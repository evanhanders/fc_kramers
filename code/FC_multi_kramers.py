"""
Dedalus script for 2D or 3D compressible convection in a polytrope,
with specified number of density scale heights of stratification.

Usage:
    FC_poly_kramers.py [options] 

Options:
    --Rayleigh=<Rayleigh>                Rayleigh number [default: 1e4]
    --Prandtl=<Prandtl>                  Prandtl number = nu/kappa [default: 1]
    --n_rho_cz=<n_rho_cz>                Density scale heights across unstable layer [default: 3]
    --epsilon=<epsilon>                  The level of superadiabaticity of our polytrope background [default: 1e-4]
    --gamma=<gamma>                      Gamma of ideal gas (cp/cv) [default: 5/3]
    --aspect=<aspect_ratio>              Physical aspect ratio of the atmosphere [default: 4]

    --no_init_bvp                       If flagged, don't solve a bvp for initial HS balance.


    --nz=<nz>                            vertical z (chebyshev) resolution [default: 128]
    --nx=<nx>                            Horizontal x (Fourier) resolution; if not set, nx=4*nz
    --ny=<ny>                            Horizontal y (Fourier) resolution; if not set, ny=nx (3D only) 
    --3D                                 Do 3D run
    --mesh=<mesh>                        Processor mesh if distributing 3D run in 2D 

    --run_time=<run_time>                Run time, in hours [default: 23.5]
    --run_time_buoy=<run_time_buoy>      Run time, in buoyancy times
    --run_time_iter=<run_time_iter>      Run time, number of iterations; if not set, n_iter=np.inf

    --fixed_T                            Fixed Temperature boundary conditions (top and bottom; default if no BCs specified)
    --mixed_flux_T                       Fixed T (top) and flux (bottom) BCs
    --mixed_T_flux                       Fixed flux (top) and T (bottom) BCs
    --fixed_flux                         Fixed flux boundary conditions (top and bottom)
    --no_slip                            If flagged, use no-slip BCs (otherwise use stress free)
    --const_nu                           If flagged, use constant nu 
    --const_chi                          If flagged, use constant chi 

    --kram_a=<a>                         rho scaling, rho^(-1-a) [default: 1]
    --kram_b=<b>                         T scaling, T^(3-b) [default: -1e-4]
    --split_diffusivities                If true, split diffusivities betwen LHS and RHS to reduce bandwidth
    
    --restart=<restart_file>             Restart from checkpoint
    --start_new_files                    Start new files while checkpointing

    --rk222                              Use RK222 as timestepper
    --safety_factor=<safety_factor>      Determines CFL Danger.  Higher=Faster [default: 0.2]
    
    --root_dir=<root_dir>                Root directory to save data dir in [default: ./]
    --label=<label>                      Additional label for run output directory
    --out_cadence=<out_cad>              The fraction of a buoyancy time to output data at [default: 0.1]
    --writes=<writes>                    Writes per file [default: 20]
    --no_coeffs                          If flagged, coeffs will not be output
    --no_volumes                         If flagged, volumes will not be output (3D)
    --no_join                            If flagged, skip join operation at end of run.

    --verbose                            Do extra output (Peclet and Nusselt numbers) to screen
    --fully_nonlinear                    If flagged, evolve full form of Kramer's opacity kappa.

    --do_bvp                             If flagged, do BVPs at regular intervals when Re > 1 to converge faster
    --num_bvps=<num>                     Max number of bvps to solve [default: 3]
    --bvp_equil_time=<time>              How long to wait after a previous BVP before starting to average for next one, in tbuoy [default: 30]
    --bvp_final_equil_time=<time>        How long to wait after last bvp before ending simulation 
    --bvp_transient_time=<time>          How long to wait at beginning of run before starting to average for next one, in tbuoy [default: 20]
    --min_bvp_time=<time>                Minimum avg time for a bvp (in tbuoy) [default: 30]
    --first_bvp_time=<time>                Minimum avg time for a bvp (in tbuoy) [default: 20]
    --bvp_resolution_factor=<mult>       an int, how many times larger than nz should the bvp nz be? [default: 1]
    --bvp_convergence_factor=<fact>      How well converged time averages need to be for BVP [default: 1e-3]
    --first_bvp_convergence_factor=<fact>      How well converged time averages need to be for BVP [default: 1e-2]


"""
import logging

import numpy as np



def FC_multitrope(Rayleigh=1e4, Prandtl=1, aspect_ratio=4, kram_a=1, kram_b=-3.5,
                 nz=128, nx=None, ny=None, threeD=False, mesh=None,
                 split_diffusivities=False,
                 n_rho_cz=3, epsilon=1e-4, gamma=5/3,
                 run_time=23.5, run_time_buoyancies=None, run_time_iter=np.inf,
                 fixed_T=False, fixed_flux=False, mixed_flux_T=False, mixed_T_flux=False,
                 restart=None, start_new_files=False,
                 rk222=False, safety_factor=0.2,
                 max_writes=20, no_slip=False, fully_nonlinear=False,
                 data_dir='./', out_cadence=0.1, no_coeffs=False, no_volumes=False, no_join=False, 
                 do_bvp=False, bvp_equil_time=10, bvp_transient_time=20, bvp_resolution_factor=1, bvp_convergence_factor=1e-2,
                 num_bvps=3, bvp_final_equil_time=None, verbose=False, min_bvp_time=20, first_bvp_time=20, first_bvp_convergence_factor=1e-2,
                 init_bvp=True):

    import dedalus.public as de
    from dedalus.tools  import post
    from dedalus.extras import flow_tools

    import time
    import os
    import sys
    from stratified_dynamics import multitropes
    from tools.checkpointing import Checkpoint
 
   
    checkpoint_min   = 30
    
    initial_time = time.time()

    logger.info("Starting Dedalus script {:s}".format(sys.argv[0]))

    if nx is None:
        nx = int(np.round(nz*aspect_ratio))
    if threeD and ny is None:
        ny = nx


    bc_dict = {
            'stress_free'             : False,
            'no_slip'                 : False,
            'fixed_flux'              : False,
            'mixed_flux_temperature'  : False,
            'mixed_temperature_flux'  : False,
            'fixed_temperature'       : False
              }
    if no_slip:
        bc_dict['no_slip'] = True
    else:
        bc_dict['stress_free'] = True

    if fixed_flux:
        bc_dict['fixed_flux'] = True
    elif fixed_T:
        bc_dict['fixed_temperature'] = True
    elif mixed_T_flux:
        bc_dict['mixed_temperature_flux'] = True
    else:
        bc_dict['mixed_flux_temperature'] = True


    atmosphere = multitropes.FC_multitrope_2d_kramers(bc_dict, nx=nx, nz=nz, gamma=gamma, aspect_ratio=aspect_ratio, fig_dir=data_dir, fully_nonlinear=fully_nonlinear, kram_a=kram_a, kram_b=kram_b, no_equil=not(init_bvp))



    ncc_cutoff = 1e-10
        
    atmosphere.set_IVP_problem(Rayleigh, Prandtl, ncc_cutoff=ncc_cutoff, split_diffusivities=split_diffusivities)

    bc_dict = {
            'stress_free'             : False,
            'no_slip'                 : False,
            'fixed_flux'              : False,
            'mixed_flux_temperature'  : False,
            'mixed_temperature_flux'  : False,
            'fixed_temperature'       : False
              }
    if no_slip:
        bc_dict['no_slip'] = True
    else:
        bc_dict['stress_free'] = True

    if fixed_flux:
        bc_dict['fixed_flux'] = True
    elif fixed_T:
        bc_dict['fixed_temperature'] = True
    elif mixed_T_flux:
        bc_dict['mixed_temperature_flux'] = True
    else:
        bc_dict['mixed_flux_temperature'] = True

    atmosphere.set_BC(**bc_dict)

    problem = atmosphere.get_problem()

    if atmosphere.domain.distributor.rank == 0:
        if not os.path.exists('{:s}/'.format(data_dir)):
            os.mkdir('{:s}/'.format(data_dir))

    if rk222:
        logger.info("timestepping using RK222")
        ts = de.timesteppers.RK222
        cfl_safety_factor = safety_factor*2
    else:
        logger.info("timestepping using RK443")
        ts = de.timesteppers.RK443
        cfl_safety_factor = safety_factor*4

    # Build solver
    solver = problem.build_solver(ts)

    #Check atmosphere
    logger.info("thermal_time = {:g}, top_thermal_time = {:g}".format(atmosphere.thermal_time,\
                                                                    atmosphere.top_thermal_time))
    logger.info("full atm HS check")
    atmosphere.check_atmosphere(make_plots = False, rho=atmosphere.get_full_rho(solver), T=atmosphere.get_full_T(solver))

    if restart is None or start_new_files:
        mode = "overwrite"
    else:
        mode = "append"

    logger.info('checkpointing in {}'.format(data_dir))
    checkpoint = Checkpoint(data_dir)

    if restart is None:
        T1 = solver.state['T1']
        T1['g'] = 0
        atmosphere.set_IC(solver)
        dt = None
    else:
        logger.info("restarting from {}".format(restart))
        dt = checkpoint.restart(restart, solver)

    checkpoint.set_checkpoint(solver, wall_dt=checkpoint_min*60, mode=mode)
    
    if run_time_buoyancies != None:
        solver.stop_sim_time    = solver.sim_time + run_time_buoyancies*atmosphere.buoyancy_time
    else:
        solver.stop_sim_time    = 100*atmosphere.thermal_time
    
    solver.stop_iteration   = solver.iteration + run_time_iter
    solver.stop_wall_time   = run_time*3600
    report_cadence = 1
    output_time_cadence = out_cadence*atmosphere.buoyancy_time
    Hermitian_cadence = 100
    
    logger.info("stopping after {:g} time units".format(solver.stop_sim_time))
    logger.info("output cadence = {:g}".format(output_time_cadence))
   
    if threeD:
        analysis_tasks = atmosphere.initialize_output(solver, data_dir, sim_dt=output_time_cadence, coeffs_output=not(no_coeffs), mode=mode,max_writes=max_writes, volumes_output=not(no_volumes))
    else:
        analysis_tasks = atmosphere.initialize_output(solver, data_dir, sim_dt=output_time_cadence, coeffs_output=not(no_coeffs), mode=mode,max_writes=max_writes)

    #Set up timestep defaults
    max_dt = output_time_cadence
#    max_dt = atmosphere.thermal_time
    if dt is None: dt = max_dt
        
    cfl_cadence = 1
    cfl_threshold = 0.1
    CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=cfl_cadence, safety=cfl_safety_factor,
                         max_change=1.5, min_change=0.1, max_dt=max_dt, threshold=cfl_threshold)
    if threeD:
        CFL.add_velocities(('u', 'v', 'w'))
    else:
        CFL.add_velocities(('u', 'w'))

    # Flow properties
    flow = flow_tools.GlobalFlowProperty(solver, cadence=1)
    flow.add_property("Re_rms", name='Re')
    flow.add_property("Ma_ad_rms", name='Ma')
    flow.add_property("Pe_rms", name='Pe')


    if verbose:
        flow.add_property("Pe_rms", name='Pe')
        flow.add_property("Nusselt_AB17", name='Nusselt')
    if do_bvp:
        if not isinstance(bvp_final_equil_time, type(None)):
            bvp_final_equil_time *= atmosphere.buoyancy_time
        if not dynamic_diffusivities:
            raise NotImplementedError('BVP method only implemented for constant kappa formulation')
        bvp_solver = FC_BVP_Solver(polytropes.FC_polytrope_2d_kappa_mu, nx, ny, nz, \
                                   flow, atmosphere.domain.dist.comm_cart, \
                                   solver, num_bvps, bvp_equil_time*atmosphere.buoyancy_time,\
                                   threeD=threeD,\
                                   bvp_transient_time=bvp_transient_time*atmosphere.buoyancy_time,\
                                   bvp_run_threshold=bvp_convergence_factor, 
                                   bvp_l2_check_time=atmosphere.buoyancy_time, mesh=mesh,\
                                   first_bvp_time=first_bvp_time*atmosphere.buoyancy_time,\
                                   first_run_threshold=first_bvp_convergence_factor,\
                                   plot_dir='{}/bvp_plots/'.format(data_dir),
                                   min_avg_dt=1e-10*atmosphere.buoyancy_time, 
                                   final_equil_time=bvp_final_equil_time,
                                   min_bvp_time=min_bvp_time*atmosphere.buoyancy_time)
        bc_dict.pop('stress_free')
        bc_dict.pop('no_slip')

 
    start_iter=solver.iteration
    start_sim_time = solver.sim_time

#    print('T0', atmosphere.T0['g'])

    try:
        start_time = time.time()
        start_iter = solver.iteration
        logger.info('starting main loop')
        good_solution = True
        first_step = True
        continue_bvps = True
        while solver.ok and good_solution and continue_bvps:
            dt = CFL.compute_dt()
            # advance
            solver.step(dt)

            effective_iter = solver.iteration - start_iter
            Re_avg = flow.grid_average('Re')

            if threeD and effective_iter % Hermitian_cadence == 0:
                for field in solver.state.fields:
                    field.require_grid_space()

            if do_bvp:
                bvp_solver.update_avgs(dt, Re_avg, np.sqrt(Rayleigh*np.exp(n_rho_cz)**2/4/1000))
                if bvp_solver.check_if_solve():
                    atmo_kwargs = { 'constant_kappa' : const_kappa,
                                    'constant_mu'    : const_mu,
                                    'epsilon'        : epsilon,
                                    'gamma'          : gamma,
                                    'n_rho_cz'       : n_rho_cz,
                                    'nz'             : nz*bvp_resolution_factor
                                   }
                    diff_args   = { 'Rayleigh'       : Rayleigh, 
                                    'Prandtl'        : Prandtl
                                  }
                    bvp_solver.solve_BVP(atmo_kwargs, diff_args, bc_dict)
                if bvp_solver.terminate_IVP():
                    continue_bvps = False



            # update lists
            if effective_iter % report_cadence == 0:
                log_string = 'Iteration: {:5d}, Time: {:8.3e} ({:8.3e}), dt: {:8.3e}, '.format(solver.iteration-start_iter, solver.sim_time, (solver.sim_time-start_sim_time)/atmosphere.buoyancy_time, dt)
                if verbose:
                    log_string += '\n\t\tRe: {:8.5e}/{:8.5e}'.format(Re_avg, flow.max('Re'))
                    log_string += '; Pe: {:8.5e}/{:8.5e}'.format(flow.grid_average('Pe'), flow.max('Pe'))
                    log_string += '; Nu: {:8.5e}/{:8.5e}'.format(flow.grid_average('Nusselt'), flow.max('Nusselt'))
                else:
                    log_string += 'Re: {:8.2e}/{:8.2e}'.format(Re_avg, flow.max('Re'))
                    log_string += '; Pe: {:8.2e}/{:8.2e}'.format(flow.grid_average('Pe'), flow.max('Pe'))
                    log_string += '; Ma: {:8.2e}/{:8.2e}'.format(flow.grid_average('Ma'), flow.max('Ma'))
                logger.info(log_string)
                
            if not np.isfinite(Re_avg):
                good_solution = False
                logger.info("Terminating run.  Trapped on Reynolds = {}".format(Re_avg))
                    
            if first_step:
                if verbose:
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt
                    fig = plt.figure()
                    ax = fig.add_subplot(1,1,1)
                    ax.spy(solver.pencils[0].L, markersize=1, markeredgewidth=0.0)
                    fig.savefig(data_dir+"sparsity_pattern.png", dpi=1200)

                    import scipy.sparse.linalg as sla
                    LU = sla.splu(solver.pencils[0].LHS.tocsc(), permc_spec='NATURAL')
                    fig = plt.figure()
                    ax = fig.add_subplot(1,2,1)
                    ax.spy(LU.L.A, markersize=1, markeredgewidth=0.0)
                    ax = fig.add_subplot(1,2,2)
                    ax.spy(LU.U.A, markersize=1, markeredgewidth=0.0)
                    fig.savefig(data_dir+"sparsity_pattern_LU.png", dpi=1200)

                    logger.info("{} nonzero entries in LU".format(LU.nnz))
                    logger.info("{} nonzero entries in LHS".format(solver.pencils[0].LHS.tocsc().nnz))
                    logger.info("{} fill in factor".format(LU.nnz/solver.pencils[0].LHS.tocsc().nnz))
                first_step = False
                start_time = time.time()
    except:
        logger.error('Exception raised, triggering end of main loop.')
    finally:
        end_time = time.time()

        # Print statistics
        elapsed_time = end_time - start_time
        elapsed_sim_time = solver.sim_time
        N_iterations = solver.iteration-1
        logger.info('main loop time: {:e}'.format(elapsed_time))
        logger.info('Iterations: {:d}'.format(N_iterations))
        logger.info('iter/sec: {:g}'.format(N_iterations/(elapsed_time)))
        if N_iterations > 0:
            logger.info('Average timestep: {:e}'.format(elapsed_sim_time / N_iterations))
        
        if not no_join:
            logger.info('beginning join operation')
            try:
                final_checkpoint = Checkpoint(data_dir, checkpoint_name='final_checkpoint')
                final_checkpoint.set_checkpoint(solver, wall_dt=1, mode="append")
                solver.step(dt) #clean this up in the future...works for now.
                post.merge_process_files(data_dir+'/final_checkpoint/', cleanup=False)
            except:
                print('cannot save final checkpoint')
                
            logger.info(data_dir+'/checkpoint/')
            post.merge_process_files(data_dir+'/checkpoint/', cleanup=False)
            
            for task in analysis_tasks.keys():
                logger.info(analysis_tasks[task].base_path)
                post.merge_process_files(analysis_tasks[task].base_path, cleanup=False)

        if (atmosphere.domain.distributor.rank==0):

            logger.info('main loop time: {:e}'.format(elapsed_time))
            if start_iter > 1:
                logger.info('Iterations (this run): {:d}'.format(N_iterations - start_iter))
                logger.info('Iterations (total): {:d}'.format(N_iterations - start_iter))
            logger.info('iter/sec: {:g}'.format(N_iterations/(elapsed_time)))
            if N_iterations > 0:
                logger.info('Average timestep: {:e}'.format(elapsed_sim_time / N_iterations))
 
            N_TOTAL_CPU = atmosphere.domain.distributor.comm_cart.size

            # Print statistics
            print('-' * 40)
            total_time = end_time-initial_time
            main_loop_time = end_time - start_time
            startup_time = start_time-initial_time
            n_steps = solver.iteration-1
            print('  startup time:', startup_time)
            print('main loop time:', main_loop_time)
            print('    total time:', total_time)
            if n_steps > 0:
                print('    iterations:', n_steps)
                print(' loop sec/iter:', main_loop_time/n_steps)
                print('    average dt:', solver.sim_time/n_steps)
                print("          N_cores, Nx, Nz, startup     main loop,   main loop/iter, main loop/iter/grid, n_cores*main loop/iter/grid")
                print('scaling:',
                    ' {:d} {:d} {:d}'.format(N_TOTAL_CPU,nx,nz),
                    ' {:8.3g} {:8.3g} {:8.3g} {:8.3g} {:8.3g}'.format(startup_time,
                                                                    main_loop_time, 
                                                                    main_loop_time/n_steps, 
                                                                    main_loop_time/n_steps/(nx*nz), 
                                                                    N_TOTAL_CPU*main_loop_time/n_steps/(nx*nz)))
            print('-' * 40)

if __name__ == "__main__":
    from docopt import docopt
    args = docopt(__doc__)
    from numpy import inf as np_inf
    from fractions import Fraction

    import os
    import sys
    # save data in directory named after script
    #   these lines really are all about setting up the output directory name
    data_dir = args['--root_dir']
    if data_dir[-1] != '/':
        data_dir += '/'
    data_dir += sys.argv[0].split('.py')[0]

    #BCs

    if args['--fixed_T'] and args['--verbose']:
        data_dir += '_fixed'
    elif args['--fixed_flux'] and args['--verbose']:
        data_dir += '_flux'

    if args['--3D']:
        data_dir +='_3D'
    else:
        data_dir +='_2D'
    data_dir += "_nrhocz{}_Ra{}_Pr{}".format(args['--n_rho_cz'], args['--Rayleigh'], args['--Prandtl'])
    data_dir += "_b{}_a{}".format(args['--kram_b'], args['--aspect'])
    
    if args['--label'] == None:
        data_dir += '/'
    else:
        data_dir += '_{}/'.format(args['--label'])

    from dedalus.tools.config import config
    
    config['logging']['filename'] = os.path.join(data_dir,'logs/dedalus_log')
    config['logging']['file_level'] = 'DEBUG'

    import mpi4py.MPI
    if mpi4py.MPI.COMM_WORLD.rank == 0:
        if not os.path.exists('{:s}/'.format(data_dir)):
            os.mkdir('{:s}/'.format(data_dir))
        logdir = os.path.join(data_dir,'logs')
        if not os.path.exists(logdir):
            os.mkdir(logdir)

    logger = logging.getLogger(__name__)
    logger.info("saving run in: {}".format(data_dir))


    #Timestepper type
    if args['--rk222']:
        rk222=True
    else:
        rk222=False

    #Restarting options
    if args['--start_new_files']:
        start_new_files = True
    else:
        start_new_files = False

    #Resolution
    nx = args['--nx']
    if nx is not None:
        nx = int(nx)
    ny =  args['--ny']
    if ny is not None:
        ny = int(ny)
    nz = int(args['--nz'])

    mesh = args['--mesh']
    if mesh is not None:
        mesh = mesh.split(',')
        mesh = [int(mesh[0]), int(mesh[1])]

    run_time_buoy = args['--run_time_buoy']
    if run_time_buoy != None:
        run_time_buoy = float(run_time_buoy)
        
    run_time_iter = args['--run_time_iter']
    if run_time_iter != None:
        run_time_iter = int(float(run_time_iter))
    else:
        run_time_iter = np_inf

    bvp_final_equil_time = args['--bvp_final_equil_time']
    if not isinstance(bvp_final_equil_time, type(None)):
        bvp_final_equil_time = float(bvp_final_equil_time)
        
    FC_multitrope(Rayleigh=float(args['--Rayleigh']),
                 Prandtl=float(args['--Prandtl']),
                 mesh=mesh,
                 nx = nx,
                 ny = ny,
                 nz = nz,
                 kram_a = float(args['--kram_a']),
                 kram_b = float(args['--kram_b']),
                 aspect_ratio=float(args['--aspect']),
                 n_rho_cz=float(args['--n_rho_cz']),
                 epsilon=float(args['--epsilon']),
                 gamma=float(Fraction(args['--gamma'])),
                 run_time=float(args['--run_time']),
                 run_time_buoyancies=run_time_buoy,
                 run_time_iter=run_time_iter,
                 fixed_T=args['--fixed_T'],
                 fixed_flux=args['--fixed_flux'],
                 mixed_flux_T=args['--mixed_flux_T'],
                 mixed_T_flux=args['--mixed_T_flux'],
                 restart=(args['--restart']),
                 start_new_files=start_new_files,
                 rk222=rk222,
                 safety_factor=float(args['--safety_factor']),
                 out_cadence=float(args['--out_cadence']),
                 max_writes=int(float(args['--writes'])),
                 data_dir=data_dir,
                 fully_nonlinear=args['--fully_nonlinear'],
                 no_coeffs=args['--no_coeffs'],
                 no_volumes=args['--no_volumes'],
                 no_join=args['--no_join'],
                 do_bvp=args['--do_bvp'],
                 num_bvps=int(args['--num_bvps']),
                 bvp_equil_time=float(args['--bvp_equil_time']),
                 bvp_final_equil_time=bvp_final_equil_time,
                 bvp_transient_time=float(args['--bvp_transient_time']),
                 bvp_resolution_factor=int(args['--bvp_resolution_factor']),
                 bvp_convergence_factor=float(args['--bvp_convergence_factor']),
                 min_bvp_time=float(args['--min_bvp_time']),
                 verbose=args['--verbose'],
                 no_slip=args['--no_slip'],
                 first_bvp_convergence_factor=float(args['--first_bvp_convergence_factor']),
                 first_bvp_time=float(args['--first_bvp_time']),
                 split_diffusivities=args['--split_diffusivities'],
                 init_bvp=not(args['--no_init_bvp']))
