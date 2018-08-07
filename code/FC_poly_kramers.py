"""
Dedalus script for 2D or 3D compressible convection in a polytrope
with a Kramer's-like opacity.

Usage:
    FC_poly_kramers.py [options] 

Options:
    ### Atmospheric Parameters
    --Rayleigh=<Rayleigh>                Rayleigh number [default: 1e5]
    --Prandtl=<Prandtl>                  Prandtl number = nu/kappa [default: 1]
    --n_rho_cz=<n_rho_cz>                Density scale heights across unstable layer [default: 3]
    --kram_a=<a>                         rho scaling, rho^(-1-a) [default: 1]
    --kram_b=<b>                         T scaling, T^(3-b) [default: -1e-4]

    ### Boundary conditions
    --mixed_flux_T                       Fixed T (top) and flux (bottom) BCs (default)
    --mixed_T_flux                       Fixed flux (top) and T (bottom) BCs
    --fixed_T                            Fixed Temperature boundary conditions (top and bottom)
    --fixed_flux                         Fixed flux boundary conditions (top and bottom)
    --no_slip                            If flagged, use no-slip BCs (otherwise use stress free)
 
    ### Initial condition switches
    --read_atmo_file=<file>              If a file is provided, read the initial T0/rho0 from there
    --restart=<restart_file>             Restart from checkpoint

    ### Dedalus parameters
    --nz=<nz>                            vertical z (chebyshev) resolution [default: 128]
    --nx=<nx>                            Horizontal x (Fourier) resolution; if not set, nx=4*nz
    --ny=<ny>                            Horizontal y (Fourier) resolution; if not set, ny=nx (3D only) 
    --max_ncc_bandwidth=<n>              Max size to expand nccs on LHS
    --3D                                 Do 3D run
    --mesh=<mesh>                        Processor mesh if distributing 3D run in 2D 
    --gamma=<gamma>                      Gamma of ideal gas (cp/cv) [default: 5/3]
    --aspect=<aspect_ratio>              Physical aspect ratio of the atmosphere [default: 4]
    
    ### End of simulation switches
    --run_time=<run_time>                Run time, in hours [default: 23.5]
    --run_time_buoy=<run_time_buoy>      Run time, in buoyancy times
    --run_time_iter=<run_time_iter>      Run time, number of iterations; if not set, n_iter=np.inf

    ### Timestepper switches
    --rk222                              Use RK222 as timestepper
    --safety_factor=<safety_factor>      Determines CFL Danger.  Higher=Faster [default: 0.2]
   
    ### File output switches 
    --root_dir=<root_dir>                Root directory to save data dir in [default: ./]
    --label=<label>                      Additional label for run output directory
    --out_cadence=<out_cad>              The fraction of a buoyancy time to output data at [default: 0.1]
    --writes=<writes>                    Writes per file [default: 20]
    --overwrite                          Force 'overwrite' mode on file writing
    --no_coeffs                          If flagged, coeffs will not be output
    --no_volumes                         If flagged, volumes will not be output (3D)
    --no_join                            If flagged, skip join operation at end of run.
"""
import logging

import numpy as np
import h5py

def read_atmosphere(read_atmo_file, solver, nz):
    T1 = solver.state['T1']
    T1_z = solver.state['T1_z']
    ln_rho1 = solver.state['ln_rho1']

    atmo = h5py.File(read_atmo_file, 'r') 
    T1_IC = atmo['tasks']['T1'].value[0,:]
    ln_rho1_IC = atmo['tasks']['ln_rho1'].value[0,:]

    T1['c']      += T1_IC[:nz]
    ln_rho1['c'] += ln_rho1_IC[:nz]
    T1.differentiate('z', out=T1_z)
    [f.set_scales(1, keep_data=True) for f in (T1, T1_z, ln_rho1)]
    import matplotlib.pyplot as plt
    plt.plot(T1['g'][0,:], ls='--')
    plt.plot(ln_rho1['g'][0,:], ls='--')
    plt.savefig('inits.png')
    atmo.close()

def FC_polytrope(Rayleigh=1e4, Prandtl=1, n_rho_cz=3, kram_a=1, kram_b=-3.5,
                 fixed_T=False, fixed_flux=False, mixed_flux_T=False, mixed_T_flux=False, no_slip=False,
                 read_atmo_file=None, restart=None, 
                 nz=128, nx=None, ny=None, max_ncc_bandwidth=None, threeD=False, mesh=None, gamma=5/3, aspect_ratio=4,
                 run_time=23.5, run_time_buoyancies=None, run_time_iter=np.inf,
                 rk222=False, safety_factor=0.2,
                 data_dir='./', out_cadence=0.1, max_writes=20, overwrite=False, no_coeffs=False, no_volumes=False, no_join=False):

    import dedalus.public as de
    from dedalus.tools  import post
    from dedalus.extras import flow_tools

    import time
    import os
    import sys
    from stratified_dynamics import polytropes
    from tools.checkpointing import Checkpoint
 
   
    checkpoint_min   = 30
    
    initial_time = time.time()

    logger.info("Starting Dedalus script {:s}".format(sys.argv[0]))

    if nx is None:
        nx = int(np.round(nz*aspect_ratio))
    if threeD and ny is None:
        ny = nx

    # Create atmosphere
    atmosphere_kwargs = { 
                          'kram_a'              : kram_a,
                          'kram_b'              : kram_b,
                          'n_rho_cz'            : n_rho_cz,
                          'nx'                  : nx,
                          'nz'                  : nz,
                          'gamma'               : gamma,
                          'aspect_ratio'        : aspect_ratio,
                          'fig_dir'             : data_dir,
                          'max_ncc_bandwidth'   : max_ncc_bandwidth,
                        }
    atmosphere = polytropes.FC_polytrope_2d_kramers(**atmosphere_kwargs)

    # Create problem, set BCs
    ncc_cutoff = 1e-10
    atmosphere.set_IVP_problem(Rayleigh, Prandtl, ncc_cutoff=ncc_cutoff)

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

    # Setup timestepper
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

    # Check atmosphere
    logger.info("thermal_time = {:g}, top_thermal_time = {:g}".format(atmosphere.thermal_time, atmosphere.top_thermal_time))
    logger.info("full atm HS check")
    atmosphere.check_atmosphere(make_plots = False, rho=atmosphere.get_full_rho(solver), T=atmosphere.get_full_T(solver))

    # Setup output type
    if restart is None or overwrite:
        mode = "overwrite"
    else:
        mode = "append"

    # Setup checkpointing & initial conditions   
    logger.info('checkpointing in {}'.format(data_dir))
    checkpoint = Checkpoint(data_dir)

    if restart is None:
        atmosphere.set_IC(solver)
        if read_atmo_file is not None:
            read_atmosphere(read_atmo_file, solver, nz)
        dt = None
    else:
        logger.info("restarting from {}".format(restart))
        dt = checkpoint.restart(restart, solver)
    checkpoint.set_checkpoint(solver, wall_dt=checkpoint_min*60, mode=mode)

    #Set up stop conditions
    if run_time_buoyancies != None:
        solver.stop_sim_time    = solver.sim_time + run_time_buoyancies*atmosphere.buoyancy_time
    else:
        solver.stop_sim_time    = 100*atmosphere.thermal_time
    
    solver.stop_iteration   = solver.iteration + run_time_iter
    solver.stop_wall_time   = run_time*3600
    report_cadence = 1
    output_time_cadence = out_cadence*atmosphere.buoyancy_time
    Hermitian_cadence = 100 #necessary for 3D
    
    logger.info("stopping after {:g} time units".format(solver.stop_sim_time))
    logger.info("output cadence = {:g}".format(output_time_cadence))
   
    #Initialize output files
    if threeD:
        analysis_tasks = atmosphere.initialize_output(solver, data_dir, sim_dt=output_time_cadence, coeffs_output=not(no_coeffs), mode=mode,max_writes=max_writes, volumes_output=not(no_volumes))
    else:
        analysis_tasks = atmosphere.initialize_output(solver, data_dir, sim_dt=output_time_cadence, coeffs_output=not(no_coeffs), mode=mode,max_writes=max_writes)

    #Set up timestep defaults and CFL.
    max_dt = output_time_cadence
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
    flow.add_property("interp(Re_rms,   z={})".format(0.95*atmosphere.Lz), name='Re_near_top')
    flow.add_property("Ma_ad_rms", name='Ma')
    flow.add_property("Pe_rms", name='Pe')
 
    start_iter=solver.iteration
    start_sim_time = solver.sim_time
    try:
        start_time = time.time()
        start_iter = solver.iteration
        logger.info('starting main loop')
        good_solution = True
        first_step = True
        while solver.ok and good_solution:
            dt = CFL.compute_dt()
            # advance
            solver.step(dt)

            effective_iter = solver.iteration - start_iter
            Re_avg = flow.grid_average('Re')

            if threeD and effective_iter % Hermitian_cadence == 0:
                for field in solver.state.fields:
                    field.require_grid_space()

            # update lists
            if effective_iter % report_cadence == 0:
                log_string = 'Iteration: {:5d}, Time: {:8.3e} ({:8.3e}), dt: {:8.3e}, '.format(solver.iteration-start_iter, solver.sim_time, (solver.sim_time-start_sim_time)/atmosphere.buoyancy_time, dt)
                log_string += 'Re: {:8.2e}/{:8.2e}'.format(Re_avg, flow.max('Re'))
                log_string += '; Pe: {:8.2e}/{:8.2e}'.format(flow.grid_average('Pe'), flow.max('Pe'))
                log_string += '; Ma: {:8.2e}/{:8.2e}'.format(flow.grid_average('Ma'), flow.max('Ma'))
                log_string += '; Re (near top): {:8.5e}'.format(flow.grid_average('Re_near_top'))
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
       
        final_checkpoint = Checkpoint(data_dir, checkpoint_name='final_checkpoint')
        final_checkpoint.set_checkpoint(solver, wall_dt=1, mode="append")
        solver.step(dt) #clean this up in the future...works for now.
    
        if not no_join:
            logger.info('beginning join operation')
            logger.info(data_dir+'/final_checkpoint/')   
            post.merge_process_files(data_dir+'/final_checkpoint/', cleanup=False)
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

    #Resolution
    nx, ny, nz = args['--nx'], args['--ny'], args['--nz']
    nz = int(nz)
    if nx is not None: nx = int(nx)
    if ny is not None: ny = int(ny)

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

    max_ncc_bandwidth = args['--max_ncc_bandwidth']
    if max_ncc_bandwidth is not None:
        max_ncc_bandwidth = int(max_ncc_bandwidth)
        
    FC_polytrope(Rayleigh=float(args['--Rayleigh']),
                 Prandtl=float(args['--Prandtl']),
                 mesh=mesh,
                 nx = nx,
                 ny = ny,
                 nz = nz,
                 kram_a = float(args['--kram_a']),
                 kram_b = float(args['--kram_b']),
                 aspect_ratio=float(args['--aspect']),
                 n_rho_cz=float(args['--n_rho_cz']),
                 gamma=float(Fraction(args['--gamma'])),
                 run_time=float(args['--run_time']),
                 run_time_buoyancies=run_time_buoy,
                 run_time_iter=run_time_iter,
                 fixed_T=args['--fixed_T'],
                 fixed_flux=args['--fixed_flux'],
                 mixed_flux_T=args['--mixed_flux_T'],
                 mixed_T_flux=args['--mixed_T_flux'],
                 no_slip=args['--no_slip'],
                 restart=(args['--restart']),
                 overwrite=args['--overwrite'],
                 rk222=rk222,
                 safety_factor=float(args['--safety_factor']),
                 out_cadence=float(args['--out_cadence']),
                 max_writes=int(float(args['--writes'])),
                 data_dir=data_dir,
                 no_coeffs=args['--no_coeffs'],
                 no_volumes=args['--no_volumes'],
                 no_join=args['--no_join'],
                 max_ncc_bandwidth=max_ncc_bandwidth,
                 read_atmo_file=args['--read_atmo_file'])
