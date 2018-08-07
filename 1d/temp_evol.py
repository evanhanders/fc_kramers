"""
Dedalus script for 2D or 3D compressible convection in a polytrope,
with specified number of density scale heights of stratification.

Usage:
    FC_poly_kramers.py [options] 

Options:
    --Rayleigh=<Rayleigh>                Rayleigh number [default: 1e7]
    --Prandtl=<Prandtl>                  Prandtl number = nu/kappa [default: 1]
    --n_rho_cz=<n_rho_cz>                Density scale heights across unstable layer [default: 3]
    --kram_b=<b>                         Kramer's b exponent [default: -0.35]
   
    --restart=<restart_file>             Restart from checkpoint
    --label=<label>                      If not none, add this to the end of the file directory.
    
    --run_iters=<r>                      Iterations to run for [default: 5000]
    --tbuoy_frac=<f>                     Fraction of a buoyancy time for dt [default: 0.5]
    
    --nz=<n>                             Number of z-coeffs [default: 512]
    --cool                               If true, run with a cooling function
"""

import numpy as np
import matplotlib.pyplot as plt

from dedalus import public as de
from dedalus.tools  import post
from dedalus.extras import flow_tools
from checkpointing import Checkpoint

import logging
logger = logging.getLogger(__name__)
import os

def new_ncc(domain):
    field = domain.new_field()
    return field


from docopt import docopt
args = docopt(__doc__)


output_iter=10
b = float(args['--kram_b'])
m_ad = 1.5
g = Cp = 2.5
Cv = 1.5
gamma = 5/3
n_rho = float(args['--n_rho_cz'])
Lz = np.exp(n_rho/m_ad) - 1
Ra = float(args['--Rayleigh'])
Pr = float(args['--Prandtl'])
kappa_0 = np.sqrt(g*Lz**3*(-b)/Cp/Ra/Pr)
t_buoy = np.sqrt(Lz/g/(-b/Cp))
base_dt = t_buoy*float(args['--tbuoy_frac'])
nz      = int(args['--nz'])


data_dir = './Ra{:.2g}_b{:.2g}'.format(Ra, b)

print('buoyancy time: {:.4g} // b: {:.4g} // Pr {:.4g} // kappa_0 {:.4g}'.format(t_buoy, b, Pr, kappa_0))

restart=args['--restart']


# Bases and domain
z_basis = de.Chebyshev('z', nz, interval=(0, Lz), dealias=3/2)
domain = de.Domain([z_basis], np.float64)

z = domain.grid(-1)

T0 = new_ncc(domain)
rho0 = new_ncc(domain)
T0_z = new_ncc(domain)
T0_zz = new_ncc(domain)

T0['g'] = (1 + Lz - z)
T0.differentiate('z', out=T0_z)
T0_z.differentiate('z', out=T0_zz)
rho0['g'] = (1 + Lz - z)**m_ad

from scipy.special import erf
cool = new_ncc(domain)
cool_z = domain.new_field()#new_ncc(domain)
mid = 0.95*Lz
sig = 0.05*Lz
if args['--cool']:
    cool['g'] = (1 + erf((z-mid)/sig))/2
    data_dir += '_cool'
else:
    cool['g'] = 0
cool.differentiate('z', out=cool_z)

label = args['--label']
if label is not None:
    data_dir += '_{:s}'.format(label)
data_dir += '/'

if not os.path.exists('{:s}/'.format(data_dir)):
    os.mkdir('{:s}/'.format(data_dir))
    os.mkdir('{:s}/figs/'.format(data_dir))



# Problem
problem = de.IVP(domain, variables=['T1', 'T1_z', 'ln_rho1', 'w'], ncc_cutoff=1e-10)
#problem = de.IVP(domain, variables=['T1', 'T1_z', 'rho1', 'M1'], ncc_cutoff=1e-10)
problem.parameters['a'] = 1
problem.parameters['b'] = b
problem.parameters['g'] = g
problem.parameters['gamma'] = gamma
problem.parameters['Cp'] = Cp
problem.parameters['Cv'] = Cv
problem.parameters['Lz'] = Lz
problem.parameters['rho0'] = rho0
problem.parameters['T0'] = T0
problem.parameters['T0_z'] = T0_z
problem.parameters['T0_zz'] = T0_zz
problem.parameters['cool'] = cool
problem.parameters['cooling_mag'] = kappa_0*((Lz+1)**(-b) - 1)
problem.parameters['cool_z'] = cool_z
problem.parameters['kappa_0'] = kappa_0
problem.substitutions['T_full']   = '(T0 + T1)'
problem.substitutions['T_z_full']   = '(T0_z + T1_z)'
problem.substitutions['rho_full']   = '(rho0*exp(ln_rho1))'
problem.substitutions['rho_fluc']   = '(rho_full - rho0)'
problem.substitutions['ln_rho0']   = '(log(rho0))'
problem.substitutions['kappa(T, r)'] = '(kappa_0*(T)**(3-b)*(r)**(-1-a))'
problem.substitutions['kappa_flux(T, Tz, r)'] = '(-kappa(T,r)*Tz)'
problem.substitutions['kappa_flux_full'] = 'kappa_flux(T_full, T_z_full, rho_full)'
problem.substitutions['kappa_flux_init'] = 'kappa_flux(T0, T0_z, rho0)'
#problem.substitutions['cooling_mag'] = '(interp(kappa_flux_full, z={}) - right(kappa_flux_full))'.format(0.8*Lz)
problem.substitutions['cooling_z']   = 'cooling_mag*cool_z'
problem.add_equation("T1_z - dz(T1) = 0")
#problem.add_equation("dz(M1) - rho1 = 0")
problem.add_equation("dt(ln_rho1) + w*dz(ln_rho0) + dz(w) = -w*dz(ln_rho1)")
problem.add_equation("dt(w) + T1_z + T0*dz(ln_rho1) + T1*dz(ln_rho0) = - T1*dz(ln_rho1) - w*dz(w)")
problem.add_equation("dt(T1) + w*T0_z + (gamma-1)*T0*dz(w) - dz(T1_z) = -w*T1_z - (gamma-1)*T1*dz(w) - dz(kappa_flux_full)/(rho_full*Cv) - dz(T1_z)  - cooling_z/(rho_full*Cv)")

problem.add_bc("right(T1) = 0")
problem.add_bc("left(T1_z) = left(-T0_z - kappa_flux_init/kappa(T_full, rho_full))")
#problem.add_bc("left(M1) = 0")
#problem.add_bc("right(M1) = 0")
problem.add_bc("left(w) = 0")
problem.add_bc("right(w) = 0")

for key in ['T1', 'T1_z', 'w']:
    problem.meta[key]['z']['dirichlet'] = True

t_therm_bot = Lz**2 / (kappa_0 * (Lz+1)**(-b)/(Lz+1)**m_ad/Cp)
print(t_therm_bot/t_buoy)

# Build solver
solver = problem.build_solver(de.timesteppers.RK443)

checkpoint = Checkpoint(data_dir)
if restart is None:
    #set ICs
    pass
else:
    logger.info("restarting from {}".format(restart))
    dt = checkpoint.restart(restart, solver)
checkpoint.set_checkpoint(solver, iter=200, mode='overwrite')


solver.stop_sim_time = solver.sim_time + t_therm_bot#10a0*t_buoy
solver.stop_wall_time = np.inf#50*t_buoy
solver.stop_iteration = solver.iteration + int(args['--run_iters'])


diagnostics = solver.evaluator.add_dictionary_handler(group='diagnostics')
diagnostics.add_task("dz(T_full)/gamma/T_full - (gamma-1)*dz(rho_full)/gamma/rho_full", name='gradS/Cp')
diagnostics.add_task("cooling_mag*cool", name='cooling')
diagnostics.add_task("kappa(T_full, rho_full)", name='Kappa')
diagnostics.add_task("T_z_full", name='Tz')
diagnostics.add_task("kappa_flux_full", name='KapFlux')
diagnostics.add_task("dz(kappa_flux_full)", name='DivKapFlux')
diagnostics.add_task("kappa_flux_full + cooling_mag*cool", name='Flux')
diagnostics.add_task("dz(kappa_flux_full) + cooling_mag*cool_z", name='TE')
diagnostics.add_task("dz(T_full) + T_full*dz(rho_full)/rho_full + g", name='HSE')


analysis_profile = solver.evaluator.add_file_handler(data_dir+"/profiles", max_writes=100, mode="overwrite", sim_dt=t_buoy)
analysis_profile.add_task("kappa_flux_full", name='KapFlux')
analysis_profile.add_task("dz(kappa_flux_full)", name='DivKapFlux')
analysis_profile.add_task("dz(T_full) + T_full*dz(rho_full)/rho_full + g", name='HSE')
analysis_profile.add_task("T1", name='T1')
analysis_profile.add_task("T1_z", name='T1_z')
analysis_profile.add_task("ln_rho1", name='ln_rho1')
analysis_profile.add_task("w", name='w')
 

# Initial conditions
T1 = solver.state['T1']
w = solver.state['w']
T1_z = solver.state['T1_z']
ln_rho1 = solver.state['ln_rho1']


# Main loop
dt = base_dt

cfl_cadence=10
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=cfl_cadence, safety=0.4, max_change=1.5, min_change=0.5, max_dt=base_dt, threshold=0.1)
CFL.add_velocities(('w'))

init_iter = solver.iteration

iteration = 1
try:
    while solver.ok:
        dt = CFL.compute_dt()
        solver.step(dt)
        if solver.iteration % output_iter == 0:

            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration-init_iter, solver.sim_time/t_buoy, dt/t_buoy))
            solver.evaluator.evaluate_group("diagnostics")
            logger.info('HSE iterate:  ({:.4e})--({:.4e})'.format(min(diagnostics['HSE']['g']),max(diagnostics['HSE']['g'])))
            logger.info('TE iterate:   ({:.4e})--({:.4e})'.format(min(diagnostics['TE']['g']),max(diagnostics['TE']['g'])))
            logger.info('Flux iterate: ({:.4e})--({:.4e})'.format(min(diagnostics['Flux']['g']),max(diagnostics['Flux']['g'])))
    #        print(diagnostics['KapFlux']['g'])
    #        print(diagnostics['Kappa']['g'])

            T1.set_scales(1, keep_data=True)
            T0.set_scales(1, keep_data=True)
            T1_z.set_scales(1, keep_data=True)
            T0_z.set_scales(1, keep_data=True)
            ln_rho1.set_scales(1, keep_data=True)
            rho0.set_scales(1, keep_data=True)
            T = T0['g'] + T1['g']
            rho = rho0['g']*np.exp(ln_rho1['g'])
            T_z = T0_z['g'] + T1_z['g']
            fig = plt.figure(figsize=(6,12))
            ax = fig.add_subplot(5,1,1)
            plt.plot(z, T, 'r', label='T')
            plt.plot(z, T0['g'], 'r--')
            plt.plot(z, rho, 'b', label='rho')
            plt.plot(z, rho0['g'], 'b--')
            ax.set_xlim(0, Lz)
            ax.set_ylabel('T (r) / rho (b)')

            ax2 = ax.twinx()
            plt.plot(z, T1['g'], 'r-.', label='T')
            plt.plot(z, rho-rho0['g'], 'b-.', label='rho')
            ax2.set_ylabel('Fluc (dash-dot)')
            ax.legend(loc='upper right')
            ax.set_yscale('log')
            ax = fig.add_subplot(5,1,2)
            ax.axhline(1e-5, c='k')
            ax.axhline(1e-10, c='k')
            ax.axhline(1e-15, c='k')
            ax.plot(np.abs(T1['c']), 'r')
            ax.plot(np.abs(ln_rho1['c']), 'b')
            ax.set_xlim(0, nz)
            ax.set_yscale('log')
            ax.set_ylim(1e-20, 1e0)
            ax.set_ylabel('Coefficient power')
            ax = fig.add_subplot(5,1,3)
            plt.plot(z, diagnostics['KapFlux']['g'], c='r')
    #        plt.plot(z, -kappa_0*rho**(-2)*T**(3-b)*T_z)
            plt.plot(z, -kappa_0*rho0['g']**(-2)*T0['g']**(3-b)*T0_z['g'], ls='--', c='orange')
            plt.plot(z, diagnostics['cooling']['g'], ls='--', c='b')
            plt.plot(z, diagnostics['Flux']['g'], c='k')
            ax.set_xlim(0, Lz)
            ax.axhline(kappa_0*(Lz+1)**(-b), ls='-.', c='grey')
            ax.set_ylim(kappa_0*0.9, kappa_0*(Lz+1)**(-b)*1.1)
            ax.set_ylabel('Radiative flux')
            ax.set_yscale('log')

            ax = fig.add_subplot(5,1,4)
            ax2 = ax.twinx()
            ax2.plot(z, diagnostics['HSE']['g'], ls='-.', c='r', lw=2)
            ax2.set_ylabel('HSE (red)')
            ax2.set_ylim(np.min(diagnostics['HSE']['g']), np.max(diagnostics['HSE']['g']))
            ax.plot(z, diagnostics['TE']['g']/kappa_0, c='b', lw=2)
            ax.set_xlim(0, Lz)
            ax.set_ylabel(r'TE/$\kappa_0$ (blue)')

            ax = fig.add_subplot(5,1,5)
            w.set_scales(1, keep_data=True)
            ax.plot(z, w['g'], c='r')
            ax.set_xlim(0, Lz)
            ax.axhline(0, c='r', ls='--')
            ax.set_ylabel('w (red)')
            
            ax2 = ax.twinx()
            ax2.plot(z, diagnostics['gradS/Cp']['g'], c='k')
            ax2.axhline(0, c='k', ls='--')
            ax2.set_ylabel('gradS/Cp (black)')

            plt.savefig('./{:s}/figs/T_{:04d}.png'.format(data_dir, iteration), bbox_inches='tight', figsize=(6,12))
            iteration += 1
            plt.close()
except:
    logger.info('error thrown, merging and exiting')
finally:
    final_checkpoint = Checkpoint(data_dir, checkpoint_name='final_checkpoint')
    final_checkpoint.set_checkpoint(solver, wall_dt=1, mode="overwrite")
    solver.step(dt) #clean this up in the future...works for now.

    post.merge_process_files(data_dir+'/final_checkpoint/', cleanup=False)
    post.merge_process_files(data_dir+'/checkpoint/', cleanup=False)
    post.merge_process_files(data_dir+'/profiles/', cleanup=False)
