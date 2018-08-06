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
"""

import numpy as np
import matplotlib.pyplot as plt

from dedalus import public as de
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
n_rho = float(args['--n_rho_cz'])
Lz = np.exp(n_rho/m_ad) - 1
Ra = float(args['--Rayleigh'])
Pr = float(args['--Prandtl'])
kappa_0 = np.sqrt(g*Lz**3*(-b)/Cp/Ra/Pr)
t_buoy = np.sqrt(Lz/g/(-b/Cp))
base_dt = t_buoy/5

print('buoyancy time: {:.4g} // b: {:.4g} // Pr {:.4g} // kappa_0 {:.4g}'.format(t_buoy, b, Pr, kappa_0))

restart=args['--restart']

data_dir = './Ra{:.2g}_b{:.2g}/'.format(Ra, b)
if not os.path.exists('{:s}/'.format(data_dir)):
    os.mkdir('{:s}/'.format(data_dir))
    os.mkdir('{:s}/figs/'.format(data_dir))


# Bases and domain
z_basis = de.Chebyshev('z', 128, interval=(0, Lz), dealias=3/2)
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
cool['g'] = 0#(1 + erf((z-mid)/sig))/2
cool.differentiate('z', out=cool_z)

# Problem
problem = de.IVP(domain, variables=['T1', 'T1_z', 'rho1', 'w'], ncc_cutoff=1e-10)
#problem = de.IVP(domain, variables=['T1', 'T1_z', 'rho1', 'M1'], ncc_cutoff=1e-10)
problem.parameters['a'] = 1
problem.parameters['b'] = b
problem.parameters['g'] = g
problem.parameters['Cp'] = Cp
problem.parameters['Cv'] = Cv
problem.parameters['Lz'] = Lz
problem.parameters['rho0'] = rho0
problem.parameters['T0'] = T0
problem.parameters['T0_z'] = T0_z
problem.parameters['T0_zz'] = T0_zz
problem.parameters['cool'] = cool
problem.parameters['cool_z'] = cool_z
problem.parameters['kappa_0'] = kappa_0
problem.substitutions['T_full']   = '(T0 + T1)'
problem.substitutions['T_z_full']   = '(T0_z + T1_z)'
problem.substitutions['rho_full']   = '(rho0 + rho1)'
problem.substitutions['kappa(T, r)'] = '(kappa_0*(T)**(3-b)*(r)**(-1-a))'
problem.substitutions['kappa_flux(T, Tz, r)'] = '(-kappa(T,r)*Tz)'
problem.substitutions['kappa_flux_full'] = 'kappa_flux(T_full, T_z_full, rho_full)'
problem.substitutions['kappa_flux_init'] = 'kappa_flux(T0, T0_z, rho0)'
problem.substitutions['cooling_mag'] = '(interp(kappa_flux_full, z={}) - right(kappa_flux_full))'.format(0.8*Lz)
problem.substitutions['cooling_z']   = 'cooling_mag*cool_z'
problem.add_equation("T1_z - dz(T1) = 0")
#problem.add_equation("dz(M1) - rho1 = 0")
problem.add_equation("dt(rho1) + dz(rho0*w) = -dz(rho1*w)")
problem.add_equation("dt(w) + rho0*T1_z + rho1*T0_z + T0*dz(rho1) + T1*dz(rho0) + rho1*g= - rho1*T1_z - T1*dz(rho1)")
problem.add_equation("dt(T1) - dz(T1_z) = -dz(kappa_flux_full)/(rho_full*Cv) - dz(T1_z)  - cooling_z/(rho_full*Cv)")

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
solver.stop_sim_time = t_therm_bot#10a0*t_buoy
solver.stop_wall_time = np.inf#50*t_buoy
solver.stop_iteration = 5000

checkpoint = Checkpoint(data_dir)
if restart is None:
    #set ICs
    pass
else:
    logger.info("restarting from {}".format(restart))
    dt = checkpoint.restart(restart, solver)
checkpoint.set_checkpoint(solver, iter=200, mode='overwrite')


diagnostics = solver.evaluator.add_dictionary_handler(group='diagnostics')
diagnostics.add_task("kappa(T_full, rho_full)", name='Kappa')
diagnostics.add_task("T_z_full", name='Tz')
diagnostics.add_task("kappa_flux_full", name='KapFlux')
diagnostics.add_task("dz(kappa_flux_full)", name='DivKapFlux')
diagnostics.add_task("dz(T_full) + T_full*dz(rho_full)/rho_full + g", name='HSE')


analysis_profile = solver.evaluator.add_file_handler(data_dir+"/profiles", max_writes=100, mode="overwrite", sim_dt=t_buoy*10)
analysis_profile.add_task("kappa_flux_full", name='KapFlux')
analysis_profile.add_task("dz(kappa_flux_full)", name='DivKapFlux')
analysis_profile.add_task("dz(T_full) + T_full*dz(rho_full)/rho_full + g", name='HSE')
 

# Initial conditions
T1 = solver.state['T1']
T1_z = solver.state['T1_z']
rho1 = solver.state['rho1']

# Main loop
dt = base_dt
iteration = 1
while solver.ok:
    solver.step(dt)
    
    if False:#iteration % 2 == 0:
        for f in T1, T1_z, rho1:
            f.set_scales(0.25, keep_data=True)
            f['c']
            f['g']
            f.set_scales(1, keep_data=True)

    if solver.iteration % output_iter == 0:

        logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time/t_buoy, dt))
        solver.evaluator.evaluate_group("diagnostics")
        logger.info('HSE iterate: ({})--({})'.format(min(diagnostics['HSE']['g']),max(diagnostics['HSE']['g'])))
        logger.info('KapFlux iterate: ({})--({})'.format(min(diagnostics['KapFlux']['g']),max(diagnostics['KapFlux']['g'])))
#        print(diagnostics['KapFlux']['g'])
#        print(diagnostics['Kappa']['g'])

        T1.set_scales(1, keep_data=True)
        T0.set_scales(1, keep_data=True)
        T1_z.set_scales(1, keep_data=True)
        T0_z.set_scales(1, keep_data=True)
        rho1.set_scales(1, keep_data=True)
        rho0.set_scales(1, keep_data=True)
        T = T0['g'] + T1['g']
        rho = rho0['g'] + rho1['g']
        T_z = T0_z['g'] + T1_z['g']
        fig = plt.figure()
        ax = fig.add_subplot(3,1,1)
        plt.plot(z, T, 'r', label='T')
        plt.plot(z, T0['g'], 'r--')
        plt.plot(z, rho, 'b', label='rho')
        plt.plot(z, rho0['g'], 'b--')

        ax2 = ax.twinx()
        plt.plot(z, T1['g'], 'r-.', label='T')
        plt.plot(z, rho1['g'], 'b-.', label='rho')
        ax.legend(loc='upper right')
        ax.set_yscale('log')
        ax = fig.add_subplot(3,1,2)
        ax.plot(np.abs(T1['c']), 'r')
        ax.plot(np.abs(rho1['c']), 'b')
        ax.set_yscale('log')
        ax.set_ylim(1e-20, 1e0)
        ax.set_ylabel('Coefficient power')
        ax = fig.add_subplot(3,1,3)
        plt.plot(z, diagnostics['KapFlux']['g'])
#        plt.plot(z, -kappa_0*rho**(-2)*T**(3-b)*T_z)
        plt.plot(z, -kappa_0*rho0['g']**(-2)*T0['g']**(3-b)*T0_z['g'], ls='--')
        ax.set_ylim(kappa_0*0.9, kappa_0*(Lz+1)**(-b)*1.1)
        ax.set_ylabel('Radiative flux')
        ax.set_yscale('log')
        ax2 = ax.twinx()
        ax2.plot(z, diagnostics['HSE']['g'], ls='-.', c='k', lw=2)
        ax2.set_ylabel('HSE')
        ax2.set_ylim(np.min(diagnostics['HSE']['g']), np.max(diagnostics['HSE']['g']))
        plt.savefig('./{:s}/figs/T_{:04d}.png'.format(data_dir, iteration), bbox_inches='tight')
        iteration += 1
        plt.close()

