import numpy as np
import matplotlib.pyplot as plt
from dedalus import public as de
from dedalus.extras import flow_tools
from scipy.special import erf
from dedalus.core.evaluator import Evaluator
from mpi4py import MPI

import logging
logger = logging.getLogger(__name__)

try:
    from equations import *
except:
    from sys import path
    path.insert(0, './stratified_dynamics')
    from stratified_dynamics.equations import *


class equilibrium_solver(Equations):

    def __init__(self, nz, Lz, dimensions=1, dealias=3/2, grid_dtype=np.float64, **kwargs):
        self.nz = nz
        self.Lz = Lz
        super(equilibrium_solver, self).__init__(dimensions=dimensions, **kwargs)
        self._set_domain(nz=nz, Lz=Lz,  comm=MPI.COMM_SELF, dealias=dealias,
                         grid_dtype=grid_dtype)

    def set_parameters(self):
        pass

    def run_BVP(self, bc_dict, *args, tolerance=1e-10, **kwargs):
        self.set_parameters(*args, **kwargs)
        self.set_subs()
        self.set_equations()
        self.set_bcs(bc_dict)
        solver = self.problem.build_solver()
        self.define_diagnostics(solver)
        # Iterations
        pert = solver.perturbations.data
        pert.fill(1+tolerance)
        while np.sum(np.abs(pert)) > tolerance:
            solver.newton_iteration(damping=1)
            logger.info('Perturbation norm: {}'.format(np.sum(np.abs(pert))))
            solver.evaluator.evaluate_group("diagnostics")
            self.output_diagnostics()
        self.final_diagnostics()
        return solver

    def define_diagnostics(self, solver):
        self.diagnostics = solver.evaluator.add_dictionary_handler(group='diagnostics')
    
    def output_diagnostics(self):
        pass
    
    def final_diagnostics(self):
        pass


class FC_equilibrium_solver(equilibrium_solver):

    def __init__(self, *args, ncc_cutoff=1e-10, dealias=3/2, **kwargs):
        super(FC_equilibrium_solver, self).__init__(*args, dealias=dealias, **kwargs)
        self.problem = de.NLBVP(self.domain, variables=['T1', 'ln_rho1', 'T1_z', 'M1'], ncc_cutoff=ncc_cutoff)

    def set_parameters(self, T, rho, g=2.5, Cp=2.5, gamma=5/3):
        T0, rho0 = self._new_ncc(), self._new_ncc()
        T0['g'] = T
        rho0['g'] = rho
        self.problem.parameters['g']         = g
        self.problem.parameters['Cp']        = Cp
        self.problem.parameters['gamma']     = gamma
        self.problem.parameters['Lz']        = self.Lz
        self.problem.parameters['T0']        = T0
        self.problem.parameters['rho0']      = rho0

    def set_subs(self):
        self.problem.substitutions['grad_T_ad'] = '-g/Cp'
        self.problem.substitutions['m_ad']      = '1/(gamma-1)'

        self.problem.substitutions['T0_z']      = 'dz(T0)'
        self.problem.substitutions['ln_rho0']   = 'log(rho0)'
        self.problem.substitutions['rho_full']   = 'rho0*exp(ln_rho1)'
        self.problem.substitutions['ln_rho_full']= '(ln_rho0 + ln_rho1)'
        self.problem.substitutions['T_full']   = '(T0 + T1)'
        self.problem.substitutions['T_full_z']   = '(T0_z + T1_z)'
        self.problem.substitutions['s']   = '(log(T_full) - (gamma-1)*ln_rho_full)/gamma'

        self.define_kappa()
        self.problem.substitutions['kappa_fluc'] = '(kappa(T_full, ln_rho_full) - kappa(T0, ln_rho0))'
        self.problem.substitutions['FluxKap(T, ln_rho)']     = '-kappa(T, ln_rho)*dz(T)'
        self.problem.substitutions['dz_ln_kappa_fluc'] = "(dz_ln_kappa(T_full, ln_rho_full) - dz_ln_kappa(T0, ln_rho0))"

    def define_kappa(self):
        self.problem.substitutions['kappa(T, ln_rho)'] = '1'
        self.problem.substitutions['ln_kappa(T, ln_rho)'] = "0"
        self.problem.substitutions['dz_ln_kappa(T, ln_rho)'] = "0"

    def set_equations(self):
        self.problem.add_equation("T1_z - dz(T1) = 0")
        self.problem.add_equation("dz(M1) = rho_full - rho0")
        self.problem.add_equation("T0*dz(ln_rho1) + T1*dz(ln_rho0) + T1_z = -T1*dz(ln_rho1) - g - T0*dz(ln_rho0) - T0_z")
        self.set_thermal_equilibrium()

    def set_bcs(self, bc_dict):

#        flux_str = "{}(T1_z) = {}(-(kappa_fluc*T0_z)/kappa(T_full, ln_rho_full) )"
#        flux_str = "{}(kappa(T0, ln_rho0)*T1_z) = {}(-(kappa_fluc*T_full_z) )"
        flux_str = "{}((1+a)*ln_rho1) = {}((3-b)*log(1 + T1/T0) + log(1 + T1_z/T0_z))"
        if bc_dict['mixed_flux_temperature']:
            logger.info("Fixed T (top) / Fixed flux (bot)")
            self.problem.add_bc("right(T1)          = 0")
            self.problem.add_bc(flux_str.format("left", "left"))
        elif bc_dict['mixed_temperature_flux']:
            logger.info("Fixed flux (top) / Fixed T (bot)")
            self.problem.add_bc("left(T1)          = 0")
            self.problem.add_bc(flux_str.format("right", "right"))
        else:
            logger.error("Boundary conditions for fixed flux / fixed temperature not implemented.")
            import sys
            sys.exit()
    
        self.problem.add_bc("left(M1) = 0")
        self.problem.add_bc("right(M1) = 0")

    def define_diagnostics(self, solver):
        super(FC_equilibrium_solver, self).define_diagnostics(solver)
        self.diagnostics.add_task('FluxKap(T_full, ln_rho_full)',name='flux')
        self.diagnostics.add_task('FluxKap(T0, ln_rho0)',name='flux0')
        self.diagnostics.add_task('-kappa(ln_rho_full,T_full)*grad_T_ad',name='flux_ad')
        self.diagnostics.add_task('dz(-kappa(ln_rho_full,T_full)*dz(T_full))',name='div_flux')
        self.diagnostics.add_task('1/gamma*dz(T_full)/T_full - (gamma-1)/gamma*dz(ln_rho_full)',name='dsdz_Cp')
        self.diagnostics.add_task('1/gamma*log(T_full) - (gamma-1)/gamma*ln_rho_full',name='s_Cp')
        self.diagnostics.add_task('T_full',name='T')
        self.diagnostics.add_task('ln_rho_full',name='ln_rho')
        self.diagnostics.add_task('dz(T_full) + T_full*dz(ln_rho_full) + g',name='HSE')
        self.diagnostics.add_task('dz(FluxKap(T_full, ln_rho_full))',name='TE')

    def output_diagnostics(self):
        logger.info('rho iterate:  {}--{}'.format(self.diagnostics['ln_rho']['g'][-1],self.diagnostics['ln_rho']['g'][0]))
        logger.info('flux iterate: ({})--({})'.format(min(self.diagnostics['flux']['g']),max(self.diagnostics['flux']['g'])))

    def final_diagnostics(self):
        ln_rho_bot = self.diagnostics['ln_rho'].interpolate(z=0)['g'][0]
        ln_rho_top = self.diagnostics['ln_rho'].interpolate(z=self.Lz)['g'][0]
        logger.info("flux(z):\n {}".format(self.diagnostics['flux']['g']))
        logger.info("s(z)/Cp:\n {}".format(self.diagnostics['s_Cp']['g']))
        logger.info("dsdz(z)/Cp:\n {}".format(self.diagnostics['dsdz_Cp']['g']))
        logger.info("HSE:\n {}".format(self.diagnostics['HSE']['g']))
        logger.info("TE:\n {}".format(self.diagnostics['TE']['g']))
        logger.info("n_rho_fluc = {}".format(ln_rho_bot - ln_rho_top))
        logger.info("delta s: {}".format(np.max(self.diagnostics['s_Cp'].interpolate(z=self.Lz)['g']) - np.max(self.diagnostics['s_Cp'].interpolate(z=0)['g'])))

    def set_thermal_equilibrium(self):
        self.problem.add_equation("-dz(T1_z) - T1_z*dz_ln_kappa(T0, ln_rho0) = dz(T0_z) + (T0_z*dz_ln_kappa(T0, ln_rho0)) + T0_z*dz_ln_kappa_fluc + T1_z*dz_ln_kappa_fluc")
        

    

class FC_kramers_equilibrium_solver(FC_equilibrium_solver):
        
    def __init__(self, *args, dealias=2, **kwargs):
        super(FC_kramers_equilibrium_solver, self).__init__(*args, dealias=dealias, **kwargs)
        
    def set_parameters(self, a, b, *args, **kwargs):
        super(FC_kramers_equilibrium_solver, self).set_parameters(*args, **kwargs)
        logger.info('setting kramers equilibrium solver with a: {}, b: {}'.format(a,b))
        self.problem.parameters['a'] = a
        self.problem.parameters['b'] = b

    def define_kappa(self):
        self.problem.substitutions['kappa(T, ln_rho)'] = "((exp(ln_rho))**(-1-a)*(T)**(3-b))"
        self.problem.substitutions['ln_kappa(T, ln_rho)'] = "(-(1+a)*ln_rho + (3-b)*log(T))"
        self.problem.substitutions['dz_ln_kappa(T, ln_rho)'] = "(-(1+a)*dz(ln_rho) + (3-b)*dz(T)/T)"

#    def set_thermal_equilibrium(self):
#        self.problem.add_equation(("-(1+a)*(dz(ln_rho1)*dz(T0)+dz(ln_rho0)*dz(T1)) + dz(T1_z) ="
#                      " (1+a)*(dz(ln_rho0)*dz(T0)+dz(ln_rho1)*dz(T1))"
#                      "-(3-b)*(dz(T_full)**2/T_full) - dz(T0_z)"))
#

