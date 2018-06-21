import numpy as np
import scipy.special as scp
import os
from mpi4py import MPI

from collections import OrderedDict

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from tools import analysis
except:
    from sys import path
    path.insert(0, './tools')
    from ..tools import analysis

import logging
logger = logging.getLogger(__name__.split('.')[-1])

from dedalus import public as de
from dedalus.core.field import Field

from scipy.optimize import minimize_scalar

class Atmosphere:
    def __init__(self, verbose=False, fig_dir='./', **kwargs):
        self._set_domain(**kwargs) # this should not happen here; should happen in equations

        self.make_plots = verbose
        self.fig_dir = fig_dir + '/'

        if self.fig_dir[-1] != '/':
            self.fig_dir += '/'
        if self.domain.dist.comm_cart.rank == 0 and not os.path.exists(self.fig_dir):
            os.mkdir(self.fig_dir)

    def evaluate_at_point(self, f, z=0):
        return f.interpolate(z=z)

    def value_at_boundary(self, field):
        orig_scale = field.meta[:]['scale']
        try:
            field_top    = self.evaluate_at_point(field, z=self.Lz)['g'][0][0]
            if not np.isfinite(field_top):
                logger.info("Likely interpolation error at top boundary; setting field=1")
                logger.info("orig_scale: {}".format(orig_scale))
                field_top = 1
            field_bottom = self.evaluate_at_point(field, z=0)['g'][0][0]
            field.set_scales(orig_scale, keep_data=True)
        except:
            logger.debug("field at top shape {}".format(field['g'].shape))
            field_top = None
            field_bottom = None

        return field_bottom, field_top

    def _set_atmosphere(self):
        self.necessary_quantities = OrderedDict()

        self.phi = self._new_ncc()
        self.necessary_quantities['phi'] = self.phi

        self.del_ln_rho0 = self._new_ncc()
        self.rho0 = self._new_ncc()
        self.necessary_quantities['del_ln_rho0'] = self.del_ln_rho0
        self.necessary_quantities['rho0'] = self.rho0

        self.del_s0 = self._new_ncc()
        self.necessary_quantities['del_s0'] = self.del_s0

        self.T0_zz = self._new_ncc()
        self.T0_z = self._new_ncc()
        self.T0 = self._new_ncc()
        self.necessary_quantities['T0_zz'] = self.T0_zz
        self.necessary_quantities['T0_z'] = self.T0_z
        self.necessary_quantities['T0'] = self.T0

        self.del_P0 = self._new_ncc()
        self.P0 = self._new_ncc()
        self.necessary_quantities['del_P0'] = self.del_P0
        self.necessary_quantities['P0'] = self.P0

        self.nu = self._new_ncc()
        self.chi = self._new_ncc()
        self.nu_l = self._new_ncc()
        self.chi_l = self._new_ncc()
        self.del_chi_l = self._new_ncc()
        self.del_nu_l = self._new_ncc()
        self.necessary_quantities['nu_l'] = self.nu_l
        self.necessary_quantities['chi_l'] = self.chi_l
        self.necessary_quantities['del_chi_l'] = self.del_chi_l
        self.necessary_quantities['del_nu_l'] = self.del_nu_l
        self.nu_r = self._new_ncc()
        self.chi_r = self._new_ncc()
        self.del_chi_r = self._new_ncc()
        self.del_nu_r = self._new_ncc()
        self.necessary_quantities['nu_r'] = self.nu_r
        self.necessary_quantities['chi_r'] = self.chi_r
        self.necessary_quantities['del_chi_r'] = self.del_chi_r
        self.necessary_quantities['del_nu_r'] = self.del_nu_r

        self.scale = self._new_ncc()
        self.scale_continuity = self._new_ncc()
        self.scale_energy = self._new_ncc()
        self.scale_momentum = self._new_ncc()
        self.necessary_quantities['scale'] = self.scale
        self.necessary_quantities['scale_continuity'] = self.scale_continuity
        self.necessary_quantities['scale_energy'] = self.scale_energy
        self.necessary_quantities['scale_momentum'] = self.scale_momentum

    def copy_atmosphere(self, atmosphere):
        '''
        Copies values from a target atmosphere into the current atmosphere.
        '''
        self.necessary_quantities = atmosphere.necessary_quantities

    def plot_atmosphere(self):
        for key in self.problem.parameters:
            try:
                self.problem.parameters[key].require_layout(self.domain.dist.layouts[1])
                logger.debug("plotting atmospheric quantity {}".format(key))
                fig_q = plt.figure()
                ax = fig_q.add_subplot(2,1,1)
                quantity = self.problem.parameters[key]
                quantity.set_scales(1, keep_data=True)
                ax.plot(self.z[0,:], quantity['g'][0,:])
                if np.min(quantity['g'][0,:]) != np.max(quantity['g'][0,:]):
                    ax.set_ylim(np.min(quantity['g'][0,:])-0.05*np.abs(np.min(quantity['g'][0,:])),
                            np.max(quantity['g'][0,:])+0.05*np.abs(np.max(quantity['g'][0,:])))
                ax.set_xlabel('z')
                ax.set_ylabel(key)

                ax = fig_q.add_subplot(2,1,2)
                power_spectrum = np.abs(quantity['c'][0,:]*np.conj(quantity['c'][0,:]))
                ax.plot(np.arange(len(quantity['c'][0,:])), power_spectrum)
                ax.axhline(y=1e-20, color='black', linestyle='dashed') # ncc_cutoff = 1e-10
                ax.set_xlabel('z')
                ax.set_ylabel("Tn power spectrum: {}".format(key))
                ax.set_yscale("log", nonposy='clip')
                ax.set_xscale("log", nonposx='clip')

                fig_q.savefig(self.fig_dir+"{:s}_p{:d}.png".format(key, self.domain.distributor.rank), dpi=300)
                plt.close(fig_q)
            except:
                logger.debug("printing atmospheric quantity {}={}".format(key, self.problem.parameters[key]))

        fig_m = plt.figure()
        ax_m = fig_m.add_subplot(1,1,1)
        del_ln_T0 = self.problem.parameters['T0_z']['g'][0,:]/self.problem.parameters['T0']['g'][0,:]
        ax_m.plot(self.z[0,:], self.problem.parameters['del_ln_rho0']['g'][0,:]/del_ln_T0, label=r'$m$')
        ax_m.axhline(y=self.m_ad, label=r'$m_{ad}$', color='black', linestyle='dashed')
        ax_m.legend()
        fig_m.savefig(self.fig_dir+"polytropic_m_p{}.png".format(self.domain.distributor.rank), dpi=300)

        for key in ['T0', 'P0', 'rho0', 'del_s0']:
            self.necessary_quantities[key].require_layout(self.domain.dist.layouts[1])
        fig_atm = plt.figure()
        axT = fig_atm.add_subplot(2,2,1)
        axT.plot(self.z[0,:], self.necessary_quantities['T0']['g'][0,:])
        axT.set_ylabel('T0')
        axP = fig_atm.add_subplot(2,2,2)
        axP.semilogy(self.z[0,:], self.necessary_quantities['P0']['g'][0,:])
        axP.set_ylabel('P0')
        axR = fig_atm.add_subplot(2,2,3)
        axR.semilogy(self.z[0,:], self.necessary_quantities['rho0']['g'][0,:])
        axR.set_ylabel(r'$\rho0$')
        axS = fig_atm.add_subplot(2,2,4)
        analysis.semilogy_posneg(axS, self.z[0,:], self.necessary_quantities['del_s0']['g'][0,:], color_neg='red')

        axS.set_ylabel(r'$\nabla s0$')
        fig_atm.savefig(self.fig_dir+"quantities_p{}.png".format(self.domain.distributor.rank), dpi=300)

        fig_atm = plt.figure()
        ax1 = fig_atm.add_subplot(2,2,1)
        ax2 = fig_atm.add_subplot(2,2,2)
        ax3 = fig_atm.add_subplot(2,2,3)
        ax4 = fig_atm.add_subplot(2,2,4)

        Cv_inv = self.gamma-1
        ax1.plot(self.z[0,:], 1/Cv_inv*np.log(self.T0['g'][0,:]) - 1/Cv_inv*(self.gamma-1)*np.log(self.rho0['g'][0,:]), label='s', linewidth=2)
        ax1.axhline(y=0, color='black', linestyle='dashed')
        ax1.set_ylabel(r'$s$')

        ax2.plot(self.z[0,:], self.del_s0['g'][0,:], label=r'$\nabla s$', linewidth=2)
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")
        ax2.set_ylabel(r'$\nabla s$')

        ax3.plot(self.z[0,:], self.problem.parameters['del_ln_rho0']['g'][0,:]/del_ln_T0, label=r'$m(z)$')
        ax3.axhline(y=self.m_ad, color='black', linestyle='dashed', label=r'$m_{ad}$')
        ax3.set_ylabel(r'$m(z)=\nabla \ln \rho/\nabla \ln T$')
        ax3.legend(loc="center left")

        ax4.plot(self.z[0,:], np.log(self.P0['g'][0,:]),   label=r'$\ln$P',    linewidth=2)
        ax4.plot(self.z[0,:], np.log(self.rho0['g'][0,:]), label=r'$\ln \rho$', linewidth=2)
        ax4.yaxis.tick_right()
        ax4.yaxis.set_label_position("right")
        ax4.set_ylabel(r'$\ln$P,$\ln \rho$')
        ax4.legend()
        plt.tight_layout()
        fig_atm.savefig(self.fig_dir+"atmosphere_parameters_p{}.png".format(self.domain.distributor.rank), dpi=300)

    def check_that_atmosphere_is_set(self):
        for key in self.necessary_quantities:
            quantity = self.necessary_quantities[key]['g']
            quantity_set = quantity.any()
            if not quantity_set:
                logger.info("WARNING: atmosphere {} is all zeros on process 0".format(key))

    def test_hydrostatic_balance(self, P_z=None, P=None, T=None, rho=None, make_plots=False):

        if rho is None:
            logger.error("HS balance test requires rho (currently)")
            raise

        if P_z is None:
            if P is None:
                if T is None:
                    logger.error("HS balance test requires P_z, P or T")
                    raise
                else:
                    T_scales = T.meta[:]['scale']
                    rho_scales = rho.meta[:]['scale']
                    if rho_scales != 1:
                        rho.set_scales(1, keep_data=True)
                    if T_scales != 1:
                        T.set_scales(1, keep_data=True)
                    P = self._new_field()
                    T.set_scales(self.domain.dealias, keep_data=True)
                    rho.set_scales(self.domain.dealias, keep_data=True)
                    P.set_scales(self.domain.dealias, keep_data=False)
                    P['g'] = T['g']*rho['g']
                    T.set_scales(T_scales, keep_data=True)
                    rho.set_scales(rho_scales, keep_data=True)

            P_z = self._new_field()
            P.differentiate('z', out=P_z)
            P_z.set_scales(1, keep_data=True)

        rho_scales = rho.meta[:]['scale']
        rho.set_scales(1, keep_data=True)
        # error in hydrostatic balance diagnostic
        HS_balance = P_z['g']+self.g*rho['g']
        relative_error = HS_balance/P_z['g']
        rho.set_scales(rho_scales, keep_data=True)

        HS_average = self._new_field()
        HS_average['g'] = HS_balance
        if self.dimensions > 1:
            HS_average.integrate('x')
            HS_average['g'] /= self.Lx
        HS_average.set_scales(1, keep_data=True)

        relative_error_avg = self._new_field()
        relative_error_avg['g'] = relative_error
        if self.dimensions > 1:
            relative_error_avg.integrate('x')
            relative_error_avg['g'] /= self.Lx
        relative_error_avg.set_scales(1, keep_data=True)

        if self.make_plots or make_plots:
            fig = plt.figure()
            ax1 = fig.add_subplot(2,1,1)
            if self.dimensions > 1:
                ax1.plot(self.z[0,:], P_z['g'][0,:])
                ax1.plot(self.z[0,:], -self.g*rho['g'][0,:])
            else:
                ax1.plot(self.z[:], P_z['g'][:])
                ax1.plot(self.z[:], -self.g*rho['g'][:])
            ax1.set_ylabel(r'$\nabla P$ and $\rho g$')
            ax1.set_xlabel('z')

            ax2 = fig.add_subplot(2,1,2)
            if self.dimensions > 1:
                ax2.semilogy(self.z[0,:], np.abs(relative_error[0,:]))
                ax2.semilogy(self.z[0,:], np.abs(relative_error_avg['g'][0,:]))
            else:
                ax2.semilogy(self.z[:], np.abs(relative_error[:]))
                ax2.semilogy(self.z[:], np.abs(relative_error_avg['g'][:]))
            ax2.set_ylabel(r'$|\nabla P + \rho g |/|\nabla P|$')
            ax2.set_xlabel('z')
            fig.savefig(self.fig_dir+"HS_balance_p{}.png".format(self.domain.distributor.rank), dpi=300)

        max_rel_err = self.domain.dist.comm_cart.allreduce(np.max(np.abs(relative_error)), op=MPI.MAX)
        max_rel_err_avg = self.domain.dist.comm_cart.allreduce(np.max(np.abs(relative_error_avg['g'])), op=MPI.MAX)
        logger.info('max error in HS balance: point={} avg={}'.format(max_rel_err, max_rel_err_avg))

    def check_atmosphere(self, make_plots=False, **kwargs):
        if self.make_plots or make_plots:
            try:
                self.plot_atmosphere()
            except:
                logger.info("Problems in plot_atmosphere: atm full of NaNs?")
        self.test_hydrostatic_balance(make_plots=make_plots, **kwargs)
        self.check_that_atmosphere_is_set()

class Polytrope(Atmosphere):
    '''
    Single polytrope, stable or unstable.
    '''
    def __init__(self,
                 nx=256, Lx=None,
                 ny=256, Ly=None,
                 nz=128, Lz=None,
                 aspect_ratio=4,
                 n_rho_cz = 3,
                 m_cz=None, epsilon=1e-4, gamma=5/3,
                 constant_kappa=True, constant_mu=True,
                 **kwargs):
        
        self.atmosphere_name = 'single polytrope'
        self.aspect_ratio    = aspect_ratio
        self.n_rho_cz        = n_rho_cz

        self._set_atmosphere_parameters(gamma=gamma, epsilon=epsilon, poly_m=m_cz)
        if m_cz is None:
            m_cz = self.poly_m

        if Lz is None:
            if n_rho_cz is not None:
                Lz = self._calculate_Lz_cz(n_rho_cz, m_cz)
            else:
                logger.error("Either Lz or n_rho must be set")
                raise
        if Lx is None:
            Lx = Lz*aspect_ratio
        if Ly is None:
            Ly = Lx
        self.d_conv          = Lz
            
        super(Polytrope, self).__init__(nx=nx, ny=ny, nz=nz, Lx=Lx, Ly=Ly, Lz=Lz, **kwargs)
        logger.info("   Lx = {:g}, Lz = {:g}".format(self.Lx, self.Lz))
        self.z0 = 1. + self.Lz
       
        self.constant_kappa = constant_kappa
        self.constant_mu    = constant_mu
        if self.constant_kappa == False and self.constant_mu == False:
            self.constant_diffusivities = True
        else:
            self.constant_diffusivities = False

        self._set_atmosphere()
        self._set_timescales()

    def _calculate_Lz_cz(self, n_rho_cz, m_cz):
        '''
        Calculate Lz based on the number of density scale heights and the initial polytrope.
        '''
        #The absolute value allows for negative m_cz.
        Lz_cz = np.exp(n_rho_cz/np.abs(m_cz))-1
        return Lz_cz
    
    def _set_atmosphere_parameters(self, gamma=5/3, epsilon=0, poly_m=None, g=None):
        # polytropic atmosphere characteristics
        self.gamma = gamma
        self.Cv = 1/(self.gamma-1)
        self.Cp = self.gamma*self.Cv
        self.epsilon = epsilon

        self.m_ad = 1/(self.gamma-1)

        # trap on poly_m/epsilon conflicts?
        if poly_m is None:
            self.poly_m = self.m_ad - self.epsilon
        else:
            self.poly_m = poly_m

        self.m_cz = self.poly_m

        if g is None:
            self.g = self.poly_m + 1
        else:
            self.g = g

        logger.info("polytropic atmosphere parameters:")
        logger.info("   poly_m = {:g}, epsilon = {:g}, gamma = {:g}".format(self.poly_m, self.epsilon, self.gamma))
    
    def _set_atmosphere(self):
        super(Polytrope, self)._set_atmosphere()

        self.del_ln_rho_factor = -self.poly_m
        self.del_ln_rho0['g'] = self.del_ln_rho_factor/(self.z0 - self.z)
        self.rho0['g'] = (self.z0 - self.z)**self.poly_m

        self.del_s0_factor = - self.epsilon 
        self.delta_s = self.del_s0_factor*np.log(self.z0)
        self.del_s0['g'] = self.del_s0_factor/(self.z0 - self.z)
 
        self.T0_zz['g'] = 0        
        self.T0_z['g'] = -1
        self.T0['g'] = self.z0 - self.z       

        self.P0['g'] = (self.z0 - self.z)**(self.poly_m+1)
        self.P0.differentiate('z', out=self.del_P0)
        self.del_P0.set_scales(1, keep_data=True)
        self.P0.set_scales(1, keep_data=True)
        
        if self.constant_diffusivities:
            self.scale['g']            = (self.z0 - self.z)
            self.scale_continuity['g'] = (self.z0 - self.z)
            self.scale_momentum['g']   = (self.z0 - self.z)
            self.scale_energy['g']     = (self.z0 - self.z)
        else:
            # consider whether to scale nccs involving chi differently (e.g., energy equation)
            self.scale['g']            = (self.z0 - self.z)
            self.scale_continuity['g'] = (self.z0 - self.z)
            self.scale_momentum['g']   = (self.z0 - self.z)# **np.ceil(self.m_cz)
            self.scale_energy['g']     = (self.z0 - self.z)# **np.ceil(self.m_cz)

        # choose a particular gauge for phi (g*z0); and -grad(phi)=g_vec=-g*z_hat
        # double negative is correct.
        self.phi['g'] = -self.g*(self.z0 - self.z)

        rho0_max, rho0_min = self.value_at_boundary(self.rho0)
        if rho0_max is not None:
            try:
		# For "strange" resolutions (e.g., 96x192), sometimes this crashes.  Need to investigate. (9/12/2017)
                rho0_ratio = rho0_max/rho0_min
                logger.info("   density: min {}  max {}".format(rho0_min, rho0_max))
                logger.info("   density scale heights = {:g} (measured)".format(np.log(rho0_ratio)))
                logger.info("   density scale heights = {:g} (target)".format(np.log((self.z0)**self.poly_m)))
            except:
                if self.domain.distributor.comm_cart.rank == 0:
                    logger.error("Something went wrong with reporting density range")
            
        H_rho_top = (self.z0-self.Lz)/self.poly_m
        H_rho_bottom = (self.z0)/self.poly_m
        logger.info("   H_rho = {:g} (top)  {:g} (bottom)".format(H_rho_top,H_rho_bottom))
        if self.delta_x != None:
            logger.info("   H_rho/delta x = {:g} (top)  {:g} (bottom)".format(H_rho_top/self.delta_x,
                                                                          H_rho_bottom/self.delta_x))
        
    def _set_timescales(self, atmosphere=None):
        if atmosphere is None:
            atmosphere=self
            
        # min of global quantity
        atmosphere.min_BV_time = self.domain.dist.comm_cart.allreduce(np.min(np.sqrt(np.abs(self.g*self.del_s0['g']/self.Cp))), op=MPI.MIN)
        atmosphere.freefall_time = np.sqrt(self.Lz/self.g)
        atmosphere.buoyancy_time = np.sqrt(np.abs(self.Lz*self.Cp / (self.g * self.delta_s)))
        
        logger.info("atmospheric timescales:")
        logger.info("   min_BV_time = {:g}, freefall_time = {:g}, buoyancy_time = {:g}".format(atmosphere.min_BV_time,
                                                                                               atmosphere.freefall_time,
                                                                                               atmosphere.buoyancy_time))
    def _set_diffusivities(self, Rayleigh=1e6, Prandtl=1, split_diffusivities=False):
       
        logger.info("problem parameters:")
        logger.info("   Ra = {:g}, Pr = {:g}".format(Rayleigh, Prandtl))
        self.Rayleigh, self.Prandtl = Rayleigh, Prandtl

        # set nu and chi at top based on Rayleigh number
        self.nu_top = nu_top = np.sqrt(Prandtl*(self.Lz**3*np.abs(self.delta_s/self.Cp)*self.g)/Rayleigh)
        self.chi_top = chi_top = nu_top/Prandtl

        if self.constant_diffusivities:
            # take constant nu, chi
            nu = nu_top
            chi = chi_top

            logger.info("   using constant nu, chi")
            logger.info("   nu = {:g}, chi = {:g}".format(nu, chi))
        else:
            if self.constant_kappa:
                self.rho0.set_scales(1, keep_data=True)
                if not split_diffusivities:
                    chi_l = chi_top/(self.rho0['g'])
                    chi_r = 0
                else:
                    if self.poly_m < 1:
                        chi_l = np.exp(self.n_rho_cz)*chi_top/(self.z0 - self.z)
                    else:
                        chi_l = chi_top/(self.z0 - self.z)
                    chi_r = chi_top/(self.rho0['g']) - chi_l
                logger.info('using constant kappa')
            else:
                chi_l = chi_top
                chi_r = 0
                logger.info('using constant chi')
            if self.constant_mu:
                self.rho0.set_scales(1, keep_data=True)
                if not split_diffusivities:
                    nu_l  = nu_top/(self.rho0['g'])
                    nu_r = 0
                else:
                    if self.poly_m < 1:
                        nu_l  = np.exp(self.n_rho_cz)*nu_top/(self.z0 - self.z)
                    else:
                        nu_l  = nu_top/(self.z0 - self.z)
                    nu_r  = nu_top/(self.rho0['g']) - nu_l
                logger.info('using constant mu')
            else:
                nu_l  = nu_top
                nu_r = 0
                logger.info('using constant nu')

      
            logger.info("   nu_top = {:g}, chi_top = {:g}".format(nu_top, chi_top))

        #Allows for atmosphere reuse
        self.chi_l.set_scales(1, keep_data=True)
        self.nu_l.set_scales(1, keep_data=True)
        self.chi_r.set_scales(1, keep_data=True)
        self.nu_r.set_scales(1, keep_data=True)
        self.nu.set_scales(1, keep_data=True)
        self.chi.set_scales(1, keep_data=True)
        self.nu_l['g'] = nu_l
        self.chi_l['g'] = chi_l
        self.nu_r['g'] = nu_r
        self.chi_r['g'] = chi_r
        self.nu['g'] = nu_l + nu_r
        self.chi['g'] = chi_l + chi_r

        self.chi_l.differentiate('z', out=self.del_chi_l)
        self.chi_l.set_scales(1, keep_data=True)
        self.nu_l.differentiate('z', out=self.del_nu_l)
        self.nu_l.set_scales(1, keep_data=True)
        self.chi_r.differentiate('z', out=self.del_chi_r)
        self.chi_r.set_scales(1, keep_data=True)
        self.nu_r.differentiate('z', out=self.del_nu_r)
        self.nu_r.set_scales(1, keep_data=True)

        # determine characteristic timescales; use chi and nu at middle of domain for bulk timescales.
        self.thermal_time = self.Lz**2/np.mean(self.chi.interpolate(z=self.Lz/2)['g'])
        self.top_thermal_time = 1/chi_top

        self.viscous_time = self.Lz**2/np.mean(self.nu.interpolate(z=self.Lz/2)['g'])
        self.top_viscous_time = 1/nu_top

        if self.dimensions == 2:
            self.thermal_time = self.thermal_time#[0]
            self.viscous_time = self.viscous_time#[0]
        if self.dimensions > 2:
            #Need to communicate across processes if mesh is weird in 3D
            therm = np.zeros(1, dtype=np.float64)
            visc  = np.zeros(1, dtype=np.float64)
            therm_rcv, visc_rcv = np.zeros_like(therm), np.zeros_like(visc)
            therm[0] = np.mean(self.thermal_time)
            visc[0]  = np.mean(self.viscous_time)
            if np.isnan(therm): therm[0] = 0
            if np.isnan(visc):  visc[0]  = 0
            self.domain.dist.comm_cart.Allreduce(therm, therm_rcv, op=MPI.MAX)
            self.thermal_time = therm_rcv[0]
            self.domain.dist.comm_cart.Allreduce(visc, visc_rcv, op=MPI.MAX)
            self.viscous_time = visc_rcv[0]

        logger.info("thermal_time = {}, top_thermal_time = {}".format(self.thermal_time,
                                                                      self.top_thermal_time))
        self.nu.set_scales(1, keep_data=True)
        self.chi.set_scales(1, keep_data=True)

    def save_atmosphere_file(self, data_dir):
        #This creates an output file that contains all of the useful atmospheric info at the beginning of the run
        out_dir = data_dir + '/atmosphere/'
        out_file = out_dir + 'atmosphere.h5'
        if self.domain.dist.rank == 0:
            if not os.path.exists('{:s}'.format(out_dir)):
                os.mkdir('{:s}'.format(out_dir))
            f = h5py.File('{:s}'.format(out_file), 'w')
        indxs = [0]*self.dimensions
        indxs[-1] = range(self.nz)
        key_set = list(self.problem.parameters.keys())
        extended_keys = ['chi','nu','del_chi','del_nu']
        key_set.extend(extended_keys)
        logger.debug("Outputing atmosphere parameters for {}".format(key_set))
        for key in key_set:
            # Figure out what type of data we're dealing with
            if 'scale' in key:
                continue
            if key in extended_keys:
                field_key = True
            elif type(self.problem.parameters[key]) == Field:
                field_key = True
                self.problem.parameters[key].set_scales(1, keep_data=True)
            else:
                field_key = False

            # Get the proper data
            if field_key:
                try:
                    if key in extended_keys:
                        self.problem.parameters[key+'_l'].require_layout(self.domain.dist.layouts[1])
                        self.problem.parameters[key+'_r'].require_layout(self.domain.dist.layouts[1])
                        array = self.problem.parameters[key+'_l'].data[indxs] +\
                                self.problem.parameters[key+'_r'].data[indxs]
                    else:
                        self.problem.parameters[key].require_layout(self.domain.dist.layouts[1])
                        array = self.problem.parameters[key].data[indxs]
                except:
                    if self.domain.dist.rank == 0:
                        logger.error("key error on atmosphere output {}".format(key))
                    array = 0
                if self.domain.dist.rank == 0:
                    f[key] = array
            elif self.domain.dist.rank == 0:
                f[key] = self.problem.parameters[key]
        
        z_value = self.domain.bases[-1].grid(1)
        if self.domain.dist.rank == 0:
            f['z'] = z_value

        if self.domain.dist.rank == 0:
            f['dimensions']     = self.dimensions
            if self.dimensions > 1:
                f['nx']             = self.nx
            if self.dimensions > 2:
                f['ny']             = self.ny
            f['nz']             = self.nz
            f['m_ad']           = self.m_ad
            f['m']              = self.m_ad - self.epsilon
            f['epsilon']        = self.epsilon
            f['n_rho_cz']       = self.n_rho_cz
            f['rayleigh']       = self.Rayleigh
            f['prandtl']        = self.Prandtl
            f['aspect_ratio']   = self.aspect_ratio
            f['atmosphere_name']= self.atmosphere_name
            f['t_buoy']         = self.buoyancy_time
            f['t_therm']        = self.thermal_time
            f.close()




class KramerPolytrope(Polytrope):
    '''
    A single polytropic layer with a perfectly adiabatic stratficiation.
    The thermodynamic variables are non-dimensionalized at the top of the
    atmosphere, such that
    
        T = rho = 1 at z = Lz

    The depth of the atmosphere is specified by the number of density
    scale heights, Lz = np.exp(n_rho_cz/m_ad) - 1, where m_ad = (gamma-1)^-1.
    The stratification follows

        T(z)   = (1 + Lz - z)
        rho(z) = (1 + Lz - z)^m_ad

    When rayleigh = epsilon = 1, the flux entering through the bottom of the
    atmosphere *should* be one non-dimensional unit. Epsilon controls the
    size of perturbations from the adiabatic state (as in Anders&Brown2017),
    and Ra controls the level of turbulence.  The conductivity is

        kappa = kappa_0 * rho^(-1-a) * T^(3-b),

    where a = 1 and b = -7/2 falls off like a kramers opacity for free-free
    interactions.  Since kappa can shrink by orders of magnitude over the
    depth of the atmosphere, there is a subgrid scale (SGS) flux, which is
    zero throughout most of the domain but which carries
    the flux across the upper boundary and sets the characteristic scale of
    entropy perturbations (through epsilon).

    The prandtl number sets the viscous diffusivity based off of the SGS
    diffusivity at the upper boundary. In general, this means that Pr >> 1
    unless Pr is very small. Nu, the viscous diffusivity, is constant
    with height, so Pr varies  with depth.
    '''
    def __init__(self,
                 kram_a=1, kram_b=-7/2,
                 **kwargs):

        self.kram_a, self.kram_b = kram_a, kram_b
        kwargs['epsilon'] = 0
        super(KramerPolytrope, self).__init__(**kwargs)
        self.delta_s = self.epsilon = self.kram_b
        self._set_timescales()

    def _set_atmosphere_parameters(self, **kwargs):
        super(KramerPolytrope, self)._set_atmosphere_parameters(**kwargs)

        self.m_kram = (3 - self.kram_b)/(1 + self.kram_a)
        logger.info("   poly_kram = {:g}".format(self.m_kram))

    def _set_timescales(self, **kwargs):
        super(KramerPolytrope, self)._set_timescales(**kwargs)

    def _set_diffusivities(self, Rayleigh, Prandtl, split_diffusivities=False):
        logger.info("problem parameters:")
        logger.info("   Ra = {:g}, Pr = {:g}".format(Rayleigh, Prandtl))
        self.Rayleigh, self.Prandtl = Rayleigh, Prandtl

        # set chi at top based on Rayleigh number. We're treating Ra as being propto chi^-2 in this formulation.
        self.chi_top = chi_top = np.sqrt(np.abs(self.delta_s)*self.Lz**3 * self.g \
                                        /(Rayleigh*Prandtl))
        
        kappa_0 = self.chi_top / (np.exp(self.n_rho_cz*(-(1+self.kram_a) + (3 - self.kram_b)/self.poly_m)))
        self.kappa = self._new_ncc()
        self.kappa['g'] = kappa_0 *   (self.T0['g']/np.exp(self.n_rho_cz/self.poly_m))**(3-self.kram_b)*\
                                      (self.rho0['g']/np.exp(self.n_rho_cz))**(-(1+self.kram_a)) 

        self.rho0.set_scales(1, keep_data=True)
        self.chi.set_scales(1, keep_data=True)
        self.kappa.set_scales(1, keep_data=True)
        self.chi['g'] = self.kappa['g']/self.rho0['g']

        self.nu_top = nu_top = Prandtl*np.max(self.chi.interpolate(z=self.Lz)['g'])
        self.nu = self._new_ncc()
        self.nu['g'] = self.nu_top
        logger.info("chi top: {}; nu top: {}".format(self.chi_top, self.nu_top))
        logger.info("Pr top: {}, Pr bot: {}".format(self.nu_top, self.nu_top/np.max(self.chi.interpolate(z=self.Lz)['g']), self.nu_top/np.max(self.chi.interpolate(z=0)['g'])))
        self.kappa.set_scales(1, keep_data=True)
        self.T0_z.set_scales(1, keep_data=True)

        self.problem.parameters['kram_a']   = self.kram_a
        self.problem.parameters['kram_b'] = self.kram_b
        self.problem.parameters['κ_C'] = self.kappa

        self.thermal_time = self.Lz**2/np.mean(self.chi.interpolate(z=self.Lz/2)['g'])
        self.top_thermal_time = 1/chi_top

        self.viscous_time = self.Lz**2/np.mean(self.nu.interpolate(z=self.Lz/2)['g'])
        self.top_viscous_time = 1/nu_top


        if self.dimensions == 2:
            self.thermal_time = self.thermal_time#[0]
            self.viscous_time = self.viscous_time#[0]
        if self.dimensions > 2:
            #Need to communicate across processes if mesh is weird in 3D
            therm = np.zeros(1, dtype=np.float64)
            visc  = np.zeros(1, dtype=np.float64)
            therm_rcv, visc_rcv = np.zeros_like(therm), np.zeros_like(visc)
            therm[0] = np.mean(self.thermal_time)
            visc[0]  = np.mean(self.viscous_time)
            if np.isnan(therm): therm[0] = 0
            if np.isnan(visc):  visc[0]  = 0
            self.domain.dist.comm_cart.Allreduce(therm, therm_rcv, op=MPI.MAX)
            self.thermal_time = therm_rcv[0]
            self.domain.dist.comm_cart.Allreduce(visc, visc_rcv, op=MPI.MAX)
            self.viscous_time = visc_rcv[0]

        logger.info("thermal_time = {}, top_thermal_time = {}".format(self.thermal_time, self.top_thermal_time))
        self.nu.set_scales(1, keep_data=True)
        self.chi.set_scales(1, keep_data=True)


class Multitrope(Atmosphere):
    '''
    Multiple joined polytropes ("tropes").  Each trope starts in hydrostatic and thermal equlibrium.
    The thermal diffusion profile κ determines the temperature gradients, and κ is currently joined
    by a smooth matching function specified in "match_Phi()".

    Parameters
    ----------
    nx : int
          Horizontal (Fourier) resolution
    nz : list of ints
          set of resolutions for possible compound domain; a single entry means a single-chebyshev will span all layers
    n_rho : list of floats
          target density contrast in each trope; order matches m (default: [1, 3])
    m : list of floats
          polytropic index of each trope; order matches n_rho (default: [3, 1.5*(1-0.01)] or cz above rz with stiffness of 100)
    g : float
          gravity; constant with depth
    gamma : float
          ratio of specific heats for ideal gas; gamma = c_p/c_v (default: 5/3)
    constant_Prandtl : logical
          maintain constant Prandtl number Pr=μ/κ (default: True)
    aspect_ratio : float
          aspect ratio of domain (horizontal/vertical) for computation (default: 4)
    width : float
    overshoot_pad : float
    pad_for_overshoot : logical


    '''
    def __init__(self, nx=256,
                 aspect_ratio=4,
                 gamma=5/3,
                 nz=[128, 128],
                 n_rho = [1, 3],
                 m = [3, 1.5*(1-0.01)],
                 reference_index = None,
                 g = None,
                 width=None,
                 overshoot_pad = None,
                 pad_for_overshoot = False,
                 constant_Prandtl=True,
                 **kwargs):
        # polytropic atmosphere characteristics
        # stiffness = (m_rz - m_ad)/(m_ad - m_cz) = (m_rz - m_ad)/epsilon

        self.atmosphere_name = 'multitrope'

        if reference_index is None:
            reference_index = m.index(min(m))
        self.reference_index = reference_index
        self.gamma = gamma
        self.Cv = 1/(self.gamma-1)
        self.Cp = self.gamma*self.Cv
        self.m_ad = 1/(gamma-1)
        self.m = m
        self.epsilon = self.m_ad - m[reference_index]

        if g is None:
            self.g = self.m[reference_index] + 1

            self.g = self.m[-1] + 1

        else:
            self.g = g

        self.n_rho = n_rho

        self.L_list = L_list = self._calculate_Lz(n_rho, m)

        self.aspect_ratio = aspect_ratio

        Lx = L_list[reference_index]*aspect_ratio

        self.match_center = L_list[0]
        self.Lz = np.sum(L_list)

        # this is going to widen the tanh and move the location of del(s)=0 as n_rho_cz increases...
        # match_width = 4% of Lz_cz, somewhat analgous to Rogers & Glatzmaier 2005
        erf_v_tanh = 18.5/(np.sqrt(np.pi)*6.5/2)
        if width is None:
            width = 0.04*erf_v_tanh
        logger.info("erf width factor is {} of L_ref (total: {})".format(width, width*L_list[reference_index]))
        self.match_width = width*L_list[reference_index] # adjusted by ~3x for erf() vs tanh()

        if pad_for_overshoot:
            if overshoot_pad is None:
                overshoot_pad = 2*self.match_width
            logger.info("using overshoot_pad = {} and match_width = {}".format(overshoot_pad, self.match_width))

            L_list[0] -= overshoot_pad
            L_list[1] += overshoot_pad
        super(Multitrope, self).__init__(nx=nx, nz=nz, Lx=Lx, Lz=L_list, **kwargs)

        self.constant_Prandtl = constant_Prandtl
        self.constant_diffusivities = False

        self.Lz_ref = L_list[reference_index]
        self.z_cz = self.Lz_ref + 1
        self._set_atmosphere()
        logger.info("Done set_atmosphere")

        # Set tapering function for initial conditions
        self.IC_taper = self._new_ncc()
        if self.reference_index == 0:
            self.IC_taper['g'] = self.match_Phi(self.z, self.Lz_ref, 0.1*self.Lz_ref)
            self.IC_taper['g'] *= np.sin(np.pi*(self.z)/self.Lz_ref)
        else:
            self.IC_taper['g'] = 1-self.match_Phi(self.z, self.Lz_ref, 0.1*self.Lz_ref)
            self.IC_taper['g'] *= np.sin(np.pi*(self.z-self.Lz_ref)/(self.Lz - self.Lz_ref))

        T0_max, T0_min = self.value_at_boundary(self.T0)
        P0_max, P0_min = self.value_at_boundary(self.P0)
        rho0_max, rho0_min = self.value_at_boundary(self.rho0)

        logger.info("   temperature: min {}  max {}".format(T0_min, T0_max))
        logger.info("   pressure: min {}  max {}".format(P0_min, P0_max))
        logger.info("   density: min {}  max {}".format(rho0_min, rho0_max))

        if rho0_max is not None:
            rho0_ratio = rho0_max/rho0_min
            logger.info("   density scale heights = {:g}".format(np.log(rho0_ratio)))
            logger.info("   target n_rho = {}".format(self.n_rho))
            logger.info("   target n_rho_total = {:g}".format(np.sum(self.n_rho)))
        H_rho_top = (self.z_cz-self.Lz_ref)/self.m[reference_index]
        H_rho_bottom = (self.z_cz)/self.m[reference_index]
        logger.info("   H_rho = {:g} (top CZ)  {:g} (bottom CZ)".format(H_rho_top,H_rho_bottom))
        if self.dimensions > 1:
            logger.info("   H_rho/delta x = {:g} (top CZ)  {:g} (bottom CZ)".format(H_rho_top/self.delta_x,\
                                                                          H_rho_bottom/self.delta_x))

        logger.info("   m = {}".format(m))
        logger.info("   m - m_ad = {}".format(np.array(self.m)-self.m_ad))

        self._set_timescales()

    def _set_timescales(self, atmosphere=None):
        if atmosphere is None:
            atmosphere=self
        # min of global quantity
        BV_time = np.sqrt(np.abs(self.g*self.del_s0['g']/self.Cp))
        if BV_time.shape[-1] == 0:
            logger.debug("BV_time {}, shape {}".format(BV_time, BV_time.shape))
            BV_time = np.array([np.inf])

        self.min_BV_time = self.domain.dist.comm_cart.allreduce(np.min(BV_time), op=MPI.MIN)
        self.freefall_time = np.sqrt(self.Lz_ref/self.g)
        self.buoyancy_time = np.sqrt(self.Lz_ref/self.g/np.abs(self.epsilon))

        logger.info("atmospheric timescales:")
        logger.info("   min_BV_time = {:g}, freefall_time = {:g}, buoyancy_time = {:g}".format(self.min_BV_time,
                                                                                               self.freefall_time,
                                                                                               self.buoyancy_time))

    def _calculate_Lz(self, n_rho_list, m_list):
        '''
        Estimate the depth of the CZ and the RZ.
        '''
        # T = del_T*(z-z_interface) + T_interface
        # del_T = -g/(m+1) = -(m_cz+1)/(m+1)
        # example: cz: z_interface = L_cz (top), T_interface = 1, del_T = -1
        #     .: T = -1*(z - L_cz) + 1 = (L_cz + 1 - z) = (z0 - z)
        # this recovers the Lecoanet et al 2014 notation
        #
        # n_rho =  ln(rho_bot/rho_i) = m*ln(T/T_i)
        #     T =  T_i*exp(n_rho/m)
        # del_T = -(m_i+1)/(m + 1)
        #
        # L = (T - T_i)/del_T = (np.exp(n_rho/m)-1)*T_i/del_T
        #
        # we build lengths from the top of the atmosphere down, but specify tropes from the bottom up:
        L_list = []
        kappa_list = []
        first_trope_built = False
        for n_rho, m in zip(reversed(n_rho_list), reversed(m_list)):
            if not first_trope_built:
                T_i = 1
                del_T = -1
                first_trope_built = True
            else:
                del_T = -(m_i+1)/(m+1)
            L = (np.exp(n_rho/m)-1)* T_i/(-del_T)
            logger.info("T:{},del_T:{}".format(T_i, del_T))
            # update for next cycle
            T_i += -del_T*L
            m_i = m
            L_list.append(L)
            logger.info("calc_Lz {}, {}, {}".format(m, n_rho, L))

        rev_L_list = reversed(L_list)
        L_list = []
        for L in rev_L_list:
            L_list.append(L)

        logger.info("Calculating scales {}".format(L_list))
        return L_list

    def match_Phi(self, z, center, width, f=scp.erf):
        return 1/2*(1-f((z-center)/width))

    def _set_atmosphere(self, atmosphere_type2=False):
        super(Multitrope, self)._set_atmosphere()

        self.delta_s = self.epsilon*np.log(self.z_cz)
        logger.info("Atmosphere delta s is {}".format(self.delta_s))

        # choose a particular gauge for phi (g*z0); and -grad(phi)=g_vec=-g*z_hat
        # double negative is correct.
        self.phi['g'] = -self.g*(self.z_cz - self.z)

        # this doesn't work: numerical instability blowup, and doesn't reduce bandwidth much at all
        #self.scale['g'] = (self.z_cz - self.z)
        # this seems to work fine; bandwidth only a few terms worse.
        self.scale['g'] = 1.
        self.scale_continuity['g'] = 1.
        self.scale_momentum['g'] = 1.
        self.scale_energy['g'] = 1.

        self.kappa = self._new_ncc()
        # general for 2 m's, but not for more than 2 m's
        # Think about step function specification closer.
        kappa_ratio = []
        for m in self.m:
            kappa_ratio.append((m+1)/(self.m[-1] + 1))
        #kappa_ratio = (self.m[1] + 1)/(self.m[0] + 1)
        # specify kappa as smoothly matched profile
        logger.info("match_center {}".format(self.match_center))
        Phi = self.match_Phi(self.z, self.match_center, self.match_width)
        inv_Phi = 1-Phi
        # a simple, smooth step function with amplitudes 1 and kappa_ratio
        # on either side of the matching region
        profile = Phi*kappa_ratio[0] + inv_Phi*kappa_ratio[1]
        logger.info("kappa ratio: {}".format(kappa_ratio))
        #if kappa_ratio < 1:
        #    profile = Phi/kappa_ratio + inv_Phi
        #else:
        #    profile = Phi + inv_Phi*kappa_ratio

        self.kappa['g'] = profile

        logger.info("Solving for T0")
        # start with an arbitrary -1 at the top, which will be rescaled after _set_diffusivites
        flux_top = -1
        self.T0_z['g'] = flux_top/self.kappa['g']

        self.T0_z.antidifferentiate('z',('right',0), out=self.T0)
        # need T0_zz in multitrope
        self.T0_z.differentiate('z', out=self.T0_zz)
        self.T0['g'] += 1
        self.T0.set_scales(1, keep_data=True)

        self.del_ln_P0 = self._new_ncc()
        self.ln_P0 = self._new_ncc()
        self.necessary_quantities['ln_P0'] = self.ln_P0
        self.necessary_quantities['del_ln_P0'] = self.del_ln_P0

        logger.info("Solving for P0")
        # assumes ideal gas equation of state
        self.del_ln_P0['g'] = -self.g/self.T0['g']
        self.del_ln_P0.antidifferentiate('z',('right',0),out=self.ln_P0)
        self.ln_P0.set_scales(1, keep_data=True)
        self.P0['g'] = np.exp(self.ln_P0['g'])
        self.del_ln_P0.set_scales(1, keep_data=True)
        self.del_P0['g'] = self.del_ln_P0['g']*self.P0['g']
        self.del_P0.set_scales(1, keep_data=True)

        self.rho0['g'] = self.P0['g']/self.T0['g']

        self.rho0.differentiate('z', out=self.del_ln_rho0)
        self.del_ln_rho0['g'] = self.del_ln_rho0['g']/self.rho0['g']

        self.rho0.set_scales(1, keep_data=True)
        self.del_ln_P0.set_scales(1, keep_data=True)
        self.del_ln_rho0.set_scales(1, keep_data=True)
        self.del_s0['g'] = 1/self.gamma*self.del_ln_P0['g'] - self.del_ln_rho0['g']

    def _set_diffusivities(self, Rayleigh=1e6, Prandtl=1, split_diffusivities=False):
        #TODO: Implement split_diffusivities
        logger.info("problem parameters (multitrope):")
        logger.info("   Ra = {:g}, Pr = {:g}".format(Rayleigh, Prandtl))
        self.Rayleigh = Rayleigh_top = Rayleigh
        self.Prandtl = Prandtl_top = Prandtl

        self.constant_mu = False
        self.constant_kappa = False
        # inputs:
        # Rayleigh_top = g dS L_cz**3/(chi_top**2 * Pr_top)
        # Prandtl_top = nu_top/chi_top
        logger.info('setting chi')
        self.chi_top = np.sqrt((self.g*(self.delta_s/self.Cp)*self.Lz_ref**3)/(Rayleigh_top*Prandtl_top))
        if self.reference_index == 0:
            # try to rescale chi appropriately so that the
            # Rayleigh number is set at the top of the CZ
            # to the desired value by removing the density
            # scaling from the rz.  This is a guess.
            # NOTE: currently it's setting us to Prandtl = 1/exp(n_rho_rz), basically, in oz land.
            self.chi_top = np.exp(self.n_rho[self.reference_index])*self.chi_top

        #Set Prandtl number at same place as Ra.
        self.nu_top = self.chi_top*Prandtl_top

        self.kappa['g'] *= self.chi_top
        self.kappa.set_scales(self.domain.dealias, keep_data=True)
        self.rho0.set_scales(self.domain.dealias, keep_data=True)
        self.chi_l.set_scales(self.domain.dealias, keep_data=True)
        if self.rho0['g'].shape[-1] != 0:
            self.chi_l['g'] = self.kappa['g']/self.rho0['g']
            self.chi_l.differentiate('z', out=self.del_chi_l)
            self.chi_l.set_scales(1, keep_data=True)

        logger.info("setting nu")
        if self.constant_Prandtl:
            self.kappa.set_scales(self.domain.dealias, keep_data=True)
            self.rho0.set_scales(self.domain.dealias, keep_data=True)
            self.nu_l.set_scales(self.domain.dealias, keep_data=True)
            if self.rho0['g'].shape[-1] != 0:
                self.nu_l['g'] = (self.nu_top/self.chi_top)*self.kappa['g']/self.rho0['g']
                self.nu_l.differentiate('z', out=self.del_nu_l)
                self.nu_l.set_scales(1, keep_data=True)
        else:
            self.nu_l['g'] = self.nu_top
            self.nu_l.differentiate('z', out=self.del_nu_l)

        # rescale kappa to correct values based on Rayleigh number derived chi

        self.nu_r['g'] = 0
        self.chi_r['g'] = 0
        self.chi_l.set_scales(1, keep_data=True)
        self.nu_l.set_scales(1, keep_data=True)
        self.chi_r.set_scales(1, keep_data=True)
        self.nu_r.set_scales(1, keep_data=True)
        self.nu.set_scales(1, keep_data=True)
        self.chi.set_scales(1, keep_data=True)
        self.nu['g'] = self.nu_l['g'] + self.nu_r['g']
        self.chi['g'] = self.chi_l['g'] + self.chi_r['g']

        self.chi_l.differentiate('z', out=self.del_chi_l)
        self.chi_l.set_scales(1, keep_data=True)
        self.nu_l.differentiate('z', out=self.del_nu_l)
        self.nu_l.set_scales(1, keep_data=True)
        self.chi_r.differentiate('z', out=self.del_chi_r)
        self.chi_r.set_scales(1, keep_data=True)
        self.nu_r.differentiate('z', out=self.del_nu_r)
        self.nu_r.set_scales(1, keep_data=True)

        self.top_thermal_time = 1/self.chi_top
        self.thermal_time = self.Lz_ref**2/self.chi_top
        logger.info("done times")
        logger.info("   nu_top = {:g}, chi_top = {:g}".format(self.nu_top, self.chi_top))
        logger.info("thermal_time = {:g}, top_thermal_time = {:g}".format(self.thermal_time,
                                                                          self.top_thermal_time))
    def get_flux(self, rho, T):
        rho.set_scales(1,keep_data=True)
        T_z = self._new_ncc()
        T.differentiate('z', out=T_z)
        T_z.set_scales(1,keep_data=True)
        chi = self.chi_l
        chi_l.set_scales(1,keep_data=True)
        flux = self._new_ncc()
        flux['g'] = rho['g']*T_z['g']*chi_l['g']
        return flux
