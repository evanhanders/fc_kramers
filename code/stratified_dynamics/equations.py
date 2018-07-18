import numpy as np
from mpi4py import MPI
import scipy.special as scp

from collections import OrderedDict

import logging
logger = logging.getLogger(__name__.split('.')[-1])

from dedalus import public as de


class Equations():
    def __init__(self, dimensions=2, max_ncc_bandwidth=None):
        self.dimensions=dimensions
        self.problem_type = ''
        self.max_ncc_bandwidth=max_ncc_bandwidth
        print(self.max_ncc_bandwidth)
        pass

    def _set_domain(self, nx=256, Lx=4,
                          ny=256, Ly=4,
                          nz=128, Lz=1,
                          grid_dtype=np.float64, comm=MPI.COMM_WORLD, mesh=None, dealias=3/2):
        # the naming conventions here force cartesian, generalize to spheres etc. make sense?
        self.mesh=mesh
        
        if not isinstance(nz, list):
            nz = [nz]
        if not isinstance(Lz, list):
            Lz = [Lz]
            
        if len(nz)>1:
            logger.info("Setting compound basis in vertical (z) direction")
            z_basis_list = []
            Lz_interface = 0.
            for Lz_i, nz_i in zip(Lz,nz):
                Lz_top = Lz_i+Lz_interface
                z_basis = de.Chebyshev('z', nz_i, interval=[Lz_interface, Lz_top], dealias=dealias)
                z_basis_list.append(z_basis)
                Lz_interface = Lz_top
            self.compound = True
            z_basis = de.Compound('z', tuple(z_basis_list),  dealias=dealias)
        elif len(nz)==1:
            logger.info("Setting single chebyshev basis in vertical (z) direction")
            self.compound = False
            z_basis = de.Chebyshev('z', nz[0], interval=[0, np.sum(Lz)], dealias=dealias)
        
        if self.dimensions > 1:
            x_basis = de.Fourier(  'x', nx, interval=[0., Lx], dealias=dealias)
        if self.dimensions > 2:
            y_basis = de.Fourier(  'y', ny, interval=[0., Ly], dealias=dealias)
        if self.dimensions == 1:
            bases = [z_basis]
        elif self.dimensions == 2:
            bases = [x_basis, z_basis]
        elif self.dimensions == 3:
            bases = [x_basis, y_basis, z_basis]
        else:
            logger.error('>3 dimensions not implemented')
        
        self.domain = de.Domain(bases, grid_dtype=grid_dtype, comm=comm, mesh=mesh)
        
        self.z = self.domain.grid(-1) # need to access globally-sized z-basis
        self.Lz = self.domain.bases[-1].interval[1] - self.domain.bases[-1].interval[0] # global size of Lz
        self.nz = self.domain.bases[-1].coeff_size

        self.z_dealias = self.domain.grid(axis=-1, scales=self.domain.dealias)

        if self.dimensions == 1:
            self.x, self.Lx, self.nx, self.delta_x = None, 0, None, None
            self.y, self.Ly, self.ny, self.delta_y = None, 0, None, None
        if self.dimensions > 1:
            self.x = self.domain.grid(0)
            self.Lx = Lx#self.domain.bases[0].interval[1] - self.domain.bases[0].interval[0] # global size of Lx
            self.nx = nx#self.domain.bases[0].coeff_size
            self.delta_x = self.Lx/self.nx
        if self.dimensions > 2:
            self.y = self.domain.grid(1)
            self.Ly = Ly#self.domain.bases[1].interval[1] - self.domain.bases[0].interval[0] # global size of Lx
            self.ny = ny#self.domain.bases[1].coeff_size
            self.delta_y = self.Ly/self.ny
            
        if self.dimensions == 3:
            out_string = "   Lx = {:g}, Ly = {:g}, Lz = {:g} ({}))".format(self.Lx, self.Ly, self.Lz, Lz)            
        elif self.dimensions == 2:
            out_string = "   Lx = {:g}, Lz = {:g} ({}))".format(self.Lx, self.Lz, Lz)
        elif self.dimensions == 1:
            out_string = "   Lz = {:g} ({})".format(self.Lz, Lz)
        logger.info(out_string)

    def set_IVP_problem(self, *args, ncc_cutoff=1e-10, **kwargs):
        self.problem_type = 'IVP'
        self.problem = de.IVP(self.domain, variables=self.variables, ncc_cutoff=ncc_cutoff)
        self.set_equations(*args, **kwargs)

    def set_eigenvalue_problem(self, *args, ncc_cutoff=1e-10, **kwargs):
        # should be set EVP for consistency with set IVP.  Why do we have P_problem.  Why not IVP, EVP.
        self.problem_type = 'EVP'
        self.problem = de.EVP(self.domain, variables=self.variables, eigenvalue='omega', ncc_cutoff=ncc_cutoff, tolerance=1e-10)
        self.problem.substitutions['dt(f)'] = "omega*f"
        self.set_equations(*args, **kwargs)

    def get_problem(self):
        return self.problem

    def _new_ncc(self):
        # is this used at all in equations.py (other than rxn), or just in atmospheres?
        # the naming conventions here force cartesian, generalize to spheres etc. make sense?
        # should "necessary quantities" logic occur here?
        field = self.domain.new_field()
        if self.dimensions > 1:
            field.meta['x']['constant'] = True
        if self.dimensions > 2:
            field.meta['y']['constant'] = True            
        return field

    def _new_field(self):
        field = self.domain.new_field()
        return field

    def _set_subs(self):
        pass

    def global_noise(self, seed=42, **kwargs):            
        # Random perturbations, initialized globally for same results in parallel
        gshape = self.domain.dist.grid_layout.global_shape(scales=self.domain.dealias)
        slices = self.domain.dist.grid_layout.slices(scales=self.domain.dealias)
        rand = np.random.RandomState(seed=seed)
        noise = rand.standard_normal(gshape)[slices]

        # filter in k-space
        noise_field = self._new_field()
        noise_field.set_scales(self.domain.dealias, keep_data=False)
        noise_field['g'] = noise
        self.filter_field(noise_field, **kwargs)

        return noise_field

    def filter_field(self, field,frac=0.25, fancy_filter=False):
        dom = field.domain
        logger.info("filtering field {} with frac={} using a set-scales approach".format(field.name,frac))
        orig_scale = field.meta[:]['scale']
        field.set_scales(frac, keep_data=True)
        field['c']
        field['g']
        field.set_scales(orig_scale, keep_data=True)

    def _gather_field(self, field):
        field.set_scales(1, keep_data=True)
        z_profile_local = np.zeros(self.nz)
        z_profile_global = np.zeros_like(z_profile_local)
        field.set_scales(1, keep_data=True)
        if self.mesh is None:
            n_per_proc = len(z_profile_local)/self.domain.dist.comm_cart.size
            rank = self.domain.dist.comm_cart.rank
            indices = [0]*len(field['g'].shape)
            indices[-1] = range(field['g'].shape[-1])
            z_profile_local[rank*n_per_proc:(rank+1)*n_per_proc] = field['g'][indices]
            self.domain.dist.comm_cart.Allreduce(z_profile_local, z_profile_global, op=MPI.SUM)
        else:
            n_per_proc = len(z_profile_local)/self.mesh[0]
            rank = self.domain.dist.comm_cart.rank
            if self.rank < self.mesh[0]:
                z_profile_local[rank*n_per_proc:(rank+1)*n_per_proc] = field['g'][0,0,:]
            self.domain.dist.comm_cart.Allreduce(z_profile_local, z_profile_global, op=MPI.SUM)
        return z_profile_global  

    def _set_field(self, field, profile):
        if self.mesh is None:
            n_per_proc = len(profile)/self.domain.dist.comm_cart.size
            rank = self.domain.dist.comm_cart.rank
            field.set_scales(1, keep_data=True)
            field['g'] = profile[rank*n_per_proc:(rank+1)*n_per_proc]
        else:
            n_per_proc = len(profile)/self.mesh[0]
            rank = self.domain.dist.comm_cart.rank % self.mesh[0]
            field.set_scales(1, keep_data=True)
            field['g'] = profile[rank*n_per_proc:(rank+1)*n_per_proc]
            
class FC_equations(Equations):
    def __init__(self, **kwargs):
        super(FC_equations, self).__init__(**kwargs)

    def _set_parameters(self):
        '''
        Basic parameters needed for fully compressible equations in stratified atmosphere.
        '''
        self.problem.parameters['Lz'] = self.Lz
        if self.dimensions > 1:
            self.problem.parameters['Lx'] = self.Lx
        if self.dimensions > 2:
            self.problem.parameters['Ly'] = self.Ly

        # these are all ideal gas; also should be in atmosphere, breaks consistency currently.
        # momentum equation and thermal equation also probably bake in ideal gas presently.
        # EOS related on a fumdanental level.
        self.problem.parameters['gamma'] = self.gamma
        self.problem.parameters['Cv'] = 1/(self.gamma-1)
        self.problem.parameters['Cv_inv'] = self.gamma-1
        self.problem.parameters['Cp'] = self.gamma/(self.gamma-1)
        self.problem.parameters['Cp_inv'] = (self.gamma-1)/self.gamma
        
        # thermodynamic quantities
        # these assume stuff is stored in self. and have particular names.  They come from atmosphere things.
        # go to NCC dictionary?  All keys could be defined on init, and this could all be handled by a 3-line for loop.
        # need an analysis dictionary and keyset as well, since some things used there and not in eqns.
        self.problem.parameters['T0'] = self.T0
        self.problem.parameters['T0_z'] = self.T0_z
        self.problem.parameters['T0_zz'] = self.T0_zz
        
        self.problem.parameters['rho0'] = self.rho0
        self.problem.parameters['del_ln_rho0'] = self.del_ln_rho0
                    
        self.problem.parameters['del_s0'] = self.del_s0

        # gravity
        self.problem.parameters['g']  = self.g
        self.problem.parameters['phi']  = self.phi

        # scaling factor to reduce NCC bandwidth of all equations
        self.problem.parameters['scale'] = self.scale
        self.problem.parameters['scale_continuity'] = self.scale_continuity
        self.problem.parameters['scale_momentum'] = self.scale_momentum
        self.problem.parameters['scale_energy'] = self.scale_energy

        # diffusivities
        self.problem.parameters['nu_l'] = self.nu_l
        self.problem.parameters['chi_l'] = self.chi_l
        self.problem.parameters['del_chi_l'] = self.del_chi_l
        self.problem.parameters['del_nu_l'] = self.del_nu_l
        self.problem.parameters['nu_r'] = self.nu_r
        self.problem.parameters['chi_r'] = self.chi_r
        self.problem.parameters['del_chi_r'] = self.del_chi_r
        self.problem.parameters['del_nu_r'] = self.del_nu_r

        # Thermo subs that are used later, but before set_subs() is called; okay or not okay?
        self.problem.parameters['delta_s_atm'] = self.delta_s

        # this first one (rho_full) is the one that doesn't fit in set_subs() when we used kappa and mu as primary variables, because the viscous subs need this to get anywhere.
        self.problem.substitutions['rho_full'] = 'rho0*exp(ln_rho1)' 
        self.problem.substitutions['rho_fluc'] = 'rho0*(exp(ln_rho1)-1)'
        self.problem.substitutions['ln_rho0']  = 'log(rho0)'
        self.problem.substitutions['ln_rho_full'] = '(ln_rho0 + ln_rho1)'
        self.problem.substitutions['T_full']      = '(T0 + T1)'
        self.problem.substitutions['s_fluc'] = '((1/Cv_inv)*log(1+T1/T0) - ln_rho1)'
        self.problem.substitutions['s_mean'] = '((1/Cv_inv)*log(T0) - ln_rho0)'
        self.problem.substitutions['m_ad']    = '((gamma-1)**-1)'

    def _set_operators(self):
        # differential operators
        self.problem.substitutions['Lap(f, f_z)'] = "(dx(dx(f)) + dy(dy(f)) + dz(f_z))"
        self.problem.substitutions['Div(fx, fy, fz_z)'] = "(dx(fx) + dy(fy) + fz_z)"
        self.problem.substitutions['Div_u'] = "Div(u, v, w_z)"
        self.problem.substitutions['UdotGrad(f, f_z)'] = "(u*dx(f) + v*dy(f) + w*(f_z))"
        
        self.problem.substitutions["σxx"] = "(2*dx(u) - 2/3*Div_u)"
        self.problem.substitutions["σyy"] = "(2*dy(v) - 2/3*Div_u)"
        self.problem.substitutions["σzz"] = "(2*w_z   - 2/3*Div_u)"
        self.problem.substitutions["σxy"] = "(dx(v) + dy(u))"
        self.problem.substitutions["σxz"] = "(dx(w) +  u_z )"
        self.problem.substitutions["σyz"] = "(dy(w) +  v_z )"

        self.problem.substitutions['ω_x'] = '(dy(w) - v_z)'        
        self.problem.substitutions['ω_y'] = '( u_z  - dx(w))'        
        self.problem.substitutions['ω_z'] = '(dx(v) - dy(u))'        
        self.problem.substitutions['enstrophy']   = '(ω_x**2 + ω_y**2 + ω_z**2)'

    def _set_diffusion_subs(self):        
        # define nu and chi for output
        if self.split_diffusivities:
            self.problem.substitutions['nu']  = '(nu_l + nu_r)'
            self.problem.substitutions['del_nu']  = '(del_nu_l + del_nu_r)'
            self.problem.substitutions['chi'] = '(chi_l + chi_r)'
            self.problem.substitutions['del_chi'] = '(del_chi_l + del_chi_r)'
        else:
            self.problem.substitutions['nu']  = '(nu_l)'
            self.problem.substitutions['del_nu']  = '(del_nu_l)'
            self.problem.substitutions['chi'] = '(chi_l)'
            self.problem.substitutions['del_chi'] = '(del_chi_l)'

        self.viscous_term_u_l = " nu_l*(Lap(u, u_z) + 1/3*Div(dx(u), dx(v), dx(w_z)))"
        self.viscous_term_v_l = " nu_l*(Lap(v, v_z) + 1/3*Div(dy(u), dy(v), dy(w_z)))"
        self.viscous_term_w_l = " nu_l*(Lap(w, w_z) + 1/3*Div(  u_z, v_z, dz(w_z)))"
        self.viscous_term_u_r = " nu_r*(Lap(u, u_z) + 1/3*Div(dx(u), dx(v), dx(w_z)))"
        self.viscous_term_v_r = " nu_r*(Lap(v, v_z) + 1/3*Div(dy(u), dy(v), dy(w_z)))"
        self.viscous_term_w_r = " nu_r*(Lap(w, w_z) + 1/3*Div(  u_z, v_z, dz(w_z)))"
        # here, nu and chi are constants                
        if not self.constant_mu:
            self.viscous_term_u_l += " + (nu_l*del_ln_rho0 + del_nu_l) * σxz"
            self.viscous_term_w_l += " + (nu_l*del_ln_rho0 + del_nu_l) * σzz"
            self.viscous_term_v_l += " + (nu_l*del_ln_rho0 + del_nu_l) * σyz"
            self.viscous_term_u_r += " + (nu_r*del_ln_rho0 + del_nu_r) * σxz"
            self.viscous_term_w_r += " + (nu_r*del_ln_rho0 + del_nu_r) * σzz"
            self.viscous_term_v_r += " + (nu_r*del_ln_rho0 + del_nu_r) * σyz"

        self.problem.substitutions['L_visc_w'] = self.viscous_term_w_l
        self.problem.substitutions['L_visc_u'] = self.viscous_term_u_l
        self.problem.substitutions['L_visc_v'] = self.viscous_term_v_l
        
        self.nonlinear_viscous_u = " nu*(dx(ln_rho1)*σxx + dy(ln_rho1)*σxy + dz(ln_rho1)*σxz)"
        self.nonlinear_viscous_v = " nu*(dx(ln_rho1)*σxy + dy(ln_rho1)*σyy + dz(ln_rho1)*σyz)"
        self.nonlinear_viscous_w = " nu*(dx(ln_rho1)*σxz + dy(ln_rho1)*σyz + dz(ln_rho1)*σzz)"
        if self.split_diffusivities:
            self.nonlinear_viscous_u += " + {}".format(self.viscous_term_u_r)
            self.nonlinear_viscous_v += " + {}".format(self.viscous_term_v_r)
            self.nonlinear_viscous_w += " + {}".format(self.viscous_term_w_r)
 
        self.problem.substitutions['R_visc_u'] = self.nonlinear_viscous_u
        self.problem.substitutions['R_visc_v'] = self.nonlinear_viscous_v
        self.problem.substitutions['R_visc_w'] = self.nonlinear_viscous_w

        # double check implementation of variabile chi and background coupling term.
        self.linear_thermal_diff_l    = " Cv_inv*(chi_l*(Lap(T1, T1_z) + T0_z*dz(ln_rho1)))"
        self.linear_thermal_diff_r    = " Cv_inv*(chi_r*(Lap(T1, T1_z) + T0_z*dz(ln_rho1)))"
        self.nonlinear_thermal_diff   = " Cv_inv*chi*(dx(T1)*dx(ln_rho1) + dy(T1)*dy(ln_rho1) + T1_z*dz(ln_rho1))"
        self.source =                   " Cv_inv*(chi*(T0_zz))"
        if not self.constant_kappa:
            self.linear_thermal_diff_l += '+ Cv_inv*(chi_l*del_ln_rho0 + del_chi_l)*T1_z'
            self.linear_thermal_diff_r += '+ Cv_inv*(chi_r*del_ln_rho0 + del_chi_r)*T1_z'
            self.source                += '+ Cv_inv*(chi*del_ln_rho0 + del_chi)*T0_z'

        if self.split_diffusivities:
            self.nonlinear_thermal_diff += " + {}".format(self.linear_thermal_diff_r)
        self.problem.substitutions['L_thermal']    = self.linear_thermal_diff_l
        self.problem.substitutions['R_thermal']   = self.nonlinear_thermal_diff
        self.problem.substitutions['source_terms'] = self.source

        self.problem.substitutions['R_visc_heat'] = " Cv_inv*nu*(dx(u)*σxx + dy(v)*σyy + w_z*σzz + σxy**2 + σxz**2 + σyz**2)"

        self.problem.substitutions['kappa_flux_mean'] = '-rho0*chi*dz(T0)'
        self.problem.substitutions['kappa_flux_fluc'] = '(-rho_full*chi*dz(T1) - rho_fluc*chi*dz(T0))'
        
    def _set_subs(self):
        # does both analysis subs and equation subs currently.
        self.problem.substitutions['plane_std(A)'] = 'sqrt(plane_avg((A - plane_avg(A))**2))'
        # other anaylsis operations (vol avg, etc.) currently set in 2-D and 3-D extensions.  Good or bad?

        # output parameters
        self.problem.substitutions['Rayleigh_global'] = 'g*Lz**3*delta_s_atm*Cp_inv/(nu*chi)'
        self.problem.substitutions['Rayleigh_local']  = 'g*Lz**4*dz(s_mean+s_fluc)*Cp_inv/(nu*chi)'

        self.problem.substitutions['epsilon_0'] = 'log(T0**(1/(gamma-1))/rho0)/log(T0)'
        self.problem.substitutions['epsilon'] = 'log(T_full**(1/(gamma-1))/rho_full)/log(T_full)'
        
        self.problem.substitutions['vel_rms'] = 'sqrt(u**2 + v**2 + w**2)'
        self.problem.substitutions['KE'] = 'rho_full*(vel_rms**2)/2'
        self.problem.substitutions['PE'] = 'rho_full*phi'
        self.problem.substitutions['PE_fluc'] = 'rho_fluc*phi'
        self.problem.substitutions['IE'] = 'rho_full*Cv*(T1+T0)'
        self.problem.substitutions['IE_fluc'] = 'rho_full*Cv*T1+rho_fluc*Cv*T0'
        self.problem.substitutions['P'] = 'rho_full*(T1+T0)'
        self.problem.substitutions['P_fluc'] = 'rho_full*T1+rho_fluc*T0'
        self.problem.substitutions['h'] = '(IE + P)'
        self.problem.substitutions['h_fluc'] = '(IE_fluc + P_fluc)'
        self.problem.substitutions['u_rms'] = 'sqrt(u**2)'
        self.problem.substitutions['v_rms'] = 'sqrt(v**2)'
        self.problem.substitutions['w_rms'] = 'sqrt(w**2)'
        self.problem.substitutions['Re_rms'] = 'vel_rms*Lz/nu'
        self.problem.substitutions['Pe_rms'] = 'vel_rms*Lz/chi'
        self.problem.substitutions['Ma_iso_rms'] = '(vel_rms/sqrt(T_full))'
        self.problem.substitutions['Ma_ad_rms'] = '(vel_rms/(sqrt(gamma*T_full)))'
        #self.problem.substitutions['lambda_microscale'] = 'sqrt(plane_avg(vel_rms)/plane_avg(enstrophy))'
        #self.problem.substitutions['Re_microscale'] = 'vel_rms*lambda_microscale/nu'
        #self.problem.substitutions['Pe_microscale'] = 'vel_rms*lambda_microscale/chi'
        
        self.problem.substitutions['h_flux_z'] = 'w*(h)'
        self.problem.substitutions['kappa_flux_z'] = '(kappa_flux_mean + kappa_flux_fluc)'
        self.problem.substitutions['KE_flux_z'] = 'w*(KE)'
        self.problem.substitutions['PE_flux_z'] = 'w*(PE)'
        self.problem.substitutions['viscous_flux_z'] = '- rho_full * nu * (u*σxz + v*σyz + w*σzz)'
        self.problem.substitutions['convective_flux_z'] = '(viscous_flux_z + KE_flux_z + PE_flux_z + h_flux_z)'
        
        self.problem.substitutions['evolved_avg_kappa'] = 'vol_avg(rho_full*chi)'
        self.problem.substitutions['kappa_adiabatic_flux_z_G75']  = '(rho0*chi*g/Cp)'
        self.problem.substitutions['kappa_adiabatic_flux_z_AB17'] = '(evolved_avg_kappa*g/Cp)'
        self.problem.substitutions['kappa_reference_flux_z_G75'] = '(-chi*rho0*(right(T1+T0)-left(T1+T0))/Lz)'
        self.problem.substitutions['Nusselt_norm_G75']   = '(kappa_reference_flux_z_G75 - kappa_adiabatic_flux_z_G75)'
        self.problem.substitutions['Nusselt_norm_AB17']   = 'vol_avg(kappa_flux_z - kappa_adiabatic_flux_z_AB17)'
        self.problem.substitutions['all_flux_minus_adiabatic_G75'] = '(convective_flux_z+kappa_flux_z-kappa_adiabatic_flux_z_G75)'
        self.problem.substitutions['all_flux_minus_adiabatic_AB17'] = '(convective_flux_z+kappa_flux_z-kappa_adiabatic_flux_z_AB17)'
        self.problem.substitutions['Nusselt_G75'] = '((all_flux_minus_adiabatic_G75)/(Nusselt_norm_G75))'
        self.problem.substitutions['Nusselt_AB17'] = '((all_flux_minus_adiabatic_AB17)/(Nusselt_norm_AB17))'
        
    def set_BC(self,
               fixed_flux=None, fixed_temperature=None, mixed_flux_temperature=None, mixed_temperature_flux=None,
               stress_free=None, no_slip=None):

        self.dirichlet_set = []

        self.set_thermal_BC(fixed_flux=fixed_flux, fixed_temperature=fixed_temperature,
                            mixed_flux_temperature=mixed_flux_temperature, mixed_temperature_flux=mixed_temperature_flux)
        
        self.set_velocity_BC(stress_free=stress_free, no_slip=no_slip)
        
        for key in self.dirichlet_set:
            self.problem.meta[key]['z']['dirichlet'] = True
            
    def set_thermal_BC(self, fixed_flux=None, fixed_temperature=None, mixed_flux_temperature=None, mixed_temperature_flux=None):
        # not(None) logic is going to be deprecated in future python releases.  What is the best way to use None as a function argument and in logic?  "if A is None" vs "if not(A)" and "if A".  Gabo will check.
        if not(fixed_flux) and not(fixed_temperature) and not(mixed_temperature_flux) and not(mixed_flux_temperature):
            mixed_flux_temperature = True

        # is this EVP aware check still needed?  What's going wrong with the EVP homogenization?  Why does it need to be done by hand?  Check if this is still actually broken, determine why.
        if 'EVP' in self.problem_type:
            l_flux_rhs_str = "0"
            r_flux_rhs_str = "0"
        else:
            l_flux_rhs_str = " left((exp(-ln_rho1)-1+ln_rho1)*T0_z)"
            r_flux_rhs_str = "right((exp(-ln_rho1)-1+ln_rho1)*T0_z)"
        # thermal boundary conditions
        if fixed_flux:
            logger.info("Thermal BC: fixed flux (full form)")
            self.problem.add_bc( "left(T1_z + ln_rho1*T0_z) = {:s}".format(l_flux_rhs_str))
            self.problem.add_bc("right(T1_z + ln_rho1*T0_z) = {:s}".format(r_flux_rhs_str))
            self.dirichlet_set.append('T1_z')
            self.dirichlet_set.append('ln_rho1')
        elif fixed_temperature:
            logger.info("Thermal BC: fixed temperature (T1)")
            self.problem.add_bc( "left(T1) = 0")
            self.problem.add_bc("right(T1) = 0")
            self.dirichlet_set.append('T1')
        elif mixed_flux_temperature:
            logger.info("Thermal BC: fixed flux/fixed temperature")
            self.problem.add_bc("left(T1_z + ln_rho1*T0_z) =  {:s}".format(l_flux_rhs_str))
            self.problem.add_bc("right(T1)  = 0")
            self.dirichlet_set.append('T1_z')
            self.dirichlet_set.append('T1')
            self.dirichlet_set.append('ln_rho1')
        elif mixed_temperature_flux:
            logger.info("Thermal BC: fixed temperature/fixed flux")
            logger.info("warning; these are not fully correct fixed flux conditions yet")
            self.problem.add_bc("left(T1)    = 0")
            self.problem.add_bc("right(T1_z + ln_rho1*T0_z) = {:s}".format(r_flux_rhs_str))
            self.dirichlet_set.append('T1_z')
            self.dirichlet_set.append('T1')
            self.dirichlet_set.append('ln_rho1')
        else:
            logger.error("Incorrect thermal boundary conditions specified")
            raise

    def set_velocity_BC(self, stress_free=None, no_slip=None):
        if not(stress_free) and not(no_slip):
            stress_free = True
            
        # horizontal velocity boundary conditions
        if stress_free:
            logger.info("Horizontal velocity BC: stress free")
            self.problem.add_bc( "left(u_z) = 0")
            self.problem.add_bc("right(u_z) = 0")
            self.dirichlet_set.append('u_z')
        elif no_slip:
            logger.info("Horizontal velocity BC: no slip")
            self.problem.add_bc( "left(u) = 0")
            self.problem.add_bc("right(u) = 0")
            self.dirichlet_set.append('u')
        else:
            logger.error("Incorrect horizontal velocity boundary conditions specified")
            raise

        # vertical velocity boundary conditions
        logger.info("Vertical velocity BC: impenetrable")
        self.problem.add_bc( "left(w) = 0")
        self.problem.add_bc("right(w) = 0")
        self.dirichlet_set.append('w')
        
    def set_IC(self, solver, A0=1e-6, **kwargs):
        # initial conditions
        T_IC = solver.state['T1']
        T_z_IC = solver.state['T1_z']
        ln_rho_IC = solver.state['ln_rho1']
            
        noise = self.global_noise(**kwargs)
        noise.set_scales(self.domain.dealias, keep_data=True)
        T_IC.set_scales(self.domain.dealias, keep_data=True)
        self.T0.set_scales(self.domain.dealias, keep_data=True)
        T_IC['g'] += A0*np.sin(np.pi*self.z_dealias/self.Lz)*noise['g']*self.T0['g']*self.epsilon
        T_IC.differentiate('z', out=T_z_IC)
        logger.info("Starting with T1 perturbations of amplitude A0 = {:g}".format(A0))

    def get_full_T(self, solver):
        T1 = solver.state['T1']
        T_scales = T1.meta[:]['scale']
        T1.set_scales(self.domain.dealias, keep_data=True)
        self.T0.set_scales(self.domain.dealias, keep_data=True)
        T = self._new_field()
        T.set_scales(self.domain.dealias, keep_data=False)
        T['g'] = self.T0['g'] + T1['g']
        T.set_scales(T_scales, keep_data=True)
        T1.set_scales(T_scales, keep_data=True)
        return T

    def get_full_rho(self, solver):
        ln_rho1 = solver.state['ln_rho1']
        rho_scales = ln_rho1.meta[:]['scale']
        self.rho0.set_scales(rho_scales, keep_data=True)
        rho = self._new_field()
        rho['g'] = self.rho0['g']*np.exp(ln_rho1['g'])
        rho.set_scales(rho_scales, keep_data=True)
        ln_rho1.set_scales(rho_scales, keep_data=True)
        return rho

    def check_system(self, solver, **kwargs):
        T = self.get_full_T(solver)
        rho = self.get_full_rho(solver)

        self.check_atmosphere(T=T, rho=rho, **kwargs)
        
    def initialize_output(self, solver, data_dir, coeffs_output=False,
                          max_writes=20, mode="overwrite", **kwargs):

        self.analysis_tasks = analysis_tasks = OrderedDict()

        analysis_profile = solver.evaluator.add_file_handler(data_dir+"profiles", max_writes=max_writes, parallel=False,
                                                             mode=mode, **kwargs)
        analysis_profile.add_task("plane_avg(T1)", name="T1")
        analysis_profile.add_task("plane_avg(T_full)", name="T_full")
        analysis_profile.add_task("plane_avg(Ma_iso_rms)", name="Ma_iso")
        analysis_profile.add_task("plane_avg(Ma_ad_rms)", name="Ma_ad")
        analysis_profile.add_task("plane_avg(ln_rho1)", name="ln_rho1")
        analysis_profile.add_task("plane_avg(rho_full)", name="rho_full")
        analysis_profile.add_task("plane_avg(KE)", name="KE")
        analysis_profile.add_task("plane_avg(PE)", name="PE")
        analysis_profile.add_task("plane_avg(IE)", name="IE")
        analysis_profile.add_task("plane_avg(PE_fluc)", name="PE_fluc")
        analysis_profile.add_task("plane_avg(IE_fluc)", name="IE_fluc")
        analysis_profile.add_task("plane_avg(KE + PE + IE)", name="TE")
        analysis_profile.add_task("plane_avg(KE + PE_fluc + IE_fluc)", name="TE_fluc")

        analysis_profile.add_task("plane_avg(KE_flux_z)", name="KE_flux_z")
        analysis_profile.add_task("plane_avg(PE_flux_z)", name="PE_flux_z")
        analysis_profile.add_task("plane_avg(w*(IE))", name="IE_flux_z")
        analysis_profile.add_task("plane_avg(w*(P))",  name="P_flux_z")
        analysis_profile.add_task("plane_avg(h_flux_z)",  name="enthalpy_flux_z")
        analysis_profile.add_task("plane_avg(viscous_flux_z)",  name="viscous_flux_z")
        analysis_profile.add_task("plane_avg(kappa_flux_z)", name="kappa_flux_z")
        analysis_profile.add_task("plane_avg(kappa_flux_z) - vol_avg(kappa_flux_z)", name="kappa_flux_fluc_z")
        analysis_profile.add_task("plane_avg(kappa_flux_z - kappa_adiabatic_flux_z_G75)", name="kappa_flux_z_minus_ad_G75")
        analysis_profile.add_task("plane_avg(kappa_flux_z - kappa_adiabatic_flux_z_AB17)", name="kappa_flux_z_minus_ad_AB17")
        analysis_profile.add_task("plane_avg(kappa_flux_z-kappa_adiabatic_flux_z_G75)/vol_avg(Nusselt_norm_G75)", name="norm_kappa_flux_z_G75")
        analysis_profile.add_task("plane_avg(kappa_flux_z-kappa_adiabatic_flux_z_AB17)/vol_avg(Nusselt_norm_AB17)", name="norm_kappa_flux_z_AB17")
        analysis_profile.add_task("plane_avg(Nusselt_G75)", name="Nusselt_G75")
        analysis_profile.add_task("plane_avg(Nusselt_AB17)", name="Nusselt_AB17")
        analysis_profile.add_task("plane_avg(u_rms)", name="u_rms")
        analysis_profile.add_task("plane_avg(w_rms)", name="w_rms")
        analysis_profile.add_task("plane_avg(vel_rms)", name="vel_rms")
        analysis_profile.add_task("plane_avg(Re_rms)", name="Re_rms")
        analysis_profile.add_task("plane_avg(Pe_rms)", name="Pe_rms")
        analysis_profile.add_task("plane_avg(enstrophy)", name="enstrophy")
        analysis_profile.add_task("plane_std(enstrophy)", name="enstrophy_std")        
        analysis_profile.add_task("plane_avg(Rayleigh_global)", name="Rayleigh_global")
        analysis_profile.add_task("plane_avg(Rayleigh_local)",  name="Rayleigh_local")
        analysis_profile.add_task("plane_avg(s_fluc)", name="s_fluc")
        analysis_profile.add_task("plane_std(s_fluc)", name="s_fluc_std")
        analysis_profile.add_task("plane_avg(s_mean)", name="s_mean")
        analysis_profile.add_task("plane_avg(s_fluc + s_mean)", name="s_tot")
        analysis_profile.add_task("plane_avg(dz(s_fluc))", name="grad_s_fluc")        
        analysis_profile.add_task("plane_avg(dz(s_mean))", name="grad_s_mean")        
        analysis_profile.add_task("plane_avg(dz(s_fluc + s_mean))", name="grad_s_tot")
        analysis_profile.add_task("plane_avg(g*dz(s_fluc)*Cp_inv)", name="brunt_squared_fluc")        
        analysis_profile.add_task("plane_avg(g*dz(s_mean)*Cp_inv)", name="brunt_squared_mean")        
        analysis_profile.add_task("plane_avg(g*dz(s_fluc + s_mean)*Cp_inv)", name="brunt_squared_tot")

        analysis_tasks['profile'] = analysis_profile

        analysis_scalar = solver.evaluator.add_file_handler(data_dir+"scalar", max_writes=max_writes, parallel=False,
                                                            mode=mode, **kwargs)
        analysis_scalar.add_task("vol_avg(KE)", name="KE")
        analysis_scalar.add_task("vol_avg(PE)", name="PE")
        analysis_scalar.add_task("vol_avg(IE)", name="IE")
        analysis_scalar.add_task("vol_avg(PE_fluc)", name="PE_fluc")
        analysis_scalar.add_task("vol_avg(IE_fluc)", name="IE_fluc")
        analysis_scalar.add_task("vol_avg(KE + PE + IE)", name="TE")
        analysis_scalar.add_task("vol_avg(KE + PE_fluc + IE_fluc)", name="TE_fluc")
        analysis_scalar.add_task("vol_avg(u_rms)", name="u_rms")
        analysis_scalar.add_task("vol_avg(w_rms)", name="w_rms")
        analysis_scalar.add_task("vol_avg(Re_rms)", name="Re_rms")
        analysis_scalar.add_task("vol_avg(Pe_rms)", name="Pe_rms")
        analysis_scalar.add_task("vol_avg(Ma_iso_rms)", name="Ma_iso")
        analysis_scalar.add_task("vol_avg(Ma_ad_rms)", name="Ma_ad")
        analysis_scalar.add_task("vol_avg(enstrophy)", name="enstrophy")
        analysis_scalar.add_task("vol_avg(Nusselt_G75)", name="Nusselt_G75")
        analysis_scalar.add_task("vol_avg(Nusselt_AB17)", name="Nusselt_AB17")
        analysis_scalar.add_task("vol_avg(Nusselt_norm_G75)", name="Nusselt_norm_G75")
        analysis_scalar.add_task("vol_avg(Nusselt_norm_AB17)", name="Nusselt_norm_AB17")
        analysis_scalar.add_task("log(left(plane_avg(rho_full))/right(plane_avg(rho_full)))", name="n_rho")
        analysis_scalar.add_task("(plane_avg(right(kappa_flux_z)) - plane_avg(left(kappa_flux_z)))", name="flux_equilibration")
        analysis_scalar.add_task("(plane_avg(right(kappa_flux_z)) - plane_avg(left(kappa_flux_z)))/plane_avg(left(kappa_flux_z))",name="flux_equilibration_pct")
            
        analysis_tasks['scalar'] = analysis_scalar

        if coeffs_output:
            analysis_coeff = solver.evaluator.add_file_handler(data_dir+"coeffs", max_writes=max_writes, parallel=False,
                                                               mode=mode, **kwargs)
            analysis_coeff.add_task("s_fluc", name="s", layout='c')
            analysis_coeff.add_task("s_fluc - plane_avg(s_fluc)", name="s'", layout='c')
            analysis_coeff.add_task("T1+T0", name="T", layout='c')
            analysis_coeff.add_task("T1+T0 - plane_avg(T1+T0)", name="T'", layout='c')
            analysis_coeff.add_task("ln_rho1+ln_rho0", name="ln_rho", layout='c')
            analysis_coeff.add_task("ln_rho1+ln_rho0 - plane_avg(ln_rho1+ln_rho0)", name="ln_rho'", layout='c')
            analysis_coeff.add_task("u", name="u", layout='c')
            analysis_coeff.add_task("w", name="w", layout='c')
            analysis_coeff.add_task("enstrophy", name="enstrophy", layout='c')
            analysis_coeff.add_task("ω_y", name="vorticity", layout='c')
            analysis_tasks['coeff'] = analysis_coeff
        
        return analysis_tasks

    def _atmosphere_splitter(self):
        nccs = OrderedDict([
               ('ln_rho0',               'log(rho0)'), 
               ( 'del_ln_rho0',          'del_ln_rho0'), 
               ( 'T0',                   'T0'), 
               ( 'T0_z',                 'T0_z')])
                
        if self.max_ncc_bandwidth is not None:
            splitter = NCC_Splitter(self, nccs)
            splitter.split_NCCs(num_coeffs=self.max_ncc_bandwidth)
        else:
            for nm, string in nccs.items():
                self.problem.substitutions['{:s}_L'.format(nm)] = '{:s}'.format(string)
                self.problem.substitutions['{:s}_R'.format(nm)] = '0'
 
    
class FC_equations_2d(FC_equations):
    def __init__(self, **kwargs):
        super(FC_equations_2d, self).__init__(**kwargs)
        self.equation_set = 'Fully Compressible (FC) Navier-Stokes'
        self.variables = ['u','u_z','w','w_z','T1', 'T1_z', 'ln_rho1']
        
    def _set_subs(self):
        # 2-D specific subs
        self.problem.substitutions['dy(A)']       = '(0*A)'
        
        # analysis operators
        if self.dimensions == 1:
            self.problem.substitutions['plane_avg(A)'] = '(A)'
            self.problem.substitutions['vol_avg(A)']   = 'integ(A)/Lz'
        else:
            self.problem.substitutions['plane_avg(A)'] = 'integ(A, "x")/Lx'
            self.problem.substitutions['vol_avg(A)']   = 'integ(A)/Lx/Lz'

        self._set_operators()
        self._set_diffusion_subs()
        super(FC_equations_2d, self)._set_subs()
        
    def set_equations(self, Rayleigh, Prandtl,
                      kx = 0,
                      split_diffusivities=False):

        if self.dimensions == 1:
            self.problem.parameters['j'] = 1j
            self.problem.substitutions['dx(f)'] = "j*kx*(f)"
            self.problem.parameters['kx'] = kx
            
        self.problem.substitutions['v']           = '(0)'
        self.problem.substitutions['v_z']         = '(0)'


        self.split_diffusivities = split_diffusivities
        self._set_diffusivities(Rayleigh=Rayleigh, Prandtl=Prandtl,
                                split_diffusivities=split_diffusivities)
        
        self._set_parameters()
        self._set_subs()
        self._atmosphere_splitter()
        
        self.problem.add_equation("dz(u) - u_z = 0")
        self.problem.add_equation("dz(w) - w_z = 0")
        self.problem.add_equation("dz(T1) - T1_z = 0")
            
        logger.debug("Setting z-momentum equation")
        self.problem.add_equation(("(scale_momentum)*( dt(w) + T1_z     + T0_L*dz(ln_rho1) + T1*del_ln_rho0_L - L_visc_w) = "
                                   "(scale_momentum)*(-UdotGrad(w, w_z) - (T0_R + T1)*dz(ln_rho1) - T1*del_ln_rho0_R + R_visc_w)"))
        
        logger.debug("Setting x-momentum equation")
        self.problem.add_equation(("(scale_momentum)*( dt(u) + dx(T1)   + T0_L*dx(ln_rho1)                  - L_visc_u) = "
                                   "(scale_momentum)*(-UdotGrad(u, u_z) - (T0_R + T1)*dx(ln_rho1) + R_visc_u)"))

        logger.debug("Setting continuity equation")
        self.problem.add_equation(("(scale_continuity)*( dt(ln_rho1)   + w*del_ln_rho0_L + Div_u ) = "
                                   "(scale_continuity)*(-UdotGrad(ln_rho1, dz(ln_rho1)) - w*del_ln_rho0_R)"))

        logger.debug("Setting energy equation")
        self.problem.add_equation(("(scale_energy)*( dt(T1)   + w*T0_z_L  + (gamma-1)*T0_L*Div_u -  L_thermal) = "
                                   "(scale_energy)*(-UdotGrad(T1, T1_z) - w*T0_z_R - (gamma-1)*(T1 + T0_R)*Div_u + R_thermal + R_visc_heat + source_terms)")) 
                            

    def initialize_output(self, solver, data_dir, coeffs_output=False,
                          max_writes=20, mode="overwrite", **kwargs):

        analysis_tasks = super().initialize_output(solver, data_dir, coeffs_output=coeffs_output, max_writes=max_writes, mode=mode, **kwargs)
        
        analysis_slice = solver.evaluator.add_file_handler(data_dir+"slices", max_writes=max_writes, parallel=False,
                                                            mode=mode, **kwargs)
        analysis_slice.add_task("s_fluc", name="s")
        analysis_slice.add_task("s_fluc - plane_avg(s_fluc)", name="s'")
        analysis_slice.add_task("u", name="u")
        analysis_slice.add_task("w", name="w")
        analysis_slice.add_task("enstrophy", name="enstrophy")
        analysis_slice.add_task("ω_y", name="vorticity")
        analysis_tasks['slice'] = analysis_slice

        return analysis_tasks


class NCC_Splitter():
    
    def __init__(self, equations, ncc_dict):
        self.equations  =   equations
        self.namespace  =   self.equations.problem.namespace
        self.domain     =   self.equations.domain
        self.nccs       =   ncc_dict #Extremely important that this is an OrderedDict in parallel.
        self.eval_nccs  =   OrderedDict() 

    def _clear_problem_namespace(self):
        del self.equations.problem.namespace
        self.equations.problem.namespace

    def _evaluate_NCCs(self):
        from dedalus.core.future import FutureField
        for name, ncc in self.nccs.items():
            logger.debug('splitting NCC {}: {}'.format(name, ncc))
            self.eval_nccs[name] = self.equations._new_ncc()
            self.eval_nccs[name].set_scales(self.equations.domain.dealias)
            self.eval_nccs[name]['g'] = FutureField.parse(ncc, self.namespace, self.domain).evaluate()['g']

    def split_NCCs(self, num_coeffs=None):
        if num_coeffs is None:
            num_coeffs = self.equations.nz
        self._evaluate_NCCs()
        for nm, f in self.eval_nccs.items():
            f.set_scales(1, keep_data=True)
            rhs_field = self.equations._new_ncc()
            rhs_field['g'] = np.copy(f['g'])
            f.set_scales(num_coeffs/self.equations.nz, keep_data=True)
            f['c']
            f['g']
            f.set_scales(1, keep_data=True)
            rhs_field.set_scales(1, keep_data=True)
            rhs_field['g'] -= f['g']
            self.equations.problem.parameters['{:s}_L'.format(nm)] = f
            self.equations.problem.parameters['{:s}_R'.format(nm)] = rhs_field
        self._clear_problem_namespace()
    
 

class FC_equations_2d_kappa_mu(FC_equations_2d):

    def _set_diffusion_subs(self):
        # define nu and chi for outputs
        self.problem.substitutions['nu']  = 'μ/rho0*exp(-ln_rho1)'
        self.problem.substitutions['chi'] = 'κ/rho0/Cp*exp(-ln_rho1)'
        
        self.problem.substitutions['KapLapT(kap, Tmp, Tmp_z)'] = "(kap * Lap(Tmp, Tmp_z))"
        self.problem.substitutions['GradKapGradT(kap, Tmp, Tmp_z)']   = "(dx(kap)*dx(Tmp) + dy(kap)*dy(Tmp) + dz(kap)*Tmp_z)"
        self.problem.substitutions['κ1']   = '(κ1_T*T1 + κ1_rho*ln_rho1)'

        if self.problem_type == 'EVP':
            self.problem.substitutions['rhs_adjust'] = '0'
            self.problem.substitutions['exp_ln_rho1'] = '1'
        else:
            self.problem.substitutions['rhs_adjust'] = '(exp(-ln_rho1)-1)'
            self.problem.substitutions['exp_ln_rho1'] = 'exp(ln_rho1)'

        #Language:
        # D = "divided by"
        # δ = "grad"
        nccs = OrderedDict([
                ('κ0_D_rho0',              'κ0/rho0'), 
                ('μ0_D_rho0',              'μ/rho0'), 
                ('δμ0_D_rho0',             'dz(μ)/rho0'), 
                ('κ1_T_δT0',               'κ1_T*T0_z'), 
                ('κ1_rho_δT0',             'κ1_rho*T0_z'),
                ('κ0',                     'κ0'),
                ('κ1_T',                   'κ1_T'),
                ('κ1_rho',                 'κ1_rho'),
                ('δκ0_D_rho0',             'dz(κ0)/rho0'), 
                ('δκ1T_δT0_D_rho0',        'dz(κ1_T)*T0_z/rho0'), 
                ('κ1T_δT0_D_rho0',         'κ1_T*T0_z/rho0'), 
                ('κ1T_δδT0_D_rho0',        'κ1_T*dz(T0_z)/rho0'), 
                ('δκ1rho_δT0_D_rho0',      'dz(κ1_rho)*T0_z/rho0'),
                ('κ1rho_δT0_D_rho0',       'κ1_rho*T0_z/rho0'),
                ('κ1rho_δδT0_D_rho0',       'κ1_rho*dz(T0_z)/rho0') ])
        if self.max_ncc_bandwidth is not None:
            splitter = NCC_Splitter(self, nccs)
            splitter.split_NCCs(num_coeffs=self.max_ncc_bandwidth)
        else:
            for nm, string in nccs.items():
                self.problem.substitutions['{:s}_L'.format(nm)] = '{:s}'.format(string)
                self.problem.substitutions['{:s}_R'.format(nm)] = '0'
            

        
        self.problem.substitutions['L_visc_u_t(mu_D_rho0, δmu_D_rho0)'] = "( mu_D_rho0*(Lap(u, u_z) + 1/3*Div(dx(u), dx(v), dx(w_z))) + δmu_D_rho0*(σxz))"
        self.problem.substitutions['L_visc_v_t(mu_D_rho0, δmu_D_rho0)'] = "( mu_D_rho0*(Lap(v, v_z) + 1/3*Div(dy(u), dy(v), dy(w_z))) + δmu_D_rho0*(σyz))"
        self.problem.substitutions['L_visc_w_t(mu_D_rho0, δmu_D_rho0)'] = "( mu_D_rho0*(Lap(w, w_z) + 1/3*Div(  u_z, dz(v), dz(w_z))) + δmu_D_rho0*(σzz))"                


        
        self.problem.substitutions['L_visc_u'] = "L_visc_u_t(μ0_D_rho0_L, δμ0_D_rho0_L)"
        self.problem.substitutions['L_visc_v'] = "L_visc_v_t(μ0_D_rho0_L, δμ0_D_rho0_L)"
        self.problem.substitutions['L_visc_w'] = "L_visc_w_t(μ0_D_rho0_L, δμ0_D_rho0_L)"

        self.problem.substitutions['R_visc_u'] = "((L_visc_u)*rhs_adjust + L_visc_u_t(μ0_D_rho0_R, δμ0_D_rho0_R)/exp_ln_rho1)"
        self.problem.substitutions['R_visc_v'] = "((L_visc_v)*rhs_adjust + L_visc_v_t(μ0_D_rho0_R, δμ0_D_rho0_R)/exp_ln_rho1)"
        self.problem.substitutions['R_visc_w'] = "((L_visc_w)*rhs_adjust + L_visc_w_t(μ0_D_rho0_R, δμ0_D_rho0_R)/exp_ln_rho1)"

        self.problem.substitutions['κ1_δδT0_D_rho_L'] = '(κ1rho_δδT0_D_rho0_L*ln_rho1 + κ1T_δδT0_D_rho0_L*T1)'
        self.problem.substitutions['κ1_δδT0_D_rho_R'] = '(κ1rho_δδT0_D_rho0_R*ln_rho1 + κ1T_δδT0_D_rho0_R*T1)'
        self.problem.substitutions['κ1_δδT0_D_rho'] =   '(κ1_δδT0_D_rho_L + κ1_δδT0_D_rho_R)'

        self.problem.substitutions['δκ0_δT1_D_rho_L'] = '(δκ0_D_rho0_L*T1_z)'
        self.problem.substitutions['δκ0_δT1_D_rho_R'] = '(δκ0_D_rho0_R*T1_z)'
        self.problem.substitutions['δκ0_δT1_D_rho'] =   '(δκ0_δT1_D_rho_L + δκ0_δT1_D_rho_R)'

        self.problem.substitutions['δκ1_δT0_D_rho_L'] = ('(δκ1T_δT0_D_rho0_L*T1 + δκ1rho_δT0_D_rho0_L*ln_rho1'
                                                                 '+κ1rho_δT0_D_rho0_L*dz(ln_rho1) + κ1T_δT0_D_rho0_L*T1_z)')
        self.problem.substitutions['δκ1_δT0_D_rho_R'] = ('(δκ1T_δT0_D_rho0_R*T1 + δκ1rho_δT0_D_rho0_R*ln_rho1'
                                                                 '+κ1rho_δT0_D_rho0_R*dz(ln_rho1) + κ1T_δT0_D_rho0_R*T1_z)')
        self.problem.substitutions['δκ1_δT0_D_rho'] = '(δκ1_δT0_D_rho_L + δκ1_δT0_D_rho_R)'

        self.problem.substitutions['T_L(κ0_D_rho, κ1_δδT0_D_rho, δκ0_δT1_D_rho, δκ1_δT0_D_rho)'] = \
                                                    ('(Cv_inv)*(KapLapT(κ0_D_rho, T1, T1_z) '
                                                     ' + κ1_δδT0_D_rho '
                                                     ' + δκ0_δT1_D_rho '
                                                     ' + δκ1_δT0_D_rho )')
        self.problem.substitutions['L_thermal']   = 'T_L(κ0_D_rho0_L, κ1_δδT0_D_rho_L, δκ0_δT1_D_rho_L, δκ1_δT0_D_rho_L)'
        self.problem.substitutions['L_thermal_R'] = 'T_L(κ0_D_rho0_R, κ1_δδT0_D_rho_R, δκ0_δT1_D_rho_R, δκ1_δT0_D_rho_R)'
#        self.problem.substitutions['T_L(κ0, κ1)'] = \
#                                                    ('(Cv_inv/rho0)*(KapLapT(κ0, T1, T1_z) '
#                                                     ' + KapLapT(κ1, T0, T0_z) '
#                                                     ' + GradKapGradT(κ0, T1, T1_z) '
#                                                     ' + GradKapGradT(κ1, T0, T0_z) )')
#        self.problem.substitutions['L_thermal'] = 'T_L(κ0, κ1)'
#        self.problem.substitutions['L_thermal_R'] = '0'
        self.problem.substitutions['R_thermal'] = ('( L_thermal_R + (L_thermal + L_thermal_R)*rhs_adjust'
                                                   '+ (Cv_inv/(rho0*exp_ln_rho1))*(KapLapT(κ_NL, (T0+T1), (T0_z+T1_z))'
                                                   '+ GradKapGradT(κ_NL, (T0+T1), (T0_z+T1_z))'
                                                   '+ κ0*dz(T0_z) + dz(κ0)*T0_z'
                                                   '+ KapLapT(κ1, T1, T1_z) + GradKapGradT(κ1, T1, T1_z)))' )
        self.problem.substitutions['source_terms'] = '0'
        self.problem.substitutions['R_visc_heat']  = " μ/rho_full*Cv_inv*(dx(u)*σxx + dy(v)*σyy + w_z*σzz + σxy**2 + σxz**2 + σyz**2)"

        self.problem.substitutions['kappa_flux_mean'] = '-κ0*dz(T0)'
        self.problem.substitutions['kappa_flux_fluc'] = '(-(κ0*dz(T1) + κ1*dz(T_full) + κ_NL*dz(T_full)))'
            

    def _set_diffusivities(self, *args, **kwargs):
        super(FC_equations_2d_kappa_mu, self)._set_diffusivities(*args, **kwargs)
        self.kappa1_T = self._new_ncc()
        self.kappa1_rho = self._new_ncc()
        self.problem.parameters['κ0'] = self.kappa

        self.mu = self._new_ncc()
        self.mu['g'] = self.nu['g']*self.rho0['g']
        self.problem.parameters['μ'] = self.mu

        self.problem.parameters['κ1_T'] = self.kappa1_T
        self.problem.parameters['κ1_rho'] = self.kappa1_rho
        self.problem.substitutions['κ'] = 'κ0'
        self.problem.substitutions['κ_NL'] = '(κ - κ0 - κ1_T*T1 - κ1_rho*ln_rho1)'
                    
    def set_thermal_BC(self, fixed_flux=None, fixed_temperature=None, mixed_flux_temperature=None, mixed_temperature_flux=None):
        if not(fixed_flux) and not(fixed_temperature) and not(mixed_temperature_flux) and not(mixed_flux_temperature):
            mixed_flux_temperature = True

        # thermal boundary conditions
        if fixed_flux:
            logger.info("Thermal BC: fixed flux (full form)")
            self.problem.add_bc( "left( κ0_L*T1_z + κ1_T_δT0_L*T1 + κ1_rho_δT0_L*ln_rho1 )  = -left(κ0_R*T1_z + κ1_T_δT0_R*T1 + κ1_rho_δT0_R*ln_rho1 + κ1*T1_z + κ_NL*(T0_z + T1_z))")
            self.problem.add_bc( "right( κ0_L*T1_z + κ1_T_δT0_L*T1 + κ1_rho_δT0_L*ln_rho1 )  = -right(κ0_R*T1_z + κ1_T_δT0_R*T1 + κ1_rho_δT0_R*ln_rho1 + κ1*T1_z + κ_NL*(T0_z + T1_z))")
            self.dirichlet_set.append('T1_z')
        elif fixed_temperature:
            logger.info("Thermal BC: fixed temperature (T1)")
            self.problem.add_bc( "left(T1) = 0")
            self.problem.add_bc("right(T1) = 0")
            self.dirichlet_set.append('T1')
        elif mixed_flux_temperature:
            logger.info("Thermal BC: fixed flux/fixed temperature")
            self.problem.add_bc( "left( κ0_L*T1_z + κ1_T_δT0_L*T1 + κ1_rho_δT0_L*ln_rho1 )  = -left(κ0_R*T1_z + κ1_T_δT0_R*T1 + κ1_rho_δT0_R*ln_rho1 + κ1*T1_z + κ_NL*(T0_z + T1_z))")
            self.problem.add_bc("right(T1)  = 0")
            self.dirichlet_set.append('T1_z')
            self.dirichlet_set.append('T1')
        elif mixed_temperature_flux:
            logger.info("Thermal BC: fixed temperature/fixed flux")
            self.problem.add_bc("left(T1)    = 0")
            self.problem.add_bc( "right( κ0_L*T1_z + κ1_T_δT0_L*T1 + κ1_rho_δT0_L*ln_rho1 )  = -right(κ0_R*T1_z + κ1_T_δT0_R*T1 + κ1_rho_δT0_R*ln_rho1 + κ1*T1_z + κ_NL*(T0_z + T1_z))")
            self.dirichlet_set.append('T1_z')
            self.dirichlet_set.append('T1')
        else:
            logger.error("Incorrect thermal boundary conditions specified")
            raise

class FC_equations_2d_kramers(FC_equations_2d_kappa_mu):
    """
    A 2D formulation of the fully compressible equations with a nonlinear kramers opacity.
    
    In such a formulation, the radiative conductivity is
    κ(ρ,T) = κ_0 · ρ^(-(1+a)) · T^(3-b)
    where the constants "a" and "b" are determined by the physics of the problem being studied.
    For radiative conductivity by free-free interactions, such as that deep in the solar CZ,
    we have a = 1 and b = -7/2 = -3.5. Thus, in a highly stratified domain, this conductivity
    can drop by orders of magnitude with height.

    Currently, this is implemented such that nu is constant with height and time, 
    and kappa is fully nonlinear.
    """

    def _set_diffusivities(self, *args, **kwargs):
        """
        This function assumes that the super() call properly sets the variables
        chi_top and nu_top, the values of the thermal and viscous diffusivities
        at the top of the domain.

        Note that kappa = rho * Cp * chi.
        """
        self.problem.substitutions['κ'] = 'κ0*((T0+T1)/T0)**(3-kram_b)*(exp(ln_rho1))**(-1-kram_a)'
        self.problem.substitutions['κ_NL'] = '(κ - κ0 - κ1_T*T1 - κ1_rho*ln_rho1)'
        self.problem.substitutions['chi'] = 'κ/rho0/exp(ln_rho1)/Cp'
        self.kappa = self._new_ncc()

        #TODO: put these back to just _set_diffusivities, and when initializing the atmosphere store the atmosphere/equations classes.
        kappa_0, T_ref, rho_ref, Prandtl = super(FC_equations_2d_kramers, self)._set_diffusivity_constants(*args, **kwargs)

        self.T0.set_scales(1, keep_data=True)
        self.rho0.set_scales(1, keep_data=True)
        self.kappa['g'] = kappa_0 *   (self.T0['g']/T_ref)**(3-self.kram_b)*\
                                      (self.rho0['g']/rho_ref)**(-(1+self.kram_a)) 
        self.problem.parameters['κ0'] = self.kappa


        [f.set_scales(1, keep_data=True) for f in (self.rho0, self.chi, self.kappa)]
        self.chi['g'] = self.kappa['g']/self.rho0['g']/self.Cp

        self.nu = self._new_ncc()
        self.chi.set_scales(1, keep_data=True)
        self.nu['g'] = self.chi['g']*Prandtl
        self.nu_top = nu_top = np.max(self.nu.interpolate(z=self.Lz)['g'])
        logger.info("chi top: {}; nu top: {}".format(np.max(self.chi.interpolate(z=self.Lz)['g']), np.max(self.nu.interpolate(z=self.Lz)['g'])))
        logger.info("Pr: {}".format(self.nu['g']/self.chi['g']))
        self.kappa.set_scales(1, keep_data=True)
        self.T0_z.set_scales(1, keep_data=True)

        self.problem.parameters['kram_a']   = self.kram_a
        self.problem.parameters['kram_b']   = self.kram_b

        self.thermal_time = self.Lz**2/np.mean(self.chi.interpolate(z=self.Lz/2)['g'])
        self.top_thermal_time = 1/np.mean(self.chi.interpolate(z=self.Lz)['g'])

        self.viscous_time = self.Lz**2/np.mean(self.nu.interpolate(z=self.Lz/2)['g'])
        self.top_viscous_time = 1/np.mean(self.nu.interpolate(z=self.Lz)['g'])


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

        self.kappa1_T = self._new_ncc()
        self.kappa1_rho = self._new_ncc()
        [f.set_scales(1, keep_data=True) for f in (self.T0, self.T0_z, self.rho0, self.del_ln_rho0)]
        self.kappa1_T['g']   =  (3 - self.kram_b) * self.kappa['g'] * self.T0_z['g'] / self.T0['g']
        self.kappa1_rho['g'] = -(1 + self.kram_a) * self.kappa['g'] * self.del_ln_rho0['g'] * self.rho0['g']
        self.problem.parameters['κ1_T'] = self.kappa1_T
        self.problem.parameters['κ1_rho'] = self.kappa1_rho

        self.mu = self._new_ncc()
        self.mu['g'] = self.nu['g']*self.rho0['g']
        self.problem.parameters['μ'] = self.mu

    def initialize_output(self, solver, data_dir, coeffs_output=False,
                          max_writes=20, mode="overwrite", **kwargs):

        analysis_tasks = super().initialize_output(solver, data_dir, coeffs_output=coeffs_output, max_writes=max_writes, mode=mode, **kwargs)
            
        analysis_tasks['profile'].add_task("plane_avg(κ)", name="kappa")
        analysis_tasks['profile'].add_task("plane_std(κ)", name="kappa_std")
        analysis_tasks['profile'].add_task("plane_avg(κ1_T)", name="kappa1_T")
        analysis_tasks['profile'].add_task("plane_avg(κ1_rho)", name="kappa1_rho")

        return analysis_tasks


   
class FC_equations_3d(FC_equations):
    def __init__(self, **kwargs):
        super(FC_equations_3d, self).__init__(**kwargs)
        self.equation_set = 'Fully Compressible (FC) Navier-Stokes in 3-D'
        self.variables = ['u','u_z','v','v_z','w','w_z','T1', 'T1_z', 'ln_rho1']
    
    def _set_subs(self, **kwargs):                    
        # analysis operators
        if self.dimensions != 1:
            self.problem.substitutions['plane_avg(A)'] = 'integ(A, "x", "y")/Lx/Ly'
            self.problem.substitutions['vol_avg(A)']   = 'integ(A)/Lx/Ly/Lz'
        else:
            self.problem.substitutions['plane_avg(A)'] = 'A'
            self.problem.substitutions['vol_avg(A)']   = 'integ(A)/Lz'
            
        self._set_operators()
        self._set_diffusion_subs()
        super(FC_equations_3d, self)._set_subs(**kwargs)
                        
    def set_equations(self, Rayleigh, Prandtl, Taylor=None, theta=0,
                      kx = 0, ky = 0,
                      split_diffusivities=False):
        
        if self.dimensions == 1:
            self.problem.parameters['j'] = 1j
            self.problem.substitutions['dx(f)'] = "j*kx*(f)"
            self.problem.parameters['kx'] = kx
            self.problem.substitutions['dy(f)'] = "j*ky*(f)"
            self.problem.parameters['ky'] = ky

        self.split_diffusivities = split_diffusivities
        self._set_diffusivities(Rayleigh=Rayleigh, Prandtl=Prandtl)
        self._set_parameters()
        self._set_subs()
    
        if Taylor:
            self.rotating = True
            self.problem.parameters['θ'] = theta
            self.problem.parameters['Ω'] = omega = np.sqrt(Taylor*self.nu_top**2/(4*self.Lz**4))
            logger.info("Rotating f-plane with Ω = {} and θ = {} (Ta = {})".format(omega, theta, Taylor))
            self.problem.substitutions['Ωx'] = '0'
            self.problem.substitutions['Ωy'] = 'Ω*sin(θ)'
            self.problem.substitutions['Ωz'] = 'Ω*cos(θ)'
            self.problem.substitutions['Coriolis_x'] = '(2*Ωy*w - 2*Ωz*v)'
            self.problem.substitutions['Coriolis_y'] = '(2*Ωz*u - 2*Ωx*w)'
            self.problem.substitutions['Coriolis_z'] = '(2*Ωx*v - 2*Ωy*u)'
            self.problem.substitutions['Rossby'] = '(sqrt(enstrophy)/(2*Ω))'
        else:
            self.rotating = False
            self.problem.substitutions['Coriolis_x'] = '0'
            self.problem.substitutions['Coriolis_y'] = '0'
            self.problem.substitutions['Coriolis_z'] = '0'
       
        self.problem.add_equation("dz(u) - u_z = 0")
        self.problem.add_equation("dz(v) - v_z = 0")
        self.problem.add_equation("dz(w) - w_z = 0")
        self.problem.add_equation("dz(T1) - T1_z = 0")

        logger.debug("Setting continuity equation")
        self.problem.add_equation(("(scale_continuity)*( dt(ln_rho1)   + w*del_ln_rho0 + Div_u ) = "
                                   "(scale_continuity)*(-UdotGrad(ln_rho1, dz(ln_rho1)))"))

        logger.debug("Setting z-momentum equation")
        self.problem.add_equation(("(scale_momentum)*( dt(w) + Coriolis_z + T1_z   + T0*dz(ln_rho1) + T1*del_ln_rho0 - L_visc_w) = "
                                   "(scale_momentum)*(-T1*dz(ln_rho1) - UdotGrad(w, w_z) + R_visc_w)"))
        
        logger.debug("Setting x-momentum equation")
        self.problem.add_equation(("(scale_momentum)*( dt(u) + Coriolis_x + dx(T1) + T0*dx(ln_rho1)                  - L_visc_u) = "
                                   "(scale_momentum)*(-T1*dx(ln_rho1) - UdotGrad(u, u_z) + R_visc_u)"))

        logger.debug("Setting y-momentum equation")
        self.problem.add_equation(("(scale_momentum)*( dt(v) + Coriolis_y + dy(T1) + T0*dy(ln_rho1)                  - L_visc_v) = "
                                   "(scale_momentum)*(-T1*dy(ln_rho1) - UdotGrad(v, v_z) + R_visc_v)"))

        logger.debug("Setting energy equation")
        self.problem.add_equation(("(scale_energy)*( dt(T1)   + w*T0_z + (gamma-1)*T0*Div_u -  L_thermal) = "
                                   "(scale_energy)*(-UdotGrad(T1, T1_z)    - (gamma-1)*T1*Div_u + R_thermal + R_visc_heat + source_terms)"))
        

    def set_BC(self, **kwargs):        
        super(FC_equations_3d, self).set_BC(**kwargs)
        # stress free boundary conditions.
        self.problem.add_bc("left(v_z) = 0")
        self.problem.add_bc("right(v_z) = 0")
        self.dirichlet_set.append('v_z')
        for key in self.dirichlet_set:
            self.problem.meta[key]['z']['dirichlet'] = True

        
    def initialize_output(self, solver, data_dir, coeffs_output=False, volumes_output=False,
                          max_writes=20, mode="overwrite", **kwargs):

        analysis_tasks = super().initialize_output(solver, data_dir, coeffs_output=coeffs_output, max_writes=max_writes, mode=mode, **kwargs)
        
        analysis_slice = solver.evaluator.add_file_handler(data_dir+"slices", max_writes=max_writes, parallel=False,
                                                           mode=mode, **kwargs)
        analysis_slice.add_task("interp(s_fluc,                     y={})".format(self.Ly/2), name="s")
        analysis_slice.add_task("interp(s_fluc - plane_avg(s_fluc), y={})".format(self.Ly/2), name="s'")
        analysis_slice.add_task("interp(enstrophy,                  y={})".format(self.Ly/2), name="enstrophy")
        analysis_slice.add_task("interp(ω_y,                        y={})".format(self.Ly/2), name="vorticity")
        analysis_slice.add_task("interp(s_fluc,                     z={})".format(0.95*self.Lz), name="s near top")
        analysis_slice.add_task("interp(s_fluc - plane_avg(s_fluc), z={})".format(0.95*self.Lz), name="s' near top")
        analysis_slice.add_task("interp(enstrophy,                  z={})".format(0.95*self.Lz), name="enstrophy near top")
        analysis_slice.add_task("interp(ω_z,                        z={})".format(0.95*self.Lz), name="vorticity_z near top")
        analysis_slice.add_task("interp(s_fluc,                     z={})".format(0.5*self.Lz),  name="s midplane")
        analysis_slice.add_task("interp(s_fluc - plane_avg(s_fluc), z={})".format(0.5*self.Lz),  name="s' midplane")
        analysis_slice.add_task("interp(enstrophy,                  z={})".format(0.5*self.Lz),  name="enstrophy midplane")
        analysis_slice.add_task("interp(ω_z,                        z={})".format(0.5*self.Lz),  name="vorticity_z midplane")
        analysis_tasks['slice'] = analysis_slice

        if volumes_output:
            analysis_volume = solver.evaluator.add_file_handler(data_dir+"volumes", max_writes=max_writes, parallel=False, 
                                                                mode=mode, **kwargs)
            analysis_volume.add_task("enstrophy", name="enstrophy")
            analysis_volume.add_task("s_fluc+s_mean", name="s_tot")
            analysis_tasks['volume'] = analysis_volume

        if self.rotating:
            analysis_scalar = analysis_tasks['scalar']
            analysis_scalar.add_task("vol_avg(Rossby)", name="Rossby")

            analysis_profile = analysis_tasks['profile']
            analysis_profile.add_task("plane_avg(Rossby)", name="Rossby")
            
        return analysis_tasks
