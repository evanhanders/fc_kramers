import numpy as np
import h5py
import matplotlib.pyplot as plt
from stratified_dynamics import polytropes, bvps_equilibration

bc_dict = {
        'stress_free'             : True,
        'no_slip'                 : False,
        'fixed_flux'              : False,
        'mixed_flux_temperature'  : True,
        'mixed_temperature_flux'  : False,
        'fixed_temperature'       : False
          }

def read_atmosphere(read_atmo_file, T, ln_rho, nz):
    atmo = h5py.File(read_atmo_file, 'r') 
    T1_IC = atmo['tasks']['T1'].value[0,:]
    ln_rho1_IC = atmo['tasks']['ln_rho1'].value[0,:]

    T1.set_scales(len(T1_IC)/nz, keep_data=True)
    ln_rho1.set_scales(len(T1_IC)/nz, keep_data=True)
    T1['g']       += T1_IC
    ln_rho1['g']  += ln_rho1_IC
    T1.differentiate('z', out=T1_z)
    atmo.close()



n_rho_cz=3
#atmo_file='initials/ICs_b-1e-07.h5'
atmo_file=None

increase_factor = -0.5
decrease_factor = increase_factor*2

nz = 1024
true_kram_b = -1
kram_b = true_kram_b
atmosphere = polytropes.FC_polytrope_2d_kramers(nz=nz, kram_b=kram_b, dimensions=1, n_rho_cz=n_rho_cz, grid_dtype=np.float64)
if atmo_file is not None:
    ln_rho0 = atmosphere._new_ncc()
    read_atmosphere(atmo_file, atmosphere.T0, ln_rho0, atmosphere.nz)
    ln_rho0.set_scales(1, keep_data=True)
    atmosphere.rho0.set_scales(1, keep_data=True)
    ln_rho0['g'] += np.log(atmosphere.rho0['g'])
    atmosphere.rho0['g'] = np.exp(ln_rho0['g'])
atmosphere.T0.differentiate('z', out=atmosphere.T0_z)
atmosphere.T0_z.differentiate('z', out=atmosphere.T0_zz)
atmosphere.rho0.differentiate('z', out=atmosphere.del_ln_rho0)
atmosphere.del_ln_rho0['g'] /= atmosphere.rho0['g']



tol=1e-7
ncc_cutoff=1e-8#np.abs(true_kram_b)*1e-6

flux_factor = 1
while(True):
    print('EQUILIBRATING AT KRAM B = {}'.format(kram_b))

    equilibration = bvps_equilibration.FC_kramers_equilibrium_solver(atmosphere.nz, atmosphere.Lz, grid_dtype=atmosphere.rho0['g'].dtype)
    
    atmosphere.T0.set_scales(1, keep_data=True)
    atmosphere.rho0.set_scales(1, keep_data=True)
    T, rho = atmosphere.T0['g'], atmosphere.rho0['g']
    
    equil_solver = equilibration.run_BVP(bc_dict, atmosphere.kram_a, atmosphere.kram_b, T, rho, g=atmosphere.g, Cp=atmosphere.Cp, gamma=atmosphere.gamma, tolerance=tol)
    ln_T1e, ln_rho1e = equil_solver.state['ln_T1'], equil_solver.state['ln_rho1']
    ln_T1e.set_scales(1, keep_data=True)
    ln_rho1e.set_scales(1, keep_data=True)

    this_lnt, this_lnrho = atmosphere._new_ncc(), atmosphere._new_ncc()
    atmosphere._set_field(this_lnt, ln_T1e['g'])
    atmosphere._set_field(this_lnrho, ln_rho1e['g'])

    atmosphere.T0['g'] *= np.exp(this_lnt['g'])#T1e['g']
    atmosphere.T0.differentiate('z', out=atmosphere.T0_z)
    atmosphere.T0_z.differentiate('z', out=atmosphere.T0_zz)
    atmosphere.rho0['g'] *= np.exp(this_lnrho['g'])#ln_rho1e['g'])
    atmosphere.rho0.differentiate('z', out=atmosphere.del_ln_rho0)
    atmosphere.del_ln_rho0['g'] /= atmosphere.rho0['g']

    if np.isnan(np.max(atmosphere.T0['g'])):
        kram_b -= decrease_factor
        print('DECREASING KRAM B BY FACTOR OF {} TO {}'.format(decrease_factor, kram_b))
        atmosphere = polytropes.FC_polytrope_2d_kramers(bc_dict, nz=nz, kram_b=kram_b, no_equil=True, dimensions=1, n_rho_cz=n_rho_cz, read_atmo_file=atmo_file)
        flux_factor = 1
    elif kram_b > true_kram_b:
        kram_b += increase_factor
        print('INCREASING KRAM B BY FACTOR OF root(2) TO {}'.format(kram_b))
        new_atmosphere = polytropes.FC_polytrope_2d_kramers(bc_dict, nz=nz, kram_b=kram_b, no_equil=True, dimensions=1, n_rho_cz=n_rho_cz)
        atmosphere.T0.set_scales(1, keep_data=True)
        new_atmosphere.T0['g']  = atmosphere.T0['g']
        new_atmosphere.T0.differentiate('z', out=new_atmosphere.T0_z)
        new_atmosphere.T0_z.differentiate('z', out=new_atmosphere.T0_zz)
        atmosphere.rho0.set_scales(1, keep_data=True)
        new_atmosphere.rho0.set_scales(1, keep_data=True)
        new_atmosphere.rho0['g'] = atmosphere.rho0['g']
        new_atmosphere.rho0.differentiate('z', out=new_atmosphere.del_ln_rho0)
        new_atmosphere.del_ln_rho0['g'] /= new_atmosphere.rho0['g']
        atmosphere = new_atmosphere
        flux_factor = 1/np.max(new_atmosphere.T0.interpolate(z=0)['g'])**(kram_b*(1 - 1/increase_factor))
        fig = plt.figure()
        ax = fig.add_subplot(2,2,1)
        atmosphere.T0.set_scales(1, keep_data=True)
        plt.plot(atmosphere.z, atmosphere.T0['g'])
        plt.ylabel('T')
        ax = fig.add_subplot(2,2,2)
        atmosphere.rho0.set_scales(1, keep_data=True)
        plt.plot(atmosphere.z, atmosphere.rho0['g'])
        plt.ylabel('rho')
        ax = fig.add_subplot(2,2,3)
        atmosphere.T0.set_scales(1, keep_data=True)
        plt.plot(np.arange(len(atmosphere.z)), np.abs(atmosphere.T0['g']))
        plt.yscale('log')
        plt.ylabel('T coeff')
        ax = fig.add_subplot(2,2,4)
        atmosphere.rho0.set_scales(1, keep_data=True)
        plt.plot(np.arange(len(atmosphere.z)), np.abs(atmosphere.rho0['g']))
        plt.yscale('log')
        plt.ylabel('rho coeff')
        plt.savefig('T_rho_b{:.4g}.png'.format(kram_b/increase_factor), dpi=300)

        print(atmosphere.T0['g'], atmosphere.rho0['c'])

    else:
        fig = plt.figure()
        ax = fig.add_subplot(2,2,1)
        atmosphere.T0.set_scales(1, keep_data=True)
        plt.plot(atmosphere.z, atmosphere.T0['g'])
        plt.ylabel('T')
        ax = fig.add_subplot(2,2,2)
        atmosphere.rho0.set_scales(1, keep_data=True)
        plt.plot(atmosphere.z, atmosphere.rho0['g'])
        plt.ylabel('rho')
        ax = fig.add_subplot(2,2,3)
        atmosphere.T0.set_scales(1, keep_data=True)
        plt.plot(np.arange(len(atmosphere.z)), np.abs(atmosphere.T0['g']))
        plt.yscale('log')
        plt.ylabel('T coeff')
        ax = fig.add_subplot(2,2,4)
        atmosphere.rho0.set_scales(1, keep_data=True)
        plt.plot(np.arange(len(atmosphere.z)), np.abs(atmosphere.rho0['g']))
        plt.yscale('log')
        plt.ylabel('rho coeff')
        plt.savefig('T_rho_b{:.4g}.png'.format(kram_b), dpi=300)
        break

f = h5py.File('initials/ICs_b{:.4g}.h5'.format(kram_b), 'w')
grp = f.create_group('tasks')
grp.create_dataset(name='T1', shape=(1, nz), dtype=np.float64)
atmosphere.T0.set_scales(1, keep_data=True)
atmosphere.T0['g'] -= (atmosphere.Lz + 1 - atmosphere.z)
atmosphere.T0.set_scales(1, keep_data=True)
f['tasks']['T1'][0,:] = atmosphere.T0['g']
atmosphere.T0_z['g'] -= -1
grp.create_dataset(name='T1_z', shape=(1, nz), dtype=np.float64)
atmosphere.T0_z.set_scales(1, keep_data=True)
f['tasks']['T1_z'][0,:] = atmosphere.T0_z['g']
atmosphere.rho0.set_scales(1, keep_data=True)
atmosphere.rho0['g'] = np.log(atmosphere.rho0['g'])
atmosphere.rho0.set_scales(1, keep_data=True)
atmosphere.rho0['g'] -= atmosphere.poly_m*np.log(1 + atmosphere.Lz - atmosphere.z)
grp.create_dataset(name='ln_rho1', shape=(1, nz), dtype=np.float64)
atmosphere.rho0.set_scales(1, keep_data=True)
f['tasks']['ln_rho1'][0,:] = atmosphere.rho0['g']

f.close()
