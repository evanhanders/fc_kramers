import numpy as np
import h5py
import matplotlib.pyplot as plt
from stratified_dynamics import polytropes

bc_dict = {
        'stress_free'             : True,
        'no_slip'                 : False,
        'fixed_flux'              : False,
        'mixed_flux_temperature'  : True,
        'mixed_temperature_flux'  : False,
        'fixed_temperature'       : False
          }

n_rho_cz=3

increase_factor = -0.02
decrease_factor = increase_factor*2

nz = 1024
true_kram_b = -1
kram_b = -0.5
atmosphere = polytropes.FC_polytrope_2d_kramers(bc_dict, nz=nz, kram_b=kram_b, no_equil=True, dimensions=1, n_rho_cz=n_rho_cz)


tol=1e-7
ncc_cutoff=np.abs(true_kram_b)*1e-6

flux_factor = 1
while(True):
    print('EQUILIBRATING AT KRAM B = {}'.format(kram_b))
    atmosphere._equilibrate_atmosphere(bc_dict, ncc_cutoff=ncc_cutoff, tolerance=tol)
    if np.isnan(np.max(atmosphere.T0['g'])):
        kram_b -= decrease_factor
        print('DECREASING KRAM B BY FACTOR OF {} TO {}'.format(decrease_factor, kram_b))
        atmosphere = polytropes.FC_polytrope_2d_kramers(bc_dict, nz=nz, kram_b=kram_b, no_equil=True, dimensions=1, n_rho_cz=n_rho_cz)
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
        plt.plot(np.arange(len(atmosphere.z)), np.abs(atmosphere.T0['c']))
        plt.yscale('log')
        plt.ylabel('T coeff')
        ax = fig.add_subplot(2,2,4)
        atmosphere.rho0.set_scales(1, keep_data=True)
        plt.plot(np.arange(len(atmosphere.z)), np.abs(atmosphere.rho0['c']))
        plt.yscale('log')
        plt.ylabel('rho coeff')
        plt.savefig('T_rho_b{:.4g}.png'.format(kram_b/increase_factor), dpi=300)

        print(atmosphere.T0['c'], atmosphere.rho0['c'])

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
        plt.plot(np.arange(len(atmosphere.z)), np.abs(atmosphere.T0['c']))
        plt.yscale('log')
        plt.ylabel('T coeff')
        ax = fig.add_subplot(2,2,4)
        atmosphere.rho0.set_scales(1, keep_data=True)
        plt.plot(np.arange(len(atmosphere.z)), np.abs(atmosphere.rho0['c']))
        plt.yscale('log')
        plt.ylabel('rho coeff')
        plt.savefig('T_rho_b{:.4g}.png'.format(kram_b), dpi=300)
        break

f = h5py.File('initials/ICs_b{:.4g}.h5'.format(kram_b), 'w')
for nm, fd in [('T0', atmosphere.T0), ('rho0', atmosphere.rho0)]:
    fd.set_scales(1, keep_data=True)
    f[nm] = fd['g']
f.close()
