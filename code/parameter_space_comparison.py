"""
Script for plotting a parameter space study of Nu v Ra.

Usage:
    parameter_space_plots.py --calculate
    parameter_space_plots.py

Options:
    --calculate     If flagged, touch dedalus output files and do time averages.  If not, use post-processed data from before
"""


import matplotlib.style
import matplotlib
matplotlib.use('Agg')
matplotlib.style.use('classic')
matplotlib.rcParams.update({'font.size': 11})
import matplotlib.pyplot as plt
from matplotlib.colors import ColorConverter
import os
import numpy as np
import glob
from docopt import docopt
from mpi4py import MPI
comm_world = MPI.COMM_WORLD
from collections import OrderedDict
import h5py
import matplotlib.gridspec as gridspec


import dedalus.public as de
plt.rc('font',family='Times New Roman')


fields = ['Re_rms', 'Ma_ad']
info = OrderedDict()

bs = []
ras = []
for i, d in enumerate(glob.glob('./pleiades/poly/FC*/')):
    ra = d.split('_Ra')[-1].split('_')[0]
    b = d.split('_b')[-1].split('_')[0]
    if b not in bs:
        bs += [b]
    if ra not in ras:
        ras += [ra]

    info['{:s}_{:s}'.format(ra, b)] = OrderedDict()
    try:
        with h5py.File('{:s}/scalar_plots/scalar_values.h5'.format(d), 'r') as f:
            for k in fields:
                info['{:s}_{:s}'.format(ra, b)][k] = f[k].value
            info['{:s}_{:s}'.format(ra, b)]['sim_time'] = f['sim_time'].value

    except:
        print('cannot find file in {:s}'.format(d))

print(info.keys())
crit_ra = dict()
crit_ra['-1e-4'] = 1.98e5



plt.figure(figsize=(8, 4))
gs     = gridspec.GridSpec(*(1000,1000))
gs_info = (((0,0), 950, 450), ((0, 550), 950, 450))
for i,k in enumerate(fields):
    ax = plt.subplot(gs.new_subplotspec(*gs_info[i]))
#    bx = ax.twiny()
    for b in bs:
        x_vals = []
        vals = []
        for ra in ras:
            run_key = '{:s}_{:s}'.format(ra, b)
            x_vals += [float(ra)]
            vals += [np.mean(info[run_key][k][-1000:])]
        ax.plot(np.array(x_vals)/crit_ra[b], vals, label='b={:s}'.format(b), lw=0, marker='o')
    ax.set_ylabel(k)
    ax.set_xlabel('Ra/Ra_crit')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.grid(which='both')
    ax.legend(loc='best')

            
plt.savefig('parameter_space_comparison.png'.format(k), dpi=400, bbox_inches='tight')
