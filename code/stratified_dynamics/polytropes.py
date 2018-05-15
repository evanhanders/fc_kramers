import numpy as np
import os
from mpi4py import MPI

import logging
logger = logging.getLogger(__name__.split('.')[-1])

try:
    from equations import *
    from atmospheres import KramerPolytrope
except:
    from sys import path
    path.insert(0, './stratified_dynamics')
    from stratified_dynamics.equations import *
    from stratified_dynamics.atmospheres import KramerPolytrope


class FC_polytrope_2d_kramers(FC_equations_2d_kramers, KramerPolytrope):
    def __init__(self, dimensions=2, *args, fully_nonlinear=False, **kwargs):
        super(FC_polytrope_2d_kramers, self).__init__(dimensions=dimensions) 
        KramerPolytrope.__init__(self, *args, **kwargs)
        self.fully_nonlinear = fully_nonlinear
        logger.info("solving {} in a {} atmosphere".format(self.equation_set, self.atmosphere_name))

    def initialize_output(self, solver, data_dir, *args, **kwargs):
        super(FC_polytrope_2d_kramers, self).initialize_output(solver, data_dir, *args, **kwargs)
        self.save_atmosphere_file(data_dir)
        return self.analysis_tasks



                     
#class FC_polytrope_3d(FC_equations_3d, Polytrope):
#    def __init__(self, dimensions=3, *args, **kwargs):
#        super(FC_polytrope_3d, self).__init__(dimensions=dimensions) 
#        Polytrope.__init__(self, dimensions=dimensions, *args, **kwargs)
#        logger.info("solving {} in a {} atmosphere".format(self.equation_set, self.atmosphere_name))
#
#    def set_equations(self, *args, **kwargs):
#        super(FC_polytrope_3d, self).set_equations(*args, **kwargs)
#        self.test_hydrostatic_balance(T=self.T0, rho=self.rho0)
#        
#    def initialize_output(self, solver, data_dir, *args, **kwargs):
#        super(FC_polytrope_3d, self).initialize_output(solver, data_dir, *args, **kwargs)
#        self.save_atmosphere_file(data_dir)
#        return self.analysis_tasks
#
#       
