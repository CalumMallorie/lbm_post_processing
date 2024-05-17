"""
This package provides modules for processing and analyzing simulation data.

Modules
-------
read_input_parameters
    Contains the CrossSlotParameters class for managing simulation parameters.
process_trajectory
    Contains the Trajectory and ProcessTrajectory classes for handling 
    particle trajectory data.
process_particle
    Contains the ProcessParticle class for handling particle data.
helper_functions
    Contains various helper functions such as search_sim_directory and 
    vector_radial_coordinates.
"""

from .read_input_parameters import CrossSlotParameters
from .process_trajectory import Trajectory
from .process_trajectory import ProcessTrajectory
from .process_particle import ProcessParticle
from .helper_functions import search_sim_directory, vector_radial_coordinates