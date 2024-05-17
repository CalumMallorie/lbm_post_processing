import os
from .read_input_parameters import CrossSlotParameters
import numpy as np
from .process_trajectory import Trajectory
import pandas as pd

def expand_parameters(parameters):
    """
    For use when parameters contains a list, this function breaks up the 
    list into separate dictionaries, each containing one value from the
    list.
    """
    # Base case: if parameters is empty, return a list containing an empty dictionary
    if not parameters:
        return [{}]

    expanded_parameters = []
    for key, value in parameters.items():
        # Remove the key-value pair from the parameters
        remaining_params = parameters.copy()
        del remaining_params[key]

        # If the value is a list, create a new dictionary for each value in the list
        if isinstance(value, list):
            for val in value:
                for expanded_dict in expand_parameters(remaining_params):
                    expanded_dict[key] = val
                    expanded_parameters.append(expanded_dict)
        else:
            # If the value is not a list, just add it to the dictionaries from the recursive call
            for expanded_dict in expand_parameters(remaining_params):
                expanded_dict[key] = value
                expanded_parameters.append(expanded_dict)

        # Only need to iterate once as we recursively handle all keys
        break

    return expanded_parameters

def search_sim_directory(simulations_location: str, parameters: dict = {}):
    matching_dirs = []
    expanded_parameters_list = expand_parameters(parameters)

    for expanded_params in expanded_parameters_list:
        for directory in os.listdir(simulations_location):
            dir_path = os.path.join(simulations_location, directory)
            if os.path.isdir(dir_path):
                try:
                    if simulation_crash_check(dir_path):
                        continue
                except FileNotFoundError:
                    print(('No log file found in {}'.format(dir_path)))
                sim_metrics = CrossSlotParameters(dir_path)
                if sim_metrics.matches_parameters(expanded_params):
                    matching_dirs.append(dir_path)
    
    # Remove duplicates if necessary
    matching_dirs = list(set(matching_dirs))
    return matching_dirs

def simulation_crash_check(filepath:str):
    """
    This function checks whether a simulation has crashed or not.
    It does this by reading the logfile and searching for the string
    'The Simulation is aborted; trying to dump data'.
    """

    logfile = os.path.join(filepath, 'log.txt')
    with open(logfile, 'r') as f:
        for line in f:
            if 'The Simulation is aborted; trying to dump data' in line:
                return True
    return False

def vector_radial_coordinates(cartesian_vector: np.ndarray, trajectory: Trajectory):
        """
        converts a vector at each point on the trajectory to 
        radial, tangential and axial coordinates
        """
        # get the particle position
        particle_position = trajectory.return_coordinates(normalise=False)

        # define the particle displacement vector from cross slot centre
        centre = trajectory.input_parameters.lattice_centre()
        centre_displacement = centre - particle_position

        # define the radial unit vector
        radial_displacement = np.array([
            centre_displacement[:, 0],
            np.zeros_like(particle_position[:, 1]), # y component = zero
            centre_displacement[:, 2]]
        ).T
        radial_unit_vector = radial_displacement / np.linalg.norm(
            radial_displacement, axis=1
            )[:, np.newaxis]

        # define the tangential unit vector
        y_axis_unit_vector = np.array([0, 1, 0])
        tangential_unit_vector = np.cross(
            radial_unit_vector, y_axis_unit_vector
        )
        tangential_unit_vector /= np.linalg.norm(
            tangential_unit_vector, axis=1
            )[:, np.newaxis] # normalize the tangential unit vector

        # make a 2d vector of the y_axis_unit_vector
        y_axis_unit_vector_2d = np.tile(
            y_axis_unit_vector, (len(particle_position), 1)
        )

        # Project the original vector onto the new basis vectors
        radial_component = np.einsum(
            'ij,ij->i', cartesian_vector, radial_unit_vector
        )
        tangential_component = np.einsum(
            'ij,ij->i', cartesian_vector, tangential_unit_vector
        )
        axial_component = np.einsum(
            'ij,ij->i', cartesian_vector, y_axis_unit_vector_2d
        )

        transformed = np.array([
            radial_component, tangential_component, axial_component
        ]).T

        return transformed