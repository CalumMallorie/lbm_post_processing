"""
helper_functions.py
===================

This module contains various helper functions used for post-processing
simulation data. Functions include parameter expansion, directory search,
simulation crash check, and vector coordinate transformation.

Functions
---------
expand_parameters(parameters):
    Expands dictionary parameters containing lists into separate
    dictionaries, each with one value from the list.

search_sim_directory(simulations_location, parameters):
    Searches for directories matching the given parameters.

simulation_crash_check(filepath):
    Checks if a simulation has crashed by reading the log file.

vector_radial_coordinates(cartesian_vector, trajectory):
    Converts a vector at each point on a trajectory to radial,
    tangential, and axial coordinates.

Classes
-------
CrossSlotParameters
    Class to manage simulation parameters.

Trajectory
    Class to manage particle trajectory data.
"""

import os
from .read_input_parameters import CrossSlotParameters
import numpy as np
import pandas as pd

def expand_parameters(parameters: dict):
    """
    Expands dictionary parameters containing lists into separate
    dictionaries, each with one value from the list.

    Parameters
    ----------
    parameters : dict
        Dictionary of parameters to expand. Values can be lists.

    Returns
    -------
    list of dict
        List of dictionaries, each containing one value from the original
        lists.
    """
    if not parameters:
        return [{}]

    expanded_parameters = []
    for key, value in parameters.items():
        remaining_params = parameters.copy()
        del remaining_params[key]

        if isinstance(value, list):
            for val in value:
                for expanded_dict in expand_parameters(remaining_params):
                    expanded_dict[key] = val
                    expanded_parameters.append(expanded_dict)
        else:
            for expanded_dict in expand_parameters(remaining_params):
                expanded_dict[key] = value
                expanded_parameters.append(expanded_dict)
        break

    return expanded_parameters

def search_sim_directory(simulations_location: str, parameters: dict = {}, exclude: list = []):
    """
    Searches for directories matching the given parameters.

    Parameters
    ----------
    simulations_location : str
        Path to the location containing simulation directories.
    parameters : dict, optional
        Dictionary of parameters to match (default is {}).
    exclude : list, optional
        List of directory or file names to ignore (default is []).

    Returns
    -------
    list of str
        List of paths to directories matching the given parameters.
    """
    matching_dirs = []
    expanded_parameters_list = expand_parameters(parameters)

    for expanded_params in expanded_parameters_list:
        for directory in os.listdir(simulations_location):
            if directory in exclude:
                continue
            dir_path = os.path.join(simulations_location, directory)
            if os.path.isdir(dir_path):
                try:
                    if simulation_crash_check(dir_path):
                        continue
                except FileNotFoundError:
                    print('No log file found in {}'.format(dir_path))
                sim_metrics = CrossSlotParameters(dir_path)
                if sim_metrics.matches_parameters(expanded_params):
                    matching_dirs.append(dir_path)
    
    matching_dirs = list(set(matching_dirs))
    return matching_dirs

def simulation_crash_check(filepath: str):
    """
    Checks if a simulation has crashed by reading the log file.

    Parameters
    ----------
    filepath : str
        Path to the simulation directory.

    Returns
    -------
    bool
        True if the simulation has crashed, False otherwise.
    """
    logfile = os.path.join(filepath, 'log.txt')
    with open(logfile, 'r') as f:
        for line in f:
            if 'The Simulation is aborted; trying to dump data' in line:
                return True
    return False

def vector_radial_coordinates(cartesian_vector: np.ndarray, trajectory):
    """
    Converts a vector at each point on a trajectory to radial, tangential,
    and axial coordinates.

    Parameters
    ----------
    cartesian_vector : np.ndarray
        Vector in Cartesian coordinates.
    trajectory : Trajectory
        Trajectory object containing particle position data.

    Returns
    -------
    np.ndarray
        Transformed vector in radial, tangential, and axial coordinates.
    """
    particle_position = trajectory.return_coordinates(normalise=False)
    centre = trajectory.input_parameters.lattice_centre()
    centre_displacement = centre - particle_position

    radial_displacement = np.array([
        centre_displacement[:, 0],
        np.zeros_like(particle_position[:, 1]),
        centre_displacement[:, 2]
    ]).T
    radial_unit_vector = radial_displacement / np.linalg.norm(
        radial_displacement, axis=1
    )[:, np.newaxis]

    y_axis_unit_vector = np.array([0, 1, 0])
    tangential_unit_vector = np.cross(
        radial_unit_vector, y_axis_unit_vector
    )
    tangential_unit_vector /= np.linalg.norm(
        tangential_unit_vector, axis=1
    )[:, np.newaxis]

    y_axis_unit_vector_2d = np.tile(
        y_axis_unit_vector, (len(particle_position), 1)
    )

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
