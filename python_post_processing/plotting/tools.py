from ..data_classes.simulation_metrics import SimulationMetrics
from ..data_classes.simulation_crash_check import simulation_crash_check
import os
import matplotlib.pyplot as plt

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

def search_sim_directory(simulations_location: str, parameters: dict):
    matching_dirs = []
    expanded_parameters_list = expand_parameters(parameters)

    for expanded_params in expanded_parameters_list:
        for directory in os.listdir(simulations_location):
            dir_path = os.path.join(simulations_location, directory)
            if os.path.isdir(dir_path):
                if simulation_crash_check(dir_path):
                    continue
                sim_metrics = SimulationMetrics(dir_path)
                if sim_metrics.matches_parameters(expanded_params):
                    matching_dirs.append(dir_path)
    
    # Remove duplicates if necessary
    matching_dirs = list(set(matching_dirs))
    return matching_dirs

def setup_plot():
    fig, ax = plt.subplots(figsize=(4,4))
    return fig, ax