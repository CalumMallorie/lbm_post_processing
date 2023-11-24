import os

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