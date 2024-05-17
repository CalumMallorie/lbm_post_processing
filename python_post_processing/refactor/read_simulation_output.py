import os 
import pandas as pd
import numpy as np

from read_input_parameters import CrossSlotParameters

class Trajectory:
    def __init__(self, filepath: str, junction_only: bool = True):
        self.input_parameters = CrossSlotParameters(filepath)
        if junction_only:
            unfiltered_datafile = self.read_particle_datafile(filepath)
            self.datafile = self.filter_by_junction(unfiltered_datafile)
        else:
            self.datafile = self.read_particle_datafile(filepath)
        return 
    
    def read_particle_datafile(self, filepath: str):
        df = []
        try:
            path = os.path.join(filepath, 'Particles', 'Particle_0.dat')
            df = pd.read_csv(path, sep=' ', skiprows=22, index_col=False)
        except FileNotFoundError:
            print(f"File {path} not found.")
        except PermissionError:
            print(f"Permission denied for file {path}.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")
        return df
    
    def filter_by_junction(self, df: pd.DataFrame):
        centre = self.input_parameters.lattice_centre()
        height = self.input_parameters.channel_height()
        inlet_width = self.input_parameters.inlet_width()
        outlet_width = self.input_parameters.outlet_width()

        x_min, x_max = centre[0] - 0.5*outlet_width, centre[0] + 0.5*outlet_width
        y_min, y_max = centre[1] - 0.5*inlet_width, centre[1] + 0.5*inlet_width
        z_min, z_max = centre[2] - 0.5*height, centre[2] + 0.5*height

        # Apply the filter
        filtered_df = df[(df['x'] >= x_min) & (df['x'] <= x_max) &
                        (df['y'] >= y_min) & (df['y'] <= y_max) &
                        (df['z'] >= z_min) & (df['z'] <= z_max)]
        
        return filtered_df
    
    def return_coordinates(self, normalise: bool = False):
        coordinates = np.array(self.datafile.loc[:, ['x', 'y', 'z']])
        if normalise:
            centre = self.input_parameters.lattice_centre()
            height = self.input_parameters.channel_height()
            inlet_width = self.input_parameters.inlet_width()
            outlet_width = self.input_parameters.outlet_width()

            normalised_coordinates_x = (coordinates[:, 0]-centre[0])/(0.5*outlet_width)
            normalised_coordinates_y = (coordinates[:, 1]-centre[1])/(0.5*inlet_width)
            normalised_coordinates_z = (coordinates[:, 2]-centre[2])/(0.5*height)
            return np.array([normalised_coordinates_x, normalised_coordinates_y, normalised_coordinates_z]).T
        else:
            return coordinates
        
    def return_velocity(self):
        return np.array(self.datafile.loc[:, ['v_x', 'v_y', 'v_z']])
    
    def return_times(self, normalise: bool = False):
        if normalise:
            times = np.array(self.datafile['time']) 
            return (times - times[-1])/self.input_parameters.advection_time()
        else:
            return np.array(self.datafile['time'])
