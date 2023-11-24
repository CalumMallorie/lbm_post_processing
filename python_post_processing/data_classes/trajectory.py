from .simulation_metrics import SimulationMetrics
from .simulation_crash_check import simulation_crash_check
import pandas as pd
import numpy as np
import os

class Trajectory:
    """Class """
    def __init__(self, filepath: str, junction_only: bool = True):
        # Check if the simulation has crashed
        if simulation_crash_check(filepath):
            raise ValueError(f"Cannot process trajectory; simulation in file {filepath} has crashed")
        self.filepath = filepath
        self.simulation_metrics = SimulationMetrics(filepath)

        if junction_only:
            self.particle_data = self.filter_by_junction(self.read_particle_datafile(filepath))
        else:
            self.particle_data = self.read_particle_datafile(filepath)

        self.coordinates =  self.extract_coordinates(self.particle_data)
        self.normalised_coordinates = self.normalise(self.coordinates)

        self.times = np.array(self.particle_data['time'])
        self.velocity = np.array(self.particle_data.loc[:, ['v_x', 'v_y', 'v_z']])
        
    def read_particle_datafile(self, filepath: str):
        df = []
        path = os.path.join(filepath, 'Particles', 'Particle_0.dat')
        df = pd.read_csv(path, sep=' ', skiprows=22, index_col=False)
        return df
    
    def filter_by_junction(self, df: pd.DataFrame):
        centre = self.simulation_metrics.cross_slot_data.centre
        height = self.simulation_metrics.cross_slot_data.channel_height
        inlet_width = self.simulation_metrics.cross_slot_data.inlet_width
        outlet_width = self.simulation_metrics.cross_slot_data.outlet_width

        x_min, x_max = centre[0] - 0.5*outlet_width, centre[0] + 0.5*outlet_width
        y_min, y_max = centre[1] - 0.5*inlet_width, centre[1] + 0.5*inlet_width
        z_min, z_max = centre[2] - 0.5*height, centre[2] + 0.5*height

        # Apply the filter
        filtered_df = df[(df['x'] >= x_min) & (df['x'] <= x_max) &
                        (df['y'] >= y_min) & (df['y'] <= y_max) &
                        (df['z'] >= z_min) & (df['z'] <= z_max)]
        
        return filtered_df
    
    def extract_coordinates(self, df: pd.DataFrame):
        return np.array(df.loc[:, ['x', 'y', 'z']])
    
    def normalise(self, coordinates):

        centre = self.simulation_metrics.cross_slot_data.centre
        height = self.simulation_metrics.cross_slot_data.channel_height
        inlet_width = self.simulation_metrics.cross_slot_data.inlet_width
        outlet_width = self.simulation_metrics.cross_slot_data.outlet_width

        normalised_coordinates_x = (coordinates[:, 0]-centre[0])/(0.5*outlet_width)
        normalised_coordinates_y = (coordinates[:, 1]-centre[1])/(0.5*inlet_width)
        normalised_coordinates_z = (coordinates[:, 2]-centre[2])/(height)
        return np.array([normalised_coordinates_x, normalised_coordinates_y, normalised_coordinates_z]).T
    


