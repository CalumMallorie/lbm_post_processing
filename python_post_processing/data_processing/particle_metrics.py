from ..data_classes.sim_input import CrossSlot
from ..data_classes.trajectory import Trajectory
from ..data_classes.simulation_crash_check import simulation_crash_check

import os
import json
import numpy as np
import pandas as pd
import pyvista as pv

class ParticleGlobalMetrics:
    """
    Class for processing global particle data as trajectory timeseries. 
    Global here means that each metric corresponds to the whole particle,
    rather than over the particle surface.
    """

    def __init__(self, filepath: str, junction_only: bool = True) -> None:
        if simulation_crash_check(filepath):
            raise ValueError(
                f"Cannot process trajectory; simulation in file" +
                f"{filepath} has crashed"
                )
        
        self.filepath = filepath
        self.junction_only = junction_only

        # Load the cross slot simulation input data
        self.cross_slot = CrossSlot(filepath)

        # Read and process the traction data
        self.integrated_traction_df = self.process_traction_data(
            filepath, junction_only
            )

        # Read and process the principal axes data
        self.principal_axes = self.process_principal_axes(
            filepath, junction_only
            )

    def read_particle_datafile(self, filepath: str) -> pd.DataFrame:
            path = os.path.join(
                filepath, 'Particles', 'Particle_0.dat'
                )
            df = pd.read_csv(
                path, sep=' ', skiprows=22, index_col=False
                )
            return df
    
    def read_particle_axes_file(self, filepath: str):
        df = []
        path = os.path.join(filepath, 'Particles', 'Axes_0.dat')
        df = pd.read_csv(path, sep=' ', skiprows=6, index_col=False)
        return df
    
    def read_particle_vtks(self):
        simulation_filepath = self.filepath
        # Read the particle vtks
        vtk_directory = os.path.join(simulation_filepath, 'VTKParticles')
        particle_vtks = {}
        for file in os.listdir(vtk_directory):
            if file.endswith(".vtp") and 'Axes' not in file:
                timestep = int(file.split('_')[1].split('.')[0].split('t')[1])
                vtk_location = os.path.join(vtk_directory, file)
                particle_vtk = pv.read(vtk_location)
                particle_vtks[timestep] = particle_vtk
        
        return particle_vtks
    
    def process_principal_axes(self, filepath: str, junction_only: bool):
        # Read the particle axes data
        axes_data = self.read_particle_axes_file(filepath)
        axes_data['timestep'] = axes_data['time'].astype(int)
        axes_data = self.calculate_shape_factor(axes_data)
        axes_data = self.calculate_taylor_deformation(axes_data)

        if junction_only:
            filtered =  self.filter_df_by_junction(axes_data)
        else:
            filtered =  axes_data

        # Calculate the shape factor
        return filtered
        
    def calculate_shape_factor(self, axes_data: pd.DataFrame):
        # Calculate the shape factor from the axes data. The shape
        # factor allows particles to be differentiated between prolate,
        # oblate, and spherical.

        max = axes_data['a']
        mid = axes_data['b']
        min = axes_data['c']
        axes_data['shape_factor'] = 2*((mid-max)/(mid+max)) + ((max-min)/(max+min))
        return axes_data
    
    def calculate_taylor_deformation(self, axes_data: pd.DataFrame):
        # Taylor deformation TD = (a-c)/(a+c)
        axes_data['taylor_deformation'] = (axes_data['a']-axes_data['c'])/(axes_data['a']+axes_data['c'])
        return axes_data
    
    def process_traction_data(self, filepath: str, junction_only: bool):
        # Read and convert the integrated traction force data, and 
        # filter it to only include timesteps within the junction if 
        # junction_only is True

        traction_data = self.read_integrated_traction_forces(filepath)
        converted_df = self.convert_integrated_traction_forces(
            traction_data
            )
        if junction_only:
            return self.filter_df_by_junction(converted_df)
        else:
            return converted_df
    
    def read_integrated_traction_forces(self, filepath: str) -> pd.DataFrame:
        # load json file
        json_path = "TractionParticles/integrated_forces.json"
        with open(os.path.join(filepath, json_path)) as json_file:
            data = json.load(json_file)
        data = pd.DataFrame(data)
        return data

    def convert_integrated_traction_forces(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Reads the json file that stores the integrated traction forces.
        The data is stored in a pandas DataFrame. The force columns are 
        converted from lists to numpy arrays, and the timestep column is
        converted to an int. The magnitudes of the forces are also calculated.
        """

        # Function to convert lists to numpy arrays
        def convert_to_array(item):
            if isinstance(item, list):
                return np.array(item)
            return item

        # Apply convert_to_array to each element in the DataFrame to convert
        # lists to numpy arrays
        data = data.applymap(convert_to_array)

        # convert the timestep column to int
        data['timestep'] = data['timestep'].astype(int)

        # sort the dataframe by timestep
        data = data.sort_values(by=['timestep'])

        # calculate the magnitudes of the forces
        data['total_traction_magnitude'] = data['total_traction'].apply(np.linalg.norm)
        data['totaldev_force_magnitude'] = data['total_dev_force'].apply(np.linalg.norm)
        data['total_press_force_mag'] = data['total_press_force'].apply(np.linalg.norm)
        
        return data

    def filter_df_by_junction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters the dataframe to only include timesteps within the junction.
        This works by looking at the particle trajectory and finding the x, y, z
        coordinates of the junction. Then the dataframe is filtered to only
        include timesteps where the particle is within the junction.
        """

        particle_dataframe = self.read_particle_datafile(self.filepath)

        centre = self.cross_slot.centre
        height = self.cross_slot.channel_height
        inlet_width = self.cross_slot.inlet_width
        outlet_width = self.cross_slot.outlet_width

        x_min, x_max = centre[0] - 0.5*outlet_width, centre[0] + 0.5*outlet_width
        y_min, y_max = centre[1] - 0.5*inlet_width, centre[1] + 0.5*inlet_width
        z_min, z_max = centre[2] - 0.5*height, centre[2] + 0.5*height

        # Apply the filter
        filtered_particle_dataframe = particle_dataframe[(particle_dataframe['x'] >= x_min) & (particle_dataframe['x'] <= x_max) &
                        (particle_dataframe['y'] >= y_min) & (particle_dataframe['y'] <= y_max) &
                        (particle_dataframe['z'] >= z_min) & (particle_dataframe['z'] <= z_max)]
        
        # Filter the integrated traction dataframe. This is done by finding the
        # timesteps that are present in the filtered particle dataframe.
        filtered_df = df[df['timestep'].isin(filtered_particle_dataframe['time'])]
        return filtered_df
    
    def convert_traction_forces_to_radial(self):

        # split up the traction contributions into vector timeseries
        total_traction = np.vstack(
            self.integrated_traction_df['total_traction'].to_numpy()
        )
        total_dev_force = np.vstack(
            self.integrated_traction_df['total_dev_force'].to_numpy()
        )
        total_press_force = np.vstack(
            self.integrated_traction_df['total_press_force'].to_numpy()
        )

        # find the position vector in cartesian coordinates
        trajectory = Trajectory(
            self.filepath,junction_only=self.junction_only
        )
        particle_position = trajectory.coordinates

        # define the particle displacement vector from cross slot centre
        centre = self.cross_slot.centre
        centre_displacement = particle_position - centre

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

        # Project the total traction forces onto the new basis vectors
        radial_traction = np.einsum(
            'ij,ij->i', total_traction, radial_unit_vector
        )
        tangential_traction = np.einsum(
            'ij,ij->i', total_traction, tangential_unit_vector
        )
        axial_traction = np.einsum(
            'ij,ij->i', total_traction, y_axis_unit_vector_2d
        )

        # Project the total deviatoric forces onto the new basis vectors
        radial_dev_force = np.einsum(
            'ij,ij->i', total_dev_force, radial_unit_vector
        )
        tangential_dev_force = np.einsum(
            'ij,ij->i', total_dev_force, tangential_unit_vector
        )
        axial_dev_force = np.einsum(
            'ij,ij->i', total_dev_force, y_axis_unit_vector_2d
        )

        # Project the total pressure forces onto the new basis vectors
        radial_press_force = np.einsum(
            'ij,ij->i', total_press_force, radial_unit_vector
        )
        tangential_press_force = np.einsum(
            'ij,ij->i', total_press_force, tangential_unit_vector
        )
        axial_press_force = np.einsum(
            'ij,ij->i', total_press_force, y_axis_unit_vector_2d
        )


        # Add the radial and tangential traction forces to dataframe
        integrated_radial_traction = pd.DataFrame({
            'timestep': self.integrated_traction_df['timestep'],
            'total_traction_radial': radial_traction,
            'total_traction_tangential': tangential_traction,
            'total_traction_axial': axial_traction,
            'dev_traction_radial': radial_dev_force,
            'dev_traction_tangential': tangential_dev_force,
            'dev_traction_axial': axial_dev_force,
            'press_traction_radial': radial_press_force,
            'press_traction_tangential': tangential_press_force,
            'press_traction_axial': axial_press_force,
        })

        return integrated_radial_traction
    
    def average_radial_traction_forces(self):
        # Convert the traction forces to radial
        df = self.convert_traction_forces_to_radial()
        df.drop('timestep', axis=1, inplace=True)

        # Average the radial traction forces
        averages = df.mean()
        return averages
    
    def particle_projected_area(self, direction):
        particle_vtks = self.read_particle_vtks()
        projected_areas = []

        for timestep, vtk in particle_vtks.items():
            try:
                projected_mesh = vtk.project_points_to_plane(
                    origin=self.cross_slot.centre,
                    normal=direction
                )
                particle_area = projected_mesh.area
                projected_areas.append({'timestep': timestep, 'projected_area': particle_area})
            except Exception as e:
                print(f"Error processing timestep {timestep}: {e}")

        initial_area = projected_areas[0]['projected_area']
        for item in projected_areas:
            item['projected_area'] /= initial_area
        projected_areas_df = pd.DataFrame(projected_areas)
        # projected_areas_df = self.filter_df_by_junction(projected_areas_df)
        return projected_areas_df
            

            