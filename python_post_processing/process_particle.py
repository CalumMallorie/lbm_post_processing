"""
process_particle.py
====================

This module provides the `ProcessParticle` class for processing global
particle data as trajectory timeseries. The metrics correspond to the whole
particle rather than its surface.

Classes
-------
ProcessParticle
    Class for processing global particle data as trajectory timeseries.

Dependencies
------------
- os
- numpy
- pandas
- pyvista
- scipy.linalg.eigh
- concurrent.futures.ProcessPoolExecutor


Example Usage
-------------
    from process_particle import ProcessParticle
    
    # Initialize with path to simulation directory
    particle_processor = ProcessParticle('path/to/simulation')
    
    # Read particle data
    particle_data = particle_processor.read_particle_datafile('path/to/simulation')
    
    # Compute shape factor
    shape_factor = particle_processor.shape_factor()
"""

from .read_input_parameters import CrossSlotParameters
from .process_trajectory import Trajectory
from .helper_functions import vector_radial_coordinates

import os
import numpy as np
import pandas as pd
import pyvista as pv
from scipy.linalg import eigh
from concurrent.futures import ProcessPoolExecutor

class ProcessParticle:
    """
    Class for processing global particle data as trajectory timeseries.
    Global here means that each metric corresponds to the whole particle,
    rather than over the particle surface.

    Attributes
    ----------
    filepath : str
        Path to the simulation directory.
    junction_only : bool
        Whether to process only junction data.
    cross_slot : CrossSlotParameters
        Instance of CrossSlotParameters for extracting simulation parameters.
    manual_axes : pd.DataFrame or None
        Placeholder for the particle axes data if they need manual calculation.

    Methods
    -------
    __init__(self, filepath: str, junction_only: bool = True)
        Initializes the ProcessParticle class with the path to the simulation directory.
    read_particle_datafile(self, filepath: str) -> pd.DataFrame
        Reads the particle data file and returns it as a pandas DataFrame.
    read_particle_axes_file(self, filepath: str) -> pd.DataFrame
        Reads the particle axes data from a specified file.
    return_particle_axes(self) -> pd.DataFrame
        Returns the particle axes data as a pandas DataFrame.
    manual_extract_axes(self) -> pd.DataFrame
        Manually extracts the particle axes data.
    read_particle_vtks(self) -> dict
        Reads particle VTK files and returns them as a dictionary.
    read_local_fluid_vtks(self) -> dict
        Reads local fluid VTK files and returns them as a dictionary.
    process_principal_axes(self, filepath: str, junction_only: bool) -> pd.DataFrame
        Processes the principal axes data.
    shape_factor(self) -> float
        Calculates the shape factor from the particle axes data.
    taylor_deformation(self) -> pd.Series
        Calculates the Taylor deformation.
    read_integrated_traction_forces(self) -> pd.DataFrame
        Reads the integrated traction forces from a file.
    filter_df_by_junction(self, df: pd.DataFrame) -> pd.DataFrame
        Filters the DataFrame to include timesteps within the junction.
    radial_traction_forces(self, option: str) -> np.ndarray
        Calculates the radial traction forces.
    stresslet(self) -> tuple
        Calculates the stresslet and its eigenvalues and eigenvectors.
    torque(self, option: str = None) -> np.ndarray
        Calculates the torque on the particle.
    particle_projected_area(self, direction: str) -> pd.DataFrame
        Calculates the projected area of the particle.
    dissipation(self, index: float) -> dict
        Calculates the fluid dissipation in the particle volume.
    compute_inertia_tensor(self, vtk: pv.PolyData) -> np.ndarray
        Computes the inertia tensor for the particle.
    inertia_tensor_timeseries(self) -> dict
        Computes the inertia tensor timeseries for the particle.
    thickness_span_cytovale(self) -> pd.DataFrame
        Computes the thickness span of the particle.
    max_distances(self, points: np.ndarray, plane: str) -> float
        Calculates the maximum distance in the specified plane.
    process_vtk(self, timestep: int, vtk: pv.PolyData, plane: str) -> tuple
        Processes a VTK file to compute distances.
    max_stretch_plane(self, plane: str) -> dict
        Calculates the maximum stretch in the specified plane.
    """

    def __init__(self, filepath: str, junction_only: bool = True) -> None:
        """Initialize the ProcessParticle class with the path to the
        simulation directory.

        Args
        ----
        filepath : str
            The path to the simulation directory.
        junction_only : bool, optional
            Whether to process only junction data (default is True).
        """
        self.filepath = filepath
        self.junction_only = junction_only

        # Load the cross slot simulation input data
        self.cross_slot = CrossSlotParameters(filepath)
        
        # placeholder for the particle axes data if they need manual calc
        self.manual_axes = None

    def read_particle_datafile(self, filepath: str) -> pd.DataFrame:
        """Read the particle data file and return it as a pandas DataFrame.

        Args
        ----
        filepath : str
            The path to the directory containing the 'Particles' folder.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the particle data.
        """
        path = os.path.join(
            filepath, 'Particles', 'Particle_0.dat'
            )
        df = pd.read_csv(
            path, sep=' ', skiprows=22, index_col=False
            )
        return df
    
    def read_particle_axes_file(self, filepath: str):
        """
        Read the particle axes data from a specified file, or manually extract
        the data if the file is not found.

        Args:
            filepath: The path to the directory containing the 'Particles'
                    folder and the 'Axes_0.dat' file.

        Returns:
            A pandas DataFrame containing the axes data, or None if the file 
            could not be read.
        """
        try:
            file_path = os.path.join(filepath, 'Particles', 'Axes_0.dat')
            df = pd.read_csv(file_path, sep=' ', skiprows=6, index_col=False)
            return df
        except FileNotFoundError:
            return self.manual_extract_axes()
        
    def return_particle_axes(self):
        """Return the particle axes data as a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the particle axes data.
        """
        if self.junction_only:
            return self.filter_df_by_junction(self.read_particle_axes_file(self.filepath))
        else:
            return self.read_particle_axes_file(self.filepath)
            
    def manual_extract_axes(self):
        """Manually extract the particle axes data.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the manually extracted particle axes data.
        """
        def compute_areas_normals_vectorized(mesh: pv.PolyData):
            # Extract vertices for all faces, assuming triangular faces
            # Faces are stored in a flat array with a fixed pattern: [n_vertices, idx1, idx2, idx3, ...]
            faces = mesh.faces.reshape(-1, 4)[:, 1:]  # Reshape and skip 'n_vertices'
            vertices = mesh.points[faces]  # Shape: (n_faces, 3, 3)

            # Compute vectors for cross product
            vec0 = vertices[:, 1, :] - vertices[:, 0, :]
            vec1 = vertices[:, 2, :] - vertices[:, 0, :]
            cross_products = np.cross(vec0, vec1)

            # Compute areas (0.5 * norm of cross product)
            areas = 0.5 * np.linalg.norm(cross_products, axis=1)

            # Compute normals (normalize cross products)
            normals = -cross_products / np.linalg.norm(cross_products, axis=1)[:, np.newaxis]

            return areas, normals

        def compute_inertia_tensor_volume(mesh: pv.PolyData):
            centre = mesh.center
            areas, normals = compute_areas_normals_vectorized(mesh)
            
            # Extract vertices for all faces and compute centroids
            faces = mesh.faces.reshape((-1, 4))[:, 1:]
            centroids = np.mean(mesh.points[faces], axis=1)
            pv = centroids - centre  # Position vectors of centroids relative to mesh center
            
            # Compute sp for all faces
            sp = np.einsum('ij,ij->i', pv, normals)
            
            # Pre-compute squared and product terms
            pv_squared = np.sum(pv**2, axis=1)
            pv_products = np.array([pv[:, 0]*pv[:, 1], pv[:, 0]*pv[:, 2], pv[:, 1]*pv[:, 2]]).T
            
            # Contributions to inertia tensor components
            contributions = np.zeros((len(faces), 3, 3))
            contributions[:, 0, 0] = areas / 5 * (pv_squared * sp - pv[:, 0]**2 * sp)
            contributions[:, 1, 1] = areas / 5 * (pv_squared * sp - pv[:, 1]**2 * sp)
            contributions[:, 2, 2] = areas / 5 * (pv_squared * sp - pv[:, 2]**2 * sp)
            contributions[:, 0, 1] = contributions[:, 1, 0] = -areas / 5 * pv_products[:, 0] * sp
            contributions[:, 0, 2] = contributions[:, 2, 0] = -areas / 5 * pv_products[:, 1] * sp
            contributions[:, 1, 2] = contributions[:, 2, 1] = -areas / 5 * pv_products[:, 2] * sp
            
            # Sum contributions to get the inertia tensor
            inertia_tensor = np.sum(contributions, axis=0)
            
            return inertia_tensor
        
        if self.manual_axes is not None:
            return self.manual_axes
        
        particle_vtks = self.read_particle_vtks()
        manual_axes = pd.DataFrame()

        for timestep, vtk in particle_vtks.items():
            try:
                # Compute the inertia tensor for the particle
                inertia_tensor = compute_inertia_tensor_volume(vtk)

                # Diagonalize the inertia tensor
                eigenvalues, eigenvectors = eigh(inertia_tensor)

                # Compute a, b, c assuming ellipsoidal shapes
                volume = vtk.volume  # Assuming mesh volume can be directly obtained
                a = np.sqrt(2.5 * (eigenvalues[1] + eigenvalues[2] - eigenvalues[0]) / volume)
                b = np.sqrt(2.5 * (eigenvalues[0] + eigenvalues[2] - eigenvalues[1]) / volume)
                c = np.sqrt(2.5 * (eigenvalues[0] + eigenvalues[1] - eigenvalues[2]) / volume)

                # Add the data to the DataFrame
                manual_axes = manual_axes._append({
                    'time': timestep,
                    'a': a,
                    'b': b,
                    'c': c
                }, ignore_index=True)
            except Exception as e:
                print(f"Error processing timestep {timestep}: {e}")
        self.manual_axes = manual_axes
        return manual_axes

    
    def read_particle_vtks(self) -> dict:
        """Reads particle VTK files and returns them as a dictionary.

        Returns
        -------
        dict
            Dictionary where keys are timesteps and values are VTK objects.
        """
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
    
    def read_local_fluid_vtks(self):
        """Reads local fluid VTK files and returns them as a dictionary.

        Returns
        -------
        dict
            Dictionary where keys are timesteps and values are VTK objects.
        """
        simulation_filepath = self.filepath
        # timesteps to read:
        times_to_read = Trajectory(
            simulation_filepath, 
            junction_only=self.junction_only
        ).return_times()

        # Read the fluid vtks
        vtk_directory = os.path.join(simulation_filepath, 'VTKLocalFluid')
        fluid_vtks = {}
        for file in os.listdir(vtk_directory):
            if file.endswith(".vtr"):
                timestep = int(file.split('_')[1].split('.')[0].split('t')[1])
                if timestep in times_to_read:
                    vtk_location = os.path.join(vtk_directory, file)
                    fluid_vtk = pv.read(vtk_location)
                    fluid_vtks[timestep] = fluid_vtk

        return fluid_vtks
    
    def process_principal_axes(self, filepath: str, junction_only: bool):
        """Processes the principal axes data.

        Args
        ----
        filepath : str
            The path to the directory containing the 'Particles' folder.
        junction_only : bool
            Whether to process only junction data.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the processed principal axes data.
        """
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
    
    def shape_factor(self):
        """Calculates the shape factor from the particle axes data.

        The shape factor distinguishes particles as prolate, oblate,
        or spherical based on their geometry.

        Returns
        -------
        float
            The shape factor of the particle, or None if the calculation
            fails due to missing or invalid data.
        """
        try:
            axes_data = self.read_particle_axes_file(self.filepath)
            
            if self.junction_only:
                axes_data = self.filter_df_by_junction(axes_data)

            major_axis = axes_data['a']
            intermediate_axis = axes_data['b']
            minor_axis = axes_data['c']

            shape_factor_value = (2 * ((intermediate_axis - major_axis) / 
                                    (intermediate_axis + major_axis)) + 
                                ((major_axis - minor_axis) / (major_axis + minor_axis)))
            return shape_factor_value
        except Exception as e:
            # Log the error or handle it as needed
            print(f"Failed to calculate the shape factor: {e}")
            return None

    def taylor_deformation(self):
        """Calculates the Taylor deformation.

        Returns
        -------
        pd.Series
            Series containing the Taylor deformation values.
        """
        # Taylor deformation TD = (a-c)/(a+c)

        if self.junction_only:
            axes_data = self.read_particle_axes_file(self.filepath)
            axes_data = self.filter_df_by_junction(axes_data)
        else:
            axes_data = self.read_particle_axes_file(self.filepath)

        taylor_deformation = (axes_data['a']-axes_data['c'])/(axes_data['a']+axes_data['c'])
        return taylor_deformation
    
    def read_integrated_traction_forces(self) -> pd.DataFrame:
        """Reads the integrated traction forces from a file.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the integrated traction forces data.
        """
        # load the integrated traction forces
        path = os.path.join(self.filepath, 'converted', 'force_analysis.csv')
        df = pd.read_csv(path)

        df.rename(columns={'timestep': 'time'}, inplace=True)
        if self.junction_only:
            return self.filter_df_by_junction(df)
        else:
            return df

    def filter_df_by_junction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filters the DataFrame to include timesteps within the junction.

        This works by looking at the particle trajectory and finding the x, y,
        z coordinates of the junction. Then the DataFrame is filtered to only
        include timesteps where the particle is within the junction.

        Args
        ----
        df : pd.DataFrame
            DataFrame to be filtered.

        Returns
        -------
        pd.DataFrame
            Filtered DataFrame.
        """

        particle_dataframe = self.read_particle_datafile(self.filepath)

        centre = self.cross_slot.lattice_centre()
        height = self.cross_slot.channel_height()
        inlet_width = self.cross_slot.inlet_width()
        outlet_width = self.cross_slot.outlet_width()

        x_min, x_max = centre[0] - 0.5*outlet_width, centre[0] + 0.5*outlet_width
        y_min, y_max = centre[1] - 0.5*inlet_width, centre[1] + 0.5*inlet_width
        z_min, z_max = centre[2] - 0.5*height, centre[2] + 0.5*height

        # Apply the filter
        filtered_particle_dataframe = particle_dataframe[(particle_dataframe['x'] >= x_min) & (particle_dataframe['x'] <= x_max) &
                        (particle_dataframe['y'] >= y_min) & (particle_dataframe['y'] <= y_max) &
                        (particle_dataframe['z'] >= z_min) & (particle_dataframe['z'] <= z_max)]
        
        # Filter the integrated traction dataframe. This is done by finding the
        # timesteps that are present in the filtered particle dataframe.
        filtered_df = df[df['time'].isin(filtered_particle_dataframe['time'])]
        return filtered_df
    
    def radial_traction_forces(self, option: str):
        """Calculates the radial traction forces.

        Args
        ----
        option : str
            The type of force to calculate ('traction', 'press', 'dev').

        Returns
        -------
        np.ndarray
            Array containing the radial traction forces.
        """
        integrated_traction_df = self.read_integrated_traction_forces()

        traction = integrated_traction_df[['traction_forces_x', 'traction_forces_y', 'traction_forces_z']].to_numpy()
        press_force = integrated_traction_df[['press_forces_x', 'press_forces_y', 'press_forces_z']].to_numpy()
        dev_force = integrated_traction_df[['dev_forces_x', 'dev_forces_y', 'dev_forces_z']].to_numpy()

        # find the position vector in cartesian coordinates
        trajectory = Trajectory(
            self.filepath,junction_only=self.junction_only
        )

        traction_radial = vector_radial_coordinates(traction, trajectory)
        press_force_radial = vector_radial_coordinates(press_force, trajectory)
        dev_force_radial = vector_radial_coordinates(dev_force, trajectory)

        if option == 'traction':
            return traction_radial
        elif option == 'press':
            return press_force_radial
        elif option == 'dev':
            return dev_force_radial
        
    def stresslet(self):
        """Calculates the stresslet and its eigenvalues and eigenvectors.

        Returns
        -------
        tuple
            A tuple containing two lists: eigenvalues and eigenvectors.
        """
        integrated_traction_df = self.read_integrated_traction_forces()

        # the df contains columns:
        # time, 
        # stresslet_x_x,  stresslet_x_y,  stresslet_x_z
        # stresslet_y_x,  stresslet_y_y,  stresslet_y_z
        # stresslet_z_x,  stresslet_z_y,  stresslet_z_z

        # Prepare to store results
        eigenvalues_list = []
        eigenvectors_list = []

        # Iterate through each row to calculate eigenvalues and eigenvectors
        for index, row in integrated_traction_df.iterrows():
            # Reshape the stresslet components into a 3x3 matrix
            stresslet_matrix = np.array([
                [row['stresslet_x_x'], row['stresslet_x_y'], row['stresslet_x_z']],
                [row['stresslet_y_x'], row['stresslet_y_y'], row['stresslet_y_z']],
                [row['stresslet_z_x'], row['stresslet_z_y'], row['stresslet_z_z']]
            ])

            # Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eig(stresslet_matrix)

            # Append results to lists
            eigenvalues_list.append(eigenvalues)
            eigenvectors_list.append(eigenvectors)

        return eigenvalues_list, eigenvectors_list



        
    def torque(self, option: str = None):
        """Calculates the torque on the particle.

        Args
        ----
        option : str, optional
            The type of force to use for the calculation ('traction', 'press', 'dev').
            If None, the default traction force is used (default is None).

        Returns
        -------
        np.ndarray
            Array containing the torque values.
        """

        # get the traction
        integrated_traction_df = self.read_integrated_traction_forces()
        if option == None:
            traction = integrated_traction_df[['traction_forces_x', 'traction_forces_y', 'traction_forces_z']].to_numpy()
        elif option == 'press':
            traction = integrated_traction_df[['press_forces_x', 'press_forces_y', 'press_forces_z']].to_numpy()
        elif option == 'dev':   
            traction = integrated_traction_df[['dev_forces_x', 'dev_forces_y', 'dev_forces_z']].to_numpy()

        # remove the y component of the traction
        traction[:,1] = 0

        # get the timeseries of positions
        trajectory = Trajectory(
            self.filepath,junction_only=self.junction_only
        )
        position = trajectory.return_coordinates(normalise=False)

        # convert the positions relative to the outlet centreline
        # This means the reference point is the x, z coordinates of the
        # outlet centreline, and the y coordinate of the particle.
        relative_position = position - self.cross_slot.lattice_centre()
        relative_position[:,1] = 0
        
        # calculate the torque
        torque = np.cross(relative_position, traction)
        return torque
    
    def particle_projected_area(self, direction):
        """Calculates the projected area of the particle.

        Args
        ----
        direction : str
            The direction in which to project the particle.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the projected area at each timestep.
        """
        particle_vtks = self.read_particle_vtks()
        projected_areas = []

        for timestep, vtk in particle_vtks.items():
            try:
                # project the particle mesh onto the plane defined by the direction
                projected_mesh = vtk.project_points_to_plane(
                    origin=self.cross_slot.lattice_centre(),
                    normal=direction,
                )

                particle_area = projected_mesh.area/2
                projected_areas.append({'time': timestep, 'projected_area': particle_area})
            except Exception as e:
                print(f"Error processing timestep {timestep}: {e}")

        projected_areas_df = pd.DataFrame(projected_areas)

        if self.junction_only:
            return self.filter_df_by_junction(projected_areas_df)
        else:
            return projected_areas_df
            
    def dissipation(self, index: float) -> dict:
        """Calculates the fluid dissipation in the particle volume.

        Args
        ----
        index : float
            The index value for the threshold filter.

        Returns
        -------
        dict
            Dictionary where keys are timesteps and values are dissipation values.
        """
        def calculate_viscous_stress_tensor(vtk: pv.PolyData):
            tauLoc = vtk.point_data["tauLoc"] # tau is spatially varying
            dynamic_viscosity = (tauLoc - 0.5) / 3

            # Compute the gradient of the velocity field
            gradients = vtk.compute_derivative(scalars="velocityVector")
            velocity_gradients = gradients.point_data['gradient'].reshape(-1, 3, 3)

            # Calculate viscous stress tensor using vectorized operations
            viscous_stress_tensor = dynamic_viscosity[:, None, None] * (velocity_gradients + velocity_gradients.transpose(0, 2, 1))
            
            return viscous_stress_tensor
        
        def calculate_strain_rate_tensor(vtk: pv.PolyData):
            # Compute the gradient of the velocity field
            gradients = vtk.compute_derivative(scalars="velocityVector")
            velocity_gradients = gradients.point_data['gradient'].reshape(-1, 3, 3)

            # Calculate the strain rate tensor as the symmetric part of the velocity gradient
            strain_rate_tensor = 0.5 * (velocity_gradients + velocity_gradients.transpose(0, 2, 1))

            return strain_rate_tensor
        
        def compute_viscous_dissipation(viscous_stress_tensor: np.array, strain_rate_tensor: np.array):
            # Calculate the viscous dissipation as the dot product of the stress tensor and the strain rate tensor
            viscous_dissipation = np.einsum('ijk,ijk->i', viscous_stress_tensor, strain_rate_tensor)

            return viscous_dissipation
        
        def compute_viscous_dissipation_from_stress(vtk: pv.PolyData):
            viscous_stress_tensor = vtk.point_data["viscousStressTensor"].reshape(-1, 3, 3)
            tauLoc = vtk.point_data["tauLoc"] # tau is spatially varying
            dynamic_viscosity = (tauLoc - 0.5) / 3
            contracted_stress_tensor = np.einsum('ijk,ijk->i', viscous_stress_tensor, viscous_stress_tensor)
            viscous_dissipation = contracted_stress_tensor / (2*dynamic_viscosity)
            return viscous_dissipation
        
        fluid_vtks = self.read_local_fluid_vtks()
        dissipation_timeseries = {}
        for timestep, vtk in fluid_vtks.items():
            try:
                fluid_inside_particle = vtk.threshold(
                    value=index, 
                    scalars='index',
                    preference='point'
                )
                # viscous_stress_tensor = calculate_viscous_stress_tensor(fluid_inside_particle)
                # strain_rate_tensor = calculate_strain_rate_tensor(fluid_inside_particle)
                # viscous_dissipation = compute_viscous_dissipation(viscous_stress_tensor, strain_rate_tensor)
                viscous_dissipation = compute_viscous_dissipation_from_stress(fluid_inside_particle)

                dissipation_timeseries[timestep] = viscous_dissipation.sum() # assuming dV=1

            except Exception as e:
                print(f"Error processing timestep {timestep}: {e}")
        
        return dissipation_timeseries
    
    def compute_inertia_tensor(self, vtk: pv.PolyData):
        """Computes the inertia tensor for the particle.

        Args
        ----
        vtk : pv.PolyData
            The VTK object representing the particle mesh.

        Returns
        -------
        np.ndarray
            The inertia tensor as a 3x3 numpy array.
        """
        # Calculate the inertia tensor
        inertia_tensor = vtk.compute_inertia_tensor()
        return inertia_tensor
        

    def inertia_tensor_timeseries(self):
        """Computes the inertia tensor timeseries for the particle.

        Returns
        -------
        dict
            Dictionary where keys are timesteps and values are inertia tensors.
        """
        # Calculate the inertia tensor for the particle
        particle_vtks = self.read_particle_vtks()
        inertia_tensors = {}

        for timestep, vtk in particle_vtks.items():
            try:
                inertia_tensor = vtk.compute_inertia_tensor()
                inertia_tensors[timestep] = inertia_tensor
            except Exception as e:
                print(f"Error processing timestep {timestep}: {e}")
        
        return inertia_tensors
    
    def thickness_span_cytovale(self) -> pd.DataFrame:
        """Computes the thickness span of the particle.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the thickness span at each timestep.
        """
        particle_vtks = self.read_particle_vtks()

        areas = []

        # Sort the dictionary by timestep
        particle_vtks = dict(sorted(particle_vtks.items()))

        for timestep, vtk in particle_vtks.items():
            try:
                # project the particle mesh onto the plane defined by the direction
                projected_mesh = vtk.project_points_to_plane(
                    origin=self.cross_slot.lattice_centre(),
                    normal=[0,0,1], # z direction
                )


                # the thickness span is the average thickness of the projected
                # particle mesh in the x direction, over a span of half the 
                # particle radius from the particle centroid in the positive and
                # negative y direction.

                span = self.cross_slot.xml_data.radius/2
                centroid = vtk.center
                y_min, y_max = centroid[1] - span, centroid[1] + span

                # clip the mesh to the span
                clipped_mesh = projected_mesh.clip_box([-np.inf, np.inf, y_min, y_max, -np.inf, np.inf], invert=False)

                # calculate the clipped area
                clipped_area = clipped_mesh.area
                areas.append({'time': timestep, 'area': clipped_area})


            except Exception as e:
                print(f"Error processing timestep {timestep}: {e}")

        if self.junction_only:
            return self.filter_df_by_junction(pd.DataFrame(areas))
        else:
            return pd.DataFrame(areas)

    def max_stretch_plane(self, plane):
        """
        Calculates the maximum stretch in the specified plane.

        Args
        ----
        plane : str
            The plane in which to calculate the maximum stretch ('xy', 'yz', 'xz').

        Returns
        -------
        dict
            Dictionary where keys are timesteps and values are maximum stretch distances.
        """
        
        def max_distances(points, plane):
            """
            Calculates the maximum distance in the specified plane.

            Args
            ----
            points : np.ndarray
                Array of points representing the particle mesh.
            plane : str
                The plane in which to calculate distances ('xy', 'yz', 'xz').

            Returns
            -------
            float
                The maximum distance in the specified plane.
            """
            if plane not in ["xy", "yz", "xz"]:
                raise ValueError(f"Invalid plane '{plane}'. Choose from 'xy', 'yz', or 'xz'.")

            # Extract coordinates based on the plane
            if plane == "xy":
                coord1, coord2 = points[:, 0], points[:, 1]
            elif plane == "yz":
                coord1, coord2 = points[:, 1], points[:, 2]
            elif plane == "xz":
                coord1, coord2 = points[:, 0], points[:, 2]

            # Compute all pairwise squared distances using broadcasting
            d1 = coord1[:, np.newaxis] - coord1
            d2 = coord2[:, np.newaxis] - coord2
            distances_sq = d1**2 + d2**2

            # Find the maximum squared distance and take its square root
            max_distance = np.sqrt(np.max(distances_sq))
            return max_distance

        def process_vtk(timestep, vtk, plane):
            """
            Processes a VTK file to compute pairwise distance between all points.

            Args
            ----
            timestep : int
                The current timestep.
            vtk : pv.PolyData
                The VTK object representing the particle mesh.
            plane : str
                The plane in which to calculate distances ('xy', 'yz', 'xz').

            Returns
            -------
            tuple
                A tuple containing the timestep and the calculated distance.
            """
            points = vtk.points
            distance = max_distances(points, plane=plane)
            return timestep, distance
        
        vtks = self.read_particle_vtks()
        
        vtks_df = pd.DataFrame(columns=["time", "vtk"])
        vtks_df["time"] = vtks.keys()
        vtks_df["vtk"] = vtks.values()
        if self.junction_only:
            vtks_junction = self.filter_df_by_junction(vtks_df)
        else:
            vtks_junction = vtks_df

        vtks = vtks_junction["vtk"].to_dict()

        distances = {}

        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_vtk, timestep, vtk, plane) 
                       for timestep, vtk in vtks.items()]
            
            for future in futures:
                result = future.result()
                if result is not None:
                    timestep, distance = result
                    distances[timestep] = distance

        # Sort distances by timestep
        distances = {k: v for k, v in sorted(distances.items(), key=lambda item: item[0])}
        return distances