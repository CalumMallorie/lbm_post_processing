"""
process_trajectory.py
======================

This module provides classes for processing particle trajectory data from
simulations. It includes functionality for reading trajectory data,
filtering it by junction, and calculating various metrics related to the
trajectory.

Classes
-------
Trajectory
    Class for handling and analyzing particle trajectory data.

ProcessTrajectory
    Class for processing and analyzing specific metrics of particle 
    trajectory data.

Dependencies
------------
- numpy
- pandas
- os
- pyvista
- scipy.optimize.curve_fit

Example Usage
-------------
    from process_trajectory import Trajectory, ProcessTrajectory
    
    # Initialize a Trajectory object with the path to the simulation directory
    traj = Trajectory('path/to/simulation')
    
    # Initialize a ProcessTrajectory object
    proc_traj = ProcessTrajectory('path/to/simulation')
    
    # Get peak positions
    peaks = proc_traj.peak_positions()
"""

import numpy as np
import pandas as pd
import os
import pyvista as pv
from scipy.optimize import curve_fit

from .read_input_parameters import CrossSlotParameters
from .helper_functions import vector_radial_coordinates

class Trajectory:
    """
    Class for handling and analyzing particle trajectory data.

    Attributes
    ----------
    filepath : str
        Path to the simulation directory.
    input_parameters : CrossSlotParameters
        Instance of CrossSlotParameters for extracting simulation parameters.
    datafile : pd.DataFrame
        DataFrame containing the trajectory data.
    
    Methods
    -------
    __init__(self, filepath: str, junction_only: bool = True)
        Initializes the Trajectory class with the path to the simulation directory.
    read_particle_datafile(self, filepath: str) -> pd.DataFrame
        Reads the particle data file and returns it as a pandas DataFrame.
    return_datafile(self) -> pd.DataFrame
        Returns the trajectory data as a pandas DataFrame.
    filter_by_junction(self, df: pd.DataFrame) -> pd.DataFrame
        Filters the trajectory data by junction.
    return_coordinates(self, normalise: bool = False) -> np.ndarray
        Returns the particle coordinates.
    return_velocity(self) -> np.ndarray
        Returns the particle velocities.
    return_times(self, normalise: bool = False) -> np.ndarray
        Returns the times from the trajectory data.
    export_csv(self, save_path: str, normalise: bool = False)
        Exports the trajectory data to a CSV file.
    """
    def __init__(self, filepath: str, junction_only: bool = True):
        """Initialize the Trajectory class with the path to the simulation directory.
        
        Args
        ----
        filepath : str
            The path to the simulation directory.
        junction_only : bool, optional
            Whether to process only junction data (default is True).
        """
        self.filepath = filepath
        self.input_parameters = CrossSlotParameters(filepath)
        if junction_only:
            unfiltered_datafile = self.read_particle_datafile(filepath)
            self.datafile = self.filter_by_junction(unfiltered_datafile)
        else:
            self.datafile = self.read_particle_datafile(filepath)
        
        # check time entries in datafile are monotonic
        if not self.datafile['time'].is_monotonic_increasing:
            raise ValueError(f"Time entries in datafile {filepath} are not monotonic.")
    
    def read_particle_datafile(self, filepath: str):
        """Reads the particle data file and returns it as a pandas DataFrame.
        
        Args
        ----
        filepath : str
            The path to the directory containing the 'Particles' folder.
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing the particle data.
        """
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
    
    def return_datafile(self):
        """Returns the trajectory data as a pandas DataFrame.
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing the trajectory data.
        """
        return self.datafile
    
    def filter_by_junction(self, df: pd.DataFrame):
        """Filters the trajectory data by junction.
        
        Args
        ----
        df : pd.DataFrame
            DataFrame containing the trajectory data.
        
        Returns
        -------
        pd.DataFrame
            Filtered DataFrame.
        """
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
        """Returns the particle coordinates.
        
        Args
        ----
        normalise : bool, optional
            Whether to normalise the coordinates (default is False).
        
        Returns
        -------
        np.ndarray
            Array containing the particle coordinates.
        """
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
        """Returns the particle velocities.
        
        Returns
        -------
        np.ndarray
            Array containing the particle velocities.
        """
        return np.array(self.datafile.loc[:, ['v_x', 'v_y', 'v_z']])
    
    def return_times(self, normalise: bool = False):
        """Returns the times from the trajectory data.
        
        Args
        ----
        normalise : bool, optional
            Whether to normalise the times (default is False).
        
        Returns
        -------
        np.ndarray
            Array containing the times.
        """
        if normalise:
            try:
                times = np.array(self.datafile['time']) 
                return (times - times[0])/self.input_parameters.advection_time()
            except IndexError:
                raise IndexError(f"No times found inside the junction region for simulation {os.path.basename(self.filepath)}")
            except Exception as e:
                print(f"An unexpected error occurred: {str(e)}")
        else:
            return np.array(self.datafile['time'])
        
    def export_csv(self, save_path: str, normalise: bool = False):
        """Exports the trajectory data to a CSV file.
        
        Args
        ----
        save_path : str
            The path where the CSV file will be saved.
        normalise : bool, optional
            Whether to normalise the data before exporting (default is False).
        """
        save_dir = os.path.dirname(save_path)
        # Create the save directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        data = self.return_coordinates(normalise=normalise)
        times = self.return_times(normalise=normalise)
        timeseries = np.column_stack((times, data))

        with open(save_path, 'w') as f:
            f.write("time,x,y,z\n")
            for row in timeseries:
                f.write(f"{row[0]},{row[1]},{row[2]},{row[3]}\n")

class ProcessTrajectory:
    """
    Class for processing and analyzing specific metrics of particle trajectory data.

    Attributes
    ----------
    filepath : str
        Path to the simulation directory.
    junction_only : bool
        Whether to process only junction data.
    trajectory : Trajectory
        Instance of the Trajectory class for handling trajectory data.
    input_parameters : CrossSlotParameters
        Instance of CrossSlotParameters for extracting simulation parameters.
    
    Methods
    -------
    __init__(self, filepath: str, junction_only: bool = True)
        Initializes the ProcessTrajectory class with the path to the simulation directory.
    peak_positions(self, normalise: bool = True) -> np.ndarray
        Returns the positions of the peaks in the trajectory.
    peak_times(self, normalise: bool = False) -> np.ndarray
        Returns the times at which the peaks occur.
    num_peaks_xy(self) -> int
        Returns the number of peaks in the xy-plane.
    peak_amplitudes_xy(self) -> np.ndarray
        Returns the amplitudes of the peaks in the xy-plane.
    residence_time(self) -> float
        Returns the residence time of the particle in the junction.
    PQ345(self, normalise: bool = True) -> float
        Calculates the PQ345 parameter.
    cumulative_angular_displacement(self) -> np.ndarray
        Calculates the cumulative angular displacement.
    total_revolutions(self) -> float
        Calculates the total number of revolutions.
    angular_velocity(self) -> np.ndarray
        Calculates the angular velocity.
    orbital_radius(self) -> np.ndarray
        Calculates the orbital radius.
    orbital_radius_gradient(self) -> np.ndarray
        Calculates the gradient of the orbital radius.
    vector_radial_coordinates(self, cartesian_vector: np.ndarray) -> np.ndarray
        Converts a vector at each point on the trajectory to radial, tangential, and axial coordinates.
    particle_radial_velocity(self) -> np.ndarray
        Calculates the particle velocity in radial coordinates.
    unpeterbed_fluid_radial_velocity(self, fluid_location: str) -> np.ndarray
        Calculates the fluid velocity in radial coordinates.
    unpeterbed_fluid_velocity_over_trajectory(self, fluid_data: pv.PolyData) -> np.ndarray
        Calculates the difference between the fluid velocity and the particle velocity at each point on the trajectory.
    fit_damped_sine(self) -> np.ndarray
        Returns the parameters of a damped sine wave fit to the trajectory.
    """
    def __init__(self, filepath, junction_only: bool = True):
        """Initialize the ProcessTrajectory class with the path to the simulation directory.
        
        Args
        ----
        filepath : str
            The path to the simulation directory.
        junction_only : bool, optional
            Whether to process only junction data (default is True).
        """
        self.filepath = filepath
        self.junction_only = junction_only
        self.trajectory = Trajectory(filepath, junction_only=junction_only)
        self.input_parameters = CrossSlotParameters(filepath)

    def peak_positions(self, normalise: bool = True) -> np.ndarray:
        """Returns the positions of the peaks in the trajectory.
        
        Args
        ----
        normalise : bool, optional
            Whether to normalise the coordinates (default is True).
        
        Returns
        -------
        np.ndarray
            Array containing the positions of the peaks.
        """
        coordinates = self.trajectory.return_coordinates(normalise=normalise)
        x_diffs = np.diff(coordinates[:, 0])
        turning_points = np.where(x_diffs[:-1] * x_diffs[1:] < 0)[0]
        x_turn = coordinates[turning_points+1, 0]
        y_turn = coordinates[turning_points+1, 1]
        z_turn = coordinates[turning_points+1, 2]
        return np.array([x_turn, y_turn, z_turn]).T
    
    def peak_times(self, normalise: bool = False) -> np.ndarray:
        """Returns the times at which the peaks occur.
        
        Args
        ----
        normalise : bool, optional
            Whether to normalise the times (default is False).
        
        Returns
        -------
        np.ndarray
            Array containing the times of the peaks.
        """
        coordinates = self.trajectory.return_coordinates(normalise=True)
        x_diffs = np.diff(coordinates[:, 0])
        turning_points = np.where(x_diffs[:-1] * x_diffs[1:] < 0)[0]
        times = self.trajectory.return_times(normalise=normalise)
        return times[turning_points+1]

    def num_peaks_xy(self) -> int:
        """Returns the number of peaks in the xy-plane.
        
        Returns
        -------
        int
            Number of peaks in the xy-plane.
        """
        peaks = self.peak_positions()
        num_peaks = peaks.shape[0]
        return num_peaks
    
    def peak_amplitudes_xy(self) -> np.ndarray:
        """Returns the amplitudes of the peaks in the xy-plane.
        
        Returns
        -------
        np.ndarray
            Array containing the amplitudes of the peaks.
        """
        peaks = self.peak_positions()
        x_deviations = np.abs(peaks[:,0])  # Compute the absolute values of the x deviations using NumPy
        amp_peaks = x_deviations
        return amp_peaks
    
    def residence_time(self) -> float:
        """Returns the residence time of the particle in the junction.
        
        Returns
        -------
        float
            Residence time of the particle.
        """
        junction_times = self.trajectory.return_times(normalise=True)
        if len(junction_times) < 2:
            return None
        else:
            res_time = junction_times[-1] - junction_times[0]
            return res_time
        
    def diff345(self, normalise=True) -> float:
        """Calculates the diff345 parameter.
        
        Args
        ----
        normalise : bool, optional
            Whether to normalise the coordinates (default is True).
        
        Returns
        -------
        float
            The diff345 parameter.
        """
        peaks = self.peak_positions(normalise=normalise)
        if np.shape(peaks)[0]<5:
            return None
        else:
            P3 = peaks[2,0]
            P4 = peaks[3,0]
            P5 = peaks[4,0]

            d34 = abs(P3-P4)
            d45 = abs(P4-P5)

            diff345 = d34 + d45

            return diff345
    
    def cumulative_angular_displacement(self) -> np.ndarray:
        """Calculates the cumulative angular displacement.
        
        Returns
        -------
        np.ndarray
            Array containing the cumulative angular displacement.
        """
        coordiantes = self.trajectory.return_coordinates(normalise=True)

        # we only care about the angular displacement around the y axis:
        x = coordiantes[:,0]
        z = coordiantes[:,2]

        # Calculate the angle at each point (negative to match the convention of the angular velocity)
        theta = -np.arctan2(z, x)

        # Compute the difference in angles between successive points
        d_theta = np.diff(theta)

        # Adjust for angle wrapping
        d_theta = np.mod(d_theta + np.pi, 2 * np.pi) - np.pi

        # Cumulative sum of angular changes
        cumulative_theta = np.cumsum(d_theta)
        cumulative_theta = np.insert(cumulative_theta, 0, 0)  # Insert starting angle (0) at the beginning

        return cumulative_theta
    
    # The following methods all rely on the TrajectoryLocalMetrics class
    def total_revolutions(self) -> float:
        """Calculates the total number of revolutions.
        
        Returns
        -------
        float
            Total number of revolutions.
        """
        # get the cumulative angular displacement
        cumulative_angular_displacement = self.cumulative_angular_displacement()

        # get the total number of revolutions
        total_revolutions = cumulative_angular_displacement[-1] / np.pi

        return total_revolutions
    
    def angular_velocity(self) -> np.ndarray:
        """Calculates the angular velocity.
        
        Returns
        -------
        np.ndarray
            Array containing the angular velocity.
        """
        # normal vector is along the y axis (outlet centreline)
        n_hat = np.array([0.0, 1.0, 0.0])

        # reference point
        r_ref = CrossSlotParameters(self.filepath).lattice_centre()

        path = self.trajectory.return_coordinates(normalise=False)
        velocity = self.trajectory.return_velocity()

        # initialise angular velocity
        angular_velocity = np.zeros((velocity.shape[0],1))
        for i in range(path.shape[0]):
            r = path[i,:]
            v = velocity[i]
            r_rel = r - r_ref

            r_perp = r_rel - np.dot(r_rel, n_hat) * n_hat
            v_perp = v - np.dot(v, n_hat) * n_hat

            omega = np.cross(r_perp, v_perp) / np.linalg.norm(r_perp)**2

            angular_velocity[i] = omega[1]

        return angular_velocity
    
    def orbital_radius(self) -> np.ndarray:
        """Calculates the orbital radius.
        
        Returns
        -------
        np.ndarray
            Array containing the orbital radius.
        """
        trajectory = self.trajectory.return_coordinates(normalise=False)
        x = trajectory[:,0]
        z = trajectory[:,2]

        centre = self.input_parameters.lattice_centre()
        x -= centre[0]
        z -= centre[2]

        r = np.sqrt(x**2 + z**2)
        return r
    
    def orbital_radius_gradient(self) -> np.ndarray:
        """Calculates the gradient of the orbital radius.
        
        Returns
        -------
        np.ndarray
            Array containing the gradient of the orbital radius.
        """
        scaled_time = self.trajectory.return_times(normalise=True)

        normalised_coordinates = self.trajectory.return_coordinates(normalise=True)
        orbital_radius = self.orbital_radius()

        # filter the arrays to begin at the point where the particle
        # crosses the centreline for the first time:

        crossing_index = np.where(normalised_coordinates[:,0]>0)[0][0]

        # if crossing_indices.size == 0:
        #     raise ValueError("No crossing point found.")
        
        # crossing_index = crossing_indices[0]
        
        scaled_time = scaled_time[crossing_index:-1]
        orbital_radius = orbital_radius[crossing_index:-1]

        # now calculate the gradient of the orbital radius:
        fit = np.polyfit(scaled_time, orbital_radius, 1)

        return fit
    
    def particle_radial_velocity(self) -> np.ndarray:
        """Calculates the particle velocity in radial coordinates.
        
        Returns
        -------
        np.ndarray
            Array containing the particle radial velocity.
        """
        velocity = self.trajectory.return_velocity()

        return vector_radial_coordinates(velocity, self.trajectory)
    
    def unpeterbed_fluid_radial_velocity(self, fluid_location: str) -> np.ndarray:
        """Calculates the fluid velocity in radial coordinates.
        
        Args
        ----
        fluid_location : str
            Path to the fluid data file.
        
        Returns
        -------
        np.ndarray
            Array containing the fluid radial velocity.
        """
        vtk = pv.read(fluid_location)
        fluid_velocity = self.unpeterbed_fluid_velocity_over_trajectory(vtk)

        return self.vector_radial_coordinates(fluid_velocity)    
    
    def unpeterbed_fluid_velocity_over_trajectory(self, fluid_data: pv.PolyData) -> np.ndarray:
        """Calculates the difference between the fluid velocity and the particle velocity at each point on the trajectory.
        
        Args
        ----
        fluid_data : pv.PolyData
            The PolyData object representing the fluid mesh.
        
        Returns
        -------
        np.ndarray
            Array containing the fluid velocity differences.
        """
        # Retrieve trajectory data
        trajectory = self.trajectory.return_coordinates(normalise=False)

        # Create a PolyData object from the entire trajectory
        trajectory_polydata = pv.PolyData(trajectory)

        # Interpolate the fluid velocity for the entire trajectory
        interpolated_data = trajectory_polydata.interpolate(fluid_data)

        # Extract the fluid velocity data
        results = np.array(interpolated_data.point_data['velocityVector'])

        return results
    
    

    def fit_damped_sine(self) -> np.ndarray:
        """Returns the parameters of a damped sine wave fit to the trajectory.
        
        Returns
        -------
        np.ndarray
            Parameters of the damped sine wave fit.
        """
        def damped_wave(t, A, omega, gamma, phi, c) -> np.ndarray:
            """
            Returns a damped sine wave function.

            Args
            ----
            A: float
                amplitude
            omega: float
                angular frequency
            gamma: float
                damping coefficient
            phi: float
                phase
            c: float
                offset
            """
            damping = np.exp(-gamma * t)
            oscillation = np.cos((omega + phi*t) * t)
            drift = c
            return A * damping * oscillation + drift

        # Extract the time and position data
        t = self.trajectory.return_times(normalise=True)
        t = t - self.peak_times(normalise=True)[0]
        
        y = self.trajectory.return_coordinates(normalise=True)[:, 0]

        # Find the index of the first positive y value
        first_positive_y_index = np.argmax(y > 0)

        # Keep all points after the first positive y value
        t = t[first_positive_y_index:]
        y = y[first_positive_y_index:]

        # Initial guess
        A = 0.2 # Amplitude
        omega = 10 # Frequency
        phi = 0 # Frequency damping
        gamma = 0 # Amplitude damping
        c = 0 # skewing
        p0 = [A, omega, gamma, phi, c]

        # Fit the damped sine wave
        popt, pcov = curve_fit(damped_wave, t, y, p0=p0)

        return popt