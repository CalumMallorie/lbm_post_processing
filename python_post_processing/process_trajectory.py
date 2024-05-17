from .read_input_parameters import CrossSlotParameters
import numpy as np
import pandas as pd
import os
import pyvista as pv
from scipy.optimize import curve_fit

class Trajectory:
    def __init__(self, filepath: str, junction_only: bool = True):
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
        return self.datafile
    
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
        """
        Exports the trajectory to a CSV file.
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
    def __init__(self, filepath, junction_only: bool = True):
        self.filepath = filepath
        self.junction_only = junction_only
        self.trajectory = Trajectory(filepath, junction_only=junction_only)
        self.input_parameters = CrossSlotParameters(filepath)

    def peak_positions(self, normalise: bool = True):
        coordinates = self.trajectory.return_coordinates(normalise=normalise)
        x_diffs = np.diff(coordinates[:, 0])
        turning_points = np.where(x_diffs[:-1] * x_diffs[1:] < 0)[0]
        x_turn = coordinates[turning_points+1, 0]
        y_turn = coordinates[turning_points+1, 1]
        z_turn = coordinates[turning_points+1, 2]
        return np.array([x_turn, y_turn, z_turn]).T
    
    def peak_times(self, normalise: bool = False):
        coordinates = self.trajectory.return_coordinates(normalise=True)
        x_diffs = np.diff(coordinates[:, 0])
        turning_points = np.where(x_diffs[:-1] * x_diffs[1:] < 0)[0]
        times = self.trajectory.return_times(normalise=normalise)
        return times[turning_points+1]

    def num_peaks_xy(self):
        peaks = self.peak_positions()
        num_peaks = peaks.shape[0]
        return num_peaks
    
    def peak_amplitudes_xy(self):
        peaks = self.peak_positions()
        x_deviations = np.abs(peaks[:,0])  # Compute the absolute values of the x deviations using NumPy
        amp_peaks = x_deviations
        return amp_peaks
    
    def residence_time(self):
        junction_times = self.trajectory.return_times(normalise=True)
        if len(junction_times) < 2:
            return None
        else:
            res_time = junction_times[-1] - junction_times[0]
            return res_time
        
    def PQ345(self, normalise=True):
        "function to calculate the PQ_345 parameter"
        peaks = self.peak_positions(normalise=normalise)
        if np.shape(peaks)[0]<5:
            return None
        else:
            P3 = peaks[2,0]
            P4 = peaks[3,0]
            P5 = peaks[4,0]

            d34 = abs(P3-P4)
            d45 = abs(P4-P5)

            PQ345 = d34 + d45

            return PQ345
    
    def cumulative_angular_displacement(self):
        "calculates the angular "
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
    def total_revolutions(self):
        "calculates the total number of revolutions in terms of degrees, allowing for incomplete revolutions."

        # get the cumulative angular displacement
        cumulative_angular_displacement = self.cumulative_angular_displacement()

        # get the total number of revolutions
        total_revolutions = cumulative_angular_displacement[-1] / np.pi

        return total_revolutions
    
    def angular_velocity(self):
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
    
    def orbital_radius(self):
        trajectory = self.trajectory.return_coordinates(normalise=False)
        x = trajectory[:,0]
        z = trajectory[:,2]

        centre = self.input_parameters.lattice_centre()
        x -= centre[0]
        z -= centre[2]

        r = np.sqrt(x**2 + z**2)
        return r
    
    def orbital_radius_gradient(self):
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
    
    def vector_radial_coordinates(self, cartesian_vector):
        """
        converts a vector at each point on the trajectory to 
        radial, tangential and axial coordinates
        """
        # get the particle position
        particle_position = self.trajectory.return_coordinates(normalise=False)

        # define the particle displacement vector from cross slot centre
        centre = self.input_parameters.lattice_centre()
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

        # Project the total velocity forces onto the new basis vectors
        radial_velocity = np.einsum(
            'ij,ij->i', cartesian_vector, radial_unit_vector
        )
        tangential_velocity = np.einsum(
            'ij,ij->i', cartesian_vector, tangential_unit_vector
        )
        axial_velocity = np.einsum(
            'ij,ij->i', cartesian_vector, y_axis_unit_vector_2d
        )

        radial_vector = np.array([
            radial_velocity, tangential_velocity, axial_velocity
        ]).T

        return radial_vector
    
    def particle_radial_velocity(self):
        """
        calculates the particle velocity in radial coordinates
        """
        velocity = self.trajectory.return_velocity()

        return self.vector_radial_coordinates(velocity)
    
    def unpeterbed_fluid_radial_velocity(self, fluid_location: str):
        """
        calculates the fluid velocity in radial coordinates
        """
        vtk = pv.read(fluid_location)
        fluid_velocity = self.unpeterbed_fluid_velocity_over_trajectory(vtk)

        return self.vector_radial_coordinates(fluid_velocity)    
    
    # def unpeterbed_fluid_velocity_over_trajectory(self, fluid_location: str):
    #     """
    #     calculates the difference between the fluid velocity and the 
    #     particle velocity at each point on the trajectory. 
    #     """

    #     # If the velocity has been calculated already, use that.
    #     if self.junction_only:
    #         interpolated_velocity_filepath = os.path.join(
    #             self.trajectory.filepath, 'unpeterbed_interpolated_velocity_junction.npy'
    #         )
    #     else:
    #         interpolated_velocity_filepath = os.path.join(
    #             self.trajectory.filepath, 'unpeterbed_interpolated_velocity_all.npy'
    #         )

    #     if os.path.exists(interpolated_velocity_filepath):
    #         velocity = np.load(interpolated_velocity_filepath)
    #         return velocity
        
    #     # If the velocity has not been calculated, calculate it now.

    #     # start with the particle trajectory:
    #     trajectory = self.trajectory.return_coordinates(normalise=False)
    #     times = self.trajectory.return_times(normalise=False)

    #     # load the fluid data
    #     fluid_data = self.read_fluid_vtk(fluid_location)

    #     # Loop over the trajectory timesteps. For each timestep, load
    #     # the fluid data and interpolate the fluid velocity at the
    #     # particle position. Then calculate the difference between the
    #     # fluid velocity and the particle velocity.
    #     results = np.array([])
    #     for i, time in enumerate(times):
            
    #         print('processing timestep', time, 'of', times[-1])

    #         # get the particle position and convert to pv datatype
    #         particle_position = trajectory[i,:]
    #         particle_position = pv.PolyData(particle_position)
    #         fluid_velocity = particle_position.interpolate(fluid_data)
    #         fluid_velocity_array = np.array(fluid_velocity.point_data['velocityVector'])
    #         results = np.append(results,fluid_velocity_array)
        
    #     results = np.reshape(results, (len(times),3))
    #     # Save the numpy array
    #     np.save(interpolated_velocity_filepath, results)

    #     return results

    def unpeterbed_fluid_velocity_over_trajectory(self, fluid_data: pv.PolyData):
        """
        Calculates the difference between the fluid velocity and the 
        particle velocity at each point on the trajectory using vectorized
        operations for improved performance.
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
        """
        Returns the parameters of a damped sine wave fit to the trajectory.
        returns:
        popt: np.ndarray, parameters of the damped sine wave fit, [A, omega, gamma, phi, c]
        """
        def damped_wave(t, A, omega, gamma, phi, c) -> np.ndarray:
            """
            Returns a damped sine wave function.
            args:
            A: float, amplitude
            omega: float, angular frequency
            gamma: float, damping coefficient
            phi: float, phase
            c: float, offset
            """
            damping = np.exp(gamma * t)
            oscillation = np.cos((omega + phi*t) * t)
            drift = c
            return A * damping * oscillation + drift
        
        # Fit a damped sine wave to the trajectory.

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