from ..data_classes.trajectory import Trajectory
from ..data_classes.sim_input import CrossSlot
from ..data_classes.simulation_crash_check import simulation_crash_check
import numpy as np
import pyvista as pv
import os 
import pandas as pd

class TrajectoryGlobalMetrics:
    def __init__(self, trajectory: Trajectory):
        self.trajectory = trajectory

    def dump(self):
        methods_output = {}
        for method_name in dir(self):
            # skip this class
            if method_name == "dump":
                continue

            # Check if it's a callable method and not a built-in one
            if callable(getattr(self, method_name)) and not method_name.startswith("__"):
                methods_output[method_name] = getattr(self, method_name)()
        return methods_output

    def peak_positions(self):
        coordinates = self.trajectory.normalised_coordinates
        x_diffs = np.diff(coordinates[:, 0])
        turning_points = np.where(x_diffs[:-1] * x_diffs[1:] < 0)[0]
        x_turn = coordinates[turning_points+1, 0]
        y_turn = coordinates[turning_points+1, 1]
        z_turn = coordinates[turning_points+1, 2]
        return np.array([x_turn, y_turn, z_turn]).T
    
    def peak_times(self):
        coordinates = self.trajectory.normalised_coordinates
        x_diffs = np.diff(coordinates[:, 0])
        turning_points = np.where(x_diffs[:-1] * x_diffs[1:] < 0)[0]
        return self.trajectory.times[turning_points+1]

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
        junction_times = self.trajectory.times
        if len(junction_times) < 2:
            return None
        else:
            res_time = junction_times[-1] - junction_times[0]
            return res_time
        
    def PQ345(self):
        "function to calculate the PQ_345 parameter"
        peaks = self.peak_positions()
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
    
    # The following methods all rely on the TrajectoryLocalMetrics class
    def total_revolutions(self):
        "calculates the total number of revolutions in terms of degrees, allowing for incomplete revolutions."

        # get the cumulative angular displacement
        cumulative_angular_displacement = TrajectoryLocalMetrics(self.trajectory).cumulative_angular_displacement()

        # get the total number of revolutions
        total_revolutions = cumulative_angular_displacement[-1] / np.pi

        return total_revolutions
    
    def average_angular_velocity(self):
        "calculates the average angular velocity in terms of radians per timestep"

        # get the angular velocity
        # the reference point needs to be the cross-slot centre in original coordiantes
        centre = self.trajectory.simulation_metrics.cross_slot_data.centre
        angular_velocity = TrajectoryLocalMetrics(self.trajectory).angular_velocity(reference_point=centre)

        # get the average angular velocity
        average_angular_velocity = np.mean(angular_velocity)

        return average_angular_velocity
    
    def average_orbital_radius(self):
        "calculates the average orbital radius in terms of the channel width"

        # get the orbital radius
        orbital_radius = TrajectoryLocalMetrics(self.trajectory).orbital_radius()

        # get the average orbital radius
        average_orbital_radius = np.mean(orbital_radius)

        return average_orbital_radius


class TrajectoryLocalMetrics:
    def __init__(self, trajectory: Trajectory):
        self.trajectory = trajectory
        self.coordinates = trajectory.coordinates
        self.velocity = trajectory.velocity
        self.times = trajectory.times
        self.cross_slot = trajectory.simulation_metrics.cross_slot_data

    def read_fluid_vtk(self, filepath: str) -> pv.PolyData:
        """
        reads the fluid vtk file and returns the fluid data as a
        pyvista object
        """
        
        # read the fluid velocity from the fluid vtk file:
        fluid_data = pv.read(filepath)

        return fluid_data

    def angular_velocity(self, reference_point):
        # normal vector
        n_hat = np.array([0.0, 1.0, 0.0])

        # reference point
        r_ref = reference_point

        path = self.trajectory.coordinates
        velocity = self.trajectory.velocity

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
        trajectory = self.trajectory.normalised_coordinates

        x = trajectory[:,0]
        z = trajectory[:,2]

        r = np.sqrt(x**2 + z**2)
        return r
    
    def cumulative_angular_displacement(self):
        "calculates the angular "
        trajectory = self.trajectory.normalised_coordinates

        # we only care about the angular displacement around the y axis:
        x = trajectory[:,0]
        z = trajectory[:,2]

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
    
    def orbital_radius_gradient(self):
        times = self.trajectory.times
        advection_time = self.trajectory.simulation_metrics.advection_time()
        scaled_time = (times-times[0])/advection_time

        normalised_coordinates = self.trajectory.normalised_coordinates
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
        
    def unpeterbed_fluid_velocity_over_trajectory(self):
        """
        calculates the difference between the fluid velocity and the 
        particle velocity at each point on the trajectory. 
        """

        # get the fluid data
        fluid_location = "/home/calum/biofm_projects/PAPERS/ViscoelasticParticle/fluidVTK/junction.vtk"

        # If the velocity has been calculated already, use that.
        if self.trajectory.junction_only:
            interpolated_velocity_filepath = os.path.join(
                self.trajectory.filepath, 'unpeterbed_interpolated_velocity_junction.npy'
            )
        else:
            interpolated_velocity_filepath = os.path.join(
                self.trajectory.filepath, 'unpeterbed_interpolated_velocity_all.npy'
            )

        if os.path.exists(interpolated_velocity_filepath):
            velocity = np.load(interpolated_velocity_filepath)
            return velocity
        
        # If the velocity has not been calculated, calculate it now.

        # start with the particle trajectory:
        trajectory = self.trajectory.coordinates
        times = self.trajectory.times

        # load the fluid data
        fluid_data = self.read_fluid_vtk(fluid_location)

        # Loop over the trajectory timesteps. For each timestep, load
        # the fluid data and interpolate the fluid velocity at the
        # particle position. Then calculate the difference between the
        # fluid velocity and the particle velocity.
        results = np.array([])
        for i, time in enumerate(times):
            
            print('processing timestep', time, 'of', times[-1])

            # get the particle position and convert to pv datatype
            particle_position = trajectory[i,:]
            particle_position = pv.PolyData(particle_position)
            fluid_velocity = particle_position.interpolate(fluid_data)
            fluid_velocity_array = np.array(fluid_velocity.point_data['velocityVector'])
            results = np.append(results,fluid_velocity_array)
        
        results = np.reshape(results, (len(times),3))
        # Save the numpy array
        np.save(interpolated_velocity_filepath, results)

        return results
    
    def local_fluid_velocity_over_trajectory(self):
        """
        calculates the difference between the fluid velocity and the 
        particle velocity at each point on the trajectory. 
        """

        # get the fluid data
        fluid_folder = os.path.join(
            self.trajectory.filepath, 'VTKLocalFluid'
        )

        # If the velocity has been calculated already, use that.
        if self.trajectory.junction_only:
            interpolated_velocity_filepath = os.path.join(
                fluid_folder, 'interpolated_velocity_junction.npy'
            )
        else:
            interpolated_velocity_filepath = os.path.join(
                fluid_folder, 'interpolated_velocity_all.npy'
            )

        if os.path.exists(interpolated_velocity_filepath):
            velocity = np.load(interpolated_velocity_filepath)
            return velocity
        
        # If the velocity has not been calculated, calculate it now.

        # start with the particle trajectory:
        trajectory = self.trajectory.coordinates
        times = self.trajectory.times

        # Loop over the trajectory timesteps. For each timestep, load
        # the fluid data and interpolate the fluid velocity at the
        # particle position. Then calculate the difference between the
        # fluid velocity and the particle velocity.
        results = np.array([])
        for i, time in enumerate(times):
            
            print('processing timestep', time, 'of', times[-1])
            fluid_vtk_path = os.path.join(
                fluid_folder, f"localFluid_t{time}.vtr")
            fluid_data = self.read_fluid_vtk(fluid_vtk_path)

            # get the particle position and convert to pv datatype
            particle_position = trajectory[i,:]
            particle_position = pv.PolyData(particle_position)
            fluid_velocity = particle_position.interpolate(fluid_data)
            fluid_velocity_array = np.array(fluid_velocity.point_data['velocityVector'])
            results = np.append(results,fluid_velocity_array)
        
        results = np.reshape(results, (len(times),3))
        # Save the numpy array
        np.save(interpolated_velocity_filepath, results)

        return results

    
    def vector_radial_coordinates(self, cartesian_vector):
        """
        converts a vector at each point on the trajectory to 
        radial, tangential and axial coordinates
        """
        # get the particle position
        particle_position = self.trajectory.coordinates

        # define the particle displacement vector from cross slot centre
        centre = self.cross_slot.centre
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
        velocity = self.trajectory.velocity

        return self.vector_radial_coordinates(velocity)

        
    
    def fluid_radial_velocity(self):
        """
        calculates the fluid velocity in radial coordinates
        """
        fluid_velocity = self.unpeterbed_fluid_velocity_over_trajectory()

        return self.vector_radial_coordinates(fluid_velocity)    












