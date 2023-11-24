from ..data_classes.trajectory import Trajectory
from ..data_classes.simulation_metrics import SimulationMetrics
from ..data_classes.sim_input import CrossSlot
from ..data_processing.trajectory_metrics import TrajectoryGlobalMetrics, TrajectoryLocalMetrics
from ..data_processing.particle_metrics import ParticleGlobalMetrics
from .tools import search_sim_directory

import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_trajectory(ax:plt.Axes,
                    filepath: str,
                    plane: str, 
                    label:str = None,
                    color=None):
    
    trajectory = Trajectory(filepath)

    x_coords = trajectory.normalised_coordinates[:, 0]
    y_coords = trajectory.normalised_coordinates[:, 1]
    z_coords = trajectory.normalised_coordinates[:, 2]

    if plane == 'xy':
        plt.plot(y_coords, x_coords, label=label, color=color)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.ylim(-0.4,0.4)
        plt.xlim(0,1)
        plt.hlines(0, 0, 1, colors='k', linestyles='dashed', linewidth=0.5)
    elif plane == 'xz':
        plt.plot(x_coords, z_coords, label=label, color=color)
        plt.xlabel('x')
        plt.ylabel('z')
    elif plane == 'yz':
        plt.plot(y_coords, z_coords, label=label, color=color)
        plt.xlabel('y')
        plt.ylabel('z')
    else:
        raise ValueError("The plane must be one of 'xy', 'xz' or 'yz'.")

def plot_multiple_trajectories(ax: plt.Axes, 
                                sim_dirs: list,
                                plane: str, 
                                label_by: str = None, 
                                label_format: str = "{:.2f}", 
                                label_prefix: str = None,
                                colormap='viridis'):
    if not isinstance(sim_dirs, list):
        sim_dirs = [sim_dirs]
    
    dir_label_pairs = []
    for directory in sim_dirs:
        sim_metrics = SimulationMetrics(directory)
        label = getattr(sim_metrics, label_by)() if label_by else None
        if label is not None:
            label = label_format.format(label)
        if label_prefix is not None:
            label = label_prefix + label
        dir_label_pairs.append((directory, label))
    dir_label_pairs.sort(key=lambda pair: float(pair[1].split('=')[1]))

    cmap = cm.get_cmap(colormap, len(dir_label_pairs))
    for i, (directory, label) in enumerate(dir_label_pairs):
        color = cmap(i)
        plot_trajectory(ax, directory, plane=plane, label=label, color=color)
    if label_by:
        ax.legend()

def plot_angular_velocity(filepath: str):
    cross_slot = CrossSlot(filepath)
    trajectory = Trajectory(filepath)
    trajectory_metrics_local = TrajectoryLocalMetrics(trajectory)
    angular_velocity = trajectory_metrics_local.angular_velocity(reference_point=cross_slot.centre)
    plt.plot(trajectory.times, angular_velocity)
    plt.xlabel('time')
    plt.ylabel('angular velocity')

def plot_orbital_radius(filepath: str):
    trajectory = Trajectory(filepath)
    trajectory_metrics_local = TrajectoryLocalMetrics(trajectory)
    orbital_radius = trajectory_metrics_local.orbital_radius()
    plt.plot(trajectory.times, orbital_radius)
    plt.xlabel('time')
    plt.ylabel('orbital radius')

def plot_angular_velocity_against_orbital_radius(filepath: str):#
    cross_slot = CrossSlot(filepath)
    trajectory = Trajectory(filepath)
    trajectory_metrics_local = TrajectoryLocalMetrics(trajectory)
    orbital_radius = trajectory_metrics_local.orbital_radius()
    angular_velocity = trajectory_metrics_local.angular_velocity(reference_point=cross_slot.centre)

    plt.plot(orbital_radius, angular_velocity)
    plt.xlabel('orbital radius')
    plt.ylabel('angular velocity')

def plot_integrated_traction_forces(sim_dir: str, parameters: dict):

    filepath = search_sim_directory(sim_dir, parameters)
    if len(filepath) !=1:
        raise ValueError(f"Expected to find one matching directory, found {len(filepath)}")
    else:
        filepath = filepath[0]

    print(f'plotting integrated traction forces for: {filepath}')
    particle_metrics = ParticleGlobalMetrics(filepath)

    sim_metrics = SimulationMetrics(filepath)
    advection_time = sim_metrics.advection_time()

    scaled_timeseries = particle_metrics.integrated_traction_df['timestep']/advection_time

    plt.plot(scaled_timeseries, particle_metrics.integrated_traction_df['total_traction_magnitude'], label='total traction')
    # plt.plot(scaled_timeseries, particle_metrics.integrated_traction_df['totaldev_force_magnitude'], label='total deviatoric')
    # plt.plot(scaled_timeseries, particle_metrics.integrated_traction_df['total_press_force_mag'], label='total pressure')
    plt.xlabel('t/t_adv')
    plt.ylabel('force')
    plt.legend()

def plot_principal_axes(sim_dir:str, parameters:dict):
    filepath = search_sim_directory(sim_dir, parameters)

    if len(filepath) !=1:
            raise ValueError(f"Expected to find one matching directory, found {len(filepath)}")
    else:
        filepath = filepath[0]

    particle_metrics = ParticleGlobalMetrics(filepath)
    sim_metrics = SimulationMetrics(filepath)

    advection_time = sim_metrics.advection_time()
    time = particle_metrics.integrated_traction_df['timestep']

    scaled_timeseries = time/advection_time

    principal_axes = particle_metrics.principal_axes

    plt.plot(scaled_timeseries, principal_axes['a'], label='a')
    plt.plot(scaled_timeseries, principal_axes['b'], label='b')
    plt.plot(scaled_timeseries, principal_axes['c'], label='c')
    plt.plot(scaled_timeseries, principal_axes['shape_factor'], label='SDI')
    plt.xlabel('t/t_adv')
    plt.ylabel('principal axes length')
    plt.legend()

def plot_shape_factor(sim_dir:str, parameters:dict):
    filepaths = search_sim_directory(sim_dir, parameters)

    for filepath in filepaths:
        particle_metrics = ParticleGlobalMetrics(filepath)
        sim_metrics = SimulationMetrics(filepath)
        viscosity_ratio = sim_metrics.viscosity_ratio()

        advection_time = sim_metrics.advection_time()
        time = particle_metrics.integrated_traction_df['timestep']

        scaled_timeseries = time/advection_time

        principal_axes = particle_metrics.principal_axes

        plt.plot(scaled_timeseries, principal_axes['shape_factor'], label=f'VR={viscosity_ratio}')
        plt.xlabel('t/t_adv')
        plt.ylabel('principal axes length')
        plt.legend()

