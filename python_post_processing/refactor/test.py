from read_simulation_output import Trajectory
import matplotlib.pyplot as plt
import os

filepath = '/home/calum/calum-ws/SIMULATIONS/viscoelastic_eqinit/simulations_eqInit'

for filename in os.listdir(filepath):
    if filename.endswith('.json'):
        continue

    sim_path = os.path.join(filepath, filename)
    traj = Trajectory(sim_path)
    params = traj.input_parameters
    normalised_coordinates = traj.return_coordinates(normalise=True)

    plt.plot(normalised_coordinates[:, 0], normalised_coordinates[:, 1], label=f'{params.viscosity_ratio()}')

plt.legend()
plt.show()

