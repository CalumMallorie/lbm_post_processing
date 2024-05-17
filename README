# Python Post-Processing of LBM-IBM-FEM Cross-Slot Simulations

This project provides tools for post-processing simulation data of deformable particle trajectories through a cross-slot channel. The simulations model the cells as viscoelastic capsules with viscosity contrast, coupled to a lattice-Boltzmann fluid simulation. The project includes several Python modules for reading, processing, and analyzing simulation data.

## Modules

### my_xml_tools.py

This module reads and parses XML files to extract simulation parameters.

**Classes:**
- `ReadSimXML`: Reads and parses XML files within a specified directory.

### read_input_parameters.py

This module calculates derived parameters for cross-slot simulations using data extracted from XML files.

**Classes:**
- `CrossSlotParameters`: Uses `CrossSlotXML` to extract and calculate simulation parameters.

### helper_functions.py

This module provides utility functions for parameter expansion, directory searching, and vector transformations.

**Functions:**
- `expand_parameters(parameters)`: Expands parameter lists into separate dictionaries.
- `search_sim_directory(simulations_location, parameters)`: Searches directories matching simulation parameters.
- `simulation_crash_check(filepath)`: Checks for simulation crashes.
- `vector_radial_coordinates(cartesian_vector, trajectory)`: Converts vectors to radial, tangential, and axial coordinates.

### process_particle.py

This module processes global particle data as trajectory timeseries, calculating various particle metrics.

**Classes:**
- `ProcessParticle`: Processes particle data and calculates metrics such as shape factor, Taylor deformation, and more.

### process_trajectory.py

This module handles particle trajectory data, filtering it by junction, and calculating various metrics related to the trajectory.

**Classes:**
- `Trajectory`: Handles and analyzes particle trajectory data.
- `ProcessTrajectory`: Processes and analyzes specific metrics of particle trajectory data.


## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/your-repo/python_post_processing.git
    ```
2. Navigate to the project directory:
    ```sh
    cd python_post_processing
    ```
3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```