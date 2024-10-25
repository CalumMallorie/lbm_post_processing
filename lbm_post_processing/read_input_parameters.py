"""
read_input_parameters.py
=========================

This module provides the `CrossSlotParameters` class for calculating
various derived parameters for cross-slot simulations based on XML
file data. It utilizes the `CrossSlotXML` class for extracting data
from BioFM XML input files.

Classes
-------
CrossSlotParameters
    Calculates derived parameters for cross-slot simulations based on
    XML file data.

CrossSlotXML
    Extracts data from BioFM XML input files using the `ReadSimXML`
    tool.

Dependencies
------------
- numpy
- my_xml_tools.ReadSimXML

Example Usage
-------------
    from read_input_parameters import CrossSlotParameters
    
    # Initialize with path to XML file
    params = CrossSlotParameters('path/to/xml/file')
    
    # Get a dictionary of all parameter values
    param_values = params.dump()
"""

from .my_xml_tools import ReadSimXML

class CrossSlotParameters:
    """Calculate various derived parameters for cross-slot simulations
    based on XML file data.

    This class uses the CrossSlotXML class to extract data from the
    BioFM XML input files.

    Attributes:
        xml_data (CrossSlotXML): Instance of CrossSlotXML for data extraction.
    """
    def __init__(self, filepath):
        """Initialize the class with the path to the simulation directory.

        Args:
            filepath (str): The path to the simulation directory.

        Raises:
            ValueError: If the XML file cannot be read or parsed correctly.
        """
        try:
            self.xml_data = CrossSlotXML(filepath)
        except Exception as e:
            raise ValueError(f"Failed to read or parse the XML file at {filepath}: {e}")

    def dump(self):
        """Returns a dictionary of method outputs.

        This method iterates through all methods of the class and
        collects their outputs in a dictionary.

        Returns:
            dict: Dictionary containing the outputs of all methods.
        """
        methods_output = {}
        for method_name in dir(self):
            if method_name == "dump" or method_name == "matches_parameters":
                continue
            if callable(getattr(self, method_name)) and not method_name.startswith("__"):
                methods_output[method_name] = getattr(self, method_name)()
        return methods_output
    
    def matches_parameters(self, parameters: dict):
        """Check if the calculated parameters match the given parameters.

        Args:
            parameters (dict): A dictionary of parameter names and values
            to match.

        Returns:
            bool: True if all parameters match, False otherwise.

        Raises:
            ValueError: If a parameter is not available or cannot be
            calculated.
        """
        for key, value in parameters.items():
            if not hasattr(self, key):
                raise ValueError(
                    f"The parameter '{key}' is not available in the"
                    " SimulationMetrics class."
                )
            if getattr(self, key)() != value:
                return False
        return True
    
    # The following methods are used to calculate the derived parameters:
    def lattice_centre(self):
        """Returns the lattice centre coordinates."""
        return self.xml_data.centre
    
    def channel_height(self):
        """Returns the channel height."""
        return self.xml_data.channel_height
    
    def inlet_width(self):
        """Returns the inlet width."""
        return self.xml_data.inlet_width
    
    def outlet_width(self):
        """Returns the outlet width."""
        return self.xml_data.outlet_width
    
    def stenosis_width(self):
        """Returns the stenosis width."""
        return self.xml_data.stenosis_width
    
    def stenosis_length(self):
        """Returns the stenosis length."""
        return self.xml_data.stenosis_length
    
    def inlet_velocity(self):
        """Returns the inlet velocity."""
        return self.xml_data.inlet_velocity

    def lbm_viscosity(self):
        """Returns the LBM viscosity."""
        return (1/3)*(self.xml_data.tau - 0.5)
    
    def shear_rate(self):
        """Returns the shear rate."""
        return 12 * self.xml_data.inlet_velocity / self.xml_data.channel_height
    
    def advection_time(self):
        """Returns the advection time."""
        return self.xml_data.inlet_width/self.xml_data.inlet_velocity
    
    def hydraulic_diameter(self):
        """Returns the hydraulic diameter."""
        return 2 * (self.xml_data.channel_height * self.xml_data.inlet_width) / (self.xml_data.channel_height + self.xml_data.inlet_width)
    
    def capillary_number(self):
        """Returns the capillary number."""
        return self.lbm_viscosity() * self.shear_rate() * self.xml_data.radius / self.xml_data.ks
    
    def reynolds_number_hydraulic_diameter(self):
        """Returns the Reynolds number based on hydraulic diameter."""
        return self.xml_data.inlet_velocity * self.hydraulic_diameter() / self.lbm_viscosity()
    
    def reynolds_number_width(self):
        """Returns the Reynolds number based on width."""
        return self.xml_data.inlet_velocity * self.xml_data.inlet_width / self.lbm_viscosity()
    
    def normalised_initial_position(self):
        """Returns the normalized initial position."""
        return [
            2 * (self.xml_data.particle_start_pos[0] - self.xml_data.centre[0]) / self.xml_data.inlet_width,
            2 * (self.xml_data.particle_start_pos[1] - self.xml_data.centre[1]) / self.xml_data.outlet_width,
            2 * (self.xml_data.particle_start_pos[2] - self.xml_data.centre[2]) / self.xml_data.channel_height,
        ]
    
    def boussinesq_shear(self):
        """Returns the Boussinesq shear."""
        return self.xml_data.membrane_viscosity_shear/(self.lbm_viscosity()*self.xml_data.radius)

    def boussinesq_dilational(self):
        """Returns the Boussinesq dilational."""
        return self.xml_data.membrane_viscosity_dilation/(self.lbm_viscosity()*self.xml_data.radius)

    def viscosity_ratio(self):
        """Returns the viscosity ratio."""
        return self.xml_data.viscosity_ratio

class CrossSlotXML:
    """Extracts data from BioFM XML input files using the `ReadSimXML` tool.

    This class handles the extraction of various simulation parameters from
    the XML file provided.

    Attributes
    ----------
    xml_reader : ReadSimXML
        Instance of ReadSimXML for reading the XML file.
    tau : float
        LBM relaxation parameter.
    inlet_velocity : float
        Inlet velocity.
    channel_height : float
        Channel height.
    inlet_width : float
        Inlet width.
    outlet_width : float
        Outlet width.
    stenosis_width : float
        Stenosis width.
    stenosis_length : float
        Stenosis length.
    centre : list of float
        Centre coordinates.
    kV : float
        Mesh property kV.
    kA : float
        Mesh property kA.
    kalpha : float
        Mesh property kalpha.
    ks : float
        Mesh property ks.
    membrane_viscosity_shear : float
        Shear viscosity of the membrane.
    membrane_viscosity_dilation : float
        Dilational viscosity of the membrane.
    kMaxwell_shear : float
        Maxwell shear property.
    kMaxwell_dilation : float
        Maxwell dilation property.
    viscosity_ratio : float
        Viscosity ratio.
    radius : float
        Particle radius.
    particle_start_pos : list of float
        Initial particle position.
    """

    def __init__(self, filepath: str):
        """Initializes the XML reader and populates properties from the XML.

        Args
        ----
        filepath : str
            Path to the XML file.
        """
        self.xml_reader = ReadSimXML(filepath)

        if filepath:
            self.populate_from_xml()

    def populate_from_xml(self):
        """Populates class attributes from the XML file."""
        self.tau = self.xml_reader.search(['LBM', 'relaxation', 'tau'])
        self.inlet_velocity = self.xml_reader.search(['boundaries', 'CrossSlot', 'inletVelocity'])
        self.channel_height = -2 + self.xml_reader.search(['lattice', 'size', 'NZ'])
        self.inlet_width = self.xml_reader.search(['boundaries', 'CrossSlot', 'inletWidth'])
        self.outlet_width = self.xml_reader.search(['boundaries', 'CrossSlot', 'outletWidth'])
        self.stenosis_width = self.xml_reader.search(['boundaries', 'CrossSlot', 'stenosisWidth'])
        self.stenosis_length = self.xml_reader.search(['boundaries', 'CrossSlot', 'stenosisLength'])
        self.centre = [
            self.xml_reader.search(['lattice', 'size', 'NX']) / 2,
            self.xml_reader.search(['lattice', 'size', 'NY']) / 2,
            self.xml_reader.search(['lattice', 'size', 'NZ']) / 2
        ]

        self.kV = self.xml_reader.search(['mesh', 'physics', 'kV'])
        self.kA = self.xml_reader.search(['mesh', 'physics', 'kA'])
        self.kalpha = self.xml_reader.search(['mesh', 'physics', 'kalpha']) 
        self.ks = self.xml_reader.search(['mesh', 'physics', 'kS'])
        self.membrane_viscosity_shear = self.xml_reader.search(['mesh', 'physics', 'ShearViscosity'])
        self.membrane_viscosity_dilation = self.xml_reader.search(['mesh', 'physics', 'DilationalViscosity'])
        self.kMaxwell_shear = self.xml_reader.search(['mesh', 'physics', 'kMaxwell_shear'])
        self.kMaxwell_dilation = self.xml_reader.search(['mesh', 'physics', 'kMaxwell_dilation'])
        self.viscosity_ratio = self.xml_reader.search(['viscosityContrast', 'viscosity', 'ratio'])

        self.radius = self.xml_reader.search(['mesh', 'general', 'radius'])
        self.particle_start_pos = [
            self.xml_reader.search(['particle', 'X']),
            self.xml_reader.search(['particle', 'Y']),
            self.xml_reader.search(['particle', 'Z']),
        ]
