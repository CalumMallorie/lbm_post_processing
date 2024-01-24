from my_xml_tools import ReadSimXML

class CrossSlotParameters:
    """Calculate various derived parameters for cross-slot simulations
    based on XML file data.

    This class uses the CrossSlotXML class to extract data from the
    BioFM XML input files. 

    Attributes:
        xml_data (CrossSlotXML): Instance of CrossSlotXML for data extraction.
    """
    def __init__(self, filepath):
        """Initialize the class with the path to an XML file.

        Args:
            filepath (str): The path to the simulation directory.

        Raises:
            ValueError: If the XML file cannot be read or parsed correctly.
        """
        try:
            self.xml_data = CrossSlotXML(filepath)
        except Exception as e:
            raise ValueError(f"Failed to read or parse the XML file: {e}")

    def dump(self):
        methods_output = {}
        for method_name in dir(self):
            # skip this attribute
            if method_name == "dump":
                continue
            # skip the matches_parameters method
            if method_name == "matches_parameters":
                continue

            # Check if it's a callable method and not a built-in one
            if callable(getattr(self, method_name)) and not method_name.startswith("__"):
                methods_output[method_name] = getattr(self, method_name)()
        return methods_output
    
    def matches_parameters(self, parameters: dict):
        """Check if the calculated parameters match the given parameters.

        Args:
            parameters (dict): A dictionary of parameter names and
            values to match.

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
                    "SimulationMetrics class."
                )
            if getattr(self, key)() != value:
                return False
        return True
    
    # The following methods are used to calculate the derived parameters:
    def lbm_viscosity(self):
        return (1/3)*(self.xml_data.tau - 0.5)
    
    def shear_rate(self):
        return 12 * self.xml_data.inlet_velocity / self.xml_data.channel_height
    
    def advection_time(self):
        return self.xml_data.inlet_width/self.xml_data.inlet_velocity
    
    def hydraulic_diameter(self):
        return 2 * (self.xml_data.channel_height * self.xml_data.inlet_width) / (self.xml_data.channel_height + self.xml_data.inlet_width)
    
    def capillary_number(self):
        return self.lbm_viscosity() * self.shear_rate() * self.mesh_data.radius / self.mesh_data.ks
    
    def reynolds_number_hydraulic_diameter(self):
        return self.xml_data.inlet_velocity * self.hydraulic_diameter() / self.lbm_viscosity()
    
    def reynolds_number_width(self):
        return self.xml_data.inlet_velocity * self.xml_data.inlet_width / self.lbm_viscosity()
    
    def normalised_initial_position(self):
        return [
            2 * (self.xml_data.particle_start_pos[0] - self.xml_data.centre[0]) / self.xml_data.inlet_width,
            2 * (self.xml_data.particle_start_pos[1] - self.xml_data.centre[1]) / self.xml_data.outlet_width,
            2 * (self.xml_data.particle_start_pos[2] - self.xml_data.centre[2]) / self.xml_data.channel_height,
        ]
    
    def boussinesq_shear(self):
        return self.xml_data.membrane_viscosity_shear/(self.lbm_viscosity()*self.xml_data.radius)

    def boussinesq_dilational(self):
        return self.xml_data.membrane_viscosity_dilation/(self.lbm_viscosity()*self.xml_data.radius)

    def viscosity_ratio(self):
        return self.xml_data.viscosity_ratio

class CrossSlotXML:
    def __init__(self, filepath: str):
        # Initialise the xml reader
        self.xml_reader = ReadSimXML(filepath)

        # If a file path is provided, populate properties from XML
        if filepath:
            self.populate_from_xml(filepath)

    def populate_from_xml(self):
        # lattice and geometry properties
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

        # mesh properties
        self.kV = self.xml_reader.search(['mesh', 'physics', 'kV'])
        self.kA = self.xml_reader.search(['mesh', 'physics', 'kA'])
        self.kalpha = self.xml_reader.search(['mesh', 'physics', 'kalpha']) 
        self.ks = self.xml_reader.search(['mesh', 'physics', 'kS'])
        self.membrane_viscosity_shear = self.xml_reader.search(['mesh', 'physics', 'ShearViscosity'])
        self.membrane_viscosity_dilation = self.xml_reader.search(['mesh', 'physics', 'DilationalViscosity'])
        self.kMaxwell_shear = self.xml_reader.search(['mesh', 'physics', 'kMaxwell_shear'])
        self.kMaxwell_dilation = self.xml_reader.search(['mesh', 'physics', 'kMaxwell_dilation'])
        self.viscosity_ratio = self.xml_reader.search(['viscosityContrast', 'viscosity', 'ratio'])

        # initial particle position and size
        self.radius = self.xml_reader.search(['mesh', 'general', 'radius'])
        self.particle_start_pos = [
            self.xml_reader.search(['particle', 'X']),
            self.xml_reader.search(['particle', 'Y']),
            self.xml_reader.search(['particle', 'Z']),
        ]
