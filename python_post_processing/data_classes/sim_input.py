from ..xml.parameter_reader import ParameterReader

class Mesh:
    def __init__(self, filepath: str = None):
        # Initialize properties with default values or None
        self.kV = None
        self.kA = None
        self.kalpha = None
        self.ks = None
        self.membrane_viscosity_shear = None
        self.membrane_viscosity_dilation = None
        self.kMaxwell_shear = None
        self.kMaxwell_dilation = None
        self.viscosity_ratio = None
        self.radius = None
        self.x0 = None
        self.y0 = None
        self.z0 = None

        # If a file path is provided, populate properties from XML
        if filepath:
            self.populate_from_xml(filepath)

    def populate_from_xml(self, filepath: str):
        # add the xml parameter reader for the given filepath
        self.parameter_reader = ParameterReader(filepath)

        # mesh properties
        self.kV = self.parameter_reader.get_value_from_files(['mesh', 'physics', 'kV'])
        self.kA = self.parameter_reader.get_value_from_files(['mesh', 'physics', 'kA'])
        self.kalpha = self.parameter_reader.get_value_from_files(['mesh', 'physics', 'kalpha']) 
        self.ks = self.parameter_reader.get_value_from_files(['mesh', 'physics', 'kS'])
        self.membrane_viscosity_shear = self.parameter_reader.get_value_from_files(['mesh', 'physics', 'ShearViscosity'])
        self.membrane_viscosity_dilation = self.parameter_reader.get_value_from_files(['mesh', 'physics', 'DilationalViscosity'])
        self.kMaxwell_shear = self.parameter_reader.get_value_from_files(['mesh', 'physics', 'kMaxwell_shear'])
        self.kMaxwell_dilation = self.parameter_reader.get_value_from_files(['mesh', 'physics', 'kMaxwell_dilation'])
        self.viscosity_ratio = self.parameter_reader.get_value_from_files(['viscosityContrast', 'viscosity', 'ratio'])

        # initial particle position and size
        self.radius = self.parameter_reader.get_value_from_files(['mesh', 'general', 'radius'])
        self.x0 = self.parameter_reader.get_value_from_files(['particle', 'X'])
        self.y0 = self.parameter_reader.get_value_from_files(['particle', 'Y'])
        self.z0 = self.parameter_reader.get_value_from_files(['particle', 'Z'])


class CrossSlot:
    def __init__(self, filepath: str = None):
        # Initialize properties with default values or None
        self.tau = None
        self.inlet_velocity = None
        self.channel_height = None
        self.inlet_width = None
        self.outlet_width = None
        self.particle_z_init = None
        self.centre = [None, None, None]

        # If a file path is provided, populate properties from XML
        if filepath:
            self.parameter_reader = ParameterReader(filepath)

            # Populate properties from XML file
            self.tau = self.parameter_reader.get_value_from_files(['LBM', 'relaxation', 'tau'])
            self.inlet_velocity = self.parameter_reader.get_value_from_files(['boundaries', 'CrossSlot', 'inletVelocity'])
            self.channel_height = -2 + self.parameter_reader.get_value_from_files(['lattice', 'size', 'NZ'])
            self.inlet_width = self.parameter_reader.get_value_from_files(['boundaries', 'CrossSlot', 'inletWidth'])
            self.outlet_width = self.parameter_reader.get_value_from_files(['boundaries', 'CrossSlot', 'outletWidth'])
            self.particle_z_init = self.parameter_reader.get_value_from_files(['particle', 'Z'])
            self.centre = [
                self.parameter_reader.get_value_from_files(['lattice', 'size', 'NX']) / 2,
                self.parameter_reader.get_value_from_files(['lattice', 'size', 'NY']) / 2,
                self.parameter_reader.get_value_from_files(['lattice', 'size', 'NZ']) / 2
            ]

