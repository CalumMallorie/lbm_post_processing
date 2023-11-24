from .sim_input import Mesh, CrossSlot

class SimulationMetrics:
    def __init__(self, filepath):
        self.filepath = filepath
        self.mesh_data = Mesh(filepath)
        self.cross_slot_data = CrossSlot(filepath)

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
        
    def lbm_viscosity(self):
        return (1/3)*(self.cross_slot_data.tau - 0.5)
    
    def shear_rate(self):
        return 12 * self.cross_slot_data.inlet_velocity / self.cross_slot_data.channel_height
    
    def advection_time(self):
        return self.cross_slot_data.inlet_width/self.cross_slot_data.inlet_velocity
    
    def hydraulic_diameter(self):
        return 2 * (self.cross_slot_data.channel_height * self.cross_slot_data.inlet_width) / (self.cross_slot_data.channel_height + self.cross_slot_data.inlet_width)
    
    def capillary_number(self):
        return self.lbm_viscosity() * self.shear_rate() * self.mesh_data.radius / self.mesh_data.ks
    
    def reynolds_number_hydraulic_diameter(self):
        return self.cross_slot_data.inlet_velocity * self.hydraulic_diameter() / self.lbm_viscosity()
    
    def reynolds_number_width(self):
        return self.cross_slot_data.inlet_velocity * self.cross_slot_data.inlet_width / self.lbm_viscosity()
    
    def normalised_initial_z_position(self):
        return 2 * (self.mesh_data.z0 - self.cross_slot_data.centre[2]) / self.cross_slot_data.channel_height
    
    def boussinesq_shear(self):
        return self.mesh_data.membrane_viscosity_shear/(self.lbm_viscosity()*self.mesh_data.radius)

    def boussinesq_dilational(self):
        return self.mesh_data.membrane_viscosity_dilation/(self.lbm_viscosity()*self.mesh_data.radius)

    def viscosity_ratio(self):
        return self.mesh_data.viscosity_ratio
    
    def matches_parameters(self, parameters: dict):
        for key, value in parameters.items():
            if not hasattr(self, key):
                raise ValueError(f"The parameter '{key}' is not available in the SimulationMetrics class.")
            if getattr(self, key)() != value:
                return False
        return True
