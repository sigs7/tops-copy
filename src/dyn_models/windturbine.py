from src.dyn_models.utils import DAEModel
import numpy as np

class WindTurbine(DAEModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bus_idx = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])
        self.bus_idx_red = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])
    
    def bus_ref_spec(self):
        return {'terminal': self.par['bus']}

    def load_flow_pv(self):
        return self.bus_idx['terminal'], -self.par['P']*self.par['N_par'], self.par['V']

    def state_list(self):
        return

    def input_list(self):
        return
    
    def state_derivatives(self, dx, x, v):
        return
    
    def init_from_load_flow(self, x_0, v_0, S):
        return
    
    def current_injections(self, x, v):
        return
    
    def dyn_const_adm(self):
        return
    
    def p_e(self, x, v):
        return
        
    def q_e(self, x, v):
        return

    def s_e(self, x, v):
        return

    def v_t(self, x, v):
        return