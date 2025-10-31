"""  
Goal: Model a VSC with UIC control (without angle state)

1. Create a UIC-style VSC without PLL
2. Uses stationary reference frame (no dq transformation)
3. Simple grid-forming control

- Use existing solvers
- Use existing blocks
- Differential equations are divided into steps (functions with different objectives)

"""

from .blocks import *

class UIC_sig(DAEModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.bus_idx = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])
        self.bus_idx_red = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])
        

    def load_flow_pv(self):
        return self.bus_idx['terminal'], -self.par['p_ref']*self.par['S_n'], self.par['v_ref']

    def int_par_list(self):
        return ['f']

    def bus_ref_spec(self):
        return {'terminal': self.par['bus']}
    
    def state_list(self):
        """Return list of states for this VSC"""
        return ['vi_x', 'vi_y', 'x_filter']

    def input_list(self):
        """Input list for VSC"""
        return ['p_ref', 'q_ref', 'v_ref']

    def state_derivatives(self, dx, x, v):
        dX = self.local_view(dx)
        X = self.local_view(x)
        par = self.par
        s_ref = self.p_ref(x, v) + 1j*self.q_ref(x, v)
        v_ref = self.v_ref(x, v)
        v_t = v[self.bus_idx_red['terminal']]
        vi = X['vi_x'] + 1j*X['vi_y']
       
        i_ref = np.conj(s_ref/v_t)
        i_a = self.i_a(x, v)
        theta = np.angle(vi, deg=False)
        
        i_error = (i_ref - i_a)
        v_error = (v_ref - abs(vi))*np.exp(1j*(theta))

        dvi = 1j*100*np.pi*par['Ki']*i_error+100*np.pi*par['Kv']*v_error +1j*vi*X['x_filter']*par['perfect_tracking']
        
        delta_omega = (dvi/vi).imag
        dX['x_filter'] = (1/par['T_filter'])*(delta_omega-X['x_filter'])
        perfect_tracking_addition = 1j*vi*X['x_filter']*par['perfect_tracking']
        
        dX['vi_x'] = np.real(dvi)
        dX['vi_y'] = np.imag(dvi)
        
        # Debug
        if not hasattr(self, '_debug_counter'):
            self._debug_counter = 0
        self._debug_counter += 1
        if self._debug_counter == 1 or self._debug_counter == 7500 or self._debug_counter == 100 or self._debug_counter == 500 or self._debug_counter == 1000 or (self._debug_counter % 5000 == 0 and self._debug_counter <= 60000): 
            print('Debug values (iteration', self._debug_counter, '):')
            print('  X[vi_x]:', X['vi_x'])
            print('  X[vi_y]:', X['vi_y'])
            print('vi: ', vi)
            print('theta: ', theta)
            print('v_t: ', v_t)
            print('  p_ref:', s_ref.real)
            print('  q_ref:', s_ref.imag)
            print('p_e: ', self.p_e(x, v))
            print('q_e: ', self.q_e(x, v))
            print('  v_ref:', v_ref)
            print('  i_ref:', i_ref)
            print('i_a: ', self.i_a(x, v))
            print('i_inj: ', self.i_inj(x, v))
            print('dvi: ', dvi)
            print('dX[vi_x]:', dX['vi_x'])
            print('dX[vi_y]:', dX['vi_y'])
            print('v_error: ', v_error)
            print('i_error: ', i_error)
            print('x_filter: ', X['x_filter'])
            print('delta_omega: ', delta_omega)
            print('dX[x_filter]:', dX['x_filter'])
            print('perfect_tracking_addition: ', perfect_tracking_addition)
            #print('perfect_tracking_addition: ', perfect_tracking_addition)

        return

    def i_inj(self, x, v):
        """Norton equivalent"""
        X = self.local_view(x)
        vi = X['vi_x'] + 1j*X['vi_y']
        xf = self.par['xf'][0]
        current = vi/(1j*xf)
        return current
    
    def current_injections(self, x, v):
        i_n_r = self.par['S_n'][0] / self.sys_par['s_n'] 
        return self.bus_idx_red['terminal'], self.i_inj(x, v) * i_n_r
    
    def init_from_load_flow(self, x_0, v_0, S):
        """Initialize from load flow solution"""
        par = self.par
        X = self.local_view(x_0)

        v_t = v_0[self.bus_idx_red['terminal']]
        current = np.conj(S/v_t)
        vi = v_t + 1j*current*par['xf']

        self._input_values['v_ref'] = abs(vi)
        self._input_values['p_ref'] = S.real
        self._input_values['q_ref'] = S.imag
        
        X['vi_x'] = np.real(vi)
        X['vi_y'] = np.imag(vi)
        X['x_filter'] = 0.0

        print('vi_x init', X['vi_x'])
        print('vi_y init', X['vi_y'])
        print('p_ref init', S.real)
        print('q_ref init', S.imag)
        print('p_e init', self.p_e(x_0, v_0))
        print('q_e init', self.q_e(x_0, v_0))
        print('v_ref init', abs(vi))
        print('v_t init', v_t)
        return

    def dyn_const_adm(self):
        par = self.par
        idx_bus = self.bus_idx['terminal']
        bus_v_n = self.sys_par['bus_v_n'][idx_bus]
        Y = 1/(1j*par['xf'])
        return Y, (idx_bus,)*2

    def v_t(self, x, v):
        """Terminal voltage"""
        return v[self.bus_idx_red['terminal']]

    def v_q(self,x,v):
        return (self.v_t(x,v)*np.exp(-1j*self.local_view(x)['angle'])).imag
    
    def i_a(self, x, v):
        par = self.par
        X = self.local_view(x)
        v_t = v[self.bus_idx_red['terminal']]
        vi = X['vi_x'] + 1j*X['vi_y']
        i_a = -(v_t-vi)/(1j*par['xf'])
        return i_a

    def s_e(self, x, v):
        return self.v_t(x, v)*np.conj(self.i_a(x, v))

    def p_e(self, x, v):
        return self.s_e(x, v).real

    def q_e(self, x, v):
        return self.s_e(x, v).imag
