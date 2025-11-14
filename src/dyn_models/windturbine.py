from src.dyn_models.utils import DAEModel
import numpy as np

class WindTurbine(DAEModel):

    """
    Inputs:
    'windturbine': {
        'WT': [
            [
                'name', 'UIC', 'H_m', 'H_e', 'K', 'D', 'K_pitch','rho', 'R', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6','x_param', 'dt'],
            [
                'WT1', 'UIC1',  1.0,    1.0, 1.0, 0.5,    5.0,   1.225,  45.0, 0.5176, 116.0, 0.4, 5.0, 21.0, 0.0068,2.0, 0.01]
        ],
    }
    """

    def connections(self):
        return [
            {
                'input': 'P_e',
                'source': {
                    'container': 'vsc',
                    'mdl': 'UIC_sig',
                    'id': self.par['WT'],
                },
                'output': 'p_e',
            },
            {
                'output': 'P_ref',
                'destination': {
                    'container': 'vsc',
                    'mdl': 'UIC_sig',
                    'id': self.par['WT'],
                },
                'input': 'p_ref',
            }
        ]

    def state_list(self):
        return ['omega_m', 'omega_e', 'theta_m', 'theta_e', 'pitch_angle']

    def input_list(self):
        return ['P_e']

    def output_list(self):
        return ['P_ref']
    
    def state_derivatives(self, dx, x, v):
        dX = self.local_view(dx)
        X = self.local_view(x)
        par = self.par

        Pm = self.P_m(x, v)
        Pe = self.P_e(x, v)
        omega_m_ref = Pe/(Pm/X['omega_m'])
        
        # swing eqs for wt dynamics
        dX['theta_m'] = X['omega_m']
        dX['theta_e'] = X['omega_e']
        dX['omega_m'] = (1/par['H_m']) * (Pm/X['omega_m'] - par['K'] * (X['theta_m'] - X['theta_e']) - par['D'] * (X['omega_m'] - X['omega_e']))
        dX['omega_e'] = (1/par['H_e']) * (Pe/X['omega_e'] + par['K'] * (X['theta_m'] - X['theta_e']) + par['D'] * (X['omega_m'] - X['omega_e']))

        # pitch control
        dX['pitch_angle'] = par['K_pitch'] * (omega_m_ref - X['omega_m'])

        if not hasattr(self, '_debug_counter'):
            self._debug_counter = 0
        self._debug_counter += 1
        if self._debug_counter == 1 or self._debug_counter == 7500 or self._debug_counter == 100 or self._debug_counter == 500 or self._debug_counter == 1000 or (self._debug_counter % 5000 == 0 and self._debug_counter <= 60000): 
            print('Debug values (iteration', self._debug_counter, '):')
            print('  X[omega_m]:', X['omega_m'])
            print('  X[omega_e]:', X['omega_e'])
            print('  X[theta_m]:', X['theta_m'])
            print('  X[theta_e]:', X['theta_e'])
            print('  X[pitch_angle]:', X['pitch_angle'])
            print('  Pm:', Pm)
            print('  Pe:', Pe)
            print('  omega_m_ref:', omega_m_ref)

        return
    
    def init_from_connections(self, x_0, v_0, S):
        X = self.local_view(x_0)
        par = self.par
        self._input_values['P_e'] = self.P_e(x_0, v_0)
        tip_speed_ratio = 7.0 # assume start value to be ideal? dont know if 7 is ideal
        
        X['theta_m'] = 0.0
        X['theta_e'] = 0.0
        X['omega_m'] = tip_speed_ratio * self.wind_speed(x_0, v_0) / par['R'] # use actual values for mechanical and electric speed, not relative
        X['omega_e'] = X['omega_m']
        X['pitch_angle'] = 0.0 # ok to start at 0

        return

    def P_m(self, x, v):
        par = self.par
        wind_speed = self.wind_speed(x, v)
        Cp = self.Cp(x, v)
        return 0.5 * par['rho'] * np.pi * par['R']**2 * wind_speed**3 * Cp

    def P_ref(self, x, v):
        return self.P_m(x, v)

    def Cp(self, x, v):
        X = self.local_view(x)
        par = self.par
        c1 = par['c1']
        c2 = par['c2']
        c3 = par['c3']
        c4 = par['c4']
        c5 = par['c5']
        c6 = par['c6']
        x_param = par['x_param']

        lam = X['omega_m']*par['R']/self.wind_speed(x, v)
        big_lam = 1/(lam+0.08*X['pitch_angle']) - 0.035/(1+X['pitch_angle']**3)

        Cp = c1 * (c2 * 1/(big_lam) - c3 * X['pitch_angle'] - c4 * X['pitch_angle']**x_param - c5) * np.exp(- c6 * (1/big_lam)) 
        return Cp

    def wind_speed(self, x, v):
        ## change to actual wind speed - from file?
        #par = self.par
        #t0 = - par['dt']
        #t += t0 + par['dt']
        u0 = 10 #par['u0'] # mean wind speed
        Ak = 0 #par['Ak'] # amplitude of kth harmonic
        omega_k = 0 #par['omega_k'] # frequency of kth harmonic 0.1 to 10 Hz
        ug_max = 0 #par['ug_max'] # maximum wind gust speed # val = 10 m/s
        omega_gust = 1/10 #1/par['T_gust'] # frequency of gust # val = 1/10 s to 1/50 s
        
        ug = (2 * ug_max) #/(1 + np.exp(-4 * np.sin(omega_gust * t) - 1))

        u_wind = u0 #* (1 + Ak * np.sin(omega_k * t)) + ug
        return u_wind

