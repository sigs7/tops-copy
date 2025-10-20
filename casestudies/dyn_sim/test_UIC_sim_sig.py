from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import src.dynamic as dps
import src.solvers as dps_sol
import importlib
importlib.reload(dps)
import sys


if __name__ == '__main__':

    # region Model loading and initialisation stage
    import casestudies.ps_data.test_UIC as model_data
    model = model_data.load()
    ps = dps.PowerSystemModel(model=model)  # Load into a PowerSystemModel object

    ps.power_flow()  # Power flow calculation

    ps.init_dyn_sim()  # Initialise dynamic variables
    x0 = ps.x0.copy()  # Initial states

    t = 0
    result_dict = defaultdict(list)
    t_end = 60  # Simulation time

    # Solver
    sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, x0, t_end, max_step=5e-3)
    print('Largest mismatch: ', max(abs(ps.state_derivatives(0, ps.x_0, ps.v_0))))
    print(ps.state_derivatives(0, ps.x_0, ps.v_0))
    # endregion

    # region Runtime variables
    # Additional plot variables
    P_m_stored = []
    P_e_stored = []
    E_f_stored = []
    v_bus = []
    I_stored = []
    theta_stored = []
    p_ref_stored = []
    q_ref_stored = []
    v_ref_stored = []
    vi_mag_stored = []
    P_actual_stored = []
    Q_actual_stored = []

    event_flag1 = False  # Create a different flag for each system event
    # endregion

    # Simulation loop starts here!
    while t < t_end:
        sys.stdout.write("\r%d%%" % (t/(t_end)*100))
        # Update dt in the VSC model for accurate omega calculation
        ps.vsc['UIC_sig'].dt = sol.dt
        
        result = sol.step()
        x = sol.y
        v = sol.v
        t = sol.t

        sc_bus_idx = ps.vsc['UIC_sig'].bus_idx_red['terminal'][0]

        # Short circuit
        if 30 <= t <= 30.05 and event_flag1:
            ps.y_bus_red_mod[(sc_bus_idx,) * 2] = 1e5
        else:
            ps.y_bus_red_mod[(sc_bus_idx,) * 2] = 0

        # region Store variables
        result_dict['Global', 't'].append(sol.t)
        [result_dict[tuple(desc)].append(state) for desc, state in zip(ps.state_desc, x)]
        # Store additional variables

        I_vsc = ps.y_bus_red_full[0, 1] * (v[0] - v[1])
        I_stored.append(np.abs(I_vsc))  # Stores magnitude of VSC current
        v_bus.append(np.abs(v[0]))  # Stores magnitude of bus voltage
        
        # Calculate theta and vi magnitude directly from state vector x
        # Get vi_x and vi_y from the VSC model's local view
        vsc_local = ps.vsc['UIC_sig'].local_view(x)
        vi_x = vsc_local['vi_x'][0]
        vi_y = vsc_local['vi_y'][0]
        vi = vi_x + 1j*vi_y
        theta = vi / abs(vi) if abs(vi) > 1e-10 else 0  # Avoid division by zero
        theta_stored.append(np.angle(theta))  # Store angle in radians
        
        # Store references and actual values
        vi_mag_stored.append(abs(vi))
        p_ref_stored.append(ps.vsc['UIC_sig']._input_values['p_ref'][0])
        q_ref_stored.append(ps.vsc['UIC_sig']._input_values['q_ref'][0])
        v_ref_stored.append(ps.vsc['UIC_sig']._input_values['v_ref'][0])
        
        # Calculate actual P and Q
        s_actual = ps.vsc['UIC_sig'].s_e(x, v)
        P_actual_stored.append(s_actual.real[0])
        Q_actual_stored.append(s_actual.imag[0])
        # endregion

    # Convert dict to pandas dataframe
    result = pd.DataFrame(result_dict, columns=pd.MultiIndex.from_tuples(result_dict))

    # region Plotting
    t_stored = result[('Global', 't')]

    fig, ax = plt.subplots(4)
    fig.suptitle('UIC VSC: vi_x, vi_y, current, and theta')
    ax[0].plot(t_stored, result.xs(key='vi_x', axis='columns', level=1), label='vi_x')
    ax[0].plot(t_stored, result.xs(key='vi_y', axis='columns', level=1), label='vi_y')
    ax[0].legend()
    ax[0].set_ylabel('Controller states')
    ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.14f'))
    
    ax[1].plot(t_stored, np.array(I_stored), label='I_vsc')
    ax[1].legend()
    ax[1].set_ylabel('VSC Current (pu)')
    ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.14f'))

    ax[2].plot(t_stored, np.array(v_bus), label='V_bus')
    ax[2].legend()
    ax[2].set_ylabel('Voltage (pu)')
    ax[2].yaxis.set_major_formatter(FormatStrFormatter('%.14f'))
    
    ax[3].plot(t_stored, np.array(theta_stored)*180/np.pi, label='theta')
    ax[3].legend()
    ax[3].set_ylabel('Theta (degrees)')
    ax[3].set_xlabel('Time (s)')
    ax[3].yaxis.set_major_formatter(FormatStrFormatter('%.14f'))
    
    # Save figure to testing folder with dynamic naming
    plt.tight_layout()
    suffix = '_with_short_circuit' if event_flag1 else '_no_disturbance'
    filename1 = f'testing/uic_sig_simulation_results{suffix}.png'
    plt.savefig(filename1, dpi=300, bbox_inches='tight')
    print(f'Plot saved to: {filename1}')
    
    # Create second figure with references
    fig2, ax2 = plt.subplots(3, figsize=(10, 8))
    fig2.suptitle(f'UIC VSC: References vs Actual Values{suffix.replace("_", " ").title()}')
    
    # Plot 1: vi magnitude vs v_ref
    ax2[0].plot(t_stored, np.array(vi_mag_stored), label='|vi| actual', linewidth=2)
    ax2[0].plot(t_stored, np.array(v_ref_stored), '--', label='v_ref', linewidth=2)
    ax2[0].legend()
    ax2[0].set_ylabel('Voltage magnitude (pu)')
    ax2[0].grid(True)
    ax2[0].yaxis.set_major_formatter(FormatStrFormatter('%.14f'))
    
    # Plot 2: P vs p_ref
    ax2[1].plot(t_stored, np.array(P_actual_stored), label='P actual', linewidth=2)
    ax2[1].plot(t_stored, np.array(p_ref_stored), '--', label='p_ref', linewidth=2)
    ax2[1].legend()
    ax2[1].set_ylabel('Active power (pu)')
    ax2[1].grid(True)
    ax2[1].yaxis.set_major_formatter(FormatStrFormatter('%.14f'))
    
    # Plot 3: Q vs q_ref
    ax2[2].plot(t_stored, np.array(Q_actual_stored), label='Q actual', linewidth=2)
    ax2[2].plot(t_stored, np.array(q_ref_stored), '--', label='q_ref', linewidth=2)
    ax2[2].legend()
    ax2[2].set_ylabel('Reactive power (pu)')
    ax2[2].set_xlabel('Time (s)')
    ax2[2].grid(True)
    ax2[2].yaxis.set_major_formatter(FormatStrFormatter('%.14f'))
    
    plt.tight_layout()
    filename2 = f'testing/uic_sig_references_vs_actual{suffix}.png'
    plt.savefig(filename2, dpi=300, bbox_inches='tight')
    print(f'Plot saved to: {filename2}')
    # endregion
