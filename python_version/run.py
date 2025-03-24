
"""
Main simulation file. This file sets up simulation parameters, creates a Reactor
object and a controller (either STC or PID), runs the simulation loop, and then
calls the plotting routine to display the results.
"""

import numpy as np
from reactor import get_drum_params, Reactor
from controller import STCController, PIDController
from plot import plot_results

def generate_reference(time, time_points, pow_arr):
    """Generates the reference signal using linear interpolation."""
    ref = np.interp(time, time_points, pow_arr)
    return ref

def main():
    # Choose controller type: 'STC' or 'PID'
    controller_type = 'PID'  # change to 'PID' for PID or STC for STC controller simulation
    num_drums = 1
    dt = 0.1
    T_sim = 6000
    time = np.arange(0, T_sim + dt, dt)
    nt = len(time)
    
    # Define piecewise linear reference time points (in seconds) and corresponding power levels
    time_points = np.array([0, 20, 30, 50, 60, 80, 90, 110, 130, 200]) * 30
    if num_drums == 8:
        pow_arr = np.array([1, 1, 0.5, 0.5, 1, 1, 0.5, 0.5, 1, 1])
    elif num_drums == 4:
        pow_arr = np.array([0.3, 0.3, 1, 1, 0.6, 0.6, 0.8, 0.8, 1, 1])
    elif num_drums == 2:
        pow_arr = np.array([0.7, 0.7, 0.4, 0.4, 0.4, 0.8, 0.8, 0.8, 1, 1])
    elif num_drums == 1:
        pow_arr = np.array([0.9, 0.9, 0.7, 0.7, 0.5, 0.5, 0.7, 0.7, 1, 1])
    else:
        raise ValueError("Invalid number of drums specified.")
    
    ref = generate_reference(time, time_points, pow_arr)
    
    # Get reactor/drum parameters
    Rho_d0, Reactivity_per_degree, u0, W = get_drum_params(num_drums)
    
    # Additional reactor parameters (common to both controllers)
    Sig_x = 2.65e-22
    yi = 0.061
    yx = 0.002
    lamda_x = 2.09e-5
    lamda_I = 2.87e-5
    Sum_f = 0.3358
    G = 3.2e-11
    V = 400 * 200
    P_0 = 22e6
    Pi = P_0 / (G * Sum_f * V)
    Xe0 = (yi + yx) * Sum_f * Pi / (lamda_x + Sig_x * Pi)
    I0 = yi * Sum_f * Pi / lamda_I
    
    # Set the initial state based on number of drums (state vector length is 12)
    if num_drums == 8:
        x0 = np.array([pow_arr[0]] * 7 + [I0, Xe0, 900, 898, 883])
    elif num_drums == 4:
        x0 = np.array([pow_arr[0]] * 7 + [I0, Xe0, 875, 873.5, 870])
    elif num_drums == 2:
        x0 = np.array([pow_arr[0]] * 7 + [I0, Xe0, 890, 888, 877])
    elif num_drums == 1:
        x0 = np.array([pow_arr[0]] * 7 + [I0, Xe0, 897, 895, 881])
    
    # Build reactor parameters dictionary
    reactor_params = {
        'Rho_d0': Rho_d0,
        'Reactivity_per_degree': Reactivity_per_degree,
        'Xe0': Xe0,
        'I0': I0,
        'Pi': Pi,
        'Tf0': 1105,
        'Tm0': 1087
    }
    
    # Create Reactor instance
    reactor_sim = Reactor(num_drums, x0, reactor_params)
    
    # Create the controller instance based on chosen type
    if controller_type == 'STC':
        lambda_ = 0.001
        T_C = 0.5
        max_val = 180
        min_val = 0
        max_rate = 0.5
        controller = STCController(lambda_, W, T_C, max_val, min_val, max_rate, u0)
    elif controller_type == 'PID':
        # Adjust PID gains as in the original pid.py
        Kp = 2 * 26.11e-5 / Reactivity_per_degree
        Ki = 5 * 26.11e-5 / Reactivity_per_degree
        Kd = 0.001 * 26.11e-5 / Reactivity_per_degree
        Kaw = 0.3 * 26.11e-5 / Reactivity_per_degree
        T_C = 0.2
        max_val = 180
        min_val = 0
        max_rate = 0.5 * 26.11e-5 / Reactivity_per_degree
        controller = PIDController(Kp, Ki, Kd, Kaw, T_C, max_val, min_val, max_rate, u0)
    else:
        raise ValueError("Invalid controller type.")
    
    # Preallocate arrays to store simulation results
    state_history = np.zeros((nt, len(x0)))
    control_history = np.zeros(nt)
    rho_history = np.zeros(nt)
    
    state_history[0, :] = x0
    control_history[0] = u0
    
    # Simulation loop (Euler integration)
    for i in range(nt - 1):
        dx, rho_val = reactor_sim.reactor_dae(time[i], state_history[i, :], control_history[i])
        rho_history[i] = rho_val
        state_history[i + 1, :] = state_history[i, :] + dx * dt
        # Update the control signal using the controller's update method
        control_history[i + 1] = controller.update(time[i], state_history[i + 1, 0], ref[i + 1])
    
    # Compute final reactivity at the last time step
    _, rho_history[-1] = reactor_sim.reactor_dae(time[-1], state_history[-1, :], control_history[-1])
    
    # Compute error indices (using downsampled signals) as in the original file
    downsample_factor = 10
    t_ds = time[::downsample_factor]
    y_ds = state_history[::downsample_factor, 0]
    r_ds = ref[::downsample_factor]
    e_ds = r_ds - y_ds
    MAE = np.mean(np.abs(e_ds))
    IAE = np.trapz(np.abs(e_ds), t_ds)
    ITAE = np.trapz(t_ds * np.abs(e_ds), t_ds)
    ISE = np.trapz(e_ds ** 2, t_ds)
    ITSE = np.trapz(t_ds * e_ds ** 2, t_ds)
    print(f"Mean Absolute Error (MAE): {MAE:.4f}")
    print(f"Integral Absolute Error (IAE): {IAE:.4f}")
    print(f"Integral Time Absolute Error (ITAE): {ITAE:.4f}")
    print(f"Integral Square Error (ISE): {ISE:.4f}")
    print(f"Integral Time Square Error (ITSE): {ITSE:.4f}")
    
    # (Optional) Print control effort based on the raw control input derivative
    du = np.zeros_like(control_history)
    du[1:] = np.diff(control_history) / dt
    control_effort = np.linalg.norm(du)
    print(f"Control Effort: {control_effort}")
    
    # Call the plot function to visualize the simulation results
    plot_results(time, state_history, control_history, rho_history, ref, controller_type, f"{controller_type}_power_simulation.png")

if __name__ == '__main__':
    main()


