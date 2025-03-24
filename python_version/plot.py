#!/usr/bin/env python3
"""
This module contains the plotting function that visualizes the simulation results.
It creates a figure with subplots for core power, drum rotation and speed, core reactivity,
and temperatures of fuel, moderator, and coolant.
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

def plot_results(time, x, u, rho_arr, ref, controller_type, filename):
    """
    Plots simulation results.
    
    Args:
      time: 1D array of time values (in seconds).
      x: 2D array of state history.
      u: 1D array of control inputs.
      rho_arr: 1D array of reactivity values.
      ref: 1D array of reference values.
      controller_type: String indicating the controller used (for plot title).
      filename: Name of the file to save the figure.
    """
    t_min = time / 60.0  # convert time to minutes
    dt = time[1] - time[0] if len(time) > 1 else 1

    # Compute the derivative of the control input
    du = np.zeros_like(u)
    du[1:] = np.diff(u) / dt

    # For STC, apply a moving average filter (window size 30) to match original behavior
    if controller_type == 'STC':
        window_size = 30
        du_filtered = du.copy()
        for i in range(window_size, len(du) - window_size):
            du_filtered[i] = np.mean(du[i - window_size:i + window_size + 1])
        # In the original code, the plot uses indices 100:-100 to avoid edge effects
        t_plot = t_min[100:-100]
        du_plot = du_filtered[100:-100]
    else:
        t_plot = t_min
        du_plot = du

    font_size = 12
    font_weight = 'bold'
    
    fig = plt.figure(figsize=(12, 16))
    gs = gridspec.GridSpec(4, 2, figure=fig)
    
    # Plot 1: Core Power (Actual vs Desired)
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(t_min, x[:, 0] * 100, linewidth=2, label='Actual power')
    ax0.plot(t_min, ref * 100, '--', linewidth=2, label='Desired power')
    ax0.grid(True)
    ax0.set_xlabel('Time (min)', fontsize=font_size, fontweight=font_weight)
    ax0.set_ylabel('Relative Power (%)', fontsize=font_size, fontweight=font_weight)
    ax0.set_title(f'{controller_type} Controller: Core Power Simulation', fontsize=font_size, fontweight=font_weight)
    ax0.set_ylim([0, 120])
    ax0.set_xlim([0, np.max(t_min)])
    ax0.legend(fontsize=font_size)
    
    # Plot 2: Drum Rotation (Control Input)
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(t_min, u, linewidth=2)
    ax1.grid(True)
    ax1.set_xlabel('Time (min)', fontsize=font_size, fontweight=font_weight)
    ax1.set_ylabel('Rotation (deg)', fontsize=font_size, fontweight=font_weight)
    ax1.set_title('Drum Rotation', fontsize=font_size, fontweight=font_weight)
    ax1.set_xlim([0, np.max(t_min)])
    
    # Plot 3: Control Drum Speed
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(t_plot, du_plot, linewidth=2)
    ax2.grid(True)
    ax2.set_xlabel('Time (min)', fontsize=font_size, fontweight=font_weight)
    ax2.set_ylabel('Speed (deg/s)', fontsize=font_size, fontweight=font_weight)
    ax2.set_title('Control Drum Speed', fontsize=font_size, fontweight=font_weight)
    ax2.set_xlim([0, np.max(t_min)])
    
    # Plot 4: Core Reactivity
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(t_min, rho_arr * 1e5, linewidth=2)
    ax3.grid(True)
    ax3.set_xlabel('Time (min)', fontsize=font_size, fontweight=font_weight)
    ax3.set_ylabel('Reactivity (pcm)', fontsize=font_size, fontweight=font_weight)
    ax3.set_title('Core Reactivity', fontsize=font_size, fontweight=font_weight)
    ax3.set_xlim([0, np.max(t_min)])
    
    # Plot 5: Temperature of Fuel
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.plot(t_min, x[:, 9], linewidth=2)
    ax4.grid(True)
    ax4.set_xlabel('Time (min)', fontsize=font_size, fontweight=font_weight)
    ax4.set_ylabel('Temperature (K)', fontsize=font_size, fontweight=font_weight)
    ax4.set_title('Temperature of Fuel', fontsize=font_size, fontweight=font_weight)
    ax4.set_xlim([0, np.max(t_min)])
    
    # Plot 6: Temperature of Moderator
    ax5 = fig.add_subplot(gs[3, 0])
    ax5.plot(t_min, x[:, 10], linewidth=2)
    ax5.grid(True)
    ax5.set_xlabel('Time (min)', fontsize=font_size, fontweight=font_weight)
    ax5.set_ylabel('Temperature (K)', fontsize=font_size, fontweight=font_weight)
    ax5.set_title('Temperature of Moderator', fontsize=font_size, fontweight=font_weight)
    ax5.set_xlim([0, np.max(t_min)])
    
    # Plot 7: Temperature of Coolant
    ax6 = fig.add_subplot(gs[3, 1])
    ax6.plot(t_min, x[:, 11], linewidth=2)
    ax6.grid(True)
    ax6.set_xlabel('Time (min)', fontsize=font_size, fontweight=font_weight)
    ax6.set_ylabel('Temperature (K)', fontsize=font_size, fontweight=font_weight)
    ax6.set_title('Temperature of Coolant', fontsize=font_size, fontweight=font_weight)
    ax6.set_xlim([0, np.max(t_min)])
    
    fig.tight_layout()
    plt.savefig(filename)
    plt.show()


