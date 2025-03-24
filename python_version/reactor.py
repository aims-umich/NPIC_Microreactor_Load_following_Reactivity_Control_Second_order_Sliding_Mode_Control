#!/usr/bin/env python3
"""
This module defines the reactor dynamics.
It includes a helper function to get drum parameters and a Reactor class that
computes the reactor differential–algebraic equations (DAE).
"""

import numpy as np

def get_drum_params(num_drums):
    """Return drum-specific parameters: Rho_d0, Reactivity_per_degree, u0, and W."""
    if num_drums == 8:
        Rho_d0 = -0.033085599
        Reactivity_per_degree = 26.11e-5
        u0 = 77.56  # 8 drums
        W = 0.06
    elif num_drums == 4:
        Rho_d0 = -0.033085599 + 0.013980296
        Reactivity_per_degree = 16.11e-5
        u0 = 108.5  # 4 drums
        W = 0.1
    elif num_drums == 2:
        Rho_d0 = -0.033085599 + 0.0074
        Reactivity_per_degree = 7.33e-5
        u0 = 165.5  # 2 drums
        W = 0.2
    elif num_drums == 1:
        Rho_d0 = -0.033085599 + 0.0071 + 0.0082
        Reactivity_per_degree = 2.77e-5
        u0 = 170  # 1 drum
        W = 0.49
    else:
        Rho_d0 = -0.033085599
        Reactivity_per_degree = 26.11e-5
        u0 = 77.56  # Default to 8 drums
        W = 0.04
    return Rho_d0, Reactivity_per_degree, u0, W

class Reactor:
    def __init__(self, num_drums, initial_state, parameters):
        """
        Initializes the reactor simulation.
        
        Args:
          num_drums: Number of control drums.
          initial_state: Initial state vector (length 12).
          parameters: Dictionary containing reactor parameters (Rho_d0, Reactivity_per_degree,
                      Xe0, I0, Pi, and optionally Tf0 and Tm0).
        """
        self.num_drums = num_drums
        self.params = parameters
        self.x = initial_state.copy()  # current state (not used internally in simulation loop)
    
    def reactor_dae(self, t, x, u):
        """Compute the reactor differential–algebraic equations (DAE) and reactivity."""
        Rho_d0 = self.params['Rho_d0']
        Reactivity_per_degree = self.params['Reactivity_per_degree']
        Xe0 = self.params['Xe0']
        I0 = self.params['I0']
        Pi = self.params['Pi']
        
        # Constants
        Sig_x   = 2.65e-22
        yi      = 0.061
        yx      = 0.002
        lamda_x = 2.09e-5
        lamda_I = 2.87e-5
        Sum_f   = 0.3358

        l       = 1.68e-3
        beta    = 0.0048
        beta_1  = 1.42481e-4
        beta_2  = 9.24281e-4
        beta_3  = 7.79956e-4
        beta_4  = 2.06583e-3
        beta_5  = 6.71175e-4
        beta_6  = 2.17806e-4
        Lamda_1 = 1.272e-2
        Lamda_2 = 3.174e-2
        Lamda_3 = 1.160e-1
        Lamda_4 = 3.110e-1
        Lamda_5 = 1.400e+0
        Lamda_6 = 3.870e+0

        cp_f    = 977
        cp_m    = 1697
        cp_c    = 5188.6
        M_f     = 2002
        M_m     = 11573
        M_c     = 500
        mu_f    = M_f * cp_f
        mu_m    = M_m * cp_m
        mu_c    = M_c * cp_c
        f_f     = 0.96
        P_0     = 22e6
        Tf0     = self.params.get('Tf0', 1105)
        Tm0     = self.params.get('Tm0', 1087)
        T_in    = 864
        T_out   = 1106
        Tc0     = (T_in + T_out) / 2
        K_fm    = f_f * P_0 / (Tf0 - Tm0)
        K_mc    = P_0 / (Tm0 - Tc0)
        M_dot   = 1.75e+1
        alpha_f = -2.875e-5
        alpha_m = -3.696e-5
        alpha_c = 0.0

        # Unpack state vector x (assumed to have 12 elements)
        n_r = x[0]
        Cr1 = x[1]
        Cr2 = x[2]
        Cr3 = x[3]
        Cr4 = x[4]
        Cr5 = x[5]
        Cr6 = x[6]
        X   = x[7]
        I   = x[8]
        Tf  = x[9]
        Tm  = x[10]
        Tc  = x[11]
        
        Rho_d1 = Rho_d0 + u * Reactivity_per_degree

        if self.num_drums in [8, 2, 1]:
            rho = (Rho_d1 + alpha_f * (Tf - Tf0) + alpha_c * (Tc - Tc0) +
                   alpha_m * (Tm - Tm0) - Sig_x * (X - Xe0) / Sum_f)
        elif self.num_drums == 4:
            rho = (Rho_d1 + alpha_f * (Tf - 900.42) + alpha_c * (Tc - 888.261) +
                   alpha_m * (Tm - 898.261) - Sig_x * (X - Xe0) / Sum_f)
        else:
            raise ValueError("Invalid number of drums specified.")
        
        dx = np.zeros(12)
        dx[0] = ((rho - beta) / l) * n_r + (beta_1 / l) * Cr1 + (beta_2 / l) * Cr2 + \
                (beta_3 / l) * Cr3 + (beta_4 / l) * Cr4 + (beta_5 / l) * Cr5 + (beta_6 / l) * Cr6
        dx[1] = Lamda_1 * n_r - Lamda_1 * Cr1
        dx[2] = Lamda_2 * n_r - Lamda_2 * Cr2
        dx[3] = Lamda_3 * n_r - Lamda_3 * Cr3
        dx[4] = Lamda_4 * n_r - Lamda_4 * Cr4
        dx[5] = Lamda_5 * n_r - Lamda_5 * Cr5
        dx[6] = Lamda_6 * n_r - Lamda_6 * Cr6
        dx[7] = 0.002 * Sum_f * Pi + lamda_I * I - Sig_x * X * Pi - lamda_x * X
        dx[8] = yi * Sum_f * Pi - lamda_I * I
        dx[9] = f_f * P_0 / mu_f * n_r - K_fm / mu_f * (Tf - Tc)
        dx[10] = (1 - f_f) * P_0 / mu_m * n_r + (K_fm * (Tf - Tm) - K_mc * (Tm - Tc)) / mu_m
        dx[11] = K_mc * (Tm - Tc) / mu_c - 2 * M_dot * cp_c * (Tc - T_in) / mu_c
        
        return dx, rho


