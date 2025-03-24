#!/usr/bin/env python3
"""
This module defines the controller classes.
It contains two controllers:
  - STCController: Implements a supertwisting controller.
  - PIDController: Implements a PID controller with anti-windup and derivative filtering.
"""

import numpy as np

class STCController:
    def __init__(self, lambda_, W, T_C, max_val, min_val, max_rate, u0):
        self.lambda_ = lambda_
        self.W = W
        self.T_C = T_C
        self.max_val = max_val
        self.min_val = min_val
        self.max_rate = max_rate
        self.u = u0  # internal controller state (analogous to persistent variable)
        self.err_prev = 0
        self.t_prev = 0
        self.deriv_prev = 0
        self.command_sat_prev = u0
        self.command_prev = u0

    def update(self, t, measurement, setpoint):
        """Compute the new control command at time t."""
        err = setpoint - measurement
        T = t - self.t_prev
        denom = T + self.T_C if (T + self.T_C) != 0 else self.T_C
        deriv_filt = (err - self.err_prev + self.T_C * self.deriv_prev) / denom
        self.err_prev = err
        self.deriv_prev = deriv_filt
        s = err + deriv_filt
        self.u = self.u + self.W * np.sign(s) * T
        command = self.lambda_ * np.sqrt(abs(s)) * np.sign(s) + self.u

        # Saturation limits
        if command > self.max_val:
            command_sat = self.max_val
        elif command < self.min_val:
            command_sat = self.min_val
        else:
            command_sat = command

        # Rate limiting
        if command_sat > self.command_sat_prev + self.max_rate * T:
            command_sat = self.command_sat_prev + self.max_rate * T
        elif command_sat < self.command_sat_prev - self.max_rate * T:
            command_sat = self.command_sat_prev - self.max_rate * T

        self.command_sat_prev = command_sat
        self.t_prev = t
        return command_sat

class PIDController:
    def __init__(self, Kp, Ki, Kd, Kaw, T_C, max_val, min_val, max_rate, u0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.Kaw = Kaw
        self.T_C = T_C
        self.max_val = max_val
        self.min_val = min_val
        self.max_rate = max_rate
        self.integral = u0
        self.err_prev = 0
        self.t_prev = 0
        self.deriv_prev = 0
        self.command_prev = u0
        self.command_sat_prev = u0

    def update(self, t, measurement, setpoint):
        """Compute the new control command at time t."""
        err = setpoint - measurement
        T_period = t - self.t_prev
        self.t_prev = t

        # Update the integral term with anti-windup correction
        self.integral = self.integral + self.Ki * err * T_period + \
                        self.Kaw * (self.command_sat_prev - self.command_prev) * T_period

        denom = T_period + self.T_C if (T_period + self.T_C) != 0 else self.T_C
        deriv_filt = (err - self.err_prev + self.T_C * self.deriv_prev) / denom
        self.err_prev = err
        self.deriv_prev = deriv_filt

        command = self.Kp * err + self.integral + self.Kd * deriv_filt
        self.command_prev = command

        # Saturation limits
        if command > self.max_val:
            command_sat = self.max_val
        elif command < self.min_val:
            command_sat = self.min_val
        else:
            command_sat = command

        # Rate limiting
        if command_sat > self.command_sat_prev + self.max_rate * T_period:
            command_sat = self.command_sat_prev + self.max_rate * T_period
        elif command_sat < self.command_sat_prev - self.max_rate * T_period:
            command_sat = self.command_sat_prev - self.max_rate * T_period

        self.command_sat_prev = command_sat
        return command_sat


