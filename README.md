# NPIC_Microreactor_Load_following_Reactivity_Control_Second_order_Sliding_Mode_Control

---

# Microreactor Load-following Reactivity Control with Second-order Sliding Mode Control

This repository contains simulation code for applying a second-order super-twisting sliding mode control to microreactor load-following reactivity control, as described in our paper. The simulations are available in both MATLAB and Python.

## Overview

The project provides two simulation environments:
- **MATLAB**
- **Python**

Each environment is set up so you can easily run the simulation.

## MATLAB Version

The MATLAB simulation consists of two files:
- **STC.m**
- **PID.m**

### How to Run:
1. Open MATLAB and navigate to the repository folder.
2. Run either `STC.m` or `PID.m` individually to generate the simulation results.
3. **Note:** You must select the number of control drums used during the simulation by selecting num_drums to be 8,4,2,1 within each file.

## Python Version

The Python simulation is organized into a folder containing the following files:
1. `run.py`
2. `controller.py`
3. `plot.py`
4. `reactor.py`

### How to Run:
1. Ensure you have Python installed on your computer.
2. Install the required package by running:
   ```bash
   pip install matplotlib 
   ```
   *(Installing matplotlib will also install numpy as a dependency.)*
3. Open `run.py` in your preferred editor.
4. Customize the simulation by editing the variables `controller_type` and `num_drums` in `run.py` to change the controller settings and the number of control drums.
5. Run the simulation by executing:
   ```bash
   python run.py
   ```
