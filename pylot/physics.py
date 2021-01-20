"""Defines the physics engine for Pylot."""

import time
import copy
import json

import numpy as np

from pylot.helpers import import_value
from pylot.airplanes import MachUpXAirplane, LinearizedAirplane
from pylot.integrators import RK4Integrator, ABM4Integrator


def run_physics(input_dict, units, graphics_dict, graphics_ready_flag, game_over_flag, quit_flag, view_flag, pause_flag, data_flag, state_manager, control_manager):
    """Runs the physics on a separate process."""
    # Note that this was a member function of Simulator, but bound methods
    # cannot be passed as the target to multiprocessing.Process() on
    # Windows. This irks me Bill...

    # Get simulation options
    sim_dict = input_dict["simulation"]
    real_time = sim_dict.get("real_time", True)
    t_start = sim_dict.get("start_time", 0.0)
    t_final = sim_dict.get("final_time", np.inf)
    render_graphics = sim_dict.get("enable_graphics", False)
    enable_interface = sim_dict.get("enable_interface", render_graphics)
    if not real_time:
        dt = sim_dict.get("timestep", 0.05)

    # Load aircraft
    aircraft = load_aircraft(input_dict, units, quit_flag, view_flag, pause_flag, data_flag, enable_interface)

    # Initialize integrator
    if input_dict["simulation"].get("integrator", "RK4") == "RK4":
        integrator = RK4Integrator(aircraft)
    else:
        integrator = ABM4Integrator(aircraft)

    # Pass airplane graphics information to parent process
    if render_graphics:
        aircraft_graphics_info = aircraft.get_graphics_info()
        for key, value in aircraft_graphics_info.items():
            graphics_dict[key] = value

        # Give initial state to graphics
        graphics_dict["position"] = aircraft.y[6:9]
        graphics_dict["orientation"] = aircraft.y[9:]

        # Wait for graphics to load
        while not graphics_ready_flag.value:
            continue

    # Initial computer time
    t0 = time.time()

    # If we're running real time, get an initial guess for how long each sim step is going to take
    if real_time:
        integrator.step(t_start, 0.0, store=False)
        aircraft.normalize()
        aircraft.output_state(t_start)
        aircraft.controller.output_controls(t_start, aircraft.controls)
        t1 = time.time()
        dt = t1-t0
        t0 = t1

    # Otherwise, still perform the necessary output actions
    else:
        aircraft.output_state(t_start)
        aircraft.controller.output_controls(t_start, aircraft.controls)


    # Initialize simulation time index
    t = copy.copy(t_start)

    # Simulation loop
    first = True
    while t <= t_final and not (quit_flag.value or game_over_flag.value):

        # Integrate
        if first or integrator=="RK4":
            integrator.step(t, dt)

        # Normalize
        aircraft.normalize()

        # Step in time
        if real_time:
            t1 = time.time()
            dt = t1-t0
            t0 = t1
        t += dt

        # Write output
        aircraft.output_state(t)
        aircraft.controller.output_controls(t, aircraft.controls)

        # Handle graphics only things
        if render_graphics:

            # Pass information to graphics
            state_manager[:13] = aircraft.y[:]
            state_manager[13] = dt
            state_manager[14] = t
            state_manager[15] = time.time()
            for key, value in aircraft.controls.items():
                control_manager[key] = value

            while pause_flag.value and not quit_flag.value:

                # The physics isn't stepping...
                state_manager[13] = 0.0

            else:
                if real_time:
                    t0 = time.time() # So as to not throw off the integration

    # If we exit the loop due to a timeout, let the graphics know we're done
    if t > t_final:
        quit_flag.value = 1

    aircraft.finalize()


def load_aircraft(input_dict, units, quit_flag, view_flag, pause_flag, data_flag, enable_interface):
    # Loads the aircraft from the input file

    # Read in aircraft input
    aircraft_name = input_dict["aircraft"]["name"]
    aircraft_file = input_dict["aircraft"]["file"]

    if isinstance(aircraft_file, str):
        with open(aircraft_file, 'r') as aircraft_file_handle:
            aircraft_dict = json.load(aircraft_file_handle)
    elif isinstance(aircraft_file, dict):
        aircraft_dict = copy.copy(aircraft_file)

    # Get density model, controller, and output file
    density = import_value("density", input_dict.get("atmosphere", {}), units, [0.0023769, "slug/ft^3"])

    # Linear aircraft
    if aircraft_dict["aero_model"]["type"] == "linearized_coefficients":
        aircraft = LinearizedAirplane(aircraft_name, aircraft_dict, density, units, input_dict["aircraft"], quit_flag, view_flag, pause_flag, data_flag, enable_interface)
    
    # MachUpX aircraft
    else:
        aircraft = MachUpXAirplane(aircraft_name, aircraft_dict, density, units, input_dict["aircraft"], quit_flag, view_flag, pause_flag, data_flag, enable_interface)

    return aircraft


def RK4(aircraft, t, dt):
    """Performs Runge-Kutta integration for the given aircraft.

    Parameters
    ----------
    aircraft : BaseAircraft
        Aircraft to integrate the state of.

    t : float
        Initial time.

    dt : float
        Time step.

    """
    y0 = copy.deepcopy(aircraft.y)

    # Determine k0
    k0 = aircraft.dy_dt(t)

    # Determine k1
    aircraft.y = y0+0.5*dt*k0 
    k1 = aircraft.dy_dt(t+0.5*dt)

    # Determine k2
    aircraft.y = y0+0.5*dt*k1
    k2 = aircraft.dy_dt(t+0.5*dt)

    # Determine k3
    aircraft.y = y0+dt*k2
    k3 = aircraft.dy_dt(t+dt)

    # Calculate y
    aircraft.y = y0+0.16666666666666666666667*(k0+2*k1+2*k2+k3)*dt
