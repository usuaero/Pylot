import numpy as np
import time
import copy
from .airplanes import MachUpXAirplane, LinearizedAirplane
import json
from .helpers import import_value

def run_physics(input_dict, units, graphics_dict, graphics_ready_flag, game_over_flag, quit_flag, view_flag, pause_flag, data_flag, state_manager, control_manager):
    """Runs the physics on a separate process."""
    # Note that this was a member function of Simulator, but bound methods
    # cannot be passed as the target to multiprocessing.Process() on
    # Windows. This irks me Bill...

    # Get simulation params
    real_time = input_dict["simulation"].get("real_time", True)
    t_start = input_dict["simulation"].get("start_time", 0.0)
    t_final = input_dict["simulation"].get("final_time", np.inf)
    render_graphics = input_dict["simulation"].get("enable_graphics", False)
    enable_interface = input_dict["simulation"].get("enable_interface", render_graphics)
    if not real_time:
        dt = input_dict["simulation"].get("dt", 0.01)

    # Load aircraft
    aircraft = load_aircraft(input_dict, units, quit_flag, view_flag, pause_flag, data_flag, enable_interface)

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

    # Get an initial guess for how long each sim step is going to take
    t0 = time.time()
    if real_time:
        RK4(aircraft, t_start, 0.0)
        aircraft.normalize()
        aircraft.output_state(t_start)
        aircraft.controller.output_controls(t_start, aircraft.controls)
        t1 = time.time()
        dt = t1-t0
        t0 = t1
    else:
        aircraft.output_state(t0)

    t = copy.copy(t_start)

    # Simulation loop
    while t <= t_final and not (quit_flag.value or game_over_flag.value):

        # Integrate
        RK4(aircraft, t, dt)

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


def load_aircraft(input_dict, units, quit_flag, view_flag, pause_flag, data_flag, enable_interface):
    # Loads the aircraft from the input file

    # Read in aircraft input
    aircraft_name = input_dict["aircraft"]["name"]
    aircraft_file = input_dict["aircraft"]["file"]
    with open(aircraft_file, 'r') as aircraft_file_handle:
        aircraft_dict = json.load(aircraft_file_handle)

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