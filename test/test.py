import pylot
import json
import numpy as np

if __name__=="__main__":

    # Load input
    input_file = "test/state_input.json"
    with open(input_file, 'r') as input_handle:
        input_dict = json.load(input_handle)

    # Record first flight with joystick
    input_dict["aircraft"]["controller"] = "joystick"
    input_dict["aircraft"]["control_output"] = "original_controls.csv"
    sim = pylot.Simulator(input_dict)
    sim.run_sim()

    # Run auto sim with standard density
    standard_file = "standard_states.txt"
    input_dict["simulation"]["real_time"] = False
    input_dict["simulation"]["enable_graphics"] = False
    input_dict["simulation"]["timestep"] = 0.01
    input_dict["atmosphere"]["density"] = "standard"
    input_dict["aircraft"]["controller"] = "original_controls.csv"
    input_dict["aircraft"].pop("control_output", None)
    input_dict["aircraft"]["state_output"] = standard_file
    sim = pylot.Simulator(input_dict)
    sim.run_sim()

    # Run auto sim with constant density
    constant_file = "constant_states.txt"
    input_dict["atmosphere"]["density"] = 0.0023769
    input_dict["aircraft"]["controller"] = "original_controls.csv"
    input_dict["aircraft"].pop("control_output", None)
    input_dict["aircraft"]["state_output"] = constant_file
    sim = pylot.Simulator(input_dict)
    sim.run_sim()

    # Read in state files
    with open(standard_file, 'r') as standard_handle:
        standard_states = np.genfromtxt(standard_handle, skip_header=1)
    with open(constant_file, 'r') as constant_handle:
        constant_states = np.genfromtxt(constant_handle, skip_header=1)

    # Print out sizes and final times
    print("Standard: {0}".format(standard_states.shape))
    print("    Final time: {0} s".format(standard_states[-1,0]))
    print("Constant: {0}".format(constant_states.shape))
    print("    Final time: {0} s".format(constant_states[-1,0]))