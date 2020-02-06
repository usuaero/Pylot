# User Interface
Pylot is run from the command line using the "-m" option. For example

    $ python -m pylot example_input.json

## Controlling the Aircraft
The aircraft can be controlled in real-time using either a joystick or the keyboard. This is specified in the [input file](creating_input_files). Please note that the specific function of the keyboard/joystick inputs is determined by how the input files are configured. The mapping from the input axes or channels to the aircraft controls is up to the user.

### General Interface Controls
The following keyboard keys affect the simulator interface, regardless of the controller specified.

| Key   | Action                        |
| ----- | ----------------------------- |
| i     | Toggle flight data display    |
| p     | Pause                         |
| SPACE | Cycle through camera views    |
| q     | Exit simulation               |

As explained in [Creating Input Files](creating_input_files), the user can set whether these keys are active. Please note that the pynput module will capture keystrokes whether the simulator window is active or not (i.e. if the sim is running in one window and you're editing a document in another, your keystrokes will affect the simulation). The default behavior is for these keys to be active only if the graphics are running.

### Keyboard Control
The keyboard controls four independent axes or channels using the key pairs A-D, W-S, UP-DOWN, and LEFT-RIGHT. Holding down any of these keys results in an incremental change in the respective control. If all keys are released, the current control state is maintained.

The example files have all been configured such that UP-DOWN controls elevator deflection, LEFT-RIGHT controls aileron deflection, W-S controls throttle, and A-D controls rudder deflection.

### Joystick Control
Joystick control is more ambiguous than keyboard control due to the variety of possible joysticks being used. The developers have used a Logitech Extreme 3D pro joystick, though any USB joystick should serve equally well. You may have to experiment with your joystick to determine which movement of the joystick corresponds to which axis.

The example files have all been configured such that the pitch axis controls elevator deflection, the roll axis controls aileron deflection, the yaw axis controls rudder deflection, and the auxiliary axis controls throttle. Again, this is for the developers' joystick and may not necessarily be the same with your setup. however, changing this configuration to your setup should be straightforward.