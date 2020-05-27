"""Classes defining controllers for simulated aircraft."""

from abc import abstractmethod
import pynput
import inputs
from math import degrees, radians
import numpy as np
import copy
import multiprocessing as mp

class BaseController:
    """An abstract aircraft controller class.
    """

    def __init__(self, control_dict, quit_flag, view_flag, pause_flag, data_flag, enable_interface, control_output):

        # Initialize
        self._UI_inputs = {}
        self._keys_pressed = []
        self._input_dict = control_dict

        # Store flags
        self._data_flag = data_flag
        self._view_flag = view_flag
        self._quit_flag = quit_flag
        self._pause_flag = pause_flag

        # Set up user interface
        self._enable_interface = enable_interface
        if self._enable_interface:
            # Key press listener function
            def on_press(key):

                # Get key
                try:
                    k = key.char
                except:
                    k = key.name

                # Check action

                # Toggle flight data
                if k == 'i':
                    self._data_flag.value = not self._data_flag.value

                # Toggle pause
                elif k == 'p':
                    self._pause_flag.value = not self._pause_flag.value

                # Quit
                elif k == 'q':
                    self._quit_flag.value = not self._quit_flag.value

                # Toggle view
                elif k == 'space':
                    self._view_flag.value = (self._view_flag.value+1)%3

                # Store other keystroke
                elif k in ['w', 's', 'a', 'd', 'left', 'right', 'up', 'down']:
                    self._keys_pressed.append(k)

            # Key release listener function
            def on_release(key):

                # Get key
                try:
                    k = key.char
                except:
                    k = key.name

                # Remove those from the list
                if k in ['w', 's', 'a', 'd', 'left', 'right', 'up', 'down'] and k in self._keys_pressed:
                    self._keys_pressed = list(filter(lambda a: a != k, self._keys_pressed))

            # Initialize keyboard listener
            self._keyboard_listener = pynput.keyboard.Listener(on_press=on_press, on_release=on_release)
            self._keyboard_listener.start()

        # Get control names
        self._controls = []
        for key, value in self._input_dict.items():
            self._controls.append(key)
        self._num_controls = len(self._controls)

        # Initialize storage
        if control_output is not None:

            # Check for csv
            if ".csv" not in control_output:
                raise IOError("Control output file must be .csv")

            # Open file
            self._write_controls = True
            self._control_output = open(control_output, 'w')
        else:
            self._write_controls = False

        # Store mapping
        if control_output is not None or isinstance(self, TimeSequenceController):
            self._column_mapping = {}
            self._output_cols = [""]*self._num_controls
            for key, value in self._input_dict.items():
                try:
                    self._column_mapping[key] = value["column_index"]
                    self._output_cols[self._column_mapping[key]-1] = key
                except KeyError:
                    if control_output is not None:
                        raise IOError("'column_index' must be specified for each control if the controls are to be output.")
                    else:
                        raise IOError("'column_index' must be specified for each control if a time-sequence controller is used.")


    def __del__(self):
        if self._enable_interface:
            self._keyboard_listener.stop()
        if self._write_controls:
            self._control_output.close()


    def get_control_names(self):
        """Returns the names of the controls handled by this controller."""
        return self._controls


    def get_input(self):
        """Returns a dictionary of inputs from the user for controlling pause, view, etc."""

        inputs = copy.deepcopy(self._UI_inputs)
        self._UI_inputs= {}
        return inputs


    def output_controls(self, t, control_dict):
        # Writes controls to csv
        if self._write_controls:
            line = ["{0}".format(t)]
            for i in range(self._num_controls):
                line.append(",{0}".format(control_dict[self._output_cols[i]]))
            line.append("\n")
            self._control_output.write("".join(line))

    
    @abstractmethod
    def get_control(self, t, state_vec, prev_controls):
        """ABSTRACT METHOD. Returns the controls based on the inputted state.

        Parameters
        ----------
        t : float
            Time index in seconds.

        state_vec : list
            State vector of the aircraft being controlled. It is given in the form

                [u, v, w, p, q, r, x, y, z, e0, ex, ey, ez]

            where u, v, and w are the body-fixed velocity components, p, q, and r are
            the body-fixed angular velocity components, x, y, and z are the Earth-fixed
            position coordinates, and e0, ex, ey, and ez are the quaternion encoding
            a rotation from the local NED frame to the body-fixed frame.

        prev_controls : dict
            Previous control values.

        Returns
        -------
        control_state : dict
            Updated control state.
        """
        pass


    def get_limits(self):
        """Returns the control limits for this controller."""

        # This will only work if the limits are defined
        try:
            limits = {}
            for name in self._controls:
                
                # Get max deflections for angular controls
                if self._angular_control[name]:
                    limits[name] = (-self._control_limits[name], self._control_limits[name])

                # Other controls just go between 0.0 and 1.0
                else:
                    limits[name] = (0.0, 1.0)

            return limits

        except:
            return None


class NoController(BaseController):
    """A controller that holds constant the initial controls.

    Parameters
    ----------
    control_dict : dict
        A dictionary of control names and specifications.
    """

    def __init__(self, control_dict, quit_flag, view_flag, pause_flag, data_flag, enable_interface, control_output):
        super().__init__(control_dict, quit_flag, view_flag, pause_flag, data_flag, enable_interface, control_output)

    def get_control(self, t, state_vec, prev_controls):
        return prev_controls


class JoystickController(BaseController):
    """A controller for controlling a 4-channel aircraft using a standard joystick.

    Parameters
    ----------
    control_dict : dict
        A dictionary of control names and specifications.
    """

    def __init__(self, control_dict, quit_flag, view_flag, pause_flag, data_flag, enable_interface, control_output):
        super().__init__(control_dict, quit_flag, view_flag, pause_flag, data_flag, enable_interface, control_output)

        # Check for device
        self._avail_pads = inputs.devices.gamepads
        if len(self._avail_pads) == 0:
            raise RuntimeError("Couldn't find any joysticks!")
        elif len(self._avail_pads) > 1:
            raise RuntimeError("More than one joystick detected!")

        # Set off listener
        self._manager = mp.Manager()
        self._joy_def = self._manager.list()
        self._joy_def[:] = [0.0]*4
        self._joy_listener = mp.Process(target=joystick_listener, args=(self._joy_def, self._quit_flag))
        self._joy_listener.start()

        # Get mapping and limits
        self._axis_mapping = {}
        self._control_limits = {}
        self._angular_control = {} # True for angular deflection, False for 0 to 1.
        for key, value in control_dict.items():

            # See if limits have been defined
            limits = value.get("max_deflection", None)
            if limits is not None: # The limits are defined
                self._control_limits[key] = limits
                self._angular_control[key] = True
            else:
                self._angular_control[key] = False
            
            # Get the mapping
            self._axis_mapping[key] = value["input_axis"]

        # Set variable for knowing if the user has perturbed from the trim state yet
        self._perturbed_set = False
        self._perturbed = False
        self._joy_init = [0.0]*4


    def get_control(self, t, state_vec, prev_controls):
        """Returns the controls based on the inputted state and keyboard/joystick inputs.

        Parameters
        ----------
        state_vec : list
            State vector of the entity being controlled.

        prev_controls : dict
            Previous control values.

        Returns
        -------
        controls : dict
            Dictionary of controls.
        """

        # Set perturbation condition
        if not self._perturbed_set:
            self._joy_init[:] = self._joy_def[:]
            self._perturbed_set = True

        # Check if we're perturbed from the start control set
        if not self._perturbed:
            if (np.array(self._joy_def) != np.array(self._joy_init)).any():
                self._perturbed = True
            else:
                return prev_controls # No point in parsing things if nothing's changed

        # Parse new controls
        control_state = {}
        setting_list = []
        for name in self._controls:
            if self._angular_control[name]:
                setting = (self._joy_def[self._axis_mapping[name]]**3)*-self._control_limits[name]
            else:
                setting = (-self._joy_def[self._axis_mapping[name]]+1.)*0.5
            control_state[name] = setting
            setting_list.append(setting)

        return control_state


class KeyboardController(BaseController):
    """A controller for controlling an aircraft with ailerons, elevators, and rudder, and a throttle using a standard keyboard.

    Parameters
    ----------
    control_dict : dict
        A dictionary of control names and specifications.
    """

    def __init__(self, control_dict, quit_flag, view_flag, pause_flag, data_flag, enable_interface, control_output):
        super().__init__(control_dict, quit_flag, view_flag, pause_flag, data_flag, enable_interface, control_output)

        # Initialize user inputs
        self._UP = False
        self._DOWN = False
        self._RIGHT = False
        self._LEFT = False
        self._WW = False
        self._SS = False
        self._AA = False
        self._DD = False

        # Get mapping and limits
        self._axis_mapping = {}
        self._control_limits = {}
        self._angular_control = {} # True for angular deflection, False for 0 to 1.
        for key, value in control_dict.items():

            # See if limits have been defined
            limits = value.get("max_deflection", None)
            if limits is not None: # The limits are defined
                self._control_limits[key] = limits
                self._angular_control[key] = True
            else:
                self._angular_control[key] = False
            
            # Get the mapping
            self._axis_mapping[key] = value["input_axis"]
            
            # Store reverse mapping
            self._control_reverse_mapping = [0]*4
            for key, value in self._axis_mapping.items():
                self._control_reverse_mapping[value] = key

            # Store control names
            self._controls.append(key)

        # Set variable for knowing if the user has perturbed from the trim state yet
        self._perturbed = False


    def get_control(self, t, state_vec, prev_controls):
        """Returns the controls based on the inputted state and keyboard/joystick inputs.

        Parameters
        ----------
        state_vec : list
            State vector of the entity being controlled.

        prev_controls : dict
            Previous control values.

        Returns
        -------
        controls : dict
            Dictionary of controls.
        """

        # Check for perturbation
        if not self._perturbed and len(self._keys_pressed) > 0:
            self._perturbed = True

        if self._perturbed:
            # Parse new controls
            control_state = {}
            for i in range(4):
                name = self._control_reverse_mapping[i]
                defl = 0.0

                # Get axis input
                if i == 0: # Input roll axis
                    if 'left' in self._keys_pressed and not 'right' in self._keys_pressed:
                        defl = 1.0
                    elif not 'left' in self._keys_pressed and 'right' in self._keys_pressed:
                        defl = -1.0

                elif i == 1: # Input pitch axis
                    if 'up' in self._keys_pressed and not 'down' in self._keys_pressed:
                        defl = 1.0
                    elif not 'up' in self._keys_pressed and 'down' in self._keys_pressed:
                        defl = -1.0

                elif i == 2: # Input yaw axis
                    if 'a' in self._keys_pressed and not 'd' in self._keys_pressed:
                        defl = 1.0
                    elif not 'a' in self._keys_pressed and 'd' in self._keys_pressed:
                        defl = -1.0

                else: # Input throttle axis
                    if 'w' in self._keys_pressed and not 's' in self._keys_pressed:
                        defl = 1.0
                    elif not 'w' in self._keys_pressed and 's' in self._keys_pressed:
                        defl = -1.0

                # Apply deflection
                if self._angular_control[name]:
                    sensitivity = 0.01
                    control_state[name] = min(self._control_limits[name], max(prev_controls[name]+sensitivity*defl, -self._control_limits[name]))
                else:
                    sensitivity = 0.001
                    control_state[name] = min(1.0, max(prev_controls[name]+defl*sensitivity, 0.0))

        else: # Otherwise, send back the previous controls
            control_state = copy.deepcopy(prev_controls)

        return control_state


class TimeSequenceController(BaseController):
    """A controller for controlling an aircraft with ailerons, elevators, and rudder, and a throttle using a standard keyboard.

    Parameters
    ----------
    control_dict : dict
        A dictionary of control names and specifications.
    """


    def __init__(self, control_dict, quit_flag, view_flag, pause_flag, data_flag, enable_interface, control_output):
        super().__init__(control_dict, quit_flag, view_flag, pause_flag, data_flag, enable_interface, control_output)

        self._input_dict = control_dict


    def read_control_file(self, control_file):
        """Reads in a time sequence input file of control settings."""

        # Read in file
        self._control_data = np.genfromtxt(control_file, delimiter=',')

        # Get final time
        self._t_end = np.max(self._control_data[:,0])


    def get_control(self, t, state_vec, prev_controls):
        """Returns the controls based on the inputted state.

        Parameters
        ----------
        t : float
            Time index

        state_vec : list
            State vector of the entity being controlled.

        prev_controls : dict
            Previous control values.
        """

        # Get control
        control_state = {}
        for name in self._controls:
            i = self._column_mapping[name]
            default = prev_controls[name]
            control_state[name] = np.interp(t, self._control_data[:,0], self._control_data[:,i], left=default, right=default)

        # Check if we've reached the end
        if t > self._t_end:
            self._quit_flag.value = 1

        return control_state


def joystick_listener(axes_def, quit_flag):
    """Listens to the joystick input and posts latest values to the manager list."""

    # While the game is still going
    while not quit_flag.value:

        # Wait for events
        events = inputs.get_gamepad()

        # Parse events
        try:
            for event in events:
                if event.ev_type == 'Absolute':

                    # Roll axis
                    if event.code == 'ABS_X':
                        axes_def[0] = event.state/511.5-1.0

                    # Pitch axis
                    elif event.code == 'ABS_Y':
                        axes_def[1] = event.state/511.5-1.0

                    # Yaw axis
                    elif event.code == 'ABS_RZ':
                        axes_def[2] = event.state/127.5-1.0

                    # Throttle axis
                    elif event.code == 'ABS_THROTTLE':
                        axes_def[3] = event.state/127.5-1.0

        except BrokenPipeError:
            return

        except FileNotFoundError:
            return

        except ConnectionResetError:
            return

    return