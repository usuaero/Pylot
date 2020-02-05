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

    def __init__(self, quit_flag, view_flag, pause_flag, data_flag):

        # Initialize controls
        self._controls = []
        self._inputs = {}
        self._control_keys = []

        # Store flags
        self._data_flag = data_flag
        self._view_flag = view_flag
        self._quit_flag = quit_flag
        self._pause_flag = pause_flag

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
                self._control_keys.append(k)

        # Key release listener function
        def on_release(key):

            # Get key
            try:
                k = key.char
            except:
                k = key.name

            # Remove those from the list
            if k in ['w', 's', 'a', 'd', 'left', 'right', 'up', 'down'] and k in self._control_keys:
                self._control_keys = list(filter(lambda a: a != k, self._control_keys))

        # Initialize keyboard listener
        self._keyboard_listener = pynput.keyboard.Listener(on_press=on_press, on_release=on_release)
        self._keyboard_listener.start()


    def __del__(self):
        self._keyboard_listener.stop()


    def get_control_names(self):
        """Returns the names of the controls handled by this controller."""
        return self._controls


    def get_input(self):
        """Returns a dictionary of inputs from the user for controlling pause, view, etc."""

        inputs = copy.deepcopy(self._inputs)
        self._inputs= {}
        return inputs

    
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

    def __init__(self, control_dict, quit_flag, view_flag, pause_flag, data_flag):
        super().__init__(quit_flag, view_flag, pause_flag, data_flag)

        # Get control names
        for key in list(control_dict.keys()):
            self._controls.append(key)

    def get_control(self, t, state_vec, prev_controls):
        return prev_controls


class JoystickController(BaseController):
    """A controller for controlling a 4-channel aircraft using a standard joystick.

    Parameters
    ----------
    control_dict : dict
        A dictionary of control names and specifications.
    """

    def __init__(self, control_dict, quit_flag, view_flag, pause_flag, data_flag):
        super().__init__(quit_flag, view_flag, pause_flag, data_flag)

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
        self._control_mapping = {}
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
            self._control_mapping[key] = value["input_axis"]

            # Store control names
            self._controls.append(key)

        # Set variable for knowing if the user has perturbed from the trim state yet
        self._perturbed = False
        self._joy_init = [0.0]*4
        self._joy_init[:] = self._joy_def[:]

        # For me to create a control input csv for testing
        self._store_input = False
        if self._store_input:
            self._storage_file = open("control_input.csv", 'w')


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
                setting = (self._joy_def[self._control_mapping[name]]**3)*-self._control_limits[name]
            else:
                setting = (-self._joy_def[self._control_mapping[name]]+1.)*0.5
            control_state[name] = setting
            setting_list.append(setting)

        if self._store_input:
            line = "{0},{1},{2},{3},{4}\n".format(t, *setting_list)
            self._storage_file.write(line)

        return control_state


    def __del__(self):
        if self._store_input:
            self._storage_file.close()


class KeyboardController(BaseController):
    """A controller for controlling an aircraft with ailerons, elevators, and rudder, and a throttle using a standard keyboard.

    Parameters
    ----------
    control_dict : dict
        A dictionary of control names and specifications.
    """

    def __init__(self, control_dict, quit_flag, view_flag, pause_flag, data_flag):
        super().__init__(quit_flag, view_flag, pause_flag, data_flag)

        # Initialize user inputs
        self._UP = False
        self._DOWN = False
        self._RIGHT = False
        self._LEFT = False
        self._WW = False
        self._SS = False
        self._AA = False
        self._DD = False
        self._RESET= False

        # Get mapping and limits
        self._control_mapping = {}
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
            self._control_mapping[key] = value["input_axis"]
            
            # Store reverse mapping
            self._control_reverse_mapping = [0]*4
            for key, value in self._control_mapping.items():
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
        if not self._perturbed and len(self._control_keys) > 0:
            self._perturbed = True

        if self._perturbed:
            # Parse new controls
            control_state = {}
            for i in range(4):
                name = self._control_reverse_mapping[i]
                defl = 0.0

                # Get axis input
                if i == 0: # Input roll axis
                    if 'left' in self._control_keys and not 'right' in self._control_keys:
                        defl = 1.0
                    elif not 'left' in self._control_keys and 'right' in self._control_keys:
                        defl = -1.0

                elif i == 1: # Input pitch axis
                    if 'up' in self._control_keys and not 'down' in self._control_keys:
                        defl = 1.0
                    elif not 'up' in self._control_keys and 'down' in self._control_keys:
                        defl = -1.0

                elif i == 2: # Input yaw axis
                    if 'a' in self._control_keys and not 'd' in self._control_keys:
                        defl = 1.0
                    elif not 'a' in self._control_keys and 'd' in self._control_keys:
                        defl = -1.0

                else: # Input throttle axis
                    if 'w' in self._control_keys and not 's' in self._control_keys:
                        defl = 1.0
                    elif not 'w' in self._control_keys and 's' in self._control_keys:
                        defl = -1.0

                # Apply deflection
                if self._angular_control[name]:
                    control_state[name] = min(self._control_limits[name], max(prev_controls[name]+0.01*defl, -self._control_limits[name]))
                else:
                    control_state[name] = min(1.0, max(prev_controls[name]+defl*0.0001, 0.0))

            return control_state

        else: # Otherwise, send back the previous controls
            return prev_controls


class TimeSequenceController(BaseController):
    """A controller for controlling an aircraft with ailerons, elevators, and rudder, and a throttle using a standard keyboard.

    Parameters
    ----------
    control_dict : dict
        A dictionary of control names and specifications.
    """


    def __init__(self, control_dict, quit_flag, view_flag, pause_flag, data_flag):
        super().__init__(quit_flag, view_flag, pause_flag, data_flag)

        # Store column mapping
        self._control_mapping = {}
        for key, value in control_dict.items():
            self._controls.append(key)
            self._control_mapping[key] = value["column_index"]


    def set_input(self, control_file):
        """Reads in a time sequence input file of control settings."""
        self._control_data = np.genfromtxt(control_file, delimiter=',')


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
        controls = {}
        for name in self._controls:
            i = self._control_mapping[name]
            default = prev_controls[name]
            controls[name] = np.interp(t, self._control_data[:,0], self._control_data[:,i], left=default, right=default)

        return controls


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

    return