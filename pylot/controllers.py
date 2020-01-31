"""Classes defining controllers for simulated aircraft."""

from abc import abstractmethod
import pygame.event
import pygame.joystick
from math import degrees, radians
import numpy as np

class BaseController:
    """An abstract aircraft controller class.
    """

    def __init__(self):

        # Initialize controls
        self._controls = []


    def get_control_names(self):
        """Returns the names of the controls handled by this controller."""
        return self._controls


    def get_input(self):
        """Returns a dictionary of inputs from the user for controlling pause, view, etc."""

        inputs= {}

        # Check events
        for event in pygame.event.get():

            # Check for key down events
            if event.type == pygame.KEYDOWN:

                # Toggle flight data display
                if event.key == pygame.K_i:
                    inputs["data"] = True

                # Pause simulation
                elif event.key == pygame.K_p:
                    inputs["pause"] = True

                # Quit game
                elif event.key == pygame.K_q:
                    inputs["quit"] = True

                # Switch view
                elif event.key == pygame.K_SPACE:
                    inputs["view"] = True

                # If it's not used here, put it back on the queue
                else:
                    pygame.event.post(event)

            # Check for general quit condition
            elif event.type == pygame.QUIT:
                inputs["quit"] = True
                
            # Put keyup events back on the queue
            elif event.type == pygame.KEYUP:
                pygame.event.post(event)

        return inputs

    
    @abstractmethod
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

    def __init__(self, control_dict):
        super().__init__()

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

    def __init__(self, control_dict):
        super().__init__()

        # Initialize user inputs
        pygame.joystick.init()
        if pygame.joystick.get_count()>0.:
            self._joy = pygame.joystick.Joystick(0)
            self._joy.init()
            self._joy_trim = np.zeros(3)
            self._trimming = False
        else:
            raise IOError("No joystick detected.")

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
        self._joy_init = np.zeros(4)
        self._joy_init[0] = self._joy.get_axis(0)
        self._joy_init[1] = self._joy.get_axis(1)
        self._joy_init[2] = self._joy.get_axis(2)
        self._joy_init[3] = self._joy.get_axis(3)

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

        # Get the joystick positions
        joy_def = np.zeros(4)
        joy_def[0] = self._joy.get_axis(0)
        joy_def[1] = self._joy.get_axis(1)
        joy_def[2] = self._joy.get_axis(2)
        joy_def[3] = self._joy.get_axis(3)

        # Check if we're perturbed from the start control set
        if not self._perturbed:
            if (joy_def != self._joy_init).any():
                self._perturbed = True
            else:
                return prev_controls # No point in parsing things if nothing's changed

        # Parse new controls
        control_state = {}
        setting_list = []
        for name in self._controls:
            if self._angular_control[name]:
                setting = (joy_def[self._control_mapping[name]]**3)*-self._control_limits[name]
            else:
                setting = (-joy_def[self._control_mapping[name]]+1.)*0.5
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

    def __init__(self, control_dict):
        super().__init__()

        # Initialize user inputs
        self._thr = 0.
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

        # Check for keyboard inputs
        for event in pygame.event.get():

            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_UP:
                    self._UP = True
                elif event.key == pygame.K_DOWN:
                    self._DOWN = True
                elif event.key == pygame.K_LEFT:
                    self._LEFT = True
                elif event.key == pygame.K_RIGHT:
                    self._RIGHT = True

                elif event.key == pygame.K_w:
                    self._WW = True
                elif event.key == pygame.K_s:
                    self._SS = True
                elif event.key == pygame.K_a:
                    self._AA = True
                elif event.key == pygame.K_d:
                    self._DD = True
                else:
                    pygame.event.post(event)

            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_UP:
                    self._UP = False
                elif event.key == pygame.K_DOWN:
                    self._DOWN = False
                elif event.key == pygame.K_LEFT:
                    self._LEFT = False
                elif event.key == pygame.K_RIGHT:
                    self._RIGHT = False

                elif event.key == pygame.K_w:
                    self._WW = False
                elif event.key == pygame.K_s:
                    self._SS = False
                elif event.key == pygame.K_a:
                    self._AA = False
                elif event.key == pygame.K_d:
                    self._DD = False
                else:
                    pygame.event.post(event)
            else:
                pygame.event.post(event)

        # Check for perturbation
        if not self._perturbed and (self._RIGHT or self._LEFT or self._UP or self._DOWN or self._WW or self._SS or self._DD or self._AA):
            self._perturbed = True

        if self._perturbed:
            # Parse new controls
            control_state = {}
            for i in range(4):
                name = self._control_reverse_mapping[i]
                defl = 0.0

                # Get axis input
                if i == 0: # Input roll axis
                    if self._LEFT and not self._RIGHT:
                        defl = 1.0
                    elif not self._LEFT and self._RIGHT:
                        defl = -1.0

                elif i == 1: # Input pitch axis
                    if self._UP and not self._DOWN:
                        defl = 1.0
                    elif not self._UP and self._DOWN:
                        defl = -1.0

                elif i == 2: # Input yaw axis
                    if self._AA and not self._DD:
                        defl = 1.0
                    elif not self._AA and self._DD:
                        defl = -1.0

                else: # Input throttle axis
                    if self._WW and not self._SS:
                        defl = 1.0
                    elif not self._WW and self._SS:
                        defl = -1.0

                # Apply deflection
                if self._angular_control[name]:
                    control_state[name] = min(self._control_limits[name], max(prev_controls[name]+0.01*defl, -self._control_limits[name]))
                else:
                    control_state[name] = min(1.0, max(prev_controls[name]+defl*0.01, 0.0))

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


    def __init__(self, control_dict):
        super().__init__()

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