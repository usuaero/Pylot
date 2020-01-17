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
        self._controls = []


    def get_control_names(self):
        return self._controls


    def get_input(self):
        """Returns a dictionary of inputs from the user for controlling pause, view, etc."""

        inputs= {}

        # Check events
        for event in pygame.event.get():

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
                    inputs["fpv"] = True

                else:
                    pygame.event.post(event)

            elif event.type == pygame.KEYDOWN or event.type == pygame.KEYUP: # Only put key events back on the queue
                pygame.event.post(event)

        return inputs

    
    @abstractmethod
    def get_control(self, state_vec, prev_controls):
        """Returns the controls based on the inputted state.

        Parameters
        ----------
        state_vec : list
            State vector of the entity being controlled.

        prev_controls : dict
            Previous control values.
        """
        pass


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
        for key, value in control_dict.items():
            self._controls.append(key)

    def get_control(self, state_vec, prev_controls):
        return prev_controls


class JoystickAircraftController(BaseController):
    """A controller for controlling a 4-channel aircraft using a standard joystick.

    Parameters
    ----------
    control_dict : dict
        A dictionary of control names and specifications.
    """

    def __init__(self, control_dict):
        super().__init__()

        # Initialize pygame
        pygame.init()

        # Initialize user inputs
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


    def get_control(self, state_vec, prev_controls):
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
        for name in self._controls:
            if self._angular_control[name]:
                control_state[name] = (joy_def[self._control_mapping[name]]**3)*-self._control_limits[name]
            else:
                control_state[name] = (-joy_def[self._control_mapping[name]]+1.)*0.5

        return control_state


class KeyboardAircraftController(BaseController):
    """A controller for controlling an aircraft with ailerons, elevators, and rudder, and a throttle using a standard keyboard.

    Parameters
    ----------
    control_dict : dict
        A dictionary of control names and specifications.
    """

    def __init__(self, control_dict):
        super().__init__()

        # Initialize pygame
        pygame.init()

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


    def get_control(self, state_vec, prev_controls):
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