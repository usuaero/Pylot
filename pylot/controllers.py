"""Classes defining controllers for simulated aircraft."""

from abc import abstractmethod
import pygame
from pygame.locals import *
from math import degrees, radians
import numpy as np

class BaseController:
    """An abstract aircraft controller class.
    """

    def __init__(self):
        pass

    
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
    """A controller that implements no control."""

    def get_control(self, state_vec, prev_controls):
        return prev_controls


class JoystickAircraftController(BaseController):
    """A controller for controlling an aircraft with ailerons, elevators, and rudder, and a throttle using a standard joystick.

    Parameters
    ----------
    control_limits : dict, optional
        A dictionary containing the deflection limits for each control. Defaults to 20 deg for each.

    control_responsiveness : float, optional
        Amount to increment the controls each time step if using the keyboard.
    """

    def __init__(self, **kwargs):

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

        # Get deflection limits
        limits = kwargs.get("control_limits", {})
        self._da_max = radians(limits.get("aileron", 20.0))
        self._de_max = radians(limits.get("elevator", 20.0))
        self._dr_max = radians(limits.get("rudder", 20.0))

        # Set variable for knowing if the user has perturbed from the trim state yet
        self._perturbed = False
        self._throttle_perturbed = False
        self._init_thr_pos = self._joy.get_axis(3)


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

        # Erase trim
        if self._joy.get_button(3):
            self._joy_trim = np.zeros(3)

        # Check if we're perturbed from the start control set
        if not self._perturbed and (joy_def[:3] != 0.0).any():
            self._perturbed = True

        # Check if the throttle has been moved
        if not self._throttle_perturbed and joy_def[3] != self._init_thr_pos:
            self._throttle_perturbed = True

        # Store trim
        if not self._trimming and self._joy.get_button(2):
            self._trimming = True
            self._joy_trim = self._joy_trim+joy_def[:3]

        # Check if the user has released the joystick after setting trim
        if self._trimming and (joy_def[:3] == 0.0).all():
            self._trimming = False

        # Don't apply the trim until the user releases the joystick
        if self._trimming:
            controls = {
                "aileron": (joy_def[0]**3)*-self._da_max,
                "elevator": (joy_def[1]**3)*-self._de_max,
                "rudder": (joy_def[2]**3)*-self._dr_max
            }
        else:
            controls = {
                "aileron": (joy_def[0]+self._joy_trim[0])**3*-self._da_max,
                "elevator": (joy_def[1]+self._joy_trim[1])**3*-self._de_max,
                "rudder": (joy_def[2]+self._joy_trim[2])**3*-self._dr_max
            }

        # Don't apply new throttle until it's been moved
        if self._throttle_perturbed:
            controls["throttle"] = (-joy_def[3]+1.)*0.5
        else:
            controls["throttle"] = prev_controls["throttle"]

        # Check if we've been perturbed from the trim state
        if not self._perturbed:
            return prev_controls
        else:
            return controls


class KeyboardAircraftController(BaseController):
    """A controller for controlling an aircraft with ailerons, elevators, and rudder, and a throttle using a standard keyboard.

    Parameters
    ----------
    control_limits : dict, optional
        A dictionary containing the deflection limits for each control. Defaults to 20 deg for each.

    control_responsiveness : float, optional
        Amount to increment the controls each time step if using the keyboard.
    """

    def __init__(self, **kwargs):

        # Initialize pygame
        pygame.init()

        # Initialize user inputs
        self._KEYBOARD = True
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

        # Get deflection limits
        limits = kwargs.get("control_limits", {})
        self._da_max = radians(limits.get("aileron", 20.0))
        self._de_max = radians(limits.get("elevator", 20.0))
        self._dr_max = radians(limits.get("rudder", 20.0))
        self._responsiveness = 0.001

        # Set variable for knowing if the user has perturbed from the trim state yet
        self._perturbed = False
        self._throttle_perturbed = False
        self._init_thr_pos = self._joy.get_axis(3)


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

        # Apply controls
        if self._perturbed:

            # Elevator
            if self._UP == True and self._DOWN == False and prev_controls["elevator"] < self._de_max:
                ele = 1.
            elif self._UP == False and self._DOWN == True and prev_controls["elevator"] > -self._de_max:
                ele = -1.
            else:
                ele = 0.

            # Aileron
            if self._LEFT == True and self._RIGHT == False and prev_controls["aileron"] < self._da_max:
                ail = 1.
            elif self._LEFT == False and self._RIGHT == True and prev_controls["aileron"] > -self._da_max:
                ail = -1.
            else:
                ail = 0.

            # Rudder
            if self._AA == True and self._DD == False and prev_controls["rudder"] < self._dr_max:
                rud = 1.
            elif self._AA == False and self._DD == True and prev_controls["rudder"] > -self._dr_max:
                rud = -1.
            else:
                rud = 0.

            # Elevator
            thr = prev_controls["throttle"]
            if self._WW == True and self._SS == False and thr<=1.0-self._responsiveness:
                thr += self._responsiveness*5
            elif self._WW == False and self._SS == True and thr>=0.0+self._responsiveness:
                thr -= self._responsiveness*5

            controls = {
                "aileron" : prev_controls["aileron"]+self._responsiveness*ail,
                "elevator" : prev_controls["elevator"]+self._responsiveness*ele,
                "rudder" : prev_controls["rudder"]+self._responsiveness*rud,
                "throttle" : thr
            }

        # Check if we've been perturbed from the trim state
        if not self._perturbed:
            return prev_controls
        else:
            return controls