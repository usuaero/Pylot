# Building Custom Controllers

Pylot implements controllers using an object-oriented approach. The aircraft has a controller object which is queries at each time step to get the current control state. The aircraft passes its current state and controls to the controller, and the controller determines the new control state. Pylot comes with four built in controller classes (NoController, JoystickController, KeyboardController, and TimeSequenceController) which are all derived from the class BaseController.

A user can derive their own custom controller from BaseController (found in pylot/controllers.py). For Pylot to recognize this controller, it must be named UserDefinedController and placed in the current working directory in a file named user_defined_controller.py. BaseController contains one abstract method (```get_control```) that must be defined in the derived class. This method takes the time index, the current aircraft state vector, and the previous control state as arguments and returns the new control state as a dictionary. To ensure proper Pylot behavior, non-abstract methods should not be redefined.

UserDefinedController should be declared as follows

```python
from pylot.controllers import BaseController

class UserDefinedController(BaseController):

    def __init__(self, control_dict, quit_flag, view_flag, pause_flag, data_flag, enable_interface):
        """Initializer.
        
        Parameters
        ----------
        control_dict: dict
            Dictionary of control parameters declared in the airplane input JSON.

        All other parameters are simply necessary for Pylot and should not be altered.
        """

        super().__init__(quit_flag, view_flag, pause_flag, data_flag, enable_interface)


    def get_control(self, t, state_vec, prev_controls):
        """Returns the controls based on the inputted state and keyboard/joystick inputs.

        Parameters
        ----------
        t : float
            Time index.

        state_vec : list
            State vector of the entity being controlled.

        prev_controls : dict
            Previous control values.

        Returns
        -------
        controls : dict
            Dictionary of controls where the keys are the names of the controls and the values
            are the control settings. Not that angular controls are specified in degrees. Non-
            angular controls (such as throttle) vary from 0 to 1.
        """
```

Happy experimenting!