Building Custom Controllers
===========================

Pylot implements controllers using an object-oriented approach. The aircraft has a controller object which is queries at each time step to get the current control state. The aircraft passes its current state and controls to the controller, and the controller determines the new control state. Pylot comes with four built in controller classes (NoController, JoystickController, KeyboardController, and TimeSequenceController) which are all derived from the class BaseController.

A user can derive their own custom controller from BaseController (found in pylot/controllers.py). For Pylot to recognize this controller, it must be named UserDefinedController and placed in the current working directory in a file named user_defined_controller.py. BaseController contains one abstract method (```get_control```) that must be defined in the derived class. This method takes the time index, the current aircraft state vector, and the previous control state as arguments and returns the new control state as a dictionary. To ensure proper Pylot behavior, non-abstract methods should not be redefined.

BaseController has the following structure:

.. automodule:: pylot
.. autoclass:: BaseController
   :members:

Again, only ```get_control``` should be defined in any derived classes. All other methods should not be redefined.