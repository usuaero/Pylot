Building Custom Controllers
===========================

A user defined controller can be derived from BaseController (found in pylot/controllers.py). For Pylot to recognize this controller, it must be named UserDefinedController and placed in the current working directory in a file named user_defined_controller.py. BaseController contains a number of abstract methods that must be defined in the derived class. To ensure proper Pylot behavior, non-abstract methods should not be redefined.

BaseController has the following structure:

.. automodule:: pylot
.. autoclass:: BaseController
   :members: