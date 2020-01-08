# Classes describing the aircraft used by the simulator

import math as m
import numpy as np
import machupX as mx
from abc import abstractmethod

class BaseAircraft:
    """A base class for aircraft to be used in the simulator.

    Parameters
    ----------
    name : str
        Name of the aircraft.

    input_dict : dict
        Dictionary describing the airplane.
    """

    def __init__(self, name, input_dict):

        # Store input
        self._input_dict = input_dict

        # Initialize state
        self.y = np.zeros(13)


    def dy_dt(self, t):
        pass


    @abstractmethod
    def get_FM(self, t):
        pass


    @abstractmethod
    def trim(self, **kwargs):
        pass


    @abstractmethod
    def get_graphics_obj(self):
        pass


class LinearizedAirplane(BaseAircraft):
    """An airplane defined by a linearized model of aerodynamic coefficients.
    
    Parameters
    ----------
    name : str
        Name of the aircraft.

    input_dict : dict
        Dictionary describing the airplane.
    """

    def __init__(self, name, input_dict):
        super.__init__(name, input_dict)


class MachUpXAirplane(BaseAircraft):
    """An airplane defined by MachUpX.
    
    Parameters
    ----------
    name : str
        Name of the aircraft.

    input_dict : dict
        Dictionary describing the airplane.
    """

    def __init__(self, name, input_dict):
        super.__init__(name, input_dict)