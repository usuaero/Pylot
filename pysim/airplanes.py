# Classes describing the aircraft used by the simulator

import math as m
import numpy as np
import machupX as mx
from abc import abstractmethod

class BaseAirplane:
    """A base class for aircraft to be used in the simulator.

    Parameters
    ----------
    input_dict : dict
        Dictionary describing the airplane.
    """

    def __init__(self, input_dict):

        # Store input
        self._input_dict = input_dict


    def dy_dt(self, t):
        pass


    @abstractmethod
    def get_FM(self, t):
        pass


class LinearAirplane(BaseAirplane):
    """An airplane defined by a linearized model of aerodynamic coefficients.
    
    Parameters
    ----------
    input_dict : dict
        Dictionary describing the airplane.
    """

    def __init__(self, input_dict):
        super.__init__(input_dict)