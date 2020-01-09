# Classes describing the aircraft used by the simulator

import math as m
import numpy as np
import machupX as mx
from abc import abstractmethod
from .helpers import *

class BaseAircraft:
    """A base class for aircraft to be used in the simulator.

    Parameters
    ----------
    name : str
        Name of the aircraft.

    input_dict : dict
        Dictionary describing the airplane.
    """

    def __init__(self, name, input_dict, density, state_output):

        # Store input
        self._input_dict = input_dict

        # Initialize state
        self.y = np.zeros(13)

        # Setup output
        self._output_state = state_output is not None
        if self._output_state:
            self._output_handle = open(state_output, 'w')
            header = "   Time              u                 v                 w                 p                 q                 r                 x                 y                 z                 eo                e1                e2                e3"
            print(header, file=self._output_handle)

        # Determine units and set gravity
        self._units = self._input_dict.get("units", "English")
        if self._units == "English":
            self._g = 32.2
        elif self._units == "SI":
            self._g = 9.81
        else:
            raise IOError("{0} is not a valid units specification.".format(self._units))

        # Set mass properties
        self._W = import_value("weight", self._input_dict, self._units, None)
        self._m_inv = self._g/self._W

        self._I_xx = import_value("Ixx", self._input_dict["inertia"], self._units, None)
        self._I_yy = import_value("Iyy", self._input_dict["inertia"], self._units, None)
        self._I_zz = import_value("Izz", self._input_dict["inertia"], self._units, None)
        self._I_xy = import_value("Ixy", self._input_dict["inertia"], self._units, None)
        self._I_xz = import_value("Ixz", self._input_dict["inertia"], self._units, None)
        self._I_yz = import_value("Iyz", self._input_dict["inertia"], self._units, None)

        self._I_inv = np.linalg.inv(np.array([[self._I_xx, -self._I_xy, -self._I_xz],
                                              [-self._I_xy, self._I_yy, -self._I_yz],
                                              [-self._I_xz, -self._I_yz, self._I_zz]]))

        self._I_diff_yz = self._I_yy-self._I_zz
        self._I_diff_zx = self._I_zz-self._I_xx
        self._I_diff_xy = self._I_xx-self._I_yy

        # Set angular momentums
        self._hx, self._hy, self._hz = self._input_dict.get("angular_momentum", [0.0, 0.0, 0.0])


    def __del__(self):
        if self.output_state:
            self._output_handle.close()


    def output_state(self, t):
        if self._output_state:
            s = ["{:>18.9E}".format(t)]
            for yi in self.y:
                s.append("{:>18.9E}".format(yi))
            print("".join(s), file=self._output_handle)


    def dy_dt(self, t):

        # Get forces and moments
        FM = self.get_FM(t)

        # Extract state
        u = self.y[0]
        v = self.y[1]
        w = self.y[2]
        p = self.y[3]
        q = self.y[4]
        r = self.y[5]
        q0 = self.y[9]
        qx = self.y[10]
        qy = self.y[11]
        qz = self.y[12]

        # Apply Newton's equations
        dy = np.zeros(13)

        # Linear acceleration
        dy[0] = 2*self._g*(qx*qz-qy*q0) + self._m_inv*FM[0] + r*v-q*w
        dy[1] = 2*self._g*(qy*qz+qx*q0) + self._m_inv*FM[1] + p*w-r*u
        dy[2] = self._g*(qz*qz+q0*q0-qx*qx-qy*qy) + self._m_inv*FM[2] + q*u-p*v

        # Angular acceleration
        pq = p*q
        qr = q*r
        pr = p*r
        p2 = p*p
        q2 = q*q
        r2 = r*r
        M = [0.0, 0.0, 0.0]
        M[0] = -self._hz*q + self._hy*r + FM[3] + self._I_diff_yz*qr + self._I_yz*(q2-r2)+self._I_xz*pq-self._I_xy*pr
        M[1] =  self._hz*p - self._hx*r + FM[4] + self._I_diff_zx*pr + self._I_xz*(r2-p2)+self._I_xy*qr-self._I_yz*pq
        M[2] = -self._hy*p + self._hx*q + FM[5] + self._I_diff_xy*pq + self._I_xy*(p2-q2)+self._I_yz*pr-self._I_xz*qr

        dy[3] = self._I_inv[0,0]*M[0] + self._I_inv[0,1]*M[1] + self._I_inv[0,2]*M[2]
        dy[4] = self._I_inv[1,0]*M[0] + self._I_inv[1,1]*M[1] + self._I_inv[1,2]*M[2]
        dy[5] = self._I_inv[2,0]*M[0] + self._I_inv[2,1]*M[1] + self._I_inv[2,2]*M[2]

        # Translation
        dy[6:9] = Body2Fixed(self.y[:3], self.y[9:])

        # Rotation
        dy[9] = 0.5*(-qx*p-qy*q-qz*r)
        dy[10] = 0.5*(q0*p-qz*q+qy*r)
        dy[11] = 0.5*(qz*p+q0*q-qx*r)
        dy[12] = 0.5*(-qy*p+qx*q+q0*r)

        return dy


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

    def __init__(self, name, input_dict, density, state_output):
        super().__init__(name, input_dict, density, state_output)


class MachUpXAirplane(BaseAircraft):
    """An airplane defined by MachUpX.
    
    Parameters
    ----------
    name : str
        Name of the aircraft.

    input_dict : dict
        Dictionary describing the airplane.
    """

    def __init__(self, name, input_dict, density, state_output):
        super().__init__(name, input_dict, density, state_output)