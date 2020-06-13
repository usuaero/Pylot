from .helpers import import_value, Body2Fixed, Fixed2Body
from .std_atmos import statee, statsi
import numpy as np
import math as m


class Engine:
    """An engine for a simulated aircraft.

    Parameters
    ----------
    name : str
        Name of the engine.

    offset : vector
        Location of the engine in body-fixed coordinates. Defaults to [0.0, 0.0, 0.0].

    T0 : float
    
    T1 : float

    T2 : float
    
    a : float

    control : str
        Name of the control that sets the throttle for this engine. Defaults to "throttle".

    CD : float, optional
        Coefficient of drag for the powerplant. Defaults to 0.0.

    area : float, optional
        Reference area for redimensionalizing the drag coefficient. Defaults to 1.0.

    units : str
        Unit system used for the engine.
    """

    def __init__(self, name, **kwargs):

        # Store parameters
        self._name = name
        self._units = kwargs.get("units")
        self._position = import_value("position", kwargs, self._units, [0.0, 0.0, 0.0])
        self._direction = import_value("direction", kwargs, self._units, [1.0, 0.0, 0.0])
        self._T0 = import_value("T0", kwargs, self._units, 0.0)
        self._T1 = import_value("T1", kwargs, self._units, 0.0)
        self._T2 = import_value("T2", kwargs, self._units, 0.0)
        self._a = import_value("a", kwargs, self._units, 1.0)
        self._control = kwargs.get("control", "throttle")
        self._ref_area = import_value("area", kwargs, self._units, 1.0)
        self._CD = import_value("CD", kwargs, self._units, 0.0)
        self._drag_param = self._ref_area*self._CD

        # Normalize direction vector
        self._direction /= np.linalg.norm(self._direction)

        # Determine air density at sea level
        if self._units == "English":
            self._rho0 = statee(0)[-1]
        else:
            self._rho0 = statsi(0)[-1]


    def get_thrust_FM(self, controls, rho, u_inf, V):
        """Returns the forces and moments due to thrust from this engine.

        Parameters
        ----------
        controls : dict
            Dictionary of control settings.

        rho : float
            Air density.

        u_inf : ndarray
            Freestream direction vector.

        V : float
            Airspeed.

        Returns
        -------
        FM : vector
            Forces and moments due to thrust.
        """

        FM = np.zeros(6)

        # Get throttle setting
        tau = controls.get(self._control, 0.0)

        # Calculate thrust magnitude
        T = tau*(rho/self._rho0)**self._a*(self._T0+self._T1*V+self._T2*V*V)

        # Calculate drag
        D = -0.5*rho*V*V*self._drag_param*u_inf

        # Set thrust vector
        F = T*self._direction+D
        FM[:3] = F

        # Set moments
        FM[3:] = np.cross(self._position, F)

        return FM


    def get_unit_thrust_moment(self):
        """Returns the thrust moment vector assuming a thrust magnitude of unity.
        """
        return np.cross(self._position, self._direction)


    def get_thrust_deriv(self, control, rho, V):
        """Returns the derivative of the thrust vector with respect to the given control.
        """
        if control == self._control:
            return self._direction*(rho/self._rho0)**self._a*(self._T0+self._T1*V+self._T2*V*V)
        else:
            return np.zeros(3)


    def get_thrust_moment_deriv(self, control, rho, V):
        """Returns the derivative of the thrust moment with respect to the given control.
        """
        if control == self._control:
            return np.cross(self._position, self._direction*(rho/self._rho0)**self._a*(self._T0+self._T1*V+self._T2*V*V))
        else:
            return np.zeros(3)


class LandingGear:
    """A single landing gear for an aircraft

    Parameters
    ----------
    name : str
        Name of the engine.

    tip_loc : list
        Location of the tip of the landing gear in body-fixed coordinates.

    shock_stiffness : float
        Spring constant for the shock system.

    shock_damping : float
        Damping constant for the shock system.

    units : str
        Unit system for the shocks.
    """

    def __init__(self, name, **kwargs):

        # Load params
        self.name = name
        self._units = kwargs.get("units")
        self._pos = import_value("tip_loc", kwargs, self._units, None)
        self._k = import_value("shock_stiffness", kwargs, self._units, None)
        self._c = import_value("shock_damping", kwargs, self._units, None)
        self._u_f_roll = import_value("rolling_friction_coef", kwargs, self._units, 0.0)
        self._u_f_slid = import_value("sliding_friction_coef", kwargs, self._units, 0.0)
        self._ref_area = import_value("area", kwargs, self._units, 1.0)
        self._CD = import_value("CD", kwargs, self._units, 0.0)
        self._drag_param = self._ref_area*self._CD


    def get_landing_FM(self, y, controls, rho, u_inf, V):
        """Returns the forces and moments generated by this landing gear.

        Parameters
        ----------
        y : list
            State vector of aircraft.

        Returns
        -------
        FM : list
            Forces and moments due to landing interactions.
        """

        FM = np.zeros(6)

        # Determine if this strut is interacting with the ground
        z = y[8]
        q = y[9:]
        depth = Body2Fixed(self._pos, q)[2]+z
        if depth > 0.0:

            # Determine velocity of the tip
            v = y[:3]
            w = y[3:6]
            v_tip = np.cross(w, self._pos)+v

            # Determine how fast the depth is changing
            v_tip_f = Body2Fixed(v_tip, q)
            velocity = v_tip_f[2]

            # Determine normal force exerted by the shock
            N = depth*self._k+velocity*self._c
            F_f = [0.0, 0.0, -N]
            F = Fixed2Body(F_f, q)

            # Determine friction forces on the wheel
            if v_tip[0] != 0.0:
                F[0] -= self._u_f_roll*N*np.sign(v_tip[0])
            if v_tip[1] != 0.0:
                F[1] -= self._u_f_slid*N*np.sign(v_tip[1])
            FM[:3] = F

            # Determine moment vector
            FM[3:] = np.cross(self._pos, F)

        # Get drag
        FM[:3] += -0.5*rho*V*V*self._drag_param*u_inf

        # Set moments
        FM[3:] = np.cross(self._pos, FM[:3])

        return FM