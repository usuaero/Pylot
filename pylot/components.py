from .helpers import import_value, Body2Fixed, Fixed2Body, Quat2Euler, cross
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
        self._aircraft_CG = kwargs.get("CG")

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
        FM[3:] = cross(self._position-self._aircraft_CG, F)

        return FM


    def get_unit_thrust_moment(self):
        """Returns the thrust moment vector assuming a thrust magnitude of unity.
        """
        return cross(self._position, self._direction)


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
            return cross(self._position, self._direction*(rho/self._rho0)**self._a*(self._T0+self._T1*V+self._T2*V*V))
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
        self._pos = import_value("position", kwargs, self._units, None)
        self._k = import_value("stiffness", kwargs, self._units, None)
        self._c = import_value("damping", kwargs, self._units, 0.0)
        self._u_f_roll = import_value("rolling_friction_coef", kwargs, self._units, 0.0)
        self._u_f_slid = import_value("sliding_friction_coef", kwargs, self._units, 0.0)
        self._ref_area = import_value("area", kwargs, self._units, 1.0)
        self._CD = import_value("CD", kwargs, self._units, 0.0)
        self._drag_param = self._ref_area*self._CD
        self._steer_cntrl = kwargs.get("steering_control", None)
        self._aircraft_CG = kwargs.get("CG")
        if kwargs.get("steering_reversed", False):
            self._steer_orient = -1.0
        else:
            self._steer_orient = 1.0


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
            v_tip = cross(w, self._pos)+v

            # Determine how fast the depth is changing
            v_tip_f = Body2Fixed(v_tip, q)
            velocity = v_tip_f[2]

            # Determine normal force exerted by the shock
            N = depth*self._k+velocity*self._c*float(velocity>0.0) # So the airplane doesn't stick to the ground... ;)
            F_f = [0.0, 0.0, -N]

            # Determine the direction the wheel is pointing
            psi = Quat2Euler(q)[2]

            # Get steering deflection
            if self._steer_cntrl is not None:
                psi += self._steer_orient*m.radians(controls.get(self._steer_cntrl, 0.0))

            # Determine rolling and sliding directions
            C_psi = m.cos(psi)
            S_psi = m.sin(psi)
            u_roll = np.asarray([C_psi, S_psi, 0.0])
            u_slid = np.asarray([-S_psi, C_psi, 0.0])

            # Determine rolling and sliding velocities
            v_roll = v_tip_f[0]*u_roll[0]+v_tip_f[1]*u_roll[1]
            v_slid = v_tip_f[0]*u_slid[0]+v_tip_f[1]*u_slid[1]

            # Determine friction forces on the wheel
            F_f -= u_roll*self._u_f_roll*N*np.sign(v_roll)
            F_f -= u_slid*self._u_f_slid*N*np.sign(v_slid)
            FM[:3] = Fixed2Body(F_f, q)

        # Get drag
        FM[:3] += -0.5*rho*V*V*self._drag_param*u_inf

        # Set moments
        FM[3:] = cross(self._pos-self._aircraft_CG, FM[:3])

        return FM