# Classes describing the aircraft used by the simulator

import os
import copy

import math as m
import numpy as np
import machupX as mx
import scipy.optimize as opt

from abc import abstractmethod
from pylot.helpers import import_value, Euler2Quat, Body2Fixed, NormalizeQuaternion, NormalizeQuaternionNearOne, Fixed2Body, cross
from pylot.std_atmos import statee, statsi
from pylot.controllers import NoController, KeyboardController, JoystickController, TimeSequenceController
from pylot.components import Engine, LandingGear

class BaseAircraft:
    """A base class for aircraft to be used in the simulator.

    Parameters
    ----------
    name : str
        Name of the aircraft.

    input_dict : dict
        Dictionary describing the airplane.
    """

    def __init__(self, name, input_dict, density, units, param_dict, quit_flag, view_flag, pause_flag, data_flag, enable_interface):

        # Store input
        self.name = name
        self._input_dict = input_dict

        # Initialize state
        self.y = np.zeros(13)

        # Determine units and set gravity
        self._units = units
        if self._units == "English":
            self._g = 32.2
        elif self._units == "SI":
            self._g = 9.81
        else:
            raise IOError("{0} is not a valid units specification.".format(self._units))

        # Setup output
        state_output = param_dict.get("state_output", None)
        self._output_state = state_output is not None
        if self._output_state:
            self._output_handle = open(state_output, 'w')
            if self._units == "English":
                header = "   Time[s]           u[ft/s]           v[ft/s]           w[ft/s]           p[rad/s]          q[rad/s]          r[rad/s]          x[ft]             y[ft]             z[ft]             eo                e1                e2                e3"
            else:
                header = "   Time[s]           u[m/s]            v[m/s]            w[m/s]            p[rad/s]          q[rad/s]          r[rad/s]          x[m]              y[m]              z[m]              eo                e1                e2                e3"
            print(header, file=self._output_handle)

        # Set mass properties
        self._CG = import_value("CG", self._input_dict, self._units, [0.0, 0.0, 0.0])
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
        self._hx, self._hy, self._hz = import_value("angular_momentum", self._input_dict, self._units, [0.0, 0.0, 0.0])

        # Load engines
        self._engines = []
        self._num_engines = 0
        for key, value in self._input_dict.get("engines", {}).items():
            self._engines.append(Engine(key, **value, units=self._units, CG=self._CG))
            self._num_engines += 1

        # Load landing gear
        self._landing_gear = []
        self._num_landing_gear = 0
        for key, value in self._input_dict.get("landing_gear", {}).items():
            self._landing_gear.append(LandingGear(key, **value, units=self._units, CG=self._CG))
            self._num_landing_gear += 1

        # Get position of bungee hook
        self._hook_pos = import_value("launch_hook_position", self._input_dict, self._units, [0.0, 0.0, 0.0])
        self._hooked = False

        # Load controls
        controller = param_dict.get("controller", None)
        self._initialize_controller(controller, quit_flag, view_flag, pause_flag, data_flag, enable_interface, param_dict.get("control_output", None))

        # Get stall angle of attack
        self._alpha_stall = m.radians(self._input_dict["aero_model"].get("stall_angle_of_attack", 15.0))
        self._beta_stall = m.radians(self._input_dict["aero_model"].get("stall_sideslip_angle", 200.0))


    def _initialize_controller(self, control_type, quit_flag, view_flag, pause_flag, data_flag, enable_interface, control_output):
        # Sets up the control input for the aircraft

        # No control input
        if control_type is None:
            self.controller = NoController(self._input_dict.get("controls", {}), quit_flag, view_flag, pause_flag, data_flag, enable_interface, control_output)

        # Joystick
        elif control_type == "joystick":
            self.controller = JoystickController(self._input_dict.get("controls", {}), quit_flag, view_flag, pause_flag, data_flag, enable_interface, control_output)

        # Keyboard
        elif control_type == "keyboard":
            self.controller = KeyboardController(self._input_dict.get("controls", {}), quit_flag, view_flag, pause_flag, data_flag, enable_interface, control_output)

        # User-defined
        elif control_type == "user-defined":
            from user_defined_controller import UserDefinedController
            self.controller = UserDefinedController(self._input_dict.get("controls", {}), quit_flag, view_flag, pause_flag, data_flag, enable_interface, control_output)

        # Time sequence file
        elif ".csv" in control_type:
            self.controller = TimeSequenceController(self._input_dict.get("controls", {}), quit_flag, view_flag, pause_flag, data_flag, enable_interface, control_output)
            self.controller.read_control_file(control_type)

        else:
            raise IOError("{0} is not a valid controller specification.".format(control_type))

        # Setup storage
        self._control_names = self.controller.get_control_names()
        self.controls = {}


    def finalize(self):
        if self._output_state:
            self._output_handle.close()

        self.controller.finalize()


    def output_state(self, t):
        if self._output_state:
            s = ["{:>18.9E}".format(t)]
            for yi in self.y:
                s.append("{:>18.9E}".format(yi))
            print("".join(s), file=self._output_handle)


    def _correct_stall(self, CL, CD, CS, Cl, Cm, Cn, alpha, beta, S_a, S_B, C_a, C_B):
        # Corrects the aerodnamic coefficients for stall
        # TODO : Make this better!

        # Blending coefficient
        k = 100

        # Get stall values
        CL_stall = 2.0*S_a*S_a*C_a*np.sign(alpha)
        CD_stall = 1-C_a*S_a*np.sign(alpha)
        Cm_stall = -0.5*S_a
        CS_stall = 0.2*S_B*S_B*C_B*np.sign(beta)
        Cl_stall = -CS_stall*0.05
        Cn_stall = CS_stall*0.1

        # Blending functions
        lon_stall_weight = 1/(1+m.exp(-k*(-self._alpha_stall-alpha)))+1/(1+m.exp(-k*(alpha-self._alpha_stall)))
        lat_stall_weight = 1/(1+m.exp(-k*(-self._beta_stall-beta)))+1/(1+m.exp(-k*(beta-self._beta_stall)))

        # Blend
        CL = CL*(1-lon_stall_weight)+CL_stall*lon_stall_weight
        CD = CD*(1-lon_stall_weight)+CD_stall*lon_stall_weight
        Cm = Cm*(1-lon_stall_weight)+Cm_stall*lon_stall_weight
        CS = CS*(1-lat_stall_weight)+CS_stall*lat_stall_weight
        Cl = Cl*(1-lat_stall_weight)+Cl_stall*lat_stall_weight
        Cn = Cn*(1-lat_stall_weight)+Cn_stall*lat_stall_weight

        return CL, CD, CS, Cl, Cm, Cn


    def dy_dt(self, t):
        """Calculates the derivative of the state vector with respect to time
        at the current state and time.

        Parameters
        ----------
        t : float
            Current simulation time.

        Returns
        -------
        ndarray
            Time reate of change of each state variable.
        """

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


    def _get_elevation(self, alpha, beta, phi, gamma):
        # Calculates the elevation angle, theta, based on the other known angles

        # Calculate constants
        C_a = m.cos(alpha)
        S_a = m.sin(alpha)
        C_B = m.cos(beta)
        S_B = m.sin(beta)
        C_phi = m.cos(phi)
        S_phi = m.sin(phi)
        S_gamma = m.sin(gamma)

        # More constants
        D = m.sqrt(1-S_a*S_a*S_B*S_B)
        A = S_a*C_B/D
        B = C_a*S_B/D
        C = C_a*C_B/D
        E = B*S_phi+A*C_phi
        E2 = E*E
        C2 = C*C

        # Get two solutions
        first = C*S_gamma
        second = E*m.sqrt(C2+E2-S_gamma*S_gamma)
        denom = C2+E2
        theta1 = m.asin((first+second)/denom)
        theta2 = m.asin((first-second)/denom)

        # Check which one is closer
        if abs(C*m.sin(theta1)-E*m.cos(theta1)-S_gamma) < abs(C*m.sin(theta2)-E*m.cos(theta2)-S_gamma):
            return theta1
        else:
            return theta2


    def _get_rotation_rates(self, C_phi, S_phi, C_theta, S_theta, g, u, w):
        # Returns the rotation rates in a steady, coordinated turn

        omega = g*S_phi*C_theta/(S_theta*w+C_phi*C_theta*u)
        return -S_theta*omega, S_phi*C_theta*omega, C_phi*C_theta*omega


    def normalize(self):
        """Normalizes the orientation quaternion."""
        self.y[9:] = NormalizeQuaternion(self.y[9:])


    def _trim(self, trim_dict):
        # Trims the aircraft at the given condition

        # Get params
        self._V0 = trim_dict["airspeed"]
        self.y[6:9] = trim_dict["position"]
        self._climb = m.radians(trim_dict.get("climb_angle", 0.0))
        self._bank = m.radians(trim_dict.get("bank_angle", 0.0))
        self._heading = m.radians(trim_dict.get("heading", 0.0))
        self._trim_verbose = trim_dict.get("verbose", False)

        # Parse controls
        self._avail_controls = trim_dict.get("trim_controls", list(self.controls.keys()))
        self._fixed_controls = trim_dict.get("fixed_controls", {})
        for name in self._control_names:
            if name in self._avail_controls:
                continue
            self._fixed_controls[name] = self._fixed_controls.get(name, 0.0)

        # Check we have enough
        if len(self._avail_controls) != 4:
            raise IOError("Exactly 4 controls must be used to trim the aircraft. Got {0}.".format(len(self._avail_controls)))

        # Initialize output
        if self._trim_verbose:
            print("Trimming at {0} deg bank and {1} deg climb...".format(m.degrees(self._bank), m.degrees(self._climb)))
            header = ["{0:>20}{1:>20}".format("Alpha [deg]", "Beta [deg]")]
            for name in self._avail_controls:
                header.append("{0:>20}".format(name.title()))
            header.append("{0:>20}".format("Elevation [deg]"))
            print("".join(header))

        # Solve for trim
        trim_val_guess = np.zeros(6)
        x, info_dict, ier, mesg = opt.fsolve(self._trim_residual_function, trim_val_guess, full_output=True)
        trim_settings = x

        # Output results of trim
        if self._trim_verbose:
            if ier != 1:
                print("No trim solution found. Scipy returned '{0}'.".format(mesg))
            print("\nFinal trim residuals: {0}".format(info_dict["fvec"]))

        # Parse trimmed state
        alpha = trim_settings[0]
        beta = trim_settings[1]
        controls = copy.deepcopy(self._fixed_controls)
        for i, name in enumerate(self._avail_controls):
            controls[name] = trim_settings[i+2]

        # Set state
        self._set_state_in_coordinated_turn(alpha, beta, controls)


    def _trim_residual_function(self, trim_vals):
        # Returns the trim residuals as a function of the control inputs

        # Unpack args
        alpha = trim_vals[0]
        beta = trim_vals[1]
        controls = copy.deepcopy(self._fixed_controls)
        for i, name in enumerate(self._avail_controls):
            controls[name] = trim_vals[i+2]

        # Calculate elevation angle
        theta = self._get_elevation(alpha, beta, self._bank, self._climb)

        # Output
        if self._trim_verbose:
            print("{0:>20.10f}{1:>20.10f}{2:>20.10f}{3:>20.10f}{4:>20.10f}{5:>20.10f}{6:>20.10f}".format(m.degrees(alpha), m.degrees(beta), *trim_vals[2:], m.degrees(theta)))

        # Set state
        self._set_state_in_coordinated_turn(alpha, beta, controls)

        # Get residuals
        dy_dt = self.dy_dt(0.0)
        return dy_dt[:6]


    def _set_state_in_coordinated_turn(self, alpha, beta, controls):

        # Set state
        theta = self._get_elevation(alpha, beta, self._bank, self._climb)
        C_theta = m.cos(theta)
        S_theta = m.sin(theta)
        C_phi = m.cos(self._bank)
        S_phi = m.sin(self._bank)
        C_a = m.cos(alpha)
        S_a = m.sin(alpha)
        C_B = m.cos(beta)
        S_B = m.sin(beta)
        D = m.sqrt(1-S_a*S_a*S_B*S_B)
        u = self._V0*C_a*C_B/D
        v = self._V0*C_a*S_B/D
        w = self._V0*S_a*C_B/D
        self.y[0] = u
        self.y[1] = v
        self.y[2] = w
        self.y[3:6] = self._get_rotation_rates(C_phi, S_phi, C_theta, S_theta, self._g, u, w)
        self.y[9:] = Euler2Quat([self._bank, theta, self._heading])

        # Set controls
        self.controls = controls


    def _initialize_state(self, param_dict):
        # Sets the state from the input by calling _trim or _set_initial_state
        trim = param_dict.get("trim", False)
        initial_state = param_dict.get("initial_state", False)
        landed = param_dict.get("landed", False)
        bungee = param_dict.get("elastic_launch", False)
        if len([x for x in [trim, initial_state, landed, bungee] if x is not False])>1:
            raise IOError("Please specify only one initial condition.")
        elif isinstance(trim, dict):
            self._trim(trim)
        elif isinstance(initial_state, dict):
            self._set_initial_state(initial_state)
        elif isinstance(landed, dict):
            self._set_landed(landed)
        elif isinstance(bungee, dict):
            self._set_elastic_launch(bungee)
        else:
            raise IOError("An initial condition must be specified.")


    def _set_initial_state(self, initial_state):
        # Sets the initial state of the aircraft without trimming

        # Store controls
        for name in self._control_names:
            self.controls[name] = initial_state.get("control_state", {}).get(name, 0.0)

        # Store state
        self.y[0:3] = import_value("velocity", initial_state, self._units, None)
        self.y[3:6] = import_value("angular_rates", initial_state, self._units, [0.0, 0.0, 0.0])
        self.y[6:9] = import_value("position", initial_state, self._units, None)
        try:
            self.y[9:] = import_value("orientation", initial_state, self._units, [1.0, 0.0, 0.0, 0.0])
        except ValueError:
            self.y[9:] = Euler2Quat(import_value("orientation", initial_state, self._units, [1.0, 0.0, 0.0, 0.0]))

        # Make sure we have some x velocity
        if self.y[0] == 0.0:
            self.y[0] = 1e-10


    def _set_landed(self, landed_dict):
        # Sets the aircraft landed at the given position at the given heading

        # Determine if we have landing gear to land on
        if len(self._landing_gear) == 0:
            raise IOError("Plane cannot be started in landing condition because no landing gear has been specified.")

        # Get user parameters
        pos = import_value("position", landed_dict, self._units, [0.0, 0.0, 0.0])
        heading = m.radians(import_value("heading", landed_dict, self._units, 0.0))

        # Determine height of landing gear
        max_height = 0.0
        for gear in self._landing_gear:
            if gear._pos[2]>max_height:
                max_height = gear._pos[2]

        # Set state
        pos[2] = -max_height
        self.y[0] = 1e-10 # To keep the get_FM methods from crashing
        self.y[1:3] = 0.0
        self.y[3:6] = 0.0
        self.y[6:9] = pos
        self.y[9:] = Euler2Quat([0.0, 0.0, heading])


    def _set_elastic_launch(self, bungee_dict):
        # Sets up a bungee launch

        # Set initial state normally, except velocity need not be specified
        bungee_dict["velocity"] = bungee_dict.get("velocity", [0.0, 0.0, 0.0])
        self._set_initial_state(bungee_dict)

        # Set up bungee parameters
        self._hooked = True
        self._anchor_pos = import_value("anchor_position", bungee_dict, self._units, [0.0, 0.0, 0.0])
        self._k_launch = import_value("stiffness", bungee_dict, self._units, None)
        self._l_unstretched = import_value("unstretched_length", bungee_dict, self._units, 0.0)
        self._launch_time = import_value("launch_time", bungee_dict, self._units, 0.0)


    def _component_effects(self, t, rho, u_inf, V):
        # Gives the forces and moments due to engines, landing gear, etc.

        # Get effect of engines
        FM = np.zeros(6)
        for engine in self._engines:
            FM += engine.get_thrust_FM(self.controls, rho, u_inf, V)

        # Get effect of landing_gear
        for gear in self._landing_gear:
            FM += gear.get_landing_FM(self.y, self.controls, rho, u_inf, V)

        # Get effect of bungee
        if self._hooked and t > self._launch_time:

            # Get bungee vector
            d_vec = Fixed2Body(-self.y[6:9]+self._anchor_pos, self.y[9:])-self._hook_pos

            # Check if we've come off the hook
            if d_vec[0] < 0.0:
                self._hooked = False

            else:

                # Determine the length of the bungee
                d = m.sqrt(d_vec[0]*d_vec[0]+d_vec[1]*d_vec[1]+d_vec[2]*d_vec[2])
                if d > self._l_unstretched:

                    # Calculate force
                    F = self._k_launch*(d-self._l_unstretched)

                    # Turn into a vector
                    F_vec = d_vec/d*F
                    FM[:3] += F_vec
                    FM[3:] += cross(self._hook_pos-self._CG, F_vec)

        return FM


    # These methods must be defined in any derived class. Any of the preceding methods can also be redefined.
    @abstractmethod
    def get_FM(self, t):
        pass


    @abstractmethod
    def get_graphics_info(self):
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

    def __init__(self, name, input_dict, density, units, param_dict, quit_flag, view_flag, pause_flag, data_flag, enable_interface):
        super().__init__(name, input_dict, density, units, param_dict, quit_flag, view_flag, pause_flag, data_flag, enable_interface)

        # Initialize density
        self._get_density = self._initialize_density(density)

        # Read in coefficients
        self._import_coefficients()

        # Import reference params
        self._import_reference_params()

        # Initialize accelerations
        self._t_prev = 0.0
        self._a_prev = None
        self._B_prev = None
        self._a_hat = 0.0
        self._B_hat = 0.0

        # Set initial state
        self._initialize_state(param_dict)


    def _initialize_density(self, density):
        # Sets up the density getter

        if density == "standard": # Standard atmospheric profile

            # English units
            if self._units == "English":
                def density_getter(alt):
                    return statee(alt)[-1]

            # SI units
            else:
                def density_getter(alt):
                    return statsi(alt)[-1]

        elif isinstance(density, float): # Constant
            self._density = density
            def density_getter(alt):
                return self._density

        else:
            raise IOError("{0} is not a valid density specification.".format(density))

        return density_getter


    def _import_coefficients(self):
        # Reads aerodynamic coefficients and derivatives in from input file

        # Lift
        self._CL0 = self._input_dict["coefficients"]["CL0"]
        self._CL_a = self._input_dict["coefficients"]["CL,a"]
        self._CL_a_hat = self._input_dict["coefficients"]["CL,a_hat"]
        self._CL_q = self._input_dict["coefficients"]["CL,q_bar"]

        # Drag
        self._CD0 = self._input_dict["coefficients"]["CD0"]
        self._CD1 = self._input_dict["coefficients"]["CD1"]
        self._CD2 = self._input_dict["coefficients"]["CD2"]
        self._CD3 = self._input_dict["coefficients"]["CD3"]
        self._CD_q = self._input_dict["coefficients"]["CD,q_bar"]
        self._CD_a_hat = self._input_dict["coefficients"]["CD,a_hat"]

        # Sideforce
        self._CS_b = self._input_dict["coefficients"]["CS,b"]
        self._CS_b_hat = self._input_dict["coefficients"]["CS,b_hat"]
        self._CS_p = self._input_dict["coefficients"]["CS,p_bar"]
        self._CS_r = self._input_dict["coefficients"]["CS,r_bar"]

        # Rolling Moment
        self._Cl_b = self._input_dict["coefficients"]["Cl,b"]
        self._Cl_b_hat = self._input_dict["coefficients"]["Cl,b_hat"]
        self._Cl_p = self._input_dict["coefficients"]["Cl,p_bar"]
        self._Cl_r = self._input_dict["coefficients"]["Cl,r_bar"]

        # Pitching moment
        self._Cm0 = self._input_dict["coefficients"]["Cm0"]
        self._Cm_a = self._input_dict["coefficients"]["Cm,a"]
        self._Cm_a_hat = self._input_dict["coefficients"]["Cm,a_hat"]
        self._Cm_q = self._input_dict["coefficients"]["Cm,q_bar"]

        # Yawing moment
        self._Cn_b = self._input_dict["coefficients"]["Cn,b"]
        self._Cn_b_hat = self._input_dict["coefficients"]["Cn,b_hat"]
        self._Cn_p = self._input_dict["coefficients"]["Cn,p_bar"]
        self._Cn_r = self._input_dict["coefficients"]["Cn,r_bar"]

        # Parse control derivatives and reference control settings
        self._control_derivs = {}
        for name in self._control_names:

            # Get derivatives
            self._control_derivs[name] = {}
            derivs = self._input_dict["coefficients"].get(name, {})
            self._control_derivs[name]["CL"] = derivs.get("CL", 0.0)
            self._control_derivs[name]["CD"] = derivs.get("CD", 0.0)
            self._control_derivs[name]["Cm"] = derivs.get("Cm", 0.0)
            self._control_derivs[name]["CS"] = derivs.get("CS", 0.0)
            self._control_derivs[name]["Cl"] = derivs.get("Cl", 0.0)
            self._control_derivs[name]["Cn"] = derivs.get("Cn", 0.0)


    def _import_reference_params(self):
        # Parses the reference state from the input file

        # Get reference lengths and area
        self._Sw = self._input_dict["reference"].get("area", None)
        self._bw = self._input_dict["reference"].get("lateral_length", None)
        self._cw = self._input_dict["reference"].get("longitudinal_length", None)

        # Check we have all the reference lengths we need
        if self._Sw is not None and self._bw is not None and self._cw is None:
            self._cw = self._Sw/self._bw
        elif self._Sw is None and self._bw is not None and self._cw is not None:
            self._Sw = self._cw*self._bw
        elif self._Sw is not None and self._bw is None and self._cw is not None:
            self._bw = self._Sw/self._cw
        elif not (self._Sw is not None and self._bw is not None and self._cw is not None):
            raise IOError("At least two of area, lateral length, or longitudinal length must be specified.")


    def _obsolete_trim(self, trim_dict):
        # Trims the aircraft according to the input conditions

        # Get initial params
        V0 = trim_dict["airspeed"]
        self.y[6:9] = trim_dict["position"]
        climb = m.radians(trim_dict["climb_angle"])
        bank = m.radians(trim_dict["bank_angle"])
        heading = m.radians(trim_dict["heading"])
        verbose = trim_dict.get("verbose", False)

        # Parse controls
        avail_controls = trim_dict.get("trim_controls", list(self.controls.keys()))
        fixed_controls = trim_dict.get("fixed_controls", {})
        for name in self._control_names:
            if name in avail_controls:
                continue
            fixed_controls[name] = fixed_controls.get(name, 0.0)

        if len(avail_controls) != 4:
            raise IOError("Exactly 4 controls must be used to trim the aircraft. Got {0}.".format(len(avail_controls)))

        # Reference parameters
        V0_inv = 1.0/V0
        const = 0.5*V0_inv
        g = self._g
        rho = self._get_density(-self.y[8])
        redim = 0.5*rho*V0*V0*self._Sw
        redim_inv = 1.0/redim
        CW = self._W*redim_inv

        # Get thrust and thrust moment derivatives
        thrust_derivs = {}
        thrust_moment_derivs = {}
        for control in avail_controls:
            thrust_derivs[control] = np.zeros(3)
            thrust_moment_derivs[control] = np.zeros(3)
            for engine in self._engines:
                # These derivatives must be converted to degrees because all controls are handled
                # in radians for the trim algorithm, regardless of whether they're actually angular.
                thrust_derivs[control] += np.degrees(engine.get_thrust_deriv(control, rho, V0))
                thrust_moment_derivs[control] += np.degrees(engine.get_thrust_moment_deriv(control, rho, V0))

        # Initial guess
        alpha = 0.0
        beta = 0.0
        theta = self._get_elevation(alpha, beta, bank, climb)
        A = np.zeros((6,6))
        B = np.zeros(6)
        old_trim_vals = np.zeros(6) # These are deviations from the reference setting
        old_trim_vals[4] = m.radians(-5.0) # For comparing with Troy
        C_phi = m.cos(bank)
        S_phi = m.sin(bank)
        theta = self._get_elevation(0.0, 0.0, bank, climb)

        # Initialize output
        if verbose:
            print("Trimming at {0} deg bank and {1} deg climb...".format(m.degrees(bank), m.degrees(climb)))
            header = ["{0:>20}{1:>20}".format("Alpha [deg]", "Beta [deg]")]
            for name in avail_controls:
                header.append("{0:>20}".format(name.title()))
            header.append("{0:>20}".format("Elevation [deg]"))
            print("".join(header))

        # Iterate until trim values converge
        approx_error = 1
        iterations = 0
        while approx_error > 1e-25:

            # Extract trim values
            alpha = old_trim_vals[0]
            beta = old_trim_vals[1]

            # Calulate trig values
            C_theta = m.cos(theta)
            S_theta = m.sin(theta)
            C_a = m.cos(alpha)
            S_a = m.sin(alpha)
            C_B = m.cos(beta)
            S_B = m.sin(beta)

            # Get states
            D = m.sqrt(1-S_a*S_a*S_B*S_B)
            u = V0*C_a*C_B/D
            v = V0*C_a*S_B/D
            w = V0*S_a*C_B/D

            # Get rotation rates
            p, q, r = self._get_rotation_rates(C_phi, S_phi, C_theta, S_theta, g, u, w)
            pr = p*r
            pq = p*q
            qr = q*r
            p2 = p*p
            q2 = q*q
            r2 = r*r
            p_bar = self._bw*const*p
            q_bar = self._cw*const*q
            r_bar = self._bw*const*r

            # Calculate aerodynamic coefficients
            CL = self._CL0+self._CL_a*alpha+self._CL_q*q_bar
            CS = self._CS_b*beta+self._CS_p*p_bar+self._CS_r*r_bar
            CD = self._CD0+self._CD_q*q_bar

            # Determine influence of trim controls
            for i,name in enumerate(avail_controls):
                control_deriv = self._control_derivs[name]
                deflection = old_trim_vals[2+i]
                CL += deflection*control_deriv["CL"]
                CD += deflection*control_deriv["CD"]
                CS += deflection*control_deriv["CS"]

            # Determine influence of fixed controls
            for key,value in fixed_controls.items():
                control_deriv = self._control_derivs[key]
                CL += m.radians(value-self._control_ref[key])*control_deriv["CL"]
                CD += m.radians(value-self._control_ref[key])*control_deriv["CD"]
                CS += m.radians(value-self._control_ref[key])*control_deriv["CS"]

            # Factor in terms involving CL and CS
            CD += self._CD1*CL+self._CD2*CL*CL+self._CD3*CS*CS

            # Populate A matrix
            # Terms dependent on alpha
            A[2,0] = -self._CL_a*C_a

            # Terms dependent on beta
            A[1,1] = self._CS_b*C_B
            A[3,1] = self._bw*self._Cl_b

            # Now loop through trim controls
            for i,name in enumerate(avail_controls):
                A[0,2+i] = redim_inv*thrust_derivs[name][0]-self._control_derivs[name]["CD"]*u*V0_inv
                A[1,2+i] = redim_inv*thrust_derivs[name][1]+self._control_derivs[name]["CS"]*C_B
                A[2,2+i] = redim_inv*thrust_derivs[name][2]-self._control_derivs[name]["CL"]*C_a
                A[3,2+i] = redim_inv*thrust_moment_derivs[name][0]+self._control_derivs[name]["Cl"]*self._bw
                A[4,2+i] = redim_inv*thrust_moment_derivs[name][1]+self._control_derivs[name]["Cm"]*self._cw
                A[5,2+i] = redim_inv*thrust_moment_derivs[name][2]+self._control_derivs[name]["Cn"]*self._bw

            # Populate B vector
            # Aerodynamic contributions
            B[0] = -CL*S_a+CS*S_B+(self._CD0+self._CD1*CL+self._CD2*CL*CL+self._CD3*CS*CS+self._CD_q*q_bar)*u*V0_inv
            B[1] = (-self._CS_p*p_bar-self._CS_r*r_bar)*C_B+CD*v*V0_inv
            B[2] = (self._CL0+self._CL_q*q_bar)*C_a+CD*w*V0_inv
            B[3] = -self._bw*(self._Cl_p*p_bar+(self._Cl_r/self._CL0)*CL*r_bar)
            B[4] = -self._cw*(self._Cm0+(self._Cm_a/self._CL_a)*(CL*u*V0_inv-self._CL0+CD*w*V0_inv)+self._Cm_q*q_bar)
            B[5] = -self._bw*((self._Cn_b/self._CS_b)*(CS*u*V0_inv-CD*v*V0_inv)+(self._Cn_p/self._CL0)*CL*p_bar+self._Cn_r*r_bar)

            # Contributions of fixed controls
            for key, value in fixed_controls.items():
                # Control-specific information
                control_deriv = self._control_derivs[key]
                delta_control = m.radians(value-self._control_ref[key])

                # Change in aerodynamics due to controls
                CD_control = delta_control*control_deriv["CD"]
                CL_control = delta_control*control_deriv["CL"]
                CY_control = delta_control*control_deriv["CS"]

                # Apply to B vector
                B[0] += -CL_control*S_a+CY_control*S_B+CD_control*u*V0_inv
                B[1] += -CY_control*C_B+CD_control*v*V0_inv
                B[2] +=  CL_control*C_a+CD_control*w*V0_inv
                B[3] += -self._bw*control_deriv["Cl"]*delta_control
                B[4] += -self._cw*control_deriv["Cm"]*delta_control
                B[5] += -self._bw*control_deriv["Cn"]*delta_control

            # Inertial and gyroscopic contributions
            B[0] += CW*(S_theta-(r*v-q*w)/g)
            B[1] += CW*(-S_phi*C_theta-(p*w-r*u)/g)
            B[2] += CW*(-C_phi*C_theta-(q*u-p*v)/g)
            B[3] += redim_inv*( self._hz*q-self._hy*r-self._I_diff_yz*qr-self._I_yz*(q2-r2)-self._I_xz*pq+self._I_xy*pr)
            B[4] += redim_inv*(-self._hz*p+self._hx*r-self._I_diff_zx*pr-self._I_xz*(r2-p2)-self._I_xy*qr+self._I_yz*pq)
            B[5] += redim_inv*( self._hy*p-self._hx*q-self._I_diff_xy*pq-self._I_xy*(p2-q2)-self._I_yz*pr+self._I_xz*qr)

            # Solve
            trim_vals = np.linalg.solve(A,B)

            # Update for next iteration
            alpha = trim_vals[0]
            beta = trim_vals[1]
            theta = self._get_elevation(alpha, beta, bank, climb)
            approx_error = np.max(np.abs(trim_vals-old_trim_vals))
            old_trim_vals = np.copy(trim_vals)

            # Output
            if verbose:
                output = ["{0:>20.10f}{1:>20.10f}".format(m.degrees(alpha), m.degrees(beta))]
                for i, name in enumerate(avail_controls):
                    output.append("{0:>20.10f}".format(m.degrees(old_trim_vals[i+2]+m.radians(self._control_ref[name]))))
                output.append("{0:>20.10f}".format(m.degrees(theta)))
                print("".join(output))

            # Check for non-convergence
            iterations += 1
            if iterations > 100:
                if verbose:
                    print("Unable to trim at the desired attitude.")
                break

        # Apply trim state
        alpha = trim_vals[0]
        beta = trim_vals[1]
        theta = self._get_elevation(alpha, beta, bank, climb)
        C_theta = m.cos(theta)
        S_theta = m.sin(theta)
        C_phi = m.cos(bank)
        S_phi = m.sin(bank)
        C_a = m.cos(alpha)
        S_a = m.sin(alpha)
        C_B = m.cos(beta)
        S_B = m.sin(beta)
        D = m.sqrt(1-S_a*S_a*S_B*S_B)
        u = V0*C_a*C_B/D
        v = V0*C_a*S_B/D
        w = V0*S_a*C_B/D
        self.y[0] = u
        self.y[1] = v
        self.y[2] = w
        self.y[3:6] = self._get_rotation_rates(C_phi, S_phi, C_theta, S_theta, g, u, w)
        self.y[9:] = Euler2Quat([bank, theta, heading])

        # Apply trim controls
        # Variable
        for i,name in enumerate(avail_controls):
            self.controls[name] = m.degrees(trim_vals[i+2]+m.radians(self._control_ref[name]))
        # Fixed
        for key, value in fixed_controls:
            self.controls[key] = value


    def get_FM(self, t):
        """Returns the aerodynamic forces and moments."""
        np.set_printoptions(precision=12)

        # Get control state
        self.controls = self.controller.get_control(t, self.y, self.controls)

        # Declare force and moment vector
        FM = np.zeros(6)

        # Get states
        rho = self._get_density(-self.y[8])
        u = self.y[0]
        v = self.y[1]
        w = self.y[2]
        p = self.y[3]
        q = self.y[4]
        r = self.y[5]
        V = m.sqrt(u*u+v*v+w*w)
        V_inv = 1.0/V
        a = m.atan2(w,u)
        B = m.asin(v/V)
        const = 0.5*V_inv
        p_bar = self._bw*p*const
        q_bar = self._cw*q*const
        r_bar = self._bw*r*const
        u_inf = self.y[0:3]*V_inv

        # Get accelerations
        dt = t-self._t_prev
        if dt>1e-10 and self._a_prev is not None and self._B_prev is not None:
            self._a_hat = 0.5*self._cw*V_inv*(a-self._a_prev)/dt
            self._B_hat = 0.5*self._bw*V_inv*(B-self._B_prev)/dt

        # Store for next evaluation
        self._t_prev = t
        self._a_prev = a
        self._B_prev = B

        # Get redimensionalizer
        redim = 0.5*rho*V*V*self._Sw

        # Determine coefficients without knowing final values for CL and CS
        CL = self._CL0+self._CL_a*a+self._CL_q*q_bar+self._CL_a_hat*self._a_hat
        CS = self._CS_b*B+self._CS_p*p_bar+self._CS_r*r_bar+self._CS_b_hat*self._B_hat
        CD = self._CD0+self._CD_q*q_bar+self._CD_a_hat*self._a_hat
        Cl = self._Cl_b*B+self._Cl_p*p_bar+self._Cl_r*r_bar+self._Cl_b_hat*self._B_hat
        Cm = self._Cm0+self._Cm_a*a+self._Cm_q*q_bar+self._Cm_a_hat*self._a_hat
        Cn = self._Cn_b*B+self._Cn_p*p_bar+self._Cn_r*r_bar+self._Cn_b_hat*self._B_hat

        # Determine influence of controls
        for key, value in self.controls.items():
            control_deriv = self._control_derivs[key]
            control_def = m.radians(value)
            CL += control_def*control_deriv["CL"]
            CD += control_def*control_deriv["CD"]
            CS += control_def*control_deriv["CS"]
            Cl += control_def*control_deriv["Cl"]
            Cm += control_def*control_deriv["Cm"]
            Cn += control_def*control_deriv["Cn"]

        # Factor in drag polar terms
        CD += self._CD1*CL+self._CD2*CL*CL+self._CD3*CS*CS

        # Get trig vals
        C_a = m.cos(a)
        S_a = w/u*C_a
        C_B = m.cos(B)
        S_B = v/V

        # Correct for stall
        CL, CD, CS, Cl, Cm, Cn = self._correct_stall(CL, CD, CS, Cl, Cm, Cn, a, B, S_a, S_B, C_a, C_B)

        # Apply aerodynamic angles and dimensionalize
        FM[0] = redim*(CL*S_a-CS*C_a*S_B-CD*C_a*C_B)
        FM[1] = redim*(CS*C_B-CD*S_B)
        FM[2] = redim*(-CL*C_a-CS*S_a*S_B-CD*S_a*C_B)
        FM[3] = redim*Cl*self._bw
        FM[4] = redim*Cm*self._cw
        FM[5] = redim*Cn*self._bw

        # Get component effects
        FM_e = self._component_effects(t, rho, u_inf, V)
        FM += FM_e

        return FM


    def get_graphics_info(self):
        """Returns the .obj, .vs, .fs, and texture .jpg files for the aircraft."""

        # Get path to graphics objects
        self._pylot_path = os.path.dirname(__file__)
        self._graphics_path = os.path.join(self._pylot_path, "graphics")
        self._objects_path = os.path.join(self._graphics_path, "objects")
        self._shaders_path = os.path.join(self._graphics_path, "shaders")
        self._textures_path = os.path.join(self._graphics_path, "textures")

        # Load input
        graphics_dict = self._input_dict.get("graphics", {})
        info = {}
        info["obj_file"] = graphics_dict.get("obj_file", os.path.join(self._objects_path, "Cessna.obj"))
        info["v_shader_file"] = graphics_dict.get("vertex_shader_file", os.path.join(self._shaders_path, "aircraft.vs"))
        info["f_shader_file"] = graphics_dict.get("face_shader_file", os.path.join(self._shaders_path, "aircraft.fs"))
        info["texture_file"] = graphics_dict.get("texture_file", os.path.join(self._textures_path, "cessna_texture.jpg"))
        info["l_ref_lon"] = self._cw
        info["l_ref_lat"] = self._bw

        return info


class MachUpXAirplane(BaseAircraft):
    """An airplane defined by MachUpX.
    
    Parameters
    ----------
    name : str
        Name of the aircraft.

    input_dict : dict
        Dictionary describing the airplane.
    """

    def __init__(self, name, input_dict, density, units, param_dict, quit_flag, view_flag, pause_flag, data_flag, enable_interface):
        super().__init__(name, input_dict, density, units, param_dict, quit_flag, view_flag, pause_flag, data_flag, enable_interface)

        # Create MachUpX scene
        import machupX as mx
        scene_dict = {
            "solver" : {
                "type" : self._input_dict["aero_model"].get("solver", "linear"),
            },
            "units" : units,
            "scene" : {
                "atmosphere" : {
                    "density" : density
                },
                "aircraft" : {
                    name : {
                        "file" : param_dict["file"],
                        "state" : {
                            "position" : [0.0, 0.0, 0.0],
                            "velocity" : 100 # These are arbitrary, as the first call to get_FM() will set the actual state
                        }
                    }
                }
            }
        }
        self._mx_scene = mx.Scene(scene_dict)

        # Get reference params
        self._Sw, self._cw, self._bw = self._mx_scene.get_aircraft_reference_geometry(name)

        # Set initial state
        self._initialize_state(param_dict)


    def _update_machupx_state(self):
        # Passes the sim object state to MachUpX

        # Set MachUpX state
        mx_state = {
            "position" : list(self.y[6:9]),
            "velocity" : list(self.y[0:3]),
            "orientation" : list(self.y[9:]),
            "angular_rates" : list(self.y[3:6])
        }
        self._mx_scene.set_aircraft_state(state=mx_state, aircraft=self.name)

        # Update controls
        self._mx_scene.set_aircraft_control_state(control_state=self.controls, aircraft=self.name)


    def get_FM(self, t):

        # Initialize
        FM = np.zeros(6)
        self.controls = self.controller.get_control(t, self.y, self.controls)
        self._update_machupx_state()

        # Get redimensionalizer
        rho = self._mx_scene._get_density(self.y[6:9])
        u = self.y[0]
        v = self.y[1]
        w = self.y[2]
        V = m.sqrt(u*u+v*v+w*w)
        redim = 0.5*rho*V*V*self._Sw
        u_inf = self.y[0:3]/V
        a = m.atan2(w,u)
        B = m.asin(u_inf[1])
        C_B = m.cos(B)
        S_B = u_inf[1]
        C_a = m.cos(a)
        S_a = w/u*C_a

        # Get MachUpX predicted coefficients
        coefs = self._mx_scene.solve_forces(dimensional=False, body_frame=True, wind_frame=True)[self.name]["total"]

        # Correct for stall
        CL, CD, CS, Cl, Cm, Cn = self._correct_stall(coefs["CL"], coefs["CD"], coefs["CS"], coefs["Cl"], coefs["Cm"], coefs["Cn"], a, B, S_a, S_B, C_a, C_B)

        # Get forces
        FM[0] = redim*(CL*S_a-CS*C_a*S_B-CD*C_a*C_B)
        FM[1] = redim*(CS*C_B-CD*S_B)
        FM[2] = redim*(-CL*C_a-CS*S_a*S_B-CD*S_a*C_B)
        FM[3] = redim*Cl*self._bw
        FM[4] = redim*Cm*self._cw
        FM[5] = redim*Cn*self._bw

        # Get component effects
        FM += self._component_effects(t, rho, u_inf, V)

        return FM


    def get_graphics_info(self):
        """Returns the .obj, .vs, .fs, and texture .jpg files for the aircraft."""

        # Get path to graphics objects
        self._pylot_path = os.path.dirname(__file__)
        self._graphics_path = os.path.join(self._pylot_path, "graphics")
        self._objects_path = os.path.join(self._graphics_path, "objects")
        self._shaders_path = os.path.join(self._graphics_path, "shaders")
        self._textures_path = os.path.join(self._graphics_path, "textures")

        # Load inputs
        graphics_dict = self._input_dict.get("graphics", {})
        info = {}
        info["obj_file"] = graphics_dict.get("obj_file", None)
        info["v_shader_file"] = graphics_dict.get("vertex_shader_file", os.path.join(self._shaders_path, "aircraft.vs"))
        info["f_shader_file"] = graphics_dict.get("face_shader_file", os.path.join(self._shaders_path, "aircraft.fs"))
        
        # Get the obj file from MachUpX
        if info["obj_file"] is None:

            # The user may not have the FreeCAD modules installed, so we'll need to fail gracefully if so
            try:

                # Import necessary files
                print("\nGenerating 3D model from MachUpX...")
                import FreeCAD
                import Mesh

                # Create stl using MachUpX (involves temporarily resetting the state)
                stl_file = "airplane.stl"
                obj_file = "airplane.obj"
                reset_state = {
                    "position" : [0.0, 0.0, 0.0],
                    "velocity" : [10.0, 0.0, 0.0],
                    "orientation" : [1.0, 0.0, 0.0, 0.0]
                }
                self._mx_scene.set_aircraft_state(state=reset_state, aircraft=self.name)
                self._mx_scene.export_stl(filename=stl_file, section_resolution=50, aircraft=self.name)
                self._update_machupx_state()
                
                # Convert to obj
                Mesh.open(stl_file)
                o = FreeCAD.getDocument("Unnamed").findObjects()[0]
                Mesh.export([o], obj_file)
                print("\n")

                # Save
                info["obj_file"] = graphics_dict.get("obj_file", obj_file)
                info["texture_file"] = graphics_dict.get("texture_file", os.path.join(self._textures_path, "grey.jpg"))

            # Just render a Cessna... ;)
            except ImportError:
                print("\nFreeCAD modules not found. Reverting to Cessna model...")
                info["obj_file"] = graphics_dict.get("obj_file", os.path.join(self._objects_path, "Cessna.obj"))
                info["texture_file"] = graphics_dict.get("texture_file", os.path.join(self._textures_path, "cessna_texture.jpg"))

        # Pass off reference lengths
        info["l_ref_lon"] = self._cw
        info["l_ref_lat"] = self._bw

        return info
