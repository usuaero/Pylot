# Classes describing the aircraft used by the simulator

import math as m
import numpy as np
import machupX as mx
from abc import abstractmethod
from .helpers import *
from .std_atmos import *
from .controllers import *
from .engine import *

class BaseAircraft:
    """A base class for aircraft to be used in the simulator.

    Parameters
    ----------
    name : str
        Name of the aircraft.

    input_dict : dict
        Dictionary describing the airplane.
    """

    def __init__(self, name, input_dict, density, units, param_dict):

        # Store input
        self._input_dict = input_dict

        # Initialize state
        self.y = np.zeros(13)

        # Setup output
        state_output = param_dict.get("state_output", None)
        self._output_state = state_output is not None
        if self._output_state:
            self._output_handle = open(state_output, 'w')
            header = "   Time              u                 v                 w                 p                 q                 r                 x                 y                 z                 eo                e1                e2                e3"
            print(header, file=self._output_handle)

        # Determine units and set gravity
        self._units = units
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
        self._hx, self._hy, self._hz = import_value("angular_momentum", self._input_dict, self._units, [0.0, 0.0, 0.0])

        # Load engines
        self._engines = []
        self._num_engines = 0
        for key, value in self._input_dict.get("engines", {}).items():
            self._engines.append(Engine(key, **value, units=self._units))
            self._num_engines += 1

        # Load controls
        controller = param_dict.get("controller", None)
        self._initialize_controller(controller)


    def _initialize_controller(self, control_type):
        # Sets up the control input for the aircraft

        # No control input
        if control_type is None:
            self._controller = NoController(self._input_dict.get("controls", {}))

        # Joystick
        elif control_type == "joystick":
            self._controller = JoystickAircraftController(self._input_dict.get("controls", {}))

        # Keyboard
        elif control_type == "keyboard":
            pass

        # User-defined
        elif control_type == "user-defined":
            pass

        # Time sequence file
        else:
            pass

        # Setup storage
        self._control_names = self._controller.get_control_names()
        self._controls = {}


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


    def normalize(self):
        """Normalizes the orientation quaternion."""
        self.y[9:] = NormalizeQuaternion(self.y[9:])


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

    def __init__(self, name, input_dict, density, units, param_dict):
        super().__init__(name, input_dict, density, units, param_dict)

        # Initialize density
        self._get_density = self._initialize_density(density)

        # Read in coefficients
        self._import_coefficients()

        # Import reference params
        self._import_reference_state()

        # Set initial state
        trim = param_dict.get("trim", False)
        initial_state = param_dict.get("initial_state", False)
        if trim and initial_state:
            raise IOError("Both a trim condition and an initial state may not be specified.")
        elif trim:
            self._trim(trim)
        elif initial_state:
            self._set_initial_state(initial_state)
        else:
            raise IOError("An initial condition was not specified!")


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


    def _import_reference_state(self):
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

        # Parse the reference state
        rho_ref = import_value("density", self._input_dict["reference"], self._units, None)
        V_ref = import_value("airspeed", self._input_dict["reference"], self._units, None)
        L_ref = import_value("lift", self._input_dict["reference"], self._units, None)
        self._CL_ref = L_ref/(0.5*rho_ref*V_ref*V_ref*self._Sw)

        # Determine drag polar
        self._CD2 = self._CD_a2/(2*self._CL_a*self._CL_a)
        self._CD1 = self._CD_a/self._CL_a-2*self._CD2*self._CL_ref
        self._CD0 = self._CD-self._CD1*self._CL_ref-self._CD2*self._CL_ref*self._CL_ref

        # Determine reference aerodynamic moments
        # This assumes thrust is evenly distributed among all engines
        engine_CT = self._CD/self._num_engines
        engine_CM = np.zeros(3)
        for engine in self._engines:
            engine_CM += engine.get_unit_thrust_moment()*engine_CT

        self._Cl_ref = -engine_CM[0]/self._bw
        self._Cm_ref = -engine_CM[1]/self._cw
        self._Cn_ref = -engine_CM[2]/self._bw


    def _import_coefficients(self):
        # Reads aerodynamic coefficients and derivatives in from input file
        self._CL_a = self._input_dict["coefficients"]["CL,a"]
        self._CD = self._input_dict["coefficients"]["CD"]
        self._CD_a = self._input_dict["coefficients"]["CD,a"]
        self._CD_a2 = self._input_dict["coefficients"]["CD,a,a"]
        self._CD3 = self._input_dict["coefficients"]["CD3"]
        self._Cm_a = self._input_dict["coefficients"]["Cm,a"]
        self._CY_b = self._input_dict["coefficients"]["CY,b"]
        self._Cl_b = self._input_dict["coefficients"]["Cl,b"]
        self._Cn_b = self._input_dict["coefficients"]["Cn,b"]
        self._CL_q = self._input_dict["coefficients"]["CL,q"]
        self._CD_q = self._input_dict["coefficients"]["CD,q"]
        self._Cm_q = self._input_dict["coefficients"]["Cm,q"]
        self._CY_p = self._input_dict["coefficients"]["CY,p"]
        self._Cl_p = self._input_dict["coefficients"]["Cl,p"]
        self._Cn_p = self._input_dict["coefficients"]["Cn,p"]
        self._CY_r = self._input_dict["coefficients"]["CY,r"]
        self._Cl_r = self._input_dict["coefficients"]["Cl,r"]
        self._Cn_r = self._input_dict["coefficients"]["Cn,r"]

        # Parse control derivatives and reference control settings
        self._control_derivs = {}
        self._control_ref = {}
        for name in self._control_names:

            # Get derivatives
            self._control_derivs[name] = {}
            derivs = self._input_dict["coefficients"].get(name, {})
            self._control_derivs[name]["CL"] = derivs.get("CL", 0.0)
            self._control_derivs[name]["CD"] = derivs.get("CD", 0.0)
            self._control_derivs[name]["Cm"] = derivs.get("Cm", 0.0)
            self._control_derivs[name]["CY"] = derivs.get("CY", 0.0)
            self._control_derivs[name]["Cl"] = derivs.get("Cl", 0.0)
            self._control_derivs[name]["Cn"] = derivs.get("Cn", 0.0)

            # Get reference control deflections
            self._control_ref[name] = self._input_dict["reference"].get("controls", {}).get(name, 0.0)


    def _trim(self, trim_dict):
        # Trims the aircraft according to the input conditions

        # Get initial params
        V0 = trim_dict["airspeed"]
        self.y[6:9] = trim_dict["position"]
        climb = m.radians(trim_dict["climb_angle"])
        bank = m.radians(trim_dict["bank_angle"])
        heading = m.radians(trim_dict["heading"])
        verbose = trim_dict["verbose"]

        # Parse controls
        avail_controls = trim_dict.get("trim_controls", list(self._controls.keys()))
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
        C_phi = cos(bank)
        S_phi = sin(bank)

        # Initialize output
        if verbose:
            print("Trimming at {0} deg bank and {1} deg climb...".format(degrees(bank), degrees(climb)))
            header = ["{0:>20}{1:>20}".format("Alpha [deg]", "Beta [deg]")]
            for name in avail_controls:
                header.append("{0:>20}".format(name))
            header.append("{0:>20}".format("Elevation [deg]"))
            print("".join(header))

        # Iterate until trim values converge
        approx_error = 1
        iterations = 0
        while approx_error > 1e-25:

            # Extract trim values
            alpha = old_trim_vals[0]
            beta = old_trim_vals[1]
            theta = self._get_elevation(alpha, beta, bank, climb)
            print(theta)

            # Calulate trig values
            C_theta = cos(theta)
            S_theta = sin(theta)
            C_a = cos(alpha)
            S_a = sin(alpha)
            C_B = cos(beta)
            S_B = sin(beta)

            # Get states
            D = sqrt(1-S_a*S_a*S_B*S_B)
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
            print("p",p)
            print("q",q)
            print("r",r)

            # Calculate aerodynamic coefficients
            CL = self._CL_ref+self._CL_a*alpha+self._CL_q*q_bar
            CS = self._CY_b*beta+self._CY_p*p_bar+self._CY_r*r_bar
            CD = self._CD0+self._CD_q*q_bar

            # Determine influence of trim controls
            for i,name in enumerate(avail_controls):
                control_deriv = self._control_derivs[name]
                deflection = old_trim_vals[2+i]
                CL += deflection*control_deriv["CL"]
                CD += deflection*control_deriv["CD"]
                CS += deflection*control_deriv["CY"]

            # Determine influence of fixed controls
            for key,value in fixed_controls.items():
                control_deriv = self._control_derivs[key]
                CL += m.radians(value-self._control_ref[key])*control_deriv["CL"]
                CD += m.radians(value-self._control_ref[key])*control_deriv["CD"]
                CS += m.radians(value-self._control_ref[key])*control_deriv["CY"]

            # Factor in terms involving CL and CS
            CD += self._CD1*CL+self._CD2*CL*CL+self._CD3*CS*CS
            print("CL", CL)
            print("CD", CD)
            print("CS", CS)

            # Populate A matrix
            # Terms dependent on alpha
            A[2,0] = -self._CL_a*C_a

            # Terms dependent on beta
            A[1,1] = self._CY_b*C_B
            A[3,1] = self._bw*self._Cl_b

            # Now loop through trim controls
            for i,name in enumerate(avail_controls):
                A[0,2+i] = redim_inv*thrust_derivs[name][0]-self._control_derivs[name]["CD"]*u*V0_inv
                A[1,2+i] = redim_inv*thrust_derivs[name][1]+self._control_derivs[name]["CY"]*C_B
                A[2,2+i] = redim_inv*thrust_derivs[name][2]-self._control_derivs[name]["CL"]*C_a
                A[3,2+i] = redim_inv*thrust_moment_derivs[name][0]+self._control_derivs[name]["Cl"]*self._bw
                A[4,2+i] = redim_inv*thrust_moment_derivs[name][1]+self._control_derivs[name]["Cm"]*self._cw
                A[5,2+i] = redim_inv*thrust_moment_derivs[name][2]+self._control_derivs[name]["Cn"]*self._bw

            # Populate B vector
            # Aerodynamic contributions
            B[0] = -CL*S_a+CS*S_B+(self._CD0+self._CD1*CL+self._CD2*CL*CL+self._CD3*CS*CS+self._CD_q*q_bar)*u*V0_inv
            B[1] = (-self._CY_p*p_bar-self._CY_r*r_bar)*C_B+CD*v*V0_inv
            B[2] = (self._CL_ref+self._CL_q*q_bar)*C_a+CD*w*V0_inv
            B[3] = -self._bw*(self._Cl_ref+self._Cl_p*p_bar+(self._Cl_r/self._CL_ref)*CL*r_bar)
            B[4] = -self._cw*(self._Cm_ref+(self._Cm_a/self._CL_a)*(CL*u*V0_inv-self._CL_ref+CD*w*V0_inv)+self._Cm_q*q_bar)
            B[5] = -self._bw*(self._Cn_ref+(self._Cn_b/self._CY_b)*(CS*u*V0_inv-CD*v*V0_inv)+(self._Cn_p/self._CL_ref)*CL*p_bar+self._Cn_r*r_bar)

            # Contributions of fixed controls
            for key, value in fixed_controls.items():
                control_deriv = self._control_derivs[key]
                delta_control = m.radians(value-self._control_ref[key])
                # TODO: Add CL, CD, and CY to all these equations
                CD_control = delta_control*control_deriv["CD"]
                CL_control = delta_control*control_deriv["CL"]
                CY_control = delta_control*control_deriv["CY"]
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

            print(A)
            print(B)

            # Solve
            trim_vals = np.linalg.solve(A,B)
            print(trim_vals)

            # Update for next iteration
            alpha = trim_vals[0]
            beta = trim_vals[1]
            approx_error = np.max(np.abs(trim_vals-old_trim_vals))
            theta = self._get_elevation(alpha, beta, bank, climb)
            old_trim_vals = np.copy(trim_vals)

            # Output
            if verbose:
                output = ["{0:>20.10f}{1:>20.10f}".format(degrees(alpha), degrees(beta))]
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
        C_theta = cos(theta)
        S_theta = sin(theta)
        C_phi = cos(bank)
        S_phi = sin(bank)
        C_a = cos(alpha)
        S_a = sin(alpha)
        C_B = cos(beta)
        S_B = sin(beta)
        D = sqrt(1-S_a*S_a*S_B*S_B)
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
            self._controls[name] = m.degrees(trim_vals[i+2]+m.radians(self._control_ref[name]))
        # Fixed
        for key, value in fixed_controls:
            self._controls[key] = value


    def _get_elevation(self, alpha, beta, phi, gamma):
        # Calculates the elevation angle, theta, based on the other known angles

        # Calculate constants
        C_a = cos(alpha)
        S_a = sin(alpha)
        C_B = cos(beta)
        S_B = sin(beta)
        C_phi = cos(phi)
        S_phi = sin(phi)
        S_gamma = sin(gamma)

        D = sqrt(1-S_a*S_a*S_B*S_B)
        E = S_phi*C_a*S_B+C_phi*S_a*C_B
        E2 = E*E
        C_aB = C_a*C_B
        C_aB2 = C_aB*C_aB

        # Get two possibilities
        A = C_aB*D*S_gamma
        B = E*sqrt(C_aB2-D*D*S_gamma*S_gamma+E2)
        C = C_aB2+E2
        theta1 = asin((A+B)/C)
        theta2 = asin((A-B)/C)
        print(theta1)
        print(theta2)

        # Check which one is closest
        if abs(D*S_gamma+sin(theta1)*C_aB-cos(theta1)*E) < abs(D*S_gamma+sin(theta2)*C_aB-cos(theta2)*E):
            return theta1
        else:
            return theta2


    def _get_rotation_rates(self, C_phi, S_phi, C_theta, S_theta, g, u, w):
        # Returns the rotation rates in a steady, coordinated turn

        omega = g*S_phi*C_theta/(S_theta*w+C_phi*C_theta*u)
        return -S_theta*omega, S_phi*C_theta*omega, C_phi*C_theta*omega


    def _set_initial_state(self, state_dict):
        # Sets the initial state of the aircraft according to the input

        # Set values
        self.y[:3] = import_value("velocity", state_dict, self._units, None)
        self.y[3:6] = import_value("angular_rates", state_dict, self._units, [0.0, 0.0, 0.0])
        self.y[6:9] = import_value("position", state_dict, self._units, None)
        orientation = import_value("orientation", state_dict, self._units, [1.0, 0.0, 0.0, 0.0])
        if len(orientation) == 3: # Euler angles
            self.y[9:] = Euler2Quat(orientation)
        else:
            self.y[9:] = orientation

        # Set controls
        for name in self._control_names:
            self._controls[name] = state_dict.get("control_state", {}).get(name, 0.0)


    def get_FM(self, t):
        """Returns the aerodynamic forces and moments."""

        # Get control state
        self._controls = self._controller.get_control(self.y, self._controls)

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
        B = m.atan2(v,u)
        const = 0.5*V_inv
        p_bar = self._bw*p*const
        q_bar = self._cw*q*const
        r_bar = self._bw*r*const

        # Get redimensionalizer
        redim = 0.5*rho*V*V*self._Sw

        # Determine coefficients without knowing final values for CL and CS
        CL = self._CL_ref+self._CL_a*a+self._CL_q*q_bar
        CS = self._CY_b*B+self._CY_p*p_bar+self._CY_r*r_bar
        CD = self._CD0+self._CD_q*q_bar
        Cl = self._Cl_ref+self._Cl_b*B+self._Cl_p*p_bar
        Cm = self._Cm_ref+self._Cm_q*q_bar
        Cn = self._Cn_ref+self._Cn_r*r_bar

        # Determine influence of controls
        for key, value in self._controls.items():
            control_deriv = self._control_derivs[key]
            CL += m.radians(value-self._control_ref[key])*control_deriv["CL"]
            CD += m.radians(value-self._control_ref[key])*control_deriv["CD"]
            CS += m.radians(value-self._control_ref[key])*control_deriv["CY"]
            Cl += m.radians(value-self._control_ref[key])*control_deriv["Cl"]
            Cm += m.radians(value-self._control_ref[key])*control_deriv["Cm"]
            Cn += m.radians(value-self._control_ref[key])*control_deriv["Cn"]

        # Factor in terms involving CL and CS
        CD += self._CD1*CL+self._CD2*CL*CL+self._CD3*CS*CS
        Cl += (self._Cl_r/self._CL_ref)*CL*r_bar
        Cm += (self._Cm_a/self._CL_a)*(CL*u*V_inv-self._CL_ref+CD*w*V_inv)
        Cn += (self._Cn_b/self._CY_b)*(CS*u*V_inv-CD*v*V_inv)+(self._Cn_p/self._CL_ref)*CL*p_bar

        # Apply aerodynamic angles and dimensionalize
        FM[0] = redim*(CL*m.sin(a)-CS*m.sin(B)-CD*u*V_inv)
        FM[1] = redim*(CS*cos(B)-CD*v*V_inv)
        FM[2] = redim*(-CL*cos(a)-CD*w*V_inv)
        FM[3] = redim*Cl*self._bw
        FM[4] = redim*Cm*self._cw
        FM[5] = redim*Cn*self._bw

        # Get effect of engines
        for engine in self._engines:
            FM += engine.get_thrust_FM(self._controls, rho, V)

        return FM


class MachUpXAirplane(BaseAircraft):
    """An airplane defined by MachUpX.
    
    Parameters
    ----------
    name : str
        Name of the aircraft.

    input_dict : dict
        Dictionary describing the airplane.
    """

    def __init__(self, name, input_dict, density, units, param_dict):
        super().__init__(name, input_dict, density, units, param_dict)