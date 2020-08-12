from math import sqrt, cos, sin, atan2, asin, pi
import numpy as np
import os
import copy

def QuatMult(q0, q1):
    q00 = q0[0]
    q0x = q0[1]
    q0y = q0[2]
    q0z = q0[3]
    q10 = q1[0]
    q1x = q1[1]
    q1y = q1[2]
    q1z = q1[3]

    return [q00*q10-q0x*q1x-q0y*q1y-q0z*q1z, q00*q1x+q0x*q10+q0y*q1z-q0z*q1y, q00*q1y-q0x*q1z+q0y*q10+q0z*q1x, q00*q1z+q0x*q1y-q0y*q1x+q0z*q10]

def Euler2Quat(E):
    phi = 0.5*E[0]
    theta = 0.5*E[1]
    psi = 0.5*E[2]
    C_phi = cos(phi)
    S_phi = sin(phi)
    C_theta = cos(theta)
    S_theta = sin(theta)
    C_psi = cos(psi)
    S_psi = sin(psi)
    CphCth = C_phi*C_theta
    SthSps = S_theta*S_psi
    SthCps = S_theta*C_psi
    SphCth = S_phi*C_theta
    return [CphCth*C_psi+S_phi*SthSps, SphCth*C_psi-C_phi*SthSps, C_phi*SthCps+SphCth*S_psi, CphCth*S_psi-S_phi*SthCps]

def NormalizeQuaternion(q):
    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]
    m = sqrt(q0*q0+q1*q1+q2*q2+q3*q3)
    return [e/m for e in q]

def NormalizeQuaternionNearOne(q):
    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]
    m = 1.5-0.5*(q0*q0+q1*q1+q2*q2+q3*q3)
    return [e*m for e in q]

def Quat2Euler(q):
    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]
    x = q0*q2-q1*q3
    if x != 0.5 and x != -0.5:
        q02 = q0*q0
        qx2 = q1*q1
        qy2 = q2*q2
        qz2 = q3*q3
        return [atan2(2*(q0*q1+q2*q3), q02+qz2-qx2-qy2), asin(2*x), atan2(2*(q0*q3+q1*q2), q02+qx2-qy2-qz2)]
    else:
        return [2.*asin(q1*1.4142135623730951), pi*x, 0.]

def Body2Fixed(v, q):
    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]
    q00 = q0*q0
    qxx = q1*q1
    qyy = q2*q2
    qzz = q3*q3
    q0x = 2*q0*q1
    q0y = 2*q0*q2
    q0z = 2*q0*q3
    qxy = 2*q1*q2
    qxz = 2*q1*q3
    qyz = 2*q2*q3
    v0 = v[0]
    v1 = v[1]
    v2 = v[2]
    return [(qxx+q00-qyy-qzz)*v0 + (qxy-q0z)*v1 + (qxz+q0y)*v2, (qxy+q0z)*v0 + (qyy+q00-qxx-qzz)*v1 + (qyz-q0x)*v2, (qxz-q0y)*v0 + (qyz+q0x)*v1 + (qzz+q00-qxx-qyy)*v2]

def Fixed2Body(v, q):
    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]
    q00 = q0*q0
    qxx = q1*q1
    qyy = q2*q2
    qzz = q3*q3
    q0x = 2*q0*q1
    q0y = 2*q0*q2
    q0z = 2*q0*q3
    qxy = 2*q1*q2
    qxz = 2*q1*q3
    qyz = 2*q2*q3
    v0 = v[0]
    v1 = v[1]
    v2 = v[2]
    return [(qxx+q00-qyy-qzz)*v0 + (qxy+q0z)*v1 + (qxz-q0y)*v2, (qxy-q0z)*v0 + (qyy+q00-qxx-qzz)*v1 + (qyz+q0x)*v2, (qxz+q0y)*v0 + (qyz-q0x)*v1 + (qzz+q00-qxx-qyy)*v2]

def check_filepath(input_filename, correct_ext):
    # Check correct file extension and that file exists
    if correct_ext not in input_filename:
        raise IOError("File {0} has the wrong extension. Expected a {1} file.".format(input_filename, correct_ext))
    if not os.path.exists(input_filename):
        raise IOError("Cannot find file {0}.".format(input_filename))

def convert_units(in_value, units, system):
    # Converts the given value from the specified units to the default for the system chosen
    if units == "-":
        return in_value

    to_english_default = {
        "ft" : 1.0,
        "in" : 0.083333333,
        "m" : 3.28084,
        "cm" : 0.0328084,
        "ft/s" : 1.0,
        "m/s" : 3.28084,
        "mph" : 1.466666666,
        "kph" : 0.9113444,
        "kn" : 1.687811,
        "ft^2" : 1.0,
        "m^2" : 10.76391,
        "slug/ft^3" : 1.0,
        "kg/m^3" : 0.0019403203,
        "lbf" : 1.0,
        "N" : 0.22480894244319,
        "deg" : 1.0,
        "rad" : 57.29578,
        "deg/s" : 0.01745329,
        "rad/s" : 1.0,
        "slug ft^2" : 1.0,
        "kg m^2" : 0.7375621419,
        "slug ft^2/s" : 1.0,
        "kg m^2/s" : 0.7375621419,
        "lbf/ft" : 1.0,
        "N/m" : 0.0685217659
    }

    to_si_default = {
        "ft" : 0.3048,
        "in" : 0.0254,
        "m" : 1.0,
        "cm" : 0.01,
        "ft/s" : 0.3048,
        "m/s" : 1.0,
        "mph" : 0.44704,
        "kph" : 0.277777777,
        "kn" : 0.514444444,
        "ft^2" : 0.09290304,
        "m^2" : 1.0,
        "slug/ft^3" : 515.378819,
        "kg/m^3" : 1.0,
        "lbf" : 4.4482216,
        "N" : 1.0,
        "deg" : 1.0,
        "rad" : 57.29578,
        "deg/s" : 0.01745329,
        "rad/s" : 1.0,
        "slug ft^2" : 1.355817961,
        "kg m^2" : 1.0,
        "slug ft^2/s" : 1.355817961,
        "kg m^2/s" : 1.0,
        "lbf/ft" : 14.593902928003786,
        "N/m" : 1.0
    }
    try:
        if system == "English":
            return in_value*to_english_default[units.strip(' \t\r\n')]
        else:
            return in_value*to_si_default[units.strip(' \t\r\n')]
    except KeyError:
        raise IOError("Improper units specified; {0} is not an allowable unit definition.".format(units))


def import_value(key, dict_of_vals, system, default_value):
    # Imports value from a dictionary. Handles importing arrays from files and 
    # unit conversions. If default_value is -1, then this value must be 
    # specified in the input (i.e. an error is thrown if -1 is returned).

    val = dict_of_vals.get(key, default_value)
    
    if val is None:
        raise IOError("Key '{0}' is not optional. Please specify.".format(key))
    is_array = False

    if isinstance(val, float): # Float without units
        return_value = val

    elif isinstance(val, int): # Integer values should be converted to floats
        return_value = float(val)

    elif isinstance(val, str) and ".csv" in val: # Filepath containing array
        check_filepath(val, ".csv")
        with open(val, 'r') as array_file:
            val = np.genfromtxt(array_file, delimiter=',', dtype=None, encoding='utf-8')
            is_array = True
            
    elif isinstance(val, str): # Simply a string value
        return_value = val

    elif isinstance(val, list):
        if any(isinstance(row, list) for row in val): # Array
            is_array = True

        elif val[0] == "elliptic": # User wants an elliptic chord distribution
            if len(val) == 3: # Unit specified
                root_chord = convert_units(val[1], val[2], system)
            else:
                root_chord = val[1]
            return_value = ("elliptic", root_chord)
        
        elif isinstance(val[-1], str): # Float or vector with units
            converted_val = vectorized_convert_units(val[:-1], val[-1], system)

            try:
                return_value = converted_val.item() # Float
            except ValueError:
                return_value = converted_val # Vector or quaternion

        elif len(val) == 3 or len(val) == 4: # Vector or quaternion without units
            return_value = np.asarray(val)

        else:
            raise ValueError("Did not recognize value format {0}.".format(val))

    else:
        raise ValueError("Did not recognize value format {0}.".format(val))

    if is_array:
        if isinstance(val[-1][0], str): # Array with units
            val = np.asarray(val)
            units = val[-1,:]
            data = val[:-1,:].astype(float)
            return_value = vectorized_convert_units(data, units, system)

        else: # Array without units
            #TODO: Allow this to handle arrays specifying a distribution of airfoils
            return_value = np.asarray(val)

    return return_value


vectorized_convert_units = np.vectorize(convert_units)


def cross(v0, v1):
    """Calculates the cross product of v0 and v1."""
    v00, v01, v02 = v0
    v10, v11, v12 = v1
    return np.array([v01*v12-v11*v02, v10*v02-v00*v12, v00*v11-v10*v01])