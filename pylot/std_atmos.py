# Functions describing a standard atmosphere

import numpy as np
import copy

def atm_print():
    """Prints both English and Metric standard atmosphere tables.
    """

    # Initialize file
    metric_filename = "stdatmos_si.txt"
    with open(metric_filename, 'w') as output_handle:

        # Create header
        output_handle.write("Geometric Geopotential                                         Speed of\n")
        output_handle.write("Altitude    Altitude   Temperature  Pressure      Density       Sound  \n")
        output_handle.write("  (m)         (m)          (K)      (N/m**2)     (kg/m**3)      (m/s)  \n")
        output_handle.write("-----------------------------------------------------------------------\n")

        # Loop through altitudes
        for i in range(51):

            # Calculate properties
            h = i*2000.0
            z, t, p, d = statsi(h)
            a = np.sqrt(1.4*287.0528*t)

            # Write to file
            write_string = "{0:<10}{1:<13.5f}{2:<13.5f}{3:<14.5e}{4:<13.5e}{5:<8.4f}\n".format(h, z, t, p, d, a)
            output_handle.write(write_string)

    # Initialize file
    english_filename = "stdatmos_ee.txt"
    with open(english_filename, 'w') as output_handle:

        # Create header
        output_handle.write("Geometric Geopotential                                          Speed of\n")
        output_handle.write("Altitude    Altitude   Temperature   Pressure      Density       Sound  \n")
        output_handle.write("  (ft)        (ft)         (R)      (lbf/ft^2)   (slugs/ft^3)    (ft/s) \n")
        output_handle.write("------------------------------------------------------------------------\n")

        # Loop through altitudes
        for i in range(51):

            # Calculate properties
            h = i*5000.0
            z, t, p, d = statee(h)
            a = np.sqrt(1.4*287.0528*t/1.8)/0.3048

            # Write to file
            write_string = "{0:<10}{1:<13.5f}{2:<13.5f}{3:<14.5e}{4:<13.5e}{5:<8.4f}\n".format(h, z, t, p, d, a)
            output_handle.write(write_string)



def statsi(h):
    """Calculates standard atmosphere data in SI units.

    Parameters
    ----------
    h : float
        geometric altitude in meters

    Returns
    -------
    z : float
        Geopotential altitude in meters.

    t : float
        Temperature in K.

    p : float
        Pressure in Pa.

    d : float
        Density in kg/m^3.
    """

    # Define constants
    zsa = np.array([0.0, 11000.0, 20000.0, 32000.0, 47000.0, 52000.0, 61000.0, 79000.0, 9.9e20])
    Tsa = np.array([288.15, 216.65, 216.65, 228.65, 270.65, 270.65,252.65, 180.65, 180.65])
    g = 9.80665
    R = 287.0528
    Re = 6346766.0
    Psa = 101325.0

    # Calculate geopotential altitude
    z = Re*h/(Re+h)

    # Loop through atmosphere layers
    for i in range(8):
        
        # Calculate layer temperature gradient
        Lt = -(Tsa[i+1]-Tsa[i])/(zsa[i+1]-zsa[i])

        # If no temperature gradient
        if Lt == 0.0:

            # Are we in this layer of the atmosphere?
            if z <= zsa[i+1]:
                t = Tsa[i] # Temp isn't changing
                p = Psa*np.exp(-g*(z-zsa[i])/R/Tsa[i])
                d = p/R/t
                break

            # We need to go higher
            else:
                Psa *= np.exp(-g*(zsa[i+1]-zsa[i])/R/Tsa[i])

        # Temperature gradient
        else:
            ex = g/R/Lt
            if z <= zsa[i+1]:
                t = Tsa[i]-Lt*(z-zsa[i])
                p = Psa*(t/Tsa[i])**ex
                d = p/R/t
                break
            else:
                Psa *= (Tsa[i+1]/Tsa[i])**ex

    # We have left the atmosphere...
    else:
        t = Tsa[-1]
        p = 0.0
        d = 0.0

    return z, t, p, d


def statee(h):
    """Calculates standard atmosphere data in English units.

    Parameters
    ----------
    h : float
        Geometric altitude in feet

    Returns
    -------
    z : float
        Geopotential altitude in feet.

    t : float
        Temperature in R.

    p : float
        Pressure in lbf/ft^2.

    d : float
        Density in slugs/ft^3.
    """
    # Convert height to SI
    hsi = h*0.3048

    # Get data
    zsi, tsi, psi, dsi = statsi(hsi)

    # Convert back to English
    z = zsi/0.3048
    t = tsi*1.8
    p = psi*0.02088543
    d = dsi*0.001940320

    return z, t, p, d