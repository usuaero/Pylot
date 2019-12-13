# Class containing the simulator

import math as m
import numpy as mp
from helpers import *
from airplanes import *
import json
import copy
import time

class Simulator:
    """A class for flight simulation using RK4 integration.

    Parameters
    ----------
    input_dict : dict
        Dictionary describing the simulation and world parameters.
    """

    def __init__(self, input_dict):

        # Store input
        self._input_dict = input_dict
        self._units = self._input_dict["run"].get("units", "English")

        # Get simulation parameters
        self._real_time = self._input_dict["run"].get("real_time", True)
        if not self._real_time:
            self._dt = self._input_dict["run"].get("dt", 0.01)
            self._t0 = self._input_dict["run"].get("start_time", 0.01)
            self._tf = self._input_dict["run"].get("final_time", 0.01)

        # Initialize aircraft
        self._aircraft = []
        for key, value in self._input_dict["scene"]["aircraft"]:

            # Read in file
            with open(value["file"], 'r') as aircraft_file_handle:
                aircraft_dict = json.load(aircraft_file_handle)

            # Linear aircraft
            if "coefficients" in list(value.keys()):
                aircraft = LinearAirplane(key, aircraft_dict)
            
            # MachUpX aircraft
            else:
                aircraft = MachUpXAirplane(key, aircraft_dict)

            # Store
            self._aircraft.append(aircraft)

            # TODO: Atmospheric properties

            # Trim
            aircraft.trim(**value["initial"])
            

    def run_sim(self):
        """Runs the simulator according to the defined inputs.
        """
        pass


    def _RK4(self, aircraft, t, dt):
        """Performs Runge-Kutta integration for the given aircraft.

        Parameters
        ----------
        aircraft : BaseAircraft
            Aircraft to integrate the state of.

        t : float
            Initial time.

        dt : float
            Time step.

        """
        y0 = copy.deepcopy(aircraft.y)

        # Determine k0
        k0 = aircraft.dy_dt(t)

        # Determine k1
        aircraft.y = y0+0.5*dt*k0 
        k1 = aircraft.dy_dt(t+0.5*dt)

        # Determine k2
        aircraft.y = y0+0.5*dt*k1
        k2 = aircraft.dy_dt(t+0.5*dt)

        # Determine k3
        aircraft.y = y0+dt*k2
        k3 = aircraft.dy_dt(t+dt)

        # Calculate y
        aircraft.y = y0+0.166666666666666666667*(k0+2*k1+2*k2+k3)*dt
