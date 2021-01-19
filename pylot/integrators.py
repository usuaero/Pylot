"""Defines various high-order numerical integrators for Pylot to use."""

import copy

import numpy as np


class RK4Integrator:
    """Performs Runge-Kutta integration for the given aircraft.

    Parameters
    ----------
    aircraft : BaseAircraft
        Aircraft to integrate the state of.
    """
    
    
    def __init__(self, aircraft):

        # Store aircraft
        self._aircraft = aircraft


    def step(self, t, dt):
        """Steps the Runge-Kutta integration forward.

        Parameters
        ----------
        t : float
            Initial time.

        dt : float
            Time step.

        """
        y0 = copy.deepcopy(self._aircraft.y)

        # Determine k0
        k0 = self._aircraft.dy_dt(t)

        # Determine k1
        self._aircraft.y = y0+0.5*dt*k0 
        k1 = self._aircraft.dy_dt(t+0.5*dt)

        # Determine k2
        self._aircraft.y = y0+0.5*dt*k1
        k2 = self._aircraft.dy_dt(t+0.5*dt)

        # Determine k3
        self._aircraft.y = y0+dt*k2
        k3 = self._aircraft.dy_dt(t+dt)

        # Calculate y
        self._aircraft.y = y0+0.16666666666666666666667*(k0+2*k1+2*k2+k3)*dt