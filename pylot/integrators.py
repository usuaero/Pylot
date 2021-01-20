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


    def step(self, t, dt, **kwargs):
        """Steps the Runge-Kutta integration forward.

        Parameters
        ----------
        t : float
            Initial time.

        dt : float
            Time step.

        """

        # Store the current state
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

        # Return the first derivative to use in the A-B-M method
        return k0


class ABM4Integrator:
    """Performs multi-step Adams-Bashforth-Moulton integration for the given aircraft.

    Parameters
    ----------
    aircraft : BaseAircraft
        Aircraft to integrate the state of.
    """
    
    
    def __init__(self, aircraft):

        # Store aircraft
        self._aircraft = aircraft

        # Store derivatives
        self._f = np.zeros((3,13))

        # Keep track of number of stored derivatives so we know when we can switch to the implicit scheme
        self._n_stored = 0

        # Create an RK4 integrator to use for starting off
        self._RK4 = RK4Integrator(self._aircraft)


    def step(self, t, dt, **kwargs):
        """Steps the A-B-M integration forward. Uses a single application of the corrector.

        Parameters
        ----------
        t : float
            Initial time.

        dt : float
            Time step.

        store : bool
            Whether this step should be stored to use in the implicit integration.
        """

        # Determine which integrator to use based on how many derivatives we have stored
        if self._n_stored < 3:

            # Step RK4 integrator
            f = self._RK4.step(t, dt)

            # Store derivatives
            if kwargs.get('store'):
                self._n_stored += 1
                self._f[2-self._n_stored,:] = f

        else:

            # Store the current state
            y0 = copy.deepcopy(self._aircraft.y)

            # Get the current derivative
            f0 = self._aircraft.dy_dt(t)

            # Predictor
            self._aircraft.y = y0+dt*(2.2916666666666665*f0-2.4583333333333335*self._f[0]+1.5416666666666667*self._f[1]-0.375*self._f[2])

            # Get derivative at predicted state
            f1 = self._aircraft.dy_dt(t+dt)

            # Corrector
            self._aircraft.y = y0+dt*(0.375*f1+0.7916666666666666*f0-0.20833333333333334*self._f[0]+0.041666666666666664*self._f[1])

            # Store derivatives for next step
            self._f = np.roll(self._f, 1, axis=0)
            self._f[0] = f0
        