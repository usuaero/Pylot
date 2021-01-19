"""For investigating how the timestep of the RK4 integrator can affect the stability of a simulated system."""

import numpy as np
import matplotlib.pyplot as plt


def f(y, t):
    """State function.

    Parameters
    ----------
    y : ndarray
        Vector of states.

    Returns
    -------
    dy_dt : ndarray
        Vector of derivatives.
    """

    # y'' + 4y' + 4y = 0
    # y1 = y
    # y2 = y'
    # y1' = y2
    # y2' = -4y2 - 4y1

    return np.array([y[1], -4.0*y[1]-4.0*y[0]])


def RK4(f, y_i, t_i, dt):
    """Performs Runge-Kutta integration.

    Parameters
    ----------
    f : callable
        Derivative function.

    y_i : ndarray
        Current state.

    t_i : float
        Current time.

    dt : float
        Time step.

    Returns
    -------
    y_j : ndarray
        State at next time step.
    """

    # Determine k constants
    k0 = f(y_i, t_i)
    k1 = f(y_i+0.5*dt*k0, t_i+0.5*dt)
    k2 = f(y_i+0.5*dt*k1, t_i+0.5*dt)
    k3 = f(y_i+dt*k2, t_i+dt)

    # Calculate y at next timestep
    return y_i+0.16666666666666666666667*(k0+2.0*k1+2.0*k2+k3)*dt


if __name__=="__main__":

    # Initialize simulation
    y0 = [4.0, 4.0]
    t0 = 0.0
    t_f = 50.0

    # Set up plot
    fig, ax = plt.subplots(nrows=1, ncols=2)

    # Loop through timesteps
    dts = np.logspace(-3.0, 0.5, num=15)
    for dt in dts:

        n_steps = int((t_f-t0)/dt)
        y = np.copy(y0)
        states = [y0]
        times = np.linspace(t0, t0+(n_steps+1)*dt, n_steps+1)

        # Sim loop
        for t in times[1:]:

            y = RK4(f, y, t, dt)
            states.append(y)

        states = np.array(states)

        # Plot
        ax[0].plot(times, states[:,0], label=str(dt))
        ax[0].legend()
        ax[0].set_xlabel('t [s]')
        ax[1].plot(times, states[:,1], label=str(dt))
        ax[1].legend()
        ax[1].set_xlabel('t [s]')

    plt.show()