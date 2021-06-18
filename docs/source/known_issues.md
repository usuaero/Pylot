# Known Issues

## Integrator Divergence
At times, the user may encounter a blank, blue screen with the message "Integrator divergence. See documentation.". This screen appears when the numerical integrator (either RK4 or ABM4) has gone unstable and the state of the aircraft has diverged. This is a numerical artifact and is not indicative of the actual physics.

At this time, we believe this to be caused by the integrator being unable to capture the behavior of the roll mode of the aircraft due to a large timestep being used relative to the time constant of the roll mode. We have noticed this issue appears consistently with small aircraft (which have very short roll modes) being simulated using the MachUpX aerodynamic model (this leads to long simulation timesteps). However, it may occur in other cases.

A few things may be done to reduce the chances of this error occurring:

* Reduce the grid resolution of the MachUpX model so as to speed up the lifting-line calculations.
* Switch to the ABM4 integrator if using the RK4 integrator (we assume you understand the implications of using the ABM4 integrator).
* Increase the Ixx inertial parameter. Yes, this is actually changing the physics. It is up to you to decide if this is appropriate for your application.

We are investigating ways to fix this more permanently and automatically.