# Developer Notes
The purpose of this document is to explain the inner workings of Pylot for those who are developing it. Please update this as often as you can.

## Adding Different Aircraft Models to Pylot
Pylot has been developed using an object-oriented approach. As such, it is simple to add new aircraft models. The aircraft for Pylot are defined in the airplanes.py file. All aircraft are derived classes of ```BaseAircraft```. 