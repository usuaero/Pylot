"""Creates a simulator and runs it."""

import sys
import json
import pygame
from pylot.simulator import *

def simulate(filename):

    # Initialize sim
    with open(filename, 'r') as input_handle:
        input_dict = json.load(input_handle)
    sim = Simulator(input_dict, verbose=True)

    # Run
    sim.run_sim()


if __name__=="__main__":

    # Get filename
    input_filename = sys.argv[-1]

    # Run sim
    simulate(input_filename)