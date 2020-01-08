"""Creates a simulator and runs it."""

import sys
import json
from .simulator import *

def simulate(filename):

    # Initialize sim
    with open(filename, 'r') as input_handle:
        input_dict = json.load(input_handle)
    sim = Simulator(input_dict)


if __name__=="__main__":

    # Get filename
    input_filename = sys.argv[-1]

    # Run sim
    simulate(input_filename)