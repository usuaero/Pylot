# Class containing the simulator

import math as m
import numpy as np
import multiprocessing as mp
from .helpers import *
from .airplanes import *
import json
import copy
import time
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from .graphics import *

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
        self._units = self._input_dict.get("units", "English")

        # Get simulation parameters
        self._real_time = self._input_dict["simulation"].get("real_time", True)
        if not self._real_time:
            self._dt = self._input_dict["simulation"].get("dt", 0.01)
            self._t0 = self._input_dict["simulation"].get("start_time", 0.0)
            self._tf = self._input_dict["simulation"].get("final_time", np.inf)

        # Initialize inter-process communication
        manager = mp.Manager()
        self._state_manager = manager.list()
        self._graphics_conn, self._physics_conn = mp.Pipe()

        # Kick off physics process
        self._physics_process = mp.Process(target=self._run_physics, args=())

        # Initialize graphics
        self._render_graphics = self._input_dict["simulation"].get("enable_graphics", False)
        if self._render_graphics:
            self._initialize_graphics()


    def _run_physics(self):
        # Handles physics on a separate process

        # Load aircraft
        self._load_aircraft()

        # TODO: Atmospheric properties

        # Get an initial guess for how long each sim step is going to take
        t0 = time.time()
        if self._real_time:
            self._RK4(self._aircraft, self._t0, 0.0)
            self._aircraft.normalize()
            self._aircraft.output_state(self._t0)
            t1 = time.time()
            self._dt = t1-t0
            t0 = t1

        t = copy.copy(self._t0)

        # Simulation loop
        while t <= self._tf:

            # Integrate
            self._RK4(self._aircraft, t, self._dt)

            # Normalize
            self._aircraft.normalize()

            # Output
            self._aircraft.output_state(t)

            # Step in time
            if self._real_time:
                t1 = time.time()
                self._dt = t1-t0
                t0 = t1
            t += self._dt

            # Pass information to graphics
            if self._render_graphics:
                self._state_manager = list(self._aircraft.y)

            # Check for exit condition


    def _load_aircraft(self):
        # Loads the aircraft from the input file

        # Read in aircraft input
        aircraft_name = self._input_dict["aircraft"]["name"]
        aircraft_file = self._input_dict["aircraft"]["file"]
        with open(aircraft_file, 'r') as aircraft_file_handle:
            aircraft_dict = json.load(aircraft_file_handle)

        # Create managers for passing information between the processes
        manager = mp.Manager()
        self._current_state = manager.list()
        self._current_controls = manager.dict()

        # Get density model and output file
        density = import_value("density", self._input_dict.get("atmosphere", {}), self._units, [0.0023769, "slug/ft^3"])
        state_output = self._input_dict["aircraft"].get("state_output", None)

        # Linear aircraft
        if aircraft_dict["aero_model"]["type"] == "linearized_coefficients":
            self._aircraft = LinearizedAirplane(aircraft_name, aircraft_dict, density, state_output, self._units)
        
        # MachUpX aircraft
        else:
            self._aircraft = MachUpXAirplane(aircraft_name, aircraft_dict, density, state_output, self._units)

        # TODO: Trim/set initial state


    def _initialize_graphics(self):
        # Initializes the graphics

        # Initialize pygame module
        pygame.init()

        # Setup window size
        width, height = 1800,900
        pygame.display.set_icon(pygame.image.load('gsim/res/gameicon.jpg'))
        _ = pygame.display.set_mode((width,height), HWSURFACE|OPENGL|DOUBLEBUF)
        pygame.display.set_caption("GSim")
        glViewport(0,0,width,height)
        glEnable(GL_DEPTH_TEST)
        
        # Boolean variables for camera view, lose screen, and pause
        self._FPV = True
        self._LOSE = False
        self._PAUSE = False
        self._DATA = True

        # SIMULATION FRAMERATE
        self._target_framerate = self._input_dict["simulation"].get("target_framerate", 30)

        # Initialize graphics objects
        # Loading screen is rendered and displayed while the rest of the objects are read and prepared
        glClearColor(0.,0.,0.,1.0)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        loading = Text(150)
        loading.draw(-0.2,-0.05,"Loading...",(0,255,0,1))
        pygame.display.flip()

        # Initialize game over screen
        self._gameover = Text(150)

        # Initialize graphics for aircraft
        self._aircraft_graphics = self._aircraft.get_graphics_obj()
        self._aircraft_graphics.set_orientation(self._aircraft.y[9:])
        self._aircraft_graphics.set_position(self._aircraft.y[6:9])

        # Initialize HUD
        self._HUD = HeadsUp(width, height)

        # Initialize flight data overlay
        self._data = FlightData()
        self._stall_warning = Text(100)

        # Initialize ground
        self._ground_quad = []
        self._quad_size = 20000
        self._ground_positions = [[0., 0., 0.],
                                  [0., self._quad_size, 0.],
                                  [self._quad_size, 0., 0.],
                                  [self._quad_size, self._quad_size, 0.]]
        ground_orientations = [[1., 0., 0., 0.],
                               [0., 0., 0., 1.],
                               [0., 0., 1., 0.],
                               [0., 1., 0., 0.]] # I'm not storing these because they don't change
        for i in range(4):
            self._ground_quad.append(Mesh("gsim/res/field.obj","gsim/shaders/field.vs","gsim/shaders/field.fs","gsim/res/field_texture.jpg",width,height))
            self._ground_quad[i].set_position(self._ground_positions[i])
            self._ground_quad[i].set_orientation(ground_orientations[i])

        # Initialize camera object
        self._cam = Camera()

        # Clock object for tracking frames and timestep
        self._clock = pygame.time.Clock()

        # Ticks clock before starting game loop
        self._clock.tick_busy_loop()

        # Initialize storage of velocities for determining g's
        self._prev_vels = [0.0, 0.0, 0.0]
        


    def run_sim(self):
        """Runs the simulation according to the defined inputs.
        """

        # Kick off the physics
        self._physics_process.start()

        # Run graphics loop
        if self._render_graphics:
            while True:
                # Update graphics
                pass

        else: # Just wait for the physics to finish
            self._physics_process.join()


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
