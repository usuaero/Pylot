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
import os

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
        self._t0 = self._input_dict["simulation"].get("start_time", 0.0)
        self._tf = self._input_dict["simulation"].get("final_time", np.inf)
        if not self._real_time:
            self._dt = self._input_dict["simulation"].get("dt", 0.01)

        # Initialize inter-process communication
        manager = mp.Manager()
        self._state_manager = manager.list()
        self._state_manager[:] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self._graphics_ready = manager.Value('i', 0)
        self._aircraft_graphics_info = manager.dict()

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

        # Pass airplane graphics information to parent process
        if self._render_graphics:
            aircraft_graphics_info = self._aircraft.get_graphics_info()
            for key, value in aircraft_graphics_info.items():
                self._aircraft_graphics_info[key] = value

            # Give initial state to graphics
            self._aircraft_graphics_info["position"] = self._aircraft.y[6:9]
            self._aircraft_graphics_info["orientation"] = self._aircraft.y[9:]

            # Wait fo graphics to load
            while not self._graphics_ready.value:
                continue

        # Get an initial guess for how long each sim step is going to take
        t0 = time.time()
        if self._real_time:
            self._RK4(self._aircraft, self._t0, 0.0)
            self._aircraft.normalize()
            self._aircraft.output_state(self._t0)
            t1 = time.time()
            self._dt = t1-t0
            t0 = t1
        else:
            self._aircraft.output_state(self._t0)

        t = copy.copy(self._t0)

        # Simulation loop
        while t <= self._tf:

            # Integrate
            self._RK4(self._aircraft, t, self._dt)

            # Normalize
            self._aircraft.normalize()

            # Step in time
            if self._real_time:
                t1 = time.time()
                self._dt = t1-t0
                t0 = t1
            t += self._dt

            # Output
            self._aircraft.output_state(t)

            # Pass information to graphics
            if self._render_graphics:
                self._state_manager[:13] = self._aircraft.y[:]
                self._state_manager[13] = self._dt

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

        # Get density model, controller, and output file
        density = import_value("density", self._input_dict.get("atmosphere", {}), self._units, [0.0023769, "slug/ft^3"])

        # Linear aircraft
        if aircraft_dict["aero_model"]["type"] == "linearized_coefficients":
            self._aircraft = LinearizedAirplane(aircraft_name, aircraft_dict, density, self._units, self._input_dict["aircraft"])
        
        # MachUpX aircraft
        else:
            self._aircraft = MachUpXAirplane(aircraft_name, aircraft_dict, density, self._units, self._input_dict["aircraft"])


    def _initialize_graphics(self):
        # Initializes the graphics

        # Get path to graphics objects
        self._pylot_path = os.path.dirname(__file__)
        self._graphics_path = os.path.join(self._pylot_path,os.path.pardir,"graphics")
        self._cessna_path = os.path.join(self._graphics_path, "Cessna")
        self._res_path = os.path.join(self._graphics_path, "res")
        self._shaders_path = os.path.join(self._graphics_path, "shaders")

        # Initialize pygame module
        pygame.init()

        # Setup window size
        self._width, self._height = 1800,900
        pygame.display.set_icon(pygame.image.load(os.path.join(self._res_path, 'gameicon.jpg')))
        screen = pygame.display.set_mode((self._width,self._height), HWSURFACE|OPENGL|DOUBLEBUF)
        pygame.display.set_caption("Pylot Flight Simulator, (C) USU AeroLab")
        glViewport(0,0,self._width,self._height)
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

        # Initialize HUD
        self._HUD = HeadsUp(self._width, self._height, self._res_path, self._shaders_path)

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
            self._ground_quad.append(Mesh(
                os.path.join(self._res_path, "field.obj"),
                os.path.join(self._shaders_path, "field.vs"),
                os.path.join(self._shaders_path, "field.fs"),
                os.path.join(self._res_path, "field_texture.jpg"),
                self._width,
                self._height))
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

        # Get graphics going
        if self._render_graphics:

            # Wait for physics to initialize then import aircraft object
            while True:
                try:
                    obj_path = self._aircraft_graphics_info["obj_file"]
                    v_shader_path = self._aircraft_graphics_info["v_shader_file"]
                    f_shader_path = self._aircraft_graphics_info["f_shader_file"]
                    texture_path = self._aircraft_graphics_info["texture_file"]
                    self._aircraft_graphics = Mesh(obj_path, v_shader_path, f_shader_path, texture_path, self._width, self._height)
                    self._aircraft_graphics.set_position(self._aircraft_graphics_info["position"])
                    self._aircraft_graphics.set_orientation(self._aircraft_graphics_info["orientation"])
                    break

                except KeyError: # If it's not there, just keep waiting
                    continue

            # Let the physics know we're good to go
            self._graphics_ready.value = 1

            # Run graphics loop
            self._start_time = time.time()
            while True:

                # Update graphics
                self._update_graphics()

        else: # Just wait for the physics to finish
            self._physics_process.join()


    def _update_graphics(self):
        # Does a step in graphics

        #set default background color for sky
        glClearColor(0.65,1.0,1.0,1.0)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        # Get state from state manager
        y = np.array(self._state_manager[:13])
        dt_physics = self._state_manager[13]

        #timestep for simulation is based on framerate
        dt_graphics = self._clock.tick(self._target_framerate)/1000.

        # Update graphics for each aircraft
        self._aircraft_graphics.set_orientation(swap_quat(y[9:]))
        self._aircraft_graphics.set_position(y[6:9])

        # Check for crashing into the ground
        if y[8] > 0.0:
            self._LOSE = True

        # Parse state of pilot aircraft
        aircraft_condition = {
            "Position" : y[6:9],
            "Orientation" : y[9:],
            "Velocity" : y[:3]
        }
        u = y[0]
        v = y[1]
        w = y[2]
        V = sqrt(u*u+v*v+w*w)
        E = np.degrees(Quat2Euler(y[9:]))
        V_f = Body2Fixed(y[:3], y[9:])
        a = m.atan2(w,u)
        B = m.atan2(v,u)

        # Store data
        flight_data = {
            "Graphics Time Step" : dt_graphics,
            "Physics Time Step" : dt_physics,
            "Airspeed" : V,
            "AoA" : m.degrees(a),
            "Sideslip" : m.degrees(B),
            "Altitude" : -y[8],
            "Latitude" : y[6]/131479714.0*360.0,
            "Longitude" : y[7]/131259396.0*360.0,
            "Time" : time.time()-self._start_time,
            "Bank" : E[0],
            "Elevation" : E[1],
            "Heading" : E[2],
            "Gnd Speed" : m.sqrt(V_f[0]*V_f[0]+V_f[1]*V_f[1])*0.68181818181818181818,
            "Gnd Track" : E[2],
            "Climb" : -V_f[2]*60,
            "Throttle" : 0.0,
            "Elevator" : 0.0,
            "Ailerons" : 0.0,
            "Rudder" : 0.0,
            "Flaps" : 0.0,
            "Axial G-Force" : 0.0,
            "Side G-Force" : 0.0,
            "Normal G-Force" : 0.0,
            "Roll Rate" : m.degrees(y[3]),
            "Pitch Rate" : m.degrees(y[4]),
            "Yaw Rate" : m.degrees(y[5])
        }

        # Store veloties
        self._prev_vels = V_f

        # If you get a game over, display lose screen
        if self._LOSE == True:
            glClearColor(0,0,0,1.0)
            glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

            self._gameover.draw(-0.2,-0.05,"Game Over",(0,255,0,1))
            self._PAUSE = True
	
        # Otherwise, render graphics
        else:
            if not self._FPV: # Third person view
                view = self._cam.third_view(self._aircraft_graphics, offset=[-70, 0., -30])
	
            # Cockpit view
            else:
                view = self._cam.cockpit_view(self._aircraft_graphics)
                self._HUD.render(aircraft_condition,view)

            # Determine pilot displacement in quad widths
            x_pos = y[6]
            y_pos = y[7]
            if x_pos > 0.0:
                N_quads_x = (x_pos+self._quad_size//2)//self._quad_size
            else:
                N_quads_x = (x_pos-self._quad_size//2)//self._quad_size+1
            if y_pos > 0.0:
                N_quads_y = (y_pos+self._quad_size//2)//self._quad_size
            else:
                N_quads_y = (y_pos-self._quad_size//2)//self._quad_size+1

            # Set positions based on aircraft position within the current quad
            curr_quad_x = N_quads_x*self._quad_size
            curr_quad_y = N_quads_y*self._quad_size

            if x_pos > curr_quad_x and y_pos > curr_quad_y:
                self._ground_positions = [[N_quads_x*self._quad_size, N_quads_y*self._quad_size, 0.],
                                          [N_quads_x*self._quad_size, (N_quads_y+1)*self._quad_size, 0.],
                                          [(N_quads_x+1)*self._quad_size, N_quads_y*self._quad_size, 0.],
                                          [(N_quads_x+1)*self._quad_size, (N_quads_y+1)*self._quad_size, 0.]]
            elif x_pos < curr_quad_x and y_pos > curr_quad_y:
                self._ground_positions = [[N_quads_x*self._quad_size, N_quads_y*self._quad_size, 0.],
                                          [N_quads_x*self._quad_size, (N_quads_y+1)*self._quad_size, 0.],
                                          [(N_quads_x-1)*self._quad_size, N_quads_y*self._quad_size, 0.],
                                          [(N_quads_x-1)*self._quad_size, (N_quads_y+1)*self._quad_size, 0.]]
            elif x_pos < curr_quad_x and y_pos < curr_quad_y:
                self._ground_positions = [[N_quads_x*self._quad_size, N_quads_y*self._quad_size, 0.],
                                          [N_quads_x*self._quad_size, (N_quads_y-1)*self._quad_size, 0.],
                                          [(N_quads_x-1)*self._quad_size, N_quads_y*self._quad_size, 0.],
                                          [(N_quads_x-1)*self._quad_size, (N_quads_y-1)*self._quad_size, 0.]]
            else:
                self._ground_positions = [[N_quads_x*self._quad_size, N_quads_y*self._quad_size, 0.],
                                          [N_quads_x*self._quad_size, (N_quads_y-1)*self._quad_size, 0.],
                                          [(N_quads_x+1)*self._quad_size, N_quads_y*self._quad_size, 0.],
                                          [(N_quads_x+1)*self._quad_size, (N_quads_y-1)*self._quad_size, 0.]]

            # Swap tiling to keep the order the same
            if (N_quads_x%2 != 0):
                self._ground_positions[0], self._ground_positions[2] = self._ground_positions[2], self._ground_positions[0]
                self._ground_positions[1], self._ground_positions[3] = self._ground_positions[3], self._ground_positions[1]
            if (N_quads_y%2 != 0):
                self._ground_positions[0], self._ground_positions[1] = self._ground_positions[1], self._ground_positions[0]
                self._ground_positions[2], self._ground_positions[3] = self._ground_positions[3], self._ground_positions[2]

            # Update ground graphics
            for i, quad in enumerate(self._ground_quad):
                quad.set_position(self._ground_positions[i])
                quad.set_view(view)
                quad.render()

            # Update airplanes
            if not self._FPV: # Don't render the pilot aircraft in HUD mode
                self._aircraft_graphics.set_view(view)
                self._aircraft_graphics.render()

            # Check for MachUp falling apart
            if np.isnan(u):
                error_msg = Text(100)
                error_msg.draw(-1.0, 0.5, "MachUpX error: invalid flight regime.", color=(255,0,0,1))

            # Display flight data
            elif self._DATA:
                self._data.render(flight_data)

        # Update screen display
        pygame.display.flip()


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
        aircraft.y = y0+0.16666666666666666666667*(k0+2*k1+2*k2+k3)*dt