# Class containing the simulator

import numpy as np
import multiprocessing as mp
import math as m
import json
import copy
import time
import pygame.display
import pygame.image
import os
from .physics import run_physics, load_aircraft, RK4
from .helpers import Quat2Euler
from pygame.locals import HWSURFACE, OPENGL, DOUBLEBUF
from OpenGL.GL import glClear, glClearColor
from .graphics import *

class Simulator:
    """A class for flight simulation using RK4 integration.

    Parameters
    ----------
    input_dict : dict
        Dictionary describing the simulation and world parameters.
    """

    def __init__(self, input_dict):

        # Print welcome
        print("\n--------------------------------------------------")
        print(  "              Welcome to Pylot!                   ")
        print(  "                 USU AeroLab                      ")
        print(  "--------------------------------------------------")

        # Store input
        self._input_dict = input_dict
        self._units = self._input_dict.get("units", "English")

        # Get simulation parameters
        self._render_graphics = self._input_dict["simulation"].get("enable_graphics", False)

        # Initialize inter-process communication
        self._manager = mp.Manager()
        self._state_manager = self._manager.list()
        self._state_manager[:] = [0.0]*16
        self._quit = self._manager.Value('i', 0)
        self._pause = self._manager.Value('i', 0)
        self._graphics_ready = self._manager.Value('i', 0)
        self._view = self._manager.Value('i', 1)
        self._flight_data = self._manager.Value('i', 1)
        self._aircraft_graphics_info = self._manager.dict()
        self._control_settings = self._manager.dict()

        # Kick off physics process
        self._physics_process = mp.Process(target=run_physics, args=(self._input_dict,
                                                                     self._units,
                                                                     self._aircraft_graphics_info,
                                                                     self._graphics_ready,
                                                                     self._quit,
                                                                     self._view,
                                                                     self._pause,
                                                                     self._flight_data,
                                                                     self._state_manager,
                                                                     self._control_settings))

        # Initialize graphics
        if self._render_graphics:

            # Initialize pygame modules
            pygame.display.init()
            pygame.font.init()

            self._initialize_graphics()


    def _initialize_graphics(self):
        # Initializes the graphics

        # Get path to graphics objects
        self._set_graphics_paths()

        # Initialize game window
        self._initialize_game_window()
        
        # Get target framerate
        self._target_framerate = self._input_dict["simulation"].get("target_framerate", 30)

        # Render loading screen
        glClearColor(0.,0.,0.,1.0)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT|GL_ACCUM_BUFFER_BIT|GL_STENCIL_BUFFER_BIT)
        loading = Text(150)
        loading.draw(-(450/self._width),-0.05,"Loading...",(0,255,0,1))
        pygame.display.flip()

        # Initialize game over screen
        self._gameover = Text(150)

        # Initialize HUD
        self._HUD = HeadsUp(self._width, self._height, self._objects_path, self._shaders_path, self._textures_path, self._screen)

        # Initialize flight data overlay
        self._data = FlightData(self._units)
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
            self._ground_quad.append(self._create_mesh("field.obj", "field.vs", "field.fs", "field_texture.jpg", self._ground_positions[i], ground_orientations[i]))

        # Initialize scenery
        try:
            self._sky = self._create_mesh("sky.obj", "sky.vs", "sky.fs", "clouds.jpg", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
        except:
            self._sky = self._create_mesh("sky.obj", "sky.vs", "sky.fs", "clouds_low_res.jpg", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
        self._airstrip = self._create_mesh("airstrip.obj", "field.vs", "field.fs", "landing.jpg", [0.0, 0.0, -1.0], [1.0, 0.0, 0.0, 0.0])
        self._tent = self._create_mesh("tent.obj", "aircraft.vs", "aircraft.fs", "Tent.jpeg", [0.0, 25.0, 0.0], [1.0, 0.0, 0.0, 0.0])
        self._bench = self._create_mesh("wood_bench.obj", "aircraft.vs", "aircraft.fs", "bench.jpg", [10.0, 25.0, 0.0], [1.0, 0.0, 0.0, 0.0])
        self._trees = []
        for i in range(10):
            theta = np.random.rand(1)*2*np.pi
            rho = np.random.rand(1)*70+30
            pos = [rho*np.cos(theta), rho*np.sin(theta), 0.0]
            self._trees.append(self._create_mesh("spruce.obj", "field.vs", "field.fs", "tree_texture.jpg", pos, [1.0, 0.0, 0.0, 0.0]))

        # Initialize camera object
        self._cam = Camera()

        # Clock object for tracking frames and timestep
        self._clock = pygame.time.Clock()

        # Ticks clock before starting game loop
        self._clock.tick_busy_loop()


    def _create_mesh(self, obj, vs, fs, texture, position, orientation):
        # Creates a mesh graphics object

        mesh = Mesh(os.path.join(self._objects_path, obj),
                os.path.join(self._shaders_path, vs),
                os.path.join(self._shaders_path, fs),
                os.path.join(self._textures_path, texture),
                self._width,
                self._height)
        mesh.set_position(position)
        mesh.set_orientation(orientation)
        return mesh


    def _set_graphics_paths(self):
        # Gets the absolute paths to the graphics files

        self._pylot_path = os.path.dirname(__file__)
        self._graphics_path = os.path.join(self._pylot_path, os.path.pardir, "graphics")
        self._objects_path = os.path.join(self._graphics_path, "objects")
        self._shaders_path = os.path.join(self._graphics_path, "shaders")
        self._textures_path = os.path.join(self._graphics_path, "textures")


    def _initialize_game_window(self):
        # Sets up the pygame window

        # Get monitor size
        self._width, self._height = self._input_dict["simulation"].get("screen_resolution", [1800, 900])

        # Initialize window
        pygame.display.set_icon(pygame.image.load(os.path.join(self._textures_path, 'gameicon.jpg')))
        self._screen = pygame.display.set_mode((self._width,self._height), HWSURFACE|OPENGL|DOUBLEBUF|pygame.RESIZABLE)
        pygame.display.set_caption("Pylot Flight Simulator, (C) USU AeroLab")
        glViewport(0,0,self._width,self._height)
        glEnable(GL_DEPTH_TEST)


    def run_sim(self):
        """Runs the simulation according to the defined inputs.
        """

        print("Running simulation...")
        # Kick off the physics
        self._physics_process.start()

        # Get graphics going
        if self._render_graphics:

            # Wait for physics to initialize then import aircraft object
            while True:
                try:
                    # Get graphics files
                    obj_path = self._aircraft_graphics_info["obj_file"]
                    v_shader_path = self._aircraft_graphics_info["v_shader_file"]
                    f_shader_path = self._aircraft_graphics_info["f_shader_file"]
                    texture_path = self._aircraft_graphics_info["texture_file"]

                    # Initialize graphics object
                    self._aircraft_graphics = Mesh(obj_path, v_shader_path, f_shader_path, texture_path, self._width, self._height)
                    self._aircraft_graphics.set_position(self._aircraft_graphics_info["position"])
                    self._aircraft_graphics.set_orientation(self._aircraft_graphics_info["orientation"])

                    # Delete object file and stl file generated by MachUpX
                    if obj_path == "airplane.obj":
                        os.remove(obj_path)
                        os.remove("airplane.stl")

                    # Get reference lengths for setting camera offset
                    self._bw = self._aircraft_graphics_info["l_ref_lat"]
                    self._cw = self._aircraft_graphics_info["l_ref_lon"]

                    break

                except KeyError: # If it's not there, just keep waiting
                    continue

            # Let the physics know we're good to go
            self._graphics_ready.value = 1

            # Run graphics loop
            while not self._quit.value:

                # Update graphics
                self._update_graphics()

        # Wait for the physics to finish
        self._physics_process.join()
        self._physics_process.close()
        self._manager.shutdown()

        # Print quit message
        print("\n--------------------------------------------------")
        print(  "           Pylot exited successfully.             ")
        print(  "                  Thank you!                      ")
        print(  "--------------------------------------------------")


    def _update_graphics(self):
        # Does a step in graphics

        # Set default background color for sky
        glClearColor(0.65,1.0,1.0,1.0)

        # Clear GL buffers
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT|GL_ACCUM_BUFFER_BIT|GL_STENCIL_BUFFER_BIT)

        # Check pygame event queue
        events = pygame.event.get()
        for event in events:

            # Quit by 'x'
            if event.type == pygame.QUIT:
                self._quit.value = 1

            ## Resize window
            # I can't get this working. Feel free to try yourself.
            #elif event.type == pygame.VIDEORESIZE:
            #    self._width = event.w
            #    self._height = event.h
            #    self._screen = pygame.display.set_mode((self._width, self._height), HWSURFACE|OPENGL|DOUBLEBUF|pygame.RESIZABLE)
            #    self._HUD.resize(self._width, self._height)
            #    self._sky.resize(self._width, self._height)
            #    self._aircraft_graphics.resize(self._width, self._height)
            #    self._airstrip.resize(self._width, self._height)
            #    self._bench.resize(self._width, self._height)
            #    self._tent.resize(self._width, self._height)
            #    for tree in self._trees:
            #        tree.resize(self._width, self._height)
            #    for quad in self._ground_quad:
            #        quad.resize(self._width, self._height)

        # Check for quitting
        if self._quit.value:
            return True

        # Get state from state manager
        y = np.array(copy.deepcopy(self._state_manager[:13]))

        # Check to see if the physics has finished the first loop
        if (y == 0.0).all():
            return False

        # Get timing information from physics
        dt_physics = self._state_manager[13]
        t_physics = self._state_manager[14]
        graphics_delay = time.time()-self._state_manager[15] # Included to compensate for the fact that these physics results may be old or brand new

        # Graphics timestep
        dt_graphics = self._clock.tick(self._target_framerate)/1000.

        # Update aircraft position and orientation
        self._aircraft_graphics.set_orientation(swap_quat(y[9:]))
        self._aircraft_graphics.set_position(y[6:9])

        # Check for crashing into the ground
        if y[8] > 0.0:

            # Display Game Over screen and quit physics
            glClearColor(0,0,0,1.0)
            glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT|GL_ACCUM_BUFFER_BIT|GL_STENCIL_BUFFER_BIT)
            self._gameover.draw(-(450/self._width),-0.05,"Game Over",(0,255,0,1))
            self._quit.value = 1
	
        # Otherwise, render graphics
        else:
            # Third person view
            if self._view.value == 0:
                view = self._cam.third_view(self._aircraft_graphics, t_physics, graphics_delay, y[0], offset=[-self._bw, 0.0, -self._cw])
                self._aircraft_graphics.set_view(view)
                self._aircraft_graphics.render()
	
            # Cockpit view
            elif self._view.value == 1:
                self._cam.pos_storage.clear()
                self._cam.up_storage.clear()
                self._cam.target_storage.clear()
                self._cam.time_storage.clear()
                view = self._cam.cockpit_view(self._aircraft_graphics)
                self._HUD.render(y[:3], self._aircraft_graphics, view)

            # Ground view
            elif self._view.value == 2:
                view = self._cam.ground_view(self._aircraft_graphics, t_physics, graphics_delay)
                self._aircraft_graphics.set_view(view)
                self._aircraft_graphics.render()


            # Determine aircraft displacement in quad widths
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

            # Display scenery
            #self._sky.set_position([y[6], y[7], 0.0])
            self._sky.set_view(view)
            self._sky.render()
            self._airstrip.set_view(view)
            self._airstrip.render()
            self._tent.set_view(view)
            self._tent.render()
            self._bench.set_view(view)
            self._bench.render()
            for tree in self._trees:
                tree.set_view(view)
                tree.render()

            # Check for the aerodynamic model falling apart
            if np.isnan(y[0]):
                error_msg = Text(100)
                error_msg.draw(-1.0, 0.5, "Pylot encountered a physics error...", color=(255,0,0,1))

            # Display flight data
            elif self._flight_data.value:
                flight_data = self._get_flight_data(y, dt_graphics, dt_physics, t_physics)
                self._data.render(flight_data, self._control_settings)

        # Update screen display
        pygame.display.flip()


    def _get_flight_data(self, y, dt_graphics, dt_physics, t_physics):
        # Parses state of aircraft
        u = y[0]
        v = y[1]
        w = y[2]
        V = m.sqrt(u*u+v*v+w*w)
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
            "Time" : t_physics,
            "Bank" : E[0],
            "Elevation" : E[1],
            "Heading" : E[2],
            "Gnd Speed" : m.sqrt(V_f[0]*V_f[0]+V_f[1]*V_f[1])*0.68181818181818181818,
            "Gnd Track" : E[2],
            "Climb" : -V_f[2]*60,
            "Axial G-Force" : 0.0,
            "Side G-Force" : 0.0,
            "Normal G-Force" : 0.0,
            "Roll Rate" : m.degrees(y[3]),
            "Pitch Rate" : m.degrees(y[4]),
            "Yaw Rate" : m.degrees(y[5])
        }
    
        return flight_data