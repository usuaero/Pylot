import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import OpenGL.GL.shaders
from pyrr import matrix44, vector3,Vector4
from math import atan2, asin
import pygame.image
import pygame.font
import pygame.draw
import os

from PIL import Image


def _load_shader(shader_file):
    shader_source = ""
    with open(shader_file) as f:
        shader_source = f.read()
    f.close()
    return str.encode(shader_source)

def compile_shader(vs, fs):
    vert_shader = _load_shader(vs)
    frag_shader = _load_shader(fs)

    shader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(vert_shader, GL_VERTEX_SHADER),
                                              OpenGL.GL.shaders.compileShader(frag_shader, GL_FRAGMENT_SHADER))
    return shader

def load_texture(path):
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    # Set the texture wrapping parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    # Set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    # load image
    image = Image.open(path)
    img_data = np.array(list(image.getdata()), np.uint8)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
    glEnable(GL_TEXTURE_2D)
    return texture

def create_from_inverse_of_quaternion(quat):
    return np.array([[1.0 - 2.0 * (quat[1]**2 + quat[2]**2), 2.0 * (quat[0] * quat[1] + quat[3] * quat[2]),2.0 * (quat[0] * quat[2] - quat[3] * quat[1]),0.],
                     [2.0 * (quat[0] * quat[1] - quat[3] * quat[2]),1.0 - 2.0 * ((quat[0]**2) + quat[2]**2),2.0 * (quat[1] * quat[2] + quat[3] * quat[0]),0.],
                     [2.0 * (quat[0] * quat[2] + quat[3] * quat[1]),2.0 * (quat[1] * quat[2] - quat[3] * quat[0]),1.0 - 2.0 * ((quat[0]**2) + quat[1]**2),0.],
                     [0.,0.,0.,1.]])

def create_from_translation(vec):
    return np.array([[1.,0.,0.,0.],
                     [0.,1.,0.,0.],
                     [0.,0.,1.,0.],
                     [vec[0],vec[1],vec[2],1.]])

def create_from_z_rotation(rotation):
    s = np.sin(rotation)
    c = np.cos(rotation)
    return np.array([[c,-s,0.,0.],
                    [s,c,0.,0.],
                    [0.,0.,1.,0.],
                    [0.,0.,0.,1.]])


#calculates quaternion product
def quatProd(A, B):
    a = A[0]*B[0]-A[1]*B[1]-A[2]*B[2]-A[3]*B[3]
    b = A[0]*B[1]+A[1]*B[0]+A[2]*B[3]-A[3]*B[2]
    c = A[0]*B[2]-A[1]*B[3]+A[2]*B[0]+A[3]*B[1]
    d = A[0]*B[3]+A[1]*B[2]-A[2]*B[1]+A[3]*B[0]
    return [a,b,c,d]
	 

#returns quaterion conjugate
def quatConj(e):
    return [e[0],-e[1],-e[2],-e[3]]


#calculates Euler angles from Quaternion according to 11.7.9 (returns angles in degrees)
def eulFromQuat(e):
    phi = atan2(2*(e[0]*e[1]+e[2]*e[3]),(e[0]**2+e[3]**2-e[1]**2-e[2]**2))*180./np.pi
    theta = asin(2*(e[0]*e[2]-e[1]*e[3]))*180./np.pi
    gamma = atan2(2*(e[0]*e[3]+e[1]*e[3]),(e[0]**2+e[1]**2-e[2]**2-e[3]**2))*180./np.pi
    if gamma<0.:
        gamma+=360.
    E = np.array([gamma,theta,phi])
    return E

#transforms vector in body fixed coordinates to earth fixed coordinates
def Body2Fixed(vec,e):
    x,y,z = vec
    e0,ex,ey,ez = e

    To =  x*ex+y*ey+z*ez
    Tx =  x*e0-y*ez+z*ey
    Ty =  x*ez+y*e0-z*ex
    Tz = -x*ey+y*ex+z*e0

    a = e0*Tx+ex*To+ey*Tz-ez*Ty
    b = e0*Ty-ex*Tz+ey*To+ez*Tx
    c = e0*Tz+ex*Ty-ey*Tx+ez*To


    return [a,b,c]

#changes quaternion from (e0,ex,ey,ez) to (x,y,z,w)
def swap_quat(e):
    return [e[1],e[2],e[3],e[0]]

def vector_normalise(vec):
    if vec[0] == 0. and vec[1] == 0. and vec[2] == 0.:
        return np.array([0.,0.,0.])
    else:
        mag =np.sqrt(vec[0]**2+vec[1]**2+vec[2]**2)
        return np.array([vec[0]/mag,vec[1]/mag,vec[2]/mag])

class Mesh:
    def __init__(self, filename, vertexshadername, fragmentshadername, texturename, width, height):
        self.vert_coords = []
        self.text_coords = []
        self.norm_coords = []
        self.vert_index = []
        self.text_index = []
        self.norm_index = []
        self.model = []
        self.projection_matrix = matrix44.create_perspective_projection_matrix(60.0, width/height,0.1,10000)
        self.position = [0.,0.,0.]
        self.orientation = [0.,0.,0.,1.]

        for line in open(filename, 'r'):
            if line.startswith('#'):continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                self.vert_coords.append(values[1:4])
            if values[0] == 'vt':
                self.text_coords.append(values[1:3])
            if values[0] == 'vn':
                self.norm_coords.append(values[1:4])
            if values[0] == 'f':
                vi = []
                ti = []
                ni = []
                for v in values[1:4]:
                    w = v.split('/')
                    vi.append(int(w[0])-1)
                    try:
                        ti.append(int(w[1])-1)
                    except ValueError:
                        pass
                    ni.append(int(w[2])-1)
                self.vert_index.append(vi)
                self.text_index.append(ti)
                self.norm_index.append(ni)
        self.vert_index = [y for x in self.vert_index for y in x]
        self.text_index = [y for x in self.text_index for y in x]
        self.norm_index = [y for x in self.norm_index for y in x]


        for i in self.vert_index:
            self.model.extend(self.vert_coords[i])
        for i in self.text_index:
            self.model.extend(self.text_coords[i])
        for i in self.norm_index:
            self.model.extend(self.norm_coords[i])
        self.model = np.array(self.model, dtype = 'float32')
        self.texture_offset = len(self.vert_index)*12
        self.normal_offset = (self.texture_offset+len(self.text_index)*8)

        self.shader = compile_shader(vertexshadername, fragmentshadername)
        self.model_loc = glGetUniformLocation(self.shader, "model")
        self.view_loc = glGetUniformLocation(self.shader, "view")
        self.proj_loc = glGetUniformLocation(self.shader, "proj")
        self.orientation_loc = glGetUniformLocation(self.shader, "orientation")

        glUseProgram(self.shader)
        glUniformMatrix4fv(self.proj_loc,1,GL_FALSE,self.projection_matrix)
        glUseProgram(0)

        self.texture = load_texture(texturename)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER,self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.model.itemsize*len(self.model), self.model, GL_STATIC_DRAW)
        #position
        glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE, self.model.itemsize*3, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        #texture
        glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,self.model.itemsize*2, ctypes.c_void_p(self.texture_offset))
        glEnableVertexAttribArray(1)
        #normals
        glVertexAttribPointer(2,3,GL_FLOAT,GL_FALSE,self.model.itemsize*3, ctypes.c_void_p(self.normal_offset))
        glEnableVertexAttribArray(2)
        glBindVertexArray(0)

        self.set_position(self.position)
        self.set_orientation(self.orientation)

    def set_position(self,position):
        self.position = position
        self.model_matrix = create_from_translation(self.position)
        glUseProgram(self.shader)
        glUniformMatrix4fv(self.model_loc,1,GL_FALSE,self.model_matrix)
        glUseProgram(0)

    def set_orientation(self,orientation):
        self.orientation = orientation
        self.orientation_matrix = create_from_inverse_of_quaternion(self.orientation)
        glUseProgram(self.shader)
        glUniformMatrix4fv(self.orientation_loc,1, GL_FALSE, self.orientation_matrix)
        glUseProgram(0)

    def set_orientation_z(self,rotation):
        self.orientation_matrix = create_from_z_rotation(-np.radians(rotation))
        glUseProgram(self.shader)
        glUniformMatrix4fv(self.orientation_loc,1, GL_FALSE, self.orientation_matrix)
        glUseProgram(0)

    def change_projection_matrix(self, fov, aspect_ratio, near, far):
        self.projection_matrix = matrix44.create_perspective_projection_matrix(fov,aspect_ratio,near,far)
        glUseProgram(self.shader)
        glUniformMatrix4fv(self.proj_loc,1,GL_FALSE,self.projection_matrix)
        glUseProgram(0)
    def set_view(self,view):
        glUseProgram(self.shader)
        glUniformMatrix4fv(self.view_loc,1,GL_FALSE,view)
        glUseProgram(0)

    def render(self):
        glBindVertexArray(self.vao)
        glUseProgram(self.shader)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glDrawArrays(GL_TRIANGLES,0,len(self.vert_index))
        glBindTexture(GL_TEXTURE_2D, 0)
        glUseProgram(0)
        glBindVertexArray(0)
		
class Text:
    def __init__(self, size):
        self.font = pygame.font.Font(None, size) 

    def draw(self, x, y, text, color = None):                                                
        position = (x, y, 0)  
        if color:
            textSurface = self.font.render(text, True, color)   
        else:
            textSurface = self.font.render(text, True, (0,0,0,1))                                     
        textData = pygame.image.tostring(textSurface, "RGBA", True) 

        glEnable(GL_BLEND) 
        glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA)           
        glRasterPos3d(*position)                                                
        glDrawPixels(textSurface.get_width(), textSurface.get_height(),GL_RGBA, GL_UNSIGNED_BYTE, textData) 
        glDisable(GL_BLEND)


class FlightData:
    def __init__(self):
        self.text = Text(36)

    def render(self,flight_data):
        #left side data
        self.text.draw(-0.9,0.90,"Airspeed")
        self.text.draw(-0.9,0.84,str(round(flight_data["Airspeed"],1))+" fps")
        self.text.draw(-0.9,0.75,"AoA")
        self.text.draw(-0.9,0.69,str(round(flight_data["AoA"],5))+" deg")
        self.text.draw(-0.9,0.60,"Sideslip")
        self.text.draw(-0.9,0.54,str(round(flight_data["Sideslip"],5))+" deg")
        self.text.draw(-0.9,0.45,"Altitude")
        self.text.draw(-0.9,0.39,str(round(flight_data["Altitude"],2))+" ft")
        self.text.draw(-0.9,0.30,"Latitude")
        self.text.draw(-0.9,0.24,str(round(flight_data["Latitude"],6))+" deg")
        self.text.draw(-0.9,0.15,"Longitude")
        self.text.draw(-0.9,0.09,str(round(flight_data["Longitude"],6))+" deg")
        self.text.draw(-0.9,0.0,"Bank")
        self.text.draw(-0.9,-0.06,str(round(flight_data["Bank"],2))+" deg")
        self.text.draw(-0.9,-0.15,"Elevation")
        self.text.draw(-0.9,-0.21,str(round(flight_data["Elevation"],2))+" deg")
        self.text.draw(-0.9,-0.30,"Heading")
        self.text.draw(-0.9,-0.36,str(round(flight_data["Heading"],2))+" deg")
        self.text.draw(-0.9,-0.45,"Gnd Speed")
        self.text.draw(-0.9,-0.51,str(round(flight_data["Gnd Speed"],1))+" mph")
        self.text.draw(-0.9,-0.60,"Gnd Track")
        self.text.draw(-0.9,-0.66,str(round(flight_data["Gnd Track"],2))+" deg")
        self.text.draw(-0.9,-0.75,"Climb")
        self.text.draw(-0.9,-0.81,str(round(flight_data["Climb"],0))+" fpm")


        #right side data
        self.text.draw(0.7,0.90,"Axial G-Force")
        self.text.draw(0.7,0.84,str(round(flight_data["Axial G-Force"],4))+" g's")
        self.text.draw(0.7,0.75,"Side G-Force")
        self.text.draw(0.7,0.69,str(round(flight_data["Side G-Force"],4))+" g's")
        self.text.draw(0.7,0.60,"Normal G-Force")
        self.text.draw(0.7,0.54,str(round(flight_data["Normal G-Force"],4))+" g's")
        self.text.draw(0.7,0.45,"Roll Rate")
        self.text.draw(0.7,0.39,str(round(flight_data["Roll Rate"],2))+" deg/s")
        self.text.draw(0.7,0.30,"Pitch Rate")
        self.text.draw(0.7,0.24,str(round(flight_data["Pitch Rate"],2))+" deg/s")
        self.text.draw(0.7,0.15,"Yaw Rate")
        self.text.draw(0.7,0.09,str(round(flight_data["Yaw Rate"],2))+" deg/s")
        self.text.draw(0.7,0.00,"Throttle")
        self.text.draw(0.7,-0.06,str(round(flight_data["Throttle"],0))+" %")
        self.text.draw(0.7,-0.15,"Elevator")
        self.text.draw(0.7,-0.21,str(round(flight_data["Elevator"],1))+" deg")
        self.text.draw(0.7,-0.30,"Ailerons")
        self.text.draw(0.7,-0.36,str(round(flight_data["Ailerons"],1))+" deg")
        self.text.draw(0.7,-0.45,"Rudder")
        self.text.draw(0.7,-0.51,str(round(flight_data["Rudder"],1))+" deg")
        self.text.draw(0.7,-0.60,"Flaps")
        self.text.draw(0.7,-0.66,str(round(flight_data["Flaps"],1))+" deg")
        self.text.draw(0.7,-0.75,"Time")
        self.text.draw(0.7,-0.81,str(round(flight_data["Time"],1))+" sec")

        #bottom data
        self.text.draw(0.1,-0.75,"Graphics Time Step: " +str(flight_data["Graphics Time Step"])+" sec")
        self.text.draw(-0.6,-0.75,"Physics Time Step: " +str(round(flight_data["Physics Time Step"],6))+" sec")

class HeadsUp:
    def __init__(self,width, height, res_path, shaders_path, screen):
        #initialize HUD objects
        self.view = np.identity(4)
        self.screen = screen
        self.width = width
        self.height = height

        #initialize pitch ladder
        self.ladder = Mesh(os.path.join(res_path, "ladder.obj"),
            os.path.join(shaders_path, "HUD.vs"),
            os.path.join(shaders_path, "HUD.fs"),
            os.path.join(res_path, "HUD_texture.jpg"),
            width,height)

        #initialize flight path indicator
        self.flightPath = Mesh(os.path.join(res_path, "flightPath.obj"),
            os.path.join(shaders_path, "HUD.vs"),
            os.path.join(shaders_path, "HUD.fs"),
            os.path.join(res_path, "HUD_texture.jpg"),
            width,height)

        #initialize crosshair
        self.crosshair = Mesh(os.path.join(res_path, "crosshair.obj"),
            os.path.join(shaders_path, "HUD.vs"),
            os.path.join(shaders_path, "HUD.fs"),
            os.path.join(res_path, "HUD_texture.jpg"),
            width,height)
        self.crosshair.set_position([0.,0.,-0.5])
        self.crosshair.set_view(self.view)

        #initialize bank angle indicator
        self.bank = Mesh(os.path.join(res_path, "bank.obj"),
            os.path.join(shaders_path, "HUD.vs"),
            os.path.join(shaders_path, "HUD.fs"),
            os.path.join(res_path, "HUD_texture.jpg"),
            width,height)
        self.bank.set_position([0.,-0.205,-0.075])
        self.bank.set_view(self.view)

        #initialize compass
        self.compass = Mesh(os.path.join(res_path, "compass.obj"),
            os.path.join(shaders_path, "HUD.vs"),
            os.path.join(shaders_path, "HUD.fs"),
            os.path.join(res_path, "HUD_texture.jpg"),
            width,height)
        self.compass.set_view(self.view)

        #initialize speedometer
        self.speed = Mesh(os.path.join(res_path, "speedometer.obj"),
            os.path.join(shaders_path, "HUD.vs"),
            os.path.join(shaders_path, "HUD.fs"),
            os.path.join(res_path, "HUD_texture.jpg"),
            width,height)
        self.speed.set_view(self.view)

        #initialize altimeter
        self.alt = Mesh(os.path.join(res_path, "altimeter.obj"),
            os.path.join(shaders_path, "HUD.vs"),
            os.path.join(shaders_path, "HUD.fs"),
            os.path.join(res_path, "HUD_texture.jpg"),
            width,height)
        self.alt.set_view(self.view)

        #initialize viewports for HUD objects
        self.viewport = Frame(0.75,0.75,width,height,[0.,0.,-1.], shaders_path)
        self.speed_viewport = Frame(1.25,0.3,width,height,[0.,0.,-1.], shaders_path)
        self.bank_viewport = Frame(0.5,0.125,width, height,[0.,0.26,-1.], shaders_path)


        #change HUD matrices to fit in center_viewport
        self.compass.change_projection_matrix(60,self.viewport.external_aspect_ratio,0.1,10)
        self.bank.change_projection_matrix(60,self.bank_viewport.external_aspect_ratio,0.05,10)
        self.crosshair.change_projection_matrix(60,self.viewport.external_aspect_ratio,0.1,10)
        self.speed.change_projection_matrix(60,self.speed_viewport.external_aspect_ratio,0.1,10)
        self.alt.change_projection_matrix(60,self.speed_viewport.external_aspect_ratio,0.1,10)


    def render(self, aircraft_condition, world_view):
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        velocity_b = aircraft_condition["Velocity"]
        orientation_b = aircraft_condition["Orientation"]
        position_f = aircraft_condition["Position"]

        Vo = np.sqrt(velocity_b[0]**2+velocity_b[1]**2+velocity_b[2]**2)

        #determine position of flight path indicator
        velocity_f = Body2Fixed(velocity_b, orientation_b)
        flight_path = vector_normalise(velocity_f)
        cam_position = position_f
        self.flightPath.set_position(cam_position+0.6*flight_path)
        self.flightPath.set_orientation(swap_quat(orientation_b))

        #calculate changes in pitch ladder and compass positions
        euler = eulFromQuat(orientation_b)			
        self.ladder.set_position(cam_position)
        self.ladder.set_orientation_z(euler[0])
        self.compass.set_position([-euler[0]*0.2,-0.22,-0.75])
        self.speed.set_position([-0.35,Vo*-0.002,-0.25])
        self.alt.set_position([0.35,position_f[2]*0.002,-0.25])
        self.bank.set_orientation_z(euler[2])

        #render ladder
        self.ladder.set_view(world_view)
        self.flightPath.set_view(world_view)
        self.ladder.render()
        self.flightPath.render()

        #altimeter and airspeed indicator viewframe
        self.speed_viewport.start_draw_to_frame(0.)
        self.speed.render()
        self.alt.render()
        self.speed_viewport.end_draw_to_frame()

        #crosshair and compass viewframe
        self.viewport.start_draw_to_frame(0.)
        self.crosshair.render()
        self.compass.render()
        self.viewport.end_draw_to_frame()

        #bank indicator viewport
        self.bank_viewport.start_draw_to_frame(0.)
        self.bank.render()
        self.bank_viewport.end_draw_to_frame()

        #render viewports
        self.speed_viewport.draw_frame()
        self.viewport.draw_frame()
        self.bank_viewport.draw_frame()


class Frame:
    def __init__(self,x,y,width,height,position,shaders_path):

        self.plane = [-0.5*x, -0.5*y, 0.0, 0.0, 0.0,
                       0.5*x, -0.5*y, 0.0, 1.0, 0.0,
                       0.5*x,  0.5*y, 0.0, 1.0, 1.0,
                      -0.5*x,  0.5*y, 0.0, 0.0, 1.0]
	
        self.plane = np.array(self.plane, dtype=np.float32)

        self.plane_indices = [0, 1, 2, 2, 3, 0]
        self.plane_indices = np.array(self.plane_indices, dtype=np.uint32)
        self.external_aspect_ratio = x/y

        aspect_ratio = float(width/height)
        self.shader = compile_shader(os.path.join(shaders_path, "blank.vs"), os.path.join(shaders_path, "blank.fs"))
        self.model_loc = glGetUniformLocation(self.shader, "model")
        self.proj_loc = glGetUniformLocation(self.shader, "proj")
        self.projection = matrix44.create_perspective_projection_matrix(45.0, aspect_ratio, 0.1, 100.0)

        # plane VAO
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, self.plane.itemsize * len(self.plane), self.plane, GL_STATIC_DRAW)
        self.EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.plane_indices.itemsize * len(self.plane_indices), self.plane_indices, GL_STATIC_DRAW)
        # position
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.plane.itemsize * 5, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        # textures
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, self.plane.itemsize * 5, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)
        glBindVertexArray(0)

        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        # texture wrapping params
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        # texture filtering params
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glBindTexture(GL_TEXTURE_2D, 0)

        self.depth_buff = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self.depth_buff)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height)

        self.FBO = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.FBO)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.texture, 0)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self.depth_buff)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        position_matrix = create_from_translation(position)
        glUseProgram(self.shader)
        glUniformMatrix4fv(self.model_loc, 1, GL_FALSE, position_matrix)
        glUniformMatrix4fv(self.proj_loc,1,GL_FALSE,self.projection)
        glUseProgram(0)


    def start_draw_to_frame(self,opacity):
        glBindFramebuffer(GL_FRAMEBUFFER,self.FBO)
        glClearColor(0.,0.,0.,opacity)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)


    def end_draw_to_frame(self):
        glBindFramebuffer(GL_FRAMEBUFFER,0)


    def draw_frame(self):
        glBindVertexArray(self.vao)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glUseProgram(self.shader)
        glDrawElements(GL_TRIANGLES, len(self.plane_indices), GL_UNSIGNED_INT, None)
        glUseProgram(0)		
        glBindVertexArray(0)


class Camera:
    def __init__(self):
        self.pos_storage = []
        self.up_storage = []
        self.target_storage =[]
        self.IDENTITY = np.identity(4)
        self.IDENTITY2 = np.identity(4)


    def third_view(self, graphics_aircraft, offset=[-10., 0., -2.]):
        """creates view matrix such that camera is positioned behind and slightly above graphics_aircraft. camera location and orientation is tied to graphics_aircraft

        Parameters
        ----------
        graphics_aircraft: graphics_aircraft object used in graphics 

        Returns
        -------
        view matrix

        Raises
        ------

        Notes
        -----
        This function does several conversions between (e0,ex,ey,ez) and (x,y,z,w) forms of quaternions. It should not be altered.
        """

        #third person camera view of plane
        quat_orientation = [graphics_aircraft.orientation[3], graphics_aircraft.orientation[0], graphics_aircraft.orientation[1], graphics_aircraft.orientation[2]]
        graphics_aircraft_to_camera = Body2Fixed(offset, quat_orientation)
		
        cam_up = [0.,0.,-1.]
        rotated_cam_up = Body2Fixed(cam_up,quat_orientation)

        self.pos_storage.append(graphics_aircraft.position+graphics_aircraft_to_camera)
        self.up_storage.append(np.array(rotated_cam_up))
        self.target_storage.append(graphics_aircraft.position)

        # Latency. Stores position, target, and up in lists and pulls out and uses old values to create a delayed effect
        # An average is taken to smooth out the camera position
        delay = 5
	
        if len(self.pos_storage)<=delay:
            self.camera_pos = self.pos_storage[0]
            self.camera_up = self.up_storage[0]
            self.target = self.target_storage[0]

        else:
            self.camera_pos = 0.25*(self.pos_storage.pop(0)+self.pos_storage[0]+self.pos_storage[1]+self.pos_storage[2])
            self.camera_up = 0.25*(self.up_storage.pop(0)+self.up_storage[0]+self.up_storage[1]+self.up_storage[2])
            self.target = 0.25*(self.target_storage.pop(0)+self.target_storage[0]+self.target_storage[1]+self.target_storage[2])

        return self.look_at(self.camera_pos, self.target, self.camera_up)	

    def cockpit_view(self, graphics_aircraft):
        """creates view matrix such that camera is positioned at the graphics_aircraft location, as if in the cockpit

        Parameters
        ----------
        graphics_aircraft: graphics_aircraft object used in graphics 

        Returns
        -------
        view matrix

        Raises
        ------

        Notes
        -----

        """

        graphics_aircraft_forward = Body2Fixed([1.0,0.,0.],[graphics_aircraft.orientation[3],graphics_aircraft.orientation[0],graphics_aircraft.orientation[1],graphics_aircraft.orientation[2]])
        graphics_aircraft_up = Body2Fixed([0.,0.,-1.],[graphics_aircraft.orientation[3],graphics_aircraft.orientation[0],graphics_aircraft.orientation[1],graphics_aircraft.orientation[2]])
        return self.look_at(graphics_aircraft.position, graphics_aircraft.position+graphics_aircraft_forward, graphics_aircraft_up)


    def look_at(self, position, target, world_up):
        """creates view matrix based on three arguments: camera position, camera target, and camera up

        Parameters
        ----------
        position: vector position of the camera
        target: vector point the camera is looking at
        up: vector pointing in the up direction

        Returns
        -------
        view matrix

        Raises
        ------

        Notes
        -----

        """
        #returns view matrix based on position, target, and up vector
        # 1.Position = known
        # 2.Calculate cameraDirection
        zaxis = vector_normalise(position - target)
        # 3.Get positive right axis vector
        xaxis = vector_normalise(np.cross(vector_normalise(world_up), zaxis))
        # 4.Calculate the camera up vector
        yaxis = vector3.cross(zaxis, xaxis)

        # create translation and rotation matrix
        translation = self.IDENTITY
        translation[3][0] = -position[0]
        translation[3][1] = -position[1]
        translation[3][2] = -position[2]


        rotation = self.IDENTITY2
        rotation[0][0] = xaxis[0]
        rotation[1][0] = xaxis[1]
        rotation[2][0] = xaxis[2]
        rotation[0][1] = yaxis[0]
        rotation[1][1] = yaxis[1]
        rotation[2][1] = yaxis[2]
        rotation[0][2] = zaxis[0]
        rotation[1][2] = zaxis[1]
        rotation[2][2] = zaxis[2]


        return np.matmul(translation, rotation)
