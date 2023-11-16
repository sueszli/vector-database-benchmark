from __future__ import print_function
from string import Template
import logging
from vaex.ui.qt import *
import numpy as np
import vaex.ui.colormaps
import time
from OpenGL.GL import *
from OpenGL.GL.framebufferobjects import *
from OpenGL.GL.ARB.shadow import *
from OpenGL.GL import shaders
import OpenGL
try:
    from PyQt4 import QtGui, QtCore
    from PyQt4 import QtOpenGL
except ImportError:
    try:
        from PyQt5 import QtGui, QtCore
        from PyQt5 import QtOpenGL
    except ImportError:
        from PySide import QtGui, QtCore
        from PySide import QtOpenGL
logger = logging.getLogger('vaex.ui.volr')
ray_cast_fragment_source = Template("\n#version 120\nvarying vec4 vertex_color;\nuniform sampler1D texture_colormap;\nuniform sampler2D texture;\nuniform sampler3D cube;\nuniform sampler3D gradient;\nuniform vec2 size; // size of screen/fbo, to convert between pixels and uniform\n//uniform vec2 minmax2d;\nuniform vec2 minmax3d;\n//uniform vec2 minmax3d_total;\n//uniform float maxvalue2d;\n//uniform float maxvalue3d;\nuniform float alpha_mod; // mod3\nuniform float mod4;  // mafnifier\nuniform float mod5; // blend color and line integral\nuniform float mod6;\n\nuniform sampler1D transfer_function;\n\nuniform float brightness;\nuniform float ambient_coefficient;\nuniform float diffuse_coefficient;\nuniform float specular_coefficient;\nuniform float specular_exponent;\nuniform float background_opacity;\nuniform float foreground_opacity;\nuniform float depth_peel;\n\n\nvoid main() {\n    int steps = ${iterations};\n    vec3 ray_end = vec3(texture2D(texture, vec2(gl_FragCoord.x/size.x, gl_FragCoord.y/size.y)));\n    vec3 ray_start = vertex_color.xyz;\n    //ray_start.z = 1. - ray_start.z;\n    //ray_end.z = 1. - ray_end.z;\n    float length = 0.;\n    vec3 ray_dir = (ray_end - ray_start);\n    vec3 ray_delta = ray_dir / float(steps);\n    float ray_length = sqrt(ray_dir.x*ray_dir.x + ray_dir.y*ray_dir.y + ray_dir.z*ray_dir.z);\n    vec3 ray_pos = ray_start;\n    float value = 0.;\n    //mat3 direction_matrix = inverse(mat3(transpose(inverse(gl_ModelViewProjectionMatrix))));\n    mat3 mat_temp = mat3(gl_ModelViewProjectionMatrix[0].xyz, gl_ModelViewProjectionMatrix[1].xyz, gl_ModelViewProjectionMatrix[2].xyz);\n    mat3 direction_matrix = mat_temp;\n    vec3 light_pos = (vec3(-100.,100., -100) * direction_matrix).zyx;\n    //vec3 light_pos = (direction_matrix * vec3(-5.,5., -100));\n    //vec3 origin = (direction_matrix * vec3(0., 0., 0)).xyz;\n    vec3 origin = (vec4(0., 0., 0., 0.)).xyz;\n    //vec3 light_pos = (vec4(-1000., 0., -1000, 1.)).xyz;\n    //mat3 mod = inverse(mat3(gl_ModelViewProjectionMatrix));\n    vec4 color = vec4(0, 0, 0, 0);\n    vec3 light_dir = light_pos - origin;\n    mat3 rotation = mat3(gl_ModelViewMatrix);\n    light_dir = vec3(-1,-1,1) * rotation;\n    light_dir = normalize(light_dir);// / sqrt(light_dir.x*light_dir.x + light_dir.y*light_dir.y + light_dir.z*light_dir.z);\n    float alpha_total = 0.;\n    //float normalize = log(maxvalue);\n    float intensity_total;\n    float data_min = minmax3d.x;\n    float data_max = minmax3d.y;\n    float data_scale = 1./(data_max - data_min);\n    float delta = 1.0/256./2;\n    //vec3 light_dir = vec3(1,1,-1);\n    vec3 eye = vec3(0, 0, 1) * rotation;\n    float depth_factor = 0.;\n\n    for(int i = 0; i < ${iterations}; i++) {\n        /*vec3 normal = texture3D(gradient, ray_pos).zyx;\n        normal = normal/ sqrt(normal.x*normal.x + normal.y*normal.y + normal.z*normal.z);\n        float cosangle = dot(light_dir, normal);\n        cosangle = clamp(cosangle, 0., 1.);*/\n\n\n        vec4 sample = texture3D(cube, ray_pos);\n        vec4 sample_x = texture3D(cube, ray_pos + vec3(delta, 0, 0));\n        vec4 sample_y = texture3D(cube, ray_pos + vec3(0, delta, 0));\n        vec4 sample_z = texture3D(cube, ray_pos + vec3(0, 0, delta));\n        if(1==1) {\n            vec3 normal = normalize(vec3((sample_x[0]-sample[0])/delta, (sample_y[0]-sample[0])/delta, (sample_z[0]-sample[0])/delta));\n            normal = -vec3(normal.x, normal.y, -normal.z);\n            //normal *= -sign(normal.z);\n            // since the 'surfaces' have two sides, the absolute of the normal is fine\n            float cosangle_light = max(abs(dot(light_dir, normal)), 0.);\n            float cosangle_eye = max(abs(dot(eye, normal)), 0.);\n\n            float data_value = (sample[0]/1000. - data_min) * data_scale;\n            depth_factor = clamp(depth_factor + (data_value > depth_peel ? 1. : 0.), 0., 1.);\n            vec4 color_sample = texture1D(transfer_function, data_value);\n\n\n            //vec4 color_sample = texture1D(texture_colormap, data_value);// * clamp(cosangle, 0.1, 1.);\n            float alpha_sample = color_sample.a * sign(data_value) * sign(1.-data_value) / float(steps) * 100.* ray_length; //function_opacities[j]*intensity * sign(data_value) * sign(1.-data_value) / float(steps) * 100.* ray_length ;//clamp(1.-chisq, 0., 1.) * 0.5;//1./128.* length(color_sample) * 100.;\n            alpha_sample = clamp(alpha_sample * foreground_opacity, 0., 1.);\n            alpha_sample *= depth_factor;\n            color_sample = color_sample * (ambient_coefficient + diffuse_coefficient*cosangle_light + specular_coefficient * pow(cosangle_eye, specular_exponent));\n            //color_sample = vec4(normal, 1.);\n            color = color + (1.0 - alpha_total) * color_sample * alpha_sample;\n            alpha_total = clamp(alpha_total + alpha_sample, 0., 1.);\n            if(alpha_total >= 1.)\n                break;\n        }\n        if(1==1) {\n            vec3 normal = normalize(-vec3((sample_x[1]-sample[1])/delta, (sample_y[1]-sample[1])/delta, (sample_z[1]-sample[1])/delta));\n            normal = -vec3(normal.x, normal.y, -normal.z);\n            float cosangle_light = max(dot(light_dir, normal), 0.);\n            float cosangle_eye = max(dot(eye, normal), 0.);\n\n            float data_value = (sample[1]/1000. - data_min) * data_scale;\n            vec4 color_sample = texture1D(transfer_function, data_value);\n\n            //vec4 color_sample = texture1D(texture_colormap, data_value);// * clamp(cosangle, 0.1, 1.);\n            float alpha_sample = color_sample.a * sign(data_value) * sign(1.-data_value) / float(steps) * 100.* ray_length; //function_opacities[j]*intensity * sign(data_value) * sign(1.-data_value) / float(steps) * 100.* ray_length ;//clamp(1.-chisq, 0., 1.) * 0.5;//1./128.* length(color_sample) * 100.;\n            alpha_sample = clamp(alpha_sample * background_opacity, 0., 1.);\n            color_sample = color_sample * (ambient_coefficient + diffuse_coefficient*cosangle_light + specular_coefficient * pow(cosangle_eye, specular_exponent));\n            //color_sample = vec4(normal, 1.);\n            color = color + (1.0 - alpha_total) * color_sample * alpha_sample;\n            alpha_total = clamp(alpha_total + alpha_sample, 0., 1.);\n            if(alpha_total >= 1.)\n                break;\n        }\n        ray_pos += ray_delta;\n    }\n    gl_FragColor = vec4(color.rgb, alpha_total) * brightness; //brightness;\n    //gl_FragColor = vec4(ray_pos.xyz, 1) * 100.; //brightness;\n}\n")
GL_R32F = 33326

class VolumeRenderWidget(QtOpenGL.QGLWidget):

    def __init__(self, parent=None, function_count=3):
        if False:
            return 10
        super(VolumeRenderWidget, self).__init__(parent)
        self.mouse_button_down = False
        self.mouse_button_down_right = False
        (self.mouse_x, self.mouse_y) = (0, 0)
        self.angle1 = 0
        self.angle2 = 0
        self.mod1 = 0
        self.mod2 = 0
        self.mod3 = 0
        self.mod4 = 0
        self.mod5 = 0
        self.mod6 = 0
        self.orbit_angle = 0
        self.orbit_delay = 50
        self.orbiting = False
        self.function_count = function_count
        self.function_opacities = [0.1 / 2 ** (function_count - 1 - k) for k in range(function_count)]
        self.function_sigmas = [0.05] * function_count
        self.function_means = np.arange(function_count) / float(function_count - 1) * 0.8 + 0.1
        self.brightness = 2.0
        self.min_level = 0.0
        self.max_level = 1.0
        self.min_level_vector3d = 0.0
        self.max_level_vector3d = 1.0
        self.vector3d_scale = 1.0
        self.vector3d_auto_scale = True
        self.vector3d_auto_scale_scale = 1.0
        self.texture_function_size = 1024 * 4
        self.ambient_coefficient = 0.5
        self.diffuse_coefficient = 0.8
        self.specular_coefficient = 0.5
        self.specular_exponent = 5.0
        self.draw_vectors = True
        self.background_opacity = 0.1
        self.foreground_opacity = 1.0
        (self.texture_cube, self.texture_gradient) = (None, None)
        self.setMouseTracking(True)
        shortcut = QtGui.QShortcut(QtGui.QKeySequence('space'), self)
        shortcut.activated.connect(self.toggle)
        self.texture_index = 1
        self.colormap_index = 0
        self.texture_size = 512
        self.grid_gl = None
        self.post_init = lambda : 1
        self.arrow_model = Arrow(0, 0, 0, 4.0)
        self.update_timer = QtCore.QTimer(self)
        self.update_timer.timeout.connect(self.orbit_progress)
        self.update_timer.setInterval(1000 / 50)
        self.ray_iterations = 500
        self.depth_peel = 0.0

    def set_iterations(self, iterations):
        if False:
            return 10
        self.ray_iterations = iterations
        self.shader_ray_cast = self.create_shader_ray_cast()

    def setResolution(self, size):
        if False:
            return 10
        self.texture_size = size
        self.makeCurrent()
        for texture in [self.texture_backside, self.texture_final]:
            glBindTexture(GL_TEXTURE_2D, texture)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, self.texture_size, self.texture_size, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
            glBindTexture(GL_TEXTURE_2D, 0)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.texture_backside, 0)
        glBindRenderbuffer(GL_RENDERBUFFER, self.render_buffer)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, self.texture_size, self.texture_size)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self.render_buffer)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def orbit_start(self):
        if False:
            return 10
        self.orbit_time_previous = time.time()
        self.stutter_last = time.time()
        self.update_timer.start()

    def orbit_stop(self):
        if False:
            i = 10
            return i + 15
        self.orbiting = False
        self.update_timer.stop()

    def orbit_progress(self):
        if False:
            print('Hello World!')
        orbit_time_now = time.time()
        delta_time = orbit_time_now - self.orbit_time_previous
        self.orbit_angle += delta_time / 4.0 * 360
        self.updateGL()
        glFinish()
        fps = 1.0 / delta_time
        if 1:
            stutter_time = time.time()
            print('.', fps, stutter_time - self.stutter_last)
            self.stutter_last = stutter_time
        self.orbit_time_previous = orbit_time_now

    def toggle(self, ignore=None):
        if False:
            i = 10
            return i + 15
        print('toggle')
        self.texture_index += 1
        self.update()

    def create_shader_color(self):
        if False:
            for i in range(10):
                print('nop')
        self.vertex_shader_color = shaders.compileShader('\n            varying vec4 vertex_color;\n            void main() {\n                gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;\n                vertex_color =  gl_Vertex /80. + vec4(0.5, 0.5, 0.5, 0.);\n                vertex_color.z = 1.-vertex_color.z;\n            }', GL_VERTEX_SHADER)
        self.fragment_shader_color = shaders.compileShader('\n            varying vec4 vertex_color;\n            void main() {\n                gl_FragColor = vertex_color;\n            }', GL_FRAGMENT_SHADER)
        return shaders.compileProgram(self.vertex_shader_color, self.fragment_shader_color)

    def create_shader_ray_cast(self):
        if False:
            print('Hello World!')
        self.vertex_shader = shaders.compileShader('\n            varying vec4 vertex_color;\n            void main() {\n                gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;\n                //vertex_color = gl_Color;\n                vertex_color =  gl_Vertex.x > 1.5 ? vec4(1,0,0,0) : vec4(0,1,0,0)  ;// vec4(gl_Color) + vec4(1, 0, 0, 0);\n                vertex_color =  gl_Vertex /80. + vec4(0.5, 0.5, 0.5, 0.);\n                vertex_color.z = 1.-vertex_color.z;\n            }', GL_VERTEX_SHADER)
        self.fragment_shader = shaders.compileShader(ray_cast_fragment_source.substitute(iterations=self.ray_iterations), GL_FRAGMENT_SHADER)
        return shaders.compileProgram(self.vertex_shader, self.fragment_shader, validate=False)

    def create_shader_vectorfield(self):
        if False:
            for i in range(10):
                print('nop')
        self.vertex_shader_color = shaders.compileShader('\n            #extension GL_ARB_draw_instanced : enable\n            varying vec4 vertex_color;\n            void main() {\n                float x = floor(float(gl_InstanceIDARB)/(8.*8.)) + 0.5;\n                float y = mod(floor(float(gl_InstanceIDARB)/8.), 8.) + 0.5;\n                float z = mod(float(gl_InstanceIDARB), 8.) + 0.5;\n                vec4 pos = (gl_Vertex + vec4(x*80./8., y*80./8., z*80./8., 0));\n                gl_Position = gl_ModelViewProjectionMatrix * pos;\n                vertex_color =  pos /80. + vec4(0.5, 0.5, 0.5, 0.);\n                //vertex_color.z = 1. - vertex_color.z;\n            }', GL_VERTEX_SHADER)
        self.fragment_shader_color = shaders.compileShader('\n            varying vec4 vertex_color;\n            void main() {\n                gl_FragColor = vertex_color;\n            }', GL_FRAGMENT_SHADER)
        return shaders.compileProgram(self.vertex_shader_color, self.fragment_shader_color)

    def create_shader_vectorfield_color(self):
        if False:
            print('Hello World!')
        self.vertex_shader_color = shaders.compileShader("\n        #version 120\n            #extension GL_ARB_draw_instanced : enable\n            varying vec4 vertex_color;\n            uniform sampler3D vectorfield;\n            uniform int grid_size;\n            uniform int use_light;\n            uniform vec3 light_color;\n            uniform vec3 lightdir;\n            uniform float count_level_min;\n            uniform float count_level_max;\n            uniform float vector3d_scale;\n            uniform float vector3d_auto_scale_scale;\n\n            void main() {\n                float grid_size_f = float(grid_size);\n                float x = floor(float(gl_InstanceIDARB)/(grid_size_f*grid_size_f))/grid_size_f;\n                float y = mod(floor(float(gl_InstanceIDARB)/grid_size_f), grid_size_f)/grid_size_f;\n                float z = mod(float(gl_InstanceIDARB), grid_size_f)/grid_size_f;\n                vec3 uniform_center = vec3(x, y, z);\n                vec4 sample = texture3D(vectorfield, uniform_center.yzx);\n                vec3 velocity = sample.xyz;\n                velocity.x *= -1.;\n                velocity.y *= -1.;\n                velocity.z *= 1.;\n                float counts = sample.a;\n                float scale = (counts >= count_level_min) && (counts <= count_level_max) ? 1. : 0.0;\n                float speed = length(velocity);\n                vec3 direction = normalize(velocity) ;// / speed;\n                // form two orthogonal vector to define a rotation matrix\n                // the rotation around the vector's axis doesn't matter\n                vec3 some_axis = normalize(vec3(0., 1., 1.));\n                //vec3 some_axis2 = normalize(vec3(1., 0., 1.));\n                vec3 axis1 = normalize(cross(direction, some_axis));\n                // + (1-length(cross(direction, some_axis)))*cross(direction, some_axis2));\n                vec3 axis2 = normalize(cross(direction, axis1));\n                mat3 rotation_and_scaling = mat3(axis1, axis2, direction * (speed) * 20. * vector3d_scale * vector3d_auto_scale_scale);\n                mat3 rotation_and_scaling_inverse_transpose = mat3(axis1, axis2, direction / (speed) / 20. / vector3d_scale / vector3d_auto_scale_scale);\n\n\n                vec3 pos = gl_Vertex.xyz;//\n                pos.z -= 0.5;\n                pos.z = -pos.z;\n                pos *= scale;\n                pos = rotation_and_scaling * pos;\n                vec3 center = (uniform_center - vec3(0.5,0.5,0.5) + 1./grid_size_f/2.) * 80.;\n                center.z = -center.z;\n                vec4 transformed_pos = vec4(pos + center, 1);\n                //transformed_pos.z = 80. - transformed_pos.z;\n                vertex_color =  transformed_pos/80. + vec4(0.5, 0.5, 0.5, 1.); //vec4(uniform_center + gl_ModelViewMatrix*pos, 0.);// + vec4(0.5, 0.5, 0.0, 1.);\n                vertex_color.z = 1. - vertex_color.z;\n                gl_Position = gl_ModelViewProjectionMatrix * transformed_pos;\n                if(use_light == 1) {\n                    float fraction = 0.5;\n                    vec3 normal =  normalize(mat3(gl_ModelViewMatrix) * rotation_and_scaling_inverse_transpose * gl_Normal);\n                    //vec3 normal = normalize(gl_NormalMatrix * gl_Normal);\n                    //mat3 rotation = mat3(m);\n                    vec3 lightdir_t = normalize(lightdir);\n                    vertex_color = vec4(light_color * fraction + max(dot(lightdir_t, normal), 0.), 1.);\n                    //vertex_color = vec4(normal, 1.0); //vec4(lightdir_t, 1.);\n                }\n\n            }", GL_VERTEX_SHADER)
        self.fragment_shader_color = shaders.compileShader('\n            varying vec4 vertex_color;\n            void main() {\n                gl_FragColor = vertex_color;\n            }', GL_FRAGMENT_SHADER)
        return shaders.compileProgram(self.vertex_shader_color, self.fragment_shader_color)

    def paintGL(self):
        if False:
            i = 10
            return i + 15
        if 1:
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            glTranslated(0.0, 0.0, -15.0)
            glRotated(self.orbit_angle, 0.0, 1.0, 0.0)
            glRotated(self.angle1, 1.0, 0.0, 0.0)
            glRotated(self.angle2, 0.0, 1.0, 0.0)
            if 0:
                glClearColor(0.0, 0.0, 0.0, 1.0)
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                glBegin(GL_TRIANGLES)
                glColor3f(1.0, 0.0, 0.0)
                glVertex3f(-30, -30, -20)
                glColor3f(0.0, 2.0, 0.0)
                glVertex3f(30, -30, -20)
                glColor3f(0.0, 0.0, 1.0)
                glVertex3f(0, 15, -20)
                glEnd()
            elif self.grid_gl is not None:
                self.draw_backside()
                self.draw_frontside()
                self.draw_to_screen()
            else:
                glViewport(0, 0, self.texture_size, self.texture_size)
                glBindFramebuffer(GL_FRAMEBUFFER, 0)
                glClearColor(0.0, 0.0, 0.0, 1.0)
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    def draw_backside(self):
        if False:
            for i in range(10):
                print('nop')
        glViewport(0, 0, self.texture_size, self.texture_size)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.texture_backside, 0)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_FRONT)
        glShadeModel(GL_SMOOTH)
        glUseProgram(self.shader_color)
        self.cube(size=80)
        self.wireframe(size=80.0)
        glCullFace(GL_BACK)
        if 0:
            self.cube(size=10)
        glUseProgram(self.shader_vectorfield_color)
        if self.vectorgrid is not None and self.draw_vectors:
            loc = glGetUniformLocation(self.shader_vectorfield_color, 'vectorfield')
            glUniform1i(loc, 0)
            loc = glGetUniformLocation(self.shader_vectorfield_color, 'grid_size')
            glUniform1i(loc, self.vectorgrid.shape[0])
            loc = glGetUniformLocation(self.shader_vectorfield_color, 'use_light')
            glUniform1i(loc, 0)
            (mi, ma) = (np.nanmin(self.vectorgrid_counts), np.nanmax(self.vectorgrid_counts))
            loc = glGetUniformLocation(self.shader_vectorfield_color, 'count_level_min')
            glUniform1f(loc, 10 ** (np.log10(ma) * self.min_level_vector3d))
            loc = glGetUniformLocation(self.shader_vectorfield_color, 'count_level_max')
            glUniform1f(loc, 10 ** (np.log10(ma) * self.max_level_vector3d))
            loc = glGetUniformLocation(self.shader_vectorfield_color, 'vector3d_scale')
            glUniform1f(loc, self.vector3d_scale)
            loc = glGetUniformLocation(self.shader_vectorfield_color, 'vector3d_auto_scale_scale')
            glUniform1f(loc, self.vector3d_auto_scale_scale if self.vector3d_auto_scale else 1.0)
            glActiveTexture(GL_TEXTURE0)
            glEnable(GL_TEXTURE_3D)
            glBindTexture(GL_TEXTURE_3D, self.texture_cube_vector)
            self.arrow_model.drawGL(self.vectorgrid.shape[0] ** 3)
        glUseProgram(0)
        glActiveTexture(GL_TEXTURE0)
        glDisable(GL_TEXTURE_3D)

    def arrow(self, x, y, z, scale):
        if False:
            return 10
        headfraction = 0.4
        baseradius = 0.1 * scale
        headradius = 0.2 * scale
        glBegin(GL_QUADS)
        for part in range(10):
            angle = np.radians(part / 10.0 * 360)
            angle2 = np.radians((part + 1) / 10.0 * 360)
            glNormal3f(np.cos(angle), np.sin(angle), 0.0)
            glVertex3f(x + baseradius * np.cos(angle), y + baseradius * np.sin(angle), z + scale / 2 - headfraction * scale)
            glNormal3f(np.cos(angle), np.sin(angle), 0.0)
            glVertex3f(x + baseradius * np.cos(angle), y + baseradius * np.sin(angle), z - scale / 2)
            glNormal3f(np.cos(angle2), np.sin(angle2), 0.0)
            glVertex3f(x + baseradius * np.cos(angle2), y + baseradius * np.sin(angle2), z - scale / 2)
            glNormal3f(np.cos(angle2), np.sin(angle2), 0.0)
            glVertex3f(x + baseradius * np.cos(angle2), y + baseradius * np.sin(angle2), z + scale / 2 - headfraction * scale)
        glEnd()
        glBegin(GL_TRIANGLE_FAN)
        glNormal3f(0, 0, -1)
        glVertex3f(x, y, z - scale / 2)
        for part in range(10 + 1):
            angle = np.radians(-part / 10.0 * 360)
            glVertex3f(x + baseradius * np.cos(angle), y + baseradius * np.sin(angle), z - scale / 2)
        glEnd()
        glBegin(GL_TRIANGLES)
        a = headradius - baseradius
        b = headfraction * scale
        headangle = np.arctan(a / b)
        for part in range(10 + 1):
            angle = np.radians(-part / 10.0 * 360)
            anglemid = np.radians(-(part + 0.5) / 10.0 * 360)
            angle2 = np.radians(-(part + 1) / 10.0 * 360)
            glNormal3f(np.cos(anglemid) * np.cos(headangle), np.sin(anglemid) * np.cos(headangle), np.sin(headangle))
            glVertex3f(x, y, z + scale / 2)
            glNormal3f(np.cos(angle2) * np.cos(headangle), np.sin(angle2) * np.cos(headangle), np.sin(headangle))
            glVertex3f(x + headradius * np.cos(angle2), y + headradius * np.sin(angle2), z + scale / 2 - headfraction * scale)
            glNormal3f(np.cos(angle) * np.cos(headangle), np.sin(angle) * np.cos(headangle), np.sin(headangle))
            glVertex3f(x + headradius * np.cos(angle), y + headradius * np.sin(angle), z + scale / 2 - headfraction * scale)
        glEnd()
        glBegin(GL_QUADS)
        glNormal3f(0, 0, -1)
        for part in range(10):
            angle = np.radians(part / 10.0 * 360)
            angle2 = np.radians((part + 1) / 10.0 * 360)
            glVertex3f(x + baseradius * np.cos(angle), y + baseradius * np.sin(angle), z + scale / 2 - headfraction * scale)
            glVertex3f(x + baseradius * np.cos(angle2), y + baseradius * np.sin(angle2), z + scale / 2 - headfraction * scale)
            glVertex3f(x + headradius * np.cos(angle2), y + headradius * np.sin(angle2), z + scale / 2 - headfraction * scale)
            glVertex3f(x + headradius * np.cos(angle), y + headradius * np.sin(angle), z + scale / 2 - headfraction * scale)
        glEnd()

    def draw_frontside(self):
        if False:
            return 10
        glViewport(0, 0, self.texture_size, self.texture_size)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.texture_final, 0)
        glClearColor(1.0, 1.0, 1.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glShadeModel(GL_SMOOTH)
        glDisable(GL_BLEND)
        glColor3f(0, 0, 0)
        self.wireframe(size=80.0)
        if 0:
            self.cube(size=10)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        g = 0.5
        glMaterialfv(GL_FRONT, GL_SPECULAR, [g, g, g, 1.0])
        glMaterialfv(GL_FRONT, GL_DIFFUSE, [g, g, g, 1.0])
        glPushMatrix()
        glLoadIdentity()
        glLightfv(GL_LIGHT0, GL_POSITION, [0.1, 0.1, 1, 0.0])
        glPopMatrix()
        a = 0.5
        glLightfv(GL_LIGHT0, GL_AMBIENT, [a, a, a, 0.0])
        glMaterialfv(GL_FRONT, GL_SHININESS, [50.0])
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glEnable(GL_COLOR_MATERIAL)
        glColor3f(0.5, 0, 0)
        if self.vectorgrid is not None and self.draw_vectors:
            glUseProgram(self.shader_vectorfield_color)
            loc = glGetUniformLocation(self.shader_vectorfield_color, 'vectorfield')
            glUniform1i(loc, 0)
            loc = glGetUniformLocation(self.shader_vectorfield_color, 'grid_size')
            glUniform1i(loc, self.vectorgrid.shape[0])
            loc = glGetUniformLocation(self.shader_vectorfield_color, 'use_light')
            glUniform1i(loc, 1)
            loc = glGetUniformLocation(self.shader_vectorfield_color, 'light_color')
            glUniform3f(loc, 1.0, 0.0, 0.0)
            loc = glGetUniformLocation(self.shader_vectorfield_color, 'lightdir')
            glUniform3f(loc, -1.0, -1.0, 1.0)
            (mi, ma) = (np.nanmin(self.vectorgrid_counts), np.nanmax(self.vectorgrid_counts))
            loc = glGetUniformLocation(self.shader_vectorfield_color, 'count_level_min')
            glUniform1f(loc, 10 ** (np.log10(ma) * self.min_level_vector3d))
            loc = glGetUniformLocation(self.shader_vectorfield_color, 'count_level_max')
            glUniform1f(loc, 10 ** (np.log10(ma) * self.max_level_vector3d))
            loc = glGetUniformLocation(self.shader_vectorfield_color, 'vector3d_scale')
            glUniform1f(loc, self.vector3d_scale)
            loc = glGetUniformLocation(self.shader_vectorfield_color, 'vector3d_auto_scale_scale')
            glUniform1f(loc, self.vector3d_auto_scale_scale if self.vector3d_auto_scale else 1.0)
            glActiveTexture(GL_TEXTURE0)
            glEnable(GL_TEXTURE_3D)
            glBindTexture(GL_TEXTURE_3D, self.texture_cube_vector)
            self.arrow_model.drawGL(self.vectorgrid.shape[0] ** 3)
            glDisable(GL_TEXTURE_3D)
        glUseProgram(0)
        glDisable(GL_LIGHTING)
        glDisable(GL_LIGHT0)
        glEnable(GL_BLEND)
        glBlendEquation(GL_FUNC_ADD, GL_FUNC_ADD)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glUseProgram(self.shader_ray_cast)
        loc = glGetUniformLocation(self.shader_ray_cast, 'texture')
        glUniform1i(loc, 0)
        glBindTexture(GL_TEXTURE_2D, self.texture_backside)
        glEnable(GL_TEXTURE_2D)
        glActiveTexture(GL_TEXTURE0)
        loc = glGetUniformLocation(self.shader_ray_cast, 'cube')
        glUniform1i(loc, 1)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_3D, self.texture_cube)
        loc = glGetUniformLocation(self.shader_ray_cast, 'texture_colormap')
        glUniform1i(loc, 2)
        glActiveTexture(GL_TEXTURE2)
        index = 16
        glBindTexture(GL_TEXTURE_1D, self.textures_colormap[index])
        glEnable(GL_TEXTURE_1D)
        if 1:
            loc = glGetUniformLocation(self.shader_ray_cast, 'transfer_function')
            glUniform1i(loc, 3)
            glActiveTexture(GL_TEXTURE3)
            glBindTexture(GL_TEXTURE_1D, self.texture_function)
            rgb = self.colormap_data[self.colormap_index]
            x = np.arange(self.texture_function_size) / (self.texture_function_size - 1.0)
            y = x * 0.0
            for i in range(3):
                y += np.exp(-((x - self.function_means[i]) / self.function_sigmas[i]) ** 2) * self.function_opacities[i]
            self.function_data[:, 0] = rgb[:, 0]
            self.function_data[:, 1] = rgb[:, 1]
            self.function_data[:, 2] = rgb[:, 2]
            self.function_data[:, 3] = (y * 255).astype(np.uint8)
            self.function_data_1d = self.function_data.reshape(-1)
            glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA8, self.texture_function_size, 0, GL_RGBA, GL_UNSIGNED_BYTE, self.function_data_1d)
            glEnable(GL_TEXTURE_1D)
        if 0:
            loc = glGetUniformLocation(self.shader_ray_cast, 'gradient')
            glUniform1i(loc, 4)
            glActiveTexture(GL_TEXTURE4)
            glEnable(GL_TEXTURE_3D)
            glBindTexture(GL_TEXTURE_3D, self.texture_gradient)
        glActiveTexture(GL_TEXTURE0)
        size = glGetUniformLocation(self.shader_ray_cast, 'size')
        glUniform2f(size, self.texture_size, self.texture_size)
        depth_peel = glGetUniformLocation(self.shader_ray_cast, 'depth_peel')
        glUniform1f(depth_peel, self.depth_peel)
        minmax = glGetUniformLocation(self.shader_ray_cast, 'minmax3d')
        glUniform2f(minmax, self.min_level, self.max_level)
        glUniform1f(glGetUniformLocation(self.shader_ray_cast, 'brightness'), self.brightness)
        glUniform1f(glGetUniformLocation(self.shader_ray_cast, 'background_opacity'), self.background_opacity)
        glUniform1f(glGetUniformLocation(self.shader_ray_cast, 'foreground_opacity'), self.foreground_opacity)
        glUniform1fv(glGetUniformLocation(self.shader_ray_cast, 'function_means'), self.function_count, self.function_means)
        glUniform1fv(glGetUniformLocation(self.shader_ray_cast, 'function_sigmas'), self.function_count, self.function_sigmas)
        glUniform1fv(glGetUniformLocation(self.shader_ray_cast, 'function_opacities'), self.function_count, self.function_opacities)
        for name in ['ambient_coefficient', 'diffuse_coefficient', 'specular_coefficient', 'specular_exponent']:
            glUniform1f(glGetUniformLocation(self.shader_ray_cast, name), getattr(self, name))
        alpha_mod = glGetUniformLocation(self.shader_ray_cast, 'alpha_mod')
        glUniform1f(alpha_mod, 10 ** self.mod3)
        for i in range(4, 7):
            name = 'mod' + str(i)
            mod = glGetUniformLocation(self.shader_ray_cast, name)
            glUniform1f(mod, 10 ** getattr(self, name))
        self.shader_ray_cast.check_validate()
        glShadeModel(GL_SMOOTH)
        self.cube(size=80)
        glUseProgram(0)
        glActiveTexture(GL_TEXTURE4)
        glBindTexture(GL_TEXTURE_3D, 0)
        glEnable(GL_TEXTURE_2D)
        glActiveTexture(GL_TEXTURE3)
        glBindTexture(GL_TEXTURE_1D, 0)
        glEnable(GL_TEXTURE_2D)
        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_1D, 0)
        glEnable(GL_TEXTURE_2D)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, 0)
        glEnable(GL_TEXTURE_2D)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glEnable(GL_TEXTURE_2D)
        glDisable(GL_BLEND)

    def draw_to_screen(self):
        if False:
            print('Hello World!')
        w = self.width()
        h = self.height()
        glViewport(0, 0, w, h)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glClearColor(1.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glCullFace(GL_BACK)
        glBindTexture(GL_TEXTURE_2D, self.textures[self.texture_index % len(self.textures)])
        glEnable(GL_TEXTURE_2D)
        glLoadIdentity()
        glBegin(GL_QUADS)
        w = 50
        z = -1
        glTexCoord2f(0, 0)
        glVertex3f(-w, -w, z)
        glTexCoord2f(1, 0)
        glVertex3f(w, -w, z)
        glTexCoord2f(1, 1)
        glVertex3f(w, w, z)
        glTexCoord2f(0, 1)
        glVertex3f(-w, w, z)
        glEnd()
        glBindTexture(GL_TEXTURE_2D, 0)

    def draw_to_screen_(self):
        if False:
            while True:
                i = 10
        w = self.width()
        h = self.height()
        glViewport(0, 0, w, h)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glClearColor(0.0, 1.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glBindTexture(GL_TEXTURE_1D, 0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glEnable(GL_TEXTURE_3D)
        glBindTexture(GL_TEXTURE_3D, self.texture_cube)
        glEnable(GL_TEXTURE_3D)
        glColor3f(1, 0, 0)
        glLoadIdentity()
        glBegin(GL_QUADS)
        w = 20
        z = -1
        glTexCoord3f(0, 0, 0.5)
        glVertex3f(-w, -w, z)
        glTexCoord3f(1, 0, 0.5)
        glVertex3f(w, -w, z)
        glTexCoord3f(1, 1, 0.5)
        glVertex3f(w, w, z)
        glTexCoord3f(0, 1, 0.5)
        glVertex3f(-w, w, z)
        glEnd()
        glBindTexture(GL_TEXTURE_3D, 0)

    def wireframe(self, size, color_axis=False):
        if False:
            for i in range(10):
                print('nop')
        w = size / 2
        glLineWidth(2.0)
        glBegin(GL_LINES)
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(-w, -w, w)
        glVertex3f(w, -w, w)
        glEnd()
        glBegin(GL_LINES)
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(-w, -w, w)
        glVertex3f(-w, w, w)
        glEnd()
        glBegin(GL_LINES)
        glColor3f(0.0, 0.0, 1.0)
        glVertex3f(-w, -w, -w)
        glVertex3f(-w, -w, w)
        glEnd()
        glLineWidth(1.0)
        glColor3f(0.0, 0.0, 0.0)
        glBegin(GL_LINES)
        glVertex3f(-w, w, -w)
        glVertex3f(-w, w, w)
        glEnd()
        glBegin(GL_LINES)
        glVertex3f(w, w, -w)
        glVertex3f(w, w, w)
        glEnd()
        glBegin(GL_LINES)
        glVertex3f(w, -w, -w)
        glVertex3f(w, -w, w)
        glEnd()
        glBegin(GL_LINES)
        glVertex3f(w, -w, -w)
        glVertex3f(w, w, -w)
        glEnd()
        glBegin(GL_LINES)
        glVertex3f(w, -w, w)
        glVertex3f(w, w, w)
        glEnd()
        glBegin(GL_LINES)
        glVertex3f(-w, -w, -w)
        glVertex3f(-w, w, -w)
        glEnd()
        glBegin(GL_LINES)
        glVertex3f(-w, w, -w)
        glVertex3f(w, w, -w)
        glEnd()
        glBegin(GL_LINES)
        glVertex3f(-w, -w, -w)
        glVertex3f(w, -w, -w)
        glEnd()
        glBegin(GL_LINES)
        glVertex3f(-w, w, w)
        glVertex3f(w, w, w)
        glEnd()

    def cube(self, size, gl_type=GL_QUADS):
        if False:
            i = 10
            return i + 15
        w = size / 2.0

        def vertex(x, y, z):
            if False:
                while True:
                    i = 10
            glVertex3f(x, y, z)
        if 1:
            glBegin(gl_type)
            vertex(-w, -w, -w)
            vertex(-w, w, -w)
            vertex(w, w, -w)
            vertex(w, -w, -w)
            glEnd()
        if 1:
            glBegin(gl_type)
            vertex(-w, -w, w)
            vertex(w, -w, w)
            vertex(w, w, w)
            vertex(-w, w, w)
            glEnd()
        if 1:
            glBegin(gl_type)
            vertex(w, -w, w)
            vertex(w, -w, -w)
            vertex(w, w, -w)
            vertex(w, w, w)
            glEnd()
        if 1:
            glBegin(gl_type)
            vertex(-w, -w, -w)
            vertex(-w, -w, w)
            vertex(-w, w, w)
            vertex(-w, w, -w)
            glEnd()
        if 1:
            glBegin(gl_type)
            vertex(w, w, -w)
            vertex(-w, w, -w)
            vertex(-w, w, w)
            vertex(w, w, w)
            glEnd()
        if 1:
            glBegin(gl_type)
            vertex(-w, -w, -w)
            vertex(w, -w, -w)
            vertex(w, -w, w)
            vertex(-w, -w, w)
            glEnd()

    def resizeGL(self, w, h):
        if False:
            while True:
                i = 10
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-50, 50, -50, 50, -150.0, 150.0)
        glViewport(0, 0, w, h)

    def initializeGL(self):
        if False:
            print('Hello World!')
        colormaps = vaex.ui.colormaps.colormaps
        (Nx, Ny) = (self.texture_function_size, 16)
        self.colormap_data = np.zeros((len(colormaps), Nx, 3), dtype=np.uint8)
        import matplotlib.cm
        self.textures_colormap = glGenTextures(len(colormaps))
        for (i, colormap_name) in enumerate(colormaps):
            colormap = matplotlib.cm.get_cmap(colormap_name)
            mapping = matplotlib.cm.ScalarMappable(cmap=colormap)
            x = np.arange(Nx) / (Nx - 1.0)
            rgba = mapping.to_rgba(x, bytes=True).reshape(Nx, 4)
            rgb = rgba[:, 0:3] * 1
            self.colormap_data[i] = rgb
            if i == 0:
                print(rgb[0], rgb[-1], end=' ')
            texture = self.textures_colormap[i]
            glBindTexture(GL_TEXTURE_1D, texture)
            glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)
            glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexImage1D(GL_TEXTURE_1D, 0, GL_RGB8, Nx, 0, GL_RGB, GL_UNSIGNED_BYTE, self.colormap_data[i])
            glBindTexture(GL_TEXTURE_1D, 0)
        if 1:
            self.texture_function = glGenTextures(1)
            texture = self.texture_function
            glBindTexture(GL_TEXTURE_1D, texture)
            glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)
            glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            self.function_data = np.zeros((self.texture_function_size, 4), dtype=np.uint8)
            x = np.arange(self.texture_function_size) * 255 / (self.texture_function_size - 1.0)
            self.function_data[:, 0] = x
            self.function_data[:, 1] = x
            self.function_data[:, 2] = 0
            self.function_data[:, 3] = x
            self.function_data_1d = self.function_data.reshape(-1)
            glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA8, Nx, 0, GL_RGBA, GL_UNSIGNED_BYTE, self.function_data_1d)
            glBindTexture(GL_TEXTURE_1D, 0)
        if 1:
            N = 1024 * 4
            self.surface_data = np.zeros((N, 3), dtype=np.uint8)
            self.texture_surface = glGenTextures(1)
            glBindTexture(GL_TEXTURE_1D, self.texture_surface)
            glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)
            glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexImage1D(GL_TEXTURE_1D, 0, GL_RGB8, Nx, 0, GL_RGB, GL_UNSIGNED_BYTE, self.surface_data)
            glBindTexture(GL_TEXTURE_1D, 0)
        if 0:
            f = glCreateShaderObject(GL_FRAGMENT_SHADER)
            fragment_source = 'void main(){ gl_FragColor=gl_FragCoord/512.0; }'
            glShaderSource(f, 1, fs, None)
            glCompileShaderARB(f)
            self.program = glCreateProgramObjectARB()
            glAttachObjectARB(self.program, f)
        self.fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        self.textures = (self.texture_backside, self.texture_final) = glGenTextures(2)
        print('textures', self.textures)
        for texture in [self.texture_backside, self.texture_final]:
            glBindTexture(GL_TEXTURE_2D, texture)
            glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, self.texture_size, self.texture_size, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
            glBindTexture(GL_TEXTURE_2D, 0)
        glFramebufferTexture2D(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.texture_backside, 0)
        self.render_buffer = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self.render_buffer)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, self.texture_size, self.texture_size)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self.render_buffer)
        self.shader_ray_cast = self.create_shader_ray_cast()
        self.shader_color = self.create_shader_color()
        self.shader_vectorfield = self.create_shader_vectorfield()
        self.shader_vectorfield_color = self.create_shader_vectorfield_color()
        self.post_init()
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def loadTable(self, args, column_names, grid_size=128, grid_size_vector=16):
        if False:
            for i in range(10):
                print('nop')
        import vaex.vaexfast
        dataset = vaex.dataset.load_file(sys.argv[1])
        (x, y, z, vx, vy, vz) = [dataset.columns[name] for name in sys.argv[2:]]
        (x, y, z, vx, vy, vz) = [k.astype(np.float64) - k.mean() for k in [x, y, z, vx, vy, vz]]
        grid3d = np.zeros((grid_size, grid_size, grid_size), dtype=np.float64)
        vectorgrid = np.zeros((4, grid_size_vector, grid_size_vector, grid_size_vector), dtype=np.float64)
        s = 0.0
        (mi, ma) = (-4, 4)
        print('histogram3d')
        vaex.vaexfast.histogram3d(x, y, z, None, grid3d, mi + s, ma + s, mi, ma, mi, ma)
        if 0:
            vx = vx - vx.mean()
            vy = vy - vy.mean()
            vz = vz - vz.mean()
        vaex.vaexfast.histogram3d(x, y, z, vx, vectorgrid[0], mi + s, ma + s, mi, ma, mi, ma)
        vaex.vaexfast.histogram3d(x, y, z, vy, vectorgrid[1], mi + s, ma + s, mi, ma, mi, ma)
        vaex.vaexfast.histogram3d(x, y, z, vz, vectorgrid[2], mi + s, ma + s, mi, ma, mi, ma)
        print(vx)
        print(vectorgrid[0])
        print(vaex.vaexfast.resize(vectorgrid[0], 4))
        print(vaex.vaexfast.resize(vectorgrid[1], 4))
        print(vaex.vaexfast.resize(vectorgrid[2], 4))
        print(vaex.vaexfast.resize(grid3d, 4))
        print('$' * 80)
        vectorgrid[3][:] = vaex.vaexfast.resize(grid3d, grid_size_vector)
        for i in range(3):
            vectorgrid[i] /= vectorgrid[3]
        if 1:
            vmax = max([np.nanmax(vectorgrid[0]), np.nanmax(vectorgrid[1]), np.nanmax(vectorgrid[2])])
            for i in range(3):
                vectorgrid[i] *= 1
        vectorgrid = np.swapaxes(vectorgrid, 0, 3)
        self.setGrid(grid3d, vectorgrid=vectorgrid)

    def setGrid(self, grid, grid_background=None, vectorgrid=None):
        if False:
            print('Hello World!')
        self.mod1 = 0
        self.mod2 = 0
        self.mod3 = 0
        self.mod4 = 0
        self.mod5 = 0
        self.mod6 = 0
        if vectorgrid is not None:
            self.vectorgrid = vectorgrid.astype(np.float32)
            self.vectorgrid_counts = self.vectorgrid[:, :, :, 3]
        else:
            self.vectorgrid = None

        def normalise(ar):
            if False:
                i = 10
                return i + 15
            mask = ~np.isinf(ar)
            (mi, ma) = (np.nanmin(ar[mask]), np.nanmax(ar[mask]))
            res = (ar - mi) / (ma - mi) * 1000.0
            res[~mask] = mi
            return res
        if grid_background is not None:
            self.grid_gl = np.zeros(grid.shape + (2,), np.float32)
            self.grid_gl[:, :, :, 0] = normalise(grid.astype(np.float32))
            self.grid_gl[:, :, :, 1] = normalise(grid_background.astype(np.float32))
        else:
            self.grid_gl = np.zeros(grid.shape + (1,), np.float32)
            self.grid_gl[:, :, :, 0] = normalise(grid.astype(np.float32))
        if 0:
            self.grid_gradient = np.gradient(self.grid)
            length = np.sqrt(self.grid_gradient[0] ** 2 + self.grid_gradient[1] ** 2 + self.grid_gradient[2] ** 2)
            self.grid_gradient[0] = self.grid_gradient[0] / length
            self.grid_gradient[1] = self.grid_gradient[1] / length
            self.grid_gradient[2] = self.grid_gradient[2] / length
            self.grid_gradient_data = np.zeros(self.grid.shape + (3,), dtype=np.float32)
            self.grid_gradient_data[:, :, :, 0] = self.grid_gradient[0]
            self.grid_gradient_data[:, :, :, 1] = self.grid_gradient[1]
            self.grid_gradient_data[:, :, :, 2] = self.grid_gradient[2]
            self.grid_gradient_data[:, :, :, 2] = 1.0
            self.grid_gradient = self.grid_gradient_data
            del self.grid_gradient_data
            print(self.grid_gradient.shape)
        for texture in [self.texture_cube, self.texture_gradient]:
            logger.debug('deleting texture: %r', texture)
            if texture is not None:
                glDeleteTextures(int(texture))
        self.texture_cube = glGenTextures(1)
        self.texture_gradient = glGenTextures(1)
        logger.debug('texture: %s %s', self.texture_cube, self.texture_gradient)
        logger.debug('texture types: %s %s', type(self.texture_cube), type(self.texture_gradient))
        if 0:
            self.rgb3d = np.zeros(self.grid.shape + (3,), dtype=np.uint8)
            self.rgb3d[:, :, :, 0] = self.grid
            self.rgb3d[:, :, :, 1] = self.grid
            self.rgb3d[:, :, :, 2] = self.grid
        glBindTexture(GL_TEXTURE_3D, self.texture_cube)
        (width, height, depth) = grid.shape[::-1]
        print('dims', width, height, depth)
        if grid_background is not None:
            glTexImage3D(GL_TEXTURE_3D, 0, GL_RG32F, width, height, depth, 0, GL_RG, GL_FLOAT, self.grid_gl)
        else:
            glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, width, height, depth, 0, GL_RED, GL_FLOAT, self.grid_gl)
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        if 1:
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER)
        glBindTexture(GL_TEXTURE_3D, 0)
        if self.vectorgrid is not None:
            assert self.vectorgrid.shape[0] == self.vectorgrid.shape[1] == self.vectorgrid.shape[2], 'wrong shape %r' % self.vectorgrid.shape
            vectorflat = self.vectorgrid.reshape((-1, 4))
            (vx, vy, vz, counts) = vectorflat.T
            mask = counts > 0
            target_length = np.nanmean(np.sqrt(vx[mask] ** 2 + vy[mask] ** 2 + vz[mask] ** 2))
            self.vector3d_auto_scale_scale = 1.0 / target_length / self.vectorgrid.shape[0]
            print('#' * 200)
            print(self.vector3d_auto_scale_scale, target_length)
            self.texture_cube_vector_size = self.vectorgrid.shape[0]
            self.texture_cube_vector = glGenTextures(1)
            glBindTexture(GL_TEXTURE_3D, self.texture_cube_vector)
            (_, width, height, depth) = self.vectorgrid.shape[::-1]
            print('dims vector', width, height, depth)
            glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA32F, width, height, depth, 0, GL_RGBA, GL_FLOAT, self.vectorgrid)
            glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
            glBindTexture(GL_TEXTURE_3D, 0)
        if 0:
            glBindTexture(GL_TEXTURE_3D, self.texture_gradient)
            glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB32F, width, height, depth, 0, GL_RGB, GL_FLOAT, self.grid_gradient)
            glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            if 1:
                glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
                glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
                glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER)
            glBindTexture(GL_TEXTURE_3D, 0)
        if 0:
            from matplotlib import pyplot as plt
            plt.subplot(221)
            plt.imshow(np.log10(grids_2d[0] + 1))
            plt.subplot(222)
            plt.imshow(np.log10(grids_2d[1] + 1))
            plt.subplot(223)
            plt.imshow(np.log10(grids_2d[2] + 1))
            plt.subplot(224)
            plt.imshow(np.log10(self.grid[128] + 1))
            plt.show()
        self.update()

    def mouseMoveEvent(self, event):
        if False:
            i = 10
            return i + 15
        (x, y) = (event.x(), event.y())
        dx = x - self.mouse_x
        dy = y - self.mouse_y
        speed = 1.0
        speed_mod = 0.1 / 5.0 / 5.0
        if self.mouse_button_down:
            self.angle2 += dx * speed
            self.angle1 += dy * speed
            print(self.angle1, self.angle2)
        if self.mouse_button_down_right:
            if QtGui.QApplication.keyboardModifiers() == QtCore.Qt.NoModifier:
                self.min_level += dx * speed_mod / 10.0
                self.max_level += -dy * speed_mod / 10.0
                print('mod1/2', self.min_level, self.max_level)
            if QtGui.QApplication.keyboardModifiers() == QtCore.Qt.AltModifier or QtGui.QApplication.keyboardModifiers() == QtCore.Qt.ControlModifier:
                self.mod3 += dx * speed_mod
                self.mod4 += -dy * speed_mod
                print('mod3/4', self.mod3, self.mod4)
            if QtGui.QApplication.keyboardModifiers() == QtCore.Qt.ShiftModifier:
                self.mod5 += dx * speed_mod
                self.mod6 += -dy * speed_mod
                print('mod5/6', self.mod5, self.mod6)
        (self.mouse_x, self.mouse_y) = (x, y)
        if not self.update_timer.isActive() and (self.mouse_button_down or self.mouse_button_down_right):
            self.updateGL()

    def mousePressEvent(self, event):
        if False:
            print('Hello World!')
        if event.button() == QtCore.Qt.LeftButton:
            self.mouse_button_down = True
        if event.button() == QtCore.Qt.RightButton:
            self.mouse_button_down_right = True

    def mouseReleaseEvent(self, event):
        if False:
            print('Hello World!')
        if event.button() == QtCore.Qt.LeftButton:
            self.mouse_button_down = False
        if event.button() == QtCore.Qt.RightButton:
            self.mouse_button_down_right = False

    def write(self):
        if False:
            for i in range(10):
                print('nop')
        colormap_name = 'afmhot'
        import matplotlib.cm
        colormap = matplotlib.cm.get_cmap(colormap_name)
        mapping = matplotlib.cm.ScalarMappable(cmap=colormap)
        data = np.zeros((128 * 8, 128 * 16, 4), dtype=np.uint8)
        (mi, ma) = (1 * 10 ** self.mod1, self.data3d.max() * 10 ** self.mod2)
        intensity_normalized = (np.log(self.data3d + 1.0) - np.log(mi)) / (np.log(ma) - np.log(mi))
        import PIL.Image
        for y2d in range(8):
            for x2d in range(16):
                zindex = x2d + y2d * 16
                I = intensity_normalized[zindex]
                rgba = mapping.to_rgba(I, bytes=True)
                print(rgba.shape)
                subdata = data[y2d * 128:(y2d + 1) * 128, x2d * 128:(x2d + 1) * 128]
                for i in range(3):
                    subdata[:, :, i] = rgba[:, :, i]
                subdata[:, :, 3] = (intensity_normalized[zindex] * 255).astype(np.uint8)
                if 0:
                    filename = 'cube%03d.png' % zindex
                    img = PIL.Image.frombuffer('RGB', (128, 128), subdata[:, :, 0:3] * 1)
                    print('saving to', filename)
                    img.save(filename)
        img = PIL.Image.frombuffer('RGBA', (128 * 16, 128 * 8), data)
        filename = 'cube.png'
        print('saving to', filename)
        img.save(filename)
        filename = 'colormap.png'
        print('saving to', filename)
        (height, width) = self.colormap_data.shape[:2]
        img = PIL.Image.frombuffer('RGB', (width, height), self.colormap_data)
        img.save(filename)

class TestWidget(QtGui.QMainWindow):

    def __init__(self, parent):
        if False:
            print('Hello World!')
        super(TestWidget, self).__init__(parent)
        self.resize(700, 700)
        self.show()
        self.raise_()
        shortcut = QtGui.QShortcut(QtGui.QKeySequence('Cmd+Q'), self)
        shortcut.activated.connect(self.myclose)
        self.main = VolumeRenderWidget(self)
        self.setCentralWidget(self.main)

    def myclose(self, ignore=None):
        if False:
            while True:
                i = 10
        self.hide()

class Arrow(object):

    def begin(self, type):
        if False:
            print('Hello World!')
        self.type = type

    def end(self):
        if False:
            for i in range(10):
                print('nop')
        self.offset = len(self.vertices)

    def vertex3f(self, x, y, z):
        if False:
            i = 10
            return i + 15
        self.vertices.append([x, y, z])
        self.normals.append(list(self.current_normal))

    def normal3f(self, x, y, z):
        if False:
            i = 10
            return i + 15
        self.current_normal = [x, y, z]

    def tri(self, i1, i2, i3):
        if False:
            while True:
                i = 10
        self.indices.append(self.offset + i1)
        self.indices.append(self.offset + i2)
        self.indices.append(self.offset + i3)

    def drawGL(self, instances=1):
        if False:
            i = 10
            return i + 15
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        vertices_ptr = self.vertices.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        glVertexPointer(3, GL_FLOAT, 0, vertices_ptr)
        normal_ptr = self.normals.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        glNormalPointer(GL_FLOAT, 0, normal_ptr)
        indices_ptr = self.indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        glDrawElementsInstanced(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, indices_ptr, instances)
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)

    def __init__(self, x, y, z, scale=80, headfraction=0.4, baseradius=0.1, headradius=0.2):
        if False:
            while True:
                i = 10
        self.vertices = []
        self.normals = []
        self.indices = []
        self.offset = 0
        headfraction = 0.5
        baseradius = 0.1 * scale / 1.5
        headradius = 0.2 * scale
        self.begin(GL_QUADS)
        parts = 10
        for part in range(parts):
            angle = np.radians(part / 10.0 * 360)
            angle2 = np.radians((part + 1) / 10.0 * 360)
            self.normal3f(np.cos(angle), np.sin(angle), 0.0)
            self.vertex3f(x + baseradius * np.cos(angle), y + baseradius * np.sin(angle), z + scale / 2 - headfraction * scale)
            self.normal3f(np.cos(angle), np.sin(angle), 0.0)
            self.vertex3f(x + baseradius * np.cos(angle), y + baseradius * np.sin(angle), z - scale / 2)
        for part in range(10 + 1):
            self.tri((part * 2 + 0) % (10 * 2), (part * 2 + 1) % (10 * 2), (part * 2 + 2) % (10 * 2))
            self.tri((part * 2 + 2) % (10 * 2), (part * 2 + 1) % (10 * 2), (part * 2 + 3) % (10 * 2))
        self.end()
        self.begin(GL_TRIANGLE_FAN)
        self.normal3f(0, 0, -1)
        for part in range(10):
            angle = np.radians(-part / 10.0 * 360)
            self.vertex3f(x + baseradius * np.cos(angle), y + baseradius * np.sin(angle), z - scale / 2)
        self.vertex3f(x, y, z - scale / 2)
        for part in range(parts + 1):
            self.tri(parts, part, (part + 1) % parts)
        self.end()
        a = headradius - baseradius
        b = headfraction * scale
        headangle = np.arctan(a / b)
        for part in range(10 + 1):
            self.begin(GL_TRIANGLES)
            angle = np.radians(-part / 10.0 * 360)
            anglemid = np.radians(-(part + 0.5) / 10.0 * 360)
            angle2 = np.radians(-(part + 1) / 10.0 * 360)
            self.normal3f(np.cos(anglemid) * np.cos(headangle), np.sin(anglemid) * np.cos(headangle), np.sin(headangle))
            self.vertex3f(x, y, z + scale / 2)
            self.normal3f(np.cos(angle2) * np.cos(headangle), np.sin(angle2) * np.cos(headangle), np.sin(headangle))
            self.vertex3f(x + headradius * np.cos(angle2), y + headradius * np.sin(angle2), z + scale / 2 - headfraction * scale)
            self.normal3f(np.cos(angle) * np.cos(headangle), np.sin(angle) * np.cos(headangle), np.sin(headangle))
            self.vertex3f(x + headradius * np.cos(angle), y + headradius * np.sin(angle), z + scale / 2 - headfraction * scale)
            self.tri(0, 1, 2)
            self.end()
        self.begin(GL_QUADS)
        self.normal3f(0, 0, -1)
        for part in range(10):
            angle = np.radians(-part / 10.0 * 360)
            angle2 = np.radians(-(part + 1) / 10.0 * 360)
            self.vertex3f(x + baseradius * np.cos(angle), y + baseradius * np.sin(angle), z + scale / 2 - headfraction * scale)
            self.vertex3f(x + headradius * np.cos(angle), y + headradius * np.sin(angle), z + scale / 2 - headfraction * scale)
        for part in range(10 + 1):
            self.tri((part * 2 + 0) % (10 * 2), (part * 2 + 1) % (10 * 2), (part * 2 + 2) % (10 * 2))
            self.tri((part * 2 + 2) % (10 * 2), (part * 2 + 1) % (10 * 2), (part * 2 + 3) % (10 * 2))
        self.end()
        self.vertices = np.array(self.vertices, dtype=np.float32)
        self.normals = np.array(self.normals, dtype=np.float32)
        self.indices = np.array(self.indices, dtype=np.uint32)
if __name__ == '__main__':
    colormaps = vaex.ui.colormaps.colormaps
    import json
    js = json.dumps(vaex.ui.colormaps.colormaps)
    print(js)
    app = QtGui.QApplication(sys.argv)
    widget = TestWidget(None)

    def load():
        if False:
            i = 10
            return i + 15
        widget.main.loadTable(sys.argv[1], sys.argv[2:])
    widget.main.post_init = load
    sys.exit(app.exec_())