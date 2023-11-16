"""
GPU-based ray tracing example.

GLSL port of the following Python example:
    https://gist.github.com/rossant/6046463
    https://pbs.twimg.com/media/BPpbJTiCIAEoEPl.png

TODO:
    * Once uniform structs are supported, refactor the code to encapsulate
      objects (spheres, planes, lights) in structures.
    * Customizable engine with an arbitrary number of objects.
"""
from math import cos
from vispy import app, gloo
vertex = '\n#version 120\n\nattribute vec2 a_position;\nvarying vec2 v_position;\nvoid main()\n{\n    gl_Position = vec4(a_position, 0.0, 1.0);\n    v_position = a_position;\n}\n'
fragment = '\n#version 120\n\nconst float M_PI = 3.14159265358979323846;\nconst float INFINITY = 1000000000.;\nconst int PLANE = 1;\nconst int SPHERE_0 = 2;\nconst int SPHERE_1 = 3;\n\nuniform float u_aspect_ratio;\nvarying vec2 v_position;\n\nuniform vec3 sphere_position_0;\nuniform float sphere_radius_0;\nuniform vec3 sphere_color_0;\n\nuniform vec3 sphere_position_1;\nuniform float sphere_radius_1;\nuniform vec3 sphere_color_1;\n\nuniform vec3 plane_position;\nuniform vec3 plane_normal;\n\nuniform float light_intensity;\nuniform vec2 light_specular;\nuniform vec3 light_position;\nuniform vec3 light_color;\n\nuniform float ambient;\nuniform vec3 O;\n\nfloat intersect_sphere(vec3 O, vec3 D, vec3 S, float R) {\n    float a = dot(D, D);\n    vec3 OS = O - S;\n    float b = 2. * dot(D, OS);\n    float c = dot(OS, OS) - R * R;\n    float disc = b * b - 4. * a * c;\n    if (disc > 0.) {\n        float distSqrt = sqrt(disc);\n        float q = (-b - distSqrt) / 2.0;\n        if (b >= 0.) {\n            q = (-b + distSqrt) / 2.0;\n        }\n        float t0 = q / a;\n        float t1 = c / q;\n        t0 = min(t0, t1);\n        t1 = max(t0, t1);\n        if (t1 >= 0.) {\n            if (t0 < 0.) {\n                return t1;\n            }\n            else {\n                return t0;\n            }\n        }\n    }\n    return INFINITY;\n}\n\nfloat intersect_plane(vec3 O, vec3 D, vec3 P, vec3 N) {\n    float denom = dot(D, N);\n    if (abs(denom) < 1e-6) {\n        return INFINITY;\n    }\n    float d = dot(P - O, N) / denom;\n    if (d < 0.) {\n        return INFINITY;\n    }\n    return d;\n}\n\nvec3 run(float x, float y) {\n    vec3 Q = vec3(x, y, 0.);\n    vec3 D = normalize(Q - O);\n    int depth = 0;\n    float t_plane, t0, t1;\n    vec3 rayO = O;\n    vec3 rayD = D;\n    vec3 col = vec3(0.0, 0.0, 0.0);\n    vec3 col_ray;\n    float reflection = 1.;\n\n    int object_index;\n    vec3 object_color;\n    vec3 object_normal;\n    float object_reflection;\n    vec3 M;\n    vec3 N, toL, toO;\n\n    while (depth < 5) {\n\n        /* start trace_ray */\n\n        t_plane = intersect_plane(rayO, rayD, plane_position, plane_normal);\n        t0 = intersect_sphere(rayO, rayD, sphere_position_0, sphere_radius_0);\n        t1 = intersect_sphere(rayO, rayD, sphere_position_1, sphere_radius_1);\n\n        if (t_plane < min(t0, t1)) {\n            // Plane.\n            M = rayO + rayD * t_plane;\n            object_normal = plane_normal;\n            // Plane texture.\n            if (mod(int(2*M.x), 2) == mod(int(2*M.z), 2)) {\n                object_color = vec3(1., 1., 1.);\n            }\n            else {\n                object_color = vec3(0., 0., 0.);\n            }\n            object_reflection = .25;\n            object_index = PLANE;\n        }\n        else if (t0 < t1) {\n            // Sphere 0.\n            M = rayO + rayD * t0;\n            object_normal = normalize(M - sphere_position_0);\n            object_color = sphere_color_0;\n            object_reflection = .5;\n            object_index = SPHERE_0;\n        }\n        else if (t1 < t0) {\n            // Sphere 1.\n            M = rayO + rayD * t1;\n            object_normal = normalize(M - sphere_position_1);\n            object_color = sphere_color_1;\n            object_reflection = .5;\n            object_index = SPHERE_1;\n        }\n        else {\n            break;\n        }\n\n        N = object_normal;\n        toL = normalize(light_position - M);\n        toO = normalize(O - M);\n\n        // Shadow of the spheres on the plane.\n        if (object_index == PLANE) {\n            t0 = intersect_sphere(M + N * .0001, toL,\n                                  sphere_position_0, sphere_radius_0);\n            t1 = intersect_sphere(M + N * .0001, toL,\n                                  sphere_position_1, sphere_radius_1);\n            if (min(t0, t1) < INFINITY) {\n                break;\n            }\n        }\n\n        col_ray = vec3(ambient, ambient, ambient);\n        col_ray += light_intensity * max(dot(N, toL), 0.) * object_color;\n        col_ray += light_specular.x * light_color *\n            pow(max(dot(N, normalize(toL + toO)), 0.), light_specular.y);\n\n        /* end trace_ray */\n\n        rayO = M + N * .0001;\n        rayD = normalize(rayD - 2. * dot(rayD, N) * N);\n        col += reflection * col_ray;\n        reflection *= object_reflection;\n\n        depth++;\n    }\n\n    return clamp(col, 0., 1.);\n}\n\nvoid main() {\n    vec2 pos = v_position;\n    gl_FragColor = vec4(run(pos.x*u_aspect_ratio, pos.y), 1.);\n}\n'

class Canvas(app.Canvas):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        app.Canvas.__init__(self, position=(300, 100), size=(800, 600), keys='interactive')
        self.program = gloo.Program(vertex, fragment)
        self.program['a_position'] = [(-1.0, -1.0), (-1.0, +1.0), (+1.0, -1.0), (+1.0, +1.0)]
        self.program['sphere_position_0'] = (0.75, 0.1, 1.0)
        self.program['sphere_radius_0'] = 0.6
        self.program['sphere_color_0'] = (0.0, 0.0, 1.0)
        self.program['sphere_position_1'] = (-0.75, 0.1, 2.25)
        self.program['sphere_radius_1'] = 0.6
        self.program['sphere_color_1'] = (0.5, 0.223, 0.5)
        self.program['plane_position'] = (0.0, -0.5, 0.0)
        self.program['plane_normal'] = (0.0, 1.0, 0.0)
        self.program['light_intensity'] = 1.0
        self.program['light_specular'] = (1.0, 50.0)
        self.program['light_position'] = (5.0, 5.0, -10.0)
        self.program['light_color'] = (1.0, 1.0, 1.0)
        self.program['ambient'] = 0.05
        self.program['O'] = (0.0, 0.0, -1.0)
        self.activate_zoom()
        self._timer = app.Timer('auto', connect=self.on_timer, start=True)
        self.show()

    def on_timer(self, event):
        if False:
            for i in range(10):
                print('nop')
        t = event.elapsed
        self.program['sphere_position_0'] = (+0.75, 0.1, 2.0 + 1.0 * cos(4 * t))
        self.program['sphere_position_1'] = (-0.75, 0.1, 2.0 - 1.0 * cos(4 * t))
        self.update()

    def on_resize(self, event):
        if False:
            for i in range(10):
                print('nop')
        self.activate_zoom()

    def activate_zoom(self):
        if False:
            i = 10
            return i + 15
        (width, height) = self.size
        gloo.set_viewport(0, 0, *self.physical_size)
        self.program['u_aspect_ratio'] = width / float(height)

    def on_draw(self, event):
        if False:
            while True:
                i = 10
        self.program.draw('triangle_strip')
if __name__ == '__main__':
    canvas = Canvas()
    app.run()