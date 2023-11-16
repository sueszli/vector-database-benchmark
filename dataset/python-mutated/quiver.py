from vispy import app, gloo
vertex = '\nattribute vec2 position;\nvoid main()\n{\n    gl_Position = vec4(position, 0.0, 1.0);\n}\n'
fragment = '\n#include "math/constants.glsl"\n#include "arrows/arrows.glsl"\n#include "antialias/antialias.glsl"\n\nuniform vec2 iResolution;\nuniform vec2 iMouse;\nvoid main()\n{\n    const float M_PI = 3.14159265358979323846;\n    const float SQRT_2 = 1.4142135623730951;\n    const float linewidth = 3.0;\n    const float antialias =  1.0;\n    const float rows = 32.0;\n    const float cols = 32.0;\n\n    float body = min(iResolution.x/cols, iResolution.y/rows) / SQRT_2;\n    vec2 texcoord = gl_FragCoord.xy;\n    vec2 size   = iResolution.xy / vec2(cols,rows);\n    vec2 center = (floor(texcoord/size) + vec2(0.5,0.5)) * size;\n    texcoord -= center;\n    float theta = M_PI-atan(center.y-iMouse.y,  center.x-iMouse.x);\n    float cos_theta = cos(theta);\n    float sin_theta = sin(theta);\n\n\n    texcoord = vec2(cos_theta*texcoord.x - sin_theta*texcoord.y,\n                    sin_theta*texcoord.x + cos_theta*texcoord.y);\n\n    float d = arrow_stealth(texcoord, body, 0.25*body, linewidth, antialias);\n    gl_FragColor = filled(d, linewidth, antialias, vec4(0,0,0,1));\n}\n'
canvas = app.Canvas(size=(2 * 512, 2 * 512), keys='interactive')
canvas.context.set_state(blend=True, blend_func=('src_alpha', 'one_minus_src_alpha'), blend_equation='func_add')

@canvas.connect
def on_draw(event):
    if False:
        i = 10
        return i + 15
    gloo.clear('white')
    program.draw('triangle_strip')

@canvas.connect
def on_resize(event):
    if False:
        for i in range(10):
            print('nop')
    program['iResolution'] = event.size
    gloo.set_viewport(0, 0, event.size[0], event.size[1])

@canvas.connect
def on_mouse_move(event):
    if False:
        while True:
            i = 10
    (x, y) = event.pos
    program['iMouse'] = (x, canvas.size[1] - y)
    canvas.update()
program = gloo.Program(vertex, fragment, count=4)
(dx, dy) = (1, 1)
program['position'] = ((-dx, -dy), (-dx, +dy), (+dx, -dy), (+dx, +dy))
program['iResolution'] = (2 * 512, 2 * 512)
program['iMouse'] = (0.0, 0.0)
if __name__ == '__main__':
    canvas.show()
    app.run()