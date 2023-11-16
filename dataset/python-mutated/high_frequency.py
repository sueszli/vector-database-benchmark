from vispy import gloo, app, keys
VERT_SHADER = '\nattribute vec2 a_position;\nvoid main (void)\n{\n    gl_Position = vec4(a_position, 0.0, 1.0);\n}\n'
FRAG_SHADER = '\nuniform vec2 u_resolution;\nuniform float u_global_time;\n\n// --- Your function here ---\nfloat function( float x )\n{\n    float d = 3.0 - 2.0*(1.0+cos(u_global_time/5.0))/2.0;\n    return sin(pow(x,d))*sin(x);\n}\n// --- Your function here ---\n\n\nfloat sample(vec2 uv)\n{\n    const int samples = 128;\n    const float fsamples = float(samples);\n    vec2 maxdist = vec2(0.5,1.0)/40.0;\n    vec2 halfmaxdist = vec2(0.5) * maxdist;\n\n    float stepsize = maxdist.x / fsamples;\n    float initial_offset_x = -0.5 * fsamples * stepsize;\n    uv.x += initial_offset_x;\n    float hit = 0.0;\n    for( int i=0; i<samples; ++i )\n    {\n        float x = uv.x + stepsize * float(i);\n        float y = uv.y;\n        float fx = function(x);\n        float dist = abs(y-fx);\n        hit += step(dist, halfmaxdist.y);\n    }\n    const float arbitraryFactor = 4.5;\n    const float arbitraryExp = 0.95;\n    return arbitraryFactor * pow( hit / fsamples, arbitraryExp );\n}\n\nvoid main(void)\n{\n    vec2 uv = gl_FragCoord.xy / u_resolution.xy;\n    float ymin = -2.0;\n    float ymax = +2.0;\n    float xmin = 0.0;\n    float xmax = xmin + (ymax-ymin)* u_resolution.x / u_resolution.y;\n\n    vec2 xy = vec2(xmin,ymin) + uv*vec2(xmax-xmin, ymax-ymin);\n    gl_FragColor = vec4(0.0,0.0,0.0, sample(xy));\n}\n'

class Canvas(app.Canvas):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        app.Canvas.__init__(self, size=(800, 600), keys='interactive')
        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self.program['u_global_time'] = 0
        self.program['a_position'] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
        self.apply_zoom()
        gloo.set_state(blend=True, blend_func=('src_alpha', 'one_minus_src_alpha'))
        self._timer = app.Timer('auto', connect=self.on_timer_event, start=True)
        self.show()

    def on_resize(self, event):
        if False:
            print('Hello World!')
        self.apply_zoom()

    def on_draw(self, event):
        if False:
            for i in range(10):
                print('nop')
        gloo.clear('white')
        self.program.draw(mode='triangle_strip')

    def on_timer_event(self, event):
        if False:
            print('Hello World!')
        if self._timer.running:
            self.program['u_global_time'] += event.dt
        self.update()

    def on_key_press(self, event):
        if False:
            print('Hello World!')
        if event.key is keys.SPACE:
            if self._timer.running:
                self._timer.stop()
            else:
                self._timer.start()

    def apply_zoom(self):
        if False:
            return 10
        self.program['u_resolution'] = self.physical_size
        gloo.set_viewport(0, 0, *self.physical_size)
if __name__ == '__main__':
    c = Canvas()
    app.run()