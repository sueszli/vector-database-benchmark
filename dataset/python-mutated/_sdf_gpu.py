"""
Jump flooding algoritm for EDT using GLSL code:
Author: Stefan Gustavson (stefan.gustavson@gmail.com)
2010-08-24. This code is in the public domain.

Adapted to `vispy` by Eric Larson <larson.eric.d@gmail.com>.
"""
import numpy as np
from ...gloo import Program, FrameBuffer, VertexBuffer, Texture2D, set_viewport, set_state
vert_seed = '\nattribute vec2 a_position;\nattribute vec2 a_texcoord;\nvarying vec2 v_uv;\n\nvoid main( void )\n{\n  v_uv = a_texcoord.xy;\n  gl_Position = vec4(a_position.xy, 0., 1.);\n}\n'
vert = '\nuniform float u_texw;\nuniform float u_texh;\nuniform float u_step;\nattribute vec2 a_position;\nattribute vec2 a_texcoord;\nvarying float v_stepu;\nvarying float v_stepv;\nvarying vec2 v_uv;\n\nvoid main( void )\n{\n  v_uv = a_texcoord.xy;\n  v_stepu = u_step / u_texw; // Saves a division in the fragment shader\n  v_stepv = u_step / u_texh;\n  gl_Position = vec4(a_position.xy, 0., 1.);\n}\n'
frag_seed = '\nuniform sampler2D u_texture;\nvarying vec2 v_uv;\n\nvoid main( void )\n{\n  float pixel = texture2D(u_texture, v_uv).r;\n  vec4 myzero = vec4(128. / 255., 128. / 255., 0., 0.);  // Zero\n  vec4 myinfinity = vec4(0., 0., 0., 0.);                // Infinity\n  // Pixels >= 0.5 are objects, others are background\n  gl_FragColor = pixel >= 0.5 ? myzero : myinfinity;\n}\n'
frag_flood = '\nuniform sampler2D u_texture;\nvarying float v_stepu;\nvarying float v_stepv;\nvarying vec2 v_uv;\n\nvec2 remap(vec4 floatdata) {\n    vec2 scaleddata = vec2(floatdata.x * 65280. + floatdata.z * 255.,\n                           floatdata.y * 65280. + floatdata.w * 255.);\n    return scaleddata / 32768. - 1.0;\n}\n\nvec4 remap_inv(vec2 floatvec) {\n    vec2 data = (floatvec + 1.0) * 32768.;\n    float x = floor(data.x / 256.);\n    float y = floor(data.y / 256.);\n    return vec4(x, y, data.x - x * 256., data.y - y * 256.) / 255.;\n}\n\nvoid main( void )\n{\n  // Search for better distance vectors among 8 candidates\n  vec2 stepvec; // Relative offset to candidate being tested\n  vec2 newvec;  // Absolute position of that candidate\n  vec3 newseed; // Closest point from that candidate (.xy) and its dist (.z)\n  vec3 bestseed; // Closest seed so far\n  bestseed.xy = remap(texture2D(u_texture, v_uv).rgba);\n  bestseed.z = length(bestseed.xy);\n\n  stepvec = vec2(-v_stepu, -v_stepv);\n  newvec = v_uv + stepvec;\n  if (all(bvec4(lessThan(newvec, vec2(1.0)), greaterThan(newvec, vec2(0.0))))){\n    newseed.xy = remap(texture2D(u_texture, newvec).rgba);\n    if(newseed.x > -0.99999) { // if the new seed is not "indeterminate dist"\n      newseed.xy = newseed.xy + stepvec;\n      newseed.z = length(newseed.xy);\n      if(newseed.z < bestseed.z) {\n        bestseed = newseed;\n      }\n    }\n  }\n\n  stepvec = vec2(-v_stepu, 0.0);\n  newvec = v_uv + stepvec;\n  if (all(bvec4(lessThan(newvec, vec2(1.0)), greaterThan(newvec, vec2(0.0))))){\n    newseed.xy = remap(texture2D(u_texture, newvec).rgba);\n    if(newseed.x > -0.99999) { // if the new seed is not "indeterminate dist"\n      newseed.xy = newseed.xy + stepvec;\n      newseed.z = length(newseed.xy);\n      if(newseed.z < bestseed.z) {\n        bestseed = newseed;\n      }\n    }\n  }\n\n  stepvec = vec2(-v_stepu, v_stepv);\n  newvec = v_uv + stepvec;\n  if (all(bvec4(lessThan(newvec, vec2(1.0)), greaterThan(newvec, vec2(0.0))))){\n    newseed.xy = remap(texture2D(u_texture, newvec).rgba);\n    if(newseed.x > -0.99999) { // if the new seed is not "indeterminate dist"\n      newseed.xy = newseed.xy + stepvec;\n      newseed.z = length(newseed.xy);\n      if(newseed.z < bestseed.z) {\n        bestseed = newseed;\n      }\n    }\n  }\n\n  stepvec = vec2(0.0, -v_stepv);\n  newvec = v_uv + stepvec;\n  if (all(bvec4(lessThan(newvec, vec2(1.0)), greaterThan(newvec, vec2(0.0))))){\n    newseed.xy = remap(texture2D(u_texture, newvec).rgba);\n    if(newseed.x > -0.99999) { // if the new seed is not "indeterminate dist"\n      newseed.xy = newseed.xy + stepvec;\n      newseed.z = length(newseed.xy);\n      if(newseed.z < bestseed.z) {\n        bestseed = newseed;\n      }\n    }\n  }\n\n  stepvec = vec2(0.0, v_stepv);\n  newvec = v_uv + stepvec;\n  if (all(bvec4(lessThan(newvec, vec2(1.0)), greaterThan(newvec, vec2(0.0))))){\n    newseed.xy = remap(texture2D(u_texture, newvec).rgba);\n    if(newseed.x > -0.99999) { // if the new seed is not "indeterminate dist"\n      newseed.xy = newseed.xy + stepvec;\n      newseed.z = length(newseed.xy);\n      if(newseed.z < bestseed.z) {\n        bestseed = newseed;\n      }\n    }\n  }\n\n  stepvec = vec2(v_stepu, -v_stepv);\n  newvec = v_uv + stepvec;\n  if (all(bvec4(lessThan(newvec, vec2(1.0)), greaterThan(newvec, vec2(0.0))))){\n    newseed.xy = remap(texture2D(u_texture, newvec).rgba);\n    if(newseed.x > -0.99999) { // if the new seed is not "indeterminate dist"\n      newseed.xy = newseed.xy + stepvec;\n      newseed.z = length(newseed.xy);\n      if(newseed.z < bestseed.z) {\n        bestseed = newseed;\n      }\n    }\n  }\n\n  stepvec = vec2(v_stepu, 0.0);\n  newvec = v_uv + stepvec;\n  if (all(bvec4(lessThan(newvec, vec2(1.0)), greaterThan(newvec, vec2(0.0))))){\n    newseed.xy = remap(texture2D(u_texture, newvec).rgba);\n    if(newseed.x > -0.99999) { // if the new seed is not "indeterminate dist"\n      newseed.xy = newseed.xy + stepvec;\n      newseed.z = length(newseed.xy);\n      if(newseed.z < bestseed.z) {\n        bestseed = newseed;\n      }\n    }\n  }\n\n  stepvec = vec2(v_stepu, v_stepv);\n  newvec = v_uv + stepvec;\n  if (all(bvec4(lessThan(newvec, vec2(1.0)), greaterThan(newvec, vec2(0.0))))){\n    newseed.xy = remap(texture2D(u_texture, newvec).rgba);\n    if(newseed.x > -0.99999) { // if the new seed is not "indeterminate dist"\n      newseed.xy = newseed.xy + stepvec;\n      newseed.z = length(newseed.xy);\n      if(newseed.z < bestseed.z) {\n        bestseed = newseed;\n      }\n    }\n  }\n\n  gl_FragColor = remap_inv(bestseed.xy);\n}\n'
frag_insert = '\n\nuniform sampler2D u_texture;\nuniform sampler2D u_pos_texture;\nuniform sampler2D u_neg_texture;\nvarying float v_stepu;\nvarying float v_stepv;\nvarying vec2 v_uv;\n\nvec2 remap(vec4 floatdata) {\n    vec2 scaled_data = vec2(floatdata.x * 65280. + floatdata.z * 255.,\n                            floatdata.y * 65280. + floatdata.w * 255.);\n    return scaled_data / 32768. - 1.0;\n}\n\nvoid main( void )\n{\n    float pixel = texture2D(u_texture, v_uv).r;\n    // convert distance from normalized units -> pixels\n    vec2 rescale = vec2(v_stepu, v_stepv);\n    float shrink = 8.;\n    rescale = rescale * 256. / shrink;\n    // Without the division, 1 RGB increment = 1 px distance\n    vec2 pos_distvec = remap(texture2D(u_pos_texture, v_uv).rgba) / rescale;\n    vec2 neg_distvec = remap(texture2D(u_neg_texture, v_uv).rgba) / rescale;\n    if (pixel <= 0.5)\n        gl_FragColor = vec4(0.5 - length(pos_distvec));\n    else\n        gl_FragColor = vec4(0.5 - (shrink - 1.) / 256. + length(neg_distvec));\n}\n'

class SDFRendererGPU(object):

    def __init__(self):
        if False:
            return 10
        self.program_seed = Program(vert_seed, frag_seed)
        self.program_flood = Program(vert, frag_flood)
        self.program_insert = Program(vert, frag_insert)
        self.programs = [self.program_seed, self.program_flood, self.program_insert]
        self.fbo_to = [FrameBuffer(), FrameBuffer(), FrameBuffer()]
        vtype = np.dtype([('a_position', np.float32, 2), ('a_texcoord', np.float32, 2)])
        vertices = np.zeros(4, dtype=vtype)
        vertices['a_position'] = [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]
        vertices['a_texcoord'] = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
        vertices = VertexBuffer(vertices)
        self.program_insert['u_step'] = 1.0
        for program in self.programs:
            program.bind(vertices)

    def render_to_texture(self, data, texture, offset, size):
        if False:
            print('Hello World!')
        'Render a SDF to a texture at a given offset and size\n\n        Parameters\n        ----------\n        data : array\n            Must be 2D with type np.ubyte.\n        texture : instance of Texture2D\n            The texture to render to.\n        offset : tuple of int\n            Offset (x, y) to render to inside the texture.\n        size : tuple of int\n            Size (w, h) to render inside the texture.\n        '
        assert isinstance(texture, Texture2D)
        set_state(blend=False, depth_test=False)
        orig_tex = Texture2D(255 - data, format='luminance', wrapping='clamp_to_edge', interpolation='nearest')
        edf_neg_tex = self._render_edf(orig_tex)
        orig_tex[:, :, 0] = data
        edf_pos_tex = self._render_edf(orig_tex)
        self.program_insert['u_texture'] = orig_tex
        self.program_insert['u_pos_texture'] = edf_pos_tex
        self.program_insert['u_neg_texture'] = edf_neg_tex
        self.fbo_to[-1].color_buffer = texture
        with self.fbo_to[-1]:
            set_viewport(tuple(offset) + tuple(size))
            self.program_insert.draw('triangle_strip')

    def _render_edf(self, orig_tex):
        if False:
            print('Hello World!')
        'Render an EDF to a texture'
        sdf_size = orig_tex.shape[:2]
        comp_texs = []
        for _ in range(2):
            tex = Texture2D(sdf_size + (4,), format='rgba', interpolation='nearest', wrapping='clamp_to_edge')
            comp_texs.append(tex)
        self.fbo_to[0].color_buffer = comp_texs[0]
        self.fbo_to[1].color_buffer = comp_texs[1]
        for program in self.programs[1:]:
            (program['u_texh'], program['u_texw']) = sdf_size
        last_rend = 0
        with self.fbo_to[last_rend]:
            set_viewport(0, 0, sdf_size[1], sdf_size[0])
            self.program_seed['u_texture'] = orig_tex
            self.program_seed.draw('triangle_strip')
        stepsize = (np.array(sdf_size) // 2).max()
        while stepsize > 0:
            self.program_flood['u_step'] = stepsize
            self.program_flood['u_texture'] = comp_texs[last_rend]
            last_rend = 1 if last_rend == 0 else 0
            with self.fbo_to[last_rend]:
                set_viewport(0, 0, sdf_size[1], sdf_size[0])
                self.program_flood.draw('triangle_strip')
            stepsize //= 2
        return comp_texs[last_rend]