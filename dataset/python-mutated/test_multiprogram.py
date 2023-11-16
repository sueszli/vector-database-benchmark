from vispy.visuals.shaders import MultiProgram, Function, StatementList
from vispy.visuals.transforms import STTransform, MatrixTransform

def test_multiprogram():
    if False:
        print('Hello World!')
    vert = '\n    uniform float u_scale;\n    \n    void main() {\n        gl_Position = $transform(vec4(0, 0, 0, 0));\n    }\n    '
    frag = '\n    void main() {\n        gl_FragColor = $color;\n        $post_hook\n    }\n    '
    mp = MultiProgram(vert, frag)
    p1 = mp.add_program()
    p2 = mp.add_program('p2')
    assert 'p2' in mp._programs
    mp.add_program('junk')
    assert 'junk' not in mp._programs and len(mp._programs) == 2
    mp['u_scale'] = 2
    assert p1['u_scale'] == 2
    assert p2['u_scale'] == 2
    p1['u_scale'] = 3
    assert p1['u_scale'] == 3
    assert p2['u_scale'] == 2
    mp.frag['color'] = (1, 1, 1, 1)
    assert p1.frag['color'].value == (1, 1, 1, 1)
    assert p2.frag['color'].value == (1, 1, 1, 1)
    func = Function('\n    void filter() {\n        gl_FragColor.r = 0.5;\n    }\n    ')
    p1.frag['post_hook'] = StatementList()
    p2.frag['post_hook'] = StatementList()
    p2.frag['post_hook'].add(func())
    tr1 = STTransform()
    tr2 = MatrixTransform()
    p1.vert['transform'] = tr1
    p2.vert['transform'] = tr2
    assert 'st_transform_map' in p1.vert.compile()
    assert 'affine_transform_map' in p2.vert.compile()
    assert 'filter' not in p1.frag.compile()
    assert 'filter' in p2.frag.compile()
    mp.vert = vert + '\n//test\n'
    mp.vert['transform'] = tr1
    assert '//test' in p1.vert.compile()
    p3 = mp.add_program()
    assert p3['u_scale'] == 2
    assert p3.frag['color'].value == (1, 1, 1, 1)
    assert '//test' in p3.vert.compile()
    assert 'st_transform_map' in p3.vert.compile()