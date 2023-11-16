from kivy.tools.gallery import parse_docstring_info

def test_parse_docstring_info():
    if False:
        while True:
            i = 10
    assert 'error' in parse_docstring_info('No Docstring')
    assert 'error' in parse_docstring_info("'''No Docstring Title'''")
    assert 'error' in parse_docstring_info("'''No Sentence\n======\nPeriods'''")
    assert 'error' in parse_docstring_info("'\nSingle Quotes\n===\n\nNo singles.'")
    d = parse_docstring_info("'''\n3D Rendering Monkey Head\n========================\n\nThis example demonstrates using OpenGL to display a\nrotating monkey head. This\nincludes loading a Blender OBJ file, shaders written in OpenGL's Shading\nLanguage (GLSL), and using scheduled callbacks.\n\nThe file monkey.obj is a OBJ file output form the Blender free 3D creation\nsoftware. The file is text, listing vertices and faces. It is loaded\ninto a scene using objloader.py's ObjFile class. The file simple.glsl is\na simple vertex and fragment shader written in GLSL.\n'''\nblah blah\nblah blah\n")
    assert 'error' not in d
    assert '3D Rendering' in d['docstring'] and 'This example' in d['docstring']
    assert '3D Rendering' in d['title']
    assert 'monkey head' in d['first_sentence']
if __name__ == '__main__':
    test_parse_docstring_info()