from vispy.visuals.mesh import MeshVisual
from vispy.visuals.filters import WireframeFilter

def test_empty_mesh_wireframe():
    if False:
        for i in range(10):
            print('nop')
    'Test that an empty mesh does not cause issues with wireframe filter'
    mesh = MeshVisual()
    wf = WireframeFilter()
    mesh.attach(wf)
    wf.enabled = True