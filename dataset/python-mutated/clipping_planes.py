"""
Clipping planes with volume and markers
=======================================
Controls:
- x/y/z/o - add new clipping plane with normal along x/y/z or [1,1,1] oblique axis
- r - remove a clipping plane
"""
import numpy as np
from vispy import app, scene, io
from vispy.visuals.filters.clipping_planes import PlanesClipper
canvas = scene.SceneCanvas(keys='interactive', size=(800, 600), show=True)
view = canvas.central_widget.add_view()
vol = np.load(io.load_data_file('volume/stent.npz'))['arr_0']
volume = scene.visuals.Volume(vol, parent=view.scene, threshold=0.225)
np.random.seed(1)
points = np.random.rand(100, 3) * (128, 128, 128)
markers = scene.visuals.Markers(pos=points, parent=view.scene)
markers.transform = scene.STTransform(translate=(0, 0, 128))
clipper = PlanesClipper()
markers.attach(clipper)
fov = 60.0
cam = scene.cameras.TurntableCamera(parent=view.scene, fov=fov, name='Turntable')
view.camera = cam
volume_center = (np.array(vol.shape) / 2)[::-1]
clip_modes = {'x': np.array([[volume_center, [1, 0, 0]]]), 'y': np.array([[volume_center, [0, 1, 0]]]), 'z': np.array([[volume_center, [0, 0, 1]]]), 'o': np.array([[volume_center, [1, 1, 1]]])}

def add_clip(mode):
    if False:
        while True:
            i = 10
    if mode not in clip_modes:
        return
    clipping_planes = np.concatenate([volume.clipping_planes, clip_modes[mode]])
    volume.clipping_planes = clipping_planes
    clipper.clipping_planes = clipping_planes

def remove_clip():
    if False:
        i = 10
        return i + 15
    if volume.clipping_planes.shape[0] > 0:
        volume.clipping_planes = volume.clipping_planes[:-1]
        clipper.clipping_planes = clipper.clipping_planes[:-1]

@canvas.events.key_press.connect
def on_key_press(event):
    if False:
        print('Hello World!')
    if event.text in 'xyzo':
        add_clip(event.text)
    elif event.text == 'r':
        remove_clip()
if __name__ == '__main__':
    print(__doc__)
    app.run()