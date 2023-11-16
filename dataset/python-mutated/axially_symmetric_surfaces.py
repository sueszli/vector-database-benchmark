"""
   * Demonstrates the usage of SurfacePlot() using meshes created in spherical
     and cylindrical coordinates to generate surfaces with rotational symmetry.
   * Meshes can be generated in xy, yz or zx axes. See create_cylinder().
   * Also this example demonstrates the creation of surfaces with discontinuity
     See create_circular_hole()
"""
import sys
import numpy as np
from vispy import scene
from vispy.scene.visuals import SurfacePlot
from vispy.scene.visuals import XYZAxis

def create_cylinder(radius, length, center=(0.0, 0.0, 0.0)):
    if False:
        return 10
    ' Creates the data of a cylinder oriented along z axis whose center, radius and length are given as inputs\n        Based on the example given at: https://stackoverflow.com/a/49311446/2602319\n    '
    z = np.linspace(0, length, 100)
    theta = np.linspace(0, 2 * np.pi, 100)
    (theta_grid, z_grid) = np.meshgrid(theta, z)
    x_grid = radius * np.cos(theta_grid) + center[0]
    y_grid = radius * np.sin(theta_grid) + center[1]
    z_grid = z_grid + center[2]
    return (x_grid, y_grid, z_grid)

def create_paraboloid(a, b, c, radius=1.0, center=(0.0, 0.0, 0.0)):
    if False:
        return 10
    '\n    Creates the data of a paraboloid whose center, radius and a, b, c parameters are given as inputs\n    '
    r = np.linspace(0, radius, 100)
    theta = np.linspace(0, 2 * np.pi, 100)
    (R, THETA) = np.meshgrid(r, theta)
    (x_grid, y_grid) = (R * np.cos(THETA), R * np.sin(THETA))
    z_grid = c * ((x_grid / a) ** 2 + (y_grid / b) ** 2)
    return (x_grid + center[0], y_grid + center[1], z_grid + center[2])

def create_sphere(radius=1.0, center=(0.0, 0.0, 0.0)):
    if False:
        while True:
            i = 10
    '\n    Creates the data of a sphere whose center, and radius are given as inputs\n    '
    theta = np.linspace(0, np.pi, 50)
    phi = np.linspace(0, 2 * np.pi, 50)
    (PHI, THETA) = np.meshgrid(phi, theta)
    RHO = radius
    x_grid = RHO * np.sin(THETA) * np.cos(PHI) + center[0]
    y_grid = RHO * np.sin(THETA) * np.sin(PHI) + center[1]
    z_grid = RHO * np.cos(THETA) + center[2]
    return (x_grid, y_grid, z_grid)

def create_circular_hole(x_grid, y_grid, hole_radius=0.5, center=(0.0, 0.0)):
    if False:
        print('Hello World!')
    X = np.where((x_grid - center[0]) ** 2 + (y_grid - center[1]) ** 2 <= hole_radius ** 2, np.NAN, x_grid)
    Y = np.where((x_grid - center[0]) ** 2 + (y_grid - center[1]) ** 2 <= hole_radius ** 2, np.NAN, y_grid)
    return (X, Y)
canvas = scene.SceneCanvas(keys='interactive', size=(800, 600), show=True)
view = canvas.central_widget.add_view()
camera = scene.cameras.TurntableCamera(fov=60, elevation=30, azimuth=50, parent=view.scene)
camera.set_range((-4, 4), (-4, 4), (-4, 4))
view.camera = camera
XYZAxis(parent=view.scene)
(x_grid, y_grid, z_grid) = create_paraboloid(6, 3, 3, radius=2, center=(0, 0, 3))
(x_grid, y_grid) = create_circular_hole(x_grid, y_grid, hole_radius=0.5, center=(0.0, 0.0))
SurfacePlot(x_grid, y_grid, z_grid, color='aquamarine', parent=view.scene)
(x_grid, y_grid, z_grid) = create_cylinder(1, 3, center=(0, 0, 0))
SurfacePlot(x_grid, y_grid, z_grid, color='lightgreen', parent=view.scene)
(x_grid, y_grid, z_grid) = create_sphere(0.5, center=(0, 0, 4))
SurfacePlot(x_grid, y_grid, z_grid, color='lightseagreen', parent=view.scene)
if __name__ == '__main__' and sys.flags.interactive == 0:
    canvas.app.run()