import os
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def custom_draw_geometry_with_camera_trajectory(pcd, camera_trajectory_path, render_option_path, output_path):
    if False:
        while True:
            i = 10
    custom_draw_geometry_with_camera_trajectory.index = -1
    custom_draw_geometry_with_camera_trajectory.trajectory = o3d.io.read_pinhole_camera_trajectory(camera_trajectory_path)
    custom_draw_geometry_with_camera_trajectory.vis = o3d.visualization.Visualizer()
    image_path = os.path.join(output_path, 'image')
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    depth_path = os.path.join(output_path, 'depth')
    if not os.path.exists(depth_path):
        os.makedirs(depth_path)
    print('Saving color images in ' + image_path)
    print('Saving depth images in ' + depth_path)

    def move_forward(vis):
        if False:
            print('Hello World!')
        ctr = vis.get_view_control()
        glb = custom_draw_geometry_with_camera_trajectory
        if glb.index >= 0:
            print('Capture image {:05d}'.format(glb.index))
            vis.capture_depth_image(os.path.join(depth_path, '{:05d}.png'.format(glb.index)), False)
            vis.capture_screen_image(os.path.join(image_path, '{:05d}.png'.format(glb.index)), False)
            '\n            depth = vis.capture_depth_float_buffer()\n            image = vis.capture_screen_float_buffer()\n            plt.imsave(os.path.join(depth_path, "{:05d}.png".format(glb.index)),\n                       np.asarray(depth),\n                       dpi=1)\n            plt.imsave(os.path.join(image_path, "{:05d}.png".format(glb.index)),\n                       np.asarray(image),\n                       dpi=1)\n            '
        glb.index = glb.index + 1
        if glb.index < len(glb.trajectory.parameters):
            ctr.convert_from_pinhole_camera_parameters(glb.trajectory.parameters[glb.index])
        else:
            custom_draw_geometry_with_camera_trajectory.vis.destroy_window()
        return False
    vis = custom_draw_geometry_with_camera_trajectory.vis
    vis.create_window()
    vis.add_geometry(pcd)
    vis.get_render_option().load_from_json(render_option_path)
    vis.register_animation_callback(move_forward)
    vis.run()
if __name__ == '__main__':
    if not o3d._build_config['ENABLE_HEADLESS_RENDERING']:
        print('Headless rendering is not enabled. Please rebuild Open3D with ENABLE_HEADLESS_RENDERING=ON')
        exit(1)
    sample_data = o3d.data.DemoCustomVisualization()
    pcd = o3d.io.read_point_cloud(sample_data.point_cloud_path)
    print('Customized visualization playing a camera trajectory. Press ctrl+z to terminate.')
    custom_draw_geometry_with_camera_trajectory(pcd, sample_data.camera_trajectory_path, sample_data.render_option_path, 'HeadlessRenderingOutput')