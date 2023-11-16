import open3d as o3d

def custom_key_action_without_kb_repeat_delay(pcd):
    if False:
        print('Hello World!')
    rotating = False
    vis = o3d.visualization.VisualizerWithKeyCallback()

    def key_action_callback(vis, action, mods):
        if False:
            return 10
        nonlocal rotating
        print(action)
        if action == 1:
            rotating = True
        elif action == 0:
            rotating = False
        elif action == 2:
            pass
        return True

    def animation_callback(vis):
        if False:
            while True:
                i = 10
        nonlocal rotating
        if rotating:
            ctr = vis.get_view_control()
            ctr.rotate(10.0, 0.0)
    vis.register_key_action_callback(32, key_action_callback)
    vis.register_animation_callback(animation_callback)
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()
if __name__ == '__main__':
    ply_data = o3d.data.PLYPointCloud()
    pcd = o3d.io.read_point_cloud(ply_data.path)
    print('Customized visualization with smooth key action (without keyboard repeat delay)')
    custom_key_action_without_kb_repeat_delay(pcd)