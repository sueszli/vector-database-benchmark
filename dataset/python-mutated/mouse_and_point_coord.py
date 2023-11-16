import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

class ExampleApp:

    def __init__(self, cloud):
        if False:
            i = 10
            return i + 15
        app = gui.Application.instance
        self.window = app.create_window('Open3D - GetCoord Example', 1024, 768)
        self.window.set_on_layout(self._on_layout)
        self.widget3d = gui.SceneWidget()
        self.window.add_child(self.widget3d)
        self.info = gui.Label('')
        self.info.visible = False
        self.window.add_child(self.info)
        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)
        mat = rendering.MaterialRecord()
        mat.shader = 'defaultUnlit'
        mat.point_size = 3 * self.window.scaling
        self.widget3d.scene.add_geometry('Point Cloud', cloud, mat)
        bounds = self.widget3d.scene.bounding_box
        center = bounds.get_center()
        self.widget3d.setup_camera(60, bounds, center)
        self.widget3d.look_at(center, center - [0, 0, 3], [0, -1, 0])
        self.widget3d.set_on_mouse(self._on_mouse_widget3d)

    def _on_layout(self, layout_context):
        if False:
            for i in range(10):
                print('nop')
        r = self.window.content_rect
        self.widget3d.frame = r
        pref = self.info.calc_preferred_size(layout_context, gui.Widget.Constraints())
        self.info.frame = gui.Rect(r.x, r.get_bottom() - pref.height, pref.width, pref.height)

    def _on_mouse_widget3d(self, event):
        if False:
            while True:
                i = 10
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_modifier_down(gui.KeyModifier.CTRL):

            def depth_callback(depth_image):
                if False:
                    i = 10
                    return i + 15
                x = event.x - self.widget3d.frame.x
                y = event.y - self.widget3d.frame.y
                depth = np.asarray(depth_image)[y, x]
                if depth == 1.0:
                    text = ''
                else:
                    world = self.widget3d.scene.camera.unproject(x, y, depth, self.widget3d.frame.width, self.widget3d.frame.height)
                    text = '({:.3f}, {:.3f}, {:.3f})'.format(world[0], world[1], world[2])

                def update_label():
                    if False:
                        for i in range(10):
                            print('nop')
                    self.info.text = text
                    self.info.visible = text != ''
                    self.window.set_needs_layout()
                gui.Application.instance.post_to_main_thread(self.window, update_label)
            self.widget3d.scene.scene.render_to_depth_image(depth_callback)
            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

def main():
    if False:
        print('Hello World!')
    app = gui.Application.instance
    app.initialize()
    pcd_data = o3d.data.DemoICPPointClouds()
    cloud = o3d.io.read_point_cloud(pcd_data.paths[0])
    ex = ExampleApp(cloud)
    app.run()
if __name__ == '__main__':
    main()