"""Online 3D depth video processing pipeline.

- Connects to a RGBD camera or RGBD video file (currently
  RealSense camera and bag file format are supported).
- Captures / reads color and depth frames. Allow recording from camera.
- Convert frames to point cloud, optionally with normals.
- Visualize point cloud video and results.
- Save point clouds and RGBD images for selected frames.

For this example, Open3D must be built with -DBUILD_LIBREALSENSE=ON
"""
import os
import json
import time
import logging as log
import argparse
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

class PipelineModel:
    """Controls IO (camera, video file, recording, saving frames). Methods run
    in worker threads."""

    def __init__(self, update_view, camera_config_file=None, rgbd_video=None, device=None):
        if False:
            while True:
                i = 10
        "Initialize.\n\n        Args:\n            update_view (callback): Callback to update display elements for a\n                frame.\n            camera_config_file (str): Camera configuration json file.\n            rgbd_video (str): RS bag file containing the RGBD video. If this is\n                provided, connected cameras are ignored.\n            device (str): Compute device (e.g.: 'cpu:0' or 'cuda:0').\n        "
        self.update_view = update_view
        if device:
            self.device = device.lower()
        else:
            self.device = 'cuda:0' if o3d.core.cuda.is_available() else 'cpu:0'
        self.o3d_device = o3d.core.Device(self.device)
        self.video = None
        self.camera = None
        self.flag_capture = False
        self.cv_capture = threading.Condition()
        self.recording = False
        self.flag_record = False
        if rgbd_video:
            self.video = o3d.t.io.RGBDVideoReader.create(rgbd_video)
            self.rgbd_metadata = self.video.metadata
            self.status_message = f'Video {rgbd_video} opened.'
        else:
            now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            filename = f'{now}.bag'
            self.camera = o3d.t.io.RealSenseSensor()
            if camera_config_file:
                with open(camera_config_file) as ccf:
                    self.camera.init_sensor(o3d.t.io.RealSenseSensorConfig(json.load(ccf)), filename=filename)
            else:
                self.camera.init_sensor(filename=filename)
            self.camera.start_capture(start_record=False)
            self.rgbd_metadata = self.camera.get_metadata()
            self.status_message = f'Camera {self.rgbd_metadata.serial_number} opened.'
        log.info(self.rgbd_metadata)
        self.extrinsics = o3d.core.Tensor.eye(4, dtype=o3d.core.Dtype.Float32, device=self.o3d_device)
        self.intrinsic_matrix = o3d.core.Tensor(self.rgbd_metadata.intrinsics.intrinsic_matrix, dtype=o3d.core.Dtype.Float32, device=self.o3d_device)
        self.depth_max = 3.0
        self.pcd_stride = 2
        self.flag_normals = False
        self.flag_save_rgbd = False
        self.flag_save_pcd = False
        self.pcd_frame = None
        self.rgbd_frame = None
        self.executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix='Capture-Save')
        self.flag_exit = False

    @property
    def max_points(self):
        if False:
            while True:
                i = 10
        'Max points in one frame for the camera or RGBD video resolution.'
        return self.rgbd_metadata.width * self.rgbd_metadata.height

    @property
    def vfov(self):
        if False:
            print('Hello World!')
        'Camera or RGBD video vertical field of view.'
        return np.rad2deg(2 * np.arctan(self.intrinsic_matrix[1, 2].item() / self.intrinsic_matrix[1, 1].item()))

    def run(self):
        if False:
            while True:
                i = 10
        'Run pipeline.'
        n_pts = 0
        frame_id = 0
        t1 = time.perf_counter()
        if self.video:
            self.rgbd_frame = self.video.next_frame()
        else:
            self.rgbd_frame = self.camera.capture_frame(wait=True, align_depth_to_color=True)
        pcd_errors = 0
        while not self.flag_exit and (self.video is None or (self.video and (not self.video.is_eof()))):
            if self.video:
                future_rgbd_frame = self.executor.submit(self.video.next_frame)
            else:
                future_rgbd_frame = self.executor.submit(self.camera.capture_frame, wait=True, align_depth_to_color=True)
            if self.flag_save_pcd:
                self.save_pcd()
                self.flag_save_pcd = False
            try:
                self.rgbd_frame = self.rgbd_frame.to(self.o3d_device)
                self.pcd_frame = o3d.t.geometry.PointCloud.create_from_rgbd_image(self.rgbd_frame, self.intrinsic_matrix, self.extrinsics, self.rgbd_metadata.depth_scale, self.depth_max, self.pcd_stride, self.flag_normals)
                depth_in_color = self.rgbd_frame.depth.colorize_depth(self.rgbd_metadata.depth_scale, 0, self.depth_max)
            except RuntimeError:
                pcd_errors += 1
            if self.pcd_frame.is_empty():
                log.warning(f'No valid depth data in frame {frame_id})')
                continue
            n_pts += self.pcd_frame.point.positions.shape[0]
            if frame_id % 60 == 0 and frame_id > 0:
                (t0, t1) = (t1, time.perf_counter())
                log.debug(f'\nframe_id = {frame_id}, \t {(t1 - t0) * 1000.0 / 60:0.2f}ms/frame \t {(t1 - t0) * 1000000000.0 / n_pts} ms/Mp\t')
                n_pts = 0
            frame_elements = {'color': self.rgbd_frame.color.cpu(), 'depth': depth_in_color.cpu(), 'pcd': self.pcd_frame.cpu(), 'status_message': self.status_message}
            self.update_view(frame_elements)
            if self.flag_save_rgbd:
                self.save_rgbd()
                self.flag_save_rgbd = False
            self.rgbd_frame = future_rgbd_frame.result()
            with self.cv_capture:
                self.cv_capture.wait_for(predicate=lambda : self.flag_capture or self.flag_exit)
            self.toggle_record()
            frame_id += 1
        if self.camera:
            self.camera.stop_capture()
        else:
            self.video.close()
        self.executor.shutdown()
        log.debug(f'create_from_depth_image() errors = {pcd_errors}')

    def toggle_record(self):
        if False:
            print('Hello World!')
        if self.camera is not None:
            if self.flag_record and (not self.recording):
                self.camera.resume_record()
                self.recording = True
            elif not self.flag_record and self.recording:
                self.camera.pause_record()
                self.recording = False

    def save_pcd(self):
        if False:
            print('Hello World!')
        'Save current point cloud.'
        now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = f'{self.rgbd_metadata.serial_number}_pcd_{now}.ply'
        self.pcd_frame.point.colors = (self.pcd_frame.point.colors * 255).to(o3d.core.Dtype.UInt8)
        self.executor.submit(o3d.t.io.write_point_cloud, filename, self.pcd_frame, write_ascii=False, compressed=True, print_progress=False)
        self.status_message = f'Saving point cloud to {filename}.'

    def save_rgbd(self):
        if False:
            print('Hello World!')
        'Save current RGBD image pair.'
        now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = f'{self.rgbd_metadata.serial_number}_color_{now}.jpg'
        self.executor.submit(o3d.t.io.write_image, filename, self.rgbd_frame.color)
        filename = f'{self.rgbd_metadata.serial_number}_depth_{now}.png'
        self.executor.submit(o3d.t.io.write_image, filename, self.rgbd_frame.depth)
        self.status_message = f'Saving RGBD images to {filename[:-3]}.{{jpg,png}}.'

class PipelineView:
    """Controls display and user interface. All methods must run in the main thread."""

    def __init__(self, vfov=60, max_pcd_vertices=1 << 20, **callbacks):
        if False:
            i = 10
            return i + 15
        'Initialize.\n\n        Args:\n            vfov (float): Vertical field of view for the 3D scene.\n            max_pcd_vertices (int): Maximum point clud verties for which memory\n                is allocated.\n            callbacks (dict of kwargs): Callbacks provided by the controller\n                for various operations.\n        '
        self.vfov = vfov
        self.max_pcd_vertices = max_pcd_vertices
        gui.Application.instance.initialize()
        self.window = gui.Application.instance.create_window('Open3D || Online RGBD Video Processing', 1280, 960)
        self.window.set_on_layout(self.on_layout)
        self.window.set_on_close(callbacks['on_window_close'])
        self.pcd_material = o3d.visualization.rendering.MaterialRecord()
        self.pcd_material.shader = 'defaultLit'
        self.pcd_material.point_size = int(4 * self.window.scaling)
        self.pcdview = gui.SceneWidget()
        self.window.add_child(self.pcdview)
        self.pcdview.enable_scene_caching(True)
        self.pcdview.scene = rendering.Open3DScene(self.window.renderer)
        self.pcdview.scene.set_background([1, 1, 1, 1])
        self.pcdview.scene.set_lighting(rendering.Open3DScene.LightingProfile.SOFT_SHADOWS, [0, -6, 0])
        self.pcd_bounds = o3d.geometry.AxisAlignedBoundingBox([-3, -3, 0], [3, 3, 6])
        self.camera_view()
        em = self.window.theme.font_size
        self.panel = gui.Vert(em, gui.Margins(em, em, em, em))
        self.panel.preferred_width = int(360 * self.window.scaling)
        self.window.add_child(self.panel)
        toggles = gui.Horiz(em)
        self.panel.add_child(toggles)
        toggle_capture = gui.ToggleSwitch('Capture / Play')
        toggle_capture.is_on = False
        toggle_capture.set_on_clicked(callbacks['on_toggle_capture'])
        toggles.add_child(toggle_capture)
        self.flag_normals = False
        self.toggle_normals = gui.ToggleSwitch('Colors / Normals')
        self.toggle_normals.is_on = False
        self.toggle_normals.set_on_clicked(callbacks['on_toggle_normals'])
        toggles.add_child(self.toggle_normals)
        view_buttons = gui.Horiz(em)
        self.panel.add_child(view_buttons)
        view_buttons.add_stretch()
        camera_view = gui.Button('Camera view')
        camera_view.set_on_clicked(self.camera_view)
        view_buttons.add_child(camera_view)
        birds_eye_view = gui.Button("Bird's eye view")
        birds_eye_view.set_on_clicked(self.birds_eye_view)
        view_buttons.add_child(birds_eye_view)
        view_buttons.add_stretch()
        save_toggle = gui.Horiz(em)
        self.panel.add_child(save_toggle)
        save_toggle.add_child(gui.Label('Record / Save'))
        self.toggle_record = None
        if callbacks['on_toggle_record'] is not None:
            save_toggle.add_fixed(1.5 * em)
            self.toggle_record = gui.ToggleSwitch('Video')
            self.toggle_record.is_on = False
            self.toggle_record.set_on_clicked(callbacks['on_toggle_record'])
            save_toggle.add_child(self.toggle_record)
        save_buttons = gui.Horiz(em)
        self.panel.add_child(save_buttons)
        save_buttons.add_stretch()
        save_pcd = gui.Button('Save Point cloud')
        save_pcd.set_on_clicked(callbacks['on_save_pcd'])
        save_buttons.add_child(save_pcd)
        save_rgbd = gui.Button('Save RGBD frame')
        save_rgbd.set_on_clicked(callbacks['on_save_rgbd'])
        save_buttons.add_child(save_rgbd)
        save_buttons.add_stretch()
        self.video_size = (int(240 * self.window.scaling), int(320 * self.window.scaling), 3)
        self.show_color = gui.CollapsableVert('Color image')
        self.show_color.set_is_open(False)
        self.panel.add_child(self.show_color)
        self.color_video = gui.ImageWidget(o3d.geometry.Image(np.zeros(self.video_size, dtype=np.uint8)))
        self.show_color.add_child(self.color_video)
        self.show_depth = gui.CollapsableVert('Depth image')
        self.show_depth.set_is_open(False)
        self.panel.add_child(self.show_depth)
        self.depth_video = gui.ImageWidget(o3d.geometry.Image(np.zeros(self.video_size, dtype=np.uint8)))
        self.show_depth.add_child(self.depth_video)
        self.status_message = gui.Label('')
        self.panel.add_child(self.status_message)
        self.flag_exit = False
        self.flag_gui_init = False

    def update(self, frame_elements):
        if False:
            return 10
        "Update visualization with point cloud and images. Must run in main\n        thread since this makes GUI calls.\n\n        Args:\n            frame_elements: dict {element_type: geometry element}.\n                Dictionary of element types to geometry elements to be updated\n                in the GUI:\n                    'pcd': point cloud,\n                    'color': rgb image (3 channel, uint8),\n                    'depth': depth image (uint8),\n                    'status_message': message\n        "
        if not self.flag_gui_init:
            dummy_pcd = o3d.t.geometry.PointCloud({'positions': o3d.core.Tensor.zeros((self.max_pcd_vertices, 3), o3d.core.Dtype.Float32), 'colors': o3d.core.Tensor.zeros((self.max_pcd_vertices, 3), o3d.core.Dtype.Float32), 'normals': o3d.core.Tensor.zeros((self.max_pcd_vertices, 3), o3d.core.Dtype.Float32)})
            if self.pcdview.scene.has_geometry('pcd'):
                self.pcdview.scene.remove_geometry('pcd')
            self.pcd_material.shader = 'normals' if self.flag_normals else 'defaultLit'
            self.pcdview.scene.add_geometry('pcd', dummy_pcd, self.pcd_material)
            self.flag_gui_init = True
        if os.name == 'nt':
            self.pcdview.scene.remove_geometry('pcd')
            self.pcdview.scene.add_geometry('pcd', frame_elements['pcd'], self.pcd_material)
        else:
            update_flags = rendering.Scene.UPDATE_POINTS_FLAG | rendering.Scene.UPDATE_COLORS_FLAG | (rendering.Scene.UPDATE_NORMALS_FLAG if self.flag_normals else 0)
            self.pcdview.scene.scene.update_geometry('pcd', frame_elements['pcd'], update_flags)
        if self.show_color.get_is_open() and 'color' in frame_elements:
            sampling_ratio = self.video_size[1] / frame_elements['color'].columns
            self.color_video.update_image(frame_elements['color'].resize(sampling_ratio).cpu())
        if self.show_depth.get_is_open() and 'depth' in frame_elements:
            sampling_ratio = self.video_size[1] / frame_elements['depth'].columns
            self.depth_video.update_image(frame_elements['depth'].resize(sampling_ratio).cpu())
        if 'status_message' in frame_elements:
            self.status_message.text = frame_elements['status_message']
        self.pcdview.force_redraw()

    def camera_view(self):
        if False:
            for i in range(10):
                print('nop')
        'Callback to reset point cloud view to the camera'
        self.pcdview.setup_camera(self.vfov, self.pcd_bounds, [0, 0, 0])
        self.pcdview.scene.camera.look_at([0, 0, 1], [0, 0, 0], [0, -1, 0])

    def birds_eye_view(self):
        if False:
            for i in range(10):
                print('nop')
        'Callback to reset point cloud view to birds eye (overhead) view'
        self.pcdview.setup_camera(self.vfov, self.pcd_bounds, [0, 0, 0])
        self.pcdview.scene.camera.look_at([0, 0, 1.5], [0, 3, 1.5], [0, -1, 0])

    def on_layout(self, layout_context):
        if False:
            while True:
                i = 10
        'Callback on window initialize / resize'
        frame = self.window.content_rect
        self.pcdview.frame = frame
        panel_size = self.panel.calc_preferred_size(layout_context, self.panel.Constraints())
        self.panel.frame = gui.Rect(frame.get_right() - panel_size.width, frame.y, panel_size.width, panel_size.height)

class PipelineController:
    """Entry point for the app. Controls the PipelineModel object for IO and
    processing  and the PipelineView object for display and UI. All methods
    operate on the main thread.
    """

    def __init__(self, camera_config_file=None, rgbd_video=None, device=None):
        if False:
            return 10
        "Initialize.\n\n        Args:\n            camera_config_file (str): Camera configuration json file.\n            rgbd_video (str): RS bag file containing the RGBD video. If this is\n                provided, connected cameras are ignored.\n            device (str): Compute device (e.g.: 'cpu:0' or 'cuda:0').\n        "
        self.pipeline_model = PipelineModel(self.update_view, camera_config_file, rgbd_video, device)
        self.pipeline_view = PipelineView(1.25 * self.pipeline_model.vfov, self.pipeline_model.max_points, on_window_close=self.on_window_close, on_toggle_capture=self.on_toggle_capture, on_save_pcd=self.on_save_pcd, on_save_rgbd=self.on_save_rgbd, on_toggle_record=self.on_toggle_record if rgbd_video is None else None, on_toggle_normals=self.on_toggle_normals)
        threading.Thread(name='PipelineModel', target=self.pipeline_model.run).start()
        gui.Application.instance.run()

    def update_view(self, frame_elements):
        if False:
            while True:
                i = 10
        'Updates view with new data. May be called from any thread.\n\n        Args:\n            frame_elements (dict): Display elements (point cloud and images)\n                from the new frame to be shown.\n        '
        gui.Application.instance.post_to_main_thread(self.pipeline_view.window, lambda : self.pipeline_view.update(frame_elements))

    def on_toggle_capture(self, is_enabled):
        if False:
            print('Hello World!')
        'Callback to toggle capture.'
        self.pipeline_model.flag_capture = is_enabled
        if not is_enabled:
            self.on_toggle_record(False)
            if self.pipeline_view.toggle_record is not None:
                self.pipeline_view.toggle_record.is_on = False
        else:
            with self.pipeline_model.cv_capture:
                self.pipeline_model.cv_capture.notify()

    def on_toggle_record(self, is_enabled):
        if False:
            return 10
        'Callback to toggle recording RGBD video.'
        self.pipeline_model.flag_record = is_enabled

    def on_toggle_normals(self, is_enabled):
        if False:
            return 10
        'Callback to toggle display of normals'
        self.pipeline_model.flag_normals = is_enabled
        self.pipeline_view.flag_normals = is_enabled
        self.pipeline_view.flag_gui_init = False

    def on_window_close(self):
        if False:
            while True:
                i = 10
        'Callback when the user closes the application window.'
        self.pipeline_model.flag_exit = True
        with self.pipeline_model.cv_capture:
            self.pipeline_model.cv_capture.notify_all()
        return True

    def on_save_pcd(self):
        if False:
            for i in range(10):
                print('nop')
        'Callback to save current point cloud.'
        self.pipeline_model.flag_save_pcd = True

    def on_save_rgbd(self):
        if False:
            i = 10
            return i + 15
        'Callback to save current RGBD image pair.'
        self.pipeline_model.flag_save_rgbd = True
if __name__ == '__main__':
    log.basicConfig(level=log.INFO)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--camera-config', help='RGBD camera configuration JSON file')
    parser.add_argument('--rgbd-video', help='RGBD video file (RealSense bag)')
    parser.add_argument('--device', help='Device to run computations. e.g. cpu:0 or cuda:0 Default is CUDA GPU if available, else CPU.')
    args = parser.parse_args()
    if args.camera_config and args.rgbd_video:
        log.critical('Please provide only one of --camera-config and --rgbd-video arguments')
    else:
        PipelineController(args.camera_config, args.rgbd_video, args.device)