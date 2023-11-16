import numpy
from typing import Optional
from PyQt6 import QtCore
from PyQt6.QtCore import QCoreApplication
from PyQt6.QtGui import QImage
from UM.Logger import Logger
from cura.PreviewPass import PreviewPass
from UM.Application import Application
from UM.Math.AxisAlignedBox import AxisAlignedBox
from UM.Math.Matrix import Matrix
from UM.Math.Vector import Vector
from UM.Scene.Camera import Camera
from UM.Scene.Iterator.DepthFirstIterator import DepthFirstIterator
from UM.Scene.SceneNode import SceneNode
from UM.Qt.QtRenderer import QtRenderer

class Snapshot:

    @staticmethod
    def getImageBoundaries(image: QImage):
        if False:
            print('Hello World!')
        pixel_array = image.bits().asarray(image.sizeInBytes())
        (width, height) = (image.width(), image.height())
        pixels = numpy.frombuffer(pixel_array, dtype=numpy.uint8).reshape([height, width, 4])
        nonzero_pixels = numpy.nonzero(pixels)
        (min_y, min_x, min_a_) = numpy.amin(nonzero_pixels, axis=1)
        (max_y, max_x, max_a_) = numpy.amax(nonzero_pixels, axis=1)
        return (min_x, max_x, min_y, max_y)

    @staticmethod
    def isometricSnapshot(width: int=300, height: int=300, *, node: Optional[SceneNode]=None) -> Optional[QImage]:
        if False:
            while True:
                i = 10
        '\n        Create an isometric snapshot of the scene.\n\n        :param width: width of the aspect ratio default 300\n        :param height: height of the aspect ratio default 300\n        :param node: node of the scene default is the root of the scene\n        :return: None when there is no model on the build plate otherwise it will return an image\n\n        '
        if node is None:
            root = Application.getInstance().getController().getScene().getRoot()
        iso_view_dir = Vector(-1, -1, -1).normalized()
        bounds = Snapshot.nodeBounds(node)
        if bounds is None:
            Logger.log('w', 'There appears to be nothing to render')
            return None
        camera = Camera('snapshot')
        tangent_space_x_direction = iso_view_dir.cross(Vector.Unit_Y).normalized()
        tangent_space_y_direction = tangent_space_x_direction.cross(iso_view_dir).normalized()
        x_points = [p.dot(tangent_space_x_direction) for p in bounds.points]
        y_points = [p.dot(tangent_space_y_direction) for p in bounds.points]
        min_x = min(x_points)
        max_x = max(x_points)
        min_y = min(y_points)
        max_y = max(y_points)
        camera_width = max_x - min_x
        camera_height = max_y - min_y
        if camera_width == 0 or camera_height == 0:
            Logger.log('w', 'There appears to be nothing to render')
            return None
        if camera_width / camera_height > width / height:
            camera_height = camera_width * height / width
        else:
            camera_width = camera_height * width / height
        ortho_matrix = Matrix()
        ortho_matrix.setOrtho(-camera_width / 2, camera_width / 2, -camera_height / 2, camera_height / 2, -10000, 10000)
        camera.setPerspective(False)
        camera.setProjectionMatrix(ortho_matrix)
        camera.setPosition(bounds.center)
        camera.lookAt(bounds.center + iso_view_dir)
        renderer = QtRenderer()
        render_pass = PreviewPass(width, height, root=node)
        renderer.setViewportSize(width, height)
        renderer.setWindowSize(width, height)
        render_pass.setCamera(camera)
        renderer.addRenderPass(render_pass)
        renderer.beginRendering()
        renderer.render()
        return render_pass.getOutput()

    @staticmethod
    def nodeBounds(root_node: SceneNode) -> Optional[AxisAlignedBox]:
        if False:
            for i in range(10):
                print('nop')
        axis_aligned_box = None
        for node in DepthFirstIterator(root_node):
            if not getattr(node, '_outside_buildarea', False):
                if node.callDecoration('isSliceable') and node.getMeshData() and node.isVisible() and (not node.callDecoration('isNonThumbnailVisibleMesh')):
                    if axis_aligned_box is None:
                        axis_aligned_box = node.getBoundingBox()
                    else:
                        axis_aligned_box = axis_aligned_box + node.getBoundingBox()
        return axis_aligned_box

    @staticmethod
    def snapshot(width=300, height=300):
        if False:
            for i in range(10):
                print('nop')
        'Return a QImage of the scene\n\n        Uses PreviewPass that leaves out some elements Aspect ratio assumes a square\n\n        :param width: width of the aspect ratio default 300\n        :param height: height of the aspect ratio default 300\n        :return: None when there is no model on the build plate otherwise it will return an image\n        '
        scene = Application.getInstance().getController().getScene()
        active_camera = scene.getActiveCamera() or scene.findCamera('3d')
        (render_width, render_height) = (width, height) if active_camera is None else active_camera.getWindowSize()
        render_width = int(render_width)
        render_height = int(render_height)
        QCoreApplication.processEvents()
        preview_pass = PreviewPass(render_width, render_height)
        root = scene.getRoot()
        camera = Camera('snapshot', root)
        bbox = Snapshot.nodeBounds(root)
        if bbox is None:
            Logger.log('w', 'Unable to create snapshot as we seem to have an empty buildplate')
            return None
        look_at = bbox.center
        size = max(bbox.width, bbox.height, bbox.depth * 0.5)
        looking_from_offset = Vector(-1, 1, 2)
        if size > 0:
            looking_from_offset = looking_from_offset * size * 1.75
        camera.setPosition(look_at + looking_from_offset)
        camera.lookAt(look_at)
        satisfied = False
        size = None
        fovy = 30
        while not satisfied:
            if size is not None:
                satisfied = True
            projection_matrix = Matrix()
            projection_matrix.setPerspective(fovy, render_width / render_height, 1, 500)
            camera.setProjectionMatrix(projection_matrix)
            preview_pass.setCamera(camera)
            preview_pass.render()
            pixel_output = preview_pass.getOutput()
            try:
                (min_x, max_x, min_y, max_y) = Snapshot.getImageBoundaries(pixel_output)
            except (ValueError, AttributeError):
                Logger.logException('w', 'Failed to crop the snapshot!')
                return None
            size = max((max_x - min_x) / render_width, (max_y - min_y) / render_height)
            if size > 0.5 or satisfied:
                satisfied = True
            else:
                fovy *= 0.5
        if max_x - min_x >= max_y - min_y:
            (min_y, max_y) = (int((max_y + min_y) / 2 - (max_x - min_x) / 2), int((max_y + min_y) / 2 + (max_x - min_x) / 2))
        else:
            (min_x, max_x) = (int((max_x + min_x) / 2 - (max_y - min_y) / 2), int((max_x + min_x) / 2 + (max_y - min_y) / 2))
        cropped_image = pixel_output.copy(min_x, min_y, max_x - min_x, max_y - min_y)
        scaled_image = cropped_image.scaled(width, height, aspectRatioMode=QtCore.Qt.AspectRatioMode.IgnoreAspectRatio, transformMode=QtCore.Qt.TransformationMode.SmoothTransformation)
        return scaled_image