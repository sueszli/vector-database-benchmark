from importlib import reload
import unittest.mock as mock
from apps.blender.resources.images.entrypoints.scripts.render_tools import scenefileeditor
from golem.testutils import TempDirFixture

class TestSceneFileEditor(TempDirFixture):

    def test_crop_file_generation_dummy(self):
        if False:
            print('Hello World!')
        'Test blender script generation with simplistic template.'
        filepath = self.temp_file_name('tmpscene')
        with open(filepath, 'w') as f:
            f.write('%(resolution_x)d\n%(resolution_y)d\n%(border_min_x).3f\n%(border_max_x).3f\n%(border_min_y).3f\n%(border_max_y).3f\n%(use_compositing)r\n%(samples)d')
        result = scenefileeditor._generate_blender_crop_file(resolution=(1, 2), borders_x=(3.01, 3.02), borders_y=(4.01, 4.02), use_compositing=True, samples=5, template_path=filepath)
        expected = '1\n2\n3.010\n3.020\n4.010\n4.020\nTrue\n5'
        self.assertEqual(result, expected)

    def test_crop_file_generation_full(self):
        if False:
            return 10
        'Mocks blender by providing bpy and tests whether generated script\n         acted as expected.'
        resolution = (1, 2)
        borders_x = (3.01, 3.02)
        borders_y = (4.01, 4.02)
        use_compositing = True
        samples = 5
        expected_attributes = {'resolution_x': resolution[0], 'resolution_y': resolution[1], 'border_min_x': borders_x[0], 'border_max_x': borders_x[1], 'border_min_y': borders_y[0], 'border_max_y': borders_y[1], 'use_compositing': use_compositing, 'tile_x': 0, 'tile_y': 0, 'resolution_percentage': 100, 'use_border': True, 'use_crop_to_border': True}
        result = scenefileeditor._generate_blender_crop_file(resolution=resolution, borders_x=borders_x, borders_y=borders_y, use_compositing=use_compositing, samples=samples, template_path=scenefileeditor.BLENDER_CROP_TEMPLATE_PATH)
        scene_m = mock.MagicMock()
        scene_m.render = mock.NonCallableMock()
        bpy_m = mock.MagicMock()
        bpy_m.context.scene = scene_m
        bpy_m.ops.render.render.return_value = None
        bpy_m.ops.file.report_missing_files.return_value = None
        result = result.replace('import bpy', '')
        globs = dict(globals())
        globs['bpy'] = bpy_m
        exec(result, globs)
        for name in expected_attributes:
            expected = expected_attributes[name]
            value = getattr(scene_m.render, name)
            self.assertEqual(value, expected, 'Value of scene.render.%s expected:%r got:%r' % (name, expected, value))
        bpy_m.ops.render.render.assert_not_called()
        bpy_m.ops.file.report_missing_files.assert_called_once_with()

    def tearDown(self):
        if False:
            print('Hello World!')
        super(TestSceneFileEditor, self).tearDown()
        reload(scenefileeditor)