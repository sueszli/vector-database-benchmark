from unittest.mock import MagicMock, patch
import pytest
from UM.Math.Polygon import Polygon
from UM.Math.Vector import Vector
from cura.BuildVolume import BuildVolume, PRIME_CLEARANCE
import numpy

@pytest.fixture
def build_volume() -> BuildVolume:
    if False:
        i = 10
        return i + 15
    mocked_application = MagicMock()
    mocked_platform = MagicMock(name='platform')
    with patch('cura.BuildVolume.Platform', mocked_platform):
        return BuildVolume(mocked_application)

def test_buildVolumeSetSizes(build_volume):
    if False:
        for i in range(10):
            print('nop')
    build_volume.setWidth(10)
    assert build_volume.getDiagonalSize() == 10
    build_volume.setWidth(0)
    build_volume.setHeight(100)
    assert build_volume.getDiagonalSize() == 100
    build_volume.setHeight(0)
    build_volume.setDepth(200)
    assert build_volume.getDiagonalSize() == 200

def test_buildMesh(build_volume):
    if False:
        while True:
            i = 10
    mesh = build_volume._buildMesh(0, 100, 0, 100, 0, 100, 1)
    result_vertices = numpy.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 100.0, 0.0], [100.0, 100.0, 0.0], [100.0, 0.0, 0.0], [100.0, 100.0, 0.0], [0.0, 0.0, 100.0], [100.0, 0.0, 100.0], [0.0, 0.0, 100.0], [0.0, 100.0, 100.0], [0.0, 100.0, 100.0], [100.0, 100.0, 100.0], [100.0, 0.0, 100.0], [100.0, 100.0, 100.0], [0.0, 0.0, 0.0], [0.0, 0.0, 100.0], [100.0, 0.0, 0.0], [100.0, 0.0, 100.0], [0.0, 100.0, 0.0], [0.0, 100.0, 100.0], [100.0, 100.0, 0.0], [100.0, 100.0, 100.0]], dtype=numpy.float32)
    assert numpy.array_equal(result_vertices, mesh.getVertices())

def test_buildGridMesh(build_volume):
    if False:
        return 10
    mesh = build_volume._buildGridMesh(0, 100, 0, 100, 0, 100, 1)
    result_vertices = numpy.array([[0.0, -1.0, 0.0], [100.0, -1.0, 100.0], [100.0, -1.0, 0.0], [0.0, -1.0, 0.0], [0.0, -1.0, 100.0], [100.0, -1.0, 100.0]])
    assert numpy.array_equal(result_vertices, mesh.getVertices())

def test_clamp(build_volume):
    if False:
        while True:
            i = 10
    assert build_volume._clamp(0, 0, 200) == 0
    assert build_volume._clamp(0, -200, 200) == 0
    assert build_volume._clamp(300, -200, 200) == 200

class TestCalculateBedAdhesionSize:
    setting_property_dict = {'adhesion_type': {'value': 'brim'}, 'skirt_brim_line_width': {'value': 0}, 'initial_layer_line_width_factor': {'value': 0}, 'brim_line_count': {'value': 0}, 'machine_width': {'value': 200}, 'machine_depth': {'value': 200}, 'skirt_line_count': {'value': 0}, 'skirt_gap': {'value': 0}, 'brim_gap': {'value': 0}, 'raft_margin': {'value': 0}, 'material_shrinkage_percentage': {'value': 100.0}, 'material_shrinkage_percentage_xy': {'value': 100.0}, 'material_shrinkage_percentage_z': {'value': 100.0}}

    def getPropertySideEffect(*args, **kwargs):
        if False:
            return 10
        properties = TestCalculateBedAdhesionSize.setting_property_dict.get(args[1])
        if properties:
            return properties.get(args[2])

    def createAndSetGlobalStack(self, build_volume):
        if False:
            return 10
        mocked_stack = MagicMock(name='mocked_stack')
        mocked_stack.getProperty = MagicMock(side_effect=self.getPropertySideEffect)
        mocked_extruder = MagicMock(name='mocked_extruder')
        mocked_extruder.getProperty = MagicMock(side_effect=self.getPropertySideEffect)
        mocked_stack.extruderList = [mocked_extruder]
        build_volume._global_container_stack = mocked_stack

    def test_noGlobalStack(self, build_volume: BuildVolume):
        if False:
            for i in range(10):
                print('nop')
        assert build_volume._calculateBedAdhesionSize([]) is None

    @pytest.mark.parametrize('setting_dict, result', [({}, 0), ({'adhesion_type': {'value': 'skirt'}}, 0), ({'adhesion_type': {'value': 'raft'}}, 0), ({'adhesion_type': {'value': 'none'}}, 0), ({'adhesion_type': {'value': 'skirt'}, 'skirt_line_count': {'value': 2}, 'initial_layer_line_width_factor': {'value': 1}, 'skirt_brim_line_width': {'value': 2}}, 0), ({'adhesion_type': {'value': 'skirt'}, 'prime_tower_brim_enable': {'value': True}, 'skirt_brim_line_width': {'value': 2}, 'initial_layer_line_width_factor': {'value': 3}}, 0), ({'brim_line_count': {'value': 1}, 'skirt_brim_line_width': {'value': 2}, 'initial_layer_line_width_factor': {'value': 3}}, 0), ({'brim_line_count': {'value': 2}, 'skirt_brim_line_width': {'value': 2}, 'initial_layer_line_width_factor': {'value': 3}}, 0), ({'brim_line_count': {'value': 9000000}, 'skirt_brim_line_width': {'value': 90000}, 'initial_layer_line_width_factor': {'value': 9000}}, 0)])
    def test_singleExtruder(self, build_volume: BuildVolume, setting_dict, result):
        if False:
            for i in range(10):
                print('nop')
        self.createAndSetGlobalStack(build_volume)
        patched_dictionary = self.setting_property_dict.copy()
        patched_dictionary.update(setting_dict)
        patched_dictionary.update({'skirt_brim_extruder_nr': {'value': 0}, 'raft_base_extruder_nr': {'value': 0}, 'raft_interface_extruder_nr': {'value': 0}, 'raft_surface_extruder_nr': {'value': 0}})
        with patch.dict(self.setting_property_dict, patched_dictionary):
            assert build_volume._calculateBedAdhesionSize([]) == result

class TestComputeDisallowedAreasStatic:
    setting_property_dict = {'machine_disallowed_areas': {'value': [[[-200, 112.5], [-82, 112.5], [-84, 102.5], [-115, 102.5]]]}, 'machine_width': {'value': 200}, 'machine_depth': {'value': 200}, 'material_shrinkage_percentage': {'value': 100.0}, 'material_shrinkage_percentage_xy': {'value': 100.0}, 'material_shrinkage_percentage_z': {'value': 100.0}}

    def getPropertySideEffect(*args, **kwargs):
        if False:
            return 10
        properties = TestComputeDisallowedAreasStatic.setting_property_dict.get(args[1])
        if properties:
            return properties.get(args[2])

    def test_computeDisallowedAreasStaticNoExtruder(self, build_volume: BuildVolume):
        if False:
            while True:
                i = 10
        mocked_stack = MagicMock()
        mocked_stack.getProperty = MagicMock(side_effect=self.getPropertySideEffect)
        build_volume._global_container_stack = mocked_stack
        assert build_volume._computeDisallowedAreasStatic(0, []) == {}

    def test_computeDisalowedAreasStaticSingleExtruder(self, build_volume: BuildVolume):
        if False:
            for i in range(10):
                print('nop')
        mocked_stack = MagicMock()
        mocked_stack.getProperty = MagicMock(side_effect=self.getPropertySideEffect)
        mocked_extruder = MagicMock()
        mocked_extruder.getProperty = MagicMock(side_effect=self.getPropertySideEffect)
        mocked_extruder.getId = MagicMock(return_value='zomg')
        build_volume._global_container_stack = mocked_stack
        with patch('cura.Settings.ExtruderManager.ExtruderManager.getInstance'):
            result = build_volume._computeDisallowedAreasStatic(0, [mocked_extruder])
            assert result == {'zomg': [Polygon([[-84.0, 102.5], [-115.0, 102.5], [-200.0, 112.5], [-82.0, 112.5]]), Polygon([[-100.0, -100.0], [-100.0, 100.0], [-99.9, 99.9], [-99.9, -99.9]]), Polygon([[100.0, 100.0], [100.0, -100.0], [99.9, -99.9], [99.9, 99.9]]), Polygon([[-100.0, 100.0], [100.0, 100.0], [99.9, 99.9], [-99.9, 99.9]]), Polygon([[100.0, -100.0], [-100.0, -100.0], [-99.9, -99.9], [99.9, -99.9]])]}

    def test_computeDisalowedAreasMutliExtruder(self, build_volume):
        if False:
            i = 10
            return i + 15
        mocked_stack = MagicMock()
        mocked_stack.getProperty = MagicMock(side_effect=self.getPropertySideEffect)
        mocked_extruder = MagicMock()
        mocked_extruder.getProperty = MagicMock(side_effect=self.getPropertySideEffect)
        mocked_extruder.getId = MagicMock(return_value='zomg')
        extruder_manager = MagicMock()
        extruder_manager.getActiveExtruderStacks = MagicMock(return_value=[mocked_stack])
        build_volume._global_container_stack = mocked_stack
        with patch('cura.Settings.ExtruderManager.ExtruderManager.getInstance', MagicMock(return_value=extruder_manager)):
            result = build_volume._computeDisallowedAreasStatic(0, [mocked_extruder])
            assert result == {'zomg': [Polygon([[-84.0, 102.5], [-115.0, 102.5], [-200.0, 112.5], [-82.0, 112.5]]), Polygon([[-100.0, -100.0], [-100.0, 100.0], [-99.9, 99.9], [-99.9, -99.9]]), Polygon([[100.0, 100.0], [100.0, -100.0], [99.9, -99.9], [99.9, 99.9]]), Polygon([[-100.0, 100.0], [100.0, 100.0], [99.9, 99.9], [-99.9, 99.9]]), Polygon([[100.0, -100.0], [-100.0, -100.0], [-99.9, -99.9], [99.9, -99.9]])]}

class TestUpdateRaftThickness:
    setting_property_dict = {'raft_base_thickness': {'value': 1}, 'raft_interface_layers': {'value': 2}, 'raft_interface_thickness': {'value': 1}, 'raft_surface_layers': {'value': 3}, 'raft_surface_thickness': {'value': 1}, 'raft_airgap': {'value': 1}, 'layer_0_z_overlap': {'value': 1}, 'adhesion_type': {'value': 'raft'}, 'material_shrinkage_percentage': {'value': 100.0}, 'material_shrinkage_percentage_xy': {'value': 100.0}, 'material_shrinkage_percentage_z': {'value': 100.0}}

    def getPropertySideEffect(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        properties = TestUpdateRaftThickness.setting_property_dict.get(args[1])
        if properties:
            return properties.get(args[2])

    def createMockedStack(self):
        if False:
            i = 10
            return i + 15
        mocked_global_stack = MagicMock(name='mocked_global_stack')
        mocked_global_stack.getProperty = MagicMock(side_effect=self.getPropertySideEffect)
        extruder_stack = MagicMock()
        return mocked_global_stack

    def test_simple(self, build_volume: BuildVolume):
        if False:
            print('Hello World!')
        build_volume.raftThicknessChanged = MagicMock()
        mocked_global_stack = self.createMockedStack()
        build_volume._global_container_stack = mocked_global_stack
        assert build_volume.getRaftThickness() == 0
        build_volume._updateRaftThickness()
        assert build_volume.getRaftThickness() == 6
        assert build_volume.raftThicknessChanged.emit.call_count == 1

    def test_adhesionIsNotRaft(self, build_volume: BuildVolume):
        if False:
            return 10
        patched_dictionary = self.setting_property_dict.copy()
        patched_dictionary['adhesion_type'] = {'value': 'not_raft'}
        mocked_global_stack = self.createMockedStack()
        build_volume._global_container_stack = mocked_global_stack
        assert build_volume.getRaftThickness() == 0
        with patch.dict(self.setting_property_dict, patched_dictionary):
            build_volume._updateRaftThickness()
        assert build_volume.getRaftThickness() == 0

    def test_noGlobalStack(self, build_volume: BuildVolume):
        if False:
            for i in range(10):
                print('nop')
        build_volume.raftThicknessChanged = MagicMock()
        assert build_volume.getRaftThickness() == 0
        build_volume._updateRaftThickness()
        assert build_volume.getRaftThickness() == 0
        assert build_volume.raftThicknessChanged.emit.call_count == 0

class TestComputeDisallowedAreasPrimeBlob:
    setting_property_dict = {'machine_width': {'value': 50}, 'machine_depth': {'value': 100}, 'prime_blob_enable': {'value': True}, 'extruder_prime_pos_x': {'value': 25}, 'extruder_prime_pos_y': {'value': 50}, 'machine_center_is_zero': {'value': True}, 'material_shrinkage_percentage': {'value': 100.0}, 'material_shrinkage_percentage_xy': {'value': 100.0}, 'material_shrinkage_percentage_z': {'value': 100.0}}

    def getPropertySideEffect(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        properties = TestComputeDisallowedAreasPrimeBlob.setting_property_dict.get(args[1])
        if properties:
            return properties.get(args[2])

    def test_noGlobalContainer(self, build_volume: BuildVolume):
        if False:
            while True:
                i = 10
        assert build_volume._computeDisallowedAreasPrimeBlob(12, []) == {}

    def test_noExtruders(self, build_volume: BuildVolume):
        if False:
            i = 10
            return i + 15
        mocked_stack = MagicMock()
        mocked_stack.getProperty = MagicMock(side_effect=self.getPropertySideEffect)
        build_volume._global_container_stack = mocked_stack
        assert build_volume._computeDisallowedAreasPrimeBlob(12, []) == {}

    def test_singleExtruder(self, build_volume: BuildVolume):
        if False:
            print('Hello World!')
        mocked_global_stack = MagicMock(name='mocked_global_stack')
        mocked_global_stack.getProperty = MagicMock(side_effect=self.getPropertySideEffect)
        mocked_extruder_stack = MagicMock(name='mocked_extruder_stack')
        mocked_extruder_stack.getId = MagicMock(return_value='0')
        mocked_extruder_stack.getProperty = MagicMock(side_effect=self.getPropertySideEffect)
        build_volume._global_container_stack = mocked_global_stack
        resulting_polygon = Polygon.approximatedCircle(PRIME_CLEARANCE)
        resulting_polygon = resulting_polygon.getMinkowskiHull(Polygon.approximatedCircle(12))
        resulting_polygon = resulting_polygon.translate(25, -50)
        assert build_volume._computeDisallowedAreasPrimeBlob(12, [mocked_extruder_stack]) == {'0': [resulting_polygon]}

class TestCalculateExtraZClearance:
    setting_property_dict = {'retraction_hop': {'value': 12}, 'retraction_hop_enabled': {'value': True}, 'material_shrinkage_percentage': {'value': 100.0}, 'material_shrinkage_percentage_xy': {'value': 100.0}, 'material_shrinkage_percentage_z': {'value': 100.0}}

    def getPropertySideEffect(*args, **kwargs):
        if False:
            print('Hello World!')
        properties = TestCalculateExtraZClearance.setting_property_dict.get(args[1])
        if properties:
            return properties.get(args[2])

    def test_noContainerStack(self, build_volume: BuildVolume):
        if False:
            while True:
                i = 10
        assert build_volume._calculateExtraZClearance([]) == 0

    def test_withRetractionHop(self, build_volume: BuildVolume):
        if False:
            print('Hello World!')
        mocked_global_stack = MagicMock(name='mocked_global_stack')
        mocked_extruder = MagicMock()
        mocked_extruder.getProperty = MagicMock(side_effect=self.getPropertySideEffect)
        build_volume._global_container_stack = mocked_global_stack
        assert build_volume._calculateExtraZClearance([mocked_extruder]) == 12

    def test_withoutRetractionHop(self, build_volume: BuildVolume):
        if False:
            print('Hello World!')
        mocked_global_stack = MagicMock(name='mocked_global_stack')
        mocked_extruder = MagicMock()
        mocked_extruder.getProperty = MagicMock(side_effect=self.getPropertySideEffect)
        build_volume._global_container_stack = mocked_global_stack
        patched_dictionary = self.setting_property_dict.copy()
        patched_dictionary['retraction_hop_enabled'] = {'value': False}
        with patch.dict(self.setting_property_dict, patched_dictionary):
            assert build_volume._calculateExtraZClearance([mocked_extruder]) == 0

class TestRebuild:
    setting_property_dict = {'material_shrinkage_percentage': {'value': 100.0}, 'material_shrinkage_percentage_xy': {'value': 100.0}, 'material_shrinkage_percentage_z': {'value': 100.0}}

    def getPropertySideEffect(*args, **kwargs):
        if False:
            while True:
                i = 10
        properties = TestCalculateExtraZClearance.setting_property_dict.get(args[1])
        if properties:
            return properties.get(args[2])

    def test_zeroWidthHeightDepth(self, build_volume: BuildVolume):
        if False:
            return 10
        build_volume.rebuild()
        assert build_volume.getMeshData() is None

    def test_engineIsNotRead(self, build_volume: BuildVolume):
        if False:
            for i in range(10):
                print('nop')
        build_volume.setWidth(10)
        build_volume.setHeight(10)
        build_volume.setDepth(10)
        build_volume.rebuild()
        assert build_volume.getMeshData() is None

    def test_noGlobalStack(self, build_volume: BuildVolume):
        if False:
            i = 10
            return i + 15
        build_volume.setWidth(10)
        build_volume.setHeight(10)
        build_volume.setDepth(10)
        build_volume._onEngineCreated()
        build_volume.rebuild()
        assert build_volume.getMeshData() is None

    def test_updateBoundingBox(self, build_volume: BuildVolume):
        if False:
            for i in range(10):
                print('nop')
        build_volume.setWidth(10)
        build_volume.setHeight(10)
        build_volume.setDepth(10)
        mocked_global_stack = MagicMock()
        mocked_global_stack.getProperty = MagicMock(side_effect=self.getPropertySideEffect)
        build_volume._global_container_stack = mocked_global_stack
        build_volume.getEdgeDisallowedSize = MagicMock(return_value=0)
        build_volume.updateNodeBoundaryCheck = MagicMock()
        build_volume._onEngineCreated()
        build_volume.rebuild()
        bounding_box = build_volume.getBoundingBox()
        assert bounding_box.minimum == Vector(-5.0, -1.0, -5.0)
        assert bounding_box.maximum == Vector(5.0, 10.0, 5.0)

class TestUpdateMachineSizeProperties:
    setting_property_dict = {'machine_width': {'value': 50}, 'machine_depth': {'value': 100}, 'machine_height': {'value': 200}, 'machine_shape': {'value': 'DERP!'}, 'material_shrinkage_percentage': {'value': 100.0}, 'material_shrinkage_percentage_xy': {'value': 100.0}, 'material_shrinkage_percentage_z': {'value': 100.0}}

    def getPropertySideEffect(*args, **kwargs):
        if False:
            while True:
                i = 10
        properties = TestUpdateMachineSizeProperties.setting_property_dict.get(args[1])
        if properties:
            return properties.get(args[2])

    def test_noGlobalStack(self, build_volume: BuildVolume):
        if False:
            i = 10
            return i + 15
        build_volume._updateMachineSizeProperties()
        assert build_volume._width == 0
        assert build_volume._height == 0
        assert build_volume._depth == 0
        assert build_volume._shape == ''

    def test_happy(self, build_volume: BuildVolume):
        if False:
            print('Hello World!')
        mocked_global_stack = MagicMock(name='mocked_global_stack')
        mocked_global_stack.getProperty = MagicMock(side_effect=self.getPropertySideEffect)
        build_volume._global_container_stack = mocked_global_stack
        build_volume._updateMachineSizeProperties()
        assert build_volume._width == 50
        assert build_volume._height == 200
        assert build_volume._depth == 100
        assert build_volume._shape == 'DERP!'

class TestGetEdgeDisallowedSize:
    setting_property_dict = {}
    bed_adhesion_size = 1

    @pytest.fixture()
    def build_volume(self, build_volume):
        if False:
            return 10
        build_volume._calculateBedAdhesionSize = MagicMock(return_value=1)
        return build_volume

    def getPropertySideEffect(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        properties = TestGetEdgeDisallowedSize.setting_property_dict.get(args[1])
        if properties:
            return properties.get(args[2])

    def createMockedStack(self):
        if False:
            for i in range(10):
                print('nop')
        mocked_global_stack = MagicMock(name='mocked_global_stack')
        mocked_global_stack.getProperty = MagicMock(side_effect=self.getPropertySideEffect)
        return mocked_global_stack

    def test_noGlobalContainer(self, build_volume: BuildVolume):
        if False:
            i = 10
            return i + 15
        assert build_volume.getEdgeDisallowedSize() == 0

    def test_unknownAdhesion(self, build_volume: BuildVolume):
        if False:
            i = 10
            return i + 15
        build_volume._global_container_stack = self.createMockedStack()
        with patch('cura.Settings.ExtruderManager.ExtruderManager.getInstance'):
            build_volume.getEdgeDisallowedSize()

    def test_oneAtATime(self, build_volume: BuildVolume):
        if False:
            return 10
        build_volume._global_container_stack = self.createMockedStack()
        with patch('cura.Settings.ExtruderManager.ExtruderManager.getInstance'):
            with patch.dict(self.setting_property_dict, {'print_sequence': {'value': 'one_at_a_time'}}):
                assert build_volume.getEdgeDisallowedSize() == 0.1