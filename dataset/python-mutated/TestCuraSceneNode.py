from UM.Math.Polygon import Polygon
from UM.Scene.SceneNodeDecorator import SceneNodeDecorator
from cura.Scene.CuraSceneNode import CuraSceneNode
import pytest
from unittest.mock import patch

class MockedConvexHullDecorator(SceneNodeDecorator):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()

    def getConvexHull(self):
        if False:
            return 10
        return Polygon([[5, 5], [-5, 5], [-5, -5], [5, -5]])

    def getPrintingArea(self):
        if False:
            return 10
        return Polygon([[5, 5], [-5, 5], [-5, -5], [5, -5]])

class InvalidConvexHullDecorator(SceneNodeDecorator):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()

    def getConvexHull(self):
        if False:
            for i in range(10):
                print('nop')
        return Polygon()

@pytest.fixture()
def cura_scene_node():
    if False:
        i = 10
        return i + 15
    with patch('cura.Scene.CuraSceneNode.SettingOverrideDecorator', SceneNodeDecorator):
        return CuraSceneNode()

class TestCollidesWithAreas:

    def test_noConvexHull(self, cura_scene_node):
        if False:
            while True:
                i = 10
        assert not cura_scene_node.collidesWithAreas([Polygon([[10, 10], [-10, 10], [-10, -10], [10, -10]])])

    def test_convexHullIntersects(self, cura_scene_node):
        if False:
            return 10
        cura_scene_node.addDecorator(MockedConvexHullDecorator())
        assert cura_scene_node.collidesWithAreas([Polygon([[10, 10], [-10, 10], [-10, -10], [10, -10]])])

    def test_convexHullNoIntersection(self, cura_scene_node):
        if False:
            print('Hello World!')
        cura_scene_node.addDecorator(MockedConvexHullDecorator())
        assert not cura_scene_node.collidesWithAreas([Polygon([[60, 60], [40, 60], [40, 40], [60, 40]])])

    def test_invalidConvexHull(self, cura_scene_node):
        if False:
            print('Hello World!')
        cura_scene_node.addDecorator(InvalidConvexHullDecorator())
        assert not cura_scene_node.collidesWithAreas([Polygon([[10, 10], [-10, 10], [-10, -10], [10, -10]])])

def test_outsideBuildArea(cura_scene_node):
    if False:
        while True:
            i = 10
    cura_scene_node.setOutsideBuildArea(True)
    assert cura_scene_node.isOutsideBuildArea