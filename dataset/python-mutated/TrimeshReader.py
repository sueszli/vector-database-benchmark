from typing import Any, List, Union, TYPE_CHECKING
import numpy
import os.path
import trimesh
from UM.Mesh.MeshData import MeshData, calculateNormalsFromIndexedVertices
from UM.Mesh.MeshReader import MeshReader
from UM.MimeTypeDatabase import MimeTypeDatabase, MimeType
from UM.Scene.GroupDecorator import GroupDecorator
from cura.CuraApplication import CuraApplication
from cura.Scene.BuildPlateDecorator import BuildPlateDecorator
from cura.Scene.ConvexHullDecorator import ConvexHullDecorator
from cura.Scene.CuraSceneNode import CuraSceneNode
from cura.Scene.SliceableObjectDecorator import SliceableObjectDecorator
if TYPE_CHECKING:
    from UM.Scene.SceneNode import SceneNode

class TrimeshReader(MeshReader):
    """Class that leverages Trimesh to import files."""

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__()
        self._supported_extensions = ['.dae', '.gltf', '.glb', '.ply', '.zae']
        MimeTypeDatabase.addMimeType(MimeType(name='model/vnd.collada+xml', comment='COLLADA Digital Asset Exchange', suffixes=['dae']))
        MimeTypeDatabase.addMimeType(MimeType(name='model/gltf-binary', comment='glTF Binary', suffixes=['glb']))
        MimeTypeDatabase.addMimeType(MimeType(name='model/gltf+json', comment='glTF Embedded JSON', suffixes=['gltf']))
        MimeTypeDatabase.addMimeType(MimeType(name='application/x-ply', comment='Stanford Triangle Format', suffixes=['ply']))
        MimeTypeDatabase.addMimeType(MimeType(name='model/vnd.collada+xml+zip', comment='Compressed COLLADA Digital Asset Exchange', suffixes=['zae']))

    def _read(self, file_name: str) -> Union['SceneNode', List['SceneNode']]:
        if False:
            for i in range(10):
                print('nop')
        "Reads a file using Trimesh.\n\n        :param file_name: The file path. This is assumed to be one of the file\n        types that Trimesh can read. It will not be checked again.\n        :return: A scene node that contains the file's contents.\n        "
        if file_name.lower().endswith('.gltf'):
            mesh_or_scene = trimesh.load(open(file_name, 'r', encoding='utf-8'), file_type='gltf')
        else:
            mesh_or_scene = trimesh.load(file_name)
        meshes = []
        if isinstance(mesh_or_scene, trimesh.Trimesh):
            meshes = [mesh_or_scene]
        elif isinstance(mesh_or_scene, trimesh.Scene):
            meshes = [mesh for mesh in mesh_or_scene.geometry.values()]
        active_build_plate = CuraApplication.getInstance().getMultiBuildPlateModel().activeBuildPlate
        nodes = []
        for mesh in meshes:
            if not isinstance(mesh, trimesh.Trimesh):
                continue
            mesh.merge_vertices()
            mesh.remove_unreferenced_vertices()
            mesh.fix_normals()
            mesh_data = self._toMeshData(mesh, file_name)
            file_base_name = os.path.basename(file_name)
            new_node = CuraSceneNode()
            new_node.setMeshData(mesh_data)
            new_node.setSelectable(True)
            new_node.setName(file_base_name if len(meshes) == 1 else '{file_base_name} {counter}'.format(file_base_name=file_base_name, counter=str(len(nodes) + 1)))
            new_node.addDecorator(BuildPlateDecorator(active_build_plate))
            new_node.addDecorator(SliceableObjectDecorator())
            nodes.append(new_node)
        if len(nodes) == 1:
            return nodes[0]
        group_node = CuraSceneNode()
        group_node.addDecorator(GroupDecorator())
        group_node.addDecorator(ConvexHullDecorator())
        group_node.addDecorator(BuildPlateDecorator(active_build_plate))
        for node in nodes:
            node.setParent(group_node)
        return group_node

    def _toMeshData(self, tri_node: trimesh.base.Trimesh, file_name: str='') -> MeshData:
        if False:
            for i in range(10):
                print('nop')
        "Converts a Trimesh to Uranium's MeshData.\n\n        :param tri_node: A Trimesh containing the contents of a file that was just read.\n        :param file_name: The full original filename used to watch for changes\n        :return: Mesh data from the Trimesh in a way that Uranium can understand it.\n        "
        tri_faces = tri_node.faces
        tri_vertices = tri_node.vertices
        indices_list = []
        vertices_list = []
        index_count = 0
        face_count = 0
        for tri_face in tri_faces:
            face = []
            for tri_index in tri_face:
                vertices_list.append(tri_vertices[tri_index])
                face.append(index_count)
                index_count += 1
            indices_list.append(face)
            face_count += 1
        vertices = numpy.asarray(vertices_list, dtype=numpy.float32)
        indices = numpy.asarray(indices_list, dtype=numpy.int32)
        normals = calculateNormalsFromIndexedVertices(vertices, indices, face_count)
        mesh_data = MeshData(vertices=vertices, indices=indices, normals=normals, file_name=file_name)
        return mesh_data