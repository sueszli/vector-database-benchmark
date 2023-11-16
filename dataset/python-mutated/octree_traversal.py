import open3d as o3d
import numpy as np

def f_traverse(node, node_info):
    if False:
        print('Hello World!')
    early_stop = False
    if isinstance(node, o3d.geometry.OctreeInternalNode):
        if isinstance(node, o3d.geometry.OctreeInternalPointNode):
            n = 0
            for child in node.children:
                if child is not None:
                    n += 1
            print('{}{}: Internal node at depth {} has {} children and {} points ({})'.format('    ' * node_info.depth, node_info.child_index, node_info.depth, n, len(node.indices), node_info.origin))
            early_stop = len(node.indices) < 250
    elif isinstance(node, o3d.geometry.OctreeLeafNode):
        if isinstance(node, o3d.geometry.OctreePointColorLeafNode):
            print('{}{}: Leaf node at depth {} has {} points with origin {}'.format('    ' * node_info.depth, node_info.child_index, node_info.depth, len(node.indices), node_info.origin))
    else:
        raise NotImplementedError('Node type not recognized!')
    return early_stop
if __name__ == '__main__':
    N = 2000
    armadillo_data = o3d.data.ArmadilloMesh()
    pcd = o3d.io.read_triangle_mesh(armadillo_data.path).sample_points_poisson_disk(N)
    pcd.scale(1 / np.max(pcd.get_max_bound() - pcd.get_min_bound()), center=pcd.get_center())
    pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(N, 3)))
    octree = o3d.geometry.Octree(max_depth=4)
    octree.convert_from_point_cloud(pcd, size_expand=0.01)
    print('Displaying input octree ...')
    o3d.visualization.draw([octree])
    print('Traversing octree ...')
    octree.traverse(f_traverse)