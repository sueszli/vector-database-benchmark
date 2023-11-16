import open3d as o3d
import numpy as np
import argparse

class PoseGraphWrapper:

    def __init__(self, dict_nodes=None, dict_edges=None):
        if False:
            print('Hello World!')
        '\n        \\input dict_nodes: index -> 6x1 pose\n        \\input dict_edges: Tuple(index, index) -> Tuple(6x1 pose, 6x6 information, is_loop)\n        They are the major maintained members.\n        Wrapped PoseGraph is the intermediate member that needs to be synchronized via _dicts2graph before performing batch ops.\n        '
        self.dict_nodes = {} if dict_nodes is None else dict_nodes
        self.dict_edges = {} if dict_edges is None else dict_edges
        self.pose_graph = o3d.pipelines.registration.PoseGraph()

    def add_node(self, index, pose):
        if False:
            print('Hello World!')
        if index in self.dict_nodes:
            print('Warning: node {} already exists, overwriting'.format(index))
        self.dict_nodes[index] = pose

    def add_edge(self, index_src, index_dst, pose_src2dst, info_src2dst, is_loop):
        if False:
            i = 10
            return i + 15
        if (index_src, index_dst) in self.dict_edges:
            print('Warning: edge ({}, {}) already exists, overwriting'.format(index_src, index_dst))
        self.dict_edges[index_src, index_dst] = (pose_src2dst, info_src2dst, is_loop)

    def _dicts2graph(self):
        if False:
            return 10
        pose_graph = o3d.pipelines.registration.PoseGraph()
        n_nodes = len(self.dict_nodes)
        if n_nodes < 3:
            print('Only {} nodes found, abort pose graph construction'.format(n_nodes))
        nodes2indices = {}
        for (i, k) in enumerate(sorted(self.dict_nodes.keys())):
            nodes2indices[i] = k
        for i in range(n_nodes):
            k = nodes2indices[i]
            pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(self.dict_nodes[k]))
        for (i, j) in self.dict_edges:
            if not i in self.dict_nodes or not j in self.dict_nodes:
                print(f'Edge node ({i} {j}) not found, abort pose graph construction')
            (trans, info, is_loop) = self.dict_edges[i, j]
            ki = nodes2indices[i]
            kj = nodes2indices[j]
            pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(ki, kj, trans, info, uncertain=is_loop))
        return pose_graph

    def _graph2dicts(self):
        if False:
            return 10
        nodes = self.pose_graph.nodes
        edges = self.pose_graph.edges
        dict_nodes = {}
        dict_edges = {}
        for (i, node) in enumerate(nodes):
            dict_nodes[i] = node.pose
        for edge in edges:
            dict_edges[edge.source_node_id, edge.target_node_id] = (edge.transformation, edge.information, edge.uncertain)
        return (dict_nodes, dict_edges)

    def solve_(self, dist_threshold=0.07, preference_loop_closure=0.1):
        if False:
            print('Hello World!')
        self.pose_graph = self._dicts2graph()
        method = o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt()
        criteria = o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria()
        option = o3d.pipelines.registration.GlobalOptimizationOption(max_correspondence_distance=dist_threshold, edge_prune_threshold=0.25, preference_loop_closure=preference_loop_closure, reference_node=0)
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
        o3d.pipelines.registration.global_optimization(self.pose_graph, method, criteria, option)
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Info)
        (self.dict_nodes, self.dict_edges) = self._graph2dicts()

    @staticmethod
    def load(fname):
        if False:
            while True:
                i = 10
        ans = PoseGraphWrapper()
        ans.pose_graph = o3d.io.read_pose_graph(fname)
        (ans.dict_nodes, ans.dict_edges) = ans._graph2dicts()
        return ans

    def save(self, fname):
        if False:
            i = 10
            return i + 15
        self.pose_graph = self._dicts2graph()
        o3d.io.write_pose_graph(fname, self.pose_graph)

    def export_extrinsics(self):
        if False:
            print('Hello World!')
        return [np.linalg.inv(self.dict_nodes[k]) for k in sorted(self.dict_nodes)]
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_pose_graph', required=True)
    args = parser.parse_args()
    pose_graph = PoseGraphWrapper.load(args.path_pose_graph)
    pose_graph.solve()
    pose_graph.save('test.json')
    pose_graph = PoseGraphWrapper.load('test.json')
    pose_graph.solve()