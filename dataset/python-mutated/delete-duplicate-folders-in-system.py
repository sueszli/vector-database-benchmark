import collections

class Solution(object):

    def deleteDuplicateFolder(self, paths):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type paths: List[List[str]]\n        :rtype: List[List[str]]\n        '

        def mark(node, lookup, node_ids):
            if False:
                i = 10
                return i + 15
            id_pairs = []
            for (subfolder_id, child) in node.iteritems():
                if child == '_del':
                    continue
                id_pairs.append((subfolder_id, mark(child, lookup, node_ids)))
            id_pairs.sort()
            node_id = node_ids[tuple(id_pairs)]
            if node_id:
                if node_id in lookup:
                    lookup[node_id]['_del']
                    node['_del']
                else:
                    lookup[node_id] = node
            return node_id

        def sweep(node, id_folders, path, result):
            if False:
                return 10
            if path:
                result.append([id_folders[i] for i in path])
            for (subfolder_id, child) in node.iteritems():
                if '_del' in child:
                    continue
                path.append(subfolder_id)
                sweep(child, id_folders, path, result)
                path.pop()
        _trie = lambda : collections.defaultdict(_trie)
        trie = _trie()
        folder_ids = collections.defaultdict()
        folder_ids.default_factory = folder_ids.__len__
        id_folders = {}
        for path in paths:
            node = trie
            for folder in path:
                if folder_ids[folder] not in id_folders:
                    id_folders[folder_ids[folder]] = folder
                node = node[folder_ids[folder]]
        node_ids = collections.defaultdict()
        node_ids.default_factory = node_ids.__len__
        mark(trie, {}, node_ids)
        result = []
        sweep(trie, id_folders, [], result)
        return result
import collections

class Solution2(object):

    def deleteDuplicateFolder(self, paths):
        if False:
            while True:
                i = 10
        '\n        :type paths: List[List[str]]\n        :rtype: List[List[str]]\n        '

        def mark(node, lookup):
            if False:
                for i in range(10):
                    print('nop')
            serialized_tree = '(' + ''.join((subfolder + mark(child, lookup) for (subfolder, child) in sorted(node.iteritems()) if child != '_del')) + ')'
            if serialized_tree != '()':
                if serialized_tree in lookup:
                    lookup[serialized_tree]['_del']
                    node['_del']
                else:
                    lookup[serialized_tree] = node
            return serialized_tree

        def sweep(node, path, result):
            if False:
                print('Hello World!')
            if path:
                result.append(path[:])
            for (subfolder, child) in node.iteritems():
                if '_del' in child:
                    continue
                path.append(subfolder)
                sweep(child, path, result)
                path.pop()
        _trie = lambda : collections.defaultdict(_trie)
        trie = _trie()
        for path in paths:
            reduce(dict.__getitem__, path, trie)
        mark(trie, {})
        result = []
        sweep(trie, [], result)
        return result