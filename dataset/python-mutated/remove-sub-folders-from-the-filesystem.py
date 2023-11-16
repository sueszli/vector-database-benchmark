import collections
import itertools

class Solution(object):

    def removeSubfolders(self, folder):
        if False:
            print('Hello World!')
        '\n        :type folder: List[str]\n        :rtype: List[str]\n        '

        def dfs(curr, path, result):
            if False:
                for i in range(10):
                    print('nop')
            if '_end' in curr:
                result.append('/' + '/'.join(path))
                return
            for c in curr:
                if c == '_end':
                    continue
                path.append(c)
                dfs(curr[c], path, result)
                path.pop()
        _trie = lambda : collections.defaultdict(_trie)
        trie = _trie()
        for f in folder:
            f_list = f.split('/')
            reduce(dict.__getitem__, itertools.islice(f_list, 1, len(f_list)), trie).setdefault('_end')
        result = []
        dfs(trie, [], result)
        return result