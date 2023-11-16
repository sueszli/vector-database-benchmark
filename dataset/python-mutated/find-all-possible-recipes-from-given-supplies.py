import collections
import itertools

class Solution(object):

    def findAllRecipes(self, recipes, ingredients, supplies):
        if False:
            while True:
                i = 10
        '\n        :type recipes: List[str]\n        :type ingredients: List[List[str]]\n        :type supplies: List[str]\n        :rtype: List[str]\n        '
        indegree = collections.defaultdict(int)
        adj = collections.defaultdict(list)
        for (r, ingredient) in itertools.izip(recipes, ingredients):
            indegree[r] = len(ingredient)
            for ing in ingredient:
                adj[ing].append(r)
        result = []
        recipes = set(recipes)
        q = supplies
        while q:
            new_q = []
            for u in q:
                if u in recipes:
                    result.append(u)
                for v in adj[u]:
                    indegree[v] -= 1
                    if not indegree[v]:
                        new_q.append(v)
            q = new_q
        return result