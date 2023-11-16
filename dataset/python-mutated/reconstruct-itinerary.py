import collections

class Solution(object):

    def findItinerary(self, tickets):
        if False:
            return 10
        '\n        :type tickets: List[List[str]]\n        :rtype: List[str]\n        '
        adj = collections.defaultdict(list)
        for ticket in tickets:
            adj[ticket[0]].append(ticket[1])
        for x in adj.itervalues():
            x.sort(reverse=True)
        origin = 'JFK'
        result = []
        stk = [origin]
        while stk:
            while adj[stk[-1]]:
                stk.append(adj[stk[-1]].pop())
            result.append(stk.pop())
        result.reverse()
        return result
import collections

class Solution2(object):

    def findItinerary(self, tickets):
        if False:
            return 10
        '\n        :type tickets: List[List[str]]\n        :rtype: List[str]\n        '

        def route_helper(origin, ticket_cnt, graph, ans):
            if False:
                return 10
            if ticket_cnt == 0:
                return True
            for (i, (dest, valid)) in enumerate(graph[origin]):
                if valid:
                    graph[origin][i][1] = False
                    ans.append(dest)
                    if route_helper(dest, ticket_cnt - 1, graph, ans):
                        return ans
                    ans.pop()
                    graph[origin][i][1] = True
            return False
        graph = collections.defaultdict(list)
        for ticket in tickets:
            graph[ticket[0]].append([ticket[1], True])
        for k in graph.keys():
            graph[k].sort()
        origin = 'JFK'
        ans = [origin]
        route_helper(origin, len(tickets), graph, ans)
        return ans