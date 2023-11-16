class Solution(object):

    def canVisitAllRooms(self, rooms):
        if False:
            return 10
        '\n        :type rooms: List[List[int]]\n        :rtype: bool\n        '
        lookup = set([0])
        stack = [0]
        while stack:
            node = stack.pop()
            for nei in rooms[node]:
                if nei not in lookup:
                    lookup.add(nei)
                    if len(lookup) == len(rooms):
                        return True
                    stack.append(nei)
        return len(lookup) == len(rooms)