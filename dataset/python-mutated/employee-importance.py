import collections
"\n# Employee info\nclass Employee(object):\n    def __init__(self, id, importance, subordinates):\n        # It's the unique id of each node.\n        # unique id of this employee\n        self.id = id\n        # the importance value of this employee\n        self.importance = importance\n        # the id of direct subordinates\n        self.subordinates = subordinates\n"

class Solution(object):

    def getImportance(self, employees, id):
        if False:
            i = 10
            return i + 15
        '\n        :type employees: Employee\n        :type id: int\n        :rtype: int\n        '
        if employees[id - 1] is None:
            return 0
        result = employees[id - 1].importance
        for id in employees[id - 1].subordinates:
            result += self.getImportance(employees, id)
        return result

class Solution2(object):

    def getImportance(self, employees, id):
        if False:
            return 10
        '\n        :type employees: Employee\n        :type id: int\n        :rtype: int\n        '
        (result, q) = (0, collections.deque([id]))
        while q:
            curr = q.popleft()
            employee = employees[curr - 1]
            result += employee.importance
            for id in employee.subordinates:
                q.append(id)
        return result