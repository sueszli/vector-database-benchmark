from sortedcontainers import SortedList

class TodoList(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.__tasks = []
        self.__user_task_ids = collections.defaultdict(SortedList)

    def addTask(self, userId, taskDescription, dueDate, tags):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type userId: int\n        :type taskDescription: str\n        :type dueDate: int\n        :type tags: List[str]\n        :rtype: int\n        '
        self.__tasks.append([dueDate, taskDescription, set(tags)])
        self.__user_task_ids[userId].add((dueDate, len(self.__tasks)))
        return len(self.__tasks)

    def getAllTasks(self, userId):
        if False:
            while True:
                i = 10
        '\n        :type userId: int\n        :rtype: List[str]\n        '
        if userId not in self.__user_task_ids:
            return []
        return [self.__tasks[i - 1][1] for (_, i) in self.__user_task_ids[userId]]

    def getTasksForTag(self, userId, tag):
        if False:
            print('Hello World!')
        '\n        :type userId: int\n        :type tag: str\n        :rtype: List[str]\n        '
        if userId not in self.__user_task_ids:
            return []
        return [self.__tasks[i - 1][1] for (_, i) in self.__user_task_ids[userId] if tag in self.__tasks[i - 1][-1]]

    def completeTask(self, userId, taskId):
        if False:
            return 10
        '\n        :type userId: int\n        :type taskId: int\n        :rtype: None\n        '
        if not (taskId - 1 < len(self.__tasks) and userId in self.__user_task_ids):
            return
        self.__user_task_ids[userId].discard((self.__tasks[taskId - 1][0], taskId))
from sortedcontainers import SortedList

class TodoList2(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.__tasks = []
        self.__user_task_ids = collections.defaultdict(SortedList)

    def addTask(self, userId, taskDescription, dueDate, tags):
        if False:
            return 10
        '\n        :type userId: int\n        :type taskDescription: str\n        :type dueDate: int\n        :type tags: List[str]\n        :rtype: int\n        '
        self.__tasks.append([dueDate, taskDescription, set(tags)])
        self.__user_task_ids[userId].add((dueDate, len(self.__tasks)))
        for tag in self.__tasks[-1][-1]:
            self.__user_task_ids[userId, tag].add((dueDate, len(self.__tasks)))
        return len(self.__tasks)

    def getAllTasks(self, userId):
        if False:
            return 10
        '\n        :type userId: int\n        :rtype: List[str]\n        '
        if userId not in self.__user_task_ids:
            return []
        return [self.__tasks[i - 1][1] for (_, i) in self.__user_task_ids[userId]]

    def getTasksForTag(self, userId, tag):
        if False:
            i = 10
            return i + 15
        '\n        :type userId: int\n        :type tag: str\n        :rtype: List[str]\n        '
        if (userId, tag) not in self.__user_task_ids:
            return []
        return [self.__tasks[i - 1][1] for (_, i) in self.__user_task_ids[userId, tag]]

    def completeTask(self, userId, taskId):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type userId: int\n        :type taskId: int\n        :rtype: None\n        '
        if not (taskId - 1 < len(self.__tasks) and userId in self.__user_task_ids):
            return
        self.__user_task_ids[userId].discard((self.__tasks[taskId - 1][0], taskId))
        for tag in self.__tasks[taskId - 1][-1]:
            self.__user_task_ids[userId, tag].discard((self.__tasks[taskId - 1][0], taskId))