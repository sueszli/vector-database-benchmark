from collections import deque
from enum import Enum

class State(Enum):
    unvisited = 0
    visited = 1

class Graph(object):

    def bfs(self, source, dest):
        if False:
            i = 10
            return i + 15
        if source is None:
            return False
        queue = deque()
        queue.append(source)
        source.visit_state = State.visited
        while queue:
            node = queue.popleft()
            print(node)
            if dest is node:
                return True
            for adjacent_node in node.adj_nodes.values():
                if adjacent_node.visit_state == State.unvisited:
                    queue.append(adjacent_node)
                    adjacent_node.visit_state = State.visited
        return False

class Person(object):

    def __init__(self, id, name):
        if False:
            i = 10
            return i + 15
        self.id = id
        self.name = name
        self.friend_ids = []

class LookupService(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.lookup = {}

    def get_person(self, person_id):
        if False:
            for i in range(10):
                print('nop')
        person_server = self.lookup[person_id]
        return person_server.people[person_id]

class PersonServer(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self.people = {}

    def get_people(self, ids):
        if False:
            print('Hello World!')
        results = []
        for id in ids:
            if id in self.people:
                results.append(self.people[id])
        return results

class UserGraphService(object):

    def __init__(self, person_ids, lookup):
        if False:
            print('Hello World!')
        self.lookup = lookup
        self.person_ids = person_ids
        self.visited_ids = set()

    def bfs(self, source, dest):
        if False:
            return 10
        pass