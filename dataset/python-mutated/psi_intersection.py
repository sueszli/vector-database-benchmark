import logging
import threading
from bigdl.ppml.utils.log4Error import invalidOperationError

class PsiIntersection(object):

    def __init__(self, max_collection=1) -> None:
        if False:
            while True:
                i = 10
        self.intersection = []
        self._thread_intersection = []
        self.max_collection = int(max_collection)
        self.condition = threading.Condition()
        self._lock = threading.Lock()
        self.collection = []

    def find_intersection(self, a, b):
        if False:
            for i in range(10):
                print('nop')
        return list(set(a) & set(b))

    def add_collection(self, collection):
        if False:
            i = 10
            return i + 15
        with self._lock:
            invalidOperationError(len(self.collection) < self.max_collection, f'PSI collection is full, got: {len(self.collection)}/{self.max_collection}')
            self.collection.append(collection)
            logging.debug(f'PSI got collection {len(self.collection)}/{self.max_collection}')
            if len(self.collection) == self.max_collection:
                current_intersection = self.collection[0]
                for i in range(1, len(self.collection)):
                    current_intersection = self.find_intersection(current_intersection, self.collection[i])
                self.intersection = current_intersection
                self.collection.clear()

    def get_intersection(self):
        if False:
            for i in range(10):
                print('nop')
        with self._lock:
            return self.intersection