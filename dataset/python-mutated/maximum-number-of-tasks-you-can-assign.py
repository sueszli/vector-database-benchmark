from sortedcontainers import SortedList

class Solution(object):

    def maxTaskAssign(self, tasks, workers, pills, strength):
        if False:
            print('Hello World!')
        '\n        :type tasks: List[int]\n        :type workers: List[int]\n        :type pills: int\n        :type strength: int\n        :rtype: int\n        '

        def check(tasks, workers, pills, strength, x):
            if False:
                return 10
            t = SortedList(tasks[:x])
            for worker in workers[-x:]:
                i = t.bisect_right(worker) - 1
                if i != -1:
                    t.pop(i)
                    continue
                if pills:
                    i = t.bisect_right(worker + strength) - 1
                    if i != -1:
                        t.pop(i)
                        pills -= 1
                        continue
                return False
            return True
        tasks.sort()
        workers.sort()
        (left, right) = (1, min(len(workers), len(tasks)))
        while left <= right:
            mid = left + (right - left) // 2
            if not check(tasks, workers, pills, strength, mid):
                right = mid - 1
            else:
                left = mid + 1
        return right
from sortedcontainers import SortedList

class Solution2(object):

    def maxTaskAssign(self, tasks, workers, pills, strength):
        if False:
            print('Hello World!')
        '\n        :type tasks: List[int]\n        :type workers: List[int]\n        :type pills: int\n        :type strength: int\n        :rtype: int\n        '

        def check(tasks, workers, pills, strength, x):
            if False:
                print('Hello World!')
            w = SortedList(workers[-x:])
            for task in tasks[-x:]:
                i = w.bisect_left(task)
                if i != len(w):
                    w.pop(i)
                    continue
                if pills:
                    i = w.bisect_left(task - strength)
                    if i != len(w):
                        w.pop(i)
                        pills -= 1
                        continue
                return False
            return True
        tasks.sort(reverse=True)
        workers.sort()
        (left, right) = (1, min(len(workers), len(tasks)))
        while left <= right:
            mid = left + (right - left) // 2
            if not check(tasks, workers, pills, strength, mid):
                right = mid - 1
            else:
                left = mid + 1
        return right
import bisect

class Solution3(object):

    def maxTaskAssign(self, tasks, workers, pills, strength):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type tasks: List[int]\n        :type workers: List[int]\n        :type pills: int\n        :type strength: int\n        :rtype: int\n        '

        def check(tasks, workers, pills, strength, x):
            if False:
                i = 10
                return i + 15
            t = tasks[:x]
            for worker in workers[-x:]:
                i = bisect.bisect_right(t, worker) - 1
                if i != -1:
                    t.pop(i)
                    continue
                if pills:
                    i = bisect.bisect_right(t, worker + strength) - 1
                    if i != -1:
                        t.pop(i)
                        pills -= 1
                        continue
                return False
            return True
        tasks.sort()
        workers.sort()
        (left, right) = (1, min(len(workers), len(tasks)))
        while left <= right:
            mid = left + (right - left) // 2
            if not check(tasks, workers, pills, strength, mid):
                right = mid - 1
            else:
                left = mid + 1
        return right
import bisect

class Solution4(object):

    def maxTaskAssign(self, tasks, workers, pills, strength):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type tasks: List[int]\n        :type workers: List[int]\n        :type pills: int\n        :type strength: int\n        :rtype: int\n        '

        def check(tasks, workers, pills, strength, x):
            if False:
                for i in range(10):
                    print('nop')
            w = workers[-x:]
            for task in tasks[-x:]:
                i = bisect.bisect_left(w, task)
                if i != len(w):
                    w.pop(i)
                    continue
                if pills:
                    i = bisect.bisect_left(w, task - strength)
                    if i != len(w):
                        w.pop(i)
                        pills -= 1
                        continue
                return False
            return True
        tasks.sort(reverse=True)
        workers.sort()
        (left, right) = (1, min(len(workers), len(tasks)))
        while left <= right:
            mid = left + (right - left) // 2
            if not check(tasks, workers, pills, strength, mid):
                right = mid - 1
            else:
                left = mid + 1
        return right