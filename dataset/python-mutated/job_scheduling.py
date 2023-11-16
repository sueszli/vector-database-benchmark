"""
Python program for weighted job scheduling using Dynamic
Programming and Binary Search
"""

class Job:
    """
    Class to represent a job
    """

    def __init__(self, start, finish, profit):
        if False:
            while True:
                i = 10
        self.start = start
        self.finish = finish
        self.profit = profit

def binary_search(job, start_index):
    if False:
        print('Hello World!')
    '\n    A Binary Search based function to find the latest job\n    (before current job) that doesn\'t conflict with current\n    job.  "index" is index of the current job.  This function\n    returns -1 if all jobs before index conflict with it.\n    The array jobs[] is sorted in increasing order of finish\n    time.\n    '
    left = 0
    right = start_index - 1
    while left <= right:
        mid = (left + right) // 2
        if job[mid].finish <= job[start_index].start:
            if job[mid + 1].finish <= job[start_index].start:
                left = mid + 1
            else:
                return mid
        else:
            right = mid - 1
    return -1

def schedule(job):
    if False:
        for i in range(10):
            print('nop')
    '\n    The main function that returns the maximum possible\n    profit from given array of jobs\n    '
    job = sorted(job, key=lambda j: j.finish)
    length = len(job)
    table = [0 for _ in range(length)]
    table[0] = job[0].profit
    for i in range(1, length):
        incl_prof = job[i].profit
        pos = binary_search(job, i)
        if pos != -1:
            incl_prof += table[pos]
        table[i] = max(incl_prof, table[i - 1])
    return table[length - 1]