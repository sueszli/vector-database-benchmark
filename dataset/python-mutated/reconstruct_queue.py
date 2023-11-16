def reconstruct_queue(people):
    if False:
        i = 10
        return i + 15
    '\n    :type people: List[List[int]]\n    :rtype: List[List[int]]\n    '
    queue = []
    people.sort(key=lambda x: (-x[0], x[1]))
    for (h, k) in people:
        queue.insert(k, [h, k])
    return queue