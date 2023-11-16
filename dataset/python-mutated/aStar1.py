"""
aStar1.py: 不行！？？
"""
import sys
from multiprocessing import Queue
from PIL import Image
start = (400, 984)
end = (398, 25)

def iswhite(value):
    if False:
        i = 10
        return i + 15
    if value == (255, 255, 255):
        return True

def getadjacent(n):
    if False:
        for i in range(10):
            print('nop')
    (x, y) = n
    return [(x - 1, y), (x, y - 1), (x + 1, y), (x, y + 1)]

def BFS(start, end, pixels):
    if False:
        i = 10
        return i + 15
    queue = Queue()
    queue.put([start])
    while not queue.empty():
        path = queue.get()
        pixel = path[-1]
        if pixel == end:
            return path
        for adjacent in getadjacent(pixel):
            (x, y) = adjacent
            if iswhite(pixels[x, y]):
                pixels[x, y] = (127, 127, 127)
                new_path = list(path)
                new_path.append(adjacent)
                queue.put(new_path)
    print('Queue has been exhausted. No answer was found.')
if __name__ == '__main__':
    base_img = Image.open(sys.argv[1])
    base_pixels = base_img.load()
    print(base_pixels)
    path = BFS(start, end, base_pixels)
    if path is None:
        print('path is None')
        exit(-1)
    print('path:', path)
    path_img = Image.open(sys.argv[1])
    path_pixels = path_img.load()
    for position in path:
        (x, y) = position
        path_pixels[x, y] = (255, 0, 0)
    path_img.save(sys.argv[2])