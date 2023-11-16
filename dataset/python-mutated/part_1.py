import hug

@hug.get()
def part1():
    if False:
        print('Hello World!')
    'This view will be at the path ``/part1``'
    return 'part1'