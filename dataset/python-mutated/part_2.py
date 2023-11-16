import hug

@hug.get()
def part2():
    if False:
        return 10
    'This view will be at the path ``/part2``'
    return 'Part 2'