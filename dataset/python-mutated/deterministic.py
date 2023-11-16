def unique_list(duplicates_list):
    if False:
        while True:
            i = 10
    '\n    Return unique list preserving the order.\n    https://stackoverflow.com/a/480227\n    '
    used = set()
    unique = [x for x in duplicates_list if not (x in used or used.add(x))]
    return unique