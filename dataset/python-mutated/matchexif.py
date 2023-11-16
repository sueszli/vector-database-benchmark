from collections import defaultdict
from itertools import combinations
from hscommon.trans import tr
from core.engine import Match

def getmatches(files, match_scaled, j):
    if False:
        for i in range(10):
            print('nop')
    timestamp2pic = defaultdict(set)
    for picture in j.iter_with_progress(files, tr('Read EXIF of %d/%d pictures')):
        timestamp = picture.exif_timestamp
        if timestamp:
            timestamp2pic[timestamp].add(picture)
    if '0000:00:00 00:00:00' in timestamp2pic:
        del timestamp2pic['0000:00:00 00:00:00']
    matches = []
    for pictures in timestamp2pic.values():
        for (p1, p2) in combinations(pictures, 2):
            if not match_scaled and p1.dimensions != p2.dimensions:
                continue
            matches.append(Match(p1, p2, 100))
    return matches