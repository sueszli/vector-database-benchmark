from pokemongo_bot.walkers.polyline_walker import PolylineWalker
from pokemongo_bot.walkers.step_walker import StepWalker

def walker_factory(name, bot, dest_lat, dest_lng, dest_alt=None, *args, **kwargs):
    if False:
        while True:
            i = 10
    '\n    Charlie and the Walker Factory\n    '
    if 'StepWalker' == name:
        ret = StepWalker(bot, dest_lat, dest_lng, dest_alt)
    elif 'PolylineWalker' == name:
        ret = PolylineWalker(bot, dest_lat, dest_lng)
    return ret