from robot.api.deco import keyword

@keyword('${match} in ${both} libraries')
def match_in_both_libraries(match, both):
    if False:
        for i in range(10):
            print('nop')
    assert match == 'Match'
    assert both == 'both'

@keyword('Follow search ${order} in libraries')
def follow_search_order_in_libraries(order):
    if False:
        i = 10
        return i + 15
    assert order == 'order'

@keyword('${match} libraries')
def match_libraries(match):
    if False:
        print('Hello World!')
    assert False

@keyword('Unresolvable ${conflict} in library')
def unresolvable_conflict_in_library(conflict):
    if False:
        while True:
            i = 10
    assert False

@keyword('${possible} conflict in library')
def possible_conflict_in_library(possible):
    if False:
        print('Hello World!')
    assert possible == 'No'