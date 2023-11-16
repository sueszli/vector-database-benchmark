from robot.api.deco import keyword

@keyword('Get ${Match} With Search Order')
def get_best_match_ever_with_search_order(Match):
    if False:
        for i in range(10):
            print('nop')
    raise AssertionError('Should not be run due to a better matchin same library.')

@keyword('Get Best ${Match:\\w+} With Search Order')
def get_best_match_with_search_order(Match):
    if False:
        i = 10
        return i + 15
    raise AssertionError('Should not be run due to a better matchin same library.')

@keyword('Get Best ${Match} With Search Order')
def get_best_match_with_search_order(Match):
    if False:
        for i in range(10):
            print('nop')
    assert Match == 'Match Ever'
    return 'embedded2'