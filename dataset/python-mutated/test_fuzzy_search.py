from ulauncher.utils.fuzzy_search import _normalize, get_matching_blocks, get_score

def test_normalize():
    if False:
        print('Hello World!')
    assert _normalize('Virransäästö') == 'virransaasto'
    assert _normalize('Éditeur d’image GIMP') == 'editeur dimage gimp'
    assert _normalize('Ögbelgilengen Uyğulamalar') == 'ogbelgilengen uygulamalar'
    assert _normalize('Füße') == 'fusse'

def test_get_matching_indexes():
    if False:
        print('Hello World!')
    assert get_matching_blocks('thfima', 'Thunar File Manager') == ([(0, 'Th'), (7, 'Fi'), (12, 'Ma')], 6)

def test_get_score():
    if False:
        return 10
    assert get_score('calc', 'Contacts') < get_score('calc', 'LibreOffice Calc')
    assert get_score('pla', 'Pycharm') < get_score('pla', 'Google Play Music')
    assert get_score('', 'LibreOffice Calc') == 0
    assert get_score('0', 'LibreOffice Calc') == 0