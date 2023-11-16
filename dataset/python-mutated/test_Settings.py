from ulauncher.utils.Settings import Settings

class TestSettings:

    def test_defaults(self):
        if False:
            i = 10
            return i + 15
        assert Settings().theme_name == 'light'
        assert Settings(theme_name='asdf').theme_name == 'asdf'

    def test_dash_to_underscore(self):
        if False:
            for i in range(10):
                print('nop')
        s = Settings()
        assert s.theme_name == 'light'
        s.update({'theme-name': 'asdf'})
        assert not hasattr(s, 'theme-name')
        assert hasattr(s, 'theme_name')
        assert s.theme_name == 'asdf'