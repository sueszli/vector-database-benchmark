from Translate import Translate

class TestTranslate:

    def testTranslateStrict(self):
        if False:
            print('Hello World!')
        translate = Translate()
        data = '\n            translated = _("original")\n            not_translated = "original"\n        '
        data_translated = translate.translateData(data, {'_(original)': 'translated'})
        assert 'translated = _("translated")' in data_translated
        assert 'not_translated = "original"' in data_translated

    def testTranslateStrictNamed(self):
        if False:
            i = 10
            return i + 15
        translate = Translate()
        data = '\n            translated = _("original", "original named")\n            translated_other = _("original", "original other named")\n            not_translated = "original"\n        '
        data_translated = translate.translateData(data, {'_(original, original named)': 'translated'})
        assert 'translated = _("translated")' in data_translated
        assert 'not_translated = "original"' in data_translated

    def testTranslateUtf8(self):
        if False:
            i = 10
            return i + 15
        translate = Translate()
        data = '\n            greeting = "Hi again árvztűrőtökörfúrógép!"\n        '
        data_translated = translate.translateData(data, {'Hi again árvztűrőtökörfúrógép!': 'Üdv újra árvztűrőtökörfúrógép!'})
        assert data_translated == '\n            greeting = "Üdv újra árvztűrőtökörfúrógép!"\n        '

    def testTranslateEscape(self):
        if False:
            for i in range(10):
                print('nop')
        _ = Translate()
        _['Hello'] = 'Szia'
        data = '{_[Hello]} {username}!'
        username = "Hacker<script>alert('boom')</script>"
        data_translated = _(data)
        assert 'Szia' in data_translated
        assert '<' not in data_translated
        assert data_translated == 'Szia Hacker&lt;script&gt;alert(&#x27;boom&#x27;)&lt;/script&gt;!'
        user = {'username': "Hacker<script>alert('boom')</script>"}
        data = '{_[Hello]} {user[username]}!'
        data_translated = _(data)
        assert 'Szia' in data_translated
        assert '<' not in data_translated
        assert data_translated == 'Szia Hacker&lt;script&gt;alert(&#x27;boom&#x27;)&lt;/script&gt;!'
        users = [{'username': "Hacker<script>alert('boom')</script>"}]
        data = '{_[Hello]} {users[0][username]}!'
        data_translated = _(data)
        assert 'Szia' in data_translated
        assert '<' not in data_translated
        assert data_translated == 'Szia Hacker&lt;script&gt;alert(&#x27;boom&#x27;)&lt;/script&gt;!'