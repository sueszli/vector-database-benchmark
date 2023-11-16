from ckan.lib import config_tool

def changes_builder(action, key, value, section='app:main', commented=False):
    if False:
        i = 10
        return i + 15
    changes = config_tool.Changes()
    changes.add(action, config_tool.Option(section, key, value, commented))
    return changes

class TestMakeChanges:

    def test_edit(self):
        if False:
            print('Hello World!')
        config_lines = '\n[app:main]\nckan.site_title = CKAN\n        '.split('\n')
        out = config_tool.make_changes(config_lines, [], changes_builder('edit', 'ckan.site_title', 'New Title'))
        assert out == '\n[app:main]\nckan.site_title = New Title\n        '.split('\n'), out

    def test_new(self):
        if False:
            return 10
        config_lines = '\n[app:main]\nckan.site_title = CKAN\n        '.split('\n')
        out = config_tool.make_changes(config_lines, [], changes_builder('add', 'ckan.option', 'New stuff'))
        assert out == '\n[app:main]\nckan.option = New stuff\nckan.site_title = CKAN\n        '.split('\n'), out

    def test_new_section(self):
        if False:
            i = 10
            return i + 15
        config_lines = '\n'.split('\n')
        out = config_tool.make_changes(config_lines, ['logger'], changes_builder('add', 'keys', 'root, ckan, ckanext', section='logger'))
        assert out == '\n\n[logger]\nkeys = root, ckan, ckanext\n'.split('\n'), out

    def test_new_section_before_appmain(self):
        if False:
            i = 10
            return i + 15
        config_lines = '\n[app:main]\nckan.site_title = CKAN\n'.split('\n')
        out = config_tool.make_changes(config_lines, ['logger'], changes_builder('add', 'keys', 'root, ckan, ckanext', section='logger'))
        assert out == '\n[logger]\nkeys = root, ckan, ckanext\n\n[app:main]\nckan.site_title = CKAN\n'.split('\n'), out

    def test_edit_commented_line(self):
        if False:
            print('Hello World!')
        config_lines = '\n[app:main]\n#ckan.site_title = CKAN\n        '.split('\n')
        out = config_tool.make_changes(config_lines, [], changes_builder('edit', 'ckan.site_title', 'New Title'))
        assert out == '\n[app:main]\nckan.site_title = New Title\n        '.split('\n'), out

    def test_comment_out_line(self):
        if False:
            print('Hello World!')
        config_lines = '\n[app:main]\nckan.site_title = CKAN\n        '.split('\n')
        out = config_tool.make_changes(config_lines, [], changes_builder('edit', 'ckan.site_title', 'New Title', commented=True))
        assert out == '\n[app:main]\n#ckan.site_title = New Title\n        '.split('\n'), out

    def test_edit_repeated_commented_line(self):
        if False:
            for i in range(10):
                print('nop')
        config_lines = '\n[app:main]\n#ckan.site_title = CKAN1\nckan.site_title = CKAN2\nckan.site_title = CKAN3\n#ckan.site_title = CKAN4\n        '.split('\n')
        out = config_tool.make_changes(config_lines, [], changes_builder('edit', 'ckan.site_title', 'New Title'))
        assert out == '\n[app:main]\nckan.site_title = New Title\n#ckan.site_title = CKAN2\n#ckan.site_title = CKAN3\n#ckan.site_title = CKAN4\n        '.split('\n'), out

class TestParseConfig:

    def test_parse_basic(self):
        if False:
            return 10
        input_lines = '\n[app:main]\nckan.site_title = CKAN\n'.split('\n')
        out = config_tool.parse_config(input_lines)
        assert str(out) == "{'app:main-ckan.site_title': <Option [app:main] ckan.site_title = CKAN>}"

    def test_parse_sections(self):
        if False:
            return 10
        input_lines = '\n[logger]\nkeys = root, ckan, ckanext\nlevel = WARNING\n\n[app:main]\nckan.site_title = CKAN\n'.split('\n')
        out = sorted(config_tool.parse_config(input_lines).items())
        assert str(out) == "[('app:main-ckan.site_title', <Option [app:main] ckan.site_title = CKAN>), ('logger-keys', <Option [logger] keys = root, ckan, ckanext>), ('logger-level', <Option [logger] level = WARNING>)]"

    def test_parse_comment(self):
        if False:
            return 10
        input_lines = '\n[app:main]\n#ckan.site_title = CKAN\n'.split('\n')
        out = config_tool.parse_config(input_lines)
        assert str(out) == "{'app:main-ckan.site_title': <Option [app:main] #ckan.site_title = CKAN>}"

class TestParseOptionString:

    def test_parse_basic(self):
        if False:
            print('Hello World!')
        input_line = 'ckan.site_title = CKAN'
        out = config_tool.parse_option_string('app:main', input_line)
        assert repr(out) == '<Option [app:main] ckan.site_title = CKAN>'
        assert str(out) == 'ckan.site_title = CKAN'

    def test_parse_extra_spaces(self):
        if False:
            print('Hello World!')
        input_line = 'ckan.site_title  =  CKAN '
        out = config_tool.parse_option_string('app:main', input_line)
        assert repr(out) == '<Option [app:main] ckan.site_title  =  CKAN >'
        assert str(out) == 'ckan.site_title  =  CKAN '
        assert out.key == 'ckan.site_title'
        assert out.value == 'CKAN'

    def test_parse_invalid_space(self):
        if False:
            return 10
        input_line = ' ckan.site_title = CKAN'
        out = config_tool.parse_option_string('app:main', input_line)
        assert out is None