import sys
import pytest
from unittest import mock
from apprise import NotifyFormat
from apprise import ConfigFormat
from apprise import ContentIncludeMode
from apprise import Apprise
from apprise import AppriseConfig
from apprise import AppriseAsset
from apprise.config.ConfigBase import ConfigBase
from apprise.plugins.NotifyBase import NotifyBase
from apprise.common import CONFIG_SCHEMA_MAP
from apprise.common import NOTIFY_SCHEMA_MAP
from apprise.config import __load_matrix
from apprise.config.ConfigFile import ConfigFile
import logging
logging.disable(logging.CRITICAL)

def test_apprise_config(tmpdir):
    if False:
        i = 10
        return i + 15
    '\n    API: AppriseConfig basic testing\n\n    '
    ac = AppriseConfig()
    assert len(ac) == 0
    assert not ac
    assert len(ac.servers()) == 0
    t = tmpdir.mkdir('simple-formatting').join('apprise')
    t.write('\n    # A comment line over top of a URL\n    mailto://usera:pass@gmail.com\n\n    # A line with mulitiple tag assignments to it\n    taga,tagb=gnome://\n\n    # Event if there is accidental leading spaces, this configuation\n    # is accepting of htat and will not exclude them\n                tagc=kde://\n\n    # A very poorly structured url\n    sns://:@/\n\n    # Just 1 token provided causes exception\n    sns://T1JJ3T3L2/\n\n    # XML\n    xml://localhost/?+HeaderEntry=Test&:IgnoredEntry=Ignored\n    ')
    ac = AppriseConfig(paths=str(t))
    assert len(ac) == 1
    assert ac
    assert len(ac.servers()) == 4
    assert isinstance(ac[0].url(), str)
    t = tmpdir.mkdir('strange-lines').join('apprise')
    t.write('\n    # basicly this consists of defined tags and no url\n    tag=\n    ')
    ac = AppriseConfig(paths=str(t), asset=AppriseAsset())
    assert len(ac) == 1
    assert len(ac.servers()) == 0
    cb = ConfigBase()
    assert ac.add(configs=cb, asset=AppriseAsset(), tag='test') is True
    assert ac.add(configs=['file://?'], asset=AppriseAsset(), tag='test') is False
    assert ac.add(configs=object()) is False
    ac = AppriseConfig(paths='file://{}?format=text'.format(str(t)))
    assert len(ac) == 1
    assert len(ac.servers()) == 0
    istr = '\n        # Iñtërnâtiônàlization Testing\n        windows://'
    t = tmpdir.mkdir('internationalization').join('apprise')
    with open(str(t), 'wb') as f:
        f.write(istr.encode('latin-1'))
    ac = AppriseConfig(paths=str(t))
    assert len(ac) == 1
    assert len(ac.servers()) == 0
    count = 0
    for entry in ac:
        count += 1
    assert len(ac) == count
    ac = AppriseConfig(paths='file://{}?encoding=latin-1'.format(str(t)))
    assert len(ac) == 1
    assert len(ac.servers()) == 1
    assert isinstance(ac[0].url(), str)
    assert isinstance(ac.pop(0), ConfigBase) is True
    assert len(ac) == 0
    t = tmpdir.mkdir('buffer-handling').join('apprise')
    buf = 'gnome://'
    t.write(buf)
    ac.clear()
    ac = AppriseConfig(paths=str(t))
    ac[0].max_buffer_size = len(buf)
    assert len(ac) == 1
    assert len(ac.servers()) == 1
    ac[0].max_buffer_size = len(buf) - 1
    assert len(ac.servers()) == 1

def test_apprise_multi_config_entries(tmpdir):
    if False:
        print('Hello World!')
    '\n    API: AppriseConfig basic multi-adding functionality\n\n    '
    t = tmpdir.mkdir('apprise-multi-add').join('apprise')
    buf = '\n    good://hostname\n    '
    t.write(buf)
    te = tmpdir.join('apprise-multi-add', 'apprise-empty')
    te.write('')

    class GoodNotification(NotifyBase):

        def __init__(self, **kwargs):
            if False:
                return 10
            super().__init__(notify_format=NotifyFormat.HTML, **kwargs)

        def notify(self, **kwargs):
            if False:
                print('Hello World!')
            return True

        def url(self, **kwargs):
            if False:
                i = 10
                return i + 15
            return ''
    NOTIFY_SCHEMA_MAP['good'] = GoodNotification
    ac = AppriseConfig()
    assert len(ac) == 0
    assert ac.add(configs=(str(t), str(t))) is True
    assert ac.add(configs=(ConfigFile(path=str(te)), ConfigFile(path=str(t)))) is True
    assert ac.add(configs=(object(), object())) is False
    assert ac.add(configs=object()) is False
    try:
        ac.server_pop(len(ac.servers()))
        assert False
    except IndexError:
        assert True
    while len(ac.servers()) > 0:
        assert isinstance(ac.server_pop(len(ac.servers()) - 1), NotifyBase) is True

def test_apprise_add_config():
    if False:
        print('Hello World!')
    '\n    API AppriseConfig.add_config()\n\n    '
    content = '\n    # A comment line over top of a URL\n    mailto://usera:pass@gmail.com\n\n    # A line with mulitiple tag assignments to it\n    taga,tagb=gnome://\n\n    # Event if there is accidental leading spaces, this configuation\n    # is accepting of htat and will not exclude them\n                tagc=kde://\n\n    # A very poorly structured url\n    sns://:@/\n\n    # Just 1 token provided causes exception\n    sns://T1JJ3T3L2/\n    '
    ac = AppriseConfig()
    assert ac.add_config(content=content) is True
    assert len(ac) == 1
    assert ac[0].config_format is ConfigFormat.TEXT
    assert ac
    assert len(ac.servers()) == 3
    assert isinstance(ac[0].url(), str)
    assert ac.add_config(content=object()) is False
    assert ac.add_config(content=42) is False
    assert ac.add_config(content=None) is False
    assert len(ac) == 1
    assert ac.add_config(content=content, asset=AppriseAsset(), tag='a') is True
    assert len(ac) == 2
    assert len(ac.servers()) == 6
    content = "\n    # A YAML File\n    urls:\n       - mailto://usera:pass@gmail.com\n       - gnome://:\n          tag: taga,tagb\n\n       - json://localhost:\n          +HeaderEntry1: 'a header entry'\n          -HeaderEntryDepricated: 'a deprecated entry'\n          :HeaderEntryIgnored: 'an ignored header entry'\n\n       - xml://localhost:\n          +HeaderEntry1: 'a header entry'\n          -HeaderEntryDepricated: 'a deprecated entry'\n          :HeaderEntryIgnored: 'an ignored header entry'\n    "
    ac = AppriseConfig()
    assert ac.add_config(content=content) is True
    assert len(ac) == 1
    assert ac[0].config_format is ConfigFormat.YAML
    assert ac
    assert len(ac.servers()) == 4
    content = 'invalid'
    ac = AppriseConfig()
    assert ac.add_config(content=content) is False
    assert len(ac.servers()) == 0

def test_apprise_config_tagging(tmpdir):
    if False:
        return 10
    '\n    API: AppriseConfig tagging\n\n    '
    t = tmpdir.mkdir('tagging').join('apprise')
    buf = 'gnome://'
    t.write(buf)
    ac = AppriseConfig()
    assert ac.add(configs=str(t), asset=AppriseAsset(), tag='a') is True
    assert ac.add(configs=str(t), asset=AppriseAsset(), tag='b') is True
    assert ac.add(configs=str(t), asset=AppriseAsset(), tag='a,b') is True
    assert len(ac.servers(tag='a')) == 2
    assert len(ac.servers(tag='a,b')) == 3
    assert len(ac.servers(tag=[('a', 'b')])) == 1
    assert len(ac.servers(tag='all')) == 3
    ac = AppriseConfig()
    assert ac.add(configs=str(t), asset=AppriseAsset(), tag='a,always') is True
    assert ac.add(configs=str(t), asset=AppriseAsset(), tag='b') is True
    assert ac.add(configs=str(t), asset=AppriseAsset(), tag='c,d') is True
    assert len(ac.servers(tag='a')) == 1
    assert len(ac.servers(tag='a,b')) == 2
    assert len(ac.servers(tag='e')) == 1
    assert len(ac.servers(tag='e', match_always=False)) == 0
    assert len(ac.servers(tag='all')) == 3
    assert len(ac.servers(tag='d')) == 2
    assert len(ac.servers(tag='d', match_always=False)) == 1

def test_apprise_config_instantiate():
    if False:
        for i in range(10):
            print('nop')
    '\n    API: AppriseConfig.instantiate()\n\n    '
    assert AppriseConfig.instantiate('file://?', suppress_exceptions=True) is None
    assert AppriseConfig.instantiate('invalid://?', suppress_exceptions=True) is None

    class BadConfig(ConfigBase):
        allow_cross_includes = ContentIncludeMode.ALWAYS

        def __init__(self, **kwargs):
            if False:
                while True:
                    i = 10
            super().__init__(**kwargs)
            raise TypeError()

        @staticmethod
        def parse_url(url, *args, **kwargs):
            if False:
                print('Hello World!')
            return ConfigBase.parse_url(url, verify_host=False)
    CONFIG_SCHEMA_MAP['bad'] = BadConfig
    with pytest.raises(TypeError):
        AppriseConfig.instantiate('bad://path', suppress_exceptions=False)
    assert AppriseConfig.instantiate('bad://path', suppress_exceptions=True) is None

def test_invalid_apprise_config(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    '\n    Parse invalid configuration includes\n\n    '

    class BadConfig(ConfigBase):
        allow_cross_includes = ContentIncludeMode.ALWAYS

        def __init__(self, **kwargs):
            if False:
                while True:
                    i = 10
            super().__init__(**kwargs)
            raise TypeError()

        @staticmethod
        def parse_url(url, *args, **kwargs):
            if False:
                return 10
            return ConfigBase.parse_url(url, verify_host=False)
    CONFIG_SCHEMA_MAP['bad'] = BadConfig
    t = tmpdir.mkdir('apprise-bad-obj').join('invalid')
    buf = '\n    # Include an invalid schema\n    include invalid://\n\n    # An unparsable valid schema\n    include https://\n\n    # A valid configuration that will throw an exception\n    include bad://\n\n    # Include ourselves (So our recursive includes fails as well)\n    include {}\n\n    '.format(str(t))
    t.write(buf)
    ac = AppriseConfig(recursion=2, insecure_includes=True, cache=False)
    assert len(ac) == 0
    assert ac.add(configs=str(t), asset=AppriseAsset()) is True
    assert len(ac) == 1
    assert len(ac.servers()) == 0

def test_apprise_config_with_apprise_obj(tmpdir):
    if False:
        return 10
    '\n    API: ConfigBase - parse valid config\n\n    '
    t = tmpdir.mkdir('apprise-obj').join('apprise')
    buf = '\n    good://hostname\n    localhost=good://localhost\n    '
    t.write(buf)

    class GoodNotification(NotifyBase):

        def __init__(self, **kwargs):
            if False:
                return 10
            super().__init__(notify_format=NotifyFormat.HTML, **kwargs)

        def notify(self, **kwargs):
            if False:
                return 10
            return True

        def url(self, **kwargs):
            if False:
                while True:
                    i = 10
            return ''
    NOTIFY_SCHEMA_MAP['good'] = GoodNotification
    ac = AppriseConfig(cache=False)
    assert len(ac) == 0
    assert ac.add(configs=str(t), asset=AppriseAsset(), tag='a') is True
    assert len(ac) == 1
    assert len(ac.servers()) == 2
    ac.server_pop(0)
    assert len(ac.servers()) == 1
    a = Apprise()
    assert a.add(servers=ac) is True
    assert len(a) == 1
    assert a.notify(body='apprise configuration power!') is True
    assert a.add(servers=[AppriseConfig(str(t)), AppriseConfig(str(t))]) is True
    assert len(a) == 5
    assert a.add(servers=object()) is False
    assert a.add(servers=[object(), object()]) is False
    assert len(a) == 5
    ref = a[0]
    assert isinstance(ref, NotifyBase) is True
    assert len(a) == 5
    ref_popped = a.pop(0)
    assert isinstance(ref_popped, NotifyBase) is True
    assert len(a) == 4
    assert ref == ref_popped
    try:
        a.pop(len(a))
        assert False
    except IndexError:
        assert True
    assert len(a) == 4
    try:
        a[len(a)]
        assert False
    except IndexError:
        assert True
    ref = a[len(a) - 1]
    assert isinstance(ref, NotifyBase) is True
    assert len(a) == 4
    ref_popped = a.pop(len(a) - 1)
    assert isinstance(ref_popped, NotifyBase) is True
    assert ref == ref_popped
    assert len(a) == 3
    assert a.add(servers=[ConfigFile(path=str(t)), 'good://another.host', GoodNotification(**{'host': 'nuxref.com'})]) is True
    assert len(a) == 7
    ref = a[len(a) - 1]
    assert isinstance(ref, NotifyBase) is True
    ref_popped = a.pop(len(a) - 1)
    assert isinstance(ref_popped, NotifyBase) is True
    assert ref == ref_popped
    assert len(a) == 6
    while len(a) > 0:
        assert isinstance(a.pop(len(a) - 1), NotifyBase) is True

def test_recursive_config_inclusion(tmpdir):
    if False:
        print('Hello World!')
    '\n    API: Apprise() Recursive Config Inclusion\n\n    '

    class ConfigCrossPostAlways(ConfigFile):
        """
        A dummy config that is set to always allow inclusion
        """
        service_name = 'always'
        protocol = 'always'
        allow_cross_includes = ContentIncludeMode.ALWAYS

    class ConfigCrossPostStrict(ConfigFile):
        """
        A dummy config that is set to strict inclusion
        """
        service_name = 'strict'
        protocol = 'strict'
        allow_cross_includes = ContentIncludeMode.STRICT

    class ConfigCrossPostNever(ConfigFile):
        """
        A dummy config that is set to never allow inclusion
        """
        service_name = 'never'
        protocol = 'never'
        allow_cross_includes = ContentIncludeMode.NEVER
    CONFIG_SCHEMA_MAP['never'] = ConfigCrossPostNever
    CONFIG_SCHEMA_MAP['strict'] = ConfigCrossPostStrict
    CONFIG_SCHEMA_MAP['always'] = ConfigCrossPostAlways
    suite = tmpdir.mkdir('apprise_config_recursion')
    cfg01 = suite.join('cfg01.cfg')
    cfg02 = suite.mkdir('dir1').join('cfg02.cfg')
    cfg03 = suite.mkdir('dir2').join('cfg03.cfg')
    cfg04 = suite.mkdir('dir3').join('cfg04.cfg')
    cfg01.write('\n# json entry\njson://localhost:8080\n\n# absolute path inclusion to ourselves\ninclude {}'.format(str(cfg01)))
    cfg02.write('\n# json entry\njson://localhost:8080\n\n# recursively include ourselves\ninclude cfg02.cfg')
    cfg03.write("\n# xml entry\nxml://localhost:8080\n\n# relative path inclusion\ninclude ../dir1/cfg02.cfg\n\n# test that we can't include invalid entries\ninclude invalid://entry\n\n# Include non includable type\ninclude memory://")
    cfg04.write('\n# xml entry\nxml://localhost:8080\n\n# always include of our file\ninclude always://{}\n\n# never include of our file\ninclude never://{}\n\n# strict include of our file\ninclude strict://{}'.format(str(cfg04), str(cfg04), str(cfg04)))
    ac = AppriseConfig()
    assert len(ac) == 0
    assert ac.add(configs=str(cfg01)) is True
    assert len(ac) == 1
    assert len(ac.servers()) == 1
    ac = AppriseConfig(recursion=1)
    assert ac.add(configs=str(cfg01)) is True
    assert len(ac) == 1
    assert len(ac.servers()) == 2
    ac = AppriseConfig(recursion=10)
    assert len(ac) == 0
    assert ac.add(configs=str(cfg02)) is True
    assert len(ac) == 1
    assert len(ac.servers()) == 11
    ac = AppriseConfig(recursion=1)
    assert len(ac) == 0
    assert ac.add(configs=str(cfg04)) is True
    assert len(ac) == 1
    assert len(ac.servers()) == 2
    ac = AppriseConfig(recursion=1, insecure_includes=True)
    assert len(ac) == 0
    assert ac.add(configs=str(cfg04)) is True
    assert len(ac) == 1
    assert len(ac.servers()) == 3

def test_apprise_config_matrix_load():
    if False:
        print('Hello World!')
    '\n    API: AppriseConfig() matrix initialization\n\n    '
    import apprise

    class ConfigDummy(ConfigBase):
        """
        A dummy wrapper for testing the different options in the load_matrix
        function
        """
        service_name = 'dummy'
        protocol = ('uh', 'oh')
        secure_protocol = ('no', 'yes')

    class ConfigDummy2(ConfigBase):
        """
        A dummy wrapper for testing the different options in the load_matrix
        function
        """
        service_name = 'dummy2'
        secure_protocol = ('true', 'false')

    class ConfigDummy3(ConfigBase):
        """
        A dummy wrapper for testing the different options in the load_matrix
        function
        """
        service_name = 'dummy3'
        secure_protocol = 'true'

    class ConfigDummy4(ConfigBase):
        """
        A dummy wrapper for testing the different options in the load_matrix
        function
        """
        service_name = 'dummy4'
        protocol = 'true'
    apprise.config.ConfigDummy = ConfigDummy
    apprise.config.ConfigDummy2 = ConfigDummy2
    apprise.config.ConfigDummy3 = ConfigDummy3
    apprise.config.ConfigDummy4 = ConfigDummy4
    __load_matrix()
    __load_matrix()

def test_configmatrix_dynamic_importing(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    '\n    API: Apprise() Config Matrix Importing\n\n    '
    suite = tmpdir.mkdir('apprise_config_test_suite')
    suite.join('__init__.py').write('')
    module_name = 'badconfig'
    sys.path.insert(0, str(suite))
    base = suite.mkdir(module_name)
    base.join('__init__.py').write('')
    base.join('ConfigBadFile1.py').write('\nclass ConfigBadFile1:\n    pass')
    base.join('ConfigBadFile2.py').write('\nclass BadClassName:\n    pass')
    base.join('ConfigBadFile3.py').write('raise ImportError()')
    base.join('ConfigGoober.py').write("\nfrom apprise.config import ConfigBase\nclass ConfigGoober(ConfigBase):\n    # This class tests the fact we have a new class name, but we're\n    # trying to over-ride items previously used\n\n    # The default simple (insecure) protocol (used by ConfigHTTP)\n    protocol = ('http', 'goober')\n\n    # The default secure protocol (used by ConfigHTTP)\n    secure_protocol = 'https'\n\n    @staticmethod\n    def parse_url(url, *args, **kwargs):\n        # always parseable\n        return ConfigBase.parse_url(url, verify_host=False)")
    base.join('ConfigBugger.py').write("\nfrom apprise.config import ConfigBase\nclass ConfigBugger(ConfigBase):\n    # This class tests the fact we have a new class name, but we're\n    # trying to over-ride items previously used\n\n    # The default simple (insecure) protocol (used by ConfigHTTP), the other\n    # isn't\n    protocol = ('http', 'bugger-test' )\n\n    # The default secure protocol (used by ConfigHTTP), the other isn't\n    secure_protocol = ('https', ['garbage'])\n\n    @staticmethod\n    def parse_url(url, *args, **kwargs):\n        # always parseable\n        return ConfigBase.parse_url(url, verify_host=False)")
    __load_matrix(path=str(base), name=module_name)

@mock.patch('os.path.getsize')
def test_config_base_parse_inaccessible_text_file(mock_getsize, tmpdir):
    if False:
        print('Hello World!')
    '\n    API: ConfigBase.parse_inaccessible_text_file\n\n    '
    t = tmpdir.mkdir('inaccessible').join('apprise')
    buf = 'gnome://'
    t.write(buf)
    mock_getsize.return_value = None
    mock_getsize.side_effect = OSError
    ac = AppriseConfig(paths=str(t))
    assert len(ac) == 1
    assert len(ac.servers()) == 0

def test_config_base_parse_yaml_file01(tmpdir):
    if False:
        return 10
    '\n    API: ConfigBase.parse_yaml_file (#1)\n\n    '
    t = tmpdir.mkdir('empty-file').join('apprise.yml')
    t.write('')
    ac = AppriseConfig(paths=str(t))
    assert len(ac) == 1
    assert len(ac.servers()) == 0

def test_config_base_parse_yaml_file02(tmpdir):
    if False:
        print('Hello World!')
    '\n    API: ConfigBase.parse_yaml_file (#2)\n\n    '
    t = tmpdir.mkdir('matching-tags').join('apprise.yml')
    t.write('urls:\n  - pover://nsisxnvnqixq39t0cw54pxieyvtdd9@2jevtmstfg5a7hfxndiybasttxxfku:\n    - tag: test1\n  - pover://rg8ta87qngcrkc6t4qbykxktou0uug@tqs3i88xlufexwl8t4asglt4zp5wfn:\n    - tag: test2\n  - pover://jcqgnlyq2oetea4qg3iunahj8d5ijm@evalvutkhc8ipmz2lcgc70wtsm0qpb:\n    - tag: test3')
    ac = AppriseConfig(paths=str(t))
    assert len(ac) == 1
    assert len(ac.servers()) == 3
    a = Apprise()
    assert a.add(servers=ac) is True
    assert len(a) == 3
    assert sum((1 for _ in a.find('no-match'))) == 0
    assert sum((1 for _ in a.find('all'))) == 3
    assert sum((1 for _ in a.find('test1'))) == 1
    assert sum((1 for _ in a.find('test2'))) == 1
    assert sum((1 for _ in a.find('test3'))) == 1
    assert sum((1 for _ in a.find('test1, test3'))) == 2

def test_config_base_parse_yaml_file03(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    '\n    API: ConfigBase.parse_yaml_file (#3)\n\n    '
    t = tmpdir.mkdir('bad-first-entry').join('apprise.yml')
    t.write('urls:\n  - pover://nsisxnvnqixq39t0cw54pxieyvtdd9@2jevtmstfg5a7hfxndiybasttxxfku:\n    -tag: test1\n  - pover://rg8ta87qngcrkc6t4qbykxktou0uug@tqs3i88xlufexwl8t4asglt4zp5wfn:\n    - tag: test2\n  - pover://jcqgnlyq2oetea4qg3iunahj8d5ijm@evalvutkhc8ipmz2lcgc70wtsm0qpb:\n    - tag: test3')
    ac = AppriseConfig(paths=str(t))
    assert len(ac) == 1
    assert len(ac.servers()) == 3
    a = Apprise()
    assert a.add(servers=ac) is True
    assert len(a) == 3
    assert sum((1 for _ in a.find('no-match'))) == 0
    assert sum((1 for _ in a.find('all'))) == 3
    assert sum((1 for _ in a.find('test1'))) == 0
    assert sum((1 for _ in a.find('test2'))) == 1
    assert sum((1 for _ in a.find('test3'))) == 1
    assert sum((1 for _ in a.find('test1, test3'))) == 1

def test_config_base_parse_yaml_file04(tmpdir):
    if False:
        i = 10
        return i + 15
    '\n    API: ConfigBase.parse_yaml_file (#4)\n\n    Test the always keyword\n\n    '
    t = tmpdir.mkdir('always-keyword').join('apprise.yml')
    t.write('urls:\n  - pover://nsisxnvnqixq39t0cw54pxieyvtdd9@2jevtmstfg5a7hfxndiybasttxxfku:\n    - tag: test1,always\n  - pover://rg8ta87qngcrkc6t4qbykxktou0uug@tqs3i88xlufexwl8t4asglt4zp5wfn:\n    - tag: test2\n  - pover://jcqgnlyq2oetea4qg3iunahj8d5ijm@evalvutkhc8ipmz2lcgc70wtsm0qpb:\n    - tag: test3')
    ac = AppriseConfig(paths=str(t))
    assert len(ac) == 1
    assert len(ac.servers()) == 3
    a = Apprise()
    assert a.add(servers=ac) is True
    assert len(a) == 3
    assert sum((1 for _ in a.find('no-match'))) == 1
    assert sum((1 for _ in a.find('no-match', match_always=False))) == 0
    assert sum((1 for _ in a.find('all'))) == 3
    assert sum((1 for _ in a.find('test1'))) == 1
    assert sum((1 for _ in a.find('test1', match_always=False))) == 1
    assert sum((1 for _ in a.find('test2'))) == 2
    assert sum((1 for _ in a.find('test2', match_always=False))) == 1
    assert sum((1 for _ in a.find('test3'))) == 2
    assert sum((1 for _ in a.find('test3', match_always=False))) == 1
    assert sum((1 for _ in a.find('test1, test3'))) == 2

def test_apprise_config_template_parse(tmpdir):
    if False:
        return 10
    '\n    API: AppriseConfig parsing of templates\n\n    '
    ac = AppriseConfig()
    t = tmpdir.mkdir('template-testing').join('apprise.yml')
    t.write('\n\n    tag:\n      - company\n\n    # A comment line over top of a URL\n    urls:\n       - mailto://user:pass@example.com:\n          - to: user1@gmail.com\n            cc: test@hotmail.com\n\n          - to: user2@gmail.com\n            tag: co-worker\n    ')
    ac = AppriseConfig(paths=str(t))
    assert len(ac.servers()) == 2
    assert (False, 'user1@gmail.com') in ac[0][0].targets
    assert 'test@hotmail.com' in ac[0][0].cc
    assert 'company' in ac[0][1].tags
    assert (False, 'user2@gmail.com') in ac[0][1].targets
    assert 'company' in ac[0][1].tags
    assert 'co-worker' in ac[0][1].tags
    tokens = {'bcc': 'user@test.com', 'to': 'user1@abc.com', 'targets': 'user2@abc.com, user3@abc.com   \t', 'ignore': 'not-used'}
    result = ConfigBase._special_token_handler('mailto', tokens)
    assert 'to' not in result
    assert 'bcc' in result
    assert 'targets' in result
    assert 'ignore' in result
    assert len(result['targets']) == 2
    assert 'user1@abc.com' in result['targets']
    assert 'user2@abc.com, user3@abc.com   \t' in result['targets']
    tokens = {'bcc': 'user@test.com', 'to': 'user1@abc.com', 'targets': ['user2@abc.com', 'user3@abc.com'], 'ignore': 'not-used'}
    result = ConfigBase._special_token_handler('mailto', tokens)
    assert 'to' not in result
    assert 'bcc' in result
    assert 'targets' in result
    assert 'ignore' in result
    assert len(result['targets']) == 3
    assert 'user1@abc.com' in result['targets']
    assert 'user2@abc.com' in result['targets']
    assert 'user3@abc.com' in result['targets']
    t.write('\n    # A comment line over top of a URL\n    urls:\n       - mailtos://user:pass@example.com:\n          - smtp: smtp3-dev.google.gmail.com\n            to:\n              - John Smith <user1@gmail.com>\n              - Jason Tater <user2@gmail.com>\n              - user3@gmail.com\n\n          - to: Henry Fisher <user4@gmail.com>, Jason Archie <user5@gmail.com>\n            smtp_host: smtp5-dev.google.gmail.com\n            tag: drinking-buddy\n\n       # provide case where the URL includes some input too\n       # In both of these cases, the cc and targets (to) get over-ridden\n       # by values below\n       - mailtos://user:pass@example.com/arnold@imdb.com/?cc=bill@micro.com/:\n            to:\n              - override01@gmail.com\n            cc:\n              - override02@gmail.com\n\n       - sinch://:\n          - spi: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n            token: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n\n            # Test a case where we expect a string, but yaml reads it in as\n            # a number\n            from: 10005243890\n            to: +1(123)555-1234\n    ')
    ac = AppriseConfig(paths=str(t))
    assert len(ac.servers()) == 4
    assert len(ac[0][0].targets) == 3
    assert ('John Smith', 'user1@gmail.com') in ac[0][0].targets
    assert ('Jason Tater', 'user2@gmail.com') in ac[0][0].targets
    assert (False, 'user3@gmail.com') in ac[0][0].targets
    assert ac[0][0].smtp_host == 'smtp3-dev.google.gmail.com'
    assert len(ac[0][1].targets) == 2
    assert ('Henry Fisher', 'user4@gmail.com') in ac[0][1].targets
    assert ('Jason Archie', 'user5@gmail.com') in ac[0][1].targets
    assert 'drinking-buddy' in ac[0][1].tags
    assert ac[0][1].smtp_host == 'smtp5-dev.google.gmail.com'
    assert len(ac[0][2].targets) == 1
    assert len(ac[0][2].cc) == 1
    assert (False, 'override01@gmail.com') in ac[0][2].targets
    assert 'override02@gmail.com' in ac[0][2].cc
    assert len(ac[0][3].targets) == 1
    assert ac[0][3].service_plan_id == 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
    assert ac[0][3].source == '+10005243890'
    assert ac[0][3].targets[0] == '+11235551234'