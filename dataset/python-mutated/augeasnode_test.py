"""Tests for AugeasParserNode classes"""
import os
from typing import List
from unittest import mock
import pytest
from certbot import errors
from certbot_apache._internal import assertions
from certbot_apache._internal import augeasparser
from certbot_apache._internal.tests import util

def _get_augeasnode_mock(filepath):
    if False:
        while True:
            i = 10
    ' Helper function for mocking out DualNode instance with an AugeasNode '

    def augeasnode_mock(metadata):
        if False:
            for i in range(10):
                print('nop')
        return augeasparser.AugeasBlockNode(name=assertions.PASS, ancestor=None, filepath=filepath, metadata=metadata)
    return augeasnode_mock

class AugeasParserNodeTest(util.ApacheTest):
    """Test AugeasParserNode using available test configurations"""

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        with mock.patch('certbot_apache._internal.configurator.ApacheConfigurator.get_parsernode_root') as mock_parsernode:
            mock_parsernode.side_effect = _get_augeasnode_mock(os.path.join(self.config_path, 'apache2.conf'))
            self.config = util.get_apache_configurator(self.config_path, self.vhost_path, self.config_dir, self.work_dir, use_parsernode=True)
        self.vh_truth = util.get_vh_truth(self.temp_dir, 'debian_apache_2_4/multiple_vhosts')

    def test_save(self):
        if False:
            return 10
        with mock.patch('certbot_apache._internal.parser.ApacheParser.save') as mock_save:
            self.config.parser_root.save('A save message')
        assert mock_save.called is True
        assert mock_save.call_args[0][0] == 'A save message'

    def test_unsaved_files(self):
        if False:
            return 10
        with mock.patch('certbot_apache._internal.parser.ApacheParser.unsaved_files') as mock_uf:
            mock_uf.return_value = ['first', 'second']
            files = self.config.parser_root.unsaved_files()
        assert files == ['first', 'second']

    def test_get_block_node_name(self):
        if False:
            for i in range(10):
                print('nop')
        from certbot_apache._internal.augeasparser import AugeasBlockNode
        block = AugeasBlockNode(name=assertions.PASS, ancestor=None, filepath=assertions.PASS, metadata={'augeasparser': mock.Mock(), 'augeaspath': '/files/anything'})
        testcases = {'/some/path/FirstNode/SecondNode': 'SecondNode', '/some/path/FirstNode/SecondNode/': 'SecondNode', 'OnlyPathItem': 'OnlyPathItem', '/files/etc/apache2/apache2.conf/VirtualHost': 'VirtualHost', '/Anything': 'Anything'}
        for test in testcases:
            assert block._aug_get_name(test) == testcases[test]

    def test_find_blocks(self):
        if False:
            return 10
        blocks = self.config.parser_root.find_blocks('VirtualHost', exclude=False)
        assert len(blocks) == 12

    def test_find_blocks_case_insensitive(self):
        if False:
            print('Hello World!')
        vhs = self.config.parser_root.find_blocks('VirtualHost')
        vhs2 = self.config.parser_root.find_blocks('viRtuAlHoST')
        assert len(vhs) == len(vhs2)

    def test_find_directive_found(self):
        if False:
            return 10
        directives = self.config.parser_root.find_directives('Listen')
        assert len(directives) == 1
        assert directives[0].filepath.endswith('/apache2/ports.conf') is True
        assert directives[0].parameters == (u'80',)

    def test_find_directive_notfound(self):
        if False:
            for i in range(10):
                print('nop')
        directives = self.config.parser_root.find_directives('Nonexistent')
        assert len(directives) == 0

    def test_find_directive_from_block(self):
        if False:
            return 10
        blocks = self.config.parser_root.find_blocks('virtualhost')
        found = False
        for vh in blocks:
            if vh.filepath.endswith('sites-enabled/certbot.conf'):
                servername = vh.find_directives('servername')
                assert servername[0].parameters[0] == 'certbot.demo'
                found = True
        assert found is True

    def test_find_comments(self):
        if False:
            i = 10
            return i + 15
        rootcomment = self.config.parser_root.find_comments('This is the main Apache server configuration file. ')
        assert len(rootcomment) == 1
        assert rootcomment[0].filepath.endswith('debian_apache_2_4/multiple_vhosts/apache2/apache2.conf') is True

    def test_set_parameters(self):
        if False:
            return 10
        servernames = self.config.parser_root.find_directives('servername')
        names: List[str] = []
        for servername in servernames:
            names += servername.parameters
        assert 'going_to_set_this' not in names
        servernames[0].set_parameters(['something', 'going_to_set_this'])
        servernames = self.config.parser_root.find_directives('servername')
        names = []
        for servername in servernames:
            names += servername.parameters
        assert 'going_to_set_this' in names

    def test_set_parameters_atinit(self):
        if False:
            while True:
                i = 10
        from certbot_apache._internal.augeasparser import AugeasDirectiveNode
        servernames = self.config.parser_root.find_directives('servername')
        setparam = 'certbot_apache._internal.augeasparser.AugeasDirectiveNode.set_parameters'
        with mock.patch(setparam) as mock_set:
            AugeasDirectiveNode(name=servernames[0].name, parameters=['test', 'setting', 'these'], ancestor=assertions.PASS, metadata=servernames[0].metadata)
            assert mock_set.called is True
            assert mock_set.call_args_list[0][0][0] == ['test', 'setting', 'these']

    def test_set_parameters_delete(self):
        if False:
            return 10
        servername = self.config.parser_root.find_directives('servername')[0]
        servername.set_parameters(['thisshouldnotexistpreviously', 'another', 'third'])
        servernames = self.config.parser_root.find_directives('servername')
        found = False
        for servername in servernames:
            if 'thisshouldnotexistpreviously' in servername.parameters:
                assert len(servername.parameters) == 3
                servername.set_parameters(['thisshouldnotexistpreviously'])
                found = True
        assert found is True
        servernames = self.config.parser_root.find_directives('servername')
        found = False
        for servername in servernames:
            if 'thisshouldnotexistpreviously' in servername.parameters:
                assert len(servername.parameters) == 1
                servername.set_parameters(['thisshouldnotexistpreviously'])
                found = True
        assert found is True

    def test_add_child_comment(self):
        if False:
            i = 10
            return i + 15
        newc = self.config.parser_root.add_child_comment('The content')
        comments = self.config.parser_root.find_comments('The content')
        assert len(comments) == 1
        assert newc.metadata['augeaspath'] == comments[0].metadata['augeaspath']
        assert newc.comment == comments[0].comment

    def test_delete_child(self):
        if False:
            return 10
        listens = self.config.parser_root.find_directives('Listen')
        assert len(listens) == 1
        self.config.parser_root.delete_child(listens[0])
        listens = self.config.parser_root.find_directives('Listen')
        assert len(listens) == 0

    def test_delete_child_not_found(self):
        if False:
            return 10
        listen = self.config.parser_root.find_directives('Listen')[0]
        listen.metadata['augeaspath'] = '/files/something/nonexistent'
        with pytest.raises(errors.PluginError):
            self.config.parser_root.delete_child(listen)

    def test_add_child_block(self):
        if False:
            return 10
        nb = self.config.parser_root.add_child_block('NewBlock', ['first', 'second'])
        (rpath, _, directive) = nb.metadata['augeaspath'].rpartition('/')
        assert rpath == self.config.parser_root.metadata['augeaspath']
        assert directive.startswith('NewBlock') is True

    def test_add_child_block_beginning(self):
        if False:
            return 10
        self.config.parser_root.add_child_block('Beginning', position=0)
        parser = self.config.parser_root.parser
        root_path = self.config.parser_root.metadata['augeaspath']
        first = parser.aug.match('{}/*[1]'.format(root_path))
        assert first[0].endswith('Beginning') is True

    def test_add_child_block_append(self):
        if False:
            i = 10
            return i + 15
        self.config.parser_root.add_child_block('VeryLast')
        parser = self.config.parser_root.parser
        root_path = self.config.parser_root.metadata['augeaspath']
        last = parser.aug.match('{}/*[last()]'.format(root_path))
        assert last[0].endswith('VeryLast') is True

    def test_add_child_block_append_alt(self):
        if False:
            i = 10
            return i + 15
        self.config.parser_root.add_child_block('VeryLastAlt', position=99999)
        parser = self.config.parser_root.parser
        root_path = self.config.parser_root.metadata['augeaspath']
        last = parser.aug.match('{}/*[last()]'.format(root_path))
        assert last[0].endswith('VeryLastAlt') is True

    def test_add_child_block_middle(self):
        if False:
            while True:
                i = 10
        self.config.parser_root.add_child_block('Middle', position=5)
        parser = self.config.parser_root.parser
        root_path = self.config.parser_root.metadata['augeaspath']
        middle = parser.aug.match('{}/*[6]'.format(root_path))
        assert middle[0].endswith('Middle') is True

    def test_add_child_block_existing_name(self):
        if False:
            i = 10
            return i + 15
        parser = self.config.parser_root.parser
        root_path = self.config.parser_root.metadata['augeaspath']
        new_block = parser.aug.match('{}/VirtualHost[2]'.format(root_path))
        assert len(new_block) == 0
        vh = self.config.parser_root.add_child_block('VirtualHost')
        new_block = parser.aug.match('{}/VirtualHost[2]'.format(root_path))
        assert len(new_block) == 1
        assert vh.metadata['augeaspath'].endswith('VirtualHost[2]') is True

    def test_node_init_error_bad_augeaspath(self):
        if False:
            return 10
        from certbot_apache._internal.augeasparser import AugeasBlockNode
        parameters = {'name': assertions.PASS, 'ancestor': None, 'filepath': assertions.PASS, 'metadata': {'augeasparser': mock.Mock(), 'augeaspath': '/files/path/endswith/slash/'}}
        with pytest.raises(errors.PluginError):
            AugeasBlockNode(**parameters)

    def test_node_init_error_missing_augeaspath(self):
        if False:
            i = 10
            return i + 15
        from certbot_apache._internal.augeasparser import AugeasBlockNode
        parameters = {'name': assertions.PASS, 'ancestor': None, 'filepath': assertions.PASS, 'metadata': {'augeasparser': mock.Mock()}}
        with pytest.raises(errors.PluginError):
            AugeasBlockNode(**parameters)

    def test_add_child_directive(self):
        if False:
            for i in range(10):
                print('nop')
        self.config.parser_root.add_child_directive('ThisWasAdded', ['with', 'parameters'], position=0)
        dirs = self.config.parser_root.find_directives('ThisWasAdded')
        assert len(dirs) == 1
        assert dirs[0].parameters == ('with', 'parameters')
        assert dirs[0].metadata['augeaspath'].endswith('[1]') is True

    def test_add_child_directive_exception(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(errors.PluginError):
            self.config.parser_root.add_child_directive('ThisRaisesErrorBecauseMissingParameters')

    def test_parsed_paths(self):
        if False:
            return 10
        paths = self.config.parser_root.parsed_paths()
        assert len(paths) == 6

    def test_find_ancestors(self):
        if False:
            return 10
        vhsblocks = self.config.parser_root.find_blocks('VirtualHost')
        macro_test = False
        nonmacro_test = False
        for vh in vhsblocks:
            if '/macro/' in vh.metadata['augeaspath'].lower():
                ancs = vh.find_ancestors('Macro')
                assert len(ancs) == 1
                macro_test = True
            else:
                ancs = vh.find_ancestors('Macro')
                assert len(ancs) == 0
                nonmacro_test = True
        assert macro_test is True
        assert nonmacro_test is True

    def test_find_ancestors_bad_path(self):
        if False:
            i = 10
            return i + 15
        self.config.parser_root.metadata['augeaspath'] = ''
        ancs = self.config.parser_root.find_ancestors('Anything')
        assert len(ancs) == 0