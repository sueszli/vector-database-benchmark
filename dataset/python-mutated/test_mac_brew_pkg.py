"""
    :codeauthor: Nicole Thomas <nicole@saltstack.com>
"""
import textwrap
import pytest
import salt.modules.mac_brew_pkg as mac_brew
import salt.utils.pkg
from salt.exceptions import CommandExecutionError
from tests.support.mock import MagicMock, Mock, patch

@pytest.fixture
def TAPS_STRING():
    if False:
        i = 10
        return i + 15
    return 'homebrew/dupes\nhomebrew/science\nhomebrew/x11'

@pytest.fixture
def TAPS_LIST():
    if False:
        for i in range(10):
            print('nop')
    return ['homebrew/dupes', 'homebrew/science', 'homebrew/x11']

@pytest.fixture
def HOMEBREW_BIN():
    if False:
        print('Hello World!')
    return '/usr/local/bin/brew'

@pytest.fixture
def configure_loader_modules():
    if False:
        print('Hello World!')
    return {mac_brew: {'__opts__': {'user': MagicMock(return_value='bar')}}}

def custom_call_brew(*cmd, failhard=True):
    if False:
        print('Hello World!')
    result = dict()
    if cmd == ('info', '--json=v2', '--installed'):
        result = {'stdout': textwrap.dedent('                {\n                  "casks": [\n                    {\n                      "appcast": null,\n                      "artifacts": [\n                        [\n                          "Day-3.0/Day-O.app"\n                        ],\n                        {\n                          "signal": {},\n                          "trash": "~/Library/Preferences/com.shauninman.Day-O.plist"\n                        }\n                      ],\n                      "auto_updates": null,\n                      "caveats": null,\n                      "conflicts_with": null,\n                      "container": null,\n                      "depends_on": {},\n                      "desc": null,\n                      "homepage": "https://shauninman.com/archive/2020/04/08/day_o_mac_menu_bar_clock_for_catalina",\n                      "installed": "3.0.1",\n                      "name": [\n                        "Day-O"\n                      ],\n                      "outdated": false,\n                      "sha256": "4963f503c1e47bfa0f8bdbbbe5694d6a7242d298fb44ff68af80d42f1eaebaf9",\n                      "token": "day-o",\n                      "full_token": "day-o",\n                      "tap": "homebrew/cask",\n                      "url": "https://shauninman.com/assets/downloads/Day-3.0.zip",\n                      "version": "3.0.1"\n                    },\n                    {\n                      "appcast": null,\n                      "artifacts": [\n                        [\n                          "iTerm.app"\n                        ],\n                        {\n                          "signal": {},\n                          "trash": [\n                            "~/Library/Application Support/iTerm",\n                            "~/Library/Application Support/iTerm2",\n                            "~/Library/Application Support/com.apple.sharedfilelist/com.apple.LSSharedFileList.ApplicationRecentDocuments/com.googlecode.iterm2.sfl*",\n                            "~/Library/Caches/com.googlecode.iterm2",\n                            "~/Library/Preferences/com.googlecode.iterm2.plist",\n                            "~/Library/Saved Application State/com.googlecode.iterm2.savedState"\n                          ]\n                        }\n                      ],\n                      "auto_updates": true,\n                      "caveats": null,\n                      "conflicts_with": {\n                        "cask": [\n                          "iterm2-beta"\n                        ]\n                      },\n                      "container": null,\n                      "depends_on": {\n                        "macos": {\n                          ">=": [\n                            "10.12"\n                          ]\n                        }\n                      },\n                      "desc": "Terminal emulator as alternative to Apple\'s Terminal app",\n                      "homepage": "https://www.iterm2.com/",\n                      "installed": "3.4.3",\n                      "name": [\n                        "iTerm2"\n                      ],\n                      "outdated": false,\n                      "sha256": "9ed73844838bddf797eadf37e5f7da3771308c3f74d38cd422c18eebaaa8f6b9",\n                      "token": "iterm2",\n                      "full_token": "custom/tap/iterm2",\n                      "tap": "custom/tap",\n                      "url": "https://iterm2.com/downloads/stable/iTerm2-3_4_3.zip",\n                      "version": "3.4.3"\n                    },\n                    {\n                      "token": "discord",\n                      "full_token": "discord",\n                      "tap": null,\n                      "name": [\n                        "Discord"\n                      ],\n                      "desc": "Voice and text chat software",\n                      "homepage": "https://discord.com/",\n                      "url": "https://dl.discordapp.net/apps/osx/0.0.268/Discord.dmg",\n                      "appcast": null,\n                      "version": "0.0.268",\n                      "versions": {\n                      },\n                      "installed": "0.0.266",\n                      "outdated": false,\n                      "sha256": "dfe12315b717ed06ac24d3eaacb700618e96cbb449ed63d2afadcdb70ad09c55",\n                      "artifacts": [\n                        {\n                          "app": [\n                            "Discord.app"\n                          ]\n                        },\n                        {\n                          "zap": [\n                            {\n                              "trash": [\n                                "~/Library/Application Support/discord",\n                                "~/Library/Caches/com.hnc.Discord",\n                                "~/Library/Caches/com.hnc.Discord.ShipIt",\n                                "~/Library/Cookies/com.hnc.Discord.binarycookies",\n                                "~/Library/Preferences/com.hnc.Discord.helper.plist",\n                                "~/Library/Preferences/com.hnc.Discord.plist",\n                                "~/Library/Saved Application State/com.hnc.Discord.savedState"\n                              ]\n                            }\n                          ]\n                        }\n                      ],\n                      "caveats": null,\n                      "depends_on": {\n                      },\n                      "conflicts_with": null,\n                      "container": null,\n                      "auto_updates": true\n                    }\n                  ],\n                  "formulae": [\n                    {\n                      "aliases": [],\n                      "bottle": {\n                        "stable": {\n                          "cellar": ":any",\n                          "files": {\n                            "arm64_big_sur": {\n                              "sha256": "674b3ae41c399f1e8e44c271b0e6909babff9fcd2e04a2127d25e2407ea4dd33",\n                              "url": "https://homebrew.bintray.com/bottles/jq-1.6.arm64_big_sur.bottle.1.tar.gz"\n                            },\n                            "big_sur": {\n                              "sha256": "bf0f8577632af7b878b6425476f5b1ab9c3bf66d65affb0c455048a173a0b6bf",\n                              "url": "https://homebrew.bintray.com/bottles/jq-1.6.big_sur.bottle.1.tar.gz"\n                            },\n                            "catalina": {\n                              "sha256": "820a3c85fcbb63088b160c7edf125d7e55fc2c5c1d51569304499c9cc4b89ce8",\n                              "url": "https://homebrew.bintray.com/bottles/jq-1.6.catalina.bottle.1.tar.gz"\n                            },\n                            "high_sierra": {\n                              "sha256": "dffcffa4ea13e8f0f2b45c5121e529077e135ae9a47254c32182231662ee9b72",\n                              "url": "https://homebrew.bintray.com/bottles/jq-1.6.high_sierra.bottle.1.tar.gz"\n                            },\n                            "mojave": {\n                              "sha256": "71f0e76c5b22e5088426c971d5e795fe67abee7af6c2c4ae0cf4c0eb98ed21ff",\n                              "url": "https://homebrew.bintray.com/bottles/jq-1.6.mojave.bottle.1.tar.gz"\n                            },\n                            "sierra": {\n                              "sha256": "bb4d19dc026c2d72c53eed78eaa0ab982e9fcad2cd2acc6d13e7a12ff658e877",\n                              "url": "https://homebrew.bintray.com/bottles/jq-1.6.sierra.bottle.1.tar.gz"\n                            }\n                          },\n                          "prefix": "/usr/local",\n                          "rebuild": 1,\n                          "root_url": "https://homebrew.bintray.com/bottles"\n                        }\n                      },\n                      "bottle_disabled": false,\n                      "build_dependencies": [],\n                      "caveats": null,\n                      "conflicts_with": [],\n                      "dependencies": [\n                        "oniguruma"\n                      ],\n                      "deprecated": false,\n                      "deprecation_date": null,\n                      "deprecation_reason": null,\n                      "desc": "Lightweight and flexible command-line JSON processor",\n                      "disable_date": null,\n                      "disable_reason": null,\n                      "disabled": false,\n                      "full_name": "jq",\n                      "homepage": "https://stedolan.github.io/jq/",\n                      "installed": [\n                        {\n                          "built_as_bottle": true,\n                          "installed_as_dependency": false,\n                          "installed_on_request": true,\n                          "poured_from_bottle": true,\n                          "runtime_dependencies": [\n                            {\n                              "full_name": "oniguruma",\n                              "version": "6.9.6"\n                            }\n                          ],\n                          "used_options": [],\n                          "version": "1.6"\n                        }\n                      ],\n                      "keg_only": false,\n                      "license": "MIT",\n                      "linked_keg": "1.6",\n                      "name": "jq",\n                      "oldname": null,\n                      "optional_dependencies": [],\n                      "options": [],\n                      "outdated": false,\n                      "pinned": false,\n                      "recommended_dependencies": [],\n                      "requirements": [],\n                      "revision": 0,\n                      "urls": {\n                        "stable": {\n                          "revision": null,\n                          "tag": null,\n                          "url": "https://github.com/stedolan/jq/releases/download/jq-1.6/jq-1.6.tar.gz"\n                        }\n                      },\n                      "uses_from_macos": [],\n                      "version_scheme": 0,\n                      "versioned_formulae": [],\n                      "versions": {\n                        "bottle": true,\n                        "head": "HEAD",\n                        "stable": "1.6"\n                      }\n                    },\n                    {\n                      "aliases": [],\n                      "bottle": {\n                        "stable": {\n                          "cellar": ":any",\n                          "files": {\n                            "arm64_big_sur": {\n                              "sha256": "c84206005787304416ed81094bd3a0cdd2ae8eb62649db5a3a44fa14b276d09f",\n                              "url": "https://homebrew.bintray.com/bottles/xz-5.2.5.arm64_big_sur.bottle.tar.gz"\n                            },\n                            "big_sur": {\n                              "sha256": "4fbd4a9e3eb49c27e83bd125b0e76d386c0e12ae1139d4dc9e31841fb8880a35",\n                              "url": "https://homebrew.bintray.com/bottles/xz-5.2.5.big_sur.bottle.tar.gz"\n                            },\n                            "catalina": {\n                              "sha256": "2dcc8e0121c934d1e34ffdb37fcd70f0f7b5c2f4755f2f7cbcf360e9e54cb43b",\n                              "url": "https://homebrew.bintray.com/bottles/xz-5.2.5.catalina.bottle.tar.gz"\n                            },\n                            "high_sierra": {\n                              "sha256": "1491b2b20c40c3cb0b990f520768d7e876e4ab4a7dc1da9994d0150da34ba5c6",\n                              "url": "https://homebrew.bintray.com/bottles/xz-5.2.5.high_sierra.bottle.tar.gz"\n                            },\n                            "mojave": {\n                              "sha256": "44483961b5d2b535b0ece1936c9d40b4bc7d9c7281646cca0fb476291ab9d4dc",\n                              "url": "https://homebrew.bintray.com/bottles/xz-5.2.5.mojave.bottle.tar.gz"\n                            }\n                          },\n                          "prefix": "/usr/local",\n                          "rebuild": 0,\n                          "root_url": "https://homebrew.bintray.com/bottles"\n                        }\n                      },\n                      "bottle_disabled": false,\n                      "build_dependencies": [],\n                      "caveats": null,\n                      "conflicts_with": [],\n                      "dependencies": [],\n                      "deprecated": false,\n                      "deprecation_date": null,\n                      "deprecation_reason": null,\n                      "desc": "General-purpose data compression with high compression ratio",\n                      "disable_date": null,\n                      "disable_reason": null,\n                      "disabled": false,\n                      "full_name": "xz",\n                      "homepage": "https://tukaani.org/xz/",\n                      "installed": [\n                        {\n                          "built_as_bottle": true,\n                          "installed_as_dependency": true,\n                          "installed_on_request": false,\n                          "poured_from_bottle": true,\n                          "runtime_dependencies": [],\n                          "used_options": [],\n                          "version": "5.2.5"\n                        }\n                      ],\n                      "keg_only": false,\n                      "license": "GPL-2.0",\n                      "linked_keg": "5.2.5",\n                      "name": "xz",\n                      "oldname": null,\n                      "optional_dependencies": [],\n                      "options": [],\n                      "outdated": false,\n                      "pinned": false,\n                      "recommended_dependencies": [],\n                      "requirements": [],\n                      "revision": 0,\n                      "urls": {\n                        "stable": {\n                          "revision": null,\n                          "tag": null,\n                          "url": "https://downloads.sourceforge.net/project/lzmautils/xz-5.2.5.tar.gz"\n                        }\n                      },\n                      "uses_from_macos": [],\n                      "version_scheme": 0,\n                      "versioned_formulae": [],\n                      "versions": {\n                        "bottle": true,\n                        "head": null,\n                        "stable": "5.2.5"\n                      }\n                    }\n                  ]\n                }\n                '), 'stderr': '', 'retcode': 0}
    return result

def custom_add_pkg(ret, name, newest_version):
    if False:
        for i in range(10):
            print('nop')
    ret[name] = newest_version
    return ret

def test_list_taps(TAPS_STRING, TAPS_LIST, HOMEBREW_BIN):
    if False:
        return 10
    '\n    Tests the return of the list of taps\n    '
    mock_taps = MagicMock(return_value={'stdout': TAPS_STRING, 'retcode': 0})
    mock_user = MagicMock(return_value='foo')
    mock_cmd = MagicMock(return_value='')
    with patch('salt.utils.path.which', MagicMock(return_value=HOMEBREW_BIN)):
        with patch.dict(mac_brew.__salt__, {'file.get_user': mock_user, 'cmd.run_all': mock_taps, 'cmd.run': mock_cmd}):
            assert mac_brew._list_taps() == TAPS_LIST

def test_tap_installed(TAPS_LIST):
    if False:
        i = 10
        return i + 15
    '\n    Tests if tap argument is already installed or not\n    '
    with patch('salt.modules.mac_brew_pkg._list_taps', MagicMock(return_value=TAPS_LIST)):
        assert mac_brew._tap('homebrew/science')

def test_tap_failure(HOMEBREW_BIN):
    if False:
        return 10
    '\n    Tests if the tap installation failed\n    '
    mock_failure = MagicMock(return_value={'stdout': '', 'stderr': '', 'retcode': 1})
    mock_user = MagicMock(return_value='foo')
    mock_cmd = MagicMock(return_value='')
    with patch('salt.utils.path.which', MagicMock(return_value=HOMEBREW_BIN)):
        with patch.dict(mac_brew.__salt__, {'cmd.run_all': mock_failure, 'file.get_user': mock_user, 'cmd.run': mock_cmd}), patch('salt.modules.mac_brew_pkg._list_taps', MagicMock(return_value={})):
            assert not mac_brew._tap('homebrew/test')

def test_tap(TAPS_LIST, HOMEBREW_BIN):
    if False:
        for i in range(10):
            print('nop')
    '\n    Tests adding unofficial GitHub repos to the list of brew taps\n    '
    mock_failure = MagicMock(return_value={'retcode': 0})
    mock_user = MagicMock(return_value='foo')
    mock_cmd = MagicMock(return_value='')
    with patch('salt.utils.path.which', MagicMock(return_value=HOMEBREW_BIN)):
        with patch.dict(mac_brew.__salt__, {'cmd.run_all': mock_failure, 'file.get_user': mock_user, 'cmd.run': mock_cmd}), patch('salt.modules.mac_brew_pkg._list_taps', MagicMock(return_value=TAPS_LIST)):
            assert mac_brew._tap('homebrew/test')

def test_homebrew_bin(HOMEBREW_BIN):
    if False:
        for i in range(10):
            print('nop')
    '\n    Tests the path to the homebrew binary\n    '
    mock_path = MagicMock(return_value='/usr/local')
    with patch('salt.utils.path.which', MagicMock(return_value=HOMEBREW_BIN)):
        with patch.dict(mac_brew.__salt__, {'cmd.run': mock_path}):
            assert mac_brew._homebrew_bin() == HOMEBREW_BIN

def test_list_pkgs_removed():
    if False:
        print('Hello World!')
    '\n    Tests removed implementation\n    '
    assert mac_brew.list_pkgs(removed=True) == {}

def test_list_pkgs_versions_true():
    if False:
        return 10
    '\n    Tests if pkg.list_pkgs is already in context and is a list\n    '
    mock_context = {'foo': ['bar']}
    with patch.dict(mac_brew.__context__, {'pkg.list_pkgs': mock_context}):
        assert mac_brew.list_pkgs(versions_as_list=True) == mock_context

def test_list_pkgs_homebrew_cask_pakages():
    if False:
        i = 10
        return i + 15
    '\n    Tests if pkg.list_pkgs list properly homebrew cask packages\n    '
    expected_pkgs = {'homebrew/cask/day-o': '3.0.1', 'day-o': '3.0.1', 'homebrew/cask/discord': '0.0.266', 'discord': '0.0.266', 'custom/tap/iterm2': '3.4.3', 'iterm2': '3.4.3', 'jq': '1.6', 'xz': '5.2.5'}
    with patch('salt.modules.mac_brew_pkg._call_brew', custom_call_brew), patch.dict(mac_brew.__salt__, {'pkg_resource.add_pkg': custom_add_pkg, 'pkg_resource.sort_pkglist': MagicMock()}):
        assert mac_brew.list_pkgs(versions_as_list=True) == expected_pkgs

def test_list_pkgs_no_context():
    if False:
        return 10
    '\n    Tests removed implementation\n    '
    expected_pkgs = {'zsh': '5.7.1', 'homebrew/cask/macvim': '8.1.151', 'homebrew/cask-fonts/font-firacode-nerd-font': '2.0.0'}
    with patch('salt.modules.mac_brew_pkg._call_brew', custom_call_brew), patch.dict(mac_brew.__salt__, {'pkg_resource.add_pkg': custom_add_pkg, 'pkg_resource.sort_pkglist': MagicMock()}), patch.object(mac_brew, '_list_pkgs_from_context') as list_pkgs_context_mock:
        pkgs = mac_brew.list_pkgs(versions_as_list=True, use_context=False)
        list_pkgs_context_mock.assert_not_called()
        list_pkgs_context_mock.reset_mock()
        pkgs = mac_brew.list_pkgs(versions_as_list=True, use_context=False)
        list_pkgs_context_mock.assert_not_called()
        list_pkgs_context_mock.reset_mock()

def test_version():
    if False:
        print('Hello World!')
    '\n    Tests version name returned\n    '
    mock_version = MagicMock(return_value='0.1.5')
    with patch.dict(mac_brew.__salt__, {'pkg_resource.version': mock_version}):
        assert mac_brew.version('foo') == '0.1.5'

def test_remove():
    if False:
        for i in range(10):
            print('nop')
    '\n    Tests if package to be removed exists\n    '
    mock_params = MagicMock(return_value=({'foo': None}, 'repository'))
    with patch('salt.modules.mac_brew_pkg.list_pkgs', return_value={'test': '0.1.5'}), patch.dict(mac_brew.__salt__, {'pkg_resource.parse_targets': mock_params}):
        assert mac_brew.remove('foo') == {}

def test_refresh_db_failure(HOMEBREW_BIN):
    if False:
        return 10
    '\n    Tests an update of homebrew package repository failure\n    '
    mock_user = MagicMock(return_value='foo')
    mock_failure = MagicMock(return_value={'stdout': '', 'stderr': '', 'retcode': 1})
    with patch('salt.utils.path.which', MagicMock(return_value='/usr/local/bin/brew')):
        with patch.dict(mac_brew.__salt__, {'file.get_user': mock_user, 'cmd.run_all': mock_failure}), patch('salt.modules.mac_brew_pkg._homebrew_bin', MagicMock(return_value=HOMEBREW_BIN)):
            with patch.object(salt.utils.pkg, 'clear_rtag', Mock()):
                pytest.raises(CommandExecutionError, mac_brew.refresh_db)

def test_refresh_db(HOMEBREW_BIN):
    if False:
        return 10
    '\n    Tests a successful update of homebrew package repository\n    '
    mock_user = MagicMock(return_value='foo')
    mock_success = MagicMock(return_value={'retcode': 0})
    with patch('salt.utils.path.which', MagicMock(return_value=HOMEBREW_BIN)):
        with patch.dict(mac_brew.__salt__, {'file.get_user': mock_user, 'cmd.run_all': mock_success}), patch('salt.modules.mac_brew_pkg._homebrew_bin', MagicMock(return_value=HOMEBREW_BIN)):
            with patch.object(salt.utils.pkg, 'clear_rtag', Mock()):
                assert mac_brew.refresh_db()

def test_install():
    if False:
        return 10
    '\n    Tests if package to be installed exists\n    '
    mock_params = MagicMock(return_value=[None, None])
    with patch.dict(mac_brew.__salt__, {'pkg_resource.parse_targets': mock_params}):
        assert mac_brew.install('name=foo') == {}

def test_hold(HOMEBREW_BIN):
    if False:
        return 10
    '\n    Tests holding if package is installed\n    '
    mock_user = MagicMock(return_value='foo')
    mock_cmd = MagicMock(return_value='')
    mock_cmd_all = MagicMock(return_value={'pid': 12345, 'retcode': 0, 'stderr': '', 'stdout': ''})
    _expected = {'foo': {'changes': {'new': 'hold', 'old': 'install'}, 'comment': 'Package foo is now being held.', 'name': 'foo', 'result': True}}
    mock_params = MagicMock(return_value=({'foo': None}, 'repository'))
    with patch('salt.utils.path.which', MagicMock(return_value=HOMEBREW_BIN)):
        with patch('salt.modules.mac_brew_pkg.list_pkgs', return_value={'foo': '0.1.5'}), patch.dict(mac_brew.__salt__, {'file.get_user': mock_user, 'pkg_resource.parse_targets': mock_params, 'cmd.run_all': mock_cmd_all, 'cmd.run': mock_cmd}):
            assert mac_brew.hold('foo') == _expected

def test_hold_not_installed(HOMEBREW_BIN):
    if False:
        for i in range(10):
            print('nop')
    '\n    Tests holding if package is not installed\n    '
    mock_user = MagicMock(return_value='foo')
    mock_cmd = MagicMock(return_value='')
    mock_cmd_all = MagicMock(return_value={'pid': 12345, 'retcode': 0, 'stderr': '', 'stdout': ''})
    _expected = {'foo': {'changes': {}, 'comment': 'Package foo does not have a state.', 'name': 'foo', 'result': False}}
    mock_params = MagicMock(return_value=({'foo': None}, 'repository'))
    with patch('salt.utils.path.which', MagicMock(return_value=HOMEBREW_BIN)):
        with patch('salt.modules.mac_brew_pkg.list_pkgs', return_value={}), patch.dict(mac_brew.__salt__, {'file.get_user': mock_user, 'pkg_resource.parse_targets': mock_params, 'cmd.run_all': mock_cmd_all, 'cmd.run': mock_cmd}):
            assert mac_brew.hold('foo') == _expected

def test_hold_pinned():
    if False:
        return 10
    '\n    Tests holding if package is already pinned\n    '
    mock_user = MagicMock(return_value='foo')
    mock_cmd = MagicMock(return_value='')
    mock_cmd_all = MagicMock(return_value={'pid': 12345, 'retcode': 0, 'stderr': '', 'stdout': ''})
    _expected = {'foo': {'changes': {}, 'comment': 'Package foo is already set to be held.', 'name': 'foo', 'result': True}}
    mock_params = MagicMock(return_value=({'foo': None}, 'repository'))
    with patch('salt.modules.mac_brew_pkg.list_pkgs', return_value={'foo': '0.1.5'}), patch('salt.modules.mac_brew_pkg._list_pinned', return_value=['foo']), patch.dict(mac_brew.__salt__, {'file.get_user': mock_user, 'pkg_resource.parse_targets': mock_params, 'cmd.run_all': mock_cmd_all, 'cmd.run': mock_cmd}):
        assert mac_brew.hold('foo') == _expected

def test_unhold(HOMEBREW_BIN):
    if False:
        for i in range(10):
            print('nop')
    '\n    Tests unholding if package is installed\n    '
    mock_user = MagicMock(return_value='foo')
    mock_cmd = MagicMock(return_value='')
    mock_cmd_all = MagicMock(return_value={'pid': 12345, 'retcode': 0, 'stderr': '', 'stdout': ''})
    _expected = {'foo': {'changes': {'new': 'install', 'old': 'hold'}, 'comment': 'Package foo is no longer being held.', 'name': 'foo', 'result': True}}
    mock_params = MagicMock(return_value=({'foo': None}, 'repository'))
    with patch('salt.utils.path.which', MagicMock(return_value=HOMEBREW_BIN)):
        with patch('salt.modules.mac_brew_pkg.list_pkgs', return_value={'foo': '0.1.5'}), patch('salt.modules.mac_brew_pkg._list_pinned', return_value=['foo']), patch.dict(mac_brew.__salt__, {'file.get_user': mock_user, 'pkg_resource.parse_targets': mock_params, 'cmd.run_all': mock_cmd_all, 'cmd.run': mock_cmd}):
            assert mac_brew.unhold('foo') == _expected

def test_unhold_not_installed():
    if False:
        i = 10
        return i + 15
    '\n    Tests unholding if package is not installed\n    '
    mock_user = MagicMock(return_value='foo')
    mock_cmd = MagicMock(return_value='')
    mock_cmd_all = MagicMock(return_value={'pid': 12345, 'retcode': 0, 'stderr': '', 'stdout': ''})
    _expected = {'foo': {'changes': {}, 'comment': 'Package foo does not have a state.', 'name': 'foo', 'result': False}}
    mock_params = MagicMock(return_value=({'foo': None}, 'repository'))
    with patch('salt.modules.mac_brew_pkg.list_pkgs', return_value={}), patch('salt.modules.mac_brew_pkg._list_pinned', return_value=['foo']), patch.dict(mac_brew.__salt__, {'file.get_user': mock_user, 'pkg_resource.parse_targets': mock_params, 'cmd.run_all': mock_cmd_all, 'cmd.run': mock_cmd}):
        assert mac_brew.unhold('foo') == _expected

def test_unhold_not_pinned():
    if False:
        return 10
    '\n    Tests unholding if package is not installed\n    '
    mock_user = MagicMock(return_value='foo')
    mock_cmd = MagicMock(return_value='')
    mock_cmd_all = MagicMock(return_value={'pid': 12345, 'retcode': 0, 'stderr': '', 'stdout': ''})
    _expected = {'foo': {'changes': {}, 'comment': 'Package foo is already set not to be held.', 'name': 'foo', 'result': True}}
    mock_params = MagicMock(return_value=({'foo': None}, 'repository'))
    with patch('salt.modules.mac_brew_pkg.list_pkgs', return_value={'foo': '0.1.5'}), patch('salt.modules.mac_brew_pkg._list_pinned', return_value=[]), patch.dict(mac_brew.__salt__, {'file.get_user': mock_user, 'pkg_resource.parse_targets': mock_params, 'cmd.run_all': mock_cmd_all, 'cmd.run': mock_cmd}):
        assert mac_brew.unhold('foo') == _expected

def test_info_installed(HOMEBREW_BIN):
    if False:
        for i in range(10):
            print('nop')
    '\n    Tests info_installed method\n    '
    mock_user = MagicMock(return_value='foo')
    mock_cmd = MagicMock(return_value='')
    mock_cmd_all = MagicMock(return_value={'pid': 12345, 'retcode': 0, 'stderr': '', 'stdout': textwrap.dedent('                {\n                  "formulae": [\n                    {\n                      "name": "salt",\n                      "full_name": "cdalvaro/tap/salt",\n                      "tap": "cdalvaro/tap",\n                      "aliases": []\n                    },\n                    {\n                      "name": "vim",\n                      "full_name": "vim",\n                      "tap": "homebrew/core",\n                      "aliases": []\n                    }\n                  ],\n                  "casks": [\n                    {\n                      "token": "visual-studio-code",\n                      "full_token": "visual-studio-code",\n                      "tap": null,\n                      "name": [\n                        "MicrosoftVisualStudioCode",\n                        "VSCode"\n                      ]\n                    }\n                  ]\n                }\n             ')})
    _expected = {'cdalvaro/tap/salt': {'name': 'salt', 'full_name': 'cdalvaro/tap/salt', 'tap': 'cdalvaro/tap', 'aliases': []}, 'vim': {'name': 'vim', 'full_name': 'vim', 'tap': 'homebrew/core', 'aliases': []}, 'visual-studio-code': {'token': 'visual-studio-code', 'full_token': 'visual-studio-code', 'tap': None, 'name': ['MicrosoftVisualStudioCode', 'VSCode']}}
    with patch('salt.utils.path.which', MagicMock(return_value=HOMEBREW_BIN)):
        with patch('salt.modules.mac_brew_pkg.list_pkgs', return_value={}), patch('salt.modules.mac_brew_pkg._list_pinned', return_value=['foo']), patch.dict(mac_brew.__salt__, {'file.get_user': mock_user, 'cmd.run_all': mock_cmd_all, 'cmd.run': mock_cmd}):
            assert mac_brew.info_installed('cdalvaro/tap/salt', 'vim', 'visual-studio-code') == _expected

def test_list_upgrades(HOMEBREW_BIN):
    if False:
        while True:
            i = 10
    '\n    Tests list_upgrades method\n    '
    mock_user = MagicMock(return_value='foo')
    mock_cmd = MagicMock(return_value='')
    mock_cmd_all = MagicMock(return_value={'pid': 12345, 'retcode': 0, 'stderr': '', 'stdout': textwrap.dedent('                {\n                  "formulae": [\n                    {\n                      "name": "cmake",\n                      "installed_versions": ["3.19.3"],\n                      "current_version": "3.19.4",\n                      "pinned": false,\n                      "pinned_version": null\n                    },\n                    {\n                      "name": "fzf",\n                      "installed_versions": ["0.25.0"],\n                      "current_version": "0.25.1",\n                      "pinned": false,\n                      "pinned_version": null\n                    }\n                  ],\n                  "casks": [\n                    {\n                      "name": "ksdiff",\n                      "installed_versions": "2.2.0,122",\n                      "current_version": "2.3.6,123-jan-18-2021"\n                    }\n                  ]\n                }\n                ')})
    _expected = {'cmake': '3.19.4', 'fzf': '0.25.1', 'ksdiff': '2.3.6,123-jan-18-2021'}
    with patch('salt.utils.path.which', MagicMock(return_value=HOMEBREW_BIN)):
        with patch('salt.modules.mac_brew_pkg.list_pkgs', return_value={}), patch('salt.modules.mac_brew_pkg._list_pinned', return_value=['foo']), patch.dict(mac_brew.__salt__, {'file.get_user': mock_user, 'cmd.run_all': mock_cmd_all, 'cmd.run': mock_cmd}):
            assert mac_brew.list_upgrades(refresh=False, include_casks=True) == _expected