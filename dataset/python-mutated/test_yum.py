from __future__ import annotations
import unittest
from ansible.modules.yum import YumModule
yum_plugin_load_error = '\nPlugin "product-id" can\'t be imported\nPlugin "search-disabled-repos" can\'t be imported\nPlugin "subscription-manager" can\'t be imported\nPlugin "product-id" can\'t be imported\nPlugin "search-disabled-repos" can\'t be imported\nPlugin "subscription-manager" can\'t be imported\n'
wrapped_output_1 = '\nЗагружены модули: fastestmirror\nLoading mirror speeds from cached hostfile\n * base: mirror.h1host.ru\n * extras: mirror.h1host.ru\n * updates: mirror.h1host.ru\n\nvms-agent.x86_64                            0.0-9                            dev\n'
wrapped_output_2 = '\nЗагружены модули: fastestmirror\nLoading mirror speeds from cached hostfile\n * base: mirror.corbina.net\n * extras: mirror.corbina.net\n * updates: mirror.corbina.net\n\nempty-empty-empty-empty-empty-empty-empty-empty-empty-empty-empty-empty-empty-empty-empty-empty-empty-empty-empty-empty.x86_64\n                                    0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.1-0\n                                                                         addons\nlibtiff.x86_64                      4.0.3-27.el7_3                       updates\n'
wrapped_output_3 = '\nLoaded plugins: fastestmirror, langpacks\nLoading mirror speeds from cached hostfile\n\nceph.x86_64                               1:11.2.0-0.el7                    ceph\nceph-base.x86_64                          1:11.2.0-0.el7                    ceph\nceph-common.x86_64                        1:11.2.0-0.el7                    ceph\nceph-mds.x86_64                           1:11.2.0-0.el7                    ceph\nceph-mon.x86_64                           1:11.2.0-0.el7                    ceph\nceph-osd.x86_64                           1:11.2.0-0.el7                    ceph\nceph-selinux.x86_64                       1:11.2.0-0.el7                    ceph\nlibcephfs1.x86_64                         1:11.0.2-0.el7                    ceph\nlibrados2.x86_64                          1:11.2.0-0.el7                    ceph\nlibradosstriper1.x86_64                   1:11.2.0-0.el7                    ceph\nlibrbd1.x86_64                            1:11.2.0-0.el7                    ceph\nlibrgw2.x86_64                            1:11.2.0-0.el7                    ceph\npython-cephfs.x86_64                      1:11.2.0-0.el7                    ceph\npython-rados.x86_64                       1:11.2.0-0.el7                    ceph\npython-rbd.x86_64                         1:11.2.0-0.el7                    ceph\n'
wrapped_output_4 = '\nipxe-roms-qemu.noarch                 20160127-1.git6366fa7a.el7\n                                                            rhelosp-9.0-director-puddle\nquota.x86_64                          1:4.01-11.el7_2.1     rhelosp-rhel-7.2-z\nquota-nls.noarch                      1:4.01-11.el7_2.1     rhelosp-rhel-7.2-z\nrdma.noarch                           7.2_4.1_rc6-2.el7     rhelosp-rhel-7.2-z\nscreen.x86_64                         4.1.0-0.23.20120314git3c2946.el7_2\n                                                            rhelosp-rhel-7.2-z\nsos.noarch                            3.2-36.el7ost.2       rhelosp-9.0-puddle\nsssd-client.x86_64                    1.13.0-40.el7_2.12    rhelosp-rhel-7.2-z\n'
unwrapped_output_rhel7 = '\n\nLoaded plugins: etckeeper, product-id, search-disabled-repos, subscription-\n              : manager\nThis system is not registered to Red Hat Subscription Management. You can use subscription-manager to register.\n\nNetworkManager-openvpn.x86_64         1:1.2.6-1.el7           epel\nNetworkManager-openvpn-gnome.x86_64   1:1.2.6-1.el7           epel\ncabal-install.x86_64                  1.16.1.0-2.el7          epel\ncgit.x86_64                           1.1-1.el7               epel\npython34-libs.x86_64                  3.4.5-3.el7             epel\npython34-test.x86_64                  3.4.5-3.el7             epel\npython34-tkinter.x86_64               3.4.5-3.el7             epel\npython34-tools.x86_64                 3.4.5-3.el7             epel\nqgit.x86_64                           2.6-4.el7               epel\nrdiff-backup.x86_64                   1.2.8-12.el7            epel\nstoken-libs.x86_64                    0.91-1.el7              epel\nxlockmore.x86_64                      5.49-2.el7              epel\n'
wrapped_output_rhel7_obsoletes_postfix = '\nObsoleting Packages\nddashboard.x86_64                     0.2.0.1-1.el7_3         mhlavink-developerdashboard\n    developerdashboard.x86_64         0.1.12.2-1.el7_2        @mhlavink-developerdashboard\npython-bugzilla.noarch                1.2.2-3.el7_2.1         mhlavink-developerdashboard\n    python-bugzilla-develdashboardfixes.noarch\n                                      1.2.2-3.el7             @mhlavink-developerdashboard\npython2-futures.noarch                3.0.5-1.el7             epel\n    python-futures.noarch             3.0.3-1.el7             @epel\npython2-pip.noarch                    8.1.2-5.el7             epel\n    python-pip.noarch                 7.1.0-1.el7             @epel\npython2-pyxdg.noarch                  0.25-6.el7              epel\n    pyxdg.noarch                      0.25-5.el7              @epel\npython2-simplejson.x86_64             3.10.0-1.el7            epel\n    python-simplejson.x86_64          3.3.3-1.el7             @epel\nSecurity: kernel-3.10.0-327.28.2.el7.x86_64 is an installed security update\nSecurity: kernel-3.10.0-327.22.2.el7.x86_64 is the currently running version\n'
wrapped_output_multiple_empty_lines = '\nLoaded plugins: langpacks, product-id, search-disabled-repos, subscription-manager\n\nThis system is not registered with an entitlement server. You can use subscription-manager to register.\n\n\nscreen.x86_64                         4.1.0-0.23.20120314git3c2946.el7_2\n                                                            rhelosp-rhel-7.2-z\nsos.noarch                            3.2-36.el7ost.2       rhelosp-9.0-puddle\n'
longname = '\nLoaded plugins: fastestmirror, priorities, rhnplugin\nThis system is receiving updates from RHN Classic or Red Hat Satellite.\nLoading mirror speeds from cached hostfile\n\nxxxxxxxxxxxxxxxxxxxxxxxxxx.noarch\n                        1.16-1            xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\nglibc.x86_64            2.17-157.el7_3.1  xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
unwrapped_output_rhel7_obsoletes = unwrapped_output_rhel7 + wrapped_output_rhel7_obsoletes_postfix
unwrapped_output_rhel7_expected_new_obsoletes_pkgs = ['ddashboard', 'python-bugzilla', 'python2-futures', 'python2-pip', 'python2-pyxdg', 'python2-simplejson']
unwrapped_output_rhel7_expected_old_obsoletes_pkgs = ['developerdashboard', 'python-bugzilla-develdashboardfixes', 'python-futures', 'python-pip', 'pyxdg', 'python-simplejson']
unwrapped_output_rhel7_expected_updated_pkgs = ['NetworkManager-openvpn', 'NetworkManager-openvpn-gnome', 'cabal-install', 'cgit', 'python34-libs', 'python34-test', 'python34-tkinter', 'python34-tools', 'qgit', 'rdiff-backup', 'stoken-libs', 'xlockmore']

class TestYumUpdateCheckParse(unittest.TestCase):

    def _assert_expected(self, expected_pkgs, result):
        if False:
            while True:
                i = 10
        for expected_pkg in expected_pkgs:
            self.assertIn(expected_pkg, result)
        self.assertEqual(len(result), len(expected_pkgs))
        self.assertIsInstance(result, dict)

    def test_empty_output(self):
        if False:
            i = 10
            return i + 15
        (res, obs) = YumModule.parse_check_update('')
        expected_pkgs = []
        self._assert_expected(expected_pkgs, res)

    def test_longname(self):
        if False:
            for i in range(10):
                print('nop')
        (res, obs) = YumModule.parse_check_update(longname)
        expected_pkgs = ['xxxxxxxxxxxxxxxxxxxxxxxxxx', 'glibc']
        self._assert_expected(expected_pkgs, res)

    def test_plugin_load_error(self):
        if False:
            i = 10
            return i + 15
        (res, obs) = YumModule.parse_check_update(yum_plugin_load_error)
        expected_pkgs = []
        self._assert_expected(expected_pkgs, res)

    def test_wrapped_output_1(self):
        if False:
            print('Hello World!')
        (res, obs) = YumModule.parse_check_update(wrapped_output_1)
        expected_pkgs = ['vms-agent']
        self._assert_expected(expected_pkgs, res)

    def test_wrapped_output_2(self):
        if False:
            return 10
        (res, obs) = YumModule.parse_check_update(wrapped_output_2)
        expected_pkgs = ['empty-empty-empty-empty-empty-empty-empty-empty-empty-empty-empty-empty-empty-empty-empty-empty-empty-empty-empty-empty', 'libtiff']
        self._assert_expected(expected_pkgs, res)

    def test_wrapped_output_3(self):
        if False:
            i = 10
            return i + 15
        (res, obs) = YumModule.parse_check_update(wrapped_output_3)
        expected_pkgs = ['ceph', 'ceph-base', 'ceph-common', 'ceph-mds', 'ceph-mon', 'ceph-osd', 'ceph-selinux', 'libcephfs1', 'librados2', 'libradosstriper1', 'librbd1', 'librgw2', 'python-cephfs', 'python-rados', 'python-rbd']
        self._assert_expected(expected_pkgs, res)

    def test_wrapped_output_4(self):
        if False:
            print('Hello World!')
        (res, obs) = YumModule.parse_check_update(wrapped_output_4)
        expected_pkgs = ['ipxe-roms-qemu', 'quota', 'quota-nls', 'rdma', 'screen', 'sos', 'sssd-client']
        self._assert_expected(expected_pkgs, res)

    def test_wrapped_output_rhel7(self):
        if False:
            print('Hello World!')
        (res, obs) = YumModule.parse_check_update(unwrapped_output_rhel7)
        self._assert_expected(unwrapped_output_rhel7_expected_updated_pkgs, res)

    def test_wrapped_output_rhel7_obsoletes(self):
        if False:
            i = 10
            return i + 15
        (res, obs) = YumModule.parse_check_update(unwrapped_output_rhel7_obsoletes)
        self._assert_expected(unwrapped_output_rhel7_expected_updated_pkgs + unwrapped_output_rhel7_expected_new_obsoletes_pkgs, res)
        self._assert_expected(unwrapped_output_rhel7_expected_old_obsoletes_pkgs, obs)

    def test_wrapped_output_multiple_empty_lines(self):
        if False:
            for i in range(10):
                print('nop')
        (res, obs) = YumModule.parse_check_update(wrapped_output_multiple_empty_lines)
        self._assert_expected(['screen', 'sos'], res)