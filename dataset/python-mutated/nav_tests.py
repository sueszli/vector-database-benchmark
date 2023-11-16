import sys
import unittest
from mkdocs.structure.files import File, Files, set_exclusions
from mkdocs.structure.nav import Section, _get_by_type, get_navigation
from mkdocs.structure.pages import Page
from mkdocs.tests.base import dedent, load_config

class SiteNavigationTests(unittest.TestCase):
    maxDiff = None

    def test_simple_nav(self):
        if False:
            while True:
                i = 10
        nav_cfg = [{'Home': 'index.md'}, {'About': 'about.md'}]
        expected = dedent("\n            Page(title='Home', url='/')\n            Page(title='About', url='/about/')\n            ")
        cfg = load_config(nav=nav_cfg, site_url='http://example.com/')
        fs = [File(list(item.values())[0], cfg.docs_dir, cfg.site_dir, cfg.use_directory_urls) for item in nav_cfg]
        files = Files(fs)
        site_navigation = get_navigation(files, cfg)
        self.assertEqual(str(site_navigation).strip(), expected)
        self.assertEqual(len(site_navigation.items), 2)
        self.assertEqual(len(site_navigation.pages), 2)
        self.assertEqual(repr(site_navigation.homepage), "Page(title='Home', url='/')")

    def test_nav_no_directory_urls(self):
        if False:
            i = 10
            return i + 15
        nav_cfg = [{'Home': 'index.md'}, {'About': 'about.md'}]
        expected = dedent("\n            Page(title='Home', url='/index.html')\n            Page(title='About', url='/about.html')\n            ")
        cfg = load_config(nav=nav_cfg, use_directory_urls=False, site_url='http://example.com/')
        fs = [File(list(item.values())[0], cfg.docs_dir, cfg.site_dir, cfg.use_directory_urls) for item in nav_cfg]
        files = Files(fs)
        site_navigation = get_navigation(files, cfg)
        self.assertEqual(str(site_navigation).strip(), expected)
        self.assertEqual(len(site_navigation.items), 2)
        self.assertEqual(len(site_navigation.pages), 2)
        self.assertEqual(repr(site_navigation.homepage), "Page(title='Home', url='/index.html')")

    def test_nav_missing_page(self):
        if False:
            for i in range(10):
                print('nop')
        nav_cfg = [{'Home': 'index.md'}]
        expected = dedent("\n            Page(title='Home', url='/')\n            ")
        cfg = load_config(nav=nav_cfg, site_url='http://example.com/')
        fs = [File('index.md', cfg.docs_dir, cfg.site_dir, cfg.use_directory_urls), File('page_not_in_nav.md', cfg.docs_dir, cfg.site_dir, cfg.use_directory_urls)]
        files = Files(fs)
        site_navigation = get_navigation(files, cfg)
        self.assertEqual(str(site_navigation).strip(), expected)
        self.assertEqual(len(site_navigation.items), 1)
        self.assertEqual(len(site_navigation.pages), 1)
        for file in files:
            self.assertIsInstance(file.page, Page)

    def test_nav_no_title(self):
        if False:
            while True:
                i = 10
        nav_cfg = ['index.md', {'About': 'about.md'}]
        expected = dedent("\n            Page(title=[blank], url='/')\n            Page(title='About', url='/about/')\n            ")
        cfg = load_config(nav=nav_cfg, site_url='http://example.com/')
        fs = [File(nav_cfg[0], cfg.docs_dir, cfg.site_dir, cfg.use_directory_urls), File(nav_cfg[1]['About'], cfg.docs_dir, cfg.site_dir, cfg.use_directory_urls)]
        files = Files(fs)
        site_navigation = get_navigation(files, cfg)
        self.assertEqual(str(site_navigation).strip(), expected)
        self.assertEqual(len(site_navigation.items), 2)
        self.assertEqual(len(site_navigation.pages), 2)

    def test_nav_external_links(self):
        if False:
            for i in range(10):
                print('nop')
        nav_cfg = [{'Home': 'index.md'}, {'Local': '/local.html'}, {'External': 'http://example.com/external.html'}]
        expected = dedent("\n            Page(title='Home', url='/')\n            Link(title='Local', url='/local.html')\n            Link(title='External', url='http://example.com/external.html')\n            ")
        cfg = load_config(nav=nav_cfg, site_url='http://example.com/')
        fs = [File('index.md', cfg.docs_dir, cfg.site_dir, cfg.use_directory_urls)]
        files = Files(fs)
        with self.assertLogs('mkdocs', level='DEBUG') as cm:
            site_navigation = get_navigation(files, cfg)
        self.assertEqual(cm.output, ["INFO:mkdocs.structure.nav:An absolute path to '/local.html' is included in the 'nav' configuration, which presumably points to an external resource.", "DEBUG:mkdocs.structure.nav:An external link to 'http://example.com/external.html' is included in the 'nav' configuration."])
        self.assertEqual(str(site_navigation).strip(), expected)
        self.assertEqual(len(site_navigation.items), 3)
        self.assertEqual(len(site_navigation.pages), 1)

    def test_nav_bad_links(self):
        if False:
            for i in range(10):
                print('nop')
        nav_cfg = [{'Home': 'index.md'}, {'Missing': 'missing.html'}, {'Bad External': 'example.com'}]
        expected = dedent("\n            Page(title='Home', url='/')\n            Link(title='Missing', url='missing.html')\n            Link(title='Bad External', url='example.com')\n            ")
        cfg = load_config(nav=nav_cfg, site_url='http://example.com/')
        fs = [File('index.md', cfg.docs_dir, cfg.site_dir, cfg.use_directory_urls)]
        files = Files(fs)
        with self.assertLogs('mkdocs') as cm:
            site_navigation = get_navigation(files, cfg)
        self.assertEqual(cm.output, ["WARNING:mkdocs.structure.nav:A relative path to 'missing.html' is included in the 'nav' configuration, which is not found in the documentation files.", "WARNING:mkdocs.structure.nav:A relative path to 'example.com' is included in the 'nav' configuration, which is not found in the documentation files."])
        self.assertEqual(str(site_navigation).strip(), expected)
        self.assertEqual(len(site_navigation.items), 3)
        self.assertEqual(len(site_navigation.pages), 1)

    def test_indented_nav(self):
        if False:
            while True:
                i = 10
        nav_cfg = [{'Home': 'index.md'}, {'API Guide': [{'Running': 'api-guide/running.md'}, {'Testing': 'api-guide/testing.md'}, {'Debugging': 'api-guide/debugging.md'}, {'Advanced': [{'Part 1': 'api-guide/advanced/part-1.md'}]}]}, {'About': [{'Release notes': 'about/release-notes.md'}, {'License': '/license.html'}]}, {'External': 'https://example.com/'}]
        expected = dedent("\n            Page(title='Home', url='/')\n            Section(title='API Guide')\n                Page(title='Running', url='/api-guide/running/')\n                Page(title='Testing', url='/api-guide/testing/')\n                Page(title='Debugging', url='/api-guide/debugging/')\n                Section(title='Advanced')\n                    Page(title='Part 1', url='/api-guide/advanced/part-1/')\n            Section(title='About')\n                Page(title='Release notes', url='/about/release-notes/')\n                Link(title='License', url='/license.html')\n            Link(title='External', url='https://example.com/')\n            ")
        cfg = load_config(nav=nav_cfg, site_url='http://example.com/')
        fs = ['index.md', 'api-guide/running.md', 'api-guide/testing.md', 'api-guide/debugging.md', 'api-guide/advanced/part-1.md', 'about/release-notes.md']
        files = Files([File(s, cfg.docs_dir, cfg.site_dir, cfg.use_directory_urls) for s in fs])
        site_navigation = get_navigation(files, cfg)
        self.assertEqual(str(site_navigation).strip(), expected)
        self.assertEqual(len(site_navigation.items), 4)
        self.assertEqual(len(site_navigation.pages), 6)
        self.assertEqual(repr(site_navigation.homepage), "Page(title='Home', url='/')")
        self.assertIsNone(site_navigation.items[0].parent)
        self.assertEqual(site_navigation.items[0].ancestors, [])
        self.assertIsNone(site_navigation.items[1].parent)
        self.assertEqual(site_navigation.items[1].ancestors, [])
        self.assertEqual(len(site_navigation.items[1].children), 4)
        self.assertEqual(repr(site_navigation.items[1].children[0].parent), "Section(title='API Guide')")
        self.assertEqual(site_navigation.items[1].children[0].ancestors, [site_navigation.items[1]])
        self.assertEqual(repr(site_navigation.items[1].children[1].parent), "Section(title='API Guide')")
        self.assertEqual(site_navigation.items[1].children[1].ancestors, [site_navigation.items[1]])
        self.assertEqual(repr(site_navigation.items[1].children[2].parent), "Section(title='API Guide')")
        self.assertEqual(site_navigation.items[1].children[2].ancestors, [site_navigation.items[1]])
        self.assertEqual(repr(site_navigation.items[1].children[3].parent), "Section(title='API Guide')")
        self.assertEqual(site_navigation.items[1].children[3].ancestors, [site_navigation.items[1]])
        self.assertEqual(len(site_navigation.items[1].children[3].children), 1)
        self.assertEqual(repr(site_navigation.items[1].children[3].children[0].parent), "Section(title='Advanced')")
        self.assertEqual(site_navigation.items[1].children[3].children[0].ancestors, [site_navigation.items[1].children[3], site_navigation.items[1]])
        self.assertIsNone(site_navigation.items[2].parent)
        self.assertEqual(len(site_navigation.items[2].children), 2)
        self.assertEqual(repr(site_navigation.items[2].children[0].parent), "Section(title='About')")
        self.assertEqual(site_navigation.items[2].children[0].ancestors, [site_navigation.items[2]])
        self.assertEqual(repr(site_navigation.items[2].children[1].parent), "Section(title='About')")
        self.assertEqual(site_navigation.items[2].children[1].ancestors, [site_navigation.items[2]])
        self.assertIsNone(site_navigation.items[3].parent)
        self.assertEqual(site_navigation.items[3].ancestors, [])
        self.assertIsNone(site_navigation.items[3].children)

    def test_nested_ungrouped_nav(self):
        if False:
            while True:
                i = 10
        nav_cfg = [{'Home': 'index.md'}, {'Contact': 'about/contact.md'}, {'License Title': 'about/sub/license.md'}]
        expected = dedent("\n            Page(title='Home', url='/')\n            Page(title='Contact', url='/about/contact/')\n            Page(title='License Title', url='/about/sub/license/')\n            ")
        cfg = load_config(nav=nav_cfg, site_url='http://example.com/')
        fs = [File(list(item.values())[0], cfg.docs_dir, cfg.site_dir, cfg.use_directory_urls) for item in nav_cfg]
        files = Files(fs)
        site_navigation = get_navigation(files, cfg)
        self.assertEqual(str(site_navigation).strip(), expected)
        self.assertEqual(len(site_navigation.items), 3)
        self.assertEqual(len(site_navigation.pages), 3)

    def test_nested_ungrouped_nav_no_titles(self):
        if False:
            print('Hello World!')
        nav_cfg = ['index.md', 'about/contact.md', 'about/sub/license.md']
        expected = dedent("\n            Page(title=[blank], url='/')\n            Page(title=[blank], url='/about/contact/')\n            Page(title=[blank], url='/about/sub/license/')\n            ")
        cfg = load_config(nav=nav_cfg, site_url='http://example.com/')
        fs = [File(item, cfg.docs_dir, cfg.site_dir, cfg.use_directory_urls) for item in nav_cfg]
        files = Files(fs)
        site_navigation = get_navigation(files, cfg)
        self.assertEqual(str(site_navigation).strip(), expected)
        self.assertEqual(len(site_navigation.items), 3)
        self.assertEqual(len(site_navigation.pages), 3)
        self.assertEqual(repr(site_navigation.homepage), "Page(title=[blank], url='/')")

    @unittest.skipUnless(sys.platform.startswith('win'), 'requires Windows')
    def test_nested_ungrouped_no_titles_windows(self):
        if False:
            while True:
                i = 10
        nav_cfg = ['index.md', 'about\\contact.md', 'about\\sub\\license.md']
        expected = dedent("\n            Page(title=[blank], url='/')\n            Page(title=[blank], url='/about/contact/')\n            Page(title=[blank], url='/about/sub/license/')\n            ")
        cfg = load_config(nav=nav_cfg, site_url='http://example.com/')
        fs = [File(item, cfg.docs_dir, cfg.site_dir, cfg.use_directory_urls) for item in nav_cfg]
        files = Files(fs)
        site_navigation = get_navigation(files, cfg)
        self.assertEqual(str(site_navigation).strip(), expected)
        self.assertEqual(len(site_navigation.items), 3)
        self.assertEqual(len(site_navigation.pages), 3)

    def test_nav_from_files(self):
        if False:
            return 10
        expected = dedent("\n            Page(title=[blank], url='/')\n            Page(title=[blank], url='/about/')\n            ")
        cfg = load_config(site_url='http://example.com/')
        fs = [File('index.md', cfg.docs_dir, cfg.site_dir, cfg.use_directory_urls), File('about.md', cfg.docs_dir, cfg.site_dir, cfg.use_directory_urls)]
        files = Files(fs)
        site_navigation = get_navigation(files, cfg)
        self.assertEqual(str(site_navigation).strip(), expected)
        self.assertEqual(len(site_navigation.items), 2)
        self.assertEqual(len(site_navigation.pages), 2)
        self.assertEqual(repr(site_navigation.homepage), "Page(title=[blank], url='/')")

    def test_nav_from_nested_files(self):
        if False:
            while True:
                i = 10
        expected = dedent("\n            Page(title=[blank], url='/')\n            Section(title='About')\n                Page(title=[blank], url='/about/license/')\n                Page(title=[blank], url='/about/release-notes/')\n            Section(title='Api guide')\n                Page(title=[blank], url='/api-guide/debugging/')\n                Page(title=[blank], url='/api-guide/running/')\n                Page(title=[blank], url='/api-guide/testing/')\n                Section(title='Advanced')\n                    Page(title=[blank], url='/api-guide/advanced/part-1/')\n            ")
        cfg = load_config(site_url='http://example.com/')
        fs = ['index.md', 'about/license.md', 'about/release-notes.md', 'api-guide/debugging.md', 'api-guide/running.md', 'api-guide/testing.md', 'api-guide/advanced/part-1.md']
        files = Files([File(s, cfg.docs_dir, cfg.site_dir, cfg.use_directory_urls) for s in fs])
        site_navigation = get_navigation(files, cfg)
        self.assertEqual(str(site_navigation).strip(), expected)
        self.assertEqual(len(site_navigation.items), 3)
        self.assertEqual(len(site_navigation.pages), 7)
        self.assertEqual(repr(site_navigation.homepage), "Page(title=[blank], url='/')")

    def test_nav_with_exclusion(self):
        if False:
            for i in range(10):
                print('nop')
        expected = dedent("\n            Page(title=[blank], url='index.html')\n            Section(title='About')\n                Page(title=[blank], url='about/license.html')\n                Page(title=[blank], url='about/release-notes.html')\n            Section(title='Api guide')\n                Page(title=[blank], url='api-guide/running.html')\n                Page(title=[blank], url='api-guide/testing.html')\n            ")
        cfg = load_config(use_directory_urls=False, not_in_nav='*ging.md\n/foo.md\n')
        fs = ['index.md', 'foo.md', 'about/license.md', 'about/release-notes.md', 'api-guide/debugging.md', 'api-guide/running.md', 'api-guide/testing.md']
        files = Files([File(s, cfg.docs_dir, cfg.site_dir, cfg.use_directory_urls) for s in fs])
        set_exclusions(files, cfg)
        site_navigation = get_navigation(files, cfg)
        self.assertEqual(str(site_navigation).strip(), expected)
        self.assertEqual(len(site_navigation.items), 3)
        self.assertEqual(len(site_navigation.pages), 5)

    def test_nav_page_subclass(self):
        if False:
            for i in range(10):
                print('nop')

        class PageSubclass(Page):
            pass
        nav_cfg = [{'Home': 'index.md'}, {'About': 'about.md'}]
        expected = dedent("\n            PageSubclass(title=[blank], url='/')\n            PageSubclass(title=[blank], url='/about/')\n            ")
        cfg = load_config(nav=nav_cfg, site_url='http://example.com/')
        fs = [File(list(item.values())[0], cfg.docs_dir, cfg.site_dir, cfg.use_directory_urls) for item in nav_cfg]
        files = Files(fs)
        for file in files:
            PageSubclass(None, file, cfg)
        site_navigation = get_navigation(files, cfg)
        self.assertEqual(str(site_navigation).strip(), expected)
        self.assertEqual(len(site_navigation.items), 2)
        self.assertEqual(len(site_navigation.pages), 2)
        self.assertEqual(repr(site_navigation.homepage), "PageSubclass(title=[blank], url='/')")

    def test_active(self):
        if False:
            i = 10
            return i + 15
        nav_cfg = [{'Home': 'index.md'}, {'API Guide': [{'Running': 'api-guide/running.md'}, {'Testing': 'api-guide/testing.md'}, {'Debugging': 'api-guide/debugging.md'}, {'Advanced': [{'Part 1': 'api-guide/advanced/part-1.md'}]}]}, {'About': [{'Release notes': 'about/release-notes.md'}, {'License': 'about/license.md'}]}]
        cfg = load_config(nav=nav_cfg, site_url='http://example.com/')
        fs = ['index.md', 'api-guide/running.md', 'api-guide/testing.md', 'api-guide/debugging.md', 'api-guide/advanced/part-1.md', 'about/release-notes.md', 'about/license.md']
        files = Files([File(s, cfg.docs_dir, cfg.site_dir, cfg.use_directory_urls) for s in fs])
        site_navigation = get_navigation(files, cfg)
        self.assertTrue(all((page.active is False for page in site_navigation.pages)))
        self.assertTrue(all((item.active is False for item in site_navigation.items)))
        site_navigation.items[1].children[3].children[0].active = True
        self.assertTrue(site_navigation.items[1].children[3].children[0].active)
        self.assertTrue(site_navigation.items[1].children[3].active)
        self.assertTrue(site_navigation.items[1].active)
        self.assertFalse(site_navigation.items[0].active)
        self.assertFalse(site_navigation.items[1].children[0].active)
        self.assertFalse(site_navigation.items[1].children[1].active)
        self.assertFalse(site_navigation.items[1].children[2].active)
        self.assertFalse(site_navigation.items[2].active)
        self.assertFalse(site_navigation.items[2].children[0].active)
        self.assertFalse(site_navigation.items[2].children[1].active)
        site_navigation.items[1].children[3].children[0].active = False
        self.assertFalse(site_navigation.items[1].children[3].children[0].active)
        self.assertFalse(site_navigation.items[1].children[3].active)
        self.assertFalse(site_navigation.items[1].active)

    def test_get_by_type_nested_sections(self):
        if False:
            print('Hello World!')
        nav_cfg = [{'Section 1': [{'Section 2': [{'Page': 'page.md'}]}]}]
        cfg = load_config(nav=nav_cfg, site_url='http://example.com/')
        fs = [File('page.md', cfg.docs_dir, cfg.site_dir, cfg.use_directory_urls)]
        files = Files(fs)
        site_navigation = get_navigation(files, cfg)
        self.assertEqual(len(_get_by_type(site_navigation, Section)), 2)