from djangocms_text_ckeditor.models import Text
from cms.api import add_plugin, create_page
from cms.models import Page
from cms.models.placeholdermodel import Placeholder
from cms.models.pluginmodel import CMSPlugin
from cms.tests.test_plugins import PluginsTestBaseCase
from cms.utils.copy_plugins import copy_plugins_to
from cms.utils.plugins import reorder_plugins

class NestedPluginsTestCase(PluginsTestBaseCase):

    def reorder_positions(self, plugin=None, parent=None):
        if False:
            return 10
        if parent:
            parent_id = parent.pk
            plugin = parent
        else:
            parent_id = plugin.parent_id
        x = 0
        for p in CMSPlugin.objects.filter(parent_id=parent_id, language=plugin.language, placeholder_id=plugin.placeholder_id):
            p.position = x
            p.save()
            x += 1

    def copy_placeholders_and_check_results(self, placeholders):
        if False:
            return 10
        '\n        This function is not itself a test; rather, it can be used by any test\n        that has created placeholders. It will check that whatever the plugin\n        structure in the placeholder, it will be copied accurately when they are\n        copied.\n\n        placeholders is a list of placeholders\n        '
        for original_placeholder in placeholders:
            original_plugins = original_placeholder.get_plugins()
            copied_placeholder = Placeholder.objects.create(slot=original_placeholder.slot)
            copy_plugins_to(original_placeholder.get_plugins(), copied_placeholder)
            copied_plugins = copied_placeholder.get_plugins()
            self.assertEqual(original_plugins.count(), copied_plugins.count())
            for (original, copy) in zip(original_plugins, copied_plugins):
                self.assertEqual(Text.objects.get(id=original.id).body, Text.objects.get(id=copy.id).body)
            original_plugins_list = []
            copied_plugins_list = []

            def plugin_list_from_tree(roots, plugin_list):
                if False:
                    return 10
                for plugin in roots:
                    plugin_list.append(plugin)
                    plugin_list_from_tree(plugin.get_children(), plugin_list)
            plugin_list_from_tree(original_plugins.filter(depth=1), original_plugins_list)
            plugin_list_from_tree(copied_plugins.filter(depth=1), copied_plugins_list)
            self.assertEqual(len(original_plugins_list), original_plugins.count())
            self.assertEqual(len(copied_plugins_list), copied_plugins.count())
            for (original, copy) in zip(original_plugins_list, copied_plugins_list):
                original_text_plugin = Text.objects.get(id=original.id)
                copied_text_plugin = Text.objects.get(id=copy.id)
                self.assertNotEqual(original.id, copy.id)
                self.assertEqual(original_text_plugin.body, copied_text_plugin.body)
                self.assertEqual(original_text_plugin.depth, copied_text_plugin.depth)
                self.assertEqual(original_text_plugin.position, copied_text_plugin.position)
                self.assertEqual(original_text_plugin.numchild, copied_text_plugin.numchild)
                self.assertEqual(original_text_plugin.get_descendant_count(), copied_text_plugin.get_descendant_count())
                self.assertEqual(original_text_plugin.get_ancestors().count(), copied_text_plugin.get_ancestors().count())
        return copied_placeholder

    def test_plugin_fix_tree(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Tests CMSPlugin.fix_tree by creating a plugin structure, setting the\n        position value to Null for all the plugins and then rebuild the tree.\n\n        The structure below isn't arbitrary, but has been designed to test\n        various conditions, including:\n\n        * nodes four levels deep\n        * siblings with and without children\n\n             1\n                 2\n                     4\n                          10\n                     8\n                 3\n                     9\n             5\n                 6\n                 7\n        "
        placeholder = Placeholder(slot='some_slot')
        placeholder.save()
        plugin_1 = add_plugin(placeholder, 'TextPlugin', 'en', body='01')
        plugin_1 = self.reload(plugin_1)
        plugin_2 = add_plugin(placeholder, 'TextPlugin', 'en', body='02', target=plugin_1)
        plugin_1 = self.reload(plugin_1)
        plugin_3 = add_plugin(placeholder, 'TextPlugin', 'en', body='03', target=plugin_1)
        plugin_2 = self.reload(plugin_2)
        plugin_4 = add_plugin(placeholder, 'TextPlugin', 'en', body='04', target=plugin_2)
        plugin_1 = self.reload(plugin_1)
        plugin_5 = add_plugin(placeholder, 'TextPlugin', 'en', body='05')
        left = CMSPlugin.objects.filter(parent__isnull=True).order_by('path')[0]
        plugin_5 = self.reload(plugin_5)
        plugin_5 = plugin_5.move(left, pos='right')
        self.reorder_positions(plugin_5)
        self.reorder_positions(plugin_2)
        plugin_5 = self.reload(plugin_5)
        plugin_6 = add_plugin(placeholder, 'TextPlugin', 'en', body='06', target=plugin_5)
        plugin_5 = self.reload(plugin_5)
        plugin_7 = add_plugin(placeholder, 'TextPlugin', 'en', body='07', target=plugin_5)
        plugin_2 = self.reload(plugin_2)
        plugin_8 = add_plugin(placeholder, 'TextPlugin', 'en', body='08', target=plugin_2)
        plugin_3 = self.reload(plugin_3)
        plugin_9 = add_plugin(placeholder, 'TextPlugin', 'en', body='09', target=plugin_3)
        plugin_4 = self.reload(plugin_4)
        plugin_10 = add_plugin(placeholder, 'TextPlugin', 'en', body='10', target=plugin_4)
        plugins = CMSPlugin.objects.filter(placeholder=placeholder)
        original_plugin_positions = dict(plugins.order_by('position').values_list('pk', 'position'))
        original_plugin_ids = list(plugins.order_by('position', 'path').values_list('pk', flat=True))
        CMSPlugin.objects.update(position=1)
        CMSPlugin.fix_tree()
        new_plugin_positions = dict(plugins.order_by('position').values_list('pk', 'position'))
        new_plugin_ids = list(plugins.order_by('position', 'path').values_list('pk', flat=True))
        self.assertDictEqual(original_plugin_positions, new_plugin_positions)
        self.assertSequenceEqual(original_plugin_ids, new_plugin_ids)
        reorder_plugins(placeholder, None, 'en', [plugin_5.pk, plugin_1.pk])
        reordered_plugins = list(placeholder.get_plugins().order_by('position', 'path'))
        CMSPlugin.fix_tree()
        new_plugins = list(placeholder.get_plugins().order_by('position', 'path'))
        self.assertSequenceEqual(reordered_plugins, new_plugins, 'Plugin order not preserved during fix_tree().')

    def test_plugin_deep_nesting_and_copying(self):
        if False:
            while True:
                i = 10
        "\n        Create a deeply-nested plugin structure, tests its properties, and tests\n        that it is copied accurately when the placeholder containing them is\n        copied.\n\n        The structure below isn't arbitrary, but has been designed to test\n        various conditions, including:\n\n        * nodes four levels deep\n        * multiple successive level increases\n        * multiple successive level decreases\n        * successive nodes on the same level followed by level changes\n        * multiple level decreases between successive nodes\n        * siblings with and without children\n        * nodes and branches added to the tree out of sequence\n\n        First we create the structure:\n\n             11\n             1\n                 2\n                     12\n                     4\n                          10\n                     8\n                 3\n                     9\n             5\n                 6\n                 7\n                 13\n             14\n\n        and then we move it all around.\n        "
        placeholder = Placeholder(slot='some_slot')
        placeholder.save()
        plugin_1 = add_plugin(placeholder, 'TextPlugin', 'en', body='01')
        plugin_1 = self.reload(plugin_1)
        plugin_2 = add_plugin(placeholder, 'TextPlugin', 'en', body='02', target=plugin_1)
        self.assertSequenceEqual(CMSPlugin.objects.get(id=plugin_1.pk).get_children(), [CMSPlugin.objects.get(id=plugin_2.pk)])
        plugin_1 = self.reload(plugin_1)
        plugin_3 = add_plugin(placeholder, 'TextPlugin', 'en', body='03', target=plugin_1)
        self.assertSequenceEqual(CMSPlugin.objects.get(id=plugin_1.pk).get_children(), [CMSPlugin.objects.get(id=plugin_2.pk), CMSPlugin.objects.get(id=plugin_3.pk)])
        plugin_2 = self.reload(plugin_2)
        plugin_4 = add_plugin(placeholder, 'TextPlugin', 'en', body='04', target=plugin_2)
        self.assertSequenceEqual(CMSPlugin.objects.get(id=plugin_2.pk).get_children(), [CMSPlugin.objects.get(id=plugin_4.pk)])
        self.assertSequenceEqual(CMSPlugin.objects.get(id=plugin_1.pk).get_descendants(), [CMSPlugin.objects.get(id=plugin_2.pk), CMSPlugin.objects.get(id=plugin_4.pk), CMSPlugin.objects.get(id=plugin_3.pk)])
        plugin_1 = self.reload(plugin_1)
        plugin_5 = add_plugin(placeholder, 'TextPlugin', 'en', body='05')
        left = CMSPlugin.objects.filter(parent__isnull=True).order_by('path')[0]
        plugin_5 = self.reload(plugin_5)
        plugin_5 = plugin_5.move(left, pos='right')
        self.reorder_positions(plugin_5)
        self.reorder_positions(plugin_2)
        plugin_5 = self.reload(plugin_5)
        plugin_6 = add_plugin(placeholder, 'TextPlugin', 'en', body='06', target=plugin_5)
        self.assertSequenceEqual(CMSPlugin.objects.get(id=plugin_5.pk).get_children(), [CMSPlugin.objects.get(id=plugin_6.pk)])
        plugin_5 = self.reload(plugin_5)
        plugin_7 = add_plugin(placeholder, 'TextPlugin', 'en', body='07', target=plugin_5)
        self.assertSequenceEqual(CMSPlugin.objects.get(id=plugin_5.pk).get_children(), [CMSPlugin.objects.get(id=plugin_6.pk), CMSPlugin.objects.get(id=plugin_7.pk)])
        self.assertSequenceEqual(CMSPlugin.objects.get(id=plugin_5.pk).get_descendants(), [CMSPlugin.objects.get(id=plugin_6.pk), CMSPlugin.objects.get(id=plugin_7.pk)])
        plugin_2 = self.reload(plugin_2)
        plugin_8 = add_plugin(placeholder, 'TextPlugin', 'en', body='08', target=plugin_2)
        self.assertSequenceEqual(CMSPlugin.objects.get(id=plugin_2.pk).get_children(), [CMSPlugin.objects.get(id=plugin_4.pk), CMSPlugin.objects.get(id=plugin_8.pk)])
        plugin_3 = self.reload(plugin_3)
        plugin_9 = add_plugin(placeholder, 'TextPlugin', 'en', body='09', target=plugin_3)
        self.assertSequenceEqual(CMSPlugin.objects.get(id=plugin_3.pk).get_children(), [CMSPlugin.objects.get(id=plugin_9.pk)])
        plugin_4 = self.reload(plugin_4)
        plugin_10 = add_plugin(placeholder, 'TextPlugin', 'en', body='10', target=plugin_4)
        self.assertSequenceEqual(CMSPlugin.objects.get(id=plugin_4.pk).get_children(), [CMSPlugin.objects.get(id=plugin_10.pk)])
        original_plugins = placeholder.get_plugins()
        self.assertEqual(original_plugins.count(), 10)
        plugin_1 = self.reload(plugin_1)
        plugin_11 = add_plugin(placeholder, 'TextPlugin', 'en', body='11', target=plugin_1, position='left')
        self.assertSequenceEqual(CMSPlugin.objects.get(id=plugin_1.pk).get_children(), [CMSPlugin.objects.get(id=plugin_2.pk), CMSPlugin.objects.get(id=plugin_3.pk)])
        plugin_4 = self.reload(plugin_4)
        plugin_12 = add_plugin(placeholder, 'TextPlugin', 'en', body='12', target=plugin_4, position='left')
        self.assertSequenceEqual(CMSPlugin.objects.get(id=plugin_2.pk).get_children(), [CMSPlugin.objects.get(id=plugin_12.pk), CMSPlugin.objects.get(id=plugin_4.pk), CMSPlugin.objects.get(id=plugin_8.pk)])
        plugin_7 = self.reload(plugin_7)
        plugin_13 = add_plugin(placeholder, 'TextPlugin', 'en', body='13', target=plugin_7, position='right')
        self.assertSequenceEqual(CMSPlugin.objects.get(id=plugin_5.pk).get_children(), [CMSPlugin.objects.get(id=plugin_6.pk), CMSPlugin.objects.get(id=plugin_7.pk), CMSPlugin.objects.get(id=plugin_13.pk)])
        plugin_5 = self.reload(plugin_5)
        plugin_14 = add_plugin(placeholder, 'TextPlugin', 'en', body='14')
        self.assertSequenceEqual(CMSPlugin.objects.filter(depth=1).order_by('path'), [CMSPlugin.objects.get(id=plugin_11.pk), CMSPlugin.objects.get(id=plugin_1.pk), CMSPlugin.objects.get(id=plugin_5.pk), CMSPlugin.objects.get(id=plugin_14.pk)])
        self.copy_placeholders_and_check_results([placeholder])
        plugin_2 = self.reload(plugin_2)
        plugin_1 = self.reload(plugin_1)
        old_parent = plugin_2.parent
        plugin_2.parent_id = plugin_1.parent_id
        plugin_2.save()
        plugin_2 = plugin_2.move(target=plugin_1, pos='left')
        self.reorder_positions(parent=old_parent)
        self.reorder_positions(plugin_2)
        self.copy_placeholders_and_check_results([placeholder])
        plugin_6 = self.reload(plugin_6)
        plugin_7 = self.reload(plugin_7)
        old_parent = plugin_6.parent
        plugin_6.parent_id = plugin_7.parent_id
        plugin_6.save()
        plugin_6 = plugin_6.move(target=plugin_7, pos='right')
        self.reorder_positions(parent=old_parent)
        self.reorder_positions(plugin_6)
        self.copy_placeholders_and_check_results([placeholder])
        plugin_2 = self.reload(plugin_2)
        plugin_3 = self.reload(plugin_3)
        old_parent = plugin_3.parent
        plugin_3.parent_id = plugin_2.parent_id
        plugin_3.save()
        plugin_3 = plugin_3.move(target=plugin_2, pos='left')
        self.reorder_positions(parent=old_parent)
        self.reorder_positions(plugin_3)
        self.copy_placeholders_and_check_results([placeholder])
        plugin_2 = self.reload(plugin_2)
        plugin_3 = self.reload(plugin_3)
        old_parent = plugin_3.parent
        plugin_3.parent_id = plugin_2.pk
        plugin_3.save()
        plugin_3 = plugin_3.move(target=plugin_2, pos='first-child')
        self.reorder_positions(CMSPlugin.objects.filter(placeholder_id=plugin_3.placeholder_id, language=plugin_3.language, depth=1)[0])
        self.reorder_positions(plugin_3)
        self.copy_placeholders_and_check_results([placeholder])
        plugin_3 = self.reload(plugin_3)
        plugin_7 = self.reload(plugin_7)
        old_parent = plugin_7.parent
        plugin_7.parent_id = plugin_3.parent_id
        plugin_7.save()
        plugin_7 = plugin_7.move(target=plugin_3, pos='right')
        self.reorder_positions(parent=old_parent)
        self.reorder_positions(plugin_7)
        self.copy_placeholders_and_check_results([placeholder])

    def test_nested_plugin_on_page(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Validate a textplugin with a nested link plugin\n        mptt values are correctly showing a parent child relationship\n        of a nested plugin\n        '
        with self.settings(CMS_PERMISSION=False):
            page_one = create_page('Three Placeholder', 'col_three.html', 'en', position='last-child', published=True, in_navigation=True)
            page_one_ph_two = page_one.placeholders.get(slot='col_left')
            pre_nesting_body = '<p>the nested text plugin with a link inside</p>'
            text_plugin = add_plugin(page_one_ph_two, 'TextPlugin', 'en', body=pre_nesting_body)
            page_one_ph_two = self.reload(page_one_ph_two)
            text_plugin = self.reload(text_plugin)
            link_plugin = add_plugin(page_one_ph_two, 'LinkPlugin', 'en', target=text_plugin)
            link_plugin.name = 'django-cms Link'
            link_plugin.external_link = 'https://www.django-cms.org'
            link_plugin.parent = text_plugin
            link_plugin.save()
            link_plugin = self.reload(link_plugin)
            text_plugin = self.reload(text_plugin)
            msg = 'parent plugin right is not updated, child not inserted correctly'
            self.assertTrue(text_plugin.position == link_plugin.position, msg=msg)
            msg = 'link has no parent'
            self.assertFalse(link_plugin.parent is None, msg=msg)
            msg = 'parent plugin path is not updated, child not inserted correctly'
            self.assertTrue(text_plugin.path == link_plugin.path[:4], msg=msg)
            msg = 'child level is not bigger than parent level'
            self.assertTrue(text_plugin.depth < link_plugin.depth, msg=msg)
            in_txt = '<img id="plugin_obj_%s" title="Link" alt="Link" src="/static/cms/img/icons/plugins/link.png">'
            nesting_body = f'{text_plugin.body}<p>{in_txt % link_plugin.id}</p>'
            text_plugin.body = nesting_body
            text_plugin.save()
            text_plugin = self.reload(text_plugin)
            self.assertEqual(text_plugin.get_descendants().exclude(placeholder=text_plugin.placeholder).count(), 0)
            post_add_plugin_count = CMSPlugin.objects.count()
            self.assertEqual(post_add_plugin_count, 2)

    def test_copy_page_nested_plugin(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test to verify that page copy with a nested plugin works\n        page one - 3 placeholder\n                    col_sidebar: 1 text plugin\n                    col_left: 1 text plugin with nested link plugin\n                    col_right: no plugin\n        page two (copy target)\n        Verify copied page, placeholders, plugins and body text\n        '
        with self.settings(CMS_PERMISSION=False):
            page_one = create_page('Three Placeholder', 'col_three.html', 'en', position='last-child', published=True, in_navigation=True)
            page_one_ph_one = page_one.placeholders.get(slot='col_sidebar')
            page_one_ph_two = page_one.placeholders.get(slot='col_left')
            page_one.placeholders.get(slot='col_right')
            text_plugin_en = add_plugin(page_one_ph_one, 'TextPlugin', 'en', body='Hello World')
            self.assertEqual(text_plugin_en.id, CMSPlugin.objects.all()[0].id)
            self.assertEqual(text_plugin_en.get_children().count(), 0)
            pre_add_plugin_count = CMSPlugin.objects.count()
            self.assertEqual(pre_add_plugin_count, 1)
            pre_nesting_body = '<p>the nested text plugin with a link inside</p>'
            text_plugin_two = add_plugin(page_one_ph_two, 'TextPlugin', 'en', body=pre_nesting_body)
            text_plugin_two = self.reload(text_plugin_two)
            page_one_ph_two = self.reload(page_one_ph_two)
            text_plugin_two = self.reload(text_plugin_two)
            link_plugin = add_plugin(page_one_ph_two, 'LinkPlugin', 'en', target=text_plugin_two)
            link_plugin.name = 'django-cms Link'
            link_plugin.external_link = 'https://www.django-cms.org'
            link_plugin.parent = text_plugin_two
            link_plugin.save()
            link_plugin = self.reload(link_plugin)
            text_plugin_two = self.reload(text_plugin_two)
            in_txt = '<cms-plugin id="%s" title="Link" alt="Link"></cms-plugin>'
            nesting_body = f'{text_plugin_two.body}<p>{in_txt % link_plugin.id}</p>'
            text_plugin_two.body = nesting_body
            text_plugin_two.save()
            text_plugin_two = self.reload(text_plugin_two)
            self.assertEqual(text_plugin_two.get_children().count(), 1)
            post_add_plugin_count = CMSPlugin.objects.filter(placeholder__page__publisher_is_draft=True).count()
            self.assertEqual(post_add_plugin_count, 3)
            page_one.save()
            page_one = self.reload(page_one)
            page_one_ph_one = page_one.placeholders.get(slot='col_sidebar')
            page_one_ph_two = page_one.placeholders.get(slot='col_left')
            page_one_ph_three = page_one.placeholders.get(slot='col_right')
            org_placeholder_one_plugins = page_one_ph_one.get_plugins()
            self.assertEqual(len(org_placeholder_one_plugins), 1)
            org_placeholder_two_plugins = page_one_ph_two.get_plugins()
            self.assertEqual(len(org_placeholder_two_plugins), 2)
            org_placeholder_three_plugins = page_one_ph_three.get_plugins()
            self.assertEqual(len(org_placeholder_three_plugins), 0)
            self.assertEqual(page_one.placeholders.count(), 3)
            placeholder_count = Placeholder.objects.filter(page__publisher_is_draft=True).count()
            self.assertEqual(placeholder_count, 3)
            self.assertEqual(CMSPlugin.objects.filter(placeholder__page__publisher_is_draft=True).count(), 3)
            page_copy_target = create_page('Three Placeholder - page copy target', 'col_three.html', 'en', position='last-child', published=True, in_navigation=True)
            all_page_count = Page.objects.drafts().count()
            pre_copy_placeholder_count = Placeholder.objects.filter(page__publisher_is_draft=True).count()
            self.assertEqual(pre_copy_placeholder_count, 6)
            superuser = self.get_superuser()
            with self.login_user_context(superuser):
                page_two = self.copy_page(page_one, page_copy_target)
            after_copy_page_plugin_count = CMSPlugin.objects.filter(placeholder__page__publisher_is_draft=True).count()
            self.assertEqual(after_copy_page_plugin_count, 6)
            after_copy_page_count = Page.objects.drafts().count()
            after_copy_placeholder_count = Placeholder.objects.filter(page__publisher_is_draft=True).count()
            self.assertGreater(after_copy_page_count, all_page_count, 'no new page after copy')
            self.assertGreater(after_copy_page_plugin_count, post_add_plugin_count, 'plugin count is not grown')
            self.assertGreater(after_copy_placeholder_count, pre_copy_placeholder_count, 'placeholder count is not grown')
            self.assertEqual(after_copy_page_count, 3, 'no new page after copy')
            page_one = self.reload(page_one)
            page_one_ph_one = page_one.placeholders.get(slot='col_sidebar')
            page_one_ph_two = page_one.placeholders.get(slot='col_left')
            page_one_ph_three = page_one.placeholders.get(slot='col_right')
            found_page = page_one_ph_one.page if page_one_ph_one else None
            self.assertEqual(found_page, page_one)
            found_page = page_one_ph_two.page if page_one_ph_two else None
            self.assertEqual(found_page, page_one)
            found_page = page_one_ph_three.page if page_one_ph_three else None
            self.assertEqual(found_page, page_one)
            page_two = self.reload(page_two)
            page_two_ph_one = page_two.placeholders.get(slot='col_sidebar')
            page_two_ph_two = page_two.placeholders.get(slot='col_left')
            page_two_ph_three = page_two.placeholders.get(slot='col_right')
            found_page = page_two_ph_one.page if page_two_ph_one else None
            self.assertEqual(found_page, page_two)
            found_page = page_two_ph_two.page if page_two_ph_two else None
            self.assertEqual(found_page, page_two)
            found_page = page_two_ph_three.page if page_two_ph_three else None
            self.assertEqual(found_page, page_two)
            msg = 'placehoder ids copy:{} org:{} copied page {} are identical - tree broken'.format(page_two_ph_one.pk, page_one_ph_one.pk, page_two.pk)
            self.assertNotEqual(page_two_ph_one.pk, page_one_ph_one.pk, msg)
            msg = 'placehoder ids copy:{} org:{} copied page {} are identical - tree broken'.format(page_two_ph_two.pk, page_one_ph_two.pk, page_two.pk)
            self.assertNotEqual(page_two_ph_two.pk, page_one_ph_two.pk, msg)
            msg = 'placehoder ids copy:{} org:{} copied page {} are identical - tree broken'.format(page_two_ph_three.pk, page_one_ph_three.pk, page_two.pk)
            self.assertNotEqual(page_two_ph_three.pk, page_one_ph_three.pk, msg)
            org_placeholder_one_plugins = page_one_ph_one.get_plugins()
            self.assertEqual(len(org_placeholder_one_plugins), 1)
            org_placeholder_two_plugins = page_one_ph_two.get_plugins()
            self.assertEqual(len(org_placeholder_two_plugins), 2)
            org_placeholder_three_plugins = page_one_ph_three.get_plugins()
            self.assertEqual(len(org_placeholder_three_plugins), 0)
            copied_placeholder_one_plugins = page_two_ph_one.get_plugins()
            self.assertEqual(len(copied_placeholder_one_plugins), 1)
            copied_placeholder_two_plugins = page_two_ph_two.get_plugins()
            self.assertEqual(len(copied_placeholder_two_plugins), 2)
            copied_placeholder_three_plugins = page_two_ph_three.get_plugins()
            self.assertEqual(len(copied_placeholder_three_plugins), 0)
            count_plugins_copied = len(copied_placeholder_one_plugins)
            count_plugins_org = len(org_placeholder_one_plugins)
            msg = f'plugin count {count_plugins_copied} {count_plugins_org} for placeholder one not equal'
            self.assertEqual(count_plugins_copied, count_plugins_org, msg)
            count_plugins_copied = len(copied_placeholder_two_plugins)
            count_plugins_org = len(org_placeholder_two_plugins)
            msg = f'plugin count {count_plugins_copied} {count_plugins_org} for placeholder two not equal'
            self.assertEqual(count_plugins_copied, count_plugins_org, msg)
            count_plugins_copied = len(copied_placeholder_three_plugins)
            count_plugins_org = len(org_placeholder_three_plugins)
            msg = f'plugin count {count_plugins_copied} {count_plugins_org} for placeholder three not equal'
            self.assertEqual(count_plugins_copied, count_plugins_org, msg)
            org_nested_text_plugin = None
            for x in org_placeholder_two_plugins:
                if x.plugin_type == 'TextPlugin':
                    instance = x.get_plugin_instance()[0]
                    if instance.body.startswith(pre_nesting_body):
                        org_nested_text_plugin = instance
                        break
            copied_nested_text_plugin = None
            for x in copied_placeholder_two_plugins:
                if x.plugin_type == 'TextPlugin':
                    instance = x.get_plugin_instance()[0]
                    if instance.body.startswith(pre_nesting_body):
                        copied_nested_text_plugin = instance
                        break
            msg = 'original nested text plugin not found'
            self.assertNotEqual(org_nested_text_plugin, None, msg=msg)
            msg = 'copied nested text plugin not found'
            self.assertNotEqual(copied_nested_text_plugin, None, msg=msg)
            org_link_child_plugin = org_nested_text_plugin.get_children()[0]
            copied_link_child_plugin = copied_nested_text_plugin.get_children()[0]
            msg = 'org plugin and copied plugin are the same'
            self.assertTrue(org_link_child_plugin.id != copied_link_child_plugin.id, msg)
            needle = '%s'
            msg = 'child plugin id differs to parent in body'
            self.assertTrue(org_nested_text_plugin.body.find(needle % org_link_child_plugin.id) != -1, msg)
            msg = 'copy: child plugin id differs to parent in body'
            self.assertTrue(copied_nested_text_plugin.body.find(needle % copied_link_child_plugin.id) != -1, msg)
            msg = 'child link plugin id differs to parent body'
            self.assertTrue(org_nested_text_plugin.body.find(needle % copied_link_child_plugin.id) == -1, msg)
            msg = 'copy: child link plugin id differs to parent body'
            self.assertTrue(copied_nested_text_plugin.body.find(needle % org_link_child_plugin.id) == -1, msg)
            org_placeholder = org_link_child_plugin.placeholder
            copied_placeholder = copied_link_child_plugin.placeholder
            msg = 'placeholder of the original plugin and copied plugin are the same'
            ok = org_placeholder.id != copied_placeholder.id
            self.assertTrue(ok, msg)

    def test_copy_page_nested_plugin_moved_parent_plugin(self):
        if False:
            i = 10
            return i + 15
        '\n        Test to verify that page copy with a nested plugin works\n        when a plugin with child got moved to another placeholder\n        page one - 3 placeholder\n                    col_sidebar:\n                        1 text plugin\n                    col_left: 1 text plugin with nested link plugin\n                    col_right: no plugin\n        page two (copy target)\n        step2: move the col_left text plugin to col_right\n                    col_sidebar:\n                        1 text plugin\n                    col_left: no plugin\n                    col_right: 1 text plugin with nested link plugin\n        verify the copied page structure\n        '
        with self.settings(CMS_PERMISSION=False):
            page_one = create_page('Three Placeholder', 'col_three.html', 'en', position='last-child', published=True, in_navigation=True)
            page_one_ph_one = page_one.placeholders.get(slot='col_sidebar')
            page_one_ph_two = page_one.placeholders.get(slot='col_left')
            page_one.placeholders.get(slot='col_right')
            text_plugin_en = add_plugin(page_one_ph_one, 'TextPlugin', 'en', body='Hello World')
            self.assertEqual(text_plugin_en.id, CMSPlugin.objects.all()[0].id)
            self.assertEqual(text_plugin_en.get_children().count(), 0)
            pre_add_plugin_count = CMSPlugin.objects.count()
            self.assertEqual(pre_add_plugin_count, 1)
            pre_nesting_body = '<p>the nested text plugin with a link inside</p>'
            text_plugin_two = add_plugin(page_one_ph_two, 'TextPlugin', 'en', body=pre_nesting_body)
            text_plugin_two = self.reload(text_plugin_two)
            page_one_ph_two = self.reload(page_one_ph_two)
            text_plugin_two = self.reload(text_plugin_two)
            link_plugin = add_plugin(page_one_ph_two, 'LinkPlugin', 'en', target=text_plugin_two)
            link_plugin.name = 'django-cms Link'
            link_plugin.external_link = 'https://www.django-cms.org'
            link_plugin.parent = text_plugin_two
            link_plugin.save()
            link_plugin = self.reload(link_plugin)
            text_plugin_two = self.reload(text_plugin_two)
            in_txt = '<cms-plugin id="%s" title="Link" alt="Link"></cms-plugin>'
            nesting_body = f'{text_plugin_two.body}<p>{in_txt % link_plugin.id}</p>'
            text_plugin_two.body = nesting_body
            text_plugin_two.save()
            text_plugin_two = self.reload(text_plugin_two)
            self.assertEqual(text_plugin_two.get_children().count(), 1)
            post_add_plugin_count = CMSPlugin.objects.count()
            self.assertEqual(post_add_plugin_count, 3)
            page_one.save()
            page_one = self.reload(page_one)
            page_one_ph_one = page_one.placeholders.get(slot='col_sidebar')
            page_one_ph_two = page_one.placeholders.get(slot='col_left')
            page_one_ph_three = page_one.placeholders.get(slot='col_right')
            org_placeholder_one_plugins = page_one_ph_one.get_plugins()
            self.assertEqual(len(org_placeholder_one_plugins), 1)
            org_placeholder_two_plugins = page_one_ph_two.get_plugins()
            self.assertEqual(len(org_placeholder_two_plugins), 2)
            org_placeholder_three_plugins = page_one_ph_three.get_plugins()
            self.assertEqual(len(org_placeholder_three_plugins), 0)
            self.assertEqual(page_one.placeholders.count(), 3)
            placeholder_count = Placeholder.objects.filter(page__publisher_is_draft=True).count()
            self.assertEqual(placeholder_count, 3)
            self.assertEqual(CMSPlugin.objects.count(), 3)
            page_copy_target = create_page('Three Placeholder - page copy target', 'col_three.html', 'en', position='last-child', published=True, in_navigation=True)
            all_page_count = Page.objects.drafts().count()
            pre_copy_placeholder_count = Placeholder.objects.filter(page__publisher_is_draft=True).count()
            self.assertEqual(pre_copy_placeholder_count, 6)
            superuser = self.get_superuser()
            with self.login_user_context(superuser):
                post_data = {'placeholder_id': page_one_ph_three.id, 'plugin_id': text_plugin_two.id, 'target_language': 'en', 'plugin_parent': ''}
                edit_url = self.get_move_plugin_uri(text_plugin_two)
                response = self.client.post(edit_url, post_data)
                self.assertEqual(response.status_code, 200)
                page_one = self.reload(page_one)
                self.reload(text_plugin_two)
                page_one_ph_one = page_one.placeholders.get(slot='col_sidebar')
                page_one_ph_two = page_one.placeholders.get(slot='col_left')
                page_one_ph_three = page_one.placeholders.get(slot='col_right')
                org_placeholder_one_plugins = page_one_ph_one.get_plugins()
                self.assertEqual(len(org_placeholder_one_plugins), 1)
                org_placeholder_two_plugins = page_one_ph_two.get_plugins()
                self.assertEqual(len(org_placeholder_two_plugins), 0)
                org_placeholder_three_plugins = page_one_ph_three.get_plugins()
                self.assertEqual(len(org_placeholder_three_plugins), 2)
                page_two = self.copy_page(page_one, page_copy_target)
            after_copy_page_plugin_count = CMSPlugin.objects.count()
            self.assertEqual(after_copy_page_plugin_count, 6)
            after_copy_page_count = Page.objects.drafts().count()
            after_copy_placeholder_count = Placeholder.objects.filter(page__publisher_is_draft=True).count()
            self.assertGreater(after_copy_page_count, all_page_count, 'no new page after copy')
            self.assertGreater(after_copy_page_plugin_count, post_add_plugin_count, 'plugin count is not grown')
            self.assertGreater(after_copy_placeholder_count, pre_copy_placeholder_count, 'placeholder count is not grown')
            self.assertEqual(after_copy_page_count, 3, 'no new page after copy')
            page_one = self.reload(page_one)
            page_one_ph_one = page_one.placeholders.get(slot='col_sidebar')
            page_one_ph_two = page_one.placeholders.get(slot='col_left')
            page_one_ph_three = page_one.placeholders.get(slot='col_right')
            found_page = page_one_ph_one.page if page_one_ph_one else None
            self.assertEqual(found_page, page_one)
            found_page = page_one_ph_two.page if page_one_ph_two else None
            self.assertEqual(found_page, page_one)
            found_page = page_one_ph_three.page if page_one_ph_three else None
            self.assertEqual(found_page, page_one)
            page_two = self.reload(page_two)
            page_two_ph_one = page_two.placeholders.get(slot='col_sidebar')
            page_two_ph_two = page_two.placeholders.get(slot='col_left')
            page_two_ph_three = page_two.placeholders.get(slot='col_right')
            found_page = page_two_ph_one.page if page_two_ph_one else None
            self.assertEqual(found_page, page_two)
            found_page = page_two_ph_two.page if page_two_ph_two else None
            self.assertEqual(found_page, page_two)
            found_page = page_two_ph_three.page if page_two_ph_three else None
            self.assertEqual(found_page, page_two)
            msg = 'placehoder ids copy:{} org:{} copied page {} are identical - tree broken'.format(page_two_ph_one.pk, page_one_ph_one.pk, page_two.pk)
            self.assertNotEqual(page_two_ph_one.pk, page_one_ph_one.pk, msg)
            msg = 'placehoder ids copy:{} org:{} copied page {} are identical - tree broken'.format(page_two_ph_two.pk, page_one_ph_two.pk, page_two.pk)
            self.assertNotEqual(page_two_ph_two.pk, page_one_ph_two.pk, msg)
            msg = 'placehoder ids copy:{} org:{} copied page {} are identical - tree broken'.format(page_two_ph_three.pk, page_one_ph_three.pk, page_two.pk)
            self.assertNotEqual(page_two_ph_three.pk, page_one_ph_three.pk, msg)
            org_placeholder_one_plugins = page_one_ph_one.get_plugins()
            self.assertEqual(len(org_placeholder_one_plugins), 1)
            org_placeholder_two_plugins = page_one_ph_two.get_plugins()
            self.assertEqual(len(org_placeholder_two_plugins), 0)
            org_placeholder_three_plugins = page_one_ph_three.get_plugins()
            self.assertEqual(len(org_placeholder_three_plugins), 2)
            copied_placeholder_one_plugins = page_two_ph_one.get_plugins()
            self.assertEqual(len(copied_placeholder_one_plugins), 1)
            copied_placeholder_two_plugins = page_two_ph_two.get_plugins()
            self.assertEqual(len(copied_placeholder_two_plugins), 0)
            copied_placeholder_three_plugins = page_two_ph_three.get_plugins()
            self.assertEqual(len(copied_placeholder_three_plugins), 2)
            count_plugins_copied = len(copied_placeholder_one_plugins)
            count_plugins_org = len(org_placeholder_one_plugins)
            msg = f'plugin count {count_plugins_copied} {count_plugins_org} for placeholder one not equal'
            self.assertEqual(count_plugins_copied, count_plugins_org, msg)
            count_plugins_copied = len(copied_placeholder_two_plugins)
            count_plugins_org = len(org_placeholder_two_plugins)
            msg = f'plugin count {count_plugins_copied} {count_plugins_org} for placeholder two not equal'
            self.assertEqual(count_plugins_copied, count_plugins_org, msg)
            count_plugins_copied = len(copied_placeholder_three_plugins)
            count_plugins_org = len(org_placeholder_three_plugins)
            msg = f'plugin count {count_plugins_copied} {count_plugins_org} for placeholder three not equal'
            self.assertEqual(count_plugins_copied, count_plugins_org, msg)
            org_nested_text_plugin = None
            for x in org_placeholder_three_plugins:
                if x.plugin_type == 'TextPlugin':
                    instance = x.get_plugin_instance()[0]
                    if instance.body.startswith(pre_nesting_body):
                        org_nested_text_plugin = instance
                        break
            copied_nested_text_plugin = None
            for x in copied_placeholder_three_plugins:
                if x.plugin_type == 'TextPlugin':
                    instance = x.get_plugin_instance()[0]
                    if instance.body.startswith(pre_nesting_body):
                        copied_nested_text_plugin = instance
                        break
            msg = 'original nested text plugin not found'
            self.assertNotEqual(org_nested_text_plugin, None, msg=msg)
            msg = 'copied nested text plugin not found'
            self.assertNotEqual(copied_nested_text_plugin, None, msg=msg)
            org_link_child_plugin = org_nested_text_plugin.get_children()[0]
            copied_link_child_plugin = copied_nested_text_plugin.get_children()[0]
            msg = 'org plugin and copied plugin are the same'
            self.assertNotEqual(org_link_child_plugin.id, copied_link_child_plugin.id, msg)
            needle = '%s'
            msg = 'child plugin id differs to parent in body'
            self.assertTrue(org_nested_text_plugin.body.find(needle % org_link_child_plugin.id) != -1, msg)
            msg = 'copy: child plugin id differs to parent in body plugin_obj_id'
            self.assertTrue(copied_nested_text_plugin.body.find(needle % copied_link_child_plugin.id) != -1, msg)
            msg = 'child link plugin id differs to parent body'
            self.assertTrue(org_nested_text_plugin.body.find(needle % copied_link_child_plugin.id) == -1, msg)
            msg = 'copy: child link plugin id differs to parent body'
            self.assertTrue(copied_nested_text_plugin.body.find(needle % org_link_child_plugin.id) == -1, msg)
            org_placeholder = org_link_child_plugin.placeholder
            copied_placeholder = copied_link_child_plugin.placeholder
            msg = 'placeholder of the original plugin and copied plugin are the same'
            self.assertNotEqual(org_placeholder.id, copied_placeholder.id, msg)

    def test_add_child_plugin(self):
        if False:
            for i in range(10):
                print('nop')
        page_one = create_page('Three Placeholder', 'col_three.html', 'en', position='last-child', published=True, in_navigation=True)
        page_one_ph_one = page_one.placeholders.get(slot='col_sidebar')
        text_plugin_en = add_plugin(page_one_ph_one, 'TextPlugin', 'en', body='Hello World')
        superuser = self.get_superuser()
        with self.login_user_context(superuser):
            post_data = {'name': 'test', 'external_link': 'http://www.example.org/'}
            add_url = self.get_add_plugin_uri(page_one_ph_one, 'LinkPlugin', parent=text_plugin_en)
            response = self.client.post(add_url, post_data)
            self.assertEqual(response.status_code, 200)
            self.assertTemplateUsed(response, 'admin/cms/page/plugin/confirm_form.html')
        link_plugin = CMSPlugin.objects.get(parent_id=text_plugin_en.pk)
        self.assertEqual(link_plugin.parent_id, text_plugin_en.pk)
        self.assertEqual(link_plugin.path, '00010001')