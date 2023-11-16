import base64
import operator
import re
from odoo import api, fields, models, tools, _
from odoo.exceptions import ValidationError
from odoo.http import request
from odoo.modules import get_module_resource
from odoo.tools.safe_eval import safe_eval
MENU_ITEM_SEPARATOR = '/'
NUMBER_PARENS = re.compile('\\(([0-9]+)\\)')

class IrUiMenu(models.Model):
    _name = 'ir.ui.menu'
    _order = 'sequence,id'
    _parent_store = True

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super(IrUiMenu, self).__init__(*args, **kwargs)
        self.pool['ir.model.access'].register_cache_clearing_method(self._name, 'clear_caches')
    name = fields.Char(string='Menu', required=True, translate=True)
    active = fields.Boolean(default=True)
    sequence = fields.Integer(default=10)
    child_id = fields.One2many('ir.ui.menu', 'parent_id', string='Child IDs')
    parent_id = fields.Many2one('ir.ui.menu', string='Parent Menu', index=True, ondelete='restrict')
    parent_left = fields.Integer(index=True)
    parent_right = fields.Integer(index=True)
    groups_id = fields.Many2many('res.groups', 'ir_ui_menu_group_rel', 'menu_id', 'gid', string='Groups', help="If you have groups, the visibility of this menu will be based on these groups. If this field is empty, Odoo will compute visibility based on the related object's read access.")
    complete_name = fields.Char(compute='_compute_complete_name', string='Full Path')
    web_icon = fields.Char(string='Web Icon File')
    action = fields.Reference(selection=[('ir.actions.report.xml', 'ir.actions.report.xml'), ('ir.actions.act_window', 'ir.actions.act_window'), ('ir.actions.act_url', 'ir.actions.act_url'), ('ir.actions.server', 'ir.actions.server'), ('ir.actions.client', 'ir.actions.client')])
    web_icon_data = fields.Binary(string='Web Icon Image', compute='_compute_web_icon', store=True, attachment=True)

    @api.depends('name', 'parent_id.complete_name')
    def _compute_complete_name(self):
        if False:
            while True:
                i = 10
        for menu in self:
            menu.complete_name = menu._get_full_name()

    def _get_full_name(self, level=6):
        if False:
            i = 10
            return i + 15
        ' Return the full name of ``self`` (up to a certain level). '
        if level <= 0:
            return '...'
        if self.parent_id:
            return self.parent_id._get_full_name(level - 1) + MENU_ITEM_SEPARATOR + (self.name or '')
        else:
            return self.name

    @api.depends('web_icon')
    def _compute_web_icon(self):
        if False:
            print('Hello World!')
        " Returns the image associated to `web_icon`.\n            `web_icon` can either be:\n              - an image icon [module, path]\n              - a built icon [icon_class, icon_color, background_color]\n            and it only has to call `read_image` if it's an image.\n        "
        for menu in self:
            if menu.web_icon and len(menu.web_icon.split(',')) == 2:
                menu.web_icon_data = self.read_image(menu.web_icon)

    def read_image(self, path):
        if False:
            print('Hello World!')
        if not path:
            return False
        path_info = path.split(',')
        icon_path = get_module_resource(path_info[0], path_info[1])
        icon_image = False
        if icon_path:
            with tools.file_open(icon_path, 'rb') as icon_file:
                icon_image = base64.encodestring(icon_file.read())
        return icon_image

    @api.constrains('parent_id')
    def _check_parent_id(self):
        if False:
            while True:
                i = 10
        if not self._check_recursion():
            raise ValidationError(_('Error! You cannot create recursive menus.'))

    @api.model
    @tools.ormcache('frozenset(self.env.user.groups_id.ids)', 'debug')
    def _visible_menu_ids(self, debug=False):
        if False:
            print('Hello World!')
        ' Return the ids of the menu items visible to the user. '
        context = {'ir.ui.menu.full_list': True}
        menus = self.with_context(context).search([])
        groups = self.env.user.groups_id
        if not debug:
            groups = groups - self.env.ref('base.group_no_one')
        menus = menus.filtered(lambda menu: not menu.groups_id or menu.groups_id & groups)
        action_menus = menus.filtered(lambda m: m.action and m.action.exists())
        folder_menus = menus - action_menus
        visible = self.browse()
        access = self.env['ir.model.access']
        model_fname = {'ir.actions.act_window': 'res_model', 'ir.actions.report.xml': 'model', 'ir.actions.server': 'model_id'}
        for menu in action_menus:
            fname = model_fname.get(menu.action._name)
            if not fname or not menu.action[fname] or access.check(menu.action[fname], 'read', False):
                visible += menu
                menu = menu.parent_id
                while menu and menu in folder_menus and (menu not in visible):
                    visible += menu
                    menu = menu.parent_id
        return set(visible.ids)

    @api.multi
    @api.returns('self')
    def _filter_visible_menus(self):
        if False:
            while True:
                i = 10
        ' Filter `self` to only keep the menu items that should be visible in\n            the menu hierarchy of the current user.\n            Uses a cache for speeding up the computation.\n        '
        visible_ids = self._visible_menu_ids(request.debug if request else False)
        return self.filtered(lambda menu: menu.id in visible_ids)

    @api.model
    def search(self, args, offset=0, limit=None, order=None, count=False):
        if False:
            return 10
        menus = super(IrUiMenu, self).search(args, offset=0, limit=None, order=order, count=False)
        if menus:
            if not self._context.get('ir.ui.menu.full_list'):
                menus = menus._filter_visible_menus()
            if offset:
                menus = menus[long(offset):]
            if limit:
                menus = menus[:long(limit)]
        return len(menus) if count else menus

    @api.multi
    def name_get(self):
        if False:
            i = 10
            return i + 15
        return [(menu.id, menu._get_full_name()) for menu in self]

    @api.model
    def create(self, values):
        if False:
            for i in range(10):
                print('nop')
        self.clear_caches()
        return super(IrUiMenu, self).create(values)

    @api.multi
    def write(self, values):
        if False:
            for i in range(10):
                print('nop')
        self.clear_caches()
        return super(IrUiMenu, self).write(values)

    @api.multi
    def unlink(self):
        if False:
            return 10
        extra = {'ir.ui.menu.full_list': True}
        direct_children = self.with_context(**extra).search([('parent_id', 'in', self.ids)])
        direct_children.write({'parent_id': False})
        self.clear_caches()
        return super(IrUiMenu, self).unlink()

    @api.multi
    def copy(self, default=None):
        if False:
            i = 10
            return i + 15
        record = super(IrUiMenu, self).copy(default=default)
        match = NUMBER_PARENS.search(record.name)
        if match:
            next_num = int(match.group(1)) + 1
            record.name = NUMBER_PARENS.sub('(%d)' % next_num, record.name)
        else:
            record.name = record.name + '(1)'
        return record

    @api.multi
    def get_needaction_data(self):
        if False:
            while True:
                i = 10
        ' Return for each menu entry in ``self``:\n            - whether it uses the needaction mechanism (needaction_enabled)\n            - the needaction counter of the related action, taking into account\n              the action domain\n        '
        menu_ids = set()
        for menu in self:
            menu_ids.add(menu.id)
            ctx = {}
            if menu.action and menu.action.type in ('ir.actions.act_window', 'ir.actions.client') and menu.action.context:
                with tools.ignore(Exception):
                    eval_ctx = tools.UnquoteEvalContext(self._context)
                    ctx = safe_eval(menu.action.context, locals_dict=eval_ctx, nocopy=True) or {}
            menu_refs = ctx.get('needaction_menu_ref')
            if menu_refs:
                if not isinstance(menu_refs, list):
                    menu_refs = [menu_refs]
                for menu_ref in menu_refs:
                    record = self.env.ref(menu_ref, False)
                    if record and record._name == 'ir.ui.menu':
                        menu_ids.add(record.id)
        res = {}
        for menu in self.browse(menu_ids):
            res[menu.id] = {'needaction_enabled': False, 'needaction_counter': False}
            if menu.action and menu.action.type in ('ir.actions.act_window', 'ir.actions.client') and menu.action.res_model:
                if menu.action.res_model in self.env:
                    model = self.env[menu.action.res_model]
                    if model._needaction:
                        if menu.action.type == 'ir.actions.act_window':
                            eval_context = self.env['ir.actions.act_window']._get_eval_context()
                            dom = safe_eval(menu.action.domain or '[]', eval_context)
                        else:
                            dom = safe_eval(menu.action.params_store or '{}', {'uid': self._uid}).get('domain')
                        res[menu.id]['needaction_enabled'] = model._needaction
                        res[menu.id]['needaction_counter'] = model._needaction_count(dom)
        return res

    @api.model
    @api.returns('self')
    def get_user_roots(self):
        if False:
            return 10
        ' Return all root menu ids visible for the user.\n\n        :return: the root menu ids\n        :rtype: list(int)\n        '
        return self.search([('parent_id', '=', False)])

    @api.model
    @tools.ormcache_context('self._uid', keys=('lang',))
    def load_menus_root(self):
        if False:
            for i in range(10):
                print('nop')
        fields = ['name', 'sequence', 'parent_id', 'action', 'web_icon_data']
        menu_roots = self.get_user_roots()
        menu_roots_data = menu_roots.read(fields) if menu_roots else []
        return {'id': False, 'name': 'root', 'parent_id': [-1, ''], 'children': menu_roots_data, 'all_menu_ids': menu_roots.ids}

    @api.model
    @tools.ormcache_context('self._uid', 'debug', keys=('lang',))
    def load_menus(self, debug):
        if False:
            for i in range(10):
                print('nop')
        " Loads all menu items (all applications and their sub-menus).\n\n        :return: the menu root\n        :rtype: dict('children': menu_nodes)\n        "
        fields = ['name', 'sequence', 'parent_id', 'action', 'web_icon', 'web_icon_data']
        menu_roots = self.get_user_roots()
        menu_roots_data = menu_roots.read(fields) if menu_roots else []
        menu_root = {'id': False, 'name': 'root', 'parent_id': [-1, ''], 'children': menu_roots_data, 'all_menu_ids': menu_roots.ids}
        if not menu_roots_data:
            return menu_root
        menus = self.search([('id', 'child_of', menu_roots.ids)])
        menu_items = menus.read(fields)
        menu_items.extend(menu_roots_data)
        menu_root['all_menu_ids'] = menus.ids
        menu_items_map = {menu_item['id']: menu_item for menu_item in menu_items}
        for menu_item in menu_items:
            parent = menu_item['parent_id'] and menu_item['parent_id'][0]
            if parent in menu_items_map:
                menu_items_map[parent].setdefault('children', []).append(menu_item)
        for menu_item in menu_items:
            menu_item.setdefault('children', []).sort(key=operator.itemgetter('sequence'))
        return menu_root