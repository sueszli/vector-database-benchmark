import ast
from odoo import api, fields, models, _
from odoo.exceptions import UserError

class IrFilters(models.Model):
    _name = 'ir.filters'
    _description = 'Filters'
    _order = 'model_id, name, id desc'
    name = fields.Char(string='Filter Name', translate=True, required=True)
    user_id = fields.Many2one('res.users', string='User', ondelete='cascade', default=lambda self: self._uid, help='The user this filter is private to. When left empty the filter is public and available to all users.')
    domain = fields.Text(default='[]', required=True)
    context = fields.Text(default='{}', required=True)
    sort = fields.Text(default='[]', required=True)
    model_id = fields.Selection(selection='_list_all_models', string='Model', required=True)
    is_default = fields.Boolean(string='Default filter')
    action_id = fields.Many2one('ir.actions.actions', string='Action', ondelete='cascade', help='The menu action this filter applies to. When left empty the filter applies to all menus for this model.')
    active = fields.Boolean(default=True)

    @api.model
    def _list_all_models(self):
        if False:
            i = 10
            return i + 15
        self._cr.execute('SELECT model, name FROM ir_model ORDER BY name')
        return self._cr.fetchall()

    @api.multi
    def copy(self, default=None):
        if False:
            i = 10
            return i + 15
        self.ensure_one()
        default = dict(default or {}, name=_('%s (copy)') % self.name)
        return super(IrFilters, self).copy(default)

    @api.multi
    def _get_eval_domain(self):
        if False:
            for i in range(10):
                print('nop')
        self.ensure_one()
        return ast.literal_eval(self.domain)

    @api.model
    def _get_action_domain(self, action_id=None):
        if False:
            print('Hello World!')
        'Return a domain component for matching filters that are visible in the\n           same context (menu/view) as the given action.'
        if action_id:
            return [('action_id', 'in', [action_id, False])]
        return [('action_id', '=', False)]

    @api.model
    def get_filters(self, model, action_id=None):
        if False:
            while True:
                i = 10
        'Obtain the list of filters available for the user on the given model.\n\n        :param action_id: optional ID of action to restrict filters to this action\n            plus global filters. If missing only global filters are returned.\n            The action does not have to correspond to the model, it may only be\n            a contextual action.\n        :return: list of :meth:`~osv.read`-like dicts containing the\n            ``name``, ``is_default``, ``domain``, ``user_id`` (m2o tuple),\n            ``action_id`` (m2o tuple) and ``context`` of the matching ``ir.filters``.\n        '
        action_domain = self._get_action_domain(action_id)
        filters = self.search(action_domain + [('model_id', '=', model), ('user_id', 'in', [self._uid, False])])
        user_context = self.env.user.context_get()
        return filters.with_context(user_context).read(['name', 'is_default', 'domain', 'context', 'user_id', 'sort'])

    @api.model
    def _check_global_default(self, vals, matching_filters):
        if False:
            print('Hello World!')
        " _check_global_default(dict, list(dict), dict) -> None\n\n        Checks if there is a global default for the model_id requested.\n\n        If there is, and the default is different than the record being written\n        (-> we're not updating the current global default), raise an error\n        to avoid users unknowingly overwriting existing global defaults (they\n        have to explicitly remove the current default before setting a new one)\n\n        This method should only be called if ``vals`` is trying to set\n        ``is_default``\n\n        :raises odoo.exceptions.UserError: if there is an existing default and\n                                            we're not updating it\n        "
        domain = self._get_action_domain(vals.get('action_id'))
        defaults = self.search(domain + [('model_id', '=', vals['model_id']), ('user_id', '=', False), ('is_default', '=', True)])
        if not defaults:
            return
        if matching_filters and matching_filters[0]['id'] == defaults.id:
            return
        raise UserError(_('There is already a shared filter set as default for %(model)s, delete or change it before setting a new default') % {'model': vals.get('model_id')})

    @api.model
    @api.returns('self', lambda value: value.id)
    def create_or_replace(self, vals):
        if False:
            for i in range(10):
                print('nop')
        action_id = vals.get('action_id')
        current_filters = self.get_filters(vals['model_id'], action_id)
        matching_filters = [f for f in current_filters if f['name'].lower() == vals['name'].lower() if (f['user_id'] and f['user_id'][0]) == vals.get('user_id')]
        if vals.get('is_default'):
            if vals.get('user_id'):
                domain = self._get_action_domain(action_id)
                defaults = self.search(domain + [('model_id', '=', vals['model_id']), ('user_id', '=', vals['user_id']), ('is_default', '=', True)])
                if defaults:
                    defaults.write({'is_default': False})
            else:
                self._check_global_default(vals, matching_filters)
        if matching_filters:
            matching_filter = self.browse(matching_filters[0]['id'])
            matching_filter.write(vals)
            return matching_filter
        return self.create(vals)
    _sql_constraints = [('name_model_uid_unique', 'unique (name, model_id, user_id, action_id)', 'Filter names must be unique')]

    @api.model_cr_context
    def _auto_init(self):
        if False:
            for i in range(10):
                print('nop')
        result = super(IrFilters, self)._auto_init()
        self._cr.execute('DROP INDEX IF EXISTS ir_filters_name_model_uid_unique_index')
        self._cr.execute("SELECT indexname FROM pg_indexes WHERE indexname = 'ir_filters_name_model_uid_unique_action_index'")
        if not self._cr.fetchone():
            self._cr.execute('CREATE UNIQUE INDEX "ir_filters_name_model_uid_unique_action_index" ON ir_filters\n                                (lower(name), model_id, COALESCE(user_id,-1), COALESCE(action_id,-1))')
        return result