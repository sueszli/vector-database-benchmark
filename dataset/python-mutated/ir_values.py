from ast import literal_eval
from odoo import api, fields, models, tools, _
from odoo.exceptions import AccessError, MissingError
from odoo.tools import pickle
EXCLUDED_FIELDS = set(('report_sxw_content', 'report_rml_content', 'report_sxw', 'report_rml', 'report_sxw_content_data', 'report_rml_content_data', 'search_view'))
ACTION_SLOTS = ['client_action_multi', 'client_print_multi', 'client_action_relate', 'tree_but_open', 'tree_but_action']

class IrValues(models.Model):
    """Holds internal model-specific action bindings and user-defined default
       field values. definitions. This is a legacy internal model, mixing
       two different concepts, and will likely be updated or replaced in a
       future version by cleaner, separate models. You should not depend
       explicitly on it.

       The purpose of each ``ir.values`` entry depends on its type, defined
       by the ``key`` column:

        * 'default': user-defined default values, used when creating new
          records of this model:
        * 'action': binding of an action to a particular *action slot* of
          this model, making the action easily available in the user
          interface for this model.

       The ``key2`` column acts as a qualifier, further refining the type
       of the entry. The possible values are:

        * for 'default' entries: an optional condition restricting the
          cases where this particular default value will be applicable,
          or ``False`` for no condition
        * for 'action' entries: the ``key2`` qualifier is one of the available
          action slots, defining how this action can be invoked:

            * ``'client_print_multi'`` for report printing actions that will
              be available on views displaying items from this model
            * ``'client_action_multi'`` for assistants (wizards) actions
              that will be available in views displaying objects of this model
            * ``'client_action_relate'`` for links towards related documents
              that should be available in views displaying objects of this model
            * ``'tree_but_open'`` for actions that will be triggered when
              double-clicking an item from this model in a hierarchical tree view

       Each entry is specific to a model (``model`` column), and for ``'actions'``
       type, may even be made specific to a given record of that model when the
       ``res_id`` column contains a record ID (``False`` means it's global for
       all records).

       The content of the entry is defined by the ``value`` column, which may either
       contain an arbitrary value, or a reference string defining the action that
       should be executed.

       .. rubric:: Usage: default values
       
       The ``'default'`` entries are usually defined manually by the
       users, and set by their UI clients calling :meth:`~.set_default`.
       These default values are then automatically used by the
       ORM every time a new record is about to be created, i.e. when
       :meth:`~odoo.models.Model.default_get`
       or :meth:`~odoo.models.Model.create` are called.

       .. rubric:: Usage: action bindings

       Business applications will usually bind their actions during
       installation, and Odoo UI clients will apply them as defined,
       based on the list of actions included in the result of
       :meth:`~odoo.models.Model.fields_view_get`,
       or directly returned by explicit calls to :meth:`~.get_actions`.
    """
    _name = 'ir.values'
    name = fields.Char(required=True)
    model = fields.Char(string='Model Name', index=True, required=True, help='Model to which this entry applies')
    model_id = fields.Many2one('ir.model', string='Model (change only)', help='Model to which this entry applies - helper field for setting a model, will automatically set the correct model name')
    action_id = fields.Many2one('ir.actions.actions', string='Action (change only)', help='Action bound to this entry - helper field for binding an action, will automatically set the correct reference')
    value = fields.Text(help='Default value (pickled) or reference to an action')
    value_unpickle = fields.Text(string='Default value or action reference', compute='_value_unpickle', inverse='_value_pickle')
    key = fields.Selection([('action', 'Action'), ('default', 'Default')], string='Type', index=True, required=True, default='action', help='- Action: an action attached to one slot of the given model\n- Default: a default value for a model field')
    key2 = fields.Char(string='Qualifier', index=True, default='tree_but_open', help='For actions, one of the possible action slots: \n  - client_action_multi\n  - client_print_multi\n  - client_action_relate\n  - tree_but_open\nFor defaults, an optional condition')
    res_id = fields.Integer(string='Record ID', index=True, help='Database identifier of the record to which this applies. 0 = for all records')
    user_id = fields.Many2one('res.users', string='User', ondelete='cascade', index=True, help='If set, action binding only applies for this user.')
    company_id = fields.Many2one('res.company', string='Company', ondelete='cascade', index=True, help='If set, action binding only applies for this company')

    @api.depends('key', 'value')
    def _value_unpickle(self):
        if False:
            return 10
        for record in self:
            value = record.value
            if record.key == 'default' and value:
                with tools.ignore(Exception):
                    value = str(pickle.loads(value))
            record.value_unpickle = value

    def _value_pickle(self):
        if False:
            for i in range(10):
                print('nop')
        context = dict(self._context)
        context.pop(self.CONCURRENCY_CHECK_FIELD, None)
        for record in self.with_context(context):
            value = record.value_unpickle
            if record.model in self.env and record.name in self.env[record.model]._fields:
                field = self.env[record.model]._fields[record.name]
                if field.type not in ['char', 'text', 'html', 'selection']:
                    value = literal_eval(value)
            if record.key == 'default':
                value = pickle.dumps(value)
            record.value = value

    @api.onchange('model_id')
    def onchange_object_id(self):
        if False:
            for i in range(10):
                print('nop')
        if self.model_id:
            self.model = self.model_id.model

    @api.onchange('action_id')
    def onchange_action_id(self):
        if False:
            print('Hello World!')
        if self.action_id:
            self.value_unpickle = self.action_id

    @api.model_cr_context
    def _auto_init(self):
        if False:
            i = 10
            return i + 15
        res = super(IrValues, self)._auto_init()
        self._cr.execute("SELECT indexname FROM pg_indexes WHERE indexname = 'ir_values_key_model_key2_res_id_user_id_idx'")
        if not self._cr.fetchone():
            self._cr.execute('CREATE INDEX ir_values_key_model_key2_res_id_user_id_idx ON ir_values (key, model, key2, res_id, user_id)')
        return res

    @api.model
    def create(self, vals):
        if False:
            print('Hello World!')
        self.clear_caches()
        return super(IrValues, self).create(vals)

    @api.multi
    def write(self, vals):
        if False:
            return 10
        self.clear_caches()
        return super(IrValues, self).write(vals)

    @api.multi
    def unlink(self):
        if False:
            return 10
        self.clear_caches()
        return super(IrValues, self).unlink()

    @api.model
    @api.returns('self', lambda value: value.id)
    def set_default(self, model, field_name, value, for_all_users=True, company_id=False, condition=False):
        if False:
            print('Hello World!')
        "Defines a default value for the given model and field_name. Any previous\n           default for the same scope (model, field_name, value, for_all_users, company_id, condition)\n           will be replaced and lost in the process.\n\n           Defaults can be later retrieved via :meth:`~.get_defaults`, which will return\n           the highest priority default for any given field. Defaults that are more specific\n           have a higher priority, in the following order (highest to lowest):\n\n               * specific to user and company\n               * specific to user only\n               * specific to company only\n               * global to everyone\n\n           :param string model: model name\n           :param string field_name: field name to which the default applies\n           :param value: the default field value to set\n           :type value: any serializable Python value\n           :param bool for_all_users: whether the default should apply to everybody or only\n                                      the user calling the method\n           :param int company_id: optional ID of the company to which the default should\n                                  apply. If omitted, the default will be global. If True\n                                  is passed, the current user's company will be used.\n           :param string condition: optional condition specification that can be used to\n                                    restrict the applicability of the default values\n                                    (e.g. based on another field's value). This is an\n                                    opaque string as far as the API is concerned, but client\n                                    stacks typically use single-field conditions in the\n                                    form ``'key=stringified_value'``.\n                                    (Currently, the condition is trimmed to 200 characters,\n                                    so values that share the same first 200 characters always\n                                    match)\n           :return: the newly created ir.values entry\n        "
        if isinstance(value, unicode):
            value = value.encode('utf8')
        if company_id is True:
            company_id = self.env.user.company_id.id
        search_criteria = [('key', '=', 'default'), ('key2', '=', condition and condition[:200]), ('model', '=', model), ('name', '=', field_name), ('user_id', '=', False if for_all_users else self._uid), ('company_id', '=', company_id)]
        self.search(search_criteria).unlink()
        return self.create({'name': field_name, 'value': pickle.dumps(value), 'model': model, 'key': 'default', 'key2': condition and condition[:200], 'user_id': False if for_all_users else self._uid, 'company_id': company_id})

    @api.model
    def get_default(self, model, field_name, for_all_users=True, company_id=False, condition=False):
        if False:
            print('Hello World!')
        ' Return the default value defined for model, field_name, users, company and condition.\n            Return ``None`` if no such default exists.\n        '
        search_criteria = [('key', '=', 'default'), ('key2', '=', condition and condition[:200]), ('model', '=', model), ('name', '=', field_name), ('user_id', '=', False if for_all_users else self._uid), ('company_id', '=', company_id)]
        defaults = self.search(search_criteria)
        return pickle.loads(defaults.value.encode('utf-8')) if defaults else None

    @api.model
    def get_defaults(self, model, condition=False):
        if False:
            print('Hello World!')
        "Returns any default values that are defined for the current model and user,\n           (and match ``condition``, if specified), previously registered via\n           :meth:`~.set_default`.\n\n           Defaults are global to a model, not field-specific, but an optional\n           ``condition`` can be provided to restrict matching default values\n           to those that were defined for the same condition (usually based\n           on another field's value).\n\n           Default values also have priorities depending on whom they apply\n           to: only the highest priority value will be returned for any\n           field. See :meth:`~.set_default` for more details.\n\n           :param string model: model name\n           :param string condition: optional condition specification that can be used to\n                                    restrict the applicability of the default values\n                                    (e.g. based on another field's value). This is an\n                                    opaque string as far as the API is concerned, but client\n                                    stacks typically use single-field conditions in the\n                                    form ``'key=stringified_value'``.\n                                    (Currently, the condition is trimmed to 200 characters,\n                                    so values that share the same first 200 characters always\n                                    match)\n           :return: list of default values tuples of the form ``(id, field_name, value)``\n                    (``id`` is the ID of the default entry, usually irrelevant)\n        "
        query = ' SELECT v.id, v.name, v.value FROM ir_values v\n                    LEFT JOIN res_users u ON (v.user_id = u.id)\n                    WHERE v.key = %%s AND v.model = %%s\n                        AND (v.user_id = %%s OR v.user_id IS NULL)\n                        AND (v.company_id IS NULL OR\n                             v.company_id = (SELECT company_id FROM res_users WHERE id = %%s)\n                            )\n                    %s\n                    ORDER BY v.user_id, u.company_id'
        params = ('default', model, self._uid, self._uid)
        if condition:
            query = query % 'AND v.key2 = %s'
            params += (condition[:200],)
        else:
            query = query % 'AND v.key2 IS NULL'
        self._cr.execute(query, params)
        defaults = {}
        for row in self._cr.dictfetchall():
            value = pickle.loads(row['value'].encode('utf-8'))
            defaults.setdefault(row['name'], (row['id'], row['name'], value))
        return defaults.values()

    @api.model
    @tools.ormcache('self._uid', 'model', 'condition')
    def get_defaults_dict(self, model, condition=False):
        if False:
            while True:
                i = 10
        ' Returns a dictionary mapping field names with their corresponding\n            default value. This method simply improves the returned value of\n            :meth:`~.get_defaults`.\n        '
        return dict(((f, v) for (i, f, v) in self.get_defaults(model, condition)))

    @api.model
    @api.returns('self', lambda value: value.id)
    def set_action(self, name, action_slot, model, action, res_id=False):
        if False:
            while True:
                i = 10
        "Binds an the given action to the given model's action slot - for later\n           retrieval via :meth:`~.get_actions`. Any existing binding of the same action\n           to the same slot is first removed, allowing an update of the action's name.\n           See the class description for more details about the various action\n           slots: :class:`~ir_values`.\n\n           :param string name: action label, usually displayed by UI client\n           :param string action_slot: the action slot to which the action should be\n                                      bound to - one of ``client_action_multi``,\n                                      ``client_print_multi``, ``client_action_relate``,\n                                      ``tree_but_open``.\n           :param string model: model name\n           :param string action: action reference, in the form ``'model,id'``\n           :param int res_id: optional record id - will bind the action only to a\n                              specific record of the model, not all records.\n           :return: the newly created ir.values entry\n        "
        assert isinstance(action, basestring) and ',' in action, 'Action definition must be an action reference, e.g. "ir.actions.act_window,42"'
        assert action_slot in ACTION_SLOTS, 'Action slot (%s) must be one of: %r' % (action_slot, ACTION_SLOTS)
        search_criteria = [('key', '=', 'action'), ('key2', '=', action_slot), ('model', '=', model), ('res_id', '=', res_id or 0), ('value', '=', action)]
        self.search(search_criteria).unlink()
        return self.create({'key': 'action', 'key2': action_slot, 'model': model, 'res_id': res_id, 'name': name, 'value': action})

    @api.model
    @tools.ormcache_context('self._uid', 'action_slot', 'model', 'res_id', keys=('lang',))
    def get_actions(self, action_slot, model, res_id=False):
        if False:
            for i in range(10):
                print('nop')
        "Retrieves the list of actions bound to the given model's action slot.\n           See the class description for more details about the various action\n           slots: :class:`~.ir_values`.\n\n           :param string action_slot: the action slot to which the actions should be\n                                      bound to - one of ``client_action_multi``,\n                                      ``client_print_multi``, ``client_action_relate``,\n                                      ``tree_but_open``.\n           :param string model: model name\n           :param int res_id: optional record id - will bind the action only to a\n                              specific record of the model, not all records.\n           :return: list of action tuples of the form ``(id, name, action_def)``,\n                    where ``id`` is the ID of the default entry, ``name`` is the\n                    action label, and ``action_def`` is a dict containing the\n                    action definition as obtained by calling\n                    :meth:`~odoo.models.Model.read` on the action record.\n        "
        assert action_slot in ACTION_SLOTS, 'Illegal action slot value: %s' % action_slot
        query = ' SELECT v.id, v.name, v.value FROM ir_values v\n                    WHERE v.key = %s AND v.key2 = %s AND v.model = %s\n                        AND (v.res_id = %s OR v.res_id IS NULL OR v.res_id = 0)\n                    ORDER BY v.id '
        self._cr.execute(query, ('action', action_slot, model, res_id or None))
        actions = []
        for (id, name, value) in self._cr.fetchall():
            if not value:
                continue
            (action_model, action_id) = value.split(',')
            if action_model not in self.env:
                continue
            action = self.env[action_model].browse(int(action_id))
            actions.append((id, name, action))
        results = {}
        for (id, name, action) in actions:
            fields = [field for field in action._fields if field not in EXCLUDED_FIELDS]
            try:
                action_def = {field: action._fields[field].convert_to_read(action[field], action) for field in fields}
                if action._name in ('ir.actions.report.xml', 'ir.actions.act_window'):
                    if action.groups_id and (not action.groups_id & self.env.user.groups_id):
                        if name == 'Menuitem':
                            raise AccessError(_('You do not have the permission to perform this operation!!!'))
                        continue
                results[name] = (id, name, action_def)
            except (AccessError, MissingError):
                continue
        return sorted(results.values())