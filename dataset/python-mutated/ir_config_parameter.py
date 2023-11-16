"""
Store database-specific configuration parameters
"""
import uuid
import logging
from odoo import api, fields, models
from odoo.tools import config, ormcache, mute_logger
_logger = logging.getLogger(__name__)
'\nA dictionary holding some configuration parameters to be initialized when the database is created.\n'
_default_parameters = {'database.secret': lambda : (str(uuid.uuid4()), ['base.group_erp_manager']), 'database.uuid': lambda : (str(uuid.uuid1()), []), 'database.create_date': lambda : (fields.Datetime.now(), ['base.group_user']), 'web.base.url': lambda : ('http://localhost:%s' % config.get('xmlrpc_port'), [])}

class IrConfigParameter(models.Model):
    """Per-database storage of configuration key-value pairs."""
    _name = 'ir.config_parameter'
    _rec_name = 'key'
    key = fields.Char(required=True, index=True)
    value = fields.Text(required=True)
    group_ids = fields.Many2many('res.groups', 'ir_config_parameter_groups_rel', 'icp_id', 'group_id', string='Groups')
    _sql_constraints = [('key_uniq', 'unique (key)', 'Key must be unique.')]

    @api.model_cr
    @mute_logger('odoo.addons.base.ir.ir_config_parameter')
    def init(self, force=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initializes the parameters listed in _default_parameters.\n        It overrides existing parameters if force is ``True``.\n        '
        for (key, func) in _default_parameters.iteritems():
            params = self.sudo().search([('key', '=', key)])
            if force or not params:
                (value, groups) = func()
                params.set_param(key, value, groups=groups)

    @api.model
    def get_param(self, key, default=False):
        if False:
            i = 10
            return i + 15
        'Retrieve the value for a given key.\n\n        :param string key: The key of the parameter value to retrieve.\n        :param string default: default value if parameter is missing.\n        :return: The value of the parameter, or ``default`` if it does not exist.\n        :rtype: string\n        '
        return self._get_param(key) or default

    @api.model
    @ormcache('self._uid', 'key')
    def _get_param(self, key):
        if False:
            print('Hello World!')
        params = self.search_read([('key', '=', key)], fields=['value'], limit=1)
        return params[0]['value'] if params else None

    @api.model
    def set_param(self, key, value, groups=()):
        if False:
            i = 10
            return i + 15
        'Sets the value of a parameter.\n\n        :param string key: The key of the parameter value to set.\n        :param string value: The value to set.\n        :param list of string groups: List of group (xml_id allowed) to read this key.\n        :return: the previous value of the parameter or False if it did\n                 not exist.\n        :rtype: string\n        '
        self._get_param.clear_cache(self)
        param = self.search([('key', '=', key)])
        gids = []
        for group_xml in groups:
            group = self.env.ref(group_xml, raise_if_not_found=False)
            if group:
                gids.append((4, group.id))
            else:
                _logger.warning('Potential Security Issue: Group [%s] is not found.' % group_xml)
        vals = {'value': value}
        if gids:
            vals.update(group_ids=gids)
        if param:
            old = param.value
            if value is not False and value is not None:
                param.write(vals)
            else:
                param.unlink()
            return old
        else:
            vals.update(key=key)
            if value is not False and value is not None:
                self.create(vals)
            return False

    @api.multi
    def write(self, vals):
        if False:
            i = 10
            return i + 15
        self.clear_caches()
        return super(IrConfigParameter, self).write(vals)

    @api.multi
    def unlink(self):
        if False:
            print('Hello World!')
        self.clear_caches()
        return super(IrConfigParameter, self).unlink()