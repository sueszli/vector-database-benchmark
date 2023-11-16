from operator import itemgetter
from odoo import api, fields, models, _
from odoo.exceptions import UserError
TYPE2FIELD = {'char': 'value_text', 'float': 'value_float', 'boolean': 'value_integer', 'integer': 'value_integer', 'text': 'value_text', 'binary': 'value_binary', 'many2one': 'value_reference', 'date': 'value_datetime', 'datetime': 'value_datetime', 'selection': 'value_text'}

class Property(models.Model):
    _name = 'ir.property'
    name = fields.Char(index=True)
    res_id = fields.Char(string='Resource', index=True, help='If not set, acts as a default value for new resources')
    company_id = fields.Many2one('res.company', string='Company', index=True)
    fields_id = fields.Many2one('ir.model.fields', string='Field', ondelete='cascade', required=True, index=True)
    value_float = fields.Float()
    value_integer = fields.Integer()
    value_text = fields.Text()
    value_binary = fields.Binary()
    value_reference = fields.Char()
    value_datetime = fields.Datetime()
    type = fields.Selection([('char', 'Char'), ('float', 'Float'), ('boolean', 'Boolean'), ('integer', 'Integer'), ('text', 'Text'), ('binary', 'Binary'), ('many2one', 'Many2One'), ('date', 'Date'), ('datetime', 'DateTime'), ('selection', 'Selection')], required=True, default='many2one', index=True)

    @api.multi
    def _update_values(self, values):
        if False:
            print('Hello World!')
        value = values.pop('value', None)
        if not value:
            return values
        prop = None
        type_ = values.get('type')
        if not type_:
            if self:
                prop = self[0]
                type_ = prop.type
            else:
                type_ = self._fields['type'].default(self)
        field = TYPE2FIELD.get(type_)
        if not field:
            raise UserError(_('Invalid type'))
        if field == 'value_reference':
            if isinstance(value, models.BaseModel):
                value = '%s,%d' % (value._name, value.id)
            elif isinstance(value, (int, long)):
                field_id = values.get('fields_id')
                if not field_id:
                    if not prop:
                        raise ValueError()
                    field_id = prop.fields_id
                else:
                    field_id = self.env['ir.model.fields'].browse(field_id)
                value = '%s,%d' % (field_id.relation, value)
        values[field] = value
        return values

    @api.multi
    def write(self, values):
        if False:
            i = 10
            return i + 15
        return super(Property, self).write(self._update_values(values))

    @api.model
    def create(self, values):
        if False:
            while True:
                i = 10
        return super(Property, self).create(self._update_values(values))

    @api.multi
    def get_by_record(self):
        if False:
            for i in range(10):
                print('nop')
        self.ensure_one()
        if self.type in ('char', 'text', 'selection'):
            return self.value_text
        elif self.type == 'float':
            return self.value_float
        elif self.type == 'boolean':
            return bool(self.value_integer)
        elif self.type == 'integer':
            return self.value_integer
        elif self.type == 'binary':
            return self.value_binary
        elif self.type == 'many2one':
            if not self.value_reference:
                return False
            (model, resource_id) = self.value_reference.split(',')
            return self.env[model].browse(int(resource_id)).exists()
        elif self.type == 'datetime':
            return self.value_datetime
        elif self.type == 'date':
            if not self.value_datetime:
                return False
            return fields.Date.to_string(fields.Datetime.from_string(self.value_datetime))
        return False

    @api.model
    def get(self, name, model, res_id=False):
        if False:
            print('Hello World!')
        domain = self._get_domain(name, model)
        if domain is not None:
            domain = [('res_id', '=', res_id)] + domain
            prop = self.search(domain, limit=1, order='company_id')
            if prop:
                return prop.get_by_record()
        return False

    def _get_domain(self, prop_name, model):
        if False:
            while True:
                i = 10
        self._cr.execute('SELECT id FROM ir_model_fields WHERE name=%s AND model=%s', (prop_name, model))
        res = self._cr.fetchone()
        if not res:
            return None
        company_id = self._context.get('force_company') or self.env['res.company']._company_default_get(model, res[0]).id
        return [('fields_id', '=', res[0]), ('company_id', 'in', [company_id, False])]

    @api.model
    def get_multi(self, name, model, ids):
        if False:
            i = 10
            return i + 15
        ' Read the property field `name` for the records of model `model` with\n            the given `ids`, and return a dictionary mapping `ids` to their\n            corresponding value.\n        '
        if not ids:
            return {}
        domain = self._get_domain(name, model)
        if domain is None:
            return dict.fromkeys(ids, False)
        refs = {'%s,%s' % (model, id): id for id in ids}
        refs[False] = False
        domain += [('res_id', 'in', list(refs))]
        props = self.search(domain, order='company_id asc')
        result = {}
        for prop in props:
            id = refs.pop(prop.res_id, None)
            if id is not None:
                result[id] = prop.get_by_record()
        default_value = result.pop(False, False)
        for id in ids:
            result.setdefault(id, default_value)
        return result

    @api.model
    def set_multi(self, name, model, values, default_value=None):
        if False:
            i = 10
            return i + 15
        ' Assign the property field `name` for the records of model `model`\n            with `values` (dictionary mapping record ids to their value).\n            If the value for a given record is the same as the default\n            value, the property entry will not be stored, to avoid bloating\n            the database.\n            If `default_value` is provided, that value will be used instead\n            of the computed default value, to determine whether the value\n            for a record should be stored or not.\n        '

        def clean(value):
            if False:
                for i in range(10):
                    print('nop')
            return value.id if isinstance(value, models.BaseModel) else value
        if not values:
            return
        if not default_value:
            domain = self._get_domain(name, model)
            if domain is None:
                raise Exception()
            default_value = clean(self.get(name, model))
        self._cr.execute('SELECT id FROM ir_model_fields WHERE name=%s AND model=%s', (name, model))
        field_id = self._cr.fetchone()[0]
        company_id = self.env.context.get('force_company') or self.env['res.company']._company_default_get(model, field_id).id
        refs = {'%s,%s' % (model, id): id for id in values}
        props = self.search([('fields_id', '=', field_id), ('company_id', '=', company_id), ('res_id', 'in', list(refs))])
        for prop in props:
            id = refs.pop(prop.res_id)
            value = clean(values[id])
            if value == default_value:
                prop.unlink()
            elif value != clean(prop.get_by_record()):
                prop.write({'value': value})
        for (ref, id) in refs.iteritems():
            value = clean(values[id])
            if value != default_value:
                self.create({'fields_id': field_id, 'company_id': company_id, 'res_id': ref, 'name': name, 'value': value, 'type': self.env[model]._fields[name].type})

    @api.model
    def search_multi(self, name, model, operator, value):
        if False:
            print('Hello World!')
        ' Return a domain for the records that match the given condition. '
        default_matches = False
        include_zero = False
        field = self.env[model]._fields[name]
        if field.type == 'many2one':
            comodel = field.comodel_name

            def makeref(value):
                if False:
                    while True:
                        i = 10
                return value and '%s,%s' % (comodel, value)
            if operator == '=':
                value = makeref(value)
                if value is False:
                    default_matches = True
            elif operator in ('!=', '<=', '<', '>', '>='):
                value = makeref(value)
            elif operator in ('in', 'not in'):
                value = map(makeref, value)
            elif operator in ('=like', '=ilike', 'like', 'not like', 'ilike', 'not ilike'):
                target = self.env[comodel]
                target_names = target.name_search(value, operator=operator, limit=None)
                target_ids = map(itemgetter(0), target_names)
                (operator, value) = ('in', map(makeref, target_ids))
        elif field.type in ('integer', 'float'):
            if value == 0 and operator == '=':
                operator = '!='
                include_zero = True
            elif value <= 0 and operator == '>=':
                operator = '<'
                include_zero = True
            elif value < 0 and operator == '>':
                operator = '<='
                include_zero = True
            elif value >= 0 and operator == '<=':
                operator = '>'
                include_zero = True
            elif value > 0 and operator == '<':
                operator = '>='
                include_zero = True
        domain = self._get_domain(name, model)
        if domain is None:
            raise Exception()
        props = self.search(domain + [(TYPE2FIELD[field.type], operator, value)])
        good_ids = []
        for prop in props:
            if prop.res_id:
                (res_model, res_id) = prop.res_id.split(',')
                good_ids.append(int(res_id))
            else:
                default_matches = True
        if include_zero:
            return [('id', 'not in', good_ids)]
        elif default_matches:
            all_ids = []
            props = self.search(domain + [('res_id', '!=', False)])
            for prop in props:
                (res_model, res_id) = prop.res_id.split(',')
                all_ids.append(int(res_id))
            bad_ids = list(set(all_ids) - set(good_ids))
            return [('id', 'not in', bad_ids)]
        else:
            return [('id', 'in', good_ids)]