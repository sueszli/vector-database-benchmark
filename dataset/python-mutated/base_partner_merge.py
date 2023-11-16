from ast import literal_eval
from email.utils import parseaddr
import functools
import htmlentitydefs
import itertools
import logging
import operator
import psycopg2
import re
from .validate_email import validate_email
from odoo import api, fields, models
from odoo import SUPERUSER_ID, _
from odoo.exceptions import ValidationError, UserError
from odoo.tools import mute_logger
_logger = logging.getLogger('base.partner.merge')

def html_entity_decode_char(m, defs=htmlentitydefs.entitydefs):
    if False:
        print('Hello World!')
    try:
        return defs[m.group(1)]
    except KeyError:
        return m.group(0)

def html_entity_decode(string):
    if False:
        print('Hello World!')
    pattern = re.compile('&(\\w+?);')
    return pattern.sub(html_entity_decode_char, string)

def sanitize_email(email):
    if False:
        i = 10
        return i + 15
    assert isinstance(email, basestring) and email
    result = re.subn(';|/|:', ',', html_entity_decode(email or ''))[0].split(',')
    emails = [parseaddr(email)[1] for item in result for email in item.split()]
    return [email.lower() for email in emails if validate_email(email)]

class MergePartnerLine(models.TransientModel):
    _name = 'base.partner.merge.line'
    _order = 'min_id asc'
    wizard_id = fields.Many2one('base.partner.merge.automatic.wizard', 'Wizard')
    min_id = fields.Integer('MinID')
    aggr_ids = fields.Char('Ids', required=True)

class MergePartnerAutomatic(models.TransientModel):
    """
        The idea behind this wizard is to create a list of potential partners to
        merge. We use two objects, the first one is the wizard for the end-user.
        And the second will contain the partner list to merge.
    """
    _name = 'base.partner.merge.automatic.wizard'

    @api.model
    def default_get(self, fields):
        if False:
            i = 10
            return i + 15
        res = super(MergePartnerAutomatic, self).default_get(fields)
        active_ids = self.env.context.get('active_ids')
        if self.env.context.get('active_model') == 'res.partner' and active_ids:
            res['state'] = 'selection'
            res['partner_ids'] = active_ids
            res['dst_partner_id'] = self._get_ordered_partner(active_ids)[-1].id
        return res
    group_by_email = fields.Boolean('Email')
    group_by_name = fields.Boolean('Name')
    group_by_is_company = fields.Boolean('Is Company')
    group_by_vat = fields.Boolean('VAT')
    group_by_parent_id = fields.Boolean('Parent Company')
    state = fields.Selection([('option', 'Option'), ('selection', 'Selection'), ('finished', 'Finished')], readonly=True, required=True, string='State', default='option')
    number_group = fields.Integer('Group of Contacts', readonly=True)
    current_line_id = fields.Many2one('base.partner.merge.line', string='Current Line')
    line_ids = fields.One2many('base.partner.merge.line', 'wizard_id', string='Lines')
    partner_ids = fields.Many2many('res.partner', string='Contacts')
    dst_partner_id = fields.Many2one('res.partner', string='Destination Contact')
    exclude_contact = fields.Boolean('A user associated to the contact')
    exclude_journal_item = fields.Boolean('Journal Items associated to the contact')
    maximum_group = fields.Integer('Maximum of Group of Contacts')

    def _get_fk_on(self, table):
        if False:
            print('Hello World!')
        " return a list of many2one relation with the given table.\n            :param table : the name of the sql table to return relations\n            :returns a list of tuple 'table name', 'column name'.\n        "
        query = "\n            SELECT cl1.relname as table, att1.attname as column\n            FROM pg_constraint as con, pg_class as cl1, pg_class as cl2, pg_attribute as att1, pg_attribute as att2\n            WHERE con.conrelid = cl1.oid\n                AND con.confrelid = cl2.oid\n                AND array_lower(con.conkey, 1) = 1\n                AND con.conkey[1] = att1.attnum\n                AND att1.attrelid = cl1.oid\n                AND cl2.relname = %s\n                AND att2.attname = 'id'\n                AND array_lower(con.confkey, 1) = 1\n                AND con.confkey[1] = att2.attnum\n                AND att2.attrelid = cl2.oid\n                AND con.contype = 'f'\n        "
        self._cr.execute(query, (table,))
        return self._cr.fetchall()

    @api.model
    def _update_foreign_keys(self, src_partners, dst_partner):
        if False:
            print('Hello World!')
        ' Update all foreign key from the src_partner to dst_partner. All many2one fields will be updated.\n            :param src_partners : merge source res.partner recordset (does not include destination one)\n            :param dst_partner : record of destination res.partner\n        '
        _logger.debug('_update_foreign_keys for dst_partner: %s for src_partners: %s', dst_partner.id, str(src_partners.ids))
        Partner = self.env['res.partner']
        relations = self._get_fk_on('res_partner')
        for (table, column) in relations:
            if 'base_partner_merge_' in table:
                continue
            query = "SELECT column_name FROM information_schema.columns WHERE table_name LIKE '%s'" % table
            self._cr.execute(query, ())
            columns = []
            for data in self._cr.fetchall():
                if data[0] != column:
                    columns.append(data[0])
            query_dic = {'table': table, 'column': column, 'value': columns[0]}
            if len(columns) <= 1:
                query = '\n                    UPDATE "%(table)s" as ___tu\n                    SET %(column)s = %%s\n                    WHERE\n                        %(column)s = %%s AND\n                        NOT EXISTS (\n                            SELECT 1\n                            FROM "%(table)s" as ___tw\n                            WHERE\n                                %(column)s = %%s AND\n                                ___tu.%(value)s = ___tw.%(value)s\n                        )' % query_dic
                for partner in src_partners:
                    self._cr.execute(query, (dst_partner.id, partner.id, dst_partner.id))
            else:
                try:
                    with mute_logger('odoo.sql_db'), self._cr.savepoint():
                        query = 'UPDATE "%(table)s" SET %(column)s = %%s WHERE %(column)s IN %%s' % query_dic
                        self._cr.execute(query, (dst_partner.id, tuple(src_partners.ids)))
                        if column == Partner._parent_name and table == 'res_partner':
                            query = '\n                                WITH RECURSIVE cycle(id, parent_id) AS (\n                                        SELECT id, parent_id FROM res_partner\n                                    UNION\n                                        SELECT  cycle.id, res_partner.parent_id\n                                        FROM    res_partner, cycle\n                                        WHERE   res_partner.id = cycle.parent_id AND\n                                                cycle.id != cycle.parent_id\n                                )\n                                SELECT id FROM cycle WHERE id = parent_id AND id = %s\n                            '
                            self._cr.execute(query, (dst_partner.id,))
                except psycopg2.Error:
                    query = 'DELETE FROM %(table)s WHERE %(column)s IN %%s' % query_dic
                    self._cr.execute(query, (tuple(src_partners.ids),))

    @api.model
    def _update_reference_fields(self, src_partners, dst_partner):
        if False:
            while True:
                i = 10
        ' Update all reference fields from the src_partner to dst_partner.\n            :param src_partners : merge source res.partner recordset (does not include destination one)\n            :param dst_partner : record of destination res.partner\n        '
        _logger.debug('_update_reference_fields for dst_partner: %s for src_partners: %r', dst_partner.id, src_partners.ids)

        def update_records(model, src, field_model='model', field_id='res_id'):
            if False:
                print('Hello World!')
            Model = self.env[model] if model in self.env else None
            if Model is None:
                return
            records = Model.sudo().search([(field_model, '=', 'res.partner'), (field_id, '=', src.id)])
            try:
                with mute_logger('odoo.sql_db'), self._cr.savepoint():
                    return records.sudo().write({field_id: dst_partner.id})
            except psycopg2.Error:
                return records.sudo().unlink()
        update_records = functools.partial(update_records)
        for partner in src_partners:
            update_records('calendar', src=partner, field_model='model_id.model')
            update_records('ir.attachment', src=partner, field_model='res_model')
            update_records('mail.followers', src=partner, field_model='res_model')
            update_records('mail.message', src=partner)
            update_records('marketing.campaign.workitem', src=partner, field_model='object_id.model')
            update_records('ir.model.data', src=partner)
        records = self.env['ir.model.fields'].search([('ttype', '=', 'reference')])
        for record in records.sudo():
            try:
                Model = self.env[record.model]
                field = Model._fields[record.name]
            except KeyError:
                continue
            if field.compute is not None:
                continue
            for partner in src_partners:
                records_ref = Model.sudo().search([(record.name, '=', 'res.partner,%d' % partner.id)])
                values = {record.name: 'res.partner,%d' % dst_partner.id}
                records_ref.sudo().write(values)

    @api.model
    def _update_values(self, src_partners, dst_partner):
        if False:
            for i in range(10):
                print('nop')
        ' Update values of dst_partner with the ones from the src_partners.\n            :param src_partners : recordset of source res.partner\n            :param dst_partner : record of destination res.partner\n        '
        _logger.debug('_update_values for dst_partner: %s for src_partners: %r', dst_partner.id, src_partners.ids)
        model_fields = dst_partner._fields

        def write_serializer(item):
            if False:
                return 10
            if isinstance(item, models.BaseModel):
                return item.id
            else:
                return item
        values = dict()
        for (column, field) in model_fields.iteritems():
            if field.type not in ('many2many', 'one2many') and field.compute is None:
                for item in itertools.chain(src_partners, [dst_partner]):
                    if item[column]:
                        values[column] = write_serializer(item[column])
        values.pop('id', None)
        parent_id = values.pop('parent_id', None)
        dst_partner.write(values)
        if parent_id and parent_id != dst_partner.id:
            try:
                dst_partner.write({'parent_id': parent_id})
            except ValidationError:
                _logger.info('Skip recursive partner hierarchies for parent_id %s of partner: %s', parent_id, dst_partner.id)

    def _merge(self, partner_ids, dst_partner=None):
        if False:
            i = 10
            return i + 15
        ' private implementation of merge partner\n            :param partner_ids : ids of partner to merge\n            :param dst_partner : record of destination res.partner\n        '
        Partner = self.env['res.partner']
        partner_ids = Partner.browse(partner_ids).exists()
        if len(partner_ids) < 2:
            return
        if len(partner_ids) > 3:
            raise UserError(_('For safety reasons, you cannot merge more than 3 contacts together. You can re-open the wizard several times if needed.'))
        child_ids = self.env['res.partner']
        for partner_id in partner_ids:
            child_ids |= Partner.search([('id', 'child_of', [partner_id.id])]) - partner_id
        if partner_ids & child_ids:
            raise UserError(_('You cannot merge a contact with one of his parent.'))
        if SUPERUSER_ID != self.env.uid and len(set((partner.email for partner in partner_ids))) > 1:
            raise UserError(_('All contacts must have the same email. Only the Administrator can merge contacts with different emails.'))
        if dst_partner and dst_partner in partner_ids:
            src_partners = partner_ids - dst_partner
        else:
            ordered_partners = self._get_ordered_partner(partner_ids.ids)
            dst_partner = ordered_partners[-1]
            src_partners = ordered_partners[:-1]
        _logger.info('dst_partner: %s', dst_partner.id)
        if SUPERUSER_ID != self.env.uid and 'account.move.line' in self.env and self.env['account.move.line'].sudo().search([('partner_id', 'in', [partner.id for partner in src_partners])]):
            raise UserError(_('Only the destination contact may be linked to existing Journal Items. Please ask the Administrator if you need to merge several contacts linked to existing Journal Items.'))
        self._update_foreign_keys(src_partners, dst_partner)
        self._update_reference_fields(src_partners, dst_partner)
        self._update_values(src_partners, dst_partner)
        _logger.info('(uid = %s) merged the partners %r with %s', self._uid, src_partners.ids, dst_partner.id)
        dst_partner.message_post(body='%s %s' % (_('Merged with the following partners:'), ', '.join(('%s <%s> (ID %s)' % (p.name, p.email or 'n/a', p.id) for p in src_partners))))
        src_partners.unlink()

    @api.model
    def _generate_query(self, fields, maximum_group=100):
        if False:
            return 10
        ' Build the SQL query on res.partner table to group them according to given criteria\n            :param fields : list of column names to group by the partners\n            :param maximum_group : limit of the query\n        '
        sql_fields = []
        for field in fields:
            if field in ['email', 'name']:
                sql_fields.append('lower(%s)' % field)
            elif field in ['vat']:
                sql_fields.append("replace(%s, ' ', '')" % field)
            else:
                sql_fields.append(field)
        group_fields = ', '.join(sql_fields)
        filters = []
        for field in fields:
            if field in ['email', 'name', 'vat']:
                filters.append((field, 'IS NOT', 'NULL'))
        criteria = ' AND '.join(('%s %s %s' % (field, operator, value) for (field, operator, value) in filters))
        text = ['SELECT min(id), array_agg(id)', 'FROM res_partner']
        if criteria:
            text.append('WHERE %s' % criteria)
        text.extend(['GROUP BY %s' % group_fields, 'HAVING COUNT(*) >= 2', 'ORDER BY min(id)'])
        if maximum_group:
            text.append('LIMIT %s' % maximum_group)
        return ' '.join(text)

    @api.model
    def _compute_selected_groupby(self):
        if False:
            while True:
                i = 10
        ' Returns the list of field names the partner can be grouped (as merge\n            criteria) according to the option checked on the wizard\n        '
        groups = []
        group_by_prefix = 'group_by_'
        for field_name in self._fields:
            if field_name.startswith(group_by_prefix):
                if getattr(self, field_name, False):
                    groups.append(field_name[len(group_by_prefix):])
        if not groups:
            raise UserError(_('You have to specify a filter for your selection'))
        return groups

    @api.model
    def _partner_use_in(self, aggr_ids, models):
        if False:
            while True:
                i = 10
        ' Check if there is no occurence of this group of partner in the selected model\n            :param aggr_ids : stringified list of partner ids separated with a comma (sql array_agg)\n            :param models : dict mapping a model name with its foreign key with res_partner table\n        '
        return any((self.env[model].search_count([(field, 'in', aggr_ids)]) for (model, field) in models.iteritems()))

    @api.model
    def _get_ordered_partner(self, partner_ids):
        if False:
            print('Hello World!')
        ' Helper : returns a `res.partner` recordset ordered by create_date/active fields\n            :param partner_ids : list of partner ids to sort\n        '
        return self.env['res.partner'].browse(partner_ids).sorted(key=lambda p: (p.active, p.create_date), reverse=True)

    @api.multi
    def _compute_models(self):
        if False:
            while True:
                i = 10
        ' Compute the different models needed by the system if you want to exclude some partners. '
        model_mapping = {}
        if self.exclude_contact:
            model_mapping['res.users'] = 'partner_id'
        if 'account.move.line' in self.env and self.exclude_journal_item:
            model_mapping['account.move.line'] = 'partner_id'
        return model_mapping

    @api.multi
    def action_skip(self):
        if False:
            while True:
                i = 10
        " Skip this wizard line. Don't compute any thing, and simply redirect to the new step."
        if self.current_line_id:
            self.current_line_id.unlink()
        return self._action_next_screen()

    @api.multi
    def _action_next_screen(self):
        if False:
            while True:
                i = 10
        ' return the action of the next screen ; this means the wizard is set to treat the\n            next wizard line. Each line is a subset of partner that can be merged together.\n            If no line left, the end screen will be displayed (but an action is still returned).\n        '
        self.invalidate_cache()
        values = {}
        if self.line_ids:
            current_line = self.line_ids[0]
            current_partner_ids = literal_eval(current_line.aggr_ids)
            values.update({'current_line_id': current_line.id, 'partner_ids': [(6, 0, current_partner_ids)], 'dst_partner_id': self._get_ordered_partner(current_partner_ids)[-1].id, 'state': 'selection'})
        else:
            values.update({'current_line_id': False, 'partner_ids': [], 'state': 'finished'})
        self.write(values)
        return {'type': 'ir.actions.act_window', 'res_model': self._name, 'res_id': self.id, 'view_mode': 'form', 'target': 'new'}

    @api.multi
    def _process_query(self, query):
        if False:
            while True:
                i = 10
        ' Execute the select request and write the result in this wizard\n            :param query : the SQL query used to fill the wizard line\n        '
        self.ensure_one()
        model_mapping = self._compute_models()
        self._cr.execute(query)
        counter = 0
        for (min_id, aggr_ids) in self._cr.fetchall():
            partners = self.env['res.partner'].search([('id', 'in', aggr_ids)])
            if len(partners) < 2:
                continue
            if model_mapping and self._partner_use_in(partners.ids, model_mapping):
                continue
            self.env['base.partner.merge.line'].create({'wizard_id': self.id, 'min_id': min_id, 'aggr_ids': partners.ids})
            counter += 1
        self.write({'state': 'selection', 'number_group': counter})
        _logger.info('counter: %s', counter)

    @api.multi
    def action_start_manual_process(self):
        if False:
            for i in range(10):
                print('nop')
        " Start the process 'Merge with Manual Check'. Fill the wizard according to the group_by and exclude\n            options, and redirect to the first step (treatment of first wizard line). After, for each subset of\n            partner to merge, the wizard will be actualized.\n                - Compute the selected groups (with duplication)\n                - If the user has selected the 'exclude_xxx' fields, avoid the partners\n        "
        self.ensure_one()
        groups = self._compute_selected_groupby()
        query = self._generate_query(groups, self.maximum_group)
        self._process_query(query)
        return self._action_next_screen()

    @api.multi
    def action_start_automatic_process(self):
        if False:
            return 10
        " Start the process 'Merge Automatically'. This will fill the wizard with the same mechanism as 'Merge\n            with Manual Check', but instead of refreshing wizard with the current line, it will automatically process\n            all lines by merging partner grouped according to the checked options.\n        "
        self.ensure_one()
        self.action_start_manual_process()
        self.invalidate_cache()
        for line in self.line_ids:
            partner_ids = literal_eval(line.aggr_ids)
            self._merge(partner_ids)
            line.unlink()
            self._cr.commit()
        self.write({'state': 'finished'})
        return {'type': 'ir.actions.act_window', 'res_model': self._name, 'res_id': self.id, 'view_mode': 'form', 'target': 'new'}

    @api.multi
    def parent_migration_process_cb(self):
        if False:
            print('Hello World!')
        self.ensure_one()
        query = '\n            SELECT\n                min(p1.id),\n                array_agg(DISTINCT p1.id)\n            FROM\n                res_partner as p1\n            INNER join\n                res_partner as p2\n            ON\n                p1.email = p2.email AND\n                p1.name = p2.name AND\n                (p1.parent_id = p2.id OR p1.id = p2.parent_id)\n            WHERE\n                p2.id IS NOT NULL\n            GROUP BY\n                p1.email,\n                p1.name,\n                CASE WHEN p1.parent_id = p2.id THEN p2.id\n                    ELSE p1.id\n                END\n            HAVING COUNT(*) >= 2\n            ORDER BY\n                min(p1.id)\n        '
        self._process_query(query)
        for line in self.line_ids:
            partner_ids = literal_eval(line.aggr_ids)
            self._merge(partner_ids)
            line.unlink()
            self._cr.commit()
        self.write({'state': 'finished'})
        self._cr.execute('\n            UPDATE\n                res_partner\n            SET\n                is_company = NULL,\n                parent_id = NULL\n            WHERE\n                parent_id = id\n        ')
        return {'type': 'ir.actions.act_window', 'res_model': self._name, 'res_id': self.id, 'view_mode': 'form', 'target': 'new'}

    @api.multi
    def action_update_all_process(self):
        if False:
            return 10
        self.ensure_one()
        self.parent_migration_process_cb()
        wizard = self.create({'group_by_vat': True, 'group_by_email': True, 'group_by_name': True})
        wizard.action_start_automatic_process()
        self._cr.execute('\n            UPDATE\n                res_partner\n            SET\n                is_company = NULL\n            WHERE\n                parent_id IS NOT NULL AND\n                is_company IS NOT NULL\n        ')
        return self._action_next_screen()

    @api.multi
    def action_merge(self):
        if False:
            for i in range(10):
                print('nop')
        ' Merge Contact button. Merge the selected partners, and redirect to\n            the end screen (since there is no other wizard line to process.\n        '
        if not self.partner_ids:
            self.write({'state': 'finished'})
            return {'type': 'ir.actions.act_window', 'res_model': self._name, 'res_id': self.id, 'view_mode': 'form', 'target': 'new'}
        self._merge(self.partner_ids.ids, self.dst_partner_id)
        if self.current_line_id:
            self.current_line_id.unlink()
        return self._action_next_screen()