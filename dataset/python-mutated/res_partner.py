import base64
import datetime
import hashlib
import pytz
import threading
import urllib2
import urlparse
from email.utils import formataddr
from lxml import etree
from odoo import api, fields, models, tools, SUPERUSER_ID, _
from odoo.modules import get_module_resource
from odoo.osv.expression import get_unaccent_wrapper
from odoo.exceptions import UserError, ValidationError
from odoo.osv.orm import browse_record
WARNING_MESSAGE = [('no-message', 'No Message'), ('warning', 'Warning'), ('block', 'Blocking Message')]
WARNING_HELP = _('Selecting the "Warning" option will notify user with the message, Selecting "Blocking Message" will throw an exception with the message and block the flow. The Message has to be written in the next field.')
ADDRESS_FORMAT_CLASSES = {'%(city)s %(state_code)s\n%(zip)s': 'o_city_state', '%(zip)s %(city)s': 'o_zip_city'}
ADDRESS_FIELDS = ('street', 'street2', 'zip', 'city', 'state_id', 'country_id')

@api.model
def _lang_get(self):
    if False:
        i = 10
        return i + 15
    return self.env['res.lang'].get_installed()

@api.model
def _tz_get(self):
    if False:
        for i in range(10):
            print('nop')
    return [(tz, tz) for tz in sorted(pytz.all_timezones, key=lambda tz: tz if not tz.startswith('Etc/') else '_')]

class FormatAddress(object):

    @api.model
    def fields_view_get_address(self, arch):
        if False:
            print('Hello World!')
        address_format = self.env.user.company_id.country_id.address_format or ''
        for (format_pattern, format_class) in ADDRESS_FORMAT_CLASSES.iteritems():
            if format_pattern in address_format:
                doc = etree.fromstring(arch)
                for address_node in doc.xpath("//div[@class='o_address_format']"):
                    address_node.attrib['class'] += ' ' + format_class
                    if format_class.startswith('o_zip'):
                        zip_fields = address_node.xpath("//field[@name='zip']")
                        city_fields = address_node.xpath("//field[@name='city']")
                        if zip_fields and city_fields:
                            city_fields[0].addprevious(zip_fields[0])
                arch = etree.tostring(doc)
                break
        return arch

class PartnerCategory(models.Model):
    _description = 'Partner Tags'
    _name = 'res.partner.category'
    _order = 'parent_left, name'
    _parent_store = True
    _parent_order = 'name'
    name = fields.Char(string='Tag Name', required=True, translate=True)
    color = fields.Integer(string='Color Index')
    parent_id = fields.Many2one('res.partner.category', string='Parent Category', index=True, ondelete='cascade')
    child_ids = fields.One2many('res.partner.category', 'parent_id', string='Child Tags')
    active = fields.Boolean(default=True, help='The active field allows you to hide the category without removing it.')
    parent_left = fields.Integer(string='Left parent', index=True)
    parent_right = fields.Integer(string='Right parent', index=True)
    partner_ids = fields.Many2many('res.partner', column1='category_id', column2='partner_id', string='Partners')

    @api.constrains('parent_id')
    def _check_parent_id(self):
        if False:
            print('Hello World!')
        if not self._check_recursion():
            raise ValidationError(_('Error ! You can not create recursive tags.'))

    @api.multi
    def name_get(self):
        if False:
            i = 10
            return i + 15
        " Return the categories' display name, including their direct\n            parent by default.\n\n            If ``context['partner_category_display']`` is ``'short'``, the short\n            version of the category name (without the direct parent) is used.\n            The default is the long version.\n        "
        if self._context.get('partner_category_display') == 'short':
            return super(PartnerCategory, self).name_get()
        res = []
        for category in self:
            names = []
            current = category
            while current:
                names.append(current.name)
                current = current.parent_id
            res.append((category.id, ' / '.join(reversed(names))))
        return res

    @api.model
    def name_search(self, name, args=None, operator='ilike', limit=100):
        if False:
            print('Hello World!')
        args = args or []
        if name:
            name = name.split(' / ')[-1]
            args = [('name', operator, name)] + args
        return self.search(args, limit=limit).name_get()

class PartnerTitle(models.Model):
    _name = 'res.partner.title'
    _order = 'name'
    name = fields.Char(string='Title', required=True, translate=True)
    shortcut = fields.Char(string='Abbreviation', translate=True)
    _sql_constraints = [('name_uniq', 'unique (name)', 'Title name already exists !')]

class Partner(models.Model, FormatAddress):
    _description = 'Partner'
    _name = 'res.partner'
    _order = 'display_name'

    def _default_category(self):
        if False:
            print('Hello World!')
        return self.env['res.partner.category'].browse(self._context.get('category_id'))

    def _default_company(self):
        if False:
            while True:
                i = 10
        return self.env['res.company']._company_default_get('res.partner')
    name = fields.Char(index=True)
    display_name = fields.Char(compute='_compute_display_name', store=True, index=True)
    date = fields.Date(index=True)
    title = fields.Many2one('res.partner.title')
    parent_id = fields.Many2one('res.partner', string='Related Company', index=True)
    parent_name = fields.Char(related='parent_id.name', readonly=True, string='Parent name')
    child_ids = fields.One2many('res.partner', 'parent_id', string='Contacts', domain=[('active', '=', True)])
    ref = fields.Char(string='Internal Reference', index=True)
    lang = fields.Selection(_lang_get, string='Language', default=lambda self: self.env.lang, help='If the selected language is loaded in the system, all documents related to this contact will be printed in this language. If not, it will be English.')
    tz = fields.Selection(_tz_get, string='Timezone', default=lambda self: self._context.get('tz'), help="The partner's timezone, used to output proper date and time values inside printed reports. It is important to set a value for this field. You should use the same timezone that is otherwise used to pick and render date and time values: your computer's timezone.")
    tz_offset = fields.Char(compute='_compute_tz_offset', string='Timezone offset', invisible=True)
    user_id = fields.Many2one('res.users', string='Salesperson', help='The internal user that is in charge of communicating with this contact if any.')
    vat = fields.Char(string='TIN', help='Tax Identification Number. Fill it if the company is subjected to taxes. Used by the some of the legal statements.')
    bank_ids = fields.One2many('res.partner.bank', 'partner_id', string='Banks')
    website = fields.Char(help='Website of Partner or Company')
    comment = fields.Text(string='Notes')
    category_id = fields.Many2many('res.partner.category', column1='partner_id', column2='category_id', string='Tags', default=_default_category)
    credit_limit = fields.Float(string='Credit Limit')
    barcode = fields.Char(oldname='ean13')
    active = fields.Boolean(default=True)
    customer = fields.Boolean(string='Is a Customer', default=True, help='Check this box if this contact is a customer.')
    supplier = fields.Boolean(string='Is a Vendor', help="Check this box if this contact is a vendor. If it's not checked, purchase people will not see it when encoding a purchase order.")
    employee = fields.Boolean(help='Check this box if this contact is an Employee.')
    function = fields.Char(string='Job Position')
    type = fields.Selection([('contact', 'Contact'), ('invoice', 'Invoice address'), ('delivery', 'Shipping address'), ('other', 'Other address')], string='Address Type', default='contact', help='Used to select automatically the right address according to the context in sales and purchases documents.')
    street = fields.Char()
    street2 = fields.Char()
    zip = fields.Char(change_default=True)
    city = fields.Char()
    state_id = fields.Many2one('res.country.state', string='State', ondelete='restrict')
    country_id = fields.Many2one('res.country', string='Country', ondelete='restrict')
    email = fields.Char()
    email_formatted = fields.Char('Formatted Email', compute='_compute_email_formatted', help='Format email address "Name <email@domain>"')
    phone = fields.Char()
    fax = fields.Char()
    mobile = fields.Char()
    is_company = fields.Boolean(string='Is a Company', default=False, help='Check if the contact is a company, otherwise it is a person')
    company_type = fields.Selection(string='Company Type', selection=[('person', 'Individual'), ('company', 'Company')], compute='_compute_company_type', readonly=False)
    company_id = fields.Many2one('res.company', 'Company', index=True, default=_default_company)
    color = fields.Integer(string='Color Index', default=0)
    user_ids = fields.One2many('res.users', 'partner_id', string='Users', auto_join=True)
    partner_share = fields.Boolean('Share Partner', compute='_compute_partner_share', store=True, help='Either customer (no user), either shared user. Indicated the current partner is a customer without access or with a limited access created for sharing data.')
    contact_address = fields.Char(compute='_compute_contact_address', string='Complete Address')
    commercial_partner_id = fields.Many2one('res.partner', compute='_compute_commercial_partner', string='Commercial Entity', store=True, index=True)
    commercial_company_name = fields.Char('Company Name Entity', compute='_compute_commercial_company_name', store=True)
    company_name = fields.Char('Company Name')
    image = fields.Binary('Image', attachment=True, help='This field holds the image used as avatar for this contact, limited to 1024x1024px')
    image_medium = fields.Binary('Medium-sized image', attachment=True, help='Medium-sized image of this contact. It is automatically resized as a 128x128px image, with aspect ratio preserved. Use this field in form views or some kanban views.')
    image_small = fields.Binary('Small-sized image', attachment=True, help='Small-sized image of this contact. It is automatically resized as a 64x64px image, with aspect ratio preserved. Use this field anywhere a small image is required.')
    self = fields.Many2one(comodel_name=_name, compute='_compute_get_ids')
    _sql_constraints = [('check_name', "CHECK( (type='contact' AND name IS NOT NULL) or (type!='contact') )", 'Contacts require a name.')]

    @api.depends('is_company', 'name', 'parent_id.name', 'type', 'company_name')
    def _compute_display_name(self):
        if False:
            for i in range(10):
                print('nop')
        diff = dict(show_address=None, show_address_only=None, show_email=None)
        names = dict(self.with_context(**diff).name_get())
        for partner in self:
            partner.display_name = names.get(partner.id)

    @api.depends('tz')
    def _compute_tz_offset(self):
        if False:
            i = 10
            return i + 15
        for partner in self:
            partner.tz_offset = datetime.datetime.now(pytz.timezone(partner.tz or 'GMT')).strftime('%z')

    @api.depends('user_ids.share')
    def _compute_partner_share(self):
        if False:
            return 10
        for partner in self:
            partner.partner_share = not partner.user_ids or any((user.share for user in partner.user_ids))

    @api.depends(lambda self: self._display_address_depends())
    def _compute_contact_address(self):
        if False:
            return 10
        for partner in self:
            partner.contact_address = partner._display_address()

    @api.one
    def _compute_get_ids(self):
        if False:
            for i in range(10):
                print('nop')
        self.self = self.id

    @api.depends('is_company', 'parent_id.commercial_partner_id')
    def _compute_commercial_partner(self):
        if False:
            while True:
                i = 10
        for partner in self:
            if partner.is_company or not partner.parent_id:
                partner.commercial_partner_id = partner
            else:
                partner.commercial_partner_id = partner.parent_id.commercial_partner_id

    @api.depends('company_name', 'parent_id.is_company', 'commercial_partner_id.name')
    def _compute_commercial_company_name(self):
        if False:
            i = 10
            return i + 15
        for partner in self:
            p = partner.commercial_partner_id
            partner.commercial_company_name = p.is_company and p.name or partner.company_name

    @api.model
    def _get_default_image(self, partner_type, is_company, parent_id):
        if False:
            print('Hello World!')
        if getattr(threading.currentThread(), 'testing', False) or self._context.get('install_mode'):
            return False
        (colorize, img_path, image) = (False, False, False)
        if partner_type in ['other'] and parent_id:
            parent_image = self.browse(parent_id).image
            image = parent_image and parent_image.decode('base64') or None
        if not image and partner_type == 'invoice':
            img_path = get_module_resource('base', 'static/src/img', 'money.png')
        elif not image and partner_type == 'delivery':
            img_path = get_module_resource('base', 'static/src/img', 'truck.png')
        elif not image and is_company:
            img_path = get_module_resource('base', 'static/src/img', 'company_image.png')
        elif not image:
            img_path = get_module_resource('base', 'static/src/img', 'avatar.png')
            colorize = True
        if img_path:
            with open(img_path, 'rb') as f:
                image = f.read()
        if image and colorize:
            image = tools.image_colorize(image)
        return tools.image_resize_image_big(image.encode('base64'))

    @api.model
    def fields_view_get(self, view_id=None, view_type='form', toolbar=False, submenu=False):
        if False:
            i = 10
            return i + 15
        if not view_id and view_type == 'form' and self._context.get('force_email'):
            view_id = self.env.ref('base.view_partner_simple_form').id
        res = super(Partner, self).fields_view_get(view_id=view_id, view_type=view_type, toolbar=toolbar, submenu=submenu)
        if view_type == 'form':
            res['arch'] = self.fields_view_get_address(res['arch'])
        return res

    @api.constrains('parent_id')
    def _check_parent_id(self):
        if False:
            return 10
        if not self._check_recursion():
            raise ValidationError(_('You cannot create recursive Partner hierarchies.'))

    @api.multi
    def copy(self, default=None):
        if False:
            for i in range(10):
                print('nop')
        self.ensure_one()
        default = dict(default or {}, name=_('%s (copy)') % self.name)
        return super(Partner, self).copy(default)

    @api.onchange('parent_id')
    def onchange_parent_id(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.parent_id:
            return
        result = {}
        partner = getattr(self, '_origin', self)
        if partner.parent_id and partner.parent_id != self.parent_id:
            result['warning'] = {'title': _('Warning'), 'message': _('Changing the company of a contact should only be done if it was never correctly set. If an existing contact starts working for a new company then a new contact should be created under that new company. You can use the "Discard" button to abandon this change.')}
        if partner.type == 'contact' or self.type == 'contact':
            address_fields = self._address_fields()
            if any((self.parent_id[key] for key in address_fields)):

                def convert(value):
                    if False:
                        while True:
                            i = 10
                    return value.id if isinstance(value, models.BaseModel) else value
                result['value'] = {key: convert(self.parent_id[key]) for key in address_fields}
        return result

    @api.onchange('country_id')
    def _onchange_country_id(self):
        if False:
            for i in range(10):
                print('nop')
        if self.country_id:
            return {'domain': {'state_id': [('country_id', '=', self.country_id.id)]}}
        else:
            return {'domain': {'state_id': []}}

    @api.onchange('email')
    def onchange_email(self):
        if False:
            return 10
        if not self.image and (not self._context.get('yaml_onchange')) and self.email:
            self.image = self._get_gravatar_image(self.email)

    @api.depends('name', 'email')
    def _compute_email_formatted(self):
        if False:
            for i in range(10):
                print('nop')
        for partner in self:
            partner.email_formatted = formataddr((partner.name, partner.email))

    @api.depends('is_company')
    def _compute_company_type(self):
        if False:
            print('Hello World!')
        for partner in self:
            partner.company_type = 'company' if partner.is_company else 'person'

    @api.onchange('company_type')
    def onchange_company_type(self):
        if False:
            for i in range(10):
                print('nop')
        self.is_company = self.company_type == 'company'

    @api.multi
    def _update_fields_values(self, fields):
        if False:
            return 10
        ' Returns dict of write() values for synchronizing ``fields`` '
        values = {}
        for fname in fields:
            field = self._fields[fname]
            if field.type == 'many2one':
                values[fname] = self[fname].id
            elif field.type == 'one2many':
                raise AssertionError(_('One2Many fields cannot be synchronized as part of `commercial_fields` or `address fields`'))
            elif field.type == 'many2many':
                values[fname] = [(6, 0, self[fname].ids)]
            else:
                values[fname] = self[fname]
        return values

    @api.model
    def _address_fields(self):
        if False:
            print('Hello World!')
        'Returns the list of address fields that are synced from the parent.'
        return list(ADDRESS_FIELDS)

    @api.multi
    def update_address(self, vals):
        if False:
            while True:
                i = 10
        addr_vals = {key: vals[key] for key in self._address_fields() if key in vals}
        if addr_vals:
            return super(Partner, self).write(addr_vals)

    @api.model
    def _commercial_fields(self):
        if False:
            print('Hello World!')
        " Returns the list of fields that are managed by the commercial entity\n        to which a partner belongs. These fields are meant to be hidden on\n        partners that aren't `commercial entities` themselves, and will be\n        delegated to the parent `commercial entity`. The list is meant to be\n        extended by inheriting classes. "
        return ['vat', 'credit_limit']

    @api.multi
    def _commercial_sync_from_company(self):
        if False:
            for i in range(10):
                print('nop')
        ' Handle sync of commercial fields when a new parent commercial entity is set,\n        as if they were related fields '
        commercial_partner = self.commercial_partner_id
        if commercial_partner != self:
            sync_vals = commercial_partner._update_fields_values(self._commercial_fields())
            self.write(sync_vals)

    @api.multi
    def _commercial_sync_to_children(self):
        if False:
            while True:
                i = 10
        ' Handle sync of commercial fields to descendants '
        commercial_partner = self.commercial_partner_id
        sync_vals = commercial_partner._update_fields_values(self._commercial_fields())
        sync_children = self.child_ids.filtered(lambda c: not c.is_company)
        for child in sync_children:
            child._commercial_sync_to_children()
        sync_children._compute_commercial_partner()
        return sync_children.write(sync_vals)

    @api.multi
    def _fields_sync(self, values):
        if False:
            for i in range(10):
                print('nop')
        ' Sync commercial fields and address fields from company and to children after create/update,\n        just as if those were all modeled as fields.related to the parent '
        if values.get('parent_id') or values.get('type', 'contact'):
            if values.get('parent_id'):
                self._commercial_sync_from_company()
            if self.parent_id and self.type == 'contact':
                onchange_vals = self.onchange_parent_id().get('value', {})
                self.update_address(onchange_vals)
        if self.child_ids:
            if self.commercial_partner_id == self:
                commercial_fields = self._commercial_fields()
                if any((field in values for field in commercial_fields)):
                    self._commercial_sync_to_children()
            for child in self.child_ids.filtered(lambda c: not c.is_company):
                if child.commercial_partner_id != self.commercial_partner_id:
                    self._commercial_sync_to_children()
                    break
            address_fields = self._address_fields()
            if any((field in values for field in address_fields)):
                contacts = self.child_ids.filtered(lambda c: c.type == 'contact')
                contacts.update_address(values)

    @api.multi
    def _handle_first_contact_creation(self):
        if False:
            while True:
                i = 10
        ' On creation of first contact for a company (or root) that has no address, assume contact address\n        was meant to be company address '
        parent = self.parent_id
        address_fields = self._address_fields()
        if (parent.is_company or not parent.parent_id) and len(parent.child_ids) == 1 and any((self[f] for f in address_fields)) and (not any((parent[f] for f in address_fields))):
            addr_vals = self._update_fields_values(address_fields)
            parent.update_address(addr_vals)

    def _clean_website(self, website):
        if False:
            i = 10
            return i + 15
        (scheme, netloc, path, params, query, fragment) = urlparse.urlparse(website)
        if not scheme:
            if not netloc:
                (netloc, path) = (path, '')
            website = urlparse.urlunparse(('http', netloc, path, params, query, fragment))
        return website

    @api.multi
    def write(self, vals):
        if False:
            for i in range(10):
                print('nop')
        if vals.get('website'):
            vals['website'] = self._clean_website(vals['website'])
        if vals.get('parent_id'):
            vals['company_name'] = False
        if vals.get('company_id'):
            company = self.env['res.company'].browse(vals['company_id'])
            for partner in self:
                if partner.user_ids:
                    companies = set((user.company_id for user in partner.user_ids))
                    if len(companies) > 1 or company not in companies:
                        raise UserError(_('You can not change the company as the partner/user has multiple user linked with different companies.'))
        tools.image_resize_images(vals)
        result = True
        if 'is_company' in vals and self.user_has_groups('base.group_partner_manager') and (not self.env.uid == SUPERUSER_ID):
            result = super(Partner, self).sudo().write({'is_company': vals.get('is_company')})
            del vals['is_company']
        result = result and super(Partner, self).write(vals)
        for partner in self:
            if any((u.has_group('base.group_user') for u in partner.user_ids if u != self.env.user)):
                self.env['res.users'].check_access_rights('write')
            partner._fields_sync(vals)
        return result

    @api.model
    def create(self, vals):
        if False:
            i = 10
            return i + 15
        if vals.get('website'):
            vals['website'] = self._clean_website(vals['website'])
        if vals.get('parent_id'):
            vals['company_name'] = False
        if not vals.get('image'):
            vals['image'] = self._get_default_image(vals.get('type'), vals.get('is_company'), vals.get('parent_id'))
        tools.image_resize_images(vals)
        partner = super(Partner, self).create(vals)
        partner._fields_sync(vals)
        partner._handle_first_contact_creation()
        return partner

    @api.multi
    def create_company(self):
        if False:
            while True:
                i = 10
        self.ensure_one()
        if self.company_name:
            values = dict(name=self.company_name, is_company=True)
            values.update(self._update_fields_values(self._address_fields()))
            new_company = self.create(values)
            self.write({'parent_id': new_company.id, 'child_ids': [(1, partner_id, dict(parent_id=new_company.id)) for partner_id in self.child_ids.ids]})
        return True

    @api.multi
    def open_commercial_entity(self):
        if False:
            for i in range(10):
                print('nop')
        ' Utility method used to add an "Open Company" button in partner views '
        self.ensure_one()
        return {'type': 'ir.actions.act_window', 'res_model': 'res.partner', 'view_mode': 'form', 'res_id': self.commercial_partner_id.id, 'target': 'current', 'flags': {'form': {'action_buttons': True}}}

    @api.multi
    def open_parent(self):
        if False:
            return 10
        ' Utility method used to add an "Open Parent" button in partner views '
        self.ensure_one()
        address_form_id = self.env.ref('base.view_partner_address_form').id
        return {'type': 'ir.actions.act_window', 'res_model': 'res.partner', 'view_mode': 'form', 'views': [(address_form_id, 'form')], 'res_id': self.parent_id.id, 'target': 'new', 'flags': {'form': {'action_buttons': True}}}

    @api.multi
    def name_get(self):
        if False:
            while True:
                i = 10
        res = []
        for partner in self:
            name = partner.name or ''
            if partner.company_name or partner.parent_id:
                if not name and partner.type in ['invoice', 'delivery', 'other']:
                    name = dict(self.fields_get(['type'])['type']['selection'])[partner.type]
                if not partner.is_company:
                    name = '%s, %s' % (partner.commercial_company_name or partner.parent_id.name, name)
            if self._context.get('show_address_only'):
                name = partner._display_address(without_company=True)
            if self._context.get('show_address'):
                name = name + '\n' + partner._display_address(without_company=True)
            name = name.replace('\n\n', '\n')
            name = name.replace('\n\n', '\n')
            if self._context.get('show_email') and partner.email:
                name = '%s <%s>' % (name, partner.email)
            if self._context.get('html_format'):
                name = name.replace('\n', '<br/>')
            res.append((partner.id, name))
        return res

    def _parse_partner_name(self, text, context=None):
        if False:
            while True:
                i = 10
        " Supported syntax:\n            - 'Raoul <raoul@grosbedon.fr>': will find name and email address\n            - otherwise: default, everything is set as the name "
        emails = tools.email_split(text.replace(' ', ','))
        if emails:
            email = emails[0]
            name = text[:text.index(email)].replace('"', '').replace('<', '').strip()
        else:
            (name, email) = (text, '')
        return (name, email)

    @api.model
    def name_create(self, name):
        if False:
            while True:
                i = 10
        " Override of orm's name_create method for partners. The purpose is\n            to handle some basic formats to create partners using the\n            name_create.\n            If only an email address is received and that the regex cannot find\n            a name, the name will have the email value.\n            If 'force_email' key in context: must find the email address. "
        (name, email) = self._parse_partner_name(name)
        if self._context.get('force_email') and (not email):
            raise UserError(_("Couldn't create contact without email address!"))
        if not name and email:
            name = email
        partner = self.create({self._rec_name: name or email, 'email': email or self.env.context.get('default_email', False)})
        return partner.name_get()[0]

    @api.model
    def _search(self, args, offset=0, limit=None, order=None, count=False, access_rights_uid=None):
        if False:
            for i in range(10):
                print('nop')
        " Override search() to always show inactive children when searching via ``child_of`` operator. The ORM will\n        always call search() with a simple domain of the form [('parent_id', 'in', [ids])]. "
        if len(args) == 1 and len(args[0]) == 3 and (args[0][:2] == ('parent_id', 'in')) and (args[0][2] != [False]):
            self = self.with_context(active_test=False)
        return super(Partner, self)._search(args, offset=offset, limit=limit, order=order, count=count, access_rights_uid=access_rights_uid)

    @api.model
    def name_search(self, name, args=None, operator='ilike', limit=100):
        if False:
            while True:
                i = 10
        if args is None:
            args = []
        if name and operator in ('=', 'ilike', '=ilike', 'like', '=like'):
            self.check_access_rights('read')
            where_query = self._where_calc(args)
            self._apply_ir_rules(where_query, 'read')
            (from_clause, where_clause, where_clause_params) = where_query.get_sql()
            where_str = where_clause and ' WHERE %s AND ' % where_clause or ' WHERE '
            search_name = name
            if operator in ('ilike', 'like'):
                search_name = '%%%s%%' % name
            if operator in ('=ilike', '=like'):
                operator = operator[1:]
            unaccent = get_unaccent_wrapper(self.env.cr)
            query = "SELECT id\n                         FROM res_partner\n                      {where} ({email} {operator} {percent}\n                           OR {display_name} {operator} {percent}\n                           OR {reference} {operator} {percent})\n                           -- don't panic, trust postgres bitmap\n                     ORDER BY {display_name} {operator} {percent} desc,\n                              {display_name}\n                    ".format(where=where_str, operator=operator, email=unaccent('email'), display_name=unaccent('display_name'), reference=unaccent('ref'), percent=unaccent('%s'))
            where_clause_params += [search_name] * 4
            if limit:
                query += ' limit %s'
                where_clause_params.append(limit)
            self.env.cr.execute(query, where_clause_params)
            partner_ids = map(lambda x: x[0], self.env.cr.fetchall())
            if partner_ids:
                return self.browse(partner_ids).name_get()
            else:
                return []
        return super(Partner, self).name_search(name, args, operator=operator, limit=limit)

    @api.model
    def find_or_create(self, email):
        if False:
            while True:
                i = 10
        ' Find a partner with the given ``email`` or use :py:method:`~.name_create`\n            to create one\n\n            :param str email: email-like string, which should contain at least one email,\n                e.g. ``"Raoul Grosbedon <r.g@grosbedon.fr>"``'
        assert email, 'an email is required for find_or_create to work'
        emails = tools.email_split(email)
        if emails:
            email = emails[0]
        partners = self.search([('email', '=ilike', email)], limit=1)
        return partners.id or self.name_create(email)[0]

    def _get_gravatar_image(self, email):
        if False:
            i = 10
            return i + 15
        gravatar_image = False
        email_hash = hashlib.md5(email.lower()).hexdigest()
        url = 'https://www.gravatar.com/avatar/' + email_hash
        try:
            image_content = urllib2.urlopen(url + '?d=404&s=128', timeout=5).read()
            gravatar_image = base64.b64encode(image_content)
        except Exception:
            pass
        return gravatar_image

    @api.multi
    def _email_send(self, email_from, subject, body, on_error=None):
        if False:
            while True:
                i = 10
        for partner in self.filtered('email'):
            tools.email_send(email_from, [partner.email], subject, body, on_error)
        return True

    @api.multi
    def address_get(self, adr_pref=None):
        if False:
            i = 10
            return i + 15
        " Find contacts/addresses of the right type(s) by doing a depth-first-search\n        through descendants within company boundaries (stop at entities flagged ``is_company``)\n        then continuing the search at the ancestors that are within the same company boundaries.\n        Defaults to partners of type ``'default'`` when the exact type is not found, or to the\n        provided partner itself if no type ``'default'`` is found either. "
        adr_pref = set(adr_pref or [])
        if 'contact' not in adr_pref:
            adr_pref.add('contact')
        result = {}
        visited = set()
        for partner in self:
            current_partner = partner
            while current_partner:
                to_scan = [current_partner]
                while to_scan:
                    record = to_scan.pop(0)
                    visited.add(record)
                    if record.type in adr_pref and (not result.get(record.type)):
                        result[record.type] = record.id
                    if len(result) == len(adr_pref):
                        return result
                    to_scan = [c for c in record.child_ids if c not in visited if not c.is_company] + to_scan
                if current_partner.is_company or not current_partner.parent_id:
                    break
                current_partner = current_partner.parent_id
        default = result.get('contact', self.id or False)
        for adr_type in adr_pref:
            result[adr_type] = result.get(adr_type) or default
        return result

    @api.model
    def view_header_get(self, view_id, view_type):
        if False:
            while True:
                i = 10
        res = super(Partner, self).view_header_get(view_id, view_type)
        if res:
            return res
        if not self._context.get('category_id'):
            return False
        return _('Partners: ') + self.env['res.partner.category'].browse(self._context['category_id']).name

    @api.model
    @api.returns('self')
    def main_partner(self):
        if False:
            while True:
                i = 10
        ' Return the main partner '
        return self.env.ref('base.main_partner')

    @api.multi
    def _display_address(self, without_company=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        The purpose of this function is to build and return an address formatted accordingly to the\n        standards of the country where it belongs.\n\n        :param address: browse record of the res.partner to format\n        :returns: the address formatted in a display that fit its country habits (or the default ones\n            if not country is specified)\n        :rtype: string\n        '
        address_format = self.country_id.address_format or '%(street)s\n%(street2)s\n%(city)s %(state_code)s %(zip)s\n%(country_name)s'
        args = {'state_code': self.state_id.code or '', 'state_name': self.state_id.name or '', 'country_code': self.country_id.code or '', 'country_name': self.country_id.name or '', 'company_name': self.commercial_company_name or ''}
        for field in self._address_fields():
            args[field] = getattr(self, field) or ''
        if without_company:
            args['company_name'] = ''
        elif self.commercial_company_name:
            address_format = '%(company_name)s\n' + address_format
        return address_format % args

    def _display_address_depends(self):
        if False:
            print('Hello World!')
        return self._address_fields() + ['country_id.address_format', 'country_id.code', 'country_id.name', 'company_name', 'state_id.code', 'state_id.name']