import json
import math
import re
import time
from odoo import api, fields, models, tools, _
CURRENCY_DISPLAY_PATTERN = re.compile('(\\w+)\\s*(?:\\((.*)\\))?')

class Currency(models.Model):
    _name = 'res.currency'
    _description = 'Currency'
    _order = 'name'
    name = fields.Char(string='Currency', size=3, required=True, help='Currency Code (ISO 4217)')
    symbol = fields.Char(help='Currency sign, to be used when printing amounts.', required=True)
    rate = fields.Float(compute='_compute_current_rate', string='Current Rate', digits=(12, 6), help='The rate of the currency to the currency of rate 1.')
    rate_ids = fields.One2many('res.currency.rate', 'currency_id', string='Rates')
    rounding = fields.Float(string='Rounding Factor', digits=(12, 6), default=0.01)
    decimal_places = fields.Integer(compute='_compute_decimal_places')
    active = fields.Boolean(default=True)
    position = fields.Selection([('after', 'After Amount'), ('before', 'Before Amount')], default='after', string='Symbol Position', help='Determines where the currency symbol should be placed after or before the amount.')
    date = fields.Date(compute='_compute_date')
    _sql_constraints = [('unique_name', 'unique (name)', 'The currency code must be unique!')]

    @api.multi
    def _compute_current_rate(self):
        if False:
            while True:
                i = 10
        date = self._context.get('date') or fields.Datetime.now()
        company_id = self._context.get('company_id') or self.env['res.users']._get_company().id
        query = 'SELECT c.id, (SELECT r.rate FROM res_currency_rate r\n                                  WHERE r.currency_id = c.id AND r.name <= %s\n                                    AND (r.company_id IS NULL OR r.company_id = %s)\n                               ORDER BY r.company_id, r.name DESC\n                                  LIMIT 1) AS rate\n                   FROM res_currency c\n                   WHERE c.id IN %s'
        self._cr.execute(query, (date, company_id, tuple(self.ids)))
        currency_rates = dict(self._cr.fetchall())
        for currency in self:
            currency.rate = currency_rates.get(currency.id) or 1.0

    @api.multi
    @api.depends('rounding')
    def _compute_decimal_places(self):
        if False:
            while True:
                i = 10
        for currency in self:
            if 0 < currency.rounding < 1:
                currency.decimal_places = int(math.ceil(math.log10(1 / currency.rounding)))
            else:
                currency.decimal_places = 0

    @api.multi
    @api.depends('rate_ids.name')
    def _compute_date(self):
        if False:
            while True:
                i = 10
        for currency in self:
            currency.date = currency.rate_ids[:1].name

    @api.model
    def name_search(self, name='', args=None, operator='ilike', limit=100):
        if False:
            print('Hello World!')
        results = super(Currency, self).name_search(name, args, operator=operator, limit=limit)
        if not results:
            name_match = CURRENCY_DISPLAY_PATTERN.match(name)
            if name_match:
                results = super(Currency, self).name_search(name_match.group(1), args, operator=operator, limit=limit)
        return results

    @api.multi
    def name_get(self):
        if False:
            print('Hello World!')
        return [(currency.id, tools.ustr(currency.name)) for currency in self]

    @api.multi
    def round(self, amount):
        if False:
            print('Hello World!')
        "Return ``amount`` rounded  according to ``self``'s rounding rules.\n\n           :param float amount: the amount to round\n           :return: rounded float\n        "
        return tools.float_round(amount, precision_rounding=self.rounding)

    @api.multi
    def compare_amounts(self, amount1, amount2):
        if False:
            return 10
        "Compare ``amount1`` and ``amount2`` after rounding them according to the\n           given currency's precision..\n           An amount is considered lower/greater than another amount if their rounded\n           value is different. This is not the same as having a non-zero difference!\n\n           For example 1.432 and 1.431 are equal at 2 digits precision,\n           so this method would return 0.\n           However 0.006 and 0.002 are considered different (returns 1) because\n           they respectively round to 0.01 and 0.0, even though\n           0.006-0.002 = 0.004 which would be considered zero at 2 digits precision.\n\n           :param float amount1: first amount to compare\n           :param float amount2: second amount to compare\n           :return: (resp.) -1, 0 or 1, if ``amount1`` is (resp.) lower than,\n                    equal to, or greater than ``amount2``, according to\n                    ``currency``'s rounding.\n\n           With the new API, call it like: ``currency.compare_amounts(amount1, amount2)``.\n        "
        return tools.float_compare(amount1, amount2, precision_rounding=self.rounding)

    @api.multi
    def is_zero(self, amount):
        if False:
            print('Hello World!')
        "Returns true if ``amount`` is small enough to be treated as\n           zero according to current currency's rounding rules.\n           Warning: ``is_zero(amount1-amount2)`` is not always equivalent to\n           ``compare_amounts(amount1,amount2) == 0``, as the former will round after\n           computing the difference, while the latter will round before, giving\n           different results for e.g. 0.006 and 0.002 at 2 digits precision.\n\n           :param float amount: amount to compare with currency's zero\n\n           With the new API, call it like: ``currency.is_zero(amount)``.\n        "
        return tools.float_is_zero(amount, precision_rounding=self.rounding)

    @api.model
    def _get_conversion_rate(self, from_currency, to_currency):
        if False:
            return 10
        from_currency = from_currency.with_env(self.env)
        to_currency = to_currency.with_env(self.env)
        return to_currency.rate / from_currency.rate

    @api.model
    def _compute(self, from_currency, to_currency, from_amount, round=True):
        if False:
            print('Hello World!')
        if to_currency == from_currency:
            amount = to_currency.round(from_amount) if round else from_amount
        else:
            rate = self._get_conversion_rate(from_currency, to_currency)
            amount = to_currency.round(from_amount * rate) if round else from_amount * rate
        return amount

    @api.multi
    def compute(self, from_amount, to_currency, round=True):
        if False:
            print('Hello World!')
        ' Convert `from_amount` from currency `self` to `to_currency`. '
        (self, to_currency) = (self or to_currency, to_currency or self)
        assert self, 'compute from unknown currency'
        assert to_currency, 'compute to unknown currency'
        if self == to_currency:
            to_amount = from_amount
        else:
            to_amount = from_amount * self._get_conversion_rate(self, to_currency)
        return to_currency.round(to_amount) if round else to_amount

    @api.model
    def get_format_currencies_js_function(self):
        if False:
            print('Hello World!')
        ' Returns a string that can be used to instanciate a javascript function that formats numbers as currencies.\n            That function expects the number as first parameter and the currency id as second parameter.\n            If the currency id parameter is false or undefined, the company currency is used.\n        '
        company_currency = self.env.user.with_env(self.env).company_id.currency_id
        function = ''
        for currency in self.search([]):
            symbol = currency.symbol or currency.name
            format_number_str = "openerp.web.format_value(arguments[0], {type: 'float', digits: [69,%s]}, 0.00)" % currency.decimal_places
            if currency.position == 'after':
                return_str = "return %s + '\\xA0' + %s;" % (format_number_str, json.dumps(symbol))
            else:
                return_str = "return %s + '\\xA0' + %s;" % (json.dumps(symbol), format_number_str)
            function += 'if (arguments[1] === %s) { %s }' % (currency.id, return_str)
            if currency == company_currency:
                company_currency_format = return_str
                function = 'if (arguments[1] === false || arguments[1] === undefined) {' + company_currency_format + ' }' + function
        return function

    def _select_companies_rates(self):
        if False:
            while True:
                i = 10
        return '\n            SELECT\n                r.currency_id,\n                COALESCE(r.company_id, c.id) as company_id,\n                r.rate,\n                r.name AS date_start,\n                (SELECT name FROM res_currency_rate r2\n                 WHERE r2.name > r.name AND\n                       r2.currency_id = r.currency_id AND\n                       (r2.company_id is null or r2.company_id = c.id)\n                 ORDER BY r2.name ASC\n                 LIMIT 1) AS date_end\n            FROM res_currency_rate r\n            JOIN res_company c ON (r.company_id is null or r.company_id = c.id)\n        '

class CurrencyRate(models.Model):
    _name = 'res.currency.rate'
    _description = 'Currency Rate'
    _order = 'name desc'
    name = fields.Datetime(string='Date', required=True, index=True, default=lambda self: fields.Date.today() + ' 00:00:00')
    rate = fields.Float(digits=(12, 6), help='The rate of the currency to the currency of rate 1')
    currency_id = fields.Many2one('res.currency', string='Currency', readonly=True)
    company_id = fields.Many2one('res.company', string='Company', default=lambda self: self.env.user._get_company())

    @api.model
    def name_search(self, name, args=None, operator='ilike', limit=80):
        if False:
            return 10
        if operator in ['=', '!=']:
            try:
                date_format = '%Y-%m-%d'
                if self._context.get('lang'):
                    langs = self.env['res.lang'].search([('code', '=', self._context['lang'])])
                    if langs:
                        date_format = langs.date_format
                name = time.strftime('%Y-%m-%d', time.strptime(name, date_format))
            except ValueError:
                try:
                    args.append(('rate', operator, float(name)))
                except ValueError:
                    return []
                name = ''
                operator = 'ilike'
        return super(CurrencyRate, self).name_search(name, args=args, operator=operator, limit=limit)