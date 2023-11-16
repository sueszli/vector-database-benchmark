import logging
import re
from odoo import tools, models, fields, api, _
from odoo.exceptions import ValidationError
_logger = logging.getLogger(__name__)
UPC_EAN_CONVERSIONS = [('none', 'Never'), ('ean2upc', 'EAN-13 to UPC-A'), ('upc2ean', 'UPC-A to EAN-13'), ('always', 'Always')]

class BarcodeNomenclature(models.Model):
    _name = 'barcode.nomenclature'
    name = fields.Char(string='Nomenclature Name', size=32, required=True, help='An internal identification of the barcode nomenclature')
    rule_ids = fields.One2many('barcode.rule', 'barcode_nomenclature_id', string='Rules', help='The list of barcode rules')
    upc_ean_conv = fields.Selection(UPC_EAN_CONVERSIONS, string='UPC/EAN Conversion', required=True, default='always', help='UPC Codes can be converted to EAN by prefixing them with a zero. This setting determines if a UPC/EAN barcode should be automatically converted in one way or another when trying to match a rule with the other encoding.')

    def ean_checksum(self, ean):
        if False:
            return 10
        code = list(ean)
        if len(code) != 13:
            return -1
        oddsum = evensum = total = 0
        code = code[:-1]
        for i in range(len(code)):
            if i % 2 == 0:
                evensum += int(code[i])
            else:
                oddsum += int(code[i])
        total = oddsum * 3 + evensum
        return int((10 - total % 10) % 10)

    def ean8_checksum(self, ean):
        if False:
            return 10
        code = list(ean)
        if len(code) != 8:
            return -1
        sum1 = int(ean[1]) + int(ean[3]) + int(ean[5])
        sum2 = int(ean[0]) + int(ean[2]) + int(ean[4]) + int(ean[6])
        total = sum1 + 3 * sum2
        return int((10 - total % 10) % 10)

    def check_ean(self, ean):
        if False:
            i = 10
            return i + 15
        return re.match('^\\d+$', ean) and self.ean_checksum(ean) == int(ean[-1])

    def check_encoding(self, barcode, encoding):
        if False:
            while True:
                i = 10
        if encoding == 'ean13':
            return len(barcode) == 13 and re.match('^\\d+$', barcode) and (self.ean_checksum(barcode) == int(barcode[-1]))
        elif encoding == 'ean8':
            return len(barcode) == 8 and re.match('^\\d+$', barcode) and (self.ean8_checksum(barcode) == int(barcode[-1]))
        elif encoding == 'upca':
            return len(barcode) == 12 and re.match('^\\d+$', barcode) and (self.ean_checksum('0' + barcode) == int(barcode[-1]))
        elif encoding == 'any':
            return True
        else:
            return False

    def sanitize_ean(self, ean):
        if False:
            while True:
                i = 10
        ean = ean[0:13]
        ean = ean + (13 - len(ean)) * '0'
        return ean[0:12] + str(self.ean_checksum(ean))

    def sanitize_upc(self, upc):
        if False:
            for i in range(10):
                print('nop')
        return self.sanitize_ean('0' + upc)[1:]

    def match_pattern(self, barcode, pattern):
        if False:
            return 10
        match = {'value': 0, 'base_code': barcode, 'match': False}
        barcode = barcode.replace('\\', '\\\\').replace('{', '\\{').replace('}', '\\}').replace('.', '\\.')
        numerical_content = re.search('[{][N]*[D]*[}]', pattern)
        if numerical_content:
            num_start = numerical_content.start()
            num_end = numerical_content.end()
            value_string = barcode[num_start:num_end - 2]
            whole_part_match = re.search('[{][N]*[D}]', numerical_content.group())
            decimal_part_match = re.search('[{N][D]*[}]', numerical_content.group())
            whole_part = value_string[:whole_part_match.end() - 2]
            decimal_part = '0.' + value_string[decimal_part_match.start():decimal_part_match.end() - 1]
            if whole_part == '':
                whole_part = '0'
            match['value'] = int(whole_part) + float(decimal_part)
            match['base_code'] = barcode[:num_start] + (num_end - num_start - 2) * '0' + barcode[num_end - 2:]
            match['base_code'] = match['base_code'].replace('\\\\', '\\').replace('\\{', '{').replace('\\}', '}').replace('\\.', '.')
            pattern = pattern[:num_start] + (num_end - num_start - 2) * '0' + pattern[num_end:]
        match['match'] = re.match(pattern, match['base_code'][:len(pattern)])
        return match

    def parse_barcode(self, barcode):
        if False:
            i = 10
            return i + 15
        parsed_result = {'encoding': '', 'type': 'error', 'code': barcode, 'base_code': barcode, 'value': 0}
        rules = []
        for rule in self.rule_ids:
            rules.append({'type': rule.type, 'encoding': rule.encoding, 'sequence': rule.sequence, 'pattern': rule.pattern, 'alias': rule.alias})
        for rule in rules:
            cur_barcode = barcode
            if rule['encoding'] == 'ean13' and self.check_encoding(barcode, 'upca') and (self.upc_ean_conv in ['upc2ean', 'always']):
                cur_barcode = '0' + cur_barcode
            elif rule['encoding'] == 'upca' and self.check_encoding(barcode, 'ean13') and (barcode[0] == '0') and (self.upc_ean_conv in ['ean2upc', 'always']):
                cur_barcode = cur_barcode[1:]
            if not self.check_encoding(barcode, rule['encoding']):
                continue
            match = self.match_pattern(cur_barcode, rule['pattern'])
            if match['match']:
                if rule['type'] == 'alias':
                    barcode = rule['alias']
                    parsed_result['code'] = barcode
                else:
                    parsed_result['encoding'] = rule['encoding']
                    parsed_result['type'] = rule['type']
                    parsed_result['value'] = match['value']
                    parsed_result['code'] = cur_barcode
                    if rule['encoding'] == 'ean13':
                        parsed_result['base_code'] = self.sanitize_ean(match['base_code'])
                    elif rule['encoding'] == 'upca':
                        parsed_result['base_code'] = self.sanitize_upc(match['base_code'])
                    else:
                        parsed_result['base_code'] = match['base_code']
                    return parsed_result
        return parsed_result

class BarcodeRule(models.Model):
    _name = 'barcode.rule'
    _order = 'sequence asc'
    name = fields.Char(string='Rule Name', size=32, required=True, help='An internal identification for this barcode nomenclature rule')
    barcode_nomenclature_id = fields.Many2one('barcode.nomenclature', string='Barcode Nomenclature')
    sequence = fields.Integer(string='Sequence', help='Used to order rules such that rules with a smaller sequence match first')
    encoding = fields.Selection([('any', _('Any')), ('ean13', 'EAN-13'), ('ean8', 'EAN-8'), ('upca', 'UPC-A')], string='Encoding', required=True, default='any', help='This rule will apply only if the barcode is encoded with the specified encoding')
    type = fields.Selection([('alias', _('Alias')), ('product', _('Unit Product'))], string='Type', required=True, default='product')
    pattern = fields.Char(string='Barcode Pattern', size=32, help='The barcode matching pattern', required=True, default='.*')
    alias = fields.Char(string='Alias', size=32, default='0', help='The matched pattern will alias to this barcode', required=True)

    @api.one
    @api.constrains('pattern')
    def _check_pattern(self):
        if False:
            while True:
                i = 10
        p = self.pattern.replace('\\\\', 'X').replace('\\{', 'X').replace('\\}', 'X')
        findall = re.findall('[{]|[}]', p)
        if len(findall) == 2:
            if not re.search('[{][N]*[D]*[}]', p):
                raise ValidationError(_('There is a syntax error in the barcode pattern ') + self.pattern + _(": braces can only contain N's followed by D's."))
            elif re.search('[{][}]', p):
                raise ValidationError(_('There is a syntax error in the barcode pattern ') + self.pattern + _(': empty braces.'))
        elif len(findall) != 0:
            raise ValidationError(_('There is a syntax error in the barcode pattern ') + self.pattern + _(': a rule can only contain one pair of braces.'))
        elif p == '*':
            raise ValidationError(_(" '*' is not a valid Regex Barcode Pattern. Did you mean '.*' ?"))