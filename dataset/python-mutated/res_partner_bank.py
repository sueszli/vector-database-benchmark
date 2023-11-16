import re
from odoo import api, models, _
from odoo.exceptions import UserError, ValidationError

def normalize_iban(iban):
    if False:
        while True:
            i = 10
    return re.sub('[\\W_]', '', iban or '')

def pretty_iban(iban):
    if False:
        i = 10
        return i + 15
    ' return iban in groups of four characters separated by a single space '
    return ' '.join([iban[i:i + 4] for i in range(0, len(iban), 4)])

def get_bban_from_iban(iban):
    if False:
        i = 10
        return i + 15
    ' Returns the basic bank account number corresponding to an IBAN.\n        Note : the BBAN is not the same as the domestic bank account number !\n        The relation between IBAN, BBAN and domestic can be found here : http://www.ecbs.org/iban.htm\n    '
    return normalize_iban(iban)[4:]

def validate_iban(iban):
    if False:
        for i in range(10):
            print('nop')
    iban = normalize_iban(iban)
    if not iban:
        raise ValidationError(_('No IBAN !'))
    country_code = iban[:2].lower()
    if country_code not in _map_iban_template:
        raise ValidationError(_('The IBAN is invalid, it should begin with the country code'))
    iban_template = _map_iban_template[country_code]
    if len(iban) != len(iban_template.replace(' ', '')):
        raise ValidationError(_('The IBAN does not seem to be correct. You should have entered something like this %s\nWhere B = National bank code, S = Branch code, C = Account No, k = Check digit') % iban_template)
    check_chars = iban[4:] + iban[:4]
    digits = int(''.join((str(int(char, 36)) for char in check_chars)))
    if digits % 97 != 1:
        raise ValidationError(_('This IBAN does not pass the validation check, please verify it.'))

class ResPartnerBank(models.Model):
    _inherit = 'res.partner.bank'

    @api.one
    @api.depends('acc_number')
    def _compute_acc_type(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            validate_iban(self.acc_number)
            self.acc_type = 'iban'
        except ValidationError:
            super(ResPartnerBank, self)._compute_acc_type()

    def get_bban(self):
        if False:
            i = 10
            return i + 15
        if self.acc_type != 'iban':
            raise UserError(_('Cannot compute the BBAN because the account number is not an IBAN.'))
        return get_bban_from_iban(self.acc_number)

    @api.model
    def create(self, vals):
        if False:
            while True:
                i = 10
        if vals.get('acc_type') == 'iban' and vals.get('acc_number'):
            vals['acc_number'] = pretty_iban(normalize_iban(vals['acc_number']))
        return super(ResPartnerBank, self).create(vals)

    @api.multi
    def write(self, vals):
        if False:
            return 10
        if vals.get('acc_type') == 'iban' and vals.get('acc_number'):
            vals['acc_number'] = pretty_iban(normalize_iban(vals['acc_number']))
        return super(ResPartnerBank, self).write(vals)

    @api.one
    @api.constrains('acc_number')
    def _check_iban(self):
        if False:
            while True:
                i = 10
        if self.acc_type == 'iban':
            validate_iban(self.acc_number)
_map_iban_template = {'ad': 'ADkk BBBB SSSS CCCC CCCC CCCC', 'ae': 'AEkk BBBC CCCC CCCC CCCC CCC', 'al': 'ALkk BBBS SSSK CCCC CCCC CCCC CCCC', 'at': 'ATkk BBBB BCCC CCCC CCCC', 'az': 'AZkk BBBB CCCC CCCC CCCC CCCC CCCC', 'ba': 'BAkk BBBS SSCC CCCC CCKK', 'be': 'BEkk BBBC CCCC CCXX', 'bg': 'BGkk BBBB SSSS DDCC CCCC CC', 'bh': 'BHkk BBBB CCCC CCCC CCCC CC', 'br': 'BRkk BBBB BBBB SSSS SCCC CCCC CCCT N', 'ch': 'CHkk BBBB BCCC CCCC CCCC C', 'cr': 'CRkk BBBC CCCC CCCC CCCC C', 'cy': 'CYkk BBBS SSSS CCCC CCCC CCCC CCCC', 'cz': 'CZkk BBBB SSSS SSCC CCCC CCCC', 'de': 'DEkk BBBB BBBB CCCC CCCC CC', 'dk': 'DKkk BBBB CCCC CCCC CC', 'do': 'DOkk BBBB CCCC CCCC CCCC CCCC CCCC', 'ee': 'EEkk BBSS CCCC CCCC CCCK', 'es': 'ESkk BBBB SSSS KKCC CCCC CCCC', 'fi': 'FIkk BBBB BBCC CCCC CK', 'fo': 'FOkk CCCC CCCC CCCC CC', 'fr': 'FRkk BBBB BGGG GGCC CCCC CCCC CKK', 'gb': 'GBkk BBBB SSSS SSCC CCCC CC', 'ge': 'GEkk BBCC CCCC CCCC CCCC CC', 'gi': 'GIkk BBBB CCCC CCCC CCCC CCC', 'gl': 'GLkk BBBB CCCC CCCC CC', 'gr': 'GRkk BBBS SSSC CCCC CCCC CCCC CCC', 'gt': 'GTkk BBBB MMTT CCCC CCCC CCCC CCCC', 'hr': 'HRkk BBBB BBBC CCCC CCCC C', 'hu': 'HUkk BBBS SSSC CCCC CCCC CCCC CCCC', 'ie': 'IEkk BBBB SSSS SSCC CCCC CC', 'il': 'ILkk BBBS SSCC CCCC CCCC CCC', 'is': 'ISkk BBBB SSCC CCCC XXXX XXXX XX', 'it': 'ITkk KBBB BBSS SSSC CCCC CCCC CCC', 'jo': 'JOkk BBBB NNNN CCCC CCCC CCCC CCCC CC', 'kw': 'KWkk BBBB CCCC CCCC CCCC CCCC CCCC CC', 'kz': 'KZkk BBBC CCCC CCCC CCCC', 'lb': 'LBkk BBBB CCCC CCCC CCCC CCCC CCCC', 'li': 'LIkk BBBB BCCC CCCC CCCC C', 'lt': 'LTkk BBBB BCCC CCCC CCCC', 'lu': 'LUkk BBBC CCCC CCCC CCCC', 'lv': 'LVkk BBBB CCCC CCCC CCCC C', 'mc': 'MCkk BBBB BGGG GGCC CCCC CCCC CKK', 'md': 'MDkk BBCC CCCC CCCC CCCC CCCC', 'me': 'MEkk BBBC CCCC CCCC CCCC KK', 'mk': 'MKkk BBBC CCCC CCCC CKK', 'mr': 'MRkk BBBB BSSS SSCC CCCC CCCC CKK', 'mt': 'MTkk BBBB SSSS SCCC CCCC CCCC CCCC CCC', 'mu': 'MUkk BBBB BBSS CCCC CCCC CCCC CCCC CC', 'nl': 'NLkk BBBB CCCC CCCC CC', 'no': 'NOkk BBBB CCCC CCK', 'pk': 'PKkk BBBB CCCC CCCC CCCC CCCC', 'pl': 'PLkk BBBS SSSK CCCC CCCC CCCC CCCC', 'ps': 'PSkk BBBB XXXX XXXX XCCC CCCC CCCC C', 'pt': 'PTkk BBBB SSSS CCCC CCCC CCCK K', 'qa': 'QAkk BBBB CCCC CCCC CCCC CCCC CCCC C', 'ro': 'ROkk BBBB CCCC CCCC CCCC CCCC', 'rs': 'RSkk BBBC CCCC CCCC CCCC KK', 'sa': 'SAkk BBCC CCCC CCCC CCCC CCCC', 'se': 'SEkk BBBB CCCC CCCC CCCC CCCC', 'si': 'SIkk BBSS SCCC CCCC CKK', 'sk': 'SKkk BBBB SSSS SSCC CCCC CCCC', 'sm': 'SMkk KBBB BBSS SSSC CCCC CCCC CCC', 'tn': 'TNkk BBSS SCCC CCCC CCCC CCCC', 'tr': 'TRkk BBBB BRCC CCCC CCCC CCCC CC', 'vg': 'VGkk BBBB CCCC CCCC CCCC CCCC', 'xk': 'XKkk BBBB CCCC CCCC CCCC'}