import logging
from odoo import api, models
_logger = logging.getLogger(__name__)

class AccountChartTemplate(models.Model):
    _inherit = 'account.chart.template'

    @api.multi
    def process_translations(self, langs, in_field, in_ids, out_ids):
        if False:
            i = 10
            return i + 15
        '\n        This method copies translations values of templates into new Accounts/Taxes/Journals for languages selected\n\n        :param langs: List of languages to load for new records\n        :param in_field: Name of the translatable field of source templates\n        :param in_ids: Recordset of ids of source object\n        :param out_ids: Recordset of ids of destination object\n\n        :return: True\n        '
        xlat_obj = self.env['ir.translation']
        for lang in langs:
            value = xlat_obj._get_ids(in_ids._name + ',' + in_field, 'model', lang, in_ids.ids)
            counter = 0
            for element in in_ids.with_context(lang=None):
                if value[element.id]:
                    xlat_obj.create({'name': out_ids._name + ',' + in_field, 'type': 'model', 'res_id': out_ids[counter].id, 'lang': lang, 'src': element.name, 'value': value[element.id]})
                else:
                    _logger.info('Language: %s. Translation from template: there is no translation available for %s!' % (lang, element.name))
                counter += 1
        return True

    @api.multi
    def process_coa_translations(self):
        if False:
            return 10
        installed_langs = dict(self.env['res.lang'].get_installed())
        company_obj = self.env['res.company']
        for chart_template_id in self:
            langs = []
            if chart_template_id.spoken_languages:
                for lang in chart_template_id.spoken_languages.split(';'):
                    if lang not in installed_langs:
                        continue
                    else:
                        langs.append(lang)
                if langs:
                    company_ids = company_obj.search([('chart_template_id', '=', chart_template_id.id)])
                    for company in company_ids:
                        chart_template_id._process_accounts_translations(company.id, langs, 'name')
                        chart_template_id._process_taxes_translations(company.id, langs, 'name')
                        chart_template_id._process_taxes_translations(company.id, langs, 'description')
                        chart_template_id._process_fiscal_pos_translations(company.id, langs, 'name')
        return True

    @api.multi
    def _process_accounts_translations(self, company_id, langs, field):
        if False:
            for i in range(10):
                print('nop')
        in_ids = self.env['account.account.template'].search([('chart_template_id', '=', self.id)], order='id')
        out_ids = self.env['account.account'].search([('company_id', '=', company_id)], order='id')
        return self.process_translations(langs, field, in_ids, out_ids)

    @api.multi
    def _process_taxes_translations(self, company_id, langs, field):
        if False:
            print('Hello World!')
        in_ids = self.env['account.tax.template'].search([('chart_template_id', '=', self.id)], order='id')
        out_ids = self.env['account.tax'].search([('company_id', '=', company_id)], order='id')
        return self.process_translations(langs, field, in_ids, out_ids)

    @api.multi
    def _process_fiscal_pos_translations(self, company_id, langs, field):
        if False:
            i = 10
            return i + 15
        in_ids = self.env['account.fiscal.position.template'].search([('chart_template_id', '=', self.id)], order='id')
        out_ids = self.env['account.fiscal.position'].search([('company_id', '=', company_id)], order='id')
        return self.process_translations(langs, field, in_ids, out_ids)

class BaseLanguageInstall(models.TransientModel):
    """ Install Language"""
    _inherit = 'base.language.install'

    @api.multi
    def lang_install(self):
        if False:
            while True:
                i = 10
        self.ensure_one()
        already_installed = self.env['res.lang'].search_count([('code', '=', self.lang)])
        res = super(BaseLanguageInstall, self).lang_install()
        if already_installed:
            return res
        for coa in self.env['account.chart.template'].search([('spoken_languages', '!=', False)]):
            if self.lang in coa.spoken_languages.split(';'):
                for company in self.env['res.company'].search([('chart_template_id', '=', coa.id)]):
                    coa._process_accounts_translations(company.id, [self.lang], 'name')
                    coa._process_taxes_translations(company.id, [self.lang], 'name')
                    coa._process_taxes_translations(company.id, [self.lang], 'description')
                    coa._process_fiscal_pos_translations(company.id, [self.lang], 'name')
        return res