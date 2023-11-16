import odoo

def migrate(cr, version):
    if False:
        print('Hello World!')
    registry = odoo.registry(cr.dbname)
    from odoo.addons.account.models.chart_template import migrate_tags_on_taxes
    migrate_tags_on_taxes(cr, registry)