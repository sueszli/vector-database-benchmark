def convert_field(cr, model, field, target_model):
    if False:
        for i in range(10):
            print('nop')
    table = model.replace('.', '_')
    cr.execute('SELECT 1\n                    FROM information_schema.columns\n                   WHERE table_name = %s\n                     AND column_name = %s\n               ', (table, field))
    if not cr.fetchone():
        return
    cr.execute('SELECT id FROM ir_model_fields WHERE model=%s AND name=%s', (model, field))
    [fields_id] = cr.fetchone()
    cr.execute("\n        INSERT INTO ir_property(name, type, fields_id, company_id, res_id, value_reference)\n        SELECT %(field)s, 'many2one', %(fields_id)s, company_id, CONCAT('{model},', id),\n               CONCAT('{target_model},', {field})\n          FROM {table} t\n         WHERE {field} IS NOT NULL\n           AND NOT EXISTS(SELECT 1\n                            FROM ir_property\n                           WHERE fields_id=%(fields_id)s\n                             AND company_id=t.company_id\n                             AND res_id=CONCAT('{model},', t.id))\n    ".format(**locals()), locals())
    cr.execute('ALTER TABLE "{0}" DROP COLUMN "{1}" CASCADE'.format(table, field))

def migrate(cr, version):
    if False:
        return 10
    convert_field(cr, 'res.partner', 'property_purchase_currency_id', 'res.currency')
    convert_field(cr, 'product.template', 'property_account_creditor_price_difference', 'account.account')