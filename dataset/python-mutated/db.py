import odoo.modules
import logging
_logger = logging.getLogger(__name__)

def is_initialized(cr):
    if False:
        for i in range(10):
            print('nop')
    " Check if a database has been initialized for the ORM.\n\n    The database can be initialized with the 'initialize' function below.\n\n    "
    cr.execute("SELECT relname FROM pg_class WHERE relkind='r' AND relname='ir_module_module'")
    return len(cr.fetchall()) > 0

def initialize(cr):
    if False:
        print('Hello World!')
    ' Initialize a database with for the ORM.\n\n    This executes base/base.sql, creates the ir_module_categories (taken\n    from each module descriptor file), and creates the ir_module_module\n    and ir_model_data entries.\n\n    '
    f = odoo.modules.get_module_resource('base', 'base.sql')
    if not f:
        m = "File not found: 'base.sql' (provided by module 'base')."
        _logger.critical(m)
        raise IOError(m)
    base_sql_file = odoo.tools.misc.file_open(f)
    try:
        cr.execute(base_sql_file.read())
        cr.commit()
    finally:
        base_sql_file.close()
    for i in odoo.modules.get_modules():
        mod_path = odoo.modules.get_module_path(i)
        if not mod_path:
            continue
        info = odoo.modules.load_information_from_description_file(i)
        if not info:
            continue
        categories = info['category'].split('/')
        category_id = create_categories(cr, categories)
        if info['installable']:
            state = 'uninstalled'
        else:
            state = 'uninstallable'
        cr.execute('INSERT INTO ir_module_module                 (author, website, name, shortdesc, description,                     category_id, auto_install, state, web, license, application, icon, sequence, summary)                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id', (info['author'], info['website'], i, info['name'], info['description'], category_id, info['auto_install'], state, info['web'], info['license'], info['application'], info['icon'], info['sequence'], info['summary']))
        id = cr.fetchone()[0]
        cr.execute('INSERT INTO ir_model_data             (name,model,module, res_id, noupdate) VALUES (%s,%s,%s,%s,%s)', ('module_' + i, 'ir.module.module', 'base', id, True))
        dependencies = info['depends']
        for d in dependencies:
            cr.execute('INSERT INTO ir_module_module_dependency                     (module_id,name) VALUES (%s, %s)', (id, d))
    while True:
        cr.execute("SELECT m.name FROM ir_module_module m WHERE m.auto_install AND state != 'to install'\n                      AND NOT EXISTS (\n                          SELECT 1 FROM ir_module_module_dependency d JOIN ir_module_module mdep ON (d.name = mdep.name)\n                                   WHERE d.module_id = m.id AND mdep.state != 'to install'\n                      )")
        to_auto_install = [x[0] for x in cr.fetchall()]
        if not to_auto_install:
            break
        cr.execute("UPDATE ir_module_module SET state='to install' WHERE name in %s", (tuple(to_auto_install),))
    cr.commit()

def create_categories(cr, categories):
    if False:
        return 10
    " Create the ir_module_category entries for some categories.\n\n    categories is a list of strings forming a single category with its\n    parent categories, like ['Grand Parent', 'Parent', 'Child'].\n\n    Return the database id of the (last) category.\n\n    "
    p_id = None
    category = []
    while categories:
        category.append(categories[0])
        xml_id = 'module_category_' + '_'.join(map(lambda x: x.lower(), category)).replace('&', 'and').replace(' ', '_')
        cr.execute('SELECT res_id FROM ir_model_data WHERE name=%s AND module=%s AND model=%s', (xml_id, 'base', 'ir.module.category'))
        c_id = cr.fetchone()
        if not c_id:
            cr.execute('INSERT INTO ir_module_category                     (name, parent_id)                     VALUES (%s, %s) RETURNING id', (categories[0], p_id))
            c_id = cr.fetchone()[0]
            cr.execute('INSERT INTO ir_model_data (module, name, res_id, model)                        VALUES (%s, %s, %s, %s)', ('base', xml_id, c_id, 'ir.module.category'))
        else:
            c_id = c_id[0]
        p_id = c_id
        categories = categories[1:]
    return p_id

def has_unaccent(cr):
    if False:
        while True:
            i = 10
    ' Test if the database has an unaccent function.\n\n    The unaccent is supposed to be provided by the PostgreSQL unaccent contrib\n    module but any similar function will be picked by OpenERP.\n\n    '
    cr.execute("SELECT proname FROM pg_proc WHERE proname='unaccent'")
    return len(cr.fetchall()) > 0