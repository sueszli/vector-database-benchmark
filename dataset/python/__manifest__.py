# -*- coding: utf-8 -*-
# @COPYRIGHT_begin
#
# Copyright [2016] Michał Szczygieł (m4gik), M4GiK Software
#
# @COPYRIGHT_end
{
    'name': "KamsERP module for order management",
    'summary': 'System to manage orders from KQS Store',
    'description': """
        TODO
    """,

    'author': "Michał Szczygieł",
    'website': "http://www.kams.com.pl",

    # Categories can be used to filter modules in modules listing
    # Check https://github.com/odoo/odoo/blob/master/openerp/addons/base/module/module_data.xml
    # for the full list
    'category': 'Sales Management',
    'version': '0.1',

    # any module necessary for this one to work correctly
    'depends': ['portal_sale', 'web_kanban_gauge', 'account_accountant', 'stock'],

    'qweb': [
        'static/src/xml/widget.xml',
    ],

    # always loaded
    'data': [
        'security/ir.model.access.csv',
        'security/kams_erp_security.xml',
        'views/kams_erp_order_view.xml',
        'views/kams_erp_product_view.xml',
        'views/kams_erp_category_view.xml',
        'views/kams_erp_manufacturer_view.xml',
        'views/kams_erp_stock_quant_view.xml',
        'views/kams_erp_menu_view.xml',
        'views/kams_erp_sale_data.xml',
        'views/kams_erp_partner_view.xml',
        'views/qweb.xml',
    ],

    # only loaded in demonstration mode
    'demo': [],
    'installable': True,
    'auto_install': False,
    'application': True,
}