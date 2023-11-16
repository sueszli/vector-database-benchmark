import frappe
COLUMNS = [{'label': 'Table', 'fieldname': 'table', 'fieldtype': 'Data', 'width': 200}, {'label': 'Size (MB)', 'fieldname': 'size', 'fieldtype': 'Float'}, {'label': 'Data (MB)', 'fieldname': 'data_size', 'fieldtype': 'Float'}, {'label': 'Index (MB)', 'fieldname': 'index_size', 'fieldtype': 'Float'}]

def execute(filters=None):
    if False:
        print('Hello World!')
    frappe.only_for('System Manager')
    data = frappe.db.multisql({'mariadb': '\n\t\t\t\tSELECT table_name AS `table`,\n\t\t\t\t\t\tround(((data_length + index_length) / 1024 / 1024), 2) `size`,\n\t\t\t\t\t\tround((data_length / 1024 / 1024), 2) as data_size,\n\t\t\t\t\t\tround((index_length / 1024 / 1024), 2) as index_size\n\t\t\t\tFROM information_schema.TABLES\n\t\t\t\tORDER BY (data_length + index_length) DESC;\n\t\t\t', 'postgres': '\n\t\t\t\tSELECT\n\t\t\t\t  table_name as "table",\n\t\t\t\t  round(pg_total_relation_size(quote_ident(table_name)) / 1024 / 1024, 2) as "size",\n\t\t\t\t  round(pg_relation_size(quote_ident(table_name)) / 1024 / 1024, 2) as "data_size",\n\t\t\t\t  round(pg_indexes_size(quote_ident(table_name)) / 1024 / 1024, 2) as "index_size"\n\t\t\t\tFROM information_schema.tables\n\t\t\t\tWHERE table_schema = \'public\'\n\t\t\t\tORDER BY 2 DESC;\n\t\t\t'}, as_dict=1)
    return (COLUMNS, data)