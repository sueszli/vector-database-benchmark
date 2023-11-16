import frappe

def execute():
    if False:
        i = 10
        return i + 15
    if frappe.db.db_type != 'mariadb':
        return
    all_tables = frappe.db.get_tables()
    final_deletion_map = frappe._dict()
    for table in all_tables:
        indexes_to_keep_map = frappe._dict()
        indexes_to_delete = []
        index_info = frappe.db.sql(f'SHOW INDEX FROM `{table}`\n\t\t\t\tWHERE Seq_in_index = 1\n\t\t\t\tAND Non_unique=0', as_dict=1)
        for index in index_info:
            if not indexes_to_keep_map.get(index.Column_name):
                indexes_to_keep_map[index.Column_name] = index
            else:
                indexes_to_delete.append(index.Key_name)
        if indexes_to_delete:
            final_deletion_map[table] = indexes_to_delete
    for (table_name, index_list) in final_deletion_map.items():
        for index in index_list:
            try:
                if is_clustered_index(table_name, index):
                    continue
                frappe.db.sql_ddl(f'ALTER TABLE `{table_name}` DROP INDEX `{index}`')
            except Exception as e:
                frappe.log_error('Failed to drop index')
                print(f'x Failed to drop index {index} from {table_name}\n {str(e)}')
            else:
                print(f'✓ dropped {index} index from {table}')

def is_clustered_index(table, index_name):
    if False:
        i = 10
        return i + 15
    return bool(frappe.db.sql(f'SHOW INDEX FROM `{table}`\n\t\t\tWHERE Key_name = "{index_name}"\n\t\t\t\tAND Seq_in_index = 2\n\t\t\t', as_dict=True))