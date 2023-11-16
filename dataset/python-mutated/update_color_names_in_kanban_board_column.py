import frappe

def execute():
    if False:
        i = 10
        return i + 15
    indicator_map = {'blue': 'Blue', 'orange': 'Orange', 'red': 'Red', 'green': 'Green', 'darkgrey': 'Gray', 'gray': 'Gray', 'purple': 'Purple', 'yellow': 'Yellow', 'lightblue': 'Light Blue'}
    for d in frappe.get_all('Kanban Board Column', fields=['name', 'indicator']):
        color_name = indicator_map.get(d.indicator, 'Gray')
        frappe.db.set_value('Kanban Board Column', d.name, 'indicator', color_name)