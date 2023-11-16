import remi.gui as gui
from remi import start, App
from remi_ext import TreeTable, SingleRowSelectionTable

class MyApp(App):

    def __init__(self, *args):
        if False:
            print('Hello World!')
        super(MyApp, self).__init__(*args)

    def main(self):
        if False:
            return 10
        self.wid = gui.VBox(style={'margin': '5px auto', 'padding': '10px'})
        table = [('', '#ff9', 'cable', '1', '2', '3'), ('cable', '#ff9', '1-core cable', '1', '2', '3'), ('cable', '#ff9', 'multi core cable', '1', '2', '3'), ('multi core cable', '#ff9', '2-core cable', '1', '2', '3'), ('multi core cable', '#ff9', '3-core cable', '1', '2', '3'), ('3-core cable', '#ff9', '3-core armoured cable', '1', '2', '3'), ('cable', '#ff9', 'armoured cable', '1', '2', '3'), ('armoured cable', '#ff9', '3-core armoured cable', '1', '2', '3')]
        heads_color = '#dfd'
        uoms_color = '#ffd'
        heads = ['heads', heads_color, 'object', 'one', 'two', 'three']
        uoms = ['uom', uoms_color, '', 'mm', 'cm', 'dm']
        self.My_TreeTable(table, heads, heads2=uoms)
        return self.wid

    def My_TreeTable(self, table, heads, heads2=None):
        if False:
            while True:
                i = 10
        ' Define and display a table\n            in which the values in first column form one or more trees.\n        '
        self.Define_TreeTable(heads, heads2)
        self.Display_TreeTable(table)

    def Define_TreeTable(self, heads, heads2=None):
        if False:
            print('Hello World!')
        ' Define a TreeTable with a heading row\n            and optionally a second heading row.\n        '
        display_heads = []
        display_heads.append(tuple(heads[2:]))
        self.tree_table = TreeTable()
        self.tree_table.append_from_list(display_heads, fill_title=True)
        if heads2 is not None:
            heads2_color = heads2[1]
            row_widget = gui.TableRow()
            for (index, field) in enumerate(heads2[2:]):
                row_item = gui.TableItem(text=field, style={'background-color': heads2_color})
                row_widget.append(row_item, field)
            self.tree_table.append(row_widget, heads2[0])
        self.wid.append(self.tree_table)

    def Display_TreeTable(self, table):
        if False:
            for i in range(10):
                print('nop')
        " Display a table in which the values in first column form one or more trees.\n            The table has row with fields that are strings of identifiers/names.\n            First convert each row into a row_widget and item_widgets\n            that are displayed in a TableTree.\n            Each input row shall start with a parent field (field[0])\n            that determines the tree hierarchy but that is not displayed on that row.\n            The parent widget becomes an attribute of the first child widget.\n            Field[1] is the row color, field[2:] contains the row values.\n            Top child(s) shall have a parent field value that is blank ('').\n            The input table rows shall be in the correct sequence.\n        "
        parent_names = []
        hierarchy = {}
        indent_level = 0
        widget_dict = {}
        for row in table:
            parent_name = row[0]
            row_color = row[1]
            child_name = row[2]
            row_widget = gui.TableRow(style={'background-color': row_color})
            openness = 'true'
            row_widget.attributes['treeopen'] = openness
            for (index, field) in enumerate(row[2:]):
                field_color = '#ffff99'
                row_item = gui.TableItem(text=field, style={'text-align': 'left', 'background-color': field_color})
                row_widget.append(row_item, field)
                if index == 0:
                    row_item.parent = parent_name
                    child_id = row_item
            print('parent-child:', parent_name, child_name)
            if parent_name == '':
                hierarchy[child_name] = 0
                parent_names.append(child_name)
                target_level = 0
            elif parent_name in parent_names:
                hierarchy[child_name] = hierarchy[parent_name] + 1
                target_level = hierarchy[child_name]
            else:
                print('Error: Parent name "{}" does not appear in network'.format(parent_name))
                return
            print('indent, target-pre:', indent_level, target_level, parent_name, child_name)
            if target_level > indent_level:
                self.tree_table.begin_fold()
                indent_level += 1
            if target_level < indent_level:
                while target_level < indent_level:
                    indent_level += -1
                    self.tree_table.end_fold()
            print('indent, target-post:', indent_level, target_level, parent_name, child_name)
            if child_name not in parent_names:
                parent_names.append(child_name)
            self.tree_table.append(row_widget, child_name)
if __name__ == '__main__':
    start(MyApp, address='127.0.0.1', port=8081, multiple_instance=False, enable_file_cache=True, update_interval=0.1, start_browser=True)