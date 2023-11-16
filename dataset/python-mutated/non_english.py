import open3d.visualization.gui as gui
import os.path
import platform
basedir = os.path.dirname(os.path.realpath(__file__))
MODE_SERIF = 'serif'
MODE_COMMON_HANYU = 'common'
MODE_SERIF_AND_COMMON_HANYU = 'serif+common'
MODE_COMMON_HANYU_EN = 'hanyu_en+common'
MODE_ALL_HANYU = 'all'
MODE_CUSTOM_CHARS = 'custom'
mode = MODE_SERIF_AND_COMMON_HANYU
if platform.system() == 'Darwin':
    serif = 'Times New Roman'
    hanzi = 'STHeiti Light'
    chess = '/System/Library/Fonts/Apple Symbols.ttf'
elif platform.system() == 'Windows':
    serif = 'c:/windows/fonts/times.ttf'
    hanzi = 'c:/windows/fonts/msyh.ttc'
    chess = 'c:/windows/fonts/seguisym.ttf'
else:
    serif = 'DejaVuSerif'
    hanzi = 'NotoSansCJK'
    chess = '/usr/share/fonts/truetype/freefont/FreeSerif.ttf'

def main():
    if False:
        print('Hello World!')
    gui.Application.instance.initialize()
    font = None
    if mode == MODE_SERIF:
        font = gui.FontDescription(serif)
    elif mode == MODE_COMMON_HANYU:
        font = gui.FontDescription()
        font.add_typeface_for_language(hanzi, 'zh')
    elif mode == MODE_SERIF_AND_COMMON_HANYU:
        font = gui.FontDescription(serif)
        font.add_typeface_for_language(hanzi, 'zh')
    elif mode == MODE_COMMON_HANYU_EN:
        font = gui.FontDescription(hanzi)
        font.add_typeface_for_language(hanzi, 'zh')
    elif mode == MODE_ALL_HANYU:
        font = gui.FontDescription()
        font.add_typeface_for_language(hanzi, 'zh_all')
    elif mode == MODE_CUSTOM_CHARS:
        range = [9812, 9813, 9814, 9815, 9816, 9817]
        font = gui.FontDescription()
        font.add_typeface_for_code_points(chess, range)
    if font is not None:
        gui.Application.instance.set_font(gui.Application.DEFAULT_FONT_ID, font)
    w = ExampleWindow()
    gui.Application.instance.run()

class ExampleWindow:
    MENU_CHECKABLE = 1
    MENU_DISABLED = 2
    MENU_QUIT = 3

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.window = gui.Application.instance.create_window('Test', 400, 768)
        w = self.window
        em = w.theme.font_size
        layout = gui.Vert(0, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))
        if gui.Application.instance.menubar is None:
            menubar = gui.Menu()
            test_menu = gui.Menu()
            test_menu.add_item('An option', ExampleWindow.MENU_CHECKABLE)
            test_menu.set_checked(ExampleWindow.MENU_CHECKABLE, True)
            test_menu.add_item('Unavailable feature', ExampleWindow.MENU_DISABLED)
            test_menu.set_enabled(ExampleWindow.MENU_DISABLED, False)
            test_menu.add_separator()
            test_menu.add_item('Quit', ExampleWindow.MENU_QUIT)
            menubar.add_menu('Test', test_menu)
            gui.Application.instance.menubar = menubar
        w.set_on_menu_item_activated(ExampleWindow.MENU_CHECKABLE, self._on_menu_checkable)
        w.set_on_menu_item_activated(ExampleWindow.MENU_QUIT, self._on_menu_quit)
        self._fileedit = gui.TextEdit()
        filedlgbutton = gui.Button('...')
        filedlgbutton.horizontal_padding_em = 0.5
        filedlgbutton.vertical_padding_em = 0
        filedlgbutton.set_on_clicked(self._on_filedlg_button)
        fileedit_layout = gui.Horiz()
        fileedit_layout.add_child(gui.Label('Model file'))
        fileedit_layout.add_child(self._fileedit)
        fileedit_layout.add_fixed(0.25 * em)
        fileedit_layout.add_child(filedlgbutton)
        layout.add_child(fileedit_layout)
        collapse = gui.CollapsableVert('Widgets', 0.33 * em, gui.Margins(em, 0, 0, 0))
        if mode == MODE_CUSTOM_CHARS:
            self._label = gui.Label('♔♕♖♗♘♙')
        elif mode == MODE_ALL_HANYU:
            self._label = gui.Label('天地玄黃，宇宙洪荒。日月盈昃，辰宿列張。')
        else:
            self._label = gui.Label('锄禾日当午，汗滴禾下土。谁知盘中餐，粒粒皆辛苦。')
        self._label.text_color = gui.Color(1.0, 0.5, 0.0)
        collapse.add_child(self._label)
        cb = gui.Checkbox('Enable some really cool effect')
        cb.set_on_checked(self._on_cb)
        collapse.add_child(cb)
        color = gui.ColorEdit()
        color.color_value = self._label.text_color
        color.set_on_value_changed(self._on_color)
        collapse.add_child(color)
        combo = gui.Combobox()
        combo.add_item('Show point labels')
        combo.add_item('Show point velocity')
        combo.add_item('Show bounding boxes')
        combo.set_on_selection_changed(self._on_combo)
        collapse.add_child(combo)
        logo = gui.ImageWidget(basedir + '/icon-32.png')
        collapse.add_child(logo)
        lv = gui.ListView()
        lv.set_items(['Ground', 'Trees', 'BuildingsCars', 'People'])
        lv.selected_index = lv.selected_index + 2
        lv.set_on_selection_changed(self._on_list)
        collapse.add_child(lv)
        tree = gui.TreeView()
        tree.add_text_item(tree.get_root_item(), 'Camera')
        geo_id = tree.add_text_item(tree.get_root_item(), 'Geometries')
        mesh_id = tree.add_text_item(geo_id, 'Mesh')
        tree.add_text_item(mesh_id, 'Triangles')
        tree.add_text_item(mesh_id, 'Albedo texture')
        tree.add_text_item(mesh_id, 'Normal map')
        points_id = tree.add_text_item(geo_id, 'Points')
        tree.can_select_items_with_children = True
        tree.set_on_selection_changed(self._on_tree)
        tree.selected_item = points_id
        collapse.add_child(tree)
        intedit = gui.NumberEdit(gui.NumberEdit.INT)
        intedit.int_value = 0
        intedit.set_limits(1, 19)
        intedit.int_value = intedit.int_value + 2
        doubleedit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        numlayout = gui.Horiz()
        numlayout.add_child(gui.Label('int'))
        numlayout.add_child(intedit)
        numlayout.add_fixed(em)
        numlayout.add_child(gui.Label('double'))
        numlayout.add_child(doubleedit)
        collapse.add_child(numlayout)
        self._progress = gui.ProgressBar()
        self._progress.value = 0.25
        self._progress.value = self._progress.value + 0.08
        prog_layout = gui.Horiz(em)
        prog_layout.add_child(gui.Label('Progress...'))
        prog_layout.add_child(self._progress)
        collapse.add_child(prog_layout)
        slider = gui.Slider(gui.Slider.INT)
        slider.set_limits(5, 13)
        slider.set_on_value_changed(self._on_slider)
        collapse.add_child(slider)
        tedit = gui.TextEdit()
        tedit.placeholder_text = 'Edit me some text here'
        tedit.set_on_text_changed(self._on_text_changed)
        tedit.set_on_value_changed(self._on_value_changed)
        collapse.add_child(tedit)
        vedit = gui.VectorEdit()
        vedit.vector_value = [1, 2, 3]
        vedit.set_on_value_changed(self._on_vedit)
        collapse.add_child(vedit)
        vgrid = gui.VGrid(2)
        vgrid.add_child(gui.Label('Trees'))
        vgrid.add_child(gui.Label('12 items'))
        vgrid.add_child(gui.Label('People'))
        vgrid.add_child(gui.Label('2 (93% certainty)'))
        vgrid.add_child(gui.Label('Cars'))
        vgrid.add_child(gui.Label('5 (87% certainty)'))
        collapse.add_child(vgrid)
        tabs = gui.TabControl()
        tab1 = gui.Vert()
        tab1.add_child(gui.Checkbox('Enable option 1'))
        tab1.add_child(gui.Checkbox('Enable option 2'))
        tab1.add_child(gui.Checkbox('Enable option 3'))
        tabs.add_tab('Options', tab1)
        tab2 = gui.Vert()
        tab2.add_child(gui.Label('No plugins detected'))
        tab2.add_stretch()
        tabs.add_tab('Plugins', tab2)
        collapse.add_child(tabs)
        button_layout = gui.Horiz()
        ok_button = gui.Button('Ok')
        ok_button.set_on_clicked(self._on_ok)
        button_layout.add_stretch()
        button_layout.add_child(ok_button)
        layout.add_child(collapse)
        layout.add_child(button_layout)
        w.add_child(layout)

    def _on_filedlg_button(self):
        if False:
            print('Hello World!')
        filedlg = gui.FileDialog(gui.FileDialog.OPEN, 'Select file', self.window.theme)
        filedlg.add_filter('.obj .ply .stl', 'Triangle mesh (.obj, .ply, .stl)')
        filedlg.add_filter('', 'All files')
        filedlg.set_on_cancel(self._on_filedlg_cancel)
        filedlg.set_on_done(self._on_filedlg_done)
        self.window.show_dialog(filedlg)

    def _on_filedlg_cancel(self):
        if False:
            while True:
                i = 10
        self.window.close_dialog()

    def _on_filedlg_done(self, path):
        if False:
            return 10
        self._fileedit.text_value = path
        self.window.close_dialog()

    def _on_cb(self, is_checked):
        if False:
            i = 10
            return i + 15
        if is_checked:
            text = 'Sorry, effects are unimplemented'
        else:
            text = 'Good choice'
        self.show_message_dialog('There might be a problem...', text)

    def show_message_dialog(self, title, message):
        if False:
            return 10
        dlg = gui.Dialog(title)
        em = self.window.theme.font_size
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label(message))
        ok_button = gui.Button('Ok')
        ok_button.set_on_clicked(self._on_dialog_ok)
        button_layout = gui.Horiz()
        button_layout.add_stretch()
        button_layout.add_child(ok_button)
        dlg_layout.add_child(button_layout)
        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)

    def _on_dialog_ok(self):
        if False:
            print('Hello World!')
        self.window.close_dialog()

    def _on_color(self, new_color):
        if False:
            print('Hello World!')
        self._label.text_color = new_color

    def _on_combo(self, new_val, new_idx):
        if False:
            for i in range(10):
                print('nop')
        print(new_idx, new_val)

    def _on_list(self, new_val, is_dbl_click):
        if False:
            for i in range(10):
                print('nop')
        print(new_val)

    def _on_tree(self, new_item_id):
        if False:
            return 10
        print(new_item_id)

    def _on_slider(self, new_val):
        if False:
            for i in range(10):
                print('nop')
        self._progress.value = new_val / 20.0

    def _on_text_changed(self, new_text):
        if False:
            print('Hello World!')
        print('edit:', new_text)

    def _on_value_changed(self, new_text):
        if False:
            return 10
        print('value:', new_text)

    def _on_vedit(self, new_val):
        if False:
            return 10
        print(new_val)

    def _on_ok(self):
        if False:
            i = 10
            return i + 15
        gui.Application.instance.quit()

    def _on_menu_checkable(self):
        if False:
            return 10
        gui.Application.instance.menubar.set_checked(ExampleWindow.MENU_CHECKABLE, not gui.Application.instance.menubar.is_checked(ExampleWindow.MENU_CHECKABLE))

    def _on_menu_quit(self):
        if False:
            print('Hello World!')
        gui.Application.instance.quit()

class MessageBox:

    def __init__(self, title, message):
        if False:
            for i in range(10):
                print('nop')
        self._window = None
        dlg = gui.Dialog(title)
        em = self.window.theme.font_size
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label(message))
        ok_button = gui.Button('Ok')
        ok_button.set_on_clicked(self._on_ok)
        button_layout = gui.Horiz()
        button_layout.add_stretch()
        button_layout.add_child(ok_button)
        dlg_layout.add_child(button_layout)
        dlg.add_child(dlg_layout)

    def show(self, window):
        if False:
            print('Hello World!')
        self._window = window

    def _on_ok(self):
        if False:
            for i in range(10):
                print('nop')
        self._window.close_dialog()
if __name__ == '__main__':
    main()