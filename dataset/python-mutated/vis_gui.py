import glob
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import os
import platform
import sys
isMacOS = platform.system() == 'Darwin'

class Settings:
    UNLIT = 'defaultUnlit'
    LIT = 'defaultLit'
    NORMALS = 'normals'
    DEPTH = 'depth'
    DEFAULT_PROFILE_NAME = 'Bright day with sun at +Y [default]'
    POINT_CLOUD_PROFILE_NAME = 'Cloudy day (no direct sun)'
    CUSTOM_PROFILE_NAME = 'Custom'
    LIGHTING_PROFILES = {DEFAULT_PROFILE_NAME: {'ibl_intensity': 45000, 'sun_intensity': 45000, 'sun_dir': [0.577, -0.577, -0.577], 'use_ibl': True, 'use_sun': True}, 'Bright day with sun at -Y': {'ibl_intensity': 45000, 'sun_intensity': 45000, 'sun_dir': [0.577, 0.577, 0.577], 'use_ibl': True, 'use_sun': True}, 'Bright day with sun at +Z': {'ibl_intensity': 45000, 'sun_intensity': 45000, 'sun_dir': [0.577, 0.577, -0.577], 'use_ibl': True, 'use_sun': True}, 'Less Bright day with sun at +Y': {'ibl_intensity': 35000, 'sun_intensity': 50000, 'sun_dir': [0.577, -0.577, -0.577], 'use_ibl': True, 'use_sun': True}, 'Less Bright day with sun at -Y': {'ibl_intensity': 35000, 'sun_intensity': 50000, 'sun_dir': [0.577, 0.577, 0.577], 'use_ibl': True, 'use_sun': True}, 'Less Bright day with sun at +Z': {'ibl_intensity': 35000, 'sun_intensity': 50000, 'sun_dir': [0.577, 0.577, -0.577], 'use_ibl': True, 'use_sun': True}, POINT_CLOUD_PROFILE_NAME: {'ibl_intensity': 60000, 'sun_intensity': 50000, 'use_ibl': True, 'use_sun': False}}
    DEFAULT_MATERIAL_NAME = 'Polished ceramic [default]'
    PREFAB = {DEFAULT_MATERIAL_NAME: {'metallic': 0.0, 'roughness': 0.7, 'reflectance': 0.5, 'clearcoat': 0.2, 'clearcoat_roughness': 0.2, 'anisotropy': 0.0}, 'Metal (rougher)': {'metallic': 1.0, 'roughness': 0.5, 'reflectance': 0.9, 'clearcoat': 0.0, 'clearcoat_roughness': 0.0, 'anisotropy': 0.0}, 'Metal (smoother)': {'metallic': 1.0, 'roughness': 0.3, 'reflectance': 0.9, 'clearcoat': 0.0, 'clearcoat_roughness': 0.0, 'anisotropy': 0.0}, 'Plastic': {'metallic': 0.0, 'roughness': 0.5, 'reflectance': 0.5, 'clearcoat': 0.5, 'clearcoat_roughness': 0.2, 'anisotropy': 0.0}, 'Glazed ceramic': {'metallic': 0.0, 'roughness': 0.5, 'reflectance': 0.9, 'clearcoat': 1.0, 'clearcoat_roughness': 0.1, 'anisotropy': 0.0}, 'Clay': {'metallic': 0.0, 'roughness': 1.0, 'reflectance': 0.5, 'clearcoat': 0.1, 'clearcoat_roughness': 0.287, 'anisotropy': 0.0}}

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.mouse_model = gui.SceneWidget.Controls.ROTATE_CAMERA
        self.bg_color = gui.Color(1, 1, 1)
        self.show_skybox = False
        self.show_axes = False
        self.use_ibl = True
        self.use_sun = True
        self.new_ibl_name = None
        self.ibl_intensity = 45000
        self.sun_intensity = 45000
        self.sun_dir = [0.577, -0.577, -0.577]
        self.sun_color = gui.Color(1, 1, 1)
        self.apply_material = True
        self._materials = {Settings.LIT: rendering.MaterialRecord(), Settings.UNLIT: rendering.MaterialRecord(), Settings.NORMALS: rendering.MaterialRecord(), Settings.DEPTH: rendering.MaterialRecord()}
        self._materials[Settings.LIT].base_color = [0.9, 0.9, 0.9, 1.0]
        self._materials[Settings.LIT].shader = Settings.LIT
        self._materials[Settings.UNLIT].base_color = [0.9, 0.9, 0.9, 1.0]
        self._materials[Settings.UNLIT].shader = Settings.UNLIT
        self._materials[Settings.NORMALS].shader = Settings.NORMALS
        self._materials[Settings.DEPTH].shader = Settings.DEPTH
        self.material = self._materials[Settings.LIT]

    def set_material(self, name):
        if False:
            while True:
                i = 10
        self.material = self._materials[name]
        self.apply_material = True

    def apply_material_prefab(self, name):
        if False:
            i = 10
            return i + 15
        assert self.material.shader == Settings.LIT
        prefab = Settings.PREFAB[name]
        for (key, val) in prefab.items():
            setattr(self.material, 'base_' + key, val)

    def apply_lighting_profile(self, name):
        if False:
            print('Hello World!')
        profile = Settings.LIGHTING_PROFILES[name]
        for (key, val) in profile.items():
            setattr(self, key, val)

class AppWindow:
    MENU_OPEN = 1
    MENU_EXPORT = 2
    MENU_QUIT = 3
    MENU_SHOW_SETTINGS = 11
    MENU_ABOUT = 21
    DEFAULT_IBL = 'default'
    MATERIAL_NAMES = ['Lit', 'Unlit', 'Normals', 'Depth']
    MATERIAL_SHADERS = [Settings.LIT, Settings.UNLIT, Settings.NORMALS, Settings.DEPTH]

    def __init__(self, width, height):
        if False:
            for i in range(10):
                print('nop')
        self.settings = Settings()
        resource_path = gui.Application.instance.resource_path
        self.settings.new_ibl_name = resource_path + '/' + AppWindow.DEFAULT_IBL
        self.window = gui.Application.instance.create_window('Open3D', width, height)
        w = self.window
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(w.renderer)
        self._scene.set_on_sun_direction_changed(self._on_sun_dir)
        em = w.theme.font_size
        separation_height = int(round(0.5 * em))
        self._settings_panel = gui.Vert(0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        view_ctrls = gui.CollapsableVert('View controls', 0.25 * em, gui.Margins(em, 0, 0, 0))
        self._arcball_button = gui.Button('Arcball')
        self._arcball_button.horizontal_padding_em = 0.5
        self._arcball_button.vertical_padding_em = 0
        self._arcball_button.set_on_clicked(self._set_mouse_mode_rotate)
        self._fly_button = gui.Button('Fly')
        self._fly_button.horizontal_padding_em = 0.5
        self._fly_button.vertical_padding_em = 0
        self._fly_button.set_on_clicked(self._set_mouse_mode_fly)
        self._model_button = gui.Button('Model')
        self._model_button.horizontal_padding_em = 0.5
        self._model_button.vertical_padding_em = 0
        self._model_button.set_on_clicked(self._set_mouse_mode_model)
        self._sun_button = gui.Button('Sun')
        self._sun_button.horizontal_padding_em = 0.5
        self._sun_button.vertical_padding_em = 0
        self._sun_button.set_on_clicked(self._set_mouse_mode_sun)
        self._ibl_button = gui.Button('Environment')
        self._ibl_button.horizontal_padding_em = 0.5
        self._ibl_button.vertical_padding_em = 0
        self._ibl_button.set_on_clicked(self._set_mouse_mode_ibl)
        view_ctrls.add_child(gui.Label('Mouse controls'))
        h = gui.Horiz(0.25 * em)
        h.add_stretch()
        h.add_child(self._arcball_button)
        h.add_child(self._fly_button)
        h.add_child(self._model_button)
        h.add_stretch()
        view_ctrls.add_child(h)
        h = gui.Horiz(0.25 * em)
        h.add_stretch()
        h.add_child(self._sun_button)
        h.add_child(self._ibl_button)
        h.add_stretch()
        view_ctrls.add_child(h)
        self._show_skybox = gui.Checkbox('Show skymap')
        self._show_skybox.set_on_checked(self._on_show_skybox)
        view_ctrls.add_fixed(separation_height)
        view_ctrls.add_child(self._show_skybox)
        self._bg_color = gui.ColorEdit()
        self._bg_color.set_on_value_changed(self._on_bg_color)
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label('BG Color'))
        grid.add_child(self._bg_color)
        view_ctrls.add_child(grid)
        self._show_axes = gui.Checkbox('Show axes')
        self._show_axes.set_on_checked(self._on_show_axes)
        view_ctrls.add_fixed(separation_height)
        view_ctrls.add_child(self._show_axes)
        self._profiles = gui.Combobox()
        for name in sorted(Settings.LIGHTING_PROFILES.keys()):
            self._profiles.add_item(name)
        self._profiles.add_item(Settings.CUSTOM_PROFILE_NAME)
        self._profiles.set_on_selection_changed(self._on_lighting_profile)
        view_ctrls.add_fixed(separation_height)
        view_ctrls.add_child(gui.Label('Lighting profiles'))
        view_ctrls.add_child(self._profiles)
        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(view_ctrls)
        advanced = gui.CollapsableVert('Advanced lighting', 0, gui.Margins(em, 0, 0, 0))
        advanced.set_is_open(False)
        self._use_ibl = gui.Checkbox('HDR map')
        self._use_ibl.set_on_checked(self._on_use_ibl)
        self._use_sun = gui.Checkbox('Sun')
        self._use_sun.set_on_checked(self._on_use_sun)
        advanced.add_child(gui.Label('Light sources'))
        h = gui.Horiz(em)
        h.add_child(self._use_ibl)
        h.add_child(self._use_sun)
        advanced.add_child(h)
        self._ibl_map = gui.Combobox()
        for ibl in glob.glob(gui.Application.instance.resource_path + '/*_ibl.ktx'):
            self._ibl_map.add_item(os.path.basename(ibl[:-8]))
        self._ibl_map.selected_text = AppWindow.DEFAULT_IBL
        self._ibl_map.set_on_selection_changed(self._on_new_ibl)
        self._ibl_intensity = gui.Slider(gui.Slider.INT)
        self._ibl_intensity.set_limits(0, 200000)
        self._ibl_intensity.set_on_value_changed(self._on_ibl_intensity)
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label('HDR map'))
        grid.add_child(self._ibl_map)
        grid.add_child(gui.Label('Intensity'))
        grid.add_child(self._ibl_intensity)
        advanced.add_fixed(separation_height)
        advanced.add_child(gui.Label('Environment'))
        advanced.add_child(grid)
        self._sun_intensity = gui.Slider(gui.Slider.INT)
        self._sun_intensity.set_limits(0, 200000)
        self._sun_intensity.set_on_value_changed(self._on_sun_intensity)
        self._sun_dir = gui.VectorEdit()
        self._sun_dir.set_on_value_changed(self._on_sun_dir)
        self._sun_color = gui.ColorEdit()
        self._sun_color.set_on_value_changed(self._on_sun_color)
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label('Intensity'))
        grid.add_child(self._sun_intensity)
        grid.add_child(gui.Label('Direction'))
        grid.add_child(self._sun_dir)
        grid.add_child(gui.Label('Color'))
        grid.add_child(self._sun_color)
        advanced.add_fixed(separation_height)
        advanced.add_child(gui.Label('Sun (Directional light)'))
        advanced.add_child(grid)
        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(advanced)
        material_settings = gui.CollapsableVert('Material settings', 0, gui.Margins(em, 0, 0, 0))
        self._shader = gui.Combobox()
        self._shader.add_item(AppWindow.MATERIAL_NAMES[0])
        self._shader.add_item(AppWindow.MATERIAL_NAMES[1])
        self._shader.add_item(AppWindow.MATERIAL_NAMES[2])
        self._shader.add_item(AppWindow.MATERIAL_NAMES[3])
        self._shader.set_on_selection_changed(self._on_shader)
        self._material_prefab = gui.Combobox()
        for prefab_name in sorted(Settings.PREFAB.keys()):
            self._material_prefab.add_item(prefab_name)
        self._material_prefab.selected_text = Settings.DEFAULT_MATERIAL_NAME
        self._material_prefab.set_on_selection_changed(self._on_material_prefab)
        self._material_color = gui.ColorEdit()
        self._material_color.set_on_value_changed(self._on_material_color)
        self._point_size = gui.Slider(gui.Slider.INT)
        self._point_size.set_limits(1, 10)
        self._point_size.set_on_value_changed(self._on_point_size)
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label('Type'))
        grid.add_child(self._shader)
        grid.add_child(gui.Label('Material'))
        grid.add_child(self._material_prefab)
        grid.add_child(gui.Label('Color'))
        grid.add_child(self._material_color)
        grid.add_child(gui.Label('Point size'))
        grid.add_child(self._point_size)
        material_settings.add_child(grid)
        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(material_settings)
        w.set_on_layout(self._on_layout)
        w.add_child(self._scene)
        w.add_child(self._settings_panel)
        if gui.Application.instance.menubar is None:
            if isMacOS:
                app_menu = gui.Menu()
                app_menu.add_item('About', AppWindow.MENU_ABOUT)
                app_menu.add_separator()
                app_menu.add_item('Quit', AppWindow.MENU_QUIT)
            file_menu = gui.Menu()
            file_menu.add_item('Open...', AppWindow.MENU_OPEN)
            file_menu.add_item('Export Current Image...', AppWindow.MENU_EXPORT)
            if not isMacOS:
                file_menu.add_separator()
                file_menu.add_item('Quit', AppWindow.MENU_QUIT)
            settings_menu = gui.Menu()
            settings_menu.add_item('Lighting & Materials', AppWindow.MENU_SHOW_SETTINGS)
            settings_menu.set_checked(AppWindow.MENU_SHOW_SETTINGS, True)
            help_menu = gui.Menu()
            help_menu.add_item('About', AppWindow.MENU_ABOUT)
            menu = gui.Menu()
            if isMacOS:
                menu.add_menu('Example', app_menu)
                menu.add_menu('File', file_menu)
                menu.add_menu('Settings', settings_menu)
            else:
                menu.add_menu('File', file_menu)
                menu.add_menu('Settings', settings_menu)
                menu.add_menu('Help', help_menu)
            gui.Application.instance.menubar = menu
        w.set_on_menu_item_activated(AppWindow.MENU_OPEN, self._on_menu_open)
        w.set_on_menu_item_activated(AppWindow.MENU_EXPORT, self._on_menu_export)
        w.set_on_menu_item_activated(AppWindow.MENU_QUIT, self._on_menu_quit)
        w.set_on_menu_item_activated(AppWindow.MENU_SHOW_SETTINGS, self._on_menu_toggle_settings_panel)
        w.set_on_menu_item_activated(AppWindow.MENU_ABOUT, self._on_menu_about)
        self._apply_settings()

    def _apply_settings(self):
        if False:
            while True:
                i = 10
        bg_color = [self.settings.bg_color.red, self.settings.bg_color.green, self.settings.bg_color.blue, self.settings.bg_color.alpha]
        self._scene.scene.set_background(bg_color)
        self._scene.scene.show_skybox(self.settings.show_skybox)
        self._scene.scene.show_axes(self.settings.show_axes)
        if self.settings.new_ibl_name is not None:
            self._scene.scene.scene.set_indirect_light(self.settings.new_ibl_name)
            self.settings.new_ibl_name = None
        self._scene.scene.scene.enable_indirect_light(self.settings.use_ibl)
        self._scene.scene.scene.set_indirect_light_intensity(self.settings.ibl_intensity)
        sun_color = [self.settings.sun_color.red, self.settings.sun_color.green, self.settings.sun_color.blue]
        self._scene.scene.scene.set_sun_light(self.settings.sun_dir, sun_color, self.settings.sun_intensity)
        self._scene.scene.scene.enable_sun_light(self.settings.use_sun)
        if self.settings.apply_material:
            self._scene.scene.update_material(self.settings.material)
            self.settings.apply_material = False
        self._bg_color.color_value = self.settings.bg_color
        self._show_skybox.checked = self.settings.show_skybox
        self._show_axes.checked = self.settings.show_axes
        self._use_ibl.checked = self.settings.use_ibl
        self._use_sun.checked = self.settings.use_sun
        self._ibl_intensity.int_value = self.settings.ibl_intensity
        self._sun_intensity.int_value = self.settings.sun_intensity
        self._sun_dir.vector_value = self.settings.sun_dir
        self._sun_color.color_value = self.settings.sun_color
        self._material_prefab.enabled = self.settings.material.shader == Settings.LIT
        c = gui.Color(self.settings.material.base_color[0], self.settings.material.base_color[1], self.settings.material.base_color[2], self.settings.material.base_color[3])
        self._material_color.color_value = c
        self._point_size.double_value = self.settings.material.point_size

    def _on_layout(self, layout_context):
        if False:
            print('Hello World!')
        r = self.window.content_rect
        self._scene.frame = r
        width = 17 * layout_context.theme.font_size
        height = min(r.height, self._settings_panel.calc_preferred_size(layout_context, gui.Widget.Constraints()).height)
        self._settings_panel.frame = gui.Rect(r.get_right() - width, r.y, width, height)

    def _set_mouse_mode_rotate(self):
        if False:
            i = 10
            return i + 15
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)

    def _set_mouse_mode_fly(self):
        if False:
            i = 10
            return i + 15
        self._scene.set_view_controls(gui.SceneWidget.Controls.FLY)

    def _set_mouse_mode_sun(self):
        if False:
            for i in range(10):
                print('nop')
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_SUN)

    def _set_mouse_mode_ibl(self):
        if False:
            i = 10
            return i + 15
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_IBL)

    def _set_mouse_mode_model(self):
        if False:
            while True:
                i = 10
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_MODEL)

    def _on_bg_color(self, new_color):
        if False:
            for i in range(10):
                print('nop')
        self.settings.bg_color = new_color
        self._apply_settings()

    def _on_show_skybox(self, show):
        if False:
            i = 10
            return i + 15
        self.settings.show_skybox = show
        self._apply_settings()

    def _on_show_axes(self, show):
        if False:
            for i in range(10):
                print('nop')
        self.settings.show_axes = show
        self._apply_settings()

    def _on_use_ibl(self, use):
        if False:
            return 10
        self.settings.use_ibl = use
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_use_sun(self, use):
        if False:
            return 10
        self.settings.use_sun = use
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_lighting_profile(self, name, index):
        if False:
            while True:
                i = 10
        if name != Settings.CUSTOM_PROFILE_NAME:
            self.settings.apply_lighting_profile(name)
            self._apply_settings()

    def _on_new_ibl(self, name, index):
        if False:
            for i in range(10):
                print('nop')
        self.settings.new_ibl_name = gui.Application.instance.resource_path + '/' + name
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_ibl_intensity(self, intensity):
        if False:
            return 10
        self.settings.ibl_intensity = int(intensity)
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_sun_intensity(self, intensity):
        if False:
            return 10
        self.settings.sun_intensity = int(intensity)
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_sun_dir(self, sun_dir):
        if False:
            while True:
                i = 10
        self.settings.sun_dir = sun_dir
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_sun_color(self, color):
        if False:
            print('Hello World!')
        self.settings.sun_color = color
        self._apply_settings()

    def _on_shader(self, name, index):
        if False:
            print('Hello World!')
        self.settings.set_material(AppWindow.MATERIAL_SHADERS[index])
        self._apply_settings()

    def _on_material_prefab(self, name, index):
        if False:
            return 10
        self.settings.apply_material_prefab(name)
        self.settings.apply_material = True
        self._apply_settings()

    def _on_material_color(self, color):
        if False:
            print('Hello World!')
        self.settings.material.base_color = [color.red, color.green, color.blue, color.alpha]
        self.settings.apply_material = True
        self._apply_settings()

    def _on_point_size(self, size):
        if False:
            for i in range(10):
                print('nop')
        self.settings.material.point_size = int(size)
        self.settings.apply_material = True
        self._apply_settings()

    def _on_menu_open(self):
        if False:
            i = 10
            return i + 15
        dlg = gui.FileDialog(gui.FileDialog.OPEN, 'Choose file to load', self.window.theme)
        dlg.add_filter('.ply .stl .fbx .obj .off .gltf .glb', 'Triangle mesh files (.ply, .stl, .fbx, .obj, .off, .gltf, .glb)')
        dlg.add_filter('.xyz .xyzn .xyzrgb .ply .pcd .pts', 'Point cloud files (.xyz, .xyzn, .xyzrgb, .ply, .pcd, .pts)')
        dlg.add_filter('.ply', 'Polygon files (.ply)')
        dlg.add_filter('.stl', 'Stereolithography files (.stl)')
        dlg.add_filter('.fbx', 'Autodesk Filmbox files (.fbx)')
        dlg.add_filter('.obj', 'Wavefront OBJ files (.obj)')
        dlg.add_filter('.off', 'Object file format (.off)')
        dlg.add_filter('.gltf', 'OpenGL transfer files (.gltf)')
        dlg.add_filter('.glb', 'OpenGL binary transfer files (.glb)')
        dlg.add_filter('.xyz', 'ASCII point cloud files (.xyz)')
        dlg.add_filter('.xyzn', 'ASCII point cloud with normals (.xyzn)')
        dlg.add_filter('.xyzrgb', 'ASCII point cloud files with colors (.xyzrgb)')
        dlg.add_filter('.pcd', 'Point Cloud Data files (.pcd)')
        dlg.add_filter('.pts', '3D Points files (.pts)')
        dlg.add_filter('', 'All files')
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_load_dialog_done)
        self.window.show_dialog(dlg)

    def _on_file_dialog_cancel(self):
        if False:
            while True:
                i = 10
        self.window.close_dialog()

    def _on_load_dialog_done(self, filename):
        if False:
            while True:
                i = 10
        self.window.close_dialog()
        self.load(filename)

    def _on_menu_export(self):
        if False:
            i = 10
            return i + 15
        dlg = gui.FileDialog(gui.FileDialog.SAVE, 'Choose file to save', self.window.theme)
        dlg.add_filter('.png', 'PNG files (.png)')
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_export_dialog_done)
        self.window.show_dialog(dlg)

    def _on_export_dialog_done(self, filename):
        if False:
            return 10
        self.window.close_dialog()
        frame = self._scene.frame
        self.export_image(filename, frame.width, frame.height)

    def _on_menu_quit(self):
        if False:
            return 10
        gui.Application.instance.quit()

    def _on_menu_toggle_settings_panel(self):
        if False:
            for i in range(10):
                print('nop')
        self._settings_panel.visible = not self._settings_panel.visible
        gui.Application.instance.menubar.set_checked(AppWindow.MENU_SHOW_SETTINGS, self._settings_panel.visible)

    def _on_menu_about(self):
        if False:
            return 10
        em = self.window.theme.font_size
        dlg = gui.Dialog('About')
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label('Open3D GUI Example'))
        ok = gui.Button('OK')
        ok.set_on_clicked(self._on_about_ok)
        h = gui.Horiz()
        h.add_stretch()
        h.add_child(ok)
        h.add_stretch()
        dlg_layout.add_child(h)
        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)

    def _on_about_ok(self):
        if False:
            while True:
                i = 10
        self.window.close_dialog()

    def load(self, path):
        if False:
            while True:
                i = 10
        self._scene.scene.clear_geometry()
        geometry = None
        geometry_type = o3d.io.read_file_geometry_type(path)
        mesh = None
        if geometry_type & o3d.io.CONTAINS_TRIANGLES:
            mesh = o3d.io.read_triangle_model(path)
        if mesh is None:
            print('[Info]', path, 'appears to be a point cloud')
            cloud = None
            try:
                cloud = o3d.io.read_point_cloud(path)
            except Exception:
                pass
            if cloud is not None:
                print('[Info] Successfully read', path)
                if not cloud.has_normals():
                    cloud.estimate_normals()
                cloud.normalize_normals()
                geometry = cloud
            else:
                print('[WARNING] Failed to read points', path)
        if geometry is not None or mesh is not None:
            try:
                if mesh is not None:
                    self._scene.scene.add_model('__model__', mesh)
                else:
                    self._scene.scene.add_geometry('__model__', geometry, self.settings.material)
                bounds = self._scene.scene.bounding_box
                self._scene.setup_camera(60, bounds, bounds.get_center())
            except Exception as e:
                print(e)

    def export_image(self, path, width, height):
        if False:
            while True:
                i = 10

        def on_image(image):
            if False:
                for i in range(10):
                    print('nop')
            img = image
            quality = 9
            if path.endswith('.jpg'):
                quality = 100
            o3d.io.write_image(path, img, quality)
        self._scene.scene.scene.render_to_image(on_image)

def main():
    if False:
        return 10
    gui.Application.instance.initialize()
    w = AppWindow(1024, 768)
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.exists(path):
            w.load(path)
        else:
            w.window.show_message_box('Error', "Could not open file '" + path + "'")
    gui.Application.instance.run()
if __name__ == '__main__':
    main()