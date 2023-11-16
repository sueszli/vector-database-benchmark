"""Data Editor component from glide-data-grid."""
from __future__ import annotations
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
from reflex.base import Base
from reflex.components.component import Component, NoSSRComponent
from reflex.components.literals import LiteralRowMarker
from reflex.utils import console, format, imports, types
from reflex.utils.serializers import serializer
from reflex.vars import ImportVar, Var, get_unique_variable_name

class GridColumnIcons(Enum):
    """An Enum for the available icons in DataEditor."""
    Array = 'array'
    AudioUri = 'audio_uri'
    Boolean = 'boolean'
    HeaderCode = 'code'
    Date = 'date'
    Email = 'email'
    Emoji = 'emoji'
    GeoDistance = 'geo_distance'
    IfThenElse = 'if_then_else'
    Image = 'image'
    JoinStrings = 'join_strings'
    Lookup = 'lookup'
    Markdown = 'markdown'
    Math = 'math'
    Number = 'number'
    Phone = 'phone'
    Reference = 'reference'
    Rollup = 'rollup'
    RowID = 'row_id'
    SingleValue = 'single_value'
    SplitString = 'split_string'
    String = 'string'
    TextTemplate = 'text_template'
    Time = 'time'
    Uri = 'uri'
    VideoUri = 'video_uri'

class DataEditorTheme(Base):
    """The theme for the DataEditor component."""
    accent_color: Optional[str] = None
    accent_fg: Optional[str] = None
    accent_light: Optional[str] = None
    base_font_style: Optional[str] = None
    bg_bubble: Optional[str] = None
    bg_bubble_selected: Optional[str] = None
    bg_cell: Optional[str] = None
    bg_cell_medium: Optional[str] = None
    bg_header: Optional[str] = None
    bg_header_has_focus: Optional[str] = None
    bg_header_hovered: Optional[str] = None
    bg_icon_header: Optional[str] = None
    bg_search_result: Optional[str] = None
    border_color: Optional[str] = None
    cell_horizontal_padding: Optional[int] = None
    cell_vertical_padding: Optional[int] = None
    drilldown_border: Optional[str] = None
    editor_font_size: Optional[str] = None
    fg_icon_header: Optional[str] = None
    font_family: Optional[str] = None
    header_bottom_border_color: Optional[str] = None
    header_font_style: Optional[str] = None
    horizontal_border_color: Optional[str] = None
    line_height: Optional[int] = None
    link_color: Optional[str] = None
    text_bubble: Optional[str] = None
    text_dark: Optional[str] = None
    text_group_header: Optional[str] = None
    text_header: Optional[str] = None
    text_header_selected: Optional[str] = None
    text_light: Optional[str] = None
    text_medium: Optional[str] = None

class DataEditor(NoSSRComponent):
    """The DataEditor Component."""
    tag = 'DataEditor'
    is_default = True
    library: str = '@glideapps/glide-data-grid@^5.3.0'
    lib_dependencies: List[str] = ['lodash', 'marked', 'react-responsive-carousel']
    rows: Var[int]
    columns: Var[List[Dict[str, Any]]]
    data: Var[List[List[Any]]]
    get_cell_content: Var[str]
    get_cell_for_selection: Var[bool]
    on_paste: Var[bool]
    draw_focus_ring: Var[bool]
    fixed_shadow_x: Var[bool]
    fixed_shadow_y: Var[bool]
    freeze_columns: Var[int]
    group_header_height: Var[int]
    header_height: Var[int]
    max_column_auto_width: Var[int]
    max_column_width: Var[int]
    min_column_width: Var[int]
    row_height: Var[int]
    row_markers: Var[LiteralRowMarker]
    row_marker_start_index: Var[int]
    row_marker_width: Var[int]
    smooth_scroll_x: Var[bool]
    smooth_scroll_y: Var[bool]
    vertical_border: Var[bool]
    column_select: Var[str]
    prevent_diagonal_scrolling: Var[bool]
    overscroll_x: Var[int]
    overscroll_y: Var[int]
    scroll_offset_x: Var[int]
    scroll_offset_y: Var[int]
    theme: Var[Union[DataEditorTheme, Dict]]

    def _get_imports(self):
        if False:
            return 10
        return imports.merge_imports(super()._get_imports(), {'': {ImportVar(tag=f'{format.format_library_name(self.library)}/dist/index.css')}, self.library: {ImportVar(tag='GridCellKind')}, '/utils/helpers/dataeditor.js': {ImportVar(tag=f'formatDataEditorCells', is_default=False, install=False)}})

    def get_event_triggers(self) -> Dict[str, Callable]:
        if False:
            for i in range(10):
                print('nop')
        'The event triggers of the component.\n\n        Returns:\n            The dict describing the event triggers.\n        '

        def edit_sig(pos, data: dict[str, Any]):
            if False:
                while True:
                    i = 10
            return [pos, data]
        return {'on_cell_activated': lambda pos: [pos], 'on_cell_clicked': lambda pos: [pos], 'on_cell_context_menu': lambda pos: [pos], 'on_cell_edited': edit_sig, 'on_group_header_clicked': edit_sig, 'on_group_header_context_menu': lambda grp_idx, data: [grp_idx, data], 'on_group_header_renamed': lambda idx, val: [idx, val], 'on_header_clicked': lambda pos: [pos], 'on_header_context_menu': lambda pos: [pos], 'on_header_menu_click': lambda col, pos: [col, pos], 'on_item_hovered': lambda pos: [pos], 'on_delete': lambda selection: [selection], 'on_finished_editing': lambda new_value, movement: [new_value, movement], 'on_row_appended': lambda : [], 'on_selection_cleared': lambda : [], 'on_column_resize': lambda col, width: [col, width]}

    def _get_hooks(self) -> str | None:
        if False:
            for i in range(10):
                print('nop')
        editor_id = get_unique_variable_name()
        data_callback = f'getData_{editor_id}'
        self.get_cell_content = Var.create(data_callback, _var_is_local=False)
        code = [f'function {data_callback}([col, row]){{']
        columns_path = f'{self.columns._var_full_name}'
        data_path = f'{self.data._var_full_name}'
        code.extend([f'    return formatDataEditorCells(col, row, {columns_path}, {data_path});', '  }'])
        return '\n'.join(code)

    @classmethod
    def create(cls, *children, **props) -> Component:
        if False:
            return 10
        'Create the DataEditor component.\n\n        Args:\n            *children: The children of the data editor.\n            **props: The props of the data editor.\n\n        Raises:\n            ValueError: invalid input.\n\n        Returns:\n            The DataEditor component.&\n        '
        from reflex.el.elements import Div
        columns = props.get('columns', [])
        data = props.get('data', [])
        rows = props.get('rows', None)
        if rows is None:
            props['rows'] = data.length() if isinstance(data, Var) else len(data)
        if not isinstance(columns, Var) and len(columns):
            if types.is_dataframe(type(data)) or (isinstance(data, Var) and types.is_dataframe(data._var_type)):
                raise ValueError('Cannot pass in both a pandas dataframe and columns to the data_editor component.')
            else:
                props['columns'] = [format.format_data_editor_column(col) for col in columns]
        if 'theme' in props:
            theme = props.get('theme')
            if isinstance(theme, Dict):
                props['theme'] = DataEditorTheme(**theme)
        props.setdefault('get_cell_for_selection', True)
        props.setdefault('on_paste', False)
        if props.pop('get_cell_content', None) is not None:
            console.warn('get_cell_content is not user configurable, the provided value will be discarded')
        grid = super().create(*children, **props)
        return Div.create(grid, width=props.pop('width', '100%'), height=props.pop('height', '100%'))

    def _get_app_wrap_components(self) -> dict[tuple[int, str], Component]:
        if False:
            print('Hello World!')
        'Get the app wrap components for the component.\n\n        Returns:\n            The app wrap components.\n        '
        from reflex.el.elements import Div

        class Portal(Div):

            def get_ref(self):
                if False:
                    for i in range(10):
                        print('nop')
                return None
        return {(-1, 'DataEditorPortal'): Portal.create(id='portal')}

@serializer
def serialize_dataeditortheme(theme: DataEditorTheme):
    if False:
        i = 10
        return i + 15
    'The serializer for the data editor theme.\n\n    Args:\n        theme: The theme to serialize.\n\n    Returns:\n        The serialized theme.\n    '
    return format.json_dumps({format.to_camel_case(k): v for (k, v) in theme.__dict__.items() if v is not None})