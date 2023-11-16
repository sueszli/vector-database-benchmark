from typing import List, Any, Callable, Union, Tuple
from contextlib import contextmanager
import warnings
import functools
import inspect
import dearpygui._dearpygui as internal_dpg
from dearpygui._dearpygui import mvBuffer
from dearpygui._dearpygui import mvVec4
from dearpygui._dearpygui import mvMat4

def run_callbacks(jobs):
    if False:
        for i in range(10):
            print('nop')
    ' New in 1.2. Runs callbacks from the callback queue and checks arguments. '
    if jobs is None:
        pass
    else:
        for job in jobs:
            if job[0] is None:
                pass
            else:
                sig = inspect.signature(job[0])
                args = []
                for arg in range(len(sig.parameters)):
                    args.append(job[arg + 1])
                job[0](*args)

def get_major_version():
    if False:
        return 10
    ' return Dear PyGui Major Version '
    return internal_dpg.get_app_configuration()['major_version']

def get_minor_version():
    if False:
        while True:
            i = 10
    ' return Dear PyGui Minor Version '
    return internal_dpg.get_app_configuration()['minor_version']

def get_dearpygui_version():
    if False:
        i = 10
        return i + 15
    ' return Dear PyGui Version '
    return internal_dpg.get_app_configuration()['version']

def configure_item(item: Union[int, str], **kwargs) -> None:
    if False:
        i = 10
        return i + 15
    'Configures an item after creation.'
    internal_dpg.configure_item(item, **kwargs)

def configure_app(**kwargs) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Configures an item after creation.'
    internal_dpg.configure_app(**kwargs)

def configure_viewport(item: Union[int, str], **kwargs) -> None:
    if False:
        return 10
    'Configures a viewport after creation.'
    internal_dpg.configure_viewport(item, **kwargs)

def start_dearpygui():
    if False:
        return 10
    'Prepares viewport (if not done already). sets up, cleans up, and runs main event loop.\n\n    Returns:\n        None\n    '
    if not internal_dpg.is_viewport_ok():
        raise RuntimeError('Viewport was not created and shown.')
        return
    while internal_dpg.is_dearpygui_running():
        internal_dpg.render_dearpygui_frame()

@contextmanager
def mutex():
    if False:
        print('Hello World!')
    ' Handles locking/unlocking render thread mutex. '
    try:
        yield internal_dpg.lock_mutex()
    finally:
        internal_dpg.unlock_mutex()

@contextmanager
def popup(parent: Union[int, str], mousebutton: int=internal_dpg.mvMouseButton_Right, modal: bool=False, tag: Union[int, str]=0, min_size: Union[List[int], Tuple[int, ...]]=[100, 100], max_size: Union[List[int], Tuple[int, ...]]=[30000, 30000], no_move: bool=False, no_background: bool=False) -> int:
    if False:
        return 10
    "A window that will be displayed when a parent item is hovered and the corresponding mouse button has been clicked. By default a popup will shrink fit the items it contains.\n    This is useful for context windows, and simple modal window popups.\n    When popups are used a modal they have more avaliable settings (i.e. title, resize, width, height) These\n    can be set by using configure item. \n    This is a light wrapper over window. For more control over a modal|popup window use a normal window with the modal|popup keyword \n    and set the item handler and mouse events manually.\n\n    Args:\n        parent: The UI item that will need to be hovered.\n        **mousebutton: The mouse button that will trigger the window to popup.\n        **modal: Will force the user to interact with the popup.\n        **min_size: New in 1.4. Minimum window size.\n        **max_size: New in 1.4. Maximum window size.\n        **no_move: New in 1.4. Prevents the window from moving based on user input.\n        **no_background: New in 1.4. Sets Background and border alpha to transparent.\n\n    Returns:\n        item's uuid\n    "
    try:
        if tag == 0:
            _internal_popup_id = internal_dpg.generate_uuid()
        else:
            _internal_popup_id = tag
        _handler_reg_id = internal_dpg.add_item_handler_registry()
        internal_dpg.add_item_clicked_handler(mousebutton, parent=internal_dpg.last_item(), callback=lambda : internal_dpg.configure_item(_internal_popup_id, show=True))
        internal_dpg.bind_item_handler_registry(parent, _handler_reg_id)
        if modal:
            internal_dpg.add_window(modal=True, show=False, tag=_internal_popup_id, autosize=True, min_size=min_size, max_size=max_size, no_move=no_move, no_background=no_background)
        else:
            internal_dpg.add_window(popup=True, show=False, tag=_internal_popup_id, autosize=True, min_size=min_size, max_size=max_size, no_move=no_move, no_background=no_background)
        internal_dpg.push_container_stack(internal_dpg.last_container())
        yield _internal_popup_id
    finally:
        internal_dpg.pop_container_stack()

def show_style_editor() -> None:
    if False:
        i = 10
        return i + 15
    'Shows the standard style editor window\n\n    Returns:\n        None\n    '
    internal_dpg.show_tool(internal_dpg.mvTool_Style)

def show_metrics() -> None:
    if False:
        while True:
            i = 10
    'Shows the standard metrics window\n\n    Returns:\n        None\n    '
    internal_dpg.show_tool(internal_dpg.mvTool_Metrics)

def show_about() -> None:
    if False:
        return 10
    'Shows the standard about window\n\n    Returns:\n        None\n    '
    internal_dpg.show_tool(internal_dpg.mvTool_About)

def show_debug() -> None:
    if False:
        while True:
            i = 10
    'Shows the standard debug window\n\n    Returns:\n        None\n    '
    internal_dpg.show_tool(internal_dpg.mvTool_Debug)

def show_documentation() -> None:
    if False:
        i = 10
        return i + 15
    'Shows the standard documentation window\n\n    Returns:\n        None\n    '
    internal_dpg.show_tool(internal_dpg.mvTool_Doc)

def show_font_manager() -> None:
    if False:
        i = 10
        return i + 15
    'Shows a debug tool for the font manager\n\n    Returns:\n        None\n    '
    internal_dpg.show_tool(internal_dpg.mvTool_Font)

def show_item_registry() -> None:
    if False:
        while True:
            i = 10
    'Shows the item hierarchy of your application\n\n    Returns:\n        None\n    '
    internal_dpg.show_tool(internal_dpg.mvTool_ItemRegistry)

def get_item_slot(item: Union[int, str]) -> Union[int, None]:
    if False:
        while True:
            i = 10
    "Returns an item's target slot.\n\n    Returns:\n        slot as a int\n    "
    return internal_dpg.get_item_info(item)['target']

def is_item_container(item: Union[int, str]) -> Union[bool, None]:
    if False:
        while True:
            i = 10
    'Checks if item is a container.\n\n    Returns:\n        status as a bool\n    '
    return internal_dpg.get_item_info(item)['container']

def get_item_parent(item: Union[int, str]) -> Union[int, None]:
    if False:
        return 10
    "Gets the item's parent.\n\n    Returns:\n        parent as a int or None\n    "
    return internal_dpg.get_item_info(item)['parent']

def get_item_children(item: Union[int, str], slot: int=-1) -> Union[dict, List[int], None]:
    if False:
        while True:
            i = 10
    "Provides access to the item's children slots.\n\n    Returns:\n        A 2-D tuple of children slots ex. ((child_slot_1),(child_slot_2),(child_slot_3),...) or a single slot if slot is used.\n    "
    if slot < 0 or slot > 4:
        return internal_dpg.get_item_info(item)['children']
    return internal_dpg.get_item_info(item)['children'][slot]

def get_item_type(item: Union[int, str]) -> Union[str]:
    if False:
        return 10
    "Gets the item's type.\n\n    Returns:\n        type as a string or None\n    "
    return internal_dpg.get_item_info(item)['type']

def get_item_theme(item: Union[int, str]) -> int:
    if False:
        while True:
            i = 10
    "Gets the item's theme.\n\n    Returns:\n        theme's uuid\n    "
    return internal_dpg.get_item_info(item)['theme']

def get_item_font(item: Union[int, str]) -> int:
    if False:
        for i in range(10):
            print('nop')
    "Gets the item's font.\n\n    Returns:\n        font's uuid\n    "
    return internal_dpg.get_item_info(item)['font']

def get_item_disabled_theme(item: Union[int, str]) -> int:
    if False:
        while True:
            i = 10
    "Gets the item's disabled theme.\n\n    Returns:\n        theme's uuid\n    "
    return internal_dpg.get_item_info(item)['disabled_theme']

def enable_item(item: Union[int, str]):
    if False:
        return 10
    'Enables the item.\n\n    Args:\n        **item: Item to enable.\n\n    Returns:\n        None\n    '
    internal_dpg.configure_item(item, enabled=True)

def disable_item(item: Union[int, str]):
    if False:
        for i in range(10):
            print('nop')
    'Disables the item.\n\n    Args:\n        **item: Item to disable.\n\n    Returns:\n        None\n    '
    internal_dpg.configure_item(item, enabled=False)

def set_item_label(item: Union[int, str], label: str):
    if False:
        print('Hello World!')
    'Sets the item\'s displayed label, anything after the characters "##" in the name will not be shown.\n\n    Args:\n        item: Item label will be applied to.\n        label: Displayed name to be applied.\n\n    Returns:\n        None\n    '
    internal_dpg.configure_item(item, label=label)

def set_item_source(item: Union[int, str], source: Union[int, str]):
    if False:
        return 10
    'Sets the item\'s value, to the source\'s value. Widget\'s value will now be "linked" to source\'s value.\n\n    Args:\n        item: Item to me linked.\n        source: Source to link to.\n\n    Returns:\n        None\n    '
    internal_dpg.configure_item(item, source=source)

def set_item_pos(item: Union[int, str], pos: List[float]):
    if False:
        print('Hello World!')
    "Sets the item's position.\n\n    Args:\n        item: Item the absolute position will be applied to.\n        pos: X and Y positions relative to parent of the item.\n\n    Returns:\n        None\n    "
    internal_dpg.configure_item(item, pos=pos)

def set_item_width(item: Union[int, str], width: int):
    if False:
        return 10
    "Sets the item's width.\n\n    Args:\n        item: Item the Width will be applied to.\n        width: Width to be applied.\n\n    Returns:\n        None\n    "
    internal_dpg.configure_item(item, width=width)

def set_item_height(item: Union[int, str], height: int):
    if False:
        return 10
    "Sets the item's height.\n\n    Args:\n        item: Item the Height will be applied to.\n        height: Height to be applied.\n\n    Returns:\n        None\n    "
    internal_dpg.configure_item(item, height=height)

def set_item_indent(item: Union[int, str], indent: int):
    if False:
        return 10
    "Sets the item's indent.\n\n    Args:\n        item: Item the Height will be applied to.\n        height: Height to be applied.\n\n    Returns:\n        None\n    "
    internal_dpg.configure_item(item, indent=indent)

def set_item_track_offset(item: Union[int, str], offset: float):
    if False:
        i = 10
        return i + 15
    "Sets the item's track offset.\n\n    Args:\n        item: Item the Height will be applied to.\n        height: Height to be applied.\n\n    Returns:\n        None\n    "
    internal_dpg.configure_item(item, track_offset=offset)

def set_item_payload_type(item: Union[int, str], payload_type: str):
    if False:
        return 10
    "Sets the item's payload type.\n\n    Args:\n        item: Item the Height will be applied to.\n        height: Height to be applied.\n\n    Returns:\n        None\n    "
    internal_dpg.configure_item(item, payload_type=str)

def set_item_callback(item: Union[int, str], callback: Callable):
    if False:
        print('Hello World!')
    "Sets the item's callack.\n\n    Args:\n        item: Item the callback will be applied to.\n        callback: Callback to be applied.\n\n    Returns:\n        None\n    "
    internal_dpg.configure_item(item, callback=callback)

def set_item_drag_callback(item: Union[int, str], callback: Callable):
    if False:
        i = 10
        return i + 15
    "Sets the item's drag callack.\n\n    Args:\n        item: Item the callback will be applied to.\n        callback: Callback to be applied.\n\n    Returns:\n        None\n    "
    internal_dpg.configure_item(item, drag_callback=callback)

def set_item_drop_callback(item: Union[int, str], callback: Callable):
    if False:
        i = 10
        return i + 15
    "Sets the item's drop callack.\n\n    Args:\n        item: Item the callback will be applied to.\n        callback: Callback to be applied.\n\n    Returns:\n        None\n    "
    internal_dpg.configure_item(item, drop_callback=callback)

def track_item(item: Union[int, str]):
    if False:
        i = 10
        return i + 15
    'Track item in scroll region.\n\n    Args:\n        item: Item the callback will be applied to.\n        callback: Callback to be applied.\n\n    Returns:\n        None\n    '
    internal_dpg.configure_item(item, tracked=True)

def untrack_item(item: Union[int, str]):
    if False:
        i = 10
        return i + 15
    'Track item in scroll region.\n\n    Args:\n        item: Item the callback will be applied to.\n        callback: Callback to be applied.\n\n    Returns:\n        None\n    '
    internal_dpg.configure_item(item, tracked=False)

def set_item_user_data(item: Union[int, str], user_data: Any):
    if False:
        print('Hello World!')
    "Sets the item's callack_data to any python object.\n\n    Args:\n        item: Item the callback will be applied to.\n        user_data: Callback_data to be applied.\n\n    Returns:\n        None\n    "
    internal_dpg.configure_item(item, user_data=user_data)

def show_item(item: Union[int, str]):
    if False:
        return 10
    'Shows the item.\n\n    Args:\n        item: Item to show.\n\n    Returns:\n        None\n    '
    internal_dpg.configure_item(item, show=True)

def hide_item(item: Union[int, str], *, children_only: bool=False):
    if False:
        print('Hello World!')
    'Hides the item.\n\n    Args:\n        **item: Item to hide.\n\n    Returns:\n        None\n    '
    if children_only:
        children = get_item_children(item)
        for child in children:
            internal_dpg.configure_item(child, show=False)
    else:
        internal_dpg.configure_item(item, show=False)

def get_item_label(item: Union[int, str]) -> Union[str, None]:
    if False:
        return 10
    "Gets the item's label.\n\n    Returns:\n        label as a string or None\n    "
    return internal_dpg.get_item_configuration(item)['label']

def get_item_filter_key(item: Union[int, str]) -> Union[str, None]:
    if False:
        print('Hello World!')
    "Gets the item's filter key.\n\n    Returns:\n        filter key as a string or None\n    "
    return internal_dpg.get_item_configuration(item)['filter_key']

def is_item_tracked(item: Union[int, str]) -> Union[bool, None]:
    if False:
        print('Hello World!')
    'Checks if item is tracked.\n\n    Returns:\n        tracked as a bool or None\n    '
    return internal_dpg.get_item_configuration(item)['tracked']

def is_item_search_delayed(item: Union[int, str]) -> Union[bool, None]:
    if False:
        return 10
    'Checks if item is search delayed.\n\n    Returns:\n        tracked as a bool or None\n    '
    return internal_dpg.get_item_configuration(item)['delay_search']

def get_item_indent(item: Union[int, str]) -> Union[int, None]:
    if False:
        while True:
            i = 10
    "Gets the item's indent.\n\n    Returns:\n        indent as a int or None\n    "
    return internal_dpg.get_item_configuration(item)['indent']

def get_item_track_offset(item: Union[int, str]) -> Union[float, None]:
    if False:
        return 10
    "Gets the item's track offset.\n\n    Returns:\n        track offset as a int or None\n    "
    return internal_dpg.get_item_configuration(item)['track_offset']

def get_item_width(item: Union[int, str]) -> Union[int, None]:
    if False:
        while True:
            i = 10
    "Gets the item's width.\n\n    Returns:\n        width as a int or None\n    "
    return internal_dpg.get_item_configuration(item)['width']

def get_item_height(item: Union[int, str]) -> Union[int, None]:
    if False:
        while True:
            i = 10
    "Gets the item's height.\n\n    Returns:\n        height as a int or None\n    "
    return internal_dpg.get_item_configuration(item)['height']

def get_item_callback(item: Union[int, str]) -> Union[Callable, None]:
    if False:
        while True:
            i = 10
    "Gets the item's callback.\n\n    Returns:\n        callback as a callable or None\n    "
    return internal_dpg.get_item_configuration(item)['callback']

def get_item_drag_callback(item: Union[int, str]) -> Union[Callable, None]:
    if False:
        i = 10
        return i + 15
    "Gets the item's drag callback.\n\n    Returns:\n        callback as a callable or None\n    "
    return internal_dpg.get_item_configuration(item)['drag_callback']

def get_item_drop_callback(item: Union[int, str]) -> Union[Callable, None]:
    if False:
        return 10
    "Gets the item's drop callback.\n\n    Returns:\n        callback as a callable or None\n    "
    return internal_dpg.get_item_configuration(item)['drop_callback']

def get_item_user_data(item: Union[int, str]) -> Union[Any, None]:
    if False:
        for i in range(10):
            print('nop')
    "Gets the item's callback data.\n\n    Returns:\n        callback data as a python object or None\n    "
    return internal_dpg.get_item_configuration(item)['user_data']

def get_item_source(item: Union[int, str]) -> Union[str, None]:
    if False:
        return 10
    "Gets the item's source.\n\n    Returns:\n        source as a string or None\n    "
    return internal_dpg.get_item_configuration(item)['source']

def is_item_hovered(item: Union[int, str]) -> Union[bool, None]:
    if False:
        for i in range(10):
            print('nop')
    'Checks if item is hovered.\n\n    Returns:\n        status as a bool\n    '
    return internal_dpg.get_item_state(item)['hovered']

def is_item_active(item: Union[int, str]) -> Union[bool, None]:
    if False:
        print('Hello World!')
    'Checks if item is active.\n\n    Returns:\n        status as a bool\n    '
    return internal_dpg.get_item_state(item)['active']

def is_item_focused(item: Union[int, str]) -> Union[bool, None]:
    if False:
        for i in range(10):
            print('nop')
    'Checks if item is focused.\n\n    Returns:\n        status as a bool\n    '
    return internal_dpg.get_item_state(item)['focused']

def is_item_clicked(item: Union[int, str]) -> Union[bool, None]:
    if False:
        for i in range(10):
            print('nop')
    'Checks if item is clicked.\n\n    Returns:\n        status as a bool\n    '
    return internal_dpg.get_item_state(item)['clicked']

def is_item_left_clicked(item: Union[int, str]) -> Union[bool, None]:
    if False:
        print('Hello World!')
    'Checks if item is left clicked.\n\n    Returns:\n        status as a bool\n    '
    return internal_dpg.get_item_state(item)['left_clicked']

def is_item_right_clicked(item: Union[int, str]) -> Union[bool, None]:
    if False:
        i = 10
        return i + 15
    'Checks if item is right clicked.\n\n    Returns:\n        status as a bool\n    '
    return internal_dpg.get_item_state(item)['right_clicked']

def is_item_middle_clicked(item: Union[int, str]) -> Union[bool, None]:
    if False:
        i = 10
        return i + 15
    'Checks if item is middle clicked.\n\n    Returns:\n        status as a bool\n    '
    return internal_dpg.get_item_state(item)['middle_clicked']

def is_item_visible(item: Union[int, str]) -> Union[bool, None]:
    if False:
        print('Hello World!')
    'Checks if item is visible.\n\n    Returns:\n        status as a bool\n    '
    return internal_dpg.get_item_state(item)['visible']

def is_item_edited(item: Union[int, str]) -> Union[bool, None]:
    if False:
        print('Hello World!')
    'Checks if item is edited.\n\n    Returns:\n        status as a bool\n    '
    return internal_dpg.get_item_state(item)['edited']

def is_item_activated(item: Union[int, str]) -> Union[bool, None]:
    if False:
        i = 10
        return i + 15
    'Checks if item is activated.\n\n    Returns:\n        status as a bool\n    '
    return internal_dpg.get_item_state(item)['activated']

def is_item_deactivated(item: Union[int, str]) -> Union[bool, None]:
    if False:
        return 10
    'Checks if item is deactivated.\n\n    Returns:\n        status as a bool\n    '
    return internal_dpg.get_item_state(item)['deactivated']

def is_item_deactivated_after_edit(item: Union[int, str]) -> Union[bool, None]:
    if False:
        print('Hello World!')
    'Checks if item is deactivated_after_edit.\n\n    Returns:\n        status as a bool\n    '
    return internal_dpg.get_item_state(item)['deactivated_after_edit']

def is_item_toggled_open(item: Union[int, str]) -> Union[bool, None]:
    if False:
        for i in range(10):
            print('nop')
    'Checks if item is toggled_open.\n\n    Returns:\n        status as a bool\n    '
    return internal_dpg.get_item_state(item)['toggled_open']

def is_item_ok(item: Union[int, str]) -> Union[bool, None]:
    if False:
        i = 10
        return i + 15
    'Checks if item is ok and can be used.\n\n    Returns:\n        status as a bool\n    '
    return internal_dpg.get_item_state(item)['ok']

def is_item_shown(item: Union[int, str]) -> Union[bool, None]:
    if False:
        for i in range(10):
            print('nop')
    'Checks if item is shown.\n\n    Returns:\n        status as a bool\n    '
    return internal_dpg.get_item_configuration(item)['show']

def is_item_enabled(item: Union[int, str]) -> Union[bool, None]:
    if False:
        i = 10
        return i + 15
    'Checks if item is enabled.\n\n    Returns:\n        status as a bool\n    '
    return internal_dpg.get_item_configuration(item)['enabled']

def get_item_pos(item: Union[int, str]) -> List[int]:
    if False:
        return 10
    "Returns item's position.\n\n    Returns:\n        position\n    "
    return internal_dpg.get_item_state(item)['pos']

def get_available_content_region(item: Union[int, str]) -> List[int]:
    if False:
        return 10
    "Returns item's available content region.\n\n    Returns:\n        position\n    "
    return internal_dpg.get_item_state(item)['content_region_avail']

def get_item_rect_size(item: Union[int, str]) -> List[int]:
    if False:
        print('Hello World!')
    "Returns item's available content region.\n\n    Returns:\n        position\n    "
    return internal_dpg.get_item_state(item)['rect_size']

def get_item_rect_min(item: Union[int, str]) -> List[int]:
    if False:
        return 10
    "Returns item's minimum content region.\n\n    Returns:\n        position\n    "
    return internal_dpg.get_item_state(item)['rect_min']

def get_item_rect_max(item: Union[int, str]) -> List[int]:
    if False:
        for i in range(10):
            print('nop')
    "Returns item's maximum content region.\n\n    Returns:\n        position\n    "
    return internal_dpg.get_item_state(item)['rect_max']

def set_viewport_clear_color(color: List[int]):
    if False:
        i = 10
        return i + 15
    "Sets the viewport's clear color.\n\n    Returns:\n        None\n    "
    internal_dpg.configure_viewport(0, clear_color=color)

def set_viewport_small_icon(icon: str):
    if False:
        while True:
            i = 10
    "Sets the viewport's small icon. Must be ico for windows.\n\n    Returns:\n        None\n    "
    internal_dpg.configure_viewport(0, small_icon=icon)

def set_viewport_large_icon(icon: str):
    if False:
        for i in range(10):
            print('nop')
    "Sets the viewport's large icon. Must be ico for windows.\n\n    Returns:\n        None\n    "
    internal_dpg.configure_viewport(0, large_icon=icon)

def set_viewport_pos(pos: List[float]):
    if False:
        return 10
    "Sets the viewport's position.\n\n    Returns:\n        None\n    "
    internal_dpg.configure_viewport(0, x_pos=pos[0], y_pos=pos[1])

def set_viewport_width(width: int):
    if False:
        for i in range(10):
            print('nop')
    "Sets the viewport's width.\n\n    Returns:\n        None\n    "
    internal_dpg.configure_viewport(0, width=width)

def set_viewport_height(height: int):
    if False:
        print('Hello World!')
    "Sets the viewport's height.\n\n    Returns:\n        None\n    "
    internal_dpg.configure_viewport(0, height=height)

def set_viewport_min_width(width: int):
    if False:
        while True:
            i = 10
    "Sets the viewport's minimum width.\n\n    Returns:\n        None\n    "
    internal_dpg.configure_viewport(0, min_width=width)

def set_viewport_max_width(width: int):
    if False:
        return 10
    "Sets the viewport's max width.\n\n    Returns:\n        None\n    "
    internal_dpg.configure_viewport(0, max_width=width)

def set_viewport_min_height(height: int):
    if False:
        for i in range(10):
            print('nop')
    "Sets the viewport's minimum height.\n\n    Returns:\n        None\n    "
    internal_dpg.configure_viewport(0, min_height=height)

def set_viewport_max_height(height: int):
    if False:
        return 10
    "Sets the viewport's max width.\n\n    Returns:\n        None\n    "
    internal_dpg.configure_viewport(0, max_height=height)

def set_viewport_title(title: str):
    if False:
        while True:
            i = 10
    "Sets the viewport's title.\n\n    Returns:\n        None\n    "
    internal_dpg.configure_viewport(0, title=title)

def set_viewport_always_top(value: bool):
    if False:
        i = 10
        return i + 15
    'Sets the viewport always on top.\n\n    Returns:\n        None\n    '
    internal_dpg.configure_viewport(0, always_on_top=value)

def set_viewport_resizable(value: bool):
    if False:
        for i in range(10):
            print('nop')
    'Sets the viewport resizable.\n\n    Returns:\n        None\n    '
    internal_dpg.configure_viewport(0, resizable=value)

def set_viewport_vsync(value: bool):
    if False:
        for i in range(10):
            print('nop')
    'Sets the viewport vsync.\n\n    Returns:\n        None\n    '
    internal_dpg.configure_viewport(0, vsync=value)

def set_viewport_decorated(value: bool):
    if False:
        for i in range(10):
            print('nop')
    'Sets the viewport to be decorated.\n\n    Returns:\n        None\n    '
    internal_dpg.configure_viewport(0, decorated=value)

def get_viewport_clear_color() -> List[int]:
    if False:
        i = 10
        return i + 15
    "Gets the viewport's clear color.\n\n    Returns:\n        List[int]\n    "
    return internal_dpg.get_viewport_configuration()['clear_color']

def get_viewport_pos() -> List[float]:
    if False:
        i = 10
        return i + 15
    "Gets the viewport's position.\n\n    Returns:\n        viewport position.\n    "
    config = internal_dpg.get_viewport_configuration()
    x_pos = config['x_pos']
    y_pos = config['y_pos']
    return [x_pos, y_pos]

def get_viewport_width() -> int:
    if False:
        return 10
    "Gets the viewport's width.\n\n    Returns:\n        viewport width\n    "
    return internal_dpg.get_viewport_configuration()['width']

def get_viewport_client_width() -> int:
    if False:
        i = 10
        return i + 15
    "Gets the viewport's client width.\n\n    Returns:\n        viewport width\n    "
    return internal_dpg.get_viewport_configuration()['client_width']

def get_viewport_client_height() -> int:
    if False:
        for i in range(10):
            print('nop')
    "Gets the viewport's client height.\n\n    Returns:\n        viewport width\n    "
    return internal_dpg.get_viewport_configuration()['client_height']

def get_viewport_height() -> int:
    if False:
        print('Hello World!')
    "Gets the viewport's height.\n\n    Returns:\n        int\n    "
    return internal_dpg.get_viewport_configuration()['height']

def get_viewport_min_width() -> int:
    if False:
        return 10
    "Gets the viewport's minimum width.\n\n    Returns:\n        int\n    "
    return internal_dpg.get_viewport_configuration()['min_width']

def get_viewport_max_width() -> int:
    if False:
        return 10
    "Gets the viewport's max width.\n\n    Returns:\n        int\n    "
    return internal_dpg.get_viewport_configuration()['max_width']

def get_viewport_min_height() -> int:
    if False:
        for i in range(10):
            print('nop')
    "Gets the viewport's minimum height.\n\n    Returns:\n        int\n    "
    return internal_dpg.get_viewport_configuration()['min_height']

def get_viewport_max_height() -> int:
    if False:
        return 10
    "Gets the viewport's max width.\n\n    Returns:\n        int\n    "
    return internal_dpg.get_viewport_configuration()['max_height']

def get_viewport_title() -> str:
    if False:
        for i in range(10):
            print('nop')
    "Gets the viewport's title.\n\n    Returns:\n        str\n    "
    return internal_dpg.get_viewport_configuration()['title']

def is_viewport_always_top() -> bool:
    if False:
        while True:
            i = 10
    'Checks the viewport always on top flag.\n\n    Returns:\n        bool\n    '
    return internal_dpg.get_viewport_configuration()['always_on_top']

def is_viewport_resizable() -> bool:
    if False:
        i = 10
        return i + 15
    'Checks the viewport resizable flag.\n\n    Returns:\n        bool\n    '
    return internal_dpg.get_viewport_configuration()['resizable']

def is_viewport_vsync_on() -> bool:
    if False:
        print('Hello World!')
    'Checks the viewport vsync flag.\n\n    Returns:\n        bool\n    '
    return internal_dpg.get_viewport_configuration()['vsync']

def is_viewport_decorated() -> bool:
    if False:
        print('Hello World!')
    'Checks if the viewport is docorated.\n\n    Returns:\n        bool\n    '
    return internal_dpg.get_viewport_configuration()['decorated']

def deprecated(reason):
    if False:
        for i in range(10):
            print('nop')
    string_types = (type(b''), type(u''))
    if isinstance(reason, string_types):

        def decorator(func1):
            if False:
                print('Hello World!')
            fmt1 = 'Call to deprecated function {name} ({reason}).'

            @functools.wraps(func1)
            def new_func1(*args, **kwargs):
                if False:
                    print('Hello World!')
                warnings.simplefilter('always', DeprecationWarning)
                warnings.warn(fmt1.format(name=func1.__name__, reason=reason), category=DeprecationWarning, stacklevel=2)
                warnings.simplefilter('default', DeprecationWarning)
                return func1(*args, **kwargs)
            return new_func1
        return decorator
    elif inspect.isfunction(reason):
        func2 = reason
        fmt2 = 'Call to deprecated function {name}.'

        @functools.wraps(func2)
        def new_func2(*args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            warnings.simplefilter('always', DeprecationWarning)
            warnings.warn(fmt2.format(name=func2.__name__), category=DeprecationWarning, stacklevel=2)
            warnings.simplefilter('default', DeprecationWarning)
            return func2(*args, **kwargs)
        return new_func2

@deprecated("Use 'configure_app(docking=True, docking_space=dock_space)'.")
def enable_docking(dock_space=False):
    if False:
        for i in range(10):
            print('nop')
    ' deprecated function '
    internal_dpg.configure_app(docking=True, docking_space=dock_space)

@deprecated("Use 'configure_app(init_file=file)'.")
def set_init_file(file='dpg.ini'):
    if False:
        i = 10
        return i + 15
    ' deprecated function '
    internal_dpg.configure_app(init_file=file)

@deprecated("Use 'configure_app(init_file=file, load_init_file=True)'.")
def load_init_file(file):
    if False:
        for i in range(10):
            print('nop')
    ' deprecated function '
    internal_dpg.configure_app(init_file=file, load_init_file=True)

@deprecated('Use: `is_viewport_ok(...)`')
def is_viewport_created():
    if False:
        i = 10
        return i + 15
    ' deprecated function '
    return internal_dpg.is_viewport_ok()

@deprecated('Use: \ncreate_viewport()\nsetup_dearpygui()\nshow_viewport()')
def setup_viewport():
    if False:
        print('Hello World!')
    ' deprecated function '
    internal_dpg.create_viewport()
    internal_dpg.setup_dearpygui()
    internal_dpg.show_viewport()

@deprecated('Use: `bind_item_theme(...)`')
def set_item_theme(item, theme):
    if False:
        print('Hello World!')
    ' deprecated function '
    return internal_dpg.bind_item_theme(item, theme)

@deprecated('Use: `bind_item_type_disabled_theme(...)`')
def set_item_type_disabled_theme(item, theme):
    if False:
        i = 10
        return i + 15
    ' deprecated function '
    return internal_dpg.bind_item_type_disabled_theme(item, theme)

@deprecated('Use: `bind_item_type_theme(...)`')
def set_item_type_theme(item, theme):
    if False:
        i = 10
        return i + 15
    ' deprecated function '
    return internal_dpg.bind_item_type_theme(item, theme)

@deprecated('Use: `bind_item_font(...)`')
def set_item_font(item, font):
    if False:
        print('Hello World!')
    ' deprecated function '
    return internal_dpg.bind_item_font(item, font)

@deprecated('Use: `add_item_activated_handler(...)`')
def add_activated_handler(parent, **kwargs):
    if False:
        while True:
            i = 10
    ' deprecated function '
    return internal_dpg.add_item_activated_handler(parent, **kwargs)

@deprecated('Use: `add_item_active_handler(...)`')
def add_active_handler(parent, **kwargs):
    if False:
        while True:
            i = 10
    ' deprecated function '
    return internal_dpg.add_item_active_handler(parent, **kwargs)

@deprecated('Use: `add_item_clicked_handler(...)`')
def add_clicked_handler(parent, button=-1, **kwargs):
    if False:
        return 10
    ' deprecated function '
    return internal_dpg.add_item_clicked_handler(parent, button, **kwargs)

@deprecated('Use: `add_item_deactived_after_edit_handler(...)`')
def add_deactivated_after_edit_handler(parent, **kwargs):
    if False:
        while True:
            i = 10
    ' deprecated function '
    return internal_dpg.add_item_deactivated_after_edit_handler(parent, **kwargs)

@deprecated('Use: `add_item_deactivated_handler(...)`')
def add_deactivated_handler(parent, **kwargs):
    if False:
        while True:
            i = 10
    ' deprecated function '
    return internal_dpg.add_item_deactivated_handler(parent, **kwargs)

@deprecated('Use: `add_item_edited_handler(...)`')
def add_edited_handler(parent, **kwargs):
    if False:
        while True:
            i = 10
    ' deprecated function '
    return internal_dpg.add_item_edited_handler(parent, **kwargs)

@deprecated('Use: `add_item_focus_handler(...)`')
def add_focus_handler(parent, **kwargs):
    if False:
        i = 10
        return i + 15
    ' deprecated function '
    return internal_dpg.add_item_focus_handler(parent, **kwargs)

@deprecated('Use: `add_item_hover_handler(...)`')
def add_hover_handler(parent, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    ' deprecated function '
    return internal_dpg.add_item_hover_handler(parent, **kwargs)

@deprecated('Use: `add_item_resize_handler(...)`')
def add_resize_handler(parent, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    ' deprecated function '
    return internal_dpg.add_item_resize_handler(parent, **kwargs)

@deprecated('Use: `add_item_toggled_open_handler(...)`')
def add_toggled_open_handler(parent, **kwargs):
    if False:
        return 10
    ' deprecated function '
    return internal_dpg.add_item_toggled_open_handler(parent, **kwargs)

@deprecated('Use: `add_item_visible_handler(...)`')
def add_visible_handler(parent, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    ' deprecated function '
    return internal_dpg.add_item_visible_handler(parent, **kwargs)

@deprecated('Use: `bind_colormap(...)`')
def set_colormap(item, source):
    if False:
        while True:
            i = 10
    ' deprecated function '
    return internal_dpg.bind_colormap(item, source)

@deprecated('Use: `bind_theme(0)`')
def reset_default_theme(item, source):
    if False:
        while True:
            i = 10
    ' deprecated function '
    return internal_dpg.bind_theme(item, source)

@deprecated
def set_staging_mode(mode):
    if False:
        i = 10
        return i + 15
    ' deprecated function '
    pass

@deprecated
def add_table_next_column(**kwargs):
    if False:
        while True:
            i = 10
    ' deprecated function '
    pass

@deprecated('Use: add_stage')
def add_staging_container(**kwargs):
    if False:
        i = 10
        return i + 15
    ' deprecated function '
    return internal_dpg.add_stage(**kwargs)

@deprecated('Use: stage')
@contextmanager
def staging_container(**kwargs):
    if False:
        return 10
    "\n\tdeprecated function\n\tArgs:\n\t\t**label (str): Overrides 'name' as label.\n\t\t**user_data (Any): User data for callbacks.\n\t\t**use_internal_label (bool): Use generated internal label instead of user specified (appends ### uuid).\n\t\t**id (Union[int, str]): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        warnings.warn("'staging_container' is deprecated and was changed to 'stage'", DeprecationWarning, 2)
        widget = internal_dpg.add_stage_container(**kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@deprecated('Use: add_spacer(...)')
def add_spacing(**kwargs):
    if False:
        return 10
    "\t(deprecated function) Adds vertical spacing. \n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks.\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tcount (int, optional): Number of spacings to add the size is dependant on the curret style.\n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'count' in kwargs.keys():
        count = kwargs['count']
        kwargs.pop('count', None)
        internal_dpg.add_group(**kwargs)
        internal_dpg.push_container_stack(internal_dpg.last_container())
        for i in range(count):
            internal_dpg.add_spacer()
        result_id = internal_dpg.pop_container_stack()
    else:
        result_id = internal_dpg.add_spacer(**kwargs)
    return result_id

@deprecated('Use: add_spacer(...)')
def add_dummy(**kwargs):
    if False:
        i = 10
        return i + 15
    "\t(deprecated function) Adds a spacer or 'dummy' object.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks.\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_spacer(**kwargs)

@deprecated('Use: `destroy_context()`')
def cleanup_dearpygui():
    if False:
        for i in range(10):
            print('nop')
    ' deprecated function '
    return internal_dpg.destroy_context()

@deprecated('Use: group(horizontal=True)')
def add_same_line(**kwargs):
    if False:
        while True:
            i = 10
    ' deprecated function '
    last_item = internal_dpg.last_item()
    group = internal_dpg.add_group(horizontal=True, **kwargs)
    internal_dpg.move_item(last_item, parent=group)
    internal_dpg.capture_next_item(lambda s: internal_dpg.move_item(s, parent=group))
    return group

@deprecated('Use: `add_child_window()`')
def add_child(**kwargs):
    if False:
        return 10
    "\t(deprecated function) Adds an embedded child window. Will show scrollbars when items do not fit.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tborder (bool, optional): Shows/Hides the border around the sides.\n\t\tautosize_x (bool, optional): Autosize the window to its parents size in x.\n\t\tautosize_y (bool, optional): Autosize the window to its parents size in y.\n\t\tno_scrollbar (bool, optional):  Disable scrollbars (window can still scroll with mouse or programmatically).\n\t\thorizontal_scrollbar (bool, optional): Allow horizontal scrollbar to appear (off by default).\n\t\tmenubar (bool, optional): Shows/Hides the menubar at the top.\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_child_window(**kwargs)

@deprecated('Use: `child_window()`')
@contextmanager
def child(**kwargs):
    if False:
        i = 10
        return i + 15
    "\t(deprecated function) Adds an embedded child window. Will show scrollbars when items do not fit.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tborder (bool, optional): Shows/Hides the border around the sides.\n\t\tautosize_x (bool, optional): Autosize the window to its parents size in x.\n\t\tautosize_y (bool, optional): Autosize the window to its parents size in y.\n\t\tno_scrollbar (bool, optional):  Disable scrollbars (window can still scroll with mouse or programmatically).\n\t\thorizontal_scrollbar (bool, optional): Allow horizontal scrollbar to appear (off by default).\n\t\tmenubar (bool, optional): Shows/Hides the menubar at the top.\n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        widget = internal_dpg.add_child_window(**kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@deprecated('Use: Just not recommended')
def setup_registries() -> None:
    if False:
        for i in range(10):
            print('nop')
    'Adds default registries for fonts, handlers, textures, colormaps, and values.'
    internal_dpg.add_font_registry(tag=internal_dpg.mvReservedUUID_0)
    internal_dpg.add_handler_registry(tag=internal_dpg.mvReservedUUID_1)
    internal_dpg.add_texture_registry(tag=internal_dpg.mvReservedUUID_2)
    internal_dpg.add_value_registry(tag=internal_dpg.mvReservedUUID_3)
    internal_dpg.add_colormap_registry(tag=internal_dpg.mvReservedUUID_4)

@deprecated('Use: `set_frame_callback()`')
def set_start_callback(callback):
    if False:
        while True:
            i = 10
    ' deprecated function '
    return internal_dpg.set_frame_callback(3, callback)

@contextmanager
def child_window(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, width: int=0, height: int=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', drop_callback: Callable=None, show: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], filter_key: str='', delay_search: bool=False, tracked: bool=False, track_offset: float=0.5, border: bool=True, autosize_x: bool=False, autosize_y: bool=False, no_scrollbar: bool=False, horizontal_scrollbar: bool=False, menubar: bool=False, no_scroll_with_mouse: bool=False, **kwargs) -> Union[int, str]:
    if False:
        i = 10
        return i + 15
    "\t Adds an embedded child window. Will show scrollbars when items do not fit.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tborder (bool, optional): Shows/Hides the border around the sides.\n\t\tautosize_x (bool, optional): Autosize the window to its parents size in x.\n\t\tautosize_y (bool, optional): Autosize the window to its parents size in y.\n\t\tno_scrollbar (bool, optional):  Disable scrollbars (window can still scroll with mouse or programmatically).\n\t\thorizontal_scrollbar (bool, optional): Allow horizontal scrollbar to appear (off by default).\n\t\tmenubar (bool, optional): Shows/Hides the menubar at the top.\n\t\tno_scroll_with_mouse (bool, optional): Disable user vertically scrolling with mouse wheel.\n\t\tid (Union[int, str], optional): (deprecated) \n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        if 'id' in kwargs.keys():
            warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
            tag = kwargs['id']
        widget = internal_dpg.add_child_window(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, width=width, height=height, indent=indent, parent=parent, before=before, payload_type=payload_type, drop_callback=drop_callback, show=show, pos=pos, filter_key=filter_key, delay_search=delay_search, tracked=tracked, track_offset=track_offset, border=border, autosize_x=autosize_x, autosize_y=autosize_y, no_scrollbar=no_scrollbar, horizontal_scrollbar=horizontal_scrollbar, menubar=menubar, no_scroll_with_mouse=no_scroll_with_mouse, **kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def clipper(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, width: int=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, show: bool=True, delay_search: bool=False, **kwargs) -> Union[int, str]:
    if False:
        print('Hello World!')
    "\t Helper to manually clip large list of items. Increases performance by not searching or drawing widgets outside of the clipped region.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\tid (Union[int, str], optional): (deprecated) \n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        if 'id' in kwargs.keys():
            warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
            tag = kwargs['id']
        widget = internal_dpg.add_clipper(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, width=width, indent=indent, parent=parent, before=before, show=show, delay_search=delay_search, **kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def collapsing_header(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', drag_callback: Callable=None, drop_callback: Callable=None, show: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], filter_key: str='', delay_search: bool=False, tracked: bool=False, track_offset: float=0.5, closable: bool=False, default_open: bool=False, open_on_double_click: bool=False, open_on_arrow: bool=False, leaf: bool=False, bullet: bool=False, **kwargs) -> Union[int, str]:
    if False:
        i = 10
        return i + 15
    "\t Adds a collapsing header to add items to. Must be closed with the end command.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tclosable (bool, optional): Adds the ability to hide this widget by pressing the (x) in the top right of widget.\n\t\tdefault_open (bool, optional): Sets the collapseable header open by default.\n\t\topen_on_double_click (bool, optional): Need double-click to open node.\n\t\topen_on_arrow (bool, optional): Only open when clicking on the arrow part.\n\t\tleaf (bool, optional): No collapsing, no arrow (use as a convenience for leaf nodes).\n\t\tbullet (bool, optional): Display a bullet instead of arrow.\n\t\tid (Union[int, str], optional): (deprecated) \n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        if 'id' in kwargs.keys():
            warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
            tag = kwargs['id']
        widget = internal_dpg.add_collapsing_header(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, indent=indent, parent=parent, before=before, payload_type=payload_type, drag_callback=drag_callback, drop_callback=drop_callback, show=show, pos=pos, filter_key=filter_key, delay_search=delay_search, tracked=tracked, track_offset=track_offset, closable=closable, default_open=default_open, open_on_double_click=open_on_double_click, open_on_arrow=open_on_arrow, leaf=leaf, bullet=bullet, **kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def colormap_registry(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, show: bool=False, **kwargs) -> Union[int, str]:
    if False:
        i = 10
        return i + 15
    "\t Adds a colormap registry.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated) \n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        if 'id' in kwargs.keys():
            warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
            tag = kwargs['id']
        widget = internal_dpg.add_colormap_registry(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, show=show, **kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def custom_series(x: Union[List[float], Tuple[float, ...]], y: Union[List[float], Tuple[float, ...]], channel_count: int, *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, before: Union[int, str]=0, source: Union[int, str]=0, callback: Callable=None, show: bool=True, y1: Any=[], y2: Any=[], y3: Any=[], tooltip: bool=True, **kwargs) -> Union[int, str]:
    if False:
        return 10
    "\t Adds a custom series to a plot. New in 1.6.\n\n\tArgs:\n\t\tx (Any): \n\t\ty (Any): \n\t\tchannel_count (int): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\ty1 (Any, optional): \n\t\ty2 (Any, optional): \n\t\ty3 (Any, optional): \n\t\ttooltip (bool, optional): Show tooltip when plot is hovered.\n\t\tid (Union[int, str], optional): (deprecated) \n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        if 'id' in kwargs.keys():
            warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
            tag = kwargs['id']
        widget = internal_dpg.add_custom_series(x, y, channel_count, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, before=before, source=source, callback=callback, show=show, y1=y1, y2=y2, y3=y3, tooltip=tooltip, **kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def drag_payload(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, show: bool=True, drag_data: Any=None, drop_data: Any=None, payload_type: str='$$DPG_PAYLOAD', **kwargs) -> Union[int, str]:
    if False:
        for i in range(10):
            print('nop')
    "\t User data payload for drag and drop operations.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tdrag_data (Any, optional): Drag data\n\t\tdrop_data (Any, optional): Drop data\n\t\tpayload_type (str, optional): \n\t\tid (Union[int, str], optional): (deprecated) \n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        if 'id' in kwargs.keys():
            warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
            tag = kwargs['id']
        widget = internal_dpg.add_drag_payload(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, show=show, drag_data=drag_data, drop_data=drop_data, payload_type=payload_type, **kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def draw_layer(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, before: Union[int, str]=0, show: bool=True, perspective_divide: bool=False, depth_clipping: bool=False, cull_mode: int=0, **kwargs) -> Union[int, str]:
    if False:
        for i in range(10):
            print('nop')
    "\t New in 1.1. Creates a layer useful for grouping drawlist items.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tperspective_divide (bool, optional): New in 1.1. apply perspective divide\n\t\tdepth_clipping (bool, optional): New in 1.1. apply depth clipping\n\t\tcull_mode (int, optional): New in 1.1. culling mode, mvCullMode_* constants. Only works with triangles currently.\n\t\tid (Union[int, str], optional): (deprecated) \n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        if 'id' in kwargs.keys():
            warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
            tag = kwargs['id']
        widget = internal_dpg.add_draw_layer(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, before=before, show=show, perspective_divide=perspective_divide, depth_clipping=depth_clipping, cull_mode=cull_mode, **kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def draw_node(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, before: Union[int, str]=0, show: bool=True, **kwargs) -> Union[int, str]:
    if False:
        for i in range(10):
            print('nop')
    "\t New in 1.1. Creates a drawing node to associate a transformation matrix. Child node matricies will concatenate.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated) \n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        if 'id' in kwargs.keys():
            warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
            tag = kwargs['id']
        widget = internal_dpg.add_draw_node(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, before=before, show=show, **kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def drawlist(width: int, height: int, *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, before: Union[int, str]=0, callback: Callable=None, show: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], filter_key: str='', delay_search: bool=False, tracked: bool=False, track_offset: float=0.5, **kwargs) -> Union[int, str]:
    if False:
        while True:
            i = 10
    "\t Adds a drawing canvas.\n\n\tArgs:\n\t\twidth (int): \n\t\theight (int): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tid (Union[int, str], optional): (deprecated) \n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        if 'id' in kwargs.keys():
            warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
            tag = kwargs['id']
        widget = internal_dpg.add_drawlist(width, height, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, before=before, callback=callback, show=show, pos=pos, filter_key=filter_key, delay_search=delay_search, tracked=tracked, track_offset=track_offset, **kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def file_dialog(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, width: int=0, height: int=0, callback: Callable=None, show: bool=True, default_path: str='', default_filename: str='.', file_count: int=0, modal: bool=False, directory_selector: bool=False, min_size: Union[List[int], Tuple[int, ...]]=[100, 100], max_size: Union[List[int], Tuple[int, ...]]=[30000, 30000], cancel_callback: Callable=None, **kwargs) -> Union[int, str]:
    if False:
        print('Hello World!')
    "\t Displays a file or directory selector depending on keywords. Displays a file dialog by default. Callback will be ran when the file or directory picker is closed. The app_data arguemnt will be populated with information related to the file and directory as a dictionary.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tdefault_path (str, optional): Path that the file dialog will default to when opened.\n\t\tdefault_filename (str, optional): Default name that will show in the file name input.\n\t\tfile_count (int, optional): Number of visible files in the dialog.\n\t\tmodal (bool, optional): Forces user interaction with the file selector.\n\t\tdirectory_selector (bool, optional): Shows only directory/paths as options. Allows selection of directory/paths only.\n\t\tmin_size (Union[List[int], Tuple[int, ...]], optional): Minimum window size.\n\t\tmax_size (Union[List[int], Tuple[int, ...]], optional): Maximum window size.\n\t\tcancel_callback (Callable, optional): Callback called when cancel button is clicked.\n\t\tid (Union[int, str], optional): (deprecated) \n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        if 'id' in kwargs.keys():
            warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
            tag = kwargs['id']
        widget = internal_dpg.add_file_dialog(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, width=width, height=height, callback=callback, show=show, default_path=default_path, default_filename=default_filename, file_count=file_count, modal=modal, directory_selector=directory_selector, min_size=min_size, max_size=max_size, cancel_callback=cancel_callback, **kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def filter_set(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, width: int=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, show: bool=True, delay_search: bool=False, **kwargs) -> Union[int, str]:
    if False:
        while True:
            i = 10
    "\t Helper to parse and apply text filters (e.g. aaaaa[, bbbbb][, ccccc])\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\tid (Union[int, str], optional): (deprecated) \n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        if 'id' in kwargs.keys():
            warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
            tag = kwargs['id']
        widget = internal_dpg.add_filter_set(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, width=width, indent=indent, parent=parent, before=before, show=show, delay_search=delay_search, **kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def font(file: str, size: int, *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, pixel_snapH: bool=False, parent: Union[int, str]=internal_dpg.mvReservedUUID_0, **kwargs) -> Union[int, str]:
    if False:
        i = 10
        return i + 15
    "\t Adds font to a font registry.\n\n\tArgs:\n\t\tfile (str): \n\t\tsize (int): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tpixel_snapH (bool, optional): Align every glyph to pixel boundary. Useful e.g. if you are merging a non-pixel aligned font with the default font, or rendering text piece-by-piece (e.g. for coloring).\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated) \n\t\tdefault_font (bool, optional): (deprecated) \n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        if 'id' in kwargs.keys():
            warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
            tag = kwargs['id']
        if 'default_font' in kwargs.keys():
            warnings.warn('default_font keyword removed', DeprecationWarning, 2)
            kwargs.pop('default_font', None)
        widget = internal_dpg.add_font(file, size, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, pixel_snapH=pixel_snapH, parent=parent, **kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def font_registry(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, show: bool=True, **kwargs) -> Union[int, str]:
    if False:
        print('Hello World!')
    "\t Adds a font registry.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated) \n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        if 'id' in kwargs.keys():
            warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
            tag = kwargs['id']
        widget = internal_dpg.add_font_registry(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, show=show, **kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def group(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, width: int=0, height: int=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', drag_callback: Callable=None, drop_callback: Callable=None, show: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], filter_key: str='', delay_search: bool=False, tracked: bool=False, track_offset: float=0.5, horizontal: bool=False, horizontal_spacing: float=-1, xoffset: float=0.0, **kwargs) -> Union[int, str]:
    if False:
        return 10
    "\t Creates a group that other widgets can belong to. The group allows item commands to be issued for all of its members.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\thorizontal (bool, optional): Forces child widgets to be added in a horizontal layout.\n\t\thorizontal_spacing (float, optional): Spacing for the horizontal layout.\n\t\txoffset (float, optional): Offset from containing window x item location within group.\n\t\tid (Union[int, str], optional): (deprecated) \n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        if 'id' in kwargs.keys():
            warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
            tag = kwargs['id']
        widget = internal_dpg.add_group(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, width=width, height=height, indent=indent, parent=parent, before=before, payload_type=payload_type, drag_callback=drag_callback, drop_callback=drop_callback, show=show, pos=pos, filter_key=filter_key, delay_search=delay_search, tracked=tracked, track_offset=track_offset, horizontal=horizontal, horizontal_spacing=horizontal_spacing, xoffset=xoffset, **kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def handler_registry(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, show: bool=True, **kwargs) -> Union[int, str]:
    if False:
        for i in range(10):
            print('nop')
    "\t Adds a handler registry.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated) \n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        if 'id' in kwargs.keys():
            warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
            tag = kwargs['id']
        widget = internal_dpg.add_handler_registry(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, show=show, **kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def item_handler_registry(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, show: bool=True, **kwargs) -> Union[int, str]:
    if False:
        i = 10
        return i + 15
    "\t Adds an item handler registry.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated) \n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        if 'id' in kwargs.keys():
            warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
            tag = kwargs['id']
        widget = internal_dpg.add_item_handler_registry(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, show=show, **kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def menu(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', drop_callback: Callable=None, show: bool=True, enabled: bool=True, filter_key: str='', delay_search: bool=False, tracked: bool=False, track_offset: float=0.5, **kwargs) -> Union[int, str]:
    if False:
        i = 10
        return i + 15
    "\t Adds a menu to an existing menu bar.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tid (Union[int, str], optional): (deprecated) \n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        if 'id' in kwargs.keys():
            warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
            tag = kwargs['id']
        widget = internal_dpg.add_menu(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, indent=indent, parent=parent, before=before, payload_type=payload_type, drop_callback=drop_callback, show=show, enabled=enabled, filter_key=filter_key, delay_search=delay_search, tracked=tracked, track_offset=track_offset, **kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def menu_bar(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, indent: int=-1, parent: Union[int, str]=0, show: bool=True, delay_search: bool=False, **kwargs) -> Union[int, str]:
    if False:
        i = 10
        return i + 15
    "\t Adds a menu bar to a window.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\tid (Union[int, str], optional): (deprecated) \n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        if 'id' in kwargs.keys():
            warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
            tag = kwargs['id']
        widget = internal_dpg.add_menu_bar(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, indent=indent, parent=parent, show=show, delay_search=delay_search, **kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def node(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, before: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', drag_callback: Callable=None, drop_callback: Callable=None, show: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], filter_key: str='', delay_search: bool=False, tracked: bool=False, track_offset: float=0.5, draggable: bool=True, **kwargs) -> Union[int, str]:
    if False:
        i = 10
        return i + 15
    "\t Adds a node to a node editor.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdraggable (bool, optional): Allow node to be draggable.\n\t\tid (Union[int, str], optional): (deprecated) \n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        if 'id' in kwargs.keys():
            warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
            tag = kwargs['id']
        widget = internal_dpg.add_node(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, before=before, payload_type=payload_type, drag_callback=drag_callback, drop_callback=drop_callback, show=show, pos=pos, filter_key=filter_key, delay_search=delay_search, tracked=tracked, track_offset=track_offset, draggable=draggable, **kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def node_attribute(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, show: bool=True, filter_key: str='', tracked: bool=False, track_offset: float=0.5, attribute_type: int=0, shape: int=1, category: str='general', **kwargs) -> Union[int, str]:
    if False:
        i = 10
        return i + 15
    "\t Adds a node attribute to a node.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tattribute_type (int, optional): mvNode_Attr_Input, mvNode_Attr_Output, or mvNode_Attr_Static.\n\t\tshape (int, optional): Pin shape.\n\t\tcategory (str, optional): Category\n\t\tid (Union[int, str], optional): (deprecated) \n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        if 'id' in kwargs.keys():
            warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
            tag = kwargs['id']
        widget = internal_dpg.add_node_attribute(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, indent=indent, parent=parent, before=before, show=show, filter_key=filter_key, tracked=tracked, track_offset=track_offset, attribute_type=attribute_type, shape=shape, category=category, **kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def node_editor(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, width: int=0, height: int=0, parent: Union[int, str]=0, before: Union[int, str]=0, callback: Callable=None, show: bool=True, filter_key: str='', delay_search: bool=False, tracked: bool=False, track_offset: float=0.5, delink_callback: Callable=None, menubar: bool=False, minimap: bool=False, minimap_location: int=2, **kwargs) -> Union[int, str]:
    if False:
        for i in range(10):
            print('nop')
    "\t Adds a node editor.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdelink_callback (Callable, optional): Callback ran when a link is detached.\n\t\tmenubar (bool, optional): Shows or hides the menubar.\n\t\tminimap (bool, optional): Shows or hides the Minimap. New in 1.6.\n\t\tminimap_location (int, optional): mvNodeMiniMap_Location_* constants. New in 1.6.\n\t\tid (Union[int, str], optional): (deprecated) \n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        if 'id' in kwargs.keys():
            warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
            tag = kwargs['id']
        widget = internal_dpg.add_node_editor(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, width=width, height=height, parent=parent, before=before, callback=callback, show=show, filter_key=filter_key, delay_search=delay_search, tracked=tracked, track_offset=track_offset, delink_callback=delink_callback, menubar=menubar, minimap=minimap, minimap_location=minimap_location, **kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def plot(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, width: int=0, height: int=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', callback: Callable=None, drag_callback: Callable=None, drop_callback: Callable=None, show: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], filter_key: str='', delay_search: bool=False, tracked: bool=False, track_offset: float=0.5, no_title: bool=False, no_menus: bool=False, no_box_select: bool=False, no_mouse_pos: bool=False, no_highlight: bool=False, no_child: bool=False, query: bool=False, crosshairs: bool=False, anti_aliased: bool=False, equal_aspects: bool=False, use_local_time: bool=False, use_ISO8601: bool=False, use_24hour_clock: bool=False, pan_button: int=internal_dpg.mvMouseButton_Left, pan_mod: int=-1, fit_button: int=internal_dpg.mvMouseButton_Left, context_menu_button: int=internal_dpg.mvMouseButton_Right, box_select_button: int=internal_dpg.mvMouseButton_Right, box_select_mod: int=-1, box_select_cancel_button: int=internal_dpg.mvMouseButton_Left, query_button: int=internal_dpg.mvMouseButton_Middle, query_mod: int=-1, query_toggle_mod: int=internal_dpg.mvKey_Control, horizontal_mod: int=internal_dpg.mvKey_Alt, vertical_mod: int=internal_dpg.mvKey_Shift, **kwargs) -> Union[int, str]:
    if False:
        while True:
            i = 10
    "\t Adds a plot which is used to hold series, and can be drawn to with draw commands.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tno_title (bool, optional): the plot title will not be displayed\n\t\tno_menus (bool, optional): the user will not be able to open context menus with right-click\n\t\tno_box_select (bool, optional): the user will not be able to box-select with right-click drag\n\t\tno_mouse_pos (bool, optional): the mouse position, in plot coordinates, will not be displayed inside of the plot\n\t\tno_highlight (bool, optional): plot items will not be highlighted when their legend entry is hovered\n\t\tno_child (bool, optional): a child window region will not be used to capture mouse scroll (can boost performance for single ImGui window applications)\n\t\tquery (bool, optional): the user will be able to draw query rects with middle - mouse or CTRL + right - click drag\n\t\tcrosshairs (bool, optional): the default mouse cursor will be replaced with a crosshair when hovered\n\t\tanti_aliased (bool, optional): plot lines will be software anti-aliased (not recommended for high density plots, prefer MSAA)\n\t\tequal_aspects (bool, optional): primary x and y axes will be constrained to have the same units/pixel (does not apply to auxiliary y-axes)\n\t\tuse_local_time (bool, optional): axis labels will be formatted for your timezone when\n\t\tuse_ISO8601 (bool, optional): dates will be formatted according to ISO 8601 where applicable (e.g. YYYY-MM-DD, YYYY-MM, --MM-DD, etc.)\n\t\tuse_24hour_clock (bool, optional): times will be formatted using a 24 hour clock\n\t\tpan_button (int, optional): enables panning when held\n\t\tpan_mod (int, optional): optional modifier that must be held for panning\n\t\tfit_button (int, optional): fits visible data when double clicked\n\t\tcontext_menu_button (int, optional): opens plot context menu (if enabled) when clicked\n\t\tbox_select_button (int, optional): begins box selection when pressed and confirms selection when released\n\t\tbox_select_mod (int, optional): begins box selection when pressed and confirms selection when released\n\t\tbox_select_cancel_button (int, optional): cancels active box selection when pressed\n\t\tquery_button (int, optional): begins query selection when pressed and end query selection when released\n\t\tquery_mod (int, optional): optional modifier that must be held for query selection\n\t\tquery_toggle_mod (int, optional): when held, active box selections turn into queries\n\t\thorizontal_mod (int, optional): expands active box selection/query horizontally to plot edge when held\n\t\tvertical_mod (int, optional): expands active box selection/query vertically to plot edge when held\n\t\tid (Union[int, str], optional): (deprecated) \n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        if 'id' in kwargs.keys():
            warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
            tag = kwargs['id']
        widget = internal_dpg.add_plot(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, width=width, height=height, indent=indent, parent=parent, before=before, payload_type=payload_type, callback=callback, drag_callback=drag_callback, drop_callback=drop_callback, show=show, pos=pos, filter_key=filter_key, delay_search=delay_search, tracked=tracked, track_offset=track_offset, no_title=no_title, no_menus=no_menus, no_box_select=no_box_select, no_mouse_pos=no_mouse_pos, no_highlight=no_highlight, no_child=no_child, query=query, crosshairs=crosshairs, anti_aliased=anti_aliased, equal_aspects=equal_aspects, use_local_time=use_local_time, use_ISO8601=use_ISO8601, use_24hour_clock=use_24hour_clock, pan_button=pan_button, pan_mod=pan_mod, fit_button=fit_button, context_menu_button=context_menu_button, box_select_button=box_select_button, box_select_mod=box_select_mod, box_select_cancel_button=box_select_cancel_button, query_button=query_button, query_mod=query_mod, query_toggle_mod=query_toggle_mod, horizontal_mod=horizontal_mod, vertical_mod=vertical_mod, **kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def plot_axis(axis: int, *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', drop_callback: Callable=None, show: bool=True, no_gridlines: bool=False, no_tick_marks: bool=False, no_tick_labels: bool=False, log_scale: bool=False, invert: bool=False, lock_min: bool=False, lock_max: bool=False, time: bool=False, **kwargs) -> Union[int, str]:
    if False:
        for i in range(10):
            print('nop')
    "\t Adds an axis to a plot.\n\n\tArgs:\n\t\taxis (int): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tno_gridlines (bool, optional): \n\t\tno_tick_marks (bool, optional): \n\t\tno_tick_labels (bool, optional): \n\t\tlog_scale (bool, optional): \n\t\tinvert (bool, optional): \n\t\tlock_min (bool, optional): \n\t\tlock_max (bool, optional): \n\t\ttime (bool, optional): \n\t\tid (Union[int, str], optional): (deprecated) \n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        if 'id' in kwargs.keys():
            warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
            tag = kwargs['id']
        widget = internal_dpg.add_plot_axis(axis, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, payload_type=payload_type, drop_callback=drop_callback, show=show, no_gridlines=no_gridlines, no_tick_marks=no_tick_marks, no_tick_labels=no_tick_labels, log_scale=log_scale, invert=invert, lock_min=lock_min, lock_max=lock_max, time=time, **kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def stage(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, **kwargs) -> Union[int, str]:
    if False:
        while True:
            i = 10
    "\t Adds a stage.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tid (Union[int, str], optional): (deprecated) \n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        if 'id' in kwargs.keys():
            warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
            tag = kwargs['id']
        widget = internal_dpg.add_stage(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, **kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def subplots(rows: int, columns: int, *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, width: int=0, height: int=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, callback: Callable=None, show: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], filter_key: str='', delay_search: bool=False, tracked: bool=False, track_offset: float=0.5, row_ratios: Union[List[float], Tuple[float, ...]]=[], column_ratios: Union[List[float], Tuple[float, ...]]=[], no_title: bool=False, no_menus: bool=False, no_resize: bool=False, no_align: bool=False, link_rows: bool=False, link_columns: bool=False, link_all_x: bool=False, link_all_y: bool=False, column_major: bool=False, **kwargs) -> Union[int, str]:
    if False:
        while True:
            i = 10
    "\t Adds a collection of plots.\n\n\tArgs:\n\t\trows (int): \n\t\tcolumns (int): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\trow_ratios (Union[List[float], Tuple[float, ...]], optional): \n\t\tcolumn_ratios (Union[List[float], Tuple[float, ...]], optional): \n\t\tno_title (bool, optional): \n\t\tno_menus (bool, optional): the user will not be able to open context menus with right-click\n\t\tno_resize (bool, optional): resize splitters between subplot cells will be not be provided\n\t\tno_align (bool, optional): subplot edges will not be aligned vertically or horizontally\n\t\tlink_rows (bool, optional): link the y-axis limits of all plots in each row (does not apply auxiliary y-axes)\n\t\tlink_columns (bool, optional): link the x-axis limits of all plots in each column\n\t\tlink_all_x (bool, optional): link the x-axis limits in every plot in the subplot\n\t\tlink_all_y (bool, optional): link the y-axis limits in every plot in the subplot (does not apply to auxiliary y-axes)\n\t\tcolumn_major (bool, optional): subplots are added in column major order instead of the default row major order\n\t\tid (Union[int, str], optional): (deprecated) \n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        if 'id' in kwargs.keys():
            warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
            tag = kwargs['id']
        widget = internal_dpg.add_subplots(rows, columns, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, width=width, height=height, indent=indent, parent=parent, before=before, callback=callback, show=show, pos=pos, filter_key=filter_key, delay_search=delay_search, tracked=tracked, track_offset=track_offset, row_ratios=row_ratios, column_ratios=column_ratios, no_title=no_title, no_menus=no_menus, no_resize=no_resize, no_align=no_align, link_rows=link_rows, link_columns=link_columns, link_all_x=link_all_x, link_all_y=link_all_y, column_major=column_major, **kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def tab(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', drop_callback: Callable=None, show: bool=True, filter_key: str='', delay_search: bool=False, tracked: bool=False, track_offset: float=0.5, closable: bool=False, no_tooltip: bool=False, order_mode: bool=0, **kwargs) -> Union[int, str]:
    if False:
        while True:
            i = 10
    "\t Adds a tab to a tab bar.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tclosable (bool, optional): Creates a button on the tab that can hide the tab.\n\t\tno_tooltip (bool, optional): Disable tooltip for the given tab.\n\t\torder_mode (bool, optional): set using a constant: mvTabOrder_Reorderable: allows reordering, mvTabOrder_Fixed: fixed ordering, mvTabOrder_Leading: adds tab to front, mvTabOrder_Trailing: adds tab to back\n\t\tid (Union[int, str], optional): (deprecated) \n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        if 'id' in kwargs.keys():
            warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
            tag = kwargs['id']
        widget = internal_dpg.add_tab(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, indent=indent, parent=parent, before=before, payload_type=payload_type, drop_callback=drop_callback, show=show, filter_key=filter_key, delay_search=delay_search, tracked=tracked, track_offset=track_offset, closable=closable, no_tooltip=no_tooltip, order_mode=order_mode, **kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def tab_bar(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, callback: Callable=None, show: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], filter_key: str='', delay_search: bool=False, tracked: bool=False, track_offset: float=0.5, reorderable: bool=False, **kwargs) -> Union[int, str]:
    if False:
        i = 10
        return i + 15
    "\t Adds a tab bar.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\treorderable (bool, optional): Allows for the user to change the order of the tabs.\n\t\tid (Union[int, str], optional): (deprecated) \n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        if 'id' in kwargs.keys():
            warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
            tag = kwargs['id']
        widget = internal_dpg.add_tab_bar(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, indent=indent, parent=parent, before=before, callback=callback, show=show, pos=pos, filter_key=filter_key, delay_search=delay_search, tracked=tracked, track_offset=track_offset, reorderable=reorderable, **kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def table(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, width: int=0, height: int=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, source: Union[int, str]=0, callback: Callable=None, show: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], filter_key: str='', delay_search: bool=False, header_row: bool=True, clipper: bool=False, inner_width: int=0, policy: int=0, freeze_rows: int=0, freeze_columns: int=0, sort_multi: bool=False, sort_tristate: bool=False, resizable: bool=False, reorderable: bool=False, hideable: bool=False, sortable: bool=False, context_menu_in_body: bool=False, row_background: bool=False, borders_innerH: bool=False, borders_outerH: bool=False, borders_innerV: bool=False, borders_outerV: bool=False, no_host_extendX: bool=False, no_host_extendY: bool=False, no_keep_columns_visible: bool=False, precise_widths: bool=False, no_clip: bool=False, pad_outerX: bool=False, no_pad_outerX: bool=False, no_pad_innerX: bool=False, scrollX: bool=False, scrollY: bool=False, no_saved_settings: bool=False, **kwargs) -> Union[int, str]:
    if False:
        print('Hello World!')
    "\t Adds a table.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\theader_row (bool, optional): show headers at the top of the columns\n\t\tclipper (bool, optional): Use clipper (rows must be same height).\n\t\tinner_width (int, optional): \n\t\tpolicy (int, optional): \n\t\tfreeze_rows (int, optional): \n\t\tfreeze_columns (int, optional): \n\t\tsort_multi (bool, optional): Hold shift when clicking headers to sort on multiple column.\n\t\tsort_tristate (bool, optional): Allow no sorting, disable default sorting.\n\t\tresizable (bool, optional): Enable resizing columns\n\t\treorderable (bool, optional): Enable reordering columns in header row (need calling TableSetupColumn() + TableHeadersRow() to display headers)\n\t\thideable (bool, optional): Enable hiding/disabling columns in context menu.\n\t\tsortable (bool, optional): Enable sorting. Call TableGetSortSpecs() to obtain sort specs. Also see ImGuiTableFlags_SortMulti and ImGuiTableFlags_SortTristate.\n\t\tcontext_menu_in_body (bool, optional): Right-click on columns body/contents will display table context menu. By default it is available in TableHeadersRow().\n\t\trow_background (bool, optional): Set each RowBg color with ImGuiCol_TableRowBg or ImGuiCol_TableRowBgAlt (equivalent of calling TableSetBgColor with ImGuiTableBgFlags_RowBg0 on each row manually)\n\t\tborders_innerH (bool, optional): Draw horizontal borders between rows.\n\t\tborders_outerH (bool, optional): Draw horizontal borders at the top and bottom.\n\t\tborders_innerV (bool, optional): Draw vertical borders between columns.\n\t\tborders_outerV (bool, optional): Draw vertical borders on the left and right sides.\n\t\tno_host_extendX (bool, optional): Make outer width auto-fit to columns, overriding outer_size.x value. Only available when ScrollX/ScrollY are disabled and Stretch columns are not used.\n\t\tno_host_extendY (bool, optional): Make outer height stop exactly at outer_size.y (prevent auto-extending table past the limit). Only available when ScrollX/ScrollY are disabled. Data below the limit will be clipped and not visible.\n\t\tno_keep_columns_visible (bool, optional): Disable keeping column always minimally visible when ScrollX is off and table gets too small. Not recommended if columns are resizable.\n\t\tprecise_widths (bool, optional): Disable distributing remainder width to stretched columns (width allocation on a 100-wide table with 3 columns: Without this flag: 33,33,34. With this flag: 33,33,33). With larger number of columns, resizing will appear to be less smooth.\n\t\tno_clip (bool, optional): Disable clipping rectangle for every individual columns.\n\t\tpad_outerX (bool, optional): Default if BordersOuterV is on. Enable outer-most padding. Generally desirable if you have headers.\n\t\tno_pad_outerX (bool, optional): Default if BordersOuterV is off. Disable outer-most padding.\n\t\tno_pad_innerX (bool, optional): Disable inner padding between columns (double inner padding if BordersOuterV is on, single inner padding if BordersOuterV is off).\n\t\tscrollX (bool, optional): Enable horizontal scrolling. Require 'outer_size' parameter of BeginTable() to specify the container size. Changes default sizing policy. Because this create a child window, ScrollY is currently generally recommended when using ScrollX.\n\t\tscrollY (bool, optional): Enable vertical scrolling.\n\t\tno_saved_settings (bool, optional): Never load/save settings in .ini file.\n\t\tid (Union[int, str], optional): (deprecated) \n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        if 'id' in kwargs.keys():
            warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
            tag = kwargs['id']
        widget = internal_dpg.add_table(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, width=width, height=height, indent=indent, parent=parent, before=before, source=source, callback=callback, show=show, pos=pos, filter_key=filter_key, delay_search=delay_search, header_row=header_row, clipper=clipper, inner_width=inner_width, policy=policy, freeze_rows=freeze_rows, freeze_columns=freeze_columns, sort_multi=sort_multi, sort_tristate=sort_tristate, resizable=resizable, reorderable=reorderable, hideable=hideable, sortable=sortable, context_menu_in_body=context_menu_in_body, row_background=row_background, borders_innerH=borders_innerH, borders_outerH=borders_outerH, borders_innerV=borders_innerV, borders_outerV=borders_outerV, no_host_extendX=no_host_extendX, no_host_extendY=no_host_extendY, no_keep_columns_visible=no_keep_columns_visible, precise_widths=precise_widths, no_clip=no_clip, pad_outerX=pad_outerX, no_pad_outerX=no_pad_outerX, no_pad_innerX=no_pad_innerX, scrollX=scrollX, scrollY=scrollY, no_saved_settings=no_saved_settings, **kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def table_cell(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, height: int=0, parent: Union[int, str]=0, before: Union[int, str]=0, show: bool=True, filter_key: str='', **kwargs) -> Union[int, str]:
    if False:
        return 10
    "\t Adds a table.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\theight (int, optional): Height of the item.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tid (Union[int, str], optional): (deprecated) \n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        if 'id' in kwargs.keys():
            warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
            tag = kwargs['id']
        widget = internal_dpg.add_table_cell(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, height=height, parent=parent, before=before, show=show, filter_key=filter_key, **kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def table_row(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, height: int=0, parent: Union[int, str]=0, before: Union[int, str]=0, show: bool=True, filter_key: str='', **kwargs) -> Union[int, str]:
    if False:
        for i in range(10):
            print('nop')
    "\t Adds a table row.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\theight (int, optional): Height of the item.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tid (Union[int, str], optional): (deprecated) \n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        if 'id' in kwargs.keys():
            warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
            tag = kwargs['id']
        widget = internal_dpg.add_table_row(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, height=height, parent=parent, before=before, show=show, filter_key=filter_key, **kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def template_registry(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, **kwargs) -> Union[int, str]:
    if False:
        print('Hello World!')
    "\t Adds a template registry.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tid (Union[int, str], optional): (deprecated) \n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        if 'id' in kwargs.keys():
            warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
            tag = kwargs['id']
        widget = internal_dpg.add_template_registry(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, **kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def texture_registry(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, show: bool=False, **kwargs) -> Union[int, str]:
    if False:
        print('Hello World!')
    "\t Adds a dynamic texture.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated) \n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        if 'id' in kwargs.keys():
            warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
            tag = kwargs['id']
        widget = internal_dpg.add_texture_registry(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, show=show, **kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def theme(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, **kwargs) -> Union[int, str]:
    if False:
        print('Hello World!')
    "\t Adds a theme.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tid (Union[int, str], optional): (deprecated) \n\t\tdefault_theme (bool, optional): (deprecated) \n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        if 'id' in kwargs.keys():
            warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
            tag = kwargs['id']
        if 'default_theme' in kwargs.keys():
            warnings.warn('default_theme keyword removed', DeprecationWarning, 2)
            kwargs.pop('default_theme', None)
        widget = internal_dpg.add_theme(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, **kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def theme_component(item_type: int=0, *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, before: Union[int, str]=0, enabled_state: bool=True, **kwargs) -> Union[int, str]:
    if False:
        print('Hello World!')
    "\t Adds a theme component.\n\n\tArgs:\n\t\titem_type (int, optional): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tenabled_state (bool, optional): \n\t\tid (Union[int, str], optional): (deprecated) \n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        if 'id' in kwargs.keys():
            warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
            tag = kwargs['id']
        widget = internal_dpg.add_theme_component(item_type, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, before=before, enabled_state=enabled_state, **kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def tooltip(parent: Union[int, str], *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, show: bool=True, delay: float=0.0, hide_on_activity: bool=False, **kwargs) -> Union[int, str]:
    if False:
        while True:
            i = 10
    "\t Adds a tooltip window.\n\n\tArgs:\n\t\tparent (Union[int, str]): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tdelay (float, optional): Activation delay: time, in seconds, during which the mouse should stay still in order to display the tooltip.  May be zero for instant activation.\n\t\thide_on_activity (bool, optional): Hide the tooltip if the user has moved the mouse.  If False, the tooltip will follow mouse pointer.\n\t\tid (Union[int, str], optional): (deprecated) \n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        if 'id' in kwargs.keys():
            warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
            tag = kwargs['id']
        widget = internal_dpg.add_tooltip(parent, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, show=show, delay=delay, hide_on_activity=hide_on_activity, **kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def tree_node(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', drag_callback: Callable=None, drop_callback: Callable=None, show: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], filter_key: str='', delay_search: bool=False, tracked: bool=False, track_offset: float=0.5, default_open: bool=False, open_on_double_click: bool=False, open_on_arrow: bool=False, leaf: bool=False, bullet: bool=False, selectable: bool=False, **kwargs) -> Union[int, str]:
    if False:
        while True:
            i = 10
    "\t Adds a tree node to add items to.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_open (bool, optional): Sets the tree node open by default.\n\t\topen_on_double_click (bool, optional): Need double-click to open node.\n\t\topen_on_arrow (bool, optional): Only open when clicking on the arrow part.\n\t\tleaf (bool, optional): No collapsing, no arrow (use as a convenience for leaf nodes).\n\t\tbullet (bool, optional): Display a bullet instead of arrow.\n\t\tselectable (bool, optional): Makes the tree selectable.\n\t\tid (Union[int, str], optional): (deprecated) \n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        if 'id' in kwargs.keys():
            warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
            tag = kwargs['id']
        widget = internal_dpg.add_tree_node(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, indent=indent, parent=parent, before=before, payload_type=payload_type, drag_callback=drag_callback, drop_callback=drop_callback, show=show, pos=pos, filter_key=filter_key, delay_search=delay_search, tracked=tracked, track_offset=track_offset, default_open=default_open, open_on_double_click=open_on_double_click, open_on_arrow=open_on_arrow, leaf=leaf, bullet=bullet, selectable=selectable, **kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def value_registry(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, **kwargs) -> Union[int, str]:
    if False:
        while True:
            i = 10
    "\t Adds a value registry.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tid (Union[int, str], optional): (deprecated) \n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        if 'id' in kwargs.keys():
            warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
            tag = kwargs['id']
        widget = internal_dpg.add_value_registry(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, **kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def viewport_drawlist(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, show: bool=True, filter_key: str='', delay_search: bool=False, front: bool=True, **kwargs) -> Union[int, str]:
    if False:
        for i in range(10):
            print('nop')
    "\t A container that is used to present draw items or layers directly to the viewport. By default this will draw to the back of the viewport. Layers and draw items should be added to this widget as children.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\tfront (bool, optional): Draws to the front of the view port instead of the back.\n\t\tid (Union[int, str], optional): (deprecated) \n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        if 'id' in kwargs.keys():
            warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
            tag = kwargs['id']
        widget = internal_dpg.add_viewport_drawlist(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, show=show, filter_key=filter_key, delay_search=delay_search, front=front, **kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def viewport_menu_bar(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, indent: int=-1, parent: Union[int, str]=0, show: bool=True, delay_search: bool=False, **kwargs) -> Union[int, str]:
    if False:
        return 10
    "\t Adds a menubar to the viewport.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\tid (Union[int, str], optional): (deprecated) \n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        if 'id' in kwargs.keys():
            warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
            tag = kwargs['id']
        widget = internal_dpg.add_viewport_menu_bar(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, indent=indent, parent=parent, show=show, delay_search=delay_search, **kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def window(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, width: int=0, height: int=0, indent: int=-1, show: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], delay_search: bool=False, min_size: Union[List[int], Tuple[int, ...]]=[100, 100], max_size: Union[List[int], Tuple[int, ...]]=[30000, 30000], menubar: bool=False, collapsed: bool=False, autosize: bool=False, no_resize: bool=False, no_title_bar: bool=False, no_move: bool=False, no_scrollbar: bool=False, no_collapse: bool=False, horizontal_scrollbar: bool=False, no_focus_on_appearing: bool=False, no_bring_to_front_on_focus: bool=False, no_close: bool=False, no_background: bool=False, modal: bool=False, popup: bool=False, no_saved_settings: bool=False, no_open_over_existing_popup: bool=True, no_scroll_with_mouse: bool=False, on_close: Callable=None, **kwargs) -> Union[int, str]:
    if False:
        for i in range(10):
            print('nop')
    "\t Creates a new window for following items to be added to.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\tmin_size (Union[List[int], Tuple[int, ...]], optional): Minimum window size.\n\t\tmax_size (Union[List[int], Tuple[int, ...]], optional): Maximum window size.\n\t\tmenubar (bool, optional): Shows or hides the menubar.\n\t\tcollapsed (bool, optional): Collapse the window.\n\t\tautosize (bool, optional): Autosized the window to fit it's items.\n\t\tno_resize (bool, optional): Allows for the window size to be changed or fixed.\n\t\tno_title_bar (bool, optional): Title name for the title bar of the window.\n\t\tno_move (bool, optional): Allows for the window's position to be changed or fixed.\n\t\tno_scrollbar (bool, optional):  Disable scrollbars. (window can still scroll with mouse or programmatically)\n\t\tno_collapse (bool, optional): Disable user collapsing window by double-clicking on it.\n\t\thorizontal_scrollbar (bool, optional): Allow horizontal scrollbar to appear. (off by default)\n\t\tno_focus_on_appearing (bool, optional): Disable taking focus when transitioning from hidden to visible state.\n\t\tno_bring_to_front_on_focus (bool, optional): Disable bringing window to front when taking focus. (e.g. clicking on it or programmatically giving it focus)\n\t\tno_close (bool, optional): Disable user closing the window by removing the close button.\n\t\tno_background (bool, optional): Sets Background and border alpha to transparent.\n\t\tmodal (bool, optional): Fills area behind window according to the theme and disables user ability to interact with anything except the window.\n\t\tpopup (bool, optional): Fills area behind window according to the theme, removes title bar, collapse and close. Window can be closed by selecting area in the background behind the window.\n\t\tno_saved_settings (bool, optional): Never load/save settings in .ini file.\n\t\tno_open_over_existing_popup (bool, optional): Don't open if there's already a popup\n\t\tno_scroll_with_mouse (bool, optional): Disable user vertically scrolling with mouse wheel.\n\t\ton_close (Callable, optional): Callback ran when window is closed.\n\t\tid (Union[int, str], optional): (deprecated) \n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        if 'id' in kwargs.keys():
            warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
            tag = kwargs['id']
        widget = internal_dpg.add_window(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, width=width, height=height, indent=indent, show=show, pos=pos, delay_search=delay_search, min_size=min_size, max_size=max_size, menubar=menubar, collapsed=collapsed, autosize=autosize, no_resize=no_resize, no_title_bar=no_title_bar, no_move=no_move, no_scrollbar=no_scrollbar, no_collapse=no_collapse, horizontal_scrollbar=horizontal_scrollbar, no_focus_on_appearing=no_focus_on_appearing, no_bring_to_front_on_focus=no_bring_to_front_on_focus, no_close=no_close, no_background=no_background, modal=modal, popup=popup, no_saved_settings=no_saved_settings, no_open_over_existing_popup=no_open_over_existing_popup, no_scroll_with_mouse=no_scroll_with_mouse, on_close=on_close, **kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

def add_2d_histogram_series(x: Union[List[float], Tuple[float, ...]], y: Union[List[float], Tuple[float, ...]], *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, before: Union[int, str]=0, source: Union[int, str]=0, show: bool=True, xbins: int=-1, ybins: int=-1, xmin_range: float=0.0, xmax_range: float=1.0, ymin_range: float=0.0, ymax_range: float=1.0, density: bool=False, outliers: bool=True, **kwargs) -> Union[int, str]:
    if False:
        print('Hello World!')
    "\t Adds a 2d histogram series.\n\n\tArgs:\n\t\tx (Any): \n\t\ty (Any): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\txbins (int, optional): \n\t\tybins (int, optional): \n\t\txmin_range (float, optional): \n\t\txmax_range (float, optional): \n\t\tymin_range (float, optional): \n\t\tymax_range (float, optional): \n\t\tdensity (bool, optional): \n\t\toutliers (bool, optional): \n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_2d_histogram_series(x, y, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, before=before, source=source, show=show, xbins=xbins, ybins=ybins, xmin_range=xmin_range, xmax_range=xmax_range, ymin_range=ymin_range, ymax_range=ymax_range, density=density, outliers=outliers, **kwargs)

def add_3d_slider(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, width: int=0, height: int=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, source: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', callback: Callable=None, drag_callback: Callable=None, drop_callback: Callable=None, show: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], filter_key: str='', tracked: bool=False, track_offset: float=0.5, default_value: Union[List[float], Tuple[float, ...]]=(0.0, 0.0, 0.0, 0.0), max_x: float=100.0, max_y: float=100.0, max_z: float=100.0, min_x: float=0.0, min_y: float=0.0, min_z: float=0.0, scale: float=1.0, **kwargs) -> Union[int, str]:
    if False:
        return 10
    "\t Adds a 3D box slider.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (Union[List[float], Tuple[float, ...]], optional): \n\t\tmax_x (float, optional): Applies upper limit to slider.\n\t\tmax_y (float, optional): Applies upper limit to slider.\n\t\tmax_z (float, optional): Applies upper limit to slider.\n\t\tmin_x (float, optional): Applies lower limit to slider.\n\t\tmin_y (float, optional): Applies lower limit to slider.\n\t\tmin_z (float, optional): Applies lower limit to slider.\n\t\tscale (float, optional): Size of the widget.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_3d_slider(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, width=width, height=height, indent=indent, parent=parent, before=before, source=source, payload_type=payload_type, callback=callback, drag_callback=drag_callback, drop_callback=drop_callback, show=show, pos=pos, filter_key=filter_key, tracked=tracked, track_offset=track_offset, default_value=default_value, max_x=max_x, max_y=max_y, max_z=max_z, min_x=min_x, min_y=min_y, min_z=min_z, scale=scale, **kwargs)

def add_alias(alias: str, item: Union[int, str], **kwargs) -> None:
    if False:
        print('Hello World!')
    '\t Adds an alias.\n\n\tArgs:\n\t\talias (str): \n\t\titem (Union[int, str]): \n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.add_alias(alias, item, **kwargs)

def add_area_series(x: Union[List[float], Tuple[float, ...]], y: Union[List[float], Tuple[float, ...]], *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, before: Union[int, str]=0, source: Union[int, str]=0, show: bool=True, fill: Union[List[int], Tuple[int, ...]]=(0, 0, 0, -255), contribute_to_bounds: bool=True, **kwargs) -> Union[int, str]:
    if False:
        return 10
    "\t Adds an area series to a plot.\n\n\tArgs:\n\t\tx (Any): \n\t\ty (Any): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tfill (Union[List[int], Tuple[int, ...]], optional): \n\t\tcontribute_to_bounds (bool, optional): \n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_area_series(x, y, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, before=before, source=source, show=show, fill=fill, contribute_to_bounds=contribute_to_bounds, **kwargs)

def add_bar_series(x: Union[List[float], Tuple[float, ...]], y: Union[List[float], Tuple[float, ...]], *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, before: Union[int, str]=0, source: Union[int, str]=0, show: bool=True, weight: float=1.0, horizontal: bool=False, **kwargs) -> Union[int, str]:
    if False:
        i = 10
        return i + 15
    "\t Adds a bar series to a plot.\n\n\tArgs:\n\t\tx (Any): \n\t\ty (Any): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tweight (float, optional): \n\t\thorizontal (bool, optional): \n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_bar_series(x, y, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, before=before, source=source, show=show, weight=weight, horizontal=horizontal, **kwargs)

def add_bool_value(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, source: Union[int, str]=0, default_value: bool=False, parent: Union[int, str]=internal_dpg.mvReservedUUID_3, **kwargs) -> Union[int, str]:
    if False:
        return 10
    "\t Adds a bool value.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tdefault_value (bool, optional): \n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_bool_value(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, source=source, default_value=default_value, parent=parent, **kwargs)

def add_button(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, width: int=0, height: int=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', callback: Callable=None, drag_callback: Callable=None, drop_callback: Callable=None, show: bool=True, enabled: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], filter_key: str='', tracked: bool=False, track_offset: float=0.5, small: bool=False, arrow: bool=False, direction: int=0, **kwargs) -> Union[int, str]:
    if False:
        i = 10
        return i + 15
    "\t Adds a button.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tsmall (bool, optional): Shrinks the size of the button to the text of the label it contains. Useful for embedding in text.\n\t\tarrow (bool, optional): Displays an arrow in place of the text string. This requires the direction keyword.\n\t\tdirection (int, optional): Sets the cardinal direction for the arrow by using constants mvDir_Left, mvDir_Up, mvDir_Down, mvDir_Right, mvDir_None. Arrow keyword must be set to True.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_button(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, width=width, height=height, indent=indent, parent=parent, before=before, payload_type=payload_type, callback=callback, drag_callback=drag_callback, drop_callback=drop_callback, show=show, enabled=enabled, pos=pos, filter_key=filter_key, tracked=tracked, track_offset=track_offset, small=small, arrow=arrow, direction=direction, **kwargs)

def add_candle_series(dates: Union[List[float], Tuple[float, ...]], opens: Union[List[float], Tuple[float, ...]], closes: Union[List[float], Tuple[float, ...]], lows: Union[List[float], Tuple[float, ...]], highs: Union[List[float], Tuple[float, ...]], *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, before: Union[int, str]=0, source: Union[int, str]=0, show: bool=True, bull_color: Union[List[int], Tuple[int, ...]]=(0, 255, 113, 255), bear_color: Union[List[int], Tuple[int, ...]]=(218, 13, 79, 255), weight: float=0.25, tooltip: bool=True, time_unit: int=5, **kwargs) -> Union[int, str]:
    if False:
        while True:
            i = 10
    "\t Adds a candle series to a plot.\n\n\tArgs:\n\t\tdates (Any): \n\t\topens (Any): \n\t\tcloses (Any): \n\t\tlows (Any): \n\t\thighs (Any): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tbull_color (Union[List[int], Tuple[int, ...]], optional): \n\t\tbear_color (Union[List[int], Tuple[int, ...]], optional): \n\t\tweight (float, optional): \n\t\ttooltip (bool, optional): \n\t\ttime_unit (int, optional): mvTimeUnit_* constants. Default mvTimeUnit_Day.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_candle_series(dates, opens, closes, lows, highs, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, before=before, source=source, show=show, bull_color=bull_color, bear_color=bear_color, weight=weight, tooltip=tooltip, time_unit=time_unit, **kwargs)

def add_char_remap(source: int, target: int, *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, **kwargs) -> Union[int, str]:
    if False:
        while True:
            i = 10
    "\t Remaps a character.\n\n\tArgs:\n\t\tsource (int): \n\t\ttarget (int): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_char_remap(source, target, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, **kwargs)

def add_checkbox(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, source: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', callback: Callable=None, drag_callback: Callable=None, drop_callback: Callable=None, show: bool=True, enabled: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], filter_key: str='', tracked: bool=False, track_offset: float=0.5, default_value: bool=False, **kwargs) -> Union[int, str]:
    if False:
        while True:
            i = 10
    "\t Adds a checkbox.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (bool, optional): Sets the default value of the checkmark\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_checkbox(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, indent=indent, parent=parent, before=before, source=source, payload_type=payload_type, callback=callback, drag_callback=drag_callback, drop_callback=drop_callback, show=show, enabled=enabled, pos=pos, filter_key=filter_key, tracked=tracked, track_offset=track_offset, default_value=default_value, **kwargs)

def add_child_window(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, width: int=0, height: int=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', drop_callback: Callable=None, show: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], filter_key: str='', delay_search: bool=False, tracked: bool=False, track_offset: float=0.5, border: bool=True, autosize_x: bool=False, autosize_y: bool=False, no_scrollbar: bool=False, horizontal_scrollbar: bool=False, menubar: bool=False, no_scroll_with_mouse: bool=False, **kwargs) -> Union[int, str]:
    if False:
        print('Hello World!')
    "\t Adds an embedded child window. Will show scrollbars when items do not fit.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tborder (bool, optional): Shows/Hides the border around the sides.\n\t\tautosize_x (bool, optional): Autosize the window to its parents size in x.\n\t\tautosize_y (bool, optional): Autosize the window to its parents size in y.\n\t\tno_scrollbar (bool, optional):  Disable scrollbars (window can still scroll with mouse or programmatically).\n\t\thorizontal_scrollbar (bool, optional): Allow horizontal scrollbar to appear (off by default).\n\t\tmenubar (bool, optional): Shows/Hides the menubar at the top.\n\t\tno_scroll_with_mouse (bool, optional): Disable user vertically scrolling with mouse wheel.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_child_window(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, width=width, height=height, indent=indent, parent=parent, before=before, payload_type=payload_type, drop_callback=drop_callback, show=show, pos=pos, filter_key=filter_key, delay_search=delay_search, tracked=tracked, track_offset=track_offset, border=border, autosize_x=autosize_x, autosize_y=autosize_y, no_scrollbar=no_scrollbar, horizontal_scrollbar=horizontal_scrollbar, menubar=menubar, no_scroll_with_mouse=no_scroll_with_mouse, **kwargs)

def add_clipper(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, width: int=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, show: bool=True, delay_search: bool=False, **kwargs) -> Union[int, str]:
    if False:
        for i in range(10):
            print('nop')
    "\t Helper to manually clip large list of items. Increases performance by not searching or drawing widgets outside of the clipped region.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_clipper(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, width=width, indent=indent, parent=parent, before=before, show=show, delay_search=delay_search, **kwargs)

def add_collapsing_header(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', drag_callback: Callable=None, drop_callback: Callable=None, show: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], filter_key: str='', delay_search: bool=False, tracked: bool=False, track_offset: float=0.5, closable: bool=False, default_open: bool=False, open_on_double_click: bool=False, open_on_arrow: bool=False, leaf: bool=False, bullet: bool=False, **kwargs) -> Union[int, str]:
    if False:
        print('Hello World!')
    "\t Adds a collapsing header to add items to. Must be closed with the end command.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tclosable (bool, optional): Adds the ability to hide this widget by pressing the (x) in the top right of widget.\n\t\tdefault_open (bool, optional): Sets the collapseable header open by default.\n\t\topen_on_double_click (bool, optional): Need double-click to open node.\n\t\topen_on_arrow (bool, optional): Only open when clicking on the arrow part.\n\t\tleaf (bool, optional): No collapsing, no arrow (use as a convenience for leaf nodes).\n\t\tbullet (bool, optional): Display a bullet instead of arrow.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_collapsing_header(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, indent=indent, parent=parent, before=before, payload_type=payload_type, drag_callback=drag_callback, drop_callback=drop_callback, show=show, pos=pos, filter_key=filter_key, delay_search=delay_search, tracked=tracked, track_offset=track_offset, closable=closable, default_open=default_open, open_on_double_click=open_on_double_click, open_on_arrow=open_on_arrow, leaf=leaf, bullet=bullet, **kwargs)

def add_color_button(default_value: Union[List[int], Tuple[int, ...]]=(0, 0, 0, 255), *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, width: int=0, height: int=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', callback: Callable=None, drag_callback: Callable=None, drop_callback: Callable=None, show: bool=True, enabled: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], filter_key: str='', tracked: bool=False, track_offset: float=0.5, no_alpha: bool=False, no_border: bool=False, no_drag_drop: bool=False, **kwargs) -> Union[int, str]:
    if False:
        i = 10
        return i + 15
    "\t Adds a color button.\n\n\tArgs:\n\t\tdefault_value (Union[List[int], Tuple[int, ...]], optional): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tno_alpha (bool, optional): Removes the displayed slider that can change alpha channel.\n\t\tno_border (bool, optional): Disable border around the image.\n\t\tno_drag_drop (bool, optional): Disable ability to drag and drop small preview (color square) to apply colors to other items.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_color_button(default_value, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, width=width, height=height, indent=indent, parent=parent, before=before, payload_type=payload_type, callback=callback, drag_callback=drag_callback, drop_callback=drop_callback, show=show, enabled=enabled, pos=pos, filter_key=filter_key, tracked=tracked, track_offset=track_offset, no_alpha=no_alpha, no_border=no_border, no_drag_drop=no_drag_drop, **kwargs)

def add_color_edit(default_value: Union[List[int], Tuple[int, ...]]=(0, 0, 0, 255), *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, width: int=0, height: int=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, source: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', callback: Callable=None, drag_callback: Callable=None, drop_callback: Callable=None, show: bool=True, enabled: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], filter_key: str='', tracked: bool=False, track_offset: float=0.5, no_alpha: bool=False, no_picker: bool=False, no_options: bool=False, no_small_preview: bool=False, no_inputs: bool=False, no_tooltip: bool=False, no_label: bool=False, no_drag_drop: bool=False, alpha_bar: bool=False, alpha_preview: int=0, display_mode: int=1048576, display_type: int=8388608, input_mode: int=134217728, **kwargs) -> Union[int, str]:
    if False:
        i = 10
        return i + 15
    "\t Adds an RGBA color editor. Left clicking the small color preview will provide a color picker. Click and draging the small color preview will copy the color to be applied on any other color widget.\n\n\tArgs:\n\t\tdefault_value (Union[List[int], Tuple[int, ...]], optional): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tno_alpha (bool, optional): Removes the displayed slider that can change alpha channel.\n\t\tno_picker (bool, optional): Disable picker popup when color square is clicked.\n\t\tno_options (bool, optional): Disable toggling options menu when right-clicking on inputs/small preview.\n\t\tno_small_preview (bool, optional): Disable colored square preview next to the inputs. (e.g. to show only the inputs). This only displays if the side preview is not shown.\n\t\tno_inputs (bool, optional): Disable inputs sliders/text widgets. (e.g. to show only the small preview colored square)\n\t\tno_tooltip (bool, optional): Disable tooltip when hovering the preview.\n\t\tno_label (bool, optional): Disable display of inline text label.\n\t\tno_drag_drop (bool, optional): Disable ability to drag and drop small preview (color square) to apply colors to other items.\n\t\talpha_bar (bool, optional): Show vertical alpha bar/gradient in picker.\n\t\talpha_preview (int, optional): mvColorEdit_AlphaPreviewNone, mvColorEdit_AlphaPreview, or mvColorEdit_AlphaPreviewHalf\n\t\tdisplay_mode (int, optional): mvColorEdit_rgb, mvColorEdit_hsv, or mvColorEdit_hex\n\t\tdisplay_type (int, optional): mvColorEdit_uint8 or mvColorEdit_float\n\t\tinput_mode (int, optional): mvColorEdit_input_rgb or mvColorEdit_input_hsv\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_color_edit(default_value, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, width=width, height=height, indent=indent, parent=parent, before=before, source=source, payload_type=payload_type, callback=callback, drag_callback=drag_callback, drop_callback=drop_callback, show=show, enabled=enabled, pos=pos, filter_key=filter_key, tracked=tracked, track_offset=track_offset, no_alpha=no_alpha, no_picker=no_picker, no_options=no_options, no_small_preview=no_small_preview, no_inputs=no_inputs, no_tooltip=no_tooltip, no_label=no_label, no_drag_drop=no_drag_drop, alpha_bar=alpha_bar, alpha_preview=alpha_preview, display_mode=display_mode, display_type=display_type, input_mode=input_mode, **kwargs)

def add_color_picker(default_value: Union[List[int], Tuple[int, ...]]=(0, 0, 0, 255), *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, width: int=0, height: int=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, source: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', callback: Callable=None, drag_callback: Callable=None, drop_callback: Callable=None, show: bool=True, enabled: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], filter_key: str='', tracked: bool=False, track_offset: float=0.5, no_alpha: bool=False, no_side_preview: bool=False, no_small_preview: bool=False, no_inputs: bool=False, no_tooltip: bool=False, no_label: bool=False, alpha_bar: bool=False, display_rgb: bool=False, display_hsv: bool=False, display_hex: bool=False, picker_mode: int=33554432, alpha_preview: int=0, display_type: int=8388608, input_mode: int=134217728, **kwargs) -> Union[int, str]:
    if False:
        print('Hello World!')
    "\t Adds an RGB color picker. Right click the color picker for options. Click and drag the color preview to copy the color and drop on any other color widget to apply. Right Click allows the style of the color picker to be changed.\n\n\tArgs:\n\t\tdefault_value (Union[List[int], Tuple[int, ...]], optional): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tno_alpha (bool, optional): Removes the displayed slider that can change alpha channel.\n\t\tno_side_preview (bool, optional): Disable bigger color preview on right side of the picker, use small colored square preview instead , unless small preview is also hidden.\n\t\tno_small_preview (bool, optional): Disable colored square preview next to the inputs. (e.g. to show only the inputs). This only displays if the side preview is not shown.\n\t\tno_inputs (bool, optional): Disable inputs sliders/text widgets. (e.g. to show only the small preview colored square)\n\t\tno_tooltip (bool, optional): Disable tooltip when hovering the preview.\n\t\tno_label (bool, optional): Disable display of inline text label.\n\t\talpha_bar (bool, optional): Show vertical alpha bar/gradient in picker.\n\t\tdisplay_rgb (bool, optional): Override _display_ type among RGB/HSV/Hex.\n\t\tdisplay_hsv (bool, optional): Override _display_ type among RGB/HSV/Hex.\n\t\tdisplay_hex (bool, optional): Override _display_ type among RGB/HSV/Hex.\n\t\tpicker_mode (int, optional): mvColorPicker_bar or mvColorPicker_wheel\n\t\talpha_preview (int, optional): mvColorEdit_AlphaPreviewNone, mvColorEdit_AlphaPreview, or mvColorEdit_AlphaPreviewHalf\n\t\tdisplay_type (int, optional): mvColorEdit_uint8 or mvColorEdit_float\n\t\tinput_mode (int, optional): mvColorEdit_input_rgb or mvColorEdit_input_hsv\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_color_picker(default_value, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, width=width, height=height, indent=indent, parent=parent, before=before, source=source, payload_type=payload_type, callback=callback, drag_callback=drag_callback, drop_callback=drop_callback, show=show, enabled=enabled, pos=pos, filter_key=filter_key, tracked=tracked, track_offset=track_offset, no_alpha=no_alpha, no_side_preview=no_side_preview, no_small_preview=no_small_preview, no_inputs=no_inputs, no_tooltip=no_tooltip, no_label=no_label, alpha_bar=alpha_bar, display_rgb=display_rgb, display_hsv=display_hsv, display_hex=display_hex, picker_mode=picker_mode, alpha_preview=alpha_preview, display_type=display_type, input_mode=input_mode, **kwargs)

def add_color_value(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, source: Union[int, str]=0, default_value: Union[List[float], Tuple[float, ...]]=(0.0, 0.0, 0.0, 0.0), parent: Union[int, str]=internal_dpg.mvReservedUUID_3, **kwargs) -> Union[int, str]:
    if False:
        print('Hello World!')
    "\t Adds a color value.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tdefault_value (Union[List[float], Tuple[float, ...]], optional): \n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_color_value(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, source=source, default_value=default_value, parent=parent, **kwargs)

def add_colormap(colors: List[Union[List[int], Tuple[int, ...]]], qualitative: bool, *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, show: bool=True, parent: Union[int, str]=internal_dpg.mvReservedUUID_4, **kwargs) -> Union[int, str]:
    if False:
        i = 10
        return i + 15
    "\t Adds a legend that pairs colors with normalized value 0.0->1.0. Each color will be  This is typically used with a heat series. (ex. [[0, 0, 0, 255], [255, 255, 255, 255]] will be mapped to a soft transition from 0.0-1.0)\n\n\tArgs:\n\t\tcolors (Any): colors that will be mapped to the normalized value 0.0->1.0\n\t\tqualitative (bool): Qualitative will create hard transitions for color boundries across the value range when enabled.\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_colormap(colors, qualitative, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, show=show, parent=parent, **kwargs)

def add_colormap_button(default_value: Union[List[int], Tuple[int, ...]]=(0, 0, 0, 255), *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, width: int=0, height: int=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', callback: Callable=None, drag_callback: Callable=None, drop_callback: Callable=None, show: bool=True, enabled: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], filter_key: str='', tracked: bool=False, track_offset: float=0.5, **kwargs) -> Union[int, str]:
    if False:
        while True:
            i = 10
    "\t Adds a button that a color map can be bound to.\n\n\tArgs:\n\t\tdefault_value (Union[List[int], Tuple[int, ...]], optional): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_colormap_button(default_value, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, width=width, height=height, indent=indent, parent=parent, before=before, payload_type=payload_type, callback=callback, drag_callback=drag_callback, drop_callback=drop_callback, show=show, enabled=enabled, pos=pos, filter_key=filter_key, tracked=tracked, track_offset=track_offset, **kwargs)

def add_colormap_registry(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, show: bool=False, **kwargs) -> Union[int, str]:
    if False:
        i = 10
        return i + 15
    "\t Adds a colormap registry.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_colormap_registry(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, show=show, **kwargs)

def add_colormap_scale(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, width: int=0, height: int=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, source: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', drop_callback: Callable=None, show: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], colormap: Union[int, str]=0, min_scale: float=0.0, max_scale: float=1.0, **kwargs) -> Union[int, str]:
    if False:
        return 10
    "\t Adds a legend that pairs values with colors. This is typically used with a heat series. \n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tcolormap (Union[int, str], optional): mvPlotColormap_* constants or mvColorMap uuid from a color map registry\n\t\tmin_scale (float, optional): Sets the min number of the color scale. Typically is the same as the min scale from the heat series.\n\t\tmax_scale (float, optional): Sets the max number of the color scale. Typically is the same as the max scale from the heat series.\n\t\tid (Union[int, str], optional): (deprecated) \n\t\tdrag_callback (Callable, optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    if 'drag_callback' in kwargs.keys():
        warnings.warn('drag_callback keyword removed', DeprecationWarning, 2)
        kwargs.pop('drag_callback', None)
    return internal_dpg.add_colormap_scale(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, width=width, height=height, indent=indent, parent=parent, before=before, source=source, payload_type=payload_type, drop_callback=drop_callback, show=show, pos=pos, colormap=colormap, min_scale=min_scale, max_scale=max_scale, **kwargs)

def add_colormap_slider(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, width: int=0, height: int=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', callback: Callable=None, drop_callback: Callable=None, show: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], filter_key: str='', tracked: bool=False, track_offset: float=0.5, default_value: float=0.0, **kwargs) -> Union[int, str]:
    if False:
        i = 10
        return i + 15
    "\t Adds a color slider that a color map can be bound to.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (float, optional): \n\t\tid (Union[int, str], optional): (deprecated) \n\t\tdrag_callback (Callable, optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    if 'drag_callback' in kwargs.keys():
        warnings.warn('drag_callback keyword removed', DeprecationWarning, 2)
        kwargs.pop('drag_callback', None)
    return internal_dpg.add_colormap_slider(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, width=width, height=height, indent=indent, parent=parent, before=before, payload_type=payload_type, callback=callback, drop_callback=drop_callback, show=show, pos=pos, filter_key=filter_key, tracked=tracked, track_offset=track_offset, default_value=default_value, **kwargs)

def add_combo(items: Union[List[str], Tuple[str, ...]]=(), *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, width: int=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, source: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', callback: Callable=None, drag_callback: Callable=None, drop_callback: Callable=None, show: bool=True, enabled: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], filter_key: str='', tracked: bool=False, track_offset: float=0.5, default_value: str='', popup_align_left: bool=False, no_arrow_button: bool=False, no_preview: bool=False, height_mode: int=1, **kwargs) -> Union[int, str]:
    if False:
        return 10
    "\t Adds a combo dropdown that allows a user to select a single option from a drop down window. All items will be shown as selectables on the dropdown.\n\n\tArgs:\n\t\titems (Union[List[str], Tuple[str, ...]], optional): A tuple of items to be shown in the drop down window. Can consist of any combination of types but will convert all items to strings to be shown.\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (str, optional): Sets a selected item from the drop down by specifying the string value.\n\t\tpopup_align_left (bool, optional): Align the contents on the popup toward the left.\n\t\tno_arrow_button (bool, optional): Display the preview box without the square arrow button indicating dropdown activity.\n\t\tno_preview (bool, optional): Display only the square arrow button and not the selected value.\n\t\theight_mode (int, optional): Controlls the number of items shown in the dropdown by the constants mvComboHeight_Small, mvComboHeight_Regular, mvComboHeight_Large, mvComboHeight_Largest\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_combo(items, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, width=width, indent=indent, parent=parent, before=before, source=source, payload_type=payload_type, callback=callback, drag_callback=drag_callback, drop_callback=drop_callback, show=show, enabled=enabled, pos=pos, filter_key=filter_key, tracked=tracked, track_offset=track_offset, default_value=default_value, popup_align_left=popup_align_left, no_arrow_button=no_arrow_button, no_preview=no_preview, height_mode=height_mode, **kwargs)

def add_custom_series(x: Union[List[float], Tuple[float, ...]], y: Union[List[float], Tuple[float, ...]], channel_count: int, *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, before: Union[int, str]=0, source: Union[int, str]=0, callback: Callable=None, show: bool=True, y1: Any=[], y2: Any=[], y3: Any=[], tooltip: bool=True, **kwargs) -> Union[int, str]:
    if False:
        i = 10
        return i + 15
    "\t Adds a custom series to a plot. New in 1.6.\n\n\tArgs:\n\t\tx (Any): \n\t\ty (Any): \n\t\tchannel_count (int): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\ty1 (Any, optional): \n\t\ty2 (Any, optional): \n\t\ty3 (Any, optional): \n\t\ttooltip (bool, optional): Show tooltip when plot is hovered.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_custom_series(x, y, channel_count, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, before=before, source=source, callback=callback, show=show, y1=y1, y2=y2, y3=y3, tooltip=tooltip, **kwargs)

def add_date_picker(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', callback: Callable=None, drag_callback: Callable=None, drop_callback: Callable=None, show: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], filter_key: str='', tracked: bool=False, track_offset: float=0.5, default_value: dict={'month_day': 14, 'year': 20, 'month': 5}, level: int=0, **kwargs) -> Union[int, str]:
    if False:
        for i in range(10):
            print('nop')
    "\t Adds a data picker.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (dict, optional): \n\t\tlevel (int, optional): Use avaliable constants. mvDatePickerLevel_Day, mvDatePickerLevel_Month, mvDatePickerLevel_Year\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_date_picker(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, indent=indent, parent=parent, before=before, payload_type=payload_type, callback=callback, drag_callback=drag_callback, drop_callback=drop_callback, show=show, pos=pos, filter_key=filter_key, tracked=tracked, track_offset=track_offset, default_value=default_value, level=level, **kwargs)

def add_double4_value(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, source: Union[int, str]=0, default_value: Any=(0.0, 0.0, 0.0, 0.0), parent: Union[int, str]=internal_dpg.mvReservedUUID_3, **kwargs) -> Union[int, str]:
    if False:
        print('Hello World!')
    "\t Adds a double value.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tdefault_value (Any, optional): \n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_double4_value(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, source=source, default_value=default_value, parent=parent, **kwargs)

def add_double_value(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, source: Union[int, str]=0, default_value: float=0.0, parent: Union[int, str]=internal_dpg.mvReservedUUID_3, **kwargs) -> Union[int, str]:
    if False:
        while True:
            i = 10
    "\t Adds a double value.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tdefault_value (float, optional): \n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_double_value(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, source=source, default_value=default_value, parent=parent, **kwargs)

def add_drag_double(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, width: int=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, source: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', callback: Callable=None, drag_callback: Callable=None, drop_callback: Callable=None, show: bool=True, enabled: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], filter_key: str='', tracked: bool=False, track_offset: float=0.5, default_value: float=0.0, format: str='%0.3f', speed: float=1.0, min_value: float=0.0, max_value: float=100.0, no_input: bool=False, clamped: bool=False, **kwargs) -> Union[int, str]:
    if False:
        return 10
    "\t Adds drag for a single double value. Useful when drag float is not accurate enough. Directly entry can be done with double click or CTRL+Click. Min and Max alone are a soft limit for the drag. Use clamped keyword to also apply limits to the direct entry modes.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (float, optional): \n\t\tformat (str, optional): Determines the format the float will be displayed as use python string formatting.\n\t\tspeed (float, optional): Sets the sensitivity the float will be modified while dragging.\n\t\tmin_value (float, optional): Applies a limit only to draging entry only.\n\t\tmax_value (float, optional): Applies a limit only to draging entry only.\n\t\tno_input (bool, optional): Disable direct entry methods or Enter key allowing to input text directly into the widget.\n\t\tclamped (bool, optional): Applies the min and max limits to direct entry methods also such as double click and CTRL+Click.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_drag_double(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, width=width, indent=indent, parent=parent, before=before, source=source, payload_type=payload_type, callback=callback, drag_callback=drag_callback, drop_callback=drop_callback, show=show, enabled=enabled, pos=pos, filter_key=filter_key, tracked=tracked, track_offset=track_offset, default_value=default_value, format=format, speed=speed, min_value=min_value, max_value=max_value, no_input=no_input, clamped=clamped, **kwargs)

def add_drag_doublex(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, width: int=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, source: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', callback: Callable=None, drag_callback: Callable=None, drop_callback: Callable=None, show: bool=True, enabled: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], filter_key: str='', tracked: bool=False, track_offset: float=0.5, default_value: Any=(0.0, 0.0, 0.0, 0.0), size: int=4, format: str='%0.3f', speed: float=1.0, min_value: float=0.0, max_value: float=100.0, no_input: bool=False, clamped: bool=False, **kwargs) -> Union[int, str]:
    if False:
        i = 10
        return i + 15
    "\t Adds drag input for a set of double values up to 4. Useful when drag float is not accurate enough. Directly entry can be done with double click or CTRL+Click. Min and Max alone are a soft limit for the drag. Use clamped keyword to also apply limits to the direct entry modes.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (Any, optional): \n\t\tsize (int, optional): Number of doubles to be displayed.\n\t\tformat (str, optional): Determines the format the float will be displayed as use python string formatting.\n\t\tspeed (float, optional): Sets the sensitivity the float will be modified while dragging.\n\t\tmin_value (float, optional): Applies a limit only to draging entry only.\n\t\tmax_value (float, optional): Applies a limit only to draging entry only.\n\t\tno_input (bool, optional): Disable direct entry methods or Enter key allowing to input text directly into the widget.\n\t\tclamped (bool, optional): Applies the min and max limits to direct entry methods also such as double click and CTRL+Click.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_drag_doublex(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, width=width, indent=indent, parent=parent, before=before, source=source, payload_type=payload_type, callback=callback, drag_callback=drag_callback, drop_callback=drop_callback, show=show, enabled=enabled, pos=pos, filter_key=filter_key, tracked=tracked, track_offset=track_offset, default_value=default_value, size=size, format=format, speed=speed, min_value=min_value, max_value=max_value, no_input=no_input, clamped=clamped, **kwargs)

def add_drag_float(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, width: int=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, source: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', callback: Callable=None, drag_callback: Callable=None, drop_callback: Callable=None, show: bool=True, enabled: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], filter_key: str='', tracked: bool=False, track_offset: float=0.5, default_value: float=0.0, format: str='%0.3f', speed: float=1.0, min_value: float=0.0, max_value: float=100.0, no_input: bool=False, clamped: bool=False, **kwargs) -> Union[int, str]:
    if False:
        print('Hello World!')
    "\t Adds drag for a single float value. Directly entry can be done with double click or CTRL+Click. Min and Max alone are a soft limit for the drag. Use clamped keyword to also apply limits to the direct entry modes.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (float, optional): \n\t\tformat (str, optional): Determines the format the float will be displayed as use python string formatting.\n\t\tspeed (float, optional): Sets the sensitivity the float will be modified while dragging.\n\t\tmin_value (float, optional): Applies a limit only to draging entry only.\n\t\tmax_value (float, optional): Applies a limit only to draging entry only.\n\t\tno_input (bool, optional): Disable direct entry methods or Enter key allowing to input text directly into the widget.\n\t\tclamped (bool, optional): Applies the min and max limits to direct entry methods also such as double click and CTRL+Click.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_drag_float(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, width=width, indent=indent, parent=parent, before=before, source=source, payload_type=payload_type, callback=callback, drag_callback=drag_callback, drop_callback=drop_callback, show=show, enabled=enabled, pos=pos, filter_key=filter_key, tracked=tracked, track_offset=track_offset, default_value=default_value, format=format, speed=speed, min_value=min_value, max_value=max_value, no_input=no_input, clamped=clamped, **kwargs)

def add_drag_floatx(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, width: int=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, source: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', callback: Callable=None, drag_callback: Callable=None, drop_callback: Callable=None, show: bool=True, enabled: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], filter_key: str='', tracked: bool=False, track_offset: float=0.5, default_value: Union[List[float], Tuple[float, ...]]=(0.0, 0.0, 0.0, 0.0), size: int=4, format: str='%0.3f', speed: float=1.0, min_value: float=0.0, max_value: float=100.0, no_input: bool=False, clamped: bool=False, **kwargs) -> Union[int, str]:
    if False:
        while True:
            i = 10
    "\t Adds drag input for a set of float values up to 4. Directly entry can be done with double click or CTRL+Click. Min and Max alone are a soft limit for the drag. Use clamped keyword to also apply limits to the direct entry modes.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (Union[List[float], Tuple[float, ...]], optional): \n\t\tsize (int, optional): Number of floats to be displayed.\n\t\tformat (str, optional): Determines the format the float will be displayed as use python string formatting.\n\t\tspeed (float, optional): Sets the sensitivity the float will be modified while dragging.\n\t\tmin_value (float, optional): Applies a limit only to draging entry only.\n\t\tmax_value (float, optional): Applies a limit only to draging entry only.\n\t\tno_input (bool, optional): Disable direct entry methods or Enter key allowing to input text directly into the widget.\n\t\tclamped (bool, optional): Applies the min and max limits to direct entry methods also such as double click and CTRL+Click.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_drag_floatx(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, width=width, indent=indent, parent=parent, before=before, source=source, payload_type=payload_type, callback=callback, drag_callback=drag_callback, drop_callback=drop_callback, show=show, enabled=enabled, pos=pos, filter_key=filter_key, tracked=tracked, track_offset=track_offset, default_value=default_value, size=size, format=format, speed=speed, min_value=min_value, max_value=max_value, no_input=no_input, clamped=clamped, **kwargs)

def add_drag_int(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, width: int=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, source: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', callback: Callable=None, drag_callback: Callable=None, drop_callback: Callable=None, show: bool=True, enabled: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], filter_key: str='', tracked: bool=False, track_offset: float=0.5, default_value: int=0, format: str='%d', speed: float=1.0, min_value: int=0, max_value: int=100, no_input: bool=False, clamped: bool=False, **kwargs) -> Union[int, str]:
    if False:
        print('Hello World!')
    "\t Adds drag for a single int value. Directly entry can be done with double click or CTRL+Click. Min and Max alone are a soft limit for the drag. Use clamped keyword to also apply limits to the direct entry modes.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (int, optional): \n\t\tformat (str, optional): Determines the format the float will be displayed as use python string formatting.\n\t\tspeed (float, optional): Sets the sensitivity the float will be modified while dragging.\n\t\tmin_value (int, optional): Applies a limit only to draging entry only.\n\t\tmax_value (int, optional): Applies a limit only to draging entry only.\n\t\tno_input (bool, optional): Disable direct entry methods or Enter key allowing to input text directly into the widget.\n\t\tclamped (bool, optional): Applies the min and max limits to direct entry methods also such as double click and CTRL+Click.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_drag_int(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, width=width, indent=indent, parent=parent, before=before, source=source, payload_type=payload_type, callback=callback, drag_callback=drag_callback, drop_callback=drop_callback, show=show, enabled=enabled, pos=pos, filter_key=filter_key, tracked=tracked, track_offset=track_offset, default_value=default_value, format=format, speed=speed, min_value=min_value, max_value=max_value, no_input=no_input, clamped=clamped, **kwargs)

def add_drag_intx(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, width: int=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, source: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', callback: Callable=None, drag_callback: Callable=None, drop_callback: Callable=None, show: bool=True, enabled: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], filter_key: str='', tracked: bool=False, track_offset: float=0.5, default_value: Union[List[int], Tuple[int, ...]]=(0, 0, 0, 0), size: int=4, format: str='%d', speed: float=1.0, min_value: int=0, max_value: int=100, no_input: bool=False, clamped: bool=False, **kwargs) -> Union[int, str]:
    if False:
        print('Hello World!')
    "\t Adds drag input for a set of int values up to 4. Directly entry can be done with double click or CTRL+Click. Min and Max alone are a soft limit for the drag. Use clamped keyword to also apply limits to the direct entry modes.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (Union[List[int], Tuple[int, ...]], optional): \n\t\tsize (int, optional): Number of ints to be displayed.\n\t\tformat (str, optional): Determines the format the int will be displayed as use python string formatting.\n\t\tspeed (float, optional): Sets the sensitivity the float will be modified while dragging.\n\t\tmin_value (int, optional): Applies a limit only to draging entry only.\n\t\tmax_value (int, optional): Applies a limit only to draging entry only.\n\t\tno_input (bool, optional): Disable direct entry methods or Enter key allowing to input text directly into the widget.\n\t\tclamped (bool, optional): Applies the min and max limits to direct entry methods also such as double click and CTRL+Click.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_drag_intx(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, width=width, indent=indent, parent=parent, before=before, source=source, payload_type=payload_type, callback=callback, drag_callback=drag_callback, drop_callback=drop_callback, show=show, enabled=enabled, pos=pos, filter_key=filter_key, tracked=tracked, track_offset=track_offset, default_value=default_value, size=size, format=format, speed=speed, min_value=min_value, max_value=max_value, no_input=no_input, clamped=clamped, **kwargs)

def add_drag_line(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, before: Union[int, str]=0, source: Union[int, str]=0, callback: Callable=None, show: bool=True, default_value: Any=0.0, color: Union[List[int], Tuple[int, ...]]=(0, 0, 0, -255), thickness: float=1.0, show_label: bool=True, vertical: bool=True, **kwargs) -> Union[int, str]:
    if False:
        while True:
            i = 10
    "\t Adds a drag line to a plot.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tdefault_value (Any, optional): \n\t\tcolor (Union[List[int], Tuple[int, ...]], optional): \n\t\tthickness (float, optional): \n\t\tshow_label (bool, optional): \n\t\tvertical (bool, optional): \n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_drag_line(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, before=before, source=source, callback=callback, show=show, default_value=default_value, color=color, thickness=thickness, show_label=show_label, vertical=vertical, **kwargs)

def add_drag_payload(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, show: bool=True, drag_data: Any=None, drop_data: Any=None, payload_type: str='$$DPG_PAYLOAD', **kwargs) -> Union[int, str]:
    if False:
        return 10
    "\t User data payload for drag and drop operations.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tdrag_data (Any, optional): Drag data\n\t\tdrop_data (Any, optional): Drop data\n\t\tpayload_type (str, optional): \n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_drag_payload(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, show=show, drag_data=drag_data, drop_data=drop_data, payload_type=payload_type, **kwargs)

def add_drag_point(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, before: Union[int, str]=0, source: Union[int, str]=0, callback: Callable=None, show: bool=True, default_value: Any=(0.0, 0.0), color: Union[List[int], Tuple[int, ...]]=(0, 0, 0, -255), thickness: float=1.0, show_label: bool=True, **kwargs) -> Union[int, str]:
    if False:
        print('Hello World!')
    "\t Adds a drag point to a plot.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tdefault_value (Any, optional): \n\t\tcolor (Union[List[int], Tuple[int, ...]], optional): \n\t\tthickness (float, optional): \n\t\tshow_label (bool, optional): \n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_drag_point(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, before=before, source=source, callback=callback, show=show, default_value=default_value, color=color, thickness=thickness, show_label=show_label, **kwargs)

def add_draw_layer(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, before: Union[int, str]=0, show: bool=True, perspective_divide: bool=False, depth_clipping: bool=False, cull_mode: int=0, **kwargs) -> Union[int, str]:
    if False:
        i = 10
        return i + 15
    "\t New in 1.1. Creates a layer useful for grouping drawlist items.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tperspective_divide (bool, optional): New in 1.1. apply perspective divide\n\t\tdepth_clipping (bool, optional): New in 1.1. apply depth clipping\n\t\tcull_mode (int, optional): New in 1.1. culling mode, mvCullMode_* constants. Only works with triangles currently.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_draw_layer(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, before=before, show=show, perspective_divide=perspective_divide, depth_clipping=depth_clipping, cull_mode=cull_mode, **kwargs)

def add_draw_node(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, before: Union[int, str]=0, show: bool=True, **kwargs) -> Union[int, str]:
    if False:
        return 10
    "\t New in 1.1. Creates a drawing node to associate a transformation matrix. Child node matricies will concatenate.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_draw_node(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, before=before, show=show, **kwargs)

def add_drawlist(width: int, height: int, *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, before: Union[int, str]=0, callback: Callable=None, show: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], filter_key: str='', delay_search: bool=False, tracked: bool=False, track_offset: float=0.5, **kwargs) -> Union[int, str]:
    if False:
        while True:
            i = 10
    "\t Adds a drawing canvas.\n\n\tArgs:\n\t\twidth (int): \n\t\theight (int): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_drawlist(width, height, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, before=before, callback=callback, show=show, pos=pos, filter_key=filter_key, delay_search=delay_search, tracked=tracked, track_offset=track_offset, **kwargs)

def add_dynamic_texture(width: int, height: int, default_value: Union[List[float], Tuple[float, ...]], *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=internal_dpg.mvReservedUUID_2, **kwargs) -> Union[int, str]:
    if False:
        return 10
    "\t Adds a dynamic texture.\n\n\tArgs:\n\t\twidth (int): \n\t\theight (int): \n\t\tdefault_value (Union[List[float], Tuple[float, ...]]): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_dynamic_texture(width, height, default_value, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, **kwargs)

def add_error_series(x: Union[List[float], Tuple[float, ...]], y: Union[List[float], Tuple[float, ...]], negative: Union[List[float], Tuple[float, ...]], positive: Union[List[float], Tuple[float, ...]], *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, before: Union[int, str]=0, source: Union[int, str]=0, show: bool=True, contribute_to_bounds: bool=True, horizontal: bool=False, **kwargs) -> Union[int, str]:
    if False:
        i = 10
        return i + 15
    "\t Adds an error series to a plot.\n\n\tArgs:\n\t\tx (Any): \n\t\ty (Any): \n\t\tnegative (Any): \n\t\tpositive (Any): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tcontribute_to_bounds (bool, optional): \n\t\thorizontal (bool, optional): \n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_error_series(x, y, negative, positive, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, before=before, source=source, show=show, contribute_to_bounds=contribute_to_bounds, horizontal=horizontal, **kwargs)

def add_file_dialog(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, width: int=0, height: int=0, callback: Callable=None, show: bool=True, default_path: str='', default_filename: str='.', file_count: int=0, modal: bool=False, directory_selector: bool=False, min_size: Union[List[int], Tuple[int, ...]]=[100, 100], max_size: Union[List[int], Tuple[int, ...]]=[30000, 30000], cancel_callback: Callable=None, **kwargs) -> Union[int, str]:
    if False:
        while True:
            i = 10
    "\t Displays a file or directory selector depending on keywords. Displays a file dialog by default. Callback will be ran when the file or directory picker is closed. The app_data arguemnt will be populated with information related to the file and directory as a dictionary.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tdefault_path (str, optional): Path that the file dialog will default to when opened.\n\t\tdefault_filename (str, optional): Default name that will show in the file name input.\n\t\tfile_count (int, optional): Number of visible files in the dialog.\n\t\tmodal (bool, optional): Forces user interaction with the file selector.\n\t\tdirectory_selector (bool, optional): Shows only directory/paths as options. Allows selection of directory/paths only.\n\t\tmin_size (Union[List[int], Tuple[int, ...]], optional): Minimum window size.\n\t\tmax_size (Union[List[int], Tuple[int, ...]], optional): Maximum window size.\n\t\tcancel_callback (Callable, optional): Callback called when cancel button is clicked.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_file_dialog(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, width=width, height=height, callback=callback, show=show, default_path=default_path, default_filename=default_filename, file_count=file_count, modal=modal, directory_selector=directory_selector, min_size=min_size, max_size=max_size, cancel_callback=cancel_callback, **kwargs)

def add_file_extension(extension: str, *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, width: int=0, height: int=0, parent: Union[int, str]=0, before: Union[int, str]=0, custom_text: str='', color: Union[List[int], Tuple[int, ...]]=(-255, 0, 0, 255), **kwargs) -> Union[int, str]:
    if False:
        return 10
    "\t Creates a file extension filter option in the file dialog.\n\n\tArgs:\n\t\textension (str): Extension that will show as an when the parent is a file dialog.\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tcustom_text (str, optional): Replaces the displayed text in the drop down for this extension.\n\t\tcolor (Union[List[int], Tuple[int, ...]], optional): Color for the text that will be shown with specified extensions.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_file_extension(extension, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, width=width, height=height, parent=parent, before=before, custom_text=custom_text, color=color, **kwargs)

def add_filter_set(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, width: int=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, show: bool=True, delay_search: bool=False, **kwargs) -> Union[int, str]:
    if False:
        return 10
    "\t Helper to parse and apply text filters (e.g. aaaaa[, bbbbb][, ccccc])\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_filter_set(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, width=width, indent=indent, parent=parent, before=before, show=show, delay_search=delay_search, **kwargs)

def add_float4_value(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, source: Union[int, str]=0, default_value: Union[List[float], Tuple[float, ...]]=(0.0, 0.0, 0.0, 0.0), parent: Union[int, str]=internal_dpg.mvReservedUUID_3, **kwargs) -> Union[int, str]:
    if False:
        return 10
    "\t Adds a float4 value.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tdefault_value (Union[List[float], Tuple[float, ...]], optional): \n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_float4_value(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, source=source, default_value=default_value, parent=parent, **kwargs)

def add_float_value(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, source: Union[int, str]=0, default_value: float=0.0, parent: Union[int, str]=internal_dpg.mvReservedUUID_3, **kwargs) -> Union[int, str]:
    if False:
        for i in range(10):
            print('nop')
    "\t Adds a float value.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tdefault_value (float, optional): \n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_float_value(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, source=source, default_value=default_value, parent=parent, **kwargs)

def add_float_vect_value(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, source: Union[int, str]=0, default_value: Union[List[float], Tuple[float, ...]]=(), parent: Union[int, str]=internal_dpg.mvReservedUUID_3, **kwargs) -> Union[int, str]:
    if False:
        i = 10
        return i + 15
    "\t Adds a float vect value.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tdefault_value (Union[List[float], Tuple[float, ...]], optional): \n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_float_vect_value(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, source=source, default_value=default_value, parent=parent, **kwargs)

def add_font(file: str, size: int, *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, pixel_snapH: bool=False, parent: Union[int, str]=internal_dpg.mvReservedUUID_0, **kwargs) -> Union[int, str]:
    if False:
        return 10
    "\t Adds font to a font registry.\n\n\tArgs:\n\t\tfile (str): \n\t\tsize (int): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tpixel_snapH (bool, optional): Align every glyph to pixel boundary. Useful e.g. if you are merging a non-pixel aligned font with the default font, or rendering text piece-by-piece (e.g. for coloring).\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated) \n\t\tdefault_font (bool, optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    if 'default_font' in kwargs.keys():
        warnings.warn('default_font keyword removed', DeprecationWarning, 2)
        kwargs.pop('default_font', None)
    return internal_dpg.add_font(file, size, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, pixel_snapH=pixel_snapH, parent=parent, **kwargs)

def add_font_chars(chars: Union[List[int], Tuple[int, ...]], *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, **kwargs) -> Union[int, str]:
    if False:
        print('Hello World!')
    "\t Adds specific font characters to a font.\n\n\tArgs:\n\t\tchars (Union[List[int], Tuple[int, ...]]): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_font_chars(chars, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, **kwargs)

def add_font_range(first_char: int, last_char: int, *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, **kwargs) -> Union[int, str]:
    if False:
        return 10
    "\t Adds a range of font characters to a font.\n\n\tArgs:\n\t\tfirst_char (int): \n\t\tlast_char (int): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_font_range(first_char, last_char, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, **kwargs)

def add_font_range_hint(hint: int, *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, **kwargs) -> Union[int, str]:
    if False:
        return 10
    "\t Adds a range of font characters (mvFontRangeHint_ constants).\n\n\tArgs:\n\t\thint (int): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_font_range_hint(hint, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, **kwargs)

def add_font_registry(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, show: bool=True, **kwargs) -> Union[int, str]:
    if False:
        while True:
            i = 10
    "\t Adds a font registry.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_font_registry(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, show=show, **kwargs)

def add_group(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, width: int=0, height: int=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', drag_callback: Callable=None, drop_callback: Callable=None, show: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], filter_key: str='', delay_search: bool=False, tracked: bool=False, track_offset: float=0.5, horizontal: bool=False, horizontal_spacing: float=-1, xoffset: float=0.0, **kwargs) -> Union[int, str]:
    if False:
        for i in range(10):
            print('nop')
    "\t Creates a group that other widgets can belong to. The group allows item commands to be issued for all of its members.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\thorizontal (bool, optional): Forces child widgets to be added in a horizontal layout.\n\t\thorizontal_spacing (float, optional): Spacing for the horizontal layout.\n\t\txoffset (float, optional): Offset from containing window x item location within group.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_group(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, width=width, height=height, indent=indent, parent=parent, before=before, payload_type=payload_type, drag_callback=drag_callback, drop_callback=drop_callback, show=show, pos=pos, filter_key=filter_key, delay_search=delay_search, tracked=tracked, track_offset=track_offset, horizontal=horizontal, horizontal_spacing=horizontal_spacing, xoffset=xoffset, **kwargs)

def add_handler_registry(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, show: bool=True, **kwargs) -> Union[int, str]:
    if False:
        while True:
            i = 10
    "\t Adds a handler registry.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_handler_registry(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, show=show, **kwargs)

def add_heat_series(x: Union[List[float], Tuple[float, ...]], rows: int, cols: int, *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, before: Union[int, str]=0, source: Union[int, str]=0, show: bool=True, scale_min: float=0.0, scale_max: float=1.0, bounds_min: Any=(0.0, 0.0), bounds_max: Any=(1.0, 1.0), format: str='%0.1f', contribute_to_bounds: bool=True, **kwargs) -> Union[int, str]:
    if False:
        print('Hello World!')
    "\t Adds a heat series to a plot.\n\n\tArgs:\n\t\tx (Any): \n\t\trows (int): \n\t\tcols (int): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tscale_min (float, optional): Sets the color scale min. Typically paired with the color scale widget scale_min.\n\t\tscale_max (float, optional): Sets the color scale max. Typically paired with the color scale widget scale_max.\n\t\tbounds_min (Any, optional): \n\t\tbounds_max (Any, optional): \n\t\tformat (str, optional): \n\t\tcontribute_to_bounds (bool, optional): \n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_heat_series(x, rows, cols, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, before=before, source=source, show=show, scale_min=scale_min, scale_max=scale_max, bounds_min=bounds_min, bounds_max=bounds_max, format=format, contribute_to_bounds=contribute_to_bounds, **kwargs)

def add_histogram_series(x: Union[List[float], Tuple[float, ...]], *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, before: Union[int, str]=0, source: Union[int, str]=0, show: bool=True, bins: int=-1, bar_scale: float=1.0, min_range: float=0.0, max_range: float=1.0, cumlative: bool=False, density: bool=False, outliers: bool=True, contribute_to_bounds: bool=True, **kwargs) -> Union[int, str]:
    if False:
        return 10
    "\t Adds a histogram series to a plot.\n\n\tArgs:\n\t\tx (Any): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tbins (int, optional): \n\t\tbar_scale (float, optional): \n\t\tmin_range (float, optional): \n\t\tmax_range (float, optional): \n\t\tcumlative (bool, optional): \n\t\tdensity (bool, optional): \n\t\toutliers (bool, optional): \n\t\tcontribute_to_bounds (bool, optional): \n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_histogram_series(x, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, before=before, source=source, show=show, bins=bins, bar_scale=bar_scale, min_range=min_range, max_range=max_range, cumlative=cumlative, density=density, outliers=outliers, contribute_to_bounds=contribute_to_bounds, **kwargs)

def add_hline_series(x: Union[List[float], Tuple[float, ...]], *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, before: Union[int, str]=0, source: Union[int, str]=0, show: bool=True, **kwargs) -> Union[int, str]:
    if False:
        for i in range(10):
            print('nop')
    "\t Adds an infinite horizontal line series to a plot.\n\n\tArgs:\n\t\tx (Any): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_hline_series(x, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, before=before, source=source, show=show, **kwargs)

def add_image(texture_tag: Union[int, str], *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, width: int=0, height: int=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, source: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', drag_callback: Callable=None, drop_callback: Callable=None, show: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], filter_key: str='', tracked: bool=False, track_offset: float=0.5, tint_color: Union[List[float], Tuple[float, ...]]=(255, 255, 255, 255), border_color: Union[List[float], Tuple[float, ...]]=(0, 0, 0, 0), uv_min: Union[List[float], Tuple[float, ...]]=(0.0, 0.0), uv_max: Union[List[float], Tuple[float, ...]]=(1.0, 1.0), **kwargs) -> Union[int, str]:
    if False:
        return 10
    "\t Adds an image from a specified texture. uv_min and uv_max represent the normalized texture coordinates of the original image that will be shown. Using range (0.0,0.0)->(1.0,1.0) for texture coordinates will generally display the entire texture.\n\n\tArgs:\n\t\ttexture_tag (Union[int, str]): The texture_tag should come from a texture that was added to a texture registry.\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\ttint_color (Union[List[float], Tuple[float, ...]], optional): Applies a color tint to the entire texture.\n\t\tborder_color (Union[List[float], Tuple[float, ...]], optional): Displays a border of the specified color around the texture. If the theme style has turned off the border it will not be shown.\n\t\tuv_min (Union[List[float], Tuple[float, ...]], optional): Normalized texture coordinates min point.\n\t\tuv_max (Union[List[float], Tuple[float, ...]], optional): Normalized texture coordinates max point.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_image(texture_tag, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, width=width, height=height, indent=indent, parent=parent, before=before, source=source, payload_type=payload_type, drag_callback=drag_callback, drop_callback=drop_callback, show=show, pos=pos, filter_key=filter_key, tracked=tracked, track_offset=track_offset, tint_color=tint_color, border_color=border_color, uv_min=uv_min, uv_max=uv_max, **kwargs)

def add_image_button(texture_tag: Union[int, str], *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, width: int=0, height: int=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, source: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', callback: Callable=None, drag_callback: Callable=None, drop_callback: Callable=None, show: bool=True, enabled: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], filter_key: str='', tracked: bool=False, track_offset: float=0.5, frame_padding: int=-1, tint_color: Union[List[float], Tuple[float, ...]]=(255, 255, 255, 255), background_color: Union[List[float], Tuple[float, ...]]=(0, 0, 0, 0), uv_min: Union[List[float], Tuple[float, ...]]=(0.0, 0.0), uv_max: Union[List[float], Tuple[float, ...]]=(1.0, 1.0), **kwargs) -> Union[int, str]:
    if False:
        i = 10
        return i + 15
    "\t Adds an button with a texture. uv_min and uv_max represent the normalized texture coordinates of the original image that will be shown. Using range (0.0,0.0)->(1.0,1.0) texture coordinates will generally display the entire texture\n\n\tArgs:\n\t\ttexture_tag (Union[int, str]): The texture_tag should come from a texture that was added to a texture registry.\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tframe_padding (int, optional): Empty space around the outside of the texture. Button will show around the texture.\n\t\ttint_color (Union[List[float], Tuple[float, ...]], optional): Applies a color tint to the entire texture.\n\t\tbackground_color (Union[List[float], Tuple[float, ...]], optional): Displays a border of the specified color around the texture.\n\t\tuv_min (Union[List[float], Tuple[float, ...]], optional): Normalized texture coordinates min point.\n\t\tuv_max (Union[List[float], Tuple[float, ...]], optional): Normalized texture coordinates max point.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_image_button(texture_tag, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, width=width, height=height, indent=indent, parent=parent, before=before, source=source, payload_type=payload_type, callback=callback, drag_callback=drag_callback, drop_callback=drop_callback, show=show, enabled=enabled, pos=pos, filter_key=filter_key, tracked=tracked, track_offset=track_offset, frame_padding=frame_padding, tint_color=tint_color, background_color=background_color, uv_min=uv_min, uv_max=uv_max, **kwargs)

def add_image_series(texture_tag: Union[int, str], bounds_min: Union[List[float], Tuple[float, ...]], bounds_max: Union[List[float], Tuple[float, ...]], *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, before: Union[int, str]=0, source: Union[int, str]=0, show: bool=True, uv_min: Union[List[float], Tuple[float, ...]]=(0.0, 0.0), uv_max: Union[List[float], Tuple[float, ...]]=(1.0, 1.0), tint_color: Union[List[int], Tuple[int, ...]]=(255, 255, 255, 255), **kwargs) -> Union[int, str]:
    if False:
        for i in range(10):
            print('nop')
    "\t Adds an image series to a plot.\n\n\tArgs:\n\t\ttexture_tag (Union[int, str]): \n\t\tbounds_min (Any): \n\t\tbounds_max (Any): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tuv_min (Union[List[float], Tuple[float, ...]], optional): normalized texture coordinates\n\t\tuv_max (Union[List[float], Tuple[float, ...]], optional): normalized texture coordinates\n\t\ttint_color (Union[List[int], Tuple[int, ...]], optional): \n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_image_series(texture_tag, bounds_min, bounds_max, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, before=before, source=source, show=show, uv_min=uv_min, uv_max=uv_max, tint_color=tint_color, **kwargs)

def add_input_double(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, width: int=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, source: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', callback: Callable=None, drag_callback: Callable=None, drop_callback: Callable=None, show: bool=True, enabled: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], filter_key: str='', tracked: bool=False, track_offset: float=0.5, default_value: float=0.0, format: str='%.3f', min_value: float=0.0, max_value: float=100.0, step: float=0.1, step_fast: float=1.0, min_clamped: bool=False, max_clamped: bool=False, on_enter: bool=False, readonly: bool=False, **kwargs) -> Union[int, str]:
    if False:
        return 10
    "\t Adds input for an double. Useful when input float is not accurate enough. +/- buttons can be activated by setting the value of step.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (float, optional): \n\t\tformat (str, optional): Determines the format the float will be displayed as use python string formatting.\n\t\tmin_value (float, optional): Value for lower limit of input. By default this limits the step buttons. Use min_clamped to limit manual input.\n\t\tmax_value (float, optional): Value for upper limit of input. By default this limits the step buttons. Use max_clamped to limit manual input.\n\t\tstep (float, optional): Increment to change value by when the step buttons are pressed. Setting this and step_fast to a value of 0 or less will turn off step buttons.\n\t\tstep_fast (float, optional): Increment to change value by when ctrl + step buttons are pressed. Setting this and step to a value of 0 or less will turn off step buttons.\n\t\tmin_clamped (bool, optional): Activates and deactivates the enforcment of min_value.\n\t\tmax_clamped (bool, optional): Activates and deactivates the enforcment of max_value.\n\t\ton_enter (bool, optional): Only runs callback on enter key press.\n\t\treadonly (bool, optional): Activates read only mode where no text can be input but text can still be highlighted.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_input_double(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, width=width, indent=indent, parent=parent, before=before, source=source, payload_type=payload_type, callback=callback, drag_callback=drag_callback, drop_callback=drop_callback, show=show, enabled=enabled, pos=pos, filter_key=filter_key, tracked=tracked, track_offset=track_offset, default_value=default_value, format=format, min_value=min_value, max_value=max_value, step=step, step_fast=step_fast, min_clamped=min_clamped, max_clamped=max_clamped, on_enter=on_enter, readonly=readonly, **kwargs)

def add_input_doublex(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, width: int=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, source: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', callback: Callable=None, drag_callback: Callable=None, drop_callback: Callable=None, show: bool=True, enabled: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], filter_key: str='', tracked: bool=False, track_offset: float=0.5, default_value: Any=(0.0, 0.0, 0.0, 0.0), format: str='%.3f', min_value: float=0.0, max_value: float=100.0, size: int=4, min_clamped: bool=False, max_clamped: bool=False, on_enter: bool=False, readonly: bool=False, **kwargs) -> Union[int, str]:
    if False:
        while True:
            i = 10
    "\t Adds multi double input for up to 4 double values. Useful when input float mulit is not accurate enough.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (Any, optional): \n\t\tformat (str, optional): Determines the format the float will be displayed as use python string formatting.\n\t\tmin_value (float, optional): Value for lower limit of input for each cell. Use min_clamped to turn on.\n\t\tmax_value (float, optional): Value for upper limit of input for each cell. Use max_clamped to turn on.\n\t\tsize (int, optional): Number of components displayed for input.\n\t\tmin_clamped (bool, optional): Activates and deactivates the enforcment of min_value.\n\t\tmax_clamped (bool, optional): Activates and deactivates the enforcment of max_value.\n\t\ton_enter (bool, optional): Only runs callback on enter key press.\n\t\treadonly (bool, optional): Activates read only mode where no text can be input but text can still be highlighted.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_input_doublex(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, width=width, indent=indent, parent=parent, before=before, source=source, payload_type=payload_type, callback=callback, drag_callback=drag_callback, drop_callback=drop_callback, show=show, enabled=enabled, pos=pos, filter_key=filter_key, tracked=tracked, track_offset=track_offset, default_value=default_value, format=format, min_value=min_value, max_value=max_value, size=size, min_clamped=min_clamped, max_clamped=max_clamped, on_enter=on_enter, readonly=readonly, **kwargs)

def add_input_float(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, width: int=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, source: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', callback: Callable=None, drag_callback: Callable=None, drop_callback: Callable=None, show: bool=True, enabled: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], filter_key: str='', tracked: bool=False, track_offset: float=0.5, default_value: float=0.0, format: str='%.3f', min_value: float=0.0, max_value: float=100.0, step: float=0.1, step_fast: float=1.0, min_clamped: bool=False, max_clamped: bool=False, on_enter: bool=False, readonly: bool=False, **kwargs) -> Union[int, str]:
    if False:
        while True:
            i = 10
    "\t Adds input for an float. +/- buttons can be activated by setting the value of step.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (float, optional): \n\t\tformat (str, optional): Determines the format the float will be displayed as use python string formatting.\n\t\tmin_value (float, optional): Value for lower limit of input. By default this limits the step buttons. Use min_clamped to limit manual input.\n\t\tmax_value (float, optional): Value for upper limit of input. By default this limits the step buttons. Use max_clamped to limit manual input.\n\t\tstep (float, optional): Increment to change value by when the step buttons are pressed. Setting this and step_fast to a value of 0 or less will turn off step buttons.\n\t\tstep_fast (float, optional): Increment to change value by when ctrl + step buttons are pressed. Setting this and step to a value of 0 or less will turn off step buttons.\n\t\tmin_clamped (bool, optional): Activates and deactivates the enforcment of min_value.\n\t\tmax_clamped (bool, optional): Activates and deactivates the enforcment of max_value.\n\t\ton_enter (bool, optional): Only runs callback on enter key press.\n\t\treadonly (bool, optional): Activates read only mode where no text can be input but text can still be highlighted.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_input_float(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, width=width, indent=indent, parent=parent, before=before, source=source, payload_type=payload_type, callback=callback, drag_callback=drag_callback, drop_callback=drop_callback, show=show, enabled=enabled, pos=pos, filter_key=filter_key, tracked=tracked, track_offset=track_offset, default_value=default_value, format=format, min_value=min_value, max_value=max_value, step=step, step_fast=step_fast, min_clamped=min_clamped, max_clamped=max_clamped, on_enter=on_enter, readonly=readonly, **kwargs)

def add_input_floatx(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, width: int=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, source: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', callback: Callable=None, drag_callback: Callable=None, drop_callback: Callable=None, show: bool=True, enabled: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], filter_key: str='', tracked: bool=False, track_offset: float=0.5, default_value: Union[List[float], Tuple[float, ...]]=(0.0, 0.0, 0.0, 0.0), format: str='%.3f', min_value: float=0.0, max_value: float=100.0, size: int=4, min_clamped: bool=False, max_clamped: bool=False, on_enter: bool=False, readonly: bool=False, **kwargs) -> Union[int, str]:
    if False:
        i = 10
        return i + 15
    "\t Adds multi float input for up to 4 float values.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (Union[List[float], Tuple[float, ...]], optional): \n\t\tformat (str, optional): Determines the format the float will be displayed as use python string formatting.\n\t\tmin_value (float, optional): Value for lower limit of input for each cell. Use min_clamped to turn on.\n\t\tmax_value (float, optional): Value for upper limit of input for each cell. Use max_clamped to turn on.\n\t\tsize (int, optional): Number of components displayed for input.\n\t\tmin_clamped (bool, optional): Activates and deactivates the enforcment of min_value.\n\t\tmax_clamped (bool, optional): Activates and deactivates the enforcment of max_value.\n\t\ton_enter (bool, optional): Only runs callback on enter key press.\n\t\treadonly (bool, optional): Activates read only mode where no text can be input but text can still be highlighted.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_input_floatx(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, width=width, indent=indent, parent=parent, before=before, source=source, payload_type=payload_type, callback=callback, drag_callback=drag_callback, drop_callback=drop_callback, show=show, enabled=enabled, pos=pos, filter_key=filter_key, tracked=tracked, track_offset=track_offset, default_value=default_value, format=format, min_value=min_value, max_value=max_value, size=size, min_clamped=min_clamped, max_clamped=max_clamped, on_enter=on_enter, readonly=readonly, **kwargs)

def add_input_int(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, width: int=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, source: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', callback: Callable=None, drag_callback: Callable=None, drop_callback: Callable=None, show: bool=True, enabled: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], filter_key: str='', tracked: bool=False, track_offset: float=0.5, default_value: int=0, min_value: int=0, max_value: int=100, step: int=1, step_fast: int=100, min_clamped: bool=False, max_clamped: bool=False, on_enter: bool=False, readonly: bool=False, **kwargs) -> Union[int, str]:
    if False:
        print('Hello World!')
    "\t Adds input for an int. +/- buttons can be activated by setting the value of step.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (int, optional): \n\t\tmin_value (int, optional): Value for lower limit of input. By default this limits the step buttons. Use min_clamped to limit manual input.\n\t\tmax_value (int, optional): Value for upper limit of input. By default this limits the step buttons. Use max_clamped to limit manual input.\n\t\tstep (int, optional): Increment to change value by when the step buttons are pressed. Setting this and step_fast to a value of 0 or less will turn off step buttons.\n\t\tstep_fast (int, optional): Increment to change value by when ctrl + step buttons are pressed. Setting this and step to a value of 0 or less will turn off step buttons.\n\t\tmin_clamped (bool, optional): Activates and deactivates the enforcment of min_value.\n\t\tmax_clamped (bool, optional): Activates and deactivates the enforcment of max_value.\n\t\ton_enter (bool, optional): Only runs callback on enter key press.\n\t\treadonly (bool, optional): Activates read only mode where no text can be input but text can still be highlighted.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_input_int(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, width=width, indent=indent, parent=parent, before=before, source=source, payload_type=payload_type, callback=callback, drag_callback=drag_callback, drop_callback=drop_callback, show=show, enabled=enabled, pos=pos, filter_key=filter_key, tracked=tracked, track_offset=track_offset, default_value=default_value, min_value=min_value, max_value=max_value, step=step, step_fast=step_fast, min_clamped=min_clamped, max_clamped=max_clamped, on_enter=on_enter, readonly=readonly, **kwargs)

def add_input_intx(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, width: int=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, source: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', callback: Callable=None, drag_callback: Callable=None, drop_callback: Callable=None, show: bool=True, enabled: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], filter_key: str='', tracked: bool=False, track_offset: float=0.5, default_value: Union[List[int], Tuple[int, ...]]=(0, 0, 0, 0), min_value: int=0, max_value: int=100, size: int=4, min_clamped: bool=False, max_clamped: bool=False, on_enter: bool=False, readonly: bool=False, **kwargs) -> Union[int, str]:
    if False:
        return 10
    "\t Adds multi int input for up to 4 integer values.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (Union[List[int], Tuple[int, ...]], optional): \n\t\tmin_value (int, optional): Value for lower limit of input for each cell. Use min_clamped to turn on.\n\t\tmax_value (int, optional): Value for upper limit of input for each cell. Use max_clamped to turn on.\n\t\tsize (int, optional): Number of components displayed for input.\n\t\tmin_clamped (bool, optional): Activates and deactivates the enforcment of min_value.\n\t\tmax_clamped (bool, optional): Activates and deactivates the enforcment of max_value.\n\t\ton_enter (bool, optional): Only runs callback on enter.\n\t\treadonly (bool, optional): Activates read only mode where no text can be input but text can still be highlighted.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_input_intx(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, width=width, indent=indent, parent=parent, before=before, source=source, payload_type=payload_type, callback=callback, drag_callback=drag_callback, drop_callback=drop_callback, show=show, enabled=enabled, pos=pos, filter_key=filter_key, tracked=tracked, track_offset=track_offset, default_value=default_value, min_value=min_value, max_value=max_value, size=size, min_clamped=min_clamped, max_clamped=max_clamped, on_enter=on_enter, readonly=readonly, **kwargs)

def add_input_text(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, width: int=0, height: int=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, source: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', callback: Callable=None, drag_callback: Callable=None, drop_callback: Callable=None, show: bool=True, enabled: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], filter_key: str='', tracked: bool=False, track_offset: float=0.5, default_value: str='', hint: str='', multiline: bool=False, no_spaces: bool=False, uppercase: bool=False, tab_input: bool=False, decimal: bool=False, hexadecimal: bool=False, readonly: bool=False, password: bool=False, scientific: bool=False, on_enter: bool=False, **kwargs) -> Union[int, str]:
    if False:
        print('Hello World!')
    "\t Adds input for text.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (str, optional): \n\t\thint (str, optional): Displayed only when value is an empty string. Will reappear if input value is set to empty string. Will not show if default value is anything other than default empty string.\n\t\tmultiline (bool, optional): Allows for multiline text input.\n\t\tno_spaces (bool, optional): Filter out spaces and tabs.\n\t\tuppercase (bool, optional): Automatically make all inputs uppercase.\n\t\ttab_input (bool, optional): Allows tabs to be input into the string value instead of changing item focus.\n\t\tdecimal (bool, optional): Only allow characters 0123456789.+-*/\n\t\thexadecimal (bool, optional): Only allow characters 0123456789ABCDEFabcdef\n\t\treadonly (bool, optional): Activates read only mode where no text can be input but text can still be highlighted.\n\t\tpassword (bool, optional): Display all input characters as '*'.\n\t\tscientific (bool, optional): Only allow characters 0123456789.+-*/eE (Scientific notation input)\n\t\ton_enter (bool, optional): Only runs callback on enter key press.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_input_text(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, width=width, height=height, indent=indent, parent=parent, before=before, source=source, payload_type=payload_type, callback=callback, drag_callback=drag_callback, drop_callback=drop_callback, show=show, enabled=enabled, pos=pos, filter_key=filter_key, tracked=tracked, track_offset=track_offset, default_value=default_value, hint=hint, multiline=multiline, no_spaces=no_spaces, uppercase=uppercase, tab_input=tab_input, decimal=decimal, hexadecimal=hexadecimal, readonly=readonly, password=password, scientific=scientific, on_enter=on_enter, **kwargs)

def add_int4_value(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, source: Union[int, str]=0, default_value: Union[List[int], Tuple[int, ...]]=(0, 0, 0, 0), parent: Union[int, str]=internal_dpg.mvReservedUUID_3, **kwargs) -> Union[int, str]:
    if False:
        while True:
            i = 10
    "\t Adds a int4 value.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tdefault_value (Union[List[int], Tuple[int, ...]], optional): \n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_int4_value(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, source=source, default_value=default_value, parent=parent, **kwargs)

def add_int_value(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, source: Union[int, str]=0, default_value: int=0, parent: Union[int, str]=internal_dpg.mvReservedUUID_3, **kwargs) -> Union[int, str]:
    if False:
        for i in range(10):
            print('nop')
    "\t Adds a int value.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tdefault_value (int, optional): \n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_int_value(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, source=source, default_value=default_value, parent=parent, **kwargs)

def add_item_activated_handler(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, callback: Callable=None, show: bool=True, **kwargs) -> Union[int, str]:
    if False:
        i = 10
        return i + 15
    "\t Adds a activated handler.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_item_activated_handler(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, callback=callback, show=show, **kwargs)

def add_item_active_handler(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, callback: Callable=None, show: bool=True, **kwargs) -> Union[int, str]:
    if False:
        print('Hello World!')
    "\t Adds a active handler.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_item_active_handler(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, callback=callback, show=show, **kwargs)

def add_item_clicked_handler(button: int=-1, *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, callback: Callable=None, show: bool=True, **kwargs) -> Union[int, str]:
    if False:
        print('Hello World!')
    "\t Adds a clicked handler.\n\n\tArgs:\n\t\tbutton (int, optional): Submits callback for all mouse buttons\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_item_clicked_handler(button, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, callback=callback, show=show, **kwargs)

def add_item_deactivated_after_edit_handler(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, callback: Callable=None, show: bool=True, **kwargs) -> Union[int, str]:
    if False:
        i = 10
        return i + 15
    "\t Adds a deactivated after edit handler.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_item_deactivated_after_edit_handler(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, callback=callback, show=show, **kwargs)

def add_item_deactivated_handler(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, callback: Callable=None, show: bool=True, **kwargs) -> Union[int, str]:
    if False:
        print('Hello World!')
    "\t Adds a deactivated handler.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_item_deactivated_handler(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, callback=callback, show=show, **kwargs)

def add_item_double_clicked_handler(button: int=-1, *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, callback: Callable=None, show: bool=True, **kwargs) -> Union[int, str]:
    if False:
        print('Hello World!')
    "\t Adds a double click handler.\n\n\tArgs:\n\t\tbutton (int, optional): Submits callback for all mouse buttons\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_item_double_clicked_handler(button, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, callback=callback, show=show, **kwargs)

def add_item_edited_handler(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, callback: Callable=None, show: bool=True, **kwargs) -> Union[int, str]:
    if False:
        print('Hello World!')
    "\t Adds an edited handler.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_item_edited_handler(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, callback=callback, show=show, **kwargs)

def add_item_focus_handler(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, callback: Callable=None, show: bool=True, **kwargs) -> Union[int, str]:
    if False:
        return 10
    "\t Adds a focus handler.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_item_focus_handler(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, callback=callback, show=show, **kwargs)

def add_item_handler_registry(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, show: bool=True, **kwargs) -> Union[int, str]:
    if False:
        return 10
    "\t Adds an item handler registry.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_item_handler_registry(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, show=show, **kwargs)

def add_item_hover_handler(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, callback: Callable=None, show: bool=True, **kwargs) -> Union[int, str]:
    if False:
        return 10
    "\t Adds a hover handler.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_item_hover_handler(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, callback=callback, show=show, **kwargs)

def add_item_resize_handler(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, callback: Callable=None, show: bool=True, **kwargs) -> Union[int, str]:
    if False:
        return 10
    "\t Adds a resize handler.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_item_resize_handler(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, callback=callback, show=show, **kwargs)

def add_item_toggled_open_handler(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, callback: Callable=None, show: bool=True, **kwargs) -> Union[int, str]:
    if False:
        return 10
    "\t Adds a togged open handler.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_item_toggled_open_handler(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, callback=callback, show=show, **kwargs)

def add_item_visible_handler(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, callback: Callable=None, show: bool=True, **kwargs) -> Union[int, str]:
    if False:
        return 10
    "\t Adds a visible handler.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_item_visible_handler(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, callback=callback, show=show, **kwargs)

def add_key_down_handler(key: int=-1, *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, callback: Callable=None, show: bool=True, parent: Union[int, str]=internal_dpg.mvReservedUUID_1, **kwargs) -> Union[int, str]:
    if False:
        return 10
    "\t Adds a key down handler.\n\n\tArgs:\n\t\tkey (int, optional): Submits callback for all keys\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_key_down_handler(key, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, callback=callback, show=show, parent=parent, **kwargs)

def add_key_press_handler(key: int=-1, *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, callback: Callable=None, show: bool=True, parent: Union[int, str]=internal_dpg.mvReservedUUID_1, **kwargs) -> Union[int, str]:
    if False:
        while True:
            i = 10
    "\t Adds a key press handler.\n\n\tArgs:\n\t\tkey (int, optional): Submits callback for all keys\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_key_press_handler(key, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, callback=callback, show=show, parent=parent, **kwargs)

def add_key_release_handler(key: int=-1, *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, callback: Callable=None, show: bool=True, parent: Union[int, str]=internal_dpg.mvReservedUUID_1, **kwargs) -> Union[int, str]:
    if False:
        i = 10
        return i + 15
    "\t Adds a key release handler.\n\n\tArgs:\n\t\tkey (int, optional): Submits callback for all keys\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_key_release_handler(key, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, callback=callback, show=show, parent=parent, **kwargs)

def add_knob_float(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, width: int=0, height: int=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, source: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', callback: Callable=None, drag_callback: Callable=None, drop_callback: Callable=None, show: bool=True, enabled: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], filter_key: str='', tracked: bool=False, track_offset: float=0.5, default_value: float=0.0, min_value: float=0.0, max_value: float=100.0, **kwargs) -> Union[int, str]:
    if False:
        return 10
    "\t Adds a knob that rotates based on change in x mouse position.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (float, optional): \n\t\tmin_value (float, optional): Applies lower limit to value.\n\t\tmax_value (float, optional): Applies upper limit to value.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_knob_float(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, width=width, height=height, indent=indent, parent=parent, before=before, source=source, payload_type=payload_type, callback=callback, drag_callback=drag_callback, drop_callback=drop_callback, show=show, enabled=enabled, pos=pos, filter_key=filter_key, tracked=tracked, track_offset=track_offset, default_value=default_value, min_value=min_value, max_value=max_value, **kwargs)

def add_line_series(x: Union[List[float], Tuple[float, ...]], y: Union[List[float], Tuple[float, ...]], *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, before: Union[int, str]=0, source: Union[int, str]=0, show: bool=True, **kwargs) -> Union[int, str]:
    if False:
        return 10
    "\t Adds a line series to a plot.\n\n\tArgs:\n\t\tx (Any): \n\t\ty (Any): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_line_series(x, y, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, before=before, source=source, show=show, **kwargs)

def add_listbox(items: Union[List[str], Tuple[str, ...]]=(), *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, width: int=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, source: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', callback: Callable=None, drag_callback: Callable=None, drop_callback: Callable=None, show: bool=True, enabled: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], filter_key: str='', tracked: bool=False, track_offset: float=0.5, default_value: str='', num_items: int=3, **kwargs) -> Union[int, str]:
    if False:
        return 10
    "\t Adds a listbox. If height is not large enough to show all items a scroll bar will appear.\n\n\tArgs:\n\t\titems (Union[List[str], Tuple[str, ...]], optional): A tuple of items to be shown in the listbox. Can consist of any combination of types. All items will be displayed as strings.\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (str, optional): String value of the item that will be selected by default.\n\t\tnum_items (int, optional): Expands the height of the listbox to show specified number of items.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_listbox(items, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, width=width, indent=indent, parent=parent, before=before, source=source, payload_type=payload_type, callback=callback, drag_callback=drag_callback, drop_callback=drop_callback, show=show, enabled=enabled, pos=pos, filter_key=filter_key, tracked=tracked, track_offset=track_offset, default_value=default_value, num_items=num_items, **kwargs)

def add_loading_indicator(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, width: int=0, height: int=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', drop_callback: Callable=None, show: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], style: int=0, circle_count: int=8, speed: float=1.0, radius: float=3.0, thickness: float=1.0, color: Union[List[int], Tuple[int, ...]]=(51, 51, 55, 255), secondary_color: Union[List[int], Tuple[int, ...]]=(29, 151, 236, 103), **kwargs) -> Union[int, str]:
    if False:
        i = 10
        return i + 15
    "\t Adds a rotating animated loading symbol.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tstyle (int, optional): 0 is rotating dots style, 1 is rotating bar style.\n\t\tcircle_count (int, optional): Number of dots show if dots or size of circle if circle.\n\t\tspeed (float, optional): Speed the anamation will rotate.\n\t\tradius (float, optional): Radius size of the loading indicator.\n\t\tthickness (float, optional): Thickness of the circles or line.\n\t\tcolor (Union[List[int], Tuple[int, ...]], optional): Color of the growing center circle.\n\t\tsecondary_color (Union[List[int], Tuple[int, ...]], optional): Background of the dots in dot mode.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_loading_indicator(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, width=width, height=height, indent=indent, parent=parent, before=before, payload_type=payload_type, drop_callback=drop_callback, show=show, pos=pos, style=style, circle_count=circle_count, speed=speed, radius=radius, thickness=thickness, color=color, secondary_color=secondary_color, **kwargs)

def add_menu(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', drop_callback: Callable=None, show: bool=True, enabled: bool=True, filter_key: str='', delay_search: bool=False, tracked: bool=False, track_offset: float=0.5, **kwargs) -> Union[int, str]:
    if False:
        while True:
            i = 10
    "\t Adds a menu to an existing menu bar.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_menu(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, indent=indent, parent=parent, before=before, payload_type=payload_type, drop_callback=drop_callback, show=show, enabled=enabled, filter_key=filter_key, delay_search=delay_search, tracked=tracked, track_offset=track_offset, **kwargs)

def add_menu_bar(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, indent: int=-1, parent: Union[int, str]=0, show: bool=True, delay_search: bool=False, **kwargs) -> Union[int, str]:
    if False:
        for i in range(10):
            print('nop')
    "\t Adds a menu bar to a window.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_menu_bar(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, indent=indent, parent=parent, show=show, delay_search=delay_search, **kwargs)

def add_menu_item(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', callback: Callable=None, drop_callback: Callable=None, show: bool=True, enabled: bool=True, filter_key: str='', tracked: bool=False, track_offset: float=0.5, default_value: bool=False, shortcut: str='', check: bool=False, **kwargs) -> Union[int, str]:
    if False:
        for i in range(10):
            print('nop')
    "\t Adds a menu item to an existing menu. Menu items act similar to selectables and has a bool value. When placed in a menu the checkmark will reflect its value.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (bool, optional): This value also controls the checkmark when shown.\n\t\tshortcut (str, optional): Displays text on the menu item. Typically used to show a shortcut key command.\n\t\tcheck (bool, optional): Displays a checkmark on the menu item when it is selected and placed in a menu.\n\t\tid (Union[int, str], optional): (deprecated) \n\t\tdrag_callback (Callable, optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    if 'drag_callback' in kwargs.keys():
        warnings.warn('drag_callback keyword removed', DeprecationWarning, 2)
        kwargs.pop('drag_callback', None)
    return internal_dpg.add_menu_item(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, indent=indent, parent=parent, before=before, payload_type=payload_type, callback=callback, drop_callback=drop_callback, show=show, enabled=enabled, filter_key=filter_key, tracked=tracked, track_offset=track_offset, default_value=default_value, shortcut=shortcut, check=check, **kwargs)

def add_mouse_click_handler(button: int=-1, *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, callback: Callable=None, show: bool=True, parent: Union[int, str]=internal_dpg.mvReservedUUID_1, **kwargs) -> Union[int, str]:
    if False:
        i = 10
        return i + 15
    "\t Adds a mouse click handler.\n\n\tArgs:\n\t\tbutton (int, optional): Submits callback for all mouse buttons\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_mouse_click_handler(button, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, callback=callback, show=show, parent=parent, **kwargs)

def add_mouse_double_click_handler(button: int=-1, *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, callback: Callable=None, show: bool=True, parent: Union[int, str]=internal_dpg.mvReservedUUID_1, **kwargs) -> Union[int, str]:
    if False:
        i = 10
        return i + 15
    "\t Adds a mouse double click handler.\n\n\tArgs:\n\t\tbutton (int, optional): Submits callback for all mouse buttons\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_mouse_double_click_handler(button, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, callback=callback, show=show, parent=parent, **kwargs)

def add_mouse_down_handler(button: int=-1, *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, callback: Callable=None, show: bool=True, parent: Union[int, str]=internal_dpg.mvReservedUUID_1, **kwargs) -> Union[int, str]:
    if False:
        while True:
            i = 10
    "\t Adds a mouse down handler.\n\n\tArgs:\n\t\tbutton (int, optional): Submits callback for all mouse buttons\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_mouse_down_handler(button, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, callback=callback, show=show, parent=parent, **kwargs)

def add_mouse_drag_handler(button: int=-1, threshold: float=10.0, *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, callback: Callable=None, show: bool=True, parent: Union[int, str]=internal_dpg.mvReservedUUID_1, **kwargs) -> Union[int, str]:
    if False:
        print('Hello World!')
    "\t Adds a mouse drag handler.\n\n\tArgs:\n\t\tbutton (int, optional): Submits callback for all mouse buttons\n\t\tthreshold (float, optional): The threshold the mouse must be dragged before the callback is ran\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_mouse_drag_handler(button, threshold, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, callback=callback, show=show, parent=parent, **kwargs)

def add_mouse_move_handler(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, callback: Callable=None, show: bool=True, parent: Union[int, str]=internal_dpg.mvReservedUUID_1, **kwargs) -> Union[int, str]:
    if False:
        for i in range(10):
            print('nop')
    "\t Adds a mouse move handler.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_mouse_move_handler(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, callback=callback, show=show, parent=parent, **kwargs)

def add_mouse_release_handler(button: int=-1, *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, callback: Callable=None, show: bool=True, parent: Union[int, str]=internal_dpg.mvReservedUUID_1, **kwargs) -> Union[int, str]:
    if False:
        while True:
            i = 10
    "\t Adds a mouse release handler.\n\n\tArgs:\n\t\tbutton (int, optional): Submits callback for all mouse buttons\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_mouse_release_handler(button, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, callback=callback, show=show, parent=parent, **kwargs)

def add_mouse_wheel_handler(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, callback: Callable=None, show: bool=True, parent: Union[int, str]=internal_dpg.mvReservedUUID_1, **kwargs) -> Union[int, str]:
    if False:
        i = 10
        return i + 15
    "\t Adds a mouse wheel handler.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_mouse_wheel_handler(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, callback=callback, show=show, parent=parent, **kwargs)

def add_node(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, before: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', drag_callback: Callable=None, drop_callback: Callable=None, show: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], filter_key: str='', delay_search: bool=False, tracked: bool=False, track_offset: float=0.5, draggable: bool=True, **kwargs) -> Union[int, str]:
    if False:
        while True:
            i = 10
    "\t Adds a node to a node editor.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdraggable (bool, optional): Allow node to be draggable.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_node(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, before=before, payload_type=payload_type, drag_callback=drag_callback, drop_callback=drop_callback, show=show, pos=pos, filter_key=filter_key, delay_search=delay_search, tracked=tracked, track_offset=track_offset, draggable=draggable, **kwargs)

def add_node_attribute(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, show: bool=True, filter_key: str='', tracked: bool=False, track_offset: float=0.5, attribute_type: int=0, shape: int=1, category: str='general', **kwargs) -> Union[int, str]:
    if False:
        for i in range(10):
            print('nop')
    "\t Adds a node attribute to a node.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tattribute_type (int, optional): mvNode_Attr_Input, mvNode_Attr_Output, or mvNode_Attr_Static.\n\t\tshape (int, optional): Pin shape.\n\t\tcategory (str, optional): Category\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_node_attribute(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, indent=indent, parent=parent, before=before, show=show, filter_key=filter_key, tracked=tracked, track_offset=track_offset, attribute_type=attribute_type, shape=shape, category=category, **kwargs)

def add_node_editor(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, width: int=0, height: int=0, parent: Union[int, str]=0, before: Union[int, str]=0, callback: Callable=None, show: bool=True, filter_key: str='', delay_search: bool=False, tracked: bool=False, track_offset: float=0.5, delink_callback: Callable=None, menubar: bool=False, minimap: bool=False, minimap_location: int=2, **kwargs) -> Union[int, str]:
    if False:
        return 10
    "\t Adds a node editor.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdelink_callback (Callable, optional): Callback ran when a link is detached.\n\t\tmenubar (bool, optional): Shows or hides the menubar.\n\t\tminimap (bool, optional): Shows or hides the Minimap. New in 1.6.\n\t\tminimap_location (int, optional): mvNodeMiniMap_Location_* constants. New in 1.6.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_node_editor(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, width=width, height=height, parent=parent, before=before, callback=callback, show=show, filter_key=filter_key, delay_search=delay_search, tracked=tracked, track_offset=track_offset, delink_callback=delink_callback, menubar=menubar, minimap=minimap, minimap_location=minimap_location, **kwargs)

def add_node_link(attr_1: Union[int, str], attr_2: Union[int, str], *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, show: bool=True, **kwargs) -> Union[int, str]:
    if False:
        while True:
            i = 10
    "\t Adds a node link between 2 node attributes.\n\n\tArgs:\n\t\tattr_1 (Union[int, str]): \n\t\tattr_2 (Union[int, str]): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_node_link(attr_1, attr_2, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, show=show, **kwargs)

def add_pie_series(x: float, y: float, radius: float, values: Union[List[float], Tuple[float, ...]], labels: Union[List[str], Tuple[str, ...]], *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, before: Union[int, str]=0, source: Union[int, str]=0, show: bool=True, format: str='%0.2f', angle: float=90.0, normalize: bool=False, **kwargs) -> Union[int, str]:
    if False:
        i = 10
        return i + 15
    "\t Adds an pie series to a plot.\n\n\tArgs:\n\t\tx (float): \n\t\ty (float): \n\t\tradius (float): \n\t\tvalues (Any): \n\t\tlabels (Union[List[str], Tuple[str, ...]]): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tformat (str, optional): \n\t\tangle (float, optional): \n\t\tnormalize (bool, optional): \n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_pie_series(x, y, radius, values, labels, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, before=before, source=source, show=show, format=format, angle=angle, normalize=normalize, **kwargs)

def add_plot(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, width: int=0, height: int=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', callback: Callable=None, drag_callback: Callable=None, drop_callback: Callable=None, show: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], filter_key: str='', delay_search: bool=False, tracked: bool=False, track_offset: float=0.5, no_title: bool=False, no_menus: bool=False, no_box_select: bool=False, no_mouse_pos: bool=False, no_highlight: bool=False, no_child: bool=False, query: bool=False, crosshairs: bool=False, anti_aliased: bool=False, equal_aspects: bool=False, use_local_time: bool=False, use_ISO8601: bool=False, use_24hour_clock: bool=False, pan_button: int=internal_dpg.mvMouseButton_Left, pan_mod: int=-1, fit_button: int=internal_dpg.mvMouseButton_Left, context_menu_button: int=internal_dpg.mvMouseButton_Right, box_select_button: int=internal_dpg.mvMouseButton_Right, box_select_mod: int=-1, box_select_cancel_button: int=internal_dpg.mvMouseButton_Left, query_button: int=internal_dpg.mvMouseButton_Middle, query_mod: int=-1, query_toggle_mod: int=internal_dpg.mvKey_Control, horizontal_mod: int=internal_dpg.mvKey_Alt, vertical_mod: int=internal_dpg.mvKey_Shift, **kwargs) -> Union[int, str]:
    if False:
        i = 10
        return i + 15
    "\t Adds a plot which is used to hold series, and can be drawn to with draw commands.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tno_title (bool, optional): the plot title will not be displayed\n\t\tno_menus (bool, optional): the user will not be able to open context menus with right-click\n\t\tno_box_select (bool, optional): the user will not be able to box-select with right-click drag\n\t\tno_mouse_pos (bool, optional): the mouse position, in plot coordinates, will not be displayed inside of the plot\n\t\tno_highlight (bool, optional): plot items will not be highlighted when their legend entry is hovered\n\t\tno_child (bool, optional): a child window region will not be used to capture mouse scroll (can boost performance for single ImGui window applications)\n\t\tquery (bool, optional): the user will be able to draw query rects with middle - mouse or CTRL + right - click drag\n\t\tcrosshairs (bool, optional): the default mouse cursor will be replaced with a crosshair when hovered\n\t\tanti_aliased (bool, optional): plot lines will be software anti-aliased (not recommended for high density plots, prefer MSAA)\n\t\tequal_aspects (bool, optional): primary x and y axes will be constrained to have the same units/pixel (does not apply to auxiliary y-axes)\n\t\tuse_local_time (bool, optional): axis labels will be formatted for your timezone when\n\t\tuse_ISO8601 (bool, optional): dates will be formatted according to ISO 8601 where applicable (e.g. YYYY-MM-DD, YYYY-MM, --MM-DD, etc.)\n\t\tuse_24hour_clock (bool, optional): times will be formatted using a 24 hour clock\n\t\tpan_button (int, optional): enables panning when held\n\t\tpan_mod (int, optional): optional modifier that must be held for panning\n\t\tfit_button (int, optional): fits visible data when double clicked\n\t\tcontext_menu_button (int, optional): opens plot context menu (if enabled) when clicked\n\t\tbox_select_button (int, optional): begins box selection when pressed and confirms selection when released\n\t\tbox_select_mod (int, optional): begins box selection when pressed and confirms selection when released\n\t\tbox_select_cancel_button (int, optional): cancels active box selection when pressed\n\t\tquery_button (int, optional): begins query selection when pressed and end query selection when released\n\t\tquery_mod (int, optional): optional modifier that must be held for query selection\n\t\tquery_toggle_mod (int, optional): when held, active box selections turn into queries\n\t\thorizontal_mod (int, optional): expands active box selection/query horizontally to plot edge when held\n\t\tvertical_mod (int, optional): expands active box selection/query vertically to plot edge when held\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_plot(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, width=width, height=height, indent=indent, parent=parent, before=before, payload_type=payload_type, callback=callback, drag_callback=drag_callback, drop_callback=drop_callback, show=show, pos=pos, filter_key=filter_key, delay_search=delay_search, tracked=tracked, track_offset=track_offset, no_title=no_title, no_menus=no_menus, no_box_select=no_box_select, no_mouse_pos=no_mouse_pos, no_highlight=no_highlight, no_child=no_child, query=query, crosshairs=crosshairs, anti_aliased=anti_aliased, equal_aspects=equal_aspects, use_local_time=use_local_time, use_ISO8601=use_ISO8601, use_24hour_clock=use_24hour_clock, pan_button=pan_button, pan_mod=pan_mod, fit_button=fit_button, context_menu_button=context_menu_button, box_select_button=box_select_button, box_select_mod=box_select_mod, box_select_cancel_button=box_select_cancel_button, query_button=query_button, query_mod=query_mod, query_toggle_mod=query_toggle_mod, horizontal_mod=horizontal_mod, vertical_mod=vertical_mod, **kwargs)

def add_plot_annotation(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, before: Union[int, str]=0, source: Union[int, str]=0, show: bool=True, default_value: Any=(0.0, 0.0), offset: Union[List[float], Tuple[float, ...]]=(0.0, 0.0), color: Union[List[int], Tuple[int, ...]]=(0, 0, 0, -255), clamped: bool=True, **kwargs) -> Union[int, str]:
    if False:
        return 10
    "\t Adds an annotation to a plot.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tdefault_value (Any, optional): \n\t\toffset (Union[List[float], Tuple[float, ...]], optional): \n\t\tcolor (Union[List[int], Tuple[int, ...]], optional): \n\t\tclamped (bool, optional): \n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_plot_annotation(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, before=before, source=source, show=show, default_value=default_value, offset=offset, color=color, clamped=clamped, **kwargs)

def add_plot_axis(axis: int, *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', drop_callback: Callable=None, show: bool=True, no_gridlines: bool=False, no_tick_marks: bool=False, no_tick_labels: bool=False, log_scale: bool=False, invert: bool=False, lock_min: bool=False, lock_max: bool=False, time: bool=False, **kwargs) -> Union[int, str]:
    if False:
        i = 10
        return i + 15
    "\t Adds an axis to a plot.\n\n\tArgs:\n\t\taxis (int): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tno_gridlines (bool, optional): \n\t\tno_tick_marks (bool, optional): \n\t\tno_tick_labels (bool, optional): \n\t\tlog_scale (bool, optional): \n\t\tinvert (bool, optional): \n\t\tlock_min (bool, optional): \n\t\tlock_max (bool, optional): \n\t\ttime (bool, optional): \n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_plot_axis(axis, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, payload_type=payload_type, drop_callback=drop_callback, show=show, no_gridlines=no_gridlines, no_tick_marks=no_tick_marks, no_tick_labels=no_tick_labels, log_scale=log_scale, invert=invert, lock_min=lock_min, lock_max=lock_max, time=time, **kwargs)

def add_plot_legend(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', drop_callback: Callable=None, show: bool=True, location: int=5, horizontal: bool=False, outside: bool=False, **kwargs) -> Union[int, str]:
    if False:
        return 10
    "\t Adds a plot legend to a plot.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tlocation (int, optional): location, mvPlot_Location_*\n\t\thorizontal (bool, optional): \n\t\toutside (bool, optional): \n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_plot_legend(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, payload_type=payload_type, drop_callback=drop_callback, show=show, location=location, horizontal=horizontal, outside=outside, **kwargs)

def add_progress_bar(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, width: int=0, height: int=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, source: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', drag_callback: Callable=None, drop_callback: Callable=None, show: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], filter_key: str='', tracked: bool=False, track_offset: float=0.5, overlay: str='', default_value: float=0.0, **kwargs) -> Union[int, str]:
    if False:
        i = 10
        return i + 15
    "\t Adds a progress bar.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\toverlay (str, optional): Overlayed text onto the bar that typically used to display the value of the progress.\n\t\tdefault_value (float, optional): Normalized value to fill the bar from 0.0 to 1.0.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_progress_bar(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, width=width, height=height, indent=indent, parent=parent, before=before, source=source, payload_type=payload_type, drag_callback=drag_callback, drop_callback=drop_callback, show=show, pos=pos, filter_key=filter_key, tracked=tracked, track_offset=track_offset, overlay=overlay, default_value=default_value, **kwargs)

def add_radio_button(items: Union[List[str], Tuple[str, ...]]=(), *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, source: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', callback: Callable=None, drag_callback: Callable=None, drop_callback: Callable=None, show: bool=True, enabled: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], filter_key: str='', tracked: bool=False, track_offset: float=0.5, default_value: str='', horizontal: bool=False, **kwargs) -> Union[int, str]:
    if False:
        print('Hello World!')
    "\t Adds a set of radio buttons. If items keyword is empty, nothing will be shown.\n\n\tArgs:\n\t\titems (Union[List[str], Tuple[str, ...]], optional): A tuple of items to be shown as radio options. Can consist of any combination of types. All types will be shown as strings.\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (str, optional): Default selected radio option. Set by using the string value of the item.\n\t\thorizontal (bool, optional): Displays the radio options horizontally.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_radio_button(items, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, indent=indent, parent=parent, before=before, source=source, payload_type=payload_type, callback=callback, drag_callback=drag_callback, drop_callback=drop_callback, show=show, enabled=enabled, pos=pos, filter_key=filter_key, tracked=tracked, track_offset=track_offset, default_value=default_value, horizontal=horizontal, **kwargs)

def add_raw_texture(width: int, height: int, default_value: Union[List[float], Tuple[float, ...]], *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, format: int=internal_dpg.mvFormat_Float_rgba, parent: Union[int, str]=internal_dpg.mvReservedUUID_2, **kwargs) -> Union[int, str]:
    if False:
        return 10
    "\t Adds a raw texture.\n\n\tArgs:\n\t\twidth (int): \n\t\theight (int): \n\t\tdefault_value (Union[List[float], Tuple[float, ...]]): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tformat (int, optional): Data format.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_raw_texture(width, height, default_value, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, format=format, parent=parent, **kwargs)

def add_scatter_series(x: Union[List[float], Tuple[float, ...]], y: Union[List[float], Tuple[float, ...]], *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, before: Union[int, str]=0, source: Union[int, str]=0, show: bool=True, **kwargs) -> Union[int, str]:
    if False:
        print('Hello World!')
    "\t Adds a scatter series to a plot.\n\n\tArgs:\n\t\tx (Any): \n\t\ty (Any): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_scatter_series(x, y, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, before=before, source=source, show=show, **kwargs)

def add_selectable(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, width: int=0, height: int=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, source: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', callback: Callable=None, drag_callback: Callable=None, drop_callback: Callable=None, show: bool=True, enabled: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], filter_key: str='', tracked: bool=False, track_offset: float=0.5, default_value: bool=False, span_columns: bool=False, disable_popup_close: bool=False, **kwargs) -> Union[int, str]:
    if False:
        for i in range(10):
            print('nop')
    "\t Adds a selectable. Similar to a button but can indicate its selected state.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (bool, optional): \n\t\tspan_columns (bool, optional): Forces the selectable to span the width of all columns if placed in a table.\n\t\tdisable_popup_close (bool, optional): Disable closing a modal or popup window.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_selectable(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, width=width, height=height, indent=indent, parent=parent, before=before, source=source, payload_type=payload_type, callback=callback, drag_callback=drag_callback, drop_callback=drop_callback, show=show, enabled=enabled, pos=pos, filter_key=filter_key, tracked=tracked, track_offset=track_offset, default_value=default_value, span_columns=span_columns, disable_popup_close=disable_popup_close, **kwargs)

def add_separator(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, show: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], **kwargs) -> Union[int, str]:
    if False:
        while True:
            i = 10
    "\t Adds a horizontal line separator.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_separator(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, indent=indent, parent=parent, before=before, show=show, pos=pos, **kwargs)

def add_series_value(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, source: Union[int, str]=0, default_value: Any=(), parent: Union[int, str]=internal_dpg.mvReservedUUID_3, **kwargs) -> Union[int, str]:
    if False:
        while True:
            i = 10
    "\t Adds a plot series value.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tdefault_value (Any, optional): \n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_series_value(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, source=source, default_value=default_value, parent=parent, **kwargs)

def add_shade_series(x: Union[List[float], Tuple[float, ...]], y1: Union[List[float], Tuple[float, ...]], *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, before: Union[int, str]=0, source: Union[int, str]=0, show: bool=True, y2: Any=[], **kwargs) -> Union[int, str]:
    if False:
        i = 10
        return i + 15
    "\t Adds a shade series to a plot.\n\n\tArgs:\n\t\tx (Any): \n\t\ty1 (Any): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\ty2 (Any, optional): \n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_shade_series(x, y1, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, before=before, source=source, show=show, y2=y2, **kwargs)

def add_simple_plot(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, width: int=0, height: int=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, source: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', drag_callback: Callable=None, drop_callback: Callable=None, show: bool=True, filter_key: str='', tracked: bool=False, track_offset: float=0.5, default_value: Union[List[float], Tuple[float, ...]]=(), overlay: str='', histogram: bool=False, autosize: bool=True, min_scale: float=0.0, max_scale: float=0.0, **kwargs) -> Union[int, str]:
    if False:
        i = 10
        return i + 15
    "\t Adds a simple plot for visualization of a 1 dimensional set of values.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (Union[List[float], Tuple[float, ...]], optional): \n\t\toverlay (str, optional): overlays text (similar to a plot title)\n\t\thistogram (bool, optional): \n\t\tautosize (bool, optional): \n\t\tmin_scale (float, optional): \n\t\tmax_scale (float, optional): \n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_simple_plot(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, width=width, height=height, indent=indent, parent=parent, before=before, source=source, payload_type=payload_type, drag_callback=drag_callback, drop_callback=drop_callback, show=show, filter_key=filter_key, tracked=tracked, track_offset=track_offset, default_value=default_value, overlay=overlay, histogram=histogram, autosize=autosize, min_scale=min_scale, max_scale=max_scale, **kwargs)

def add_slider_double(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, width: int=0, height: int=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, source: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', callback: Callable=None, drag_callback: Callable=None, drop_callback: Callable=None, show: bool=True, enabled: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], filter_key: str='', tracked: bool=False, track_offset: float=0.5, default_value: float=0.0, vertical: bool=False, no_input: bool=False, clamped: bool=False, min_value: float=0.0, max_value: float=100.0, format: str='%.3f', **kwargs) -> Union[int, str]:
    if False:
        while True:
            i = 10
    "\t Adds slider for a single double value. Useful when slider float is not accurate enough. Directly entry can be done with double click or CTRL+Click. Min and Max alone are a soft limit for the slider. Use clamped keyword to also apply limits to the direct entry modes.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (float, optional): \n\t\tvertical (bool, optional): Sets orientation of the slidebar and slider to vertical.\n\t\tno_input (bool, optional): Disable direct entry methods double-click or ctrl+click or Enter key allowing to input text directly into the item.\n\t\tclamped (bool, optional): Applies the min and max limits to direct entry methods also such as double click and CTRL+Click.\n\t\tmin_value (float, optional): Applies a limit only to sliding entry only.\n\t\tmax_value (float, optional): Applies a limit only to sliding entry only.\n\t\tformat (str, optional): Determines the format the float will be displayed as use python string formatting.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_slider_double(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, width=width, height=height, indent=indent, parent=parent, before=before, source=source, payload_type=payload_type, callback=callback, drag_callback=drag_callback, drop_callback=drop_callback, show=show, enabled=enabled, pos=pos, filter_key=filter_key, tracked=tracked, track_offset=track_offset, default_value=default_value, vertical=vertical, no_input=no_input, clamped=clamped, min_value=min_value, max_value=max_value, format=format, **kwargs)

def add_slider_doublex(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, width: int=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, source: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', callback: Callable=None, drag_callback: Callable=None, drop_callback: Callable=None, show: bool=True, enabled: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], filter_key: str='', tracked: bool=False, track_offset: float=0.5, default_value: Any=(0.0, 0.0, 0.0, 0.0), size: int=4, no_input: bool=False, clamped: bool=False, min_value: float=0.0, max_value: float=100.0, format: str='%.3f', **kwargs) -> Union[int, str]:
    if False:
        while True:
            i = 10
    "\t Adds multi slider for up to 4 double values. Usueful for when multi slide float is not accurate enough. Directly entry can be done with double click or CTRL+Click. Min and Max alone are a soft limit for the slider. Use clamped keyword to also apply limits to the direct entry modes.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (Any, optional): \n\t\tsize (int, optional): Number of doubles to be displayed.\n\t\tno_input (bool, optional): Disable direct entry methods double-click or ctrl+click or Enter key allowing to input text directly into the item.\n\t\tclamped (bool, optional): Applies the min and max limits to direct entry methods also such as double click and CTRL+Click.\n\t\tmin_value (float, optional): Applies a limit only to sliding entry only.\n\t\tmax_value (float, optional): Applies a limit only to sliding entry only.\n\t\tformat (str, optional): Determines the format the int will be displayed as use python string formatting.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_slider_doublex(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, width=width, indent=indent, parent=parent, before=before, source=source, payload_type=payload_type, callback=callback, drag_callback=drag_callback, drop_callback=drop_callback, show=show, enabled=enabled, pos=pos, filter_key=filter_key, tracked=tracked, track_offset=track_offset, default_value=default_value, size=size, no_input=no_input, clamped=clamped, min_value=min_value, max_value=max_value, format=format, **kwargs)

def add_slider_float(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, width: int=0, height: int=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, source: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', callback: Callable=None, drag_callback: Callable=None, drop_callback: Callable=None, show: bool=True, enabled: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], filter_key: str='', tracked: bool=False, track_offset: float=0.5, default_value: float=0.0, vertical: bool=False, no_input: bool=False, clamped: bool=False, min_value: float=0.0, max_value: float=100.0, format: str='%.3f', **kwargs) -> Union[int, str]:
    if False:
        i = 10
        return i + 15
    "\t Adds slider for a single float value. Directly entry can be done with double click or CTRL+Click. Min and Max alone are a soft limit for the slider. Use clamped keyword to also apply limits to the direct entry modes.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (float, optional): \n\t\tvertical (bool, optional): Sets orientation of the slidebar and slider to vertical.\n\t\tno_input (bool, optional): Disable direct entry methods double-click or ctrl+click or Enter key allowing to input text directly into the item.\n\t\tclamped (bool, optional): Applies the min and max limits to direct entry methods also such as double click and CTRL+Click.\n\t\tmin_value (float, optional): Applies a limit only to sliding entry only.\n\t\tmax_value (float, optional): Applies a limit only to sliding entry only.\n\t\tformat (str, optional): Determines the format the float will be displayed as use python string formatting.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_slider_float(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, width=width, height=height, indent=indent, parent=parent, before=before, source=source, payload_type=payload_type, callback=callback, drag_callback=drag_callback, drop_callback=drop_callback, show=show, enabled=enabled, pos=pos, filter_key=filter_key, tracked=tracked, track_offset=track_offset, default_value=default_value, vertical=vertical, no_input=no_input, clamped=clamped, min_value=min_value, max_value=max_value, format=format, **kwargs)

def add_slider_floatx(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, width: int=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, source: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', callback: Callable=None, drag_callback: Callable=None, drop_callback: Callable=None, show: bool=True, enabled: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], filter_key: str='', tracked: bool=False, track_offset: float=0.5, default_value: Union[List[float], Tuple[float, ...]]=(0.0, 0.0, 0.0, 0.0), size: int=4, no_input: bool=False, clamped: bool=False, min_value: float=0.0, max_value: float=100.0, format: str='%.3f', **kwargs) -> Union[int, str]:
    if False:
        i = 10
        return i + 15
    "\t Adds multi slider for up to 4 float values. Directly entry can be done with double click or CTRL+Click. Min and Max alone are a soft limit for the slider. Use clamped keyword to also apply limits to the direct entry modes.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (Union[List[float], Tuple[float, ...]], optional): \n\t\tsize (int, optional): Number of floats to be displayed.\n\t\tno_input (bool, optional): Disable direct entry methods double-click or ctrl+click or Enter key allowing to input text directly into the item.\n\t\tclamped (bool, optional): Applies the min and max limits to direct entry methods also such as double click and CTRL+Click.\n\t\tmin_value (float, optional): Applies a limit only to sliding entry only.\n\t\tmax_value (float, optional): Applies a limit only to sliding entry only.\n\t\tformat (str, optional): Determines the format the int will be displayed as use python string formatting.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_slider_floatx(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, width=width, indent=indent, parent=parent, before=before, source=source, payload_type=payload_type, callback=callback, drag_callback=drag_callback, drop_callback=drop_callback, show=show, enabled=enabled, pos=pos, filter_key=filter_key, tracked=tracked, track_offset=track_offset, default_value=default_value, size=size, no_input=no_input, clamped=clamped, min_value=min_value, max_value=max_value, format=format, **kwargs)

def add_slider_int(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, width: int=0, height: int=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, source: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', callback: Callable=None, drag_callback: Callable=None, drop_callback: Callable=None, show: bool=True, enabled: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], filter_key: str='', tracked: bool=False, track_offset: float=0.5, default_value: int=0, vertical: bool=False, no_input: bool=False, clamped: bool=False, min_value: int=0, max_value: int=100, format: str='%d', **kwargs) -> Union[int, str]:
    if False:
        return 10
    "\t Adds slider for a single int value. Directly entry can be done with double click or CTRL+Click. Min and Max alone are a soft limit for the slider. Use clamped keyword to also apply limits to the direct entry modes.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (int, optional): \n\t\tvertical (bool, optional): Sets orientation of the slidebar and slider to vertical.\n\t\tno_input (bool, optional): Disable direct entry methods double-click or ctrl+click or Enter key allowing to input text directly into the item.\n\t\tclamped (bool, optional): Applies the min and max limits to direct entry methods also such as double click and CTRL+Click.\n\t\tmin_value (int, optional): Applies a limit only to sliding entry only.\n\t\tmax_value (int, optional): Applies a limit only to sliding entry only.\n\t\tformat (str, optional): Determines the format the int will be displayed as use python string formatting.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_slider_int(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, width=width, height=height, indent=indent, parent=parent, before=before, source=source, payload_type=payload_type, callback=callback, drag_callback=drag_callback, drop_callback=drop_callback, show=show, enabled=enabled, pos=pos, filter_key=filter_key, tracked=tracked, track_offset=track_offset, default_value=default_value, vertical=vertical, no_input=no_input, clamped=clamped, min_value=min_value, max_value=max_value, format=format, **kwargs)

def add_slider_intx(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, width: int=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, source: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', callback: Callable=None, drag_callback: Callable=None, drop_callback: Callable=None, show: bool=True, enabled: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], filter_key: str='', tracked: bool=False, track_offset: float=0.5, default_value: Union[List[int], Tuple[int, ...]]=(0, 0, 0, 0), size: int=4, no_input: bool=False, clamped: bool=False, min_value: int=0, max_value: int=100, format: str='%d', **kwargs) -> Union[int, str]:
    if False:
        while True:
            i = 10
    "\t Adds multi slider for up to 4 int values. Directly entry can be done with double click or CTRL+Click. Min and Max alone are a soft limit for the slider. Use clamped keyword to also apply limits to the direct entry modes.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (Union[List[int], Tuple[int, ...]], optional): \n\t\tsize (int, optional): Number of ints to be displayed.\n\t\tno_input (bool, optional): Disable direct entry methods double-click or ctrl+click or Enter key allowing to input text directly into the item.\n\t\tclamped (bool, optional): Applies the min and max limits to direct entry methods also such as double click and CTRL+Click.\n\t\tmin_value (int, optional): Applies a limit only to sliding entry only.\n\t\tmax_value (int, optional): Applies a limit only to sliding entry only.\n\t\tformat (str, optional): Determines the format the int will be displayed as use python string formatting.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_slider_intx(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, width=width, indent=indent, parent=parent, before=before, source=source, payload_type=payload_type, callback=callback, drag_callback=drag_callback, drop_callback=drop_callback, show=show, enabled=enabled, pos=pos, filter_key=filter_key, tracked=tracked, track_offset=track_offset, default_value=default_value, size=size, no_input=no_input, clamped=clamped, min_value=min_value, max_value=max_value, format=format, **kwargs)

def add_spacer(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, width: int=0, height: int=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, show: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], **kwargs) -> Union[int, str]:
    if False:
        return 10
    "\t Adds a spacer item that can be used to help with layouts or can be used as a placeholder item.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_spacer(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, width=width, height=height, indent=indent, parent=parent, before=before, show=show, pos=pos, **kwargs)

def add_stage(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, **kwargs) -> Union[int, str]:
    if False:
        return 10
    "\t Adds a stage.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_stage(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, **kwargs)

def add_stair_series(x: Union[List[float], Tuple[float, ...]], y: Union[List[float], Tuple[float, ...]], *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, before: Union[int, str]=0, source: Union[int, str]=0, show: bool=True, **kwargs) -> Union[int, str]:
    if False:
        return 10
    "\t Adds a stair series to a plot.\n\n\tArgs:\n\t\tx (Any): \n\t\ty (Any): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_stair_series(x, y, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, before=before, source=source, show=show, **kwargs)

def add_static_texture(width: int, height: int, default_value: Union[List[float], Tuple[float, ...]], *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=internal_dpg.mvReservedUUID_2, **kwargs) -> Union[int, str]:
    if False:
        while True:
            i = 10
    "\t Adds a static texture.\n\n\tArgs:\n\t\twidth (int): \n\t\theight (int): \n\t\tdefault_value (Union[List[float], Tuple[float, ...]]): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_static_texture(width, height, default_value, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, **kwargs)

def add_stem_series(x: Union[List[float], Tuple[float, ...]], y: Union[List[float], Tuple[float, ...]], *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, source: Union[int, str]=0, show: bool=True, **kwargs) -> Union[int, str]:
    if False:
        while True:
            i = 10
    "\t Adds a stem series to a plot.\n\n\tArgs:\n\t\tx (Any): \n\t\ty (Any): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_stem_series(x, y, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, indent=indent, parent=parent, before=before, source=source, show=show, **kwargs)

def add_string_value(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, source: Union[int, str]=0, default_value: str='', parent: Union[int, str]=internal_dpg.mvReservedUUID_3, **kwargs) -> Union[int, str]:
    if False:
        for i in range(10):
            print('nop')
    "\t Adds a string value.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tdefault_value (str, optional): \n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_string_value(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, source=source, default_value=default_value, parent=parent, **kwargs)

def add_subplots(rows: int, columns: int, *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, width: int=0, height: int=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, callback: Callable=None, show: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], filter_key: str='', delay_search: bool=False, tracked: bool=False, track_offset: float=0.5, row_ratios: Union[List[float], Tuple[float, ...]]=[], column_ratios: Union[List[float], Tuple[float, ...]]=[], no_title: bool=False, no_menus: bool=False, no_resize: bool=False, no_align: bool=False, link_rows: bool=False, link_columns: bool=False, link_all_x: bool=False, link_all_y: bool=False, column_major: bool=False, **kwargs) -> Union[int, str]:
    if False:
        for i in range(10):
            print('nop')
    "\t Adds a collection of plots.\n\n\tArgs:\n\t\trows (int): \n\t\tcolumns (int): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\trow_ratios (Union[List[float], Tuple[float, ...]], optional): \n\t\tcolumn_ratios (Union[List[float], Tuple[float, ...]], optional): \n\t\tno_title (bool, optional): \n\t\tno_menus (bool, optional): the user will not be able to open context menus with right-click\n\t\tno_resize (bool, optional): resize splitters between subplot cells will be not be provided\n\t\tno_align (bool, optional): subplot edges will not be aligned vertically or horizontally\n\t\tlink_rows (bool, optional): link the y-axis limits of all plots in each row (does not apply auxiliary y-axes)\n\t\tlink_columns (bool, optional): link the x-axis limits of all plots in each column\n\t\tlink_all_x (bool, optional): link the x-axis limits in every plot in the subplot\n\t\tlink_all_y (bool, optional): link the y-axis limits in every plot in the subplot (does not apply to auxiliary y-axes)\n\t\tcolumn_major (bool, optional): subplots are added in column major order instead of the default row major order\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_subplots(rows, columns, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, width=width, height=height, indent=indent, parent=parent, before=before, callback=callback, show=show, pos=pos, filter_key=filter_key, delay_search=delay_search, tracked=tracked, track_offset=track_offset, row_ratios=row_ratios, column_ratios=column_ratios, no_title=no_title, no_menus=no_menus, no_resize=no_resize, no_align=no_align, link_rows=link_rows, link_columns=link_columns, link_all_x=link_all_x, link_all_y=link_all_y, column_major=column_major, **kwargs)

def add_tab(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', drop_callback: Callable=None, show: bool=True, filter_key: str='', delay_search: bool=False, tracked: bool=False, track_offset: float=0.5, closable: bool=False, no_tooltip: bool=False, order_mode: bool=0, **kwargs) -> Union[int, str]:
    if False:
        print('Hello World!')
    "\t Adds a tab to a tab bar.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tclosable (bool, optional): Creates a button on the tab that can hide the tab.\n\t\tno_tooltip (bool, optional): Disable tooltip for the given tab.\n\t\torder_mode (bool, optional): set using a constant: mvTabOrder_Reorderable: allows reordering, mvTabOrder_Fixed: fixed ordering, mvTabOrder_Leading: adds tab to front, mvTabOrder_Trailing: adds tab to back\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_tab(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, indent=indent, parent=parent, before=before, payload_type=payload_type, drop_callback=drop_callback, show=show, filter_key=filter_key, delay_search=delay_search, tracked=tracked, track_offset=track_offset, closable=closable, no_tooltip=no_tooltip, order_mode=order_mode, **kwargs)

def add_tab_bar(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, callback: Callable=None, show: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], filter_key: str='', delay_search: bool=False, tracked: bool=False, track_offset: float=0.5, reorderable: bool=False, **kwargs) -> Union[int, str]:
    if False:
        while True:
            i = 10
    "\t Adds a tab bar.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\treorderable (bool, optional): Allows for the user to change the order of the tabs.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_tab_bar(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, indent=indent, parent=parent, before=before, callback=callback, show=show, pos=pos, filter_key=filter_key, delay_search=delay_search, tracked=tracked, track_offset=track_offset, reorderable=reorderable, **kwargs)

def add_tab_button(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', callback: Callable=None, drag_callback: Callable=None, drop_callback: Callable=None, show: bool=True, filter_key: str='', tracked: bool=False, track_offset: float=0.5, no_reorder: bool=False, leading: bool=False, trailing: bool=False, no_tooltip: bool=False, **kwargs) -> Union[int, str]:
    if False:
        i = 10
        return i + 15
    "\t Adds a tab button to a tab bar.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tno_reorder (bool, optional): Disable reordering this tab or having another tab cross over this tab. Fixes the position of this tab in relation to the order of neighboring tabs at start. \n\t\tleading (bool, optional): Enforce the tab position to the left of the tab bar (after the tab list popup button).\n\t\ttrailing (bool, optional): Enforce the tab position to the right of the tab bar (before the scrolling buttons).\n\t\tno_tooltip (bool, optional): Disable tooltip for the given tab.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_tab_button(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, indent=indent, parent=parent, before=before, payload_type=payload_type, callback=callback, drag_callback=drag_callback, drop_callback=drop_callback, show=show, filter_key=filter_key, tracked=tracked, track_offset=track_offset, no_reorder=no_reorder, leading=leading, trailing=trailing, no_tooltip=no_tooltip, **kwargs)

def add_table(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, width: int=0, height: int=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, source: Union[int, str]=0, callback: Callable=None, show: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], filter_key: str='', delay_search: bool=False, header_row: bool=True, clipper: bool=False, inner_width: int=0, policy: int=0, freeze_rows: int=0, freeze_columns: int=0, sort_multi: bool=False, sort_tristate: bool=False, resizable: bool=False, reorderable: bool=False, hideable: bool=False, sortable: bool=False, context_menu_in_body: bool=False, row_background: bool=False, borders_innerH: bool=False, borders_outerH: bool=False, borders_innerV: bool=False, borders_outerV: bool=False, no_host_extendX: bool=False, no_host_extendY: bool=False, no_keep_columns_visible: bool=False, precise_widths: bool=False, no_clip: bool=False, pad_outerX: bool=False, no_pad_outerX: bool=False, no_pad_innerX: bool=False, scrollX: bool=False, scrollY: bool=False, no_saved_settings: bool=False, **kwargs) -> Union[int, str]:
    if False:
        while True:
            i = 10
    "\t Adds a table.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\theader_row (bool, optional): show headers at the top of the columns\n\t\tclipper (bool, optional): Use clipper (rows must be same height).\n\t\tinner_width (int, optional): \n\t\tpolicy (int, optional): \n\t\tfreeze_rows (int, optional): \n\t\tfreeze_columns (int, optional): \n\t\tsort_multi (bool, optional): Hold shift when clicking headers to sort on multiple column.\n\t\tsort_tristate (bool, optional): Allow no sorting, disable default sorting.\n\t\tresizable (bool, optional): Enable resizing columns\n\t\treorderable (bool, optional): Enable reordering columns in header row (need calling TableSetupColumn() + TableHeadersRow() to display headers)\n\t\thideable (bool, optional): Enable hiding/disabling columns in context menu.\n\t\tsortable (bool, optional): Enable sorting. Call TableGetSortSpecs() to obtain sort specs. Also see ImGuiTableFlags_SortMulti and ImGuiTableFlags_SortTristate.\n\t\tcontext_menu_in_body (bool, optional): Right-click on columns body/contents will display table context menu. By default it is available in TableHeadersRow().\n\t\trow_background (bool, optional): Set each RowBg color with ImGuiCol_TableRowBg or ImGuiCol_TableRowBgAlt (equivalent of calling TableSetBgColor with ImGuiTableBgFlags_RowBg0 on each row manually)\n\t\tborders_innerH (bool, optional): Draw horizontal borders between rows.\n\t\tborders_outerH (bool, optional): Draw horizontal borders at the top and bottom.\n\t\tborders_innerV (bool, optional): Draw vertical borders between columns.\n\t\tborders_outerV (bool, optional): Draw vertical borders on the left and right sides.\n\t\tno_host_extendX (bool, optional): Make outer width auto-fit to columns, overriding outer_size.x value. Only available when ScrollX/ScrollY are disabled and Stretch columns are not used.\n\t\tno_host_extendY (bool, optional): Make outer height stop exactly at outer_size.y (prevent auto-extending table past the limit). Only available when ScrollX/ScrollY are disabled. Data below the limit will be clipped and not visible.\n\t\tno_keep_columns_visible (bool, optional): Disable keeping column always minimally visible when ScrollX is off and table gets too small. Not recommended if columns are resizable.\n\t\tprecise_widths (bool, optional): Disable distributing remainder width to stretched columns (width allocation on a 100-wide table with 3 columns: Without this flag: 33,33,34. With this flag: 33,33,33). With larger number of columns, resizing will appear to be less smooth.\n\t\tno_clip (bool, optional): Disable clipping rectangle for every individual columns.\n\t\tpad_outerX (bool, optional): Default if BordersOuterV is on. Enable outer-most padding. Generally desirable if you have headers.\n\t\tno_pad_outerX (bool, optional): Default if BordersOuterV is off. Disable outer-most padding.\n\t\tno_pad_innerX (bool, optional): Disable inner padding between columns (double inner padding if BordersOuterV is on, single inner padding if BordersOuterV is off).\n\t\tscrollX (bool, optional): Enable horizontal scrolling. Require 'outer_size' parameter of BeginTable() to specify the container size. Changes default sizing policy. Because this create a child window, ScrollY is currently generally recommended when using ScrollX.\n\t\tscrollY (bool, optional): Enable vertical scrolling.\n\t\tno_saved_settings (bool, optional): Never load/save settings in .ini file.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_table(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, width=width, height=height, indent=indent, parent=parent, before=before, source=source, callback=callback, show=show, pos=pos, filter_key=filter_key, delay_search=delay_search, header_row=header_row, clipper=clipper, inner_width=inner_width, policy=policy, freeze_rows=freeze_rows, freeze_columns=freeze_columns, sort_multi=sort_multi, sort_tristate=sort_tristate, resizable=resizable, reorderable=reorderable, hideable=hideable, sortable=sortable, context_menu_in_body=context_menu_in_body, row_background=row_background, borders_innerH=borders_innerH, borders_outerH=borders_outerH, borders_innerV=borders_innerV, borders_outerV=borders_outerV, no_host_extendX=no_host_extendX, no_host_extendY=no_host_extendY, no_keep_columns_visible=no_keep_columns_visible, precise_widths=precise_widths, no_clip=no_clip, pad_outerX=pad_outerX, no_pad_outerX=no_pad_outerX, no_pad_innerX=no_pad_innerX, scrollX=scrollX, scrollY=scrollY, no_saved_settings=no_saved_settings, **kwargs)

def add_table_cell(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, height: int=0, parent: Union[int, str]=0, before: Union[int, str]=0, show: bool=True, filter_key: str='', **kwargs) -> Union[int, str]:
    if False:
        while True:
            i = 10
    "\t Adds a table.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\theight (int, optional): Height of the item.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_table_cell(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, height=height, parent=parent, before=before, show=show, filter_key=filter_key, **kwargs)

def add_table_column(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, width: int=0, parent: Union[int, str]=0, before: Union[int, str]=0, show: bool=True, enabled: bool=True, init_width_or_weight: float=0.0, default_hide: bool=False, default_sort: bool=False, width_stretch: bool=False, width_fixed: bool=False, no_resize: bool=False, no_reorder: bool=False, no_hide: bool=False, no_clip: bool=False, no_sort: bool=False, no_sort_ascending: bool=False, no_sort_descending: bool=False, no_header_width: bool=False, prefer_sort_ascending: bool=True, prefer_sort_descending: bool=False, indent_enable: bool=False, indent_disable: bool=False, **kwargs) -> Union[int, str]:
    if False:
        print('Hello World!')
    "\t Adds a table column.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tinit_width_or_weight (float, optional): \n\t\tdefault_hide (bool, optional): Default as a hidden/disabled column.\n\t\tdefault_sort (bool, optional): Default as a sorting column.\n\t\twidth_stretch (bool, optional): Column will stretch. Preferable with horizontal scrolling disabled (default if table sizing policy is _SizingStretchSame or _SizingStretchProp).\n\t\twidth_fixed (bool, optional): Column will not stretch. Preferable with horizontal scrolling enabled (default if table sizing policy is _SizingFixedFit and table is resizable).\n\t\tno_resize (bool, optional): Disable manual resizing.\n\t\tno_reorder (bool, optional): Disable manual reordering this column, this will also prevent other columns from crossing over this column.\n\t\tno_hide (bool, optional): Disable ability to hide/disable this column.\n\t\tno_clip (bool, optional): Disable clipping for this column (all NoClip columns will render in a same draw command).\n\t\tno_sort (bool, optional): Disable ability to sort on this field (even if ImGuiTableFlags_Sortable is set on the table).\n\t\tno_sort_ascending (bool, optional): Disable ability to sort in the ascending direction.\n\t\tno_sort_descending (bool, optional): Disable ability to sort in the descending direction.\n\t\tno_header_width (bool, optional): Disable header text width contribution to automatic column width.\n\t\tprefer_sort_ascending (bool, optional): Make the initial sort direction Ascending when first sorting on this column (default).\n\t\tprefer_sort_descending (bool, optional): Make the initial sort direction Descending when first sorting on this column.\n\t\tindent_enable (bool, optional): Use current Indent value when entering cell (default for column 0).\n\t\tindent_disable (bool, optional): Ignore current Indent value when entering cell (default for columns > 0). Indentation changes _within_ the cell will still be honored.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_table_column(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, width=width, parent=parent, before=before, show=show, enabled=enabled, init_width_or_weight=init_width_or_weight, default_hide=default_hide, default_sort=default_sort, width_stretch=width_stretch, width_fixed=width_fixed, no_resize=no_resize, no_reorder=no_reorder, no_hide=no_hide, no_clip=no_clip, no_sort=no_sort, no_sort_ascending=no_sort_ascending, no_sort_descending=no_sort_descending, no_header_width=no_header_width, prefer_sort_ascending=prefer_sort_ascending, prefer_sort_descending=prefer_sort_descending, indent_enable=indent_enable, indent_disable=indent_disable, **kwargs)

def add_table_row(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, height: int=0, parent: Union[int, str]=0, before: Union[int, str]=0, show: bool=True, filter_key: str='', **kwargs) -> Union[int, str]:
    if False:
        return 10
    "\t Adds a table row.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\theight (int, optional): Height of the item.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_table_row(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, height=height, parent=parent, before=before, show=show, filter_key=filter_key, **kwargs)

def add_template_registry(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, **kwargs) -> Union[int, str]:
    if False:
        while True:
            i = 10
    "\t Adds a template registry.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_template_registry(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, **kwargs)

def add_text(default_value: str='', *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, source: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', drag_callback: Callable=None, drop_callback: Callable=None, show: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], filter_key: str='', tracked: bool=False, track_offset: float=0.5, wrap: int=-1, bullet: bool=False, color: Union[List[int], Tuple[int, ...]]=(-255, 0, 0, 255), show_label: bool=False, **kwargs) -> Union[int, str]:
    if False:
        i = 10
        return i + 15
    "\t Adds text. Text can have an optional label that will display to the right of the text.\n\n\tArgs:\n\t\tdefault_value (str, optional): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\twrap (int, optional): Number of pixels from the start of the item until wrapping starts.\n\t\tbullet (bool, optional): Places a bullet to the left of the text.\n\t\tcolor (Union[List[int], Tuple[int, ...]], optional): Color of the text (rgba).\n\t\tshow_label (bool, optional): Displays the label to the right of the text.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_text(default_value, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, indent=indent, parent=parent, before=before, source=source, payload_type=payload_type, drag_callback=drag_callback, drop_callback=drop_callback, show=show, pos=pos, filter_key=filter_key, tracked=tracked, track_offset=track_offset, wrap=wrap, bullet=bullet, color=color, show_label=show_label, **kwargs)

def add_text_point(x: float, y: float, *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, before: Union[int, str]=0, source: Union[int, str]=0, show: bool=True, x_offset: int=..., y_offset: int=..., vertical: bool=False, **kwargs) -> Union[int, str]:
    if False:
        for i in range(10):
            print('nop')
    "\t Adds a label series to a plot.\n\n\tArgs:\n\t\tx (float): \n\t\ty (float): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tx_offset (int, optional): \n\t\ty_offset (int, optional): \n\t\tvertical (bool, optional): \n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_text_point(x, y, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, before=before, source=source, show=show, x_offset=x_offset, y_offset=y_offset, vertical=vertical, **kwargs)

def add_texture_registry(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, show: bool=False, **kwargs) -> Union[int, str]:
    if False:
        for i in range(10):
            print('nop')
    "\t Adds a dynamic texture.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_texture_registry(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, show=show, **kwargs)

def add_theme(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, **kwargs) -> Union[int, str]:
    if False:
        return 10
    "\t Adds a theme.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tid (Union[int, str], optional): (deprecated) \n\t\tdefault_theme (bool, optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    if 'default_theme' in kwargs.keys():
        warnings.warn('default_theme keyword removed', DeprecationWarning, 2)
        kwargs.pop('default_theme', None)
    return internal_dpg.add_theme(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, **kwargs)

def add_theme_color(target: int=0, value: Union[List[int], Tuple[int, ...]]=(0, 0, 0, 255), *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, category: int=0, **kwargs) -> Union[int, str]:
    if False:
        print('Hello World!')
    "\t Adds a theme color.\n\n\tArgs:\n\t\ttarget (int, optional): \n\t\tvalue (Union[List[int], Tuple[int, ...]], optional): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tcategory (int, optional): Options include mvThemeCat_Core, mvThemeCat_Plots, mvThemeCat_Nodes.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_theme_color(target, value, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, category=category, **kwargs)

def add_theme_component(item_type: int=0, *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, before: Union[int, str]=0, enabled_state: bool=True, **kwargs) -> Union[int, str]:
    if False:
        while True:
            i = 10
    "\t Adds a theme component.\n\n\tArgs:\n\t\titem_type (int, optional): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tenabled_state (bool, optional): \n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_theme_component(item_type, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, before=before, enabled_state=enabled_state, **kwargs)

def add_theme_style(target: int=0, x: float=1.0, y: float=-1.0, *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, category: int=0, **kwargs) -> Union[int, str]:
    if False:
        while True:
            i = 10
    "\t Adds a theme style.\n\n\tArgs:\n\t\ttarget (int, optional): \n\t\tx (float, optional): \n\t\ty (float, optional): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tcategory (int, optional): Options include mvThemeCat_Core, mvThemeCat_Plots, mvThemeCat_Nodes.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_theme_style(target, x, y, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, category=category, **kwargs)

def add_time_picker(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', callback: Callable=None, drag_callback: Callable=None, drop_callback: Callable=None, show: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], filter_key: str='', tracked: bool=False, track_offset: float=0.5, default_value: dict={'hour': 14, 'min': 32, 'sec': 23}, hour24: bool=False, **kwargs) -> Union[int, str]:
    if False:
        while True:
            i = 10
    "\t Adds a time picker.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (dict, optional): \n\t\thour24 (bool, optional): Show 24 hour clock instead of 12 hour.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_time_picker(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, indent=indent, parent=parent, before=before, payload_type=payload_type, callback=callback, drag_callback=drag_callback, drop_callback=drop_callback, show=show, pos=pos, filter_key=filter_key, tracked=tracked, track_offset=track_offset, default_value=default_value, hour24=hour24, **kwargs)

def add_tooltip(parent: Union[int, str], *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, show: bool=True, delay: float=0.0, hide_on_activity: bool=False, **kwargs) -> Union[int, str]:
    if False:
        while True:
            i = 10
    "\t Adds a tooltip window.\n\n\tArgs:\n\t\tparent (Union[int, str]): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tdelay (float, optional): Activation delay: time, in seconds, during which the mouse should stay still in order to display the tooltip.  May be zero for instant activation.\n\t\thide_on_activity (bool, optional): Hide the tooltip if the user has moved the mouse.  If False, the tooltip will follow mouse pointer.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_tooltip(parent, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, show=show, delay=delay, hide_on_activity=hide_on_activity, **kwargs)

def add_tree_node(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, indent: int=-1, parent: Union[int, str]=0, before: Union[int, str]=0, payload_type: str='$$DPG_PAYLOAD', drag_callback: Callable=None, drop_callback: Callable=None, show: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], filter_key: str='', delay_search: bool=False, tracked: bool=False, track_offset: float=0.5, default_open: bool=False, open_on_double_click: bool=False, open_on_arrow: bool=False, leaf: bool=False, bullet: bool=False, selectable: bool=False, **kwargs) -> Union[int, str]:
    if False:
        for i in range(10):
            print('nop')
    "\t Adds a tree node to add items to.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_open (bool, optional): Sets the tree node open by default.\n\t\topen_on_double_click (bool, optional): Need double-click to open node.\n\t\topen_on_arrow (bool, optional): Only open when clicking on the arrow part.\n\t\tleaf (bool, optional): No collapsing, no arrow (use as a convenience for leaf nodes).\n\t\tbullet (bool, optional): Display a bullet instead of arrow.\n\t\tselectable (bool, optional): Makes the tree selectable.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_tree_node(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, indent=indent, parent=parent, before=before, payload_type=payload_type, drag_callback=drag_callback, drop_callback=drop_callback, show=show, pos=pos, filter_key=filter_key, delay_search=delay_search, tracked=tracked, track_offset=track_offset, default_open=default_open, open_on_double_click=open_on_double_click, open_on_arrow=open_on_arrow, leaf=leaf, bullet=bullet, selectable=selectable, **kwargs)

def add_value_registry(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, **kwargs) -> Union[int, str]:
    if False:
        i = 10
        return i + 15
    "\t Adds a value registry.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_value_registry(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, **kwargs)

def add_viewport_drawlist(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, show: bool=True, filter_key: str='', delay_search: bool=False, front: bool=True, **kwargs) -> Union[int, str]:
    if False:
        while True:
            i = 10
    "\t A container that is used to present draw items or layers directly to the viewport. By default this will draw to the back of the viewport. Layers and draw items should be added to this widget as children.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\tfront (bool, optional): Draws to the front of the view port instead of the back.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_viewport_drawlist(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, show=show, filter_key=filter_key, delay_search=delay_search, front=front, **kwargs)

def add_viewport_menu_bar(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, indent: int=-1, parent: Union[int, str]=0, show: bool=True, delay_search: bool=False, **kwargs) -> Union[int, str]:
    if False:
        while True:
            i = 10
    "\t Adds a menubar to the viewport.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_viewport_menu_bar(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, indent=indent, parent=parent, show=show, delay_search=delay_search, **kwargs)

def add_vline_series(x: Union[List[float], Tuple[float, ...]], *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, before: Union[int, str]=0, source: Union[int, str]=0, show: bool=True, **kwargs) -> Union[int, str]:
    if False:
        print('Hello World!')
    "\t Adds an infinite vertical line series to a plot.\n\n\tArgs:\n\t\tx (Any): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_vline_series(x, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, before=before, source=source, show=show, **kwargs)

def add_window(*, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, width: int=0, height: int=0, indent: int=-1, show: bool=True, pos: Union[List[int], Tuple[int, ...]]=[], delay_search: bool=False, min_size: Union[List[int], Tuple[int, ...]]=[100, 100], max_size: Union[List[int], Tuple[int, ...]]=[30000, 30000], menubar: bool=False, collapsed: bool=False, autosize: bool=False, no_resize: bool=False, no_title_bar: bool=False, no_move: bool=False, no_scrollbar: bool=False, no_collapse: bool=False, horizontal_scrollbar: bool=False, no_focus_on_appearing: bool=False, no_bring_to_front_on_focus: bool=False, no_close: bool=False, no_background: bool=False, modal: bool=False, popup: bool=False, no_saved_settings: bool=False, no_open_over_existing_popup: bool=True, no_scroll_with_mouse: bool=False, on_close: Callable=None, **kwargs) -> Union[int, str]:
    if False:
        print('Hello World!')
    "\t Creates a new window for following items to be added to.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\tmin_size (Union[List[int], Tuple[int, ...]], optional): Minimum window size.\n\t\tmax_size (Union[List[int], Tuple[int, ...]], optional): Maximum window size.\n\t\tmenubar (bool, optional): Shows or hides the menubar.\n\t\tcollapsed (bool, optional): Collapse the window.\n\t\tautosize (bool, optional): Autosized the window to fit it's items.\n\t\tno_resize (bool, optional): Allows for the window size to be changed or fixed.\n\t\tno_title_bar (bool, optional): Title name for the title bar of the window.\n\t\tno_move (bool, optional): Allows for the window's position to be changed or fixed.\n\t\tno_scrollbar (bool, optional):  Disable scrollbars. (window can still scroll with mouse or programmatically)\n\t\tno_collapse (bool, optional): Disable user collapsing window by double-clicking on it.\n\t\thorizontal_scrollbar (bool, optional): Allow horizontal scrollbar to appear. (off by default)\n\t\tno_focus_on_appearing (bool, optional): Disable taking focus when transitioning from hidden to visible state.\n\t\tno_bring_to_front_on_focus (bool, optional): Disable bringing window to front when taking focus. (e.g. clicking on it or programmatically giving it focus)\n\t\tno_close (bool, optional): Disable user closing the window by removing the close button.\n\t\tno_background (bool, optional): Sets Background and border alpha to transparent.\n\t\tmodal (bool, optional): Fills area behind window according to the theme and disables user ability to interact with anything except the window.\n\t\tpopup (bool, optional): Fills area behind window according to the theme, removes title bar, collapse and close. Window can be closed by selecting area in the background behind the window.\n\t\tno_saved_settings (bool, optional): Never load/save settings in .ini file.\n\t\tno_open_over_existing_popup (bool, optional): Don't open if there's already a popup\n\t\tno_scroll_with_mouse (bool, optional): Disable user vertically scrolling with mouse wheel.\n\t\ton_close (Callable, optional): Callback ran when window is closed.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.add_window(label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, width=width, height=height, indent=indent, show=show, pos=pos, delay_search=delay_search, min_size=min_size, max_size=max_size, menubar=menubar, collapsed=collapsed, autosize=autosize, no_resize=no_resize, no_title_bar=no_title_bar, no_move=no_move, no_scrollbar=no_scrollbar, no_collapse=no_collapse, horizontal_scrollbar=horizontal_scrollbar, no_focus_on_appearing=no_focus_on_appearing, no_bring_to_front_on_focus=no_bring_to_front_on_focus, no_close=no_close, no_background=no_background, modal=modal, popup=popup, no_saved_settings=no_saved_settings, no_open_over_existing_popup=no_open_over_existing_popup, no_scroll_with_mouse=no_scroll_with_mouse, on_close=on_close, **kwargs)

def apply_transform(item: Union[int, str], transform: Any, **kwargs) -> None:
    if False:
        while True:
            i = 10
    '\t New in 1.1. Applies a transformation matrix to a layer.\n\n\tArgs:\n\t\titem (Union[int, str]): Drawing node to apply transform to.\n\t\ttransform (Any): Transformation matrix.\n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.apply_transform(item, transform, **kwargs)

def bind_colormap(item: Union[int, str], source: Union[int, str], **kwargs) -> None:
    if False:
        while True:
            i = 10
    '\t Sets the color map for widgets that accept it.\n\n\tArgs:\n\t\titem (Union[int, str]): item that the color map will be applied to\n\t\tsource (Union[int, str]): The colormap tag. This should come from a colormap that was added to a colormap registry.  Built in color maps are accessible through their corresponding constants mvPlotColormap_Twilight, mvPlotColormap_***\n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.bind_colormap(item, source, **kwargs)

def bind_font(font: Union[int, str], **kwargs) -> Union[int, str]:
    if False:
        while True:
            i = 10
    '\t Binds a global font.\n\n\tArgs:\n\t\tfont (Union[int, str]): \n\tReturns:\n\t\tUnion[int, str]\n\t'
    return internal_dpg.bind_font(font, **kwargs)

def bind_item_font(item: Union[int, str], font: Union[int, str], **kwargs) -> None:
    if False:
        i = 10
        return i + 15
    "\t Sets an item's font.\n\n\tArgs:\n\t\titem (Union[int, str]): \n\t\tfont (Union[int, str]): \n\tReturns:\n\t\tNone\n\t"
    return internal_dpg.bind_item_font(item, font, **kwargs)

def bind_item_handler_registry(item: Union[int, str], handler_registry: Union[int, str], **kwargs) -> None:
    if False:
        i = 10
        return i + 15
    '\t Binds an item handler registry to an item.\n\n\tArgs:\n\t\titem (Union[int, str]): \n\t\thandler_registry (Union[int, str]): \n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.bind_item_handler_registry(item, handler_registry, **kwargs)

def bind_item_theme(item: Union[int, str], theme: Union[int, str], **kwargs) -> None:
    if False:
        print('Hello World!')
    '\t Binds a theme to an item.\n\n\tArgs:\n\t\titem (Union[int, str]): \n\t\ttheme (Union[int, str]): \n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.bind_item_theme(item, theme, **kwargs)

def bind_theme(theme: Union[int, str], **kwargs) -> None:
    if False:
        return 10
    '\t Binds a global theme.\n\n\tArgs:\n\t\ttheme (Union[int, str]): \n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.bind_theme(theme, **kwargs)

def capture_next_item(callback: Callable, *, user_data: Any=None, **kwargs) -> None:
    if False:
        print('Hello World!')
    '\t Captures the next item.\n\n\tArgs:\n\t\tcallback (Callable): \n\t\tuser_data (Any, optional): New in 1.3. Optional user data to send to the callback\n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.capture_next_item(callback, user_data=user_data, **kwargs)

def clear_selected_links(node_editor: Union[int, str], **kwargs) -> None:
    if False:
        print('Hello World!')
    "\t Clears a node editor's selected links.\n\n\tArgs:\n\t\tnode_editor (Union[int, str]): \n\tReturns:\n\t\tNone\n\t"
    return internal_dpg.clear_selected_links(node_editor, **kwargs)

def clear_selected_nodes(node_editor: Union[int, str], **kwargs) -> None:
    if False:
        return 10
    "\t Clears a node editor's selected nodes.\n\n\tArgs:\n\t\tnode_editor (Union[int, str]): \n\tReturns:\n\t\tNone\n\t"
    return internal_dpg.clear_selected_nodes(node_editor, **kwargs)

def create_context(**kwargs) -> None:
    if False:
        while True:
            i = 10
    '\t Creates the Dear PyGui context.\n\n\tArgs:\n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.create_context(**kwargs)

def create_fps_matrix(eye: Union[List[float], Tuple[float, ...]], pitch: float, yaw: float, **kwargs) -> Any:
    if False:
        return 10
    "\t New in 1.1. Create a 'first person shooter' matrix.\n\n\tArgs:\n\t\teye (Union[List[float], Tuple[float, ...]]): eye position\n\t\tpitch (float): pitch (in radians)\n\t\tyaw (float): yaw (in radians)\n\tReturns:\n\t\tAny\n\t"
    return internal_dpg.create_fps_matrix(eye, pitch, yaw, **kwargs)

def create_lookat_matrix(eye: Union[List[float], Tuple[float, ...]], target: Union[List[float], Tuple[float, ...]], up: Union[List[float], Tuple[float, ...]], **kwargs) -> Any:
    if False:
        for i in range(10):
            print('nop')
    "\t New in 1.1. Creates a 'Look at matrix'.\n\n\tArgs:\n\t\teye (Union[List[float], Tuple[float, ...]]): eye position\n\t\ttarget (Union[List[float], Tuple[float, ...]]): target position\n\t\tup (Union[List[float], Tuple[float, ...]]): up vector\n\tReturns:\n\t\tAny\n\t"
    return internal_dpg.create_lookat_matrix(eye, target, up, **kwargs)

def create_orthographic_matrix(left: float, right: float, bottom: float, top: float, zNear: float, zFar: float, **kwargs) -> Any:
    if False:
        return 10
    '\t New in 1.1. Creates an orthographic matrix.\n\n\tArgs:\n\t\tleft (float): left plane\n\t\tright (float): right plane\n\t\tbottom (float): bottom plane\n\t\ttop (float): top plane\n\t\tzNear (float): Near clipping plane.\n\t\tzFar (float): Far clipping plane.\n\tReturns:\n\t\tAny\n\t'
    return internal_dpg.create_orthographic_matrix(left, right, bottom, top, zNear, zFar, **kwargs)

def create_perspective_matrix(fov: float, aspect: float, zNear: float, zFar: float, **kwargs) -> Any:
    if False:
        for i in range(10):
            print('nop')
    '\t New in 1.1. Creates a perspective matrix.\n\n\tArgs:\n\t\tfov (float): Field of view (in radians)\n\t\taspect (float): Aspect ratio (width/height)\n\t\tzNear (float): Near clipping plane.\n\t\tzFar (float): Far clipping plane.\n\tReturns:\n\t\tAny\n\t'
    return internal_dpg.create_perspective_matrix(fov, aspect, zNear, zFar, **kwargs)

def create_rotation_matrix(angle: float, axis: Union[List[float], Tuple[float, ...]], **kwargs) -> Any:
    if False:
        return 10
    '\t New in 1.1. Applies a transformation matrix to a layer.\n\n\tArgs:\n\t\tangle (float): angle to rotate\n\t\taxis (Union[List[float], Tuple[float, ...]]): axis to rotate around\n\tReturns:\n\t\tAny\n\t'
    return internal_dpg.create_rotation_matrix(angle, axis, **kwargs)

def create_scale_matrix(scales: Union[List[float], Tuple[float, ...]], **kwargs) -> Any:
    if False:
        print('Hello World!')
    '\t New in 1.1. Applies a transformation matrix to a layer.\n\n\tArgs:\n\t\tscales (Union[List[float], Tuple[float, ...]]): scale values per axis\n\tReturns:\n\t\tAny\n\t'
    return internal_dpg.create_scale_matrix(scales, **kwargs)

def create_translation_matrix(translation: Union[List[float], Tuple[float, ...]], **kwargs) -> Any:
    if False:
        return 10
    '\t New in 1.1. Creates a translation matrix.\n\n\tArgs:\n\t\ttranslation (Union[List[float], Tuple[float, ...]]): translation vector\n\tReturns:\n\t\tAny\n\t'
    return internal_dpg.create_translation_matrix(translation, **kwargs)

def create_viewport(*, title: str='Dear PyGui', small_icon: str='', large_icon: str='', width: int=1280, height: int=800, x_pos: int=100, y_pos: int=100, min_width: int=250, max_width: int=10000, min_height: int=250, max_height: int=10000, resizable: bool=True, vsync: bool=True, always_on_top: bool=False, decorated: bool=True, clear_color: Union[List[float], Tuple[float, ...]]=(0, 0, 0, 255), disable_close: bool=False, **kwargs) -> None:
    if False:
        i = 10
        return i + 15
    "\t Creates a viewport. Viewports are required.\n\n\tArgs:\n\t\ttitle (str, optional): Sets the title of the viewport.\n\t\tsmall_icon (str, optional): Sets the small icon that is found in the viewport's decorator bar. Must be ***.ico on windows and either ***.ico or ***.png on mac.\n\t\tlarge_icon (str, optional): Sets the large icon that is found in the task bar while the app is running. Must be ***.ico on windows and either ***.ico or ***.png on mac.\n\t\twidth (int, optional): Sets the width of the drawable space on the viewport. Does not inclue the border.\n\t\theight (int, optional): Sets the height of the drawable space on the viewport. Does not inclue the border or decorator bar.\n\t\tx_pos (int, optional): Sets x position the viewport will be drawn in screen coordinates.\n\t\ty_pos (int, optional): Sets y position the viewport will be drawn in screen coordinates.\n\t\tmin_width (int, optional): Applies a minimuim limit to the width of the viewport.\n\t\tmax_width (int, optional): Applies a maximum limit to the width of the viewport.\n\t\tmin_height (int, optional): Applies a minimuim limit to the height of the viewport.\n\t\tmax_height (int, optional): Applies a maximum limit to the height of the viewport.\n\t\tresizable (bool, optional): Enables and Disables user ability to resize the viewport.\n\t\tvsync (bool, optional): Enables and Disables the renderloop vsync limit. vsync frame value is set by refresh rate of display.\n\t\talways_on_top (bool, optional): Forces the viewport to always be drawn ontop of all other viewports.\n\t\tdecorated (bool, optional): Enabled and disabled the decorator bar at the top of the viewport.\n\t\tclear_color (Union[List[float], Tuple[float, ...]], optional): Sets the color of the back of the viewport.\n\t\tdisable_close (bool, optional): Disables the viewport close button. can be used with set_exit_callback\n\tReturns:\n\t\tNone\n\t"
    return internal_dpg.create_viewport(title=title, small_icon=small_icon, large_icon=large_icon, width=width, height=height, x_pos=x_pos, y_pos=y_pos, min_width=min_width, max_width=max_width, min_height=min_height, max_height=max_height, resizable=resizable, vsync=vsync, always_on_top=always_on_top, decorated=decorated, clear_color=clear_color, disable_close=disable_close, **kwargs)

def delete_item(item: Union[int, str], *, children_only: bool=False, slot: int=-1, **kwargs) -> None:
    if False:
        return 10
    '\t Deletes an item..\n\n\tArgs:\n\t\titem (Union[int, str]): \n\t\tchildren_only (bool, optional): \n\t\tslot (int, optional): \n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.delete_item(item, children_only=children_only, slot=slot, **kwargs)

def destroy_context(**kwargs) -> None:
    if False:
        while True:
            i = 10
    '\t Destroys the Dear PyGui context.\n\n\tArgs:\n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.destroy_context(**kwargs)

def does_alias_exist(alias: str, **kwargs) -> bool:
    if False:
        print('Hello World!')
    '\t Checks if an alias exist.\n\n\tArgs:\n\t\talias (str): \n\tReturns:\n\t\tbool\n\t'
    return internal_dpg.does_alias_exist(alias, **kwargs)

def does_item_exist(item: Union[int, str], **kwargs) -> bool:
    if False:
        for i in range(10):
            print('nop')
    '\t Checks if an item exist..\n\n\tArgs:\n\t\titem (Union[int, str]): \n\tReturns:\n\t\tbool\n\t'
    return internal_dpg.does_item_exist(item, **kwargs)

def draw_arrow(p1: Union[List[float], Tuple[float, ...]], p2: Union[List[float], Tuple[float, ...]], *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, before: Union[int, str]=0, show: bool=True, color: Union[List[int], Tuple[int, ...]]=(255, 255, 255, 255), thickness: float=1.0, size: int=4, **kwargs) -> Union[int, str]:
    if False:
        print('Hello World!')
    "\t Adds an arrow.\n\n\tArgs:\n\t\tp1 (Union[List[float], Tuple[float, ...]]): Arrow tip.\n\t\tp2 (Union[List[float], Tuple[float, ...]]): Arrow tail.\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tcolor (Union[List[int], Tuple[int, ...]], optional): \n\t\tthickness (float, optional): \n\t\tsize (int, optional): \n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.draw_arrow(p1, p2, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, before=before, show=show, color=color, thickness=thickness, size=size, **kwargs)

def draw_bezier_cubic(p1: Union[List[float], Tuple[float, ...]], p2: Union[List[float], Tuple[float, ...]], p3: Union[List[float], Tuple[float, ...]], p4: Union[List[float], Tuple[float, ...]], *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, before: Union[int, str]=0, show: bool=True, color: Union[List[int], Tuple[int, ...]]=(255, 255, 255, 255), thickness: float=1.0, segments: int=0, **kwargs) -> Union[int, str]:
    if False:
        return 10
    "\t Adds a cubic bezier curve.\n\n\tArgs:\n\t\tp1 (Union[List[float], Tuple[float, ...]]): First point in curve.\n\t\tp2 (Union[List[float], Tuple[float, ...]]): Second point in curve.\n\t\tp3 (Union[List[float], Tuple[float, ...]]): Third point in curve.\n\t\tp4 (Union[List[float], Tuple[float, ...]]): Fourth point in curve.\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tcolor (Union[List[int], Tuple[int, ...]], optional): \n\t\tthickness (float, optional): \n\t\tsegments (int, optional): Number of segments to approximate bezier curve.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.draw_bezier_cubic(p1, p2, p3, p4, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, before=before, show=show, color=color, thickness=thickness, segments=segments, **kwargs)

def draw_bezier_quadratic(p1: Union[List[float], Tuple[float, ...]], p2: Union[List[float], Tuple[float, ...]], p3: Union[List[float], Tuple[float, ...]], *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, before: Union[int, str]=0, show: bool=True, color: Union[List[int], Tuple[int, ...]]=(255, 255, 255, 255), thickness: float=1.0, segments: int=0, **kwargs) -> Union[int, str]:
    if False:
        for i in range(10):
            print('nop')
    "\t Adds a quadratic bezier curve.\n\n\tArgs:\n\t\tp1 (Union[List[float], Tuple[float, ...]]): First point in curve.\n\t\tp2 (Union[List[float], Tuple[float, ...]]): Second point in curve.\n\t\tp3 (Union[List[float], Tuple[float, ...]]): Third point in curve.\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tcolor (Union[List[int], Tuple[int, ...]], optional): \n\t\tthickness (float, optional): \n\t\tsegments (int, optional): Number of segments to approximate bezier curve.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.draw_bezier_quadratic(p1, p2, p3, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, before=before, show=show, color=color, thickness=thickness, segments=segments, **kwargs)

def draw_circle(center: Union[List[float], Tuple[float, ...]], radius: float, *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, before: Union[int, str]=0, show: bool=True, color: Union[List[int], Tuple[int, ...]]=(255, 255, 255, 255), fill: Union[List[int], Tuple[int, ...]]=(0, 0, 0, -255), thickness: float=1.0, segments: int=0, **kwargs) -> Union[int, str]:
    if False:
        return 10
    "\t Adds a circle\n\n\tArgs:\n\t\tcenter (Union[List[float], Tuple[float, ...]]): \n\t\tradius (float): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tcolor (Union[List[int], Tuple[int, ...]], optional): \n\t\tfill (Union[List[int], Tuple[int, ...]], optional): \n\t\tthickness (float, optional): \n\t\tsegments (int, optional): Number of segments to approximate circle.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.draw_circle(center, radius, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, before=before, show=show, color=color, fill=fill, thickness=thickness, segments=segments, **kwargs)

def draw_ellipse(pmin: Union[List[float], Tuple[float, ...]], pmax: Union[List[float], Tuple[float, ...]], *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, before: Union[int, str]=0, show: bool=True, color: Union[List[int], Tuple[int, ...]]=(255, 255, 255, 255), fill: Union[List[int], Tuple[int, ...]]=(0, 0, 0, -255), thickness: float=1.0, segments: int=32, **kwargs) -> Union[int, str]:
    if False:
        i = 10
        return i + 15
    "\t Adds an ellipse.\n\n\tArgs:\n\t\tpmin (Union[List[float], Tuple[float, ...]]): Min point of bounding rectangle.\n\t\tpmax (Union[List[float], Tuple[float, ...]]): Max point of bounding rectangle.\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tcolor (Union[List[int], Tuple[int, ...]], optional): \n\t\tfill (Union[List[int], Tuple[int, ...]], optional): \n\t\tthickness (float, optional): \n\t\tsegments (int, optional): Number of segments to approximate bezier curve.\n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.draw_ellipse(pmin, pmax, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, before=before, show=show, color=color, fill=fill, thickness=thickness, segments=segments, **kwargs)

def draw_image(texture_tag: Union[int, str], pmin: Union[List[float], Tuple[float, ...]], pmax: Union[List[float], Tuple[float, ...]], *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, before: Union[int, str]=0, show: bool=True, uv_min: Union[List[float], Tuple[float, ...]]=(0.0, 0.0), uv_max: Union[List[float], Tuple[float, ...]]=(1.0, 1.0), color: Union[List[int], Tuple[int, ...]]=(255, 255, 255, 255), **kwargs) -> Union[int, str]:
    if False:
        return 10
    "\t Adds an image (for a drawing).\n\n\tArgs:\n\t\ttexture_tag (Union[int, str]): \n\t\tpmin (Union[List[float], Tuple[float, ...]]): Point of to start drawing texture.\n\t\tpmax (Union[List[float], Tuple[float, ...]]): Point to complete drawing texture.\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tuv_min (Union[List[float], Tuple[float, ...]], optional): Normalized coordinates on texture that will be drawn.\n\t\tuv_max (Union[List[float], Tuple[float, ...]], optional): Normalized coordinates on texture that will be drawn.\n\t\tcolor (Union[List[int], Tuple[int, ...]], optional): \n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.draw_image(texture_tag, pmin, pmax, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, before=before, show=show, uv_min=uv_min, uv_max=uv_max, color=color, **kwargs)

def draw_image_quad(texture_tag: Union[int, str], p1: Union[List[float], Tuple[float, ...]], p2: Union[List[float], Tuple[float, ...]], p3: Union[List[float], Tuple[float, ...]], p4: Union[List[float], Tuple[float, ...]], *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, before: Union[int, str]=0, show: bool=True, uv1: Union[List[float], Tuple[float, ...]]=(0.0, 0.0), uv2: Union[List[float], Tuple[float, ...]]=(1.0, 0.0), uv3: Union[List[float], Tuple[float, ...]]=(1.0, 1.0), uv4: Union[List[float], Tuple[float, ...]]=(0.0, 1.0), color: Union[List[int], Tuple[int, ...]]=(255, 255, 255, 255), **kwargs) -> Union[int, str]:
    if False:
        print('Hello World!')
    "\t Adds an image (for a drawing).\n\n\tArgs:\n\t\ttexture_tag (Union[int, str]): \n\t\tp1 (Union[List[float], Tuple[float, ...]]): \n\t\tp2 (Union[List[float], Tuple[float, ...]]): \n\t\tp3 (Union[List[float], Tuple[float, ...]]): \n\t\tp4 (Union[List[float], Tuple[float, ...]]): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tuv1 (Union[List[float], Tuple[float, ...]], optional): Normalized coordinates on texture that will be drawn.\n\t\tuv2 (Union[List[float], Tuple[float, ...]], optional): Normalized coordinates on texture that will be drawn.\n\t\tuv3 (Union[List[float], Tuple[float, ...]], optional): Normalized coordinates on texture that will be drawn.\n\t\tuv4 (Union[List[float], Tuple[float, ...]], optional): Normalized coordinates on texture that will be drawn.\n\t\tcolor (Union[List[int], Tuple[int, ...]], optional): \n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.draw_image_quad(texture_tag, p1, p2, p3, p4, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, before=before, show=show, uv1=uv1, uv2=uv2, uv3=uv3, uv4=uv4, color=color, **kwargs)

def draw_line(p1: Union[List[float], Tuple[float, ...]], p2: Union[List[float], Tuple[float, ...]], *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, before: Union[int, str]=0, show: bool=True, color: Union[List[int], Tuple[int, ...]]=(255, 255, 255, 255), thickness: float=1.0, **kwargs) -> Union[int, str]:
    if False:
        while True:
            i = 10
    "\t Adds a line.\n\n\tArgs:\n\t\tp1 (Union[List[float], Tuple[float, ...]]): Start of line.\n\t\tp2 (Union[List[float], Tuple[float, ...]]): End of line.\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tcolor (Union[List[int], Tuple[int, ...]], optional): \n\t\tthickness (float, optional): \n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.draw_line(p1, p2, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, before=before, show=show, color=color, thickness=thickness, **kwargs)

def draw_polygon(points: List[List[float]], *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, before: Union[int, str]=0, show: bool=True, color: Union[List[int], Tuple[int, ...]]=(255, 255, 255, 255), fill: Union[List[int], Tuple[int, ...]]=(0, 0, 0, -255), thickness: float=1.0, **kwargs) -> Union[int, str]:
    if False:
        i = 10
        return i + 15
    "\t Adds a polygon.\n\n\tArgs:\n\t\tpoints (List[List[float]]): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tcolor (Union[List[int], Tuple[int, ...]], optional): \n\t\tfill (Union[List[int], Tuple[int, ...]], optional): \n\t\tthickness (float, optional): \n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.draw_polygon(points, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, before=before, show=show, color=color, fill=fill, thickness=thickness, **kwargs)

def draw_polyline(points: List[List[float]], *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, before: Union[int, str]=0, show: bool=True, closed: bool=False, color: Union[List[int], Tuple[int, ...]]=(255, 255, 255, 255), thickness: float=1.0, **kwargs) -> Union[int, str]:
    if False:
        while True:
            i = 10
    "\t Adds a polyline.\n\n\tArgs:\n\t\tpoints (List[List[float]]): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tclosed (bool, optional): Will close the polyline by returning to the first point.\n\t\tcolor (Union[List[int], Tuple[int, ...]], optional): \n\t\tthickness (float, optional): \n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.draw_polyline(points, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, before=before, show=show, closed=closed, color=color, thickness=thickness, **kwargs)

def draw_quad(p1: Union[List[float], Tuple[float, ...]], p2: Union[List[float], Tuple[float, ...]], p3: Union[List[float], Tuple[float, ...]], p4: Union[List[float], Tuple[float, ...]], *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, before: Union[int, str]=0, show: bool=True, color: Union[List[int], Tuple[int, ...]]=(255, 255, 255, 255), fill: Union[List[int], Tuple[int, ...]]=(0, 0, 0, -255), thickness: float=1.0, **kwargs) -> Union[int, str]:
    if False:
        for i in range(10):
            print('nop')
    "\t Adds a quad.\n\n\tArgs:\n\t\tp1 (Union[List[float], Tuple[float, ...]]): \n\t\tp2 (Union[List[float], Tuple[float, ...]]): \n\t\tp3 (Union[List[float], Tuple[float, ...]]): \n\t\tp4 (Union[List[float], Tuple[float, ...]]): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tcolor (Union[List[int], Tuple[int, ...]], optional): \n\t\tfill (Union[List[int], Tuple[int, ...]], optional): \n\t\tthickness (float, optional): \n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.draw_quad(p1, p2, p3, p4, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, before=before, show=show, color=color, fill=fill, thickness=thickness, **kwargs)

def draw_rectangle(pmin: Union[List[float], Tuple[float, ...]], pmax: Union[List[float], Tuple[float, ...]], *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, before: Union[int, str]=0, show: bool=True, color: Union[List[int], Tuple[int, ...]]=(255, 255, 255, 255), color_upper_left: Union[List[int], Tuple[int, ...]]=(255, 255, 255, 255), color_upper_right: Union[List[int], Tuple[int, ...]]=(255, 255, 255, 255), color_bottom_right: Union[List[int], Tuple[int, ...]]=(255, 255, 255, 255), color_bottom_left: Union[List[int], Tuple[int, ...]]=(255, 255, 255, 255), fill: Union[List[int], Tuple[int, ...]]=(0, 0, 0, -255), multicolor: bool=False, rounding: float=0.0, thickness: float=1.0, **kwargs) -> Union[int, str]:
    if False:
        while True:
            i = 10
    "\t Adds a rectangle.\n\n\tArgs:\n\t\tpmin (Union[List[float], Tuple[float, ...]]): Min point of bounding rectangle.\n\t\tpmax (Union[List[float], Tuple[float, ...]]): Max point of bounding rectangle.\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tcolor (Union[List[int], Tuple[int, ...]], optional): \n\t\tcolor_upper_left (Union[List[int], Tuple[int, ...]], optional): 'multicolor' must be set to 'True'\n\t\tcolor_upper_right (Union[List[int], Tuple[int, ...]], optional): 'multicolor' must be set to 'True'\n\t\tcolor_bottom_right (Union[List[int], Tuple[int, ...]], optional): 'multicolor' must be set to 'True'\n\t\tcolor_bottom_left (Union[List[int], Tuple[int, ...]], optional): 'multicolor' must be set to 'True'\n\t\tfill (Union[List[int], Tuple[int, ...]], optional): \n\t\tmulticolor (bool, optional): \n\t\trounding (float, optional): Number of pixels of the radius that will round the corners of the rectangle. Note: doesn't work with multicolor\n\t\tthickness (float, optional): \n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.draw_rectangle(pmin, pmax, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, before=before, show=show, color=color, color_upper_left=color_upper_left, color_upper_right=color_upper_right, color_bottom_right=color_bottom_right, color_bottom_left=color_bottom_left, fill=fill, multicolor=multicolor, rounding=rounding, thickness=thickness, **kwargs)

def draw_text(pos: Union[List[float], Tuple[float, ...]], text: str, *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, before: Union[int, str]=0, show: bool=True, color: Union[List[int], Tuple[int, ...]]=(255, 255, 255, 255), size: float=10.0, **kwargs) -> Union[int, str]:
    if False:
        for i in range(10):
            print('nop')
    "\t Adds text (drawlist).\n\n\tArgs:\n\t\tpos (Union[List[float], Tuple[float, ...]]): Top left point of bounding text rectangle.\n\t\ttext (str): Text to draw.\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tcolor (Union[List[int], Tuple[int, ...]], optional): \n\t\tsize (float, optional): \n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.draw_text(pos, text, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, before=before, show=show, color=color, size=size, **kwargs)

def draw_triangle(p1: Union[List[float], Tuple[float, ...]], p2: Union[List[float], Tuple[float, ...]], p3: Union[List[float], Tuple[float, ...]], *, label: str=None, user_data: Any=None, use_internal_label: bool=True, tag: Union[int, str]=0, parent: Union[int, str]=0, before: Union[int, str]=0, show: bool=True, color: Union[List[int], Tuple[int, ...]]=(255, 255, 255, 255), fill: Union[List[int], Tuple[int, ...]]=(0, 0, 0, -255), thickness: float=1.0, **kwargs) -> Union[int, str]:
    if False:
        while True:
            i = 10
    "\t Adds a triangle.\n\n\tArgs:\n\t\tp1 (Union[List[float], Tuple[float, ...]]): \n\t\tp2 (Union[List[float], Tuple[float, ...]]): \n\t\tp3 (Union[List[float], Tuple[float, ...]]): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tcolor (Union[List[int], Tuple[int, ...]], optional): \n\t\tfill (Union[List[int], Tuple[int, ...]], optional): \n\t\tthickness (float, optional): \n\t\tid (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'id' in kwargs.keys():
        warnings.warn('id keyword renamed to tag', DeprecationWarning, 2)
        tag = kwargs['id']
    return internal_dpg.draw_triangle(p1, p2, p3, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, parent=parent, before=before, show=show, color=color, fill=fill, thickness=thickness, **kwargs)

def empty_container_stack(**kwargs) -> None:
    if False:
        while True:
            i = 10
    '\t Emptyes the container stack.\n\n\tArgs:\n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.empty_container_stack(**kwargs)

def fit_axis_data(axis: Union[int, str], **kwargs) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\t Sets the axis boundaries max/min in the data series currently on the plot.\n\n\tArgs:\n\t\taxis (Union[int, str]): \n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.fit_axis_data(axis, **kwargs)

def focus_item(item: Union[int, str], **kwargs) -> None:
    if False:
        while True:
            i = 10
    '\t Focuses an item.\n\n\tArgs:\n\t\titem (Union[int, str]): \n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.focus_item(item, **kwargs)

def generate_uuid(**kwargs) -> Union[int, str]:
    if False:
        while True:
            i = 10
    '\t Generate a new UUID.\n\n\tArgs:\n\tReturns:\n\t\tUnion[int, str]\n\t'
    return internal_dpg.generate_uuid(**kwargs)

def get_active_window(**kwargs) -> Union[int, str]:
    if False:
        while True:
            i = 10
    '\t Returns the active window.\n\n\tArgs:\n\tReturns:\n\t\tUnion[int, str]\n\t'
    return internal_dpg.get_active_window(**kwargs)

def get_alias_id(alias: str, **kwargs) -> Union[int, str]:
    if False:
        for i in range(10):
            print('nop')
    '\t Returns the ID associated with an alias.\n\n\tArgs:\n\t\talias (str): \n\tReturns:\n\t\tUnion[int, str]\n\t'
    return internal_dpg.get_alias_id(alias, **kwargs)

def get_aliases(**kwargs) -> Union[List[str], Tuple[str, ...]]:
    if False:
        while True:
            i = 10
    '\t Returns all aliases.\n\n\tArgs:\n\tReturns:\n\t\tUnion[List[str], Tuple[str, ...]]\n\t'
    return internal_dpg.get_aliases(**kwargs)

def get_all_items(**kwargs) -> Union[List[int], Tuple[int, ...]]:
    if False:
        return 10
    '\t Returns all items.\n\n\tArgs:\n\tReturns:\n\t\tUnion[List[int], Tuple[int, ...]]\n\t'
    return internal_dpg.get_all_items(**kwargs)

def get_app_configuration(**kwargs) -> dict:
    if False:
        for i in range(10):
            print('nop')
    '\t Returns app configuration.\n\n\tArgs:\n\tReturns:\n\t\tdict\n\t'
    return internal_dpg.get_app_configuration(**kwargs)

def get_axis_limits(axis: Union[int, str], **kwargs) -> Union[List[float], Tuple[float, ...]]:
    if False:
        i = 10
        return i + 15
    '\t Get the specified axis limits.\n\n\tArgs:\n\t\taxis (Union[int, str]): \n\tReturns:\n\t\tUnion[List[float], Tuple[float, ...]]\n\t'
    return internal_dpg.get_axis_limits(axis, **kwargs)

def get_callback_queue(**kwargs) -> Any:
    if False:
        return 10
    '\t New in 1.2. Returns and clears callback queue.\n\n\tArgs:\n\tReturns:\n\t\tAny\n\t'
    return internal_dpg.get_callback_queue(**kwargs)

def get_clipboard_text(**kwargs) -> str:
    if False:
        print('Hello World!')
    '\t New in 1.3. Gets the clipboard text.\n\n\tArgs:\n\tReturns:\n\t\tstr\n\t'
    return internal_dpg.get_clipboard_text(**kwargs)

def get_colormap_color(colormap: Union[int, str], index: int, **kwargs) -> Union[List[int], Tuple[int, ...]]:
    if False:
        print('Hello World!')
    '\t Returns a color from a colormap given an index >= 0. (ex. 0 will be the first color in the color list of the color map) Modulo will be performed against the number of items in the color list.\n\n\tArgs:\n\t\tcolormap (Union[int, str]): The colormap tag. This should come from a colormap that was added to a colormap registry. Built in color maps are accessible through their corresponding constants mvPlotColormap_Twilight, mvPlotColormap_***\n\t\tindex (int): Desired position of the color in the colors list value of the colormap being quiered \n\tReturns:\n\t\tUnion[List[int], Tuple[int, ...]]\n\t'
    return internal_dpg.get_colormap_color(colormap, index, **kwargs)

def get_delta_time(**kwargs) -> float:
    if False:
        print('Hello World!')
    '\t Returns time since last frame.\n\n\tArgs:\n\tReturns:\n\t\tfloat\n\t'
    return internal_dpg.get_delta_time(**kwargs)

def get_drawing_mouse_pos(**kwargs) -> Union[List[int], Tuple[int, ...]]:
    if False:
        i = 10
        return i + 15
    '\t Returns mouse position in drawing.\n\n\tArgs:\n\tReturns:\n\t\tUnion[List[int], Tuple[int, ...]]\n\t'
    return internal_dpg.get_drawing_mouse_pos(**kwargs)

def get_file_dialog_info(file_dialog: Union[int, str], **kwargs) -> dict:
    if False:
        for i in range(10):
            print('nop')
    '\t Returns information related to the file dialog. Typically used while the file dialog is in use to query data about the state or info related to the file dialog.\n\n\tArgs:\n\t\tfile_dialog (Union[int, str]): \n\tReturns:\n\t\tdict\n\t'
    return internal_dpg.get_file_dialog_info(file_dialog, **kwargs)

def get_frame_count(**kwargs) -> int:
    if False:
        i = 10
        return i + 15
    '\t Returns frame count.\n\n\tArgs:\n\tReturns:\n\t\tint\n\t'
    return internal_dpg.get_frame_count(**kwargs)

def get_frame_rate(**kwargs) -> float:
    if False:
        return 10
    '\t Returns the average frame rate across 120 frames.\n\n\tArgs:\n\tReturns:\n\t\tfloat\n\t'
    return internal_dpg.get_frame_rate(**kwargs)

def get_global_font_scale(**kwargs) -> float:
    if False:
        while True:
            i = 10
    '\t Returns global font scale.\n\n\tArgs:\n\tReturns:\n\t\tfloat\n\t'
    return internal_dpg.get_global_font_scale(**kwargs)

def get_item_alias(item: Union[int, str], **kwargs) -> str:
    if False:
        print('Hello World!')
    "\t Returns an item's alias.\n\n\tArgs:\n\t\titem (Union[int, str]): \n\tReturns:\n\t\tstr\n\t"
    return internal_dpg.get_item_alias(item, **kwargs)

def get_item_configuration(item: Union[int, str], **kwargs) -> dict:
    if False:
        return 10
    "\t Returns an item's configuration.\n\n\tArgs:\n\t\titem (Union[int, str]): \n\tReturns:\n\t\tdict\n\t"
    return internal_dpg.get_item_configuration(item, **kwargs)

def get_item_info(item: Union[int, str], **kwargs) -> dict:
    if False:
        for i in range(10):
            print('nop')
    "\t Returns an item's information.\n\n\tArgs:\n\t\titem (Union[int, str]): \n\tReturns:\n\t\tdict\n\t"
    return internal_dpg.get_item_info(item, **kwargs)

def get_item_state(item: Union[int, str], **kwargs) -> dict:
    if False:
        i = 10
        return i + 15
    "\t Returns an item's state.\n\n\tArgs:\n\t\titem (Union[int, str]): \n\tReturns:\n\t\tdict\n\t"
    return internal_dpg.get_item_state(item, **kwargs)

def get_item_types(**kwargs) -> dict:
    if False:
        i = 10
        return i + 15
    '\t Returns an item types.\n\n\tArgs:\n\tReturns:\n\t\tdict\n\t'
    return internal_dpg.get_item_types(**kwargs)

def get_mouse_drag_delta(**kwargs) -> float:
    if False:
        while True:
            i = 10
    '\t Returns mouse drag delta.\n\n\tArgs:\n\tReturns:\n\t\tfloat\n\t'
    return internal_dpg.get_mouse_drag_delta(**kwargs)

def get_mouse_pos(*, local: bool=True, **kwargs) -> Union[List[int], Tuple[int, ...]]:
    if False:
        print('Hello World!')
    '\t Returns mouse position.\n\n\tArgs:\n\t\tlocal (bool, optional): \n\tReturns:\n\t\tUnion[List[int], Tuple[int, ...]]\n\t'
    return internal_dpg.get_mouse_pos(local=local, **kwargs)

def get_platform(**kwargs) -> int:
    if False:
        return 10
    '\t New in 1.6. Returns platform constant.\n\n\tArgs:\n\tReturns:\n\t\tint\n\t'
    return internal_dpg.get_platform(**kwargs)

def get_plot_mouse_pos(**kwargs) -> Union[List[int], Tuple[int, ...]]:
    if False:
        return 10
    '\t Returns mouse position in plot.\n\n\tArgs:\n\tReturns:\n\t\tUnion[List[int], Tuple[int, ...]]\n\t'
    return internal_dpg.get_plot_mouse_pos(**kwargs)

def get_plot_query_area(plot: Union[int, str], **kwargs) -> Union[List[float], Tuple[float, ...]]:
    if False:
        return 10
    "\t Returns the last/current query area of the plot. (Requires plot 'query' kwarg to be enabled)\n\n\tArgs:\n\t\tplot (Union[int, str]): \n\tReturns:\n\t\tUnion[List[float], Tuple[float, ...]]\n\t"
    return internal_dpg.get_plot_query_area(plot, **kwargs)

def get_selected_links(node_editor: Union[int, str], **kwargs) -> List[List[str]]:
    if False:
        return 10
    "\t Returns a node editor's selected links.\n\n\tArgs:\n\t\tnode_editor (Union[int, str]): \n\tReturns:\n\t\tList[List[str]]\n\t"
    return internal_dpg.get_selected_links(node_editor, **kwargs)

def get_selected_nodes(node_editor: Union[int, str], **kwargs) -> Union[List[int], Tuple[int, ...]]:
    if False:
        return 10
    "\t Returns a node editor's selected nodes.\n\n\tArgs:\n\t\tnode_editor (Union[int, str]): \n\tReturns:\n\t\tUnion[List[int], Tuple[int, ...]]\n\t"
    return internal_dpg.get_selected_nodes(node_editor, **kwargs)

def get_text_size(text: str, *, wrap_width: float=-1.0, font: Union[int, str]=0, **kwargs) -> Union[List[float], Tuple[float, ...]]:
    if False:
        i = 10
        return i + 15
    '\t Returns width/height of text with specified font (must occur after 1st frame).\n\n\tArgs:\n\t\ttext (str): \n\t\twrap_width (float, optional): Wrap width to use (-1.0 turns wrap off).\n\t\tfont (Union[int, str], optional): Font to use.\n\tReturns:\n\t\tUnion[List[float], Tuple[float, ...]]\n\t'
    return internal_dpg.get_text_size(text, wrap_width=wrap_width, font=font, **kwargs)

def get_total_time(**kwargs) -> float:
    if False:
        for i in range(10):
            print('nop')
    '\t Returns total time since Dear PyGui has started.\n\n\tArgs:\n\tReturns:\n\t\tfloat\n\t'
    return internal_dpg.get_total_time(**kwargs)

def get_value(item: Union[int, str], **kwargs) -> Any:
    if False:
        i = 10
        return i + 15
    "\t Returns an item's value.\n\n\tArgs:\n\t\titem (Union[int, str]): \n\tReturns:\n\t\tAny\n\t"
    return internal_dpg.get_value(item, **kwargs)

def get_values(items: Union[List[int], Tuple[int, ...]], **kwargs) -> Any:
    if False:
        i = 10
        return i + 15
    '\t Returns values of a list of items.\n\n\tArgs:\n\t\titems (Union[List[int], Tuple[int, ...]]): \n\tReturns:\n\t\tAny\n\t'
    return internal_dpg.get_values(items, **kwargs)

def get_viewport_configuration(item: Union[int, str], **kwargs) -> dict:
    if False:
        print('Hello World!')
    "\t Returns a viewport's configuration.\n\n\tArgs:\n\t\titem (Union[int, str]): \n\tReturns:\n\t\tdict\n\t"
    return internal_dpg.get_viewport_configuration(item, **kwargs)

def get_windows(**kwargs) -> Union[List[int], Tuple[int, ...]]:
    if False:
        for i in range(10):
            print('nop')
    '\t Returns all windows.\n\n\tArgs:\n\tReturns:\n\t\tUnion[List[int], Tuple[int, ...]]\n\t'
    return internal_dpg.get_windows(**kwargs)

def get_x_scroll(item: Union[int, str], **kwargs) -> float:
    if False:
        print('Hello World!')
    '\t Undocumented\n\n\tArgs:\n\t\titem (Union[int, str]): \n\tReturns:\n\t\tfloat\n\t'
    return internal_dpg.get_x_scroll(item, **kwargs)

def get_x_scroll_max(item: Union[int, str], **kwargs) -> float:
    if False:
        i = 10
        return i + 15
    '\t Undocumented\n\n\tArgs:\n\t\titem (Union[int, str]): \n\tReturns:\n\t\tfloat\n\t'
    return internal_dpg.get_x_scroll_max(item, **kwargs)

def get_y_scroll(item: Union[int, str], **kwargs) -> float:
    if False:
        return 10
    '\t Undocumented\n\n\tArgs:\n\t\titem (Union[int, str]): \n\tReturns:\n\t\tfloat\n\t'
    return internal_dpg.get_y_scroll(item, **kwargs)

def get_y_scroll_max(item: Union[int, str], **kwargs) -> float:
    if False:
        i = 10
        return i + 15
    '\t Undocumented\n\n\tArgs:\n\t\titem (Union[int, str]): \n\tReturns:\n\t\tfloat\n\t'
    return internal_dpg.get_y_scroll_max(item, **kwargs)

def highlight_table_cell(table: Union[int, str], row: int, column: int, color: Union[List[int], Tuple[int, ...]], **kwargs) -> None:
    if False:
        while True:
            i = 10
    '\t Highlight specified table cell.\n\n\tArgs:\n\t\ttable (Union[int, str]): \n\t\trow (int): \n\t\tcolumn (int): \n\t\tcolor (Union[List[int], Tuple[int, ...]]): \n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.highlight_table_cell(table, row, column, color, **kwargs)

def highlight_table_column(table: Union[int, str], column: int, color: Union[List[int], Tuple[int, ...]], **kwargs) -> None:
    if False:
        print('Hello World!')
    '\t Highlight specified table column.\n\n\tArgs:\n\t\ttable (Union[int, str]): \n\t\tcolumn (int): \n\t\tcolor (Union[List[int], Tuple[int, ...]]): \n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.highlight_table_column(table, column, color, **kwargs)

def highlight_table_row(table: Union[int, str], row: int, color: Union[List[int], Tuple[int, ...]], **kwargs) -> None:
    if False:
        i = 10
        return i + 15
    '\t Highlight specified table row.\n\n\tArgs:\n\t\ttable (Union[int, str]): \n\t\trow (int): \n\t\tcolor (Union[List[int], Tuple[int, ...]]): \n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.highlight_table_row(table, row, color, **kwargs)

def is_dearpygui_running(**kwargs) -> bool:
    if False:
        i = 10
        return i + 15
    '\t Checks if Dear PyGui is running\n\n\tArgs:\n\tReturns:\n\t\tbool\n\t'
    return internal_dpg.is_dearpygui_running(**kwargs)

def is_key_down(key: int, **kwargs) -> bool:
    if False:
        print('Hello World!')
    '\t Checks if key is down.\n\n\tArgs:\n\t\tkey (int): \n\tReturns:\n\t\tbool\n\t'
    return internal_dpg.is_key_down(key, **kwargs)

def is_key_pressed(key: int, **kwargs) -> bool:
    if False:
        i = 10
        return i + 15
    '\t Checks if key is pressed.\n\n\tArgs:\n\t\tkey (int): \n\tReturns:\n\t\tbool\n\t'
    return internal_dpg.is_key_pressed(key, **kwargs)

def is_key_released(key: int, **kwargs) -> bool:
    if False:
        i = 10
        return i + 15
    '\t Checks if key is released.\n\n\tArgs:\n\t\tkey (int): \n\tReturns:\n\t\tbool\n\t'
    return internal_dpg.is_key_released(key, **kwargs)

def is_mouse_button_clicked(button: int, **kwargs) -> bool:
    if False:
        i = 10
        return i + 15
    '\t Checks if mouse button is clicked.\n\n\tArgs:\n\t\tbutton (int): \n\tReturns:\n\t\tbool\n\t'
    return internal_dpg.is_mouse_button_clicked(button, **kwargs)

def is_mouse_button_double_clicked(button: int, **kwargs) -> bool:
    if False:
        i = 10
        return i + 15
    '\t Checks if mouse button is double clicked.\n\n\tArgs:\n\t\tbutton (int): \n\tReturns:\n\t\tbool\n\t'
    return internal_dpg.is_mouse_button_double_clicked(button, **kwargs)

def is_mouse_button_down(button: int, **kwargs) -> bool:
    if False:
        for i in range(10):
            print('nop')
    '\t Checks if mouse button is down.\n\n\tArgs:\n\t\tbutton (int): \n\tReturns:\n\t\tbool\n\t'
    return internal_dpg.is_mouse_button_down(button, **kwargs)

def is_mouse_button_dragging(button: int, threshold: float, **kwargs) -> bool:
    if False:
        return 10
    '\t Checks if mouse button is down and dragging.\n\n\tArgs:\n\t\tbutton (int): \n\t\tthreshold (float): \n\tReturns:\n\t\tbool\n\t'
    return internal_dpg.is_mouse_button_dragging(button, threshold, **kwargs)

def is_mouse_button_released(button: int, **kwargs) -> bool:
    if False:
        for i in range(10):
            print('nop')
    '\t Checks if mouse button is released.\n\n\tArgs:\n\t\tbutton (int): \n\tReturns:\n\t\tbool\n\t'
    return internal_dpg.is_mouse_button_released(button, **kwargs)

def is_plot_queried(plot: Union[int, str], **kwargs) -> bool:
    if False:
        i = 10
        return i + 15
    "\t Returns true if the plot is currently being queried. (Requires plot 'query' kwarg to be enabled)\n\n\tArgs:\n\t\tplot (Union[int, str]): \n\tReturns:\n\t\tbool\n\t"
    return internal_dpg.is_plot_queried(plot, **kwargs)

def is_table_cell_highlighted(table: Union[int, str], row: int, column: int, **kwargs) -> bool:
    if False:
        i = 10
        return i + 15
    '\t Checks if a table cell is highlighted.\n\n\tArgs:\n\t\ttable (Union[int, str]): \n\t\trow (int): \n\t\tcolumn (int): \n\tReturns:\n\t\tbool\n\t'
    return internal_dpg.is_table_cell_highlighted(table, row, column, **kwargs)

def is_table_column_highlighted(table: Union[int, str], column: int, **kwargs) -> bool:
    if False:
        for i in range(10):
            print('nop')
    '\t Checks if a table column is highlighted.\n\n\tArgs:\n\t\ttable (Union[int, str]): \n\t\tcolumn (int): \n\tReturns:\n\t\tbool\n\t'
    return internal_dpg.is_table_column_highlighted(table, column, **kwargs)

def is_table_row_highlighted(table: Union[int, str], row: int, **kwargs) -> bool:
    if False:
        print('Hello World!')
    '\t Checks if a table row is highlighted.\n\n\tArgs:\n\t\ttable (Union[int, str]): \n\t\trow (int): \n\tReturns:\n\t\tbool\n\t'
    return internal_dpg.is_table_row_highlighted(table, row, **kwargs)

def is_viewport_ok(**kwargs) -> bool:
    if False:
        i = 10
        return i + 15
    '\t Checks if a viewport has been created and shown.\n\n\tArgs:\n\tReturns:\n\t\tbool\n\t'
    return internal_dpg.is_viewport_ok(**kwargs)

def last_container(**kwargs) -> Union[int, str]:
    if False:
        print('Hello World!')
    '\t Returns the last container item added.\n\n\tArgs:\n\tReturns:\n\t\tUnion[int, str]\n\t'
    return internal_dpg.last_container(**kwargs)

def last_item(**kwargs) -> Union[int, str]:
    if False:
        return 10
    '\t Returns the last item added.\n\n\tArgs:\n\tReturns:\n\t\tUnion[int, str]\n\t'
    return internal_dpg.last_item(**kwargs)

def last_root(**kwargs) -> Union[int, str]:
    if False:
        for i in range(10):
            print('nop')
    '\t Returns the last root added (registry or window).\n\n\tArgs:\n\tReturns:\n\t\tUnion[int, str]\n\t'
    return internal_dpg.last_root(**kwargs)

def load_image(file: str, *, gamma: float=1.0, gamma_scale_factor: float=1.0, **kwargs) -> Any:
    if False:
        for i in range(10):
            print('nop')
    '\t Loads an image. Returns width, height, channels, mvBuffer\n\n\tArgs:\n\t\tfile (str): \n\t\tgamma (float, optional): Gamma correction factor. (default is 1.0 to avoid automatic gamma correction on loading.\n\t\tgamma_scale_factor (float, optional): Gamma scale factor.\n\tReturns:\n\t\tAny\n\t'
    return internal_dpg.load_image(file, gamma=gamma, gamma_scale_factor=gamma_scale_factor, **kwargs)

def lock_mutex(**kwargs) -> None:
    if False:
        print('Hello World!')
    '\t Locks render thread mutex.\n\n\tArgs:\n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.lock_mutex(**kwargs)

def maximize_viewport(**kwargs) -> None:
    if False:
        return 10
    '\t Maximizes the viewport.\n\n\tArgs:\n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.maximize_viewport(**kwargs)

def minimize_viewport(**kwargs) -> None:
    if False:
        print('Hello World!')
    '\t Minimizes a viewport.\n\n\tArgs:\n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.minimize_viewport(**kwargs)

def move_item(item: Union[int, str], *, parent: Union[int, str]=0, before: Union[int, str]=0, **kwargs) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\t Moves an item to a new location.\n\n\tArgs:\n\t\titem (Union[int, str]): \n\t\tparent (Union[int, str], optional): \n\t\tbefore (Union[int, str], optional): \n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.move_item(item, parent=parent, before=before, **kwargs)

def move_item_down(item: Union[int, str], **kwargs) -> None:
    if False:
        return 10
    '\t Moves an item down.\n\n\tArgs:\n\t\titem (Union[int, str]): \n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.move_item_down(item, **kwargs)

def move_item_up(item: Union[int, str], **kwargs) -> None:
    if False:
        while True:
            i = 10
    '\t Moves an item up.\n\n\tArgs:\n\t\titem (Union[int, str]): \n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.move_item_up(item, **kwargs)

def output_frame_buffer(file: str='', *, callback: Callable=None, **kwargs) -> Any:
    if False:
        print('Hello World!')
    '\t Outputs frame buffer as a png if file is specified or through the second argument of a callback if specified. Render loop must have been started.\n\n\tArgs:\n\t\tfile (str, optional): \n\t\tcallback (Callable, optional): Callback will return framebuffer as an array through the second arg.\n\tReturns:\n\t\tAny\n\t'
    return internal_dpg.output_frame_buffer(file, callback=callback, **kwargs)

def pop_container_stack(**kwargs) -> Union[int, str]:
    if False:
        print('Hello World!')
    '\t Pops the top item off the parent stack and return its ID.\n\n\tArgs:\n\tReturns:\n\t\tUnion[int, str]\n\t'
    return internal_dpg.pop_container_stack(**kwargs)

def push_container_stack(item: Union[int, str], **kwargs) -> bool:
    if False:
        i = 10
        return i + 15
    '\t Pushes an item onto the container stack.\n\n\tArgs:\n\t\titem (Union[int, str]): \n\tReturns:\n\t\tbool\n\t'
    return internal_dpg.push_container_stack(item, **kwargs)

def remove_alias(alias: str, **kwargs) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\t Removes an alias.\n\n\tArgs:\n\t\talias (str): \n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.remove_alias(alias, **kwargs)

def render_dearpygui_frame(**kwargs) -> None:
    if False:
        while True:
            i = 10
    '\t Render a single Dear PyGui frame.\n\n\tArgs:\n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.render_dearpygui_frame(**kwargs)

def reorder_items(container: Union[int, str], slot: int, new_order: Union[List[int], Tuple[int, ...]], **kwargs) -> None:
    if False:
        i = 10
        return i + 15
    "\t Reorders an item's children.\n\n\tArgs:\n\t\tcontainer (Union[int, str]): \n\t\tslot (int): \n\t\tnew_order (Union[List[int], Tuple[int, ...]]): \n\tReturns:\n\t\tNone\n\t"
    return internal_dpg.reorder_items(container, slot, new_order, **kwargs)

def reset_axis_ticks(axis: Union[int, str], **kwargs) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\t Removes the manually set axis ticks and applies the default axis ticks\n\n\tArgs:\n\t\taxis (Union[int, str]): \n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.reset_axis_ticks(axis, **kwargs)

def reset_pos(item: Union[int, str], **kwargs) -> None:
    if False:
        while True:
            i = 10
    "\t Resets an item's position after using 'set_item_pos'.\n\n\tArgs:\n\t\titem (Union[int, str]): \n\tReturns:\n\t\tNone\n\t"
    return internal_dpg.reset_pos(item, **kwargs)

def sample_colormap(colormap: Union[int, str], t: float, **kwargs) -> Union[List[int], Tuple[int, ...]]:
    if False:
        while True:
            i = 10
    '\t Returns a color from a colormap given t between 0.0-1.0.\n\n\tArgs:\n\t\tcolormap (Union[int, str]): The colormap tag. This should come from a colormap that was added to a colormap registry. Built in color maps are accessible through their corresponding constants mvPlotColormap_Twilight, mvPlotColormap_***\n\t\tt (float): Value of the colormap to sample between 0.0-1.0\n\tReturns:\n\t\tUnion[List[int], Tuple[int, ...]]\n\t'
    return internal_dpg.sample_colormap(colormap, t, **kwargs)

def save_image(file: str, width: int, height: int, data: Any, *, components: int=4, quality: int=50, **kwargs) -> None:
    if False:
        while True:
            i = 10
    '\t Saves an image. Possible formats: png, bmp, tga, hdr, jpg.\n\n\tArgs:\n\t\tfile (str): \n\t\twidth (int): \n\t\theight (int): \n\t\tdata (Any): \n\t\tcomponents (int, optional): Number of components (1-4). Default of 4.\n\t\tquality (int, optional): Stride in bytes (only used for jpg).\n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.save_image(file, width, height, data, components=components, quality=quality, **kwargs)

def save_init_file(file: str, **kwargs) -> None:
    if False:
        print('Hello World!')
    '\t Save dpg.ini file.\n\n\tArgs:\n\t\tfile (str): \n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.save_init_file(file, **kwargs)

def set_axis_limits(axis: Union[int, str], ymin: float, ymax: float, **kwargs) -> None:
    if False:
        while True:
            i = 10
    '\t Sets limits on the axis for pan and zoom.\n\n\tArgs:\n\t\taxis (Union[int, str]): \n\t\tymin (float): \n\t\tymax (float): \n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.set_axis_limits(axis, ymin, ymax, **kwargs)

def set_axis_limits_auto(axis: Union[int, str], **kwargs) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\t Removes all limits on specified axis.\n\n\tArgs:\n\t\taxis (Union[int, str]): \n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.set_axis_limits_auto(axis, **kwargs)

def set_axis_ticks(axis: Union[int, str], label_pairs: Any, **kwargs) -> None:
    if False:
        while True:
            i = 10
    "\t Replaces axis ticks with 'label_pairs' argument.\n\n\tArgs:\n\t\taxis (Union[int, str]): \n\t\tlabel_pairs (Any): Tuples of label and value in the form '((label, axis_value), (label, axis_value), ...)'\n\tReturns:\n\t\tNone\n\t"
    return internal_dpg.set_axis_ticks(axis, label_pairs, **kwargs)

def set_clip_space(item: Union[int, str], top_left_x: float, top_left_y: float, width: float, height: float, min_depth: float, max_depth: float, **kwargs) -> None:
    if False:
        print('Hello World!')
    "\t New in 1.1. Set the clip space for depth clipping and 'viewport' transformation.\n\n\tArgs:\n\t\titem (Union[int, str]): draw layer to set clip space\n\t\ttop_left_x (float): angle to rotate\n\t\ttop_left_y (float): angle to rotate\n\t\twidth (float): angle to rotate\n\t\theight (float): angle to rotate\n\t\tmin_depth (float): angle to rotate\n\t\tmax_depth (float): angle to rotate\n\tReturns:\n\t\tNone\n\t"
    return internal_dpg.set_clip_space(item, top_left_x, top_left_y, width, height, min_depth, max_depth, **kwargs)

def set_clipboard_text(text: str, **kwargs) -> None:
    if False:
        return 10
    '\t New in 1.3. Sets the clipboard text.\n\n\tArgs:\n\t\ttext (str): \n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.set_clipboard_text(text, **kwargs)

def set_exit_callback(callback: Callable, *, user_data: Any=None, **kwargs) -> str:
    if False:
        i = 10
        return i + 15
    '\t Sets a callback to run on last frame.\n\n\tArgs:\n\t\tcallback (Callable): \n\t\tuser_data (Any, optional): New in 1.3. Optional user data to send to the callback\n\tReturns:\n\t\tstr\n\t'
    return internal_dpg.set_exit_callback(callback, user_data=user_data, **kwargs)

def set_frame_callback(frame: int, callback: Callable, *, user_data: Any=None, **kwargs) -> str:
    if False:
        i = 10
        return i + 15
    '\t Sets a callback to run on first frame.\n\n\tArgs:\n\t\tframe (int): \n\t\tcallback (Callable): \n\t\tuser_data (Any, optional): New in 1.3. Optional user data to send to the callback\n\tReturns:\n\t\tstr\n\t'
    return internal_dpg.set_frame_callback(frame, callback, user_data=user_data, **kwargs)

def set_global_font_scale(scale: float, **kwargs) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\t Sets global font scale.\n\n\tArgs:\n\t\tscale (float): \n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.set_global_font_scale(scale, **kwargs)

def set_item_alias(item: Union[int, str], alias: str, **kwargs) -> None:
    if False:
        for i in range(10):
            print('nop')
    "\t Sets an item's alias.\n\n\tArgs:\n\t\titem (Union[int, str]): \n\t\talias (str): \n\tReturns:\n\t\tNone\n\t"
    return internal_dpg.set_item_alias(item, alias, **kwargs)

def set_item_children(item: Union[int, str], source: Union[int, str], slot: int, **kwargs) -> None:
    if False:
        return 10
    "\t Sets an item's children.\n\n\tArgs:\n\t\titem (Union[int, str]): \n\t\tsource (Union[int, str]): \n\t\tslot (int): \n\tReturns:\n\t\tNone\n\t"
    return internal_dpg.set_item_children(item, source, slot, **kwargs)

def set_primary_window(window: Union[int, str], value: bool, **kwargs) -> None:
    if False:
        i = 10
        return i + 15
    '\t Sets the primary window.\n\n\tArgs:\n\t\twindow (Union[int, str]): \n\t\tvalue (bool): \n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.set_primary_window(window, value, **kwargs)

def set_table_row_color(table: Union[int, str], row: int, color: Union[List[int], Tuple[int, ...]], **kwargs) -> None:
    if False:
        while True:
            i = 10
    '\t Set table row color.\n\n\tArgs:\n\t\ttable (Union[int, str]): \n\t\trow (int): \n\t\tcolor (Union[List[int], Tuple[int, ...]]): \n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.set_table_row_color(table, row, color, **kwargs)

def set_value(item: Union[int, str], value: Any, **kwargs) -> None:
    if False:
        i = 10
        return i + 15
    "\t Set's an item's value.\n\n\tArgs:\n\t\titem (Union[int, str]): \n\t\tvalue (Any): \n\tReturns:\n\t\tNone\n\t"
    return internal_dpg.set_value(item, value, **kwargs)

def set_viewport_resize_callback(callback: Callable, *, user_data: Any=None, **kwargs) -> str:
    if False:
        print('Hello World!')
    '\t Sets a callback to run on viewport resize.\n\n\tArgs:\n\t\tcallback (Callable): \n\t\tuser_data (Any, optional): New in 1.3. Optional user data to send to the callback\n\tReturns:\n\t\tstr\n\t'
    return internal_dpg.set_viewport_resize_callback(callback, user_data=user_data, **kwargs)

def set_x_scroll(item: Union[int, str], value: float, **kwargs) -> None:
    if False:
        while True:
            i = 10
    '\t Undocumented\n\n\tArgs:\n\t\titem (Union[int, str]): \n\t\tvalue (float): \n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.set_x_scroll(item, value, **kwargs)

def set_y_scroll(item: Union[int, str], value: float, **kwargs) -> None:
    if False:
        return 10
    '\t Undocumented\n\n\tArgs:\n\t\titem (Union[int, str]): \n\t\tvalue (float): \n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.set_y_scroll(item, value, **kwargs)

def setup_dearpygui(**kwargs) -> None:
    if False:
        i = 10
        return i + 15
    '\t Sets up Dear PyGui\n\n\tArgs:\n\t\tviewport (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tNone\n\t'
    if 'viewport' in kwargs.keys():
        warnings.warn('viewport keyword removed', DeprecationWarning, 2)
        kwargs.pop('viewport', None)
    return internal_dpg.setup_dearpygui(**kwargs)

def show_imgui_demo(**kwargs) -> None:
    if False:
        return 10
    '\t Shows the imgui demo.\n\n\tArgs:\n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.show_imgui_demo(**kwargs)

def show_implot_demo(**kwargs) -> None:
    if False:
        return 10
    '\t Shows the implot demo.\n\n\tArgs:\n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.show_implot_demo(**kwargs)

def show_item_debug(item: Union[int, str], **kwargs) -> None:
    if False:
        while True:
            i = 10
    "\t Shows an item's debug window\n\n\tArgs:\n\t\titem (Union[int, str]): \n\tReturns:\n\t\tNone\n\t"
    return internal_dpg.show_item_debug(item, **kwargs)

def show_tool(tool: Union[int, str], **kwargs) -> str:
    if False:
        print('Hello World!')
    '\t Shows a built in tool.\n\n\tArgs:\n\t\ttool (Union[int, str]): \n\tReturns:\n\t\tstr\n\t'
    return internal_dpg.show_tool(tool, **kwargs)

def show_viewport(*, minimized: bool=False, maximized: bool=False, **kwargs) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\t Shows the main viewport.\n\n\tArgs:\n\t\tminimized (bool, optional): Sets the state of the viewport to minimized\n\t\tmaximized (bool, optional): Sets the state of the viewport to maximized\n\t\tviewport (Union[int, str], optional): (deprecated) \n\tReturns:\n\t\tNone\n\t'
    if 'viewport' in kwargs.keys():
        warnings.warn('viewport keyword removed', DeprecationWarning, 2)
        kwargs.pop('viewport', None)
    return internal_dpg.show_viewport(minimized=minimized, maximized=maximized, **kwargs)

def split_frame(*, delay: int=32, **kwargs) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\t Waits one frame.\n\n\tArgs:\n\t\tdelay (int, optional): Minimal delay in in milliseconds\n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.split_frame(delay=delay, **kwargs)

def stop_dearpygui(**kwargs) -> None:
    if False:
        return 10
    '\t Stops Dear PyGui\n\n\tArgs:\n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.stop_dearpygui(**kwargs)

def toggle_viewport_fullscreen(**kwargs) -> None:
    if False:
        i = 10
        return i + 15
    '\t Toggle viewport fullscreen mode..\n\n\tArgs:\n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.toggle_viewport_fullscreen(**kwargs)

def top_container_stack(**kwargs) -> Union[int, str]:
    if False:
        for i in range(10):
            print('nop')
    '\t Returns the item on the top of the container stack.\n\n\tArgs:\n\tReturns:\n\t\tUnion[int, str]\n\t'
    return internal_dpg.top_container_stack(**kwargs)

def unhighlight_table_cell(table: Union[int, str], row: int, column: int, **kwargs) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\t Unhighlight specified table cell.\n\n\tArgs:\n\t\ttable (Union[int, str]): \n\t\trow (int): \n\t\tcolumn (int): \n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.unhighlight_table_cell(table, row, column, **kwargs)

def unhighlight_table_column(table: Union[int, str], column: int, **kwargs) -> None:
    if False:
        while True:
            i = 10
    '\t Unhighlight specified table column.\n\n\tArgs:\n\t\ttable (Union[int, str]): \n\t\tcolumn (int): \n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.unhighlight_table_column(table, column, **kwargs)

def unhighlight_table_row(table: Union[int, str], row: int, **kwargs) -> None:
    if False:
        print('Hello World!')
    '\t Unhighlight specified table row.\n\n\tArgs:\n\t\ttable (Union[int, str]): \n\t\trow (int): \n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.unhighlight_table_row(table, row, **kwargs)

def unlock_mutex(**kwargs) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\t Unlocks render thread mutex\n\n\tArgs:\n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.unlock_mutex(**kwargs)

def unset_table_row_color(table: Union[int, str], row: int, **kwargs) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\t Remove user set table row color.\n\n\tArgs:\n\t\ttable (Union[int, str]): \n\t\trow (int): \n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.unset_table_row_color(table, row, **kwargs)

def unstage(item: Union[int, str], **kwargs) -> None:
    if False:
        print('Hello World!')
    '\t Unstages an item.\n\n\tArgs:\n\t\titem (Union[int, str]): \n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.unstage(item, **kwargs)
mvGraphicsBackend_D3D11 = internal_dpg.mvGraphicsBackend_D3D11
mvGraphicsBackend_D3D12 = internal_dpg.mvGraphicsBackend_D3D12
mvGraphicsBackend_VULKAN = internal_dpg.mvGraphicsBackend_VULKAN
mvGraphicsBackend_METAL = internal_dpg.mvGraphicsBackend_METAL
mvGraphicsBackend_OPENGL = internal_dpg.mvGraphicsBackend_OPENGL
mvMouseButton_Left = internal_dpg.mvMouseButton_Left
mvMouseButton_Right = internal_dpg.mvMouseButton_Right
mvMouseButton_Middle = internal_dpg.mvMouseButton_Middle
mvMouseButton_X1 = internal_dpg.mvMouseButton_X1
mvMouseButton_X2 = internal_dpg.mvMouseButton_X2
mvKey_0 = internal_dpg.mvKey_0
mvKey_1 = internal_dpg.mvKey_1
mvKey_2 = internal_dpg.mvKey_2
mvKey_3 = internal_dpg.mvKey_3
mvKey_4 = internal_dpg.mvKey_4
mvKey_5 = internal_dpg.mvKey_5
mvKey_6 = internal_dpg.mvKey_6
mvKey_7 = internal_dpg.mvKey_7
mvKey_8 = internal_dpg.mvKey_8
mvKey_9 = internal_dpg.mvKey_9
mvKey_A = internal_dpg.mvKey_A
mvKey_B = internal_dpg.mvKey_B
mvKey_C = internal_dpg.mvKey_C
mvKey_D = internal_dpg.mvKey_D
mvKey_E = internal_dpg.mvKey_E
mvKey_F = internal_dpg.mvKey_F
mvKey_G = internal_dpg.mvKey_G
mvKey_H = internal_dpg.mvKey_H
mvKey_I = internal_dpg.mvKey_I
mvKey_J = internal_dpg.mvKey_J
mvKey_K = internal_dpg.mvKey_K
mvKey_L = internal_dpg.mvKey_L
mvKey_M = internal_dpg.mvKey_M
mvKey_N = internal_dpg.mvKey_N
mvKey_O = internal_dpg.mvKey_O
mvKey_P = internal_dpg.mvKey_P
mvKey_Q = internal_dpg.mvKey_Q
mvKey_R = internal_dpg.mvKey_R
mvKey_S = internal_dpg.mvKey_S
mvKey_T = internal_dpg.mvKey_T
mvKey_U = internal_dpg.mvKey_U
mvKey_V = internal_dpg.mvKey_V
mvKey_W = internal_dpg.mvKey_W
mvKey_X = internal_dpg.mvKey_X
mvKey_Y = internal_dpg.mvKey_Y
mvKey_Z = internal_dpg.mvKey_Z
mvKey_Back = internal_dpg.mvKey_Back
mvKey_Tab = internal_dpg.mvKey_Tab
mvKey_Clear = internal_dpg.mvKey_Clear
mvKey_Return = internal_dpg.mvKey_Return
mvKey_Shift = internal_dpg.mvKey_Shift
mvKey_Control = internal_dpg.mvKey_Control
mvKey_Alt = internal_dpg.mvKey_Alt
mvKey_Pause = internal_dpg.mvKey_Pause
mvKey_Capital = internal_dpg.mvKey_Capital
mvKey_Escape = internal_dpg.mvKey_Escape
mvKey_Spacebar = internal_dpg.mvKey_Spacebar
mvKey_Prior = internal_dpg.mvKey_Prior
mvKey_Next = internal_dpg.mvKey_Next
mvKey_End = internal_dpg.mvKey_End
mvKey_Home = internal_dpg.mvKey_Home
mvKey_Left = internal_dpg.mvKey_Left
mvKey_Up = internal_dpg.mvKey_Up
mvKey_Right = internal_dpg.mvKey_Right
mvKey_Down = internal_dpg.mvKey_Down
mvKey_Select = internal_dpg.mvKey_Select
mvKey_Print = internal_dpg.mvKey_Print
mvKey_Execute = internal_dpg.mvKey_Execute
mvKey_PrintScreen = internal_dpg.mvKey_PrintScreen
mvKey_Insert = internal_dpg.mvKey_Insert
mvKey_Delete = internal_dpg.mvKey_Delete
mvKey_Help = internal_dpg.mvKey_Help
mvKey_LWin = internal_dpg.mvKey_LWin
mvKey_RWin = internal_dpg.mvKey_RWin
mvKey_Apps = internal_dpg.mvKey_Apps
mvKey_Sleep = internal_dpg.mvKey_Sleep
mvKey_NumPad0 = internal_dpg.mvKey_NumPad0
mvKey_NumPad1 = internal_dpg.mvKey_NumPad1
mvKey_NumPad2 = internal_dpg.mvKey_NumPad2
mvKey_NumPad3 = internal_dpg.mvKey_NumPad3
mvKey_NumPad4 = internal_dpg.mvKey_NumPad4
mvKey_NumPad5 = internal_dpg.mvKey_NumPad5
mvKey_NumPad6 = internal_dpg.mvKey_NumPad6
mvKey_NumPad7 = internal_dpg.mvKey_NumPad7
mvKey_NumPad8 = internal_dpg.mvKey_NumPad8
mvKey_NumPad9 = internal_dpg.mvKey_NumPad9
mvKey_Multiply = internal_dpg.mvKey_Multiply
mvKey_Add = internal_dpg.mvKey_Add
mvKey_Separator = internal_dpg.mvKey_Separator
mvKey_Subtract = internal_dpg.mvKey_Subtract
mvKey_Decimal = internal_dpg.mvKey_Decimal
mvKey_Divide = internal_dpg.mvKey_Divide
mvKey_F1 = internal_dpg.mvKey_F1
mvKey_F2 = internal_dpg.mvKey_F2
mvKey_F3 = internal_dpg.mvKey_F3
mvKey_F4 = internal_dpg.mvKey_F4
mvKey_F5 = internal_dpg.mvKey_F5
mvKey_F6 = internal_dpg.mvKey_F6
mvKey_F7 = internal_dpg.mvKey_F7
mvKey_F8 = internal_dpg.mvKey_F8
mvKey_F9 = internal_dpg.mvKey_F9
mvKey_F10 = internal_dpg.mvKey_F10
mvKey_F11 = internal_dpg.mvKey_F11
mvKey_F12 = internal_dpg.mvKey_F12
mvKey_F13 = internal_dpg.mvKey_F13
mvKey_F14 = internal_dpg.mvKey_F14
mvKey_F15 = internal_dpg.mvKey_F15
mvKey_F16 = internal_dpg.mvKey_F16
mvKey_F17 = internal_dpg.mvKey_F17
mvKey_F18 = internal_dpg.mvKey_F18
mvKey_F19 = internal_dpg.mvKey_F19
mvKey_F20 = internal_dpg.mvKey_F20
mvKey_F21 = internal_dpg.mvKey_F21
mvKey_F22 = internal_dpg.mvKey_F22
mvKey_F23 = internal_dpg.mvKey_F23
mvKey_F24 = internal_dpg.mvKey_F24
mvKey_F25 = internal_dpg.mvKey_F25
mvKey_NumLock = internal_dpg.mvKey_NumLock
mvKey_ScrollLock = internal_dpg.mvKey_ScrollLock
mvKey_LShift = internal_dpg.mvKey_LShift
mvKey_RShift = internal_dpg.mvKey_RShift
mvKey_LControl = internal_dpg.mvKey_LControl
mvKey_RControl = internal_dpg.mvKey_RControl
mvKey_LMenu = internal_dpg.mvKey_LMenu
mvKey_RMenu = internal_dpg.mvKey_RMenu
mvKey_Browser_Back = internal_dpg.mvKey_Browser_Back
mvKey_Browser_Forward = internal_dpg.mvKey_Browser_Forward
mvKey_Browser_Refresh = internal_dpg.mvKey_Browser_Refresh
mvKey_Browser_Stop = internal_dpg.mvKey_Browser_Stop
mvKey_Browser_Search = internal_dpg.mvKey_Browser_Search
mvKey_Browser_Favorites = internal_dpg.mvKey_Browser_Favorites
mvKey_Browser_Home = internal_dpg.mvKey_Browser_Home
mvKey_Volume_Mute = internal_dpg.mvKey_Volume_Mute
mvKey_Volume_Down = internal_dpg.mvKey_Volume_Down
mvKey_Volume_Up = internal_dpg.mvKey_Volume_Up
mvKey_Media_Next_Track = internal_dpg.mvKey_Media_Next_Track
mvKey_Media_Prev_Track = internal_dpg.mvKey_Media_Prev_Track
mvKey_Media_Stop = internal_dpg.mvKey_Media_Stop
mvKey_Media_Play_Pause = internal_dpg.mvKey_Media_Play_Pause
mvKey_Launch_Mail = internal_dpg.mvKey_Launch_Mail
mvKey_Launch_Media_Select = internal_dpg.mvKey_Launch_Media_Select
mvKey_Launch_App1 = internal_dpg.mvKey_Launch_App1
mvKey_Launch_App2 = internal_dpg.mvKey_Launch_App2
mvKey_Colon = internal_dpg.mvKey_Colon
mvKey_Plus = internal_dpg.mvKey_Plus
mvKey_Comma = internal_dpg.mvKey_Comma
mvKey_Minus = internal_dpg.mvKey_Minus
mvKey_Period = internal_dpg.mvKey_Period
mvKey_Slash = internal_dpg.mvKey_Slash
mvKey_Tilde = internal_dpg.mvKey_Tilde
mvKey_Open_Brace = internal_dpg.mvKey_Open_Brace
mvKey_Backslash = internal_dpg.mvKey_Backslash
mvKey_Close_Brace = internal_dpg.mvKey_Close_Brace
mvKey_Quote = internal_dpg.mvKey_Quote
mvAll = internal_dpg.mvAll
mvTool_About = internal_dpg.mvTool_About
mvTool_Debug = internal_dpg.mvTool_Debug
mvTool_Doc = internal_dpg.mvTool_Doc
mvTool_ItemRegistry = internal_dpg.mvTool_ItemRegistry
mvTool_Metrics = internal_dpg.mvTool_Metrics
mvTool_Style = internal_dpg.mvTool_Style
mvTool_Font = internal_dpg.mvTool_Font
mvFontAtlas = internal_dpg.mvFontAtlas
mvAppUUID = internal_dpg.mvAppUUID
mvInvalidUUID = internal_dpg.mvInvalidUUID
mvDir_None = internal_dpg.mvDir_None
mvDir_Left = internal_dpg.mvDir_Left
mvDir_Right = internal_dpg.mvDir_Right
mvDir_Up = internal_dpg.mvDir_Up
mvDir_Down = internal_dpg.mvDir_Down
mvComboHeight_Small = internal_dpg.mvComboHeight_Small
mvComboHeight_Regular = internal_dpg.mvComboHeight_Regular
mvComboHeight_Large = internal_dpg.mvComboHeight_Large
mvComboHeight_Largest = internal_dpg.mvComboHeight_Largest
mvPlatform_Windows = internal_dpg.mvPlatform_Windows
mvPlatform_Apple = internal_dpg.mvPlatform_Apple
mvPlatform_Linux = internal_dpg.mvPlatform_Linux
mvColorEdit_AlphaPreviewNone = internal_dpg.mvColorEdit_AlphaPreviewNone
mvColorEdit_AlphaPreview = internal_dpg.mvColorEdit_AlphaPreview
mvColorEdit_AlphaPreviewHalf = internal_dpg.mvColorEdit_AlphaPreviewHalf
mvColorEdit_uint8 = internal_dpg.mvColorEdit_uint8
mvColorEdit_float = internal_dpg.mvColorEdit_float
mvColorEdit_rgb = internal_dpg.mvColorEdit_rgb
mvColorEdit_hsv = internal_dpg.mvColorEdit_hsv
mvColorEdit_hex = internal_dpg.mvColorEdit_hex
mvColorEdit_input_rgb = internal_dpg.mvColorEdit_input_rgb
mvColorEdit_input_hsv = internal_dpg.mvColorEdit_input_hsv
mvPlotColormap_Default = internal_dpg.mvPlotColormap_Default
mvPlotColormap_Deep = internal_dpg.mvPlotColormap_Deep
mvPlotColormap_Dark = internal_dpg.mvPlotColormap_Dark
mvPlotColormap_Pastel = internal_dpg.mvPlotColormap_Pastel
mvPlotColormap_Paired = internal_dpg.mvPlotColormap_Paired
mvPlotColormap_Viridis = internal_dpg.mvPlotColormap_Viridis
mvPlotColormap_Plasma = internal_dpg.mvPlotColormap_Plasma
mvPlotColormap_Hot = internal_dpg.mvPlotColormap_Hot
mvPlotColormap_Cool = internal_dpg.mvPlotColormap_Cool
mvPlotColormap_Pink = internal_dpg.mvPlotColormap_Pink
mvPlotColormap_Jet = internal_dpg.mvPlotColormap_Jet
mvPlotColormap_Twilight = internal_dpg.mvPlotColormap_Twilight
mvPlotColormap_RdBu = internal_dpg.mvPlotColormap_RdBu
mvPlotColormap_BrBG = internal_dpg.mvPlotColormap_BrBG
mvPlotColormap_PiYG = internal_dpg.mvPlotColormap_PiYG
mvPlotColormap_Spectral = internal_dpg.mvPlotColormap_Spectral
mvPlotColormap_Greys = internal_dpg.mvPlotColormap_Greys
mvColorPicker_bar = internal_dpg.mvColorPicker_bar
mvColorPicker_wheel = internal_dpg.mvColorPicker_wheel
mvTabOrder_Reorderable = internal_dpg.mvTabOrder_Reorderable
mvTabOrder_Fixed = internal_dpg.mvTabOrder_Fixed
mvTabOrder_Leading = internal_dpg.mvTabOrder_Leading
mvTabOrder_Trailing = internal_dpg.mvTabOrder_Trailing
mvTimeUnit_Us = internal_dpg.mvTimeUnit_Us
mvTimeUnit_Ms = internal_dpg.mvTimeUnit_Ms
mvTimeUnit_S = internal_dpg.mvTimeUnit_S
mvTimeUnit_Min = internal_dpg.mvTimeUnit_Min
mvTimeUnit_Hr = internal_dpg.mvTimeUnit_Hr
mvTimeUnit_Day = internal_dpg.mvTimeUnit_Day
mvTimeUnit_Mo = internal_dpg.mvTimeUnit_Mo
mvTimeUnit_Yr = internal_dpg.mvTimeUnit_Yr
mvDatePickerLevel_Day = internal_dpg.mvDatePickerLevel_Day
mvDatePickerLevel_Month = internal_dpg.mvDatePickerLevel_Month
mvDatePickerLevel_Year = internal_dpg.mvDatePickerLevel_Year
mvCullMode_None = internal_dpg.mvCullMode_None
mvCullMode_Back = internal_dpg.mvCullMode_Back
mvCullMode_Front = internal_dpg.mvCullMode_Front
mvFontRangeHint_Default = internal_dpg.mvFontRangeHint_Default
mvFontRangeHint_Japanese = internal_dpg.mvFontRangeHint_Japanese
mvFontRangeHint_Korean = internal_dpg.mvFontRangeHint_Korean
mvFontRangeHint_Chinese_Full = internal_dpg.mvFontRangeHint_Chinese_Full
mvFontRangeHint_Chinese_Simplified_Common = internal_dpg.mvFontRangeHint_Chinese_Simplified_Common
mvFontRangeHint_Cyrillic = internal_dpg.mvFontRangeHint_Cyrillic
mvFontRangeHint_Thai = internal_dpg.mvFontRangeHint_Thai
mvFontRangeHint_Vietnamese = internal_dpg.mvFontRangeHint_Vietnamese
mvNode_PinShape_Circle = internal_dpg.mvNode_PinShape_Circle
mvNode_PinShape_CircleFilled = internal_dpg.mvNode_PinShape_CircleFilled
mvNode_PinShape_Triangle = internal_dpg.mvNode_PinShape_Triangle
mvNode_PinShape_TriangleFilled = internal_dpg.mvNode_PinShape_TriangleFilled
mvNode_PinShape_Quad = internal_dpg.mvNode_PinShape_Quad
mvNode_PinShape_QuadFilled = internal_dpg.mvNode_PinShape_QuadFilled
mvNode_Attr_Input = internal_dpg.mvNode_Attr_Input
mvNode_Attr_Output = internal_dpg.mvNode_Attr_Output
mvNode_Attr_Static = internal_dpg.mvNode_Attr_Static
mvPlotBin_Sqrt = internal_dpg.mvPlotBin_Sqrt
mvPlotBin_Sturges = internal_dpg.mvPlotBin_Sturges
mvPlotBin_Rice = internal_dpg.mvPlotBin_Rice
mvPlotBin_Scott = internal_dpg.mvPlotBin_Scott
mvXAxis = internal_dpg.mvXAxis
mvYAxis = internal_dpg.mvYAxis
mvPlotMarker_None = internal_dpg.mvPlotMarker_None
mvPlotMarker_Circle = internal_dpg.mvPlotMarker_Circle
mvPlotMarker_Square = internal_dpg.mvPlotMarker_Square
mvPlotMarker_Diamond = internal_dpg.mvPlotMarker_Diamond
mvPlotMarker_Up = internal_dpg.mvPlotMarker_Up
mvPlotMarker_Down = internal_dpg.mvPlotMarker_Down
mvPlotMarker_Left = internal_dpg.mvPlotMarker_Left
mvPlotMarker_Right = internal_dpg.mvPlotMarker_Right
mvPlotMarker_Cross = internal_dpg.mvPlotMarker_Cross
mvPlotMarker_Plus = internal_dpg.mvPlotMarker_Plus
mvPlotMarker_Asterisk = internal_dpg.mvPlotMarker_Asterisk
mvPlot_Location_Center = internal_dpg.mvPlot_Location_Center
mvPlot_Location_North = internal_dpg.mvPlot_Location_North
mvPlot_Location_South = internal_dpg.mvPlot_Location_South
mvPlot_Location_West = internal_dpg.mvPlot_Location_West
mvPlot_Location_East = internal_dpg.mvPlot_Location_East
mvPlot_Location_NorthWest = internal_dpg.mvPlot_Location_NorthWest
mvPlot_Location_NorthEast = internal_dpg.mvPlot_Location_NorthEast
mvPlot_Location_SouthWest = internal_dpg.mvPlot_Location_SouthWest
mvPlot_Location_SouthEast = internal_dpg.mvPlot_Location_SouthEast
mvNodeMiniMap_Location_BottomLeft = internal_dpg.mvNodeMiniMap_Location_BottomLeft
mvNodeMiniMap_Location_BottomRight = internal_dpg.mvNodeMiniMap_Location_BottomRight
mvNodeMiniMap_Location_TopLeft = internal_dpg.mvNodeMiniMap_Location_TopLeft
mvNodeMiniMap_Location_TopRight = internal_dpg.mvNodeMiniMap_Location_TopRight
mvTable_SizingFixedFit = internal_dpg.mvTable_SizingFixedFit
mvTable_SizingFixedSame = internal_dpg.mvTable_SizingFixedSame
mvTable_SizingStretchProp = internal_dpg.mvTable_SizingStretchProp
mvTable_SizingStretchSame = internal_dpg.mvTable_SizingStretchSame
mvFormat_Float_rgba = internal_dpg.mvFormat_Float_rgba
mvFormat_Float_rgb = internal_dpg.mvFormat_Float_rgb
mvThemeCat_Core = internal_dpg.mvThemeCat_Core
mvThemeCat_Plots = internal_dpg.mvThemeCat_Plots
mvThemeCat_Nodes = internal_dpg.mvThemeCat_Nodes
mvThemeCol_Text = internal_dpg.mvThemeCol_Text
mvThemeCol_TextDisabled = internal_dpg.mvThemeCol_TextDisabled
mvThemeCol_WindowBg = internal_dpg.mvThemeCol_WindowBg
mvThemeCol_ChildBg = internal_dpg.mvThemeCol_ChildBg
mvThemeCol_Border = internal_dpg.mvThemeCol_Border
mvThemeCol_PopupBg = internal_dpg.mvThemeCol_PopupBg
mvThemeCol_BorderShadow = internal_dpg.mvThemeCol_BorderShadow
mvThemeCol_FrameBg = internal_dpg.mvThemeCol_FrameBg
mvThemeCol_FrameBgHovered = internal_dpg.mvThemeCol_FrameBgHovered
mvThemeCol_FrameBgActive = internal_dpg.mvThemeCol_FrameBgActive
mvThemeCol_TitleBg = internal_dpg.mvThemeCol_TitleBg
mvThemeCol_TitleBgActive = internal_dpg.mvThemeCol_TitleBgActive
mvThemeCol_TitleBgCollapsed = internal_dpg.mvThemeCol_TitleBgCollapsed
mvThemeCol_MenuBarBg = internal_dpg.mvThemeCol_MenuBarBg
mvThemeCol_ScrollbarBg = internal_dpg.mvThemeCol_ScrollbarBg
mvThemeCol_ScrollbarGrab = internal_dpg.mvThemeCol_ScrollbarGrab
mvThemeCol_ScrollbarGrabHovered = internal_dpg.mvThemeCol_ScrollbarGrabHovered
mvThemeCol_ScrollbarGrabActive = internal_dpg.mvThemeCol_ScrollbarGrabActive
mvThemeCol_CheckMark = internal_dpg.mvThemeCol_CheckMark
mvThemeCol_SliderGrab = internal_dpg.mvThemeCol_SliderGrab
mvThemeCol_SliderGrabActive = internal_dpg.mvThemeCol_SliderGrabActive
mvThemeCol_Button = internal_dpg.mvThemeCol_Button
mvThemeCol_ButtonHovered = internal_dpg.mvThemeCol_ButtonHovered
mvThemeCol_ButtonActive = internal_dpg.mvThemeCol_ButtonActive
mvThemeCol_Header = internal_dpg.mvThemeCol_Header
mvThemeCol_HeaderHovered = internal_dpg.mvThemeCol_HeaderHovered
mvThemeCol_HeaderActive = internal_dpg.mvThemeCol_HeaderActive
mvThemeCol_Separator = internal_dpg.mvThemeCol_Separator
mvThemeCol_SeparatorHovered = internal_dpg.mvThemeCol_SeparatorHovered
mvThemeCol_SeparatorActive = internal_dpg.mvThemeCol_SeparatorActive
mvThemeCol_ResizeGrip = internal_dpg.mvThemeCol_ResizeGrip
mvThemeCol_ResizeGripHovered = internal_dpg.mvThemeCol_ResizeGripHovered
mvThemeCol_ResizeGripActive = internal_dpg.mvThemeCol_ResizeGripActive
mvThemeCol_Tab = internal_dpg.mvThemeCol_Tab
mvThemeCol_TabHovered = internal_dpg.mvThemeCol_TabHovered
mvThemeCol_TabActive = internal_dpg.mvThemeCol_TabActive
mvThemeCol_TabUnfocused = internal_dpg.mvThemeCol_TabUnfocused
mvThemeCol_TabUnfocusedActive = internal_dpg.mvThemeCol_TabUnfocusedActive
mvThemeCol_DockingPreview = internal_dpg.mvThemeCol_DockingPreview
mvThemeCol_DockingEmptyBg = internal_dpg.mvThemeCol_DockingEmptyBg
mvThemeCol_PlotLines = internal_dpg.mvThemeCol_PlotLines
mvThemeCol_PlotLinesHovered = internal_dpg.mvThemeCol_PlotLinesHovered
mvThemeCol_PlotHistogram = internal_dpg.mvThemeCol_PlotHistogram
mvThemeCol_PlotHistogramHovered = internal_dpg.mvThemeCol_PlotHistogramHovered
mvThemeCol_TableHeaderBg = internal_dpg.mvThemeCol_TableHeaderBg
mvThemeCol_TableBorderStrong = internal_dpg.mvThemeCol_TableBorderStrong
mvThemeCol_TableBorderLight = internal_dpg.mvThemeCol_TableBorderLight
mvThemeCol_TableRowBg = internal_dpg.mvThemeCol_TableRowBg
mvThemeCol_TableRowBgAlt = internal_dpg.mvThemeCol_TableRowBgAlt
mvThemeCol_TextSelectedBg = internal_dpg.mvThemeCol_TextSelectedBg
mvThemeCol_DragDropTarget = internal_dpg.mvThemeCol_DragDropTarget
mvThemeCol_NavHighlight = internal_dpg.mvThemeCol_NavHighlight
mvThemeCol_NavWindowingHighlight = internal_dpg.mvThemeCol_NavWindowingHighlight
mvThemeCol_NavWindowingDimBg = internal_dpg.mvThemeCol_NavWindowingDimBg
mvThemeCol_ModalWindowDimBg = internal_dpg.mvThemeCol_ModalWindowDimBg
mvPlotCol_Line = internal_dpg.mvPlotCol_Line
mvPlotCol_Fill = internal_dpg.mvPlotCol_Fill
mvPlotCol_MarkerOutline = internal_dpg.mvPlotCol_MarkerOutline
mvPlotCol_MarkerFill = internal_dpg.mvPlotCol_MarkerFill
mvPlotCol_ErrorBar = internal_dpg.mvPlotCol_ErrorBar
mvPlotCol_FrameBg = internal_dpg.mvPlotCol_FrameBg
mvPlotCol_PlotBg = internal_dpg.mvPlotCol_PlotBg
mvPlotCol_PlotBorder = internal_dpg.mvPlotCol_PlotBorder
mvPlotCol_LegendBg = internal_dpg.mvPlotCol_LegendBg
mvPlotCol_LegendBorder = internal_dpg.mvPlotCol_LegendBorder
mvPlotCol_LegendText = internal_dpg.mvPlotCol_LegendText
mvPlotCol_TitleText = internal_dpg.mvPlotCol_TitleText
mvPlotCol_InlayText = internal_dpg.mvPlotCol_InlayText
mvPlotCol_XAxis = internal_dpg.mvPlotCol_XAxis
mvPlotCol_XAxisGrid = internal_dpg.mvPlotCol_XAxisGrid
mvPlotCol_YAxis = internal_dpg.mvPlotCol_YAxis
mvPlotCol_YAxisGrid = internal_dpg.mvPlotCol_YAxisGrid
mvPlotCol_YAxis2 = internal_dpg.mvPlotCol_YAxis2
mvPlotCol_YAxisGrid2 = internal_dpg.mvPlotCol_YAxisGrid2
mvPlotCol_YAxis3 = internal_dpg.mvPlotCol_YAxis3
mvPlotCol_YAxisGrid3 = internal_dpg.mvPlotCol_YAxisGrid3
mvPlotCol_Selection = internal_dpg.mvPlotCol_Selection
mvPlotCol_Query = internal_dpg.mvPlotCol_Query
mvPlotCol_Crosshairs = internal_dpg.mvPlotCol_Crosshairs
mvNodeCol_NodeBackground = internal_dpg.mvNodeCol_NodeBackground
mvNodeCol_NodeBackgroundHovered = internal_dpg.mvNodeCol_NodeBackgroundHovered
mvNodeCol_NodeBackgroundSelected = internal_dpg.mvNodeCol_NodeBackgroundSelected
mvNodeCol_NodeOutline = internal_dpg.mvNodeCol_NodeOutline
mvNodeCol_TitleBar = internal_dpg.mvNodeCol_TitleBar
mvNodeCol_TitleBarHovered = internal_dpg.mvNodeCol_TitleBarHovered
mvNodeCol_TitleBarSelected = internal_dpg.mvNodeCol_TitleBarSelected
mvNodeCol_Link = internal_dpg.mvNodeCol_Link
mvNodeCol_LinkHovered = internal_dpg.mvNodeCol_LinkHovered
mvNodeCol_LinkSelected = internal_dpg.mvNodeCol_LinkSelected
mvNodeCol_Pin = internal_dpg.mvNodeCol_Pin
mvNodeCol_PinHovered = internal_dpg.mvNodeCol_PinHovered
mvNodeCol_BoxSelector = internal_dpg.mvNodeCol_BoxSelector
mvNodeCol_BoxSelectorOutline = internal_dpg.mvNodeCol_BoxSelectorOutline
mvNodeCol_GridBackground = internal_dpg.mvNodeCol_GridBackground
mvNodeCol_GridLine = internal_dpg.mvNodeCol_GridLine
mvNodesCol_GridLinePrimary = internal_dpg.mvNodesCol_GridLinePrimary
mvNodesCol_MiniMapBackground = internal_dpg.mvNodesCol_MiniMapBackground
mvNodesCol_MiniMapBackgroundHovered = internal_dpg.mvNodesCol_MiniMapBackgroundHovered
mvNodesCol_MiniMapOutline = internal_dpg.mvNodesCol_MiniMapOutline
mvNodesCol_MiniMapOutlineHovered = internal_dpg.mvNodesCol_MiniMapOutlineHovered
mvNodesCol_MiniMapNodeBackground = internal_dpg.mvNodesCol_MiniMapNodeBackground
mvNodesCol_MiniMapNodeBackgroundHovered = internal_dpg.mvNodesCol_MiniMapNodeBackgroundHovered
mvNodesCol_MiniMapNodeBackgroundSelected = internal_dpg.mvNodesCol_MiniMapNodeBackgroundSelected
mvNodesCol_MiniMapNodeOutline = internal_dpg.mvNodesCol_MiniMapNodeOutline
mvNodesCol_MiniMapLink = internal_dpg.mvNodesCol_MiniMapLink
mvNodesCol_MiniMapLinkSelected = internal_dpg.mvNodesCol_MiniMapLinkSelected
mvNodesCol_MiniMapCanvas = internal_dpg.mvNodesCol_MiniMapCanvas
mvNodesCol_MiniMapCanvasOutline = internal_dpg.mvNodesCol_MiniMapCanvasOutline
mvStyleVar_Alpha = internal_dpg.mvStyleVar_Alpha
mvStyleVar_WindowPadding = internal_dpg.mvStyleVar_WindowPadding
mvStyleVar_WindowRounding = internal_dpg.mvStyleVar_WindowRounding
mvStyleVar_WindowBorderSize = internal_dpg.mvStyleVar_WindowBorderSize
mvStyleVar_WindowMinSize = internal_dpg.mvStyleVar_WindowMinSize
mvStyleVar_WindowTitleAlign = internal_dpg.mvStyleVar_WindowTitleAlign
mvStyleVar_ChildRounding = internal_dpg.mvStyleVar_ChildRounding
mvStyleVar_ChildBorderSize = internal_dpg.mvStyleVar_ChildBorderSize
mvStyleVar_PopupRounding = internal_dpg.mvStyleVar_PopupRounding
mvStyleVar_PopupBorderSize = internal_dpg.mvStyleVar_PopupBorderSize
mvStyleVar_FramePadding = internal_dpg.mvStyleVar_FramePadding
mvStyleVar_FrameRounding = internal_dpg.mvStyleVar_FrameRounding
mvStyleVar_FrameBorderSize = internal_dpg.mvStyleVar_FrameBorderSize
mvStyleVar_ItemSpacing = internal_dpg.mvStyleVar_ItemSpacing
mvStyleVar_ItemInnerSpacing = internal_dpg.mvStyleVar_ItemInnerSpacing
mvStyleVar_IndentSpacing = internal_dpg.mvStyleVar_IndentSpacing
mvStyleVar_CellPadding = internal_dpg.mvStyleVar_CellPadding
mvStyleVar_ScrollbarSize = internal_dpg.mvStyleVar_ScrollbarSize
mvStyleVar_ScrollbarRounding = internal_dpg.mvStyleVar_ScrollbarRounding
mvStyleVar_GrabMinSize = internal_dpg.mvStyleVar_GrabMinSize
mvStyleVar_GrabRounding = internal_dpg.mvStyleVar_GrabRounding
mvStyleVar_TabRounding = internal_dpg.mvStyleVar_TabRounding
mvStyleVar_ButtonTextAlign = internal_dpg.mvStyleVar_ButtonTextAlign
mvStyleVar_SelectableTextAlign = internal_dpg.mvStyleVar_SelectableTextAlign
mvPlotStyleVar_LineWeight = internal_dpg.mvPlotStyleVar_LineWeight
mvPlotStyleVar_Marker = internal_dpg.mvPlotStyleVar_Marker
mvPlotStyleVar_MarkerSize = internal_dpg.mvPlotStyleVar_MarkerSize
mvPlotStyleVar_MarkerWeight = internal_dpg.mvPlotStyleVar_MarkerWeight
mvPlotStyleVar_FillAlpha = internal_dpg.mvPlotStyleVar_FillAlpha
mvPlotStyleVar_ErrorBarSize = internal_dpg.mvPlotStyleVar_ErrorBarSize
mvPlotStyleVar_ErrorBarWeight = internal_dpg.mvPlotStyleVar_ErrorBarWeight
mvPlotStyleVar_DigitalBitHeight = internal_dpg.mvPlotStyleVar_DigitalBitHeight
mvPlotStyleVar_DigitalBitGap = internal_dpg.mvPlotStyleVar_DigitalBitGap
mvPlotStyleVar_PlotBorderSize = internal_dpg.mvPlotStyleVar_PlotBorderSize
mvPlotStyleVar_MinorAlpha = internal_dpg.mvPlotStyleVar_MinorAlpha
mvPlotStyleVar_MajorTickLen = internal_dpg.mvPlotStyleVar_MajorTickLen
mvPlotStyleVar_MinorTickLen = internal_dpg.mvPlotStyleVar_MinorTickLen
mvPlotStyleVar_MajorTickSize = internal_dpg.mvPlotStyleVar_MajorTickSize
mvPlotStyleVar_MinorTickSize = internal_dpg.mvPlotStyleVar_MinorTickSize
mvPlotStyleVar_MajorGridSize = internal_dpg.mvPlotStyleVar_MajorGridSize
mvPlotStyleVar_MinorGridSize = internal_dpg.mvPlotStyleVar_MinorGridSize
mvPlotStyleVar_PlotPadding = internal_dpg.mvPlotStyleVar_PlotPadding
mvPlotStyleVar_LabelPadding = internal_dpg.mvPlotStyleVar_LabelPadding
mvPlotStyleVar_LegendPadding = internal_dpg.mvPlotStyleVar_LegendPadding
mvPlotStyleVar_LegendInnerPadding = internal_dpg.mvPlotStyleVar_LegendInnerPadding
mvPlotStyleVar_LegendSpacing = internal_dpg.mvPlotStyleVar_LegendSpacing
mvPlotStyleVar_MousePosPadding = internal_dpg.mvPlotStyleVar_MousePosPadding
mvPlotStyleVar_AnnotationPadding = internal_dpg.mvPlotStyleVar_AnnotationPadding
mvPlotStyleVar_FitPadding = internal_dpg.mvPlotStyleVar_FitPadding
mvPlotStyleVar_PlotDefaultSize = internal_dpg.mvPlotStyleVar_PlotDefaultSize
mvPlotStyleVar_PlotMinSize = internal_dpg.mvPlotStyleVar_PlotMinSize
mvNodeStyleVar_GridSpacing = internal_dpg.mvNodeStyleVar_GridSpacing
mvNodeStyleVar_NodeCornerRounding = internal_dpg.mvNodeStyleVar_NodeCornerRounding
mvNodeStyleVar_NodePadding = internal_dpg.mvNodeStyleVar_NodePadding
mvNodeStyleVar_NodeBorderThickness = internal_dpg.mvNodeStyleVar_NodeBorderThickness
mvNodeStyleVar_LinkThickness = internal_dpg.mvNodeStyleVar_LinkThickness
mvNodeStyleVar_LinkLineSegmentsPerLength = internal_dpg.mvNodeStyleVar_LinkLineSegmentsPerLength
mvNodeStyleVar_LinkHoverDistance = internal_dpg.mvNodeStyleVar_LinkHoverDistance
mvNodeStyleVar_PinCircleRadius = internal_dpg.mvNodeStyleVar_PinCircleRadius
mvNodeStyleVar_PinQuadSideLength = internal_dpg.mvNodeStyleVar_PinQuadSideLength
mvNodeStyleVar_PinTriangleSideLength = internal_dpg.mvNodeStyleVar_PinTriangleSideLength
mvNodeStyleVar_PinLineThickness = internal_dpg.mvNodeStyleVar_PinLineThickness
mvNodeStyleVar_PinHoverRadius = internal_dpg.mvNodeStyleVar_PinHoverRadius
mvNodeStyleVar_PinOffset = internal_dpg.mvNodeStyleVar_PinOffset
mvNodesStyleVar_MiniMapPadding = internal_dpg.mvNodesStyleVar_MiniMapPadding
mvNodesStyleVar_MiniMapOffset = internal_dpg.mvNodesStyleVar_MiniMapOffset
mvInputText = internal_dpg.mvInputText
mvButton = internal_dpg.mvButton
mvRadioButton = internal_dpg.mvRadioButton
mvTabBar = internal_dpg.mvTabBar
mvTab = internal_dpg.mvTab
mvImage = internal_dpg.mvImage
mvMenuBar = internal_dpg.mvMenuBar
mvViewportMenuBar = internal_dpg.mvViewportMenuBar
mvMenu = internal_dpg.mvMenu
mvMenuItem = internal_dpg.mvMenuItem
mvChildWindow = internal_dpg.mvChildWindow
mvGroup = internal_dpg.mvGroup
mvSliderFloat = internal_dpg.mvSliderFloat
mvSliderInt = internal_dpg.mvSliderInt
mvFilterSet = internal_dpg.mvFilterSet
mvDragFloat = internal_dpg.mvDragFloat
mvDragInt = internal_dpg.mvDragInt
mvInputFloat = internal_dpg.mvInputFloat
mvInputInt = internal_dpg.mvInputInt
mvColorEdit = internal_dpg.mvColorEdit
mvClipper = internal_dpg.mvClipper
mvColorPicker = internal_dpg.mvColorPicker
mvTooltip = internal_dpg.mvTooltip
mvCollapsingHeader = internal_dpg.mvCollapsingHeader
mvSeparator = internal_dpg.mvSeparator
mvCheckbox = internal_dpg.mvCheckbox
mvListbox = internal_dpg.mvListbox
mvText = internal_dpg.mvText
mvCombo = internal_dpg.mvCombo
mvPlot = internal_dpg.mvPlot
mvSimplePlot = internal_dpg.mvSimplePlot
mvDrawlist = internal_dpg.mvDrawlist
mvWindowAppItem = internal_dpg.mvWindowAppItem
mvSelectable = internal_dpg.mvSelectable
mvTreeNode = internal_dpg.mvTreeNode
mvProgressBar = internal_dpg.mvProgressBar
mvSpacer = internal_dpg.mvSpacer
mvImageButton = internal_dpg.mvImageButton
mvTimePicker = internal_dpg.mvTimePicker
mvDatePicker = internal_dpg.mvDatePicker
mvColorButton = internal_dpg.mvColorButton
mvFileDialog = internal_dpg.mvFileDialog
mvTabButton = internal_dpg.mvTabButton
mvDrawNode = internal_dpg.mvDrawNode
mvNodeEditor = internal_dpg.mvNodeEditor
mvNode = internal_dpg.mvNode
mvNodeAttribute = internal_dpg.mvNodeAttribute
mvTable = internal_dpg.mvTable
mvTableColumn = internal_dpg.mvTableColumn
mvTableRow = internal_dpg.mvTableRow
mvDrawLine = internal_dpg.mvDrawLine
mvDrawArrow = internal_dpg.mvDrawArrow
mvDrawTriangle = internal_dpg.mvDrawTriangle
mvDrawImageQuad = internal_dpg.mvDrawImageQuad
mvDrawCircle = internal_dpg.mvDrawCircle
mvDrawEllipse = internal_dpg.mvDrawEllipse
mvDrawBezierCubic = internal_dpg.mvDrawBezierCubic
mvDrawBezierQuadratic = internal_dpg.mvDrawBezierQuadratic
mvDrawQuad = internal_dpg.mvDrawQuad
mvDrawRect = internal_dpg.mvDrawRect
mvDrawText = internal_dpg.mvDrawText
mvDrawPolygon = internal_dpg.mvDrawPolygon
mvDrawPolyline = internal_dpg.mvDrawPolyline
mvDrawImage = internal_dpg.mvDrawImage
mvDragFloatMulti = internal_dpg.mvDragFloatMulti
mvDragIntMulti = internal_dpg.mvDragIntMulti
mvSliderFloatMulti = internal_dpg.mvSliderFloatMulti
mvSliderIntMulti = internal_dpg.mvSliderIntMulti
mvInputIntMulti = internal_dpg.mvInputIntMulti
mvInputFloatMulti = internal_dpg.mvInputFloatMulti
mvDragPoint = internal_dpg.mvDragPoint
mvDragLine = internal_dpg.mvDragLine
mvAnnotation = internal_dpg.mvAnnotation
mvLineSeries = internal_dpg.mvLineSeries
mvScatterSeries = internal_dpg.mvScatterSeries
mvStemSeries = internal_dpg.mvStemSeries
mvStairSeries = internal_dpg.mvStairSeries
mvBarSeries = internal_dpg.mvBarSeries
mvErrorSeries = internal_dpg.mvErrorSeries
mvVLineSeries = internal_dpg.mvVLineSeries
mvHLineSeries = internal_dpg.mvHLineSeries
mvHeatSeries = internal_dpg.mvHeatSeries
mvImageSeries = internal_dpg.mvImageSeries
mvPieSeries = internal_dpg.mvPieSeries
mvShadeSeries = internal_dpg.mvShadeSeries
mvLabelSeries = internal_dpg.mvLabelSeries
mvHistogramSeries = internal_dpg.mvHistogramSeries
mv2dHistogramSeries = internal_dpg.mv2dHistogramSeries
mvCandleSeries = internal_dpg.mvCandleSeries
mvAreaSeries = internal_dpg.mvAreaSeries
mvColorMapScale = internal_dpg.mvColorMapScale
mvSlider3D = internal_dpg.mvSlider3D
mvKnobFloat = internal_dpg.mvKnobFloat
mvLoadingIndicator = internal_dpg.mvLoadingIndicator
mvNodeLink = internal_dpg.mvNodeLink
mvTextureRegistry = internal_dpg.mvTextureRegistry
mvStaticTexture = internal_dpg.mvStaticTexture
mvDynamicTexture = internal_dpg.mvDynamicTexture
mvStage = internal_dpg.mvStage
mvDrawLayer = internal_dpg.mvDrawLayer
mvViewportDrawlist = internal_dpg.mvViewportDrawlist
mvFileExtension = internal_dpg.mvFileExtension
mvPlotLegend = internal_dpg.mvPlotLegend
mvPlotAxis = internal_dpg.mvPlotAxis
mvHandlerRegistry = internal_dpg.mvHandlerRegistry
mvKeyDownHandler = internal_dpg.mvKeyDownHandler
mvKeyPressHandler = internal_dpg.mvKeyPressHandler
mvKeyReleaseHandler = internal_dpg.mvKeyReleaseHandler
mvMouseMoveHandler = internal_dpg.mvMouseMoveHandler
mvMouseWheelHandler = internal_dpg.mvMouseWheelHandler
mvMouseClickHandler = internal_dpg.mvMouseClickHandler
mvMouseDoubleClickHandler = internal_dpg.mvMouseDoubleClickHandler
mvMouseDownHandler = internal_dpg.mvMouseDownHandler
mvMouseReleaseHandler = internal_dpg.mvMouseReleaseHandler
mvMouseDragHandler = internal_dpg.mvMouseDragHandler
mvHoverHandler = internal_dpg.mvHoverHandler
mvActiveHandler = internal_dpg.mvActiveHandler
mvFocusHandler = internal_dpg.mvFocusHandler
mvVisibleHandler = internal_dpg.mvVisibleHandler
mvEditedHandler = internal_dpg.mvEditedHandler
mvActivatedHandler = internal_dpg.mvActivatedHandler
mvDeactivatedHandler = internal_dpg.mvDeactivatedHandler
mvDeactivatedAfterEditHandler = internal_dpg.mvDeactivatedAfterEditHandler
mvToggledOpenHandler = internal_dpg.mvToggledOpenHandler
mvClickedHandler = internal_dpg.mvClickedHandler
mvDoubleClickedHandler = internal_dpg.mvDoubleClickedHandler
mvDragPayload = internal_dpg.mvDragPayload
mvResizeHandler = internal_dpg.mvResizeHandler
mvFont = internal_dpg.mvFont
mvFontRegistry = internal_dpg.mvFontRegistry
mvTheme = internal_dpg.mvTheme
mvThemeColor = internal_dpg.mvThemeColor
mvThemeStyle = internal_dpg.mvThemeStyle
mvThemeComponent = internal_dpg.mvThemeComponent
mvFontRangeHint = internal_dpg.mvFontRangeHint
mvFontRange = internal_dpg.mvFontRange
mvFontChars = internal_dpg.mvFontChars
mvCharRemap = internal_dpg.mvCharRemap
mvValueRegistry = internal_dpg.mvValueRegistry
mvIntValue = internal_dpg.mvIntValue
mvFloatValue = internal_dpg.mvFloatValue
mvFloat4Value = internal_dpg.mvFloat4Value
mvInt4Value = internal_dpg.mvInt4Value
mvBoolValue = internal_dpg.mvBoolValue
mvStringValue = internal_dpg.mvStringValue
mvDoubleValue = internal_dpg.mvDoubleValue
mvDouble4Value = internal_dpg.mvDouble4Value
mvColorValue = internal_dpg.mvColorValue
mvFloatVectValue = internal_dpg.mvFloatVectValue
mvSeriesValue = internal_dpg.mvSeriesValue
mvRawTexture = internal_dpg.mvRawTexture
mvSubPlots = internal_dpg.mvSubPlots
mvColorMap = internal_dpg.mvColorMap
mvColorMapRegistry = internal_dpg.mvColorMapRegistry
mvColorMapButton = internal_dpg.mvColorMapButton
mvColorMapSlider = internal_dpg.mvColorMapSlider
mvTemplateRegistry = internal_dpg.mvTemplateRegistry
mvTableCell = internal_dpg.mvTableCell
mvItemHandlerRegistry = internal_dpg.mvItemHandlerRegistry
mvInputDouble = internal_dpg.mvInputDouble
mvInputDoubleMulti = internal_dpg.mvInputDoubleMulti
mvDragDouble = internal_dpg.mvDragDouble
mvDragDoubleMulti = internal_dpg.mvDragDoubleMulti
mvSliderDouble = internal_dpg.mvSliderDouble
mvSliderDoubleMulti = internal_dpg.mvSliderDoubleMulti
mvCustomSeries = internal_dpg.mvCustomSeries
mvReservedUUID_0 = internal_dpg.mvReservedUUID_0
mvReservedUUID_1 = internal_dpg.mvReservedUUID_1
mvReservedUUID_2 = internal_dpg.mvReservedUUID_2
mvReservedUUID_3 = internal_dpg.mvReservedUUID_3
mvReservedUUID_4 = internal_dpg.mvReservedUUID_4
mvReservedUUID_5 = internal_dpg.mvReservedUUID_5
mvReservedUUID_6 = internal_dpg.mvReservedUUID_6
mvReservedUUID_7 = internal_dpg.mvReservedUUID_7
mvReservedUUID_8 = internal_dpg.mvReservedUUID_8
mvReservedUUID_9 = internal_dpg.mvReservedUUID_9