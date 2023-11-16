"""The main Chat app."""
import reflex as rx
from ..styles import *
from ..webui import styles
from ..webui.components import chat, modal, navbar, sidebar

def chatapp_page() -> rx.Component:
    if False:
        i = 10
        return i + 15
    'The main app.\n\n    Returns:\n        The UI for the main app.\n    '
    return rx.box(rx.vstack(navbar(), chat.chat(), chat.action_bar(), sidebar(), modal(), bg=styles.bg_dark_color, color=styles.text_light_color, min_h='100vh', align_items='stretch', spacing='0', style=template_content_style), style=template_page_style)