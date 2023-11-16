from typing import Tuple
from nicegui import ui

class leaflet(ui.element, component='leaflet.js'):

    def __init__(self) -> None:
        if False:
            return 10
        super().__init__()
        ui.add_head_html('<link href="https://unpkg.com/leaflet@1.6.0/dist/leaflet.css" rel="stylesheet"/>')
        ui.add_head_html('<script src="https://unpkg.com/leaflet@1.6.0/dist/leaflet.js"></script>')

    def set_location(self, location: Tuple[float, float]) -> None:
        if False:
            return 10
        self.run_method('set_location', location[0], location[1])