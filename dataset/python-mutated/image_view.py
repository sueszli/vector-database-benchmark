from typing import Dict, Any
from annotation_gui_gcp.lib.views.web_view import WebView, distinct_colors

def point_color(point_id: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    hex_color = distinct_colors[divmod(hash(point_id), 19)[1]]
    return hex_color

class ImageView(WebView):

    def __init__(self, main_ui, web_app, route_prefix, image_keys, is_geo_reference):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(main_ui, web_app, route_prefix)
        self.main_ui = main_ui
        self.image_manager = main_ui.image_manager
        self.is_geo_reference = is_geo_reference
        self.image_list = image_keys
        self.app.add_url_rule(f'{route_prefix}/image/<int:max_sz>/<path:path>', f'{route_prefix}_image', view_func=self.get_image)

    def get_candidate_images(self):
        if False:
            while True:
                i = 10
        return self.image_list

    def get_image(self, path, max_sz):
        if False:
            for i in range(10):
                print('nop')
        return self.image_manager.get_image(path, max_sz)

    def add_remove_update_point_observation(self, image_id, point_coordinates=None, precision=None):
        if False:
            print('Hello World!')
        gcp_manager = self.main_ui.gcp_manager
        active_gcp = self.main_ui.curr_point
        if active_gcp is None:
            print('No point selected in the main UI. Doing nothing')
            return
        gcp_manager.remove_point_observation(active_gcp, image_id, remove_latlon=self.is_geo_reference)
        if point_coordinates is not None:
            assert precision is not None
            self.main_ui.gcp_manager.add_point_observation(active_gcp, image_id, point_coordinates, precision=precision, geo=None)
        self.main_ui.populate_gcp_list()

    def display_points(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def highlight_gcp_reprojection(self, *args, **kwargs):
        if False:
            print('Hello World!')
        pass

    def populate_image_list(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        pass

    def sync_to_client(self):
        if False:
            i = 10
            return i + 15
        '\n        Sends all the data required to initialize or sync the image view\n        '
        image_list = self.get_candidate_images()
        all_points_this_view = {image: self.main_ui.gcp_manager.get_visible_points_coords(image) for image in image_list}
        data = {'points': all_points_this_view, 'selected_point': self.main_ui.curr_point, 'colors': {point_id: point_color(point_id) for point_id in self.main_ui.gcp_manager.points}}
        self.send_sse_message(data)

    def process_client_message(self, data: Dict[str, Any]) -> None:
        if False:
            while True:
                i = 10
        command = data['event']
        if command not in ('add_or_update_point_observation', 'remove_point_observation'):
            raise ValueError(f'Unknown command {command}')
        if data['point_id'] != self.main_ui.curr_point:
            print(data['point_id'], self.main_ui.curr_point)
            print('Frontend sending an update for some other point. Ignoring')
            return
        if command == 'add_or_update_point_observation':
            self.add_remove_update_point_observation(image_id=data['image_id'], point_coordinates=data['xy'], precision=data['norm_precision'])
        else:
            self.add_remove_update_point_observation(image_id=data['image_id'], point_coordinates=None)