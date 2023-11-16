from __future__ import annotations
import logging
log = logging.getLogger(__name__)

def init_robots(app):
    if False:
        for i in range(10):
            print('nop')
    '\n    Add X-Robots-Tag header.\n\n    Use it to avoid search engines indexing airflow. This mitigates some of the risk\n    associated with exposing Airflow to the public internet, however it does not\n    address the real security risks associated with such a deployment.\n\n    See also: https://developers.google.com/search/docs/advanced/robots/robots_meta_tag#xrobotstag\n    '

    def apply_robot_tag(response):
        if False:
            for i in range(10):
                print('nop')
        response.headers['X-Robots-Tag'] = 'noindex, nofollow'
        return response
    app.after_request(apply_robot_tag)