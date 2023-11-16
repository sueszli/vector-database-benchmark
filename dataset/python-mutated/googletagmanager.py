from pylons import app_globals as g
from pylons import tmpl_context as c
from pylons import request
from r2.controllers.reddit_base import MinimalController
from r2.lib.pages import GoogleTagManagerJail, GoogleTagManager
from r2.lib.validator import validate, VGTMContainerId

class GoogleTagManagerController(MinimalController):

    def pre(self):
        if False:
            for i in range(10):
                print('nop')
        if request.host != g.media_domain:
            self.abort404()
        MinimalController.pre(self)
        c.allow_framing = True

    @validate(container_id=VGTMContainerId('id'))
    def GET_jail(self, container_id):
        if False:
            i = 10
            return i + 15
        return GoogleTagManagerJail(container_id=container_id).render()

    @validate(container_id=VGTMContainerId('id'))
    def GET_gtm(self, container_id):
        if False:
            while True:
                i = 10
        return GoogleTagManager(container_id=container_id).render()