from pylons import request
from pylons import app_globals as g
from reddit_base import RedditController
from r2.lib.pages import AdminPage, AdminAwards
from r2.lib.pages import AdminAwardGive, AdminAwardWinners
from r2.lib.validator import *

class AwardsController(RedditController):

    @validate(VAdmin())
    def GET_index(self):
        if False:
            print('Hello World!')
        res = AdminPage(content=AdminAwards(), title='awards').render()
        return res

    @validate(VAdmin(), award=VAwardByCodename('awardcn'), recipient=nop('recipient'), desc=nop('desc'), url=nop('url'), hours=nop('hours'))
    def GET_give(self, award, recipient, desc, url, hours):
        if False:
            print('Hello World!')
        if award is None:
            abort(404, 'page not found')
        res = AdminPage(content=AdminAwardGive(award, recipient, desc, url, hours), title='give an award').render()
        return res

    @validate(VAdmin(), award=VAwardByCodename('awardcn'))
    def GET_winners(self, award):
        if False:
            return 10
        if award is None:
            abort(404, 'page not found')
        res = AdminPage(content=AdminAwardWinners(award), title='award winners').render()
        return res