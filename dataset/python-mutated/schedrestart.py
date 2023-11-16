from qqbot import QQBotSched as qqbotsched
from qqbot.utf8logger import INFO

class g(object):
    pass

def onPlug(bot):
    if False:
        i = 10
        return i + 15
    g.t = bot.conf.pluginsConf.get(__name__, '8:00')
    (g.hour, g.minute) = g.t.split(':')

    @qqbotsched(hour=g.hour, minute=g.minute)
    def schedRestart(_bot):
        if False:
            i = 10
            return i + 15
        _bot.FreshRestart()
    INFO('已创建计划任务：每天 %s 重启（需要手工扫码）', g.t)

def onUnplug(bot):
    if False:
        i = 10
        return i + 15
    INFO('已删除计划任务：每天 %s 重启（需要手工扫码）', g.t)