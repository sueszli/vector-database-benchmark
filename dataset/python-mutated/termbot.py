from qqbot.utf8logger import ERROR
from qqbot.mainloop import Put
from qqbot.common import Unquote, STR2BYTES, JsonDumps, BYTES2STR
(cmdFuncs, usage) = ({}, {})

class TermBot(object):

    def onTermCommand(bot, command):
        if False:
            return 10
        command = BYTES2STR(command)
        if command.startswith('GET /'):
            http = True
            end = command.find('\r\n')
            if end == -1 or not command[:end - 3].endswith(' HTTP/'):
                argv = []
            else:
                url = command[5:end - 9].rstrip('/')
                if url == 'favicon.ico':
                    return b''
                argv = [Unquote(x) for x in url.split('/')]
        else:
            http = False
            argv = command.strip().split(None, 3)
        if argv and argv[0] in cmdFuncs:
            try:
                (result, err) = cmdFuncs[argv[0]](bot, argv[1:], http)
            except Exception as e:
                (result, err) = (None, '运行命令过程中出错：' + str(type(e)) + str(e))
                ERROR(err, exc_info=True)
        else:
            (result, err) = (None, 'QQBot 命令格式错误')
        if http:
            rep = {'result': result, 'err': err}
            rep = STR2BYTES(JsonDumps(rep, ensure_ascii=False, indent=4))
            rep = b'HTTP/1.1 200 OK\r\n' + b'Connection: close\r\n' + b'Content-Length: ' + STR2BYTES(str(len(rep))) + b'\r\n' + b'Content-Type: text/plain;charset=utf-8\r\n\r\n' + rep
        else:
            rep = STR2BYTES(str(err or result)) + b'\r\n'
        return rep

def cmd_help(bot, args, http=False):
    if False:
        return 10
    '1 help'
    if len(args) == 0:
        return (usage['term'], None)
    else:
        return (None, 'QQBot 命令格式错误')

def cmd_stop(bot, args, http=False):
    if False:
        for i in range(10):
            print('nop')
    '1 stop'
    if len(args) == 0:
        Put(bot.Stop)
        return ('QQBot已停止', None)
    else:
        return (None, 'QQBot 命令格式错误')

def cmd_restart(bot, args, http=False):
    if False:
        while True:
            i = 10
    '1 restart'
    if len(args) == 0:
        Put(bot.Restart)
        return ('QQBot已重启（自动登录）', None)
    else:
        return (None, 'QQBot 命令格式错误')

def cmd_fresh_restart(bot, args, http=False):
    if False:
        for i in range(10):
            print('nop')
    '1 fresh-restart'
    if len(args) == 0:
        Put(bot.FreshRestart)
        return ('QQBot已重启（手工登录）', None)
    else:
        return (None, 'QQBot 命令格式错误')

def cmd_list(bot, args, http=False):
    if False:
        for i in range(10):
            print('nop')
    '2 list buddy|group|discuss [qq|name|key=val]\n       2 list group-member|discuss-member oqq|oname|okey=oval [qq|name|key=val]'
    if len(args) in (1, 2) and args[0] in ('buddy', 'group', 'discuss'):
        if not http:
            return (bot.StrOfList(*args), None)
        else:
            return bot.ObjOfList(*args)
    elif len(args) in (2, 3) and args[1] and (args[0] in ('group-member', 'discuss-member')):
        if not http:
            return (bot.StrOfList(*args), None)
        else:
            return bot.ObjOfList(*args)
    else:
        return (None, 'QQBot 命令格式错误')

def cmd_update(bot, args, http=False):
    if False:
        return 10
    '2 update buddy|group|discuss\n       2 update group-member|discuss-member oqq|oname|okey=oval'
    if len(args) == 1 and args[0] in ('buddy', 'group', 'discuss'):
        return (bot.Update(args[0]), None)
    elif len(args) == 2 and args[1] and (args[0] in ('group-member', 'discuss-member')):
        cl = bot.List(args[0][:-7], args[1])
        if cl is None:
            return (None, 'QQBot 在向 QQ 服务器请求数据获取联系人资料的过程中发生错误')
        elif not cl:
            return (None, '%s-%s 不存在' % (args[0], args[1]))
        else:
            return ([bot.Update(c) for c in cl], None)
    else:
        return (None, 'QQBot 命令格式错误')

def cmd_send(bot, args, http=False):
    if False:
        print('Hello World!')
    '3 send buddy|group|discuss qq|name|key=val message'
    if len(args) == 3 and args[0] in ('buddy', 'group', 'discuss'):
        cl = bot.List(args[0], args[1])
        if cl is None:
            return (None, 'QQBot 在向 QQ 服务器请求数据获取联系人资料的过程中发生错误')
        elif not cl:
            return (None, '%s-%s 不存在' % (args[0], args[1]))
        else:
            msg = args[2].replace('\\n', '\n').replace('\\t', '\t')
            result = [bot.SendTo(c, msg) for c in cl]
            if not http:
                result = '\n'.join(result)
            return (result, None)
    else:
        return (None, 'QQBot 命令格式错误')

def group_operation(bot, ginfo, minfos, func, exArgs, http):
    if False:
        for i in range(10):
            print('nop')
    gl = bot.List('group', ginfo)
    if gl is None:
        return (None, '错误：向 QQ 服务器请求群列表失败')
    elif not gl:
        return (None, '错误：群%s 不存在' % ginfo)
    result = []
    for g in gl:
        (membsResult, membs) = ([], [])
        for minfo in minfos:
            ml = bot.List(g, minfo)
            if ml is None:
                membsResult.append('错误：向 QQ 服务器请求%s的成员列表失败' % g)
            elif not ml:
                membsResult.append('错误：%s[成员“%s”]不存在' % (g, minfo))
            else:
                membs.extend(ml)
        if membs:
            membsResult.extend(func(g, membs, *exArgs))
        if not http:
            result.append('\n'.join(membsResult))
        else:
            result.append({'group': g.__dict__, 'membs_result': membsResult})
    if not http:
        result = '\n\n'.join(result)
    return (result, None)

def cmd_group_kick(bot, args, http=False):
    if False:
        i = 10
        return i + 15
    '4 group-kick ginfo minfo1,minfo2,minfo3'
    if len(args) == 2:
        ginfo = args[0]
        minfos = args[1].split(',')
        return group_operation(bot, ginfo, minfos, bot.GroupKick, [], http)
    else:
        return (None, 'QQBot 命令格式错误')

def cmd_group_set_admin(bot, args, http=False):
    if False:
        for i in range(10):
            print('nop')
    '4 group-set-admin ginfo minfo1,minfo2,minfo3'
    if len(args) == 2:
        ginfo = args[0]
        minfos = args[1].split(',')
        return group_operation(bot, ginfo, minfos, bot.GroupSetAdmin, [True], http)
    else:
        return (None, 'QQBot 命令格式错误')

def cmd_group_unset_admin(bot, args, http=False):
    if False:
        for i in range(10):
            print('nop')
    '4 group-unset-admin ginfo minfo1,minfo2,minfo3'
    if len(args) == 2:
        ginfo = args[0]
        minfos = args[1].split(',')
        return group_operation(bot, ginfo, minfos, bot.GroupSetAdmin, [False], http)
    else:
        return (None, 'QQBot 命令格式错误')

def cmd_group_shut(bot, args, http=False):
    if False:
        i = 10
        return i + 15
    '4 group-shut ginfo minfo1,minfo2,minfo3 t'
    if len(args) in (2, 3):
        ginfo = args[0]
        minfos = args[1].split(',')
        if len(args) == 3 and args[2].isdigit() and (int(args[2]) > 60):
            t = int(args[2])
        else:
            t = 60
        return group_operation(bot, ginfo, minfos, bot.GroupShut, [t], http)
    else:
        return (None, 'QQBot 命令格式错误')

def cmd_group_set_card(bot, args, http=False):
    if False:
        for i in range(10):
            print('nop')
    '4 group-set-card ginfo minfo1,minfo2,minfo3 card'
    if len(args) == 3:
        ginfo = args[0]
        minfos = args[1].split(',')
        card = args[2]
        return group_operation(bot, ginfo, minfos, bot.GroupSetCard, [card], http)
    else:
        return (None, 'QQBot 命令格式错误')

def cmd_group_unset_card(bot, args, http=False):
    if False:
        i = 10
        return i + 15
    '4 group-unset-card ginfo minfo1,minfo2,minfo3'
    if len(args) == 2:
        ginfo = args[0]
        minfos = args[1].split(',')
        card = ''
        return group_operation(bot, ginfo, minfos, bot.GroupSetCard, [card], http)
    else:
        return (None, 'QQBot 命令格式错误')

def cmd_plug(bot, args, http=False):
    if False:
        for i in range(10):
            print('nop')
    '5 plug myplugin'
    if len(args) == 1:
        return (bot.Plug(args[0]), None)
    else:
        return (None, 'QQBot 命令格式错误')

def cmd_unplug(bot, args, http=False):
    if False:
        while True:
            i = 10
    '5 unplug myplugin'
    if len(args) == 1:
        return (bot.Unplug(args[0]), None)
    else:
        return (None, 'QQBot 命令格式错误')

def cmd_plugins(bot, args, http=False):
    if False:
        return 10
    '5 plugins'
    if len(args) == 0:
        if not http:
            return ('已加载插件：%s' % bot.Plugins(), None)
        else:
            return (bot.Plugins(), None)
    else:
        return (None, 'QQBot 命令格式错误')
for (name, attr) in dict(globals().items()).items():
    if name.startswith('cmd_'):
        cmdFuncs[name[4:].replace('_', '-')] = attr
usage['term'] = 'QQBot 命令：\n1） 帮助、停机和重启命令\n    qq help|stop|restart\n\n2） 联系人查询命令\n    qq list buddy|group|discuss [qq|name|key=val]\n    qq list group-member|discuss-member oqq|oname|okey=oval [qq|name|key=val]\n\n3） 联系人更新命令\n    qq update buddy|group|discuss\n    qq update group-member|discuss-member oqq|oname|okey=oval\n\n4） 消息发送命令\n    qq send buddy|group|discuss qq|name|key=val message\n\n5） 群管理命令： 设置/取消管理员 、 设置/删除群名片 、 群成员禁言 以及 踢除群成员\n    qq group-set-admin ginfo minfo1,minfo2,...\n    qq group-unset-admin ginfo minfo1,minfo2,...\n    qq group-set-card ginfo minfo1,minfo2,... card\n    qq group-unset-card ginfo minfo1,minfo2,...\n    qq group-shut ginfo minfo1,minfo2,... [t]\n    qq group-kick ginfo minfo1,minfo2,...\n\n6） 加载/卸载/显示插件\n    qq plug/unplug myplugin\n    qq plugins'