"""
增加/修改 qq 命令的方法：在 qqbot.termbot 模块的 cmdFuncs 字典中 增加/修改 自定义函数

本插件加载后： 运行 qq mycommand xxx ，返回 “['xxx', False]”
              请求 http://127.0.0.1:8188/mycommand/xxx ，返回 “{"result": "[['xxx'], True]",  "err": null}”
"""

def cmd_mycommand(bot, args, http=False):
    if False:
        for i in range(10):
            print('nop')
    result = str([args, http])
    err = None
    return (result, err)

def onPlug(bot):
    if False:
        while True:
            i = 10
    from qqbot.termbot import cmdFuncs
    cmdFuncs['mycommand'] = cmd_mycommand