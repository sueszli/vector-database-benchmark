SH_FLAG = True
SZ_FLAG = False

def reverse_repurchase(context):
    if False:
        for i in range(10):
            print('nop')
    cash = context.portfolio.cash
    if SH_FLAG:
        amount = int(cash / 100 / 1000) * 1000
        order('204001.SS', -amount)
    if SZ_FLAG:
        amount = int(cash / 100 / 10) * 10
        order('131810.SZ', -amount)

def initialize(context):
    if False:
        i = 10
        return i + 15
    '\n    初始化\n    '
    run_daily(context, reverse_repurchase, time='14:58')