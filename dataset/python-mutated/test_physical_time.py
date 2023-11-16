__config__ = {'base': {'start_date': '2015-01-01', 'end_date': '2015-12-31', 'frequency': '1d', 'accounts': {'future': 1000000}}, 'extra': {'log_level': 'error'}}

def test_physical_time():
    if False:
        for i in range(10):
            print('nop')
    '\n    测试 physical_time 的使用\n    '
    from rqalpha.mod.rqalpha_mod_sys_scheduler.scheduler import physical_time

    def _day(context, bar_dict):
        if False:
            i = 10
            return i + 15
        context.counter += 1

    def init(context):
        if False:
            return 10
        context.counter = 0
        scheduler.run_daily(_day, time_rule=physical_time(hour=9, minute=31))
        context.days = 0

    def handle_bar(context, bar_dict):
        if False:
            while True:
                i = 10
        context.days += 1
        assert context.counter == context.days
    return locals()