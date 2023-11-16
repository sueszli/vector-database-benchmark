import fire
import sys
sys.path.append('..')
from jsl_monitor import ReachTargetJSL
from realtime_monitor_ts import ReachTarget

def main(monitor_type='jsl'):
    if False:
        print('Hello World!')
    if monitor_type == 'jsl':
        obj = ReachTargetJSL()
    else:
        obj = ReachTarget()
    obj.monitor()
if __name__ == '__main__':
    fire.Fire(main)