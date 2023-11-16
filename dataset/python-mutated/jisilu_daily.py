from datahub.jisilu import Jisilu
import fire

def run(remote='qq'):
    if False:
        i = 10
        return i + 15
    obj = Jisilu(remote=remote)
    obj.daily_update()
if __name__ == '__main__':
    fire.Fire(run)