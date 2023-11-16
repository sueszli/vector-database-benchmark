"""To test the cases of importing NAS without certain DL libraries installed."""
import argparse
import subprocess
import sys
import pytest
masked_packages = ['torch', 'torch_none', 'tensorflow', 'tianshou']

def import_related(mask_out):
    if False:
        return 10
    import nni
    nni.set_default_framework(mask_out)
    import nni.nas
    import nni.nas.evaluator
    import nni.nas.hub
    import nni.nas.strategy
    import nni.nas.experiment

def import_rl_strategy_without_tianshou():
    if False:
        print('Hello World!')
    from nni.nas.strategy import PolicyBasedRL
    with pytest.raises(ImportError, match='tianshou'):
        PolicyBasedRL()

def main():
    if False:
        while True:
            i = 10
    parser = argparse.ArgumentParser()
    parser.add_argument('masked', choices=masked_packages)
    args = parser.parse_args()
    if args.masked == 'torch':
        sys.modules['torch'] = None
        import_related('tensorflow')
    elif args.masked == 'torch_none':
        sys.modules['torch'] = None
        import_related('none')
    elif args.masked == 'tensorflow':
        sys.modules['tensorflow'] = None
        import_related('pytorch')
    elif args.masked == 'tianshou':
        sys.modules['tianshou'] = None
        import_rl_strategy_without_tianshou()
    else:
        raise ValueError(f'Unknown masked package: {args.masked}')

@pytest.mark.parametrize('framework', masked_packages)
def test_import_without_framework(framework):
    if False:
        i = 10
        return i + 15
    subprocess.run([sys.executable, __file__, framework], check=True)
if __name__ == '__main__':
    main()