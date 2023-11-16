def test_load_args():
    if False:
        for i in range(10):
            print('nop')
    import subprocess
    import sys
    subprocess.call([sys.executable, 'load_args.py', '--output_dir', 'data/'], shell=True)