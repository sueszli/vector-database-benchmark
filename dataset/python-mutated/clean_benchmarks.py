import os
import shutil
from pathlib import Path
from typer import run

def main():
    if False:
        for i in range(10):
            print('nop')
    benchmarks = Path('benchmark')
    for benchmark in benchmarks.iterdir():
        if benchmark.is_dir():
            print(f'Cleaning {benchmark}')
            for path in benchmark.iterdir():
                if path.name in ['prompt', 'main_prompt']:
                    continue
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    os.remove(path)
if __name__ == '__main__':
    run(main)