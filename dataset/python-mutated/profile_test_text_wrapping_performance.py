import cProfile
import datetime
import shutil
from pathlib import Path
from borb.pdf import Paragraph
from tests.pdf.canvas.layout.text.paragraph.test_text_wrapping_performance import TestTextWrappingPerformance

def print_cache_information(x) -> None:
    if False:
        print('Hello World!')
    for (k, v) in x.__dict__.items():
        if '_cache_for_' in str(k) and '_hit_rate' in str(k):
            hit_ratio: float = v[0] / (v[0] + v[1])
            miss_ratio: float = 1.0 - hit_ratio
            print(f'{x.__name__}, {k}, hit: {hit_ratio}, miss: {miss_ratio}')
if __name__ == '__main__':
    root_dir: Path = Path(__file__).parent
    artifacts_dir: Path = root_dir / 'artifacts_profile_test_text_wrapping_performance'
    if not artifacts_dir.exists():
        artifacts_dir.mkdir()
    profiler_output_path: Path = artifacts_dir / ('borb_profiler_output_' + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '.prof')
    latest_profiler_output_path: Path = artifacts_dir / 'latest.prof'
    cProfile.run('TestTextWrappingPerformance().test_layout_odyssey()', profiler_output_path)
    shutil.copy(profiler_output_path, latest_profiler_output_path)
    print_cache_information(Paragraph)