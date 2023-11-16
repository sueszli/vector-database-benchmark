import os
import random
import sys
import tempfile
from collections import OrderedDict, defaultdict
from operator import itemgetter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from scalene import sparkline
from scalene.scalene_json import ScaleneJSON
from scalene.scalene_leak_analysis import ScaleneLeakAnalysis
from scalene.scalene_statistics import Filename, LineNumber, ScaleneStatistics
from scalene.syntaxline import SyntaxLine

class ScaleneOutput:
    max_sparkline_len_file = 27
    max_sparkline_len_line = 9
    highlight_percentage = 33
    highlight_color = 'bold red'
    memory_color = 'dark_green'
    gpu_color = 'yellow4'
    copy_volume_color = 'yellow4'

    def __init__(self) -> None:
        if False:
            return 10
        self.output_file = ''
        self.html = False
        self.gpu = False

    def output_top_memory(self, title: str, console: Console, mallocs: Dict[LineNumber, float]) -> None:
        if False:
            i = 10
            return i + 15
        if mallocs:
            printed_header = False
            number = 1
            print_top_mallocs_count = 5
            print_top_mallocs_threshold_mb = 1
            for (malloc_lineno, value) in mallocs.items():
                if value <= print_top_mallocs_threshold_mb:
                    break
                if number > print_top_mallocs_count:
                    break
                if not printed_header:
                    console.print(title)
                    printed_header = True
                output_str = f'({str(number)}) {malloc_lineno:5.0f}: {mallocs[malloc_lineno]:5.0f} MB'
                console.print(Markdown(output_str, style=self.memory_color))
                number += 1

    def output_profile_line(self, json: ScaleneJSON, fname: Filename, line_no: LineNumber, line: SyntaxLine, console: Console, tbl: Table, stats: ScaleneStatistics, profile_this_code: Callable[[Filename, LineNumber], bool], force_print: bool=False, suppress_lineno_print: bool=False, is_function_summary: bool=False, profile_memory: bool=False, reduced_profile: bool=False) -> bool:
        if False:
            while True:
                i = 10
        'Print at most one line of the profile (true == printed one).'
        obj = json.output_profile_line(fname=fname, fname_print=fname, line_no=line_no, line=line, stats=stats, profile_this_code=profile_this_code, force_print=force_print)
        if not obj:
            return False
        if -1 < obj['n_peak_mb'] < 1:
            obj['n_peak_mb'] = 0
        n_cpu_percent_c_str: str = '' if obj['n_cpu_percent_c'] < 1 else f"{obj['n_cpu_percent_c']:5.0f}%"
        n_gpu_percent_str: str = '' if obj['n_gpu_percent'] < 1 else f"{obj['n_gpu_percent']:3.0f}%"
        n_cpu_percent_python_str: str = '' if obj['n_cpu_percent_python'] < 1 else f"{obj['n_cpu_percent_python']:5.0f}%"
        n_growth_mem_str = ''
        if obj['n_peak_mb'] < 1024:
            n_growth_mem_str = '' if not obj['n_peak_mb'] and (not obj['n_usage_fraction']) else f"{obj['n_peak_mb']:5.0f}M"
        else:
            n_growth_mem_str = '' if not obj['n_peak_mb'] and (not obj['n_usage_fraction']) else f"{obj['n_peak_mb'] / 1024:5.2f}G"
        sys_str: str = '' if obj['n_sys_percent'] < 1 else f"{obj['n_sys_percent']:4.0f}%"
        if not is_function_summary:
            print_line_no = '' if suppress_lineno_print else str(line_no)
        else:
            print_line_no = '' if fname not in stats.firstline_map else str(stats.firstline_map[fname])
        if profile_memory:
            spark_str: str = ''
            samples = obj['memory_samples']
            if len(samples) > ScaleneOutput.max_sparkline_len_line:
                random_samples = sorted(random.sample(samples, ScaleneOutput.max_sparkline_len_line))
            else:
                random_samples = samples
            sparkline_samples = [random_samples[i][1] * obj['n_usage_fraction'] for i in range(len(random_samples))]
            if random_samples:
                (_, _, spark_str) = sparkline.generate(sparkline_samples, 0, stats.max_footprint)
            ncpps: Any = ''
            ncpcs: Any = ''
            nufs: Any = ''
            ngpus: Any = ''
            n_usage_fraction_str: str = '' if obj['n_usage_fraction'] < 0.01 else f"{100 * obj['n_usage_fraction']:4.0f}%"
            if obj['n_usage_fraction'] >= self.highlight_percentage or obj['n_cpu_percent_c'] + obj['n_cpu_percent_python'] + obj['n_gpu_percent'] >= self.highlight_percentage:
                ncpps = Text.assemble((n_cpu_percent_python_str, self.highlight_color))
                ncpcs = Text.assemble((n_cpu_percent_c_str, self.highlight_color))
                nufs = Text.assemble((spark_str + n_usage_fraction_str, self.highlight_color))
                ngpus = Text.assemble((n_gpu_percent_str, self.highlight_color))
            else:
                ncpps = n_cpu_percent_python_str
                ncpcs = n_cpu_percent_c_str
                ngpus = n_gpu_percent_str
                nufs = spark_str + n_usage_fraction_str
            if reduced_profile and (not ncpps + ncpcs + nufs + ngpus):
                return False
            n_python_fraction_str: str = '' if obj['n_python_fraction'] < 0.01 else f"{obj['n_python_fraction'] * 100:4.0f}%"
            n_copy_mb_s_str: str = '' if obj['n_copy_mb_s'] < 0.5 else f"{obj['n_copy_mb_s']:6.0f}"
            if self.gpu:
                tbl.add_row(print_line_no, ncpps, ncpcs, sys_str, ngpus, n_python_fraction_str, n_growth_mem_str, nufs, n_copy_mb_s_str, line)
            else:
                tbl.add_row(print_line_no, ncpps, ncpcs, sys_str, n_python_fraction_str, n_growth_mem_str, nufs, n_copy_mb_s_str, line)
        else:
            if obj['n_cpu_percent_c'] + obj['n_cpu_percent_python'] + obj['n_gpu_percent'] >= self.highlight_percentage:
                ncpps = Text.assemble((n_cpu_percent_python_str, self.highlight_color))
                ncpcs = Text.assemble((n_cpu_percent_c_str, self.highlight_color))
                ngpus = Text.assemble((n_gpu_percent_str, self.highlight_color))
            else:
                ncpps = n_cpu_percent_python_str
                ncpcs = n_cpu_percent_c_str
                ngpus = n_gpu_percent_str
            if reduced_profile and (not ncpps + ncpcs + ngpus):
                return False
            if self.gpu:
                tbl.add_row(print_line_no, ncpps, ncpcs, sys_str, ngpus, line)
            else:
                tbl.add_row(print_line_no, ncpps, ncpcs, sys_str, line)
        return True

    def output_profiles(self, column_width: int, stats: ScaleneStatistics, pid: int, profile_this_code: Callable[[Filename, LineNumber], bool], python_alias_dir: Path, program_path: Path, program_args: Optional[List[str]], profile_memory: bool=True, reduced_profile: bool=False) -> bool:
        if False:
            i = 10
            return i + 15
        'Write the profile out.'
        json = ScaleneJSON()
        json.gpu = self.gpu
        if not pid:
            stats.merge_stats(python_alias_dir)
        if not stats.total_cpu_samples and (not stats.total_memory_malloc_samples) and (not stats.total_memory_free_samples):
            return False
        all_instrumented_files: List[Filename] = list(set(list(stats.cpu_samples_python.keys()) + list(stats.cpu_samples_c.keys()) + list(stats.memory_free_samples.keys()) + list(stats.memory_malloc_samples.keys())))
        if not all_instrumented_files:
            return False
        mem_usage_line: Union[Text, str] = ''
        growth_rate = 0.0
        if profile_memory:
            samples = stats.memory_footprint_samples
            if len(samples) > 0:
                if len(samples) > ScaleneOutput.max_sparkline_len_file:
                    random_samples = sorted(random.sample(samples, ScaleneOutput.max_sparkline_len_file))
                else:
                    random_samples = samples
                sparkline_samples = [item[1] for item in random_samples]
                (_, _, spark_str) = sparkline.generate(sparkline_samples[:ScaleneOutput.max_sparkline_len_file], 0, stats.max_footprint)
                if stats.allocation_velocity[1] > 0:
                    growth_rate = 100.0 * stats.allocation_velocity[0] / stats.allocation_velocity[1]
                mem_usage_line = Text.assemble('Memory usage: ', (spark_str, self.memory_color), f' (max: {ScaleneJSON.memory_consumed_str(stats.max_footprint)}, growth rate: {growth_rate:3.0f}%)\n')
        null = tempfile.TemporaryFile(mode='w+')
        console = Console(width=column_width, record=True, force_terminal=True, file=null, force_jupyter=False)
        report_files: List[Filename] = []
        for fname in sorted(all_instrumented_files, key=lambda f: (-stats.cpu_samples[f], f)):
            fname = Filename(fname)
            try:
                percent_cpu_time = 100 * stats.cpu_samples[fname] / stats.total_cpu_samples
            except ZeroDivisionError:
                percent_cpu_time = 0
            if stats.malloc_samples[fname] < ScaleneJSON.malloc_threshold and percent_cpu_time < ScaleneJSON.cpu_percent_threshold:
                continue
            report_files.append(fname)
        if pid:
            stats.output_stats(pid, python_alias_dir)
            return True
        if not report_files:
            return False
        for fname in report_files:
            fname_print = fname
            import re
            if (result := re.match('_ipython-input-([0-9]+)-.*', fname_print)):
                fname_print = Filename(f'[{result.group(1)}]')
            percent_cpu_time = 100 * stats.cpu_samples[fname] / stats.total_cpu_samples if stats.total_cpu_samples else 0
            new_title = mem_usage_line + f'{fname_print}: % of time = {percent_cpu_time:6.2f}% ({ScaleneJSON.time_consumed_str(percent_cpu_time / 100.0 * stats.elapsed_time * 1000.0)}) out of {ScaleneJSON.time_consumed_str(stats.elapsed_time * 1000.0)}.'
            mem_usage_line = ''
            tbl = Table(box=box.MINIMAL_HEAVY_HEAD, title=new_title, collapse_padding=True, width=column_width - 1)
            tbl.add_column(Markdown('Line', style='dim'), style='dim', justify='right', no_wrap=True, width=4)
            tbl.add_column(Markdown('Time  ' + '\n' + '_Python_', style='blue'), style='blue', no_wrap=True, width=6)
            tbl.add_column(Markdown('––––––  \n_native_', style='blue'), style='blue', no_wrap=True, width=6)
            tbl.add_column(Markdown('––––––  \n_system_', style='blue'), style='blue', no_wrap=True, width=6)
            if self.gpu:
                tbl.add_column(Markdown('––––––  \n_GPU_', style=self.gpu_color), style=self.gpu_color, no_wrap=True, width=6)
            other_columns_width = 0
            if profile_memory:
                tbl.add_column(Markdown('Memory  \n_Python_', style=self.memory_color), style=self.memory_color, no_wrap=True, width=7)
                tbl.add_column(Markdown('––––––  \n_peak_', style=self.memory_color), style=self.memory_color, no_wrap=True, width=6)
                tbl.add_column(Markdown('–––––––––––  \n_timeline_/%', style=self.memory_color), style=self.memory_color, no_wrap=True, width=15)
                tbl.add_column(Markdown('Copy  \n_(MB/s)_', style=self.copy_volume_color), style=self.copy_volume_color, no_wrap=True, width=6)
                other_columns_width = 75 + (6 if self.gpu else 0)
            else:
                other_columns_width = 37 + (5 if self.gpu else 0)
            tbl.add_column('\n' + fname_print, width=column_width - other_columns_width, no_wrap=True)
            if fname == '<BOGUS>':
                continue
            if not fname:
                continue
            full_fname = os.path.normpath(os.path.join(program_path, fname))
            try:
                with open(full_fname, 'r') as source_file:
                    code_lines = source_file.read()
            except (FileNotFoundError, OSError):
                continue
            did_print = True
            syntax_highlighted = Syntax(code_lines, 'python', theme='default' if self.html else 'vim', line_numbers=False, code_width=None)
            capture_console = Console(width=column_width - other_columns_width, force_terminal=True)
            formatted_lines = [SyntaxLine(segments) for segments in capture_console.render_lines(syntax_highlighted)]
            for (line_no, line) in enumerate(formatted_lines, start=1):
                old_did_print = did_print
                did_print = self.output_profile_line(json=json, fname=fname, line_no=LineNumber(line_no), line=line, console=console, tbl=tbl, stats=stats, profile_this_code=profile_this_code, profile_memory=profile_memory, force_print=False, suppress_lineno_print=False, is_function_summary=False, reduced_profile=reduced_profile)
                if old_did_print and (not did_print):
                    tbl.add_row('...')
                old_did_print = did_print
            fn_stats = stats.build_function_stats(fname)
            print_fn_summary = False
            all_samples = set()
            all_samples |= set(fn_stats.cpu_samples_python.keys())
            all_samples |= set(fn_stats.cpu_samples_c.keys())
            all_samples |= set(fn_stats.memory_malloc_samples.keys())
            all_samples |= set(fn_stats.memory_free_samples.keys())
            for fn_name in all_samples:
                if fn_name == fname:
                    continue
                print_fn_summary = True
                break
            if print_fn_summary:
                try:
                    tbl.add_row(None, end_section=True)
                except TypeError:
                    tbl.add_row(None)
                txt = Text.assemble(f'function summary for {fname_print}', style='bold italic')
                if profile_memory:
                    if self.gpu:
                        tbl.add_row('', '', '', '', '', '', '', '', '', txt)
                    else:
                        tbl.add_row('', '', '', '', '', '', '', '', txt)
                elif self.gpu:
                    tbl.add_row('', '', '', '', '', txt)
                else:
                    tbl.add_row('', '', '', '', txt)
                for fn_name in sorted(fn_stats.cpu_samples_python, key=lambda k: stats.firstline_map[k]):
                    if fn_name == fname:
                        continue
                    syntax_highlighted = Syntax(fn_name, 'python', theme='default' if self.html else 'vim', line_numbers=False, code_width=None)
                    self.output_profile_line(json=json, fname=fn_name, line_no=LineNumber(1), line=syntax_highlighted, console=console, tbl=tbl, stats=fn_stats, profile_this_code=profile_this_code, profile_memory=profile_memory, force_print=True, suppress_lineno_print=True, is_function_summary=True, reduced_profile=reduced_profile)
            console.print(tbl)
            avg_mallocs: Dict[LineNumber, float] = defaultdict(float)
            for line_no in stats.bytei_map[fname]:
                n_malloc_mb = stats.memory_aggregate_footprint[fname][line_no]
                if (count := stats.memory_malloc_count[fname][line_no]):
                    avg_mallocs[line_no] = n_malloc_mb / count
                else:
                    avg_mallocs[line_no] = n_malloc_mb
            avg_mallocs = OrderedDict(sorted(avg_mallocs.items(), key=itemgetter(1), reverse=True))
            peak_mallocs: Dict[LineNumber, float] = defaultdict(float)
            for line_no in stats.bytei_map[fname]:
                peak_mallocs[line_no] = stats.memory_max_footprint[fname][line_no]
            peak_mallocs = OrderedDict(sorted(peak_mallocs.items(), key=itemgetter(1), reverse=True))
            self.output_top_memory('Top AVERAGE memory consumption, by line:', console, avg_mallocs)
            self.output_top_memory('Top PEAK memory consumption, by line:', console, peak_mallocs)
            leaks = ScaleneLeakAnalysis.compute_leaks(growth_rate, stats, avg_mallocs, fname)
            if len(leaks) > 0:
                for leak in sorted(leaks, key=itemgetter(1), reverse=True):
                    output_str = f'Possible memory leak identified at line {str(leak[0])} (estimated likelihood: {leak[1] * 100:3.0f}%, velocity: {leak[2] / stats.elapsed_time:3.0f} MB/s)'
                    console.print(output_str)
        if self.html:
            md = Markdown('generated by the [scalene](https://github.com/plasma-umass/scalene) profiler')
            console.print(md)
            if not self.output_file:
                self.output_file = '/dev/stdout'
            console.save_html(self.output_file, clear=False)
        elif self.output_file:
            console.save_text(self.output_file, styles=False, clear=False)
        else:
            sys.stdout.write(console.export_text(styles=True))
        return True