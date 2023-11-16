import os
import re
path = '_build/default/tests'
print('Coverage statistics for files under matching/')

def report_summary_stat() -> str:
    if False:
        for i in range(10):
            print('nop')
    stat = os.popen('bisect-ppx-report summary').read()
    patt = re.compile('Coverage:\\s+\\d+/\\d+\\s+\\((\\d+\\.\\d*)%\\)')
    mobj = patt.match(stat)
    if mobj is not None:
        return mobj.group(1)
    raise Exception('')

def report_summary_for_file_stat(file: str) -> str:
    if False:
        print('Hello World!')
    stat = os.popen('bisect-ppx-report summary --per-file').readlines()
    patt = re.compile(f'\\s*(\\d+.\\d*)\\s+%\\s+\\d+/\\d+\\s+{file}')
    for line in stat:
        mobj = patt.match(line)
        if mobj is not None:
            return mobj.group(1)
    raise Exception('')
metrics_URL = 'https://dashboard.semgrep.dev/api/metric'

def add_metric(category: str, value: str) -> None:
    if False:
        i = 10
        return i + 15
    path = f'semgrep.core.{category}.coverage.matching.pct'
    cmd = f"curl -X POST {metrics_URL}/{path} -d '{value}' 2> /dev/null"
    out = os.popen(cmd).read()
    patt = re.compile('.*successfully recorded')
    if patt.match(out) is None:
        raise Exception(f'Could not push the metric: {out}')
os.system('dune runtest --instrument-with bisect_ppx --force 2> /dev/null')
global_stat = report_summary_stat()
print(f'Aggregated coverage for all languages: {global_stat}%')
add_metric('all', global_stat)
languages = ['Python', 'Javascript', 'Typescript', 'JSON', 'Java', 'C', 'Go', 'OCaml', 'Ruby', 'PHP']
for lang in languages:
    os.system(f'rm -f {path}/*.coverage')
    os.system(f'cd {path};./test.exe {lang} > /dev/null')
    lang_stat = report_summary_stat()
    print(f'Coverage stat for {lang}: {lang_stat}%')
    add_metric(lang.lower(), lang_stat)
subsystems = [('eval', 'matching/Eval_generic.ml')]
for (test, file) in subsystems:
    os.system(f'rm -f {path}/*.coverage')
    os.system(f'cd {path};./test.exe {test} > /dev/null')
    stat = report_summary_for_file_stat(file)
    print(f'Coverage stat for {test} in file {file}: {stat}%')