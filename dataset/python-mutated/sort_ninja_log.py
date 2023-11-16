import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom
parser = argparse.ArgumentParser()
parser.add_argument('log_file', type=str, default='.ninja_log', help='.ninja_log file')
parser.add_argument('--fmt', type=str, default='csv', choices=['csv', 'xml', 'html'], help='output format (to stdout)')
parser.add_argument('--msg', type=str, default=None, help='optional text file to include at the top of the html output')
parser.add_argument('--cmp_log', type=str, default=None, help='optional baseline ninja_log to compare results')
args = parser.parse_args()
log_file = args.log_file
output_fmt = args.fmt
cmp_file = args.cmp_log

def build_log_map(log_file):
    if False:
        while True:
            i = 10
    entries = {}
    log_path = os.path.dirname(os.path.abspath(log_file))
    with open(log_file) as log:
        last = 0
        files = {}
        for line in log:
            entry = line.split()
            if len(entry) > 4:
                obj_file = entry[3]
                file_size = os.path.getsize(os.path.join(log_path, obj_file)) if os.path.exists(obj_file) else 0
                start = int(entry[0])
                end = int(entry[1])
                if end < last:
                    files = {}
                last = end
                files.setdefault(entry[4], (entry[3], start, end, file_size))
        for entry in files.values():
            entries[entry[0]] = (entry[1], entry[2], entry[3])
    return entries

def output_xml(entries, sorted_list, args):
    if False:
        for i in range(10):
            print('nop')
    root = ET.Element('testsuites')
    testsuite = ET.Element('testsuite', attrib={'name': 'build-time', 'tests': str(len(sorted_list)), 'failures': str(0), 'errors': str(0)})
    root.append(testsuite)
    for name in sorted_list:
        entry = entries[name]
        build_time = float(entry[1] - entry[0]) / 1000
        item = ET.Element('testcase', attrib={'classname': 'BuildTime', 'name': name, 'time': str(build_time)})
        testsuite.append(item)
    tree = ET.ElementTree(root)
    xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent='   ')
    print(xmlstr)

def time_to_width(value, end):
    if False:
        print('Hello World!')
    r = float(value) / float(end) * 1000.0
    return int(r)

def assign_entries_to_threads(entries):
    if False:
        print('Hello World!')
    sorted_keys = sorted(list(entries.keys()), key=lambda k: entries[k][1], reverse=True)
    results = {}
    threads = []
    for name in sorted_keys:
        entry = entries[name]
        tid = -1
        for t in range(len(threads)):
            if threads[t] >= entry[1]:
                threads[t] = entry[0]
                tid = t
                break
        if tid < 0:
            threads.append(entry[0])
            tid = len(threads) - 1
        if tid not in results.keys():
            results[tid] = []
        results[tid].append(name)
    end_time = entries[sorted_keys[0]][1]
    return (results, end_time)

def format_build_time(input_time):
    if False:
        for i in range(10):
            print('nop')
    build_time = abs(input_time)
    build_time_str = str(build_time) + ' ms'
    if build_time > 120000:
        minutes = int(build_time / 60000)
        seconds = int((build_time / 60000 - minutes) * 60)
        build_time_str = '{:d}:{:02d} min'.format(minutes, seconds)
    elif build_time > 1000:
        build_time_str = '{:.3f} s'.format(build_time / 1000)
    if input_time < 0:
        build_time_str = '-' + build_time_str
    return build_time_str

def format_file_size(input_size):
    if False:
        i = 10
        return i + 15
    file_size = abs(input_size)
    file_size_str = ''
    if file_size > 1000000:
        file_size_str = '{:.3f} MB'.format(file_size / 1000000)
    elif file_size > 1000:
        file_size_str = '{:.3f} KB'.format(file_size / 1000)
    elif file_size > 0:
        file_size_str = str(file_size) + ' bytes'
    if input_size < 0:
        file_size_str = '-' + file_size_str
    return file_size_str

def output_html(entries, sorted_list, cmp_entries, args):
    if False:
        return 10
    print('<html><head><title>Build Metrics Report</title>')
    print('</head><body>')
    if args.msg is not None:
        msg_file = Path(args.msg)
        if msg_file.is_file():
            msg = msg_file.read_text()
            print('<p>', msg, '</p>')
    (threads, end_time) = assign_entries_to_threads(entries)
    summary = {'red': 0, 'yellow': 0, 'green': 0, 'white': 0}
    red = "bgcolor='#FFBBD0'"
    yellow = "bgcolor='#FFFF80'"
    green = "bgcolor='#AAFFBD'"
    white = "bgcolor='#FFFFFF'"
    print("<table id='chart' width='1000px' bgcolor='#BBBBBB'>")
    for tid in range(len(threads)):
        names = threads[tid]
        names = sorted(names, key=lambda k: entries[k][0])
        last_entry = entries[names[len(names) - 1]]
        last_time = time_to_width(last_entry[1], end_time)
        print("<tr><td><table width='", last_time, "px' border='0' cellspacing='1' cellpadding='0'><tr>", sep='')
        prev_end = 0
        for name in names:
            entry = entries[name]
            start = entry[0]
            end = entry[1]
            if prev_end > 0 and start > prev_end:
                size = time_to_width(start - prev_end, end_time)
                print("<td width='", size, "px'></td>")
            prev_end = end + int(end_time / 500)
            build_time = end - start
            build_time_str = format_build_time(build_time)
            color = white
            if build_time > 300000:
                color = red
                summary['red'] += 1
            elif build_time > 120000:
                color = yellow
                summary['yellow'] += 1
            elif build_time > 1000:
                color = green
                summary['green'] += 1
            else:
                summary['white'] += 1
            size = max(time_to_width(build_time, end_time), 2)
            print("<td height='20px' width='", size, "px' ", sep='', end='')
            print(color, "title='", end='')
            print(name, '\n', build_time_str, "' ", sep='', end='')
            print("align='center' nowrap>", end='')
            print("<font size='-2' face='courier'>", end='')
            file_name = os.path.basename(name)
            if len(file_name) + 3 > size / 7:
                abbr_size = int(size / 7) - 3
                if abbr_size > 1:
                    print(file_name[:abbr_size], '...', sep='', end='')
            else:
                print(file_name, end='')
            print('</font></td>')
            entries[name] = (build_time, color, entry[2])
        print("<td width='*'></td></tr></table></td></tr>")
    print('</table><br/>')
    print("<table id='detail' bgcolor='#EEEEEE'>")
    print('<tr><th>File</th>', '<th>Compile time</th>', '<th>Size</th>', sep='')
    if cmp_entries:
        print('<th>t-cmp</th>', sep='')
    print('</tr>')
    for name in sorted_list:
        entry = entries[name]
        build_time = entry[0]
        color = entry[1]
        file_size = entry[2]
        build_time_str = format_build_time(build_time)
        file_size_str = format_file_size(file_size)
        print('<tr ', color, '><td>', name, '</td>', sep='', end='')
        print("<td align='right'>", build_time_str, '</td>', sep='', end='')
        print("<td align='right'>", file_size_str, '</td>', sep='', end='')
        cmp_entry = cmp_entries[name] if cmp_entries and name in cmp_entries else None
        if cmp_entry:
            diff_time = build_time - (cmp_entry[1] - cmp_entry[0])
            diff_time_str = format_build_time(diff_time)
            diff_color = white
            diff_percent = int(diff_time / build_time * 100)
            if build_time > 60000:
                if diff_percent > 20:
                    diff_color = red
                    diff_time_str = '<b>' + diff_time_str + '</b>'
                elif diff_percent < -20:
                    diff_color = green
                    diff_time_str = '<b>' + diff_time_str + '</b>'
                elif diff_percent > 0:
                    diff_color = yellow
            print("<td align='right' ", diff_color, '>', diff_time_str, '</td>', sep='', end='')
        print('</tr>')
    print('</table><br/>')
    print("<table id='legend' border='2' bgcolor='#EEEEEE'>")
    print('<tr><td', red, '>time &gt; 5 minutes</td>')
    print("<td align='right'>", summary['red'], '</td></tr>')
    print('<tr><td', yellow, '>2 minutes &lt; time &lt; 5 minutes</td>')
    print("<td align='right'>", summary['yellow'], '</td></tr>')
    print('<tr><td', green, '>1 second &lt; time &lt; 2 minutes</td>')
    print("<td align='right'>", summary['green'], '</td></tr>')
    print('<tr><td', white, '>time &lt; 1 second</td>')
    print("<td align='right'>", summary['white'], '</td></tr>')
    print('</table>')
    if cmp_entries:
        print("<table id='legend' border='2' bgcolor='#EEEEEE'>")
        print('<tr><td', red, '>time increase &gt; 20%</td></tr>')
        print('<tr><td', yellow, '>time increase &gt; 0</td></tr>')
        print('<tr><td', green, '>time decrease &gt; 20%</td></tr>')
        print('<tr><td', white, '>time change &lt; 20%% or build time &lt; 1 minute</td></tr>')
        print('</table>')
    print('</body></html>')

def output_csv(entries, sorted_list, cmp_entries, args):
    if False:
        return 10
    print('time,size,file', end='')
    if cmp_entries:
        print(',diff', end='')
    print()
    for name in sorted_list:
        entry = entries[name]
        build_time = entry[1] - entry[0]
        file_size = entry[2]
        cmp_entry = cmp_entries[name] if cmp_entries and name in cmp_entries else None
        print(build_time, file_size, name, sep=',', end='')
        if cmp_entry:
            diff_time = build_time - (cmp_entry[1] - cmp_entry[0])
            print(',', diff_time, sep='', end='')
        print()
entries = build_log_map(log_file)
if len(entries) == 0:
    print('Could not parse', log_file)
    exit()
sorted_list = sorted(list(entries.keys()), key=lambda k: entries[k][1] - entries[k][0], reverse=True)
cmp_entries = build_log_map(cmp_file) if cmp_file else None
if output_fmt == 'xml':
    output_xml(entries, sorted_list, args)
elif output_fmt == 'html':
    output_html(entries, sorted_list, cmp_entries, args)
else:
    output_csv(entries, sorted_list, cmp_entries, args)