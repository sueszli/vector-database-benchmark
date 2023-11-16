import argparse
import os
from collections import defaultdict
html_template = '<html><head><style media="screen" type="text/css">\nimg{{\n  width:100%;\n  max-width:800px;\n}}\n</style>\n</head><body>\n{failed}\n{body}\n</body></html>\n'
subdir_template = '<h2>{subdir}</h2><table>\n<thead><td>name</td><td>actual</td><td>expected</td><td>diff</td></thead>\n{rows}\n</table>\n'
failed_template = '<h2>Only Failed</h2><table>\n<thead><td>name</td><td>actual</td><td>expected</td><td>diff</td></thead>\n{rows}\n</table>\n'
row_template = '<tr><td>{0}{1}</td><td>{2}</td><td><a href="{3}"><img src="{3}"></a></td><td>{4}</td></tr>'
linked_image_template = '<a href="{0}"><img src="{0}"></a>'

def run(show_browser=True):
    if False:
        while True:
            i = 10
    '\n    Build a website for visual comparison\n    '
    image_dir = 'result_images'
    _subdirs = (name for name in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, name)))
    failed_rows = []
    body_sections = []
    for subdir in sorted(_subdirs):
        if subdir == 'test_compare_images':
            continue
        pictures = defaultdict(dict)
        for file in os.listdir(os.path.join(image_dir, subdir)):
            if os.path.isdir(os.path.join(image_dir, subdir, file)):
                continue
            (fn, fext) = os.path.splitext(file)
            if fext != '.png':
                continue
            if '-failed-diff' in fn:
                pictures[fn[:-12]]['f'] = '/'.join((subdir, file))
            elif '-expected' in fn:
                pictures[fn[:-9]]['e'] = '/'.join((subdir, file))
            else:
                pictures[fn]['c'] = '/'.join((subdir, file))
        subdir_rows = []
        for (name, test) in sorted(pictures.items()):
            expected_image = test.get('e', '')
            actual_image = test.get('c', '')
            if 'f' in test:
                status = ' (failed)'
                failed = f'''<a href="{test['f']}">diff</a>'''
                current = linked_image_template.format(actual_image)
                failed_rows.append(row_template.format(name, '', current, expected_image, failed))
            elif 'c' not in test:
                status = ' (failed)'
                failed = '--'
                current = '(Failure in test, no image produced)'
                failed_rows.append(row_template.format(name, '', current, expected_image, failed))
            else:
                status = ' (passed)'
                failed = '--'
                current = linked_image_template.format(actual_image)
            subdir_rows.append(row_template.format(name, status, current, expected_image, failed))
        body_sections.append(subdir_template.format(subdir=subdir, rows='\n'.join(subdir_rows)))
    if failed_rows:
        failed = failed_template.format(rows='\n'.join(failed_rows))
    else:
        failed = ''
    body = ''.join(body_sections)
    html = html_template.format(failed=failed, body=body)
    index = os.path.join(image_dir, 'index.html')
    with open(index, 'w') as f:
        f.write(html)
    show_message = not show_browser
    if show_browser:
        try:
            import webbrowser
            webbrowser.open(index)
        except Exception:
            show_message = True
    if show_message:
        print(f'Open {index} in a browser for a visual comparison.')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-browser', action='store_true', help="Don't show browser after creating index page.")
    args = parser.parse_args()
    run(show_browser=not args.no_browser)