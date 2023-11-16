"""Make HTML tables that report where TF and TFLite failed to convert models.

This is primarily used by generate_examples.py. See it or
`make_report_table` for more details on usage.
"""
import html
import json
import re
FAILED = 'FAILED'
SUCCESS = 'SUCCESS'
NOTRUN = 'NOTRUN'

def make_report_table(fp, title, reports):
    if False:
        print('Hello World!')
    'Make an HTML report of the success/failure reports.\n\n  Args:\n    fp: File-like object in which to put the html.\n    title: "Title of the zip file this pertains to."\n    reports: a list of conversion attempts. (report_args, report_vals) i.e.\n      ({"shape": [1,2,3], "type": "tf.float32"},\n       {"tf": "SUCCESS", "tflite_converter": "FAILURE",\n        "tf_log": "", "tflite_converter_log": "Unsupported type."})\n  '
    reports.sort(key=lambda x: x[1]['tflite_converter'], reverse=False)
    reports.sort(key=lambda x: x[1]['tf'], reverse=True)

    def result_cell(x, row, col):
        if False:
            return 10
        'Produce a cell with the condition string `x`.'
        s = html.escape(repr(x), quote=True)
        color = '#44ff44' if x == SUCCESS else '#ff4444' if x == FAILED else '#eeeeee'
        handler = 'ShowLog(%d, %d)' % (row, col)
        fp.write("<td style='background-color: %s' onclick='%s'>%s</td>\n" % (color, handler, s))
    fp.write('<html>\n<head>\n<title>tflite report</title>\n<style>\nbody { font-family: Arial; }\nth { background-color: #555555; color: #eeeeee; }\ntd { vertical-align: top; }\ntd.horiz {width: 50%;}\npre { white-space: pre-wrap; word-break: keep-all; }\ntable {width: 100%;}\n</style>\n</head>\n')
    fp.write('<script> \n')
    fp.write('\nfunction ShowLog(row, col) {\n\nvar log = document.getElementById("log");\nlog.innerHTML = "<pre>" + data[row][col]  + "</pre>";\n}\n')
    fp.write('var data = \n')
    logs = json.dumps([[escape_and_normalize(x[1]['tf_log']), escape_and_normalize(x[1]['tflite_converter_log'])] for x in reports])
    fp.write(logs)
    fp.write(';</script>\n')
    fp.write('\n<body>\n<h1>TensorFlow Lite Conversion</h1>\n<h2>%s</h2>\n' % title)
    param_keys = {}
    for (params, _) in reports:
        for k in params.keys():
            param_keys[k] = True
    fp.write('<table>\n')
    fp.write("<tr><td class='horiz'>\n")
    fp.write("<div style='height:1000px; overflow:auto'>\n")
    fp.write('<table>\n')
    fp.write('<tr>\n')
    for p in param_keys:
        fp.write('<th>%s</th>\n' % html.escape(p, quote=True))
    fp.write('<th>TensorFlow</th>\n')
    fp.write('<th>TensorFlow Lite Converter</th>\n')
    fp.write('</tr>\n')
    for (idx, (params, vals)) in enumerate(reports):
        fp.write('<tr>\n')
        for p in param_keys:
            fp.write('  <td>%s</td>\n' % html.escape(repr(params.get(p, None)), quote=True))
        result_cell(vals['tf'], idx, 0)
        result_cell(vals['tflite_converter'], idx, 1)
        fp.write('</tr>\n')
    fp.write('</table>\n')
    fp.write('</div>\n')
    fp.write('</td>\n')
    fp.write("<td class='horiz' id='log'></td></tr>\n")
    fp.write('</table>\n')
    fp.write('<script>\n')
    fp.write('</script>\n')
    fp.write('\n    </body>\n    </html>\n    ')

def escape_and_normalize(log):
    if False:
        i = 10
        return i + 15
    log = re.sub('/tmp/[^ ]+ ', '/NORMALIZED_TMP_FILE_PATH ', log)
    log = re.sub('/build/work/[^/]+', '/NORMALIZED_BUILD_PATH', log)
    return html.escape(log, quote=True)