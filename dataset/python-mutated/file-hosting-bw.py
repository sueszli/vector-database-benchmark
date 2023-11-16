__license__ = 'GPL v3'
__copyright__ = '2014, Kovid Goyal <kovid at kovidgoyal.net>'
import os, subprocess, socket
BASE = '/srv/download/bw'

def main():
    if False:
        return 10
    if not os.path.exists(BASE):
        os.makedirs(BASE)
    os.chdir(BASE)
    for name in 'hours days months top10 summary'.split():
        subprocess.check_call(['vnstati', '--' + name, '-o', name + '.png'])
    html = '    <!DOCTYPE html>\n    <html>\n    <head><title>Bandwidth usage for {host}</title></head>\n    <body>\n    <style> .float {{ float: left; margin-right:30px; margin-left:30px; text-align:center; width: 500px; }}</style>\n    <h1>Bandwidth usage for {host}</h1>\n    <div class="float">\n    <h2>Summary</h2>\n    <img src="summary.png"/>\n    </div>\n    <div class="float">\n    <h2>Hours</h2>\n    <img src="hours.png"/>\n    </div>\n    <div class="float">\n    <h2>Days</h2>\n    <img src="days.png"/>\n    </div>\n    <div class="float">\n    <h2>Months</h2>\n    <img src="months.png"/>\n    </div>\n    <div class="float">\n    <h2>Top10</h2>\n    <img src="top10.png"/>\n    </div>\n    </body>\n    </html>\n    '.format(host=socket.gethostname())
    with open('index.html', 'wb') as f:
        f.write(html.encode('utf-8'))
if __name__ == '__main__':
    main()