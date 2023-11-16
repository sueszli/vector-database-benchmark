import sys
from timeit import Timer
from tornado.options import options, define, parse_command_line
from tornado.template import Template
define('num', default=100, help='number of iterations')
define('dump', default=False, help='print template generated code and exit')
context = {'page_title': "mitsuhiko's benchmark", 'table': [dict(a=1, b=2, c=3, d=4, e=5, f=6, g=7, h=8, i=9, j=10) for x in range(1000)]}
tmpl = Template('<!doctype html>\n<html>\n  <head>\n    <title>{{ page_title }}</title>\n  </head>\n  <body>\n    <div class="header">\n      <h1>{{ page_title }}</h1>\n    </div>\n    <ul class="navigation">\n    {% for href, caption in [         (\'index.html\', \'Index\'),         (\'downloads.html\', \'Downloads\'),         (\'products.html\', \'Products\')       ] %}\n      <li><a href="{{ href }}">{{ caption }}</a></li>\n    {% end %}\n    </ul>\n    <div class="table">\n      <table>\n      {% for row in table %}\n        <tr>\n        {% for cell in row %}\n          <td>{{ cell }}</td>\n        {% end %}\n        </tr>\n      {% end %}\n      </table>\n    </div>\n  </body>\n</html>')

def render():
    if False:
        while True:
            i = 10
    tmpl.generate(**context)

def main():
    if False:
        for i in range(10):
            print('nop')
    parse_command_line()
    if options.dump:
        print(tmpl.code)
        sys.exit(0)
    t = Timer(render)
    results = t.timeit(options.num) / options.num
    print('%0.3f ms per iteration' % (results * 1000))
if __name__ == '__main__':
    main()