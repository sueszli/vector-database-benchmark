"""Get DOM elements using selectors."""
import webview

def get_elements(window):
    if False:
        for i in range(10):
            print('nop')
    heading = window.get_elements('#heading')
    content = window.get_elements('.content')
    print('Heading:\n %s ' % heading[0]['outerHTML'])
    print('Content 1:\n %s ' % content[0]['outerHTML'])
    print('Content 2:\n %s ' % content[1]['outerHTML'])
if __name__ == '__main__':
    html = '\n      <html>\n        <body>\n          <h1 id="heading">Heading</h1>\n          <div class="content">Content 1</div>\n          <div class="content">Content 2</div>\n        </body>\n      </html>\n    '
    window = webview.create_window('Get elements example', html=html)
    webview.start(get_elements, window, debug=True)