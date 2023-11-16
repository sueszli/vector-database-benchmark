"""
Create multiple windows, some of which have their own servers, both before and after `webview.start()` is called.
"""
import bottle
import webview
windows = []

def serverDescription(server):
    if False:
        print('Hello World!')
    return f"{str(server).replace('<', '').replace('>', '')}"
app1 = bottle.Bottle()

@app1.route('/')
def hello():
    if False:
        i = 10
        return i + 15
    return '<h1>Second Window</h1><p>This one is a web app and has its own server.</p>'
app2 = bottle.Bottle()

@app2.route('/')
def hello():
    if False:
        return 10
    head = '  <head>\n                    <style type="text/css">\n                        table {\n                          font-family: arial, sans-serif;\n                          border-collapse: collapse;\n                          width: 100%;\n                        }\n\n                        td, th {\n                          border: 1px solid #dddddd;\n                          text-align: center;\n                          padding: 8px;\n                        }\n\n                        tr:nth-child(even) {\n                          background-color: #dddddd;\n                        }\n                    </style>\n                </head>\n            '
    body = f" <body>\n                    <h1>Third Window</h1>\n                    <p>This one is another web app and has its own server. It was started after webview.start.</p>\n                    <p>Server Descriptions: </p>\n                    <table>\n                        <tr>\n                            <th>Window</th>\n                            <th>Object</th>\n                            <th>IP Address</th>\n                        </tr>\n                        <tr>\n                            <td>Global Server</td>\n                            <td>{serverDescription(webview.http.global_server)}</td>\n                            <td>{(webview.http.global_server.address if not webview.http.global_server is None else 'None')}</td>\n                        </tr>\n                        <tr>\n                            <td>First Window</td>\n                            <td>{serverDescription(windows[0]._server)}</td>\n                            <td>{(windows[0]._server.address if not windows[0]._server is None else 'None')}</td>\n                        </tr>\n                        <tr>\n                            <td>Second Window</td>\n                            <td>{serverDescription(windows[1]._server)}</td>\n                            <td>{windows[1]._server.address}</td>\n                        </tr>\n                        <tr>\n                            <td>Third Window</td>\n                            <td>{serverDescription(windows[2]._server)}</td>\n                            <td>{windows[2]._server.address}</td>\n                        </tr>\n                    </table>\n                </body>\n            "
    return head + body

def third_window():
    if False:
        while True:
            i = 10
    windows.append(webview.create_window('Window #3', url=app2))
if __name__ == '__main__':
    windows.append(webview.create_window('Window #1', html='<h1>First window</h1><p>This one is static HTML and just uses the global server for api calls.</p>'))
    windows.append(webview.create_window('Window #2', url=app1, http_port=3333))
    webview.start(third_window, debug=True, http_server=True, http_port=3334)