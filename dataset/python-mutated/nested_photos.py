from reactpy import component, html, run

@component
def Photo():
    if False:
        while True:
            i = 10
    return html.img({'src': 'https://picsum.photos/id/274/500/300', 'style': {'width': '30%'}, 'alt': 'Ray Charles'})

@component
def Gallery():
    if False:
        i = 10
        return i + 15
    return html.section(html.h1('Famous Musicians'), Photo(), Photo(), Photo())
run(Gallery)