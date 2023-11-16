from reactpy import component, html, run

@component
def Photo():
    if False:
        while True:
            i = 10
    return html.img({'src': 'https://picsum.photos/id/237/500/300', 'style': {'width': '50%'}, 'alt': 'Puppy'})
run(Photo)