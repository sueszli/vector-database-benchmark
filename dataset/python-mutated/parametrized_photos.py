from reactpy import component, html, run

@component
def Photo(alt_text, image_id):
    if False:
        while True:
            i = 10
    return html.img({'src': f'https://picsum.photos/id/{image_id}/500/200', 'style': {'width': '50%'}, 'alt': alt_text})

@component
def Gallery():
    if False:
        while True:
            i = 10
    return html.section(html.h1('Photo Gallery'), Photo('Landscape', image_id=830), Photo('City', image_id=274), Photo('Puppy', image_id=237))
run(Gallery)