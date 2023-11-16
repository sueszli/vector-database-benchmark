from reactpy import component, event, html, run

@component
def DoNotChangePages():
    if False:
        for i in range(10):
            print('nop')
    return html.div(html.p('Normally clicking this link would take you to a new page'), html.a({'on_click': event(lambda event: None, prevent_default=True), 'href': 'https://google.com'}, 'https://google.com'))
run(DoNotChangePages)