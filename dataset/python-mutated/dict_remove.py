from reactpy import component, html, run, use_state

@component
def Definitions():
    if False:
        for i in range(10):
            print('nop')
    (term_to_add, set_term_to_add) = use_state(None)
    (definition_to_add, set_definition_to_add) = use_state(None)
    (all_terms, set_all_terms) = use_state({})

    def handle_term_to_add_change(event):
        if False:
            print('Hello World!')
        set_term_to_add(event['target']['value'])

    def handle_definition_to_add_change(event):
        if False:
            while True:
                i = 10
        set_definition_to_add(event['target']['value'])

    def handle_add_click(event):
        if False:
            while True:
                i = 10
        if term_to_add and definition_to_add:
            set_all_terms({**all_terms, term_to_add: definition_to_add})
            set_term_to_add(None)
            set_definition_to_add(None)

    def make_delete_click_handler(term_to_delete):
        if False:
            print('Hello World!')

        def handle_click(event):
            if False:
                for i in range(10):
                    print('nop')
            set_all_terms({t: d for (t, d) in all_terms.items() if t != term_to_delete})
        return handle_click
    return html.div(html.button({'on_click': handle_add_click}, 'add term'), html.label('Term: ', html.input({'value': term_to_add, 'on_change': handle_term_to_add_change})), html.label('Definition: ', html.input({'value': definition_to_add, 'on_change': handle_definition_to_add_change})), html.hr(), [html.div({'key': term}, html.button({'on_click': make_delete_click_handler(term)}, 'delete term'), html.dt(term), html.dd(definition)) for (term, definition) in all_terms.items()])
run(Definitions)