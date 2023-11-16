from reactpy import component, html, run, use_state

@component
def CounterList():
    if False:
        print('Hello World!')
    (counters, set_counters) = use_state([0, 0, 0])

    def make_increment_click_handler(index):
        if False:
            i = 10
            return i + 15

        def handle_click(event):
            if False:
                print('Hello World!')
            new_value = counters[index] + 1
            set_counters(counters[:index] + [new_value] + counters[index + 1:])
        return handle_click
    return html.ul([html.li({'key': index}, count, html.button({'on_click': make_increment_click_handler(index)}, '+1')) for (index, count) in enumerate(counters)])
run(CounterList)