import reactpy

def increment(last_count):
    if False:
        print('Hello World!')
    return last_count + 1

def decrement(last_count):
    if False:
        print('Hello World!')
    return last_count - 1

@reactpy.component
def Counter():
    if False:
        for i in range(10):
            print('nop')
    initial_count = 0
    (count, set_count) = reactpy.hooks.use_state(initial_count)
    return reactpy.html.div(f'Count: {count}', reactpy.html.button({'on_click': lambda event: set_count(initial_count)}, 'Reset'), reactpy.html.button({'on_click': lambda event: set_count(increment)}, '+'), reactpy.html.button({'on_click': lambda event: set_count(decrement)}, '-'))
reactpy.run(Counter)