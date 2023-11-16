from reactpy import component, html, run

@component
def Button(display_text, on_click):
    if False:
        i = 10
        return i + 15
    return html.button({'on_click': on_click}, display_text)

@component
def PlayButton(movie_name):
    if False:
        while True:
            i = 10

    def handle_click(event):
        if False:
            for i in range(10):
                print('nop')
        print(f'Playing {movie_name}')
    return Button(f'Play {movie_name}', on_click=handle_click)

@component
def FastForwardButton():
    if False:
        for i in range(10):
            print('nop')

    def handle_click(event):
        if False:
            i = 10
            return i + 15
        print('Skipping ahead')
    return Button('Fast forward', on_click=handle_click)

@component
def App():
    if False:
        return 10
    return html.div(PlayButton('Buena Vista Social Club'), FastForwardButton())
run(App)