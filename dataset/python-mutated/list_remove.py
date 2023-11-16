from reactpy import component, html, run, use_state

@component
def ArtistList():
    if False:
        for i in range(10):
            print('nop')
    (artist_to_add, set_artist_to_add) = use_state('')
    (artists, set_artists) = use_state(['Marta Colvin Andrade', 'Lamidi Olonade Fakeye', 'Louise Nevelson'])

    def handle_change(event):
        if False:
            for i in range(10):
                print('nop')
        set_artist_to_add(event['target']['value'])

    def handle_add_click(event):
        if False:
            i = 10
            return i + 15
        if artist_to_add not in artists:
            set_artists([*artists, artist_to_add])
            set_artist_to_add('')

    def make_handle_delete_click(index):
        if False:
            for i in range(10):
                print('nop')

        def handle_click(event):
            if False:
                i = 10
                return i + 15
            set_artists(artists[:index] + artists[index + 1:])
        return handle_click
    return html.div(html.h1('Inspiring sculptors:'), html.input({'value': artist_to_add, 'on_change': handle_change}), html.button({'on_click': handle_add_click}, 'add'), html.ul([html.li({'key': name}, name, html.button({'on_click': make_handle_delete_click(index)}, 'delete')) for (index, name) in enumerate(artists)]))
run(ArtistList)