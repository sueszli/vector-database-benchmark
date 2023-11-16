from urllib.error import URLError
import pandas as pd
import pydeck as pdk
import streamlit as st
from streamlit.hello.utils import show_code

def mapping_demo():
    if False:
        for i in range(10):
            print('nop')

    @st.cache_data
    def from_data_file(filename):
        if False:
            return 10
        url = 'https://raw.githubusercontent.com/streamlit/example-data/master/hello/v1/%s' % filename
        return pd.read_json(url)
    try:
        ALL_LAYERS = {'Bike Rentals': pdk.Layer('HexagonLayer', data=from_data_file('bike_rental_stats.json'), get_position=['lon', 'lat'], radius=200, elevation_scale=4, elevation_range=[0, 1000], extruded=True), 'Bart Stop Exits': pdk.Layer('ScatterplotLayer', data=from_data_file('bart_stop_stats.json'), get_position=['lon', 'lat'], get_color=[200, 30, 0, 160], get_radius='[exits]', radius_scale=0.05), 'Bart Stop Names': pdk.Layer('TextLayer', data=from_data_file('bart_stop_stats.json'), get_position=['lon', 'lat'], get_text='name', get_color=[0, 0, 0, 200], get_size=10, get_alignment_baseline="'bottom'"), 'Outbound Flow': pdk.Layer('ArcLayer', data=from_data_file('bart_path_stats.json'), get_source_position=['lon', 'lat'], get_target_position=['lon2', 'lat2'], get_source_color=[200, 30, 0, 160], get_target_color=[200, 30, 0, 160], auto_highlight=True, width_scale=0.0001, get_width='outbound', width_min_pixels=3, width_max_pixels=30)}
        st.sidebar.markdown('### Map Layers')
        selected_layers = [layer for (layer_name, layer) in ALL_LAYERS.items() if st.sidebar.checkbox(layer_name, True)]
        if selected_layers:
            st.pydeck_chart(pdk.Deck(map_style=None, initial_view_state={'latitude': 37.76, 'longitude': -122.4, 'zoom': 11, 'pitch': 50}, layers=selected_layers))
        else:
            st.error('Please choose at least one layer above.')
    except URLError as e:
        st.error('\n            **This demo requires internet access.**\n            Connection error: %s\n        ' % e.reason)
st.set_page_config(page_title='Mapping Demo', page_icon='🌍')
st.markdown('# Mapping Demo')
st.sidebar.header('Mapping Demo')
st.write('This demo shows how to use\n[`st.pydeck_chart`](https://docs.streamlit.io/library/api-reference/charts/st.pydeck_chart)\nto display geospatial data.')
mapping_demo()
show_code(mapping_demo)