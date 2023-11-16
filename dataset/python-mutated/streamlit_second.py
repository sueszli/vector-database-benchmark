import networkx as nx
from typing import List
import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from scripts.utils import get_filenames_from_dir
from streamlit_extras import add_vertical_space as avs
from annotated_text import annotated_text, parameters
from streamlit_extras.badges import badge
import nltk
st.set_page_config(page_title='Resume Matcher', page_icon='Assets/img/favicon.ico', initial_sidebar_state='auto')
nltk.download('punkt')
parameters.SHOW_LABEL_SEPARATOR = False
parameters.BORDER_RADIUS = 3
parameters.PADDING = '0.5 0.25rem'

def create_star_graph(nodes_and_weights, title):
    if False:
        for i in range(10):
            print('nop')
    G = nx.Graph()
    central_node = 'resume'
    G.add_node(central_node)
    for (node, weight) in nodes_and_weights:
        G.add_node(node)
        G.add_edge(central_node, node, weight=weight * 100)
    pos = nx.spring_layout(G)
    edge_x = []
    edge_y = []
    for edge in G.edges():
        (x0, y0) = pos[edge[0]]
        (x1, y1) = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
    node_x = []
    node_y = []
    for node in G.nodes():
        (x, y) = pos[node]
        node_x.append(x)
        node_y.append(y)
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text', marker=dict(showscale=True, colorscale='Rainbow', reversescale=True, color=[], size=10, colorbar=dict(thickness=15, title='Node Connections', xanchor='left', titleside='right'), line_width=2))
    node_adjacencies = []
    node_text = []
    for node in G.nodes():
        adjacencies = list(G.adj[node])
        node_adjacencies.append(len(adjacencies))
        node_text.append(f'{node}<br># of connections: {len(adjacencies)}')
    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text
    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(title=title, titlefont_size=16, showlegend=False, hovermode='closest', margin=dict(b=20, l=5, r=5, t=40), xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    st.plotly_chart(fig)

def create_annotated_text(input_string: str, word_list: List[str], annotation: str, color_code: str):
    if False:
        print('Hello World!')
    tokens = nltk.word_tokenize(input_string)
    word_set = set(word_list)
    annotated_text = []
    for token in tokens:
        if token in word_set:
            annotated_text.append((token, annotation, color_code))
        else:
            annotated_text.append(token)
    return annotated_text

def read_json(filename):
    if False:
        print('Hello World!')
    with open(filename) as f:
        data = json.load(f)
    return data

def tokenize_string(input_string):
    if False:
        print('Hello World!')
    tokens = nltk.word_tokenize(input_string)
    return tokens
st.title(':blue[Resume Matcher]')
with st.sidebar:
    st.image('Assets/img/header_image.png')
    st.subheader('Free and Open Source ATS to help your resume pass the screening stage.')
    st.markdown('Check the website [www.resumematcher.fyi](https://www.resumematcher.fyi/)')
    st.markdown('Give Resume Matcher a ⭐ on [GitHub](https://github.com/srbhr/resume-matcher)')
    badge(type='github', name='srbhr/Resume-Matcher')
    st.markdown('For updates follow me on Twitter.')
    badge(type='twitter', name='_srbhr_')
    st.markdown('If you like the project and would like to further help in development please consider 👇')
    badge(type='buymeacoffee', name='srbhr')
st.divider()
avs.add_vertical_space(1)
resume_names = get_filenames_from_dir('Data/Processed/Resumes')
output = st.selectbox(f'There are {len(resume_names)} resumes present. Please select one from the menu below:', resume_names)
avs.add_vertical_space(5)
selected_file = read_json('Data/Processed/Resumes/' + output)
avs.add_vertical_space(2)
st.markdown('#### Parsed Resume Data')
st.caption("This text is parsed from your resume. This is how it'll look like after getting parsed by an ATS.")
st.caption('Utilize this to understand how to make your resume ATS friendly.')
avs.add_vertical_space(3)
st.write(selected_file['clean_data'])
avs.add_vertical_space(3)
st.write("Now let's take a look at the extracted keywords from the resume.")
annotated_text(create_annotated_text(selected_file['clean_data'], selected_file['extracted_keywords'], 'KW', '#0B666A'))
avs.add_vertical_space(5)
st.write("Now let's take a look at the extracted entities from the resume.")
create_star_graph(selected_file['keyterms'], 'Entities from Resume')
df2 = pd.DataFrame(selected_file['keyterms'], columns=['keyword', 'value'])
keyword_dict = {}
for (keyword, value) in selected_file['keyterms']:
    keyword_dict[keyword] = value * 100
fig = go.Figure(data=[go.Table(header=dict(values=['Keyword', 'Value'], font=dict(size=12), fill_color='#070A52'), cells=dict(values=[list(keyword_dict.keys()), list(keyword_dict.values())], line_color='darkslategray', fill_color='#6DA9E4'))])
st.plotly_chart(fig)
st.divider()
fig = px.treemap(df2, path=['keyword'], values='value', color_continuous_scale='Rainbow', title='Key Terms/Topics Extracted from your Resume')
st.write(fig)
avs.add_vertical_space(5)
job_descriptions = get_filenames_from_dir('Data/Processed/JobDescription')
output = st.selectbox(f'There are {len(job_descriptions)} job descriptions present. Please select one from the menu below:', job_descriptions)
avs.add_vertical_space(5)
selected_jd = read_json('Data/Processed/JobDescription/' + output)
avs.add_vertical_space(2)
st.markdown('#### Job Description')
st.caption("Currently in the pipeline I'm parsing this from PDF but it'll be from txt or copy paste.")
avs.add_vertical_space(3)
st.write(selected_jd['clean_data'])
st.markdown('#### Common Words between Job Description and Resumes Highlighted.')
annotated_text(create_annotated_text(selected_file['clean_data'], selected_jd['extracted_keywords'], 'JD', '#F24C3D'))
st.write("Now let's take a look at the extracted entities from the job description.")
create_star_graph(selected_jd['keyterms'], 'Entities from Job Description')
df2 = pd.DataFrame(selected_jd['keyterms'], columns=['keyword', 'value'])
keyword_dict = {}
for (keyword, value) in selected_jd['keyterms']:
    keyword_dict[keyword] = value * 100
fig = go.Figure(data=[go.Table(header=dict(values=['Keyword', 'Value'], font=dict(size=12), fill_color='#070A52'), cells=dict(values=[list(keyword_dict.keys()), list(keyword_dict.values())], line_color='darkslategray', fill_color='#6DA9E4'))])
st.plotly_chart(fig)
st.divider()
fig = px.treemap(df2, path=['keyword'], values='value', color_continuous_scale='Rainbow', title='Key Terms/Topics Extracted from the selected Job Description')
st.write(fig)
avs.add_vertical_space(5)
st.divider()
st.markdown('## Vector Similarity Scores')
st.caption('Powered by Qdrant Vector Search')
st.info('These are pre-computed queries', icon='ℹ')
st.warning('Running Qdrant or Sentence Transformers without having capacity is not recommended', icon='⚠')
data = [{'text': "{'resume': 'Alfred Pennyworth", 'query': 'Job Description Product Manager', 'score': 0.62658}, {'text': "{'resume': 'Barry Allen", 'query': 'Job Description Product Manager', 'score': 0.43777737}, {'text': "{'resume': 'Bruce Wayne ", 'query': 'Job Description Product Manager', 'score': 0.39835533}, {'text': "{'resume': 'JOHN DOE", 'query': 'Job Description Product Manager', 'score': 0.3915512}, {'text': "{'resume': 'Harvey Dent", 'query': 'Job Description Product Manager', 'score': 0.3519544}, {'text': "{'resume': 'Barry Allen", 'query': 'Job Description Senior Full Stack Engineer', 'score': 0.6541866}, {'text': "{'resume': 'Alfred Pennyworth", 'query': 'Job Description Senior Full Stack Engineer', 'score': 0.59806436}, {'text': "{'resume': 'JOHN DOE", 'query': 'Job Description Senior Full Stack Engineer', 'score': 0.5951386}, {'text': "{'resume': 'Bruce Wayne ", 'query': 'Job Description Senior Full Stack Engineer', 'score': 0.57700855}, {'text': "{'resume': 'Harvey Dent", 'query': 'Job Description Senior Full Stack Engineer', 'score': 0.38489106}, {'text': "{'resume': 'Barry Allen", 'query': 'Job Description Front End Engineer', 'score': 0.76813436}, {'text': "{'resume': 'Bruce Wayne'", 'query': 'Job Description Front End Engineer', 'score': 0.60440844}, {'text': "{'resume': 'JOHN DOE", 'query': 'Job Description Front End Engineer', 'score': 0.56080043}, {'text': "{'resume': 'Alfred Pennyworth", 'query': 'Job Description Front End Engineer', 'score': 0.5395049}, {'text': "{'resume': 'Harvey Dent", 'query': 'Job Description Front End Engineer', 'score': 0.3859515}, {'text': "{'resume': 'JOHN DOE", 'query': 'Job Description Java Developer', 'score': 0.5449441}, {'text': "{'resume': 'Alfred Pennyworth", 'query': 'Job Description Java Developer', 'score': 0.53476423}, {'text': "{'resume': 'Barry Allen", 'query': 'Job Description Java Developer', 'score': 0.5313871}, {'text': "{'resume': 'Bruce Wayne ", 'query': 'Job Description Java Developer', 'score': 0.44446343}, {'text': "{'resume': 'Harvey Dent", 'query': 'Job Description Java Developer', 'score': 0.3616274}]
df = pd.DataFrame(data)
df1 = df[df['query'] == 'Job Description Product Manager'].sort_values(by='score', ascending=False)
df2 = df[df['query'] == 'Job Description Senior Full Stack Engineer'].sort_values(by='score', ascending=False)
df3 = df[df['query'] == 'Job Description Front End Engineer'].sort_values(by='score', ascending=False)
df4 = df[df['query'] == 'Job Description Java Developer'].sort_values(by='score', ascending=False)

def plot_df(df, title):
    if False:
        print('Hello World!')
    fig = px.bar(df, x='text', y=df['score'] * 100, title=title)
    st.plotly_chart(fig)
st.markdown('### Bar plots of scores based on similarity to Job Description.')
st.subheader(':blue[Legend]')
st.text('Alfred Pennyworth :  Product Manager')
st.text('Barry Allen :  Front End Developer')
st.text('Harvey Dent :  Machine Learning Engineer')
st.text('Bruce Wayne :  Fullstack Developer (MERN)')
st.text('John Doe :  Fullstack Developer (Java)')
plot_df(df1, 'Job Description Product Manager 10+ Years of Exper')
plot_df(df2, 'Job Description Senior Full Stack Engineer 5+ Year')
plot_df(df3, 'Job Description Front End Engineer 2 Years of Expe')
plot_df(df4, 'Job Description Java Developer 3 Years of Experien')
avs.add_vertical_space(3)
st.markdown('[:arrow_up: Back to Top](#resume-matcher)')