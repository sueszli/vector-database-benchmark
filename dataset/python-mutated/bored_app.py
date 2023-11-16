from pywebio import *
from pywebio.output import *
from pywebio.input import *
from pywebio.pin import *
import requests
from typing import List
import requests
import spacy

def extract_noun_phrases(text: str):
    if False:
        for i in range(10):
            print('nop')
    'Extract noun phrases from a text'
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    return [chunk.text for chunk in doc.noun_chunks]

def get_query(phrase: str):
    if False:
        print('Hello World!')
    'Turn noun phrase into a query by replacing space with +'
    return '+'.join(phrase.split(' '))

def get_query_for_noun_phrases(text: str):
    if False:
        return 10
    'Turn list of noun phrases into a list of queries'
    noun_phrases = extract_noun_phrases(text)
    return [get_query(phrase) for phrase in noun_phrases]

def get_books(query: str):
    if False:
        i = 10
        return i + 15
    'Get the first 3 books based on the query'
    api = f'https://openlibrary.org/search.json?title={query}'
    response = requests.get(api)
    content = response.json()['docs'][:3]
    return content

def get_books_of_text(text: str):
    if False:
        i = 10
        return i + 15
    'Get books given a test'
    queries = get_query_for_noun_phrases(text)
    books = []
    for query in queries:
        books.extend(get_books(query))
    return books

def get_activity_content(inputs: dict):
    if False:
        for i in range(10):
            print('nop')
    'Get a random activity using Bored API'
    if inputs['value'] == 'random':
        api = 'https://www.boredapi.com/api/activity'
    else:
        api = f"https://www.boredapi.com/api/activity?type={inputs['value']}"
    response = requests.get(api)
    content = response.json()
    return content

def display_activity_for_boredom():
    if False:
        while True:
            i = 10
    put_markdown("# Find things to do when you're bored")
    activity_types = ['random', 'education', 'recreational', 'social', 'diy', 'charity', 'cooking', 'relaxation', 'music', 'busywork']
    put_select(name='type', label='Activity Type', options=activity_types)

def create_book_table(books: List[dict]):
    if False:
        for i in range(10):
            print('nop')
    if books == []:
        put_markdown('No books with this topic is found')
        return
    book_table = [[book['title'], book.get('author_name', ['_'])[0]] for book in books]
    book_table.insert(0, ['Title', 'Author'])
    put_table(book_table)

def main():
    if False:
        for i in range(10):
            print('nop')
    display_activity_for_boredom()
    while True:
        new_inputs = pin_wait_change(['type'])
        with use_scope('activity', clear=True):
            content = get_activity_content(new_inputs)
            put_markdown(f"## Your activity for today: {content['activity']}")
            put_markdown(f"Number of participants: {content['participants']}\n            Price: {content['price']}\n            Accessibility: {content['accessibility']}")
            put_markdown('## Books related to this activity you might be interested in:')
            with put_loading():
                books = get_books_of_text(content['activity'])
                create_book_table(books)
if __name__ == '__main__':
    start_server(main, port=36635, debug=True)