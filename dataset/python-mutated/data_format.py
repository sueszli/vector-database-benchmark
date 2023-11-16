"""
One of the reasons people use Python is that it simplifies parsing and
converting data to serialized objects for further analysis. This module
shows how to parse and process these data formats: JSON, XML and CSV.
"""
import json
from csv import DictReader
from dataclasses import dataclass, fields
from io import StringIO
from xml.etree import ElementTree as ETree
_JSON_DATA = '\n[\n    {\n        "author": "John",\n        "title": "Summer",\n        "body": "Summer time is hot"\n    },\n    {\n        "author": "Jane",\n        "title": "Winter",\n        "body": "Winter time is cold"\n    }\n]\n'
_XML_DATA = '\n<notepad>\n    <note>\n        <author>John</author>\n        <title>Summer</title>\n        <body>Summer time is hot</body>\n    </note>\n    <note>\n        <author>Jane</author>\n        <title>Winter</title>\n        <body>Winter time is cold</body>\n    </note>\n</notepad>\n'
_CSV_DATA = '\nJohn,Summer,Summer time is hot\nJane,Winter,Winter time is cold\n'

@dataclass
class Note:
    """Note model.

    We notice that each data format has the notion of a record with fields
    associated with them. To streamline the creation and comparison of
    these records, we define an in-memory model of what it is.
    """
    author: str
    title: str
    body: str

    @classmethod
    def from_data(cls, data):
        if False:
            return 10
        'Create note from dictionary data.'
        return cls(**data)

    @classmethod
    def fields(cls):
        if False:
            while True:
                i = 10
        'Get field names to simplify parsing logic.'
        return tuple((field.name for field in fields(cls)))

def main():
    if False:
        for i in range(10):
            print('nop')
    json_content = json.load(StringIO(_JSON_DATA))
    json_notes = [Note.from_data(data) for data in json_content]
    assert all((isinstance(note, Note) for note in json_notes))
    tree = ETree.parse(StringIO(_XML_DATA))
    xml_notes = [Note.from_data({field: note_el.findtext(field) for field in Note.fields()}) for note_el in tree.getroot()]
    assert all((isinstance(note, Note) for note in xml_notes))
    csv_reader = DictReader(StringIO(_CSV_DATA), fieldnames=Note.fields())
    csv_notes = [Note.from_data(row) for row in csv_reader]
    assert all((isinstance(note, Note) for note in csv_notes))
    for (json_note, xml_note, csv_note) in zip(json_notes, xml_notes, csv_notes):
        assert json_note == xml_note == csv_note
if __name__ == '__main__':
    main()