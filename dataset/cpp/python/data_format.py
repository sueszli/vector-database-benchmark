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

# Data in JSON format. For more info on this format:
# https://fileinfo.com/extension/json
_JSON_DATA = """
[
    {
        "author": "John",
        "title": "Summer",
        "body": "Summer time is hot"
    },
    {
        "author": "Jane",
        "title": "Winter",
        "body": "Winter time is cold"
    }
]
"""

# Data in XML format. For more info on this format:
# https://fileinfo.com/extension/xml
_XML_DATA = """
<notepad>
    <note>
        <author>John</author>
        <title>Summer</title>
        <body>Summer time is hot</body>
    </note>
    <note>
        <author>Jane</author>
        <title>Winter</title>
        <body>Winter time is cold</body>
    </note>
</notepad>
"""

# Data in CSV format. For more info on this format:
# https://fileinfo.com/extension/csv
_CSV_DATA = """
John,Summer,Summer time is hot
Jane,Winter,Winter time is cold
"""


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
        """Create note from dictionary data."""
        return cls(**data)

    @classmethod
    def fields(cls):
        """Get field names to simplify parsing logic."""
        return tuple(field.name for field in fields(cls))


def main():
    # Let's use `json.load` to parse note data from a JSON file
    # https://docs.python.org/3/library/json.html
    json_content = json.load(StringIO(_JSON_DATA))
    json_notes = [Note.from_data(data) for data in json_content]
    assert all(isinstance(note, Note) for note in json_notes)

    # Let's use `ElementTree.parse` to parse note data from a XML file
    # https://docs.python.org/3/library/xml.html
    tree = ETree.parse(StringIO(_XML_DATA))
    xml_notes = [
        Note.from_data({
            field: note_el.findtext(field)
            for field in Note.fields()
        }) for note_el in tree.getroot()
    ]
    assert all(isinstance(note, Note) for note in xml_notes)

    # Let's use `csv.DictReader` to parse note data from a CSV file
    # https://docs.python.org/3/library/csv.html
    csv_reader = DictReader(StringIO(_CSV_DATA), fieldnames=Note.fields())
    csv_notes = [Note.from_data(row) for row in csv_reader]
    assert all(isinstance(note, Note) for note in csv_notes)

    # All three formats have similar `Note` objects
    for json_note, xml_note, csv_note in zip(json_notes, xml_notes, csv_notes):
        assert json_note == xml_note == csv_note


if __name__ == "__main__":
    main()
