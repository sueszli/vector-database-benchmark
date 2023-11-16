"""
SQLite module tests
"""

from txtai.embeddings import Embeddings

from .testrdbms import Common


# pylint: disable=R0904
class TestSQLite(Common.TestRDBMS):
    """
    Embeddings with content stored in SQLite tests.
    """

    @classmethod
    def setUpClass(cls):
        """
        Initialize test data.
        """

        cls.data = [
            "US tops 5 million confirmed virus cases",
            "Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg",
            "Beijing mobilises invasion craft along coast as Taiwan tensions escalate",
            "The National Park Service warns against sacrificing slower friends in a bear attack",
            "Maine man wins $1M from $25 lottery ticket",
            "Make huge profits without work, earn up to $100,000 a day",
        ]

        # Content backend
        cls.backend = "sqlite"

        # Create embeddings model, backed by sentence-transformers & transformers
        cls.embeddings = Embeddings({"path": "sentence-transformers/nli-mpnet-base-v2", "content": cls.backend})

    @classmethod
    def tearDownClass(cls):
        """
        Cleanup data.
        """

        if cls.embeddings:
            cls.embeddings.close()

    def testFunction(self):
        """
        Test custom functions
        """

        embeddings = Embeddings(
            {
                "path": "sentence-transformers/nli-mpnet-base-v2",
                "content": self.backend,
                "functions": [{"name": "length", "function": "testdatabase.testsqlite.length"}],
            }
        )

        # Create an index for the list of text
        embeddings.index([(uid, text, None) for uid, text in enumerate(self.data)])

        # Search for best match
        result = embeddings.search("select length(text) length from txtai where id = 0", 1)[0]

        self.assertEqual(result["length"], 39)


def length(text):
    """
    Custom SQL function.
    """

    return len(text)
