import textacy
from textacy import extract

class KeytermExtractor:
    """
    A class for extracting keyterms from a given text using various algorithms.
    """

    def __init__(self, raw_text: str, top_n_values: int=20):
        if False:
            i = 10
            return i + 15
        '\n        Initialize the KeytermExtractor object.\n\n        Args:\n            raw_text (str): The raw input text.\n            top_n_values (int): The number of top keyterms to extract.\n        '
        self.raw_text = raw_text
        self.text_doc = textacy.make_spacy_doc(self.raw_text, lang='en_core_web_md')
        self.top_n_values = top_n_values

    def get_keyterms_based_on_textrank(self):
        if False:
            i = 10
            return i + 15
        '\n        Extract keyterms using the TextRank algorithm.\n\n        Returns:\n            List[str]: A list of top keyterms based on TextRank.\n        '
        return list(extract.keyterms.textrank(self.text_doc, normalize='lemma', topn=self.top_n_values))

    def get_keyterms_based_on_sgrank(self):
        if False:
            while True:
                i = 10
        '\n        Extract keyterms using the SGRank algorithm.\n\n        Returns:\n            List[str]: A list of top keyterms based on SGRank.\n        '
        return list(extract.keyterms.sgrank(self.text_doc, normalize='lemma', topn=self.top_n_values))

    def get_keyterms_based_on_scake(self):
        if False:
            return 10
        '\n        Extract keyterms using the sCAKE algorithm.\n\n        Returns:\n            List[str]: A list of top keyterms based on sCAKE.\n        '
        return list(extract.keyterms.scake(self.text_doc, normalize='lemma', topn=self.top_n_values))

    def get_keyterms_based_on_yake(self):
        if False:
            print('Hello World!')
        '\n        Extract keyterms using the YAKE algorithm.\n\n        Returns:\n            List[str]: A list of top keyterms based on YAKE.\n        '
        return list(extract.keyterms.yake(self.text_doc, normalize='lemma', topn=self.top_n_values))

    def bi_gramchunker(self):
        if False:
            while True:
                i = 10
        '\n        Chunk the text into bigrams.\n\n        Returns:\n            List[str]: A list of bigrams.\n        '
        return list(textacy.extract.basics.ngrams(self.text_doc, n=2, filter_stops=True, filter_nums=True, filter_punct=True))

    def tri_gramchunker(self):
        if False:
            return 10
        '\n        Chunk the text into trigrams.\n\n        Returns:\n            List[str]: A list of trigrams.\n        '
        return list(textacy.extract.basics.ngrams(self.text_doc, n=3, filter_stops=True, filter_nums=True, filter_punct=True))