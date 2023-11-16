from dataclasses import dataclass
from typing import Optional

@dataclass(init=False, repr=False)
class Token:
    """
    A simple token representation, keeping track of the token's text, offset in the passage it was
    taken from, POS tag, dependency relation, and similar information.  These fields match spacy's
    exactly, so we can just use a spacy token for this.

    # Parameters

    text : `str`, optional
        The original text represented by this token.
    idx : `int`, optional
        The character offset of this token into the tokenized passage.
    idx_end : `int`, optional
        The character offset one past the last character in the tokenized passage.
    lemma_ : `str`, optional
        The lemma of this token.
    pos_ : `str`, optional
        The coarse-grained part of speech of this token.
    tag_ : `str`, optional
        The fine-grained part of speech of this token.
    dep_ : `str`, optional
        The dependency relation for this token.
    ent_type_ : `str`, optional
        The entity type (i.e., the NER tag) for this token.
    text_id : `int`, optional
        If your tokenizer returns integers instead of strings (e.g., because you're doing byte
        encoding, or some hash-based embedding), set this with the integer.  If this is set, we
        will bypass the vocabulary when indexing this token, regardless of whether `text` is also
        set.  You can `also` set `text` with the original text, if you want, so that you can
        still use a character-level representation in addition to a hash-based word embedding.
    type_id : `int`, optional
        Token type id used by some pretrained language models like original BERT

        The other fields on `Token` follow the fields on spacy's `Token` object; this is one we
        added, similar to spacy's `lex_id`.
    """
    __slots__ = ['text', 'idx', 'idx_end', 'lemma_', 'pos_', 'tag_', 'dep_', 'ent_type_', 'text_id', 'type_id']
    text: Optional[str]
    idx: Optional[int]
    idx_end: Optional[int]
    lemma_: Optional[str]
    pos_: Optional[str]
    tag_: Optional[str]
    dep_: Optional[str]
    ent_type_: Optional[str]
    text_id: Optional[int]
    type_id: Optional[int]

    def __init__(self, text: str=None, idx: int=None, idx_end: int=None, lemma_: str=None, pos_: str=None, tag_: str=None, dep_: str=None, ent_type_: str=None, text_id: int=None, type_id: int=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        assert text is None or isinstance(text, str)
        self.text = text
        self.idx = idx
        self.idx_end = idx_end
        self.lemma_ = lemma_
        self.pos_ = pos_
        self.tag_ = tag_
        self.dep_ = dep_
        self.ent_type_ = ent_type_
        self.text_id = text_id
        self.type_id = type_id

    def __str__(self):
        if False:
            print('Hello World!')
        return self.text

    def __repr__(self):
        if False:
            while True:
                i = 10
        return self.__str__()

    def ensure_text(self) -> str:
        if False:
            return 10
        "\n        Return the `text` field, raising an exception if it's `None`.\n        "
        if self.text is None:
            raise ValueError('Unexpected null text for token')
        else:
            return self.text

def show_token(token: Token) -> str:
    if False:
        for i in range(10):
            print('nop')
    return f'{token.text} (idx: {token.idx}) (idx_end: {token.idx_end}) (lemma: {token.lemma_}) (pos: {token.pos_}) (tag: {token.tag_}) (dep: {token.dep_}) (ent_type: {token.ent_type_}) (text_id: {token.text_id}) (type_id: {token.type_id}) '