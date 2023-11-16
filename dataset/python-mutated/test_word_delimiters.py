import fitz
import string

def test_delimiters():
    if False:
        return 10
    'Test changing word delimiting characters.'
    doc = fitz.open()
    page = doc.new_page()
    text = 'word1,word2 - word3. word4?word5.'
    page.insert_text((50, 50), text)
    words0 = [w[4] for w in page.get_text('words')]
    assert words0 == ['word1,word2', '-', 'word3.', 'word4?word5.']
    words1 = [w[4] for w in page.get_text('words', delimiters=string.punctuation)]
    assert words0 != words1
    assert ' '.join(words1) == 'word1 word2 word3 word4 word5'
    assert [w[4] for w in page.get_text('words')] == words0