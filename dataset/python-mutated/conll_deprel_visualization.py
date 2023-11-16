from stanza.models.common.constant import is_right_to_left
import spacy
import argparse
from spacy import displacy
from spacy.tokens import Doc
from stanza.utils import conll
from stanza.utils.visualization import dependency_visualization as viz

def conll_to_visual(conll_file, pipeline, sent_count=10, display_all=False):
    if False:
        while True:
            i = 10
    '\n    Takes in a conll file and visualizes it by converting the conll file to a Stanza Document object\n    and visualizing it with the visualize_doc method.\n\n    Input should be a proper conll file.\n\n    The pipeline for the conll file to be processed in must be provided as well.\n\n    Optionally, the sent_count argument can be tweaked to display a different amount of sentences.\n\n    To display all of the sentences in a conll file, the display_all argument can optionally be set to True.\n    BEWARE: setting this argument for a large conll file may result in too many renderings, resulting in a crash.\n    '
    doc = conll.CoNLL.conll2doc(conll_file)
    if display_all:
        viz.visualize_doc(conll.CoNLL.conll2doc(conll_file), pipeline)
    else:
        visualization_options = {'compact': True, 'bg': '#09a3d5', 'color': 'white', 'distance': 100, 'font': 'Source Sans Pro', 'offset_x': 30, 'arrow_spacing': 20}
        nlp = spacy.blank('en')
        (sentences_to_visualize, rtl, num_sentences) = ([], is_right_to_left(pipeline), len(doc.sentences))
        for i in range(sent_count):
            if i >= num_sentences:
                break
            sentence = doc.sentences[i]
            (words, lemmas, heads, deps, tags) = ([], [], [], [], [])
            sentence_words = sentence.words
            if rtl:
                sentence_words = reversed(sentence.words)
                sent_len = len(sentence.words)
            for word in sentence_words:
                words.append(word.text)
                lemmas.append(word.lemma)
                deps.append(word.deprel)
                tags.append(word.upos)
                if rtl and word.head == 0:
                    heads.append(sent_len - word.id)
                elif rtl and word.head != 0:
                    heads.append(sent_len - word.head)
                elif not rtl and word.head == 0:
                    heads.append(word.id - 1)
                elif not rtl and word.head != 0:
                    heads.append(word.head - 1)
            document_result = Doc(nlp.vocab, words=words, lemmas=lemmas, heads=heads, deps=deps, pos=tags)
            sentences_to_visualize.append(document_result)
        print(sentences_to_visualize)
        for line in sentences_to_visualize:
            displacy.render(line, style='dep', options=visualization_options)

def main():
    if False:
        i = 10
        return i + 15
    parser = argparse.ArgumentParser()
    parser.add_argument('--conll_file', type=str, default='C:\\Users\\Alex\\stanza\\demo\\en_test.conllu.txt', help='File path of the CoNLL file to visualize dependencies of')
    parser.add_argument('--pipeline', type=str, default='en', help="Language code of the language pipeline to use (ex: 'en' for English)")
    parser.add_argument('--sent_count', type=int, default=10, help='Number of sentences to visualize from CoNLL file')
    parser.add_argument('--display_all', type=bool, default=False, help='Whether or not to visualize all of the sentences from the file. Overrides sent_count if set to True')
    args = parser.parse_args()
    conll_to_visual(args.conll_file, args.pipeline, args.sent_count, args.display_all)
    return
if __name__ == '__main__':
    main()