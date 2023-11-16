"""Swedish search language: includes the JS Swedish stemmer."""
from __future__ import annotations
from typing import TYPE_CHECKING, Dict
import snowballstemmer
from sphinx.search import SearchLanguage, parse_stop_word
swedish_stopwords = parse_stop_word("\n| source: http://snowball.tartarus.org/algorithms/swedish/stop.txt\noch            | and\ndet            | it, this/that\natt            | to (with infinitive)\ni              | in, at\nen             | a\njag            | I\nhon            | she\nsom            | who, that\nhan            | he\npå             | on\nden            | it, this/that\nmed            | with\nvar            | where, each\nsig            | him(self) etc\nför            | for\nså             | so (also: seed)\ntill           | to\när             | is\nmen            | but\nett            | a\nom             | if; around, about\nhade           | had\nde             | they, these/those\nav             | of\nicke           | not, no\nmig            | me\ndu             | you\nhenne          | her\ndå             | then, when\nsin            | his\nnu             | now\nhar            | have\ninte           | inte någon = no one\nhans           | his\nhonom          | him\nskulle         | 'sake'\nhennes         | her\ndär            | there\nmin            | my\nman            | one (pronoun)\nej             | nor\nvid            | at, by, on (also: vast)\nkunde          | could\nnågot          | some etc\nfrån           | from, off\nut             | out\nnär            | when\nefter          | after, behind\nupp            | up\nvi             | we\ndem            | them\nvara           | be\nvad            | what\növer           | over\nän             | than\ndig            | you\nkan            | can\nsina           | his\nhär            | here\nha             | have\nmot            | towards\nalla           | all\nunder          | under (also: wonder)\nnågon          | some etc\neller          | or (else)\nallt           | all\nmycket         | much\nsedan          | since\nju             | why\ndenna          | this/that\nsjälv          | myself, yourself etc\ndetta          | this/that\nåt             | to\nutan           | without\nvarit          | was\nhur            | how\ningen          | no\nmitt           | my\nni             | you\nbli            | to be, become\nblev           | from bli\noss            | us\ndin            | thy\ndessa          | these/those\nnågra          | some etc\nderas          | their\nblir           | from bli\nmina           | my\nsamma          | (the) same\nvilken         | who, that\ner             | you, your\nsådan          | such a\nvår            | our\nblivit         | from bli\ndess           | its\ninom           | within\nmellan         | between\nsådant         | such a\nvarför         | why\nvarje          | each\nvilka          | who, that\nditt           | thy\nvem            | who\nvilket         | who, that\nsitta          | his\nsådana         | such a\nvart           | each\ndina           | thy\nvars           | whose\nvårt           | our\nvåra           | our\nert            | your\nera            | your\nvilkas         | whose\n")

class SearchSwedish(SearchLanguage):
    lang = 'sv'
    language_name = 'Swedish'
    js_stemmer_rawcode = 'swedish-stemmer.js'
    stopwords = swedish_stopwords

    def init(self, options: dict) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.stemmer = snowballstemmer.stemmer('swedish')

    def stem(self, word: str) -> str:
        if False:
            print('Hello World!')
        return self.stemmer.stemWord(word.lower())