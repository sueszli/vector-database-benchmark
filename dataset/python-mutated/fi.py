"""Finnish search language: includes the JS Finnish stemmer."""
from __future__ import annotations
from typing import TYPE_CHECKING, Dict
import snowballstemmer
from sphinx.search import SearchLanguage, parse_stop_word
finnish_stopwords = parse_stop_word('\n| source: http://snowball.tartarus.org/algorithms/finnish/stop.txt\n| forms of BE\n\nolla\nolen\nolet\non\nolemme\nolette\novat\nole        | negative form\n\noli\nolisi\nolisit\nolisin\nolisimme\nolisitte\nolisivat\nolit\nolin\nolimme\nolitte\nolivat\nollut\nolleet\n\nen         | negation\net\nei\nemme\nette\neivät\n\n|Nom   Gen    Acc    Part   Iness   Elat    Illat  Adess   Ablat   Allat   Ess    Trans\nminä   minun  minut  minua  minussa minusta minuun minulla minulta minulle               | I\nsinä   sinun  sinut  sinua  sinussa sinusta sinuun sinulla sinulta sinulle               | you\nhän    hänen  hänet  häntä  hänessä hänestä häneen hänellä häneltä hänelle               | he she\nme     meidän meidät meitä  meissä  meistä  meihin meillä  meiltä  meille                | we\nte     teidän teidät teitä  teissä  teistä  teihin teillä  teiltä  teille                | you\nhe     heidän heidät heitä  heissä  heistä  heihin heillä  heiltä  heille                | they\n\ntämä   tämän         tätä   tässä   tästä   tähän  tällä   tältä   tälle   tänä   täksi  | this\ntuo    tuon          tuota  tuossa  tuosta  tuohon tuolla  tuolta  tuolle  tuona  tuoksi | that\nse     sen           sitä   siinä   siitä   siihen sillä   siltä   sille   sinä   siksi  | it\nnämä   näiden        näitä  näissä  näistä  näihin näillä  näiltä  näille  näinä  näiksi | these\nnuo    noiden        noita  noissa  noista  noihin noilla  noilta  noille  noina  noiksi | those\nne     niiden        niitä  niissä  niistä  niihin niillä  niiltä  niille  niinä  niiksi | they\n\nkuka   kenen kenet   ketä   kenessä kenestä keneen kenellä keneltä kenelle kenenä keneksi| who\nketkä  keiden ketkä  keitä  keissä  keistä  keihin keillä  keiltä  keille  keinä  keiksi | (pl)\nmikä   minkä minkä   mitä   missä   mistä   mihin  millä   miltä   mille   minä   miksi  | which what\nmitkä                                                                                    | (pl)\n\njoka   jonka         jota   jossa   josta   johon  jolla   jolta   jolle   jona   joksi  | who which\njotka  joiden        joita  joissa  joista  joihin joilla  joilta  joille  joina  joiksi | (pl)\n\n| conjunctions\n\nettä   | that\nja     | and\njos    | if\nkoska  | because\nkuin   | than\nmutta  | but\nniin   | so\nsekä   | and\nsillä  | for\ntai    | or\nvaan   | but\nvai    | or\nvaikka | although\n\n\n| prepositions\n\nkanssa  | with\nmukaan  | according to\nnoin    | about\npoikki  | across\nyli     | over, across\n\n| other\n\nkun    | when\nniin   | so\nnyt    | now\nitse   | self\n')

class SearchFinnish(SearchLanguage):
    lang = 'fi'
    language_name = 'Finnish'
    js_stemmer_rawcode = 'finnish-stemmer.js'
    stopwords = finnish_stopwords

    def init(self, options: dict) -> None:
        if False:
            i = 10
            return i + 15
        self.stemmer = snowballstemmer.stemmer('finnish')

    def stem(self, word: str) -> str:
        if False:
            while True:
                i = 10
        return self.stemmer.stemWord(word.lower())