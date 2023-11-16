"""French search language: includes the JS French stemmer."""
from __future__ import annotations
from typing import TYPE_CHECKING, Dict
import snowballstemmer
from sphinx.search import SearchLanguage, parse_stop_word
french_stopwords = parse_stop_word("\n| source: http://snowball.tartarus.org/algorithms/french/stop.txt\nau             |  a + le\naux            |  a + les\navec           |  with\nce             |  this\nces            |  these\ndans           |  with\nde             |  of\ndes            |  de + les\ndu             |  de + le\nelle           |  she\nen             |  `of them' etc\net             |  and\neux            |  them\nil             |  he\nje             |  I\nla             |  the\nle             |  the\nleur           |  their\nlui            |  him\nma             |  my (fem)\nmais           |  but\nme             |  me\nmême           |  same; as in moi-même (myself) etc\nmes            |  me (pl)\nmoi            |  me\nmon            |  my (masc)\nne             |  not\nnos            |  our (pl)\nnotre          |  our\nnous           |  we\non             |  one\nou             |  where\npar            |  by\npas            |  not\npour           |  for\nqu             |  que before vowel\nque            |  that\nqui            |  who\nsa             |  his, her (fem)\nse             |  oneself\nses            |  his (pl)\nson            |  his, her (masc)\nsur            |  on\nta             |  thy (fem)\nte             |  thee\ntes            |  thy (pl)\ntoi            |  thee\nton            |  thy (masc)\ntu             |  thou\nun             |  a\nune            |  a\nvos            |  your (pl)\nvotre          |  your\nvous           |  you\n\n               |  single letter forms\n\nc              |  c'\nd              |  d'\nj              |  j'\nl              |  l'\nà              |  to, at\nm              |  m'\nn              |  n'\ns              |  s'\nt              |  t'\ny              |  there\n\n               | forms of être (not including the infinitive):\nété\nétée\nétées\nétés\nétant\nsuis\nes\nest\nsommes\nêtes\nsont\nserai\nseras\nsera\nserons\nserez\nseront\nserais\nserait\nserions\nseriez\nseraient\nétais\nétait\nétions\nétiez\nétaient\nfus\nfut\nfûmes\nfûtes\nfurent\nsois\nsoit\nsoyons\nsoyez\nsoient\nfusse\nfusses\nfût\nfussions\nfussiez\nfussent\n\n               | forms of avoir (not including the infinitive):\nayant\neu\neue\neues\neus\nai\nas\navons\navez\nont\naurai\nauras\naura\naurons\naurez\nauront\naurais\naurait\naurions\nauriez\nauraient\navais\navait\navions\naviez\navaient\neut\neûmes\neûtes\neurent\naie\naies\nait\nayons\nayez\naient\neusse\neusses\neût\neussions\neussiez\neussent\n\n               | Later additions (from Jean-Christophe Deschamps)\nceci           |  this\ncela           |  that (added 11 Apr 2012. Omission reported by Adrien Grand)\ncelà           |  that (incorrect, though common)\ncet            |  this\ncette          |  this\nici            |  here\nils            |  they\nles            |  the (pl)\nleurs          |  their (pl)\nquel           |  which\nquels          |  which\nquelle         |  which\nquelles        |  which\nsans           |  without\nsoi            |  oneself\n")

class SearchFrench(SearchLanguage):
    lang = 'fr'
    language_name = 'French'
    js_stemmer_rawcode = 'french-stemmer.js'
    stopwords = french_stopwords

    def init(self, options: dict) -> None:
        if False:
            i = 10
            return i + 15
        self.stemmer = snowballstemmer.stemmer('french')

    def stem(self, word: str) -> str:
        if False:
            print('Hello World!')
        return self.stemmer.stemWord(word.lower())