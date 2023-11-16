"""Hungarian search language: includes the JS Hungarian stemmer."""
from __future__ import annotations
from typing import TYPE_CHECKING, Dict
import snowballstemmer
from sphinx.search import SearchLanguage, parse_stop_word
hungarian_stopwords = parse_stop_word('\n| source: http://snowball.tartarus.org/algorithms/hungarian/stop.txt\n| prepared by Anna Tordai\na\nahogy\nahol\naki\nakik\nakkor\nalatt\náltal\náltalában\namely\namelyek\namelyekben\namelyeket\namelyet\namelynek\nami\namit\namolyan\namíg\namikor\nát\nabban\nahhoz\nannak\narra\narról\naz\nazok\nazon\nazt\nazzal\nazért\naztán\nazután\nazonban\nbár\nbe\nbelül\nbenne\ncikk\ncikkek\ncikkeket\ncsak\nde\ne\neddig\negész\negy\negyes\negyetlen\negyéb\negyik\negyre\nekkor\nel\nelég\nellen\nelő\nelőször\nelőtt\nelső\nén\néppen\nebben\nehhez\nemilyen\nennek\nerre\nez\nezt\nezek\nezen\nezzel\nezért\nés\nfel\nfelé\nhanem\nhiszen\nhogy\nhogyan\nigen\nígy\nilletve\nill.\nill\nilyen\nilyenkor\nison\nismét\nitt\njó\njól\njobban\nkell\nkellett\nkeresztül\nkeressünk\nki\nkívül\nközött\nközül\nlegalább\nlehet\nlehetett\nlegyen\nlenne\nlenni\nlesz\nlett\nmaga\nmagát\nmajd\nmajd\nmár\nmás\nmásik\nmeg\nmég\nmellett\nmert\nmely\nmelyek\nmi\nmit\nmíg\nmiért\nmilyen\nmikor\nminden\nmindent\nmindenki\nmindig\nmint\nmintha\nmivel\nmost\nnagy\nnagyobb\nnagyon\nne\nnéha\nnekem\nneki\nnem\nnéhány\nnélkül\nnincs\nolyan\nott\nössze\nő\nők\nőket\npedig\npersze\nrá\ns\nsaját\nsem\nsemmi\nsok\nsokat\nsokkal\nszámára\nszemben\nszerint\nszinte\ntalán\ntehát\nteljes\ntovább\ntovábbá\ntöbb\núgy\nugyanis\núj\nújabb\nújra\nután\nutána\nutolsó\nvagy\nvagyis\nvalaki\nvalami\nvalamint\nvaló\nvagyok\nvan\nvannak\nvolt\nvoltam\nvoltak\nvoltunk\nvissza\nvele\nviszont\nvolna\n')

class SearchHungarian(SearchLanguage):
    lang = 'hu'
    language_name = 'Hungarian'
    js_stemmer_rawcode = 'hungarian-stemmer.js'
    stopwords = hungarian_stopwords

    def init(self, options: dict) -> None:
        if False:
            i = 10
            return i + 15
        self.stemmer = snowballstemmer.stemmer('hungarian')

    def stem(self, word: str) -> str:
        if False:
            print('Hello World!')
        return self.stemmer.stemWord(word.lower())