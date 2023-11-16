import re
from pprint import pprint

class PhraseFilter:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.phrases = {}

    def run(self, text, storyid):
        if False:
            i = 10
            return i + 15
        chunks = self.chunk(text)
        self.count_phrases(chunks, storyid)

    def print_phrases(self):
        if False:
            return 10
        pprint(self.phrases)

    def get_phrases(self):
        if False:
            return 10
        return self.phrases.keys()

    def chunk(self, text):
        if False:
            print('Hello World!')
        chunks = [t.strip() for t in re.split('[^a-zA-Z-]+', text) if t]
        return chunks

    def _lowercase(self, chunks):
        if False:
            i = 10
            return i + 15
        return [c.lower() for c in chunks]

    def count_phrases(self, chunks, storyid):
        if False:
            return 10
        for l in range(1, len(chunks) + 1):
            combinations = self._get_combinations(chunks, l)
            for phrase in combinations:
                if phrase not in self.phrases:
                    self.phrases[phrase] = []
                if storyid not in self.phrases[phrase]:
                    self.phrases[phrase].append(storyid)

    def _get_combinations(self, chunks, length):
        if False:
            print('Hello World!')
        combinations = []
        for (i, chunk) in enumerate(chunks):
            combination = []
            for l in range(length):
                if i + l < len(chunks):
                    combination.append(chunks[i + l])
            combinations.append(' '.join(combination))
        return combinations

    def pare_phrases(self):
        if False:
            while True:
                i = 10
        for (phrase, counts) in self.phrases.items():
            if len(counts) < 2:
                del self.phrases[phrase]
                continue
            if len(phrase) < 4:
                del self.phrases[phrase]
                continue
        for phrase in self.phrases.keys():
            for phrase2 in self.phrases.keys():
                if phrase in self.phrases and len(phrase2) > len(phrase) and (phrase in phrase2) and (phrase != phrase2):
                    del self.phrases[phrase]
if __name__ == '__main__':
    phrasefilter = PhraseFilter()
    phrasefilter.run('House of the Day: 123 Atlantic Ave. #3', 1)
    phrasefilter.run('House of the Day: 456 Plankton St. #3', 4)
    phrasefilter.run('Coop of the Day: 321 Pacific St.', 2)
    phrasefilter.run('Streetlevel: 393 Pacific St.', 11)
    phrasefilter.run('Coop of the Day: 456 Jefferson Ave.', 3)
    phrasefilter.run('Extra, Extra', 5)
    phrasefilter.run('Extra, Extra', 6)
    phrasefilter.run('Early Addition', 7)
    phrasefilter.run('Early Addition', 8)
    phrasefilter.run('Development Watch', 9)
    phrasefilter.run('Streetlevel', 10)
    phrasefilter.pare_phrases()
    phrasefilter.print_phrases()