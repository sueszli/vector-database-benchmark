class Solution(object):

    def uniqueMorseRepresentations(self, words):
        if False:
            i = 10
            return i + 15
        '\n        :type words: List[str]\n        :rtype: int\n        '
        MORSE = ['.-', '-...', '-.-.', '-..', '.', '..-.', '--.', '....', '..', '.---', '-.-', '.-..', '--', '-.', '---', '.--.', '--.-', '.-.', '...', '-', '..-', '...-', '.--', '-..-', '-.--', '--..']
        lookup = {''.join((MORSE[ord(c) - ord('a')] for c in word)) for word in words}
        return len(lookup)