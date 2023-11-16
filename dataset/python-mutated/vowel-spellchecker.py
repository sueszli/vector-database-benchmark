class Solution(object):

    def spellchecker(self, wordlist, queries):
        if False:
            i = 10
            return i + 15
        '\n        :type wordlist: List[str]\n        :type queries: List[str]\n        :rtype: List[str]\n        '
        vowels = set(['a', 'e', 'i', 'o', 'u'])

        def todev(word):
            if False:
                while True:
                    i = 10
            return ''.join(('*' if c.lower() in vowels else c.lower() for c in word))
        words = set(wordlist)
        caps = {}
        vows = {}
        for word in wordlist:
            caps.setdefault(word.lower(), word)
            vows.setdefault(todev(word), word)

        def check(query):
            if False:
                for i in range(10):
                    print('nop')
            if query in words:
                return query
            lower = query.lower()
            if lower in caps:
                return caps[lower]
            devow = todev(lower)
            if devow in vows:
                return vows[devow]
            return ''
        return map(check, queries)