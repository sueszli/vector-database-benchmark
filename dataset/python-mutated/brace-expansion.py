import itertools

class Solution(object):

    def expand(self, S):
        if False:
            print('Hello World!')
        '\n        :type S: str\n        :rtype: List[str]\n        '

        def form_words(options):
            if False:
                for i in range(10):
                    print('nop')
            words = map(''.join, itertools.product(*options))
            words.sort()
            return words

        def generate_option(expr, i):
            if False:
                for i in range(10):
                    print('nop')
            option_set = set()
            while i[0] != len(expr) and expr[i[0]] != '}':
                i[0] += 1
                for option in generate_words(expr, i):
                    option_set.add(option)
            i[0] += 1
            option = list(option_set)
            option.sort()
            return option

        def generate_words(expr, i):
            if False:
                print('Hello World!')
            options = []
            while i[0] != len(expr) and expr[i[0]] not in ',}':
                tmp = []
                if expr[i[0]] not in '{,}':
                    tmp.append(expr[i[0]])
                    i[0] += 1
                elif expr[i[0]] == '{':
                    tmp = generate_option(expr, i)
                options.append(tmp)
            return form_words(options)
        return generate_words(S, [0])

class Solution2(object):

    def expand(self, S):
        if False:
            print('Hello World!')
        '\n        :type S: str\n        :rtype: List[str]\n        '

        def form_words(options):
            if False:
                print('Hello World!')
            words = []
            total = 1
            for opt in options:
                total *= len(opt)
            for i in xrange(total):
                tmp = []
                for opt in reversed(options):
                    (i, c) = divmod(i, len(opt))
                    tmp.append(opt[c])
                tmp.reverse()
                words.append(''.join(tmp))
            words.sort()
            return words

        def generate_option(expr, i):
            if False:
                i = 10
                return i + 15
            option_set = set()
            while i[0] != len(expr) and expr[i[0]] != '}':
                i[0] += 1
                for option in generate_words(expr, i):
                    option_set.add(option)
            i[0] += 1
            option = list(option_set)
            option.sort()
            return option

        def generate_words(expr, i):
            if False:
                while True:
                    i = 10
            options = []
            while i[0] != len(expr) and expr[i[0]] not in ',}':
                tmp = []
                if expr[i[0]] not in '{,}':
                    tmp.append(expr[i[0]])
                    i[0] += 1
                elif expr[i[0]] == '{':
                    tmp = generate_option(expr, i)
                options.append(tmp)
            return form_words(options)
        return generate_words(S, [0])