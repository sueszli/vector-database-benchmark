class BrowserHistory(object):

    def __init__(self, homepage):
        if False:
            print('Hello World!')
        '\n        :type homepage: str\n        '
        self.__history = [homepage]
        self.__curr = 0

    def visit(self, url):
        if False:
            while True:
                i = 10
        '\n        :type url: str\n        :rtype: None\n        '
        while len(self.__history) > self.__curr + 1:
            self.__history.pop()
        self.__history.append(url)
        self.__curr += 1

    def back(self, steps):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type steps: int\n        :rtype: str\n        '
        self.__curr = max(self.__curr - steps, 0)
        return self.__history[self.__curr]

    def forward(self, steps):
        if False:
            i = 10
            return i + 15
        '\n        :type steps: int\n        :rtype: str\n        '
        self.__curr = min(self.__curr + steps, len(self.__history) - 1)
        return self.__history[self.__curr]