__all__ = ['WordsSearch']
__author__ = 'Lin Zhijun'
__date__ = '2020.05.16'

class TrieNode:

    def __init__(self):
        if False:
            return 10
        self.Index = 0
        self.Index = 0
        self.Layer = 0
        self.End = False
        self.Char = ''
        self.Results = []
        self.m_values = {}
        self.Failure = None
        self.Parent = None

    def Add(self, c):
        if False:
            return 10
        if c in self.m_values:
            return self.m_values[c]
        node = TrieNode()
        node.Parent = self
        node.Char = c
        self.m_values[c] = node
        return node

    def SetResults(self, index):
        if False:
            i = 10
            return i + 15
        if self.End == False:
            self.End = True
        self.Results.append(index)

class TrieNode2:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.End = False
        self.Results = []
        self.m_values = {}
        self.minflag = 65535
        self.maxflag = 0

    def Add(self, c, node3):
        if False:
            return 10
        if self.minflag > c:
            self.minflag = c
        if self.maxflag < c:
            self.maxflag = c
        self.m_values[c] = node3

    def SetResults(self, index):
        if False:
            print('Hello World!')
        if self.End == False:
            self.End = True
        if (index in self.Results) == False:
            self.Results.append(index)

    def HasKey(self, c):
        if False:
            while True:
                i = 10
        return c in self.m_values

    def TryGetValue(self, c):
        if False:
            print('Hello World!')
        if self.minflag <= c and self.maxflag >= c:
            if c in self.m_values:
                return self.m_values[c]
        return None

class WordsSearch:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self._first = {}
        self._keywords = []
        self._indexs = []

    def SetKeywords(self, keywords):
        if False:
            return 10
        self._keywords = keywords
        self._indexs = []
        for i in range(len(keywords)):
            self._indexs.append(i)
        root = TrieNode()
        allNodeLayer = {}
        for i in range(len(self._keywords)):
            p = self._keywords[i]
            nd = root
            for j in range(len(p)):
                nd = nd.Add(ord(p[j]))
                if nd.Layer == 0:
                    nd.Layer = j + 1
                    if nd.Layer in allNodeLayer:
                        allNodeLayer[nd.Layer].append(nd)
                    else:
                        allNodeLayer[nd.Layer] = []
                        allNodeLayer[nd.Layer].append(nd)
            nd.SetResults(i)
        allNode = []
        allNode.append(root)
        for key in allNodeLayer.keys():
            for nd in allNodeLayer[key]:
                allNode.append(nd)
        allNodeLayer = None
        for i in range(len(allNode)):
            if i == 0:
                continue
            nd = allNode[i]
            nd.Index = i
            r = nd.Parent.Failure
            c = nd.Char
            while r != None and (c in r.m_values) == False:
                r = r.Failure
            if r == None:
                nd.Failure = root
            else:
                nd.Failure = r.m_values[c]
                for key2 in nd.Failure.Results:
                    nd.SetResults(key2)
        root.Failure = root
        allNode2 = []
        for i in range(len(allNode)):
            allNode2.append(TrieNode2())
        for i in range(len(allNode2)):
            oldNode = allNode[i]
            newNode = allNode2[i]
            for key in oldNode.m_values:
                index = oldNode.m_values[key].Index
                newNode.Add(key, allNode2[index])
            for index in range(len(oldNode.Results)):
                item = oldNode.Results[index]
                newNode.SetResults(item)
            oldNode = oldNode.Failure
            while oldNode != root:
                for key in oldNode.m_values:
                    if newNode.HasKey(key) == False:
                        index = oldNode.m_values[key].Index
                        newNode.Add(key, allNode2[index])
                for index in range(len(oldNode.Results)):
                    item = oldNode.Results[index]
                    newNode.SetResults(item)
                oldNode = oldNode.Failure
        allNode = None
        root = None
        self._first = allNode2[0]

    def FindFirst(self, text):
        if False:
            i = 10
            return i + 15
        ptr = None
        for index in range(len(text)):
            t = ord(text[index])
            tn = None
            if ptr == None:
                tn = self._first.TryGetValue(t)
            else:
                tn = ptr.TryGetValue(t)
                if tn == None:
                    tn = self._first.TryGetValue(t)
            if tn != None:
                if tn.End:
                    item = tn.Results[0]
                    keyword = self._keywords[item]
                    return {'Keyword': keyword, 'Success': True, 'End': index, 'Start': index + 1 - len(keyword), 'Index': self._indexs[item]}
            ptr = tn
        return None

    def FindAll(self, text):
        if False:
            i = 10
            return i + 15
        ptr = None
        list = []
        for index in range(len(text)):
            t = ord(text[index])
            tn = None
            if ptr == None:
                tn = self._first.TryGetValue(t)
            else:
                tn = ptr.TryGetValue(t)
                if tn == None:
                    tn = self._first.TryGetValue(t)
            if tn != None:
                if tn.End:
                    for j in range(len(tn.Results)):
                        item = tn.Results[j]
                        keyword = self._keywords[item]
                        list.append({'Keyword': keyword, 'Success': True, 'End': index, 'Start': index + 1 - len(keyword), 'Index': self._indexs[item]})
            ptr = tn
        return list

    def ContainsAny(self, text):
        if False:
            for i in range(10):
                print('nop')
        ptr = None
        for index in range(len(text)):
            t = ord(text[index])
            tn = None
            if ptr == None:
                tn = self._first.TryGetValue(t)
            else:
                tn = ptr.TryGetValue(t)
                if tn == None:
                    tn = self._first.TryGetValue(t)
            if tn != None:
                if tn.End:
                    return True
            ptr = tn
        return False

    def Replace(self, text, replaceChar='*'):
        if False:
            return 10
        result = list(text)
        ptr = None
        for i in range(len(text)):
            t = ord(text[i])
            tn = None
            if ptr == None:
                tn = self._first.TryGetValue(t)
            else:
                tn = ptr.TryGetValue(t)
                if tn == None:
                    tn = self._first.TryGetValue(t)
            if tn != None:
                if tn.End:
                    maxLength = len(self._keywords[tn.Results[0]])
                    start = i + 1 - maxLength
                    for j in range(start, i + 1):
                        result[j] = replaceChar
            ptr = tn
        return ''.join(result)
if __name__ == '__main__':
    s = '中国|国人|zg人|乾清宫'
    test = '我是中国人'
    search = WordsSearch()
    search.SetKeywords(s.split('|'))
    print('-----------------------------------  WordsSearch  -----------------------------------')
    print('WordsSearch FindFirst is run.')
    f = search.FindFirst(test)
    if f['Keyword'] != '中国':
        print('WordsSearch FindFirst is error.............................')
    print('WordsSearch FindFirst is run.')
    all = search.FindAll('乾清宫')
    if all[0]['Keyword'] != '乾清宫':
        print('WordsSearch FindFirst is error.............................')
    print('WordsSearch FindAll is run.')
    all = search.FindAll(test)
    if all[0]['Keyword'] != '中国':
        print('WordsSearch FindAll is error.............................')
    if all[1]['Keyword'] != '国人':
        print('WordsSearch FindAll is error.............................')
    if all[0]['Start'] != 2:
        print('WordsSearch FindAll is error.............................')
    if all[0]['End'] != 3:
        print('WordsSearch FindAll is error.............................')
    if len(all) != 2:
        print('WordsSearch FindAll is error.............................')
    print('WordsSearch ContainsAny is run.')
    b = search.ContainsAny(test)
    if b == False:
        print('WordsSearch ContainsAny is error.............................')
    print('WordsSearch Replace  is run.')
    txt = search.Replace(test)
    if txt != '我是***':
        print('WordsSearch Replace  is error.............................')
    print('-----------------------------------  Test End  -----------------------------------')