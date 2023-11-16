class HtmlParser(object):

    def getUrls(self, url):
        if False:
            i = 10
            return i + 15
        '\n       :type url: str\n       :rtype List[str]\n       '
        pass

class Solution(object):

    def crawl(self, startUrl, htmlParser):
        if False:
            i = 10
            return i + 15
        '\n        :type startUrl: str\n        :type htmlParser: HtmlParser\n        :rtype: List[str]\n        '
        SCHEME = 'http://'

        def hostname(url):
            if False:
                return 10
            pos = url.find('/', len(SCHEME))
            if pos == -1:
                return url
            return url[:pos]
        result = [startUrl]
        lookup = set(result)
        for from_url in result:
            name = hostname(from_url)
            for to_url in htmlParser.getUrls(from_url):
                if to_url not in lookup and name == hostname(to_url):
                    result.append(to_url)
                    lookup.add(to_url)
        return result