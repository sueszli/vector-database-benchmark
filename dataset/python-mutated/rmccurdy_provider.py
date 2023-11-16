from scylla.providers.plain_text_provider import PlainTextProvider

class RmccurdyProvider(PlainTextProvider):

    def urls(self) -> [str]:
        if False:
            for i in range(10):
                print('nop')
        return ['https://www.rmccurdy.com/scripts/proxy/good.txt']