from module.network import RequestContent

def search_url(e):
    if False:
        return 10
    return f'https://api.bgm.tv/search/subject/{e}?responseGroup=large'

def bgm_parser(title):
    if False:
        i = 10
        return i + 15
    url = search_url(title)
    with RequestContent() as req:
        contents = req.get_json(url)
        if contents:
            return contents[0]
        else:
            return None