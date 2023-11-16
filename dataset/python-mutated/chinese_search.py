def setup(app):
    if False:
        i = 10
        return i + 15
    import sphinx.search as search
    import zh
    search.languages['zh_CN'] = zh.SearchChinese