from module.rss import RSSEngine

def update_main_rss(rss_link: str):
    if False:
        return 10
    with RSSEngine() as engine:
        engine.add_rss(rss_link, 'main', True)