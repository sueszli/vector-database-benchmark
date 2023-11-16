def find_browser():
    if False:
        return 10
    'Find the default browser if possible and if compatible.'
    import webbrowser
    incompatible_browsers = {'www-browser', 'links', 'elinks', 'lynx', 'w3m', 'links2', 'links-g'}
    try:
        browser = webbrowser.get()
    except webbrowser.Error:
        return None
    return None if browser.name in incompatible_browsers else browser