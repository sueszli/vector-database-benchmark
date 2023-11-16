def do_not_use_static_url(request):
    if False:
        print('Hello World!')

    def exception():
        if False:
            while True:
                i = 10
        raise Exception('Do not use STATIC_URL in templates. Use the {% static %} templatetag (or {% versioned_static %} within admin templates) instead.')
    return {'STATIC_URL': lambda : exception()}