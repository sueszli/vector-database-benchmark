def setup(app):
    if False:
        return 10
    app.add_crossref_type(directivename='setting', rolename='setting', indextemplate='pair: %s; setting')
    app.add_crossref_type(directivename='templatetag', rolename='ttag', indextemplate='pair: %s; template tag')