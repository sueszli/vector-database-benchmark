""" Run scripts to generate docs for Flexx
"""
import genuiclasses
import genexamples
import gencommonast

def init():
    if False:
        print('Hello World!')
    print('GENERATING DOCS ...')
    print('  Generating docs for UI classes.')
    genuiclasses.main()
    print('  Generating examples.')
    genexamples.main()

def clean(app, *args):
    if False:
        print('Hello World!')
    genuiclasses.clean()
    genexamples.clean()

def setup(app):
    if False:
        return 10
    init()
    app.connect('build-finished', clean)