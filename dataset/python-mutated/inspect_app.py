"""
Script that prints out the routes of an App instance.
"""
import argparse
import importlib
import os
import sys
import falcon
from falcon.inspect import inspect_app
from falcon.inspect import inspect_routes
from falcon.inspect import StringVisitor
sys.path.append(os.getcwd())

def make_parser():
    if False:
        i = 10
        return i + 15
    'Create the parser or the application.'
    parser = argparse.ArgumentParser(description='Example: falcon-inspect-app myprogram:app')
    parser.add_argument('-r', '--route_only', action='store_true', help='Prints only the information regarding the routes')
    parser.add_argument('-v', '--verbose', action='store_true', help='More verbose output')
    parser.add_argument('-i', '--internal', action='store_true', help='Print also internal falcon route methods and error handlers')
    parser.add_argument('app_module', help='The module and app to inspect. Example: myapp.somemodule:api')
    return parser

def load_app(parser, args):
    if False:
        i = 10
        return i + 15
    try:
        (module, instance) = args.app_module.split(':', 1)
    except ValueError:
        parser.error('The app_module must include a colon between the module and instance')
    try:
        app = getattr(importlib.import_module(module), instance)
    except AttributeError:
        parser.error('{!r} not found in module {!r}'.format(instance, module))
    if not isinstance(app, falcon.App):
        if callable(app):
            app = app()
            if not isinstance(app, falcon.App):
                parser.error('{} did not return a falcon.App instance'.format(args.app_module))
        else:
            parser.error('The instance must be of falcon.App or be a callable without args that returns falcon.App')
    return app

def route_main():
    if False:
        return 10
    print('The "falcon-print-routes" command is deprecated. Please use "falcon-inspect-app"')
    main()

def main():
    if False:
        while True:
            i = 10
    parser = make_parser()
    args = parser.parse_args()
    app = load_app(parser, args)
    if args.route_only:
        routes = inspect_routes(app)
        visitor = StringVisitor(args.verbose, args.internal)
        for route in routes:
            print(visitor.process(route))
    else:
        print(inspect_app(app).to_string(args.verbose, args.internal))
if __name__ == '__main__':
    main()