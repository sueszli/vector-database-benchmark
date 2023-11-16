from pyramid.httpexceptions import HTTPGone, HTTPMovedPermanently, HTTPNotFound
from pyramid.response import Response
from pyramid.view import forbidden_view_config, view_config
from trove_classifiers import sorted_classifiers
from warehouse.classifiers.models import Classifier

def _exc_with_message(exc, message):
    if False:
        i = 10
        return i + 15
    resp = exc(message)
    resp.status = f'{resp.status_code} {message}'
    return resp

@view_config(route_name='legacy.api.pypi.file_upload', require_csrf=False, require_methods=['POST'])
@view_config(route_name='legacy.api.pypi.submit', require_csrf=False, require_methods=['POST'])
@view_config(route_name='legacy.api.pypi.submit_pkg_info', require_csrf=False, require_methods=['POST'])
@view_config(route_name='legacy.api.pypi.doc_upload', require_csrf=False, require_methods=['POST'])
def forklifted(request):
    if False:
        for i in range(10):
            print('nop')
    settings = request.registry.settings
    domain = settings.get('forklift.domain', settings.get('warehouse.domain', request.domain))
    information_url = 'TODO'
    return _exc_with_message(HTTPGone, 'This API has moved to https://{}/legacy/. See {} for more information.'.format(domain, information_url))

@view_config(route_name='legacy.api.pypi.doap')
def doap(request):
    if False:
        return 10
    return _exc_with_message(HTTPGone, 'DOAP is no longer supported.')

@forbidden_view_config(request_param=':action')
def forbidden_legacy(exc, request):
    if False:
        i = 10
        return i + 15
    return exc

@view_config(route_name='legacy.api.pypi.list_classifiers')
def list_classifiers(request):
    if False:
        print('Hello World!')
    return Response(text='\n'.join(sorted_classifiers), content_type='text/plain; charset=utf-8')

@view_config(route_name='legacy.api.pypi.search')
def search(request):
    if False:
        print('Hello World!')
    return HTTPMovedPermanently(request.route_path('search', _query={'q': request.params.get('term')}))

@view_config(route_name='legacy.api.pypi.browse')
def browse(request):
    if False:
        i = 10
        return i + 15
    classifier_id = request.params.get('c')
    if not classifier_id:
        raise HTTPNotFound
    try:
        int(classifier_id)
    except ValueError:
        raise HTTPNotFound
    classifier = request.db.get(Classifier, classifier_id)
    if not classifier:
        raise HTTPNotFound
    return HTTPMovedPermanently(request.route_path('search', _query={'c': classifier.classifier}))

@view_config(route_name='legacy.api.pypi.files')
def files(request):
    if False:
        i = 10
        return i + 15
    name = request.params.get('name')
    version = request.params.get('version')
    if not name or not version:
        raise HTTPNotFound
    return HTTPMovedPermanently(request.route_path('packaging.release', name=name, version=version, _anchor='files'))

@view_config(route_name='legacy.api.pypi.display')
def display(request):
    if False:
        while True:
            i = 10
    name = request.params.get('name')
    version = request.params.get('version')
    if not name:
        raise HTTPNotFound
    if version:
        return HTTPMovedPermanently(request.route_path('packaging.release', name=name, version=version))
    return HTTPMovedPermanently(request.route_path('packaging.project', name=name))