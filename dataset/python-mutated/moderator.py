import inspect
import warnings

def use_draft(request):
    if False:
        print('Hello World!')
    warnings.warn(f'{inspect.stack()[0][3]} is deprecated and will be removed in django CMS 4.1', DeprecationWarning, stacklevel=2)
    return _use_draft(request)

def _use_draft(request):
    if False:
        for i in range(10):
            print('nop')
    is_staff = request.user.is_authenticated and request.user.is_staff
    return is_staff and (not request.session.get('cms_preview'))

def get_model_queryset(model, request=None):
    if False:
        while True:
            i = 10
    'Decision function used in frontend - says which model should be used.\n    Public models are used unless looking at preview or edit versions of the page.\n    '
    warnings.warn(f'{inspect.stack()[0][3]} is deprecated and will be removed in django CMS 4.1', DeprecationWarning, stacklevel=2)
    if request and _use_draft(request):
        return model.objects.drafts()
    return model.objects.public()

def get_title_queryset(request=None):
    if False:
        while True:
            i = 10
    from cms.models import Title
    warnings.warn(f'{inspect.stack()[0][3]} is deprecated and will be removed in django CMS 4.1', DeprecationWarning, stacklevel=2)
    return Title.objects.all()

def get_cmsplugin_queryset(request=None):
    if False:
        return 10
    from cms.models import CMSPlugin
    warnings.warn(f'{inspect.stack()[0][3]} is deprecated and will be removed in django CMS 4.1', DeprecationWarning, stacklevel=2)
    return CMSPlugin.objects.all()