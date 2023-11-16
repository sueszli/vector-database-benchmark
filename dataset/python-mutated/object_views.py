import logging
from copy import deepcopy
from django.contrib import messages
from django.db import transaction
from django.db.models import ProtectedError
from django.shortcuts import redirect, render
from django.urls import reverse
from django.utils.html import escape
from django.utils.safestring import mark_safe
from extras.signals import clear_webhooks
from utilities.error_handlers import handle_protectederror
from utilities.exceptions import AbortRequest, PermissionsViolation
from utilities.forms import ConfirmationForm, restrict_form_fields
from utilities.htmx import is_htmx
from utilities.permissions import get_permission_for_model
from utilities.utils import get_viewname, normalize_querydict, prepare_cloned_fields
from utilities.views import GetReturnURLMixin
from .base import BaseObjectView
from .mixins import ActionsMixin, TableMixin
from .utils import get_prerequisite_model
__all__ = ('ComponentCreateView', 'ObjectChildrenView', 'ObjectDeleteView', 'ObjectEditView', 'ObjectView')

class ObjectView(BaseObjectView):
    """
    Retrieve a single object for display.

    Note: If `template_name` is not specified, it will be determined automatically based on the queryset model.

    Attributes:
        tab: A ViewTab instance for the view
    """
    tab = None

    def get_required_permission(self):
        if False:
            i = 10
            return i + 15
        return get_permission_for_model(self.queryset.model, 'view')

    def get_template_name(self):
        if False:
            i = 10
            return i + 15
        "\n        Return self.template_name if defined. Otherwise, dynamically resolve the template name using the queryset\n        model's `app_label` and `model_name`.\n        "
        if self.template_name is not None:
            return self.template_name
        model_opts = self.queryset.model._meta
        return f'{model_opts.app_label}/{model_opts.model_name}.html'

    def get(self, request, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        GET request handler. `*args` and `**kwargs` are passed to identify the object being queried.\n\n        Args:\n            request: The current request\n        '
        instance = self.get_object(**kwargs)
        return render(request, self.get_template_name(), {'object': instance, 'tab': self.tab, **self.get_extra_context(request, instance)})

class ObjectChildrenView(ObjectView, ActionsMixin, TableMixin):
    """
    Display a table of child objects associated with the parent object. For example, NetBox uses this to display
    the set of child IP addresses within a parent prefix.

    Attributes:
        child_model: The model class which represents the child objects
        table: The django-tables2 Table class used to render the child objects list
        filterset: A django-filter FilterSet that is applied to the queryset
        actions: Supported actions for the model. When adding custom actions, bulk action names must
            be prefixed with `bulk_`. Default actions: add, import, export, bulk_edit, bulk_delete
        action_perms: A dictionary mapping supported actions to a set of permissions required for each
    """
    child_model = None
    table = None
    filterset = None

    def get_children(self, request, parent):
        if False:
            print('Hello World!')
        '\n        Return a QuerySet of child objects.\n\n        Args:\n            request: The current request\n            parent: The parent object\n        '
        raise NotImplementedError(f'{self.__class__.__name__} must implement get_children()')

    def prep_table_data(self, request, queryset, parent):
        if False:
            for i in range(10):
                print('nop')
        '\n        Provides a hook for subclassed views to modify data before initializing the table.\n\n        Args:\n            request: The current request\n            queryset: The filtered queryset of child objects\n            parent: The parent object\n        '
        return queryset

    def get(self, request, *args, **kwargs):
        if False:
            return 10
        '\n        GET handler for rendering child objects.\n        '
        instance = self.get_object(**kwargs)
        child_objects = self.get_children(request, instance)
        if self.filterset:
            child_objects = self.filterset(request.GET, child_objects, request=request).qs
        actions = self.get_permitted_actions(request.user, model=self.child_model)
        has_bulk_actions = any([a.startswith('bulk_') for a in actions])
        table_data = self.prep_table_data(request, child_objects, instance)
        table = self.get_table(table_data, request, has_bulk_actions)
        if is_htmx(request):
            return render(request, 'htmx/table.html', {'object': instance, 'table': table})
        return render(request, self.get_template_name(), {'object': instance, 'child_model': self.child_model, 'base_template': f'{instance._meta.app_label}/{instance._meta.model_name}.html', 'table': table, 'table_config': f'{table.name}_config', 'actions': actions, 'tab': self.tab, 'return_url': request.get_full_path(), **self.get_extra_context(request, instance)})

class ObjectEditView(GetReturnURLMixin, BaseObjectView):
    """
    Create or edit a single object.

    Attributes:
        form: The form used to create or edit the object
    """
    template_name = 'generic/object_edit.html'
    form = None

    def dispatch(self, request, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        self._permission_action = 'change' if kwargs else 'add'
        return super().dispatch(request, *args, **kwargs)

    def get_required_permission(self):
        if False:
            i = 10
            return i + 15
        return get_permission_for_model(self.queryset.model, self._permission_action)

    def get_object(self, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Return an object for editing. If no keyword arguments have been specified, this will be a new instance.\n        '
        if not kwargs:
            return self.queryset.model()
        return super().get_object(**kwargs)

    def alter_object(self, obj, request, url_args, url_kwargs):
        if False:
            print('Hello World!')
        '\n        Provides a hook for views to modify an object before it is processed. For example, a parent object can be\n        defined given some parameter from the request URL.\n\n        Args:\n            obj: The object being edited\n            request: The current request\n            url_args: URL path args\n            url_kwargs: URL path kwargs\n        '
        return obj

    def get_extra_addanother_params(self, request):
        if False:
            print('Hello World!')
        '\n        Return a dictionary of extra parameters to use on the Add Another button.\n        '
        return {}

    def get(self, request, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        GET request handler.\n\n        Args:\n            request: The current request\n        '
        obj = self.get_object(**kwargs)
        obj = self.alter_object(obj, request, args, kwargs)
        model = self.queryset.model
        initial_data = normalize_querydict(request.GET)
        form = self.form(instance=obj, initial=initial_data)
        restrict_form_fields(form, request.user)
        if is_htmx(request):
            return render(request, 'htmx/form.html', {'form': form})
        return render(request, self.template_name, {'model': model, 'object': obj, 'form': form, 'return_url': self.get_return_url(request, obj), 'prerequisite_model': get_prerequisite_model(self.queryset), **self.get_extra_context(request, obj)})

    def post(self, request, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        POST request handler.\n\n        Args:\n            request: The current request\n        '
        logger = logging.getLogger('netbox.views.ObjectEditView')
        obj = self.get_object(**kwargs)
        if obj.pk and hasattr(obj, 'snapshot'):
            obj.snapshot()
        obj = self.alter_object(obj, request, args, kwargs)
        form = self.form(data=request.POST, files=request.FILES, instance=obj)
        restrict_form_fields(form, request.user)
        if form.is_valid():
            logger.debug('Form validation was successful')
            try:
                with transaction.atomic():
                    object_created = form.instance.pk is None
                    obj = form.save()
                    if not self.queryset.filter(pk=obj.pk).exists():
                        raise PermissionsViolation()
                msg = '{} {}'.format('Created' if object_created else 'Modified', self.queryset.model._meta.verbose_name)
                logger.info(f'{msg} {obj} (PK: {obj.pk})')
                if hasattr(obj, 'get_absolute_url'):
                    msg = mark_safe(f'{msg} <a href="{obj.get_absolute_url()}">{escape(obj)}</a>')
                else:
                    msg = f'{msg} {obj}'
                messages.success(request, msg)
                if '_addanother' in request.POST:
                    redirect_url = request.path
                    params = prepare_cloned_fields(obj)
                    params.update(self.get_extra_addanother_params(request))
                    if params:
                        if 'return_url' in request.GET:
                            params['return_url'] = request.GET.get('return_url')
                        redirect_url += f'?{params.urlencode()}'
                    return redirect(redirect_url)
                return_url = self.get_return_url(request, obj)
                return redirect(return_url)
            except (AbortRequest, PermissionsViolation) as e:
                logger.debug(e.message)
                form.add_error(None, e.message)
                clear_webhooks.send(sender=self)
        else:
            logger.debug('Form validation failed')
        return render(request, self.template_name, {'object': obj, 'form': form, 'return_url': self.get_return_url(request, obj), **self.get_extra_context(request, obj)})

class ObjectDeleteView(GetReturnURLMixin, BaseObjectView):
    """
    Delete a single object.
    """
    template_name = 'generic/object_delete.html'

    def get_required_permission(self):
        if False:
            for i in range(10):
                print('nop')
        return get_permission_for_model(self.queryset.model, 'delete')

    def get(self, request, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        GET request handler.\n\n        Args:\n            request: The current request\n        '
        obj = self.get_object(**kwargs)
        form = ConfirmationForm(initial=request.GET)
        if is_htmx(request):
            viewname = get_viewname(self.queryset.model, action='delete')
            form_url = reverse(viewname, kwargs={'pk': obj.pk})
            return render(request, 'htmx/delete_form.html', {'object': obj, 'object_type': self.queryset.model._meta.verbose_name, 'form': form, 'form_url': form_url, **self.get_extra_context(request, obj)})
        return render(request, self.template_name, {'object': obj, 'form': form, 'return_url': self.get_return_url(request, obj), **self.get_extra_context(request, obj)})

    def post(self, request, *args, **kwargs):
        if False:
            return 10
        '\n        POST request handler.\n\n        Args:\n            request: The current request\n        '
        logger = logging.getLogger('netbox.views.ObjectDeleteView')
        obj = self.get_object(**kwargs)
        form = ConfirmationForm(request.POST)
        if hasattr(obj, 'snapshot'):
            obj.snapshot()
        if form.is_valid():
            logger.debug('Form validation was successful')
            try:
                obj.delete()
            except ProtectedError as e:
                logger.info('Caught ProtectedError while attempting to delete object')
                handle_protectederror([obj], request, e)
                return redirect(obj.get_absolute_url())
            except AbortRequest as e:
                logger.debug(e.message)
                messages.error(request, mark_safe(e.message))
                return redirect(obj.get_absolute_url())
            msg = 'Deleted {} {}'.format(self.queryset.model._meta.verbose_name, obj)
            logger.info(msg)
            messages.success(request, msg)
            return_url = form.cleaned_data.get('return_url')
            if return_url and return_url.startswith('/'):
                return redirect(return_url)
            return redirect(self.get_return_url(request, obj))
        else:
            logger.debug('Form validation failed')
        return render(request, self.template_name, {'object': obj, 'form': form, 'return_url': self.get_return_url(request, obj), **self.get_extra_context(request, obj)})

class ComponentCreateView(GetReturnURLMixin, BaseObjectView):
    """
    Add one or more components (e.g. interfaces, console ports, etc.) to a Device or VirtualMachine.
    """
    template_name = 'generic/object_edit.html'
    form = None
    model_form = None

    def get_required_permission(self):
        if False:
            print('Hello World!')
        return get_permission_for_model(self.queryset.model, 'add')

    def alter_object(self, instance, request):
        if False:
            print('Hello World!')
        return instance

    def initialize_form(self, request):
        if False:
            return 10
        data = request.POST if request.method == 'POST' else None
        initial_data = normalize_querydict(request.GET)
        form = self.form(data=data, initial=initial_data)
        return form

    def get(self, request):
        if False:
            for i in range(10):
                print('nop')
        form = self.initialize_form(request)
        instance = self.alter_object(self.queryset.model(), request)
        if is_htmx(request):
            return render(request, 'htmx/form.html', {'form': form})
        return render(request, self.template_name, {'object': instance, 'form': form, 'return_url': self.get_return_url(request)})

    def post(self, request):
        if False:
            i = 10
            return i + 15
        logger = logging.getLogger('netbox.views.ComponentCreateView')
        form = self.initialize_form(request)
        instance = self.alter_object(self.queryset.model(), request)
        form.instance._replicated_base = hasattr(self.form, 'replication_fields')
        if form.is_valid():
            new_components = []
            data = deepcopy(request.POST)
            pattern_count = len(form.cleaned_data[self.form.replication_fields[0]])
            for i in range(pattern_count):
                for field_name in self.form.replication_fields:
                    if form.cleaned_data.get(field_name):
                        data[field_name] = form.cleaned_data[field_name][i]
                if hasattr(form, 'get_iterative_data'):
                    data.update(form.get_iterative_data(i))
                component_form = self.model_form(data)
                if component_form.is_valid():
                    new_components.append(component_form)
                else:
                    form.errors.update(component_form.errors)
                    break
            if not form.errors and (not component_form.errors):
                try:
                    with transaction.atomic():
                        new_objs = []
                        for component_form in new_components:
                            obj = component_form.save()
                            new_objs.append(obj)
                        if self.queryset.filter(pk__in=[obj.pk for obj in new_objs]).count() != len(new_objs):
                            raise PermissionsViolation
                        messages.success(request, 'Added {} {}'.format(len(new_components), self.queryset.model._meta.verbose_name_plural))
                        if '_addanother' in request.POST:
                            return redirect(request.get_full_path())
                        else:
                            return redirect(self.get_return_url(request))
                except (AbortRequest, PermissionsViolation) as e:
                    logger.debug(e.message)
                    form.add_error(None, e.message)
                    clear_webhooks.send(sender=self)
        return render(request, self.template_name, {'object': instance, 'form': form, 'return_url': self.get_return_url(request)})