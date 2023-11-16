from django import forms
from django.utils.translation import gettext_lazy as _
from dcim.models import *
from netbox.forms import NetBoxModelForm
from utilities.forms.fields import DynamicModelChoiceField, DynamicModelMultipleChoiceField, ExpandableNameField
from utilities.forms.widgets import APISelect
from . import model_forms
__all__ = ('ComponentCreateForm', 'ConsolePortCreateForm', 'ConsolePortTemplateCreateForm', 'ConsoleServerPortCreateForm', 'ConsoleServerPortTemplateCreateForm', 'DeviceBayCreateForm', 'DeviceBayTemplateCreateForm', 'FrontPortCreateForm', 'FrontPortTemplateCreateForm', 'InterfaceCreateForm', 'InterfaceTemplateCreateForm', 'InventoryItemCreateForm', 'InventoryItemTemplateCreateForm', 'ModuleBayCreateForm', 'ModuleBayTemplateCreateForm', 'PowerOutletCreateForm', 'PowerOutletTemplateCreateForm', 'PowerPortCreateForm', 'PowerPortTemplateCreateForm', 'RearPortCreateForm', 'RearPortTemplateCreateForm', 'VirtualChassisCreateForm')

class ComponentCreateForm(forms.Form):
    """
    Subclass this form when facilitating the creation of one or more component or component template objects based on
    a name pattern.
    """
    name = ExpandableNameField(label=_('Name'))
    label = ExpandableNameField(label=_('Label'), required=False, help_text=_('Alphanumeric ranges are supported. (Must match the number of objects being created.)'))
    replication_fields = ('name', 'label')

    def clean(self):
        if False:
            print('Hello World!')
        super().clean()
        if not (patterns := self.cleaned_data.get(self.replication_fields[0])):
            return
        pattern_count = len(patterns)
        for field_name in self.replication_fields:
            value_count = len(self.cleaned_data[field_name])
            if self.cleaned_data[field_name] and value_count != pattern_count:
                raise forms.ValidationError({field_name: _('The provided pattern specifies {value_count} values, but {pattern_count} are expected.').format(value_count=value_count, pattern_count=pattern_count)}, code='label_pattern_mismatch')

class ConsolePortTemplateCreateForm(ComponentCreateForm, model_forms.ConsolePortTemplateForm):

    class Meta(model_forms.ConsolePortTemplateForm.Meta):
        exclude = ('name', 'label')

class ConsoleServerPortTemplateCreateForm(ComponentCreateForm, model_forms.ConsoleServerPortTemplateForm):

    class Meta(model_forms.ConsoleServerPortTemplateForm.Meta):
        exclude = ('name', 'label')

class PowerPortTemplateCreateForm(ComponentCreateForm, model_forms.PowerPortTemplateForm):

    class Meta(model_forms.PowerPortTemplateForm.Meta):
        exclude = ('name', 'label')

class PowerOutletTemplateCreateForm(ComponentCreateForm, model_forms.PowerOutletTemplateForm):

    class Meta(model_forms.PowerOutletTemplateForm.Meta):
        exclude = ('name', 'label')

class InterfaceTemplateCreateForm(ComponentCreateForm, model_forms.InterfaceTemplateForm):

    class Meta(model_forms.InterfaceTemplateForm.Meta):
        exclude = ('name', 'label')

class FrontPortTemplateCreateForm(ComponentCreateForm, model_forms.FrontPortTemplateForm):
    rear_port = forms.MultipleChoiceField(choices=[], label=_('Rear ports'), help_text=_('Select one rear port assignment for each front port being created.'), widget=forms.SelectMultiple(attrs={'size': 6}))
    fieldsets = ((None, ('device_type', 'module_type', 'name', 'label', 'type', 'color', 'rear_port', 'description')),)

    class Meta(model_forms.FrontPortTemplateForm.Meta):
        exclude = ('name', 'label', 'rear_port', 'rear_port_position')

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
        if 'device_type' in self.initial or self.data.get('device_type'):
            parent = DeviceType.objects.get(pk=self.initial.get('device_type') or self.data.get('device_type'))
        elif 'module_type' in self.initial or self.data.get('module_type'):
            parent = ModuleType.objects.get(pk=self.initial.get('module_type') or self.data.get('module_type'))
        else:
            return
        occupied_port_positions = [(front_port.rear_port_id, front_port.rear_port_position) for front_port in parent.frontporttemplates.all()]
        choices = []
        rear_ports = parent.rearporttemplates.all()
        for rear_port in rear_ports:
            for i in range(1, rear_port.positions + 1):
                if (rear_port.pk, i) not in occupied_port_positions:
                    choices.append(('{}:{}'.format(rear_port.pk, i), '{}:{}'.format(rear_port.name, i)))
        self.fields['rear_port'].choices = choices

    def clean(self):
        if False:
            i = 10
            return i + 15
        frontport_count = len(self.cleaned_data['name'])
        rearport_count = len(self.cleaned_data['rear_port'])
        if frontport_count != rearport_count:
            raise forms.ValidationError({'rear_port': _('The number of front port templates to be created ({frontport_count}) must match the selected number of rear port positions ({rearport_count}).').format(frontport_count=frontport_count, rearport_count=rearport_count)})

    def get_iterative_data(self, iteration):
        if False:
            i = 10
            return i + 15
        (rear_port, position) = self.cleaned_data['rear_port'][iteration].split(':')
        return {'rear_port': int(rear_port), 'rear_port_position': int(position)}

class RearPortTemplateCreateForm(ComponentCreateForm, model_forms.RearPortTemplateForm):

    class Meta(model_forms.RearPortTemplateForm.Meta):
        exclude = ('name', 'label')

class DeviceBayTemplateCreateForm(ComponentCreateForm, model_forms.DeviceBayTemplateForm):

    class Meta(model_forms.DeviceBayTemplateForm.Meta):
        exclude = ('name', 'label')

class ModuleBayTemplateCreateForm(ComponentCreateForm, model_forms.ModuleBayTemplateForm):
    position = ExpandableNameField(label=_('Position'), required=False, help_text=_('Alphanumeric ranges are supported. (Must match the number of objects being created.)'))
    replication_fields = ('name', 'label', 'position')

    class Meta(model_forms.ModuleBayTemplateForm.Meta):
        exclude = ('name', 'label', 'position')

class InventoryItemTemplateCreateForm(ComponentCreateForm, model_forms.InventoryItemTemplateForm):

    class Meta(model_forms.InventoryItemTemplateForm.Meta):
        exclude = ('name', 'label')

class ConsolePortCreateForm(ComponentCreateForm, model_forms.ConsolePortForm):

    class Meta(model_forms.ConsolePortForm.Meta):
        exclude = ('name', 'label')

class ConsoleServerPortCreateForm(ComponentCreateForm, model_forms.ConsoleServerPortForm):

    class Meta(model_forms.ConsoleServerPortForm.Meta):
        exclude = ('name', 'label')

class PowerPortCreateForm(ComponentCreateForm, model_forms.PowerPortForm):

    class Meta(model_forms.PowerPortForm.Meta):
        exclude = ('name', 'label')

class PowerOutletCreateForm(ComponentCreateForm, model_forms.PowerOutletForm):

    class Meta(model_forms.PowerOutletForm.Meta):
        exclude = ('name', 'label')

class InterfaceCreateForm(ComponentCreateForm, model_forms.InterfaceForm):

    class Meta(model_forms.InterfaceForm.Meta):
        exclude = ('name', 'label')

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super().__init__(*args, **kwargs)
        if 'module' in self.fields:
            self.fields['name'].help_text += _('The string <code>{module}</code> will be replaced with the position of the assigned module, if any.')

class FrontPortCreateForm(ComponentCreateForm, model_forms.FrontPortForm):
    device = DynamicModelChoiceField(label=_('Device'), queryset=Device.objects.all(), selector=True, widget=APISelect(attrs={'hx-get': '.', 'hx-include': f'#form_fields', 'hx-target': f'#form_fields'}))
    rear_port = forms.MultipleChoiceField(choices=[], label=_('Rear ports'), help_text=_('Select one rear port assignment for each front port being created.'), widget=forms.SelectMultiple(attrs={'size': 6}))
    fieldsets = ((None, ('device', 'module', 'name', 'label', 'type', 'color', 'rear_port', 'mark_connected', 'description', 'tags')),)

    class Meta(model_forms.FrontPortForm.Meta):
        exclude = ('name', 'label', 'rear_port', 'rear_port_position')

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
        if (device_id := (self.data.get('device') or self.initial.get('device'))):
            device = Device.objects.get(pk=device_id)
        else:
            return
        occupied_port_positions = [(front_port.rear_port_id, front_port.rear_port_position) for front_port in device.frontports.all()]
        choices = []
        rear_ports = RearPort.objects.filter(device=device)
        for rear_port in rear_ports:
            for i in range(1, rear_port.positions + 1):
                if (rear_port.pk, i) not in occupied_port_positions:
                    choices.append(('{}:{}'.format(rear_port.pk, i), '{}:{}'.format(rear_port.name, i)))
        self.fields['rear_port'].choices = choices

    def clean(self):
        if False:
            for i in range(10):
                print('nop')
        frontport_count = len(self.cleaned_data['name'])
        rearport_count = len(self.cleaned_data['rear_port'])
        if frontport_count != rearport_count:
            raise forms.ValidationError({'rear_port': _('The number of front ports to be created ({frontport_count}) must match the selected number of rear port positions ({rearport_count}).').format(frontport_count=frontport_count, rearport_count=rearport_count)})

    def get_iterative_data(self, iteration):
        if False:
            i = 10
            return i + 15
        (rear_port, position) = self.cleaned_data['rear_port'][iteration].split(':')
        return {'rear_port': int(rear_port), 'rear_port_position': int(position)}

class RearPortCreateForm(ComponentCreateForm, model_forms.RearPortForm):

    class Meta(model_forms.RearPortForm.Meta):
        exclude = ('name', 'label')

class DeviceBayCreateForm(ComponentCreateForm, model_forms.DeviceBayForm):

    class Meta(model_forms.DeviceBayForm.Meta):
        exclude = ('name', 'label')

class ModuleBayCreateForm(ComponentCreateForm, model_forms.ModuleBayForm):
    position = ExpandableNameField(label=_('Position'), required=False, help_text=_('Alphanumeric ranges are supported. (Must match the number of objects being created.)'))
    replication_fields = ('name', 'label', 'position')

    class Meta(model_forms.ModuleBayForm.Meta):
        exclude = ('name', 'label', 'position')

class InventoryItemCreateForm(ComponentCreateForm, model_forms.InventoryItemForm):

    class Meta(model_forms.InventoryItemForm.Meta):
        exclude = ('name', 'label')

class VirtualChassisCreateForm(NetBoxModelForm):
    region = DynamicModelChoiceField(label=_('Region'), queryset=Region.objects.all(), required=False, initial_params={'sites': '$site'})
    site_group = DynamicModelChoiceField(label=_('Site group'), queryset=SiteGroup.objects.all(), required=False, initial_params={'sites': '$site'})
    site = DynamicModelChoiceField(label=_('Site'), queryset=Site.objects.all(), required=False, query_params={'region_id': '$region', 'group_id': '$site_group'})
    rack = DynamicModelChoiceField(label=_('Rack'), queryset=Rack.objects.all(), required=False, null_option='None', query_params={'site_id': '$site'})
    members = DynamicModelMultipleChoiceField(label=_('Members'), queryset=Device.objects.all(), required=False, query_params={'site_id': '$site', 'rack_id': '$rack'})
    initial_position = forms.IntegerField(label=_('Initial position'), initial=1, required=False, help_text=_('Position of the first member device. Increases by one for each additional member.'))

    class Meta:
        model = VirtualChassis
        fields = ['name', 'domain', 'description', 'region', 'site_group', 'site', 'rack', 'members', 'initial_position', 'tags']

    def clean(self):
        if False:
            i = 10
            return i + 15
        super().clean()
        if self.cleaned_data['members'] and self.cleaned_data['initial_position'] is None:
            raise forms.ValidationError({'initial_position': _('A position must be specified for the first VC member.')})

    def save(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        instance = super().save(*args, **kwargs)
        if instance.pk and self.cleaned_data['members']:
            initial_position = self.cleaned_data.get('initial_position', 1)
            for (i, member) in enumerate(self.cleaned_data['members'], start=initial_position):
                member.virtual_chassis = instance
                member.vc_position = i
                member.save()
        return instance