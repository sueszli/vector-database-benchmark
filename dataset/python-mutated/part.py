"""Functionality for Part import template.

Primarily Part import tools.
"""
from InvenTree.helpers import DownloadFile, GetExportFormats
from .admin import PartImportResource
from .models import Part

def IsValidPartFormat(fmt):
    if False:
        i = 10
        return i + 15
    'Test if a file format specifier is in the valid list of part import template file formats.'
    return fmt.strip().lower() in GetExportFormats()

def MakePartTemplate(fmt):
    if False:
        while True:
            i = 10
    'Generate a part import template file (for user download).'
    fmt = fmt.strip().lower()
    if not IsValidPartFormat(fmt):
        fmt = 'csv'
    query = Part.objects.filter(pk=None)
    dataset = PartImportResource().export(queryset=query, importing=True)
    data = dataset.export(fmt)
    filename = 'InvenTree_Part_Template.' + fmt
    return DownloadFile(data, filename)