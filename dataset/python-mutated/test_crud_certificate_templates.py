import re
import typing
import uuid
import google.auth
from conftest import LOCATION
from create_certificate_template import create_certificate_template
from delete_certificate_template import delete_certificate_template
from list_certificate_templates import list_certificate_templates
from update_certificate_template import update_certificate_template
PROJECT = google.auth.default()[1]
COMMON_NAME = 'COMMON_NAME'
ORGANIZATION = 'ORGANIZATION'
CA_DURATION = 1000000

def generate_name() -> str:
    if False:
        print('Hello World!')
    return 'i' + uuid.uuid4().hex[:10]

def test_create_delete_certificate_template(capsys: typing.Any) -> None:
    if False:
        print('Hello World!')
    TEMPLATE_NAME = generate_name()
    create_certificate_template(PROJECT, LOCATION, TEMPLATE_NAME)
    delete_certificate_template(PROJECT, LOCATION, TEMPLATE_NAME)
    (out, _) = capsys.readouterr()
    assert re.search(f'Operation result: name: "projects/{PROJECT}/locations/{LOCATION}/certificateTemplates/{TEMPLATE_NAME}"', out)
    assert re.search(f'Deleted certificate template: {TEMPLATE_NAME}', out)

def test_list_certificate_templates(certificate_template, capsys: typing.Any) -> None:
    if False:
        i = 10
        return i + 15
    TEMPLATE_NAME = certificate_template
    list_certificate_templates(PROJECT, LOCATION)
    (out, _) = capsys.readouterr()
    assert 'Available certificate templates:' in out
    assert f'{TEMPLATE_NAME}\n' in out

def test_update_certificate_template(certificate_template, capsys: typing.Any) -> None:
    if False:
        while True:
            i = 10
    TEMPLATE_NAME = certificate_template
    update_certificate_template(PROJECT, LOCATION, TEMPLATE_NAME)
    (out, _) = capsys.readouterr()
    assert 'Successfully updated the certificate template!' in out