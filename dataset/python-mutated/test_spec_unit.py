import io
import docker
import pytest

def build_docker_image(text: str, tag: str) -> docker.models.images.Image:
    if False:
        print('Hello World!')
    '\n    Really for this test we dont need to remove the image since we access it by a string name\n    and remove it also by a string name. But maybe we wanna use it somewhere\n    '
    client = docker.from_env()
    fileobj = io.BytesIO(bytes(text, 'utf-8'))
    (image, _) = client.images.build(fileobj=fileobj, tag=tag, forcerm=True, rm=True)
    return image

@pytest.fixture
def correct_connector_image() -> str:
    if False:
        i = 10
        return i + 15
    dockerfile_text = '\n        FROM scratch\n        ENV AIRBYTE_ENTRYPOINT "python /airbyte/integration_code/main.py"\n        ENTRYPOINT ["python", "/airbyte/integration_code/main.py"]\n        '
    tag = 'my-valid-one'
    build_docker_image(dockerfile_text, tag)
    yield tag
    client = docker.from_env()
    client.images.remove(image=tag, force=True)

@pytest.fixture
def connector_image_without_env():
    if False:
        for i in range(10):
            print('nop')
    dockerfile_text = '\n        FROM scratch\n        ENTRYPOINT ["python", "/airbyte/integration_code/main.py"]\n        '
    tag = 'my-no-env'
    build_docker_image(dockerfile_text, tag)
    yield tag
    client = docker.from_env()
    client.images.remove(image=tag, force=True)

@pytest.fixture
def connector_image_with_ne_properties():
    if False:
        i = 10
        return i + 15
    dockerfile_text = '\n        FROM scratch\n        ENV AIRBYTE_ENTRYPOINT "python /airbyte/integration_code/main.py"\n        ENTRYPOINT ["python3", "/airbyte/integration_code/main.py"]\n        '
    tag = 'my-ne-properties'
    build_docker_image(dockerfile_text, tag)
    yield tag
    client = docker.from_env()
    client.images.remove(image=tag, force=True)