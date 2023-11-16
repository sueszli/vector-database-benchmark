"""
Script for generating the model zoo docs page contents
``docs/source/user_guide/model_zoo/models.rst``.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import logging
import os
import re
from jinja2 import Environment, BaseLoader
import eta.core.utils as etau
import fiftyone.zoo as foz
logger = logging.getLogger(__name__)
_HEADER = '\n.. _model-zoo-models:\n\nAvailable Zoo Models\n====================\n\n.. default-role:: code\n\nThis page lists all of the models available in the Model Zoo.\n\n.. note::\n\n    Check out the :ref:`API reference <model-zoo-api>` for complete\n    instructions for using the Model Zoo.\n'
_SECTION_TEMPLATE = "\n.. _model-zoo-{{ link_name }}-models:\n\n{{ header_name }} models\n{{ '-' * (header_name|length + 7) }}\n"
_MODEL_TEMPLATE = '\n.. _model-zoo-{{ name }}:\n\n{{ header_name }}\n{{ \'_\' * header_name|length }}\n\n{{ description }}.\n\n**Details**\n\n-   Model name: ``{{ name }}``\n-   Model source: {{ source }}\n{% if size %}\n-   Model size: {{ size }}\n{% endif %}\n-   Exposes embeddings? {{ exposes_embeddings }}\n-   Tags: ``{{ tags }}``\n\n**Requirements**\n\n{% if base_packages %}\n-   Packages: ``{{ base_packages }}``\n\n{% endif %}\n-   CPU support\n\n    -   {{ supports_cpu }}\n{% if cpu_packages %}\n    -   Packages: ``{{ cpu_packages }}``\n{% endif %}\n\n-   GPU support\n\n    -   {{ supports_gpu }}\n{% if gpu_packages %}\n    -   Packages: ``{{ gpu_packages }}``\n{% endif %}\n\n**Example usage**\n\n.. code-block:: python\n    :linenos:\n\n    import fiftyone as fo\n    import fiftyone.zoo as foz\n\n{% if \'imagenet\' in name %}\n    dataset = foz.load_zoo_dataset(\n        "imagenet-sample",\n        dataset_name=fo.get_default_dataset_name(),\n        max_samples=50,\n        shuffle=True,\n    )\n{% else %}\n    dataset = foz.load_zoo_dataset(\n        "coco-2017",\n        split="validation",\n        dataset_name=fo.get_default_dataset_name(),\n        max_samples=50,\n        shuffle=True,\n    )\n{% endif %}\n\n{% if \'segment-anything\' in tags %}\n    model = foz.load_zoo_model("{{ name }}")\n\n    # Segment inside boxes\n    dataset.apply_model(\n        model,\n        label_field="segmentations",\n        prompt_field="ground_truth",  # can contain Detections or Keypoints\n    )\n\n    # Full automatic segmentations\n    dataset.apply_model(model, label_field="auto")\n\n    session = fo.launch_app(dataset)\n{% elif \'dinov2\' in name %}\n    model = foz.load_zoo_model("{{ name }}")\n\n    embeddings = dataset.compute_embeddings(model)\n{% else %}\n    model = foz.load_zoo_model("{{ name }}")\n\n    dataset.apply_model(model, label_field="predictions")\n\n    session = fo.launch_app(dataset)\n{% endif %}\n\n{% if \'clip\' in tags %}\n    #\n    # Make zero-shot predictions with custom classes\n    #\n\n    model = foz.load_zoo_model(\n        "{{ name }}",\n        text_prompt="A photo of a",\n        classes=["person", "dog", "cat", "bird", "car", "tree", "chair"],\n    )\n\n    dataset.apply_model(model, label_field="predictions")\n    session.refresh()\n{% endif %}\n'
_CARD_SECTION_START = '\n.. raw:: html\n\n    <div id="tutorial-cards-container">\n\n    <nav class="navbar navbar-expand-lg navbar-light tutorials-nav col-12">\n        <div class="tutorial-tags-container">\n            <div id="dropdown-filter-tags">\n                <div class="tutorial-filter-menu">\n                    <div class="tutorial-filter filter-btn all-tag-selected" data-tag="all">All</div>\n                </div>\n            </div>\n        </div>\n    </nav>\n\n    <hr class="tutorials-hr">\n\n    <div class="row">\n\n    <div id="tutorial-cards">\n    <div class="list">\n'
_CARD_SECTION_END = '\n.. raw:: html\n\n    </div>\n\n    <div class="pagination d-flex justify-content-center"></div>\n\n    </div>\n\n    </div>\n'
_CARD_MODEL_TEMPLATE = '\n.. customcarditem::\n    :header: {{ header }}\n    :description: {{ description }}\n    :link: {{ link }}\n    :tags: {{ tags }}\n'

def _render_section_content(template, all_models, print_source, header_name):
    if False:
        return 10
    models = []
    for (model_name, source, _) in all_models:
        if source != print_source:
            continue
        zoo_model = foz.get_zoo_model(model_name)
        tags_str = ', '.join(zoo_model.tags)
        models.append({'name': model_name, 'tags_str': tags_str})
    col1_width = 2 * max((len(m['name']) for m in models)) + 22
    col2_width = max((len(m['tags_str']) for m in models)) + 2
    return template.render(link_name=print_source, header_name=header_name, col1_width=col1_width, col2_width=col2_width, models=models)

def _render_model_content(template, model_name):
    if False:
        for i in range(10):
            print('nop')
    zoo_model = foz.get_zoo_model(model_name)
    if 'torch' in zoo_model.tags:
        source = 'torch'
    elif any((t in zoo_model.tags for t in ('tf', 'tf1', 'tf2'))):
        source = 'tensorflow'
    else:
        source = 'other'
    header_name = model_name
    if zoo_model.size_bytes is not None:
        size_str = etau.to_human_bytes_str(zoo_model.size_bytes, decimals=2)
        size_str = size_str[:-2] + ' ' + size_str[-2:]
    else:
        size_str = None
    if 'embeddings' in zoo_model.tags:
        exposes_embeddings = 'yes'
    else:
        exposes_embeddings = 'no'
    tags_str = ', '.join(zoo_model.tags)
    base_packages = zoo_model.requirements.packages
    if base_packages is not None:
        base_packages = ', '.join(base_packages)
    if zoo_model.supports_cpu:
        supports_cpu = 'yes'
    else:
        supports_cpu = 'no'
    cpu_packages = zoo_model.requirements.cpu_packages
    if cpu_packages is not None:
        cpu_packages = ', '.join(cpu_packages)
    if zoo_model.supports_gpu:
        supports_gpu = 'yes'
    else:
        supports_gpu = 'no'
    gpu_packages = zoo_model.requirements.gpu_packages
    if gpu_packages is not None:
        gpu_packages = ', '.join(gpu_packages)
    content = template.render(name=zoo_model.name, header_name=header_name, description=zoo_model.description, source=zoo_model.source, size=size_str, exposes_embeddings=exposes_embeddings, tags=tags_str, base_packages=base_packages, supports_cpu=supports_cpu, cpu_packages=cpu_packages, supports_gpu=supports_gpu, gpu_packages=gpu_packages)
    return (source, content)

def _render_card_model_content(template, model_name):
    if False:
        for i in range(10):
            print('nop')
    zoo_model = foz.get_zoo_model(model_name)
    tags = []
    for tag in zoo_model.tags:
        if tag == 'tf1':
            tags.append('TensorFlow-1')
        elif tag == 'tf2':
            tags.append('TensorFlow-2')
        elif tag == 'tf':
            tags.append('TensorFlow')
        elif tag == 'torch':
            tags.append('PyTorch')
        else:
            tags.append(tag.capitalize().replace(' ', '-'))
    tags = ','.join(tags)
    link = 'models.html#%s' % zoo_model.name
    description = zoo_model.description
    description = description.replace('`_', '"')
    description = description.replace('`', '"')
    description = re.sub(' <.*>', '', description)
    content = template.render(header=zoo_model.name, description=description, link=link, tags=tags)
    return content

def _generate_section(template, all_models, print_source, header_name):
    if False:
        while True:
            i = 10
    content = [_render_section_content(template, all_models, print_source, header_name)]
    for (_, source, model_content) in all_models:
        if source == print_source:
            content.append(model_content)
    return content

def main():
    if False:
        return 10
    environment = Environment(loader=BaseLoader, trim_blocks=True, lstrip_blocks=True)
    section_template = environment.from_string(_SECTION_TEMPLATE)
    model_template = environment.from_string(_MODEL_TEMPLATE)
    card_model_template = environment.from_string(_CARD_MODEL_TEMPLATE)
    models = []
    for model_name in foz.list_zoo_models():
        (source, content) = _render_model_content(model_template, model_name)
        models.append((model_name, source, content))
    content = [_HEADER]
    content.append(_CARD_SECTION_START)
    for model_name in foz.list_zoo_models():
        card_content = _render_card_model_content(card_model_template, model_name)
        content.append(card_content)
    content.append(_CARD_SECTION_END)
    content.extend(_generate_section(section_template, models, 'torch', 'Torch'))
    content.extend(_generate_section(section_template, models, 'tensorflow', 'TensorFlow'))
    docs_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
    outpath = os.path.join(docs_dir, 'source/user_guide/model_zoo/models.rst')
    print("Writing '%s'" % outpath)
    etau.write_file('\n'.join(content), outpath)
if __name__ == '__main__':
    main()