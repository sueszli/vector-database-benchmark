"""
Utilities for working with annotations in
`Labelbox format <https://labelbox.com/docs/exporting-data/export-format-detail>`_.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
from copy import copy, deepcopy
import logging
import os
import requests
from uuid import uuid4
import warnings
import webbrowser
import numpy as np
import eta.core.image as etai
import eta.core.serial as etas
import eta.core.utils as etau
import eta.core.web as etaw
import fiftyone.core.fields as fof
import fiftyone.core.labels as fol
import fiftyone.core.media as fomm
import fiftyone.core.metadata as fom
import fiftyone.core.sample as fos
import fiftyone.core.utils as fou
import fiftyone.core.validation as fov
import fiftyone.utils.annotations as foua
lb = fou.lazy_import('labelbox', callback=lambda : fou.ensure_import('labelbox'))
lbs = fou.lazy_import('labelbox.schema')
lbo = fou.lazy_import('labelbox.schema.ontology')
lbr = fou.lazy_import('labelbox.schema.review')
logger = logging.getLogger(__name__)

class LabelboxBackendConfig(foua.AnnotationBackendConfig):
    """Class for configuring :class:`LabelboxBackend` instances.

    Args:
        name: the name of the backend
        label_schema: a dictionary containing the description of label fields,
            classes and attribute to annotate
        media_field ("filepath"): string field name containing the paths to
            media files on disk to upload
        url (None): the url of the Labelbox server
        api_key (None): the Labelbox API key
        project_name (None): a name for the Labelbox project that will be
            created. The default is ``"FiftyOne_<dataset_name>"``
        members (None): an optional list of ``(email, role)`` tuples specifying
            the email addresses and roles of users to add to the project. If a
            user is not a member of the project's organization, an email
            invitation will be sent to them. The supported roles are
            ``["LABELER", "REVIEWER", "TEAM_MANAGER", "ADMIN"]``
        classes_as_attrs (True): whether to show every object class at the top
            level of the editor (False) or whether to show the label field at
            the top level and annotate the class as a required attribute of
            each object (True)
    """

    def __init__(self, name, label_schema, media_field='filepath', url=None, api_key=None, project_name=None, members=None, classes_as_attrs=True, **kwargs):
        if False:
            return 10
        super().__init__(name, label_schema, media_field=media_field, **kwargs)
        self.url = url
        self.project_name = project_name
        self.members = members
        self.classes_as_attrs = classes_as_attrs
        self._api_key = api_key

    @property
    def api_key(self):
        if False:
            return 10
        return self._api_key

    @api_key.setter
    def api_key(self, value):
        if False:
            print('Hello World!')
        self._api_key = value

    @property
    def _experimental(self):
        if False:
            print('Hello World!')
        if self.members:
            return True
        return False

    def load_credentials(self, url=None, api_key=None):
        if False:
            i = 10
            return i + 15
        self._load_parameters(url=url, api_key=api_key)

class LabelboxBackend(foua.AnnotationBackend):
    """Class for interacting with the Labelbox annotation backend."""

    @property
    def supported_media_types(self):
        if False:
            i = 10
            return i + 15
        return [fomm.IMAGE, fomm.VIDEO]

    @property
    def supported_label_types(self):
        if False:
            while True:
                i = 10
        return ['classification', 'classifications', 'detection', 'detections', 'instance', 'instances', 'polyline', 'polylines', 'polygon', 'polygons', 'keypoint', 'keypoints', 'segmentation', 'scalar']

    @property
    def supported_scalar_types(self):
        if False:
            print('Hello World!')
        return [fof.IntField, fof.FloatField, fof.StringField, fof.BooleanField]

    @property
    def supported_attr_types(self):
        if False:
            for i in range(10):
                print('nop')
        return ['text', 'select', 'radio', 'checkbox']

    @property
    def supports_keyframes(self):
        if False:
            while True:
                i = 10
        return False

    @property
    def supports_video_sample_fields(self):
        if False:
            while True:
                i = 10
        return False

    @property
    def requires_label_schema(self):
        if False:
            while True:
                i = 10
        return True

    def recommend_attr_tool(self, name, value):
        if False:
            print('Hello World!')
        if isinstance(value, bool):
            return {'type': 'radio', 'values': [True, False]}
        return {'type': 'text'}

    def requires_attr_values(self, attr_type):
        if False:
            print('Hello World!')
        return attr_type != 'text'

    def _connect_to_api(self):
        if False:
            print('Hello World!')
        return LabelboxAnnotationAPI(self.config.name, self.config.url, api_key=self.config.api_key, _experimental=self.config._experimental)

    def upload_annotations(self, samples, anno_key, launch_editor=False):
        if False:
            for i in range(10):
                print('nop')
        api = self.connect_to_api()
        logger.info('Uploading media to Labelbox...')
        results = api.upload_samples(samples, anno_key, self)
        logger.info('Upload complete')
        if launch_editor:
            results.launch_editor()
        return results

    def download_annotations(self, results):
        if False:
            i = 10
            return i + 15
        api = self.connect_to_api()
        logger.info('Downloading labels from Labelbox...')
        annotations = api.download_annotations(results)
        logger.info('Download complete')
        return annotations

class LabelboxAnnotationAPI(foua.AnnotationAPI):
    """A class to facilitate connection to and management of projects in
    Labelbox.

    On initializiation, this class constructs a client based on the provided
    server url and credentials.

    This API provides methods to easily upload, download, create, and delete
    projects and data through the formatted urls specified by the Labelbox API.

    Additionally, samples and label schemas can be uploaded and annotations
    downloaded through this class.

    Args:
        name: the name of the backend
        url: url of the Labelbox server
        api_key (None): the Labelbox API key
    """

    def __init__(self, name, url, api_key=None, _experimental=False):
        if False:
            i = 10
            return i + 15
        if '://' not in url:
            protocol = 'http'
            base_url = url
        else:
            (protocol, base_url) = url.split('://')
        self._name = name
        self._url = base_url
        self._protocol = protocol
        self._api_key = api_key
        self._experimental = _experimental
        self._roles = None
        self._tool_types_map = None
        self._setup()

    def _setup(self):
        if False:
            return 10
        if not self._url:
            raise ValueError('You must provide/configure the `url` of the Labelbox server')
        api_key = self._api_key
        if api_key is None:
            api_key = self._prompt_api_key(self._name)
        self._client = lb.client.Client(api_key=api_key, endpoint=self.base_graphql_url, enable_experimental=self._experimental)
        self._tool_types_map = {'detections': lbo.Tool.Type.BBOX, 'detection': lbo.Tool.Type.BBOX, 'instance': lbo.Tool.Type.SEGMENTATION, 'instances': lbo.Tool.Type.SEGMENTATION, 'segmentation': lbo.Tool.Type.SEGMENTATION, 'polyline': lbo.Tool.Type.LINE, 'polylines': lbo.Tool.Type.LINE, 'polygon': lbo.Tool.Type.POLYGON, 'polygons': lbo.Tool.Type.POLYGON, 'keypoint': lbo.Tool.Type.POINT, 'keypoints': lbo.Tool.Type.POINT, 'classification': lbo.Classification, 'classifications': lbo.Classification, 'scalar': lbo.Classification}

    @property
    def roles(self):
        if False:
            i = 10
            return i + 15
        if self._roles is None:
            self._roles = self._client.get_roles()
        return self._roles

    @property
    def attr_type_map(self):
        if False:
            print('Hello World!')
        return {'text': lbo.Classification.Type.TEXT, 'select': lbo.Classification.Type.DROPDOWN, 'radio': lbo.Classification.Type.RADIO, 'checkbox': lbo.Classification.Type.CHECKLIST}

    @property
    def attr_list_types(self):
        if False:
            for i in range(10):
                print('nop')
        return ['checkbox']

    @property
    def base_api_url(self):
        if False:
            while True:
                i = 10
        return '%s://api.%s' % (self._protocol, self._url)

    @property
    def base_graphql_url(self):
        if False:
            print('Hello World!')
        return '%s/graphql' % self.base_api_url

    @property
    def projects_url(self):
        if False:
            print('Hello World!')
        return '%s/projects' % self.base_api_url

    def project_url(self, project_id):
        if False:
            return 10
        return '%s/%s' % (self.projects_url, project_id)

    def editor_url(self, project_id):
        if False:
            while True:
                i = 10
        return '%s://editor.%s/?project=%s' % (self._protocol, self._url, project_id)

    def get_project_users(self, project=None, project_id=None):
        if False:
            for i in range(10):
                print('nop')
        'Returns a list of users that are assigned to the given project.\n\n        Provide either ``project`` or ``project_id`` to this method.\n\n        Args:\n            project: a ``labelbox.schema.project.Project``\n            project_id: the project ID\n\n        Returns:\n            a list of ``labelbox.schema.user.User`` objects\n        '
        if project is None:
            if project_id is None:
                raise ValueError('Either `project` or `project_id` must be provided')
            project = self.get_project(project_id)
        project_users = []
        project_id = project.uid
        users = list(project.organization().users())
        for user in users:
            if project in user.projects():
                project_users.append(user)
        return users

    def add_member(self, project, email, role):
        if False:
            while True:
                i = 10
        'Adds a member to the given Labelbox project with the given\n        project-level role.\n\n        If the user is not a member of the project\'s parent organization, an\n        email invitivation will be sent.\n\n        Args:\n            project: the ``labelbox.schema.project.Project``\n            email: the email of the user\n            role: the role for the user. Supported values are\n                ``["LABELER", "REVIEWER", "TEAM_MANAGER", "ADMIN"]``\n        '
        if not self._experimental:
            raise ValueError('This method can only be used if the `LabelboxAnnotationAPI` object was initialized with `_experimental=True`')
        if role not in self.roles or role == 'NONE':
            raise ValueError("Unsupported user role '%s'" % role)
        role_id = self.roles[role]
        organization = self._client.get_organization()
        existing_users = {u.email: u for u in organization.users()}
        if email in existing_users:
            user = existing_users[email]
            user.upsert_project_role(project, role_id)
            return
        limit = organization.invite_limit()
        if limit.remaining == 0:
            logger.warning("Your organization has reached its limit of %d members. Cannot invite new member %s to project '%s'", limit.limit, email, project.name)
            return
        project_role = lbs.organization.ProjectRole(project=project, role=role_id)
        organization.invite_user(email, self.roles['NONE'], project_roles=[project_role])

    def list_datasets(self):
        if False:
            i = 10
            return i + 15
        'Retrieves the list of datasets in your Labelbox account.\n\n        Returns:\n            a list of dataset IDs\n        '
        datasets = self._client.get_datasets()
        return [d.uid for d in datasets]

    def delete_datasets(self, dataset_ids):
        if False:
            while True:
                i = 10
        'Deletes the given datasets from the Labelbox server.\n\n        Args:\n            dataset_ids: an iterable of dataset IDs\n        '
        logger.info('Deleting datasets...')
        with fou.ProgressBar() as pb:
            for dataset_id in pb(list(dataset_ids)):
                dataset = self._client.get_dataset(dataset_id)
                dataset.delete()

    def list_projects(self):
        if False:
            for i in range(10):
                print('nop')
        'Retrieves the list of projects in your Labelbox account.\n\n        Returns:\n            a list of project IDs\n        '
        projects = self._client.get_projects()
        return [p.uid for p in projects]

    def get_project(self, project_id):
        if False:
            i = 10
            return i + 15
        'Retrieves the ``labelbox.schema.project.Project`` for the project\n        with the given ID.\n\n        Args:\n            project_id: the project ID\n\n        Returns:\n            a ``labelbox.schema.project.Project``\n        '
        return self._client.get_project(project_id)

    def delete_project(self, project_id, delete_batches=False, delete_ontologies=True):
        if False:
            while True:
                i = 10
        'Deletes the given project from the Labelbox server.\n\n        Args:\n            project_id: the project ID\n            delete_batches (False): whether to delete the attached batches as\n                well\n            delete_ontologies (True): whether to delete the attached\n                ontologies as well\n        '
        project = self._client.get_project(project_id)
        logger.info("Deleting project '%s'...", project_id)
        if delete_batches:
            for batch in project.batches():
                batch.delete_labels()
                lb.DataRow.bulk_delete(data_rows=list(batch.export_data_rows(include_metadata=False)))
                batch.delete()
        ontology = project.ontology()
        project.delete()
        if delete_ontologies:
            self._client.delete_unused_ontology(ontology.uid)

    def delete_projects(self, project_ids, delete_batches=False):
        if False:
            return 10
        'Deletes the given projects from the Labelbox server.\n\n        Args:\n            project_ids: an iterable of project IDs\n            delete_batches (False): whether to delete the attached batches as\n                well\n        '
        for project_id in project_ids:
            self.delete_project(project_id, delete_batches=delete_batches)

    def delete_unused_ontologies(self):
        if False:
            i = 10
            return i + 15
        'Deletes unused ontologies from the Labelbox server.'
        deleted_ontologies = []
        unused_ontologies = self._client.get_unused_ontologies()
        while unused_ontologies:
            for o in unused_ontologies:
                deleted_ontologies.append(o)
                self._client.delete_unused_ontology(o)
            unused_ontologies = self._client.get_unused_ontologies()
        logger.info('Deleted %d ontologies.' % len(deleted_ontologies))

    def launch_editor(self, url=None):
        if False:
            i = 10
            return i + 15
        'Launches the Labelbox editor in your default web browser.\n\n        Args:\n            url (None): an optional URL to open. By default, the base URL of\n                the server is opened\n        '
        if url is None:
            url = self.projects_url
        webbrowser.open(url, new=2)

    def _skip_existing_global_keys(self, media_paths, sample_ids):
        if False:
            while True:
                i = 10
        lb_logger = logging.getLogger('labelbox.client')
        lb_logger_level = lb_logger.level
        lb_logger.setLevel(logging.ERROR)
        response = self._client.get_data_row_ids_for_global_keys(sample_ids)
        lb_logger.setLevel(lb_logger_level)
        results = response['results']
        new_media_paths = []
        new_sample_ids = []
        for (i, result) in enumerate(results):
            if result == '':
                new_media_paths.append(media_paths[i])
                new_sample_ids.append(sample_ids[i])
        num_existing_samples = len(media_paths) - len(new_media_paths)
        if num_existing_samples:
            logger.info('Found %d data row(s) with a global key matching a sample id. These samples will not be reuploaded...' % num_existing_samples)
        return (new_media_paths, new_sample_ids)

    def upload_data(self, samples, dataset_name, media_field='filepath'):
        if False:
            while True:
                i = 10
        'Uploads the media for the given samples to Labelbox.\n\n        This method uses ``labelbox.schema.dataset.Dataset.create_data_rows()``\n        to add data in batches, and sets the global key of each DataRow to the\n        ID of the corresponding sample.\n\n        Args:\n            samples: a :class:`fiftyone.core.collections.SampleCollection`\n                containing the media to upload\n            dataset_name: the name of the Labelbox dataset created if data\n                needs to be uploaded\n            media_field ("filepath"): string field name containing the paths to\n                media files on disk to upload\n        '
        (media_paths, sample_ids) = samples.values([media_field, 'id'])
        (media_paths, sample_ids) = self._skip_existing_global_keys(media_paths, sample_ids)
        upload_info = []
        with fou.ProgressBar(iters_str='samples') as pb:
            for (media_path, sample_id) in pb(zip(media_paths, sample_ids)):
                item_url = self._client.upload_file(media_path)
                upload_info.append({lb.DataRow.row_data: item_url, lb.DataRow.global_key: sample_id})
        if upload_info:
            lb_dataset = self._client.create_dataset(name=dataset_name)
            task = lb_dataset.create_data_rows(upload_info)
            task.wait_till_done()
            if task.errors:
                logger.warning('Datarow creation failed with error: %s' % task.errors)

    def upload_samples(self, samples, anno_key, backend):
        if False:
            i = 10
            return i + 15
        "Uploads the given samples to Labelbox according to the given\n        backend's annotation and server configuration.\n\n        Args:\n            samples: a :class:`fiftyone.core.collections.SampleCollection`\n            anno_key: the annotation key\n            backend: a :class:`LabelboxBackend` to use to perform the upload\n\n        Returns:\n            a :class:`LabelboxAnnotationResults`\n        "
        config = backend.config
        label_schema = config.label_schema
        media_field = config.media_field
        project_name = config.project_name
        members = config.members
        classes_as_attrs = config.classes_as_attrs
        is_video = samples.media_type == fomm.VIDEO
        for (label_field, label_info) in label_schema.items():
            if label_info['existing_field']:
                raise ValueError("Cannot use existing field '%s'; the Labelbox backend does not yet support uploading existing labels" % label_field)
        if project_name is None:
            _dataset_name = samples._root_dataset.name.replace(' ', '_')
            project_name = 'FiftyOne_%s' % _dataset_name
        self.upload_data(samples, project_name, media_field=media_field)
        global_keys = samples.values('id')
        project = self._setup_project(project_name, global_keys, label_schema, classes_as_attrs, is_video)
        if members:
            for (email, role) in members:
                self.add_member(project, email, role)
        project_id = project.uid
        id_map = {}
        frame_id_map = self._build_frame_id_map(samples)
        return LabelboxAnnotationResults(samples, config, anno_key, id_map, project_id, frame_id_map, backend=backend)

    def download_annotations(self, results):
        if False:
            for i in range(10):
                print('nop')
        'Downloads the annotations from the Labelbox server for the given\n        results instance and parses them into the appropriate FiftyOne types.\n\n        Args:\n            results: a :class:`LabelboxAnnotationResults`\n\n        Returns:\n            the annotations dict\n        '
        project_id = results.project_id
        frame_id_map = results.frame_id_map
        classes_as_attrs = results.config.classes_as_attrs
        label_schema = results.config.label_schema
        project = self._client.get_project(project_id)
        labels_json = self._download_project_labels(project=project)
        is_video = results._samples.media_type == fomm.VIDEO
        annotations = {}
        if classes_as_attrs:
            class_attr = 'class_name'
        else:
            class_attr = False
        for d in labels_json:
            labelbox_id = d['DataRow ID']
            sample_id = d['Global Key']
            if sample_id is None:
                logger.warning("Skipping DataRow '%s' with no sample ID", labelbox_id)
                continue
            metadata = self._get_sample_metadata(project, sample_id)
            if metadata is None:
                logger.warning("Skipping sample '%s' with no metadata, likely due to not finding a DataRow with a matching Global Key", sample_id)
                continue
            frame_size = (metadata['width'], metadata['height'])
            if is_video:
                video_d_list = self._get_video_labels(d['Label'])
                frames = {}
                for label_d in video_d_list:
                    frame_number = label_d['frameNumber']
                    frame_id = frame_id_map[sample_id][frame_number]
                    labels_dict = _parse_image_labels(label_d, frame_size, class_attr=class_attr)
                    if not classes_as_attrs:
                        labels_dict = self._process_label_fields(label_schema, labels_dict)
                    frames[frame_id] = labels_dict
                self._add_video_labels_to_results(annotations, frames, sample_id, label_schema)
            else:
                labels_dict = _parse_image_labels(d['Label'], frame_size, class_attr=class_attr)
                if not classes_as_attrs:
                    labels_dict = self._process_label_fields(label_schema, labels_dict)
                annotations = self._add_labels_to_results(annotations, labels_dict, sample_id, label_schema)
        return annotations

    def _process_label_fields(self, label_schema, labels_dict):
        if False:
            for i in range(10):
                print('nop')
        unexpected_types = ['segmentation', 'detections', 'keypoints', 'polylines']
        field_map = {}
        for (label_field, label_info) in label_schema.items():
            label_type = label_info['type']
            mapped_type = _UNIQUE_TYPE_MAP.get(label_type, label_type)
            field_map[mapped_type] = label_field
        _labels_dict = {}
        for (field_or_type, label_info) in labels_dict.items():
            if field_or_type in unexpected_types:
                label_field = field_map[field_or_type]
                _labels_dict[label_field] = {}
                if field_or_type in label_info:
                    label_info = label_info[field_or_type]
                _labels_dict[label_field][field_or_type] = label_info
            else:
                _labels_dict[field_or_type] = label_info
        return _labels_dict

    def _build_frame_id_map(self, samples):
        if False:
            while True:
                i = 10
        if samples.media_type != fomm.VIDEO:
            return {}
        samples.ensure_frames()
        (sample_ids, frame_numbers, frame_ids) = samples.values(['id', 'frames.frame_number', 'frames.id'])
        frame_id_map = {}
        for (sample_id, fns, fids) in zip(sample_ids, frame_numbers, frame_ids):
            frame_id_map[sample_id] = {fn: fid for (fn, fid) in zip(fns, fids)}
        return frame_id_map

    def _setup_project(self, project_name, global_keys, label_schema, classes_as_attrs, is_video):
        if False:
            for i in range(10):
                print('nop')
        media_type = lb.MediaType.Video if is_video else lb.MediaType.Image
        project = self._client.create_project(name=project_name, media_type=media_type)
        project.create_batch(name=str(uuid4()), global_keys=global_keys)
        self._setup_editor(project, label_schema, classes_as_attrs)
        if project.setup_complete is None:
            raise ValueError("Failed to create Labelbox project '%s'" % project_name)
        return project

    def _setup_editor(self, project, label_schema, classes_as_attrs):
        if False:
            for i in range(10):
                print('nop')
        editor = next(self._client.get_labeling_frontends(where=lb.LabelingFrontend.name == 'Editor'))
        tools = []
        classifications = []
        label_types = {}
        _multiple_types = ['scalar', 'classification', 'classifications']
        for (label_field, label_info) in label_schema.items():
            label_type = label_info['type']
            if label_type not in _multiple_types:
                unique_label_type = _UNIQUE_TYPE_MAP.get(label_type, label_type)
                if unique_label_type in label_types and (not classes_as_attrs):
                    raise ValueError("Only one field of each label type is allowed when `classes_as_attrs=False`; but found fields '%s' and '%s' of type '%s'" % (label_field, label_types[label_type], label_type))
                label_types[unique_label_type] = label_field
            (field_tools, field_classifications) = self._create_ontology_tools(label_info, label_field, classes_as_attrs)
            tools.extend(field_tools)
            classifications.extend(field_classifications)
        ontology_builder = lbo.OntologyBuilder(tools=tools, classifications=classifications)
        project.setup(editor, ontology_builder.asdict())

    def _create_ontology_tools(self, label_info, label_field, classes_as_attrs):
        if False:
            for i in range(10):
                print('nop')
        label_type = label_info['type']
        classes = label_info['classes']
        attr_schema = label_info['attributes']
        general_attrs = self._build_attributes(attr_schema)
        if label_type in ['scalar', 'classification', 'classifications']:
            tools = []
            classifications = self._build_classifications(classes, label_field, general_attrs, label_type, label_field)
        else:
            tools = self._build_tools(classes, label_field, label_type, general_attrs, classes_as_attrs)
            classifications = []
        return (tools, classifications)

    def _build_attributes(self, attr_schema):
        if False:
            while True:
                i = 10
        attributes = []
        for (attr_name, attr_info) in attr_schema.items():
            attr_type = attr_info['type']
            class_type = self.attr_type_map[attr_type]
            if attr_type == 'text':
                attr = lbo.Classification(class_type=class_type, name=attr_name)
            else:
                attr_values = attr_info['values']
                options = [lbo.Option(value=str(v)) for v in attr_values]
                attr = lbo.Classification(class_type=class_type, name=attr_name, options=options)
            attributes.append(attr)
        return attributes

    def _build_classifications(self, classes, name, general_attrs, label_type, label_field):
        if False:
            for i in range(10):
                print('nop')
        'Returns the classifications for the given label field. Generally,\n        the classification is a dropdown selection for given classes, but can\n        be a text entry for scalars without provided classes.\n\n        Attributes are available for Classification and Classifications types\n        in nested dropdowns.\n        '
        classifications = []
        options = []
        for c in classes:
            if isinstance(c, dict):
                sub_classes = c['classes']
                attrs = self._build_attributes(c['attributes']) + general_attrs
            else:
                sub_classes = [c]
                attrs = general_attrs
            if label_type == 'scalar':
                attrs = []
            for sc in sub_classes:
                if label_type == 'scalar':
                    sub_attrs = attrs
                else:
                    prefix = 'field:%s_class:%s_attr:' % (label_field, str(sc))
                    sub_attrs = deepcopy(attrs)
                    for attr in sub_attrs:
                        attr.name = prefix + attr.name
                options.append(lbo.Option(value=str(sc), options=sub_attrs))
        if label_type == 'scalar' and (not classes):
            classification = lbo.Classification(class_type=lbo.Classification.Type.TEXT, name=name)
            classifications.append(classification)
        elif label_type == 'classifications':
            classification = lbo.Classification(class_type=lbo.Classification.Type.CHECKLIST, name=name, options=options)
            classifications.append(classification)
        else:
            classification = lbo.Classification(class_type=lbo.Classification.Type.RADIO, name=name, options=options)
            classifications.append(classification)
        return classifications

    def _build_tools(self, classes, label_field, label_type, general_attrs, classes_as_attrs):
        if False:
            print('Hello World!')
        tools = []
        if classes_as_attrs:
            tool_type = self._tool_types_map[label_type]
            attributes = self._create_classes_as_attrs(classes, general_attrs)
            tools.append(lbo.Tool(name=label_field, tool=tool_type, classifications=attributes))
        else:
            for c in classes:
                if isinstance(c, dict):
                    subset_classes = c['classes']
                    subset_attr_schema = c['attributes']
                    subset_attrs = self._build_attributes(subset_attr_schema)
                    all_attrs = general_attrs + subset_attrs
                    for sc in subset_classes:
                        tool = self._build_tool_for_class(sc, label_type, all_attrs)
                        tools.append(tool)
                else:
                    tool = self._build_tool_for_class(c, label_type, general_attrs)
                    tools.append(tool)
        return tools

    def _build_tool_for_class(self, class_name, label_type, attributes):
        if False:
            i = 10
            return i + 15
        tool_type = self._tool_types_map[label_type]
        return lbo.Tool(name=str(class_name), tool=tool_type, classifications=attributes)

    def _create_classes_as_attrs(self, classes, general_attrs):
        if False:
            i = 10
            return i + 15
        'Creates radio attributes for all classes and formats all\n        class-specific attributes.\n        '
        options = []
        for c in classes:
            if isinstance(c, dict):
                subset_attrs = self._build_attributes(c['attributes'])
                for sc in c['classes']:
                    options.append(lbo.Option(value=str(sc), options=subset_attrs))
            else:
                options.append(lbo.Option(value=str(c)))
        classes_attr = lbo.Classification(class_type=lbo.Classification.Type.RADIO, name='class_name', options=options, required=True)
        return [classes_attr] + general_attrs

    def _get_sample_metadata(self, project, sample_id):
        if False:
            while True:
                i = 10
        metadata = None
        try:
            data_row_id_response = self._client.get_data_row_ids_for_global_keys([sample_id])
            data_row_id = data_row_id_response['results'][0]
            data_row = self._client.get_data_row(data_row_id)
            metadata = data_row.media_attributes
        except:
            pass
        return metadata

    def _get_video_labels(self, label_dict):
        if False:
            i = 10
            return i + 15
        url = label_dict['frames']
        headers = {'Authorization': 'Bearer %s' % self._api_key}
        response = requests.get(url, headers=headers)
        return etas.load_ndjson(response.text)

    def _download_project_labels(self, project_id=None, project=None):
        if False:
            print('Hello World!')
        if project is None:
            if project_id is None:
                raise ValueError('Either `project_id` or `project` must be provided')
            project = self._client.get_project(project_id)
        return download_labels_from_labelbox(project)

    def _add_labels_to_results(self, results, labels_dict, sample_id, label_schema):
        if False:
            return 10
        'Adds the labels in ``labels_dict`` to ``results``.\n\n        results::\n\n            <label_field>: {\n                <label_type>: {\n                    <sample_id>: {\n                        <label_id>:\n                            <fo.Label> or <label - for scalars>\n                    }\n                }\n            }\n\n        labels_dict::\n\n            {\n                <label_field>: {\n                    <label_type>: [<fo.Label>, ...]\n                }\n            }\n        '
        attributes = self._gather_classification_attributes(labels_dict, label_schema)
        results = self._parse_expected_label_fields(results, labels_dict, sample_id, label_schema, attributes)
        return results

    def _add_video_labels_to_results(self, results, frames_dict, sample_id, label_schema):
        if False:
            print('Hello World!')
        'Adds the video labels in ``frames_dict`` to ``results``.\n\n        results::\n\n            <label_field>: {\n                <label_type>: {\n                    <sample_id>: {\n                        <frame_id>: {\n                            <label_id>: <fo.Label>\n                        }\n                        or <label - for scalars>\n                    }\n                }\n            }\n\n        frames_dict::\n\n            {\n                <frame_id>: {\n                    <label_field>: {\n                        <label_type>: [<fo.Label>, ...]\n                    }\n                }\n            }\n        '
        for (frame_id, labels_dict) in frames_dict.items():
            attributes = self._gather_classification_attributes(labels_dict, label_schema)
            results = self._parse_expected_label_fields(results, labels_dict, sample_id, label_schema, attributes, frame_id=frame_id)
        return results

    def _gather_classification_attributes(self, labels_dict, label_schema):
        if False:
            i = 10
            return i + 15
        attributes = {}
        for (label_field, labels) in labels_dict.items():
            if label_field not in label_schema:
                if 'field:' not in label_field or '_class:' not in label_field or '_attr:' not in label_field:
                    logger.warning("Ignoring invalid classification label field '%s'", label_field)
                    continue
                (label_field, substr) = label_field.replace('field:', '').split('_class:')
                (class_name, attr_name) = substr.split('_attr:')
                if isinstance(labels, fol.Classification):
                    val = _parse_attribute(labels.label)
                elif isinstance(labels, fol.Classifications):
                    attr_type = _get_attr_type(label_schema, label_field, attr_name, class_name=class_name)
                    val = [_parse_attribute(c.label) for c in labels.classifications]
                    if attr_type not in self.attr_list_types:
                        if val:
                            val = val[0]
                        else:
                            val = None
                else:
                    logger.warning("Ignoring invalid label of type %s in label field '%s'. Expected a %s or %s" % (type(labels), label_field, fol.Classification, fol.Classifications))
                    continue
                if label_field not in attributes:
                    attributes[label_field] = {}
                if class_name not in attributes[label_field]:
                    attributes[label_field][class_name] = {}
                attributes[label_field][class_name][attr_name] = val
        return attributes

    def _parse_expected_label_fields(self, results, labels_dict, sample_id, label_schema, attributes, frame_id=None):
        if False:
            for i in range(10):
                print('nop')
        for (label_field, labels) in labels_dict.items():
            if label_field in label_schema:
                label_info = label_schema[label_field]
                mask_targets = label_info.get('mask_targets', None)
                expected_type = label_info['type']
                if isinstance(labels, dict):
                    label_results = self._convert_label_types(labels, expected_type, sample_id, frame_id=frame_id, mask_targets=mask_targets)
                else:
                    label_info = label_schema[label_field]
                    expected_type = label_info['type']
                    if expected_type == 'classifications':
                        if label_field in attributes:
                            for c in labels.classifications:
                                class_name = str(c.label)
                                if class_name in attributes[label_field]:
                                    for (attr_name, attr_val) in attributes[label_field][class_name].items():
                                        c[attr_name] = attr_val
                        result_type = 'classifications'
                        sample_results = {c.id: c for c in labels.classifications}
                    elif expected_type == 'classification':
                        if label_field in attributes:
                            class_name = str(labels.label)
                            if class_name in attributes[label_field]:
                                for (attr_name, attr_val) in attributes[label_field][class_name].items():
                                    labels[attr_name] = attr_val
                        result_type = 'classifications'
                        sample_results = {labels.id: labels}
                    else:
                        result_type = 'scalar'
                        sample_results = _parse_attribute(labels.label)
                    if frame_id is not None:
                        sample_results = {frame_id: sample_results}
                    label_results = {result_type: {sample_id: sample_results}}
                label_results = {label_field: label_results}
                results = self._merge_results(results, label_results)
        return results

    def _convert_label_types(self, labels_dict, expected_type, sample_id, frame_id=None, mask_targets=None):
        if False:
            for i in range(10):
                print('nop')
        output_labels = {}
        for (lb_type, labels_list) in labels_dict.items():
            if lb_type == 'detections':
                fo_type = 'detections'
            if lb_type == 'keypoints':
                fo_type = 'keypoints'
            if lb_type == 'polylines':
                if expected_type in ['detections', 'instances']:
                    fo_type = 'detections'
                elif expected_type == 'segmentation':
                    fo_type = 'segmentation'
                else:
                    fo_type = 'polylines'
            if lb_type == 'segmentation':
                if expected_type == 'segmentation':
                    fo_type = 'segmentation'
                else:
                    fo_type = 'detections'
                labels_list = self._convert_segmentations(labels_list, fo_type, mask_targets=mask_targets)
            if fo_type not in output_labels:
                output_labels[fo_type] = {}
            if sample_id not in output_labels[fo_type]:
                output_labels[fo_type][sample_id] = {}
            if labels_list:
                if frame_id is not None:
                    if frame_id not in output_labels[fo_type][sample_id]:
                        output_labels[fo_type][sample_id][frame_id] = {}
            for label in labels_list:
                if frame_id is not None:
                    output_labels[fo_type][sample_id][frame_id][label.id] = label
                else:
                    output_labels[fo_type][sample_id][label.id] = label
        return output_labels

    def _convert_segmentations(self, labels_list, label_type, mask_targets=None):
        if False:
            return 10
        labels = []
        for seg_dict in labels_list:
            mask = seg_dict['mask']
            label = str(seg_dict['label'])
            attrs = seg_dict['attributes']
            labels.append(fol.Detection.from_mask(mask, label, **attrs))
        if label_type != 'segmentation':
            return labels
        frame_size = (mask.shape[1], mask.shape[0])
        detections = fol.Detections(detections=labels)
        segmentation = detections.to_segmentation(frame_size=frame_size, mask_targets=mask_targets)
        return [segmentation]

    def _merge_results(self, results, new_results):
        if False:
            i = 10
            return i + 15
        if isinstance(new_results, dict):
            for (key, val) in new_results.items():
                if key not in results:
                    results[key] = val
                else:
                    results[key] = self._merge_results(results[key], val)
        return results

class LabelboxAnnotationResults(foua.AnnotationResults):
    """Class that stores all relevant information needed to monitor the
    progress of an annotation run sent to Labelbox and download the results.
    """

    def __init__(self, samples, config, anno_key, id_map, project_id, frame_id_map, backend=None):
        if False:
            return 10
        super().__init__(samples, config, anno_key, id_map, backend=backend)
        self.project_id = project_id
        self.frame_id_map = frame_id_map

    def launch_editor(self):
        if False:
            print('Hello World!')
        'Launches the Labelbox editor and loads the project for this\n        annotation run.\n        '
        api = self.connect_to_api()
        project_id = self.project_id
        editor_url = api.editor_url(project_id)
        logger.info("Launching editor at '%s'...", editor_url)
        api.launch_editor(url=editor_url)

    def get_status(self):
        if False:
            i = 10
            return i + 15
        'Gets the status of the annotation run.\n\n        Returns:\n            a dict of status information\n        '
        return self._get_status()

    def print_status(self):
        if False:
            for i in range(10):
                print('nop')
        'Prints the status of the annotation run.'
        self._get_status(log=True)

    def cleanup(self):
        if False:
            i = 10
            return i + 15
        'Deletes the project associated with this annotation run from the\n        Labelbox server.\n        '
        if self.project_id is not None:
            api = self.connect_to_api()
            api.delete_project(self.project_id)
        self.project_id = None

    def _get_status(self, log=False):
        if False:
            i = 10
            return i + 15
        api = self.connect_to_api()
        project = api.get_project(self.project_id)
        created_at = project.created_at
        updated_at = project.updated_at
        num_labeled_samples = len(list(project.labels()))
        members = list(project.members())
        positive = project.review_metrics(lbr.Review.NetScore.Positive)
        negative = project.review_metrics(lbr.Review.NetScore.Negative)
        zero = project.review_metrics(lbr.Review.NetScore.Zero)
        status = {}
        status['name'] = project.name
        status['id'] = project.uid
        status['created'] = created_at
        status['updated'] = updated_at
        status['num_labeled_samples'] = num_labeled_samples
        status['members'] = members
        status['review'] = {'positive': positive, 'negative': negative, 'zero': zero}
        if log:
            logger.info('\nProject: %s\nID: %s\nCreated at: %s\nUpdated at: %s\nNumber of labeled samples: %d\nMembers:\n', project.name, project.uid, str(created_at), str(updated_at), num_labeled_samples)
            if not members:
                logger.info('\t-')
            for member in members:
                user = member.user()
                role = member.role()
                logger.info('\tUser: %s\n\tName: %s\n\tRole: %s\n\tEmail: %s\n\tID: %s\n', user.name, user.nickname, role.name, user.email, user.uid)
            logger.info('\nReviews:\n\tPositive: %d\n\tNegative: %d\n\tZero: %d', positive, negative, zero)
        return status

    @classmethod
    def _from_dict(cls, d, samples, config, anno_key):
        if False:
            for i in range(10):
                print('nop')
        return cls(samples, config, anno_key, d['id_map'], d['project_id'], d['frame_id_map'])

def import_from_labelbox(dataset, json_path, label_prefix=None, download_dir=None, labelbox_id_field='labelbox_id'):
    if False:
        while True:
            i = 10
    'Imports the labels from the Labelbox project into the FiftyOne dataset.\n\n    The ``labelbox_id_field`` of the FiftyOne samples are used to associate the\n    corresponding Labelbox labels.\n\n    If a ``download_dir`` is provided, any Labelbox IDs with no matching\n    FiftyOne sample are added to the FiftyOne dataset, and their media is\n    downloaded into ``download_dir``.\n\n    The provided ``json_path`` should contain a JSON file in the following\n    format::\n\n        [\n            {\n                "DataRow ID": <labelbox-id>,\n                "Labeled Data": <url-or-None>,\n                "Label": {...}\n            }\n        ]\n\n    When importing image labels, the ``Label`` field should contain a dict of\n    `Labelbox image labels <https://labelbox.com/docs/exporting-data/export-format-detail#images>`_::\n\n        {\n            "objects": [...],\n            "classifications": [...]\n        }\n\n    When importing video labels, the ``Label`` field should contain a dict as\n    follows::\n\n        {\n            "frames": <url-or-filepath>\n        }\n\n    where the ``frames`` field can either contain a URL, in which case the\n    file is downloaded from the web, or the path to NDJSON file on disk of\n    `Labelbox video labels <https://labelbox.com/docs/exporting-data/export-format-detail#video>`_::\n\n        {"frameNumber": 1, "objects": [...], "classifications": [...]}\n        {"frameNumber": 2, "objects": [...], "classifications": [...]}\n        ...\n\n    Args:\n        dataset: a :class:`fiftyone.core.dataset.Dataset`\n        json_path: the path to the Labelbox JSON export to load\n        label_prefix (None): a prefix to prepend to the sample label field(s)\n            that are created, separated by an underscore\n        download_dir (None): a directory into which to download the media for\n            any Labelbox IDs with no corresponding sample with the matching\n            ``labelbox_id_field`` value. This can be omitted if all IDs are\n            already present or you do not wish to download media and add new\n            samples\n        labelbox_id_field ("labelbox_id"): the sample field to lookup/store the\n            IDs of the Labelbox DataRows\n    '
    fov.validate_collection(dataset, media_type=(fomm.IMAGE, fomm.VIDEO))
    is_video = dataset.media_type == fomm.VIDEO
    if download_dir:
        filename_maker = fou.UniqueFilenameMaker(output_dir=download_dir)
    if labelbox_id_field not in dataset.get_field_schema():
        dataset.add_sample_field(labelbox_id_field, fof.StringField)
    id_map = {k: v for (k, v) in zip(*dataset.values([labelbox_id_field, 'id']))}
    if label_prefix:
        label_key = lambda k: label_prefix + '_' + k
    else:
        label_key = lambda k: k
    d_list = etas.read_json(json_path)
    with fou.ProgressBar() as pb:
        for d in pb(d_list):
            labelbox_id = d['DataRow ID']
            if labelbox_id in id_map:
                sample = dataset[id_map[labelbox_id]]
            elif download_dir:
                image_url = d['Labeled Data']
                filepath = filename_maker.get_output_path(image_url)
                etaw.download_file(image_url, path=filepath, quiet=True)
                sample = fos.Sample(filepath=filepath)
                dataset.add_sample(sample)
            else:
                logger.info("Skipping labels for unknown Labelbox ID '%s'; provide a `download_dir` if you wish to download media and create samples for new media", labelbox_id)
                continue
            if sample.metadata is None:
                if is_video:
                    sample.metadata = fom.VideoMetadata.build_for(sample.filepath)
                else:
                    sample.metadata = fom.ImageMetadata.build_for(sample.filepath)
            if is_video:
                frame_size = (sample.metadata.frame_width, sample.metadata.frame_height)
                frames = _parse_video_labels(d['Label'], frame_size)
                sample.frames.merge({frame_number: {label_key(fname): flabel for (fname, flabel) in frame_dict.items()} for (frame_number, frame_dict) in frames.items()})
            else:
                frame_size = (sample.metadata.width, sample.metadata.height)
                labels_dict = _parse_image_labels(d['Label'], frame_size)
                sample.update_fields({label_key(k): v for (k, v) in labels_dict.items()})
            sample.save()

def export_to_labelbox(sample_collection, ndjson_path, video_labels_dir=None, labelbox_id_field='labelbox_id', label_field=None, frame_labels_field=None):
    if False:
        while True:
            i = 10
    'Exports labels from the FiftyOne samples to Labelbox format.\n\n    This function is useful for loading predictions into Labelbox for\n    `model-assisted labeling <https://labelbox.com/docs/automation/model-assisted-labeling>`_.\n\n    You can use :meth:`upload_labels_to_labelbox` to upload the exported labels\n    to a Labelbox project.\n\n    You can use :meth:`upload_media_to_labelbox` to upload sample media to\n    Labelbox and populate the ``labelbox_id_field`` field, if necessary.\n\n    The IDs of the Labelbox DataRows corresponding to each sample must be\n    stored in the ``labelbox_id_field`` of the samples. Any samples with no\n    value in ``labelbox_id_field`` will be skipped.\n\n    When exporting frame labels for video datasets, the ``frames`` key of the\n    exported labels will contain the paths on disk to per-sample NDJSON files\n    that are written to ``video_labels_dir`` as follows::\n\n        video_labels_dir/\n            <labelbox-id1>.json\n            <labelbox-id2>.json\n            ...\n\n    where each NDJSON file contains the frame labels for the video with the\n    corresponding Labelbox ID.\n\n    Args:\n        sample_collection: a\n            :class:`fiftyone.core.collections.SampleCollection`\n        ndjson_path: the path to write an NDJSON export of the labels\n        video_labels_dir (None): a directory to write the per-sample video\n            labels. Only applicable for video datasets\n        labelbox_id_field ("labelbox_id"): the sample field to lookup/store the\n            IDs of the Labelbox DataRows\n        label_field (None): optional label field(s) to export. Can be any of\n            the following:\n\n            -   the name of a label field to export\n            -   a glob pattern of label field(s) to export\n            -   a list or tuple of label field(s) to export\n            -   a dictionary mapping label field names to keys to use when\n                constructing the exported labels\n\n            By default, no labels are exported\n        frame_labels_field (None): optional frame label field(s) to export.\n            Only applicable to video datasets. Can be any of the following:\n\n            -   the name of a frame label field to export\n            -   a glob pattern of frame label field(s) to export\n            -   a list or tuple of frame label field(s) to export\n            -   a dictionary mapping frame label field names to keys to use\n                when constructing the exported frame labels\n\n            By default, no frame labels are exported\n    '
    fov.validate_collection(sample_collection, media_type=(fomm.IMAGE, fomm.VIDEO))
    is_video = sample_collection.media_type == fomm.VIDEO
    label_fields = sample_collection._parse_label_field(label_field, allow_coercion=False, force_dict=True, required=False)
    if is_video:
        frame_label_fields = sample_collection._parse_frame_labels_field(frame_labels_field, allow_coercion=False, force_dict=True, required=False)
        if frame_label_fields and video_labels_dir is None:
            raise ValueError('Must provide `video_labels_dir` when exporting frame labels for video datasets')
    sample_collection.compute_metadata()
    etau.ensure_empty_file(ndjson_path)
    with fou.ProgressBar() as pb:
        for sample in pb(sample_collection):
            labelbox_id = sample[labelbox_id_field]
            if labelbox_id is None:
                logger.warning("Skipping sample '%s' with no '%s' value", sample.id, labelbox_id_field)
                continue
            if is_video:
                frame_size = (sample.metadata.frame_width, sample.metadata.frame_height)
            else:
                frame_size = (sample.metadata.width, sample.metadata.height)
            if label_fields:
                labels_dict = _get_labels(sample, label_fields)
                annos = _to_labelbox_image_labels(labels_dict, frame_size, labelbox_id)
                etas.write_ndjson(annos, ndjson_path, append=True)
            if is_video and frame_label_fields:
                frames = _get_frame_labels(sample, frame_label_fields)
                video_annos = _to_labelbox_video_labels(frames, frame_size, labelbox_id)
                video_labels_path = os.path.join(video_labels_dir, labelbox_id + '.json')
                etas.write_ndjson(video_annos, video_labels_path)
                anno = _make_video_anno(video_labels_path, data_row_id=labelbox_id)
                etas.write_ndjson([anno], ndjson_path, append=True)

def download_labels_from_labelbox(labelbox_project, outpath=None):
    if False:
        i = 10
        return i + 15
    'Downloads the labels for the given Labelbox project.\n\n    Args:\n        labelbox_project: a ``labelbox.schema.project.Project``\n        outpath (None): the path to write the JSON export on disk\n\n    Returns:\n        ``None`` if an ``outpath`` is provided, or the loaded JSON itself if no\n        ``outpath`` is provided\n    '
    export_url = labelbox_project.export_labels()
    if outpath:
        etaw.download_file(export_url, path=outpath)
        return None
    labels_bytes = etaw.download_file(export_url)
    return etas.load_json(labels_bytes)

def upload_media_to_labelbox(labelbox_dataset, sample_collection, labelbox_id_field='labelbox_id'):
    if False:
        i = 10
        return i + 15
    'Uploads the raw media for the FiftyOne samples to Labelbox.\n\n    The IDs of the Labelbox DataRows that are created are stored in the\n    ``labelbox_id_field`` of the samples.\n\n    Args:\n        labelbox_dataset: a ``labelbox.schema.dataset.Dataset`` to which to\n            add the media\n        sample_collection: a\n            :class:`fiftyone.core.collections.SampleCollection`\n        labelbox_id_field ("labelbox_id"): the sample field in which to store\n            the IDs of the Labelbox DataRows\n    '
    with fou.ProgressBar() as pb:
        for sample in pb(sample_collection):
            try:
                has_id = sample[labelbox_id_field] is not None
            except:
                has_id = False
            if has_id:
                logger.warning("Skipping sample '%s' with an existing '%s' value", sample.id, labelbox_id_field)
                continue
            filepath = sample.filepath
            data_row = labelbox_dataset.create_data_row(row_data=filepath)
            sample[labelbox_id_field] = data_row.uid
            sample.save()

def upload_labels_to_labelbox(labelbox_project, annos_or_ndjson_path, batch_size=None):
    if False:
        i = 10
        return i + 15
    'Uploads labels to a Labelbox project.\n\n    Use this function to load predictions into Labelbox for\n    `model-assisted labeling <https://labelbox.com/docs/automation/model-assisted-labeling>`_.\n\n    Use :meth:`export_to_labelbox` to export annotations in the format expected\n    by this method.\n\n    Args:\n        labelbox_project: a ``labelbox.schema.project.Project``\n        annos_or_ndjson_path: a list of annotation dicts or the path to an\n            NDJSON file on disk containing annotations\n        batch_size (None): an optional batch size to use when uploading the\n            annotations. By default, ``annos_or_ndjson_path`` is passed\n            directly to ``labelbox_project.upload_annotations()``\n    '
    if batch_size is None:
        name = '%s-upload-request' % labelbox_project.name
        return labelbox_project.upload_annotations(name, annos_or_ndjson_path)
    if etau.is_str(annos_or_ndjson_path):
        annos = etas.read_ndjson(annos_or_ndjson_path)
    else:
        annos = annos_or_ndjson_path
    requests = []
    count = 0
    for anno_batch in fou.iter_batches(annos, batch_size):
        count += 1
        name = '%s-upload-request-%d' % (labelbox_project.name, count)
        request = labelbox_project.upload_annotations(name, anno_batch)
        requests.append(request)
    return requests

def convert_labelbox_export_to_import(inpath, outpath=None, video_outdir=None):
    if False:
        return 10
    "Converts a Labelbox NDJSON export generated by\n    :meth:`export_to_labelbox` into the format expected by\n    :meth:`import_from_labelbox`.\n\n    The output JSON file will have the same format that is generated when\n    `exporting a Labelbox project's labels <https://labelbox.com/docs/exporting-data/export-overview>`_.\n\n    The ``Labeled Data`` fields of the output labels will be ``None``.\n\n    Args:\n        inpath: the path to an NDJSON file generated (for example) by\n            :meth:`export_to_labelbox`\n        outpath (None): the path to write a JSON file containing the converted\n            labels. If omitted, the input file will be overwritten\n        video_outdir (None): a directory to write the converted video frame\n            labels (if applicable). If omitted, the input frame label files\n            will be overwritten\n    "
    if outpath is None:
        outpath = inpath
    din_list = etas.read_ndjson(inpath)
    dout_map = {}
    for din in din_list:
        uuid = din.pop('dataRow')['id']
        din.pop('uuid')
        if 'frames' in din:
            frames_inpath = din['frames']
            if video_outdir is not None:
                frames_outpath = os.path.join(video_outdir, os.path.basename(frames_inpath))
            else:
                frames_outpath = frames_inpath
            _convert_labelbox_frames_export_to_import(frames_inpath, frames_outpath)
            dout_map[uuid] = {'DataRow ID': uuid, 'Labeled Data': None, 'Label': {'frames': frames_outpath}}
            continue
        if uuid not in dout_map:
            dout_map[uuid] = {'DataRow ID': uuid, 'Labeled Data': None, 'Label': {'objects': [], 'classifications': []}}
        _ingest_label(din, dout_map[uuid]['Label'])
    dout = list(dout_map.values())
    etas.write_json(dout, outpath)

def _convert_labelbox_frames_export_to_import(inpath, outpath):
    if False:
        print('Hello World!')
    din_list = etas.read_ndjson(inpath)
    dout_map = {}
    for din in din_list:
        frame_number = din.pop('frameNumber')
        din.pop('dataRow')
        din.pop('uuid')
        if frame_number not in dout_map:
            dout_map[frame_number] = {'frameNumber': frame_number, 'objects': [], 'classifications': []}
        _ingest_label(din, dout_map[frame_number])
    dout = [dout_map[fn] for fn in sorted(dout_map.keys())]
    etas.write_ndjson(dout, outpath)

def _ingest_label(din, d_label):
    if False:
        return 10
    if any((k in din for k in ('bbox', 'polygon', 'line', 'point', 'mask'))):
        if 'mask' in din:
            din['instanceURI'] = din.pop('mask')['instanceURI']
        d_label['objects'].append(din)
    else:
        d_label['classifications'].append(din)

def _get_labels(sample_or_frame, label_fields):
    if False:
        print('Hello World!')
    labels_dict = {}
    for (field, key) in label_fields.items():
        value = sample_or_frame[field]
        if value is not None:
            labels_dict[key] = value
    return labels_dict

def _get_frame_labels(sample, frame_label_fields):
    if False:
        for i in range(10):
            print('nop')
    frames = {}
    for (frame_number, frame) in sample.frames.items():
        frames[frame_number] = _get_labels(frame, frame_label_fields)
    return frames

def _to_labelbox_image_labels(labels_dict, frame_size, data_row_id):
    if False:
        for i in range(10):
            print('nop')
    annotations = []
    for (name, label) in labels_dict.items():
        if isinstance(label, (fol.Classification, fol.Classifications)):
            anno = _to_global_classification(name, label, data_row_id)
            annotations.append(anno)
        elif isinstance(label, (fol.Detection, fol.Detections)):
            annos = _to_detections(label, frame_size, data_row_id)
            annotations.extend(annos)
        elif isinstance(label, (fol.Polyline, fol.Polylines)):
            annos = _to_polylines(label, frame_size, data_row_id)
            annotations.extend(annos)
        elif isinstance(label, (fol.Keypoint, fol.Keypoints)):
            annos = _to_points(label, frame_size, data_row_id)
            annotations.extend(annos)
        elif isinstance(label, fol.Segmentation):
            annos = _to_mask(name, label, data_row_id)
            annotations.extend(annos)
        elif label is not None:
            msg = "Ignoring unsupported label type '%s'" % label.__class__
            warnings.warn(msg)
    return annotations

def _to_labelbox_video_labels(frames, frame_size, data_row_id):
    if False:
        for i in range(10):
            print('nop')
    annotations = []
    for (frame_number, labels_dict) in frames.items():
        frame_annos = _to_labelbox_image_labels(labels_dict, frame_size, data_row_id)
        for anno in frame_annos:
            anno['frameNumber'] = frame_number
            annotations.append(anno)
    return annotations

def _to_global_classification(name, label, data_row_id):
    if False:
        i = 10
        return i + 15
    anno = _make_base_anno(name, data_row_id=data_row_id)
    anno.update(_make_classification_answer(label))
    return anno

def _get_nested_classifications(label):
    if False:
        print('Hello World!')
    classifications = []
    for (name, value) in label.iter_attributes():
        if etau.is_str(value) or isinstance(value, (list, tuple)):
            anno = _make_base_anno(name)
            anno.update(_make_classification_answer(value))
            classifications.append(anno)
        else:
            msg = "Ignoring unsupported attribute type '%s'" % type(value)
            warnings.warn(msg)
            continue
    return classifications

def _to_mask(name, label, data_row_id):
    if False:
        while True:
            i = 10
    mask = np.asarray(label.get_mask())
    if mask.ndim < 3 or mask.dtype != np.uint8:
        raise ValueError('Segmentation masks must be stored as RGB color uint8 images')
    try:
        instance_uri = label.instance_uri
    except:
        raise ValueError('You must populate the `instance_uri` field of segmentation masks')
    colors = np.unique(np.reshape(mask, (-1, 3)), axis=0).tolist()
    annos = []
    base_anno = _make_base_anno(name, data_row_id=data_row_id)
    for color in colors:
        anno = copy(base_anno)
        anno['mask'] = _make_mask(instance_uri, color)
        annos.append(anno)
    return annos

def _to_detections(label, frame_size, data_row_id):
    if False:
        return 10
    if isinstance(label, fol.Detections):
        detections = label.detections
    else:
        detections = [label]
    annos = []
    for detection in detections:
        anno = _make_base_anno(detection.label, data_row_id=data_row_id)
        anno['bbox'] = _make_bbox(detection.bounding_box, frame_size)
        classifications = _get_nested_classifications(detection)
        if classifications:
            anno['classifications'] = classifications
        annos.append(anno)
    return annos

def _to_polylines(label, frame_size, data_row_id):
    if False:
        print('Hello World!')
    if isinstance(label, fol.Polylines):
        polylines = label.polylines
    else:
        polylines = [label]
    annos = []
    for polyline in polylines:
        field = 'polygon' if polyline.filled else 'line'
        classifications = _get_nested_classifications(polyline)
        for points in polyline.points:
            anno = _make_base_anno(polyline.label, data_row_id=data_row_id)
            anno[field] = [_make_point(point, frame_size) for point in points]
            if classifications:
                anno['classifications'] = classifications
            annos.append(anno)
    return annos

def _to_points(label, frame_size, data_row_id):
    if False:
        i = 10
        return i + 15
    if isinstance(label, fol.Keypoints):
        keypoints = label.keypoints
    else:
        keypoints = [keypoints]
    annos = []
    for keypoint in keypoints:
        classifications = _get_nested_classifications(keypoint)
        for point in keypoint.points:
            anno = _make_base_anno(keypoint.label, data_row_id=data_row_id)
            anno['point'] = _make_point(point, frame_size)
            if classifications:
                anno['classifications'] = classifications
            annos.append(anno)
    return annos

def _make_base_anno(value, data_row_id=None):
    if False:
        for i in range(10):
            print('nop')
    anno = {'uuid': str(uuid4()), 'schemaId': None, 'title': value, 'value': value}
    if data_row_id:
        anno['dataRow'] = {'id': data_row_id}
    return anno

def _make_video_anno(labels_path, data_row_id=None):
    if False:
        i = 10
        return i + 15
    anno = {'uuid': str(uuid4()), 'frames': labels_path}
    if data_row_id:
        anno['dataRow'] = {'id': data_row_id}
    return anno

def _make_classification_answer(label):
    if False:
        return 10
    if isinstance(label, fol.Classification):
        return {'answer': label.label}
    if isinstance(label, fol.Classifications):
        return {'answers': [{'value': c.label} for c in label.classifications]}
    if etau.is_str(label):
        return {'answer': label}
    if isinstance(label, (list, tuple)):
        return {'answers': [{'value': value} for value in label]}
    raise ValueError('Cannot convert %s to a classification' % label.__class__)

def _make_bbox(bounding_box, frame_size):
    if False:
        for i in range(10):
            print('nop')
    (x, y, w, h) = bounding_box
    (width, height) = frame_size
    return {'left': round(x * width, 1), 'top': round(y * height, 1), 'width': round(w * width, 1), 'height': round(h * height, 1)}

def _make_point(point, frame_size):
    if False:
        for i in range(10):
            print('nop')
    (x, y) = point
    (width, height) = frame_size
    return {'x': round(x * width, 1), 'y': round(y * height, 1)}

def _make_mask(instance_uri, color):
    if False:
        return 10
    return {'instanceURI': instance_uri, 'colorRGB': list(color)}

def _parse_video_labels(video_label_d, frame_size):
    if False:
        i = 10
        return i + 15
    url_or_filepath = video_label_d['frames']
    label_d_list = _download_or_load_ndjson(url_or_filepath)
    frames = {}
    for label_d in label_d_list:
        frame_number = label_d['frameNumber']
        frames[frame_number] = _parse_image_labels(label_d, frame_size)
    return frames

def _parse_image_labels(label_d, frame_size, class_attr=None):
    if False:
        while True:
            i = 10
    labels = {}
    cd_list = label_d.get('classifications', [])
    classifications = _parse_classifications(cd_list)
    labels.update(classifications)
    od_list = label_d.get('objects', [])
    objects = _parse_objects(od_list, frame_size, class_attr=class_attr)
    labels.update(objects)
    return labels

def _parse_classifications(cd_list):
    if False:
        i = 10
        return i + 15
    labels = {}
    for cd in cd_list:
        name = cd['value']
        if 'answer' in cd:
            answer = cd['answer']
            if isinstance(answer, list):
                labels[name] = fol.Classifications(classifications=[fol.Classification(label=a['value']) for a in answer])
            elif isinstance(answer, dict):
                labels[name] = fol.Classification(label=answer['value'])
            else:
                labels[name] = fol.Classification(label=answer)
        if 'answers' in cd:
            answers = cd['answers']
            labels[name] = fol.Classifications(classifications=[fol.Classification(label=a['value']) for a in answers])
    return labels

def _parse_attributes(cd_list):
    if False:
        print('Hello World!')
    attributes = {}
    for cd in cd_list:
        name = cd['value']
        if 'answer' in cd:
            answer = cd['answer']
            if isinstance(answer, list):
                answers = [_parse_attribute(a['value']) for a in answer]
                if len(answers) == 1:
                    answers = answers[0]
                attributes[name] = answers
            elif isinstance(answer, dict):
                attributes[name] = _parse_attribute(answer['value'])
            else:
                attributes[name] = _parse_attribute(answer)
        if 'answers' in cd:
            answer = cd['answers']
            attributes[name] = [_parse_attribute(a['value']) for a in answer]
    return attributes

def _parse_objects(od_list, frame_size, class_attr=None):
    if False:
        print('Hello World!')
    detections = []
    polylines = []
    keypoints = []
    segmentations = []
    mask = None
    mask_instance_uri = None
    label_fields = {}
    for od in od_list:
        attributes = _parse_attributes(od.get('classifications', []))
        load_fo_seg = class_attr is not None
        if class_attr and class_attr in attributes:
            label_field = od['title']
            label = attributes.pop(class_attr)
            if label_field not in label_fields:
                label_fields[label_field] = {}
        else:
            label = od['value']
            label_field = None
        if 'bbox' in od:
            bounding_box = _parse_bbox(od['bbox'], frame_size)
            det = fol.Detection(label=label, bounding_box=bounding_box, **attributes)
            if label_field is None:
                detections.append(det)
            else:
                if 'detections' not in label_fields[label_field]:
                    label_fields[label_field]['detections'] = []
                label_fields[label_field]['detections'].append(det)
        elif 'polygon' in od:
            points = _parse_points(od['polygon'], frame_size)
            polyline = fol.Polyline(label=label, points=[points], closed=True, filled=True, **attributes)
            if label_field is None:
                polylines.append(polyline)
            else:
                if 'polylines' not in label_fields[label_field]:
                    label_fields[label_field]['polylines'] = []
                label_fields[label_field]['polylines'].append(polyline)
        elif 'line' in od:
            points = _parse_points(od['line'], frame_size)
            polyline = fol.Polyline(label=label, points=[points], closed=True, filled=False, **attributes)
            if label_field is None:
                polylines.append(polyline)
            else:
                if 'polylines' not in label_fields[label_field]:
                    label_fields[label_field]['polylines'] = []
                label_fields[label_field]['polylines'].append(polyline)
        elif 'point' in od:
            point = _parse_point(od['point'], frame_size)
            keypoint = fol.Keypoint(label=label, points=[point], **attributes)
            if label_field is None:
                keypoints.append(keypoint)
            else:
                if 'keypoints' not in label_fields[label_field]:
                    label_fields[label_field]['keypoints'] = []
                label_fields[label_field]['keypoints'].append(keypoint)
        elif 'instanceURI' in od:
            if not load_fo_seg:
                if mask is None:
                    mask_instance_uri = od['instanceURI']
                    mask = _parse_mask(mask_instance_uri)
                    segmentation = {'mask': current_mask, 'label': label, 'attributes': attributes}
                elif od['instanceURI'] != mask_instance_uri:
                    msg = 'Only one segmentation mask per image/frame is allowed; skipping additional mask(s)'
                    warnings.warn(msg)
            else:
                current_mask_instance_uri = od['instanceURI']
                current_mask = _parse_mask(current_mask_instance_uri)
                segmentation = {'mask': current_mask, 'label': label, 'attributes': attributes}
                if label_field is not None:
                    if 'segmentation' not in label_fields[label_field]:
                        label_fields[label_field]['segmentation'] = []
                    label_fields[label_field]['segmentation'].append(segmentation)
                else:
                    segmentations.append(segmentation)
        else:
            msg = 'Ignoring unsupported label'
            warnings.warn(msg)
    labels = {}
    if detections:
        labels['detections'] = fol.Detections(detections=detections)
    if polylines:
        labels['polylines'] = fol.Polylines(polylines=polylines)
    if keypoints:
        labels['keypoints'] = fol.Keypoints(keypoints=keypoints)
    if mask is not None:
        labels['segmentation'] = mask
    elif segmentations:
        labels['segmentation'] = segmentations
    labels.update(label_fields)
    return labels

def _parse_bbox(bd, frame_size):
    if False:
        return 10
    (width, height) = frame_size
    x = bd['left'] / width
    y = bd['top'] / height
    w = bd['width'] / width
    h = bd['height'] / height
    return [x, y, w, h]

def _parse_points(pd_list, frame_size):
    if False:
        while True:
            i = 10
    return [_parse_point(pd, frame_size) for pd in pd_list]

def _parse_point(pd, frame_size):
    if False:
        while True:
            i = 10
    (width, height) = frame_size
    return (pd['x'] / width, pd['y'] / height)

def _parse_mask(instance_uri):
    if False:
        print('Hello World!')
    img_bytes = etaw.download_file(instance_uri, quiet=True)
    return etai.decode(img_bytes)

def _download_or_load_ndjson(url_or_filepath):
    if False:
        while True:
            i = 10
    if url_or_filepath.startswith('http'):
        ndjson_bytes = etaw.download_file(url_or_filepath, quiet=True)
        return etas.load_ndjson(ndjson_bytes)
    return etas.read_ndjson(url_or_filepath)

def _parse_attribute(value):
    if False:
        for i in range(10):
            print('nop')
    if value in {'True', 'true'}:
        return True
    if value in {'False', 'false'}:
        return False
    try:
        return int(value)
    except:
        pass
    try:
        return float(value)
    except:
        pass
    if value == 'None':
        return None
    return value

def _get_attr_type(label_schema, label_field, attr_name, class_name=None):
    if False:
        for i in range(10):
            print('nop')
    label_info = label_schema.get(label_field, {})
    classes = label_info.get('classes', [])
    global_attrs = label_info.get('attributes', {})
    if attr_name in global_attrs:
        return global_attrs[attr_name]['type']
    else:
        for _class in classes:
            if isinstance(_class, dict):
                _classes = _class['classes']
                if class_name in _classes:
                    _attrs = _class['attributes']
                    if attr_name in _attrs:
                        return _attrs[attr_name]['type']
    return None
_UNIQUE_TYPE_MAP = {'instance': 'segmentation', 'instances': 'segmentation', 'polygons': 'polylines', 'polygon': 'polylines', 'polyline': 'polylines'}