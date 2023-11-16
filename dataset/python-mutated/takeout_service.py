"""Functions to export the data of all user related models from a given
user_id.
"""
from __future__ import annotations
import json
import logging
import re
from core import feconf
from core import utils
from core.domain import fs_services
from core.domain import takeout_domain
from core.domain import user_services
from core.platform import models
from typing import List, Type
MYPY = False
if MYPY:
    from mypy_imports import base_models
(base_models,) = models.Registry.import_models([models.Names.BASE_MODEL])

def get_models_which_should_be_exported() -> List[Type[base_models.BaseModel]]:
    if False:
        print('Hello World!')
    'Returns list of models to export.\n\n    Returns:\n        list(datastore_services.Model). List of models whose data should be\n        exported.\n    '
    exempt_base_classes = ['BaseCommitLogEntryModel', 'BaseMapReduceBatchResultsModel', 'BaseModel', 'BaseSnapshotContentModel', 'BaseSnapshotMetadataModel', 'VersionedModel']
    return [model_class for model_class in models.Registry.get_all_storage_model_classes() if model_class.get_model_association_to_user() != base_models.MODEL_ASSOCIATION_TO_USER.NOT_CORRESPONDING_TO_USER and (not model_class.__name__ in exempt_base_classes)]

def export_data_for_user(user_id: str) -> takeout_domain.TakeoutData:
    if False:
        print('Hello World!')
    'Exports selected models according to model defined export_data functions.\n\n    Args:\n        user_id: str. The user_id of the user whose data is being exported.\n\n    Returns:\n        dict. Dictionary containing all user data in the following format:\n        {\n            <MODEL_NAME>_data: <dict of data in format as specified by\n                                model export policy>\n        }.\n\n    Raises:\n        NotImplementedError. Takeout for profile users is not implemented.\n    '
    user_settings = user_services.get_user_settings(user_id, strict=False)
    if user_settings is not None and feconf.ROLE_ID_MOBILE_LEARNER in user_settings.roles:
        raise NotImplementedError('Takeout for profile users is not yet supported.')
    exported_data = {}
    models_to_export = get_models_which_should_be_exported()
    for model in models_to_export:
        split_name = re.findall('[A-Z][^A-Z]*', model.__name__)[:-1]
        exported_model_data = model.export_data(user_id)
        exported_model_data_json_string = json.dumps(exported_model_data)
        user_id_match_object = re.search(feconf.USER_ID_REGEX, exported_model_data_json_string)
        if user_id_match_object:
            logging.error('[TAKEOUT] User ID (%s) found in the JSON generated for %s and user with ID %s' % (user_id_match_object.group(0), model.__name__, user_id))
        final_name = '_'.join([x.lower() for x in split_name])
        exported_data[final_name] = exported_model_data
    takeout_image_files: List[takeout_domain.TakeoutImage] = []
    if user_settings is not None:
        if user_settings.username is not None:
            fs = fs_services.GcsFileSystem(feconf.ENTITY_TYPE_USER, user_settings.username)
            filename_png = 'profile_picture.png'
            filename_webp = 'profile_picture.webp'
            image_data_png = utils.convert_image_binary_to_data_url(fs.get(filename_png), 'png')
            image_data_webp = utils.convert_image_binary_to_data_url(fs.get(filename_webp), 'webp')
            takeout_image_files.append(takeout_domain.TakeoutImage(image_data_png, 'user_settings_profile_picture.png'))
            takeout_image_files.append(takeout_domain.TakeoutImage(image_data_webp, 'user_settings_profile_picture.webp'))
    return takeout_domain.TakeoutData(exported_data, takeout_image_files)