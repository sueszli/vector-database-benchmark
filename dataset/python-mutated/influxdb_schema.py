from marshmallow import EXCLUDE, Schema, fields, post_load, pre_load
from .dataclasses import FeatureEvaluationData, UsageData

class FeatureEvaluationDataSchema(Schema):
    count = fields.Integer(allow_none=True)
    day = fields.Date(allow_none=True)

    class Meta:
        unknown = EXCLUDE

    @pre_load
    def preprocess(self, data, **kwargs):
        if False:
            return 10
        day = data.pop('datetime')
        (_, count) = data.popitem()
        return {'day': day, 'count': count}

    @post_load
    def make_usage_data(self, data, **kwargs):
        if False:
            i = 10
            return i + 15
        return FeatureEvaluationData(**data)

class UsageDataSchema(Schema):
    flags = fields.Integer(data_key='Flags', allow_none=True)
    traits = fields.Integer(data_key='Traits', allow_none=True)
    identities = fields.Integer(data_key='Identities', allow_none=True)
    environment_document = fields.Integer(data_key='Environment-document', allow_none=True)
    day = fields.Date(data_key='name', allow_none=True)

    class Meta:
        unknown = EXCLUDE

    @post_load
    def make_usage_data(self, data, **kwargs):
        if False:
            return 10
        return UsageData(**data)