from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Optional, Union
from flask import current_app, g
from flask_appbuilder import Model
from marshmallow import post_load, pre_load, Schema, ValidationError
from sqlalchemy.orm.exc import NoResultFound
from superset.utils.core import get_user_id

def validate_owner(value: int) -> None:
    if False:
        while True:
            i = 10
    try:
        current_app.appbuilder.get_session.query(current_app.appbuilder.sm.user_model.id).filter_by(id=value).one()
    except NoResultFound as ex:
        raise ValidationError(f'User {value} does not exist') from ex

class BaseSupersetSchema(Schema):
    """
    Extends Marshmallow schema so that we can pass a Model to load
    (following marshmallow-sqlalchemy pattern). This is useful
    to perform partial model merges on HTTP PUT
    """
    __class_model__: Model = None

    def __init__(self, **kwargs: Any) -> None:
        if False:
            return 10
        self.instance: Optional[Model] = None
        super().__init__(**kwargs)

    def load(self, data: Union[Mapping[str, Any], Iterable[Mapping[str, Any]]], many: Optional[bool]=None, partial: Union[bool, Sequence[str], set[str], None]=None, instance: Optional[Model]=None, **kwargs: Any) -> Any:
        if False:
            while True:
                i = 10
        self.instance = instance
        if many is None:
            many = False
        if partial is None:
            partial = False
        return super().load(data, many=many, partial=partial, **kwargs)

    @post_load
    def make_object(self, data: dict[Any, Any], discard: Optional[list[str]]=None) -> Model:
        if False:
            i = 10
            return i + 15
        '\n        Creates a Model object from POST or PUT requests. PUT will use self.instance\n        previously fetched from the endpoint handler\n\n        :param data: Schema data payload\n        :param discard: List of fields to not set on the model\n        '
        discard = discard or []
        if not self.instance:
            self.instance = self.__class_model__()
        for field in data:
            if field not in discard:
                setattr(self.instance, field, data.get(field))
        return self.instance

class BaseOwnedSchema(BaseSupersetSchema):
    """
    Implements owners validation,pre load and post_load
    (to populate the owners field) on Marshmallow schemas
    """
    owners_field_name = 'owners'

    @post_load
    def make_object(self, data: dict[str, Any], discard: Optional[list[str]]=None) -> Model:
        if False:
            i = 10
            return i + 15
        discard = discard or []
        discard.append(self.owners_field_name)
        instance = super().make_object(data, discard)
        if 'owners' not in data and g.user not in instance.owners:
            instance.owners.append(g.user)
        if self.owners_field_name in data:
            self.set_owners(instance, data[self.owners_field_name])
        return instance

    @pre_load
    def pre_load(self, data: dict[Any, Any]) -> None:
        if False:
            while True:
                i = 10
        if not self.instance:
            data[self.owners_field_name] = data.get(self.owners_field_name, [])

    @staticmethod
    def set_owners(instance: Model, owners: list[int]) -> None:
        if False:
            for i in range(10):
                print('nop')
        owner_objs = []
        user_id = get_user_id()
        if user_id and user_id not in owners:
            owners.append(user_id)
        for owner_id in owners:
            user = current_app.appbuilder.get_session.query(current_app.appbuilder.sm.user_model).get(owner_id)
            owner_objs.append(user)
        instance.owners = owner_objs