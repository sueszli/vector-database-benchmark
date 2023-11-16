from __future__ import annotations
from importlib import import_module
from typing import TYPE_CHECKING
from sqlalchemy import Column, Integer, String
from airflow.models.base import Base
from airflow.utils import timezone
from airflow.utils.sqlalchemy import ExtendedJSON, UtcDateTime
if TYPE_CHECKING:
    from airflow.callbacks.callback_requests import CallbackRequest

class DbCallbackRequest(Base):
    """Used to handle callbacks through database."""
    __tablename__ = 'callback_request'
    id = Column(Integer(), nullable=False, primary_key=True)
    created_at = Column(UtcDateTime, default=timezone.utcnow, nullable=False)
    priority_weight = Column(Integer(), nullable=False)
    callback_data = Column(ExtendedJSON, nullable=False)
    callback_type = Column(String(20), nullable=False)
    processor_subdir = Column(String(2000), nullable=True)

    def __init__(self, priority_weight: int, callback: CallbackRequest):
        if False:
            while True:
                i = 10
        self.created_at = timezone.utcnow()
        self.priority_weight = priority_weight
        self.processor_subdir = callback.processor_subdir
        self.callback_data = callback.to_json()
        self.callback_type = callback.__class__.__name__

    def get_callback_request(self) -> CallbackRequest:
        if False:
            return 10
        module = import_module('airflow.callbacks.callback_requests')
        callback_class = getattr(module, self.callback_type)
        from_json = getattr(callback_class, 'from_json')
        return from_json(self.callback_data)