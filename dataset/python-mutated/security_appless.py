from __future__ import annotations
from typing import TYPE_CHECKING
from airflow.auth.managers.fab.security_manager.override import FabAirflowSecurityManagerOverride
if TYPE_CHECKING:
    from flask_session import Session

class FakeAppBuilder:
    """Stand-in class to replace a Flask App Builder.

    The only purpose is to provide the ``self.appbuilder.get_session`` interface
    for ``ApplessAirflowSecurityManager`` so it can be used without a real Flask
    app, which is slow to create.
    """

    def __init__(self, session: Session | None=None) -> None:
        if False:
            while True:
                i = 10
        self.get_session = session

class ApplessAirflowSecurityManager(FabAirflowSecurityManagerOverride):
    """Security Manager that doesn't need the whole flask app."""

    def __init__(self, session: Session | None=None):
        if False:
            return 10
        self.appbuilder = FakeAppBuilder(session)