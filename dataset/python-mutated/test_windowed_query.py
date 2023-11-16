import math
import pytest
from sqlalchemy import select
from sqlalchemy.orm import aliased
from warehouse.packaging.models import Project
from warehouse.utils.db.windowed_query import windowed_query
from ....common.db.packaging import ProjectFactory

@pytest.mark.parametrize('window_size', [1, 2])
def test_windowed_query(db_session, query_recorder, window_size):
    if False:
        for i in range(10):
            print('nop')
    projects = ProjectFactory.create_batch(10)
    project_set = {(project.name, project.id) for project in projects}
    expected = math.ceil(len(projects) / window_size) + 1
    subquery = select(Project.normalized_name).order_by(Project.id).subquery()
    pa = aliased(Project, subquery)
    query = select(Project.name).select_from(pa).distinct(Project.id)
    with query_recorder:
        assert set(windowed_query(db_session, query, Project.id, window_size)) == project_set
    assert len(query_recorder.queries) == expected