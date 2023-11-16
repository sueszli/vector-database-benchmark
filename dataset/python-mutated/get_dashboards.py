from flask_appbuilder import SQLA
from superset.models.dashboard import Dashboard

def get_dashboards_ids(db: SQLA, dashboard_slugs: list[str]) -> list[int]:
    if False:
        print('Hello World!')
    result = db.session.query(Dashboard.id).filter(Dashboard.slug.in_(dashboard_slugs)).all()
    return [row[0] for row in result]