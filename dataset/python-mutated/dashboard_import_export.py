import logging
from sqlalchemy.orm import Session
from superset.models.dashboard import Dashboard
logger = logging.getLogger(__name__)

def export_dashboards(session: Session) -> str:
    if False:
        return 10
    'Returns all dashboards metadata as a json dump'
    logger.info('Starting export')
    dashboards = session.query(Dashboard)
    dashboard_ids = set()
    for dashboard in dashboards:
        dashboard_ids.add(dashboard.id)
    data = Dashboard.export_dashboards(dashboard_ids)
    return data