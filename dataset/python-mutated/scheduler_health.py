from __future__ import annotations
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
from sqlalchemy import select
from airflow.configuration import conf
from airflow.jobs.job import Job
from airflow.jobs.scheduler_job_runner import SchedulerJobRunner
from airflow.utils.net import get_hostname
from airflow.utils.session import create_session
log = logging.getLogger(__name__)

class HealthServer(BaseHTTPRequestHandler):
    """Small webserver to serve scheduler health check."""

    def do_GET(self):
        if False:
            i = 10
            return i + 15
        if self.path == '/health':
            try:
                with create_session() as session:
                    scheduler_job = session.scalar(select(Job).filter_by(job_type=SchedulerJobRunner.job_type).filter_by(hostname=get_hostname()).order_by(Job.latest_heartbeat.desc()).limit(1))
                if scheduler_job and scheduler_job.is_alive():
                    self.send_response(200)
                    self.end_headers()
                else:
                    self.send_error(503)
            except Exception:
                log.exception('Exception when executing Health check')
                self.send_error(503)
        else:
            self.send_error(404)

def serve_health_check():
    if False:
        return 10
    health_check_port = conf.getint('scheduler', 'SCHEDULER_HEALTH_CHECK_SERVER_PORT')
    httpd = HTTPServer(('0.0.0.0', health_check_port), HealthServer)
    httpd.serve_forever()
if __name__ == '__main__':
    serve_health_check()