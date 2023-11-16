import os
from django.db import models
from ..jenkins import get_job_status
from .base import StatusCheck, StatusCheckResult

class JenkinsStatusCheck(StatusCheck):
    jenkins_config = models.ForeignKey('JenkinsConfig')

    @property
    def check_category(self):
        if False:
            print('Hello World!')
        return 'Jenkins check'

    @property
    def failing_short_status(self):
        if False:
            while True:
                i = 10
        return 'Job failing on Jenkins'

    def _run(self):
        if False:
            return 10
        result = StatusCheckResult(status_check=self)
        try:
            status = get_job_status(self.jenkins_config, self.name)
            active = status['active']
            result.job_number = status['job_number']
            result.consecutive_failures = status['consecutive_failures']
            if status['status_code'] == 404:
                result.error = u'Job %s not found on Jenkins' % self.name
                result.succeeded = False
                return result
            elif status['status_code'] > 400:
                raise Exception(u'returned %s' % status['status_code'])
        except Exception as e:
            result.error = u'Error fetching from Jenkins - %s' % e.message
            result.succeeded = True
            return result
        if not active:
            result.error = u'Job "%s" disabled on Jenkins' % self.name
            result.succeeded = False
        else:
            if self.max_queued_build_time and status['blocked_build_time']:
                if status['blocked_build_time'] > self.max_queued_build_time * 60:
                    result.succeeded = False
                    result.error = u'Job "%s" has blocked build waiting for %ss (> %sm)' % (self.name, int(status['blocked_build_time']), self.max_queued_build_time)
                    result.job_number = status['queued_job_number']
                else:
                    result.succeeded = status['succeeded']
            else:
                result.succeeded = status['succeeded']
            if not status['succeeded']:
                message = u'Job "%s" failing on Jenkins (%s)' % (self.name, status['consecutive_failures'])
                if result.error:
                    result.error += u'; %s' % message
                else:
                    result.error = message
                result.raw_data = status
        return result

    def calculate_debounced_passing(self, recent_results, debounce=0):
        if False:
            for i in range(10):
                print('nop')
        '\n        `debounce` is the number of previous job failures we need (not including this)\n        to mark a search as passing or failing\n        Returns:\n          True if passing given debounce factor\n          False if failing\n        '
        last_result = recent_results[0]
        return last_result.consecutive_failures <= debounce

class JenkinsConfig(models.Model):
    name = models.CharField(max_length=30, blank=False)
    jenkins_api = models.CharField(max_length=2000, blank=False)
    jenkins_user = models.CharField(max_length=2000, blank=False)
    jenkins_pass = models.CharField(max_length=2000, blank=False)

    def __str__(self):
        if False:
            while True:
                i = 10
        return self.name

def create_default_jenkins_config():
    if False:
        print('Hello World!')
    if not JenkinsConfig.objects.exists():
        if os.environ.get('JENKINS_API'):
            JenkinsConfig.objects.create(name='Default Jenkins', jenkins_api=os.environ.get('JENKINS_API', 'http://jenkins.example.com'), jenkins_user=os.environ.get('JENKINS_USER', ''), jenkins_pass=os.environ.get('JENKINS_PASS', ''))