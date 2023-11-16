from __future__ import annotations
from unittest.mock import MagicMock, patch
import pytest
from airflow.exceptions import AirflowException, AirflowSkipException
from airflow.providers.jenkins.hooks.jenkins import JenkinsHook
from airflow.providers.jenkins.sensors.jenkins import JenkinsBuildSensor

class TestJenkinsBuildSensor:

    @pytest.mark.parametrize('build_number, build_state, result', [(None, True, ''), (3, True, '')])
    @patch('jenkins.Jenkins')
    def test_poke_buliding(self, mock_jenkins, build_number, build_state, result):
        if False:
            i = 10
            return i + 15
        target_build_number = build_number or 10
        jenkins_mock = MagicMock()
        jenkins_mock.get_job_info.return_value = {'lastBuild': {'number': target_build_number}}
        jenkins_mock.get_build_info.return_value = {'building': build_state}
        mock_jenkins.return_value = jenkins_mock
        with patch.object(JenkinsHook, 'get_connection') as mock_get_connection:
            mock_get_connection.return_value = MagicMock()
            sensor = JenkinsBuildSensor(dag=None, jenkins_connection_id='fake_jenkins_connection', task_id='sensor_test', job_name='a_job_on_jenkins', build_number=target_build_number, target_states=['SUCCESS'])
            output = sensor.poke(None)
            assert output == (not build_state)
            assert jenkins_mock.get_job_info.call_count == 0 if build_number else 1
            jenkins_mock.get_build_info.assert_called_once_with('a_job_on_jenkins', target_build_number)

    @pytest.mark.parametrize('soft_fail, expected_exception', ((False, AirflowException), (True, AirflowSkipException)))
    @pytest.mark.parametrize('build_number, build_state, result', [(1, False, 'SUCCESS'), (2, False, 'FAILED')])
    @patch('jenkins.Jenkins')
    def test_poke_finish_building(self, mock_jenkins, build_number, build_state, result, soft_fail, expected_exception):
        if False:
            while True:
                i = 10
        target_build_number = build_number or 10
        jenkins_mock = MagicMock()
        jenkins_mock.get_job_info.return_value = {'lastBuild': {'number': target_build_number}}
        jenkins_mock.get_build_info.return_value = {'building': build_state, 'result': result}
        mock_jenkins.return_value = jenkins_mock
        with patch.object(JenkinsHook, 'get_connection') as mock_get_connection:
            mock_get_connection.return_value = MagicMock()
            sensor = JenkinsBuildSensor(dag=None, jenkins_connection_id='fake_jenkins_connection', task_id='sensor_test', job_name='a_job_on_jenkins', build_number=target_build_number, target_states=['SUCCESS'], soft_fail=soft_fail)
            if result not in sensor.target_states:
                with pytest.raises(expected_exception):
                    sensor.poke(None)
                    assert jenkins_mock.get_build_info.call_count == 2
            else:
                output = sensor.poke(None)
                assert output == (not build_state)
                assert jenkins_mock.get_job_info.call_count == 0 if build_number else 1
                assert jenkins_mock.get_build_info.call_count == 2