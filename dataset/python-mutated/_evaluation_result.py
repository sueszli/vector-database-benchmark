from typing import Dict
from azure.ai.generative.evaluate._utils import _get_ai_studio_url

class EvaluationResult(object):

    def __init__(self, metrics_summary: Dict[str, float], artifacts: Dict[str, str], **kwargs):
        if False:
            print('Hello World!')
        self._metrics_summary = metrics_summary
        self._artifacts = artifacts
        self._tracking_uri = kwargs.get('tracking_uri')
        self._evaluation_id = kwargs.get('evaluation_id')
        if self._tracking_uri:
            self._studio_url = _get_ai_studio_url(self._tracking_uri, self._evaluation_id)

    @property
    def metrics_summary(self) -> Dict[str, float]:
        if False:
            return 10
        return self._metrics_summary

    @property
    def artifacts(self) -> Dict[str, str]:
        if False:
            while True:
                i = 10
        return self._artifacts

    @property
    def tracking_uri(self) -> str:
        if False:
            while True:
                i = 10
        return self._tracking_uri

    @property
    def studio_url(self) -> str:
        if False:
            print('Hello World!')
        return self._studio_url

    def download_evaluation_artifacts(self, path: str) -> str:
        if False:
            return 10
        from mlflow.artifacts import download_artifacts
        for (artifact, artifact_uri) in self.artifacts.items():
            download_artifacts(artifact_uri=artifact_uri, tracking_uri=self.tracking_uri, dst_path=path)