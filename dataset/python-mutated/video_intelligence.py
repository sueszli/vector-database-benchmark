"""This module contains Google Cloud Vision operators."""
from __future__ import annotations
from typing import TYPE_CHECKING, Sequence
from google.api_core.gapic_v1.method import DEFAULT, _MethodDefault
from google.cloud.videointelligence_v1 import Feature, VideoContext
from google.protobuf.json_format import MessageToDict
from airflow.providers.google.cloud.hooks.video_intelligence import CloudVideoIntelligenceHook
from airflow.providers.google.cloud.operators.cloud_base import GoogleCloudBaseOperator
if TYPE_CHECKING:
    from google.api_core.retry import Retry
    from airflow.utils.context import Context

class CloudVideoIntelligenceDetectVideoLabelsOperator(GoogleCloudBaseOperator):
    """
    Performs video annotation, annotating video labels.

    .. seealso::
        For more information on how to use this operator, take a look at the guide:
        :ref:`howto/operator:CloudVideoIntelligenceDetectVideoLabelsOperator`.

    :param input_uri: Input video location. Currently, only Google Cloud Storage URIs are supported,
        which must be specified in the following format: ``gs://bucket-id/object-id``.
    :param input_content: The video data bytes.
        If unset, the input video(s) should be specified via ``input_uri``.
        If set, ``input_uri`` should be unset.
    :param output_uri: Optional, location where the output (in JSON format) should be stored. Currently, only
        Google Cloud Storage URIs are supported, which must be specified in the following format:
        ``gs://bucket-id/object-id``.
    :param video_context: Optional, Additional video context and/or feature-specific parameters.
    :param location: Optional, cloud region where annotation should take place. Supported cloud regions:
        us-east1, us-west1, europe-west1, asia-east1. If no region is specified, a region will be determined
        based on video file location.
    :param retry: Retry object used to determine when/if to retry requests.
        If None is specified, requests will not be retried.
    :param timeout: Optional, The amount of time, in seconds, to wait for the request to complete.
        Note that if retry is specified, the timeout applies to each individual attempt.
    :param gcp_conn_id: Optional, The connection ID used to connect to Google Cloud.
        Defaults to ``google_cloud_default``.
    :param impersonation_chain: Optional service account to impersonate using short-term
        credentials, or chained list of accounts required to get the access_token
        of the last account in the list, which will be impersonated in the request.
        If set as a string, the account must grant the originating account
        the Service Account Token Creator IAM role.
        If set as a sequence, the identities from the list must grant
        Service Account Token Creator IAM role to the directly preceding identity, with first
        account from the list granting this role to the originating account (templated).
    """
    template_fields: Sequence[str] = ('input_uri', 'output_uri', 'gcp_conn_id', 'impersonation_chain')

    def __init__(self, *, input_uri: str, input_content: bytes | None=None, output_uri: str | None=None, video_context: dict | VideoContext | None=None, location: str | None=None, retry: Retry | _MethodDefault=DEFAULT, timeout: float | None=None, gcp_conn_id: str='google_cloud_default', impersonation_chain: str | Sequence[str] | None=None, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.input_uri = input_uri
        self.input_content = input_content
        self.output_uri = output_uri
        self.video_context = video_context
        self.location = location
        self.retry = retry
        self.gcp_conn_id = gcp_conn_id
        self.timeout = timeout
        self.impersonation_chain = impersonation_chain

    def execute(self, context: Context):
        if False:
            while True:
                i = 10
        hook = CloudVideoIntelligenceHook(gcp_conn_id=self.gcp_conn_id, impersonation_chain=self.impersonation_chain)
        operation = hook.annotate_video(input_uri=self.input_uri, input_content=self.input_content, video_context=self.video_context, location=self.location, retry=self.retry, features=[Feature.LABEL_DETECTION], timeout=self.timeout)
        self.log.info('Processing video for label annotations')
        result = MessageToDict(operation.result()._pb)
        self.log.info('Finished processing.')
        return result

class CloudVideoIntelligenceDetectVideoExplicitContentOperator(GoogleCloudBaseOperator):
    """
    Performs video annotation, annotating explicit content.

    .. seealso::
        For more information on how to use this operator, take a look at the guide:
        :ref:`howto/operator:CloudVideoIntelligenceDetectVideoExplicitContentOperator`

    :param input_uri: Input video location. Currently, only Google Cloud Storage URIs are supported,
        which must be specified in the following format: ``gs://bucket-id/object-id``.
    :param input_content: The video data bytes.
        If unset, the input video(s) should be specified via ``input_uri``.
        If set, ``input_uri`` should be unset.
    :param output_uri: Optional, location where the output (in JSON format) should be stored. Currently, only
        Google Cloud Storage URIs are supported, which must be specified in the following format:
        ``gs://bucket-id/object-id``.
    :param video_context: Optional, Additional video context and/or feature-specific parameters.
    :param location: Optional, cloud region where annotation should take place. Supported cloud regions:
        us-east1, us-west1, europe-west1, asia-east1. If no region is specified, a region will be determined
        based on video file location.
    :param retry: Retry object used to determine when/if to retry requests.
        If None is specified, requests will not be retried.
    :param timeout: Optional, The amount of time, in seconds, to wait for the request to complete.
        Note that if retry is specified, the timeout applies to each individual attempt.
    :param gcp_conn_id: Optional, The connection ID used to connect to Google Cloud
        Defaults to ``google_cloud_default``.
    :param impersonation_chain: Optional service account to impersonate using short-term
        credentials, or chained list of accounts required to get the access_token
        of the last account in the list, which will be impersonated in the request.
        If set as a string, the account must grant the originating account
        the Service Account Token Creator IAM role.
        If set as a sequence, the identities from the list must grant
        Service Account Token Creator IAM role to the directly preceding identity, with first
        account from the list granting this role to the originating account (templated).
    """
    template_fields: Sequence[str] = ('input_uri', 'output_uri', 'gcp_conn_id', 'impersonation_chain')

    def __init__(self, *, input_uri: str, output_uri: str | None=None, input_content: bytes | None=None, video_context: dict | VideoContext | None=None, location: str | None=None, retry: Retry | _MethodDefault=DEFAULT, timeout: float | None=None, gcp_conn_id: str='google_cloud_default', impersonation_chain: str | Sequence[str] | None=None, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.input_uri = input_uri
        self.output_uri = output_uri
        self.input_content = input_content
        self.video_context = video_context
        self.location = location
        self.retry = retry
        self.gcp_conn_id = gcp_conn_id
        self.timeout = timeout
        self.impersonation_chain = impersonation_chain

    def execute(self, context: Context):
        if False:
            i = 10
            return i + 15
        hook = CloudVideoIntelligenceHook(gcp_conn_id=self.gcp_conn_id, impersonation_chain=self.impersonation_chain)
        operation = hook.annotate_video(input_uri=self.input_uri, input_content=self.input_content, video_context=self.video_context, location=self.location, retry=self.retry, features=[Feature.EXPLICIT_CONTENT_DETECTION], timeout=self.timeout)
        self.log.info('Processing video for explicit content annotations')
        result = MessageToDict(operation.result()._pb)
        self.log.info('Finished processing.')
        return result

class CloudVideoIntelligenceDetectVideoShotsOperator(GoogleCloudBaseOperator):
    """
    Performs video annotation, annotating video shots.

    .. seealso::
        For more information on how to use this operator, take a look at the guide:
        :ref:`howto/operator:CloudVideoIntelligenceDetectVideoShotsOperator`

    :param input_uri: Input video location. Currently, only Google Cloud Storage URIs are supported,
        which must be specified in the following format: ``gs://bucket-id/object-id``.
    :param input_content: The video data bytes.
        If unset, the input video(s) should be specified via ``input_uri``.
        If set, ``input_uri`` should be unset.
    :param output_uri: Optional, location where the output (in JSON format) should be stored. Currently, only
        Google Cloud Storage URIs are supported, which must be specified in the following format:
        ``gs://bucket-id/object-id``.
    :param video_context: Optional, Additional video context and/or feature-specific parameters.
    :param location: Optional, cloud region where annotation should take place. Supported cloud regions:
        us-east1, us-west1, europe-west1, asia-east1. If no region is specified, a region will be determined
        based on video file location.
    :param retry: Retry object used to determine when/if to retry requests.
        If None is specified, requests will not be retried.
    :param timeout: Optional, The amount of time, in seconds, to wait for the request to complete.
        Note that if retry is specified, the timeout applies to each individual attempt.
    :param gcp_conn_id: Optional, The connection ID used to connect to Google Cloud.
        Defaults to ``google_cloud_default``.
    :param impersonation_chain: Optional service account to impersonate using short-term
        credentials, or chained list of accounts required to get the access_token
        of the last account in the list, which will be impersonated in the request.
        If set as a string, the account must grant the originating account
        the Service Account Token Creator IAM role.
        If set as a sequence, the identities from the list must grant
        Service Account Token Creator IAM role to the directly preceding identity, with first
        account from the list granting this role to the originating account (templated).
    """
    template_fields: Sequence[str] = ('input_uri', 'output_uri', 'gcp_conn_id', 'impersonation_chain')

    def __init__(self, *, input_uri: str, output_uri: str | None=None, input_content: bytes | None=None, video_context: dict | VideoContext | None=None, location: str | None=None, retry: Retry | _MethodDefault=DEFAULT, timeout: float | None=None, gcp_conn_id: str='google_cloud_default', impersonation_chain: str | Sequence[str] | None=None, **kwargs) -> None:
        if False:
            return 10
        super().__init__(**kwargs)
        self.input_uri = input_uri
        self.output_uri = output_uri
        self.input_content = input_content
        self.video_context = video_context
        self.location = location
        self.retry = retry
        self.gcp_conn_id = gcp_conn_id
        self.timeout = timeout
        self.impersonation_chain = impersonation_chain

    def execute(self, context: Context):
        if False:
            while True:
                i = 10
        hook = CloudVideoIntelligenceHook(gcp_conn_id=self.gcp_conn_id, impersonation_chain=self.impersonation_chain)
        operation = hook.annotate_video(input_uri=self.input_uri, input_content=self.input_content, video_context=self.video_context, location=self.location, retry=self.retry, features=[Feature.SHOT_CHANGE_DETECTION], timeout=self.timeout)
        self.log.info('Processing video for video shots annotations')
        result = MessageToDict(operation.result()._pb)
        self.log.info('Finished processing.')
        return result