from typing import Awaitable, Callable, Dict, Optional, Sequence, Tuple, Union
import warnings
from google.api_core import gapic_v1, grpc_helpers_async
from google.auth import credentials as ga_credentials
from google.auth.transport.grpc import SslCredentials
from google.protobuf import empty_pb2
import grpc
from grpc.experimental import aio
from google.cloud.container_v1beta1.types import cluster_service
from .base import DEFAULT_CLIENT_INFO, ClusterManagerTransport
from .grpc import ClusterManagerGrpcTransport

class ClusterManagerGrpcAsyncIOTransport(ClusterManagerTransport):
    """gRPC AsyncIO backend transport for ClusterManager.

    Google Kubernetes Engine Cluster Manager v1beta1

    This class defines the same methods as the primary client, so the
    primary client can load the underlying transport implementation
    and call it.

    It sends protocol buffers over the wire using gRPC (which is built on
    top of HTTP/2); the ``grpcio`` package must be installed.
    """
    _grpc_channel: aio.Channel
    _stubs: Dict[str, Callable] = {}

    @classmethod
    def create_channel(cls, host: str='container.googleapis.com', credentials: Optional[ga_credentials.Credentials]=None, credentials_file: Optional[str]=None, scopes: Optional[Sequence[str]]=None, quota_project_id: Optional[str]=None, **kwargs) -> aio.Channel:
        if False:
            while True:
                i = 10
        'Create and return a gRPC AsyncIO channel object.\n        Args:\n            host (Optional[str]): The host for the channel to use.\n            credentials (Optional[~.Credentials]): The\n                authorization credentials to attach to requests. These\n                credentials identify this application to the service. If\n                none are specified, the client will attempt to ascertain\n                the credentials from the environment.\n            credentials_file (Optional[str]): A file with credentials that can\n                be loaded with :func:`google.auth.load_credentials_from_file`.\n                This argument is ignored if ``channel`` is provided.\n            scopes (Optional[Sequence[str]]): A optional list of scopes needed for this\n                service. These are only used when credentials are not specified and\n                are passed to :func:`google.auth.default`.\n            quota_project_id (Optional[str]): An optional project to use for billing\n                and quota.\n            kwargs (Optional[dict]): Keyword arguments, which are passed to the\n                channel creation.\n        Returns:\n            aio.Channel: A gRPC AsyncIO channel object.\n        '
        return grpc_helpers_async.create_channel(host, credentials=credentials, credentials_file=credentials_file, quota_project_id=quota_project_id, default_scopes=cls.AUTH_SCOPES, scopes=scopes, default_host=cls.DEFAULT_HOST, **kwargs)

    def __init__(self, *, host: str='container.googleapis.com', credentials: Optional[ga_credentials.Credentials]=None, credentials_file: Optional[str]=None, scopes: Optional[Sequence[str]]=None, channel: Optional[aio.Channel]=None, api_mtls_endpoint: Optional[str]=None, client_cert_source: Optional[Callable[[], Tuple[bytes, bytes]]]=None, ssl_channel_credentials: Optional[grpc.ChannelCredentials]=None, client_cert_source_for_mtls: Optional[Callable[[], Tuple[bytes, bytes]]]=None, quota_project_id: Optional[str]=None, client_info: gapic_v1.client_info.ClientInfo=DEFAULT_CLIENT_INFO, always_use_jwt_access: Optional[bool]=False, api_audience: Optional[str]=None) -> None:
        if False:
            return 10
        "Instantiate the transport.\n\n        Args:\n            host (Optional[str]):\n                 The hostname to connect to.\n            credentials (Optional[google.auth.credentials.Credentials]): The\n                authorization credentials to attach to requests. These\n                credentials identify the application to the service; if none\n                are specified, the client will attempt to ascertain the\n                credentials from the environment.\n                This argument is ignored if ``channel`` is provided.\n            credentials_file (Optional[str]): A file with credentials that can\n                be loaded with :func:`google.auth.load_credentials_from_file`.\n                This argument is ignored if ``channel`` is provided.\n            scopes (Optional[Sequence[str]]): A optional list of scopes needed for this\n                service. These are only used when credentials are not specified and\n                are passed to :func:`google.auth.default`.\n            channel (Optional[aio.Channel]): A ``Channel`` instance through\n                which to make calls.\n            api_mtls_endpoint (Optional[str]): Deprecated. The mutual TLS endpoint.\n                If provided, it overrides the ``host`` argument and tries to create\n                a mutual TLS channel with client SSL credentials from\n                ``client_cert_source`` or application default SSL credentials.\n            client_cert_source (Optional[Callable[[], Tuple[bytes, bytes]]]):\n                Deprecated. A callback to provide client SSL certificate bytes and\n                private key bytes, both in PEM format. It is ignored if\n                ``api_mtls_endpoint`` is None.\n            ssl_channel_credentials (grpc.ChannelCredentials): SSL credentials\n                for the grpc channel. It is ignored if ``channel`` is provided.\n            client_cert_source_for_mtls (Optional[Callable[[], Tuple[bytes, bytes]]]):\n                A callback to provide client certificate bytes and private key bytes,\n                both in PEM format. It is used to configure a mutual TLS channel. It is\n                ignored if ``channel`` or ``ssl_channel_credentials`` is provided.\n            quota_project_id (Optional[str]): An optional project to use for billing\n                and quota.\n            client_info (google.api_core.gapic_v1.client_info.ClientInfo):\n                The client info used to send a user-agent string along with\n                API requests. If ``None``, then default info will be used.\n                Generally, you only need to set this if you're developing\n                your own client library.\n            always_use_jwt_access (Optional[bool]): Whether self signed JWT should\n                be used for service account credentials.\n\n        Raises:\n            google.auth.exceptions.MutualTlsChannelError: If mutual TLS transport\n              creation failed for any reason.\n          google.api_core.exceptions.DuplicateCredentialArgs: If both ``credentials``\n              and ``credentials_file`` are passed.\n        "
        self._grpc_channel = None
        self._ssl_channel_credentials = ssl_channel_credentials
        self._stubs: Dict[str, Callable] = {}
        if api_mtls_endpoint:
            warnings.warn('api_mtls_endpoint is deprecated', DeprecationWarning)
        if client_cert_source:
            warnings.warn('client_cert_source is deprecated', DeprecationWarning)
        if channel:
            credentials = False
            self._grpc_channel = channel
            self._ssl_channel_credentials = None
        elif api_mtls_endpoint:
            host = api_mtls_endpoint
            if client_cert_source:
                (cert, key) = client_cert_source()
                self._ssl_channel_credentials = grpc.ssl_channel_credentials(certificate_chain=cert, private_key=key)
            else:
                self._ssl_channel_credentials = SslCredentials().ssl_credentials
        elif client_cert_source_for_mtls and (not ssl_channel_credentials):
            (cert, key) = client_cert_source_for_mtls()
            self._ssl_channel_credentials = grpc.ssl_channel_credentials(certificate_chain=cert, private_key=key)
        super().__init__(host=host, credentials=credentials, credentials_file=credentials_file, scopes=scopes, quota_project_id=quota_project_id, client_info=client_info, always_use_jwt_access=always_use_jwt_access, api_audience=api_audience)
        if not self._grpc_channel:
            self._grpc_channel = type(self).create_channel(self._host, credentials=self._credentials, credentials_file=None, scopes=self._scopes, ssl_credentials=self._ssl_channel_credentials, quota_project_id=quota_project_id, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])
        self._prep_wrapped_messages(client_info)

    @property
    def grpc_channel(self) -> aio.Channel:
        if False:
            return 10
        'Create the channel designed to connect to this service.\n\n        This property caches on the instance; repeated calls return\n        the same channel.\n        '
        return self._grpc_channel

    @property
    def list_clusters(self) -> Callable[[cluster_service.ListClustersRequest], Awaitable[cluster_service.ListClustersResponse]]:
        if False:
            for i in range(10):
                print('nop')
        'Return a callable for the list clusters method over gRPC.\n\n        Lists all clusters owned by a project in either the\n        specified zone or all zones.\n\n        Returns:\n            Callable[[~.ListClustersRequest],\n                    Awaitable[~.ListClustersResponse]]:\n                A function that, when called, will call the underlying RPC\n                on the server.\n        '
        if 'list_clusters' not in self._stubs:
            self._stubs['list_clusters'] = self.grpc_channel.unary_unary('/google.container.v1beta1.ClusterManager/ListClusters', request_serializer=cluster_service.ListClustersRequest.serialize, response_deserializer=cluster_service.ListClustersResponse.deserialize)
        return self._stubs['list_clusters']

    @property
    def get_cluster(self) -> Callable[[cluster_service.GetClusterRequest], Awaitable[cluster_service.Cluster]]:
        if False:
            while True:
                i = 10
        'Return a callable for the get cluster method over gRPC.\n\n        Gets the details for a specific cluster.\n\n        Returns:\n            Callable[[~.GetClusterRequest],\n                    Awaitable[~.Cluster]]:\n                A function that, when called, will call the underlying RPC\n                on the server.\n        '
        if 'get_cluster' not in self._stubs:
            self._stubs['get_cluster'] = self.grpc_channel.unary_unary('/google.container.v1beta1.ClusterManager/GetCluster', request_serializer=cluster_service.GetClusterRequest.serialize, response_deserializer=cluster_service.Cluster.deserialize)
        return self._stubs['get_cluster']

    @property
    def create_cluster(self) -> Callable[[cluster_service.CreateClusterRequest], Awaitable[cluster_service.Operation]]:
        if False:
            for i in range(10):
                print('nop')
        "Return a callable for the create cluster method over gRPC.\n\n        Creates a cluster, consisting of the specified number and type\n        of Google Compute Engine instances.\n\n        By default, the cluster is created in the project's `default\n        network <https://cloud.google.com/compute/docs/networks-and-firewalls#networks>`__.\n\n        One firewall is added for the cluster. After cluster creation,\n        the Kubelet creates routes for each node to allow the containers\n        on that node to communicate with all other instances in the\n        cluster.\n\n        Finally, an entry is added to the project's global metadata\n        indicating which CIDR range the cluster is using.\n\n        Returns:\n            Callable[[~.CreateClusterRequest],\n                    Awaitable[~.Operation]]:\n                A function that, when called, will call the underlying RPC\n                on the server.\n        "
        if 'create_cluster' not in self._stubs:
            self._stubs['create_cluster'] = self.grpc_channel.unary_unary('/google.container.v1beta1.ClusterManager/CreateCluster', request_serializer=cluster_service.CreateClusterRequest.serialize, response_deserializer=cluster_service.Operation.deserialize)
        return self._stubs['create_cluster']

    @property
    def update_cluster(self) -> Callable[[cluster_service.UpdateClusterRequest], Awaitable[cluster_service.Operation]]:
        if False:
            print('Hello World!')
        'Return a callable for the update cluster method over gRPC.\n\n        Updates the settings for a specific cluster.\n\n        Returns:\n            Callable[[~.UpdateClusterRequest],\n                    Awaitable[~.Operation]]:\n                A function that, when called, will call the underlying RPC\n                on the server.\n        '
        if 'update_cluster' not in self._stubs:
            self._stubs['update_cluster'] = self.grpc_channel.unary_unary('/google.container.v1beta1.ClusterManager/UpdateCluster', request_serializer=cluster_service.UpdateClusterRequest.serialize, response_deserializer=cluster_service.Operation.deserialize)
        return self._stubs['update_cluster']

    @property
    def update_node_pool(self) -> Callable[[cluster_service.UpdateNodePoolRequest], Awaitable[cluster_service.Operation]]:
        if False:
            for i in range(10):
                print('nop')
        'Return a callable for the update node pool method over gRPC.\n\n        Updates the version and/or image type of a specific\n        node pool.\n\n        Returns:\n            Callable[[~.UpdateNodePoolRequest],\n                    Awaitable[~.Operation]]:\n                A function that, when called, will call the underlying RPC\n                on the server.\n        '
        if 'update_node_pool' not in self._stubs:
            self._stubs['update_node_pool'] = self.grpc_channel.unary_unary('/google.container.v1beta1.ClusterManager/UpdateNodePool', request_serializer=cluster_service.UpdateNodePoolRequest.serialize, response_deserializer=cluster_service.Operation.deserialize)
        return self._stubs['update_node_pool']

    @property
    def set_node_pool_autoscaling(self) -> Callable[[cluster_service.SetNodePoolAutoscalingRequest], Awaitable[cluster_service.Operation]]:
        if False:
            return 10
        'Return a callable for the set node pool autoscaling method over gRPC.\n\n        Sets the autoscaling settings of a specific node\n        pool.\n\n        Returns:\n            Callable[[~.SetNodePoolAutoscalingRequest],\n                    Awaitable[~.Operation]]:\n                A function that, when called, will call the underlying RPC\n                on the server.\n        '
        if 'set_node_pool_autoscaling' not in self._stubs:
            self._stubs['set_node_pool_autoscaling'] = self.grpc_channel.unary_unary('/google.container.v1beta1.ClusterManager/SetNodePoolAutoscaling', request_serializer=cluster_service.SetNodePoolAutoscalingRequest.serialize, response_deserializer=cluster_service.Operation.deserialize)
        return self._stubs['set_node_pool_autoscaling']

    @property
    def set_logging_service(self) -> Callable[[cluster_service.SetLoggingServiceRequest], Awaitable[cluster_service.Operation]]:
        if False:
            while True:
                i = 10
        'Return a callable for the set logging service method over gRPC.\n\n        Sets the logging service for a specific cluster.\n\n        Returns:\n            Callable[[~.SetLoggingServiceRequest],\n                    Awaitable[~.Operation]]:\n                A function that, when called, will call the underlying RPC\n                on the server.\n        '
        if 'set_logging_service' not in self._stubs:
            self._stubs['set_logging_service'] = self.grpc_channel.unary_unary('/google.container.v1beta1.ClusterManager/SetLoggingService', request_serializer=cluster_service.SetLoggingServiceRequest.serialize, response_deserializer=cluster_service.Operation.deserialize)
        return self._stubs['set_logging_service']

    @property
    def set_monitoring_service(self) -> Callable[[cluster_service.SetMonitoringServiceRequest], Awaitable[cluster_service.Operation]]:
        if False:
            while True:
                i = 10
        'Return a callable for the set monitoring service method over gRPC.\n\n        Sets the monitoring service for a specific cluster.\n\n        Returns:\n            Callable[[~.SetMonitoringServiceRequest],\n                    Awaitable[~.Operation]]:\n                A function that, when called, will call the underlying RPC\n                on the server.\n        '
        if 'set_monitoring_service' not in self._stubs:
            self._stubs['set_monitoring_service'] = self.grpc_channel.unary_unary('/google.container.v1beta1.ClusterManager/SetMonitoringService', request_serializer=cluster_service.SetMonitoringServiceRequest.serialize, response_deserializer=cluster_service.Operation.deserialize)
        return self._stubs['set_monitoring_service']

    @property
    def set_addons_config(self) -> Callable[[cluster_service.SetAddonsConfigRequest], Awaitable[cluster_service.Operation]]:
        if False:
            i = 10
            return i + 15
        'Return a callable for the set addons config method over gRPC.\n\n        Sets the addons for a specific cluster.\n\n        Returns:\n            Callable[[~.SetAddonsConfigRequest],\n                    Awaitable[~.Operation]]:\n                A function that, when called, will call the underlying RPC\n                on the server.\n        '
        if 'set_addons_config' not in self._stubs:
            self._stubs['set_addons_config'] = self.grpc_channel.unary_unary('/google.container.v1beta1.ClusterManager/SetAddonsConfig', request_serializer=cluster_service.SetAddonsConfigRequest.serialize, response_deserializer=cluster_service.Operation.deserialize)
        return self._stubs['set_addons_config']

    @property
    def set_locations(self) -> Callable[[cluster_service.SetLocationsRequest], Awaitable[cluster_service.Operation]]:
        if False:
            i = 10
            return i + 15
        'Return a callable for the set locations method over gRPC.\n\n        Sets the locations for a specific cluster. Deprecated. Use\n        `projects.locations.clusters.update <https://cloud.google.com/kubernetes-engine/docs/reference/rest/v1beta1/projects.locations.clusters/update>`__\n        instead.\n\n        Returns:\n            Callable[[~.SetLocationsRequest],\n                    Awaitable[~.Operation]]:\n                A function that, when called, will call the underlying RPC\n                on the server.\n        '
        if 'set_locations' not in self._stubs:
            self._stubs['set_locations'] = self.grpc_channel.unary_unary('/google.container.v1beta1.ClusterManager/SetLocations', request_serializer=cluster_service.SetLocationsRequest.serialize, response_deserializer=cluster_service.Operation.deserialize)
        return self._stubs['set_locations']

    @property
    def update_master(self) -> Callable[[cluster_service.UpdateMasterRequest], Awaitable[cluster_service.Operation]]:
        if False:
            return 10
        'Return a callable for the update master method over gRPC.\n\n        Updates the master for a specific cluster.\n\n        Returns:\n            Callable[[~.UpdateMasterRequest],\n                    Awaitable[~.Operation]]:\n                A function that, when called, will call the underlying RPC\n                on the server.\n        '
        if 'update_master' not in self._stubs:
            self._stubs['update_master'] = self.grpc_channel.unary_unary('/google.container.v1beta1.ClusterManager/UpdateMaster', request_serializer=cluster_service.UpdateMasterRequest.serialize, response_deserializer=cluster_service.Operation.deserialize)
        return self._stubs['update_master']

    @property
    def set_master_auth(self) -> Callable[[cluster_service.SetMasterAuthRequest], Awaitable[cluster_service.Operation]]:
        if False:
            print('Hello World!')
        'Return a callable for the set master auth method over gRPC.\n\n        Sets master auth materials. Currently supports\n        changing the admin password or a specific cluster,\n        either via password generation or explicitly setting the\n        password.\n\n        Returns:\n            Callable[[~.SetMasterAuthRequest],\n                    Awaitable[~.Operation]]:\n                A function that, when called, will call the underlying RPC\n                on the server.\n        '
        if 'set_master_auth' not in self._stubs:
            self._stubs['set_master_auth'] = self.grpc_channel.unary_unary('/google.container.v1beta1.ClusterManager/SetMasterAuth', request_serializer=cluster_service.SetMasterAuthRequest.serialize, response_deserializer=cluster_service.Operation.deserialize)
        return self._stubs['set_master_auth']

    @property
    def delete_cluster(self) -> Callable[[cluster_service.DeleteClusterRequest], Awaitable[cluster_service.Operation]]:
        if False:
            print('Hello World!')
        "Return a callable for the delete cluster method over gRPC.\n\n        Deletes the cluster, including the Kubernetes\n        endpoint and all worker nodes.\n\n        Firewalls and routes that were configured during cluster\n        creation are also deleted.\n\n        Other Google Compute Engine resources that might be in\n        use by the cluster, such as load balancer resources, are\n        not deleted if they weren't present when the cluster was\n        initially created.\n\n        Returns:\n            Callable[[~.DeleteClusterRequest],\n                    Awaitable[~.Operation]]:\n                A function that, when called, will call the underlying RPC\n                on the server.\n        "
        if 'delete_cluster' not in self._stubs:
            self._stubs['delete_cluster'] = self.grpc_channel.unary_unary('/google.container.v1beta1.ClusterManager/DeleteCluster', request_serializer=cluster_service.DeleteClusterRequest.serialize, response_deserializer=cluster_service.Operation.deserialize)
        return self._stubs['delete_cluster']

    @property
    def list_operations(self) -> Callable[[cluster_service.ListOperationsRequest], Awaitable[cluster_service.ListOperationsResponse]]:
        if False:
            return 10
        'Return a callable for the list operations method over gRPC.\n\n        Lists all operations in a project in the specified\n        zone or all zones.\n\n        Returns:\n            Callable[[~.ListOperationsRequest],\n                    Awaitable[~.ListOperationsResponse]]:\n                A function that, when called, will call the underlying RPC\n                on the server.\n        '
        if 'list_operations' not in self._stubs:
            self._stubs['list_operations'] = self.grpc_channel.unary_unary('/google.container.v1beta1.ClusterManager/ListOperations', request_serializer=cluster_service.ListOperationsRequest.serialize, response_deserializer=cluster_service.ListOperationsResponse.deserialize)
        return self._stubs['list_operations']

    @property
    def get_operation(self) -> Callable[[cluster_service.GetOperationRequest], Awaitable[cluster_service.Operation]]:
        if False:
            i = 10
            return i + 15
        'Return a callable for the get operation method over gRPC.\n\n        Gets the specified operation.\n\n        Returns:\n            Callable[[~.GetOperationRequest],\n                    Awaitable[~.Operation]]:\n                A function that, when called, will call the underlying RPC\n                on the server.\n        '
        if 'get_operation' not in self._stubs:
            self._stubs['get_operation'] = self.grpc_channel.unary_unary('/google.container.v1beta1.ClusterManager/GetOperation', request_serializer=cluster_service.GetOperationRequest.serialize, response_deserializer=cluster_service.Operation.deserialize)
        return self._stubs['get_operation']

    @property
    def cancel_operation(self) -> Callable[[cluster_service.CancelOperationRequest], Awaitable[empty_pb2.Empty]]:
        if False:
            print('Hello World!')
        'Return a callable for the cancel operation method over gRPC.\n\n        Cancels the specified operation.\n\n        Returns:\n            Callable[[~.CancelOperationRequest],\n                    Awaitable[~.Empty]]:\n                A function that, when called, will call the underlying RPC\n                on the server.\n        '
        if 'cancel_operation' not in self._stubs:
            self._stubs['cancel_operation'] = self.grpc_channel.unary_unary('/google.container.v1beta1.ClusterManager/CancelOperation', request_serializer=cluster_service.CancelOperationRequest.serialize, response_deserializer=empty_pb2.Empty.FromString)
        return self._stubs['cancel_operation']

    @property
    def get_server_config(self) -> Callable[[cluster_service.GetServerConfigRequest], Awaitable[cluster_service.ServerConfig]]:
        if False:
            print('Hello World!')
        'Return a callable for the get server config method over gRPC.\n\n        Returns configuration info about the Google\n        Kubernetes Engine service.\n\n        Returns:\n            Callable[[~.GetServerConfigRequest],\n                    Awaitable[~.ServerConfig]]:\n                A function that, when called, will call the underlying RPC\n                on the server.\n        '
        if 'get_server_config' not in self._stubs:
            self._stubs['get_server_config'] = self.grpc_channel.unary_unary('/google.container.v1beta1.ClusterManager/GetServerConfig', request_serializer=cluster_service.GetServerConfigRequest.serialize, response_deserializer=cluster_service.ServerConfig.deserialize)
        return self._stubs['get_server_config']

    @property
    def get_json_web_keys(self) -> Callable[[cluster_service.GetJSONWebKeysRequest], Awaitable[cluster_service.GetJSONWebKeysResponse]]:
        if False:
            i = 10
            return i + 15
        'Return a callable for the get json web keys method over gRPC.\n\n        Gets the public component of the cluster signing keys\n        in JSON Web Key format.\n        This API is not yet intended for general use, and is not\n        available for all clusters.\n\n        Returns:\n            Callable[[~.GetJSONWebKeysRequest],\n                    Awaitable[~.GetJSONWebKeysResponse]]:\n                A function that, when called, will call the underlying RPC\n                on the server.\n        '
        if 'get_json_web_keys' not in self._stubs:
            self._stubs['get_json_web_keys'] = self.grpc_channel.unary_unary('/google.container.v1beta1.ClusterManager/GetJSONWebKeys', request_serializer=cluster_service.GetJSONWebKeysRequest.serialize, response_deserializer=cluster_service.GetJSONWebKeysResponse.deserialize)
        return self._stubs['get_json_web_keys']

    @property
    def list_node_pools(self) -> Callable[[cluster_service.ListNodePoolsRequest], Awaitable[cluster_service.ListNodePoolsResponse]]:
        if False:
            print('Hello World!')
        'Return a callable for the list node pools method over gRPC.\n\n        Lists the node pools for a cluster.\n\n        Returns:\n            Callable[[~.ListNodePoolsRequest],\n                    Awaitable[~.ListNodePoolsResponse]]:\n                A function that, when called, will call the underlying RPC\n                on the server.\n        '
        if 'list_node_pools' not in self._stubs:
            self._stubs['list_node_pools'] = self.grpc_channel.unary_unary('/google.container.v1beta1.ClusterManager/ListNodePools', request_serializer=cluster_service.ListNodePoolsRequest.serialize, response_deserializer=cluster_service.ListNodePoolsResponse.deserialize)
        return self._stubs['list_node_pools']

    @property
    def get_node_pool(self) -> Callable[[cluster_service.GetNodePoolRequest], Awaitable[cluster_service.NodePool]]:
        if False:
            return 10
        'Return a callable for the get node pool method over gRPC.\n\n        Retrieves the requested node pool.\n\n        Returns:\n            Callable[[~.GetNodePoolRequest],\n                    Awaitable[~.NodePool]]:\n                A function that, when called, will call the underlying RPC\n                on the server.\n        '
        if 'get_node_pool' not in self._stubs:
            self._stubs['get_node_pool'] = self.grpc_channel.unary_unary('/google.container.v1beta1.ClusterManager/GetNodePool', request_serializer=cluster_service.GetNodePoolRequest.serialize, response_deserializer=cluster_service.NodePool.deserialize)
        return self._stubs['get_node_pool']

    @property
    def create_node_pool(self) -> Callable[[cluster_service.CreateNodePoolRequest], Awaitable[cluster_service.Operation]]:
        if False:
            for i in range(10):
                print('nop')
        'Return a callable for the create node pool method over gRPC.\n\n        Creates a node pool for a cluster.\n\n        Returns:\n            Callable[[~.CreateNodePoolRequest],\n                    Awaitable[~.Operation]]:\n                A function that, when called, will call the underlying RPC\n                on the server.\n        '
        if 'create_node_pool' not in self._stubs:
            self._stubs['create_node_pool'] = self.grpc_channel.unary_unary('/google.container.v1beta1.ClusterManager/CreateNodePool', request_serializer=cluster_service.CreateNodePoolRequest.serialize, response_deserializer=cluster_service.Operation.deserialize)
        return self._stubs['create_node_pool']

    @property
    def delete_node_pool(self) -> Callable[[cluster_service.DeleteNodePoolRequest], Awaitable[cluster_service.Operation]]:
        if False:
            i = 10
            return i + 15
        'Return a callable for the delete node pool method over gRPC.\n\n        Deletes a node pool from a cluster.\n\n        Returns:\n            Callable[[~.DeleteNodePoolRequest],\n                    Awaitable[~.Operation]]:\n                A function that, when called, will call the underlying RPC\n                on the server.\n        '
        if 'delete_node_pool' not in self._stubs:
            self._stubs['delete_node_pool'] = self.grpc_channel.unary_unary('/google.container.v1beta1.ClusterManager/DeleteNodePool', request_serializer=cluster_service.DeleteNodePoolRequest.serialize, response_deserializer=cluster_service.Operation.deserialize)
        return self._stubs['delete_node_pool']

    @property
    def complete_node_pool_upgrade(self) -> Callable[[cluster_service.CompleteNodePoolUpgradeRequest], Awaitable[empty_pb2.Empty]]:
        if False:
            while True:
                i = 10
        'Return a callable for the complete node pool upgrade method over gRPC.\n\n        CompleteNodePoolUpgrade will signal an on-going node\n        pool upgrade to complete.\n\n        Returns:\n            Callable[[~.CompleteNodePoolUpgradeRequest],\n                    Awaitable[~.Empty]]:\n                A function that, when called, will call the underlying RPC\n                on the server.\n        '
        if 'complete_node_pool_upgrade' not in self._stubs:
            self._stubs['complete_node_pool_upgrade'] = self.grpc_channel.unary_unary('/google.container.v1beta1.ClusterManager/CompleteNodePoolUpgrade', request_serializer=cluster_service.CompleteNodePoolUpgradeRequest.serialize, response_deserializer=empty_pb2.Empty.FromString)
        return self._stubs['complete_node_pool_upgrade']

    @property
    def rollback_node_pool_upgrade(self) -> Callable[[cluster_service.RollbackNodePoolUpgradeRequest], Awaitable[cluster_service.Operation]]:
        if False:
            while True:
                i = 10
        'Return a callable for the rollback node pool upgrade method over gRPC.\n\n        Rolls back a previously Aborted or Failed NodePool\n        upgrade. This makes no changes if the last upgrade\n        successfully completed.\n\n        Returns:\n            Callable[[~.RollbackNodePoolUpgradeRequest],\n                    Awaitable[~.Operation]]:\n                A function that, when called, will call the underlying RPC\n                on the server.\n        '
        if 'rollback_node_pool_upgrade' not in self._stubs:
            self._stubs['rollback_node_pool_upgrade'] = self.grpc_channel.unary_unary('/google.container.v1beta1.ClusterManager/RollbackNodePoolUpgrade', request_serializer=cluster_service.RollbackNodePoolUpgradeRequest.serialize, response_deserializer=cluster_service.Operation.deserialize)
        return self._stubs['rollback_node_pool_upgrade']

    @property
    def set_node_pool_management(self) -> Callable[[cluster_service.SetNodePoolManagementRequest], Awaitable[cluster_service.Operation]]:
        if False:
            return 10
        'Return a callable for the set node pool management method over gRPC.\n\n        Sets the NodeManagement options for a node pool.\n\n        Returns:\n            Callable[[~.SetNodePoolManagementRequest],\n                    Awaitable[~.Operation]]:\n                A function that, when called, will call the underlying RPC\n                on the server.\n        '
        if 'set_node_pool_management' not in self._stubs:
            self._stubs['set_node_pool_management'] = self.grpc_channel.unary_unary('/google.container.v1beta1.ClusterManager/SetNodePoolManagement', request_serializer=cluster_service.SetNodePoolManagementRequest.serialize, response_deserializer=cluster_service.Operation.deserialize)
        return self._stubs['set_node_pool_management']

    @property
    def set_labels(self) -> Callable[[cluster_service.SetLabelsRequest], Awaitable[cluster_service.Operation]]:
        if False:
            i = 10
            return i + 15
        'Return a callable for the set labels method over gRPC.\n\n        Sets labels on a cluster.\n\n        Returns:\n            Callable[[~.SetLabelsRequest],\n                    Awaitable[~.Operation]]:\n                A function that, when called, will call the underlying RPC\n                on the server.\n        '
        if 'set_labels' not in self._stubs:
            self._stubs['set_labels'] = self.grpc_channel.unary_unary('/google.container.v1beta1.ClusterManager/SetLabels', request_serializer=cluster_service.SetLabelsRequest.serialize, response_deserializer=cluster_service.Operation.deserialize)
        return self._stubs['set_labels']

    @property
    def set_legacy_abac(self) -> Callable[[cluster_service.SetLegacyAbacRequest], Awaitable[cluster_service.Operation]]:
        if False:
            print('Hello World!')
        'Return a callable for the set legacy abac method over gRPC.\n\n        Enables or disables the ABAC authorization mechanism\n        on a cluster.\n\n        Returns:\n            Callable[[~.SetLegacyAbacRequest],\n                    Awaitable[~.Operation]]:\n                A function that, when called, will call the underlying RPC\n                on the server.\n        '
        if 'set_legacy_abac' not in self._stubs:
            self._stubs['set_legacy_abac'] = self.grpc_channel.unary_unary('/google.container.v1beta1.ClusterManager/SetLegacyAbac', request_serializer=cluster_service.SetLegacyAbacRequest.serialize, response_deserializer=cluster_service.Operation.deserialize)
        return self._stubs['set_legacy_abac']

    @property
    def start_ip_rotation(self) -> Callable[[cluster_service.StartIPRotationRequest], Awaitable[cluster_service.Operation]]:
        if False:
            while True:
                i = 10
        'Return a callable for the start ip rotation method over gRPC.\n\n        Starts master IP rotation.\n\n        Returns:\n            Callable[[~.StartIPRotationRequest],\n                    Awaitable[~.Operation]]:\n                A function that, when called, will call the underlying RPC\n                on the server.\n        '
        if 'start_ip_rotation' not in self._stubs:
            self._stubs['start_ip_rotation'] = self.grpc_channel.unary_unary('/google.container.v1beta1.ClusterManager/StartIPRotation', request_serializer=cluster_service.StartIPRotationRequest.serialize, response_deserializer=cluster_service.Operation.deserialize)
        return self._stubs['start_ip_rotation']

    @property
    def complete_ip_rotation(self) -> Callable[[cluster_service.CompleteIPRotationRequest], Awaitable[cluster_service.Operation]]:
        if False:
            i = 10
            return i + 15
        'Return a callable for the complete ip rotation method over gRPC.\n\n        Completes master IP rotation.\n\n        Returns:\n            Callable[[~.CompleteIPRotationRequest],\n                    Awaitable[~.Operation]]:\n                A function that, when called, will call the underlying RPC\n                on the server.\n        '
        if 'complete_ip_rotation' not in self._stubs:
            self._stubs['complete_ip_rotation'] = self.grpc_channel.unary_unary('/google.container.v1beta1.ClusterManager/CompleteIPRotation', request_serializer=cluster_service.CompleteIPRotationRequest.serialize, response_deserializer=cluster_service.Operation.deserialize)
        return self._stubs['complete_ip_rotation']

    @property
    def set_node_pool_size(self) -> Callable[[cluster_service.SetNodePoolSizeRequest], Awaitable[cluster_service.Operation]]:
        if False:
            while True:
                i = 10
        'Return a callable for the set node pool size method over gRPC.\n\n        SetNodePoolSizeRequest sets the size of a node pool. The new\n        size will be used for all replicas, including future replicas\n        created by modifying\n        [NodePool.locations][google.container.v1beta1.NodePool.locations].\n\n        Returns:\n            Callable[[~.SetNodePoolSizeRequest],\n                    Awaitable[~.Operation]]:\n                A function that, when called, will call the underlying RPC\n                on the server.\n        '
        if 'set_node_pool_size' not in self._stubs:
            self._stubs['set_node_pool_size'] = self.grpc_channel.unary_unary('/google.container.v1beta1.ClusterManager/SetNodePoolSize', request_serializer=cluster_service.SetNodePoolSizeRequest.serialize, response_deserializer=cluster_service.Operation.deserialize)
        return self._stubs['set_node_pool_size']

    @property
    def set_network_policy(self) -> Callable[[cluster_service.SetNetworkPolicyRequest], Awaitable[cluster_service.Operation]]:
        if False:
            print('Hello World!')
        'Return a callable for the set network policy method over gRPC.\n\n        Enables or disables Network Policy for a cluster.\n\n        Returns:\n            Callable[[~.SetNetworkPolicyRequest],\n                    Awaitable[~.Operation]]:\n                A function that, when called, will call the underlying RPC\n                on the server.\n        '
        if 'set_network_policy' not in self._stubs:
            self._stubs['set_network_policy'] = self.grpc_channel.unary_unary('/google.container.v1beta1.ClusterManager/SetNetworkPolicy', request_serializer=cluster_service.SetNetworkPolicyRequest.serialize, response_deserializer=cluster_service.Operation.deserialize)
        return self._stubs['set_network_policy']

    @property
    def set_maintenance_policy(self) -> Callable[[cluster_service.SetMaintenancePolicyRequest], Awaitable[cluster_service.Operation]]:
        if False:
            print('Hello World!')
        'Return a callable for the set maintenance policy method over gRPC.\n\n        Sets the maintenance policy for a cluster.\n\n        Returns:\n            Callable[[~.SetMaintenancePolicyRequest],\n                    Awaitable[~.Operation]]:\n                A function that, when called, will call the underlying RPC\n                on the server.\n        '
        if 'set_maintenance_policy' not in self._stubs:
            self._stubs['set_maintenance_policy'] = self.grpc_channel.unary_unary('/google.container.v1beta1.ClusterManager/SetMaintenancePolicy', request_serializer=cluster_service.SetMaintenancePolicyRequest.serialize, response_deserializer=cluster_service.Operation.deserialize)
        return self._stubs['set_maintenance_policy']

    @property
    def list_usable_subnetworks(self) -> Callable[[cluster_service.ListUsableSubnetworksRequest], Awaitable[cluster_service.ListUsableSubnetworksResponse]]:
        if False:
            return 10
        'Return a callable for the list usable subnetworks method over gRPC.\n\n        Lists subnetworks that can be used for creating\n        clusters in a project.\n\n        Returns:\n            Callable[[~.ListUsableSubnetworksRequest],\n                    Awaitable[~.ListUsableSubnetworksResponse]]:\n                A function that, when called, will call the underlying RPC\n                on the server.\n        '
        if 'list_usable_subnetworks' not in self._stubs:
            self._stubs['list_usable_subnetworks'] = self.grpc_channel.unary_unary('/google.container.v1beta1.ClusterManager/ListUsableSubnetworks', request_serializer=cluster_service.ListUsableSubnetworksRequest.serialize, response_deserializer=cluster_service.ListUsableSubnetworksResponse.deserialize)
        return self._stubs['list_usable_subnetworks']

    @property
    def check_autopilot_compatibility(self) -> Callable[[cluster_service.CheckAutopilotCompatibilityRequest], Awaitable[cluster_service.CheckAutopilotCompatibilityResponse]]:
        if False:
            while True:
                i = 10
        'Return a callable for the check autopilot compatibility method over gRPC.\n\n        Checks the cluster compatibility with Autopilot mode,\n        and returns a list of compatibility issues.\n\n        Returns:\n            Callable[[~.CheckAutopilotCompatibilityRequest],\n                    Awaitable[~.CheckAutopilotCompatibilityResponse]]:\n                A function that, when called, will call the underlying RPC\n                on the server.\n        '
        if 'check_autopilot_compatibility' not in self._stubs:
            self._stubs['check_autopilot_compatibility'] = self.grpc_channel.unary_unary('/google.container.v1beta1.ClusterManager/CheckAutopilotCompatibility', request_serializer=cluster_service.CheckAutopilotCompatibilityRequest.serialize, response_deserializer=cluster_service.CheckAutopilotCompatibilityResponse.deserialize)
        return self._stubs['check_autopilot_compatibility']

    @property
    def list_locations(self) -> Callable[[cluster_service.ListLocationsRequest], Awaitable[cluster_service.ListLocationsResponse]]:
        if False:
            while True:
                i = 10
        'Return a callable for the list locations method over gRPC.\n\n        Fetches locations that offer Google Kubernetes\n        Engine.\n\n        Returns:\n            Callable[[~.ListLocationsRequest],\n                    Awaitable[~.ListLocationsResponse]]:\n                A function that, when called, will call the underlying RPC\n                on the server.\n        '
        if 'list_locations' not in self._stubs:
            self._stubs['list_locations'] = self.grpc_channel.unary_unary('/google.container.v1beta1.ClusterManager/ListLocations', request_serializer=cluster_service.ListLocationsRequest.serialize, response_deserializer=cluster_service.ListLocationsResponse.deserialize)
        return self._stubs['list_locations']

    def close(self):
        if False:
            return 10
        return self.grpc_channel.close()
__all__ = ('ClusterManagerGrpcAsyncIOTransport',)