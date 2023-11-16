from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto
__protobuf__ = proto.module(package='google.cloud.baremetalsolution.v2', manifest={'SSHKey', 'ListSSHKeysRequest', 'ListSSHKeysResponse', 'CreateSSHKeyRequest', 'DeleteSSHKeyRequest'})

class SSHKey(proto.Message):
    """An SSH key, used for authorizing with the interactive serial
    console feature.

    Attributes:
        name (str):
            Output only. The name of this SSH key.
            Currently, the only valid value for the location
            is "global".
        public_key (str):
            The public SSH key. This must be in OpenSSH .authorized_keys
            format.
    """
    name: str = proto.Field(proto.STRING, number=1)
    public_key: str = proto.Field(proto.STRING, number=2)

class ListSSHKeysRequest(proto.Message):
    """Message for listing the public SSH keys in a project.

    Attributes:
        parent (str):
            Required. The parent containing the SSH keys.
            Currently, the only valid value for the location
            is "global".
        page_size (int):
            The maximum number of items to return.
        page_token (str):
            The next_page_token value returned from a previous List
            request, if any.
    """
    parent: str = proto.Field(proto.STRING, number=1)
    page_size: int = proto.Field(proto.INT32, number=2)
    page_token: str = proto.Field(proto.STRING, number=3)

class ListSSHKeysResponse(proto.Message):
    """Message for response of ListSSHKeys.

    Attributes:
        ssh_keys (MutableSequence[google.cloud.bare_metal_solution_v2.types.SSHKey]):
            The SSH keys registered in the project.
        next_page_token (str):
            Token to retrieve the next page of results,
            or empty if there are no more results in the
            list.
    """

    @property
    def raw_page(self):
        if False:
            for i in range(10):
                print('nop')
        return self
    ssh_keys: MutableSequence['SSHKey'] = proto.RepeatedField(proto.MESSAGE, number=1, message='SSHKey')
    next_page_token: str = proto.Field(proto.STRING, number=90)

class CreateSSHKeyRequest(proto.Message):
    """Message for registering a public SSH key in a project.

    Attributes:
        parent (str):
            Required. The parent containing the SSH keys.
        ssh_key (google.cloud.bare_metal_solution_v2.types.SSHKey):
            Required. The SSH key to register.
        ssh_key_id (str):
            Required. The ID to use for the key, which will become the
            final component of the key's resource name.

            This value must match the regex: [a-zA-Z0-9@.-_]{1,64}
    """
    parent: str = proto.Field(proto.STRING, number=1)
    ssh_key: 'SSHKey' = proto.Field(proto.MESSAGE, number=2, message='SSHKey')
    ssh_key_id: str = proto.Field(proto.STRING, number=3)

class DeleteSSHKeyRequest(proto.Message):
    """Message for deleting an SSH key from a project.

    Attributes:
        name (str):
            Required. The name of the SSH key to delete.
            Currently, the only valid value for the location
            is "global".
    """
    name: str = proto.Field(proto.STRING, number=1)
__all__ = tuple(sorted(__protobuf__.manifest))