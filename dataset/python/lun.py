# -*- coding: utf-8 -*-
# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from __future__ import annotations

from typing import MutableMapping, MutableSequence

from google.protobuf import timestamp_pb2  # type: ignore
import proto  # type: ignore

__protobuf__ = proto.module(
    package="google.cloud.baremetalsolution.v2",
    manifest={
        "Lun",
        "GetLunRequest",
        "ListLunsRequest",
        "ListLunsResponse",
        "EvictLunRequest",
    },
)


class Lun(proto.Message):
    r"""A storage volume logical unit number (LUN).

    Attributes:
        name (str):
            Output only. The name of the LUN.
        id (str):
            An identifier for the LUN, generated by the
            backend.
        state (google.cloud.bare_metal_solution_v2.types.Lun.State):
            The state of this storage volume.
        size_gb (int):
            The size of this LUN, in gigabytes.
        multiprotocol_type (google.cloud.bare_metal_solution_v2.types.Lun.MultiprotocolType):
            The LUN multiprotocol type ensures the
            characteristics of the LUN are optimized for
            each operating system.
        storage_volume (str):
            Display the storage volume for this LUN.
        shareable (bool):
            Display if this LUN can be shared between
            multiple physical servers.
        boot_lun (bool):
            Display if this LUN is a boot LUN.
        storage_type (google.cloud.bare_metal_solution_v2.types.Lun.StorageType):
            The storage type for this LUN.
        wwid (str):
            The WWID for this LUN.
        expire_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Time after which LUN will be fully deleted. It
            is filled only for LUNs in COOL_OFF state.
        instances (MutableSequence[str]):
            Output only. Instances this Lun is attached
            to.
    """

    class State(proto.Enum):
        r"""The possible states for the LUN.

        Values:
            STATE_UNSPECIFIED (0):
                The LUN is in an unknown state.
            CREATING (1):
                The LUN is being created.
            UPDATING (2):
                The LUN is being updated.
            READY (3):
                The LUN is ready for use.
            DELETING (4):
                The LUN has been requested to be deleted.
            COOL_OFF (5):
                The LUN is in cool off state. It will be deleted after
                ``expire_time``.
        """
        STATE_UNSPECIFIED = 0
        CREATING = 1
        UPDATING = 2
        READY = 3
        DELETING = 4
        COOL_OFF = 5

    class MultiprotocolType(proto.Enum):
        r"""Display the operating systems present for the LUN
        multiprotocol type.

        Values:
            MULTIPROTOCOL_TYPE_UNSPECIFIED (0):
                Server has no OS specified.
            LINUX (1):
                Server with Linux OS.
        """
        MULTIPROTOCOL_TYPE_UNSPECIFIED = 0
        LINUX = 1

    class StorageType(proto.Enum):
        r"""The storage types for a LUN.

        Values:
            STORAGE_TYPE_UNSPECIFIED (0):
                The storage type for this LUN is unknown.
            SSD (1):
                This storage type for this LUN is SSD.
            HDD (2):
                This storage type for this LUN is HDD.
        """
        STORAGE_TYPE_UNSPECIFIED = 0
        SSD = 1
        HDD = 2

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    id: str = proto.Field(
        proto.STRING,
        number=10,
    )
    state: State = proto.Field(
        proto.ENUM,
        number=2,
        enum=State,
    )
    size_gb: int = proto.Field(
        proto.INT64,
        number=3,
    )
    multiprotocol_type: MultiprotocolType = proto.Field(
        proto.ENUM,
        number=4,
        enum=MultiprotocolType,
    )
    storage_volume: str = proto.Field(
        proto.STRING,
        number=5,
    )
    shareable: bool = proto.Field(
        proto.BOOL,
        number=6,
    )
    boot_lun: bool = proto.Field(
        proto.BOOL,
        number=7,
    )
    storage_type: StorageType = proto.Field(
        proto.ENUM,
        number=8,
        enum=StorageType,
    )
    wwid: str = proto.Field(
        proto.STRING,
        number=9,
    )
    expire_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=11,
        message=timestamp_pb2.Timestamp,
    )
    instances: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=12,
    )


class GetLunRequest(proto.Message):
    r"""Message for requesting storage lun information.

    Attributes:
        name (str):
            Required. Name of the resource.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListLunsRequest(proto.Message):
    r"""Message for requesting a list of storage volume luns.

    Attributes:
        parent (str):
            Required. Parent value for ListLunsRequest.
        page_size (int):
            Requested page size. The server might return
            fewer items than requested. If unspecified,
            server will pick an appropriate default.
        page_token (str):
            A token identifying a page of results from
            the server.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=2,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=3,
    )


class ListLunsResponse(proto.Message):
    r"""Response message containing the list of storage volume luns.

    Attributes:
        luns (MutableSequence[google.cloud.bare_metal_solution_v2.types.Lun]):
            The list of luns.
        next_page_token (str):
            A token identifying a page of results from
            the server.
        unreachable (MutableSequence[str]):
            Locations that could not be reached.
    """

    @property
    def raw_page(self):
        return self

    luns: MutableSequence["Lun"] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message="Lun",
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )
    unreachable: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=3,
    )


class EvictLunRequest(proto.Message):
    r"""Request for skip lun cooloff and delete it.

    Attributes:
        name (str):
            Required. The name of the lun.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
