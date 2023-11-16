from . import _range as prange
from typing import Any, Dict, List, Optional, Union, Iterable, cast, overload
from .partition_key import _Undefined, _Empty, PartitionKey
MaximumExclusiveEffectivePartitionKey = 255
MinimumInclusiveEffectivePartitionKey = 0

def GetEPKRangeForPrefixPartitionKey(partitionKeyDefinition: PartitionKey) -> prange.Range:
    if False:
        print('Hello World!')
    minEPK = GetEffectivePartitionKeyString(partitionKeyDefinition, False)
    maxEPK = minEPK + MaximumExclusiveEffectivePartitionKey
    return prange.Range(minEPK, maxEPK)

def GetEffectivePartitionKeyForHashPartitioning():
    if False:
        return 10
    pass

def GetEffectivePartitionKeyForHashPartitioningV2():
    if False:
        for i in range(10):
            print('nop')
    pass

def GetEffectivePartitionKeyForMultiHashPartitioningV2():
    if False:
        while True:
            i = 10
    pass

def ToHexEncodedBinaryString(path: Union[str, list]) -> str:
    if False:
        i = 10
        return i + 15
    pass

def GetEffectivePartitionKeyString(partitionKeyDefinition: PartitionKey, strict: bool) -> str:
    if False:
        return 10
    if type(partitionKeyDefinition) == _Empty:
        return MinimumInclusiveEffectivePartitionKey
    if partitionKeyDefinition.kind == 'Hash':
        if partitionKeyDefinition.version == 1:
            GetEffectivePartitionKeyForHashPartitioning()
        elif partitionKeyDefinition.version == 2:
            GetEffectivePartitionKeyForHashPartitioningV2()
    elif partitionKeyDefinition.kind == 'MultiHash':
        GetEffectivePartitionKeyForMultiHashPartitioningV2()
    else:
        return ToHexEncodedBinaryString(partitionKeyDefinition.paths)