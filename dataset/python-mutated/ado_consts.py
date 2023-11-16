adXactUnspecified = -1
adXactBrowse = 256
adXactChaos = 16
adXactCursorStability = 4096
adXactIsolated = 1048576
adXactReadCommitted = 4096
adXactReadUncommitted = 256
adXactRepeatableRead = 65536
adXactSerializable = 1048576
adUseClient = 3
adUseServer = 2
adOpenDynamic = 2
adOpenForwardOnly = 0
adOpenKeyset = 1
adOpenStatic = 3
adOpenUnspecified = -1
adCmdText = 1
adCmdStoredProc = 4
adSchemaTables = 20
adParamInput = 1
adParamInputOutput = 3
adParamOutput = 2
adParamReturnValue = 4
adParamUnknown = 0
directions = {0: 'Unknown', 1: 'Input', 2: 'Output', 3: 'InputOutput', 4: 'Return'}

def ado_direction_name(ado_dir):
    if False:
        print('Hello World!')
    try:
        return 'adParam' + directions[ado_dir]
    except:
        return 'unknown direction (' + str(ado_dir) + ')'
adStateClosed = 0
adStateOpen = 1
adStateConnecting = 2
adStateExecuting = 4
adStateFetching = 8
adFldMayBeNull = 64
adModeUnknown = 0
adModeRead = 1
adModeWrite = 2
adModeReadWrite = 3
adModeShareDenyRead = 4
adModeShareDenyWrite = 8
adModeShareExclusive = 12
adModeShareDenyNone = 16
adModeRecursive = 4194304
adXactCommitRetaining = 131072
adXactAbortRetaining = 262144
ado_error_TIMEOUT = -2147217871
adArray = 8192
adEmpty = 0
adBSTR = 8
adBigInt = 20
adBinary = 128
adBoolean = 11
adChapter = 136
adChar = 129
adCurrency = 6
adDBDate = 133
adDBTime = 134
adDBTimeStamp = 135
adDate = 7
adDecimal = 14
adDouble = 5
adError = 10
adFileTime = 64
adGUID = 72
adIDispatch = 9
adIUnknown = 13
adInteger = 3
adLongVarBinary = 205
adLongVarChar = 201
adLongVarWChar = 203
adNumeric = 131
adPropVariant = 138
adSingle = 4
adSmallInt = 2
adTinyInt = 16
adUnsignedBigInt = 21
adUnsignedInt = 19
adUnsignedSmallInt = 18
adUnsignedTinyInt = 17
adUserDefined = 132
adVarBinary = 204
adVarChar = 200
adVarNumeric = 139
adVarWChar = 202
adVariant = 12
adWChar = 130
AUTO_FIELD_MARKER = -1000
adTypeNames = {adBSTR: 'adBSTR', adBigInt: 'adBigInt', adBinary: 'adBinary', adBoolean: 'adBoolean', adChapter: 'adChapter', adChar: 'adChar', adCurrency: 'adCurrency', adDBDate: 'adDBDate', adDBTime: 'adDBTime', adDBTimeStamp: 'adDBTimeStamp', adDate: 'adDate', adDecimal: 'adDecimal', adDouble: 'adDouble', adEmpty: 'adEmpty', adError: 'adError', adFileTime: 'adFileTime', adGUID: 'adGUID', adIDispatch: 'adIDispatch', adIUnknown: 'adIUnknown', adInteger: 'adInteger', adLongVarBinary: 'adLongVarBinary', adLongVarChar: 'adLongVarChar', adLongVarWChar: 'adLongVarWChar', adNumeric: 'adNumeric', adPropVariant: 'adPropVariant', adSingle: 'adSingle', adSmallInt: 'adSmallInt', adTinyInt: 'adTinyInt', adUnsignedBigInt: 'adUnsignedBigInt', adUnsignedInt: 'adUnsignedInt', adUnsignedSmallInt: 'adUnsignedSmallInt', adUnsignedTinyInt: 'adUnsignedTinyInt', adUserDefined: 'adUserDefined', adVarBinary: 'adVarBinary', adVarChar: 'adVarChar', adVarNumeric: 'adVarNumeric', adVarWChar: 'adVarWChar', adVariant: 'adVariant', adWChar: 'adWChar'}

def ado_type_name(ado_type):
    if False:
        return 10
    return adTypeNames.get(ado_type, 'unknown type (' + str(ado_type) + ')')
adoErrors = {3707: 'adErrBoundToCommand', 3732: 'adErrCannotComplete', 3748: 'adErrCantChangeConnection', 3220: 'adErrCantChangeProvider', 3724: 'adErrCantConvertvalue', 3725: 'adErrCantCreate', 3747: 'adErrCatalogNotSet', 3726: 'adErrColumnNotOnThisRow', 3421: 'adErrDataConversion', 3721: 'adErrDataOverflow', 3738: 'adErrDelResOutOfScope', 3750: 'adErrDenyNotSupported', 3751: 'adErrDenyTypeNotSupported', 3251: 'adErrFeatureNotAvailable', 3749: 'adErrFieldsUpdateFailed', 3219: 'adErrIllegalOperation', 3246: 'adErrInTransaction', 3719: 'adErrIntegrityViolation', 3001: 'adErrInvalidArgument', 3709: 'adErrInvalidConnection', 3708: 'adErrInvalidParamInfo', 3714: 'adErrInvalidTransaction', 3729: 'adErrInvalidURL', 3265: 'adErrItemNotFound', 3021: 'adErrNoCurrentRecord', 3715: 'adErrNotExecuting', 3710: 'adErrNotReentrant', 3704: 'adErrObjectClosed', 3367: 'adErrObjectInCollection', 3420: 'adErrObjectNotSet', 3705: 'adErrObjectOpen', 3002: 'adErrOpeningFile', 3712: 'adErrOperationCancelled', 3734: 'adErrOutOfSpace', 3720: 'adErrPermissionDenied', 3742: 'adErrPropConflicting', 3739: 'adErrPropInvalidColumn', 3740: 'adErrPropInvalidOption', 3741: 'adErrPropInvalidValue', 3743: 'adErrPropNotAllSettable', 3744: 'adErrPropNotSet', 3745: 'adErrPropNotSettable', 3746: 'adErrPropNotSupported', 3000: 'adErrProviderFailed', 3706: 'adErrProviderNotFound', 3003: 'adErrReadFile', 3731: 'adErrResourceExists', 3730: 'adErrResourceLocked', 3735: 'adErrResourceOutOfScope', 3722: 'adErrSchemaViolation', 3723: 'adErrSignMismatch', 3713: 'adErrStillConnecting', 3711: 'adErrStillExecuting', 3728: 'adErrTreePermissionDenied', 3727: 'adErrURLDoesNotExist', 3737: 'adErrURLNamedRowDoesNotExist', 3736: 'adErrUnavailable', 3716: 'adErrUnsafeOperation', 3733: 'adErrVolumeNotFound', 3004: 'adErrWriteFile'}