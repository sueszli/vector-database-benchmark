#define _HIDPI_
#define _HIDPI_NO_FUNCTION_MACROS_
#include <ntddk.h>
#include <hidpddi.h>

#include "hidparser.h"
#include "hidp.h"

#define UNIMPLEMENTED DebugFunction("%s is UNIMPLEMENTED\n", __FUNCTION__)

VOID
NTAPI
HidP_FreeCollectionDescription(
    IN PHIDP_DEVICE_DESC   DeviceDescription)
{
    //
    // free collection
    //
    HidParser_FreeCollectionDescription(DeviceDescription);
}


HIDAPI
NTSTATUS
NTAPI
HidP_GetCaps(
    IN PHIDP_PREPARSED_DATA  PreparsedData,
    OUT PHIDP_CAPS  Capabilities)
{
    //
    // get caps
    //
    return HidParser_GetCaps(PreparsedData, Capabilities);
}

NTSTATUS
TranslateStatusForUpperLayer(
    IN NTSTATUS Status)
{
    //
    // now we are handling only this values, for others just return
    // status as it is.
    //
    switch (Status)
    {
    case HIDP_STATUS_INTERNAL_ERROR:
        return STATUS_INSUFFICIENT_RESOURCES;
    case HIDP_STATUS_INVALID_REPORT_TYPE:
        return HIDP_STATUS_INVALID_REPORT_TYPE;
    case HIDP_STATUS_BUFFER_TOO_SMALL:
        return STATUS_BUFFER_TOO_SMALL;
    case HIDP_STATUS_USAGE_NOT_FOUND:
        return STATUS_NO_DATA_DETECTED;
    default:
        return Status;
    }
}

NTSTATUS
NTAPI
HidP_GetCollectionDescription(
    IN PHIDP_REPORT_DESCRIPTOR ReportDesc,
    IN ULONG DescLength,
    IN POOL_TYPE PoolType,
    OUT PHIDP_DEVICE_DESC DeviceDescription)
{
    NTSTATUS Status;

    //
    // get description;
    //
    Status = HidParser_GetCollectionDescription(ReportDesc, DescLength, PoolType, DeviceDescription);
    return TranslateStatusForUpperLayer(Status);
}

HIDAPI
ULONG
NTAPI
HidP_MaxUsageListLength(
    IN HIDP_REPORT_TYPE  ReportType,
    IN USAGE  UsagePage  OPTIONAL,
    IN PHIDP_PREPARSED_DATA  PreparsedData)
{
    //
    // sanity check
    //
    ASSERT(ReportType == HidP_Input || ReportType == HidP_Output || ReportType == HidP_Feature);

    //
    // get usage length
    //
    return HidParser_MaxUsageListLength(PreparsedData, ReportType, UsagePage);
}

HIDAPI
NTSTATUS
NTAPI
HidP_GetSpecificValueCaps(
    IN HIDP_REPORT_TYPE  ReportType,
    IN USAGE  UsagePage,
    IN USHORT  LinkCollection,
    IN USAGE  Usage,
    OUT PHIDP_VALUE_CAPS  ValueCaps,
    IN OUT PUSHORT  ValueCapsLength,
    IN PHIDP_PREPARSED_DATA  PreparsedData)
{
    //
    // sanity check
    //
    ASSERT(ReportType == HidP_Input || ReportType == HidP_Output || ReportType == HidP_Feature);

    //
    // get value caps
    //
    return HidParser_GetSpecificValueCaps(PreparsedData, ReportType, UsagePage, LinkCollection, Usage, ValueCaps, ValueCapsLength);
}

HIDAPI
NTSTATUS
NTAPI
HidP_GetUsages(
    IN HIDP_REPORT_TYPE ReportType,
    IN USAGE UsagePage,
    IN USHORT LinkCollection  OPTIONAL,
    OUT PUSAGE UsageList,
    IN OUT PULONG UsageLength,
    IN PHIDP_PREPARSED_DATA  PreparsedData,
    IN PCHAR Report,
    IN ULONG ReportLength)
{
    //
    // sanity check
    //
    ASSERT(ReportType == HidP_Input || ReportType == HidP_Output || ReportType == HidP_Feature);

    //
    // get usages
    //
    return HidParser_GetUsages(PreparsedData, ReportType, UsagePage, LinkCollection, UsageList, UsageLength, Report, ReportLength);
}


#undef HidP_GetButtonCaps

HIDAPI
NTSTATUS
NTAPI
HidP_UsageListDifference(
    IN PUSAGE  PreviousUsageList,
    IN PUSAGE  CurrentUsageList,
    OUT PUSAGE  BreakUsageList,
    OUT PUSAGE  MakeUsageList,
    IN ULONG  UsageListLength)
{
    return HidParser_UsageListDifference(PreviousUsageList, CurrentUsageList, BreakUsageList, MakeUsageList, UsageListLength);
}

HIDAPI
NTSTATUS
NTAPI
HidP_GetUsagesEx(
    IN HIDP_REPORT_TYPE  ReportType,
    IN USHORT  LinkCollection,
    OUT PUSAGE_AND_PAGE  ButtonList,
    IN OUT ULONG  *UsageLength,
    IN PHIDP_PREPARSED_DATA  PreparsedData,
    IN PCHAR  Report,
    IN ULONG  ReportLength)
{
    return HidP_GetUsages(ReportType, HID_USAGE_PAGE_UNDEFINED, LinkCollection, &ButtonList->Usage, UsageLength, PreparsedData, Report, ReportLength);
}

HIDAPI
NTSTATUS
NTAPI
HidP_UsageAndPageListDifference(
    IN PUSAGE_AND_PAGE  PreviousUsageList,
    IN PUSAGE_AND_PAGE  CurrentUsageList,
    OUT PUSAGE_AND_PAGE  BreakUsageList,
    OUT PUSAGE_AND_PAGE  MakeUsageList,
    IN ULONG  UsageListLength)
{
    return HidParser_UsageAndPageListDifference(PreviousUsageList, CurrentUsageList, BreakUsageList, MakeUsageList, UsageListLength);
}

HIDAPI
NTSTATUS
NTAPI
HidP_GetScaledUsageValue(
    IN HIDP_REPORT_TYPE  ReportType,
    IN USAGE  UsagePage,
    IN USHORT  LinkCollection  OPTIONAL,
    IN USAGE  Usage,
    OUT PLONG  UsageValue,
    IN PHIDP_PREPARSED_DATA  PreparsedData,
    IN PCHAR  Report,
    IN ULONG  ReportLength)
{
    //
    // sanity check
    //
    ASSERT(ReportType == HidP_Input || ReportType == HidP_Output || ReportType == HidP_Feature);

    //
    // get scaled usage value
    //
    return HidParser_GetScaledUsageValue(PreparsedData, ReportType, UsagePage, LinkCollection, Usage, UsageValue, Report, ReportLength);
}

HIDAPI
NTSTATUS
NTAPI
HidP_GetUsageValue(
    IN HIDP_REPORT_TYPE  ReportType,
    IN USAGE  UsagePage,
    IN USHORT  LinkCollection,
    IN USAGE  Usage,
    OUT PULONG  UsageValue,
    IN PHIDP_PREPARSED_DATA  PreparsedData,
    IN PCHAR  Report,
    IN ULONG  ReportLength)
{
    //
    // sanity check
    //
    ASSERT(ReportType == HidP_Input || ReportType == HidP_Output || ReportType == HidP_Feature);

    //
    // get scaled usage value
    //
    return HidParser_GetUsageValue(PreparsedData, ReportType, UsagePage, LinkCollection, Usage, UsageValue, Report, ReportLength);
}


HIDAPI
NTSTATUS
NTAPI
HidP_TranslateUsageAndPagesToI8042ScanCodes(
    IN PUSAGE_AND_PAGE  ChangedUsageList,
    IN ULONG  UsageListLength,
    IN HIDP_KEYBOARD_DIRECTION  KeyAction,
    IN OUT PHIDP_KEYBOARD_MODIFIER_STATE  ModifierState,
    IN PHIDP_INSERT_SCANCODES  InsertCodesProcedure,
    IN PVOID  InsertCodesContext)
{
    //
    // translate usage pages
    //
    return HidParser_TranslateUsageAndPagesToI8042ScanCodes(ChangedUsageList, UsageListLength, KeyAction, ModifierState, InsertCodesProcedure, InsertCodesContext);
}

HIDAPI
NTSTATUS
NTAPI
HidP_GetButtonCaps(
    HIDP_REPORT_TYPE ReportType,
    PHIDP_BUTTON_CAPS ButtonCaps,
    PUSHORT ButtonCapsLength,
    PHIDP_PREPARSED_DATA PreparsedData)
{
    return HidP_GetSpecificButtonCaps(ReportType, HID_USAGE_PAGE_UNDEFINED, 0, 0, ButtonCaps, ButtonCapsLength, PreparsedData);
}

HIDAPI
NTSTATUS
NTAPI
HidP_GetSpecificButtonCaps(
    IN HIDP_REPORT_TYPE ReportType,
    IN USAGE UsagePage,
    IN USHORT LinkCollection,
    IN USAGE Usage,
    OUT PHIDP_BUTTON_CAPS ButtonCaps,
    IN OUT PUSHORT ButtonCapsLength,
    IN PHIDP_PREPARSED_DATA  PreparsedData)
{
    UNIMPLEMENTED;
    ASSERT(FALSE);
    return STATUS_NOT_IMPLEMENTED;
}

HIDAPI
NTSTATUS
NTAPI
HidP_GetData(
    IN HIDP_REPORT_TYPE  ReportType,
    OUT PHIDP_DATA  DataList,
    IN OUT PULONG  DataLength,
    IN PHIDP_PREPARSED_DATA  PreparsedData,
    IN PCHAR  Report,
    IN ULONG  ReportLength)
{
    UNIMPLEMENTED;
    ASSERT(FALSE);
    return STATUS_NOT_IMPLEMENTED;
}

HIDAPI
NTSTATUS
NTAPI
HidP_GetExtendedAttributes(
    IN HIDP_REPORT_TYPE  ReportType,
    IN USHORT DataIndex,
    IN PHIDP_PREPARSED_DATA  PreparsedData,
    OUT PHIDP_EXTENDED_ATTRIBUTES  Attributes,
    IN OUT PULONG  LengthAttributes)
{
    UNIMPLEMENTED;
    ASSERT(FALSE);
    return STATUS_NOT_IMPLEMENTED;
}

HIDAPI
NTSTATUS
NTAPI
HidP_GetLinkCollectionNodes(
    OUT PHIDP_LINK_COLLECTION_NODE  LinkCollectionNodes,
    IN OUT PULONG  LinkCollectionNodesLength,
    IN PHIDP_PREPARSED_DATA  PreparsedData)
{
    UNIMPLEMENTED;
    ASSERT(FALSE);
    return STATUS_NOT_IMPLEMENTED;
}

NTSTATUS
NTAPI
HidP_SysPowerEvent(
    IN PCHAR HidPacket,
    IN USHORT HidPacketLength,
    IN PHIDP_PREPARSED_DATA Ppd,
    OUT PULONG OutputBuffer)
{
    UNIMPLEMENTED;
    ASSERT(FALSE);
    return STATUS_NOT_IMPLEMENTED;
}

NTSTATUS
NTAPI
HidP_SysPowerCaps(
    IN PHIDP_PREPARSED_DATA Ppd,
    OUT PULONG OutputBuffer)
{
    UNIMPLEMENTED;
    ASSERT(FALSE);
    return STATUS_NOT_IMPLEMENTED;
}

HIDAPI
NTSTATUS
NTAPI
HidP_GetUsageValueArray(
    IN HIDP_REPORT_TYPE  ReportType,
    IN USAGE  UsagePage,
    IN USHORT  LinkCollection  OPTIONAL,
    IN USAGE  Usage,
    OUT PCHAR  UsageValue,
    IN USHORT  UsageValueByteLength,
    IN PHIDP_PREPARSED_DATA  PreparsedData,
    IN PCHAR  Report,
    IN ULONG  ReportLength)
{
    UNIMPLEMENTED;
    ASSERT(FALSE);
    return STATUS_NOT_IMPLEMENTED;
}


HIDAPI
NTSTATUS
NTAPI
HidP_UnsetUsages(
    IN HIDP_REPORT_TYPE  ReportType,
    IN USAGE  UsagePage,
    IN USHORT  LinkCollection,
    IN PUSAGE  UsageList,
    IN OUT PULONG  UsageLength,
    IN PHIDP_PREPARSED_DATA  PreparsedData,
    IN OUT PCHAR  Report,
    IN ULONG  ReportLength)
{
    UNIMPLEMENTED;
    ASSERT(FALSE);
    return STATUS_NOT_IMPLEMENTED;
}

HIDAPI
NTSTATUS
NTAPI
HidP_TranslateUsagesToI8042ScanCodes(
    IN PUSAGE  ChangedUsageList,
    IN ULONG  UsageListLength,
    IN HIDP_KEYBOARD_DIRECTION  KeyAction,
    IN OUT PHIDP_KEYBOARD_MODIFIER_STATE  ModifierState,
    IN PHIDP_INSERT_SCANCODES  InsertCodesProcedure,
    IN PVOID  InsertCodesContext)
{
    UNIMPLEMENTED;
    ASSERT(FALSE);
    return STATUS_NOT_IMPLEMENTED;
}

HIDAPI
NTSTATUS
NTAPI
HidP_SetUsages(
    IN HIDP_REPORT_TYPE  ReportType,
    IN USAGE  UsagePage,
    IN USHORT  LinkCollection,
    IN PUSAGE  UsageList,
    IN OUT PULONG  UsageLength,
    IN PHIDP_PREPARSED_DATA  PreparsedData,
    IN OUT PCHAR  Report,
    IN ULONG  ReportLength)
{
    UNIMPLEMENTED;
    ASSERT(FALSE);
    return STATUS_NOT_IMPLEMENTED;
}

HIDAPI
NTSTATUS
NTAPI
HidP_SetUsageValueArray(
    IN HIDP_REPORT_TYPE  ReportType,
    IN USAGE  UsagePage,
    IN USHORT  LinkCollection  OPTIONAL,
    IN USAGE  Usage,
    IN PCHAR  UsageValue,
    IN USHORT  UsageValueByteLength,
    IN PHIDP_PREPARSED_DATA  PreparsedData,
    OUT PCHAR  Report,
    IN ULONG  ReportLength)
{
    UNIMPLEMENTED;
    ASSERT(FALSE);
    return STATUS_NOT_IMPLEMENTED;
}

HIDAPI
NTSTATUS
NTAPI
HidP_SetUsageValue(
    IN HIDP_REPORT_TYPE  ReportType,
    IN USAGE  UsagePage,
    IN USHORT  LinkCollection,
    IN USAGE  Usage,
    IN ULONG  UsageValue,
    IN PHIDP_PREPARSED_DATA  PreparsedData,
    IN OUT PCHAR  Report,
    IN ULONG  ReportLength)
{
    UNIMPLEMENTED;
    ASSERT(FALSE);
    return STATUS_NOT_IMPLEMENTED;
}

HIDAPI
NTSTATUS
NTAPI
HidP_SetScaledUsageValue(
    IN HIDP_REPORT_TYPE  ReportType,
    IN USAGE  UsagePage,
    IN USHORT  LinkCollection  OPTIONAL,
    IN USAGE  Usage,
    IN LONG  UsageValue,
    IN PHIDP_PREPARSED_DATA  PreparsedData,
    IN OUT PCHAR  Report,
    IN ULONG  ReportLength)
{
    UNIMPLEMENTED;
    ASSERT(FALSE);
    return STATUS_NOT_IMPLEMENTED;
}

HIDAPI
NTSTATUS
NTAPI
HidP_SetData(
    IN HIDP_REPORT_TYPE  ReportType,
    IN PHIDP_DATA  DataList,
    IN OUT PULONG  DataLength,
    IN PHIDP_PREPARSED_DATA  PreparsedData,
    IN OUT PCHAR  Report,
    IN ULONG  ReportLength)
{
    UNIMPLEMENTED;
    ASSERT(FALSE);
    return STATUS_NOT_IMPLEMENTED;
}

HIDAPI
ULONG
NTAPI
HidP_MaxDataListLength(
    IN HIDP_REPORT_TYPE  ReportType,
    IN PHIDP_PREPARSED_DATA  PreparsedData)
{
    UNIMPLEMENTED;
    ASSERT(FALSE);
    return STATUS_NOT_IMPLEMENTED;
}

HIDAPI
NTSTATUS
NTAPI
HidP_InitializeReportForID(
    IN HIDP_REPORT_TYPE  ReportType,
    IN UCHAR  ReportID,
    IN PHIDP_PREPARSED_DATA  PreparsedData,
    IN OUT PCHAR  Report,
    IN ULONG  ReportLength)
{
    UNIMPLEMENTED;
    ASSERT(FALSE);
    return STATUS_NOT_IMPLEMENTED;
}

#undef HidP_GetValueCaps

HIDAPI
NTSTATUS
NTAPI
HidP_GetValueCaps(
    HIDP_REPORT_TYPE ReportType,
    PHIDP_VALUE_CAPS ValueCaps,
    PUSHORT ValueCapsLength,
    PHIDP_PREPARSED_DATA PreparsedData)
{
    return HidP_GetSpecificValueCaps(ReportType,
                                     HID_USAGE_PAGE_UNDEFINED,
                                     HIDP_LINK_COLLECTION_UNSPECIFIED,
                                     0,
                                     ValueCaps,
                                     ValueCapsLength,
                                     PreparsedData);
}
