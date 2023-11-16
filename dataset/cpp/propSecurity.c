/*******************************************************************************
*
*  (C) COPYRIGHT AUTHORS, 2015 - 2022
*
*  TITLE:       PROPSECURITY.C
*
*  VERSION:     2.01
*
*  DATE:        18 Dec 2022
*
* THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF
* ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED
* TO THE IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A
* PARTICULAR PURPOSE.
*
*******************************************************************************/
#include "global.h"
#include "propSecurityConsts.h"

typedef struct _ObjectSecurityVtbl ObjectSecurityVtbl, * PObjectSecurityVtbl;

//class
typedef struct _IObjectSecurity {
    ObjectSecurityVtbl* lpVtbl;
    ULONG RefCount;
    ULONG psiFlags;
    ULONG dwAccessMax;
    GENERIC_MAPPING GenericMapping;
    ACCESS_MASK ValidAccessMask;
    HINSTANCE hInstance;
    PROP_OBJECT_INFO* ObjectContext;
    PSI_ACCESS AccessTable;//dynamically allocated access table
    POPENOBJECTMETHOD OpenObjectMethod;
    PCLOSEOBJECTMETHOD CloseObjectMethod;
} IObjectSecurity, * PIObjectSecurity;


//Vtbl prototypes

typedef HRESULT(STDMETHODCALLTYPE* pQueryInterface)(
    _In_ IObjectSecurity* This,
    _In_ REFIID riid,
    _Out_ void** ppvObject);

typedef ULONG(STDMETHODCALLTYPE* pAddRef)(
    _In_ IObjectSecurity* This);

typedef ULONG(STDMETHODCALLTYPE* pRelease)(
    _In_ IObjectSecurity* This);

// *** ISecurityInformation methods ***
typedef HRESULT(STDMETHODCALLTYPE* pGetObjectInformation)(
    _In_ IObjectSecurity* This,
    _Out_ PSI_OBJECT_INFO pObjectInfo);

typedef HRESULT(STDMETHODCALLTYPE* pGetSecurity)(
    _In_ IObjectSecurity* This,
    _In_ SECURITY_INFORMATION RequestedInformation,
    _Out_ PSECURITY_DESCRIPTOR* ppSecurityDescriptor,
    _In_ BOOL fDefault);

typedef HRESULT(STDMETHODCALLTYPE* pSetSecurity)(
    _In_ IObjectSecurity* This,
    _In_ SECURITY_INFORMATION SecurityInformation,
    _In_ PSECURITY_DESCRIPTOR pSecurityDescriptor);

typedef HRESULT(STDMETHODCALLTYPE* pGetAccessRights)(
    _In_ IObjectSecurity* This,
    _In_ const GUID* pguidObjectType,
    _In_ DWORD dwFlags,
    _Out_ PSI_ACCESS* ppAccess,
    _Out_ ULONG* pcAccesses,
    _Out_ ULONG* piDefaultAccess);

typedef HRESULT(STDMETHODCALLTYPE* pMapGeneric)(
    _In_ IObjectSecurity* This,
    _In_ const GUID* pguidObjectType,
    _In_ UCHAR* pAceFlags,
    _In_ ACCESS_MASK* pMask);

typedef HRESULT(STDMETHODCALLTYPE* pGetInheritTypes)(
    _In_ IObjectSecurity* This,
    _Out_ PSI_INHERIT_TYPE* ppInheritTypes,
    _Out_ ULONG* pcInheritTypes);

typedef HRESULT(STDMETHODCALLTYPE* pPropertySheetPageCallback)(
    _In_ IObjectSecurity* This,
    _In_ HWND hwnd,
    _In_ UINT uMsg,
    _In_ SI_PAGE_TYPE uPage);

typedef struct _ObjectSecurityVtbl {
    pQueryInterface             QueryInterface;
    pAddRef                     AddRef;
    pRelease                    Release;
    pGetObjectInformation       GetObjectInformation;
    pGetSecurity                GetSecurity;
    pSetSecurity                SetSecurity;
    pGetAccessRights            GetAccessRights;
    pMapGeneric                 MapGeneric;
    pGetInheritTypes            GetInheritTypes;
    pPropertySheetPageCallback  PropertySheetPageCallback;
} ObjectSecurityVtbl, * PObjectSecurityVtbl;

/*
* propSecurityObjectSupported
*
* Purpose:
*
* Check we can show security for the given object type.
*
*/
BOOL propSecurityObjectSupported(
    _In_ WOBJ_OBJECT_TYPE nTypeIndex
)
{
    WOBJ_OBJECT_TYPE SecuritySupportedTypes[] = {
        ObjectTypeDesktop,
        ObjectTypeDevice,
        ObjectTypeDirectory,
        ObjectTypeEvent,
        ObjectTypeEventPair,
        ObjectTypeFile,
        ObjectTypeIoCompletion,
        ObjectTypeJob,
        ObjectTypeKey,
        ObjectTypeMemoryPartition,
        ObjectTypeMutant,
        ObjectTypePort,
        ObjectTypeProcess,
        ObjectTypeRegistryTransaction,
        ObjectTypeSection,
        ObjectTypeSemaphore,
        ObjectTypeSession,
        ObjectTypeSymbolicLink,
        ObjectTypeThread,
        ObjectTypeTimer,
        ObjectTypeToken,
        ObjectTypeWinstation
    };

    UINT i;
    for (i = 0; i < RTL_NUMBER_OF(SecuritySupportedTypes); i++) {
        if (SecuritySupportedTypes[i] == nTypeIndex)
            return TRUE;
    }

    return FALSE;
}

/*
* propGetAccessTable
*
* Purpose:
*
* Return access rights table and set generic mappings depending on object type.
*
*/
PSI_ACCESS propGetAccessTable(
    _In_ IObjectSecurity* This
)
{
    SI_ACCESS* AccessTable = NULL;

    switch (This->ObjectContext->ObjectTypeIndex) {

    case ObjectTypeDirectory:
        This->dwAccessMax = MAX_KNOWN_DIRECTORY_ACCESS_VALUE;
        AccessTable = (PSI_ACCESS)&DirectoryAccessValues;
        break;

    case ObjectTypeFile:
    case ObjectTypeDevice:
        This->dwAccessMax = MAX_KNOWN_FILE_ACCESS_VALUE;
        AccessTable = (PSI_ACCESS)&FileAccessValues;
        break;

    case ObjectTypeSection:
        This->dwAccessMax = MAX_KNOWN_SECTION_ACCESS_VALUE;
        AccessTable = (PSI_ACCESS)&SectionAccessValues;
        break;

    case ObjectTypeEvent:
        This->dwAccessMax = MAX_KNOWN_EVENT_ACCESS_VALUE;
        AccessTable = (PSI_ACCESS)&EventAccessValues;
        break;

    case ObjectTypeMutant:
        This->dwAccessMax = MAX_KNOWN_MUTANT_ACCESS_VALUE;
        AccessTable = (PSI_ACCESS)&MutantAccessValues;
        break;

    case ObjectTypeDesktop:
        This->dwAccessMax = MAX_KNOWN_DESKTOP_ACCESS_VALUE;
        AccessTable = (PSI_ACCESS)&DesktopAccessValues;
        break;

    case ObjectTypeWinstation:
        This->dwAccessMax = MAX_KNOWN_WINSTATION_ACCESS_VALUE;
        AccessTable = (PSI_ACCESS)&WinStationAccessValues;
        break;

    case ObjectTypeKey:
        This->dwAccessMax = MAX_KNOWN_KEY_ACCESS_VALUE;
        AccessTable = (PSI_ACCESS)&KeyAccessValues;
        break;

    case ObjectTypeSemaphore:
        This->dwAccessMax = MAX_KNOWN_SEMAPHORE_ACCESS_VALUE;
        AccessTable = (PSI_ACCESS)&SemaphoreAccessValues;
        break;

    case ObjectTypeSymbolicLink:
        This->dwAccessMax = MAX_KNOWN_SYMLINK_ACCESS_VALUE;
        AccessTable = (PSI_ACCESS)&SymlinkAccessValues;
        break;

    case ObjectTypeTimer:
        This->dwAccessMax = MAX_KNOWN_TIMER_ACCESS_VALUE;
        AccessTable = (PSI_ACCESS)&TimerAccessValues;
        break;

    case ObjectTypeJob:
        This->dwAccessMax = MAX_KNOWN_JOB_ACCESS_VALUE;
        AccessTable = (PSI_ACCESS)&JobAccessValues;
        break;

    case ObjectTypeSession:
        This->dwAccessMax = MAX_KNOWN_SESSION_ACCESS_VALUE;
        AccessTable = (PSI_ACCESS)&SessionAccessValues;
        break;

    case ObjectTypeIoCompletion:
    case ObjectTypeIoCompletionReserve:
        This->dwAccessMax = MAX_KNOWN_IOCOMPLETION_ACCESS_VALUE;
        AccessTable = (PSI_ACCESS)&IoCompletionAccessValues;
        break;

    case ObjectTypeMemoryPartition:
        This->dwAccessMax = MAX_KNOWN_MEMORYPARTITION_ACCESS_VALUE;
        AccessTable = (PSI_ACCESS)&MemoryPartitionAccessValues;
        break;

    case ObjectTypeProcess:
        This->dwAccessMax = MAX_KNOWN_PROCESS_ACCESS_VALUE;
        AccessTable = (PSI_ACCESS)&ProcessAccessValues;
        break;

    case ObjectTypeThread:
        This->dwAccessMax = MAX_KNOWN_THREAD_ACCESS_VALUE;
        AccessTable = (PSI_ACCESS)&ThreadAccessValues;
        break;

    case ObjectTypeToken:
        This->dwAccessMax = MAX_KNOWN_TOKEN_ACCESS_VALUE;
        AccessTable = (PSI_ACCESS)&TokenAccessValues;
        break;

    case ObjectTypePort:
        This->dwAccessMax = MAX_KNOWN_PORT_ACCESS_VALUE;
        AccessTable = (PSI_ACCESS)&PortAccessValues;
        break;

    case ObjectTypeRegistryTransaction:
        This->dwAccessMax = MAX_KNOWN_TRANSACTION_ACCESS_VALUE;
        AccessTable = (PSI_ACCESS)&TransactionAccessValues;
        break;
    }

    return AccessTable;
}

/*
* propGetObjectAccessMask
*
* Purpose:
*
* Query required access mask to get/set object security.
*
*/
ACCESS_MASK propGetObjectAccessMask(
    _In_ SECURITY_INFORMATION SecurityInformation,
    _In_ BOOL fSet
)
{
    ACCESS_MASK AccessMask = 0;

    if (fSet) {
        if ((SecurityInformation & OWNER_SECURITY_INFORMATION) ||
            (SecurityInformation & GROUP_SECURITY_INFORMATION) ||
            (SecurityInformation & LABEL_SECURITY_INFORMATION))
        {
            AccessMask |= WRITE_OWNER;
        }

        if ((SecurityInformation & DACL_SECURITY_INFORMATION) ||
            (SecurityInformation & ATTRIBUTE_SECURITY_INFORMATION) ||
            (SecurityInformation & PROTECTED_DACL_SECURITY_INFORMATION) ||
            (SecurityInformation & UNPROTECTED_DACL_SECURITY_INFORMATION))
        {
            AccessMask |= WRITE_DAC;
        }

        if ((SecurityInformation & SACL_SECURITY_INFORMATION) ||
            (SecurityInformation & SCOPE_SECURITY_INFORMATION) ||
            (SecurityInformation & PROTECTED_SACL_SECURITY_INFORMATION) ||
            (SecurityInformation & UNPROTECTED_SACL_SECURITY_INFORMATION))
        {
            AccessMask |= ACCESS_SYSTEM_SECURITY;
        }

        if (SecurityInformation & BACKUP_SECURITY_INFORMATION) {

            AccessMask |= WRITE_DAC | WRITE_OWNER | ACCESS_SYSTEM_SECURITY;
        }
    }
    else {
        //get
        if ((SecurityInformation & OWNER_SECURITY_INFORMATION) ||
            (SecurityInformation & GROUP_SECURITY_INFORMATION) ||
            (SecurityInformation & DACL_SECURITY_INFORMATION) ||
            (SecurityInformation & LABEL_SECURITY_INFORMATION) ||
            (SecurityInformation & ATTRIBUTE_SECURITY_INFORMATION) ||
            (SecurityInformation & SCOPE_SECURITY_INFORMATION))
        {
            AccessMask |= READ_CONTROL;
        }

        if (SecurityInformation & SACL_SECURITY_INFORMATION) {
            AccessMask |= ACCESS_SYSTEM_SECURITY;
        }

        if (SecurityInformation & BACKUP_SECURITY_INFORMATION) {
            AccessMask |= READ_CONTROL | ACCESS_SYSTEM_SECURITY;
        }

    }
    return AccessMask;
}

HRESULT STDMETHODCALLTYPE QueryInterface(
    _In_ IObjectSecurity* This,
    _In_ REFIID riid,
    _Out_ void** ppvObject
)
{
#if defined(__cplusplus)
    if (IsEqualIID(riid, IID_ISecurityInformation) ||
        IsEqualIID(riid, IID_IUnknown))
#else
    if (IsEqualIID(riid, &IID_ISecurityInformation) ||
        IsEqualIID(riid, &IID_IUnknown))
#endif
    {
        *ppvObject = This;
        This->lpVtbl->AddRef(This);
        return S_OK;
    }

    *ppvObject = NULL;
    return E_NOINTERFACE;
}

ULONG STDMETHODCALLTYPE AddRef(
    _In_ IObjectSecurity* This
)
{
    return ++This->RefCount;
}

ULONG STDMETHODCALLTYPE Release(
    _In_ IObjectSecurity* This
)
{
    This->RefCount--;

    if (This->RefCount == 0) {
        if (This->AccessTable) {
            supHeapFree(This->AccessTable);
        }
        supHeapFree(This);
        return S_OK;
    }
    return This->RefCount;
}

HRESULT STDMETHODCALLTYPE GetObjectInformation(
    _In_ IObjectSecurity* This,
    _Out_ PSI_OBJECT_INFO pObjectInfo
)
{
    pObjectInfo->dwFlags = This->psiFlags;
    pObjectInfo->hInstance = This->hInstance;
    pObjectInfo->pszPageTitle = TEXT("Security");
    pObjectInfo->pszObjectName = This->ObjectContext->NtObjectName.Buffer;
    return S_OK;
}

HRESULT STDMETHODCALLTYPE GetAccessRights(
    _In_ IObjectSecurity* This,
    _In_ const GUID* pguidObjectType,
    _In_ DWORD dwFlags,
    _Out_ PSI_ACCESS* ppAccess,
    _Out_ ULONG* pcAccesses,
    _Out_ ULONG* piDefaultAccess
)
{
    UNREFERENCED_PARAMETER(pguidObjectType);
    UNREFERENCED_PARAMETER(dwFlags);

    *ppAccess = This->AccessTable;
    *pcAccesses = This->dwAccessMax;
    *piDefaultAccess = 0;

    return S_OK;
}

HRESULT STDMETHODCALLTYPE GetSecurity(
    _In_ IObjectSecurity* This,
    _In_ SECURITY_INFORMATION RequestedInformation,
    _Out_ PSECURITY_DESCRIPTOR* ppSecurityDescriptor,
    _In_ BOOL fDefault
)
{
    HRESULT                hResult;
    HANDLE                 hObject;
    ULONG                  bytesNeeded;
    NTSTATUS               status;
    ACCESS_MASK            DesiredAccess;
    PSECURITY_DESCRIPTOR   PSD;

    if (fDefault) {
        return E_NOTIMPL;
    }

    //open object
    hObject = NULL;
    DesiredAccess = propGetObjectAccessMask(RequestedInformation, FALSE);

    if (!This->OpenObjectMethod(This->ObjectContext, &hObject, DesiredAccess)) {
        return HRESULT_FROM_WIN32(GetLastError());
    }

    //query object SD
    //warning: system free SD with LocalFree on security dialog destroy
    bytesNeeded = 0x100;
    PSD = LocalAlloc(LPTR, bytesNeeded);
    if (PSD == NULL) {
        hResult = HRESULT_FROM_WIN32(GetLastError());
        goto Done;
    }

    status = NtQuerySecurityObject(hObject, RequestedInformation,
        PSD, bytesNeeded, &bytesNeeded);

    if (status == STATUS_BUFFER_TOO_SMALL) {
        LocalFree(PSD);
        PSD = LocalAlloc(LPTR, bytesNeeded);
        if (PSD == NULL) {
            hResult = HRESULT_FROM_WIN32(GetLastError());
            goto Done;
        }
        status = NtQuerySecurityObject(
            hObject,
            RequestedInformation,
            PSD,
            bytesNeeded,
            &bytesNeeded
        );
    }

    hResult = HRESULT_FROM_WIN32(RtlNtStatusToDosError(status));
    *ppSecurityDescriptor = PSD;

Done:
    //cleanup
    This->CloseObjectMethod(This->ObjectContext, hObject);
    return hResult;
}

HRESULT STDMETHODCALLTYPE SetSecurity(
    _In_ IObjectSecurity* This,
    _In_ SECURITY_INFORMATION SecurityInformation,
    _In_ PSECURITY_DESCRIPTOR pSecurityDescriptor
)
{
    NTSTATUS       status;
    HANDLE         hObject = NULL;
    ACCESS_MASK    DesiredAccess;

    DesiredAccess = propGetObjectAccessMask(SecurityInformation, TRUE);
    if (!This->OpenObjectMethod(This->ObjectContext, &hObject, DesiredAccess)) {
        return HRESULT_FROM_WIN32(GetLastError());
    }

    status = NtSetSecurityObject(hObject, SecurityInformation, pSecurityDescriptor);

    //cleanup
    This->CloseObjectMethod(This->ObjectContext, hObject);
    return HRESULT_FROM_WIN32(RtlNtStatusToDosError(status));
}

HRESULT STDMETHODCALLTYPE MapGeneric(
    _In_ IObjectSecurity* This,
    _In_ const GUID* pguidObjectType,
    _In_ UCHAR* pAceFlags,
    _In_ ACCESS_MASK* pMask
)
{
    UNREFERENCED_PARAMETER(pguidObjectType);
    UNREFERENCED_PARAMETER(pAceFlags);

    RtlMapGenericMask(pMask, &This->GenericMapping);
    return S_OK;
}

HRESULT STDMETHODCALLTYPE GetInheritTypes(
    _In_ IObjectSecurity* This,
    _Out_ PSI_INHERIT_TYPE* ppInheritTypes,
    _Out_ ULONG* pcInheritTypes
)
{
    UNREFERENCED_PARAMETER(This);
    UNREFERENCED_PARAMETER(ppInheritTypes);
    UNREFERENCED_PARAMETER(pcInheritTypes);

    return E_NOTIMPL;
}

HRESULT STDMETHODCALLTYPE PropertySheetPageCallback(
    _In_ IObjectSecurity* This,
    _In_ HWND hwnd,
    _In_ UINT uMsg,
    _In_ SI_PAGE_TYPE uPage
)
{
    UNREFERENCED_PARAMETER(This);
    UNREFERENCED_PARAMETER(hwnd);
    UNREFERENCED_PARAMETER(uMsg);
    UNREFERENCED_PARAMETER(uPage);

    return E_NOTIMPL;
}

//class methods
ObjectSecurityVtbl g_Vtbl = {
    QueryInterface,
    AddRef,
    Release,
    GetObjectInformation,
    GetSecurity,
    SetSecurity,
    GetAccessRights,
    MapGeneric,
    GetInheritTypes,
    PropertySheetPageCallback
};

/*
* propSecurityConstructor
*
* Purpose:
*
* Initialize class object, query type info, Vtbl, AccessTable, object specific methods.
*
*/
HRESULT propSecurityConstructor(
    _In_ IObjectSecurity* This,
    _In_ PROP_OBJECT_INFO* Context,
    _In_ POPENOBJECTMETHOD OpenObjectMethod,
    _In_opt_ PCLOSEOBJECTMETHOD CloseObjectMethod,
    _In_ ULONG psiFlags
)
{
    ULONG                       bytesNeeded = 0L;
    NTSTATUS                    status;
    SIZE_T                      Size;
    HRESULT                     hResult;
    HANDLE                      hObject = NULL;
    PSI_ACCESS                  TypeAccessTable = NULL;
    POBJECT_TYPE_INFORMATION    TypeInfo = NULL;

    This->ObjectContext = Context;
    This->OpenObjectMethod = OpenObjectMethod;

    if (CloseObjectMethod == NULL) {
        This->CloseObjectMethod = (PCLOSEOBJECTMETHOD)supCloseObjectFromContext;
    }
    else {
        This->CloseObjectMethod = CloseObjectMethod;
    }

    do {

        if (!This->OpenObjectMethod(Context, &hObject, READ_CONTROL)) {
            hResult = E_ACCESSDENIED;
            break;
        }

        status = supQueryObjectInformation(hObject,
            ObjectTypeInformation,
            &TypeInfo,
            &bytesNeeded);

        if (!NT_SUCCESS(status)) {
            hResult = HRESULT_FROM_WIN32(RtlNtStatusToDosError(status));
            break;
        }

        This->GenericMapping = TypeInfo->GenericMapping;
        This->ValidAccessMask = TypeInfo->ValidAccessMask;

        supHeapFree(TypeInfo);
        TypeInfo = NULL;

        This->lpVtbl = &g_Vtbl;
        This->hInstance = g_WinObj.hInstance;
        This->psiFlags = psiFlags;

        TypeAccessTable = propGetAccessTable(This);

        //allocate access table
        Size = (MAX_KNOWN_GENERAL_ACCESS_VALUE + (SIZE_T)This->dwAccessMax) * sizeof(SI_ACCESS);
        This->AccessTable = (PSI_ACCESS)supHeapAlloc(Size);
        if (This->AccessTable == NULL) {
            hResult = HRESULT_FROM_WIN32(GetLastError());
            break;
        }

        //copy object specific access table if it present
        if (TypeAccessTable && This->dwAccessMax) {

            RtlCopyMemory(This->AccessTable,
                TypeAccessTable,
                This->dwAccessMax * sizeof(SI_ACCESS));

        }

        if (This->ValidAccessMask & DELETE) {
            RtlCopyMemory(&This->AccessTable[This->dwAccessMax++],
                &GeneralAccessValues[0], sizeof(SI_ACCESS));
        }
        if (This->ValidAccessMask & READ_CONTROL) {
            RtlCopyMemory(&This->AccessTable[This->dwAccessMax++],
                &GeneralAccessValues[1], sizeof(SI_ACCESS));
        }
        if (This->ValidAccessMask & WRITE_DAC) {
            RtlCopyMemory(&This->AccessTable[This->dwAccessMax++],
                &GeneralAccessValues[2], sizeof(SI_ACCESS));
        }
        if (This->ValidAccessMask & WRITE_OWNER) {
            RtlCopyMemory(&This->AccessTable[This->dwAccessMax++],
                &GeneralAccessValues[3], sizeof(SI_ACCESS));
        }
        if (This->ValidAccessMask & SYNCHRONIZE) {
            RtlCopyMemory(&This->AccessTable[This->dwAccessMax++],
                &GeneralAccessValues[4], sizeof(SI_ACCESS));
        }
        hResult = S_OK;

    } while (FALSE);

    //cleanup
    if (hObject) This->CloseObjectMethod(Context, hObject);
    if (TypeInfo) {
        supHeapFree(TypeInfo);
    }
    return hResult;
}

/*
* propSecurityCreatePage
*
* Purpose:
*
* Create Security page.
*
*
* Page creation methods call sequence:
* AddRef->QueryInterface->GetObjectInformation->PropertySheetPageCallback.
*
* Page query info call sequence:
* PropertySheetPageCallback->GetObjectInformation->GetAccessRights->GetSecurity->MapGeneric.
*
* Page close call sequence:
* PropertySheetPageCallback->Release.
*
*/
HPROPSHEETPAGE propSecurityCreatePage(
    _In_ PROP_OBJECT_INFO* Context,
    _In_ POPENOBJECTMETHOD OpenObjectMethod,
    _In_opt_ PCLOSEOBJECTMETHOD CloseObjectMethod,
    _In_ ULONG psiFlags
)
{
    IObjectSecurity* psi;

    if (!propSecurityObjectSupported(Context->ObjectTypeIndex)) {
        return NULL;
    }

    psi = (IObjectSecurity*)supHeapAlloc(sizeof(IObjectSecurity));
    if (psi == NULL)
        return NULL;

    if (S_OK != propSecurityConstructor(
        psi,
        Context,
        OpenObjectMethod,
        CloseObjectMethod,
        psiFlags))
    {
        supHeapFree(psi);
        return NULL;
    }

    return CreateSecurityPage((LPSECURITYINFO)psi);
}
