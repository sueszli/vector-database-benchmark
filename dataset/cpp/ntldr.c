/************************************************************************************
*
*  (C) COPYRIGHT AUTHORS, 2014 - 2023
*
*  TITLE:       NTLDR.C
*
*  VERSION:     1.22
*
*  DATE:        25 Jul 2023
*
*  NT loader raw parsing related code.
*
*  Depends on:    ntos.h
*                 apisetx.h
*                 minirtl
*
* THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF
* ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED
* TO THE IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A
* PARTICULAR PURPOSE.
*
************************************************************************************/

#include "ntldr.h"

PFNNTLDR_EXCEPT_FILTER NtpLdrExceptionFilter = NULL;

INT NtLdrExceptionFilter(
    _In_ UINT ExceptionCode,
    _In_ EXCEPTION_POINTERS* ExceptionPointers);

#define NTLDR_EXCEPTION_FILTER NtLdrExceptionFilter(GetExceptionCode(), GetExceptionInformation())

/*
* NtLdrExceptionFilter
*
* Purpose:
*
* Default exception filter with optional custom callback.
*
*/
INT NtLdrExceptionFilter(
    _In_ UINT ExceptionCode,
    _In_ EXCEPTION_POINTERS* ExceptionPointers
)
{
    if (NtpLdrExceptionFilter)
        return NtpLdrExceptionFilter(ExceptionCode, ExceptionPointers);

    return EXCEPTION_EXECUTE_HANDLER;
}

/*
* NtRawGetProcAddress
*
* Purpose:
*
* Custom GPA.
*
*/
NTSTATUS NtRawGetProcAddress(
    _In_ LPVOID Module,
    _In_ LPCSTR ProcName,
    _In_ PRESOLVE_INFO Pointer
)
{
    PIMAGE_NT_HEADERS NtHeaders;
    PIMAGE_EXPORT_DIRECTORY exp;
    PDWORD fntable, nametable;
    PWORD ordtable;
    ULONG mid, high, low;
    ULONG_PTR fnptr, exprva, expsize;
    int r;

    if (Module == NULL)
        return STATUS_INVALID_PARAMETER_1;

    NtHeaders = RtlImageNtHeader(Module);
    if (NtHeaders->OptionalHeader.NumberOfRvaAndSizes <= IMAGE_DIRECTORY_ENTRY_EXPORT)
        return STATUS_OBJECT_NAME_NOT_FOUND;

    exprva = NtHeaders->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT].VirtualAddress;
    if (exprva == 0)
        return STATUS_INVALID_IMAGE_FORMAT;

    expsize = NtHeaders->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT].Size;

    exp = (PIMAGE_EXPORT_DIRECTORY)((ULONG_PTR)Module + exprva);
    fntable = (PDWORD)((ULONG_PTR)Module + exp->AddressOfFunctions);

    if ((ULONG_PTR)ProcName < 0x10000) {
        // ProcName is ordinal
        if (
            ((ULONG_PTR)ProcName < (ULONG_PTR)exp->Base) ||
            ((ULONG_PTR)ProcName >= (ULONG_PTR)exp->Base + exp->NumberOfFunctions))
            return STATUS_OBJECT_NAME_NOT_FOUND;

        fnptr = fntable[(ULONG_PTR)ProcName - exp->Base];

    }
    else {
        // ProcName is ANSI string
        nametable = (PDWORD)((ULONG_PTR)Module + exp->AddressOfNames);
        ordtable = (PWORD)((ULONG_PTR)Module + exp->AddressOfNameOrdinals);

        if (exp->NumberOfNames == 0)
            return STATUS_OBJECT_NAME_NOT_FOUND;

        low = 0;
        high = exp->NumberOfNames;

        do {
            mid = low + (high - low) / 2;
            r = _strcmp_a(ProcName, (LPCSTR)((ULONG_PTR)Module + nametable[mid]));

            if (r > 0)
            {
                low = mid + 1;
            }
            else
            {
                if (r < 0)
                    high = mid;
                else
                    break;
            }
        } while (low < high);

        if (r == 0)
            fnptr = fntable[ordtable[mid]];
        else
            return STATUS_OBJECT_NAME_NOT_FOUND;
    }

    if ((fnptr >= exprva) && (fnptr < exprva + expsize))
        Pointer->ResultType = ForwarderString;
    else
        Pointer->ResultType = FunctionCode;

    Pointer->Function = (LPVOID)((ULONG_PTR)Module + fnptr);
    return STATUS_SUCCESS;
}

/*
* NtRawEnumSyscallExports
*
* Purpose:
*
* Enumerate syscall module exports to the table.
*
*/
_Success_(return != 0)
ULONG NtRawEnumSyscallExports(
    _In_ HANDLE HeapHandle,
    _In_ LPVOID Module,
    _Out_ PRAW_SYSCALL_ENTRY * SyscallTable
)
{
    PIMAGE_EXPORT_DIRECTORY pExportDirectory;
    PDWORD FnPtrTable, NameTable;
    PWORD NameOrdTable;
    ULONG_PTR fnptr;
    ULONG i, j, result, exportSize;
    PRAW_SYSCALL_ENTRY newEntry;

    pExportDirectory = (PIMAGE_EXPORT_DIRECTORY)RtlImageDirectoryEntryToData(
        Module,
        TRUE,
        IMAGE_DIRECTORY_ENTRY_EXPORT,
        &exportSize);

    if (pExportDirectory == NULL)
        return 0;

    FnPtrTable = (PDWORD)((ULONG_PTR)Module + pExportDirectory->AddressOfFunctions);
    NameTable = (PDWORD)((ULONG_PTR)Module + pExportDirectory->AddressOfNames);
    NameOrdTable = (PWORD)((ULONG_PTR)Module + pExportDirectory->AddressOfNameOrdinals);

    result = 0;

    for (i = 0; i < pExportDirectory->NumberOfFunctions; ++i) {

        fnptr = (ULONG_PTR)Module + FnPtrTable[i];
        if (*(PDWORD)fnptr != 0xb8d18b4c) //mov r10, rcx; mov eax
            continue;

        newEntry = (PRAW_SYSCALL_ENTRY)RtlAllocateHeap(HeapHandle,
            HEAP_ZERO_MEMORY, sizeof(RAW_SYSCALL_ENTRY));

        if (newEntry == NULL)
            break;

        newEntry->Index = *(PDWORD)(fnptr + 4);

        for (j = 0; j < pExportDirectory->NumberOfNames; ++j)
        {
            if (NameOrdTable[j] == i)
            {
                _strncpy_a(&newEntry->Name[0],
                    sizeof(newEntry->Name),
                    (LPCSTR)((ULONG_PTR)Module + NameTable[j]),
                    sizeof(newEntry->Name));

                break;
            }
        }

        ++result;

        *SyscallTable = newEntry;
        SyscallTable = &newEntry->NextEntry;
    }

    return result;
}

/*
* NtRawIATEntryToImport
*
* Purpose:
*
* Resolve function name.
*
*/
_Success_(return != NULL)
LPCSTR NtRawIATEntryToImport(
    _In_ LPVOID Module,
    _In_ LPVOID IATEntry,
    _Out_opt_ LPCSTR * ImportModuleName
)
{
    PIMAGE_IMPORT_DESCRIPTOR pImageImportDescriptor;
    ULONG_PTR* rname;
    ULONG size;
    LPVOID* raddr;

    if (ImportModuleName)
        *ImportModuleName = NULL;

    pImageImportDescriptor = (PIMAGE_IMPORT_DESCRIPTOR)RtlImageDirectoryEntryToData(
        Module,
        TRUE,
        IMAGE_DIRECTORY_ENTRY_IMPORT,
        &size);

    if (pImageImportDescriptor == NULL)
        return 0;

    while (pImageImportDescriptor->Name != 0) {
        raddr = (LPVOID*)((ULONG_PTR)Module + pImageImportDescriptor->FirstThunk);
        if (pImageImportDescriptor->OriginalFirstThunk == 0)
            rname = (ULONG_PTR*)raddr;
        else
            rname = (ULONG_PTR*)((ULONG_PTR)Module + pImageImportDescriptor->OriginalFirstThunk);

        while (*rname != 0) {
            if (IATEntry == raddr)
            {
                if (((*rname) & IMAGE_ORDINAL_FLAG) == 0)
                {
                    if (ImportModuleName) {
                        *ImportModuleName = (LPCSTR)((ULONG_PTR)Module + pImageImportDescriptor->Name);
                    }
                    return (LPCSTR) & ((PIMAGE_IMPORT_BY_NAME)((ULONG_PTR)Module + *rname))->Name;
                }
            }

            ++rname;
            ++raddr;
        }
        ++pImageImportDescriptor;
    }

    return NULL;
}

/*
* ApiSetpSearchForApiSetHost
*
* Purpose:
*
* Resolve alias name if present.
* Directly ripped from ntdll!ApiSetpSearchForApiSetHost.
*
*/
PAPI_SET_VALUE_ENTRY_V6 ApiSetpSearchForApiSetHost(
    _In_ PAPI_SET_NAMESPACE_ENTRY_V6 Entry,
    _In_ PWCHAR ApiSetToResolve,
    _In_ USHORT ApiSetToResolveLength,
    _In_ PVOID Namespace)
{
    API_SET_VALUE_ENTRY_V6* ValueEntry;
    API_SET_VALUE_ENTRY_V6* AliasValueEntry, * Result = NULL;
    ULONG AliasCount, i, AliasIndex;
    PWCHAR AliasName;
    LONG CompareResult;

    ValueEntry = API_SET_TO_VALUE_ENTRY(Namespace, Entry, 0);
    AliasCount = Entry->Count;

    if (AliasCount >= 1) {

        i = 1;

        do {
            AliasIndex = (AliasCount + i) >> 1;
            AliasValueEntry = API_SET_TO_VALUE_ENTRY(Namespace, Entry, AliasIndex);
            AliasName = API_SET_TO_VALUE_NAME(Namespace, AliasValueEntry);

            CompareResult = RtlCompareUnicodeStrings(ApiSetToResolve,
                ApiSetToResolveLength,
                AliasName,
                AliasValueEntry->NameLength >> 1,
                TRUE);

            if (CompareResult < 0) {
                AliasCount = AliasIndex - 1;
            }
            else {
                if (CompareResult == 0) {

                    Result = API_SET_TO_VALUE_ENTRY(Namespace,
                        Entry,
                        ((AliasCount + i) >> 1));

                    break;
                }
                i = (AliasCount + 1);
            }

        } while (i <= AliasCount);

    }
    else {
        Result = ValueEntry;
    }

    return Result;
}

/*
* ApiSetpSearchForApiSet
*
* Purpose:
*
* Find apiset entry by hash from it name.
*
*/
PAPI_SET_NAMESPACE_ENTRY_V6 ApiSetpSearchForApiSet(
    _In_ PVOID Namespace,
    _In_ PWCHAR ResolveName,
    _In_ USHORT ResolveNameEffectiveLength)
{
    ULONG LookupHash = 0, i, c, HashIndex, EntryCount, EntryHash;
    WCHAR ch;

    PWCHAR NamespaceEntryName;
    API_SET_HASH_ENTRY_V6* LookupHashEntry;
    PAPI_SET_NAMESPACE_ENTRY_V6 NamespaceEntry = NULL;
    PAPI_SET_NAMESPACE_ARRAY_V6 ApiSetNamespace = (PAPI_SET_NAMESPACE_ARRAY_V6)Namespace;

    if ((ApiSetNamespace->Count == 0) || (ResolveNameEffectiveLength == 0))
        return NULL;

    //
    // Calculate lookup hash.
    //
    for (i = 0; i < ResolveNameEffectiveLength; i++) {
        ch = locase_w(ResolveName[i]);
        LookupHash = LookupHash * ApiSetNamespace->HashMultiplier + ch;
    }

    //
    // Search for hash.
    //
    c = 0;
    EntryCount = ApiSetNamespace->Count - 1;
    do {

        HashIndex = (EntryCount + c) >> 1;

        LookupHashEntry = API_SET_TO_HASH_ENTRY(ApiSetNamespace, HashIndex);
        EntryHash = LookupHashEntry->Hash;

        if (LookupHash < EntryHash) {
            EntryCount = HashIndex - 1;
            if (c > EntryCount)
                return NULL;
            continue;
        }

        if (EntryHash == LookupHash) {
            //
            // Hash found, query namespace entry and break.
            //
            NamespaceEntry = API_SET_TO_NAMESPACE_ENTRY(ApiSetNamespace, LookupHashEntry);
            break;
        }

        c = HashIndex + 1;

        if (c > EntryCount)
            return NULL;

    } while (1);

    if (NamespaceEntry == NULL)
        return NULL;

    //
    // Verify entry name.
    //
    NamespaceEntryName = API_SET_TO_NAMESPACE_ENTRY_NAME(ApiSetNamespace, NamespaceEntry);

    if (0 == RtlCompareUnicodeStrings(ResolveName,
        ResolveNameEffectiveLength,
        NamespaceEntryName,
        (NamespaceEntry->HashNameLength >> 1),
        TRUE))
    {
        return NamespaceEntry;
    }

    return NULL;
}

/*
* NtRawApiSetResolveLibrary
*
* Purpose:
*
* Resolve apiset library name.
*
*/
NTSTATUS NtRawApiSetResolveLibrary(
    _In_ PVOID Namespace,
    _In_ PCUNICODE_STRING ApiSetToResolve,
    _In_opt_ PCUNICODE_STRING ApiSetParentName,
    _Inout_ PUNICODE_STRING ResolvedHostLibraryName
)
{
    NTSTATUS Status = STATUS_APISET_NOT_PRESENT;
    PWCHAR BufferPtr;
    USHORT Length;
    ULONG64 SchemaPrefix;
    API_SET_NAMESPACE_ENTRY_V6* ResolvedEntry;
    API_SET_VALUE_ENTRY_V6* HostLibraryEntry = NULL;
    PAPI_SET_NAMESPACE_ARRAY_V6 ApiSetNamespace = (PAPI_SET_NAMESPACE_ARRAY_V6)Namespace;

    __try {

        //
        // Only Win10+ version supported.
        //
        if (ApiSetNamespace->Version != API_SET_SCHEMA_VERSION_V6)
            return STATUS_UNKNOWN_REVISION;

        //
        // Must include something except prefix.
        //
        if (ApiSetToResolve->Length <= API_SET_PREFIX_NAME_U_SIZE)
            return STATUS_INVALID_PARAMETER_2;

        //
        // Check prefix.
        //
        SchemaPrefix = API_SET_TO_UPPER_PREFIX(((ULONG64*)ApiSetToResolve->Buffer)[0]);
        if ((SchemaPrefix != API_SET_PREFIX_API) && (SchemaPrefix != API_SET_PREFIX_EXT)) //API- or EXT- only
            return STATUS_INVALID_PARAMETER;

        //
        // Calculate length without everything after last hyphen including dll suffix.
        //
        BufferPtr = (PWCHAR)RtlOffsetToPointer(ApiSetToResolve->Buffer, ApiSetToResolve->Length);

        Length = ApiSetToResolve->Length;

        do {
            if (Length <= 1)
                break;

            Length -= sizeof(WCHAR);
            --BufferPtr;

        } while (*BufferPtr != L'-');

        Length = (USHORT)Length >> 1;

        //
        // Resolve apiset entry.
        //
        ResolvedEntry = ApiSetpSearchForApiSet(
            Namespace,
            ApiSetToResolve->Buffer,
            Length);

        if (ResolvedEntry == NULL)
            return STATUS_INVALID_PARAMETER;

        //
        // If parent name specified and resolved entry has more than 1 value entry check it out.
        //
        if (ApiSetParentName && ResolvedEntry->Count > 1) {

            HostLibraryEntry = ApiSetpSearchForApiSetHost(ResolvedEntry,
                ApiSetParentName->Buffer,
                ApiSetParentName->Length >> 1,
                Namespace);

        }
        else {

            //
            // If resolved apiset entry has value check it out.
            //
            if (ResolvedEntry->Count > 0) {
                HostLibraryEntry = API_SET_TO_VALUE_ENTRY(Namespace, ResolvedEntry, 0);
            }
        }

        //
        // Set output parameter if host library resolved.
        //
        if (HostLibraryEntry) {
            if (!API_SET_EMPTY_NAMESPACE_VALUE(HostLibraryEntry)) {

                //
                // Host library name is not null terminated, handle that.
                //
                BufferPtr = (PWSTR)RtlAllocateHeap(NtCurrentPeb()->ProcessHeap, HEAP_ZERO_MEMORY,
                    HostLibraryEntry->ValueLength + sizeof(UNICODE_NULL));

                if (BufferPtr) {

                    RtlCopyMemory(BufferPtr,
                        (PWSTR)RtlOffsetToPointer(Namespace, HostLibraryEntry->ValueOffset),
                        (SIZE_T)HostLibraryEntry->ValueLength);

                    ResolvedHostLibraryName->Length = (USHORT)HostLibraryEntry->ValueLength;
                    ResolvedHostLibraryName->MaximumLength = (USHORT)HostLibraryEntry->ValueLength;
                    ResolvedHostLibraryName->Buffer = BufferPtr;
                    Status = STATUS_SUCCESS;
                }
            }
        }
    }
    __except (NTLDR_EXCEPTION_FILTER)
    {
        return GetExceptionCode();
    }

    return Status;
}
