/*
 * COPYRIGHT:       See COPYING in the top level directory
 * PROJECT:         ReactOS net command
 * FILE:            base/applications/network/net/cmdLocalGroup.c
 * PURPOSE:
 *
 * PROGRAMMERS:     Eric Kohl
 */

#include "net.h"


static
int
CompareInfo(const void *a,
            const void *b)
{
    return _wcsicmp(((PLOCALGROUP_INFO_0)a)->lgrpi0_name,
                    ((PLOCALGROUP_INFO_0)b)->lgrpi0_name);
}


static
NET_API_STATUS
EnumerateLocalGroups(VOID)
{
    PLOCALGROUP_INFO_0 pBuffer = NULL;
    PSERVER_INFO_100 pServer = NULL;
    DWORD dwRead = 0, dwTotal = 0;
    DWORD i;
    DWORD_PTR ResumeHandle = 0;
    NET_API_STATUS Status;


    Status = NetServerGetInfo(NULL,
                              100,
                              (LPBYTE*)&pServer);
    if (Status != NERR_Success)
        return Status;

    ConPuts(StdOut, L"\n");
    PrintMessageStringV(4405, pServer->sv100_name);
    ConPuts(StdOut, L"\n");
    PrintPadding(L'-', 79);
    ConPuts(StdOut, L"\n");

    NetApiBufferFree(pServer);

    Status = NetLocalGroupEnum(NULL,
                               0,
                               (LPBYTE*)&pBuffer,
                               MAX_PREFERRED_LENGTH,
                               &dwRead,
                               &dwTotal,
                               &ResumeHandle);
    if (Status != NERR_Success)
        return Status;

    qsort(pBuffer,
          dwRead,
          sizeof(PLOCALGROUP_INFO_0),
          CompareInfo);

    for (i = 0; i < dwRead; i++)
    {
        if (pBuffer[i].lgrpi0_name)
            ConPrintf(StdOut, L"*%s\n", pBuffer[i].lgrpi0_name);
    }

    NetApiBufferFree(pBuffer);

    return NERR_Success;
}


static
NET_API_STATUS
DisplayLocalGroup(LPWSTR lpGroupName)
{
    PLOCALGROUP_INFO_1 pGroupInfo = NULL;
    PLOCALGROUP_MEMBERS_INFO_3 pMembers = NULL;
    PSERVER_INFO_100 pServer = NULL;
    LPWSTR *pNames = NULL;
    DWORD dwRead = 0;
    DWORD dwTotal = 0;
    DWORD_PTR ResumeHandle = 0;
    DWORD i;
    DWORD len;
    INT nPaddedLength = 18;
    NET_API_STATUS Status;

    Status = NetLocalGroupGetInfo(NULL,
                                  lpGroupName,
                                  1,
                                  (LPBYTE*)&pGroupInfo);
    if (Status != NERR_Success)
        return Status;

    Status = NetLocalGroupGetMembers(NULL,
                                     lpGroupName,
                                     3,
                                     (LPBYTE*)&pMembers,
                                     MAX_PREFERRED_LENGTH,
                                     &dwRead,
                                     &dwTotal,
                                     &ResumeHandle);
    if (Status != NERR_Success)
        goto done;

    Status = NetServerGetInfo(NULL,
                              100,
                              (LPBYTE*)&pServer);
    if (Status != NERR_Success)
        goto done;

    pNames = RtlAllocateHeap(RtlGetProcessHeap(),
                             HEAP_ZERO_MEMORY,
                             dwRead * sizeof(LPWSTR));
    if (pNames == NULL)
    {
        Status = ERROR_OUTOFMEMORY;
        goto done;
    }

    len = wcslen(pServer->sv100_name);
    for (i = 0; i < dwRead; i++)
    {
         if (!wcsncmp(pMembers[i].lgrmi3_domainandname, pServer->sv100_name, len))
             pNames[i] = &pMembers[i].lgrmi3_domainandname[len + 1];
         else
             pNames[i] = pMembers[i].lgrmi3_domainandname;
    }

    PrintPaddedMessageString(4406, nPaddedLength);
    ConPrintf(StdOut, L"%s\n", pGroupInfo->lgrpi1_name);

    PrintPaddedMessageString(4407, nPaddedLength);
    ConPrintf(StdOut, L"%s\n", pGroupInfo->lgrpi1_comment);

    ConPuts(StdOut, L"\n");

    PrintMessageString(4408);
    ConPuts(StdOut, L"\n");

    PrintPadding(L'-', 79);
    ConPuts(StdOut, L"\n");

    for (i = 0; i < dwRead; i++)
    {
        if (pNames[i])
            ConPrintf(StdOut, L"%s\n", pNames[i]);
    }

done:
    if (pNames != NULL)
        RtlFreeHeap(RtlGetProcessHeap(), 0, pNames);

    if (pServer != NULL)
        NetApiBufferFree(pServer);

    if (pMembers != NULL)
        NetApiBufferFree(pMembers);

    if (pGroupInfo != NULL)
        NetApiBufferFree(pGroupInfo);

    return Status;
}


INT
cmdLocalGroup(
    INT argc,
    WCHAR **argv)
{
    INT i, j;
    INT result = 0;
    ULONG dwMemberCount = 0;
    BOOL bAdd = FALSE;
    BOOL bDelete = FALSE;
#if 0
    BOOL bDomain = FALSE;
#endif
    LPWSTR lpGroupName = NULL;
    LPWSTR lpComment = NULL;
    LPLOCALGROUP_MEMBERS_INFO_3 lpMembers = NULL;
    LOCALGROUP_INFO_0 Info0;
    LOCALGROUP_INFO_1 Info1;
    LOCALGROUP_INFO_1002 Info1002;
    NET_API_STATUS Status;

    if (argc == 2)
    {
        Status = EnumerateLocalGroups();
        ConPrintf(StdOut, L"Status: %lu\n", Status);
        return 0;
    }
    else if (argc == 3)
    {
        Status = DisplayLocalGroup(argv[2]);
        ConPrintf(StdOut, L"Status: %lu\n", Status);
        return 0;
    }

    i = 2;
    if (argv[i][0] != L'/')
    {
        lpGroupName = argv[i];
        i++;
    }

    for (j = i; j < argc; j++)
    {
        if (argv[j][0] == L'/')
            break;

        dwMemberCount++;
    }

    ConPrintf(StdOut, L"Member count: %lu\n", dwMemberCount);

    if (dwMemberCount > 0)
    {
        lpMembers = RtlAllocateHeap(RtlGetProcessHeap(),
                                    HEAP_ZERO_MEMORY,
                                    dwMemberCount * sizeof(LPLOCALGROUP_MEMBERS_INFO_3));
        if (lpMembers == NULL)
            return 0;
    }

    j = 0;
    for (; i < argc; i++)
    {
        if (argv[i][0] == L'/')
            break;

        lpMembers[j].lgrmi3_domainandname = argv[i];
        j++;
    }

    for (; i < argc; i++)
    {
        if (_wcsicmp(argv[i], L"/help") == 0)
        {
            PrintMessageString(4381);
            ConPuts(StdOut, L"\n");
            PrintNetMessage(MSG_LOCALGROUP_SYNTAX);
            PrintNetMessage(MSG_LOCALGROUP_HELP);
            return 0;
        }
        else if (_wcsicmp(argv[i], L"/add") == 0)
        {
            bAdd = TRUE;
        }
        else if (_wcsicmp(argv[i], L"/delete") == 0)
        {
            bDelete = TRUE;
        }
        else if (_wcsnicmp(argv[i], L"/comment:", 9) == 0)
        {
            lpComment = &argv[i][9];
        }
        else if (_wcsicmp(argv[i], L"/domain") == 0)
        {
            ConPuts(StdErr, L"The /DOMAIN option is not supported yet.\n");
#if 0
            bDomain = TRUE;
#endif
        }
        else
        {
            PrintErrorMessage(3506/*, argv[i]*/);
            result = 1;
            goto done;
        }
    }

    if (lpGroupName == NULL)
    {
        result = 1;
        goto done;
    }

    if (bAdd && bDelete)
    {
        result = 1;
        goto done;
    }

#if 0
    ConPrintf(StdOut, L"Group:\n  %s\n", lpGroupName);

    if (lpMembers != NULL)
    {
        ConPuts(StdOut, L"\nMembers:\n");
        for (i = 0; i < dwMemberCount; i++)
            ConPrintf(StdOut, L"  %s\n", lpMembers[i].lgrmi3_domainandname);
    }

    if (lpComment != NULL)
    {
        ConPrintf(StdOut, L"\nComment:\n  %s\n", lpComment);
    }
#endif

    if (lpMembers == NULL)
    {
        if (!bAdd && !bDelete && lpComment != NULL)
        {
            /* Set group comment */
            Info1002.lgrpi1002_comment = lpComment;
            Status = NetLocalGroupSetInfo(NULL,
                                          lpGroupName,
                                          1002,
                                          (LPBYTE)&Info1002,
                                          NULL);
            ConPrintf(StdOut, L"Status: %lu\n", Status);
        }
        else if (bAdd && !bDelete)
        {
            /* Add the group */
            if (lpComment == NULL)
            {
                Info0.lgrpi0_name = lpGroupName;
            }
            else
            {
                Info1.lgrpi1_name = lpGroupName;
                Info1.lgrpi1_comment = lpComment;
            }

            Status = NetLocalGroupAdd(NULL,
                                      (lpComment == NULL) ? 0 : 1,
                                      (lpComment == NULL) ? (LPBYTE)&Info0 : (LPBYTE)&Info1,
                                      NULL);
            ConPrintf(StdOut, L"Status: %lu\n", Status);
        }
        else if (!bAdd && bDelete && lpComment == NULL)
        {
            /* Delete the group */
            Status = NetLocalGroupDel(NULL,
                                      lpGroupName);
            ConPrintf(StdOut, L"Status: %lu\n", Status);
        }
        else
        {
            result = 1;
        }
    }
    else
    {
        if (bAdd && !bDelete && lpComment == NULL)
        {
            /* Add group members */
            Status = NetLocalGroupAddMembers(NULL,
                                             lpGroupName,
                                             3,
                                             (LPBYTE)lpMembers,
                                             dwMemberCount);
            ConPrintf(StdOut, L"Status: %lu\n", Status);
        }
        else if (!bAdd && bDelete && lpComment == NULL)
        {
            /* Delete group members */
            Status = NetLocalGroupDelMembers(NULL,
                                             lpGroupName,
                                             3,
                                             (LPBYTE)lpMembers,
                                             dwMemberCount);
            ConPrintf(StdOut, L"Status: %lu\n", Status);
        }
        else
        {
            result = 1;
        }
    }

done:
    if (lpMembers != NULL)
        RtlFreeHeap(RtlGetProcessHeap(), 0, lpMembers);

    if (result != 0)
    {
        PrintMessageString(4381);
        ConPuts(StdOut, L"\n");
        PrintNetMessage(MSG_LOCALGROUP_SYNTAX);
    }

    return result;
}

/* EOF */
