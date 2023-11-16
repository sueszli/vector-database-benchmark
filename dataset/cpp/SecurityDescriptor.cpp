/*
 * regexpl - Console Registry Explorer
 *
 * Copyright (C) 2000-2005 Nedko Arnaudov <nedko@users.sourceforge.net>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

// SecurityDescriptor.cpp: implementation of the CSecurityDescriptor class.
//
//////////////////////////////////////////////////////////////////////

#include "ph.h"
#include "SecurityDescriptor.h"

BOOL GetTextualSid(
    PSID pSid,            // binary Sid
    LPTSTR TextualSid,    // buffer for Textual representation of Sid
    LPDWORD lpdwBufferLen // required/provided TextualSid buffersize
    )
{
    PSID_IDENTIFIER_AUTHORITY psia;
    DWORD dwSubAuthorities;
    DWORD dwSidRev=SID_REVISION;
    DWORD dwCounter;
    DWORD dwSidSize;

    // Validate the binary SID.

    if(!IsValidSid(pSid)) return FALSE;

    // Get the identifier authority value from the SID.

    psia = GetSidIdentifierAuthority(pSid);

    // Get the number of subauthorities in the SID.

    dwSubAuthorities = *GetSidSubAuthorityCount(pSid);

    // Compute the buffer length.
    // S-SID_REVISION- + IdentifierAuthority- + subauthorities- + NULL

    dwSidSize=(15 + 12 + (12 * dwSubAuthorities) + 1) * sizeof(TCHAR);

    // Check input buffer length.
    // If too small, indicate the proper size and set last error.

    if (*lpdwBufferLen < dwSidSize)
    {
        *lpdwBufferLen = dwSidSize;
        SetLastError(ERROR_INSUFFICIENT_BUFFER);
        return FALSE;
    }

    // Add 'S' prefix and revision number to the string.

    dwSidSize=wsprintf(TextualSid, TEXT("S-%lu-"), dwSidRev );

    // Add SID identifier authority to the string.

    if ( (psia->Value[0] != 0) || (psia->Value[1] != 0) )
    {
        dwSidSize+=wsprintf(TextualSid + lstrlen(TextualSid),
                    TEXT("0x%02hx%02hx%02hx%02hx%02hx%02hx"),
                    (USHORT)psia->Value[0],
                    (USHORT)psia->Value[1],
                    (USHORT)psia->Value[2],
                    (USHORT)psia->Value[3],
                    (USHORT)psia->Value[4],
                    (USHORT)psia->Value[5]);
    }
    else
    {
        dwSidSize+=wsprintf(TextualSid + lstrlen(TextualSid),
                    TEXT("%lu"),
                    (ULONG)(psia->Value[5]      )   +
                    (ULONG)(psia->Value[4] <<  8)   +
                    (ULONG)(psia->Value[3] << 16)   +
                    (ULONG)(psia->Value[2] << 24)   );
    }

    // Add SID subauthorities to the string.
    //
    for (dwCounter=0 ; dwCounter < dwSubAuthorities ; dwCounter++)
    {
        dwSidSize+=wsprintf(TextualSid + dwSidSize, TEXT("-%lu"),
                    *GetSidSubAuthority(pSid, dwCounter) );
    }

    return TRUE;
}

const TCHAR * GetSidTypeName(SID_NAME_USE Use)
{
	switch(Use)
	{
	case SidTypeUser:
		return _T("User SID");
	case SidTypeGroup:
		return _T("Group SID");
	case SidTypeDomain:
		return _T("Domain SID");
	case SidTypeAlias:
		return _T("Alias SID");
	case SidTypeWellKnownGroup:
		return _T("SID for a well-known group");
	case SidTypeDeletedAccount:
		return _T("SID for a deleted account");
	case SidTypeInvalid:
		return _T("Invalid SID");
	case SidTypeUnknown:
		return _T("Unknown SID type");
	default:
		return _T("Error. Cannot recognize SID type.");
	}
}

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

CSecurityDescriptor::CSecurityDescriptor()
{
	m_pSecurityDescriptor = NULL;
	m_pCurrentACEHeader = NULL;
}

CSecurityDescriptor::~CSecurityDescriptor()
{
}

void CSecurityDescriptor::AssociateDescriptor(PSECURITY_DESCRIPTOR pSecurityDescriptor)
{
	m_pSecurityDescriptor = pSecurityDescriptor;
}

DWORD CSecurityDescriptor::BeginDACLInteration()
{
	if (!GetSecurityDescriptorDacl(m_pSecurityDescriptor,&m_blnDACLPresent,&m_pDACL,&m_blnDACLDefaulted))
	{
		throw GetLastError();
	}
	return ERROR_SUCCESS;
}

BOOL CSecurityDescriptor::DescriptorContainsDACL()
{
	return m_blnDACLPresent;
}

DWORD CSecurityDescriptor::BeginSACLInteration()
{
	if (!GetSecurityDescriptorSacl(m_pSecurityDescriptor,&m_blnSACLPresent,&m_pSACL,&m_blnSACLDefaulted))
		throw GetLastError();
	return ERROR_SUCCESS;
}

BOOL CSecurityDescriptor::DescriptorContainsSACL()
{
	return m_blnSACLPresent;
}

BOOL CSecurityDescriptor::HasNULLDACL()
{
	ASSERT(m_blnDACLPresent);
	return (m_pDACL == NULL);
}

BOOL CSecurityDescriptor::HasValidDACL()
{
	ASSERT(m_blnDACLPresent);
	ASSERT(m_pDACL != NULL);
	return IsValidAcl(m_pDACL);
}

BOOL CSecurityDescriptor::HasNULLSACL()
{
	ASSERT(m_blnSACLPresent);
	return (m_pSACL == NULL);
}

BOOL CSecurityDescriptor::HasValidSACL()
{
	ASSERT(m_blnSACLPresent);
	ASSERT(m_pSACL != NULL);
	return IsValidAcl(m_pSACL);
}

DWORD CSecurityDescriptor::GetDACLEntriesCount()
{
	ACL_SIZE_INFORMATION SizeInfo;
	if (!GetAclInformation(m_pDACL,&SizeInfo,sizeof(SizeInfo),AclSizeInformation))
		throw GetLastError();
	return SizeInfo.AceCount;
}

DWORD CSecurityDescriptor::GetSACLEntriesCount()
{
	ACL_SIZE_INFORMATION SizeInfo;
	if (!GetAclInformation(m_pSACL,&SizeInfo,sizeof(SizeInfo),AclSizeInformation))
		throw GetLastError();
	return SizeInfo.AceCount;
}

CSecurityDescriptor::ACEntryType CSecurityDescriptor::GetDACLEntry(DWORD nIndex)
{
	void *pACE;
	if (!GetAce(m_pDACL,nIndex,&pACE)) throw GetLastError();
	m_pCurrentACEHeader = (PACE_HEADER)pACE;
	if (m_pCurrentACEHeader->AceType == ACCESS_ALLOWED_ACE_TYPE)
	{
		return AccessAlowed;
	}
	if (m_pCurrentACEHeader->AceType == ACCESS_DENIED_ACE_TYPE)
	{
		return AccessDenied;
	}
	return Unknown;
}

CSecurityDescriptor::ACEntryType CSecurityDescriptor::GetSACLEntry(DWORD nIndex, BOOL& blnFailedAccess, BOOL& blnSeccessfulAccess)
{
	void *pACE;
	if (!GetAce(m_pSACL,nIndex,&pACE)) throw GetLastError();
	m_pCurrentACEHeader = (PACE_HEADER)pACE;
	if (m_pCurrentACEHeader->AceType == SYSTEM_AUDIT_ACE_TYPE)
	{
		blnFailedAccess = m_pCurrentACEHeader->AceFlags & FAILED_ACCESS_ACE_FLAG;
		blnSeccessfulAccess = m_pCurrentACEHeader->AceFlags & SUCCESSFUL_ACCESS_ACE_FLAG;
		return SystemAudit;
	}
	return Unknown;
}

PSID CSecurityDescriptor::GetCurrentACE_SID()
{
	ASSERT(m_pCurrentACEHeader != NULL);
	switch(m_pCurrentACEHeader->AceType)
	{
	case ACCESS_ALLOWED_ACE_TYPE:
		return ((PSID)&(((ACCESS_ALLOWED_ACE *)m_pCurrentACEHeader)->SidStart));
	case ACCESS_DENIED_ACE_TYPE:
		return ((PSID)&(((ACCESS_DENIED_ACE *)m_pCurrentACEHeader)->SidStart));
	case SYSTEM_AUDIT_ACE_TYPE:
		return ((PSID)&(((SYSTEM_AUDIT_ACE *)m_pCurrentACEHeader)->SidStart));
	default:
		ASSERT(FALSE);	// Do not call this function for unknown ACE types !!!
		return NULL;
	}
}

void CSecurityDescriptor::GetCurrentACE_AccessMask(DWORD& dwMask)
{
	ASSERT(m_pCurrentACEHeader != NULL);
	switch(m_pCurrentACEHeader->AceType)
	{
	case ACCESS_ALLOWED_ACE_TYPE:
		dwMask = (((ACCESS_ALLOWED_ACE *)m_pCurrentACEHeader)->Mask);
		return;
	case ACCESS_DENIED_ACE_TYPE:
		dwMask = (((ACCESS_DENIED_ACE *)m_pCurrentACEHeader)->Mask);
		return;
	case SYSTEM_AUDIT_ACE_TYPE:
		dwMask = (((SYSTEM_AUDIT_ACE *)m_pCurrentACEHeader)->Mask);
		return;
	default:
		ASSERT(FALSE);	// Do not call this function for unknown ACE types !!!
		return;
	}
}

void CSecurityDescriptor::GetCurrentACE_Flags(BYTE& bFlags)
{
	ASSERT(m_pCurrentACEHeader != NULL);
  bFlags = m_pCurrentACEHeader->AceFlags;
}
