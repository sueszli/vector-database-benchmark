// Copyright 2012 Nexenta Systems, Inc.  All rights reserved.
// Copyright (C) 2002 Microsoft Corporation
// All rights reserved.
//
// THIS CODE AND INFORMATION IS PROVIDED "AS IS"
// WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESSED
// OR IMPLIED, INCLUDING BUT NOT LIMITED
// TO THE IMPLIED WARRANTIES OF MERCHANTIBILITY
// AND/OR FITNESS FOR A PARTICULAR PURPOSE.
//
// Date    - 10/08/2002
// Author  - Sanj Surati

/////////////////////////////////////////////////////////////
//
// SPNEGO.C
//
// SPNEGO Token Handler Source File
//
// Contains implementation of SPNEGO Token Handling API
// as defined in SPNEGO.H.
//
/////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <memory.h>
#include "spnego.h"
#include "derparse.h"
#include "spnegoparse.h"

//
// Defined in DERPARSE.C
//

extern MECH_OID g_stcMechOIDList [];


/**********************************************************************/
/**                                                                  **/
/**                                                                  **/
/**                                                                  **/
/**                                                                  **/
/**               SPNEGO Token Handler API implementation            **/
/**                                                                  **/
/**                                                                  **/
/**                                                                  **/
/**                                                                  **/
/**********************************************************************/


/////////////////////////////////////////////////////////////////////////////
//
// Function:
//    spnegoInitFromBinary
//
// Parameters:
//    [in]  pbTokenData       -  Binary Token Data
//    [in]  ulLength          -  Length of binary Token Data
//    [out] phSpnegoToken     -  SPNEGO_TOKEN_HANDLE pointer
//
// Returns:
//    int   Success - SPNEGO_E_SUCCESS
//          Failure - SPNEGO API Error code
//
// Comments :
//    Initializes a SPNEGO_TOKEN_HANDLE from the supplied
//    binary data.  Data is copied locally.  Returned data structure
//    must be freed by calling spnegoFreeData().
//
////////////////////////////////////////////////////////////////////////////

int spnegoInitFromBinary( unsigned char* pbTokenData, unsigned long ulLength, SPNEGO_TOKEN_HANDLE* phSpnegoToken )
{
   int            nReturn = SPNEGO_E_INVALID_PARAMETER;
   SPNEGO_TOKEN** ppSpnegoToken = (SPNEGO_TOKEN**) phSpnegoToken;

   // Pass off to a handler function that allows tighter control over how the token structure
   // is handled.  In this case, we want the token data copied and we want the associated buffer
   // freed.
   nReturn = InitTokenFromBinary( SPNEGO_TOKEN_INTERNAL_COPYDATA,
                                 SPNEGO_TOKEN_INTERNAL_FLAGS_FREEDATA, pbTokenData,
                                 ulLength, ppSpnegoToken );

   return nReturn;
}

/////////////////////////////////////////////////////////////////////////////
//
// Function:
//    spnegoCreateNegTokenHint
//
// Parameters:
//    [in]  pMechTypeList     -  List of MechTypes (OIDs) to include
//    [in]  MechTypeCnt       -  Length of MechTypes array
//    [in]  pbPrincipal       -  Principal name for MechListMIC
//    [out] phSpnegoToken     -  SPNEGO_TOKEN_HANDLE pointer
//
// Returns:
//    int   Success - SPNEGO_E_SUCCESS
//          Failure - SPNEGO API Error code
//
// Comments :
//    Initializes a SPNEGO_TOKEN_HANDLE for a NegTokenInit type token
//    from the supplied parameters.  The token created is the "hint"
//    used (for example) in the response to an SMB negotiate protocol.
//    Returned data structure must be freed by calling spnegoFreeData().
//
//    The "hint" tells the client what authentication methods this
//    server supports (the ones in the MechTypeList).  The Principal
//    name historically was the server's own SPN, but recent versions
//    of windows only supply: "not_defined_in_RFC4178@please_ignore"
//    So if you want to be nice to your clients, provide the host SPN,
//    otherwise provide the bogus SPN string like recent windows.
//
////////////////////////////////////////////////////////////////////////////

int spnegoCreateNegTokenHint( SPNEGO_MECH_OID *pMechTypeList, int MechTypeCnt,
	unsigned char *pbPrincipal, SPNEGO_TOKEN_HANDLE* phSpnegoToken )
{
	int   nReturn;
	long  nTokenLength = 0L;
	long  nInternalTokenLength = 0L;
	unsigned long ulPrincipalLen;
	unsigned char* pbMechListMIC;
	unsigned long ulMechListMICLen;
	unsigned char* pbTokenData = NULL;
	SPNEGO_TOKEN** ppSpnegoToken = (SPNEGO_TOKEN**) phSpnegoToken;

	if ( NULL == ppSpnegoToken || NULL == pbPrincipal )
		return (SPNEGO_E_INVALID_PARAMETER);

	/*
	 * Get the actual token size
	 */
	ulPrincipalLen = strlen((char *)pbPrincipal);
	ulMechListMICLen = ASNDerCalcElementLength( ulPrincipalLen, NULL );
	nReturn = CalculateMinSpnegoInitTokenSize(
		0, /* ulMechTokenLen */
		ulMechListMICLen,
		pMechTypeList,
		MechTypeCnt,
		0, /* nReqFlagsAvailable */
		&nTokenLength,
		&nInternalTokenLength );
	if ( nReturn != SPNEGO_E_SUCCESS )
		return (nReturn);

	// Allocate a buffer to hold the data.
	pbTokenData = calloc( 1, nTokenLength );

	if ( NULL == pbTokenData )
		return ( SPNEGO_E_OUT_OF_MEMORY );

	/*
	 * Construct the MechListMIC
	 */
	pbMechListMIC = pbTokenData + (nTokenLength - ulMechListMICLen);
	(void) ASNDerWriteElement( pbMechListMIC, SPNEGO_NEGINIT_ELEMENT_MECHTYPES,
				   GENERALSTR, pbPrincipal, ulPrincipalLen );

	// Now write the token
	nReturn = CreateSpnegoInitToken(
		pMechTypeList,
		MechTypeCnt,
		0, /* ContextFlags */
		NULL, 0, /* MechToken, len */
		pbMechListMIC,
		ulMechListMICLen,
		pbTokenData,
		nTokenLength,
		nInternalTokenLength );
	if ( nReturn != SPNEGO_E_SUCCESS ) {
		free( pbTokenData );
		return (nReturn);
	}

	// This will copy our allocated pointer, and ensure that the sructure cleans
	// up the data later
	nReturn = InitTokenFromBinary( SPNEGO_TOKEN_INTERNAL_COPYPTR,
				       SPNEGO_TOKEN_INTERNAL_FLAGS_FREEDATA,
				       pbTokenData, nTokenLength, ppSpnegoToken );

	// Cleanup on failure
	if ( nReturn != SPNEGO_E_SUCCESS ) {
		free( pbTokenData );
		return (nReturn);
	}

	return (SPNEGO_E_SUCCESS);
}

/////////////////////////////////////////////////////////////////////////////
//
// Function:
//    spnegoCreateNegTokenInit
//
// Parameters:
//    [in]  MechType          -  MechType to specify in MechTypeList element
//    [in]  ucContextFlags    -  Context Flags element value
//    [in]  pbMechToken       -  Pointer to binary MechToken Data
//    [in]  ulMechTokenLen    -  Length of MechToken Data
//    [in]  pbMechListMIC     -  Pointer to binary MechListMIC Data
//    [in]  ulMechListMICLen  -  Length of MechListMIC Data
//    [out] phSpnegoToken     -  SPNEGO_TOKEN_HANDLE pointer
//
// Returns:
//    int   Success - SPNEGO_E_SUCCESS
//          Failure - SPNEGO API Error code
//
// Comments :
//    Initializes a SPNEGO_TOKEN_HANDLE for a NegTokenInit type
//    from the supplied parameters.  ucContextFlags may be 0 or must be
//    a valid flag combination.  MechToken data can be NULL - if not, it
//    must correspond to the MechType.  MechListMIC can also be NULL.
//    Returned data structure must be freed by calling spnegoFreeData().
//
////////////////////////////////////////////////////////////////////////////

int spnegoCreateNegTokenInit( SPNEGO_MECH_OID MechType,
          unsigned char ucContextFlags, unsigned char* pbMechToken,
          unsigned long ulMechTokenLen, unsigned char* pbMechListMIC,
          unsigned long ulMechListMICLen, SPNEGO_TOKEN_HANDLE* phSpnegoToken )
{
   int   nReturn = SPNEGO_E_INVALID_PARAMETER;
   long  nTokenLength = 0L;
   long  nInternalTokenLength = 0L;
   unsigned char* pbTokenData = NULL;
   SPNEGO_TOKEN** ppSpnegoToken = (SPNEGO_TOKEN**) phSpnegoToken;

   if ( NULL != ppSpnegoToken &&
         IsValidMechOid( MechType ) &&
         IsValidContextFlags( ucContextFlags ) )
   {
      // Get the actual token size

      if ( ( nReturn = CalculateMinSpnegoInitTokenSize( ulMechTokenLen, ulMechListMICLen,
							&MechType, 1, ( ucContextFlags != 0L ),
                                                         &nTokenLength, &nInternalTokenLength ) )
                        == SPNEGO_E_SUCCESS )
      {
         // Allocate a buffer to hold the data.
         pbTokenData = calloc( 1, nTokenLength );

         if ( NULL != pbTokenData )
         {

            // Now write the token
            if ( ( nReturn = CreateSpnegoInitToken( &MechType, 1,
                                                 ucContextFlags, pbMechToken,
                                                 ulMechTokenLen, pbMechListMIC,
                                                 ulMechListMICLen, pbTokenData,
                                                 nTokenLength, nInternalTokenLength ) )
                              == SPNEGO_E_SUCCESS )
            {

               // This will copy our allocated pointer, and ensure that the sructure cleans
               // up the data later
               nReturn = InitTokenFromBinary( SPNEGO_TOKEN_INTERNAL_COPYPTR,
                                             SPNEGO_TOKEN_INTERNAL_FLAGS_FREEDATA,
                                             pbTokenData, nTokenLength, ppSpnegoToken );

            }

            // Cleanup on failure
            if ( SPNEGO_E_SUCCESS != nReturn )
            {
               free( pbTokenData );
            }

         }  // IF alloc succeeded
         else
         {
            nReturn = SPNEGO_E_OUT_OF_MEMORY;
         }

      }  // If calculated token size

   }  // IF Valid Parameters

   return nReturn;
}

/////////////////////////////////////////////////////////////////////////////
//
// Function:
//    spnegoCreateNegTokenTarg
//
// Parameters:
//    [in]  MechType          -  MechType to specify in supported MechType element
//    [in]  spnegoNegResult   -  NegResult value
//    [in]  pbMechToken       -  Pointer to response MechToken Data
//    [in]  ulMechTokenLen    -  Length of MechToken Data
//    [in]  pbMechListMIC     -  Pointer to binary MechListMIC Data
//    [in]  ulMechListMICLen  -  Length of MechListMIC Data
//    [out] phSpnegoToken     -  SPNEGO_TOKEN_HANDLE pointer
//
// Returns:
//    int   Success - SPNEGO_E_SUCCESS
//          Failure - SPNEGO API Error code
//
// Comments :
//    Initializes a SPNEGO_TOKEN_HANDLE for a NegTokenTarg type
//    from the supplied parameters.  MechToken data can be NULL - if not,
//    it must correspond to the MechType.  MechListMIC can also be NULL.
//    Returned data structure must be freed by calling spnegoFreeData().
//
////////////////////////////////////////////////////////////////////////////

int spnegoCreateNegTokenTarg( SPNEGO_MECH_OID MechType,
          SPNEGO_NEGRESULT spnegoNegResult, unsigned char* pbMechToken,
          unsigned long ulMechTokenLen, unsigned char* pbMechListMIC,
          unsigned long ulMechListMICLen, SPNEGO_TOKEN_HANDLE* phSpnegoToken )
{
   int   nReturn = SPNEGO_E_INVALID_PARAMETER;
   long  nTokenLength = 0L;
   long  nInternalTokenLength = 0L;
   unsigned char* pbTokenData = NULL;
   SPNEGO_TOKEN** ppSpnegoToken = (SPNEGO_TOKEN**) phSpnegoToken;

   //
   // spnego_mech_oid_NotUsed and spnego_negresult_NotUsed
   // are okay here, however a valid MechOid is required
   // if spnego_negresult_success or spnego_negresult_incomplete
   // is specified.
   //

   if ( NULL != ppSpnegoToken &&

         ( IsValidMechOid( MechType ) ||
            spnego_mech_oid_NotUsed == MechType ) &&

         ( IsValidNegResult( spnegoNegResult ) ||
            spnego_negresult_NotUsed == spnegoNegResult ) )
   {

      // Get the actual token size

      if ( ( nReturn = CalculateMinSpnegoTargTokenSize( MechType, spnegoNegResult, ulMechTokenLen,
                                                         ulMechListMICLen, &nTokenLength,
                                                         &nInternalTokenLength ) )
                        == SPNEGO_E_SUCCESS )
      {
         // Allocate a buffer to hold the data.
         pbTokenData = calloc( 1, nTokenLength );

         if ( NULL != pbTokenData )
         {

            // Now write the token
            if ( ( nReturn = CreateSpnegoTargToken( MechType,
                                                 spnegoNegResult, pbMechToken,
                                                 ulMechTokenLen, pbMechListMIC,
                                                 ulMechListMICLen, pbTokenData,
                                                 nTokenLength, nInternalTokenLength ) )
                              == SPNEGO_E_SUCCESS )
            {

               // This will copy our allocated pointer, and ensure that the sructure cleans
               // up the data later
               nReturn = InitTokenFromBinary( SPNEGO_TOKEN_INTERNAL_COPYPTR,
                                             SPNEGO_TOKEN_INTERNAL_FLAGS_FREEDATA,
                                             pbTokenData, nTokenLength, ppSpnegoToken );

            }

            // Cleanup on failure
            if ( SPNEGO_E_SUCCESS != nReturn )
            {
               free( pbTokenData );
            }

         }  // IF alloc succeeded
         else
         {
            nReturn = SPNEGO_E_OUT_OF_MEMORY;
         }

      }  // If calculated token size

   }  // IF Valid Parameters

   return nReturn;
}

/////////////////////////////////////////////////////////////////////////////
//
// Function:
//    spnegoTokenGetBinary
//
// Parameters:
//    [in]     hSpnegoToken   -  Initialized SPNEGO_TOKEN_HANDLE
//    [out]    pbTokenData    -  Buffer to copy token into
//    [in/out] pulDataLen     -  Length of pbTokenData buffer, filled out
//                               with actual size used upon function return.
//
// Returns:
//    int   Success - SPNEGO_E_SUCCESS
//          Failure - SPNEGO API Error code
//
// Comments :
//    Copies binary SPNEGO token data from hSpnegoToken into the user
//    supplied buffer.  If pbTokenData is NULL, or the value in pulDataLen
//    is too small, the function will return SPNEGO_E_BUFFER_TOO_SMALL and
//    fill out pulDataLen with the minimum required buffer size.
//
////////////////////////////////////////////////////////////////////////////

int spnegoTokenGetBinary( SPNEGO_TOKEN_HANDLE hSpnegoToken, unsigned char* pbTokenData,
                           unsigned long * pulDataLen )
{
   int   nReturn = SPNEGO_E_INVALID_PARAMETER;
   SPNEGO_TOKEN*  pSpnegoToken = (SPNEGO_TOKEN*) hSpnegoToken;

   // Check parameters - pbTokenData is optional
   if (  IsValidSpnegoToken( pSpnegoToken ) &&
         NULL != pulDataLen )
   {

      // Check for Buffer too small conditions
      if ( NULL == pbTokenData ||
            pSpnegoToken->ulBinaryDataLen > *pulDataLen )
      {
         *pulDataLen = pSpnegoToken->ulBinaryDataLen;
         nReturn = SPNEGO_E_BUFFER_TOO_SMALL;
      }
      else
      {
         memcpy( pbTokenData, pSpnegoToken->pbBinaryData, pSpnegoToken->ulBinaryDataLen );
         *pulDataLen = pSpnegoToken->ulBinaryDataLen;
         nReturn = SPNEGO_E_SUCCESS;
      }

   }  // IF parameters OK

   return nReturn;;
}

/////////////////////////////////////////////////////////////////////////////
//
// Function:
//    spnegoFreeData
//
// Parameters:
//    [in]     hSpnegoToken   -  Initialized SPNEGO_TOKEN_HANDLE
//
// Returns:
//    void
//
// Comments :
//    Frees up resources consumed by hSpnegoToken.  The supplied data
//    pointer is invalidated by this function.
//
////////////////////////////////////////////////////////////////////////////

void spnegoFreeData( SPNEGO_TOKEN_HANDLE hSpnegoToken )
{
   FreeSpnegoToken( (SPNEGO_TOKEN*) hSpnegoToken);
   return;
}

/////////////////////////////////////////////////////////////////////////////
//
// Function:
//    spnegoGetTokenType
//
// Parameters:
//    [in]  hSpnegoToken      -  Initialized SPNEGO_TOKEN_HANDLE
//    [out] piTokenType       -  Filled out with token type value.
//
// Returns:
//    int   Success - SPNEGO_E_SUCCESS
//          Failure - SPNEGO API Error code
//
// Comments :
//    The function will analyze hSpnegoToken and return the appropriate
//    type in piTokenType.
//
////////////////////////////////////////////////////////////////////////////

int spnegoGetTokenType( SPNEGO_TOKEN_HANDLE hSpnegoToken, int * piTokenType )
{
   int   nReturn = SPNEGO_E_INVALID_PARAMETER;
   SPNEGO_TOKEN*  pSpnegoToken = (SPNEGO_TOKEN*) hSpnegoToken;

   // Check parameters
   if (  IsValidSpnegoToken( pSpnegoToken ) &&
         NULL != piTokenType &&
         pSpnegoToken)
   {

      // Check that the type in the structure makes sense
      if ( SPNEGO_TOKEN_INIT == pSpnegoToken->ucTokenType ||
            SPNEGO_TOKEN_TARG == pSpnegoToken->ucTokenType )
      {
         *piTokenType = pSpnegoToken->ucTokenType;
         nReturn = SPNEGO_E_SUCCESS;
      }

   }  // IF parameters OK

   return nReturn;
}

/////////////////////////////////////////////////////////////////////////////
//
// Function:
//    spnegoIsMechTypeAvailable
//
// Parameters:
//    [in]  hSpnegoToken      -  Initialized SPNEGO_TOKEN_HANDLE
//    [in]  MechOID           -  MechOID to search MechTypeList for
//    [out] piMechTypeIndex   -  Filled out with index in MechTypeList
//                               element if MechOID is found.
//
// Returns:
//    int   Success - SPNEGO_E_SUCCESS
//          Failure - SPNEGO API Error code
//
// Comments :
//    hSpnegoToken must reference a token of type NegTokenInit.  The
//    function will search the MechTypeList element for an OID corresponding
//    to the specified MechOID.  If one is found, the index (0 based) will
//    be passed into the piMechTypeIndex parameter.
//
////////////////////////////////////////////////////////////////////////////

// Returns the Initial Mech Type in the MechList element in the NegInitToken.
int spnegoIsMechTypeAvailable( SPNEGO_TOKEN_HANDLE hSpnegoToken, SPNEGO_MECH_OID MechOID, int * piMechTypeIndex )
{
   int   nReturn = SPNEGO_E_INVALID_PARAMETER;
   SPNEGO_TOKEN*  pSpnegoToken = (SPNEGO_TOKEN*) hSpnegoToken;

   // Check parameters
   if (  IsValidSpnegoToken( pSpnegoToken ) &&
         NULL != piMechTypeIndex &&
         IsValidMechOid( MechOID ) &&
         SPNEGO_TOKEN_INIT == pSpnegoToken->ucTokenType )
   {

      // Check if MechList is available
      if ( pSpnegoToken->aElementArray[SPNEGO_INIT_MECHTYPES_ELEMENT].iElementPresent
            == SPNEGO_TOKEN_ELEMENT_AVAILABLE )
      {
         // Locate the MechOID in the list element
         nReturn = FindMechOIDInMechList(
                     &pSpnegoToken->aElementArray[SPNEGO_INIT_MECHTYPES_ELEMENT],
                     MechOID, piMechTypeIndex );
      }
      else
      {
         nReturn = SPNEGO_E_ELEMENT_UNAVAILABLE;
      }

   }  // IF parameters OK

   return nReturn;;
}

/////////////////////////////////////////////////////////////////////////////
//
// Function:
//    spnegoGetContextFlags
//
// Parameters:
//    [in]  hSpnegoToken      -  Initialized SPNEGO_TOKEN_HANDLE
//    [out] pucContextFlags   -  Filled out with ContextFlags value.
//
// Returns:
//    int   Success - SPNEGO_E_SUCCESS
//          Failure - SPNEGO API Error code
//
// Comments :
//    hSpnegoToken must reference a token of type NegTokenInit.  The
//    function will copy data from the ContextFlags element into the
//    location pucContextFlags points to.  Note that the function will
//    fail if the actual ContextFlags data appears invalid.
//
////////////////////////////////////////////////////////////////////////////

int spnegoGetContextFlags( SPNEGO_TOKEN_HANDLE hSpnegoToken, unsigned char* pucContextFlags )
{
   int   nReturn = SPNEGO_E_INVALID_PARAMETER;
   SPNEGO_TOKEN*  pSpnegoToken = (SPNEGO_TOKEN*) hSpnegoToken;

   // Check parameters
   if (  IsValidSpnegoToken( pSpnegoToken ) &&
         NULL != pucContextFlags &&
         SPNEGO_TOKEN_INIT == pSpnegoToken->ucTokenType )
   {

      // Check if ContextFlags is available
      if ( pSpnegoToken->aElementArray[SPNEGO_INIT_REQFLAGS_ELEMENT].iElementPresent
            == SPNEGO_TOKEN_ELEMENT_AVAILABLE )
      {
         // The length should be two, the value should show a 1 bit difference in the difference byte, and
         // the value must be valid
         if ( pSpnegoToken->aElementArray[SPNEGO_INIT_REQFLAGS_ELEMENT].nDatalength == SPNEGO_NEGINIT_MAXLEN_REQFLAGS &&
               pSpnegoToken->aElementArray[SPNEGO_INIT_REQFLAGS_ELEMENT].pbData[0] == SPNEGO_NEGINIT_REQFLAGS_BITDIFF &&
               IsValidContextFlags( pSpnegoToken->aElementArray[SPNEGO_INIT_REQFLAGS_ELEMENT].pbData[1] ) )
         {
            *pucContextFlags = pSpnegoToken->aElementArray[SPNEGO_INIT_REQFLAGS_ELEMENT].pbData[1];
            nReturn = SPNEGO_E_SUCCESS;
         }
         else
         {
            nReturn = SPNEGO_E_INVALID_ELEMENT;
         }

      }
      else
      {
         nReturn = SPNEGO_E_ELEMENT_UNAVAILABLE;
      }

   }  // IF parameters OK

   return nReturn;;
}

/////////////////////////////////////////////////////////////////////////////
//
// Function:
//    spnegoGetNegotiationResult
//
// Parameters:
//    [in]  hSpnegoToken      -  Initialized SPNEGO_TOKEN_HANDLE
//    [out] pnegResult        -  Filled out with NegResult value.
//
// Returns:
//    int   Success - SPNEGO_E_SUCCESS
//          Failure - SPNEGO API Error code
//
// Comments :
//    hSpnegoToken must reference a token of type NegTokenTarg.  The
//    function will copy data from the NegResult element into the
//    location pointed to by pnegResult.  Note that the function will
//    fail if the actual NegResult data appears invalid.
//
////////////////////////////////////////////////////////////////////////////

int spnegoGetNegotiationResult( SPNEGO_TOKEN_HANDLE hSpnegoToken, SPNEGO_NEGRESULT* pnegResult )
{
   int   nReturn = SPNEGO_E_INVALID_PARAMETER;
   SPNEGO_TOKEN*  pSpnegoToken = (SPNEGO_TOKEN*) hSpnegoToken;

   // Check parameters
   if (  IsValidSpnegoToken( pSpnegoToken ) &&
         NULL != pnegResult &&
         SPNEGO_TOKEN_TARG == pSpnegoToken->ucTokenType )
   {

      // Check if NegResult is available
      if ( pSpnegoToken->aElementArray[SPNEGO_TARG_NEGRESULT_ELEMENT].iElementPresent
            == SPNEGO_TOKEN_ELEMENT_AVAILABLE )
      {
         // Must be 1 byte long and a valid value
         if ( pSpnegoToken->aElementArray[SPNEGO_TARG_NEGRESULT_ELEMENT].nDatalength == SPNEGO_NEGTARG_MAXLEN_NEGRESULT &&
               IsValidNegResult( *pSpnegoToken->aElementArray[SPNEGO_TARG_NEGRESULT_ELEMENT].pbData ) )
         {
            *pnegResult = *pSpnegoToken->aElementArray[SPNEGO_TARG_NEGRESULT_ELEMENT].pbData;
            nReturn = SPNEGO_E_SUCCESS;
         }
         else
         {
            nReturn = SPNEGO_E_INVALID_ELEMENT;
         }
      }
      else
      {
         nReturn = SPNEGO_E_ELEMENT_UNAVAILABLE;
      }

   }  // IF parameters OK

   return nReturn;;
}

/////////////////////////////////////////////////////////////////////////////
//
// Function:
//    spnegoGetSupportedMechType
//
// Parameters:
//    [in]  hSpnegoToken      -  Initialized SPNEGO_TOKEN_HANDLE
//    [out] pMechOID          -  Filled out with Supported MechType value.
//
// Returns:
//    int   Success - SPNEGO_E_SUCCESS
//          Failure - SPNEGO API Error code
//
// Comments :
//    hSpnegoToken must reference a token of type NegTokenTarg.  The
//    function will check the Supported MechType element, and if it
//    corresponds to a supported MechType ( spnego_mech_oid_Kerberos_V5_Legacy
//    or spnego_mech_oid_Kerberos_V5 ), will set the location pointed
//    to by pMechOID equal to the appropriate value.
//
////////////////////////////////////////////////////////////////////////////

int spnegoGetSupportedMechType( SPNEGO_TOKEN_HANDLE hSpnegoToken, SPNEGO_MECH_OID* pMechOID  )
{
   int   nReturn = SPNEGO_E_INVALID_PARAMETER;
   int   nCtr = 0L;
   long  nLength = 0L;
   SPNEGO_TOKEN*  pSpnegoToken = (SPNEGO_TOKEN*) hSpnegoToken;

   // Check parameters
   if (  IsValidSpnegoToken( pSpnegoToken ) &&
         NULL != pMechOID &&
         SPNEGO_TOKEN_TARG == pSpnegoToken->ucTokenType )
   {

      // Check if MechList is available
      if ( pSpnegoToken->aElementArray[SPNEGO_TARG_SUPPMECH_ELEMENT].iElementPresent
            == SPNEGO_TOKEN_ELEMENT_AVAILABLE )
      {

         for ( nCtr = 0;
               nReturn != SPNEGO_E_SUCCESS &&
               g_stcMechOIDList[nCtr].eMechanismOID != spnego_mech_oid_NotUsed;
               nCtr++ )
         {

            if ( ( nReturn = ASNDerCheckOID(
                        pSpnegoToken->aElementArray[SPNEGO_TARG_SUPPMECH_ELEMENT].pbData,
                        nCtr,
                        pSpnegoToken->aElementArray[SPNEGO_TARG_SUPPMECH_ELEMENT].nDatalength,
                        &nLength ) ) == SPNEGO_E_SUCCESS )
            {
               *pMechOID = nCtr;
            }

         }  // For enum MechOIDs


      }
      else
      {
         nReturn = SPNEGO_E_ELEMENT_UNAVAILABLE;
      }

   }  // IF parameters OK

   return nReturn;;
}

/////////////////////////////////////////////////////////////////////////////
//
// Function:
//    spnegoTokenGetMechToken
//
// Parameters:
//    [in]     hSpnegoToken   -  Initialized SPNEGO_TOKEN_HANDLE
//    [out]    pbTokenData    -  Buffer to copy MechToken into
//    [in/out] pulDataLen     -  Length of pbTokenData buffer, filled out
//                               with actual size used upon function return.
//
// Returns:
//    int   Success - SPNEGO_E_SUCCESS
//          Failure - SPNEGO API Error code
//
// Comments :
//    hSpnegoToken can point to either NegTokenInit or a NegTokenTarg token.
//    The function will copy the MechToken (the initial MechToken if
//    NegTokenInit, the response MechToken if NegTokenTarg) from the
//    underlying token into the buffer pointed to by pbTokenData.  If
//    pbTokenData is NULL, or the value in pulDataLen is too small, the
//    function will return SPNEGO_E_BUFFER_TOO_SMALL and fill out pulDataLen
//    with the minimum required buffer size.  The token can then be passed
//    to a GSS-API function for processing.
//
////////////////////////////////////////////////////////////////////////////

int spnegoGetMechToken( SPNEGO_TOKEN_HANDLE hSpnegoToken, unsigned char* pbTokenData, unsigned long* pulDataLen )
{
   int   nReturn = SPNEGO_E_INVALID_PARAMETER;
   SPNEGO_TOKEN*  pSpnegoToken = (SPNEGO_TOKEN*) hSpnegoToken;
   SPNEGO_ELEMENT*   pSpnegoElement = NULL;

   // Check parameters
   if (  IsValidSpnegoToken( pSpnegoToken ) &&
         NULL != pulDataLen )
   {

      // Point at the proper Element
      if ( SPNEGO_TOKEN_INIT == pSpnegoToken->ucTokenType )
      {
         pSpnegoElement = &pSpnegoToken->aElementArray[SPNEGO_INIT_MECHTOKEN_ELEMENT];
      }
      else
      {
         pSpnegoElement = &pSpnegoToken->aElementArray[SPNEGO_TARG_RESPTOKEN_ELEMENT];
      }

      // Check if MechType is available
      if ( SPNEGO_TOKEN_ELEMENT_AVAILABLE == pSpnegoElement->iElementPresent  )
      {
         // Check for Buffer too small conditions
         if ( NULL == pbTokenData ||
               pSpnegoElement->nDatalength > *pulDataLen )
         {
            *pulDataLen = pSpnegoElement->nDatalength;
            nReturn = SPNEGO_E_BUFFER_TOO_SMALL;
         }
         else
         {
            // Copy Memory
            memcpy( pbTokenData, pSpnegoElement->pbData, pSpnegoElement->nDatalength );
            *pulDataLen = pSpnegoElement->nDatalength;
            nReturn = SPNEGO_E_SUCCESS;
         }
      }
      else
      {
         nReturn = SPNEGO_E_ELEMENT_UNAVAILABLE;
      }

   }  // IF parameters OK

   return nReturn;;
}

/////////////////////////////////////////////////////////////////////////////
//
// Function:
//    spnegoTokenGetMechListMIC
//
// Parameters:
//    [in]     hSpnegoToken   -  Initialized SPNEGO_TOKEN_HANDLE
//    [out]    pbTokenData    -  Buffer to copy MechListMIC data into
//    [in/out] pulDataLen     -  Length of pbTokenData buffer, filled out
//                               with actual size used upon function return.
//
// Returns:
//    int   Success - SPNEGO_E_SUCCESS
//          Failure - SPNEGO API Error code
//
// Comments :
//    hSpnegoToken can point to either NegTokenInit or a NegTokenTarg token.
//    The function will copy the MechListMIC data from the underlying token
//    into the buffer pointed to by pbTokenData.  If pbTokenData is NULL,
//    or the value in pulDataLen is too small, the function will return
//    SPNEGO_E_BUFFER_TOO_SMALL and fill out pulDataLen with the minimum
//    required buffer size.
//
////////////////////////////////////////////////////////////////////////////

int spnegoGetMechListMIC( SPNEGO_TOKEN_HANDLE hSpnegoToken, unsigned char* pbMICData, unsigned long* pulDataLen )
{
   int   nReturn = SPNEGO_E_INVALID_PARAMETER;
   SPNEGO_TOKEN*  pSpnegoToken = (SPNEGO_TOKEN*) hSpnegoToken;
   SPNEGO_ELEMENT*   pSpnegoElement = NULL;

   // Check parameters
   if (  IsValidSpnegoToken( pSpnegoToken ) &&
         NULL != pulDataLen )
   {

      // Point at the proper Element
      if ( SPNEGO_TOKEN_INIT == pSpnegoToken->ucTokenType )
      {
         pSpnegoElement = &pSpnegoToken->aElementArray[SPNEGO_INIT_MECHLISTMIC_ELEMENT];
      }
      else
      {
         pSpnegoElement = &pSpnegoToken->aElementArray[SPNEGO_TARG_MECHLISTMIC_ELEMENT];
      }

      // Check if MechType is available
      if ( SPNEGO_TOKEN_ELEMENT_AVAILABLE == pSpnegoElement->iElementPresent  )
      {
         // Check for Buffer too small conditions
         if ( NULL == pbMICData ||
               pSpnegoElement->nDatalength > *pulDataLen )
         {
            *pulDataLen = pSpnegoElement->nDatalength;
            nReturn = SPNEGO_E_BUFFER_TOO_SMALL;
         }
         else
         {
            // Copy Memory
            memcpy( pbMICData, pSpnegoElement->pbData, pSpnegoElement->nDatalength );
            *pulDataLen = pSpnegoElement->nDatalength;
            nReturn = SPNEGO_E_SUCCESS;
         }
      }
      else
      {
         nReturn = SPNEGO_E_ELEMENT_UNAVAILABLE;
      }

   }  // IF parameters OK

   return nReturn;;
}

