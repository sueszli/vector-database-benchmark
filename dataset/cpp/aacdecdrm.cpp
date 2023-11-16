/* ***** BEGIN LICENSE BLOCK *****  
 * Source last modified: $Id: aacdecdrm.cpp,v 1.2 2005/03/14 19:14:31 bobclark Exp $ 
 *   
 * Portions Copyright (c) 1995-2005 RealNetworks, Inc. All Rights Reserved.  
 *       
 * The contents of this file, and the files included with this file, 
 * are subject to the current version of the RealNetworks Public 
 * Source License (the "RPSL") available at 
 * http://www.helixcommunity.org/content/rpsl unless you have licensed 
 * the file under the current version of the RealNetworks Community 
 * Source License (the "RCSL") available at 
 * http://www.helixcommunity.org/content/rcsl, in which case the RCSL 
 * will apply. You may also obtain the license terms directly from 
 * RealNetworks.  You may not use this file except in compliance with 
 * the RPSL or, if you have a valid RCSL with RealNetworks applicable 
 * to this file, the RCSL.  Please see the applicable RPSL or RCSL for 
 * the rights, obligations and limitations governing use of the 
 * contents of the file. 
 *   
 * This file is part of the Helix DNA Technology. RealNetworks is the 
 * developer of the Original Code and owns the copyrights in the 
 * portions it created. 
 *   
 * This file, and the files included with this file, is distributed 
 * and made available on an 'AS IS' basis, WITHOUT WARRANTY OF ANY 
 * KIND, EITHER EXPRESS OR IMPLIED, AND REALNETWORKS HEREBY DISCLAIMS 
 * ALL SUCH WARRANTIES, INCLUDING WITHOUT LIMITATION, ANY WARRANTIES 
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, QUIET 
 * ENJOYMENT OR NON-INFRINGEMENT. 
 *  
 * Technology Compatibility Kit Test Suite(s) Location:  
 *    http://www.helixcommunity.org/content/tck  
 *  
 * Contributor(s):  
 *   
 * ***** END LICENSE BLOCK ***** */  

#include "hxtypes.h"
#include "hxresult.h"
#include "hxcom.h"
#include "hxassert.h"
#include "hxacodec.h"
#include "baseobj.h"
#include "racodec.h"
#include "aacdecdll.h"
#include "aacdecdrm.h"
#include "aacconstants.h"

CAACDecDRM::CAACDecDRM()
    : m_bSecure(FALSE)
{
}

CAACDecDRM::~CAACDecDRM()
{
}

STDMETHODIMP CAACDecDRM::QueryInterface(REFIID riid, void** ppvObj)
{
    return CAACDec::QueryInterface(riid, ppvObj);
}

STDMETHODIMP_(ULONG32) CAACDecDRM::AddRef()
{
    return InterlockedIncrement(&m_lRefCount);
}

STDMETHODIMP_(ULONG32) CAACDecDRM::Release()
{
    if (InterlockedDecrement(&m_lRefCount) > 0)
    {
        return m_lRefCount;
    }

    delete this;
    return 0;
}

STDMETHODIMP CAACDecDRM::Decode(const UCHAR* data, UINT32 nBytes, UINT32 &nBytesConsumed, INT16 *samplesOut, UINT32& nSamplesOut, HXBOOL eof)
{
    return CAACDec::Decode(data, nBytes, nBytesConsumed, samplesOut, nSamplesOut, eof);
}

STDMETHODIMP CAACDecDRM::GoSecure()
{
    return HXR_OK;
}

STDMETHODIMP_(HXBOOL) CAACDecDRM::IsSecure()
{
    return m_bSecure;
}

