//
// Copyright (C) 2007 Pingtel Corp., certain elements licensed under a Contributor Agreement.
// Contributors retain copyright to elements licensed under a Contributor Agreement.
// Licensed to the User under the LGPL license.
//
// $$
////////////////////////////////////////////////////////////////////////
//////

#ifdef HAVE_GIPS /* [ */

// APPLICATION INCLUDES
#include "mp/MpConnection.h"
#include "mp/MpdGIPSPCMU.h"
#include "mp/GIPS/GIPS_API.h"
#include "mp/MprDejitter.h"
const MpCodecInfo MpdGIPSPCMU::smCodecInfo(
         SdpCodec::SDP_CODEC_GIPS_PCMU, GIPS_API_VERSION, true,
         8000, 8, 1, 160, 64000, 1280, 1280, 1280, 160);

#ifdef DEBUG_SHORT_PACKET_CRASH /* [ */
static int packetsIn = 0;
#endif /* DEBUG_SHORT_PACKET_CRASH ] */

MpdGIPSPCMU::MpdGIPSPCMU(int payloadType)
   : MpDecoderBase(payloadType, &smCodecInfo)
{
   // osPrintf("MpdGIPSPCMU::MpdGIPSPCMU(%d)\n", payloadType);
#ifdef DEBUG_SHORT_PACKET_CRASH /* [ */
   packetsIn = 0;
#endif /* DEBUG_SHORT_PACKET_CRASH ] */
}

MpdGIPSPCMU::~MpdGIPSPCMU()
{
   freeDecode();
}

OsStatus MpdGIPSPCMU::initDecode(MpConnection* pConnection)
{
   //Get NetEq pointer
   mpJBState = pConnection->getJBinst();

   // Set the payload number for NetEq
   NETEQ_GIPS_10MS16B_initCodepoint(mpJBState, const_cast <char*> ("PCMU"),
                                    8000, getPayloadType());

   // osPrintf("MpdGIPSPCMU::initDecode: payloadType=%d\n", getPayloadType());
   return OS_SUCCESS;
}

OsStatus MpdGIPSPCMU::freeDecode(void)
{
   return OS_SUCCESS;
}

#ifdef OLD_GIPS /* [ */
/************************************************************************
 * Check for odd length packets (not a multiple of 10 milliseconds).
 * It seems that our ancient version of NetEQ causes a program exception
 * if the payload is not a multiple of 80 samples.
 */
static int invalidLen(int l, unsigned char* p)
{
   /* l is #bytes in packet, including header.  The basic RTP header is 12
    * bytes long, with an additional 4 bytes for each CSRC ID; the CSRC
    * count is the low 4 bits of the first byte of the header
    */

   l = l - (12 + (((*p) & 0xf) * sizeof(int)));

   /* Most packets are 20 milliseconds (160 samples), so avoid doing the
    * expensive division in that case:
    */
   return (160 == l) ? 0 : (l % 80);
}

int MpdGIPSPCMU::decodeIn(MpBufPtr pPacket)
{
   int thisLen;
   unsigned char* pHeader;

   thisLen = MpBuf_getContentLen(pPacket);
   pHeader = (unsigned char*)MpBuf_getStorage(pPacket);

   if (invalidLen(thisLen, pHeader)) {

    /* DISCARD AN ODD LENGTH PACKET */

      // osPrintf("MpdGIPSPCMU::decodeIn - BAD packet, SEQ#%d,"
      // " payload is %d bytes\n", MprDejitter::getSeqNum(pPacket), thisLen);
      return OS_SUCCESS; /* (nobody cares, don't rock the boat!) */
      thisLen = 0;
   }
   return thisLen;
}
#endif /* OLD_GIPS ] */
#endif /* HAVE_GIPS ] */
