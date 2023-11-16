/******************************************************************************
 *
 *  Copyright (c) 2016, The Linux Foundation. All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are
 *   met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of The Linux Foundation nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED "AS IS" AND ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
 * IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
  ******************************************************************************/
/******************************************************************************
 *
 *  Utility functions to help build and parse AAC Codec Information
 *
 ******************************************************************************/

#include "bt_target.h"
#include <string.h>
#include "a2d_api.h"
#include "a2d_int.h"
#include "a2d_aac.h"
#include "bt_utils.h"

#if (A2D_M24_INCLUDED == TRUE)
static UINT8 A2D_UINT32_BitsSet(UINT32 num);
/******************************************************************************
**
** Function         A2D_BldAacInfo
**
** Description      This function builds byte sequence for
**                  Aac Codec Capabilities.
** Input :           media_type:  Audio or MultiMedia.
**                  p_ie: AAC Codec Information Element
**
** Output :          p_result: codec info.
**
** Returns          A2D_SUCCESS if successful.
**                  Error otherwise.
******************************************************************************/
tA2D_STATUS A2D_BldAacInfo(UINT8 media_type, tA2D_AAC_CIE *p_ie, UINT8 *p_result)
{
    tA2D_STATUS status;
    if( p_ie == NULL || p_result == NULL ||
        (p_ie->object_type & ~A2D_AAC_IE_OBJ_TYPE_MSK) ||
        (p_ie->samp_freq & ~A2D_AAC_IE_SAMP_FREQ_MSK) ||
        (p_ie->channels & ~A2D_AAC_IE_CHANNELS_MSK) ||
        (p_ie->bit_rate & ~A2D_AAC_IE_BIT_RATE_MSK) ||
        (p_ie->vbr & ~A2D_AAC_IE_VBR_MSK) )
    {
        /* return invalid params if invalid bit is set */
        status = A2D_INVALID_PARAMS;
    }
    else
    {
        status = A2D_SUCCESS;
        *p_result++ = A2D_AAC_INFO_LEN;
        *p_result++ = media_type;
        *p_result++ = A2D_MEDIA_CT_M24;

        /* Codec information */
        *p_result++ = p_ie->object_type; // object type

        *p_result++ = (UINT8)(p_ie->samp_freq >> 8);

        *p_result++ = (p_ie->samp_freq & 0x00F0)|p_ie->channels;
        *p_result++ = p_ie->vbr | ((p_ie->bit_rate >> 16)& 0x007F);
        *p_result++ = (p_ie->bit_rate >> 8)& 0x00FF;
        *p_result   = p_ie->bit_rate & 0x000000FF;
    }
    return status;
}

/******************************************************************************
**
** Function         A2D_ParsAacInfo
**
** Description      This function parse byte sequence for
**                  Aac Codec Capabilities.
** Input :          p_info:  input byte sequence.
**                  for_caps: True for getcap, false otherwise
**
** Output :          p_ie: Aac codec information.
**
** Returns          A2D_SUCCESS if successful.
**                  Error otherwise.
******************************************************************************/
tA2D_STATUS A2D_ParsAacInfo(tA2D_AAC_CIE *p_ie, UINT8 *p_info, BOOLEAN for_caps)
{
    tA2D_STATUS status;
    UINT8   losc;
    UINT8   media_type;

    if( p_ie == NULL || p_info == NULL)
        status = A2D_INVALID_PARAMS;
    else
    {
        losc            = *p_info++;
        media_type      = *p_info++;
        /* Check for wrong length, media type */
        if(losc != A2D_AAC_INFO_LEN || *p_info != A2D_MEDIA_CT_M24)
            status = A2D_WRONG_CODEC;
        else
        {
            p_info++;
            /* obj type */
            p_ie->object_type = *p_info & A2D_AAC_IE_OBJ_TYPE_MSK; p_info++;
            /* samping freq */
            p_ie->samp_freq     = *p_info; p_info++;
            p_ie->samp_freq = p_ie->samp_freq << 8;
            p_ie->samp_freq |= (*p_info & 0xF0);
            /* channels */
            p_ie->channels = *p_info & A2D_AAC_IE_CHANNELS_MSK; p_info++;
            /* variable bit rate */
            p_ie->vbr =       *p_info & A2D_AAC_IE_VBR_MSK;
            /* bit rate */
            p_ie->bit_rate = *p_info & 0x7F;p_ie->bit_rate = p_ie->bit_rate << 8; p_info++;
            p_ie->bit_rate |= *p_info;p_ie->bit_rate = p_ie->bit_rate << 8; p_info++;
            p_ie->bit_rate |= *p_info;
            status = A2D_SUCCESS;

            if(for_caps == FALSE)
            {
                if(A2D_UINT32_BitsSet(p_ie->object_type) != A2D_SET_ONE_BIT)
                    status = A2D_BAD_OBJ_TYPE;
                if(A2D_UINT32_BitsSet(p_ie->samp_freq) != A2D_SET_ONE_BIT)
                    status = A2D_BAD_SAMP_FREQ;
                if(A2D_UINT32_BitsSet(p_ie->channels) != A2D_SET_ONE_BIT)
                    status = A2D_BAD_CHANNEL;
            }
        }
    }
    return status;
}
/******************************************************************************
** Function         A2D_UINT32_BitsSet
**
** Description      Check the given number of 32bit  for the number of bits set
** Returns          A2D_SET_ONE_BIT, if one and only one bit is set
**                  A2D_SET_ZERO_BIT, if all bits clear
**                  A2D_SET_MULTL_BIT, if multiple bits are set
******************************************************************************/
static UINT8 A2D_UINT32_BitsSet(UINT32 num)
{
    UINT8   count;
    BOOLEAN res;
    if(num == 0)
        res = A2D_SET_ZERO_BIT;
    else
    {
        count = (num & (num - 1));
        res = ((count==0)?A2D_SET_ONE_BIT:A2D_SET_MULTL_BIT);
    }
    return res;
}
#endif /* A2D_M24_INCLUDED == TRUE */
