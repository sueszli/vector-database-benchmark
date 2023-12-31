/* IBM_PROLOG_BEGIN_TAG                                                   */
/* This is an automatically generated prolog.                             */
/*                                                                        */
/* $Source: src/usr/fapi2/dimmBadDqBitmapFuncs.C $                        */
/*                                                                        */
/* OpenPOWER HostBoot Project                                             */
/*                                                                        */
/* Contributors Listed Below - COPYRIGHT 2017,2023                        */
/* [+] International Business Machines Corp.                              */
/*                                                                        */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/*                                                                        */
/* IBM_PROLOG_END_TAG                                                     */
#include <dimmBadDqBitmapFuncs.H>
#include <string.h>
#include <attribute_service.H>
#include <target.H>
#include <errl/errlmanager.H>
#include <lib/shared/exp_consts.H>

using namespace TARGETING;

extern "C"
{

using namespace mss;

//------------------------------------------------------------------------------
// Utility function to check parameters and get the Bad DQ bitmap
//------------------------------------------------------------------------------
fapi2::ReturnCode dimmBadDqCheckParamGetBitmap( const fapi2::Target
    <fapi2::TARGET_TYPE_MEM_PORT|fapi2::TARGET_TYPE_OCMB_CHIP>& i_fapiTrgt,
    const uint8_t i_port,
    const uint8_t i_dimm,
    const uint8_t i_rank,
    TARGETING::TargetHandle_t & o_dimmTrgt,
    uint8_t (&o_dqBitmap)[MAX_RANK_PER_DIMM][BAD_DQ_BYTE_COUNT])
{
    fapi2::ReturnCode l_rc;

    do
    {
        // Check parameters.
        if ( (i_dimm >= exp::sizes::MAX_DIMM_PER_PORT) ||
             (i_rank >= exp::sizes::MAX_RANK_PER_DIMM) )
        {
            FAPI_ERR( "dimmBadDqCheckParamGetBitmap: Bad parameter. "
                      "i_dimm:%d i_rank:%d", i_dimm, i_rank );
            l_rc = fapi2::FAPI2_RC_INVALID_ATTR_GET;
            break;
        }

        errlHndl_t l_errl = nullptr;
        TARGETING::TargetHandle_t l_trgt = nullptr;

        l_errl = fapi2::platAttrSvc::getTargetingTarget(i_fapiTrgt, l_trgt);
        if ( l_errl )
        {
            FAPI_ERR( "dimmBadDqCheckParamGetBitmap: Error from "
                      "getTargetingTarget" );
            // Add the error log pointer as data to the ReturnCode
            addErrlPtrToReturnCode(l_rc, l_errl);
            break;
        }

        // Get all functional DIMMs
        TargetHandleList l_dimmList;
        getChildAffinityTargets( l_dimmList, l_trgt, CLASS_NA, TYPE_DIMM );

        // Find the DIMM with the correct port/dimm slct
        uint8_t l_port = 0;
        uint8_t l_dimm = 0;

        for ( auto &dimmTrgt : l_dimmList )
        {
            // Get and compare the port
            // TODO: check dimm ATTR_MEM_PORT if support for it is added back
            TargetHandle_t memport = getAffinityParent(dimmTrgt, TYPE_MEM_PORT);
            l_port = memport->getAttr<ATTR_REL_POS>();

            if ( l_port == i_port )
            {
                // Get and compare the dimm
                l_dimm = dimmTrgt->getAttr<ATTR_POS_ON_MEM_PORT>();

                if ( l_dimm == i_dimm )
                {
                    o_dimmTrgt = dimmTrgt;
                    fapi2::Target<fapi2::TARGET_TYPE_DIMM> l_fapiDimm(dimmTrgt);
                    // Port and dimm are correct, get the Bad DQ bitmap
                    l_rc = FAPI_ATTR_GET( fapi2::ATTR_BAD_DQ_BITMAP,
                                          l_fapiDimm,
                                          o_dqBitmap );
                    if ( l_rc ) break;
                }
            }
        }

        if ( l_rc )
        {
            FAPI_ERR( "dimmBadDqCheckParamGetBitmap: Error getting "
                      "ATTR_BAD_DQ_BITMAP." );
        }

    }while(0);

    return l_rc;
}

//------------------------------------------------------------------------------
fapi2::ReturnCode p10DimmGetBadDqBitmap( const fapi2::Target
    <fapi2::TARGET_TYPE_MEM_PORT|fapi2::TARGET_TYPE_OCMB_CHIP>& i_fapiTrgt,
    const uint8_t i_dimm,
    const uint8_t i_rank,
    uint8_t (&o_data)[BAD_DQ_BYTE_COUNT],
    const uint8_t i_port)
{
    FAPI_INF( ">>p10DimmGetBadDqBitmap. %d:%d", i_dimm, i_rank );

    fapi2::ReturnCode l_rc;

    do
    {
        uint8_t l_dqBitmap[MAX_RANK_PER_DIMM][BAD_DQ_BYTE_COUNT];
        TARGETING::TargetHandle_t l_dimmTrgt = nullptr;

        // Check parameters and get Bad Dq Bitmap
        l_rc = dimmBadDqCheckParamGetBitmap( i_fapiTrgt, i_port, i_dimm, i_rank,
                                             l_dimmTrgt, l_dqBitmap );
        if ( l_rc )
        {
            FAPI_ERR( "p10DimmGetBadDqBitmap: Error from "
                      "dimmBadDqCheckParamGetBitmap." );
            break;
        }
        // Write contents of DQ bitmap for specific rank to o_data.
        memcpy( o_data, l_dqBitmap[i_rank], BAD_DQ_BYTE_COUNT );
    }while(0);

    FAPI_INF( "<<p10DimmGetBadDqBitmap" );

    return l_rc;
}

//------------------------------------------------------------------------------
fapi2::ReturnCode p10DimmSetBadDqBitmap( const fapi2::Target
    <fapi2::TARGET_TYPE_MEM_PORT|fapi2::TARGET_TYPE_OCMB_CHIP>& i_fapiTrgt,
    const uint8_t i_dimm,
    const uint8_t i_rank,
    const uint8_t (&i_data)[BAD_DQ_BYTE_COUNT],
    const uint8_t i_port)
{
    FAPI_INF( ">>p10DimmSetBadDqBitmap. %d:%d", i_dimm, i_rank );

    fapi2::ReturnCode l_rc;

    do
    {
        // Get the Bad DQ Bitmap by querying ATTR_BAD_DQ_BITMAP.
        uint8_t l_dqBitmap[MAX_RANK_PER_DIMM][BAD_DQ_BYTE_COUNT];
        TARGETING::TargetHandle_t l_dimmTrgt = nullptr;

        // Check parameters and get Bad Dq Bitmap
        l_rc = dimmBadDqCheckParamGetBitmap( i_fapiTrgt, i_port, i_dimm, i_rank,
                                             l_dimmTrgt, l_dqBitmap );
        if ( l_rc )
        {
            FAPI_ERR("p10DimmSetBadDqBitmap: Error getting ATTR_BAD_DQ_BITMAP.");
            break;
        }
        // Add the rank bitmap to the DIMM bitmap and write the bitmap.
        memcpy( l_dqBitmap[i_rank], i_data, BAD_DQ_BYTE_COUNT );

        fapi2::Target<fapi2::TARGET_TYPE_DIMM> l_fapiDimm( l_dimmTrgt );
        l_rc = FAPI_ATTR_SET( fapi2::ATTR_BAD_DQ_BITMAP, l_fapiDimm,
                              l_dqBitmap );

        if ( l_rc )
        {
            FAPI_ERR("p10DimmSetBadDqBitmap: Error setting ATTR_BAD_DQ_BITMAP.");
        }
    }while(0);

    FAPI_INF( "<<p10DimmSetBadDqBitmap" );

    return l_rc;
}

} // extern "C"
