/* IBM_PROLOG_BEGIN_TAG                                                   */
/* This is an automatically generated prolog.                             */
/*                                                                        */
/* $Source: src/usr/diag/prdf/plat/mem/prdfMemVcm_ipl.C $                 */
/*                                                                        */
/* OpenPOWER HostBoot Project                                             */
/*                                                                        */
/* Contributors Listed Below - COPYRIGHT 2016,2023                        */
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

/** @file prdfMemVcm_ipl.C */

// Platform includes
#include <prdfMemDqBitmap.H>
#include <prdfMemVcm.H>

using namespace TARGETING;

namespace PRDF
{

using namespace PlatServices;

//##############################################################################
//
//                          Generic template functions
//
//##############################################################################

template<TARGETING::TYPE T>
uint32_t VcmEvent<T>::falseAlarm( STEP_CODE_DATA_STRUCT & io_sc )
{
    #define PRDF_FUNC "[VcmEvent::falseAlarm] "

    uint32_t o_rc = SUCCESS;

    PRDF_TRAC( PRDF_FUNC "Chip mark false alarm: 0x%08x,0x%02x",
               iv_chip->getHuid(), getKey() );

    io_sc.service_data->setSignature( iv_chip->getHuid(),
                                      PRDFSIG_VcmFalseAlarm );

    do
    {
        // If DRAM repairs are disabled, make the error log predictive.
        if ( areDramRepairsDisabled() )
        {
            io_sc.service_data->setServiceCall();
            break; // Nothing more to do.
        }

        // Remove the chip mark.
        o_rc = MarkStore::clearChipMark<T>( iv_chip, iv_rank, iv_port );
        if ( SUCCESS != o_rc )
        {
            PRDF_ERR( PRDF_FUNC "clearChipMark(0x%08x,0x%02x,%x) failed",
                      iv_chip->getHuid(), getKey(), iv_port );
            break;
        }

    } while (0);

    return o_rc;

    #undef PRDF_FUNC
}

//------------------------------------------------------------------------------

template<TARGETING::TYPE T>
bool __iueCheck( uint32_t i_eccAttns );

template<> inline
bool __iueCheck<TYPE_OCMB_CHIP>( uint32_t i_eccAttns )
{
    return ( 0 != (i_eccAttns & MAINT_IUE) );
}

//------------------------------------------------------------------------------

template<TARGETING::TYPE T>
uint32_t VcmEvent<T>::checkEcc( const uint32_t & i_eccAttns,
                                STEP_CODE_DATA_STRUCT & io_sc,
                                bool & o_done )
{
    #define PRDF_FUNC "[VcmEvent<T>::checkEcc] "

    uint32_t o_rc = SUCCESS;

    do
    {
        // IUEs are reported as UEs during read operations. Therefore, we will
        // treat IUEs like UEs for these scrub operations simply to maintain
        // consistency during all of Memory Diagnostics.
        if ( (i_eccAttns & MAINT_UE) || __iueCheck<T>(i_eccAttns) )
        {
            PRDF_TRAC( PRDF_FUNC "UE Detected: 0x%08x,0x%02x",
                       iv_chip->getHuid(), getKey() );

            io_sc.service_data->setSignature( iv_chip->getHuid(),
                                              (i_eccAttns & MAINT_UE)
                                                           ? PRDFSIG_MaintUE
                                                           : PRDFSIG_MaintIUE );

            // At this point we don't actually have an address for the UE. The
            // best we can do is get the address in which the command stopped.
            MemAddr addr;
            o_rc = getMemMaintAddr<T>( iv_chip, addr );
            if ( SUCCESS != o_rc )
            {
                PRDF_ERR( PRDF_FUNC "getMemMaintAddr(0x%08x) failed",
                          iv_chip->getHuid() );
                break;
            }

            o_rc = MemEcc::handleMemUe<T>( iv_chip, addr, UE_TABLE::SCRUB_UE,
                                           io_sc );
            if ( SUCCESS != o_rc )
            {
                PRDF_ERR( PRDF_FUNC "handleMemUe(0x%08x,0x%02x) failed",
                          iv_chip->getHuid(), getKey() );
                break;
            }

            // Leave the mark in place and abort this procedure.
            o_done = true; break;
        }
        // For Odyssey, AUEs found during IPL/memdiags will be handled the same
        // as UEs. Pause on AUE will be set in the superfast reads for memdiags.
        else if ( 0 != (i_eccAttns & MAINT_AUE) )
        {
            // Add the signature to the multi-signature list. Also, since
            // this will be a predictive callout, change the primary
            // signature as well.
            io_sc.service_data->AddSignatureList( iv_chip->getTrgt(),
                                                  PRDFSIG_MaintAUE );
            io_sc.service_data->setSignature( iv_chip->getHuid(),
                                              PRDFSIG_MaintAUE );

            // At this point we don't actually have an address for the UE. The
            // best we can do is get the address in which the command stopped.
            MemAddr addr;
            o_rc = getMemMaintAddr<T>( iv_chip, addr );
            if ( SUCCESS != o_rc )
            {
                PRDF_ERR( PRDF_FUNC "getMemMaintAddr(0x%08x) failed",
                          iv_chip->getHuid() );
                break;
            }

            // Do memory UE handling.
            o_rc = MemEcc::handleMemUe<T>( iv_chip, addr, UE_TABLE::SCRUB_AUE,
                                           io_sc );
            if ( SUCCESS != o_rc )
            {
                PRDF_ERR( PRDF_FUNC "MAINT_AUE: handleMemUe<T>(0x%08x) failed",
                          iv_chip->getHuid() );
                break;
            }

            // Leave the mark in place and abort this procedure.
            o_done = true; break;
        }

    } while (0);

    return o_rc;

    #undef PRDF_FUNC
}

//------------------------------------------------------------------------------

// Avoid linker errors with the template.
template class VcmEvent<TYPE_OCMB_CHIP>;

} // end namespace PRDF

