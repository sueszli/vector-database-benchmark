/* IBM_PROLOG_BEGIN_TAG                                                   */
/* This is an automatically generated prolog.                             */
/*                                                                        */
/* $Source: src/import/chips/p10/procedures/hwp/corecache/p10_hcd_powerbus_purge.C $ */
/*                                                                        */
/* OpenPOWER HostBoot Project                                             */
/*                                                                        */
/* Contributors Listed Below - COPYRIGHT 2018,2021                        */
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



///
/// @file  p10_hcd_powerbus_purge.C
/// @brief Purge the PowerBus interface
///


// *HWP HWP Owner          : David Du         <daviddu@us.ibm.com>
// *HWP Backup HWP Owner   : Greg Still       <stillgs@us.ibm.com>
// *HWP FW Owner           : Prem Shanker Jha <premjha2@in.ibm.com>
// *HWP Team               : PM
// *HWP Consumed by        : SBE:QME
// *HWP Level              : 2
///
// EKB-Mirror-To: hw/ppe


//------------------------------------------------------------------------------
// Includes
//------------------------------------------------------------------------------

#include "p10_hcd_powerbus_purge.H"
#include "p10_hcd_common.H"

#ifdef __PPE_QME
    #include "p10_ppe_c.H"
    using namespace scomt::ppe_c;
#else
    #include "p10_scom_c.H"
    using namespace scomt::c;
#endif


//------------------------------------------------------------------------------
// Constant Definitions
//------------------------------------------------------------------------------

enum P10_HCD_POWERBUS_PURGE_CONSTANTS
{
    HCD_POWERBUS_PURGE_DONE_POLL_TIMEOUT_HW_NS    = 100000, // 10^5ns = 100us timeout
    HCD_POWERBUS_PURGE_DONE_POLL_DELAY_HW_NS      = 1000,   // 1us poll loop delay
    HCD_POWERBUS_PURGE_DONE_POLL_DELAY_SIM_CYCLE  = 32000,  // 32k sim cycle delay
    HCD_POWERBUS_INTERFACE_QUIESCE_DELAY_HW_NS      = 10,   // 10ns quiesce delay
    HCD_POWERBUS_INTERFACE_QUIESCE_DELAY_SIM_CYCLE  = 64,   // 64 sim cycle delay
};

//------------------------------------------------------------------------------
// Procedure: p10_hcd_powerbus_purge
//------------------------------------------------------------------------------

fapi2::ReturnCode
p10_hcd_powerbus_purge(
    const fapi2::Target < fapi2::TARGET_TYPE_CORE | fapi2::TARGET_TYPE_MULTICAST, fapi2::MULTICAST_AND > & i_target)
{
    fapi2::buffer<uint64_t> l_scomData = 0;
    fapi2::buffer<buffer_t> l_mmioData = 0;
    uint32_t                l_timeout  = 0;
    uint32_t                l_powerbus_purge_done = 0;

    FAPI_INF(">>p10_hcd_powerbus_purge");

    FAPI_DBG("Assert PB_PURGE_REQ via PCR_SCSR[12]");
    FAPI_TRY( HCD_PUTMMIO_C( i_target, QME_SCSR_WO_OR, MMIO_LOAD32H( BIT32(12) ) ) );

    FAPI_DBG("Wait for PB_PURGE_DONE via PCR_SCSR[44]");
    l_timeout = HCD_POWERBUS_PURGE_DONE_POLL_TIMEOUT_HW_NS /
                HCD_POWERBUS_PURGE_DONE_POLL_DELAY_HW_NS;

    do
    {

        FAPI_TRY( HCD_GETMMIO_C( i_target, MMIO_LOWADDR(QME_SCSR), l_mmioData ) );

        // use multicastAND to check 1
        MMIO_GET32L(l_powerbus_purge_done);

        if( ( l_powerbus_purge_done & BIT64SH(44) ) == BIT64SH(44) )
        {
            break;
        }

        fapi2::delay(HCD_POWERBUS_PURGE_DONE_POLL_DELAY_HW_NS,
                     HCD_POWERBUS_PURGE_DONE_POLL_DELAY_SIM_CYCLE);
    }
    while( (--l_timeout) != 0 );

    HCD_ASSERT4((l_timeout != 0),
                POWERBUS_PURGE_DONE_TIMEOUT,
                set_POWERBUS_PURGE_DONE_POLL_TIMEOUT_HW_NS, HCD_POWERBUS_PURGE_DONE_POLL_TIMEOUT_HW_NS,
                set_QME_SCSR, l_mmioData,
                set_MC_CORE_TARGET, i_target,
                set_CORE_SELECT, i_target.getCoreSelect(),
                "ERROR: PowerBus Purge Done Timeout");

    // Note: Do not drop Powerbus Purge until L3 becomes available again.

    FAPI_DBG("Assert NCU_PM_RCMD_DIS_CFG via NCU_RCMD_QUIESCE_REG[0]");
    FAPI_TRY( HCD_PUTSCOM_C( i_target, 0x20010658, SCOM_1BIT(0) ) );

    FAPI_DBG("Assert L3_PM_RCMD_DIS_CFG via PM_LCO_DIS_REG[1]");
    // This register doesnt have OR/CLR interface and having two functional bits
    // [0] L3_PM_LCO_DIS_CFG
    // [1] L3_PM_RCMD_DIS_CFG
    // Here set both bit0 and bit1 as bit0 would be previously set before l3 purge
    FAPI_TRY( HCD_PUTSCOM_C( i_target, 0x20010616, SCOM_LOAD32H( BITS32(0, 2) ) ) );

    FAPI_DBG("Wait ~20 Cache clocks");
    fapi2::delay(HCD_POWERBUS_INTERFACE_QUIESCE_DELAY_HW_NS,
                 HCD_POWERBUS_INTERFACE_QUIESCE_DELAY_SIM_CYCLE);

fapi_try_exit:

    FAPI_INF("<<p10_hcd_powerbus_purge");

    return fapi2::current_err;

}
