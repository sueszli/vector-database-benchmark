/* IBM_PROLOG_BEGIN_TAG                                                   */
/* This is an automatically generated prolog.                             */
/*                                                                        */
/* $Source: src/import/chips/ocmb/explorer/procedures/hwp/memory/lib/dimm/ddr4/exp_mrs06.C $ */
/*                                                                        */
/* OpenPOWER HostBoot Project                                             */
/*                                                                        */
/* Contributors Listed Below - COPYRIGHT 2020,2023                        */
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
/// @file exp_mrs06.C
/// @brief Run and manage the DDR4 MRS06 loading
///
// *HWP HWP Owner: Sneha Kadam <Sneha.Kadam1@ibm.com>
// *HWP HWP Backup: Louis Stermole <stermole@us.ibm.com>
// *HWP Team: Memory
// *HWP Level: 3
// *HWP Consumed by: FSP:HB

#include <fapi2.H>
#include <lib/shared/exp_consts.H>
#include <lib/ecc/ecc_traits_explorer.H>
#include <lib/dimm/exp_mrs_traits.H>
#include <lib/ccs/ccs_traits_explorer.H>
#include <generic/memory/lib/dimm/ddr4/mrs_load_ddr4.H>
#include <generic/memory/lib/dimm/ddr4/mrs06.H>

namespace mss
{

namespace ddr4
{

///
/// @brief mrs0_data ctor
/// @param[in] a fapi2::TARGET_TYPE_DIMM target
/// @param[out] fapi2::ReturnCode FAPI2_RC_SUCCESS iff ok
/// @note Burst Length will always be set to fixed x8 (0)
/// @note Burst Chop (x4) is not supported
///
template<>
mrs06_data<mss::mc_type::EXPLORER>::mrs06_data( const fapi2::Target<fapi2::TARGET_TYPE_DIMM>& i_target,
        fapi2::ReturnCode& o_rc ):
    iv_tccd_l(0)
{
    const auto l_port_target = mss::find_target<fapi2::TARGET_TYPE_MEM_PORT>(i_target);
    uint8_t l_vrefdq_train_value[mss::exp::MAX_RANK_PER_DIMM][mss::exp::MAX_NIBBLES_PER_PORT] = {0};
    uint8_t l_vrefdq_train_range[mss::exp::MAX_RANK_PER_DIMM][mss::exp::MAX_NIBBLES_PER_PORT] = {0};
    uint8_t l_vrefdq_train_enable[mss::exp::MAX_RANK_PER_DIMM][mss::exp::MAX_NIBBLES_PER_PORT] = {0};

    FAPI_TRY( mss::attr::get_exp_resp_vref_dq_train_value(i_target, l_vrefdq_train_value), "Error in mrs06_data()" );
    FAPI_TRY( mss::attr::get_exp_resp_vref_dq_train_range(i_target, l_vrefdq_train_range), "Error in mrs06_data()" );
    FAPI_TRY( mss::attr::get_exp_resp_vref_dq_train_enable(i_target, l_vrefdq_train_enable), "Error in mrs06_data()" );
    FAPI_TRY( FAPI_ATTR_GET(fapi2::ATTR_MEM_EFF_DRAM_TCCD_L, l_port_target, iv_tccd_l), "Error in mrs06_data()" );

    for (int i = 0; i < mss::exp::MAX_RANK_PER_DIMM; i++)
    {
        // Using DRAM 0 values due to missing values
        // From skipped DRAMs or spares that DNE
        iv_vrefdq_train_value[i] = l_vrefdq_train_value[i][0];
        iv_vrefdq_train_range[i] = l_vrefdq_train_range[i][0];
        iv_vrefdq_train_enable[i] = l_vrefdq_train_enable[i][0];
    }

    o_rc = fapi2::FAPI2_RC_SUCCESS;
    return;

fapi_try_exit:
    o_rc = fapi2::current_err;
    FAPI_ERR("%s unable to get attributes for mrs06", mss::c_str(i_target));
    return;
}

template<>
fapi2::ReturnCode (*mrs06_data<mss::mc_type::EXPLORER>::make_ccs_instruction)(const
        fapi2::Target<fapi2::TARGET_TYPE_DIMM>& i_target,
        const mrs06_data<mss::mc_type::EXPLORER>& i_data,
        ccs::instruction_t<mss::mc_type::EXPLORER>& io_inst,
        const uint64_t i_rank) = &mrs06;

template<>
fapi2::ReturnCode (*mrs06_data<mss::mc_type::EXPLORER>::decode)(const ccs::instruction_t<mss::mc_type::EXPLORER>&
        i_inst,
        const uint64_t i_rank) = &mrs06_decode;

} // ns ddr4

} // ns mss
