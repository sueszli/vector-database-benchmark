/* IBM_PROLOG_BEGIN_TAG                                                   */
/* This is an automatically generated prolog.                             */
/*                                                                        */
/* $Source: src/usr/vpd/vpd_ecc_api_algorithms.C $                        */
/*                                                                        */
/* OpenPOWER HostBoot Project                                             */
/*                                                                        */
/* Contributors Listed Below - COPYRIGHT 2020,2022                        */
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

#include "ipvpd.H"   // vpdeccCreateEcc, vpdeccCheckData
#include <stdint.h>
#include <ecc/vpd_ecc_api_wrapper.h>

#ifdef CONFIG_COMPILE_VPD_ECC_ALGORITHMS
// Create a reference to the symbol seepromGetEcc, which is provided by the
// "real" ECC implementation. This will cause the build to fail if the real ECC
// implementation is not linked in, but the user specified the flag
// CONFIG_COMPILE_VPD_ECC_ALGORITHMS in the build config file.
extern size_t seepromGetEcc;
size_t* seepromGetEcc_reference = &seepromGetEcc;
#else
// Create a reference to the symbol seepromGetEccShim, which is provided by the
// "no-op" ECC implementation. This will cause the build to fail if the no-op
// ECC implementation is not linked in and the user didn't specify
// CONFIG_COMPILE_VPD_ECC_ALGORITHMS in the build config file.
extern size_t seepromGetEccShim;
size_t* seepromGetEccShim_reference = &seepromGetEccShim;
#endif

// ------------------------------------------------------------------
// IpVpdFacade::vpdeccCreateEcc
// ------------------------------------------------------------------
int IpVpdFacade::vpdeccCreateEcc(
                const unsigned char* i_recordData, size_t  i_recordLength,
                unsigned char*       o_eccData,    size_t* io_eccLength)
{
    return ::vpdeccCreateEcc_wrapper(i_recordData, i_recordLength, o_eccData, io_eccLength);
} // vpdeccCreateEcc

// ------------------------------------------------------------------
// IpVpdFacade::vpdeccCheckData
// ------------------------------------------------------------------
int IpVpdFacade::vpdeccCheckData(
                unsigned char*       io_recordData, size_t i_recordLength,
                const unsigned char* i_eccData,     size_t i_eccLength)
{
    return ::vpdeccCheckData_wrapper(io_recordData, i_recordLength, i_eccData, i_eccLength);
} // vpdeccCheckData
