/* IBM_PROLOG_BEGIN_TAG                                                   */
/* This is an automatically generated prolog.                             */
/*                                                                        */
/* $Source: src/usr/vpd/pvpd.C $                                          */
/*                                                                        */
/* OpenPOWER HostBoot Project                                             */
/*                                                                        */
/* Contributors Listed Below - COPYRIGHT 2013,2022                        */
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
// ----------------------------------------------
// Includes
// ----------------------------------------------
#include <string.h>
#include <endian.h>
#include <trace/interface.H>
#include <errl/errlentry.H>
#include <errl/errlmanager.H>
#include <targeting/common/targetservice.H>
#include <targeting/common/util.H>
#include <targeting/common/utilFilter.H>
#include <devicefw/driverif.H>
#include <vfs/vfs.H>
#include <vpd/vpdreasoncodes.H>
#include <vpd/pvpdenums.H>
#include <vpd/vpd_if.H>
#include <eeprom/eepromif.H>
#include "pvpd.H"
#include "vpd.H"
#include <initservice/initserviceif.H>


// ----------------------------------------------
// Trace definitions
// ----------------------------------------------
extern trace_desc_t* g_trac_vpd;


// ------------------------
// Macros for unit testing
//#define TRACUCOMP(args...)  TRACFCOMP(args)
#define TRACUCOMP(args...)
//#define TRACSSCOMP(args...)  TRACFCOMP(args)
#define TRACSSCOMP(args...)

namespace PVPD
{
    // ----------------------------------------------
    // Globals
    // ----------------------------------------------
    mutex_t g_mutex = MUTEX_INITIALIZER;


    /**
     * @brief This function will perform the steps required to do a read from
     *      the Hostboot PVPD data.
     *
     * @param[in] i_opType - Operation Type - See DeviceFW::OperationType in
     *       driververif.H
     *
     * @param[in] i_target - Processor Target device
     *
     * @param [in/out] io_buffer - Pointer to the data that was read from
     *       the target device.  This parameter, when set to NULL, will return
     *       the keyword size value in io_buflen.
     *
     * @param [in/out] io_buflen - Length of the buffer to be read or written
     *       to/from the target.  This value should indicate the size of the
     *       io_buffer parameter that has been allocated.  Being returned it
     *       will indicate the number of valid bytes in the buffer being
     *       returned. This parameter will contain the size of a keyword when
     *       the io_buffer parameter is passed in NULL.
     *
     * @param [in] i_accessType - Access Type - See DeviceFW::AccessType in
     *       usrif.H
     *
     * @param [in] i_args - This is an argument list for the device driver
     *       framework.
     *
     * @return errlHndl_t - NULL if successful, otherwise a pointer to the
     *       error log.
     */
    errlHndl_t pvpdRead ( DeviceFW::OperationType i_opType,
                          TARGETING::Target * i_target,
                          void * io_buffer,
                          size_t & io_buflen,
                          int64_t i_accessType,
                          va_list i_args )
    {
        errlHndl_t err = NULL;
        IpVpdFacade::input_args_t args;
        args.record = ((pvpdRecord)va_arg( i_args, uint64_t ));
        args.keyword = ((pvpdKeyword)va_arg( i_args, uint64_t ));

        TRACSSCOMP( g_trac_vpd,
                    ENTER_MRK"pvpdRead()" );

        err = Singleton<PvpdFacade>::instance().read(i_target,
                                                     io_buffer,
                                                     io_buflen,
                                                     args);

        return err;
    }


    /**
     * @brief This function will perform the steps required to do a write to
     *      the Hostboot PVPD data.
     *
     * @param[in] i_opType - Operation Type - See DeviceFW::OperationType in
     *       driververif.H
     *
     * @param[in] i_target - Processor Target device
     *
     * @param [in/out] io_buffer - Pointer to the data that was read from
     *       the target device.  It will also be used to contain data to
     *       be written to the device.
     *
     * @param [in/out] io_buflen - Length of the buffer to be read or written
     *       to/from the target.  This value should indicate the size of the
     *       io_buffer parameter that has been allocated.  Being returned it
     *       will indicate the number of valid bytes in the buffer being
     *       returned.
     *
     * @param [in] i_accessType - Access Type - See DeviceFW::AccessType in
     *       usrif.H
     *
     * @param [in] i_args - This is an argument list for the device driver
     *       framework.
     *
     * @return errlHndl_t - NULL if successful, otherwise a pointer to the
     *       error log.
     */
    errlHndl_t pvpdWrite ( DeviceFW::OperationType i_opType,
                           TARGETING::Target * i_target,
                           void * io_buffer,
                           size_t & io_buflen,
                           int64_t i_accessType,
                           va_list i_args )
    {
        errlHndl_t err = NULL;
        IpVpdFacade::input_args_t args;
        args.record = ((pvpdRecord)va_arg( i_args, uint64_t ));
        args.keyword = ((pvpdKeyword)va_arg( i_args, uint64_t ));

        TRACSSCOMP( g_trac_vpd,
                    ENTER_MRK"pvpdWrite()" );


        err = Singleton<PvpdFacade>::instance().write(i_target,
                                                      io_buffer,
                                                      io_buflen,
                                                      args);

        return err;
    }

    // Register with the routing code
    DEVICE_REGISTER_ROUTE( DeviceFW::READ,
                           DeviceFW::PVPD,
                           TARGETING::TYPE_NODE,
                           pvpdRead );
    DEVICE_REGISTER_ROUTE( DeviceFW::WRITE,
                           DeviceFW::PVPD,
                           TARGETING::TYPE_NODE,
                           pvpdWrite );
    DEVICE_REGISTER_ROUTE( DeviceFW::READ,
                           DeviceFW::PVPD,
                           TARGETING::TYPE_TPM,
                           pvpdRead );
    DEVICE_REGISTER_ROUTE( DeviceFW::WRITE,
                           DeviceFW::PVPD,
                           TARGETING::TYPE_TPM,
                           pvpdWrite );
    DEVICE_REGISTER_ROUTE( DeviceFW::READ,
                           DeviceFW::VPD,
                           TARGETING::TYPE_NODE,
                           pvpdRead );
    DEVICE_REGISTER_ROUTE( DeviceFW::WRITE,
                           DeviceFW::VPD,
                           TARGETING::TYPE_NODE,
                           pvpdWrite );
    DEVICE_REGISTER_ROUTE( DeviceFW::READ,
                           DeviceFW::VPD,
                           TARGETING::TYPE_TPM,
                           pvpdRead );
    DEVICE_REGISTER_ROUTE( DeviceFW::WRITE,
                           DeviceFW::VPD,
                           TARGETING::TYPE_TPM,
                           pvpdWrite );

}; // end namespace PVPD

#if !defined(__HOSTBOOT_RUNTIME)
// --------------------------------------------------------
// Presence Detection
//---------------------------------------------------------

/**
 * @brief Performs a presence detect operation on a Node card.
 *
 * There is no FSI presence detection, just Planar vpd detection.
 * Presence is always returned as Success (unless the unlikely case of too
 * small of a buffer passed). A problem with planar EEPROM is logged but
 * not passed up so that the enclosure and everything inside is not
 * deconfigured.
 *
 * @param[in]   i_opType        Operation type, see DeviceFW::OperationType
 *                              in driverif.H
 * @param[in]   i_target        Presence detect target
 * @param[in/out] io_buffer     Read: Pointer to output data storage
 *                              Write: Pointer to input data storage
 * @param[in/out] io_buflen     Input: size of io_buffer (in bytes, always 1)
 *                              Output: Success = 1, Failure = 0
 * @param[in]   i_accessType    DeviceFW::AccessType enum (userif.H)
 * @param[in]   i_args          This is an argument list for DD framework.
 *                              In this function, there are no arguments.
 * @return  errlHndl_t
 */
errlHndl_t nodePresenceDetect(DeviceFW::OperationType i_opType,
                              TARGETING::Target* i_target,
                              void* io_buffer,
                              size_t& io_buflen,
                              int64_t i_accessType,
                              va_list i_args)
{
    errlHndl_t l_errl = NULL;
    bool pvpd_present = true;
    TRACFCOMP(g_trac_vpd,ENTER_MRK"nodePresenceDetect(%.8X)",
              TARGETING::get_huid(i_target));

    if (unlikely(io_buflen < sizeof(bool)))
    {
        TRACFCOMP(g_trac_vpd,
                  ERR_MRK "nodePresenceDetect> Invalid data length: %d",
                  io_buflen);
        /*@
         * @errortype
         * @moduleid     VPD::VPD_PVPD_PRESENCEDETECT
         * @reasoncode   VPD::VPD_INVALID_LENGTH
         * @userdata1    Data Length
         * @devdesc      presenceDetect> Invalid data length (!= 1 bytes)
         * @custdesc     An internal firmware error occurred.
         */
        l_errl =
                new ERRORLOG::ErrlEntry(ERRORLOG::ERRL_SEV_UNRECOVERABLE,
                                        VPD::VPD_PVPD_PRESENCEDETECT,
                                        VPD::VPD_INVALID_LENGTH,
                                        TO_UINT64(io_buflen),
                                        true /*SW error*/);
        io_buflen = 0;
        return l_errl;
    }

    pvpd_present = VPD::pvpdPresent( i_target );
    if( !pvpd_present )
    {
        TRACFCOMP(g_trac_vpd,
                  ERR_MRK "nodePresenceDetect> failed presence detect");
        pvpd_present = true;  //node PVPD always returns present
    }

    //Fsp sets PN/SN so if there is none, do it here
    if(!INITSERVICE::spBaseServicesEnabled())
    {
        // set part and serial number attributes for current target
        VPD::setPartAndSerialNumberAttributes( i_target );
    }

    // Always return presence.
    // A returned error deconfigures the node and stops the IPL.
    memcpy(io_buffer, &pvpd_present, sizeof(pvpd_present));
    io_buflen = sizeof(pvpd_present);

    TRACFCOMP(g_trac_vpd,EXIT_MRK"nodePresenceDetect(%.8X)",
              TARGETING::get_huid(i_target));
    return l_errl;
}

// Register as the presence detect for nodes.
DEVICE_REGISTER_ROUTE(DeviceFW::READ,
                      DeviceFW::PRESENT,
                      TARGETING::TYPE_NODE,
                      nodePresenceDetect);
#endif //#if !defined(__HOSTBOOT_RUNTIME)


bool VPD::pvpdPresent( TARGETING::Target * i_target )
{
    TRACSSCOMP( g_trac_vpd, ENTER_MRK"pvpdPresent()");
#ifndef  __HOSTBOOT_RUNTIME

    return EEPROM::eepromPresence( i_target );

#else
    return Singleton<PvpdFacade>::instance().hasVpdPresent( i_target,
                                                            PVPD::VINI,
                                                            PVPD::PN );
#endif
}


//PVPD Class Functions
/**
 * @brief  Constructor for Planar VPD
 */
PvpdFacade::PvpdFacade() :
IpVpdFacade(PVPD::pvpdRecords,
            (sizeof(PVPD::pvpdRecords)/sizeof(PVPD::pvpdRecords[0])),
            PVPD::pvpdKeywords,
            (sizeof(PVPD::pvpdKeywords)/sizeof(PVPD::pvpdKeywords[0])),
            PVPD::g_mutex,
            VPD::VPD_WRITE_CACHE)
{
    TRACUCOMP(g_trac_vpd, "PvpdFacade::PvpdFacade> " );
}

