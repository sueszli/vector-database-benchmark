/* IBM_PROLOG_BEGIN_TAG                                                   */
/* This is an automatically generated prolog.                             */
/*                                                                        */
/* $Source: src/usr/htmgt/runtime/rt_occ.C $                              */
/*                                                                        */
/* OpenPOWER HostBoot Project                                             */
/*                                                                        */
/* Contributors Listed Below - COPYRIGHT 2014,2022                        */
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
#include    <runtime/interface.h>
#include    <util/utillidmgr.H>
#include    <htmgt/htmgt.H>
#include    "../htmgt_utility.H"
#include    "../htmgt_occmanager.H"
#include    <targeting/common/commontargeting.H>
#include    <targeting/runtime/rt_targeting.H>
#include    <targeting/translateTarget.H>
#include    <runtime/runtime_reasoncodes.H>
#include    <util/runtime/rt_fwreq_helper.H>
#include    <map>
#include    <targeting/common/utilFilter.H>

using namespace TARGETING;
extern trace_desc_t* g_trac_hbrt;

const std::map<OCC_RESET_REASON, HTMGT::occResetReason>g_PMComplexResetReasonMap =
    {
        {OCC_RESET_REASON_ERROR, HTMGT::OCC_RESET_REASON_ERROR},
        {OCC_RESET_REASON_CODE_UPDATE, HTMGT::OCC_RESET_REASON_CODE_UPDATE},
        {OCC_RESET_REASON_CANCEL_CODE_UPDATE, HTMGT::OCC_RESET_REASON_CANCEL_CODE_UPDATE},
        // FIXME RTC: 214351 TMGT_REQUEST currently doesn't map
        {OCC_RESET_REASON_TMGT_REQUEST, HTMGT::OCC_RESET_REASON_NONE},
        {OCC_RESET_REASON_NA, HTMGT::OCC_RESET_REASON_NONE},
    };

namespace HTMGT
{

    //------------------------------------------------------------------------

    void process_occ_error (uint64_t i_chipId)
    {
        TRACFCOMP(g_trac_hbrt, ENTER_MRK" process_occ_error");
#ifdef CONFIG_HTMGT
        TARGETING::Target* l_reportingOccTarget = NULL;
        errlHndl_t err = RT_TARG::getHbTarget(i_chipId,l_reportingOccTarget);
        if (err)
        {
            TMGT_ERR ("process_occ_error: getHbTarget"
                      " failed at %d chipId", i_chipId);
            errlCommit (err, RUNTIME_COMP_ID);
        }
        else
        {
            HTMGT::processOccError(l_reportingOccTarget);
        }
#else
        TMGT_ERR("Unexpected call to process_occ_error(%d)"
                 " when HTMGT is not enabled", i_chipId);
#endif
        TRACFCOMP(g_trac_hbrt, EXIT_MRK" process_occ_error");
    }

    //------------------------------------------------------------------------

    void process_occ_reset (uint64_t i_chipId)
    {
        TRACFCOMP(g_trac_hbrt, ENTER_MRK" process_occ_reset");
#ifdef CONFIG_HTMGT
        TARGETING::Target* l_failedOccTarget = NULL;
        errlHndl_t err = RT_TARG::getHbTarget(i_chipId,l_failedOccTarget);
        if (err)
        {
            TMGT_ERR ("process_occ_reset: getHbTarget"
                      " failed at %d chipId", i_chipId);
            errlCommit (err, RUNTIME_COMP_ID);
        }
        else
        {
            HTMGT::processOccReset(l_failedOccTarget);
        }
#else
        TMGT_ERR("Unexpected call to process_occ_reset(%d)"
                 " when HTMGT is not enabled", i_chipId);
#endif
        TRACFCOMP(g_trac_hbrt, EXIT_MRK" process_occ_reset");
    }

    //------------------------------------------------------------------------

    int enable_occ_actuation (int i_occ_activation)
    {
        TRACFCOMP(g_trac_hbrt, ENTER_MRK" enable_occ_actuation");
        int rc = 0;
#ifdef CONFIG_HTMGT
        errlHndl_t err = HTMGT::enableOccActuation(0 != i_occ_activation);
        if (err)
        {
            rc = err->reasonCode();
            if (0 == rc)
            {
                // If there was a failure, be sure to return non-zero status
                rc = -1;
            }
            TMGT_ERR ("enable_occ_actuation: OCC state change"
                      " failed with rc=0x%04X (actuate=%d)",
                      err->reasonCode(), i_occ_activation);
            errlCommit (err, RUNTIME_COMP_ID);
        }
#else
        rc = -1;
        TMGT_ERR("Unexpected call to enable_occ_actuation(%d)"
                 " when HTMGT is not enabled", i_occ_activation);
#endif
        TRACFCOMP(g_trac_hbrt, EXIT_MRK" enable_occ_actuation: rc=0x%X",rc);
        return rc;
    }

    //------------------------------------------------------------------------

    int htmgt_pass_thru (uint16_t   i_cmdLength,
                             uint8_t *  i_cmdData,
                             uint16_t * o_rspLength,
                             uint8_t *  o_rspData)
    {
        TRACFCOMP(g_trac_hbrt, ENTER_MRK" mfg_htmgt_pass_thru");
        int rc = 0;
#ifdef CONFIG_HTMGT
        errlHndl_t err = HTMGT::passThruCommand(i_cmdLength, i_cmdData,
                                                *o_rspLength, o_rspData);
        if (err)
        {
            rc = err->reasonCode();
            if (0 == rc)
            {
                // If there was a failure, be sure to return non-zero status
                rc = -1;
            }
            if ((i_cmdLength > 0) && (NULL != i_cmdData))
            {
                TMGT_ERR ("htmgt_pass_thru: command 0x%02X"
                          " (%d bytes) failed with rc=0x%04X",
                          i_cmdData[0], i_cmdLength, err->reasonCode());
            }
            errlCommit (err, RUNTIME_COMP_ID);
        }
#else
        o_rspLength = 0;
#endif
        TRACFCOMP(g_trac_hbrt, EXIT_MRK" mfg_htmgt_pass_thru: rc=0x%X",rc);
        return rc;
    }


    void alert_phyp()
    {
#if defined(__HOSTBOOT_RUNTIME) && !defined(CONFIG_FSP_BUILD)
        errlHndl_t l_errl = nullptr;

        TargetHandleList l_procChips;
        getAllChips(l_procChips, TYPE_PROC, true);

        TRACFCOMP(g_trac_hbrt, "alert_phyp: %d proc(s) found",
                  l_procChips.size());

        for (const auto & l_procChip: l_procChips)
        {
            // Inform PHYP that we are about to reset the PM complex on
            //  this chip (BMC systems only)

            // Create the firmware_request request struct
            hostInterfaces::hbrt_fw_msg l_req_msg;
            memset(&l_req_msg, 0, sizeof(l_req_msg));  // clear it all
            l_req_msg.io_type = hostInterfaces::HBRT_FW_MSG_TYPE_PM_RESET_ALERT;

            // Get the Proc Chip Id
            TARGETING::rtChipId_t l_chipId = 0;
            l_errl = TARGETING::getRtTarget(l_procChip, l_chipId);
            if(l_errl)
            {
                TRACFCOMP(g_trac_hbrt,
                          ERR_MRK"alert_phyp: getRtTarget ERROR for %.8X",
                          TARGETING::get_huid(l_procChip) );
            }
            else
            {
                TRACFCOMP(g_trac_hbrt, "alert_phyp: Sending "
                          "HBRT_FW_MSG_TYPE_PM_RESET_ALERT(%d) message to phyp",
                          l_chipId );

                l_req_msg.pmreset_alert.procId = l_chipId;

                // actual msg size (one type of hbrt_fw_msg)
                uint64_t l_req_msg_size = hostInterfaces::HBRT_FW_MSG_BASE_SIZE+
                    sizeof(l_req_msg.pmreset_alert);

                // Create the firmware_request response struct to receive data
                hostInterfaces::hbrt_fw_msg l_resp_fw_msg;
                uint64_t l_resp_fw_msg_size = sizeof(l_resp_fw_msg);
                memset(&l_resp_fw_msg, 0, l_resp_fw_msg_size);

                // Make the firmware_request call
                l_errl = firmware_request_helper(l_req_msg_size,
                                                 &l_req_msg,
                                                 &l_resp_fw_msg_size,
                                                 &l_resp_fw_msg);
                if (l_errl)
                {
                    TRACFCOMP(g_trac_hbrt, ERR_MRK"alert_phyp: firmware_request"
                              " failed w/rc=0x%04X", l_errl->reasonCode());
                }
            }

            // commit the log and continue
            if (l_errl)
            {
                l_errl->setSev(ERRORLOG::ERRL_SEV_PREDICTIVE);
                l_errl->collectTrace("HTMGT");
                errlCommit(l_errl, ISTEP_COMP_ID);
            }
        }
#endif
    }


    //------------------------------------------------------------------------
    int reset_pm_complex_with_reason(const OCC_RESET_REASON i_reason,
                                     const uint64_t i_chipId)
    {
        uint16_t l_rc = 0;
#ifdef CONFIG_HTMGT
        errlHndl_t l_errl = nullptr;
        TARGETING::Target* l_proc = nullptr;

        TRACFCOMP(g_trac_hbrt,ENTER_MRK
                  "reset_pm_complex_with_reason: i_reason=%d, i_chipId=%d ",
                  i_reason, i_chipId);

        do{

        // If the system is in safemode then ignore request to reset OCCs
        Target* l_sys = UTIL::assertGetToplevelTarget();
        uint8_t safeMode = 0;
        if(!l_sys->tryGetAttr<TARGETING::ATTR_HTMGT_SAFEMODE>(safeMode) ||
           !safeMode)
        {
            // If a code update is started early, it is possible for PHYP to call
            // reset_pm_complex_with_reason before HBRT has even started for the first time,
            // so flag the PM_COMPLEX_LOAD_REQ to indicate to SKIP_FIRST_LOAD
            // (which will be handled in load_and_start_pm_complex).
            if ((l_sys->getAttr<ATTR_PM_COMPLEX_LOAD_REQ>()) == PM_COMPLEX_LOAD_TYPE_LOAD)
            {
                l_sys->setAttr<ATTR_PM_COMPLEX_LOAD_REQ>(PM_COMPLEX_LOAD_TYPE_SKIP_FIRST_LOAD);
                TRACFCOMP(g_trac_hbrt, "reset_pm_complex_with_reason SET to "
                          "SKIP_FIRST_LOAD ATTR_PM_COMPLEX_LOAD_REQ=0x%X",
                          l_sys->getAttr<ATTR_PM_COMPLEX_LOAD_REQ>());
                break;
            }

            // Only pass in i_chipId's conversion to l_proc if it is an error scenario
            // NOTE: have to use '::' to avoid collision with enum HTMGT::occResetReason
            if (i_reason == ::OCC_RESET_REASON_ERROR)
            {
                l_errl = RT_TARG::getHbTarget(i_chipId,
                                              l_proc);
                if(l_errl)
                {
                    TRACFCOMP(g_trac_hbrt,ERR_MRK"Could not get TARGETING::Target* for chip ID 0x%08lx!", i_chipId);
                    l_rc = ERRL_GETRC_SAFE(l_errl);
                    errlCommit(l_errl, RUNTIME_COMP_ID);
                    break;
                }
            }

            occResetReason l_resetReason = HTMGT::OCC_RESET_REASON_NONE;
            auto l_mappedReason = g_PMComplexResetReasonMap.find(i_reason);
            if(l_mappedReason == g_PMComplexResetReasonMap.end())
            {
                TRACFCOMP(g_trac_hbrt, ERR_MRK"Could not map OCC reset reason 0x%08x for chip ID 0x%08x",
                          i_reason, i_chipId);
                /*@
                 * @errortype
                 * @moduleid   RUNTIME::MOD_PM_RT_RESET_W_REASON
                 * @reasoncode RUNTIME::RC_COULD_NOT_MAP_RESET_REASON
                 * @userdata1  Input OCC reason code
                 * @userdata2  Chip ID of the input chip
                 * @devdesc    Could not map OCC reset reason from PHYP to HTMGT
                 * @custdesc   A failure occurred during runtime
                 */
                l_errl = new ERRORLOG::ErrlEntry(ERRORLOG::ERRL_SEV_UNRECOVERABLE,
                                                 RUNTIME::MOD_PM_RT_RESET_W_REASON,
                                                 RUNTIME::RC_COULD_NOT_MAP_RESET_REASON,
                                                 i_reason,
                                                 i_chipId);
                l_rc = ERRL_GETRC_SAFE(l_errl);
                errlCommit(l_errl, RUNTIME_COMP_ID);
                break;
            }
            else
            {
                l_resetReason = l_mappedReason->second;
            }

            l_errl = OccManager::resetOccs(l_proc,
                                           false, //i_skipCountIncrement
                                           false, //i_skipComm
                                           l_resetReason);
            if(l_errl)
            {
                l_rc = ERRL_GETRC_SAFE(l_errl);
                errlCommit(l_errl, RUNTIME_COMP_ID);
                break;
            }
        }
        else // system is in safe mode (already in reset)
        {
            TRACFCOMP(g_trac_hbrt, "reset_pm_complex_with_reason: System is in safe mode. "
                      "Alert PHYP that we would normally reset the PM complex");
            alert_phyp();
        }

        // ONLY -AFTER- the resetOccs successfully completes do we flag the PM_COMPLEX_LOAD_REQ
        // For this special case we want to make sure that the PM Complex is RE-loaded
        // NOTE: have to use '::' to avoid collision with enum HTMGT::occResetReason
        if (i_reason == ::OCC_RESET_REASON_CODE_UPDATE)
        {
            Target* l_sys = UTIL::assertGetToplevelTarget();
            auto pm_type = PM_COMPLEX_LOAD_TYPE_RELOAD;
            l_sys->setAttr<ATTR_PM_COMPLEX_LOAD_REQ>(pm_type);
            TRACFCOMP(g_trac_hbrt, INFO_MRK
                      "reset_pm_complex_with_reason: SET to RELOAD ATTR_PM_COMPLEX_LOAD_REQ=0x%X", pm_type);
        }

        }while(0);
#endif

        TRACFCOMP(g_trac_hbrt,EXIT_MRK"reset_pm_complex_with_reason: rc=%d ", l_rc);

        return l_rc;
    }

    //------------------------------------------------------------------------

    struct registerOcc
    {
        registerOcc()
        {
            runtimeInterfaces_t * rt_intf = getRuntimeInterfaces();
            rt_intf->process_occ_error =
                            DISABLE_MCTP_WRAPPER(process_occ_error);
            rt_intf->process_occ_reset =
                            DISABLE_MCTP_WRAPPER(process_occ_reset);
            rt_intf->enable_occ_actuation =
                            DISABLE_MCTP_WRAPPER(enable_occ_actuation);
            rt_intf->mfg_htmgt_pass_thru =
                            DISABLE_MCTP_WRAPPER(htmgt_pass_thru);
            rt_intf->reset_pm_complex_with_reason =
                            DISABLE_MCTP_WRAPPER(reset_pm_complex_with_reason);
        }
    };

    registerOcc g_registerOcc;


    //------------------------------------------------------------------------

    void process_occ_clr_msgs( void )
    {
#ifdef CONFIG_HTMGT
        // a NULL parameter will cause processOccError() to poll
        //   all of the OCCs (since the parm was invalid)
        TARGETING::Target * l_DummyOccTarget = nullptr;
        HTMGT::processOccError(l_DummyOccTarget);
#else
        TMGT_ERR("Unexpected call to process_occ_clr_msgs"
                " when HTMGT is not enabled");
#endif
    }

    //------------------------------------------------------------------------

    struct registerOccStartup
    {
        registerOccStartup()
        {
            // Register interface for Host to call
            postInitCalls_t * rt_post = getPostInitCalls();
            rt_post->callClearPendingOccMsgs = &process_occ_clr_msgs;
        }

    };

    registerOccStartup g_registerOccStartup;

} // end namespace HTMGT
