/* IBM_PROLOG_BEGIN_TAG                                                   */
/* This is an automatically generated prolog.                             */
/*                                                                        */
/* $Source: src/usr/isteps/pbusLinkSvc.C $                                */
/*                                                                        */
/* OpenPOWER HostBoot Project                                             */
/*                                                                        */
/* Contributors Listed Below - COPYRIGHT 2012,2021                        */
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
#include    "pbusLinkSvc.H"
#include  <isteps/istep_reasoncodes.H>

namespace   EDI_EI_INITIALIZATION
{

using   namespace   TARGETING;
using   namespace   ISTEP;

PbusLinkSvc & PbusLinkSvc::getTheInstance()
{
    return Singleton<PbusLinkSvc>::instance();
}

PbusLinkSvc::PbusLinkSvc()
{
    mutex_init(&iv_mutex);
}

PbusLinkSvc::~PbusLinkSvc()
{
    mutex_destroy(&iv_mutex);
}

errlHndl_t PbusLinkSvc::getPbusConnections( TargetPairs_t & o_PbusConnections,
                                            const IOHS_CONFIG_MODE i_busType,
                                            const bool i_noDuplicate )
{
    errlHndl_t l_errl = nullptr;
    o_PbusConnections.clear();

    mutex_lock(&iv_mutex);

    if (iv_busConnections[i_busType].empty())
    {
        l_errl = collectPbusConnections(i_busType);
    }

    if (l_errl == nullptr)
    {
        TargetPairs_t * const l_PbusConnections = (i_noDuplicate
                                                   ? &iv_uniqueBusConnections[i_busType]
                                                   : &iv_busConnections[i_busType] );

        o_PbusConnections.insert( l_PbusConnections->begin(),
                                  l_PbusConnections->end() );
    }

    mutex_unlock(&iv_mutex);

    return l_errl;
}

/**
 * @brief If i_target is an IOHS or a child of an IOHS, return the
 * IOHS_CONFIG_MODE attribute from that IOHS; otherwise return
 * CONFIG_MODE_INVALID.
 */
static ATTR_IOHS_CONFIG_MODE_type getBusConfigMode(const Target* const i_target)
{
    ATTR_IOHS_CONFIG_MODE_type mode = IOHS_CONFIG_MODE_INVALID;

    if (i_target->getAttr<ATTR_TYPE>() == TYPE_IOHS)
    {
        mode = i_target->getAttr<ATTR_IOHS_CONFIG_MODE>();
    }
    else
    {
        TargetHandleList iohsParent;
        getParentAffinityTargets(iohsParent, i_target, CLASS_NA, TYPE_IOHS, false);

        if (!iohsParent.empty())
        {
            mode = iohsParent.front()->getAttr<ATTR_IOHS_CONFIG_MODE>();
        }
    }

    return mode;
}

errlHndl_t PbusLinkSvc::collectPbusConnections( const IOHS_CONFIG_MODE i_busType )
{
    errlHndl_t l_errl = nullptr;

    // Get all functional IOHS chiplets
    TARGETING::TargetHandleList l_iohsList;
    getAllChiplets(l_iohsList, TYPE_IOHS);
    TRACFCOMP( TARGETING::g_trac_targeting,
             "PbusLinkSvc::collectPbusConnections - getAllChiplets(TYPE_IOHS) "
             "returned %d entries", l_iohsList.size());

    // select the appropriate maps to work with
    TargetPairs_t* const l_pPbusConnections = &iv_busConnections[i_busType];
    TargetPairs_t* const l_pPbusUniqueConnections = &iv_uniqueBusConnections[i_busType];

    // Collect all functional IOHS connections
    for (const auto l_iohs : l_iohsList)
    {
        if (l_iohs->getAttr<ATTR_IOHS_CONFIG_MODE>() != i_busType)
        {
            continue; // Skip buses of types we're not requesting
        }

        // exit for loop if error already encountered
        if (nullptr != l_errl)
        {
            break;
        }

        TargetHandleList l_busses;
        getChildAffinityTargets(l_busses, l_iohs, CLASS_NA, TYPE_SMPGROUP);

        for (const auto l_bus : l_busses)
        {
            // get other endpoint target
            TRACFCOMP( TARGETING::g_trac_targeting,
                       "Get other endpoint target for target HUID %.8X",
                       TARGETING::get_huid(l_bus) );

            const TARGETING::Target * l_dstTgt =
                l_bus->getAttr<ATTR_PEER_TARGET>();

            TRACFCOMP( TARGETING::g_trac_targeting,
                       "Other endpoint target HUID %.8X",
                       TARGETING::get_huid(l_dstTgt) );

            // connection is existing, not to itself and is a real target
            if ((l_dstTgt != nullptr) && (l_dstTgt != l_bus))
            {
                const auto l_dstType = getBusConfigMode(l_dstTgt);
                if (l_dstType != i_busType)
                {
                    TRACFCOMP(TARGETING::g_trac_targeting,
                              "Both endpoints' bus type mismatch; "
                              "target HUID %.8X: dest HUID %.8X",
                              TARGETING::get_huid(l_bus),
                              TARGETING::get_huid(l_dstTgt));

                    // Mixed bus type connection
                    /*@
                     * @errortype    ERRL_SEV_UNRECOVERABLE
                     * @moduleid     MOD_EDI_EI_IO_RUN_TRAINING
                     * @reasoncode   RC_MIXED_PBUS_CONNECTION
                     * @userdata1    Endpoint1 bus type
                     * @userdata2    Endpoint2 bus type
                     * @custdesc     Platform generated error. Check User Data.
                     * @devdesc      Mixed bus connection of two types
                     */
                    l_errl = new ERRORLOG::ErrlEntry(
                        ERRORLOG::ERRL_SEV_UNRECOVERABLE,
                        MOD_EDI_EI_IO_RUN_TRAINING,
                        RC_MIXED_PBUS_CONNECTION,
                        i_busType,
                        l_dstType );
                    break;
                }

                // Get the chip parents of endpoints
                const TARGETING::Target * l_endp1Parent = getParentChip(l_bus);
                const TARGETING::Target * l_endp2Parent = getParentChip(l_dstTgt);

                if (l_endp1Parent == l_endp2Parent)
                {
                    TRACFCOMP(TARGETING::g_trac_targeting,
                              "Both endpoints from same chip; "
                              "target HUID %.8X dest HUID %.8X",
                              TARGETING::get_huid(l_bus),
                              TARGETING::get_huid(l_dstTgt));

                    // connection of same chip
                    /*@
                     * @errortype    ERRL_SEV_UNRECOVERABLE
                     * @moduleid     MOD_EDI_EI_IO_RUN_TRAINING
                     * @reasoncode   RC_SAME_CHIP_PBUS_CONNECTION
                     * @custdesc     Platform generated error.
                     * @devdesc      Both endpoint connections of same chip
                     */
                    l_errl = new ERRORLOG::ErrlEntry(
                        ERRORLOG::ERRL_SEV_UNRECOVERABLE,
                        MOD_EDI_EI_IO_RUN_TRAINING,
                        RC_SAME_CHIP_PBUS_CONNECTION );
                    break;
                }

                if (l_dstTgt->getAttr<ATTR_HWAS_STATE>().functional)
                {
                    TRACFCOMP( TARGETING::g_trac_targeting,
                               "ADDING pair HUID %.8X,%.8X TYPE %d",
                               TARGETING::get_huid(l_bus),
                               TARGETING::get_huid(l_dstTgt),
                               i_busType);
                    // save the pair if not yet done so
                    (*l_pPbusConnections)[l_bus] = l_dstTgt;
                    (*l_pPbusUniqueConnections)[l_bus] = l_dstTgt;
                }
            }
        }
    } // for l_bus_iter

    // Validate pbus connections and strike out duplicates for the Unique
    // connection map
    TargetPairs_t::iterator l_jtr;

    for (const auto l_PbusConnection : (*l_pPbusUniqueConnections))
    {
        // exit for loop if error already encountered
        if (nullptr != l_errl)
        {
            break;
        }

        const TARGETING::Target *l_ptr1 = l_PbusConnection.first;
        const TARGETING::Target *l_ptr2 = l_PbusConnection.second;
        l_jtr = l_pPbusUniqueConnections->find(l_ptr2);
        if ( (l_jtr == l_pPbusUniqueConnections->end()) ||
             (l_jtr->second != l_ptr1) )
        {
            // Connection is conflicting, e.g.
            //   endp1 -> endp2 but endp2 -> endp3.
            //   endp1 -> endp2 but endp2 -> endp2 (itself) or not existing
            EntityPath l_path;
            l_path = l_ptr2->getAttr<ATTR_PHYS_PATH>();
            char *l_pathString = l_path.toString();
            TARG_ERR( "First endpoint's PEER_TARGET is %s", l_pathString );
            free (l_pathString);
            if (l_jtr != l_pPbusUniqueConnections->end())
            {
                l_path = l_jtr->second->getAttr<ATTR_PHYS_PATH>();
                l_pathString = l_path.toString();
                TARG_ERR("Second endpoint's PEER_TARGET is %s", l_pathString);
                free (l_pathString);
            }
            else
            {
                if (l_ptr2)
                {
                    l_path = l_ptr2->getAttr<ATTR_PHYS_PATH>();
                    l_pathString = l_path.toString();
                    TARG_ERR("Second endpoint's PEER_TARGET is itself, %s",
                                                                l_pathString);
                    free (l_pathString);
                }
                else
                {
                    TARG_ERR("Second endpoint's PEER_TARGET is not existing");
                }
            }

            /*@
             * @errortype    ERRL_SEV_UNRECOVERABLE
             * @moduleid     MOD_EDI_EI_IO_RUN_TRAINING
             * @reasoncode   RC_CONFLICT_PBUS_CONNECTION
             * @userdata1    Bus endpoint target pointer1
             * @userdata2    Bus endpoint target pointer2
             * @custdesc     Platform generated error. Check User Data.
             * @devdesc      Connection is conflicting.
             *               endp1 -> endp2 but endp2 -> endp3.
             *               endp1 -> endp2 but
             *               endp2 -> endp2 (itself) or not existing
             */
            l_errl = new ERRORLOG::ErrlEntry(ERRORLOG::ERRL_SEV_UNRECOVERABLE,
                                         MOD_EDI_EI_IO_RUN_TRAINING,
                                         RC_CONFLICT_PBUS_CONNECTION,
                                         reinterpret_cast<uint64_t>(l_ptr1),
                                         reinterpret_cast<uint64_t>(l_ptr2));
            break;
        }
        else
        {
          l_pPbusUniqueConnections->erase(l_jtr);
        }
    }

    return l_errl;
}

}
