/* IBM_PROLOG_BEGIN_TAG                                                   */
/* This is an automatically generated prolog.                             */
/*                                                                        */
/* $Source: src/usr/pldm/extended/pldm_entity_ids.C $                     */
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

/* @file pldm_entity_ids.C
 *
 * @brief Implementation for interface in pldm_entity_ids.H
 */

// Targeting
#include <targeting/common/targetservice.H>
#include <targeting/common/utilFilter.H>

// PLDM
#include <pldm/pldm_trace.H>
#include <pldm/extended/pldm_entity_ids.H>
#include <pldm/extended/pdr_manager.H>
#include <pldm/extended/hb_pdrs.H>
#include <pldm/extended/pldm_fru.H>
#include <pldm/pldm_reasoncodes.H>
#include <pldm/pldm_errl.H>

// Error logs
#include <errl/errlmanager.H>

// libpldm header from pldm subtree
#include <openbmc/pldm/libpldm/include/libpldm/pdr.h>

using namespace ERRORLOG;

namespace
{

using namespace PLDM;
using namespace TARGETING;

// These are entities whose RSID is generated by Hostboot. We encode the RSID so
// that we can retrieve the Target instance associated with the entity.
const entity_type hb_entity_types[] =
{
    ENTITY_TYPE_PROCESSOR,
    ENTITY_TYPE_LOGICAL_PROCESSOR,
    ENTITY_TYPE_DIMM
};

// These are entities whose RSID is NOT generated by Hostboot. We don't have a
// strategy yet to associate them with the right Target instance, so we are
// relying on them having one and only one instance and associating them that
// way.
struct unique_entity_info
{
    TARGETING::CLASS target_class;
    TARGETING::TYPE target_type;
    entity_type ent_type;
};

const unique_entity_info foreign_entity_types[] =
{
    { CLASS_ENC, TYPE_NODE, ENTITY_TYPE_BACKPLANE },
    { CLASS_SYS, TYPE_SYS,  ENTITY_TYPE_CHASSIS },
    { CLASS_SYS, TYPE_SYS,  ENTITY_TYPE_LOGICAL_SYSTEM },
    { CLASS_CHIP, TYPE_TPM, ENTITY_TYPE_TPM }
};

const unique_entity_info hb_connector_info_attr_types[] =
{
    { CLASS_LOGICAL_CARD, TYPE_DIMM, ENTITY_TYPE_DIMM_SLOT },
    { CLASS_CHIP,         TYPE_PROC, ENTITY_TYPE_SOCKET },
};

/**
 * @brief Update all connector target's ATTR_CONNECTOR_PLDM_ENTITY_ID_INFO
 *        containerId to that of the backplane
 * @param[in] i_backplane_containerId containerId of backplane
 */
void updateConnectorInfoAttr(const uint16_t i_backplane_containerId)
{
    // This variable is to prevent setting ATTR_CONNECTOR_PLDM_ENTITY_ID_INFO
    // multiple times to the same backplane value
    static uint16_t l_backplane_container = 0xFFFF;

    PLDM_DBG(">> updateConnectorInfoAttr(0x%04X)", i_backplane_containerId);
    ATTR_CONNECTOR_PLDM_ENTITY_ID_INFO_type targeting_entity_id = { };

    TargetHandleList connector_targets;

    if (i_backplane_containerId != l_backplane_container)
    {
        for (const auto& connector_info : hb_connector_info_attr_types)
        {
            // grab all targets of ATTR_TYPE
            getClassResources(connector_targets, connector_info.target_class, connector_info.target_type, UTIL_FILTER_ALL);

            for (const auto& connector_target : connector_targets)
            {
                targeting_entity_id = connector_target->getAttr<ATTR_CONNECTOR_PLDM_ENTITY_ID_INFO>();
                PLDM_DBG("updateConnectorInfoAttr(backplaneId): Found target 0x%08x -> 0x%04x/0x%04x/0x%04x (entitytype/instanceNum/containerId)",
                    get_huid(connector_target),
                    targeting_entity_id.entityType,
                    targeting_entity_id.entityInstanceNumber,
                    targeting_entity_id.containerId);
                if (targeting_entity_id.containerId != i_backplane_containerId)
                {
                    targeting_entity_id.containerId = i_backplane_containerId;
                    PLDM_INF("updateConnectorInfoAttr(backplaneId): Set CONNECTOR_PLDM_ENTITY_ID for"
                        " HUID 0x%08x to 0x%04x/0x%04x/0x%04x (entitytype/instanceNum/containerId)",
                        get_huid(connector_target),
                        targeting_entity_id.entityType,
                        targeting_entity_id.entityInstanceNumber,
                        targeting_entity_id.containerId);
                    connector_target->setAttr<ATTR_CONNECTOR_PLDM_ENTITY_ID_INFO>(targeting_entity_id);
                }
                else
                {
                    PLDM_DBG("updateConnectorInfoAttr(backplaneId): Skip setting HUID 0x%08X "
                        "CONNECTOR_PLDM_ENTITY_ID as its containerId (0x%04x) already is set correctly",
                        get_huid(connector_target),
                        targeting_entity_id.containerId);
                }
            }
        }
        l_backplane_container = i_backplane_containerId;
    }
    PLDM_DBG("<< updateConnectorInfoAttr(0x%04X)", i_backplane_containerId);
}

/* @brief Update ATTR_CONNECTOR_PLDM_ENTITY_ID_INFO attached to child_target
 *        with its parent socket/slot normalized information
 * @param[in] i_child_target Target with ATTR_CONNECTOR_PLDM_ENTITY_ID_INFO
 * @param[in] i_child_entity_info Normalized entity id information of child target
 */
void updateConnectorInfoAttr(Target* const i_child_target,
                          const TARGETING::ATTR_PLDM_ENTITY_ID_INFO_type& i_child_entity_info)
{
    errlHndl_t errl = nullptr;
    TARGETING::ATTR_CONNECTOR_PLDM_ENTITY_ID_INFO_type connectorInfo = { };
    uint16_t childContainerId = le16toh(i_child_entity_info.containerId);
    uint16_t childInstanceNum = le16toh(i_child_entity_info.entityInstanceNumber);
    std::vector<pldm_entity> vAssocEntities = {};
    do
    {
        vAssocEntities = thePdrManager().findEntityAssociationsByContainer(childContainerId);
        if (vAssocEntities.size() == 1)
        {
            const pldm_entity entity = vAssocEntities[0];

            // convert to host endian for comparison and tracing
            connectorInfo.entityType = le16toh(entity.entity_type);
            connectorInfo.entityInstanceNumber = le16toh(entity.entity_instance_num);
            connectorInfo.containerId = le16toh(entity.entity_container_id);

            // check this is an expected entity type using this container
            if ( (connectorInfo.entityType != ENTITY_TYPE_DIMM_SLOT) &&
                 (connectorInfo.entityType != ENTITY_TYPE_SOCKET) )
            {
                // need to go back another parent to find DCM socket connection
                if (connectorInfo.entityType != ENTITY_TYPE_PROCESSOR_MODULE)
                {
                    PLDM_ERR("updateConnectorInfoAttr: unsupported entityType 0x%04X", connectorInfo.entityType);
                    /*@
                     * @moduleid   MOD_UPDATE_CONNECTOR_INFO_ATTR
                     * @reasoncode RC_UNSUPPORTED_TYPE
                     * @userdata1  The child Target HUID
                     * @userdata2  Entity type
                     * @devdesc    Unsupported entity type for connector type.  Possible BMC normalization failure.
                     * @custdesc   A software error occurred during system boot
                     */
                    errl = new ErrlEntry(ERRL_SEV_PREDICTIVE,
                                         MOD_UPDATE_CONNECTOR_INFO_ATTR,
                                         RC_UNSUPPORTED_TYPE,
                                         get_huid(i_child_target),
                                         connectorInfo.entityType,
                                         ErrlEntry::NO_SW_CALLOUT);
                    addBmcErrorCallouts(errl);
                    break;
                }
                vAssocEntities.clear();
                // switch to the DCM container ID
                PLDM_DBG("updateConnectorInfoAttr: switch to DCM container 0x%04X, instanceNumber 0x%04X",
                  connectorInfo.containerId, connectorInfo.entityInstanceNumber);
                childContainerId = connectorInfo.containerId;
                childInstanceNum = connectorInfo.entityInstanceNumber;
                continue;
            }

            // socket or dimm_slot instance numbers should match their child entity's instance number
            if (childInstanceNum != connectorInfo.entityInstanceNumber)
            {
                PLDM_ERR("updateConnectorInfoAttr: entity_instance_num mismatch (assoc = 0x%04X, child = 0x%04X)",
                  connectorInfo.entityInstanceNumber, childInstanceNum);
                /*@
                 * @moduleid          MOD_UPDATE_CONNECTOR_INFO_ATTR
                 * @reasoncode        RC_MISMATCHED_ENTITY_INSTANCE
                 * @userdata1         The child Target HUID
                 * @userdata2[0:31]   Parent connector entity instance number
                 * @userdata2[32:63]  Expected child entity instance number
                 * @devdesc    Connection entity instance number not in sync with child value
                 * @custdesc   A software error occurred during system boot
                 */
                errl = new ErrlEntry(ERRL_SEV_PREDICTIVE,
                                     MOD_UPDATE_CONNECTOR_INFO_ATTR,
                                     RC_MISMATCHED_ENTITY_INSTANCE,
                                     get_huid(i_child_target),
                                     TWO_UINT32_TO_UINT64(
                                        connectorInfo.entityInstanceNumber,
                                        childInstanceNum),
                                     ErrlEntry::NO_SW_CALLOUT);
                addBmcErrorCallouts(errl);
                break;
            }

            PLDM_INF("updateConnectorInfoAttr: Set CONNECTOR_PLDM_ENTITY_ID for"
                " HUID 0x%08x (%s) to 0x%04x/0x%04x/0x%04x (entitytype/instanceNum/containerId)",
                get_huid(i_child_target),
                attrToString<ATTR_TYPE>(i_child_target->getAttr<ATTR_TYPE>()),
                connectorInfo.entityType,
                connectorInfo.entityInstanceNumber,
                connectorInfo.containerId);

            // convert to LE format
            connectorInfo.entityType = htole16(connectorInfo.entityType);
            connectorInfo.entityInstanceNumber = htole16(connectorInfo.entityInstanceNumber),
            connectorInfo.containerId = htole16(connectorInfo.containerId);
            i_child_target->setAttr<ATTR_CONNECTOR_PLDM_ENTITY_ID_INFO>(connectorInfo);

            // now update all socket types to same container (backplane)
            updateConnectorInfoAttr(connectorInfo.containerId);
            break;
        }
        else
        {
            PLDM_ERR("updateConnectorInfoAttr: PLDM socket or dimm_slot entity should only have 1 child, found %d", vAssocEntities.size());
            for (const auto & pEntityId : vAssocEntities)
            {
                PLDM_ERR("updateConnectorInfoAttr: Entity 0x%04x/0x%04x/0x%04x (entitytype/instanceNum/containerId)",
                   pEntityId.entity_type, pEntityId.entity_instance_num, pEntityId.entity_container_id);
            }
            /*@
             * @moduleid    MOD_UPDATE_CONNECTOR_INFO_ATTR
             * @reasoncode  RC_INVALID_RECORD_COUNT
             * @userdata1   The child Target HUID
             * @userdata2[0:31]   container_id searched
             * @userdata2[32:63]  Number of association records found matching container_id
             * @devdesc     Only one association record should be found.
             * @custdesc    A software error occurred during system boot
             */
            errl = new ErrlEntry(ERRL_SEV_PREDICTIVE,
                                 MOD_UPDATE_CONNECTOR_INFO_ATTR,
                                 RC_INVALID_RECORD_COUNT,
                                 get_huid(i_child_target),
                                 TWO_UINT32_TO_UINT64(
                                     childContainerId,
                                     vAssocEntities.size()),
                                 ErrlEntry::NO_SW_CALLOUT);
            addBmcErrorCallouts(errl);

            break;
        }
    } while (1);

    // commit errors but continue the IPL as
    // this will just limit LED identification of empty hw slots
    if (errl)
    {
        errl->collectTrace(PLDM_COMP_NAME);
        errlCommit(errl, PLDM_COMP_ID);
    }
}


/* @brief Update the given target's PLDM_ENTITY_ID_INFO attribute
 *        with a PLDM entity ID
 *
 * @param[in] i_target  The target to update
 * @param[in] i_rsid    FRU Record Set ID to use for Entity ID info
 * @return errlHndl_t   Error if any, otherwise nullptr
 */
errlHndl_t updateTargetEntityIdAttribute(Target* const i_target,
                                         const fru_record_set_id_t i_rsid)
{
    errlHndl_t errl = nullptr;

    do
    {

    pldm_entity ent { };

    if (!thePdrManager().findEntityByFruRecordSetId(i_rsid, ent))
    {
        PLDM_ERR("Cannot find entity by FRU RSID 0x%04x", i_rsid);

        /*@
         * @errortype  ERRL_SEV_UNRECOVERABLE
         * @moduleid   MOD_PLDM_ENTITY_IDS
         * @reasoncode RC_NO_ENTITY_FROM_RSID
         * @userdata1  The Target HUID
         * @userdata2  The FRU Record Set ID
         * @devdesc    Software problem, cannot find Entity from FRU Record Set ID
         * @custdesc   A software error occurred during system boot
         */
        errl = new ErrlEntry(ERRL_SEV_UNRECOVERABLE,
                             MOD_PLDM_ENTITY_IDS,
                             RC_NO_ENTITY_FROM_RSID,
                             get_huid(i_target),
                             i_rsid,
                             ErrlEntry::ADD_SW_CALLOUT);

        addBmcErrorCallouts(errl);
        break;
    }

    PLDM_INF("Set PLDM_ENTITY_ID for HUID 0x%08x (%s) to 0x%04x/0x%04x/0x%04x",
             get_huid(i_target),
             attrToString<ATTR_TYPE>(i_target->getAttr<ATTR_TYPE>()),
             ent.entity_type,
             ent.entity_instance_num,
             ent.entity_container_id);

    // This union groups all PLDM entity ID information attributes into the
    // same storage location to be exploited below.  All PLDM entity ID
    // information attribute flavors need to move in lock step if they are ever
    // changed
    const union {
        TARGETING::ATTR_PLDM_ENTITY_ID_INFO_type         generic;
        TARGETING::ATTR_CHASSIS_PLDM_ENTITY_ID_INFO_type chassis;
        TARGETING::ATTR_SYSTEM_PLDM_ENTITY_ID_INFO_type  system;
    } entity_info = {

        // The attribute stores its values in little-endian
        .generic = {
            .entityType = static_cast<uint16_t>(htole16(ent.entity_type)),
            .entityInstanceNumber = static_cast<uint16_t>(htole16(ent.entity_instance_num)),
            .containerId = static_cast<uint16_t>(htole16(ent.entity_container_id))
         }
    };
    TARGETING::ATTR_CONNECTOR_PLDM_ENTITY_ID_INFO_type connectorInfo = { };

    switch(ent.entity_type)
    {
        // System and chassis PLDM entity ID information map to specially named
        // attributes (with the same format as PLDM_ENTITY_ID_INFO) that reside
        // on the system target
        case ENTITY_TYPE_LOGICAL_SYSTEM:
            PLDM_INF("Writing logical system PLDM entity ID info: 0x%04x/0x%04x/0x%04x (entitytype/instanceNum/containerId)",
                entity_info.system.entityType,
                entity_info.system.entityInstanceNumber,
                entity_info.system.containerId);
            i_target->setAttr<ATTR_SYSTEM_PLDM_ENTITY_ID_INFO>(entity_info.system);
            break;
        case ENTITY_TYPE_CHASSIS:
            PLDM_INF("Writing chassis PLDM entity ID info: 0x%04x/0x%04x/0x%04x (entitytype/instanceNum/containerId)",
                entity_info.chassis.entityType,
                entity_info.chassis.entityInstanceNumber,
                entity_info.chassis.containerId);
            i_target->setAttr<ATTR_CHASSIS_PLDM_ENTITY_ID_INFO>(entity_info.chassis);
            break;
        default:
            i_target->setAttr<ATTR_PLDM_ENTITY_ID_INFO>(entity_info.generic);
            if (UTIL::assertGetToplevelTarget()->getAttr<ATTR_PLDM_CONNECTOR_PDRS_ENABLED>() &&
                i_target->tryGetAttr<ATTR_CONNECTOR_PLDM_ENTITY_ID_INFO>(connectorInfo))
            {
                updateConnectorInfoAttr(i_target, entity_info.generic);
            }
            break;
    }

    } while (false);

    return errl;
}

}

errlHndl_t PLDM::assignTargetEntityIds()
{
    PLDM_ENTER("assignTargetEntityIds");

    errlHndl_t errl = nullptr;

    do
    {

    // These RSIDs are not produced by Hostboot so we can't decode them to
    // determine which Target it corresponds to. Instead we rely on the fact
    // that there is only one Target object on single-node machines. This block
    // deals with that.
    for (const auto& entity_info : foreign_entity_types)
    {
        const auto target_rsid = thePdrManager().findFruRecordSetIdsByType(entity_info.ent_type);

        // We don't have a way to associate multiple backplane RSIDs (which is
        // created by the BMC) with Targets at this point, so throw an error
        if (target_rsid.size() > 1)
        {
            PLDM_ERR("Got %llu targets of type %s, expected 0 or 1",
                     target_rsid.size(),
                     attrToString<ATTR_TYPE>(entity_info.target_type));

            /*@
             * @errortype  ERRL_SEV_UNRECOVERABLE
             * @moduleid   MOD_PLDM_ENTITY_IDS
             * @reasoncode RC_EXPECTED_UNIQUE_ENTITY
             * @userdata1  The number of targets found
             * @userdata2[0:31]  The class of target
             * @userdata2[32:63] The type of target
             * @devdesc    Software problem, wanted 0 or 1 target, got multiple
             * @custdesc   A software error occurred during system boot
             */
            errl = new ErrlEntry(ERRL_SEV_UNRECOVERABLE,
                                 MOD_PLDM_ENTITY_IDS,
                                 RC_EXPECTED_UNIQUE_ENTITY,
                                 target_rsid.size(),
                                 TWO_UINT32_TO_UINT64(entity_info.target_class,
                                                      entity_info.target_type),
                                 ErrlEntry::ADD_SW_CALLOUT);

            addBmcErrorCallouts(errl);
            break;
        }

        TargetHandleList target;

        getClassResources(target,
                          entity_info.target_class,
                          entity_info.target_type,
                          UTIL_FILTER_ALL);

        if (target.size() != target_rsid.size())
        {
            PLDM_ERR("Hostboot has %llu targets of type %s, but BMC has %llu PLDM entities of type %d",
                     target.size(),
                     attrToString<ATTR_TYPE>(entity_info.target_type),
                     target_rsid.size(),
                     entity_info.ent_type);

            /*@
             * @errortype        ERRL_SEV_INFORMATIONAL
             * @moduleid         MOD_PLDM_ENTITY_IDS
             * @reasoncode       RC_EXPECTED_UNIQUE_TARGET
             * @userdata1[0:31]  The number of targets found
             * @userdata1[32:63] The number of PLDM entities found
             * @userdata2[0:31]  The target class
             * @userdata2[32:63] The target type
             * @devdesc          Software problem, mismatching number of Targets and Entities
             * @custdesc         A software error occurred during system boot
             */
            errl = new ErrlEntry(ERRL_SEV_INFORMATIONAL,
                                 MOD_PLDM_ENTITY_IDS,
                                 RC_EXPECTED_UNIQUE_TARGET,
                                 TWO_UINT32_TO_UINT64(target.size(), target_rsid.size()),
                                 TWO_UINT32_TO_UINT64(entity_info.target_class, entity_info.target_type),
                                 ErrlEntry::ADD_SW_CALLOUT);

            addBmcErrorCallouts(errl);
            addPldmFrData(errl);
            errlCommit(errl, PLDM_COMP_ID);
            continue;
        }

        if (!target.empty())
        {
            errl = updateTargetEntityIdAttribute(target[0], target_rsid[0]);

            if (errl)
            {
                break;
            }
        }
    }

    if (errl)
    {
        break;
    }

    // Now deal with records whose RSIDs Hostboot did encode with the
    // corresponding Target's class/type/instance.
    for (const entity_type ent_type : hb_entity_types)
    {
        const auto fru_rsids = thePdrManager().findFruRecordSetIdsByType(ent_type, thePdrManager().hostbootTerminusId());

        for (const auto rsid : fru_rsids)
        {
            Target* const target = getTargetFromHostbootFruRecordSetID(rsid);

            // This means that either there was a programming error, or else the
            // BMC did not preserve our RSIDs when we sent our PDRs to them and
            // refetched them at the end of the PDR exchange.
            if (!target)
            {
                PLDM_ERR("Cannot find target from RSID 0x%04x", rsid);

                /*@
                 * @errortype  ERRL_SEV_UNRECOVERABLE
                 * @moduleid   MOD_PLDM_ENTITY_IDS
                 * @reasoncode RC_NO_TARGET_FROM_RSID
                 * @userdata1  The FRU Record Set ID that does not correspond to a Target
                 * @userdata2  The entity type of the corresponding Record Set
                 * @devdesc    Software problem, cannot find the Target from FRU Record Set ID
                 * @custdesc   A software error occurred during system boot
                 */
                errl = new ErrlEntry(ERRL_SEV_UNRECOVERABLE,
                                     MOD_PLDM_ENTITY_IDS,
                                     RC_NO_TARGET_FROM_RSID,
                                     rsid,
                                     ent_type,
                                     ErrlEntry::ADD_SW_CALLOUT);

                addBmcErrorCallouts(errl);
                break;
            }

            errl = updateTargetEntityIdAttribute(target, rsid);

            if (errl)
            {
                break;
            }
        }

        if (errl)
        {
            break;
        }
    }

    } while (false);

    // checks for PLDM error and adds flight recorder data to log
    addPldmFrData(errl);

    PLDM_EXIT("assignTargetEntityIds");

    return errl;
}
