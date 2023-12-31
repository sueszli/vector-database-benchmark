/* IBM_PROLOG_BEGIN_TAG                                                   */
/* This is an automatically generated prolog.                             */
/*                                                                        */
/* $Source: src/usr/targeting/common/util.C $                             */
/*                                                                        */
/* OpenPOWER HostBoot Project                                             */
/*                                                                        */
/* Contributors Listed Below - COPYRIGHT 2012,2023                        */
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
/**
 *  @file targeting/common/util.C
 *
 *  @brief Provides miscellaneous utility functions to targeting, including
 *     a check for whether system is in simulation or not.
 */

//******************************************************************************
// Includes
//******************************************************************************
#include <stdint.h>
#include <targeting/common/attributes.H>
#include <targeting/common/targetservice.H>
#include <targeting/common/utilFilter.H>
#include <targeting/common/trace.H>
#ifdef __HOSTBOOT_MODULE
#include <targeting/common/predicates/predicates.H>
#include <target_states.H>
#include <util/crc32.H>
#endif


namespace TARGETING
{

/**
 * @brief Checks to see if we are running in a hardware simulation
 *    environment, i.e. VPO/VBU  (not Simics)
 */
bool is_vpo( void )
{
    struct IsVpoFunctor
    {
        static bool operate()
        {
            bool rc = false;

            Target* sys = NULL;
            targetService().getTopLevelTarget(sys);

            uint8_t vpo_mode = 0;
            if (unlikely(sys &&
                         sys->tryGetAttr<ATTR_IS_SIMULATION>(vpo_mode) &&
                         (1 == vpo_mode)))
            {
                rc = true;
            }
            return rc;
        }
    };

#ifdef __HOSTBOOT_MODULE
    // In Hostboot this value cannot change, so cache it as a static.
    static bool is_vpo_mode = IsVpoFunctor::operate();
    return is_vpo_mode;
#else
    // On FSP, assumption is that the user could change this, so we cannot
    // cache it as a static.  Read from the attribute directly.
    return IsVpoFunctor::operate();
#endif
};

/**
 * @brief set HWAS Changed flag to subscription mask
 *
 *   This will be used by the HCDB service - when the target has
 *   changed, this will get called to tell the appropriate services
 *   that the change has occurred.
 */
void update_hwas_changed_mask(Target * i_target)
{
    i_target->setAttr<ATTR_HWAS_STATE_CHANGED_FLAG>(
            i_target->getAttr<ATTR_HWAS_STATE_CHANGED_SUBSCRIPTION_MASK>());
}

void extractGroupAndChip(const topoId_t i_topologyId,
                         groupId_t& o_group,
                         chipId_t& o_chip)
{
    TARGETING::Target* l_sys = TARGETING::UTIL::assertGetToplevelTarget();

    topoMode_t l_topologyMode = l_sys->getAttr<ATTR_PROC_FABRIC_TOPOLOGY_MODE>();

    // NOTE: mode 0 -> GGGC
    //       mode 1 -> GGCC
    topologyIdBits_t l_idBits;
    l_idBits.topoId = i_topologyId;
    if(l_topologyMode == TARGETING::PROC_FABRIC_TOPOLOGY_MODE_MODE0)
    {
        o_group = l_idBits.mode0.group;
        o_chip = l_idBits.mode0.chip;
    }
    else
    {
        o_group = l_idBits.mode1.group;
        o_chip = l_idBits.mode1.chip;
    }
};

/**
 * @brief set HWAS Changed flag to specific bits
 *
 *   This will be used by different services when the target needs processing.
 */
void update_hwas_changed_mask(Target * i_target, const uint64_t i_bits)
{
    i_target->setAttr<ATTR_HWAS_STATE_CHANGED_FLAG>(
            i_target->getAttr<ATTR_HWAS_STATE_CHANGED_FLAG>() |
            (i_target->getAttr<ATTR_HWAS_STATE_CHANGED_SUBSCRIPTION_MASK>() &
                i_bits));
}

/**
 * @brief clear bit in HWAS Changed Mask
 *
 *   This will be used by the appropriate services when they have handled
 *   the change flag for this target.
 */
void clear_hwas_changed_bit(Target * i_target, const HWAS_CHANGED_BIT i_bit)
{
    ATTR_HWAS_STATE_CHANGED_FLAG_type hwasChangedState =
        i_target->getAttr<ATTR_HWAS_STATE_CHANGED_FLAG>();
    hwasChangedState &= ~i_bit;
    i_target->setAttr<ATTR_HWAS_STATE_CHANGED_FLAG>(hwasChangedState);
}

/**
 * @brief   Checks if we are loading no Payload (PAYLOAD_KIND_NONE)
 */
bool is_no_load(void)
{
    TARGETING::Target * sys = NULL;
    TARGETING::targetService().getTopLevelTarget( sys );
    assert(sys != NULL);
    return (PAYLOAD_KIND_NONE == sys->getAttr<TARGETING::ATTR_PAYLOAD_KIND>());
}

bool orderByNodeAndPosition(  Target* i_firstProc,
                              Target* i_secondProc)
{
    uint8_t topoId0 = i_firstProc->getAttr<ATTR_PROC_FABRIC_TOPOLOGY_ID>();
    uint8_t topoId1 = i_secondProc->getAttr<ATTR_PROC_FABRIC_TOPOLOGY_ID>();

    return topoId0 < topoId1;
}

uint8_t  is_fused_mode( )
{
    uint8_t  l_fused;
    TARGETING::Target * sys = NULL;
    TARGETING::targetService().getTopLevelTarget( sys );
    assert(sys != NULL);

    /*
        Fused Mode

        ATTR_FUSED_CORE_MODE / ATTR_FUSED_CORE_OPTION

        smt4_only / default_cores    --> notfused
        smt4_only / normal_cores     --> notfused
        smt4_only / fused_cores      --> fused

        smt8_only / default_cores    --> fused
        smt8_only / normal_cores     --> notfused
        smt8_only / fused_cores      --> fused
    */

    // FUSED_CORE_MODE_HB represents what we read from the PVR
    uint8_t l_mode = sys->getAttr<ATTR_FUSED_CORE_MODE_HB>();

    // FUSED_CORE_OPTION is used as an override to force a mode on universal parts
    uint8_t l_option = sys->getAttr<ATTR_FUSED_CORE_OPTION>();

    if (FUSED_CORE_OPTION_USING_DEFAULT_CORES == l_option)
    {
        if (FUSED_CORE_MODE_HB_SMT4_ONLY == l_mode)
        {
            l_fused = false;
        }
        else // SMT8_ONLY
        {
            l_fused = true;
        }
    }
    else if (FUSED_CORE_OPTION_USING_NORMAL_CORES == l_option)
    {
        l_fused = false;
        if (FUSED_CORE_MODE_HB_SMT8_ONLY == l_mode)
        {
            TARG_ERR("FUSED_CORE_OPTION wants SMT4 but hardware is set to SMT8");
        }
    }
    else // USING_FUSED_CORES
    {
        l_fused = true;
        if (FUSED_CORE_MODE_HB_SMT4_ONLY == l_mode)
        {
            TARG_ERR("FUSED_CORE_OPTION wants SMT8 but hardware is set to SMT4");
        }
    }

    return(l_fused);

} // end is_fused_mode


bool arePmicsInBlueprint()
{
    static uint8_t l_pmics_flag = 0x00;
    constexpr uint8_t BLUEPRINT_CHECKED_BIT = 0x80;

    if ((l_pmics_flag & BLUEPRINT_CHECKED_BIT) != BLUEPRINT_CHECKED_BIT)
    {
        // check if system has PMICs
        TARGETING::TargetHandleList l_pmicList;
        TARGETING::Target* l_sys = TARGETING::UTIL::assertGetToplevelTarget();

        // Only get PMIC targets
        const PredicateCTM l_pmicPred(CLASS_NA, TYPE_PMIC);

        TARGETING::targetService().getAssociated( l_pmicList, l_sys,
            TargetService::CHILD, TargetService::ALL, &l_pmicPred);

        if (l_pmicList.empty())
        {
            // No PMICs found
            l_pmics_flag |= BLUEPRINT_CHECKED_BIT;
        }
        else
        {
            // Found PMICs (return true)
            l_pmics_flag |= (BLUEPRINT_CHECKED_BIT | 0x01);
        }
    }
    return (l_pmics_flag & (~BLUEPRINT_CHECKED_BIT));
}

bool isNVDIMM( const TARGETING::Target * i_target )
{
    // no nvdimm support in p10
    return false;
}

TARGETING::TargetHandle_t getTargetFromLocationCode(const std::vector<uint8_t>& i_location_code,
                                                    const TARGETING::TYPE i_type)
{
    TARGETING::TargetHandle_t out_target = NULL;
    // TODO RTC:208810 lookup target from location code / handle multiple instances
    // for now hardcode lookup for node target
    if(i_type == TARGETING::TYPE_NODE)
    {
        TARGETING::TargetHandleList l_nodeEncList;

        getEncResources(l_nodeEncList, TARGETING::TYPE_NODE,
                        TARGETING::UTIL_FILTER_ALL);

        assert(l_nodeEncList.size() == 1);

        out_target = l_nodeEncList[0];
    }
    else if (i_type == TARGETING::TYPE_SYS)
    {
        TARGETING::targetService().getTopLevelTarget(out_target);
        assert(out_target != NULL);
    }
    else if (i_type == TARGETING::TYPE_TPM)
    {
        TARGETING::TargetHandleList l_tpms;
        getChipResources(l_tpms, TARGETING::TYPE_TPM, TARGETING::UTIL_FILTER_ALL);

        assert(l_tpms.size() == 1);

        out_target = l_tpms[0];
    }

    return out_target;
}


TARGETING::TargetHandleList getProcNVDIMMs( TARGETING::Target * i_proc )
{
    TargetHandleList o_nvdimmList;

    TARGETING::ATTR_MODEL_type l_chipModel =
        i_proc->getAttr<TARGETING::ATTR_MODEL>();

    // NVDIMM only present on NIMBUS systems
    if (l_chipModel == TARGETING::MODEL_NIMBUS)
    {
        TargetHandleList l_dimmTargetList;
        getChildAffinityTargets(l_dimmTargetList, i_proc, CLASS_NA, TYPE_DIMM);

        for (TargetHandleList::iterator it = l_dimmTargetList.begin();
                 it != l_dimmTargetList.end(); ++it)
        {
            TARGETING::Target* l_dimm = *it;
            if (TARGETING::isNVDIMM(l_dimm))
            {
                // Found a valid NVDIMM
                o_nvdimmList.push_back(l_dimm);
            }
        }
    }

    return o_nvdimmList;
}


#ifdef __HOSTBOOT_MODULE
errlHndl_t getAttrMetadataPtr(void* i_targetingPtr,
                              section_metadata_mem_layout_t*& o_metadataPtr)
{
    errlHndl_t l_errl = nullptr;
    o_metadataPtr = nullptr;

    do {

    if(i_targetingPtr == nullptr)
    {
        TRACFCOMP(g_trac_targeting,
                  ERR_MRK"getAttrMetadataPtr: Bad targeting pointer passed");
        /*@
         * @errortype
         * @severity   ERRORLOG::ERRL_SEV_UNRECOVERABLE
         * @moduleid   TARG_GET_ATTR_METADATA_PTR
         * @reasoncode TARG_RC_BAD_TARGETING_PTR
         * @devdesc    nullptr was passed as the targeting pointer
         * @custdesc   Failure during the boot of the system
         */
        l_errl = new ERRORLOG::ErrlEntry(ERRORLOG::ERRL_SEV_UNRECOVERABLE,
                                         TARG_GET_ATTR_METADATA_PTR,
                                         TARG_RC_BAD_TARGETING_PTR,
                                         0,
                                         0,
                                         ERRORLOG::ErrlEntry::ADD_SW_CALLOUT);
        break;
    }

    const TARGETING::TargetingHeader* l_targetingHeader =
        reinterpret_cast<TARGETING::TargetingHeader*>(i_targetingPtr);

    if(l_targetingHeader->eyeCatcher != TARGETING::PNOR_TARG_EYE_CATCHER)
    {
        TRACFCOMP(g_trac_targeting,
                  ERR_MRK"getAttrMetadataPtr: Targeting header is incorrect. Expected 0x%x; actual 0x%x",
                  TARGETING::PNOR_TARG_EYE_CATCHER,
                  l_targetingHeader->eyeCatcher);
        /*@
         * @errortype
         * @severity   ERRORLOG::ERRL_SEV_UNRECOVERABLE
         * @moduleid   TARG_GET_ATTR_METADATA_PTR
         * @reasoncode TARG_RC_BAD_EYECATCH
         * @userdata1  Expected targeting eye catcher
         * @userdata2  Actual eye catcher
         * @devdesc    Incorrect pointer to targeting was passed
         * @custdesc   Failure during the boot of the system
         */
        l_errl = new ERRORLOG::ErrlEntry(ERRORLOG::ERRL_SEV_UNRECOVERABLE,
                                         TARG_GET_ATTR_METADATA_PTR,
                                         TARG_RC_BAD_EYECATCH,
                                         PNOR_TARG_EYE_CATCHER,
                                         l_targetingHeader->eyeCatcher,
                                         ERRORLOG::ErrlEntry::ADD_SW_CALLOUT);
        break;
    }

    const size_t l_sectionCount = l_targetingHeader->numSections;
    const TargetingSection* l_sectionPtr =
        reinterpret_cast<TargetingSection*>(
            reinterpret_cast<uint8_t*>(i_targetingPtr) +
            sizeof(TargetingHeader) +
            l_targetingHeader->offsetToSections);

    for(size_t i = 0; i < l_sectionCount; ++i, ++l_sectionPtr)
    {
        if(l_sectionPtr->sectionType == SECTION_TYPE_HB_METADATA)
        {
            o_metadataPtr = reinterpret_cast<section_metadata_mem_layout_t*>(
                                reinterpret_cast<uint8_t*>(i_targetingPtr) +
                                l_sectionPtr->sectionOffset);
            break;
        }
    }

    if(o_metadataPtr == nullptr)
    {
        TRACFCOMP(g_trac_targeting,
                  ERR_MRK"getAttrMetadataPtr: Could not find metadata section");
        /*@
         * @errortype
         * @severity   ERRORLOG::ERRL_SEV_UNRECOVERABLE
         * @moduleid   TARG_GET_ATTR_METADATA_PTR
         * @reasoncode TARG_RC_NO_METADATA
         * @userdata1  The number of sections included in the HBD image
         * @devdesc    HBD metadata section was not found in targeting image
         * @custdesc   Failure during the boot of the system
         */
        l_errl = new ERRORLOG::ErrlEntry(ERRORLOG::ERRL_SEV_UNRECOVERABLE,
                                         TARG_GET_ATTR_METADATA_PTR,
                                         TARG_RC_NO_METADATA,
                                         l_sectionCount,
                                         0,
                                         ERRORLOG::ErrlEntry::ADD_SW_CALLOUT);
        break;
    }

    } while(0);
    return l_errl;
}

errlHndl_t parseAttrMetadataSection(section_metadata_mem_layout_t const* i_attrMetadataPtr,
                                    attr_metadata_map& o_map)
{
    errlHndl_t l_errl = nullptr;

    do {
    if(i_attrMetadataPtr == nullptr)
    {
        TRACFCOMP(g_trac_targeting,
                  ERR_MRK"parseAttrMetadataSection: nullptr passed for metadata section");
        /*@
         * @errortype
         * @severity   ERRORLOG::ERRL_SEV_UNRECOVERABLE
         * @moduleid TARG_PARSE_ATTR_METADATA
         * @reasoncode TARG_RC_BAD_METADATA_PTR
         * @devdesc nullptr passed for metadata section pointer
         * @custdesc Failure during the boot of the system
         */
        l_errl = new ERRORLOG::ErrlEntry(ERRORLOG::ERRL_SEV_UNRECOVERABLE,
                                         TARG_PARSE_ATTR_METADATA,
                                         TARG_RC_BAD_METADATA_PTR,
                                         0,
                                         0,
                                         ERRORLOG::ErrlEntry::ADD_SW_CALLOUT);
        break;
    }
    // First is the number of attributes in the section (uint32_t)
    uint32_t const l_numAttrs = i_attrMetadataPtr->numAttrs;
    // After are structs attributeId(uint32_t),size(uint32_t),persistency(uint8_t)
    section_metadata_t const * l_sectionMetadata = i_attrMetadataPtr->attrMetadata;


    for(size_t i = 0; i < l_numAttrs; ++i, ++l_sectionMetadata)
    {
        o_map[l_sectionMetadata->attrId] = l_sectionMetadata->attrMetadata;
    }
    }while(0);

    return l_errl;
}

errlHndl_t parseRWAttributeData(rw_attr_section_t const* i_rwAttributePtr,
                                huid_rw_attrs_map& o_parsedData)
{
    errlHndl_t l_errl = nullptr;

    do
    {

    if(i_rwAttributePtr == nullptr)
    {
        TRACFCOMP(g_trac_targeting,
                  ERR_MRK"parseRWAttributeData: Bad RW Attribute Section pointer passed");
        /*@
         * @errortype
         * @severity   ERRORLOG::ERRL_SEV_UNRECOVERABLE
         * @moduleid   TARG_PARSE_RW_ATTR_DATA
         * @reasoncode TARG_RC_BAD_RW_ATTR_PTR
         * @devdesc    nullptr passed for RW attribute data pointer
         * @custdesc   Failure during the boot of the system
         */
        l_errl = new ERRORLOG::ErrlEntry(ERRORLOG::ERRL_SEV_UNRECOVERABLE,
                                         TARG_PARSE_RW_ATTR_DATA,
                                         TARG_RC_BAD_RW_ATTR_PTR,
                                         0,
                                         0,
                                         ERRORLOG::ErrlEntry::ADD_SW_CALLOUT);
        break;
    }

    if(i_rwAttributePtr->version != CURRENT_HBD_PERSISTENT_VERSION)
    {
        TRACFCOMP(g_trac_targeting, ERR_MRK"parseRWAttributeData: version mismatch; expected version: %d; actual version: %d",
                  CURRENT_HBD_PERSISTENT_VERSION, i_rwAttributePtr->version);
        /*@
         * @errortype
         * @severity   ERRORLOG::ERRL_SEV_INFORMATIONAL
         * @moduleid   TARG_PARSE_RW_DATA
         * @reasoncode TARG_RC_BAD_VERSION
         * @userdata1  Expected persistent data version
         * @userdata2  Actual persistent data version
         * @devdesc    The HBD persistent data has unexpected version. The data
         *             can no longer be used and will be updated to the current
         *             version.
         * @custdesc   Failure during the boot of the system
         */
        l_errl = new ERRORLOG::ErrlEntry(ERRORLOG::ERRL_SEV_INFORMATIONAL,
                                         TARG_PARSE_RW_DATA,
                                         TARG_RC_BAD_VERSION,
                                         CURRENT_HBD_PERSISTENT_VERSION,
                                         i_rwAttributePtr->version,
                                         ERRORLOG::ErrlEntry::ADD_SW_CALLOUT);
        break;
    }

    // Compute the hash of the entire partition for comparison
    const uint8_t* l_rwDataPtr = reinterpret_cast<const uint8_t*>(&(i_rwAttributePtr->version));
    uint32_t l_crcHash = Util::crc32_calc(l_rwDataPtr, i_rwAttributePtr->dataSize);

    if(i_rwAttributePtr->dataHash != l_crcHash)
    {
        TRACFCOMP(g_trac_targeting, ERR_MRK"parseRWAttributeData: computed hash of RW data doesn't match current hash. Computed hash: 0x%x; current hash: 0x%x",
                 l_crcHash, i_rwAttributePtr->dataHash);
        /*@
         * @errortype
         * @severity   ERRORLOG::ERRL_SEV_INFORMATIONAL
         * @moduleid   TARG_PARSE_RW_DATA
         * @reasoncode TARG_RC_BAD_HASH
         * @userdata1  The computed hash of the data
         * @userdata2  The hash of the data in the image
         * @devdesc    The computed hash of RW partition doesn't match the
         *             hash in the image (probably indicates image corruption)
         * @custdesc  Failure during the boot of the system
         */
        l_errl = new ERRORLOG::ErrlEntry(ERRORLOG::ERRL_SEV_INFORMATIONAL,
                                         TARG_PARSE_RW_DATA,
                                         TARG_RC_BAD_HASH,
                                         l_crcHash,
                                         i_rwAttributePtr->dataHash,
                                         ERRORLOG::ErrlEntry::ADD_SW_CALLOUT);
        break;
    }

    // After the hash is the 4-byte number of attributes included
    const uint32_t l_numAttributes = i_rwAttributePtr->numAttributes;
    // Then the actual attribute data in format: attr hash, HUID, size, persistency, value
    const rw_attr_memory_layout_t* l_attrDataPtr = i_rwAttributePtr->attrArray;

    for(size_t i = 0; i < l_numAttributes; ++i)
    {
        o_parsedData[l_attrDataPtr->huid][l_attrDataPtr->attrHash].metadata.attrSize =
                l_attrDataPtr->attrData.metadata.attrSize;
        o_parsedData[l_attrDataPtr->huid][l_attrDataPtr->attrHash].metadata.attrPersistency =
                l_attrDataPtr->attrData.metadata.attrPersistency;
        // Make sure the vector has enough space to fit the value of the attr
        o_parsedData[l_attrDataPtr->huid][l_attrDataPtr->attrHash].value.resize(l_attrDataPtr->attrData.metadata.attrSize);
        memcpy(o_parsedData[l_attrDataPtr->huid][l_attrDataPtr->attrHash].value.data(),
               l_attrDataPtr->attrData.value,
               l_attrDataPtr->attrData.metadata.attrSize);

        // Compute the next attribute pointer. We need to skip (size of the
        // attribute) bytes over the value of the current attribute.
        l_attrDataPtr = reinterpret_cast<const rw_attr_memory_layout_t*>(
                            reinterpret_cast<const uint8_t*>(l_attrDataPtr) +
                            sizeof(rw_attr_memory_layout_t) +
                            l_attrDataPtr->attrData.metadata.attrSize);
    }
    } while(0);
    return l_errl;
}

void update_sppe_target_state()
{
    TARG_INF("update_sppe_target_state: enter");

    for( const auto l_tgt : composable(getTargetsWithAttr<TARGETING::ATTR_SPPE_TARGET_STATE>)())
    {
        uint8_t l_sppeState = 0; // fapi:TargetState; zero is not defined

        // get the current ATTR_SPPE_TARGET_STATE
        l_sppeState = l_tgt->getAttr<ATTR_SPPE_TARGET_STATE>();

        // get HWAS state
        HwasState l_hwasState;
        assert( l_tgt->tryGetAttr<ATTR_HWAS_STATE>(l_hwasState),
               "update_sppe_target_state: no ATTR_HWAS_STATE for tgt 0x%.8X",
               get_huid(l_tgt));

        // Convert hwasState to SPPE State
        // NOTE: SPPE State only expects one valid state at a time.  For example,
        //       functional state is not ORed with present state
        if (l_hwasState.functional == true)
        {
            l_sppeState = fapi2::TARGET_STATE_FUNCTIONAL;
        }
        else if (l_hwasState.present)
        {
            l_sppeState = fapi2::TARGET_STATE_PRESENT;
        }
        // else leave the state as-is

        l_tgt->setAttr<ATTR_SPPE_TARGET_STATE>(l_sppeState);

        TARG_DBG("update_sppe_target_state: Setting 0x%.8X ATTR_SPPE_TARGET_STATE to 0x%X",
                 get_huid(l_tgt), l_sppeState);
    }

    TARG_INF("update_sppe_target_state: exit");
}


#endif // __HOSTBOOT_MODULE

} // end namespace TARGETING
