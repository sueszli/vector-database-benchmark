/* IBM_PROLOG_BEGIN_TAG                                                   */
/* This is an automatically generated prolog.                             */
/*                                                                        */
/* $Source: src/usr/diag/prdf/common/framework/config/iipSystem.C $       */
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

// Module Description **************************************************
//
// Description: This module provides the implementation for the PRD
//              System class.
//
// End Module Description **********************************************

/*--------------------------------------------------------------------*/
/* Emit the virtual function tables and inline function defintions in
 this translation unit.                                             */
/*--------------------------------------------------------------------*/
#ifdef __GNUC__
#endif

//----------------------------------------------------------------------
//  Includes
//----------------------------------------------------------------------

#include <prdfMain.H>
#include <iipSystem.h>
#include <iipResolution.h>
#include <iipsdbug.h>
#include <iipchip.h>
#include <iipDomain.h>
#include <iipServiceDataCollector.h>
#include <iipResolutionFactory.h>
#include <prdfPlatServices.H>
#include <prdfGlobal.H>
#include <prdfRuleMetaData.H>
#include <prdfMfgThresholdMgr.H>

namespace PRDF
{

//----------------------------------------------------------------------
//  User Types
//----------------------------------------------------------------------

//----------------------------------------------------------------------
//  Constants
//----------------------------------------------------------------------

//----------------------------------------------------------------------
//  Macros
//----------------------------------------------------------------------

//----------------------------------------------------------------------
//  Internal Function Prototypes
//----------------------------------------------------------------------

//----------------------------------------------------------------------
//  Global Variables
//----------------------------------------------------------------------

//---------------------------------------------------------------------
// Member Function Specifications
//---------------------------------------------------------------------

System::System( Resolution & noSystemAttentions ) :
    chips(),
    prioritizedDomains(),
    noAttnResolution(noSystemAttentions)
{}

System::~System(void)
{
    for(ChipMapType::iterator chipIterator = chips.begin();
        chipIterator != chips.end();chipIterator++)
    {
        delete (*chipIterator);
    }

    // change to delete prioritiezed Domains instead of domains dg04
    for(DomainContainerType::iterator domainIterator = prioritizedDomains.begin();
        domainIterator != prioritizedDomains.end();domainIterator++)
    {
        delete (*domainIterator);
    }
    // clear the Resolution factory
    ResolutionFactory::Access().Reset();

    // clear the MfgThresholdMgr
    MfgThresholdMgr::getInstance()->reset();

    // clear the threshold policies
    ThresholdResolution::reset();
    //clears list of all the RuleMetaData instances
    for( RuleMetaDataList::iterator l_itMetaData = iv_listRuleData.begin();
            l_itMetaData != iv_listRuleData.end(); l_itMetaData++ )
    {
        delete( l_itMetaData->second );
    }
    iv_listRuleData.clear();

}

CHIP_CLASS * System::GetChip(TARGETING::TargetHandle_t i_pchipHandle )
{
    CHIP_CLASS * chipPtr = nullptr;

    //  chips.LookUp(chipPtr, chipId);
    for(uint32_t i = 0; i < chips.size(); ++i)
    {
        if(chips[i]->getTrgt() == i_pchipHandle)
        {
            chipPtr = chips[i];
            break;
        }
    }

    return(chipPtr);
}

Domain * System::GetDomain(DOMAIN_ID domainId)
{
    Domain * domainPtr = nullptr;

    //  domains.LookUp(domainPtr, domainId);
    for(uint32_t i = 0; i < prioritizedDomains.size(); ++i)
    {
        if(prioritizedDomains[i]->GetId() == domainId)
        {
            domainPtr = prioritizedDomains[i];
            break;
        }
    }

    return(domainPtr);
}

void System::AddChips(ChipContainerIterator begin,
                      ChipContainerIterator end)
{
    size_t l_size = 0;
    if(begin < end) l_size = end-begin;
    else l_size = begin-end;
    chips.reserve(chips.size()+l_size);
    while(begin != end)
    {
        chips.push_back(*begin);
        begin++;
    }
}

void System::AddDomains(DomainContainerIterator begin,
                        DomainContainerIterator end)
{
    size_t l_size = 0;
    if(begin < end) l_size = end-begin;
    else l_size = begin-end;
    prioritizedDomains.reserve(prioritizedDomains.size()+l_size);
    // The Configurator owns determining the priority of the domains.
    while(begin != end)
    {
        prioritizedDomains.push_back(*begin);        // dg01 - add: Keep order given
        begin++;
    }
}

void System::RemoveStoppedChips(TARGETING::TargetHandle_t i_pchipHandle)
{
    PRDF_TRAC("System::RemoveStoppedChips chip: 0x%08x",
                  PlatServices::getHuid(i_pchipHandle));
    for(DomainContainerType::iterator domainIterator = prioritizedDomains.begin();
        domainIterator != prioritizedDomains.end(); domainIterator++)
    {
        (*domainIterator)->Remove(i_pchipHandle);
    }
}

// --------------------------------------------------------------------

void System::RemoveNonFunctionalChips()
{
    ChipMapType l_chips;

    for(ChipMapType::iterator chipIterator = chips.begin();
        chipIterator != chips.end();
        chipIterator++)
    {
        if(!PlatServices::isFunctional((*chipIterator)->getTrgt()))
        {
            l_chips.push_back(*chipIterator);
        }
    }

    // The reason for l_chips is because we can't remove elements from a vector
    // as we are iterating on it. Otherwise, it will foul up the iterators.
    // Now, this is not the most efficient way to remove the chips, because this
    // implementation will remove (put at the end of the vector) the chip then
    // erase it and repeat for each element. Instead, it should remove all chips
    // then erase all removed chips. However, for the scope of this code, the
    // efficiency increase is nominal so it can be left for future improvements.
    for(ChipMapType::iterator chipIterator = l_chips.begin();
        chipIterator != l_chips.end();
        chipIterator++)
    {
        RemoveStoppedChips((*chipIterator)->getTrgt());
    }
}

// --------------------------------------------------------------------

void System::Initialize(void)
{
    uint32_t rc = SUCCESS;
    //SYSTEM_DEBUG_CLASS sysdebug; dg08d

    for(ChipMapType::iterator chipIterator = chips.begin();
        (chipIterator != chips.end()) && (rc == SUCCESS);chipIterator++)
    {
        rc = (*chipIterator)->Initialize();
        // if rc then an error log was (will be) committed
        if(rc != SUCCESS)
        {
            PRDF_ERR( "System::Initialize of chips failed. rc = %x", rc );
        }
    }

    // change domains to priortizedDomains  dg04
    for(DomainContainerType::iterator domainIterator = prioritizedDomains.begin();
        (domainIterator != prioritizedDomains.end()) && (rc == SUCCESS);
        domainIterator++)
    {
        rc = (*domainIterator)->Initialize();

        if(rc != SUCCESS)
        {
            PRDF_ERR( "System::Initialize of chips failed. rc = %x", rc );
        }
    }
}

//------------------------------------------------------------------------------

int32_t System::Analyze( STEP_CODE_DATA_STRUCT & io_sc )
{
    #define PRDF_FUNC "[System::Analyze] "
    SYSTEM_DEBUG_CLASS sysdebug;
    Domain * domainAtAttentionPtr = nullptr;
    ServiceDataCollector * l_saved_sdc = nullptr;
    ServiceDataCollector * l_temp_sdc = nullptr;

    int32_t rc = (prioritizedDomains.empty() ? NO_DOMAINS_IN_SYSTEM : SUCCESS);
    int32_t l_saved_rc = 0;
    int32_t l_primaryRc = 0;
    if(rc == SUCCESS)
    {
        ATTENTION_TYPE priAttnType = io_sc.service_data->getPrimaryAttnType();

        // If the primary attention type is a system checkstop or unit checkstop
        // then start with recoverable attentions because it is possible they
        // may be the root cause of the checkstops.
        ATTENTION_TYPE startAttnType = priAttnType;
        if ( (MACHINE_CHECK == priAttnType) || (UNIT_CS == priAttnType) )
            startAttnType = RECOVERABLE;

        for ( ATTENTION_TYPE secAttnType = startAttnType;
              domainAtAttentionPtr == nullptr && secAttnType >= priAttnType;
              --secAttnType )
        {
            DomainContainerType::iterator domainIterator;

            for( domainIterator = prioritizedDomains.begin();
                 domainIterator != prioritizedDomains.end() &&
                 domainAtAttentionPtr == nullptr; )
            {
                bool l_continueInDomain = false;
                domainAtAttentionPtr = ((*domainIterator)->Query(secAttnType)) ? (*domainIterator) : nullptr;
                if(domainAtAttentionPtr != nullptr)
                {
                    io_sc.service_data->setSecondaryAttnType(secAttnType);
                    rc = domainAtAttentionPtr->Analyze(io_sc, secAttnType);
                    if((rc == PRD_SCAN_COMM_REGISTER_ZERO) ||
                        (rc == PRD_POWER_FAULT) )
                    {
                        // save sdc, and continue
                        if(l_saved_sdc == nullptr)
                        {
                            l_saved_sdc = new ServiceDataCollector(
                                                *io_sc.service_data);
                            l_saved_rc = rc;
                        }

                        if( ( PRD_SCAN_COMM_REGISTER_ZERO == rc ) &&
                            ( io_sc.service_data->isPrimaryPass() ) &&
                             io_sc.service_data->isSecondaryErrFound() )
                        {
                            //So, the chip was reporting attention but none of
                            //the FIR read had any Primary bit set. But some
                            //secondary bits are on though. So, we would like
                            //to investigate if there are other chips in
                            //the domain which are reporting primary attention.
                            l_continueInDomain = true;
                            io_sc.service_data->clearSecondaryErrFlag();
                        }

                        domainAtAttentionPtr = nullptr;

                        if(rc == PRD_POWER_FAULT)
                        {
                            PRDF_ERR( PRDF_FUNC "Power Fault detected!" );
                            break;
                        }
                    }

                    else if ( ( priAttnType == MACHINE_CHECK )
                             && ( secAttnType != MACHINE_CHECK ) )
                    {
                        // We were asked to analyze MACHINE XTOP, but we found
                        // another attention and did analyze that. In this case
                        // we want to add additional signature of MACHINE XSTOP
                        // attention to error log.

                        // l_temp_sdc is not nullptr. It is a code bug. Do not do
                        // anything for error isolation pass.
                        if ( nullptr != l_temp_sdc )
                        {
                            PRDF_ERR( PRDF_FUNC "l_temp_sdc is not nullptr" );
                            continue;
                        }

                        // Do a setup for error isolation pass for MACHINE
                        // XTOP. In this pass, we are only interested in
                        // error signature.
                        domainAtAttentionPtr = nullptr;
                        l_temp_sdc = new ServiceDataCollector (
                                                    *io_sc.service_data );
                        // Set up Error Isolation Pass Flag.
                        io_sc.service_data->setIsolationOnlyPass();
                        // The original capture data is held with l_temp_sdc
                        // when the copy contructor was called above. If we
                        // continue to use io_sc.service_data as is, it
                        // still contains all of the current capture data. So
                        // any additional capture data will most likely be
                        // duplicates (which will actually be a third copy of
                        // the registers). The duplicates will be removed
                        // eventually when mergeData() is called later, but we
                        // can prevent a lot of duplication and have a improve
                        // preformance slightly by clearing the capture data for
                        // the isolation pass.
                        io_sc.service_data->GetCaptureData().Clear();
                        // Set the outer for loop iteration variable secAttnType so
                        // that we analyze MACHINE XSTOP in next iteration.
                        secAttnType = MACHINE_CHECK + 1;
                        // save primary rc.
                        l_primaryRc = rc;
                        break;
                    }

                } // end domainAtAttentionPtr != nullptr

                //so if a chip of a domain is at attention and gave us dd02, we
                //would like to see other chips of the domain before moving on
                //to next domain.

                if( !l_continueInDomain )
                {
                    domainIterator++;
                }

            } // end inner for loop
        } // end outer for loop

        // if ptr is nullptr && we don't have a saved SDC than we have noAttns
        // if ptr is nullptr && we have a saved SDC then we have an attn with no-bits-on
        // otherwise we are done - already did the analysis
        if ( domainAtAttentionPtr == nullptr)
        {
            if(l_saved_sdc == nullptr)
            {
                rc = noAttnResolution.Resolve(io_sc);
            }
            else
            {
                *io_sc.service_data = *l_saved_sdc;
                sysdebug.CalloutThoseAtAttention(io_sc);
                rc = l_saved_rc;
            }
        }

        // l_temp_sdc will not be nullptr if we go to ERROR ISOLATION ONLY PASS.
        // In that case get the secondary signature and update this.
        if ( nullptr != l_temp_sdc )
        {
            // Merge SUE flag, but only when the UERE flag is not set. We want
            // to merge the flag in cases where we happen to get another attn
            // at around the same time as a CS that has the SUE flag set, so
            // when we analyze to that other attn, we account for the SUE flag
            // from the CS and adjust the gard policy to null accordingly.
            // However, we do not want to merge the flags when we get the SUE
            // and UERE flags together, as then we end up adjusting the gard
            // policy to null when analyzing the SueSource, which we don't want.
            if( io_sc.service_data->IsSUE() &&
                !io_sc.service_data->IsUERE() )
            {
                l_temp_sdc->SetSUE();
            }

            // We are having only one secondary signature. Secondary signature
            // here is mainly used for XSTOP error. XSTOP error can only happen
            // on Fabric domain. For fabric domain, we use Fabric sorting which
            // will give us first chip which is root cause for the XSTOP error.
            // So signature we get after fabric sorting is enough for FFDC
            // perspective.

            l_temp_sdc->AddSignatureList(
                           *( io_sc.service_data->GetErrorSignature() ));

            // merge debug scom data from the two analysis
            l_temp_sdc->GetCaptureData().mergeData(
                            io_sc.service_data->GetCaptureData());

            // merge trace array data from the two analysis
            l_temp_sdc->getTraceArrayData().mergeData(
                        io_sc.service_data->getTraceArrayData());

            *io_sc.service_data = *l_temp_sdc;
            delete l_temp_sdc;

            // We will discard any error we get in ERROR ISOLATION ONLY PASS,
            // and restore rc found in primary pass.
            rc = l_primaryRc;
        }

        if(l_saved_sdc != nullptr) delete l_saved_sdc; //dg05a

    }
    #undef PRDF_FUNC
    return(rc);
}

RuleMetaData* System::getChipMetaData( TARGETING::TYPE i_type,
                                        const char *i_fileName,
                                        ScanFacility & i_scanFactory,
                                        ResolutionFactory & i_reslFactory,
                                        errlHndl_t & o_errl )
{
    if( nullptr == iv_listRuleData[i_fileName] )
    {
        iv_listRuleData[i_fileName] = new RuleMetaData( i_fileName,i_scanFactory,
                                                    i_reslFactory,i_type,
                                                    o_errl );
    }
    return iv_listRuleData[i_fileName];

}

} // end namespace PRDF

