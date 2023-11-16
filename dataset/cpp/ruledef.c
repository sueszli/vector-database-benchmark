   /*******************************************************/
   /*      "C" Language Integrated Production System      */
   /*                                                     */
   /*             CLIPS Version 6.30  10/19/06            */
   /*                                                     */
   /*                   DEFRULE MODULE                    */
   /*******************************************************/

/*************************************************************/
/* Purpose: Defines basic defrule primitive functions such   */
/*   as allocating and deallocating, traversing, and finding */
/*   defrule data structures.                                */
/*                                                           */
/* Principal Programmer(s):                                  */
/*      Gary D. Riley                                        */
/*                                                           */
/* Contributing Programmer(s):                               */
/*      Brian L. Dantes                                      */
/*                                                           */
/* Revision History:                                         */
/*      6.24: Removed CONFLICT_RESOLUTION_STRATEGIES         */
/*            compilation flag.                              */
/*                                                           */
/*            Renamed BOOLEAN macro type to intBool.         */
/*                                                           */
/*            Corrected code to remove run-time program      */
/*            compiler warnings.                             */
/*                                                           */
/*      6.30: Added support for hashed alpha memories.       */
/*                                                           */
/*            Added additional developer statistics to help  */
/*            analyze join network performance.              */
/*                                                           */
/*            Added salience groups to improve performance   */
/*            with large numbers of activations of different */
/*            saliences.                                     */
/*                                                           */
/*************************************************************/

#define _RULEDEF_SOURCE_

#include "setup.h"

#if DEFRULE_CONSTRUCT

#include <stdio.h>
#define _STDIO_INCLUDED_

#include "agenda.h"
#include "drive.h"
#include "engine.h"
#include "envrnmnt.h"
#include "memalloc.h"
#include "pattern.h"
#include "retract.h"
#include "reteutil.h"
#include "rulebsc.h"
#include "rulecom.h"
#include "rulepsr.h"
#include "ruledlt.h"

#if BLOAD || BLOAD_AND_BSAVE || BLOAD_ONLY
#include "bload.h"
#include "rulebin.h"
#endif

#if CONSTRUCT_COMPILER && (! RUN_TIME)
#include "rulecmp.h"
#endif

#include "ruledef.h"

/***************************************/
/* LOCAL INTERNAL FUNCTION DEFINITIONS */
/***************************************/

   static void                   *AllocateModule(void *,EXEC_STATUS);
   static void                    ReturnModule(void *,EXEC_STATUS,void *);
   static void                    InitializeDefruleModules(void *,EXEC_STATUS);
   static void                    DeallocateDefruleData(void *,EXEC_STATUS);
   static void                    DestroyDefruleAction(void *,EXEC_STATUS,struct constructHeader *,void *);
#if RUN_TIME   
   static void                    AddBetaMemoriesToRule(void *,EXEC_STATUS,struct joinNode *);
#endif

/**********************************************************/
/* InitializeDefrules: Initializes the defrule construct. */
/**********************************************************/
globle void InitializeDefrules(
  void *theEnv,
  EXEC_STATUS)
  {   
   unsigned long i;
   AllocateEnvironmentData(theEnv,execStatus,DEFRULE_DATA,sizeof(struct defruleData),DeallocateDefruleData);

   InitializeEngine(theEnv,execStatus);
   InitializeAgenda(theEnv,execStatus);
   InitializePatterns(theEnv,execStatus);
   InitializeDefruleModules(theEnv,execStatus);

   AddReservedPatternSymbol(theEnv,execStatus,"and",NULL);
   AddReservedPatternSymbol(theEnv,execStatus,"not",NULL);
   AddReservedPatternSymbol(theEnv,execStatus,"or",NULL);
   AddReservedPatternSymbol(theEnv,execStatus,"test",NULL);
   AddReservedPatternSymbol(theEnv,execStatus,"logical",NULL);
   AddReservedPatternSymbol(theEnv,execStatus,"exists",NULL);
   AddReservedPatternSymbol(theEnv,execStatus,"forall",NULL);

   DefruleBasicCommands(theEnv,execStatus);

   DefruleCommands(theEnv,execStatus);

   DefruleData(theEnv,execStatus)->DefruleConstruct =
      AddConstruct(theEnv,execStatus,"defrule","defrules",
                   ParseDefrule,EnvFindDefrule,
                   GetConstructNamePointer,GetConstructPPForm,
                   GetConstructModuleItem,EnvGetNextDefrule,SetNextConstruct,
                   EnvIsDefruleDeletable,EnvUndefrule,ReturnDefrule);

   DefruleData(theEnv,execStatus)->AlphaMemoryTable = (ALPHA_MEMORY_HASH **)
                  gm3(theEnv,execStatus,sizeof (ALPHA_MEMORY_HASH *) * ALPHA_MEMORY_HASH_SIZE);

   for (i = 0; i < ALPHA_MEMORY_HASH_SIZE; i++) DefruleData(theEnv,execStatus)->AlphaMemoryTable[i] = NULL;

   DefruleData(theEnv,execStatus)->BetaMemoryResizingFlag = TRUE;
   
   DefruleData(theEnv,execStatus)->RightPrimeJoins = NULL;
   DefruleData(theEnv,execStatus)->LeftPrimeJoins = NULL;   
  }
  
/**************************************************/
/* DeallocateDefruleData: Deallocates environment */
/*    data for the defrule construct.             */
/**************************************************/
static void DeallocateDefruleData(
  void *theEnv,
  EXEC_STATUS)
  {
   struct defruleModule *theModuleItem;
   void *theModule;
   struct activation *theActivation, *tmpActivation;
   struct salienceGroup *theGroup, *tmpGroup;

#if BLOAD || BLOAD_AND_BSAVE
   if (Bloaded(theEnv,execStatus))
     { return; }
#endif
   
   DoForAllConstructs(theEnv,execStatus,DestroyDefruleAction,DefruleData(theEnv,execStatus)->DefruleModuleIndex,FALSE,NULL);

   for (theModule = EnvGetNextDefmodule(theEnv,execStatus,NULL);
        theModule != NULL;
        theModule = EnvGetNextDefmodule(theEnv,execStatus,theModule))
     {
      theModuleItem = (struct defruleModule *)
                      GetModuleItem(theEnv,execStatus,(struct defmodule *) theModule,
                                    DefruleData(theEnv,execStatus)->DefruleModuleIndex);
                                    
      theActivation = theModuleItem->agenda;
      while (theActivation != NULL)
        {
         tmpActivation = theActivation->next;
         
         rtn_struct(theEnv,execStatus,activation,theActivation);
         
         theActivation = tmpActivation;
        }
        
      theGroup = theModuleItem->groupings;
      while (theGroup != NULL)
        {
         tmpGroup = theGroup->next;
         
         rtn_struct(theEnv,execStatus,salienceGroup,theGroup);
         
         theGroup = tmpGroup;
        }        

#if ! RUN_TIME                                    
      rtn_struct(theEnv,execStatus,defruleModule,theModuleItem);
#endif
     }   
     
   rm3(theEnv,execStatus,DefruleData(theEnv,execStatus)->AlphaMemoryTable,sizeof (ALPHA_MEMORY_HASH *) * ALPHA_MEMORY_HASH_SIZE);
  }
  
/********************************************************/
/* DestroyDefruleAction: Action used to remove defrules */
/*   as a result of DestroyEnvironment.                 */
/********************************************************/
#if WIN_BTC
#pragma argsused
#endif
static void DestroyDefruleAction(
  void *theEnv,
  EXEC_STATUS,
  struct constructHeader *theConstruct,
  void *buffer)
  {
#if MAC_MCW || WIN_MCW || MAC_XCD
#pragma unused(buffer)
#endif
   struct defrule *theDefrule = (struct defrule *) theConstruct;
   
   DestroyDefrule(theEnv,execStatus,theDefrule);
  }

/*****************************************************/
/* InitializeDefruleModules: Initializes the defrule */
/*   construct for use with the defmodule construct. */
/*****************************************************/
static void InitializeDefruleModules(
  void *theEnv,
  EXEC_STATUS)
  {
   DefruleData(theEnv,execStatus)->DefruleModuleIndex = RegisterModuleItem(theEnv,execStatus,"defrule",
                                    AllocateModule,
                                    ReturnModule,
#if BLOAD_AND_BSAVE || BLOAD || BLOAD_ONLY
                                    BloadDefruleModuleReference,
#else
                                    NULL,
#endif
#if CONSTRUCT_COMPILER && (! RUN_TIME)
                                    DefruleCModuleReference,
#else
                                    NULL,
#endif
                                    EnvFindDefrule);
  }

/***********************************************/
/* AllocateModule: Allocates a defrule module. */
/***********************************************/
static void *AllocateModule(
  void *theEnv,
  EXEC_STATUS)
  {
   struct defruleModule *theItem;

   theItem = get_struct(theEnv,execStatus,defruleModule);
   theItem->agenda = NULL;
   theItem->groupings = NULL;
   return((void *) theItem);
  }

/*********************************************/
/* ReturnModule: Deallocates a defrule module. */
/*********************************************/
static void ReturnModule(
  void *theEnv,
  EXEC_STATUS,
  void *theItem)
  {
   FreeConstructHeaderModule(theEnv,execStatus,(struct defmoduleItemHeader *) theItem,DefruleData(theEnv,execStatus)->DefruleConstruct);
   rtn_struct(theEnv,execStatus,defruleModule,theItem);
  }

/************************************************************/
/* GetDefruleModuleItem: Returns a pointer to the defmodule */
/*  item for the specified defrule or defmodule.            */
/************************************************************/
globle struct defruleModule *GetDefruleModuleItem(
  void *theEnv,
  EXEC_STATUS,
  struct defmodule *theModule)
  {   
   return((struct defruleModule *) GetConstructModuleItemByIndex(theEnv,execStatus,theModule,DefruleData(theEnv,execStatus)->DefruleModuleIndex)); 
  }

/*******************************************************************/
/* EnvFindDefrule: Searches for a defrule in the list of defrules. */
/*   Returns a pointer to the defrule if found, otherwise NULL.    */
/*******************************************************************/
globle void *EnvFindDefrule(
  void *theEnv,
  EXEC_STATUS,
  char *defruleName)
  {   
   return(FindNamedConstruct(theEnv,execStatus,defruleName,DefruleData(theEnv,execStatus)->DefruleConstruct)); 
  }

/************************************************************/
/* EnvGetNextDefrule: If passed a NULL pointer, returns the */
/*   first defrule in the ListOfDefrules. Otherwise returns */
/*   the next defrule following the defrule passed as an    */
/*   argument.                                              */
/************************************************************/
globle void *EnvGetNextDefrule(
  void *theEnv,
  EXEC_STATUS,
  void *defrulePtr)
  {   
   return((void *) GetNextConstructItem(theEnv,execStatus,(struct constructHeader *) defrulePtr,DefruleData(theEnv,execStatus)->DefruleModuleIndex)); 
  }

/*******************************************************/
/* EnvIsDefruleDeletable: Returns TRUE if a particular */
/*   defrule can be deleted, otherwise returns FALSE.  */
/*******************************************************/
globle intBool EnvIsDefruleDeletable(
  void *theEnv,
  EXEC_STATUS,
  void *vTheDefrule)
  {
   struct defrule *theDefrule;

   if (! ConstructsDeletable(theEnv,execStatus))
     { return FALSE; }

   for (theDefrule = (struct defrule *) vTheDefrule;
        theDefrule != NULL;
        theDefrule = theDefrule->disjunct)
     { if (theDefrule->executing) return(FALSE); }

   if (EngineData(theEnv,execStatus)->MatchOperationInProgress) return(FALSE);

   return(TRUE);
  }

#if RUN_TIME

/******************************************/
/* DefruleRunTimeInitialize:  Initializes */
/*   defrule in a run-time module.        */
/******************************************/
globle void DefruleRunTimeInitialize(
  void *theEnv,
  EXEC_STATUS,
  struct joinLink *rightPrime,
  struct joinLink *leftPrime)
  {
   struct defmodule *theModule;
   struct defrule *theRule, *theDisjunct;

   DefruleData(theEnv,execStatus)->RightPrimeJoins = rightPrime;
   DefruleData(theEnv,execStatus)->LeftPrimeJoins = leftPrime;   

   SaveCurrentModule(theEnv,execStatus);

   for (theModule = (struct defmodule *) EnvGetNextDefmodule(theEnv,execStatus,NULL);
        theModule != NULL;
        theModule = (struct defmodule *) EnvGetNextDefmodule(theEnv,execStatus,theModule))
     {
      EnvSetCurrentModule(theEnv,execStatus,(void *) theModule);
      for (theRule = EnvGetNextDefrule(theEnv,execStatus,NULL);
           theRule != NULL;
           theRule = EnvGetNextDefrule(theEnv,execStatus,theRule))
        { 
         for (theDisjunct = theRule;
              theDisjunct != NULL;
              theDisjunct = theDisjunct->disjunct)
           { AddBetaMemoriesToRule(theEnv,execStatus,theDisjunct->lastJoin); }
        }
     }
     
   RestoreCurrentModule(theEnv,execStatus);
  }


/******************************************/
/* AddBetaMemoriesToRule:     */
/******************************************/
static void AddBetaMemoriesToRule(
  void *theEnv,
  EXEC_STATUS,
  struct joinNode *theNode)
  {
   AddBetaMemoriesToJoin(theEnv,execStatus,theNode);
   
   if (theNode->lastLevel != NULL)
     { AddBetaMemoriesToRule(theEnv,execStatus,theNode->lastLevel); }
     
   if (theNode->joinFromTheRight)
     { AddBetaMemoriesToRule(theEnv,execStatus,theNode->rightSideEntryStructure); }
  }
  
#endif

#if RUN_TIME || BLOAD_ONLY || BLOAD || BLOAD_AND_BSAVE

/******************************************/
/* AddBetaMemoriesToJoin:     */
/******************************************/
globle void AddBetaMemoriesToJoin(
  void *theEnv,
  EXEC_STATUS,
  struct joinNode *theNode)
  {   
   if ((theNode->leftMemory != NULL) || (theNode->rightMemory != NULL))
     { return; }

   //if ((! theNode->firstJoin) || theNode->patternIsExists)

   if ((! theNode->firstJoin) || theNode->patternIsExists || theNode-> patternIsNegated || theNode->joinFromTheRight)
     {
      if (theNode->leftHash == NULL)
        {
         theNode->leftMemory = get_struct(theEnv,execStatus,betaMemory); 
         theNode->leftMemory->beta = (struct partialMatch **) genalloc(theEnv,execStatus,sizeof(struct partialMatch *));
         theNode->leftMemory->beta[0] = NULL;
         theNode->leftMemory->size = 1;
         theNode->leftMemory->count = 0;
         theNode->leftMemory->last = NULL;
        }
      else
        {
         theNode->leftMemory = get_struct(theEnv,execStatus,betaMemory); 
         theNode->leftMemory->beta = (struct partialMatch **) genalloc(theEnv,execStatus,sizeof(struct partialMatch *) * INITIAL_BETA_HASH_SIZE);
         memset(theNode->leftMemory->beta,0,sizeof(struct partialMatch *) * INITIAL_BETA_HASH_SIZE);
         theNode->leftMemory->size = INITIAL_BETA_HASH_SIZE;
         theNode->leftMemory->count = 0;
         theNode->leftMemory->last = NULL;
        }

 //     if (theNode->firstJoin && theNode->patternIsExists)
      if (theNode->firstJoin && (theNode->patternIsExists || theNode-> patternIsNegated || theNode->joinFromTheRight))
        {
         theNode->leftMemory->beta[0] = CreateEmptyPartialMatch(theEnv,execStatus); 
         theNode->leftMemory->beta[0]->owner = theNode;
        }
     }
   else
     { theNode->leftMemory = NULL; }

   if (theNode->joinFromTheRight)
     {
      if (theNode->leftHash == NULL)
        {
         theNode->rightMemory = get_struct(theEnv,execStatus,betaMemory); 
         theNode->rightMemory->beta = (struct partialMatch **) genalloc(theEnv,execStatus,sizeof(struct partialMatch *));
         theNode->rightMemory->last = (struct partialMatch **) genalloc(theEnv,execStatus,sizeof(struct partialMatch *));
         theNode->rightMemory->beta[0] = NULL;
         theNode->rightMemory->last[0] = NULL;
         theNode->rightMemory->size = 1;
         theNode->rightMemory->count = 0;
        }
      else
        {
         theNode->rightMemory = get_struct(theEnv,execStatus,betaMemory); 
         theNode->rightMemory->beta = (struct partialMatch **) genalloc(theEnv,execStatus,sizeof(struct partialMatch *) * INITIAL_BETA_HASH_SIZE);
         theNode->rightMemory->last = (struct partialMatch **) genalloc(theEnv,execStatus,sizeof(struct partialMatch *) * INITIAL_BETA_HASH_SIZE);
         memset(theNode->rightMemory->beta,0,sizeof(struct partialMatch **) * INITIAL_BETA_HASH_SIZE);
         memset(theNode->rightMemory->last,0,sizeof(struct partialMatch **) * INITIAL_BETA_HASH_SIZE);
         theNode->rightMemory->size = INITIAL_BETA_HASH_SIZE;
         theNode->rightMemory->count = 0;
        }
     }

   else if (theNode->firstJoin && (theNode->rightSideEntryStructure == NULL))
     {
      theNode->rightMemory = get_struct(theEnv,execStatus,betaMemory); 
      theNode->rightMemory->beta = (struct partialMatch **) genalloc(theEnv,execStatus,sizeof(struct partialMatch *));
      theNode->rightMemory->last = (struct partialMatch **) genalloc(theEnv,execStatus,sizeof(struct partialMatch *));
      theNode->rightMemory->beta[0] = CreateEmptyPartialMatch(theEnv,execStatus);
      theNode->rightMemory->beta[0]->owner = theNode;
      theNode->rightMemory->beta[0]->rhsMemory = TRUE;
      theNode->rightMemory->last[0] = theNode->rightMemory->beta[0];
      theNode->rightMemory->size = 1;
      theNode->rightMemory->count = 1;    
     }

   else
     { theNode->rightMemory = NULL; }
  }

#endif

#endif /* DEFRULE_CONSTRUCT */


