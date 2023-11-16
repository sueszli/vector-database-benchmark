   /*******************************************************/
   /*      "C" Language Integrated Production System      */
   /*                                                     */
   /*             CLIPS Version 6.30  10/19/06            */
   /*                                                     */
   /*                 FACT MANAGER MODULE                 */
   /*******************************************************/

/*************************************************************/
/* Purpose: Provides core routines for maintaining the fact  */
/*   list including assert/retract operations, data          */
/*   structure creation/deletion, printing, slot access,     */
/*   and other utility functions.                            */
/*                                                           */
/* Principal Programmer(s):                                  */
/*      Gary D. Riley                                        */
/*                                                           */
/* Contributing Programmer(s):                               */
/*      Brian L. Dantes                                      */
/*                                                           */
/* Revision History:                                         */
/*      6.23: Added support for templates maintaining their  */
/*            own list of facts.                             */
/*                                                           */
/*      6.24: Removed LOGICAL_DEPENDENCIES compilation flag. */
/*                                                           */
/*            Renamed BOOLEAN macro type to intBool.         */
/*                                                           */
/*            AssignFactSlotDefaults function does not       */
/*            properly handle defaults for multifield slots. */
/*            DR0869                                         */
/*                                                           */
/*            Support for ppfact command.                    */
/*                                                           */
/*************************************************************/


#define _FACTMNGR_SOURCE_

#include <stdio.h>
#define _STDIO_INCLUDED_

#include "setup.h"

#if DEFTEMPLATE_CONSTRUCT && DEFRULE_CONSTRUCT

#include "constant.h"
#include "symbol.h"
#include "memalloc.h"
#include "exprnpsr.h"
#include "argacces.h"
#include "scanner.h"
#include "router.h"
#include "strngrtr.h"
#include "match.h"
#include "factbld.h"
#include "factqury.h"
#include "reteutil.h"
#include "retract.h"
#include "factcmp.h"
#include "filecom.h"
#include "factfun.h"
#include "fact_command.h"
#include "constrct.h"
#include "factrhs.h"
#include "factmch.h"
#include "watch.h"
#include "utility.h"
#include "factbin.h"
#include "fact_manager.h"
#include "facthsh.h"
#include "default.h"
#include "commline.h"
#include "envrnmnt.h"
#include "sysdep.h"

#include "engine.h"
#include "lgcldpnd.h"
#include "drive.h"
#include "ruledlt.h"

#include "tmpltbsc.h"
#include "tmpltdef.h"
#include "tmpltutl.h"
#include "tmpltfun.h"


#include <unistd.h>
#include <assert.h>

/***************************************/
/* LOCAL INTERNAL FUNCTION DEFINITIONS */
/***************************************/

   static void                    ResetFacts(void *,EXEC_STATUS);
   static int                     ClearFactsReady(void *,EXEC_STATUS);
   static void                    RemoveGarbageFacts(void *,EXEC_STATUS);
   static void                    DeallocateFactData(void *,EXEC_STATUS);

/**************************************************************/
/* InitializeFacts: Initializes the fact data representation. */
/*   Facts are only available when both the defrule and       */
/*   deftemplate constructs are available.                    */
/**************************************************************/
globle void InitializeFacts(
  void *theEnv,
  EXEC_STATUS)
  {
   struct patternEntityRecord factInfo = { { "FACT_ADDRESS", FACT_ADDRESS,1,0,0,
                                                     PrintFactIdentifier,
                                                     PrintFactIdentifierInLongForm,
                                                     EnvRetract,
                                                     NULL,
                                                     EnvGetNextFact,
                                                     EnvIncrementFactCount,
                                                     EnvDecrementFactCount,NULL,NULL,NULL,NULL,NULL
                                                   },
                                                   DecrementFactBasisCount,
                                                   IncrementFactBasisCount,
                                                   MatchFactFunction,
                                                   NULL
                                                 };
                                                 
   struct fact dummyFact = { { NULL, NULL, 0, 0L }, NULL, NULL, -1L, 0, 0, 1,
                                  NULL, NULL, NULL, NULL, { 1, 0, 0UL, NULL, { { 0, NULL } } } };
   
   AllocateEnvironmentData(theEnv,execStatus,FACTS_DATA,sizeof(struct factsData),DeallocateFactData);

   memcpy(&FactData(theEnv,execStatus)->FactInfo,&factInfo,sizeof(struct patternEntityRecord)); 
   dummyFact.factHeader.theInfo = &FactData(theEnv,execStatus)->FactInfo;    
   memcpy(&FactData(theEnv,execStatus)->DummyFact,&dummyFact,sizeof(struct fact));  
   FactData(theEnv,execStatus)->LastModuleIndex = -1;

   /*=========================================*/
   /* Initialize the fact hash table (used to */
   /* quickly determine if a fact exists).    */
   /*=========================================*/

   InitializeFactHashTable(theEnv,execStatus);

   /*============================================*/
   /* Initialize the fact callback functions for */
   /* use with the reset and clear commands.     */
   /*============================================*/

   EnvAddResetFunction(theEnv,execStatus,"facts",ResetFacts,60);
   AddClearReadyFunction(theEnv,execStatus,"facts",ClearFactsReady,0);

   /*=============================*/
   /* Initialize periodic garbage */
   /* collection for facts.       */
   /*=============================*/

   AddCleanupFunction(theEnv,execStatus,"facts",RemoveGarbageFacts,0);

   /*===================================*/
   /* Initialize fact pattern matching. */
   /*===================================*/

   InitializeFactPatterns(theEnv,execStatus);

   /*==================================*/
   /* Initialize the facts keyword for */
   /* use with the watch command.      */
   /*==================================*/

#if DEBUGGING_FUNCTIONS
   AddWatchItem(theEnv,execStatus,"facts",0,&FactData(theEnv,execStatus)->WatchFacts,80,DeftemplateWatchAccess,DeftemplateWatchPrint);
#endif

   /*=========================================*/
   /* Initialize fact commands and functions. */
   /*=========================================*/

   FactCommandDefinitions(theEnv,execStatus);
   FactFunctionDefinitions(theEnv,execStatus);
   
   /*==============================*/
   /* Initialize fact set queries. */
   /*==============================*/
  
#if FACT_SET_QUERIES
   SetupFactQuery(theEnv,execStatus);
#endif

   /*==================================*/
   /* Initialize fact patterns for use */
   /* with the bload/bsave commands.   */
   /*==================================*/

#if (BLOAD || BLOAD_ONLY || BLOAD_AND_BSAVE) && (! RUN_TIME)
   FactBinarySetup(theEnv,execStatus);
#endif

   /*===================================*/
   /* Initialize fact patterns for use  */
   /* with the constructs-to-c command. */
   /*===================================*/

#if CONSTRUCT_COMPILER && (! RUN_TIME)
   FactPatternsCompilerSetup(theEnv,execStatus);
#endif
  }
  
/***********************************/
/* DeallocateFactData: Deallocates */
/*   environment data for facts.   */
/***********************************/
static void DeallocateFactData(
  void *theEnv,
  EXEC_STATUS)
  {
   struct factHashEntry *tmpFHEPtr, *nextFHEPtr;
   struct fact *tmpFactPtr, *nextFactPtr;
   unsigned long i;
   struct patternMatch *theMatch, *tmpMatch;
   
   for (i = 0; i < FactData(theEnv,execStatus)->FactHashTableSize; i++) 
     {
      tmpFHEPtr = FactData(theEnv,execStatus)->FactHashTable[i];
      
      while (tmpFHEPtr != NULL)
        {
         nextFHEPtr = tmpFHEPtr->next;
         rtn_struct(theEnv,execStatus,factHashEntry,tmpFHEPtr);
         tmpFHEPtr = nextFHEPtr;
        }
     }
  
   rm3(theEnv,execStatus,FactData(theEnv,execStatus)->FactHashTable,
       sizeof(struct factHashEntry *) * FactData(theEnv,execStatus)->FactHashTableSize);
                 
   tmpFactPtr = FactData(theEnv,execStatus)->FactList;
   while (tmpFactPtr != NULL)
     {
      nextFactPtr = tmpFactPtr->nextFact;

      theMatch = (struct patternMatch *) tmpFactPtr->list;        
      while (theMatch != NULL)
        {
         tmpMatch = theMatch->next;
         rtn_struct(theEnv,execStatus,patternMatch,theMatch);
         theMatch = tmpMatch;
        }

      ReturnEntityDependencies(theEnv,execStatus,(struct patternEntity *) tmpFactPtr);

      ReturnFact(theEnv,execStatus,tmpFactPtr);
      tmpFactPtr = nextFactPtr; 
     }
     
   tmpFactPtr = FactData(theEnv,execStatus)->GarbageFacts;
   while (tmpFactPtr != NULL)
     {
      nextFactPtr = tmpFactPtr->nextFact;

      ReturnFact(theEnv,execStatus,tmpFactPtr);
      tmpFactPtr = nextFactPtr; 
     }
  }

/**********************************************/
/* PrintFactWithIdentifier: Displays a single */
/*   fact preceded by its fact identifier.    */
/**********************************************/
globle void PrintFactWithIdentifier(
  void *theEnv,
  EXEC_STATUS,
  char *logicalName,
  struct fact *factPtr)
  {
   char printSpace[20];

   gensprintf(printSpace,"f-%-5lld ",factPtr->factIndex);
   EnvPrintRouter(theEnv,execStatus,logicalName,printSpace);
   PrintFact(theEnv,execStatus,logicalName,factPtr,FALSE,FALSE);
  }

/****************************************************/
/* PrintFactIdentifier: Displays a fact identifier. */
/****************************************************/
globle void PrintFactIdentifier(
  void *theEnv,
  EXEC_STATUS,
  char *logicalName,
  void *factPtr)
  {
   char printSpace[20];

   gensprintf(printSpace,"f-%lld",((struct fact *) factPtr)->factIndex);
   EnvPrintRouter(theEnv,execStatus,logicalName,printSpace);
  }

/********************************************/
/* PrintFactIdentifierInLongForm: Display a */
/*   fact identifier in a longer format.    */
/********************************************/
globle void PrintFactIdentifierInLongForm(
  void *theEnv,
  EXEC_STATUS,
  char *logicalName,
  void *factPtr)
  {
   if (PrintUtilityData(theEnv,execStatus)->AddressesToStrings) EnvPrintRouter(theEnv,execStatus,logicalName,"\"");
   if (factPtr != (void *) &FactData(theEnv,execStatus)->DummyFact)
     {
      EnvPrintRouter(theEnv,execStatus,logicalName,"<Fact-");
      PrintLongInteger(theEnv,execStatus,logicalName,((struct fact *) factPtr)->factIndex);
      EnvPrintRouter(theEnv,execStatus,logicalName,">");
     }
   else
     { EnvPrintRouter(theEnv,execStatus,logicalName,"<Dummy Fact>"); }

   if (PrintUtilityData(theEnv,execStatus)->AddressesToStrings) EnvPrintRouter(theEnv,execStatus,logicalName,"\"");
  }

/*******************************************/
/* DecrementFactBasisCount: Decrements the */
/*   partial match busy count of a fact    */
/*******************************************/
globle void DecrementFactBasisCount(
  void *theEnv,
  EXEC_STATUS,
  void *vFactPtr)
  {
   struct fact *factPtr = (struct fact *) vFactPtr;
   struct multifield *theSegment;
   int i;

   EnvDecrementFactCount(theEnv,execStatus,factPtr);

   theSegment = &factPtr->theProposition;

   for (i = 0 ; i < (int) theSegment->multifieldLength ; i++)
     {
      AtomDeinstall(theEnv,execStatus,theSegment->theFields[i].type,theSegment->theFields[i].value);
     }
  }

/*******************************************/
/* IncrementFactBasisCount: Increments the */
/*   partial match busy count of a fact.   */
/*******************************************/
globle void IncrementFactBasisCount(
  void *theEnv,
  EXEC_STATUS,
  void *vFactPtr)
  {
   struct fact *factPtr = (struct fact *) vFactPtr;
   struct multifield *theSegment;
   int i;

   EnvIncrementFactCount(theEnv,execStatus,factPtr);

   theSegment = &factPtr->theProposition;

   for (i = 0 ; i < (int) theSegment->multifieldLength ; i++)
     {
      AtomInstall(theEnv,execStatus,theSegment->theFields[i].type,theSegment->theFields[i].value);
     }
  }

/**************************************************/
/* PrintFact: Displays the printed representation */
/*   of a fact containing the relation name and   */
/*   all of the fact's slots or fields.           */
/**************************************************/
globle void PrintFact(
  void *theEnv,
  EXEC_STATUS,
  char *logicalName,
  struct fact *factPtr,
  int seperateLines,
  int ignoreDefaults)
  {
   struct multifield *theMultifield;

   /*=========================================*/
   /* Print a deftemplate (non-ordered) fact. */
   /*=========================================*/

   if (factPtr->whichDeftemplate->implied == FALSE)
     {
      PrintTemplateFact(theEnv,execStatus,logicalName,factPtr,seperateLines,ignoreDefaults);
      return;
     }

   /*==============================*/
   /* Print an ordered fact (which */
   /* has an implied deftemplate). */
   /*==============================*/

   EnvPrintRouter(theEnv,execStatus,logicalName,"(");

   EnvPrintRouter(theEnv,execStatus,logicalName,factPtr->whichDeftemplate->header.name->contents);

   theMultifield = (struct multifield *) factPtr->theProposition.theFields[0].value;
   if (theMultifield->multifieldLength != 0)
     {
      EnvPrintRouter(theEnv,execStatus,logicalName," ");
      PrintMultifield(theEnv,execStatus,logicalName,theMultifield,0,
                      (long) (theMultifield->multifieldLength - 1),
                      FALSE);
     }

   EnvPrintRouter(theEnv,execStatus,logicalName,")");
  }

/*********************************************/
/* MatchFactFunction: Filters a fact through */
/*   the appropriate fact pattern network.   */
/*********************************************/
globle void MatchFactFunction(
  void *theEnv,EXEC_STATUS,
  void *vTheFact)
  {
   struct fact *theFact = (struct fact *) vTheFact;

   FactPatternMatch(theEnv,execStatus,theFact,theFact->whichDeftemplate->patternNetwork,0,NULL,NULL);
  }

/*********************************************************/
/* EnvRetract: C access routine for the retract command. */
/*********************************************************/
globle intBool EnvRetract(
  void *theEnv,
  EXEC_STATUS,
  void *vTheFact)
  {
   struct fact *theFact = (struct fact *) vTheFact;
   struct deftemplate *theTemplate = theFact->whichDeftemplate;

   /*===========================================*/
   /* A fact can not be retracted while another */
   /* fact is being asserted or retracted.      */
   /*===========================================*/

   if (EngineData(theEnv,execStatus)->MatchOperationInProgress)
     {
      PrintErrorID(theEnv,execStatus,"FACTMNGR",1,TRUE);
      EnvPrintRouter(theEnv,execStatus,WERROR,"Facts may not be retracted during pattern-matching\n");
      return(FALSE);
     }

   /*====================================*/
   /* A NULL fact pointer indicates that */
   /* all facts should be retracted.     */
   /*====================================*/

   if (theFact == NULL)
     {
      RemoveAllFacts(theEnv,execStatus);
      return(TRUE);
     }

   /*======================================================*/
   /* Check to see if the fact has already been retracted. */
   /*======================================================*/

   if (theFact->garbage) return(FALSE);

   /*============================*/
   /* Print retraction output if */
   /* facts are being watched.   */
   /*============================*/

#if DEBUGGING_FUNCTIONS
   if (theFact->whichDeftemplate->watch)
     {
      EnvPrintRouter(theEnv,execStatus,WTRACE,"<== ");
      PrintFactWithIdentifier(theEnv,execStatus,WTRACE,theFact);
      EnvPrintRouter(theEnv,execStatus,WTRACE,"\n");
     }
#endif

   /*==================================*/
   /* Set the change flag to indicate  */
   /* the fact-list has been modified. */
   /*==================================*/

   FactData(theEnv,execStatus)->ChangeToFactList = TRUE;

   /*===============================================*/
   /* Remove any links between the fact and partial */
   /* matches in the join network. These links are  */
   /* used to keep track of logical dependencies.   */
   /*===============================================*/

   RemoveEntityDependencies(theEnv,execStatus,(struct patternEntity *) theFact);

   /*===========================================*/
   /* Remove the fact from the fact hash table. */
   /*===========================================*/

   RemoveHashedFact(theEnv,execStatus,theFact);

   /*=========================================*/
   /* Remove the fact from its template list. */
   /*=========================================*/
   
   if (theFact == theTemplate->lastFact)
     { theTemplate->lastFact = theFact->previousTemplateFact; }

   if (theFact->previousTemplateFact == NULL)
     {
      theTemplate->factList = theTemplate->factList->nextTemplateFact;
      if (theTemplate->factList != NULL)
        { theTemplate->factList->previousTemplateFact = NULL; }
     }
   else
     {
      theFact->previousTemplateFact->nextTemplateFact = theFact->nextTemplateFact;
      if (theFact->nextTemplateFact != NULL)
        { theFact->nextTemplateFact->previousTemplateFact = theFact->previousTemplateFact; }
     }
  
   /*=====================================*/
   /* Remove the fact from the fact list. */
   /*=====================================*/

   if (theFact == FactData(theEnv,execStatus)->LastFact)
     { FactData(theEnv,execStatus)->LastFact = theFact->previousFact; }

   if (theFact->previousFact == NULL)
     {
      FactData(theEnv,execStatus)->FactList = FactData(theEnv,execStatus)->FactList->nextFact;
      if (FactData(theEnv,execStatus)->FactList != NULL)
        { FactData(theEnv,execStatus)->FactList->previousFact = NULL; }
     }
   else
     {
      theFact->previousFact->nextFact = theFact->nextFact;
      if (theFact->nextFact != NULL)
        { theFact->nextFact->previousFact = theFact->previousFact; }
     }

   /*==================================*/
   /* Update busy counts and ephemeral */
   /* garbage information.             */
   /*==================================*/

   FactDeinstall(theEnv,execStatus,theFact);
   UtilityData(theEnv,execStatus)->EphemeralItemCount++;
   UtilityData(theEnv,execStatus)->EphemeralItemSize += sizeof(struct fact) + (sizeof(struct field) * theFact->theProposition.multifieldLength);

   /*========================================*/
   /* Add the fact to the fact garbage list. */
   /*========================================*/

   theFact->nextFact = FactData(theEnv,execStatus)->GarbageFacts;
   FactData(theEnv,execStatus)->GarbageFacts = theFact;
   theFact->garbage = TRUE;

   /*===================================================*/
   /* Reset the evaluation error flag since expressions */
   /* will be evaluated as part of the retract.         */
   /*===================================================*/

   SetEvaluationError(theEnv,execStatus,FALSE);

   /*===========================================*/
   /* Loop through the list of all the patterns */
   /* that matched the fact and process the     */
   /* retract operation for each one.           */
   /*===========================================*/

   EngineData(theEnv,execStatus)->MatchOperationInProgress = TRUE;
   NetworkRetract(theEnv,execStatus,(struct patternMatch *) theFact->list);
   EngineData(theEnv,execStatus)->MatchOperationInProgress = FALSE;

   /*=========================================*/
   /* Free partial matches that were released */
   /* by the retraction of the fact.          */
   /*=========================================*/

   if (EngineData(theEnv,execStatus)->ExecutingRule == NULL)
     { FlushGarbagePartialMatches(theEnv,execStatus); }

   /*=========================================*/
   /* Retract other facts that were logically */
   /* dependent on the fact just retracted.   */
   /*=========================================*/

   ForceLogicalRetractions(theEnv,execStatus);

   /*===========================================*/
   /* Force periodic cleanup if the retract was */
   /* executed from an embedded application.    */
   /*===========================================*/

   if ((execStatus->CurrentEvaluationDepth == 0) && (! CommandLineData(theEnv,execStatus)->EvaluatingTopLevelCommand) &&
       (execStatus->CurrentExpression == NULL))
     { PeriodicCleanup(theEnv,execStatus,TRUE,FALSE); }

   /*==================================*/
   /* Return TRUE to indicate the fact */
   /* was successfully retracted.      */
   /*==================================*/

   return(TRUE);
  }

/*******************************************************************/
/* RemoveGarbageFacts: Returns facts that have been retracted to   */
/*   the pool of available memory. It is necessary to postpone     */
/*   returning the facts to memory because RHS actions retrieve    */
/*   their variable bindings directly from the fact data structure */
/*   and the facts may be in use in other data structures.         */
/*******************************************************************/
static void RemoveGarbageFacts(
  void *theEnv,
  EXEC_STATUS)
  {
   struct fact *factPtr, *nextPtr, *lastPtr = NULL;

   factPtr = FactData(theEnv,execStatus)->GarbageFacts;

   while (factPtr != NULL)
     {
      nextPtr = factPtr->nextFact;
      if ((factPtr->factHeader.busyCount == 0) &&
          (((int) factPtr->depth) > execStatus->CurrentEvaluationDepth))
        {
         UtilityData(theEnv,execStatus)->EphemeralItemCount--;
         UtilityData(theEnv,execStatus)->EphemeralItemSize -= sizeof(struct fact) + (sizeof(struct field) * factPtr->theProposition.multifieldLength);
         ReturnFact(theEnv,execStatus,factPtr);
         if (lastPtr == NULL) FactData(theEnv,execStatus)->GarbageFacts = nextPtr;
         else lastPtr->nextFact = nextPtr;
        }
      else
        { lastPtr = factPtr; }

      factPtr = nextPtr;
     }
  }



/*
 * Used to transfer the parameters from EnvAssert to
 * the pattern matching and retracting process.
 */
struct paramsForFactMatchAndRetract
{
  void *theEnv;
  struct executionStatus* execStatus;
  struct fact *theFact;
  struct factPatternNode *patternPtr;
  int offset;
  struct multifieldMarker *markers;
  struct multifieldMarker *endMark;
};


/*************************************************************************/
/* ParallelFactPatternMatch: Parallel Wrapper function for queued match  */
/*                           jobs.                                       */
/*************************************************************************/
static void * APR_THREAD_FUNC ParallelFactMatchAndLogicRetract(apr_thread_t *thread, void *parameters)
{
  struct paramsForFactMatchAndRetract * const params = (struct paramsForFactMatchAndRetract*)parameters;
  
  struct executionStatus localExecStatus = { NULL };
  localExecStatus.RunningInParallel = TRUE;
  
  FactPatternMatch(params->theEnv,
                   (params->execStatus) ? params->execStatus : & localExecStatus,
                   params->theFact,
                   params->patternPtr,
                   params->offset,
                   params->markers,
                   params->endMark);

  // STEFAN: don't do that anymore for the moment
  // EngineData(params->theEnv)->MatchOperationInProgress = FALSE;
  
  
  /*===================================================*/
  /* Retract other facts that were logically dependent */
  /* on the non-existence of the fact just asserted.   */
  /*===================================================*/
  
  ForceLogicalRetractions(params->theEnv, params->execStatus);
  
  if (params->execStatus)
    free(params->execStatus);
  
  free(params);
  
  return NULL;
}

/******************************************************************/
/* SpawnMatchingTask: Put a matching operation on the task queue. */
/******************************************************************/
globle void SpawnMatchingTask(void* theEnv,EXEC_STATUS,struct fact *theFact,
                              struct factPatternNode *entryNodeOnRootLevel) {
  struct paramsForFactMatchAndRetract *parameters = 
        (struct paramsForFactMatchAndRetract *)malloc(sizeof(
                                        struct paramsForFactMatchAndRetract));

  if (!parameters) {
    SystemError(theEnv,execStatus,"malloc failed",1);
  }
  else {
    parameters->theEnv     = theEnv;
    parameters->execStatus = NULL;
    parameters->theFact    = theFact;
    parameters->patternPtr = entryNodeOnRootLevel;
    parameters->offset     = 0;
    parameters->markers    = NULL;
    parameters->endMark    = NULL;
    
    apr_status_t rv;
    rv = apr_thread_pool_push(Env(theEnv,execStatus)->matcherThreadPool,
                              ParallelFactMatchAndLogicRetract,
                              parameters,
                              0, NULL);
    if (rv) {
      SystemError(theEnv,execStatus,"Putting task on thread pool failed",1);
    }
  }
}

/********************************************************/
/* EnvAssert: C access routine for the assert function. */
/********************************************************/
globle void *EnvAssert(
  void *theEnv,
  EXEC_STATUS,
  void *vTheFact,
  int  goParallel)
  {
   unsigned long hashValue;
   unsigned long length, i;
   struct field *theField;
   struct fact *theFact = (struct fact *) vTheFact;
   intBool duplicate;

   /*==========================================*/
   /* A fact can not be asserted while another */
   /* fact is being asserted or retracted.     */
   /*==========================================*/

   if (EngineData(theEnv,execStatus)->MatchOperationInProgress)
     {
      ReturnFact(theEnv,execStatus,theFact);
      PrintErrorID(theEnv,execStatus,"FACTMNGR",2,TRUE);
      EnvPrintRouter(theEnv,execStatus,WERROR,"Facts may not be asserted during pattern-matching\n");
      return(NULL);
     }

   /*=============================================================*/
   /* Replace invalid data types in the fact with the symbol nil. */
   /*=============================================================*/

   length = theFact->theProposition.multifieldLength;
   theField = theFact->theProposition.theFields;

   for (i = 0; i < length; i++)
     {
      if (theField[i].type == RVOID)
        {
         theField[i].type = SYMBOL;
         theField[i].value = (void *) EnvAddSymbol(theEnv,execStatus,"nil");
        }
     }

   /*========================================================*/
   /* If fact assertions are being checked for duplications, */
   /* then search the fact list for a duplicate fact.        */
   /*========================================================*/

   hashValue = HandleFactDuplication(theEnv,execStatus,theFact,&duplicate);
   if (duplicate) return(NULL);

   /*==========================================================*/
   /* If necessary, add logical dependency links between the   */
   /* fact and the partial match which is its logical support. */
   /*==========================================================*/

   if (AddLogicalDependencies(theEnv,execStatus,(struct patternEntity *) theFact,FALSE) == FALSE)
     {
      ReturnFact(theEnv,execStatus,theFact);
      return(NULL);
     }

   /*======================================*/
   /* Add the fact to the fact hash table. */
   /*======================================*/

    // STEFAN: GLOBAL STUFF...
    
   AddHashedFact(theEnv,execStatus,theFact,hashValue);

   /*================================*/
   /* Add the fact to the fact list. */
   /*================================*/

    // STEFAN: GLOBAL STUFF...
    
   theFact->nextFact = NULL;
   theFact->list = NULL;
   theFact->previousFact = FactData(theEnv,execStatus)->LastFact;
   if (FactData(theEnv,execStatus)->LastFact == NULL)
     { FactData(theEnv,execStatus)->FactList = theFact; }
   else
     { FactData(theEnv,execStatus)->LastFact->nextFact = theFact; }
   FactData(theEnv,execStatus)->LastFact = theFact;

   /*====================================*/
   /* Add the fact to its template list. */
   /*====================================*/
   
    // STEFAN: GLOBAL STUFF...
    
   theFact->previousTemplateFact = theFact->whichDeftemplate->lastFact;
   theFact->nextTemplateFact = NULL;
   
   if (theFact->whichDeftemplate->lastFact == NULL)
     { theFact->whichDeftemplate->factList = theFact; }
   else
     { theFact->whichDeftemplate->lastFact->nextTemplateFact = theFact; }
     
   theFact->whichDeftemplate->lastFact = theFact;
   
   /*==================================*/
   /* Set the fact index and time tag. */
   /*==================================*/
    
    // STEFAN: GLOBAL STUFF...

   theFact->factIndex = FactData(theEnv,execStatus)->NextFactIndex++;
   theFact->factHeader.timeTag = DefruleData(theEnv,execStatus)->CurrentEntityTimeTag++;

   /*=====================*/
   /* Update busy counts. */
   /*=====================*/

   FactInstall(theEnv,execStatus,theFact);

   /*==========================*/
   /* Print assert output if   */
   /* facts are being watched. */
   /*==========================*/

#if DEBUGGING_FUNCTIONS
   if (theFact->whichDeftemplate->watch)
     {
      EnvPrintRouter(theEnv,execStatus,WTRACE,"==> ");
      PrintFactWithIdentifier(theEnv,execStatus,WTRACE,theFact);
      EnvPrintRouter(theEnv,execStatus,WTRACE,"\n");
     }
#endif

   /*==================================*/
   /* Set the change flag to indicate  */
   /* the fact-list has been modified. */
   /*==================================*/

   FactData(theEnv,execStatus)->ChangeToFactList = TRUE;

   /*==========================================*/
   /* Check for constraint errors in the fact. */
   /*==========================================*/

   CheckTemplateFact(theEnv,execStatus,theFact);

   /*===================================================*/
   /* Reset the evaluation error flag since expressions */
   /* will be evaluated as part of the assert .         */
   /*===================================================*/

   SetEvaluationError(theEnv,execStatus,FALSE);

   /*=============================================*/
   /* Pattern match the fact using the associated */
   /* deftemplate's pattern network.              */
   /*=============================================*/
  
    if (goParallel) {  
      // STEFAN: Lets go parallel
      struct factPatternNode *entryNodeOnRootLevel = 
                                    theFact->whichDeftemplate->patternNetwork;
      
      while (entryNodeOnRootLevel) {
        SpawnMatchingTask(theEnv, execStatus, theFact, entryNodeOnRootLevel);
        entryNodeOnRootLevel = entryNodeOnRootLevel->rightNode;
      }
    }
    else {
      EngineData(theEnv,execStatus)->MatchOperationInProgress = TRUE;
      
      FactPatternMatch(theEnv,execStatus,
                       theFact,
                       theFact->whichDeftemplate->patternNetwork,
                       0,
                       NULL,
                       NULL);
      
      EngineData(theEnv,execStatus)->MatchOperationInProgress = FALSE;
      
      
      /*===================================================*/
      /* Retract other facts that were logically dependent */
      /* on the non-existence of the fact just asserted.   */
      /*===================================================*/
      
      ForceLogicalRetractions(theEnv,execStatus);
    }

   /*=========================================*/
   /* Free partial matches that were released */
   /* by the assertion of the fact.           */
   /*=========================================*/

   if (EngineData(theEnv,execStatus)->ExecutingRule == NULL) FlushGarbagePartialMatches(theEnv,execStatus);

   /*==========================================*/
   /* Force periodic cleanup if the assert was */
   /* executed from an embedded application.   */
   /*==========================================*/

   if ((execStatus->CurrentEvaluationDepth == 0) && (! CommandLineData(theEnv,execStatus)->EvaluatingTopLevelCommand) &&
       (execStatus->CurrentExpression == NULL))
     { PeriodicCleanup(theEnv,execStatus,TRUE,FALSE); }

   /*===============================*/
   /* Return a pointer to the fact. */
   /*===============================*/

   return((void *) theFact);
  }

/**************************************/
/* RemoveAllFacts: Loops through the  */
/*   fact-list and removes each fact. */
/**************************************/
globle void RemoveAllFacts(
  void *theEnv,
  EXEC_STATUS)
  {
   while (FactData(theEnv,execStatus)->FactList != NULL)
     { EnvRetract(theEnv,execStatus,(void *) FactData(theEnv,execStatus)->FactList); }
  }

/************************************************/
/* EnvCreateFact: Creates a fact data structure */
/*   of the specified deftemplate.              */
/************************************************/
globle struct fact *EnvCreateFact(
  void *theEnv,
  EXEC_STATUS,
  void *vTheDeftemplate)
  {
   struct deftemplate *theDeftemplate = (struct deftemplate *) vTheDeftemplate;
   struct fact *newFact;
   int i;

   /*=================================*/
   /* A deftemplate must be specified */
   /* in order to create a fact.      */
   /*=================================*/

   if (theDeftemplate == NULL) return(NULL);

   /*============================================*/
   /* Create a fact for an explicit deftemplate. */
   /*============================================*/

   if (theDeftemplate->implied == FALSE)
     {
      newFact = CreateFactBySize(theEnv,execStatus,theDeftemplate->numberOfSlots);
      for (i = 0;
           i < (int) theDeftemplate->numberOfSlots;
           i++)
        { newFact->theProposition.theFields[i].type = RVOID; }
     }

   /*===========================================*/
   /* Create a fact for an implied deftemplate. */
   /*===========================================*/

   else
     {
      newFact = CreateFactBySize(theEnv,execStatus,1);
      newFact->theProposition.theFields[0].type = MULTIFIELD;
      newFact->theProposition.theFields[0].value = CreateMultifield2(theEnv,execStatus,0L);
     }

   /*===============================*/
   /* Return a pointer to the fact. */
   /*===============================*/

   newFact->whichDeftemplate = theDeftemplate;

   return(newFact);
  }

/******************************************/
/* EnvGetFactSlot: Returns the slot value */
/*   from the specified slot of a fact.   */
/******************************************/
globle intBool EnvGetFactSlot(
  void *theEnv,
  EXEC_STATUS,
  void *vTheFact,
  char *slotName,
  DATA_OBJECT *theValue)
  {
   struct fact *theFact = (struct fact *) vTheFact;
   struct deftemplate *theDeftemplate;
   short whichSlot;

   /*===============================================*/
   /* Get the deftemplate associated with the fact. */
   /*===============================================*/

   theDeftemplate = theFact->whichDeftemplate;

   /*==============================================*/
   /* Handle retrieving the slot value from a fact */
   /* having an implied deftemplate. An implied    */
   /* facts has a single multifield slot.          */
   /*==============================================*/

   if (theDeftemplate->implied)
     {
      if (slotName != NULL) return(FALSE);
      theValue->type = theFact->theProposition.theFields[0].type;
      theValue->value = theFact->theProposition.theFields[0].value;
      SetpDOBegin(theValue,1);
      SetpDOEnd(theValue,((struct multifield *) theValue->value)->multifieldLength);
      return(TRUE);
     }

   /*===================================*/
   /* Make sure the slot name requested */
   /* corresponds to a valid slot name. */
   /*===================================*/

   if (FindSlot(theDeftemplate,(SYMBOL_HN *) EnvAddSymbol(theEnv,execStatus,slotName),&whichSlot) == NULL)
     { return(FALSE); }

   /*======================================================*/
   /* Return the slot value. If the slot value wasn't set, */
   /* then return FALSE to indicate that an appropriate    */
   /* slot value wasn't available.                         */
   /*======================================================*/

   theValue->type = theFact->theProposition.theFields[whichSlot-1].type;
   theValue->value = theFact->theProposition.theFields[whichSlot-1].value;
   if (theValue->type == MULTIFIELD)
     {
      SetpDOBegin(theValue,1);
      SetpDOEnd(theValue,((struct multifield *) theValue->value)->multifieldLength);
     }

   if (theValue->type == RVOID) return(FALSE);

   return(TRUE);
  }

/****************************************/
/* GetFactSlot: Returns the slot value  */
/*   from the specified slot of a fact. */
/****************************************/
#if ALLOW_ENVIRONMENT_GLOBALS
globle intBool GetFactSlot(
  void *vTheFact,
  char *slotName,
  DATA_OBJECT *theValue)
  {
   return(EnvGetFactSlot(GetCurrentEnvironment(),GetCurrentExecutionStatus(),vTheFact,slotName,theValue));
  }
#endif

/***************************************/
/* EnvPutFactSlot: Sets the slot value */
/*   of the specified slot of a fact.  */
/***************************************/
globle intBool EnvPutFactSlot(
  void *theEnv,
  EXEC_STATUS,
  void *vTheFact,
  char *slotName,
  DATA_OBJECT *theValue)
  {
   struct fact *theFact = (struct fact *) vTheFact;
   struct deftemplate *theDeftemplate;
   struct templateSlot *theSlot;
   short whichSlot;

   /*===============================================*/
   /* Get the deftemplate associated with the fact. */
   /*===============================================*/

   theDeftemplate = theFact->whichDeftemplate;

   /*============================================*/
   /* Handle setting the slot value of a fact    */
   /* having an implied deftemplate. An implied  */
   /* facts has a single multifield slot.        */
   /*============================================*/

   if (theDeftemplate->implied)
     {
      if ((slotName != NULL) || (theValue->type != MULTIFIELD))
        { return(FALSE); }

      if (theFact->theProposition.theFields[0].type == MULTIFIELD)
        { ReturnMultifield(theEnv,execStatus,(struct multifield *) theFact->theProposition.theFields[0].value); }

      theFact->theProposition.theFields[0].type = theValue->type;
      theFact->theProposition.theFields[0].value = DOToMultifield(theEnv,execStatus,theValue);
      
      return(TRUE);
     }

   /*===================================*/
   /* Make sure the slot name requested */
   /* corresponds to a valid slot name. */
   /*===================================*/

   if ((theSlot = FindSlot(theDeftemplate,(SYMBOL_HN *) EnvAddSymbol(theEnv,execStatus,slotName),&whichSlot)) == NULL)
     { return(FALSE); }

   /*=============================================*/
   /* Make sure a single field value is not being */
   /* stored in a multifield slot or vice versa.  */
   /*=============================================*/

   if (((theSlot->multislot == 0) && (theValue->type == MULTIFIELD)) ||
       ((theSlot->multislot == 1) && (theValue->type != MULTIFIELD)))
     { return(FALSE); }

   /*=====================*/
   /* Set the slot value. */
   /*=====================*/

   if (theFact->theProposition.theFields[whichSlot-1].type == MULTIFIELD)
     { ReturnMultifield(theEnv,execStatus,(struct multifield *) theFact->theProposition.theFields[whichSlot-1].value); }

   theFact->theProposition.theFields[whichSlot-1].type = theValue->type;

   if (theValue->type == MULTIFIELD)
     { theFact->theProposition.theFields[whichSlot-1].value = DOToMultifield(theEnv,execStatus,theValue); }
   else
     { theFact->theProposition.theFields[whichSlot-1].value = theValue->value; }
   
   return(TRUE);
  }

/********************************************************/
/* EnvAssignFactSlotDefaults: Sets a fact's slot values */
/*   to its default value if the value of the slot has  */
/*   not yet been set.                                  */
/********************************************************/
globle intBool EnvAssignFactSlotDefaults(
  void *theEnv,
  EXEC_STATUS,
  void *vTheFact)
  {
   struct fact *theFact = (struct fact *) vTheFact;
   struct deftemplate *theDeftemplate;
   struct templateSlot *slotPtr;
   int i;
   DATA_OBJECT theResult;

   /*===============================================*/
   /* Get the deftemplate associated with the fact. */
   /*===============================================*/

   theDeftemplate = theFact->whichDeftemplate;

   /*================================================*/
   /* The value for the implied multifield slot of   */
   /* an implied deftemplate is set to a multifield  */
   /* of length zero when the fact is created.       */
   /*================================================*/

   if (theDeftemplate->implied) return(TRUE);

   /*============================================*/
   /* Loop through each slot of the deftemplate. */
   /*============================================*/

   for (i = 0, slotPtr = theDeftemplate->slotList;
        i < (int) theDeftemplate->numberOfSlots;
        i++, slotPtr = slotPtr->next)
     {
      /*===================================*/
      /* If the slot's value has been set, */
      /* then move on to the next slot.    */
      /*===================================*/

      if (theFact->theProposition.theFields[i].type != RVOID) continue;

      /*======================================================*/
      /* Assign the default value for the slot if one exists. */
      /*======================================================*/
      
      if (DeftemplateSlotDefault(theEnv,execStatus,theDeftemplate,slotPtr,&theResult,FALSE))
        {
         theFact->theProposition.theFields[i].type = theResult.type;
         theFact->theProposition.theFields[i].value = theResult.value;
        }
     }

   /*==========================================*/
   /* Return TRUE to indicate that the default */
   /* values have been successfully set.       */
   /*==========================================*/

   return(TRUE);
  }
  
/********************************************************/
/* DeftemplateSlotDefault: Determines the default value */
/*   for the specified slot of a deftemplate.           */
/********************************************************/
globle intBool DeftemplateSlotDefault(
  void *theEnv,
  EXEC_STATUS,
  struct deftemplate *theDeftemplate,
  struct templateSlot *slotPtr,
  DATA_OBJECT *theResult,
  int garbageMultifield)
  {
   /*================================================*/
   /* The value for the implied multifield slot of an */
   /* implied deftemplate does not have a default.    */
   /*=================================================*/

   if (theDeftemplate->implied) return(FALSE);

   /*===============================================*/
   /* If the (default ?NONE) attribute was declared */
   /* for the slot, then return FALSE to indicate   */
   /* the default values for the fact couldn't be   */
   /* supplied since this attribute requires that a */
   /* default value can't be used for the slot.     */
   /*===============================================*/

   if (slotPtr->noDefault) return(FALSE);

   /*==============================================*/
   /* Otherwise if a static default was specified, */
   /* use this as the default value.               */
   /*==============================================*/

   else if (slotPtr->defaultPresent)
     {
      if (slotPtr->multislot)
        {
         StoreInMultifield(theEnv,execStatus,theResult,slotPtr->defaultList,garbageMultifield);
        }
      else
        {
         theResult->type = slotPtr->defaultList->type;
         theResult->value = slotPtr->defaultList->value;
        }
     }

   /*================================================*/
   /* Otherwise if a dynamic-default was specified,  */
   /* evaluate it and use this as the default value. */
   /*================================================*/

   else if (slotPtr->defaultDynamic)
     {
      if (! EvaluateAndStoreInDataObject(theEnv,execStatus,(int) slotPtr->multislot,
                                         (EXPRESSION *) slotPtr->defaultList,
                                         theResult,garbageMultifield))
        { return(FALSE); }
     }

   /*====================================*/
   /* Otherwise derive the default value */
   /* from the slot's constraints.       */
   /*====================================*/

   else
     {
      DeriveDefaultFromConstraints(theEnv,execStatus,slotPtr->constraints,theResult,
                                  (int) slotPtr->multislot,garbageMultifield);
     }

   /*==========================================*/
   /* Return TRUE to indicate that the default */
   /* values have been successfully set.       */
   /*==========================================*/

   return(TRUE);
  }

/***************************************************************/
/* CopyFactSlotValues: Copies the slot values from one fact to */
/*   another. Both facts must have the same relation name.     */
/***************************************************************/
globle intBool CopyFactSlotValues(
  void *theEnv,
  EXEC_STATUS,
  void *vTheDestFact,
  void *vTheSourceFact)
  {
   struct fact *theDestFact = (struct fact *) vTheDestFact;
   struct fact *theSourceFact = (struct fact *) vTheSourceFact;
   struct deftemplate *theDeftemplate;
   struct templateSlot *slotPtr;
   int i;

   /*===================================*/
   /* Both facts must be the same type. */
   /*===================================*/

   theDeftemplate = theSourceFact->whichDeftemplate;
   if (theDestFact->whichDeftemplate != theDeftemplate)
     { return(FALSE); }

   /*===================================================*/
   /* Loop through each slot of the deftemplate copying */
   /* the source fact value to the destination fact.    */
   /*===================================================*/

   for (i = 0, slotPtr = theDeftemplate->slotList;
        i < (int) theDeftemplate->numberOfSlots;
        i++, slotPtr = slotPtr->next)
     {
      theDestFact->theProposition.theFields[i].type =
         theSourceFact->theProposition.theFields[i].type;
      if (theSourceFact->theProposition.theFields[i].type != MULTIFIELD)
        {
         theDestFact->theProposition.theFields[i].value =
           theSourceFact->theProposition.theFields[i].value;
        }
      else
        {
         theDestFact->theProposition.theFields[i].value =
           CopyMultifield(theEnv,execStatus,(struct multifield *) theSourceFact->theProposition.theFields[i].value);
        }
     }

   /*========================================*/
   /* Return TRUE to indicate that fact slot */
   /* values were successfully copied.       */
   /*========================================*/

   return(TRUE);
  }

/*********************************************/
/* CreateFactBySize: Allocates a fact data   */
/*   structure based on the number of slots. */
/*********************************************/
globle struct fact *CreateFactBySize(
  void *theEnv,
  EXEC_STATUS,
  unsigned size)
  {
   struct fact *theFact;
   unsigned newSize;

   if (size <= 0) newSize = 1;
   else newSize = size;

   theFact = get_var_struct(theEnv,execStatus,fact,sizeof(struct field) * (newSize - 1));

   theFact->depth = (unsigned) execStatus->CurrentEvaluationDepth;
   theFact->garbage = FALSE;
   theFact->factIndex = 0LL;
   theFact->factHeader.busyCount = 0;
   theFact->factHeader.theInfo = &FactData(theEnv,execStatus)->FactInfo;
   theFact->factHeader.dependents = NULL;
   theFact->whichDeftemplate = NULL;
   theFact->nextFact = NULL;
   theFact->previousFact = NULL;
   theFact->previousTemplateFact = NULL;
   theFact->nextTemplateFact = NULL;
   theFact->list = NULL;

   theFact->theProposition.multifieldLength = size;
   theFact->theProposition.depth = (short) execStatus->CurrentEvaluationDepth;
   theFact->theProposition.busyCount = 0;

   return(theFact);
  }

/*********************************************/
/* ReturnFact: Returns a fact data structure */
/*   to the pool of free memory.             */
/*********************************************/
globle void ReturnFact(
  void *theEnv,
  EXEC_STATUS,
  struct fact *theFact)
  {
   struct multifield *theSegment, *subSegment;
   long newSize, i;

   theSegment = &theFact->theProposition;

   for (i = 0; i < theSegment->multifieldLength; i++)
     {
      if (theSegment->theFields[i].type == MULTIFIELD)
        {
         subSegment = (struct multifield *) theSegment->theFields[i].value;
         if (subSegment->busyCount == 0)
           { ReturnMultifield(theEnv,execStatus,subSegment); }
         else
           { AddToMultifieldList(theEnv,execStatus,subSegment); }
        }
     }

   if (theFact->theProposition.multifieldLength == 0) newSize = 1;
   else newSize = theFact->theProposition.multifieldLength;
      
   rtn_var_struct(theEnv,execStatus,fact,sizeof(struct field) * (newSize - 1),theFact);
  }

/*************************************************************/
/* FactInstall: Increments the fact, deftemplate, and atomic */
/*   data value busy counts associated with the fact.        */
/*************************************************************/
globle void FactInstall(
  void *theEnv,
  EXEC_STATUS,
  struct fact *newFact)
  {
   struct multifield *theSegment;
   int i;

   FactData(theEnv,execStatus)->NumberOfFacts++;
   newFact->whichDeftemplate->busyCount++;
   theSegment = &newFact->theProposition;

   for (i = 0 ; i < (int) theSegment->multifieldLength ; i++)
     {
      AtomInstall(theEnv,execStatus,theSegment->theFields[i].type,theSegment->theFields[i].value);
     }

   newFact->factHeader.busyCount++;
  }

/***************************************************************/
/* FactDeinstall: Decrements the fact, deftemplate, and atomic */
/*   data value busy counts associated with the fact.          */
/***************************************************************/
globle void FactDeinstall(
  void *theEnv,
  EXEC_STATUS,
  struct fact *newFact)
  {
   struct multifield *theSegment;
   int i;

   FactData(theEnv,execStatus)->NumberOfFacts--;
   theSegment = &newFact->theProposition;
   newFact->whichDeftemplate->busyCount--;

   for (i = 0 ; i < (int) theSegment->multifieldLength ; i++)
     {
      AtomDeinstall(theEnv,execStatus,theSegment->theFields[i].type,theSegment->theFields[i].value);
     }

   newFact->factHeader.busyCount--;
  }

/************************************************/
/* EnvIncrementFactCount: Increments the number */
/*   of references to a specified fact.         */
/************************************************/
#if WIN_BTC
#pragma argsused
#endif
globle void EnvIncrementFactCount(
  void *theEnv,
  EXEC_STATUS,		  
  void *factPtr)
  {
#if MAC_MCW || WIN_MCW || MAC_XCD
#pragma unused(theEnv,execStatus)
#endif

   ((struct fact *) factPtr)->factHeader.busyCount++;
  }

/************************************************/
/* EnvDecrementFactCount: Decrements the number */
/*   of references to a specified fact.         */
/************************************************/
#if WIN_BTC
#pragma argsused
#endif
globle void EnvDecrementFactCount(
  void *theEnv,
  EXEC_STATUS,
  void *factPtr)
  {
#if MAC_MCW || WIN_MCW || MAC_XCD
#pragma unused(theEnv,execStatus)
#endif

   ((struct fact *) factPtr)->factHeader.busyCount--;
  }

/*********************************************************/
/* EnvGetNextFact: If passed a NULL pointer, returns the */
/*   first fact in the fact-list. Otherwise returns the  */
/*   next fact following the fact passed as an argument. */
/*********************************************************/
globle void *EnvGetNextFact(
  void *theEnv,
  EXEC_STATUS,
  void *factPtr)
  {
   if (factPtr == NULL)
     { return((void *) FactData(theEnv,execStatus)->FactList); }

   if (((struct fact *) factPtr)->garbage) return(NULL);

   return((void *) ((struct fact *) factPtr)->nextFact);
  }

/**************************************************/
/* GetNextFactInScope: Returns the next fact that */
/*   is in scope of the current module. Works in  */
/*   a similar fashion to GetNextFact, but skips  */
/*   facts that are out of scope.                 */
/**************************************************/
globle void *GetNextFactInScope(
  void *theEnv,
  EXEC_STATUS,
  void *vTheFact)
  {
   struct fact *theFact = (struct fact *) vTheFact;

   /*=======================================================*/
   /* If fact passed as an argument is a NULL pointer, then */
   /* we're just beginning a traversal of the fact list. If */
   /* the module index has changed since that last time the */
   /* fact list was traversed by this routine, then         */
   /* determine all of the deftemplates that are in scope   */
   /* of the current module.                                */
   /*=======================================================*/

   if (theFact == NULL)
     {
      theFact = FactData(theEnv,execStatus)->FactList;
      if (FactData(theEnv,execStatus)->LastModuleIndex != DefmoduleData(theEnv,execStatus)->ModuleChangeIndex)
        {
         UpdateDeftemplateScope(theEnv,execStatus);
         FactData(theEnv,execStatus)->LastModuleIndex = DefmoduleData(theEnv,execStatus)->ModuleChangeIndex;
        }
     }

   /*==================================================*/
   /* Otherwise, if the fact passed as an argument has */
   /* been retracted, then there's no way to determine */
   /* the next fact, so return a NULL pointer.         */
   /*==================================================*/

   else if (((struct fact *) theFact)->garbage)
     { return(NULL); }

   /*==================================================*/
   /* Otherwise, start the search for the next fact in */
   /* scope with the fact immediately following the    */
   /* fact passed as an argument.                      */
   /*==================================================*/

   else
     { theFact = theFact->nextFact; }

   /*================================================*/
   /* Continue traversing the fact-list until a fact */
   /* is found that's associated with a deftemplate  */
   /* that's in scope.                               */
   /*================================================*/

   while (theFact != NULL)
     {
      if (theFact->whichDeftemplate->inScope) return((void *) theFact);

      theFact = theFact->nextFact;
     }

   return(NULL);
  }

/****************************************/
/* EnvGetFactPPForm: Returns the pretty */
/*   print representation of a fact.    */
/****************************************/
globle void EnvGetFactPPForm(
  void *theEnv,
  EXEC_STATUS,
  char *buffer,
  unsigned bufferLength,
  void *theFact)
  {
   OpenStringDestination(theEnv,execStatus,"FactPPForm",buffer,bufferLength);
   PrintFactWithIdentifier(theEnv,execStatus,"FactPPForm",(struct fact *) theFact);
   CloseStringDestination(theEnv,execStatus,"FactPPForm");
  }

/**********************************/
/* EnvFactIndex: C access routine */
/*   for the fact-index function. */
/**********************************/
#if WIN_BTC
#pragma argsused
#endif
globle long long EnvFactIndex(
  void *theEnv,
  void *factPtr)
  {
#if MAC_MCW || WIN_MCW || MAC_XCD
#pragma unused(theEnv)
#endif

   return(((struct fact *) factPtr)->factIndex);
  }

/**********************************/
/* FactIndex: C access routine    */
/*   for the fact-index function. */
/**********************************/
#if ALLOW_ENVIRONMENT_GLOBALS
globle long long FactIndex(
  void *factPtr)
  {
   return(EnvFactIndex(GetCurrentEnvironment(),factPtr));
  }
#endif

/*************************************/
/* EnvAssertString: C access routine */
/*   for the assert-string function. */
/*************************************/
globle void *EnvAssertString(
  void *theEnv,
  EXEC_STATUS,
  char *theString)
  {
   struct fact *theFact;

   if ((theFact = StringToFact(theEnv,execStatus,theString)) == NULL) return(NULL);

   return((void *) EnvAssert(theEnv,execStatus,(void *) theFact, FALSE));
  }

/******************************************************/
/* EnvGetFactListChanged: Returns the flag indicating */
/*   whether a change to the fact-list has been made. */
/******************************************************/
globle int EnvGetFactListChanged(
  void *theEnv,EXEC_STATUS)
  {
   return(FactData(theEnv,execStatus)->ChangeToFactList); 
  }

/***********************************************************/
/* EnvSetFactListChanged: Sets the flag indicating whether */
/*   a change to the fact-list has been made.              */
/***********************************************************/
globle void EnvSetFactListChanged(
  void *theEnv,EXEC_STATUS,
  int value)
  {
   FactData(theEnv,execStatus)->ChangeToFactList = value;
  }

/****************************************/
/* GetNumberOfFacts: Returns the number */
/* of facts in the fact-list.           */
/****************************************/
globle unsigned long GetNumberOfFacts(
  void *theEnv,EXEC_STATUS)
  {   
   return(FactData(theEnv,execStatus)->NumberOfFacts); 
  }

/***********************************************************/
/* ResetFacts: Reset function for facts. Sets the starting */
/*   fact index to zero and removes all facts.             */
/***********************************************************/
static void ResetFacts(
  void *theEnv,EXEC_STATUS)
  {
   /*====================================*/
   /* Initialize the fact index to zero. */
   /*====================================*/

   FactData(theEnv,execStatus)->NextFactIndex = 0L;

   /*======================================*/
   /* Remove all facts from the fact list. */
   /*======================================*/

   RemoveAllFacts(theEnv,execStatus);
  }

/************************************************************/
/* ClearFactsReady: Clear ready function for facts. Returns */
/*   TRUE if facts were successfully removed and the clear  */
/*   command can continue, otherwise FALSE.                 */
/************************************************************/
static int ClearFactsReady(
  void *theEnv,EXEC_STATUS)
  {
   /*====================================*/
   /* Initialize the fact index to zero. */
   /*====================================*/

   FactData(theEnv,execStatus)->NextFactIndex = 0L;

   /*======================================*/
   /* Remove all facts from the fact list. */
   /*======================================*/

   RemoveAllFacts(theEnv,execStatus);

   /*==============================================*/
   /* If for some reason there are any facts still */
   /* remaining, don't continue with the clear.    */
   /*==============================================*/

   if (EnvGetNextFact(theEnv,execStatus,NULL) != NULL) return(FALSE);

   /*=============================*/
   /* Return TRUE to indicate the */
   /* clear command can continue. */
   /*=============================*/

   return(TRUE);
  }

/***************************************************/
/* FindIndexedFact: Returns a pointer to a fact in */
/*   the fact list with the specified fact index.  */
/***************************************************/
globle struct fact *FindIndexedFact(
  void *theEnv,
  EXEC_STATUS,
  long long factIndexSought)
  {
   struct fact *theFact;

   for (theFact = (struct fact *) EnvGetNextFact(theEnv,execStatus,NULL);
        theFact != NULL;
        theFact = (struct fact *) EnvGetNextFact(theEnv,execStatus,theFact))
     {
      if (theFact->factIndex == factIndexSought)
        { return(theFact); }
     }

   return(NULL);
  }

#endif /* DEFTEMPLATE_CONSTRUCT && DEFRULE_CONSTRUCT */

