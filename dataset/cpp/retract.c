   /*******************************************************/
   /*      "C" Language Integrated Production System      */
   /*                                                     */
   /*             CLIPS Version 6.30  10/19/06            */
   /*                                                     */
   /*                   RETRACT MODULE                    */
   /*******************************************************/

/*************************************************************/
/* Purpose:  Handles join network activity associated with   */
/*   with the removal of a data entity such as a fact or     */
/*   instance.                                               */
/*                                                           */
/* Principal Programmer(s):                                  */
/*      Gary D. Riley                                        */
/*                                                           */
/* Contributing Programmer(s):                               */
/*                                                           */
/* Revision History:                                         */
/*                                                           */
/*      6.24: Removed LOGICAL_DEPENDENCIES compilation flag. */
/*                                                           */
/*            Renamed BOOLEAN macro type to intBool.         */
/*                                                           */
/*            Rule with exists CE has incorrect activation.  */
/*            DR0867                                         */
/*                                                           */
/*      6.30: Added support for hashed alpha memories.       */
/*                                                           */
/*            Added additional developer statistics to help  */
/*            analyze join network performance.              */
/*                                                           */
/*            Removed pseudo-facts used in not CEs.          */
/*                                                           */
/*************************************************************/

#define _RETRACT_SOURCE_

#include <stdio.h>
#define _STDIO_INCLUDED_
#include <stdlib.h>

#include "setup.h"

#if DEFRULE_CONSTRUCT

#include "agenda.h"
#include "argacces.h"
#include "constant.h"
#include "drive.h"
#include "engine.h"
#include "envrnmnt.h"
#include "lgcldpnd.h"
#include "match.h"
#include "memalloc.h"
#include "network.h"
#include "reteutil.h"
#include "router.h"
#include "symbol.h"

#include "retract.h"

/***************************************/
/* LOCAL INTERNAL FUNCTION DEFINITIONS */
/***************************************/

   static void                    ReturnMarkers(void *,EXEC_STATUS,struct multifieldMarker *);
   static intBool                 FindNextConflictingMatch(void *,EXEC_STATUS,struct partialMatch *,
                                                           struct partialMatch *,
                                                           struct joinNode *,struct partialMatch *);
   static intBool                 PartialMatchDefunct(void *,EXEC_STATUS,struct partialMatch *);
   static void                    NegEntryRetractAlpha(void *,EXEC_STATUS,struct partialMatch *);
   static void                    NegEntryRetractBeta(void *,EXEC_STATUS,struct joinNode *,struct partialMatch *,
                                                      struct partialMatch *);

/************************************************************/
/* NetworkRetract:  Retracts a data entity (such as a fact  */
/*   or instance) from the pattern and join networks given  */
/*   a pointer to the list of patterns which the data       */
/*   entity matched.                                        */
/************************************************************/
globle void NetworkRetract(
  void *theEnv,
  EXEC_STATUS,
  struct patternMatch *listOfMatchedPatterns)
  {
   struct patternMatch *tempMatch, *nextMatch;

   tempMatch = listOfMatchedPatterns;
   while (tempMatch != NULL)
     {
      nextMatch = tempMatch->next;

      if (tempMatch->theMatch->children != NULL)
        { PosEntryRetractAlpha(theEnv,execStatus,tempMatch->theMatch); }

      if (tempMatch->theMatch->blockList != NULL)
        { NegEntryRetractAlpha(theEnv,execStatus,tempMatch->theMatch); }
      
      /*===================================================*/
      /* Remove from the alpha memory of the pattern node. */
      /*===================================================*/

      RemoveAlphaMemoryMatches(theEnv,execStatus,tempMatch->matchingPattern,
                               tempMatch->theMatch,
                               tempMatch->theMatch->binds[0].gm.theMatch);
      
      rtn_struct(theEnv,execStatus,patternMatch,tempMatch);

      tempMatch = nextMatch;
     }
  }

/***************************************************************/
/* PosEntryRetractAlpha:           */
/***************************************************************/
globle void PosEntryRetractAlpha(
  void *theEnv,
  EXEC_STATUS,
  struct partialMatch *alphaMatch)
  {
   struct partialMatch *betaMatch, *tempMatch;
   struct joinNode *joinPtr;
   
   betaMatch = alphaMatch->children;
   while (betaMatch != NULL)
     {
      joinPtr = (struct joinNode *) betaMatch->owner;
      
      if (betaMatch->children != NULL)
        { PosEntryRetractBeta(theEnv,execStatus,betaMatch,betaMatch->children); }

      if (betaMatch->rhsMemory)
        { NegEntryRetractAlpha(theEnv,execStatus,betaMatch); }
      
      /* Remove the beta match. */
      
	  if ((joinPtr->ruleToActivate != NULL) ?
		  (betaMatch->marker != NULL) : FALSE)
		{ RemoveActivation(theEnv,execStatus,(struct activation *) betaMatch->marker,TRUE,TRUE); }

	  tempMatch = betaMatch->nextRightChild;

	  if (betaMatch->rhsMemory)
		{ UnlinkBetaPMFromNodeAndLineage(theEnv,execStatus,joinPtr,betaMatch,RHS); }
	  else
		{ UnlinkBetaPMFromNodeAndLineage(theEnv,execStatus,joinPtr,betaMatch,LHS); }

      DeletePartialMatches(theEnv,execStatus,betaMatch);
      
      betaMatch = tempMatch;     
     }
  }
  
/***************************************************************/
/* NegEntryRetractAlpha:           */
/***************************************************************/
static void NegEntryRetractAlpha(
  void *theEnv,
  EXEC_STATUS,
  struct partialMatch *alphaMatch)
  {
   struct partialMatch *betaMatch;
   struct joinNode *joinPtr;
   
   betaMatch = alphaMatch->blockList;
   while (betaMatch != NULL)
     {
      joinPtr = (struct joinNode *) betaMatch->owner;
      
      if ((! joinPtr->patternIsNegated) &&
          (! joinPtr->patternIsExists) &&
          (! joinPtr->joinFromTheRight))
        {               
         SystemError(theEnv,execStatus,"RETRACT",117);
         betaMatch = betaMatch->nextBlocked;
         continue;
        }

      NegEntryRetractBeta(theEnv,execStatus,joinPtr,alphaMatch,betaMatch);
      betaMatch = alphaMatch->blockList;
     }
  }

/***************************************************************/
/* NegEntryRetractBeta:           */
/***************************************************************/
static void NegEntryRetractBeta(
  void *theEnv,
  EXEC_STATUS,
  struct joinNode *joinPtr,
  struct partialMatch *alphaMatch,
  struct partialMatch *betaMatch)
  {
   /*======================================================*/
   /* Try to find another RHS partial match which prevents */
   /* the LHS partial match from being satisifed.          */
   /*======================================================*/

   RemoveBlockedLink(betaMatch);

   if (FindNextConflictingMatch(theEnv,execStatus,betaMatch,alphaMatch->nextInMemory,joinPtr,alphaMatch))
     { return; }
   else if (joinPtr->patternIsExists)
     { 
      if (betaMatch->children != NULL)
        { PosEntryRetractBeta(theEnv,execStatus,betaMatch,betaMatch->children); }
      return; 
     }
   else if (joinPtr->firstJoin && (joinPtr->patternIsNegated || joinPtr->joinFromTheRight) && (! joinPtr->patternIsExists)) 
     {
      if (joinPtr->secondaryNetworkTest != NULL)
        {
         if (EvaluateSecondaryNetworkTest(theEnv,execStatus,betaMatch,joinPtr) == FALSE)
           { return; }
        }     
     
      EPMDrive(theEnv,execStatus,betaMatch,joinPtr);

      return;
     }

   if (joinPtr->secondaryNetworkTest != NULL)
     {
      if (EvaluateSecondaryNetworkTest(theEnv,execStatus,betaMatch,joinPtr) == FALSE)
        { return; }
     }     
      
   /*=========================================================*/
   /* If the LHS partial match now has no RHS partial matches */
   /* that conflict with it, then it satisfies the conditions */
   /* of the RHS not CE. Create a partial match and send it   */
   /* to the joins below.                                     */
   /*=========================================================*/
      
   /*===============================*/
   /* Create the new partial match. */
   /*===============================*/

   PPDrive(theEnv,execStatus,betaMatch,NULL,joinPtr);
  }

/***************************************************************/
/* PosEntryRetractBeta:           */
/***************************************************************/
globle void PosEntryRetractBeta(
  void *theEnv,
  EXEC_STATUS,
  struct partialMatch *parentMatch,
  struct partialMatch *betaMatch)
  {
   struct partialMatch *tempMatch;
 
   while (betaMatch != NULL)
     {
      if (betaMatch->children != NULL)
        {
         betaMatch = betaMatch->children;
         continue;
        }

      if (betaMatch->nextLeftChild != NULL)
        { tempMatch = betaMatch->nextLeftChild; }
      else
        { 
         tempMatch = betaMatch->leftParent;
         betaMatch->leftParent->children = NULL; 
        }

      if (betaMatch->blockList != NULL)
        { NegEntryRetractAlpha(theEnv,execStatus,betaMatch); }
      else if ((((struct joinNode *) betaMatch->owner)->ruleToActivate != NULL) ?
               (betaMatch->marker != NULL) : FALSE)
        { RemoveActivation(theEnv,execStatus,(struct activation *) betaMatch->marker,TRUE,TRUE); }
      
      if (betaMatch->rhsMemory)
        { UnlinkNonLeftLineage(theEnv,execStatus,(struct joinNode *) betaMatch->owner,betaMatch,RHS); }
      else
        { UnlinkNonLeftLineage(theEnv,execStatus,(struct joinNode *) betaMatch->owner,betaMatch,LHS); } 

      if (betaMatch->dependents != NULL) RemoveLogicalSupport(theEnv,execStatus,betaMatch);
      ReturnPartialMatch(theEnv,execStatus,betaMatch);
    
      if (tempMatch == parentMatch) return;
      betaMatch = tempMatch;      
     }
  }

/******************************************************************/
/* FindNextConflictingMatch: Finds the next conflicting partial   */
/*    match in the right memory of a join that prevents a partial */
/*    match in the beta memory of the join from being satisfied.  */
/******************************************************************/
static intBool FindNextConflictingMatch(
  void *theEnv,
  EXEC_STATUS,
  struct partialMatch *theBind,
  struct partialMatch *possibleConflicts,
  struct joinNode *theJoin,
  struct partialMatch *skipMatch)
  {
   int result, restore = FALSE;
   struct partialMatch *oldLHSBinds = NULL;
   struct partialMatch *oldRHSBinds = NULL;
   struct joinNode *oldJoin = NULL;

   /*====================================*/
   /* Check each of the possible partial */
   /* matches which could conflict.      */
   /*====================================*/

#if DEVELOPER
   if (possibleConflicts != NULL)
     { EngineData(theEnv,execStatus)->leftToRightLoops++; }
#endif     
   /*====================================*/
   /* Set up the evaluation environment. */
   /*====================================*/
   
   if (possibleConflicts != NULL)
     {
      oldLHSBinds = LocalEngineData(theEnv,execStatus).LHSBinds;
      oldRHSBinds = LocalEngineData(theEnv,execStatus).RHSBinds;
      oldJoin = EngineData(theEnv,execStatus)->GlobalJoin;
      LocalEngineData(theEnv,execStatus).LHSBinds = theBind;
      EngineData(theEnv,execStatus)->GlobalJoin = theJoin;
      restore = TRUE;
     }

   for (;
        possibleConflicts != NULL;
        possibleConflicts = possibleConflicts->nextInMemory)
     {
      theJoin->memoryCompares++;
      
      /*=====================================*/
      /* Initially indicate that the partial */
      /* match doesn't conflict.             */
      /*=====================================*/

      result = FALSE;

      if (skipMatch == possibleConflicts)
        { /* Do Nothing */ }
        
       /*======================================================*/
       /* 6.05 Bug Fix. It is possible that a pattern entity   */
       /* (e.g. instance) in a partial match is 'out of date'  */
       /* with respect to the lazy evaluation scheme use by    */
       /* negated patterns. In other words, the object may     */
       /* have changed since it was last pushed through the    */
       /* network, and thus the partial match may be invalid.  */
       /* If so, the partial match must be ignored here.       */
       /*======================================================*/

      else if (PartialMatchDefunct(theEnv,execStatus,possibleConflicts))
        { /* Do Nothing */ }

      /*================================================*/
      /* If the join doesn't have a network expression  */
      /* to be evaluated, then partial match conflicts. */
      /*================================================*/

      else if (theJoin->networkTest == NULL)
        { result = TRUE; }

      /*=================================================*/
      /* Otherwise, if the join has a network expression */
      /* to evaluate, then evaluate it.                  */
      /*=================================================*/

      else
        {
#if DEVELOPER
         if (theJoin->networkTest)
           { 
            EngineData(theEnv,execStatus)->leftToRightComparisons++; 
            EngineData(theEnv,execStatus)->findNextConflictingComparisons++; 
           }
#endif
         LocalEngineData(theEnv,execStatus).RHSBinds = possibleConflicts;

         result = EvaluateJoinExpression(theEnv,execStatus,theJoin->networkTest,theJoin);
         if (execStatus->EvaluationError)
           {
            result = TRUE;
            execStatus->EvaluationError = FALSE;
           }
        
#if DEVELOPER
         if (result != FALSE)
          { EngineData(theEnv,execStatus)->leftToRightSucceeds++; }
#endif
        }
        
      /*==============================================*/
      /* If the network expression evaluated to TRUE, */
      /* then partial match being examined conflicts. */
      /* Point the beta memory partial match to the   */
      /* conflicting partial match and return TRUE to */
      /* indicate a conflict was found.               */
      /*==============================================*/

      if (result != FALSE)
        {
         AddBlockedLink(theBind,possibleConflicts);
         LocalEngineData(theEnv,execStatus).LHSBinds = oldLHSBinds;
         LocalEngineData(theEnv,execStatus).RHSBinds = oldRHSBinds;
         EngineData(theEnv,execStatus)->GlobalJoin = oldJoin;
         return(TRUE);
        }
     }

   if (restore)
     {
      LocalEngineData(theEnv,execStatus).LHSBinds = oldLHSBinds;
      LocalEngineData(theEnv,execStatus).RHSBinds = oldRHSBinds;
      EngineData(theEnv,execStatus)->GlobalJoin = oldJoin;
     }

   /*========================*/
   /* No conflict was found. */
   /*========================*/

   return(FALSE);
  }

/***********************************************************/
/* PartialMatchDefunct: Determines if any pattern entities */
/*   contained within the partial match have changed since */
/*   this partial match was generated. Assumes counterf is */
/*   FALSE.                                                */
/***********************************************************/
static intBool PartialMatchDefunct(
  void *theEnv,
  EXEC_STATUS,
  struct partialMatch *thePM)
  {
   register unsigned short i;
   register struct patternEntity * thePE;

   for (i = 0 ; i < thePM->bcount ; i++)
     {
      if (thePM->binds[i].gm.theMatch == NULL) continue;
      thePE = thePM->binds[i].gm.theMatch->matchingItem;
      if (thePE && thePE->theInfo->synchronized &&
          !(*thePE->theInfo->synchronized)(theEnv,execStatus,thePE))
        return(TRUE);
     }
   return(FALSE);
  }

/***************************************************/
/* DeletePartialMatches: Returns a list of partial */
/*   matches to the pool of free memory.           */
/***************************************************/
void DeletePartialMatches(
  void *theEnv,
  EXEC_STATUS,
  struct partialMatch *listOfPMs)
  {
   struct partialMatch *nextPM;

   while (listOfPMs != NULL)
     {
      /*============================================*/
      /* Remember the next partial match to delete. */
      /*============================================*/

      nextPM = listOfPMs->nextInMemory;

      /*================================================*/
      /* Remove the links between the partial match and */
      /* any data entities that it is attached to as a  */
      /* result of a logical CE.                        */
      /*================================================*/

      if (listOfPMs->dependents != NULL) RemoveLogicalSupport(theEnv,execStatus,listOfPMs);

      /*==========================================================*/
      /* If the partial match is being deleted from a beta memory */
      /* and the partial match isn't associated with a satisfied  */
      /* not CE, then it can be immediately returned to the pool  */
      /* of free memory. Otherwise, it's could be in use (either  */
      /* to retrieve variables from the LHS or by the activation  */
      /* of the rule). Since a not CE creates a "pseudo" data     */
      /* entity, the beta partial match which stores this pseudo  */
      /* data entity can not be deleted immediately (for the same */
      /* reason an alpha memory partial match can't be deleted    */
      /* immediately).                                            */
      /*==========================================================*/

      ReturnPartialMatch(theEnv,execStatus,listOfPMs);

      /*====================================*/
      /* Move on to the next partial match. */
      /*====================================*/

      listOfPMs = nextPM;
     }
  }

/**************************************************************/
/* ReturnPartialMatch: Returns the data structures associated */
/*   with a partial match to the pool of free memory.         */
/**************************************************************/
globle void ReturnPartialMatch(
  void *theEnv,
  EXEC_STATUS,
  struct partialMatch *waste)
  {
   /*==============================================*/
   /* If the partial match is in use, then put it  */
   /* on a garbage list to be processed later when */
   /* the partial match is not in use.             */
   /*==============================================*/

   if (waste->busy)
     {
      waste->nextInMemory = EngineData(theEnv,execStatus)->GarbagePartialMatches;
      EngineData(theEnv,execStatus)->GarbagePartialMatches = waste;
      return;
     }

   /*======================================================*/
   /* If we're dealing with an alpha memory partial match, */
   /* then return the multifield markers associated with   */
   /* the partial match (if any) along with the alphaMatch */
   /* data structure.                                      */
   /*======================================================*/

   if (waste->betaMemory == FALSE)
     {
      if (waste->binds[0].gm.theMatch->markers != NULL)
        { ReturnMarkers(theEnv,execStatus,waste->binds[0].gm.theMatch->markers); }
      rm(theEnv,execStatus,waste->binds[0].gm.theMatch,(int) sizeof(struct alphaMatch));
     }

   /*=================================================*/
   /* Remove any links between the partial match and  */
   /* a data entity that were created with the use of */
   /* the logical CE.                                 */
   /*=================================================*/

   if (waste->dependents != NULL) RemovePMDependencies(theEnv,execStatus,waste);

   /*======================================================*/
   /* Return the partial match to the pool of free memory. */
   /*======================================================*/

   rtn_var_struct(theEnv,execStatus,partialMatch,(int) sizeof(struct genericMatch *) *
                  (waste->bcount - 1),
                  waste);
  }

/***************************************************************/
/* DestroyPartialMatch: Returns the data structures associated */
/*   with a partial match to the pool of free memory.          */
/***************************************************************/
globle void DestroyPartialMatch(
  void *theEnv,
  EXEC_STATUS,
  struct partialMatch *waste)
  {
   /*======================================================*/
   /* If we're dealing with an alpha memory partial match, */
   /* then return the multifield markers associated with   */
   /* the partial match (if any) along with the alphaMatch */
   /* data structure.                                      */
   /*======================================================*/

   if (waste->betaMemory == FALSE)
     {
      if (waste->binds[0].gm.theMatch->markers != NULL)
        { ReturnMarkers(theEnv,execStatus,waste->binds[0].gm.theMatch->markers); }
      rm(theEnv,execStatus,waste->binds[0].gm.theMatch,(int) sizeof(struct alphaMatch));
     }
     
   /*=================================================*/
   /* Remove any links between the partial match and  */
   /* a data entity that were created with the use of */
   /* the logical CE.                                 */
   /*=================================================*/

   if (waste->dependents != NULL) DestroyPMDependencies(theEnv,execStatus,waste);

   /*======================================================*/
   /* Return the partial match to the pool of free memory. */
   /*======================================================*/

   rtn_var_struct(theEnv,execStatus,partialMatch,(int) sizeof(struct genericMatch *) *
                  (waste->bcount - 1),
                  waste);
  }

/******************************************************/
/* ReturnMarkers: Returns a linked list of multifield */
/*   markers associated with a data entity matching a */
/*   pattern to the pool of free memory.              */
/******************************************************/
static void ReturnMarkers(
  void *theEnv,
  EXEC_STATUS,
  struct multifieldMarker *waste)
  {
   struct multifieldMarker *temp;

   while (waste != NULL)
     {
      temp = waste->next;
      rtn_struct(theEnv,execStatus,multifieldMarker,waste);
      waste = temp;
     }
  }
  
/*************************************************************/
/* FlushGarbagePartialMatches:  Returns partial matches and  */
/*   associated structures that were removed as part of a    */
/*   retraction. It is necessary to postpone returning these */
/*   structures to memory because RHS actions retrieve their */
/*   variable bindings directly from the fact and instance   */
/*   data structures through the alpha memory bindings.      */
/*************************************************************/
globle void FlushGarbagePartialMatches(
  void *theEnv,
  EXEC_STATUS)
  {
   struct partialMatch *pmPtr;
   struct alphaMatch *amPtr;

   /*===================================================*/
   /* Return the garbage partial matches collected from */
   /* the alpha memories of the pattern networks.       */
   /*===================================================*/

   while (EngineData(theEnv,execStatus)->GarbageAlphaMatches != NULL)
     {
      amPtr = EngineData(theEnv,execStatus)->GarbageAlphaMatches->next;
      rtn_struct(theEnv,execStatus,alphaMatch,EngineData(theEnv,execStatus)->GarbageAlphaMatches);
      EngineData(theEnv,execStatus)->GarbageAlphaMatches = amPtr;
     }

   /*==============================================*/
   /* Return the garbage partial matches collected */
   /* from the beta memories of the join networks. */
   /*==============================================*/

   while (EngineData(theEnv,execStatus)->GarbagePartialMatches != NULL)
     {
      /*=====================================================*/
      /* Remember the next garbage partial match to process. */
      /*=====================================================*/

      pmPtr = EngineData(theEnv,execStatus)->GarbagePartialMatches->nextInMemory;

      /*============================================*/
      /* Dispose of the garbage partial match being */
      /* examined and move on to the next one.      */
      /*============================================*/

      EngineData(theEnv,execStatus)->GarbagePartialMatches->busy = FALSE;
      ReturnPartialMatch(theEnv,execStatus,EngineData(theEnv,execStatus)->GarbagePartialMatches);
      EngineData(theEnv,execStatus)->GarbagePartialMatches = pmPtr;
     }
  }

#endif /* DEFRULE_CONSTRUCT */

