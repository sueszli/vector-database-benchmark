   /*******************************************************/
   /*      "C" Language Integrated Production System      */
   /*                                                     */
   /*             CLIPS Version 6.24  07/01/05            */
   /*                                                     */
   /*            CONSTRAINT BLOAD/BSAVE MODULE            */
   /*******************************************************/

/*************************************************************/
/* Purpose: Implements the binary save/load feature for      */
/*    constraint records.                                    */
/*                                                           */
/* Principal Programmer(s):                                  */
/*      Gary D. Riley                                        */
/*                                                           */
/* Contributing Programmer(s):                               */
/*      Brian L. Dantes                                      */
/*                                                           */
/* Revision History:                                         */
/*      6.24: Added allowed-classes slot facet.              */
/*                                                           */
/*************************************************************/

#define _CSTRNBIN_SOURCE_

#include "setup.h"

#if (BLOAD || BLOAD_ONLY || BLOAD_AND_BSAVE) && (! RUN_TIME)

#include "constant.h"
#include "envrnmnt.h"
#include "memalloc.h"
#include "router.h"
#include "bload.h"

#if BLOAD_AND_BSAVE
#include "bsave.h"
#endif

#include "cstrnbin.h"

/*******************/
/* DATA STRUCTURES */
/*******************/

struct bsaveConstraintRecord
  {
   unsigned int anyAllowed : 1;
   unsigned int symbolsAllowed : 1;
   unsigned int stringsAllowed : 1;
   unsigned int floatsAllowed : 1;
   unsigned int integersAllowed : 1;
   unsigned int instanceNamesAllowed : 1;
   unsigned int instanceAddressesAllowed : 1;
   unsigned int externalAddressesAllowed : 1;
   unsigned int factAddressesAllowed : 1;
   unsigned int anyRestriction : 1;
   unsigned int symbolRestriction : 1;
   unsigned int stringRestriction : 1;
   unsigned int numberRestriction : 1;
   unsigned int floatRestriction : 1;
   unsigned int integerRestriction : 1;
   unsigned int classRestriction : 1;
   unsigned int instanceNameRestriction : 1;
   unsigned int multifieldsAllowed : 1;
   unsigned int singlefieldsAllowed : 1;
   long classList;
   long restrictionList;
   long minValue;
   long maxValue;
   long minFields;
   long maxFields;
  };

typedef struct bsaveConstraintRecord BSAVE_CONSTRAINT_RECORD;

/***************************************/
/* LOCAL INTERNAL FUNCTION DEFINITIONS */
/***************************************/

#if BLOAD_AND_BSAVE
   static void                    CopyToBsaveConstraintRecord(void *,EXEC_STATUS,CONSTRAINT_RECORD *,BSAVE_CONSTRAINT_RECORD *);
#endif
   static void                    CopyFromBsaveConstraintRecord(void *,EXEC_STATUS,void *,long);

#if BLOAD_AND_BSAVE

/**************************************************/
/* WriteNeededConstraints: Writes the constraints */
/*   in the constraint table to the binary image  */
/*   currently being saved.                       */
/**************************************************/
globle void WriteNeededConstraints(
  void *theEnv,
  EXEC_STATUS,
  FILE *fp)
  {
   int i;
   unsigned short theIndex = 0;
   unsigned long int numberOfUsedConstraints = 0;
   CONSTRAINT_RECORD *tmpPtr;
   BSAVE_CONSTRAINT_RECORD bsaveConstraints;

   /*================================*/
   /* Get the number of constraints. */
   /*================================*/

   for (i = 0; i < SIZE_CONSTRAINT_HASH; i++)
     {
      for (tmpPtr = ConstraintData(theEnv,execStatus)->ConstraintHashtable[i];
           tmpPtr != NULL;
           tmpPtr = tmpPtr->next)
        {
         tmpPtr->bsaveIndex = theIndex++;
         numberOfUsedConstraints++;
        }
     }

   /*=============================================*/
   /* If dynamic constraint checking is disabled, */
   /* then no constraints are saved.              */
   /*=============================================*/

   if ((! EnvGetDynamicConstraintChecking(theEnv,execStatus)) && (numberOfUsedConstraints != 0))
     {
      numberOfUsedConstraints = 0;
      PrintWarningID(theEnv,execStatus,"CSTRNBIN",1,FALSE);
      EnvPrintRouter(theEnv,execStatus,WWARNING,"Constraints are not saved with a binary image\n");
      EnvPrintRouter(theEnv,execStatus,WWARNING,"  when dynamic constraint checking is disabled.\n");
     }

   /*============================================*/
   /* Write out the number of constraints in the */
   /* constraint table followed by each of the   */
   /* constraints in the constraint table.       */
   /*============================================*/

   GenWrite(&numberOfUsedConstraints,sizeof(unsigned long int),fp);
   if (numberOfUsedConstraints == 0) return;

   for (i = 0 ; i < SIZE_CONSTRAINT_HASH; i++)
     {
      for (tmpPtr = ConstraintData(theEnv,execStatus)->ConstraintHashtable[i];
           tmpPtr != NULL;
           tmpPtr = tmpPtr->next)
        {
         CopyToBsaveConstraintRecord(theEnv,execStatus,tmpPtr,&bsaveConstraints);
         GenWrite(&bsaveConstraints,sizeof(BSAVE_CONSTRAINT_RECORD),fp);
        }
     }
  }

/****************************************************/
/* CopyToBsaveConstraintRecord: Copies a constraint */
/*   record to the data structure used for storing  */
/*   constraints in a binary image.                 */
/****************************************************/
static void CopyToBsaveConstraintRecord(
  void *theEnv,
  EXEC_STATUS,
  CONSTRAINT_RECORD *constraints,
  BSAVE_CONSTRAINT_RECORD *bsaveConstraints)
  {
   bsaveConstraints->anyAllowed = constraints->anyAllowed;
   bsaveConstraints->symbolsAllowed = constraints->symbolsAllowed;
   bsaveConstraints->stringsAllowed = constraints->stringsAllowed;
   bsaveConstraints->floatsAllowed = constraints->floatsAllowed;
   bsaveConstraints->integersAllowed = constraints->integersAllowed;
   bsaveConstraints->instanceNamesAllowed = constraints->instanceNamesAllowed;
   bsaveConstraints->instanceAddressesAllowed = constraints->instanceAddressesAllowed;
   bsaveConstraints->externalAddressesAllowed = constraints->externalAddressesAllowed;
   bsaveConstraints->multifieldsAllowed = constraints->multifieldsAllowed;
   bsaveConstraints->singlefieldsAllowed = constraints->singlefieldsAllowed;
   bsaveConstraints->factAddressesAllowed = constraints->factAddressesAllowed;
   bsaveConstraints->anyRestriction = constraints->anyRestriction;
   bsaveConstraints->symbolRestriction = constraints->symbolRestriction;
   bsaveConstraints->stringRestriction = constraints->stringRestriction;
   bsaveConstraints->floatRestriction = constraints->floatRestriction;
   bsaveConstraints->integerRestriction = constraints->integerRestriction;
   bsaveConstraints->classRestriction = constraints->classRestriction;
   bsaveConstraints->instanceNameRestriction = constraints->instanceNameRestriction;

   bsaveConstraints->restrictionList = HashedExpressionIndex(theEnv,execStatus,constraints->restrictionList);
   bsaveConstraints->classList = HashedExpressionIndex(theEnv,execStatus,constraints->classList);
   bsaveConstraints->minValue = HashedExpressionIndex(theEnv,execStatus,constraints->minValue);
   bsaveConstraints->maxValue = HashedExpressionIndex(theEnv,execStatus,constraints->maxValue);
   bsaveConstraints->minFields = HashedExpressionIndex(theEnv,execStatus,constraints->minFields);
   bsaveConstraints->maxFields = HashedExpressionIndex(theEnv,execStatus,constraints->maxFields);
  }

#endif /* BLOAD_AND_BSAVE */

/********************************************************/
/* ReadNeededConstraints: Reads in the constraints used */
/*   by the binary image currently being loaded.        */
/********************************************************/
globle void ReadNeededConstraints(
  void *theEnv,
  EXEC_STATUS)
  {
   GenReadBinary(theEnv,execStatus,(void *) &ConstraintData(theEnv,execStatus)->NumberOfConstraints,sizeof(unsigned long int));
   if (ConstraintData(theEnv,execStatus)->NumberOfConstraints == 0) return;

   ConstraintData(theEnv,execStatus)->ConstraintArray = (CONSTRAINT_RECORD *)
           genalloc(theEnv,execStatus,(sizeof(CONSTRAINT_RECORD) * ConstraintData(theEnv,execStatus)->NumberOfConstraints));

   BloadandRefresh(theEnv,execStatus,ConstraintData(theEnv,execStatus)->NumberOfConstraints,sizeof(BSAVE_CONSTRAINT_RECORD),
                   CopyFromBsaveConstraintRecord);
  }

/*****************************************************/
/* CopyFromBsaveConstraintRecord: Copies values to a */
/*   constraint record from the data structure used  */
/*   for storing constraints in a binary image.      */
/*****************************************************/
static void CopyFromBsaveConstraintRecord(
  void *theEnv,
  EXEC_STATUS,
  void *buf,
  long theIndex)
  {
   BSAVE_CONSTRAINT_RECORD *bsaveConstraints;
   CONSTRAINT_RECORD *constraints;

   bsaveConstraints = (BSAVE_CONSTRAINT_RECORD *) buf;
   constraints = (CONSTRAINT_RECORD *) &ConstraintData(theEnv,execStatus)->ConstraintArray[theIndex];

   constraints->anyAllowed = bsaveConstraints->anyAllowed;
   constraints->symbolsAllowed = bsaveConstraints->symbolsAllowed;
   constraints->stringsAllowed = bsaveConstraints->stringsAllowed;
   constraints->floatsAllowed = bsaveConstraints->floatsAllowed;
   constraints->integersAllowed = bsaveConstraints->integersAllowed;
   constraints->instanceNamesAllowed = bsaveConstraints->instanceNamesAllowed;
   constraints->instanceAddressesAllowed = bsaveConstraints->instanceAddressesAllowed;
   constraints->externalAddressesAllowed = bsaveConstraints->externalAddressesAllowed;
   constraints->voidAllowed = FALSE;
   constraints->multifieldsAllowed = bsaveConstraints->multifieldsAllowed;
   constraints->singlefieldsAllowed = bsaveConstraints->singlefieldsAllowed;
   constraints->factAddressesAllowed = bsaveConstraints->factAddressesAllowed;
   constraints->anyRestriction = bsaveConstraints->anyRestriction;
   constraints->symbolRestriction = bsaveConstraints->symbolRestriction;
   constraints->stringRestriction = bsaveConstraints->stringRestriction;
   constraints->floatRestriction = bsaveConstraints->floatRestriction;
   constraints->integerRestriction = bsaveConstraints->integerRestriction;
   constraints->classRestriction = bsaveConstraints->classRestriction;
   constraints->instanceNameRestriction = bsaveConstraints->instanceNameRestriction;

   constraints->restrictionList = HashedExpressionPointer(bsaveConstraints->restrictionList);
   constraints->classList = HashedExpressionPointer(bsaveConstraints->classList);
   constraints->minValue = HashedExpressionPointer(bsaveConstraints->minValue);
   constraints->maxValue = HashedExpressionPointer(bsaveConstraints->maxValue);
   constraints->minFields = HashedExpressionPointer(bsaveConstraints->minFields);
   constraints->maxFields = HashedExpressionPointer(bsaveConstraints->maxFields);
   constraints->multifield = NULL;
  }

/********************************************************/
/* ClearBloadedConstraints: Releases memory associated  */
/*   with constraints loaded from binary image          */
/********************************************************/
globle void ClearBloadedConstraints(
  void *theEnv,
  EXEC_STATUS)
  {
   if (ConstraintData(theEnv,execStatus)->NumberOfConstraints != 0)
     {
      genfree(theEnv,execStatus,(void *) ConstraintData(theEnv,execStatus)->ConstraintArray,
                     (sizeof(CONSTRAINT_RECORD) * ConstraintData(theEnv,execStatus)->NumberOfConstraints));
      ConstraintData(theEnv,execStatus)->NumberOfConstraints = 0;
     }
  }

#endif /* (BLOAD || BLOAD_ONLY || BLOAD_AND_BSAVE) && (! RUN_TIME) */





