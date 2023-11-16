   /*******************************************************/
   /*      "C" Language Integrated Production System      */
   /*                                                     */
   /*             CLIPS Version 6.24  06/05/06            */
   /*                                                     */
   /*                FACT COMMANDS MODULE                 */
   /*******************************************************/

/*************************************************************/
/* Purpose: Provides the facts, assert, retract, save-facts, */
/*   load-facts, set-fact-duplication, get-fact-duplication, */
/*   assert-string, and fact-index commands and functions.   */
/*                                                           */
/* Principal Programmer(s):                                  */
/*      Gary D. Riley                                        */
/*                                                           */
/* Contributing Programmer(s):                               */
/*      Brian L. Dantes                                      */
/*                                                           */
/* Revision History:                                         */
/*      6.23: Correction for FalseSymbol/TrueSymbol. DR0859  */
/*                                                           */
/*      6.24: Added environment parameter to GenClose.       */
/*            Added environment parameter to GenOpen.        */
/*                                                           */
/*            Renamed BOOLEAN macro type to intBool.         */
/*                                                           */
/*************************************************************/

#include <stdio.h>
#define _STDIO_INCLUDED_
#include <string.h>

#include "setup.h"

#if DEFTEMPLATE_CONSTRUCT

#define _FACTCOM_SOURCE_

#include "memalloc.h"
#include "envrnmnt.h"
#include "exprnpsr.h"
#include "fact_manager.h"
#include "argacces.h"
#include "match.h"
#include "router.h"
#include "scanner.h"
#include "constant.h"
#include "factrhs.h"
#include "factmch.h"
#include "extnfunc.h"
#include "tmpltpsr.h"
#include "tmpltutl.h"
#include "facthsh.h"
#include "modulutl.h"
#include "strngrtr.h"
#include "tmpltdef.h"
#include "tmpltfun.h"
#include "sysdep.h"

#if BLOAD_AND_BSAVE || BLOAD || BLOAD_ONLY
#include "bload.h"
#endif

#include "fact_command.h"

#define INVALID     -2L
#define UNSPECIFIED -1L

# include <assert.h>
# include <unistd.h>

/***************************************/
/* LOCAL INTERNAL FUNCTION DEFINITIONS */
/***************************************/

#if (! RUN_TIME)
   static struct expr            *AssertParse(void *,EXEC_STATUS,struct expr *,char *);
#endif
#if DEBUGGING_FUNCTIONS
   static long long               GetFactsArgument(void *,EXEC_STATUS,int,int);
#endif
   static struct expr            *StandardLoadFact(void *,EXEC_STATUS,char *,struct token *);
   static DATA_OBJECT_PTR         GetSaveFactsDeftemplateNames(void *,EXEC_STATUS,struct expr *,int,int *,int *);

/***************************************/
/* FactCommandDefinitions: Initializes */
/*   fact commands and functions.      */
/***************************************/
globle void FactCommandDefinitions(
  void *theEnv,
  EXEC_STATUS)
  {
#if ! RUN_TIME
#if DEBUGGING_FUNCTIONS
   EnvDefineFunction2(theEnv,execStatus,"facts", 'v', PTIEF FactsCommand,        "FactsCommand", "*4iu");
#endif

   EnvDefineFunction(theEnv,execStatus,"assert", 'u', PTIEF AssertCommand,  "AssertCommand");
   EnvDefineFunction2(theEnv,execStatus,"retract", 'v', PTIEF RetractCommand, "RetractCommand","1*z");
   EnvDefineFunction2(theEnv,execStatus,"assert-string", 'u', PTIEF AssertStringFunction,   "AssertStringFunction", "11s");
   EnvDefineFunction2(theEnv,execStatus,"str-assert", 'u', PTIEF AssertStringFunction,   "AssertStringFunction", "11s");

   EnvDefineFunction2(theEnv,execStatus,"get-fact-duplication",'b',
                   GetFactDuplicationCommand,"GetFactDuplicationCommand", "00");
   EnvDefineFunction2(theEnv,execStatus,"set-fact-duplication",'b',
                   SetFactDuplicationCommand,"SetFactDuplicationCommand", "11");

   EnvDefineFunction2(theEnv,execStatus,"save-facts", 'b', PTIEF SaveFactsCommand, "SaveFactsCommand", "1*wk");
   EnvDefineFunction2(theEnv,execStatus,"load-facts", 'b', PTIEF LoadFactsCommand, "LoadFactsCommand", "11k");
   EnvDefineFunction2(theEnv,execStatus,"fact-index", 'g', PTIEF FactIndexFunction,"FactIndexFunction", "11y");

   AddFunctionParser(theEnv,execStatus,"assert",AssertParse);
   FuncSeqOvlFlags(theEnv,execStatus,"assert",FALSE,FALSE);
#else
#if MAC_MCW || WIN_MCW || MAC_XCD
#pragma unused(theEnv,execStatus)
#endif
#endif
  }

static void AssertOrProcessEvent(
   void *theEnv,
   EXEC_STATUS,
   DATA_OBJECT_PTR rv,
   int doAssert)
  {
   struct deftemplate *theDeftemplate;
   struct field *theField;
   DATA_OBJECT theValue;
   struct expr *theExpression;
   struct templateSlot *slotPtr;
   struct fact *newFact;
   int error = FALSE;
   int i;
   struct fact *theFact;
   
   /*===================================================*/
   /* Set the default return value to the symbol FALSE. */
   /*===================================================*/

   SetpType(rv,SYMBOL);
   SetpValue(rv,EnvFalseSymbol(theEnv,execStatus));

   /*================================*/
   /* Get the deftemplate associated */
   /* with the fact being asserted.  */
   /*================================*/

   theExpression = GetFirstArgument();
   theDeftemplate = (struct deftemplate *) theExpression->value;

   /*=======================================*/
   /* Create the fact and store the name of */
   /* the deftemplate as the 1st field.     */
   /*=======================================*/

   if (theDeftemplate->implied == FALSE)
     {
      newFact = CreateFactBySize(theEnv,execStatus,theDeftemplate->numberOfSlots);
      slotPtr = theDeftemplate->slotList;
     }
   else
     {
      newFact = CreateFactBySize(theEnv,execStatus,1);
      if (theExpression->nextArg == NULL)
        {
         newFact->theProposition.theFields[0].type = MULTIFIELD;
         newFact->theProposition.theFields[0].value = CreateMultifield2(theEnv,execStatus,0L);
        }
      slotPtr = NULL;
     }

   newFact->whichDeftemplate = theDeftemplate;

   /*===================================================*/
   /* Evaluate the expression associated with each slot */
   /* and store the result in the appropriate slot of   */
   /* the newly created fact.                           */
   /*===================================================*/

   theField = newFact->theProposition.theFields;

   for (theExpression = theExpression->nextArg, i = 0;
        theExpression != NULL;
        theExpression = theExpression->nextArg, i++)
     {
      /*===================================================*/
      /* Evaluate the expression to be stored in the slot. */
      /*===================================================*/

      EvaluateExpression(theEnv,execStatus,theExpression,&theValue);

      /*============================================================*/
      /* A multifield value can't be stored in a single field slot. */
      /*============================================================*/

      if ((slotPtr != NULL) ?
          (slotPtr->multislot == FALSE) && (theValue.type == MULTIFIELD) :
          FALSE)
        {
         MultiIntoSingleFieldSlotError(theEnv,execStatus,slotPtr,theDeftemplate);
         theValue.type = SYMBOL;
         theValue.value = EnvFalseSymbol(theEnv,execStatus);
         error = TRUE;
        }

      /*==============================*/
      /* Store the value in the slot. */
      /*==============================*/

      theField[i].type = theValue.type;
      theField[i].value = theValue.value;

      /*========================================*/
      /* Get the information for the next slot. */
      /*========================================*/

      if (slotPtr != NULL) slotPtr = slotPtr->next;
     }

   /*============================================*/
   /* If an error occured while generating the   */
   /* fact's slot values, then abort the assert. */
   /*============================================*/

   if (error)
     {
      ReturnFact(theEnv,execStatus,newFact);
      return;
     }

   /*================================*/
   /* Add the fact to the fact-list. */
   /*================================*/

   theFact = (struct fact *) EnvAssert(theEnv,execStatus,(void *) newFact, !doAssert);

   /*========================================*/
   /* The asserted fact is the return value. */
   /*========================================*/

   if (theFact != NULL)
     {
      SetpType(rv,FACT_ADDRESS);
      SetpValue(rv,(void *) theFact);
     }

   return;
  }


/***************************************/
/* AssertCommand: H/L access routine   */
/*   for the assert function.          */
/***************************************/
globle void AssertCommand(
   void *theEnv,
   EXEC_STATUS,
   DATA_OBJECT_PTR rv)
   {
     AssertOrProcessEvent(theEnv,execStatus, rv, TRUE);
   }    


/*
 * Used to transfer the parameters from EnvAssert to
 * the pattern matching and retracting process.
 */
struct paramsForProcessEvent
{
  void *theEnv;
  struct executionStatus* execStatus;
};


static void * APR_THREAD_FUNC ProcessEventOnFactThread(apr_thread_t *thread, void *parameters)
{
  struct paramsForProcessEvent* const params = (struct paramsForProcessEvent*)parameters;
  
  DATA_OBJECT result;
  AssertOrProcessEvent(params->theEnv,params->execStatus, &result, FALSE);
  
  ExpressionDeinstall(params->theEnv,params->execStatus,
                      params->execStatus->CurrentExpression);
  
  ReturnExpression(params->theEnv,params->execStatus,params->execStatus->CurrentExpression);
  //rtn_struct(theEnv,execStatus,expr,tmp);
  //free(params->execStatus->CurrentExpression);
  free(params->execStatus);
  free(params);
  
  return NULL;
}


globle void ProcessEventCommand(
   void *theEnv,EXEC_STATUS)
   {
     // STEFAN: hand over to the FactThread
       
     struct paramsForProcessEvent * parameters = (struct paramsForProcessEvent *)malloc(sizeof(struct paramsForProcessEvent));
     
     if (!parameters) {
       SystemError(theEnv,execStatus,"malloc failed",1);
     }
     else {
       parameters->theEnv = theEnv;
       
       struct executionStatus* newExecStatus = CreateExecutionStatus();
       *newExecStatus = *execStatus;  // copy the old one to the thread-local
       
       parameters->execStatus = newExecStatus;
       
       // Copy the expression to be independent of the main thread
       EXPRESSION* newExpr = PackExpression(theEnv,execStatus,
                                  parameters->execStatus->CurrentExpression);
       
       
       // increase the reference pointer, as all allocations will be released
       // by the thread that is receiving the task/job
       ExpressionInstall(theEnv,execStatus,newExpr);
       
       apr_status_t apr_rv;
       apr_rv = apr_thread_pool_push(Env(theEnv,execStatus)->factThreadPool,
                                 ProcessEventOnFactThread,
                                 parameters,
                                 0, NULL);
       if (apr_rv) {
         SystemError(theEnv,execStatus,"Putting task on thread pool failed",1);
       }
     }
     
/*     while ((apr_thread_pool_tasks_count(Env(theEnv,execStatus)->factThreadPool) > 0)
            || ((apr_thread_pool_busy_count(Env(theEnv,execStatus)->factThreadPool)) > 0)) {
       usleep(100);
     }
     
     assert(apr_thread_pool_idle_count(Env(theEnv,execStatus)->factThreadPool) == 1);*/
   }

/****************************************/
/* RetractCommand: H/L access routine   */
/*   for the retract command.           */
/****************************************/
globle void RetractCommand(
  void *theEnv,EXEC_STATUS)
  {
   long long factIndex;
   struct fact *ptr;
   struct expr *theArgument;
   DATA_OBJECT theResult;
   int argNumber;

   /*================================*/
   /* Iterate through each argument. */
   /*================================*/

   for (theArgument = GetFirstArgument(), argNumber = 1;
        theArgument != NULL;
        theArgument = GetNextArgument(theArgument), argNumber++)
     {
      /*========================*/
      /* Evaluate the argument. */
      /*========================*/

      EvaluateExpression(theEnv,execStatus,theArgument,&theResult);

      /*===============================================*/
      /* If the argument evaluates to an integer, then */
      /* it's assumed to be the fact index of the fact */
      /* to be retracted.                              */
      /*===============================================*/

      if (theResult.type == INTEGER)
        {
         /*==========================================*/
         /* A fact index must be a positive integer. */
         /*==========================================*/

         factIndex = ValueToLong(theResult.value);
         if (factIndex < 0)
           {
            ExpectedTypeError1(theEnv,execStatus,"retract",argNumber,"fact-address, fact-index, or the symbol *");
            return;
           }

         /*================================================*/
         /* See if a fact with the specified index exists. */
         /*================================================*/

         ptr = FindIndexedFact(theEnv,execStatus,factIndex);

         /*=====================================*/
         /* If the fact exists then retract it, */
         /* otherwise print an error message.   */
         /*=====================================*/

         if (ptr != NULL)
           { EnvRetract(theEnv,execStatus,(void *) ptr); }
         else
           {
            char tempBuffer[20];
            gensprintf(tempBuffer,"f-%lld",factIndex);
            CantFindItemErrorMessage(theEnv,execStatus,"fact",tempBuffer);
           }
        }

      /*===============================================*/
      /* Otherwise if the argument evaluates to a fact */
      /* address, we can directly retract it.          */
      /*===============================================*/

      else if (theResult.type == FACT_ADDRESS)
        { EnvRetract(theEnv,execStatus,theResult.value); }

      /*============================================*/
      /* Otherwise if the argument evaluates to the */
      /* symbol *, then all facts are retracted.    */
      /*============================================*/

      else if ((theResult.type == SYMBOL) ?
               (strcmp(ValueToString(theResult.value),"*") == 0) : FALSE)
        {
         RemoveAllFacts(theEnv,execStatus);
         return;
        }

      /*============================================*/
      /* Otherwise the argument has evaluated to an */
      /* illegal value for the retract command.     */
      /*============================================*/

      else
        {
         ExpectedTypeError1(theEnv,execStatus,"retract",argNumber,"fact-address, fact-index, or the symbol *");
         SetEvaluationError(theEnv,execStatus,TRUE);
        }
     }
  }

/***************************************************/
/* SetFactDuplicationCommand: H/L access routine   */
/*   for the set-fact-duplication command.         */
/***************************************************/
globle int SetFactDuplicationCommand(
  void *theEnv,EXEC_STATUS)
  {
   int oldValue;
   DATA_OBJECT theValue;

   /*=====================================================*/
   /* Get the old value of the fact duplication behavior. */
   /*=====================================================*/

   oldValue = EnvGetFactDuplication(theEnv,execStatus);

   /*============================================*/
   /* Check for the correct number of arguments. */
   /*============================================*/

   if (EnvArgCountCheck(theEnv,execStatus,"set-fact-duplication",EXACTLY,1) == -1)
     { return(oldValue); }

   /*========================*/
   /* Evaluate the argument. */
   /*========================*/

   EnvRtnUnknown(theEnv,execStatus,1,&theValue);

   /*===============================================================*/
   /* If the argument evaluated to FALSE, then the fact duplication */
   /* behavior is disabled, otherwise it is enabled.                */
   /*===============================================================*/

   if ((theValue.value == EnvFalseSymbol(theEnv,execStatus)) && (theValue.type == SYMBOL))
     { EnvSetFactDuplication(theEnv,execStatus,FALSE); }
   else
     { EnvSetFactDuplication(theEnv,execStatus,TRUE); }

   /*========================================================*/
   /* Return the old value of the fact duplication behavior. */
   /*========================================================*/

   return(oldValue);
  }

/***************************************************/
/* GetFactDuplicationCommand: H/L access routine   */
/*   for the get-fact-duplication command.         */
/***************************************************/
globle int GetFactDuplicationCommand(
  void *theEnv,EXEC_STATUS)
  {
   int currentValue;

   /*=========================================================*/
   /* Get the current value of the fact duplication behavior. */
   /*=========================================================*/

   currentValue = EnvGetFactDuplication(theEnv,execStatus);

   /*============================================*/
   /* Check for the correct number of arguments. */
   /*============================================*/

   if (EnvArgCountCheck(theEnv,execStatus,"get-fact-duplication",EXACTLY,0) == -1)
     { return(currentValue); }

   /*============================================================*/
   /* Return the current value of the fact duplication behavior. */
   /*============================================================*/

   return(currentValue);
  }

/*******************************************/
/* FactIndexFunction: H/L access routine   */
/*   for the fact-index function.          */
/*******************************************/
globle long long FactIndexFunction(
  void *theEnv,EXEC_STATUS)
  {
   DATA_OBJECT item;

   /*============================================*/
   /* Check for the correct number of arguments. */
   /*============================================*/

   if (EnvArgCountCheck(theEnv,execStatus,"fact-index",EXACTLY,1) == -1) return(-1LL);

   /*========================*/
   /* Evaluate the argument. */
   /*========================*/

   EnvRtnUnknown(theEnv,execStatus,1,&item);

   /*======================================*/
   /* The argument must be a fact address. */
   /*======================================*/

   if (GetType(item) != FACT_ADDRESS)
     {
      ExpectedTypeError1(theEnv,execStatus,"fact-index",1,"fact-address");
      return(-1L);
     }

   /*================================================*/
   /* Return the fact index associated with the fact */
   /* address. If the fact has been retracted, then  */
   /* return -1 for the fact index.                  */
   /*================================================*/

   if (((struct fact *) GetValue(item))->garbage) return(-1LL);

   return (EnvFactIndex(theEnv,GetValue(item)));
  }

#if DEBUGGING_FUNCTIONS

/**************************************/
/* FactsCommand: H/L access routine   */
/*   for the facts command.           */
/**************************************/
globle void FactsCommand(
  void *theEnv,EXEC_STATUS)
  {
   int argumentCount;
   long long start = UNSPECIFIED, end = UNSPECIFIED, max = UNSPECIFIED;
   struct defmodule *theModule;
   DATA_OBJECT theValue;
   int argOffset;

   /*=========================================================*/
   /* Determine the number of arguments to the facts command. */
   /*=========================================================*/

   if ((argumentCount = EnvArgCountCheck(theEnv,execStatus,"facts",NO_MORE_THAN,4)) == -1) return;

   /*==================================*/
   /* The default module for the facts */
   /* command is the current module.   */
   /*==================================*/

   theModule = ((struct defmodule *) EnvGetCurrentModule(theEnv,execStatus));

   /*==========================================*/
   /* If no arguments were specified, then use */
   /* the default values to list the facts.    */
   /*==========================================*/

   if (argumentCount == 0)
     {
      EnvFacts(theEnv,execStatus,WDISPLAY,theModule,start,end,max);
      return;
     }

   /*========================================================*/
   /* Since there are one or more arguments, see if a module */
   /* or start index was specified as the first argument.    */
   /*========================================================*/

   EnvRtnUnknown(theEnv,execStatus,1,&theValue);

   /*===============================================*/
   /* If the first argument is a symbol, then check */
   /* to see that a valid module was specified.     */
   /*===============================================*/

   if (theValue.type == SYMBOL)
     {
      theModule = (struct defmodule *) EnvFindDefmodule(theEnv,execStatus,ValueToString(theValue.value));
      if ((theModule == NULL) && (strcmp(ValueToString(theValue.value),"*") != 0))
        {
         SetEvaluationError(theEnv,execStatus,TRUE);
         CantFindItemErrorMessage(theEnv,execStatus,"defmodule",ValueToString(theValue.value));
         return;
        }

      if ((start = GetFactsArgument(theEnv,execStatus,2,argumentCount)) == INVALID) return;

      argOffset = 1;
     }

   /*================================================*/
   /* Otherwise if the first argument is an integer, */
   /* check to see that a valid index was specified. */
   /*================================================*/

   else if (theValue.type == INTEGER)
     {
      start = DOToLong(theValue);
      if (start < 0)
        {
         ExpectedTypeError1(theEnv,execStatus,"facts",1,"symbol or positive number");
         SetHaltExecution(theEnv,execStatus,TRUE);
         SetEvaluationError(theEnv,execStatus,TRUE);
         return;
        }
      argOffset = 0;
     }

   /*==========================================*/
   /* Otherwise the first argument is invalid. */
   /*==========================================*/

   else
     {
      ExpectedTypeError1(theEnv,execStatus,"facts",1,"symbol or positive number");
      SetHaltExecution(theEnv,execStatus,TRUE);
      SetEvaluationError(theEnv,execStatus,TRUE);
      return;
     }

   /*==========================*/
   /* Get the other arguments. */
   /*==========================*/

   if ((end = GetFactsArgument(theEnv,execStatus,2 + argOffset,argumentCount)) == INVALID) return;
   if ((max = GetFactsArgument(theEnv,execStatus,3 + argOffset,argumentCount)) == INVALID) return;

   /*=================*/
   /* List the facts. */
   /*=================*/

   EnvFacts(theEnv,execStatus,WDISPLAY,theModule,start,end,max);
  }

/*****************************************************/
/* EnvFacts: C access routine for the facts command. */
/*****************************************************/
globle void EnvFacts(
  void *theEnv,EXEC_STATUS,
  char *logicalName,
  void *vTheModule,
  long long start,
  long long end,
  long long max)
  {
   struct fact *factPtr;
   long count = 0;
   struct defmodule *oldModule, *theModule = (struct defmodule *) vTheModule;
   int allModules = FALSE;

   /*==========================*/
   /* Save the current module. */
   /*==========================*/

   oldModule = ((struct defmodule *) EnvGetCurrentModule(theEnv,execStatus));

   /*=========================================================*/
   /* Determine if facts from all modules are to be displayed */
   /* or just facts from the current module.                  */
   /*=========================================================*/

   if (theModule == NULL) allModules = TRUE;
   else EnvSetCurrentModule(theEnv,execStatus,(void *) theModule);

   /*=====================================*/
   /* Get the first fact to be displayed. */
   /*=====================================*/

   if (allModules) factPtr = (struct fact *) EnvGetNextFact(theEnv,execStatus,NULL);
   else factPtr = (struct fact *) GetNextFactInScope(theEnv,execStatus,NULL);

   /*===============================*/
   /* Display facts until there are */
   /* no more facts to display.     */
   /*===============================*/

   while (factPtr != NULL)
     {
      /*==================================================*/
      /* Abort the display of facts if the Halt Execution */
      /* flag has been set (normally by user action).     */
      /*==================================================*/

      if (GetHaltExecution(theEnv,execStatus) == TRUE)
        {
         EnvSetCurrentModule(theEnv,execStatus,(void *) oldModule);
         return;
        }

      /*===============================================*/
      /* If the maximum fact index of facts to display */
      /* has been reached, then stop displaying facts. */
      /*===============================================*/

      if ((factPtr->factIndex > end) && (end != UNSPECIFIED))
        {
         PrintTally(theEnv,execStatus,logicalName,count,"fact","facts");
         EnvSetCurrentModule(theEnv,execStatus,(void *) oldModule);
         return;
        }

      /*================================================*/
      /* If the maximum number of facts to be displayed */
      /* has been reached, then stop displaying facts.  */
      /*================================================*/

      if (max == 0)
        {
         PrintTally(theEnv,execStatus,logicalName,count,"fact","facts");
         EnvSetCurrentModule(theEnv,execStatus,(void *) oldModule);
         return;
        }

      /*======================================================*/
      /* If the index of the fact is greater than the minimum */
      /* starting fact index, then display the fact.          */
      /*======================================================*/

      if (factPtr->factIndex >= start)
        {
         PrintFactWithIdentifier(theEnv,execStatus,logicalName,factPtr);
         EnvPrintRouter(theEnv,execStatus,logicalName,"\n");
         count++;
         if (max > 0) max--;
        }

      /*========================================*/
      /* Proceed to the next fact to be listed. */
      /*========================================*/

      if (allModules) factPtr = (struct fact *) EnvGetNextFact(theEnv,execStatus,factPtr);
      else factPtr = (struct fact *) GetNextFactInScope(theEnv,execStatus,factPtr);
     }

   /*===================================================*/
   /* Print the total of the number of facts displayed. */
   /*===================================================*/

   PrintTally(theEnv,execStatus,logicalName,count,"fact","facts");

   /*=============================*/
   /* Restore the current module. */
   /*=============================*/

   EnvSetCurrentModule(theEnv,execStatus,(void *) oldModule);
  }

/****************************************************************/
/* GetFactsArgument: Returns an argument for the facts command. */
/*  A return value of -1 indicates that no value was specified. */
/*  A return value of -2 indicates that the value specified is  */
/*  invalid.                                                    */
/****************************************************************/
static long long GetFactsArgument(
  void *theEnv,
  EXEC_STATUS,
  int whichOne,
  int argumentCount)
  {
   long long factIndex;
   DATA_OBJECT theValue;

   if (whichOne > argumentCount) return(UNSPECIFIED);

   if (EnvArgTypeCheck(theEnv,execStatus,"facts",whichOne,INTEGER,&theValue) == FALSE) return(INVALID);

   factIndex = DOToLong(theValue);

   if (factIndex < 0)
     {
      ExpectedTypeError1(theEnv,execStatus,"facts",whichOne,"positive number");
      SetHaltExecution(theEnv,execStatus,TRUE);
      SetEvaluationError(theEnv,execStatus,TRUE);
      return(INVALID);
     }

   return(factIndex);
  }

#endif /* DEBUGGING_FUNCTIONS */

/**********************************************/
/* AssertStringFunction: H/L access routine   */
/*   for the assert-string function.          */
/**********************************************/
globle void AssertStringFunction(
  void *theEnv,
  EXEC_STATUS,
  DATA_OBJECT_PTR returnValue)
  {
   DATA_OBJECT argPtr;
   struct fact *theFact;

   /*===================================================*/
   /* Set the default return value to the symbol FALSE. */
   /*===================================================*/

   SetpType(returnValue,SYMBOL);
   SetpValue(returnValue,EnvFalseSymbol(theEnv,execStatus));

   /*=====================================================*/
   /* Check for the correct number and type of arguments. */
   /*=====================================================*/

   if (EnvArgCountCheck(theEnv,execStatus,"assert-string",EXACTLY,1) == -1) return;

   if (EnvArgTypeCheck(theEnv,execStatus,"assert-string",1,STRING,&argPtr) == FALSE)
     { return; }

   /*==========================================*/
   /* Call the driver routine for converting a */
   /* string to a fact and then assert it.     */
   /*==========================================*/

   theFact = (struct fact *) EnvAssertString(theEnv,execStatus,DOToString(argPtr));
   if (theFact != NULL)
     {
      SetpType(returnValue,FACT_ADDRESS);
      SetpValue(returnValue,(void *) theFact);
     }

   return;
  }

/******************************************/
/* SaveFactsCommand: H/L access routine   */
/*   for the save-facts command.          */
/******************************************/
globle int SaveFactsCommand(
  void *theEnv,
  EXEC_STATUS)
  {
   char *fileName;
   int numArgs, saveCode = LOCAL_SAVE;
   char *argument;
   DATA_OBJECT theValue;
   struct expr *theList = NULL;

   /*============================================*/
   /* Check for the correct number of arguments. */
   /*============================================*/

   if ((numArgs = EnvArgCountCheck(theEnv,execStatus,"save-facts",AT_LEAST,1)) == -1) return(FALSE);

   /*=================================================*/
   /* Get the file name to which facts will be saved. */
   /*=================================================*/

   if ((fileName = GetFileName(theEnv,execStatus,"save-facts",1)) == NULL) return(FALSE);

   /*=============================================================*/
   /* If specified, the second argument to save-facts indicates   */
   /* whether just facts local to the current module or all facts */
   /* visible to the current module will be saved.                */
   /*=============================================================*/

   if (numArgs > 1)
     {
      if (EnvArgTypeCheck(theEnv,execStatus,"save-facts",2,SYMBOL,&theValue) == FALSE) return(FALSE);

      argument = DOToString(theValue);

      if (strcmp(argument,"local") == 0)
        { saveCode = LOCAL_SAVE; }
      else if (strcmp(argument,"visible") == 0)
        { saveCode = VISIBLE_SAVE; }
      else
        {
         ExpectedTypeError1(theEnv,execStatus,"save-facts",2,"symbol with value local or visible");
         return(FALSE);
        }
     }

   /*======================================================*/
   /* Subsequent arguments indicate that only those facts  */
   /* associated with the specified deftemplates should be */
   /* saved to the file.                                   */
   /*======================================================*/

   if (numArgs > 2) theList = GetFirstArgument()->nextArg->nextArg;

   /*====================================*/
   /* Call the SaveFacts driver routine. */
   /*====================================*/

   if (EnvSaveFacts(theEnv,execStatus,fileName,saveCode,theList) == FALSE)
     { return(FALSE); }

   return(TRUE);
  }

/******************************************/
/* LoadFactsCommand: H/L access routine   */
/*   for the load-facts command.          */
/******************************************/
globle int LoadFactsCommand(
  void *theEnv,EXEC_STATUS)
  {
   char *fileName;

   /*============================================*/
   /* Check for the correct number of arguments. */
   /*============================================*/

   if (EnvArgCountCheck(theEnv,execStatus,"load-facts",EXACTLY,1) == -1) return(FALSE);

   /*====================================================*/
   /* Get the file name from which facts will be loaded. */
   /*====================================================*/

   if ((fileName = GetFileName(theEnv,execStatus,"load-facts",1)) == NULL) return(FALSE);

   /*====================================*/
   /* Call the LoadFacts driver routine. */
   /*====================================*/

   if (EnvLoadFacts(theEnv,execStatus,fileName) == FALSE) return(FALSE);

   return(TRUE);
  }

/**************************************************************/
/* EnvSaveFacts: C access routine for the save-facts command. */
/**************************************************************/
globle intBool EnvSaveFacts(
  void *theEnv,EXEC_STATUS,
  char *fileName,
  int saveCode,
  struct expr *theList)
  {
   int tempValue1, tempValue2, tempValue3;
   struct fact *theFact;
   FILE *filePtr;
   struct defmodule *theModule;
   DATA_OBJECT_PTR theDOArray;
   int count, i, printFact, error;

   /*======================================================*/
   /* Open the file. Use either "fast save" or I/O Router. */
   /*======================================================*/

   if ((filePtr = GenOpen(theEnv,execStatus,fileName,"w")) == NULL)
     {
      OpenErrorMessage(theEnv,execStatus,"save-facts",fileName);
      return(FALSE);
     }

   SetFastSave(theEnv,execStatus,filePtr);

   /*===========================================*/
   /* Set the print flags so that addresses and */
   /* strings are printed properly to the file. */
   /*===========================================*/

   tempValue1 = PrintUtilityData(theEnv,execStatus)->PreserveEscapedCharacters;
   PrintUtilityData(theEnv,execStatus)->PreserveEscapedCharacters = TRUE;
   tempValue2 = PrintUtilityData(theEnv,execStatus)->AddressesToStrings;
   PrintUtilityData(theEnv,execStatus)->AddressesToStrings = TRUE;
   tempValue3 = PrintUtilityData(theEnv,execStatus)->InstanceAddressesToNames;
   PrintUtilityData(theEnv,execStatus)->InstanceAddressesToNames = TRUE;

   /*===================================================*/
   /* Determine the list of specific facts to be saved. */
   /*===================================================*/

   theDOArray = GetSaveFactsDeftemplateNames(theEnv,execStatus,theList,saveCode,&count,&error);

   if (error)
     {
      PrintUtilityData(theEnv,execStatus)->PreserveEscapedCharacters = tempValue1;
      PrintUtilityData(theEnv,execStatus)->AddressesToStrings = tempValue2;
      PrintUtilityData(theEnv,execStatus)->InstanceAddressesToNames = tempValue3;
      GenClose(theEnv,execStatus,filePtr);
      SetFastSave(theEnv,execStatus,NULL);
      return(FALSE);
     }

   /*=================*/
   /* Save the facts. */
   /*=================*/

   theModule = ((struct defmodule *) EnvGetCurrentModule(theEnv,execStatus));

   for (theFact = (struct fact *) GetNextFactInScope(theEnv,execStatus,NULL);
        theFact != NULL;
        theFact = (struct fact *) GetNextFactInScope(theEnv,execStatus,theFact))
     {
      /*===========================================================*/
      /* If we're doing a local save and the facts's corresponding */
      /* deftemplate isn't in the current module, then don't save  */
      /* the fact.                                                 */
      /*===========================================================*/

      if ((saveCode == LOCAL_SAVE) &&
          (theFact->whichDeftemplate->header.whichModule->theModule != theModule))
        { printFact = FALSE; }

      /*=====================================================*/
      /* Otherwise, if the list of facts to be printed isn't */
      /* restricted, then set the print flag to TRUE.        */
      /*=====================================================*/

      else if (theList == NULL)
        { printFact = TRUE; }

      /*=======================================================*/
      /* Otherwise see if the fact's corresponding deftemplate */
      /* is in the list of deftemplates whose facts are to be  */
      /* saved. If it's in the list, then set the print flag   */
      /* to TRUE, otherwise set it to FALSE.                   */
      /*=======================================================*/

      else
        {
         printFact = FALSE;
         for (i = 0; i < count; i++)
           {
            if (theDOArray[i].value == (void *) theFact->whichDeftemplate)
              {
               printFact = TRUE;
               break;
              }
           }
        }

      /*===================================*/
      /* If the print flag is set to TRUE, */
      /* then save the fact to the file.   */
      /*===================================*/

      if (printFact)
        {
         PrintFact(theEnv,execStatus,(char *) filePtr,theFact,FALSE,FALSE);
         EnvPrintRouter(theEnv,execStatus,(char *) filePtr,"\n");
        }
     }

   /*==========================*/
   /* Restore the print flags. */
   /*==========================*/

   PrintUtilityData(theEnv,execStatus)->PreserveEscapedCharacters = tempValue1;
   PrintUtilityData(theEnv,execStatus)->AddressesToStrings = tempValue2;
   PrintUtilityData(theEnv,execStatus)->InstanceAddressesToNames = tempValue3;

   /*=================*/
   /* Close the file. */
   /*=================*/

   GenClose(theEnv,execStatus,filePtr);
   SetFastSave(theEnv,execStatus,NULL);

   /*==================================*/
   /* Free the deftemplate name array. */
   /*==================================*/

   if (theList != NULL) rm3(theEnv,execStatus,theDOArray,(long) sizeof(DATA_OBJECT) * count);

   /*===================================*/
   /* Return TRUE to indicate no errors */
   /* occurred while saving the facts.  */
   /*===================================*/

   return(TRUE);
  }

/*******************************************************************/
/* GetSaveFactsDeftemplateNames: Retrieves the list of deftemplate */
/*   names for saving specific facts with the save-facts command.  */
/*******************************************************************/
static DATA_OBJECT_PTR GetSaveFactsDeftemplateNames(
  void *theEnv,EXEC_STATUS,
  struct expr *theList,
  int saveCode,
  int *count,
  int *error)
  {
   struct expr *tempList;
   DATA_OBJECT_PTR theDOArray;
   int i, tempCount;
   struct deftemplate *theDeftemplate = NULL;

   /*=============================*/
   /* Initialize the error state. */
   /*=============================*/

   *error = FALSE;

   /*=====================================================*/
   /* If no deftemplate names were specified as arguments */
   /* then the deftemplate name list is empty.            */
   /*=====================================================*/

   if (theList == NULL)
     {
      *count = 0;
      return(NULL);
     }

   /*======================================*/
   /* Determine the number of deftemplate  */
   /* names to be stored in the name list. */
   /*======================================*/

   for (tempList = theList, *count = 0;
        tempList != NULL;
        tempList = tempList->nextArg, (*count)++)
     { /* Do Nothing */ }

   /*=========================================*/
   /* Allocate the storage for the name list. */
   /*=========================================*/

   theDOArray = (DATA_OBJECT_PTR) gm3(theEnv,execStatus,(long) sizeof(DATA_OBJECT) * *count);

   /*=====================================*/
   /* Loop through each of the arguments. */
   /*=====================================*/

   for (tempList = theList, i = 0;
        i < *count;
        tempList = tempList->nextArg, i++)
     {
      /*========================*/
      /* Evaluate the argument. */
      /*========================*/

      EvaluateExpression(theEnv,execStatus,tempList,&theDOArray[i]);

      if (execStatus->EvaluationError)
        {
         *error = TRUE;
         rm3(theEnv,execStatus,theDOArray,(long) sizeof(DATA_OBJECT) * *count);
         return(NULL);
        }

      /*======================================*/
      /* A deftemplate name must be a symbol. */
      /*======================================*/

      if (theDOArray[i].type != SYMBOL)
        {
         *error = TRUE;
         ExpectedTypeError1(theEnv,execStatus,"save-facts",3+i,"symbol");
         rm3(theEnv,execStatus,theDOArray,(long) sizeof(DATA_OBJECT) * *count);
         return(NULL);
        }

      /*===================================================*/
      /* Find the deftemplate. For a local save, look only */
      /* in the current module. For a visible save, look   */
      /* in all visible modules.                           */
      /*===================================================*/

      if (saveCode == LOCAL_SAVE)
        {
         theDeftemplate = (struct deftemplate *)
                         EnvFindDeftemplate(theEnv,execStatus,ValueToString(theDOArray[i].value));
         if (theDeftemplate == NULL)
           {
            *error = TRUE;
            ExpectedTypeError1(theEnv,execStatus,"save-facts",3+i,"local deftemplate name");
            rm3(theEnv,execStatus,theDOArray,(long) sizeof(DATA_OBJECT) * *count);
            return(NULL);
           }
        }
      else if (saveCode == VISIBLE_SAVE)
        {
         theDeftemplate = (struct deftemplate *)
           FindImportedConstruct(theEnv,execStatus,"deftemplate",NULL,
                                 ValueToString(theDOArray[i].value),
                                 &tempCount,TRUE,NULL);
         if (theDeftemplate == NULL)
           {
            *error = TRUE;
            ExpectedTypeError1(theEnv,execStatus,"save-facts",3+i,"visible deftemplate name");
            rm3(theEnv,execStatus,theDOArray,(long) sizeof(DATA_OBJECT) * *count);
            return(NULL);
           }
        }

      /*==================================*/
      /* Add a pointer to the deftemplate */
      /* to the array being created.      */
      /*==================================*/

      theDOArray[i].type = DEFTEMPLATE_PTR;
      theDOArray[i].value = (void *) theDeftemplate;
     }

   /*===================================*/
   /* Return the array of deftemplates. */
   /*===================================*/

   return(theDOArray);
  }

/**************************************************************/
/* EnvLoadFacts: C access routine for the load-facts command. */
/**************************************************************/
globle intBool EnvLoadFacts(
  void *theEnv,EXEC_STATUS,
  char *fileName)
  {
   FILE *filePtr;
   struct token theToken;
   struct expr *testPtr;
   DATA_OBJECT rv;

   /*======================================================*/
   /* Open the file. Use either "fast save" or I/O Router. */
   /*======================================================*/

   if ((filePtr = GenOpen(theEnv,execStatus,fileName,"r")) == NULL)
     {
      OpenErrorMessage(theEnv,execStatus,"load-facts",fileName);
      return(FALSE);
     }

   SetFastLoad(theEnv,execStatus,filePtr);

   /*=================*/
   /* Load the facts. */
   /*=================*/

   theToken.type = LPAREN;
   while (theToken.type != STOP)
     {
      testPtr = StandardLoadFact(theEnv,execStatus,(char *) filePtr,&theToken);
      if (testPtr == NULL) theToken.type = STOP;
      else EvaluateExpression(theEnv,execStatus,testPtr,&rv);
      ReturnExpression(theEnv,execStatus,testPtr);
     }

   /*=================*/
   /* Close the file. */
   /*=================*/

   SetFastLoad(theEnv,execStatus,NULL);
   GenClose(theEnv,execStatus,filePtr);

   /*================================================*/
   /* Return TRUE if no error occurred while loading */
   /* the facts, otherwise return FALSE.             */
   /*================================================*/

   if (execStatus->EvaluationError) return(FALSE);
   return(TRUE);
  }

/*********************************************/
/* EnvLoadFactsFromString: C access routine. */
/*********************************************/
globle intBool EnvLoadFactsFromString(
  void *theEnv,EXEC_STATUS,
  char *theString,
  int theMax)
  {
   char * theStrRouter = "*** load-facts-from-string ***";
   struct token theToken;
   struct expr *testPtr;
   DATA_OBJECT rv;

   /*==========================*/
   /* Initialize string router */
   /*==========================*/

   if ((theMax == -1) ? (!OpenStringSource(theEnv,execStatus,theStrRouter,theString,0)) :
                        (!OpenTextSource(theEnv,execStatus,theStrRouter,theString,0,(unsigned) theMax)))
     return(FALSE);

   /*=================*/
   /* Load the facts. */
   /*=================*/

   theToken.type = LPAREN;
   while (theToken.type != STOP)
     {
      testPtr = StandardLoadFact(theEnv,execStatus,theStrRouter,&theToken);
      if (testPtr == NULL) theToken.type = STOP;
      else EvaluateExpression(theEnv,execStatus,testPtr,&rv);
      ReturnExpression(theEnv,execStatus,testPtr);
     }

   /*=================*/
   /* Close router.   */
   /*=================*/

   CloseStringSource(theEnv,execStatus,theStrRouter);

   /*================================================*/
   /* Return TRUE if no error occurred while loading */
   /* the facts, otherwise return FALSE.             */
   /*================================================*/

   if (execStatus->EvaluationError) return(FALSE);
   return(TRUE);
  }

/**************************************************************************/
/* StandardLoadFact: Loads a single fact from the specified logical name. */
/**************************************************************************/
static struct expr *StandardLoadFact(
  void *theEnv,EXEC_STATUS,
  char *logicalName,
  struct token *theToken)
  {
   int error = FALSE;
   struct expr *temp;

   GetToken(theEnv,execStatus,logicalName,theToken);
   if (theToken->type != LPAREN) return(NULL);

   temp = GenConstant(theEnv,execStatus,FCALL,FindFunction(theEnv,execStatus,"assert"));
   temp->argList = GetRHSPattern(theEnv,execStatus,logicalName,theToken,&error,
                                  TRUE,FALSE,TRUE,RPAREN);

   if (error == TRUE)
     {
      EnvPrintRouter(theEnv,execStatus,WERROR,"Function load-facts encountered an error\n");
      SetEvaluationError(theEnv,execStatus,TRUE);
      ReturnExpression(theEnv,execStatus,temp);
      return(NULL);
     }

   if (ExpressionContainsVariables(temp,TRUE))
     {
      ReturnExpression(theEnv,execStatus,temp);
      return(NULL);
     }

   return(temp);
  }

#if (! RUN_TIME)

/****************************************************************/
/* AssertParse: Driver routine for parsing the assert function. */
/****************************************************************/
static struct expr *AssertParse(
  void *theEnv,EXEC_STATUS,
  struct expr *top,
  char *logicalName)
  {
   int error;
   struct expr *rv;
   struct token theToken;

   ReturnExpression(theEnv,execStatus,top);
   SavePPBuffer(theEnv,execStatus," ");
   IncrementIndentDepth(theEnv,execStatus,8);
   rv = BuildRHSAssert(theEnv,execStatus,logicalName,&theToken,&error,TRUE,TRUE,"assert command","assert");
   DecrementIndentDepth(theEnv,execStatus,8);
   return(rv);
  }

#endif /* (! RUN_TIME) */

#endif /* DEFTEMPLATE_CONSTRUCT */


