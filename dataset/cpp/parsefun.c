   /*******************************************************/
   /*      "C" Language Integrated Production System      */
   /*                                                     */
   /*             CLIPS Version 6.24  06/02/06            */
   /*                                                     */
   /*               PARSING FUNCTIONS MODULE              */
   /*******************************************************/

/*************************************************************/
/* Purpose: Contains the code for several parsing related    */
/*   functions including...                                  */
/*                                                           */
/* Principal Programmer(s):                                  */
/*      Gary D. Riley                                        */
/*                                                           */
/* Contributing Programmer(s):                               */
/*                                                           */
/* Revision History:                                         */
/*                                                           */
/*      6.23: Correction for FalseSymbol/TrueSymbol. DR0859  */
/*                                                           */
/*      6.24: Corrected code to remove run-time program      */
/*            compiler warnings.                             */
/*                                                           */
/*************************************************************/

#define _PARSEFUN_SOURCE_

#include "setup.h"

#include <string.h>

#include "argacces.h"
#include "cstrcpsr.h"
#include "envrnmnt.h"
#include "exprnpsr.h"
#include "extnfunc.h"
#include "memalloc.h"
#include "multifld.h"
#include "prcdrpsr.h"
#include "router.h"
#include "strngrtr.h"
#include "utility.h"

#include "parsefun.h"

#define PARSEFUN_DATA 11

struct parseFunctionData
  { 
   char *ErrorString;
   size_t ErrorCurrentPosition;
   size_t ErrorMaximumPosition;
   char *WarningString;
   size_t WarningCurrentPosition;
   size_t WarningMaximumPosition;
  };

#define ParseFunctionData(theEnv,execStatus) ((struct parseFunctionData *) GetEnvironmentData(theEnv,execStatus,PARSEFUN_DATA))

/***************************************/
/* LOCAL INTERNAL FUNCTION DEFINITIONS */
/***************************************/

#if (! RUN_TIME) && (! BLOAD_ONLY)
   static int                     FindErrorCapture(void *,EXEC_STATUS,char *);
   static int                     PrintErrorCapture(void *,EXEC_STATUS,char *,char *);
   static void                    DeactivateErrorCapture(void *,EXEC_STATUS);
   static void                    SetErrorCaptureValues(void *,EXEC_STATUS,DATA_OBJECT_PTR);
#endif

/*****************************************/
/* ParseFunctionDefinitions: Initializes */
/*   the parsing related functions.      */
/*****************************************/
globle void ParseFunctionDefinitions(
  void *theEnv,
  EXEC_STATUS)
  {
   AllocateEnvironmentData(theEnv,execStatus,PARSEFUN_DATA,sizeof(struct parseFunctionData),NULL);

#if ! RUN_TIME
   EnvDefineFunction2(theEnv,execStatus,"check-syntax",'u',PTIEF CheckSyntaxFunction,"CheckSyntaxFunction","11s");
#endif
  }

#if (! RUN_TIME) && (! BLOAD_ONLY)
/*******************************************/
/* CheckSyntaxFunction: H/L access routine */
/*   for the check-syntax function.        */
/*******************************************/
globle void CheckSyntaxFunction(
  void *theEnv,
  EXEC_STATUS,
  DATA_OBJECT *returnValue)
  {
   DATA_OBJECT theArg;

   /*===============================*/
   /* Set up a default return value */
   /* (TRUE for problems found).    */
   /*===============================*/

   SetpType(returnValue,SYMBOL);
   SetpValue(returnValue,EnvTrueSymbol(theEnv,execStatus));

   /*=====================================================*/
   /* Function check-syntax expects exactly one argument. */
   /*=====================================================*/

   if (EnvArgCountCheck(theEnv,execStatus,"check-syntax",EXACTLY,1) == -1) return;

   /*========================================*/
   /* The argument should be of type STRING. */
   /*========================================*/

   if (EnvArgTypeCheck(theEnv,execStatus,"check-syntax",1,STRING,&theArg) == FALSE)
     { return; }

   /*===================*/
   /* Check the syntax. */
   /*===================*/

   CheckSyntax(theEnv,execStatus,DOToString(theArg),returnValue);
  }

/*********************************/
/* CheckSyntax: C access routine */
/*   for the build function.     */
/*********************************/
globle int CheckSyntax(
  void *theEnv,
  EXEC_STATUS,
  char *theString,
  DATA_OBJECT_PTR returnValue)
  {
   char *name;
   struct token theToken;
   struct expr *top;
   short rv;

   /*==============================*/
   /* Set the default return value */
   /* (TRUE for problems found).   */
   /*==============================*/

   SetpType(returnValue,SYMBOL);
   SetpValue(returnValue,EnvTrueSymbol(theEnv,execStatus));

   /*===========================================*/
   /* Create a string source router so that the */
   /* string can be used as an input source.    */
   /*===========================================*/

   if (OpenStringSource(theEnv,execStatus,"check-syntax",theString,0) == 0)
     { return(TRUE); }

   /*=================================*/
   /* Only expressions and constructs */
   /* can have their syntax checked.  */
   /*=================================*/

   GetToken(theEnv,execStatus,"check-syntax",&theToken);

   if (theToken.type != LPAREN)
     {
      CloseStringSource(theEnv,execStatus,"check-syntax");
      SetpValue(returnValue,EnvAddSymbol(theEnv,execStatus,"MISSING-LEFT-PARENTHESIS"));
      return(TRUE);
     }

   /*========================================*/
   /* The next token should be the construct */
   /* type or function name.                 */
   /*========================================*/

   GetToken(theEnv,execStatus,"check-syntax",&theToken);
   if (theToken.type != SYMBOL)
     {
      CloseStringSource(theEnv,execStatus,"check-syntax");
      SetpValue(returnValue,EnvAddSymbol(theEnv,execStatus,"EXPECTED-SYMBOL-AFTER-LEFT-PARENTHESIS"));
      return(TRUE);
     }

   name = ValueToString(theToken.value);

   /*==============================================*/
   /* Set up a router to capture the error output. */
   /*==============================================*/

   EnvAddRouter(theEnv,execStatus,"error-capture",40,
              FindErrorCapture, PrintErrorCapture,
              NULL, NULL, NULL);

   /*================================*/
   /* Determine if it's a construct. */
   /*================================*/

   if (FindConstruct(theEnv,execStatus,name))
     {
      ConstructData(theEnv,execStatus)->CheckSyntaxMode = TRUE;
      rv = (short) ParseConstruct(theEnv,execStatus,name,"check-syntax");
      GetToken(theEnv,execStatus,"check-syntax",&theToken);
      ConstructData(theEnv,execStatus)->CheckSyntaxMode = FALSE;

      if (rv)
        {
         EnvPrintRouter(theEnv,execStatus,WERROR,"\nERROR:\n");
         PrintInChunks(theEnv,execStatus,WERROR,GetPPBuffer(theEnv,execStatus));
         EnvPrintRouter(theEnv,execStatus,WERROR,"\n");
        }

      DestroyPPBuffer(theEnv,execStatus);

      CloseStringSource(theEnv,execStatus,"check-syntax");

      if ((rv != FALSE) || (ParseFunctionData(theEnv,execStatus)->WarningString != NULL))
        {
         SetErrorCaptureValues(theEnv,execStatus,returnValue);
         DeactivateErrorCapture(theEnv,execStatus);
         return(TRUE);
        }

      if (theToken.type != STOP)
        {
         SetpValue(returnValue,EnvAddSymbol(theEnv,execStatus,"EXTRANEOUS-INPUT-AFTER-LAST-PARENTHESIS"));
         DeactivateErrorCapture(theEnv,execStatus);
         return(TRUE);
        }

      SetpType(returnValue,SYMBOL);
      SetpValue(returnValue,EnvFalseSymbol(theEnv,execStatus));
      DeactivateErrorCapture(theEnv,execStatus);
      return(FALSE);
     }

   /*=======================*/
   /* Parse the expression. */
   /*=======================*/

   top = Function2Parse(theEnv,execStatus,"check-syntax",name);
   GetToken(theEnv,execStatus,"check-syntax",&theToken);
   ClearParsedBindNames(theEnv,execStatus);
   CloseStringSource(theEnv,execStatus,"check-syntax");

   if (top == NULL)
     {
      SetErrorCaptureValues(theEnv,execStatus,returnValue);
      DeactivateErrorCapture(theEnv,execStatus);
      return(TRUE);
     }

   if (theToken.type != STOP)
     {
      SetpValue(returnValue,EnvAddSymbol(theEnv,execStatus,"EXTRANEOUS-INPUT-AFTER-LAST-PARENTHESIS"));
      DeactivateErrorCapture(theEnv,execStatus);
      ReturnExpression(theEnv,execStatus,top);
      return(TRUE);
     }

   DeactivateErrorCapture(theEnv,execStatus);

   ReturnExpression(theEnv,execStatus,top);
   SetpType(returnValue,SYMBOL);
   SetpValue(returnValue,EnvFalseSymbol(theEnv,execStatus));
   return(FALSE);
  }

/**************************************************/
/* DeactivateErrorCapture: Deactivates the error  */
/*   capture router and the strings used to store */
/*   the captured information.                    */
/**************************************************/
static void DeactivateErrorCapture(
  void *theEnv,
  EXEC_STATUS)
  {   
   if (ParseFunctionData(theEnv,execStatus)->ErrorString != NULL)
     {
      rm(theEnv,execStatus,ParseFunctionData(theEnv,execStatus)->ErrorString,ParseFunctionData(theEnv,execStatus)->ErrorMaximumPosition);
      ParseFunctionData(theEnv,execStatus)->ErrorString = NULL;
     }

   if (ParseFunctionData(theEnv,execStatus)->WarningString != NULL)
     {
      rm(theEnv,execStatus,ParseFunctionData(theEnv,execStatus)->WarningString,ParseFunctionData(theEnv,execStatus)->WarningMaximumPosition);
      ParseFunctionData(theEnv,execStatus)->WarningString = NULL;
     }

   ParseFunctionData(theEnv,execStatus)->ErrorCurrentPosition = 0;
   ParseFunctionData(theEnv,execStatus)->ErrorMaximumPosition = 0;
   ParseFunctionData(theEnv,execStatus)->WarningCurrentPosition = 0;
   ParseFunctionData(theEnv,execStatus)->WarningMaximumPosition = 0;

   EnvDeleteRouter(theEnv,execStatus,"error-capture");
  }

/******************************************************************/
/* SetErrorCaptureValues: Stores the error/warnings captured when */
/*   parsing an expression or construct into a multifield value.  */
/*   The first field contains the output sent to the WERROR       */
/*   logical name and the second field contains the output sent   */
/*   to the WWARNING logical name. FALSE is stored in either      */
/*   position if no output was sent to those logical names.       */
/******************************************************************/
static void SetErrorCaptureValues(
  void *theEnv,
  EXEC_STATUS,
  DATA_OBJECT_PTR returnValue)
  {
   struct multifield *theMultifield;

   theMultifield = (struct multifield *) EnvCreateMultifield(theEnv,execStatus,2L);

   if (ParseFunctionData(theEnv,execStatus)->ErrorString != NULL)
     {
      SetMFType(theMultifield,1,STRING);
      SetMFValue(theMultifield,1,EnvAddSymbol(theEnv,execStatus,ParseFunctionData(theEnv,execStatus)->ErrorString));
     }
   else
     {
      SetMFType(theMultifield,1,SYMBOL);
      SetMFValue(theMultifield,1,EnvFalseSymbol(theEnv,execStatus));
     }

   if (ParseFunctionData(theEnv,execStatus)->WarningString != NULL)
     {
      SetMFType(theMultifield,2,STRING);
      SetMFValue(theMultifield,2,EnvAddSymbol(theEnv,execStatus,ParseFunctionData(theEnv,execStatus)->WarningString));
     }
   else
     {
      SetMFType(theMultifield,2,SYMBOL);
      SetMFValue(theMultifield,2,EnvFalseSymbol(theEnv,execStatus));
     }

   SetpType(returnValue,MULTIFIELD);
   SetpDOBegin(returnValue,1);
   SetpDOEnd(returnValue,2);
   SetpValue(returnValue,(void *) theMultifield);
  }

/**********************************/
/* FindErrorCapture: Find routine */
/*   for the check-syntax router. */
/**********************************/
#if WIN_BTC
#pragma argsused
#endif
static int FindErrorCapture(
  void *theEnv,
  EXEC_STATUS,
  char *logicalName)
  {
#if MAC_MCW || WIN_MCW || MAC_XCD
#pragma unused(theEnv,execStatus)
#endif

   if ((strcmp(logicalName,WERROR) == 0) ||
       (strcmp(logicalName,WWARNING) == 0))
     { return(TRUE); }

   return(FALSE);
  }

/************************************/
/* PrintErrorCapture: Print routine */
/*   for the check-syntax router.   */
/************************************/
static int PrintErrorCapture(
  void *theEnv,
  EXEC_STATUS,
  char *logicalName,
  char *str)
  {
   if (strcmp(logicalName,WERROR) == 0)
     {
      ParseFunctionData(theEnv,execStatus)->ErrorString = AppendToString(theEnv,execStatus,str,ParseFunctionData(theEnv,execStatus)->ErrorString,
                                   &ParseFunctionData(theEnv,execStatus)->ErrorCurrentPosition,
                                   &ParseFunctionData(theEnv,execStatus)->ErrorMaximumPosition);
     }
   else if (strcmp(logicalName,WWARNING) == 0)
     {
      ParseFunctionData(theEnv,execStatus)->WarningString = AppendToString(theEnv,execStatus,str,ParseFunctionData(theEnv,execStatus)->WarningString,
                                     &ParseFunctionData(theEnv,execStatus)->WarningCurrentPosition,
                                     &ParseFunctionData(theEnv,execStatus)->WarningMaximumPosition);
     }

   return(1);
  }

#else
/****************************************************/
/* CheckSyntaxFunction: This is the non-functional  */
/*   stub provided for use with a run-time version. */
/****************************************************/
globle void CheckSyntaxFunction(
  void *theEnv,
  EXEC_STATUS,
  DATA_OBJECT *returnValue)
  {
   PrintErrorID(theEnv,execStatus,"PARSEFUN",1,FALSE);
   EnvPrintRouter(theEnv,execStatus,WERROR,"Function check-syntax does not work in run time modules.\n");
   SetpType(returnValue,SYMBOL);
   SetpValue(returnValue,EnvTrueSymbol(theEnv,execStatus));
  }

/************************************************/
/* CheckSyntax: This is the non-functional stub */
/*   provided for use with a run-time version.  */
/************************************************/
globle int CheckSyntax(
  void *theEnv,
  EXEC_STATUS,
  char *theString,
  DATA_OBJECT_PTR returnValue)
  {
#if (MAC_MCW || WIN_MCW) && (RUN_TIME || BLOAD_ONLY)
#pragma unused(theString)
#pragma unused(returnValue)
#endif

   PrintErrorID(theEnv,execStatus,"PARSEFUN",1,FALSE);
   EnvPrintRouter(theEnv,execStatus,WERROR,"Function check-syntax does not work in run time modules.\n");
   SetpType(returnValue,SYMBOL);
   SetpValue(returnValue,EnvTrueSymbol(theEnv,execStatus));
   return(TRUE);
  }

#endif /* (! RUN_TIME) && (! BLOAD_ONLY) */


