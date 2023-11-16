   /*******************************************************/
   /*      "C" Language Integrated Production System      */
   /*                                                     */
   /*               CLIPS Version 6.24  06/05/06          */
   /*                                                     */
   /*                                                     */
   /*******************************************************/

/*************************************************************/
/* Purpose: Deffunction Parsing Routines                     */
/*                                                           */
/* Principal Programmer(s):                                  */
/*      Brian L. Dantes                                      */
/*                                                           */
/* Contributing Programmer(s):                               */
/*                                                           */
/* Revision History:                                         */
/*                                                           */
/*      6.24: Renamed BOOLEAN macro type to intBool.         */
/*                                                           */
/*            If the last construct in a loaded file is a    */
/*            deffunction or defmethod with no closing right */
/*            parenthesis, an error should be issued, but is */
/*            not. DR0872                                    */
/*                                                           */
/*            Added pragmas to prevent unused variable       */
/*            warnings.                                      */
/*                                                           */
/*************************************************************/

/* =========================================
   *****************************************
               EXTERNAL DEFINITIONS
   =========================================
   ***************************************** */
#include "setup.h"

#if DEFFUNCTION_CONSTRUCT && (! BLOAD_ONLY) && (! RUN_TIME)

#if BLOAD || BLOAD_AND_BSAVE
#include "bload.h"
#endif

#if DEFRULE_CONSTRUCT
#include "network.h"
#endif

#if DEFGENERIC_CONSTRUCT
#include "genrccom.h"
#endif

#include "constant.h"
#include "cstrcpsr.h"
#include "constrct.h"
#include "dffnxfun.h"
#include "envrnmnt.h"
#include "expressn.h"
#include "exprnpsr.h"
#include "extnfunc.h"
#include "memalloc.h"
#include "prccode.h"
#include "router.h"
#include "scanner.h"
#include "symbol.h"

#define _DFFNXPSR_SOURCE_
#include "dffnxpsr.h"

/* =========================================
   *****************************************
      INTERNALLY VISIBLE FUNCTION HEADERS
   =========================================
   ***************************************** */

static intBool ValidDeffunctionName(void *,EXEC_STATUS,char *);
static DEFFUNCTION *AddDeffunction(void *,EXEC_STATUS,SYMBOL_HN *,EXPRESSION *,int,int,int,int);

/* =========================================
   *****************************************
          EXTERNALLY VISIBLE FUNCTIONS
   =========================================
   ***************************************** */

/***************************************************************************
  NAME         : ParseDeffunction
  DESCRIPTION  : Parses the deffunction construct
  INPUTS       : The input logical name
  RETURNS      : FALSE if successful parse, TRUE otherwise
  SIDE EFFECTS : Creates valid deffunction definition
  NOTES        : H/L Syntax :
                 (deffunction <name> [<comment>]
                    (<single-field-varible>* [<multifield-variable>])
                    <action>*)
 ***************************************************************************/
globle intBool ParseDeffunction(
  void *theEnv,
  EXEC_STATUS,
  char *readSource)
  {
   SYMBOL_HN *deffunctionName;
   EXPRESSION *actions;
   EXPRESSION *parameterList;
   SYMBOL_HN *wildcard;
   int min,max,lvars,DeffunctionError = FALSE;
   short overwrite = FALSE, owMin = 0, owMax = 0;
   DEFFUNCTION *dptr;

   SetPPBufferStatus(theEnv,execStatus,ON);

   FlushPPBuffer(theEnv,execStatus);
   SetIndentDepth(theEnv,execStatus,3);
   SavePPBuffer(theEnv,execStatus,"(deffunction ");

#if BLOAD || BLOAD_AND_BSAVE
   if ((Bloaded(theEnv,execStatus) == TRUE) && (! ConstructData(theEnv,execStatus)->CheckSyntaxMode))
     {
      CannotLoadWithBloadMessage(theEnv,execStatus,"deffunctions");
      return(TRUE);
     }
#endif

   /* =====================================================
      Parse the name and comment fields of the deffunction.
      ===================================================== */
   deffunctionName = GetConstructNameAndComment(theEnv,execStatus,readSource,&DeffunctionData(theEnv,execStatus)->DFInputToken,"deffunction",
                                                EnvFindDeffunction,NULL,
                                                "!",TRUE,TRUE,TRUE);
   if (deffunctionName == NULL)
     return(TRUE);

   if (ValidDeffunctionName(theEnv,execStatus,ValueToString(deffunctionName)) == FALSE)
     return(TRUE);

   /*==========================*/
   /* Parse the argument list. */
   /*==========================*/
   parameterList = ParseProcParameters(theEnv,execStatus,readSource,&DeffunctionData(theEnv,execStatus)->DFInputToken,NULL,&wildcard,
                                       &min,&max,&DeffunctionError,NULL);
   if (DeffunctionError)
     return(TRUE);

   /*===================================================================*/
   /* Go ahead and add the deffunction so it can be recursively called. */
   /*===================================================================*/

   if (ConstructData(theEnv,execStatus)->CheckSyntaxMode)
     {
      dptr = (DEFFUNCTION *) EnvFindDeffunction(theEnv,execStatus,ValueToString(deffunctionName));
      if (dptr == NULL)
        { dptr = AddDeffunction(theEnv,execStatus,deffunctionName,NULL,min,max,0,TRUE); }
      else
        {
         overwrite = TRUE;
         owMin = (short) dptr->minNumberOfParameters;
         owMax = (short) dptr->maxNumberOfParameters;
         dptr->minNumberOfParameters = min;
         dptr->maxNumberOfParameters = max;
        }
     }
   else
     { dptr = AddDeffunction(theEnv,execStatus,deffunctionName,NULL,min,max,0,TRUE); }

   if (dptr == NULL)
     {
      ReturnExpression(theEnv,execStatus,parameterList);
      return(TRUE);
     }

   /*==================================================*/
   /* Parse the actions contained within the function. */
   /*==================================================*/

   PPCRAndIndent(theEnv,execStatus);

   ExpressionData(theEnv,execStatus)->ReturnContext = TRUE;
   actions = ParseProcActions(theEnv,execStatus,"deffunction",readSource,
                              &DeffunctionData(theEnv,execStatus)->DFInputToken,parameterList,wildcard,
                              NULL,NULL,&lvars,NULL);

   /*=============================================================*/
   /* Check for the closing right parenthesis of the deffunction. */
   /*=============================================================*/

   if ((DeffunctionData(theEnv,execStatus)->DFInputToken.type != RPAREN) && /* DR0872 */
       (actions != NULL))
     {
      SyntaxErrorMessage(theEnv,execStatus,"deffunction");
      
      ReturnExpression(theEnv,execStatus,parameterList);
      ReturnPackedExpression(theEnv,execStatus,actions);

      if (overwrite)
        {
         dptr->minNumberOfParameters = owMin;
         dptr->maxNumberOfParameters = owMax;
        }

      if ((dptr->busy == 0) && (! overwrite))
        {
         RemoveConstructFromModule(theEnv,execStatus,(struct constructHeader *) dptr);
         RemoveDeffunction(theEnv,execStatus,dptr);
        }

      return(TRUE);
     }

   if (actions == NULL)
     {
      ReturnExpression(theEnv,execStatus,parameterList);
      if (overwrite)
        {
         dptr->minNumberOfParameters = owMin;
         dptr->maxNumberOfParameters = owMax;
        }

      if ((dptr->busy == 0) && (! overwrite))
        {
         RemoveConstructFromModule(theEnv,execStatus,(struct constructHeader *) dptr);
         RemoveDeffunction(theEnv,execStatus,dptr);
        }
      return(TRUE);
     }

   /*==============================================*/
   /* If we're only checking syntax, don't add the */
   /* successfully parsed deffunction to the KB.   */
   /*==============================================*/

   if (ConstructData(theEnv,execStatus)->CheckSyntaxMode)
     {
      ReturnExpression(theEnv,execStatus,parameterList);
      ReturnPackedExpression(theEnv,execStatus,actions);
      if (overwrite)
        {
         dptr->minNumberOfParameters = owMin;
         dptr->maxNumberOfParameters = owMax;
        }
      else
        {
         RemoveConstructFromModule(theEnv,execStatus,(struct constructHeader *) dptr);
         RemoveDeffunction(theEnv,execStatus,dptr);
        }
      return(FALSE);
     }

   /*=============================*/
   /* Reformat the closing token. */
   /*=============================*/

   PPBackup(theEnv,execStatus);
   PPBackup(theEnv,execStatus);
   SavePPBuffer(theEnv,execStatus,DeffunctionData(theEnv,execStatus)->DFInputToken.printForm);
   SavePPBuffer(theEnv,execStatus,"\n");

   /*======================*/
   /* Add the deffunction. */
   /*======================*/

   AddDeffunction(theEnv,execStatus,deffunctionName,actions,min,max,lvars,FALSE);

   ReturnExpression(theEnv,execStatus,parameterList);

   return(DeffunctionError);
  }

/* =========================================
   *****************************************
          INTERNALLY VISIBLE FUNCTIONS
   =========================================
   ***************************************** */

/************************************************************
  NAME         : ValidDeffunctionName
  DESCRIPTION  : Determines if a new deffunction of the given
                 name can be defined in the current module
  INPUTS       : The new deffunction name
  RETURNS      : TRUE if OK, FALSE otherwise
  SIDE EFFECTS : Error message printed if not OK
  NOTES        : GetConstructNameAndComment() (called before
                 this function) ensures that the deffunction
                 name does not conflict with one from
                 another module
 ************************************************************/
static intBool ValidDeffunctionName(
  void *theEnv,
  EXEC_STATUS,
  char *theDeffunctionName)
  {
   struct constructHeader *theDeffunction;
#if DEFGENERIC_CONSTRUCT
   struct defmodule *theModule;
   struct constructHeader *theDefgeneric;
#endif

   /* ============================================
      A deffunction cannot be named the same as a
      construct type, e.g, defclass, defrule, etc.
      ============================================ */
   if (FindConstruct(theEnv,execStatus,theDeffunctionName) != NULL)
     {
      PrintErrorID(theEnv,execStatus,"DFFNXPSR",1,FALSE);
      EnvPrintRouter(theEnv,execStatus,WERROR,"Deffunctions are not allowed to replace constructs.\n");
      return(FALSE);
     }

   /* ============================================
      A deffunction cannot be named the same as a
      pre-defined system function, e.g, watch,
      list-defrules, etc.
      ============================================ */
   if (FindFunction(theEnv,execStatus,theDeffunctionName) != NULL)
     {
      PrintErrorID(theEnv,execStatus,"DFFNXPSR",2,FALSE);
      EnvPrintRouter(theEnv,execStatus,WERROR,"Deffunctions are not allowed to replace external functions.\n");
      return(FALSE);
     }

#if DEFGENERIC_CONSTRUCT
   /* ============================================
      A deffunction cannot be named the same as a
      generic function (either in this module or
      imported from another)
      ============================================ */
   theDefgeneric =
     (struct constructHeader *) LookupDefgenericInScope(theEnv,execStatus,theDeffunctionName);
   if (theDefgeneric != NULL)
     {
      theModule = GetConstructModuleItem(theDefgeneric)->theModule;
      if (theModule != ((struct defmodule *) EnvGetCurrentModule(theEnv,execStatus)))
        {
         PrintErrorID(theEnv,execStatus,"DFFNXPSR",5,FALSE);
         EnvPrintRouter(theEnv,execStatus,WERROR,"Defgeneric ");
         EnvPrintRouter(theEnv,execStatus,WERROR,EnvGetDefgenericName(theEnv,execStatus,(void *) theDefgeneric));
         EnvPrintRouter(theEnv,execStatus,WERROR," imported from module ");
         EnvPrintRouter(theEnv,execStatus,WERROR,EnvGetDefmoduleName(theEnv,execStatus,(void *) theModule));
         EnvPrintRouter(theEnv,execStatus,WERROR," conflicts with this deffunction.\n");
         return(FALSE);
        }
      else
        {
         PrintErrorID(theEnv,execStatus,"DFFNXPSR",3,FALSE);
         EnvPrintRouter(theEnv,execStatus,WERROR,"Deffunctions are not allowed to replace generic functions.\n");
        }
      return(FALSE);
     }
#endif

   theDeffunction = (struct constructHeader *) EnvFindDeffunction(theEnv,execStatus,theDeffunctionName);
   if (theDeffunction != NULL)
     {
      /* ===========================================
         And a deffunction in the current module can
         only be redefined if it is not executing.
         =========================================== */
      if (((DEFFUNCTION *) theDeffunction)->executing)
        {
         PrintErrorID(theEnv,execStatus,"DFNXPSR",4,FALSE);
         EnvPrintRouter(theEnv,execStatus,WERROR,"Deffunction ");
         EnvPrintRouter(theEnv,execStatus,WERROR,EnvGetDeffunctionName(theEnv,execStatus,(void *) theDeffunction));
         EnvPrintRouter(theEnv,execStatus,WERROR," may not be redefined while it is executing.\n");
         return(FALSE);
        }
     }
   return(TRUE);
  }


/****************************************************
  NAME         : AddDeffunction
  DESCRIPTION  : Adds a deffunction to the list of
                 deffunctions
  INPUTS       : 1) The symbolic name
                 2) The action expressions
                 3) The minimum number of arguments
                 4) The maximum number of arguments
                    (can be -1)
                 5) The number of local variables
                 6) A flag indicating if this is
                    a header call so that the
                    deffunction can be recursively
                    called
  RETURNS      : The new deffunction (NULL on errors)
  SIDE EFFECTS : Deffunction structures allocated
  NOTES        : Assumes deffunction is not executing
 ****************************************************/
#if WIN_BTC
#pragma argsused
#endif
static DEFFUNCTION *AddDeffunction(
  void *theEnv,
  EXEC_STATUS,
  SYMBOL_HN *name,
  EXPRESSION *actions,
  int min,
  int max,
  int lvars,
  int headerp)
  {
   DEFFUNCTION *dfuncPtr;
   unsigned oldbusy;
#if DEBUGGING_FUNCTIONS
   unsigned DFHadWatch = FALSE;
#else
#if MAC_MCW || WIN_MCW || MAC_XCD
#pragma unused(headerp)
#endif
#endif

   /*===============================================================*/
   /* If the deffunction doesn't exist, create a new structure to   */
   /* contain it and add it to the List of deffunctions. Otherwise, */
   /* use the existing structure and remove the pretty print form   */
   /* and interpretive code.                                        */
   /*===============================================================*/
   dfuncPtr = (DEFFUNCTION *) EnvFindDeffunction(theEnv,execStatus,ValueToString(name));
   if (dfuncPtr == NULL)
     {
      dfuncPtr = get_struct(theEnv,execStatus,deffunctionStruct);
      InitializeConstructHeader(theEnv,execStatus,"deffunction",(struct constructHeader *) dfuncPtr,name);
      IncrementSymbolCount(name);
      dfuncPtr->code = NULL;
      dfuncPtr->minNumberOfParameters = min;
      dfuncPtr->maxNumberOfParameters = max;
      dfuncPtr->numberOfLocalVars = lvars;
      dfuncPtr->busy = 0;
      dfuncPtr->executing = 0;
     }
   else
     {
#if DEBUGGING_FUNCTIONS
      DFHadWatch = EnvGetDeffunctionWatch(theEnv,execStatus,(void *) dfuncPtr);
#endif
      dfuncPtr->minNumberOfParameters = min;
      dfuncPtr->maxNumberOfParameters = max;
      dfuncPtr->numberOfLocalVars = lvars;
      oldbusy = dfuncPtr->busy;
      ExpressionDeinstall(theEnv,execStatus,dfuncPtr->code);
      dfuncPtr->busy = oldbusy;
      ReturnPackedExpression(theEnv,execStatus,dfuncPtr->code);
      dfuncPtr->code = NULL;
      SetDeffunctionPPForm((void *) dfuncPtr,NULL);

      /* =======================================
         Remove the deffunction from the list so
         that it can be added at the end
         ======================================= */
      RemoveConstructFromModule(theEnv,execStatus,(struct constructHeader *) dfuncPtr);
     }

   AddConstructToModule((struct constructHeader *) dfuncPtr);

   /* ==================================
      Install the new interpretive code.
      ================================== */

   if (actions != NULL)
     {
      /* ===============================
         If a deffunction is recursive,
         do not increment its busy count
         based on self-references
         =============================== */
      oldbusy = dfuncPtr->busy;
      ExpressionInstall(theEnv,execStatus,actions);
      dfuncPtr->busy = oldbusy;
      dfuncPtr->code = actions;
     }

   /* ===============================================================
      Install the pretty print form if memory is not being conserved.
      =============================================================== */

#if DEBUGGING_FUNCTIONS
   EnvSetDeffunctionWatch(theEnv,execStatus,DFHadWatch ? TRUE : DeffunctionData(theEnv,execStatus)->WatchDeffunctions,(void *) dfuncPtr);
   if ((EnvGetConserveMemory(theEnv,execStatus) == FALSE) && (headerp == FALSE))
     SetDeffunctionPPForm((void *) dfuncPtr,CopyPPBuffer(theEnv,execStatus));
#endif
   return(dfuncPtr);
  }

#endif /* DEFFUNCTION_CONSTRUCT && (! BLOAD_ONLY) && (! RUN_TIME) */

