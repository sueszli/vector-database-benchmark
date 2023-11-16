   /*******************************************************/
   /*      "C" Language Integrated Production System      */
   /*                                                     */
   /*               CLIPS Version 6.24  06/05/06          */
   /*                                                     */
   /*          INSTANCE-SET QUERIES PARSER MODULE         */
   /*******************************************************/

/*************************************************************/
/* Purpose: Instance_set Queries Parsing Routines            */
/*                                                           */
/* Principal Programmer(s):                                  */
/*      Brian L. Dantes                                      */
/*                                                           */
/* Contributing Programmer(s):                               */
/*                                                           */
/*                                                           */
/* Revision History:                                         */
/*      6.23: Changed name of variable exp to theExp         */
/*            because of Unix compiler warnings of shadowed  */
/*            definitions.                                   */
/*                                                           */
/*      6.24: Renamed BOOLEAN macro type to intBool.         */
/*                                                           */
/*************************************************************/

/* =========================================
   *****************************************
               EXTERNAL DEFINITIONS
   =========================================
   ***************************************** */
#include "setup.h"

#if INSTANCE_SET_QUERIES && (! RUN_TIME)

#include <string.h>

#include "classcom.h"
#include "envrnmnt.h"
#include "exprnpsr.h"
#include "extnfunc.h"
#include "insquery.h"
#include "prcdrpsr.h"
#include "prntutil.h"
#include "router.h"
#include "scanner.h"
#include "strngrtr.h"

#define _INSQYPSR_SOURCE_
#include "insqypsr.h"

/* =========================================
   *****************************************
                   CONSTANTS
   =========================================
   ***************************************** */
#define INSTANCE_SLOT_REF ':'

/* =========================================
   *****************************************
      INTERNALLY VISIBLE FUNCTION HEADERS
   =========================================
   ***************************************** */

static EXPRESSION *ParseQueryRestrictions(void *,EXEC_STATUS,EXPRESSION *,char *,struct token *);
static intBool ReplaceClassNameWithReference(void *,EXEC_STATUS,EXPRESSION *);
static int ParseQueryTestExpression(void *,EXEC_STATUS,EXPRESSION *,char *);
static int ParseQueryActionExpression(void *,EXEC_STATUS,EXPRESSION *,char *,EXPRESSION *,struct token *);
static void ReplaceInstanceVariables(void *,EXEC_STATUS,EXPRESSION *,EXPRESSION *,int,int);
static void ReplaceSlotReference(void *,EXEC_STATUS,EXPRESSION *,EXPRESSION *,
                                 struct FunctionDefinition *,int);
static int IsQueryFunction(EXPRESSION *);

/* =========================================
   *****************************************
          EXTERNALLY VISIBLE FUNCTIONS
   =========================================
   ***************************************** */

/***********************************************************************
  NAME         : ParseQueryNoAction
  DESCRIPTION  : Parses the following functions :
                   (any-instancep)
                   (find-first-instance)
                   (find-all-instances)
  INPUTS       : 1) The address of the top node of the query function
                 2) The logical name of the input
  RETURNS      : The completed expression chain, or NULL on errors
  SIDE EFFECTS : The expression chain is extended, or the "top" node
                 is deleted on errors
  NOTES        : H/L Syntax :

                 (<function> <query-block>)

                 <query-block>  :== (<instance-var>+) <query-expression>
                 <instance-var> :== (<var-name> <class-name>+)

                 Parses into following form :

                 <query-function>
                      |
                      V
                 <query-expression>  ->

                 <class-1a> -> <class-1b> -> (QDS) ->

                 <class-2a> -> <class-2b> -> (QDS) -> ...
 ***********************************************************************/
globle EXPRESSION *ParseQueryNoAction(
  void *theEnv,
  EXEC_STATUS,
  EXPRESSION *top,
  char *readSource)
  {
   EXPRESSION *insQuerySetVars;
   struct token queryInputToken;

   insQuerySetVars = ParseQueryRestrictions(theEnv,execStatus,top,readSource,&queryInputToken);
   if (insQuerySetVars == NULL)
     return(NULL);
   IncrementIndentDepth(theEnv,execStatus,3);
   PPCRAndIndent(theEnv,execStatus);
   if (ParseQueryTestExpression(theEnv,execStatus,top,readSource) == FALSE)
     {
      DecrementIndentDepth(theEnv,execStatus,3);
      ReturnExpression(theEnv,execStatus,insQuerySetVars);
      return(NULL);
     }
   DecrementIndentDepth(theEnv,execStatus,3);
   GetToken(theEnv,execStatus,readSource,&queryInputToken);
   if (GetType(queryInputToken) != RPAREN)
     {
      SyntaxErrorMessage(theEnv,execStatus,"instance-set query function");
      ReturnExpression(theEnv,execStatus,top);
      ReturnExpression(theEnv,execStatus,insQuerySetVars);
      return(NULL);
     }
   ReplaceInstanceVariables(theEnv,execStatus,insQuerySetVars,top->argList,TRUE,0);
   ReturnExpression(theEnv,execStatus,insQuerySetVars);
   return(top);
  }

/***********************************************************************
  NAME         : ParseQueryAction
  DESCRIPTION  : Parses the following functions :
                   (do-for-instance)
                   (do-for-all-instances)
                   (delayed-do-for-all-instances)
  INPUTS       : 1) The address of the top node of the query function
                 2) The logical name of the input
  RETURNS      : The completed expression chain, or NULL on errors
  SIDE EFFECTS : The expression chain is extended, or the "top" node
                 is deleted on errors
  NOTES        : H/L Syntax :

                 (<function> <query-block> <query-action>)

                 <query-block>  :== (<instance-var>+) <query-expression>
                 <instance-var> :== (<var-name> <class-name>+)

                 Parses into following form :

                 <query-function>
                      |
                      V
                 <query-expression> -> <query-action>  ->

                 <class-1a> -> <class-1b> -> (QDS) ->

                 <class-2a> -> <class-2b> -> (QDS) -> ...
 ***********************************************************************/
globle EXPRESSION *ParseQueryAction(
  void *theEnv,
  EXEC_STATUS,
  EXPRESSION *top,
  char *readSource)
  {
   EXPRESSION *insQuerySetVars;
   struct token queryInputToken;

   insQuerySetVars = ParseQueryRestrictions(theEnv,execStatus,top,readSource,&queryInputToken);
   if (insQuerySetVars == NULL)
     return(NULL);
   IncrementIndentDepth(theEnv,execStatus,3);
   PPCRAndIndent(theEnv,execStatus);
   if (ParseQueryTestExpression(theEnv,execStatus,top,readSource) == FALSE)
     {
      DecrementIndentDepth(theEnv,execStatus,3);
      ReturnExpression(theEnv,execStatus,insQuerySetVars);
      return(NULL);
     }
   PPCRAndIndent(theEnv,execStatus);
   if (ParseQueryActionExpression(theEnv,execStatus,top,readSource,insQuerySetVars,&queryInputToken) == FALSE)
     {
      DecrementIndentDepth(theEnv,execStatus,3);
      ReturnExpression(theEnv,execStatus,insQuerySetVars);
      return(NULL);
     }
   DecrementIndentDepth(theEnv,execStatus,3);
   
   if (GetType(queryInputToken) != RPAREN)
     {
      SyntaxErrorMessage(theEnv,execStatus,"instance-set query function");
      ReturnExpression(theEnv,execStatus,top);
      ReturnExpression(theEnv,execStatus,insQuerySetVars);
      return(NULL);
     }
   ReplaceInstanceVariables(theEnv,execStatus,insQuerySetVars,top->argList,TRUE,0);
   ReplaceInstanceVariables(theEnv,execStatus,insQuerySetVars,top->argList->nextArg,FALSE,0);
   ReturnExpression(theEnv,execStatus,insQuerySetVars);
   return(top);
  }

/* =========================================
   *****************************************
          INTERNALLY VISIBLE FUNCTIONS
   =========================================
   ***************************************** */

/***************************************************************
  NAME         : ParseQueryRestrictions
  DESCRIPTION  : Parses the class restrictions for a query
  INPUTS       : 1) The top node of the query expression
                 2) The logical name of the input
                 3) Caller's token buffer
  RETURNS      : The instance-variable expressions
  SIDE EFFECTS : Entire query expression deleted on errors
                 Nodes allocated for restrictions and instance
                   variable expressions
                 Class restrictions attached to query-expression
                   as arguments
  NOTES        : Expects top != NULL
 ***************************************************************/
static EXPRESSION *ParseQueryRestrictions(
  void *theEnv,
  EXEC_STATUS,
  EXPRESSION *top,
  char *readSource,
  struct token *queryInputToken)
  {
   EXPRESSION *insQuerySetVars = NULL,*lastInsQuerySetVars = NULL,
              *classExp = NULL,*lastClassExp,
              *tmp,*lastOne = NULL;
   int error = FALSE;

   SavePPBuffer(theEnv,execStatus," ");
   GetToken(theEnv,execStatus,readSource,queryInputToken);
   if (queryInputToken->type != LPAREN)
     goto ParseQueryRestrictionsError1;
   GetToken(theEnv,execStatus,readSource,queryInputToken);
   if (queryInputToken->type != LPAREN)
     goto ParseQueryRestrictionsError1;
   while (queryInputToken->type == LPAREN)
     {
      GetToken(theEnv,execStatus,readSource,queryInputToken);
      if (queryInputToken->type != SF_VARIABLE)
        goto ParseQueryRestrictionsError1;
      tmp = insQuerySetVars;
      while (tmp != NULL)
        {
         if (tmp->value == queryInputToken->value)
           {
            PrintErrorID(theEnv,execStatus,"INSQYPSR",1,FALSE);
            EnvPrintRouter(theEnv,execStatus,WERROR,"Duplicate instance member variable name in function ");
            EnvPrintRouter(theEnv,execStatus,WERROR,ValueToString(ExpressionFunctionCallName(top)));
            EnvPrintRouter(theEnv,execStatus,WERROR,".\n");
            goto ParseQueryRestrictionsError2;
           }
         tmp = tmp->nextArg;
        }
      tmp = GenConstant(theEnv,execStatus,SF_VARIABLE,queryInputToken->value);
      if (insQuerySetVars == NULL)
        insQuerySetVars = tmp;
      else
        lastInsQuerySetVars->nextArg = tmp;
      lastInsQuerySetVars = tmp;
      SavePPBuffer(theEnv,execStatus," ");
      classExp = ArgumentParse(theEnv,execStatus,readSource,&error);
      if (error)
        goto ParseQueryRestrictionsError2;
      if (classExp == NULL)
        goto ParseQueryRestrictionsError1;
      if (ReplaceClassNameWithReference(theEnv,execStatus,classExp) == FALSE)
        goto ParseQueryRestrictionsError2;
      lastClassExp = classExp;
      SavePPBuffer(theEnv,execStatus," ");
      while ((tmp = ArgumentParse(theEnv,execStatus,readSource,&error)) != NULL)
        {
         if (ReplaceClassNameWithReference(theEnv,execStatus,tmp) == FALSE)
           goto ParseQueryRestrictionsError2;
         lastClassExp->nextArg = tmp;
         lastClassExp = tmp;
         SavePPBuffer(theEnv,execStatus," ");
        }
      if (error)
        goto ParseQueryRestrictionsError2;
      PPBackup(theEnv,execStatus);
      PPBackup(theEnv,execStatus);
      SavePPBuffer(theEnv,execStatus,")");
      tmp = GenConstant(theEnv,execStatus,SYMBOL,(void *) InstanceQueryData(theEnv,execStatus)->QUERY_DELIMETER_SYMBOL);
      lastClassExp->nextArg = tmp;
      lastClassExp = tmp;
      if (top->argList == NULL)
        top->argList = classExp;
      else
        lastOne->nextArg = classExp;
      lastOne = lastClassExp;
      classExp = NULL;
      SavePPBuffer(theEnv,execStatus," ");
      GetToken(theEnv,execStatus,readSource,queryInputToken);
     }
   if (queryInputToken->type != RPAREN)
     goto ParseQueryRestrictionsError1;
   PPBackup(theEnv,execStatus);
   PPBackup(theEnv,execStatus);
   SavePPBuffer(theEnv,execStatus,")");
   return(insQuerySetVars);

ParseQueryRestrictionsError1:
   SyntaxErrorMessage(theEnv,execStatus,"instance-set query function");

ParseQueryRestrictionsError2:
   ReturnExpression(theEnv,execStatus,classExp);
   ReturnExpression(theEnv,execStatus,top);
   ReturnExpression(theEnv,execStatus,insQuerySetVars);
   return(NULL);
  }

/***************************************************
  NAME         : ReplaceClassNameWithReference
  DESCRIPTION  : In parsing an instance-set query,
                 this function replaces a constant
                 class name with an actual pointer
                 to the class
  INPUTS       : The expression
  RETURNS      : TRUE if all OK, FALSE
                 if class cannot be found
  SIDE EFFECTS : The expression type and value are
                 modified if class is found
  NOTES        : Searches current and imported
                 modules for reference
 ***************************************************/
static intBool ReplaceClassNameWithReference(
  void *theEnv,
  EXEC_STATUS,
  EXPRESSION *theExp)
  {
   char *theClassName;
   void *theDefclass;

   if (theExp->type == SYMBOL)
     {
      theClassName = ValueToString(theExp->value);
      theDefclass = (void *) LookupDefclassByMdlOrScope(theEnv,execStatus,theClassName);
      if (theDefclass == NULL)
        {
         CantFindItemErrorMessage(theEnv,execStatus,"class",theClassName);
         return(FALSE);
        }
      theExp->type = DEFCLASS_PTR;
      theExp->value = theDefclass;
     }
   return(TRUE);
  }

/*************************************************************
  NAME         : ParseQueryTestExpression
  DESCRIPTION  : Parses the test-expression for a query
  INPUTS       : 1) The top node of the query expression
                 2) The logical name of the input
  RETURNS      : TRUE if all OK, FALSE otherwise
  SIDE EFFECTS : Entire query-expression deleted on errors
                 Nodes allocated for new expression
                 Test shoved in front of class-restrictions on
                    query argument list
  NOTES        : Expects top != NULL
 *************************************************************/
static int ParseQueryTestExpression(
  void *theEnv,
  EXEC_STATUS,
  EXPRESSION *top,
  char *readSource)
  {
   EXPRESSION *qtest;
   int error;
   struct BindInfo *oldBindList;

   error = FALSE;
   oldBindList = GetParsedBindNames(theEnv,execStatus);
   SetParsedBindNames(theEnv,execStatus,NULL);
   qtest = ArgumentParse(theEnv,execStatus,readSource,&error);
   if (error == TRUE)
     {
      SetParsedBindNames(theEnv,execStatus,oldBindList);
      ReturnExpression(theEnv,execStatus,top);
      return(FALSE);
     }
   if (qtest == NULL)
     {
      SetParsedBindNames(theEnv,execStatus,oldBindList);
      SyntaxErrorMessage(theEnv,execStatus,"instance-set query function");
      ReturnExpression(theEnv,execStatus,top);
      return(FALSE);
     }
   qtest->nextArg = top->argList;
   top->argList = qtest;
   if (ParsedBindNamesEmpty(theEnv,execStatus) == FALSE)
     {
      ClearParsedBindNames(theEnv,execStatus);
      SetParsedBindNames(theEnv,execStatus,oldBindList);
      PrintErrorID(theEnv,execStatus,"INSQYPSR",2,FALSE);
      EnvPrintRouter(theEnv,execStatus,WERROR,"Binds are not allowed in instance-set query in function ");
      EnvPrintRouter(theEnv,execStatus,WERROR,ValueToString(ExpressionFunctionCallName(top)));
      EnvPrintRouter(theEnv,execStatus,WERROR,".\n");
      ReturnExpression(theEnv,execStatus,top);
      return(FALSE);
     }
   SetParsedBindNames(theEnv,execStatus,oldBindList);
   return(TRUE);
  }

/*************************************************************
  NAME         : ParseQueryActionExpression
  DESCRIPTION  : Parses the action-expression for a query
  INPUTS       : 1) The top node of the query expression
                 2) The logical name of the input
                 3) List of query parameters
  RETURNS      : TRUE if all OK, FALSE otherwise
  SIDE EFFECTS : Entire query-expression deleted on errors
                 Nodes allocated for new expression
                 Action shoved in front of class-restrictions
                    and in back of test-expression on query
                    argument list
  NOTES        : Expects top != NULL && top->argList != NULL
 *************************************************************/
static int ParseQueryActionExpression(
  void *theEnv,
  EXEC_STATUS,
  EXPRESSION *top,
  char *readSource,
  EXPRESSION *insQuerySetVars,
  struct token *queryInputToken)
  {
   EXPRESSION *qaction,*tmpInsSetVars;
   int error;
   struct BindInfo *oldBindList,*newBindList,*prev;

   error = FALSE;
   oldBindList = GetParsedBindNames(theEnv,execStatus);
   SetParsedBindNames(theEnv,execStatus,NULL);
   ExpressionData(theEnv,execStatus)->BreakContext = TRUE;
   ExpressionData(theEnv,execStatus)->ReturnContext = ExpressionData(theEnv,execStatus)->svContexts->rtn;

   qaction = GroupActions(theEnv,execStatus,readSource,queryInputToken,TRUE,NULL,FALSE);
   PPBackup(theEnv,execStatus);
   PPBackup(theEnv,execStatus);
   SavePPBuffer(theEnv,execStatus,queryInputToken->printForm);

   ExpressionData(theEnv,execStatus)->BreakContext = FALSE;
   if (error == TRUE)
     {
      SetParsedBindNames(theEnv,execStatus,oldBindList);
      ReturnExpression(theEnv,execStatus,top);
      return(FALSE);
     }
   if (qaction == NULL)
     {
      SetParsedBindNames(theEnv,execStatus,oldBindList);
      SyntaxErrorMessage(theEnv,execStatus,"instance-set query function");
      ReturnExpression(theEnv,execStatus,top);
      return(FALSE);
     }
   qaction->nextArg = top->argList->nextArg;
   top->argList->nextArg = qaction;
   newBindList = GetParsedBindNames(theEnv,execStatus);
   prev = NULL;
   while (newBindList != NULL)
     {
      tmpInsSetVars = insQuerySetVars;
      while (tmpInsSetVars != NULL)
        {
         if (tmpInsSetVars->value == (void *) newBindList->name)
           {
            ClearParsedBindNames(theEnv,execStatus);
            SetParsedBindNames(theEnv,execStatus,oldBindList);
            PrintErrorID(theEnv,execStatus,"INSQYPSR",3,FALSE);
            EnvPrintRouter(theEnv,execStatus,WERROR,"Cannot rebind instance-set member variable ");
            EnvPrintRouter(theEnv,execStatus,WERROR,ValueToString(tmpInsSetVars->value));
            EnvPrintRouter(theEnv,execStatus,WERROR," in function ");
            EnvPrintRouter(theEnv,execStatus,WERROR,ValueToString(ExpressionFunctionCallName(top)));
            EnvPrintRouter(theEnv,execStatus,WERROR,".\n");
            ReturnExpression(theEnv,execStatus,top);
            return(FALSE);
           }
         tmpInsSetVars = tmpInsSetVars->nextArg;
        }
      prev = newBindList;
      newBindList = newBindList->next;
     }
   if (prev == NULL)
     SetParsedBindNames(theEnv,execStatus,oldBindList);
   else
     prev->next = oldBindList;
   return(TRUE);
  }

/***********************************************************************************
  NAME         : ReplaceInstanceVariables
  DESCRIPTION  : Replaces all references to instance-variables within an
                   instance query-function with function calls to query-instance
                   (which references the instance array at run-time)
  INPUTS       : 1) The instance-variable list
                 2) A boolean expression containing variable references
                 3) A flag indicating whether to allow slot references of the type
                    <instance-query-variable>:<slot-name> for direct slot access
                    or not
                 4) Nesting depth of query functions
  RETURNS      : Nothing useful
  SIDE EFFECTS : If a SF_VARIABLE node is found and is on the list of instance
                   variables, it is replaced with a query-instance function call.
  NOTES        : Other SF_VARIABLE(S) are left alone for replacement by other
                   parsers.  This implies that a user may use defgeneric,
                   defrule, and defmessage-handler variables within a query-function
                   where they do not conflict with instance-variable names.
 ***********************************************************************************/
static void ReplaceInstanceVariables(
  void *theEnv,
  EXEC_STATUS,
  EXPRESSION *vlist,
  EXPRESSION *bexp,
  int sdirect,
  int ndepth)
  {
   EXPRESSION *eptr;
   struct FunctionDefinition *rindx_func,*rslot_func;
   int posn;

   rindx_func = FindFunction(theEnv,execStatus,"(query-instance)");
   rslot_func = FindFunction(theEnv,execStatus,"(query-instance-slot)");
   while (bexp != NULL)
     {
      if (bexp->type == SF_VARIABLE)
        {
         eptr = vlist;
         posn = 0;
         while ((eptr != NULL) ? (eptr->value != bexp->value) : FALSE)
           {
            eptr = eptr->nextArg;
            posn++;
           }
         if (eptr != NULL)
           {
            bexp->type = FCALL;
            bexp->value = (void *) rindx_func;
            eptr = GenConstant(theEnv,execStatus,INTEGER,(void *) EnvAddLong(theEnv,execStatus,(long long) ndepth));
            eptr->nextArg = GenConstant(theEnv,execStatus,INTEGER,(void *) EnvAddLong(theEnv,execStatus,(long long) posn));
            bexp->argList = eptr;
           }
         else if (sdirect == TRUE)
           ReplaceSlotReference(theEnv,execStatus,vlist,bexp,rslot_func,ndepth);
        }
      if (bexp->argList != NULL)
        {
         if (IsQueryFunction(bexp))
           ReplaceInstanceVariables(theEnv,execStatus,vlist,bexp->argList,sdirect,ndepth+1);
         else
           ReplaceInstanceVariables(theEnv,execStatus,vlist,bexp->argList,sdirect,ndepth);
        }
      bexp = bexp->nextArg;
     }
  }

/*************************************************************************
  NAME         : ReplaceSlotReference
  DESCRIPTION  : Replaces instance-set query function variable
                   references of the form: <instance-variable>:<slot-name>
                   with function calls to get these instance-slots at run
                   time
  INPUTS       : 1) The instance-set variable list
                 2) The expression containing the variable
                 3) The address of the instance slot access function
                 4) Nesting depth of query functions
  RETURNS      : Nothing useful
  SIDE EFFECTS : If the variable is a slot reference, then it is replaced
                   with the appropriate function-call.
  NOTES        : None
 *************************************************************************/
static void ReplaceSlotReference(
  void *theEnv,
  EXEC_STATUS,
  EXPRESSION *vlist,
  EXPRESSION *theExp,
  struct FunctionDefinition *func,
  int ndepth)
  {
   size_t len;
   int posn,oldpp;
   size_t i;
   register char *str;
   EXPRESSION *eptr;
   struct token itkn;

   str = ValueToString(theExp->value);
   len =  strlen(str);
   if (len < 3)
     return;
   for (i = len-2 ; i >= 1 ; i--)
     {
      if ((str[i] == INSTANCE_SLOT_REF) ? (i >= 1) : FALSE)
        {
         eptr = vlist;
         posn = 0;
         while (eptr && ((i != strlen(ValueToString(eptr->value))) ||
                         strncmp(ValueToString(eptr->value),str,
                                 (STD_SIZE) i)))
           {
            eptr = eptr->nextArg;
            posn++;
           }
         if (eptr != NULL)
           {
            OpenStringSource(theEnv,execStatus,"query-var",str+i+1,0);
            oldpp = GetPPBufferStatus(theEnv,execStatus);
            SetPPBufferStatus(theEnv,execStatus,OFF);
            GetToken(theEnv,execStatus,"query-var",&itkn);
            SetPPBufferStatus(theEnv,execStatus,oldpp);
            CloseStringSource(theEnv,execStatus,"query-var");
            theExp->type = FCALL;
            theExp->value = (void *) func;
            theExp->argList = GenConstant(theEnv,execStatus,INTEGER,(void *) EnvAddLong(theEnv,execStatus,(long long) ndepth));
            theExp->argList->nextArg =
              GenConstant(theEnv,execStatus,INTEGER,(void *) EnvAddLong(theEnv,execStatus,(long long) posn));
            theExp->argList->nextArg->nextArg = GenConstant(theEnv,execStatus,itkn.type,itkn.value);
            break;
           }
        }
     }
  }

/********************************************************************
  NAME         : IsQueryFunction
  DESCRIPTION  : Determines if an expression is a query function call
  INPUTS       : The expression
  RETURNS      : TRUE if query function call, FALSE otherwise
  SIDE EFFECTS : None
  NOTES        : None
 ********************************************************************/
static int IsQueryFunction(
  EXPRESSION *theExp)
  {
   int (*fptr)(void);

   if (theExp->type != FCALL)
     return(FALSE);
   fptr = (int (*)(void)) ExpressionFunctionPointer(theExp);
   if (fptr == (int (*)(void)) AnyInstances)
     return(TRUE);
   if (fptr == (int (*)(void)) QueryFindInstance)
     return(TRUE);
   if (fptr == (int (*)(void)) QueryFindAllInstances)
     return(TRUE);
   if (fptr == (int (*)(void)) QueryDoForInstance)
     return(TRUE);
   if (fptr == (int (*)(void)) QueryDoForAllInstances)
     return(TRUE);
   if (fptr == (int (*)(void)) DelayedQueryDoForAllInstances)
     return(TRUE);
   return(FALSE);
  }

#endif

/***************************************************
  NAME         :
  DESCRIPTION  :
  INPUTS       :
  RETURNS      :
  SIDE EFFECTS :
  NOTES        :
 ***************************************************/


