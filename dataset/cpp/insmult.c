   /*******************************************************/
   /*      "C" Language Integrated Production System      */
   /*                                                     */
   /*              CLIPS Version 6.24  06/05/06           */
   /*                                                     */
   /*           INSTANCE MULTIFIELD SLOT MODULE           */
   /*******************************************************/

/*************************************************************/
/* Purpose:  Access routines for Instance Multifield Slots   */
/*                                                           */
/* Principal Programmer(s):                                  */
/*      Brian L. Dantes                                      */
/*                                                           */
/* Contributing Programmer(s):                               */
/*                                                           */
/* Revision History:                                         */
/*      6.23: Correction for FalseSymbol/TrueSymbol. DR0859  */
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

#if OBJECT_SYSTEM

#include "argacces.h"
#include "envrnmnt.h"
#include "extnfunc.h"
#include "insfun.h"
#include "msgfun.h"
#include "msgpass.h"
#include "multifun.h"
#include "router.h"

#define _INSMULT_SOURCE_
#include "insmult.h"

/* =========================================
   *****************************************
                   CONSTANTS
   =========================================
   ***************************************** */
#define INSERT         0
#define REPLACE        1
#define DELETE_OP      2

/* =========================================
   *****************************************
      INTERNALLY VISIBLE FUNCTION HEADERS
   =========================================
   ***************************************** */

static INSTANCE_TYPE *CheckMultifieldSlotInstance(void *,EXEC_STATUS,char *);
static INSTANCE_SLOT *CheckMultifieldSlotModify(void *,EXEC_STATUS,int,char *,INSTANCE_TYPE *,
                                       EXPRESSION *,long *,long *,DATA_OBJECT *);
static void AssignSlotToDataObject(DATA_OBJECT *,INSTANCE_SLOT *);

/* =========================================
   *****************************************
          EXTERNALLY VISIBLE FUNCTIONS
   =========================================
   ***************************************** */

#if (! RUN_TIME)

/***************************************************
  NAME         : SetupInstanceMultifieldCommands
  DESCRIPTION  : Defines function interfaces for
                 manipulating instance multislots
  INPUTS       : None
  RETURNS      : Nothing useful
  SIDE EFFECTS : Functions defined to KB
  NOTES        : None
 ***************************************************/
globle void SetupInstanceMultifieldCommands(
  void *theEnv,
  EXEC_STATUS)
  {
   /* ===================================
      Old version 5.1 compatibility names
      =================================== */
   EnvDefineFunction2(theEnv,execStatus,"direct-mv-replace",'b',PTIEF DirectMVReplaceCommand,
                   "DirectMVReplaceCommand","4**wii");
   EnvDefineFunction2(theEnv,execStatus,"direct-mv-insert",'b',PTIEF DirectMVInsertCommand,
                   "DirectMVInsertCommand","3**wi");
   EnvDefineFunction2(theEnv,execStatus,"direct-mv-delete",'b',PTIEF DirectMVDeleteCommand,
                   "DirectMVDeleteCommand","33iw");
   EnvDefineFunction2(theEnv,execStatus,"mv-slot-replace",'u',PTIEF MVSlotReplaceCommand,
                   "MVSlotReplaceCommand","5*uewii");
   EnvDefineFunction2(theEnv,execStatus,"mv-slot-insert",'u',PTIEF MVSlotInsertCommand,
                   "MVSlotInsertCommand","4*uewi");
   EnvDefineFunction2(theEnv,execStatus,"mv-slot-delete",'u',PTIEF MVSlotDeleteCommand,
                   "MVSlotDeleteCommand","44iew");

   /* =====================
      New version 6.0 names
      ===================== */
   EnvDefineFunction2(theEnv,execStatus,"slot-direct-replace$",'b',PTIEF DirectMVReplaceCommand,
                   "DirectMVReplaceCommand","4**wii");
   EnvDefineFunction2(theEnv,execStatus,"slot-direct-insert$",'b',PTIEF DirectMVInsertCommand,
                   "DirectMVInsertCommand","3**wi");
   EnvDefineFunction2(theEnv,execStatus,"slot-direct-delete$",'b',PTIEF DirectMVDeleteCommand,
                   "DirectMVDeleteCommand","33iw");
   EnvDefineFunction2(theEnv,execStatus,"slot-replace$",'u',PTIEF MVSlotReplaceCommand,
                   "MVSlotReplaceCommand","5*uewii");
   EnvDefineFunction2(theEnv,execStatus,"slot-insert$",'u',PTIEF MVSlotInsertCommand,
                   "MVSlotInsertCommand","4*uewi");
   EnvDefineFunction2(theEnv,execStatus,"slot-delete$",'u',PTIEF MVSlotDeleteCommand,
                   "MVSlotDeleteCommand","44iew");
  }

#endif

/***********************************************************************************
  NAME         : MVSlotReplaceCommand
  DESCRIPTION  : Allows user to replace a specified field of a multi-value slot
                 The slot is directly read (w/o a get- message) and the new
                   slot-value is placed via a put- message.
                 This function is not valid for single-value slots.
  INPUTS       : Caller's result buffer
  RETURNS      : TRUE if multi-value slot successfully modified,
                 FALSE otherwise
  SIDE EFFECTS : Put messsage sent for slot
  NOTES        : H/L Syntax : (slot-replace$ <instance> <slot>
                                 <range-begin> <range-end> <value>)
 ***********************************************************************************/
globle void MVSlotReplaceCommand(
  void *theEnv,
  EXEC_STATUS,
  DATA_OBJECT *result)
  {
   DATA_OBJECT newval,newseg,oldseg;
   INSTANCE_TYPE *ins;
   INSTANCE_SLOT *sp;
   long rb,re;
   EXPRESSION arg;

   result->type = SYMBOL;
   result->value = EnvFalseSymbol(theEnv,execStatus);
   ins = CheckMultifieldSlotInstance(theEnv,execStatus,"slot-replace$");
   if (ins == NULL)
     return;
   sp = CheckMultifieldSlotModify(theEnv,execStatus,REPLACE,"slot-replace$",ins,
                            GetFirstArgument()->nextArg,&rb,&re,&newval);
   if (sp == NULL)
     return;
   AssignSlotToDataObject(&oldseg,sp);
   if (ReplaceMultiValueField(theEnv,execStatus,&newseg,&oldseg,rb,re,&newval,"slot-replace$") == FALSE)
     return;
   arg.type = MULTIFIELD;
   arg.value = (void *) &newseg;
   arg.nextArg = NULL;
   arg.argList = NULL;
   DirectMessage(theEnv,execStatus,sp->desc->overrideMessage,ins,result,&arg);
  }

/***********************************************************************************
  NAME         : MVSlotInsertCommand
  DESCRIPTION  : Allows user to insert a specified field of a multi-value slot
                 The slot is directly read (w/o a get- message) and the new
                   slot-value is placed via a put- message.
                 This function is not valid for single-value slots.
  INPUTS       : Caller's result buffer
  RETURNS      : TRUE if multi-value slot successfully modified, FALSE otherwise
  SIDE EFFECTS : Put messsage sent for slot
  NOTES        : H/L Syntax : (slot-insert$ <instance> <slot> <index> <value>)
 ***********************************************************************************/
globle void MVSlotInsertCommand(
  void *theEnv,
  EXEC_STATUS,
  DATA_OBJECT *result)
  {
   DATA_OBJECT newval,newseg,oldseg;
   INSTANCE_TYPE *ins;
   INSTANCE_SLOT *sp;
   long theIndex;
   EXPRESSION arg;

   result->type = SYMBOL;
   result->value = EnvFalseSymbol(theEnv,execStatus);
   ins = CheckMultifieldSlotInstance(theEnv,execStatus,"slot-insert$");
   if (ins == NULL)
     return;
   sp = CheckMultifieldSlotModify(theEnv,execStatus,INSERT,"slot-insert$",ins,
                            GetFirstArgument()->nextArg,&theIndex,NULL,&newval);
   if (sp == NULL)
     return;
   AssignSlotToDataObject(&oldseg,sp);
   if (InsertMultiValueField(theEnv,execStatus,&newseg,&oldseg,theIndex,&newval,"slot-insert$") == FALSE)
     return;
   arg.type = MULTIFIELD;
   arg.value = (void *) &newseg;
   arg.nextArg = NULL;
   arg.argList = NULL;
   DirectMessage(theEnv,execStatus,sp->desc->overrideMessage,ins,result,&arg);
  }

/***********************************************************************************
  NAME         : MVSlotDeleteCommand
  DESCRIPTION  : Allows user to delete a specified field of a multi-value slot
                 The slot is directly read (w/o a get- message) and the new
                   slot-value is placed via a put- message.
                 This function is not valid for single-value slots.
  INPUTS       : Caller's result buffer
  RETURNS      : TRUE if multi-value slot successfully modified, FALSE otherwise
  SIDE EFFECTS : Put message sent for slot
  NOTES        : H/L Syntax : (slot-delete$ <instance> <slot>
                                 <range-begin> <range-end>)
 ***********************************************************************************/
globle void MVSlotDeleteCommand(
  void *theEnv,
  EXEC_STATUS,
  DATA_OBJECT *result)
  {
   DATA_OBJECT newseg,oldseg;
   INSTANCE_TYPE *ins;
   INSTANCE_SLOT *sp;
   long rb,re;
   EXPRESSION arg;

   result->type = SYMBOL;
   result->value = EnvFalseSymbol(theEnv,execStatus);
   ins = CheckMultifieldSlotInstance(theEnv,execStatus,"slot-delete$");
   if (ins == NULL)
     return;
   sp = CheckMultifieldSlotModify(theEnv,execStatus,DELETE_OP,"slot-delete$",ins,
                            GetFirstArgument()->nextArg,&rb,&re,NULL);
   if (sp == NULL)
     return;
   AssignSlotToDataObject(&oldseg,sp);
   if (DeleteMultiValueField(theEnv,execStatus,&newseg,&oldseg,rb,re,"slot-delete$") == FALSE)
     return;
   arg.type = MULTIFIELD;
   arg.value = (void *) &newseg;
   arg.nextArg = NULL;
   arg.argList = NULL;
   DirectMessage(theEnv,execStatus,sp->desc->overrideMessage,ins,result,&arg);
  }

/*****************************************************************
  NAME         : DirectMVReplaceCommand
  DESCRIPTION  : Directly replaces a slot's value
  INPUTS       : None
  RETURNS      : TRUE if put OK, FALSE otherwise
  SIDE EFFECTS : Slot modified
  NOTES        : H/L Syntax: (direct-slot-replace$ <slot>
                                <range-begin> <range-end> <value>)
 *****************************************************************/
globle intBool DirectMVReplaceCommand(
  void *theEnv,
  EXEC_STATUS)
  {
   INSTANCE_SLOT *sp;
   INSTANCE_TYPE *ins;
   long rb,re;
   DATA_OBJECT newval,newseg,oldseg;

   if (CheckCurrentMessage(theEnv,execStatus,"direct-slot-replace$",TRUE) == FALSE)
     return(FALSE);
   ins = GetActiveInstance(theEnv,execStatus);
   sp = CheckMultifieldSlotModify(theEnv,execStatus,REPLACE,"direct-slot-replace$",ins,
                            GetFirstArgument(),&rb,&re,&newval);
   if (sp == NULL)
     return(FALSE);
   AssignSlotToDataObject(&oldseg,sp);
   if (ReplaceMultiValueField(theEnv,execStatus,&newseg,&oldseg,rb,re,&newval,"direct-slot-replace$")
           == FALSE)
     return(FALSE);
   if (PutSlotValue(theEnv,execStatus,ins,sp,&newseg,&newval,"function direct-slot-replace$"))
     return(TRUE);
   return(FALSE);
  }

/************************************************************************
  NAME         : DirectMVInsertCommand
  DESCRIPTION  : Directly inserts a slot's value
  INPUTS       : None
  RETURNS      : TRUE if put OK, FALSE otherwise
  SIDE EFFECTS : Slot modified
  NOTES        : H/L Syntax: (direct-slot-insert$ <slot> <index> <value>)
 ************************************************************************/
globle intBool DirectMVInsertCommand(
  void *theEnv,
  EXEC_STATUS)
  {
   INSTANCE_SLOT *sp;
   INSTANCE_TYPE *ins;
   long theIndex;
   DATA_OBJECT newval,newseg,oldseg;

   if (CheckCurrentMessage(theEnv,execStatus,"direct-slot-insert$",TRUE) == FALSE)
     return(FALSE);
   ins = GetActiveInstance(theEnv,execStatus);
   sp = CheckMultifieldSlotModify(theEnv,execStatus,INSERT,"direct-slot-insert$",ins,
                            GetFirstArgument(),&theIndex,NULL,&newval);
   if (sp == NULL)
     return(FALSE);
   AssignSlotToDataObject(&oldseg,sp);
   if (InsertMultiValueField(theEnv,execStatus,&newseg,&oldseg,theIndex,&newval,"direct-slot-insert$")
          == FALSE)
     return(FALSE);
   if (PutSlotValue(theEnv,execStatus,ins,sp,&newseg,&newval,"function direct-slot-insert$"))
     return(TRUE);
   return(FALSE);
  }

/*****************************************************************
  NAME         : DirectMVDeleteCommand
  DESCRIPTION  : Directly deletes a slot's value
  INPUTS       : None
  RETURNS      : TRUE if put OK, FALSE otherwise
  SIDE EFFECTS : Slot modified
  NOTES        : H/L Syntax: (direct-slot-delete$ <slot>
                                <range-begin> <range-end>)
 *****************************************************************/
globle intBool DirectMVDeleteCommand(
  void *theEnv,
  EXEC_STATUS)
  {
   INSTANCE_SLOT *sp;
   INSTANCE_TYPE *ins;
   long rb,re;
   DATA_OBJECT newseg,oldseg;

   if (CheckCurrentMessage(theEnv,execStatus,"direct-slot-delete$",TRUE) == FALSE)
     return(FALSE);
   ins = GetActiveInstance(theEnv,execStatus);
   sp = CheckMultifieldSlotModify(theEnv,execStatus,DELETE_OP,"direct-slot-delete$",ins,
                                  GetFirstArgument(),&rb,&re,NULL);
   if (sp == NULL)
     return(FALSE);
   AssignSlotToDataObject(&oldseg,sp);
   if (DeleteMultiValueField(theEnv,execStatus,&newseg,&oldseg,rb,re,"direct-slot-delete$")
         == FALSE)
     return(FALSE);
   if (PutSlotValue(theEnv,execStatus,ins,sp,&newseg,&oldseg,"function direct-slot-delete$"))
     return(TRUE);
   return(FALSE);
  }

/* =========================================
   *****************************************
          INTERNALLY VISIBLE FUNCTIONS
   =========================================
   ***************************************** */

/**********************************************************************
  NAME         : CheckMultifieldSlotInstance
  DESCRIPTION  : Gets the instance for the functions slot-replace$,
                    insert and delete
  INPUTS       : The function name
  RETURNS      : The instance address, NULL on errors
  SIDE EFFECTS : None
  NOTES        : None
 **********************************************************************/
static INSTANCE_TYPE *CheckMultifieldSlotInstance(
  void *theEnv,
  EXEC_STATUS,
  char *func)
  {
   INSTANCE_TYPE *ins;
   DATA_OBJECT temp;

   if (EnvArgTypeCheck(theEnv,execStatus,func,1,INSTANCE_OR_INSTANCE_NAME,&temp) == FALSE)
     {
      SetEvaluationError(theEnv,execStatus,TRUE);
      return(NULL);
     }
   if (temp.type == INSTANCE_ADDRESS)
     {
      ins = (INSTANCE_TYPE *) temp.value;
      if (ins->garbage == 1)
        {
         StaleInstanceAddress(theEnv,execStatus,func,0);
         SetEvaluationError(theEnv,execStatus,TRUE);
         return(NULL);
        }
     }
   else
     {
      ins = FindInstanceBySymbol(theEnv,execStatus,(SYMBOL_HN *) temp.value);
      if (ins == NULL)
        NoInstanceError(theEnv,execStatus,ValueToString(temp.value),func);
     }
   return(ins);
  }

/*********************************************************************
  NAME         : CheckMultifieldSlotModify
  DESCRIPTION  : For the functions slot-replace$, insert, & delete
                    as well as direct-slot-replace$, insert, & delete
                    this function gets the slot, index, and optional
                    field-value for these functions
  INPUTS       : 1) A code indicating the type of operation
                      INSERT    (0) : Requires one index
                      REPLACE   (1) : Requires two indices
                      DELETE_OP (2) : Requires two indices
                 2) Function name-string
                 3) Instance address
                 4) Argument expression chain
                 5) Caller's buffer for index (or beginning of range)
                 6) Caller's buffer for end of range
                     (can be NULL for INSERT)
                 7) Caller's new-field value buffer
                     (can be NULL for DELETE_OP)
  RETURNS      : The address of the instance-slot,
                    NULL on errors
  SIDE EFFECTS : Caller's index buffer set
                 Caller's new-field value buffer set (if not NULL)
                   Will allocate an ephemeral segment to store more
                     than 1 new field value
                 EvaluationError set on errors
  NOTES        : Assume the argument chain is at least 2
                   expressions deep - slot, index, and optional values
 *********************************************************************/
static INSTANCE_SLOT *CheckMultifieldSlotModify(
  void *theEnv,
  EXEC_STATUS,
  int code,
  char *func,
  INSTANCE_TYPE *ins,
  EXPRESSION *args,
  long *rb,
  long *re,
  DATA_OBJECT *newval)
  {
   DATA_OBJECT temp;
   INSTANCE_SLOT *sp;
   int start;

   start = (args == GetFirstArgument()) ? 1 : 2;
   execStatus->EvaluationError = FALSE;
   EvaluateExpression(theEnv,execStatus,args,&temp);
   if (temp.type != SYMBOL)
     {
      ExpectedTypeError1(theEnv,execStatus,func,start,"symbol");
      SetEvaluationError(theEnv,execStatus,TRUE);
      return(NULL);
     }
   sp = FindInstanceSlot(theEnv,execStatus,ins,(SYMBOL_HN *) temp.value);
   if (sp == NULL)
     {
      SlotExistError(theEnv,execStatus,ValueToString(temp.value),func);
      return(NULL);
     }
   if (sp->desc->multiple == 0)
     {
      PrintErrorID(theEnv,execStatus,"INSMULT",1,FALSE);
      EnvPrintRouter(theEnv,execStatus,WERROR,"Function ");
      EnvPrintRouter(theEnv,execStatus,WERROR,func);
      EnvPrintRouter(theEnv,execStatus,WERROR," cannot be used on single-field slot ");
      EnvPrintRouter(theEnv,execStatus,WERROR,ValueToString(sp->desc->slotName->name));
      EnvPrintRouter(theEnv,execStatus,WERROR," in instance ");
      EnvPrintRouter(theEnv,execStatus,WERROR,ValueToString(ins->name));
      EnvPrintRouter(theEnv,execStatus,WERROR,".\n");
      SetEvaluationError(theEnv,execStatus,TRUE);
      return(NULL);
     }
   EvaluateExpression(theEnv,execStatus,args->nextArg,&temp);
   if (temp.type != INTEGER)
     {
      ExpectedTypeError1(theEnv,execStatus,func,start+1,"integer");
      SetEvaluationError(theEnv,execStatus,TRUE);
      return(NULL);
     }
   args = args->nextArg->nextArg;
   *rb = (long) ValueToLong(temp.value);
   if ((code == REPLACE) || (code == DELETE_OP))
     {
      EvaluateExpression(theEnv,execStatus,args,&temp);
      if (temp.type != INTEGER)
        {
         ExpectedTypeError1(theEnv,execStatus,func,start+2,"integer");
         SetEvaluationError(theEnv,execStatus,TRUE);
         return(NULL);
        }
      *re = (long) ValueToLong(temp.value);
      args = args->nextArg;
     }
   if ((code == INSERT) || (code == REPLACE))
     {
      if (EvaluateAndStoreInDataObject(theEnv,execStatus,1,args,newval,TRUE) == FALSE)
        return(NULL);
     }
   return(sp);
  }

/***************************************************
  NAME         : AssignSlotToDataObject
  DESCRIPTION  : Assigns the value of a multifield
                 slot to a data object
  INPUTS       : 1) The data object buffer
                 2) The instance slot
  RETURNS      : Nothing useful
  SIDE EFFECTS : Data object fields set
  NOTES        : Assumes slot is a multislot
 ***************************************************/
static void AssignSlotToDataObject(
  DATA_OBJECT *theDataObject,
  INSTANCE_SLOT *theSlot)
  {
   theDataObject->type = (unsigned short) theSlot->type;
   theDataObject->value = theSlot->value;
   theDataObject->begin = 0;
   SetpDOEnd(theDataObject,GetInstanceSlotLength(theSlot));
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


