   /*******************************************************/
   /*      "C" Language Integrated Production System      */
   /*                                                     */
   /*             CLIPS Version 6.24  07/01/05            */
   /*                                                     */
   /*                PRINT UTILITY MODULE                 */
   /*******************************************************/

/*************************************************************/
/* Purpose: Utility routines for printing various items      */
/*   and messages.                                           */
/*                                                           */
/* Principal Programmer(s):                                  */
/*      Gary D. Riley                                        */
/*                                                           */
/* Contributing Programmer(s):                               */
/*      Brian L. Dantes                                      */
/*                                                           */
/* Revision History:                                         */
/*                                                           */
/*      6.24: Link error occurs for the SlotExistError       */
/*            function when OBJECT_SYSTEM is set to 0 in     */
/*            setup.h. DR0865                                */
/*                                                           */
/*            Added DataObjectToString function.             */
/*                                                           */
/*            Added SlotExistError function.                 */
/*                                                           */
/*************************************************************/

#define _PRNTUTIL_SOURCE_

#include <stdio.h>
#define _STDIO_INCLUDED_
#include <string.h>

#include "setup.h"

#include "constant.h"
#include "envrnmnt.h"
#include "symbol.h"
#include "utility.h"
#include "evaluatn.h"
#include "argacces.h"
#include "router.h"
#include "multifun.h"
#include "fact/fact_manager.h"
#include "inscom.h"
#include "insmngr.h"
#include "memalloc.h"
#include "sysdep.h"

#include "prntutil.h"

/*****************************************************/
/* InitializePrintUtilityData: Allocates environment */
/*    data for print utility routines.               */
/*****************************************************/
globle void InitializePrintUtilityData(
  void *theEnv,
  EXEC_STATUS)
  {
   AllocateEnvironmentData(theEnv,execStatus,PRINT_UTILITY_DATA,sizeof(struct printUtilityData),NULL);
  }

/***********************************************************/
/* PrintInChunks:  Prints a string in chunks to accomodate */
/*   systems which have a limit on the maximum size of a   */
/*   string which can be printed.                          */
/***********************************************************/
globle void PrintInChunks(
  void *theEnv,
  EXEC_STATUS,
  char *logicalName,
  char *bigString)
  {
   char tc, *subString;

   subString = bigString;

   if (subString == NULL) return;

   while (((int) strlen(subString)) > 500)
     {
      if (execStatus->HaltExecution) return;
      tc = subString[500];
      subString[500] = EOS;
      EnvPrintRouter(theEnv,execStatus,logicalName,subString);
      subString[500] = tc;
      subString += 500;
     }

   EnvPrintRouter(theEnv,execStatus,logicalName,subString);
  }

/************************************************************/
/* PrintFloat: Controls printout of floating point numbers. */
/************************************************************/
globle void PrintFloat(
  void *theEnv,
  EXEC_STATUS,
  char *fileid,
  double number)
  {
   char *theString;

   theString = FloatToString(theEnv,execStatus,number);
   EnvPrintRouter(theEnv,execStatus,fileid,theString);
  }

/****************************************************/
/* PrintLongInteger: Controls printout of integers. */
/****************************************************/
globle void PrintLongInteger(
  void *theEnv,
  EXEC_STATUS,
  char *logicalName,
  long long number)
  {
   char printBuffer[32];

   gensprintf(printBuffer,"%lld",number);
   EnvPrintRouter(theEnv,execStatus,logicalName,printBuffer);
  }

/**************************************/
/* PrintAtom: Prints an atomic value. */
/**************************************/
globle void PrintAtom(
  void *theEnv,
  EXEC_STATUS,
  char *logicalName,
  int type,
  void *value)
  {
   struct externalAddressHashNode *theAddress;
   char buffer[20];

   switch (type)
     {
      case FLOAT:
        PrintFloat(theEnv,execStatus,logicalName,ValueToDouble(value));
        break;
      case INTEGER:
        PrintLongInteger(theEnv,execStatus,logicalName,ValueToLong(value));
        break;
      case SYMBOL:
        EnvPrintRouter(theEnv,execStatus,logicalName,ValueToString(value));
        break;
      case STRING:
        if (PrintUtilityData(theEnv,execStatus)->PreserveEscapedCharacters)
          { EnvPrintRouter(theEnv,execStatus,logicalName,StringPrintForm(theEnv,execStatus,ValueToString(value))); }
        else
          {
           EnvPrintRouter(theEnv,execStatus,logicalName,"\"");
           EnvPrintRouter(theEnv,execStatus,logicalName,ValueToString(value));
           EnvPrintRouter(theEnv,execStatus,logicalName,"\"");
          }
        break;

      case EXTERNAL_ADDRESS:
        theAddress = (struct externalAddressHashNode *) value;
        
        if (PrintUtilityData(theEnv,execStatus)->AddressesToStrings) EnvPrintRouter(theEnv,execStatus,logicalName,"\"");
        
        if ((EvaluationData(theEnv,execStatus)->ExternalAddressTypes[theAddress->type] != NULL) &&
            (EvaluationData(theEnv,execStatus)->ExternalAddressTypes[theAddress->type]->longPrintFunction != NULL))
          { (*EvaluationData(theEnv,execStatus)->ExternalAddressTypes[theAddress->type]->longPrintFunction)(theEnv,execStatus,logicalName,value); }
        else
          {
           EnvPrintRouter(theEnv,execStatus,logicalName,"<Pointer-");
        
           gensprintf(buffer,"%d-",theAddress->type);
           EnvPrintRouter(theEnv,execStatus,logicalName,buffer);
        
           gensprintf(buffer,"%p",ValueToExternalAddress(value));
           EnvPrintRouter(theEnv,execStatus,logicalName,buffer);
           EnvPrintRouter(theEnv,execStatus,logicalName,">");
          }
          
        if (PrintUtilityData(theEnv,execStatus)->AddressesToStrings) EnvPrintRouter(theEnv,execStatus,logicalName,"\"");
        break;

#if OBJECT_SYSTEM
      case INSTANCE_NAME:
        EnvPrintRouter(theEnv,execStatus,logicalName,"[");
        EnvPrintRouter(theEnv,execStatus,logicalName,ValueToString(value));
        EnvPrintRouter(theEnv,execStatus,logicalName,"]");
        break;
#endif

      case RVOID:
        break;

      default:
        if (EvaluationData(theEnv,execStatus)->PrimitivesArray[type] == NULL) break;
        if (EvaluationData(theEnv,execStatus)->PrimitivesArray[type]->longPrintFunction == NULL)
          {
           EnvPrintRouter(theEnv,execStatus,logicalName,"<unknown atom type>");
           break;
          }
        (*EvaluationData(theEnv,execStatus)->PrimitivesArray[type]->longPrintFunction)(theEnv,execStatus,logicalName,value);
        break;
     }
  }

/**********************************************************/
/* PrintTally: Prints a tally count indicating the number */
/*   of items that have been displayed. Used by functions */
/*   such as list-defrules.                               */
/**********************************************************/
globle void PrintTally(
  void *theEnv,
  EXEC_STATUS,
  char *logicalName,
  long long count,
  char *singular,
  char *plural)
  {
   if (count == 0) return;

   EnvPrintRouter(theEnv,execStatus,logicalName,"For a total of ");
   PrintLongInteger(theEnv,execStatus,logicalName,count);
   EnvPrintRouter(theEnv,execStatus,logicalName," ");

   if (count == 1) EnvPrintRouter(theEnv,execStatus,logicalName,singular);
   else EnvPrintRouter(theEnv,execStatus,logicalName,plural);

   EnvPrintRouter(theEnv,execStatus,logicalName,".\n");
  }

/********************************************/
/* PrintErrorID: Prints the module name and */
/*   error ID for an error message.         */
/********************************************/
globle void PrintErrorID(
  void *theEnv,
  EXEC_STATUS,
  char *module,
  int errorID,
  int printCR)
  {
   if (printCR) EnvPrintRouter(theEnv,execStatus,WERROR,"\n");
   EnvPrintRouter(theEnv,execStatus,WERROR,"[");
   EnvPrintRouter(theEnv,execStatus,WERROR,module);
   PrintLongInteger(theEnv,execStatus,WERROR,(long int) errorID);
   EnvPrintRouter(theEnv,execStatus,WERROR,"] ");
  }

/**********************************************/
/* PrintWarningID: Prints the module name and */
/*   warning ID for a warning message.        */
/**********************************************/
globle void PrintWarningID(
  void *theEnv,
  EXEC_STATUS,
  char *module,
  int warningID,
  int printCR)
  {
   if (printCR) EnvPrintRouter(theEnv,execStatus,WWARNING,"\n");
   EnvPrintRouter(theEnv,execStatus,WWARNING,"[");
   EnvPrintRouter(theEnv,execStatus,WWARNING,module);
   PrintLongInteger(theEnv,execStatus,WWARNING,(long int) warningID);
   EnvPrintRouter(theEnv,execStatus,WWARNING,"] WARNING: ");
  }

/***************************************************/
/* CantFindItemErrorMessage: Generic error message */
/*  when an "item" can not be found.               */
/***************************************************/
globle void CantFindItemErrorMessage(
  void *theEnv,
  EXEC_STATUS,
  char *itemType,
  char *itemName)
  {
   PrintErrorID(theEnv,execStatus,"PRNTUTIL",1,FALSE);
   EnvPrintRouter(theEnv,execStatus,WERROR,"Unable to find ");
   EnvPrintRouter(theEnv,execStatus,WERROR,itemType);
   EnvPrintRouter(theEnv,execStatus,WERROR," ");
   EnvPrintRouter(theEnv,execStatus,WERROR,itemName);
   EnvPrintRouter(theEnv,execStatus,WERROR,".\n");
  }

/*****************************************************/
/* CantFindItemInFunctionErrorMessage: Generic error */
/*  message when an "item" can not be found.         */
/*****************************************************/
globle void CantFindItemInFunctionErrorMessage(
  void *theEnv,
  EXEC_STATUS,
  char *itemType,
  char *itemName,
  char *func)
  {
   PrintErrorID(theEnv,execStatus,"PRNTUTIL",1,FALSE);
   EnvPrintRouter(theEnv,execStatus,WERROR,"Unable to find ");
   EnvPrintRouter(theEnv,execStatus,WERROR,itemType);
   EnvPrintRouter(theEnv,execStatus,WERROR," ");
   EnvPrintRouter(theEnv,execStatus,WERROR,itemName);
   EnvPrintRouter(theEnv,execStatus,WERROR," in function ");
   EnvPrintRouter(theEnv,execStatus,WERROR,func);
   EnvPrintRouter(theEnv,execStatus,WERROR,".\n");
  }

/*****************************************************/
/* CantDeleteItemErrorMessage: Generic error message */
/*  when an "item" can not be deleted.               */
/*****************************************************/
globle void CantDeleteItemErrorMessage(
  void *theEnv,
  EXEC_STATUS,
  char *itemType,
  char *itemName)
  {
   PrintErrorID(theEnv,execStatus,"PRNTUTIL",4,FALSE);
   EnvPrintRouter(theEnv,execStatus,WERROR,"Unable to delete ");
   EnvPrintRouter(theEnv,execStatus,WERROR,itemType);
   EnvPrintRouter(theEnv,execStatus,WERROR," ");
   EnvPrintRouter(theEnv,execStatus,WERROR,itemName);
   EnvPrintRouter(theEnv,execStatus,WERROR,".\n");
  }

/****************************************************/
/* AlreadyParsedErrorMessage: Generic error message */
/*  when an "item" has already been parsed.         */
/****************************************************/
globle void AlreadyParsedErrorMessage(
  void *theEnv,
  EXEC_STATUS,
  char *itemType,
  char *itemName)
  {
   PrintErrorID(theEnv,execStatus,"PRNTUTIL",5,TRUE);
   EnvPrintRouter(theEnv,execStatus,WERROR,"The ");
   if (itemType != NULL) EnvPrintRouter(theEnv,execStatus,WERROR,itemType);
   if (itemName != NULL) EnvPrintRouter(theEnv,execStatus,WERROR,itemName);
   EnvPrintRouter(theEnv,execStatus,WERROR," has already been parsed.\n");
  }

/*********************************************************/
/* SyntaxErrorMessage: Generalized syntax error message. */
/*********************************************************/
globle void SyntaxErrorMessage(
  void *theEnv,
  EXEC_STATUS,
  char *location)
  {
   PrintErrorID(theEnv,execStatus,"PRNTUTIL",2,TRUE);
   EnvPrintRouter(theEnv,execStatus,WERROR,"Syntax Error");
   if (location != NULL)
     {
      EnvPrintRouter(theEnv,execStatus,WERROR,":  Check appropriate syntax for ");
      EnvPrintRouter(theEnv,execStatus,WERROR,location);
     }

   EnvPrintRouter(theEnv,execStatus,WERROR,".\n");
   SetEvaluationError(theEnv,execStatus,TRUE);
  }

/****************************************************/
/* LocalVariableErrorMessage: Generic error message */
/*  when a local variable is accessed by an "item"  */
/*  which can not access local variables.           */
/****************************************************/
globle void LocalVariableErrorMessage(
  void *theEnv,
  EXEC_STATUS,
  char *byWhat)
  {
   PrintErrorID(theEnv,execStatus,"PRNTUTIL",6,TRUE);
   EnvPrintRouter(theEnv,execStatus,WERROR,"Local variables can not be accessed by ");
   EnvPrintRouter(theEnv,execStatus,WERROR,byWhat);
   EnvPrintRouter(theEnv,execStatus,WERROR,".\n");
  }

/******************************************/
/* SystemError: Generalized error message */
/*   for major internal errors.           */
/******************************************/
globle void SystemError(
  void *theEnv,
  EXEC_STATUS,
  char *module,
  int errorID)
  {
   PrintErrorID(theEnv,execStatus,"PRNTUTIL",3,TRUE);

   EnvPrintRouter(theEnv,execStatus,WERROR,"\n*** ");
   EnvPrintRouter(theEnv,execStatus,WERROR,APPLICATION_NAME);
   EnvPrintRouter(theEnv,execStatus,WERROR," SYSTEM ERROR ***\n");

   EnvPrintRouter(theEnv,execStatus,WERROR,"ID = ");
   EnvPrintRouter(theEnv,execStatus,WERROR,module);
   PrintLongInteger(theEnv,execStatus,WERROR,(long int) errorID);
   EnvPrintRouter(theEnv,execStatus,WERROR,"\n");

   EnvPrintRouter(theEnv,execStatus,WERROR,APPLICATION_NAME);
   EnvPrintRouter(theEnv,execStatus,WERROR," data structures are in an inconsistent or corrupted state.\n");
   EnvPrintRouter(theEnv,execStatus,WERROR,"This error may have occurred from errors in user defined code.\n");
   EnvPrintRouter(theEnv,execStatus,WERROR,"**************************\n");
  }

/*******************************************************/
/* DivideByZeroErrorMessage: Generalized error message */
/*   for when a function attempts to divide by zero.   */
/*******************************************************/
globle void DivideByZeroErrorMessage(
  void *theEnv,
  EXEC_STATUS,
  char *functionName)
  {
   PrintErrorID(theEnv,execStatus,"PRNTUTIL",7,FALSE);
   EnvPrintRouter(theEnv,execStatus,WERROR,"Attempt to divide by zero in ");
   EnvPrintRouter(theEnv,execStatus,WERROR,functionName);
   EnvPrintRouter(theEnv,execStatus,WERROR," function.\n");
  }

/*******************************************************/
/* FloatToString: Converts number to KB string format. */
/*******************************************************/
globle char *FloatToString(
  void *theEnv,
  EXEC_STATUS,
  double number)
  {
   char floatString[40];
   int i;
   char x;
   void *thePtr;

   gensprintf(floatString,"%.15g",number);

   for (i = 0; (x = floatString[i]) != '\0'; i++)
     {
      if ((x == '.') || (x == 'e'))
        {
         thePtr = EnvAddSymbol(theEnv,execStatus,floatString);
         return(ValueToString(thePtr));
        }
     }

   genstrcat(floatString,".0");

   thePtr = EnvAddSymbol(theEnv,execStatus,floatString);
   return(ValueToString(thePtr));
  }

/*******************************************************************/
/* LongIntegerToString: Converts long integer to KB string format. */
/*******************************************************************/
globle char *LongIntegerToString(
  void *theEnv,
  EXEC_STATUS,
  long long number)
  {
   char buffer[50];
   void *thePtr;

   gensprintf(buffer,"%lld",number);

   thePtr = EnvAddSymbol(theEnv,execStatus,buffer);
   return(ValueToString(thePtr));
  }

/*******************************************************************/
/* DataObjectToString: Converts a DATA_OBJECT to KB string format. */
/*******************************************************************/
globle char *DataObjectToString(
  void *theEnv,
  EXEC_STATUS,
  DATA_OBJECT *theDO)
  {
   void *thePtr;
   char *theString, *newString;
   char *prefix, *postfix;
   size_t length;
   struct externalAddressHashNode *theAddress;
   char buffer[30];
   
   switch (GetpType(theDO))
     {
      case MULTIFIELD:
         prefix = "(";
         theString = ValueToString(ImplodeMultifield(theEnv,execStatus,theDO));
         postfix = ")";
         break;
         
      case STRING:
         prefix = "\"";
         theString = DOPToString(theDO);
         postfix = "\"";
         break;
         
      case INSTANCE_NAME:
         prefix = "[";
         theString = DOPToString(theDO);
         postfix = "]";
         break;
         
      case SYMBOL:
         return(DOPToString(theDO));
         
      case FLOAT:
         return(FloatToString(theEnv,execStatus,DOPToDouble(theDO)));
         
      case INTEGER:
         return(LongIntegerToString(theEnv,execStatus,DOPToLong(theDO)));
         
      case RVOID:
         return("");

#if OBJECT_SYSTEM
      case INSTANCE_ADDRESS:
         thePtr = DOPToPointer(theDO);

         if (thePtr == (void *) &InstanceData(theEnv,execStatus)->DummyInstance)
           { return("<Dummy Instance>"); }
           
         if (((struct instance *) thePtr)->garbage)
           {
            prefix = "<Stale Instance-";
            theString = ValueToString(((struct instance *) thePtr)->name);
            postfix = ">";
           }
         else
           {
            prefix = "<Instance-";
            theString = ValueToString(GetFullInstanceName(theEnv,execStatus,(INSTANCE_TYPE *) thePtr));
            postfix = ">";
           }
           
        break;
#endif
      
      case EXTERNAL_ADDRESS:
        theAddress = (struct externalAddressHashNode *) DOPToPointer(theDO);
        /* TBD Need specific routine for creating name string. */
        gensprintf(buffer,"<Pointer-%d-%p>",(int) theAddress->type,DOPToExternalAddress(theDO));
        thePtr = EnvAddSymbol(theEnv,execStatus,buffer);
        return(ValueToString(thePtr));

#if DEFTEMPLATE_CONSTRUCT      
      case FACT_ADDRESS:
         if (DOPToPointer(theDO) == (void *) &FactData(theEnv,execStatus)->DummyFact)
           { return("<Dummy Fact>"); }
         
         thePtr = DOPToPointer(theDO);
         gensprintf(buffer,"<Fact-%lld>",((struct fact *) thePtr)->factIndex);
         thePtr = EnvAddSymbol(theEnv,execStatus,buffer);
         return(ValueToString(thePtr));
#endif
                        
      default:
         return("UNK");
     }
     
   length = strlen(prefix) + strlen(theString) + strlen(postfix) + 1;
   newString = (char *) genalloc(theEnv,execStatus,length);
   newString[0] = '\0';
   genstrcat(newString,prefix);
   genstrcat(newString,theString);
   genstrcat(newString,postfix);
   thePtr = EnvAddSymbol(theEnv,execStatus,newString);
   genfree(theEnv,execStatus,newString,length);
   return(ValueToString(thePtr));
  }
  
/************************************************************/
/* SalienceInformationError: Error message for errors which */
/*   occur during the evaluation of a salience value.       */
/************************************************************/
globle void SalienceInformationError(
  void *theEnv,
  EXEC_STATUS,
  char *constructType,
  char *constructName)
  {
   PrintErrorID(theEnv,execStatus,"PRNTUTIL",8,TRUE);
   EnvPrintRouter(theEnv,execStatus,WERROR,"This error occurred while evaluating the salience");
   if (constructName != NULL)
     {
      EnvPrintRouter(theEnv,execStatus,WERROR," for ");
      EnvPrintRouter(theEnv,execStatus,WERROR,constructType);
      EnvPrintRouter(theEnv,execStatus,WERROR," ");
      EnvPrintRouter(theEnv,execStatus,WERROR,constructName);
     }
   EnvPrintRouter(theEnv,execStatus,WERROR,".\n");
  }

/**********************************************************/
/* SalienceRangeError: Error message that is printed when */
/*   a salience value does not fall between the minimum   */
/*   and maximum salience values.                         */
/**********************************************************/
globle void SalienceRangeError(
  void *theEnv,
  EXEC_STATUS,
  int min,
  int max)
  {
   PrintErrorID(theEnv,execStatus,"PRNTUTIL",9,TRUE);
   EnvPrintRouter(theEnv,execStatus,WERROR,"Salience value out of range ");
   PrintLongInteger(theEnv,execStatus,WERROR,(long int) min);
   EnvPrintRouter(theEnv,execStatus,WERROR," to ");
   PrintLongInteger(theEnv,execStatus,WERROR,(long int) max);
   EnvPrintRouter(theEnv,execStatus,WERROR,".\n");
  }

/***************************************************************/
/* SalienceNonIntegerError: Error message that is printed when */
/*   a rule's salience does not evaluate to an integer.        */
/***************************************************************/
globle void SalienceNonIntegerError(
  void *theEnv,
  EXEC_STATUS)
  {
   PrintErrorID(theEnv,execStatus,"PRNTUTIL",10,TRUE);
   EnvPrintRouter(theEnv,execStatus,WERROR,"Salience value must be an integer value.\n");
  }

/***************************************************/
/* SlotExistError: Prints out an appropriate error */
/*   message when a slot cannot be found for a     */
/*   function. Input to the function is the slot   */
/*   name and the function name.                   */
/***************************************************/
globle void SlotExistError(
  void *theEnv,
  EXEC_STATUS,
  char *sname,
  char *func)
  {
   PrintErrorID(theEnv,execStatus,"INSFUN",3,FALSE);
   EnvPrintRouter(theEnv,execStatus,WERROR,"No such slot ");
   EnvPrintRouter(theEnv,execStatus,WERROR,sname);
   EnvPrintRouter(theEnv,execStatus,WERROR," in function ");
   EnvPrintRouter(theEnv,execStatus,WERROR,func);
   EnvPrintRouter(theEnv,execStatus,WERROR,".\n");
   SetEvaluationError(theEnv,execStatus,TRUE);
  }
