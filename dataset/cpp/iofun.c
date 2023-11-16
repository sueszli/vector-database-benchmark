   /*******************************************************/
   /*      "C" Language Integrated Production System      */
   /*                                                     */
   /*             CLIPS Version 6.30  03/02/07            */
   /*                                                     */
   /*                 I/O FUNCTIONS MODULE                */
   /*******************************************************/

/*************************************************************/
/* Purpose: Contains the code for several I/O functions      */
/*   including printout, read, open, close, remove, rename,  */
/*   format, and readline.                                   */
/*                                                           */
/* Principal Programmer(s):                                  */
/*      Brian L. Dantes                                      */
/*      Gary D. Riley                                        */
/*      Bebe Ly                                              */
/*                                                           */
/* Contributing Programmer(s):                               */
/*                                                           */
/* Revision History:                                         */
/*      6.24: Added the get-char, set-locale, and            */
/*            read-number functions.                         */
/*                                                           */
/*            Modified printing of floats in the format      */
/*            function to use the locale from the set-locale */
/*            function.                                      */
/*                                                           */
/*            Moved IllegalLogicalNameMessage function to    */
/*            argacces.c.                                    */
/*                                                           */
/*      6.30: Removed the undocumented use of t in the       */
/*            printout command to perform the same function  */
/*            as crlf.                                       */
/*                                                           */
/*            Added a+, w+, rb, ab, r+b, w+b, and a+b modes  */
/*            for the open function.                         */
/*************************************************************/

#define _IOFUN_SOURCE_

#include "setup.h"

#if IO_FUNCTIONS
#include <locale.h>
#include <stdlib.h>
#include <ctype.h>
#endif

#include <stdio.h>
#define _STDIO_INCLUDED_
#include <string.h>

#include "envrnmnt.h"
#include "router.h"
#include "strngrtr.h"
#include "filertr.h"
#include "argacces.h"
#include "extnfunc.h"
#include "scanner.h"
#include "constant.h"
#include "memalloc.h"
#include "commline.h"
#include "sysdep.h"
#include "utility.h"

#include "iofun.h"

/***************/
/* DEFINITIONS */
/***************/

#define FORMAT_MAX 512
#define FLAG_MAX    80

/********************/
/* ENVIRONMENT DATA */
/********************/

#define IO_FUNCTION_DATA 64

struct IOFunctionData
  { 
   void *locale;
   intBool useFullCRLF;
  };

#define IOFunctionData(theEnv,execStatus) ((struct IOFunctionData *) GetEnvironmentData(theEnv,execStatus,IO_FUNCTION_DATA))

/****************************************/
/* LOCAL INTERNAL FUNCTION DEFINITIONS  */
/****************************************/

#if IO_FUNCTIONS
   static void             ReadTokenFromStdin(void *,EXEC_STATUS,struct token *);
   static char            *ControlStringCheck(void *,EXEC_STATUS,int);
   static char             FindFormatFlag(char *,size_t *,char *,size_t);
   static char            *PrintFormatFlag(void *,EXEC_STATUS,char *,int,int);
   static char            *FillBuffer(void *,EXEC_STATUS,char *,size_t *,size_t *);
   static void             ReadNumber(void *,EXEC_STATUS,char *,struct token *,int);
#endif

/**************************************/
/* IOFunctionDefinitions: Initializes */
/*   the I/O functions.               */
/**************************************/
globle void IOFunctionDefinitions(
  void *theEnv,
  EXEC_STATUS)
  {
   AllocateEnvironmentData(theEnv,execStatus,IO_FUNCTION_DATA,sizeof(struct IOFunctionData),NULL);

#if IO_FUNCTIONS
   IOFunctionData(theEnv,execStatus)->useFullCRLF = FALSE;
   IOFunctionData(theEnv,execStatus)->locale = (SYMBOL_HN *) EnvAddSymbol(theEnv,execStatus,setlocale(LC_ALL,NULL));
   IncrementSymbolCount(IOFunctionData(theEnv,execStatus)->locale);
#endif

#if ! RUN_TIME
#if IO_FUNCTIONS
   EnvDefineFunction2(theEnv,execStatus,"printout",   'v', PTIEF PrintoutFunction, "PrintoutFunction", "1*");
   EnvDefineFunction2(theEnv,execStatus,"read",       'u', PTIEF ReadFunction,  "ReadFunction", "*1");
   EnvDefineFunction2(theEnv,execStatus,"open",       'b', OpenFunction,  "OpenFunction", "23*k");
   EnvDefineFunction2(theEnv,execStatus,"close",      'b', CloseFunction, "CloseFunction", "*1");
   EnvDefineFunction2(theEnv,execStatus,"get-char",   'i', GetCharFunction, "GetCharFunction", "*1");
   EnvDefineFunction2(theEnv,execStatus,"put-char",   'v', PTIEF PutCharFunction, "PutCharFunction", "12");
   EnvDefineFunction2(theEnv,execStatus,"remove",   'b', RemoveFunction,  "RemoveFunction", "11k");
   EnvDefineFunction2(theEnv,execStatus,"rename",   'b', RenameFunction, "RenameFunction", "22k");
   EnvDefineFunction2(theEnv,execStatus,"format",   's', PTIEF FormatFunction, "FormatFunction", "2**us");
   EnvDefineFunction2(theEnv,execStatus,"readline", 'k', PTIEF ReadlineFunction, "ReadlineFunction", "*1");
   EnvDefineFunction2(theEnv,execStatus,"set-locale", 'u', PTIEF SetLocaleFunction,  "SetLocaleFunction", "*1");
   EnvDefineFunction2(theEnv,execStatus,"read-number",       'u', PTIEF ReadNumberFunction,  "ReadNumberFunction", "*1");
#endif
#else
#if MAC_MCW || WIN_MCW || MAC_XCD
#pragma unused(theEnv,execStatus)
#endif
#endif
  }

#if IO_FUNCTIONS

/******************************************/
/* PrintoutFunction: H/L access routine   */
/*   for the printout function.           */
/******************************************/
globle void PrintoutFunction(
  void *theEnv,
  EXEC_STATUS)
  {
   char *dummyid;
   int i, argCount;
   DATA_OBJECT theArgument;

   /*=======================================================*/
   /* The printout function requires at least one argument. */
   /*=======================================================*/

   if ((argCount = EnvArgCountCheck(theEnv,execStatus,"printout",AT_LEAST,1)) == -1) return;

   /*=====================================================*/
   /* Get the logical name to which output is to be sent. */
   /*=====================================================*/

   dummyid = GetLogicalName(theEnv,execStatus,1,"stdout");
   if (dummyid == NULL)
     {
      IllegalLogicalNameMessage(theEnv,execStatus,"printout");
      SetHaltExecution(theEnv,execStatus,TRUE);
      SetEvaluationError(theEnv,execStatus,TRUE);
      return;
     }

   /*============================================================*/
   /* Determine if any router recognizes the output destination. */
   /*============================================================*/

   if (strcmp(dummyid,"nil") == 0)
     { return; }
   else if (QueryRouters(theEnv,execStatus,dummyid) == FALSE)
     {
      UnrecognizedRouterMessage(theEnv,execStatus,dummyid);
      return;
     }

   /*===============================================*/
   /* Print each of the arguments sent to printout. */
   /*===============================================*/

   for (i = 2; i <= argCount; i++)
     {
      EnvRtnUnknown(theEnv,execStatus,i,&theArgument);
      if (execStatus->HaltExecution) break;

      switch(GetType(theArgument))
        {
         case SYMBOL:
           if (strcmp(DOToString(theArgument),"crlf") == 0)
             {    
              if (IOFunctionData(theEnv,execStatus)->useFullCRLF)
                { EnvPrintRouter(theEnv,execStatus,dummyid,"\r\n"); }
              else
                { EnvPrintRouter(theEnv,execStatus,dummyid,"\n"); }
             }
           else if (strcmp(DOToString(theArgument),"tab") == 0)
             { EnvPrintRouter(theEnv,execStatus,dummyid,"\t"); }
           else if (strcmp(DOToString(theArgument),"vtab") == 0)
             { EnvPrintRouter(theEnv,execStatus,dummyid,"\v"); }
           else if (strcmp(DOToString(theArgument),"ff") == 0)
             { EnvPrintRouter(theEnv,execStatus,dummyid,"\f"); }
             /*
           else if (strcmp(DOToString(theArgument),"t") == 0)
             { 
              if (IOFunctionData(theEnv,execStatus)->useFullCRLF)
                { EnvPrintRouter(theEnv,execStatus,dummyid,"\r\n"); }
              else
                { EnvPrintRouter(theEnv,execStatus,dummyid,"\n"); }
             }
             */
           else
             { EnvPrintRouter(theEnv,execStatus,dummyid,DOToString(theArgument)); }
           break;

         case STRING:
           EnvPrintRouter(theEnv,execStatus,dummyid,DOToString(theArgument));
           break;

         default:
           PrintDataObject(theEnv,execStatus,dummyid,&theArgument);
           break;
        }
     }
  }

/*****************************************************/
/* SetFullCRLF: Set the flag which indicates whether */
/*   crlf is treated just as '\n' or '\r\n'.         */
/*****************************************************/
globle intBool SetFullCRLF(
  void *theEnv,
  EXEC_STATUS,
  intBool value)
  {
   intBool oldValue = IOFunctionData(theEnv,execStatus)->useFullCRLF;
   
   IOFunctionData(theEnv,execStatus)->useFullCRLF = value;
   
   return(oldValue);
  }

/*************************************************************/
/* ReadFunction: H/L access routine for the read function.   */
/*************************************************************/
globle void ReadFunction(
  void *theEnv,
  EXEC_STATUS,
  DATA_OBJECT_PTR returnValue)
  {
   struct token theToken;
   int numberOfArguments;
   char *logicalName = NULL;

   /*===============================================*/
   /* Check for an appropriate number of arguments. */
   /*===============================================*/

   if ((numberOfArguments = EnvArgCountCheck(theEnv,execStatus,"read",NO_MORE_THAN,1)) == -1)
     {
      returnValue->type = STRING;
      returnValue->value = (void *) EnvAddSymbol(theEnv,execStatus,"*** READ ERROR ***");
      return;
     }

   /*======================================================*/
   /* Determine the logical name from which input is read. */
   /*======================================================*/

   if (numberOfArguments == 0)
     { logicalName = "stdin"; }
   else if (numberOfArguments == 1)
     {
      logicalName = GetLogicalName(theEnv,execStatus,1,"stdin");
      if (logicalName == NULL)
        {
         IllegalLogicalNameMessage(theEnv,execStatus,"read");
         SetHaltExecution(theEnv,execStatus,TRUE);
         SetEvaluationError(theEnv,execStatus,TRUE);
         returnValue->type = STRING;
         returnValue->value = (void *) EnvAddSymbol(theEnv,execStatus,"*** READ ERROR ***");
         return;
        }
     }

   /*============================================*/
   /* Check to see that the logical name exists. */
   /*============================================*/

   if (QueryRouters(theEnv,execStatus,logicalName) == FALSE)
     {
      UnrecognizedRouterMessage(theEnv,execStatus,logicalName);
      SetHaltExecution(theEnv,execStatus,TRUE);
      SetEvaluationError(theEnv,execStatus,TRUE);
      returnValue->type = STRING;
      returnValue->value = (void *) EnvAddSymbol(theEnv,execStatus,"*** READ ERROR ***");
      return;
     }

   /*=======================================*/
   /* Collect input into string if the read */
   /* source is stdin, else just get token. */
   /*=======================================*/

   if (strcmp(logicalName,"stdin") == 0)
     { ReadTokenFromStdin(theEnv,execStatus,&theToken); }
   else
     { GetToken(theEnv,execStatus,logicalName,&theToken); }

   RouterData(theEnv,execStatus)->CommandBufferInputCount = 0;
   RouterData(theEnv,execStatus)->AwaitingInput = FALSE;

   /*====================================================*/
   /* Copy the token to the return value data structure. */
   /*====================================================*/

   returnValue->type = theToken.type;
   if ((theToken.type == FLOAT) || (theToken.type == STRING) ||
#if OBJECT_SYSTEM
       (theToken.type == INSTANCE_NAME) ||
#endif
       (theToken.type == SYMBOL) || (theToken.type == INTEGER))
     { returnValue->value = theToken.value; }
   else if (theToken.type == STOP)
     {
      returnValue->type = SYMBOL;
      returnValue->value = (void *) EnvAddSymbol(theEnv,execStatus,"EOF");
     }
   else if (theToken.type == UNKNOWN_VALUE)
     {
      returnValue->type = STRING;
      returnValue->value = (void *) EnvAddSymbol(theEnv,execStatus,"*** READ ERROR ***");
     }
   else
     {
      returnValue->type = STRING;
      returnValue->value = (void *) EnvAddSymbol(theEnv,execStatus,theToken.printForm);
     }

   return;
  }

/********************************************************/
/* ReadTokenFromStdin: Special routine used by the read */
/*   function to read a token from standard input.      */
/********************************************************/
static void ReadTokenFromStdin(
  void *theEnv,
  EXEC_STATUS,
  struct token *theToken)
  {
   char *inputString;
   size_t inputStringSize;
   int inchar;

   /*=============================================*/
   /* Continue processing until a token is found. */
   /*=============================================*/

   theToken->type = STOP;
   while (theToken->type == STOP)
     {
      /*===========================================*/
      /* Initialize the variables used for storing */
      /* the characters retrieved from stdin.      */
      /*===========================================*/

      inputString = NULL;
      RouterData(theEnv,execStatus)->CommandBufferInputCount = 0;
      RouterData(theEnv,execStatus)->AwaitingInput = TRUE;
      inputStringSize = 0;
      inchar = EnvGetcRouter(theEnv,execStatus,"stdin");

      /*========================================================*/
      /* Continue reading characters until a carriage return is */
      /* entered or the user halts execution (usually with      */
      /* control-c). Waiting for the carriage return prevents   */
      /* the input from being prematurely parsed (such as when  */
      /* a space is entered after a symbol has been typed).     */
      /*========================================================*/

      while ((inchar != '\n') && (inchar != '\r') && (inchar != EOF) &&
             (! GetHaltExecution(theEnv,execStatus)))
        {
         inputString = ExpandStringWithChar(theEnv,execStatus,inchar,inputString,&RouterData(theEnv,execStatus)->CommandBufferInputCount,
                                            &inputStringSize,inputStringSize + 80);
         inchar = EnvGetcRouter(theEnv,execStatus,"stdin");
        }

      /*==================================================*/
      /* Open a string input source using the characters  */
      /* retrieved from stdin and extract the first token */
      /* contained in the string.                         */
      /*==================================================*/

      OpenStringSource(theEnv,execStatus,"read",inputString,0);
      GetToken(theEnv,execStatus,"read",theToken);
      CloseStringSource(theEnv,execStatus,"read");
      if (inputStringSize > 0) rm(theEnv,execStatus,inputString,inputStringSize);

      /*===========================================*/
      /* Pressing control-c (or comparable action) */
      /* aborts the read function.                 */
      /*===========================================*/

      if (GetHaltExecution(theEnv,execStatus))
        {
         theToken->type = STRING;
         theToken->value = (void *) EnvAddSymbol(theEnv,execStatus,"*** READ ERROR ***");
        }

      /*====================================================*/
      /* Return the EOF symbol if the end of file for stdin */
      /* has been encountered. This typically won't occur,  */
      /* but is possible (for example by pressing control-d */
      /* in the UNIX operating system).                     */
      /*====================================================*/

      if ((theToken->type == STOP) && (inchar == EOF))
        {
         theToken->type = SYMBOL;
         theToken->value = (void *) EnvAddSymbol(theEnv,execStatus,"EOF");
        }
     }
  }

/*************************************************************/
/* OpenFunction: H/L access routine for the open function.   */
/*************************************************************/
globle int OpenFunction(
  void *theEnv,
  EXEC_STATUS)
  {
   int numberOfArguments;
   char *fileName, *logicalName, *accessMode = NULL;
   DATA_OBJECT theArgument;

   /*========================================*/
   /* Check for a valid number of arguments. */
   /*========================================*/

   if ((numberOfArguments = EnvArgRangeCheck(theEnv,execStatus,"open",2,3)) == -1) return(0);

   /*====================*/
   /* Get the file name. */
   /*====================*/

   if ((fileName = GetFileName(theEnv,execStatus,"open",1)) == NULL) return(0);

   /*=======================================*/
   /* Get the logical name to be associated */
   /* with the opened file.                 */
   /*=======================================*/

   logicalName = GetLogicalName(theEnv,execStatus,2,NULL);
   if (logicalName == NULL)
     {
      SetHaltExecution(theEnv,execStatus,TRUE);
      SetEvaluationError(theEnv,execStatus,TRUE);
      IllegalLogicalNameMessage(theEnv,execStatus,"open");
      return(0);
     }

   /*==================================*/
   /* Check to see if the logical name */
   /* is already in use.               */
   /*==================================*/

   if (FindFile(theEnv,execStatus,logicalName))
     {
      SetHaltExecution(theEnv,execStatus,TRUE);
      SetEvaluationError(theEnv,execStatus,TRUE);
      PrintErrorID(theEnv,execStatus,"IOFUN",2,FALSE);
      EnvPrintRouter(theEnv,execStatus,WERROR,"Logical name ");
      EnvPrintRouter(theEnv,execStatus,WERROR,logicalName);
      EnvPrintRouter(theEnv,execStatus,WERROR," already in use.\n");
      return(0);
     }

   /*===========================*/
   /* Get the file access mode. */
   /*===========================*/

   if (numberOfArguments == 2)
     { accessMode = "r"; }
   else if (numberOfArguments == 3)
     {
      if (EnvArgTypeCheck(theEnv,execStatus,"open",3,STRING,&theArgument) == FALSE) return(0);
      accessMode = DOToString(theArgument);
     }

   /*=====================================*/
   /* Check for a valid file access mode. */
   /*=====================================*/

   if ((strcmp(accessMode,"r") != 0) &&
       (strcmp(accessMode,"w") != 0) &&
       (strcmp(accessMode,"a") != 0) &&
       (strcmp(accessMode,"r+") != 0) &&
       (strcmp(accessMode,"w+") != 0) &&
       (strcmp(accessMode,"a+") != 0) &&
       (strcmp(accessMode,"rb") != 0) &&
       (strcmp(accessMode,"wb") != 0) &&
       (strcmp(accessMode,"ab") != 0) &&
       (strcmp(accessMode,"r+b") != 0) &&
       (strcmp(accessMode,"w+b") != 0) &&
       (strcmp(accessMode,"a+b") != 0))
     {
      SetHaltExecution(theEnv,execStatus,TRUE);
      SetEvaluationError(theEnv,execStatus,TRUE);
      ExpectedTypeError1(theEnv,execStatus,"open",3,"string with value \"r\", \"w\", \"a\", \"r+\", \"w+\", \"rb\", \"wb\", \"ab\", \"r+b\", or \"w+b\"");
      return(0);
     }

   /*================================================*/
   /* Open the named file and associate it with the  */
   /* specified logical name. Return TRUE if the     */
   /* file was opened successfully, otherwise FALSE. */
   /*================================================*/

   return(OpenAFile(theEnv,execStatus,fileName,accessMode,logicalName));
  }

/***************************************************************/
/* CloseFunction: H/L access routine for the close function.   */
/***************************************************************/
globle int CloseFunction(
  void *theEnv,
  EXEC_STATUS)
  {
   int numberOfArguments;
   char *logicalName;

   /*======================================*/
   /* Check for valid number of arguments. */
   /*======================================*/

   if ((numberOfArguments = EnvArgCountCheck(theEnv,execStatus,"close",NO_MORE_THAN,1)) == -1) return(0);

   /*=====================================================*/
   /* If no arguments are specified, then close all files */
   /* opened with the open command. Return TRUE if all    */
   /* files were closed successfully, otherwise FALSE.    */
   /*=====================================================*/

   if (numberOfArguments == 0) return(CloseAllFiles(theEnv,execStatus));

   /*================================*/
   /* Get the logical name argument. */
   /*================================*/

   logicalName = GetLogicalName(theEnv,execStatus,1,NULL);
   if (logicalName == NULL)
     {
      IllegalLogicalNameMessage(theEnv,execStatus,"close");
      SetHaltExecution(theEnv,execStatus,TRUE);
      SetEvaluationError(theEnv,execStatus,TRUE);
      return(0);
     }

   /*========================================================*/
   /* Close the file associated with the specified logical   */
   /* name. Return TRUE if the file was closed successfully, */
   /* otherwise false.                                       */
   /*========================================================*/

   return(CloseFile(theEnv,execStatus,logicalName));
  }

/***************************************/
/* GetCharFunction: H/L access routine */
/*   for the get-char function.        */
/***************************************/
globle int GetCharFunction(
  void *theEnv,
  EXEC_STATUS)
  {
   int numberOfArguments;
   char *logicalName;

   if ((numberOfArguments = EnvArgCountCheck(theEnv,execStatus,"get-char",NO_MORE_THAN,1)) == -1)
     { return(-1); }

   if (numberOfArguments == 0 )
     { logicalName = "stdin"; }
   else
     {
      logicalName = GetLogicalName(theEnv,execStatus,1,"stdin");
      if (logicalName == NULL)
        {
         IllegalLogicalNameMessage(theEnv,execStatus,"get-char");
         SetHaltExecution(theEnv,execStatus,TRUE);
         SetEvaluationError(theEnv,execStatus,TRUE);
         return(-1);
        }
     }

   if (QueryRouters(theEnv,execStatus,logicalName) == FALSE)
     {
      UnrecognizedRouterMessage(theEnv,execStatus,logicalName);
      SetHaltExecution(theEnv,execStatus,TRUE);
      SetEvaluationError(theEnv,execStatus,TRUE);
      return(-1);
     }

   return(EnvGetcRouter(theEnv,execStatus,logicalName));
  }

/***************************************/
/* PutCharFunction: H/L access routine */
/*   for the put-char function.        */
/***************************************/
globle void PutCharFunction(
  void *theEnv,
  EXEC_STATUS)
  {
   int numberOfArguments;
   char *logicalName;
   DATA_OBJECT theValue;
   long long theChar;
   FILE *theFile;

   if ((numberOfArguments = EnvArgRangeCheck(theEnv,execStatus,"put-char",1,2)) == -1)
     { return; }
     
   /*=======================*/
   /* Get the logical name. */
   /*=======================*/
   
   if (numberOfArguments == 1)
     { logicalName = "stdout"; }
   else
     {
      logicalName = GetLogicalName(theEnv,execStatus,1,"stdout");
      if (logicalName == NULL)
        {
         IllegalLogicalNameMessage(theEnv,execStatus,"put-char");
         SetHaltExecution(theEnv,execStatus,TRUE);
         SetEvaluationError(theEnv,execStatus,TRUE);
         return;
        }
     }

   if (QueryRouters(theEnv,execStatus,logicalName) == FALSE)
     {
      UnrecognizedRouterMessage(theEnv,execStatus,logicalName);
      SetHaltExecution(theEnv,execStatus,TRUE);
      SetEvaluationError(theEnv,execStatus,TRUE);
      return;
     }

   /*===========================*/
   /* Get the character to put. */
   /*===========================*/
   
   if (numberOfArguments == 1)
     { if (EnvArgTypeCheck(theEnv,execStatus,"put-char",1,INTEGER,&theValue) == FALSE) return; }
   else
     { if (EnvArgTypeCheck(theEnv,execStatus,"put-char",2,INTEGER,&theValue) == FALSE) return; }
     
   theChar = DOToLong(theValue);
   
   /*===================================================*/
   /* If the "fast load" option is being used, then the */
   /* logical name is actually a pointer to a file and  */
   /* we can bypass the router and directly output the  */
   /* value.                                            */
   /*===================================================*/
      
   theFile = FindFptr(theEnv,execStatus,logicalName);
   if (theFile != NULL)
     { putc((int) theChar,theFile); }
  }

/****************************************/
/* RemoveFunction: H/L access routine   */
/*   for the remove function.           */
/****************************************/
globle int RemoveFunction(
  void *theEnv,
  EXEC_STATUS)
  {
   char *theFileName;

   /*======================================*/
   /* Check for valid number of arguments. */
   /*======================================*/

   if (EnvArgCountCheck(theEnv,execStatus,"remove",EXACTLY,1) == -1) return(FALSE);

   /*====================*/
   /* Get the file name. */
   /*====================*/

   if ((theFileName = GetFileName(theEnv,execStatus,"remove",1)) == NULL) return(FALSE);

   /*==============================================*/
   /* Remove the file. Return TRUE if the file was */
   /* sucessfully removed, otherwise FALSE.        */
   /*==============================================*/

   return(genremove(theFileName));
  }

/****************************************/
/* RenameFunction: H/L access routine   */
/*   for the rename function.           */
/****************************************/
globle int RenameFunction(
  void *theEnv,
  EXEC_STATUS)
  {
   char *oldFileName, *newFileName;

   /*========================================*/
   /* Check for a valid number of arguments. */
   /*========================================*/

   if (EnvArgCountCheck(theEnv,execStatus,"rename",EXACTLY,2) == -1) return(FALSE);

   /*===========================*/
   /* Check for the file names. */
   /*===========================*/

   if ((oldFileName = GetFileName(theEnv,execStatus,"rename",1)) == NULL) return(FALSE);
   if ((newFileName = GetFileName(theEnv,execStatus,"rename",2)) == NULL) return(FALSE);

   /*==============================================*/
   /* Rename the file. Return TRUE if the file was */
   /* sucessfully renamed, otherwise FALSE.        */
   /*==============================================*/

   return(genrename(oldFileName,newFileName));
  }

/****************************************/
/* FormatFunction: H/L access routine   */
/*   for the format function.           */
/****************************************/
globle void *FormatFunction(
  void *theEnv,
  EXEC_STATUS)
  {
   int argCount;
   size_t start_pos;
   char *formatString, *logicalName;
   char formatFlagType;
   int  f_cur_arg = 3;
   size_t form_pos = 0;
   char percentBuffer[FLAG_MAX];
   char *fstr = NULL;
   size_t fmaxm = 0;
   size_t fpos = 0;
   void *hptr;
   char *theString;

   /*======================================*/
   /* Set default return value for errors. */
   /*======================================*/

   hptr = EnvAddSymbol(theEnv,execStatus,"");

   /*=========================================*/
   /* Format requires at least two arguments: */
   /* a logical name and a format string.     */
   /*=========================================*/

   if ((argCount = EnvArgCountCheck(theEnv,execStatus,"format",AT_LEAST,2)) == -1)
     { return(hptr); }

   /*========================================*/
   /* First argument must be a logical name. */
   /*========================================*/

   if ((logicalName = GetLogicalName(theEnv,execStatus,1,"stdout")) == NULL)
     {
      IllegalLogicalNameMessage(theEnv,execStatus,"format");
      SetHaltExecution(theEnv,execStatus,TRUE);
      SetEvaluationError(theEnv,execStatus,TRUE);
      return(hptr);
     }

   if (strcmp(logicalName,"nil") == 0)
     { /* do nothing */ }
   else if (QueryRouters(theEnv,execStatus,logicalName) == FALSE)
     {
      UnrecognizedRouterMessage(theEnv,execStatus,logicalName);
      return(hptr);
     }

   /*=====================================================*/
   /* Second argument must be a string.  The appropriate  */
   /* number of arguments specified by the string must be */
   /* present in the argument list.                       */
   /*=====================================================*/

   if ((formatString = ControlStringCheck(theEnv,execStatus,argCount)) == NULL)
     { return (hptr); }

   /*========================================*/
   /* Search the format string, printing the */
   /* format flags as they are encountered.  */
   /*========================================*/

   while (formatString[form_pos] != '\0')
     {
      if (formatString[form_pos] != '%')
        {
         start_pos = form_pos;
         while ((formatString[form_pos] != '%') &&
                (formatString[form_pos] != '\0'))
           { form_pos++; }
         fstr = AppendNToString(theEnv,execStatus,&formatString[start_pos],fstr,form_pos-start_pos,&fpos,&fmaxm);
        }
      else
        {
		 form_pos++;
         formatFlagType = FindFormatFlag(formatString,&form_pos,percentBuffer,FLAG_MAX);
         if (formatFlagType != ' ')
           {
            if ((theString = PrintFormatFlag(theEnv,execStatus,percentBuffer,f_cur_arg,formatFlagType)) == NULL)
              {
               if (fstr != NULL) rm(theEnv,execStatus,fstr,fmaxm);
               return (hptr);
              }
            fstr = AppendToString(theEnv,execStatus,theString,fstr,&fpos,&fmaxm);
            if (fstr == NULL) return(hptr);
            f_cur_arg++;
           }
         else
           {
            fstr = AppendToString(theEnv,execStatus,percentBuffer,fstr,&fpos,&fmaxm);
            if (fstr == NULL) return(hptr);
           }
        }
     }

   if (fstr != NULL)
     {
      hptr = EnvAddSymbol(theEnv,execStatus,fstr);
      if (strcmp(logicalName,"nil") != 0) EnvPrintRouter(theEnv,execStatus,logicalName,fstr);
      rm(theEnv,execStatus,fstr,fmaxm);
     }
   else
     { hptr = EnvAddSymbol(theEnv,execStatus,""); }

   return(hptr);
  }

/*********************************************************************/
/* ControlStringCheck:  Checks the 2nd parameter which is the format */
/*   control string to see if there are enough matching arguments.   */
/*********************************************************************/
static char *ControlStringCheck(
  void *theEnv,
  EXEC_STATUS,
  int argCount)
  {
   DATA_OBJECT t_ptr;
   char *str_array;
   char print_buff[FLAG_MAX];
   size_t i;
   int per_count;
   char formatFlag;

   if (EnvArgTypeCheck(theEnv,execStatus,"format",2,STRING,&t_ptr) == FALSE) return(NULL);

   per_count = 0;
   str_array = ValueToString(t_ptr.value);
   for (i= 0 ; str_array[i] != '\0' ; )
     {
      if (str_array[i] == '%')
        {
         i++;
         formatFlag = FindFormatFlag(str_array,&i,print_buff,FLAG_MAX);
         if (formatFlag == '-')
           { 
            PrintErrorID(theEnv,execStatus,"IOFUN",3,FALSE);
            EnvPrintRouter(theEnv,execStatus,WERROR,"Invalid format flag \"");
            EnvPrintRouter(theEnv,execStatus,WERROR,print_buff);
            EnvPrintRouter(theEnv,execStatus,WERROR,"\" specified in format function.\n");
            SetEvaluationError(theEnv,execStatus,TRUE);
            return (NULL);
           }
         else if (formatFlag != ' ')
           { per_count++; }
        }
      else
        { i++; }
     }

   if (per_count != (argCount - 2))
     {
      ExpectedCountError(theEnv,execStatus,"format",EXACTLY,per_count+2);
      SetEvaluationError(theEnv,execStatus,TRUE);
      return (NULL);
     }

   return(str_array);
  }

/***********************************************/
/* FindFormatFlag:  This function searches for */
/*   a format flag in the format string.       */
/***********************************************/
static char FindFormatFlag(
  char *formatString,
  size_t *a,
  char *formatBuffer,
  size_t bufferMax)
  {
   char inchar, formatFlagType;
   size_t copy_pos = 0;

   /*====================================================*/
   /* Set return values to the default value. A blank    */
   /* character indicates that no format flag was found  */
   /* which requires a parameter.                        */
   /*====================================================*/

   formatFlagType = ' ';

   /*=====================================================*/
   /* The format flags for carriage returns, line feeds,  */
   /* horizontal and vertical tabs, and the percent sign, */
   /* do not require a parameter.                         */
   /*=====================================================*/

   if (formatString[*a] == 'n')
     {
      gensprintf(formatBuffer,"\n");
      (*a)++;
      return(formatFlagType);
     }
   else if (formatString[*a] == 'r')
     {
      gensprintf(formatBuffer,"\r");
      (*a)++;
      return(formatFlagType);
     }
   else if (formatString[*a] == 't')
     {
      gensprintf(formatBuffer,"\t");
      (*a)++;
      return(formatFlagType);
     }
   else if (formatString[*a] == 'v')
     {
      gensprintf(formatBuffer,"\v");
      (*a)++;
      return(formatFlagType);
     }
   else if (formatString[*a] == '%')
     {
      gensprintf(formatBuffer,"%%");
      (*a)++;
      return(formatFlagType);
     }

   /*======================================================*/
   /* Identify the format flag which requires a parameter. */
   /*======================================================*/

   formatBuffer[copy_pos++] = '%';
   formatBuffer[copy_pos] = '\0';
   while ((formatString[*a] != '%') &&
          (formatString[*a] != '\0') &&
          (copy_pos < (bufferMax - 5)))
     {
      inchar = formatString[*a];
      (*a)++;

      if ( (inchar == 'd') ||
           (inchar == 'o') ||
           (inchar == 'x') ||
           (inchar == 'u'))
        {
         formatFlagType = inchar;
         formatBuffer[copy_pos++] = 'l';
         formatBuffer[copy_pos++] = 'l';
         formatBuffer[copy_pos++] = inchar;
         formatBuffer[copy_pos] = '\0';
         return(formatFlagType);
        }
      else if ( (inchar == 'c') ||
                (inchar == 's') ||
                (inchar == 'e') ||
                (inchar == 'f') ||
                (inchar == 'g') )
        {
         formatBuffer[copy_pos++] = inchar;
         formatBuffer[copy_pos] = '\0';
         formatFlagType = inchar;
         return(formatFlagType);
        }
      
      /*=======================================================*/
      /* If the type hasn't been read, then this should be the */
      /* -M.N part of the format specification (where M and N  */
      /* are integers).                                        */
      /*=======================================================*/
      
      if ( (! isdigit(inchar)) &&
           (inchar != '.') &&
           (inchar != '-') )
        { 
         formatBuffer[copy_pos++] = inchar;
         formatBuffer[copy_pos] = '\0';
         return('-'); 
        }

      formatBuffer[copy_pos++] = inchar;
      formatBuffer[copy_pos] = '\0';
     }

   return(formatFlagType);
  }

/**********************************************************************/
/* PrintFormatFlag:  Prints out part of the total format string along */
/*   with the argument for that part of the format string.            */
/**********************************************************************/
static char *PrintFormatFlag(
  void *theEnv,
  EXEC_STATUS,
  char *formatString,
  int whichArg,
  int formatType)
  {
   DATA_OBJECT theResult;
   char *theString, *printBuffer;
   size_t theLength;
   void *oldLocale;
      
   /*=================*/
   /* String argument */
   /*=================*/

   switch (formatType)
     {
      case 's':
        if (EnvArgTypeCheck(theEnv,execStatus,"format",whichArg,SYMBOL_OR_STRING,&theResult) == FALSE) return(NULL);
        theLength = strlen(formatString) + strlen(ValueToString(theResult.value)) + 200;
        printBuffer = (char *) gm2(theEnv,execStatus,(sizeof(char) * theLength));
        gensprintf(printBuffer,formatString,ValueToString(theResult.value));
        break;

      case 'c':
        EnvRtnUnknown(theEnv,execStatus,whichArg,&theResult);
        if ((GetType(theResult) == STRING) ||
            (GetType(theResult) == SYMBOL))
          {
           theLength = strlen(formatString) + 200;
           printBuffer = (char *) gm2(theEnv,execStatus,(sizeof(char) * theLength));
           gensprintf(printBuffer,formatString,(ValueToString(theResult.value))[0]);
          }
        else if (GetType(theResult) == INTEGER)
          {
           theLength = strlen(formatString) + 200;
           printBuffer = (char *) gm2(theEnv,execStatus,(sizeof(char) * theLength));
           gensprintf(printBuffer,formatString,(char) DOToLong(theResult));
          }
        else
          {
           ExpectedTypeError1(theEnv,execStatus,"format",whichArg,"symbol, string, or integer");
           return(NULL);
          }
        break;

      case 'd':
      case 'x':
      case 'o':
      case 'u':
        if (EnvArgTypeCheck(theEnv,execStatus,"format",whichArg,INTEGER_OR_FLOAT,&theResult) == FALSE) return(NULL);
        theLength = strlen(formatString) + 200;
        printBuffer = (char *) gm2(theEnv,execStatus,(sizeof(char) * theLength));
        
        oldLocale = EnvAddSymbol(theEnv,execStatus,setlocale(LC_NUMERIC,NULL));
        setlocale(LC_NUMERIC,ValueToString(IOFunctionData(theEnv,execStatus)->locale));

        if (GetType(theResult) == FLOAT)
          { gensprintf(printBuffer,formatString,(long long) ValueToDouble(theResult.value)); }
        else
          { gensprintf(printBuffer,formatString,(long long) ValueToLong(theResult.value)); }
          
        setlocale(LC_NUMERIC,ValueToString(oldLocale));
        break;

      case 'f':
      case 'g':
      case 'e':
        if (EnvArgTypeCheck(theEnv,execStatus,"format",whichArg,INTEGER_OR_FLOAT,&theResult) == FALSE) return(NULL);
        theLength = strlen(formatString) + 200;
        printBuffer = (char *) gm2(theEnv,execStatus,(sizeof(char) * theLength));

        oldLocale = EnvAddSymbol(theEnv,execStatus,setlocale(LC_NUMERIC,NULL));
        
        setlocale(LC_NUMERIC,ValueToString(IOFunctionData(theEnv,execStatus)->locale));

        if (GetType(theResult) == FLOAT)
          { gensprintf(printBuffer,formatString,ValueToDouble(theResult.value)); }
        else
          { gensprintf(printBuffer,formatString,(double) ValueToLong(theResult.value)); }
        
        setlocale(LC_NUMERIC,ValueToString(oldLocale));
        
        break;

      default:
         EnvPrintRouter(theEnv,execStatus,WERROR," Error in format, the conversion character");
         EnvPrintRouter(theEnv,execStatus,WERROR," for formatted output is not valid\n");
         return(FALSE);
     }

   theString = ValueToString(EnvAddSymbol(theEnv,execStatus,printBuffer));
   rm(theEnv,execStatus,printBuffer,sizeof(char) * theLength);
   return(theString);
  }

/******************************************/
/* ReadlineFunction: H/L access routine   */
/*   for the readline function.           */
/******************************************/
globle void ReadlineFunction(
  void *theEnv,
  EXEC_STATUS,
  DATA_OBJECT_PTR returnValue)
  {
   char *buffer;
   size_t line_max = 0;
   int numberOfArguments;
   char *logicalName;

   returnValue->type = STRING;

   if ((numberOfArguments = EnvArgCountCheck(theEnv,execStatus,"readline",NO_MORE_THAN,1)) == -1)
     {
      returnValue->value = (void *) EnvAddSymbol(theEnv,execStatus,"*** READ ERROR ***");
      return;
     }

   if (numberOfArguments == 0 )
     { logicalName = "stdin"; }
   else
     {
      logicalName = GetLogicalName(theEnv,execStatus,1,"stdin");
      if (logicalName == NULL)
        {
         IllegalLogicalNameMessage(theEnv,execStatus,"readline");
         SetHaltExecution(theEnv,execStatus,TRUE);
         SetEvaluationError(theEnv,execStatus,TRUE);
         returnValue->value = (void *) EnvAddSymbol(theEnv,execStatus,"*** READ ERROR ***");
         return;
        }
     }

   if (QueryRouters(theEnv,execStatus,logicalName) == FALSE)
     {
      UnrecognizedRouterMessage(theEnv,execStatus,logicalName);
      SetHaltExecution(theEnv,execStatus,TRUE);
      SetEvaluationError(theEnv,execStatus,TRUE);
      returnValue->value = (void *) EnvAddSymbol(theEnv,execStatus,"*** READ ERROR ***");
      return;
     }

   RouterData(theEnv,execStatus)->CommandBufferInputCount = 0;
   RouterData(theEnv,execStatus)->AwaitingInput = TRUE;
   buffer = FillBuffer(theEnv,execStatus,logicalName,&RouterData(theEnv,execStatus)->CommandBufferInputCount,&line_max);
   RouterData(theEnv,execStatus)->CommandBufferInputCount = 0;
   RouterData(theEnv,execStatus)->AwaitingInput = FALSE;

   if (GetHaltExecution(theEnv,execStatus))
     {
      returnValue->value = (void *) EnvAddSymbol(theEnv,execStatus,"*** READ ERROR ***");
      if (buffer != NULL) rm(theEnv,execStatus,buffer,(int) sizeof (char) * line_max);
      return;
     }

   if (buffer == NULL)
     {
      returnValue->value = (void *) EnvAddSymbol(theEnv,execStatus,"EOF");
      returnValue->type = SYMBOL;
      return;
     }

   returnValue->value = (void *) EnvAddSymbol(theEnv,execStatus,buffer);
   rm(theEnv,execStatus,buffer,(int) sizeof (char) * line_max);
   return;
  }

/*************************************************************/
/* FillBuffer: Read characters from a specified logical name */
/*   and places them into a buffer until a carriage return   */
/*   or end-of-file character is read.                       */
/*************************************************************/
static char *FillBuffer(
  void *theEnv,
  EXEC_STATUS,
  char *logicalName,
  size_t *currentPosition,
  size_t *maximumSize)
  {
   int c;
   char *buf = NULL;

   /*================================*/
   /* Read until end of line or eof. */
   /*================================*/

   c = EnvGetcRouter(theEnv,execStatus,logicalName);

   if (c == EOF)
     { return(NULL); }

   /*==================================*/
   /* Grab characters until cr or eof. */
   /*==================================*/

   while ((c != '\n') && (c != '\r') && (c != EOF) &&
          (! GetHaltExecution(theEnv,execStatus)))
     {
      buf = ExpandStringWithChar(theEnv,execStatus,c,buf,currentPosition,maximumSize,*maximumSize+80);
      c = EnvGetcRouter(theEnv,execStatus,logicalName);
     }

   /*==================*/
   /* Add closing EOS. */
   /*==================*/

   buf = ExpandStringWithChar(theEnv,execStatus,EOS,buf,currentPosition,maximumSize,*maximumSize+80);
   return (buf);
  }
  
/*****************************************/
/* SetLocaleFunction: H/L access routine */
/*   for the set-locale function.        */
/*****************************************/
globle void SetLocaleFunction(
  void *theEnv,
  EXEC_STATUS,
  DATA_OBJECT_PTR returnValue)
  {
   DATA_OBJECT theResult;
   int numArgs;
   
   /*======================================*/
   /* Check for valid number of arguments. */
   /*======================================*/
   
   if ((numArgs = EnvArgCountCheck(theEnv,execStatus,"set-locale",NO_MORE_THAN,1)) == -1)
     {
      returnValue->type = SYMBOL;
      returnValue->value = EnvFalseSymbol(theEnv,execStatus);
      return;
     }
     
   /*=================================*/
   /* If there are no arguments, just */
   /* return the current locale.      */
   /*=================================*/
   
   if (numArgs == 0)
     {
      returnValue->type = STRING;
      returnValue->value = IOFunctionData(theEnv,execStatus)->locale;
      return;
     }

   /*=================*/
   /* Get the locale. */
   /*=================*/
   
   if (EnvArgTypeCheck(theEnv,execStatus,"set-locale",1,STRING,&theResult) == FALSE)
     {
      returnValue->type = SYMBOL;
      returnValue->value = EnvFalseSymbol(theEnv,execStatus);
      return;
     }
     
   /*=====================================*/
   /* Return the old value of the locale. */
   /*=====================================*/
   
   returnValue->type = STRING;
   returnValue->value = IOFunctionData(theEnv,execStatus)->locale;
   
   /*======================================================*/
   /* Change the value of the locale to the one specified. */
   /*======================================================*/
   
   DecrementSymbolCount(theEnv,execStatus,(struct symbolHashNode *) IOFunctionData(theEnv,execStatus)->locale);
   IOFunctionData(theEnv,execStatus)->locale = DOToPointer(theResult);
   IncrementSymbolCount(IOFunctionData(theEnv,execStatus)->locale);
  }

/******************************************/
/* ReadNumberFunction: H/L access routine */
/*   for the read-number function.        */
/******************************************/
globle void ReadNumberFunction(
  void *theEnv,
  EXEC_STATUS,
  DATA_OBJECT_PTR returnValue)
  {
   struct token theToken;
   int numberOfArguments;
   char *logicalName = NULL;

   /*===============================================*/
   /* Check for an appropriate number of arguments. */
   /*===============================================*/

   if ((numberOfArguments = EnvArgCountCheck(theEnv,execStatus,"read",NO_MORE_THAN,1)) == -1)
     {
      returnValue->type = STRING;
      returnValue->value = (void *) EnvAddSymbol(theEnv,execStatus,"*** READ ERROR ***");
      return;
     }

   /*======================================================*/
   /* Determine the logical name from which input is read. */
   /*======================================================*/

   if (numberOfArguments == 0)
     { logicalName = "stdin"; }
   else if (numberOfArguments == 1)
     {
      logicalName = GetLogicalName(theEnv,execStatus,1,"stdin");
      if (logicalName == NULL)
        {
         IllegalLogicalNameMessage(theEnv,execStatus,"read");
         SetHaltExecution(theEnv,execStatus,TRUE);
         SetEvaluationError(theEnv,execStatus,TRUE);
         returnValue->type = STRING;
         returnValue->value = (void *) EnvAddSymbol(theEnv,execStatus,"*** READ ERROR ***");
         return;
        }
     }

   /*============================================*/
   /* Check to see that the logical name exists. */
   /*============================================*/

   if (QueryRouters(theEnv,execStatus,logicalName) == FALSE)
     {
      UnrecognizedRouterMessage(theEnv,execStatus,logicalName);
      SetHaltExecution(theEnv,execStatus,TRUE);
      SetEvaluationError(theEnv,execStatus,TRUE);
      returnValue->type = STRING;
      returnValue->value = (void *) EnvAddSymbol(theEnv,execStatus,"*** READ ERROR ***");
      return;
     }

   /*=======================================*/
   /* Collect input into string if the read */
   /* source is stdin, else just get token. */
   /*=======================================*/

   if (strcmp(logicalName,"stdin") == 0)
     { ReadNumber(theEnv,execStatus,logicalName,&theToken,TRUE); }
   else
     { ReadNumber(theEnv,execStatus,logicalName,&theToken,FALSE); }

   RouterData(theEnv,execStatus)->CommandBufferInputCount = 0;
   RouterData(theEnv,execStatus)->AwaitingInput = FALSE;

   /*====================================================*/
   /* Copy the token to the return value data structure. */
   /*====================================================*/

   returnValue->type = theToken.type;
   if ((theToken.type == FLOAT) || (theToken.type == STRING) ||
#if OBJECT_SYSTEM
       (theToken.type == INSTANCE_NAME) ||
#endif
       (theToken.type == SYMBOL) || (theToken.type == INTEGER))
     { returnValue->value = theToken.value; }
   else if (theToken.type == STOP)
     {
      returnValue->type = SYMBOL;
      returnValue->value = (void *) EnvAddSymbol(theEnv,execStatus,"EOF");
     }
   else if (theToken.type == UNKNOWN_VALUE)
     {
      returnValue->type = STRING;
      returnValue->value = (void *) EnvAddSymbol(theEnv,execStatus,"*** READ ERROR ***");
     }
   else
     {
      returnValue->type = STRING;
      returnValue->value = (void *) EnvAddSymbol(theEnv,execStatus,theToken.printForm);
     }

   return;
  }
  
/********************************************/
/* ReadNumber: Special routine used by the  */
/*   read-number function to read a number. */
/********************************************/
static void ReadNumber(
  void *theEnv,
  EXEC_STATUS,
  char *logicalName,
  struct token *theToken,
  int isStdin)
  {
   char *inputString;
   char *charPtr = NULL;
   size_t inputStringSize;
   int inchar;
   long long theLong;
   double theDouble;
   void *oldLocale;

   theToken->type = STOP;

   /*===========================================*/
   /* Initialize the variables used for storing */
   /* the characters retrieved from stdin.      */
   /*===========================================*/

   inputString = NULL;
   RouterData(theEnv,execStatus)->CommandBufferInputCount = 0;
   RouterData(theEnv,execStatus)->AwaitingInput = TRUE;
   inputStringSize = 0;
   inchar = EnvGetcRouter(theEnv,execStatus,logicalName);
            
   /*====================================*/
   /* Skip whitespace before any number. */
   /*====================================*/
      
   while (isspace(inchar) && (inchar != EOF) && 
          (! GetHaltExecution(theEnv,execStatus)))
     { inchar = EnvGetcRouter(theEnv,execStatus,logicalName); }

   /*=============================================================*/
   /* Continue reading characters until whitespace is found again */
   /* (for anything other than stdin) or a CR/LF (for stdin).     */
   /*=============================================================*/

   while ((((! isStdin) && (! isspace(inchar))) || 
          (isStdin && (inchar != '\n') && (inchar != '\r'))) &&
          (inchar != EOF) &&
          (! GetHaltExecution(theEnv,execStatus)))
     {
      inputString = ExpandStringWithChar(theEnv,execStatus,inchar,inputString,&RouterData(theEnv,execStatus)->CommandBufferInputCount,
                                         &inputStringSize,inputStringSize + 80);
      inchar = EnvGetcRouter(theEnv,execStatus,logicalName);
     }

   /*===========================================*/
   /* Pressing control-c (or comparable action) */
   /* aborts the read-number function.          */
   /*===========================================*/

   if (GetHaltExecution(theEnv,execStatus))
     {
      theToken->type = STRING;
      theToken->value = (void *) EnvAddSymbol(theEnv,execStatus,"*** READ ERROR ***");
      if (inputStringSize > 0) rm(theEnv,execStatus,inputString,inputStringSize);
      return;
     }

   /*====================================================*/
   /* Return the EOF symbol if the end of file for stdin */
   /* has been encountered. This typically won't occur,  */
   /* but is possible (for example by pressing control-d */
   /* in the UNIX operating system).                     */
   /*====================================================*/

   if (inchar == EOF)
     {
      theToken->type = SYMBOL;
      theToken->value = (void *) EnvAddSymbol(theEnv,execStatus,"EOF");
      if (inputStringSize > 0) rm(theEnv,execStatus,inputString,inputStringSize);
      return;
     }

   /*==================================================*/
   /* Open a string input source using the characters  */
   /* retrieved from stdin and extract the first token */
   /* contained in the string.                         */
   /*==================================================*/
   
   /*=======================================*/
   /* Change the locale so that numbers are */
   /* converted using the localized format. */
   /*=======================================*/
   
   oldLocale = EnvAddSymbol(theEnv,execStatus,setlocale(LC_NUMERIC,NULL));
   setlocale(LC_NUMERIC,ValueToString(IOFunctionData(theEnv,execStatus)->locale));

   /*========================================*/
   /* Try to parse the number as a long. The */
   /* terminating character must either be   */
   /* white space or the string terminator.  */
   /*========================================*/

#if WIN_MVC
   theLong = _strtoi64(inputString,&charPtr,10);
#else
   theLong = strtoll(inputString,&charPtr,10);
#endif

   if ((charPtr != inputString) && 
       (isspace(*charPtr) || (*charPtr == '\0')))
     {
      theToken->type = INTEGER;
      theToken->value = (void *) EnvAddLong(theEnv,execStatus,theLong);
      if (inputStringSize > 0) rm(theEnv,execStatus,inputString,inputStringSize);
      setlocale(LC_NUMERIC,ValueToString(oldLocale));
      return;
     }
     
   /*==========================================*/
   /* Try to parse the number as a double. The */
   /* terminating character must either be     */
   /* white space or the string terminator.    */
   /*==========================================*/

   theDouble = strtod(inputString,&charPtr);  
   if ((charPtr != inputString) && 
       (isspace(*charPtr) || (*charPtr == '\0')))
     {
      theToken->type = FLOAT;
      theToken->value = (void *) EnvAddDouble(theEnv,execStatus,theDouble);
      if (inputStringSize > 0) rm(theEnv,execStatus,inputString,inputStringSize);
      setlocale(LC_NUMERIC,ValueToString(oldLocale));
      return;
     }

   /*============================================*/
   /* Restore the "C" locale so that any parsing */
   /* of numbers uses the C format.              */
   /*============================================*/
   
   setlocale(LC_NUMERIC,ValueToString(oldLocale));

   /*=========================================*/
   /* Return "*** READ ERROR ***" to indicate */
   /* a number was not successfully parsed.   */
   /*=========================================*/
         
   theToken->type = STRING;
   theToken->value = (void *) EnvAddSymbol(theEnv,execStatus,"*** READ ERROR ***");
  }

#endif

