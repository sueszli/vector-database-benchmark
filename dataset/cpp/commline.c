   /*******************************************************/
   /*      "C" Language Integrated Production System      */
   /*                                                     */
   /*             CLIPS Version 6.24  06/05/06            */
   /*                                                     */
   /*                COMMAND LINE MODULE                  */
   /*******************************************************/

/*************************************************************/
/* Purpose: Provides a set of routines for processing        */
/*   commands entered at the top level prompt.               */
/*                                                           */
/* Principal Programmer(s):                                  */
/*      Gary D. Riley                                        */
/*                                                           */
/* Contributing Programmer(s):                               */
/*      Brian L. Dantes                                      */
/*                                                           */
/* Revision History:                                         */
/*                                                           */
/*      6.24: Renamed BOOLEAN macro type to intBool.         */
/*                                                           */
/*            Refactored several functions and added         */
/*            additional functions for use by an interface   */
/*            layered on top of CLIPS.                       */
/*                                                           */
/*      6.30: Local variables set with the bind function     */
/*            persist until a reset/clear command is issued. */
/*                                                           */
/*************************************************************/

#define _COMMLINE_SOURCE_

#include <stdio.h>
#define _STDIO_INCLUDED_
#include <string.h>
#include <ctype.h>

#include "setup.h"
#include "constant.h"

#include "argacces.h"
#include "constrct.h"
#include "cstrcpsr.h"
#include "envrnmnt.h"
#include "exprnpsr.h"
#include "filecom.h"
#include "memalloc.h"
#include "prcdrfun.h"
#include "prcdrpsr.h"
#include "router.h"
#include "scanner.h"
#include "strngrtr.h"
#include "symbol.h"
#include "sysdep.h"
#include "utility.h"

#include "commline.h"

/***************************************/
/* LOCAL INTERNAL FUNCTION DEFINITIONS */
/***************************************/

#if ! RUN_TIME
   static int                     DoString(char *,int,int *);
   static int                     DoComment(char *,int);
   static int                     DoWhiteSpace(char *,int);
   static int                     DefaultGetNextEvent(void *,EXEC_STATUS);
#endif
   static void                    DeallocateCommandLineData(void *,EXEC_STATUS);

/****************************************************/
/* InitializeCommandLineData: Allocates environment */
/*    data for command line functionality.          */
/****************************************************/
globle void InitializeCommandLineData(
  void *theEnv,
  EXEC_STATUS)
  {
   AllocateEnvironmentData(theEnv,execStatus,COMMANDLINE_DATA,sizeof(struct commandLineData),DeallocateCommandLineData);

#if ! RUN_TIME   
   CommandLineData(theEnv,execStatus)->BannerString = BANNER_STRING;
   CommandLineData(theEnv,execStatus)->EventFunction = DefaultGetNextEvent;
#endif
  }
  
/*******************************************************/
/* DeallocateCommandLineData: Deallocates environment */
/*    data for the command line functionality.        */
/******************************************************/
static void DeallocateCommandLineData(
  void *theEnv,
  EXEC_STATUS)
  {
#if ! RUN_TIME
   if (CommandLineData(theEnv,execStatus)->CommandString != NULL) 
     { rm(theEnv,execStatus,CommandLineData(theEnv,execStatus)->CommandString,CommandLineData(theEnv,execStatus)->MaximumCharacters); }
     
   if (CommandLineData(theEnv,execStatus)->CurrentCommand != NULL) 
     { ReturnExpression(theEnv,execStatus,CommandLineData(theEnv,execStatus)->CurrentCommand); }
#else
#if MAC_MCW || WIN_MCW || MAC_XCD
#pragma unused(theEnv,execStatus)
#endif
#endif
  }

#if ! RUN_TIME

/***************************************************/
/* ExpandCommandString: Appends a character to the */
/*   command string. Returns TRUE if the command   */
/*   string was successfully expanded, otherwise   */
/*   FALSE. Expanding the string also includes     */
/*   adding a backspace character which reduces    */
/*   string's length.                              */
/***************************************************/
globle int ExpandCommandString(
  void *theEnv,
  EXEC_STATUS,
  int inchar)
  {
   size_t k;

   k = RouterData(theEnv,execStatus)->CommandBufferInputCount;
   CommandLineData(theEnv,execStatus)->CommandString = ExpandStringWithChar(theEnv,execStatus,inchar,CommandLineData(theEnv,execStatus)->CommandString,&RouterData(theEnv,execStatus)->CommandBufferInputCount,
                                        &CommandLineData(theEnv,execStatus)->MaximumCharacters,CommandLineData(theEnv,execStatus)->MaximumCharacters+80);
   return((RouterData(theEnv,execStatus)->CommandBufferInputCount != k) ? TRUE : FALSE);
  }

/******************************************************************/
/* FlushCommandString: Empties the contents of the CommandString. */
/******************************************************************/
globle void FlushCommandString(
  void *theEnv,
  EXEC_STATUS)
  {
   if (CommandLineData(theEnv,execStatus)->CommandString != NULL) rm(theEnv,execStatus,CommandLineData(theEnv,execStatus)->CommandString,CommandLineData(theEnv,execStatus)->MaximumCharacters);
   CommandLineData(theEnv,execStatus)->CommandString = NULL;
   CommandLineData(theEnv,execStatus)->MaximumCharacters = 0;
   RouterData(theEnv,execStatus)->CommandBufferInputCount = 0;
   RouterData(theEnv,execStatus)->AwaitingInput = TRUE;
  }

/*********************************************************************************/
/* SetCommandString: Sets the contents of the CommandString to a specific value. */
/*********************************************************************************/
globle void SetCommandString(
  void *theEnv,
  EXEC_STATUS,
  char *str)
  {
   size_t length;

   FlushCommandString(theEnv,execStatus);
   length = strlen(str);
   CommandLineData(theEnv,execStatus)->CommandString = (char *)
                   genrealloc(theEnv,execStatus,CommandLineData(theEnv,execStatus)->CommandString,(unsigned) CommandLineData(theEnv,execStatus)->MaximumCharacters,
                              (unsigned) CommandLineData(theEnv,execStatus)->MaximumCharacters + length + 1);

   genstrcpy(CommandLineData(theEnv,execStatus)->CommandString,str);
   CommandLineData(theEnv,execStatus)->MaximumCharacters += (length + 1);
   RouterData(theEnv,execStatus)->CommandBufferInputCount += (int) length;
  }

/*************************************************************/
/* SetNCommandString: Sets the contents of the CommandString */
/*   to a specific value up to N characters.                 */
/*************************************************************/
globle void SetNCommandString(
  void *theEnv,
  EXEC_STATUS,
  char *str,
  unsigned length)
  {
   FlushCommandString(theEnv,execStatus);
   CommandLineData(theEnv,execStatus)->CommandString = (char *)
                   genrealloc(theEnv,execStatus,CommandLineData(theEnv,execStatus)->CommandString,(unsigned) CommandLineData(theEnv,execStatus)->MaximumCharacters,
                              (unsigned) CommandLineData(theEnv,execStatus)->MaximumCharacters + length + 1);

   genstrncpy(CommandLineData(theEnv,execStatus)->CommandString,str,length);
   CommandLineData(theEnv,execStatus)->CommandString[CommandLineData(theEnv,execStatus)->MaximumCharacters + length] = 0;
   CommandLineData(theEnv,execStatus)->MaximumCharacters += (length + 1);
   RouterData(theEnv,execStatus)->CommandBufferInputCount += (int) length;
  }

/******************************************************************************/
/* AppendCommandString: Appends a value to the contents of the CommandString. */
/******************************************************************************/
globle void AppendCommandString(
  void *theEnv,
  EXEC_STATUS,
  char *str)
  {
   CommandLineData(theEnv,execStatus)->CommandString = AppendToString(theEnv,execStatus,str,CommandLineData(theEnv,execStatus)->CommandString,&RouterData(theEnv,execStatus)->CommandBufferInputCount,&CommandLineData(theEnv,execStatus)->MaximumCharacters);
  }

/******************************************************************************/
/* InsertCommandString: Inserts a value in the contents of the CommandString. */
/******************************************************************************/
globle void InsertCommandString(
  void *theEnv,
  EXEC_STATUS,
  char *str,
  unsigned int position)
  {
   CommandLineData(theEnv,execStatus)->CommandString = 
      InsertInString(theEnv,execStatus,str,position,CommandLineData(theEnv,execStatus)->CommandString,
                     &RouterData(theEnv,execStatus)->CommandBufferInputCount,&CommandLineData(theEnv,execStatus)->MaximumCharacters);
  }
  
/************************************************************/
/* AppendNCommandString: Appends a value up to N characters */
/*   to the contents of the CommandString.                  */
/************************************************************/
globle void AppendNCommandString(
  void *theEnv,
  EXEC_STATUS,
  char *str,
  unsigned length)
  {
   CommandLineData(theEnv,execStatus)->CommandString = AppendNToString(theEnv,execStatus,str,CommandLineData(theEnv,execStatus)->CommandString,length,&RouterData(theEnv,execStatus)->CommandBufferInputCount,&CommandLineData(theEnv,execStatus)->MaximumCharacters);
  }

/*****************************************************************************/
/* GetCommandString: Returns a pointer to the contents of the CommandString. */
/*****************************************************************************/
globle char *GetCommandString(
  void *theEnv,
  EXEC_STATUS)
  {
   return(CommandLineData(theEnv,execStatus)->CommandString);
  }

/**************************************************************************/
/* CompleteCommand: Determines whether a string forms a complete command. */
/*   A complete command is either a constant, a variable, or a function   */
/*   call which is followed (at some point) by a carriage return. Once a  */
/*   complete command is found (not including the parenthesis),           */
/*   extraneous parenthesis and other tokens are ignored. If a complete   */
/*   command exists, then 1 is returned. 0 is returned if the command was */
/*   not complete and without errors. -1 is returned if the command       */
/*   contains an error.                                                   */
/**************************************************************************/
globle int CompleteCommand(
  char *mstring)
  {
   int i;
   char inchar;
   int depth = 0;
   int moreThanZero = 0;
   int complete;
   int error = 0;

   if (mstring == NULL) return(0);

   /*===================================================*/
   /* Loop through each character of the command string */
   /* to determine if there is a complete command.      */
   /*===================================================*/

   i = 0;
   while ((inchar = mstring[i++]) != EOS)
     {
      switch(inchar)
        {
         /*======================================================*/
         /* If a carriage return or line feed is found, there is */
         /* at least one completed token in the command buffer,  */
         /* and parentheses are balanced, then a complete        */
         /* command has been found. Otherwise, remove all white  */
         /* space beginning with the current character.          */
         /*======================================================*/

         case '\n' :
         case '\r' :
           if (error) return(-1);
           if (moreThanZero && (depth == 0)) return(1);
           i = DoWhiteSpace(mstring,i);
           break;

         /*=====================*/
         /* Remove white space. */
         /*=====================*/

         case ' ' :
         case '\f' :
         case '\t' :
           i = DoWhiteSpace(mstring,i);
           break;

         /*======================================================*/
         /* If the opening quotation of a string is encountered, */
         /* determine if the closing quotation of the string is  */
         /* in the command buffer. Until the closing quotation   */
         /* is found, a complete command can not be made.        */
         /*======================================================*/

         case '"' :
           i = DoString(mstring,i,&complete);
           if ((depth == 0) && complete) moreThanZero = TRUE;
           break;

         /*====================*/
         /* Process a comment. */
         /*====================*/

         case ';' :
           i = DoComment(mstring,i);
           if (moreThanZero && (depth == 0) && (mstring[i] != EOS))
             {
              if (error) return(-1);
              else return(1);
             }
           else if (mstring[i] != EOS) i++;
           break;

         /*====================================================*/
         /* A left parenthesis increases the nesting depth of  */
         /* the current command by 1. Don't bother to increase */
         /* the depth if the first token encountered was not   */
         /* a parenthesis (e.g. for the command string         */
         /* "red (+ 3 4", the symbol red already forms a       */
         /* complete command, so the next carriage return will */
         /* cause evaluation of red--the closing parenthesis   */
         /* for "(+ 3 4" does not have to be found).           */
         /*====================================================*/

         case '(' :
           if ((depth > 0) || (moreThanZero == FALSE))
             {
              depth++;
              moreThanZero = TRUE;
             }
           break;

         /*====================================================*/
         /* A right parenthesis decreases the nesting depth of */
         /* the current command by 1. If the parenthesis is    */
         /* the first token of the command, then an error is   */
         /* generated.                                         */
         /*====================================================*/

         case ')' :
           if (depth > 0) depth--;
           else if (moreThanZero == FALSE) error = TRUE;
           break;

         /*=====================================================*/
         /* If the command begins with any other character and  */
         /* an opening parenthesis hasn't yet been found, then  */
         /* skip all characters on the same line. If a carriage */
         /* return or line feed is found, then a complete       */
         /* command exists.                                     */
         /*=====================================================*/

         default:
           if (depth == 0)
             {
              if (isprint(inchar) || IsUTF8MultiByteStart(inchar))
                {
                 while ((inchar = mstring[i++]) != EOS)
                   {
                    if ((inchar == '\n') || (inchar == '\r'))
                      {
                       if (error) return(-1);
                       else return(1);
                      }
                   }
                 return(0);
                }
             }
           break;
        }
     }

   /*====================================================*/
   /* Return 0 because a complete command was not found. */
   /*====================================================*/

   return(0);
  }

/***********************************************************/
/* DoString: Skips over a string contained within a string */
/*   until the closing quotation mark is encountered.      */
/***********************************************************/
static int DoString(
  char *str,
  int pos,
  int *complete)
  {
   int inchar;

   /*=================================================*/
   /* Process the string character by character until */
   /* the closing quotation mark is found.            */
   /*=================================================*/

   inchar = str[pos];
   while (inchar  != '"')
     {
      /*=====================================================*/
      /* If a \ is found, then the next character is ignored */
      /* even if it is a closing quotation mark.             */
      /*=====================================================*/

      if (inchar == '\\')
        {
         pos++;
         inchar = str[pos];
        }

      /*===================================================*/
      /* If the end of input is reached before the closing */
      /* quotation mark is found, the return the last      */
      /* position that was reached and indicate that a     */
      /* complete string was not found.                    */
      /*===================================================*/

      if (inchar == EOS)
        {
         *complete = FALSE;
         return(pos);
        }

      /*================================*/
      /* Move on to the next character. */
      /*================================*/

      pos++;
      inchar = str[pos];
     }

   /*======================================================*/
   /* Indicate that a complete string was found and return */
   /* the position of the closing quotation mark.          */
   /*======================================================*/

   pos++;
   *complete = TRUE;
   return(pos);
  }

/*************************************************************/
/* DoComment: Skips over a comment contained within a string */
/*   until a line feed or carriage return is encountered.    */
/*************************************************************/
static int DoComment(
  char *str,
  int pos)
  {
   int inchar;

   inchar = str[pos];
   while ((inchar != '\n') && (inchar != '\r'))
     {
      if (inchar == EOS)
        { return(pos); }

      pos++;
      inchar = str[pos];
     }

   return(pos);
  }

/**************************************************************/
/* DoWhiteSpace: Skips over white space consisting of spaces, */
/*   tabs, and form feeds that is contained within a string.  */
/**************************************************************/
static int DoWhiteSpace(
  char *str,
  int pos)
  {
   int inchar;

   inchar = str[pos];
   while ((inchar == ' ') || (inchar == '\f') || (inchar == '\t'))
     {
      pos++;
      inchar = str[pos];
     }

   return(pos);
  }

/********************************************************************/
/* CommandLoop: Endless loop which waits for user commands and then */
/*   executes them. The command loop will bypass the EventFunction  */
/*   if there is an active batch file.                              */
/********************************************************************/
globle void CommandLoop(
  void *theEnv,
  EXEC_STATUS)
  {
   int inchar;

   EnvPrintRouter(theEnv,execStatus,WPROMPT,CommandLineData(theEnv,execStatus)->BannerString);
   SetHaltExecution(theEnv,execStatus,FALSE);
   SetEvaluationError(theEnv,execStatus,FALSE);
   PeriodicCleanup(theEnv,execStatus,TRUE,FALSE);
   PrintPrompt(theEnv,execStatus);
   RouterData(theEnv,execStatus)->CommandBufferInputCount = 0;
   RouterData(theEnv,execStatus)->AwaitingInput = TRUE;

   while (TRUE)
     {
      /*===================================================*/
      /* If a batch file is active, grab the command input */
      /* directly from the batch file, otherwise call the  */
      /* event function.                                   */
      /*===================================================*/

      if (BatchActive(theEnv,execStatus) == TRUE)
        {
         inchar = LLGetcBatch(theEnv,execStatus,"stdin",TRUE);
         if (inchar == EOF)
           { (*CommandLineData(theEnv,execStatus)->EventFunction)(theEnv,execStatus); }
         else
           { ExpandCommandString(theEnv,execStatus,(char) inchar); }
        }
      else
        { (*CommandLineData(theEnv,execStatus)->EventFunction)(theEnv,execStatus); }

      /*=================================================*/
      /* If execution was halted, then remove everything */
      /* from the command buffer.                        */
      /*=================================================*/

      if (GetHaltExecution(theEnv,execStatus) == TRUE)
        {
         SetHaltExecution(theEnv,execStatus,FALSE);
         SetEvaluationError(theEnv,execStatus,FALSE);
         FlushCommandString(theEnv,execStatus);
#if ! WINDOW_INTERFACE
         fflush(stdin);
#endif
         EnvPrintRouter(theEnv,execStatus,WPROMPT,"\n");
         PrintPrompt(theEnv,execStatus);
        }

      /*=========================================*/
      /* If a complete command is in the command */
      /* buffer, then execute it.                */
      /*=========================================*/

      ExecuteIfCommandComplete(theEnv,execStatus);
     }
  }
  
/***********************************************************/
/* CommandLoopBatch: Loop which waits for commands from a  */
/*   batch file and then executes them. Returns when there */
/*   are no longer any active batch files.                 */
/***********************************************************/
globle void CommandLoopBatch(
  void *theEnv,
	EXEC_STATUS)
  {
   SetHaltExecution(theEnv,execStatus,FALSE);
   SetEvaluationError(theEnv,execStatus,FALSE);
   PeriodicCleanup(theEnv,execStatus,TRUE,FALSE);
   PrintPrompt(theEnv,execStatus);
   RouterData(theEnv,execStatus)->CommandBufferInputCount = 0;
   RouterData(theEnv,execStatus)->AwaitingInput = TRUE;

   CommandLoopBatchDriver(theEnv,execStatus);
  }

/************************************************************/
/* CommandLoopOnceThenBatch: Loop which waits for commands  */
/*   from a batch file and then executes them. Returns when */
/*   there are no longer any active batch files.            */
/************************************************************/
globle void CommandLoopOnceThenBatch(
  void *theEnv,
	EXEC_STATUS)
  {
   if (! ExecuteIfCommandComplete(theEnv,execStatus)) return;

   CommandLoopBatchDriver(theEnv,execStatus);
  }
  
/*********************************************************/
/* CommandLoopBatchDriver: Loop which waits for commands */
/*   from a batch file and then executes them. Returns   */
/*   when there are no longer any active batch files.    */
/*********************************************************/
globle void CommandLoopBatchDriver(
  void *theEnv,
	EXEC_STATUS)
  {
   int inchar;

   while (TRUE)
     {
      if (GetHaltCommandLoopBatch(theEnv,execStatus) == TRUE)
        { 
         CloseAllBatchSources(theEnv,execStatus);
         SetHaltCommandLoopBatch(theEnv,execStatus,FALSE);
        }
        
      /*===================================================*/
      /* If a batch file is active, grab the command input */
      /* directly from the batch file, otherwise call the  */
      /* event function.                                   */
      /*===================================================*/

      if (BatchActive(theEnv,execStatus) == TRUE)
        {
         inchar = LLGetcBatch(theEnv,execStatus,"stdin",TRUE);
         if (inchar == EOF)
           { return; }
         else
           { ExpandCommandString(theEnv,execStatus,(char) inchar); }
        }
      else
        { return; }

      /*=================================================*/
      /* If execution was halted, then remove everything */
      /* from the command buffer.                        */
      /*=================================================*/

      if (GetHaltExecution(theEnv,execStatus) == TRUE)
        {
         SetHaltExecution(theEnv,execStatus,FALSE);
         SetEvaluationError(theEnv,execStatus,FALSE);
         FlushCommandString(theEnv,execStatus);
#if ! WINDOW_INTERFACE
         fflush(stdin);
#endif
         EnvPrintRouter(theEnv,execStatus,WPROMPT,"\n");
         PrintPrompt(theEnv,execStatus);
        }

      /*=========================================*/
      /* If a complete command is in the command */
      /* buffer, then execute it.                */
      /*=========================================*/

      ExecuteIfCommandComplete(theEnv,execStatus);
     }
  }

/**********************************************************/
/* ExecuteIfCommandComplete: Checks to determine if there */
/*   is a completed command and if so executes it.        */
/**********************************************************/
globle intBool ExecuteIfCommandComplete(
  void *theEnv,
	EXEC_STATUS)
  {
   if ((CompleteCommand(CommandLineData(theEnv,execStatus)->CommandString) == 0) || 
       (RouterData(theEnv,execStatus)->CommandBufferInputCount == 0) ||
       (RouterData(theEnv,execStatus)->AwaitingInput == FALSE))
     { return FALSE; }
     
   if (CommandLineData(theEnv,execStatus)->BeforeCommandExecutionFunction != NULL)
     { 
      if (! (*CommandLineData(theEnv,execStatus)->BeforeCommandExecutionFunction)(theEnv,execStatus))
        { return FALSE; }
     }
       
   FlushPPBuffer(theEnv,execStatus);
   SetPPBufferStatus(theEnv,execStatus,OFF);
   RouterData(theEnv,execStatus)->CommandBufferInputCount = 0;
   RouterData(theEnv,execStatus)->AwaitingInput = FALSE;
   RouteCommand(theEnv,execStatus,CommandLineData(theEnv,execStatus)->CommandString,TRUE);
   FlushPPBuffer(theEnv,execStatus);
   SetHaltExecution(theEnv,execStatus,FALSE);
   SetEvaluationError(theEnv,execStatus,FALSE);
   FlushCommandString(theEnv,execStatus);
   PeriodicCleanup(theEnv,execStatus,TRUE,FALSE);
   PrintPrompt(theEnv,execStatus);
         
   return TRUE;
  }

/*******************************************/
/* CommandCompleteAndNotEmpty: */
/*******************************************/
globle intBool CommandCompleteAndNotEmpty(
  void *theEnv,
  EXEC_STATUS)
  {
   if ((CompleteCommand(CommandLineData(theEnv,execStatus)->CommandString) == 0) || 
       (RouterData(theEnv,execStatus)->CommandBufferInputCount == 0) ||
       (RouterData(theEnv,execStatus)->AwaitingInput == FALSE))
     { return FALSE; }
     
   return TRUE;
  }
       
/*******************************************/
/* PrintPrompt: Prints the command prompt. */
/*******************************************/
globle void PrintPrompt(
   void *theEnv,
  EXEC_STATUS)
   {
    EnvPrintRouter(theEnv,execStatus,WPROMPT,COMMAND_PROMPT);

    if (CommandLineData(theEnv,execStatus)->AfterPromptFunction != NULL)
      { (*CommandLineData(theEnv,execStatus)->AfterPromptFunction)(theEnv,execStatus); }
   }

/*****************************************/
/* PrintBanner: Prints the CLIPS banner. */
/*****************************************/
globle void PrintBanner(
   void *theEnv,
  EXEC_STATUS)
   {
    EnvPrintRouter(theEnv,execStatus,WPROMPT,CommandLineData(theEnv,execStatus)->BannerString);
   }

/************************************************/
/* SetAfterPromptFunction: Replaces the current */
/*   value of AfterPromptFunction.              */
/************************************************/
globle void SetAfterPromptFunction(
  void *theEnv,
  EXEC_STATUS,
  int (*funptr)(void *,EXEC_STATUS))
  {
   CommandLineData(theEnv,execStatus)->AfterPromptFunction = funptr;
  }

/***********************************************************/
/* SetBeforeCommandExecutionFunction: Replaces the current */
/*   value of BeforeCommandExecutionFunction.              */
/***********************************************************/
globle void SetBeforeCommandExecutionFunction(
  void *theEnv,
  EXEC_STATUS,
  int (*funptr)(void *,EXEC_STATUS))
  {
   CommandLineData(theEnv,execStatus)->BeforeCommandExecutionFunction = funptr;
  }
  
/************************************************/
/* RouteCommand: Processes a completed command. */
/************************************************/
globle intBool RouteCommand(
  void *theEnv,
  EXEC_STATUS,
  char *command,
  int printResult)
  {
   DATA_OBJECT result;
   struct expr *top;
   char *commandName;
   struct token theToken;

   if (command == NULL)
     { return(0); }

   /*========================================*/
   /* Open a string input source and get the */
   /* first token from that source.          */
   /*========================================*/

   OpenStringSource(theEnv,execStatus,"command",command,0);

   GetToken(theEnv,execStatus,"command",&theToken);

   /*=====================*/
   /* Evaluate constants. */
   /*=====================*/

   if ((theToken.type == SYMBOL) || (theToken.type == STRING) ||
       (theToken.type == FLOAT) || (theToken.type == INTEGER) ||
       (theToken.type == INSTANCE_NAME))
     {
      CloseStringSource(theEnv,execStatus,"command");
      if (printResult)
        {
         PrintAtom(theEnv,execStatus,"stdout",theToken.type,theToken.value);
         EnvPrintRouter(theEnv,execStatus,"stdout","\n");
        }
      return(1);
     }

   /*=====================*/
   /* Evaluate variables. */
   /*=====================*/

   if ((theToken.type == GBL_VARIABLE) ||
       (theToken.type == SF_VARIABLE) ||
       (theToken.type == MF_VARIABLE))
     {
      CloseStringSource(theEnv,execStatus,"command");
      top = GenConstant(theEnv,execStatus,theToken.type,theToken.value);
      EvaluateExpression(theEnv,execStatus,top,&result);
      rtn_struct(theEnv,execStatus,expr,top);
      if (printResult)
        {
         PrintDataObject(theEnv,execStatus,"stdout",&result);
         EnvPrintRouter(theEnv,execStatus,"stdout","\n");
        }
      return(1);
     }

   /*========================================================*/
   /* If the next token isn't the beginning left parenthesis */
   /* of a command or construct, then whatever was entered   */
   /* cannot be evaluated at the command prompt.             */
   /*========================================================*/

   if (theToken.type != LPAREN)
     {
      PrintErrorID(theEnv,execStatus,"COMMLINE",1,FALSE);
      EnvPrintRouter(theEnv,execStatus,WERROR,"Expected a '(', constant, or variable\n");
      CloseStringSource(theEnv,execStatus,"command");
      return(0);
     }

   /*===========================================================*/
   /* The next token must be a function name or construct type. */
   /*===========================================================*/

   GetToken(theEnv,execStatus,"command",&theToken);
   if (theToken.type != SYMBOL)
     {
      PrintErrorID(theEnv,execStatus,"COMMLINE",2,FALSE);
      EnvPrintRouter(theEnv,execStatus,WERROR,"Expected a command.\n");
      CloseStringSource(theEnv,execStatus,"command");
      return(0);
     }

   commandName = ValueToString(theToken.value);

   /*======================*/
   /* Evaluate constructs. */
   /*======================*/

#if (! RUN_TIME) && (! BLOAD_ONLY)
   {
    int errorFlag;

    errorFlag = ParseConstruct(theEnv,execStatus,commandName,"command");
    if (errorFlag != -1)
      {
       CloseStringSource(theEnv,execStatus,"command");
       if (errorFlag == 1)
         {
          EnvPrintRouter(theEnv,execStatus,WERROR,"\nERROR:\n");
          PrintInChunks(theEnv,execStatus,WERROR,GetPPBuffer(theEnv,execStatus));
          EnvPrintRouter(theEnv,execStatus,WERROR,"\n");
         }
       DestroyPPBuffer(theEnv,execStatus);
       return(errorFlag);
      }
   }
#endif

   /*========================*/
   /* Parse a function call. */
   /*========================*/

   CommandLineData(theEnv,execStatus)->ParsingTopLevelCommand = TRUE;
   top = Function2Parse(theEnv,execStatus,"command",commandName);
   CommandLineData(theEnv,execStatus)->ParsingTopLevelCommand = FALSE;
   ClearParsedBindNames(theEnv,execStatus);

   /*================================*/
   /* Close the string input source. */
   /*================================*/

   CloseStringSource(theEnv,execStatus,"command");

   /*=========================*/
   /* Evaluate function call. */
   /*=========================*/

   if (top == NULL) return(0);
   
   ExpressionInstall(theEnv,execStatus,top);
   
   CommandLineData(theEnv,execStatus)->EvaluatingTopLevelCommand = TRUE;
   CommandLineData(theEnv,execStatus)->CurrentCommand = top;
   EvaluateExpression(theEnv,execStatus,top,&result);
   CommandLineData(theEnv,execStatus)->CurrentCommand = NULL;
   CommandLineData(theEnv,execStatus)->EvaluatingTopLevelCommand = FALSE;
   
   ExpressionDeinstall(theEnv,execStatus,top);
   ReturnExpression(theEnv,execStatus,top);
   
   if ((result.type != RVOID) && printResult)
     {
      PrintDataObject(theEnv,execStatus,"stdout",&result);
      EnvPrintRouter(theEnv,execStatus,"stdout","\n");
     }

   return(1);
  }

/*****************************************************************/
/* DefaultGetNextEvent: Default event-handling function. Handles */
/*   only keyboard events by first calling GetcRouter to get a   */
/*   character and then calling ExpandCommandString to add the   */
/*   character to the CommandString.                             */
/*****************************************************************/
static int DefaultGetNextEvent(
  void *theEnv,
  EXEC_STATUS)
  {
   int inchar;

   inchar = EnvGetcRouter(theEnv,execStatus,"stdin");

   if (inchar == EOF) inchar = '\n';

   ExpandCommandString(theEnv,execStatus,(char) inchar);
   
   return 0;
  }

/*************************************/
/* SetEventFunction: Replaces the    */
/*   current value of EventFunction. */
/*************************************/
globle int (*SetEventFunction(
	void *theEnv,
  EXEC_STATUS,
	int (*theFunction)(void *,EXEC_STATUS)))(void *,EXEC_STATUS)
  {
   int (*tmp_ptr)(void *,EXEC_STATUS);

   tmp_ptr = CommandLineData(theEnv,execStatus)->EventFunction;
   CommandLineData(theEnv,execStatus)->EventFunction = theFunction;
   return(tmp_ptr);
  }

/****************************************/
/* TopLevelCommand: Indicates whether a */
/*   top-level command is being parsed. */
/****************************************/
globle intBool TopLevelCommand(
  void *theEnv,
  EXEC_STATUS)
  {
   return(CommandLineData(theEnv,execStatus)->ParsingTopLevelCommand);
  }

/***********************************************************/
/* GetCommandCompletionString: Returns the last token in a */
/*   string if it is a valid token for command completion. */
/***********************************************************/
globle char *GetCommandCompletionString(
  void *theEnv,
  EXEC_STATUS,
  char *theString,
  size_t maxPosition)
  {
   struct token lastToken;
   struct token theToken;
   char lastChar;
   char *rs;
   size_t length;

   /*=========================*/
   /* Get the command string. */
   /*=========================*/

   if (theString == NULL) return("");

   /*=========================================================================*/
   /* If the last character in the command string is a space, character       */
   /* return, or quotation mark, then the command completion can be anything. */
   /*=========================================================================*/

   lastChar = theString[maxPosition - 1];
   if ((lastChar == ' ') || (lastChar == '"') ||
       (lastChar == '\t') || (lastChar == '\f') ||
       (lastChar == '\n') || (lastChar == '\r'))
     { return(""); }

   /*============================================*/
   /* Find the last token in the command string. */
   /*============================================*/

   OpenTextSource(theEnv,execStatus,"CommandCompletion",theString,0,maxPosition);
   ScannerData(theEnv,execStatus)->IgnoreCompletionErrors = TRUE;
   GetToken(theEnv,execStatus,"CommandCompletion",&theToken);
   CopyToken(&lastToken,&theToken);
   while (theToken.type != STOP)
     {
      CopyToken(&lastToken,&theToken);
      GetToken(theEnv,execStatus,"CommandCompletion",&theToken);
     }
   CloseStringSource(theEnv,execStatus,"CommandCompletion");
   ScannerData(theEnv,execStatus)->IgnoreCompletionErrors = FALSE;

   /*===============================================*/
   /* Determine if the last token can be completed. */
   /*===============================================*/

   if (lastToken.type == SYMBOL)
     {
      rs = ValueToString(lastToken.value);
      if (rs[0] == '[') return (&rs[1]);
      return(ValueToString(lastToken.value));
     }
   else if (lastToken.type == SF_VARIABLE)
     { return(ValueToString(lastToken.value)); }
   else if (lastToken.type == MF_VARIABLE)
     { return(ValueToString(lastToken.value)); }
   else if ((lastToken.type == GBL_VARIABLE) || (lastToken.type == MF_GBL_VARIABLE) ||
            (lastToken.type == INSTANCE_NAME))
     { return(NULL); }
   else if (lastToken.type == STRING)
     {
      length = strlen(ValueToString(lastToken.value));
      return(GetCommandCompletionString(theEnv,execStatus,ValueToString(lastToken.value),length));
     }
   else if ((lastToken.type == FLOAT) || (lastToken.type == INTEGER))
     { return(NULL); }

   return("");
  }

/****************************************************************/
/* SetHaltCommandLoopBatch: Sets the HaltCommandLoopBatch flag. */
/****************************************************************/
globle void SetHaltCommandLoopBatch(
  void *theEnv,
  EXEC_STATUS,
  int value)
  { 
   CommandLineData(theEnv,execStatus)->HaltCommandLoopBatch = value; 
  }

/*******************************************************************/
/* GetHaltCommandLoopBatch: Returns the HaltCommandLoopBatch flag. */
/*******************************************************************/
globle int GetHaltCommandLoopBatch(
  void *theEnv,
  EXEC_STATUS)
  {
   return(CommandLineData(theEnv,execStatus)->HaltCommandLoopBatch);
  }

#endif

