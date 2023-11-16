   /*******************************************************/
   /*      "C" Language Integrated Production System      */
   /*                                                     */
   /*             CLIPS Version 6.24  06/05/06            */
   /*                                                     */
   /*               TEXT PROCESSING MODULE                */
   /*******************************************************/

/*************************************************************/
/* Purpose:                                                  */
/*                                                           */
/* Principal Programmer(s):                                  */
/*      Brian L. Dantes                                      */
/*                                                           */
/* Contributing Programmer(s):                               */
/*      Gary D. Riley                                        */
/*                                                           */
/* Revision History:                                         */
/*      6.23: Modified error messages so that they were      */
/*            directly printed rather than storing them in   */
/*            a string buffer which might not be large       */
/*            enough to contain the entire message. DR0855   */
/*            Correction for FalseSymbol/TrueSymbol. DR0859  */
/*                                                           */
/*      6.24: Added get-region function.                     */
/*                                                           */
/*            Added environment parameter to GenClose.       */
/*            Added environment parameter to GenOpen.        */
/*                                                           */
/*************************************************************/


/**************************************************************************/
/**************************************************************************/
/*                        LOOKUP TABLE FUNCTIONS                          */
/*                                                                        */
/* The functions contained in this file set up and access a hierarchical  */
/* lookup system for multiple files.                                      */
/*                                                                        */
/* For usage see external documentation.                                  */
/**************************************************************************/
/**************************************************************************/

#define _TEXTPRO_SOURCE_

#include "setup.h"

#include <stdio.h>
#define _STDIO_INCLUDED_
#include <ctype.h>
#include <string.h>

#include "argacces.h"
#include "commline.h"
#include "envrnmnt.h"
#include "extnfunc.h"
#include "memalloc.h"
#include "router.h"
#include "sysdep.h"

#include "textpro.h"

#if TEXTPRO_FUNCTIONS || HELP_FUNCTIONS

#define NAMESIZE 80
#define NULLCHAR '\0'
#define BLANK (' ')
#define TAB ('\t')
#define LNFEED ('\n')

/*=========================================================*/
/*Status returns for the file loading and lookup functions */
/*=========================================================*/
#define NORMAL 0          /*Entry information found in file       */
#define NO_FILE -10       /*File not found for reference          */
#define NEW_FILE -15      /*File loaded onto internal lookup table*/
#define OLD_FILE -20      /*File was already on the lookup table  */

#define NO_TOPIC -25      /*No entry information was found in file*/
#define EXIT -30          /*Branch-up from root; exit lookup table*/
#define BRANCH_UP -35     /*Move up from subtopic entry to parent */
#define BRANCH_DOWN -40   /*Move down from main topic to subtopic */

/*=================*/
/*Entry data types */
/*=================*/
#define MENU -45    /*Entry has subtopics*/
#define INFO -50    /*Entry is a leaf; contains only information*/

/*==========================================*/
/*Entry node type for internal lookup table */
/*==========================================*/
struct entries
  {
   int level;              /*Level of entry node in the lookup tree  */
   int type;               /*Entry node data type : menu or info     */
   char name[NAMESIZE];    /*Entry node name                         */
   long int offset;        /*Location of entry info in the file      */
   struct entries *child;  /*Address of list of subtopic entries     */
   struct entries *parent; /*Address of parent topic entry           */
   struct entries *next;   /*Address of next entry at the same level */
  };

/*=========================================*/
/*File node type for internal lookup table */
/*=========================================*/
struct lists
  {
   char file[NAMESIZE];       /*File name                                */
   struct entries *topics;    /*Address of list of entry topics for file */
   struct entries *curr_menu; /*Address of current main topic in file    */
   struct lists *next;        /*Address of next file in the table        */
  };

/*==================================================*/
/*Delimeter strings for marking entries in the file */
/*==================================================*/
#define BDELIM "BEGIN-ENTRY-"
#define BDLEN 12
#define EDELIM "END-ENTRY"
#define EDLEN 9
#define BFORMAT "%d%1s%12s%s"   /*Format string for sscanf*/
#define LIT_DELIM ('$')

/*
#if WIN_BTC
#define OPEN_READ "rb"
#else
#define OPEN_READ "r"
#endif
*/
#define OPEN_READ "r"

#define TEXTPRO_DATA 8

struct textProcessingData
  { 
   struct lists *headings;
   struct entries *parent;
#if HELP_FUNCTIONS
   int HELP_INIT;
   char *help_file;
#endif
  };

#define TextProcessingData(theEnv,execStatus) ((struct textProcessingData *) GetEnvironmentData(theEnv,execStatus,TEXTPRO_DATA))

int TextLookupFetch(void *,EXEC_STATUS,char *);
int TextLookupToss(void *,EXEC_STATUS,char *);
static FILE *GetEntries(void *,EXEC_STATUS,char *,char **,char *,int *);
static FILE *GetCurrentMenu(void *,EXEC_STATUS,char *,int *);
static char *grab_string(void *,EXEC_STATUS,FILE *,char *,int);

static int findstr(char *,char *);
static void upper(char *);
static struct lists *NewFetchFile(void *,EXEC_STATUS,char *);
static struct entries *AllocateEntryNode(void *,EXEC_STATUS,FILE *,char *,char *,int);
static int AttachLeaf(void *,EXEC_STATUS,struct lists *,struct entries *,FILE *,char *,int);
static long LookupEntry(void *,EXEC_STATUS,char *,char **,char *,int *);
static void TossFunction(void *,EXEC_STATUS,struct entries *);
static void DeallocateTextProcessingData(void *,EXEC_STATUS);

/******************************************************************************/
/*============================================================================*/
/*                             INTERNAL ROUTINES                              */
/*============================================================================*/
/******************************************************************************/

/****************************************************************************/
/*LOAD FUNCTION :                                                           */
/* Input : 1) name of file to be loaded into the lookup table               */
/*         2) caller-allocated buffer to contain an error message (if any)  */
/*         3) size of error message buffer                                  */
/* Output :                                                                 */
/* This function attempts to load the file's topic information into the     */
/* lookup table according to the format below :                             */
/*                                                                          */
/* <level-num><entry-type-code>BEGIN-ENTRY-<topic-name>                     */
/*                .                                                         */
/*                .                                                         */
/*    Entry information in the form in which                                */
/*    it is to be displayed when referenced.                                */
/*                .                                                         */
/*                .                                                         */
/* END-ENTRY                                                                */
/*                                                                          */
/* The function returns the number of entries loaded if the entire file was */
/*   was correctly formatted, else it returns -1.                           */
/****************************************************************************/
globle int TextLookupFetch(
  void *theEnv,
  EXEC_STATUS,
  char *file)
  {
   FILE *fp;                     /*Pointer into stream of input file      */
   char str[256];                /*Buffer for storing input file lines    */
   int INFO_BEGIN, INFO_END;     /*Flags used to check proper syntax      */
   struct lists *lnode;          /*Used to store file node in list        */
   struct entries *enode;        /*Used to store entry node in topic list */
   int line_ct;                  /*Line count - used for error messages   */
   int entries_ct;               /*Number of entries successfully loaded. */

   fp = GenOpen(theEnv,execStatus,file,OPEN_READ);

   if (fp == NULL)
     {
      PrintErrorID(theEnv,execStatus,"TEXTPRO",1,FALSE);
      EnvPrintRouter(theEnv,execStatus,WERROR,"Could not open file \"");
      EnvPrintRouter(theEnv,execStatus,WERROR,file);
      EnvPrintRouter(theEnv,execStatus,WERROR,"\".\n");
      return(-1);
     }
     
   if ((lnode = NewFetchFile(theEnv,execStatus,file)) == NULL)
     {
      GenClose(theEnv,execStatus,fp);

      PrintErrorID(theEnv,execStatus,"TEXTPRO",2,FALSE);
      EnvPrintRouter(theEnv,execStatus,WERROR,"File \"");
      EnvPrintRouter(theEnv,execStatus,WERROR,file);
      EnvPrintRouter(theEnv,execStatus,WERROR,"\" already loaded.\n");
      return(-1);
     }

   /*===========================*/
   /*Store the file entry topics*/
   /*===========================*/
   line_ct = 0;
   entries_ct = 0;

   INFO_BEGIN = FALSE;
   INFO_END = TRUE;
   while (fgets(str,256,fp) != NULL)
     {
      line_ct++;

      /*=============================================================*/
      /*Forces the load function to ignore lines beginning with `$$' */
      /*=============================================================*/
      
      if ((str[0] != LIT_DELIM) || (str[1] != LIT_DELIM))
        {
        if (findstr(str,EDELIM) >= 0)
          {
           if (INFO_BEGIN == TRUE)
             {
              INFO_BEGIN = FALSE;
              INFO_END = TRUE;
              entries_ct++;
             }
           else
             {
              GenClose(theEnv,execStatus,fp);
              TextLookupToss(theEnv,execStatus,file);
              PrintErrorID(theEnv,execStatus,"TEXTPRO",8,FALSE);
              EnvPrintRouter(theEnv,execStatus,WERROR,"Line ");
              PrintLongInteger(theEnv,execStatus,WERROR,line_ct);
              EnvPrintRouter(theEnv,execStatus,WERROR," : Unmatched end marker.\n");
              return(-1);
             }
          }
        else if (findstr(str,BDELIM) >= 0)
          {
           if (INFO_END == TRUE)
              {
               INFO_END = FALSE;
               INFO_BEGIN = TRUE;
              }
            else
              {
               GenClose(theEnv,execStatus,fp);
               TextLookupToss(theEnv,execStatus,file);

               PrintErrorID(theEnv,execStatus,"TEXTPRO",4,FALSE);
               EnvPrintRouter(theEnv,execStatus,WERROR,"Line ");
               PrintLongInteger(theEnv,execStatus,WERROR,line_ct);
               EnvPrintRouter(theEnv,execStatus,WERROR," : Previous entry not closed.\n");

               return(-1);
              }

           if ((enode=AllocateEntryNode(theEnv,execStatus,fp,file,str,line_ct))==NULL)
             return(-1);

           /*=================================*/
           /*Store new entry node in the tree */
           /*=================================*/
           if (AttachLeaf(theEnv,execStatus,lnode,enode,fp,file,line_ct) == FALSE)
             return(-1);
          }
        }
     }
   GenClose(theEnv,execStatus,fp);
   if (INFO_END == FALSE)
     {
      TextLookupToss(theEnv,execStatus,file);
      
      PrintErrorID(theEnv,execStatus,"TEXTPRO",4,FALSE);
      EnvPrintRouter(theEnv,execStatus,WERROR,"Line ");
      PrintLongInteger(theEnv,execStatus,WERROR,line_ct);
      EnvPrintRouter(theEnv,execStatus,WERROR," : Previous entry not closed.\n");
      
      return(-1);
     }
   if (entries_ct == 0)
     TextLookupToss(theEnv,execStatus,file);
   return(entries_ct);
  }

/******************************************************************************/
/*FUNCTION UNLOAD :                                                           */
/* Input : 1) name of file to be taken off the lookup table                   */
/* Output : This functions deletes a file and all entry-topics associated with*/
/*          it from the lookup table and returns a boolean flag indicating    */
/*          failure or success.                                               */
/******************************************************************************/
globle int TextLookupToss(
  void *theEnv,
  EXEC_STATUS,
  char *file)
  {
   struct lists *plptr, *clptr;
   int l_flag;

   clptr = TextProcessingData(theEnv,execStatus)->headings;
   plptr = clptr;
   if (clptr != NULL)
     if (strcmp(clptr->file,file) != 0)
       l_flag = 1;
     else
       l_flag = 0;
   else
     l_flag = 0;
   while (l_flag > 0)
     {
      plptr = clptr;
      clptr = clptr->next;
      if (clptr != NULL)
        if (strcmp(clptr->file,file) != 0)
          l_flag = 1;
        else
          l_flag = 0;
      else
        l_flag = 0;
     }
   if (clptr == NULL)
     return(FALSE);

   TossFunction(theEnv,execStatus,clptr->topics);

   if (plptr == clptr)
     TextProcessingData(theEnv,execStatus)->headings = clptr->next;
   else
     plptr->next = clptr->next;
   rm(theEnv,execStatus,(void *) clptr,(int) sizeof(struct lists));
   return(TRUE);
  }

/******************************************************************************/
/*FUNCTION GET_ENTRIES :                                                      */
/* Input : 1) name of file to be accessed for lookup of entry                 */
/*         2) caller allocated buffer for main topic name                     */
/*         3) name of the entry to be accessed in the file                    */
/*         4) caller allocated buffer for a status code (see LOOKUP).         */
/* Output : 1) returns a pointer into the stream of the lookup file which     */
/*             indicates the starting position of the lookup information      */
/*             (NULL if the topic was not found)                              */
/* This function passes its input directly to LOOKUP.  See its description    */
/* for further detail.                                                        */
/*                                                                            */
/******************************************************************************/
static FILE *GetEntries(
  void *theEnv,
  EXEC_STATUS,
  char *file,
  char **menu,
  char *name,
  int *code)
  {
   FILE *fp;          /*Lookup file stream*/
   long int offset;   /*Offset from beginning of file to beginning of topic*/

   offset = LookupEntry(theEnv,execStatus,file,menu,name,code);
   if (offset < 0)
      return(NULL);
   fp = GenOpen(theEnv,execStatus,file,OPEN_READ);
   if (fp == NULL)
     {
      *code = NO_FILE;
      return(NULL);
     }
   if (fseek(fp,offset,0) < 0)
     {
      GenClose(theEnv,execStatus,fp);
      *code = NO_FILE;
      return(NULL);
     }
   return(fp);
  }

/******************************************************************************/
/*FUNCTION GET_CURR_MENU :                                                    */
/* Input : 1) name of file to be accessed                                     */
/*         2) caller allocated buffer for the current menu name               */
/*         3) caller allocated buffer for status code : NO_FILE, NO_TOPIC, or */
/*            NORMAL.                                                         */
/* Output : 1) returns a pointer into the file stream indicating the beginning*/
/*             of the description of the current menu for the named file      */
/*             (returns NULL if there is no current menu)                     */
/******************************************************************************/
static FILE *GetCurrentMenu(
  void *theEnv,
  EXEC_STATUS,
  char *file,
  int *status)
  {
   struct lists *lptr;   /*Used in searching the file list*/
   FILE *fp;             /*File stream*/
   int l_flag;           /*Used in looping through the file list*/

   /*=====================================*/
   /*Find the named file in the file list */
   /*=====================================*/
   lptr = TextProcessingData(theEnv,execStatus)->headings;
   if (lptr != NULL)
     if (strcmp(lptr->file,file) != 0)
       l_flag = 1;
     else
       l_flag = 0;
   else
     l_flag = 0;
   while (l_flag > 0)
     {
      lptr = lptr->next;
      if (lptr != NULL)
        if (strcmp(lptr->file,file) != 0)
          l_flag = 1;
        else
          l_flag = 0;
      else
        l_flag = 0;
     }
   if (lptr == NULL)
     {
      *status = NO_FILE;
      return(NULL);
     }

   /*============================================================*/
   /*Position the pointer in the file stream to the current menu */
   /*============================================================*/
   if (lptr->curr_menu == NULL)
     {
      *status = NO_TOPIC;
      return(NULL);
     }
   if ((fp = GenOpen(theEnv,execStatus,file,OPEN_READ)) == NULL)
     {
      *status = NO_FILE;
      return(NULL);
     }
   if (fseek(fp,lptr->curr_menu->offset,0) < 0)
     {
      GenClose(theEnv,execStatus,fp);
      *status = NO_FILE;
      return(NULL);
     }
   *status = NORMAL;
   return(fp);
  }

/******************************************************************************/
/*FUNCTION GRAB_STRING :                                                      */
/* Input : 1) file stream pointer                                             */
/*         2) caller allocated buffer for storage of read string              */
/*         3) size of caller's buffer                                         */
/* Output : This function grabs a line of text from the currently opened      */
/*          lookup file at the given file position in the stream.  If it      */
/*          encounters EOF or the closing topic delimeter, it closes the file */
/*          and returns NULL.  Otherwise, the return value is simply the      */
/*          address of the caller's buffer.                                   */
/*                                                                            */
/* Notes : 1) This function expects a file pointer into a stream of a file    */
/*            already opened!!                                                */
/*         2) The caller must close the file himself if he wishes to          */
/*            prematurely abort the complete reading of an entry.             */
/******************************************************************************/
static char *grab_string(
  void *theEnv,
  EXEC_STATUS,
  FILE *fp,
  char *buf,
  int bufsize)
  {
   if (fgets(buf,bufsize,fp) == NULL)
     {
      GenClose(theEnv,execStatus,fp);
      return(NULL);
     }
   if ((buf[0] == LIT_DELIM) && (buf[1] == LIT_DELIM))
     {
      buf[0] = BLANK;
      buf[1] = BLANK;
     }
   else if (findstr(buf,EDELIM) >= 0)
     {
      buf = NULL;
      GenClose(theEnv,execStatus,fp);
     }
   return(buf);
  }

/**************************************************************************/
/*FINDSTR FUNCTION :                                                      */
/* Input : 1) string to be searched                                       */
/*         2) string to be found                                          */
/* Output : 1) returns index of string-1 where string-2 started, if found */
/*          2) returns -1, if not found                                   */
/**************************************************************************/
static int findstr(
  char *s,
  char *t)
  {
   int i,j,k;

   for (i = 0; s[i] != '\0'; i++)
     {
      for (j = i, k = 0; t[k] != '\0' && s[j] == t[k]; j++, k++) ;
      if ((t[k] == '\0') && (k != 0))
        return(i);
     }
   return(-1);
  }

/**********************************************************************/
/*UPPER FUNCTION :                                                    */
/* Input : 1) alphanumeric string                                     */
/* Output : 1) all alphabetic characters of string are capitalized    */
/**********************************************************************/
static void upper(
  char *str)
  {
   int theIndex;

   for (theIndex = 0 ; str[theIndex] != NULLCHAR; theIndex++)
     if (islower(str[theIndex])) str[theIndex] = (char) toupper(str[theIndex]);
  }

/******************************************************************************/
/*FILE_NODE FUNCTION :                                                        */
/* Input : 1) name of a file                                                  */
/* Output : 1) returns address of an initalized NewFetchFile, if the file was    */
/*             not already on the lookup table                                */
/*          2) returns the null address, if the file was already present      */
/******************************************************************************/
static struct lists *NewFetchFile(
  void *theEnv,
  EXEC_STATUS,
  char *file)
  {
   struct lists *lptr = NULL, *lnode;

   if (TextProcessingData(theEnv,execStatus)->headings != NULL)
     {
      lptr = TextProcessingData(theEnv,execStatus)->headings;
      while (lptr->next != NULL)
        {
         if (strcmp(lptr->file,file) == 0)
           return(NULL);
         lptr = lptr->next;
        }
      if (strcmp(lptr->file,file) == 0)
        return(NULL);
     }
   lnode = (struct lists *) gm2(theEnv,execStatus,(int) sizeof(struct lists));
   genstrcpy(lnode->file,file);
   lnode->topics = NULL;
   lnode->curr_menu = NULL;
   lnode->next = NULL;
   if (TextProcessingData(theEnv,execStatus)->headings == NULL)
     TextProcessingData(theEnv,execStatus)->headings = lnode;
   else
     lptr->next = lnode;
   return(lnode);
  }

/******************************************************************************/
/*ENTRIES_NODE FUNCTION :                                                     */
/* Input : 1) file pointer                                                    */
/*         2) file name                                                       */
/*         3) input string from the file                                      */
/*         4) buffer for error messages                                       */
/*         5) size of the error message buffer                                */
/*         6) line count in the file                                          */
/* Output :                                                                   */
/*This function scans the input string for the appropriate topic entry        */
/*delimeter and, if it finds this to be correct, allocates a new entry node,  */
/*and initializes it, and returns the address to the calling routine.  If an  */
/*error is detected, the function writes an appropriate message to the        */
/*caller's buffer, deallocates the node, deletes all previous nodes from the  */
/*current file from the lookup table, closes the file, and returns the null   */
/*address.                                                                    */
/******************************************************************************/
static struct entries *AllocateEntryNode(
  void *theEnv,
  EXEC_STATUS,
  FILE *fp,
  char *file,
  char *str,
  int line_ct)
  {
   struct entries *enode;
   char bmarker[BDLEN+1],  /*Entry topic delimiting strings         */
        t_code[2];         /*Type of entry flag : menu or info      */


   /*================================================================*/
   /*Allocate a new node and scan the delimeter string for tree info */
   /*================================================================*/

   enode = (struct entries *) gm2(theEnv,execStatus,(int) sizeof(struct entries));
   if (sscanf(str,BFORMAT,
              &enode->level,t_code,bmarker,enode->name) != 4)
     {
      rm(theEnv,execStatus,(void *) enode,(int) sizeof(struct entries));
      GenClose(theEnv,execStatus,fp);
      TextLookupToss(theEnv,execStatus,file);

      PrintErrorID(theEnv,execStatus,"TEXTPRO",5,FALSE);
      EnvPrintRouter(theEnv,execStatus,WERROR,"Line ");
      PrintLongInteger(theEnv,execStatus,WERROR,line_ct);
      EnvPrintRouter(theEnv,execStatus,WERROR," : Invalid delimeter string.\n");

      return(NULL);
     }
   if (t_code[0] == 'M')
     enode->type = MENU;
   else if (t_code[0] == 'I')
     enode->type = INFO;
   else
     {
      rm(theEnv,execStatus,(void *) enode,(int) sizeof(struct entries));
      GenClose(theEnv,execStatus,fp);
      TextLookupToss(theEnv,execStatus,file);

      PrintErrorID(theEnv,execStatus,"TEXTPRO",6,FALSE);
      EnvPrintRouter(theEnv,execStatus,WERROR,"Line ");
      PrintLongInteger(theEnv,execStatus,WERROR,line_ct);
      EnvPrintRouter(theEnv,execStatus,WERROR," : Invalid entry type.\n");

      return(NULL);
     }
   if (strcmp(bmarker,BDELIM) != 0)
     {
      rm(theEnv,execStatus,(void *) enode,(int) sizeof(struct entries));
      GenClose(theEnv,execStatus,fp);
      TextLookupToss(theEnv,execStatus,file);

      PrintErrorID(theEnv,execStatus,"TEXTPRO",5,FALSE);
      EnvPrintRouter(theEnv,execStatus,WERROR,"Line ");
      PrintLongInteger(theEnv,execStatus,WERROR,line_ct);
      EnvPrintRouter(theEnv,execStatus,WERROR," : Invalid delimeter string.\n");

      return(NULL);
     }

   /*===============================================================*/
   /* For systems which have record file systems (such as VMS),     */
   /* the following statement is necessary to move the file pointer */
   /* to the beginning of the next record.                          */
   /*===============================================================*/
   ungetc(getc(fp),fp);
   enode->offset = ftell(fp);

   enode->parent = NULL;
   enode->child  = NULL;
   enode->next = NULL;
   upper(enode->name);

   return(enode);
  }

/******************************************************************************/
/*FUNCTION ATTACH_LEAF :                                                      */
/* Input : 1) address of current NewFetchFile                                    */
/*         2) address of current topic entry-node                             */
/*         3) file pointer                                                    */
/*         4) name of file                                                    */
/*         5) error message buffer                                            */
/*         6) size of error message buffer                                    */
/*         7) line count in the file                                          */
/* Output :                                                                   */
/*This function attaches the entry-node to its proper place in the tree of the*/
/*current file.  The function returns a boolean flag indicating the success   */
/*(or lack thereof) of this connection.  In the case of an error, an error    */
/*message is written to the caller's buffer, the file is closed, and the      */
/*previous file entries are deleted from the lookup table.                    */
/******************************************************************************/
static int AttachLeaf(
  void *theEnv,
  EXEC_STATUS,
  struct lists *lnode,
  struct entries *enode,
  FILE *fp,
  char *file,
  int line_ct)
  {
   int p_flag;   /*Used in searching the tree for a parent*/


   /*====================*/
   /*First topic for file*/
   /*====================*/
   if (lnode->topics == NULL)
     lnode->topics = enode;

   /*================================*/
   /*Subtopic - branch down the tree */
   /*================================*/
   else if (enode->level > TextProcessingData(theEnv,execStatus)->parent->level)
     if (TextProcessingData(theEnv,execStatus)->parent->type == MENU)
       {
        enode->parent = TextProcessingData(theEnv,execStatus)->parent;
        TextProcessingData(theEnv,execStatus)->parent->child = enode;
       }
     else
       {
        rm(theEnv,execStatus,(void *) enode,(int) sizeof(struct entries));
        GenClose(theEnv,execStatus,fp);
        TextLookupToss(theEnv,execStatus,file);

        PrintErrorID(theEnv,execStatus,"TEXTPRO",7,FALSE);
        EnvPrintRouter(theEnv,execStatus,WERROR,"Line ");
        PrintLongInteger(theEnv,execStatus,WERROR,line_ct);
        EnvPrintRouter(theEnv,execStatus,WERROR," : Non-menu entries cannot have subtopics.\n");

        return(FALSE);
       }
   /*====================================*/
   /*Brother-topic -- same level in tree */
   /*====================================*/
   else if (enode->level == TextProcessingData(theEnv,execStatus)->parent->level)
     {
      enode->parent = TextProcessingData(theEnv,execStatus)->parent->parent;
      enode->next = TextProcessingData(theEnv,execStatus)->parent->next;
      TextProcessingData(theEnv,execStatus)->parent->next = enode;
     }

   /*==========================================================*/
   /*Topic is unrelated to previous topic - branch up the tree */
   /*==========================================================*/
   else
     {
      if (TextProcessingData(theEnv,execStatus)->parent != NULL)
        p_flag = 1;
      else
        p_flag = 0;
      while (p_flag > 0)
        {
         TextProcessingData(theEnv,execStatus)->parent = TextProcessingData(theEnv,execStatus)->parent->parent;
         if (TextProcessingData(theEnv,execStatus)->parent != NULL)
           if (enode->level < TextProcessingData(theEnv,execStatus)->parent->level)
             p_flag = 1;
           else
             p_flag = 0;
         else
           p_flag = 0;
        }
      if (TextProcessingData(theEnv,execStatus)->parent != NULL)

        /*========*/
        /*Subtopic*/
        /*========*/
        if (TextProcessingData(theEnv,execStatus)->parent->level < enode->level)
          {
           enode->parent = TextProcessingData(theEnv,execStatus)->parent;
           enode->next = TextProcessingData(theEnv,execStatus)->parent->child;
           TextProcessingData(theEnv,execStatus)->parent->child = enode;
          }

        /*=============*/
        /*Brother-topic*/
        /*=============*/
        else
          {
           enode->parent = TextProcessingData(theEnv,execStatus)->parent->parent;
           enode->next = TextProcessingData(theEnv,execStatus)->parent->next;
           TextProcessingData(theEnv,execStatus)->parent->next = enode;
          }

      /*=========*/
      /*Root Node*/
      /*=========*/
      else
        {
         enode->parent = NULL;
         enode->next = lnode->topics;
         lnode->topics = enode;
        }
     }
   TextProcessingData(theEnv,execStatus)->parent = enode;
   return(TRUE);
  }

/******************************************************************************/
/*FUNCTION LOOKUP :                                                           */
/* Input : 1) name of entry-topic file to be used for reference               */
/*         2) caller allocated buffer to contain the main topic name          */
/*         3) name of the entry-topic to be found                             */
/*         4) caller allocated buffer to store the return status              */
/* Output : 1) offset from the beginning of the entry-topic file stream to the*/
/*             beginning of the requested topic (-1 if the topic not found)   */
/*          2) status code stored in caller's buffer indicating the result of */
/*             the lookup : NO_FILE, NO_TOPIC, BRANCH_UP, BRANCH_DOWN, EXIT,  */
/*             or NORMAL.                                                     */
/*                                                                            */
/* Notes : 1) If NULL is given as an entry-topic, the lookup routine branches */
/*            up one level in the tree (status BRANCH_UP).  If the current    */
/*            level of the tree is already the root, all paths are set to NULL*/
/*            (status EXIT).                                                  */
/*         2) If an entry-topic is not found, the file position of the current*/
/*            main topic (or menu) is returned (status NO_TOPIC).             */
/******************************************************************************/
static long int LookupEntry(
  void *theEnv,
  EXEC_STATUS,
  char *file,
  char **menu,
  char *name,
  int *code)
  {
   struct lists *lptr;    /*Local pointers used to move through the tree*/
   struct entries *eptr;
   int l_flag, e_flag;    /*Flags used in looping to find entry-topics*/

   /*===============================*/
   /*Find named file in lookup list */
   /*===============================*/
   lptr = TextProcessingData(theEnv,execStatus)->headings;
   if (lptr != NULL)
     if (strcmp(lptr->file,file) != 0)
       l_flag = 1;
     else
       l_flag = 0;
   else
     l_flag = 0;
   while (l_flag > 0)
     {
      lptr = lptr->next;
      if (lptr != NULL)
        if (strcmp(lptr->file,file) != 0)
          l_flag = 1;
        else
          l_flag = 0;
      else
        l_flag = 0;
     }
   if (lptr == NULL)
     {
      *code = NO_FILE;
      return(-1);
     }

   /*==================================================================*/
   /*If entry-topic was NULL, branch up one-level in the tree, or exit */
   /*the tree if already at the root.                                  */
   /*==================================================================*/
   if (name == NULL)
     {
      if (lptr->curr_menu == NULL)
        {
         *code = EXIT;
         return(-1);
        }
      else
        {
         if (lptr->curr_menu->parent == NULL)
           {
            *code = EXIT;
            lptr->curr_menu = NULL;
            *menu = NULL;
            return(-1);
           }
         lptr->curr_menu = lptr->curr_menu->parent;
         *code = BRANCH_UP;
         *menu = lptr->curr_menu->name;
         return(lptr->curr_menu->offset);
        }
     }

   /*========================================*/
   /*Find the topic in the file's topic tree */
   /*========================================*/
   upper(name);
   if (lptr->curr_menu != NULL)
     eptr = lptr->curr_menu->child;
   else
     eptr = lptr->topics;
   if (eptr != NULL)
     if (findstr(eptr->name,name) == 0)
       e_flag = 0;
     else
       e_flag = 1;
   else
     e_flag = 0;
   while (e_flag > 0)
     {
      eptr = eptr->next;
      if (eptr != NULL)
        if (findstr(eptr->name,name) == 0)
          e_flag = 0;
        else
          e_flag = 1;
      else
        e_flag = 0;
     }

   /*===================================================================*/
   /*If the topic was not found, return the position of the current menu*/
   /*===================================================================*/
   if (eptr == NULL)
     {
      *code = NO_TOPIC;
      if (lptr->curr_menu != NULL)
        {
         *menu = lptr->curr_menu->name;
         return(lptr->curr_menu->offset);
        }
      return(-1);
     }

   /*===============================================================*/
   /*If the requested topic has children, branch down to its level. */
   /*===============================================================*/
   if (eptr->type == MENU)
     {
      *code = BRANCH_DOWN;
      lptr->curr_menu = eptr;
     }
   else
     *code = NORMAL;

   if (lptr->curr_menu != NULL)
      *menu = lptr->curr_menu->name;
   return(eptr->offset);
  }

/******************************************************************************/
/*FUNCTION TOSS :                                                             */
/* Input : 1) entry-topic address                                             */
/* Output : This function recursively deletes a node and all child nodes      */
/******************************************************************************/
static void TossFunction(
  void *theEnv,
  EXEC_STATUS,
  struct entries *eptr)
  {
   struct entries *prev;

   while (eptr != NULL)
     {
      if (eptr->child != NULL)
        TossFunction(theEnv,execStatus,eptr->child);
      prev = eptr;
      eptr = eptr->next;
      rm(theEnv,execStatus,(void *) prev,(int) sizeof(struct entries));
     }
  }


/****************************************************************************/
/****************************************************************************/
/*                          TEXT PROCESSING FUNCTIONS                       */
/*                                                                          */
/* The functions contained in this file can be called to handle             */
/* external file referencing and accessing.  FetchCommand() loads a file    */
/* onto an internal run-time lookup table, TossCommand() removes the file,  */
/* PrintRegionCommand accesses the loaded file to display a requested       */
/* entry, and HelpFunction() provides an on-line help facility              */
/* using the external help data file specified in the header file setup.h.  */
/* For information on the format of the data file(s) required, see the      */
/* internal documentation in LOOKUP.C and the external documentation.       */
/*                                                                          */
/* For usage of these functions, see the external documentation.            */
/****************************************************************************/
/****************************************************************************/

#define SCREEN_LN 22   /*Typical terminal screen length -- 22 lines*/
                       /*Used for scrolling in the help facility   */

/*==========================================*/
/*Topic node for help facility's query list */
/*==========================================*/
struct topics
  {
   char name[NAMESIZE];      /*Name of the node                 */
   struct topics *end_list;  /*Pointer to end of query list     */
   struct topics *next;      /*Pointer to next topic in the list*/
  };

/******************************************************************************/
/*============================================================================*/
/*                        FUNCTION DECLARATIONS                               */
/*============================================================================*/
/******************************************************************************/

#if HELP_FUNCTIONS
static int RecognizeHelpRouters(void *,EXEC_STATUS,char *);
static int HelpPrint(void *,EXEC_STATUS,char *,char *);
static int HelpGetc(void *,EXEC_STATUS,char *);
static int HelpUngetc(void *,EXEC_STATUS,int,char *);
static struct topics *AskForNewHelpTopic(void *,EXEC_STATUS,struct topics *,char **);
#endif

static struct topics *GetCommandLineTopics(void *,EXEC_STATUS);
static FILE *FindTopicInEntries(void *,EXEC_STATUS,char *,struct topics *,char **,int *);

/******************************************************************************/
/*============================================================================*/
/*                       EXTERNAL ACCESS FUNCTIONS                            */
/*============================================================================*/
/******************************************************************************/

/******************************************************************************/
/*FUNCTION HelpFunction : (H/L function help)                               */
/* Input : Multiple or no topic requests may be passed to the help facility   */
/*         from the top level via a "stack" accessed by the          */
/*         system routines num_args() and rstring().                          */
/* Output : This function loads the help file specified in setup.h into a     */
/*          a hierarchical tree structure using the routines of LOOKUP.C.     */
/*          It then queries the user for topics, and, using the LOOKUP        */
/*          routines, branches through the tree, displaying information where */
/*          appropriate.  The function returns control once the user          */
/*          has indicated an exit from the help tree.                         */
/*                                                                            */
/* For usage see external documentation.                                      */
/******************************************************************************/

#if HELP_FUNCTIONS

globle void HelpFunction(
  void *theEnv,
  EXEC_STATUS)
  {
   int status;                     /*Return code from the lookup routines */
   FILE *fp;                       /*Pointer in to the help file stream   */
   struct topics *main_topic,      /*Pointer to the first requested topic */
                 *tptr;            /*Used in deallocating the topic list  */
   char buf[256],   /*Buffer for storing input strings from the help file */
        *menu[1];   /*Buffer for the name of the current main topic       */
#if ! WINDOW_INTERFACE
   char termbuf[2]; /*Buffer for storing the terminators of a scroll      */
   int line_cnt;    /*Line count used for scrolling purposes              */
#endif

   if (TextProcessingData(theEnv,execStatus)->HELP_INIT == FALSE)
     {
      if (TextProcessingData(theEnv,execStatus)->help_file == NULL)
        {
         TextProcessingData(theEnv,execStatus)->help_file = (char *) gm2(theEnv,execStatus,strlen(HELP_DEFAULT) + 1);
         genstrcpy(TextProcessingData(theEnv,execStatus)->help_file,HELP_DEFAULT);
        }
      EnvPrintRouter(theEnv,execStatus,WDIALOG,"Loading help file entries from ");
      EnvPrintRouter(theEnv,execStatus,WDIALOG,TextProcessingData(theEnv,execStatus)->help_file);
      EnvPrintRouter(theEnv,execStatus,WDIALOG,".\nPlease wait...\n");
      status = TextLookupFetch(theEnv,execStatus,TextProcessingData(theEnv,execStatus)->help_file);
      if (status <= 0)
        {
         return;
        }
      else
        {
         /* ================================================================
            Enables logical name "whelp" as the destination for all help I/O
            ================================================================ */
         EnvAddRouter(theEnv,execStatus,"whelp",10,RecognizeHelpRouters,HelpPrint,
                    HelpGetc,HelpUngetc,NULL);
         TextProcessingData(theEnv,execStatus)->HELP_INIT = TRUE;
        }
     }

   EnvActivateRouter(theEnv,execStatus,"whelp");

   /* ====================================================================
      The root node of the help-tree is MAIN (see external documentation.)
      Add this node to the front of the initial topic request list given
      by the user on the top level command line.
      ==================================================================== */
   main_topic = (struct topics *) gm2(theEnv,execStatus,(int) sizeof(struct topics));
   genstrcpy(main_topic->name,"MAIN");
   main_topic->next = GetCommandLineTopics(theEnv,execStatus);
   main_topic->end_list = NULL;

   EnvPrintRouter(theEnv,execStatus,"whelp","\n");

   /*============================*/
   /*Process user topic requests */
   /*============================*/
   do
     {
      fp = FindTopicInEntries(theEnv,execStatus,TextProcessingData(theEnv,execStatus)->help_file,main_topic,menu,&status);
      if (status == NO_FILE)
        {
         PrintErrorID(theEnv,execStatus,"TEXTPRO",1,FALSE);
         EnvPrintRouter(theEnv,execStatus,WERROR,"Unable to access help file.\n");
         break;
        }
      if (status == EXIT)
        break;
      if (status == NO_TOPIC)
        {
         if (fp == NULL)
           {
            /*===================================================*/
            /*The lookup routines return the file location of the*/
            /*current main topic if the requested topic is not   */
            /*found.  The help-tree has one root: MAIN (see      */
            /*external docs).  This topic should always be       */
            /*available.  Thus, if the topic was not found and   */
            /*there is no current menu, the help-file has been   */
            /*tampered with and should be corrected.             */
            /*===================================================*/
            EnvPrintRouter(theEnv,execStatus,"whelp","Root entry \"MAIN\" not found in ");
            EnvPrintRouter(theEnv,execStatus,"whelp",TextProcessingData(theEnv,execStatus)->help_file);
            EnvPrintRouter(theEnv,execStatus,"whelp",".\nSee external documentation.\n");
            break;
           }
         EnvPrintRouter(theEnv,execStatus,"whelp","\nSorry, no information available.\n\n");
        }
      if (status != BRANCH_UP)
        {
#if ! WINDOW_INTERFACE
         line_cnt = 0;
#endif

         /*======================================================*/
         /*Print lines from the information entry stopping after */
         /*every screenful of lines.  The user at that point has */
         /*the option to continue or abort the entry to continue */
         /*at the current menu level.                            */
         /*======================================================*/
         while (grab_string(theEnv,execStatus,fp,buf,256) != NULL)
           {
#if ! WINDOW_INTERFACE
            if (line_cnt >= (SCREEN_LN + 1))
              {
               EnvPrintRouter(theEnv,execStatus,"whelp","PRESS <RETURN> FOR MORE. ");
               EnvPrintRouter(theEnv,execStatus,"whelp","PRESS <A>,<RETURN> TO ABORT.");
               RouterData(theEnv,execStatus)->CommandBufferInputCount = 0;
               RouterData(theEnv,execStatus)->AwaitingInput = TRUE;
               do
                 {
                  termbuf[0] = (char) EnvGetcRouter(theEnv,execStatus,"whelp");
                  if (termbuf[0] != LNFEED)
                    {
                     if (termbuf[0] == 'a')
                       termbuf[0] = 'A';
                     if (termbuf[0] != '\b')
                       RouterData(theEnv,execStatus)->CommandBufferInputCount++;
                     else if (RouterData(theEnv,execStatus)->CommandBufferInputCount != 0)
                       RouterData(theEnv,execStatus)->CommandBufferInputCount--;
                     termbuf[1] = (char) EnvGetcRouter(theEnv,execStatus,"whelp");
                    }
                 }
               while ((termbuf[0] != LNFEED) &&
                      (termbuf[0] != 'A'));
               RouterData(theEnv,execStatus)->CommandBufferInputCount = 0;
               RouterData(theEnv,execStatus)->AwaitingInput = FALSE;
               line_cnt = 0;
               if (termbuf[0] == 'A')
                 {
                  GenClose(theEnv,execStatus,fp);
                  break;
                 }
              }
            line_cnt++;
#endif
            EnvPrintRouter(theEnv,execStatus,"whelp",buf);
           }
        }
      else if (fp != NULL)
        /*==========================================================*/
        /*If the user branched-up the help-tree, don't reprint that */
        /*menu.  However, the help file still needs to be closed.   */
        /*==========================================================*/
        GenClose(theEnv,execStatus,fp);

      main_topic = AskForNewHelpTopic(theEnv,execStatus,main_topic,menu);
      if (execStatus->HaltExecution)
        {
         while (status != EXIT)
           if ((fp = GetEntries(theEnv,execStatus,TextProcessingData(theEnv,execStatus)->help_file,menu,NULL,&status)) != NULL)
             GenClose(theEnv,execStatus,fp);
        }
     } while (status != EXIT);
   EnvDeactivateRouter(theEnv,execStatus,"whelp");

   /*========================================================*/
   /*Release any space used by the user's topic request list */
   /*========================================================*/
   while (main_topic != NULL)
     {
      tptr = main_topic;
      main_topic = main_topic->next;
      rm(theEnv,execStatus,(void *) tptr,(int) sizeof(struct topics));
     }
  }

/***************************************************************************/
/*FUNCTION HelpPathFunction : (function help-path)                         */
/* Input : Via the argument "stack", the name of the new help entries    */
/*         file, or no input.                                             */
/* Output : This function redefines the lookup file for the help facility. */
/*          If no argument is given, it displays the current file name.    */
/***************************************************************************/
globle void HelpPathFunction(
  void *theEnv,
  EXEC_STATUS)
  {
   char *help_name;
   DATA_OBJECT arg_ptr;

   if (EnvRtnArgCount(theEnv,execStatus) == 0)
     {
      EnvPrintRouter(theEnv,execStatus,WDIALOG,"The current help entries file is ");
      if (TextProcessingData(theEnv,execStatus)->help_file != NULL)
        EnvPrintRouter(theEnv,execStatus,WDIALOG,TextProcessingData(theEnv,execStatus)->help_file);
      else
        EnvPrintRouter(theEnv,execStatus,WDIALOG,HELP_DEFAULT);
      EnvPrintRouter(theEnv,execStatus,WDIALOG,"\n");
     }
   else
     {
      if (TextProcessingData(theEnv,execStatus)->help_file != NULL)
        {
         if (TextProcessingData(theEnv,execStatus)->HELP_INIT == TRUE)
           {
            EnvPrintRouter(theEnv,execStatus,WDIALOG,"Releasing help entries from file ");
            EnvPrintRouter(theEnv,execStatus,WDIALOG,TextProcessingData(theEnv,execStatus)->help_file);
            EnvPrintRouter(theEnv,execStatus,WDIALOG,"...\n");
            TextLookupToss(theEnv,execStatus,TextProcessingData(theEnv,execStatus)->help_file);
            EnvDeleteRouter(theEnv,execStatus,"whelp");
            TextProcessingData(theEnv,execStatus)->HELP_INIT = FALSE;
           }
         rm(theEnv,execStatus,(void *) TextProcessingData(theEnv,execStatus)->help_file,strlen(TextProcessingData(theEnv,execStatus)->help_file) + 1);
        }
      if (EnvArgTypeCheck(theEnv,execStatus,"help-path",1,SYMBOL_OR_STRING,&arg_ptr) == FALSE) return;
      help_name = DOToString(arg_ptr);
      TextProcessingData(theEnv,execStatus)->help_file = (char *) gm2(theEnv,execStatus,strlen(help_name) + 1);
      genstrcpy(TextProcessingData(theEnv,execStatus)->help_file,help_name);
      EnvPrintRouter(theEnv,execStatus,WDIALOG,"Help entries file reset to ");
      EnvPrintRouter(theEnv,execStatus,WDIALOG,help_name);
      EnvPrintRouter(theEnv,execStatus,WDIALOG,"\n");
     }
  }

#endif

#if TEXTPRO_FUNCTIONS

/***************************************************************************/
/*FUNCTION FetchCommand : (H/L function fetch)                             */
/* Input : Name of the file to be stored in the lookup table - passed via  */
/*         the argument "stack" and result buffer                          */
/* Output : This function loads a file into the internal lookup table and  */
/*          returns a (float) boolean flag indicating failure or success.  */
/***************************************************************************/
globle void FetchCommand(
  void *theEnv,
  EXEC_STATUS,
  DATA_OBJECT *result)
  {
   int load_ct;          /*Number of entries loaded */
   DATA_OBJECT arg_ptr;

   result->type = SYMBOL;
   result->value = EnvFalseSymbol(theEnv,execStatus);
   if (EnvArgTypeCheck(theEnv,execStatus,"fetch",1,SYMBOL_OR_STRING,&arg_ptr) == FALSE)
      return;
   load_ct = TextLookupFetch(theEnv,execStatus,DOToString(arg_ptr));
   if (load_ct <= 0)
     {
      if (load_ct == 0)
        {
         PrintErrorID(theEnv,execStatus,"TEXTPRO",3,FALSE);
         EnvPrintRouter(theEnv,execStatus,WERROR,"No entries found.\n");
        }

      return;
     }
   result->type = INTEGER;
   result->value = (void *) EnvAddLong(theEnv,execStatus,(long long) load_ct);
  }

/******************************************************************************/
/*FUNCTION PrintRegionCommand : (H/L function print-region)                 */
/* Input : Via the argument "stack", logical name for the output, the name of the */
/*         file to be accessed, and the name of the topic(s) to be looked up. */
/* Output : This function accesses a previously loaded file and prints the    */
/*          information of the topic entry requested to the screen.  The tree */
/*          structure must currently be at the correct level in order for the */
/*          topic to be accessed.  To branch down the tree, each topic in the */
/*          path to the one desired must be named.  Multiple arguments are    */
/*          allowed as in the help facility (see the external documentation.) */
/*          To branch up the tree, the special topic character `^' must be    */
/*          specified for each upwards branch.  Giving no topic name will     */
/*          cause a single branch-up in the tree.  The `?' character given at */
/*          the end of a path will return the current main topic menu.        */
/*                                                                            */
/* For usage, see the external documentation.                                 */
/******************************************************************************/
globle int PrintRegionCommand(
  void *theEnv,
  EXEC_STATUS)
  {
   struct topics *params,    /*Lookup file and list of topic requests  */
                 *tptr;      /*Used in deallocating the parameter list */
   char buf[256];            /*Buffer for the topic entry strings      */
   FILE *fp;                 /*Stream for the input file               */
   char *menu[1];            /*Buffer for the current menu name        */
   int status,               /*Lookup status return code               */
       com_code;             /*Completion flag                         */

   params = GetCommandLineTopics(theEnv,execStatus);
   fp = FindTopicInEntries(theEnv,execStatus,params->next->name,params->next->next,menu,&status);
   if ((status != NO_FILE) && (status != NO_TOPIC) && (status != EXIT))
     {
      if (strcmp(params->name,"t") == 0)
        genstrcpy(params->name,"stdout");
      EnvPrintRouter(theEnv,execStatus,params->name,"\n");
      while (grab_string(theEnv,execStatus,fp,buf,256) != NULL)
        EnvPrintRouter(theEnv,execStatus,params->name,buf);
      com_code = TRUE;
     }
   else
     {
      /* ==================================================================
         On NO_TOPIC results, the file is left open to point to the current
         menu.  This used as a check by the Help System.  In the case of
         print-region, however, we need to always make sure the file is
         closed.
         ================================================================== */
      if (fp != NULL)
        GenClose(theEnv,execStatus,fp);
      com_code = FALSE;
     }

   /* =======================================================
      Release any space used by the user's topic request list
      ======================================================= */
   while (params != NULL)
     {
      tptr = params;
      params = params->next;
      rm(theEnv,execStatus,(void *) tptr,(int) sizeof(struct topics));
     }
   return(com_code);
  }

/******************************************************************************/
/*FUNCTION GetRegionCommand : (H/L functionget-region)                 */
/******************************************************************************/
globle void *GetRegionCommand(
  void *theEnv,
  EXEC_STATUS)
  {
   struct topics *params,    /*Lookup file and list of topic requests  */
                 *tptr;      /*Used in deallocating the parameter list */
   char buf[256];            /*Buffer for the topic entry strings      */
   FILE *fp;                 /*Stream for the input file               */
   char *menu[1];            /*Buffer for the current menu name        */
   int status;               /*Lookup status return code               */
   char *theString = NULL;
   void *theResult;
   size_t oldPos = 0;
   size_t oldMax = 0;
   size_t sLength;

   params = GetCommandLineTopics(theEnv,execStatus);
   fp = FindTopicInEntries(theEnv,execStatus,params->name,params->next,menu,&status);
   if ((status != NO_FILE) && (status != NO_TOPIC) && (status != EXIT))
     {
      while (grab_string(theEnv,execStatus,fp,buf,256) != NULL)
        theString = AppendToString(theEnv,execStatus,buf,theString,&oldPos,&oldMax);
     }
   else
     {
      /* ==================================================================
         On NO_TOPIC results, the file is left open to point to the current
         menu.  This used as a check by the Help System.  In the case of
         print-region, however, we need to always make sure the file is
         closed.
         ================================================================== */
      if (fp != NULL)
        GenClose(theEnv,execStatus,fp);
     }

   /* =======================================================
      Release any space used by the user's topic request list
      ======================================================= */
   while (params != NULL)
     {
      tptr = params;
      params = params->next;
      rm(theEnv,execStatus,(void *) tptr,(int) sizeof(struct topics));
     }

   if (theString == NULL)
     { theResult = EnvAddSymbol(theEnv,execStatus,""); }
   else
     {
      sLength = strlen(theString);
      if ((sLength > 0) &&
          (((theString[sLength-1] == '\r') && (theString[sLength-2] == '\n'))
		   ||
           ((theString[sLength-1] == '\n') && (theString[sLength-2] == '\r'))))
        { theString[sLength-2] = 0; }
      theResult = EnvAddSymbol(theEnv,execStatus,theString);
     }

   if (theString != NULL)
     { genfree(theEnv,execStatus,theString,oldMax); }

   return(theResult);
  }

/***************************************************************************/
/*FUNCTION TossCommand : (H/L function toss)                             */
/* Input : Name of the file to be deleted from the lookup table (passed via*/
/*         the argument "stack")                                           */
/* Output : This function deletes the named file from the lookup table and */
/*          returns a (float) boolean flag indicating failure or success.  */
/***************************************************************************/
globle int TossCommand(
  void *theEnv,
  EXEC_STATUS)
  {
   char *file;   /*Name of the file */
   DATA_OBJECT arg_ptr;

   if (EnvArgTypeCheck(theEnv,execStatus,"toss",1,SYMBOL_OR_STRING,&arg_ptr) == FALSE)
     return (FALSE);
   file = DOToString(arg_ptr);

#if HELP_FUNCTIONS

    if (TextProcessingData(theEnv,execStatus)->help_file != NULL)
      if ((strcmp(file,TextProcessingData(theEnv,execStatus)->help_file) == 0) && (TextProcessingData(theEnv,execStatus)->HELP_INIT == TRUE))
        {
         rm(theEnv,execStatus,(void *) TextProcessingData(theEnv,execStatus)->help_file,strlen(TextProcessingData(theEnv,execStatus)->help_file) + 1);
         TextProcessingData(theEnv,execStatus)->help_file = NULL;
         TextProcessingData(theEnv,execStatus)->HELP_INIT = FALSE;
         EnvDeleteRouter(theEnv,execStatus,"whelp");
        }

#endif

   return(TextLookupToss(theEnv,execStatus,file));
  }

#endif

/******************************************************************************/
/* The following four functions are the router routines for the logical name  */
/* "whelp".  Currently, all they do is direct all accesses to standard I/O.   */
/******************************************************************************/

#if HELP_FUNCTIONS

#if WIN_BTC
#pragma argsused
#endif
static int RecognizeHelpRouters(
  void *theEnv,
  EXEC_STATUS,
  char *log_name)
  {
#if MAC_MCW || WIN_MCW || MAC_XCD
#pragma unused(theEnv,execStatus)
#endif

   if (strcmp(log_name,"whelp") == 0)
     return(TRUE);
   return(FALSE);
  }

#if WIN_BTC
#pragma argsused
#endif
static int HelpPrint(
  void *theEnv,
  EXEC_STATUS,
  char *log_name,
  char *str)
  {
#if MAC_MCW || WIN_MCW || MAC_XCD
#pragma unused(log_name)
#endif

   EnvPrintRouter(theEnv,execStatus,"stdout",str);
   return(1);
  }

#if WIN_BTC
#pragma argsused
#endif
static int HelpGetc(
  void *theEnv,
  EXEC_STATUS,
  char *log_name)
  {
#if MAC_MCW || WIN_MCW || MAC_XCD
#pragma unused(log_name)
#endif

   return(EnvGetcRouter(theEnv,execStatus,"stdin"));
  }

#if WIN_BTC
#pragma argsused
#endif
static int HelpUngetc(
  void *theEnv,
  EXEC_STATUS,
  int ch,
  char *log_name)
  {
#if MAC_MCW || WIN_MCW || MAC_XCD
#pragma unused(log_name)
#endif

   return(EnvUngetcRouter(theEnv,execStatus,ch,"stdin"));
  }

#endif

/******************************************************************************/
/*============================================================================*/
/*                            INTERNAL ROUTINES                               */
/*============================================================================*/
/******************************************************************************/

/******************************************************************************/
/*FUNCTION CMD_LINE_TOPICS :                                                  */
/* Input : None                                                               */
/* Output : This function builds a linked list of topics requested by the     */
/*          user at the H/L level using the argument "stack" routines,        */
/*          num_args() and rstring().  It returns the address of the top of   */
/*          the list or NULL if there were no command line topics.            */
/******************************************************************************/
static struct topics *GetCommandLineTopics(
  void *theEnv,
  EXEC_STATUS)
  {
   int topic_num,         /*Number of topics specified by the user */
       theIndex;             /*Used to loop through the topic list    */
   struct topics *head,   /*Address of the top of the topic list   */
                 *tnode,  /*Address of new topic node              */
                 *tptr;   /*Used to attach new node to the list    */
   DATA_OBJECT val;       /*Unknown-type H/L data structure        */

   head = NULL;
   topic_num = EnvRtnArgCount(theEnv,execStatus);
   for (theIndex = 1; theIndex <= topic_num; theIndex++)
     {
      tnode = (struct topics *) gm2(theEnv,execStatus,(int) sizeof(struct topics));
      EnvRtnUnknown(theEnv,execStatus,theIndex,&val);
      if ((GetType(val) == SYMBOL) || (GetType(val) == STRING))
        genstrncpy(tnode->name,DOToString(val),NAMESIZE-1);
      else if (GetType(val) == FLOAT)
        genstrncpy(tnode->name,FloatToString(theEnv,execStatus,DOToDouble(val)),NAMESIZE-1);
      else if (GetType(val) == INTEGER)
        genstrncpy(tnode->name,LongIntegerToString(theEnv,execStatus,DOToLong(val)),NAMESIZE-1);
      else
        genstrncpy(tnode->name,"***ERROR***",NAMESIZE-1);
      tnode->next = NULL;
      tnode->end_list = NULL;
      if (head == NULL)
        head = tnode;
      else
        {
         tptr = head;
         while (tptr->next != NULL)
           tptr = tptr->next;
         tptr->next = tnode;
        }
     }
    return(head);
  }

/******************************************************************************/
/*FUNCTION QUERY_TOPIC :                                                      */
/* Input : 1) The address of the old topic list (this routines writes over    */
/*            previously allocated memory, if available)                      */
/*         2) A buffer holding the name of the current menu in the tree       */
/* Output : This function prompts the user for a new set of topic(s) and      */
/*          displays the name of the current menu.  Each new topic is         */
/*          delineated by white-space, and this function builds a linked list */
/*          of these topics.  It returns the address of the top of this list. */
/******************************************************************************/

#if HELP_FUNCTIONS

static struct topics *AskForNewHelpTopic(
  void *theEnv,
  EXEC_STATUS,
  struct topics *old_list,
  char **menu)
  {
   int theIndex, cnt;       /*Indices of the user input buffer and topic name */
   struct topics *tmain,  /*Address of the top of the topic list            */
                 *tnode, /*Address of the new topic node                   */
                 *tptr;  /*Used to add the new node to the topic list      */
   char list[256],       /*User input buffer                               */
        name[NAMESIZE];  /*Name of the new topic in the list               */

   /*==================================================================*/
   /*Read a line of input from the user (substituting blanks for tabs) */
   /*==================================================================*/
   EnvPrintRouter(theEnv,execStatus,"whelp",*menu);
   EnvPrintRouter(theEnv,execStatus,"whelp"," Topic? ");
   RouterData(theEnv,execStatus)->CommandBufferInputCount = 0;
   RouterData(theEnv,execStatus)->AwaitingInput = TRUE;
   for ( theIndex = 0;
         ((list[theIndex] = (char) EnvGetcRouter(theEnv,execStatus,"whelp")) != LNFEED) && (theIndex < 254);
         theIndex++ , RouterData(theEnv,execStatus)->CommandBufferInputCount++)
       {
        if (execStatus->HaltExecution)
          break;
        if (list[theIndex] == TAB)
          list[theIndex] = BLANK;
        else if ((list[theIndex] == '\b') && (theIndex != 0))
          {
           theIndex -= 2;
           RouterData(theEnv,execStatus)->CommandBufferInputCount -= 2;
          }
       }
#if VAX_VMS
   EnvPrintRouter(theEnv,execStatus,"whelp","\n");
#endif

   RouterData(theEnv,execStatus)->CommandBufferInputCount = 0;
   RouterData(theEnv,execStatus)->AwaitingInput = FALSE;
   if (execStatus->HaltExecution)
     {
      EnvPrintRouter(theEnv,execStatus,"whelp","\n");
      old_list->end_list = old_list;
      return(old_list);
     }
   list[theIndex] = BLANK;
   list[theIndex+1] = NULLCHAR;

   /*=======================================*/
   /*Parse user buffer into separate topics */
   /*=======================================*/
   tmain = old_list;
   theIndex = 0; cnt = 0;
   while (list[theIndex] != NULLCHAR)
     {
      if ((list[theIndex] != BLANK) && (cnt < NAMESIZE))
        name[cnt++] = list[theIndex++];
      else if (cnt > 0)
        {
         while ((list[theIndex] != BLANK) && (list[theIndex] != NULLCHAR))
           theIndex++;
         name[cnt] = NULLCHAR;
         cnt = 0;

         /*==============================================*/
         /*Write over previous topic lists, if available */
         /*==============================================*/
         if (old_list != NULL)
           {
            genstrcpy(old_list->name,name);
            old_list = old_list->next;
           }
         else
           {
            tnode = (struct topics *) gm2(theEnv,execStatus,(int) sizeof(struct topics));
            genstrcpy(tnode->name,name);
            tnode->next = NULL;
            tnode->end_list = NULL;
            if (tmain == NULL)
              tmain = tnode;
            else
              {
               tptr = tmain;
               while (tptr->next != NULL)
                 tptr = tptr->next;
               tptr->next = tnode;
              }
           }
        }
      else
        theIndex++;
     }

  /*========================================================================*/
  /*If the new list is shorter than the previous one, we must mark the end. */
  /*========================================================================*/
  tmain->end_list = old_list;
  return(tmain);
 }


#endif

/******************************************************************************/
/*FUNCTION FIND_TOPIC :                                                       */
/* Input : 1) File to be searched for topic request                           */
/*         2) Address of topic request list                                   */
/*         3) Buffer for current menu name                                    */
/*         4) Lookup status return code                                       */
/* Output : This function flows through the user topic request path by        */
/*          calling the lookup routines.  When it reaches the last element,   */
/*          it returns a pointer into the stream of the lookup file           */
/*          indicating the beginning of the topic entry.  If any topic in the */
/*          path is not found, the function aborts and returns the address of */
/*          of the current menu in the lookup tree for the file.  The exact   */
/*          nature of the final lookup is indicated in the status buffer.     */
/******************************************************************************/
static FILE *FindTopicInEntries(
  void *theEnv,
  EXEC_STATUS,
  char *file,
  struct topics *main_topic,
  char **menu,
  int *status)
  {
   FILE *fp = NULL;                 /*Input file stream                    */
   struct topics *tptr,             /*Used to loop through the topic list  */
                 *end_list;         /*Address of the end of the topic list */

   if (main_topic != NULL)
     end_list = main_topic->end_list;
   else
     end_list = NULL;
   tptr = main_topic;
   if (tptr != end_list)
     do
       {
        if (fp != NULL)
          GenClose(theEnv,execStatus,fp);

        /*======================*/
        /*Branch up in the tree */
        /*======================*/
        if (strcmp(tptr->name,"^") == 0)
          fp = GetEntries(theEnv,execStatus,file,menu,NULL,status);

        /*=======================================================*/
        /*Return the current main topic menu of the lookup table */
        /*=======================================================*/
        else if ((strcmp(tptr->name,"?") == 0) && (tptr->next == end_list))
          fp = GetCurrentMenu(theEnv,execStatus,file,status);

        /*=====================*/
        /*Lookup topic request */
        /*=====================*/
        else
          fp = GetEntries(theEnv,execStatus,file,menu,tptr->name,status);

        if ((*status == NO_FILE) || (*status == NO_TOPIC))
          break;
        tptr = tptr->next;
       } while (tptr != end_list);
   else
     /*==================================================================*/
     /*An empty topic request list causes a single branch-up in the tree */
     /*==================================================================*/
     fp = GetEntries(theEnv,execStatus,file,menu,NULL,status);

   return(fp);
  }

/*******************************************/
/* HelpFunctionDefinitions:                */
/*******************************************/
globle void HelpFunctionDefinitions(
  void *theEnv,
  EXEC_STATUS)
  {
   AllocateEnvironmentData(theEnv,execStatus,TEXTPRO_DATA,sizeof(struct textProcessingData),DeallocateTextProcessingData);
#if ! RUN_TIME
#if HELP_FUNCTIONS
   EnvDefineFunction2(theEnv,execStatus,"help",'v',PTIEF HelpFunction,"HelpFunction",NULL);
   EnvDefineFunction2(theEnv,execStatus,"help-path",'v',PTIEF HelpPathFunction,"HelpPathFunction","*1k");
#endif

#if TEXTPRO_FUNCTIONS
   EnvDefineFunction2(theEnv,execStatus,"fetch",'u', PTIEF FetchCommand,"FetchCommand","11k");
   EnvDefineFunction2(theEnv,execStatus,"toss",'b', PTIEF TossCommand,"TossCommand","11k");
   EnvDefineFunction2(theEnv,execStatus,"print-region",'b', PTIEF PrintRegionCommand,"PrintRegionCommand","2**wk");
   EnvDefineFunction2(theEnv,execStatus,"get-region",'s', PTIEF GetRegionCommand,"GetRegionCommand","1**k");
#endif
#endif
  }
  
/*********************************************************/
/* DeallocateTextProcessingData: Deallocates environment */
/*    data for text processing routines.                 */
/*********************************************************/
static void DeallocateTextProcessingData(
  void *theEnv,
  EXEC_STATUS)
  {
   struct lists *nextptr, *clptr;

   clptr = TextProcessingData(theEnv,execStatus)->headings;
   while (clptr != NULL)
     {
      nextptr = clptr->next;
     
      TossFunction(theEnv,execStatus,clptr->topics);
      rm(theEnv,execStatus,(void *) clptr,(int) sizeof(struct lists));

      clptr = nextptr;
     }
     
#if HELP_FUNCTIONS
   if (TextProcessingData(theEnv,execStatus)->help_file != NULL)
     {
      rm(theEnv,execStatus,TextProcessingData(theEnv,execStatus)->help_file,
         strlen(TextProcessingData(theEnv,execStatus)->help_file) + 1);
     }
#endif
  }


#endif

