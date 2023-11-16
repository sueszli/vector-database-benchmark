   /*******************************************************/
   /*      "C" Language Integrated Production System      */
   /*                                                     */
   /*             CLIPS Version 6.21  06/15/03            */
   /*                                                     */
   /*            DEFGLOBAL BSAVE/BLOAD MODULE             */
   /*******************************************************/

/*************************************************************/
/* Purpose: Implements the binary save/load feature for the  */
/*    defglobal construct.                                   */
/*                                                           */
/* Principal Programmer(s):                                  */
/*      Gary D. Riley                                        */
/*                                                           */
/* Contributing Programmer(s):                               */
/*      Brian L. Dantes                                      */
/*                                                           */
/* Revision History:                                         */
/*                                                           */
/*************************************************************/

#define _GLOBLBIN_SOURCE_

#include "setup.h"

#if DEFGLOBAL_CONSTRUCT && (BLOAD || BLOAD_AND_BSAVE || BLOAD_ONLY) && (! RUN_TIME)

#include <stdio.h>
#define _STDIO_INCLUDED_

#include "memalloc.h"
#include "multifld.h"
#include "globldef.h"
#include "bload.h"
#include "bsave.h"
#include "moduldef.h"
#include "globlbsc.h"
#include "envrnmnt.h"

#include "globlbin.h"

/***************************************/
/* LOCAL INTERNAL FUNCTION DEFINITIONS */
/***************************************/

#if BLOAD_AND_BSAVE
   static void                    BsaveFind(void *,EXEC_STATUS);
   static void                    BsaveStorage(void *,EXEC_STATUS,FILE *);
   static void                    BsaveBinaryItem(void *,EXEC_STATUS,FILE *);
#endif
   static void                    BloadStorageDefglobals(void *,EXEC_STATUS);
   static void                    BloadBinaryItem(void *,EXEC_STATUS);
   static void                    UpdateDefglobalModule(void *,EXEC_STATUS,void *,long);
   static void                    UpdateDefglobal(void *,EXEC_STATUS,void *,long);
   static void                    ClearBload(void *,EXEC_STATUS);
   static void                    DeallocateDefglobalBloadData(void *,EXEC_STATUS);

/*********************************************/
/* DefglobalBinarySetup: Installs the binary */
/*   save/load feature for the defglobals.   */
/*********************************************/
globle void DefglobalBinarySetup(
  void *theEnv,
  EXEC_STATUS)
  {
   AllocateEnvironmentData(theEnv,execStatus,GLOBLBIN_DATA,sizeof(struct defglobalBinaryData),DeallocateDefglobalBloadData);
#if (BLOAD_AND_BSAVE || BLOAD)
   AddAfterBloadFunction(theEnv,execStatus,"defglobal",ResetDefglobals,50);
#endif

#if BLOAD_AND_BSAVE
   AddBinaryItem(theEnv,execStatus,"defglobal",0,BsaveFind,NULL,
                             BsaveStorage,BsaveBinaryItem,
                             BloadStorageDefglobals,BloadBinaryItem,
                             ClearBload);
#endif

#if (BLOAD || BLOAD_ONLY)
   AddBinaryItem(theEnv,execStatus,"defglobal",0,NULL,NULL,NULL,NULL,
                             BloadStorageDefglobals,BloadBinaryItem,
                             ClearBload);
#endif
  }

/*********************************************************/
/* DeallocateDefglobalBloadData: Deallocates environment */
/*    data for the defglobal bsave functionality.        */
/*********************************************************/
static void DeallocateDefglobalBloadData(
  void *theEnv,
  EXEC_STATUS)
  {
#if (BLOAD || BLOAD_ONLY || BLOAD_AND_BSAVE) && (! RUN_TIME)
   size_t space;
   long i;

   for (i = 0; i < DefglobalBinaryData(theEnv,execStatus)->NumberOfDefglobals; i++)
     {
      if (DefglobalBinaryData(theEnv,execStatus)->DefglobalArray[i].current.type == MULTIFIELD)
        { ReturnMultifield(theEnv,execStatus,(struct multifield *) DefglobalBinaryData(theEnv,execStatus)->DefglobalArray[i].current.value); }
     }

   space = DefglobalBinaryData(theEnv,execStatus)->NumberOfDefglobals * sizeof(struct defglobal);
   if (space != 0) 
     { genfree(theEnv,execStatus,(void *) DefglobalBinaryData(theEnv,execStatus)->DefglobalArray,space); }

   space =  DefglobalBinaryData(theEnv,execStatus)->NumberOfDefglobalModules * sizeof(struct defglobalModule);
   if (space != 0) 
     { genfree(theEnv,execStatus,(void *) DefglobalBinaryData(theEnv,execStatus)->ModuleArray,space); }
#endif
  }

#if BLOAD_AND_BSAVE

/****************************************************/
/* BsaveFind:  Counts the number of data structures */
/*   which must be saved in the binary image for    */
/*   the defglobals in the current environment.     */
/****************************************************/
static void BsaveFind(
  void *theEnv,
  EXEC_STATUS)
  {
   struct defglobal *defglobalPtr;
   struct defmodule *theModule;

   /*=======================================================*/
   /* If a binary image is already loaded, then temporarily */
   /* save the count values since these will be overwritten */
   /* in the process of saving the binary image.            */
   /*=======================================================*/

   SaveBloadCount(theEnv,execStatus,DefglobalBinaryData(theEnv,execStatus)->NumberOfDefglobalModules);
   SaveBloadCount(theEnv,execStatus,DefglobalBinaryData(theEnv,execStatus)->NumberOfDefglobals);

   /*============================================*/
   /* Set the count of defglobals and defglobals */
   /* module data structures to zero.            */
   /*============================================*/

   DefglobalBinaryData(theEnv,execStatus)->NumberOfDefglobals = 0;
   DefglobalBinaryData(theEnv,execStatus)->NumberOfDefglobalModules = 0;

   for (theModule = (struct defmodule *) EnvGetNextDefmodule(theEnv,execStatus,NULL);
        theModule != NULL;
        theModule = (struct defmodule *) EnvGetNextDefmodule(theEnv,execStatus,theModule))
     {
      /*================================================*/
      /* Set the current module to the module being     */
      /* examined and increment the number of defglobal */
      /* modules encountered.                           */
      /*================================================*/

      EnvSetCurrentModule(theEnv,execStatus,(void *) theModule);
      DefglobalBinaryData(theEnv,execStatus)->NumberOfDefglobalModules++;

      /*====================================================*/
      /* Loop through each defglobal in the current module. */
      /*====================================================*/

      for (defglobalPtr = (struct defglobal *) EnvGetNextDefglobal(theEnv,execStatus,NULL);
           defglobalPtr != NULL;
           defglobalPtr = (struct defglobal *) EnvGetNextDefglobal(theEnv,execStatus,defglobalPtr))
        {
         /*======================================================*/
         /* Initialize the construct header for the binary save. */
         /*======================================================*/

         MarkConstructHeaderNeededItems(&defglobalPtr->header,DefglobalBinaryData(theEnv,execStatus)->NumberOfDefglobals++);
        }
     }
  }

/*****************************************************/
/* BsaveStorage: Writes out storage requirements for */
/*   all defglobal structures to the binary file     */
/*****************************************************/
static void BsaveStorage(
  void *theEnv,
  EXEC_STATUS,
  FILE *fp)
  {
   size_t space;

   /*===========================================================*/
   /* Only two data structures are saved as part of a defglobal */
   /* binary image: the defglobal data structure and the        */
   /* defglobalModule data structure.                           */
   /*===========================================================*/

   space = sizeof(long) * 2;
   GenWrite(&space,sizeof(size_t),fp);
   GenWrite(&DefglobalBinaryData(theEnv,execStatus)->NumberOfDefglobals,sizeof(long int),fp);
   GenWrite(&DefglobalBinaryData(theEnv,execStatus)->NumberOfDefglobalModules,sizeof(long int),fp);
  }

/*********************************************/
/* BsaveBinaryItem: Writes out all defglobal */
/*   structures to the binary file           */
/*********************************************/
static void BsaveBinaryItem(
  void *theEnv,
  EXEC_STATUS,
  FILE *fp)
  {
   size_t space;
   struct defglobal *theDefglobal;
   struct bsaveDefglobal newDefglobal;
   struct defmodule *theModule;
   struct bsaveDefglobalModule tempDefglobalModule;
   struct defglobalModule *theModuleItem;

   /*==========================================================*/
   /* Write out the amount of space taken up by the defglobal  */
   /* and defglobalModule data structures in the binary image. */
   /*==========================================================*/

   space = DefglobalBinaryData(theEnv,execStatus)->NumberOfDefglobals * sizeof(struct bsaveDefglobal) +
           (DefglobalBinaryData(theEnv,execStatus)->NumberOfDefglobalModules * sizeof(struct bsaveDefglobalModule));
   GenWrite(&space,sizeof(size_t),fp);

   /*=================================================*/
   /* Write out each defglobal module data structure. */
   /*=================================================*/

   DefglobalBinaryData(theEnv,execStatus)->NumberOfDefglobals = 0;
   for (theModule = (struct defmodule *) EnvGetNextDefmodule(theEnv,execStatus,NULL);
        theModule != NULL;
        theModule = (struct defmodule *) EnvGetNextDefmodule(theEnv,execStatus,theModule))
     {
      EnvSetCurrentModule(theEnv,execStatus,(void *) theModule);

      theModuleItem = (struct defglobalModule *)
                      GetModuleItem(theEnv,execStatus,NULL,FindModuleItem(theEnv,execStatus,"defglobal")->moduleIndex);
      AssignBsaveDefmdlItemHdrVals(&tempDefglobalModule.header,
                                           &theModuleItem->header);
      GenWrite(&tempDefglobalModule,sizeof(struct bsaveDefglobalModule),fp);
     }

   /*===========================*/
   /* Write out each defglobal. */
   /*===========================*/

   DefglobalBinaryData(theEnv,execStatus)->NumberOfDefglobals = 0;
   for (theModule = (struct defmodule *) EnvGetNextDefmodule(theEnv,execStatus,NULL);
        theModule != NULL;
        theModule = (struct defmodule *) EnvGetNextDefmodule(theEnv,execStatus,theModule))
     {
      EnvSetCurrentModule(theEnv,execStatus,(void *) theModule);

      for (theDefglobal = (struct defglobal *) EnvGetNextDefglobal(theEnv,execStatus,NULL);
           theDefglobal != NULL;
           theDefglobal = (struct defglobal *) EnvGetNextDefglobal(theEnv,execStatus,theDefglobal))
        {
         AssignBsaveConstructHeaderVals(&newDefglobal.header,
                                          &theDefglobal->header);
         newDefglobal.initial = HashedExpressionIndex(theEnv,execStatus,theDefglobal->initial);

         GenWrite(&newDefglobal,sizeof(struct bsaveDefglobal),fp);
        }
     }

   /*=============================================================*/
   /* If a binary image was already loaded when the bsave command */
   /* was issued, then restore the counts indicating the number   */
   /* of defglobals and defglobal modules in the binary image     */
   /* (these were overwritten by the binary save).                */
   /*=============================================================*/

   RestoreBloadCount(theEnv,execStatus,&DefglobalBinaryData(theEnv,execStatus)->NumberOfDefglobalModules);
   RestoreBloadCount(theEnv,execStatus,&DefglobalBinaryData(theEnv,execStatus)->NumberOfDefglobals);
  }

#endif /* BLOAD_AND_BSAVE */

/***********************************************/
/* BloadStorageDefglobals: Allocates space for */
/*   the defglobals used by this binary image. */
/***********************************************/
static void BloadStorageDefglobals(
  void *theEnv,
  EXEC_STATUS)
  {
   size_t space;

   /*=======================================================*/
   /* Determine the number of defglobal and defglobalModule */
   /* data structures to be read.                           */
   /*=======================================================*/

   GenReadBinary(theEnv,execStatus,&space,sizeof(size_t));
   GenReadBinary(theEnv,execStatus,&DefglobalBinaryData(theEnv,execStatus)->NumberOfDefglobals,sizeof(long int));
   GenReadBinary(theEnv,execStatus,&DefglobalBinaryData(theEnv,execStatus)->NumberOfDefglobalModules,sizeof(long int));

   /*===================================*/
   /* Allocate the space needed for the */
   /* defglobalModule data structures.  */
   /*===================================*/

   if (DefglobalBinaryData(theEnv,execStatus)->NumberOfDefglobalModules == 0)
     {
      DefglobalBinaryData(theEnv,execStatus)->DefglobalArray = NULL;
      DefglobalBinaryData(theEnv,execStatus)->ModuleArray = NULL;
     }

   space = DefglobalBinaryData(theEnv,execStatus)->NumberOfDefglobalModules * sizeof(struct defglobalModule);
   DefglobalBinaryData(theEnv,execStatus)->ModuleArray = (struct defglobalModule *) genalloc(theEnv,execStatus,space);

   /*===================================*/
   /* Allocate the space needed for the */
   /* defglobal data structures.        */
   /*===================================*/

   if (DefglobalBinaryData(theEnv,execStatus)->NumberOfDefglobals == 0)
     {
      DefglobalBinaryData(theEnv,execStatus)->DefglobalArray = NULL;
      return;
     }

   space = (DefglobalBinaryData(theEnv,execStatus)->NumberOfDefglobals * sizeof(struct defglobal));
   DefglobalBinaryData(theEnv,execStatus)->DefglobalArray = (struct defglobal *) genalloc(theEnv,execStatus,space);
  }

/******************************************************/
/* BloadBinaryItem: Loads and refreshes the defglobal */
/*   constructs used by this binary image.            */
/******************************************************/
static void BloadBinaryItem(
  void *theEnv,
  EXEC_STATUS)
  {
   size_t space;

   /*======================================================*/
   /* Read in the amount of space used by the binary image */
   /* (this is used to skip the construct in the event it  */
   /* is not available in the version being run).          */
   /*======================================================*/

   GenReadBinary(theEnv,execStatus,&space,sizeof(size_t));

   /*=============================================*/
   /* Read in the defglobalModule data structures */
   /* and refresh the pointers.                   */
   /*=============================================*/

   BloadandRefresh(theEnv,execStatus,DefglobalBinaryData(theEnv,execStatus)->NumberOfDefglobalModules,
                   sizeof(struct bsaveDefglobalModule),
                   UpdateDefglobalModule);

   /*=======================================*/
   /* Read in the defglobal data structures */
   /* and refresh the pointers.             */
   /*=======================================*/

   BloadandRefresh(theEnv,execStatus,DefglobalBinaryData(theEnv,execStatus)->NumberOfDefglobals,
                   sizeof(struct bsaveDefglobal),
                   UpdateDefglobal);
  }

/************************************************/
/* UpdateDefglobalModule: Bload refresh routine */
/*   for defglobal module data structures.      */
/************************************************/
static void UpdateDefglobalModule(
  void *theEnv,
  EXEC_STATUS,
  void *buf,
  long obji)
  {
   struct bsaveDefglobalModule *bdmPtr;

   bdmPtr = (struct bsaveDefglobalModule *) buf;

   UpdateDefmoduleItemHeader(theEnv,execStatus,&bdmPtr->header,&DefglobalBinaryData(theEnv,execStatus)->ModuleArray[obji].header,
                             (int) sizeof(struct defglobal),
                             (void *) DefglobalBinaryData(theEnv,execStatus)->DefglobalArray);
  }

/******************************************/
/* UpdateDefglobal: Bload refresh routine */
/*   for defglobal data structures.       */
/******************************************/
static void UpdateDefglobal(
  void *theEnv,
  EXEC_STATUS,
  void *buf,
  long obji)
  {
   struct bsaveDefglobal *bdp;

   bdp = (struct bsaveDefglobal *) buf;
   UpdateConstructHeader(theEnv,execStatus,&bdp->header,&DefglobalBinaryData(theEnv,execStatus)->DefglobalArray[obji].header,
                         (int) sizeof(struct defglobalModule),(void *) DefglobalBinaryData(theEnv,execStatus)->ModuleArray,
                         (int) sizeof(struct defglobal),(void *) DefglobalBinaryData(theEnv,execStatus)->DefglobalArray);

#if DEBUGGING_FUNCTIONS
   DefglobalBinaryData(theEnv,execStatus)->DefglobalArray[obji].watch = WatchGlobals;
#endif
   DefglobalBinaryData(theEnv,execStatus)->DefglobalArray[obji].initial = HashedExpressionPointer(bdp->initial);
   DefglobalBinaryData(theEnv,execStatus)->DefglobalArray[obji].current.type = RVOID;

  }

/***************************************/
/* ClearBload: Defglobal clear routine */
/*   when a binary load is in effect.  */
/***************************************/
static void ClearBload(
  void *theEnv,
  EXEC_STATUS)
  {
   long i;
   size_t space;

   /*=======================================================*/
   /* Decrement in use counters for atomic values contained */
   /* in the construct headers. Also decrement data         */
   /* structures used to store the defglobal's value.       */
   /*=======================================================*/

   for (i = 0; i < DefglobalBinaryData(theEnv,execStatus)->NumberOfDefglobals; i++)
     {
      UnmarkConstructHeader(theEnv,execStatus,&DefglobalBinaryData(theEnv,execStatus)->DefglobalArray[i].header);

      ValueDeinstall(theEnv,execStatus,&(DefglobalBinaryData(theEnv,execStatus)->DefglobalArray[i].current));
      if (DefglobalBinaryData(theEnv,execStatus)->DefglobalArray[i].current.type == MULTIFIELD)
        { ReturnMultifield(theEnv,execStatus,(struct multifield *) DefglobalBinaryData(theEnv,execStatus)->DefglobalArray[i].current.value); }
     }

   /*==============================================================*/
   /* Deallocate the space used for the defglobal data structures. */
   /*==============================================================*/

   space = DefglobalBinaryData(theEnv,execStatus)->NumberOfDefglobals * sizeof(struct defglobal);
   if (space != 0) genfree(theEnv,execStatus,(void *) DefglobalBinaryData(theEnv,execStatus)->DefglobalArray,space);
   DefglobalBinaryData(theEnv,execStatus)->NumberOfDefglobals = 0;
   
   /*=====================================================================*/
   /* Deallocate the space used for the defglobal module data structures. */
   /*=====================================================================*/

   space = DefglobalBinaryData(theEnv,execStatus)->NumberOfDefglobalModules * sizeof(struct defglobalModule);
   if (space != 0) genfree(theEnv,execStatus,(void *) DefglobalBinaryData(theEnv,execStatus)->ModuleArray,space);
   DefglobalBinaryData(theEnv,execStatus)->NumberOfDefglobalModules = 0;
  }

/********************************************************/
/* BloadDefglobalModuleReference: Returns the defglobal */
/*   module pointer for using with the bload function.  */
/********************************************************/
globle void *BloadDefglobalModuleReference(
  void *theEnv,
  EXEC_STATUS,
  int theIndex)
  {
   return ((void *) &DefglobalBinaryData(theEnv,execStatus)->ModuleArray[theIndex]);
  }

#endif /* DEFGLOBAL_CONSTRUCT && (BLOAD || BLOAD_AND_BSAVE || BLOAD_ONLY) && (! RUN_TIME) */



