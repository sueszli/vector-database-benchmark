   /*******************************************************/
   /*      "C" Language Integrated Production System      */
   /*                                                     */
   /*             CLIPS Version 6.30  10/19/06            */
   /*                                                     */
   /*                ENVIRONMENT MODULE                   */
   /*******************************************************/

/*************************************************************/
/* Purpose: Routines for supporting multiple environments.   */
/*                                                           */
/* Principal Programmer(s):                                  */
/*      Gary D. Riley                                        */
/*                                                           */
/* Revision History:                                         */
/*                                                           */
/*      6.24: Added code to CreateEnvironment to free        */
/*            already allocated data if one of the malloc    */
/*            calls fail.                                    */
/*                                                           */
/*            Modified AllocateEnvironmentData to print a    */
/*            message if it was unable to allocate memory.   */
/*                                                           */
/*            Renamed BOOLEAN macro type to intBool.         */
/*                                                           */
/*            Added CreateRuntimeEnvironment function.       */
/*                                                           */
/*            Added support for context information when an  */
/*            environment is created (i.e a pointer from the */
/*            CLIPS environment to its parent environment).  */
/*                                                           */
/*      6.30: Added support for passing context information  */ 
/*            to user defined functions and callback         */
/*            functions.                                     */
/*                                                           */
/*************************************************************/

#define _ENVRNMNT_SOURCE_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "setup.h"

#include "memalloc.h"
#include "prntutil.h"
#include "router.h"
#include "engine.h"
#include "sysdep.h"
#include "utility.h"

#include "envrnmnt.h"

#define SIZE_ENVIRONMENT_HASH  131

/***************************************/
/* LOCAL INTERNAL FUNCTION DEFINITIONS */
/***************************************/

#if ALLOW_ENVIRONMENT_GLOBALS
   static void                    AddHashedEnvironment(struct environmentData *);
   static struct environmentData *FindEnvironment(unsigned long);
   static intBool                 RemoveHashedEnvironment(struct environmentData *, EXEC_STATUS);
   static void                    InitializeEnvironmentHashTable(void);
#endif
   static void                    RemoveEnvironmentCleanupFunctions(struct environmentData *, EXEC_STATUS);
   static void                   *CreateEnvironmentDriver(EXEC_STATUS,struct symbolHashNode **,struct floatHashNode **,
                                                          struct integerHashNode **,struct bitMapHashNode **,
                                                          struct externalAddressHashNode **);

/***************************************/
/* LOCAL INTERNAL VARIABLE DEFINITIONS */
/***************************************/

#if ALLOW_ENVIRONMENT_GLOBALS
   static unsigned long              NextEnvironmentIndex   = 0;
   static struct environmentData   **EnvironmentHashTable   = NULL;
   static struct environmentData    *CurrentEnvironment     = NULL;
   static struct executionStatus    *CurrentExecutionStatus = NULL;
#endif

/*******************************************************/
/* AllocateEnvironmentData: Allocates environment data */
/*    for the specified environment data record.       */
/*******************************************************/
globle intBool AllocateEnvironmentData(
  void *vtheEnvironment,
  EXEC_STATUS,
  unsigned int position,
  unsigned long size,
  void (*cleanupFunction)(void *,EXEC_STATUS))
  {      
   struct environmentData *theEnvironment = (struct environmentData *) vtheEnvironment;

   /*===========================================*/
   /* Environment data can't be of length zero. */
   /*===========================================*/
   
   if (size <= 0)
     {
      printf("\n[ENVRNMNT1] Environment data position %d allocated with size of 0 or less.\n",position);      
      return(FALSE);
     }
     
   /*================================================================*/
   /* Check to see if the data position exceeds the maximum allowed. */
   /*================================================================*/
   
   if (position >= MAXIMUM_ENVIRONMENT_POSITIONS)
     {
      printf("\n[ENVRNMNT2] Environment data position %d exceeds the maximum allowed.\n",position);      
      return(FALSE);
     }
     
   /*============================================================*/
   /* Check if the environment data has already been registered. */
   /*============================================================*/
   
   if (theEnvironment->theData[position] != NULL)
     {
      printf("\n[ENVRNMNT3] Environment data position %d already allocated.\n",position);      
      return(FALSE);
     }
     
   /*====================*/
   /* Allocate the data. */
   /*====================*/
   
   theEnvironment->theData[position] = malloc(size);
   if (theEnvironment->theData[position] == NULL)
     {
      printf("\n[ENVRNMNT4] Environment data position %d could not be allocated.\n",position);      
      return(FALSE);
     }
   
   memset(theEnvironment->theData[position],0,size);
   
   /*=============================*/
   /* Store the cleanup function. */
   /*=============================*/
   
   theEnvironment->cleanupFunctions[position] = cleanupFunction;
   
   /*===============================*/
   /* Data successfully registered. */
   /*===============================*/
   
   return(TRUE);
  }

/***************************************************************/
/* DeallocateEnvironmentData: Deallocates all environments     */
/*   stored in the environment hash table and then deallocates */
/*   the environment hash table.                               */
/***************************************************************/
globle intBool DeallocateEnvironmentData()
  {
#if ALLOW_ENVIRONMENT_GLOBALS
   struct environmentData *theEnvironment, *nextEnvironment;
	 // Lode: TODO add EXEC_STATUS?
   int i, rv = TRUE;
   
   for (i = 0; i < SIZE_ENVIRONMENT_HASH; i++)
     {
      for (theEnvironment = EnvironmentHashTable[i];
           theEnvironment != NULL;
          )
        {
         nextEnvironment = theEnvironment->next;
         
	     #warning Lode: Stefan wanted me to add this warning - passing NULL pointer!
         if (! DestroyEnvironment(theEnvironment,NULL))
           { rv = FALSE; }
         
         theEnvironment = nextEnvironment;
        }
     }

   free(EnvironmentHashTable);
   
   return(rv);
#else
   return(FALSE);
#endif
  }

#if ALLOW_ENVIRONMENT_GLOBALS
/*********************************************************/
/* InitializeEnvironmentHashTable: Initializes the table */
/*   entries in the environment hash table to NULL.      */
/*********************************************************/
static void InitializeEnvironmentHashTable()
  {
   int i;
    
   if (EnvironmentHashTable != NULL)
     { return; }

   EnvironmentHashTable = (struct environmentData **)
                          malloc(sizeof (struct environmentData *) * SIZE_ENVIRONMENT_HASH);

   if (EnvironmentHashTable == NULL)
     {
      printf("\n[ENVRNMNT4] Unable to initialize environment hash table.\n");      
      return;
     }

   for (i = 0; i < SIZE_ENVIRONMENT_HASH; i++) EnvironmentHashTable[i] = NULL;
  }

/*********************************************/
/* AddHashedEnvironment: Adds an environment */
/*    entry to the environment hash table.   */
/*********************************************/
static void AddHashedEnvironment(
  struct environmentData *theEnvironment)
  {

   struct environmentData *temp;
   unsigned long hashValue;
   
   if (EnvironmentHashTable == NULL)
     { InitializeEnvironmentHashTable(); }
     
   hashValue = theEnvironment->environmentIndex % SIZE_ENVIRONMENT_HASH;

   temp = EnvironmentHashTable[hashValue];
   EnvironmentHashTable[hashValue] = theEnvironment;
   theEnvironment->next = temp;
  }
  
/***************************************************/
/* RemoveHashedEnvironment: Removes an environment */
/*   entry from the environment hash table.        */
/***************************************************/
static intBool RemoveHashedEnvironment(
  struct environmentData *theEnvironment,
  EXEC_STATUS)
  {

   unsigned long hashValue;
   struct environmentData *hptr, *prev;

   hashValue = theEnvironment->environmentIndex % SIZE_ENVIRONMENT_HASH;

   for (hptr = EnvironmentHashTable[hashValue], prev = NULL;
        hptr != NULL;
        hptr = hptr->next)
     {
      if (hptr == theEnvironment)
        {
         if (prev == NULL)
           {
            EnvironmentHashTable[hashValue] = hptr->next;
            return(TRUE);
           }
         else
           {
            prev->next = hptr->next;
            return(TRUE);
           }
        }
      prev = hptr;
     }

   return(FALSE);
  }

/**********************************************************/
/* FindEnvironment: Determines if a specified environment */
/*   index has an entry in the environment hash table.    */
/**********************************************************/
static struct environmentData *FindEnvironment(
  unsigned long environmentIndex)
  {
   struct environmentData *theEnvironment;
	 // Lode: TODO add EXEC_STATUS?
   unsigned long hashValue;
   
   hashValue = environmentIndex % SIZE_ENVIRONMENT_HASH;
   
   for (theEnvironment = EnvironmentHashTable[hashValue];
        theEnvironment != NULL;
        theEnvironment = theEnvironment->next)
     {
      if (theEnvironment->environmentIndex == environmentIndex)
        { return(theEnvironment); }
     }

   return(NULL);
  }
#endif

/************************************************************/
/* CreateEnvironment: Creates an environment data structure */
/*   and initializes its content to zero/null.              */
/************************************************************/
globle void *CreateEnvironment(EXEC_STATUS)
  {
   return CreateEnvironmentDriver(execStatus,NULL,NULL,NULL,NULL,NULL);
  }

/*********************************************************************/
/* CreateExecutionStatus: Creates an execution status data structure */
/*   and initializes its content to zero/null.                       */
/*********************************************************************/
globle struct executionStatus *CreateExecutionStatus()
{
  struct executionStatus *result = (struct executionStatus *)malloc(sizeof(struct executionStatus));
  
  result->HaltExecution          = FALSE;
  result->RunningInParallel      = FALSE;
  result->EvaluationError        = FALSE;
  result->CurrentExpression      = NULL;
  result->CurrentEvaluationDepth = 0;
  result->DepthInReteNetwork     = 0;
  
	return result;
}
/**********************************************************/
/* CreateRuntimeEnvironment: Creates an environment data  */
/*   structure and initializes its content to zero/null.  */
/**********************************************************/
globle void *CreateRuntimeEnvironment(
  EXEC_STATUS,
  struct symbolHashNode **symbolTable,
  struct floatHashNode **floatTable,
  struct integerHashNode **integerTable,
  struct bitMapHashNode **bitmapTable)
  {
   return CreateEnvironmentDriver(execStatus,symbolTable,floatTable,integerTable,bitmapTable,NULL);
  }
  
/*********************************************************/
/* CreateEnvironmentDriver: Creates an environment data  */
/*   structure and initializes its content to zero/null. */
/*********************************************************/
globle void *CreateEnvironmentDriver(
  EXEC_STATUS,
  struct symbolHashNode **symbolTable,
  struct floatHashNode **floatTable,
  struct integerHashNode **integerTable,
  struct bitMapHashNode **bitmapTable,
  struct externalAddressHashNode **externalAddressTable)
  {
   struct environmentData *theEnvironment;
	  
   void *theData;
   
   theEnvironment = (struct environmentData *) malloc(sizeof(struct environmentData));
  
   if (theEnvironment == NULL)
     {
      printf("\n[ENVRNMNT5] Unable to create new environment.\n");
      return(NULL);
     }

   theData = malloc(sizeof(void *) * MAXIMUM_ENVIRONMENT_POSITIONS);
   
   if (theData == NULL)
     {
      free(theEnvironment);
      printf("\n[ENVRNMNT6] Unable to create environment data.\n");
      return(NULL);
     }

   memset(theData,0,sizeof(void *) * MAXIMUM_ENVIRONMENT_POSITIONS);

   theEnvironment->initialized = FALSE;
   theEnvironment->theData = (void **) theData;
   theEnvironment->next = NULL;
   theEnvironment->listOfCleanupEnvironmentFunctions = NULL;
#if ALLOW_ENVIRONMENT_GLOBALS
   theEnvironment->environmentIndex = NextEnvironmentIndex++;
#else
   theEnvironment->environmentIndex = 0;
#endif
   theEnvironment->context = NULL;
   theEnvironment->routerContext = NULL;
   theEnvironment->functionContext = NULL;
   theEnvironment->callbackContext = NULL;
    
    theEnvironment->memoryPool = NULL;
    theEnvironment->matcherThreadPool = NULL;
    theEnvironment->factThreadPool = NULL;
    
    // STEFAN: initalize the thread-pool related data structures
    apr_status_t rv;
    
    apr_initialize();
    
    rv = apr_pool_create(&theEnvironment->memoryPool, NULL);
    if (rv) {
      printf("\n[ENVRNMNT_THREAD] Unable to create memory pool.\n");
      return(NULL);
    }
    
    rv = apr_thread_pool_create(&theEnvironment->matcherThreadPool, 
                                1, 1, theEnvironment->memoryPool);
    if (rv) {
      printf("\n[ENVRNMNT_THREAD] Unable to create thread pool.\n");
      return(NULL);
    }
    
    rv = apr_thread_pool_create(&theEnvironment->factThreadPool,
                                1, 1, theEnvironment->memoryPool);
    if (rv) {
      printf("\n[ENVRNMNT_THREAD] Unable to create thread pool.\n");
      return(NULL);
    }
    
	  
	rv = apr_thread_rwlock_create(&theEnvironment->factHashLock, theEnvironment->memoryPool);
	if (rv) {
	  printf("\n[ENVRNMNT_THREAD] Unable to create thread read/write lock.\n");
	  return(NULL);
	}
	  

   /*=============================================*/
   /* Allocate storage for the cleanup functions. */
   /*=============================================*/

   theData = malloc(sizeof(void (*)(struct environmentData *)) * MAXIMUM_ENVIRONMENT_POSITIONS);
   
   if (theData == NULL)
     {
      free(theEnvironment->theData);
      free(theEnvironment);
      printf("\n[ENVRNMNT7] Unable to create environment data.\n");
      return(NULL);
     }

   memset(theData,0,sizeof(void (*)(struct environmentData *)) * MAXIMUM_ENVIRONMENT_POSITIONS);
   theEnvironment->cleanupFunctions = (void (**)(void *,EXEC_STATUS))theData;

#if ALLOW_ENVIRONMENT_GLOBALS
   AddHashedEnvironment(theEnvironment);
   CurrentEnvironment = theEnvironment;
#endif

   EnvInitializeEnvironment(theEnvironment,execStatus,symbolTable,floatTable,integerTable,bitmapTable,externalAddressTable);

   return(theEnvironment);
  }

#if ALLOW_ENVIRONMENT_GLOBALS
/*******************************************/
/* SetCurrentEnvironment: Sets the current */
/*   environment to the one specified.     */
/*******************************************/
globle void SetCurrentEnvironment(
  void *theEnvironment)
  {
   CurrentEnvironment = (struct environmentData *) theEnvironment;
  }
  
/**************************************************/
/* SetCurrentEnvironmentByIndex: Sets the current */
/*   environment to the one having the specified  */
/*   environment index.                           */
/**************************************************/
globle intBool SetCurrentEnvironmentByIndex(
  unsigned long environmentIndex)
  {
   struct environmentData *theEnvironment;
	 // Lode: TODO add EXEC_STATUS?

   theEnvironment = FindEnvironment(environmentIndex);
   
   if (theEnvironment == NULL)
     { return(FALSE); }
     
   SetCurrentEnvironment(theEnvironment);
   
   return(TRUE);
  }     
   
/**************************************************/
/* GetEnvironmentByIndex: Returns the environment */
/*   having the specified environment index.      */
/**************************************************/
globle void *GetEnvironmentByIndex(
  unsigned long environmentIndex)
  {
   struct environmentData *theEnvironment;
	 // Lode: TODO add EXEC_STATUS?

   theEnvironment = FindEnvironment(environmentIndex);
      
   return(theEnvironment);
  }     
   
/********************************************/
/* GetCurrentEnvironment: Returns a pointer */
/*   to the current environment.            */
/********************************************/
globle void *GetCurrentEnvironment()
  {
   return(CurrentEnvironment);
  }  

globle struct executionStatus *GetCurrentExecutionStatus() 
  {
   return(CurrentExecutionStatus);
  }

/******************************************/
/* GetEnvironmentIndex: Returns the index */
/*   of the specified environment.        */
/******************************************/
globle unsigned long GetEnvironmentIndex(
  void *theEnvironment,
  EXEC_STATUS)
  {
   return(((struct environmentData *) theEnvironment)->environmentIndex);
  } 
  
#endif

/**********************************************/
/* GetEnvironmentContext: Returns the context */
/*   of the specified environment.            */
/**********************************************/
globle void *GetEnvironmentContext(
  void *theEnvironment)
  {
   return(((struct environmentData *) theEnvironment)->context);
  } 

/*******************************************/
/* SetEnvironmentContext: Sets the context */
/*   of the specified environment.         */
/*******************************************/
globle void *SetEnvironmentContext(
  void *theEnvironment,
  EXEC_STATUS,
  void *theContext)
  {
   void *oldContext;
   
   oldContext = ((struct environmentData *) theEnvironment)->context;
  
   ((struct environmentData *) theEnvironment)->context = theContext;
   
   return oldContext;
  } 

/***************************************************/
/* GetEnvironmentRouterContext: Returns the router */
/*   context of the specified environment.         */
/***************************************************/
globle void *GetEnvironmentRouterContext(
  void *theEnvironment,
  EXEC_STATUS)
  {
   return(((struct environmentData *) theEnvironment)->routerContext);
  } 

/************************************************/
/* SetEnvironmentRouterContext: Sets the router */
/*   context of the specified environment.      */
/************************************************/
globle void *SetEnvironmentRouterContext(
  void *theEnvironment,
  EXEC_STATUS,
  void *theRouterContext)
  {
   void *oldRouterContext;
   
   oldRouterContext = ((struct environmentData *) theEnvironment)->routerContext;
  
   ((struct environmentData *) theEnvironment)->routerContext = theRouterContext;
   
   return oldRouterContext;
  } 

/*******************************************************/
/* GetEnvironmentFunctionContext: Returns the function */
/*   context of the specified environment.             */
/*******************************************************/
globle void *GetEnvironmentFunctionContext(
  void *theEnvironment,
  EXEC_STATUS)
  {
	 // Lode: TODO add EXEC_STATUS?
   return(((struct environmentData *) theEnvironment)->functionContext);
  } 

/**************************************************/
/* SetEnvironmentFunctionContext: Sets the router */
/*   context of the specified environment.        */
/**************************************************/
globle void *SetEnvironmentFunctionContext(
  void *theEnvironment,
  EXEC_STATUS,
  void *theFunctionContext)
  {
   void *oldFunctionContext;
   
   oldFunctionContext = ((struct environmentData *) theEnvironment)->functionContext;
  
   ((struct environmentData *) theEnvironment)->functionContext = theFunctionContext;
   
   return oldFunctionContext;
  } 

/*******************************************************/
/* GetEnvironmentCallbackContext: Returns the callback */
/*   context of the specified environment.             */
/*******************************************************/
globle void *GetEnvironmentCallbackContext(
  void *theEnvironment)
  {
   return(((struct environmentData *) theEnvironment)->callbackContext);
  } 

/****************************************************/
/* SetEnvironmentCallbackContext: Sets the callback */
/*   context of the specified environment.          */
/****************************************************/
globle void *SetEnvironmentCallbackContext(
  void *theEnvironment,
  EXEC_STATUS,
  void *theCallbackContext)
  {
   void *oldCallbackContext;
   
   oldCallbackContext = ((struct environmentData *) theEnvironment)->callbackContext;
  
   ((struct environmentData *) theEnvironment)->callbackContext = theCallbackContext;
   
   return oldCallbackContext;
  } 
  
/**********************************************/
/* DestroyEnvironment: Destroys the specified */
/*   environment returning all of its memory. */
/**********************************************/
globle intBool DestroyEnvironment(
  void *vtheEnvironment,
  EXEC_STATUS)
  {   
   struct environmentCleanupFunction *cleanupPtr;
   int i;
   struct memoryData *theMemData;
   intBool rv = TRUE;
   struct environmentData *theEnvironment = (struct environmentData *) vtheEnvironment;
   /*
   if (EvaluationData(theEnvironment)->CurrentExpression != NULL)
     { return(FALSE); }
     
#if DEFRULE_CONSTRUCT
   if (EngineData(theEnvironment)->ExecutingRule != NULL)
     { return(FALSE); }
#endif
*/
   theMemData = MemoryData(theEnvironment,execStatus);

   EnvReleaseMem(theEnvironment,execStatus,-1,FALSE);

   for (i = 0; i < MAXIMUM_ENVIRONMENT_POSITIONS; i++)
     {
      if (theEnvironment->cleanupFunctions[i] != NULL)
        { (*theEnvironment->cleanupFunctions[i])(theEnvironment,execStatus); }
     }
     
   free(theEnvironment->cleanupFunctions);
     
   for (cleanupPtr = theEnvironment->listOfCleanupEnvironmentFunctions;
        cleanupPtr != NULL;
        cleanupPtr = cleanupPtr->next)
     { (*cleanupPtr->func)(theEnvironment,execStatus); }

   RemoveEnvironmentCleanupFunctions(theEnvironment,execStatus);
   
   EnvReleaseMem(theEnvironment,execStatus,-1,FALSE);

#if ALLOW_ENVIRONMENT_GLOBALS
   RemoveHashedEnvironment(theEnvironment,execStatus);
#endif
     
   if ((theMemData->MemoryAmount != 0) || (theMemData->MemoryCalls != 0))
     {
      printf("\n[ENVRNMNT8] Environment data not fully deallocated.\n"); 
      printf("\n[ENVRNMNT8] MemoryAmount = %ld.\n",(long) theMemData->MemoryAmount); 
      printf("\n[ENVRNMNT8] MemoryCalls = %ld.\n",(long) theMemData->MemoryCalls); 
      rv = FALSE;     
     }
     
   free(theMemData->MemoryTable);

#if BLOCK_MEMORY
   ReturnAllBlocks(theEnvironment);
#endif
         
   for (i = 0; i < MAXIMUM_ENVIRONMENT_POSITIONS; i++)
     {
      if (theEnvironment->theData[i] != NULL)
        {
         free(theEnvironment->theData[i]);
         theEnvironment->theData[i] = NULL;
        }
     }
     
   free(theEnvironment->theData);
   
#if ALLOW_ENVIRONMENT_GLOBALS
   if (CurrentEnvironment == theEnvironment)
     { CurrentEnvironment = NULL; }
#endif

   free(theEnvironment);
   
   return(rv);
  } 
 
/**************************************************/
/* AddEnvironmentCleanupFunction: Adds a function */
/*   to the ListOfCleanupEnvironmentFunctions.    */
/**************************************************/
globle intBool AddEnvironmentCleanupFunction(
  void *vtheEnv,
  EXEC_STATUS,
  char *name,
  void (*functionPtr)(void *,EXEC_STATUS),
  int priority)
  {
   struct environmentCleanupFunction *newPtr, *currentPtr, *lastPtr = NULL;
   struct environmentData *theEnv = (struct environmentData *) vtheEnv;
     
   newPtr = (struct environmentCleanupFunction *) malloc(sizeof(struct environmentCleanupFunction));
   if (newPtr == NULL)
     { return(FALSE); }

   newPtr->name = name;
   newPtr->func = functionPtr;
   newPtr->priority = priority;

   if (theEnv->listOfCleanupEnvironmentFunctions == NULL)
     {
      newPtr->next = NULL;
      theEnv->listOfCleanupEnvironmentFunctions = newPtr;
      return(TRUE);
     }

   currentPtr = theEnv->listOfCleanupEnvironmentFunctions;
   while ((currentPtr != NULL) ? (priority < currentPtr->priority) : FALSE)
     {
      lastPtr = currentPtr;
      currentPtr = currentPtr->next;
     }

   if (lastPtr == NULL)
     {
      newPtr->next = theEnv->listOfCleanupEnvironmentFunctions;
      theEnv->listOfCleanupEnvironmentFunctions = newPtr;
     }
   else
     {
      newPtr->next = currentPtr;
      lastPtr->next = newPtr;
     }
     
   return(TRUE);
  }

/**************************************************/
/* RemoveEnvironmentCleanupFunctions: Removes the */
/*   list of environment cleanup functions.       */
/**************************************************/
static void RemoveEnvironmentCleanupFunctions(
  struct environmentData *theEnv,
  EXEC_STATUS)
  {   
   struct environmentCleanupFunction *nextPtr;
      
   while (theEnv->listOfCleanupEnvironmentFunctions != NULL)
     { 
      nextPtr = theEnv->listOfCleanupEnvironmentFunctions->next;
      free(theEnv->listOfCleanupEnvironmentFunctions);
      theEnv->listOfCleanupEnvironmentFunctions = nextPtr;
     }
  } 
