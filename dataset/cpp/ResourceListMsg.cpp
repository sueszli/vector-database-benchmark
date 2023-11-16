//
// Copyright (C) 2007 Pingtel Corp., certain elements licensed under a Contributor Agreement.
// Contributors retain copyright to elements licensed under a Contributor Agreement.
// Licensed to the User under the LGPL license.
//
//
// $$
////////////////////////////////////////////////////////////////////////
//////


// SYSTEM INCLUDES
// APPLICATION INCLUDES

#include "ResourceListMsg.h"
#include <os/OsLogger.h>

// EXTERNAL FUNCTIONS
// EXTERNAL VARIABLES
// CONSTANTS
// STATIC VARIABLE INITIALIZATIONS


/* //////////////////////////// PUBLIC //////////////////////////////////// */

/* ============================ CREATORS ================================== */

// Constructor
SubscriptionCallbackMsg::SubscriptionCallbackMsg(const char* earlyDialogHandle,
                                                 const char* dialogHandle,
                                                 SipSubscribeClient:: SubscriptionState newState,
                                                 const char* subscriptionState) :
   OsMsg(RLS_SUBSCRIPTION_MSG, 0),
   mEarlyDialogHandle(earlyDialogHandle),
   mDialogHandle(dialogHandle),
   mNewState(newState),
   mSubscriptionState(subscriptionState)
{
   Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                 "SubscriptionCallbackMsg:: earlyDialogHandle = '%s', dialogHandle = '%s', newState = %d, subscriptionState = '%s'",
                 earlyDialogHandle, dialogHandle, newState, subscriptionState);
}

// Copy constructor
SubscriptionCallbackMsg::SubscriptionCallbackMsg(const SubscriptionCallbackMsg& rSubscriptionCallbackMsg)
:  OsMsg(rSubscriptionCallbackMsg)
{
   mEarlyDialogHandle = rSubscriptionCallbackMsg.mEarlyDialogHandle;
   mDialogHandle      = rSubscriptionCallbackMsg.mDialogHandle;
   mNewState          = rSubscriptionCallbackMsg.mNewState;
   mSubscriptionState = rSubscriptionCallbackMsg.mSubscriptionState;
}

// Create a copy of this msg object (which may be of a derived type)
OsMsg* SubscriptionCallbackMsg::createCopy(void) const
{
   return new SubscriptionCallbackMsg(*this);
}

// Destructor
SubscriptionCallbackMsg::~SubscriptionCallbackMsg()
{
   // no work required
}

/* ============================ MANIPULATORS ============================== */

// Assignment operator
SubscriptionCallbackMsg&
SubscriptionCallbackMsg::operator=(const SubscriptionCallbackMsg& rhs)
{
   if (this == &rhs)            // handle the assignment to self case
      return *this;

   OsMsg::operator=(rhs);       // assign fields for parent class

   mEarlyDialogHandle = rhs.mEarlyDialogHandle;
   mDialogHandle      = rhs.mDialogHandle;
   mNewState          = rhs.mNewState;
   mSubscriptionState = rhs.mSubscriptionState;

   return *this;
}

/* ============================ ACCESSORS ================================= */

// Return the size of the message in bytes.
// This is a virtual method so that it will return the accurate size for
// the message object even if that object has been upcast to the type of
// an ancestor class.
int SubscriptionCallbackMsg::getMsgSize(void) const
{
   return sizeof(*this);
}

// Return pointer to mEarlyDialogHandle.
const UtlString* SubscriptionCallbackMsg::getEarlyDialogHandle() const
{
   return &mEarlyDialogHandle;
}

// Return pointer to mDialogHandle.
const UtlString* SubscriptionCallbackMsg::getDialogHandle() const
{
   return &mDialogHandle;
}

// Return the newState.
SipSubscribeClient::SubscriptionState SubscriptionCallbackMsg::getNewState() const
{
   return mNewState;
}

// Return pointer to mSubscriptionState.
const UtlString* SubscriptionCallbackMsg::getSubscriptionState() const
{
   return &mSubscriptionState;
}

/* ============================ INQUIRY =================================== */

/* //////////////////////////// PROTECTED ///////////////////////////////// */

/* //////////////////////////// PRIVATE /////////////////////////////////// */

/* ============================ FUNCTIONS ================================= */


/* //////////////////////////// PUBLIC //////////////////////////////////// */

/* ============================ CREATORS ================================== */

// Constructor
NotifyCallbackMsg::NotifyCallbackMsg(const char* dialogHandle,
                                     const char* content_bytes,
                                     int content_length) :
   OsMsg(RLS_NOTIFY_MSG, 0),
   mDialogHandle(dialogHandle),
   mContent(content_bytes, content_length)
{
   Os::Logger::instance().log(FAC_RLS, PRI_DEBUG,
                 "NotifyCallbackMsg:: dialogHandle = '%s', content = '%.*s'",
                 dialogHandle, content_length, content_bytes);
}

// Copy constructor
NotifyCallbackMsg::NotifyCallbackMsg(const NotifyCallbackMsg& rNotifyCallbackMsg)
:  OsMsg(rNotifyCallbackMsg)
{
   mDialogHandle = rNotifyCallbackMsg.mDialogHandle;
   mContent      = rNotifyCallbackMsg.mContent;
}

// Create a copy of this msg object (which may be of a derived type)
OsMsg* NotifyCallbackMsg::createCopy(void) const
{
   return new NotifyCallbackMsg(*this);
}

// Destructor
NotifyCallbackMsg::~NotifyCallbackMsg()
{
   // no work required
}

/* ============================ MANIPULATORS ============================== */

// Assignment operator
NotifyCallbackMsg&
NotifyCallbackMsg::operator=(const NotifyCallbackMsg& rhs)
{
   if (this == &rhs)            // handle the assignment to self case
      return *this;

   OsMsg::operator=(rhs);       // assign fields for parent class

   mDialogHandle = rhs.mDialogHandle;
   mContent      = rhs.mContent;

   return *this;
}

/* ============================ ACCESSORS ================================= */

// Return the size of the message in bytes.
// This is a virtual method so that it will return the accurate size for
// the message object even if that object has been upcast to the type of
// an ancestor class.
int NotifyCallbackMsg::getMsgSize(void) const
{
   return sizeof(*this);
}

// Return pointer to the mDialogHandle.
const UtlString* NotifyCallbackMsg::getDialogHandle() const
{
   return &mDialogHandle;
}

// Return pointer to the mContent.
const UtlString* NotifyCallbackMsg::getContent() const
{
   return &mContent;
}

/* ============================ INQUIRY =================================== */

/* //////////////////////////// PROTECTED ///////////////////////////////// */

/* //////////////////////////// PRIVATE /////////////////////////////////// */

/* ============================ FUNCTIONS ================================= */

// Constructor
SubscriptionSetMsg::SubscriptionSetMsg(
        UtlContainable *handler,
        const UtlString& callidContact) :
   OsMsg(RLS_SUBSCRIPTION_SET_MSG, 0),
   _handler(handler),
   _callidContact(callidContact)
{
}

// Copy constructor
SubscriptionSetMsg::SubscriptionSetMsg(
        const SubscriptionSetMsg& rhs)
:  OsMsg(rhs)
{
   _handler       = rhs._handler;
   _callidContact = rhs._callidContact;
}

// Create a copy of this msg object (which may be of a derived type)
OsMsg* SubscriptionSetMsg::createCopy(void) const
{
   return new SubscriptionSetMsg(*this);
}

// Destructor
SubscriptionSetMsg::~SubscriptionSetMsg()
{
   // no work required
}

/* ============================ MANIPULATORS ============================== */

// Assignment operator
SubscriptionSetMsg&
SubscriptionSetMsg::operator=(const SubscriptionSetMsg& rhs)
{
   if (this == &rhs)            // handle the assignment to self case
      return *this;

   OsMsg::operator=(rhs);       // assign fields for parent class

   _handler       = rhs._handler;
   _callidContact = rhs._callidContact;

   return *this;
}

/* ============================ ACCESSORS ================================= */

// Return the size of the message in bytes.
// This is a virtual method so that it will return the accurate size for
// the message object even if that object has been upcast to the type of
// an ancestor class.
int SubscriptionSetMsg::getMsgSize(void) const
{
   return sizeof(*this);
}

// Return the newState.
UtlContainable* SubscriptionSetMsg::getHandler() const
{
   return _handler;
}

// Return pointer to mSubscriptionState.
const UtlString* SubscriptionSetMsg::getCallidContact() const
{
   return &_callidContact;
}
