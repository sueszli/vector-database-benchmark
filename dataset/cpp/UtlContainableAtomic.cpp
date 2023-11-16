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
#include "utl/UtlContainableAtomic.h"
#include "utl/Instrumentation.h"
#include <typeinfo>
// EXTERNAL FUNCTIONS
// EXTERNAL VARIABLES
// CONSTANTS
// STATIC VARIABLE INITIALIZATIONS

/* //////////////////////////// PUBLIC //////////////////////////////////// */

/* ============================ CREATORS ================================== */

// Constructor
UtlContainableAtomic::UtlContainableAtomic()
{
  system_tap_object_created((intptr_t)this, typeid(*this).name());
}
// Copy constructor

// Destructor

/* ============================ MANIPULATORS ============================== */

// Assignment operator

/* ============================ ACCESSORS ================================= */

/* ============================ INQUIRY =================================== */

UtlContainableAtomic::~UtlContainableAtomic()
{
  system_tap_object_destroyed((intptr_t)this, typeid(*this).name());
}
/// Get hash of object.
unsigned UtlContainableAtomic::hash() const
{
   return UtlContainable::directHash();
}

/// Compare this object to another object.
int UtlContainableAtomic::compareTo(UtlContainable const* other) const
{
   // C++ permits the consistent comparison of all pointers.
   int r =
      this > other ? 1 :
      this < other ? -1 :
      0;
   return r;
}

/* //////////////////////////// PROTECTED ///////////////////////////////// */

/* //////////////////////////// PRIVATE /////////////////////////////////// */

/* ============================ FUNCTIONS ================================= */
