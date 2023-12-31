/* lockw3.c: RECURSIVE LOCKS IN WIN32
 *
 * $Id$
 * Copyright (c) 2001-2020 Ravenbrook Limited.  See end of file for license.
 *
 * .design: These are implemented using critical sections.
 *  See the section titled "Synchronization functions" in the Groups
 *  chapter of the Microsoft Win32 API Programmer's Reference.
 *  The "Synchronization" section of the Overview is also relevant.
 *
 *  Critical sections support recursive locking, so the implementation
 *  could be trivial.  This implementation counts the claims to provide
 *  extra checking.
 *
 *  The limit on the number of recursive claims is the max of
 *  ULONG_MAX and the limit imposed by critical sections, which
 *  is believed to be about UCHAR_MAX.
 *
 *  During use the claims field is updated to remember the number of
 *  claims acquired on a lock.  This field must only be modified
 *  while we are inside the critical section.
 */

#include "mpm.h"

#if !defined(MPS_OS_W3)
#error "lockw3.c is specific to MPS_OS_W3"
#endif

#include "mpswin.h"

SRCID(lockw3, "$Id$");

#if defined(LOCK)

/* .lock.win32: Win32 lock structure; uses CRITICAL_SECTION */
typedef struct LockStruct {
  Sig sig;                      /* design.mps.sig.field */
  unsigned long claims;         /* # claims held by the owning thread */
  CRITICAL_SECTION cs;          /* Win32's recursive lock thing */
} LockStruct;


size_t (LockSize)(void)
{
  return sizeof(LockStruct);
}

Bool (LockCheck)(Lock lock)
{
  CHECKS(Lock, lock);
  return TRUE;
}

void (LockInit)(Lock lock)
{
  AVER(lock != NULL);
  lock->claims = 0;
  InitializeCriticalSection(&lock->cs);
  lock->sig = LockSig;
  AVERT(Lock, lock);
}

void (LockFinish)(Lock lock)
{
  AVERT(Lock, lock);
  /* Lock should not be finished while held */
  AVER(lock->claims == 0);
  DeleteCriticalSection(&lock->cs);
  lock->sig = SigInvalid;
}

void (LockClaim)(Lock lock)
{
  AVERT(Lock, lock);
  EnterCriticalSection(&lock->cs);
  /* This should be the first claim.  Now we are inside the
   * critical section it is ok to check this. */
  AVER(lock->claims == 0); /* <design/check/#.common> */
  lock->claims = 1;
}

void (LockRelease)(Lock lock)
{
  AVERT(Lock, lock);
  AVER(lock->claims == 1);  /* The lock should only be held once */
  lock->claims = 0;  /* Must set this before leaving CS */
  LeaveCriticalSection(&lock->cs);
}

void (LockClaimRecursive)(Lock lock)
{
  AVERT(Lock, lock);
  EnterCriticalSection(&lock->cs);
  ++lock->claims;
  AVER(lock->claims > 0);
}

void (LockReleaseRecursive)(Lock lock)
{
  AVERT(Lock, lock);
  AVER(lock->claims > 0);
  --lock->claims;
  LeaveCriticalSection(&lock->cs);
}

Bool (LockIsHeld)(Lock lock)
{
  if (TryEnterCriticalSection(&lock->cs)) {
    Bool claimed = lock->claims > 0;
    LeaveCriticalSection(&lock->cs);
    return claimed;
  }
  return TRUE;
}


/* Global locking is performed by normal locks.
 * A separate lock structure is used for recursive and
 * non-recursive locks so that each may be differently ordered
 * with respect to client-allocated locks.
 */

static LockStruct globalLockStruct;
static LockStruct globalRecLockStruct;
static Lock globalLock = &globalLockStruct;
static Lock globalRecLock = &globalRecLockStruct;
static Bool globalLockInit = FALSE; /* TRUE iff initialized */

void LockInitGlobal(void)
{
  globalLock->claims = 0;
  LockInit(globalLock);
  globalRecLock->claims = 0;
  LockInit(globalRecLock);
  globalLockInit = TRUE;
}

/* lockEnsureGlobalLock -- one-time initialization of global locks
 *
 * InitOnceExecuteOnce ensures that only one thread can be running the
 * callback at a time, which allows to safely check globalLockInit. See
 * <https://docs.microsoft.com/en-us/windows/desktop/api/synchapi/nf-synchapi-initonceexecuteonce>
 * but note that at time of writing (2018-06-27) the documentation has
 * the arguments the wrong way round (parameter comes before context).
 */

static BOOL CALLBACK lockEnsureGlobalLockCallback(INIT_ONCE *init_once, void *parameter, void **context)
{
  UNUSED(init_once);
  AVER(parameter == UNUSED_POINTER);
  UNUSED(context);
  if (!globalLockInit) {
    LockInitGlobal();
  }
  return TRUE;
}

static void lockEnsureGlobalLock(void)
{
  static INIT_ONCE init_once = INIT_ONCE_STATIC_INIT;
  BOOL b = InitOnceExecuteOnce(&init_once, lockEnsureGlobalLockCallback,
                               UNUSED_POINTER, NULL);
  AVER(b);
}

void (LockClaimGlobalRecursive)(void)
{
  lockEnsureGlobalLock();
  AVER(globalLockInit);
  LockClaimRecursive(globalRecLock);
}

void (LockReleaseGlobalRecursive)(void)
{
  AVER(globalLockInit);
  LockReleaseRecursive(globalRecLock);
}

void (LockClaimGlobal)(void)
{
  lockEnsureGlobalLock();
  AVER(globalLockInit);
  LockClaim(globalLock);
}

void (LockReleaseGlobal)(void)
{
  AVER(globalLockInit);
  LockRelease(globalLock);
}

void LockSetup(void)
{
  /* Nothing to do as MPS does not support fork() on Windows. */
}

#elif defined(LOCK_NONE)
#include "lockan.c"
#else
#error "No lock configuration."
#endif


/* C. COPYRIGHT AND LICENSE
 *
 * Copyright (C) 2001-2020 Ravenbrook Limited <https://www.ravenbrook.com/>.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the
 *    distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 * IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
