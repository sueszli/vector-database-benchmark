/* than.c: ANSI THREADS MANAGER
 *
 *  $Id$
 *  Copyright (c) 2001-2020 Ravenbrook Limited.  See end of file for license.
 *
 *  This is a single-threaded implementation of the threads manager.
 *  Has stubs for thread suspension.
 *  <design/thread-manager#.impl.an>.
 */

#include "mpm.h"

SRCID(than, "$Id$");


typedef struct mps_thr_s {      /* ANSI fake thread structure */
  Sig sig;                      /* design.mps.sig.field */
  Serial serial;                /* from arena->threadSerial */
  Arena arena;                  /* owning arena */
  RingStruct arenaRing;         /* attaches to arena */
} ThreadStruct;


Bool ThreadCheck(Thread thread)
{
  CHECKS(Thread, thread);
  CHECKU(Arena, thread->arena);
  CHECKL(thread->serial < thread->arena->threadSerial);
  CHECKD_NOSIG(Ring, &thread->arenaRing);
  return TRUE;
}


Bool ThreadCheckSimple(Thread thread)
{
  CHECKS(Thread, thread);
  return TRUE;
}


Res ThreadRegister(Thread *threadReturn, Arena arena)
{
  Res res;
  Thread thread;
  Ring ring;
  void *p;

  AVER(threadReturn != NULL);

  res = ControlAlloc(&p, arena, sizeof(ThreadStruct));
  if (res != ResOK)
    return res;
  thread = (Thread)p;

  thread->arena = arena;
  RingInit(&thread->arenaRing);

  thread->sig = ThreadSig;
  thread->serial = arena->threadSerial;
  ++arena->threadSerial;

  AVERT(Thread, thread);

  ring = ArenaThreadRing(arena);

  RingAppend(ring, &thread->arenaRing);

  *threadReturn = thread;
  return ResOK;
}


void ThreadDeregister(Thread thread, Arena arena)
{
  AVERT(Thread, thread);
  AVERT(Arena, arena);

  RingRemove(&thread->arenaRing);

  thread->sig = SigInvalid;

  RingFinish(&thread->arenaRing);

  ControlFree(arena, thread, sizeof(ThreadStruct));
}


void ThreadRingSuspend(Ring threadRing, Ring deadRing)
{
  AVERT(Ring, threadRing);
  AVERT(Ring, deadRing);
}

void ThreadRingResume(Ring threadRing, Ring deadRing)
{
  AVERT(Ring, threadRing);
  AVERT(Ring, deadRing);
}

Thread ThreadRingThread(Ring threadRing)
{
  Thread thread;
  AVERT(Ring, threadRing);
  thread = RING_ELT(Thread, arenaRing, threadRing);
  AVERT(Thread, thread);
  return thread;
}


/* Must be thread-safe. <design/interface-c#.check.testt>. */

Arena ThreadArena(Thread thread)
{
  AVER(TESTT(Thread, thread));
  return thread->arena;
}


Res ThreadScan(ScanState ss, Thread thread, void *stackCold,
               mps_area_scan_t scan_area,
               void *closure)
{
  UNUSED(thread);
  return StackScan(ss, stackCold, scan_area, closure);
}


Res ThreadDescribe(Thread thread, mps_lib_FILE *stream, Count depth)
{
  Res res;

  res = WriteF(stream, depth,
               "Thread $P ($U) {\n", (WriteFP)thread, (WriteFU)thread->serial,
               "  arena $P ($U)\n",
               (WriteFP)thread->arena, (WriteFU)thread->arena->serial,
               "} Thread $P ($U)\n", (WriteFP)thread, (WriteFU)thread->serial,
               NULL);
  if (res != ResOK)
    return res;

  return ResOK;
}


void ThreadSetup(void)
{
  /* Nothing to do as ANSI platform does not have fork(). */
}


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
