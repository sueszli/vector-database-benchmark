/* poolabs.c: ABSTRACT POOL CLASSES
 *
 * $Id$
 * Copyright (c) 2001-2020 Ravenbrook Limited.  See end of file for license.
 * Portions copyright (C) 2002 Global Graphics Software.
 *
 * PURPOSE
 *
 * .purpose: This defines the abstract pool classes, giving
 * a single-inheritance framework which concrete classes
 * may utilize.  The purpose is to reduce the fragility of class
 * definitions for pool implementations when small changes are
 * made to the pool protocol.   For now, the class hierarchy for
 * the abstract classes is intended to be useful, but not to
 * represent any particular design for pool inheritance.
 *
 * HIERARCHY
 *
 * .hierarchy: define the following hierarchy of abstract pool classes:
 *    AbstractPoolClass     - implements init, finish, describe
 *     AbstractBufferPoolClass - implements the buffer protocol
 *      AbstractSegBufPoolClass - uses SegBuf buffer class
 *       AbstractCollectPoolClass - implements basic GC
 */

#include "mpm.h"

SRCID(poolabs, "$Id$");


/* Mixins:
 *
 * For now (at least) we're avoiding multiple inheritance.
 * However, there is a significant use of multiple inheritance
 * in practice amongst the pool classes, as there are several
 * orthogonal sub-protocols included in the pool protocol.
 * The following mixin functions help to provide the inheritance
 * via a simpler means than real multiple inheritance.
 */


/* PoolClassMixInBuffer -- mix in the protocol for buffer reserve / commit */

void PoolClassMixInBuffer(PoolClass klass)
{
  /* Can't check klass because it's not initialized yet */
  klass->bufferFill = PoolTrivBufferFill;
  klass->bufferEmpty = PoolTrivBufferEmpty;
  /* By default, buffered pools treat frame operations as NOOPs */
  klass->framePush = PoolTrivFramePush;
  klass->framePop = PoolTrivFramePop;
  klass->bufferClass = BufferClassGet;
}


/* PoolClassMixInCollect -- mix in the protocol for GC */

void PoolClassMixInCollect(PoolClass klass)
{
  /* Can't check klass because it's not initialized yet */
  klass->attr |= AttrGC;
  klass->rampBegin = PoolTrivRampBegin;
  klass->rampEnd = PoolTrivRampEnd;
}


/* Classes */


/* PoolAbsInit -- initialize an abstract pool instance */

Res PoolAbsInit(Pool pool, Arena arena, PoolClass klass, ArgList args)
{
  ArgStruct arg;

  AVER(pool != NULL);
  AVERT(Arena, arena);
  UNUSED(args);
  UNUSED(klass); /* used for debug pools only */

  /* Superclass init */
  InstInit(CouldBeA(Inst, pool));

  pool->arena = arena;
  RingInit(&pool->arenaRing);
  RingInit(&pool->bufferRing);
  RingInit(&pool->segRing);
  pool->bufferSerial = (Serial)0;
  pool->alignment = MPS_PF_ALIGN;
  pool->alignShift = SizeLog2(pool->alignment);
  pool->format = NULL;

  if (ArgPick(&arg, args, MPS_KEY_FORMAT)) {
    Format format = arg.val.format;
    AVERT(Format, format);
    AVER(FormatArena(format) == arena);
    pool->format = format;
    /* .init.format: Increment reference count on the format for
       consistency checking.  See .finish.format. */
    ++pool->format->poolCount;
  } else {
    pool->format = NULL;
  }

  pool->serial = ArenaGlobals(arena)->poolSerial;
  ++ArenaGlobals(arena)->poolSerial;

  /* Initialise signature last; see design.mps.sig.init */
  SetClassOfPoly(pool, CLASS(AbstractPool));
  pool->sig = PoolSig;
  AVERT(Pool, pool);

  /* Add initialized pool to list of pools in arena. */
  RingAppend(ArenaPoolRing(arena), PoolArenaRing(pool));

  return ResOK;
}


/* PoolAbsFinish -- finish an abstract pool instance */

void PoolAbsFinish(Inst inst)
{
  Pool pool = MustBeA(AbstractPool, inst);

  EVENT2(PoolFinish, pool, PoolArena(pool));

  /* Detach the pool from the arena and format, and unsig it. */
  RingRemove(PoolArenaRing(pool));

  /* .finish.format: Decrement the reference count on the format for
     consistency checking.  See .format.init. */
  if (pool->format) {
    AVER(pool->format->poolCount > 0);
    --pool->format->poolCount;
    pool->format = NULL;
  }

  pool->sig = SigInvalid;
  InstFinish(CouldBeA(Inst, pool));

  RingFinish(&pool->segRing);
  RingFinish(&pool->bufferRing);
  RingFinish(&pool->arenaRing);
}

DEFINE_CLASS(Inst, PoolClass, klass)
{
  INHERIT_CLASS(klass, PoolClass, InstClass);
  AVERT(InstClass, klass);
}

DEFINE_CLASS(Pool, AbstractPool, klass)
{
  INHERIT_CLASS(&klass->instClassStruct, AbstractPool, Inst);
  klass->instClassStruct.describe = PoolAbsDescribe;
  klass->instClassStruct.finish = PoolAbsFinish;
  klass->size = sizeof(PoolStruct);
  klass->attr = 0;
  klass->varargs = ArgTrivVarargs;
  klass->init = PoolAbsInit;
  klass->alloc = PoolNoAlloc;
  klass->free = PoolNoFree;
  klass->bufferFill = PoolNoBufferFill;
  klass->bufferEmpty = PoolNoBufferEmpty;
  klass->rampBegin = PoolNoRampBegin;
  klass->rampEnd = PoolNoRampEnd;
  klass->framePush = PoolNoFramePush;
  klass->framePop = PoolNoFramePop;
  klass->segPoolGen = PoolNoSegPoolGen;
  klass->freewalk = PoolTrivFreeWalk;
  klass->bufferClass = PoolNoBufferClass;
  klass->debugMixin = PoolNoDebugMixin;
  klass->totalSize = PoolNoSize;
  klass->freeSize = PoolNoSize;
  klass->addrObject = PoolTrivAddrObject;
  klass->sig = PoolClassSig;
  AVERT(PoolClass, klass);
}

DEFINE_CLASS(Pool, AbstractBufferPool, klass)
{
  INHERIT_CLASS(klass, AbstractBufferPool, AbstractPool);
  PoolClassMixInBuffer(klass);
  AVERT(PoolClass, klass);
}

DEFINE_CLASS(Pool, AbstractSegBufPool, klass)
{
  INHERIT_CLASS(klass, AbstractSegBufPool, AbstractBufferPool);
  klass->bufferClass = SegBufClassGet;
  klass->bufferEmpty = PoolSegBufferEmpty;
  AVERT(PoolClass, klass);
}

DEFINE_CLASS(Pool, AbstractCollectPool, klass)
{
  INHERIT_CLASS(klass, AbstractCollectPool, AbstractSegBufPool);
  PoolClassMixInCollect(klass);
  AVERT(PoolClass, klass);
}


/* PoolNo*, PoolTriv* -- Trivial and non-methods for Pool Classes
 *
 * <design/pool#.no> and <design/pool#.triv>
 */

Res PoolNoAlloc(Addr *pReturn, Pool pool, Size size)
{
  AVER(pReturn != NULL);
  AVERT(Pool, pool);
  AVER(size > 0);
  NOTREACHED;
  return ResUNIMPL;
}

Res PoolTrivAlloc(Addr *pReturn, Pool pool, Size size)
{
  AVER(pReturn != NULL);
  AVERT(Pool, pool);
  AVER(size > 0);
  return ResLIMIT;
}

void PoolNoFree(Pool pool, Addr old, Size size)
{
  AVERT(Pool, pool);
  AVER(old != NULL);
  AVER(size > 0);
  NOTREACHED;
}

void PoolTrivFree(Pool pool, Addr old, Size size)
{
  AVERT(Pool, pool);
  AVER(old != NULL);
  AVER(size > 0);
  NOOP;                         /* trivial free has no effect */
}

PoolGen PoolNoSegPoolGen(Pool pool, Seg seg)
{
  AVERT(Pool, pool);
  AVERT(Seg, seg);
  AVER(pool == SegPool(seg));
  NOTREACHED;
  return NULL;
}

Res PoolNoBufferFill(Addr *baseReturn, Addr *limitReturn,
                     Pool pool, Buffer buffer, Size size)
{
  AVER(baseReturn != NULL);
  AVER(limitReturn != NULL);
  AVERT(Pool, pool);
  AVERT(Buffer, buffer);
  AVER(size > 0);
  NOTREACHED;
  return ResUNIMPL;
}

Res PoolTrivBufferFill(Addr *baseReturn, Addr *limitReturn,
                       Pool pool, Buffer buffer, Size size)
{
  Res res;
  Addr p;

  AVER(baseReturn != NULL);
  AVER(limitReturn != NULL);
  AVERT(Pool, pool);
  AVERT(Buffer, buffer);
  AVER(size > 0);

  res = PoolAlloc(&p, pool, size);
  if (res != ResOK)
    return res;

  *baseReturn = p;
  *limitReturn = AddrAdd(p, size);
  return ResOK;
}


void PoolNoBufferEmpty(Pool pool, Buffer buffer)
{
  AVERT(Pool, pool);
  AVERT(Buffer, buffer);
  AVER(BufferIsReady(buffer));
  NOTREACHED;
}

void PoolTrivBufferEmpty(Pool pool, Buffer buffer)
{
  Addr init, limit;

  AVERT(Pool, pool);
  AVERT(Buffer, buffer);
  AVER(BufferIsReady(buffer));

  init = BufferGetInit(buffer);
  limit = BufferLimit(buffer);
  AVER(init <= limit);
  if (limit > init)
    PoolFree(pool, init, AddrOffset(init, limit));
}

void PoolSegBufferEmpty(Pool pool, Buffer buffer)
{
  Seg seg;

  AVERT(Pool, pool);
  AVERT(Buffer, buffer);
  AVER(BufferIsReady(buffer));
  seg = BufferSeg(buffer);
  AVERT(Seg, seg);

  Method(Seg, seg, bufferEmpty)(seg, buffer);
}


Res PoolAbsDescribe(Inst inst, mps_lib_FILE *stream, Count depth)
{
  Pool pool = CouldBeA(AbstractPool, inst);
  Res res;
  Ring node, nextNode;

  if (!TESTC(AbstractPool, pool))
    return ResPARAM;
  if (stream == NULL)
    return ResPARAM;

  res = InstDescribe(CouldBeA(Inst, pool), stream, depth);
  if (res != ResOK)
    return res;

  res = WriteF(stream, depth + 2,
               "serial $U\n", (WriteFU)pool->serial,
               "arena $P ($U)\n",
               (WriteFP)pool->arena, (WriteFU)pool->arena->serial,
               "alignment $W\n", (WriteFW)pool->alignment,
               "alignShift $W\n", (WriteFW)pool->alignShift,
               NULL);
  if (res != ResOK)
    return res;

  if (pool->format != NULL) {
    res = FormatDescribe(pool->format, stream, depth + 2);
    if (res != ResOK)
      return res;
  }

  RING_FOR(node, &pool->bufferRing, nextNode) {
    Buffer buffer = RING_ELT(Buffer, poolRing, node);
    res = BufferDescribe(buffer, stream, depth + 2);
    if (res != ResOK)
      return res;
  }

  return ResOK;
}


Res PoolNoTraceBegin(Pool pool, Trace trace)
{
  AVERT(Pool, pool);
  AVERT(Trace, trace);
  AVER(PoolArena(pool) == trace->arena);
  NOTREACHED;
  return ResUNIMPL;
}

Res PoolTrivTraceBegin(Pool pool, Trace trace)
{
  AVERT(Pool, pool);
  AVERT(Trace, trace);
  AVER(PoolArena(pool) == trace->arena);
  return ResOK;
}

void PoolNoRampBegin(Pool pool, Buffer buf, Bool collectAll)
{
  AVERT(Pool, pool);
  AVERT(Buffer, buf);
  AVERT(Bool, collectAll);
  NOTREACHED;
}


void PoolNoRampEnd(Pool pool, Buffer buf)
{
  AVERT(Pool, pool);
  AVERT(Buffer, buf);
  NOTREACHED;
}


void PoolTrivRampBegin(Pool pool, Buffer buf, Bool collectAll)
{
  AVERT(Pool, pool);
  AVERT(Buffer, buf);
  AVERT(Bool, collectAll);
}


void PoolTrivRampEnd(Pool pool, Buffer buf)
{
  AVERT(Pool, pool);
  AVERT(Buffer, buf);
}


Res PoolNoFramePush(AllocFrame *frameReturn, Pool pool, Buffer buf)
{
  AVER(frameReturn != NULL);
  AVERT(Pool, pool);
  AVERT(Buffer, buf);
  NOTREACHED;
  return ResUNIMPL;
}


Res PoolNoFramePop(Pool pool, Buffer buf, AllocFrame frame)
{
  AVERT(Pool, pool);
  AVERT(Buffer, buf);
  /* frame is of an abstract type & can't be checked */
  UNUSED(frame);
  NOTREACHED;
  return ResUNIMPL;
}


Res PoolTrivFramePush(AllocFrame *frameReturn, Pool pool, Buffer buf)
{
  AVER(frameReturn != NULL);
  AVERT(Pool, pool);
  AVERT(Buffer, buf);
  return ResOK;
}


Res PoolTrivFramePop(Pool pool, Buffer buf, AllocFrame frame)
{
  AVERT(Pool, pool);
  AVERT(Buffer, buf);
  /* frame is of an abstract type & can't be checked */
  UNUSED(frame);
  return ResOK;
}


void PoolTrivFreeWalk(Pool pool, FreeBlockVisitor f, void *p)
{
  AVERT(Pool, pool);
  AVER(FUNCHECK(f));
  /* p is arbitrary, hence can't be checked */
  UNUSED(p);

  /* FreeWalk doesn't have be perfect, so just pretend you didn't find any. */
  NOOP;
}


BufferClass PoolNoBufferClass(void)
{
  NOTREACHED;
  return NULL;
}


Size PoolNoSize(Pool pool)
{
  AVERT(Pool, pool);
  NOTREACHED;
  return UNUSED_SIZE;
}


Res PoolTrivAddrObject(Addr *pReturn, Pool pool, Addr addr)
{
  AVERT(Pool, pool);
  AVER(pReturn != NULL);
  UNUSED(addr);

  return ResUNIMPL;
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
