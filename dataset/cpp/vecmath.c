//
//  vecmath.c
//  C-Ray
//
//  Created by Valtteri Koskivuori on 28/12/2020.
//  Copyright © 2020-2022 Valtteri Koskivuori. All rights reserved.
//

#include <stdio.h>
#include "../nodebase.h"

#include "../../utils/hashtable.h"
#include "../../renderer/samplers/sampler.h"
#include "../../datatypes/scene.h"
#include "../../datatypes/hitrecord.h"
#include "../vectornode.h"

#include "../../datatypes/vector.h"
#include "vecmath.h"

struct vecMathNode {
	struct vectorNode node;
	const struct vectorNode *A;
	const struct vectorNode *B;
	const struct vectorNode *C;
	const struct valueNode *f;
	const enum vecOp op;
};

static bool compare(const void *A, const void *B) {
	const struct vecMathNode *this = A;
	const struct vecMathNode *other = B;
	return this->A == other->A && this->B == other->B && this->C == other->C && this->f == other->f && this->op == other->op;
}

static uint32_t hash(const void *p) {
	const struct vecMathNode *this = p;
	uint32_t h = hashInit();
	h = hashBytes(h, &this->A, sizeof(this->A));
	h = hashBytes(h, &this->B, sizeof(this->B));
	h = hashBytes(h, &this->C, sizeof(this->C));
	h = hashBytes(h, &this->f, sizeof(this->f));
	h = hashBytes(h, &this->op, sizeof(this->op));
	return h;
}

static void dump_vec_op(const enum vecOp op, char *buf, size_t bufsize) {
	switch(op) {
		case VecAdd:
			snprintf(buf, bufsize, "add");
			break;
		case VecSubtract:
			snprintf(buf, bufsize, "subtract");
			break;
		case VecMultiply:
			snprintf(buf, bufsize, "multiply");
			break;
		case VecDivide:
			snprintf(buf, bufsize, "divide");
			break;
		case VecCross:
			snprintf(buf, bufsize, "cross");
			break;
		case VecReflect:
			snprintf(buf, bufsize, "reflect");
			break;
		case VecRefract:
			snprintf(buf, bufsize, "refract");
			break;
		case VecDot:
			snprintf(buf, bufsize, "dot");
			break;
		case VecDistance:
			snprintf(buf, bufsize, "distance");
			break;
		case VecLength:
			snprintf(buf, bufsize, "length");
			break;
		case VecScale:
			snprintf(buf, bufsize, "scale");
			break;
		case VecNormalize:
			snprintf(buf, bufsize, "normalize");
			break;
		case VecWrap:
			snprintf(buf, bufsize, "wrap");
			break;
		case VecFloor:
			snprintf(buf, bufsize, "floor");
			break;
		case VecCeil:
			snprintf(buf, bufsize, "ceil");
			break;
		case VecModulo:
			snprintf(buf, bufsize, "mod");
			break;
		case VecAbs:
			snprintf(buf, bufsize, "abs");
			break;
		case VecMin:
			snprintf(buf, bufsize, "min");
			break;
		case VecMax:
			snprintf(buf, bufsize, "max");
			break;
		case VecSin:
			snprintf(buf, bufsize, "sin");
			break;
		case VecCos:
			snprintf(buf, bufsize, "cos");
			break;
		case VecTan:
			snprintf(buf, bufsize, "tan");
			break;
	}
}

static void dump(const void *node, char *dumpbuf, int bufsize) {
	struct vecMathNode *self = (struct vecMathNode *)node;
	char A[DUMPBUF_SIZE / 2] = "";
	char B[DUMPBUF_SIZE / 2] = "";
	char C[DUMPBUF_SIZE / 2] = "";
	char f[DUMPBUF_SIZE / 2] = "";
	char op[DUMPBUF_SIZE / 2] = "";
	if (self->A->base.dump) self->A->base.dump(self->A, A, sizeof(A));
	if (self->B->base.dump) self->B->base.dump(self->B, B, sizeof(B));
	if (self->C->base.dump) self->C->base.dump(self->C, C, sizeof(C));
	if (self->f->base.dump) self->f->base.dump(self->f, f, sizeof(f));
	dump_vec_op(self->op, op, sizeof(op));
	snprintf(dumpbuf, bufsize, "vecMathNode { A: %s, B: %s, C: %s, f: %s, op: %s }", A, B, C, f, op);
}

static inline float wrap(float value, float max, float min) {
	const float range = max - min;
	return (range != 0.0f) ? value - (range * floorf((value - min) / range)) : min;
}
 
static struct vectorValue eval(const struct vectorNode *node, sampler *sampler, const struct hitRecord *record) {
	struct vecMathNode *this = (struct vecMathNode *)node;
	
	const struct vector a = this->A->eval(this->A, sampler, record).v;
	const struct vector b = this->B->eval(this->B, sampler, record).v;
	const struct vector c = this->C->eval(this->C, sampler, record).v;
	const float f = this->f->eval(this->f, sampler, record);
	
	switch (this->op) {
		case VecAdd:
			return (struct vectorValue){ .v = vec_add(a, b) };
		case VecSubtract:
			return (struct vectorValue){ .v = vec_sub(a, b) };
		case VecMultiply:
			return (struct vectorValue){ .v = vec_mul(a, b) };
		case VecDivide:
			return (struct vectorValue){ .v = { a.x / b.x, a.y / b.y, a.z / b.z } };
		case VecCross:
			return (struct vectorValue){ .v = vec_cross(a, b) };
		case VecReflect:
			return (struct vectorValue){ .v = vec_reflect(a, b) };
		case VecRefract:
		{
			struct vectorValue v = { 0 };
			v.f = vec_refract(a, b, f, &v.v) ? 1.0f : 0.0f;
			return v;
		}
		case VecDot:
			return (struct vectorValue){ .f = vec_dot(a, b) };
		case VecDistance:
			return (struct vectorValue){ .f = vec_distance_to(a, b) };
		case VecLength:
			return (struct vectorValue){ .f = vec_length(a) };
		case VecScale:
			return (struct vectorValue){ .v = vec_scale(a, f) };
		case VecNormalize:
			return (struct vectorValue){ .v = vec_normalize(a) };
		case VecWrap:
			return (struct vectorValue){ .v = { wrap(a.x, b.x, c.x), wrap(a.y, b.y, c.y), wrap(a.z, b.z, c.z) } };
		case VecFloor:
			return (struct vectorValue){ .v = { .x = floorf(a.x), .y = floorf(a.y), .z = floorf(a.z) } };
		case VecCeil:
			return (struct vectorValue){ .v = { .x = ceilf(a.x), .y = ceilf(a.y), .z = ceilf(a.z) } };
		case VecModulo:
			return (struct vectorValue){ .v = { .x = fmodf(a.x, b.x), .y = fmodf(a.y, b.y), .z = fmodf(a.z, b.z) } };
		case VecAbs:
			return (struct vectorValue){ .v = { .x = fabsf(a.x), .y = fabsf(a.y), .z = fabsf(a.z) } };
		case VecMin:
			return (struct vectorValue){ .v = { .x = fminf(a.x, b.x), .y = fminf(a.y, b.y), .z = fminf(a.z, b.z) } };
		case VecMax:
			return (struct vectorValue){ .v = { .x = fmaxf(a.x, b.x), .y = fmaxf(a.y, b.y), .z = fmaxf(a.z, b.z) } };
		case VecSin:
			return (struct vectorValue){ .v = { .x = sinf(a.x), .y = sinf(a.y), .z = sinf(a.z) } };
		case VecCos:
			return (struct vectorValue){ .v = { .x = cosf(a.x), .y = cosf(a.y), .z = cosf(a.z) } };
		case VecTan:
			return (struct vectorValue){ .v = { .x = tanf(a.x), .y = tanf(a.y), .z = tanf(a.z) } };
	}
	ASSERT_NOT_REACHED();
	return (struct vectorValue){ 0 };
}

const struct vectorNode *newVecMath(const struct node_storage *s, const struct vectorNode *A, const struct vectorNode *B, const struct vectorNode *C, const struct valueNode *f, const enum vecOp op) {
	HASH_CONS(s->node_table, hash, struct vecMathNode, {
		.A = A ? A : newConstantVector(s, vec_zero()),
		.B = B ? B : newConstantVector(s, vec_zero()),
		.C = C ? C : newConstantVector(s, vec_zero()),
		.f = f ? f : newConstantValue(s, 0.0f),
		.op = op,
		.node = {
			.eval = eval,
			.base = { .compare = compare, .dump = dump }
		}
	});
}
