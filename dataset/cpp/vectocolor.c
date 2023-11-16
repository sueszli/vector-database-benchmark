//
//  vectocolor.c
//  C-Ray
//
//  Created by Valtteri Koskivuori on 15/04/2021.
//  Copyright © 2021-2022 Valtteri Koskivuori. All rights reserved.
//

#include <stdio.h>
#include "../nodebase.h"

#include "../../renderer/samplers/sampler.h"
#include "../../utils/hashtable.h"
#include "../../datatypes/scene.h"
#include "../../datatypes/hitrecord.h"
#include "../colornode.h"
#include "../vectornode.h"

#include "vectocolor.h"

struct vecToColorNode {
	struct colorNode node;
	const struct vectorNode *vec;
};

static bool compare(const void *A, const void *B) {
	const struct vecToColorNode *this = A;
	const struct vecToColorNode *other = B;
	return this->vec == other->vec;
}

static uint32_t hash(const void *p) {
	const struct vecToColorNode *this = p;
	uint32_t h = hashInit();
	h = hashBytes(h, &this->vec, sizeof(this->vec));
	return h;
}

static void dump(const void *node, char *dumpbuf, int bufsize) {
	struct vecToColorNode *self = (struct vecToColorNode *)node;
	char vec[DUMPBUF_SIZE / 2] = "";
	if (self->vec->base.dump) self->vec->base.dump(self->vec, vec, sizeof(vec));
	snprintf(dumpbuf, bufsize, "vecTovecNode { vec: %s }", vec);
}

static struct color eval(const struct colorNode *node, sampler *sampler, const struct hitRecord *record) {
	(void)record;
	struct vecToColorNode *this = (struct vecToColorNode *)node;
	struct vector vec = this->vec->eval(this->vec, sampler, record).v;
	vec = vec_max(vec, vec_zero());
	return (struct color){ vec.x, vec.y, vec.z, 0.0f };
}

const struct colorNode *newVecToColor(const struct node_storage *s, const struct vectorNode *vec) {
	HASH_CONS(s->node_table, hash, struct vecToColorNode, {
		.vec = vec ? vec : newConstantVector(s, vec_zero()),
		.node = {
			.eval = eval,
			.base = { .compare = compare, .dump = dump }
		}
	});
}
