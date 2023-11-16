//
//  glass.c
//  C-Ray
//
//  Created by Valtteri Koskivuori on 30/11/2020.
//  Copyright © 2020-2022 Valtteri Koskivuori. All rights reserved.
//

#include <stdio.h>
#include "../../datatypes/color.h"
#include "../../renderer/samplers/sampler.h"
#include "../../datatypes/vector.h"
#include "../../datatypes/material.h"
#include "../../utils/hashtable.h"
#include "../../datatypes/scene.h"
#include "../bsdfnode.h"

#include "glass.h"

struct glassBsdf {
	struct bsdfNode bsdf;
	const struct colorNode *color;
	const struct valueNode *roughness;
	const struct valueNode *IOR;
};

static bool compare(const void *A, const void *B) {
	const struct glassBsdf *this = A;
	const struct glassBsdf *other = B;
	return this->color == other->color && this->roughness == other->roughness;
}

static uint32_t hash(const void *p) {
	const struct glassBsdf *this = p;
	uint32_t h = hashInit();
	h = hashBytes(h, &this->color, sizeof(this->color));
	h = hashBytes(h, &this->roughness, sizeof(this->roughness));
	return h;
}

static void dump(const void *node, char *dumpbuf, int bufsize) {
	struct glassBsdf *self = (struct glassBsdf *)node;
	char color[DUMPBUF_SIZE / 2] = "";
	char roughness[DUMPBUF_SIZE / 2] = "";
	char IOR[DUMPBUF_SIZE / 2] = "";
	if (self->color->base.dump) self->color->base.dump(self->color, color, sizeof(color));
	if (self->roughness->base.dump) self->roughness->base.dump(self->roughness, roughness, sizeof(roughness));
	if (self->IOR->base.dump) self->IOR->base.dump(self->IOR, IOR, sizeof(IOR));
	snprintf(dumpbuf, bufsize, "glassBsdf { color: %s, roughness: %s, IOR: %s }", color, roughness, IOR);
}

static struct bsdfSample sample(const struct bsdfNode *bsdf, sampler *sampler, const struct hitRecord *record) {
	struct glassBsdf *glassBsdf = (struct glassBsdf *)bsdf;
	
	struct vector outwardNormal;
	struct vector reflected = vec_reflect(record->incident_dir, record->surfaceNormal);
	float niOverNt;
	struct vector refracted;
	float reflectionProbability;
	float cosine;
	
	float IOR = glassBsdf->IOR->eval(glassBsdf->IOR, sampler, record);
	
	if (vec_dot(record->incident_dir, record->surfaceNormal) > 0.0f) {
		outwardNormal = vec_negate(record->surfaceNormal);
		niOverNt = IOR;
		cosine = IOR * vec_dot(record->incident_dir, record->surfaceNormal) / vec_length(record->incident_dir);
	} else {
		outwardNormal = record->surfaceNormal;
		niOverNt = 1.0f / IOR;
		cosine = -(vec_dot(record->incident_dir, record->surfaceNormal) / vec_length(record->incident_dir));
	}
	
	if (vec_refract(record->incident_dir, outwardNormal, niOverNt, &refracted)) {
		reflectionProbability = schlick(cosine, IOR);
	} else {
		reflectionProbability = 1.0f;
	}
	
	float roughness = glassBsdf->roughness->eval(glassBsdf->roughness, sampler, record);
	if (roughness > 0.0f) {
		struct vector fuzz = vec_scale(vec_on_unit_sphere(sampler), roughness);
		reflected = vec_add(reflected, fuzz);
		refracted = vec_add(refracted, fuzz);
	}
	
	struct vector scatterDir = {0};
	if (getDimension(sampler) < reflectionProbability) {
		scatterDir = reflected;
	} else {
		scatterDir = refracted;
	}
	
	return (struct bsdfSample){
		.out = scatterDir,
		.weight = glassBsdf->color->eval(glassBsdf->color, sampler, record)
	};
}

const struct bsdfNode *newGlass(const struct node_storage *s, const struct colorNode *color, const struct valueNode *roughness, const struct valueNode *IOR) {
	HASH_CONS(s->node_table, hash, struct glassBsdf, {
		.color = color ? color : newConstantTexture(s, g_black_color),
		.roughness = roughness ? roughness : newConstantValue(s, 0.0f),
		.IOR = IOR ? IOR : newConstantValue(s, 1.45f),
		.bsdf = {
			.sample = sample,
			.base = { .compare = compare, .dump = dump }
		}
	});
}

