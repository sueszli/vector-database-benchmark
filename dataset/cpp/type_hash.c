/*
 * This file is part of cparser.
 * Copyright (C) 2012 Matthias Braun <matze@braunis.de>
 */
#include "type_hash.h"

#include <assert.h>
#include <stdbool.h>

#include "type_t.h"

#define HashSet         type_hash_t
#define ValueType       type_t*
#include "adt/hashset.h"
#undef ValueType
#undef HashSet

typedef struct type_hash_iterator_t  type_hash_iterator_t;
typedef struct type_hash_t           type_hash_t;

/* TODO: ^= is a bad way of combining hashes since most addresses are very
 * similar */

static unsigned hash_ptr(const void *ptr)
{
	unsigned ptr_int = ((char*) ptr - (char*) NULL);
	return ptr_int >> 3;
}

static unsigned hash_atomic_type(const atomic_type_t *type)
{
	unsigned some_prime = 27644437;
	unsigned result     = type->akind * some_prime;

	return result;
}

static unsigned hash_pointer_type(const pointer_type_t *type)
{
	return hash_ptr(type->points_to);
}

static unsigned hash_reference_type(const reference_type_t *type)
{
	return hash_ptr(type->refers_to);
}

static unsigned hash_array_type(const array_type_t *type)
{
	return hash_ptr(type->element_type);
}

static unsigned hash_compound_type(const compound_type_t *type)
{
	return hash_ptr(type->compound);
}

static unsigned hash_function_type(const function_type_t *type)
{
	unsigned result = hash_ptr(type->return_type);

	function_parameter_t *parameter = type->parameters;
	while (parameter != NULL) {
		result   ^= hash_ptr(parameter->type);
		parameter = parameter->next;
	}
	result += type->modifiers;
	result += type->linkage;
	result += type->calling_convention;

	return result;
}

static unsigned hash_enum_type(const enum_type_t *type)
{
	return hash_ptr(type->enume);
}

static unsigned hash_typeof_type(const typeof_type_t *type)
{
	unsigned result = hash_ptr(type->expression);
	result         ^= hash_ptr(type->typeof_type);

	return result;
}

static unsigned hash_type(const type_t *type)
{
	unsigned hash = 0;

	switch (type->kind) {
	case TYPE_ERROR:
		return 0;
	case TYPE_IMAGINARY:
	case TYPE_COMPLEX:
	case TYPE_ATOMIC:
		hash = hash_atomic_type(&type->atomic);
		break;
	case TYPE_ENUM:
		hash = hash_enum_type(&type->enumt);
		break;
	case TYPE_COMPOUND_STRUCT:
	case TYPE_COMPOUND_UNION:
		hash = hash_compound_type(&type->compound);
		break;
	case TYPE_FUNCTION:
		hash = hash_function_type(&type->function);
		break;
	case TYPE_POINTER:
		hash = hash_pointer_type(&type->pointer);
		break;
	case TYPE_REFERENCE:
		hash = hash_reference_type(&type->reference);
		break;
	case TYPE_ARRAY:
		hash = hash_array_type(&type->array);
		break;
	case TYPE_TYPEDEF:
		hash = hash_ptr(type->typedeft.typedefe);
		break;
	case TYPE_TYPEOF:
		hash = hash_typeof_type(&type->typeoft);
		break;
	case TYPE_VOID:
	case TYPE_BUILTIN_TEMPLATE:
		break;
	}

	unsigned some_prime = 99991;
	hash ^= some_prime * type->base.qualifiers;

	return hash;
}

static bool atomic_types_equal(const atomic_type_t *type1,
							   const atomic_type_t *type2)
{
	return type1->akind == type2->akind;
}

static bool function_types_equal(const function_type_t *type1,
                                 const function_type_t *type2)
{
	if (type1->return_type != type2->return_type)
		return false;
	if (type1->variadic != type2->variadic)
		return false;
	if (type1->unspecified_parameters != type2->unspecified_parameters)
		return false;
	if (type1->kr_style_parameters != type2->kr_style_parameters)
		return false;
	if (type1->linkage != type2->linkage)
		return false;
	if (type1->modifiers != type2->modifiers)
		return false;
	if (type1->calling_convention != type2->calling_convention)
		return false;

	function_parameter_t *param1 = type1->parameters;
	function_parameter_t *param2 = type2->parameters;
	while (param1 != NULL && param2 != NULL) {
		if (param1->type != param2->type)
			return false;
		param1 = param1->next;
		param2 = param2->next;
	}
	if (param1 != NULL || param2 != NULL)
		return false;

	return true;
}

static bool pointer_types_equal(const pointer_type_t *type1,
                                const pointer_type_t *type2)
{
	return type1->points_to == type2->points_to;
}

static bool reference_types_equal(const reference_type_t *type1,
                                  const reference_type_t *type2)
{
	return type1->refers_to == type2->refers_to;
}

static bool array_types_equal(const array_type_t *type1,
                              const array_type_t *type2)
{
	if (type1->element_type != type2->element_type)
		return false;
	if (type1->is_variable != type2->is_variable)
		return false;
	if (type1->is_static != type2->is_static)
		return false;
	if (type1->size_constant != type2->size_constant)
		return false;

	/* never identify vla types, because we need them for caching calculated
	 * sizes later in ast2firm */
	if (type1->is_vla || type2->is_vla)
		return false;

	/* TODO: compare size expressions for equality... */

	return false;
}

static bool compound_types_equal(const compound_type_t *type1,
                                 const compound_type_t *type2)
{
	return type1->compound == type2->compound;
}

static bool enum_types_equal(const enum_type_t *type1,
                             const enum_type_t *type2)
{
	return type1->enume == type2->enume;
}

static bool typedef_types_equal(const typedef_type_t *type1,
                                const typedef_type_t *type2)
{
	return type1->typedefe == type2->typedefe;
}

static bool typeof_types_equal(const typeof_type_t *type1,
                               const typeof_type_t *type2)
{
	if (type1->expression != type2->expression)
		return false;
	if (type1->typeof_type != type2->typeof_type)
		return false;

	return true;
}

static bool types_equal(const type_t *type1, const type_t *type2)
{
	if (type1 == type2)
		return true;
	if (type1->kind != type2->kind)
		return false;
	if (type1->base.qualifiers != type2->base.qualifiers)
		return false;

	switch (type1->kind) {
	case TYPE_ERROR:
	case TYPE_VOID:
	case TYPE_BUILTIN_TEMPLATE:
		return true;
	case TYPE_ATOMIC:
	case TYPE_IMAGINARY:
	case TYPE_COMPLEX:
		return atomic_types_equal(&type1->atomic, &type2->atomic);
	case TYPE_ENUM:
		return enum_types_equal(&type1->enumt, &type2->enumt);
	case TYPE_COMPOUND_STRUCT:
	case TYPE_COMPOUND_UNION:
		return compound_types_equal(&type1->compound, &type2->compound);
	case TYPE_FUNCTION:
		return function_types_equal(&type1->function, &type2->function);
	case TYPE_POINTER:
		return pointer_types_equal(&type1->pointer, &type2->pointer);
	case TYPE_REFERENCE:
		return reference_types_equal(&type1->reference, &type2->reference);
	case TYPE_ARRAY:
		return array_types_equal(&type1->array, &type2->array);
	case TYPE_TYPEOF:
		return typeof_types_equal(&type1->typeoft, &type2->typeoft);
	case TYPE_TYPEDEF:
		return typedef_types_equal(&type1->typedeft, &type2->typedeft);
	}

	abort();
}

#define HashSet                    type_hash_t
#define ValueType                  type_t*
#define NullValue                  NULL
#define DeletedValue               ((type_t*)-1)
#define Hash(this, key)            hash_type(key)
#define KeysEqual(this,key1,key2)  types_equal(key1, key2)
#define SetRangeEmpty(ptr,size)    memset(ptr, 0, (size) * sizeof(*(ptr)))

void _typehash_init(type_hash_t *hash);
#define hashset_init             _typehash_init
void _typehash_destroy(type_hash_t *hash);
#define hashset_destroy          _typehash_destroy
type_t *_typehash_insert(type_hash_t *hash, type_t *type);
#define hashset_insert           _typehash_insert
#define SCALAR_RETURN

#include "adt/hashset.c.h"

static type_hash_t typehash;

void init_typehash(void)
{
	_typehash_init(&typehash);
}

void exit_typehash(void)
{
	_typehash_destroy(&typehash);
}

type_t *typehash_insert(type_t *type)
{
	return _typehash_insert(&typehash, type);
}
