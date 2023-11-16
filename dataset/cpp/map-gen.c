/* -*- linux-c -*- 
 * map API generator
 * Copyright (C) 2005 Red Hat Inc.
 *
 * This file is part of systemtap, and is free software.  You can
 * redistribute it and/or modify it under the terms of the GNU General
 * Public License (GPL); either version 2, or (at your option) any
 * later version.
 */

/** @file map-gen.c
 * @brief Map function generator
 * This file is a template designed to be included as many times as
 * needed to generate the necessary map functions.
 */

#define JOIN(x,y) JOINx(x,y)
#define JOINx(x,y) x##_##y
#define JOIN2(x,y,z) JOIN2x(x,y,z)
#define JOIN2x(x,y,z) x##_##y##z
#define JOIN3(a,b,c,d) JOIN3x(a,b,c,d)
#define JOIN3x(a,b,c,d) a##_##b##c##d
#define JOIN4(a,b,c,d,e) JOIN4x(a,b,c,d,e)
#define JOIN4x(a,b,c,d,e) a##_##b##c##d##e
#define JOIN5(a,b,c,d,e,f) JOIN5x(a,b,c,d,e,f)
#define JOIN5x(a,b,c,d,e,f) a##_##b##c##d##e##f
#define JOIN6(a,b,c,d,e,f,g) JOIN6x(a,b,c,d,e,f,g)
#define JOIN6x(a,b,c,d,e,f,g) a##_##b##c##d##e##f##g
#define JOIN7(a,b,c,d,e,f,g,h) JOIN7x(a,b,c,d,e,f,g,h)
#define JOIN7x(a,b,c,d,e,f,g,h) a##_##b##c##d##e##f##g##h
#define JOIN8(a,b,c,d,e,f,g,h,i) JOIN8x(a,b,c,d,e,f,g,h,i)
#define JOIN8x(a,b,c,d,e,f,g,h,i) a##_##b##c##d##e##f##g##h##i
#define JOIN9(a,b,c,d,e,f,g,h,i,j) JOIN9x(a,b,c,d,e,f,g,h,i,j)
#define JOIN9x(a,b,c,d,e,f,g,h,i,j) a##_##b##c##d##e##f##g##h##i##j
#define JOIN10(a,b,c,d,e,f,g,h,i,j,k) JOIN10x(a,b,c,d,e,f,g,h,i,j,k)
#define JOIN10x(a,b,c,d,e,f,g,h,i,j,k) a##_##b##c##d##e##f##g##h##i##j##k

#include "map.h"

#if !defined(VALUE_TYPE)
#error Need to define VALUE_TYPE as STRING, STAT, or INT64
#endif

#if VALUE_TYPE == STRING
#define VALTYPE char*
#define VSTYPE char*
#define VALNAME str
#define VALN s
#define MAP_SET_VAL(a,b,c,d) _new_map_set_str(a,b,c,d)
#define MAP_GET_VAL(n) _stp_get_str(n)
#define NULLRET ""
#elif VALUE_TYPE == INT64
#define VALTYPE int64_t
#define VSTYPE int64_t
#define VALNAME int64
#define VALN i
#define MAP_SET_VAL(a,b,c,d) _new_map_set_int64(a,b,c,d)
#define MAP_GET_VAL(n) _stp_get_int64(n)
#define NULLRET (int64_t)0
#elif VALUE_TYPE == STAT
#define VALTYPE stat*
#define VSTYPE int64_t
#define VALNAME stat
#define VALN x
#define MAP_SET_VAL(a,b,c,d) _new_map_set_stat(a,b,c,d)
#define MAP_GET_VAL(n) _stp_get_stat(n)
#define NULLRET (stat*)0
#else
#error Need to define VALUE_TYPE as STRING, STAT, or INT64
#endif /* VALUE_TYPE */

//#define MAP_SET_VAL(a,b,c,d) _new_map_set_##VALNAME(a,b,c,d)

#if defined (KEY1_TYPE)
#define KEY_ARITY 1
#if KEY1_TYPE == STRING
#define KEY1TYPE char*
#define KEY1NAME str
#define KEY1N s
#define KEY1STOR char key1[MAP_STRING_LENGTH]
#define KEY1CPY(m) str_copy(m->key1, key1)
#else
#define KEY1TYPE int64_t
#define KEY1NAME int64
#define KEY1N i
#define KEY1STOR int64_t key1
#define KEY1CPY(m) m->key1=key1
#endif
#define KEY1_EQ_P JOIN(KEY1NAME,eq_p)
#define KEY1_HASH JOIN(KEY1NAME,hash)
#endif /* defined(KEY1_TYPE) */

#if defined (KEY2_TYPE)
#undef KEY_ARITY
#define KEY_ARITY 2
#if KEY2_TYPE == STRING
#define KEY2TYPE char*
#define KEY2NAME str
#define KEY2N s
#define KEY2STOR char key2[MAP_STRING_LENGTH]
#define KEY2CPY(m) str_copy(m->key2, key2)
#else
#define KEY2TYPE int64_t
#define KEY2NAME int64
#define KEY2N i
#define KEY2STOR int64_t key2
#define KEY2CPY(m) m->key2=key2
#endif
#define KEY2_EQ_P JOIN(KEY2NAME,eq_p)
#define KEY2_HASH JOIN(KEY2NAME,hash)
#endif /* defined(KEY2_TYPE) */

#if defined (KEY3_TYPE)
#undef KEY_ARITY
#define KEY_ARITY 3
#if KEY3_TYPE == STRING
#define KEY3TYPE char*
#define KEY3NAME str
#define KEY3N s
#define KEY3STOR char key3[MAP_STRING_LENGTH]
#define KEY3CPY(m) str_copy(m->key3, key3)
#else
#define KEY3TYPE int64_t
#define KEY3NAME int64
#define KEY3N i
#define KEY3STOR int64_t key3
#define KEY3CPY(m) m->key3=key3
#endif
#define KEY3_EQ_P JOIN(KEY3NAME,eq_p)
#define KEY3_HASH JOIN(KEY3NAME,hash)
#endif /* defined(KEY3_TYPE) */

#if defined (KEY4_TYPE)
#undef KEY_ARITY
#define KEY_ARITY 4
#if KEY4_TYPE == STRING
#define KEY4TYPE char*
#define KEY4NAME str
#define KEY4N s
#define KEY4STOR char key4[MAP_STRING_LENGTH]
#define KEY4CPY(m) str_copy(m->key4, key4)
#else
#define KEY4TYPE int64_t
#define KEY4NAME int64
#define KEY4N i
#define KEY4STOR int64_t key4
#define KEY4CPY(m) m->key4=key4
#endif
#define KEY4_EQ_P JOIN(KEY4NAME,eq_p)
#define KEY4_HASH JOIN(KEY4NAME,hash)
#endif /* defined(KEY4_TYPE) */

#if defined (KEY5_TYPE)
#undef KEY_ARITY
#define KEY_ARITY 5
#if KEY5_TYPE == STRING
#define KEY5TYPE char*
#define KEY5NAME str
#define KEY5N s
#define KEY5STOR char key5[MAP_STRING_LENGTH]
#define KEY5CPY(m) str_copy(m->key5, key5)
#else
#define KEY5TYPE int64_t
#define KEY5NAME int64
#define KEY5N i
#define KEY5STOR int64_t key5
#define KEY5CPY(m) m->key5=key5
#endif
#define KEY5_EQ_P JOIN(KEY5NAME,eq_p)
#define KEY5_HASH JOIN(KEY5NAME,hash)
#endif /* defined(KEY5_TYPE) */

#if defined (KEY6_TYPE)
#undef KEY_ARITY
#define KEY_ARITY 6
#if KEY6_TYPE == STRING
#define KEY6TYPE char*
#define KEY6NAME str
#define KEY6N s
#define KEY6STOR char key6[MAP_STRING_LENGTH]
#define KEY6CPY(m) str_copy(m->key6, key6)
#else
#define KEY6TYPE int64_t
#define KEY6NAME int64
#define KEY6N i
#define KEY6STOR int64_t key6
#define KEY6CPY(m) m->key6=key6
#endif
#define KEY6_EQ_P JOIN(KEY6NAME,eq_p)
#define KEY6_HASH JOIN(KEY6NAME,hash)
#endif /* defined(KEY6_TYPE) */

#if defined (KEY7_TYPE)
#undef KEY_ARITY
#define KEY_ARITY 7
#if KEY7_TYPE == STRING
#define KEY7TYPE char*
#define KEY7NAME str
#define KEY7N s
#define KEY7STOR char key7[MAP_STRING_LENGTH]
#define KEY7CPY(m) str_copy(m->key7, key7)
#else
#define KEY7TYPE int64_t
#define KEY7NAME int64
#define KEY7N i
#define KEY7STOR int64_t key7
#define KEY7CPY(m) m->key7=key7
#endif
#define KEY7_EQ_P JOIN(KEY7NAME,eq_p)
#define KEY7_HASH JOIN(KEY7NAME,hash)
#endif /* defined(KEY7_TYPE) */

#if defined (KEY7_TYPE)
#undef KEY_ARITY
#define KEY_ARITY 7
#if KEY7_TYPE == STRING
#define KEY7TYPE char*
#define KEY7NAME str
#define KEY7N s
#define KEY7STOR char key7[MAP_STRING_LENGTH]
#define KEY7CPY(m) str_copy(m->key7, key7)
#else
#define KEY7TYPE int64_t
#define KEY7NAME int64
#define KEY7N i
#define KEY7STOR int64_t key7
#define KEY7CPY(m) m->key7=key7
#endif
#define KEY7_EQ_P JOIN(KEY7NAME,eq_p)
#define KEY7_HASH JOIN(KEY7NAME,hash)
#endif /* defined(KEY7_TYPE) */

#if defined (KEY8_TYPE)
#undef KEY_ARITY
#define KEY_ARITY 8
#if KEY8_TYPE == STRING
#define KEY8TYPE char*
#define KEY8NAME str
#define KEY8N s
#define KEY8STOR char key8[MAP_STRING_LENGTH]
#define KEY8CPY(m) str_copy(m->key8, key8)
#else
#define KEY8TYPE int64_t
#define KEY8NAME int64
#define KEY8N i
#define KEY8STOR int64_t key8
#define KEY8CPY(m) m->key8=key8
#endif
#define KEY8_EQ_P JOIN(KEY8NAME,eq_p)
#define KEY8_HASH JOIN(KEY8NAME,hash)
#endif /* defined(KEY8_TYPE) */

#if defined (KEY9_TYPE)
#undef KEY_ARITY
#define KEY_ARITY 9
#if KEY9_TYPE == STRING
#define KEY9TYPE char*
#define KEY9NAME str
#define KEY9N s
#define KEY9STOR char key9[MAP_STRING_LENGTH]
#define KEY9CPY(m) str_copy(m->key9, key9)
#else
#define KEY9TYPE int64_t
#define KEY9NAME int64
#define KEY9N i
#define KEY9STOR int64_t key9
#define KEY9CPY(m) m->key9=key9
#endif
#define KEY9_EQ_P JOIN(KEY9NAME,eq_p)
#define KEY9_HASH JOIN(KEY9NAME,hash)
#endif /* defined(KEY9_TYPE) */

/* Not so many, cowboy! */
#if defined (KEY10_TYPE)
#error "excessive key arity == too many array indexes"
#endif



#if KEY_ARITY == 1
#define KEYSYM(x) JOIN2(x,KEY1N,VALN)
#define ALLKEYS(x) x##1
#define ALLKEYSD(x) KEY1TYPE x##1
#define KEYCPY(m) {KEY1CPY(m);}
#elif KEY_ARITY == 2
#define KEYSYM(x) JOIN3(x,KEY1N,KEY2N,VALN)
#define ALLKEYS(x) x##1, x##2
#define ALLKEYSD(x) KEY1TYPE x##1, KEY2TYPE x##2
#define KEYCPY(m) {KEY1CPY(m);KEY2CPY(m);}
#elif KEY_ARITY == 3
#define KEYSYM(x) JOIN4(x,KEY1N,KEY2N,KEY3N,VALN)
#define ALLKEYS(x) x##1, x##2, x##3
#define ALLKEYSD(x) KEY1TYPE x##1, KEY2TYPE x##2, KEY3TYPE x##3
#define KEYCPY(m) {KEY1CPY(m);KEY2CPY(m);KEY3CPY(m);}
#elif KEY_ARITY == 4
#define KEYSYM(x) JOIN5(x,KEY1N,KEY2N,KEY3N,KEY4N,VALN)
#define ALLKEYS(x) x##1, x##2, x##3, x##4
#define ALLKEYSD(x) KEY1TYPE x##1, KEY2TYPE x##2, KEY3TYPE x##3, KEY4TYPE x##4
#define KEYCPY(m) {KEY1CPY(m);KEY2CPY(m);KEY3CPY(m);KEY4CPY(m);}
#elif KEY_ARITY == 5
#define KEYSYM(x) JOIN6(x,KEY1N,KEY2N,KEY3N,KEY4N,KEY5N,VALN)
#define ALLKEYS(x) x##1, x##2, x##3, x##4, x##5
#define ALLKEYSD(x) KEY1TYPE x##1, KEY2TYPE x##2, KEY3TYPE x##3, KEY4TYPE x##4, KEY5TYPE x##5
#define KEYCPY(m) {KEY1CPY(m);KEY2CPY(m);KEY3CPY(m);KEY4CPY(m);KEY5CPY(m);}
#elif KEY_ARITY == 6
#define KEYSYM(x) JOIN7(x,KEY1N,KEY2N,KEY3N,KEY4N,KEY5N,KEY6N,VALN)
#define ALLKEYS(x) x##1, x##2, x##3, x##4, x##5, x##6
#define ALLKEYSD(x) KEY1TYPE x##1, KEY2TYPE x##2, KEY3TYPE x##3, KEY4TYPE x##4, KEY5TYPE x##5, KEY6TYPE x##6
#define KEYCPY(m) {KEY1CPY(m);KEY2CPY(m);KEY3CPY(m);KEY4CPY(m);KEY5CPY(m);KEY6CPY(m);}
#elif KEY_ARITY == 7
#define KEYSYM(x) JOIN8(x,KEY1N,KEY2N,KEY3N,KEY4N,KEY5N,KEY6N,KEY7N,VALN)
#define ALLKEYS(x) x##1, x##2, x##3, x##4, x##5, x##6, x##7
#define ALLKEYSD(x) KEY1TYPE x##1, KEY2TYPE x##2, KEY3TYPE x##3, KEY4TYPE x##4, KEY5TYPE x##5, KEY6TYPE x##6, KEY7TYPE x##7
#define KEYCPY(m) {KEY1CPY(m);KEY2CPY(m);KEY3CPY(m);KEY4CPY(m);KEY5CPY(m);KEY6CPY(m);KEY7CPY(m);}
#elif KEY_ARITY == 8
#define KEYSYM(x) JOIN9(x,KEY1N,KEY2N,KEY3N,KEY4N,KEY5N,KEY6N,KEY7N,KEY8N,VALN)
#define ALLKEYS(x) x##1, x##2, x##3, x##4, x##5, x##6, x##7, x##8
#define ALLKEYSD(x) KEY1TYPE x##1, KEY2TYPE x##2, KEY3TYPE x##3, KEY4TYPE x##4, KEY5TYPE x##5, KEY6TYPE x##6, KEY7TYPE x##7, KEY8TYPE x##8
#define KEYCPY(m) {KEY1CPY(m);KEY2CPY(m);KEY3CPY(m);KEY4CPY(m);KEY5CPY(m);KEY6CPY(m);KEY7CPY(m);KEY8CPY(m);}
#elif KEY_ARITY == 9
#define KEYSYM(x) JOIN10(x,KEY1N,KEY2N,KEY3N,KEY4N,KEY5N,KEY6N,KEY7N,KEY8N,KEY9N,VALN)
#define ALLKEYS(x) x##1, x##2, x##3, x##4, x##5, x##6, x##7, x##8, x##9
#define ALLKEYSD(x) KEY1TYPE x##1, KEY2TYPE x##2, KEY3TYPE x##3, KEY4TYPE x##4, KEY5TYPE x##5, KEY6TYPE x##6, KEY7TYPE x##7, KEY8TYPE x##8, KEY9TYPE x##9
#define KEYCPY(m) {KEY1CPY(m);KEY2CPY(m);KEY3CPY(m);KEY4CPY(m);KEY5CPY(m);KEY6CPY(m);KEY7CPY(m);KEY8CPY(m);KEY9CPY(m);}
#endif

/* */

struct KEYSYM(map_node) {
	/* list of other nodes in the map */
	struct list_head lnode;
	/* list of nodes with the same hash value */
	struct hlist_node hnode;
	/* pointer back to the map struct */
	struct map_root *map;

	KEY1STOR;
#if KEY_ARITY > 1
	KEY2STOR;
#if KEY_ARITY > 2
	KEY3STOR;
#if KEY_ARITY > 3
	KEY4STOR;
#if KEY_ARITY > 4
	KEY5STOR;
#if KEY_ARITY > 5
	KEY6STOR;
#if KEY_ARITY > 6
	KEY7STOR;
#if KEY_ARITY > 7
	KEY8STOR;
#if KEY_ARITY > 8
	KEY9STOR;
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
};

#define type_to_enum(type)						\
	({								\
		int ret;						\
		if (__builtin_types_compatible_p (type, char*)) 	\
			ret = STRING;					\
		else							\
			ret = INT64;					\
		ret;							\
	})

static key_data KEYSYM(map_get_key) (struct map_node *mn, int n, int *type)
{
	key_data ptr;
	struct KEYSYM(map_node) *m = (struct KEYSYM(map_node) *)mn;	

	if (n > KEY_ARITY || n < 1) {
		if (type)
			*type = END;
		return (key_data)(int64_t)0;
	}

	switch (n) {
	case 1:
		ptr = (key_data)m->key1;
		if (type)
			*type = type_to_enum(KEY1TYPE);
		break;
#if KEY_ARITY > 1
	case 2:
		ptr = (key_data)m->key2;
		if (type)
			*type = type_to_enum(KEY2TYPE);

		break;
#if KEY_ARITY > 2
	case 3:
		ptr = (key_data)m->key3;
		if (type)
			*type = type_to_enum(KEY3TYPE);
		break;
#if KEY_ARITY > 3
	case 4:
		ptr = (key_data)m->key4;
		if (type)
			*type = type_to_enum(KEY4TYPE);
		break;
#if KEY_ARITY > 4
	case 5:
		ptr = (key_data)m->key5;
		if (type)
			*type = type_to_enum(KEY5TYPE);
		break;
#if KEY_ARITY > 5
	case 6:
		ptr = (key_data)m->key6;
		if (type)
			*type = type_to_enum(KEY6TYPE);
		break;
#if KEY_ARITY > 6
	case 7:
		ptr = (key_data)m->key7;
		if (type)
			*type = type_to_enum(KEY7TYPE);
		break;
#if KEY_ARITY > 7
	case 8:
		ptr = (key_data)m->key8;
		if (type)
			*type = type_to_enum(KEY8TYPE);
		break;
#if KEY_ARITY > 8
	case 9:
		ptr = (key_data)m->key9;
		if (type)
			*type = type_to_enum(KEY9TYPE);
		break;
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
	default:
		ptr = (key_data)(int64_t)0;
		if (type)
			*type = END;
	}
	return ptr;
}


static unsigned int KEYSYM(keycheck) (ALLKEYSD(key))
{
#if KEY1_TYPE == STRING
	if (key1 == NULL)
		return 0;
#endif

#if KEY_ARITY > 1
#if KEY2_TYPE == STRING
	if (key2 == NULL)
		return 0;
#endif

#if KEY_ARITY > 2
#if KEY3_TYPE == STRING
	if (key3 == NULL)
		return 0;
#endif

#if KEY_ARITY > 3
#if KEY4_TYPE == STRING
	if (key4 == NULL)
		return 0;
#endif

#if KEY_ARITY > 4
#if KEY5_TYPE == STRING
	if (key5 == NULL)
		return 0;
#endif

#if KEY_ARITY > 5
#if KEY6_TYPE == STRING
	if (key6 == NULL)
		return 0;
#endif

#if KEY_ARITY > 6
#if KEY7_TYPE == STRING
	if (key7 == NULL)
		return 0;
#endif

#if KEY_ARITY > 7
#if KEY8_TYPE == STRING
	if (key8 == NULL)
		return 0;
#endif

#if KEY_ARITY > 8
#if KEY9_TYPE == STRING
	if (key9 == NULL)
		return 0;
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
	return 1;
}

static unsigned int KEYSYM(hash) (ALLKEYSD(key))
{
	unsigned int hash = KEY1_HASH(key1);
#if KEY_ARITY > 1
	hash = (hash << 1) ^ KEY2_HASH(key2);
#if KEY_ARITY > 2
	hash = (hash << 1) ^ KEY3_HASH(key3);
#if KEY_ARITY > 3
	hash = (hash << 1) ^ KEY4_HASH(key4);
#if KEY_ARITY > 4
	hash = (hash << 1) ^ KEY5_HASH(key5);
#if KEY_ARITY > 5
	hash = (hash << 1) ^ KEY6_HASH(key6);
#if KEY_ARITY > 6
	hash = (hash << 1) ^ KEY7_HASH(key7);
#if KEY_ARITY > 7
	hash = (hash << 1) ^ KEY8_HASH(key8);
#if KEY_ARITY > 8
	hash = (hash << 1) ^ KEY9_HASH(key9);
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
	return (unsigned int) (hash % HASH_TABLE_SIZE);
}


#if VALUE_TYPE == INT64 || VALUE_TYPE == STRING
static MAP KEYSYM(_stp_map_new) (unsigned max_entries)
{
	MAP m = _stp_map_new (max_entries, VALUE_TYPE, sizeof(struct KEYSYM(map_node)), 0);
	if (m)
		m->get_key = KEYSYM(map_get_key);
	return m;
}
#else
/* _stp_map_new_key1_key2...val (num, HIST_LINEAR, start, end, interval) */
/* _stp_map_new_key1_key2...val (num, HIST_LOG) */ 

static MAP KEYSYM(_stp_map_new) (unsigned max_entries, int htype, ...)
{
	int start=0, stop=0, interval=0;
	MAP m;

	if (htype == HIST_LINEAR) {
		va_list ap;
		va_start (ap, htype);		
		start = va_arg(ap, int);
		stop = va_arg(ap, int);
		interval = va_arg(ap, int);
		va_end (ap);
	}

	switch (htype) {
	case HIST_NONE:
		m = _stp_map_new (max_entries, STAT, sizeof(struct KEYSYM(map_node)), 0);
		break;
	case HIST_LOG:
		m = _stp_map_new_hstat_log (max_entries, sizeof(struct KEYSYM(map_node)));
		break;
	case HIST_LINEAR:
		m = _stp_map_new_hstat_linear (max_entries, sizeof(struct KEYSYM(map_node)),
					       start, stop, interval);
		break;
	default:
		_stp_warn ("Unknown histogram type %d\n", htype);
		m = NULL;
	}

	if (m)
		m->get_key = KEYSYM(map_get_key);

	return m;
}

#endif /* VALUE_TYPE */
static int KEYSYM(__stp_map_set) (MAP map, ALLKEYSD(key), VSTYPE val, int add)
{
	unsigned int hv;
	struct hlist_head *head;
	struct hlist_node *e;
	struct KEYSYM(map_node) *n;

	if (map == NULL)
		return -2;

	if (KEYSYM(keycheck) (ALLKEYS(key)) == 0)
		return -2;

	hv = KEYSYM(hash) (ALLKEYS(key));
	head = &map->hashes[hv];

	hlist_for_each(e, head) {
		n = (struct KEYSYM(map_node) *)((long)e - sizeof(struct list_head));
		if (KEY1_EQ_P(n->key1, key1)
#if KEY_ARITY > 1
		    && KEY2_EQ_P(n->key2, key2)
#if KEY_ARITY > 2
		    && KEY3_EQ_P(n->key3, key3)
#if KEY_ARITY > 3
		    && KEY4_EQ_P(n->key4, key4)
#if KEY_ARITY > 4
		    && KEY5_EQ_P(n->key5, key5)
#if KEY_ARITY > 5
		    && KEY6_EQ_P(n->key6, key6)
#if KEY_ARITY > 6
		    && KEY7_EQ_P(n->key7, key7)
#if KEY_ARITY > 7
		    && KEY8_EQ_P(n->key8, key8)
#if KEY_ARITY > 8
		    && KEY9_EQ_P(n->key9, key9)
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
			) {
			return MAP_SET_VAL(map,(struct map_node *)n, val, add);
		}
	}
	/* key not found */
	n = (struct KEYSYM(map_node)*)_new_map_create (map, head);
	if (n == NULL)
		return -1;
	KEYCPY(n);
	return MAP_SET_VAL(map,(struct map_node *)n, val, 0);
}

static int KEYSYM(_stp_map_set) (MAP map, ALLKEYSD(key), VSTYPE val)
{
	return KEYSYM(__stp_map_set) (map, ALLKEYS(key), val, 0);
}

static int KEYSYM(_stp_map_add) (MAP map, ALLKEYSD(key), VSTYPE val)
{
	return KEYSYM(__stp_map_set) (map, ALLKEYS(key), val, 1);
}


static VALTYPE KEYSYM(_stp_map_get) (MAP map, ALLKEYSD(key))
{
	unsigned int hv;
	struct hlist_head *head;
	struct hlist_node *e;
	struct KEYSYM(map_node) *n;

	if (map == NULL)
		return NULLRET;

	hv = KEYSYM(hash) (ALLKEYS(key));
	head = &map->hashes[hv];

	hlist_for_each(e, head) {
		n = (struct KEYSYM(map_node) *)((long)e - sizeof(struct list_head));
		if (KEY1_EQ_P(n->key1, key1)
#if KEY_ARITY > 1
		    && KEY2_EQ_P(n->key2, key2)
#if KEY_ARITY > 2
		    && KEY3_EQ_P(n->key3, key3)
#if KEY_ARITY > 3
		    && KEY4_EQ_P(n->key4, key4)
#if KEY_ARITY > 4
		    && KEY5_EQ_P(n->key5, key5)
#if KEY_ARITY > 5
		    && KEY6_EQ_P(n->key6, key6)
#if KEY_ARITY > 6
		    && KEY7_EQ_P(n->key7, key7)
#if KEY_ARITY > 7
		    && KEY8_EQ_P(n->key8, key8)
#if KEY_ARITY > 8
		    && KEY9_EQ_P(n->key9, key9)
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
			) {
			return MAP_GET_VAL((struct map_node *)n);
		}
	}
	/* key not found */
	return NULLRET;
}

static int KEYSYM(_stp_map_del) (MAP map, ALLKEYSD(key))
{
	unsigned int hv;
	struct hlist_head *head;
	struct hlist_node *e;
	struct KEYSYM(map_node) *n;

	if (map == NULL)
		return -1;

	hv = KEYSYM(hash) (ALLKEYS(key));
	head = &map->hashes[hv];

	hlist_for_each(e, head) {
		n = (struct KEYSYM(map_node) *)((long)e - sizeof(struct list_head));
		if (KEY1_EQ_P(n->key1, key1)
#if KEY_ARITY > 1
		    && KEY2_EQ_P(n->key2, key2)
#if KEY_ARITY > 2
		    && KEY3_EQ_P(n->key3, key3)
#if KEY_ARITY > 3
		    && KEY4_EQ_P(n->key4, key4)
#if KEY_ARITY > 4
		    && KEY5_EQ_P(n->key5, key5)
#if KEY_ARITY > 5
		    && KEY6_EQ_P(n->key6, key6)
#if KEY_ARITY > 6
		    && KEY7_EQ_P(n->key7, key7)
#if KEY_ARITY > 7
		    && KEY8_EQ_P(n->key8, key8)
#if KEY_ARITY > 8
		    && KEY9_EQ_P(n->key9, key9)
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
			) {
			_new_map_del_node(map,(struct map_node *)n);
			return 0;
		}
	}
	/* key not found */
	return 0;
}

static int KEYSYM(_stp_map_exists) (MAP map, ALLKEYSD(key))
{
	unsigned int hv;
	struct hlist_head *head;
	struct hlist_node *e;
	struct KEYSYM(map_node) *n;

	if (map == NULL)
		return 0;

	hv = KEYSYM(hash) (ALLKEYS(key));
	head = &map->hashes[hv];

	hlist_for_each(e, head) {
		n = (struct KEYSYM(map_node) *)((long)e - sizeof(struct list_head));
		if (KEY1_EQ_P(n->key1, key1)
#if KEY_ARITY > 1
		    && KEY2_EQ_P(n->key2, key2)
#if KEY_ARITY > 2
		    && KEY3_EQ_P(n->key3, key3)
#if KEY_ARITY > 3
		    && KEY4_EQ_P(n->key4, key4)
#if KEY_ARITY > 4
		    && KEY5_EQ_P(n->key5, key5)
#if KEY_ARITY > 5
		    && KEY6_EQ_P(n->key6, key6)
#if KEY_ARITY > 6
		    && KEY7_EQ_P(n->key7, key7)
#if KEY_ARITY > 7
		    && KEY8_EQ_P(n->key8, key8)
#if KEY_ARITY > 8
		    && KEY9_EQ_P(n->key9, key9)
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
			) {
			return 1;
		}
	}
	/* key not found */
	return 0;
}

#undef KEY1NAME
#undef KEY1N
#undef KEY1TYPE
#undef KEY1_TYPE
#undef KEY1STOR
#undef KEY1CPY

#undef KEY2NAME
#undef KEY2N
#undef KEY2TYPE
#undef KEY2_TYPE
#undef KEY2STOR
#undef KEY2CPY

#undef KEY3NAME
#undef KEY3N
#undef KEY3TYPE
#undef KEY3_TYPE
#undef KEY3STOR
#undef KEY3CPY

#undef KEY4NAME
#undef KEY4N
#undef KEY4TYPE
#undef KEY4_TYPE
#undef KEY4STOR
#undef KEY4CPY

#undef KEY5NAME
#undef KEY5N
#undef KEY5TYPE
#undef KEY5_TYPE
#undef KEY5STOR
#undef KEY5CPY

#undef KEY6NAME
#undef KEY6N
#undef KEY6TYPE
#undef KEY6_TYPE
#undef KEY6STOR
#undef KEY6CPY

#undef KEY7NAME
#undef KEY7N
#undef KEY7TYPE
#undef KEY7_TYPE
#undef KEY7STOR
#undef KEY7CPY

#undef KEY8NAME
#undef KEY8N
#undef KEY8TYPE
#undef KEY8_TYPE
#undef KEY8STOR
#undef KEY8CPY

#undef KEY9NAME
#undef KEY9N
#undef KEY9TYPE
#undef KEY9_TYPE
#undef KEY9STOR
#undef KEY9CPY

#undef KEY_ARITY
#undef ALLKEYS
#undef ALLKEYSD
#undef KEYCPY
#undef KEYSYM 

#undef VALUE_TYPE
#undef VALNAME
#undef VALTYPE
#undef VSTYPE
#undef VALN

#undef MAP_SET_VAL
#undef MAP_GET_VAL
#undef NULLRET
