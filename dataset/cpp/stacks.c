/**
 *
 * Phantom OS
 *
 * Copyright (C) 2005-2008 Dmitry Zavalishin, dz@dz.ru
 *
 * Virtual machine stacks implementation.
 *
 *
**/


#include "vm/internal.h"
#include "vm/internal_da.h"
#include "vm/exception.h"
#include "vm/alloc.h"
#include "vm/exec.h"

// Ok. All these methods receive ptr to root stack object da,
// which curr_da field points to the active page data area.
// so rootda is root page da, and s is curr page da.


#define   	page_push(v) 		do { s->stack[s->common.free_cell_ptr++] = v; } while(0)
#define 	page_pop() 		s->stack[--(s->common.free_cell_ptr)]
#define		page_top() 		s->stack[(s->common.free_cell_ptr)-1]

#define 	page_is_empty()		(s->common.free_cell_ptr == 0)
#define 	page_is_full() 		(s->common.free_cell_ptr >= s->common.__sSize)

#define    	no_prev() 		pvm_is_null( s->common.prev )
#define 	no_next() 		pvm_is_null( s->common.next )

#define    	set_me(to) 		do { rootda->common.curr = to; rootda->curr_da = s = (void *)&(to->da); } while(0)


#define check_underflow()     \
    do {                      \
        if( page_is_empty() ) \
        {                     \
            if( no_prev() ) pvm_exec_panic0( "stack underflow" ); \
            set_me( s->common.prev ); \
        }                             \
    } while(0);

#define check_overflow()                             \
    do {                                             \
        if( page_is_full() )                         \
        {                                            \
            if( no_next() )                          \
                {                                    \
                s->common.next = make();             \
                set_next_prev();     \
                }                                    \
            set_me( s->common.next );                \
        }                                            \
    } while(0);




/**
 *
 * Object stack goes
 *
**/

#define make()  pvm_create_ostack_object()
#define set_next_prev()  { pvm_object_da(s->common.next, object_stack)->common.prev = rootda->common.curr; pvm_object_da(s->common.next, object_stack)->common.root = rootda->common.root; }

void pvm_ostack_push( struct data_area_4_object_stack* rootda, pvm_object_t o )
{
    struct data_area_4_object_stack* s = rootda->curr_da;
    check_overflow();
    if( page_is_full() ) panic("opush page full after mkpage");
    page_push(o);
}

// push nulls to reserve stack space
void pvm_ostack_reserve( struct data_area_4_object_stack* rootda, int n_slots ) 
{
    struct data_area_4_object_stack* s = rootda->curr_da;

    pvm_object_t zero = 0;

    while( n_slots-- > 0 )
    {
        check_overflow();
        if( page_is_full() ) panic("opush page full after mkpage");

        s->stack[s->common.free_cell_ptr++] = zero;
    }
}

pvm_object_t pvm_ostack_pop( struct data_area_4_object_stack* rootda )
{
    struct data_area_4_object_stack* s = rootda->curr_da;
    check_underflow();
    if( page_is_empty() ) panic("opop page empty");
    return page_pop();
}

pvm_object_t pvm_ostack_top( struct data_area_4_object_stack* rootda )
{
    struct data_area_4_object_stack* s = rootda->curr_da;
    check_underflow();
    if( page_is_empty() ) panic("otop page empty");
    return page_top();
}

int pvm_ostack_empty( struct data_area_4_object_stack* rootda )
{
    struct data_area_4_object_stack* s = rootda->curr_da;
    return page_is_empty() && no_prev();
}


void pvm_ostack_abs_set( struct data_area_4_object_stack* rootda, int abs_pos, pvm_object_t val )
{
    unsigned int pagesize = rootda->common.__sSize;
    pvm_object_t c = rootda->common.root;

    while( abs_pos >= pagesize )
    {
        c = pvm_object_da(c,object_stack)->common.next;
        if( pvm_is_null(c) ) pvm_exec_panic0( "o abs_set: out of stack" );
        abs_pos -= pagesize;
    }

    //TODO: assert should be here instead of decrement - it is an error in bytecode compiler/implementation
    if (pvm_object_da(c,object_stack)->stack[abs_pos] != 0) ref_dec_o( pvm_object_da(c,object_stack)->stack[abs_pos] ); //decr prev value - avoid memory leak

    pvm_object_da(c,object_stack)->stack[abs_pos] = val;
}

pvm_object_t pvm_ostack_abs_get( struct data_area_4_object_stack* rootda, int abs_pos )
{
    unsigned int pagesize = rootda->common.__sSize;
    pvm_object_t c = rootda->common.root;

    while( abs_pos >= pagesize )
    {
        c = pvm_object_da(c,object_stack)->common.next;
        if( pvm_is_null(c) ) pvm_exec_panic0( "o abs_get: out of stack" );
        abs_pos -= pagesize;
    }

    return pvm_object_da(c,object_stack)->stack[abs_pos];
}

/**
 * 
 * \brief Dig element from stack.
 * 
 * \param[in]  rootda    Root stack page
 * \param[in]  depth     Distance to object we need from stack top, 0 = last element on stack.
 * 
**/
pvm_object_t  pvm_ostack_pull( struct data_area_4_object_stack* rootda, int depth )
{
    if( depth < 0 ) pvm_exec_panic0( "stack pull: overflow" );

    struct data_area_4_object_stack* s = rootda->curr_da;
    // steps up to needed slot from the bottom of the current page
    int displ = s->common.free_cell_ptr - depth - 1;

    pvm_object_t c = s->common.curr;

    while( displ < 0 )
    {
        c = pvm_object_da(c,object_stack)->common.prev;
        if( pvm_is_null(c) ) pvm_exec_panic0( "stack pull: underflow" );
        displ += s->common.__sSize;
    }

    return pvm_object_da(c,object_stack)->stack[displ];
}

// Return number of elements in stack
int pvm_ostack_count( struct data_area_4_object_stack* rootda )
{
    struct data_area_4_object_stack* s = rootda->curr_da;
    
    int count = s->common.free_cell_ptr;

    pvm_object_t c = s->common.curr;

    while( 1 )
    {
        c = pvm_object_da(c,object_stack)->common.prev;
        if( pvm_is_null(c) ) 
        break;
        count += s->common.__sSize;
    }

    return count;    
}

/**
 *
 * Integer stack goes
 *
**/

#undef make
#undef set_next_prev
#define make()  pvm_create_istack_object()
#define set_next_prev()  { pvm_object_da(s->common.next, integer_stack)->common.prev = rootda->common.curr; pvm_object_da(s->common.next, integer_stack)->common.root = rootda->common.root; }



void pvm_istack_push( struct data_area_4_integer_stack* rootda, int o )
{
    struct data_area_4_integer_stack* s = rootda->curr_da;
    check_overflow();
    page_push(o);
}

int pvm_istack_pop( struct data_area_4_integer_stack* rootda )
{
    struct data_area_4_integer_stack* s = rootda->curr_da;
    check_underflow();
    return page_pop();
}

int pvm_istack_top( struct data_area_4_integer_stack* rootda )
{
    struct data_area_4_integer_stack* s = rootda->curr_da;
    check_underflow();
    return page_top();
}

int pvm_istack_empty( struct data_area_4_integer_stack* rootda )
{
    struct data_area_4_integer_stack* s = rootda->curr_da;
    return page_is_empty() && no_prev();
}


// push 0 to reserve stack space
void pvm_istack_reserve( struct data_area_4_integer_stack* rootda, int n_slots ) 
{
    struct data_area_4_integer_stack* s = rootda->curr_da;

    while( n_slots-- > 0 )
    {
        check_overflow();
        if( page_is_full() ) panic("ipush page full after mkpage");

        s->stack[s->common.free_cell_ptr++] = 0;
    }
}




void pvm_istack_abs_set( struct data_area_4_integer_stack* rootda, int abs_pos, int val )
{
    unsigned int pagesize = rootda->common.__sSize;
    pvm_object_t c = rootda->common.root;

    while( abs_pos >= pagesize )
    {
        c = pvm_object_da(c,integer_stack)->common.next;
        if( pvm_is_null(c) ) pvm_exec_panic0( "i abs_set: out of stack" );
        abs_pos -= pagesize;
    }

    pvm_object_da(c,integer_stack)->stack[abs_pos] = val;
}

int pvm_istack_abs_get( struct data_area_4_integer_stack* rootda, int abs_pos )
{
    unsigned int pagesize = rootda->common.__sSize;
    pvm_object_t c = rootda->common.root;

    while( abs_pos >= pagesize )
    {
        c = pvm_object_da(c,integer_stack)->common.next;
        if( pvm_is_null(c) ) pvm_exec_panic0( "i abs_get: out of stack" );
        abs_pos -= pagesize;
    }

    return pvm_object_da(c,integer_stack)->stack[abs_pos];
}




/**
 *
 * Long stack goes - it's integer stack, but 64 bit access.
 * 
 * Least significant bits are above (near stack end).
 *
**/


#if 1
void pvm_lstack_push( struct data_area_4_integer_stack* rootda, int64_t o )
{
    //printf("lpush %lld; \n", o);
    pvm_istack_push( rootda, (int)(o >> 32));
    pvm_istack_push( rootda, (int)o);
}

int64_t pvm_lstack_pop( struct data_area_4_integer_stack* rootda )
{
    int64_t o;
    o = pvm_istack_pop( rootda );
    o |= ((int64_t)pvm_istack_pop( rootda )) << 32;
    return o;
}

int64_t  pvm_lstack_top( struct data_area_4_integer_stack* rootda )
{
    int64_t o;

    int low = pvm_istack_pop( rootda );
    int hi = pvm_istack_top( rootda );
    pvm_istack_push( rootda, low );    

    o = low; o |= ((int64_t)hi) << 32;
    return o;
}


void pvm_lstack_abs_set( struct data_area_4_integer_stack* rootda, int abs_pos, int64_t val )
{
    unsigned int pagesize = rootda->common.__sSize;
    pvm_object_t c = rootda->common.root;

    while( abs_pos >= pagesize )
    {
        c = pvm_object_da(c,integer_stack)->common.next;
        if( pvm_is_null(c) ) pvm_exec_panic0( "l abs_set: out of stack" );
        abs_pos -= pagesize;
    }

    pvm_object_da(c,integer_stack)->stack[abs_pos]   = (int32_t)(val >> 32);

    abs_pos++;

    if( abs_pos >= pagesize )
    {
        c = pvm_object_da(c,integer_stack)->common.next;
        if( pvm_is_null(c) ) pvm_exec_panic0( "l abs_set: out of stack" );
        abs_pos -= pagesize;
    }

    pvm_object_da(c,integer_stack)->stack[abs_pos] = (int32_t)val;
}

int64_t pvm_lstack_abs_get( struct data_area_4_integer_stack* rootda, int abs_pos )
{
    unsigned int pagesize = rootda->common.__sSize;
    pvm_object_t c = rootda->common.root;

    while( abs_pos >= pagesize )
    {
        c = pvm_object_da(c,integer_stack)->common.next;
        if( pvm_is_null(c) ) pvm_exec_panic0( "l abs_get: out of stack" );
        abs_pos -= pagesize;
    }

    int hi = pvm_object_da(c,integer_stack)->stack[abs_pos];

    abs_pos++;

    if( abs_pos >= pagesize )
    {
        c = pvm_object_da(c,integer_stack)->common.next;
        if( pvm_is_null(c) ) pvm_exec_panic0( "l abs_set: out of stack" );
        abs_pos -= pagesize;
    }

    int lo = pvm_object_da(c,integer_stack)->stack[abs_pos];

    return (((int64_t)hi) << 32) | lo;
    }


#else

// NB! We don't redefine - re-use int stack defs above!

//#undef make
//#undef set_next_prev
//#define make()  pvm_create_istack_object()
//#define set_next_prev()  { pvm_object_da(s->common.next, integer_stack)->common.prev = rootda->common.curr; pvm_object_da(s->common.next, integer_stack)->common.root = rootda->common.root; }

#define   	lpage_push(v) 		do { ((int64_t*)s->stack)[s->common.free_cell_ptr++] = v; } while(0)
#define 	lpage_pop() 		((int64_t*)s->stack)[--(s->common.free_cell_ptr)]
#define		lpage_top() 		((int64_t*)s->stack)[(s->common.free_cell_ptr)-1]

#define 	lpage_is_empty()		(s->common.free_cell_ptr < 2)
// -1 for one more element
#define 	lpage_is_full() 		(s->common.free_cell_ptr >= (s->common.__sSize-1))


#define lcheck_underflow()     \
    do {                      \
        if( lpage_is_empty() ) \
        {                     \
            if( no_prev() ) pvm_exec_panic0( "stack underflow" ); \
            set_me( s->common.prev ); \
        }                             \
    } while(0);

#define lcheck_overflow()                             \
    do {                                             \
        if( lpage_is_full() )                         \
        {                                            \
            if( no_next() )                          \
                {                                    \
                s->common.next = make();             \
                set_next_prev();     \
                }                                    \
            set_me( s->common.next );                \
        }                                            \
    } while(0);


void pvm_lstack_push( struct data_area_4_integer_stack* rootda, int64_t o )
{
    struct data_area_4_integer_stack* s = rootda->curr_da;
    lcheck_overflow();
    lpage_push(o);
}

int64_t pvm_lstack_pop( struct data_area_4_integer_stack* rootda )
{
    struct data_area_4_integer_stack* s = rootda->curr_da;
    lcheck_underflow();
    return lpage_pop();
}

int64_t pvm_lstack_top( struct data_area_4_integer_stack* rootda )
{
    struct data_area_4_integer_stack* s = rootda->curr_da;
    lcheck_underflow();
    return lpage_top();
}
/*
int pvm_istack_empty( struct data_area_4_integer_stack* rootda )
{
    struct data_area_4_integer_stack* s = rootda->curr_da;
    return page_is_empty() && no_prev();
}
*/


/*
void pvm_lstack_abs_set( struct data_area_4_integer_stack* rootda, int abs_pos, int val )
{
    unsigned int pagesize = rootda->common.__sSize;
    pvm_object_t c = rootda->common.root;

    while( abs_pos >= pagesize )
    {
        c = pvm_object_da(c,integer_stack)->common.next;
        if( pvm_is_null(c) ) pvm_exec_panic( "i abs_set: out of stack" );
        abs_pos -= pagesize;
    }

    pvm_object_da(c,integer_stack)->stack[abs_pos] = val;
}

int pvm_istack_abs_get( struct data_area_4_integer_stack* rootda, int abs_pos )
{
    unsigned int pagesize = rootda->common.__sSize;
    pvm_object_t c = rootda->common.root;

    while( abs_pos >= pagesize )
    {
        c = pvm_object_da(c,integer_stack)->common.next;
        if( pvm_is_null(c) ) pvm_exec_panic( "i abs_get: out of stack" );
        abs_pos -= pagesize;
    }

    return pvm_object_da(c,integer_stack)->stack[abs_pos];
}
*/


#endif












/**
 *
 * Exceptions stack goes
 *
**/

#undef make
#undef set_next_prev
#define make()  pvm_create_estack_object()
#define set_next_prev()  { pvm_object_da(s->common.next, exception_stack)->common.prev = rootda->common.curr; pvm_object_da(s->common.next, exception_stack)->common.root = rootda->common.root; }



void pvm_estack_push( struct data_area_4_exception_stack* rootda, struct pvm_exception_handler e )
{
    struct data_area_4_exception_stack* s = rootda->curr_da;
    check_overflow();
    page_push(e);
}


struct pvm_exception_handler pvm_estack_pop( struct data_area_4_exception_stack* rootda )
{
    struct data_area_4_exception_stack* s = rootda->curr_da;
    check_underflow();
    return page_pop();
}

struct pvm_exception_handler pvm_estack_top( struct data_area_4_exception_stack* rootda )
{
    struct data_area_4_exception_stack* s = rootda->curr_da;
    check_underflow();
    return page_top();
}

int pvm_estack_empty( struct data_area_4_exception_stack* rootda )
{
    struct data_area_4_exception_stack* s = rootda->curr_da;
    return page_is_empty() && no_prev();
}


int e_page_foreach( struct data_area_4_exception_stack* s,
                    void *pass,
                    int (*func)( void *pass, struct pvm_exception_handler *elem ) )
{
    int cell = s->common.free_cell_ptr;
//printf("check %d estack cells", cell+1 );
    while( --cell >= 0 )
        if( func( pass, s->stack+cell ) )
            return 1;
    return 0;
}


int pvm_estack_foreach(
                       struct data_area_4_exception_stack* rootda,
                       void *pass,
                       int (*func)( void *pass, struct pvm_exception_handler *elem ))
{
    struct data_area_4_exception_stack* s = rootda->curr_da;
    pvm_object_t c = s->common.curr;
    while( !pvm_is_null( c ) )
    {
        if( e_page_foreach( pvm_data_area(c,exception_stack), pass, func ) )
            return 1;
        //c = data_area(c)->prev;
        c = pvm_object_da(c,exception_stack)->common.prev;
    }
    return 0;
}


































/**
 *
 * Stack objects creation.
 * NB. We are deliberately do not use classic init by constructor
 * to speedup creation.
 * TODO: some kind of object pools to recycle stack objects?  Just in the thread object.
 *
 * ssize is number of slots in page
**/

static pvm_object_t     pvm_create_general_stack_object(pvm_object_t object_class, int ssize, int da_size )
{
    pvm_object_t ret = pvm_object_create_dynamic( object_class, da_size );

    // We use only common, so any stack da will do
    struct data_area_4_object_stack* sda = pvm_object_da(ret,object_stack);
    sda->common.root = ret;
    sda->common.curr = ret;
    sda->common.prev = 0;
    sda->common.next = 0;
    sda->common.free_cell_ptr = 0;
    sda->common.__sSize = ssize;
    sda->curr_da = (void *)sda;

    return ret;
}


pvm_object_t     pvm_create_istack_object()
{
    return pvm_create_general_stack_object(pvm_get_istack_class(), PVM_INTEGER_STACK_SIZE, sizeof(struct data_area_4_integer_stack) );
}

pvm_object_t     pvm_create_ostack_object()
{
    return pvm_create_general_stack_object(pvm_get_ostack_class(), PVM_OBJECT_STACK_SIZE, sizeof(struct data_area_4_object_stack) );
}

pvm_object_t     pvm_create_estack_object()
{
    return pvm_create_general_stack_object(pvm_get_estack_class(), PVM_EXCEPTION_STACK_SIZE, sizeof(struct data_area_4_exception_stack));
}




void pvm_internal_init_ostack(pvm_object_t os )
{
    struct data_area_4_object_stack* sda = (struct data_area_4_object_stack*) os->da;

    pvm_object_t ret;

    ret = os;
    //ret.interface = 0; // Nobody needs it there anyway

    sda->common.root = ret;  // will update later for nonroot pages
    sda->common.curr = ret;
    sda->common.prev = 0;
    sda->common.next = 0;
    sda->common.free_cell_ptr = 0;
    sda->common.__sSize = PVM_OBJECT_STACK_SIZE;
    sda->curr_da = (void *)sda;
}


#define gc_fcall( f, arg, o )   f( o, arg )

void pvm_gc_iter_ostack(gc_iterator_call_t func, pvm_object_t  os, void *arg)
{
    struct data_area_4_object_stack *da = (struct data_area_4_object_stack *)&(os->da);

    int i, max = da->common.free_cell_ptr;
    for( i = 0; i < max; i++ )
    {
        gc_fcall( func, arg, da->stack[i] );
    }

    if ( da->common.next != 0 )
        gc_fcall( func, arg, da->common.next );  //we are starting from root, so following next is enough. Tail recursion
}


void pvm_internal_init_istack(pvm_object_t os )
{
    struct data_area_4_integer_stack* sda = (struct data_area_4_integer_stack*) os->da;

    pvm_object_t ret;

    ret = os;
    //ret.interface = 0; // Nobody needs it there anyway

    sda->common.root = ret;
    sda->common.curr = ret;
    sda->common.prev = 0;
    sda->common.next = 0;
    sda->common.free_cell_ptr = 0;
    sda->common.__sSize = PVM_INTEGER_STACK_SIZE;
    sda->curr_da = (void *)sda;
}

void pvm_gc_iter_istack(gc_iterator_call_t func, pvm_object_t  os, void *arg)
{
    struct data_area_4_object_stack *da = (struct data_area_4_object_stack *)&(os->da);

    // No objects in the integer stack, but please visit linked list ifself

    if ( da->common.next != 0 )
        gc_fcall( func, arg, da->common.next );  //we are starting from root, so following next is enough. Tail recursion
}


void pvm_internal_init_estack(pvm_object_t os )
{
    struct data_area_4_exception_stack* sda = (struct data_area_4_exception_stack*) os->da;

    pvm_object_t ret;

    ret = os;
    //ret.interface = 0; // Nobody needs it there anyway

    sda->common.root = ret;
    sda->common.curr = ret;
    sda->common.prev = 0;
    sda->common.next = 0;
    sda->common.free_cell_ptr = 0;
    sda->common.__sSize = PVM_EXCEPTION_STACK_SIZE;
    sda->curr_da = (void *)sda;
}


void pvm_gc_iter_estack(gc_iterator_call_t func, pvm_object_t  os, void *arg)
{
    struct data_area_4_exception_stack *da = (struct data_area_4_exception_stack *)&(os->da);

    int i, max = da->common.free_cell_ptr;
    for( i = 0; i < max; i++ )
    {
        gc_fcall( func, arg, da->stack[i].object );
    }

    if ( da->common.next != 0 )
        gc_fcall( func, arg, da->common.next );  //we are starting from root, so following next is enough. Tail recursion
}

