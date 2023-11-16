/**
 *
 * Phantom OS
 *
 * Copyright (C) 2005-2019 Dmitry Zavalishin, dz@dz.ru
 *
 * .internal.udp class implementation
 * 
 * See <https://github.com/dzavalishin/phantomuserland/wiki/InternalClasses>
 * See <https://github.com/dzavalishin/phantomuserland/wiki/InternalMethodWritingGuide>
 *
**/


#define DEBUG_MSG_PREFIX "vm.sysc.udp"
#include <debug_ext.h>
#define debug_level_flow 0
#define debug_level_error 1
#define debug_level_info 0


#include <phantom_libc.h>
#include <vm/syscall_net.h>
#include <vm/alloc.h>
#include <kernel/snap_sync.h>
#include <kernel/net.h>
#include <kernel/json.h>

#include <errno.h>


static int debug_print = 0;





static int si_udp_tostring_5( pvm_object_t me, pvm_object_t *ret, struct data_area_4_thread *tc, int n_args, pvm_object_t *args )
{
    (void) me;
    DEBUG_INFO;
    SYSCALL_RETURN( pvm_create_string_object( "udp socket" ));
}







static int si_udp_send_19( pvm_object_t me, pvm_object_t *ret, struct data_area_4_thread *tc, int n_args, pvm_object_t *args )
{
    (void) me;
    //struct data_area_4_udp      *da = pvm_data_area( me, udp );

    DEBUG_INFO;
    

    CHECK_PARAM_COUNT(2);


    SYSCALL_THROW_STRING( "not implemented" );
}




static int si_udp_sendto_25( pvm_object_t me, pvm_object_t *ret, struct data_area_4_thread *tc, int n_args, pvm_object_t *args )
{
    (void) me;
    //struct data_area_4_udp      *da = pvm_data_area( me, udp );

    DEBUG_INFO;
    

    CHECK_PARAM_COUNT(2);


    SYSCALL_THROW_STRING( "not implemented" );
}







static int si_udp_recv_21( pvm_object_t me, pvm_object_t *ret, struct data_area_4_thread *tc, int n_args, pvm_object_t *args )
{
    (void) me;
    //struct data_area_4_udp      *da = pvm_data_area( me, udp );

    DEBUG_INFO;

    CHECK_PARAM_COUNT(2);

    SYSCALL_THROW_STRING( "not implemented" );
}





static int si_udp_recvfrom_23( pvm_object_t me, pvm_object_t *ret, struct data_area_4_thread *tc, int n_args, pvm_object_t *args )
{
    (void) me;
    //struct data_area_4_udp      *da = pvm_data_area( me, udp );

    DEBUG_INFO;

    CHECK_PARAM_COUNT(2);

    SYSCALL_THROW_STRING( "not implemented" );
}







static int si_udp_bind_16( pvm_object_t me, pvm_object_t *ret, struct data_area_4_thread *tc, int n_args, pvm_object_t *args )
{
    (void) me;
    //struct data_area_4_udp      *da = pvm_data_area( me, udp );

    DEBUG_INFO;

    CHECK_PARAM_COUNT(1);
    int addr = AS_INT(args[0]);
    (void) addr;

    //SYSCALL_PUT_THIS_THREAD_ASLEEP()

    SYSCALL_THROW_STRING( "not implemented" );
}












syscall_func_t  syscall_table_4_udp[26] =
{
    &si_void_0_construct,               &si_void_1_destruct,
    &si_void_2_class,                   &si_void_3_clone,
    &si_void_4_equals,                  &si_udp_tostring_5,
    &si_void_6_toXML,                   &si_void_7_fromXML,
    // 8
    &invalid_syscall,                   &invalid_syscall,
    &invalid_syscall,                   &invalid_syscall,
    &invalid_syscall,                   &invalid_syscall,
    &invalid_syscall,                   &si_void_15_hashcode,
    // 16
    &si_udp_bind_16,                    &invalid_syscall,
    &invalid_syscall,                   &si_udp_send_19,
    &invalid_syscall,                   &si_udp_recv_21,
    // 24
    &invalid_syscall,                   &si_udp_recvfrom_23,
    &invalid_syscall,                   &si_udp_sendto_25,

};
DECLARE_SIZE(udp);

















void pvm_internal_init_udp(pvm_object_t os)
{
    struct data_area_4_udp      *da = (struct data_area_4_udp *)os->da;

    (void) da;

    da->connected = 0;

}

pvm_object_t     pvm_create_udp_object(void)
{
    return pvm_create_object( pvm_get_udp_class() );
}

void pvm_gc_iter_udp(gc_iterator_call_t func, pvm_object_t  os, void *arg)
{
    struct data_area_4_udp      *da = (struct data_area_4_udp *)os->da;

    (void) da;

    //gc_fcall( func, arg, ot );
    //gc_fcall( func, arg, da->p_kernel_state_object );
    //gc_fcall( func, arg, da->callback );
}


void pvm_gc_finalizer_udp( pvm_object_t  os )
{
    // is it called?
    struct data_area_4_udp *da = (struct data_area_4_udp *)os->da;
    (void) da;
}

/* unused
void pvm_restart_udp( pvm_object_t o )
{
    struct data_area_4_udp *da = pvm_object_da( o, udp );

    //da->connected = 0;
    if( da->connected )
    {
        printf("restarting TCP - unimpl!");
    }

}
*/
