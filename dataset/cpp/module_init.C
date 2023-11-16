/* IBM_PROLOG_BEGIN_TAG                                                   */
/* This is an automatically generated prolog.                             */
/*                                                                        */
/* $Source: src/usr/module_init.C $                                       */
/*                                                                        */
/* OpenPOWER HostBoot Project                                             */
/*                                                                        */
/* Contributors Listed Below - COPYRIGHT 2011,2019                        */
/* [+] International Business Machines Corp.                              */
/*                                                                        */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/*                                                                        */
/* IBM_PROLOG_END_TAG                                                     */

#include <stdint.h>

void call_dtors(void * i_dso_handle);

/**
 *  @note: See documentation in src/lib/tls.C
 */
void __tls_register(void * tls_start, void * tls_end,const size_t i_module);

void __tls_unregister(void * tls_start, void * tls_end);

// This identifies the module
void*   __dso_handle = (void*) &__dso_handle;

// TLS Module ID to be updated by custom linker during final binary creation.
// Set to max ID so that if it's not replaced, it will lead to obvious crashes.
size_t TLS_MODULE_ID = 0xFFFFFFFFFFFFFFFFULL;

extern "C"
void _init(void*)
{
    // Register thread-local storage.
    extern void* tls_start_address;
    extern void* tls_end_address;
    __tls_register(&tls_start_address, &tls_end_address, TLS_MODULE_ID);

    // Call default constructors for any static objects.
    extern void (*ctor_start_address)();
    extern void (*ctor_end_address)();
    void(**ctors)() = &ctor_start_address;
    while(ctors != &ctor_end_address)
    {
	(*ctors)();
	ctors++;
    }
}

extern "C"
void _fini(void)
{
    call_dtors(__dso_handle);

    extern void* tls_start_address;
    extern void* tls_end_address;
    __tls_unregister(&tls_start_address, &tls_end_address);
}



