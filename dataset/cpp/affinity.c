// file: source/threading/affinity.c


#if defined(ZPL_SYSTEM_MACOS)
#    include <sys/sysctl.h>
#endif

ZPL_BEGIN_C_DECLS

#if defined(ZPL_SYSTEM_WINDOWS) || defined(ZPL_SYSTEM_CYGWIN)

    void zpl_affinity_init(zpl_affinity *a) {
        SYSTEM_LOGICAL_PROCESSOR_INFORMATION *start_processor_info = NULL;
        DWORD length = 0;
        zpl_b32 result  = GetLogicalProcessorInformation(NULL, &length);

        zpl_zero_item(a);

        if (!result && GetLastError() == 122l /*ERROR_INSUFFICIENT_BUFFER*/ && length > 0) {
            start_processor_info = cast(SYSTEM_LOGICAL_PROCESSOR_INFORMATION *)zpl_alloc(zpl_heap_allocator(), length);
            result = GetLogicalProcessorInformation(start_processor_info, &length);
            if (result) {
                SYSTEM_LOGICAL_PROCESSOR_INFORMATION *end_processor_info, *processor_info;

                a->is_accurate  = true;
                a->core_count   = 0;
                a->thread_count = 0;
                end_processor_info = cast(SYSTEM_LOGICAL_PROCESSOR_INFORMATION *)zpl_pointer_add(start_processor_info, length);

                for (processor_info = start_processor_info;
                     processor_info < end_processor_info;
                     processor_info++) {
                    if (processor_info->Relationship == RelationProcessorCore) {
                        zpl_isize thread = zpl_count_set_bits(processor_info->ProcessorMask);
                        if (thread == 0) {
                            a->is_accurate = false;
                        } else if (a->thread_count + thread > ZPL_WIN32_MAX_THREADS) {
                            a->is_accurate = false;
                        } else {
                            ZPL_ASSERT(a->core_count <= a->thread_count &&
                                       a->thread_count < ZPL_WIN32_MAX_THREADS);
                            a->core_masks[a->core_count++] = processor_info->ProcessorMask;
                            a->thread_count += thread;
                        }
                    }
                }
            }

            zpl_free(zpl_heap_allocator(), start_processor_info);
        }

        ZPL_ASSERT(a->core_count <= a->thread_count);
        if (a->thread_count == 0) {
            a->is_accurate   = false;
            a->core_count    = 1;
            a->thread_count  = 1;
            a->core_masks[0] = 1;
        }

    }

    void zpl_affinity_destroy(zpl_affinity *a) {
        zpl_unused(a);
    }

    zpl_b32 zpl_affinity_set(zpl_affinity *a, zpl_isize core, zpl_isize thread) {
        zpl_usize available_mask, check_mask = 1;
        ZPL_ASSERT(thread < zpl_affinity_thread_count_for_core(a, core));

        available_mask = a->core_masks[core];
        for (;;) {
            if ((available_mask & check_mask) != 0) {
                if (thread-- == 0) {
                    zpl_usize result = SetThreadAffinityMask(GetCurrentThread(), check_mask);
                    return result != 0;
                }
            }
            check_mask <<= 1; // NOTE: Onto the next bit
        }
    }

    zpl_isize zpl_affinity_thread_count_for_core(zpl_affinity *a, zpl_isize core) {
        ZPL_ASSERT(core >= 0 && core < a->core_count);
        return zpl_count_set_bits(a->core_masks[core]);
    }

#elif defined(ZPL_SYSTEM_MACOS)
    void zpl_affinity_init(zpl_affinity *a) {
        zpl_usize count, count_size = zpl_size_of(count);

        a->is_accurate      = false;
        a->thread_count     = 1;
        a->core_count       = 1;
        a->threads_per_core = 1;

        if (sysctlbyname("hw.logicalcpu", &count, &count_size, NULL, 0) == 0) {
            if (count > 0) {
                a->thread_count = count;
                // Get # of physical cores
                if (sysctlbyname("hw.physicalcpu", &count, &count_size, NULL, 0) == 0) {
                    if (count > 0) {
                        a->core_count = count;
                        a->threads_per_core = a->thread_count / count;
                        if (a->threads_per_core < 1)
                            a->threads_per_core = 1;
                        else
                            a->is_accurate = true;
                    }
                }
            }
        }

    }

    void zpl_affinity_destroy(zpl_affinity *a) {
        zpl_unused(a);
    }

    zpl_b32 zpl_affinity_set(zpl_affinity *a, zpl_isize core, zpl_isize thread_index) {
        zpl_isize index;
        thread_t thread;
        thread_affinity_policy_data_t info;
        kern_return_t result;

        ZPL_ASSERT(core < a->core_count);
        ZPL_ASSERT(thread_index < a->threads_per_core);

        index = core * a->threads_per_core + thread_index;
        thread = mach_thread_self();
        info.affinity_tag = cast(integer_t)index;
        result = thread_policy_set(thread, THREAD_AFFINITY_POLICY, cast(thread_policy_t)&info, THREAD_AFFINITY_POLICY_COUNT);
        return result == KERN_SUCCESS;
    }

    zpl_isize zpl_affinity_thread_count_for_core(zpl_affinity *a, zpl_isize core) {
        ZPL_ASSERT(core >= 0 && core < a->core_count);
        return a->threads_per_core;
    }

#elif defined(ZPL_SYSTEM_LINUX) || defined(ZPL_SYSTEM_FREEBSD) || defined(ZPL_SYSTEM_OPENBSD)
    void zpl_affinity_init(zpl_affinity *a) {
        a->core_count       = sysconf(_SC_NPROCESSORS_ONLN);
        a->threads_per_core = 1;

        a->is_accurate      = a->core_count > 0;
        a->core_count       = a->is_accurate ? a->core_count : 1;
        a->thread_count     = a->core_count;
    }

    void zpl_affinity_destroy(zpl_affinity *a) {
        zpl_unused(a);
    }

    zpl_b32 zpl_affinity_set(zpl_affinity * a, zpl_isize core, zpl_isize thread_index) {
        zpl_unused(a);
        zpl_unused(core);
        zpl_unused(thread_index);
        return true;
    }

    zpl_isize zpl_affinity_thread_count_for_core(zpl_affinity *a, zpl_isize core) {
        ZPL_ASSERT(0 <= core && core < a->core_count);
        return a->threads_per_core;
    }

#elif defined(ZPL_SYSTEM_EMSCRIPTEN)
#    error No affinity implementation for Emscripten
#else
#    error TODO: Unknown system
#endif

ZPL_END_C_DECLS
