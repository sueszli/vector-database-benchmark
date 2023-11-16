#include <l4/sys/ipc.h>
#include <l4/re/env>

extern "C" void _exit(int code) noexcept __attribute__ ((__noreturn__, __weak__));

void _exit(int code) noexcept
{
  L4Re::Env const *e;
  if (l4re_global_env && (e = L4Re::Env::env()) && e->parent().is_valid())
    e->parent()->signal(0, code);
  for (;;)
    l4_ipc_sleep(L4_IPC_NEVER);
}
