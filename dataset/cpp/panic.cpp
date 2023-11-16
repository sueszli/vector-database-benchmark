
#include <os.hpp>
#include <kernel.hpp>
#include <kernel/elf.hpp>
#include <system_log>
#include <statman>
#include <kprint>
#include <smp>

#if defined (UNITTESTS) && !defined(__MACH__)
#define THROW throw()
#else
#define THROW
#endif

// Emitted if and only if panic (unrecoverable system wide error) happens
static const char* panic_signature = "\x15\x07\t**** PANIC ****";
extern uintptr_t heap_begin;
extern uintptr_t heap_end;

struct alignas(SMP_ALIGN) context_buffer
{
  std::array<char, 512> buffer;
};
static std::vector<context_buffer> contexts;
SMP_RESIZE_EARLY_GCTOR(contexts);

// NOTE: panics cannot be per-cpu because it might not be ready yet
// NOTE: it's also used by OS::is_panicking(), used by OS::print(...)

size_t get_crash_context_length()
{
  return PER_CPU(contexts).buffer.size();
}
char*  get_crash_context_buffer()
{
  return PER_CPU(contexts).buffer.data();
}

extern "C"
void cpu_enable_panicking()
{
  //PER_CPU(contexts).panics++;
  __sync_fetch_and_add(&kernel::state().panics, 1);
}

static os::on_panic_func panic_handler = nullptr;
void os::on_panic(on_panic_func func)
{
  panic_handler = std::move(func);
}

extern "C" void double_fault(const char* why);
__attribute__((noreturn)) static void panic_epilogue(const char*);

namespace net {
  __attribute__((weak)) void print_last_packet() {}
}

extern kernel::ctor_t __plugin_ctors_start;
extern kernel::ctor_t __plugin_ctors_end;

/**
 * panic:
 * Display reason for kernel panic
 * Display last crash context value, if it exists
 * Display no-heap backtrace of stack
 * Call on_panic handler function, to allow application specific panic behavior
 * Print EOT character to stderr, to signal outside that PANIC output completed
 * If the handler returns, go to (permanent) sleep
**/
void os::panic(const char* why) noexcept
{
  cpu_enable_panicking();
  if (kernel::panics() > 4) double_fault(why);

  const int current_cpu = SMP::cpu_id();

  SMP::global_lock();

  // Tell the System log that we have paniced
  SystemLog::set_flags(SystemLog::PANIC);

  /// display informacion ...
  fprintf(stderr, "\n%s\nCPU: %d, Reason: %s\n",
          panic_signature, current_cpu, why);

  // crash context (can help determine source of crash)
  const int len = strnlen(get_crash_context_buffer(), get_crash_context_length());
  if (len > 0) {
    printf("\n\t**** CPU %d CONTEXT: ****\n %*s\n\n",
        current_cpu, len, get_crash_context_buffer());
  }

  // heap info
  typedef unsigned long ulong;
  uintptr_t heap_total = kernel::heap_max() - kernel::heap_begin();
  fprintf(stderr, "Heap is at: %p / %p  (diff=%lu)\n",
         (void*) kernel::heap_end(), (void*) kernel::heap_max(), (ulong) (kernel::heap_max() - kernel::heap_end()));
  fprintf(stderr, "Heap area: %lu / %lu Kb (allocated %zu kb)\n", // (%.2f%%)\n",
         (ulong) (kernel::heap_end() - kernel::heap_begin()) / 1024,
          (ulong) heap_total / 1024, kernel::heap_usage() / 1024); //, total * 100.0);
  fprintf(stderr, "Total memory use: ~%zu%% (%zu of %zu b)\n",
          util::bits::upercent(os::total_memuse(), kernel::memory_end()), os::total_memuse(), kernel::memory_end());

  // print plugins
  fprintf(stderr, "*** Found %u plugin constructors:\n",
          uint32_t(&__plugin_ctors_end - &__plugin_ctors_start));
  for (kernel::ctor_t* ptr = &__plugin_ctors_start; ptr < &__plugin_ctors_end; ptr++)
  {
    char buffer[4096];
    auto res = Elf::safe_resolve_symbol((void*) *ptr, buffer, sizeof(buffer));
    fprintf(stderr, "Plugin: %s (%p)\n", res.name, (void*) res.addr);
  }

  // last packet
  net::print_last_packet();

  // finally, backtrace
  fprintf(stderr, "\n*** Backtrace:");
  print_backtrace([] (const char* text, size_t len) {
    fprintf(stderr, "%.*s", (int) len, text);
  });

  fflush(stderr);
  SMP::global_unlock();

  panic_epilogue(why);
}

extern "C"
void double_fault(const char* why)
{
  SMP::global_lock();
  fprintf(stderr, "\n\n%s\nDouble fault! \nCPU: %d, Reason: %s\n",
          panic_signature, SMP::cpu_id(), why);
  SMP::global_unlock();

  panic_epilogue(why);
}



void panic_epilogue(const char* why)
{
  // Call custom on panic handler (if present).
  if (panic_handler != nullptr) {
    // Avoid recursion if the panic handler results in panic
    auto final_action = panic_handler;
    panic_handler = nullptr;
    final_action(why);
  }

#if defined(ARCH_x86)
  SMP::global_lock();
  // Signal End-Of-Transmission
  kprint("\x04");
  SMP::global_unlock();

#else
#warning "panic() handler not implemented for selected arch"
#endif

  switch (os::panic_action())
  {
  case os::Panic_action::halt:
    while (1) os::halt();
  case os::Panic_action::shutdown:
    extern __attribute__((noreturn)) void __arch_poweroff();
    __arch_poweroff();
    [[fallthrough]]; // needed for g++ bug
  case os::Panic_action::reboot:
    os::reboot();
  }

  __builtin_unreachable();
}

void __expect_fail(const char *expr, const char *file, int line, const char *func)
{
  SMP::global_lock();
  fprintf(stderr, "%s:%i:%s\n>>> %s\n",file, line, func, expr);
  fflush(NULL);
  SMP::global_unlock();
  os::panic(expr);
}

// Shutdown the machine when one of the exit functions are called
void kernel::default_exit() {
  __arch_poweroff();
  __builtin_unreachable();
}

extern "C"
void _init_syscalls()
{
  // make sure each buffer is zero length so it won't always show up in crashes
  for (auto& ctx : contexts)
      ctx.buffer[0] = 0;
}
