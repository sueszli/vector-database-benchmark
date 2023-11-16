

#include "fieldmedic.hpp"
#include <os>

//#include <atomic>

#define DIAGNOSE(TEST, TEXT, ...) \
  try { printf("%16s[%s] " TEXT "\n","", TEST ? "+" : " ",  ##__VA_ARGS__); } \
  catch (const std::runtime_error& e) {                                 \
    printf("%16s[ ] " TEXT " failed: %s\n", "", e.what()); }

extern "C" char get_single_tbss();
namespace medic{

void init(){
  using namespace diag;
  INFO("Field medic", "Checking vital signs");
  printf(
         "\t         ____n_\n"
         "\t------  | +  |_\\-; ---------\n"
         "\t ====== ;@-----@-'  ===========\n"
         "\t  _______________________________\n\n"
         );

  /* TODO:
     init_tls();
     DIAGNOSE(timers(),     "Timers active");
     DIAGNOSE(elf(),        "ELF binary intact");
     DIAGNOSE(virtmem(),    "Virtual memory active");
     DIAGNOSE(heap(),       "Heap fragments intact");
     DIAGNOSE(tls(),        "Thread local storage intact");
  */

  DIAGNOSE(stack(),      "Stack check");
  DIAGNOSE(exceptions(), "Exceptions test");

  INFO("Field medic", "Diagnose complete");
  }

  __attribute__ ((constructor))
  void register_medic() {
    os::register_plugin(medic::init, "Field medic");
  }


  namespace diag
  {
    extern thread_local medic::diag::Tl_bss_arr __tl_bss;
    extern thread_local medic::diag::Tl_data_arr __tl_data;

    /** Verify TLS data from a different translation unit */
    bool tls() {
      for (auto& c : __tl_bss) {
        if(c != '!')
          throw medic::diag::Error("unexpected .tbss value");
      }
      for (auto& i : __tl_data) {
        if (i != 42)
          return false;
      }
      return true;
    }
  }
}

extern "C" char get_single_tbss(){
  return medic::diag::__tl_bss[0];
}

extern "C" int get_single_tdata(){
  return medic::diag::__tl_data[0];
}
