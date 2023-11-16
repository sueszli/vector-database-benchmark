#include <xtd/diagnostics/debug>
#include <xtd/console>

using namespace xtd;
using namespace xtd::diagnostics;

auto main()->int {
  // Uncomment following line to remove assert dialog
  //diagnostics::debug::show_assert_dialog(false);
  
  auto index = 0;
  console::write_line("Start application");
  debug::cassert_(index > 0);
  console::write_line("End application");
}

// This code produces the following output :
//
// > If user clicks assert dialog 'Abort' button:
// Start application
//
// > If user clicks assert dialog 'Retry' button:
// Start application
// > Break the debugger on file debug_cassert.cpp line 12.
// > If user choose continue running after break.
// End application
//
// > If user clicks assert dialog 'Ignore' button:
// Start application
// End application
