#include <xtd/diagnostics/assert>
#include <xtd/console>

using namespace xtd;

auto main()->int {
  // Uncomment following line to remove assert dialog
  //diagnostics::debug::show_assert_dialog(false);
  
  auto index = 0;
  console::write_line("Start application");
  assert_(index > 0, "index must be greater than 0");
  console::write_line("End application");
}

// This code produces the following output :
//
// > If user clicks assert dialog 'Abort' button:
// Start application
//
// > If user clicks assert dialog 'Retry' button:
// Start application
// > Break the debugger on file assert_with_message.cpp line 11.
// > If user choose continue running after break.
// End application
//
// > If user clicks assert dialog 'Ignore' button:
// Start application
// End application
