#include <xtd/diagnostics/stack_frame>
#include <xtd/console>

using namespace xtd;

void trace_message(const ustring& message, const xtd::diagnostics::stack_frame& stack_frame) {
  console::write_line("stack_frame: {}\n", stack_frame);
  console::write_line("Message: {}", message);
  console::write_line("Method: {}", stack_frame.get_method());
  console::write_line("File name: {}", stack_frame.get_file_name());
  console::write_line("File line number: {}", stack_frame.get_file_line_number());
}

auto main()->int {
  trace_message("Something has happened.", current_stack_frame_);
  // trace_message("Something has happened.", current_stack_frame_); is same as :
  //
  // trace_message("Something has happened.", {__FILE__, __LINE__, __func__});
}

// This code can produce the following output:
//
// stack_frame: /!---OMITTED---!/current_stack_frame.cpp:15:0
//
// Message: Something happened.
// Method name: main
// File name: /!---OMITTED---!/current_stack_frame.cpp
// File line number: 15
