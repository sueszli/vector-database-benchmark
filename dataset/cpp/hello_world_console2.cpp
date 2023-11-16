#include <xtd/background_color>
#include <xtd/console>
#include <xtd/foreground_color>
#include <xtd/reset_color>

using namespace std;
using namespace xtd;

auto main()->int {
  console::out << background_color(console_color::blue) << foreground_color(console_color::white) << "Hello, World!" << reset_color() << endl;
}

// This code produces the following output with colors :
//
// Hello, World!
