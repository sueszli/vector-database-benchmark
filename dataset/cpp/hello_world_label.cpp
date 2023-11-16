#include <xtd/forms/application>
#include <xtd/forms/form>
#include <xtd/forms/label>

using namespace xtd;
using namespace xtd::drawing;
using namespace xtd::forms;

namespace hello_world_label_example {
  class main_form : public form {
  public:
    main_form() {
      text("Hello world (label)");
      controls().push_back(label);
      
      label.dock(dock_style::fill);
      label.font(drawing::font {label.font(), 32, font_style::bold | font_style::italic});
      label.fore_color(color::green);
      label.shadow(true);
      label.text("Hello, World!");
      label.text_align(xtd::forms::content_alignment::middle_center);
    }
    
  private:
    forms::label label;
  };
}

auto main()->int {
  application::run(hello_world_label_example::main_form {});
}
