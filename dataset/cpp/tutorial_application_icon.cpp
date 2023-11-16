#include <xtd/xtd>

namespace tutorial {
  class form_icon : public xtd::forms::form {
  public:
    form_icon() {
      text("Icon");
      start_position(xtd::forms::form_start_position::center_screen);
      icon(xtd::drawing::system_icons::gammasoft());
    }
    
    static auto main() {
      xtd::forms::application::run(form_icon());
    }
  };
}

startup_(tutorial::form_icon::main);
