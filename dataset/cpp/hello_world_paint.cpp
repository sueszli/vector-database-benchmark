#include <xtd/drawing/brushes>
#include <xtd/forms/application>
#include <xtd/forms/form>

using namespace xtd;
using namespace xtd::drawing;
using namespace xtd::forms;

namespace hello_world_paint_example {
  class main_form : public form {
  public:
    main_form() {
      text("Hello world (paint)");
      client_size({300, 300});
      
      paint += [&](object& sender, paint_event_args& e) {
        e.graphics().clear(color::from_argb(0x0, 0x20, 0x10));
        e.graphics().draw_string("Hello, World!", {system_fonts::default_font(), 32, font_style::bold | font_style::italic}, solid_brush {color::dark(color::spring_green, 2.0 / 3)}, rectangle::offset(e.clip_rectangle(), {2, 2}), string_format().alignment(string_alignment::center).line_alignment(string_alignment::center));
        e.graphics().draw_string("Hello, World!", {system_fonts::default_font(), 32, font_style::bold | font_style::italic}, brushes::spring_green(), rectangle::inflate(e.clip_rectangle(), {-2, -2}), string_format().alignment(string_alignment::center).line_alignment(string_alignment::center));
      };
    }
  };
}

auto main()->int {
  application::run(hello_world_paint_example::main_form());
}
