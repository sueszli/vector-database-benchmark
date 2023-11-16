#include <xtd/forms/application>
#include <xtd/forms/button>
#include <xtd/forms/form>
#include <xtd/forms/message_box>

using namespace xtd::forms;

class main_form : public form {
public:
  main_form() {
    text("Hello world (message_box)");
    
    button1.location({10, 10});
    button1.parent(*this);
    button1.text("&Click me");
    button1.click += [] {
      message_box::show("Hello, World!");
    };
  }
  
private:
  button button1;
};

auto main()->int {
  application::run(main_form());
}
