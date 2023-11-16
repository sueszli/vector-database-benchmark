#include <xtd/web/css/css_writer>
#include <xtd/console>
#include <strstream>

using namespace std;
using namespace xtd;
using namespace xtd::web::css;

auto main()->int {
  auto stream = stringstream {};
  auto  writer = css_writer {stream};
  
  writer.selectors()[".user_box"].properties()["display"] = property::from("none");
  writer.selectors()[".user_box"].properties()["position"] = property::from("fixed");
  writer.selectors()[".user_box"].properties()["z-index"] = property::from(100);
  writer.selectors()[".user_box"].properties()["width"] = property::from("100%");
  writer.selectors()[".user_box"].properties()["height"] = property::from("100%");
  writer.selectors()[".user_box"].properties()["left"] = property::from(300);
  writer.selectors()[".user_box"].properties()["top"] = property::from(200);
  writer.selectors()[".user_box"].properties()["background"] = property::from("#4080FA");
  writer.selectors()[".user_box"].properties()["filter"] = property::from("alpha(opacity=40)");
  writer.selectors()[".user_box"].properties()["opacity"] = property::from(0.4);
  writer.write();
  
  console::write_line("String stream :");
  console::write_line("---------------");
  console::write_line(stream.str());
}

// This code can produces the following output :
//
// String stream :
// ---------------
// .user_box {
//   background: #4080FA;
//   display: none;
//   filter: alpha(opacity=40);
//   height: 100%;
//   left: 300;
//   opacity: 0.4;
//   position: fixed;
//   top: 200;
//   width: 100%;
//   z-index: 100;
// }
