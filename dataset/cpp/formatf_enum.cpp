#include <xtd/xtd>

using namespace std;
using namespace xtd;

enum week_day {
  monday,
  tuesday,
  wednesday,
  thursday,
  friday,
  saturday,
  sunday
};

// Only this operator is needed for week_day enum to be recognized by format()
inline ostream& operator<<(ostream& os, week_day value) {
  return os << to_string(value, {{week_day::monday, "monday"}, {week_day::tuesday, "tuesday"}, {week_day::wednesday, "wednesday"}, {week_day::thursday, "thursday"}, {week_day::friday, "friday"}, {week_day::saturday, "saturday"}, {week_day::sunday, "sunday"}});
}

int main() {
  cout << format("{}", week_day::saturday) << endl;
  cout << format("0b{:b}", week_day::saturday) << endl;
  cout << format("0b{:B}", week_day::saturday) << endl;
  cout << format("{:d}", week_day::saturday) << endl;
  cout << format("{:D}", week_day::saturday) << endl;
  cout << format("{:g}", week_day::saturday) << endl;
  cout << format("{:G}", week_day::saturday) << endl;
  cout << format("0{:o}", week_day::saturday) << endl;
  cout << format("0{:O}", week_day::saturday) << endl;
  cout << format("0x{:x}", week_day::saturday) << endl;
  cout << format("0x{:X}", week_day::saturday) << endl;
}

// This code produces the following output :
//
// saturday
// 0b101
// 0b101
// 5
// 5
// saturday
// saturday
// 05
// 05
// 0x5
// 0x5
