#include <xtd/double_object>
#include <xtd/ustring>

using namespace std;
using namespace xtd;

auto main()->int {
  cout << ustring::sprintf("%f", 12.345) << endl;
  cout << ustring::sprintf("%F", 12.345) << endl;
  cout << ustring::sprintf("%e", 12.345) << endl;
  cout << ustring::sprintf("%E", 12.345) << endl;
  cout << ustring::sprintf("%g", 12.345) << endl;
  cout << ustring::sprintf("%G", 12.345) << endl;
  cout << ustring::sprintf("0x%a", 12.345) << endl;
  cout << ustring::sprintf("0x%A", 12.345) << endl;
  cout << ustring::sprintf("%G", double_object::epsilon) << endl;
  cout << ustring::sprintf("%f", double_object::NaN) << endl;
  cout << ustring::sprintf("%f", double_object::positive_infinity) << endl;
  cout << ustring::sprintf("%f", double_object::negative_infinity) << endl;
}

// This code produces the following output :
//
// 12.345000
// 12.345000
// 1.234500e+01
// 1.234500E+01
// 12.345
// 12.345
// 0x0x1.8b0a3d70a3d71p+3
// 0x0X1.8B0A3D70A3D71P+3
// 4.94066E-324
// nan
// inf
// -inf
