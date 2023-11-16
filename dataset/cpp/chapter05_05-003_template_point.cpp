///////////////////////////////////////////////////////////////////////////////
//  Copyright Christopher Kormanyos 2017 - 2018.
//  Distributed under the Boost Software License,
//  Version 1.0. (See accompanying file LICENSE_1_0.txt
//  or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// chapter05_05-002_template_point.cpp

#include <cstdint>
#include <iostream>

// Only the second template parameter
// has a default types.
template<typename x_type,
         typename y_type = std::uint16_t> // OK
class point
{
public:
  x_type my_x;
  y_type my_y;

  point(const x_type& x = x_type(),
        const y_type& y = y_type()) : my_x(x),
                                      my_y(y) { }
};

// An (x8, y16) point.
point<std::uint8_t> pt08_16
{
  UINT8_C(34),
  UINT16_C(5678)
};

int main()
{
  std::cout << "pt08_16: "
            << "(" << unsigned(pt08_16.my_x) << "," << unsigned(pt08_16.my_y) << ")"
            << std::endl;
}
