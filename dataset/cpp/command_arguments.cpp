/*
  Copyright (C) 2022-2023  Sutou Kouhei <kou@clear-code.com>

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License as published by the Free Software Foundation; either
  version 2.1 of the License, or (at your option) any later version.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

#include "grn_proc.h"

#include <groonga.hpp>

namespace grn {
  bool
  CommandArguments::arg_to_bool(grn_obj *arg, bool default_value)
  {
    return grn_proc_option_value_bool(ctx_, arg, default_value);
  }

  int32_t
  CommandArguments::arg_to_int32(grn_obj *arg, int32_t default_value)
  {
    return grn_proc_option_value_int32(ctx_, arg, default_value);
  }

  uint32_t
  CommandArguments::arg_to_uint32(grn_obj *arg, uint32_t default_value)
  {
    return grn_proc_option_value_uint32(ctx_, arg, default_value);
  }

  float
  CommandArguments::arg_to_float(grn_obj *arg, float default_value)
  {
    return grn_proc_option_value_float(ctx_, arg, default_value);
  }
} // namespace grn
