/*!A cross-platform build utility based on Lua
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Copyright (C) 2015-present, TBOOX Open Source Group.
 *
 * @author      ruki
 * @file        decompress_stream_write.c
 *
 */

/* //////////////////////////////////////////////////////////////////////////////////////
 * trace
 */
#define TB_TRACE_MODULE_NAME    "decompress_stream_write"
#define TB_TRACE_MODULE_DEBUG   (0)

/* //////////////////////////////////////////////////////////////////////////////////////
 * includes
 */
#include "prefix.h"

/* //////////////////////////////////////////////////////////////////////////////////////
 * implementation
 */

tb_int_t xm_lz4_decompress_stream_write(lua_State* lua)
{
    // check
    tb_assert_and_check_return_val(lua, 0);

    // check handle
    if (!xm_lua_ispointer(lua, 1))
    {
        lua_pushinteger(lua, -1);
        lua_pushliteral(lua, "invalid handle!");
        return 2;
    }

    // get stream
    xm_lz4_dstream_t* stream = (xm_lz4_dstream_t*)xm_lua_topointer(lua, 1);
    tb_check_return_val(stream, 0);

    // get data and size
    tb_size_t        size = 0;
    tb_byte_t const* data = tb_null;
    if (lua_isnumber(lua, 2)) data = (tb_byte_t const*)(tb_size_t)(tb_long_t)lua_tonumber(lua, 2);
    if (lua_isnumber(lua, 3)) size = (tb_size_t)lua_tonumber(lua, 3);
    if (!data || !size)
    {
        lua_pushinteger(lua, -1);
        lua_pushfstring(lua, "invalid data(%p) and size(%d)!", data, (tb_int_t)size);
        return 2;
    }

    // write data
    tb_long_t real = xm_lz4_dstream_write(stream, data, size, tb_false);
    lua_pushinteger(lua, (tb_int_t)real);
    return 1;
}
