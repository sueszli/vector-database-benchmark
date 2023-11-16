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
 * @file        openv.c
 *
 */

/* //////////////////////////////////////////////////////////////////////////////////////
 * trace
 */
#define TB_TRACE_MODULE_NAME                "process.openv"
#define TB_TRACE_MODULE_DEBUG               (0)

/* //////////////////////////////////////////////////////////////////////////////////////
 * includes
 */
#include "prefix.h"
#include "../io/prefix.h"
#if defined(TB_CONFIG_OS_MACOSX) || defined(TB_CONFIG_OS_LINUX) || defined(TB_CONFIG_OS_BSD) || defined(TB_CONFIG_OS_HAIKU)
#   include <signal.h>
#endif

/* //////////////////////////////////////////////////////////////////////////////////////
 * implementation
 */

/* p = process.openv(shellname, argv,
 * {outpath = "", errpath = "", outfile = "",
 *  errfile = "", outpipe = "", errpipe = "",
 *  infile = "", inpipe = "", inpipe = "",
 *  envs = {"PATH=xxx", "XXX=yyy"}})
 */
tb_int_t xm_process_openv(lua_State* lua)
{
    // check
    tb_assert_and_check_return_val(lua, 0);

    // check argv
    if (!lua_istable(lua, 2))
    {
        // error
        lua_pushfstring(lua, "invalid argv type(%s) for process.openv", luaL_typename(lua, 2));
        lua_error(lua);
        return 0;
    }

    // get shellname
    tb_char_t const* shellname  = lua_tostring(lua, 1);
    tb_check_return_val(shellname, 0);

    // get the arguments count
    tb_long_t argn = (tb_long_t)lua_objlen(lua, 2);
    tb_check_return_val(argn >= 0, 0);

    // get arguments
    tb_size_t           argi = 0;
    tb_char_t const**   argv = tb_nalloc0_type(1 + argn + 1, tb_char_t const*);
    tb_check_return_val(argv, 0);

    // fill arguments
    argv[0] = shellname;
    for (argi = 0; argi < argn; argi++)
    {
        // get argv[i]
        lua_pushinteger(lua, argi + 1);
        lua_gettable(lua, 2);

        // is string?
        if (lua_isstring(lua, -1))
        {
            // pass this argument
            argv[1 + argi] = lua_tostring(lua, -1);
        }
        // is path instance?
        else if (lua_istable(lua, -1))
        {
            lua_pushstring(lua, "_STR");
            lua_gettable(lua, -2);
            argv[1 + argi] = lua_tostring(lua, -1);
            lua_pop(lua, 1);
        }
        else
        {
            // error
            lua_pushfstring(lua, "invalid argv[%d] type(%s) for process.openv", (tb_int_t)argi, luaL_typename(lua, -1));
            lua_error(lua);
        }

        // pop it
        lua_pop(lua, 1);
    }

    // init attributes
    tb_process_attr_t attr = {0};

    // get option arguments
    tb_bool_t          exclusive = tb_false;
    tb_size_t          envn = 0;
    tb_char_t const*   envs[1024] = {0};
    tb_char_t const*   inpath  = tb_null;
    tb_char_t const*   outpath = tb_null;
    tb_char_t const*   errpath = tb_null;
    xm_io_file_t*      infile  = tb_null;
    xm_io_file_t*      outfile = tb_null;
    xm_io_file_t*      errfile = tb_null;
    tb_pipe_file_ref_t inpipe  = tb_null;
    tb_pipe_file_ref_t outpipe = tb_null;
    tb_pipe_file_ref_t errpipe = tb_null;
    if (lua_istable(lua, 3))
    {
        // is detached?
        lua_pushstring(lua, "detach");
        lua_gettable(lua, 3);
        if (lua_toboolean(lua, -1))
            attr.flags |= TB_PROCESS_FLAG_DETACH;
        lua_pop(lua, 1);

        // is exclusive?
        lua_pushstring(lua, "exclusive");
        lua_gettable(lua, 3);
        if (lua_toboolean(lua, -1))
            exclusive = tb_true;
        lua_pop(lua, 1);

        // get curdir
        lua_pushstring(lua, "curdir");
        lua_gettable(lua, 3);
        attr.curdir = lua_tostring(lua, -1);
        lua_pop(lua, 1);

        // get inpath
        lua_pushstring(lua, "inpath");
        lua_gettable(lua, 3);
        inpath = lua_tostring(lua, -1);
        lua_pop(lua, 1);

        // get outpath
        lua_pushstring(lua, "outpath");
        lua_gettable(lua, 3);
        outpath = lua_tostring(lua, -1);
        lua_pop(lua, 1);

        // get errpath
        lua_pushstring(lua, "errpath");
        lua_gettable(lua, 3);
        errpath = lua_tostring(lua, -1);
        lua_pop(lua, 1);

        // get infile
        if (!inpath)
        {
            lua_pushstring(lua, "infile");
            lua_gettable(lua, 3);
            infile = (xm_io_file_t*)lua_touserdata(lua, -1);
            lua_pop(lua, 1);
        }

        // get outfile
        if (!outpath)
        {
            lua_pushstring(lua, "outfile");
            lua_gettable(lua, 3);
            outfile = (xm_io_file_t*)lua_touserdata(lua, -1);
            lua_pop(lua, 1);
        }

        // get errfile
        if (!errpath)
        {
            lua_pushstring(lua, "errfile");
            lua_gettable(lua, 3);
            errfile = (xm_io_file_t*)lua_touserdata(lua, -1);
            lua_pop(lua, 1);
        }

        // get inpipe
        if (!inpath && !infile)
        {
            lua_pushstring(lua, "inpipe");
            lua_gettable(lua, 3);
            inpipe = (tb_pipe_file_ref_t)lua_touserdata(lua, -1);
            lua_pop(lua, 1);
        }

        // get outpipe
        if (!outpath && !outfile)
        {
            lua_pushstring(lua, "outpipe");
            lua_gettable(lua, 3);
            outpipe = (tb_pipe_file_ref_t)lua_touserdata(lua, -1);
            lua_pop(lua, 1);
        }

        // get errpipe
        if (!errpath && !errfile)
        {
            lua_pushstring(lua, "errpipe");
            lua_gettable(lua, 3);
            errpipe = (tb_pipe_file_ref_t)lua_touserdata(lua, -1);
            lua_pop(lua, 1);
        }

        // get environments
        lua_pushstring(lua, "envs");
        lua_gettable(lua, 3);
        if (lua_istable(lua, -1))
        {
            // get environment variables count
            tb_size_t count = (tb_size_t)lua_objlen(lua, -1);

            // get all passed environment variables
            tb_size_t i;
            for (i = 0; i < count; i++)
            {
                // get envs[i]
                lua_pushinteger(lua, i + 1);
                lua_gettable(lua, -2);

                // is string?
                if (lua_isstring(lua, -1))
                {
                    // add this environment value
                    if (envn + 1 < tb_arrayn(envs))
                        envs[envn++] = lua_tostring(lua, -1);
                    else
                    {
                        // error
                        lua_pushfstring(lua, "envs is too large(%d > %d) for process.openv", (tb_int_t)envn, tb_arrayn(envs) - 1);
                        lua_error(lua);
                    }
                }
                else
                {
                    // error
                    lua_pushfstring(lua, "invalid envs[%d] type(%s) for process.openv", (tb_int_t)i, luaL_typename(lua, -1));
                    lua_error(lua);
                }

                // pop it
                lua_pop(lua, 1);
            }
        }
        lua_pop(lua, 1);
    }

    // redirect stdin?
    if (inpath)
    {
        // redirect stdin to file
        attr.in.path = inpath;
        attr.inmode  = TB_FILE_MODE_RO;
        attr.intype  = TB_PROCESS_REDIRECT_TYPE_FILEPATH;
    }
    else if (infile && xm_io_file_is_file(infile))
    {
        tb_file_ref_t rawfile = tb_null;
        if (tb_stream_ctrl(infile->stream, TB_STREAM_CTRL_FILE_GET_FILE, &rawfile) && rawfile)
        {
            attr.in.file = rawfile;
            attr.intype  = TB_PROCESS_REDIRECT_TYPE_FILE;
        }
    }
    else if (inpipe)
    {
        attr.in.pipe = inpipe;
        attr.intype  = TB_PROCESS_REDIRECT_TYPE_PIPE;
    }

    // redirect stdout?
    if (outpath)
    {
        // redirect stdout to file
        attr.out.path = outpath;
        attr.outmode  = TB_FILE_MODE_RW | TB_FILE_MODE_TRUNC | TB_FILE_MODE_CREAT;
        attr.outtype  = TB_PROCESS_REDIRECT_TYPE_FILEPATH;
    }
    else if (outfile && xm_io_file_is_file(outfile))
    {
        tb_file_ref_t rawfile = tb_null;
        if (tb_stream_ctrl(outfile->stream, TB_STREAM_CTRL_FILE_GET_FILE, &rawfile) && rawfile)
        {
            attr.out.file = rawfile;
            attr.outtype  = TB_PROCESS_REDIRECT_TYPE_FILE;
        }
    }
    else if (outpipe)
    {
        attr.out.pipe = outpipe;
        attr.outtype  = TB_PROCESS_REDIRECT_TYPE_PIPE;
    }

    // redirect stderr?
    if (errpath)
    {
        // redirect stderr to file
        attr.err.path = errpath;
        attr.errmode  = TB_FILE_MODE_RW | TB_FILE_MODE_TRUNC | TB_FILE_MODE_CREAT;
        attr.errtype  = TB_PROCESS_REDIRECT_TYPE_FILEPATH;
    }
    else if (errfile && xm_io_file_is_file(errfile))
    {
        tb_file_ref_t rawfile = tb_null;
        if (tb_stream_ctrl(errfile->stream, TB_STREAM_CTRL_FILE_GET_FILE, &rawfile) && rawfile)
        {
            attr.err.file = rawfile;
            attr.errtype  = TB_PROCESS_REDIRECT_TYPE_FILE;
        }
    }
    else if (errpipe)
    {
        attr.err.pipe = errpipe;
        attr.errtype  = TB_PROCESS_REDIRECT_TYPE_PIPE;
    }

    // set the new environments
    if (envn > 0) attr.envp = envs;

    /* we need to ignore SIGINT and SIGQUIT if we enter exclusive mode
     * @see https://github.com/xmake-io/xmake/discussions/2893
     */
#if defined(SIGINT)
    if (exclusive) signal(SIGINT, SIG_IGN);
#endif
#if defined(SIGQUIT)
    if (exclusive) signal(SIGQUIT, SIG_IGN);
#endif

    // init process
    tb_process_ref_t process = (tb_process_ref_t)tb_process_init(shellname, argv, &attr);
    if (process) xm_lua_pushpointer(lua, (tb_pointer_t)process);
    else lua_pushnil(lua);

    // exit argv
    if (argv) tb_free(argv);
    argv = tb_null;

    // ok
    return 1;
}
