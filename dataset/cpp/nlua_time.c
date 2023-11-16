/*
 * See Licensing and Copyright notice in naev.h
 */
/**
 * @file nlua_time.c
 *
 * @brief Time manipulation Lua bindings.
 */
/** @cond */
#include <lauxlib.h>
#include <stdlib.h>

#include "naev.h"
/** @endcond */

#include "nlua_time.h"

#include "log.h"
#include "nluadef.h"
#include "ntime.h"

/* Time methods. */
static int timeL_new( lua_State *L );
static int timeL_add( lua_State *L );
static int timeL_add__( lua_State *L );
static int timeL_sub( lua_State *L );
static int timeL_sub__( lua_State *L );
static int timeL_eq( lua_State *L );
static int timeL_lt( lua_State *L );
static int timeL_le( lua_State *L );
static int timeL_get( lua_State *L );
static int timeL_str( lua_State *L );
static int timeL_inc( lua_State *L );
static int timeL_tonumber( lua_State *L );
static int timeL_fromnumber( lua_State *L );
static const luaL_Reg time_methods[] = {
   { "new", timeL_new },
   { "add", timeL_add__ },
   { "__add", timeL_add },
   { "sub", timeL_sub__ },
   { "__sub", timeL_sub },
   { "__eq", timeL_eq },
   { "__lt", timeL_lt },
   { "__le", timeL_le },
   { "get", timeL_get },
   { "str", timeL_str },
   { "__tostring", timeL_str },
   { "inc", timeL_inc },
   { "tonumber", timeL_tonumber },
   { "fromnumber", timeL_fromnumber },
   {0,0}
}; /**< Time Lua methods. */

/**
 * @brief Loads the Time Lua library.
 *
 *    @param env Lua environment.
 *    @return 0 on success.
 */
int nlua_loadTime( nlua_env env )
{
   nlua_register(env, TIME_METATABLE, time_methods, 1);
   return 0; /* No error */
}

/**
 * @brief Bindings for interacting with the time.
 *
 * Usage is generally something as follows:
 * @code
 * time_limit = time.get() + time.new( 0, 5, 0 )
 * player.msg( string.format("You only have %s left!", time.str(time.get() - time_limit)) )
 *
 * -- Do stuff here
 *
 * if time.get() > time_limit then
 *    -- Limit is up
 * end
 * @endcode
 *
 * @luamod time
 */
/**
 * @brief Gets time at index.
 *
 *    @param L Lua state to get time from.
 *    @param ind Index position to find the time.
 *    @return Time found at the index in the state.
 */
ntime_t* lua_totime( lua_State *L, int ind )
{
   return (ntime_t*) lua_touserdata(L,ind);
}
/**
 * @brief Gets time at index raising an error if isn't a time.
 *
 *    @param L Lua state to get time from.
 *    @param ind Index position to find the time.
 *    @return Time found at the index in the state.
 */
ntime_t* luaL_checktime( lua_State *L, int ind )
{
   if (lua_istime(L,ind))
      return lua_totime(L,ind);
   luaL_typerror(L, ind, TIME_METATABLE);
   return NULL;
}
/**
 * @brief Gets a time directly.
 *
 *    @param L Lua state to get time from.
 *    @param ind Index position to find the time.
 *    @return Time found at the index in the state.
 */
ntime_t luaL_validtime( lua_State *L, int ind )
{
   return *luaL_checktime( L, ind );
}
/**
 * @brief Pushes a time on the stack.
 *
 *    @param L Lua state to push time into.
 *    @param time Time to push.
 *    @return Newly pushed time.
 */
ntime_t* lua_pushtime( lua_State *L, ntime_t time )
{
   ntime_t *p = (ntime_t*) lua_newuserdata(L, sizeof(ntime_t));
   *p = time;
   luaL_getmetatable(L, TIME_METATABLE);
   lua_setmetatable(L, -2);
   return p;
}
/**
 * @brief Checks to see if ind is a time.
 *
 *    @param L Lua state to check.
 *    @param ind Index position to check.
 *    @return 1 if ind is a time.
 */
int lua_istime( lua_State *L, int ind )
{
   int ret;

   if (lua_getmetatable(L,ind)==0)
      return 0;
   lua_getfield(L, LUA_REGISTRYINDEX, TIME_METATABLE);

   ret = 0;
   if (lua_rawequal(L, -1, -2))  /* does it have the correct mt? */
      ret = 1;

   lua_pop(L, 2);  /* remove both metatables */
   return ret;
}

/**
 * @brief Creates a time. This can be absolute or relative.
 *
 * @usage t = time.new( 591, 3271, 12801 ) -- Gets a time near when the incident happened.
 *
 *    @luatparam number cycles Cycles for the new time.
 *    @luatparam number periods Periods for the new time.
 *    @luatparam number seconds Seconds for the new time.
 *    @luatreturn Time A newly created time metatable.
 * @luafunc new
 */
static int timeL_new( lua_State *L )
{
   int cycles, periods, seconds;

   /* Parameters. */
   cycles = luaL_checkint(L,1);
   periods = luaL_checkint(L,2);
   seconds = luaL_checkint(L,3);

   /* Create the time. */
   lua_pushtime( L, ntime_create( cycles, periods, seconds ) );
   return 1;
}
/**
 * @brief Adds two time metatables.
 *
 * Overrides the addition operator.
 *
 * @usage new_time = time.get() + time.new( 0, 5, 0 ) -- Adds 5 periods to the current date
 *
 *    @luatparam Time t1 Time metatable to add to.
 *    @luatparam Time t2 Time metatable added.
 * @luafunc add
 */
static int timeL_add( lua_State *L )
{
   ntime_t t1, t2;

   /* Parameters. */
   t1 = luaL_validtime( L, 1 );
   t2 = luaL_validtime( L, 2 );

   /* Add them. */
   lua_pushtime( L, t1 + t2 );
   return 1;
}

/*
 * Method version of time_add that modifies the first time.
 */
static int timeL_add__( lua_State *L )
{
   ntime_t *t1, t2;

   /* Parameters. */
   t1 = luaL_checktime( L, 1 );
   t2 = luaL_validtime( L, 2 );

   /* Add them. */
   *t1 += t2;
   lua_pushtime( L, *t1 );
   return 1;
}

/**
 * @brief Subtracts two time metatables.
 *
 * Overrides the subtraction operator.
 *
 * @usage new_time = time.get() - time.new( 0, 3, 0 ) -- Subtracts 3 periods from the current date
 *
 *    @luatparam Time t1 Time metatable to subtract from.
 *    @luatparam Time t2 Time metatable subtracted.
 * @luafunc sub
 */
static int timeL_sub( lua_State *L )
{
   ntime_t t1, t2;

   /* Parameters. */
   t1 = luaL_validtime( L, 1 );
   t2 = luaL_validtime( L, 2 );

   /* Sub them. */
   lua_pushtime( L, t1 - t2 );
   return 1;
}

/*
 * Method version of time_sub that modifies the first time.
 */
static int timeL_sub__( lua_State *L )
{
   ntime_t *t1, t2;

   /* Parameters. */
   t1 = luaL_checktime( L, 1 );
   t2 = luaL_validtime( L, 2 );

   /* Sub them. */
   *t1 -= t2;
   lua_pushtime( L, *t1 );
   return 1;
}

/**
 * @brief Checks to see if two time are equal.
 *
 * It is recommended to check with < and <= instead of ==.
 *
 * @usage if time.new( 630, 5, 78) == time.get() then -- do something if they match
 *
 *    @luatparam Time t1 Time to compare for equality.
 *    @luatparam Time t2 Time to compare for equality.
 *    @luatreturn boolean true if they're equal.
 * @luafunc __eq
 */
static int timeL_eq( lua_State *L )
{
   ntime_t t1, t2;
   t1 = luaL_validtime( L, 1 );
   t2 = luaL_validtime( L, 2 );
   lua_pushboolean( L, t1==t2 );
   return 1;
}
/**
 * @brief Checks to see if a time is strictly larger than another.
 *
 * @usage if time.new( 630, 5, 78) < time.get() then -- do something if time is past UST 630:0005.78
 *
 *    @luatparam Time t1 Time to see if is is smaller than t2.
 *    @luatparam Time t2 Time see if is larger than t1.
 *    @luatreturn boolean true if t1 < t2
 * @luafunc __lt
 */
static int timeL_lt( lua_State *L )
{
   ntime_t t1, t2;
   t1 = luaL_validtime( L, 1 );
   t2 = luaL_validtime( L, 2 );
   lua_pushboolean( L, t1<t2 );
   return 1;
}
/**
 * @brief Checks to see if a time is larger or equal to another.
 *
 * @usage if time.new( 630, 5, 78) <= time.get() then -- do something if time is past UST 630:0005.78
 *
 *    @luatparam Time t1 Time to see if is is smaller or equal to than t2.
 *    @luatparam Time t2 Time see if is larger or equal to than t1.
 *    @luatreturn boolean true if t1 <= t2
 * @luafunc __le
 */
static int timeL_le( lua_State *L )
{
   ntime_t t1, t2;
   t1 = luaL_validtime( L, 1 );
   t2 = luaL_validtime( L, 2 );
   lua_pushboolean( L, t1<=t2 );
   return 1;
}
/**
 * @brief Gets the current time in internal representation time.
 *
 * @usage t = time.get()
 *
 *    @luatreturn Time Time in internal representation time.
 * @luafunc get
 */
static int timeL_get( lua_State *L )
{
   lua_pushtime( L, ntime_get() );
   return 1;
}
/**
 * @brief Converts the time to a pretty human readable format.
 *
 * @usage strt = time.str() -- Gets current time
 * @usage strt = time.str( nil, 5 ) -- Gets current time with full decimals
 * @usage strt = time.str( time.get() + time.new(0,5,0) ) -- Gets time in 5 periods
 * @usage strt = t:str() -- Gets the string of t
 *
 *    @luatparam Time t Time to convert to pretty format.  If omitted, current time is used.
 *    @luatparam[opt=2] number d Decimals to use for displaying seconds (should be between 0 and 5).
 *    @luatreturn string The time in human readable format.
 * @luafunc str
 */
static int timeL_str( lua_State *L )
{
   ntime_t t;
   char nt[64];
   int d;

   /* Parse parameters. */
   if (!lua_isnoneornil(L,1))
      t = luaL_validtime(L,1);
   else
      t = ntime_get();
   d = luaL_optinteger(L,2,2); /* Defaults to 2 decimals. */

   /* Push string. */
   ntime_prettyBuf( nt, sizeof(nt), t, d );
   lua_pushstring(L, nt);
   return 1;
}

/**
 * @brief Increases or decreases the in-game time.
 *
 * Note that this can trigger hooks and fail missions and the likes.
 *
 * @usage time.inc( time.new(0,0,100) ) -- Increments the time by 100 seconds.
 *
 *    @luatparam Time t Amount to increment or decrement the time by.
 * @luafunc inc
 */
static int timeL_inc( lua_State *L )
{
   ntime_inc( luaL_validtime(L,1) );
   return 0;
}

/**
 * @brief Gets a number representing this time.
 *
 * The best usage for this currently is mission variables.
 *
 * @usage num = t:tonumber() -- Getting the number from a time t
 *
 *    @luatparam Time t Time to get number of.
 *    @luatreturn number Number representing time.
 * @luafunc tonumber
 */
static int timeL_tonumber( lua_State *L )
{
   ntime_t t = luaL_validtime(L,1);
   lua_pushnumber( L, t );
   return 1;
}

/**
 * @brief Creates a time from a number representing it.
 *
 * The best usage for this currently is mission variables.
 *
 * @usage t = time.fromnumber( t:tonumber() ) -- Should get the time t again
 *
 *    @luatparam number num Number to get time from.
 *    @luatreturn Time Time representing number.
 * @luafunc fromnumber
 */
static int timeL_fromnumber( lua_State *L )
{
   ntime_t t = (ntime_t) luaL_checknumber(L,1);
   lua_pushtime( L, t );
   return 1;
}
