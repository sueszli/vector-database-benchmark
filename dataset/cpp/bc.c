/* A Bison parser, made by GNU Bison 3.0.4.  */

/* Bison implementation for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015 Free Software Foundation, Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "3.0.4"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1




/* Copy the first part of user declarations.  */
#line 1 "bc.y" /* yacc.c:339  */

/*	from 4.4BSD /usr/src/usr.bin/bc/bc.y	*/
/*-
 * Copyright (c) 1991, 1993
 *	The Regents of the University of California.  All rights reserved.
 *
 * This module is believed to contain source code proprietary to AT&T.
 * Use and redistribution is subject to the Berkeley Software License
 * Agreement and your Software Agreement with AT&T (Western Electric).
 *
 *	from bc.y	8.1 (Berkeley) 6/6/93
 */
/*
 * Copyright(C) Caldera International Inc. 2001-2002. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *   Redistributions of source code and documentation must retain the
 *    above copyright notice, this list of conditions and the following
 *    disclaimer.
 *   Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *   All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *      This product includes software developed or owned by Caldera
 *      International, Inc.
 *   Neither the name of Caldera International, Inc. nor the names of
 *    other contributors may be used to endorse or promote products
 *    derived from this software without specific prior written permission.
 *
 * USE OF THE SOFTWARE PROVIDED FOR UNDER THIS LICENSE BY CALDERA
 * INTERNATIONAL, INC. AND CONTRIBUTORS ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL CALDERA INTERNATIONAL, INC. BE
 * LIABLE FOR ANY DIRECT, INDIRECT INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#if __GNUC__ >= 3 && __GNUC_MINOR__ >= 4 || __GNUC__ >= 4
#define	USED	__attribute__ ((used))
#elif defined __GNUC__
#define	USED	__attribute__ ((unused))
#else
#define	USED
#endif
static const char sccsid[] USED = "@(#)bc.sl	1.24 (gritter) 7/3/05";
#include <unistd.h>
#include <signal.h>
#include <limits.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
typedef	intptr_t	YYSTYPE;
			YYSTYPE yyval;
#define	YYSTYPE	YYSTYPE
	static int cpeek(int c, int yes, int no);
	static int getch(void);
	static intptr_t bundle(int a, ...);
	static void routput(intptr_t *p);
	static void output(intptr_t *p);
	static void conout(intptr_t p, intptr_t s);
	static void pp(intptr_t);
	static void tp(intptr_t);
	static void yyinit(int argc, char *argv[]);
	static intptr_t *getout(void);
	static intptr_t *getf(intptr_t);
	static intptr_t *geta(intptr_t);
	static void yyerror(const char *);
	static void cantopen(const char *);
	extern int yylex(void);

#if defined (__GLIBC__) && defined (_IO_getc_unlocked)
#undef	getc
#define	getc(f)		_IO_getc_unlocked(f)
#endif
#line 97 "bc.y" /* yacc.c:339  */

#define	THIS_BC_STRING_MAX	1000
static FILE *in;
static char cary[LINE_MAX + 1], *cp = { cary };
static char string[THIS_BC_STRING_MAX + 3], *str = {string};
static int crs = '0';
static int rcrs = '0';  /* reset crs */
static int bindx = 0;
static int lev = 0;
static int ln;
static char *ss;
static int bstack[10] = { 0 };
static char *numb[15] = {
  " 0", " 1", " 2", " 3", " 4", " 5",
  " 6", " 7", " 8", " 9", " 10", " 11",
  " 12", " 13", " 14" };
static intptr_t *pre, *post;

#line 168 "y.tab.c" /* yacc.c:339  */

# ifndef YY_NULLPTR
#  if defined __cplusplus && 201103L <= __cplusplus
#   define YY_NULLPTR nullptr
#  else
#   define YY_NULLPTR 0
#  endif
# endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 0
#endif


/* Debug traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif
#if YYDEBUG
extern int yydebug;
#endif

/* Token type.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
  enum yytokentype
  {
    UMINUS = 258,
    LETTER = 259,
    DIGIT = 260,
    SQRT = 261,
    LENGTH = 262,
    _IF = 263,
    FFF = 264,
    EQ = 265,
    _WHILE = 266,
    _FOR = 267,
    NE = 268,
    LE = 269,
    GE = 270,
    INCR = 271,
    DECR = 272,
    _RETURN = 273,
    _BREAK = 274,
    _DEFINE = 275,
    BASE = 276,
    OBASE = 277,
    SCALE = 278,
    EQPL = 279,
    EQMI = 280,
    EQMUL = 281,
    EQDIV = 282,
    EQREM = 283,
    EQEXP = 284,
    _AUTO = 285,
    DOT = 286,
    QSTR = 287
  };
#endif
/* Tokens.  */
#define UMINUS 258
#define LETTER 259
#define DIGIT 260
#define SQRT 261
#define LENGTH 262
#define _IF 263
#define FFF 264
#define EQ 265
#define _WHILE 266
#define _FOR 267
#define NE 268
#define LE 269
#define GE 270
#define INCR 271
#define DECR 272
#define _RETURN 273
#define _BREAK 274
#define _DEFINE 275
#define BASE 276
#define OBASE 277
#define SCALE 278
#define EQPL 279
#define EQMI 280
#define EQMUL 281
#define EQDIV 282
#define EQREM 283
#define EQEXP 284
#define _AUTO 285
#define DOT 286
#define QSTR 287

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef int YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif


extern YYSTYPE yylval;

int yyparse (void);



/* Copy the second part of user declarations.  */

#line 280 "y.tab.c" /* yacc.c:358  */

#ifdef short
# undef short
#endif

#ifdef YYTYPE_UINT8
typedef YYTYPE_UINT8 yytype_uint8;
#else
typedef unsigned char yytype_uint8;
#endif

#ifdef YYTYPE_INT8
typedef YYTYPE_INT8 yytype_int8;
#else
typedef signed char yytype_int8;
#endif

#ifdef YYTYPE_UINT16
typedef YYTYPE_UINT16 yytype_uint16;
#else
typedef unsigned short int yytype_uint16;
#endif

#ifdef YYTYPE_INT16
typedef YYTYPE_INT16 yytype_int16;
#else
typedef short int yytype_int16;
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif ! defined YYSIZE_T
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned int
# endif
#endif

#define YYSIZE_MAXIMUM ((YYSIZE_T) -1)

#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(Msgid) dgettext ("bison-runtime", Msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(Msgid) Msgid
# endif
#endif

#ifndef YY_ATTRIBUTE
# if (defined __GNUC__                                               \
      && (2 < __GNUC__ || (__GNUC__ == 2 && 96 <= __GNUC_MINOR__)))  \
     || defined __SUNPRO_C && 0x5110 <= __SUNPRO_C
#  define YY_ATTRIBUTE(Spec) __attribute__(Spec)
# else
#  define YY_ATTRIBUTE(Spec) /* empty */
# endif
#endif

#ifndef YY_ATTRIBUTE_PURE
# define YY_ATTRIBUTE_PURE   YY_ATTRIBUTE ((__pure__))
#endif

#ifndef YY_ATTRIBUTE_UNUSED
# define YY_ATTRIBUTE_UNUSED YY_ATTRIBUTE ((__unused__))
#endif

#if !defined _Noreturn \
     && (!defined __STDC_VERSION__ || __STDC_VERSION__ < 201112)
# if defined _MSC_VER && 1200 <= _MSC_VER
#  define _Noreturn __declspec (noreturn)
# else
#  define _Noreturn YY_ATTRIBUTE ((__noreturn__))
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(E) ((void) (E))
#else
# define YYUSE(E) /* empty */
#endif

#if defined __GNUC__ && 407 <= __GNUC__ * 100 + __GNUC_MINOR__
/* Suppress an incorrect diagnostic about yylval being uninitialized.  */
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN \
    _Pragma ("GCC diagnostic push") \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")\
    _Pragma ("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")
# define YY_IGNORE_MAYBE_UNINITIALIZED_END \
    _Pragma ("GCC diagnostic pop")
#else
# define YY_INITIAL_VALUE(Value) Value
#endif
#ifndef YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_END
#endif
#ifndef YY_INITIAL_VALUE
# define YY_INITIAL_VALUE(Value) /* Nothing. */
#endif


#if ! defined yyoverflow || YYERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined EXIT_SUCCESS
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
      /* Use EXIT_SUCCESS as a witness for stdlib.h.  */
#     ifndef EXIT_SUCCESS
#      define EXIT_SUCCESS 0
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's 'empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (0)
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined EXIT_SUCCESS \
       && ! ((defined YYMALLOC || defined malloc) \
             && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef EXIT_SUCCESS
#    define EXIT_SUCCESS 0
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined EXIT_SUCCESS
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined EXIT_SUCCESS
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* ! defined yyoverflow || YYERROR_VERBOSE */


#if (! defined yyoverflow \
     && (! defined __cplusplus \
         || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yytype_int16 yyss_alloc;
  YYSTYPE yyvs_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (yytype_int16) + sizeof (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

# define YYCOPY_NEEDED 1

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack_alloc, Stack)                           \
    do                                                                  \
      {                                                                 \
        YYSIZE_T yynewbytes;                                            \
        YYCOPY (&yyptr->Stack_alloc, Stack, yysize);                    \
        Stack = &yyptr->Stack_alloc;                                    \
        yynewbytes = yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
        yyptr += yynewbytes / sizeof (*yyptr);                          \
      }                                                                 \
    while (0)

#endif

#if defined YYCOPY_NEEDED && YYCOPY_NEEDED
/* Copy COUNT objects from SRC to DST.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(Dst, Src, Count) \
      __builtin_memcpy (Dst, Src, (Count) * sizeof (*(Src)))
#  else
#   define YYCOPY(Dst, Src, Count)              \
      do                                        \
        {                                       \
          YYSIZE_T yyi;                         \
          for (yyi = 0; yyi < (Count); yyi++)   \
            (Dst)[yyi] = (Src)[yyi];            \
        }                                       \
      while (0)
#  endif
# endif
#endif /* !YYCOPY_NEEDED */

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  2
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   673

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  54
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  20
/* YYNRULES -- Number of rules.  */
#define YYNRULES  121
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  226

/* YYTRANSLATE[YYX] -- Symbol number corresponding to YYX as returned
   by yylex, with out-of-bounds checking.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   287

#define YYTRANSLATE(YYX)                                                \
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, without out-of-bounds checking.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
      48,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     8,     2,     2,
      45,    40,     6,     4,    52,     5,     2,     7,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,    46,
      49,     3,    50,    51,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    43,     2,    44,     9,    53,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    41,     2,    42,    47,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,    10,    11,
      12,    13,    14,    15,    16,    17,    18,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
      32,    33,    34,    35,    36,    37,    38,    39
};

#if YYDEBUG
  /* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   116,   116,   117,   119,   128,   129,   132,   135,   136,
     138,   140,   142,   144,   146,   148,   150,   152,   154,   156,
     158,   160,   162,   164,   166,   168,   170,   172,   176,   181,
     186,   190,   192,   194,   196,   198,   200,   204,   209,   212,
     213,   217,   219,   222,   224,   226,   228,   230,   232,   234,
     238,   240,   242,   244,   246,   248,   250,   252,   254,   256,
     258,   260,   262,   264,   266,   268,   270,   272,   274,   276,
     278,   280,   282,   284,   286,   288,   290,   292,   294,   296,
     298,   300,   302,   304,   306,   308,   310,   312,   314,   316,
     318,   320,   322,   324,   326,   328,   330,   332,   334,   336,
     338,   340,   342,   344,   346,   350,   351,   354,   355,   359,
     363,   365,   367,   372,   381,   390,   391,   393,   397,   399,
     402,   403
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || 0
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "'='", "'+'", "'-'", "'*'", "'/'", "'%'",
  "'^'", "UMINUS", "LETTER", "DIGIT", "SQRT", "LENGTH", "_IF", "FFF", "EQ",
  "_WHILE", "_FOR", "NE", "LE", "GE", "INCR", "DECR", "_RETURN", "_BREAK",
  "_DEFINE", "BASE", "OBASE", "SCALE", "EQPL", "EQMI", "EQMUL", "EQDIV",
  "EQREM", "EQEXP", "_AUTO", "DOT", "QSTR", "')'", "'{'", "'}'", "'['",
  "']'", "'('", "';'", "'~'", "'\\n'", "'<'", "'>'", "'?'", "','", "'_'",
  "$accept", "start", "dlist", "stat", "EQOP", "fprefix", "BLEV", "slist",
  "tail", "re", "e", "cargs", "eora", "cons", "constant", "CRS", "def",
  "dargs", "dlets", "lora", YY_NULLPTR
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[NUM] -- (External) token number corresponding to the
   (internal) symbol number NUM (which must be that of a token).  */
static const yytype_uint16 yytoknum[] =
{
       0,   256,   257,    61,    43,    45,    42,    47,    37,    94,
     258,   259,   260,   261,   262,   263,   264,   265,   266,   267,
     268,   269,   270,   271,   272,   273,   274,   275,   276,   277,
     278,   279,   280,   281,   282,   283,   284,   285,   286,   287,
      41,   123,   125,    91,    93,    40,    59,   126,    10,    60,
      62,    63,    44,    95
};
# endif

#define YYPACT_NINF -174

#define yypact_value_is_default(Yystate) \
  (!!((Yystate) == (-174)))

#define YYTABLE_NINF -9

#define yytable_value_is_error(Yytable_value) \
  0

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const yytype_int16 yypact[] =
{
    -174,   200,  -174,  -174,   405,   456,  -174,   -35,   -24,  -174,
    -174,  -174,   -22,     6,    82,     8,  -174,     9,   575,   581,
     497,    -7,  -174,   298,   405,    56,  -174,  -174,   -37,  -174,
     658,    26,    59,    63,   462,   597,   615,   503,    64,  -174,
     405,  -174,  -174,  -174,  -174,  -174,  -174,  -174,  -174,   405,
     102,   405,   405,   405,  -174,    34,   405,    37,  -174,  -174,
    -174,    39,  -174,  -174,  -174,   342,    38,   405,  -174,  -174,
     405,   405,  -174,  -174,   405,   405,  -174,  -174,   405,   405,
    -174,  -174,    81,   354,    92,  -174,  -174,  -174,   405,   405,
     405,   405,   405,   405,   405,    -7,  -174,    51,   -34,  -174,
     405,   405,   405,   405,   405,   405,   405,   405,   405,  -174,
     658,   162,   479,  -174,   658,   -33,  -174,   658,   417,   545,
      52,   405,    20,   405,   405,  -174,   551,  -174,   658,   658,
     658,   658,   658,   557,   658,  -174,   298,  -174,   405,    50,
      35,    53,    53,    99,    99,    99,    99,  -174,    65,    76,
      63,   658,   227,   658,   658,   658,   658,   658,   658,   658,
     621,   362,  -174,   425,  -174,  -174,   405,    88,  -174,   276,
     325,  -174,  -174,  -174,   658,   405,   405,   405,   405,   405,
     405,   405,  -174,   -37,  -174,   637,   405,  -174,  -174,   405,
    -174,  -174,    94,   298,  -174,  -174,   563,   658,   658,   658,
     658,   658,   658,   249,  -174,   405,   405,   658,   658,   298,
    -174,   298,    63,    91,   658,   658,  -174,  -174,  -174,    72,
    -174,  -174,  -174,    63,  -174,  -174
};

  /* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
     Performed when YYTABLE does not specify something else to do.  Zero
     means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
       2,     0,     1,    26,     0,    85,   111,     0,     0,   113,
      25,   113,     0,     0,     0,    17,    14,     0,   103,   104,
     102,    84,     9,     0,     0,     0,    93,   110,     0,   113,
       7,    80,   109,   115,    85,   103,   104,   102,     0,    52,
       0,    58,    61,    31,    32,    33,    34,    35,    36,     0,
       0,     0,     0,     0,    38,     0,     0,    59,    71,    75,
      67,    60,    73,    77,    69,     0,     0,     0,    70,    72,
       0,     0,    74,    76,     0,     0,    66,    68,     0,     0,
      81,    39,     0,     0,    95,    42,    41,     3,     0,     0,
       0,     0,     0,     0,     0,    83,   112,   120,     0,   116,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    95,
      10,     0,    85,    79,   107,     0,   105,    12,     0,     0,
       0,     0,     0,     0,     0,    16,     0,   114,    20,    21,
      22,    23,    18,     0,    19,    24,     0,    92,     0,     0,
      49,    50,    51,    53,    54,    55,    56,    82,     0,     0,
       0,    86,     0,    87,    98,    99,   100,   101,    96,    97,
      57,     0,    78,     0,    94,    90,     0,     0,    37,     0,
       0,    15,    91,    40,    30,     0,     0,     0,     0,     0,
       0,     0,   121,     0,   117,    57,     0,    62,    64,     0,
     108,   106,     0,     0,    63,    65,     0,    43,    46,    48,
      47,    44,    45,     0,     5,     0,     0,    11,    13,     0,
      38,     0,     0,     0,    88,    89,    27,    28,    38,     0,
     118,     4,    29,     0,     6,   119
};

  /* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -174,  -174,  -174,    -1,    -4,  -174,  -173,   -65,   -25,  -108,
      -2,  -174,   -20,   -17,  -174,     1,  -174,  -174,  -174,  -142
};

  /* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,   203,    81,   102,    29,   120,    82,   136,   139,
      30,   115,   116,    31,    32,    54,    33,    98,   219,    99
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_int16 yytable[] =
{
      28,    51,    39,    87,    80,     6,   149,   162,   184,    85,
      52,    86,    55,   167,    70,    74,    79,    57,   150,   163,
      66,    53,    83,    56,    89,    90,    91,    92,    93,    94,
      88,   104,   106,   108,    58,    59,    60,   217,   110,    89,
      90,    91,    92,    93,    94,   222,    27,   111,   114,   117,
     118,   119,   176,    65,   122,   177,   178,   179,   192,    91,
      92,    93,    94,   126,    95,   128,   168,    84,   129,   130,
     220,    96,   131,   132,    97,   109,   133,   134,   147,   121,
     123,   225,   124,   127,   180,   181,   140,   141,   142,   143,
     144,   145,   146,    61,   148,   138,   175,   166,   151,   152,
     153,   154,   155,   156,   157,   158,   159,     4,    94,   182,
      62,    63,    64,   112,     6,     7,     8,   183,    85,   140,
      86,   169,   170,   135,   223,    13,    14,    85,   193,    86,
      35,    36,    37,   221,   209,   173,   174,    85,   213,    86,
      21,     0,   113,   191,     0,     0,     0,    24,     0,    38,
       0,     0,     0,    26,     0,    27,   189,     0,   204,   152,
       0,   114,     0,     0,   140,     0,    89,    90,    91,    92,
      93,    94,     0,   196,   197,   198,   199,   200,   201,   202,
       0,   206,     0,     0,   207,     0,     0,   208,     0,     0,
       0,     0,   210,     0,   224,     0,     0,     0,     0,     0,
       2,     3,     0,   214,   215,     4,   160,     0,   216,     0,
     218,     5,     6,     7,     8,     9,    10,     0,    11,    12,
       0,     0,     0,    13,    14,    15,    16,    17,    18,    19,
      20,    89,    90,    91,    92,    93,    94,     0,    21,    22,
       0,    23,     0,     0,     0,    24,    -8,    25,    -8,     0,
       3,    26,     0,    27,     4,     0,     0,     0,     0,     0,
       5,     6,     7,     8,     9,    10,     0,    11,    12,     0,
       0,   185,    13,    14,    15,    16,     0,    18,    19,    20,
      89,    90,    91,    92,    93,    94,   212,    21,    22,     0,
      23,    -8,     0,     0,    24,    -8,    25,    -8,     0,     3,
      26,     0,    27,     4,     0,     0,     0,     0,     0,     5,
       6,     7,     8,     9,    10,     0,    11,    12,     0,     0,
     194,    13,    14,    15,    16,     0,    18,    19,    20,    89,
      90,    91,    92,    93,    94,     0,    21,    22,     0,    23,
      -8,     0,     0,    24,    -8,    25,    -8,     4,     0,    26,
       0,    27,     0,    34,     6,     7,     8,     0,    89,    90,
      91,    92,    93,    94,     0,    13,    14,     4,     0,   195,
      35,    36,    37,    34,     6,     7,     8,     0,     0,     0,
      21,     0,   125,     0,     0,    13,    14,    24,     0,    38,
      35,    36,    37,    26,   137,    27,     0,     0,     0,     0,
      21,     0,     0,     0,     0,     0,   190,    24,     0,    38,
       4,     0,     0,    26,     0,    27,    34,     6,     7,     8,
       0,    89,    90,    91,    92,    93,    94,     0,    13,    14,
       4,     0,     0,    35,    36,    37,   112,     6,     7,     8,
       0,     0,     0,    21,     0,     0,     0,     0,    13,    14,
      24,     0,    38,    35,    36,    37,    26,   164,    27,    40,
       0,     0,     0,    21,     0,   100,     0,     0,     0,     0,
      24,     0,    38,     0,     0,     0,    26,     0,    27,    41,
      42,     0,   100,     0,     0,    41,    42,    43,    44,    45,
      46,    47,    48,    43,    44,    45,    46,    47,    48,    49,
      75,    50,    41,    42,     0,   101,   107,    50,     0,     0,
      43,    44,    45,    46,    47,    48,     0,     0,     0,     0,
      76,    77,   161,     0,    50,     0,    76,    77,    43,    44,
      45,    46,    47,    48,    43,    44,    45,    46,    47,    48,
       0,     0,    78,     0,     0,     0,     0,     0,    78,    89,
      90,    91,    92,    93,    94,    89,    90,    91,    92,    93,
      94,    89,    90,    91,    92,    93,    94,    89,    90,    91,
      92,    93,    94,     0,     0,     0,     0,     0,    67,     0,
       0,     0,     0,     0,    71,   165,     0,     0,     0,     0,
       0,   171,     0,     0,     0,     0,     0,   172,    68,    69,
     103,     0,     0,   211,    72,    73,    43,    44,    45,    46,
      47,    48,    43,    44,    45,    46,    47,    48,   105,     0,
      68,    69,     0,     0,   186,     0,     0,     0,    43,    44,
      45,    46,    47,    48,     0,     0,     0,     0,    72,    73,
     205,     0,     0,     0,   187,   188,    43,    44,    45,    46,
      47,    48,    43,    44,    45,    46,    47,    48,     0,     0,
     187,   188,    89,    90,    91,    92,    93,    94,    43,    44,
      45,    46,    47,    48
};

static const yytype_int16 yycheck[] =
{
       1,     5,     4,    28,    21,    12,    40,    40,   150,    46,
      45,    48,    11,   121,    18,    19,    20,    11,    52,    52,
      11,    45,    24,    45,     4,     5,     6,     7,     8,     9,
      29,    35,    36,    37,    28,    29,    30,   210,    40,     4,
       5,     6,     7,     8,     9,   218,    53,    49,    50,    51,
      52,    53,    17,    45,    56,    20,    21,    22,   166,     6,
       7,     8,     9,    65,    38,    67,    46,    11,    70,    71,
     212,    12,    74,    75,    11,    11,    78,    79,    95,    45,
      43,   223,    43,    45,    49,    50,    88,    89,    90,    91,
      92,    93,    94,    11,    43,     3,    46,    45,   100,   101,
     102,   103,   104,   105,   106,   107,   108,     5,     9,    44,
      28,    29,    30,    11,    12,    13,    14,    41,    46,   121,
      48,   123,   124,    42,    52,    23,    24,    46,    40,    48,
      28,    29,    30,    42,    40,   136,   138,    46,   203,    48,
      38,    -1,    40,   163,    -1,    -1,    -1,    45,    -1,    47,
      -1,    -1,    -1,    51,    -1,    53,   160,    -1,   183,   161,
      -1,   163,    -1,    -1,   166,    -1,     4,     5,     6,     7,
       8,     9,    -1,   175,   176,   177,   178,   179,   180,   181,
      -1,   185,    -1,    -1,   186,    -1,    -1,   189,    -1,    -1,
      -1,    -1,   193,    -1,   219,    -1,    -1,    -1,    -1,    -1,
       0,     1,    -1,   205,   206,     5,    44,    -1,   209,    -1,
     211,    11,    12,    13,    14,    15,    16,    -1,    18,    19,
      -1,    -1,    -1,    23,    24,    25,    26,    27,    28,    29,
      30,     4,     5,     6,     7,     8,     9,    -1,    38,    39,
      -1,    41,    -1,    -1,    -1,    45,    46,    47,    48,    -1,
       1,    51,    -1,    53,     5,    -1,    -1,    -1,    -1,    -1,
      11,    12,    13,    14,    15,    16,    -1,    18,    19,    -1,
      -1,    44,    23,    24,    25,    26,    -1,    28,    29,    30,
       4,     5,     6,     7,     8,     9,    37,    38,    39,    -1,
      41,    42,    -1,    -1,    45,    46,    47,    48,    -1,     1,
      51,    -1,    53,     5,    -1,    -1,    -1,    -1,    -1,    11,
      12,    13,    14,    15,    16,    -1,    18,    19,    -1,    -1,
      44,    23,    24,    25,    26,    -1,    28,    29,    30,     4,
       5,     6,     7,     8,     9,    -1,    38,    39,    -1,    41,
      42,    -1,    -1,    45,    46,    47,    48,     5,    -1,    51,
      -1,    53,    -1,    11,    12,    13,    14,    -1,     4,     5,
       6,     7,     8,     9,    -1,    23,    24,     5,    -1,    44,
      28,    29,    30,    11,    12,    13,    14,    -1,    -1,    -1,
      38,    -1,    40,    -1,    -1,    23,    24,    45,    -1,    47,
      28,    29,    30,    51,    40,    53,    -1,    -1,    -1,    -1,
      38,    -1,    -1,    -1,    -1,    -1,    44,    45,    -1,    47,
       5,    -1,    -1,    51,    -1,    53,    11,    12,    13,    14,
      -1,     4,     5,     6,     7,     8,     9,    -1,    23,    24,
       5,    -1,    -1,    28,    29,    30,    11,    12,    13,    14,
      -1,    -1,    -1,    38,    -1,    -1,    -1,    -1,    23,    24,
      45,    -1,    47,    28,    29,    30,    51,    40,    53,     3,
      -1,    -1,    -1,    38,    -1,     3,    -1,    -1,    -1,    -1,
      45,    -1,    47,    -1,    -1,    -1,    51,    -1,    53,    23,
      24,    -1,     3,    -1,    -1,    23,    24,    31,    32,    33,
      34,    35,    36,    31,    32,    33,    34,    35,    36,    43,
       3,    45,    23,    24,    -1,    43,     3,    45,    -1,    -1,
      31,    32,    33,    34,    35,    36,    -1,    -1,    -1,    -1,
      23,    24,    43,    -1,    45,    -1,    23,    24,    31,    32,
      33,    34,    35,    36,    31,    32,    33,    34,    35,    36,
      -1,    -1,    45,    -1,    -1,    -1,    -1,    -1,    45,     4,
       5,     6,     7,     8,     9,     4,     5,     6,     7,     8,
       9,     4,     5,     6,     7,     8,     9,     4,     5,     6,
       7,     8,     9,    -1,    -1,    -1,    -1,    -1,     3,    -1,
      -1,    -1,    -1,    -1,     3,    40,    -1,    -1,    -1,    -1,
      -1,    40,    -1,    -1,    -1,    -1,    -1,    40,    23,    24,
       3,    -1,    -1,    40,    23,    24,    31,    32,    33,    34,
      35,    36,    31,    32,    33,    34,    35,    36,     3,    -1,
      23,    24,    -1,    -1,     3,    -1,    -1,    -1,    31,    32,
      33,    34,    35,    36,    -1,    -1,    -1,    -1,    23,    24,
       3,    -1,    -1,    -1,    23,    24,    31,    32,    33,    34,
      35,    36,    31,    32,    33,    34,    35,    36,    -1,    -1,
      23,    24,     4,     5,     6,     7,     8,     9,    31,    32,
      33,    34,    35,    36
};

  /* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
     symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,    55,     0,     1,     5,    11,    12,    13,    14,    15,
      16,    18,    19,    23,    24,    25,    26,    27,    28,    29,
      30,    38,    39,    41,    45,    47,    51,    53,    57,    59,
      64,    67,    68,    70,    11,    28,    29,    30,    47,    64,
       3,    23,    24,    31,    32,    33,    34,    35,    36,    43,
      45,    58,    45,    45,    69,    69,    45,    11,    28,    29,
      30,    11,    28,    29,    30,    45,    11,     3,    23,    24,
      58,     3,    23,    24,    58,     3,    23,    24,    45,    58,
      67,    57,    61,    64,    11,    46,    48,    62,    69,     4,
       5,     6,     7,     8,     9,    38,    12,    11,    71,    73,
       3,    43,    58,     3,    58,     3,    58,     3,    58,    11,
      64,    64,    11,    40,    64,    65,    66,    64,    64,    64,
      60,    45,    64,    43,    43,    40,    64,    45,    64,    64,
      64,    64,    64,    64,    64,    42,    62,    40,     3,    63,
      64,    64,    64,    64,    64,    64,    64,    67,    43,    40,
      52,    64,    64,    64,    64,    64,    64,    64,    64,    64,
      44,    43,    40,    52,    40,    40,    45,    63,    46,    64,
      64,    40,    40,    57,    64,    46,    17,    20,    21,    22,
      49,    50,    44,    41,    73,    44,     3,    23,    24,    58,
      44,    66,    63,    40,    44,    44,    64,    64,    64,    64,
      64,    64,    64,    56,    62,     3,    58,    64,    64,    40,
      57,    40,    37,    61,    64,    64,    57,    60,    57,    72,
      73,    42,    60,    52,    62,    73
};

  /* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    54,    55,    55,    55,    56,    56,    57,    57,    57,
      57,    57,    57,    57,    57,    57,    57,    57,    57,    57,
      57,    57,    57,    57,    57,    57,    57,    57,    57,    57,
      57,    58,    58,    58,    58,    58,    58,    59,    60,    61,
      61,    62,    62,    63,    63,    63,    63,    63,    63,    63,
      64,    64,    64,    64,    64,    64,    64,    64,    64,    64,
      64,    64,    64,    64,    64,    64,    64,    64,    64,    64,
      64,    64,    64,    64,    64,    64,    64,    64,    64,    64,
      64,    64,    64,    64,    64,    64,    64,    64,    64,    64,
      64,    64,    64,    64,    64,    64,    64,    64,    64,    64,
      64,    64,    64,    64,    64,    65,    65,    66,    66,    67,
      68,    68,    68,    69,    70,    71,    71,    71,    72,    72,
      73,    73
};

  /* YYR2[YYN] -- Number of symbols on the right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     0,     3,     8,     1,     4,     1,     0,     1,
       3,     6,     3,     6,     1,     4,     3,     1,     3,     3,
       3,     3,     3,     3,     3,     1,     1,     7,     7,     8,
       4,     1,     1,     1,     1,     1,     1,     4,     0,     1,
       3,     1,     1,     3,     3,     3,     3,     3,     3,     1,
       3,     3,     2,     3,     3,     3,     3,     4,     2,     2,
       2,     2,     5,     5,     5,     5,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     4,     3,
       1,     2,     3,     2,     1,     1,     3,     3,     6,     6,
       4,     4,     3,     1,     4,     2,     3,     3,     3,     3,
       3,     3,     1,     1,     1,     1,     3,     1,     3,     1,
       1,     1,     2,     0,     3,     0,     1,     3,     1,     3,
       1,     3
};


#define yyerrok         (yyerrstatus = 0)
#define yyclearin       (yychar = YYEMPTY)
#define YYEMPTY         (-2)
#define YYEOF           0

#define YYACCEPT        goto yyacceptlab
#define YYABORT         goto yyabortlab
#define YYERROR         goto yyerrorlab


#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)                                  \
do                                                              \
  if (yychar == YYEMPTY)                                        \
    {                                                           \
      yychar = (Token);                                         \
      yylval = (Value);                                         \
      YYPOPSTACK (yylen);                                       \
      yystate = *yyssp;                                         \
      goto yybackup;                                            \
    }                                                           \
  else                                                          \
    {                                                           \
      yyerror (YY_("syntax error: cannot back up")); \
      YYERROR;                                                  \
    }                                                           \
while (0)

/* Error token number */
#define YYTERROR        1
#define YYERRCODE       256



/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)                        \
do {                                            \
  if (yydebug)                                  \
    YYFPRINTF Args;                             \
} while (0)

/* This macro is provided for backward compatibility. */
#ifndef YY_LOCATION_PRINT
# define YY_LOCATION_PRINT(File, Loc) ((void) 0)
#endif


# define YY_SYMBOL_PRINT(Title, Type, Value, Location)                    \
do {                                                                      \
  if (yydebug)                                                            \
    {                                                                     \
      YYFPRINTF (stderr, "%s ", Title);                                   \
      yy_symbol_print (stderr,                                            \
                  Type, Value); \
      YYFPRINTF (stderr, "\n");                                           \
    }                                                                     \
} while (0)


/*----------------------------------------.
| Print this symbol's value on YYOUTPUT.  |
`----------------------------------------*/

static void
yy_symbol_value_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
{
  FILE *yyo = yyoutput;
  YYUSE (yyo);
  if (!yyvaluep)
    return;
# ifdef YYPRINT
  if (yytype < YYNTOKENS)
    YYPRINT (yyoutput, yytoknum[yytype], *yyvaluep);
# endif
  YYUSE (yytype);
}


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

static void
yy_symbol_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
{
  YYFPRINTF (yyoutput, "%s %s (",
             yytype < YYNTOKENS ? "token" : "nterm", yytname[yytype]);

  yy_symbol_value_print (yyoutput, yytype, yyvaluep);
  YYFPRINTF (yyoutput, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

static void
yy_stack_print (yytype_int16 *yybottom, yytype_int16 *yytop)
{
  YYFPRINTF (stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++)
    {
      int yybot = *yybottom;
      YYFPRINTF (stderr, " %d", yybot);
    }
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)                            \
do {                                                            \
  if (yydebug)                                                  \
    yy_stack_print ((Bottom), (Top));                           \
} while (0)


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

static void
yy_reduce_print (yytype_int16 *yyssp, YYSTYPE *yyvsp, int yyrule)
{
  unsigned long int yylno = yyrline[yyrule];
  int yynrhs = yyr2[yyrule];
  int yyi;
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %lu):\n",
             yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr,
                       yystos[yyssp[yyi + 1 - yynrhs]],
                       &(yyvsp[(yyi + 1) - (yynrhs)])
                                              );
      YYFPRINTF (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)          \
do {                                    \
  if (yydebug)                          \
    yy_reduce_print (yyssp, yyvsp, Rule); \
} while (0)

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args)
# define YY_SYMBOL_PRINT(Title, Type, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif


#if YYERROR_VERBOSE

# ifndef yystrlen
#  if defined __GLIBC__ && defined _STRING_H
#   define yystrlen strlen
#  else
/* Return the length of YYSTR.  */
static YYSIZE_T
yystrlen (const char *yystr)
{
  YYSIZE_T yylen;
  for (yylen = 0; yystr[yylen]; yylen++)
    continue;
  return yylen;
}
#  endif
# endif

# ifndef yystpcpy
#  if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#   define yystpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
static char *
yystpcpy (char *yydest, const char *yysrc)
{
  char *yyd = yydest;
  const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
#  endif
# endif

# ifndef yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYSIZE_T
yytnamerr (char *yyres, const char *yystr)
{
  if (*yystr == '"')
    {
      YYSIZE_T yyn = 0;
      char const *yyp = yystr;

      for (;;)
        switch (*++yyp)
          {
          case '\'':
          case ',':
            goto do_not_strip_quotes;

          case '\\':
            if (*++yyp != '\\')
              goto do_not_strip_quotes;
            /* Fall through.  */
          default:
            if (yyres)
              yyres[yyn] = *yyp;
            yyn++;
            break;

          case '"':
            if (yyres)
              yyres[yyn] = '\0';
            return yyn;
          }
    do_not_strip_quotes: ;
    }

  if (! yyres)
    return yystrlen (yystr);

  return yystpcpy (yyres, yystr) - yyres;
}
# endif

/* Copy into *YYMSG, which is of size *YYMSG_ALLOC, an error message
   about the unexpected token YYTOKEN for the state stack whose top is
   YYSSP.

   Return 0 if *YYMSG was successfully written.  Return 1 if *YYMSG is
   not large enough to hold the message.  In that case, also set
   *YYMSG_ALLOC to the required number of bytes.  Return 2 if the
   required number of bytes is too large to store.  */
static int
yysyntax_error (YYSIZE_T *yymsg_alloc, char **yymsg,
                yytype_int16 *yyssp, int yytoken)
{
  YYSIZE_T yysize0 = yytnamerr (YY_NULLPTR, yytname[yytoken]);
  YYSIZE_T yysize = yysize0;
  enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
  /* Internationalized format string. */
  const char *yyformat = YY_NULLPTR;
  /* Arguments of yyformat. */
  char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
  /* Number of reported tokens (one for the "unexpected", one per
     "expected"). */
  int yycount = 0;

  /* There are many possibilities here to consider:
     - If this state is a consistent state with a default action, then
       the only way this function was invoked is if the default action
       is an error action.  In that case, don't check for expected
       tokens because there are none.
     - The only way there can be no lookahead present (in yychar) is if
       this state is a consistent state with a default action.  Thus,
       detecting the absence of a lookahead is sufficient to determine
       that there is no unexpected or expected token to report.  In that
       case, just report a simple "syntax error".
     - Don't assume there isn't a lookahead just because this state is a
       consistent state with a default action.  There might have been a
       previous inconsistent state, consistent state with a non-default
       action, or user semantic action that manipulated yychar.
     - Of course, the expected token list depends on states to have
       correct lookahead information, and it depends on the parser not
       to perform extra reductions after fetching a lookahead from the
       scanner and before detecting a syntax error.  Thus, state merging
       (from LALR or IELR) and default reductions corrupt the expected
       token list.  However, the list is correct for canonical LR with
       one exception: it will still contain any token that will not be
       accepted due to an error action in a later state.
  */
  if (yytoken != YYEMPTY)
    {
      int yyn = yypact[*yyssp];
      yyarg[yycount++] = yytname[yytoken];
      if (!yypact_value_is_default (yyn))
        {
          /* Start YYX at -YYN if negative to avoid negative indexes in
             YYCHECK.  In other words, skip the first -YYN actions for
             this state because they are default actions.  */
          int yyxbegin = yyn < 0 ? -yyn : 0;
          /* Stay within bounds of both yycheck and yytname.  */
          int yychecklim = YYLAST - yyn + 1;
          int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
          int yyx;

          for (yyx = yyxbegin; yyx < yyxend; ++yyx)
            if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR
                && !yytable_value_is_error (yytable[yyx + yyn]))
              {
                if (yycount == YYERROR_VERBOSE_ARGS_MAXIMUM)
                  {
                    yycount = 1;
                    yysize = yysize0;
                    break;
                  }
                yyarg[yycount++] = yytname[yyx];
                {
                  YYSIZE_T yysize1 = yysize + yytnamerr (YY_NULLPTR, yytname[yyx]);
                  if (! (yysize <= yysize1
                         && yysize1 <= YYSTACK_ALLOC_MAXIMUM))
                    return 2;
                  yysize = yysize1;
                }
              }
        }
    }

  switch (yycount)
    {
# define YYCASE_(N, S)                      \
      case N:                               \
        yyformat = S;                       \
      break
      YYCASE_(0, YY_("syntax error"));
      YYCASE_(1, YY_("syntax error, unexpected %s"));
      YYCASE_(2, YY_("syntax error, unexpected %s, expecting %s"));
      YYCASE_(3, YY_("syntax error, unexpected %s, expecting %s or %s"));
      YYCASE_(4, YY_("syntax error, unexpected %s, expecting %s or %s or %s"));
      YYCASE_(5, YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s"));
# undef YYCASE_
    }

  {
    YYSIZE_T yysize1 = yysize + yystrlen (yyformat);
    if (! (yysize <= yysize1 && yysize1 <= YYSTACK_ALLOC_MAXIMUM))
      return 2;
    yysize = yysize1;
  }

  if (*yymsg_alloc < yysize)
    {
      *yymsg_alloc = 2 * yysize;
      if (! (yysize <= *yymsg_alloc
             && *yymsg_alloc <= YYSTACK_ALLOC_MAXIMUM))
        *yymsg_alloc = YYSTACK_ALLOC_MAXIMUM;
      return 1;
    }

  /* Avoid sprintf, as that infringes on the user's name space.
     Don't have undefined behavior even if the translation
     produced a string with the wrong number of "%s"s.  */
  {
    char *yyp = *yymsg;
    int yyi = 0;
    while ((*yyp = *yyformat) != '\0')
      if (*yyp == '%' && yyformat[1] == 's' && yyi < yycount)
        {
          yyp += yytnamerr (yyp, yyarg[yyi++]);
          yyformat += 2;
        }
      else
        {
          yyp++;
          yyformat++;
        }
  }
  return 0;
}
#endif /* YYERROR_VERBOSE */

/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

static void
yydestruct (const char *yymsg, int yytype, YYSTYPE *yyvaluep)
{
  YYUSE (yyvaluep);
  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yytype, yyvaluep, yylocationp);

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YYUSE (yytype);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}




/* The lookahead symbol.  */
int yychar;

/* The semantic value of the lookahead symbol.  */
YYSTYPE yylval;
/* Number of syntax errors so far.  */
int yynerrs;


/*----------.
| yyparse.  |
`----------*/

int
yyparse (void)
{
    int yystate;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus;

    /* The stacks and their tools:
       'yyss': related to states.
       'yyvs': related to semantic values.

       Refer to the stacks through separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* The state stack.  */
    yytype_int16 yyssa[YYINITDEPTH];
    yytype_int16 *yyss;
    yytype_int16 *yyssp;

    /* The semantic value stack.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs;
    YYSTYPE *yyvsp;

    YYSIZE_T yystacksize;

  int yyn;
  int yyresult;
  /* Lookahead token as an internal (translated) token number.  */
  int yytoken = 0;
  /* The variables used to return semantic value and location from the
     action routines.  */


#if YYERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char *yymsg = yymsgbuf;
  YYSIZE_T yymsg_alloc = sizeof yymsgbuf;
#endif

#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  yyssp = yyss = yyssa;
  yyvsp = yyvs = yyvsa;
  yystacksize = YYINITDEPTH;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY; /* Cause a token to be read.  */
  goto yysetstate;

/*------------------------------------------------------------.
| yynewstate -- Push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
 yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;

 yysetstate:
  *yyssp = yystate;

  if (yyss + yystacksize - 1 <= yyssp)
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYSIZE_T yysize = yyssp - yyss + 1;

#ifdef yyoverflow
      {
        /* Give user a chance to reallocate the stack.  Use copies of
           these so that the &'s don't force the real ones into
           memory.  */
        YYSTYPE *yyvs1 = yyvs;
        yytype_int16 *yyss1 = yyss;

        /* Each stack pointer address is followed by the size of the
           data in use in that stack, in bytes.  This used to be a
           conditional around just the two extra args, but that might
           be undefined if yyoverflow is a macro.  */
        yyoverflow (YY_("memory exhausted"),
                    &yyss1, yysize * sizeof (*yyssp),
                    &yyvs1, yysize * sizeof (*yyvsp),
                    &yystacksize);

        yyss = yyss1;
        yyvs = yyvs1;
      }
#else /* no yyoverflow */
# ifndef YYSTACK_RELOCATE
      goto yyexhaustedlab;
# else
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
        goto yyexhaustedlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
        yystacksize = YYMAXDEPTH;

      {
        yytype_int16 *yyss1 = yyss;
        union yyalloc *yyptr =
          (union yyalloc *) YYSTACK_ALLOC (YYSTACK_BYTES (yystacksize));
        if (! yyptr)
          goto yyexhaustedlab;
        YYSTACK_RELOCATE (yyss_alloc, yyss);
        YYSTACK_RELOCATE (yyvs_alloc, yyvs);
#  undef YYSTACK_RELOCATE
        if (yyss1 != yyssa)
          YYSTACK_FREE (yyss1);
      }
# endif
#endif /* no yyoverflow */

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;

      YYDPRINTF ((stderr, "Stack size increased to %lu\n",
                  (unsigned long int) yystacksize));

      if (yyss + yystacksize - 1 <= yyssp)
        YYABORT;
    }

  YYDPRINTF ((stderr, "Entering state %d\n", yystate));

  if (yystate == YYFINAL)
    YYACCEPT;

  goto yybackup;

/*-----------.
| yybackup.  |
`-----------*/
yybackup:

  /* Do appropriate processing given the current state.  Read a
     lookahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to lookahead token.  */
  yyn = yypact[yystate];
  if (yypact_value_is_default (yyn))
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid lookahead symbol.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token: "));
      yychar = yylex ();
    }

  if (yychar <= YYEOF)
    {
      yychar = yytoken = YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yytable_value_is_error (yyn))
        goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the lookahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);

  /* Discard the shifted token.  */
  yychar = YYEMPTY;

  yystate = yyn;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END

  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- Do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     '$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
        case 3:
#line 118 "bc.y" /* yacc.c:1646  */
    { output( (intptr_t *)(yyvsp[-1]) );}
#line 1598 "y.tab.c" /* yacc.c:1646  */
    break;

  case 4:
#line 120 "bc.y" /* yacc.c:1646  */
    {	bundle( 6,pre, (yyvsp[-1]), post ,"0",numb[lev],"Q");
			conout( (yyval), (yyvsp[-6]) );
			rcrs = crs;
			output( (intptr_t *)"" );
			lev = bindx = 0;
			}
#line 1609 "y.tab.c" /* yacc.c:1646  */
    break;

  case 7:
#line 133 "bc.y" /* yacc.c:1646  */
    { bundle(2, (yyvsp[0]), "ps." ); }
#line 1615 "y.tab.c" /* yacc.c:1646  */
    break;

  case 8:
#line 135 "bc.y" /* yacc.c:1646  */
    { bundle(1, "" ); }
#line 1621 "y.tab.c" /* yacc.c:1646  */
    break;

  case 9:
#line 137 "bc.y" /* yacc.c:1646  */
    { bundle(3,"[",(yyvsp[0]),"]P");}
#line 1627 "y.tab.c" /* yacc.c:1646  */
    break;

  case 10:
#line 139 "bc.y" /* yacc.c:1646  */
    { bundle(3, (yyvsp[0]), "s", (yyvsp[-2]) ); }
#line 1633 "y.tab.c" /* yacc.c:1646  */
    break;

  case 11:
#line 141 "bc.y" /* yacc.c:1646  */
    { bundle(4, (yyvsp[0]), (yyvsp[-3]), ":", geta((yyvsp[-5]))); }
#line 1639 "y.tab.c" /* yacc.c:1646  */
    break;

  case 12:
#line 143 "bc.y" /* yacc.c:1646  */
    { bundle(6, "l", (yyvsp[-2]), (yyvsp[0]), (yyvsp[-1]), "s", (yyvsp[-2]) ); }
#line 1645 "y.tab.c" /* yacc.c:1646  */
    break;

  case 13:
#line 145 "bc.y" /* yacc.c:1646  */
    { bundle(8,(yyvsp[-3]), ";", geta((yyvsp[-5])), (yyvsp[0]), (yyvsp[-1]), (yyvsp[-3]), ":", geta((yyvsp[-5])));}
#line 1651 "y.tab.c" /* yacc.c:1646  */
    break;

  case 14:
#line 147 "bc.y" /* yacc.c:1646  */
    { bundle(2, numb[lev-bstack[bindx-1]], "Q" ); }
#line 1657 "y.tab.c" /* yacc.c:1646  */
    break;

  case 15:
#line 149 "bc.y" /* yacc.c:1646  */
    { bundle(4, (yyvsp[-1]), post, numb[lev], "Q" ); }
#line 1663 "y.tab.c" /* yacc.c:1646  */
    break;

  case 16:
#line 151 "bc.y" /* yacc.c:1646  */
    { bundle(4, "0", post, numb[lev], "Q" ); }
#line 1669 "y.tab.c" /* yacc.c:1646  */
    break;

  case 17:
#line 153 "bc.y" /* yacc.c:1646  */
    { bundle(4,"0",post,numb[lev],"Q"); }
#line 1675 "y.tab.c" /* yacc.c:1646  */
    break;

  case 18:
#line 155 "bc.y" /* yacc.c:1646  */
    { bundle(2, (yyvsp[0]), "k"); }
#line 1681 "y.tab.c" /* yacc.c:1646  */
    break;

  case 19:
#line 157 "bc.y" /* yacc.c:1646  */
    { bundle(4,"K",(yyvsp[0]),(yyvsp[-1]),"k"); }
#line 1687 "y.tab.c" /* yacc.c:1646  */
    break;

  case 20:
#line 159 "bc.y" /* yacc.c:1646  */
    { bundle(2,(yyvsp[0]), "i"); }
#line 1693 "y.tab.c" /* yacc.c:1646  */
    break;

  case 21:
#line 161 "bc.y" /* yacc.c:1646  */
    { bundle(4,"I",(yyvsp[0]),(yyvsp[-1]),"i"); }
#line 1699 "y.tab.c" /* yacc.c:1646  */
    break;

  case 22:
#line 163 "bc.y" /* yacc.c:1646  */
    { bundle(2,(yyvsp[0]),"o"); }
#line 1705 "y.tab.c" /* yacc.c:1646  */
    break;

  case 23:
#line 165 "bc.y" /* yacc.c:1646  */
    { bundle(4,"O",(yyvsp[0]),(yyvsp[-1]),"o"); }
#line 1711 "y.tab.c" /* yacc.c:1646  */
    break;

  case 24:
#line 167 "bc.y" /* yacc.c:1646  */
    { (yyval) = (yyvsp[-1]); }
#line 1717 "y.tab.c" /* yacc.c:1646  */
    break;

  case 25:
#line 169 "bc.y" /* yacc.c:1646  */
    { bundle(1,"fY"); }
#line 1723 "y.tab.c" /* yacc.c:1646  */
    break;

  case 26:
#line 171 "bc.y" /* yacc.c:1646  */
    { bundle(1,"c"); }
#line 1729 "y.tab.c" /* yacc.c:1646  */
    break;

  case 27:
#line 173 "bc.y" /* yacc.c:1646  */
    {	conout( (yyvsp[0]), (yyvsp[-5]) );
			bundle(3, (yyvsp[-2]), (yyvsp[-5]), " " );
			}
#line 1737 "y.tab.c" /* yacc.c:1646  */
    break;

  case 28:
#line 177 "bc.y" /* yacc.c:1646  */
    {	bundle(3, (yyvsp[-1]), (yyvsp[-3]), (yyvsp[-5]) );
			conout( (yyval), (yyvsp[-5]) );
			bundle(3, (yyvsp[-3]), (yyvsp[-5]), " " );
			}
#line 1746 "y.tab.c" /* yacc.c:1646  */
    break;

  case 29:
#line 182 "bc.y" /* yacc.c:1646  */
    {	bundle(5, (yyvsp[-1]), (yyvsp[-3]), "s.", (yyvsp[-5]), (yyvsp[-6]) );
			conout( (yyval), (yyvsp[-6]) );
			bundle(5, (yyvsp[-7]), "s.", (yyvsp[-5]), (yyvsp[-6]), " " );
			}
#line 1755 "y.tab.c" /* yacc.c:1646  */
    break;

  case 30:
#line 187 "bc.y" /* yacc.c:1646  */
    {	bundle(3,(yyvsp[0]),"S",(yyvsp[-2])); }
#line 1761 "y.tab.c" /* yacc.c:1646  */
    break;

  case 31:
#line 191 "bc.y" /* yacc.c:1646  */
    { (yyval) = (intptr_t)"+"; }
#line 1767 "y.tab.c" /* yacc.c:1646  */
    break;

  case 32:
#line 193 "bc.y" /* yacc.c:1646  */
    { (yyval) = (intptr_t)"-"; }
#line 1773 "y.tab.c" /* yacc.c:1646  */
    break;

  case 33:
#line 195 "bc.y" /* yacc.c:1646  */
    { (yyval) = (intptr_t)"*"; }
#line 1779 "y.tab.c" /* yacc.c:1646  */
    break;

  case 34:
#line 197 "bc.y" /* yacc.c:1646  */
    { (yyval) = (intptr_t)"/"; }
#line 1785 "y.tab.c" /* yacc.c:1646  */
    break;

  case 35:
#line 199 "bc.y" /* yacc.c:1646  */
    { (yyval) = (intptr_t)"%%"; }
#line 1791 "y.tab.c" /* yacc.c:1646  */
    break;

  case 36:
#line 201 "bc.y" /* yacc.c:1646  */
    { (yyval) = (intptr_t)"^"; }
#line 1797 "y.tab.c" /* yacc.c:1646  */
    break;

  case 37:
#line 205 "bc.y" /* yacc.c:1646  */
    { (yyval) = (yyvsp[-1]); }
#line 1803 "y.tab.c" /* yacc.c:1646  */
    break;

  case 38:
#line 209 "bc.y" /* yacc.c:1646  */
    { --bindx; }
#line 1809 "y.tab.c" /* yacc.c:1646  */
    break;

  case 40:
#line 214 "bc.y" /* yacc.c:1646  */
    { bundle(2, (yyvsp[-2]), (yyvsp[0]) ); }
#line 1815 "y.tab.c" /* yacc.c:1646  */
    break;

  case 41:
#line 218 "bc.y" /* yacc.c:1646  */
    {ln++;}
#line 1821 "y.tab.c" /* yacc.c:1646  */
    break;

  case 43:
#line 223 "bc.y" /* yacc.c:1646  */
    { bundle(3, (yyvsp[-2]), (yyvsp[0]), "=" ); }
#line 1827 "y.tab.c" /* yacc.c:1646  */
    break;

  case 44:
#line 225 "bc.y" /* yacc.c:1646  */
    { bundle(3, (yyvsp[-2]), (yyvsp[0]), ">" ); }
#line 1833 "y.tab.c" /* yacc.c:1646  */
    break;

  case 45:
#line 227 "bc.y" /* yacc.c:1646  */
    { bundle(3, (yyvsp[-2]), (yyvsp[0]), "<" ); }
#line 1839 "y.tab.c" /* yacc.c:1646  */
    break;

  case 46:
#line 229 "bc.y" /* yacc.c:1646  */
    { bundle(3, (yyvsp[-2]), (yyvsp[0]), "!=" ); }
#line 1845 "y.tab.c" /* yacc.c:1646  */
    break;

  case 47:
#line 231 "bc.y" /* yacc.c:1646  */
    { bundle(3, (yyvsp[-2]), (yyvsp[0]), "!>" ); }
#line 1851 "y.tab.c" /* yacc.c:1646  */
    break;

  case 48:
#line 233 "bc.y" /* yacc.c:1646  */
    { bundle(3, (yyvsp[-2]), (yyvsp[0]), "!<" ); }
#line 1857 "y.tab.c" /* yacc.c:1646  */
    break;

  case 49:
#line 235 "bc.y" /* yacc.c:1646  */
    { bundle(2, (yyvsp[0]), " 0!=" ); }
#line 1863 "y.tab.c" /* yacc.c:1646  */
    break;

  case 50:
#line 239 "bc.y" /* yacc.c:1646  */
    { bundle(3, (yyvsp[-2]), (yyvsp[0]), "+" ); }
#line 1869 "y.tab.c" /* yacc.c:1646  */
    break;

  case 51:
#line 241 "bc.y" /* yacc.c:1646  */
    { bundle(3, (yyvsp[-2]), (yyvsp[0]), "-" ); }
#line 1875 "y.tab.c" /* yacc.c:1646  */
    break;

  case 52:
#line 243 "bc.y" /* yacc.c:1646  */
    { bundle(3, " 0", (yyvsp[0]), "-" ); }
#line 1881 "y.tab.c" /* yacc.c:1646  */
    break;

  case 53:
#line 245 "bc.y" /* yacc.c:1646  */
    { bundle(3, (yyvsp[-2]), (yyvsp[0]), "*" ); }
#line 1887 "y.tab.c" /* yacc.c:1646  */
    break;

  case 54:
#line 247 "bc.y" /* yacc.c:1646  */
    { bundle(3, (yyvsp[-2]), (yyvsp[0]), "/" ); }
#line 1893 "y.tab.c" /* yacc.c:1646  */
    break;

  case 55:
#line 249 "bc.y" /* yacc.c:1646  */
    { bundle(3, (yyvsp[-2]), (yyvsp[0]), "%%" ); }
#line 1899 "y.tab.c" /* yacc.c:1646  */
    break;

  case 56:
#line 251 "bc.y" /* yacc.c:1646  */
    { bundle(3, (yyvsp[-2]), (yyvsp[0]), "^" ); }
#line 1905 "y.tab.c" /* yacc.c:1646  */
    break;

  case 57:
#line 253 "bc.y" /* yacc.c:1646  */
    { bundle(3,(yyvsp[-1]), ";", geta((yyvsp[-3]))); }
#line 1911 "y.tab.c" /* yacc.c:1646  */
    break;

  case 58:
#line 255 "bc.y" /* yacc.c:1646  */
    { bundle(4, "l", (yyvsp[-1]), "d1+s", (yyvsp[-1]) ); }
#line 1917 "y.tab.c" /* yacc.c:1646  */
    break;

  case 59:
#line 257 "bc.y" /* yacc.c:1646  */
    { bundle(4, "l", (yyvsp[0]), "1+ds", (yyvsp[0]) ); }
#line 1923 "y.tab.c" /* yacc.c:1646  */
    break;

  case 60:
#line 259 "bc.y" /* yacc.c:1646  */
    { bundle(4, "l", (yyvsp[0]), "1-ds", (yyvsp[0]) ); }
#line 1929 "y.tab.c" /* yacc.c:1646  */
    break;

  case 61:
#line 261 "bc.y" /* yacc.c:1646  */
    { bundle(4, "l", (yyvsp[-1]), "d1-s", (yyvsp[-1]) ); }
#line 1935 "y.tab.c" /* yacc.c:1646  */
    break;

  case 62:
#line 263 "bc.y" /* yacc.c:1646  */
    { bundle(7,(yyvsp[-2]),";",geta((yyvsp[-4])),"d1+",(yyvsp[-2]),":",geta((yyvsp[-4]))); }
#line 1941 "y.tab.c" /* yacc.c:1646  */
    break;

  case 63:
#line 265 "bc.y" /* yacc.c:1646  */
    { bundle(7,(yyvsp[-1]),";",geta((yyvsp[-3])),"1+d",(yyvsp[-1]),":",geta((yyvsp[-3]))); }
#line 1947 "y.tab.c" /* yacc.c:1646  */
    break;

  case 64:
#line 267 "bc.y" /* yacc.c:1646  */
    { bundle(7,(yyvsp[-2]),";",geta((yyvsp[-4])),"d1-",(yyvsp[-2]),":",geta((yyvsp[-4]))); }
#line 1953 "y.tab.c" /* yacc.c:1646  */
    break;

  case 65:
#line 269 "bc.y" /* yacc.c:1646  */
    { bundle(7,(yyvsp[-1]),";",geta((yyvsp[-3])),"1-d",(yyvsp[-1]),":",geta((yyvsp[-3]))); }
#line 1959 "y.tab.c" /* yacc.c:1646  */
    break;

  case 66:
#line 271 "bc.y" /* yacc.c:1646  */
    { bundle(1,"Kd1+k"); }
#line 1965 "y.tab.c" /* yacc.c:1646  */
    break;

  case 67:
#line 273 "bc.y" /* yacc.c:1646  */
    { bundle(1,"K1+dk"); }
#line 1971 "y.tab.c" /* yacc.c:1646  */
    break;

  case 68:
#line 275 "bc.y" /* yacc.c:1646  */
    { bundle(1,"Kd1-k"); }
#line 1977 "y.tab.c" /* yacc.c:1646  */
    break;

  case 69:
#line 277 "bc.y" /* yacc.c:1646  */
    { bundle(1,"K1-dk"); }
#line 1983 "y.tab.c" /* yacc.c:1646  */
    break;

  case 70:
#line 279 "bc.y" /* yacc.c:1646  */
    { bundle(1,"Id1+i"); }
#line 1989 "y.tab.c" /* yacc.c:1646  */
    break;

  case 71:
#line 281 "bc.y" /* yacc.c:1646  */
    { bundle(1,"I1+di"); }
#line 1995 "y.tab.c" /* yacc.c:1646  */
    break;

  case 72:
#line 283 "bc.y" /* yacc.c:1646  */
    { bundle(1,"Id1-i"); }
#line 2001 "y.tab.c" /* yacc.c:1646  */
    break;

  case 73:
#line 285 "bc.y" /* yacc.c:1646  */
    { bundle(1,"I1-di"); }
#line 2007 "y.tab.c" /* yacc.c:1646  */
    break;

  case 74:
#line 287 "bc.y" /* yacc.c:1646  */
    { bundle(1,"Od1+o"); }
#line 2013 "y.tab.c" /* yacc.c:1646  */
    break;

  case 75:
#line 289 "bc.y" /* yacc.c:1646  */
    { bundle(1,"O1+do"); }
#line 2019 "y.tab.c" /* yacc.c:1646  */
    break;

  case 76:
#line 291 "bc.y" /* yacc.c:1646  */
    { bundle(1,"Od1-o"); }
#line 2025 "y.tab.c" /* yacc.c:1646  */
    break;

  case 77:
#line 293 "bc.y" /* yacc.c:1646  */
    { bundle(1,"O1-do"); }
#line 2031 "y.tab.c" /* yacc.c:1646  */
    break;

  case 78:
#line 295 "bc.y" /* yacc.c:1646  */
    { bundle(4, (yyvsp[-1]), "l", getf((yyvsp[-3])), "x" ); }
#line 2037 "y.tab.c" /* yacc.c:1646  */
    break;

  case 79:
#line 297 "bc.y" /* yacc.c:1646  */
    { bundle(3, "l", getf((yyvsp[-2])), "x" ); }
#line 2043 "y.tab.c" /* yacc.c:1646  */
    break;

  case 80:
#line 299 "bc.y" /* yacc.c:1646  */
    { bundle(2, " ", (yyvsp[0]) ); }
#line 2049 "y.tab.c" /* yacc.c:1646  */
    break;

  case 81:
#line 301 "bc.y" /* yacc.c:1646  */
    { bundle(2, " .", (yyvsp[0]) ); }
#line 2055 "y.tab.c" /* yacc.c:1646  */
    break;

  case 82:
#line 303 "bc.y" /* yacc.c:1646  */
    { bundle(4, " ", (yyvsp[-2]), ".", (yyvsp[0]) ); }
#line 2061 "y.tab.c" /* yacc.c:1646  */
    break;

  case 83:
#line 305 "bc.y" /* yacc.c:1646  */
    { bundle(3, " ", (yyvsp[-1]), "." ); }
#line 2067 "y.tab.c" /* yacc.c:1646  */
    break;

  case 84:
#line 307 "bc.y" /* yacc.c:1646  */
    { (yyval) = (intptr_t)"l."; }
#line 2073 "y.tab.c" /* yacc.c:1646  */
    break;

  case 85:
#line 309 "bc.y" /* yacc.c:1646  */
    { bundle(2, "l", (yyvsp[0]) ); }
#line 2079 "y.tab.c" /* yacc.c:1646  */
    break;

  case 86:
#line 311 "bc.y" /* yacc.c:1646  */
    { bundle(3, (yyvsp[0]), "ds", (yyvsp[-2]) ); }
#line 2085 "y.tab.c" /* yacc.c:1646  */
    break;

  case 87:
#line 313 "bc.y" /* yacc.c:1646  */
    { bundle(6, "l", (yyvsp[-2]), (yyvsp[0]), (yyvsp[-1]), "ds", (yyvsp[-2]) ); }
#line 2091 "y.tab.c" /* yacc.c:1646  */
    break;

  case 88:
#line 315 "bc.y" /* yacc.c:1646  */
    { bundle(5,(yyvsp[0]),"d",(yyvsp[-3]),":",geta((yyvsp[-5]))); }
#line 2097 "y.tab.c" /* yacc.c:1646  */
    break;

  case 89:
#line 317 "bc.y" /* yacc.c:1646  */
    { bundle(9,(yyvsp[-3]),";",geta((yyvsp[-5])),(yyvsp[0]),(yyvsp[-1]),"d",(yyvsp[-3]),":",geta((yyvsp[-5]))); }
#line 2103 "y.tab.c" /* yacc.c:1646  */
    break;

  case 90:
#line 319 "bc.y" /* yacc.c:1646  */
    { bundle(2,(yyvsp[-1]),"Z"); }
#line 2109 "y.tab.c" /* yacc.c:1646  */
    break;

  case 91:
#line 321 "bc.y" /* yacc.c:1646  */
    { bundle(2,(yyvsp[-1]),"X"); }
#line 2115 "y.tab.c" /* yacc.c:1646  */
    break;

  case 92:
#line 323 "bc.y" /* yacc.c:1646  */
    { (yyval) = (yyvsp[-1]); }
#line 2121 "y.tab.c" /* yacc.c:1646  */
    break;

  case 93:
#line 325 "bc.y" /* yacc.c:1646  */
    { bundle(1, "?" ); }
#line 2127 "y.tab.c" /* yacc.c:1646  */
    break;

  case 94:
#line 327 "bc.y" /* yacc.c:1646  */
    { bundle(2, (yyvsp[-1]), "v" ); }
#line 2133 "y.tab.c" /* yacc.c:1646  */
    break;

  case 95:
#line 329 "bc.y" /* yacc.c:1646  */
    { bundle(2,"L",(yyvsp[0])); }
#line 2139 "y.tab.c" /* yacc.c:1646  */
    break;

  case 96:
#line 331 "bc.y" /* yacc.c:1646  */
    { bundle(2,(yyvsp[0]),"dk"); }
#line 2145 "y.tab.c" /* yacc.c:1646  */
    break;

  case 97:
#line 333 "bc.y" /* yacc.c:1646  */
    { bundle(4,"K",(yyvsp[0]),(yyvsp[-1]),"dk"); }
#line 2151 "y.tab.c" /* yacc.c:1646  */
    break;

  case 98:
#line 335 "bc.y" /* yacc.c:1646  */
    { bundle(2,(yyvsp[0]),"di"); }
#line 2157 "y.tab.c" /* yacc.c:1646  */
    break;

  case 99:
#line 337 "bc.y" /* yacc.c:1646  */
    { bundle(4,"I",(yyvsp[0]),(yyvsp[-1]),"di"); }
#line 2163 "y.tab.c" /* yacc.c:1646  */
    break;

  case 100:
#line 339 "bc.y" /* yacc.c:1646  */
    { bundle(2,(yyvsp[0]),"do"); }
#line 2169 "y.tab.c" /* yacc.c:1646  */
    break;

  case 101:
#line 341 "bc.y" /* yacc.c:1646  */
    { bundle(4,"O",(yyvsp[0]),(yyvsp[-1]),"do"); }
#line 2175 "y.tab.c" /* yacc.c:1646  */
    break;

  case 102:
#line 343 "bc.y" /* yacc.c:1646  */
    { bundle(1,"K"); }
#line 2181 "y.tab.c" /* yacc.c:1646  */
    break;

  case 103:
#line 345 "bc.y" /* yacc.c:1646  */
    { bundle(1,"I"); }
#line 2187 "y.tab.c" /* yacc.c:1646  */
    break;

  case 104:
#line 347 "bc.y" /* yacc.c:1646  */
    { bundle(1,"O"); }
#line 2193 "y.tab.c" /* yacc.c:1646  */
    break;

  case 106:
#line 352 "bc.y" /* yacc.c:1646  */
    { bundle(2, (yyvsp[-2]), (yyvsp[0]) ); }
#line 2199 "y.tab.c" /* yacc.c:1646  */
    break;

  case 108:
#line 356 "bc.y" /* yacc.c:1646  */
    {bundle(2,"l",geta((yyvsp[-2]))); }
#line 2205 "y.tab.c" /* yacc.c:1646  */
    break;

  case 109:
#line 360 "bc.y" /* yacc.c:1646  */
    { *cp++ = '\0'; }
#line 2211 "y.tab.c" /* yacc.c:1646  */
    break;

  case 110:
#line 364 "bc.y" /* yacc.c:1646  */
    { (yyval) = (intptr_t)cp; *cp++ = '_'; }
#line 2217 "y.tab.c" /* yacc.c:1646  */
    break;

  case 111:
#line 366 "bc.y" /* yacc.c:1646  */
    { (yyval) = (intptr_t)cp; *cp++ = (yyvsp[0]); }
#line 2223 "y.tab.c" /* yacc.c:1646  */
    break;

  case 112:
#line 368 "bc.y" /* yacc.c:1646  */
    { *cp++ = (yyvsp[0]); }
#line 2229 "y.tab.c" /* yacc.c:1646  */
    break;

  case 113:
#line 372 "bc.y" /* yacc.c:1646  */
    { (yyval) = (intptr_t)cp; *cp++ = crs++; *cp++ = '\0';
			if(crs == '[')crs+=3;
			if(crs == 'a')crs='{';
			if(crs >= 0241){yyerror("program too big");
				getout();
			}
			bstack[bindx++] = lev++; }
#line 2241 "y.tab.c" /* yacc.c:1646  */
    break;

  case 114:
#line 382 "bc.y" /* yacc.c:1646  */
    {	(yyval) = (intptr_t)getf((yyvsp[-1]));
			pre = (intptr_t *)"";
			post = (intptr_t *)"";
			lev = 1;
			bstack[bindx=0] = 0;
			}
#line 2252 "y.tab.c" /* yacc.c:1646  */
    break;

  case 116:
#line 392 "bc.y" /* yacc.c:1646  */
    { pp( (yyvsp[0]) ); }
#line 2258 "y.tab.c" /* yacc.c:1646  */
    break;

  case 117:
#line 394 "bc.y" /* yacc.c:1646  */
    { pp( (yyvsp[0]) ); }
#line 2264 "y.tab.c" /* yacc.c:1646  */
    break;

  case 118:
#line 398 "bc.y" /* yacc.c:1646  */
    { tp((yyvsp[0])); }
#line 2270 "y.tab.c" /* yacc.c:1646  */
    break;

  case 119:
#line 400 "bc.y" /* yacc.c:1646  */
    { tp((yyvsp[0])); }
#line 2276 "y.tab.c" /* yacc.c:1646  */
    break;

  case 121:
#line 404 "bc.y" /* yacc.c:1646  */
    { (yyval) = (intptr_t)geta((yyvsp[-2])); }
#line 2282 "y.tab.c" /* yacc.c:1646  */
    break;


#line 2286 "y.tab.c" /* yacc.c:1646  */
      default: break;
    }
  /* User semantic actions sometimes alter yychar, and that requires
     that yytoken be updated with the new translation.  We take the
     approach of translating immediately before every use of yytoken.
     One alternative is translating here after every semantic action,
     but that translation would be missed if the semantic action invokes
     YYABORT, YYACCEPT, or YYERROR immediately after altering yychar or
     if it invokes YYBACKUP.  In the case of YYABORT or YYACCEPT, an
     incorrect destructor might then be invoked immediately.  In the
     case of YYERROR or YYBACKUP, subsequent parser actions might lead
     to an incorrect destructor call or verbose syntax error message
     before the lookahead is translated.  */
  YY_SYMBOL_PRINT ("-> $$ =", yyr1[yyn], &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);

  *++yyvsp = yyval;

  /* Now 'shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTOKENS] + *yyssp;
  if (0 <= yystate && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTOKENS];

  goto yynewstate;


/*--------------------------------------.
| yyerrlab -- here on detecting error.  |
`--------------------------------------*/
yyerrlab:
  /* Make sure we have latest lookahead translation.  See comments at
     user semantic actions for why this is necessary.  */
  yytoken = yychar == YYEMPTY ? YYEMPTY : YYTRANSLATE (yychar);

  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
#if ! YYERROR_VERBOSE
      yyerror (YY_("syntax error"));
#else
# define YYSYNTAX_ERROR yysyntax_error (&yymsg_alloc, &yymsg, \
                                        yyssp, yytoken)
      {
        char const *yymsgp = YY_("syntax error");
        int yysyntax_error_status;
        yysyntax_error_status = YYSYNTAX_ERROR;
        if (yysyntax_error_status == 0)
          yymsgp = yymsg;
        else if (yysyntax_error_status == 1)
          {
            if (yymsg != yymsgbuf)
              YYSTACK_FREE (yymsg);
            yymsg = (char *) YYSTACK_ALLOC (yymsg_alloc);
            if (!yymsg)
              {
                yymsg = yymsgbuf;
                yymsg_alloc = sizeof yymsgbuf;
                yysyntax_error_status = 2;
              }
            else
              {
                yysyntax_error_status = YYSYNTAX_ERROR;
                yymsgp = yymsg;
              }
          }
        yyerror (yymsgp);
        if (yysyntax_error_status == 2)
          goto yyexhaustedlab;
      }
# undef YYSYNTAX_ERROR
#endif
    }



  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
         error, discard it.  */

      if (yychar <= YYEOF)
        {
          /* Return failure if at end of input.  */
          if (yychar == YYEOF)
            YYABORT;
        }
      else
        {
          yydestruct ("Error: discarding",
                      yytoken, &yylval);
          yychar = YYEMPTY;
        }
    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:

  /* Pacify compilers like GCC when the user code never invokes
     YYERROR and the label yyerrorlab therefore never appears in user
     code.  */
  if (/*CONSTCOND*/ 0)
     goto yyerrorlab;

  /* Do not reclaim the symbols of the rule whose action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;      /* Each real token shifted decrements this.  */

  for (;;)
    {
      yyn = yypact[yystate];
      if (!yypact_value_is_default (yyn))
        {
          yyn += YYTERROR;
          if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYTERROR)
            {
              yyn = yytable[yyn];
              if (0 < yyn)
                break;
            }
        }

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
        YYABORT;


      yydestruct ("Error: popping",
                  yystos[yystate], yyvsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", yystos[yyn], yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturn;

/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturn;

#if !defined yyoverflow || YYERROR_VERBOSE
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  /* Fall through.  */
#endif

yyreturn:
  if (yychar != YYEMPTY)
    {
      /* Make sure we have latest lookahead translation.  See comments at
         user semantic actions for why this is necessary.  */
      yytoken = YYTRANSLATE (yychar);
      yydestruct ("Cleanup: discarding lookahead",
                  yytoken, &yylval);
    }
  /* Do not reclaim the symbols of the rule whose action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
                  yystos[*yyssp], yyvsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
#if YYERROR_VERBOSE
  if (yymsg != yymsgbuf)
    YYSTACK_FREE (yymsg);
#endif
  return yyresult;
}
#line 407 "bc.y" /* yacc.c:1906  */

# define error 256

static int peekc = -1;
static int sargc;
static int ifile;
static char **sargv;

static char funtab[52] = {
	01,0,02,0,03,0,04,0,05,0,06,0,07,0,010,0,011,0,012,0,013,0,014,0,015,0,016,0,017,0,
	020,0,021,0,022,0,023,0,024,0,025,0,026,0,027,0,030,0,031,0,032,0 };
static char atab[52] = {
	0241,0,0242,0,0243,0,0244,0,0245,0,0246,0,0247,0,0250,0,0251,0,0252,0,0253,0,
	0254,0,0255,0,0256,0,0257,0,0260,0,0261,0,0262,0,0263,0,0264,0,0265,0,0266,0,
	0267,0,0270,0,0271,0,0272,0};
static char *letr[26] = {
  "a","b","c","d","e","f","g","h","i","j",
  "k","l","m","n","o","p","q","r","s","t",
  "u","v","w","x","y","z" } ;
/*static char *dot = { "." };*/

int
yylex(void){
	int c, ch;
restart:
	c = getch();
	peekc = -1;
	while( c == ' ' || c == '\t' ) c = getch();
	if(c == '\\'){
		getch();
		goto restart;
	}
	if( c<= 'z' && c >= 'a' ) {
		/* look ahead to look for reserved words */
		peekc = getch();
		if( peekc >= 'a' && peekc <= 'z' ){ /* must be reserved word */
			if( c=='i' && peekc=='f' ){ c=_IF; goto skip; }
			if( c=='w' && peekc=='h' ){ c=_WHILE; goto skip; }
			if( c=='f' && peekc=='o' ){ c=_FOR; goto skip; }
			if( c=='s' && peekc=='q' ){ c=SQRT; goto skip; }
			if( c=='r' && peekc=='e' ){ c=_RETURN; goto skip; }
			if( c=='b' && peekc=='r' ){ c=_BREAK; goto skip; }
			if( c=='d' && peekc=='e' ){ c=_DEFINE; goto skip; }
			if( c=='s' && peekc=='c' ){ c= SCALE; goto skip; }
			if( c=='b' && peekc=='a' ){ c=BASE; goto skip; }
			if( c=='i' && peekc == 'b'){ c=BASE; goto skip; }
			if( c=='o' && peekc=='b' ){ c=OBASE; goto skip; }
			if( c=='d' && peekc=='i' ){ c=FFF; goto skip; }
			if( c=='a' && peekc=='u' ){ c=_AUTO; goto skip; }
			if( c == 'l' && peekc=='e'){ c=LENGTH; goto skip; }
			if( c == 'q' && peekc == 'u'){getout();}
			/* could not be found */
			return( error );
		skip:	/* skip over rest of word */
			peekc = -1;
			while( (ch = getch()) >= 'a' && ch <= 'z' );
			peekc = ch;
			return( c );
		}

		/* usual case; just one single letter */

		yylval = (intptr_t)letr[c-'a'];
		return( LETTER );
	}
	if( c>= '0' && c <= '9' || c>= 'A' && c<= 'F' ){
		yylval = c;
		return( DIGIT );
	}
	switch( c ){
	case '.':	return( DOT );
	case '=':
		switch( peekc = getch() ){
		case '=': c=EQ; goto gotit;
		case '+': c=EQPL; goto gotit;
		case '-': c=EQMI; goto gotit;
		case '*': c=EQMUL; goto gotit;
		case '/': c=EQDIV; goto gotit;
		case '%': c=EQREM; goto gotit;
		case '^': c=EQEXP; goto gotit;
		default:   return( '=' );
			  gotit:     peekc = -1; return(c);
		  }
	case '+':	return( cpeek( '+', INCR, cpeek( '=', EQPL, '+') ) );
	case '-':	return( cpeek( '-', DECR, cpeek( '=', EQMI, '-') ) ) ;
	case '<':	return( cpeek( '=', LE, '<' ) );
	case '>':	return( cpeek( '=', GE, '>' ) );
	case '!':	return( cpeek( '=', NE, '!' ) );
	case '/':
		if((peekc = getch()) == '*'){
			peekc = -1;
			while((getch() != '*') || ((peekc = getch()) != '/'));
			peekc = -1;
			goto restart;
		}
		else if (peekc == '=') {
			c=EQDIV;
			goto gotit;
		}
		else return(c);
	case '*':
		return( cpeek( '=', EQMUL, '*' ) );
	case '%':
		return( cpeek( '=', EQREM, '%' ) );
	case '^':
		return( cpeek( '=', EQEXP, '^' ) );
	case '"':	
		 yylval = (intptr_t)str;
		 while((c=getch()) != '"'){*str++ = c;
			if(str >= &string[sizeof string - 1]){yyerror("string space exceeded");
			getout();
		}
	}
	 *str++ = '\0';
	return(QSTR);
	default:	 return( c );
	}
}

static int
cpeek(int c, int yes, int no){
	if( (peekc=getch()) != c ) return( no );
	else {
		peekc = -1;
		return( yes );
	}
}

static int
getch(void){
	int ch;
loop:
	ch = (peekc < 0) ? getc(in) : peekc;
	peekc = -1;
	if(ch != EOF)return(ch);
	if(++ifile > sargc){
		if(ifile >= sargc+2)getout();
		in = stdin;
		ln = 0;
		goto loop;
	}
	fclose(in);
	if((in = fopen(sargv[ifile],"r")) != NULL){
		ln = 0;
		ss = sargv[ifile];
		goto loop;
	}
	cantopen(sargv[ifile]);
	return EOF;
}
# define b_sp_max 3000
static intptr_t b_space [ b_sp_max ];
static intptr_t * b_sp_nxt = { b_space };

static int	bdebug = 0;

static intptr_t
bundle(int a, ...){
	intptr_t i, *q;
	va_list ap;

	i = a;
	q = b_sp_nxt;
	if( bdebug ) printf("bundle %ld elements at %lo\n",(long)i,  (long)q );
	va_start(ap, a);
	while(i-- > 0){
		if( b_sp_nxt >= & b_space[b_sp_max] ) yyerror( "bundling space exceeded" );
		* b_sp_nxt++ = va_arg(ap, intptr_t);
	}
	va_end(ap);
	* b_sp_nxt++ = 0;
	yyval = (intptr_t)q;
	return( (intptr_t)q );
}

static void
routput(intptr_t *p) {
	if( bdebug ) printf("routput(%lo)\n", (long)p );
	if( p >= &b_space[0] && p < &b_space[b_sp_max]){
		/* part of a bundle */
		while( *p != 0 ) routput( (intptr_t *)*p++ );
	}
	else printf( (char *)p );	 /* character string */
}

static void
output(intptr_t *p) {
	routput( p );
	b_sp_nxt = & b_space[0];
	printf( "\n" );
	fflush(stdout);
	cp = cary;
	crs = rcrs;
}

static void
conout(intptr_t p, intptr_t s) {
	printf("[");
	routput( (intptr_t *)p );
	printf("]s%s\n", (char *)s );
	fflush(stdout);
	lev--;
}

static void
yyerror(const char *s) {
	if(ifile > sargc)ss="teletype";
	fprintf(stderr, "%s on line %d, %s\n",
		s ,ss?ln+1:0,ss?ss:"command line");
	cp = cary;
	crs = rcrs;
	bindx = 0;
	lev = 0;
	b_sp_nxt = &b_space[0];
}

static void
cantopen(const char *fn)
{
	char	spc[280];
	char	*oss = ss;

	ss = 0;
	snprintf(spc, sizeof spc, "can't open input file %s", fn);
	yyerror(spc);
	ss = oss;
}

static void
pp(intptr_t s) {
	/* puts the relevant stuff on pre and post for the letter s */

	bundle(3, "S", s, pre );
	pre = (intptr_t *)yyval;
	bundle(4, post, "L", s, "s." );
	post = (intptr_t *)yyval;
}

static void
tp(intptr_t s) { /* same as pp, but for temps */
	bundle(3, "0S", s, pre );
	pre = (intptr_t *)yyval;
	bundle(4, post, "L", s, "s." );
	post = (intptr_t *)yyval;
}

static void
yyinit(int argc,char **argv) {
	signal(SIGINT, SIG_IGN);
	sargv=argv;
	sargc= -- argc;
	if(sargc == 0)in=stdin;
	else if((in = fopen(sargv[1],"r")) == NULL) {
		cantopen(sargv[1]);
		exit(0);
	}
	ifile = 1;
	ln = 0;
	ss = sargv[1];
}

static intptr_t *
getout(void){
	printf("q");
	fflush(stdout);
	exit(0);
	/*NOTREACHED*/
	return(NULL);
}

static intptr_t *
getf(intptr_t p) {
	return(intptr_t *)(&funtab[2*(*((char *)p) -0141)]);
}

static intptr_t *
geta(intptr_t p) {
	return(intptr_t *)(&atab[2*(*((char *)p) - 0141)]);
}

int
main(int argc, char **argv)
{
	extern int yyparse(void);
	const char optstring[] = "cdl";
	int p[2];
	int i;
	int cflag = 0, lflag = 0;


#ifdef	__GLIBC__
	putenv("POSIXLY_CORRECT=1");
#endif
	while ((i = getopt(argc, argv, optstring)) != EOF) {
		switch (i) {
		case 'd':
		case 'c':
			cflag = 1;
			break;
		case 'l':
			lflag = 1;
			break;
		default:
			exit(2);
		}
	}
	argv += optind - 1, argc -= optind - 1;
	if (cflag) {
		yyinit(argc, argv);
		yyparse();
		exit(0);
	}
	if (lflag) {
		*argv-- = LIBB;
		argc++;
	}
	pipe(p);
	if (fork()==0) {
		close(1);
		dup(p[1]);
		close(p[0]);
		close(p[1]);
		yyinit(argc, argv);
		yyparse();
		exit(0);
	}
	close(0);
	dup(p[0]);
	close(p[0]);
	close(p[1]);
	execl(DC, "dc", "-", NULL);
	execl("/usr/5bin/dc", "dc", "-", NULL);
	execl("/usr/local/bin/dc", "dc", "-", NULL);
	execl("/usr/contrib/bin/dc", "dc", "-", NULL);
	execl("/usr/bin/dc", "dc", "-", NULL);
	return(1);
}
