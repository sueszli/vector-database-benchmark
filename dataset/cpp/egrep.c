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
#line 40 "egrep.y" /* yacc.c:339  */

#if __GNUC__ >= 3 && __GNUC_MINOR__ >= 4 || __GNUC__ >= 4
#define	USED	__attribute__ ((used))
#elif defined __GNUC__
#define	USED	__attribute__ ((unused))
#else
#define	USED
#endif
static const char sccsid[] USED = "@(#)egrep.sl	2.22 (gritter) 5/29/05";
#line 66 "egrep.y" /* yacc.c:339  */

#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <regex.h>
#include <limits.h>
#include <stdlib.h>
#include "grep.h"
#include "alloc.h"

#include <mbtowi.h>

#define NCHARS 256
#define	NSTATES	64
#define FINAL -1
static unsigned	MAXLIN;
static unsigned	MAXPOS;
static unsigned	MAXCHARS;
static unsigned	char gotofn[NSTATES][NCHARS];
static int	lastn;
static int	state[NSTATES];
static char	out[NSTATES];
static int	line = 1;
static int	*name;
static int	*left;
static int	*right;
static int	*parent;
static int	*foll;
static int	*positions;
static wchar_t	*chars;
static int	nxtpos;
static int	initpos;
static int	nxtchar = 0;
static int	*tmpstat;
static int	*initstat;
static int	xstate;
static int	count;
static int	icount;

static void	yyerror(const char *);
static void	*erealloc(void *, size_t);
static void	more_chars(unsigned);
static void	more_lines(unsigned);
static void	more_positions(unsigned);
static int	yylex(void);
static void	synerror(void);
static int	enter(int);
static int	cclenter(int);
static int	menter(int);
static int	node(int, int, int);
static int	unary(int, int);
static void	cfoll(int);
static int	cgotofn(int, int);
static void	igotofn(void);
static int	cstate(int);
static int	member(int, int, int);
static int	notin(int);
static void	add(int *, int);
static void	follow(int);
static void	eg_build(void);
static int	eg_matchw(const char *, size_t);
static int	eg_match(const char *, size_t);
static int	eg_range(struct iblok *, char *);
static int	eg_rangew(struct iblok *, char *);

#line 144 "y.tab.c" /* yacc.c:339  */

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
    CHAR = 258,
    MCHAR = 259,
    DOT = 260,
    CCL = 261,
    NCCL = 262,
    OR = 263,
    CAT = 264,
    STAR = 265,
    PLUS = 266,
    QUEST = 267
  };
#endif
/* Tokens.  */
#define CHAR 258
#define MCHAR 259
#define DOT 260
#define CCL 261
#define NCCL 262
#define OR 263
#define CAT 264
#define STAR 265
#define PLUS 266
#define QUEST 267

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef int YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif


extern YYSTYPE yylval;

int yyparse (void);



/* Copy the second part of user declarations.  */

#line 216 "y.tab.c" /* yacc.c:358  */

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
#define YYFINAL  6
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   116

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  15
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  5
/* YYNRULES -- Number of rules.  */
#define YYNRULES  19
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  26

/* YYTRANSLATE[YYX] -- Symbol number corresponding to YYX as returned
   by yylex, with out-of-bounds checking.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   267

#define YYTRANSLATE(YYX)                                                \
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, without out-of-bounds checking.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
      13,    14,     2,     2,     2,     2,     2,     2,     2,     2,
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
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12
};

#if YYDEBUG
  /* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_uint8 yyrline[] =
{
       0,   135,   135,   140,   142,   144,   146,   150,   153,   155,
     157,   159,   161,   165,   167,   169,   171,   173,   175,   177
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || 0
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "CHAR", "MCHAR", "DOT", "CCL", "NCCL",
  "OR", "CAT", "STAR", "PLUS", "QUEST", "'('", "')'", "$accept", "s", "t",
  "b", "r", YY_NULLPTR
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[NUM] -- (External) token number corresponding to the
   (internal) symbol number NUM (which must be that of a token).  */
static const yytype_uint16 yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,    40,    41
};
# endif

#define YYPACT_NINF -6

#define yypact_value_is_default(Yystate) \
  (!!((Yystate) == (-6)))

#define YYTABLE_NINF -15

#define yytable_value_is_error(Yytable_value) \
  0

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const yytype_int8 yypact[] =
{
      -5,    -6,    10,    -6,     1,     1,    -6,    -6,    -6,    -6,
      -6,    -6,    -6,     1,    50,    64,    77,    92,    -6,    -6,
      -6,    20,   103,     1,    -6,    35
};

  /* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
     Performed when YYTABLE does not specify something else to do.  Zero
     means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
       7,     7,     0,     2,     0,     0,     1,    19,     8,     9,
      10,    11,    12,     0,     0,     0,     0,     0,    15,    16,
      17,     0,     0,     0,    18,     0
};

  /* YYPGOTO[NTERM-NUM].  */
static const yytype_int8 yypgoto[] =
{
      -6,    -6,    -6,    11,    -4
};

  /* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int8 yydefgoto[] =
{
      -1,     2,     3,     4,    21
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_int8 yytable[] =
{
      14,    15,     7,     1,     8,     9,    10,    11,    12,    16,
       6,     0,     5,    25,    13,     0,     0,     0,    25,    25,
     -14,     7,     0,   -14,   -14,   -14,   -14,   -14,   -14,     0,
      18,    19,    20,   -14,   -14,   -13,     7,     0,     8,     9,
      10,    11,    12,   -13,     0,    18,    19,    20,    13,   -13,
      -3,     7,     0,     8,     9,    10,    11,    12,    17,     0,
      18,    19,    20,    13,    -5,     7,     0,     8,     9,    10,
      11,    12,    22,     0,    18,    19,    20,    13,     7,     0,
       8,     9,    10,    11,    12,    23,     0,    18,    19,    20,
      13,    24,    -6,     7,     0,     8,     9,    10,    11,    12,
       0,     0,     0,    -4,     7,    13,     8,     9,    10,    11,
      12,     0,     0,     0,     0,     0,    13
};

static const yytype_int8 yycheck[] =
{
       4,     5,     1,     8,     3,     4,     5,     6,     7,    13,
       0,    -1,     1,    17,    13,    -1,    -1,    -1,    22,    23,
       0,     1,    -1,     3,     4,     5,     6,     7,     8,    -1,
      10,    11,    12,    13,    14,     0,     1,    -1,     3,     4,
       5,     6,     7,     8,    -1,    10,    11,    12,    13,    14,
       0,     1,    -1,     3,     4,     5,     6,     7,     8,    -1,
      10,    11,    12,    13,     0,     1,    -1,     3,     4,     5,
       6,     7,     8,    -1,    10,    11,    12,    13,     1,    -1,
       3,     4,     5,     6,     7,     8,    -1,    10,    11,    12,
      13,    14,     0,     1,    -1,     3,     4,     5,     6,     7,
      -1,    -1,    -1,     0,     1,    13,     3,     4,     5,     6,
       7,    -1,    -1,    -1,    -1,    -1,    13
};

  /* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
     symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,     8,    16,    17,    18,    18,     0,     1,     3,     4,
       5,     6,     7,    13,    19,    19,    19,     8,    10,    11,
      12,    19,     8,     8,    14,    19
};

  /* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    15,    16,    17,    17,    17,    17,    18,    19,    19,
      19,    19,    19,    19,    19,    19,    19,    19,    19,    19
};

  /* YYR2[YYN] -- Number of symbols on the right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     1,     2,     4,     3,     3,     0,     1,     1,
       1,     1,     1,     3,     2,     2,     2,     2,     3,     1
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
  YYSTYPE yyval;

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
        case 2:
#line 136 "egrep.y" /* yacc.c:1646  */
    { unary(FINAL, (yyvsp[0]));
		  line--;
		}
#line 1317 "y.tab.c" /* yacc.c:1646  */
    break;

  case 3:
#line 141 "egrep.y" /* yacc.c:1646  */
    { (yyval) = node(CAT, (yyvsp[-1]), (yyvsp[0])); }
#line 1323 "y.tab.c" /* yacc.c:1646  */
    break;

  case 4:
#line 143 "egrep.y" /* yacc.c:1646  */
    { (yyval) = node(CAT, (yyvsp[-2]), (yyvsp[-1])); }
#line 1329 "y.tab.c" /* yacc.c:1646  */
    break;

  case 5:
#line 145 "egrep.y" /* yacc.c:1646  */
    { (yyval) = node(CAT, (yyvsp[-1]), (yyvsp[0])); }
#line 1335 "y.tab.c" /* yacc.c:1646  */
    break;

  case 6:
#line 147 "egrep.y" /* yacc.c:1646  */
    { (yyval) = node(CAT, (yyvsp[-2]), (yyvsp[-1])); }
#line 1341 "y.tab.c" /* yacc.c:1646  */
    break;

  case 7:
#line 150 "egrep.y" /* yacc.c:1646  */
    { (yyval) = enter(DOT);
		   (yyval) = unary(STAR, (yyval)); }
#line 1348 "y.tab.c" /* yacc.c:1646  */
    break;

  case 8:
#line 154 "egrep.y" /* yacc.c:1646  */
    { (yyval) = enter((yyvsp[0])); }
#line 1354 "y.tab.c" /* yacc.c:1646  */
    break;

  case 9:
#line 156 "egrep.y" /* yacc.c:1646  */
    { (yyval) = menter((yyvsp[0])); }
#line 1360 "y.tab.c" /* yacc.c:1646  */
    break;

  case 10:
#line 158 "egrep.y" /* yacc.c:1646  */
    { (yyval) = enter(DOT); }
#line 1366 "y.tab.c" /* yacc.c:1646  */
    break;

  case 11:
#line 160 "egrep.y" /* yacc.c:1646  */
    { (yyval) = cclenter(CCL); }
#line 1372 "y.tab.c" /* yacc.c:1646  */
    break;

  case 12:
#line 162 "egrep.y" /* yacc.c:1646  */
    { (yyval) = cclenter(NCCL); }
#line 1378 "y.tab.c" /* yacc.c:1646  */
    break;

  case 13:
#line 166 "egrep.y" /* yacc.c:1646  */
    { (yyval) = node(OR, (yyvsp[-2]), (yyvsp[0])); }
#line 1384 "y.tab.c" /* yacc.c:1646  */
    break;

  case 14:
#line 168 "egrep.y" /* yacc.c:1646  */
    { (yyval) = node(CAT, (yyvsp[-1]), (yyvsp[0])); }
#line 1390 "y.tab.c" /* yacc.c:1646  */
    break;

  case 15:
#line 170 "egrep.y" /* yacc.c:1646  */
    { (yyval) = unary(STAR, (yyvsp[-1])); }
#line 1396 "y.tab.c" /* yacc.c:1646  */
    break;

  case 16:
#line 172 "egrep.y" /* yacc.c:1646  */
    { (yyval) = unary(PLUS, (yyvsp[-1])); }
#line 1402 "y.tab.c" /* yacc.c:1646  */
    break;

  case 17:
#line 174 "egrep.y" /* yacc.c:1646  */
    { (yyval) = unary(QUEST, (yyvsp[-1])); }
#line 1408 "y.tab.c" /* yacc.c:1646  */
    break;

  case 18:
#line 176 "egrep.y" /* yacc.c:1646  */
    { (yyval) = (yyvsp[-1]); }
#line 1414 "y.tab.c" /* yacc.c:1646  */
    break;


#line 1418 "y.tab.c" /* yacc.c:1646  */
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
#line 180 "egrep.y" /* yacc.c:1906  */

static void
yyerror(const char *s) {
	fprintf(stderr, "%s: %s\n", progname, s);
	exit(2);
}

static void *
erealloc(void *vp, size_t size)
{
	if ((vp = realloc(vp, size)) == NULL) {
		write(2, progname, strlen(progname));
		write(2, ": regular expression too long\n", 30);
		_exit(2);
	}
	return (vp);
}

static void
more_chars(unsigned need)
{
	unsigned omaxchars = MAXCHARS;
	unsigned incr;
	incr = need - MAXCHARS + 32;
	MAXCHARS += incr;
	chars = erealloc(chars, MAXCHARS * sizeof *chars);
	memset(&chars[omaxchars], 0, incr * sizeof *chars);
}

static void
more_lines(unsigned need)
{
	unsigned omaxlin = MAXLIN;
	unsigned incr;
	incr = need - MAXLIN + 32;
	MAXLIN += incr;
	name = erealloc(name, MAXLIN * sizeof *name);
	memset(&name[omaxlin], 0, incr * sizeof *name);
	left = erealloc(left, MAXLIN * sizeof *left);
	memset(&left[omaxlin], 0, incr * sizeof *left);
	right = erealloc(right, MAXLIN * sizeof *right);
	memset(&right[omaxlin], 0, incr * sizeof *right);
	parent = erealloc(parent, MAXLIN * sizeof *parent);
	memset(&parent[omaxlin], 0, incr * sizeof *parent);
	foll = erealloc(foll, MAXLIN * sizeof *foll);
	memset(&foll[omaxlin], 0, incr * sizeof *foll);
	tmpstat = erealloc(tmpstat, MAXLIN * sizeof *tmpstat);
	memset(&tmpstat[omaxlin], 0, incr * sizeof *tmpstat);
	initstat = erealloc(initstat, MAXLIN * sizeof *initstat);
	memset(&initstat[omaxlin], 0, incr * sizeof *initstat);
}

static void
more_positions(unsigned need)
{
	unsigned omaxpos = MAXPOS;
	unsigned incr;
	incr = need - MAXPOS + 48;
	MAXPOS += incr;
	positions = erealloc(positions, MAXPOS * sizeof *positions);
	memset(&positions[omaxpos], 0, incr * sizeof *positions);
}

static int
yylex(void) {
	int cclcnt, x;
	register int c;
	switch(c = nextch()) {
		case '$':
		case '^': c = '\n';
			goto defchar;
		case '|': return (OR);
		case '*': return (STAR);
		case '+': return (PLUS);
		case '?': return (QUEST);
		case '(': return (c);
		case ')': return (c);
		case '.': return (DOT);
		case EOF: return (0);
		case '\0': return (0);
		case '\n': return (OR);
		case '[': 
			x = CCL;
			cclcnt = 0;
			count = nxtchar++;
			if ((c = nextch()) == '^') {
				x = NCCL;
				c = nextch();
			}
			do {
				if (c == '\0' || c == EOF) synerror();
				if (nxtchar >= MAXCHARS) more_chars(nxtchar);
				chars[nxtchar++] = c;
				cclcnt++;
			} while ((c = nextch()) != ']');
			chars[count] = cclcnt;
			return (x);
		case '\\':
			if ((c = nextch()) == EOF) synerror();
		defchar:
		default:
			if (mbcode && c & ~(wchar_t)0177) {
				if (nxtchar >= MAXCHARS) more_chars(nxtchar);
				chars[nxtchar] = c;
				yylval = nxtchar++;
				return (MCHAR);
			}
			yylval = c; return (CHAR);
	}
}

static void
synerror(void) {
	yyerror("syntax error");
}

static int
enter(int x) {
	if(line >= MAXLIN) more_lines(line);
	name[line] = x;
	left[line] = 0;
	right[line] = 0;
	return(line++);
}

static int
cclenter(int x) {
	register int linno;
	linno = enter(x);
	right[linno] = count;
	return (linno);
}

static int
menter(int x) {
	register int linno;
	linno = enter(MCHAR);
	right[linno] = x;
	return (linno);
}

static int
node(int x, int l, int r) {
	if(line >= MAXLIN) more_lines(line);
	name[line] = x;
	left[line] = l;
	right[line] = r;
	parent[l] = line;
	parent[r] = line;
	return(line++);
}

static int
unary(int x, int d) {
	if(line >= MAXLIN) more_lines(line);
	name[line] = x;
	left[line] = d;
	right[line] = 0;
	parent[d] = line;
	return(line++);
}

static void
cfoll(int v) {
	register int i;
	if (left[v] == 0) {
		count = 0;
		for (i=1; i<=line; i++) tmpstat[i] = 0;
		follow(v);
		add(foll, v);
	}
	else if (right[v] == 0) cfoll(left[v]);
	else {
		cfoll(left[v]);
		cfoll(right[v]);
	}
}

static int
cgotofn(int s, int c)
{
	register int cc, i, k;
	int n, pos;
	int st;
	int curpos, num;
	int number, newpos;
	n = lastn;
	cc = iflag ? mbcode && c & ~(wchar_t)0177 ? towlower(c):tolower(c) : c;
	num = positions[state[s]];
	count = icount;
	for (i=3; i <= line; i++) tmpstat[i] = initstat[i];
	pos = state[s] + 1;
	for (i=0; i<num; i++) {
		curpos = positions[pos];
		if ((k = name[curpos]) >= 0)
			if (
				((cc & ~(wchar_t)(NCHARS-1)) == 0 && k == cc)
				|| (k == MCHAR && cc == chars[right[curpos]])
				|| (cc != '\n' && cc != WEOF && ((k == DOT)
				|| (k == CCL && member(cc, right[curpos], 1))
				|| (k == NCCL && member(cc, right[curpos], 0))))
			) {
				if (foll[curpos] < MAXPOS)
					number = positions[foll[curpos]];
				else
					number = 0;
				newpos = foll[curpos] + 1;
				for (k=0; k<number; k++) {
					if (tmpstat[positions[newpos]] != 1) {
						tmpstat[positions[newpos]] = 1;
						count++;
					}
					newpos++;
				}
			}
		pos++;
	}
	if (notin(n)) {
		if (++n >= NSTATES) {
			n = gotofn[0]['\n'];
			memset(gotofn, 0, sizeof gotofn);
			gotofn[0]['\n'] = n;
			memset(&out[n], 0, sizeof out - n);
			nxtpos = initpos;
			st = n + 1;
		}
		else {
			st = n + 1;
			if ((c & ~(wchar_t)(NCHARS-1)) == 0)
				gotofn[s][c] = st;
		}
		add(state, n);
		if (tmpstat[line] == 1) out[n] = 1;
	}
	else {
		st = xstate + 1;
		if ((c & ~(wchar_t)(NCHARS-1)) == 0)
			gotofn[s][c] = st;
	}
	lastn = n;
	return (st);
}

static void
igotofn(void)
{
	int n;
	count = 0;
	for (n=3; n<=line; n++) tmpstat[n] = 0;
	if (cstate(line-1)==0) {
		tmpstat[line] = 1;
		count++;
		out[0] = 1;
	}
	for (n=3; n<=line; n++) initstat[n] = tmpstat[n];
	count--;		/*leave out position 1 */
	icount = count;
	tmpstat[1] = 0;
	add(state, 0);
	cgotofn(0, '\n');
	initpos = nxtpos;
}

static int
cstate(int v) {
	register int b;
	if (left[v] == 0) {
		if (tmpstat[v] != 1) {
			tmpstat[v] = 1;
			count++;
		}
		return(1);
	}
	else if (right[v] == 0) {
		if (cstate(left[v]) == 0) return (0);
		else if (name[v] == PLUS) return (1);
		else return (0);
	}
	else if (name[v] == CAT) {
		if (cstate(left[v]) == 0 && cstate(right[v]) == 0) return (0);
		else return (1);
	}
	else { /* name[v] == OR */
		b = cstate(right[v]);
		if (cstate(left[v]) == 0 || b == 0) return (0);
		else return (1);
	}
}

static int
member(int symb, int set, int torf) {
	register int num, pos, lastc = WEOF;
	num = chars[set];
	pos = set + 1;
	num += pos;
	while (pos < num) {
		if (chars[pos] == '-' && lastc != WEOF) {
			if (++pos == num)	/* System V oddity: '[a-]' */
				return (!torf);	/* matches 'a' but not '-' */
			if (symb > lastc && symb < chars[pos])
				return (torf);
		}
		if (symb == chars[pos]) return (torf);
		lastc = chars[pos++];
	}
	return (!torf);
}

static int
notin(int n) {
	register int i, j, pos;
	for (i=0; i<=n; i++) {
		if (positions[state[i]] == count) {
			pos = state[i] + 1;
			for (j=0; j < count; j++)
				if (tmpstat[positions[pos++]] != 1) goto nxt;
			xstate = i;
			return (0);
		}
		nxt: ;
	}
	return (1);
}

static void
add(int *array, int n) {
	register int i;
	unsigned need;
	if ((need = nxtpos + count) >= MAXPOS) more_positions(need);
	array[n] = nxtpos;
	positions[nxtpos++] = count;
	for (i=3; i <= line; i++) {
		if (tmpstat[i] == 1) {
			if (nxtpos >= MAXPOS) more_positions(nxtpos);
			positions[nxtpos++] = i;
		}
	}
}

static void
follow(int v) {
	int p;
	if (v == line) return;
	p = parent[v];
	switch(name[p]) {
		case STAR:
		case PLUS:	cstate(v);
				follow(p);
				return;

		case OR:
		case QUEST:	follow(p);
				return;

		case CAT:	if (v == left[p]) {
					if (cstate(right[p]) == 0) {
						follow(p);
						return;
					}
				}
				else follow(p);
				return;
		case FINAL:	if (tmpstat[line] != 1) {
					tmpstat[line] = 1;
					count++;
				}
				return;
	}
}

static void
eg_build(void)
{
	extern int	yyparse(void);

	more_chars(0);
	more_lines(0);
	more_positions(0);
	yyparse();
	cfoll(line-1);
	igotofn();
	range = mbcode ? eg_rangew : eg_range;
}

static int
eg_matchw(const char *str, size_t sz)
{
	register const char *p;
	register int cstat, nstat;
	wint_t wc;
	int n;

	p = str;
	cstat = gotofn[0]['\n']-1;
	while (out[cstat] == 0 && p < &str[sz]) {
		if (*p & 0200) {
			if ((n = mbtowi(&wc, p, &str[sz] - p)) < 0) {
				p++;
				wc = WEOF;
			}
			else p += n;
		}
		else wc = *p++;
		if ((wc & ~(wchar_t)(NCHARS-1)) != 0 ||
				(nstat = gotofn[cstat][wc]) == 0)
			nstat = cgotofn(cstat, wc);
		cstat = nstat-1;
	}
	if (out[cstat] == 0 && p == &str[sz])
		cstat = gotofn[cstat]['\n']-1;
	return (out[cstat]);
}

static int
eg_match(const char *str, size_t sz)
{
	register const char *p;
	register int cstat, nstat;

	p = str;
	cstat = gotofn[0]['\n']-1;
	while (out[cstat] == 0 && p < &str[sz]) {
		if ((nstat = gotofn[cstat][*p++ & 0377]) == 0)
			nstat = cgotofn(cstat, p[-1] & 0377);
		cstat = nstat-1;
	}
	if (out[cstat] == 0 && p == &str[sz])
		cstat = gotofn[cstat]['\n']-1;
	return (out[cstat]);
}

static int
eg_range(struct iblok *ip, char *last)
{
	register char *p;
	register int cstat, nstat;
	int istat;

	p = ip->ib_cur;
	lineno++;
	istat = cstat = gotofn[0]['\n']-1;
	if (out[cstat]) goto found;
	for (;;) {
		if ((nstat = gotofn[cstat][*p & 0377]) == 0)
			nstat = cgotofn(cstat, *p & 0377);
		cstat = nstat-1;
		if (out[cstat]) {
		found:	for (;;) {
				if (vflag == 0) {
		succeed:		outline(ip, last, p - ip->ib_cur);
					if (qflag || lflag)
						return (1);
				}
				else {
					ip->ib_cur = p;
					while (*ip->ib_cur++ != '\n');
				}
				if ((p = ip->ib_cur) > last)
					return (0);
				lineno++;
				if ((out[(cstat=istat)]) == 0) goto brk2;
			}
		}
		if (*p++ == '\n') {
			if (vflag) {
				p--;
				goto succeed;
			}
			if ((ip->ib_cur = p) > last)
				return (0);
			lineno++;
			if (out[(cstat=istat)]) goto found;
		}
		brk2:	;
	}
}

static int
eg_rangew(struct iblok *ip, char *last)
{
	register char *p;
	register int cstat, nstat;
	int istat;
	wint_t	wc;
	int	n;

	p = ip->ib_cur;
	lineno++;
	istat = cstat = gotofn[0]['\n']-1;
	if (out[cstat]) goto found;
	for (;;) {
		if (*p & 0200) {
			if ((n = mbtowi(&wc, p, last + 1 - p)) < 0) {
				n = 1;
				wc = WEOF;
			}
		}
		else {
			wc = *p;
			n = 1;
		}
		if ((wc & ~(wchar_t)(NCHARS-1)) != 0 ||
			(nstat = gotofn[cstat][wc]) == 0)
			nstat = cgotofn(cstat, wc);
		cstat = nstat-1;
		if (out[cstat]) {
		found:	for (;;) {
				if (vflag == 0) {
		succeed:		outline(ip, last, p - ip->ib_cur);
					if (qflag || lflag)
						return (1);
				}
				else {
					ip->ib_cur = p;
					while (*ip->ib_cur++ != '\n');
				}
				if ((p = ip->ib_cur) > last)
					return (0);
				lineno++;
				if ((out[(cstat=istat)]) == 0) goto brk2;
			}
		}
		p += n;
		if (wc == '\n') {
			if (vflag) {
				p--;
				goto succeed;
			}
			if ((ip->ib_cur = p) > last)
				return (0);
			lineno++;
			if (out[(cstat=istat)]) goto found;
		}
		brk2:	;
	}
}

void
misop(void)
{
	synerror();
}

void
init(void)
{
	Eflag = 1;
	eg_select();
	options = "bce:f:hilnrRvyz";
}

void
eg_select(void)
{
	build = eg_build;
	match = mbcode ? eg_matchw : eg_match;
	matchflags &= ~(MF_NULTERM|MF_LOCONV);
}

/*
 * Some dummy routines for sharing source with other grep flavors.
 */
void
ac_select(void)
{
}

void
st_select(void)
{
}

void
rc_select(void)
{
}

char *usagemsg =
"usage: %s [ -bchilnv ] [ -e exp ] [ -f file ] [ strings ] [ file ] ...\n";
char *stdinmsg;
