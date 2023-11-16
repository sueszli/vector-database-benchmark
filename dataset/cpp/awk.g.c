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
#line 74 "awk.g.y" /* yacc.c:339  */

#include <inttypes.h>
typedef	void	*YYSTYPE;
#define	YYSTYPE	YYSTYPE
#line 114 "awk.g.y" /* yacc.c:339  */

/*	from 4.4BSD /usr/src/old/awk/awk.g.y	4.4 (Berkeley) 4/27/91	*/
/*	Sccsid @(#)awk.g.y	1.5 (gritter) 7/24/03>	*/

#include "awk.def"
#ifndef	DEBUG	
#	define	PUTS(x)
#endif

#line 81 "y.tab.c" /* yacc.c:339  */

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

/* In a future release of Bison, this section will be replaced
   by #include "y.tab.h".  */
#ifndef YY_YY_Y_TAB_H_INCLUDED
# define YY_YY_Y_TAB_H_INCLUDED
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
    FIRSTTOKEN = 258,
    FINAL = 259,
    FATAL = 260,
    LT = 261,
    LE = 262,
    GT = 263,
    GE = 264,
    EQ = 265,
    NE = 266,
    MATCH = 267,
    NOTMATCH = 268,
    APPEND = 269,
    ADD = 270,
    MINUS = 271,
    MULT = 272,
    DIVIDE = 273,
    MOD = 274,
    UMINUS = 275,
    ASSIGN = 276,
    ADDEQ = 277,
    SUBEQ = 278,
    MULTEQ = 279,
    DIVEQ = 280,
    MODEQ = 281,
    JUMP = 282,
    XBEGIN = 283,
    XEND = 284,
    NL = 285,
    PRINT = 286,
    PRINTF = 287,
    SPRINTF = 288,
    SPLIT = 289,
    IF = 290,
    ELSE = 291,
    WHILE = 292,
    FOR = 293,
    IN = 294,
    NEXT = 295,
    EXIT = 296,
    BREAK = 297,
    CONTINUE = 298,
    PROGRAM = 299,
    PASTAT = 300,
    PASTAT2 = 301,
    REGEXPR = 302,
    ASGNOP = 303,
    BOR = 304,
    AND = 305,
    NOT = 306,
    NUMBER = 307,
    VAR = 308,
    ARRAY = 309,
    FNCN = 310,
    SUBSTR = 311,
    LSUBSTR = 312,
    INDEX = 313,
    GETLINE = 314,
    RELOP = 315,
    MATCHOP = 316,
    OR = 317,
    STRING = 318,
    DOT = 319,
    CCL = 320,
    NCCL = 321,
    CHAR = 322,
    CAT = 323,
    STAR = 324,
    PLUS = 325,
    QUEST = 326,
    POSTINCR = 327,
    PREINCR = 328,
    POSTDECR = 329,
    PREDECR = 330,
    INCR = 331,
    DECR = 332,
    FIELD = 333,
    INDIRECT = 334,
    LASTTOKEN = 335
  };
#endif
/* Tokens.  */
#define FIRSTTOKEN 258
#define FINAL 259
#define FATAL 260
#define LT 261
#define LE 262
#define GT 263
#define GE 264
#define EQ 265
#define NE 266
#define MATCH 267
#define NOTMATCH 268
#define APPEND 269
#define ADD 270
#define MINUS 271
#define MULT 272
#define DIVIDE 273
#define MOD 274
#define UMINUS 275
#define ASSIGN 276
#define ADDEQ 277
#define SUBEQ 278
#define MULTEQ 279
#define DIVEQ 280
#define MODEQ 281
#define JUMP 282
#define XBEGIN 283
#define XEND 284
#define NL 285
#define PRINT 286
#define PRINTF 287
#define SPRINTF 288
#define SPLIT 289
#define IF 290
#define ELSE 291
#define WHILE 292
#define FOR 293
#define IN 294
#define NEXT 295
#define EXIT 296
#define BREAK 297
#define CONTINUE 298
#define PROGRAM 299
#define PASTAT 300
#define PASTAT2 301
#define REGEXPR 302
#define ASGNOP 303
#define BOR 304
#define AND 305
#define NOT 306
#define NUMBER 307
#define VAR 308
#define ARRAY 309
#define FNCN 310
#define SUBSTR 311
#define LSUBSTR 312
#define INDEX 313
#define GETLINE 314
#define RELOP 315
#define MATCHOP 316
#define OR 317
#define STRING 318
#define DOT 319
#define CCL 320
#define NCCL 321
#define CHAR 322
#define CAT 323
#define STAR 324
#define PLUS 325
#define QUEST 326
#define POSTINCR 327
#define PREINCR 328
#define POSTDECR 329
#define PREDECR 330
#define INCR 331
#define DECR 332
#define FIELD 333
#define INDIRECT 334
#define LASTTOKEN 335

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef int YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif


extern YYSTYPE yylval;

int yyparse (void);

#endif /* !YY_YY_Y_TAB_H_INCLUDED  */

/* Copy the second part of user declarations.  */

#line 292 "y.tab.c" /* yacc.c:358  */

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
#define YYLAST   1721

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  97
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  30
/* YYNRULES -- Number of rules.  */
#define YYNRULES  110
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  228

/* YYTRANSLATE[YYX] -- Symbol number corresponding to YYX as returned
   by yylex, with out-of-bounds checking.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   335

#define YYTRANSLATE(YYX)                                                \
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, without out-of-bounds checking.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,    70,    76,     2,     2,
      68,    91,    74,    72,    94,    73,     2,    75,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,    96,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    92,     2,    93,    69,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    89,    95,    90,     2,     2,     2,     2,
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
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    71,    77,    78,    79,    80,    81,    82,
      83,    84,    85,    86,    87,    88
};

#if YYDEBUG
  /* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   126,   126,   127,   131,   132,   133,   137,   138,   139,
     143,   144,   145,   146,   150,   151,   152,   153,   157,   158,
     159,   160,   164,   168,   169,   173,   177,   178,   182,   183,
     184,   185,   186,   189,   190,   191,   194,   197,   198,   199,
     201,   203,   205,   207,   209,   210,   211,   212,   213,   214,
     215,   216,   217,   218,   219,   220,   224,   225,   226,   230,
     231,   235,   236,   237,   238,   240,   244,   245,   246,   250,
     253,   254,   255,   259,   260,   261,   265,   266,   267,   271,
     272,   276,   276,   282,   284,   289,   290,   294,   296,   298,
     300,   302,   303,   304,   308,   309,   310,   312,   313,   314,
     315,   316,   317,   318,   319,   323,   324,   328,   332,   334,
     336
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || 0
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "FIRSTTOKEN", "FINAL", "FATAL", "LT",
  "LE", "GT", "GE", "EQ", "NE", "MATCH", "NOTMATCH", "APPEND", "ADD",
  "MINUS", "MULT", "DIVIDE", "MOD", "UMINUS", "ASSIGN", "ADDEQ", "SUBEQ",
  "MULTEQ", "DIVEQ", "MODEQ", "JUMP", "XBEGIN", "XEND", "NL", "PRINT",
  "PRINTF", "SPRINTF", "SPLIT", "IF", "ELSE", "WHILE", "FOR", "IN", "NEXT",
  "EXIT", "BREAK", "CONTINUE", "PROGRAM", "PASTAT", "PASTAT2", "REGEXPR",
  "ASGNOP", "BOR", "AND", "NOT", "NUMBER", "VAR", "ARRAY", "FNCN",
  "SUBSTR", "LSUBSTR", "INDEX", "GETLINE", "RELOP", "MATCHOP", "OR",
  "STRING", "DOT", "CCL", "NCCL", "CHAR", "'('", "'^'", "'$'", "CAT",
  "'+'", "'-'", "'*'", "'/'", "'%'", "STAR", "PLUS", "QUEST", "POSTINCR",
  "PREINCR", "POSTDECR", "PREDECR", "INCR", "DECR", "FIELD", "INDIRECT",
  "LASTTOKEN", "'{'", "'}'", "')'", "'['", "']'", "','", "'|'", "';'",
  "$accept", "program", "begin", "end", "compound_conditional",
  "compound_pattern", "conditional", "else", "field", "if", "lex_expr",
  "var", "term", "expr", "optNL", "pa_stat", "pa_stats", "pattern",
  "print_list", "pe_list", "redir", "regular_expr", "$@1", "rel_expr",
  "st", "simple_stat", "statement", "stat_list", "while", "for", YY_NULLPTR
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[NUM] -- (External) token number corresponding to the
   (internal) symbol number NUM (which must be that of a token).  */
static const yytype_uint16 yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   294,
     295,   296,   297,   298,   299,   300,   301,   302,   303,   304,
     305,   306,   307,   308,   309,   310,   311,   312,   313,   314,
     315,   316,   317,   318,   319,   320,   321,   322,    40,    94,
      36,   323,    43,    45,    42,    47,    37,   324,   325,   326,
     327,   328,   329,   330,   331,   332,   333,   334,   335,   123,
     125,    41,    91,    93,    44,   124,    59
};
# endif

#define YYPACT_NINF -105

#define yypact_value_is_default(Yystate) \
  (!!((Yystate) == (-105)))

#define YYTABLE_NINF -93

#define yytable_value_is_error(Yytable_value) \
  0

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const yytype_int16 yypact[] =
{
     807,  -105,   -83,    14,    -7,  -105,  -105,  -105,  1151,   270,
     -61,  1598,   -37,  1489,  -105,   -48,     7,    16,    32,  -105,
    -105,  1489,  1634,  1634,  -105,    52,    52,  -105,  1634,  -105,
      34,  -105,  -105,  -105,    40,    54,  1525,   -22,   -11,  -105,
    -105,  -105,  1598,  1598,    35,    41,    48,   -22,   851,   -22,
     -22,  1634,  -105,  -105,   680,  1634,   -22,  -105,   680,  -105,
    -105,  1598,  1003,  -105,    18,  1634,  -105,  1634,  1267,  1634,
    1634,   -23,    11,  1230,     0,    27,     5,  -105,  -105,    76,
    -105,  -105,  -105,   340,  -105,  1634,  -105,  -105,  1634,  1634,
    1634,  1634,  1634,  1634,    56,    54,  -105,  -105,  -105,  1489,
    1489,  -105,  1489,   -53,   -53,  1562,  1562,   725,  -105,   851,
    -105,  -105,  -105,  1304,   408,   101,  -105,  -105,   476,   929,
     -69,  1634,  1634,  1040,  1194,  -105,  1341,  1077,  1114,  -105,
    -105,  -105,  -105,    65,  -105,  1634,    71,    71,  -105,  -105,
    -105,   162,  -105,   104,  -105,   544,   -13,  -105,  -105,  1634,
    1634,  1562,  1562,  -105,    -4,  -105,  1525,  -105,     2,   -27,
      55,  -105,  -105,   125,   680,  -105,  -105,  1634,  1634,   103,
    -105,  -105,  1634,  1634,  -105,  -105,  -105,  1634,  1634,  -105,
      67,    46,    11,  1230,    27,  1562,  1562,   125,   125,   106,
     893,  -105,  -105,  -105,   -64,   966,  1378,   612,  -105,   116,
    -105,  -105,  -105,    78,   770,   -40,  -105,  1634,  -105,  1634,
    -105,  -105,   125,    82,   770,  1415,  1452,   680,   125,    93,
    -105,  -105,  -105,   680,   125,  -105,   680,  -105
};

  /* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
     Performed when YYTABLE does not specify something else to do.  Zero
     means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
       0,     3,     0,     0,    67,   106,     1,     5,     9,     0,
       0,    75,     0,     0,    28,    30,    35,     0,     0,    34,
      29,     0,     0,     0,    81,     0,     0,    23,     0,   106,
       2,    72,    32,    71,    33,    56,     0,    68,    61,    69,
      70,    93,    75,    75,     0,     0,     0,     0,     0,     0,
       0,     0,   106,     4,     0,    91,     0,   105,     0,    98,
     106,     0,    73,    38,    74,     0,    16,     0,     0,     0,
       0,    72,    71,     0,     0,    70,    33,    51,    50,     0,
      52,    53,    24,     0,     8,     0,    54,    55,     0,     0,
       0,     0,     0,     0,     0,    57,    85,    86,    66,     0,
       0,   106,     0,    88,    90,     0,     0,     0,    99,     0,
     100,   102,   103,     0,     0,    95,    94,    97,     0,     0,
       0,     0,     0,     0,     0,    36,     0,     0,     0,    17,
      27,    44,    84,     0,    65,    58,    45,    46,    47,    48,
      49,    83,    26,    14,    15,     0,    63,    79,    80,     0,
       0,     0,     0,    21,     0,    20,    18,    19,     0,    30,
       0,   101,   104,    60,     0,     7,    78,    76,    77,     0,
      31,    37,     0,     0,    82,    62,   106,    87,    89,    12,
      21,     0,    20,    18,    19,     0,     0,    60,    60,     0,
       0,    59,    22,    96,     0,     0,     0,     0,    13,    10,
      11,    25,   107,     0,     0,     0,    42,     0,    40,     0,
      43,    64,    60,     0,     0,     0,     0,     0,    60,     0,
      41,    39,   110,     0,    60,   109,     0,   108
};

  /* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -105,  -105,  -105,  -105,    36,   166,  -104,  -105,  -105,  -105,
       8,    44,   153,    -8,   -26,  -105,  -105,    20,    12,   129,
      87,    99,  -105,    50,   -30,  -103,   -43,   -28,  -105,  -105
};

  /* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     3,     4,    30,   153,    31,   154,   164,    32,    54,
     155,    34,    35,    55,   192,    37,     8,    38,    63,    64,
     149,    39,    79,   157,    98,    56,    57,     9,    58,    59
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_int16 yytable[] =
{
      36,    83,   158,    62,   160,    36,     5,   147,    96,   185,
     186,   115,   189,    73,     6,   117,    33,   108,   110,   111,
     112,    33,   166,     7,   114,   122,   116,   206,    60,    72,
     207,    65,   118,    66,    62,    62,    99,   100,    99,   100,
     109,    74,   148,   113,    67,   185,   186,   179,   181,    99,
     100,   185,   186,   119,   103,   104,   214,   123,    40,   124,
     126,   127,   128,    40,    84,    67,    76,    76,   129,    80,
      81,    75,    76,   145,    97,    68,   176,   135,   101,   161,
      76,   199,   200,   102,    69,   141,   205,   187,    85,    86,
      87,    36,    36,   188,    36,   185,   186,   156,   156,    76,
      70,   213,   130,   105,    14,    15,    76,    33,    33,   106,
      33,   219,   122,   167,   168,    20,   107,    76,   132,   143,
     144,   193,   146,   133,    86,    87,    88,    89,    90,    91,
      92,    24,    76,    76,    76,    76,    76,   163,    27,    28,
     174,   177,   178,   156,   183,    90,    91,    92,   197,    40,
      40,   190,    40,    76,   100,   191,   194,    76,   198,   203,
     182,   201,   202,    76,   195,   196,   186,    76,    76,   212,
      76,    76,    76,   218,   222,    77,    78,   156,   156,    76,
     225,    82,   156,   227,   224,    76,   217,    71,   180,    95,
     120,   150,   223,   142,     0,    11,    12,     0,   226,   215,
      76,   216,   184,     0,     0,     0,     0,     0,    95,     0,
       0,    76,    76,     0,     0,    95,     0,     0,     0,     0,
       0,    76,    76,     0,     0,    20,    95,    76,     0,     0,
      51,     0,     0,     0,    22,    23,     0,     0,     0,    76,
      76,   136,   137,   138,   139,   140,    25,    26,    27,    28,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    76,
      76,     0,    95,     0,     0,     0,    95,     0,     0,     0,
       0,    41,    95,     0,     0,     0,    95,    95,     0,    95,
      95,    95,     0,     0,     0,     0,     0,     0,    95,     0,
       0,     0,     0,     0,    95,     0,     0,     0,     0,     0,
     -92,    42,    43,    11,    12,    44,     0,    45,    46,    95,
      47,    48,    49,    50,     0,     0,     0,     0,     0,     0,
      95,    95,    14,    15,     0,    16,    17,     0,    18,    19,
      95,    95,     0,    20,     0,     0,    95,     0,    51,     0,
       0,    41,    22,    23,     0,     0,     0,     0,    95,    95,
       0,     0,     0,     0,    25,    26,    27,    28,     0,    52,
      53,     0,     0,     0,     0,     0,   -92,     0,    95,    95,
     -92,    42,    43,    11,    12,    44,     0,    45,    46,     0,
      47,    48,    49,    50,     0,     0,     0,     0,     0,     0,
       0,     0,    14,    15,     0,    16,    17,     0,    18,    19,
       0,     0,     0,    20,     0,     0,     0,     0,    51,    41,
       0,     0,    22,    23,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    25,    26,    27,    28,     0,    52,
     134,     0,     0,     0,     0,     0,   -92,     0,   -92,    42,
      43,    11,    12,    44,     0,    45,    46,     0,    47,    48,
      49,    50,     0,     0,     0,     0,     0,     0,     0,     0,
      14,    15,     0,    16,    17,     0,    18,    19,     0,     0,
       0,    20,     0,     0,     0,     0,    51,    41,     0,     0,
      22,    23,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    25,    26,    27,    28,     0,    52,   162,     0,
       0,     0,     0,     0,   -92,     0,   -92,    42,    43,    11,
      12,    44,     0,    45,    46,     0,    47,    48,    49,    50,
       0,     0,     0,     0,     0,     0,     0,     0,    14,    15,
       0,    16,    17,     0,    18,    19,     0,     0,     0,    20,
       0,     0,     0,     0,    51,    41,     0,     0,    22,    23,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      25,    26,    27,    28,     0,    52,   165,     0,     0,     0,
       0,     0,   -92,     0,   -92,    42,    43,    11,    12,    44,
       0,    45,    46,     0,    47,    48,    49,    50,     0,     0,
       0,     0,     0,     0,     0,     0,    14,    15,     0,    16,
      17,     0,    18,    19,     0,     0,     0,    20,     0,     0,
       0,     0,    51,    41,     0,     0,    22,    23,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    25,    26,
      27,    28,     0,    52,   175,     0,     0,     0,     0,     0,
     -92,     0,   -92,    42,    43,    11,    12,    44,     0,    45,
      46,     0,    47,    48,    49,    50,     0,     0,     0,     0,
       0,     0,     0,     0,    14,    15,     0,    16,    17,     0,
      18,    19,     0,     0,     0,    20,     0,     0,     0,     0,
      51,    41,     0,     0,    22,    23,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    25,    26,    27,    28,
       0,    52,   211,     0,     0,     0,     0,     0,   -92,     0,
     -92,    42,    43,    11,    12,    44,     0,    45,    46,     0,
      47,    48,    49,    50,     0,     0,    41,     0,     0,     0,
       0,     0,    14,    15,     0,    16,    17,     0,    18,    19,
       0,     0,     0,    20,     0,     0,     0,     0,    51,     0,
       0,     0,    22,    23,     0,     0,    42,    43,    11,    12,
       0,     0,     0,     0,    25,    26,    27,    28,     0,    52,
       0,    41,     0,     0,     0,     0,   -92,    14,   159,     0,
      16,    17,     0,    18,    19,     0,     0,     0,    20,     0,
       0,     0,     0,    51,     0,     0,     0,    22,    23,     0,
       0,    42,    43,    11,    12,     0,     0,    -6,     1,    25,
      26,    27,    28,     0,     0,     0,     0,     0,     0,     0,
       0,   -92,    14,    15,     0,    16,    17,     0,    18,    19,
       0,     0,     0,    20,     0,     2,    -6,    -6,    51,     0,
      -6,    -6,    22,    23,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    25,    26,    27,    28,    -6,    -6,
      -6,   -92,    -6,    -6,     0,    -6,    -6,     0,     0,     0,
      -6,     0,     0,     0,     0,    -6,     0,     0,     0,    -6,
      -6,    96,    -6,     0,    11,    12,     0,     0,     0,     0,
       0,    -6,    -6,    -6,    -6,     0,    -6,     0,     0,     0,
       0,     0,     0,    14,    15,     0,    16,    17,     0,    18,
      19,     0,     0,     0,    20,     0,     0,     0,     0,    51,
       0,     0,     0,    22,    23,     0,    11,    12,     0,     0,
       0,     0,     0,     0,     0,    25,    26,    27,    28,     0,
       0,     0,     0,     0,   151,    14,    15,    97,    16,    17,
       0,    18,    19,     0,     0,     0,    20,     0,     0,     0,
       0,   152,    11,    12,     0,    22,    23,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    25,    26,    27,
      28,    14,    15,     0,    16,    17,     0,    18,    19,   204,
       0,     0,    20,     0,     0,     0,     0,    51,     0,    11,
      12,    22,    23,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    25,    26,    27,    28,     0,    14,    15,
     131,    16,    17,   121,    18,    19,     0,     0,     0,    20,
       0,     0,     0,     0,    51,     0,    11,    12,    22,    23,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      25,    26,    27,    28,     0,    14,    15,   208,    16,    17,
     209,    18,    19,     0,     0,     0,    20,     0,     0,     0,
       0,    51,     0,    11,    12,    22,    23,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    25,    26,    27,
      28,     0,    14,    15,     0,    16,    17,   121,    18,    19,
       0,     0,     0,    20,     0,     0,     0,     0,    51,     0,
      11,    12,    22,    23,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    25,    26,    27,    28,     0,    14,
      15,     0,    16,    17,   169,    18,    19,     0,     0,     0,
      20,     0,     0,     0,     0,    51,     0,    11,    12,    22,
      23,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    25,    26,    27,    28,     0,    14,    15,     0,    16,
      17,   172,    18,    19,     0,     0,     0,    20,     0,     0,
      10,     0,    51,     0,    11,    12,    22,    23,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    25,    26,
      27,    28,    13,    14,    15,     0,    16,    17,   173,    18,
      19,     0,     0,     0,    20,     0,     0,     0,     0,    21,
       0,     0,     0,    22,    23,     0,    24,    11,    12,     0,
       0,     0,     0,     0,     0,    25,    26,    27,    28,     0,
      29,     0,     0,     0,     0,     0,    14,    15,     0,    16,
      17,     0,    18,    19,     0,     0,     0,    20,     0,     0,
       0,     0,    51,    11,    12,     0,    22,    23,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    25,    26,
      27,    28,    14,    15,     0,    16,    17,   170,    18,    19,
      93,    94,     0,    20,     0,     0,     0,     0,    51,     0,
      11,    12,    22,    23,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    25,    26,    27,    28,     0,    14,
      15,   131,    16,    17,     0,    18,    19,     0,     0,     0,
      20,     0,     0,     0,     0,    51,     0,    11,    12,    22,
      23,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    25,    26,    27,    28,     0,    14,    15,   125,    16,
      17,     0,    18,    19,     0,     0,     0,    20,     0,     0,
       0,     0,    51,     0,    11,    12,    22,    23,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    25,    26,
      27,    28,     0,    14,    15,   131,    16,    17,     0,    18,
      19,     0,     0,     0,    20,     0,     0,     0,     0,    51,
       0,    11,    12,    22,    23,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    25,    26,    27,    28,     0,
      14,    15,   171,    16,    17,     0,    18,    19,     0,     0,
       0,    20,     0,     0,     0,     0,    51,     0,    11,    12,
      22,    23,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    25,    26,    27,    28,     0,    14,    15,   210,
      16,    17,     0,    18,    19,     0,     0,     0,    20,     0,
       0,     0,     0,    51,     0,    11,    12,    22,    23,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    25,
      26,    27,    28,     0,    14,    15,   220,    16,    17,     0,
      18,    19,     0,     0,     0,    20,     0,     0,     0,     0,
      51,     0,    11,    12,    22,    23,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    25,    26,    27,    28,
      13,    14,    15,   221,    16,    17,     0,    18,    19,     0,
       0,     0,    20,     0,     0,     0,     0,    21,    11,    12,
       0,    22,    23,     0,    24,     0,     0,     0,     0,     0,
       0,     0,     0,    25,    26,    27,    28,    14,    15,     0,
      16,    17,     0,    18,    19,    93,    94,     0,    20,     0,
       0,     0,     0,    51,     0,    11,    12,    22,    23,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    25,
      26,    27,    28,   151,    14,    15,     0,    16,    17,     0,
      18,    19,     0,     0,     0,    20,     0,     0,     0,     0,
     152,    11,    12,     0,    22,    23,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    25,    26,    27,    28,
      14,    15,     0,    16,    17,     0,    18,    19,     0,     0,
       0,    20,     0,     0,     0,     0,    61,    11,    12,     0,
      22,    23,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    25,    26,    27,    28,    14,    15,     0,    16,
      17,     0,    18,    19,     0,     0,     0,    20,     0,     0,
       0,     0,    51,     0,     0,     0,    22,    23,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    25,    26,
      27,    28
};

static const yytype_int16 yycheck[] =
{
       8,    29,   106,    11,   107,    13,    89,    60,    30,    49,
      50,    54,    39,    21,     0,    58,     8,    47,    48,    49,
      50,    13,    91,    30,    52,    94,    56,    91,    89,    21,
      94,    68,    60,    13,    42,    43,    49,    50,    49,    50,
      48,    21,    95,    51,    92,    49,    50,   151,   152,    49,
      50,    49,    50,    61,    42,    43,    96,    65,     8,    67,
      68,    69,    70,    13,    30,    92,    22,    23,    91,    25,
      26,    21,    28,   101,    96,    68,    89,    85,    89,   109,
      36,   185,   186,    94,    68,    93,   190,    91,    48,    84,
      85,    99,   100,    91,   102,    49,    50,   105,   106,    55,
      68,   204,    91,    68,    52,    53,    62,    99,   100,    68,
     102,   214,    94,   121,   122,    63,    68,    73,    91,    99,
     100,   164,   102,    47,    84,    85,    72,    73,    74,    75,
      76,    75,    88,    89,    90,    91,    92,    36,    86,    87,
      75,   149,   150,   151,   152,    74,    75,    76,   176,    99,
     100,    96,   102,   109,    50,    30,    53,   113,    91,    53,
     152,   187,   188,   119,   172,   173,    50,   123,   124,    91,
     126,   127,   128,    91,   217,    22,    23,   185,   186,   135,
     223,    28,   190,   226,    91,   141,   212,    21,   152,    36,
      61,   104,   218,    94,    -1,    33,    34,    -1,   224,   207,
     156,   209,   152,    -1,    -1,    -1,    -1,    -1,    55,    -1,
      -1,   167,   168,    -1,    -1,    62,    -1,    -1,    -1,    -1,
      -1,   177,   178,    -1,    -1,    63,    73,   183,    -1,    -1,
      68,    -1,    -1,    -1,    72,    73,    -1,    -1,    -1,   195,
     196,    88,    89,    90,    91,    92,    84,    85,    86,    87,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   215,
     216,    -1,   109,    -1,    -1,    -1,   113,    -1,    -1,    -1,
      -1,     1,   119,    -1,    -1,    -1,   123,   124,    -1,   126,
     127,   128,    -1,    -1,    -1,    -1,    -1,    -1,   135,    -1,
      -1,    -1,    -1,    -1,   141,    -1,    -1,    -1,    -1,    -1,
      30,    31,    32,    33,    34,    35,    -1,    37,    38,   156,
      40,    41,    42,    43,    -1,    -1,    -1,    -1,    -1,    -1,
     167,   168,    52,    53,    -1,    55,    56,    -1,    58,    59,
     177,   178,    -1,    63,    -1,    -1,   183,    -1,    68,    -1,
      -1,     1,    72,    73,    -1,    -1,    -1,    -1,   195,   196,
      -1,    -1,    -1,    -1,    84,    85,    86,    87,    -1,    89,
      90,    -1,    -1,    -1,    -1,    -1,    96,    -1,   215,   216,
      30,    31,    32,    33,    34,    35,    -1,    37,    38,    -1,
      40,    41,    42,    43,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    52,    53,    -1,    55,    56,    -1,    58,    59,
      -1,    -1,    -1,    63,    -1,    -1,    -1,    -1,    68,     1,
      -1,    -1,    72,    73,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    84,    85,    86,    87,    -1,    89,
      90,    -1,    -1,    -1,    -1,    -1,    96,    -1,    30,    31,
      32,    33,    34,    35,    -1,    37,    38,    -1,    40,    41,
      42,    43,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      52,    53,    -1,    55,    56,    -1,    58,    59,    -1,    -1,
      -1,    63,    -1,    -1,    -1,    -1,    68,     1,    -1,    -1,
      72,    73,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    84,    85,    86,    87,    -1,    89,    90,    -1,
      -1,    -1,    -1,    -1,    96,    -1,    30,    31,    32,    33,
      34,    35,    -1,    37,    38,    -1,    40,    41,    42,    43,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    52,    53,
      -1,    55,    56,    -1,    58,    59,    -1,    -1,    -1,    63,
      -1,    -1,    -1,    -1,    68,     1,    -1,    -1,    72,    73,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      84,    85,    86,    87,    -1,    89,    90,    -1,    -1,    -1,
      -1,    -1,    96,    -1,    30,    31,    32,    33,    34,    35,
      -1,    37,    38,    -1,    40,    41,    42,    43,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    52,    53,    -1,    55,
      56,    -1,    58,    59,    -1,    -1,    -1,    63,    -1,    -1,
      -1,    -1,    68,     1,    -1,    -1,    72,    73,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    84,    85,
      86,    87,    -1,    89,    90,    -1,    -1,    -1,    -1,    -1,
      96,    -1,    30,    31,    32,    33,    34,    35,    -1,    37,
      38,    -1,    40,    41,    42,    43,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    52,    53,    -1,    55,    56,    -1,
      58,    59,    -1,    -1,    -1,    63,    -1,    -1,    -1,    -1,
      68,     1,    -1,    -1,    72,    73,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    84,    85,    86,    87,
      -1,    89,    90,    -1,    -1,    -1,    -1,    -1,    96,    -1,
      30,    31,    32,    33,    34,    35,    -1,    37,    38,    -1,
      40,    41,    42,    43,    -1,    -1,     1,    -1,    -1,    -1,
      -1,    -1,    52,    53,    -1,    55,    56,    -1,    58,    59,
      -1,    -1,    -1,    63,    -1,    -1,    -1,    -1,    68,    -1,
      -1,    -1,    72,    73,    -1,    -1,    31,    32,    33,    34,
      -1,    -1,    -1,    -1,    84,    85,    86,    87,    -1,    89,
      -1,     1,    -1,    -1,    -1,    -1,    96,    52,    53,    -1,
      55,    56,    -1,    58,    59,    -1,    -1,    -1,    63,    -1,
      -1,    -1,    -1,    68,    -1,    -1,    -1,    72,    73,    -1,
      -1,    31,    32,    33,    34,    -1,    -1,     0,     1,    84,
      85,    86,    87,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    96,    52,    53,    -1,    55,    56,    -1,    58,    59,
      -1,    -1,    -1,    63,    -1,    28,    29,    30,    68,    -1,
      33,    34,    72,    73,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    84,    85,    86,    87,    51,    52,
      53,    91,    55,    56,    -1,    58,    59,    -1,    -1,    -1,
      63,    -1,    -1,    -1,    -1,    68,    -1,    -1,    -1,    72,
      73,    30,    75,    -1,    33,    34,    -1,    -1,    -1,    -1,
      -1,    84,    85,    86,    87,    -1,    89,    -1,    -1,    -1,
      -1,    -1,    -1,    52,    53,    -1,    55,    56,    -1,    58,
      59,    -1,    -1,    -1,    63,    -1,    -1,    -1,    -1,    68,
      -1,    -1,    -1,    72,    73,    -1,    33,    34,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    84,    85,    86,    87,    -1,
      -1,    -1,    -1,    -1,    51,    52,    53,    96,    55,    56,
      -1,    58,    59,    -1,    -1,    -1,    63,    -1,    -1,    -1,
      -1,    68,    33,    34,    -1,    72,    73,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    84,    85,    86,
      87,    52,    53,    -1,    55,    56,    -1,    58,    59,    96,
      -1,    -1,    63,    -1,    -1,    -1,    -1,    68,    -1,    33,
      34,    72,    73,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    84,    85,    86,    87,    -1,    52,    53,
      91,    55,    56,    94,    58,    59,    -1,    -1,    -1,    63,
      -1,    -1,    -1,    -1,    68,    -1,    33,    34,    72,    73,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      84,    85,    86,    87,    -1,    52,    53,    91,    55,    56,
      94,    58,    59,    -1,    -1,    -1,    63,    -1,    -1,    -1,
      -1,    68,    -1,    33,    34,    72,    73,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    84,    85,    86,
      87,    -1,    52,    53,    -1,    55,    56,    94,    58,    59,
      -1,    -1,    -1,    63,    -1,    -1,    -1,    -1,    68,    -1,
      33,    34,    72,    73,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    84,    85,    86,    87,    -1,    52,
      53,    -1,    55,    56,    94,    58,    59,    -1,    -1,    -1,
      63,    -1,    -1,    -1,    -1,    68,    -1,    33,    34,    72,
      73,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    84,    85,    86,    87,    -1,    52,    53,    -1,    55,
      56,    94,    58,    59,    -1,    -1,    -1,    63,    -1,    -1,
      29,    -1,    68,    -1,    33,    34,    72,    73,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    84,    85,
      86,    87,    51,    52,    53,    -1,    55,    56,    94,    58,
      59,    -1,    -1,    -1,    63,    -1,    -1,    -1,    -1,    68,
      -1,    -1,    -1,    72,    73,    -1,    75,    33,    34,    -1,
      -1,    -1,    -1,    -1,    -1,    84,    85,    86,    87,    -1,
      89,    -1,    -1,    -1,    -1,    -1,    52,    53,    -1,    55,
      56,    -1,    58,    59,    -1,    -1,    -1,    63,    -1,    -1,
      -1,    -1,    68,    33,    34,    -1,    72,    73,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    84,    85,
      86,    87,    52,    53,    -1,    55,    56,    93,    58,    59,
      60,    61,    -1,    63,    -1,    -1,    -1,    -1,    68,    -1,
      33,    34,    72,    73,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    84,    85,    86,    87,    -1,    52,
      53,    91,    55,    56,    -1,    58,    59,    -1,    -1,    -1,
      63,    -1,    -1,    -1,    -1,    68,    -1,    33,    34,    72,
      73,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    84,    85,    86,    87,    -1,    52,    53,    91,    55,
      56,    -1,    58,    59,    -1,    -1,    -1,    63,    -1,    -1,
      -1,    -1,    68,    -1,    33,    34,    72,    73,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    84,    85,
      86,    87,    -1,    52,    53,    91,    55,    56,    -1,    58,
      59,    -1,    -1,    -1,    63,    -1,    -1,    -1,    -1,    68,
      -1,    33,    34,    72,    73,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    84,    85,    86,    87,    -1,
      52,    53,    91,    55,    56,    -1,    58,    59,    -1,    -1,
      -1,    63,    -1,    -1,    -1,    -1,    68,    -1,    33,    34,
      72,    73,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    84,    85,    86,    87,    -1,    52,    53,    91,
      55,    56,    -1,    58,    59,    -1,    -1,    -1,    63,    -1,
      -1,    -1,    -1,    68,    -1,    33,    34,    72,    73,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    84,
      85,    86,    87,    -1,    52,    53,    91,    55,    56,    -1,
      58,    59,    -1,    -1,    -1,    63,    -1,    -1,    -1,    -1,
      68,    -1,    33,    34,    72,    73,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    84,    85,    86,    87,
      51,    52,    53,    91,    55,    56,    -1,    58,    59,    -1,
      -1,    -1,    63,    -1,    -1,    -1,    -1,    68,    33,    34,
      -1,    72,    73,    -1,    75,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    84,    85,    86,    87,    52,    53,    -1,
      55,    56,    -1,    58,    59,    60,    61,    -1,    63,    -1,
      -1,    -1,    -1,    68,    -1,    33,    34,    72,    73,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    84,
      85,    86,    87,    51,    52,    53,    -1,    55,    56,    -1,
      58,    59,    -1,    -1,    -1,    63,    -1,    -1,    -1,    -1,
      68,    33,    34,    -1,    72,    73,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    84,    85,    86,    87,
      52,    53,    -1,    55,    56,    -1,    58,    59,    -1,    -1,
      -1,    63,    -1,    -1,    -1,    -1,    68,    33,    34,    -1,
      72,    73,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    84,    85,    86,    87,    52,    53,    -1,    55,
      56,    -1,    58,    59,    -1,    -1,    -1,    63,    -1,    -1,
      -1,    -1,    68,    -1,    -1,    -1,    72,    73,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    84,    85,
      86,    87
};

  /* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
     symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,     1,    28,    98,    99,    89,     0,    30,   113,   124,
      29,    33,    34,    51,    52,    53,    55,    56,    58,    59,
      63,    68,    72,    73,    75,    84,    85,    86,    87,    89,
     100,   102,   105,   107,   108,   109,   110,   112,   114,   118,
     120,     1,    31,    32,    35,    37,    38,    40,    41,    42,
      43,    68,    89,    90,   106,   110,   122,   123,   125,   126,
      89,    68,   110,   115,   116,    68,   114,    92,    68,    68,
      68,   102,   107,   110,   114,   120,   108,   109,   109,   119,
     108,   108,   109,   124,    30,    48,    84,    85,    72,    73,
      74,    75,    76,    60,    61,   109,    30,    96,   121,    49,
      50,    89,    94,   115,   115,    68,    68,    68,   121,   110,
     121,   121,   121,   110,   124,   123,   121,   123,   124,   110,
     116,    94,    94,   110,   110,    91,   110,   110,   110,    91,
      91,    91,    91,    47,    90,   110,   109,   109,   109,   109,
     109,   110,   118,   114,   114,   124,   114,    60,    95,   117,
     117,    51,    68,   101,   103,   107,   110,   120,   103,    53,
     122,   121,    90,    36,   104,    90,    91,   110,   110,    94,
      93,    91,    94,    94,    75,    90,    89,   110,   110,   103,
     101,   103,   107,   110,   120,    49,    50,    91,    91,    39,
      96,    30,   111,   123,    53,   110,   110,   124,    91,   103,
     103,   111,   111,    53,    96,   103,    91,    94,    91,    94,
      91,    90,    91,   122,    96,   110,   110,   111,    91,   122,
      91,    91,   123,   111,    91,   123,   111,   123
};

  /* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    97,    98,    98,    99,    99,    99,   100,   100,   100,
     101,   101,   101,   101,   102,   102,   102,   102,   103,   103,
     103,   103,   104,   105,   105,   106,   107,   107,   108,   108,
     108,   108,   108,   109,   109,   109,   109,   109,   109,   109,
     109,   109,   109,   109,   109,   109,   109,   109,   109,   109,
     109,   109,   109,   109,   109,   109,   110,   110,   110,   111,
     111,   112,   112,   112,   112,   112,   113,   113,   113,   114,
     114,   114,   114,   115,   115,   115,   116,   116,   116,   117,
     117,   119,   118,   120,   120,   121,   121,   122,   122,   122,
     122,   122,   122,   122,   123,   123,   123,   123,   123,   123,
     123,   123,   123,   123,   123,   124,   124,   125,   126,   126,
     126
};

  /* YYR2[YYN] -- Number of symbols on the right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     3,     1,     4,     2,     0,     4,     2,     0,
       3,     3,     2,     3,     3,     3,     2,     3,     1,     1,
       1,     1,     2,     1,     2,     5,     3,     3,     1,     1,
       1,     4,     1,     1,     1,     1,     3,     4,     2,     8,
       6,     8,     6,     6,     3,     3,     3,     3,     3,     3,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     1,
       0,     1,     4,     3,     6,     3,     3,     0,     2,     1,
       1,     1,     1,     1,     1,     0,     3,     3,     3,     1,
       1,     0,     4,     3,     3,     1,     1,     4,     2,     4,
       2,     1,     0,     1,     2,     2,     4,     2,     1,     2,
       2,     3,     2,     2,     3,     2,     0,     5,    10,     9,
       8
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
#line 126 "awk.g.y" /* yacc.c:1646  */
    { if (errorflag==0) winner = (node *)stat3(PROGRAM, (yyvsp[-2]), (yyvsp[-1]), (yyvsp[0])); }
#line 1836 "y.tab.c" /* yacc.c:1646  */
    break;

  case 3:
#line 127 "awk.g.y" /* yacc.c:1646  */
    { yyclearin; yyerror("bailing out"); }
#line 1842 "y.tab.c" /* yacc.c:1646  */
    break;

  case 4:
#line 131 "awk.g.y" /* yacc.c:1646  */
    { PUTS("XBEGIN list"); (yyval) = (yyvsp[-1]); }
#line 1848 "y.tab.c" /* yacc.c:1646  */
    break;

  case 6:
#line 133 "awk.g.y" /* yacc.c:1646  */
    { PUTS("empty XBEGIN"); (yyval) = nullstat; }
#line 1854 "y.tab.c" /* yacc.c:1646  */
    break;

  case 7:
#line 137 "awk.g.y" /* yacc.c:1646  */
    { PUTS("XEND list"); (yyval) = (yyvsp[-1]); }
#line 1860 "y.tab.c" /* yacc.c:1646  */
    break;

  case 9:
#line 139 "awk.g.y" /* yacc.c:1646  */
    { PUTS("empty END"); (yyval) = nullstat; }
#line 1866 "y.tab.c" /* yacc.c:1646  */
    break;

  case 10:
#line 143 "awk.g.y" /* yacc.c:1646  */
    { PUTS("cond||cond"); (yyval) = op2(BOR, (yyvsp[-2]), (yyvsp[0])); }
#line 1872 "y.tab.c" /* yacc.c:1646  */
    break;

  case 11:
#line 144 "awk.g.y" /* yacc.c:1646  */
    { PUTS("cond&&cond"); (yyval) = op2(AND, (yyvsp[-2]), (yyvsp[0])); }
#line 1878 "y.tab.c" /* yacc.c:1646  */
    break;

  case 12:
#line 145 "awk.g.y" /* yacc.c:1646  */
    { PUTS("!cond"); (yyval) = op1(NOT, (yyvsp[0])); }
#line 1884 "y.tab.c" /* yacc.c:1646  */
    break;

  case 13:
#line 146 "awk.g.y" /* yacc.c:1646  */
    { (yyval) = (yyvsp[-1]); }
#line 1890 "y.tab.c" /* yacc.c:1646  */
    break;

  case 14:
#line 150 "awk.g.y" /* yacc.c:1646  */
    { PUTS("pat||pat"); (yyval) = op2(BOR, (yyvsp[-2]), (yyvsp[0])); }
#line 1896 "y.tab.c" /* yacc.c:1646  */
    break;

  case 15:
#line 151 "awk.g.y" /* yacc.c:1646  */
    { PUTS("pat&&pat"); (yyval) = op2(AND, (yyvsp[-2]), (yyvsp[0])); }
#line 1902 "y.tab.c" /* yacc.c:1646  */
    break;

  case 16:
#line 152 "awk.g.y" /* yacc.c:1646  */
    { PUTS("!pat"); (yyval) = op1(NOT, (yyvsp[0])); }
#line 1908 "y.tab.c" /* yacc.c:1646  */
    break;

  case 17:
#line 153 "awk.g.y" /* yacc.c:1646  */
    { (yyval) = (yyvsp[-1]); }
#line 1914 "y.tab.c" /* yacc.c:1646  */
    break;

  case 18:
#line 157 "awk.g.y" /* yacc.c:1646  */
    { PUTS("expr"); (yyval) = op2(NE, (yyvsp[0]), valtonode(lookup("$zero&null", symtab, 0), CCON)); }
#line 1920 "y.tab.c" /* yacc.c:1646  */
    break;

  case 19:
#line 158 "awk.g.y" /* yacc.c:1646  */
    { PUTS("relexpr"); }
#line 1926 "y.tab.c" /* yacc.c:1646  */
    break;

  case 20:
#line 159 "awk.g.y" /* yacc.c:1646  */
    { PUTS("lexexpr"); }
#line 1932 "y.tab.c" /* yacc.c:1646  */
    break;

  case 21:
#line 160 "awk.g.y" /* yacc.c:1646  */
    { PUTS("compcond"); }
#line 1938 "y.tab.c" /* yacc.c:1646  */
    break;

  case 22:
#line 164 "awk.g.y" /* yacc.c:1646  */
    { PUTS("else"); }
#line 1944 "y.tab.c" /* yacc.c:1646  */
    break;

  case 23:
#line 168 "awk.g.y" /* yacc.c:1646  */
    { PUTS("field"); (yyval) = valtonode((yyvsp[0]), CFLD); }
#line 1950 "y.tab.c" /* yacc.c:1646  */
    break;

  case 24:
#line 169 "awk.g.y" /* yacc.c:1646  */
    { PUTS("ind field"); (yyval) = op1(INDIRECT, (yyvsp[0])); }
#line 1956 "y.tab.c" /* yacc.c:1646  */
    break;

  case 25:
#line 173 "awk.g.y" /* yacc.c:1646  */
    { PUTS("if(cond)"); (yyval) = (yyvsp[-2]); }
#line 1962 "y.tab.c" /* yacc.c:1646  */
    break;

  case 26:
#line 177 "awk.g.y" /* yacc.c:1646  */
    { PUTS("expr~re"); (yyval) = op2((intptr_t)(yyvsp[-1]), (yyvsp[-2]), (void *)makedfa((yyvsp[0]))); }
#line 1968 "y.tab.c" /* yacc.c:1646  */
    break;

  case 27:
#line 178 "awk.g.y" /* yacc.c:1646  */
    { PUTS("(lex_expr)"); (yyval) = (yyvsp[-1]); }
#line 1974 "y.tab.c" /* yacc.c:1646  */
    break;

  case 28:
#line 182 "awk.g.y" /* yacc.c:1646  */
    {PUTS("number"); (yyval) = valtonode((yyvsp[0]), CCON); }
#line 1980 "y.tab.c" /* yacc.c:1646  */
    break;

  case 29:
#line 183 "awk.g.y" /* yacc.c:1646  */
    { PUTS("string"); (yyval) = valtonode((yyvsp[0]), CCON); }
#line 1986 "y.tab.c" /* yacc.c:1646  */
    break;

  case 30:
#line 184 "awk.g.y" /* yacc.c:1646  */
    { PUTS("var"); (yyval) = valtonode((yyvsp[0]), CVAR); }
#line 1992 "y.tab.c" /* yacc.c:1646  */
    break;

  case 31:
#line 185 "awk.g.y" /* yacc.c:1646  */
    { PUTS("array[]"); (yyval) = op2(ARRAY, (yyvsp[-3]), (yyvsp[-1])); }
#line 1998 "y.tab.c" /* yacc.c:1646  */
    break;

  case 34:
#line 190 "awk.g.y" /* yacc.c:1646  */
    { PUTS("getline"); (yyval) = op1(GETLINE, 0); }
#line 2004 "y.tab.c" /* yacc.c:1646  */
    break;

  case 35:
#line 191 "awk.g.y" /* yacc.c:1646  */
    { PUTS("func");
			(yyval) = op2(FNCN, (yyvsp[0]), valtonode(lookup("$record", symtab, 0), CFLD));
			}
#line 2012 "y.tab.c" /* yacc.c:1646  */
    break;

  case 36:
#line 194 "awk.g.y" /* yacc.c:1646  */
    { PUTS("func()"); 
			(yyval) = op2(FNCN, (yyvsp[-2]), valtonode(lookup("$record", symtab, 0), CFLD));
			}
#line 2020 "y.tab.c" /* yacc.c:1646  */
    break;

  case 37:
#line 197 "awk.g.y" /* yacc.c:1646  */
    { PUTS("func(expr)"); (yyval) = op2(FNCN, (yyvsp[-3]), (yyvsp[-1])); }
#line 2026 "y.tab.c" /* yacc.c:1646  */
    break;

  case 38:
#line 198 "awk.g.y" /* yacc.c:1646  */
    { PUTS("sprintf"); (yyval) = op1((intptr_t)(yyvsp[-1]), (yyvsp[0])); }
#line 2032 "y.tab.c" /* yacc.c:1646  */
    break;

  case 39:
#line 200 "awk.g.y" /* yacc.c:1646  */
    { PUTS("substr(e,e,e)"); (yyval) = op3(SUBSTR, (yyvsp[-5]), (yyvsp[-3]), (yyvsp[-1])); }
#line 2038 "y.tab.c" /* yacc.c:1646  */
    break;

  case 40:
#line 202 "awk.g.y" /* yacc.c:1646  */
    { PUTS("substr(e,e,e)"); (yyval) = op3(SUBSTR, (yyvsp[-3]), (yyvsp[-1]), nullstat); }
#line 2044 "y.tab.c" /* yacc.c:1646  */
    break;

  case 41:
#line 204 "awk.g.y" /* yacc.c:1646  */
    { PUTS("split(e,e,e)"); (yyval) = op3(SPLIT, (yyvsp[-5]), (yyvsp[-3]), (yyvsp[-1])); }
#line 2050 "y.tab.c" /* yacc.c:1646  */
    break;

  case 42:
#line 206 "awk.g.y" /* yacc.c:1646  */
    { PUTS("split(e,e,e)"); (yyval) = op3(SPLIT, (yyvsp[-3]), (yyvsp[-1]), nullstat); }
#line 2056 "y.tab.c" /* yacc.c:1646  */
    break;

  case 43:
#line 208 "awk.g.y" /* yacc.c:1646  */
    { PUTS("index(e,e)"); (yyval) = op2(INDEX, (yyvsp[-3]), (yyvsp[-1])); }
#line 2062 "y.tab.c" /* yacc.c:1646  */
    break;

  case 44:
#line 209 "awk.g.y" /* yacc.c:1646  */
    {PUTS("(expr)");  (yyval) = (yyvsp[-1]); }
#line 2068 "y.tab.c" /* yacc.c:1646  */
    break;

  case 45:
#line 210 "awk.g.y" /* yacc.c:1646  */
    { PUTS("t+t"); (yyval) = op2(ADD, (yyvsp[-2]), (yyvsp[0])); }
#line 2074 "y.tab.c" /* yacc.c:1646  */
    break;

  case 46:
#line 211 "awk.g.y" /* yacc.c:1646  */
    { PUTS("t-t"); (yyval) = op2(MINUS, (yyvsp[-2]), (yyvsp[0])); }
#line 2080 "y.tab.c" /* yacc.c:1646  */
    break;

  case 47:
#line 212 "awk.g.y" /* yacc.c:1646  */
    { PUTS("t*t"); (yyval) = op2(MULT, (yyvsp[-2]), (yyvsp[0])); }
#line 2086 "y.tab.c" /* yacc.c:1646  */
    break;

  case 48:
#line 213 "awk.g.y" /* yacc.c:1646  */
    { PUTS("t/t"); (yyval) = op2(DIVIDE, (yyvsp[-2]), (yyvsp[0])); }
#line 2092 "y.tab.c" /* yacc.c:1646  */
    break;

  case 49:
#line 214 "awk.g.y" /* yacc.c:1646  */
    { PUTS("t%t"); (yyval) = op2(MOD, (yyvsp[-2]), (yyvsp[0])); }
#line 2098 "y.tab.c" /* yacc.c:1646  */
    break;

  case 50:
#line 215 "awk.g.y" /* yacc.c:1646  */
    { PUTS("-term"); (yyval) = op1(UMINUS, (yyvsp[0])); }
#line 2104 "y.tab.c" /* yacc.c:1646  */
    break;

  case 51:
#line 216 "awk.g.y" /* yacc.c:1646  */
    { PUTS("+term"); (yyval) = (yyvsp[0]); }
#line 2110 "y.tab.c" /* yacc.c:1646  */
    break;

  case 52:
#line 217 "awk.g.y" /* yacc.c:1646  */
    { PUTS("++var"); (yyval) = op1(PREINCR, (yyvsp[0])); }
#line 2116 "y.tab.c" /* yacc.c:1646  */
    break;

  case 53:
#line 218 "awk.g.y" /* yacc.c:1646  */
    { PUTS("--var"); (yyval) = op1(PREDECR, (yyvsp[0])); }
#line 2122 "y.tab.c" /* yacc.c:1646  */
    break;

  case 54:
#line 219 "awk.g.y" /* yacc.c:1646  */
    { PUTS("var++"); (yyval)= op1(POSTINCR, (yyvsp[-1])); }
#line 2128 "y.tab.c" /* yacc.c:1646  */
    break;

  case 55:
#line 220 "awk.g.y" /* yacc.c:1646  */
    { PUTS("var--"); (yyval)= op1(POSTDECR, (yyvsp[-1])); }
#line 2134 "y.tab.c" /* yacc.c:1646  */
    break;

  case 56:
#line 224 "awk.g.y" /* yacc.c:1646  */
    { PUTS("term"); }
#line 2140 "y.tab.c" /* yacc.c:1646  */
    break;

  case 57:
#line 225 "awk.g.y" /* yacc.c:1646  */
    { PUTS("expr term"); (yyval) = op2(CAT, (yyvsp[-1]), (yyvsp[0])); }
#line 2146 "y.tab.c" /* yacc.c:1646  */
    break;

  case 58:
#line 226 "awk.g.y" /* yacc.c:1646  */
    { PUTS("var=expr"); (yyval) = stat2((intptr_t)(yyvsp[-1]), (yyvsp[-2]), (yyvsp[0])); }
#line 2152 "y.tab.c" /* yacc.c:1646  */
    break;

  case 61:
#line 235 "awk.g.y" /* yacc.c:1646  */
    { PUTS("pattern"); (yyval) = stat2(PASTAT, (yyvsp[0]), genprint()); }
#line 2158 "y.tab.c" /* yacc.c:1646  */
    break;

  case 62:
#line 236 "awk.g.y" /* yacc.c:1646  */
    { PUTS("pattern {...}"); (yyval) = stat2(PASTAT, (yyvsp[-3]), (yyvsp[-1])); }
#line 2164 "y.tab.c" /* yacc.c:1646  */
    break;

  case 63:
#line 237 "awk.g.y" /* yacc.c:1646  */
    { PUTS("srch,srch"); (yyval) = pa2stat((yyvsp[-2]), (yyvsp[0]), genprint()); }
#line 2170 "y.tab.c" /* yacc.c:1646  */
    break;

  case 64:
#line 239 "awk.g.y" /* yacc.c:1646  */
    { PUTS("srch, srch {...}"); (yyval) = pa2stat((yyvsp[-5]), (yyvsp[-3]), (yyvsp[-1])); }
#line 2176 "y.tab.c" /* yacc.c:1646  */
    break;

  case 65:
#line 240 "awk.g.y" /* yacc.c:1646  */
    { PUTS("null pattern {...}"); (yyval) = stat2(PASTAT, nullstat, (yyvsp[-1])); }
#line 2182 "y.tab.c" /* yacc.c:1646  */
    break;

  case 66:
#line 244 "awk.g.y" /* yacc.c:1646  */
    { PUTS("pa_stats pa_stat"); (yyval) = linkum((yyvsp[-2]), (yyvsp[-1])); }
#line 2188 "y.tab.c" /* yacc.c:1646  */
    break;

  case 67:
#line 245 "awk.g.y" /* yacc.c:1646  */
    { PUTS("null pa_stat"); (yyval) = nullstat; }
#line 2194 "y.tab.c" /* yacc.c:1646  */
    break;

  case 68:
#line 246 "awk.g.y" /* yacc.c:1646  */
    {PUTS("pa_stats pa_stat"); (yyval) = linkum((yyvsp[-1]), (yyvsp[0])); }
#line 2200 "y.tab.c" /* yacc.c:1646  */
    break;

  case 69:
#line 250 "awk.g.y" /* yacc.c:1646  */
    { PUTS("regex");
		(yyval) = op2(MATCH, valtonode(lookup("$record", symtab, 0), CFLD), (void *)makedfa((yyvsp[0])));
		}
#line 2208 "y.tab.c" /* yacc.c:1646  */
    break;

  case 70:
#line 253 "awk.g.y" /* yacc.c:1646  */
    { PUTS("relexpr"); }
#line 2214 "y.tab.c" /* yacc.c:1646  */
    break;

  case 71:
#line 254 "awk.g.y" /* yacc.c:1646  */
    { PUTS("lexexpr"); }
#line 2220 "y.tab.c" /* yacc.c:1646  */
    break;

  case 72:
#line 255 "awk.g.y" /* yacc.c:1646  */
    { PUTS("comp pat"); }
#line 2226 "y.tab.c" /* yacc.c:1646  */
    break;

  case 73:
#line 259 "awk.g.y" /* yacc.c:1646  */
    { PUTS("expr"); }
#line 2232 "y.tab.c" /* yacc.c:1646  */
    break;

  case 74:
#line 260 "awk.g.y" /* yacc.c:1646  */
    { PUTS("pe_list"); }
#line 2238 "y.tab.c" /* yacc.c:1646  */
    break;

  case 75:
#line 261 "awk.g.y" /* yacc.c:1646  */
    { PUTS("null print_list"); (yyval) = valtonode(lookup("$record", symtab, 0), CFLD); }
#line 2244 "y.tab.c" /* yacc.c:1646  */
    break;

  case 76:
#line 265 "awk.g.y" /* yacc.c:1646  */
    {(yyval) = linkum((yyvsp[-2]), (yyvsp[0])); }
#line 2250 "y.tab.c" /* yacc.c:1646  */
    break;

  case 77:
#line 266 "awk.g.y" /* yacc.c:1646  */
    {(yyval) = linkum((yyvsp[-2]), (yyvsp[0])); }
#line 2256 "y.tab.c" /* yacc.c:1646  */
    break;

  case 78:
#line 267 "awk.g.y" /* yacc.c:1646  */
    {(yyval) = (yyvsp[-1]); }
#line 2262 "y.tab.c" /* yacc.c:1646  */
    break;

  case 81:
#line 276 "awk.g.y" /* yacc.c:1646  */
    { startreg(); }
#line 2268 "y.tab.c" /* yacc.c:1646  */
    break;

  case 82:
#line 278 "awk.g.y" /* yacc.c:1646  */
    { PUTS("/r/"); (yyval) = (yyvsp[-1]); }
#line 2274 "y.tab.c" /* yacc.c:1646  */
    break;

  case 83:
#line 283 "awk.g.y" /* yacc.c:1646  */
    { PUTS("expr relop expr"); (yyval) = op2((intptr_t)(yyvsp[-1]), (yyvsp[-2]), (yyvsp[0])); }
#line 2280 "y.tab.c" /* yacc.c:1646  */
    break;

  case 84:
#line 285 "awk.g.y" /* yacc.c:1646  */
    { PUTS("(relexpr)"); (yyval) = (yyvsp[-1]); }
#line 2286 "y.tab.c" /* yacc.c:1646  */
    break;

  case 87:
#line 295 "awk.g.y" /* yacc.c:1646  */
    { PUTS("print>stat"); (yyval) = stat3((intptr_t)(yyvsp[-3]), (yyvsp[-2]), (yyvsp[-1]), (yyvsp[0])); }
#line 2292 "y.tab.c" /* yacc.c:1646  */
    break;

  case 88:
#line 297 "awk.g.y" /* yacc.c:1646  */
    { PUTS("print list"); (yyval) = stat3((intptr_t)(yyvsp[-1]), (yyvsp[0]), nullstat, nullstat); }
#line 2298 "y.tab.c" /* yacc.c:1646  */
    break;

  case 89:
#line 299 "awk.g.y" /* yacc.c:1646  */
    { PUTS("printf>stat"); (yyval) = stat3((intptr_t)(yyvsp[-3]), (yyvsp[-2]), (yyvsp[-1]), (yyvsp[0])); }
#line 2304 "y.tab.c" /* yacc.c:1646  */
    break;

  case 90:
#line 301 "awk.g.y" /* yacc.c:1646  */
    { PUTS("printf list"); (yyval) = stat3((intptr_t)(yyvsp[-1]), (yyvsp[0]), nullstat, nullstat); }
#line 2310 "y.tab.c" /* yacc.c:1646  */
    break;

  case 91:
#line 302 "awk.g.y" /* yacc.c:1646  */
    { PUTS("expr"); (yyval) = exptostat((yyvsp[0])); }
#line 2316 "y.tab.c" /* yacc.c:1646  */
    break;

  case 92:
#line 303 "awk.g.y" /* yacc.c:1646  */
    { PUTS("null simple statement"); (yyval) = nullstat; }
#line 2322 "y.tab.c" /* yacc.c:1646  */
    break;

  case 93:
#line 304 "awk.g.y" /* yacc.c:1646  */
    { yyclearin; yyerror("illegal statement"); (yyval) = nullstat; }
#line 2328 "y.tab.c" /* yacc.c:1646  */
    break;

  case 94:
#line 308 "awk.g.y" /* yacc.c:1646  */
    { PUTS("simple stat"); }
#line 2334 "y.tab.c" /* yacc.c:1646  */
    break;

  case 95:
#line 309 "awk.g.y" /* yacc.c:1646  */
    { PUTS("if stat"); (yyval) = stat3(IF, (yyvsp[-1]), (yyvsp[0]), nullstat); }
#line 2340 "y.tab.c" /* yacc.c:1646  */
    break;

  case 96:
#line 311 "awk.g.y" /* yacc.c:1646  */
    { PUTS("if-else stat"); (yyval) = stat3(IF, (yyvsp[-3]), (yyvsp[-2]), (yyvsp[0])); }
#line 2346 "y.tab.c" /* yacc.c:1646  */
    break;

  case 97:
#line 312 "awk.g.y" /* yacc.c:1646  */
    { PUTS("while stat"); (yyval) = stat2(WHILE, (yyvsp[-1]), (yyvsp[0])); }
#line 2352 "y.tab.c" /* yacc.c:1646  */
    break;

  case 98:
#line 313 "awk.g.y" /* yacc.c:1646  */
    { PUTS("for stat"); }
#line 2358 "y.tab.c" /* yacc.c:1646  */
    break;

  case 99:
#line 314 "awk.g.y" /* yacc.c:1646  */
    { PUTS("next"); (yyval) = stat1(NEXT, 0); }
#line 2364 "y.tab.c" /* yacc.c:1646  */
    break;

  case 100:
#line 315 "awk.g.y" /* yacc.c:1646  */
    { PUTS("exit"); (yyval) = stat1(EXIT, 0); }
#line 2370 "y.tab.c" /* yacc.c:1646  */
    break;

  case 101:
#line 316 "awk.g.y" /* yacc.c:1646  */
    { PUTS("exit"); (yyval) = stat1(EXIT, (yyvsp[-1])); }
#line 2376 "y.tab.c" /* yacc.c:1646  */
    break;

  case 102:
#line 317 "awk.g.y" /* yacc.c:1646  */
    { PUTS("break"); (yyval) = stat1(BREAK, 0); }
#line 2382 "y.tab.c" /* yacc.c:1646  */
    break;

  case 103:
#line 318 "awk.g.y" /* yacc.c:1646  */
    { PUTS("continue"); (yyval) = stat1(CONTINUE, 0); }
#line 2388 "y.tab.c" /* yacc.c:1646  */
    break;

  case 104:
#line 319 "awk.g.y" /* yacc.c:1646  */
    { PUTS("{statlist}"); (yyval) = (yyvsp[-1]); }
#line 2394 "y.tab.c" /* yacc.c:1646  */
    break;

  case 105:
#line 323 "awk.g.y" /* yacc.c:1646  */
    { PUTS("stat_list stat"); (yyval) = linkum((yyvsp[-1]), (yyvsp[0])); }
#line 2400 "y.tab.c" /* yacc.c:1646  */
    break;

  case 106:
#line 324 "awk.g.y" /* yacc.c:1646  */
    { PUTS("null stat list"); (yyval) = nullstat; }
#line 2406 "y.tab.c" /* yacc.c:1646  */
    break;

  case 107:
#line 328 "awk.g.y" /* yacc.c:1646  */
    { PUTS("while(cond)"); (yyval) = (yyvsp[-2]); }
#line 2412 "y.tab.c" /* yacc.c:1646  */
    break;

  case 108:
#line 333 "awk.g.y" /* yacc.c:1646  */
    { PUTS("for(e;e;e)"); (yyval) = stat4(FOR, (yyvsp[-7]), (yyvsp[-5]), (yyvsp[-3]), (yyvsp[0])); }
#line 2418 "y.tab.c" /* yacc.c:1646  */
    break;

  case 109:
#line 335 "awk.g.y" /* yacc.c:1646  */
    { PUTS("for(e;e;e)"); (yyval) = stat4(FOR, (yyvsp[-6]), nullstat, (yyvsp[-3]), (yyvsp[0])); }
#line 2424 "y.tab.c" /* yacc.c:1646  */
    break;

  case 110:
#line 337 "awk.g.y" /* yacc.c:1646  */
    { PUTS("for(v in v)"); (yyval) = stat3(IN, (yyvsp[-5]), (yyvsp[-3]), (yyvsp[0])); }
#line 2430 "y.tab.c" /* yacc.c:1646  */
    break;


#line 2434 "y.tab.c" /* yacc.c:1646  */
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
#line 340 "awk.g.y" /* yacc.c:1906  */

