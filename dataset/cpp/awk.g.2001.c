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
#line 29 "awk.g.2001.y" /* yacc.c:339  */

#include "awk.h"
#include <unistd.h>
#include <inttypes.h>
#include <pfmt.h>
int yywrap(void) { return(1); }
#ifndef	DEBUG
#	define	PUTS(x)
#endif
Node	*beginloc = 0, *endloc = 0;
int	infunc	= 0;	/* = 1 if in arglist or body of func */
unsigned char	*curfname = 0;
Node	*arglist = 0;	/* list of args for current function */
static void setfname(Cell *);
static int constnode(Node *);
static unsigned char *strnode(Node *);
static Node *notnull(Node *);
extern	const char illstat[];

extern int	yylex(void);

#line 88 "y.tab.c" /* yacc.c:339  */

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
    FIRSTTOKEN = 258,
    PROGRAM = 259,
    PASTAT = 260,
    PASTAT2 = 261,
    XBEGIN = 262,
    XEND = 263,
    NL = 264,
    ARRAY = 265,
    MATCH = 266,
    NOTMATCH = 267,
    MATCHOP = 268,
    FINAL = 269,
    DOT = 270,
    ALL = 271,
    CCL = 272,
    NCCL = 273,
    CHAR = 274,
    MCHAR = 275,
    OR = 276,
    STAR = 277,
    QUEST = 278,
    PLUS = 279,
    AND = 280,
    BOR = 281,
    APPEND = 282,
    EQ = 283,
    GE = 284,
    GT = 285,
    LE = 286,
    LT = 287,
    NE = 288,
    IN = 289,
    ARG = 290,
    BLTIN = 291,
    BREAK = 292,
    CONTINUE = 293,
    DELETE = 294,
    DO = 295,
    EXIT = 296,
    FOR = 297,
    FUNC = 298,
    SUB = 299,
    GSUB = 300,
    IF = 301,
    INDEX = 302,
    LSUBSTR = 303,
    MATCHFCN = 304,
    NEXT = 305,
    ADD = 306,
    MINUS = 307,
    MULT = 308,
    DIVIDE = 309,
    MOD = 310,
    ASSIGN = 311,
    ASGNOP = 312,
    ADDEQ = 313,
    SUBEQ = 314,
    MULTEQ = 315,
    DIVEQ = 316,
    MODEQ = 317,
    POWEQ = 318,
    PRINT = 319,
    PRINTF = 320,
    SPRINTF = 321,
    ELSE = 322,
    INTEST = 323,
    CONDEXPR = 324,
    POSTINCR = 325,
    PREINCR = 326,
    POSTDECR = 327,
    PREDECR = 328,
    VAR = 329,
    IVAR = 330,
    VARNF = 331,
    CALL = 332,
    NUMBER = 333,
    STRING = 334,
    FIELD = 335,
    REGEXPR = 336,
    GETLINE = 337,
    RETURN = 338,
    SPLIT = 339,
    SUBSTR = 340,
    WHILE = 341,
    CAT = 342,
    NOT = 343,
    UMINUS = 344,
    POWER = 345,
    DECR = 346,
    INCR = 347,
    INDIRECT = 348,
    LASTTOKEN = 349
  };
#endif
/* Tokens.  */
#define FIRSTTOKEN 258
#define PROGRAM 259
#define PASTAT 260
#define PASTAT2 261
#define XBEGIN 262
#define XEND 263
#define NL 264
#define ARRAY 265
#define MATCH 266
#define NOTMATCH 267
#define MATCHOP 268
#define FINAL 269
#define DOT 270
#define ALL 271
#define CCL 272
#define NCCL 273
#define CHAR 274
#define MCHAR 275
#define OR 276
#define STAR 277
#define QUEST 278
#define PLUS 279
#define AND 280
#define BOR 281
#define APPEND 282
#define EQ 283
#define GE 284
#define GT 285
#define LE 286
#define LT 287
#define NE 288
#define IN 289
#define ARG 290
#define BLTIN 291
#define BREAK 292
#define CONTINUE 293
#define DELETE 294
#define DO 295
#define EXIT 296
#define FOR 297
#define FUNC 298
#define SUB 299
#define GSUB 300
#define IF 301
#define INDEX 302
#define LSUBSTR 303
#define MATCHFCN 304
#define NEXT 305
#define ADD 306
#define MINUS 307
#define MULT 308
#define DIVIDE 309
#define MOD 310
#define ASSIGN 311
#define ASGNOP 312
#define ADDEQ 313
#define SUBEQ 314
#define MULTEQ 315
#define DIVEQ 316
#define MODEQ 317
#define POWEQ 318
#define PRINT 319
#define PRINTF 320
#define SPRINTF 321
#define ELSE 322
#define INTEST 323
#define CONDEXPR 324
#define POSTINCR 325
#define PREINCR 326
#define POSTDECR 327
#define PREDECR 328
#define VAR 329
#define IVAR 330
#define VARNF 331
#define CALL 332
#define NUMBER 333
#define STRING 334
#define FIELD 335
#define REGEXPR 336
#define GETLINE 337
#define RETURN 338
#define SPLIT 339
#define SUBSTR 340
#define WHILE 341
#define CAT 342
#define NOT 343
#define UMINUS 344
#define POWER 345
#define DECR 346
#define INCR 347
#define INDIRECT 348
#define LASTTOKEN 349

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED

union YYSTYPE
{
#line 51 "awk.g.2001.y" /* yacc.c:355  */

	Node	*p;
	Cell	*cp;
	intptr_t	i;
	unsigned char	*s;

#line 320 "y.tab.c" /* yacc.c:355  */
};

typedef union YYSTYPE YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif


extern YYSTYPE yylval;

int yyparse (void);



/* Copy the second part of user declarations.  */

#line 337 "y.tab.c" /* yacc.c:358  */

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
#define YYFINAL  8
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   4663

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  111
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  43
/* YYNRULES -- Number of rules.  */
#define YYNRULES  177
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  354

/* YYTRANSLATE[YYX] -- Symbol number corresponding to YYX as returned
   by yylex, with out-of-bounds checking.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   349

#define YYTRANSLATE(YYX)                                                \
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, without out-of-bounds checking.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,   103,     2,     2,
      12,    16,   102,   100,    10,   101,     2,    15,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    93,    14,
       2,     2,     2,    92,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    18,     2,    19,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    11,    13,    17,     2,     2,     2,     2,
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
       5,     6,     7,     8,     9,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    94,    95,    96,
      97,    98,    99,   104,   105,   106,   107,   108,   109,   110
};

#if YYDEBUG
  /* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   107,   107,   109,   113,   113,   117,   117,   121,   121,
     125,   125,   129,   129,   133,   135,   137,   142,   143,   147,
     151,   151,   155,   155,   159,   160,   164,   165,   170,   171,
     175,   176,   180,   184,   185,   186,   187,   188,   189,   191,
     193,   193,   198,   199,   203,   204,   208,   209,   211,   213,
     215,   216,   221,   222,   223,   224,   228,   229,   231,   233,
     235,   237,   238,   239,   240,   241,   242,   243,   244,   249,
     250,   251,   252,   253,   254,   258,   259,   263,   264,   268,
     269,   270,   274,   274,   278,   278,   278,   278,   282,   282,
     286,   288,   292,   292,   296,   296,   300,   301,   302,   303,
     304,   305,   306,   307,   311,   311,   315,   316,   317,   319,
     320,   321,   322,   323,   324,   325,   328,   329,   330,   331,
     332,   336,   337,   341,   341,   345,   346,   347,   348,   349,
     350,   351,   352,   353,   354,   355,   356,   357,   358,   359,
     360,   361,   362,   363,   364,   365,   366,   367,   369,   372,
     373,   375,   380,   381,   383,   385,   387,   388,   389,   391,
     396,   398,   403,   405,   407,   408,   412,   413,   414,   415,
     416,   420,   421,   422,   426,   427,   428,   433
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || 0
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "FIRSTTOKEN", "PROGRAM", "PASTAT",
  "PASTAT2", "XBEGIN", "XEND", "NL", "','", "'{'", "'('", "'|'", "';'",
  "'/'", "')'", "'}'", "'['", "']'", "ARRAY", "MATCH", "NOTMATCH",
  "MATCHOP", "FINAL", "DOT", "ALL", "CCL", "NCCL", "CHAR", "MCHAR", "OR",
  "STAR", "QUEST", "PLUS", "AND", "BOR", "APPEND", "EQ", "GE", "GT", "LE",
  "LT", "NE", "IN", "ARG", "BLTIN", "BREAK", "CONTINUE", "DELETE", "DO",
  "EXIT", "FOR", "FUNC", "SUB", "GSUB", "IF", "INDEX", "LSUBSTR",
  "MATCHFCN", "NEXT", "ADD", "MINUS", "MULT", "DIVIDE", "MOD", "ASSIGN",
  "ASGNOP", "ADDEQ", "SUBEQ", "MULTEQ", "DIVEQ", "MODEQ", "POWEQ", "PRINT",
  "PRINTF", "SPRINTF", "ELSE", "INTEST", "CONDEXPR", "POSTINCR", "PREINCR",
  "POSTDECR", "PREDECR", "VAR", "IVAR", "VARNF", "CALL", "NUMBER",
  "STRING", "FIELD", "REGEXPR", "'?'", "':'", "GETLINE", "RETURN", "SPLIT",
  "SUBSTR", "WHILE", "CAT", "'+'", "'-'", "'*'", "'%'", "NOT", "UMINUS",
  "POWER", "DECR", "INCR", "INDIRECT", "LASTTOKEN", "$accept", "program",
  "and", "bor", "comma", "do", "else", "for", "funcname", "if", "lbrace",
  "nl", "opt_nl", "opt_pst", "opt_simple_stmt", "pas", "pa_pat", "pa_stat",
  "$@1", "pa_stats", "patlist", "ppattern", "pattern", "plist", "pplist",
  "prarg", "print", "pst", "rbrace", "re", "reg_expr", "$@2", "rparen",
  "simple_stmt", "st", "stmt", "stmtlist", "subop", "term", "var",
  "varlist", "varname", "while", YY_NULLPTR
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[NUM] -- (External) token number corresponding to the
   (internal) symbol number NUM (which must be that of a token).  */
static const yytype_uint16 yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
      44,   123,    40,   124,    59,    47,    41,   125,    91,    93,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   294,
     295,   296,   297,   298,   299,   300,   301,   302,   303,   304,
     305,   306,   307,   308,   309,   310,   311,   312,   313,   314,
     315,   316,   317,   318,   319,   320,   321,   322,   323,   324,
     325,   326,   327,   328,   329,   330,   331,   332,   333,   334,
     335,   336,    63,    58,   337,   338,   339,   340,   341,   342,
      43,    45,    42,    37,   343,   344,   345,   346,   347,   348,
     349
};
# endif

#define YYPACT_NINF -283

#define yypact_value_is_default(Yystate) \
  (!!((Yystate) == (-283)))

#define YYTABLE_NINF -134

#define yytable_value_is_error(Yytable_value) \
  (!!((Yytable_value) == (-134)))

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const yytype_int16 yypact[] =
{
     662,  -283,  -283,  -283,    29,  1631,  -283,    76,  -283,    28,
      28,  -283,  4383,  -283,  -283,    25,   -10,  -283,  -283,    37,
      54,    77,  -283,  -283,  -283,    89,  -283,  -283,  -283,   300,
      93,   106,  4440,  4440,  4383,   234,   234,  4440,   728,   191,
    -283,   164,  3539,  -283,  -283,   115,    36,   -47,    23,  -283,
    -283,   728,   728,  2231,    43,    50,  4144,  -283,  -283,   122,
    4383,  4383,  4383,  4201,  4440,   108,  4383,  4383,  4383,  4440,
      49,    98,    49,   -21,  -283,    56,  -283,  -283,  -283,  -283,
    -283,   168,   167,   167,   -22,  -283,  1789,   172,   175,   167,
    -283,  -283,  1789,   179,   796,  -283,  1460,   728,  3539,  4497,
     167,  -283,   862,  1460,  4383,   728,  1631,    85,  4383,  -283,
    -283,  4383,  4383,  4383,  4383,  4383,  4383,   -22,  4383,  1846,
    1903,    36,  4383,  4440,  4440,  4440,  4440,  4440,  4440,  4383,
    -283,  -283,  4383,   928,   994,  -283,  -283,  1960,   149,  1960,
     181,  -283,    65,  3539,   119,  2631,  2631,    66,  -283,    68,
      36,  4440,  2631,  2631,  2721,    49,  -283,   198,  -283,   168,
     198,  -283,  -283,   190,  1732,  -283,  1517,  4383,  -283,  1732,
    -283,  4383,  -283,   120,   137,  1060,  4383,  3951,   207,    17,
      36,   -40,  -283,  -283,  -283,  -283,    28,  1126,  -283,   234,
    3789,    83,  3789,  3789,  3789,  3789,  3789,  3789,  -283,  2811,
    -283,  3709,  -283,  3629,  2631,   207,    49,     6,     6,    49,
      49,    49,  3539,    26,  -283,  -283,  -283,  3539,   -22,  3539,
    -283,  -283,  1960,  -283,    94,  1960,  1960,  -283,  -283,    36,
      47,  1960,  -283,  -283,  4383,  -283,   206,  -283,    14,  2901,
    -283,  2901,   212,  -283,  1194,  -283,   221,   103,  4554,   -22,
    4554,  2017,  2074,    36,  2131,  4440,  4440,  4440,  4554,   728,
    -283,  -283,  4383,  1960,  1960,  -283,  -283,  3539,  -283,     7,
     222,  2991,   216,  3081,   217,   116,  2331,    33,  4258,   -22,
     222,   222,  4383,  -283,  -283,  -283,   192,  4383,  4326,    83,
    -283,  3868,  4087,  4019,  3951,    36,    36,    36,  3951,  1260,
    3539,  2431,  2531,  -283,  -283,    28,  -283,  -283,  -283,  -283,
    -283,  1960,  -283,  1960,  -283,  1574,  3179,   218,  3269,   -22,
     141,  4554,  -283,  -283,   332,  -283,   332,   728,  3359,   219,
    3449,   218,  1574,  1328,   167,  -283,   192,  3951,   223,   224,
    1394,  -283,  -283,  -283,  1328,   218,  -283,  -283,  -283,  -283,
    -283,  -283,  1328,  -283
};

  /* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
     Performed when YYTABLE does not specify something else to do.  Zero
     means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
       0,     3,    84,    85,     0,    30,     2,    27,     1,     0,
       0,    20,     0,    92,   175,   136,     0,   123,   124,     0,
       0,     0,   174,   169,   176,     0,   152,   157,   168,   146,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    33,
      42,    26,    32,   165,    90,     0,    74,   164,   166,    86,
      87,     0,     0,     0,     0,     0,     0,    17,    18,     0,
       0,     0,     0,     0,     0,   145,     0,     0,     0,     0,
     132,   164,   131,    60,    91,    74,   139,   140,   170,   103,
      21,    24,     0,     0,     0,    10,     0,     0,     0,     0,
      82,    83,     0,     0,     0,   111,     0,     0,   102,    79,
       0,   121,     0,     0,     0,     0,    31,     0,     0,     4,
       6,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    73,     0,     0,     0,     0,     0,     0,     0,     0,
     141,   142,     0,     0,     0,     8,   149,     0,     0,     0,
       0,   134,     0,    44,   171,     0,     0,     0,   137,     0,
     144,     0,     0,     0,     0,   133,    22,    25,   120,    24,
     104,   106,   107,   101,     0,   110,     0,     0,   115,     0,
     117,     0,    11,     0,   113,     0,     0,    77,    80,    99,
      55,   164,   118,    37,   122,   119,    35,     0,    43,    72,
      68,    67,    61,    62,    63,    64,    65,    66,    69,     0,
       5,    59,     7,    58,     0,    90,   128,   125,   126,   127,
     129,   130,    56,     0,    38,    39,     9,    75,     0,    76,
      93,   135,     0,   172,     0,     0,     0,   156,   138,   143,
       0,     0,    23,   105,     0,   109,     0,    29,   166,     0,
     116,     0,     0,    12,     0,    88,   114,     0,     0,     0,
       0,     0,     0,    54,     0,     0,     0,     0,     0,     0,
      34,    71,     0,     0,     0,   167,    70,    45,    94,     0,
      40,     0,    90,     0,    90,     0,     0,     0,     0,     0,
      19,   177,     0,    13,   112,    89,    81,     0,    51,    50,
      52,     0,    49,    48,    78,    96,    97,    98,    46,     0,
      57,     0,     0,   173,    95,     0,   147,   148,   151,   150,
     155,     0,   163,     0,   100,     0,     0,     0,     0,     0,
       0,     0,    36,   159,     0,   158,     0,     0,     0,    90,
       0,     0,     0,     0,     0,    53,     0,    47,     0,     0,
       0,   153,   154,   162,     0,     0,    16,   108,   161,   160,
      41,    15,     0,    14
};

  /* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -283,  -283,   -94,    -8,   201,  -283,  -283,  -283,  -283,  -283,
       9,   -59,    78,   204,  -282,  -283,   142,   144,  -283,  -283,
     -52,  -105,   244,  -167,  -283,  -283,  -283,  -283,  -283,     4,
     -96,  -283,  -224,  -161,   -58,   -31,   -50,  -283,   366,   -29,
    -283,   -38,  -283
};

  /* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     4,   119,   120,   222,    94,   244,    95,    59,    96,
      97,   160,   158,     5,   236,     6,    39,    40,   305,    41,
     142,   177,    98,    54,   178,   179,    99,     7,   246,    43,
      44,    55,   270,   100,   161,   101,   102,    45,    46,    47,
     224,    48,   103
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_int16 yytable[] =
{
      65,   133,   134,    71,    71,   237,    76,    77,    71,   247,
     147,   149,   191,    71,    38,   280,   216,   281,    51,    52,
     129,   123,   157,    14,    71,   162,   205,   258,   165,     8,
     255,   168,   132,   331,   170,    71,   135,    56,    74,    11,
      71,   132,   182,   135,    71,   265,   163,   175,   105,    60,
     345,   123,   314,   135,   256,   187,   216,   257,   279,   138,
     130,   131,    22,   173,    24,   174,    61,   130,   131,    71,
     181,   184,   185,    74,    57,   135,   135,    58,   135,   198,
     213,   221,   227,   251,   228,    49,    35,    36,    37,    62,
      50,   303,    14,   333,    71,    71,    71,    71,    71,    71,
     157,    63,   184,   184,   135,    66,   235,   344,   126,   127,
     268,   240,   128,   135,    71,    38,    71,    71,    67,   286,
     320,   352,    71,    71,    71,    71,   135,   122,   238,   272,
     274,    22,   310,    24,   144,    71,   124,   125,   126,   127,
      71,   140,   128,   288,   184,   291,   292,   293,    71,   294,
     151,   135,   289,   298,   237,   128,   184,   336,  -133,  -133,
     261,    71,   128,    71,    71,    71,    71,    71,    71,   252,
      71,   237,    71,     2,    71,    71,   156,   156,     3,   189,
     266,   159,   277,    71,   166,   -90,   -90,   167,    71,   -90,
      71,   171,   275,   218,   251,   259,   220,   251,   251,   251,
     251,   104,    11,   223,   251,   130,   131,   232,   234,   299,
      71,   290,    71,   284,   243,   329,   337,   135,   242,   181,
     278,   181,   181,   181,   282,   181,    71,    71,    71,   181,
     285,   304,   307,   309,   268,   342,   319,   233,    71,   348,
     349,   317,    71,   251,    71,   106,   186,    71,     0,    42,
     188,     0,     0,     0,   137,   139,    53,     0,     0,    71,
       0,     0,    71,    71,    71,    71,     0,     0,   184,    71,
       0,    71,    71,    71,     0,     0,   347,   340,    73,    14,
     252,   335,     0,   252,   252,   252,   252,    71,     0,    71,
     252,     0,   181,     0,     0,   338,     0,   339,     0,    71,
     143,    71,   346,     0,   145,   146,   143,   143,    71,   184,
     152,   153,   154,   351,   327,     0,     0,     0,    22,    23,
      24,   353,     0,     0,    28,     0,     0,     0,     0,   252,
     164,     0,     0,     0,     0,     0,   169,     0,     0,     0,
       0,   216,    64,    37,     0,    14,   225,   226,    42,     0,
      42,     0,   190,   230,   231,   192,   193,   194,   195,   196,
     197,     0,   199,   201,   203,     0,   204,     0,     0,     0,
       0,     0,     0,   212,     0,     0,   143,    14,     0,   254,
       0,   217,     0,   219,    22,    23,    24,     0,     0,     0,
      28,     0,     0,     0,     0,     0,     0,     0,    70,    72,
      75,     0,     0,    78,     0,   263,   264,     0,   121,    37,
       0,   239,     0,     0,     0,   241,    22,    23,    24,   121,
      53,     0,    28,     0,     0,   269,     0,     0,     0,     0,
     150,     0,     0,     0,     0,   155,     0,     0,     0,   121,
       0,    37,     0,     0,     0,     0,     0,     0,   139,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   121,   180,   267,     0,     0,   271,
     273,     0,     0,     0,     0,   276,   311,   313,   143,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   206,
     207,   208,   209,   210,   211,     0,     0,     0,     0,     0,
       0,     0,   324,   326,     0,     0,   300,   301,   302,   121,
       0,   121,   121,     0,     0,     0,     0,   229,   121,   121,
     121,   139,   316,     0,     0,     0,   318,     0,     0,     0,
     121,    53,     0,     0,     0,   121,     0,     0,     0,     0,
       0,     0,     0,   253,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   328,   121,   330,   121,   121,
     121,   121,   121,   121,     0,   121,     0,   121,     0,   121,
     121,     0,     0,     0,     0,     0,     0,     0,   121,     0,
       0,     0,     0,   121,     0,   121,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   121,     0,   121,     0,     0,
       0,     0,     0,     0,   180,     0,   180,   180,   180,     0,
     180,   295,   296,   297,   180,     0,     0,     0,     0,     0,
       0,     0,     0,   121,     0,     0,     0,   121,     0,   121,
       0,     0,   121,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   253,     0,     0,   253,   253,   253,
     253,     0,   -26,     1,   253,     0,   121,   121,   121,   -26,
     -26,     2,     0,   -26,   -26,     0,     3,   -26,     0,     0,
       0,     0,   121,     0,   121,     0,     0,   180,     0,     0,
       0,     0,     0,     0,   121,     0,   121,     0,     0,     0,
       0,     0,     0,   253,     0,     0,     0,   -26,   -26,     0,
       0,     0,     0,     0,     0,   -26,   -26,   -26,     0,   -26,
       0,   -26,     0,     0,     0,     0,     0,     0,     0,    79,
       0,     0,     0,     0,     0,     0,     0,    80,   -26,    11,
      12,     0,    81,    13,     0,     0,   -26,   -26,   -26,   -26,
     -26,   -26,   -26,     0,     0,     0,   -26,     0,   -26,   -26,
       0,     0,   -26,   -26,     0,     0,   -26,     0,     0,   -26,
     -26,   -26,     0,    14,    15,    82,    83,    84,    85,    86,
      87,     0,    17,    18,    88,    19,     0,    20,    89,     0,
       0,     0,     0,     0,     0,     0,     0,    79,     0,     0,
       0,     0,    90,    91,    21,   172,     0,    11,    12,     0,
      81,    13,    22,    23,    24,    25,    26,    27,    28,     0,
       0,     0,    29,    92,    30,    31,    93,     0,    32,    33,
       0,     0,    34,     0,     0,    35,    36,    37,     0,     0,
       0,    14,    15,    82,    83,    84,    85,    86,    87,     0,
      17,    18,    88,    19,     0,    20,    89,     0,     0,     0,
       0,     0,     0,    79,     0,     0,     0,     0,     0,     0,
      90,    91,    21,    11,    12,     0,    81,    13,     0,   183,
      22,    23,    24,    25,    26,    27,    28,     0,     0,     0,
      29,    92,    30,    31,    93,     0,    32,    33,     0,     0,
      34,     0,     0,    35,    36,    37,     0,    14,    15,    82,
      83,    84,    85,    86,    87,     0,    17,    18,    88,    19,
       0,    20,    89,     0,     0,     0,     0,     0,     0,    79,
       0,     0,     0,     0,     0,     0,    90,    91,    21,    11,
      12,     0,    81,    13,     0,   214,    22,    23,    24,    25,
      26,    27,    28,     0,     0,     0,    29,    92,    30,    31,
      93,     0,    32,    33,     0,     0,    34,     0,     0,    35,
      36,    37,     0,    14,    15,    82,    83,    84,    85,    86,
      87,     0,    17,    18,    88,    19,     0,    20,    89,     0,
       0,     0,     0,     0,     0,    79,     0,     0,     0,     0,
       0,     0,    90,    91,    21,    11,    12,     0,    81,    13,
       0,   215,    22,    23,    24,    25,    26,    27,    28,     0,
       0,     0,    29,    92,    30,    31,    93,     0,    32,    33,
       0,     0,    34,     0,     0,    35,    36,    37,     0,    14,
      15,    82,    83,    84,    85,    86,    87,     0,    17,    18,
      88,    19,     0,    20,    89,     0,     0,     0,     0,     0,
       0,    79,     0,     0,     0,     0,     0,     0,    90,    91,
      21,    11,    12,     0,    81,    13,     0,   245,    22,    23,
      24,    25,    26,    27,    28,     0,     0,     0,    29,    92,
      30,    31,    93,     0,    32,    33,     0,     0,    34,     0,
       0,    35,    36,    37,     0,    14,    15,    82,    83,    84,
      85,    86,    87,     0,    17,    18,    88,    19,     0,    20,
      89,     0,     0,     0,     0,     0,     0,    79,     0,     0,
       0,     0,     0,     0,    90,    91,    21,    11,    12,     0,
      81,    13,     0,   260,    22,    23,    24,    25,    26,    27,
      28,     0,     0,     0,    29,    92,    30,    31,    93,     0,
      32,    33,     0,     0,    34,     0,     0,    35,    36,    37,
       0,    14,    15,    82,    83,    84,    85,    86,    87,     0,
      17,    18,    88,    19,     0,    20,    89,     0,     0,     0,
       0,     0,     0,     0,     0,    79,     0,     0,     0,     0,
      90,    91,    21,   283,     0,    11,    12,     0,    81,    13,
      22,    23,    24,    25,    26,    27,    28,     0,     0,     0,
      29,    92,    30,    31,    93,     0,    32,    33,     0,     0,
      34,     0,     0,    35,    36,    37,     0,     0,     0,    14,
      15,    82,    83,    84,    85,    86,    87,     0,    17,    18,
      88,    19,     0,    20,    89,     0,     0,     0,     0,     0,
       0,    79,     0,     0,     0,     0,     0,     0,    90,    91,
      21,    11,    12,     0,    81,    13,     0,   322,    22,    23,
      24,    25,    26,    27,    28,     0,     0,     0,    29,    92,
      30,    31,    93,     0,    32,    33,     0,     0,    34,     0,
       0,    35,    36,    37,     0,    14,    15,    82,    83,    84,
      85,    86,    87,     0,    17,    18,    88,    19,     0,    20,
      89,     0,     0,     0,     0,     0,     0,     0,     0,    79,
       0,     0,     0,     0,    90,    91,    21,   304,     0,    11,
      12,     0,    81,    13,    22,    23,    24,    25,    26,    27,
      28,     0,     0,     0,    29,    92,    30,    31,    93,     0,
      32,    33,     0,     0,    34,     0,     0,    35,    36,    37,
       0,     0,     0,    14,    15,    82,    83,    84,    85,    86,
      87,     0,    17,    18,    88,    19,     0,    20,    89,     0,
       0,     0,     0,     0,     0,    79,     0,     0,     0,     0,
       0,     0,    90,    91,    21,    11,    12,     0,    81,    13,
       0,   350,    22,    23,    24,    25,    26,    27,    28,     0,
       0,     0,    29,    92,    30,    31,    93,     0,    32,    33,
       0,     0,    34,     0,     0,    35,    36,    37,     0,    14,
      15,    82,    83,    84,    85,    86,    87,     0,    17,    18,
      88,    19,     0,    20,    89,     0,     0,     0,     0,     0,
       0,    79,     0,     0,     0,     0,     0,     0,    90,    91,
      21,    11,    12,     0,    81,    13,     0,     0,    22,    23,
      24,    25,    26,    27,    28,     0,     0,     0,    29,    92,
      30,    31,    93,     0,    32,    33,     0,     0,    34,     0,
       0,    35,    36,    37,     0,    14,    15,    82,    83,    84,
      85,    86,    87,     0,    17,    18,    88,    19,    79,    20,
      89,     0,     0,     0,     0,     0,     0,     0,     0,    12,
       0,   -28,    13,     0,    90,    91,    21,     0,     0,     0,
       0,     0,     0,     0,    22,    23,    24,    25,    26,    27,
      28,     0,     0,     0,    29,    92,    30,    31,    93,     0,
      32,    33,    14,    15,    34,     0,    84,    35,    36,    37,
       0,    17,    18,     0,    19,    79,    20,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    12,     0,     0,    13,
     -28,    90,    91,    21,     0,     0,     0,     0,     0,     0,
       0,    22,    23,    24,    25,    26,    27,    28,     0,     0,
       0,    29,     0,    30,    31,     0,     0,    32,    33,    14,
      15,    34,     0,    84,    35,    36,    37,     0,    17,    18,
       0,    19,     0,    20,     0,     0,     0,     0,     9,    10,
       0,     0,    11,    12,     0,     0,    13,     0,    90,    91,
      21,     0,     0,     0,     0,     0,     0,     0,    22,    23,
      24,    25,    26,    27,    28,     0,     0,     0,    29,     0,
      30,    31,     0,     0,    32,    33,    14,    15,    34,     0,
       0,    35,    36,    37,    16,    17,    18,     0,    19,     0,
      20,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    21,     0,     0,
       0,     0,     0,     0,     0,    22,    23,    24,    25,    26,
      27,    28,     0,     0,     0,    29,     0,    30,    31,     0,
       0,    32,    33,     0,     0,    34,     0,     0,    35,    36,
      37,   156,     0,     0,    68,   107,   159,    13,     0,     0,
       0,     0,     0,     0,     0,   108,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   109,   110,     0,
     111,   112,   113,   114,   115,   116,   117,    14,    15,     0,
       0,     0,     0,     0,     0,     0,    17,    18,     0,    19,
       0,    20,     0,     0,     0,     0,     0,     0,   156,     0,
       0,    12,     0,   159,    13,     0,     0,     0,    21,     0,
       0,     0,     0,     0,     0,     0,    22,    23,    24,    25,
      26,    27,    28,     0,   118,     0,    29,     0,    30,    31,
       0,     0,    32,    33,    14,    15,    69,     0,     0,    35,
      36,    37,     0,    17,    18,     0,    19,     0,    20,     0,
       0,     0,     0,     0,     0,   200,     0,     0,    12,     0,
       0,    13,     0,     0,     0,    21,     0,     0,     0,     0,
       0,     0,     0,    22,    23,    24,    25,    26,    27,    28,
       0,     0,     0,    29,     0,    30,    31,     0,     0,    32,
      33,    14,    15,    34,     0,     0,    35,    36,    37,     0,
      17,    18,     0,    19,     0,    20,     0,     0,     0,     0,
       0,     0,   202,     0,     0,    12,     0,     0,    13,     0,
       0,     0,    21,     0,     0,     0,     0,     0,     0,     0,
      22,    23,    24,    25,    26,    27,    28,     0,     0,     0,
      29,     0,    30,    31,     0,     0,    32,    33,    14,    15,
      34,     0,     0,    35,    36,    37,     0,    17,    18,     0,
      19,     0,    20,     0,     0,     0,     0,     0,     0,   216,
       0,     0,    12,     0,     0,    13,     0,     0,     0,    21,
       0,     0,     0,     0,     0,     0,     0,    22,    23,    24,
      25,    26,    27,    28,     0,     0,     0,    29,     0,    30,
      31,     0,     0,    32,    33,    14,    15,    34,     0,     0,
      35,    36,    37,     0,    17,    18,     0,    19,     0,    20,
       0,     0,     0,     0,     0,     0,   200,     0,     0,   287,
       0,     0,    13,     0,     0,     0,    21,     0,     0,     0,
       0,     0,     0,     0,    22,    23,    24,    25,    26,    27,
      28,     0,     0,     0,    29,     0,    30,    31,     0,     0,
      32,    33,    14,    15,    34,     0,     0,    35,    36,    37,
       0,    17,    18,     0,    19,     0,    20,     0,     0,     0,
       0,     0,     0,   202,     0,     0,   287,     0,     0,    13,
       0,     0,     0,    21,     0,     0,     0,     0,     0,     0,
       0,    22,    23,    24,    25,    26,    27,    28,     0,     0,
       0,    29,     0,    30,    31,     0,     0,    32,    33,    14,
      15,    69,     0,     0,    35,    36,    37,     0,    17,    18,
       0,    19,     0,    20,     0,     0,     0,     0,     0,     0,
     216,     0,     0,   287,     0,     0,    13,     0,     0,     0,
      21,     0,     0,     0,     0,     0,     0,     0,    22,    23,
      24,    25,    26,    27,    28,     0,     0,     0,    29,     0,
      30,    31,     0,     0,    32,    33,    14,    15,    69,     0,
       0,    35,    36,    37,     0,    17,    18,     0,    19,     0,
      20,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    21,     0,     0,
       0,     0,     0,     0,     0,    22,    23,    24,    25,    26,
      27,    28,     0,     0,     0,    29,     0,    30,    31,     0,
       0,    32,    33,     0,     0,    69,     0,     0,    35,    36,
      37,   135,     0,    68,   107,     0,    13,   136,     0,     0,
       0,     0,     0,     0,   108,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   109,   110,     0,   111,
     112,   113,   114,   115,   116,   117,    14,    15,     0,     0,
       0,     0,     0,     0,     0,    17,    18,     0,    19,     0,
      20,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    21,     0,     0,
       0,     0,     0,     0,     0,    22,    23,    24,    25,    26,
      27,    28,     0,   118,     0,    29,     0,    30,    31,     0,
       0,    32,    33,     0,     0,    69,     0,     0,    35,    36,
      37,   135,     0,    68,   107,     0,    13,   312,     0,     0,
       0,     0,     0,     0,   108,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   109,   110,     0,   111,
     112,   113,   114,   115,   116,   117,    14,    15,     0,     0,
       0,     0,     0,     0,     0,    17,    18,     0,    19,     0,
      20,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    21,     0,     0,
       0,     0,     0,     0,     0,    22,    23,    24,    25,    26,
      27,    28,     0,   118,     0,    29,     0,    30,    31,     0,
       0,    32,    33,     0,     0,    69,     0,     0,    35,    36,
      37,   135,     0,    68,   107,     0,    13,   323,     0,     0,
       0,     0,     0,     0,   108,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   109,   110,     0,   111,
     112,   113,   114,   115,   116,   117,    14,    15,     0,     0,
       0,     0,     0,     0,     0,    17,    18,     0,    19,     0,
      20,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    21,     0,     0,
       0,     0,     0,     0,     0,    22,    23,    24,    25,    26,
      27,    28,     0,   118,     0,    29,     0,    30,    31,     0,
       0,    32,    33,     0,     0,    69,     0,     0,    35,    36,
      37,   135,     0,    68,   107,     0,    13,   325,     0,     0,
       0,     0,     0,     0,   108,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   109,   110,     0,   111,
     112,   113,   114,   115,   116,   117,    14,    15,     0,     0,
       0,     0,     0,     0,     0,    17,    18,     0,    19,     0,
      20,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    21,     0,     0,
       0,     0,     0,     0,     0,    22,    23,    24,    25,    26,
      27,    28,     0,   118,     0,    29,     0,    30,    31,     0,
       0,    32,    33,     0,     0,    69,     0,     0,    35,    36,
      37,   135,     0,    68,   107,     0,    13,     0,     0,     0,
       0,     0,     0,     0,   108,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   109,   110,     0,   111,
     112,   113,   114,   115,   116,   117,    14,    15,     0,     0,
       0,     0,     0,     0,     0,    17,    18,     0,    19,     0,
      20,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    21,     0,     0,
       0,     0,     0,     0,     0,    22,    23,    24,    25,    26,
      27,    28,     0,   118,     0,    29,     0,    30,    31,     0,
       0,    32,    33,    68,   107,    69,    13,   136,    35,    36,
      37,     0,     0,     0,   108,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   109,   110,     0,   111,
     112,   113,   114,   115,   116,   117,    14,    15,     0,     0,
       0,     0,     0,     0,     0,    17,    18,     0,    19,     0,
      20,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    21,     0,     0,
       0,     0,     0,     0,     0,    22,    23,    24,    25,    26,
      27,    28,     0,   118,     0,    29,     0,    30,    31,     0,
       0,    32,    33,    68,   107,    69,    13,     0,    35,    36,
      37,     0,     0,     0,   108,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   109,   110,     0,   111,
     112,   113,   114,   115,   116,   117,    14,    15,     0,     0,
       0,     0,     0,     0,     0,    17,    18,     0,    19,     0,
      20,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    21,     0,     0,
       0,     0,     0,     0,     0,    22,    23,    24,    25,    26,
      27,    28,     0,   118,   262,    29,     0,    30,    31,     0,
       0,    32,    33,    68,   107,    69,    13,   268,    35,    36,
      37,     0,     0,     0,   108,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   109,   110,     0,   111,
     112,   113,   114,   115,   116,   117,    14,    15,     0,     0,
       0,     0,     0,     0,     0,    17,    18,     0,    19,     0,
      20,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    21,     0,     0,
       0,     0,     0,     0,     0,    22,    23,    24,    25,    26,
      27,    28,     0,   118,     0,    29,     0,    30,    31,     0,
       0,    32,    33,    68,   107,    69,    13,   306,    35,    36,
      37,     0,     0,     0,   108,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   109,   110,     0,   111,
     112,   113,   114,   115,   116,   117,    14,    15,     0,     0,
       0,     0,     0,     0,     0,    17,    18,     0,    19,     0,
      20,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    21,     0,     0,
       0,     0,     0,     0,     0,    22,    23,    24,    25,    26,
      27,    28,     0,   118,     0,    29,     0,    30,    31,     0,
       0,    32,    33,    68,   107,    69,    13,   308,    35,    36,
      37,     0,     0,     0,   108,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   109,   110,     0,   111,
     112,   113,   114,   115,   116,   117,    14,    15,     0,     0,
       0,     0,     0,     0,     0,    17,    18,     0,    19,     0,
      20,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    21,     0,     0,
       0,     0,     0,     0,     0,    22,    23,    24,    25,    26,
      27,    28,     0,   118,     0,    29,     0,    30,    31,     0,
       0,    32,    33,     0,     0,    69,     0,     0,    35,    36,
      37,    68,   107,   332,    13,     0,     0,     0,     0,     0,
       0,     0,   108,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   109,   110,     0,   111,   112,   113,
     114,   115,   116,   117,    14,    15,     0,     0,     0,     0,
       0,     0,     0,    17,    18,     0,    19,     0,    20,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    21,     0,     0,     0,     0,
       0,     0,     0,    22,    23,    24,    25,    26,    27,    28,
       0,   118,     0,    29,     0,    30,    31,     0,     0,    32,
      33,    68,   107,    69,    13,   334,    35,    36,    37,     0,
       0,     0,   108,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   109,   110,     0,   111,   112,   113,
     114,   115,   116,   117,    14,    15,     0,     0,     0,     0,
       0,     0,     0,    17,    18,     0,    19,     0,    20,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    21,     0,     0,     0,     0,
       0,     0,     0,    22,    23,    24,    25,    26,    27,    28,
       0,   118,     0,    29,     0,    30,    31,     0,     0,    32,
      33,    68,   107,    69,    13,   341,    35,    36,    37,     0,
       0,     0,   108,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   109,   110,     0,   111,   112,   113,
     114,   115,   116,   117,    14,    15,     0,     0,     0,     0,
       0,     0,     0,    17,    18,     0,    19,     0,    20,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    21,     0,     0,     0,     0,
       0,     0,     0,    22,    23,    24,    25,    26,    27,    28,
       0,   118,     0,    29,     0,    30,    31,     0,     0,    32,
      33,    68,   107,    69,    13,   343,    35,    36,    37,     0,
       0,     0,   108,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   109,   110,     0,   111,   112,   113,
     114,   115,   116,   117,    14,    15,     0,     0,     0,     0,
       0,     0,     0,    17,    18,     0,    19,     0,    20,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    21,     0,     0,     0,     0,
       0,     0,     0,    22,    23,    24,    25,    26,    27,    28,
       0,   118,     0,    29,     0,    30,    31,     0,     0,    32,
      33,    68,   107,    69,    13,     0,    35,    36,    37,     0,
       0,     0,   108,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   109,   110,     0,   111,   112,   113,
     114,   115,   116,   117,    14,    15,     0,     0,     0,     0,
       0,     0,     0,    17,    18,     0,    19,     0,    20,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    21,     0,     0,     0,     0,
       0,     0,     0,    22,    23,    24,    25,    26,    27,    28,
       0,   118,     0,    29,     0,    30,    31,     0,     0,    32,
      33,    68,   107,    69,    13,     0,    35,    36,    37,     0,
       0,     0,   108,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   109,     0,     0,   111,   112,   113,
     114,   115,   116,   117,    14,    15,     0,     0,     0,     0,
       0,     0,     0,    17,    18,     0,    19,     0,    20,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    21,     0,     0,     0,     0,
       0,     0,     0,    22,    23,    24,    25,    26,    27,    28,
       0,    68,   107,    29,    13,    30,    31,     0,     0,    32,
      33,     0,   108,    69,     0,     0,    35,    36,    37,     0,
       0,     0,     0,     0,     0,     0,     0,   111,   112,   113,
     114,   115,   116,   117,    14,    15,     0,     0,     0,     0,
       0,     0,     0,    17,    18,     0,    19,     0,    20,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    21,     0,     0,     0,     0,
       0,     0,     0,    22,    23,    24,    25,    26,    27,    28,
       0,    68,  -134,    29,    13,    30,    31,     0,     0,    32,
      33,     0,  -134,    69,     0,     0,    35,    36,    37,     0,
       0,     0,     0,     0,     0,     0,     0,  -134,  -134,  -134,
    -134,  -134,  -134,  -134,    14,    15,     0,     0,     0,     0,
       0,     0,     0,    17,    18,     0,    19,     0,    20,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    21,     0,     0,     0,     0,
       0,     0,     0,    22,    23,    24,    25,    26,    27,    28,
      68,     0,     0,    13,     0,    30,    31,     0,     0,    32,
      33,   248,     0,    69,     0,     0,    35,    36,    37,     0,
       0,     0,     0,   109,   110,     0,     0,     0,     0,     0,
       0,     0,   249,    14,    15,     0,     0,     0,     0,     0,
       0,     0,    17,    18,     0,    19,     0,    20,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    21,     0,     0,     0,     0,     0,
       0,     0,    22,    23,    24,    25,    26,    27,    28,     0,
     250,   321,    29,    68,    30,    31,    13,     0,    32,    33,
       0,     0,    69,     0,   248,    35,    36,    37,     0,     0,
       0,     0,     0,     0,     0,     0,   109,   110,     0,     0,
       0,     0,     0,     0,     0,   249,    14,    15,     0,     0,
       0,     0,     0,     0,     0,    17,    18,     0,    19,     0,
      20,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    21,     0,     0,
       0,    68,     0,     0,    13,    22,    23,    24,    25,    26,
      27,    28,   248,   250,     0,    29,     0,    30,    31,     0,
       0,    32,    33,     0,   109,    69,     0,     0,    35,    36,
      37,     0,     0,   249,    14,    15,     0,     0,     0,     0,
       0,     0,     0,    17,    18,     0,    19,     0,    20,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    21,     0,     0,     0,    68,
       0,     0,    13,    22,    23,    24,    25,    26,    27,    28,
     248,     0,     0,    29,     0,    30,    31,     0,     0,    32,
      33,     0,     0,    69,     0,     0,    35,    36,    37,     0,
       0,   249,    14,    15,     0,     0,     0,     0,     0,     0,
       0,    17,    18,     0,    19,     0,    20,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    12,     0,     0,    13,
     141,     0,     0,    21,     0,     0,     0,     0,     0,     0,
       0,    22,    23,    24,    25,    26,    27,    28,     0,     0,
       0,    29,     0,    30,    31,     0,     0,    32,    33,    14,
      15,    69,     0,     0,    35,    36,    37,     0,    17,    18,
       0,    19,     0,    20,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    12,     0,     0,    13,   148,     0,     0,
      21,     0,     0,     0,     0,     0,     0,     0,    22,    23,
      24,    25,    26,    27,    28,     0,     0,     0,    29,     0,
      30,    31,     0,     0,    32,    33,    14,    15,    34,     0,
       0,    35,    36,    37,     0,    17,    18,     0,    19,     0,
      20,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      12,     0,   315,    13,     0,     0,     0,    21,     0,     0,
       0,     0,     0,     0,     0,    22,    23,    24,    25,    26,
      27,    28,     0,     0,     0,    29,     0,    30,    31,     0,
       0,    32,    33,    14,    15,    34,     0,     0,    35,    36,
      37,     0,    17,    18,     0,    19,     0,    20,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    21,     0,     0,     0,    68,     0,
       0,    13,    22,    23,    24,    25,    26,    27,    28,  -134,
       0,     0,    29,     0,    30,    31,     0,     0,    32,    33,
       0,     0,    34,     0,     0,    35,    36,    37,     0,     0,
    -134,    14,    15,     0,     0,     0,     0,     0,     0,     0,
      17,    18,     0,    19,     0,    20,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    12,     0,     0,    13,     0,
       0,     0,    21,     0,     0,     0,     0,     0,     0,     0,
      22,    23,    24,    25,    26,    27,    28,     0,     0,     0,
       0,     0,    30,    31,     0,     0,    32,    33,    14,    15,
      69,     0,     0,    35,    36,    37,     0,    17,    18,     0,
      19,     0,    20,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    68,     0,     0,    13,     0,     0,     0,    21,
       0,     0,     0,     0,     0,     0,     0,    22,    23,    24,
      25,    26,    27,    28,     0,     0,     0,    29,     0,    30,
      31,     0,     0,    32,    33,    14,    15,    34,     0,     0,
      35,    36,    37,     0,    17,    18,     0,    19,     0,    20,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   176,
       0,     0,    13,     0,     0,     0,    21,     0,     0,     0,
       0,     0,     0,     0,    22,    23,    24,    25,    26,    27,
      28,     0,     0,     0,    29,     0,    30,    31,     0,     0,
      32,    33,    14,    15,    69,     0,     0,    35,    36,    37,
       0,    17,    18,     0,    19,     0,    20,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   287,     0,     0,    13,
       0,     0,     0,    21,     0,     0,     0,     0,     0,     0,
       0,    22,    23,    24,    25,    26,    27,    28,     0,     0,
       0,    29,     0,    30,    31,     0,     0,    32,    33,    14,
      15,    69,     0,     0,    35,    36,    37,     0,    17,    18,
       0,    19,     0,    20,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      21,     0,     0,     0,     0,     0,     0,     0,    22,    23,
      24,    25,    26,    27,    28,     0,     0,     0,    29,     0,
      30,    31,     0,     0,    32,    33,     0,     0,    69,     0,
       0,    35,    36,    37
};

static const yytype_int16 yycheck[] =
{
      29,    51,    52,    32,    33,   166,    35,    36,    37,   176,
      62,    63,   108,    42,     5,   239,     9,   241,     9,    10,
      67,    15,    81,    45,    53,    83,   122,    67,    86,     0,
      13,    89,    18,   315,    92,    64,    10,    12,    34,    11,
      69,    18,   100,    10,    73,    19,    84,    97,    39,    12,
     332,    15,    19,    10,    37,   105,     9,    40,    44,    16,
     107,   108,    84,    94,    86,    96,    12,   107,   108,    98,
      99,   102,   103,    69,    84,    10,    10,    87,    10,   117,
     132,    16,    16,   177,    16,     9,   107,   108,   109,    12,
      14,    84,    45,   317,   123,   124,   125,   126,   127,   128,
     159,    12,   133,   134,    10,    12,   164,   331,   102,   103,
      16,   169,   106,    10,   143,   106,   145,   146,    12,    16,
     287,   345,   151,   152,   153,   154,    10,    12,   166,   225,
     226,    84,    16,    86,    12,   164,   100,   101,   102,   103,
     169,    91,   106,   248,   175,   250,   251,   252,   177,   254,
      42,    10,   248,   258,   315,   106,   187,    16,   102,   103,
     189,   190,   106,   192,   193,   194,   195,   196,   197,   177,
     199,   332,   201,     9,   203,   204,     9,     9,    14,    94,
     218,    14,   234,   212,    12,   102,   103,    12,   217,   106,
     219,    12,   230,    44,   288,   186,    15,   291,   292,   293,
     294,    10,    11,    84,   298,   107,   108,     9,    18,   259,
     239,   249,   241,   244,    77,   311,   321,    10,    98,   248,
      14,   250,   251,   252,    12,   254,   255,   256,   257,   258,
       9,     9,    16,    16,    16,    16,    44,   159,   267,    16,
      16,   279,   271,   337,   273,    41,   104,   276,    -1,     5,
     106,    -1,    -1,    -1,    53,    54,    12,    -1,    -1,   288,
      -1,    -1,   291,   292,   293,   294,    -1,    -1,   299,   298,
      -1,   300,   301,   302,    -1,    -1,   334,   327,    34,    45,
     288,   319,    -1,   291,   292,   293,   294,   316,    -1,   318,
     298,    -1,   321,    -1,    -1,   324,    -1,   326,    -1,   328,
      56,   330,   333,    -1,    60,    61,    62,    63,   337,   340,
      66,    67,    68,   344,   305,    -1,    -1,    -1,    84,    85,
      86,   352,    -1,    -1,    90,    -1,    -1,    -1,    -1,   337,
      86,    -1,    -1,    -1,    -1,    -1,    92,    -1,    -1,    -1,
      -1,     9,    42,   109,    -1,    45,   145,   146,   104,    -1,
     106,    -1,   108,   152,   153,   111,   112,   113,   114,   115,
     116,    -1,   118,   119,   120,    -1,   122,    -1,    -1,    -1,
      -1,    -1,    -1,   129,    -1,    -1,   132,    45,    -1,   178,
      -1,   137,    -1,   139,    84,    85,    86,    -1,    -1,    -1,
      90,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    32,    33,
      34,    -1,    -1,    37,    -1,   204,   205,    -1,    42,   109,
      -1,   167,    -1,    -1,    -1,   171,    84,    85,    86,    53,
     176,    -1,    90,    -1,    -1,   224,    -1,    -1,    -1,    -1,
      64,    -1,    -1,    -1,    -1,    69,    -1,    -1,    -1,    73,
      -1,   109,    -1,    -1,    -1,    -1,    -1,    -1,   247,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    98,    99,   222,    -1,    -1,   225,
     226,    -1,    -1,    -1,    -1,   231,   275,   276,   234,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   123,
     124,   125,   126,   127,   128,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,   301,   302,    -1,    -1,   262,   263,   264,   143,
      -1,   145,   146,    -1,    -1,    -1,    -1,   151,   152,   153,
     154,   320,   278,    -1,    -1,    -1,   282,    -1,    -1,    -1,
     164,   287,    -1,    -1,    -1,   169,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   177,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,   311,   190,   313,   192,   193,
     194,   195,   196,   197,    -1,   199,    -1,   201,    -1,   203,
     204,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   212,    -1,
      -1,    -1,    -1,   217,    -1,   219,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,   239,    -1,   241,    -1,    -1,
      -1,    -1,    -1,    -1,   248,    -1,   250,   251,   252,    -1,
     254,   255,   256,   257,   258,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   267,    -1,    -1,    -1,   271,    -1,   273,
      -1,    -1,   276,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,   288,    -1,    -1,   291,   292,   293,
     294,    -1,     0,     1,   298,    -1,   300,   301,   302,     7,
       8,     9,    -1,    11,    12,    -1,    14,    15,    -1,    -1,
      -1,    -1,   316,    -1,   318,    -1,    -1,   321,    -1,    -1,
      -1,    -1,    -1,    -1,   328,    -1,   330,    -1,    -1,    -1,
      -1,    -1,    -1,   337,    -1,    -1,    -1,    45,    46,    -1,
      -1,    -1,    -1,    -1,    -1,    53,    54,    55,    -1,    57,
      -1,    59,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,     9,    76,    11,
      12,    -1,    14,    15,    -1,    -1,    84,    85,    86,    87,
      88,    89,    90,    -1,    -1,    -1,    94,    -1,    96,    97,
      -1,    -1,   100,   101,    -1,    -1,   104,    -1,    -1,   107,
     108,   109,    -1,    45,    46,    47,    48,    49,    50,    51,
      52,    -1,    54,    55,    56,    57,    -1,    59,    60,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,     1,    -1,    -1,
      -1,    -1,    74,    75,    76,     9,    -1,    11,    12,    -1,
      14,    15,    84,    85,    86,    87,    88,    89,    90,    -1,
      -1,    -1,    94,    95,    96,    97,    98,    -1,   100,   101,
      -1,    -1,   104,    -1,    -1,   107,   108,   109,    -1,    -1,
      -1,    45,    46,    47,    48,    49,    50,    51,    52,    -1,
      54,    55,    56,    57,    -1,    59,    60,    -1,    -1,    -1,
      -1,    -1,    -1,     1,    -1,    -1,    -1,    -1,    -1,    -1,
      74,    75,    76,    11,    12,    -1,    14,    15,    -1,    17,
      84,    85,    86,    87,    88,    89,    90,    -1,    -1,    -1,
      94,    95,    96,    97,    98,    -1,   100,   101,    -1,    -1,
     104,    -1,    -1,   107,   108,   109,    -1,    45,    46,    47,
      48,    49,    50,    51,    52,    -1,    54,    55,    56,    57,
      -1,    59,    60,    -1,    -1,    -1,    -1,    -1,    -1,     1,
      -1,    -1,    -1,    -1,    -1,    -1,    74,    75,    76,    11,
      12,    -1,    14,    15,    -1,    17,    84,    85,    86,    87,
      88,    89,    90,    -1,    -1,    -1,    94,    95,    96,    97,
      98,    -1,   100,   101,    -1,    -1,   104,    -1,    -1,   107,
     108,   109,    -1,    45,    46,    47,    48,    49,    50,    51,
      52,    -1,    54,    55,    56,    57,    -1,    59,    60,    -1,
      -1,    -1,    -1,    -1,    -1,     1,    -1,    -1,    -1,    -1,
      -1,    -1,    74,    75,    76,    11,    12,    -1,    14,    15,
      -1,    17,    84,    85,    86,    87,    88,    89,    90,    -1,
      -1,    -1,    94,    95,    96,    97,    98,    -1,   100,   101,
      -1,    -1,   104,    -1,    -1,   107,   108,   109,    -1,    45,
      46,    47,    48,    49,    50,    51,    52,    -1,    54,    55,
      56,    57,    -1,    59,    60,    -1,    -1,    -1,    -1,    -1,
      -1,     1,    -1,    -1,    -1,    -1,    -1,    -1,    74,    75,
      76,    11,    12,    -1,    14,    15,    -1,    17,    84,    85,
      86,    87,    88,    89,    90,    -1,    -1,    -1,    94,    95,
      96,    97,    98,    -1,   100,   101,    -1,    -1,   104,    -1,
      -1,   107,   108,   109,    -1,    45,    46,    47,    48,    49,
      50,    51,    52,    -1,    54,    55,    56,    57,    -1,    59,
      60,    -1,    -1,    -1,    -1,    -1,    -1,     1,    -1,    -1,
      -1,    -1,    -1,    -1,    74,    75,    76,    11,    12,    -1,
      14,    15,    -1,    17,    84,    85,    86,    87,    88,    89,
      90,    -1,    -1,    -1,    94,    95,    96,    97,    98,    -1,
     100,   101,    -1,    -1,   104,    -1,    -1,   107,   108,   109,
      -1,    45,    46,    47,    48,    49,    50,    51,    52,    -1,
      54,    55,    56,    57,    -1,    59,    60,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,     1,    -1,    -1,    -1,    -1,
      74,    75,    76,     9,    -1,    11,    12,    -1,    14,    15,
      84,    85,    86,    87,    88,    89,    90,    -1,    -1,    -1,
      94,    95,    96,    97,    98,    -1,   100,   101,    -1,    -1,
     104,    -1,    -1,   107,   108,   109,    -1,    -1,    -1,    45,
      46,    47,    48,    49,    50,    51,    52,    -1,    54,    55,
      56,    57,    -1,    59,    60,    -1,    -1,    -1,    -1,    -1,
      -1,     1,    -1,    -1,    -1,    -1,    -1,    -1,    74,    75,
      76,    11,    12,    -1,    14,    15,    -1,    17,    84,    85,
      86,    87,    88,    89,    90,    -1,    -1,    -1,    94,    95,
      96,    97,    98,    -1,   100,   101,    -1,    -1,   104,    -1,
      -1,   107,   108,   109,    -1,    45,    46,    47,    48,    49,
      50,    51,    52,    -1,    54,    55,    56,    57,    -1,    59,
      60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     1,
      -1,    -1,    -1,    -1,    74,    75,    76,     9,    -1,    11,
      12,    -1,    14,    15,    84,    85,    86,    87,    88,    89,
      90,    -1,    -1,    -1,    94,    95,    96,    97,    98,    -1,
     100,   101,    -1,    -1,   104,    -1,    -1,   107,   108,   109,
      -1,    -1,    -1,    45,    46,    47,    48,    49,    50,    51,
      52,    -1,    54,    55,    56,    57,    -1,    59,    60,    -1,
      -1,    -1,    -1,    -1,    -1,     1,    -1,    -1,    -1,    -1,
      -1,    -1,    74,    75,    76,    11,    12,    -1,    14,    15,
      -1,    17,    84,    85,    86,    87,    88,    89,    90,    -1,
      -1,    -1,    94,    95,    96,    97,    98,    -1,   100,   101,
      -1,    -1,   104,    -1,    -1,   107,   108,   109,    -1,    45,
      46,    47,    48,    49,    50,    51,    52,    -1,    54,    55,
      56,    57,    -1,    59,    60,    -1,    -1,    -1,    -1,    -1,
      -1,     1,    -1,    -1,    -1,    -1,    -1,    -1,    74,    75,
      76,    11,    12,    -1,    14,    15,    -1,    -1,    84,    85,
      86,    87,    88,    89,    90,    -1,    -1,    -1,    94,    95,
      96,    97,    98,    -1,   100,   101,    -1,    -1,   104,    -1,
      -1,   107,   108,   109,    -1,    45,    46,    47,    48,    49,
      50,    51,    52,    -1,    54,    55,    56,    57,     1,    59,
      60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    12,
      -1,    14,    15,    -1,    74,    75,    76,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    84,    85,    86,    87,    88,    89,
      90,    -1,    -1,    -1,    94,    95,    96,    97,    98,    -1,
     100,   101,    45,    46,   104,    -1,    49,   107,   108,   109,
      -1,    54,    55,    -1,    57,     1,    59,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    12,    -1,    -1,    15,
      16,    74,    75,    76,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    84,    85,    86,    87,    88,    89,    90,    -1,    -1,
      -1,    94,    -1,    96,    97,    -1,    -1,   100,   101,    45,
      46,   104,    -1,    49,   107,   108,   109,    -1,    54,    55,
      -1,    57,    -1,    59,    -1,    -1,    -1,    -1,     7,     8,
      -1,    -1,    11,    12,    -1,    -1,    15,    -1,    74,    75,
      76,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    84,    85,
      86,    87,    88,    89,    90,    -1,    -1,    -1,    94,    -1,
      96,    97,    -1,    -1,   100,   101,    45,    46,   104,    -1,
      -1,   107,   108,   109,    53,    54,    55,    -1,    57,    -1,
      59,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    76,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    84,    85,    86,    87,    88,
      89,    90,    -1,    -1,    -1,    94,    -1,    96,    97,    -1,
      -1,   100,   101,    -1,    -1,   104,    -1,    -1,   107,   108,
     109,     9,    -1,    -1,    12,    13,    14,    15,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    23,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    35,    36,    -1,
      38,    39,    40,    41,    42,    43,    44,    45,    46,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    54,    55,    -1,    57,
      -1,    59,    -1,    -1,    -1,    -1,    -1,    -1,     9,    -1,
      -1,    12,    -1,    14,    15,    -1,    -1,    -1,    76,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    84,    85,    86,    87,
      88,    89,    90,    -1,    92,    -1,    94,    -1,    96,    97,
      -1,    -1,   100,   101,    45,    46,   104,    -1,    -1,   107,
     108,   109,    -1,    54,    55,    -1,    57,    -1,    59,    -1,
      -1,    -1,    -1,    -1,    -1,     9,    -1,    -1,    12,    -1,
      -1,    15,    -1,    -1,    -1,    76,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    84,    85,    86,    87,    88,    89,    90,
      -1,    -1,    -1,    94,    -1,    96,    97,    -1,    -1,   100,
     101,    45,    46,   104,    -1,    -1,   107,   108,   109,    -1,
      54,    55,    -1,    57,    -1,    59,    -1,    -1,    -1,    -1,
      -1,    -1,     9,    -1,    -1,    12,    -1,    -1,    15,    -1,
      -1,    -1,    76,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      84,    85,    86,    87,    88,    89,    90,    -1,    -1,    -1,
      94,    -1,    96,    97,    -1,    -1,   100,   101,    45,    46,
     104,    -1,    -1,   107,   108,   109,    -1,    54,    55,    -1,
      57,    -1,    59,    -1,    -1,    -1,    -1,    -1,    -1,     9,
      -1,    -1,    12,    -1,    -1,    15,    -1,    -1,    -1,    76,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    84,    85,    86,
      87,    88,    89,    90,    -1,    -1,    -1,    94,    -1,    96,
      97,    -1,    -1,   100,   101,    45,    46,   104,    -1,    -1,
     107,   108,   109,    -1,    54,    55,    -1,    57,    -1,    59,
      -1,    -1,    -1,    -1,    -1,    -1,     9,    -1,    -1,    12,
      -1,    -1,    15,    -1,    -1,    -1,    76,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    84,    85,    86,    87,    88,    89,
      90,    -1,    -1,    -1,    94,    -1,    96,    97,    -1,    -1,
     100,   101,    45,    46,   104,    -1,    -1,   107,   108,   109,
      -1,    54,    55,    -1,    57,    -1,    59,    -1,    -1,    -1,
      -1,    -1,    -1,     9,    -1,    -1,    12,    -1,    -1,    15,
      -1,    -1,    -1,    76,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    84,    85,    86,    87,    88,    89,    90,    -1,    -1,
      -1,    94,    -1,    96,    97,    -1,    -1,   100,   101,    45,
      46,   104,    -1,    -1,   107,   108,   109,    -1,    54,    55,
      -1,    57,    -1,    59,    -1,    -1,    -1,    -1,    -1,    -1,
       9,    -1,    -1,    12,    -1,    -1,    15,    -1,    -1,    -1,
      76,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    84,    85,
      86,    87,    88,    89,    90,    -1,    -1,    -1,    94,    -1,
      96,    97,    -1,    -1,   100,   101,    45,    46,   104,    -1,
      -1,   107,   108,   109,    -1,    54,    55,    -1,    57,    -1,
      59,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    76,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    84,    85,    86,    87,    88,
      89,    90,    -1,    -1,    -1,    94,    -1,    96,    97,    -1,
      -1,   100,   101,    -1,    -1,   104,    -1,    -1,   107,   108,
     109,    10,    -1,    12,    13,    -1,    15,    16,    -1,    -1,
      -1,    -1,    -1,    -1,    23,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    35,    36,    -1,    38,
      39,    40,    41,    42,    43,    44,    45,    46,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    54,    55,    -1,    57,    -1,
      59,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    76,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    84,    85,    86,    87,    88,
      89,    90,    -1,    92,    -1,    94,    -1,    96,    97,    -1,
      -1,   100,   101,    -1,    -1,   104,    -1,    -1,   107,   108,
     109,    10,    -1,    12,    13,    -1,    15,    16,    -1,    -1,
      -1,    -1,    -1,    -1,    23,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    35,    36,    -1,    38,
      39,    40,    41,    42,    43,    44,    45,    46,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    54,    55,    -1,    57,    -1,
      59,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    76,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    84,    85,    86,    87,    88,
      89,    90,    -1,    92,    -1,    94,    -1,    96,    97,    -1,
      -1,   100,   101,    -1,    -1,   104,    -1,    -1,   107,   108,
     109,    10,    -1,    12,    13,    -1,    15,    16,    -1,    -1,
      -1,    -1,    -1,    -1,    23,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    35,    36,    -1,    38,
      39,    40,    41,    42,    43,    44,    45,    46,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    54,    55,    -1,    57,    -1,
      59,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    76,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    84,    85,    86,    87,    88,
      89,    90,    -1,    92,    -1,    94,    -1,    96,    97,    -1,
      -1,   100,   101,    -1,    -1,   104,    -1,    -1,   107,   108,
     109,    10,    -1,    12,    13,    -1,    15,    16,    -1,    -1,
      -1,    -1,    -1,    -1,    23,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    35,    36,    -1,    38,
      39,    40,    41,    42,    43,    44,    45,    46,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    54,    55,    -1,    57,    -1,
      59,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    76,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    84,    85,    86,    87,    88,
      89,    90,    -1,    92,    -1,    94,    -1,    96,    97,    -1,
      -1,   100,   101,    -1,    -1,   104,    -1,    -1,   107,   108,
     109,    10,    -1,    12,    13,    -1,    15,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    23,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    35,    36,    -1,    38,
      39,    40,    41,    42,    43,    44,    45,    46,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    54,    55,    -1,    57,    -1,
      59,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    76,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    84,    85,    86,    87,    88,
      89,    90,    -1,    92,    -1,    94,    -1,    96,    97,    -1,
      -1,   100,   101,    12,    13,   104,    15,    16,   107,   108,
     109,    -1,    -1,    -1,    23,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    35,    36,    -1,    38,
      39,    40,    41,    42,    43,    44,    45,    46,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    54,    55,    -1,    57,    -1,
      59,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    76,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    84,    85,    86,    87,    88,
      89,    90,    -1,    92,    -1,    94,    -1,    96,    97,    -1,
      -1,   100,   101,    12,    13,   104,    15,    -1,   107,   108,
     109,    -1,    -1,    -1,    23,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    35,    36,    -1,    38,
      39,    40,    41,    42,    43,    44,    45,    46,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    54,    55,    -1,    57,    -1,
      59,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    76,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    84,    85,    86,    87,    88,
      89,    90,    -1,    92,    93,    94,    -1,    96,    97,    -1,
      -1,   100,   101,    12,    13,   104,    15,    16,   107,   108,
     109,    -1,    -1,    -1,    23,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    35,    36,    -1,    38,
      39,    40,    41,    42,    43,    44,    45,    46,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    54,    55,    -1,    57,    -1,
      59,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    76,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    84,    85,    86,    87,    88,
      89,    90,    -1,    92,    -1,    94,    -1,    96,    97,    -1,
      -1,   100,   101,    12,    13,   104,    15,    16,   107,   108,
     109,    -1,    -1,    -1,    23,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    35,    36,    -1,    38,
      39,    40,    41,    42,    43,    44,    45,    46,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    54,    55,    -1,    57,    -1,
      59,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    76,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    84,    85,    86,    87,    88,
      89,    90,    -1,    92,    -1,    94,    -1,    96,    97,    -1,
      -1,   100,   101,    12,    13,   104,    15,    16,   107,   108,
     109,    -1,    -1,    -1,    23,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    35,    36,    -1,    38,
      39,    40,    41,    42,    43,    44,    45,    46,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    54,    55,    -1,    57,    -1,
      59,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    76,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    84,    85,    86,    87,    88,
      89,    90,    -1,    92,    -1,    94,    -1,    96,    97,    -1,
      -1,   100,   101,    -1,    -1,   104,    -1,    -1,   107,   108,
     109,    12,    13,    14,    15,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    23,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    35,    36,    -1,    38,    39,    40,
      41,    42,    43,    44,    45,    46,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    54,    55,    -1,    57,    -1,    59,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    76,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    84,    85,    86,    87,    88,    89,    90,
      -1,    92,    -1,    94,    -1,    96,    97,    -1,    -1,   100,
     101,    12,    13,   104,    15,    16,   107,   108,   109,    -1,
      -1,    -1,    23,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    35,    36,    -1,    38,    39,    40,
      41,    42,    43,    44,    45,    46,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    54,    55,    -1,    57,    -1,    59,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    76,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    84,    85,    86,    87,    88,    89,    90,
      -1,    92,    -1,    94,    -1,    96,    97,    -1,    -1,   100,
     101,    12,    13,   104,    15,    16,   107,   108,   109,    -1,
      -1,    -1,    23,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    35,    36,    -1,    38,    39,    40,
      41,    42,    43,    44,    45,    46,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    54,    55,    -1,    57,    -1,    59,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    76,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    84,    85,    86,    87,    88,    89,    90,
      -1,    92,    -1,    94,    -1,    96,    97,    -1,    -1,   100,
     101,    12,    13,   104,    15,    16,   107,   108,   109,    -1,
      -1,    -1,    23,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    35,    36,    -1,    38,    39,    40,
      41,    42,    43,    44,    45,    46,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    54,    55,    -1,    57,    -1,    59,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    76,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    84,    85,    86,    87,    88,    89,    90,
      -1,    92,    -1,    94,    -1,    96,    97,    -1,    -1,   100,
     101,    12,    13,   104,    15,    -1,   107,   108,   109,    -1,
      -1,    -1,    23,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    35,    36,    -1,    38,    39,    40,
      41,    42,    43,    44,    45,    46,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    54,    55,    -1,    57,    -1,    59,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    76,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    84,    85,    86,    87,    88,    89,    90,
      -1,    92,    -1,    94,    -1,    96,    97,    -1,    -1,   100,
     101,    12,    13,   104,    15,    -1,   107,   108,   109,    -1,
      -1,    -1,    23,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    35,    -1,    -1,    38,    39,    40,
      41,    42,    43,    44,    45,    46,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    54,    55,    -1,    57,    -1,    59,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    76,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    84,    85,    86,    87,    88,    89,    90,
      -1,    12,    13,    94,    15,    96,    97,    -1,    -1,   100,
     101,    -1,    23,   104,    -1,    -1,   107,   108,   109,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    38,    39,    40,
      41,    42,    43,    44,    45,    46,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    54,    55,    -1,    57,    -1,    59,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    76,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    84,    85,    86,    87,    88,    89,    90,
      -1,    12,    13,    94,    15,    96,    97,    -1,    -1,   100,
     101,    -1,    23,   104,    -1,    -1,   107,   108,   109,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    38,    39,    40,
      41,    42,    43,    44,    45,    46,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    54,    55,    -1,    57,    -1,    59,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    76,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    84,    85,    86,    87,    88,    89,    90,
      12,    -1,    -1,    15,    -1,    96,    97,    -1,    -1,   100,
     101,    23,    -1,   104,    -1,    -1,   107,   108,   109,    -1,
      -1,    -1,    -1,    35,    36,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    44,    45,    46,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    54,    55,    -1,    57,    -1,    59,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    76,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    84,    85,    86,    87,    88,    89,    90,    -1,
      92,    93,    94,    12,    96,    97,    15,    -1,   100,   101,
      -1,    -1,   104,    -1,    23,   107,   108,   109,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    35,    36,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    44,    45,    46,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    54,    55,    -1,    57,    -1,
      59,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    76,    -1,    -1,
      -1,    12,    -1,    -1,    15,    84,    85,    86,    87,    88,
      89,    90,    23,    92,    -1,    94,    -1,    96,    97,    -1,
      -1,   100,   101,    -1,    35,   104,    -1,    -1,   107,   108,
     109,    -1,    -1,    44,    45,    46,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    54,    55,    -1,    57,    -1,    59,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    76,    -1,    -1,    -1,    12,
      -1,    -1,    15,    84,    85,    86,    87,    88,    89,    90,
      23,    -1,    -1,    94,    -1,    96,    97,    -1,    -1,   100,
     101,    -1,    -1,   104,    -1,    -1,   107,   108,   109,    -1,
      -1,    44,    45,    46,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    54,    55,    -1,    57,    -1,    59,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    12,    -1,    -1,    15,
      16,    -1,    -1,    76,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    84,    85,    86,    87,    88,    89,    90,    -1,    -1,
      -1,    94,    -1,    96,    97,    -1,    -1,   100,   101,    45,
      46,   104,    -1,    -1,   107,   108,   109,    -1,    54,    55,
      -1,    57,    -1,    59,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    12,    -1,    -1,    15,    16,    -1,    -1,
      76,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    84,    85,
      86,    87,    88,    89,    90,    -1,    -1,    -1,    94,    -1,
      96,    97,    -1,    -1,   100,   101,    45,    46,   104,    -1,
      -1,   107,   108,   109,    -1,    54,    55,    -1,    57,    -1,
      59,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      12,    -1,    14,    15,    -1,    -1,    -1,    76,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    84,    85,    86,    87,    88,
      89,    90,    -1,    -1,    -1,    94,    -1,    96,    97,    -1,
      -1,   100,   101,    45,    46,   104,    -1,    -1,   107,   108,
     109,    -1,    54,    55,    -1,    57,    -1,    59,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    76,    -1,    -1,    -1,    12,    -1,
      -1,    15,    84,    85,    86,    87,    88,    89,    90,    23,
      -1,    -1,    94,    -1,    96,    97,    -1,    -1,   100,   101,
      -1,    -1,   104,    -1,    -1,   107,   108,   109,    -1,    -1,
      44,    45,    46,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      54,    55,    -1,    57,    -1,    59,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    12,    -1,    -1,    15,    -1,
      -1,    -1,    76,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      84,    85,    86,    87,    88,    89,    90,    -1,    -1,    -1,
      -1,    -1,    96,    97,    -1,    -1,   100,   101,    45,    46,
     104,    -1,    -1,   107,   108,   109,    -1,    54,    55,    -1,
      57,    -1,    59,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    12,    -1,    -1,    15,    -1,    -1,    -1,    76,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    84,    85,    86,
      87,    88,    89,    90,    -1,    -1,    -1,    94,    -1,    96,
      97,    -1,    -1,   100,   101,    45,    46,   104,    -1,    -1,
     107,   108,   109,    -1,    54,    55,    -1,    57,    -1,    59,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    12,
      -1,    -1,    15,    -1,    -1,    -1,    76,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    84,    85,    86,    87,    88,    89,
      90,    -1,    -1,    -1,    94,    -1,    96,    97,    -1,    -1,
     100,   101,    45,    46,   104,    -1,    -1,   107,   108,   109,
      -1,    54,    55,    -1,    57,    -1,    59,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    12,    -1,    -1,    15,
      -1,    -1,    -1,    76,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    84,    85,    86,    87,    88,    89,    90,    -1,    -1,
      -1,    94,    -1,    96,    97,    -1,    -1,   100,   101,    45,
      46,   104,    -1,    -1,   107,   108,   109,    -1,    54,    55,
      -1,    57,    -1,    59,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      76,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    84,    85,
      86,    87,    88,    89,    90,    -1,    -1,    -1,    94,    -1,
      96,    97,    -1,    -1,   100,   101,    -1,    -1,   104,    -1,
      -1,   107,   108,   109
};

  /* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
     symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,     1,     9,    14,   112,   124,   126,   138,     0,     7,
       8,    11,    12,    15,    45,    46,    53,    54,    55,    57,
      59,    76,    84,    85,    86,    87,    88,    89,    90,    94,
      96,    97,   100,   101,   104,   107,   108,   109,   121,   127,
     128,   130,   133,   140,   141,   148,   149,   150,   152,     9,
      14,   121,   121,   133,   134,   142,    12,    84,    87,   119,
      12,    12,    12,    12,    42,   150,    12,    12,    12,   104,
     149,   150,   149,   133,   140,   149,   150,   150,   149,     1,
       9,    14,    47,    48,    49,    50,    51,    52,    56,    60,
      74,    75,    95,    98,   116,   118,   120,   121,   133,   137,
     144,   146,   147,   153,    10,   121,   124,    13,    23,    35,
      36,    38,    39,    40,    41,    42,    43,    44,    92,   113,
     114,   149,    12,    15,   100,   101,   102,   103,   106,    67,
     107,   108,    18,   147,   147,    10,    16,   115,    16,   115,
      91,    16,   131,   133,    12,   133,   133,   131,    16,   131,
     149,    42,   133,   133,   133,   149,     9,   122,   123,    14,
     122,   145,   145,   152,   133,   145,    12,    12,   145,   133,
     145,    12,     9,   146,   146,   147,    12,   132,   135,   136,
     149,   150,   145,    17,   146,   146,   127,   147,   128,    94,
     133,   141,   133,   133,   133,   133,   133,   133,   152,   133,
       9,   133,     9,   133,   133,   141,   149,   149,   149,   149,
     149,   149,   133,   131,    17,    17,     9,   133,    44,   133,
      15,    16,   115,    84,   151,   115,   115,    16,    16,   149,
     115,   115,     9,   123,    18,   145,   125,   144,   152,   133,
     145,   133,    98,    77,   117,    17,   139,   134,    23,    44,
      92,   113,   114,   149,   115,    13,    37,    40,    67,   121,
      17,   150,    93,   115,   115,    19,   152,   133,    16,   115,
     143,   133,   141,   133,   141,   152,   133,   131,    14,    44,
     143,   143,    12,     9,   146,     9,    16,    12,   132,   141,
     152,   132,   132,   132,   132,   149,   149,   149,   132,   147,
     133,   133,   133,    84,     9,   129,    16,    16,    16,    16,
      16,   115,    16,   115,    19,    14,   133,   152,   133,    44,
     134,    93,    17,    16,   115,    16,   115,   121,   133,   141,
     133,   125,    14,   143,    16,   152,    16,   132,   150,   150,
     147,    16,    16,    16,   143,   125,   146,   145,    16,    16,
      17,   146,   143,   146
};

  /* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,   111,   112,   112,   113,   113,   114,   114,   115,   115,
     116,   116,   117,   117,   118,   118,   118,   119,   119,   120,
     121,   121,   122,   122,   123,   123,   124,   124,   125,   125,
     126,   126,   127,   128,   128,   128,   128,   128,   128,   128,
     129,   128,   130,   130,   131,   131,   132,   132,   132,   132,
     132,   132,   132,   132,   132,   132,   133,   133,   133,   133,
     133,   133,   133,   133,   133,   133,   133,   133,   133,   133,
     133,   133,   133,   133,   133,   134,   134,   135,   135,   136,
     136,   136,   137,   137,   138,   138,   138,   138,   139,   139,
     140,   140,   142,   141,   143,   143,   144,   144,   144,   144,
     144,   144,   144,   144,   145,   145,   146,   146,   146,   146,
     146,   146,   146,   146,   146,   146,   146,   146,   146,   146,
     146,   147,   147,   148,   148,   149,   149,   149,   149,   149,
     149,   149,   149,   149,   149,   149,   149,   149,   149,   149,
     149,   149,   149,   149,   149,   149,   149,   149,   149,   149,
     149,   149,   149,   149,   149,   149,   149,   149,   149,   149,
     149,   149,   149,   149,   149,   149,   150,   150,   150,   150,
     150,   151,   151,   151,   152,   152,   152,   153
};

  /* YYR2[YYN] -- Number of symbols on the right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     1,     1,     1,     2,     1,     2,     1,     2,
       1,     2,     1,     2,     9,     8,     7,     1,     1,     4,
       1,     2,     1,     2,     0,     1,     0,     1,     0,     1,
       1,     3,     1,     1,     4,     3,     6,     3,     4,     4,
       0,     9,     1,     3,     1,     3,     3,     5,     3,     3,
       3,     3,     3,     5,     2,     1,     3,     5,     3,     3,
       2,     3,     3,     3,     3,     3,     3,     3,     3,     3,
       5,     4,     3,     2,     1,     3,     3,     1,     3,     0,
       1,     3,     1,     1,     1,     1,     2,     2,     1,     2,
       1,     2,     0,     4,     1,     2,     4,     4,     4,     2,
       5,     2,     1,     1,     1,     2,     2,     2,     7,     3,
       2,     1,     4,     2,     3,     2,     3,     2,     2,     2,
       2,     1,     2,     1,     1,     3,     3,     3,     3,     3,
       3,     2,     2,     2,     3,     4,     1,     3,     4,     2,
       2,     2,     2,     4,     3,     2,     1,     6,     6,     3,
       6,     6,     1,     8,     8,     6,     4,     1,     6,     6,
       8,     8,     8,     6,     1,     1,     1,     4,     1,     1,
       2,     0,     1,     3,     1,     1,     1,     4
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
#line 107 "awk.g.2001.y" /* yacc.c:1646  */
    { if (errorflag==0)
			winner = (Node *)stat3(PROGRAM, beginloc, (yyvsp[0].p), endloc); }
#line 2537 "y.tab.c" /* yacc.c:1646  */
    break;

  case 3:
#line 109 "awk.g.2001.y" /* yacc.c:1646  */
    { yyclearin; bracecheck(); vyyerror(":95:Bailing out"); }
#line 2543 "y.tab.c" /* yacc.c:1646  */
    break;

  case 10:
#line 125 "awk.g.2001.y" /* yacc.c:1646  */
    { }
#line 2549 "y.tab.c" /* yacc.c:1646  */
    break;

  case 12:
#line 129 "awk.g.2001.y" /* yacc.c:1646  */
    { }
#line 2555 "y.tab.c" /* yacc.c:1646  */
    break;

  case 14:
#line 134 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = stat4(FOR, (yyvsp[-6].p), notnull((yyvsp[-4].p)), (yyvsp[-2].p), (yyvsp[0].p)); }
#line 2561 "y.tab.c" /* yacc.c:1646  */
    break;

  case 15:
#line 136 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = stat4(FOR, (yyvsp[-5].p), NIL, (yyvsp[-2].p), (yyvsp[0].p)); }
#line 2567 "y.tab.c" /* yacc.c:1646  */
    break;

  case 16:
#line 138 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = stat3(IN, (yyvsp[-4].p), makearr((yyvsp[-2].p)), (yyvsp[0].p)); }
#line 2573 "y.tab.c" /* yacc.c:1646  */
    break;

  case 17:
#line 142 "awk.g.2001.y" /* yacc.c:1646  */
    { setfname((yyvsp[0].cp)); }
#line 2579 "y.tab.c" /* yacc.c:1646  */
    break;

  case 18:
#line 143 "awk.g.2001.y" /* yacc.c:1646  */
    { setfname((yyvsp[0].cp)); }
#line 2585 "y.tab.c" /* yacc.c:1646  */
    break;

  case 19:
#line 147 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = notnull((yyvsp[-1].p)); }
#line 2591 "y.tab.c" /* yacc.c:1646  */
    break;

  case 24:
#line 159 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.i) = 0; }
#line 2597 "y.tab.c" /* yacc.c:1646  */
    break;

  case 26:
#line 164 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.i) = 0; }
#line 2603 "y.tab.c" /* yacc.c:1646  */
    break;

  case 28:
#line 170 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = 0; }
#line 2609 "y.tab.c" /* yacc.c:1646  */
    break;

  case 30:
#line 175 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = 0; }
#line 2615 "y.tab.c" /* yacc.c:1646  */
    break;

  case 31:
#line 176 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = (yyvsp[-1].p); }
#line 2621 "y.tab.c" /* yacc.c:1646  */
    break;

  case 32:
#line 180 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = notnull((yyvsp[0].p)); }
#line 2627 "y.tab.c" /* yacc.c:1646  */
    break;

  case 33:
#line 184 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = stat2(PASTAT, (yyvsp[0].p), stat2(PRINT, rectonode(), NIL)); }
#line 2633 "y.tab.c" /* yacc.c:1646  */
    break;

  case 34:
#line 185 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = stat2(PASTAT, (yyvsp[-3].p), (yyvsp[-1].p)); }
#line 2639 "y.tab.c" /* yacc.c:1646  */
    break;

  case 35:
#line 186 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = pa2stat((yyvsp[-2].p), (yyvsp[0].p), stat2(PRINT, rectonode(), NIL)); }
#line 2645 "y.tab.c" /* yacc.c:1646  */
    break;

  case 36:
#line 187 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = pa2stat((yyvsp[-5].p), (yyvsp[-3].p), (yyvsp[-1].p)); }
#line 2651 "y.tab.c" /* yacc.c:1646  */
    break;

  case 37:
#line 188 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = stat2(PASTAT, NIL, (yyvsp[-1].p)); }
#line 2657 "y.tab.c" /* yacc.c:1646  */
    break;

  case 38:
#line 190 "awk.g.2001.y" /* yacc.c:1646  */
    { beginloc = linkum(beginloc, (yyvsp[-1].p)); (yyval.p) = 0; }
#line 2663 "y.tab.c" /* yacc.c:1646  */
    break;

  case 39:
#line 192 "awk.g.2001.y" /* yacc.c:1646  */
    { endloc = linkum(endloc, (yyvsp[-1].p)); (yyval.p) = 0; }
#line 2669 "y.tab.c" /* yacc.c:1646  */
    break;

  case 40:
#line 193 "awk.g.2001.y" /* yacc.c:1646  */
    {infunc++;}
#line 2675 "y.tab.c" /* yacc.c:1646  */
    break;

  case 41:
#line 194 "awk.g.2001.y" /* yacc.c:1646  */
    { infunc--; curfname=0; defn((Cell *)(yyvsp[-7].p), (yyvsp[-5].p), (yyvsp[-1].p)); (yyval.p) = 0; }
#line 2681 "y.tab.c" /* yacc.c:1646  */
    break;

  case 43:
#line 199 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = linkum((yyvsp[-2].p), (yyvsp[0].p)); }
#line 2687 "y.tab.c" /* yacc.c:1646  */
    break;

  case 45:
#line 204 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = linkum((yyvsp[-2].p), (yyvsp[0].p)); }
#line 2693 "y.tab.c" /* yacc.c:1646  */
    break;

  case 46:
#line 208 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op2((yyvsp[-1].i), (yyvsp[-2].p), (yyvsp[0].p)); }
#line 2699 "y.tab.c" /* yacc.c:1646  */
    break;

  case 47:
#line 210 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op3(CONDEXPR, notnull((yyvsp[-4].p)), (yyvsp[-2].p), (yyvsp[0].p)); }
#line 2705 "y.tab.c" /* yacc.c:1646  */
    break;

  case 48:
#line 212 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op2(BOR, notnull((yyvsp[-2].p)), notnull((yyvsp[0].p))); }
#line 2711 "y.tab.c" /* yacc.c:1646  */
    break;

  case 49:
#line 214 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op2(AND, notnull((yyvsp[-2].p)), notnull((yyvsp[0].p))); }
#line 2717 "y.tab.c" /* yacc.c:1646  */
    break;

  case 50:
#line 215 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op3((yyvsp[-1].i), NIL, (yyvsp[-2].p), (Node*)makedfa((yyvsp[0].s), 0)); }
#line 2723 "y.tab.c" /* yacc.c:1646  */
    break;

  case 51:
#line 217 "awk.g.2001.y" /* yacc.c:1646  */
    { if (constnode((yyvsp[0].p)))
			(yyval.p) = op3((yyvsp[-1].i), NIL, (yyvsp[-2].p), (Node*)makedfa(strnode((yyvsp[0].p)), 0));
		  else
			(yyval.p) = op3((yyvsp[-1].i), (Node *)1, (yyvsp[-2].p), (yyvsp[0].p)); }
#line 2732 "y.tab.c" /* yacc.c:1646  */
    break;

  case 52:
#line 221 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op2(INTEST, (yyvsp[-2].p), makearr((yyvsp[0].p))); }
#line 2738 "y.tab.c" /* yacc.c:1646  */
    break;

  case 53:
#line 222 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op2(INTEST, (yyvsp[-3].p), makearr((yyvsp[0].p))); }
#line 2744 "y.tab.c" /* yacc.c:1646  */
    break;

  case 54:
#line 223 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op2(CAT, (yyvsp[-1].p), (yyvsp[0].p)); }
#line 2750 "y.tab.c" /* yacc.c:1646  */
    break;

  case 56:
#line 228 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op2((yyvsp[-1].i), (yyvsp[-2].p), (yyvsp[0].p)); }
#line 2756 "y.tab.c" /* yacc.c:1646  */
    break;

  case 57:
#line 230 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op3(CONDEXPR, notnull((yyvsp[-4].p)), (yyvsp[-2].p), (yyvsp[0].p)); }
#line 2762 "y.tab.c" /* yacc.c:1646  */
    break;

  case 58:
#line 232 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op2(BOR, notnull((yyvsp[-2].p)), notnull((yyvsp[0].p))); }
#line 2768 "y.tab.c" /* yacc.c:1646  */
    break;

  case 59:
#line 234 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op2(AND, notnull((yyvsp[-2].p)), notnull((yyvsp[0].p))); }
#line 2774 "y.tab.c" /* yacc.c:1646  */
    break;

  case 60:
#line 236 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op1(NOT, op2(NE,(yyvsp[0].p),valtonode(lookup("$zero&null",symtab),CCON))); }
#line 2780 "y.tab.c" /* yacc.c:1646  */
    break;

  case 61:
#line 237 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op2((yyvsp[-1].i), (yyvsp[-2].p), (yyvsp[0].p)); }
#line 2786 "y.tab.c" /* yacc.c:1646  */
    break;

  case 62:
#line 238 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op2((yyvsp[-1].i), (yyvsp[-2].p), (yyvsp[0].p)); }
#line 2792 "y.tab.c" /* yacc.c:1646  */
    break;

  case 63:
#line 239 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op2((yyvsp[-1].i), (yyvsp[-2].p), (yyvsp[0].p)); }
#line 2798 "y.tab.c" /* yacc.c:1646  */
    break;

  case 64:
#line 240 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op2((yyvsp[-1].i), (yyvsp[-2].p), (yyvsp[0].p)); }
#line 2804 "y.tab.c" /* yacc.c:1646  */
    break;

  case 65:
#line 241 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op2((yyvsp[-1].i), (yyvsp[-2].p), (yyvsp[0].p)); }
#line 2810 "y.tab.c" /* yacc.c:1646  */
    break;

  case 66:
#line 242 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op2((yyvsp[-1].i), (yyvsp[-2].p), (yyvsp[0].p)); }
#line 2816 "y.tab.c" /* yacc.c:1646  */
    break;

  case 67:
#line 243 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op3((yyvsp[-1].i), NIL, (yyvsp[-2].p), (Node*)makedfa((yyvsp[0].s), 0)); }
#line 2822 "y.tab.c" /* yacc.c:1646  */
    break;

  case 68:
#line 245 "awk.g.2001.y" /* yacc.c:1646  */
    { if (constnode((yyvsp[0].p)))
			(yyval.p) = op3((yyvsp[-1].i), NIL, (yyvsp[-2].p), (Node*)makedfa(strnode((yyvsp[0].p)), 0));
		  else
			(yyval.p) = op3((yyvsp[-1].i), (Node *)1, (yyvsp[-2].p), (yyvsp[0].p)); }
#line 2831 "y.tab.c" /* yacc.c:1646  */
    break;

  case 69:
#line 249 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op2(INTEST, (yyvsp[-2].p), makearr((yyvsp[0].p))); }
#line 2837 "y.tab.c" /* yacc.c:1646  */
    break;

  case 70:
#line 250 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op2(INTEST, (yyvsp[-3].p), makearr((yyvsp[0].p))); }
#line 2843 "y.tab.c" /* yacc.c:1646  */
    break;

  case 71:
#line 251 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op3(GETLINE, (yyvsp[0].p), (Node*)(yyvsp[-2].i), (yyvsp[-3].p)); }
#line 2849 "y.tab.c" /* yacc.c:1646  */
    break;

  case 72:
#line 252 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op3(GETLINE, (Node*)0, (Node*)(yyvsp[-1].i), (yyvsp[-2].p)); }
#line 2855 "y.tab.c" /* yacc.c:1646  */
    break;

  case 73:
#line 253 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op2(CAT, (yyvsp[-1].p), (yyvsp[0].p)); }
#line 2861 "y.tab.c" /* yacc.c:1646  */
    break;

  case 75:
#line 258 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = linkum((yyvsp[-2].p), (yyvsp[0].p)); }
#line 2867 "y.tab.c" /* yacc.c:1646  */
    break;

  case 76:
#line 259 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = linkum((yyvsp[-2].p), (yyvsp[0].p)); }
#line 2873 "y.tab.c" /* yacc.c:1646  */
    break;

  case 78:
#line 264 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = linkum((yyvsp[-2].p), (yyvsp[0].p)); }
#line 2879 "y.tab.c" /* yacc.c:1646  */
    break;

  case 79:
#line 268 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = rectonode(); }
#line 2885 "y.tab.c" /* yacc.c:1646  */
    break;

  case 81:
#line 270 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = (yyvsp[-1].p); }
#line 2891 "y.tab.c" /* yacc.c:1646  */
    break;

  case 88:
#line 282 "awk.g.2001.y" /* yacc.c:1646  */
    { }
#line 2897 "y.tab.c" /* yacc.c:1646  */
    break;

  case 90:
#line 287 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op3(MATCH, NIL, rectonode(), (Node*)makedfa((yyvsp[0].s),0)); }
#line 2903 "y.tab.c" /* yacc.c:1646  */
    break;

  case 91:
#line 288 "awk.g.2001.y" /* yacc.c:1646  */
    {(yyval.p) = op1(NOT, notnull((yyvsp[0].p))); }
#line 2909 "y.tab.c" /* yacc.c:1646  */
    break;

  case 92:
#line 292 "awk.g.2001.y" /* yacc.c:1646  */
    {startreg();}
#line 2915 "y.tab.c" /* yacc.c:1646  */
    break;

  case 93:
#line 292 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.s) = (yyvsp[-1].s); }
#line 2921 "y.tab.c" /* yacc.c:1646  */
    break;

  case 96:
#line 300 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = stat3((yyvsp[-3].i), (yyvsp[-2].p), (Node *) (yyvsp[-1].i), (yyvsp[0].p)); }
#line 2927 "y.tab.c" /* yacc.c:1646  */
    break;

  case 97:
#line 301 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = stat3((yyvsp[-3].i), (yyvsp[-2].p), (Node *) (yyvsp[-1].i), (yyvsp[0].p)); }
#line 2933 "y.tab.c" /* yacc.c:1646  */
    break;

  case 98:
#line 302 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = stat3((yyvsp[-3].i), (yyvsp[-2].p), (Node *) (yyvsp[-1].i), (yyvsp[0].p)); }
#line 2939 "y.tab.c" /* yacc.c:1646  */
    break;

  case 99:
#line 303 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = stat3((yyvsp[-1].i), (yyvsp[0].p), NIL, NIL); }
#line 2945 "y.tab.c" /* yacc.c:1646  */
    break;

  case 100:
#line 304 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = stat2(DELETE, makearr((yyvsp[-3].p)), (yyvsp[-1].p)); }
#line 2951 "y.tab.c" /* yacc.c:1646  */
    break;

  case 101:
#line 305 "awk.g.2001.y" /* yacc.c:1646  */
    { yyclearin; vyyerror(":96:You can only delete array[element]"); (yyval.p) = stat1(DELETE, (yyvsp[0].p)); }
#line 2957 "y.tab.c" /* yacc.c:1646  */
    break;

  case 102:
#line 306 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = exptostat((yyvsp[0].p)); }
#line 2963 "y.tab.c" /* yacc.c:1646  */
    break;

  case 103:
#line 307 "awk.g.2001.y" /* yacc.c:1646  */
    { yyclearin; vyyerror(illstat); }
#line 2969 "y.tab.c" /* yacc.c:1646  */
    break;

  case 104:
#line 311 "awk.g.2001.y" /* yacc.c:1646  */
    { }
#line 2975 "y.tab.c" /* yacc.c:1646  */
    break;

  case 105:
#line 311 "awk.g.2001.y" /* yacc.c:1646  */
    { }
#line 2981 "y.tab.c" /* yacc.c:1646  */
    break;

  case 106:
#line 315 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = stat1(BREAK, NIL); }
#line 2987 "y.tab.c" /* yacc.c:1646  */
    break;

  case 107:
#line 316 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = stat1(CONTINUE, NIL); }
#line 2993 "y.tab.c" /* yacc.c:1646  */
    break;

  case 108:
#line 318 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = stat2(DO, (yyvsp[-5].p), notnull((yyvsp[-2].p))); }
#line 2999 "y.tab.c" /* yacc.c:1646  */
    break;

  case 109:
#line 319 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = stat1(EXIT, (yyvsp[-1].p)); }
#line 3005 "y.tab.c" /* yacc.c:1646  */
    break;

  case 110:
#line 320 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = stat1(EXIT, NIL); }
#line 3011 "y.tab.c" /* yacc.c:1646  */
    break;

  case 112:
#line 322 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = stat3(IF, (yyvsp[-3].p), (yyvsp[-2].p), (yyvsp[0].p)); }
#line 3017 "y.tab.c" /* yacc.c:1646  */
    break;

  case 113:
#line 323 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = stat3(IF, (yyvsp[-1].p), (yyvsp[0].p), NIL); }
#line 3023 "y.tab.c" /* yacc.c:1646  */
    break;

  case 114:
#line 324 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = (yyvsp[-1].p); }
#line 3029 "y.tab.c" /* yacc.c:1646  */
    break;

  case 115:
#line 325 "awk.g.2001.y" /* yacc.c:1646  */
    { if (infunc)
				vyyerror(":97:Next is illegal inside a function");
			  (yyval.p) = stat1(NEXT, NIL); }
#line 3037 "y.tab.c" /* yacc.c:1646  */
    break;

  case 116:
#line 328 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = stat1(RETURN, (yyvsp[-1].p)); }
#line 3043 "y.tab.c" /* yacc.c:1646  */
    break;

  case 117:
#line 329 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = stat1(RETURN, NIL); }
#line 3049 "y.tab.c" /* yacc.c:1646  */
    break;

  case 119:
#line 331 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = stat2(WHILE, (yyvsp[-1].p), (yyvsp[0].p)); }
#line 3055 "y.tab.c" /* yacc.c:1646  */
    break;

  case 120:
#line 332 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = 0; }
#line 3061 "y.tab.c" /* yacc.c:1646  */
    break;

  case 122:
#line 337 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = linkum((yyvsp[-1].p), (yyvsp[0].p)); }
#line 3067 "y.tab.c" /* yacc.c:1646  */
    break;

  case 125:
#line 345 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op2(ADD, (yyvsp[-2].p), (yyvsp[0].p)); }
#line 3073 "y.tab.c" /* yacc.c:1646  */
    break;

  case 126:
#line 346 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op2(MINUS, (yyvsp[-2].p), (yyvsp[0].p)); }
#line 3079 "y.tab.c" /* yacc.c:1646  */
    break;

  case 127:
#line 347 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op2(MULT, (yyvsp[-2].p), (yyvsp[0].p)); }
#line 3085 "y.tab.c" /* yacc.c:1646  */
    break;

  case 128:
#line 348 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op2(DIVIDE, (yyvsp[-2].p), (yyvsp[0].p)); }
#line 3091 "y.tab.c" /* yacc.c:1646  */
    break;

  case 129:
#line 349 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op2(MOD, (yyvsp[-2].p), (yyvsp[0].p)); }
#line 3097 "y.tab.c" /* yacc.c:1646  */
    break;

  case 130:
#line 350 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op2(POWER, (yyvsp[-2].p), (yyvsp[0].p)); }
#line 3103 "y.tab.c" /* yacc.c:1646  */
    break;

  case 131:
#line 351 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op1(UMINUS, (yyvsp[0].p)); }
#line 3109 "y.tab.c" /* yacc.c:1646  */
    break;

  case 132:
#line 352 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = (yyvsp[0].p); }
#line 3115 "y.tab.c" /* yacc.c:1646  */
    break;

  case 133:
#line 353 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op1(NOT, notnull((yyvsp[0].p))); }
#line 3121 "y.tab.c" /* yacc.c:1646  */
    break;

  case 134:
#line 354 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op2(BLTIN, (Node *) (yyvsp[-2].i), rectonode()); }
#line 3127 "y.tab.c" /* yacc.c:1646  */
    break;

  case 135:
#line 355 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op2(BLTIN, (Node *) (yyvsp[-3].i), (yyvsp[-1].p)); }
#line 3133 "y.tab.c" /* yacc.c:1646  */
    break;

  case 136:
#line 356 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op2(BLTIN, (Node *) (yyvsp[0].i), rectonode()); }
#line 3139 "y.tab.c" /* yacc.c:1646  */
    break;

  case 137:
#line 357 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op2(CALL, valtonode((yyvsp[-2].cp),CVAR), NIL); }
#line 3145 "y.tab.c" /* yacc.c:1646  */
    break;

  case 138:
#line 358 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op2(CALL, valtonode((yyvsp[-3].cp),CVAR), (yyvsp[-1].p)); }
#line 3151 "y.tab.c" /* yacc.c:1646  */
    break;

  case 139:
#line 359 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op1(PREDECR, (yyvsp[0].p)); }
#line 3157 "y.tab.c" /* yacc.c:1646  */
    break;

  case 140:
#line 360 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op1(PREINCR, (yyvsp[0].p)); }
#line 3163 "y.tab.c" /* yacc.c:1646  */
    break;

  case 141:
#line 361 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op1(POSTDECR, (yyvsp[-1].p)); }
#line 3169 "y.tab.c" /* yacc.c:1646  */
    break;

  case 142:
#line 362 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op1(POSTINCR, (yyvsp[-1].p)); }
#line 3175 "y.tab.c" /* yacc.c:1646  */
    break;

  case 143:
#line 363 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op3(GETLINE, (yyvsp[-2].p), (Node *)(yyvsp[-1].i), (yyvsp[0].p)); }
#line 3181 "y.tab.c" /* yacc.c:1646  */
    break;

  case 144:
#line 364 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op3(GETLINE, NIL, (Node *)(yyvsp[-1].i), (yyvsp[0].p)); }
#line 3187 "y.tab.c" /* yacc.c:1646  */
    break;

  case 145:
#line 365 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op3(GETLINE, (yyvsp[0].p), NIL, NIL); }
#line 3193 "y.tab.c" /* yacc.c:1646  */
    break;

  case 146:
#line 366 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op3(GETLINE, NIL, NIL, NIL); }
#line 3199 "y.tab.c" /* yacc.c:1646  */
    break;

  case 147:
#line 368 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op2(INDEX, (yyvsp[-3].p), (yyvsp[-1].p)); }
#line 3205 "y.tab.c" /* yacc.c:1646  */
    break;

  case 148:
#line 370 "awk.g.2001.y" /* yacc.c:1646  */
    { vyyerror(":98:Index() doesn't permit regular expressions");
		  (yyval.p) = op2(INDEX, (yyvsp[-3].p), (Node*)(yyvsp[-1].s)); }
#line 3212 "y.tab.c" /* yacc.c:1646  */
    break;

  case 149:
#line 372 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = (yyvsp[-1].p); }
#line 3218 "y.tab.c" /* yacc.c:1646  */
    break;

  case 150:
#line 374 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op3(MATCHFCN, NIL, (yyvsp[-3].p), (Node*)makedfa((yyvsp[-1].s), 1)); }
#line 3224 "y.tab.c" /* yacc.c:1646  */
    break;

  case 151:
#line 376 "awk.g.2001.y" /* yacc.c:1646  */
    { if (constnode((yyvsp[-1].p)))
			(yyval.p) = op3(MATCHFCN, NIL, (yyvsp[-3].p), (Node*)makedfa(strnode((yyvsp[-1].p)), 1));
		  else
			(yyval.p) = op3(MATCHFCN, (Node *)1, (yyvsp[-3].p), (yyvsp[-1].p)); }
#line 3233 "y.tab.c" /* yacc.c:1646  */
    break;

  case 152:
#line 380 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = valtonode((yyvsp[0].cp), CCON); }
#line 3239 "y.tab.c" /* yacc.c:1646  */
    break;

  case 153:
#line 382 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op4(SPLIT, (yyvsp[-5].p), makearr((yyvsp[-3].p)), (yyvsp[-1].p), (Node*)STRING); }
#line 3245 "y.tab.c" /* yacc.c:1646  */
    break;

  case 154:
#line 384 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op4(SPLIT, (yyvsp[-5].p), makearr((yyvsp[-3].p)), (Node*)makedfa((yyvsp[-1].s), 1), (Node *)REGEXPR); }
#line 3251 "y.tab.c" /* yacc.c:1646  */
    break;

  case 155:
#line 386 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op4(SPLIT, (yyvsp[-3].p), makearr((yyvsp[-1].p)), NIL, (Node*)STRING); }
#line 3257 "y.tab.c" /* yacc.c:1646  */
    break;

  case 156:
#line 387 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op1((yyvsp[-3].i), (yyvsp[-1].p)); }
#line 3263 "y.tab.c" /* yacc.c:1646  */
    break;

  case 157:
#line 388 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = valtonode((yyvsp[0].cp), CCON); }
#line 3269 "y.tab.c" /* yacc.c:1646  */
    break;

  case 158:
#line 390 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op4((yyvsp[-5].i), NIL, (Node*)makedfa((yyvsp[-3].s), 1), (yyvsp[-1].p), rectonode()); }
#line 3275 "y.tab.c" /* yacc.c:1646  */
    break;

  case 159:
#line 392 "awk.g.2001.y" /* yacc.c:1646  */
    { if (constnode((yyvsp[-3].p)))
			(yyval.p) = op4((yyvsp[-5].i), NIL, (Node*)makedfa(strnode((yyvsp[-3].p)), 1), (yyvsp[-1].p), rectonode());
		  else
			(yyval.p) = op4((yyvsp[-5].i), (Node *)1, (yyvsp[-3].p), (yyvsp[-1].p), rectonode()); }
#line 3284 "y.tab.c" /* yacc.c:1646  */
    break;

  case 160:
#line 397 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op4((yyvsp[-7].i), NIL, (Node*)makedfa((yyvsp[-5].s), 1), (yyvsp[-3].p), (yyvsp[-1].p)); }
#line 3290 "y.tab.c" /* yacc.c:1646  */
    break;

  case 161:
#line 399 "awk.g.2001.y" /* yacc.c:1646  */
    { if (constnode((yyvsp[-5].p)))
			(yyval.p) = op4((yyvsp[-7].i), NIL, (Node*)makedfa(strnode((yyvsp[-5].p)), 1), (yyvsp[-3].p), (yyvsp[-1].p));
		  else
			(yyval.p) = op4((yyvsp[-7].i), (Node *)1, (yyvsp[-5].p), (yyvsp[-3].p), (yyvsp[-1].p)); }
#line 3299 "y.tab.c" /* yacc.c:1646  */
    break;

  case 162:
#line 404 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op3(SUBSTR, (yyvsp[-5].p), (yyvsp[-3].p), (yyvsp[-1].p)); }
#line 3305 "y.tab.c" /* yacc.c:1646  */
    break;

  case 163:
#line 406 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op3(SUBSTR, (yyvsp[-3].p), (yyvsp[-1].p), NIL); }
#line 3311 "y.tab.c" /* yacc.c:1646  */
    break;

  case 167:
#line 413 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op2(ARRAY, makearr((yyvsp[-3].p)), (yyvsp[-1].p)); }
#line 3317 "y.tab.c" /* yacc.c:1646  */
    break;

  case 168:
#line 414 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = valtonode((yyvsp[0].cp), CFLD); }
#line 3323 "y.tab.c" /* yacc.c:1646  */
    break;

  case 169:
#line 415 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op1(INDIRECT, valtonode((yyvsp[0].cp), CVAR)); }
#line 3329 "y.tab.c" /* yacc.c:1646  */
    break;

  case 170:
#line 416 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op1(INDIRECT, (yyvsp[0].p)); }
#line 3335 "y.tab.c" /* yacc.c:1646  */
    break;

  case 171:
#line 420 "awk.g.2001.y" /* yacc.c:1646  */
    { arglist = (yyval.p) = 0; }
#line 3341 "y.tab.c" /* yacc.c:1646  */
    break;

  case 172:
#line 421 "awk.g.2001.y" /* yacc.c:1646  */
    { arglist = (yyval.p) = valtonode((yyvsp[0].cp),CVAR); }
#line 3347 "y.tab.c" /* yacc.c:1646  */
    break;

  case 173:
#line 422 "awk.g.2001.y" /* yacc.c:1646  */
    { arglist = (yyval.p) = linkum((yyvsp[-2].p),valtonode((yyvsp[0].cp),CVAR)); }
#line 3353 "y.tab.c" /* yacc.c:1646  */
    break;

  case 174:
#line 426 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = valtonode((yyvsp[0].cp), CVAR); }
#line 3359 "y.tab.c" /* yacc.c:1646  */
    break;

  case 175:
#line 427 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op1(ARG, (Node *) (yyvsp[0].i)); }
#line 3365 "y.tab.c" /* yacc.c:1646  */
    break;

  case 176:
#line 428 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = op1(VARNF, (Node *) (yyvsp[0].cp)); }
#line 3371 "y.tab.c" /* yacc.c:1646  */
    break;

  case 177:
#line 433 "awk.g.2001.y" /* yacc.c:1646  */
    { (yyval.p) = notnull((yyvsp[-1].p)); }
#line 3377 "y.tab.c" /* yacc.c:1646  */
    break;


#line 3381 "y.tab.c" /* yacc.c:1646  */
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
#line 436 "awk.g.2001.y" /* yacc.c:1906  */


static void
setfname(Cell *p)
{
	if (isarr(p))
		vyyerror(":99:%s is an array, not a function", p->nval);
	else if (isfunc(p))
		vyyerror(":100:You cannot define function %s more than once", p->nval);
	curfname = p->nval;
}

static int
constnode(Node *p)
{
	return p->ntype == NVALUE && ((Cell *) (p->narg[0]))->csub == CCON;
}

static unsigned char *strnode(Node *p)
{
	return ((Cell *)(p->narg[0]))->sval;
}

static Node *notnull(Node *n)
{
	switch (n->nobj) {
	case LE: case LT: case EQ: case NE: case GT: case GE:
	case BOR: case AND: case NOT:
		return n;
	default:
		return op2(NE, n, nullnode);
	}
}
