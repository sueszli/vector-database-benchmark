/*
*      Front End for The Small C+ Compiler
*
*      Based on the frontend from zcc096 but substantially
*      reworked
*
*      Each file is now processed in turn (all the way through)
*      And then everything is linked at the end, this makes it
*      quite a bit nicer, and a bit more standard - saves having
*      to preprocess all files and then find out there's an error
*      at the start of the first one!
*
*      $Id: zcc.c,v 1.195 2017-01-09 17:53:07 aralbrec Exp $
*/


#include        <stdio.h>
#include        <string.h>
#include        <stdlib.h>
#include        <stdarg.h>
#include        <ctype.h>
#include        <stddef.h>
#include        <stdint.h>
#include        <inttypes.h>
#include        <time.h>
#include        <sys/stat.h>
#include        "uthash.h"
#include        "utlist.h"
#include        "zcc.h"
#include        "regex/regex.h"
#include        "dirname.h"
#include        "option.h"

#ifdef WIN32
#include        <direct.h>
#include        <process.h>

#if !defined S_ISDIR
    #define S_ISDIR(m) (((m) & S_IFMT) == S_IFDIR)
#endif

#if !defined(S_ISREG) 
    #define S_ISREG(m) (((m) & S_IFMT) == S_IFREG)
#endif



#else
#include        <unistd.h>
#endif


#ifdef WIN32
#ifndef strcasecmp
#define strcasecmp(a,b) stricmp(a,b)
#endif
#endif

#if (_BSD_SOURCE || _SVID_SOURCE || _XOPEN_SOURCE >= 500)
#define mktempfile(a) mkstemp(a)
#else
#define mktempfile(a) mktemp(a)
#endif

enum {
    CPU_MAP_TOOL_Z80ASM = 0,
    CPU_MAP_TOOL_SCCZ80,
    CPU_MAP_TOOL_ZSDCC,
    CPU_MAP_TOOL_COPT,
    CPU_MAP_TOOL_CPURULES,
    CPU_MAP_TOOL_LIBNAME,
    CPU_MAP_TOOL_EZ80CLANG,
    CPU_MAP_TOOL_SIZE,
};

enum {
    CPU_TYPE_Z80 = 0,
    CPU_TYPE_Z80N,
    CPU_TYPE_Z180,
    CPU_TYPE_R2KA,
    CPU_TYPE_R3K,
    CPU_TYPE_R4K,
    CPU_TYPE_8080,
    CPU_TYPE_8085,
    CPU_TYPE_GBZ80,
    CPU_TYPE_EZ80,
    CPU_TYPE_KC160,
    CPU_TYPE_IXIY,
    CPU_TYPE_STRICT,
    CPU_TYPE_SIZE
};


typedef struct arg_s arg_t;

struct arg_s {
    char  *name;
    int    flags;
    void(*setfunc)(arg_t *arg, char *);
    void  *data;
    int   *num_ptr;
    char  *help;
    char  *defvalue;
};


typedef struct cpu_map_s cpu_map_t;

struct cpu_map_s {
    char *tool[CPU_MAP_TOOL_SIZE];
};


typedef struct pragma_m4_s pragma_m4_t;

struct pragma_m4_s {
    int         seen;
    const char *pragma;
    const char *m4_name;
};

struct tokens_list_s
{
    char                *token;
    char                 *path;
    struct tokens_list_s *next;
};

/* All our function prototypes */

static void            add_option_to_compiler(char *arg);
static struct tokens_list_s*    gather_from_list_file(char *filename);
static void            add_file_to_process(char *filename, char process_extension);

static void cmd_line_to_tokens(char* line, const char *path, struct tokens_list_s** tokens);
static void cmd_free_tokens(struct tokens_list_s* tokens);

static void            SetNumber(arg_t *argument, char *arg);
static void            SetStringConfig(arg_t *argument, char *arg);
static void            LoadConfigFile(arg_t *argument, char *arg);
static void            parse_cmdline_arg(char *arg);
static void            AddPreProc(option *arg, char *);
static void            AddPreProcIncPath(option *arg, char *);
static void            AddToArgs(option *arg, char *);
static void            AddToArgsQuoted(option *arg, char *);
static void            AddLinkLibrary(option *arg, char *);
static void            AddLinkSearchPath(option *arg, char *);
static void            usage(const char *program);
static void            print_help_text(const char *program);
static void            GlobalDefc(option *argument, char *);
static void            Alias(option *arg, char *);
static void            PragmaDefine(option *arg, char *);
static void            PragmaExport(option *arg, char *);
static void            PragmaRedirect(option *arg, char *);
static void            PragmaNeed(option *arg, char *);
static void            PragmaBytes(option *arg, char *);
static void            PragmaString(option *arg, char *);
static void            PragmaInclude(option *arg, char *);
static void            AddArray(arg_t *arg, char *);
static void            conf_opt_code_speed(option *arg, char *);
static void            write_zcc_defined(char *name, int value, int export);

static void           *mustmalloc(size_t);
static char           *muststrdup(const char *s);
static char           *zcc_strstrip(char *s);
static char           *zcc_strrstrip(char *s);
static char           *zcc_ascii_only(char *s);
static int             hassuffix(char *file, char *suffix_to_check);
static char           *stripsuffix(char *, char *);
static char           *changesuffix(char *, char *);
static char           *find_file_ext(char *filename);
static int             is_path_absolute(char *filename);
static int             process(char *, char *, char *, char *, enum iostyle, int, int, int);
static int             linkthem(char *);
static int             get_filetype_by_suffix(char *);
static void            BuildAsmLine(char *, size_t, char *);
static void            parse_cmdline_arg(char *option);
static void            BuildOptions(char **, char *);
static void            BuildOptions_start(char **, char *);
static void            BuildOptionsQuoted(char **, char *);
static void            copy_output_files_to_destdir(char *suffix, int die_on_fail);
static void            parse_configfile(const char *config_line);
static void            parse_configfile_line(char *config_line);
static void            KillEOL(char *line);
static int             add_variant_args(char *wanted, int num_choices, char **choices);

static void            configure_assembler(void);
static void            configure_compiler(void);
static void            configure_misc_options(void);
static void            configure_maths_library(char **libstring);

static void            apply_copt_rules(int filenumber, int num, char **rules, char *ext1, char *ext2, char *ext);
static void            zsdcc_asm_filter_comments(int filenumber, char *ext);
static void            remove_temporary_files(void);
static void            remove_file_with_extension(char *file, char *suffix);
static int             copyprepend_file(char *src, char *src_extension, char *dest, char *dest_extension, char *prepend);
static int             copy_file(char *src, char *src_extension, char *dest, char *dest_extension);
static int             prepend_file(char *src, char *src_extension, char *dest, char *dest_extension, char *prepend);
static int             copy_defc_file(char *name1, char *ext1, char *name2, char *ext2);
static void            tempname(char *);
static void            find_zcc_config_fileFile(const char *program, char *arg, char *buf, size_t buflen);
static void            parse_option(char *option);
static void            add_zccopt(char *fmt, ...);
static char           *replace_str(const char *str, const char *old, const char *new);
static void            setup_default_configuration(void);
static void            print_specs(void);
static int             isquote(unsigned char c);
static char           *qstrtok(char *s, const char *delim);
static char           *strip_inner_quotes(char *p);
static char           *strip_outer_quotes(char *p);
static int             zcc_asprintf(char **s, const char *fmt, ...);
static int             zcc_getdelim(char **lineptr, unsigned int *n, int delimiter, FILE *stream);
static char           *expand_macros(char *arg);

struct explicit_extension
{
    char* filename;
    char* extension;
    UT_hash_handle hh;
};

static int             createapp = 0;    /* Go the next stage and create the app */
static int             z80verbose = 0;
static int             cleanup = 1;
static int             assembleonly = 0;
static int             lstcwd = 0;
static int             compileonly = 0;
static int             m4only = 0;
static int             clangonly = 0;
static int             llvmonly = 0;
static int             makelib = 0;
static int             explicit_file_type_c = 0;
static struct explicit_extension* explicit_extensions = NULL;
static int             build_bin = 0;
static int             c_code_in_asm = 0;
static int             opt_code_size = 0;
static int             zopt = 0;
static int             verbose = 0;
static int             peepholeopt = 0;
static int             sdccpeepopt = 0;
static int             symbolson = 0;
static int             lston = 0;
static int             mapon = 0;
static int             globaldefon = 0;
static char           *globaldefrefile = NULL;
static int             preprocessonly = 0;
static int             printmacros = 0;
static int             relocate = 0;
static int             relocinfo = 0;
static int             sdcc_signed_char = 0;
static int             swallow_M = 0;
static int             c_print_specs = 0;
static int             c_zorg = -1;
static int             c_sccz80_inline_ints = 0;
static int             gargc;
/* filelist has to stay as ** because we change suffix all the time */
static int             nfiles = 0;
static char          **filelist = NULL;              /* Working filenames   */
static char          **original_filenames = NULL;    /* Original filenames  */
static char          **temporary_filenames = NULL;   /* Temporary filenames */
static char           *outputfile = NULL;
static char           *c_linker_output_file = NULL;
static char           *cpparg;
static char           *cpp_incpath_first;
static char           *cpp_incpath_last;
static char           *comparg;
static char           *clangarg;
static char           *clangcpparg = "-D\"__attribute__(x)= \" -D\"__builtin_unreachable(x)= \" ";
static char           *llvmopt;
static char           *llvmarg;
static char           *linkargs;
static char           *linker_libpath_first;
static char           *linker_libpath_last;
static char           *linklibs;
static char           *linker_linklib_first;
static char           *linker_linklib_last;
static char           *asmargs;
static char           *appmakeargs;
static char           *sccz80arg = NULL;
static char           *sdccarg = NULL;
static char           *m4arg = NULL;
static char           *coptarg = NULL;
static char           *pragincname = NULL;  /* File containing pragmas to append to zcc_opt.def */
static char           *zccopt = NULL;       /* Text to append to zcc_opt.def */
static char           *c_subtype = NULL;
static char           *c_clib = NULL;
static int             c_startup = -2;
static int             c_startupoffset = -1;
static int             c_nostdlib = 0;
static int             c_cpu = 0;
static int             c_nocrt = 0;
static char           *c_crt_incpath = NULL;
static int             processing_user_command_line_arg = 0;
static char            c_sccz80_r2l_calling;
static char            c_copy_m4_processed_files;

static char            filenamebuf[FILENAME_MAX + 1];
#ifdef WIN32
static char            tmpnambuf[FILENAME_MAX+1];
#endif

#define ASM_Z80ASM     0
#define IS_ASM(x)  ( assembler_type == (x) )
static int             assembler_type = ASM_Z80ASM;
static enum iostyle    assembler_style = outimplied;
static int             linker_output_separate_arg = 0;

static enum iostyle    compiler_style = outimplied;

#define CC_SCCZ80    0
#define CC_SDCC      1
#define CC_EZ80CLANG 2
static char           *c_compiler_type = "sccz80";
static int             compiler_type = CC_SCCZ80;

static char           *zcc_opt_dir = ".";
static char           *zcc_opt_def = "zcc_opt.def";

static char           *defaultout = "a.bin";


static char           *cfg_path = ".";

#define AF_BOOL_TRUE      1
#define AF_BOOL_FALSE     2
#define AF_MORE           4
#define AF_DEPRECATED     8


static char  *c_zcc_cfg = NULL;
static char *c_binary_dir = "";
static char  *c_install_dir = PREFIX "/";
static char  *c_options = NULL;

static char  *c_z80asm_exe = "z88dk-z80asm";

static char  *c_ez80clang_exe = "ez80-clang";
static char  *c_clang_exe = "zclang";
static char  *c_llvm_exe = "zllvm-cbe";
static char  *c_sdcc_exe = "z88dk-zsdcc";
static char  *c_sccz80_exe = "z88dk-sccz80";
static char  *c_cpp_exe = "z88dk-ucpp";
static char  *c_sdcc_preproc_exe = "z88dk-ucpp";
static char  *c_zpragma_exe = "z88dk-zpragma";
static char  *c_copt_exe = "z88dk-copt";
static char  *c_appmake_exe = "z88dk-appmake";
#ifndef WIN32
static char  *c_copycmd = "cat";
#else
static char  *c_copycmd = "type";
#endif
static char  *c_extension_config = "o";
static char  *c_incpath = NULL;
static char  *c_clangincpath = NULL;
static char  *c_m4opts = NULL;
static char  *c_coptrules1 = NULL;
static char  *c_coptrules2 = NULL;
static char  *c_coptrules3 = NULL;
static char  *c_coptrules9 = NULL;
static char  *c_coptrules_user = NULL;
static char  *c_coptrules_sccz80 = NULL;
static char  *c_coptrules_target = NULL;
static char  *coptrules_cpu = NULL;
static char  *c_ez80clang_opt = NULL;
static char  *c_sdccopt1 = NULL;
static char  *c_sdccopt2 = NULL;
static char  *c_sdccopt3 = NULL;
static char  *c_sdccopt9 = NULL;
static char  *c_sdccpeeph0 = NULL;
static char  *c_sdccpeeph1 = NULL;
static char  *c_sdccpeeph2 = NULL;
static char  *c_sdccpeeph3 = NULL;
static char  *c_sdccpeeph0cs = NULL;
static char  *c_sdccpeeph1cs = NULL;
static char  *c_sdccpeeph2cs = NULL;
static char  *c_sdccpeeph3cs = NULL;
static char  *c_crt0 = NULL;
static char  *c_linkopts = NULL;
static char  *c_asmopts = NULL;
static char  *c_altmathlib = NULL;
static char  *c_altmathflags = NULL;        /* "-math-z88 -D__NATIVE_MATH__"; */
static char  *c_startuplib = "z80_crt0";
static char  *c_genmathlib = "genmath@{ZCC_LIBCPU}";
static int    c_stylecpp = outspecified;

static char  *c_extension = NULL;
static char  *c_assembler = NULL;
static char  *c_linker = NULL;
static char  *c_compiler = NULL;
static char **c_subtype_array = NULL;
static int    c_subtype_array_num = 0;
static char **c_clib_array = NULL;
static int    c_clib_array_num = 0;
static char **c_aliases_array = NULL;
static int    c_aliases_array_num = 0;

static char **aliases = NULL;
static int    aliases_num = 0;

static char   c_generate_debug_info = 0;
static char   c_help = 0;

static arg_t  config[] = {
    { "OPTIONS", 0, SetStringConfig, &c_options, NULL, "Extra options for port" },
    { "CPP", 0, SetStringConfig, &c_cpp_exe, NULL, "Name of the cpp binary" },
    { "SDCPP", 0, SetStringConfig, &c_sdcc_preproc_exe, NULL, "Name of the SDCC cpp binary" },
    { "STYLECPP", 0, SetNumber, &c_stylecpp, NULL, "" },
    { "ZPRAGMAEXE", 0, SetStringConfig, &c_zpragma_exe, NULL, "Name of the zpragma binary" },

    { "Z80EXE", 0, SetStringConfig, &c_z80asm_exe, NULL, "Name of the z80asm binary" },
    { "LINKOPTS", 0, SetStringConfig, &c_linkopts, NULL, "Options for z80asm as linker", " -L\"DESTDIR/lib/clibs\" -I\"DESTDIR/lib\" " },
    { "ASMOPTS", 0, SetStringConfig, &c_asmopts, NULL, "Options for z80asm as assembler", "-I\"DESTDIR/lib\"" },

    { "COMPILER", AF_DEPRECATED, SetStringConfig, &c_compiler, NULL, "Name of sccz80 binary (use SCCZ80EXE)" },
    { "SCCZ80EXE", 0, SetStringConfig, &c_sccz80_exe, NULL, "Name of sccz80 binary" },
    { "ZSDCCEXE", 0, SetStringConfig, &c_sdcc_exe, NULL, "Name of the sdcc binary" },
    { "ZCLANGEXE", 0, SetStringConfig, &c_clang_exe, NULL, "Name of the clang binary" },
    { "ZLLVMEXE", 0, SetStringConfig, &c_llvm_exe, NULL, "Name of the llvm-cbe binary" },

    { "APPMAKEEXE", 0, SetStringConfig, &c_appmake_exe, NULL, "" },
    { "APPMAKER", AF_DEPRECATED, SetStringConfig, &c_appmake_exe, NULL, "Name of the applink binary (use APPMAKEEXE)" },

    { "COPTEXE", 0, SetStringConfig, &c_copt_exe, NULL, "" },
    { "COPYCMD", 0, SetStringConfig, &c_copycmd, NULL, "" },

    { "INCPATH", 0, SetStringConfig, &c_incpath, NULL, "", "-isystem\"DESTDIR/include\" " },
    { "CLANGINCPATH", 0, SetStringConfig, &c_clangincpath, NULL, "", "-isystem \"DESTDIR/include/_DEVELOPMENT/clang\" " },
    { "M4OPTS", 0, SetStringConfig, &c_m4opts, NULL, "", " -I \"DESTDIR/src/m4\" " },
    { "COPTRULES1", 0, SetStringConfig, &c_coptrules1, NULL, "", "\"DESTDIR/lib/z80rules.1\"" },
    { "COPTRULES2", 0, SetStringConfig, &c_coptrules2, NULL, "", "\"DESTDIR/lib/z80rules.2\"" },
    { "COPTRULES3", 0, SetStringConfig, &c_coptrules3, NULL, "", "\"DESTDIR/lib/z80rules.0\"" },
    { "COPTRULES9", 0, SetStringConfig, &c_coptrules9, NULL, "", "\"DESTDIR/lib/z80rules.9\"" },
    { "COPTRULESINLINE", 0, SetStringConfig, &c_coptrules_sccz80, NULL, "Optimisation file for inlining sccz80 ops", "\"DESTDIR/lib/z80rules.8\"" },
    { "COPTRULESTARGET", 0, SetStringConfig, &c_coptrules_target, NULL, "Optimisation file for target specific operations",NULL },
    { "EZ80CLANGRULES", 0, SetStringConfig, &c_ez80clang_opt, NULL, "Rules for ez80 clang", "DESTDIR/lib/clang_rules.1"},
    { "SDCCOPT1", 0, SetStringConfig, &c_sdccopt1, NULL, "", "\"DESTDIR/libsrc/_DEVELOPMENT/sdcc_opt.1\"" },
    { "SDCCOPT2", 0, SetStringConfig, &c_sdccopt2, NULL, "", "\"DESTDIR/libsrc/_DEVELOPMENT/sdcc_opt.2\"" },
    { "SDCCOPT3", 0, SetStringConfig, &c_sdccopt3, NULL, "", "\"DESTDIR/libsrc/_DEVELOPMENT/sdcc_opt.3\"" },
    { "SDCCOPT9", 0, SetStringConfig, &c_sdccopt9, NULL, "", "\"DESTDIR/libsrc/_DEVELOPMENT/sdcc_opt.9\"" },
    { "SDCCPEEP0", 0, SetStringConfig, &c_sdccpeeph0, NULL, "", " --no-peep --peep-file \"DESTDIR/libsrc/_DEVELOPMENT/sdcc_peeph.0\"" },
    { "SDCCPEEP1", 0, SetStringConfig, &c_sdccpeeph1, NULL, "", " --no-peep --peep-file \"DESTDIR/libsrc/_DEVELOPMENT/sdcc_peeph.1\"" },
    { "SDCCPEEP2", 0, SetStringConfig, &c_sdccpeeph2, NULL, "", " --no-peep --peep-file \"DESTDIR/libsrc/_DEVELOPMENT/sdcc_peeph.2\"" },
    { "SDCCPEEP3", 0, SetStringConfig, &c_sdccpeeph3, NULL, "", " --no-peep --peep-file \"DESTDIR/libsrc/_DEVELOPMENT/sdcc_peeph.3\"" },
    { "SDCCOPTSZ0", 0, SetStringConfig, &c_sdccpeeph0cs, NULL, "", " --no-peep --peep-file \"DESTDIR/libsrc/_DEVELOPMENT/sdcc_peeph_cs.0\"" },
    { "SDCCOPTSZ1", 0, SetStringConfig, &c_sdccpeeph1cs, NULL, "", " --no-peep --peep-file \"DESTDIR/libsrc/_DEVELOPMENT/sdcc_peeph_cs.1\"" },
    { "SDCCOPTSZ2", 0, SetStringConfig, &c_sdccpeeph2cs, NULL, "", " --no-peep --peep-file \"DESTDIR/libsrc/_DEVELOPMENT/sdcc_peeph_cs.2\"" },
    { "SDCCOPTSZ3", 0, SetStringConfig, &c_sdccpeeph3cs, NULL, "", " --no-peep --peep-file \"DESTDIR/libsrc/_DEVELOPMENT/sdcc_peeph_cs.3\"" },
    { "CRT0", 0, SetStringConfig, &c_crt0, NULL, "" },

    { "ALTMATHLIB", 0, SetStringConfig, &c_altmathlib, NULL, "Name of the alt maths library" },
    { "ALTMATHFLG", 0, SetStringConfig, &c_altmathflags, NULL, "Additional options for non-generic maths" },
    { "Z88MATHLIB", AF_DEPRECATED, SetStringConfig, &c_altmathlib, NULL, "Name of the alt maths library (use ALTMATHLIB)" },
    { "Z88MATHFLG", AF_DEPRECATED, SetStringConfig, &c_altmathflags, NULL, "Additional options for non-generic maths (use ALTMATHFLG)" },
    { "STARTUPLIB", 0, SetStringConfig, &c_startuplib, NULL, "" },
    { "GENMATHLIB", 0, SetStringConfig, &c_genmathlib, NULL, "" },
    { "SUBTYPE",  0, AddArray, &c_subtype_array, &c_subtype_array_num, "Add a sub-type alias and config" },
    { "CLIB",  0, AddArray, &c_clib_array, &c_clib_array_num, "Add a clib variant config" },
    { "ALIAS",  0, AddArray, &c_aliases_array, &c_aliases_array_num, "Add an alias and options" },
    { "INCLUDE", 0, LoadConfigFile, NULL, NULL, "Load a configuration file"},
    { "", 0, NULL, NULL }
};

static option options[] = {
    { 'v', "verbose", OPT_BOOL,  "Output all commands that are run (-vn suppresses)" , &verbose, NULL, 0},
    { 'h', "help", OPT_BOOL,  "Display this text" , &c_help, NULL, 0},
    { 0, "o", OPT_STRING,  "Set the basename for linker output files" , &outputfile, NULL, 0},
    { 0, "specs", OPT_BOOL,  "Print out compiler specs" , &c_print_specs, NULL, 0},

    { 0, "", OPT_HEADER, "CPU Targetting:", NULL, NULL, 0 },
    { 0, "m8080", OPT_ASSIGN|OPT_INT, "Generate output for the i8080", &c_cpu, NULL, CPU_TYPE_8080 },
    { 0, "m8085", OPT_ASSIGN|OPT_INT, "Generate output for the i8085", &c_cpu, NULL, CPU_TYPE_8085 },
    { 0, "mz80", OPT_ASSIGN|OPT_INT, "Generate output for the z80", &c_cpu, NULL, CPU_TYPE_Z80 },
    { 0, "mz80_ixiy", OPT_ASSIGN|OPT_INT, "Generate output for the z80 with ix/iy swap", &c_cpu, NULL, CPU_TYPE_IXIY },
    { 0, "mz80_strict", OPT_ASSIGN|OPT_INT, "Generate output for the documented z80", &c_cpu, NULL, CPU_TYPE_STRICT },
    { 0, "mz80n", OPT_ASSIGN|OPT_INT, "Generate output for the z80n", &c_cpu, NULL, CPU_TYPE_Z80N },
    { 0, "mz180", OPT_ASSIGN|OPT_INT, "Generate output for the z180", &c_cpu, NULL, CPU_TYPE_Z180 },
    { 0, "mr2ka", OPT_ASSIGN|OPT_INT, "Generate output for the Rabbit 2000", &c_cpu, NULL, CPU_TYPE_R2KA },
    { 0, "mr3k", OPT_ASSIGN|OPT_INT, "Generate output for the Rabbit 3000", &c_cpu, NULL, CPU_TYPE_R3K },
    { 0, "mr4k", OPT_ASSIGN|OPT_INT, "Generate output for the Rabbit 4000", &c_cpu, NULL, CPU_TYPE_R4K },
    { 0, "mgbz80", OPT_ASSIGN|OPT_INT, "Generate output for the gbz80", &c_cpu, NULL, CPU_TYPE_GBZ80 },
    { 0, "mez80_z80", OPT_ASSIGN|OPT_INT, "Generate output for the ez80 (z80 mode)", &c_cpu, NULL, CPU_TYPE_EZ80 },
    { 0, "mkc160", OPT_ASSIGN|OPT_INT, "Generate output for the KC160 (z80 mode)", &c_cpu, NULL, CPU_TYPE_KC160 },

    { 0, "", OPT_HEADER, "Target options:", NULL, NULL, 0 },
    { 0, "subtype", OPT_STRING,  "Set the target subtype" , &c_subtype, NULL, 0},
    { 0, "clib", OPT_STRING,  "Set the target clib type" , &c_clib, NULL, 0},
    { 0, "crt0", OPT_STRING,  "Override the crt0 assembler file to use" , &c_crt0, NULL, 0},
    { 0, "startuplib", OPT_STRING,  "Override STARTUPLIB - compiler base support routines" , &c_startuplib, NULL, 0},
    { 0, "no-crt", OPT_BOOL|OPT_DOUBLE_DASH,  "Link without crt0 file" , &c_nocrt, NULL, 0},
    { 0, "startupoffset", OPT_INT|OPT_PRIVATE,  "Startup offset value (internal)" , &c_startupoffset, NULL, 0},
    { 0, "startup", OPT_INT,  "Set the startup type" , &c_startup, NULL, 0},
    { 0, "zorg", OPT_INT,  "Set the origin (only certain targets)" , &c_zorg, NULL, 0},
    { 0, "nostdlib", OPT_BOOL,  "If set ignore INCPATH, STARTUPLIB", &c_nostdlib, NULL, 0},
    { 0, "pragma-redirect", OPT_FUNCTION,  "Redirect a function" , NULL, PragmaRedirect, 0},
    { 0, "pragma-define", OPT_FUNCTION,  "Define the option in zcc_opt.def" , NULL, PragmaDefine, 0},
    { 0, "pragma-output", OPT_FUNCTION,  "Define the option in zcc_opt.def (same as above)" , NULL, PragmaDefine, 0},
    { 0, "pragma-export", OPT_FUNCTION,  "Define the option in zcc_opt.def and export as public" , NULL, PragmaExport, 0},
    { 0, "pragma-need", OPT_FUNCTION,  "NEED the option in zcc_opt.def" , NULL, PragmaNeed, 0},
    { 0, "pragma-bytes", OPT_FUNCTION,  "Dump a sequence of bytes zcc_opt.def" , NULL, PragmaBytes, 0},
    { 0, "pragma-string", OPT_FUNCTION,  "Dump a string zcc_opt.def" , NULL, PragmaString, 0},
    { 0, "pragma-include", OPT_FUNCTION,  "Process include file containing pragmas" , NULL, PragmaInclude, 0},

    { 0, "", OPT_HEADER, "Lifecycle options:", NULL, NULL, 0 },
    { 0, "m4", OPT_BOOL,  "Stop after processing m4 files" , &m4only, NULL, 0},
    { 'E', "preprocess-only", OPT_BOOL|OPT_DOUBLE_DASH,  "Stop after preprocessing files" , &preprocessonly, NULL, 0},
    { 0, "dD", OPT_BOOL,  "Print macro definitions in -E mode in addition to normal output" , &printmacros, NULL, 0},
    { 'c', "compile-only", OPT_BOOL|OPT_DOUBLE_DASH,  "Stop after compiling .c .s .asm files to .o files" , &compileonly, NULL, 0},
    { 'a', "assemble-only", OPT_BOOL|OPT_DOUBLE_DASH,  "Stop after compiling .c .s files to .asm files" , &assembleonly, NULL, 0},
    { 'S', "assemble-only", OPT_BOOL|OPT_DOUBLE_DASH,  "Stop after compiling .c .s files to .asm files" , &assembleonly, NULL, 0},
    { 'x', NULL, OPT_BOOL,  "Make a library out of source files" , &makelib, NULL, 0},
    { 0, "xc", OPT_BOOL,  "Explicitly specify file type as C" , &explicit_file_type_c, NULL, 0},
    { 0, "create-app", OPT_BOOL,  "Run appmake on the resulting binary to create emulator usable file" , &createapp, NULL, 0},


    { 0, "", OPT_HEADER, "M4 options:", NULL, NULL, 0 },
    { 0, "Cm", OPT_FUNCTION,  "Add an option to m4" , &m4arg, AddToArgs, 0},
    { 0, "copy-back-after-m4", OPT_BOOL, "Copy files back after processing with m4",&c_copy_m4_processed_files, NULL, 0 },

    { 0, "", OPT_HEADER, "Preprocessor options:", NULL, NULL, 0 },
    { 0, "Cp", OPT_FUNCTION,  "Add an option to the preprocessor" , &cpparg, AddToArgs, 0},
    { 0, "D", OPT_FUNCTION|OPT_INCLUDE_OPT,  "Define a preprocessor option" , NULL, AddPreProc, 0},
    { 0, "U", OPT_FUNCTION|OPT_INCLUDE_OPT,  "Undefine a preprocessor option" , NULL, AddPreProc, 0},
    { 0, "I", OPT_FUNCTION|OPT_INCLUDE_OPT,  "Add an include directory for the preprocessor" , NULL, AddPreProcIncPath, 0},
    { 0, "iquote", OPT_FUNCTION|OPT_INCLUDE_OPT,  "Add a quoted include path for the preprocessor" , &cpparg, AddToArgsQuoted, 0},
    { 0, "isystem", OPT_FUNCTION|OPT_INCLUDE_OPT,  "Add a system include path for the preprocessor" , &cpparg, AddToArgsQuoted, 0},

    { 0, "", OPT_HEADER, "Compiler (all) options:", NULL, NULL, 0 },
    { 0, "compiler", OPT_STRING,  "Set the compiler type from the command line (sccz80,sdcc)" , &c_compiler_type, NULL, 0},
    { 0, "c-code-in-asm", OPT_BOOL|OPT_DOUBLE_DASH,  "Add C code to .asm files" , &c_code_in_asm, NULL, 0},
    { 0, "opt-code-speed", OPT_FUNCTION|OPT_DOUBLE_DASH|OPT_DEFAULT_VALUE,  "Optimize for code speed" , NULL, conf_opt_code_speed, (intptr_t)"all"},
    { 0, "debug", OPT_BOOL, "Enable debugging support", &c_generate_debug_info, NULL, 0 },
    { 0, "", OPT_HEADER, "Compiler (sccz80) options:", NULL, NULL, 0 },
    { 0, "Cc", OPT_FUNCTION,  "Add an option to sccz80" , &sccz80arg, AddToArgs, 0},
    { 0, "set-r2l-by-default", OPT_BOOL,  "(sccz80) Use r2l calling convention by default", &c_sccz80_r2l_calling, NULL, 0},
    { 0, "O", OPT_INT,  "Set the peephole optimiser setting for copt" , &peepholeopt, NULL, 0},
    { 0, "Ch", OPT_FUNCTION,  "Add an option to the sccz80 peepholer" , &coptarg, AddToArgs, 0},
    { 0, "", OPT_HEADER, "Compiler (sdcc) options:", NULL, NULL, 0 },
    { 0, "Cs", OPT_FUNCTION,  "Add an option to sdcc" , &sdccarg, AddToArgs, 0},
    { 0, "opt-code-size", OPT_BOOL|OPT_DOUBLE_DASH,  "Optimize for code size (sdcc only)" , &opt_code_size, NULL, 0},
    { 0, "SO", OPT_INT,  "Set the peephole optimiser setting for sdcc-peephole" , &sdccpeepopt, NULL, 0},
    { 0, "fsigned-char", OPT_BOOL|OPT_DOUBLE_DASH,  "Use signed chars by default" , &sdcc_signed_char, NULL, 0},
    { 0, "", OPT_HEADER, "Compiler (clang/llvm) options:", NULL, NULL, 0 },
    { 0, "Cg", OPT_FUNCTION,  "Add an option to clang" , &clangarg, AddToArgs, 0},
    { 0, "clang", OPT_BOOL,  "Stop after translating .c files to llvm ir" , &clangonly, NULL, 0},
    { 0, "llvm", OPT_BOOL,  "Stop after llvm-cbe generates new .cbe.c files" , &llvmonly, NULL, 0},
    { 0, "Co", OPT_FUNCTION,  "Add an option to llvm-opt" , &llvmopt, AddToArgs, 0},
    { 0, "Cv", OPT_FUNCTION,  "Add an option to llvm-cbe" , &llvmarg, AddToArgs, 0},
    { 0, "zopt", OPT_BOOL,  "Enable llvm-optimizer (clang only)" , &zopt, NULL, 0},
    { 0, "", OPT_HEADER, "Assembler options:", NULL, NULL, 0 },
    { 0, "Ca", OPT_FUNCTION,  "Add an option to the assembler" , &asmargs, AddToArgsQuoted, 0},
    { 0, "z80-verb", OPT_BOOL,  "Make the assembler more verbose" , &z80verbose, NULL, 0},
    { 0, "", OPT_HEADER, "Linker options:", NULL, NULL, 0 },
    { 0, "Cl", OPT_FUNCTION,  "Add an option to the linker" , &linkargs, AddToArgsQuoted, 0},
    { 0, "L", OPT_FUNCTION|OPT_INCLUDE_OPT,  "Add a library search path" , NULL, AddLinkSearchPath, 0},
    { 0, "l", OPT_FUNCTION|OPT_INCLUDE_OPT,  "Add a library" , NULL, AddLinkLibrary, 0},
    { 0, "bn", OPT_STRING,  "Set the output file for the linker stage" , &c_linker_output_file, NULL, 0},
    { 0, "reloc-info", OPT_BOOL,  "Generate binary file relocation information" , &relocinfo, NULL, 0},
    { 'm', "gen-map-file", OPT_BOOL,  "Generate an output map of the final executable" , &mapon, NULL, 0},
    { 's', "gen-symbol-file", OPT_BOOL,  "Generate a symbol map of the final executable" , &symbolson, NULL, 0},
    { 0, "list", OPT_BOOL|OPT_DOUBLE_DASH,  "Generate list files" , &lston, NULL, 0},
    { 'R', NULL, OPT_BOOL|OPT_DEPRECATED,  "Generate relocatable code (deprecated)" , &relocate, NULL, 0},
    { 0, NULL, OPT_HEADER, "Appmake options:", NULL, NULL, 0 },
    { 0, "Cz", OPT_FUNCTION,  "Add an option to appmake" , &appmakeargs, AddToArgs, 0},
   
    { 0, "", OPT_HEADER, "Misc options:", NULL, NULL, 0 },
    { 0, "g", OPT_FUNCTION|OPT_INCLUDE_OPT,  "Generate a global defc file of the final executable (-g -gp -gpf:filename)" , &globaldefrefile, GlobalDefc, 0},
    { 0, "alias", OPT_FUNCTION,  "Define a command line alias" , NULL, Alias, 0},
    { 0, "lstcwd", OPT_BOOL|OPT_DOUBLE_DASH,  "Paths in .lst files are relative to the current working dir" , &lstcwd, NULL, 0},
    { 0, "custom-copt-rules", OPT_STRING,  "Custom user copt rules" , &c_coptrules_user, NULL, 0},
    { 'M', NULL, OPT_BOOL|OPT_PRIVATE,  "Swallow -M option in configs" , &swallow_M, NULL, 0},
    { 0, "vn", OPT_BOOL_FALSE|OPT_PRIVATE,  "Turn off command tracing" , &verbose, NULL, 0},
    { 0, "no-cleanup", OPT_BOOL_FALSE, "Don't cleanup temporary files", &cleanup, NULL, 0 },
    { 0, "", 0, NULL },

};


cpu_map_t cpu_map[CPU_TYPE_SIZE] = {
    {{ "-mz80"   , "-mz80"   , "-mz80"   , "-mz80", "DESTDIR/lib/arch/z80/z80_rules.1", "", "-triple z80"   }},          /* CPU_TYPE_Z80     : CPU_MAP_TOOL_Z80ASM, CPU_MAP_TOOL_SCCZ80, CPU_MAP_TOOL_ZSDCC, CPU_TOOL_COPT, CPU_TOOL_LIBNAME */
    {{ "-mz80n"  , "-mz80n"  , "-mz80n"  , "-mz80n","DESTDIR/lib/arch/z80n/z80n_rules.1", "_z80n", "-triple z80"  }},          /* CPU_TYPE_Z80N    : CPU_MAP_TOOL_Z80ASM, CPU_MAP_TOOL_SCCZ80, CPU_MAP_TOOL_ZSDCC                */
    {{ "-mz180"  , "-mz180"  , "-mz180 -portmode=z180", "-mz180", "DESTDIR/lib/arch/z180/z180_rules.1", "_z180", "-triple z180" }},    /* CPU_TYPE_Z180    : CPU_MAP_TOOL_Z80ASM, CPU_MAP_TOOL_SCCZ80, CPU_MAP_TOOL_ZSDCC, CPU_TOOL_COPT */
    {{ "-mr2ka"  , "-mr2ka"  , "-mr2ka"  , "-mr2ka", "DESTDIR/lib/arch/rabbit/rabbit_rules.1", "_r2ka", NULL  }},          /* CPU_TYPE_R2KA     : CPU_MAP_TOOL_Z80ASM, CPU_MAP_TOOL_SCCZ80, CPU_MAP_TOOL_ZSDCC, CPU_TOOL_COPT */
    {{ "-mr3k"   , "-mr3k"   , "-mr3ka"  , "-mr3k", "DESTDIR/lib/arch/rabbit/rabbit_rules.1", "_r2ka", NULL   }},          /* CPU_TYPE_R3K     : CPU_MAP_TOOL_Z80ASM, CPU_MAP_TOOL_SCCZ80, CPU_MAP_TOOL_ZSDCC, CPU_TOOL_COPT */
    {{ "-mr4k"   , "-mr4k"   , "-mr4ka"  , "-mr4k", "DESTDIR/lib/arch/rabbit/rabbit_rules.1", "_r4k", NULL   }},          /* CPU_TYPE_R4K     : CPU_MAP_TOOL_Z80ASM, CPU_MAP_TOOL_SCCZ80, CPU_MAP_TOOL_ZSDCC, CPU_TOOL_COPT */
    {{ "-m8080"  , "-m8080"  , NULL   , "-m8080", "DESTDIR/lib/arch/8080/8080_rules.1", "_8080", NULL  }},          /* CPU_TYPE_8080    : CPU_MAP_TOOL_Z80ASM, CPU_MAP_TOOL_SCCZ80, CPU_MAP_TOOL_ZSDCC, CPU_TOOL_COPT */
    {{ "-m8085"  , "-m8085"  , NULL   , "-m8085", "DESTDIR/lib/arch/8085/8085_rules.1", "_8085", NULL  }},          /* CPU_TYPE_8085    : CPU_MAP_TOOL_Z80ASM, CPU_MAP_TOOL_SCCZ80, CPU_MAP_TOOL_ZSDCC, CPU_TOOL_COPT */
    {{ "-mgbz80" , "-mgbz80" , "-msm83" , "-mgbz80", "DESTDIR/lib/arch/gbz80/gbz80_rules.1", "_gbz80", NULL }},       /* CPU_TYPE_GBZ80   : CPU_MAP_TOOL_Z80ASM, CPU_MAP_TOOL_SCCZ80, CPU_MAP_TOOL_ZSDCC, CPU_TOOL_COPT */
    {{ "-mez80_z80"   , "-mez80_z80" ,  "-mez80_z80" ,   "-mez80", "DESTDIR/lib/arch/ez80/ez80_rules.1", "_ez80_z80",  "-triple z80" }},           /* CPU_TYPE_EZ80   : CPU_MAP_TOOL_Z80ASM, CPU_MAP_TOOL_SCCZ80, CPU_MAP_TOOL_ZSDCC, CPU_TOOL_COPT */
    {{ "-mkc160" , "-mkc160" ,  "-mz80" , "-mkc160", "DESTDIR/lib/arch/kc160/kc160_rules.1", "_kc160",  "-triple z180" }},           /* CPU_TYPE_KC160   : CPU_MAP_TOOL_Z80ASM, CPU_MAP_TOOL_SCCZ80, CPU_MAP_TOOL_ZSDCC, CPU_TOOL_COPT */
    {{ "-mz80 -IXIY"   , "-mz80"   , "-mz80"   , "-mz80", "DESTDIR/lib/arch/z80/z80_rules.1", "_ixiy",  "-triple z80"   }},          /* CPU_TYPE_IXIY     : CPU_MAP_TOOL_Z80ASM, CPU_MAP_TOOL_SCCZ80, CPU_MAP_TOOL_ZSDCC, CPU_TOOL_COPT, CPU_TOOL_LIBNAME */
    {{ "-mz80_strict",  "-mz80"   , "-mz80"   , "-mz80", "DESTDIR/lib/arch/z80/z80_rules.1", "_strict",  "-triple z80"   }},          /* CPU_TYPE_STRICT  : CPU_MAP_TOOL_Z80ASM, CPU_MAP_TOOL_SCCZ80, CPU_MAP_TOOL_ZSDCC, CPU_TOOL_COPT, CPU_TOOL_LIBNAME */
};

char *select_cpu(int n)
{
    return cpu_map[c_cpu].tool[n];
}


pragma_m4_t important_pragmas[] = {
    { 0, "startup", "__STARTUP" },
    { 0, "startupoffset", "__STARTUP_OFFSET" },
    { 0, "CRT_INCLUDE_DRIVER_INSTANTIATION", "M4__CRT_INCLUDE_DRIVER_INSTANTIATION" },
    { 0, "CRT_ITERM_EDIT_BUFFER_SIZE", "M4__CRT_ITERM_EDIT_BUFFER_SIZE" },
    { 0, "CRT_OTERM_FZX_DRAW_MODE", "M4__CRT_OTERM_FZX_DRAW_MODE" },
    { 0, "CRT_APPEND_MMAP", "M4__CRT_APPEND_MMAP" },
    { 0, "__MMAP", "M4__MMAP" },
};


static void *mustmalloc(size_t n)
{
    void           *p;

    if ((p = malloc(n)) == 0) {
        fprintf(stderr, "malloc failed\n");
        exit(1);
    }
    return (p);
}

static char *muststrdup(const char *s)
{
    char *r;

    if ((r = strdup(s)) == NULL) {
        fprintf(stderr, "strdup failed\n");
        exit(1);
    }
    return r;
}

static char *zcc_strstrip(char *s)
{
    while (isspace(*s)) ++s;
    return zcc_strrstrip(s);
}

static char *zcc_strrstrip(char *s)
{
    char *p;

    for (p = s + strlen(s); p != s; *p = 0)
        if (!isspace(*--p)) break;

    return s;
}

static char *zcc_ascii_only(char *s)
{
    char *p;

    for (p = s; *p; ++p)
        *p &= 0x7f;

    return s;
}

static int hassuffix(char *name, char *suffix)
{
    int             nlen, slen;

    {
        struct explicit_extension* exp = NULL;
        HASH_FIND_STR(explicit_extensions, name, exp);
        if (exp && (strcmp(exp->extension, suffix) == 0)) {
            return 1;
        }
    }

    nlen = strlen(name);
    slen = strlen(suffix);

    if (slen > nlen)
        return (0);
    return (strcmp(&name[nlen - slen], suffix) == 0);
}

static char *stripsuffix(char *name, char *suffix)
{
    char *p, *r;

    /* Recursively strip suffix */
    r = muststrdup(name);
    while (((p = find_file_ext(r)) != NULL) && (strcmp(p, suffix) == 0))
        *p = '\0';
    return r;
}

static char *changesuffix(char *name, char *suffix)
{
    char *p, *r;

    if ((p = find_file_ext(name)) == NULL) {
        r = mustmalloc(strlen(name) + strlen(suffix) + 1);
        sprintf(r, "%s%s", name, suffix);
    }
    else {
        r = mustmalloc(p - name + strlen(suffix) + 1);
        r[0] = '\0';
        strncat(r, name, p - name);
        strcat(r, suffix);
    }

    return (r);
}

static int explicit_file_type_defined(void)
{
    return explicit_file_type_c;
}

static char* get_explicit_file_type(void)
{
    if (explicit_file_type_c) {
        return ".c";
    }

    return NULL;
}

int process(char *suffix, char *nextsuffix, char *processor, char *extraargs, enum iostyle ios, int number, int needsuffix, int src_is_original)
{
    int             status, errs;
    int             tstore;
    char            buffer[8192], *outname;
    char           *bin_dir = c_binary_dir;

    errs = 0;

    if (!hassuffix(filelist[number], suffix))
        return (0);

#ifndef WIN32
    // On non-windows platforms m4 is a system file, so doesn't need a prefix
    if (strcasecmp(processor,"m4") == 0) {
        bin_dir = "";
    }
#endif

    outname = changesuffix(temporary_filenames[number], nextsuffix);

    switch (ios) {
    case outimplied:
        /* Dropping the suffix for Z80..cheating! */
        tstore = strlen(filelist[number]) - strlen(suffix);
        if (!needsuffix)
            filelist[number][tstore] = 0;
        snprintf(buffer, sizeof(buffer), "%s%s %s \"%s\"", bin_dir, processor, extraargs, filelist[number]);
        filelist[number][tstore] = '.';
        break;
    case outspecified:
        snprintf(buffer, sizeof(buffer), "%s%s %s \"%s\" \"%s\"", bin_dir, processor, extraargs, filelist[number], outname);
        break;
    case outspecified_flag:
        snprintf(buffer, sizeof(buffer), "%s%s %s \"%s\" -o \"%s\"", bin_dir, processor, extraargs, filelist[number], outname);
        break;
    case filter:
        snprintf(buffer, sizeof(buffer), "%s%s %s < \"%s\" > \"%s\"", bin_dir, processor, extraargs, filelist[number], outname);
        break;
    case filter_out:
        // This is only used by copy command, which is cat/type so not a z88dk binary
        snprintf(buffer, sizeof(buffer), "%s %s \"%s\" > \"%s\"", processor, extraargs, filelist[number], outname);
        break;
    case filter_outspecified_flag:
        snprintf(buffer, sizeof(buffer), "%s%s %s < \"%s\" -o \"%s\"", bin_dir, processor, extraargs, filelist[number], outname);
        break;
    }

    if (verbose) {
        printf("%s\n", buffer);
        fflush(stdout);
    }

    status = system(buffer);

    if (status != 0) {
        errs = 1;
        free(outname);
    }
    else {
        /* Free up the allocated memory */
        free(filelist[number]);
        filelist[number] = outname;
    }

    return (errs);
}


int linkthem(char *linker)
{
    int             i, len, offs, status;
    char           *temp, *cmdline;
    char            tname[FILENAME_MAX + 1];
    FILE           *out, *prj;

    if (compileonly)
    {
        len = offs = zcc_asprintf(&temp, "%s %s -o\"%s\" %s",
            linker,
            select_cpu(CPU_MAP_TOOL_Z80ASM),
            outputfile,
            linkargs);
    }
    else if (makelib)
    {
        len = offs = zcc_asprintf(&temp, "%s %s %s -d %s %s -x\"%s\"",
            linker,
            (z80verbose && IS_ASM(ASM_Z80ASM)) ? "-v" : "",
            select_cpu(CPU_MAP_TOOL_Z80ASM),
            IS_ASM(ASM_Z80ASM) ? "" : "-Mo ",
            linkargs,
            outputfile);
    }
    else
    {
        /* late assembly for first file which acts as crt */
        if (verbose) printf("\nPROCESSING CRT\n");
        if (process(".asm", c_extension, c_assembler, c_crt_incpath ? c_crt_incpath : "", assembler_style, 0, YES, NO))
            exit(1);
        if (verbose) puts("");
        free(c_crt_incpath);

        len = offs = zcc_asprintf(&temp, "%s %s -b -d %s -o%s\"%s\" %s%s%s%s%s%s%s%s%s",
            linker,
            select_cpu(CPU_MAP_TOOL_Z80ASM),
            IS_ASM(ASM_Z80ASM) ? "" : "-Mo ",
            linker_output_separate_arg ? " " : "",
            outputfile,
            (z80verbose && IS_ASM(ASM_Z80ASM)) ? "-v " : "",
            (relocate && IS_ASM(ASM_Z80ASM)) ? "-R " : "",
            globaldefon ? "-g " : "",
            (createapp || mapon) ? "-m " : "",
            (createapp || symbolson) ? "-s " : "",
            relocinfo ? "--reloc-info " : "",
            linkargs,
            (c_nostdlib == 0) ? c_linkopts : "",
            linklibs);
    }

    tname[0] = '\0';
    prj = z80verbose ? fopen(PROJFILE, "w") : NULL;

    if ((nfiles > 2) && IS_ASM(ASM_Z80ASM))
    {
        /* place source files into a list file for z80asm */

        tempname(tname);
        strcat(tname, ".lst");
        if ((out = fopen(tname, "w")) == NULL) goto USE_COMMANDLINE;

        for (i = 0; i < nfiles; ++i)
        {
            if (hassuffix(filelist[i], c_extension) || hassuffix(filelist[i],"obj"))
            {
                fprintf(out, "%s\n", filelist[i]);
                if (prj) fprintf(prj, "%s\n", original_filenames[i]);
            }
        }

        fclose(out);

        len += strlen(tname) + 5;
        cmdline = calloc(len, sizeof(char));

        snprintf(cmdline, len, "%s \"@%s\"", temp, tname);
    }
    else
    {
    USE_COMMANDLINE:

        /* place source files on the command line */

        for (i = 0; i < nfiles; i++)
            len += strlen(filelist[i]) + 7;
        len++;

        /* So the total length we need is now in len, let's malloc and do it */

        cmdline = calloc(len, sizeof(char));
        strcpy(cmdline, temp);

        for (i = 0; i < nfiles; ++i)
        {
            if (hassuffix(filelist[i], c_extension) )
            {
                offs += snprintf(cmdline + offs, len - offs, " \"%s\"", filelist[i]);
                if (prj) fprintf(prj, "%s\n", original_filenames[i]);
            }
        }
    }

    if (prj) fclose(prj);

    if (verbose) {
        printf("%s\n", cmdline);
        fflush(stdout);
    }
    status = system(cmdline);

    if (cleanup && strlen(tname))
        remove_file_with_extension(tname, ".lst");

    free(cmdline);
    free(temp);

    return (status);
}

int main(int argc, char **argv)
{
    int             i, ft;
    char           *ptr;
    char            config_filename[FILENAME_MAX + 1];
    char            asmarg[4096];    /* Hell, that should be long enough! */
    char            buffer[LINEMAX + 1];    /* For reading in option file */
    FILE           *fp;

#ifdef WIN32
    /* Randomize temporary filenames for windows (it may end up in cwd)  */
    snprintf(tmpnambuf, sizeof(tmpnambuf), "zcc%08X%04X",_getpid(),  ((unsigned int)time(NULL)) & 0xffff);
#endif

    processing_user_command_line_arg = 0;

    asmargs = linkargs = cpparg = clangarg = llvmopt = llvmarg = NULL;
    linklibs = muststrdup("");

    cpp_incpath_first = cpp_incpath_last = NULL;
    linker_libpath_first = linker_libpath_last = NULL;
    linker_linklib_first = linker_linklib_last = NULL;

    atexit(remove_temporary_files);
    add_option_to_compiler("");

    if (argc == 1) {
        print_help_text(argv[0]);
        exit(1);
    }

    // If argv[0] is a path and not "." and not the system install path, then we can
    // setup variables based on that
    snprintf(buffer,sizeof(buffer),"%s",argv[0]);
    ptr = zdirname(buffer);
    if ( strcmp(ptr,".") && strcmp(ptr, BINDIR)) {
        char ccc[FILENAME_MAX+1];
        snprintf(ccc, sizeof(ccc), "%s/../", ptr);
        c_install_dir = muststrdup(ccc);
        snprintf(ccc, sizeof(ccc), "%s/../lib/config/", ptr);
        c_zcc_cfg = muststrdup(ccc);
        snprintf(ccc, sizeof(ccc), "%s/", ptr);
        c_binary_dir = muststrdup(ccc);
    }

    /* Setup the install prefix based on ZCCCFG */
    if ((ptr = getenv("ZCCCFG")) != NULL) {
        char ccc[FILENAME_MAX+1];

        c_zcc_cfg = muststrdup(ptr);
#ifdef WIN32
        snprintf(ccc, sizeof(ccc), "%s\\..\\..\\", ptr);
#else
        snprintf(ccc, sizeof(ccc), "%s/../../", ptr);
#endif
        c_install_dir = muststrdup(ccc);
    }

    setup_default_configuration();

    char configuration[1024] = {0};

    for (gargc = 0 ; gargc < argc; gargc++) {
        char* aa = argv[gargc];
        if (aa[0] == '+') {
            strcpy(configuration, aa);
            break;
        } else if (aa[0] == '@') {
            struct tokens_list_s* tokens = gather_from_list_file(aa + 1);

            struct tokens_list_s* token;
            LL_FOREACH(tokens, token) {
                char* tk = token->token;
                if (tk[0] == '+') {
                    strcpy(configuration, tk);
                }
            }
            cmd_free_tokens(tokens);
        }
    }

    if (strlen(configuration) == 0) {
        fprintf(stderr, "A config file must be specified with +file\n\n");
        print_help_text(argv[0]);
        exit(1);
    }

    find_zcc_config_fileFile(argv[0], configuration, config_filename, sizeof(config_filename));
    cfg_path = zdirname(strdup(config_filename));
    parse_configfile(config_filename);
    

    /* Now, parse the default options list */
    if (c_options != NULL) {
        parse_option(muststrdup(c_options));
    }

    /* Add in any aliases defined by the config file */
    for ( i = 0; i < c_aliases_array_num; i++ ) {
        char buf[LINEMAX+1];
        char *ptr = c_aliases_array[i];
        char *dest = buf;
        while ( *ptr && !isspace(*ptr)) {
            *dest++ = *ptr++;
        }
        *dest = 0;
        while ( isspace(*ptr) )
           ptr++;
        aliases = realloc(aliases, (aliases_num + 2) * sizeof(aliases[0]));
        aliases[aliases_num++] = strdup(buf);
        aliases[aliases_num++] = strdup(ptr);
    }

    processing_user_command_line_arg = 1;
    argc = option_parse(&options[0], argc, argv);
    for (gargc = 0; gargc < argc; gargc++) {
        char* aa = argv[gargc];
        // We have some options left over, it may well be an alias
        if (aa[0] == '-') {
            parse_cmdline_arg(aa);
        } else if (aa[0] != '+') {
            add_file_to_process(aa, 1);
        }
    }
    processing_user_command_line_arg = 0; 

    if (c_print_specs) {
        print_specs();
        exit(0);
    }


    if (add_variant_args(c_subtype, c_subtype_array_num, c_subtype_array) == -1) {
        fprintf(stderr, "Cannot find definition for target -subtype=%s\n", c_subtype);
        exit(1);
    }
    if (add_variant_args(c_clib, c_clib_array_num, c_clib_array) == -1) {
        fprintf(stderr, "Cannot find definition for -clib=%s\n", c_clib);
        exit(1);
    }

    /* We must have at least a crt file - we can rely on defaults for everything else */
    if (c_crt0 == NULL) {
        fprintf(stderr, "No CRT0 defined in configuration file <%s>\n", config_filename);
        exit(1);
    }

    /* Setup zcc_opt.def */
    {
        char tempdir[FILENAME_MAX+1];

        unlink("zcc_opt.def");
#ifndef WIN32
        char* ret = NULL;

        while ( ret == NULL ) {
            snprintf(tempdir, sizeof(tempdir),"/tmp/tmpzccXXXXXXXX");
            ret = mkdtemp(tempdir);
        }
#else
        int ret = -1;

        while ( ret != 0 ) {
            tempname(tempdir);
            ret = mkdir(tempdir);
        }
#endif
        zcc_opt_dir = strdup(tempdir);
        strcat(tempdir,"/zcc_opt.def");
        zcc_opt_def = strdup(tempdir);
    }



    if (c_sccz80_inline_ints == 0 ) {
        c_coptrules_sccz80 = NULL;
    }

    // Find the CPU specific rules
    {
        char *rules = select_cpu(CPU_MAP_TOOL_CPURULES);
        if ( rules ) {
            char *expanded = expand_macros(rules);
            struct stat sb;

            if (stat(expanded, &sb) == 0 && S_ISREG(sb.st_mode)) {
                coptrules_cpu = expanded;
            }
        }
    }

    configure_assembler();
    configure_compiler();
    configure_misc_options();

    if (c_nostdlib == 0) {
        /* Add the startup library to the linker arguments */
        if (c_startuplib && strlen(c_startuplib)) {
            snprintf(buffer, sizeof(buffer), "-l\"%s\" ", c_startuplib);
            AddLinkLibrary(NULL, buffer);
            /* Add the default cpp path */
            AddPreProcIncPath(NULL, c_incpath);
        }
    }

    if (lston) {
        /* list on so add list options to assembler and linker */
        BuildOptions(&asmargs, "-l ");
        BuildOptions(&linkargs, "-l ");
    }

    if (c_zorg != -1) {
        write_zcc_defined("CRT_ORG_CODE", c_zorg, 0);
    }

    if ((fp = fopen(zcc_opt_def, "a")) != NULL) {
        fprintf(fp, "%s", zccopt ? zccopt : "");
        fclose(fp);
    }
    else {
        fprintf(stderr, "Could not create %s: File in use?\n", zcc_opt_def);
        exit(1);
    }

    /* process pragma-include */
    if (pragincname)
    {
        char cmdline[FILENAME_MAX + 128];

#ifdef _WIN32
        snprintf(cmdline, FILENAME_MAX + 127, "%s -zcc-opt=\"%s\" < \"%s\" > nul",c_zpragma_exe, zcc_opt_def, pragincname);
#else
        snprintf(cmdline, FILENAME_MAX + 127, "%s -zcc-opt=\"%s\" < \"%s\" > /dev/null",c_zpragma_exe, zcc_opt_def, pragincname);
#endif
        if (verbose) { printf("%s\n", cmdline); fflush(stdout); }
        if (-1 == system(cmdline)) {
            fprintf(stderr, "Could not execute %s\n", cmdline);
            exit(1);
        }
    }


    if (nfiles <= 0 || c_help) {
        print_help_text(argv[0]);
        exit(0);
    }

    /* Mangle math lib name but only for classic compiles */
    if ((c_clib == NULL) || (!strstr(c_clib, "new") && !strstr(c_clib, "sdcc") && !strstr(c_clib, "clang")))
        if (linker_linklib_first) configure_maths_library(&linker_linklib_first);   /* -lm appears here */

    /* Options that must be sequenced in specific order */
    if (compiler_type == CC_SDCC)
        BuildOptions_start(&comparg, "--constseg rodata_compiler ");

    BuildOptions(&clangarg, c_clangincpath);
    if (cpp_incpath_last) {
        BuildOptions(&cpparg, cpp_incpath_last);
        BuildOptions(&clangarg, cpp_incpath_last);
    }
    if (cpp_incpath_first) {
        BuildOptions_start(&cpparg, cpp_incpath_first);
        BuildOptions_start(&clangarg, cpp_incpath_first);
    }
    clangarg = replace_str(ptr = clangarg, "-I", "-idirafter ");
    free(ptr);
    clangarg = replace_str(ptr = clangarg, "-D__SDCC", "-D__CLANG");
    free(ptr);

    BuildOptions(&linker_libpath_first, "-L. ");
    if (linker_libpath_last)
        BuildOptions(&linker_libpath_first, linker_libpath_last);
    BuildOptions_start(&linkargs, linker_libpath_first);

    if (linker_linklib_last)
        BuildOptions(&linker_linklib_first, linker_linklib_last);
    if (linker_linklib_first)
        BuildOptions_start(&linklibs, linker_linklib_first);

    /* CLANG & LLVM options */
    BuildOptions_start(&clangarg, "--target=sdcc-z80 -S -emit-llvm ");
    if (!sdcc_signed_char) BuildOptions_start(&clangarg, "-fno-signed-char ");
    BuildOptions(&llvmarg, llvmarg ? "-disable-partial-libcall-inlining " : "-O2 -disable-partial-libcall-inlining ");
    BuildOptions(&llvmopt, llvmopt ? "-disable-simplify-libcalls -disable-loop-vectorization -disable-slp-vectorization -S " : "-O2 -disable-simplify-libcalls -disable-loop-vectorization -disable-slp-vectorization -S ");

    if (printmacros)
    {
        BuildOptions(&cpparg, "-d");
    }

    /* Peephole optimization level for sdcc */
    if (compiler_type == CC_SDCC && c_cpu != CPU_TYPE_GBZ80)
    {
        switch (sdccpeepopt)
        {
        case 0:
            add_option_to_compiler(opt_code_size ? c_sdccpeeph0cs : c_sdccpeeph0);
            break;
        case 1:
            add_option_to_compiler(opt_code_size ? c_sdccpeeph1cs : c_sdccpeeph1);
            break;
        case 2:
            add_option_to_compiler(opt_code_size ? c_sdccpeeph2cs : c_sdccpeeph2);
            break;
        default:
            add_option_to_compiler(opt_code_size ? c_sdccpeeph3cs : c_sdccpeeph3);
            break;
        }
    }

    /* m4 include path points to target's home directory */
    if ((ptr = last_path_char(c_crt0)) != NULL) {
        char *p;
        p = mustmalloc((ptr - c_crt0 + 7) * sizeof(char));
        sprintf(p, "-I \"%.*s\"",(int)( ptr - c_crt0), c_crt0);
        BuildOptions(&m4arg, p);
        free(p);
    }

    /* m4 must include zcc_opt_dir */
    {
        char  optdir[FILENAME_MAX+1];

        snprintf(optdir,sizeof(optdir),"-I \"%s\"",zcc_opt_dir);
        BuildOptions(&m4arg, optdir);
    }

    /* m4 include path finds z88dk macro definition file "z88dk.m4" */
    BuildOptions(&m4arg, c_m4opts);

    build_bin = !m4only && !clangonly && !llvmonly && !preprocessonly && !assembleonly && !compileonly && !makelib;

    /* Create M4 defines out of some pragmas for the CRT */
    /* Some pragma values need to be known at m4 time in the new c lib CRTs */

    if (build_bin) {
        if ((fp = fopen(zcc_opt_def, "r")) != NULL) {
            char buffer[LINEMAX + 1];
            int32_t val;

            while (fgets(buffer, LINEMAX, fp) != NULL) {
                for (i = 0; i < sizeof(important_pragmas) / sizeof(*important_pragmas); ++i) {
                    if (important_pragmas[i].seen == 0) {
                        char match[LINEMAX + 1];

                        snprintf(match, sizeof(match), " defc %s = %%" SCNi32, important_pragmas[i].pragma);
                        if (sscanf(buffer, match, &val) == 1) {
                            important_pragmas[i].seen = 1;
                            snprintf(buffer, sizeof(buffer), "--define=%s=%" PRId32, important_pragmas[i].m4_name, val);
                            buffer[sizeof(buffer) - 1] = 0;
                            BuildOptions(&m4arg, buffer);
                        }
                    }
                }
            }
        } else {
            fprintf(stderr, "Could not open %s: File in use?\n", zcc_opt_def);
            exit(1);
        }

        fclose(fp);
    }
    /* Activate target's crt file */
    if ((c_nocrt == 0) && build_bin) {
        /* append target crt to end of filelist */
        add_file_to_process(c_crt0, 0);
        /* move crt to front of filelist */
        ptr = original_filenames[nfiles - 1];
        memmove(&original_filenames[1], &original_filenames[0], (nfiles - 1) * sizeof(*original_filenames));
        original_filenames[0] = ptr;
        ptr = filelist[nfiles - 1];
        memmove(&filelist[1], &filelist[0], (nfiles - 1) * sizeof(*filelist));
        filelist[0] = ptr;
        ptr = temporary_filenames[nfiles - 1];
        memmove(&temporary_filenames[1], &temporary_filenames[0], (nfiles - 1) * sizeof(*temporary_filenames));
        temporary_filenames[0] = ptr;
    }

    /* crt file is now the first file in filelist */
    c_crt0 = temporary_filenames[0];

    /* PLOT TWIST - See #190 on Github                                                                           */
    /* The crt must be processed last because the new c library now processes zcc_opt.def with m4.               */
    /* With the crt first, the other .c files have not been processed yet and zcc_opt.def may not be complete.   */
    /*                                                                                                           */
    /* To solve let's do something awkward.  Process the files starting at index one but go one larger than      */
    /* the number of files.  When the one larger number is hit, set the index to zero to do the crt.             */
    /*                                                                                                           */
    /* This nastiness is marked "HACK" in the loop below.  Maybe something better will come along later.         */

    /* Parse through the files, handling each one in turn */
    for (i = 1; (i <= nfiles) && (i != 0); i += (i != 0)) { /* HACK 1 OF 2 */
        char   temp_filename[FILENAME_MAX+1];
        char   *ext;

        if (i == nfiles) i = 0;                            /* HACK 2 OF 2 */
        if (verbose) printf("\nPROCESSING %s\n", original_filenames[i]);
    SWITCH_REPEAT:
        switch ( (ft = get_filetype_by_suffix(filelist[i])))
        {
        case M4FILE:
            // Strip off the .m4 suffix and find the underlying extension
            snprintf(temp_filename,sizeof(temp_filename),"%s", filelist[i]);
            ext = find_file_ext(temp_filename);
            if ( ext != NULL ) {
                *ext = 0;
                ext = find_file_ext(temp_filename);
            }
            if ( ext == NULL) ext = "";

            if (process(".m4", ext, "m4", (m4arg == NULL) ? "" : m4arg, filter, i, YES, YES))
                exit(1);
            /* Disqualify recursive .m4 extensions */
            ft = get_filetype_by_suffix(filelist[i]);
            if (ft == M4FILE) {
                fprintf(stderr, "Cannot process recursive .m4 file %s\n", original_filenames[i]);
                exit(1);
            }
            if ( c_copy_m4_processed_files ) {
                /* Write processed file to original source location immediately */		
                 ptr = stripsuffix(original_filenames[i], ".m4");		
                 if (copy_file(filelist[i], "", ptr, "")) {		
                     fprintf(stderr, "Couldn't write output file %s\n", ptr);		
                     exit(1);		
                 }		
                 /* Copied file becomes the new original file */		
                 free(original_filenames[i]);		
                 free(filelist[i]);		
                 original_filenames[i] = ptr;		
                 filelist[i] = muststrdup(ptr);
            }
            /* No more processing for .h and .inc files */
            ft = get_filetype_by_suffix(filelist[i]);
            if ((ft == HDRFILE) || (ft == INCFILE)) continue;
            /* Continue processing macro expanded source file */
            goto SWITCH_REPEAT;
            break;
        CASE_LLFILE:
        case LLFILE:
            if (m4only || clangonly) continue;
            /* llvm-cbe translates llvm-ir to c */
            if (zopt && process(".ll", ".opt.ll", "zopt", llvmopt, outspecified_flag, i, YES, NO))
                exit(1);
            if (process(".ll", ".cbe.c", c_llvm_exe, llvmarg, outspecified_flag, i, YES, NO))
                exit(1);
            /* Write .cbe.c to original directory immediately */
            ptr = changesuffix(original_filenames[i], ".cbe.c");
            if (copy_file(filelist[i], "", ptr, "")) {
                fprintf(stderr, "Couldn't write output file %s\n", ptr);
                exit(1);
            }
            /* Copied file becomes the new original file */
            free(original_filenames[i]);
            free(filelist[i]);
            original_filenames[i] = ptr;
            filelist[i] = muststrdup(ptr);
        case CFILE:
        case CXXFILE:
            if (m4only) continue;
            /* special treatment for clang+llvm */
            if ((strcmp(c_compiler_type, "clang") == 0) && !hassuffix(filelist[i], ".cbe.c")) {
                if (process(".c", ".ll", c_clang_exe, clangarg, outspecified_flag, i, YES, NO))
                    exit(1);
                goto CASE_LLFILE;
            }
            if (clangonly || llvmonly) continue;
            if (hassuffix(filelist[i], ".cbe.c"))
                BuildOptions(&cpparg, clangcpparg);
            /* past clang+llvm related pre-processing */
            if (compiler_type == CC_SDCC || compiler_type == CC_EZ80CLANG) {
                char zpragma_args[1024];
                snprintf(zpragma_args, sizeof(zpragma_args),"-zcc-opt=\"%s\"", zcc_opt_def);
                if (process(ft == CXXFILE ? ".cpp" : ".c", ".i2", c_cpp_exe, cpparg, c_stylecpp, i, YES, YES))
                    exit(1);
                if (process(".i2", ".i", c_zpragma_exe, zpragma_args, filter, i, YES, NO))
                    exit(1);
            } else {
                char zpragma_args[1024];
                snprintf(zpragma_args, sizeof(zpragma_args),"-sccz80 -zcc-opt=\"%s\"", zcc_opt_def);

                if (process(".c", ".i2", c_cpp_exe, cpparg, c_stylecpp, i, YES, YES))
                    exit(1);
                if (process(".i2", ".i", c_zpragma_exe, zpragma_args, filter, i, YES, NO))
                    exit(1);
            }
        case CPPFILE:
            {
                char *compiler_arg = NULL;

                if (m4only || clangonly || llvmonly || preprocessonly) continue;

                if ( get_filetype_by_suffix(original_filenames[i]) == CXXFILE ) {
                    if ( compiler_type != CC_EZ80CLANG) {
                        fprintf(stderr, "Only -compiler=ez80clang supports c++\n");
                        exit(1);
                    }
                    // Nobble compiler args so we compile c++
                    compiler_arg = replace_str(comparg, "-cc1", "-cc1 -x c++");
                } else {
                    compiler_arg = strdup(comparg);
                }
            
                if (process(".i", ".opt", c_compiler, compiler_arg, compiler_style, i, YES, NO))
                    exit(1);
                free(compiler_arg);
            }
        case OPTFILE:
            if (m4only || clangonly || llvmonly || preprocessonly) continue;
            if (compiler_type == CC_SDCC) {
                char  *rules[MAX_COPT_RULE_FILES];
                int    num_rules = 0;

                /* filter comments out of asz80 asm file see issue #801 on github */
                if (peepholeopt) zsdcc_asm_filter_comments(i, ".op1");

                /* sdcc_opt.9 bugfixes critical sections and implements RST substitution */
                /* rules[num_rules++] = c_sdccopt9;                                      */

                switch (peepholeopt)
                {
                case 0:
                    rules[num_rules++] = c_sdccopt9;
                    break;
                case 1:
                    rules[num_rules++] = c_sdccopt1;
                    rules[num_rules++] = c_sdccopt9;
                    break;
                default:
                    rules[num_rules++] = c_sdccopt1;
                    rules[num_rules++] = c_sdccopt9;
                    rules[num_rules++] = c_sdccopt2;
                    break;
                }

                if ( c_coptrules_target ) {
                    rules[num_rules++] = c_coptrules_target;
                }

                if ( coptrules_cpu ) {
                    rules[num_rules++] = coptrules_cpu;
                }

                if ( c_coptrules_user ) {
                    rules[num_rules++] = c_coptrules_user;
                }



                if (peepholeopt == 0)
                    apply_copt_rules(i, num_rules, rules, ".opt", ".op1", ".s");
                else
                    apply_copt_rules(i, num_rules, rules, ".op1", ".opt", ".asm");
            } else if ( compiler_type == CC_EZ80CLANG) {
                char  *rules[MAX_COPT_RULE_FILES];
                int    num_rules = 0;

                rules[num_rules++] = c_ez80clang_opt;

                apply_copt_rules(i, num_rules, rules, ".opt", ".op1", ".asm");

            } else {
                char  *rules[MAX_COPT_RULE_FILES];
                int    num_rules = 0;

                /* z80rules.9 implements intrinsics and RST substitution */
                rules[num_rules++] = c_coptrules9;

                switch (peepholeopt) {
                case 0:
                    break;
                case 1:
                    rules[num_rules++] = c_coptrules1;
                    break;
                case 2:
                    rules[num_rules++] = c_coptrules2;
                    rules[num_rules++] = c_coptrules1;
                    break;
                default:
                    rules[num_rules++] = c_coptrules2;
                    rules[num_rules++] = c_coptrules1;
                    rules[num_rules++] = c_coptrules3;
                    break;
                }

                if ( c_coptrules_target ) {
                    rules[num_rules++] = c_coptrules_target;
                }


                if ( coptrules_cpu ) {
                    rules[num_rules++] = coptrules_cpu;
                }

                if ( c_coptrules_sccz80 ) {
                    rules[num_rules++] = c_coptrules_sccz80;
                }

                if ( c_coptrules_user ) {
                    rules[num_rules++] = c_coptrules_user;
                }

                apply_copt_rules(i, num_rules, rules, ".opt", ".op1", ".asm");
            }
            /* continue processing if this is not a .s file */
            if ((compiler_type != CC_SDCC) || (peepholeopt != 0))
                goto CASE_ASMFILE;
            /* user wants to stop at the .s file if stopping at assembly translation */
            if (assembleonly) continue;
        case SFILE:
            if (m4only || clangonly || llvmonly || preprocessonly) continue;
            /* filter comments out of asz80 asm file see issue #801 on github */
            zsdcc_asm_filter_comments(i, ".s2");
            if (process(".s2", ".asm", c_copt_exe, c_sdccopt1, filter, i, YES, NO))
                exit(1);
        CASE_ASMFILE:
        case ASMFILE:
            if (m4only || clangonly || llvmonly || preprocessonly || assembleonly)
                continue;

            /* See #16 on github.                                                                    */
            /* z80asm is unable to output object files to an arbitrary destination directory.        */
            /* We don't want to assemble files in their original source directory because that would */
            /* create a temporary object file there which may accidentally overwrite user files.     */

            /* Instead the plan is to copy the asm file to the temp directory and add the original   */
            /* source directory to the include search path                                           */

            BuildAsmLine(asmarg, sizeof(asmarg), " -s ");

            /* Check if source .asm file is in the temp directory already (indicates this is an intermediate file) */
            ptr = changesuffix(temporary_filenames[i], ".asm");
            if (strcmp(ptr, filelist[i]) == 0) {
                free(ptr);
                ptr = muststrdup(asmarg);
            } else {
                char *p, tmp[FILENAME_MAX*2 + 2];

                /* copy .asm file to temp directory */
                if (copy_file(filelist[i], "", ptr, "")) {
                    fprintf(stderr, "Couldn't write output file %s\n", ptr);
                    exit(1);
                }

                /* determine path to original source directory */
                p = last_path_char(filelist[i]);

                if (!is_path_absolute(filelist[i])) {
                    int len;
#ifdef WIN32
                    if (_getcwd(tmp, sizeof(tmp) - 1) == NULL)
                        *tmp = '\0';
#else
                    if (getcwd(tmp, sizeof(tmp) - 1) == NULL)
                        *tmp = '\0';
#endif
                    if (p) {
                        len = strlen(tmp);
                        snprintf(tmp + len, sizeof(tmp) - len - 1, "/%.*s", (int)(p - filelist[i]), filelist[i]);
                    }

                    if (*tmp == '\0')
                        strcpy(tmp, ".");
                }
                else if (p) {
                    snprintf(tmp, sizeof(tmp) - 1, "%.*s", (int)(p - filelist[i]), filelist[i]);
                } else {
                    strcpy(tmp, ".");
                }

                /* working file is now the .asm file in the temp directory */
                free(filelist[i]);
                filelist[i] = ptr;

                /* add original source directory to the include path */
                ptr = mustmalloc((strlen(asmarg) + strlen(tmp) + 7) * sizeof(char));
                sprintf(ptr, "%s -I\"%s\" ", asmarg, tmp);
            }

            /* insert module directive at front of .asm file see issue #46 on github                                                 */
            /* this is a bit of a hack - foo.asm is copied to foo.tmp and then foo.tmp is written back to foo.asm with module header */

            {
                char *p, *q, tmp[FILENAME_MAX*2 + 100];

                p = changesuffix(temporary_filenames[i], ".tmp");

                if (copy_file(filelist[i], "", p, "")) {
                    fprintf(stderr, "Couldn't write output file %s\n", p);
                    exit(1);
                }

                if ((q = last_path_char(original_filenames[i])) != NULL )
                    q++;
                else
                    q = original_filenames[i];

                snprintf(tmp, sizeof(tmp) - 3, 
						 "MODULE %s%s\n"
                         "LINE 0, \"%s\"\n\n", 
						isdigit(*q) ? "_" : "", q, original_filenames[i]);

                /* change non-alnum chars in module name to underscore */

                for (q = tmp+7; *q != '\n'; ++q)
                    if (!isalnum(*q)) *q = '_';

                if (prepend_file(p, "", filelist[i], "", tmp)) {
                    fprintf(stderr, "Couldn't append output file %s\n", p);
                    exit(1);
                }

                free(p);
            }

            /* must be late assembly for the first file when making a binary */
            if ((i == 0) && build_bin)
            {
                c_crt_incpath = ptr;
                if (verbose) printf("WILL ACT AS CRT\n");
                continue;
            }

            if (process(".asm", c_extension, c_assembler, ptr, assembler_style, i, YES, NO))
                exit(1);
            free(ptr);
            break;
        case OBJFILE2:
            if (process(".obj", c_extension, c_copycmd, "", filter_out, i, YES, YES))
                exit(1);
        case OBJFILE:
            break;
        default:
            if (strcmp(filelist[i], original_filenames[i]) == 0)
                fprintf(stderr, "Filetype of %s unrecognized\n", filelist[i]);
            else
                fprintf(stderr, "Filetype of %s (%s) unrecognized\n", filelist[i], original_filenames[i]);
            exit(1);
        }
    }

    if (verbose) printf("\nGENERATING OUTPUT\n");

    if (m4only) exit(0);

    if (clangonly) {
        if (nfiles > 1) outputfile = NULL;
        copy_output_files_to_destdir(".ll", 1);
        exit(0);
    }

    if (llvmonly) exit(0);

    if (preprocessonly) {
        if (nfiles > 1) outputfile = NULL;
        copy_output_files_to_destdir(".i", 1);
        exit(0);
    }

    if (assembleonly) {
        if (nfiles > 1) outputfile = NULL;
        copy_output_files_to_destdir(".asm", 1);
        copy_output_files_to_destdir(".s", 1);
        copy_output_files_to_destdir(".adb", 1);
        exit(0);
    }
    copy_output_files_to_destdir(".adb", 1);

    {
        char *tempofile = outputfile;
        outputfile = NULL;
        if (lston) copy_output_files_to_destdir(".lis", 1);
        if (symbolson) copy_output_files_to_destdir(".sym", 1);
        outputfile = tempofile;
    }

    {
        // Sort out linklibs for CPU markers (only for classic)
        int cpu_libs = ((c_clib == NULL) || (!strstr(c_clib, "new") && !strstr(c_clib, "sdcc") && !strstr(c_clib, "clang")));
        char *tmp = replace_str(linklibs, "@{ZCC_LIBCPU}", cpu_libs ? select_cpu(CPU_MAP_TOOL_LIBNAME) : "");

        free(linklibs);
        linklibs = tmp;
    }

    if (compileonly) {
        if ((nfiles > 1) && (outputfile != NULL)) {
            /* consolidated object file */
            outputfile = changesuffix(outputfile, ".o");
            if (linkthem(c_linker))
                exit(1);
        }
        else
        {
            /* independent object files */
            copy_output_files_to_destdir(c_extension, 1);
        }
        exit(0);
    }

    /* Set the default name as necessary */

    if (outputfile == NULL)
        outputfile = c_linker_output_file ? c_linker_output_file : defaultout;

    strcpy(filenamebuf, outputfile);

    if ((ptr = find_file_ext(filenamebuf)) != NULL)
        *ptr = 0;

    /* Link */

    if (linkthem(c_linker))
    {
        if (build_bin && lston && copy_file(c_crt0, ".lis", filenamebuf, ".lis"))
            fprintf(stderr, "Cannot copy crt0 list file\n");

        exit(1);
    }

    /* Build binary */

    if (build_bin) {

        int status = 0;

		// z80asm now generates map file with same basename as output binary, i.e. a.map
		/*
        if (mapon && copy_file(c_crt0, ".map", filenamebuf, ".map")) {
            fprintf(stderr, "Cannot copy map file\n");
            status = 1;
        }
		*/

        if (symbolson && copy_file(c_crt0, ".sym", filenamebuf, ".sym")) {
            fprintf(stderr, "Cannot copy symbols file\n");
            status = 1;
        }

		// z80asm now generates def file with same basename as output binary, i.e. a.def
		/*
		if (globaldefon && copy_defc_file(c_crt0, ".def", filenamebuf, ".def")) {
            fprintf(stderr, "Cannot create global defc file\n");
            status = 1;
        }
		*/

        if (lston && copy_file(c_crt0, ".lis", filenamebuf, ".lis")) {
            fprintf(stderr, "Cannot copy crt0 list file\n");
            status = 1;
        }

        if (c_generate_debug_info && compiler_type == CC_SDCC && copy_file(c_crt0, ".adb", filenamebuf, ".adb")) {
            // Ignore error
        }

        if (createapp) {
            /* Building an application - run the appmake command on it */
			/* z80asm now generates map file with same basename as output binary, i.e. a.map */
            snprintf(buffer, sizeof(buffer), "%s %s -b \"%s\" -c \"%s\"", c_appmake_exe, appmakeargs ? appmakeargs : "", outputfile, filenamebuf);
            if (verbose) {
                printf("%s\n", buffer);
                fflush(stdout);
            }
            if (system(buffer)) {
                fprintf(stderr, "Building application code failed\n");
                status = 1;
            }
        }
        if ( !mapon ) {
            remove_file_with_extension(outputfile, ".map");
        }

        exit(status);
    }
    
    exit(0);    /* If this point is reached, all went well */
}


static void apply_copt_rules(int filenumber, int num, char **rules, char *ext1, char *ext2, char *ext)
{
    char   argbuf[FILENAME_MAX+1];
    int    i;
    char  *input_ext;
    char  *output_ext;

    for ( i = 0; i < num ; i++ ) {
        if (i % 2 == 0) {
            input_ext = ext1;
            output_ext = ext2;
        } else {
            input_ext = ext2;
            output_ext = ext1;
        }

        if ( i == (num-1) ) {
            output_ext = ext;
        }
        snprintf(argbuf,sizeof(argbuf),"%s %s %s", select_cpu(CPU_MAP_TOOL_COPT), coptarg ? coptarg : "", rules[i]);
        if (process(input_ext, output_ext, c_copt_exe, argbuf, filter, filenumber, YES, NO))
            exit(1);
    }
}


/* filter comments out of asz80 asm file see issue #801 on github */
void zsdcc_asm_filter_comments(int filenumber, char *ext)
{
    FILE *fin;
    FILE *fout;
    char *outname;

    char *line = NULL;
    unsigned int len = 0;

    outname = changesuffix(temporary_filenames[filenumber], ext);

    if ((fin = fopen(filelist[filenumber], "r")) == NULL)
    {
        fprintf(stderr, "Error: Cannot read %s\n", filelist[filenumber]);
        exit(1);
    }

    if ((fout = fopen(outname, "w")) == NULL)
    {
        fprintf(stderr, "Error: Cannot write %s\n", outname);
        fclose(fin);
        exit(1);
    }

    /* read lines from asm file */

    while (zcc_getdelim(&line, &len, '\n', fin) > 0)
    {
        unsigned int i;

        int seen_semicolon = 0;
        int seen_nonws = 0;
        int accept_line = 0;

        unsigned int quote_type = 0;
        unsigned int quote_count = 0;

        for (i = 0; i <= strlen(line); ++i)
        {
            if (accept_line == 0)
            {
                if (seen_semicolon && (line[i] != '@'))
                    break;

                seen_semicolon = 0;

                switch (line[i])
                {
                case '\'':
#ifdef WIN32
                    if ((i >= 2) && (strnicmp(&line[i - 2], "af'", 3) == 0))
#else
                    if ((i >= 2) && (strncasecmp(&line[i - 2], "af'", 3) == 0))
#endif
                        break;
                    if (quote_count && ((quote_type & 0x1) == 0))
                    {
                        quote_count--;
                        quote_type >>= 1;
                    }
                    else
                    {
                        quote_count++;
                        quote_type <<= 1;
                    }
                    break;

                case '"':
                    if (quote_count && ((quote_type & 0x1) == 1))
                    {
                        quote_count--;
                        quote_type >>= 1;
                    }
                    else
                    {
                        quote_count++;
                        quote_type = (quote_type << 1) + 1;
                    }
                    break;

                case ';':
                    if (quote_count)
                        break;
                    if (seen_nonws == 0)
                    {
                        accept_line = 1;
                        break;
                    }
                    seen_semicolon = 1;
                    break;

                default:
                    break;
                }

                if (!isspace(line[i]))
                    seen_nonws = 1;
            }
        }

        if (seen_semicolon) line[i-1] = '\0';                         /* terminate at semicolon */
        fprintf(fout, "%s\n", zcc_ascii_only(zcc_strrstrip(line)));   /* remove trailing whitespace (copt asz80 translator) and ascii-ify source (copt) */
    }

    free(line);

    fclose(fin);
    fclose(fout);

    free(filelist[filenumber]);
    filelist[filenumber] = outname;
}



/* Filter global defc file as it is written to the destination directory.
 *
 * (globaldefon     &   0x2) = make symbols PUBLIC
 * (globaldefrefile != NULL) = file holding one regular expression per line with
 *                             leading +- indicating acceptance or rejection for matches
*/

typedef struct gfilter_s {
    int      accept;
    regex_t  preg;
} GFILTER;

void gdf_cleanup(GFILTER *filter, int nfilter)
{
    while (nfilter > 1)
        regfree(&filter[--nfilter].preg);
    free(filter);
}

int copy_defc_file(char *name1, char *ext1, char *name2, char *ext2)
{
    FILE *in, *out, *rules;
    char buffer[LINEMAX + 1];
    char *line, *ptr;
    GFILTER *filter;
    regmatch_t pmatch[3];
    int nfilter, errcode, lineno;
    unsigned int len;

    /* the first regular expression is used to parse a z80asm generated defc line */
    filter  = mustmalloc(sizeof(*filter));
    nfilter = 1;
    if (regcomp(&filter->preg, "(defc|DEFC)[\t ]+([^\t =]+)", REG_EXTENDED))
    {
        fprintf(stderr, "Cannot create regular expressions\n");
        return 1;
    }

    /* read the filters from the regular expression file */
    if (globaldefrefile)
    {
        if ((rules = fopen(globaldefrefile, "r")) == NULL)
        {
            fprintf(stderr, "Cannot open rules file %s\n", globaldefrefile);
            gdf_cleanup(filter, nfilter);
            return 1;
        }

        line = NULL;
        for (lineno = 1; zcc_getdelim(&line, &len, '\n', rules) > 0; ++lineno)
        {
            ptr = zcc_strstrip(line);
            if ((*ptr == 0) || (*ptr == ';')) continue;
            if ((filter = realloc(filter, (nfilter + 1) * sizeof(*filter))) == NULL)
            {
                fprintf(stderr, "Cannot realloc global defc filter\n");
                gdf_cleanup(filter, nfilter);
                free(line);
                fclose(rules);
                return 1;
            }
            filter[nfilter].accept = !(*ptr == '-');
            if ((*ptr == '+') || (*ptr == '-')) ++ptr;
            while (isspace(*ptr)) ++ptr;
            if ( (errcode = regcomp(&filter[nfilter].preg, ptr, REG_EXTENDED)) )
            {
                regerror(errcode, &filter[nfilter].preg, buffer, sizeof(buffer));
                fprintf(stderr, "Ignoring %s line %u: %s", globaldefrefile, lineno, buffer);
                regfree(&filter[nfilter].preg);
                continue;
            }
            ++nfilter;
        }
        free(line);
        fclose(rules);
    }

    /* open the defc file generated by z80asm */
    buffer[sizeof(LINEMAX)] = 0;
    snprintf(buffer, sizeof(buffer) - 1, "%s%s", name1, ext1);
    if ((in = fopen(buffer, "r")) == NULL)
    {
        gdf_cleanup(filter, nfilter);
        return 1;
    }

    /* create the output defc file */
    snprintf(buffer, sizeof(buffer) - 1, "%s%s", name2, ext2);
    if ((out = fopen(buffer, "w")) == NULL)
    {
        gdf_cleanup(filter, nfilter);
        fclose(in);
        return 1;
    }

    line = NULL;
    while(zcc_getdelim(&line, &len, '\n', in) > 0)
    {
        /* determine symbol name */
        if (regexec(&filter->preg, line, sizeof(pmatch)/sizeof(pmatch[0]), pmatch, 0) != 0)
        {
            fprintf(stderr, "Cannot find symbol name in:\n%s", line);
            gdf_cleanup(filter, nfilter);
            free(line);
            fclose(in);
            fclose(out);
            return 1;
        }
        snprintf(buffer, sizeof(buffer) - 1, "%.*s",(int)(pmatch[2].rm_eo - pmatch[2].rm_so), &line[pmatch[2].rm_so]);

        /* accept or reject */
        if (globaldefrefile)
        {
            for (lineno = 1; lineno < nfilter; ++lineno)
            {
                if (regexec(&filter[lineno].preg, buffer, 0, NULL, 0) == 0)
                {
                    /* symbol name matches rule */
                    if (filter[lineno].accept)
                    {
                        if (globaldefon & 0x2)
                            fprintf(out, "\nPUBLIC %s\n", buffer);
                        fprintf(out, "%s", line);
                    }
                    break;
                }
            }
        }
        else
        {
            if (globaldefon & 0x2)
                fprintf(out, "\nPUBLIC %s\n", buffer);
            fprintf(out, "%s", line);
        }
    }

    gdf_cleanup(filter, nfilter);
    free(line);

    fclose(in);
    fclose(out);

    return 0;
}


int copyprepend_file(char *name1, char *ext1, char *name2, char *ext2, char *prepend)
{
    FILE           *out;
    char            buffer[LINEMAX + 1];
    char           *cmd;
    int             ret;

    if (prepend == NULL) prepend = "";

    buffer[sizeof(LINEMAX)] = 0;
    snprintf(buffer, sizeof(buffer) - 1, "%s%s", name2, ext2);

    if ((out = fopen(buffer, "w")) == NULL)
        return 1;

    fprintf(out, "%s", prepend);
    fclose(out);

#ifdef WIN32
    snprintf(buffer, sizeof(buffer), "%s \"%s%s\" >> \"%s%s\"", c_copycmd, name1, ext1, name2, ext2);
#else
    snprintf(buffer, sizeof(buffer), "%s \"%s%s\" >> \"%s%s\"", c_copycmd, name1, ext1, name2, ext2);
#endif
#ifdef WIN32
    /* Argh....annoying */
    if ((strcmp(c_copycmd, "type") == 0) || (strcmp(c_copycmd, "copy") == 0)){
        cmd = replace_str(buffer, "/", "\\");
    }
    else {
        cmd = muststrdup(buffer);
    }
#else
    cmd = muststrdup(buffer);
#endif
    if (verbose) {
        printf("%s\n", cmd);
        fflush(stdout);
    }
    ret = (system(cmd));
    free(cmd);
    return ret;
}

int copy_file(char *name1, char *ext1, char *name2, char *ext2)
{
    return copyprepend_file(name1, ext1, name2, ext2, NULL);
}

int prepend_file(char *name1, char *ext1, char *name2, char *ext2, char *prepend)
{
    return copyprepend_file(name1, ext1, name2, ext2, prepend);
}


int get_filetype_by_suffix(char *name)
{
    char      *ext = find_file_ext(name);

    if (ext == NULL) {
        return 0;
    }
    if (strcmp(ext, ".c") == 0)
        return CFILE;
    if (strcmp(ext, ".cpp") == 0)
        return CXXFILE;
    if (strcmp(ext, ".i") == 0)
        return CPPFILE;
    if (strcmp(ext, ".opt") == 0)
        return OPTFILE;
    if (strcmp(ext, ".s") == 0)
        return SFILE;
    if (strcmp(ext, ".asm") == 0)
        return ASMFILE;
    if (strcmp(ext, ".o") == 0)
        return OBJFILE;
    if (strcmp(ext, ".obj") == 0)
        return OBJFILE2;
    if (strcmp(ext, ".m4") == 0)
        return M4FILE;
    if (strcmp(ext, ".h") == 0)
        return HDRFILE;
    if (strcmp(ext, ".inc") == 0)
        return INCFILE;
    if (strcmp(ext, ".ll") == 0)
        return LLFILE;
    return 0;
}


int add_variant_args(char *wanted, int num_choices, char **choices)
{
    int  i;

    if (wanted != NULL) {
        size_t len = strlen(wanted);
        for (i = 0; i < num_choices; i++) {
            if (strncmp(wanted, choices[i], len) == 0 && isspace(choices[i][len])) {
                parse_option(muststrdup(choices[i] + len));
                break;
            }
        }
        if (i == num_choices) {
            return -1;
        }
    }
    return 0;
}


void BuildAsmLine(char *dest, size_t destlen, char *prefix)
{
    size_t offs;
    offs = snprintf(dest, destlen, "%s", asmargs ? asmargs : " ");

    if (IS_ASM(ASM_Z80ASM)) {
        offs += snprintf(dest + offs, destlen - offs, "%s%s%s%s",
            prefix,
            z80verbose ? " -v " : " ",
            select_cpu(CPU_MAP_TOOL_Z80ASM),
            symbolson ? " -s " : " ");
    }

    snprintf(dest + offs, destlen - offs, "%s", c_asmopts);
}



void GlobalDefc(option *argument, char *arg)
{
    char *ptr = arg + 1;

    if (*ptr++ == 'g') {
        /* global defc is on */
        globaldefon = 0x1;

        if (*ptr == 'p') {
            /* make defc symbols public */
            globaldefon |= 0x2;
            ++ptr;
        }

        if (*ptr == 'f') {
            /* filename containing regular expressions */
            ++ptr;
            while (isspace(*ptr) || (*ptr == '=') || (*ptr == ':')) ++ptr;

            if (*ptr != 0) {
                globaldefrefile = muststrdup(ptr);
            }
        }
    }
}


static char *expand_macros(char *arg)
{
    char  *ptr, *nval;
    char  *rep, *start;
    char  *value = muststrdup(arg);
    char   varname[300];

    start = value;
    while ((ptr = strchr(start, '$')) != NULL) {
        if (*(ptr + 1) == '{') {
            char  *end = strchr(ptr + 1, '}');

            if (end != NULL) {
                snprintf(varname, sizeof(varname), "%.*s", (int)(end - ptr - 2), ptr + 2);
                rep = getenv(varname);
                if (rep == NULL) {
                    rep = "";
                }

                snprintf(varname, sizeof(varname), "%.*s", (int)(end - ptr + 1), ptr);
                nval = replace_str(value, varname, rep);
                free(value);
                value = nval;
                start = value + (ptr - start);
            }
        }
        else {
            start++;
        }
    }

    nval = replace_str(value, "DESTDIR", c_install_dir);
    free(value);

    return nval;
}

void SetStringConfig(arg_t *argument, char *arg)
{
    *(char **)argument->data = expand_macros(arg);
}

void SetNumber(arg_t *argument, char *arg)
{
    char *ptr = arg + 1;
    char *end;
    int   val;

    if (strncmp(ptr, argument->name, strlen(argument->name)) == 0) {
        ptr += strlen(argument->name);
    }

    while (ispunct(*ptr)) ++ptr;
    val = (int)strtol(ptr, &end, 0);

    if (end != ptr) {
        *(int *)argument->data = val;
    }
}

void AddArray(arg_t *argument, char *arg)
{
    int   i = *argument->num_ptr;
    char **arr = *(char ***)argument->data;
    *argument->num_ptr = *argument->num_ptr + 1;
    arr = realloc(arr, *argument->num_ptr * sizeof(char *));
    arr[i] = expand_macros(arg);
    *(char ***)argument->data = arr;
}


void conf_opt_code_speed(option *argument, char *arg)
{
    char *sccz80_arg = NULL;
    if ( strstr(arg,"inlineints") != NULL || strstr(arg,"all") != NULL) {
        c_sccz80_inline_ints = 1;
    }
    zcc_asprintf(&sccz80_arg,"-%s%s=%s", argument->type & OPT_DOUBLE_DASH ? "-" : "",argument->long_name, arg);
    BuildOptions(&sccz80arg, sccz80_arg);
    free(sccz80_arg);

    // Add the option to sdcc as well since it has one of the same name
    BuildOptions(&sdccarg, "--opt-code-speed");
}


void AddToArgs(option *argument, char *arg)
{
    BuildOptions(argument->value, arg);
}

void AddToArgsQuoted(option *argument, char *arg)
{
    BuildOptionsQuoted(argument->value, arg);
}

void AddPreProcIncPath(option *argument, char *arg)
{
    /* user-supplied inc path takes precedence over system-supplied inc path */
    if (processing_user_command_line_arg)
        BuildOptionsQuoted(&cpp_incpath_first, arg);
    else
        BuildOptionsQuoted(&cpp_incpath_last, arg);
}

void AddPreProc(option *argument, char *arg)
{
    BuildOptions(&cpparg, arg);
    BuildOptions(&clangarg, arg);
}


void AddLinkLibrary(option *argument, char *arg)
{
    /* user-supplied lib takes precedence over system-supplied lib */
    if (processing_user_command_line_arg)
        BuildOptionsQuoted(&linker_linklib_first, arg);
    else
        BuildOptionsQuoted(&linker_linklib_last, arg);
}

void AddLinkSearchPath(option *argument, char *arg)
{
    /* user-supplied lib path takes precedence over system-supplied lib path */
    if (processing_user_command_line_arg)
        BuildOptionsQuoted(&linker_libpath_first, arg);
    else
        BuildOptionsQuoted(&linker_libpath_last, arg);
}


/** \brief Append arg to *list
*/
void BuildOptions(char **list, char *arg)
{
    char           *val;
    char           *orig = *list;

    zcc_asprintf(&val, "%s%s ", orig ? orig : "", arg);

    free(orig);
    *list = val;
}

void BuildOptions_start(char **list, char *arg)
{
    char           *val;
    char           *orig = *list;

    zcc_asprintf(&val, "%s %s", arg, orig ? orig : "");
    free(orig);
    *list = val;
}

void BuildOptionsQuoted(char **list, char *arg)
{
    char           *val;
    char           *orig = *list;
    int             len = -1;

    if ((strchr(arg, '"') == NULL) && (strchr(arg, '\'') == NULL)) {
        if ((strncmp(arg, "-I", 2) == 0) || (strncmp(arg, "-L", 2) == 0))
            len = 2;
        else if (strncmp(arg, "-iquote", 7) == 0)
            len = 7;
        else if (strncmp(arg, "-isystem", 8) == 0)
            len = 8;
    }

    if (len > 0) {
        zcc_asprintf(&val, "%s%.*s\"%s\" ", orig ? orig : "", len, arg, arg+len);
        free(orig);
        *list = val;
    } else {
        BuildOptions(list, arg);
    }
}


void add_option_to_compiler(char *arg)
{
    BuildOptions(&comparg, arg);
}


char *find_file_ext(char *filename)
{
    {
        struct explicit_extension* explicit = NULL;
        HASH_FIND_STR(explicit_extensions, filename, explicit);
        if (explicit)
        {
            return explicit->extension;
        }
    }

    char *p;

    if ((p = last_path_char(filename)) == NULL)
        p = filename;
    return strrchr(p, '.');
}


int is_path_absolute(char *p)
{
    while (*p && isspace(*p))
        ++p;

#ifdef WIN32
    return ((*p == '/') || (*p == '\\') || (isalpha(*p) && (*(p + 1) == ':')));
#else
    return (*p == '/') || (*p == '\\');
#endif
}

static void cmd_line_to_tokens(char* line, const char *path, struct tokens_list_s** tokens)
{
    char* p;
    
    while (*line && isspace(*line))
        line++;

    if ( *line == ';' || *line == '#')
        return;
    
    p = strtok(line, " \r\n\t");

    while (p != NULL)
    {
        struct tokens_list_s* token = mustmalloc(sizeof(struct tokens_list_s));
        token->token = strdup(p);
        token->path = strdup(path);
        LL_APPEND((*tokens), token);
        p = strtok(NULL, " \r\n\t");
    }
}

void cmd_free_tokens(struct tokens_list_s* tokens)
{
    struct tokens_list_s* tmp;
    struct tokens_list_s* token;
    LL_FOREACH_SAFE(tokens, token, tmp)
    {
        LL_DELETE(tokens, token);
        free(token->path);
        free(token->token);
        free(token);
    }
}

static struct tokens_list_s* gather_from_list_file(char *filename)
{
    FILE *in;
    char *line, *p;
    unsigned int len;
    char pathname[FILENAME_MAX + 1];

    struct tokens_list_s* tokens = NULL;

    /* reject non-filenames */
    if (((filename = strtok(filename, " \r\n\t")) == NULL) || !(*filename))
        return NULL;

    /* open list file for reading */
    if ((in = fopen(filename, "r")) == NULL) {
        fprintf(stderr, "Unable to open list file \"%s\"\n", filename);
        exit(1);
    }

    /* extract path from filename */
    p = last_path_char(filename);
    memset(pathname, 0, sizeof(pathname));
    if (p != NULL)
        strncpy(pathname, filename, p - filename + 1);

    /* read filenames from list file */
    line = NULL;
    while (zcc_getdelim(&line, &len, '\n', in) > 0) {
        cmd_line_to_tokens(line, pathname, &tokens);
    }

    if (!feof(in)) {
        fprintf(stderr, "Malformed line in list file \"%s\"\n", filename);
        exit(1);
    }

    free(line);
    fclose(in);

    return tokens;
}

void add_file_to_process(char *filename, char process_extension)
{
    FILE *fclaim;
    char tname[FILENAME_MAX + 1];
    char *p;
    struct stat tmp;

    if (((p = strtok(filename, " \r\n\t")) != NULL) && *p) {
        p = strip_outer_quotes(p);

        if (*p == '@') {
            struct tokens_list_s* tokens = gather_from_list_file(p + 1);

            char outname[FILENAME_MAX * 2 + 2];


            struct tokens_list_s* token;
            LL_FOREACH(tokens, token) {
                p = token->token;

                /* check for comment line */
                if ((*p == ';') || (*p == '#'))
                    continue;

                /* clear output filename */
                *outname = '\0';

                /* sanity check */
                if (strlen(p) > FILENAME_MAX) {
                    fprintf(stderr, "Filename is too long \"%s\"\n", p);
                    exit(1);
                }

                if (p[0] == '-') {
                    parse_cmdline_arg(p);
                } else if (p[0] != '+') {

                    if ( p[0] == '@') {
                        strcpy(outname,"@");
                        p++;
                    }
                    /* prepend path if filename is not absolute */
                    if (!lstcwd && !is_path_absolute(p))
                        strcat(outname, token->path);

                    /* append rest of filename */
                    strcat(outname, p);

                    /* add file to process */

                    if (strlen(outname) >= FILENAME_MAX) {
                        fprintf(stderr, "Filename is too long \"%s\"\n", outname);
                        exit(1);
                    }

                    add_file_to_process(outname, 1);
                }

                p = strtok(NULL, " \r\n\t");
            }

            cmd_free_tokens(tokens);
        } else if ((*p != ';') && (*p != '#')) { /* ignore filename leading with semicolon or hash */
            /* Expand memory for filenames */
            if ((original_filenames = realloc(original_filenames, (nfiles + 1) * sizeof(char *))) == NULL) {
                fprintf(stderr, "Unable to realloc memory for input filenames\n");
                exit(1);
            }
            if ((filelist = realloc(filelist, (nfiles + 1) * sizeof(char *))) == NULL) {
                fprintf(stderr, "Unable to realloc memory for input filenames\n");
                exit(1);
            }
            if ((temporary_filenames = realloc(temporary_filenames, (nfiles + 1) * sizeof(char *))) == NULL) {
                fprintf(stderr, "Unable to realloc memory for input filenames\n");
                exit(1);
            }

            if (process_extension && explicit_file_type_defined()) {
                struct explicit_extension* exp = mustmalloc(sizeof(struct explicit_extension));
                exp->filename = muststrdup(p);
                exp->extension = get_explicit_file_type();
                HASH_ADD_STR(explicit_extensions, filename, exp);
                original_filenames[nfiles] = muststrdup(p);
            } else {
                /* Add this file to the list of original files */
                if (find_file_ext(p) == NULL) {
                    /* file without extension - see if it exists, exclude directories */
                    if ( stat(p, &tmp) == 0 && !S_ISDIR(tmp.st_mode) ) {
                        fprintf(stderr, "Unrecognized file type %s\n", p);
                        exit(1);
                    }
                    /* input file has no extension and does not exist so assume .asm then .o then .asm.m4 */
                    original_filenames[nfiles] = mustmalloc((strlen(p) + 8) * sizeof(char));
                    strcpy(original_filenames[nfiles], p);
                    strcat(original_filenames[nfiles], ".asm");
                    if (stat(original_filenames[nfiles], &tmp) != 0) {
                        strcpy(strrchr(original_filenames[nfiles], '.'), ".o");
                        if (stat(original_filenames[nfiles], &tmp) != 0)
                            strcpy(strrchr(original_filenames[nfiles], '.'), ".asm.m4");
                    }
                } else {
                    original_filenames[nfiles] = muststrdup(p);
                }
            }

            /* Working file is the original file */
            filelist[nfiles] = muststrdup(original_filenames[nfiles]);

            /* Now work out temporary filename */
            tempname(tname);
            temporary_filenames[nfiles] = muststrdup(tname);

            /* Claim the temporary filename */
            if ((fclaim = fopen(temporary_filenames[nfiles], "w")) == NULL) {
                fprintf(stderr, "Unable to claim temporary filename %s\n", temporary_filenames[nfiles]);
                exit(1);
            }
            fclose(fclaim);

            nfiles++;
        }
    }
}





void usage(const char *program)
{
    fprintf(stderr,"zcc - Frontend for the z88dk Cross-C Compiler - %s\n",version);
    fprintf(stderr,"Usage: %s +[target] {options} {files}\n",program);
}

void print_help_text(const char *program)
{
    int         i;

    usage(program);

    option_list(&options[0]);

    fprintf(stderr,"\nArgument Aliases:\n\n");
    for ( i = 0; i < aliases_num; i+=2 ) {
        fprintf(stderr,"%-20s %s\n", aliases[i],aliases[i+1]);
    }

    if ( c_clib_array_num ) {
        fprintf(stderr,"\n-clib options:\n\n");
        for ( i = 0; i < c_clib_array_num; i++ ) {
            char buf[LINEMAX+1];
            char *ptr = c_clib_array[i];
            char *dest = buf;
            while ( *ptr && !isspace(*ptr)) {
                *dest++ = *ptr++;
            }
            *dest = 0;
            fprintf(stderr,"-clib=%-14s\n", buf);
        }
    }
    if ( c_subtype_array_num ) {
        fprintf(stderr,"\n-subtype options:\n\n");
        for ( i = 0; i < c_subtype_array_num; i++ ) {
            char buf[LINEMAX+1];
            char *ptr = c_subtype_array[i];
            char *dest = buf;
            while ( *ptr && !isspace(*ptr)) {
                *dest++ = *ptr++;
            }
            *dest = 0;
            fprintf(stderr,"-subtype=%-11s\n", buf);
        }
    }

    exit(0);
}



void parse_cmdline_arg(char *arg)
{
    int             i;
    char           *tempargv[2];

    tempargv[1] = arg;

    if ( option_parse(&options[0], 2, &tempargv[0]) == 0 ) {
        return;
    }
   
    for ( i = 0; i < aliases_num; i+=2 ) {
        if ( strcmp(arg, aliases[i]) == 0 ) {
            parse_option(muststrdup(aliases[i+1]));
            return;
        }
    }
    add_option_to_compiler(arg);
}


void LoadConfigFile(arg_t *argument, char *arg)
{
    char   buf[FILENAME_MAX+1];
    struct stat sb;

    do {
        // 1. Try a local file/absolute path
        snprintf(buf,sizeof(buf), "%s",arg);
        if ( stat(buf, &sb) == 0 && S_ISREG(sb.st_mode)) {
            break;
        }

        // 2. Try in ZCCCFG
        if (c_zcc_cfg != NULL) {
            /* Config file in config directory */
            snprintf(buf, sizeof(buf), "%s/%s", c_zcc_cfg, arg);
            if ( stat(buf, &sb) == 0 && S_ISREG(sb.st_mode)) {
                break;
            }
        }

        // 3. Try in cfg file path
        snprintf(buf,sizeof(buf),"%s/%s",cfg_path,arg);
        if ( stat(buf, &sb) == 0 && S_ISREG(sb.st_mode)) {
            break;
        }

        // 4. Try install dir
        snprintf(buf, sizeof(buf), "%s/lib/config/%s", c_install_dir, arg);
        if ( stat(buf, &sb) == 0 && S_ISREG(sb.st_mode)) {
            break;
        }

        fprintf(stderr, "Can't open config file %s\n", arg);
        exit(1);
    } while (0);

    parse_configfile(buf);
}

void parse_configfile(const char *filename)
{
    FILE *fp;
    char  buffer[LINEMAX+1];

      /* Okay, so now we read in the options file and get some info for us */
    if ((fp = fopen(filename, "r")) == NULL) {
        fprintf(stderr, "Can't open config file %s\n", filename);
        exit(1);
    }
    while (fgets(buffer, LINEMAX, fp) != NULL) {
        if (!isupper(buffer[0]))
            continue;
        parse_configfile_line(buffer);
    }
    fclose(fp);
}

void parse_configfile_line(char *arg)
{
    arg_t          *pargs = config;

    while (pargs->setfunc) {
        if (strncmp(arg, pargs->name, strlen(pargs->name)) == 0) {
            char *val = arg + strlen(pargs->name);
            while (isspace(*val)) {
                val++;
            }
            KillEOL(val);
            (*pargs->setfunc) (pargs, val);
            return;
        }
        pargs++;
    }
    printf("Unrecognised config option: %s\n", arg);
    return;
}

static void configure_misc_options(void)
{
    char     buf[256];

    /* Setup the extension - config files just seem to specify -M, so fudge it */
    snprintf(buf, sizeof(buf), ".%s", c_extension_config && strlen(c_extension_config) ? c_extension_config : "o");
    c_extension = muststrdup(buf);

    /*
    the new c lib uses startup=-1 to mean user supplies the crt
    current working dir will be different than when using -crt0
    */

    if (c_startup >= -1) {
        write_zcc_defined("startup", c_startup, 0);
    }

    if (c_startupoffset >= 0) {
        write_zcc_defined("startupoffset", c_startupoffset, 0);
    }

    if (linkargs == NULL) {
        linkargs = muststrdup("");
    }

    if (linklibs == NULL) {
        linklibs = muststrdup("");
    }
}

static void configure_maths_library(char **libstring)
{
    char   buf[1024];

    /* By convention, -lm refers to GENMATH, -lmz to Z88MATHLIB/ALTMATHLIB */

    if (c_altmathlib) {
        if (strstr(*libstring, "-lmz ") != NULL) {
            snprintf(buf, sizeof(buf), "-l\"%s\" ", c_altmathlib);
            if ((*libstring = replace_str(*libstring, "-lmz ", buf)) == NULL) {
                fprintf(stderr, "Malloc failed\n");
                exit(1);
            }
            parse_option(c_altmathflags);
        }
    }

    if (c_genmathlib) {
        if (strstr(*libstring, "-lm ") != NULL) {
            snprintf(buf, sizeof(buf), "-l\"%s\" ", c_genmathlib);
            if ((*libstring = replace_str(*libstring, "-lm ", buf)) == NULL) {
                fprintf(stderr, "Malloc failed\n");
                exit(1);
            }
        }
    }
}

static void configure_assembler(void)
{
    char            buf[FILENAME_MAX+1];
    char           *assembler = NULL;
    char           *linker = NULL;
    int             type = ASM_Z80ASM;
    enum iostyle    style = outimplied;

    type = ASM_Z80ASM;
    linker = c_z80asm_exe;
    assembler = c_z80asm_exe;

    assembler_type = type;
    assembler_style = style;
    if (assembler) {
        c_assembler = assembler;
    }
    if (linker) {
        c_linker = linker;
    }
    if ( c_generate_debug_info) {
        mapon = 1;
        BuildOptions(&asmargs, "-m");
        BuildOptions(&asmargs, "-debug");
        BuildOptions(&linkargs, "-debug");
    }
    snprintf(buf,sizeof(buf),"-I\"%s\"",zcc_opt_dir);
    BuildOptions(&asmargs, buf);
    BuildOptions(&linkargs, buf);
}




static void configure_compiler(void)
{
    char *preprocarg;
    char  buf[256];

    compiler_type = CC_SCCZ80;
    BuildOptions(&cpparg, "-D__Z88DK");

    /* compiler= */
    if ((strcmp(c_compiler_type, "clang") == 0) || (strcmp(c_compiler_type, "sdcc") == 0)) {
        char *cpuarg = select_cpu(CPU_MAP_TOOL_ZSDCC);

        if ( cpuarg == NULL ) {
            fprintf(stderr, "Selected CPU is not supported by zsdcc\n");
            exit(1);
        }


        compiler_type = CC_SDCC;
        snprintf(buf, sizeof(buf), "%s --no-optsdcc-in-asm --c1mode --emit-externs %s %s %s ", \
            cpuarg, \
            (sdcc_signed_char ? "--fsigned-char" : ""), \
            (c_code_in_asm ? "" : "--no-c-code-in-asm"), \
            (opt_code_size ? "--opt-code-size" : ""));
        add_option_to_compiler(buf);
        if (sdccarg) {
            add_option_to_compiler(sdccarg);
        }
        if ( c_generate_debug_info) {
            add_option_to_compiler("--debug");
        }
        preprocarg = " -D__SDCC";
        BuildOptions(&cpparg, preprocarg);
        c_compiler = c_sdcc_exe;
        c_cpp_exe = c_sdcc_preproc_exe;
        compiler_style = filter_outspecified_flag;
        BuildOptions(&asmargs, "-D__SDCC");
        BuildOptions(&linkargs, "-D__SDCC");
    } else if (strcmp(c_compiler_type,"sccz80") == 0 ) {
        preprocarg = " -DSCCZ80 -DSMALL_C -D__SCCZ80";
        BuildOptions(&cpparg, preprocarg);
        BuildOptions(&asmargs, "-D__SCCZ80");
        BuildOptions(&linkargs, "-D__SCCZ80");
        /* Indicate to sccz80 what assembler we want */
        snprintf(buf, sizeof(buf), "-ext=opt %s -zcc-opt=\"%s\"", select_cpu(CPU_MAP_TOOL_SCCZ80),zcc_opt_def);
        add_option_to_compiler(buf);

        if (sccz80arg) {
            add_option_to_compiler(sccz80arg);
        }
        if (c_code_in_asm) {
            add_option_to_compiler("-cc");
        }
        if (c_sccz80_r2l_calling) {
            add_option_to_compiler("-set-r2l-by-default");
            preprocarg = " -DZ88DK_R2L_CALLING_CONVENTION";
            BuildOptions(&cpparg, preprocarg);
        }
        if ( c_generate_debug_info) {
            add_option_to_compiler("-debug-defc");
        }
        c_compiler = c_sccz80_exe;
        compiler_style = outspecified_flag;
    } else if (strcmp(c_compiler_type,"ez80clang") == 0 ) {
        char *cpuarg = select_cpu(CPU_MAP_TOOL_EZ80CLANG);

        if ( cpuarg == NULL ) {
            fprintf(stderr, "Selected CPU is not supported by ez80-clang\n");
            exit(1);
        }

        preprocarg = " -E -D__CLANG";
        BuildOptions(&cpparg, preprocarg);
        snprintf(buf, sizeof(buf), "-cc1 %s -S -O3", cpuarg);
        add_option_to_compiler(buf);
        compiler_type = CC_EZ80CLANG;
        c_compiler = c_ez80clang_exe;
        c_cpp_exe = c_ez80clang_exe;
        compiler_style = filter_outspecified_flag;
        c_stylecpp = filter_out;
    } else {
        printf("Unknown compiler type: %s\n",c_compiler_type);
        exit(1);
    }
}


void PragmaInclude(option *arg, char *val)
{
    char *ptr = strip_outer_quotes(val);

    while ((*ptr == '=') || (*ptr == ':')) ++ptr;

    if (*ptr != '\0') {
        free(pragincname);
        pragincname = muststrdup(ptr);
    }
}

void PragmaRedirect(option *arg, char *val)
{
    char *eql;
    char *ptr = val;
    char *value = "";

    if ((eql = strchr(ptr, '=')) != NULL) {
        *eql = 0;
        value = eql + 1;
    }
    if (strlen(value)) {
        add_zccopt("\nIF !DEFINED_%s\n", ptr);
        add_zccopt("\tPUBLIC %s\n", ptr);
        add_zccopt("\tEXTERN %s\n", value);
        add_zccopt("\tdefc\tDEFINED_%s = 1\n", ptr);
        add_zccopt("\tdefc %s = %s\n", ptr, value);
        add_zccopt("ENDIF\n\n");
    }
}

void Alias(option *arg, char *val)
{
    char *ptr = val;
    char *eql;

    if ((eql = strchr(ptr, '=')) != NULL) {
        *eql = 0;
        aliases = realloc(aliases, (aliases_num + 2) * sizeof(aliases[0]));
        aliases[aliases_num++] = strdup(ptr);
        aliases[aliases_num++] = strdup(eql+1);
    }
}


void PragmaDefine(option *arg, char *val)
{
    char *ptr = val;
    int   value = 0;
    char *eql;

    if ((eql = strchr(ptr, '=')) != NULL) {
        *eql = 0;
        value = (int)strtol(eql + 1, NULL, 0);
    }
    write_zcc_defined(ptr, value, 0);
}

void PragmaExport(option *arg, char *val)
{
    char *ptr = val;
    int   value = 0;
    char *eql;

    if ((eql = strchr(ptr, '=')) != NULL) {
        *eql = 0;
        value = (int)strtol(eql + 1, NULL, 0);
    }
    write_zcc_defined(ptr, value, 1);
}

void write_zcc_defined(char *name, int value, int export)
{
    add_zccopt("\nIF !DEFINED_%s\n", name);
    add_zccopt("\tdefc\tDEFINED_%s = 1\n", name);
    if (export) add_zccopt("\tPUBLIC\t%s\n", name);
    add_zccopt("\tdefc %s = %d\n", name, value);
    add_zccopt("\tIFNDEF %s\n\tENDIF\n", name);
    add_zccopt("ENDIF\n\n");
}

void PragmaNeed(option *arg, char *val)
{
    char *ptr = val;
    
    add_zccopt("\nIF !NEED_%s\n", ptr);
    add_zccopt("\tDEFINE\tNEED_%s\n", ptr);
    add_zccopt("ENDIF\n\n");
}


void PragmaBytes(option *arg, char *val)
{
    char *ptr = val;
    char *value;

    if ((value = strchr(ptr, '=')) != NULL) {
        *value++ = 0;
    }
    else {
        return;
    }

    add_zccopt("\nIF NEED_%s\n", ptr);
    add_zccopt("\tDEFINE\tDEFINED_NEED_%s\n", ptr);
    add_zccopt("\tdefb\t%s\n", value);
    add_zccopt("ENDIF\n\n");
}

void PragmaString(option *arg, char *val)
{
    char *ptr = val;
    char *value;

    if ((value = strchr(ptr, '=')) != NULL) {
        *value++ = 0;
    }
    else {
        return;
    }

    add_zccopt("\nIF NEED_%s\n", ptr);
    add_zccopt("\tDEFINE\tDEFINED_NEED_%s\n", ptr);
    add_zccopt("\tdefm\t\"%s\"\n", value);
    add_zccopt("ENDIF\n\n");
}

/** \brief Remove line feeds at the end of a line
*/
void KillEOL(char *str)
{
    char           *ptr;
    if ((ptr = strrchr(str, '\n')))
        *ptr = 0;
    if ((ptr = strrchr(str, '\r')))
        *ptr = 0;
}


/** \brief Copy any generated output files back to the directory they came from
*/
void copy_output_files_to_destdir(char *suffix, int die_on_fail)
{
    int             j;
    char           *ptr, *p, *name;
    struct stat     tmp;
    char            fname[FILENAME_MAX + 32];

    /* copy each output file having indicated extension */
    for (j = build_bin ? 1 : 0; j < nfiles; ++j) {

        /* check that original file was actually processed */
        if (((ptr = find_file_ext(original_filenames[j])) == NULL) || (strcmp(ptr, suffix) != 0)) {

            ptr = changesuffix(filelist[j], suffix);

            /* check that this file was generated */
            if (stat(ptr, &tmp) == 0) {

                /* generate output filename */
                if (outputfile != NULL)
                    strcpy(fname, outputfile);                                  /* use supplied output filename */
                else {
                    /* using original filename to create output filename */
                    name = stripsuffix(original_filenames[j], ".m4");
                    if (strcmp(suffix, c_extension) == 0) {
                        p = changesuffix(name, suffix);                         /* for .o, use original filename with extension changed to .o */
                        strcpy(fname, p);
                        free(p);
                    }
                    else {
                        p = stripsuffix(name, suffix);
                        snprintf(fname, sizeof(fname), "%s%s", p, suffix);      /* use original filename with extension appended */
                        free(p);
                    }
                    free(name);
                }

                if (verbose && preprocessonly && printmacros) {
                    FILE* f = fopen(ptr, "r");
                    if (f) {
                        static char buf[1024];
                        unsigned long nread;
                        while ((nread = fread(buf, 1, sizeof(buf), f)) > 0)
                            fwrite(buf, 1, nread, stdout);
                        fclose(f);
                    }
                }

                /* copy to output directory */
                if (copy_file(ptr, "", fname, "")) {
                    fprintf(stderr, "Couldn't copy output file %s\n", fname);
                    if (die_on_fail) {
                        free(ptr);
                        exit(1);
                    }
                }
            } 

            free(ptr);
        }
    }
}


void remove_temporary_files(void)
{
    int             j;

    if (cleanup) {    /* Default is yes */
        for (j = 0; j < nfiles; j++) {
            remove_file_with_extension(temporary_filenames[j], "");
            remove_file_with_extension(temporary_filenames[j], ".ll");
            remove_file_with_extension(temporary_filenames[j], ".i");
            remove_file_with_extension(temporary_filenames[j], ".i2");
            remove_file_with_extension(temporary_filenames[j], ".asm");
            remove_file_with_extension(temporary_filenames[j], ".s2");
            remove_file_with_extension(temporary_filenames[j], ".err");
            remove_file_with_extension(temporary_filenames[j], ".op2");
            remove_file_with_extension(temporary_filenames[j], ".op1");
            remove_file_with_extension(temporary_filenames[j], ".opt");
            remove_file_with_extension(temporary_filenames[j], ".o");
            remove_file_with_extension(temporary_filenames[j], ".map");
            remove_file_with_extension(temporary_filenames[j], ".adb");
            remove_file_with_extension(temporary_filenames[j], ".sym");
            remove_file_with_extension(temporary_filenames[j], ".def");
            remove_file_with_extension(temporary_filenames[j], ".tmp");
            remove_file_with_extension(temporary_filenames[j], ".lis");
        }
        /* Cleanup zcc_opt files */
        remove(zcc_opt_def);
        rmdir(zcc_opt_dir);
    }
}


void remove_file_with_extension(char *file, char *ext)
{
    char           *temp;
    temp = changesuffix(file, ext);
    remove(temp);
    free(temp);
}


/** \brief Get a temporary filename
*/
void tempname(char *filen)
{
#ifdef _WIN32

    char   *ptr;

    if ((ptr = _tempnam(".\\", tmpnambuf)) == NULL) {
        fprintf(stderr, "Failed to create temporary filename\n");
        exit(1);
    }
    strcpy(filen, ptr);
    free(ptr);

#elif defined(__MSDOS__) && defined(__TURBOC__)
    /* Both predefined by Borland's Turbo C/C++ and Borland C/C++ */

    if (ptr = getenv("TEMP")) {    /* From MS-DOS 5, C:\TEMP, C:\DOS,
                                    * C:\WINDOWS\TEMP or whatever or
                                    * nothing */
        strcpy(filen, ptr);    /* Directory is not guaranteed to
                                * exist */
        strcat(filen, "\\");
        tmpnam(filen + strlen(filen));    /* Adds strings like
                                           * TMP1.$$$, TMP2.$$$ */
    }
    /* Allways starts at TMP1.$$$. Does not */
    else            /* check if file already exists. So is  */
        tmpnam(filen);    /* not suitable for executing zcc more  */
    if (ptr = find_file_ext(filen))    /* than once without cleaning out
                                        * files. */
        *ptr = 0;    /* Don't want to risk too long filenames */
#else
    strcpy(filen, "/tmp/tmpXXXXXXXX");

    /* Prevent linker warning: the use of mktemp is dangerous */
    /* mktemp(filen);                                         */
    if ( ( mkstemp(filen) == -1 ) || ( unlink(filen) == -1 ) ) {   /* Automatic delete of file by unlink */
        fprintf(stderr, "Failed to create temporary filename\n");
        exit(1);
    }
#endif
}

/*
*    Find a config file to use:
*
*    Scheme is as follows:
*    Use ZCCFILE for compatibility
*    If not, use ZCCCFG/zcc.cfg
*        or  ZCCCFG/argv[1]
*    Or as a first resort argv[1]
*    Returns gc (or exits)
*
*    If ZCCCFG doesn't exist then we take the c_install_dir/lib/config/zcc.cfg
*/
void find_zcc_config_fileFile(const char *program, char *arg, char *buf, size_t buflen)
{
    FILE           *fp;

    /* Scan for an option file on the command line */
    if (arg[0] == '+') {
        snprintf(buf, buflen, "%s", arg + 1);
        if (strstr(arg, ".cfg") != NULL) {
            if ((fp = fopen(buf, "r")) != NULL) {
                /* Local config file */
                fclose(fp);
                return;
            }
        }
        if (c_zcc_cfg != NULL) {
            /* Config file in config directory */
            snprintf(buf, buflen, "%s/%s.cfg", c_zcc_cfg, arg + 1);
            return;
        } else {
            snprintf(buf, buflen, "%s/lib/config/%s.cfg", c_install_dir, arg + 1);
        }
        /*
         * User supplied invalid config file, let it fall over back
         * when
         */
        return;
    }
    /* Without a config file, we should just print usage and then exit */
    fprintf(stderr, "A config file must be specified with +file\n\n");
    print_help_text(program);
    exit(1);
}


/* Parse options - rewritten to use qstrtok which is like strtok but understands quoting */
void parse_option(char *option)
{
    char           *ptr;

    if (option != NULL) {
        ptr = qstrtok(option, " \t\r\n");
        while (ptr != NULL) {
            if (ptr[0] == '-') {
                parse_cmdline_arg(strip_inner_quotes(ptr));
            } else {
                add_file_to_process(strip_outer_quotes(ptr), 1);
            }
            ptr = qstrtok(NULL, " \t\r\n");
        }
    }
}

void add_zccopt(char *fmt, ...)
{
    char   buf[4096];
    size_t len = zccopt ? strlen(zccopt) : 0;
    size_t extra;
    va_list ap;

    va_start(ap, fmt);
    extra = vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);

    zccopt = realloc(zccopt, len + extra + 1);
    strcpy(zccopt + len, buf);
}

/* From: http://creativeandcritical.net/str-replace-c/ */
char *replace_str(const char *str, const char *old, const char *new)
{
    char *ret, *r;
    const char *p, *q;
    size_t oldlen = strlen(old);
    size_t count, retlen, newlen = strlen(new);

    if (oldlen != newlen) {
        for (count = 0, p = str; (q = strstr(p, old)) != NULL; p = q + oldlen)
            count++;
        /* this is undefined if p - str > PTRDIFF_MAX */
        retlen = p - str + strlen(p) + count * (newlen - oldlen);
    }
    else
        retlen = strlen(str);

    ret = mustmalloc(retlen + 1);

    for (r = ret, p = str; (q = strstr(p, old)) != NULL; p = q + oldlen) {
        /* this is undefined if q - p > PTRDIFF_MAX */
        ptrdiff_t l = q - p;
        memcpy(r, p, l);
        r += l;
        memcpy(r, new, newlen);
        r += newlen;
    }
    strcpy(r, p);

    return ret;
}

static void setup_default_configuration(void)
{
    char    buf[1024];
    arg_t  *pargs = config;

    while (pargs->setfunc) {
        if (pargs->defvalue) {
            snprintf(buf, sizeof(buf), "%s %s", pargs->name, pargs->defvalue);
            parse_configfile_line(buf);
        }
        pargs++;
    }
}

static void print_specs(void)
{
    arg_t  *pargs = config;
    int     i;

    while (pargs->setfunc) {
        if ((pargs->flags & AF_DEPRECATED) == 0 && pargs->setfunc == SetStringConfig) {
            if (*(char **)pargs->data != NULL && strlen(*(char **)pargs->data)) {
                printf("%-20s\t%s\n", pargs->name, *(char **)pargs->data);
            }
            else {
                printf("%-20s\t[undefined]\n", pargs->name);
            }
        }
        else if (pargs->setfunc == AddArray) {
            for (i = 0; i < *pargs->num_ptr; i++) {
                printf("%-20s\t%s\n", pargs->name, (*(char ***)pargs->data)[i]);
            }
        }
        pargs++;
    }
}


static int isquote(unsigned char c)
{
    return ((c == '"') || (c == '\''));
}


/* strtok with quoting */
static char *qstrtok(char *s, const char *delim)
{
    static char *start = NULL;
    char *ret;
    uint32_t quote_mask = 0;
    uint32_t quote_type = 0;
    int type = 0;

    /* check for new string */
    if (s != NULL)
    {
        /* clear inquote indicators */
        quote_mask = quote_type = 0;

        /* skip initial delimiters */
        for (start = s; *start; ++start)
            if ((strchr(delim, *start) == NULL) || isquote(*start))
                break;

        if (*start == '\0') start = NULL;
    }

    /* check if current string is done */
    if (start == NULL) return NULL;

    /* look for next token in current string */
    for (ret = start; *start; ++start) {
        if (quote_mask) {
            /* inside quote, ignore delim */
            if (isquote(*start)) {
                type = (*start == '"');
                if (type == (quote_type & 0x01))
                {
                    /* undoing one level of quote */
                    quote_mask >>= 1;
                    quote_type >>= 1;
                } else {
                    /* adding a level of quote */
                    if (quote_mask & 0x80000000UL)
                    {
                        fprintf(stderr, "Error: Reached maximum quoting level\n");
                        exit(1);
                    }

                    quote_mask = (quote_mask << 1) + 1;
                    quote_type = (quote_type << 1) + type;
                }
            }
        } else {
            /* behave like strtok, delim takes precedence over quoting */
            if (strchr(delim, *start))
                break;
            /* check for quoting */
            if (isquote(*start)) {
                quote_mask = 1;
                quote_type = (*start == '"');
            }
        }
    }

    if (*start == '\0')
        start = NULL;
    else
        *start++ = '\0';
    return ret;
}


/* strip away first level of quotes inside string */
static char *strip_inner_quotes(char *p)
{
    char *first, *last, *temp;
    size_t len;

    len = strlen(p);

    first = strchr(p, '"');
    temp = strchr(p, '\'');

    if ((first == NULL) || ((temp != NULL) && (first > temp)))
        first = temp;

    last = strrchr(p, '"');
    temp = strrchr(p, '\'');

    if ((last == NULL) || ((temp != NULL) && (last < temp)))
        last = temp;

    if ((first != NULL) && (first != last) && (*first == *last)) {
        memmove(first, first + 1, last - first - 1);
        memmove(last - 1, last, p + len - last - 1);
        p[len - 2] = '\0';
    }

    return p;
}


/* strip away outer quotes if string is quoted */
static char *strip_outer_quotes(char *p)
{
    size_t q = strlen(p);

    if (isquote(*p) && (p[0] == p[q - 1])) {
        p[q - 1] = '\0';
        p++;
    }

    return p;
}


static int zcc_vasprintf(char **s, const char *fmt, va_list ap)
{
    FILE       *fp;
    va_list     saveap;
    size_t      req;
    char       *ret;

    /* MSC Visual Studio 2010 does not have va_copy and va_list is a simple pointer */
#ifdef _MSC_VER
    saveap = ap;
#else
    va_copy(saveap, ap);
#endif

    /* This isn't performant, but we don't use it that much */
    if (
#ifndef WIN32
    (fp = fopen("/dev/null", "w")) != NULL
#else
        (fp = fopen("NUL", "w")) != NULL
#endif
        ) {
        req = vfprintf(fp, fmt, ap);
        fclose(fp);
        ret = calloc(req + 1, sizeof(char));
        req = vsnprintf(ret, req + 1, fmt, saveap);
        *s = ret;
    }
    else {
        *s = NULL;
        req = -1;
    }
    return req;
}



static int zcc_asprintf(char **s, const char *fmt, ...)
{
    va_list arg;
    int     res;

    va_start(arg, fmt);
    res = zcc_vasprintf(s, fmt, arg);
    va_end(arg);
    return res;
}




/*
* zcc_getdelim()
* Copyright (C) 2003 ETC s.r.o.
*
* This program is free software; you can redistribute it and/or
* modify it under the terms of the GNU General Public License
* as published by the Free Software Foundation; either version 2
* of the License, or (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program; if not, write to the Free Software
* Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA
* 02111-1307, USA.
*
* Written by Marcel Telka <marcel@telka.sk>, 2003.
*
*/

#ifndef GETDELIM_BUFFER
#define GETDELIM_BUFFER 128
#endif

static int zcc_getdelim(char **lineptr, unsigned int *n, int delimiter, FILE *stream)
{

    char *p, *np;
    int c;
    unsigned int len = 0;

    if (!lineptr || !n)
        return -1;

    /* allocate initial buffer */
    if (!*lineptr || !*n) {
        np = realloc(*lineptr, GETDELIM_BUFFER);
        if (!np)
            return -1;
        *n = GETDELIM_BUFFER;
        *lineptr = np;
    }

    p = *lineptr;

    /* read characters from stream */
    while ((c = fgetc(stream)) != EOF) {
        if (len >= *n) {
            np = realloc(*lineptr, *n * 2);
            if (!np)
                return -1;
            p = np + (p - *lineptr);
            *lineptr = np;
            *n *= 2;
        }
        *p++ = (char)c;
        len++;
        if (delimiter == c)
            break;
    }

    /* end of file without any bytes read */
    if ((c == EOF) && (len == 0))
        return -1;

    /* trailing '\0' */
    if (len >= *n) {
        np = realloc(*lineptr, *n + 1);
        if (!np)
            return -1;
        p = np + (p - *lineptr);
        *lineptr = np;
        *n += 1;
    }
    *p = '\0';

    return len;
}



/*
* Local Variables:
*  indent-tabs-mode:nil
*  require-final-newline:t
*  c-basic-offset: 4
*  eval: (c-set-offset 'case-label 0)
*  eval: (c-set-offset 'substatement-open 0)
*  eval: (c-set-offset 'access-label 0)
*  eval: (c-set-offset 'class-open 4)
*  eval: (c-set-offset 'class-close 4)
* End:
*/
