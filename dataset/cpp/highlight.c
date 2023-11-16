#include <stic.h>

#include <limits.h> /* INT_MAX */
#include <string.h> /* strcpy() */

#include <test-utils.h>

#include "../../src/cfg/config.h"
#include "../../src/lua/vlua.h"
#include "../../src/ui/color_scheme.h"
#include "../../src/ui/statusbar.h"
#include "../../src/ui/ui.h"
#include "../../src/utils/matchers.h"
#include "../../src/cmd_core.h"
#include "../../src/filelist.h"
#include "../../src/status.h"

SETUP_ONCE()
{
	cmds_init();
	cs_reset(&cfg.cs);
	lwin.list_rows = 0;
	rwin.list_rows = 0;

	curr_view = &lwin;
}

SETUP()
{
	cs_reset(&cfg.cs);
	curr_stats.cs = &cfg.cs;
}

TEARDOWN()
{
	cs_reset(&cfg.cs);
	curr_stats.cs = NULL;
}

/* General behaviour. */

TEST(all_colors_are_printed)
{
	/* On PDCurses A_STANDOUT == (A_REVERSE | A_BOLD). */
#if __PDCURSES__
# define ATTR ",standout"
#else
# define ATTR ""
#endif

	const char *expected =
		"Win        cterm=none ctermfg=white   ctermbg=black\n"
		"Directory  cterm=bold ctermfg=cyan    ctermbg=default\n"
		"Link       cterm=bold ctermfg=yellow  ctermbg=default\n"
		"BrokenLink cterm=bold ctermfg=red     ctermbg=default\n"
		"HardLink   cterm=none ctermfg=yellow  ctermbg=default\n"
		"Socket     cterm=bold ctermfg=magenta ctermbg=default\n"
		"Device     cterm=bold ctermfg=red     ctermbg=default\n"
		"Fifo       cterm=bold ctermfg=cyan    ctermbg=default\n"
		"Executable cterm=bold ctermfg=green   ctermbg=default\n"
		"Selected   cterm=bold ctermfg=magenta ctermbg=default\n"
		"CurrLine   cterm=bold,reverse" ATTR " ctermfg=default ctermbg=default\n"
		"TopLine    cterm=none ctermfg=black   ctermbg=white\n"
		"TopLineSel cterm=bold ctermfg=black   ctermbg=default\n"
		"StatusLine cterm=bold ctermfg=black   ctermbg=white\n"
		"WildMenu   cterm=underline,reverse ctermfg=white   ctermbg=black\n"
		"CmdLine    cterm=none ctermfg=white   ctermbg=black\n"
		"ErrorMsg   cterm=none ctermfg=red     ctermbg=black\n"
		"Border     cterm=none ctermfg=black   ctermbg=white\n"
		"OtherLine  cterm=none ctermfg=default ctermbg=default\n"
		"JobLine    cterm=bold,reverse" ATTR " ctermfg=black   ctermbg=white\n"
		"SuggestBox cterm=bold ctermfg=default ctermbg=default\n"
		"CmpMismatch cterm=bold ctermfg=white   ctermbg=red\n"
		"CmpUnmatched cterm=bold ctermfg=white   ctermbg=green\n"
		"CmpBlank   cterm=none ctermfg=default ctermbg=default\n"
		"AuxWin     cterm=none ctermfg=default ctermbg=default\n"
		"TabLine    cterm=none ctermfg=white   ctermbg=black\n"
		"TabLineSel cterm=bold,reverse" ATTR " ctermfg=default ctermbg=default\n"
		"User1      cterm=none ctermfg=default ctermbg=default\n"
		"User2      cterm=none ctermfg=default ctermbg=default\n"
		"User3      cterm=none ctermfg=default ctermbg=default\n"
		"User4      cterm=none ctermfg=default ctermbg=default\n"
		"User5      cterm=none ctermfg=default ctermbg=default\n"
		"User6      cterm=none ctermfg=default ctermbg=default\n"
		"User7      cterm=none ctermfg=default ctermbg=default\n"
		"User8      cterm=none ctermfg=default ctermbg=default\n"
		"User9      cterm=none ctermfg=default ctermbg=default\n"
		"User10     cterm=none ctermfg=default ctermbg=default\n"
		"User11     cterm=none ctermfg=default ctermbg=default\n"
		"User12     cterm=none ctermfg=default ctermbg=default\n"
		"User13     cterm=none ctermfg=default ctermbg=default\n"
		"User14     cterm=none ctermfg=default ctermbg=default\n"
		"User15     cterm=none ctermfg=default ctermbg=default\n"
		"User16     cterm=none ctermfg=default ctermbg=default\n"
		"User17     cterm=none ctermfg=default ctermbg=default\n"
		"User18     cterm=none ctermfg=default ctermbg=default\n"
		"User19     cterm=none ctermfg=default ctermbg=default\n"
		"User20     cterm=none ctermfg=default ctermbg=default\n"
		"OtherWin   cterm=none ctermfg=default ctermbg=default\n"
		"LineNr     cterm=none ctermfg=default ctermbg=default\n"
		"OddLine    cterm=none ctermfg=default ctermbg=default\n"
		"\n"
		"column:size cterm=bold ctermfg=red     ctermbg=red\n"
		"\n"
		"{*.jpg}    cterm=none ctermfg=red     ctermbg=blue";

	assert_success(cmds_dispatch("highlight {*.jpg} ctermfg=red\tctermbg=blue",
				&lwin, CIT_COMMAND));
	assert_success(cmds_dispatch(
				"highlight column:size ctermfg=red ctermbg=red cterm=bold", &lwin,
				CIT_COMMAND));

	ui_sb_msg("");
	assert_failure(cmds_dispatch("hi", &lwin, CIT_COMMAND));
	assert_string_equal(expected, ui_sb_last());
}

/* Colors. */

TEST(wrong_gui_color_causes_error)
{
	ui_sb_msg("");
	assert_failure(cmds_dispatch("hi Win guifg=#1234", &lwin, CIT_COMMAND));
	assert_string_equal("Unrecognized color value format: #1234", ui_sb_last());

	ui_sb_msg("");
	assert_failure(cmds_dispatch("hi Win guibg=#1234", &lwin, CIT_COMMAND));
	assert_string_equal("Unrecognized color value format: #1234", ui_sb_last());
}

TEST(gui_colors_are_parsed)
{
	assert_success(cmds_dispatch("hi Win guifg=#1234fe guibg=red gui=reverse",
				&lwin, CIT_COMMAND));
	assert_true(curr_stats.cs->color[WIN_COLOR].gui_set);
	assert_int_equal(0x1234fe, curr_stats.cs->color[WIN_COLOR].gui_fg);
	assert_int_equal(COLOR_RED, curr_stats.cs->color[WIN_COLOR].gui_bg);
	assert_int_equal(A_REVERSE, curr_stats.cs->color[WIN_COLOR].gui_attr);

	assert_success(cmds_dispatch("hi Win guifg=default", &lwin, CIT_COMMAND));
	assert_true(curr_stats.cs->color[WIN_COLOR].gui_set);
	assert_int_equal(-1, curr_stats.cs->color[WIN_COLOR].gui_fg);
	assert_int_equal(COLOR_RED, curr_stats.cs->color[WIN_COLOR].gui_bg);
	assert_int_equal(A_REVERSE, curr_stats.cs->color[WIN_COLOR].gui_attr);
}

TEST(gui_colors_are_printed)
{
	ui_sb_msg("");
	assert_success(cmds_dispatch("hi Win guifg=#1234fe guibg=red", &lwin,
				CIT_COMMAND));
	assert_failure(cmds_dispatch("hi Win", &lwin, CIT_COMMAND));
	assert_string_equal(
			"Win        cterm=none ctermfg=white   ctermbg=black\n"
			"           gui=none   guifg=#1234fe   guibg=red",
			ui_sb_last());
}

/* Attributes. */

TEST(wrong_attribute_causes_error)
{
	assert_failure(cmds_dispatch("hi Win cterm=bad", &lwin, CIT_COMMAND));
}

TEST(various_attributes_are_parsed)
{
#ifdef HAVE_A_ITALIC_DECL
	const unsigned int italic_attr = A_ITALIC;
#else
	/* If A_ITALIC is missing (it's an extension), use A_REVERSE instead. */
	const unsigned int italic_attr = A_REVERSE;
#endif

	curr_stats.cs->color[WIN_COLOR].attr = 0;

	assert_success(cmds_dispatch("hi Win cterm=bold,italic", &lwin, CIT_COMMAND));
	assert_int_equal(A_BOLD | italic_attr, curr_stats.cs->color[WIN_COLOR].attr);

	assert_success(cmds_dispatch("hi Win cterm=underline,reverse", &lwin,
				CIT_COMMAND));
	assert_int_equal(A_UNDERLINE | A_REVERSE,
			curr_stats.cs->color[WIN_COLOR].attr);

	assert_success(cmds_dispatch("hi Win cterm=standout,combine", &lwin,
				CIT_COMMAND));
	assert_int_equal(A_STANDOUT, curr_stats.cs->color[WIN_COLOR].attr);
	assert_true(curr_stats.cs->color[WIN_COLOR].combine_attrs);
}

TEST(attributes_are_printed_back_correctly)
{
	ui_sb_msg("");
	assert_failure(cmds_dispatch("highlight AuxWin", &lwin, CIT_COMMAND));
	assert_string_equal("AuxWin     cterm=none ctermfg=default ctermbg=default",
			ui_sb_last());

	assert_success(cmds_dispatch("highlight Win cterm=underline,inverse", &lwin,
				CIT_COMMAND));

	ui_sb_msg("");
	assert_success(cmds_dispatch("highlight AuxWin cterm=combine", &lwin,
				CIT_COMMAND));
	assert_failure(cmds_dispatch("highlight AuxWin", &lwin, CIT_COMMAND));
	assert_string_equal(
			"AuxWin     cterm=combine ctermfg=default ctermbg=default", ui_sb_last());

	assert_success(cmds_dispatch("highlight Win cterm=underline,inverse", &lwin,
				CIT_COMMAND));

	ui_sb_msg("");
	assert_failure(cmds_dispatch("highlight Win", &lwin, CIT_COMMAND));
	assert_string_equal(
			"Win        cterm=underline,reverse ctermfg=white   ctermbg=black",
			ui_sb_last());

	assert_success(cmds_dispatch("highlight Win cterm=italic,standout,bold",
				&lwin, CIT_COMMAND));

	ui_sb_msg("");
	assert_failure(cmds_dispatch("highlight Win", &lwin, CIT_COMMAND));
#ifdef HAVE_A_ITALIC_DECL
	assert_string_equal(
			"Win        cterm=bold,standout,italic ctermfg=white   ctermbg=black",
			ui_sb_last());
#else
	assert_string_equal(
			"Win        cterm=bold,reverse,standout ctermfg=white   ctermbg=black",
			ui_sb_last());
#endif
}

/* Generic groups. */

TEST(color_is_set)
{
	const char *const COMMANDS = "hi Win ctermfg=red ctermbg=red cterm=bold";

	curr_stats.cs->color[WIN_COLOR].fg = COLOR_BLUE;
	curr_stats.cs->color[WIN_COLOR].bg = COLOR_BLUE;
	curr_stats.cs->color[WIN_COLOR].attr = 0;
	assert_success(cmds_dispatch(COMMANDS, &lwin, CIT_COMMAND));
	assert_int_equal(COLOR_RED, curr_stats.cs->color[WIN_COLOR].fg);
	assert_int_equal(COLOR_RED, curr_stats.cs->color[WIN_COLOR].bg);
	assert_int_equal(A_BOLD, curr_stats.cs->color[WIN_COLOR].attr);
}

TEST(original_color_is_unchanged_on_parsing_error)
{
	const char *const COMMANDS = "highlight Win ctermfg=red ctersmbg=red";

	curr_stats.cs->color[WIN_COLOR].fg = COLOR_BLUE;
	assert_failure(cmds_dispatch(COMMANDS, &lwin, CIT_COMMAND));
	assert_int_equal(COLOR_BLUE, curr_stats.cs->color[WIN_COLOR].fg);
}

/* Column highlighting. */

TEST(column_name_is_wrong)
{
	curr_stats.vlua = vlua_init();

	ui_sb_msg("");
	assert_failure(cmds_dispatch("hi column:badone ctermfg=red", &lwin,
				CIT_COMMAND));
	assert_string_equal("No such column: badone", ui_sb_last());

	vlua_finish(curr_stats.vlua);
	curr_stats.vlua = NULL;
}

TEST(column_color_is_not_set)
{
	assert_null(cs_get_column_hi(curr_stats.cs, SK_BY_SIZE));
	assert_failure(cmds_dispatch("hi column:size ctermfg=bad", &lwin,
				CIT_COMMAND));
	assert_null(cs_get_column_hi(curr_stats.cs, SK_BY_SIZE));
}

TEST(column_color_is_set)
{
	assert_null(cs_get_column_hi(curr_stats.cs, SK_BY_SIZE));

	assert_success(cmds_dispatch(
				"hi column:size ctermfg=red ctermbg=red cterm=bold", &lwin,
				CIT_COMMAND));

	const col_attr_t *hi = cs_get_column_hi(curr_stats.cs, SK_BY_SIZE);
	assert_int_equal(COLOR_RED, hi->fg);
	assert_int_equal(COLOR_RED, hi->bg);
	assert_int_equal(A_BOLD, hi->attr);
}

TEST(skipped_column_color_is_not_set)
{
	assert_null(cs_get_column_hi(curr_stats.cs, SK_BY_NAME));
	assert_success(cmds_dispatch(
				"hi column:size ctermfg=red ctermbg=red cterm=bold", &lwin, CIT_COMMAND));
	assert_null(cs_get_column_hi(curr_stats.cs, SK_BY_NAME));
}

TEST(column_color_not_removed)
{
	curr_stats.vlua = vlua_init();

	ui_sb_msg("");
	assert_failure(cmds_dispatch("hi clear column:bad", &lwin, CIT_COMMAND));
	assert_string_equal("No such column: bad", ui_sb_last());

	vlua_finish(curr_stats.vlua);
	curr_stats.vlua = NULL;
}

TEST(column_color_is_removed)
{
	/* Nothing to remove yet. */
	ui_sb_msg("");
	assert_failure(cmds_dispatch("hi clear column:size", &lwin, CIT_COMMAND));
	assert_string_equal("No such group: column:size", ui_sb_last());

	assert_success(cmds_dispatch("hi column:size ctermfg=red cterm=bold", &lwin,
				CIT_COMMAND));
	assert_non_null(cs_get_column_hi(curr_stats.cs, SK_BY_SIZE));

	assert_success(cmds_dispatch("hi clear column:size", &lwin, CIT_COMMAND));
	assert_null(cs_get_column_hi(curr_stats.cs, SK_BY_SIZE));
}

TEST(column_color_is_printed)
{
	ui_sb_msg("");
	assert_success(cmds_dispatch("hi column:size", &lwin, CIT_COMMAND));
	assert_string_equal("", ui_sb_last());

	assert_success(cmds_dispatch("hi column:size ctermfg=red cterm=bold", &lwin,
				CIT_COMMAND));
	assert_non_null(cs_get_column_hi(curr_stats.cs, SK_BY_SIZE));

	ui_sb_msg("");
	assert_failure(cmds_dispatch("hi column:size", &lwin, CIT_COMMAND));
	assert_string_equal("column:size cterm=bold ctermfg=red     ctermbg=default",
			ui_sb_last());
}

/* File-specific highlight. */

TEST(empty_curly_braces)
{
	const char *const COMMANDS = "highlight {} ctermfg=red";

	assert_false(cmds_dispatch(COMMANDS, &lwin, CIT_COMMAND) == 0);
}

TEST(curly_braces_pattern_transform)
{
	const char *const COMMANDS = "highlight {*.sh}<inode/directory> ctermfg=red";

	assert_int_equal(0, cmds_dispatch(COMMANDS, &lwin, CIT_COMMAND));
	assert_string_equal("{*.sh}<inode/directory>",
			matchers_get_expr(cfg.cs.file_hi[0].matchers));
}

TEST(curly_braces_no_flags_allowed)
{
	const char *const COMMANDS = "highlight {*.sh}i ctermfg=red";

	assert_false(cmds_dispatch(COMMANDS, &lwin, CIT_COMMAND) == 0);
}

TEST(empty_re_without_flags)
{
	const char *const COMMANDS = "highlight // ctermfg=red";

	assert_false(cmds_dispatch(COMMANDS, &lwin, CIT_COMMAND) == 0);
}

TEST(empty_re_with_flags)
{
	const char *const COMMANDS = "highlight //i ctermfg=red";

	assert_false(cmds_dispatch(COMMANDS, &lwin, CIT_COMMAND) == 0);
}

TEST(pattern_is_not_unescaped)
{
	const char *const COMMANDS = "highlight /^\\./ ctermfg=red";

	assert_int_equal(0, cmds_dispatch(COMMANDS, &lwin, CIT_COMMAND));
	assert_string_equal("/^\\./", matchers_get_expr(cfg.cs.file_hi[0].matchers));
}

TEST(pattern_length_is_not_limited)
{
	const char *const COMMANDS = "highlight /\\.(7z|Z|a|ace|alz|apkg|arc|arj|bz"
		"|bz2|cab|cpio|deb|gz|jar|lha|lrz|lz|lzma|lzo|rar|rpm|rz|t7z|tZ|tar|tbz"
		"|tbz2|tgz|tlz|txz|tzo|war|xz|zip)$/ ctermfg=red";

	assert_int_equal(0, cmds_dispatch(COMMANDS, &lwin, CIT_COMMAND));
	assert_string_equal("/\\.(7z|Z|a|ace|alz|apkg|arc|arj|bz"
		"|bz2|cab|cpio|deb|gz|jar|lha|lrz|lz|lzma|lzo|rar|rpm|rz|t7z|tZ|tar|tbz"
		"|tbz2|tgz|tlz|txz|tzo|war|xz|zip)$/",
		matchers_get_expr(cfg.cs.file_hi[0].matchers));
}

TEST(i_flag)
{
	const char *const COMMANDS = "highlight /^\\./i ctermfg=red";

	assert_int_equal(0, cmds_dispatch(COMMANDS, &lwin, CIT_COMMAND));
	assert_string_equal("/^\\./i", matchers_get_expr(cfg.cs.file_hi[0].matchers));
}

TEST(I_flag)
{
	const char *const COMMANDS = "highlight /^\\./I ctermfg=red";

	assert_int_equal(0, cmds_dispatch(COMMANDS, &lwin, CIT_COMMAND));
	assert_string_equal("/^\\./I", matchers_get_expr(cfg.cs.file_hi[0].matchers));
}

TEST(wrong_flag)
{
	const char *const COMMANDS = "highlight /^\\./x ctermfg=red";

	assert_int_equal(-1, cmds_dispatch(COMMANDS, &lwin, CIT_COMMAND));
}

TEST(negation)
{
	const char *const COMMANDS = "highlight !/^\\./i ctermfg=red";

	assert_success(cmds_dispatch(COMMANDS, &lwin, CIT_COMMAND));
	assert_string_equal("!/^\\./i",
			matchers_get_expr(cfg.cs.file_hi[0].matchers));
}

TEST(highlighting_is_printed_back_correctly)
{
	const char *const COMMANDS = "highlight {*.jpg} ctermfg=red";
	assert_success(cmds_dispatch(COMMANDS, &lwin, CIT_COMMAND));

	ui_sb_msg("");
	assert_failure(cmds_dispatch("highlight {*.jpg}", &lwin, CIT_COMMAND));
	assert_string_equal("{*.jpg}    cterm=none ctermfg=red     ctermbg=default",
			ui_sb_last());
}

TEST(existing_records_are_updated)
{
	const char *const COMMANDS1 = "highlight {*.jpg} ctermfg=red";
	const char *const COMMANDS2 = "highlight {*.jpg} ctermfg=blue";

	assert_success(cmds_dispatch(COMMANDS1, &lwin, CIT_COMMAND));
	assert_success(cmds_dispatch(COMMANDS2, &lwin, CIT_COMMAND));
	assert_int_equal(1, cfg.cs.file_hi_count);

	assert_int_equal(COLOR_BLUE, cfg.cs.file_hi[0].hi.fg);
}

TEST(all_records_can_be_removed)
{
	const char *const COMMANDS1 = "highlight {*.jpg} ctermfg=red";
	const char *const COMMANDS2 = "highlight {*.avi} cterm=bold";
	const char *const COMMANDS3 = "highlight clear";

	assert_success(cmds_dispatch(COMMANDS1, &lwin, CIT_COMMAND));
	assert_success(cmds_dispatch(COMMANDS2, &lwin, CIT_COMMAND));
	assert_int_equal(2, cfg.cs.file_hi_count);
	assert_success(cmds_dispatch(COMMANDS3, &lwin, CIT_COMMAND));
	assert_int_equal(0, cfg.cs.file_hi_count);
}

TEST(records_can_be_removed)
{
	const char *const COMMANDS1 = "highlight {*.jpg} ctermfg=red";
	const char *const COMMANDS2 = "highlight clear {*.avi}";
	const char *const COMMANDS3 = "highlight clear {*.jpg}";

	assert_success(cmds_dispatch(COMMANDS1, &lwin, CIT_COMMAND));
	assert_int_equal(1, cfg.cs.file_hi_count);
	assert_failure(cmds_dispatch(COMMANDS2, &lwin, CIT_COMMAND));
	assert_int_equal(1, cfg.cs.file_hi_count);
	assert_success(cmds_dispatch(COMMANDS3, &lwin, CIT_COMMAND));
	assert_int_equal(0, cfg.cs.file_hi_count);
}

TEST(incorrect_highlight_groups_are_not_added)
{
	const char *const COMMANDS = "highlight {*.jpg} ctersmfg=red";
	assert_failure(cmds_dispatch(COMMANDS, &lwin, CIT_COMMAND));
	assert_int_equal(0, cfg.cs.file_hi_count);
}

TEST(can_color_uncolored_file)
{
	view_setup(&lwin);
	make_abs_path(lwin.curr_dir, sizeof(lwin.curr_dir), TEST_DATA_PATH,
			"color-schemes", NULL);

	assert_success(populate_dir_list(&lwin, 0));

	curr_stats.cs = &cfg.cs;
	curr_stats.load_stage = 2;

	assert_null(cs_get_file_hi(curr_stats.cs, "some.vifm",
				&lwin.dir_entry[0].hi_num));
	assert_int_equal(INT_MAX, lwin.dir_entry[0].hi_num);

	assert_success(cmds_dispatch("highlight {*.vifm} cterm=bold", &lwin,
				CIT_COMMAND));
	assert_non_null(cs_get_file_hi(curr_stats.cs, "some.vifm",
				&lwin.dir_entry[0].hi_num));
	assert_int_equal(0, lwin.dir_entry[0].hi_num);

	view_teardown(&lwin);
	curr_stats.load_stage = 0;
}

TEST(tabs_are_allowed)
{
	const char *const COMMANDS1 = "highlight\t{*.jpg} ctermfg=red\tctermbg=blue";
	const char *const COMMANDS2 = "highlight {*.avi}\tctermfg=red";
	const char *const COMMANDS3 = "highlight\t{*.mp3}\tctermfg=red";

	assert_success(cmds_dispatch(COMMANDS1, &lwin, CIT_COMMAND));
	assert_int_equal(1, cfg.cs.file_hi_count);
	assert_success(cmds_dispatch(COMMANDS2, &lwin, CIT_COMMAND));
	assert_int_equal(2, cfg.cs.file_hi_count);
	assert_success(cmds_dispatch(COMMANDS3, &lwin, CIT_COMMAND));
	assert_int_equal(3, cfg.cs.file_hi_count);

	if(cfg.cs.file_hi_count > 0)
	{
		assert_string_equal("{*.jpg}",
				matchers_get_expr(cfg.cs.file_hi[0].matchers));
	}
	if(cfg.cs.file_hi_count > 1)
	{
		assert_string_equal("{*.avi}",
				matchers_get_expr(cfg.cs.file_hi[1].matchers));
	}
	if(cfg.cs.file_hi_count > 2)
	{
		assert_string_equal("{*.mp3}",
				matchers_get_expr(cfg.cs.file_hi[2].matchers));
	}
}

/* vim: set tabstop=2 softtabstop=2 shiftwidth=2 noexpandtab cinoptions-=(0 : */
/* vim: set cinoptions+=t0 filetype=c : */
