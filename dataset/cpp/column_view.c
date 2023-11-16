/* vifm
 * Copyright (C) 2012 xaizek.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA
 */

#include "column_view.h"

#include <assert.h> /* assert() */
#include <stddef.h> /* NULL size_t */
#include <stdlib.h> /* malloc() free() */
#include <string.h> /* memmove() memset() strcpy() strdup() strlen() */

#include "../compat/reallocarray.h"
#include "../utils/macros.h"
#include "../utils/str.h"
#include "../utils/utf8.h"
#include "../utils/utils.h"

/* Character used to fill gaps in lines. */
#define GAP_FILL_CHAR ' '

/* Holds general column information. */
typedef struct
{
	int column_id;    /* Unique column id. */
	column_func func; /* Function, which prints column value. */
	void *data;       /* Data to be passed to the function. */
}
column_desc_t;

/* Column information including calculated values. */
typedef struct
{
	column_info_t info; /* Column properties specified by the client. */
	int start;          /* Start position of the column. */
	int width;          /* Calculated width of the column. */
	int print_width;    /* Print width (less or equal to the width field). */
	column_desc_t desc; /* Column description. */
}
column_t;

/* Column view description structure.  Typedef is in the header file. */
struct columns_t
{
	int max_width;  /* Maximum width of one line of the view. */
	int count;      /* Number of columns in the list. */
	column_t *list; /* Array of columns of count length. */
};

static int extend_column_desc_list(void);
static void init_new_column_desc(column_desc_t *desc, int column_id,
		column_func func, void *data);
static int column_id_present(int column_id);
static int extend_column_list(columns_t *cols);
static void init_new_column(column_t *col, column_info_t info);
static void mark_for_recalculation(columns_t *cols);
static const column_desc_t * get_column_func(int column_id);
static void adjust_match_range(size_t cut_from, size_t cut_to, size_t ell_len,
		size_t *match_from, size_t *match_to, int *use_marks);
static AlignType decorate_output(const column_t *col, format_info_t *info,
		char buf[], size_t buf_len, int max_line_width);
static int calculate_max_width(const column_t *col, int len,
		int max_line_width);
static int calculate_start_pos(const column_t *col, const char buf[],
		AlignType align);
static void fill_gap_pos(void *format_data, int from, int to, int column_id);
static int get_width_on_screen(const char str[]);
static void recalculate_if_needed(columns_t *cols, int max_width);
static void recalculate(columns_t *cols, int max_width);
static void update_widths(columns_t *cols, int max_width);
static int update_abs_and_rel_widths(columns_t *cols, int *max_width);
static void update_auto_widths(columns_t *cols, int auto_count, int max_width);
static void update_start_positions(columns_t *cols);

/* Number of registered column descriptors. */
static int col_desc_count;
/* List of column descriptors. */
static column_desc_t *col_descs;
/* Column print function. */
static column_line_print_func print_func;
/* Optional function to query column match. */
static column_line_match_func match_func;
/* String to be used in place of ellipsis. */
static const char *ellipsis = "...";
/* Strings to be used when highlighted text goes out of bounds. */
static const char *left_marks = "<<<";
static const char *right_marks = ">>>";
static const char *middle_marks = ">..<";

void
columns_set_line_print_func(column_line_print_func func)
{
	print_func = func;
}

void
columns_set_line_match_func(column_line_match_func func)
{
	match_func = func;
}

void
columns_set_ellipsis(const char ell[])
{
	ellipsis = ell;
}

int
columns_add_column_desc(int column_id, column_func func, void *data)
{
	if(!column_id_present(column_id) && extend_column_desc_list() == 0)
	{
		init_new_column_desc(&col_descs[col_desc_count - 1], column_id, func, data);
		return 0;
	}
	return 1;
}

/* Extends column descriptors list by one element, but doesn't initialize it at
 * all.  Returns zero on success. */
static int
extend_column_desc_list(void)
{
	column_desc_t *mem_ptr;
	mem_ptr = reallocarray(col_descs, col_desc_count + 1, sizeof(*mem_ptr));
	if(mem_ptr == NULL)
	{
		return 1;
	}
	col_descs = mem_ptr;
	col_desc_count++;
	return 0;
}

/* Fills column description structure with initial values. */
static void
init_new_column_desc(column_desc_t *desc, int column_id, column_func func,
		void *data)
{
	desc->column_id = column_id;
	desc->func = func;
	desc->data = data;
}

columns_t *
columns_create(void)
{
	columns_t *const result = malloc(sizeof(*result));
	if(result == NULL)
	{
		return NULL;
	}
	result->count = 0;
	result->list = NULL;
	mark_for_recalculation(result);
	return result;
}

void
columns_free(columns_t *cols)
{
	if(cols != NULL)
	{
		columns_clear(cols);
		free(cols);
	}
}

void
columns_clear(columns_t *cols)
{
	int i;
	for(i = 0U; i < cols->count; ++i)
	{
		free(cols->list[i].info.literal);
	}

	free(cols->list);
	cols->list = NULL;
	cols->count = 0;
}

void
columns_clear_column_descs(void)
{
	free(col_descs);
	col_descs = NULL;
	col_desc_count = 0;
}

void
columns_add_column(columns_t *cols, column_info_t info)
{
	assert(info.text_width <= info.full_width &&
			"Text width should be bigger than full width.");
	assert(column_id_present(info.column_id) && "Unknown column id.");
	if(extend_column_list(cols) == 0)
	{
		init_new_column(&cols->list[cols->count - 1], info);
		mark_for_recalculation(cols);
	}
}

/* Checks whether a column with such column_id is already in the list of column
 * descriptors. Returns non-zero if yes. */
static int
column_id_present(int column_id)
{
	if(column_id == FILL_COLUMN_ID)
	{
		/* This pseudo-column is always "present". */
		return 1;
	}

	int i;
	/* Validate column_id. */
	for(i = 0; i < col_desc_count; i++)
	{
		if(col_descs[i].column_id == column_id)
		{
			return 1;
		}
	}
	return 0;
}

/* Extends columns list by one element, but doesn't initialize it at all.
 * Returns zero on success. */
static int
extend_column_list(columns_t *cols)
{
	column_t *mem_ptr;
	mem_ptr = reallocarray(cols->list, cols->count + 1, sizeof(*mem_ptr));
	if(mem_ptr == NULL)
	{
		return 1;
	}
	cols->list = mem_ptr;
	cols->count++;
	return 0;
}

/* Marks columns structure as one that need to be recalculated. */
static void
mark_for_recalculation(columns_t *cols)
{
	cols->max_width = -1;
}

/* Fills column structure with initial values. */
static void
init_new_column(column_t *col, column_info_t info)
{
	col->info = info;
	if(info.literal != NULL)
	{
		col->info.literal = strdup(info.literal);
	}

	col->start = -1;
	col->width = -1;
	col->print_width = -1;

	const column_desc_t *desc = get_column_func(info.column_id);
	if(desc != NULL)
	{
		col->desc = *desc;
	}
}

/* Returns a pointer to column formatting function by the column id or NULL on
 * unknown column_id. */
static const column_desc_t *
get_column_func(int column_id)
{
	int i;
	for(i = 0; i < col_desc_count; i++)
	{
		column_desc_t *col_desc = &col_descs[i];
		if(col_desc->column_id == column_id)
		{
			return col_desc;
		}
	}

	assert((column_id == FILL_COLUMN_ID) && "Unknown column id");
	return NULL;
}

void
columns_format_line(columns_t *cols, void *format_data, int max_line_width)
{
	char prev_col_buf[1024 + 1];
	int prev_col_start = 0;
	prev_col_buf[0] = '\0';

	int i;
	int prev_col_end = 0;
	int prev_col_id = FILL_COLUMN_ID;

	recalculate_if_needed(cols, max_line_width);

	for(i = 0; i < cols->count; ++i)
	{
		/* Use big buffer to hold whole item so there will be no issues with right
		 * aligned fields. */
		char col_buffer[sizeof(prev_col_buf)];
		char full_column[sizeof(prev_col_buf)];
		const column_t *const col = &cols->list[i];
		format_info_t info = {
			.data = format_data,
			.id = col->info.column_id,
			.real_id = col->info.column_id,
			.width = col->print_width,
			.match_from = -1,
			.match_to = -1,
		};

		if(col->info.literal == NULL)
		{
			col->desc.func(col->desc.data, sizeof(col_buffer), col_buffer, &info);
		}
		else
		{
			copy_str(col_buffer, sizeof(col_buffer), col->info.literal);
		}

		if(match_func != NULL)
		{
			match_func(col_buffer, &info, &info.match_from, &info.match_to);
		}

		strcpy(full_column, col_buffer);

		AlignType align = decorate_output(col, &info, col_buffer,
				sizeof(col_buffer), max_line_width);
		const int cur_col_start = calculate_start_pos(col, col_buffer, align);
		int print_start = MIN(cur_col_start, col->start);

		/* Ensure that we are not trying to draw current column in the middle of a
		 * character inside previous column. */
		if(prev_col_end > print_start)
		{
			const int prev_col_max_width = (print_start > prev_col_start)
			                             ? (print_start - prev_col_start)
			                             : 0;
			const size_t break_point = utf8_strsnlen(prev_col_buf,
					prev_col_max_width);
			prev_col_buf[break_point] = '\0';
			int real_prev_end = prev_col_start + get_width_on_screen(prev_col_buf);
			fill_gap_pos(format_data, real_prev_end, print_start, prev_col_id);
			print_start = prev_col_end;
		}
		else
		{
			fill_gap_pos(format_data, prev_col_end, print_start, prev_col_id);
		}

		/* Gap filling is done per column.  This one is for the current one. */
		fill_gap_pos(format_data, print_start, cur_col_start, col->info.column_id);

		print_func(col_buffer, cur_col_start, align, full_column, &info);

		prev_col_end = cur_col_start + get_width_on_screen(col_buffer);
		prev_col_id = col->info.column_id;

		/* Store information about the current column for usage on the next
		 * iteration. */
		strcpy(prev_col_buf, col_buffer);
		prev_col_start = cur_col_start;
	}

	fill_gap_pos(format_data, prev_col_end, max_line_width, prev_col_id);
}

/* Adds decorations like ellipsis to the output.  Returns actual align type used
 * for the column (might not match col->info.align). */
static AlignType
decorate_output(const column_t *col, format_info_t *info, char buf[],
		size_t buf_len, int max_line_width)
{
	char * (*add_ellipsis)(const char str[], size_t max_width, const char ell[]);
	void (*get_cut_range)(const char str[], size_t max_width, size_t *cut_from,
			size_t *cut_to);

	const int len = get_width_on_screen(buf);
	const int max_col_width = calculate_max_width(col, len, max_line_width);
	const int too_long = len > max_col_width;
	AlignType result;
	const char *ell = (col->info.cropping == CT_ELLIPSIS ? ellipsis : "");
	const char *marks;
	char *ellipsed;

	if(!too_long)
	{
		return (col->info.align == AT_RIGHT ? AT_RIGHT : AT_LEFT);
	}

	if(col->info.align == AT_MIDDLE)
	{
		add_ellipsis = middle_ellipsis;
		get_cut_range = get_middle_cut_range;

		marks = middle_marks;
		result = AT_LEFT;
	}
	else if(col->info.align == AT_LEFT)
	{
		add_ellipsis = right_ellipsis;
		get_cut_range = get_right_cut_range;

		marks = right_marks;
		result = AT_LEFT;
	}
	else
	{
		add_ellipsis = left_ellipsis;
		get_cut_range = get_left_cut_range;

		marks = left_marks;
		result = AT_RIGHT;
	}

	if(info->match_from != info->match_to)
	{
		/* If the line is cut, search match offsets might need to be adjusted. */
		size_t match_from = info->match_from;
		size_t match_to = info->match_to;
		size_t cut_from = 0;
		size_t cut_to = 0;
		size_t ell_len = strlen(ell);
		size_t ell_width = utf8_strsw(ell);
		int use_marks;

		get_cut_range(buf, max_col_width - ell_width, &cut_from, &cut_to);
		adjust_match_range(cut_from, cut_to, ell_len, &match_from, &match_to,
				&use_marks);

		if(use_marks)
		{
			ell = marks;
			size_t new_ell_len = strlen(ell);
			size_t new_ell_width = utf8_strsw(ell);
			if(new_ell_len != ell_len || new_ell_width != ell_width)
			{
				/* If ellipsis length or width has changed, readjust match range. */
				match_from = info->match_from;
				match_to = info->match_to;
				get_cut_range(buf, max_col_width - new_ell_width, &cut_from, &cut_to);
				adjust_match_range(cut_from, cut_to, new_ell_len, &match_from,
						&match_to, &use_marks);
			}
		}

		info->match_from = match_from;
		info->match_to = match_to;
	}

	ellipsed = add_ellipsis(buf, max_col_width, ell);

	copy_str(buf, buf_len, ellipsed);
	free(ellipsed);

	return result;
}

/* Adjusts search match offsets according to cut range and ellipsis length.
 * Sets use_marks argument to non-zero if match range overlaps ellipsis,
 * otherwise to zero. */
static void
adjust_match_range(size_t cut_from, size_t cut_to, size_t ell_len,
		size_t *match_from, size_t *match_to, int *use_marks)
{
	*use_marks = 0;

	if(*match_from >= cut_from && *match_from < cut_to)
	{
		*match_from = cut_from;
		*use_marks = 1;
	}
	else if(*match_from >= cut_to)
	{
		*match_from = cut_from + ell_len + *match_from - cut_to;
	}

	if(*match_to > cut_from && *match_to <= cut_to)
	{
		*match_to = cut_from + ell_len;
		*use_marks = 1;
	}
	else if(*match_to > cut_to)
	{
		*match_to = cut_from + ell_len + *match_to - cut_to;
	}

	if(*match_from < cut_from && *match_to > cut_from)
	{
		*use_marks = 1;
	}
}

/* Calculates maximum width for outputting content of the col. */
static int
calculate_max_width(const column_t *col, int len, int max_line_width)
{
	if(col->info.cropping != CT_NONE)
	{
		return col->print_width;
	}

	const int left_bound = (col->info.align == AT_LEFT ? col->start : 0);
	const int max_col_width = max_line_width - left_bound;
	return MIN(len, max_col_width);
}

/* Calculates start position for outputting content of the col. */
static int
calculate_start_pos(const column_t *col, const char buf[], AlignType align)
{
	if(align == AT_LEFT)
	{
		return col->start;
	}

	const int end = col->start + col->width;
	const int len = get_width_on_screen(buf);
	return (end > len && align == AT_RIGHT) ? (end - len) : 0;
}

/* Prints gap filler (GAP_FILL_CHAR) in place of gaps.  Does nothing if to less
 * or equal to from. */
static void
fill_gap_pos(void *format_data, int from, int to, int column_id)
{
	if(to > from)
	{
		char gap[to - from + 1];
		memset(gap, GAP_FILL_CHAR, to - from);
		gap[to - from] = '\0';

		const format_info_t info = {
			.data = format_data,
			.id = FILL_COLUMN_ID,
			.real_id = column_id,
			.width = to - from,
		};
		print_func(gap, from, AT_LEFT, gap, &info);
	}
}

/* Returns number of character positions allocated by the string on the
 * screen.  On issues will try to do the best, to make visible at least
 * something. */
static int
get_width_on_screen(const char str[])
{
	int length = -1;

	wchar_t *const wide = to_wide(str);
	if(wide != NULL)
	{
		length = vifm_wcswidth(wide, (size_t)-1);
		free(wide);
	}

	if(length < 0)
	{
		length = utf8_nstrlen(str);
	}
	return length;
}

/* Checks if recalculation is needed and runs it if yes. */
static void
recalculate_if_needed(columns_t *cols, int max_width)
{
	if(!columns_matches_width(cols, max_width))
	{
		recalculate(cols, max_width);
	}
}

int
columns_matches_width(const columns_t *cols, int max_width)
{
	return cols->max_width == max_width;
}

/* Recalculates column widths and start offsets. */
static void
recalculate(columns_t *cols, int max_width)
{
	update_widths(cols, max_width);
	update_start_positions(cols);

	cols->max_width = max_width;
}

/* Recalculates column widths. */
static void
update_widths(columns_t *cols, int max_width)
{
	int width_left = max_width;
	const int auto_count = update_abs_and_rel_widths(cols, &width_left);
	update_auto_widths(cols, auto_count, width_left);
}

/* Recalculates widths of columns with absolute or percent widths.  Returns
 * number of columns with auto size type. */
static int
update_abs_and_rel_widths(columns_t *cols, int *max_width)
{
	int i;
	int auto_count = 0;
	int percent_count = 0;
	int width_left = *max_width;
	for(i = 0; i < cols->count; i++)
	{
		int effective_width;
		column_t *col = &cols->list[i];
		if(col->info.sizing == ST_ABSOLUTE)
		{
			effective_width = MIN(col->info.full_width, width_left);
		}
		else if(col->info.sizing == ST_PERCENT)
		{
			++percent_count;
			effective_width = MIN(col->info.full_width**max_width/100, width_left);
		}
		else
		{
			auto_count++;
			continue;
		}

		width_left -= effective_width;
		col->width = effective_width;
		if(col->info.sizing == ST_ABSOLUTE)
		{
			col->print_width = MIN(effective_width, col->info.text_width);
		}
		else
		{
			col->print_width = col->width;
		}
	}

	/* When there is no columns with automatic size, give unused size to last
	 * percent column if one exists.  This removes right "margin", which otherwise
	 * present. */
	if(auto_count == 0U && percent_count != 0U)
	{
		int i;
		for(i = cols->count - 1; i >= 0; --i)
		{
			column_t *col = &cols->list[i];
			if(col->info.sizing == ST_PERCENT)
			{
				col->width += width_left;
				col->print_width += width_left;
				width_left = 0U;
			}
		}
	}

	*max_width = width_left;
	return auto_count;
}

/* Recalculates widths of columns with automatic widths. */
static void
update_auto_widths(columns_t *cols, int auto_count, int max_width)
{
	int i;
	int auto_left = auto_count;
	int left = max_width;
	for(i = 0; i < cols->count; i++)
	{
		column_t *col = &cols->list[i];
		if(col->info.sizing == ST_AUTO)
		{
			auto_left--;

			col->width = (auto_left > 0) ? max_width/auto_count : left;
			col->print_width = col->width;

			left -= col->width;
		}
	}
}

/* Should be used after updating column widths. */
static void
update_start_positions(columns_t *cols)
{
	int i;
	for(i = 0; i < cols->count; i++)
	{
		if(i == 0)
		{
			cols->list[i].start = 0;
		}
		else
		{
			column_t *prev_col = &cols->list[i - 1];
			cols->list[i].start = prev_col->start + prev_col->width;
		}
	}
}

/* vim: set tabstop=2 softtabstop=2 shiftwidth=2 noexpandtab cinoptions-=(0 : */
/* vim: set cinoptions+=t0 filetype=c : */
