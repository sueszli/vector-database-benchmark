/* -*- mode:c -*- */

/* Copyright (C) 2002-2023 Alexander Chernov <cher@ejudge.ru> */

/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 */

#include "ejudge/config.h"
#include "ejudge/ej_types.h"
#include "ejudge/userlist_clnt.h"
#include "ejudge/userlist_proto.h"
#include "ejudge/contests.h"
#include "ejudge/userlist.h"
#include "ejudge/ejudge_cfg.h"
#include "ejudge/xml_utils.h"
#include "ejudge/misctext.h"
#include "ejudge/ncurses_utils.h"
#include "ejudge/compat.h"

#include "ejudge/xalloc.h"
#include "ejudge/logger.h"
#include "ejudge/osdeps.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include <time.h>
#include <ncurses.h>
#include <menu.h>
#include <panel.h>
#include <locale.h>
#include <errno.h>
#include <regex.h>
#include <langinfo.h>
#include <ctype.h>

/* special pseudofields which are used for user info list markup */
enum
{
  USERLIST_PSEUDO_FIRST = 500,
  USERLIST_PSEUDO_PASSWORDS = USERLIST_PSEUDO_FIRST,
  USERLIST_PSEUDO_GENERAL_INFO,
  USERLIST_PSEUDO_FLAGS,
  USERLIST_PSEUDO_TIMESTAMPS,

  USERLIST_PSEUDO_LAST,
};

#define _(x) x
static char const * const member_string[] =
{
  _("Contestant"),
  _("Reserve"),
  _("Coach"),
  _("Advisor"),
  _("Guest")
};
static char const * const member_string_pl[] =
{
  _("Contestants"),
  _("Reserves"),
  _("Coaches"),
  _("Advisors"),
  _("Guests")
};
static char const * const member_status_string[] =
{
  _("Empty"),
  _("School student"),
  _("Student"),
  _("Magistrant"),
  _("PhD student"),
  _("School teacher"),
  _("Professor"),
  _("Scientist"),
  _("Other")
};
static char const * const member_gender_string[] =
{
  _("Empty"),
  _("Male"),
  _("Female"),
};
#undef _

/* various sort criteria for participants */
enum
  {
    PART_SORT_NONE,                  /* no sort */
    PART_SORT_ID,
    PART_SORT_ID_REV,
    PART_SORT_LOGIN,
    PART_SORT_LOGIN_REV,
    PART_SORT_NAME,
    PART_SORT_NAME_REV,
    PART_SORT_LAST
  };

/* search flags */
enum
  {
    SRCH_REPEAT,
    SRCH_REGEX_LOGIN_FORWARD,
    SRCH_REGEX_LOGIN_BACKWARD,
    SRCH_REGEX_NAME_FORWARD,
    SRCH_REGEX_NAME_BACKWARD,
    SRCH_REGEX_TEXT_FORWARD,
    SRCH_REGEX_TEXT_BACKWARD,
    SRCH_LAST
  };

static struct userlist_clnt *server_conn;
static struct ejudge_cfg *config;
static int utf8_mode = 0;

static int
display_user_menu(unsigned char *upper, int start_item, int only_choose);
static int
display_contests_menu(unsigned char *upper, int only_choose);

static void
print_help(char const *help)
{
  wattrset(stdscr, COLOR_PAIR(3));
  wbkgdset(stdscr, COLOR_PAIR(3));
  mvwaddstr(stdscr, LINES - 1, 0, help);
  wclrtoeol(stdscr);
  wattrset(stdscr, COLOR_PAIR(1));
  wbkgdset(stdscr, COLOR_PAIR(1));
}

static void
vis_err(unsigned char const *fmt, ...)
{
  unsigned char buf[1024];
  int buflen;
  va_list args;
  int req_cols, req_lines, first_line, first_col;
  WINDOW *out_win, *in_win;
  PANEL *out_pan, *in_pan;

  va_start(args, fmt);
  buflen = vsnprintf(buf, sizeof(buf), fmt, args);
  va_end(args);

  if (buflen > COLS - 2) {
    req_lines = buflen / (COLS - 2) + 1;
    req_cols = COLS - 2;
  } else {
    req_lines = 1;
    req_cols = buflen;
  }
  first_line = (LINES - req_lines - 2) / 2;
  first_col = (COLS - req_cols - 2) / 2;
  out_win = newwin(req_lines + 2, req_cols + 2, first_line, first_col);
  in_win = newwin(req_lines, req_cols, first_line + 1, first_col + 1);
  wattrset(out_win, COLOR_PAIR(4));
  wbkgdset(out_win, COLOR_PAIR(4));
  wattrset(in_win, COLOR_PAIR(4));
  wbkgdset(in_win, COLOR_PAIR(4));
  wclear(in_win);
  wclear(out_win);
  box(out_win, 0, 0);
  waddstr(in_win, buf);
  out_pan = new_panel(out_win);
  in_pan = new_panel(in_win);
  update_panels();
  print_help("Press any key to continue");
  doupdate();
  (void) getch();
  del_panel(in_pan);
  del_panel(out_pan);
  delwin(out_win);
  delwin(in_win);
  update_panels();
  doupdate();
}

static int
generic_menu(int min_width, int max_width, /* incl. frame */
             int min_height, int max_height, /* incl. frame */
             int first_item, int nitems,
             int rec_line, int rec_col,
             unsigned char const * const *items,
             unsigned char const * const *hotkeys,
             unsigned char const *help_str,
             unsigned char const *format, ...)
{
  unsigned char buf[1024];
  int buflen, i, itemlen, c, answer = -3, cmd;
  va_list args;
  int act_width, item_width, act_height, head_width;
  unsigned char **item_strs = NULL;
  unsigned char const *pc;
  ITEM **curs_items = NULL;
  MENU *curs_menu;
  WINDOW *in_win, *out_win, *txt_win;
  PANEL *in_pan, *out_pan, *txt_pan;

  ASSERT(items);
  ASSERT(nitems >= 1);
  for (i = 0; i < nitems; i++) {
    ASSERT(items[i]);
  }

  /* FIXME: we cannot scroll yet */
  ASSERT(nitems + 3 <= LINES - 2);

  if (max_width > COLS - 2 || max_width < 4) {
    max_width = COLS - 2;
  }
  if (min_width < 4 || min_width > max_width) {
    min_width = 4;
  }
  if (max_height > LINES - 2 || max_height < 4) {
    max_height = LINES - 2;
  }
  if (min_height < 4 || min_height > max_height) {
    min_height = 4;
  }

  va_start(args, format);
  vsnprintf(buf, sizeof(buf), format, args);
  va_end(args);
  buflen = strlen(buf);
  if (buflen > max_width - 2) {
    buf[max_width - 2] = 0;
    buflen = max_width - 2;
    head_width = max_width - 2;
  } else if (buflen < min_height - 2) {
    head_width = min_height - 2;
  } else {
    head_width = buflen;
  }
  act_width = head_width;

  item_width = -1;
  for (i = 0; i < nitems; i++) {
    itemlen = strlen(items[i]);
    if (itemlen > max_width - 3) {
      itemlen = max_width - 3;
    }
    if (itemlen > item_width) {
      item_width = itemlen;
    }
  }
  ASSERT(item_width > 0);

  XCALLOC(item_strs, nitems);
  for (i = 0; i < nitems; i++) {
    item_strs[i] = malloc(item_width + 1);
    memset(item_strs[i], ' ', item_width);
    item_strs[i][item_width] = 0;
    itemlen = strlen(items[i]);
    if (itemlen > item_width) {
      itemlen = item_width;
    }
    memcpy(item_strs[i], items[i], itemlen);
  }

  /* FIXME: too dumb */
  act_height = nitems + 1;

  if (item_width + 1 > act_width) {
    act_width = item_width + 1;
  }
  if (rec_col < 0 || rec_col >= COLS) {
    rec_col = (COLS - 2 - act_width) / 2;
  }
  if (rec_col + act_width + 2 >= COLS) {
    rec_col = COLS - 3 - act_width;
  }
  if (rec_col < 0) {
    rec_col = 0;
  }
  if (rec_line < 1 || rec_line >= LINES - 1) {
    rec_line = (LINES - 4 - act_height) / 2 + 1;
  }
  if (rec_line + act_height + 2 >= LINES) {
    rec_line = LINES - 3 - act_height;
  }
  if (rec_line < 1) {
    rec_line = 1;
  }

  XCALLOC(curs_items, nitems + 1);
  for (i = 0; i < nitems; i++) {
    curs_items[i] = new_item(item_strs[i], 0);
  }
  curs_menu = new_menu(curs_items);
  set_menu_back(curs_menu, COLOR_PAIR(1));
  set_menu_fore(curs_menu, COLOR_PAIR(3));

  out_win = newwin(act_height + 2, act_width + 2, rec_line, rec_col);
  txt_win = newwin(1, act_width, rec_line + 1, rec_col + 1);
  in_win = newwin(act_height - 1, item_width + 1,
                  rec_line + 2, rec_col + 1 + (act_width - item_width-1) / 2);
  wattrset(out_win, COLOR_PAIR(1));
  wbkgdset(out_win, COLOR_PAIR(1));
  wattrset(in_win, COLOR_PAIR(1));
  wbkgdset(in_win, COLOR_PAIR(1));
  wattrset(txt_win, COLOR_PAIR(1));
  wbkgdset(txt_win, COLOR_PAIR(1));
  wclear(in_win);
  wclear(out_win);
  wclear(txt_win);
  box(out_win, 0, 0);
  out_pan = new_panel(out_win);
  txt_pan = new_panel(txt_win);
  in_pan = new_panel(in_win);
  set_menu_win(curs_menu, in_win);
  set_menu_sub(curs_menu, in_win);
  mvwaddstr(txt_win, 0, (act_width - head_width) / 2, buf);

  if (first_item >= nitems) first_item = nitems - 1;
  if (first_item < 0) first_item = 0;
  set_current_item(curs_menu, curs_items[first_item]);

  post_menu(curs_menu);
  if (!help_str) help_str = "Enter-select ^G-cancel";
  print_help(help_str);
  update_panels();
  doupdate();

  while (1) {
    c = ncurses_getkey(utf8_mode, 0);
    cmd = -1;
    switch (c) {
    case '\r': case '\n': case ' ':
      /* OK */
      cmd = -2;
      break;
    case 'G' & 31: case '\033':
      /* CANCEL */
      cmd = -3;
      break;
    case KEY_UP:
    case KEY_LEFT:
      cmd = REQ_UP_ITEM;
      break;
    case KEY_DOWN:
    case KEY_RIGHT:
      cmd = REQ_DOWN_ITEM;
      break;
    default:
      if (hotkeys && c <= 255) {
        for (i = 0; i < nitems; i++) {
          if (!hotkeys[i]) continue;
          pc = hotkeys[i];
          while (*pc && *pc != c) pc++;
          if (*pc == c) {
            set_current_item(curs_menu, curs_items[i]);
            cmd = -2;
            break;
          }
        }
      }
    }

    if (cmd < -1) break;
    if (cmd > -1) {
      menu_driver(curs_menu, cmd);
      update_panels();
      doupdate();
    }
  }

  unpost_menu(curs_menu);
  switch (cmd) {
  case -2:
    answer = item_index(current_item(curs_menu));
    if (answer < 0 || answer > nitems) answer = 0;
    break;
  case -3:
    answer = -1;
    break;
  }

  del_panel(in_pan);
  del_panel(out_pan);
  del_panel(txt_pan);
  delwin(out_win);
  delwin(txt_win);
  delwin(in_win);
  free_menu(curs_menu);
  for (i = 0; i < nitems; i++) {
    free_item(curs_items[i]);
    xfree(item_strs[i]);
  }
  xfree(item_strs);
  xfree(curs_items);
  update_panels();
  doupdate();
  return answer;
}

static int
okcancel(unsigned char const *fmt, ...)
{
  va_list args;
  unsigned char buf[1024];
  int buflen;
  WINDOW *in_win, *out_win, *txt_win;
  MENU *menu;
  ITEM *items[3];
  PANEL *in_pan, *out_pan, *txt_pan;
  int req_lines, req_cols, line0, col0;
  int answer = 0;               /* cancel */
  int c, cmd;

  va_start(args, fmt);
  buflen = vsnprintf(buf, sizeof(buf), fmt, args);
  va_end(args);

  /* calculate size */
  if (buflen > COLS - 10) {
    req_lines = buflen / (COLS - 10) + 1;
    req_cols = COLS - 10;
  } else {
    req_lines = 1;
    req_cols = buflen;
  }
  if (req_cols < 10) req_cols = 10;
  line0 = (LINES - req_lines - 4) / 2;
  col0 = (COLS - req_cols - 2) / 2;

  items[0] = new_item("Cancel", 0);
  items[1] = new_item("Ok", 0);
  items[2] = 0;
  menu = new_menu(items);
  set_menu_back(menu, COLOR_PAIR(1));
  set_menu_fore(menu, COLOR_PAIR(3));

  out_win = newwin(req_lines + 4, req_cols + 2, line0, col0);
  txt_win = newwin(req_lines, req_cols, line0 + 1, col0 + 1);
  in_win = newwin(2, 8, line0 + req_lines + 1, col0 + 1 + (req_cols - 8) / 2);
  wattrset(out_win, COLOR_PAIR(1));
  wbkgdset(out_win, COLOR_PAIR(1));
  wattrset(in_win, COLOR_PAIR(1));
  wbkgdset(in_win, COLOR_PAIR(1));
  wattrset(txt_win, COLOR_PAIR(1));
  wbkgdset(txt_win, COLOR_PAIR(1));
  wclear(in_win);
  wclear(out_win);
  wclear(txt_win);
  box(out_win, 0, 0);
  out_pan = new_panel(out_win);
  txt_pan = new_panel(txt_win);
  in_pan = new_panel(in_win);
  set_menu_win(menu, in_win);
  set_menu_sub(menu, in_win);
  waddstr(txt_win, buf);

  post_menu(menu);
  print_help("Enter-select Y-Ok N-Cancel Q-Cancel");
  update_panels();
  doupdate();

  while (1) {
    c = ncurses_getkey(utf8_mode, 0);
    switch (c) {
    case 'q': case 'G' & 31: case '\033':
      c = 'q';
      goto menu_done;
    case 'y':
      c = 'y';
      goto menu_done;
    case 'n':
      c = 'n';
      goto menu_done;
    case '\n': case '\r': case ' ':
      c = '\n';
      goto menu_done;
    }
    cmd = -1;
    switch (c) {
    case KEY_UP:
    case KEY_LEFT:
      cmd = REQ_UP_ITEM;
      break;
    case KEY_DOWN:
    case KEY_RIGHT:
      cmd = REQ_DOWN_ITEM;
      break;
    }
    if (cmd != -1) {
      menu_driver(menu, cmd);
      update_panels();
      doupdate();
    }
  }
 menu_done:
  unpost_menu(menu);
  switch (c) {
  case '\n':
    answer = item_index(current_item(menu));
    if (answer < 0 || answer > 1) answer = 0;
    break;
  case 'y':
    answer = 1;
    break;
  }

  del_panel(in_pan);
  del_panel(out_pan);
  del_panel(txt_pan);
  free_menu(menu);
  delwin(out_win);
  delwin(txt_win);
  delwin(in_win);
  free_item(items[0]);
  free_item(items[1]);
  update_panels();
  doupdate();
  return answer;
}

static int
yesno(int init_val, unsigned char const *fmt, ...)
{
  va_list args;
  unsigned char buf[1024];
  int buflen;
  WINDOW *in_win, *out_win, *txt_win;
  MENU *menu;
  ITEM *items[3];
  PANEL *in_pan, *out_pan, *txt_pan;
  int req_lines, req_cols, line0, col0;
  int answer = -1;               /* cancel */
  int c, cmd;

  va_start(args, fmt);
  buflen = vsnprintf(buf, sizeof(buf), fmt, args);
  va_end(args);

  /* calculate size */
  if (buflen > COLS - 10) {
    req_lines = buflen / (COLS - 10) + 1;
    req_cols = COLS - 10;
  } else {
    req_lines = 1;
    req_cols = buflen;
  }
  if (req_cols < 10) req_cols = 10;
  line0 = (LINES - req_lines - 4) / 2;
  col0 = (COLS - req_cols - 2) / 2;

  items[0] = new_item("No", 0);
  items[1] = new_item("Yes", 0);
  items[2] = 0;
  menu = new_menu(items);
  set_menu_back(menu, COLOR_PAIR(1));
  set_menu_fore(menu, COLOR_PAIR(3));

  if (init_val < 0) init_val = 0;
  if (init_val > 1) init_val = 1;
  set_current_item(menu, items[init_val]);

  out_win = newwin(req_lines + 4, req_cols + 2, line0, col0);
  txt_win = newwin(req_lines, req_cols, line0 + 1, col0 + 1);
  in_win = newwin(2, 8, line0 + req_lines + 1, col0 + 1 + (req_cols - 8) / 2);
  wattrset(out_win, COLOR_PAIR(1));
  wbkgdset(out_win, COLOR_PAIR(1));
  wattrset(in_win, COLOR_PAIR(1));
  wbkgdset(in_win, COLOR_PAIR(1));
  wattrset(txt_win, COLOR_PAIR(1));
  wbkgdset(txt_win, COLOR_PAIR(1));
  wclear(in_win);
  wclear(out_win);
  wclear(txt_win);
  box(out_win, 0, 0);
  out_pan = new_panel(out_win);
  txt_pan = new_panel(txt_win);
  in_pan = new_panel(in_win);
  set_menu_win(menu, in_win);
  set_menu_sub(menu, in_win);
  waddstr(txt_win, buf);

  post_menu(menu);
  print_help("Enter-select Y-Yes N-No Q-Quit");
  update_panels();
  doupdate();

  while (1) {
    c = ncurses_getkey(utf8_mode, 0);
    switch (c) {
    case 'q': case 'G' & 31: case '\033':
      c = 'q';
      goto menu_done;
    case 'y':
      c = 'y';
      goto menu_done;
    case 'n':
      c = 'n';
      goto menu_done;
    case '\n': case '\r': case ' ':
      c = '\n';
      goto menu_done;
    }
    cmd = -1;
    switch (c) {
    case KEY_UP:
    case KEY_LEFT:
      cmd = REQ_UP_ITEM;
      break;
    case KEY_DOWN:
    case KEY_RIGHT:
      cmd = REQ_DOWN_ITEM;
      break;
    }
    if (cmd != -1) {
      menu_driver(menu, cmd);
      update_panels();
      doupdate();
    }
  }
 menu_done:
  unpost_menu(menu);
  switch (c) {
  case '\n':
    answer = item_index(current_item(menu));
    if (answer < 0 || answer > 1) answer = 0;
    break;
  case 'y':
    answer = 1;
    break;
  case 'n':
    answer = 0;
    break;
  }

  del_panel(in_pan);
  del_panel(out_pan);
  del_panel(txt_pan);
  free_menu(menu);
  delwin(out_win);
  delwin(txt_win);
  delwin(in_win);
  free_item(items[0]);
  free_item(items[1]);
  update_panels();
  doupdate();
  return answer;
}

static int
display_reg_status_menu(int line, int init_val)
{
  int i;
  ITEM *items[USERLIST_REG_LAST + 1];
  int req_lines, req_cols, line0, col0;
  MENU *menu;
  WINDOW *out_win, *in_win;
  PANEL *out_pan, *in_pan;
  int selected_value = -1;
  int c, cmd;

  XMEMZERO(items, USERLIST_REG_LAST + 1);
  for (i = 0; i < USERLIST_REG_LAST; i++) {
    items[i] = new_item(userlist_unparse_reg_status(i), 0);
  }
  menu = new_menu(items);
  scale_menu(menu, &req_lines, &req_cols);
  set_menu_back(menu, COLOR_PAIR(1));
  set_menu_fore(menu, COLOR_PAIR(3));
  line0 = line - req_lines/2 - 1;
  if (line0 + req_lines + 2 >= LINES)
    line0 = LINES - 1 - req_lines - 2;
  if (line0 < 1) line0 = 1;
  col0 = COLS - 1 - req_cols - 2;
  if (col0 < 0) col0 = 0;
  out_win = newwin(req_lines + 2, req_cols + 2, line0, col0);
  in_win = newwin(req_lines, req_cols, line0 + 1, col0 + 1);
  wattrset(out_win, COLOR_PAIR(1));
  wbkgdset(out_win, COLOR_PAIR(1));
  wattrset(in_win, COLOR_PAIR(1));
  wbkgdset(in_win, COLOR_PAIR(1));
  wclear(in_win);
  wclear(out_win);
  box(out_win, 0, 0);
  out_pan = new_panel(out_win);
  in_pan = new_panel(in_win);
  set_menu_win(menu, in_win);
  set_menu_sub(menu, in_win);

  if (init_val < 0) init_val = 0;
  if (init_val >= USERLIST_REG_LAST) init_val = USERLIST_REG_LAST - 1;
  set_current_item(menu, items[init_val]);

  /*
    show_panel(out_pan);
    show_panel(in_pan);
  */
  post_menu(menu);
  print_help("Enter-select Q-quit");
  update_panels();
  doupdate();

  while (1) {
    c = ncurses_getkey(utf8_mode, 0);
    switch (c) {
    case 'q': case 'G' & 31: case '\033':
      c = 'q';
      goto menu_done;
    case '\n': case '\r': case ' ':
      c = '\n';
      goto menu_done;
    }
    cmd = -1;
    switch (c) {
    case KEY_UP:
    case KEY_LEFT:
      cmd = REQ_UP_ITEM;
      break;
    case KEY_DOWN:
    case KEY_RIGHT:
      cmd = REQ_DOWN_ITEM;
      break;
    }
    if (cmd != -1) {
      menu_driver(menu, cmd);
      update_panels();
      doupdate();
    }
  }
 menu_done:
  unpost_menu(menu);
  /*
    hide_panel(out_pan);
    hide_panel(in_pan);
    update_panels();
    doupdate();
  */
  if (c == '\n') {
    selected_value = item_index(current_item(menu));
  }

  del_panel(in_pan);
  del_panel(out_pan);
  free_menu(menu);
  delwin(out_win);
  delwin(in_win);
  update_panels();
  doupdate();
  for (i = 0; i < USERLIST_REG_LAST; i++) {
    free_item(items[i]);
  }
  return selected_value;
}

static int
display_role_menu(int line, int init_val)
{
  int i;
  ITEM *items[CONTEST_LAST_MEMBER + 1];
  int req_lines, req_cols, line0, col0;
  MENU *menu;
  WINDOW *out_win, *in_win;
  PANEL *out_pan, *in_pan;
  int selected_value = -1;
  int c, cmd;

  XMEMZERO(items, CONTEST_LAST_MEMBER + 1);
  for (i = 0; i < CONTEST_LAST_MEMBER; i++) {
    items[i] = new_item(member_string[i], 0);
  }
  menu = new_menu(items);
  scale_menu(menu, &req_lines, &req_cols);
  set_menu_back(menu, COLOR_PAIR(1));
  set_menu_fore(menu, COLOR_PAIR(3));
  line0 = line - req_lines/2 - 1;
  if (line0 + req_lines + 2 >= LINES)
    line0 = LINES - 1 - req_lines - 2;
  if (line0 < 1) line0 = 1;
  col0 = COLS - 1 - req_cols - 2;
  if (col0 < 0) col0 = 0;
  out_win = newwin(req_lines + 2, req_cols + 2, line0, col0);
  in_win = newwin(req_lines, req_cols, line0 + 1, col0 + 1);
  wattrset(out_win, COLOR_PAIR(1));
  wbkgdset(out_win, COLOR_PAIR(1));
  wattrset(in_win, COLOR_PAIR(1));
  wbkgdset(in_win, COLOR_PAIR(1));
  wclear(in_win);
  wclear(out_win);
  box(out_win, 0, 0);
  out_pan = new_panel(out_win);
  in_pan = new_panel(in_win);
  set_menu_win(menu, in_win);
  set_menu_sub(menu, in_win);

  if (init_val < 0) init_val = 0;
  if (init_val >= USERLIST_REG_LAST) init_val = CONTEST_LAST_MEMBER - 1;
  set_current_item(menu, items[init_val]);

  /*
    show_panel(out_pan);
    show_panel(in_pan);
  */
  post_menu(menu);
  print_help("Enter-select Q-quit");
  update_panels();
  doupdate();

  while (1) {
    c = ncurses_getkey(utf8_mode, 0);
    switch (c) {
    case 'q': case 'G' & 31: case '\033':
      c = 'q';
      goto menu_done;
    case '\n': case '\r': case ' ':
      c = '\n';
      goto menu_done;
    }
    cmd = -1;
    switch (c) {
    case KEY_UP:
    case KEY_LEFT:
      cmd = REQ_UP_ITEM;
      break;
    case KEY_DOWN:
    case KEY_RIGHT:
      cmd = REQ_DOWN_ITEM;
      break;
    }
    if (cmd != -1) {
      menu_driver(menu, cmd);
      update_panels();
      doupdate();
    }
  }
 menu_done:
  unpost_menu(menu);
  /*
    hide_panel(out_pan);
    hide_panel(in_pan);
    update_panels();
    doupdate();
  */
  if (c == '\n') {
    selected_value = item_index(current_item(menu));
  }

  del_panel(in_pan);
  del_panel(out_pan);
  free_menu(menu);
  delwin(out_win);
  delwin(in_win);
  update_panels();
  doupdate();
  for (i = 0; i < CONTEST_LAST_MEMBER; i++) {
    free_item(items[i]);
  }
  return selected_value;
}

static int
display_member_status_menu(int line, int init_val)
{
  int i;
  ITEM *items[USERLIST_ST_LAST + 1];
  int req_lines, req_cols, line0, col0;
  MENU *menu;
  WINDOW *out_win, *in_win;
  PANEL *out_pan, *in_pan;
  int selected_value = -1;
  int c, cmd;

  XMEMZERO(items, USERLIST_ST_LAST + 1);
  for (i = 0; i < USERLIST_ST_LAST; i++) {
    items[i] = new_item(member_status_string[i], 0);
  }
  menu = new_menu(items);
  scale_menu(menu, &req_lines, &req_cols);
  set_menu_back(menu, COLOR_PAIR(1));
  set_menu_fore(menu, COLOR_PAIR(3));
  line0 = line - req_lines/2 - 1;
  if (line0 + req_lines + 2 >= LINES)
    line0 = LINES - 1 - req_lines - 2;
  if (line0 < 1) line0 = 1;
  col0 = COLS / 2;
  if (col0 + req_cols + 2 >= COLS)
    col0 =COLS - 1 - req_cols - 2;
  if (col0 < 0) col0 = 0;
  out_win = newwin(req_lines + 2, req_cols + 2, line0, col0);
  in_win = newwin(req_lines, req_cols, line0 + 1, col0 + 1);
  wattrset(out_win, COLOR_PAIR(1));
  wbkgdset(out_win, COLOR_PAIR(1));
  wattrset(in_win, COLOR_PAIR(1));
  wbkgdset(in_win, COLOR_PAIR(1));
  wclear(in_win);
  wclear(out_win);
  box(out_win, 0, 0);
  out_pan = new_panel(out_win);
  in_pan = new_panel(in_win);
  set_menu_win(menu, in_win);
  set_menu_sub(menu, in_win);

  if (init_val < 0) init_val = 0;
  if (init_val >= USERLIST_ST_LAST) init_val = USERLIST_ST_LAST - 1;
  set_current_item(menu, items[init_val]);
  /*
    show_panel(out_pan);
    show_panel(in_pan);
  */
  post_menu(menu);
  print_help("Enter-select Q-quit");
  update_panels();
  doupdate();

  while (1) {
    c = ncurses_getkey(utf8_mode, 0);
    switch (c) {
    case 'q': case 'G' & 031: case '\033':
      c = 'q';
      goto menu_done;
    case '\n': case '\r': case ' ':
      c = '\n';
      goto menu_done;
    }
    cmd = -1;
    switch (c) {
    case KEY_UP:
    case KEY_LEFT:
      cmd = REQ_UP_ITEM;
      break;
    case KEY_DOWN:
    case KEY_RIGHT:
      cmd = REQ_DOWN_ITEM;
      break;
    }
    if (cmd != -1) {
      menu_driver(menu, cmd);
      update_panels();
      doupdate();
    }
  }
 menu_done:
  unpost_menu(menu);
  /*
    hide_panel(out_pan);
    hide_panel(in_pan);
    update_panels();
    doupdate();
  */
  if (c == '\n') {
    i = item_index(current_item(menu));
    if (i >= 0 && i < USERLIST_ST_LAST) selected_value = i;
  }

  del_panel(in_pan);
  del_panel(out_pan);
  free_menu(menu);
  delwin(out_win);
  delwin(in_win);
  update_panels();
  doupdate();
  for (i = 0; i < USERLIST_ST_LAST; i++) {
    free_item(items[i]);
  }
  return selected_value;
}

static int
display_member_gender_menu(int line, int init_val)
{
  int i;
  ITEM *items[USERLIST_SX_LAST + 1];
  int req_lines, req_cols, line0, col0;
  MENU *menu;
  WINDOW *out_win, *in_win;
  PANEL *out_pan, *in_pan;
  int selected_value = -1;
  int c, cmd;

  XMEMZERO(items, USERLIST_SX_LAST + 1);
  for (i = 0; i < USERLIST_SX_LAST; i++) {
    items[i] = new_item(member_gender_string[i], 0);
  }
  menu = new_menu(items);
  scale_menu(menu, &req_lines, &req_cols);
  set_menu_back(menu, COLOR_PAIR(1));
  set_menu_fore(menu, COLOR_PAIR(3));
  line0 = line - req_lines/2 - 1;
  if (line0 + req_lines + 2 >= LINES)
    line0 = LINES - 1 - req_lines - 2;
  if (line0 < 1) line0 = 1;
  col0 = COLS / 2;
  if (col0 + req_cols + 2 >= COLS)
    col0 =COLS - 1 - req_cols - 2;
  if (col0 < 0) col0 = 0;
  out_win = newwin(req_lines + 2, req_cols + 2, line0, col0);
  in_win = newwin(req_lines, req_cols, line0 + 1, col0 + 1);
  wattrset(out_win, COLOR_PAIR(1));
  wbkgdset(out_win, COLOR_PAIR(1));
  wattrset(in_win, COLOR_PAIR(1));
  wbkgdset(in_win, COLOR_PAIR(1));
  wclear(in_win);
  wclear(out_win);
  box(out_win, 0, 0);
  out_pan = new_panel(out_win);
  in_pan = new_panel(in_win);
  set_menu_win(menu, in_win);
  set_menu_sub(menu, in_win);

  if (init_val < 0) init_val = 0;
  if (init_val >= USERLIST_SX_LAST) init_val = USERLIST_SX_LAST - 1;
  set_current_item(menu, items[init_val]);
  /*
    show_panel(out_pan);
    show_panel(in_pan);
  */
  post_menu(menu);
  print_help("Enter-select Q-quit");
  update_panels();
  doupdate();

  while (1) {
    c = ncurses_getkey(utf8_mode, 0);
    switch (c) {
    case 'q': case 'G' & 031: case '\033':
      c = 'q';
      goto menu_done;
    case '\n': case '\r': case ' ':
      c = '\n';
      goto menu_done;
    }
    cmd = -1;
    switch (c) {
    case KEY_UP:
    case KEY_LEFT:
      cmd = REQ_UP_ITEM;
      break;
    case KEY_DOWN:
    case KEY_RIGHT:
      cmd = REQ_DOWN_ITEM;
      break;
    }
    if (cmd != -1) {
      menu_driver(menu, cmd);
      update_panels();
      doupdate();
    }
  }
 menu_done:
  unpost_menu(menu);
  /*
    hide_panel(out_pan);
    hide_panel(in_pan);
    update_panels();
    doupdate();
  */
  if (c == '\n') {
    i = item_index(current_item(menu));
    if (i >= 0 && i < USERLIST_SX_LAST) selected_value = i;
  }

  del_panel(in_pan);
  del_panel(out_pan);
  free_menu(menu);
  delwin(out_win);
  delwin(in_win);
  update_panels();
  doupdate();
  for (i = 0; i < USERLIST_SX_LAST; i++) {
    free_item(items[i]);
  }
  return selected_value;
}

static unsigned char const * const participant_sort_keys[] =
{
  "None",
  "Id",
  "Rev. Id",
  "Login",
  "Rev. Login",
  "Name",
  "Rev. Name"
};
static unsigned char const * const sort_hotkeys[] =
{
  "Nn",
  "Ii",
  "Dd",
  "Ll",
  "Gg",
  "Aa",
  "Ee",
};
static int
display_participant_sort_menu(int curval)
{
  if (curval < 0 || curval >= PART_SORT_LAST) curval = 0;
  return generic_menu(10, -1, -1, -1, curval, 7, -1, -1,
                      participant_sort_keys, sort_hotkeys,
                      "Enter-select ^G-cancel N,I,D,L,G,A,E-select option",
                      "Sort by?");
}

static unsigned char const * const search_menu_items[] =
{
  "Repeat last",
  "Forw. regex login",
  "Back. regex login",
  "Forw. regex name",
  "Back. regex name",
  "Forw. regex text",
  "Back. regex text",
};
static unsigned char const * const search_menu_hotkeys[] =
{
  "Rr",
  "Ff",
  "Bb",
  "Nn",
  "Ee",
  "Tt",
  "Xx",
};
static int
display_search_menu(int curval)
{
  if (curval < 0 || curval >= SRCH_LAST) curval = 0;
  return generic_menu(10, -1, -1, -1, curval, 7, -1, -1,
                      search_menu_items, search_menu_hotkeys,
                      "Enter-select ^G-cancel R,F,B,N,E,T,X-select option",
                      "Choose search type");
}

#define FIRST_COOKIE(u) ((struct userlist_cookie*) (u)->cookies->first_down)
#define NEXT_COOKIE(c)  ((struct userlist_cookie*) (c)->b.right)
#define FIRST_CONTEST(u) ((struct userlist_contest*)(u)->contests->first_down)
#define NEXT_CONTEST(c)  ((struct userlist_contest*)(c)->b.right)

struct user_field_desc
{
  unsigned char const *name;
  char has_value;
  char is_editable;
};
static const struct user_field_desc user_descs[] =
{
  [USERLIST_NN_ID] = { "Id", 1, 0 },
  [USERLIST_NN_IS_PRIVILEGED] = { "Privileged?", 1, 1 },
  [USERLIST_NN_IS_INVISIBLE] = { "Invisible?", 1, 1 },
  [USERLIST_NN_IS_BANNED] = { "Banned?", 1, 1 },
  [USERLIST_NN_IS_LOCKED] = { "Locked?", 1, 1 },
  [USERLIST_NN_SHOW_LOGIN] = { "Show login?", 1, 1 },
  [USERLIST_NN_SHOW_EMAIL] = { "Show email?", 1, 1 },
  [USERLIST_NN_READ_ONLY] = { "Read-only?", 1, 1 },
  [USERLIST_NN_NEVER_CLEAN] = { "Never clean?", 1, 1 },
  [USERLIST_NN_SIMPLE_REGISTRATION] = { "Simple registered?", 1, 1 },
  [USERLIST_NN_LOGIN] = { "Login", 1, 1 },
  [USERLIST_NN_EMAIL] = { "E-mail", 1, 1 },
  [USERLIST_NN_PASSWD] = { "Reg password", 1, 1 },
  [USERLIST_NN_REGISTRATION_TIME] = { "Reg time", 1, 1 },
  [USERLIST_NN_LAST_LOGIN_TIME] = { "Login time", 1, 1 },
  [USERLIST_NN_LAST_CHANGE_TIME] = { "Change time", 1, 1 },
  [USERLIST_NN_LAST_PWDCHANGE_TIME] = { "Reg. pwd time", 1, 1 },
  [USERLIST_NC_CNTS_READ_ONLY] = { "One contest Read-only?", 1, 1 },
  [USERLIST_NC_NAME] = { "Name", 1, 1 },
  [USERLIST_NC_TEAM_PASSWD] = { "Team password", 1, 1 },
  [USERLIST_NC_INST] = { "Institution", 1, 1 },
  [USERLIST_NC_INST_EN] = { "Institution (En)", 1, 1 },
  [USERLIST_NC_INSTSHORT] = { "Inst. (short)", 1, 1 },
  [USERLIST_NC_INSTSHORT_EN] = { "Inst. (short) (En)", 1, 1 },
  [USERLIST_NC_INSTNUM] = { "Inst. number", 1, 1 },
  [USERLIST_NC_FAC] = { "Faculty", 1, 1 },
  [USERLIST_NC_FAC_EN] = { "Faculty (En)", 1, 1 },
  [USERLIST_NC_FACSHORT] = { "Fac. (short)", 1, 1 },
  [USERLIST_NC_FACSHORT_EN] = { "Fac. (short) (En)", 1, 1 },
  [USERLIST_NC_HOMEPAGE] = { "Homepage", 1, 1 },
  [USERLIST_NC_CITY] = { "City", 1, 1 },
  [USERLIST_NC_CITY_EN] = { "City (En)", 1, 1 },
  [USERLIST_NC_COUNTRY] = { "Country", 1, 1 },
  [USERLIST_NC_COUNTRY_EN] = { "Country (En)", 1, 1 },
  [USERLIST_NC_REGION] = { "Region", 1, 1 },
  [USERLIST_NC_AREA] = { "Area", 1, 1 },
  [USERLIST_NC_ZIP] = { "Zip code", 1, 1 },
  [USERLIST_NC_STREET] = { "Street addr.", 1, 1 },
  [USERLIST_NC_LOCATION] = { "Location", 1, 1 },
  [USERLIST_NC_SPELLING] = { "Spelling", 1, 1 },
  [USERLIST_NC_PRINTER_NAME] = { "Printer name", 1, 1 },
  [USERLIST_NC_EXAM_ID] = { "Exam ID", 1, 1 },
  [USERLIST_NC_EXAM_CYPHER] = { "Exam Cypher", 1, 1 },
  [USERLIST_NC_LANGUAGES] = { "Prog. languages", 1, 1 },
  [USERLIST_NC_PHONE] = { "Phone", 1, 1 },
  [USERLIST_NC_FIELD0] = { "Field 0", 1, 1 },
  [USERLIST_NC_FIELD1] = { "Field 1", 1, 1 },
  [USERLIST_NC_FIELD2] = { "Field 2", 1, 1 },
  [USERLIST_NC_FIELD3] = { "Field 3", 1, 1 },
  [USERLIST_NC_FIELD4] = { "Field 4", 1, 1 },
  [USERLIST_NC_FIELD5] = { "Field 5", 1, 1 },
  [USERLIST_NC_FIELD6] = { "Field 6", 1, 1 },
  [USERLIST_NC_FIELD7] = { "Field 7", 1, 1 },
  [USERLIST_NC_FIELD8] = { "Field 8", 1, 1 },
  [USERLIST_NC_FIELD9] = { "Field 9", 1, 1 },
  [USERLIST_NC_AVATAR_STORE] = { "Avatar Store", 1, 1 },
  [USERLIST_NC_AVATAR_ID] = { "Avatar ID", 1, 1 },
  [USERLIST_NC_AVATAR_SUFFIX] = { "Avatar Suffix", 1, 1 },
  [USERLIST_NC_CREATE_TIME] = { "User info create time", 1, 1 },
  [USERLIST_NC_LAST_LOGIN_TIME] = { "Contest last login", 1, 1 },
  [USERLIST_NC_LAST_CHANGE_TIME] = { "User info change time", 1, 1 },
  [USERLIST_NC_LAST_PWDCHANGE_TIME] = { "Team passwd change time", 1, 1},
  [USERLIST_PSEUDO_PASSWORDS] = { "*Passwords*", 0, 0 },
  [USERLIST_PSEUDO_GENERAL_INFO] = { "*General info*", 0, 0 },
  [USERLIST_PSEUDO_FLAGS] = { "*Flags*", 0, 0 },
  [USERLIST_PSEUDO_TIMESTAMPS] = { "*Timestamps*", 0, 0 },
};
static const struct user_field_desc member_descs[] =
{
  [USERLIST_NM_SERIAL]     = { "Serial", 1, 1 },
  [USERLIST_NM_FIRSTNAME]  = { "Firstname", 1, 1 },
  [USERLIST_NM_FIRSTNAME_EN] = { "Firstname (En)", 1, 1 },
  [USERLIST_NM_MIDDLENAME] = { "Middlename", 1, 1 },
  [USERLIST_NM_MIDDLENAME_EN] = { "Middlename (En)", 1, 1 },
  [USERLIST_NM_SURNAME]    = { "Surname", 1, 1 },
  [USERLIST_NM_SURNAME_EN] = { "Surname (En)", 1, 1 },
  [USERLIST_NM_STATUS]     = { "Status", 1, 1 },
  [USERLIST_NM_GENDER]     = { "Gender", 1, 1 },
  [USERLIST_NM_GRADE]      = { "Grade", 1, 1 },
  [USERLIST_NM_GROUP]      = { "Group", 1, 1 },
  [USERLIST_NM_GROUP_EN]   = { "Group (En)", 1, 1 },
  [USERLIST_NM_OCCUPATION] = { "Occupation", 1, 1 },
  [USERLIST_NM_OCCUPATION_EN] = { "Occupation (En)", 1, 1 },
  [USERLIST_NM_DISCIPLINE] = { "Discipline", 1, 1 },
  [USERLIST_NM_EMAIL]      = { "E-mail", 1, 1 },
  [USERLIST_NM_HOMEPAGE]   = { "Homepage", 1, 1 },
  [USERLIST_NM_PHONE]      = { "Phone", 1, 1 },
  [USERLIST_NM_INST]       = { "Institution", 1, 1 },
  [USERLIST_NM_INST_EN]    = { "Institution (En)", 1, 1 },
  [USERLIST_NM_INSTSHORT]  = { "Inst. (short)", 1, 1 },
  [USERLIST_NM_INSTSHORT_EN] = { "Inst. (short) (En)", 1, 1 },
  [USERLIST_NM_FAC]        = { "Faculty", 1, 1 },
  [USERLIST_NM_FAC_EN]     = { "Faculty (En)", 1, 1 },
  [USERLIST_NM_FACSHORT]   = { "Fac. (short)", 1, 1 },
  [USERLIST_NM_FACSHORT_EN] = { "Fac. (short) (En)", 1, 1 },
  [USERLIST_NM_BIRTH_DATE] = { "Birth date", 1, 1 },
  [USERLIST_NM_ENTRY_DATE] = { "Inst. entry date", 1, 1 },
  [USERLIST_NM_GRADUATION_DATE] = { "Exp. grad. date", 1, 1 },
  [USERLIST_NM_CREATE_TIME] = { "Create time", 1, 1 },
  [USERLIST_NM_LAST_CHANGE_TIME] = { "Change time", 1, 1 },
};

static int
get_cookie_str(unsigned char *buf, size_t len,
               const struct userlist_cookie *cookie)
{
  return snprintf(buf, len, "%016llx %16s %s %4d %4d",
                  cookie->cookie, xml_unparse_ipv6(&cookie->ip),
                  userlist_unparse_date(cookie->expire, 1),
                  cookie->locale_id,
                  cookie->contest_id);
}

static int
estimate_contest_str(const struct userlist_contest *reg)
{
  const struct contest_desc *d = 0;
  const unsigned char *s = 0;

  if (contests_get(reg->id, &d) >= 0 && d) {
    s = d->name;
  }
  if (!s) s = "???";

  return 4 * COLS + 1 + strlen(s);
}

static unsigned char *
append_padded_string(unsigned char *buf, const unsigned char *str, int width);

static void
get_contest_str(unsigned char *buf, const struct userlist_contest *reg)
{
  const struct contest_desc *d = 0;
  const unsigned char *s = 0;

  if (contests_get(reg->id, &d) >= 0 && d) {
    s = d->name;
  }
  if (!s) s = "???";

  unsigned char *p = buf;
  p += sprintf(p, "%-6d ", reg->id);
  *p++ = (reg->flags & USERLIST_UC_BANNED)?'B':' ';
  *p++ = (reg->flags & USERLIST_UC_INVISIBLE)?'I':' ';
  *p++ = (reg->flags & USERLIST_UC_LOCKED)?'L':' ';
  *p++ = (reg->flags & USERLIST_UC_PRIVILEGED)?'P':((reg->flags & USERLIST_UC_INCOMPLETE)?'N':' ');
  *p++ = (reg->flags & USERLIST_UC_DISQUALIFIED)?'D':' ';
  *p++ = (reg->flags & USERLIST_UC_REG_READONLY)?'R':' ';
  *p++ = ' ';
  p = append_padded_string(p, userlist_unparse_reg_status(reg->status), 7);
  *p++ = ' ';
  append_padded_string(p, s, COLS - 25);
}

struct field_ref
{
  int role;                     /* -1 - main */
  int pers;                     /*  */
  int field;
};

static int
get_user_field(unsigned char *buf, size_t size,
               const struct userlist_user *u, int field, int convert_null)
{
  if (field >= USERLIST_NN_FIRST && field < USERLIST_NN_LAST) {
    return userlist_get_user_field_str(buf, size, u, field, convert_null);
  } else if (field >= USERLIST_NC_FIRST && field < USERLIST_NC_LAST) {
    if (!u->cnts0) return snprintf(buf, size, "%s", "");
    return userlist_get_user_info_field_str(buf,size,u->cnts0,field,convert_null);
  } else {
    abort();
  }
}

static int
set_user_field(struct userlist_user *u, int field, const unsigned char *value)
{
  struct userlist_user_info *ui = 0;
  if (field >= USERLIST_NN_FIRST && field < USERLIST_NN_LAST) {
    return userlist_set_user_field_str(u, field, value);
  } else if (field >= USERLIST_NC_FIRST && field < USERLIST_NC_LAST) {
    ui = userlist_get_cnts0(u);
    return userlist_set_user_info_field_str(ui, field, value);
  } else {
    abort();
  }
}

static unsigned char *
append_padded_string(unsigned char *buf, const unsigned char *str, int width)
{
  if (!utf8_mode) {
    while (*str && width > 0) {
      *buf++ = *str++;
      --width;
    }
    while (width > 0) {
      *buf++ = ' ';
      --width;
    }
    *buf = 0;
    return buf;
  } else {
    return utf8_padded_append(buf, str, width);
  }
}

static int
user_menu_estimate(struct userlist_user *u, int f)
{
  unsigned char buf[4096];

  if (f >= USERLIST_PSEUDO_FIRST && f < USERLIST_PSEUDO_LAST) {
    return COLS - 2 + 1;
  } else if (!user_descs[f].has_value) {
    return COLS - 2 + 1;
  } else {
    get_user_field(buf, sizeof(buf), u, f, 1);
    return 3 + strlen(user_descs[f].name) + COLS * 4 + strlen(buf);
  }
}

static void
user_menu_string(struct userlist_user *u, int f, unsigned char *out)
{
  unsigned char buf[256];

  if (f >= USERLIST_PSEUDO_FIRST && f < USERLIST_PSEUDO_LAST) {
    snprintf(out, COLS - 2, "%s", user_descs[f].name);
  } else if (!user_descs[f].has_value) {
    snprintf(out, COLS - 2, "%s", user_descs[f].name);
  } else {
    get_user_field(buf, sizeof(buf), u, f, 1);
    out = append_padded_string(out, user_descs[f].name, 15);
    *out++ = ':'; *out++ = ' ';
    append_padded_string(out, buf, COLS - 20);
  }
}

static int
member_menu_estimate(const struct userlist_member *m, int f)
{
  unsigned char buf[4096];

  if (!member_descs[f].has_value) {
    return COLS - 2 + 1;
  } else {
    userlist_get_member_field_str(buf, sizeof(buf), m, f, 1, 0);
    return 3 + strlen(member_descs[f].name) + COLS * 4 + strlen(buf);
  }
}

static void
member_menu_string(const struct userlist_member *m, int f, unsigned char *out)
{
  unsigned char buf[256];

  if (!member_descs[f].has_value) {
    snprintf(out, COLS - 2, "%s", member_descs[f].name);
  } else {
    userlist_get_member_field_str(buf, sizeof(buf), m, f, 1, 0);
    out = append_padded_string(out, member_descs[f].name, 15);
    *out++ = ':'; *out++ = ' ';
    append_padded_string(out, buf, COLS - 20);
  }
}

// order of general and contest-specific fields in the list
static int field_order[] =
{
  USERLIST_NN_ID,
  USERLIST_NN_LOGIN,
  USERLIST_NN_EMAIL,
  USERLIST_NC_NAME,
  USERLIST_PSEUDO_PASSWORDS,
  USERLIST_NN_PASSWD,
  USERLIST_NC_TEAM_PASSWD,
  USERLIST_PSEUDO_GENERAL_INFO,
  USERLIST_NC_INST,
  USERLIST_NC_INST_EN,
  USERLIST_NC_INSTSHORT,
  USERLIST_NC_INSTSHORT_EN,
  USERLIST_NC_INSTNUM,
  USERLIST_NC_FAC,
  USERLIST_NC_FAC_EN,
  USERLIST_NC_FACSHORT,
  USERLIST_NC_FACSHORT_EN,
  USERLIST_NC_HOMEPAGE,
  USERLIST_NC_CITY,
  USERLIST_NC_CITY_EN,
  USERLIST_NC_COUNTRY,
  USERLIST_NC_COUNTRY_EN,
  USERLIST_NC_REGION,
  USERLIST_NC_AREA,
  USERLIST_NC_ZIP,
  USERLIST_NC_STREET,
  USERLIST_NC_LOCATION,
  USERLIST_NC_SPELLING,
  USERLIST_NC_PRINTER_NAME,
  USERLIST_NC_EXAM_ID,
  USERLIST_NC_EXAM_CYPHER,
  USERLIST_NC_LANGUAGES,
  USERLIST_NC_PHONE,
  USERLIST_NC_FIELD0,
  USERLIST_NC_FIELD1,
  USERLIST_NC_FIELD2,
  USERLIST_NC_FIELD3,
  USERLIST_NC_FIELD4,
  USERLIST_NC_FIELD5,
  USERLIST_NC_FIELD6,
  USERLIST_NC_FIELD7,
  USERLIST_NC_FIELD8,
  USERLIST_NC_FIELD9,
  USERLIST_NC_AVATAR_STORE,
  USERLIST_NC_AVATAR_ID,
  USERLIST_NC_AVATAR_SUFFIX,
  USERLIST_PSEUDO_FLAGS,
  USERLIST_NN_READ_ONLY,
  USERLIST_NC_CNTS_READ_ONLY,
  USERLIST_NN_SIMPLE_REGISTRATION,
  USERLIST_NN_NEVER_CLEAN,
  USERLIST_NN_IS_PRIVILEGED,
  USERLIST_NN_IS_INVISIBLE,
  USERLIST_NN_IS_BANNED,
  USERLIST_NN_IS_LOCKED,
  USERLIST_NN_SHOW_LOGIN,
  USERLIST_NN_SHOW_EMAIL,
  USERLIST_PSEUDO_TIMESTAMPS,
  USERLIST_NN_REGISTRATION_TIME,
  USERLIST_NN_LAST_LOGIN_TIME,
  USERLIST_NN_LAST_CHANGE_TIME,
  USERLIST_NN_LAST_PWDCHANGE_TIME,
  USERLIST_NC_CREATE_TIME,
  USERLIST_NC_LAST_LOGIN_TIME,
  USERLIST_NC_LAST_CHANGE_TIME,
  USERLIST_NC_LAST_PWDCHANGE_TIME,
};
#define field_order_size (sizeof(field_order)/sizeof(field_order[0]))

static int
is_valid_user_field(int f)
{
  if (f < 0) return 0;
  if (f >= USERLIST_NC_LAST) return 0;
  if (f >= USERLIST_NN_LAST && f < USERLIST_NC_FIRST) return 0;
  return 1;
}

static int
do_display_user(unsigned char const *upper, int user_id, int contest_id,
                int *p_start_item, int *p_needs_reload)
{
  int r, tot_items = 0, first_row;
  unsigned char *xml_text = 0;
  struct userlist_user *u = 0;
  int retcode = -1, i, j, role, pers;
  unsigned char **descs = 0;
  struct field_ref *info;
  void **refs;
  struct userlist_member *m = 0;
  struct userlist_contest *reg;
  struct userlist_cookie *cookie;
  struct userlist_user_info *ui;
  ITEM **items;
  MENU *menu;
  PANEL *in_pan, *out_pan;
  WINDOW *in_win, *out_win;
  unsigned char current_level[512];
  int c, cmd, role_cnt;
  int cur_i, cur_line;
  unsigned char edit_buf[512];
  unsigned char edit_header[512];
  int new_status;
  char const *help_str = "";
  ej_cookie_t cookie_val;

  r = userlist_clnt_get_info(server_conn, ULS_PRIV_GET_USER_INFO,
                             user_id, contest_id, &xml_text);
  if (r < 0) {
    vis_err("Cannot get user information: %s", userlist_strerror(-r));
    return -1;
  }
  if (!(u = userlist_parse_user_str(xml_text))) {
    vis_err("XML parse error");
    return -1;
  }
  ui = u->cnts0;

  snprintf(current_level, COLS + 1, "%s->%s %d, %s %d", upper, "User", u->id,
           "Contest", contest_id);

  // count how much menu items we need
  tot_items = field_order_size;
  if (ui && ui->members) {
    for (role = 0; role < CONTEST_LAST_MEMBER; role++) {
      if ((role_cnt = userlist_members_count(ui->members, role)) <= 0)
        continue;
      tot_items += 1 + (USERLIST_NM_LAST - USERLIST_NM_FIRST + 1) * role_cnt;
    }
  }
  if ((r = userlist_user_count_contests(u)) > 0) {
    tot_items += r + 1;
  }
  if ((r = userlist_user_count_cookies(u)) > 0) {
    tot_items += r + 1;
  }

  XCALLOC(descs, tot_items + 1);
  XCALLOC(refs, tot_items + 1);
  XCALLOC(info, tot_items + 1);

  j = 0;
  for (i = 0; i < field_order_size; i++) {
    info[j].role = -1;
    info[j].pers = 0;
    info[j].field = field_order[i];
    descs[j] = xmalloc(user_menu_estimate(u, field_order[i]));
    user_menu_string(u, field_order[i], descs[j++]);
  }
  for (role = 0; role < CONTEST_LAST_MEMBER; role++) {
    if (!ui || !ui->members) continue;
    if ((role_cnt = userlist_members_count(ui->members, role)) <= 0)
      continue;
    info[j].role = role;
    info[j].pers = -1;
    info[j].field = 0;
    descs[j] = xmalloc(COLS - 2 + 1);
    snprintf(descs[j++], COLS - 2, "*%s*", member_string_pl[role]);

    for (pers = 0; pers < role_cnt; pers++) {
      if (!(m = (struct userlist_member*) userlist_members_get_nth(ui->members, role, pers)))
        continue;

      info[j].role = role;
      info[j].pers = pers;
      info[j].field = -1;
      refs[j] = m;
      descs[j] = xmalloc(COLS - 2 + 1);
      snprintf(descs[j++], COLS - 2, "*%s %d*", member_string[role], pers + 1);

      for (i = USERLIST_NM_FIRST; i < USERLIST_NM_LAST; i++) {
        info[j].role = role;
        info[j].pers = pers;
        info[j].field = i;
        refs[j] = m;
        descs[j] = xmalloc(member_menu_estimate(m, i));
        member_menu_string(m, i, descs[j++]);
      }
    }
  }
  if ((r = userlist_user_count_contests(u)) > 0) {
    info[j].role = -1;
    info[j].pers = 1;
    info[j].field = -1;
    descs[j] = xmalloc(COLS - 2 + 1);
    snprintf(descs[j++], COLS - 2, "*%s*", "Registrations");

    for (reg = FIRST_CONTEST(u), i = 0; reg; reg = NEXT_CONTEST(reg), i++) {
      info[j].role = -1;
      info[j].pers = 1;
      info[j].field = i;
      refs[j] = reg;
      descs[j] = xmalloc(estimate_contest_str(reg));
      get_contest_str(descs[j], reg);
      j++;
    }
  }
  if ((r = userlist_user_count_cookies(u)) > 0) {
    info[j].role = -1;
    info[j].pers = 2;
    info[j].field = -1;
    descs[j] = xmalloc(COLS - 2 + 1);
    snprintf(descs[j++], COLS - 2, "*%s*", "Cookies");

    for (cookie=FIRST_COOKIE(u),i=0;cookie;cookie=NEXT_COOKIE(cookie),i++) {
      info[j].role = -1;
      info[j].pers = 2;
      info[j].field = i;
      refs[j] = cookie;
      descs[j] = xmalloc(COLS - 2 + 1);
      get_cookie_str(descs[j], COLS - 2, cookie);
      j++;
    }
  }
  ASSERT(j == tot_items);

  XCALLOC(items, tot_items + 1);
  for (i = 0; i < tot_items; i++) {
    items[i] = new_item(descs[i], 0);
  }

  menu = new_menu(items);
  set_menu_back(menu, COLOR_PAIR(1));
  set_menu_fore(menu, COLOR_PAIR(3));
  out_win = newwin(LINES - 2, COLS, 1, 0);
  in_win = newwin(LINES - 4, COLS - 2, 2, 1);
  wattrset(out_win, COLOR_PAIR(1));
  wbkgdset(out_win, COLOR_PAIR(1));
  wattrset(in_win, COLOR_PAIR(1));
  wbkgdset(in_win, COLOR_PAIR(1));
  wclear(in_win);
  wclear(out_win);
  box(out_win, 0, 0);
  out_pan = new_panel(out_win);
  in_pan = new_panel(in_win);
  set_menu_win(menu, in_win);
  set_menu_sub(menu, in_win);
  mvwprintw(stdscr, 0, 0, "%s", current_level);
  wclrtoeol(stdscr);
  set_menu_format(menu, LINES - 4, 0);
  if (*p_start_item < 0) *p_start_item = 0;
  if (*p_start_item >= tot_items) *p_start_item = tot_items - 1;
  first_row = *p_start_item - (LINES - 4) / 2;
  if (first_row + LINES - 4 > tot_items) first_row = tot_items - (LINES - 4);
  if (first_row < 0) first_row = 0;
  set_top_row(menu, first_row);
  set_current_item(menu, items[*p_start_item]);

  while (1) {
    show_panel(out_pan);
    show_panel(in_pan);
    post_menu(menu);
    update_panels();
    doupdate();

    while (1) {
      help_str = "";
      i = item_index(current_item(menu));
      if (info[i].role == -1 && info[i].pers == 0) {
        switch (info[i].field) {
        case USERLIST_NN_ID:
        case USERLIST_PSEUDO_FLAGS:
        case USERLIST_PSEUDO_TIMESTAMPS:
        case USERLIST_PSEUDO_PASSWORDS:
        case USERLIST_PSEUDO_GENERAL_INFO:
          help_str = "C-contest A-new member Q-quit";
          break;
        case USERLIST_NN_LOGIN:
        case USERLIST_NN_EMAIL:
          help_str = "Enter-edit C-contest A-new member Q-quit";
          break;
        case USERLIST_NC_NAME:
        case USERLIST_NC_INST:
        case USERLIST_NC_INST_EN:
        case USERLIST_NC_INSTSHORT:
        case USERLIST_NC_INSTSHORT_EN:
        case USERLIST_NC_INSTNUM:
        case USERLIST_NC_FAC:
        case USERLIST_NC_FAC_EN:
        case USERLIST_NC_FACSHORT:
        case USERLIST_NC_FACSHORT_EN:
        case USERLIST_NC_HOMEPAGE:
        case USERLIST_NC_PHONE:
        case USERLIST_NC_CITY:
        case USERLIST_NC_CITY_EN:
        case USERLIST_NC_COUNTRY:
        case USERLIST_NC_COUNTRY_EN:
        case USERLIST_NC_REGION:
        case USERLIST_NC_AREA:
        case USERLIST_NC_ZIP:
        case USERLIST_NC_STREET:
        case USERLIST_NC_LOCATION:
        case USERLIST_NC_SPELLING:
        case USERLIST_NC_PRINTER_NAME:
        case USERLIST_NC_EXAM_ID:
        case USERLIST_NC_EXAM_CYPHER:
        case USERLIST_NC_LANGUAGES:
        case USERLIST_NC_FIELD0:
        case USERLIST_NC_FIELD1:
        case USERLIST_NC_FIELD2:
        case USERLIST_NC_FIELD3:
        case USERLIST_NC_FIELD4:
        case USERLIST_NC_FIELD5:
        case USERLIST_NC_FIELD6:
        case USERLIST_NC_FIELD7:
        case USERLIST_NC_FIELD8:
        case USERLIST_NC_FIELD9:
        case USERLIST_NC_AVATAR_STORE:
        case USERLIST_NC_AVATAR_ID:
        case USERLIST_NC_AVATAR_SUFFIX:
          help_str = "Enter-edit D-clear C-contest A-new member Q-quit";
          break;
        case USERLIST_NN_PASSWD:
        case USERLIST_NC_TEAM_PASSWD:
          help_str = "Enter-edit D-clear O-random Y-copy C-contest A-new member Q-quit";
          break;
        case USERLIST_NN_IS_PRIVILEGED:
        case USERLIST_NN_IS_INVISIBLE:
        case USERLIST_NN_IS_BANNED:
        case USERLIST_NN_IS_LOCKED:
        case USERLIST_NN_SHOW_LOGIN:
        case USERLIST_NN_SHOW_EMAIL:
        case USERLIST_NN_READ_ONLY:
        case USERLIST_NC_CNTS_READ_ONLY:
        case USERLIST_NN_NEVER_CLEAN:
        case USERLIST_NN_SIMPLE_REGISTRATION:
          help_str = "Enter-toggle D-reset C-contest A-new member Q-quit";
          break;
        case USERLIST_NN_REGISTRATION_TIME:
        case USERLIST_NN_LAST_LOGIN_TIME:
        case USERLIST_NN_LAST_CHANGE_TIME:
        case USERLIST_NN_LAST_PWDCHANGE_TIME:
        case USERLIST_NC_CREATE_TIME:
        case USERLIST_NC_LAST_LOGIN_TIME:
        case USERLIST_NC_LAST_CHANGE_TIME:
        case USERLIST_NC_LAST_PWDCHANGE_TIME:
          help_str = "D-clear C-contest A-new member Q-quit";
          break;
        default:
          help_str = "Q-quit";
          break;
        }
      }
      if (info[i].role == -1 && info[i].pers == 1) {
        if (info[i].field == -1) {
          help_str = "C-contest A-new member Q-quit";
        } else {
          help_str = "R-register B-(un)ban I-(in)visible C-contest A-new member Q-quit";
        }
      }
      if (info[i].role == -1 && info[i].pers == 2) {
        if (info[i].field == -1) {
          help_str = "D-delete all C-contest A-new member Q-quit";
        } else {
          help_str = "D-delete C-contest A-new member Q-quit";
        }
      }
      if (info[i].role >= 0 && info[i].pers == -1) {
        help_str = "C-contest A-new member Q-quit";
      }
      if (info[i].role >= 0 && info[i].pers >= 0 && info[i].field == -1) {
        help_str = "D-delete C-contest A-new member Q-quit";
      }
      if (info[i].role >= 0 && info[i].pers >= 0 && info[i].field == 0) {
        help_str = "C-contest A-new member Q-quit";
      }
      if (info[i].role >= 0 && info[i].pers >= 0 && info[i].field > 0) {
        help_str = "Enter-edit D-clear C-contest A-new member Q-quit";
      }
      print_help(help_str);
      update_panels();
      doupdate();

      c = ncurses_getkey(utf8_mode, 0);
      // in the following may be duplicates
      if (c == KEY_BACKSPACE || c == KEY_DC || c == 127 || c == 8) {
        c = 'd';
        break;
      }
      switch (c) {
      case '\n': case '\r': case ' ':
        c = '\n';
        goto menu_done;
      case 'q': case 'G' & 31: case '\033':
        c = 'q';
        goto menu_done;
      case 'd': case 'r': case 'b': case 'i': case 'l':
      case 'a': case 'c': case 'o': case 'y': case 'n':
      case 'p': case 'h':
      case 'u':
        goto menu_done;
      }
      cmd = -1;
      switch (c) {
      case KEY_UP:
      case KEY_LEFT:
        cmd = REQ_UP_ITEM;
        break;
      case KEY_DOWN:
      case KEY_RIGHT:
        cmd = REQ_DOWN_ITEM;
        break;
      case KEY_HOME:
        cmd = REQ_FIRST_ITEM;
        break;
      case KEY_END:
        cmd = REQ_LAST_ITEM;
        break;
      case KEY_NPAGE:
        i = item_index(current_item(menu));
        if (i + LINES - 4 >= tot_items) cmd = REQ_LAST_ITEM;
        else cmd = REQ_SCR_DPAGE;
        break;
      case KEY_PPAGE:
        i = item_index(current_item(menu));
        if (i - (LINES - 4) < 0) cmd = REQ_FIRST_ITEM;
        else cmd = REQ_SCR_UPAGE;
        break;
      }
      if (cmd != -1) {
        menu_driver(menu, cmd);
        update_panels();
        doupdate();
      }
    }
  menu_done:
    if (c == 'r' || c == 'b' || c == 'i' || c == 'l') {
      cur_i = item_index(current_item(menu));
      cur_line = i - top_row(menu) + 2;
      if (info[cur_i].role != -1) goto menu_continue;
      if (info[cur_i].pers != 1) goto menu_continue;
      if (info[cur_i].field < 0) goto menu_continue;
      reg = (struct userlist_contest*) refs[cur_i];
      switch (c) {
      case 'r':
        cur_line = i - top_row(menu) + 2;
        new_status = display_reg_status_menu(cur_line, reg->status);
        if (new_status < 0 || new_status >= USERLIST_REG_LAST)
          goto menu_continue;
        if (new_status == reg->status) goto menu_continue;
        r = userlist_clnt_change_registration(server_conn, u->id,
                                              reg->id, new_status, 0, 0);
        if (r >= 0) reg->status = new_status;
        break;
      case 'b':
        if (okcancel("Toggle BANNED status?") != 1) goto menu_continue;
        r = userlist_clnt_change_registration(server_conn, u->id,
                                              reg->id, -1, 3,
                                              USERLIST_UC_BANNED);
        if (r >= 0) reg->flags ^= USERLIST_UC_BANNED;
        break;
      case 'i':
        if (okcancel("Toggle INVISIBLE status?") != 1) goto menu_continue;
        r = userlist_clnt_change_registration(server_conn, u->id,
                                              reg->id, -1, 3,
                                              USERLIST_UC_INVISIBLE);
        if (r >= 0) reg->flags ^= USERLIST_UC_INVISIBLE;
        break;
      case 'l':
        if (okcancel("Toggle LOCKED status?") != 1) goto menu_continue;
        r = userlist_clnt_change_registration(server_conn, u->id,
                                              reg->id, -1, 3,
                                              USERLIST_UC_LOCKED);
        if (r >= 0) reg->flags ^= USERLIST_UC_LOCKED;
        break;
      case 'n':
        if (okcancel("Toggle INCOMPLETE status?") != 1) goto menu_continue;
        r = userlist_clnt_change_registration(server_conn, u->id,
                                              reg->id, -1, 3,
                                              USERLIST_UC_INCOMPLETE);
        if (r >= 0) reg->flags ^= USERLIST_UC_INCOMPLETE;
        break;
      case 'u':
        if (okcancel("Toggle DISQUALIFIED status?") != 1) goto menu_continue;
        r = userlist_clnt_change_registration(server_conn, u->id,
                                              reg->id, -1, 3,
                                              USERLIST_UC_DISQUALIFIED);
        if (r >= 0) reg->flags ^= USERLIST_UC_DISQUALIFIED;
        break;
      case 'p':
        if (okcancel("Toggle PRIVILEGED status?") != 1) goto menu_continue;
        r = userlist_clnt_change_registration(server_conn, u->id,
                                              reg->id, -1, 3,
                                              USERLIST_UC_PRIVILEGED);
        if (r >= 0) reg->flags ^= USERLIST_UC_PRIVILEGED;
        break;
      case 'h':
        if (okcancel("Toggle REG-READONLY status?") != 1) goto menu_continue;
        r = userlist_clnt_change_registration(server_conn, u->id,
                                              reg->id, -1, 3,
                                              USERLIST_UC_REG_READONLY);
        if (r >= 0) reg->flags ^= USERLIST_UC_REG_READONLY;
        break;
      }
      if (r < 0) {
        vis_err("Operation failed: %s", userlist_strerror(-r));
        goto menu_continue;
      }
      get_contest_str(descs[cur_i], reg);
    }

    /* delete field */
    if (c == 'd') {
      cur_i = item_index(current_item(menu));
      if (cur_i < 0 || cur_i >= tot_items) continue;
      cur_line = i - top_row(menu) + 2;
      r = 1;

      if (info[cur_i].role < -1 || info[cur_i].role > CONTEST_LAST_MEMBER)
        goto menu_continue;
      if (info[cur_i].role == -1) {
        if (info[cur_i].pers < 0) goto menu_continue;
        if (info[cur_i].pers == 0) {
          if (!is_valid_user_field(info[cur_i].field)) goto menu_continue;
          if (info[cur_i].field == USERLIST_NN_ID) goto menu_continue;
          if (info[cur_i].field == USERLIST_NN_LOGIN) goto menu_continue;
          if (info[cur_i].field == USERLIST_NN_EMAIL) goto menu_continue;
          r = okcancel("Clear field %s?",
                       user_descs[info[cur_i].field].name);
          if (r != 1) goto menu_continue;
          r = userlist_clnt_delete_field(server_conn, ULS_DELETE_FIELD,
                                         u->id, contest_id, 0,
                                         info[cur_i].field);
          if (r < 0) {
            vis_err("Delete failed: %s", userlist_strerror(-r));
            goto menu_continue;
          }

          if (info[cur_i].field == USERLIST_NC_NAME) {
            if (p_needs_reload) *p_needs_reload = 1;
          }

          *p_start_item = cur_i;
          retcode = 0;
          c = 'q';
          goto menu_continue;
        } else if (info[cur_i].pers == 1) {
          // registration
          if (info[cur_i].field < 0) goto menu_continue;
          reg = (struct userlist_contest*) refs[cur_i];
          r = okcancel("Delete registration for contest %d?", reg->id);
          if (r != 1) goto menu_continue;
          r = userlist_clnt_change_registration(server_conn, u->id,
                                                reg->id, -2, 0, 0);
          if (r < 0) {
            vis_err("Delete failed: %s", userlist_strerror(-r));
            goto menu_continue;
          }
          *p_start_item = 0;
          retcode = 0;
          c = 'q';
          goto menu_continue;
        } else if (info[cur_i].pers == 2) {
          // cookies
          if (info[cur_i].field == -1) {
            r = okcancel("Delete all cookies?");
            cookie_val = 0;
          } else {
            cookie = (struct userlist_cookie*) refs[cur_i];
            r = okcancel("Delete cookie %016llx?", cookie->cookie);
            cookie_val = cookie->cookie;
          }
          if (r != 1) goto menu_continue;
          r = userlist_clnt_delete_cookie(server_conn, u->id, contest_id,
                                          cookie_val,
                                          0 /* FIXME: client_key */);
          if (r < 0) {
            vis_err("Delete failed: %s", userlist_strerror(-r));
            goto menu_continue;
          }
          *p_start_item = 0;
          retcode = 0;
          c = 'q';
          goto menu_continue;
        }
      }
      if (info[cur_i].role >= 0) {
        if (!ui) goto menu_continue;
        if (!(m = (struct userlist_member*) userlist_members_get_nth(ui->members, info[cur_i].role,
                                           info[cur_i].pers)))
          goto menu_continue;
        if (info[cur_i].field < -1) goto menu_continue;
        if (info[cur_i].field > USERLIST_NM_LAST) goto menu_continue;
        if (info[cur_i].field == -1) {
          // delete the whole member
          r = okcancel("DELETE MEMBER %s_%d?",
                       member_string[info[cur_i].role],
                       info[cur_i].pers + 1);
          if (r != 1) goto menu_continue;
          r = userlist_clnt_delete_info(server_conn, ULS_PRIV_DELETE_MEMBER,
                                        u->id, contest_id, m->serial);
          if (r < 0) {
            vis_err("Delete failed: %s", userlist_strerror(-r));
            goto menu_continue;
          }

          *p_start_item = 0;
          retcode = 0;
          c = 'q';
          goto menu_continue;
        } else {
          r = okcancel("Reset field %s_%d::%s?",
                       member_string[info[cur_i].role],
                       info[cur_i].pers + 1,
                       member_descs[info[cur_i].field].name);
          if (r != 1) goto menu_continue;
          r = userlist_clnt_delete_field(server_conn, ULS_DELETE_FIELD,
                                         u->id, contest_id,
                                         m->serial, info[cur_i].field);
          if (r < 0) {
            vis_err("Delete failed: %s", userlist_strerror(-r));
            goto menu_continue;
          }

          *p_start_item = cur_i;
          retcode = 0;
          c = 'q';
          goto menu_continue;
        }
      }
      // should never get here?
      abort();
    }

    /* add member */
    if (c == 'a') {
      r = display_role_menu(LINES / 2, 0);
      if (r < 0 || r >= CONTEST_LAST_MEMBER) goto menu_continue;

      r = userlist_clnt_create_member(server_conn, u->id, contest_id, r);
      if (r < 0) {
        vis_err("Add failed: %s", userlist_strerror(-r));
        goto menu_continue;
      }

      retcode = 0;
      c = 'q';
      // FIXME: calculate starting position of the new member
      *p_start_item = 0;
      goto menu_continue;
    }

    if (c == 'c') {
      i = display_contests_menu(current_level, 1);
      // oops, we cannot check validity of contest_id
      if (i <= 0) goto menu_continue;
      /*
      if (i <= 0 || i >= contests->id_map_size || !contests->id_map[i])
        goto menu_continue;
      */
      r = okcancel("Register for contest %d?", i);
      if (r != 1) goto menu_continue;
      r = userlist_clnt_register_contest(server_conn,
                                         ULS_PRIV_REGISTER_CONTEST,
                                         u->id, i, 0, 0);
      if (r < 0) {
        vis_err("Registration failed: %s", userlist_strerror(-r));
        goto menu_continue;
      }
      retcode = 0;
      c = 'q';
      // FIXME: calculate the position of the new contest
      *p_start_item = 0;
      goto menu_continue;
    }

    if (c == 'o' || c == 'y') {
      unsigned char *msg_txt = 0;
      int cmd;

      // assign a random password or copy password
      cur_i = item_index(current_item(menu));
      if (cur_i < 0 || cur_i >= tot_items) continue;
      if (info[cur_i].role != -1) goto menu_continue;
      if (info[cur_i].pers != 0) goto menu_continue;
      if (info[cur_i].field != USERLIST_NN_PASSWD
          && info[cur_i].field != USERLIST_NC_TEAM_PASSWD)
        goto menu_continue;
      if (!user_descs[info[cur_i].field].is_editable
          || !user_descs[info[cur_i].field].has_value)
        goto menu_continue;

      switch (info[cur_i].field) {
      case USERLIST_NN_PASSWD:
        if (c == 'o') {
          msg_txt = "Assign a random registration password?";
          cmd = ULS_RANDOM_PASSWD;
        } else {
          msg_txt = "Copy registration password to team password?";
          cmd = ULS_COPY_TO_TEAM;
        }
        break;
      case USERLIST_NC_TEAM_PASSWD:
        if (c == 'o') {
          msg_txt = "Assign a random team password?";
          cmd = ULS_RANDOM_TEAM_PASSWD;
        } else {
          msg_txt = "Copy team password to registration password?";
          cmd = ULS_COPY_TO_REGISTER;
        }
        break;
      default:
        abort();
      }
      r = okcancel(msg_txt);
      if (r != 1) goto menu_continue;
      r = userlist_clnt_register_contest(server_conn, cmd, u->id,
                                         contest_id, 0, 0);
      if (r < 0) {
        vis_err("Server error: %s", userlist_strerror(-r));
        goto menu_continue;
      }
      retcode = 0;
      c = 'q';
      *p_start_item = cur_i;
      goto menu_continue;
    }

    if (c == '\n') {
      cur_i = item_index(current_item(menu));
      if (cur_i < 0 || cur_i >= tot_items) continue;
      cur_line = cur_i - top_row(menu) + 2;

      if (info[cur_i].role < -1) goto menu_continue;
      if (info[cur_i].role == -1) {
        if (info[cur_i].pers != 0) goto menu_continue;
        if (!is_valid_user_field(info[cur_i].field)) goto menu_continue;
        if (!user_descs[info[cur_i].field].is_editable
            || !user_descs[info[cur_i].field].has_value)
          goto menu_continue;

        switch (info[cur_i].field) {
        case USERLIST_NN_IS_PRIVILEGED:
        case USERLIST_NN_IS_INVISIBLE:
        case USERLIST_NN_IS_BANNED:
        case USERLIST_NN_IS_LOCKED:
        case USERLIST_NN_SHOW_LOGIN:
        case USERLIST_NN_SHOW_EMAIL:
        case USERLIST_NN_READ_ONLY:
        case USERLIST_NC_CNTS_READ_ONLY:
        case USERLIST_NN_NEVER_CLEAN:
        case USERLIST_NN_SIMPLE_REGISTRATION:
          edit_buf[0] = 0;
          get_user_field(edit_buf, sizeof(edit_buf), u, info[cur_i].field, 0);
          r = xml_parse_bool(NULL, 0, 0, 0, edit_buf, 0);
          r = yesno(r, "New value for \"%s\"",
                    user_descs[info[cur_i].field].name);
          if (r < 0 || r > 1) goto menu_continue;
          snprintf(edit_buf, sizeof(edit_buf), "%d", r);
          r = set_user_field(u, info[cur_i].field, edit_buf);
          if (!r) goto menu_continue;
          if (r < 0) {
            vis_err("Invalid field value");
            goto menu_continue;
          }
          r = userlist_clnt_edit_field(server_conn, ULS_EDIT_FIELD,
                                       u->id, contest_id, 0,
                                       info[cur_i].field, edit_buf);
          if (r < 0) {
            vis_err("Server error: %s", userlist_strerror(-r));
            goto menu_continue;
          }
          //user_menu_string(u, info[cur_i].field, descs[cur_i]);
          retcode = 0;
          c = 'q';
          *p_start_item = cur_i;
          goto menu_continue;

          /*
        case USERLIST_NN_REGISTRATION_TIME:
        case USERLIST_NN_LAST_LOGIN_TIME:
        case USERLIST_NN_LAST_CHANGE_TIME:
        case USERLIST_NN_LAST_PWDCHANGE_TIME:
        case USERLIST_NC_CREATE_TIME:
        case USERLIST_NC_LAST_CHANGE_TIME:
        case USERLIST_NC_LAST_PWDCHANGE_TIME:
          goto menu_continue;
          */
        }

        get_user_field(edit_buf, sizeof(edit_buf), u, info[cur_i].field, 0);
        snprintf(edit_header, sizeof(edit_header),
                 "%s",
                 user_descs[info[cur_i].field].name);
        r = ncurses_edit_string(cur_line, COLS, edit_header,
                                edit_buf, sizeof(edit_buf) - 1, utf8_mode);
        if (r < 0) goto menu_continue;
        r = set_user_field(u, info[cur_i].field, edit_buf);
        if (!r) goto menu_continue;
        if (r < 0) {
          vis_err("Invalid field value");
          goto menu_continue;
        }
        r = userlist_clnt_edit_field(server_conn, ULS_EDIT_FIELD,
                                     u->id, contest_id, 0,
                                     info[cur_i].field, edit_buf);
        if (r < 0) {
          vis_err("Server error: %s", userlist_strerror(-r));
          goto menu_continue;
        }
        //user_menu_string(u, info[cur_i].field, descs[cur_i]);
        if (info[cur_i].field == USERLIST_NN_LOGIN
            || info[cur_i].field == USERLIST_NC_NAME
            || info[cur_i].field == USERLIST_NN_EMAIL) {
          if (p_needs_reload) *p_needs_reload = 1;
        }
        retcode = 0;
        c = 'q';
        *p_start_item = cur_i;
        goto menu_continue;
      }
      if (info[cur_i].role >= 0) {
        if (info[cur_i].role >= CONTEST_LAST_MEMBER) goto menu_continue;
        if (!ui) goto menu_continue;
        if (info[cur_i].pers < 0 ||
            info[cur_i].pers >= userlist_members_count(ui->members, info[cur_i].role))
          goto menu_continue;
        if (info[cur_i].field < 0
            || info[cur_i].field > USERLIST_NM_LAST)
          goto menu_continue;
        if (!member_descs[info[cur_i].field].is_editable
            || !member_descs[info[cur_i].field].has_value)
          goto menu_continue;

        m = (struct userlist_member*) refs[cur_i];
        if (info[cur_i].field == USERLIST_NM_SERIAL) goto menu_continue;
        if (info[cur_i].field == USERLIST_NM_STATUS) {
          int new_status;

          new_status = display_member_status_menu(cur_line, m->status);
          if (new_status < 0 || new_status >= USERLIST_ST_LAST
              || new_status == m->status)
            goto menu_continue;
          snprintf(edit_buf, sizeof(edit_buf), "%d", new_status);
          r = userlist_set_member_field_str(m, info[cur_i].field, edit_buf);
          if (!r) goto menu_continue;
          if (r < 0) {
            vis_err("Invalid field value");
            goto menu_continue;
          }
          r = userlist_clnt_edit_field(server_conn, ULS_EDIT_FIELD,
                                       u->id, contest_id,
                                       m->serial, info[cur_i].field, edit_buf);
          if (r < 0) {
            vis_err("Server error: %s", userlist_strerror(-r));
            goto menu_continue;
          }
          //member_menu_string(m, info[cur_i].field, descs[cur_i]);
          retcode = 0;
          c = 'q';
          *p_start_item = cur_i;
          goto menu_continue;
        }
        if (info[cur_i].field == USERLIST_NM_GENDER) {
          int new_gender;

          new_gender = display_member_gender_menu(cur_line, m->gender);
          if (new_gender < 0 || new_gender >= USERLIST_SX_LAST
              || new_gender == m->gender)
            goto menu_continue;
          snprintf(edit_buf, sizeof(edit_buf), "%d", new_gender);
          r = userlist_set_member_field_str(m, info[cur_i].field, edit_buf);
          if (!r) goto menu_continue;
          if (r < 0) {
            vis_err("Invalid field value");
            goto menu_continue;
          }
          r = userlist_clnt_edit_field(server_conn, ULS_EDIT_FIELD,
                                       u->id, contest_id,
                                       m->serial, info[cur_i].field, edit_buf);
          if (r < 0) {
            vis_err("Server error: %s", userlist_strerror(-r));
            goto menu_continue;
          }
          //member_menu_string(m, info[cur_i].field, descs[cur_i]);
          retcode = 0;
          c = 'q';
          *p_start_item = cur_i;
          goto menu_continue;
        }
        if (info[cur_i].field >= 0) {
          userlist_get_member_field_str(edit_buf, sizeof(edit_buf),
                                        m, info[cur_i].field, 0, 0);
          snprintf(edit_header, sizeof(edit_header),
                   "%s_%d::%s",
                   member_string[info[cur_i].role],
                   info[cur_i].pers + 1,
                   member_descs[info[cur_i].field].name);
          r = ncurses_edit_string(cur_line, COLS, edit_header,
                                  edit_buf, sizeof(edit_buf) - 1, utf8_mode);
          if (r < 0) goto menu_continue;
          r = userlist_set_member_field_str(m, info[cur_i].field, edit_buf);
          if (!r) goto menu_continue;
          if (r < 0) {
            vis_err("Invalid field value");
            goto menu_continue;
          }
          r = userlist_clnt_edit_field(server_conn, ULS_EDIT_FIELD,
                                       u->id, contest_id,
                                       m->serial, info[cur_i].field, edit_buf);
          if (r < 0) {
            vis_err("Server error: %s", userlist_strerror(-r));
            goto menu_continue;
          }
          //member_menu_string(m, info[cur_i].field, descs[cur_i]);
          retcode = 0;
          c = 'q';
          *p_start_item = cur_i;
          goto menu_continue;
        }
      }
    }
  menu_continue:
    unpost_menu(menu);
    hide_panel(out_pan);
    hide_panel(in_pan);
    update_panels();
    doupdate();

    if (c == 'q') break;
  }

  free_menu(menu);
  del_panel(in_pan);
  del_panel(out_pan);
  delwin(out_win);
  delwin(in_win);
  for (i = 0; i < tot_items; i++) {
    free_item(items[i]);
    xfree(descs[i]);
  }
  xfree(descs);
  xfree(refs);
  xfree(info);
  xfree(items);
  return retcode;
}

static int
display_user(unsigned char const *upper, int user_id, int contest_id)
{
  int start_item = 0, r = 0, needs_reload = 0;
  while (r >= 0) {
    r = do_display_user(upper, user_id, contest_id, &start_item, &needs_reload);
  }
  return needs_reload;
}


static unsigned char const * search_regex_kind_full[] =
{
  [SRCH_REGEX_LOGIN_FORWARD]  = "Enter regexp for forward login search",
  [SRCH_REGEX_LOGIN_BACKWARD] = "Enter regexp for backward login search",
  [SRCH_REGEX_NAME_FORWARD]   = "Enter regexp for forward name search",
  [SRCH_REGEX_NAME_BACKWARD]  = "Enter regexp for backward name search",
  [SRCH_REGEX_TEXT_FORWARD]   = "Enter regexp for forward text search",
  [SRCH_REGEX_TEXT_BACKWARD]  = "Enter regexp for backward text search"
};
static regex_t search_regex_comp;
static unsigned char search_regex_buf[1024];
static int search_regex_ready;

static int
user_regmatch(char const *str)
{
  if (!str) return 0;
  return !regexec(&search_regex_comp, str, 0, 0, 0);
}

static int
user_match(struct userlist_user *u, int kind)
{
  struct userlist_user_info *ui;

  if (!u) return 0;
  ui = u->cnts0;

  switch (kind) {
  case SRCH_REGEX_LOGIN_FORWARD:
  case SRCH_REGEX_LOGIN_BACKWARD:
    return user_regmatch(u->login);

  case SRCH_REGEX_NAME_FORWARD:
  case SRCH_REGEX_NAME_BACKWARD:
    if (!ui) return 0;
    return user_regmatch(ui->name);

  case SRCH_REGEX_TEXT_FORWARD:
  case SRCH_REGEX_TEXT_BACKWARD:
    if (user_regmatch(u->login)) return 1;
    if (user_regmatch(u->email)) return 1;
    if (!ui) return 0;
    if (user_regmatch(ui->name)) return 1;
    if (user_regmatch(ui->inst)) return 1;
    if (user_regmatch(ui->inst_en)) return 1;
    if (user_regmatch(ui->instshort)) return 1;
    if (user_regmatch(ui->instshort_en)) return 1;
    if (user_regmatch(ui->fac)) return 1;
    if (user_regmatch(ui->fac_en)) return 1;
    if (user_regmatch(ui->facshort)) return 1;
    if (user_regmatch(ui->facshort_en)) return 1;
    if (user_regmatch(ui->homepage)) return 1;
    if (user_regmatch(ui->phone)) return 1;
    if (user_regmatch(ui->city)) return 1;
    if (user_regmatch(ui->city_en)) return 1;
    if (user_regmatch(ui->country)) return 1;
    if (user_regmatch(ui->country_en)) return 1;
    if (user_regmatch(ui->region)) return 1;
    if (user_regmatch(ui->area)) return 1;
    if (user_regmatch(ui->location)) return 1;
    if (user_regmatch(ui->spelling)) return 1;
    if (user_regmatch(ui->languages)) return 1;
    {
      int role, memb;
      const struct userlist_member *pm;
      int role_cnt = 0;

      for (role = 0; role < USERLIST_MB_LAST; role++) {
        if ((role_cnt = userlist_members_count(ui->members, role)) <= 0)
          continue;
        for (memb = 0; memb < role_cnt; memb++) {
          if (!(pm = userlist_members_get_nth(ui->members, role, memb)))
            continue;

          if (user_regmatch(pm->firstname)) return 1;
          if (user_regmatch(pm->firstname_en)) return 1;
          if (user_regmatch(pm->middlename)) return 1;
          if (user_regmatch(pm->middlename_en)) return 1;
          if (user_regmatch(pm->surname)) return 1;
          if (user_regmatch(pm->surname_en)) return 1;
          if (user_regmatch(pm->group)) return 1;
          if (user_regmatch(pm->group_en)) return 1;
          if (user_regmatch(pm->email)) return 1;
          if (user_regmatch(pm->homepage)) return 1;
          if (user_regmatch(pm->phone)) return 1;
          if (user_regmatch(pm->occupation)) return 1;
          if (user_regmatch(pm->occupation_en)) return 1;
          if (user_regmatch(pm->discipline)) return 1;
          if (user_regmatch(pm->inst)) return 1;
          if (user_regmatch(pm->inst_en)) return 1;
          if (user_regmatch(pm->instshort)) return 1;
          if (user_regmatch(pm->instshort_en)) return 1;
          if (user_regmatch(pm->fac)) return 1;
          if (user_regmatch(pm->fac_en)) return 1;
          if (user_regmatch(pm->facshort)) return 1;
          if (user_regmatch(pm->facshort_en)) return 1;
        }
      }
    }
    return 0;
  }
  // default action
  return 0;
}

static int
user_search(struct userlist_user **uu, int total_users, int cur_user)
{
  int search_type;
  int j, i;

  search_type = display_search_menu(0);
  if (search_type < 0) return -2;
  if (search_type >= SRCH_REGEX_LOGIN_FORWARD && search_type < SRCH_LAST) {
    if (search_regex_ready) {
      regfree(&search_regex_comp);
      search_regex_ready = 0;
    }
    j = ncurses_edit_string(LINES / 2, COLS,search_regex_kind_full[search_type],
                            search_regex_buf, sizeof(search_regex_buf) - 16,
                            utf8_mode);
    if (j <= 0) return -2;
    j = regcomp(&search_regex_comp, search_regex_buf,
                REG_EXTENDED | REG_NOSUB);
    if (j != 0) {
      unsigned char msgbuf[1024];

      regerror(j, &search_regex_comp, msgbuf, sizeof(msgbuf));
      vis_err("Invalid regexp: %s", msgbuf);
      regfree(&search_regex_comp);
      return -2;
    }
    search_regex_ready = search_type;
  } else if (search_type == SRCH_REPEAT) {
    if (!search_regex_ready) {
      vis_err("No search to repeat");
      return -2;
    }
  }

  i = cur_user;
  switch (search_regex_ready) {
  case SRCH_REGEX_LOGIN_FORWARD:
  case SRCH_REGEX_NAME_FORWARD:
  case SRCH_REGEX_TEXT_FORWARD:
    for (i++; i < total_users; i++) {
      if (user_match(uu[i], search_regex_ready)) break;
    }
    break;

  case SRCH_REGEX_LOGIN_BACKWARD:
  case SRCH_REGEX_NAME_BACKWARD:
  case SRCH_REGEX_TEXT_BACKWARD:
    for (i--; i >= 0; i--) {
      if (user_match(uu[i], search_regex_ready)) break;
    }
    break;

  default:
    vis_err("Invalid regexp search");
    return -2;
  }

  if (i < 0 || i >= total_users) {
    vis_err("No match");
    return -2;
  }
  return i;
}

static int registered_users_sort_flag = 0;
static int
registered_users_sort_func(void const *p1, void const *p2)
{
  struct userlist_user const **x1 = (struct userlist_user const **) p1;
  struct userlist_user const **x2 = (struct userlist_user const **) p2;
  const unsigned char *name1 = 0, *name2 = 0;

  switch (registered_users_sort_flag) {
  case PART_SORT_ID:
    return x1[0]->id - x2[0]->id;
  case PART_SORT_ID_REV:
    return x2[0]->id - x1[0]->id;
  case PART_SORT_LOGIN:
    return strcoll(x1[0]->login, x2[0]->login);
  case PART_SORT_LOGIN_REV:
    return strcoll(x2[0]->login, x1[0]->login);
  case PART_SORT_NAME:
    if (x1[0]->cnts0) name1 = x1[0]->cnts0->name;
    if (!name1) name1 = "";
    if (x2[0]->cnts0) name2 = x2[0]->cnts0->name;
    if (!name2) name2 = "";
    return strcoll(name1, name2);
  case PART_SORT_NAME_REV:
    if (x1[0]->cnts0) name1 = x1[0]->cnts0->name;
    if (!name1) name1 = "";
    if (x2[0]->cnts0) name2 = x2[0]->cnts0->name;
    if (!name2) name2 = "";
    return strcoll(name2, name1);
  case 0:
  default:
    return x1 - x2;
  }
}

static const unsigned char * const field_op_names[] =
{
  "Do nothing",
  "Clear field",
  "Set field",
  "Fix passwords",
  0,
};
static const unsigned char * const field_op_keys[] =
{
  "Qq",
  "Cc",
  "Ss",
  "Ff",
  0,
};
static const unsigned char * const field_names[] =
{
  "Do nothing",
  "Read-only flag",
  "One contest read-only flag",
  "Never clean flag",
  "Location field",
  "Team password field",
  0
};
static const unsigned char * const field_keys[] =
{
  "Qq",
  "Rr",
  "Ee",
  "Cc",
  "Ll",
  "Pp",
  0,
};
static const int field_codes[] =
{
  0,
  USERLIST_NN_READ_ONLY,
  USERLIST_NC_CNTS_READ_ONLY,
  USERLIST_NN_NEVER_CLEAN,
  USERLIST_NC_LOCATION,
  USERLIST_NC_TEAM_PASSWD,
  0,
};

/* information about selected users */
struct selected_users_info
{
  int contest_id;
  int total_selected;
  int used;
  int allocated;
  unsigned char *mask;
  int *ids;
};
static struct selected_users_info g_sel_users;
static struct selected_users_info sel_users;
static struct selected_users_info sel_cnts;
static struct selected_users_info sel_groups;
static struct selected_users_info sel_members;

static void
selected_mask_allocate(struct selected_users_info *info, int used)
{
  if (!info) return;
  xfree(info->mask); info->mask = 0;
  xfree(info->ids); info->ids = 0;
  info->allocated = 0;
  info->used = 0;
  info->total_selected = 0;
  if (used <= 0) return;

  info->allocated = 16;
  while (info->allocated < used) info->allocated *= 2;
  XCALLOC(info->mask, info->allocated);
  XCALLOC(info->ids, info->allocated);
  info->used = used;
}

static void
selected_mask_clear(struct selected_users_info *info)
{
  if (!info) return;
  info->total_selected = 0;
  if (info->allocated <= 0) return;
  memset(info->mask, 0, info->allocated);
}

static int
estimate_reg_user_item(
        int i,
        struct userlist_user **uu)
{
  int len = 4 * COLS + 1;
  if (uu[i] && uu[i]->login) {
    len += strlen(uu[i]->login);
  }
  if (uu[i] && uu[i]->cnts0 && uu[i]->cnts0->name) {
    len += strlen(uu[i]->cnts0->name);
  }
  return len;
}

static void
generate_reg_user_item(
        unsigned char *buf,
        int i,
        struct userlist_user **uu,
        struct userlist_contest **uc,
        unsigned char *mask)
{
  const unsigned char *name = 0;

  if (uu[i]->cnts0) name = uu[i]->cnts0->name;
  if (!name) name = "";

  *buf++ = mask[i]?'!':' ';
  buf += sprintf(buf, "%-6d", uu[i]->id);
  *buf++ = ' ';
  buf = append_padded_string(buf, uu[i]->login, 16);
  *buf++ = ' ';
  buf = append_padded_string(buf, name, COLS - 38);
  *buf++ = ' ';
  *buf++ = (uc[i]->flags & USERLIST_UC_BANNED)?'B':' ';
  *buf++ = (uc[i]->flags & USERLIST_UC_INVISIBLE)?'I':' ';
  *buf++ = (uc[i]->flags & USERLIST_UC_LOCKED)?'L':' ';
  *buf++ = (uc[i]->flags & USERLIST_UC_PRIVILEGED)?'P':((uc[i]->flags & USERLIST_UC_INCOMPLETE)?'N':' ');
  *buf++ = (uc[i]->flags & USERLIST_UC_DISQUALIFIED)?'D':' ';
  *buf++ = (uc[i]->flags & USERLIST_UC_REG_READONLY)?'R':' ';
  *buf++ = ' ';
  buf = append_padded_string(buf, userlist_unparse_reg_status(uc[i]->status), 2);
}

static unsigned char csv_path[1024];
static unsigned char csv_sep[1024];

static int
do_display_registered_users(
        unsigned char const *upper,
        int contest_id,
        int *p_cur_val,
        int only_choose)
{
  unsigned char current_level[512];
  int r, nuser, i, j, k;
  unsigned char *xml_text = 0;
  struct userlist_list *users;
  struct userlist_user **uu = 0;
  struct userlist_contest **uc = 0, *cc;
  unsigned char **descs = 0;
  ITEM **items;
  MENU *menu;
  WINDOW *in_win, *out_win;
  PANEL *in_pan, *out_pan;
  int c, cmd, cur_line, new_status;
  int first_row;
  int retcode = -1, errcode;
  const struct contest_desc *cnts = 0;
  unsigned char edit_buf[512];
  int new_contest_id = contest_id;

  if ((errcode = contests_get(contest_id, &cnts)) < 0) {
    vis_err("%s", contests_strerror(-errcode));
    return -1;
  }
  if (cnts->user_contest_num > 0) new_contest_id = cnts->user_contest_num;

  snprintf(current_level, sizeof(current_level),
           "%s->%s %d", upper, "Registered users for",
           cnts->id);

  r = userlist_clnt_list_all_users(server_conn, ULS_LIST_ALL_USERS,
                                   cnts->id, &xml_text);
  if (r < 0) {
    vis_err("Cannot get the list of users: %s", userlist_strerror(-r));
    return -1;
  }
  users = userlist_parse_str(xml_text);
  xfree(xml_text); xml_text = 0;
  if (!users) {
    vis_err("XML parse error");
    return -1;
  }

  for (i = 1, nuser = 0; i < users->user_map_size; i++) {
    if (users->user_map[i]) nuser++;
  }
  if (!nuser) {
    i = okcancel("No users registered for this contest. Add a new user?");
    if (i != 1) return -1;
    i = display_user_menu(current_level, 0, 1);
    if (i > 0) {
      r = okcancel("Add user %d?", i);
      if (r == 1) {
        r = userlist_clnt_register_contest(server_conn,
                                           ULS_PRIV_REGISTER_CONTEST,
                                           i, cnts->id, 0, 0);
        if (r < 0) {
          vis_err("Registration failed: %s", userlist_strerror(-r));
          return -1;
        } else {
          return 0;
        }
      }
    }
    return -1;
  }

  /* uu - array of user references */
  XCALLOC(uu, nuser);
  for (j = 0, i = 1; i < users->user_map_size; i++) {
    if (users->user_map[i]) uu[j++] = users->user_map[i];
  }

  if (sel_users.contest_id != contest_id || sel_users.used != nuser) {
    selected_mask_allocate(&sel_users, nuser);
  } else {
    for (j = 0; j < nuser && sel_users.ids[j] == uu[j]->id; ++j) {
    }
    if (j < nuser) {
      selected_mask_allocate(&sel_users, nuser);
    }
  }
  sel_users.contest_id = contest_id;
  for (j = 0; j < nuser; ++j) {
    sel_users.ids[j] = uu[j]->id;
  }

  if (registered_users_sort_flag > 0) {
    qsort(uu, nuser, sizeof(uu[0]), registered_users_sort_func);
  }

  XCALLOC(uc, nuser);
  for (i = 0; i < nuser; i++) {
    ASSERT(uu[i]->contests);
    for (cc = (struct userlist_contest*) uu[i]->contests->first_down;
         cc; cc = (struct userlist_contest*) cc->b.right) {
      if (cc->id == cnts->id || cc->id == new_contest_id) break;
    }
    ASSERT(cc);
    uc[i] = cc;
  }
  XCALLOC(descs, nuser);
  XCALLOC(items, nuser + 1);
  for (i = 0; i < nuser; i++) {
    descs[i] = xmalloc(estimate_reg_user_item(i, uu));
    generate_reg_user_item(descs[i], i, uu, uc, sel_users.mask);
    items[i] = new_item(descs[i], 0);
  }

  menu = new_menu(items);
  set_menu_back(menu, COLOR_PAIR(1));
  set_menu_fore(menu, COLOR_PAIR(3));
  out_win = newwin(LINES - 2, COLS, 1, 0);
  in_win = newwin(LINES - 4, COLS - 2, 2, 1);
  wattrset(out_win, COLOR_PAIR(1));
  wbkgdset(out_win, COLOR_PAIR(1));
  wattrset(in_win, COLOR_PAIR(1));
  wbkgdset(in_win, COLOR_PAIR(1));
  wclear(in_win);
  wclear(out_win);
  box(out_win, 0, 0);
  out_pan = new_panel(out_win);
  in_pan = new_panel(in_win);
  set_menu_win(menu, in_win);
  set_menu_sub(menu, in_win);
  set_menu_format(menu, LINES - 4, 0);

  if (*p_cur_val >= nuser) *p_cur_val = nuser - 1;
  if (*p_cur_val < 0) *p_cur_val = 0;
  first_row = *p_cur_val - (LINES - 4) / 2;
  if (first_row + LINES - 4 > nuser) first_row = nuser - (LINES - 4);
  if (first_row < 0) first_row = 0;
  set_top_row(menu, first_row);
  set_current_item(menu, items[*p_cur_val]);

  while (1) {
    mvwprintw(stdscr, 0, 0, "%s", current_level);
    wclrtoeol(stdscr);
    print_help("Add Register Delete (un)Ban (in)vIsible Sort Enter-edit Quit :-select Toggle 0-clear");
    show_panel(out_pan);
    show_panel(in_pan);
    post_menu(menu);
    update_panels();
    doupdate();

    while (1) {
      c = ncurses_getkey(utf8_mode, 0);
      // in the following may be duplicates
      if (c == KEY_BACKSPACE || c == KEY_DC || c == 127 || c == 8) {
        c = 'd';
        break;
      }
      switch (c) {
      case 'q': case 'G' & 31: case '\033':
        c = 'q';
        goto menu_done;
      case '\n': case '\r': case ' ':
        c = '\n';
        goto menu_done;
      case 'r': case 'd': case 'i': case 'b': case 'l': case 'a':
      case 's': case 'j': case 'e': case ';': case 'c': case 't':
      case '0': case 'f': case 'o': case 'n': case 'u': case 'v':
      case 'p': case 'h':
        goto menu_done;
      }
      cmd = -1;
      switch (c) {
      case KEY_UP:
      case KEY_LEFT:
        cmd = REQ_UP_ITEM;
        break;
      case KEY_DOWN:
      case KEY_RIGHT:
        cmd = REQ_DOWN_ITEM;
        break;
      case KEY_HOME:
        cmd = REQ_FIRST_ITEM;
        break;
      case KEY_END:
        cmd = REQ_LAST_ITEM;
        break;
      case KEY_NPAGE:
        i = item_index(current_item(menu));
        if (i + LINES - 4 >= nuser) cmd = REQ_LAST_ITEM;
        else cmd = REQ_SCR_DPAGE;
        break;
      case KEY_PPAGE:
        i = item_index(current_item(menu));
        if (i - (LINES - 4) < 0) cmd = REQ_FIRST_ITEM;
        else cmd = REQ_SCR_UPAGE;
        break;
      }
      if (cmd != -1) {
        menu_driver(menu, cmd);
        update_panels();
        doupdate();
      }
    }
  menu_done:
    if (c == ';') {
      i = item_index(current_item(menu));
      if (sel_users.mask[i]) {
        sel_users.mask[i] = 0;
        sel_users.total_selected--;
      } else {
        sel_users.mask[i] = 1;
        sel_users.total_selected++;
      }
      generate_reg_user_item(descs[i], i, uu, uc, sel_users.mask);
      menu_driver(menu, REQ_DOWN_ITEM);
    } else if (c == '0') {
      // clear all selection
      memset(sel_users.mask, 0, sel_users.allocated);
      sel_users.total_selected = 0;
      retcode = -2;
    } else if (c == 't') {
      // toggle all selection
      sel_users.total_selected = 0;
      for (j = 0; j < nuser; j++)
        sel_users.total_selected += (sel_users.mask[j] ^= 1);
      retcode = -2;
    } else if (c == 'c' && !only_choose) {
      if ((k = display_contests_menu(current_level, 1)) <= 0) continue;

      if (!sel_users.total_selected && !sel_cnts.total_selected) {
        // register the current user to the specified contest
        i = item_index(current_item(menu));
        if (okcancel("Register user %d for contest %d?", uu[i]->id, k) != 1)
          continue;
        r = userlist_clnt_register_contest(server_conn,
                                           ULS_PRIV_REGISTER_CONTEST,
                                           uu[i]->id, k, 0, 0);
        if (r < 0) {
          vis_err("Registration failed: %s", userlist_strerror(-r));
          continue;
        }
      } else if (!sel_cnts.total_selected) {
        // register the selected users to the specified contest
        if (okcancel("Register selected users for contest %d?", k) != 1)
          continue;
        for (j = 0; j < nuser; j++) {
          if (!sel_users.mask[j]) continue;
          r = userlist_clnt_register_contest(server_conn,
                                             ULS_PRIV_REGISTER_CONTEST,
                                             uu[j]->id, k, 0, 0);
          if (r < 0) {
            vis_err("Registration failed: %s", userlist_strerror(-r));
            continue;
          }
          sel_users.mask[j] = 0;
          generate_reg_user_item(descs[j], j, uu, uc, sel_users.mask);
        }
        memset(sel_users.mask, 0, sel_users.allocated);
        sel_users.total_selected = 0;
      } else if (!sel_users.total_selected) {
        // register the current user to the selected contests
        i = item_index(current_item(menu));
        if (okcancel("Register user %d for selected contests?", uu[i]->id) != 1)
          continue;
        for (k = 1; k < sel_cnts.allocated; k++) {
          if (!sel_cnts.mask[k]) continue;
          r = userlist_clnt_register_contest(server_conn,
                                             ULS_PRIV_REGISTER_CONTEST,
                                             uu[i]->id, k, 0, 0);
          if (r < 0) {
            vis_err("Registration for contest %d failed: %s", userlist_strerror(-r), k);
            continue;
          }
        }
      } else {
        // register the selected users to the selected contests
        if (okcancel("Register selected users for selected contests?", k) != 1)
          continue;
        for (j = 0; j < nuser; j++) {
          if (!sel_users.mask[j]) continue;
          for (k = 1; k < sel_cnts.allocated; k++) {
            if (!sel_cnts.mask[k]) continue;
            r = userlist_clnt_register_contest(server_conn,
                                               ULS_PRIV_REGISTER_CONTEST,
                                               uu[j]->id, k, 0, 0);
            if (r < 0) {
              vis_err("Registration of user %d to contest %d failed: %s",
                      j, k, userlist_strerror(-r));
              continue;
            }
          }
          sel_users.mask[j] = 0;
          generate_reg_user_item(descs[j], j, uu, uc, sel_users.mask);
        }
        memset(sel_users.mask, 0, sel_users.allocated);
        sel_users.total_selected = 0;
      }
    } else if (c == 'r' && !only_choose) {
      i = item_index(current_item(menu));
      cur_line = i - top_row(menu) + 2;
      new_status = display_reg_status_menu(cur_line, uc[i]->status);
      if (new_status < 0 || new_status >= USERLIST_REG_LAST) continue;
      if (!sel_users.total_selected) {
        // operation on a signle user
        if (new_status == uc[i]->status) continue;
        r = userlist_clnt_change_registration(server_conn, uu[i]->id,
                                              cnts->id, new_status, 0, 0);
        if (r < 0) {
          vis_err("Status change failed: %s", userlist_strerror(-r));
          continue;
        }
        uc[i]->status = new_status;
        generate_reg_user_item(descs[i], i, uu, uc, sel_users.mask);
      } else {
        // operation on a group of users
        if (okcancel("Set registration status for the selected users to %s?",
                     userlist_unparse_reg_status(new_status)) != 1)
          continue;
        for (j = 0; j < nuser; j++) {
          if (!sel_users.mask[j]) continue;
          if (new_status == uc[j]->status) continue;
          r = userlist_clnt_change_registration(server_conn, uu[j]->id,
                                                cnts->id, new_status, 0, 0);
          if (r < 0) {
            vis_err("Status change failed on %d: %s", uu[j]->id,
                    userlist_strerror(-r));
            continue;
          }
          uc[j]->status = new_status;
          sel_users.mask[j] = 0;
          generate_reg_user_item(descs[j], j, uu, uc, sel_users.mask);
        }
        memset(sel_users.mask, 0, sel_users.allocated);
        sel_users.total_selected = 0;
      }
    } else if (c == 'd' && !only_choose) {
      if (!sel_users.total_selected) {
        // operation on a single user
        i = item_index(current_item(menu));
        if (okcancel("Delete registration for %s?", uu[i]->login) != 1)
          continue;
        r = userlist_clnt_change_registration(server_conn, uu[i]->id,
                                              cnts->id, -2, 0, 0);
        if (r < 0) {
          vis_err("Delete failed: %s", userlist_strerror(-r));
          continue;
        }
      } else {
        // operation on the selected users
        if (okcancel("Delete registration for the selected users?") != 1)
          continue;
        for (j = 0; j < nuser; j++) {
          if (!sel_users.mask[j]) continue;
          r = userlist_clnt_change_registration(server_conn, uu[j]->id,
                                                cnts->id, -2, 0, 0);
          if (r < 0) {
            vis_err("Delete of %d failed: %s", uu[j]->id,
                    userlist_strerror(-r));
            continue;
          }
        }
      }
      memset(sel_users.mask, 0, sel_users.allocated);
      sel_users.total_selected = 0;
      retcode = -2;
    } else if (c == 'b' && !only_choose) {
      if (!sel_users.total_selected) {
        // operation on a single user
        i = item_index(current_item(menu));
        if (okcancel("Toggle BANNED status for %s?", uu[i]->login) != 1)
          continue;
        r = userlist_clnt_change_registration(server_conn, uu[i]->id,
                                              cnts->id, -1, 3,
                                              USERLIST_UC_BANNED);
        if (r < 0) {
          vis_err("Toggle flags failed: %s", userlist_strerror(-r));
          continue;
        }
        uc[i]->flags ^= USERLIST_UC_BANNED;
        generate_reg_user_item(descs[i], i, uu, uc, sel_users.mask);
      } else {
        // operation on the selected users
        if (okcancel("Toggle BANNED status for selected users?") != 1)
          continue;
        for (j = 0; j < nuser; j++) {
          if (!sel_users.mask[j]) continue;
          r = userlist_clnt_change_registration(server_conn, uu[j]->id,
                                                cnts->id, -1, 3,
                                                USERLIST_UC_BANNED);
          if (r < 0) {
            vis_err("Toggle flags failed for %d: %s",
                    uu[j]->id, userlist_strerror(-r));
            continue;
          }
          uc[j]->flags ^= USERLIST_UC_BANNED;
          sel_users.mask[j] = 0;
          generate_reg_user_item(descs[j], j, uu, uc, sel_users.mask);
        }
        memset(sel_users.mask, 0, sel_users.allocated);
        sel_users.total_selected = 0;
      }
    } else if (c == 'i' && !only_choose) {
      if (!sel_users.total_selected) {
        // operation on a single user
        i = item_index(current_item(menu));
        if (okcancel("Toggle INVISIBLE status for %s?", uu[i]->login) != 1)
          continue;
        r = userlist_clnt_change_registration(server_conn, uu[i]->id,
                                              cnts->id, -1, 3,
                                              USERLIST_UC_INVISIBLE);
        if (r < 0) {
          vis_err("Toggle flags failed: %s", userlist_strerror(-r));
          continue;
        }
        uc[i]->flags ^= USERLIST_UC_INVISIBLE;
        generate_reg_user_item(descs[i], i, uu, uc, sel_users.mask);
      } else {
        // operation on the selected users
        if (okcancel("Toggle INVISIBLE status for selected users?") != 1)
          continue;
        for (j = 0; j < nuser; j++) {
          if (!sel_users.mask[j]) continue;
          r = userlist_clnt_change_registration(server_conn, uu[j]->id,
                                                cnts->id, -1, 3,
                                                USERLIST_UC_INVISIBLE);
          if (r < 0) {
            vis_err("Toggle flags failed for %d: %s",
                    uu[j]->id, userlist_strerror(-r));
            continue;
          }
          uc[j]->flags ^= USERLIST_UC_INVISIBLE;
          sel_users.mask[j] = 0;
          generate_reg_user_item(descs[j], j, uu, uc, sel_users.mask);
        }
        memset(sel_users.mask, 0, sel_users.allocated);
        sel_users.total_selected = 0;
      }
    } else if (c == 'l' && !only_choose) {
      if (!sel_users.total_selected) {
        // operation on a single user
        i = item_index(current_item(menu));
        if (okcancel("Toggle LOCKED status for %s?", uu[i]->login) != 1)
          continue;
        r = userlist_clnt_change_registration(server_conn, uu[i]->id,
                                              cnts->id, -1, 3,
                                              USERLIST_UC_LOCKED);
        if (r < 0) {
          vis_err("Toggle flags failed: %s", userlist_strerror(-r));
          continue;
        }
        uc[i]->flags ^= USERLIST_UC_LOCKED;
        generate_reg_user_item(descs[i], i, uu, uc, sel_users.mask);
      } else {
        // operation on the selected users
        if (okcancel("Toggle LOCKED status for selected users?") != 1)
          continue;
        for (j = 0; j < nuser; j++) {
          if (!sel_users.mask[j]) continue;
          r = userlist_clnt_change_registration(server_conn, uu[j]->id,
                                                cnts->id, -1, 3,
                                                USERLIST_UC_LOCKED);
          if (r < 0) {
            vis_err("Toggle flags failed for %d: %s",
                    uu[j]->id, userlist_strerror(-r));
            continue;
          }
          uc[j]->flags ^= USERLIST_UC_LOCKED;
          sel_users.mask[j] = 0;
          generate_reg_user_item(descs[j], j, uu, uc, sel_users.mask);
        }
        memset(sel_users.mask, 0, sel_users.allocated);
        sel_users.total_selected = 0;
      }
    } else if (c == 'n' && !only_choose) {
      if (!sel_users.total_selected) {
        // operation on a single user
        i = item_index(current_item(menu));
        if (okcancel("Toggle INCOMPLETE status for %s?", uu[i]->login) != 1)
          continue;
        r = userlist_clnt_change_registration(server_conn, uu[i]->id,
                                              cnts->id, -1, 3,
                                              USERLIST_UC_INCOMPLETE);
        if (r < 0) {
          vis_err("Toggle flags failed: %s", userlist_strerror(-r));
          continue;
        }
        uc[i]->flags ^= USERLIST_UC_INCOMPLETE;
        generate_reg_user_item(descs[i], i, uu, uc, sel_users.mask);
      } else {
        // operation on the selected users
        if (okcancel("Toggle INCOMPLETE status for selected users?") != 1)
          continue;
        for (j = 0; j < nuser; j++) {
          if (!sel_users.mask[j]) continue;
          r = userlist_clnt_change_registration(server_conn, uu[j]->id,
                                                cnts->id, -1, 3,
                                                USERLIST_UC_INCOMPLETE);
          if (r < 0) {
            vis_err("Toggle flags failed for %d: %s",
                    uu[j]->id, userlist_strerror(-r));
            continue;
          }
          uc[j]->flags ^= USERLIST_UC_INCOMPLETE;
          sel_users.mask[j] = 0;
          generate_reg_user_item(descs[j], j, uu, uc, sel_users.mask);
        }
        memset(sel_users.mask, 0, sel_users.allocated);
        sel_users.total_selected = 0;
      }
    } else if (c == 'u' && !only_choose) {
      if (!sel_users.total_selected) {
        // operation on a single user
        i = item_index(current_item(menu));
        if (okcancel("Toggle DISQUALIFIED status for %s?", uu[i]->login) != 1)
          continue;
        r = userlist_clnt_change_registration(server_conn, uu[i]->id,
                                              cnts->id, -1, 3,
                                              USERLIST_UC_DISQUALIFIED);
        if (r < 0) {
          vis_err("Toggle flags failed: %s", userlist_strerror(-r));
          continue;
        }
        uc[i]->flags ^= USERLIST_UC_DISQUALIFIED;
        generate_reg_user_item(descs[i], i, uu, uc, sel_users.mask);
      } else {
        // operation on the selected users
        if (okcancel("Toggle DISQUALIFIED status for selected users?") != 1)
          continue;
        for (j = 0; j < nuser; j++) {
          if (!sel_users.mask[j]) continue;
          r = userlist_clnt_change_registration(server_conn, uu[j]->id,
                                                cnts->id, -1, 3,
                                                USERLIST_UC_DISQUALIFIED);
          if (r < 0) {
            vis_err("Toggle flags failed for %d: %s",
                    uu[j]->id, userlist_strerror(-r));
            continue;
          }
          uc[j]->flags ^= USERLIST_UC_DISQUALIFIED;
          sel_users.mask[j] = 0;
          generate_reg_user_item(descs[j], j, uu, uc, sel_users.mask);
        }
        memset(sel_users.mask, 0, sel_users.allocated);
        sel_users.total_selected = 0;
      }
    } else if (c == 'p' && !only_choose) {
      if (!sel_users.total_selected) {
        // operation on a single user
        i = item_index(current_item(menu));
        if (okcancel("Toggle PRIVILEGED status for %s?", uu[i]->login) != 1)
          continue;
        r = userlist_clnt_change_registration(server_conn, uu[i]->id,
                                              cnts->id, -1, 3,
                                              USERLIST_UC_PRIVILEGED);
        if (r < 0) {
          vis_err("Toggle flags failed: %s", userlist_strerror(-r));
          continue;
        }
        uc[i]->flags ^= USERLIST_UC_PRIVILEGED;
        generate_reg_user_item(descs[i], i, uu, uc, sel_users.mask);
      } else {
        // operation on the selected users
        if (okcancel("Toggle PRIVILEGED status for selected users?") != 1)
          continue;
        for (j = 0; j < nuser; j++) {
          if (!sel_users.mask[j]) continue;
          r = userlist_clnt_change_registration(server_conn, uu[j]->id,
                                                cnts->id, -1, 3,
                                                USERLIST_UC_PRIVILEGED);
          if (r < 0) {
            vis_err("Toggle flags failed for %d: %s",
                    uu[j]->id, userlist_strerror(-r));
            continue;
          }
          uc[j]->flags ^= USERLIST_UC_PRIVILEGED;
          sel_users.mask[j] = 0;
          generate_reg_user_item(descs[j], j, uu, uc, sel_users.mask);
        }
        memset(sel_users.mask, 0, sel_users.allocated);
        sel_users.total_selected = 0;
      }
    } else if (c == 'h' && !only_choose) {
      if (!sel_users.total_selected) {
        // operation on a single user
        i = item_index(current_item(menu));
        if (okcancel("Toggle REG_READONLY status for %s?", uu[i]->login) != 1)
          continue;
        r = userlist_clnt_change_registration(server_conn, uu[i]->id,
                                              cnts->id, -1, 3,
                                              USERLIST_UC_REG_READONLY);
        if (r < 0) {
          vis_err("Toggle flags failed: %s", userlist_strerror(-r));
          continue;
        }
        uc[i]->flags ^= USERLIST_UC_REG_READONLY;
        generate_reg_user_item(descs[i], i, uu, uc, sel_users.mask);
      } else {
        // operation on the selected users
        if (okcancel("Toggle REG_READONLY status for selected users?") != 1)
          continue;
        for (j = 0; j < nuser; j++) {
          if (!sel_users.mask[j]) continue;
          r = userlist_clnt_change_registration(server_conn, uu[j]->id,
                                                cnts->id, -1, 3,
                                                USERLIST_UC_REG_READONLY);
          if (r < 0) {
            vis_err("Toggle flags failed for %d: %s",
                    uu[j]->id, userlist_strerror(-r));
            continue;
          }
          uc[j]->flags ^= USERLIST_UC_REG_READONLY;
          sel_users.mask[j] = 0;
          generate_reg_user_item(descs[j], j, uu, uc, sel_users.mask);
        }
        memset(sel_users.mask, 0, sel_users.allocated);
        sel_users.total_selected = 0;
      }
    } else if (c == '\n' && only_choose) {
      i = item_index(current_item(menu));
      *p_cur_val = i;
      retcode = uu[i]->id;
      c = 'q';
    } else if (c == '\n') {
      i = item_index(current_item(menu));
      display_user(current_level, uu[i]->id, contest_id);
      *p_cur_val = i;
      retcode = -2;
    } else if (c == 'a' && !only_choose) {
      i = display_user_menu(current_level, 0, 1);
      if (i > 0) {
        r = okcancel("Register user %d?", i);
        if (r == 1) {
          r = userlist_clnt_register_contest(server_conn,
                                             ULS_PRIV_REGISTER_CONTEST,
                                             i, cnts->id, 0, 0);
          if (r < 0) {
            vis_err("Registration failed: %s", userlist_strerror(-r));
          } else {
            memset(sel_users.mask, 0, sel_users.allocated);
            sel_users.total_selected = 0;
            retcode = -2;
          }
        }
      }
    } else if (c == 's') {
      /* change sort criteria */
      i = display_participant_sort_menu(registered_users_sort_flag);
      if (i >= 0 && i != registered_users_sort_flag) {
        retcode = -2;
        registered_users_sort_flag = i;
      }
    } else if (c == 'j') {
      /* find a user by number */
      unsigned char number_buf[256], *endptr;
      char *tmpendptr = 0;

      memset(number_buf, 0, sizeof(number_buf));
      i = ncurses_edit_string(LINES / 2, COLS, "Jump to user id?", number_buf,
                              200, utf8_mode);
      if (i >= 0) {
        errno = 0;
        i = strtol(number_buf, &tmpendptr, 10);
        endptr = tmpendptr;
        if (!errno && !*endptr) {
          if (i <= uu[0]->id) {
            j = 0;
          } else if (i >= uu[nuser - 1]->id) {
            j = nuser - 1;
          } else {
            for (j = 0; j < nuser - 1; j++) {
              if (uu[j]->id <= i && uu[j + 1]->id > i)
                break;
            }
          }
          retcode = -2;
          *p_cur_val = j;
        }
      }
    } else if (c == 'e') {
      i = user_search(uu, nuser, item_index(current_item(menu)));
      if (i >= 0) {
        retcode = -2;
        *p_cur_val = i;
      }
    } else if (c == 'o' && !only_choose) {
      // copy user_info
      if ((k = display_contests_menu(current_level, 1)) <= 0) continue;

      if (!sel_users.total_selected && !sel_cnts.total_selected) {
        i = item_index(current_item(menu));
        if (okcancel("Copy user %d info to contest %d?", uu[i]->id, k) != 1)
          continue;
        r = userlist_clnt_copy_user_info(server_conn, ULS_COPY_ALL, uu[i]->id, cnts->id, k);
        if (r < 0) {
          vis_err("Operation failed: %s", userlist_strerror(-r));
          continue;
        }
      } else if (!sel_cnts.total_selected) {
        // copy the selected users to the specified contest
        if (okcancel("Copy selected users infos to contest %d?", k) != 1)
          continue;
        for (j = 0; j < nuser; j++) {
          if (!sel_users.mask[j]) continue;
          r = userlist_clnt_copy_user_info(server_conn, ULS_COPY_ALL, uu[j]->id, cnts->id, k);
          if (r < 0) {
            vis_err("Operation failed: %s", userlist_strerror(-r));
            continue;
          }
          sel_users.mask[j] = 0;
          generate_reg_user_item(descs[j], j, uu, uc, sel_users.mask);
        }
        memset(sel_users.mask, 0, sel_users.allocated);
        sel_users.total_selected = 0;
      }
      /*
 else if (!sel_users.total_selected) {
        // register the current user to the selected contests
        i = item_index(current_item(menu));
        if (okcancel("Register user %d for selected contests?", uu[i]->id) != 1)
          continue;
        for (k = 1; k < sel_cnts.allocated; k++) {
          if (!sel_cnts.mask[k]) continue;
          r = userlist_clnt_register_contest(server_conn,
                                             ULS_PRIV_REGISTER_CONTEST,
                                             uu[i]->id, k, 0, 0);
          if (r < 0) {
            vis_err("Registration for contest %d failed: %s", userlist_strerror(-r), k);
            continue;
          }
        }
      } else {
        // register the selected users to the selected contests
        if (okcancel("Register selected users for selected contests?", k) != 1)
          continue;
        for (j = 0; j < nuser; j++) {
          if (!sel_users.mask[j]) continue;
          for (k = 1; k < sel_cnts.allocated; k++) {
            if (!sel_cnts.mask[k]) continue;
            r = userlist_clnt_register_contest(server_conn,
                                               ULS_PRIV_REGISTER_CONTEST,
                                               uu[j]->id, k, 0, 0);
            if (r < 0) {
              vis_err("Registration of user %d to contest %d failed: %s",
                      j, k, userlist_strerror(-r));
              continue;
            }
          }
          sel_users.mask[j] = 0;
          generate_reg_user_item(descs[j], j, uu, uc, sel_users.mask);
        }
        memset(sel_users.mask, 0, sel_users.allocated);
        sel_users.total_selected = 0;
      }
      */
    } else if (c == 'f' && !only_choose) {
      int field_op, field_code = 0;
      field_op = generic_menu(10, -1, -1, -1, 0, 4, -1, -1,
                              field_op_names, field_op_keys,
                              "Enter-select ^G-cancel Q,C,S-select option",
                              "Field operation");
      if (field_op <= 0) continue;
      if (field_op != 3) {
        i = generic_menu(10, -1, -1, -1, 0, (field_op == 2)?4:6, -1, -1,
                         field_names, field_keys,
                         "Enter-select ^G-cancel Q,R,C,L-select option",
                         "Field");
        if (i <= 0) continue;
        field_code = field_codes[i];
      }

      if (!sel_users.total_selected) {
        i = item_index(current_item(menu));
        switch (field_op) {
        case 1:                 /* clear field */
          r = userlist_clnt_delete_field(server_conn, ULS_DELETE_FIELD,
                                         uu[i]->id, contest_id,
                                         0, field_code);
          break;
        case 2:                 /* set field */
          snprintf(edit_buf, sizeof(edit_buf), "%d", 1);
          r = userlist_clnt_edit_field(server_conn, ULS_EDIT_FIELD,
                                       uu[i]->id, contest_id,
                                       0, field_code, edit_buf);
          break;
        case 3:
          r = userlist_clnt_register_contest(server_conn, ULS_FIX_PASSWORD,
                                             uu[i]->id, 0, 0, 0);
          break;
        }
        if (r < 0) {
          vis_err("Operation failed: %s", userlist_strerror(-r));
          break;
        }
      } else {
        for (i = 0; i < nuser; i++) {
          if (!sel_users.mask[i]) continue;
          switch (field_op) {
          case 1:               /* clear field */
            r = userlist_clnt_delete_field(server_conn, ULS_DELETE_FIELD,
                                           uu[i]->id, contest_id,
                                           0, field_code);
            break;
          case 2:               /* set field */
            snprintf(edit_buf, sizeof(edit_buf), "%d", 1);
            r = userlist_clnt_edit_field(server_conn, ULS_EDIT_FIELD,
                                         uu[i]->id, contest_id,
                                         0, field_code, edit_buf);
          case 3:
            r = userlist_clnt_register_contest(server_conn, ULS_FIX_PASSWORD,
                                               uu[i]->id, 0, 0, 0);
            break;
          }
          if (r < 0) {
            vis_err("Operation failed for %d: %s", uu[i]->id, userlist_strerror(-r));
            break;
          }
        }
        memset(sel_users.mask, 0, sel_users.allocated);
        sel_users.total_selected = 0;
        retcode = -2;
      }
    } else if (c == 'v' && !only_choose) {
      FILE *csv_f = 0, *csv_in = 0;
      char *csv_txt = 0;
      size_t csv_z = 0;
      int curc;
      unsigned char *csv_reply = 0;

      // import CSV
      r = ncurses_edit_string(LINES / 2, COLS, "CSV file name",
                              csv_path, sizeof(csv_path) - 1, utf8_mode);
      if (r < 0) continue;
      r = ncurses_edit_string(LINES / 2, COLS, "CSV field separator",
                              csv_sep, sizeof(csv_sep) - 1, utf8_mode);
      if (r < 0) continue;
      if (strlen(csv_sep) != 1 || csv_sep[0] < ' ' || csv_sep[0] >= 127) {
        vis_err("Invalid field separator");
        continue;
      }
      if (!(csv_in = fopen(csv_path, "r"))) {
        vis_err("Cannot open file `%s'", csv_path);
        continue;
      }
      csv_f = open_memstream(&csv_txt, &csv_z);
      while ((curc = getc(csv_in)) != EOF)
        putc(curc, csv_f);
      putc(0, csv_f);
      if (!feof(csv_in) && ferror(csv_in)) {
        vis_err("Read error from `%s'", csv_path);
        close_memstream(csv_f); csv_f = 0;
        fclose(csv_in);
        xfree(csv_txt);
        continue;
      }
      fclose(csv_in); csv_in = 0;
      close_memstream(csv_f); csv_f = 0;
      if (strlen(csv_txt) + 1 != csv_z) {
        vis_err("The file `%s' is binary", csv_path);
        xfree(csv_txt);
        continue;
      }
      r = userlist_clnt_import_csv_users(server_conn, ULS_IMPORT_CSV_USERS,
                                         contest_id, csv_sep[0], 0, csv_txt,
                                         &csv_reply);
      if (r < 0) {
        xfree(csv_txt);
        vis_err("Operation failed: %s", userlist_strerror(-r));
        continue;
      }
      ncurses_view_text("Import log", csv_reply);
      xfree(csv_txt);
      xfree(csv_reply);
      retcode = -2;
    }

    unpost_menu(menu);
    hide_panel(out_pan);
    hide_panel(in_pan);
    update_panels();
    doupdate();

    if (c == 'q' || retcode == -2) break;
  }

  // cleanup
  wmove(stdscr, 0, 0);
  wclrtoeol(stdscr);
  wmove(stdscr, LINES - 1, 0);
  wclrtoeol(stdscr);
  del_panel(in_pan);
  del_panel(out_pan);
  free_menu(menu);
  delwin(out_win);
  delwin(in_win);
  update_panels();
  doupdate();
  for (i = 0; i < nuser; i++) {
    free_item(items[i]);
    xfree(descs[i]);
  }
  xfree(uu);
  xfree(uc);
  xfree(descs);
  xfree(items);
  return retcode;
}

static int
display_registered_users(
        unsigned char const *upper,
        int contest_id,
        int init_val,
        int only_choose)
{
  int val = -2;

  while (val == -2) {
    val = do_display_registered_users(upper, contest_id, &init_val, only_choose);
  }
  return val;
}

static int
display_contests_menu(unsigned char *upper, int only_choose)
{
  int ncnts = 0, i, j;
  const struct contest_desc *cc;
  unsigned char **descs;
  ITEM **items;
  MENU *menu = 0;
  WINDOW *in_win = 0, *out_win = 0;
  PANEL *out_pan = 0, *in_pan = 0;
  int c, cmd;
  unsigned char current_level[512];
  int sel_num, r = 0;
  int retval = -1;
  const unsigned char *cnts_set = 0;
  int cnts_set_card;
  int *cntsi;
  unsigned char **cnts_names = 0,*s;
  int cur_item = 0, first_row, slen;

  snprintf(current_level, sizeof(current_level),
           "%s->%s", upper, "Contest list");

  // request the set of the existing contests
  cnts_set_card = contests_get_set(&cnts_set);

  /* update the selected mask array */
  if (cnts_set_card > sel_cnts.allocated) {
    size_t new_a = sel_cnts.allocated;
    unsigned char *new_m;

    if (!new_a) new_a = 64;
    while (cnts_set_card > new_a) new_a *= 2;
    new_m = (unsigned char*) xcalloc(new_a, 1);
    if (sel_cnts.allocated > 0)
      memcpy(new_m, sel_cnts.mask, sel_cnts.allocated);
    xfree(sel_cnts.mask);
    sel_cnts.allocated = new_a;
    sel_cnts.mask = new_m;
  }

  sel_cnts.total_selected = 0;
  if (sel_cnts.allocated > 0)
    memset(sel_cnts.mask, 0, sel_cnts.allocated);

  // count the total contests
  for (i = 1; i < cnts_set_card; i++) {
    if (cnts_set[i]) ncnts++;
  }
  if (!ncnts) return -1;

  XCALLOC(cnts_names, cnts_set_card);
  for (i = 1; i < cnts_set_card; i++) {
    if (contests_get(i, &cc) >= 0) {
      cnts_names[i] = s = xstrdup(cc->name);
      // fix the contest names
      slen = strlen(s);
      while (slen > 0 && isspace(s[slen - 1])) slen--;
      s[slen] = 0;
      for (; *s; s++)
        if (*s < ' ')
          *s = ' ';
    }
  }

  XCALLOC(cntsi, ncnts);
  for (i = 1, j = 0; i < cnts_set_card; i++) {
    if (cnts_set[i]) cntsi[j++] = i;
  }
  ASSERT(j == ncnts);

  XCALLOC(descs, ncnts);
  XCALLOC(items, ncnts + 1);

 restart_menu:

  for (i = 0; i < ncnts; i++) {
    j = cntsi[i];
    const unsigned char *cnts_name = cnts_names[j];
    if (!cnts_name) cnts_name = "(removed)";
    unsigned char *disp_str = xmalloc(strlen(cnts_name) + COLS * 4 + 1);
    unsigned char *s = disp_str;
    *s++ = sel_cnts.mask[j]?'!':' ';
    s += sprintf(s, "%-8d ", j);
    append_padded_string(s, cnts_name, COLS - 13);
    xfree(descs[i]);
    descs[i] = disp_str;
  }

  // free menus from the previous iteration
  if (in_pan) del_panel(in_pan);
  if (out_pan) del_panel(out_pan);
  if (menu) free_menu(menu);
  if (out_win) delwin(out_win);
  if (in_win) delwin(in_win);
  for (i = 0; i < ncnts; i++) {
    if (items[i]) free_item(items[i]);
    items[i] = 0;
  }

  for (i = 0; i < ncnts; i++) {
    items[i] = new_item(descs[i], 0);
  }
  menu = new_menu(items);
  set_menu_back(menu, COLOR_PAIR(1));
  set_menu_fore(menu, COLOR_PAIR(3));
  out_win = newwin(LINES - 2, COLS, 1, 0);
  in_win = newwin(LINES - 4, COLS - 2, 2, 1);
  wattrset(out_win, COLOR_PAIR(1));
  wbkgdset(out_win, COLOR_PAIR(1));
  wattrset(in_win, COLOR_PAIR(1));
  wbkgdset(in_win, COLOR_PAIR(1));
  wclear(in_win);
  wclear(out_win);
  box(out_win, 0, 0);
  out_pan = new_panel(out_win);
  in_pan = new_panel(in_win);
  set_menu_win(menu, in_win);
  set_menu_sub(menu, in_win);
  set_menu_format(menu, LINES - 4, 0);

  if (cur_item < 0) cur_item = 0;
  if (cur_item >= ncnts) cur_item = ncnts - 1;
  first_row = cur_item - (LINES - 4) / 2;
  if (first_row + LINES - 4 > ncnts) first_row = ncnts - (LINES - 4);
  if (first_row < 0) first_row = 0;
  set_top_row(menu, first_row);
  set_current_item(menu, items[cur_item]);

  while (1) {
    mvwprintw(stdscr, 0, 0, "%s", current_level);
    wclrtoeol(stdscr);
    print_help("Enter-view Q-quit :-select T-toggle 0-clear");
    show_panel(out_pan);
    show_panel(in_pan);
    post_menu(menu);
    update_panels();
    doupdate();

    while (1) {
      c = ncurses_getkey(utf8_mode, 0);
      switch (c) {
      case 'q': case 'G' & 31: case '\033':
        c = 'q';
        goto menu_done;
      case '\n': case '\r': case ' ':
        c = '\n';
        goto menu_done;
      case ';': case 't': case '0':
        goto menu_done;
      }
      cmd = -1;
      switch (c) {
      case KEY_UP:
      case KEY_LEFT:
        cmd = REQ_UP_ITEM;
        break;
      case KEY_DOWN:
      case KEY_RIGHT:
        cmd = REQ_DOWN_ITEM;
        break;
      case KEY_HOME:
        cmd = REQ_FIRST_ITEM;
        break;
      case KEY_END:
        cmd = REQ_LAST_ITEM;
        break;
      case KEY_NPAGE:
        i = item_index(current_item(menu));
        if (i + LINES - 4 >= ncnts) cmd = REQ_LAST_ITEM;
        else cmd = REQ_SCR_DPAGE;
        break;
      case KEY_PPAGE:
        i = item_index(current_item(menu));
        if (i - (LINES - 4) < 0) cmd = REQ_FIRST_ITEM;
        else cmd = REQ_SCR_UPAGE;
        break;
      }
      if (cmd != -1) {
        menu_driver(menu, cmd);
        update_panels();
        doupdate();
      }
    }
  menu_done:
    unpost_menu(menu);
    hide_panel(out_pan);
    hide_panel(in_pan);
    update_panels();
    doupdate();

    if (c == ';') {
      cur_item = i = item_index(current_item(menu));
      if (i >= 0 && i < ncnts) {
        j = cntsi[i];
        if (cnts_names[j]) {
          if (sel_cnts.mask[j]) {
            sel_cnts.mask[j] = 0;
            sel_cnts.total_selected--;
          } else {
            sel_cnts.mask[j] = 1;
            sel_cnts.total_selected++;
          }
        }
      }
      if (cur_item < ncnts - 1) cur_item++;
      goto restart_menu;
    }
    if (c == 't') {
      cur_item = item_index(current_item(menu));
      sel_cnts.total_selected = 0;
      for (i = 1; i < cnts_set_card; i++)
        sel_cnts.total_selected += (sel_cnts.mask[i] ^= 1);
      goto restart_menu;
    }
    if (c == '0') {
      cur_item = item_index(current_item(menu));
      memset(sel_cnts.mask, 0, sel_cnts.allocated);
      sel_cnts.total_selected = 0;
      goto restart_menu;
    }

    if (c == 'q') break;
    if (c == '\n' && only_choose) {
      sel_num = item_index(current_item(menu));
      retval = cntsi[sel_num];
      break;
    }
    if (c == '\n') {
      sel_num = item_index(current_item(menu));
      r = display_registered_users(current_level, cntsi[sel_num], r, 0);
    }
  }

  // cleanup
  wmove(stdscr, 0, 0);
  wclrtoeol(stdscr);
  del_panel(in_pan);
  del_panel(out_pan);
  free_menu(menu);
  delwin(out_win);
  delwin(in_win);
  for (i = 0; i < ncnts; i++) {
    free_item(items[i]);
  }
  for (i = 0; i < ncnts; i++)
    xfree(descs[i]);
  for (i = 1; i < cnts_set_card; i++)
    xfree(cnts_names[i]);
  xfree(cnts_names);
  xfree(cntsi);
  xfree(descs);
  xfree(items);
  return retval;
}

static unsigned char *
make_user_menu_item(
        unsigned char *prev_item,
        const struct userlist_user *uu,
        int sel_flag)
{
  const unsigned char *name = NULL;
  if (uu->cnts0) name = uu->cnts0->name;
  if (!name) name = "";

  unsigned char *buf = xmalloc(strlen(name) + COLS * 4 + 1);
  unsigned char *s = buf;

  *s++ = sel_flag?'!':' ';
  s += sprintf(s, "%-6d ", uu->id);
  s = append_padded_string(s, uu->login, 16);
  *s++ = ' ';
  append_padded_string(s, name, COLS - 28);
  free(prev_item);
  return buf;
}

static int
do_display_user_menu(unsigned char *upper, int *p_start_item, int only_choose)
{
  int r;
  unsigned char *xml_text = 0;
  struct userlist_list *users = 0;
  int nusers, i, j, k;
  struct userlist_user **uu = 0;
  unsigned char **descs = 0;
  ITEM **items;
  MENU *menu;
  WINDOW *in_win, *out_win;
  PANEL *out_pan, *in_pan;
  int c, cmd;
  unsigned char current_level[512];
  int retval = -1;
  int first_row;
  int loc_start_item;

  snprintf(current_level, sizeof(current_level),
           "%s->%s", upper, "User list");

  r = userlist_clnt_list_all_users(server_conn, ULS_LIST_ALL_USERS,
                                   0, &xml_text);
  if (r < 0) {
    vis_err("Cannot get user list: %s", userlist_strerror(-r));
    return -1;
  }
  users = userlist_parse_str(xml_text);
  if (!users) {
    vis_err("XML parse error");
    xfree(xml_text);
    return -1;
  }
  xfree(xml_text); xml_text = 0;

  // count all users
  nusers = 0;
  for (i = 1; i < users->user_map_size; i++) {
    if (!users->user_map[i]) continue;
    nusers++;
  }
  if (!nusers) {
    j = okcancel("No users in database. Add new user?");
    if (j != 1) return -1;
    j = userlist_clnt_create_user(server_conn, ULS_CREATE_USER, 0, 0);
    if (j < 0) {
      vis_err("Add failed: %s", userlist_strerror(-j));
      return -1;
    }

    *p_start_item = 0;
    return -2;
  }

  XCALLOC(uu, nusers);
  for (i = 1, j = 0; i < users->user_map_size; i++) {
    if (!users->user_map[i]) continue;
    uu[j++] = users->user_map[i];
  }
  ASSERT(j == nusers);

  // extend selection
  if (nusers != g_sel_users.used) {
    selected_mask_allocate(&g_sel_users, nusers);
  } else {
    for (j = 0; j < nusers && g_sel_users.ids[j] == uu[j]->id; ++j) {
    }
    if (j < nusers) {
      selected_mask_allocate(&g_sel_users, nusers);
    }
  }
  for (j = 0; j < nusers; ++j) {
    g_sel_users.ids[j] = uu[j]->id;
  }

  if (registered_users_sort_flag > 0) {
    qsort(uu, nusers, sizeof(uu[0]), registered_users_sort_func);
  }

  XCALLOC(descs, nusers);
  for (i = 0; i < nusers; i++) {
    descs[i] = make_user_menu_item(NULL, uu[i], g_sel_users.mask[i]);
  }

  XCALLOC(items, nusers + 1);
  for (i = 0; i < nusers; i++) {
    items[i] = new_item(descs[i], 0);
  }
  menu = new_menu(items);
  set_menu_back(menu, COLOR_PAIR(1));
  set_menu_fore(menu, COLOR_PAIR(3));
  out_win = newwin(LINES - 2, COLS, 1, 0);
  in_win = newwin(LINES - 4, COLS - 2, 2, 1);
  wattrset(out_win, COLOR_PAIR(1));
  wbkgdset(out_win, COLOR_PAIR(1));
  wattrset(in_win, COLOR_PAIR(1));
  wbkgdset(in_win, COLOR_PAIR(1));
  wclear(in_win);
  wclear(out_win);
  box(out_win, 0, 0);
  out_pan = new_panel(out_win);
  in_pan = new_panel(in_win);
  set_menu_win(menu, in_win);
  set_menu_sub(menu, in_win);
  set_menu_format(menu, LINES - 4, 0);

  for (i = 0; i < nusers; i++)
    if (uu[i]->id == *p_start_item) break;
  if (i < nusers) loc_start_item = i;
  else loc_start_item = 0;

  if (loc_start_item < 0) loc_start_item = 0;
  if (loc_start_item >= nusers) loc_start_item = nusers - 1;
  first_row = loc_start_item - (LINES - 4)/2;
  if (first_row + LINES - 4 > nusers) first_row = nusers - (LINES - 4);
  if (first_row < 0) first_row = 0;
  set_top_row(menu, first_row);
  set_current_item(menu, items[loc_start_item]);

  while (1) {
    mvwprintw(stdscr, 0, 0, "%s", current_level);
    wclrtoeol(stdscr);
    print_help("Enter-view Add Delete Quit Sort Jump sEarch Mass Contest :-Sel Toggle 0-clear");
    show_panel(out_pan);
    show_panel(in_pan);
    post_menu(menu);
    update_panels();
    doupdate();

    while (1) {
      c = ncurses_getkey(utf8_mode, 0);
      if (c == KEY_BACKSPACE || c == KEY_DC || c == 127 || c == 8 || c == 'd') {
        c = 'd';
        goto menu_done;
      }
      switch (c) {
      case 'q': case 'G' & 31: case '\033':
        c = 'q';
        goto menu_done;
      case '\n': case '\r':
        c = '\n';
        goto menu_done;
      case 'a': case 's': case 'j': case 'e': case 'm':
      case ';': case 'c': case 't': case '0':
        goto menu_done;
      }
      cmd = -1;
      switch (c) {
      case KEY_UP:
      case KEY_LEFT:
        cmd = REQ_UP_ITEM;
        break;
      case KEY_DOWN:
      case KEY_RIGHT:
        cmd = REQ_DOWN_ITEM;
        break;
      case KEY_HOME:
        cmd = REQ_FIRST_ITEM;
        break;
      case KEY_END:
        cmd = REQ_LAST_ITEM;
        break;
      case KEY_NPAGE:
        i = item_index(current_item(menu));
        if (i + LINES - 4 >= nusers) cmd = REQ_LAST_ITEM;
        else cmd = REQ_SCR_DPAGE;
        break;
      case KEY_PPAGE:
        i = item_index(current_item(menu));
        if (i - (LINES - 4) < 0) cmd = REQ_FIRST_ITEM;
        else cmd = REQ_SCR_UPAGE;
        break;
      }
      if (cmd != -1) {
        menu_driver(menu, cmd);
        update_panels();
        doupdate();
      }
    }
  menu_done:
    if (c == 'd' && !only_choose) {
      i = item_index(current_item(menu));
      j = okcancel("REMOVE USER %d (%s)?", uu[i]->id, uu[i]->login);
      if (j != 1) goto menu_continue;
      j = userlist_clnt_delete_info(server_conn, ULS_DELETE_USER, uu[i]->id, 0, 0);
      if (j < 0) {
        vis_err("Remove failed: %s", userlist_strerror(-j));
        goto menu_continue;
      }

      // set to the first position and redraw the screen
      *p_start_item = 0;
      retval = -2;
      c = 'q';
    }
    if (c == 'a' && !only_choose) {
      j = okcancel("Add new user?");
      if (j != 1) goto menu_continue;
      j = userlist_clnt_create_user(server_conn, ULS_CREATE_USER, 0, 0);
      if (j < 0) {
        vis_err("Add failed: %s", userlist_strerror(-j));
        goto menu_continue;
      }

      // FIXME: the id of new user is not known :-(
      // set to the first position and redraw the screen
      *p_start_item = 0;
      retval = -2;
      c = 'q';
    }
    if (c == 's') {
      /* change sort criteria */
      i = display_participant_sort_menu(registered_users_sort_flag);
      if (i >= 0 && i != registered_users_sort_flag) {
        registered_users_sort_flag = i;
        *p_start_item = 0;
        retval = -2;
        c = 'q';
      }
    }
    if (c == 'j') {
      /* find a user by number */
      unsigned char number_buf[256], *endptr;
      char *tmpendptr = 0;

      memset(number_buf, 0, sizeof(number_buf));
      i = ncurses_edit_string(LINES / 2, COLS, "Jump to user id?", number_buf,
                              200, utf8_mode);
      if (i >= 0) {
        errno = 0;
        i = strtol(number_buf, &tmpendptr, 10);
        endptr = tmpendptr;
        if (!errno && !*endptr) {
          if (i <= uu[0]->id) {
            j = 0;
          } else if (i >= uu[nusers - 1]->id) {
            j = nusers - 1;
          } else {
            for (j = 0; j < nusers - 1; j++) {
              if (uu[j]->id <= i && uu[j + 1]->id > i)
                break;
            }
          }
          *p_start_item = uu[j]->id;
          retval = -2;
          c = 'q';
        }
      }
    }
    if (c == 'e') {
      i = user_search(uu, nusers, item_index(current_item(menu)));
      if (i >= 0) {
        *p_start_item = uu[i]->id;
        retval = -2;
        c = 'q';
      }
    }
    if (c == 'm' ) {
      unsigned char templ_buf[256];
      unsigned char passwd_buf[256];
      unsigned char num_buf[256];
      unsigned char valbuf[1024];
      int first_num, last_num, n, contest_num, user_id;
      // mass creating new user

      memset(templ_buf, 0, sizeof(templ_buf));
      i = ncurses_edit_string(LINES / 2, COLS, "Template for new logins?",
                              templ_buf, 200, utf8_mode);
      if (i < 0) goto menu_continue;
      memset(passwd_buf, 0, sizeof(passwd_buf));
      i = ncurses_edit_string(LINES / 2, COLS, "Template for passwords?",
                              passwd_buf, 200, utf8_mode);
      if (i < 0) goto menu_continue;
      memset(num_buf, 0, sizeof(num_buf));
      i = ncurses_edit_string(LINES / 2, COLS, "First number:", num_buf, 200,
                              utf8_mode);
      if (i < 0) goto menu_continue;
      if (sscanf(num_buf, "%d%n", &first_num, &n) != 1 || num_buf[n]
          || first_num < 0 || first_num >= 1000000) {
        vis_err("Invalid number");
        goto menu_continue;
      }
      memset(num_buf, 0, sizeof(num_buf));
      i = ncurses_edit_string(LINES / 2, COLS, "Last number:", num_buf, 200,
                              utf8_mode);
      if (i < 0) goto menu_continue;
      if (sscanf(num_buf, "%d%n", &last_num, &n) != 1 || num_buf[n]
          || last_num < 0 || last_num >= 1000000 || last_num < first_num) {
        vis_err("Invalid number");
        goto menu_continue;
      }
      memset(num_buf, 0, sizeof(num_buf));
      i = ncurses_edit_string(LINES / 2, COLS, "Contest number:", num_buf, 200,
                              utf8_mode);
      if (i < 0) goto menu_continue;
      if (sscanf(num_buf, "%d%n", &contest_num, &n) != 1 || num_buf[n]
          || contest_num <= 0 || contest_num >= 1000000) {
        vis_err("Invalid number");
        goto menu_continue;
      }

      for (i = first_num; i <= last_num; i++) {
        user_id = -1;
        j = userlist_clnt_create_user(server_conn, ULS_CREATE_USER, 0,&user_id);
        if (j < 0) {
          vis_err("Adding failed: %s", userlist_strerror(-j));
          goto menu_continue;
        }
        snprintf(valbuf, sizeof(valbuf), templ_buf, i);
        j = userlist_clnt_edit_field(server_conn, ULS_EDIT_FIELD, user_id, 0, 0,
                                     USERLIST_NN_LOGIN, valbuf);
        if (j < 0) {
          vis_err("Setting login failed: %s", userlist_strerror(-j));
          goto menu_continue;
        }
        j = userlist_clnt_edit_field(server_conn, ULS_EDIT_FIELD,
                                     user_id, contest_num, 0,
                                     USERLIST_NC_NAME, valbuf);
        if (j < 0) {
          vis_err("Setting name failed: %s", userlist_strerror(-j));
          goto menu_continue;
        }
        snprintf(valbuf, sizeof(valbuf), "N/A");
        j = userlist_clnt_edit_field(server_conn, ULS_EDIT_FIELD, user_id, 0, 0,
                                     USERLIST_NN_EMAIL, valbuf);
        if (j < 0) {
          vis_err("Setting e-mail failed: %s", userlist_strerror(-j));
          goto menu_continue;
        }
        snprintf(valbuf, sizeof(valbuf), passwd_buf, i);
        j = userlist_clnt_edit_field(server_conn, ULS_EDIT_FIELD, user_id, 0, 0,
                                     USERLIST_NN_PASSWD, valbuf);
        if (j < 0) {
          vis_err("Setting reg. password failed: %s", userlist_strerror(-j));
          goto menu_continue;
        }
        j = userlist_clnt_register_contest(server_conn, ULS_PRIV_REGISTER_CONTEST,
                                           user_id,contest_num, 0, 0);
        if (j < 0) {
          vis_err("Registration for contest %d failed: %s",
                  contest_num, userlist_strerror(-j));
          goto menu_continue;
        }
      }
      retval = -2;
      c = 'q';
    }
    if (c == ';') {
      i = item_index(current_item(menu));
      ASSERT(i >= 0 && i < nusers);
      if (g_sel_users.mask[i]) {
        g_sel_users.mask[i] = 0;
        g_sel_users.total_selected--;
      } else {
        g_sel_users.mask[i] = 1;
        g_sel_users.total_selected++;
      }
      descs[i] = make_user_menu_item(descs[i], uu[i], g_sel_users.mask[i]);
      menu_driver(menu, REQ_DOWN_ITEM);
      goto menu_continue;
    }
    if (c == 't') {
      g_sel_users.total_selected = 0;
      for (j = 0; j < nusers; j++)
        g_sel_users.total_selected += (g_sel_users.mask[j] ^= 1);
      retval = -2;
      c = 'q';
      *p_start_item = uu[item_index(current_item(menu))]->id;
    }
    if (c == '0') {
      memset(g_sel_users.mask, 0, g_sel_users.allocated);
      g_sel_users.total_selected = 0;
      retval = -2;
      c = 'q';
      *p_start_item = uu[item_index(current_item(menu))]->id;
    }
    if (c == 'c') {
      if ((k = display_contests_menu(current_level, 1)) <= 0) continue;

      if (!g_sel_users.total_selected && !sel_cnts.total_selected) {
        // register the current user to the specified contest
        i = item_index(current_item(menu));
        if (okcancel("Register user %d for contest %d?", uu[i]->id, k) != 1)
          goto menu_continue;
        r = userlist_clnt_register_contest(server_conn,
                                           ULS_PRIV_REGISTER_CONTEST,
                                           uu[i]->id, k, 0, 0);
        if (r < 0) {
          vis_err("Registration failed: %s", userlist_strerror(-r));
          goto menu_continue;
        }
      } else if (!sel_cnts.total_selected) {
        // register the selected users to the specified contest
        if (okcancel("Register selected users for contest %d?", k) != 1)
          goto menu_continue;
        for (j = 0; j < nusers; j++) {
          if (!g_sel_users.mask[j]) continue;
          r = userlist_clnt_register_contest(server_conn,
                                             ULS_PRIV_REGISTER_CONTEST,
                                             uu[j]->id, k, 0, 0);
          if (r < 0) {
            vis_err("Registration failed: %s", userlist_strerror(-r));
            goto menu_continue;
          }
          g_sel_users.mask[j] = 0;
          descs[j] = make_user_menu_item(descs[j], uu[j], g_sel_users.mask[j]);
        }
        memset(g_sel_users.mask, 0, g_sel_users.allocated);
        g_sel_users.total_selected = 0;
      } else if (!g_sel_users.total_selected) {
        // register the current user to the selected contests
        i = item_index(current_item(menu));
        if (okcancel("Register user %d for selected contests?", uu[i]->id) != 1)
          goto menu_continue;
        for (j = 1; j < sel_cnts.allocated; j++) {
          if (!sel_cnts.mask[j]) continue;
          r = userlist_clnt_register_contest(server_conn,
                                             ULS_PRIV_REGISTER_CONTEST,
                                             uu[i]->id, j, 0, 0);
          if (r < 0) {
            vis_err("Registration for contest %d failed: %s", j, userlist_strerror(-r));
            goto menu_continue;
          }
        }
      } else {
        // register the selected users to the selected contests
        if (okcancel("Register selected users for selected contests?") != 1)
          goto menu_continue;
        for (j = 0; j < nusers; j++) {
          if (!g_sel_users.mask[j]) continue;
          for (k = 1; k < sel_cnts.allocated; k++) {
            if (!sel_cnts.mask[k]) continue;
            r = userlist_clnt_register_contest(server_conn,
                                               ULS_PRIV_REGISTER_CONTEST,
                                               uu[j]->id, k, 0, 0);
            if (r < 0) {
              vis_err("Registration of user %d to contest %d failed: %s",
                      uu[j]->id, k, userlist_strerror(-r));
              goto menu_continue;
            }
          }
          g_sel_users.mask[j] = 0;
          descs[j] = make_user_menu_item(descs[j], uu[j], g_sel_users.mask[j]);
        }
        memset(g_sel_users.mask, 0, g_sel_users.allocated);
        g_sel_users.total_selected = 0;
      }
      goto menu_continue;
    }

  menu_continue:
    unpost_menu(menu);
    hide_panel(out_pan);
    hide_panel(in_pan);
    update_panels();
    doupdate();

    if (c == 'q') break;
    if (c == '\n' && only_choose) {
      i = item_index(current_item(menu));
      retval = uu[i]->id;
      *p_start_item = retval;
      break;
    }
    if (c == '\n' && !only_choose) {
      i = item_index(current_item(menu));
      j = 0;
      if (display_user(current_level, uu[i]->id, 0)) {
        // save the current user and redraw the screen
        *p_start_item = uu[i]->id;
        retval = -2;
        break;
      }
    }
  }

  // cleanup
  wmove(stdscr, 0, 0);
  wclrtoeol(stdscr);
  del_panel(in_pan);
  del_panel(out_pan);
  free_menu(menu);
  delwin(out_win);
  delwin(in_win);
  for (i = 0; i < nusers; i++) {
    free_item(items[i]);
    xfree(descs[i]);
  }
  userlist_free(&users->b);
  xfree(uu);
  xfree(descs);
  xfree(items);
  return retval;
}

static int
display_user_menu(unsigned char *upper, int start_item, int only_choose)
{
  int val = -2;

  while (val == -2) {
    val = do_display_user_menu(upper, &start_item, only_choose);
  }
  return val;
}

static unsigned char *
group_member_menu_entry(
        unsigned char *prev_entry,
        struct userlist_groupmember *gm,
        int sel_flag)
{
  unsigned char buf[512];
  unsigned char *s = buf;
  *s++ = sel_flag?'!':' ';
  s += sprintf(s, "%-6d ", gm->user->id);
  append_padded_string(s, gm->user->login, 50);
  xfree(prev_entry);
  return xstrdup(buf);
}

static int
do_display_group_members_menu(
        const unsigned char *upper,
        int group_id,
        int *p_user_id,
        int only_choose)
{
  int retval = -1;
  int user_id = 0;
  int r, i, j, contest_id;
  unsigned char current_level[512];
  unsigned char *xml_text = 0;
  struct userlist_list *users = 0;
  struct userlist_group *grp = 0;
  int member_count = 0;
  struct xml_tree *t;
  struct userlist_groupmember *gm;
  struct userlist_groupmember **uu = 0;
  unsigned char **descs = 0;
  int need_clear = 0, height = 0, cur_pos, first_row, c, cmd, done = 0;
  ITEM **items = 0;
  MENU *menu = 0;
  WINDOW *in_win = 0, *out_win = 0;
  PANEL *out_pan = 0, *in_pan = 0;

  if (p_user_id) user_id = *p_user_id;
  snprintf(current_level, sizeof(current_level), "%s->Group %d members", upper,
           group_id);
  r = userlist_clnt_list_all_users(server_conn, ULS_LIST_GROUP_USERS,
                                   group_id, &xml_text);
  if (r < 0) {
    vis_err("Cannot get group members: %s", userlist_strerror(-r));
    goto cleanup;
  }
  users = userlist_parse_str(xml_text);
  if (!users) {
    vis_err("XML parse error");
    goto cleanup;
  }
  xfree(xml_text); xml_text = 0;

  if (group_id <= 0 || group_id >= users->group_map_size
      || !(grp = users->group_map[group_id])) {
    vis_err("Invalid group");
    goto cleanup;
  }

  member_count = 0;
  if (users->groupmembers_node) {
    for (t = users->groupmembers_node->first_down; t; t = t->right) {
      ASSERT(t->tag == USERLIST_T_USERGROUPMEMBER);
      gm = (struct userlist_groupmember*) t;
      if (gm->group_id != group_id) continue;
      if (gm->user_id <= 0 || gm->user_id >= users->user_map_size
          || !users->user_map[gm->user_id])
        continue;
      gm->user = users->user_map[gm->user_id];
      ++member_count;
    }
  }
  if (!member_count) {
    j = okcancel("No members in the group. Add a new member?");
    if (j != 1) goto cleanup;
    i = display_user_menu(current_level, 0, 1);
    if (i > 0 && g_sel_users.total_selected > 0) {
      j = okcancel("Add %d users?", g_sel_users.total_selected);
      if (j != 1) {
        selected_mask_clear(&g_sel_users);
        goto cleanup;
      }
      for (i = 0; i < g_sel_users.used; ++i) {
        if (!g_sel_users.mask[i]) continue;
        j = userlist_clnt_register_contest(server_conn, ULS_CREATE_GROUP_MEMBER,
                                           g_sel_users.ids[i], group_id, 0, 0);
        if (j < 0) {
          vis_err("Member creation failed: %s", userlist_strerror(-r));
          selected_mask_clear(&g_sel_users);
          goto cleanup;
        }
      }
      selected_mask_clear(&g_sel_users);
    } else {
      if (i <= 0) goto cleanup;
      j = okcancel("Add user %d?", i);
      if (j != 1) goto cleanup;
      j = userlist_clnt_register_contest(server_conn, ULS_CREATE_GROUP_MEMBER,
                                         i, group_id, 0, 0);
      if (j < 0) {
        vis_err("Member creation failed: %s", userlist_strerror(-r));
        goto cleanup;
      }
      user_id = 0;
      retval = -2;
      goto cleanup;
    }
  }

  XCALLOC(uu, member_count);
  for (t = users->groupmembers_node->first_down, i = 0; t; t = t->right) {
    gm = (struct userlist_groupmember*) t;
    if (gm->user) {
      uu[i++] = gm;
    }
  }
  ASSERT(i == member_count);

  if (member_count != sel_members.used || group_id != sel_members.contest_id) {
    selected_mask_allocate(&sel_members, member_count);
  } else {
    for (j = 0; j < member_count; ++j) {
      if (uu[j]->user_id != sel_groups.ids[j])
        break;
    }
    if (j < member_count) {
      selected_mask_allocate(&sel_members, member_count);
    }
  }
  sel_members.contest_id = group_id;
  for (j = 0; j < member_count; ++j) {
    sel_members.ids[j] = uu[j]->user_id;
  }

  XCALLOC(descs, member_count);
  for (i = 0; i < member_count; ++i) {
    descs[i] = group_member_menu_entry(descs[i], uu[i], sel_members.mask[j]);
  }

  XCALLOC(items, member_count + 1);
  for (i = 0; i < member_count; ++i) {
    items[i] = new_item(descs[i], 0);
  }
  height = LINES - 4;
  need_clear = 1;
  menu = new_menu(items);
  set_menu_back(menu, COLOR_PAIR(1));
  set_menu_fore(menu, COLOR_PAIR(3));
  out_win = newwin(height + 2, COLS, 1, 0);
  in_win = newwin(height, COLS - 2, 2, 1);
  wattrset(out_win, COLOR_PAIR(1));
  wbkgdset(out_win, COLOR_PAIR(1));
  wattrset(in_win, COLOR_PAIR(1));
  wbkgdset(in_win, COLOR_PAIR(1));
  wclear(in_win);
  wclear(out_win);
  box(out_win, 0, 0);
  out_pan = new_panel(out_win);
  in_pan = new_panel(in_win);
  set_menu_win(menu, in_win);
  set_menu_sub(menu, in_win);
  set_menu_format(menu, height, 0);

  for (cur_pos = 0; cur_pos < member_count; ++cur_pos)
    if (uu[cur_pos]->user_id == user_id)
      break;
  if (cur_pos >= member_count)
    cur_pos = 0;
  first_row = cur_pos - height / 2;
  if (first_row + height > member_count) first_row = member_count - height;
  if (first_row < 0) first_row = 0;
  set_top_row(menu, first_row);
  set_current_item(menu, items[cur_pos]);

  do {
    mvwprintw(stdscr, 0, 0, "%s", current_level);
    wclrtoeol(stdscr);
    print_help("Quit Add Delete Contests :-Sel Toggle 0-clear");
    show_panel(out_pan);
    show_panel(in_pan);
    post_menu(menu);
    update_panels();
    doupdate();

    while (1) {
      c = ncurses_getkey(utf8_mode, 0);
      if (c == KEY_BACKSPACE || c == KEY_DC || c == 127 || c == 8 || c == 'd') {
        c = 'd';
        break;
      }
      if (c == 'q' || c == ('G' & 31) || c == '\033') {
        c = 'q';
        break;
      }
      if (c == '\n' || c == '\r') {
        c = '\n';
        break;
      }
      if (c == 'a' || c == 't' || c == ';' || c == '0' || c == 'c') {
        break;
      }

      cmd = -1;
      switch (c) {
      case KEY_UP:
      case KEY_LEFT:
        cmd = REQ_UP_ITEM;
        break;
      case KEY_DOWN:
      case KEY_RIGHT:
        cmd = REQ_DOWN_ITEM;
        break;
      case KEY_HOME:
        cmd = REQ_FIRST_ITEM;
        break;
      case KEY_END:
        cmd = REQ_LAST_ITEM;
        break;
      case KEY_NPAGE:
        i = item_index(current_item(menu));
        if (i + height >= member_count) cmd = REQ_LAST_ITEM;
        else cmd = REQ_SCR_DPAGE;
        break;
      case KEY_PPAGE:
        i = item_index(current_item(menu));
        if (i - height < 0) cmd = REQ_FIRST_ITEM;
        else cmd = REQ_SCR_UPAGE;
        break;
      }
      if (cmd != -1) {
        menu_driver(menu, cmd);
        update_panels();
        doupdate();
      }
    }

    if (c == 'd' && !only_choose) {
      if (sel_members.total_selected > 0) {
        j = okcancel("REMOVE %d GROUP MEMBERS?", sel_members.total_selected);
        if (j == 1) {
          r = 0;
          for (i = 0; i < sel_members.used; ++i) {
            if (!sel_members.mask[i]) continue;
            j = userlist_clnt_register_contest(server_conn,
                                               ULS_DELETE_GROUP_MEMBER,
                                               sel_members.ids[i],
                                               group_id, 0, 0);
            if (j < 0) ++r;
          }
          if (r > 0) {
            vis_err("Delete of %d members failed", r);
          }
          selected_mask_clear(&sel_members);
          done = 1;
          retval = -2;
          user_id = 0;
        }
      } else {
        i = item_index(current_item(menu));
        j = okcancel("REMOVE GROUP MEMBER %d (%s)?", uu[i]->user_id,
                     uu[i]->user->login);
        if (j == 1) {
          j = userlist_clnt_register_contest(server_conn,
                                             ULS_DELETE_GROUP_MEMBER,
                                             uu[i]->user_id, group_id, 0, 0);
          if (j < 0) {
            vis_err("Delete failed: %s", userlist_strerror(-j));
          } else {
            done = 1;
            retval = -2;
            user_id = 0;
          }
        }
      }
    } else if (c == 'a' && !only_choose) {
      i = display_user_menu(current_level, 0, 1);
      if (i > 0 && g_sel_users.total_selected > 0) {
        j = okcancel("Add %d users?", g_sel_users.total_selected);
        if (j == 1) {
          r = 0;
          for (i = 0; i < g_sel_users.used; ++i) {
            if (!g_sel_users.mask[i]) continue;
            j = userlist_clnt_register_contest(server_conn,
                                               ULS_CREATE_GROUP_MEMBER,
                                               g_sel_users.ids[i], group_id, 0,0);
            if (j < 0) ++r;
          }
          if (r > 0) {
            vis_err("Adding of %d members failed", r);
          }
          selected_mask_clear(&g_sel_users);
          done = 1;
          retval = -2;
          user_id = 0;
        } else {
          selected_mask_clear(&g_sel_users);
        }
      } else if (i > 0) {
        j = userlist_clnt_register_contest(server_conn, ULS_CREATE_GROUP_MEMBER,
                                           i, group_id, 0, 0);
        if (j < 0) {
          vis_err("Member creation failed: %s", userlist_strerror(-r));
        } else {
          done = 1;
          retval = -2;
          user_id = 0;
        }
      }
    } else if (c == 'q') {
      retval = -1;
      done = 1;
    } else if (c == '0') {
      for (j = 0; j < member_count; ++j) {
        if (sel_members.mask[j]) {
          sel_members.mask[j] = 0;
          --sel_members.total_selected;
          descs[j]=group_member_menu_entry(descs[j],uu[j],sel_members.mask[j]);
        }
      }
    } else if (c == 't') {
      for (j = 0; j < member_count; ++j) {
        if (sel_members.mask[j]) {
          sel_members.mask[j] = 0;
          --sel_members.total_selected;
        } else {
          sel_members.mask[j] = 1;
          ++sel_members.total_selected;
        }
        descs[j] = group_member_menu_entry(descs[j], uu[j],sel_members.mask[j]);
      }
    } else if (c == ';') {
      i = item_index(current_item(menu));
      if (sel_members.mask[i]) {
        sel_members.mask[i] = 0;
        --sel_members.total_selected;
      } else {
        sel_members.mask[i] = 1;
        ++sel_members.total_selected;
      }
      descs[i] = group_member_menu_entry(descs[i], uu[i], sel_members.mask[i]);
      menu_driver(menu, REQ_DOWN_ITEM);
    } else if (c == 'c') {
      contest_id = display_contests_menu(current_level, 1);
      if (contest_id > 0) {
        j = okcancel("Add users from contest %d?", contest_id);
        if (j == 1) {
          i = display_registered_users(current_level, contest_id, 0, 1);
          if (i > 0 && sel_users.total_selected > 0) {
            j = okcancel("Add %d users?", sel_users.total_selected);
            if (j == 1) {
              r = 0;
              for (i = 0; i < sel_users.used; ++i) {
                if (!sel_users.mask[i]) continue;
                j = userlist_clnt_register_contest(server_conn,
                                                   ULS_CREATE_GROUP_MEMBER,
                                                   sel_users.ids[i], group_id, 0,0);
                if (j < 0) ++r;
              }
              if (r > 0) {
                vis_err("Adding of %d members failed", r);
              }
              selected_mask_clear(&sel_users);
              done = 1;
              retval = -2;
              user_id = 0;
            } else {
              selected_mask_clear(&sel_users);
            }
          } else if (i > 0) {
            j = userlist_clnt_register_contest(server_conn, ULS_CREATE_GROUP_MEMBER,
                                               i, group_id, 0, 0);
            if (j < 0) {
              vis_err("Member creation failed: %s", userlist_strerror(-r));
            } else {
              done = 1;
              retval = -2;
              user_id = 0;
            }
          }
        }
      }
    }

  /*
    else if (c == 'n') {
      // edit name
      i = item_index(current_item(menu));
      buf[0] = 0;
      if (uu[i]->group_name) {
        snprintf(buf, sizeof(buf), "%s", uu[i]->group_name);
      }
      j = ncurses_edit_string(LINES / 2, COLS, "Change the group name",
                              buf, sizeof(buf) - 1, utf8_mode);
      if (j >= 0) {
        j = userlist_clnt_edit_field(server_conn, ULS_EDIT_GROUP_FIELD,
                                     uu[i]->group_id, 0,
                                     0, USERLIST_GRP_GROUP_NAME, buf);
        if (j < 0) {
          vis_err("Operation failed: %s", userlist_strerror(-j));
        } else {
          done = 1;
          retval = -2;
          group_id = uu[i]->group_id;
        }
      }
    } else if (c == 'c') {
      // edit description
      i = item_index(current_item(menu));
      buf[0] = 0;
      if (uu[i]->description) {
        snprintf(buf, sizeof(buf), "%s", uu[i]->description);
      }
      j = ncurses_edit_string(LINES / 2, COLS, "Change the group description",
                              buf, sizeof(buf) - 1, utf8_mode);
      if (j >= 0) {
        j = userlist_clnt_edit_field(server_conn, ULS_EDIT_GROUP_FIELD,
                                     uu[i]->group_id, 0,
                                     0, USERLIST_GRP_DESCRIPTION, buf);
        if (j < 0) {
          vis_err("Operation failed: %s", userlist_strerror(-j));
        } else {
          done = 1;
          retval = -2;
          group_id = uu[i]->group_id;
        }
      }
    } else if (c == 'z') {
      // delete description
      i = item_index(current_item(menu));
      j = userlist_clnt_delete_field(server_conn, ULS_DELETE_GROUP_FIELD,
                                     uu[i]->group_id, 0, 0,
                                     USERLIST_GRP_DESCRIPTION);
      if (j < 0) {
        vis_err("Operation failed: %s", userlist_strerror(-j));
      } else {
        done = 1;
        retval = -2;
        group_id = uu[i]->group_id;
      }
    } else if (c == 'm') {
      // view members
      i = item_index(current_item(menu));
      display_group_members_menu(current_level, uu[i]->group_id, 0, 0);
      done = 1;
      retval = -2;
      group_id = uu[i]->group_id;
    }

   */
    unpost_menu(menu);
    hide_panel(out_pan);
    hide_panel(in_pan);
    update_panels();
    doupdate();
  } while (!done);

 cleanup:
  if (in_pan) del_panel(in_pan);
  if (out_pan) del_panel(out_pan);
  if (in_win) delwin(in_win);
  if (out_win) delwin(out_win);
  if (need_clear) {
    wmove(stdscr, 0, 0);
    wclrtoeol(stdscr);
  }
  if (menu) free_menu(menu);
  if (items) {
    for (i = 0; i < member_count; ++i) {
      free_item(items[i]);
    }
    xfree(items);
  }
  if (member_count > 0) {
    for (i = 0; i < member_count; ++i) {
      xfree(descs[i]);
    }
  }
  xfree(descs);
  xfree(uu);
  if (users) {
    userlist_free(&users->b);
  }
  xfree(xml_text);
  if (p_user_id) *p_user_id = user_id;
  return retval;
}

static int
display_group_members_menu(
        const unsigned char *upper,
        int group_id,
        int user_id,
        int only_choose)
{
  int val = -2;

  while ((val = do_display_group_members_menu(upper, group_id, &user_id,
                                              only_choose)) == -2) {
  }
  return val;
}

static unsigned char *
group_menu_entry(
        unsigned char *prev_entry,
        struct userlist_group *grp,
        int sel_flag)
{
  const unsigned char *description = grp->description;
  if (!description) description = "";

  // FIXME: use COLS
  unsigned char buf[512];
  unsigned char *s = buf;
  *s++ = sel_flag?'!':' ';
  s += sprintf(s, "%-6d ", grp->group_id);
  s = append_padded_string(s, grp->group_name, 15);
  *s++ = ' ';
  s = append_padded_string(s, description, 50);
  xfree(prev_entry);
  return xstrdup(buf);
}

/*
  return values: -2 means restart the function
  -1 - no group selected
 */
static int
do_display_group_menu(
        const unsigned char *upper,
        int *p_group_id,
        int only_choose)
{
  int group_id = 0;
  int retval = -1;
  unsigned char current_level[512];
  int r, i, j, group_count = 0;
  unsigned char *xml_text = 0;
  struct userlist_list *users = 0;
  struct userlist_group **uu = 0;
  unsigned char **descs = 0;
  unsigned char buf[1024];
  ITEM **items = 0;
  MENU *menu = 0;
  WINDOW *in_win = 0, *out_win = 0;
  PANEL *out_pan = 0, *in_pan = 0;
  int cur_pos, first_row, height, need_clear = 0;
  int c, cmd, done = 0;

  if (p_group_id) group_id = *p_group_id;
  snprintf(current_level, sizeof(current_level), "%s->%s", upper, "Group list");

  r = userlist_clnt_list_all_users(server_conn, ULS_LIST_ALL_GROUPS,
                                   0, &xml_text);
  if (r < 0) {
    vis_err("Cannot get the list of groups: %s", userlist_strerror(-r));
    goto cleanup;
  }
  users = userlist_parse_str(xml_text);
  if (!users) {
    vis_err("XML parse error");
    goto cleanup;
  }
  xfree(xml_text); xml_text = 0;

  group_count = 0;
  for (i = 1; i < users->group_map_size; ++i) {
    group_count += (users->group_map[i] != NULL);
  }
  if (!group_count) {
    j = okcancel("No groups in the database. Add a new group?");
    if (j != 1) goto cleanup;
    j = userlist_clnt_create_user(server_conn, ULS_CREATE_GROUP, 0, 0);
    if (j < 0) {
      vis_err("Group creation failed: %s", userlist_strerror(-j));
      goto cleanup;
    }
    group_id = 0;
    retval = -2;
    goto cleanup;
  }

  XCALLOC(uu, group_count);
  for (i = 1, j = 0; i < users->group_map_size; ++i) {
    if (users->group_map[i]) {
      uu[j++] = users->group_map[i];
    }
  }
  ASSERT(j == group_count);

  if (group_count != sel_groups.used) {
    selected_mask_allocate(&sel_groups, group_count);
  } else {
    for (j = 0; j < group_count; ++j) {
      if (uu[j]->group_id != sel_groups.ids[j])
        break;
    }
    if (j < group_count) {
      selected_mask_allocate(&sel_groups, group_count);
    }
  }
  for (j = 0; j < group_count; ++j) {
    sel_groups.ids[j] = uu[j]->group_id;
  }

  XCALLOC(descs, group_count);
  for (i = 0; i < group_count; ++i) {
    descs[i] = group_menu_entry(descs[i], uu[i], sel_groups.mask[i]);
  }

  XCALLOC(items, group_count + 1);
  for (i = 0; i < group_count; ++i) {
    items[i] = new_item(descs[i], 0);
  }
  height = LINES - 4;
  need_clear = 1;
  menu = new_menu(items);
  set_menu_back(menu, COLOR_PAIR(1));
  set_menu_fore(menu, COLOR_PAIR(3));
  out_win = newwin(height + 2, COLS, 1, 0);
  in_win = newwin(height, COLS - 2, 2, 1);
  wattrset(out_win, COLOR_PAIR(1));
  wbkgdset(out_win, COLOR_PAIR(1));
  wattrset(in_win, COLOR_PAIR(1));
  wbkgdset(in_win, COLOR_PAIR(1));
  wclear(in_win);
  wclear(out_win);
  box(out_win, 0, 0);
  out_pan = new_panel(out_win);
  in_pan = new_panel(in_win);
  set_menu_win(menu, in_win);
  set_menu_sub(menu, in_win);
  set_menu_format(menu, height, 0);

  for (cur_pos = 0; cur_pos < group_count; ++cur_pos)
    if (uu[cur_pos]->group_id == group_id)
      break;
  if (cur_pos >= group_count)
    cur_pos = 0;
  first_row = cur_pos - height / 2;
  if (first_row + height > group_count) first_row = group_count - height;
  if (first_row < 0) first_row = 0;
  set_top_row(menu, first_row);
  set_current_item(menu, items[cur_pos]);

  do {
    mvwprintw(stdscr, 0, 0, "%s", current_level);
    wclrtoeol(stdscr);
    print_help("Quit Add Delete Name desCription Members :-Sel Toggle 0-clear");
    show_panel(out_pan);
    show_panel(in_pan);
    post_menu(menu);
    update_panels();
    doupdate();

    while (1) {
      c = ncurses_getkey(utf8_mode, 0);
      if (c == KEY_BACKSPACE || c == KEY_DC || c == 127 || c == 8 || c == 'd') {
        c = 'd';
        break;
      }
      if (c == 'q' || c == ('G' & 31) || c == '\033') {
        c = 'q';
        break;
      }
      if (c == '\n' || c == '\r') {
        c = '\n';
        break;
      }

      if (c == 'a' || c == 'n' || c == 'c' || c == 'z' || c == 'm'
          || c == ';' || c == 't' || c == '0') {
        break;
      }

      cmd = -1;
      switch (c) {
      case KEY_UP:
      case KEY_LEFT:
        cmd = REQ_UP_ITEM;
        break;
      case KEY_DOWN:
      case KEY_RIGHT:
        cmd = REQ_DOWN_ITEM;
        break;
      case KEY_HOME:
        cmd = REQ_FIRST_ITEM;
        break;
      case KEY_END:
        cmd = REQ_LAST_ITEM;
        break;
      case KEY_NPAGE:
        i = item_index(current_item(menu));
        if (i + height >= group_count) cmd = REQ_LAST_ITEM;
        else cmd = REQ_SCR_DPAGE;
        break;
      case KEY_PPAGE:
        i = item_index(current_item(menu));
        if (i - height < 0) cmd = REQ_FIRST_ITEM;
        else cmd = REQ_SCR_UPAGE;
        break;
      }
      if (cmd != -1) {
        menu_driver(menu, cmd);
        update_panels();
        doupdate();
      }
    }

    if (c == 'd' && !only_choose) {
      i = item_index(current_item(menu));
      j = okcancel("REMOVE GROUP %d (%s)?", uu[i]->group_id, uu[i]->group_name);
      if (j == 1) {
        j = userlist_clnt_delete_info(server_conn, ULS_DELETE_GROUP,
                                      uu[i]->group_id, 0, 0);
        if (j < 0) {
          vis_err("Delete failed: %s", userlist_strerror(-j));
        } else {
          done = 1;
          retval = -2;
          group_id = 0;
        }
      }
    } else if (c == 'a' && !only_choose) {
      j = okcancel("Create a new group?");
      if (j == 1) {
        j = userlist_clnt_create_user(server_conn, ULS_CREATE_GROUP, 0, &i);
        if (j < 0) {
          vis_err("Create failed: %s", userlist_strerror(-j));
        } else {
          done = 1;
          retval = -2;
          group_id = i;
        }
      }
    } else if (c == 'q') {
      retval = -1;
      done = 1;
    } else if (c == '\n' && only_choose) {
    } else if (c == 'n') {
      // edit name
      i = item_index(current_item(menu));
      buf[0] = 0;
      if (uu[i]->group_name) {
        snprintf(buf, sizeof(buf), "%s", uu[i]->group_name);
      }
      j = ncurses_edit_string(LINES / 2, COLS, "Change the group name",
                              buf, sizeof(buf) - 1, utf8_mode);
      if (j >= 0) {
        j = userlist_clnt_edit_field(server_conn, ULS_EDIT_GROUP_FIELD,
                                     uu[i]->group_id, 0,
                                     0, USERLIST_GRP_GROUP_NAME, buf);
        if (j < 0) {
          vis_err("Operation failed: %s", userlist_strerror(-j));
        } else {
          done = 1;
          retval = -2;
          group_id = uu[i]->group_id;
        }
      }
    } else if (c == 'c') {
      // edit description
      i = item_index(current_item(menu));
      buf[0] = 0;
      if (uu[i]->description) {
        snprintf(buf, sizeof(buf), "%s", uu[i]->description);
      }
      j = ncurses_edit_string(LINES / 2, COLS, "Change the group description",
                              buf, sizeof(buf) - 1, utf8_mode);
      if (j >= 0) {
        j = userlist_clnt_edit_field(server_conn, ULS_EDIT_GROUP_FIELD,
                                     uu[i]->group_id, 0,
                                     0, USERLIST_GRP_DESCRIPTION, buf);
        if (j < 0) {
          vis_err("Operation failed: %s", userlist_strerror(-j));
        } else {
          done = 1;
          retval = -2;
          group_id = uu[i]->group_id;
        }
      }
    } else if (c == 'z') {
      // delete description
      i = item_index(current_item(menu));
      j = userlist_clnt_delete_field(server_conn, ULS_DELETE_GROUP_FIELD,
                                     uu[i]->group_id, 0, 0,
                                     USERLIST_GRP_DESCRIPTION);
      if (j < 0) {
        vis_err("Operation failed: %s", userlist_strerror(-j));
      } else {
        done = 1;
        retval = -2;
        group_id = uu[i]->group_id;
      }
    } else if (c == 'm') {
      // view members
      i = item_index(current_item(menu));
      display_group_members_menu(current_level, uu[i]->group_id, 0, 0);
      done = 1;
      retval = -2;
      group_id = uu[i]->group_id;
    } else if (c == '0') {
      for (j = 0; j < group_count; ++j) {
        if (sel_groups.mask[j]) {
          sel_groups.mask[j] = 0;
          --sel_groups.total_selected;
          descs[j] = group_menu_entry(descs[j], uu[j], sel_groups.mask[j]);
        }
      }
    } else if (c == 't') {
      for (j = 0; j < group_count; ++j) {
        if (sel_groups.mask[j]) {
          sel_groups.mask[j] = 0;
          --sel_groups.total_selected;
        } else {
          sel_groups.mask[j] = 1;
          ++sel_groups.total_selected;
        }
        descs[j] = group_menu_entry(descs[j], uu[j], sel_groups.mask[j]);
      }
    } else if (c == ';') {
      i = item_index(current_item(menu));
      if (sel_groups.mask[i]) {
        sel_groups.mask[i] = 0;
        --sel_groups.total_selected;
      } else {
        sel_groups.mask[i] = 1;
        ++sel_groups.total_selected;
      }
      descs[i] = group_menu_entry(descs[i], uu[i], sel_groups.mask[i]);
      menu_driver(menu, REQ_DOWN_ITEM);
    }

    unpost_menu(menu);
    hide_panel(out_pan);
    hide_panel(in_pan);
    update_panels();
    doupdate();
  } while (!done);

 cleanup: ;
  if (in_pan) del_panel(in_pan);
  if (out_pan) del_panel(out_pan);
  if (in_win) delwin(in_win);
  if (out_win) delwin(out_win);
  if (need_clear) {
    wmove(stdscr, 0, 0);
    wclrtoeol(stdscr);
  }
  if (menu) free_menu(menu);
  if (items) {
    for (i = 0; i < group_count; ++i) {
      free_item(items[i]);
    }
    xfree(items);
  }
  if (descs) {
    for (i = 0; i < group_count; ++i)
      xfree(descs[i]);
    xfree(descs);
  }
  xfree(uu);
  xfree(xml_text);
  userlist_free(&users->b);
  if (p_group_id) *p_group_id = group_id;
  return retval;
}

static int
display_group_menu(
        const unsigned char *upper,
        int start_item,
        int only_choose)
{
  int val = -2;

  while (val == -2) {
    val = do_display_group_menu(upper, &start_item, only_choose);
  }
  return val;
}

static void
display_main_menu(void)
{
  ITEM *items[6];
  MENU *menu;
  WINDOW *window, *in_win;
  PANEL *panel, *in_pan;
  int req_rows, req_cols;
  int c, cmd, start_col, r = 0, cur_group = 0;
  unsigned char current_level[512];

  snprintf(current_level, sizeof(current_level), "%s", "Main menu");

  memset(items, 0, sizeof(items));
  items[0] = new_item("View contests", 0);
  items[1] = new_item("View users", 0);
  items[2] = new_item("View groups", 0);
  items[3] = new_item("Quit", 0);
  menu = new_menu(items);
  scale_menu(menu, &req_rows, &req_cols);
  set_menu_back(menu, COLOR_PAIR(1));
  set_menu_fore(menu, COLOR_PAIR(3));

  start_col = (80 - req_cols - 2) / 2;
  window = newwin(req_rows + 2, req_cols + 2, 5, start_col);
  wattrset(window, COLOR_PAIR(1));
  wbkgdset(window, COLOR_PAIR(1));
  box(window, 0, 0);
  panel = new_panel(window);
  in_win = newwin(req_rows, req_cols, 6, start_col + 1);
  in_pan = new_panel(in_win);
  set_menu_win(menu, in_win);
  set_menu_sub(menu, in_win);

  while (1) {
    mvwprintw(stdscr, 0, 0, "%s", current_level);
    wclrtoeol(stdscr);
    print_help("Enter-view C-contests U-users G-groups Q-quit");
    show_panel(panel);
    show_panel(in_pan);
    post_menu(menu);
    update_panels();
    doupdate();

    while (1) {
      c = ncurses_getkey(utf8_mode, 0);
      switch (c) {
      case 'q': case 'G' & 31: case '\033':
        c = 'q';
        goto menu_done;
      case '\n': case '\r': case ' ':
        c = '\n';
        goto menu_done;
      case 'c': case 'u': case 'g':
        goto menu_done;
      }
      cmd = -1;
      switch (c) {
      case KEY_UP:
      case KEY_LEFT:
        cmd = REQ_UP_ITEM;
        break;
      case KEY_DOWN:
      case KEY_RIGHT:
        cmd = REQ_DOWN_ITEM;
        break;
      case KEY_HOME:
        cmd = REQ_FIRST_ITEM;
        break;
      case KEY_END:
        cmd = REQ_LAST_ITEM;
        break;
      case KEY_NPAGE:
        cmd = REQ_SCR_UPAGE;
        break;
      case KEY_PPAGE:
        cmd = REQ_SCR_DPAGE;
        break;
      }
      if (cmd != -1) {
        menu_driver(menu, cmd);
        update_panels();
        doupdate();
      }
    }
  menu_done:
    unpost_menu(menu);
    hide_panel(panel);
    hide_panel(in_pan);
    update_panels();
    doupdate();

    // handle the requested action
    if (c == '\n') {
      ITEM *cur = current_item(menu);
      if (cur == items[0]) {
        c = 'c';
      } else if (cur == items[1]) {
        c = 'u';
      } else if (cur == items[2]) {
        c = 'g';
      } else if (cur == items[3]) {
        c = 'q';
      }
    }
    if (c == 'q') break;
    if (c == 'c') {
      display_contests_menu(current_level, 0);
    } else if (c == 'u') {
      r = display_user_menu(current_level, r, 0);
    } else if (c == 'g') {
      cur_group = display_group_menu(current_level, cur_group, 0);
    }

    // perform other actions
  }

  // cleanup
  del_panel(in_pan);
  del_panel(panel);
  free_menu(menu);
  delwin(window);
  delwin(in_win);
  free_item(items[0]);
  free_item(items[1]);
  free_item(items[2]);
}

int
main(int argc, char **argv)
{
  int r;
  unsigned char *ejudge_xml_path = 0;

#if defined EJUDGE_XML_PATH
  if (argc == 1) {
    //fprintf(stderr, "%s: using the default %s\n", argv[0], EJUDGE_XML_PATH);
    ejudge_xml_path = EJUDGE_XML_PATH;
  } else if (argc != 2) {
    fprintf(stderr, "%s: invalid number of arguments\n", argv[0]);
    return 1;
  } else {
    ejudge_xml_path = argv[1];
  }
#else
  if (argc != 2) {
    fprintf(stderr, "%s: invalid number of arguments\n", argv[0]);
    return 1;
  }
  ejudge_xml_path = argv[1];
#endif

  if (!(config = ejudge_cfg_parse(ejudge_xml_path, 0))) {
    fprintf(stderr, "%s: cannot parse configuration file\n", argv[0]);
    return 1;
  }
  if ((r = contests_set_directory(config->contests_dir)) < 0) {
    fprintf(stderr, "%s: %s\n",
            argv[0], contests_strerror(-r));
    return 1;
  }
  if (!(server_conn = userlist_clnt_open(config->socket_path))) {
    fprintf(stderr, "%s: cannot open server connection: %s\n",
            argv[0], os_ErrorMsg());
    return 1;
  }
  if ((r = userlist_clnt_admin_process(server_conn, 0, 0, 0)) < 0) {
    fprintf(stderr, "%s: cannot become admin process: %s\n",
            argv[0], userlist_strerror(-r));
    return 1;
  }

  setlocale(LC_ALL, "");
  if (!strcmp(nl_langinfo(CODESET), "UTF-8")) utf8_mode = 1;

  if (!(root_window = initscr())) return 1;
  cbreak();
  noecho();
  nonl();
  meta(stdscr, TRUE);
  intrflush(stdscr, FALSE);
  keypad(stdscr, TRUE);

  if (has_colors()) {
    start_color();
    init_pair(1, COLOR_WHITE, COLOR_BLUE);
    init_pair(2, COLOR_YELLOW, COLOR_BLUE);
    init_pair(3, COLOR_BLUE, COLOR_WHITE);
    init_pair(4, COLOR_YELLOW, COLOR_RED);
  }
  attrset(COLOR_PAIR(1));
  bkgdset(COLOR_PAIR(1));
  clear();

  display_main_menu();

  bkgdset(COLOR_PAIR(0));
  clear();
  refresh();
  endwin();
  return 0;
}
