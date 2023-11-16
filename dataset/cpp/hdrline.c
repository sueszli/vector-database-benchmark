/**
 * @file
 * String processing routines to generate the mail index
 *
 * @authors
 * Copyright (C) 1996-2000,2002,2007 Michael R. Elkins <me@mutt.org>
 * Copyright (C) 2016 Richard Russon <rich@flatcap.org>
 * Copyright (C) 2016 Ian Zimmerman <itz@primate.net>
 * Copyright (C) 2019 Pietro Cerutti <gahr@gahr.ch>
 *
 * @copyright
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 2 of the License, or (at your option) any later
 * version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * @page neo_hdrline String processing routines to generate the mail index
 *
 * String processing routines to generate the mail index
 */

#include "config.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "mutt/lib.h"
#include "address/lib.h"
#include "config/lib.h"
#include "email/lib.h"
#include "core/lib.h"
#include "alias/lib.h"
#include "gui/lib.h"
#include "hdrline.h"
#include "attach/lib.h"
#include "color/lib.h"
#include "ncrypt/lib.h"
#include "format_flags.h"
#include "hook.h"
#include "maillist.h"
#include "mutt_thread.h"
#include "muttlib.h"
#include "mx.h"
#include "sort.h"
#include "subjectrx.h"
#ifdef USE_NOTMUCH
#include "notmuch/lib.h"
#endif

/**
 * struct HdrFormatInfo - Data passed to index_format_str()
 */
struct HdrFormatInfo
{
  struct Mailbox *mailbox;    ///< Current Mailbox
  int msg_in_pager;           ///< Index of Email displayed in the Pager
  struct Email *email;        ///< Current Email
  const char *pager_progress; ///< String representing Pager position through Email
};

/**
 * enum FieldType - Header types
 *
 * Strings for printing headers
 */
enum FieldType
{
  DISP_TO,    ///< To: string
  DISP_CC,    ///< Cc: string
  DISP_BCC,   ///< Bcc: string
  DISP_FROM,  ///< From: string
  DISP_PLAIN, ///< Empty string
  DISP_MAX,
};

/**
 * add_index_color - Insert a color marker into a string
 * @param buf    Buffer to store marker
 * @param buflen Buffer length
 * @param flags  Flags, see #MuttFormatFlags
 * @param color  Color, e.g. #MT_COLOR_MESSAGE
 * @retval num Characters written
 *
 * The colors are stored as "magic" strings embedded in the text.
 */
static size_t add_index_color(char *buf, size_t buflen, MuttFormatFlags flags, enum ColorId color)
{
  /* only add color markers if we are operating on main index entries. */
  if (!(flags & MUTT_FORMAT_INDEX))
    return 0;

  /* this item is going to be passed to an external filter */
  if (flags & MUTT_FORMAT_NOFILTER)
    return 0;

  if (color == MT_COLOR_INDEX)
  { /* buf might be uninitialized other cases */
    const size_t len = mutt_str_len(buf);
    buf += len;
    buflen -= len;
  }

  if (buflen <= 2)
    return 0;

  buf[0] = MUTT_SPECIAL_INDEX;
  buf[1] = color;
  buf[2] = '\0';

  return 2;
}

/**
 * make_from_prefix - Create a prefix for an author field
 * @param disp   Type of field
 * @retval ptr Prefix string (do not free it)
 *
 * If $from_chars is set, pick an appropriate character from it.
 * If not, use the default prefix: "To", "Cc", etc
 */
static const char *make_from_prefix(enum FieldType disp)
{
  /* need 2 bytes at the end, one for the space, another for NUL */
  static char padded[8];
  static const char *long_prefixes[DISP_MAX] = {
    [DISP_TO] = "To ", [DISP_CC] = "Cc ", [DISP_BCC] = "Bcc ",
    [DISP_FROM] = "",  [DISP_PLAIN] = "",
  };

  const struct MbTable *c_from_chars = cs_subset_mbtable(NeoMutt->sub, "from_chars");

  if (!c_from_chars || !c_from_chars->chars || (c_from_chars->len == 0))
    return long_prefixes[disp];

  const char *pchar = mbtable_get_nth_wchar(c_from_chars, disp);
  if (mutt_str_len(pchar) == 0)
    return "";

  snprintf(padded, sizeof(padded), "%s ", pchar);
  return padded;
}

/**
 * make_from - Generate a From: field (with optional prefix)
 * @param env      Envelope of the email
 * @param buf      Buffer to store the result
 * @param buflen   Size of the buffer
 * @param do_lists Should we check for mailing lists?
 * @param flags    Format flags, see #MuttFormatFlags
 *
 * Generate the %F or %L field in $index_format.
 * This is the author, or recipient of the email.
 *
 * The field can optionally be prefixed by a character from $from_chars.
 * If $from_chars is not set, the prefix will be, "To", "Cc", etc
 */
static void make_from(struct Envelope *env, char *buf, size_t buflen,
                      bool do_lists, MuttFormatFlags flags)
{
  if (!env || !buf)
    return;

  bool me;
  enum FieldType disp;
  struct AddressList *name = NULL;

  me = mutt_addr_is_user(TAILQ_FIRST(&env->from));

  if (do_lists || me)
  {
    if (check_for_mailing_list(&env->to, make_from_prefix(DISP_TO), buf, buflen))
      return;
    if (check_for_mailing_list(&env->cc, make_from_prefix(DISP_CC), buf, buflen))
      return;
  }

  if (me && !TAILQ_EMPTY(&env->to))
  {
    disp = (flags & MUTT_FORMAT_PLAIN) ? DISP_PLAIN : DISP_TO;
    name = &env->to;
  }
  else if (me && !TAILQ_EMPTY(&env->cc))
  {
    disp = DISP_CC;
    name = &env->cc;
  }
  else if (me && !TAILQ_EMPTY(&env->bcc))
  {
    disp = DISP_BCC;
    name = &env->bcc;
  }
  else if (!TAILQ_EMPTY(&env->from))
  {
    disp = DISP_FROM;
    name = &env->from;
  }
  else
  {
    *buf = '\0';
    return;
  }

  snprintf(buf, buflen, "%s%s", make_from_prefix(disp), mutt_get_name(TAILQ_FIRST(name)));
}

/**
 * make_from_addr - Create a 'from' address for a reply email
 * @param env      Envelope of current email
 * @param buf      Buffer for the result
 * @param buflen   Length of buffer
 * @param do_lists If true, check for mailing lists
 */
static void make_from_addr(struct Envelope *env, char *buf, size_t buflen, bool do_lists)
{
  if (!env || !buf)
    return;

  bool me = mutt_addr_is_user(TAILQ_FIRST(&env->from));

  if (do_lists || me)
  {
    if (check_for_mailing_list_addr(&env->to, buf, buflen))
      return;
    if (check_for_mailing_list_addr(&env->cc, buf, buflen))
      return;
  }

  if (me && !TAILQ_EMPTY(&env->to))
    snprintf(buf, buflen, "%s", buf_string(TAILQ_FIRST(&env->to)->mailbox));
  else if (me && !TAILQ_EMPTY(&env->cc))
    snprintf(buf, buflen, "%s", buf_string(TAILQ_FIRST(&env->cc)->mailbox));
  else if (!TAILQ_EMPTY(&env->from))
    mutt_str_copy(buf, buf_string(TAILQ_FIRST(&env->from)->mailbox), buflen);
  else
    *buf = '\0';
}

/**
 * user_in_addr - Do any of the addresses refer to the user?
 * @param al AddressList
 * @retval true Any of the addresses match one of the user's addresses
 */
static bool user_in_addr(struct AddressList *al)
{
  struct Address *a = NULL;
  TAILQ_FOREACH(a, al, entries)
  if (mutt_addr_is_user(a))
    return true;
  return false;
}

/**
 * user_is_recipient - Is the user a recipient of the message
 * @param e Email to test
 * @retval enum Character index into the `$to_chars` config variable
 */
static enum ToChars user_is_recipient(struct Email *e)
{
  if (!e || !e->env)
    return FLAG_CHAR_TO_NOT_IN_THE_LIST;

  struct Envelope *env = e->env;

  if (!e->recip_valid)
  {
    e->recip_valid = true;

    if (mutt_addr_is_user(TAILQ_FIRST(&env->from)))
    {
      e->recipient = FLAG_CHAR_TO_ORIGINATOR;
    }
    else if (user_in_addr(&env->to))
    {
      if (TAILQ_NEXT(TAILQ_FIRST(&env->to), entries) || !TAILQ_EMPTY(&env->cc))
        e->recipient = FLAG_CHAR_TO_TO; /* non-unique recipient */
      else
        e->recipient = FLAG_CHAR_TO_UNIQUE; /* unique recipient */
    }
    else if (user_in_addr(&env->cc))
    {
      e->recipient = FLAG_CHAR_TO_CC;
    }
    else if (check_for_mailing_list(&env->to, NULL, NULL, 0))
    {
      e->recipient = FLAG_CHAR_TO_SUBSCRIBED_LIST;
    }
    else if (check_for_mailing_list(&env->cc, NULL, NULL, 0))
    {
      e->recipient = FLAG_CHAR_TO_SUBSCRIBED_LIST;
    }
    else if (user_in_addr(&env->reply_to))
    {
      e->recipient = FLAG_CHAR_TO_REPLY_TO;
    }
    else
    {
      e->recipient = FLAG_CHAR_TO_NOT_IN_THE_LIST;
    }
  }

  return e->recipient;
}

/**
 * thread_is_new - Does the email thread contain any new emails?
 * @param e Email
 * @retval true Thread contains new mail
 */
static bool thread_is_new(struct Email *e)
{
  return e->collapsed && (e->num_hidden > 1) && (mutt_thread_contains_unread(e) == 1);
}

/**
 * thread_is_old - Does the email thread contain any unread emails?
 * @param e Email
 * @retval true Thread contains unread mail
 */
static bool thread_is_old(struct Email *e)
{
  return e->collapsed && (e->num_hidden > 1) && (mutt_thread_contains_unread(e) == 2);
}

/**
 * index_format_str - Format a string for the index list - Implements ::format_t - @ingroup expando_api
 *
 * | Expando | Description
 * | :------ | :----------------------------------------------------------------
 * | \%a     | Address of the author
 * | \%A     | Reply-to address (if present; otherwise: address of author)
 * | \%b     | Filename of the original message folder (think mailbox)
 * | \%B     | The list to which the email was sent, or else the folder name (%b)
 * | \%C     | Current message number
 * | \%c     | Number of characters (bytes) in the body of the message
 * | \%cr    | Number of characters (bytes) in the message, including header
 * | \%D     | Date and time of message using `$date_format` and local timezone
 * | \%d     | Date and time of message using `$date_format` and sender's timezone
 * | \%e     | Current message number in thread
 * | \%E     | Number of messages in current thread
 * | \%Fp    | Like %F, but plain. No contextual formatting is applied to recipient name
 * | \%F     | Author name, or recipient name if the message is from you
 * | \%f     | Sender (address + real name), either From: or Return-Path:
 * | \%Gx    | Individual message tag (e.g. notmuch tags/imap flags)
 * | \%g     | Message tags (e.g. notmuch tags/imap flags)
 * | \%H     | Spam attribute(s) of this message
 * | \%I     | Initials of author
 * | \%i     | Message-id of the current message
 * | \%J     | Message tags (if present, tree unfolded, and != parent's tags)
 * | \%K     | The list to which the email was sent (if any; otherwise: empty)
 * | \%L     | Like %F, except 'lists' are displayed first
 * | \%l     | Number of lines in the message
 * | \%M     | Number of hidden messages if the thread is collapsed
 * | \%m     | Total number of message in the mailbox
 * | \%n     | Author's real name (or address if missing)
 * | \%N     | Message score
 * | \%O     | Like %L, except using address instead of name
 * | \%P     | Progress indicator for the built-in pager (how much of the file has been displayed)
 * | \%q     | Newsgroup name (if compiled with NNTP support)
 * | \%R     | Comma separated list of Cc: recipients
 * | \%r     | Comma separated list of To: recipients
 * | \%S     | Single character status of the message (N/O/D/d/!/r/-)
 * | \%s     | Subject of the message
 * | \%t     | 'To:' field (recipients)
 * | \%T     | The appropriate character from the `$to_chars` string
 * | \%u     | User (login) name of the author
 * | \%v     | First name of the author, or the recipient if the message is from you
 * | \%W     | Name of organization of author ('Organization:' field)
 * | \%x     | 'X-Comment-To:' field (if present and compiled with NNTP support)
 * | \%X     | Number of MIME attachments
 * | \%y     | 'X-Label:' field (if present)
 * | \%Y     | 'X-Label:' field (if present, tree unfolded, and != parent's x-label)
 * | \%zc    | Message crypto flags
 * | \%zs    | Message status flags
 * | \%zt    | Message tag flags
 * | \%Z     | Combined message flags
 * | \%\@name\@ | Insert and evaluate format-string from the matching "$index-format-hook" command
 * | \%(fmt) | Date/time when the message was received
 * | \%[fmt] | Message date/time converted to the local time zone
 * | \%{fmt} | Message date/time converted to sender's time zone
 */
static const char *index_format_str(char *buf, size_t buflen, size_t col, int cols,
                                    char op, const char *src, const char *prec,
                                    const char *if_str, const char *else_str,
                                    intptr_t data, MuttFormatFlags flags)
{
  struct HdrFormatInfo *hfi = (struct HdrFormatInfo *) data;
  char fmt[128], tmp[1024];
  char *p = NULL, *tags = NULL;
  bool optional = (flags & MUTT_FORMAT_OPTIONAL);
  const bool threads = mutt_using_threads();
  int is_index = (flags & MUTT_FORMAT_INDEX);
  size_t colorlen;

  struct Email *e = hfi->email;
  size_t msg_in_pager = hfi->msg_in_pager;
  struct Mailbox *m = hfi->mailbox;

  if (!e || !e->env)
    return src;

  const struct Address *reply_to = TAILQ_FIRST(&e->env->reply_to);
  const struct Address *from = TAILQ_FIRST(&e->env->from);
  const struct Address *to = TAILQ_FIRST(&e->env->to);
  const struct Address *cc = TAILQ_FIRST(&e->env->cc);

  const struct MbTable *c_crypt_chars = cs_subset_mbtable(NeoMutt->sub, "crypt_chars");
  const struct MbTable *c_flag_chars = cs_subset_mbtable(NeoMutt->sub, "flag_chars");
  const struct MbTable *c_to_chars = cs_subset_mbtable(NeoMutt->sub, "to_chars");
  const char *const c_date_format = cs_subset_string(NeoMutt->sub, "date_format");

  buf[0] = '\0';
  switch (op)
  {
    case 'A':
    case 'I':
      if (op == 'A')
      {
        if (reply_to && reply_to->mailbox)
        {
          colorlen = add_index_color(buf, buflen, flags, MT_COLOR_INDEX_AUTHOR);
          mutt_format_s(buf + colorlen, buflen - colorlen, prec,
                        mutt_addr_for_display(reply_to));
          add_index_color(buf + colorlen, buflen - colorlen, flags, MT_COLOR_INDEX);
          break;
        }
      }
      else
      {
        if (mutt_mb_get_initials(mutt_get_name(from), tmp, sizeof(tmp)))
        {
          colorlen = add_index_color(buf, buflen, flags, MT_COLOR_INDEX_AUTHOR);
          mutt_format_s(buf + colorlen, buflen - colorlen, prec, tmp);
          add_index_color(buf + colorlen, buflen - colorlen, flags, MT_COLOR_INDEX);
          break;
        }
      }
      FALLTHROUGH;

    case 'a':
      colorlen = add_index_color(buf, buflen, flags, MT_COLOR_INDEX_AUTHOR);
      if (from && from->mailbox)
      {
        mutt_format_s(buf + colorlen, buflen - colorlen, prec, mutt_addr_for_display(from));
      }
      else
      {
        mutt_format_s(buf + colorlen, buflen - colorlen, prec, "");
      }
      add_index_color(buf + colorlen, buflen - colorlen, flags, MT_COLOR_INDEX);
      break;

    case 'B':
    case 'K':
      if (first_mailing_list(buf, buflen, &e->env->to) ||
          first_mailing_list(buf, buflen, &e->env->cc))
      {
        mutt_str_copy(tmp, buf, sizeof(tmp));
        mutt_format_s(buf, buflen, prec, tmp);
      }
      else if (optional)
      {
        optional = false;
      }
      break;

    case 'b':
      if (m)
      {
        p = strrchr(mailbox_path(m), '/');
#ifdef USE_NOTMUCH
        if (m->type == MUTT_NOTMUCH)
        {
          char *rel_path = nm_email_get_folder_rel_db(m, e);
          if (rel_path)
            p = rel_path;
        }
#endif

        if (p)
          mutt_str_copy(buf, p + 1, buflen);
        else
          mutt_str_copy(buf, mailbox_path(m), buflen);
      }
      else
      {
        mutt_str_copy(buf, "(null)", buflen);
      }
      mutt_str_copy(tmp, buf, sizeof(tmp));
      mutt_format_s(buf, buflen, prec, tmp);
      break;

    case 'c':
      colorlen = add_index_color(buf, buflen, flags, MT_COLOR_INDEX_SIZE);
      if (src[0] == 'r')
      {
        mutt_str_pretty_size(tmp, sizeof(tmp), email_size(e));
        src++;
      }
      else
      {
        mutt_str_pretty_size(tmp, sizeof(tmp), e->body->length);
      }
      mutt_format_s(buf + colorlen, buflen - colorlen, prec, tmp);
      add_index_color(buf + colorlen, buflen - colorlen, flags, MT_COLOR_INDEX);
      break;

    case 'C':
      colorlen = add_index_color(fmt, sizeof(fmt), flags, MT_COLOR_INDEX_NUMBER);
      snprintf(fmt + colorlen, sizeof(fmt) - colorlen, "%%%sd", prec);
      add_index_color(fmt + colorlen, sizeof(fmt) - colorlen, flags, MT_COLOR_INDEX);
      snprintf(buf, buflen, fmt, e->msgno + 1);
      break;

    case 'd':
    case 'D':
    case '{':
    case '[':
    case '(':
    case '<':
      /* preprocess $date_format to handle %Z */
      {
        const char *cp = NULL;
        time_t now;
        int j = 0;

        if (optional && ((op == '[') || (op == '(')))
        {
          now = mutt_date_now();
          struct tm tm = mutt_date_localtime(now);
          now -= (op == '(') ? e->received : e->date_sent;

          char *is = (char *) prec;
          bool invert = false;
          if (*is == '>')
          {
            invert = true;
            is++;
          }

          while (*is && (*is != '?'))
          {
            int t = strtol(is, &is, 10);
            /* semi-broken (assuming 30 days in all months) */
            switch (*(is++))
            {
              case 'y':
                if (t > 1)
                {
                  t--;
                  t *= (60 * 60 * 24 * 365);
                }
                t += ((tm.tm_mon * 60 * 60 * 24 * 30) + (tm.tm_mday * 60 * 60 * 24) +
                      (tm.tm_hour * 60 * 60) + (tm.tm_min * 60) + tm.tm_sec);
                break;

              case 'm':
                if (t > 1)
                {
                  t--;
                  t *= (60 * 60 * 24 * 30);
                }
                t += ((tm.tm_mday * 60 * 60 * 24) + (tm.tm_hour * 60 * 60) +
                      (tm.tm_min * 60) + tm.tm_sec);
                break;

              case 'w':
                if (t > 1)
                {
                  t--;
                  t *= (60 * 60 * 24 * 7);
                }
                t += ((tm.tm_wday * 60 * 60 * 24) + (tm.tm_hour * 60 * 60) +
                      (tm.tm_min * 60) + tm.tm_sec);
                break;

              case 'd':
                if (t > 1)
                {
                  t--;
                  t *= (60 * 60 * 24);
                }
                t += ((tm.tm_hour * 60 * 60) + (tm.tm_min * 60) + tm.tm_sec);
                break;

              case 'H':
                if (t > 1)
                {
                  t--;
                  t *= (60 * 60);
                }
                t += ((tm.tm_min * 60) + tm.tm_sec);
                break;

              case 'M':
                if (t > 1)
                {
                  t--;
                  t *= (60);
                }
                t += (tm.tm_sec);
                break;

              default:
                break;
            }
            j += t;
          }

          if (j < 0)
            j *= -1;

          if (((now > j) || (now < (-1 * j))) ^ invert)
            optional = false;
          break;
        }

        p = buf;

        cp = ((op == 'd') || (op == 'D')) ? (NONULL(c_date_format)) : src;
        bool use_c_locale = false;
        if (*cp == '!')
        {
          use_c_locale = true;
          cp++;
        }

        size_t len = buflen - 1;
        while ((len > 0) &&
               ((((op == 'd') || (op == 'D')) && *cp) ||
                ((op == '{') && (*cp != '}')) || ((op == '[') && (*cp != ']')) ||
                ((op == '(') && (*cp != ')')) || ((op == '<') && (*cp != '>'))))
        {
          if (*cp == '%')
          {
            cp++;
            if (((*cp == 'Z') || (*cp == 'z')) && ((op == 'd') || (op == '{')))
            {
              if (len >= 5)
              {
                sprintf(p, "%c%02u%02u", e->zoccident ? '-' : '+', e->zhours, e->zminutes);
                p += 5;
                len -= 5;
              }
              else
              {
                break; /* not enough space left */
              }
            }
            else
            {
              if (len >= 2)
              {
                *p++ = '%';
                *p++ = *cp;
                len -= 2;
              }
              else
              {
                break; /* not enough space */
              }
            }
            cp++;
          }
          else
          {
            *p++ = *cp++;
            len--;
          }
        }
        *p = '\0';

        struct tm tm = { 0 };
        if ((op == '[') || (op == 'D'))
        {
          tm = mutt_date_localtime(e->date_sent);
        }
        else if (op == '(')
        {
          tm = mutt_date_localtime(e->received);
        }
        else if (op == '<')
        {
          tm = mutt_date_localtime(mutt_date_now());
        }
        else
        {
          /* restore sender's time zone */
          now = e->date_sent;
          if (e->zoccident)
            now -= (e->zhours * 3600 + e->zminutes * 60);
          else
            now += (e->zhours * 3600 + e->zminutes * 60);
          tm = mutt_date_gmtime(now);
        }

        if (use_c_locale)
          strftime_l(tmp, sizeof(tmp), buf, &tm, NeoMutt->time_c_locale);
        else
          strftime(tmp, sizeof(tmp), buf, &tm);

        colorlen = add_index_color(buf, buflen, flags, MT_COLOR_INDEX_DATE);
        mutt_format_s(buf + colorlen, buflen - colorlen, prec, tmp);
        add_index_color(buf + colorlen, buflen - colorlen, flags, MT_COLOR_INDEX);

        if ((len > 0) && (op != 'd') && (op != 'D')) /* Skip ending op */
          src = cp + 1;
        break;
      }

    case 'e':
      snprintf(fmt, sizeof(fmt), "%%%sd", prec);
      snprintf(buf, buflen, fmt, mutt_messages_in_thread(m, e, MIT_POSITION));
      break;

    case 'E':
      if (!optional)
      {
        snprintf(fmt, sizeof(fmt), "%%%sd", prec);
        snprintf(buf, buflen, fmt, mutt_messages_in_thread(m, e, MIT_NUM_MESSAGES));
      }
      else if (mutt_messages_in_thread(m, e, MIT_NUM_MESSAGES) <= 1)
      {
        optional = false;
      }
      break;

    case 'f':
    {
      struct Buffer *tmpbuf = buf_pool_get();
      mutt_addrlist_write(&e->env->from, tmpbuf, true);
      mutt_str_copy(tmp, buf_string(tmpbuf), sizeof(tmp));
      buf_pool_release(&tmpbuf);
      mutt_format_s(buf, buflen, prec, tmp);
      break;
    }

    case 'F':
      if (!optional)
      {
        const bool is_plain = (src[0] == 'p');
        colorlen = add_index_color(buf, buflen, flags, MT_COLOR_INDEX_AUTHOR);
        make_from(e->env, tmp, sizeof(tmp), false,
                  (is_plain ? MUTT_FORMAT_PLAIN : MUTT_FORMAT_NO_FLAGS));
        mutt_format_s(buf + colorlen, buflen - colorlen, prec, tmp);
        add_index_color(buf + colorlen, buflen - colorlen, flags, MT_COLOR_INDEX);

        if (is_plain)
          src++;
      }
      else if (mutt_addr_is_user(from))
      {
        optional = false;
      }
      break;

    case 'g':
      tags = driver_tags_get_transformed(&e->tags);
      if (!optional)
      {
        colorlen = add_index_color(buf, buflen, flags, MT_COLOR_INDEX_TAGS);
        mutt_format_s(buf + colorlen, buflen - colorlen, prec, NONULL(tags));
        add_index_color(buf + colorlen, buflen - colorlen, flags, MT_COLOR_INDEX);
      }
      else if (!tags)
      {
        optional = false;
      }
      FREE(&tags);
      break;

    case 'G':
    {
      char format[3] = { 0 };
      char *tag = NULL;

      if (!optional)
      {
        format[0] = op;
        format[1] = *src;
        format[2] = '\0';

        tag = mutt_hash_find(TagFormats, format);
        if (tag)
        {
          tags = driver_tags_get_transformed_for(&e->tags, tag);
          colorlen = add_index_color(buf, buflen, flags, MT_COLOR_INDEX_TAG);
          mutt_format_s(buf + colorlen, buflen - colorlen, prec, NONULL(tags));
          add_index_color(buf + colorlen, buflen - colorlen, flags, MT_COLOR_INDEX);
          FREE(&tags);
        }
        src++;
      }
      else
      {
        format[0] = op;
        format[1] = *prec;
        format[2] = '\0';

        tag = mutt_hash_find(TagFormats, format);
        if (tag)
        {
          tags = driver_tags_get_transformed_for(&e->tags, tag);
          if (!tags)
            optional = false;
          FREE(&tags);
        }
      }
      break;
    }

    case 'H':
      /* (Hormel) spam score */
      if (optional)
        optional = !buf_is_empty(&e->env->spam);

      mutt_format_s(buf, buflen, prec, buf_string(&e->env->spam));
      break;

    case 'i':
      mutt_format_s(buf, buflen, prec, e->env->message_id ? e->env->message_id : "<no.id>");
      break;

    case 'J':
    {
      bool have_tags = true;
      tags = driver_tags_get_transformed(&e->tags);
      if (tags)
      {
        if (flags & MUTT_FORMAT_TREE)
        {
          char *parent_tags = NULL;
          if (e->thread->prev && e->thread->prev->message)
          {
            parent_tags = driver_tags_get_transformed(&e->thread->prev->message->tags);
          }
          if (!parent_tags && e->thread->parent && e->thread->parent->message)
          {
            parent_tags = driver_tags_get_transformed(
                &e->thread->parent->message->tags);
          }
          if (parent_tags && mutt_istr_equal(tags, parent_tags))
            have_tags = false;
          FREE(&parent_tags);
        }
      }
      else
      {
        have_tags = false;
      }

      if (optional)
        optional = have_tags;

      colorlen = add_index_color(buf, buflen, flags, MT_COLOR_INDEX_TAGS);
      if (have_tags)
        mutt_format_s(buf + colorlen, buflen - colorlen, prec, tags);
      else
        mutt_format_s(buf + colorlen, buflen - colorlen, prec, "");
      add_index_color(buf + colorlen, buflen - colorlen, flags, MT_COLOR_INDEX);
      FREE(&tags);
      break;
    }

    case 'l':
      if (!optional)
      {
        snprintf(fmt, sizeof(fmt), "%%%sd", prec);
        colorlen = add_index_color(buf, buflen, flags, MT_COLOR_INDEX_SIZE);
        snprintf(buf + colorlen, buflen - colorlen, fmt, e->lines);
        add_index_color(buf + colorlen, buflen - colorlen, flags, MT_COLOR_INDEX);
      }
      else if (e->lines <= 0)
      {
        optional = false;
      }
      break;

    case 'L':
      if (!optional)
      {
        colorlen = add_index_color(buf, buflen, flags, MT_COLOR_INDEX_AUTHOR);
        make_from(e->env, tmp, sizeof(tmp), true, flags);
        mutt_format_s(buf + colorlen, buflen - colorlen, prec, tmp);
        add_index_color(buf + colorlen, buflen - colorlen, flags, MT_COLOR_INDEX);
      }
      else if (!check_for_mailing_list(&e->env->to, NULL, NULL, 0) &&
               !check_for_mailing_list(&e->env->cc, NULL, NULL, 0))
      {
        optional = false;
      }
      break;

    case 'm':
      if (m)
      {
        snprintf(fmt, sizeof(fmt), "%%%sd", prec);
        snprintf(buf, buflen, fmt, m->msg_count);
      }
      else
      {
        mutt_str_copy(buf, "(null)", buflen);
      }
      break;

    case 'n':
      colorlen = add_index_color(buf, buflen, flags, MT_COLOR_INDEX_AUTHOR);
      mutt_format_s(buf + colorlen, buflen - colorlen, prec, mutt_get_name(from));
      add_index_color(buf + colorlen, buflen - colorlen, flags, MT_COLOR_INDEX);
      break;

    case 'M':
      snprintf(fmt, sizeof(fmt), "%%%sd", prec);
      if (!optional)
      {
        colorlen = add_index_color(buf, buflen, flags, MT_COLOR_INDEX_COLLAPSED);
        if (threads && is_index && e->collapsed && (e->num_hidden > 1))
        {
          snprintf(buf + colorlen, buflen - colorlen, fmt, e->num_hidden);
          add_index_color(buf, buflen - colorlen, flags, MT_COLOR_INDEX);
        }
        else if (is_index && threads)
        {
          mutt_format_s(buf + colorlen, buflen - colorlen, prec, " ");
          add_index_color(buf, buflen - colorlen, flags, MT_COLOR_INDEX);
        }
        else
        {
          *buf = '\0';
        }
      }
      else
      {
        if (!(threads && is_index && e->collapsed && (e->num_hidden > 1)))
          optional = false;
      }
      break;

    case 'N':
      if (!optional)
      {
        snprintf(fmt, sizeof(fmt), "%%%sd", prec);
        snprintf(buf, buflen, fmt, e->score);
      }
      else
      {
        if (e->score == 0)
          optional = false;
      }
      break;

    case 'O':
      if (!optional)
      {
        make_from_addr(e->env, tmp, sizeof(tmp), true);
        const bool c_save_address = cs_subset_bool(NeoMutt->sub, "save_address");
        if (!c_save_address && (p = strpbrk(tmp, "%@")))
          *p = '\0';
        mutt_format_s(buf, buflen, prec, tmp);
      }
      else if (!check_for_mailing_list_addr(&e->env->to, NULL, 0) &&
               !check_for_mailing_list_addr(&e->env->cc, NULL, 0))
      {
        optional = false;
      }
      break;

    case 'P':
      mutt_str_copy(buf, hfi->pager_progress, buflen);
      break;

#ifdef USE_NNTP
    case 'q':
      mutt_format_s(buf, buflen, prec, e->env->newsgroups ? e->env->newsgroups : "");
      break;
#endif

    case 'r':
    {
      struct Buffer *tmpbuf = buf_pool_get();
      mutt_addrlist_write(&e->env->to, tmpbuf, true);
      mutt_str_copy(tmp, buf_string(tmpbuf), sizeof(tmp));
      buf_pool_release(&tmpbuf);
      if (optional && (tmp[0] == '\0'))
        optional = false;
      mutt_format_s(buf, buflen, prec, tmp);
      break;
    }

    case 'R':
    {
      struct Buffer *tmpbuf = buf_pool_get();
      mutt_addrlist_write(&e->env->cc, tmpbuf, true);
      mutt_str_copy(tmp, buf_string(tmpbuf), sizeof(tmp));
      buf_pool_release(&tmpbuf);
      if (optional && (tmp[0] == '\0'))
        optional = false;
      mutt_format_s(buf, buflen, prec, tmp);
      break;
    }

    case 's':
    {
      subjrx_apply_mods(e->env);
      char *subj = NULL;
      if (e->env->disp_subj)
        subj = e->env->disp_subj;
      else
        subj = e->env->subject;
      if (flags & MUTT_FORMAT_TREE && !e->collapsed)
      {
        if (flags & MUTT_FORMAT_FORCESUBJ)
        {
          colorlen = add_index_color(buf, buflen, flags, MT_COLOR_INDEX_SUBJECT);
          mutt_format_s(buf + colorlen, buflen - colorlen, "", NONULL(subj));
          add_index_color(buf + colorlen, buflen - colorlen, flags, MT_COLOR_INDEX);
          snprintf(tmp, sizeof(tmp), "%s%s", e->tree, buf);
          mutt_format_s_tree(buf, buflen, prec, tmp);
        }
        else
        {
          mutt_format_s_tree(buf, buflen, prec, e->tree);
        }
      }
      else
      {
        colorlen = add_index_color(buf, buflen, flags, MT_COLOR_INDEX_SUBJECT);
        mutt_format_s(buf + colorlen, buflen - colorlen, prec, NONULL(subj));
        add_index_color(buf + colorlen, buflen - colorlen, flags, MT_COLOR_INDEX);
      }
      break;
    }

    case 'S':
    {
      const char *wch = NULL;
      if (e->deleted)
        wch = mbtable_get_nth_wchar(c_flag_chars, FLAG_CHAR_DELETED);
      else if (e->attach_del)
        wch = mbtable_get_nth_wchar(c_flag_chars, FLAG_CHAR_DELETED_ATTACH);
      else if (e->tagged)
        wch = mbtable_get_nth_wchar(c_flag_chars, FLAG_CHAR_TAGGED);
      else if (e->flagged)
        wch = mbtable_get_nth_wchar(c_flag_chars, FLAG_CHAR_IMPORTANT);
      else if (e->replied)
        wch = mbtable_get_nth_wchar(c_flag_chars, FLAG_CHAR_REPLIED);
      else if (e->read && (msg_in_pager != e->msgno))
        wch = mbtable_get_nth_wchar(c_flag_chars, FLAG_CHAR_SEMPTY);
      else if (e->old)
        wch = mbtable_get_nth_wchar(c_flag_chars, FLAG_CHAR_OLD);
      else
        wch = mbtable_get_nth_wchar(c_flag_chars, FLAG_CHAR_NEW);

      snprintf(tmp, sizeof(tmp), "%s", wch);
      colorlen = add_index_color(buf, buflen, flags, MT_COLOR_INDEX_FLAGS);
      mutt_format_s(buf + colorlen, buflen - colorlen, prec, tmp);
      add_index_color(buf + colorlen, buflen - colorlen, flags, MT_COLOR_INDEX);
      break;
    }

    case 't':
      tmp[0] = '\0';
      if (!check_for_mailing_list(&e->env->to, "To ", tmp, sizeof(tmp)) &&
          !check_for_mailing_list(&e->env->cc, "Cc ", tmp, sizeof(tmp)))
      {
        if (to)
          snprintf(tmp, sizeof(tmp), "To %s", mutt_get_name(to));
        else if (cc)
          snprintf(tmp, sizeof(tmp), "Cc %s", mutt_get_name(cc));
      }
      mutt_format_s(buf, buflen, prec, tmp);
      break;

    case 'T':
    {
      int i;
      snprintf(fmt, sizeof(fmt), "%%%ss", prec);
      snprintf(buf, buflen, fmt,
               (c_to_chars && ((i = user_is_recipient(e))) < c_to_chars->len) ?
                   c_to_chars->chars[i] :
                   " ");
      break;
    }

    case 'u':
      if (from && from->mailbox)
      {
        mutt_str_copy(tmp, mutt_addr_for_display(from), sizeof(tmp));
        p = strpbrk(tmp, "%@");
        if (p)
          *p = '\0';
      }
      else
      {
        tmp[0] = '\0';
      }
      mutt_format_s(buf, buflen, prec, tmp);
      break;

    case 'v':
      if (mutt_addr_is_user(from))
      {
        if (to)
          mutt_format_s(tmp, sizeof(tmp), prec, mutt_get_name(to));
        else if (cc)
          mutt_format_s(tmp, sizeof(tmp), prec, mutt_get_name(cc));
        else
          *tmp = '\0';
      }
      else
      {
        mutt_format_s(tmp, sizeof(tmp), prec, mutt_get_name(from));
      }
      p = strpbrk(tmp, " %@");
      if (p)
        *p = '\0';
      mutt_format_s(buf, buflen, prec, tmp);
      break;

    case 'W':
      if (!optional)
      {
        mutt_format_s(buf, buflen, prec, e->env->organization ? e->env->organization : "");
      }
      else if (!e->env->organization)
      {
        optional = false;
      }
      break;

#ifdef USE_NNTP
    case 'x':
      if (!optional)
      {
        mutt_format_s(buf, buflen, prec, e->env->x_comment_to ? e->env->x_comment_to : "");
      }
      else if (!e->env->x_comment_to)
      {
        optional = false;
      }
      break;
#endif

    case 'X':
    {
      struct Message *msg = mx_msg_open(m, e);
      if (msg)
      {
        int count = mutt_count_body_parts(m, e, msg->fp);
        mx_msg_close(m, &msg);

        /* The recursion allows messages without depth to return 0. */
        if (optional)
          optional = (count != 0);

        snprintf(fmt, sizeof(fmt), "%%%sd", prec);
        snprintf(buf, buflen, fmt, count);
      }
      break;
    }

    case 'y':
      if (optional)
        optional = (e->env->x_label != NULL);

      colorlen = add_index_color(buf, buflen, flags, MT_COLOR_INDEX_LABEL);
      mutt_format_s(buf + colorlen, buflen - colorlen, prec, NONULL(e->env->x_label));
      add_index_color(buf + colorlen, buflen - colorlen, flags, MT_COLOR_INDEX);
      break;

    case 'Y':
    {
      bool label = true;
      if (e->env->x_label)
      {
        struct Email *e_tmp = NULL;
        if (flags & MUTT_FORMAT_TREE && (e->thread->prev && e->thread->prev->message &&
                                         e->thread->prev->message->env->x_label))
        {
          e_tmp = e->thread->prev->message;
        }
        else if (flags & MUTT_FORMAT_TREE &&
                 (e->thread->parent && e->thread->parent->message &&
                  e->thread->parent->message->env->x_label))
        {
          e_tmp = e->thread->parent->message;
        }
        if (e_tmp && mutt_istr_equal(e->env->x_label, e_tmp->env->x_label))
          label = false;
      }
      else
      {
        label = false;
      }

      if (optional)
        optional = label;

      colorlen = add_index_color(buf, buflen, flags, MT_COLOR_INDEX_LABEL);
      if (label)
        mutt_format_s(buf + colorlen, buflen - colorlen, prec, NONULL(e->env->x_label));
      else
        mutt_format_s(buf + colorlen, buflen - colorlen, prec, "");
      add_index_color(buf + colorlen, buflen - colorlen, flags, MT_COLOR_INDEX);
      break;
    }

    case 'z':
      if (src[0] == 's') /* status: deleted/new/old/replied */
      {
        const char *ch = NULL;
        if (e->deleted)
        {
          ch = mbtable_get_nth_wchar(c_flag_chars, FLAG_CHAR_DELETED);
        }
        else if (e->attach_del)
        {
          ch = mbtable_get_nth_wchar(c_flag_chars, FLAG_CHAR_DELETED_ATTACH);
        }
        else if (threads && thread_is_new(e))
        {
          ch = mbtable_get_nth_wchar(c_flag_chars, FLAG_CHAR_NEW_THREAD);
        }
        else if (threads && thread_is_old(e))
        {
          ch = mbtable_get_nth_wchar(c_flag_chars, FLAG_CHAR_OLD_THREAD);
        }
        else if (e->read && (msg_in_pager != e->msgno))
        {
          if (e->replied)
            ch = mbtable_get_nth_wchar(c_flag_chars, FLAG_CHAR_REPLIED);
          else
            ch = mbtable_get_nth_wchar(c_flag_chars, FLAG_CHAR_ZEMPTY);
        }
        else
        {
          if (e->old)
            ch = mbtable_get_nth_wchar(c_flag_chars, FLAG_CHAR_OLD);
          else
            ch = mbtable_get_nth_wchar(c_flag_chars, FLAG_CHAR_NEW);
        }

        snprintf(tmp, sizeof(tmp), "%s", ch);
        src++;
      }
      else if (src[0] == 'c') /* crypto */
      {
        const char *ch = NULL;
        if ((WithCrypto != 0) && (e->security & SEC_GOODSIGN))
        {
          ch = mbtable_get_nth_wchar(c_crypt_chars, FLAG_CHAR_CRYPT_GOOD_SIGN);
        }
        else if ((WithCrypto != 0) && (e->security & SEC_ENCRYPT))
        {
          ch = mbtable_get_nth_wchar(c_crypt_chars, FLAG_CHAR_CRYPT_ENCRYPTED);
        }
        else if ((WithCrypto != 0) && (e->security & SEC_SIGN))
        {
          ch = mbtable_get_nth_wchar(c_crypt_chars, FLAG_CHAR_CRYPT_SIGNED);
        }
        else if (((WithCrypto & APPLICATION_PGP) != 0) && ((e->security & PGP_KEY) == PGP_KEY))
        {
          ch = mbtable_get_nth_wchar(c_crypt_chars, FLAG_CHAR_CRYPT_CONTAINS_KEY);
        }
        else
        {
          ch = mbtable_get_nth_wchar(c_crypt_chars, FLAG_CHAR_CRYPT_NO_CRYPTO);
        }

        snprintf(tmp, sizeof(tmp), "%s", ch);
        src++;
      }
      else if (src[0] == 't') /* tagged, flagged, recipient */
      {
        const char *ch = NULL;
        if (e->tagged)
          ch = mbtable_get_nth_wchar(c_flag_chars, FLAG_CHAR_TAGGED);
        else if (e->flagged)
          ch = mbtable_get_nth_wchar(c_flag_chars, FLAG_CHAR_IMPORTANT);
        else
          ch = mbtable_get_nth_wchar(c_to_chars, user_is_recipient(e));

        snprintf(tmp, sizeof(tmp), "%s", ch);
        src++;
      }
      else /* fallthrough */
      {
        break;
      }

      colorlen = add_index_color(buf, buflen, flags, MT_COLOR_INDEX_FLAGS);
      mutt_format_s(buf + colorlen, buflen - colorlen, prec, tmp);
      add_index_color(buf + colorlen, buflen - colorlen, flags, MT_COLOR_INDEX);
      break;

    case 'Z':
    {
      /* New/Old for threads; replied; New/Old for messages */
      const char *first = NULL;
      if (threads && thread_is_new(e))
      {
        first = mbtable_get_nth_wchar(c_flag_chars, FLAG_CHAR_NEW_THREAD);
      }
      else if (threads && thread_is_old(e))
      {
        first = mbtable_get_nth_wchar(c_flag_chars, FLAG_CHAR_OLD_THREAD);
      }
      else if (e->read && (msg_in_pager != e->msgno))
      {
        if (e->replied)
          first = mbtable_get_nth_wchar(c_flag_chars, FLAG_CHAR_REPLIED);
        else
          first = mbtable_get_nth_wchar(c_flag_chars, FLAG_CHAR_ZEMPTY);
      }
      else
      {
        if (e->old)
          first = mbtable_get_nth_wchar(c_flag_chars, FLAG_CHAR_OLD);
        else
          first = mbtable_get_nth_wchar(c_flag_chars, FLAG_CHAR_NEW);
      }

      /* Marked for deletion; deleted attachments; crypto */
      const char *second = NULL;
      if (e->deleted)
        second = mbtable_get_nth_wchar(c_flag_chars, FLAG_CHAR_DELETED);
      else if (e->attach_del)
        second = mbtable_get_nth_wchar(c_flag_chars, FLAG_CHAR_DELETED_ATTACH);
      else if ((WithCrypto != 0) && (e->security & SEC_GOODSIGN))
        second = mbtable_get_nth_wchar(c_crypt_chars, FLAG_CHAR_CRYPT_GOOD_SIGN);
      else if ((WithCrypto != 0) && (e->security & SEC_ENCRYPT))
        second = mbtable_get_nth_wchar(c_crypt_chars, FLAG_CHAR_CRYPT_ENCRYPTED);
      else if ((WithCrypto != 0) && (e->security & SEC_SIGN))
        second = mbtable_get_nth_wchar(c_crypt_chars, FLAG_CHAR_CRYPT_SIGNED);
      else if (((WithCrypto & APPLICATION_PGP) != 0) && (e->security & PGP_KEY))
        second = mbtable_get_nth_wchar(c_crypt_chars, FLAG_CHAR_CRYPT_CONTAINS_KEY);
      else
        second = mbtable_get_nth_wchar(c_crypt_chars, FLAG_CHAR_CRYPT_NO_CRYPTO);

      /* Tagged, flagged and recipient flag */
      const char *third = NULL;
      if (e->tagged)
        third = mbtable_get_nth_wchar(c_flag_chars, FLAG_CHAR_TAGGED);
      else if (e->flagged)
        third = mbtable_get_nth_wchar(c_flag_chars, FLAG_CHAR_IMPORTANT);
      else
        third = mbtable_get_nth_wchar(c_to_chars, user_is_recipient(e));

      snprintf(tmp, sizeof(tmp), "%s%s%s", first, second, third);
    }

      colorlen = add_index_color(buf, buflen, flags, MT_COLOR_INDEX_FLAGS);
      mutt_format_s(buf + colorlen, buflen - colorlen, prec, tmp);
      add_index_color(buf + colorlen, buflen - colorlen, flags, MT_COLOR_INDEX);
      break;

    case '@':
    {
      if (!m)
        break;

      const char *end = src;
      static unsigned char recurse = 0;

      while ((*end != '\0') && (*end != '@'))
        end++;
      if ((*end == '@') && (recurse < 20))
      {
        recurse++;
        mutt_strn_copy(tmp, src, end - src, sizeof(tmp));
        mutt_expando_format(tmp, sizeof(tmp), col, cols,
                            NONULL(mutt_idxfmt_hook(tmp, m, e)),
                            index_format_str, data, flags);
        mutt_format_s_x(buf, buflen, prec, tmp, true);
        recurse--;

        src = end + 1;
        break;
      }
    }
      FALLTHROUGH;

    default:
      snprintf(buf, buflen, "%%%s%c", prec, op);
      break;
  }

  if (optional)
  {
    mutt_expando_format(buf, buflen, col, cols, if_str, index_format_str, data, flags);
  }
  else if (flags & MUTT_FORMAT_OPTIONAL)
  {
    mutt_expando_format(buf, buflen, col, cols, else_str, index_format_str, data, flags);
  }

  /* We return the format string, unchanged */
  return src;
}

/**
 * mutt_make_string - Create formatted strings using mailbox expandos
 * @param buf      Buffer for the result
 * @param buflen   Buffer length
 * @param cols     Number of screen columns (OPTIONAL)
 * @param s        printf-line format string
 * @param m        Mailbox
 * @param inpgr    Message shown in the pager
 * @param e        Email
 * @param flags    Flags, see #MuttFormatFlags
 * @param progress Pager progress string
 *
 * @sa index_format_str()
 */
void mutt_make_string(char *buf, size_t buflen, int cols, const char *s,
                      struct Mailbox *m, int inpgr, struct Email *e,
                      MuttFormatFlags flags, const char *progress)
{
  struct HdrFormatInfo hfi = { 0 };

  hfi.email = e;
  hfi.mailbox = m;
  hfi.msg_in_pager = inpgr;
  hfi.pager_progress = progress;

  mutt_expando_format(buf, buflen, 0, cols, s, index_format_str, (intptr_t) &hfi, flags);
}
