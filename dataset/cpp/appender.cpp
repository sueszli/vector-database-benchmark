// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "appender.h"
#include "juniperdebug.h"
#define _NEED_SUMMARY_CONFIG_IMPL
#include "SummaryConfig.h"

namespace juniper {

void
Appender::append(std::vector<char> & s, char c)
{
    JD_INVAR(JD_INPUT, c != 0, return,\
             LOG(warning, "Document source contained 0-bytes"));
    // eliminate separators:
    if (_sumconf->separator(c)) {
        return;
    }

    // eliminate multiple space characters
    if (!_preserve_white_space) {
        if (c > 0 && isspace(c)) {
            if (_last_was_space) {
                return;
            } else {
                _last_was_space = true;
            }
            c = ' '; // Never output newline or tab
        } else {
            _last_was_space = false;
        }
    }

    bool handled_as_markup;
    if (_escape_markup) {
        handled_as_markup = true;
        switch (c) {
        case '<':
            s.push_back('&');
            s.push_back('l');
            s.push_back('t');
            s.push_back(';');
            break;
        case '>':
            s.push_back('&');
            s.push_back('g');
            s.push_back('t');
            s.push_back(';');
            break;
        case '"':
            s.push_back('&');
            s.push_back('q');
            s.push_back('u');
            s.push_back('o');
            s.push_back('t');
            s.push_back(';');
            break;
        case '&':
            s.push_back('&');
            s.push_back('a');
            s.push_back('m');
            s.push_back('p');
            s.push_back(';');
            break;
        case '\'':
            s.push_back('&');
            s.push_back('#');
            s.push_back('3');
            s.push_back('9');
            s.push_back(';');
            break;
        default:
            handled_as_markup = false;
            break;
        }
        if (handled_as_markup) {
            _char_len++;
        }
    } else {
        handled_as_markup = false;
    }

    if (!handled_as_markup) {
        s.push_back(c);
        /** If at start of an UTF8 character (both highest bits or none of them set)
         *  another char is accumulated..
         */
        if (!(c & 0x80) || (c & 0x40) ) {
            _char_len++;
        }
    }
}

Appender::Appender(const SummaryConfig *sumconf)
    : _sumconf(sumconf),
      _escape_markup(false),
      _preserve_white_space(false),
      _last_was_space(false),
      _char_len(0)
{
    ConfigFlag esc_conf = _sumconf->escape_markup();

    switch (esc_conf) {
    case CF_OFF:
        _escape_markup = false;
        break;
    case CF_ON:
        _escape_markup = true;
        break;
    case CF_AUTO:
        _escape_markup = (_sumconf->highlight_on()[0] == '<' ||
                          _sumconf->highlight_off()[0] == '<' ||
                          _sumconf->dots()[0] == '<');
        break;
    }

    if (_sumconf->preserve_white_space() == CF_ON) {
        _preserve_white_space = true;
    }
}

void
Appender::append(std::vector<char>& s, const char* ds, int length) {
    for (int i = 0; i < length; i++) {
        append(s, ds[i]);
    }
}

}
