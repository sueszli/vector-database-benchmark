/*-
 * Copyright 2021 Vsevolod Stakhov
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define U_CHARSET_IS_UTF8 1
#include <unicode/utypes.h>
#include <unicode/utf8.h>
#include <unicode/uchar.h>
#include <unicode/normalizer2.h>
#include <unicode/schriter.h>
#include <unicode/coll.h>
#include <unicode/translit.h>
#include <utility>
#include <tuple>
#include <string>
#include <limits>
#include <memory>

#include "utf8_util.h"
#include "str_util.h"

#define DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL
#include "doctest/doctest.h"

const char *
rspamd_string_unicode_trim_inplace(const char *str, size_t *len)
{
	const auto *p = str, *end = str + *len;
	auto i = 0;

	while (i < *len) {
		UChar32 uc;
		auto prev_i = i;

		U8_NEXT(p, i, *len, uc);

		if (!u_isUWhiteSpace(uc) && !IS_ZERO_WIDTH_SPACE(uc)) {
			i = prev_i;
			break;
		}
	}

	p += i;
	(*len) -= i;
	i = end - p;
	auto *ret = p;

	if (i > 0) {

		while (i > 0) {
			UChar32 uc;
			auto prev_i = i;

			U8_PREV(p, 0, i, uc);

			if (!u_isUWhiteSpace(uc) && !IS_ZERO_WIDTH_SPACE(uc)) {
				i = prev_i;
				break;
			}
		}

		*len = i;
	}

	return ret;
}

enum rspamd_utf8_normalise_result
rspamd_normalise_unicode_inplace(char *start, size_t *len)
{
	UErrorCode uc_err = U_ZERO_ERROR;
	const auto *nfkc_norm = icu::Normalizer2::getNFKCInstance(uc_err);
	static icu::UnicodeSet zw_spaces{};

	if (!zw_spaces.isFrozen()) {
		/* Add zw spaces to the set */
		zw_spaces.add(0x200B);
		/* TODO: ZW non joiner, it might be used for ligatures, so it should possibly be excluded as well */
		zw_spaces.add(0x200C);
		/* See github issue #4290 for explanation. It seems that the ZWJ has many legit use cases */
		//zw_spaces.add(0x200D);
		zw_spaces.add(0xFEF);
		zw_spaces.add(0x00AD);
		zw_spaces.freeze();
	}

	int ret = RSPAMD_UNICODE_NORM_NORMAL;

	g_assert(U_SUCCESS(uc_err));

	auto uc_string = icu::UnicodeString::fromUTF8(icu::StringPiece(start, *len));
	auto is_normal = nfkc_norm->quickCheck(uc_string, uc_err);

	if (!U_SUCCESS(uc_err)) {
		return RSPAMD_UNICODE_NORM_ERROR;
	}

	/* Filter zero width spaces and push resulting string back */
	const auto filter_zw_spaces_and_push_back = [&](const icu::UnicodeString &input) -> size_t {
		icu::StringCharacterIterator it{input};
		size_t i = 0;

		while (it.hasNext()) {
			/* libicu is very 'special' if it comes to 'safe' macro */
			if (i >= *len) {
				ret |= RSPAMD_UNICODE_NORM_ERROR;
				break;
			}

			auto uc = it.next32PostInc();

			if (zw_spaces.contains(uc)) {
				ret |= RSPAMD_UNICODE_NORM_ZERO_SPACES;
			}
			else {
				UBool err = 0;

				if (uc == 0xFFFD) {
					ret |= RSPAMD_UNICODE_NORM_UNNORMAL;
				}
				U8_APPEND((uint8_t *) start, i, *len, uc, err);

				if (err) {
					ret |= RSPAMD_UNICODE_NORM_ERROR;
					break;
				}
			}
		}

		return i;
	};

	if (is_normal != UNORM_YES) {
		/* Need to normalise */
		ret |= RSPAMD_UNICODE_NORM_UNNORMAL;

		auto normalised = nfkc_norm->normalize(uc_string, uc_err);

		if (!U_SUCCESS(uc_err)) {
			return RSPAMD_UNICODE_NORM_ERROR;
		}

		*len = filter_zw_spaces_and_push_back(normalised);
	}
	else {
		*len = filter_zw_spaces_and_push_back(uc_string);
	}

	return static_cast<enum rspamd_utf8_normalise_result>(ret);
}

gchar *
rspamd_utf8_transliterate(const gchar *start, gsize len, gsize *target_len)
{
	UErrorCode uc_err = U_ZERO_ERROR;

	static std::unique_ptr<icu::Transliterator> transliterator;

	if (!transliterator) {
		UParseError parse_err;
		static const auto rules = icu::UnicodeString{":: Any-Latin;"
													 ":: [:Nonspacing Mark:] Remove;"
													 ":: [:Punctuation:] Remove;"
													 ":: [:Symbol:] Remove;"
													 ":: [:Format:] Remove;"
													 ":: Latin-ASCII;"
													 ":: Lower();"
													 ":: NULL;"
													 "[:Space Separator:] > ' '"};
		transliterator = std::unique_ptr<icu::Transliterator>(
			icu::Transliterator::createFromRules("RspamdTranslit", rules, UTRANS_FORWARD, parse_err, uc_err));

		if (U_FAILURE(uc_err) || !transliterator) {
			auto context = icu::UnicodeString(parse_err.postContext, sizeof(parse_err.preContext) / sizeof(UChar));
			g_error("fatal error: cannot init libicu transliteration engine: %s, line: %d, offset: %d",
					u_errorName(uc_err), parse_err.line, parse_err.offset);
			abort();
		}
	}

	auto uc_string = icu::UnicodeString::fromUTF8(icu::StringPiece(start, len));
	transliterator->transliterate(uc_string);

	// We assume that all characters are now ascii
	auto dest_len = uc_string.length();
	gchar *dest = (gchar *) g_malloc(dest_len + 1);
	auto sink = icu::CheckedArrayByteSink(dest, dest_len);
	uc_string.toUTF8(sink);

	*target_len = sink.NumberOfBytesWritten();
	dest[*target_len] = '\0';

	return dest;
}

struct rspamd_icu_collate_storage {
	icu::Collator *collator = nullptr;
	rspamd_icu_collate_storage()
	{
		UErrorCode uc_err = U_ZERO_ERROR;
		collator = icu::Collator::createInstance(icu::Locale::getEnglish(), uc_err);

		if (U_FAILURE(uc_err) || collator == nullptr) {
			g_error("fatal error: cannot init libicu collation engine: %s",
					u_errorName(uc_err));
			abort();
		}
		/* Ignore all difference except functional */
		collator->setStrength(icu::Collator::PRIMARY);
	}

	~rspamd_icu_collate_storage()
	{
		if (collator) {
			delete collator;
		}
	}
};

static rspamd_icu_collate_storage collate_storage;

int rspamd_utf8_strcmp_sizes(const char *s1, gsize n1, const char *s2, gsize n2)
{
	if (n1 >= std::numeric_limits<int>::max() || n2 >= std::numeric_limits<int>::max()) {
		/*
		 * It's hard to say what to do here... But libicu wants int, so we fall
		 * back to g_ascii_strcasecmp which can deal with size_t
		 */
		if (n1 == n2) {
			return g_ascii_strncasecmp(s1, s2, n1);
		}
		else {
			return n1 - n2;
		}
	}

	UErrorCode success = U_ZERO_ERROR;
	auto res = collate_storage.collator->compareUTF8({s1, (int) n1}, {s2, (int) n2},
													 success);

	switch (res) {
	case UCOL_EQUAL:
		return 0;
	case UCOL_GREATER:
		return 1;
	case UCOL_LESS:
	default:
		return -1;
	}
}

int rspamd_utf8_strcmp(const char *s1, const char *s2, gsize n)
{
	return rspamd_utf8_strcmp_sizes(s1, n, s2, n);
}

TEST_SUITE("utf8 utils")
{
	TEST_CASE("utf8 normalise")
	{
		std::tuple<const char *, const char *, int> cases[] = {
			{"abc", "abc", RSPAMD_UNICODE_NORM_NORMAL},
			{"тест", "тест", RSPAMD_UNICODE_NORM_NORMAL},
			/* Zero width spaces */
			{"\xE2\x80\x8B"
			 "те"
			 "\xE2\x80\x8B"
			 "ст",
			 "тест", RSPAMD_UNICODE_NORM_ZERO_SPACES},
			/* Special case of diacritic */
			{"13_\u0020\u0308\u0301\u038e\u03ab", "13_ ̈́ΎΫ", RSPAMD_UNICODE_NORM_UNNORMAL},
			// String containing a non-joiner character
			{"س\u200Cت", "ست", RSPAMD_UNICODE_NORM_ZERO_SPACES},
			// String containing a soft hyphen
			{"in\u00ADter\u00ADest\u00ADing", "interesting", RSPAMD_UNICODE_NORM_ZERO_SPACES},
			// String with ligature
			{"ﬁsh", "fish", RSPAMD_UNICODE_NORM_UNNORMAL},
			// String with accented characters and zero-width spaces
			{"café\u200Blatté\u200C", "cafélatté", RSPAMD_UNICODE_NORM_ZERO_SPACES},
			/* Same with zw spaces */
			{"13\u200C_\u0020\u0308\u0301\u038e\u03ab", "13_ ̈́ΎΫ",
			 RSPAMD_UNICODE_NORM_UNNORMAL | RSPAMD_UNICODE_NORM_ZERO_SPACES},
			/* Buffer overflow case */
			{"u\xC2\xC2\xC2\xC2\xC2\xC2"
			 "abcdef"
			 "abcdef",
			 "u\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD",
			 RSPAMD_UNICODE_NORM_UNNORMAL | RSPAMD_UNICODE_NORM_ERROR},
			// String with a mix of special characters, ligatures, and zero-width spaces
			{"ﬁsh\u200Bcafé\u200C\u200Dlatté\u200D\u00AD", "fishcafé\u200Dlatté\u200D", RSPAMD_UNICODE_NORM_UNNORMAL | RSPAMD_UNICODE_NORM_ZERO_SPACES},
			// Empty string
			{"", "", RSPAMD_UNICODE_NORM_NORMAL},
		};

		for (const auto &c: cases) {
			std::string cpy{std::get<0>(c)};
			auto ns = cpy.size();
			auto res = rspamd_normalise_unicode_inplace(cpy.data(), &ns);
			cpy.resize(ns);
			CHECK(cpy == std::string(std::get<1>(c)));
			CHECK(res == std::get<2>(c));
		}
	}

	TEST_CASE("utf8 trim")
	{
		std::pair<const char *, const char *> cases[] = {
			{" \u200B"
			 "abc ",
			 "abc"},
			{"   ", ""},
			{"   a", "a"},
			{"a   ", "a"},
			{"a a", "a a"},
			{"abc", "abc"},
			{"a ", "a"},
			{"   abc      ", "abc"},
			{" abc ", "abc"},
			{" \xE2\x80\x8B"
			 "a\xE2\x80\x8B"
			 "bc ",
			 "a\xE2\x80\x8B"
			 "bc"},
			{" \xE2\x80\x8B"
			 "abc\xE2\x80\x8B ",
			 "abc"},
			{" \xE2\x80\x8B"
			 "abc \xE2\x80\x8B  ",
			 "abc"},
		};

		for (const auto &c: cases) {
			std::string cpy{c.first};
			auto ns = cpy.size();
			auto *nstart = rspamd_string_unicode_trim_inplace(cpy.data(), &ns);
			std::string res{nstart, ns};
			CHECK(res == std::string{c.second});
		}
	}


	TEST_CASE("utf8 strcmp")
	{
		std::tuple<const char *, const char *, int, int> cases[] = {
			{"abc", "abc", -1, 0},
			{"", "", -1, 0},
			{"aBc", "AbC", -1, 0},
			{"abc", "ab", 2, 0},
			{"теСт", "ТесТ", -1, 0},
			{"теСт", "Тезт", 4, 0},
			{"теСт", "Тезт", -1, 1},
			{"abc", "ABD", -1, -1},
			{"\0a\0", "\0a\1", 2, 0},
			{"\0a\0", "\0b\1", 3, -1},
		};

		for (const auto &c: cases) {
			auto [s1, s2, n, expected] = c;
			if (n == -1) {
				n = MIN(strlen(s1), strlen(s2));
			}
			SUBCASE((std::string("test case: ") + s1 + " <=> " + s2).c_str())
			{
				auto ret = rspamd_utf8_strcmp(s1, s2, n);
				CHECK(ret == expected);
			}
		}
	}

	TEST_CASE("transliterate")
	{
		using namespace std::literals;
		std::tuple<std::string_view, const char *> cases[] = {
			{"abc"sv, "abc"},
			{""sv, ""},
			{"тест"sv, "test"},
			// Diacritic to ascii
			{"Ύ"sv, "y"},
			// Chinese to pinyin
			{"你好"sv, "ni hao"},
			// Japanese to romaji
			{"こんにちは"sv, "konnichiha"},
			// Devanagari to latin
			{"नमस्ते"sv, "namaste"},
			// Arabic to latin
			{"مرحبا"sv, "mrhba"},
			// Remove of punctuation
			{"a.b.c"sv, "abc"},
			// Lowercase
			{"ABC"sv, "abc"},
			// Remove zero-width spaces
			{"\xE2\x80\x8B"
			 "abc\xE2\x80\x8B"
			 "def"sv,
			 "abcdef"},
		};

		for (const auto &c: cases) {
			auto [s1, s2] = c;
			SUBCASE((std::string("test case: ") + std::string(s1) + " => " + s2).c_str())
			{
				gsize tlen;
				auto *ret = rspamd_utf8_transliterate(s1.data(), s1.length(), &tlen);
				CHECK(tlen == strlen(s2));
				CHECK(strcmp(s2, ret) == 0);
			}
		}
	}
}