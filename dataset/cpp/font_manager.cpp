// font_manager.cpp
/*
  neogfx C++ App/Game Engine
  Copyright (c) 2015, 2020 Leigh Johnston.  All Rights Reserved.
  
  This program is free software: you can redistribute it and / or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  
  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <neogfx/neogfx.hpp>
#include <filesystem>
#include <neolib/core/string_utils.hpp>
#include <neolib/core/string_utf.hpp>
#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_LCD_FILTER_H
#include FT_GLYPH_H
#include FT_OUTLINE_H
#include FT_BITMAP_H
#ifdef u8
#undef u8
#include <harfbuzz\hb.h>
#include <harfbuzz\hb-ft.h>
#include <harfbuzz\hb-ucdn\ucdn.h>
#define u8
#else
#include <harfbuzz\hb.h>
#include <harfbuzz\hb-ot.h>
#endif
#ifdef _WIN32
#include <Shlobj.h>
#endif
#include <neolib/file/file.hpp>
#include <neogfx/app/i_app.hpp>
#include <neogfx/gfx/i_rendering_engine.hpp>
#include <neogfx/gfx/i_graphics_context.hpp>
#include <neogfx/gfx/text/font_manager.hpp>
#include <neogfx/gfx/text/text_category_map.hpp>
#include <neogfx/gfx/text/glyph_text.ipp>
#include "../../gfx/text/native/native_font_face.hpp"
#include "../../gfx/text/native/native_font.hpp"

template <>
neogfx::i_font_manager& services::start_service<neogfx::i_font_manager>()
{
    return services::service<neogfx::i_rendering_engine>().font_manager();
}

namespace neogfx
{
    neolib::small_cookie item_cookie(const font_manager::id_cache_entry& aEntry)
    {
        return aEntry.id();
    }

    namespace detail
    {
        namespace platform_specific
        {
            optional<font_info> default_system_font_info(system_font_role aRole)
            {
#ifdef WIN32
#if 1 // Has Microsoft (tm) changed their mind on this? (See VS2019 font usage)
                if (service<i_font_manager>().has_font("Segoe UI", "Regular") && (aRole == system_font_role::Caption || aRole == system_font_role::Menu || aRole == system_font_role::StatusBar))
                    return font_info{ "Segoe UI", "Regular", 9 };
#endif
                if (aRole == system_font_role::Widget)
                {
                    std::wstring defaultFontFaceName = L"Microsoft Sans Serif";
                    HKEY hkeyDefaultFont;
                    if (::RegOpenKeyEx(HKEY_LOCAL_MACHINE, L"Software\\Microsoft\\Windows NT\\CurrentVersion\\FontSubstitutes",
                        0, KEY_READ, &hkeyDefaultFont) == ERROR_SUCCESS)
                    {
                        DWORD dwType;
                        wchar_t byteBuffer[LF_FACESIZE + 1];
                        DWORD dwBufferSize = sizeof(byteBuffer);
                        if (RegQueryValueEx(hkeyDefaultFont, L"MS Shell Dlg 2", NULL, &dwType,
                            (LPBYTE)&byteBuffer, &dwBufferSize) == ERROR_SUCCESS)
                        {
                            defaultFontFaceName = (LPCTSTR)byteBuffer;
                        }
                        else if (RegQueryValueEx(hkeyDefaultFont, L"MS Shell Dlg", NULL, &dwType,
                            (LPBYTE)&byteBuffer, &dwBufferSize) == ERROR_SUCCESS)
                        {
                            defaultFontFaceName = (LPCTSTR)byteBuffer;
                        }
                        ::RegCloseKey(hkeyDefaultFont);
                    }
                    return font_info(neolib::utf16_to_utf8(reinterpret_cast<const char16_t*>(defaultFontFaceName.c_str())), font_style::Normal, 8);
                }
                else
                    return {};
#else
                return {};
#endif
            }

            std::string get_system_font_directory()
            {
#ifdef WIN32
                char szPath[MAX_PATH];
                if (SUCCEEDED(SHGetFolderPathA(NULL, CSIDL_FONTS, NULL, 0, szPath)))
                    return neolib::tidy_path(szPath);
                else
                    throw std::logic_error("neogfx::detail::platform_specific::get_system_font_directory: Error");
#else
                throw std::logic_error("neogfx::detail::platform_specific::get_system_font_directory: Unknown system");
#endif
            }

            std::string get_local_font_directory()
            {
#ifdef WIN32
                char szPath[MAX_PATH];
                if (SUCCEEDED(SHGetFolderPathA(NULL, CSIDL_LOCAL_APPDATA, NULL, 0, szPath)))
                    return neolib::tidy_path(szPath) + "/Microsoft/Windows/Fonts";
                else
                    throw std::logic_error("neogfx::detail::platform_specific::get_local_font_directory: Error");
#else
                throw std::logic_error("neogfx::detail::platform_specific::get_local_font_directory: Unknown system");
#endif
            }

            fallback_font_info default_fallback_font_info()
            {
#ifdef WIN32
                // TODO: Use fallback font info from registry
                return fallback_font_info{ { "Segoe UI Symbol", "Noto Sans CJK JP", "Arial Unicode MS" } };
#else
                throw std::logic_error("neogfx::detail::platform_specific::default_fallback_font_info: Unknown system");
#endif
            }
        }
    }

    fallback_font_info::fallback_font_info(std::vector<string> aFallbackFontFamilies) :
        iFallbackFontFamilies(std::move(aFallbackFontFamilies))
    {
    }

    bool fallback_font_info::has_fallback_for(i_string const& aFontFamilyName) const
    {
        if (iFallbackFontFamilies.empty())
            return false;
        return std::find(iFallbackFontFamilies.begin(), iFallbackFontFamilies.end(), aFontFamilyName) != std::prev(iFallbackFontFamilies.end());
    }

    i_string const& fallback_font_info::fallback_for(i_string const& aFontFamilyName) const
    {
        auto f = std::find(iFallbackFontFamilies.begin(), iFallbackFontFamilies.end(), aFontFamilyName);
        if (f == iFallbackFontFamilies.end())
            return *iFallbackFontFamilies.begin();
        ++f;
        if (f == iFallbackFontFamilies.end())
            throw no_fallback();
        return *f;
    }

    class glyph_text_factory : public i_glyph_text_factory
    {
    public:
        struct cluster
        {
            std::string::size_type from;
            glyph_char::flags_e flags;
        };
        typedef std::vector<cluster> cluster_map_t;
        typedef std::tuple<const char32_t*, const char32_t*, text_direction, bool, hb_script_t> glyph_run;
        typedef std::vector<glyph_run> run_list;
    public:
        glyph_text create_glyph_text() override;
        glyph_text create_glyph_text(font const& aFont) override;
        glyph_text to_glyph_text(i_graphics_context const& aGc, char const* aUtf8Begin, char const* aUtf8End, i_font_selector const& aFontSelector, bool aAlignBaselines = true) override;
        glyph_text to_glyph_text(i_graphics_context const& aGc, char32_t const* aUtf32Begin, char32_t const* aUtf32End, i_font_selector const& aFontSelector, bool aAlignBaselines = true) override;
    };

    class glyph_shapes
    {
    public:
        struct not_using_fallback : std::logic_error { not_using_fallback() : std::logic_error("neogfx::graphics_context::glyph_shapes::not_using_fallback") {} };
    public:
        class glyphs
        {
        public:
            glyphs(const i_graphics_context& aParent, const font& aFont, const glyph_text_factory::glyph_run& aGlyphRun) :
                iParent{ aParent },
                iFont{ static_cast<font_face_handle*>(aFont.native_font_face().handle())->harfbuzzFont },
                iGlyphRun{ aGlyphRun },
                iBuf{ static_cast<font_face_handle*>(aFont.native_font_face().handle())->harfbuzzBuf },
                iGlyphCount{ 0u }
            {
                hb_buffer_set_direction(iBuf, std::get<2>(aGlyphRun) == text_direction::RTL ? HB_DIRECTION_RTL : HB_DIRECTION_LTR);
                hb_buffer_set_script(iBuf, std::get<4>(aGlyphRun));
                hb_buffer_set_cluster_level(iBuf, HB_BUFFER_CLUSTER_LEVEL_CHARACTERS);
                std::vector<std::uint32_t> reversed;
                hb_buffer_add_utf32(iBuf, reinterpret_cast<const std::uint32_t*>(std::get<0>(aGlyphRun)), static_cast<int>(std::get<1>(aGlyphRun) - std::get<0>(aGlyphRun)), 0, static_cast<int>(std::get<1>(aGlyphRun) - std::get<0>(aGlyphRun)));
                scoped_kerning sk{ aFont.kerning() };
                hb_shape(iFont, iBuf, NULL, 0);
                unsigned int glyphCount = 0;
                auto glyphInfo = hb_buffer_get_glyph_infos(iBuf, &glyphCount);
                iGlyphInfo.assign(glyphInfo, glyphInfo + glyphCount);
                auto glyphPos = hb_buffer_get_glyph_positions(iBuf, &glyphCount);
                iGlyphPos.assign(glyphPos, glyphPos + glyphCount);
                iGlyphCount = glyphCount;
            }
            ~glyphs()
            {
                hb_buffer_clear_contents(iBuf);
            }
        public:
            std::uint32_t glyph_count() const
            {
                return iGlyphCount;
            }
            const hb_glyph_info_t& glyph_info(std::uint32_t aIndex) const
            {
                return iGlyphInfo[aIndex];
            }
            const hb_glyph_position_t& glyph_position(std::uint32_t aIndex) const
            {
                return iGlyphPos[aIndex];
            }
            bool needs_fallback_font() const
            {
                for (std::uint32_t i = 0; i < glyph_count(); ++i)
                {
                    auto const tc = get_text_category(service<i_font_manager>().emoji_atlas(), std::next(std::get<0>(iGlyphRun), i), std::get<1>(iGlyphRun));
                    if (glyph_info(i).codepoint == 0 && tc != text_category::Whitespace && tc != text_category::Emoji)
                        return true;
                }
                return false;
            }
        private:
            const i_graphics_context& iParent;
            hb_font_t* iFont;
            const glyph_text_factory::glyph_run& iGlyphRun;
            hb_buffer_t* iBuf;
            std::uint32_t iGlyphCount;
            std::vector<hb_glyph_info_t> iGlyphInfo;
            std::vector<hb_glyph_position_t> iGlyphPos;
        };
        typedef std::list<glyphs> glyphs_list;
        typedef std::vector<std::pair<glyphs_list::const_iterator, std::uint32_t>> result_type;
    public:
        glyph_shapes(const i_graphics_context& aParent, const font& aFont, const glyph_text_factory::glyph_run& aGlyphRun)
        {
            thread_local std::vector<font> fontsTried;
            auto tryFont = aFont;
            fontsTried.push_back(aFont);
            iGlyphsList.emplace_back(glyphs{ aParent, tryFont, aGlyphRun });
            while (iGlyphsList.back().needs_fallback_font())
            {
                if (tryFont.has_fallback() && std::find(fontsTried.begin(), fontsTried.end(), tryFont.fallback()) == fontsTried.end())
                {
                    tryFont = tryFont.fallback();
                    fontsTried.push_back(tryFont);
                    iGlyphsList.emplace_back(glyphs{ aParent, tryFont, aGlyphRun });
                }
                else
                {
                    std::u32string lastResort{ std::get<0>(aGlyphRun), std::get<1>(aGlyphRun) };
                    for (std::uint32_t i = 0; i < iGlyphsList.back().glyph_count(); ++i)
                        if (iGlyphsList.back().glyph_info(i).codepoint == 0)
                            lastResort[iGlyphsList.back().glyph_info(i).cluster] = neolib::INVALID_CHAR32; // replacement character
                    iGlyphsList.emplace_back(glyphs{ aParent, aFont, glyph_text_factory::glyph_run{&lastResort[0], &lastResort[0] + lastResort.size(), std::get<2>(aGlyphRun), std::get<3>(aGlyphRun), std::get<4>(aGlyphRun) } });
                    break;
                }
            }
            fontsTried.clear();
            auto const g = iGlyphsList.begin();
            iResults.reserve(g->glyph_count());
            for (std::uint32_t i = 0; i < g->glyph_count(); ++i)
            {
                auto const& gi = g->glyph_info(i);
                auto glyph_match = [&](glyphs const& aGlyphs, std::uint32_t aIndex) -> bool
                {
                    auto const& hgi = aGlyphs.glyph_info(aIndex);
                    if (hgi.cluster != gi.cluster)
                        return false;
                    auto tc = get_text_category(service<i_font_manager>().emoji_atlas(), std::get<0>(aGlyphRun) + hgi.cluster, std::get<1>(aGlyphRun));
                    return hgi.codepoint != 0 || tc == text_category::Whitespace || tc == text_category::Emoji;
                };
                if (glyph_match(*g, i))
                    iResults.push_back(std::make_pair(g, i));
                else
                {
                    auto next = std::next(g);
                    bool found = false;
                    while (!found && next != iGlyphsList.end())
                    {
                        for (std::uint32_t j = 0; j < next->glyph_count(); ++j)
                        {
                            if (glyph_match(*next, j))
                            {
                                iResults.push_back(std::make_pair(next, j));
                                found = true;
                            }
                        }
                        ++next;
                    }
                }
            }
        }
    public:
        std::uint32_t glyph_count() const
        {
            return static_cast<std::uint32_t>(iResults.size());
        }
        const hb_glyph_info_t& glyph_info(std::uint32_t aIndex) const
        {
            return iResults[aIndex].first->glyph_info(iResults[aIndex].second);
        }
        const hb_glyph_position_t& glyph_position(std::uint32_t aIndex) const
        {
            return iResults[aIndex].first->glyph_position(iResults[aIndex].second);
        }
        bool using_fallback(std::uint32_t aIndex) const
        {
            return iResults[aIndex].first != iGlyphsList.begin();
        }
        std::uint32_t fallback_index(std::uint32_t aIndex) const
        {
            if (!using_fallback(aIndex))
                throw not_using_fallback();
            return static_cast<std::uint32_t>(std::distance(iGlyphsList.begin(), iResults[aIndex].first) - 1);
        }
    private:
        glyphs_list iGlyphsList;
        result_type iResults;
    };

    glyph_text glyph_text_factory::create_glyph_text()
    {
        return *make_ref<glyph_text_content>();
    }

    glyph_text glyph_text_factory::create_glyph_text(font const& aFont)
    {
        return *make_ref<glyph_text_content>(aFont);
    }

    glyph_text glyph_text_factory::to_glyph_text(i_graphics_context const& aGc, char const* aUtf8Begin, char const* aUtf8End, i_font_selector const& aFontSelector, bool aAlignBaselines)
    {
        thread_local cluster_map_t clusterMap;
        clusterMap.clear();

        thread_local std::u32string codePoints;
        codePoints.clear();

        auto& clusterMapRef = clusterMap;
        codePoints = neolib::utf8_to_utf32(std::string_view{ aUtf8Begin, aUtf8End }, [&clusterMapRef](std::string::size_type aFrom, std::u32string::size_type)
        {
            clusterMapRef.push_back(glyph_text_factory::cluster{ aFrom });
        });

        if (codePoints.empty())
            return aFontSelector.select_font(0);

        return to_glyph_text(aGc, codePoints.data(), codePoints.data() + codePoints.size(), font_selector{ [&aFontSelector, &clusterMapRef](std::u32string::size_type aIndex)->font
        {
            return aFontSelector.select_font(clusterMapRef[aIndex].from);
        } }, aAlignBaselines);
    }

    glyph_text glyph_text_factory::to_glyph_text(i_graphics_context const& aGc, char32_t const* aUtf32Begin, char32_t const* aUtf32End, i_font_selector const& aFontSelector, bool aAlignBaselines)
    {
        auto const& emojiAtlas = service<i_font_manager>().emoji_atlas();

        auto refResult = make_ref<glyph_text_content>(aFontSelector.select_font(0));
        auto& result = *refResult;

        if (aUtf32End == aUtf32Begin)
            return result;

        bool hasEmojis = false;

        thread_local std::vector<character_type> textDirections;
        textDirections.clear();

        std::u32string::size_type codePointCount = aUtf32End - aUtf32Begin;

        thread_local std::vector<std::u32string::size_type> clusters;
        clusters.clear();
        for (std::u32string::size_type c = 0; c < codePointCount; ++c)
            clusters.push_back(c);

        thread_local std::u32string adjustedCodepoints;
        adjustedCodepoints.clear();

        if (!aGc.password())
        {
            // Reverse LTR embedded in an RTL line...
            adjustedCodepoints.assign(aUtf32Begin, aUtf32End);
            auto next = adjustedCodepoints.begin();
            while (next != adjustedCodepoints.end())
            {
                auto nextEnd = std::find_if(next, adjustedCodepoints.end(), [&](auto const& ch) { return ch == U'\n' || ch == U'\r'; });
                auto nextLtrRtl = std::find_if(next, nextEnd, [&](auto const& ch)
                    { return get_text_category(emojiAtlas, ch) == text_category::LTR || get_text_category(emojiAtlas, ch) == text_category::RTL; } );
                if (nextLtrRtl != nextEnd && get_text_category(emojiAtlas, *nextLtrRtl) == text_category::RTL)
                {
                    auto nextLtrBit = nextLtrRtl;
                    while (nextLtrBit != nextEnd)
                    {
                        auto from = nextLtrBit;
                        nextLtrBit = std::find_if(nextLtrBit, nextEnd, [&](auto const& ch)
                            { return get_text_category(emojiAtlas, ch) == text_category::LTR || 
                                get_text_category(emojiAtlas, ch) == text_category::Digit; });
                        if (nextLtrBit != nextEnd)
                        {
                            while (nextLtrBit != from && get_text_category(emojiAtlas, *std::prev(nextLtrBit)) != text_category::RTL)
                                --nextLtrBit;
                        }
                        auto nextRtlBit = std::find_if(nextLtrBit, nextEnd, [&](auto const& ch)
                            { return get_text_category(emojiAtlas, ch) == text_category::RTL; });
                        if (nextRtlBit == nextEnd)
                            while (nextLtrBit != nextEnd &&
                                (get_text_category(emojiAtlas, *nextLtrBit) == text_category::Mark ||
                                 get_text_category(emojiAtlas, *nextLtrBit) == text_category::None ||
                                 get_text_category(emojiAtlas, *nextLtrBit) == text_category::Whitespace))
                                ++nextLtrBit;
                        std::reverse(nextLtrBit, nextRtlBit);
                        std::reverse(std::next(clusters.begin(), std::distance(adjustedCodepoints.begin(), nextLtrBit)), 
                            std::next(clusters.begin(), std::distance(adjustedCodepoints.begin(), nextRtlBit)));
                        for (auto& ch : std::ranges::subrange{ nextLtrBit, nextRtlBit })
                            switch (ch) { case '<': ch = '>'; break; case '>': ch = '<'; break; case '(': ch = ')'; break; 
                                case ')': ch = '('; break; case '{': ch = '}'; break; case '}': ch = '{'; break; 
                                case '[': ch = ']'; break; case ']': ch = '['; break; }
                        nextLtrBit = nextRtlBit;
                    }
                }
                next = nextEnd;
                if (next != adjustedCodepoints.end())
                    ++next;
            }
        }
        else
            adjustedCodepoints.assign(codePointCount, neolib::utf8_to_utf32(aGc.password_mask())[0]);

        auto codePoints = &adjustedCodepoints[0];

        thread_local run_list runs;
        runs.clear();

        text_category previousCategory = get_text_category(emojiAtlas, codePoints, codePoints + codePointCount);
        if (aGc.mnemonic_set() && codePoints[0] == static_cast<char32_t>(aGc.mnemonic()) && 
            (codePointCount == 1 || codePoints[1] != static_cast<char32_t>(aGc.mnemonic())))
            previousCategory = text_category::Mnemonic;
        text_direction lineDirection = get_text_direction(emojiAtlas, codePoints, codePoints + codePointCount);
        text_direction previousDirection = lineDirection;
        const char32_t* runStart = &codePoints[0];
        std::u32string::size_type lastCodePointIndex = codePointCount - 1;
        font previousFont = aFontSelector.select_font(0);
        hb_script_t previousScript = hb_unicode_script(static_cast<font_face_handle*>(previousFont.native_font_face().handle())->harfbuzzUnicodeFuncs, codePoints[0]);

        std::deque<std::pair<text_direction, bool>> directionStack;
        const char32_t LRE = U'\u202A';
        const char32_t RLE = U'\u202B';
        const char32_t LRO = U'\u202D';
        const char32_t RLO = U'\u202E';
        const char32_t PDF = U'\u202C';

        for (std::size_t codePointIndex = 0; codePointIndex <= lastCodePointIndex; ++codePointIndex)
        {
            font const currentFont = aFontSelector.select_font(codePointIndex);
            switch (codePoints[codePointIndex])
            {
            case PDF:
                if (!directionStack.empty())
                    directionStack.pop_back();
                break;
            case LRE:
                directionStack.push_back(std::make_pair(text_direction::LTR, false));
                break;
            case RLE:
                directionStack.push_back(std::make_pair(text_direction::RTL, false));
                break;
            case LRO:
                directionStack.push_back(std::make_pair(text_direction::LTR, true));
                break;
            case RLO:
                directionStack.push_back(std::make_pair(text_direction::RTL, true));
                break;
            default:
                break;
            }

            hb_unicode_funcs_t* unicodeFuncs = static_cast<font_face_handle*>(currentFont.native_font_face().handle())->harfbuzzUnicodeFuncs;
            
            text_category currentCategory = get_text_category(emojiAtlas, codePoints + codePointIndex, codePoints + codePointCount);
            
            if (aGc.mnemonic_set() && codePoints[codePointIndex] == static_cast<char32_t>(aGc.mnemonic()) &&
                (codePointCount - 1 == codePointIndex || codePoints[codePointIndex + 1] != static_cast<char32_t>(aGc.mnemonic())))
                currentCategory = text_category::Mnemonic;
            
            bool newLine = (codePoints[codePointIndex] == U'\r' || codePoints[codePointIndex] == U'\n');
            if (newLine)
                lineDirection = get_text_direction(emojiAtlas, codePoints + codePointIndex, codePoints + codePointCount);

            text_direction currentDirection = get_text_direction(emojiAtlas, codePoints + codePointIndex, codePoints + codePointCount, lineDirection, previousDirection);
            
            auto bidi_check = [&directionStack](text_category aCategory, text_direction aDirection)
            {
                if (!directionStack.empty())
                {
                    switch (aCategory)
                    {
                    case text_category::LTR:
                    case text_category::RTL:
                    case text_category::Digit:
                    case text_category::Emoji:
                        if (directionStack.back().second == true)
                            return directionStack.back().first;
                        break;
                    case text_category::Mark:
                    case text_category::None:
                    case text_category::Whitespace:
                    case text_category::Mnemonic:
                        return directionStack.back().first;
                    default:
                        break;
                    }
                }
                return aDirection;
            };
            
            if (!newLine)
                currentDirection = bidi_check(currentCategory, currentDirection);
            else
                currentDirection = text_direction::LTR;
            
            hb_script_t currentScript = hb_unicode_script(unicodeFuncs, codePoints[codePointIndex]);
            if (currentScript == HB_SCRIPT_COMMON || currentScript == HB_SCRIPT_INHERITED)
                currentScript = previousScript;

            bool newRun =
                previousFont != currentFont ||
                currentCategory == text_category::Mnemonic ||
                previousCategory == text_category::Mnemonic ||
                previousDirection != currentDirection;

            textDirections.push_back(character_type{ currentCategory, currentDirection });
            if (currentCategory == text_category::Emoji)
                hasEmojis = true;
            if (newRun && codePointIndex > 0)
            {
                runs.push_back(std::make_tuple(runStart, &codePoints[codePointIndex], previousDirection, previousCategory == text_category::Mnemonic, previousScript));
                runStart = &codePoints[codePointIndex];
            }
            previousDirection = currentDirection;
            previousCategory = currentCategory;
            previousScript = currentScript;
            if (codePointIndex == lastCodePointIndex)
                runs.push_back(std::make_tuple(runStart, &codePoints[codePointIndex + 1], previousDirection, previousCategory == text_category::Mnemonic, previousScript));
            previousFont = currentFont;
        }

        float lineStart = 0.0f;
        vec2f previousAdvance = {};
        quadf_2d previousCell = {};

        for (std::size_t i = 0; i < runs.size(); ++i)
        {
            if (std::get<3>(runs[i]))
                continue;
            
            bool drawMnemonic = (i > 0 && std::get<3>(runs[i - 1]));
            std::string::size_type sourceClusterRunStart = std::get<0>(runs[i]) - &codePoints[0];
            glyph_shapes shapes{ aGc, aFontSelector.select_font(sourceClusterRunStart), runs[i] };

            for (std::uint32_t j = 0; j < shapes.glyph_count(); ++j)
            {
                std::u32string::size_type startCluster = shapes.glyph_info(j).cluster;
                std::u32string::size_type endCluster;
                if (std::get<2>(runs[i]) != text_direction::RTL)
                {
                    std::uint32_t k = j + 1;
                    while (k < shapes.glyph_count() && shapes.glyph_info(k).cluster == startCluster)
                        ++k;
                    endCluster = (k < shapes.glyph_count() ? shapes.glyph_info(k).cluster : startCluster + 1);
                }
                else
                {
                    std::uint32_t k = j;
                    while (k > 0 && shapes.glyph_info(k).cluster == startCluster)
                        --k;
                    endCluster = (shapes.glyph_info(k).cluster != startCluster ? shapes.glyph_info(k).cluster : startCluster + 1);
                }
                startCluster += (std::get<0>(runs[i]) - &codePoints[0]);
                endCluster += (std::get<0>(runs[i]) - &codePoints[0]);

                if (textDirections[startCluster].category == text_category::Whitespace && aUtf32Begin[startCluster] == U'\r')
                {
                    if (aUtf32Begin + startCluster + 1 == aUtf32End || aUtf32Begin[startCluster + 1] != U'\n')
                        result.line_breaks().push_back(result.size());
                }

                neogfx::font selectedFont = aFontSelector.select_font(startCluster);
                neogfx::font font = selectedFont;
                if (shapes.using_fallback(j))
                {
                    font = font.has_fallback() ? font.fallback() : selectedFont;
                    for (auto fi = shapes.fallback_index(j); font != selectedFont && fi > 0; --fi)
                        font = font.has_fallback() ? font.fallback() : selectedFont;
                }
                    
                auto const& glyphPosition = shapes.glyph_position(j);

                float const cellHeight = static_cast<float>(font.height());

                vec2f advance = textDirections[startCluster].category != text_category::Emoji ?
                    vec2{ glyphPosition.x_advance / 64.0, glyphPosition.y_advance / 64.0 }.round() :
                    vec2{ cellHeight, 0.0 }.round();
                vec2f const offset = vec2{ glyphPosition.x_offset / 64.0, glyphPosition.y_offset / 64.0 }.round();

                auto& newGlyph = result.emplace_back(
                    shapes.glyph_info(j).codepoint,
                    glyph_char::cluster_range{ static_cast<std::uint32_t>(clusters[startCluster]), static_cast<std::uint32_t>(clusters[startCluster] + (endCluster - startCluster)) },
                    textDirections[startCluster],
                    glyph_char::flags_e{},
                    font.id(),
                    quadf_2d{},
                    quadf_2d{});

                if (category(newGlyph) == text_category::Whitespace)
                    newGlyph.value = codePoints[startCluster];
                else if (category(newGlyph) == text_category::Emoji)
                    try { newGlyph.value = emojiAtlas.emoji(aUtf32Begin[startCluster], font.height()); }
                    catch (...) { newGlyph.type.category = text_category::FontEmoji; }
                if ((selectedFont.style() & font_style::Underline) == font_style::Underline)
                    set_underline(newGlyph, true);
                if ((selectedFont.style() & font_style::Superscript) == font_style::Superscript)
                    set_superscript(newGlyph, true, (selectedFont.style() & font_style::BelowAscenderLine) == font_style::BelowAscenderLine);
                if ((selectedFont.style() & font_style::Subscript) == font_style::Subscript)
                    set_subscript(newGlyph, true, (selectedFont.style() & font_style::AboveBaseline) == font_style::AboveBaseline);
                if (aGc.is_subpixel_rendering_on() && !font.is_bitmap_font())
                    set_subpixel(newGlyph, true);
                if (drawMnemonic && ((j == 0 && std::get<2>(runs[i]) == text_direction::LTR) || (j == shapes.glyph_count() - 1 && std::get<2>(runs[i]) == text_direction::RTL)))
                    set_mnemonic(newGlyph, true);

                if (category(newGlyph) == text_category::Whitespace)
                {
                    if (newGlyph.value == U'\r' || newGlyph.value == U'\n')
                    {
                        lineStart = previousCell[0].x + previousAdvance.x;
                        advance = {};
                    }
                    else if (newGlyph.value == U'\t' && aGc.has_tab_stops())
                    {
                        // todo: tab stop list and tab alignment
                        if (!aGc.tab_stops().stops().empty() || aGc.tab_stops().default_stop().alignment != alignment::Left)
                            throw not_yet_implemented("Extended tab stop functionality not yet implemented");
                        auto const tabStopPos = static_cast<float>(aGc.tab_stops().default_stop().pos);
                        advance.x = tabStopPos - std::fmod((previousCell[0] + previousAdvance).x - lineStart, tabStopPos);
                    }
                }

                if (category(newGlyph) != text_category::Emoji)
                {
                    auto const& glyphTexture = font.glyph(newGlyph);
                    auto const& glyphTextureExtents = glyphTexture.texture().extents().as<float>();
                    float const cellWidth = (category(newGlyph) != text_category::Whitespace ? std::max(advance.x, glyphTextureExtents.cx) : advance.x);
                    auto const& glyphMetrics = glyphTexture.metrics();

                    newGlyph.cell = quadf_2d{
                        previousCell[0] + previousAdvance,
                        previousCell[0] + previousAdvance + vec2f{ cellWidth, 0.0f },
                        previousCell[0] + previousAdvance + vec2f{ cellWidth, cellHeight },
                        previousCell[0] + previousAdvance + vec2f{ 0.0f, cellHeight } };

                    newGlyph.shape = category(newGlyph) != text_category::Whitespace ? 
                        quadf_2d{
                            offset,
                            offset + vec2f{ glyphTextureExtents.cx, 0.0f },
                            offset + vec2f{ glyphTextureExtents.cx, glyphTextureExtents.cy },
                            offset + vec2f{ 0.0f, glyphTextureExtents.cy } } : 
                        quadf_2d{};

                    vec2f const shapeAdjust = vec2{
                        glyphMetrics.bearing.x,
                        glyphMetrics.bearing.y - glyphMetrics.extents.y + -font.descender() }.as<float>();
                    newGlyph.shape += shapeAdjust;
                }
                else
                {
                    float const cellWidth = advance.x;

                    newGlyph.cell = quadf_2d{
                        previousCell[0] + previousAdvance,
                        previousCell[0] + previousAdvance + vec2f{ cellWidth, 0.0f },
                        previousCell[0] + previousAdvance + vec2f{ cellWidth, cellHeight },
                        previousCell[0] + previousAdvance + vec2f{ 0.0f, cellHeight } };

                    newGlyph.shape = quadf_2d{
                        vec2f{ 0.0f, 0.0f },
                        vec2f{ cellWidth, 0.0f },
                        vec2f{ cellWidth, cellHeight },
                        vec2f{ 0.0f, cellHeight } };
                }

                if (aGc.logical_coordinate_system() == logical_coordinate_system::AutomaticGui)
                    for (auto& v : newGlyph.shape)
                        v.y = -v.y + cellHeight;

                previousAdvance = advance;
                previousCell = newGlyph.cell;
            }
        }
        if (hasEmojis)
        {
            auto refEmojiResult = make_ref<glyph_text_content>(aFontSelector.select_font(0));
            auto& emojiResult = *refEmojiResult;
            emojiResult.line_breaks() = result.line_breaks();
            vec2f advanceAdjust = {};
            for (auto i = result.begin(); i != result.end(); ++i)
            {
                auto cluster = i->clusters.first;
                auto chStart = aUtf32Begin[cluster];
                if (category(*i) == text_category::Emoji)
                {
                    if (!emojiResult.empty() && is_emoji(emojiResult.back()) && emojiResult.back().clusters == i->clusters)
                    {
                        // probable variant selector fubar'd by harfbuzz
                        auto s = emojiResult.back().clusters;
                        if (s.second < codePointCount && get_text_category(emojiAtlas, aUtf32Begin[s.second]) == text_category::Control)
                        {
                            ++s.first;
                            ++s.second;
                            i->clusters = s;
                            set_category(*i, text_category::Control);
                            i->cell = {};
                        }
                    }
                    thread_local std::u32string sequence;
                    sequence.clear();
                    sequence += chStart;
                    auto j = i + 1;
                    for (; j != result.end(); ++j)
                    {
                        auto ch = aUtf32Begin[cluster + (j - i)];
                        if (emojiAtlas.is_emoji(sequence + ch))
                            sequence += ch;
                        else
                            break;
                    }
                    if (sequence.size() > 1)
                    {
                        auto g = *i;
                        g.value = emojiAtlas.emoji(sequence, aFontSelector.select_font(cluster).height());
                        g.clusters = glyph_char::cluster_range{ g.clusters.first, g.clusters.first + static_cast<std::uint32_t>(sequence.size()) };
                        g.cell += advanceAdjust;
                        emojiResult.push_back(g);
                        auto toCombine = sequence.size() - 1;
                        while (toCombine--)
                        {
                            advanceAdjust -= vec2f{ i->cell[1].x - i->cell[0].x, 0.0f };
                            ++i;
                        }
                    }
                    else
                    {
                        auto g = *i;
                        g.cell += advanceAdjust;
                        emojiResult.push_back(g);
                    }
                }
                else
                {
                    auto g = *i;
                    g.cell += advanceAdjust;
                    emojiResult.push_back(g);
                }
            }
            if (aAlignBaselines)
                return emojiResult.align_baselines();
            else
                return emojiResult;
        }
        if (aAlignBaselines)
            return result.align_baselines();
        else
            return result;
    }

    font_manager::font_manager() :
        iGlyphTextFactory{ std::make_unique<neogfx::glyph_text_factory>() },
        iGlyphAtlas{ size{1024.0, 1024.0} },
        iEmojiAtlas{}
    {
        FT_Error error = FT_Init_FreeType(&iFontLib);
        if (error)
            throw error_initializing_font_library();
        error = FT_Library_SetLcdFilter(iFontLib, FT_LCD_FILTER_NONE);
        if (error)
            throw error_initializing_font_library();
        auto enumerate = [this](const std::string fontsDirectory)
        {
            if (std::filesystem::exists(fontsDirectory))
                for (std::filesystem::directory_iterator file(fontsDirectory); file != std::filesystem::directory_iterator(); ++file)
                {
                    if (!std::filesystem::is_regular_file(file->status()))
                        continue;
                    try
                    {
                        if (is_font_file(string{ file->path().string() }))
                        {
                            auto font = iNativeFonts.emplace(iNativeFonts.end(), iFontLib, file->path().string());
                            iFontFamilies[font->family_name()].push_back(font);
                        }
                    }
                    catch (native_font::failed_to_load_font&)
                    {
                    }
                    catch (...)
                    {
                        throw;
                    }
                }
        };
        enumerate(detail::platform_specific::get_system_font_directory());
        enumerate(detail::platform_specific::get_local_font_directory());
        for (auto& family : iFontFamilies)
        {
            std::optional<native_font_list::iterator> bold;
            std::optional<native_font_list::iterator> italic;
            std::optional<native_font_list::iterator> boldItalic;
            std::vector<native_font_list::iterator> emulatedBold;
            std::vector<native_font_list::iterator> emulatedItalic;
            std::vector<native_font_list::iterator> emulatedBoldItalic;
            for (auto& font : family.second)
            {
                if (font->has_style(font_style::Bold))
                    bold = font;
                if (font->has_style(font_style::Italic))
                    italic = font;
                if (font->has_style(font_style::BoldItalic))
                    boldItalic = font;
                if (font->has_style(font_style::EmulatedBold))
                    emulatedBold.push_back(font);
                if (font->has_style(font_style::EmulatedItalic))
                    emulatedItalic.push_back(font);
                if (font->has_style(font_style::EmulatedBoldItalic))
                    emulatedBoldItalic.push_back(font);
            }
            if (bold)
                for (auto& f : emulatedBold)
                    (*f).remove_style(font_style::EmulatedBold);
            if (italic)
                for (auto& f : emulatedItalic)
                    (*f).remove_style(font_style::EmulatedItalic);
            if (boldItalic)
                for (auto& f : emulatedBoldItalic)
                    (*f).remove_style(font_style::EmulatedBoldItalic);
            std::sort(family.second.begin(), family.second.end(),
                [](auto const& f1, auto const& f2) { return f1->min_style() < f2->min_style() || (f1->min_style() == f2->min_style() && f1->min_weight() < f2->min_weight()); });
        }
    }

    font_manager::~font_manager()
    {
        iIdCache.clear();
        iFontFamilies.clear();
        iNativeFonts.clear();
        FT_Done_FreeType(iFontLib);
    }

    void* font_manager::font_library_handle() const
    {
        return iFontLib;
    }

    i_optional<font_info> const& font_manager::default_system_font_info(system_font_role aRole) const
    {
        if (iDefaultSystemFontInfo[aRole] == std::nullopt)
            iDefaultSystemFontInfo[aRole] = detail::platform_specific::default_system_font_info(aRole);
        return iDefaultSystemFontInfo[aRole];
    }

    const i_fallback_font_info& font_manager::default_fallback_font_info() const
    {
        if (iDefaultFallbackFontInfo == std::nullopt)
            iDefaultFallbackFontInfo = detail::platform_specific::default_fallback_font_info();
        return *iDefaultFallbackFontInfo;
    }

    i_native_font_face& font_manager::create_default_font(const i_device_resolution& aDevice)
    {
        return create_font(service<i_app>().current_style().font_info(), aDevice);
    }

    bool font_manager::has_fallback_font(const i_native_font_face& aExistingFont) const
    {
        return default_fallback_font_info().has_fallback_for(aExistingFont.family_name());
    }
        
    i_native_font_face& font_manager::create_fallback_font(const i_native_font_face& aExistingFont)
    {
        if (!has_fallback_font(aExistingFont))
            throw no_fallback_font();
        if (aExistingFont.fallback_cached())
            return aExistingFont.fallback();
        struct : i_device_resolution
        {
            size iResolution;
            virtual dimension horizontal_dpi() const { return iResolution.cx; }
            virtual dimension vertical_dpi() const { return iResolution.cy; }
            virtual dimension ppi() const { return iResolution.magnitude() / std::sqrt(2.0); }
        } deviceResolution;
        deviceResolution.iResolution = size(aExistingFont.horizontal_dpi(), aExistingFont.vertical_dpi());
        string fallbackFontFamily = aExistingFont.family_name();
        try
        {
            bool found = false;
            while (!found)
            {
                fallbackFontFamily = default_fallback_font_info().fallback_for(fallbackFontFamily);
                found = (iFontFamilies.find(fallbackFontFamily) != iFontFamilies.end());
            }
        }
        catch (...)
        {
        }
        auto& fallbackFont = create_font(fallbackFontFamily, (aExistingFont.style() & ~font_style::Emulated), aExistingFont.size(), deviceResolution);
        return fallbackFont;
    }

    i_native_font_face& font_manager::create_font(i_string const& aFamilyName, neogfx::font_style aStyle, font::point_size aSize, const i_device_resolution& aDevice)
    {
        if (aStyle == neogfx::font_style::Emulated)
            aStyle = neogfx::font_style::Normal;
        return add_font(find_best_font(aFamilyName, aStyle, aSize).create_face(aStyle, aSize, {}, aDevice));
    }

    i_native_font_face& font_manager::create_font(i_string const& aFamilyName, i_string const& aStyleName, font::point_size aSize, const i_device_resolution& aDevice)
    {
        return add_font(find_font(aFamilyName, aStyleName, aSize).create_face(aStyleName, aSize, {}, aDevice));
    }

    i_native_font_face& font_manager::create_font(const font_info& aFontInfo, const i_device_resolution& aDevice)
    {
        return add_font(find_font(aFontInfo).create_face(aFontInfo, aDevice));
    }

    i_native_font_face& font_manager::create_font(i_native_font& aFont, neogfx::font_style aStyle, font::point_size aSize, const i_device_resolution& aDevice)
    {
        if (aStyle == neogfx::font_style::Emulated)
            aStyle = neogfx::font_style::Normal;
        return add_font(aFont.create_face(aStyle, aSize, {}, aDevice));
    }

    i_native_font_face& font_manager::create_font(i_native_font& aFont, i_string const& aStyleName, font::point_size aSize, const i_device_resolution& aDevice)
    {
        return add_font(aFont.create_face(aStyleName, aSize, {}, aDevice));
    }

    i_native_font_face& font_manager::create_font(i_native_font& aFont, const font_info& aFontInfo, const i_device_resolution& aDevice)
    {
        return add_font(aFont.create_face(aFontInfo, aDevice));
    }

    bool font_manager::is_font_file(i_string const& aFileName) const
    {
        FT_Face face;
        FT_Error error = FT_New_Face(iFontLib, aFileName.c_str(), 0, &face);
        if (error)
            return false;
        FT_Done_Face(face);
        return true;
    }

    i_native_font_face& font_manager::load_font_from_file(i_string const& aFileName, const i_device_resolution& aDevice)
    {
        throw std::logic_error("neogfx::font_manager::load_font_from_file function overload not yet implemented");
        (void)aFileName;
        (void)aDevice;
    }

    i_native_font_face& font_manager::load_font_from_file(i_string const& aFileName, neogfx::font_style aStyle, font::point_size aSize, const i_device_resolution& aDevice)
    {
        throw std::logic_error("neogfx::font_manager::load_font_from_file function overload not yet implemented");
        (void)aFileName;
        (void)aStyle;
        (void)aSize;
        (void)aDevice;
    }

    i_native_font_face& font_manager::load_font_from_file(i_string const& aFileName, i_string const& aStyleName, font::point_size aSize, const i_device_resolution& aDevice)
    {
        throw std::logic_error("neogfx::font_manager::load_font_from_file function overload not yet implemented");
        (void)aFileName;
        (void)aStyleName;
        (void)aSize;
        (void)aDevice;
    }

    i_native_font_face& font_manager::load_font_from_file(i_string const& aFileName, font_info const& aFontInfo, const i_device_resolution& aDevice)
    {
        throw std::logic_error("neogfx::font_manager::load_font_from_file function overload not yet implemented");
        (void)aFileName;
        (void)aFontInfo;
        (void)aDevice;
    }

    i_native_font_face& font_manager::load_font_from_memory(const void* aData, std::size_t aSizeInBytes, const i_device_resolution& aDevice)
    {
        throw std::logic_error("neogfx::font_manager::load_font_from_memory function overload not yet implemented");
        (void)aData;
        (void)aSizeInBytes;
        (void)aDevice;
    }

    i_native_font_face& font_manager::load_font_from_memory(const void* aData, std::size_t aSizeInBytes, neogfx::font_style aStyle, font::point_size aSize, const i_device_resolution& aDevice)
    {
        throw std::logic_error("neogfx::font_manager::load_font_from_memory function overload not yet implemented");
        (void)aData;
        (void)aSizeInBytes;
        (void)aStyle;
        (void)aSize;
        (void)aDevice;
    }

    i_native_font_face& font_manager::load_font_from_memory(const void* aData, std::size_t aSizeInBytes, i_string const& aStyleName, font::point_size aSize, const i_device_resolution& aDevice)
    {
        throw std::logic_error("neogfx::font_manager::load_font_from_memory function overload not yet implemented");
        (void)aData;
        (void)aSizeInBytes;
        (void)aStyleName;
        (void)aSize;
        (void)aDevice;
    }

    i_native_font_face& font_manager::load_font_from_memory(const void* aData, std::size_t aSizeInBytes, font_info const& aFontInfo, const i_device_resolution& aDevice)
    {
        throw std::logic_error("neogfx::font_manager::load_font_from_memory function overload not yet implemented");
        (void)aData;
        (void)aSizeInBytes;
        (void)aFontInfo;
        (void)aDevice;
    }

    std::uint32_t font_manager::font_family_count() const
    {
        return static_cast<std::uint32_t>(iFontFamilies.size());
    }

    i_string const& font_manager::font_family(std::uint32_t aFamilyIndex) const
    {
        if (aFamilyIndex < font_family_count())
            return std::next(iFontFamilies.begin(), aFamilyIndex)->first;
        throw bad_font_family_index();
    }

    std::uint32_t font_manager::font_style_count(std::uint32_t aFamilyIndex) const
    {
        if (aFamilyIndex < font_family_count())
        {
            std::uint32_t styles = 0;
            for (auto& font : std::next(iFontFamilies.begin(), aFamilyIndex)->second)
                styles += font->style_count();
            return styles;
        }
        throw bad_font_family_index();
    }

    font_style font_manager::font_style(std::uint32_t aFamilyIndex, std::uint32_t aStyleIndex) const
    {
        if (aFamilyIndex < font_family_count() && aStyleIndex < font_style_count(aFamilyIndex))
        {
            for (auto& font : std::next(iFontFamilies.begin(), aFamilyIndex)->second)
            {
                if (aStyleIndex < font->style_count())
                    return font->style(aStyleIndex);
                aStyleIndex -= font->style_count();
            }
        }
        throw bad_font_family_index();
    }

    i_string const& font_manager::font_style_name(std::uint32_t aFamilyIndex, std::uint32_t aStyleIndex) const
    {
        if (aFamilyIndex < font_family_count() && aStyleIndex < font_style_count(aFamilyIndex))
        {
            for (auto& font : std::next(iFontFamilies.begin(), aFamilyIndex)->second)
            {
                if (aStyleIndex < font->style_count())
                    return font->style_name(aStyleIndex);
                aStyleIndex -= font->style_count();
            }
        }
        throw bad_font_family_index();
    }

    font_id font_manager::allocate_font_id()
    {
        return iIdCache.next_cookie();
    }

    const font& font_manager::font_from_id(font_id aId) const
    {
        return iIdCache[aId];
    }

    i_glyph_text_factory& font_manager::glyph_text_factory() const
    {
        return *iGlyphTextFactory;
    }

    const i_texture_atlas& font_manager::glyph_atlas() const
    {
        return iGlyphAtlas;
    }

    i_texture_atlas& font_manager::glyph_atlas()
    {
        return iGlyphAtlas;
    }

    const i_emoji_atlas& font_manager::emoji_atlas() const
    {
        return iEmojiAtlas;
    }

    i_emoji_atlas& font_manager::emoji_atlas()
    {
        return iEmojiAtlas;
    }

    void font_manager::add_ref(font_id aId)
    {
        font_from_id(aId).native_font_face().add_ref();
    }

    void font_manager::release(font_id aId)
    {
        font_from_id(aId).native_font_face().release();
    }

    long font_manager::use_count(font_id aId) const
    {
        return font_from_id(aId).native_font_face().use_count();
    }

    i_native_font& font_manager::find_font(i_string const& aFamilyName, i_string const& aStyleName, font::point_size aSize)
    {
        auto family = iFontFamilies.find(aFamilyName);
        if (family == iFontFamilies.end() && default_system_font_info(system_font_role::Widget) != std::nullopt)
            family = iFontFamilies.find(default_system_font_info(system_font_role::Widget)->family_name());
        if (family == iFontFamilies.end())
            throw no_matching_font_found();
        std::multimap<std::uint32_t, native_font_list::iterator> matches;
        for (auto& f : family->second)
        {
            for (std::uint32_t s = 0; s < f->style_count(); ++s)
                if (neolib::ci_equal_to{}(f->style_name(s), aStyleName))
                    return *f;
        }
        return find_best_font(aFamilyName, font_style::Normal, aSize);
    }

    i_native_font& font_manager::find_font(font_info const& aFontInfo)
    {
        if (aFontInfo.style_name_available())
            return find_font(aFontInfo.family_name(), aFontInfo.style_name(), aFontInfo.size());
        else
            return find_best_font(aFontInfo.family_name(), aFontInfo.style(), aFontInfo.size());
    }

    namespace
    {
        std::uint32_t matching_bits(std::uint32_t lhs, std::uint32_t rhs)
        {
            if (lhs == rhs)
                return 32;
            std::uint32_t matches = 0;
            std::uint32_t test = 1;
            while (test != 0)
            {
                if ((lhs & rhs) & test)
                    ++matches;
                test <<= 1;
            }
            return matches;
        }
    }

    i_native_font& font_manager::find_best_font(i_string const& aFamilyName, neogfx::font_style aStyle, font::point_size)
    {
        if (aStyle == neogfx::font_style::Emulated)
            aStyle = neogfx::font_style::Normal;
        auto family = iFontFamilies.find(aFamilyName);
        if (family == iFontFamilies.end() && default_system_font_info(system_font_role::Widget) != std::nullopt)
            family = iFontFamilies.find(default_system_font_info(system_font_role::Widget)->family_name());
        if (family == iFontFamilies.end())
            throw no_matching_font_found();
        struct match
        {
            std::uint32_t matchingBits;
            neogfx::font_style style;
            font_weight weight;
            i_native_font* font;
        };
        std::optional<match> bestNormalFont;
        std::optional<match> bestBoldFont;
        std::optional<match> bestOtherFont;
        for (auto& f : family->second)
        {
            for (std::uint32_t s = 0; s < f->style_count(); ++s)
            {
                auto const matchingBits = matching_bits(static_cast<std::uint32_t>(f->style(s)), static_cast<std::uint32_t>(aStyle));
                auto const& styleName = f->style_name(s);
                auto const weight = font_info::weight_from_style_name(styleName);
                if (weight <= font_weight::Normal && (
                    bestNormalFont == std::nullopt ||
                    bestNormalFont->matchingBits < matchingBits ||
                    (bestNormalFont->matchingBits == matchingBits && bestNormalFont->weight < weight)))
                {
                    bestNormalFont = match{ matchingBits, f->style(s), weight, &*f };
                }
                else if (weight >= font_weight::Bold && (
                    bestBoldFont == std::nullopt ||
                    bestBoldFont->matchingBits < matchingBits ||
                    (bestBoldFont->style & font_style::Emulated) == font_style::Emulated ||
                    (bestBoldFont->matchingBits == matchingBits && bestBoldFont->weight > weight)))
                {
                    bestBoldFont = match{ matchingBits, f->style(s), weight, &*f };
                }
                else if (bestOtherFont == std::nullopt ||
                    bestOtherFont->matchingBits < matchingBits ||
                    (bestOtherFont->style & font_style::Emulated) == font_style::Emulated ||
                    (bestOtherFont->matchingBits == matchingBits && bestOtherFont->weight < weight))
                {
                    bestOtherFont = match{ matchingBits, f->style(s), weight, &*f };
                }
            }
        }
        if ((aStyle & neogfx::font_style::Bold) != neogfx::font_style::Bold)
        {
            if (bestNormalFont != std::nullopt)
                return *bestNormalFont->font;
            else if (bestOtherFont != std::nullopt)
                return *bestOtherFont->font;
            else if (bestBoldFont != std::nullopt)
                return *bestBoldFont->font;
            else
                throw no_matching_font_found();
        }
        else
        {
            if (bestBoldFont != std::nullopt)
                return *bestBoldFont->font;
            else if (bestOtherFont != std::nullopt)
                return *bestOtherFont->font;
            else if (bestNormalFont != std::nullopt)
                return *bestNormalFont->font;
            else
                throw no_matching_font_found();
        }
    }

    i_native_font_face& font_manager::add_font(const ref_ptr<i_native_font_face>& aNewFont)
    {
        if (!iIdCache.contains(aNewFont->id()))
            iIdCache.add(aNewFont->id(), font{ *aNewFont });
        // cleanup opportunity
        cleanup();
        return *aNewFont;
    }

    void font_manager::cleanup()
    {
        for (auto i = iIdCache.begin(); i != iIdCache.end();)
        {
            auto& cacheEntry = *i;
            if (cacheEntry.native_font_face().use_count() == 1)
                i = iIdCache.erase(i);
            else
                ++i;
        }
    }
}