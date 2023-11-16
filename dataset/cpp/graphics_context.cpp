// graphics_context.cpp
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
#include <neolib/core/string_utils.hpp>
#include "native/i_native_texture.hpp"
#include "text/native/native_font_face.hpp"
#include <neogfx/hid/i_native_surface.hpp>
#include <neogfx/hid/i_surface.hpp>
#include <neogfx/gfx/i_texture.hpp>
#include <neogfx/gui/widget/i_widget.hpp>
#include <neogfx/gfx/text/text_category_map.hpp>
#include <neogfx/gfx/i_rendering_engine.hpp>
#include <neogfx/gfx/i_rendering_context.hpp>
#include <neogfx/gfx/text/i_font_manager.hpp>
#include <neogfx/game/mesh.hpp>
#include <neogfx/game/rectangle.hpp>
#include <neogfx/game/ecs_helpers.hpp>
#include <neogfx/gfx/graphics_context.hpp>

namespace neogfx
{
    ping_pong_buffers create_ping_pong_buffers(i_rendering_context const& aContext, size const& aExtents, texture_sampling aSampling, optional_color const& aClearColor)
    {
        size previousExtents;
        auto& buffer1 = service<i_rendering_engine>().ping_pong_buffer1(aExtents, previousExtents, aSampling);
        buffer1.as_render_target().set_logical_coordinate_system(aContext.logical_coordinate_system());
        auto gcBuffer1 = std::make_unique<graphics_context>(buffer1);
        {
            if (aClearColor != std::nullopt)
            {
                scoped_render_target srt{ *gcBuffer1 };
                scoped_scissor ss{ *gcBuffer1, rect{ point{}, previousExtents }.inflate(1.0) };
                gcBuffer1->clear(*aClearColor);
                gcBuffer1->clear_depth_buffer();
                gcBuffer1->clear_stencil_buffer();
            }
        }
        auto& buffer2 = service<i_rendering_engine>().ping_pong_buffer2(aExtents, previousExtents, aSampling);
        buffer2.as_render_target().set_logical_coordinate_system(aContext.logical_coordinate_system());
        auto gcBuffer2 = std::make_unique<graphics_context>(buffer2);
        {
            if (aClearColor != std::nullopt)
            {
                scoped_render_target srt{ *gcBuffer2 };
                scoped_scissor ss{ *gcBuffer2, rect{ point{}, previousExtents }.inflate(1.0) };
                gcBuffer2->clear(*aClearColor);
                gcBuffer2->clear_depth_buffer();
                gcBuffer2->clear_stencil_buffer();
            }
        }
        return ping_pong_buffers{ {}, std::move(gcBuffer1), std::move(gcBuffer2) };
    }

    graphics_context::graphics_context(i_surface const& aSurface, type aType) :
        iType{ aType },
        iRenderTarget{ aSurface.native_surface() },
        iNativeGraphicsContext{ nullptr },
        iDefaultFont{},
        iExtents{ aSurface.extents() },
        iLayer{ LayerWidget },
        iSnapToPixel{ false },
        iOpacity{ 1.0 },
        iBlendingMode{ neogfx::blending_mode::Default },
        iSmoothingMode{ neogfx::smoothing_mode::None },
        iSubpixelRendering{ service<i_rendering_engine>().is_subpixel_rendering_on() }
    {
    }

    graphics_context::graphics_context(i_surface const& aSurface, font const& aDefaultFont, type aType) :
        iType{ aType },
        iRenderTarget{ aSurface.native_surface() },
        iNativeGraphicsContext{ nullptr },
        iDefaultFont{ aDefaultFont },
        iExtents{ aSurface.extents() },
        iLayer{ LayerWidget },
        iSnapToPixel{ false },
        iOpacity{ 1.0 },
        iBlendingMode{ neogfx::blending_mode::Default },
        iSmoothingMode{ neogfx::smoothing_mode::None },
        iSubpixelRendering{ service<i_rendering_engine>().is_subpixel_rendering_on() }
    {
    }

    graphics_context::graphics_context(i_widget const& aWidget, type aType) :
        iType{ aType },
        iRenderTarget{ aWidget.surface().native_surface() },
        iNativeGraphicsContext{ nullptr },
        iDefaultFont{ aWidget.font() },
        iOrigin{ aWidget.origin() },
        iExtents{ aWidget.extents() },
        iLayer{ LayerWidget },
        iSnapToPixel{ false },
        iOpacity{ 1.0 },
        iBlendingMode{ neogfx::blending_mode::Default },
        iSmoothingMode{ neogfx::smoothing_mode::None },
        iSubpixelRendering{ service<i_rendering_engine>().is_subpixel_rendering_on() }
    {
        set_logical_coordinate_system(aWidget.logical_coordinate_system());
    }

    graphics_context::graphics_context(i_texture const& aTexture, type aType) :
        iType{ aType },
        iRenderTarget{ static_cast<i_native_texture&>(aTexture.native_texture()) },
        iNativeGraphicsContext{ nullptr },
        iDefaultFont{ font() },
        iExtents{ aTexture.extents() },
        iLayer{ LayerWidget },
        iSnapToPixel{ false },
        iOpacity{ 1.0 },
        iBlendingMode{ neogfx::blending_mode::Default },
        iSmoothingMode{ neogfx::smoothing_mode::None },
        iSubpixelRendering{ service<i_rendering_engine>().is_subpixel_rendering_on() }
    {
    }

    graphics_context::graphics_context(graphics_context const& aOther) :
        iType{ aOther.iType },
        iRenderTarget{ aOther.iRenderTarget },
        iSrt{ aOther.iRenderTarget },
        iNativeGraphicsContext{ aOther.active() ? aOther.native_context().clone() : nullptr },
        iDefaultFont{ aOther.iDefaultFont },
        iOrigin{ aOther.origin() },
        iExtents{ aOther.extents() },
        iLayer{ LayerWidget },
        iLogicalCoordinateSystem{ aOther.iLogicalCoordinateSystem },
        iLogicalCoordinates{ aOther.iLogicalCoordinates },
        iSnapToPixel{ aOther.iSnapToPixel },
        iOpacity{ 1.0 },
        iBlendingMode{ neogfx::blending_mode::Default },
        iSmoothingMode{ neogfx::smoothing_mode::None },
        iSubpixelRendering{ service<i_rendering_engine>().is_subpixel_rendering_on() }
    {
    }

    graphics_context::~graphics_context()
    {
        flush();
    }

    std::unique_ptr<i_rendering_context> graphics_context::clone() const
    {
        return std::make_unique<graphics_context>(*this);
    }

    i_rendering_engine& graphics_context::rendering_engine() const
    {
        return native_context().rendering_engine();
    }

    const i_render_target& graphics_context::render_target() const
    {
        return iRenderTarget;
    }

    rect graphics_context::rendering_area(bool aConsiderScissor) const
    {
        return native_context().rendering_area(aConsiderScissor);
    }

    const graphics_operation::queue& graphics_context::queue() const
    {
        return native_context().queue();
    }

    graphics_operation::queue& graphics_context::queue()
    {
        return native_context().queue();
    }

    void graphics_context::enqueue(graphics_operation::operation const& aOperation)
    {
        native_context().enqueue(aOperation);
    }

    void graphics_context::flush()
    {
        if (attached() && active())
            native_context().flush();
    }

    delta graphics_context::to_device_units(delta const& aValue) const
    {
        return units_converter{ *this }.to_device_units(aValue);
    }

    size graphics_context::to_device_units(size const& aValue) const
    {
        return units_converter{ *this }.to_device_units(aValue);
    }

    point graphics_context::to_device_units(point const& aValue) const
    {
        return units_converter{ *this }.to_device_units(aValue);
    }

    vec2 graphics_context::to_device_units(vec2 const& aValue) const
    {
        return units_converter{ *this }.to_device_units(aValue);
    }

    rect graphics_context::to_device_units(rect const& aValue) const
    {
        return units_converter{ *this }.to_device_units(aValue);
    }

    path graphics_context::to_device_units(path const& aValue) const
    {
        path result = aValue;
        result.set_position(to_device_units(result.position()));
        for (std::size_t i = 0; i < result.sub_paths().size(); ++i)
            for (std::size_t j = 0; j < result.sub_paths()[i].size(); ++j)
                result.sub_paths()[i][j] = to_device_units(result.sub_paths()[i][j]);
        return result;
    }

    delta graphics_context::from_device_units(delta const& aValue) const
    {
        return units_converter{ *this }.from_device_units(aValue);
    }

    size graphics_context::from_device_units(size const& aValue) const
    {
        return units_converter{ *this }.from_device_units(aValue);
    }

    point graphics_context::from_device_units(point const& aValue) const
    {
        return units_converter{ *this }.from_device_units(aValue);
    }

    rect graphics_context::from_device_units(rect const& aValue) const
    {
        return units_converter{ *this }.from_device_units(aValue);
    }

    path graphics_context::from_device_units(path const& aValue) const
    {
        path result = aValue;
        result.set_position(from_device_units(result.position()));
        for (std::size_t i = 0; i < result.sub_paths().size(); ++i)
            for (std::size_t j = 0; j < result.sub_paths()[i].size(); ++j)
                result.sub_paths()[i][j] = from_device_units(result.sub_paths()[i][j]);
        return result;
    }

    layer_t graphics_context::layer() const
    {
        return iLayer;
    }

    void graphics_context::set_layer(layer_t aLayer)
    {
        if (iLayer != aLayer)
        {
            iLayer = aLayer;
            // todo:-
//            if (attached())
//                native_context().enqueue(graphics_operation::set_layer{ aLayer });
        }
    }

    neogfx::logical_coordinate_system graphics_context::logical_coordinate_system() const
    {
        if (iLogicalCoordinateSystem != std::nullopt)
            return *iLogicalCoordinateSystem;
        return render_target().logical_coordinate_system();
    }

    void graphics_context::set_logical_coordinate_system(neogfx::logical_coordinate_system aSystem) const
    {
        if (iLogicalCoordinateSystem != aSystem)
        {
            iLogicalCoordinateSystem = aSystem;
            if (attached())
                native_context().enqueue(graphics_operation::set_logical_coordinate_system{ aSystem });
        }
    }

    neogfx::logical_coordinates graphics_context::logical_coordinates() const
    {
        if (iLogicalCoordinates != std::nullopt)
            return *iLogicalCoordinates;
        return render_target().logical_coordinates();
    }

    void graphics_context::set_logical_coordinates(neogfx::logical_coordinates const& aCoordinates) const
    {
        if (iLogicalCoordinates != aCoordinates)
        {
            iLogicalCoordinates = aCoordinates;
            if (attached())
                native_context().enqueue(graphics_operation::set_logical_coordinates{ aCoordinates });
        }
    }

    vec2 graphics_context::offset() const
    {
        throw not_implemented();
    }

    void graphics_context::set_offset(optional_vec2 const& aOffset)
    {
        throw not_implemented();
    }

    bool graphics_context::gradient_set() const
    {
        throw not_implemented();
    }

    void graphics_context::apply_gradient(i_gradient_shader& aShader)
    {
        throw not_implemented();
    }

    font const& graphics_context::default_font() const
    {
        return iDefaultFont;
    }

    void graphics_context::set_default_font(font const& aDefaultFont) const
    {
        iDefaultFont = aDefaultFont;
    }

    void graphics_context::set_extents(size const& aExtents) const
    {
        iExtents = aExtents;
    }

    void graphics_context::set_origin(point const& aOrigin) const
    {
        if (iOrigin != to_device_units(aOrigin))
        {
            iOrigin = to_device_units(aOrigin);
            native_context().enqueue(graphics_operation::set_origin{ iOrigin });
        }
    }

    point graphics_context::origin() const
    {
        return from_device_units(iOrigin);
    }

    void graphics_context::clear_gradient()
    {
        native_context().enqueue(graphics_operation::clear_gradient{});
    }

    void graphics_context::set_gradient(gradient const& aGradient, rect const& aBoundingBox)
    {
        native_context().enqueue(graphics_operation::set_gradient{ aGradient.with_bounding_box(to_device_units(aBoundingBox) + iOrigin) });
    }

    void graphics_context::set_pixel(point const& aPoint, color const& aColor) const
    {
        native_context().enqueue(graphics_operation::set_pixel{ to_device_units(aPoint) + iOrigin, aColor });
    }

    void graphics_context::draw_pixel(point const& aPoint, color const& aColor) const
    {
        native_context().enqueue(graphics_operation::draw_pixel{ to_device_units(aPoint) + iOrigin, aColor });
    }

    void graphics_context::draw_line(point const& aFrom, point const& aTo, pen const& aPen) const
    {
        native_context().enqueue(graphics_operation::draw_line{ to_device_units(aFrom) + iOrigin, to_device_units(aTo) + iOrigin, aPen });
    }

    void graphics_context::draw_triangle(point const& aP0, point const& aP1, point const& aP2, pen const& aPen, brush const& aFill) const
    {
        native_context().enqueue(graphics_operation::draw_triangle{ to_device_units(aP0) + iOrigin, to_device_units(aP1) + iOrigin, to_device_units(aP2) + iOrigin, aPen, aFill });
    }

    void graphics_context::draw_rect(rect const& aRect, pen const& aPen, brush const& aFill) const
    {
        native_context().enqueue(graphics_operation::draw_rect{ to_device_units(aRect) + iOrigin, aPen, aFill });
    }

    void graphics_context::draw_rounded_rect(rect const& aRect, vec4 const& aRadius, pen const& aPen, brush const& aFill) const
    {
        native_context().enqueue(graphics_operation::draw_rounded_rect{ to_device_units(aRect) + iOrigin, aRadius, aPen, aFill });
    }

    void graphics_context::draw_checker_rect(rect const& aRect, size const& aSquareSize, pen const& aPen, brush const& aFill1, brush const& aFill2) const
    {
        native_context().enqueue(graphics_operation::draw_checker_rect{ to_device_units(aRect) + iOrigin, to_device_units(aSquareSize), aPen, aFill1, aFill2 });
    }

    void graphics_context::draw_circle(point const& aCenter, dimension aRadius, pen const& aPen, brush const& aFill) const
    {
        native_context().enqueue(graphics_operation::draw_circle{ to_device_units(aCenter) + iOrigin, aRadius, aPen, aFill });
    }

    void graphics_context::draw_ellipse(point const& aCenter, dimension aRadiusA, dimension aRadiusB, pen const& aPen, brush const& aFill) const
    {
        native_context().enqueue(graphics_operation::draw_ellipse{ to_device_units(aCenter) + iOrigin, aRadiusA, aRadiusB, aPen, aFill });
    }

    void graphics_context::draw_pie(point const& aCenter, dimension aRadius, angle aStartAngle, angle aEndAngle, pen const& aPen, brush const& aFill) const
    {
        native_context().enqueue(graphics_operation::draw_pie{ to_device_units(aCenter) + iOrigin, aRadius, aStartAngle, aEndAngle, aPen, aFill });
    }

    void graphics_context::draw_arc(point const& aCenter, dimension aRadius, angle aStartAngle, angle aEndAngle, pen const& aPen, brush const& aFill) const
    {
        native_context().enqueue(graphics_operation::draw_arc{ to_device_units(aCenter) + iOrigin, aRadius, aStartAngle, aEndAngle, aPen, aFill });
    }

    void graphics_context::draw_cubic_bezier(point const& aP0, point const& aP1, point const& aP2, point const& aP3, pen const& aPen) const
    {
        native_context().enqueue(graphics_operation::draw_cubic_bezier{ to_device_units(aP0) + iOrigin, to_device_units(aP1) + iOrigin, to_device_units(aP2) + iOrigin, to_device_units(aP3) + iOrigin, aPen });
    }

    void graphics_context::draw_path(path const& aPath, pen const& aPen, brush const& aFill) const
    {
        path path = to_device_units(aPath);
        path.set_position(path.position() + iOrigin);
        native_context().enqueue(graphics_operation::draw_path{ path, aPen, aFill });
    }

    void graphics_context::draw_shape(game::mesh const& aShape, vec3 const& aPosition, pen const& aPen, brush const& aFill) const
    {
        vec2 const toDeviceUnits = to_device_units(vec2{ 1.0, 1.0 });
        native_context().enqueue(
            graphics_operation::draw_shape{
                mat44{ 
                    { toDeviceUnits.x, 0.0, 0.0, 0.0 },
                    { 0.0, toDeviceUnits.y, 0.0, 0.0 },
                    { 0.0, 0.0, 1.0, 0.0 }, 
                    { iOrigin.x, iOrigin.y, 0.0, 1.0 } } * aShape,
                aPosition,
                aPen,
                aFill
            });
    }

    void graphics_context::draw_entities(game::i_ecs& aEcs, int32_t aLayer) const
    {
        vec2 const toDeviceUnits = to_device_units(vec2{ 1.0, 1.0 });
        native_context().enqueue(
            graphics_operation::draw_entities{ 
                aEcs, 
                aLayer,
                mat44{
                    { toDeviceUnits.x, 0.0, 0.0, 0.0 },
                    { 0.0, toDeviceUnits.y, 0.0, 0.0 },
                    { 0.0, 0.0, 1.0, 0.0 },
                    { iOrigin.x, iOrigin.y, 0.0, 1.0 } }
                });
    }

    void graphics_context::draw_focus_rect(rect const& aRect) const
    {
        scoped_snap_to_pixel sntp{ *this, true };

        push_logical_operation(neogfx::logical_operation::Xor);
        line_stipple_on(1.0, 0x5555);
        draw_rect(aRect, pen{ color::White });
        line_stipple_off();
        pop_logical_operation();
    }

    bool graphics_context::has_tab_stops() const
    {
        return iTabStops.has_value();
    }

    i_tab_stops const& graphics_context::tab_stops() const
    {
        if (iTabStops.has_value())
            return iTabStops.value();;
        throw no_tab_stops();
    }

    void graphics_context::set_tab_stops(i_tab_stops const& aTabStops)
    {
        iTabStops.emplace(aTabStops);
    }

    void graphics_context::clear_tab_stops()
    {
        iTabStops = std::nullopt;
    }

    size graphics_context::text_extent(std::string const& aText) const
    {
        return text_extent(aText, default_font());
    }

    size graphics_context::text_extent(std::string const& aText, font const& aFont) const
    {
        return text_extent(aText.begin(), aText.end(), [&aFont](std::size_t) { return aFont; });
    }

    size graphics_context::text_extent(std::string const& aText, std::function<font(std::size_t)> aFontSelector) const
    {
        return text_extent(aText.begin(), aText.end(), aFontSelector);
    }

    size graphics_context::text_extent(std::string::const_iterator aTextBegin, std::string::const_iterator aTextEnd) const
    {
        return text_extent(aTextBegin, aTextEnd, default_font());
    }

    size graphics_context::text_extent(std::string::const_iterator aTextBegin, std::string::const_iterator aTextEnd, font const& aFont) const
    {
        return text_extent(aTextBegin, aTextEnd, [&aFont](std::size_t) { return aFont; });
    }

    size graphics_context::text_extent(std::string::const_iterator aTextBegin, std::string::const_iterator aTextEnd, std::function<font(std::size_t)> aFontSelector) const
    {
        return glyph_text_extent(to_glyph_text(aTextBegin, aTextEnd, aFontSelector));
    }

    size graphics_context::multiline_text_extent(std::string const& aText) const
    {
        return multiline_text_extent(aText, default_font());
    }

    size graphics_context::multiline_text_extent(std::string const& aText, font const& aFont) const
    {
        return multiline_text_extent(aText, [&aFont](std::size_t) { return aFont; }, 0);
    }

    size graphics_context::multiline_text_extent(std::string const& aText, std::function<font(std::size_t)> aFontSelector) const
    {
        return multiline_text_extent(aText, aFontSelector, 0);
    }

    size graphics_context::multiline_text_extent(std::string const& aText, dimension aMaxWidth) const
    {
        return multiline_text_extent(aText, default_font(), aMaxWidth);
    }

    size graphics_context::multiline_text_extent(std::string const& aText, font const& aFont, dimension aMaxWidth) const
    {
        return multiline_text_extent(aText, [&aFont](std::size_t) { return aFont; }, aMaxWidth);
    }
        
    size graphics_context::multiline_text_extent(std::string const& aText, std::function<font(std::size_t)> aFontSelector, dimension aMaxWidth) const
    {
        return multiline_glyph_text_extent(to_glyph_text(aText.begin(), aText.end(), aFontSelector), aMaxWidth);
    }

    size graphics_context::glyph_text_extent(glyph_text const& aText) const
    {
        return from_device_units(aText.extents());
    }

    size graphics_context::glyph_text_extent(glyph_text const& aText, glyph_text::const_iterator aTextBegin, glyph_text::const_iterator aTextEnd) const
    {
        return from_device_units(aText.extents(aTextBegin, aTextEnd));
    }

    size graphics_context::multiline_glyph_text_extent(glyph_text const& aText, dimension aMaxWidth) const
    {
        return quad_extents(to_multiline_glyph_text(aText, aMaxWidth).bbox);
    }

    bool graphics_context::is_text_left_to_right(std::string const& aText) const
    {
        return is_text_left_to_right(aText, default_font());
    }

    bool graphics_context::is_text_left_to_right(std::string const& aText, font const& aFont) const
    {
        auto const& glyphText = to_glyph_text(aText.begin(), aText.end(), aFont);
        return glyph_text_direction(glyphText.cbegin(), glyphText.cend()) == text_direction::LTR;
    }

    bool graphics_context::is_text_right_to_left(std::string const& aText) const
    {
        return is_text_right_to_left(aText, default_font());
    }

    bool graphics_context::is_text_right_to_left(std::string const& aText, font const& aFont) const
    {
        return !is_text_left_to_right(aText, aFont);
    }

    void graphics_context::draw_text(point const& aPoint, std::string const& aText, text_format const& aTextFormat) const
    {
        draw_text(aPoint, aText, default_font(), aTextFormat);
    }

    void graphics_context::draw_text(point const& aPoint, std::string const& aText, font const& aFont, text_format const& aTextFormat) const
    {
        draw_text(aPoint.to_vec3(), aText, aFont, aTextFormat);
    }

    void graphics_context::draw_text(point const& aPoint, std::string::const_iterator aTextBegin, std::string::const_iterator aTextEnd, text_format const& aTextFormat) const
    {
        draw_text(aPoint, aTextBegin, aTextEnd, default_font(), aTextFormat);
    }

    void graphics_context::draw_text(point const& aPoint, std::string::const_iterator aTextBegin, std::string::const_iterator aTextEnd, font const& aFont, text_format const& aTextFormat) const
    {
        draw_text(aPoint.to_vec3(), aTextBegin, aTextEnd, aFont, aTextFormat);
    }

    void graphics_context::draw_text(vec3 const& aPoint, std::string const& aText, text_format const& aTextFormat) const
    {
        draw_text(aPoint, aText, default_font(), aTextFormat);
    }

    void graphics_context::draw_text(vec3 const& aPoint, std::string const& aText, font const& aFont, text_format const& aTextFormat) const
    {
        draw_text(aPoint, aText.begin(), aText.end(), aFont, aTextFormat);
    }

    void graphics_context::draw_text(vec3 const& aPoint, std::string::const_iterator aTextBegin, std::string::const_iterator aTextEnd, text_format const& aTextFormat) const
    {
        draw_text(aPoint, aTextBegin, aTextEnd, default_font() , aTextFormat);
    }

    void graphics_context::draw_text(vec3 const& aPoint, std::string::const_iterator aTextBegin, std::string::const_iterator aTextEnd, font const& aFont, text_format const& aTextFormat) const
    {
        draw_glyph_text(aPoint, to_glyph_text(aTextBegin, aTextEnd, aFont), aTextFormat);
    }

    void graphics_context::draw_multiline_text(point const& aPoint, std::string const& aText, text_format const& aTextFormat, alignment aAlignment) const
    {
        draw_multiline_text(aPoint, aText, default_font(), aTextFormat, aAlignment);
    }

    void graphics_context::draw_multiline_text(point const& aPoint, std::string const& aText, font const& aFont, text_format const& aTextFormat, alignment aAlignment) const
    {
        draw_multiline_text(aPoint.to_vec3(), aText, aFont, aTextFormat, aAlignment);
    }

    void graphics_context::draw_multiline_text(point const& aPoint, std::string const& aText, dimension aMaxWidth, text_format const& aTextFormat, alignment aAlignment) const
    {
        draw_multiline_text(aPoint, aText, default_font(), aMaxWidth, aTextFormat, aAlignment);
    }

    void graphics_context::draw_multiline_text(point const& aPoint, std::string const& aText, font const& aFont, dimension aMaxWidth, text_format const& aTextFormat, alignment aAlignment) const
    {
        draw_multiline_text(aPoint.to_vec3(), aText, aFont, aMaxWidth, aTextFormat, aAlignment);
    }
        
    void graphics_context::draw_multiline_text(vec3 const& aPoint, std::string const& aText, text_format const& aTextFormat, alignment aAlignment) const
    {
        draw_multiline_text(aPoint, aText, default_font(), aTextFormat, aAlignment);
    }

    void graphics_context::draw_multiline_text(vec3 const& aPoint, std::string const& aText, font const& aFont, text_format const& aTextFormat, alignment aAlignment) const
    {
        draw_multiline_text(aPoint, aText, aFont, 0, aTextFormat, aAlignment);
    }

    void graphics_context::draw_multiline_text(vec3 const& aPoint, std::string const& aText, dimension aMaxWidth, text_format const& aTextFormat, alignment aAlignment) const
    {
        draw_multiline_text(aPoint, aText, default_font(), aMaxWidth, aTextFormat, aAlignment);
    }
    
    void graphics_context::draw_multiline_text(vec3 const& aPoint, std::string const& aText, font const& aFont, dimension aMaxWidth, text_format const& aTextFormat, alignment aAlignment) const
    {
        draw_multiline_glyph_text(aPoint, to_multiline_glyph_text(aText, aFont, aMaxWidth, aAlignment), aTextFormat);
    }

    void graphics_context::draw_glyph_text(point const& aPoint, glyph_text const& aText, text_format const& aTextFormat) const
    {
        draw_glyph_text(aPoint.to_vec3(), aText, aTextFormat);
    }

    void graphics_context::draw_glyph_text(point const& aPoint, glyph_text const& aText, glyph_text::const_iterator aTextBegin, glyph_text::const_iterator aTextEnd, text_format const& aTextFormat) const
    {
        draw_glyph_text(aPoint.to_vec3(), aText, aTextBegin, aTextEnd, aTextFormat);
    }

    void graphics_context::draw_glyph_text(vec3 const& aPoint, glyph_text const& aText, text_format const& aTextFormat) const
    {
        draw_glyph_text(aPoint, aText, aText.cbegin(), aText.cend(), aTextFormat);
    }

    void graphics_context::draw_glyph_text(vec3 const& aPoint, glyph_text const& aText, glyph_text::const_iterator aTextBegin, glyph_text::const_iterator aTextEnd, text_format const& aTextFormat) const
    {
        if (aTextBegin == aTextEnd)
            return;
        auto adjustedPos = (to_device_units(point{ aPoint }) + iOrigin).to_vec3() + vec3{ 0.0, 0.0, aPoint.z };
        native_context().enqueue(graphics_operation::draw_glyphs{ adjustedPos, aText, aTextBegin, aTextEnd, text_format_span{ 0, aTextEnd - aTextBegin, aTextFormat }, mnemonics_shown() });
    }

    void graphics_context::draw_multiline_glyph_text(point const& aPoint, glyph_text const& aText, dimension aMaxWidth, text_format const& aTextFormat, alignment aAlignment) const
    {
        draw_multiline_glyph_text(aPoint.to_vec3(), aText, aMaxWidth, aTextFormat, aAlignment);
    }

    void graphics_context::draw_multiline_glyph_text(vec3 const& aPoint, glyph_text const& aText, dimension aMaxWidth, text_format const& aTextFormat, alignment aAlignment) const
    {
        draw_multiline_glyph_text(aPoint, to_multiline_glyph_text(aText, aMaxWidth, aAlignment), aTextFormat);
    }

    void graphics_context::draw_multiline_glyph_text(point const& aPoint, multiline_glyph_text const& aText, text_format const& aTextFormat) const
    {
        draw_multiline_glyph_text(aPoint.to_vec3(), aText, aTextFormat);
    }

    void graphics_context::draw_multiline_glyph_text(vec3 const& aPoint, multiline_glyph_text const& aText, text_format const& aTextFormat) const
    {
        for (auto& line : aText.lines)
        {
            if (line.begin == line.end)
                continue;
            auto const glyphs = std::ranges::subrange(aText.glyphText.begin() + line.begin, aText.glyphText.begin() + line.end);
            auto const adjust = line.bbox[0] - vec3{ glyphs.begin()->cell[0] };
            draw_glyph_text(aPoint + adjust, aText.glyphText, glyphs.begin(), glyphs.end(), aTextFormat);
        }
    }

    subpixel_format graphics_context::subpixel_format() const
    {
        return native_context().subpixel_format();
    }

    bool graphics_context::metrics_available() const
    {
        return true;
    }

    size graphics_context::extents() const
    {
        return iExtents;
    }

    dimension graphics_context::horizontal_dpi() const
    {
        return iRenderTarget.horizontal_dpi();
    }

    dimension graphics_context::vertical_dpi() const
    {
        return iRenderTarget.vertical_dpi();
    }

    dimension graphics_context::ppi() const
    {
        return iRenderTarget.ppi();
    }

    dimension graphics_context::em_size() const
    {
        return static_cast<dimension>(iDefaultFont.size() / 72.0 * horizontal_dpi());
    }

    bool graphics_context::device_metrics_available() const
    {
        return device_metrics().metrics_available();
    }

    const i_device_metrics& graphics_context::device_metrics() const
    {
        return *this;
    }

    bool graphics_context::attached() const
    {
        return iType == type::Attached;
    }

    bool graphics_context::active() const
    {
        return iNativeGraphicsContext != nullptr;
    }

    i_rendering_context& graphics_context::native_context() const
    {
        if (attached())
        {
            if (!active())
            {
                if (iSrt == std::nullopt)
                    iSrt.emplace(render_target());
                iNativeGraphicsContext = iRenderTarget.create_graphics_context(iBlendingMode);
            }
            return *iNativeGraphicsContext;
        }
        throw unattached();
    }

    void graphics_context::flush() const
    {
        native_context().flush();
    }

    void graphics_context::set_viewport(optional_rect const& aViewport) const
    {
        if (aViewport)
            native_context().enqueue(graphics_operation::set_viewport{ to_device_units(aViewport.value() )});
        else
            native_context().enqueue(graphics_operation::set_viewport{});
    }

    void graphics_context::set_view_transforamtion(optional_mat33 const& aViewTransforamtion) const
    {
        if (aViewTransforamtion)
            native_context().enqueue(graphics_operation::set_view_transformation{ aViewTransforamtion.value() });
        else
            native_context().enqueue(graphics_operation::set_view_transformation{});
    }

    void graphics_context::scissor_on(rect const& aRect) const
    {
        native_context().enqueue(graphics_operation::scissor_on{ to_device_units(aRect) + iOrigin });
    }

    void graphics_context::scissor_off() const
    {
        native_context().enqueue(graphics_operation::scissor_off{});
    }

    bool graphics_context::snap_to_pixel() const
    {
        return iSnapToPixel;
    }

    void graphics_context::set_snap_to_pixel(bool aSnap) const
    {
        if (iSnapToPixel != aSnap)
        {
            iSnapToPixel = aSnap;
            if (snap_to_pixel())
                native_context().enqueue(graphics_operation::snap_to_pixel_on{});
            else
                native_context().enqueue(graphics_operation::snap_to_pixel_off{});
        }
    }

    double graphics_context::opacity() const
    {
        return iOpacity;
    }

    void graphics_context::set_opacity(double aOpacity) const
    {
        if (iOpacity != aOpacity)
        {
            iOpacity = aOpacity;
            native_context().enqueue(graphics_operation::set_opacity{ aOpacity });
        }
    }

    blending_mode graphics_context::blending_mode() const
    {
        return iBlendingMode;
    }

    void graphics_context::set_blending_mode(neogfx::blending_mode aBlendingMode) const
    {
        if (iBlendingMode != aBlendingMode)
        {
            iBlendingMode = aBlendingMode;
            native_context().enqueue(graphics_operation::set_blending_mode{ aBlendingMode });
        }
    }

    smoothing_mode graphics_context::smoothing_mode() const
    {
        return iSmoothingMode;
    }

    void graphics_context::set_smoothing_mode(neogfx::smoothing_mode aSmoothingMode) const
    {
        if (iSmoothingMode != aSmoothingMode)
        {
            iSmoothingMode = aSmoothingMode;
            native_context().enqueue(graphics_operation::set_smoothing_mode{ aSmoothingMode });
        }
    }

    void graphics_context::push_logical_operation(logical_operation aLogicalOperation) const
    {
        native_context().enqueue(graphics_operation::push_logical_operation{ aLogicalOperation });
    }

    void graphics_context::pop_logical_operation() const
    {
        native_context().enqueue(graphics_operation::pop_logical_operation{});
    }

    void graphics_context::line_stipple_on(scalar aFactor, uint16_t aPattern, scalar aPosition) const
    {
        native_context().enqueue(graphics_operation::line_stipple_on{ aFactor, aPattern, aPosition });
    }

    void graphics_context::line_stipple_off() const
    {
        native_context().enqueue(graphics_operation::line_stipple_off{});
    }

    bool graphics_context::is_subpixel_rendering_on() const
    {
        return iSubpixelRendering;
    }

    void graphics_context::subpixel_rendering_on() const
    {
        if (iSubpixelRendering != true)
        {
            iSubpixelRendering = true;
            native_context().enqueue(graphics_operation::subpixel_rendering_on{});
        }
    }

    void graphics_context::subpixel_rendering_off() const
    {
        if (iSubpixelRendering != false)
        {
            iSubpixelRendering = false;
            native_context().enqueue(graphics_operation::subpixel_rendering_off{});
        }
    }

    void graphics_context::clear(color const& aColor, std::optional<scalar> const& aZpos) const
    {
        if (aZpos == std::nullopt)
            native_context().enqueue(graphics_operation::clear{ aColor });
        else
            fill_rect(rect{ render_target().target_type() == render_target_type::Surface ? 
                point{ 0.0, 0.0, aZpos ? *aZpos : 0.0 } : point{ -1.0, -1.0 }, iRenderTarget.target_texture().storage_extents() }, aColor);
    }

    void graphics_context::clear_depth_buffer() const
    {
        native_context().enqueue(graphics_operation::clear_depth_buffer{});
    }

    void graphics_context::clear_stencil_buffer() const
    {
        native_context().enqueue(graphics_operation::clear_stencil_buffer{});
    }

    void graphics_context::blit(rect const& aDestinationRect, i_graphics_context const& aSource, rect const& aSourceRect) const
    {
        scoped_blending_mode sbm{ *this, neogfx::blending_mode::Blit };
        draw_texture(aDestinationRect, aSource.render_target().target_texture(), aSourceRect);
    }

    void blur(i_graphics_context const& aDestination, rect const& aDestinationRect, i_graphics_context const& aSource, rect const& aSourceRect, blurring_algorithm aAlgorithm, scalar aParameter1, scalar aParameter2)
    {
        scoped_render_target srt{ aDestination };
        auto mesh = aDestination.logical_coordinate_system() == logical_coordinate_system::AutomaticGui ?
            to_ecs_component(aDestinationRect) : to_ecs_component(game_rect{ aDestinationRect });
        auto const& source = aSource.render_target();
        for (auto& uv : mesh.uv)
            uv = (aSourceRect.top_left() / source.extents()).to_vec2() + uv.scale((aSourceRect.extents() / source.extents()).to_vec2());
        aDestination.draw_mesh(
            mesh,
            game::material
            {
                {},
                {},
                {},
                to_ecs_component(aSource.render_target().target_texture()),
                shader_effect::Filter
            },
            optional_mat44{},
            to_ecs_component(aAlgorithm, aParameter1, aParameter2));
    }

    void graphics_context::blur(rect const& aDestinationRect, const i_graphics_context& aSource, rect const& aSourceRect, dimension aRadius, blurring_algorithm aAlgorithm, scalar aParameter1, scalar aParameter2) const
    {
        scoped_render_target srt1{ *this };
        scoped_blending_mode sbm1{ *this, neogfx::blending_mode::Blit };
        scoped_scissor ss1{ *this, aDestinationRect };
        scoped_render_target srt2{ aSource };
        scoped_blending_mode sbm2{ aSource, neogfx::blending_mode::Blit };
        scoped_scissor ss2{ aSource, aSourceRect };
        int32_t passes = static_cast<int32_t>(aRadius);
        if (passes % 2 == 0)
            ++passes;
        for (int32_t pass = 0; pass < passes; ++pass)
        {
            if (pass % 2 == 0)
                neogfx::blur(*this, aDestinationRect, aSource, aSourceRect, aAlgorithm, aParameter1, aParameter2);
            else
                neogfx::blur(aSource, aSourceRect, *this, aDestinationRect, aAlgorithm, aParameter1, aParameter2);
        }
    }

    glyph_text graphics_context::to_glyph_text(std::string const& aText) const
    {
        return to_glyph_text(aText, default_font());
    }

    glyph_text graphics_context::to_glyph_text(std::string const& aText, font const& aFont) const
    {
        return to_glyph_text(aText.begin(), aText.end(), aFont);
    }

    glyph_text graphics_context::to_glyph_text(std::string const& aText, std::function<font(std::size_t)> aFontSelector) const
    {
        return to_glyph_text(aText.begin(), aText.end(), aFontSelector);
    }

    glyph_text graphics_context::to_glyph_text(std::string::const_iterator aTextBegin, std::string::const_iterator aTextEnd) const
    {
        return to_glyph_text(aTextBegin, aTextEnd, default_font());
    }

    glyph_text graphics_context::to_glyph_text(std::string::const_iterator aTextBegin, std::string::const_iterator aTextEnd, font const& aFont) const
    {
        return to_glyph_text(aTextBegin, aTextEnd, [&aFont](std::size_t) { return aFont; });
    }

    glyph_text graphics_context::to_glyph_text(std::string::const_iterator aTextBegin, std::string::const_iterator aTextEnd, std::function<font(std::size_t)> aFontSelector) const
    {
        return service<i_font_manager>().glyph_text_factory().to_glyph_text(*this, std::string_view{ aTextBegin, aTextEnd }, aFontSelector);
    }

    glyph_text graphics_context::to_glyph_text(std::u32string const& aText) const
    {
        return to_glyph_text(aText, default_font());
    }

    glyph_text graphics_context::to_glyph_text(std::u32string const& aText, font const& aFont) const
    {
        return to_glyph_text(aText.begin(), aText.end(), aFont);
    }

    glyph_text graphics_context::to_glyph_text(std::u32string const& aText, std::function<font(std::size_t)> aFontSelector) const
    {
        return to_glyph_text(aText.begin(), aText.end(), aFontSelector);
    }

    glyph_text graphics_context::to_glyph_text(std::u32string::const_iterator aTextBegin, std::u32string::const_iterator aTextEnd) const
    {
        return to_glyph_text(aTextBegin, aTextEnd, default_font());
    }

    glyph_text graphics_context::to_glyph_text(std::u32string::const_iterator aTextBegin, std::u32string::const_iterator aTextEnd, font const& aFont) const
    {
        return to_glyph_text(aTextBegin, aTextEnd, [&aFont](std::size_t) { return aFont; });
    }

    glyph_text graphics_context::to_glyph_text(std::u32string::const_iterator aTextBegin, std::u32string::const_iterator aTextEnd, std::function<font(std::size_t)> aFontSelector) const
    {
        return service<i_font_manager>().glyph_text_factory().to_glyph_text(*this, std::u32string_view{ aTextBegin, aTextEnd }, aFontSelector);
    }

    multiline_glyph_text graphics_context::to_multiline_glyph_text(std::string const& aText, dimension aMaxWidth, alignment aAlignment) const
    {
        return to_multiline_glyph_text(aText, default_font(), aMaxWidth, aAlignment);
    }

    multiline_glyph_text graphics_context::to_multiline_glyph_text(std::string const& aText, font const& aFont, dimension aMaxWidth, alignment aAlignment) const
    {
        return to_multiline_glyph_text(aText.begin(), aText.end(), aFont, aMaxWidth, aAlignment);
    }

    multiline_glyph_text graphics_context::to_multiline_glyph_text(std::string::const_iterator aTextBegin, std::string::const_iterator aTextEnd, dimension aMaxWidth, alignment aAlignment) const
    {
        return to_multiline_glyph_text(aTextBegin, aTextEnd, default_font(), aMaxWidth, aAlignment);
    }

    multiline_glyph_text graphics_context::to_multiline_glyph_text(std::string::const_iterator aTextBegin, std::string::const_iterator aTextEnd, font const& aFont, dimension aMaxWidth, alignment aAlignment) const
    {
        return to_multiline_glyph_text(aTextBegin, aTextEnd, [aFont](std::size_t) { return aFont; }, aMaxWidth, aAlignment);
    }

    multiline_glyph_text graphics_context::to_multiline_glyph_text(std::string::const_iterator aTextBegin, std::string::const_iterator aTextEnd, std::function<font(std::size_t)> aFontSelector, dimension aMaxWidth, alignment aAlignment) const
    {
        return to_multiline_glyph_text(to_glyph_text(aTextBegin, aTextEnd, aFontSelector), aMaxWidth, aAlignment);
    }

    multiline_glyph_text graphics_context::to_multiline_glyph_text(std::u32string const& aText, dimension aMaxWidth, alignment aAlignment) const
    {
        return to_multiline_glyph_text(aText, default_font(), aMaxWidth, aAlignment);
    }

    multiline_glyph_text graphics_context::to_multiline_glyph_text(std::u32string const& aText, font const& aFont, dimension aMaxWidth, alignment aAlignment) const
    {
        return to_multiline_glyph_text(aText.begin(), aText.end(), aFont, aMaxWidth, aAlignment);
    }

    multiline_glyph_text graphics_context::to_multiline_glyph_text(std::u32string::const_iterator aTextBegin, std::u32string::const_iterator aTextEnd, dimension aMaxWidth, alignment aAlignment) const
    {
        return to_multiline_glyph_text(aTextBegin, aTextEnd, default_font(), aMaxWidth, aAlignment);
    }

    multiline_glyph_text graphics_context::to_multiline_glyph_text(std::u32string::const_iterator aTextBegin, std::u32string::const_iterator aTextEnd, font const& aFont, dimension aMaxWidth, alignment aAlignment) const
    {
        return to_multiline_glyph_text(aTextBegin, aTextEnd, [&aFont](std::size_t) { return aFont; }, aMaxWidth, aAlignment);
    }

    multiline_glyph_text graphics_context::to_multiline_glyph_text(std::u32string::const_iterator aTextBegin, std::u32string::const_iterator aTextEnd, std::function<font(std::size_t)> aFontSelector, dimension aMaxWidth, alignment aAlignment) const
    {
        return to_multiline_glyph_text(to_glyph_text(aTextBegin, aTextEnd, aFontSelector), aMaxWidth, aAlignment);
    }

    multiline_glyph_text graphics_context::to_multiline_glyph_text(glyph_text const& aText, dimension aMaxWidth, alignment aAlignment) const
    {
        multiline_glyph_text result{ aText.clone() };
        typedef std::pair<glyph_text::iterator, glyph_text::iterator> line_t;
        typedef std::vector<line_t> lines_t;
        lines_t lines;
        std::array<glyph_char, 2> delimeters = { glyph_char{ '\r', {}, text_category::Whitespace }, glyph_char{ '\n', {}, text_category::Whitespace } };
        neolib::tokens(result.glyphText.begin(), result.glyphText.end(), delimeters.begin(), delimeters.end(), lines, 0, false);
        vec3 pos;
        dimension maxLineWidth = 0.0;
        for (lines_t::const_iterator i = lines.begin(); i != lines.end(); ++i)
        {
            auto const& line = (logical_coordinates().is_gui_orientation() ? *i : *(lines.rbegin() + (i - lines.begin())));
            glyph_text::iterator lineStart = line.first;
            glyph_text::iterator lineEnd = line.second;
            if (lineStart != lineEnd)
            {
                if (aMaxWidth == 0)
                {
                    auto const& glyphs = std::ranges::subrange(lineStart, lineEnd);
                    result.lines.push_back(multiline_glyph_text::line{
                        {},
                        std::distance(result.glyphText.begin(), lineStart), std::distance(result.glyphText.begin(), lineEnd) });
                    auto const& firstGlyph = *glyphs.begin();
                    auto const xAdjust = static_cast<float>(-firstGlyph.cell[0].x);
                    for (auto& glyph : glyphs)
                        glyph.cell += vec2f{ xAdjust, static_cast<float>(pos.y) };
                    size lineExtent = from_device_units(result.glyphText.extents(lineStart, lineEnd));
                    maxLineWidth = std::max(maxLineWidth, lineExtent.cx);
                    pos.y += lineExtent.cy;
                }
                else
                {
                    dimension maxWidth = to_device_units(size(aMaxWidth, 0.0)).cx;
                    glyph_text::iterator next = lineStart;
                    while (next != line.second)
                    {
                        bool gotLine = false;
                        if (next->cell[1].x - lineStart->cell[0].x > maxWidth)
                        {
                            if (next != lineStart)
                            {
                                std::pair<glyph_text::iterator, glyph_text::iterator> wordBreak = result.glyphText.word_break(lineStart, next);
                                lineEnd = wordBreak.first;
                                next = wordBreak.second;
                                if (lineEnd == next)
                                {
                                    while (lineEnd != line.second && (lineEnd + 1)->clusters == wordBreak.first->clusters)
                                        ++lineEnd;
                                    next = lineEnd;
                                }
                            }
                            else
                                ++next;
                            gotLine = true;
                        }
                        else
                            ++next;
                        if (gotLine || next == line.second)
                        {
                            auto const& glyphs = std::ranges::subrange(lineStart, lineEnd);
                            result.lines.push_back(multiline_glyph_text::line{
                                {},
                                std::distance(result.glyphText.begin(), lineStart), std::distance(result.glyphText.begin(), lineEnd) });
                            auto const xAdjust = static_cast<float>(-lineStart->cell[0].x);
                            for (auto& glyph : glyphs)
                                glyph.cell += vec2f{ xAdjust, static_cast<float>(pos.y) };
                            size lineExtent = from_device_units(result.glyphText.extents(lineStart, lineEnd));
                            maxLineWidth = std::max(maxLineWidth, lineExtent.cx);
                            pos.y += lineExtent.cy;
                            lineStart = next;
                            lineEnd = line.second;
                        }
                    }
                }
            }
            else
                pos.y += aText.glyph_font().height();
        }

        std::optional<vec3> min;
        std::optional<vec3> max;

        for (auto& line : result.lines)
        {
            auto const glyphs = std::ranges::subrange(result.glyphText.begin() + line.begin, result.glyphText.begin() + line.end);
            auto const textDirection = glyph_text_direction(glyphs.begin(), glyphs.end());
            line.bbox[0] = glyphs.begin()->cell[0];
            line.bbox[1] = std::prev(glyphs.end())->cell[1];
            line.bbox[2] = std::prev(glyphs.end())->cell[2];
            line.bbox[3] = glyphs.begin()->cell[3];
            if (!min)
                min = vec3{ line.bbox[0] };
            else
                min = min->min(vec3{ line.bbox[0] });
            if (!max)
                max = vec3{ line.bbox[2] };
            else
                max = max->max(vec3{ line.bbox[2] });
            if ((aAlignment == alignment::Left && textDirection == text_direction::RTL) || (aAlignment == alignment::Right && textDirection == text_direction::LTR))
            {
                auto const adjust = maxLineWidth - from_device_units(size{ line.bbox[1].x - line.bbox[0].x, 0.0 }).cx;
                line.bbox += vec3{ adjust, 0.0 };
            }
            else if (aAlignment == alignment::Center)
            {
                auto const adjust = (maxLineWidth - from_device_units(size{ line.bbox[1].x - line.bbox[0].x, 0.0 }).cx) / 2;
                line.bbox += vec3{ adjust, 0.0 };
            }
        }

        if (min && max)
        {
            result.bbox[0] = *min;
            result.bbox[1] = vec3{ max->x, min->y };
            result.bbox[2] = *max;
            result.bbox[3] = vec3{ min->x, max->y };
        }

        return result;
    }

    void graphics_context::draw_glyph(point const& aPoint, glyph_text const& aText, glyph_char const& aGlyphChar, text_format const& aTextFormat) const
    {
        draw_glyph(aPoint.to_vec3(), aText, aGlyphChar, aTextFormat);
    }

    void graphics_context::draw_glyph(vec3 const& aPoint, glyph_text const& aText, glyph_char const& aGlyphChar, text_format const& aTextFormat) const
    {
        auto adjustedPos = (to_device_units(point{ aPoint }) + iOrigin).to_vec3() + vec3{ 0.0, 0.0, aPoint.z };
        native_context().enqueue(graphics_operation::draw_glyphs{ adjustedPos, aText, &aGlyphChar, std::next(&aGlyphChar), text_format_span{ 0, 1, aTextFormat }, mnemonics_shown() });
    }

    void graphics_context::draw_glyphs(point const& aPoint, glyph_text const& aText, text_format_spans const& aSpans) const
    {
        draw_glyphs(aPoint.to_vec3(), aText, aSpans);
    }

    void graphics_context::draw_glyphs(vec3 const& aPoint, glyph_text const& aText, text_format_spans const& aSpans) const
    {
        auto adjustedPos = (to_device_units(point{ aPoint }) + iOrigin).to_vec3() + vec3{ 0.0, 0.0, aPoint.z };
        native_context().enqueue(graphics_operation::draw_glyphs{ adjustedPos, aText, aText.begin(), aText.end(), aSpans, mnemonics_shown()});
    }

    char graphics_context::mnemonic() const
    {
        if (mnemonic_set())
            return iMnemonic->second;
        return '&';
    }

    bool graphics_context::mnemonic_set() const
    {
        return iMnemonic != std::nullopt;
    }

    void graphics_context::set_mnemonic(bool aShowMnemonics, char aMnemonicPrefix) const
    {
        iMnemonic = std::make_pair(aShowMnemonics, aMnemonicPrefix);
    }

    void graphics_context::unset_mnemonic() const
    {
        iMnemonic = std::nullopt;
    }

    bool graphics_context::mnemonics_shown() const
    {
        return iMnemonic != std::nullopt && iMnemonic->first;
    }

    bool graphics_context::password() const
    {
        return iPassword != std::nullopt;
    }

    std::string const& graphics_context::password_mask() const
    {
        if (password())
        {
            if (iPassword->empty())
                iPassword = "\xE2\x97\x8F";
            return *iPassword;
        }
        throw password_not_set();
    }

    void graphics_context::set_password(bool aPassword, std::string const& aMask)
    {
        if (aPassword)
            iPassword = aMask;
        else
            iPassword = std::nullopt;
    }

    void graphics_context::draw_texture(point const& aPoint, i_texture const& aTexture, color_or_gradient const& aColor, shader_effect aShaderEffect) const
    {
        draw_texture(rect{ aPoint, aTexture.extents() }, aTexture, with_bounding_box(aColor, rect{ aPoint, aTexture.extents() }), aShaderEffect);
    }

    void graphics_context::draw_texture(rect const& aRect, i_texture const& aTexture, color_or_gradient const& aColor, shader_effect aShaderEffect) const
    {
        if (logical_coordinates().is_gui_orientation())
            draw_texture(to_ecs_component(aRect), aTexture, with_bounding_box(aColor, aRect), aShaderEffect);
        else
            draw_texture(to_ecs_component(game_rect{ aRect }), aTexture, with_bounding_box(aColor, game_rect{ aRect }), aShaderEffect);
    }

    void graphics_context::draw_texture(point const& aPoint, i_texture const& aTexture, rect const& aTextureRect, color_or_gradient const& aColor, shader_effect aShaderEffect) const
    {
        draw_texture(rect{ aPoint, aTextureRect.extents() }, aTexture, aTextureRect, with_bounding_box(aColor, rect{ aPoint, aTextureRect.extents() }), aShaderEffect);
    }

    void graphics_context::draw_texture(rect const& aRect, i_texture const& aTexture, rect const& aTextureRect, color_or_gradient const& aColor, shader_effect aShaderEffect) const
    {
        if (logical_coordinates().is_gui_orientation())
            draw_texture(to_ecs_component(aRect), aTexture, aTextureRect, with_bounding_box(aColor, aRect), aShaderEffect);
        else
            draw_texture(to_ecs_component(game_rect{ aRect }), aTexture, aTextureRect, with_bounding_box(aColor, game_rect{ aRect }), aShaderEffect);
    }

    void graphics_context::draw_texture(game::mesh const& aMesh, i_texture const& aTexture, color_or_gradient const& aColor, shader_effect aShaderEffect) const
    {
        draw_mesh(
            aMesh, 
            game::material
            { 
                std::holds_alternative<color>(aColor) ? game::color{ to_ecs_component(std::get<color>(aColor)) } : std::optional<game::color>{},
                std::holds_alternative<gradient>(aColor) ? game::gradient{ to_ecs_component(std::get<gradient>(aColor)) } : std::optional<game::gradient>{},
                {},
                to_ecs_component(aTexture),
                aShaderEffect
            },
            optional_mat44{});
    }

    void graphics_context::draw_texture(game::mesh const& aMesh, i_texture const& aTexture, rect const& aTextureRect, color_or_gradient const& aColor, shader_effect aShaderEffect) const
    {
        auto adjustedMesh = aMesh;
        for (auto& uv : adjustedMesh.uv)
            uv = (aTextureRect.top_left() / aTexture.extents()).to_vec2() + uv.scale((aTextureRect.extents() / aTexture.extents()).to_vec2());
        draw_texture(adjustedMesh, aTexture, aColor, aShaderEffect);
    }

    void graphics_context::draw_mesh(game::mesh const& aMesh, game::material const& aMaterial, optional_mat44 const& aTransformation, std::optional<game::filter> const& aFilter) const
    {
        vec2 const toDeviceUnits = to_device_units(vec2{ 1.0, 1.0 });
        native_context().enqueue(
            graphics_operation::draw_mesh{
                aMesh,
                aMaterial,
                mat44{
                    { toDeviceUnits.x, 0.0, 0.0, 0.0 },
                    { 0.0, toDeviceUnits.y, 0.0, 0.0 },
                    { 0.0, 0.0, 1.0, 0.0 },
                    { iOrigin.x, iOrigin.y, 0.0, 1.0 } } * (aTransformation != std::nullopt ? *aTransformation : mat44::identity()),
                aFilter
            });
    }
}