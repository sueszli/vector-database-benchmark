/*****************************************************************************
 * Copyright (c) 2014-2023 OpenRCT2 developers
 *
 * For a complete list of all authors, please refer to contributors.md
 * Interested in contributing? Visit https://github.com/OpenRCT2/OpenRCT2
 *
 * OpenRCT2 is licensed under the GNU General Public License version 3.
 *****************************************************************************/

#include <algorithm>
#include <openrct2-ui/interface/LandTool.h>
#include <openrct2-ui/interface/Widget.h>
#include <openrct2-ui/windows/Window.h>
#include <openrct2/Context.h>
#include <openrct2/localisation/Formatter.h>
#include <openrct2/localisation/Localisation.h>
#include <openrct2/world/Park.h>
#include <openrct2/world/Scenery.h>

enum WindowClearSceneryWidgetIdx
{
    WIDX_BACKGROUND,
    WIDX_TITLE,
    WIDX_CLOSE,
    WIDX_PREVIEW,
    WIDX_DECREMENT,
    WIDX_INCREMENT,
    WIDX_SMALL_SCENERY,
    WIDX_LARGE_SCENERY,
    WIDX_FOOTPATH
};
// clang-format on
static constexpr StringId WINDOW_TITLE = STR_CLEAR_SCENERY;
static constexpr int32_t WW = 98;
static constexpr int32_t WH = 94;

static constexpr ScreenSize CLEAR_SCENERY_BUTTON = { 24, 24 };

static Widget window_clear_scenery_widgets[] = {
    WINDOW_SHIM(WINDOW_TITLE, WW, WH),
    MakeWidget(
        { 27, 17 }, { 44, 32 }, WindowWidgetType::ImgBtn, WindowColour::Primary, SPR_LAND_TOOL_SIZE_0, STR_NONE), // preview box
    MakeRemapWidget(
        { 28, 18 }, { 16, 16 }, WindowWidgetType::TrnBtn, WindowColour::Secondary, SPR_LAND_TOOL_DECREASE,
        STR_ADJUST_SMALLER_LAND_TIP), // decrement size
    MakeRemapWidget(
        { 54, 32 }, { 16, 16 }, WindowWidgetType::TrnBtn, WindowColour::Secondary, SPR_LAND_TOOL_INCREASE,
        STR_ADJUST_LARGER_LAND_TIP), // increment size
    MakeRemapWidget(
        { 7, 53 }, CLEAR_SCENERY_BUTTON, WindowWidgetType::FlatBtn, WindowColour::Secondary, SPR_G2_BUTTON_TREES,
        STR_CLEAR_SCENERY_REMOVE_SMALL_SCENERY_TIP), // small scenery
    MakeRemapWidget(
        { 37, 53 }, CLEAR_SCENERY_BUTTON, WindowWidgetType::FlatBtn, WindowColour::Secondary, SPR_G2_BUTTON_LARGE_SCENERY,
        STR_CLEAR_SCENERY_REMOVE_LARGE_SCENERY_TIP), // large scenery
    MakeRemapWidget(
        { 67, 53 }, CLEAR_SCENERY_BUTTON, WindowWidgetType::FlatBtn, WindowColour::Secondary, SPR_G2_BUTTON_FOOTPATH,
        STR_CLEAR_SCENERY_REMOVE_FOOTPATHS_TIP), // footpaths
    WIDGETS_END,
};

class CleanSceneryWindow final : public Window
{
public:
    void OnOpen() override
    {
        widgets = window_clear_scenery_widgets;
        hold_down_widgets = (1uLL << WIDX_INCREMENT) | (1uLL << WIDX_DECREMENT);
        WindowInitScrollWidgets(*this);
        WindowPushOthersBelow(*this);

        gLandToolSize = 2;
        gClearSceneryCost = MONEY64_UNDEFINED;

        gClearSmallScenery = true;
        gClearLargeScenery = false;
        gClearFootpath = false;

        Invalidate();
    }

    void OnClose() override
    {
        if (ClearSceneryToolIsActive())
            ToolCancel();
    }

    void OnMouseUp(const WidgetIndex widgetIndex) override
    {
        switch (widgetIndex)
        {
            case WIDX_CLOSE:
                Close();
                break;
            case WIDX_PREVIEW:
            {
                Formatter ft;
                ft.Add<int16_t>(MINIMUM_TOOL_SIZE);
                ft.Add<int16_t>(MAXIMUM_TOOL_SIZE);
                TextInputOpen(WIDX_PREVIEW, STR_SELECTION_SIZE, STR_ENTER_SELECTION_SIZE, ft, STR_NONE, STR_NONE, 3);
                break;
            }
            case WIDX_SMALL_SCENERY:
                gClearSmallScenery ^= 1;
                Invalidate();
                break;
            case WIDX_LARGE_SCENERY:
                gClearLargeScenery ^= 1;
                Invalidate();
                break;
            case WIDX_FOOTPATH:
                gClearFootpath ^= 1;
                Invalidate();
                break;
        }
    }

    void OnMouseDown(const WidgetIndex widgetIndex) override
    {
        switch (widgetIndex)
        {
            case WIDX_DECREMENT:
                // Decrement land tool size, if it stays within the limit
                gLandToolSize = std::max(MINIMUM_TOOL_SIZE, gLandToolSize - 1);

                // Invalidate the window
                Invalidate();
                break;
            case WIDX_INCREMENT:
                // Increment land tool size, if it stays within the limit
                gLandToolSize = std::min(MAXIMUM_TOOL_SIZE, gLandToolSize + 1);

                // Invalidate the window
                Invalidate();
                break;
        }
    }

    void OnTextInput(const WidgetIndex widgetIndex, const std::string_view text) override
    {
        if (widgetIndex != WIDX_PREVIEW || text.empty())
            return;

        try
        {
            int32_t size = std::stol(std::string(text));
            size = std::clamp(size, MINIMUM_TOOL_SIZE, MAXIMUM_TOOL_SIZE);
            gLandToolSize = size;
            Invalidate();
        }
        catch (const std::logic_error&)
        {
            // std::stol can throw std::out_of_range or std::invalid_argument
        }
    }

    void OnUpdate() override
    {
        frame_no++;
        // Close window if another tool is open
        if (!ClearSceneryToolIsActive())
            Close();
    }

    void Invalidate()
    {
        // Set the preview image button to be pressed down
        pressed_widgets = (1uLL << WIDX_PREVIEW) | (gClearSmallScenery ? (1uLL << WIDX_SMALL_SCENERY) : 0)
            | (gClearLargeScenery ? (1uLL << WIDX_LARGE_SCENERY) : 0) | (gClearFootpath ? (1uLL << WIDX_FOOTPATH) : 0);

        // Update the preview image (for tool sizes up to 7)
        window_clear_scenery_widgets[WIDX_PREVIEW].image = ImageId(LandTool::SizeToSpriteIndex(gLandToolSize));
    }

    void OnDraw(DrawPixelInfo& dpi) override
    {
        DrawWidgets(dpi);

        // Draw number for tool sizes bigger than 7
        ScreenCoordsXY screenCoords = { windowPos.x + window_clear_scenery_widgets[WIDX_PREVIEW].midX(),
                                        windowPos.y + window_clear_scenery_widgets[WIDX_PREVIEW].midY() };
        if (gLandToolSize > MAX_TOOL_SIZE_WITH_SPRITE)
        {
            auto ft = Formatter();
            ft.Add<uint16_t>(gLandToolSize);
            DrawTextBasic(dpi, screenCoords - ScreenCoordsXY{ 0, 2 }, STR_LAND_TOOL_SIZE_VALUE, ft, { TextAlignment::CENTRE });
        }

        // Draw cost amount
        if (gClearSceneryCost != MONEY64_UNDEFINED && gClearSceneryCost != 0 && !(gParkFlags & PARK_FLAGS_NO_MONEY))
        {
            auto ft = Formatter();
            ft.Add<money64>(gClearSceneryCost);
            screenCoords.x = window_clear_scenery_widgets[WIDX_PREVIEW].midX() + windowPos.x;
            screenCoords.y = window_clear_scenery_widgets[WIDX_PREVIEW].bottom + windowPos.y + 5 + 27;
            DrawTextBasic(dpi, screenCoords, STR_COST_AMOUNT, ft, { TextAlignment::CENTRE });
        }
    }

    void OnResize() override
    {
        ResizeFrame();
    }
};

WindowBase* WindowClearSceneryOpen()
{
    auto* w = static_cast<CleanSceneryWindow*>(WindowBringToFrontByClass(WindowClass::ClearScenery));

    if (w != nullptr)
        return w;

    w = WindowCreate<CleanSceneryWindow>(WindowClass::ClearScenery, ScreenCoordsXY(ContextGetWidth() - WW, 29), WW, WH, 0);

    if (w != nullptr)
        return w;

    return nullptr;
}
