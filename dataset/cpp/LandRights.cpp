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
#include <openrct2-ui/interface/Viewport.h>
#include <openrct2-ui/interface/Widget.h>
#include <openrct2-ui/windows/Window.h>
#include <openrct2/Context.h>
#include <openrct2/Game.h>
#include <openrct2/Input.h>
#include <openrct2/actions/LandBuyRightsAction.h>
#include <openrct2/core/String.hpp>
#include <openrct2/drawing/Drawing.h>
#include <openrct2/localisation/Formatter.h>
#include <openrct2/localisation/Localisation.h>
#include <openrct2/world/Park.h>

static constexpr StringId WINDOW_TITLE = STR_LAND_RIGHTS;
static constexpr int32_t WH = 94;
static constexpr int32_t WW = 98;

// clang-format off
enum WindowLandRightsWidgetIdx {
    WIDX_BACKGROUND,
    WIDX_TITLE,
    WIDX_CLOSE,
    WIDX_PREVIEW,
    WIDX_DECREMENT,
    WIDX_INCREMENT,
    WIDX_BUY_LAND_RIGHTS,
    WIDX_BUY_CONSTRUCTION_RIGHTS
};

static Widget window_land_rights_widgets[] = {
    WINDOW_SHIM(WINDOW_TITLE, WW, WH),
    MakeWidget     ({27, 17}, {44, 32}, WindowWidgetType::ImgBtn,  WindowColour::Primary , ImageId(SPR_LAND_TOOL_SIZE_0)                                           ), // preview box
    MakeRemapWidget({28, 18}, {16, 16}, WindowWidgetType::TrnBtn,  WindowColour::Tertiary, SPR_LAND_TOOL_DECREASE,      STR_ADJUST_SMALLER_LAND_RIGHTS_TIP), // decrement size
    MakeRemapWidget({54, 32}, {16, 16}, WindowWidgetType::TrnBtn,  WindowColour::Tertiary, SPR_LAND_TOOL_INCREASE,      STR_ADJUST_LARGER_LAND_RIGHTS_TIP ), // increment size
    MakeRemapWidget({22, 53}, {24, 24}, WindowWidgetType::FlatBtn, WindowColour::Tertiary, SPR_BUY_LAND_RIGHTS,         STR_BUY_LAND_RIGHTS_TIP           ), // land rights
    MakeRemapWidget({52, 53}, {24, 24}, WindowWidgetType::FlatBtn, WindowColour::Tertiary, SPR_BUY_CONSTRUCTION_RIGHTS, STR_BUY_CONSTRUCTION_RIGHTS_TIP   ), // construction rights
    WIDGETS_END,
};
// clang-format on

constexpr uint8_t LAND_RIGHTS_MODE_BUY_CONSTRUCTION_RIGHTS = 0;
constexpr uint8_t LAND_RIGHTS_MODE_BUY_LAND = 1;

class LandRightsWindow final : public Window
{
public:
    void OnOpen() override
    {
        widgets = window_land_rights_widgets;
        hold_down_widgets = (1uLL << WIDX_INCREMENT) | (1uLL << WIDX_DECREMENT);
        WindowInitScrollWidgets(*this);
        WindowPushOthersBelow(*this);
        _landRightsMode = LAND_RIGHTS_MODE_BUY_LAND;
        pressed_widgets = (1uLL << WIDX_BUY_LAND_RIGHTS);

        gLandToolSize = 1;

        ShowGridlines();
        ToolSet(*this, WIDX_BUY_LAND_RIGHTS, Tool::UpArrow);
        InputSetFlag(INPUT_FLAG_6, true);

        ShowLandRights();

        if (gLandRemainingConstructionSales == 0)
        {
            ShowConstructionRights();
        }
    }

    void OnClose() override
    {
        HideGridlines();
        if (gLandRemainingConstructionSales == 0)
        {
            HideConstructionRights();
        }

        // If the tool wasn't changed, turn tool off
        if (LandRightsToolIsActive())
            ToolCancel();
    }

    void OnMouseUp(WidgetIndex widgetIndex) override
    {
        switch (widgetIndex)
        {
            case WIDX_CLOSE:
                Close();
                break;
            case WIDX_PREVIEW:
                InputSize();
                break;
            case WIDX_BUY_LAND_RIGHTS:
                if (_landRightsMode != LAND_RIGHTS_MODE_BUY_LAND)
                {
                    ToolSet(*this, WIDX_BUY_LAND_RIGHTS, Tool::UpArrow);
                    _landRightsMode = LAND_RIGHTS_MODE_BUY_LAND;
                    ShowLandRights();
                    Invalidate();
                }
                break;
            case WIDX_BUY_CONSTRUCTION_RIGHTS:
                if (_landRightsMode != LAND_RIGHTS_MODE_BUY_CONSTRUCTION_RIGHTS)
                {
                    ToolSet(*this, WIDX_BUY_CONSTRUCTION_RIGHTS, Tool::UpArrow);
                    _landRightsMode = LAND_RIGHTS_MODE_BUY_CONSTRUCTION_RIGHTS;
                    ShowConstructionRights();
                    Invalidate();
                }
                break;
        }
    }

    void OnMouseDown(WidgetIndex widgetIndex) override
    {
        switch (widgetIndex)
        {
            case WIDX_DECREMENT:
                // Decrement land rights tool size
                gLandToolSize = std::max(MINIMUM_TOOL_SIZE, gLandToolSize - 1);

                // Invalidate the window
                Invalidate();
                break;
            case WIDX_INCREMENT:
                // Decrement land rights tool size
                gLandToolSize = std::min(MAXIMUM_TOOL_SIZE, gLandToolSize + 1);

                // Invalidate the window
                Invalidate();
                break;
        }
    }

    void OnTextInput(WidgetIndex widgetIndex, std::string_view text) override
    {
        if (text.empty())
            return;

        if (widgetIndex != WIDX_PREVIEW)
            return;

        const auto res = String::Parse<int32_t>(text);
        if (res.has_value())
        {
            int32_t size;
            size = res.value();
            size = std::max(MINIMUM_TOOL_SIZE, size);
            size = std::min(MAXIMUM_TOOL_SIZE, size);
            gLandToolSize = size;
            Invalidate();
        }
    }

    void OnUpdate() override
    {
        frame_no++;
        // Close window if another tool is open
        if (!LandRightsToolIsActive())
            Close();
    }

    void OnPrepareDraw() override
    {
        SetWidgetPressed(WIDX_PREVIEW, true);
        if (_landRightsMode == LAND_RIGHTS_MODE_BUY_LAND)
        {
            SetWidgetPressed(WIDX_BUY_LAND_RIGHTS, true);
            SetWidgetPressed(WIDX_BUY_CONSTRUCTION_RIGHTS, false);
        }
        else if (_landRightsMode == LAND_RIGHTS_MODE_BUY_CONSTRUCTION_RIGHTS)
        {
            SetWidgetPressed(WIDX_BUY_LAND_RIGHTS, false);
            SetWidgetPressed(WIDX_BUY_CONSTRUCTION_RIGHTS, true);
        }

        window_land_rights_widgets[WIDX_PREVIEW].image = ImageId(LandTool::SizeToSpriteIndex(gLandToolSize));

        if (gLandRemainingOwnershipSales == 0)
        {
            SetWidgetDisabled(WIDX_BUY_LAND_RIGHTS, true);
            window_land_rights_widgets[WIDX_BUY_LAND_RIGHTS].tooltip = STR_NO_LAND_RIGHTS_FOR_SALE_TIP;
        }
        else
        {
            SetWidgetDisabled(WIDX_BUY_LAND_RIGHTS, false);
            window_land_rights_widgets[WIDX_BUY_LAND_RIGHTS].tooltip = STR_BUY_LAND_RIGHTS_TIP;
        }

        if (gLandRemainingConstructionSales == 0)
        {
            SetWidgetDisabled(WIDX_BUY_CONSTRUCTION_RIGHTS, true);
            window_land_rights_widgets[WIDX_BUY_CONSTRUCTION_RIGHTS].tooltip = STR_NO_CONSTRUCTION_RIGHTS_FOR_SALE_TIP;
        }
        else
        {
            SetWidgetDisabled(WIDX_BUY_CONSTRUCTION_RIGHTS, false);
            window_land_rights_widgets[WIDX_BUY_CONSTRUCTION_RIGHTS].tooltip = STR_BUY_CONSTRUCTION_RIGHTS_TIP;
        }
    }

    void OnDraw(DrawPixelInfo& dpi) override
    {
        auto screenCoords = ScreenCoordsXY{ windowPos.x + window_land_rights_widgets[WIDX_PREVIEW].midX(),
                                            windowPos.y + window_land_rights_widgets[WIDX_PREVIEW].midY() };

        DrawWidgets(dpi);
        // Draw number for tool sizes bigger than 7
        if (gLandToolSize > MAX_TOOL_SIZE_WITH_SPRITE)
        {
            auto ft = Formatter();
            ft.Add<uint16_t>(gLandToolSize);
            DrawTextBasic(dpi, screenCoords - ScreenCoordsXY{ 0, 2 }, STR_LAND_TOOL_SIZE_VALUE, ft, { TextAlignment::CENTRE });
        }

        // Draw cost amount
        if (_landRightsCost != MONEY64_UNDEFINED && _landRightsCost != 0 && !(gParkFlags & PARK_FLAGS_NO_MONEY))
        {
            auto ft = Formatter();
            ft.Add<money64>(_landRightsCost);
            screenCoords = { window_land_rights_widgets[WIDX_PREVIEW].midX() + windowPos.x,
                             window_land_rights_widgets[WIDX_PREVIEW].bottom + windowPos.y + 32 };
            DrawTextBasic(dpi, screenCoords, STR_COST_AMOUNT, ft, { TextAlignment::CENTRE });
        }
    }

    void OnToolUpdate(WidgetIndex widgetIndex, const ScreenCoordsXY& screenCoords) override
    {
        MapInvalidateSelectionRect();
        gMapSelectFlags &= ~MAP_SELECT_FLAG_ENABLE;

        auto mapTile = ScreenGetMapXY(screenCoords, nullptr);

        if (!mapTile.has_value())
        {
            if (_landRightsCost != MONEY64_UNDEFINED)
            {
                _landRightsCost = MONEY64_UNDEFINED;
                WindowInvalidateByClass(WindowClass::ClearScenery);
            }
            return;
        }

        uint8_t state_changed = 0;

        if (!(gMapSelectFlags & MAP_SELECT_FLAG_ENABLE))
        {
            gMapSelectFlags |= MAP_SELECT_FLAG_ENABLE;
            state_changed++;
        }

        if (gMapSelectType != MAP_SELECT_TYPE_FULL)
        {
            gMapSelectType = MAP_SELECT_TYPE_FULL;
            state_changed++;
        }

        int16_t tool_size = gLandToolSize;
        if (tool_size == 0)
            tool_size = 1;

        int16_t tool_length = (tool_size - 1) * 32;

        // Move to tool bottom left
        mapTile->x -= (tool_size - 1) * 16;
        mapTile->y -= (tool_size - 1) * 16;
        mapTile = mapTile->ToTileStart();

        if (gMapSelectPositionA.x != mapTile->x)
        {
            gMapSelectPositionA.x = mapTile->x;
            state_changed++;
        }

        if (gMapSelectPositionA.y != mapTile->y)
        {
            gMapSelectPositionA.y = mapTile->y;
            state_changed++;
        }

        mapTile->x += tool_length;
        mapTile->y += tool_length;

        if (gMapSelectPositionB.x != mapTile->x)
        {
            gMapSelectPositionB.x = mapTile->x;
            state_changed++;
        }

        if (gMapSelectPositionB.y != mapTile->y)
        {
            gMapSelectPositionB.y = mapTile->y;
            state_changed++;
        }

        MapInvalidateSelectionRect();
        if (!state_changed)
            return;

        auto landBuyRightsAction = LandBuyRightsAction(
            { gMapSelectPositionA.x, gMapSelectPositionA.y, gMapSelectPositionB.x, gMapSelectPositionB.y },
            (_landRightsMode == LAND_RIGHTS_MODE_BUY_LAND) ? LandBuyRightSetting::BuyLand
                                                           : LandBuyRightSetting::BuyConstructionRights);
        auto res = GameActions::Query(&landBuyRightsAction);

        _landRightsCost = res.Error == GameActions::Status::Ok ? res.Cost : MONEY64_UNDEFINED;
    }

    void OnToolAbort(WidgetIndex widgetIndex) override
    {
        if (_landRightsMode == LAND_RIGHTS_MODE_BUY_LAND)
        {
            HideLandRights();
        }
        else
        {
            HideConstructionRights();
        }
    }

    void OnToolDown(WidgetIndex widgetIndex, const ScreenCoordsXY& screenCoords) override
    {
        if (_landRightsMode == LAND_RIGHTS_MODE_BUY_LAND)
        {
            if (screenCoords.x != LOCATION_NULL)
            {
                auto landBuyRightsAction = LandBuyRightsAction(
                    { gMapSelectPositionA.x, gMapSelectPositionA.y, gMapSelectPositionB.x, gMapSelectPositionB.y },
                    LandBuyRightSetting::BuyLand);
                GameActions::Execute(&landBuyRightsAction);
            }
        }
        else
        {
            if (screenCoords.x != LOCATION_NULL)
            {
                auto landBuyRightsAction = LandBuyRightsAction(
                    { gMapSelectPositionA.x, gMapSelectPositionA.y, gMapSelectPositionB.x, gMapSelectPositionB.y },
                    LandBuyRightSetting::BuyConstructionRights);
                GameActions::Execute(&landBuyRightsAction);
            }
        }
    }

    void OnToolDrag(WidgetIndex widgetIndex, const ScreenCoordsXY& screenCoords) override
    {
        if (_landRightsMode == LAND_RIGHTS_MODE_BUY_LAND)
        {
            if (screenCoords.x != LOCATION_NULL)
            {
                auto landBuyRightsAction = LandBuyRightsAction(
                    { gMapSelectPositionA.x, gMapSelectPositionA.y, gMapSelectPositionB.x, gMapSelectPositionB.y },
                    LandBuyRightSetting::BuyLand);
                GameActions::Execute(&landBuyRightsAction);
            }
        }
        else
        {
            if (screenCoords.x != LOCATION_NULL)
            {
                auto landBuyRightsAction = LandBuyRightsAction(
                    { gMapSelectPositionA.x, gMapSelectPositionA.y, gMapSelectPositionB.x, gMapSelectPositionB.y },
                    LandBuyRightSetting::BuyConstructionRights);
                GameActions::Execute(&landBuyRightsAction);
            }
        }
    }

    void OnResize() override
    {
        ResizeFrame();
    }

private:
    uint8_t _landRightsMode;
    money64 _landRightsCost;

    void InputSize()
    {
        Formatter ft;
        ft.Add<int16_t>(MINIMUM_TOOL_SIZE);
        ft.Add<int16_t>(MAXIMUM_TOOL_SIZE);
        WindowTextInputOpen(this, WIDX_PREVIEW, STR_SELECTION_SIZE, STR_ENTER_SELECTION_SIZE, ft, STR_NONE, STR_NONE, 3);
    }

    bool LandRightsToolIsActive()
    {
        if (!(InputTestFlag(INPUT_FLAG_TOOL_ACTIVE)))
            return false;
        if (gCurrentToolWidget.window_classification != WindowClass::LandRights)
            return false;
        return true;
    }
};

WindowBase* WindowLandRightsOpen()
{
    return WindowFocusOrCreate<LandRightsWindow>(
        WindowClass::LandRights, ScreenCoordsXY(ContextGetWidth() - WW, 29), WW, WH, 0);
}
