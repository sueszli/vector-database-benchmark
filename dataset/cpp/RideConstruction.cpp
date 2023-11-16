/*****************************************************************************
 * Copyright (c) 2014-2023 OpenRCT2 developers
 *
 * For a complete list of all authors, please refer to contributors.md
 * Interested in contributing? Visit https://github.com/OpenRCT2/OpenRCT2
 *
 * OpenRCT2 is licensed under the GNU General Public License version 3.
 *****************************************************************************/

#include "../ride/Construction.h"

#include <algorithm>
#include <limits>
#include <openrct2-ui/interface/Dropdown.h>
#include <openrct2-ui/interface/Viewport.h>
#include <openrct2-ui/interface/Widget.h>
#include <openrct2-ui/windows/Window.h>
#include <openrct2/Cheats.h>
#include <openrct2/Context.h>
#include <openrct2/Game.h>
#include <openrct2/Input.h>
#include <openrct2/actions/MazeSetTrackAction.h>
#include <openrct2/actions/RideDemolishAction.h>
#include <openrct2/actions/RideEntranceExitPlaceAction.h>
#include <openrct2/actions/RideSetStatusAction.h>
#include <openrct2/actions/TrackPlaceAction.h>
#include <openrct2/actions/TrackRemoveAction.h>
#include <openrct2/actions/TrackSetBrakeSpeedAction.h>
#include <openrct2/audio/audio.h>
#include <openrct2/config/Config.h>
#include <openrct2/localisation/Formatter.h>
#include <openrct2/localisation/Localisation.h>
#include <openrct2/network/network.h>
#include <openrct2/object/ObjectManager.h>
#include <openrct2/paint/tile_element/Paint.TileElement.h>
#include <openrct2/platform/Platform.h>
#include <openrct2/ride/Ride.h>
#include <openrct2/ride/RideConstruction.h>
#include <openrct2/ride/RideData.h>
#include <openrct2/ride/Track.h>
#include <openrct2/ride/TrackData.h>
#include <openrct2/sprites.h>
#include <openrct2/util/Math.hpp>
#include <openrct2/windows/Intent.h>
#include <openrct2/world/Entrance.h>
#include <openrct2/world/Footpath.h>
#include <openrct2/world/Park.h>

static constexpr StringId WINDOW_TITLE = STR_RIDE_CONSTRUCTION_WINDOW_TITLE;
static constexpr int32_t WH = 394;
static constexpr int32_t WW = 166;

static constexpr uint16_t ARROW_PULSE_DURATION = 200;
// Width of the group boxes, e.g. “Banking”
static constexpr int32_t GW = WW - 6;

using namespace OpenRCT2::TrackMetaData;

#pragma region Widgets

enum
{
    WIDX_BACKGROUND,
    WIDX_TITLE,
    WIDX_CLOSE,
    WIDX_DIRECTION_GROUPBOX,
    WIDX_SLOPE_GROUPBOX,
    WIDX_BANKING_GROUPBOX,
    WIDX_LEFT_CURVE_VERY_SMALL,
    WIDX_LEFT_CURVE_SMALL,
    WIDX_LEFT_CURVE,
    WIDX_LEFT_CURVE_LARGE,
    WIDX_STRAIGHT,
    WIDX_RIGHT_CURVE_LARGE,
    WIDX_RIGHT_CURVE,
    WIDX_RIGHT_CURVE_SMALL,
    WIDX_RIGHT_CURVE_VERY_SMALL,
    WIDX_SPECIAL_TRACK_DROPDOWN,
    WIDX_SLOPE_DOWN_STEEP,
    WIDX_SLOPE_DOWN,
    WIDX_LEVEL,
    WIDX_SLOPE_UP,
    WIDX_SLOPE_UP_STEEP,
    WIDX_CHAIN_LIFT,
    WIDX_BANK_LEFT,
    WIDX_BANK_STRAIGHT,
    WIDX_BANK_RIGHT,
    WIDX_CONSTRUCT,
    WIDX_DEMOLISH,
    WIDX_PREVIOUS_SECTION,
    WIDX_NEXT_SECTION,
    WIDX_ENTRANCE_EXIT_GROUPBOX,
    WIDX_ENTRANCE,
    WIDX_EXIT,
    WIDX_ROTATE,
    WIDX_U_TRACK,
    WIDX_O_TRACK,
    WIDX_SEAT_ROTATION_GROUPBOX,
    WIDX_SEAT_ROTATION_ANGLE_SPINNER,
    WIDX_SEAT_ROTATION_ANGLE_SPINNER_UP,
    WIDX_SEAT_ROTATION_ANGLE_SPINNER_DOWN,
    WIDX_SIMULATE,
    WIDX_SPEED_GROUPBOX = WIDX_BANKING_GROUPBOX,
    WIDX_SPEED_SETTING_SPINNER = WIDX_BANK_LEFT,
    WIDX_SPEED_SETTING_SPINNER_UP = WIDX_BANK_STRAIGHT,
    WIDX_SPEED_SETTING_SPINNER_DOWN = WIDX_BANK_RIGHT,
};

validate_global_widx(WC_RIDE_CONSTRUCTION, WIDX_CONSTRUCT);
validate_global_widx(WC_RIDE_CONSTRUCTION, WIDX_ENTRANCE);
validate_global_widx(WC_RIDE_CONSTRUCTION, WIDX_EXIT);
validate_global_widx(WC_RIDE_CONSTRUCTION, WIDX_ROTATE);

// clang-format off
static Widget _rideConstructionWidgets[] = {
    WINDOW_SHIM(WINDOW_TITLE, WW, WH),
    MakeWidget        ({  3,  17}, {     GW,  57}, WindowWidgetType::Groupbox, WindowColour::Primary  , STR_RIDE_CONSTRUCTION_DIRECTION                                                                       ),
    MakeWidget        ({  3,  76}, {     GW,  41}, WindowWidgetType::Groupbox, WindowColour::Primary  , STR_RIDE_CONSTRUCTION_SLOPE                                                                           ),
    MakeWidget        ({  3, 120}, {     GW,  41}, WindowWidgetType::Groupbox, WindowColour::Primary  , STR_RIDE_CONSTRUCTION_ROLL_BANKING                                                                    ),
    MakeWidget        ({  6,  29}, {     22,  24}, WindowWidgetType::FlatBtn,  WindowColour::Secondary, ImageId(SPR_RIDE_CONSTRUCTION_LEFT_CURVE_SMALL),  STR_RIDE_CONSTRUCTION_LEFT_CURVE_VERY_SMALL_TIP     ),
    MakeWidget        ({  6,  29}, {     22,  24}, WindowWidgetType::FlatBtn,  WindowColour::Secondary, ImageId(SPR_RIDE_CONSTRUCTION_LEFT_CURVE_SMALL),  STR_RIDE_CONSTRUCTION_LEFT_CURVE_SMALL_TIP          ),
    MakeWidget        ({ 28,  29}, {     22,  24}, WindowWidgetType::FlatBtn,  WindowColour::Secondary, ImageId(SPR_RIDE_CONSTRUCTION_LEFT_CURVE),        STR_RIDE_CONSTRUCTION_LEFT_CURVE_TIP                ),
    MakeWidget        ({ 50,  29}, {     22,  24}, WindowWidgetType::FlatBtn,  WindowColour::Secondary, ImageId(SPR_RIDE_CONSTRUCTION_LEFT_CURVE_LARGE),  STR_RIDE_CONSTRUCTION_LEFT_CURVE_LARGE_TIP          ),
    MakeWidget        ({ 72,  29}, {     22,  24}, WindowWidgetType::FlatBtn,  WindowColour::Secondary, ImageId(SPR_RIDE_CONSTRUCTION_STRAIGHT),          STR_RIDE_CONSTRUCTION_STRAIGHT_TIP                  ),
    MakeWidget        ({ 94,  29}, {     22,  24}, WindowWidgetType::FlatBtn,  WindowColour::Secondary, ImageId(SPR_RIDE_CONSTRUCTION_RIGHT_CURVE_LARGE), STR_RIDE_CONSTRUCTION_RIGHT_CURVE_LARGE_TIP         ),
    MakeWidget        ({116,  29}, {     22,  24}, WindowWidgetType::FlatBtn,  WindowColour::Secondary, ImageId(SPR_RIDE_CONSTRUCTION_RIGHT_CURVE),       STR_RIDE_CONSTRUCTION_RIGHT_CURVE_TIP               ),
    MakeWidget        ({138,  29}, {     22,  24}, WindowWidgetType::FlatBtn,  WindowColour::Secondary, ImageId(SPR_RIDE_CONSTRUCTION_RIGHT_CURVE_SMALL), STR_RIDE_CONSTRUCTION_RIGHT_CURVE_SMALL_TIP         ),
    MakeWidget        ({138,  29}, {     22,  24}, WindowWidgetType::FlatBtn,  WindowColour::Secondary, ImageId(SPR_RIDE_CONSTRUCTION_RIGHT_CURVE_SMALL), STR_RIDE_CONSTRUCTION_RIGHT_CURVE_VERY_SMALL_TIP    ),
    MakeWidget        ({  6,  55}, { GW - 6,  14}, WindowWidgetType::Button,   WindowColour::Secondary, STR_YELLOW_STRING,                                STR_RIDE_CONSTRUCTION_OTHER_TRACK_CONFIGURATIONS_TIP),
    MakeWidget        ({ 23,  88}, {     24,  24}, WindowWidgetType::FlatBtn,  WindowColour::Secondary, ImageId(SPR_RIDE_CONSTRUCTION_SLOPE_DOWN_STEEP),  STR_RIDE_CONSTRUCTION_STEEP_SLOPE_DOWN_TIP          ),
    MakeWidget        ({ 47,  88}, {     24,  24}, WindowWidgetType::FlatBtn,  WindowColour::Secondary, ImageId(SPR_RIDE_CONSTRUCTION_SLOPE_DOWN),        STR_RIDE_CONSTRUCTION_SLOPE_DOWN_TIP                ),
    MakeWidget        ({ 71,  88}, {     24,  24}, WindowWidgetType::FlatBtn,  WindowColour::Secondary, ImageId(SPR_RIDE_CONSTRUCTION_SLOPE_LEVEL),       STR_RIDE_CONSTRUCTION_LEVEL_TIP                     ),
    MakeWidget        ({ 95,  88}, {     24,  24}, WindowWidgetType::FlatBtn,  WindowColour::Secondary, ImageId(SPR_RIDE_CONSTRUCTION_SLOPE_UP),          STR_RIDE_CONSTRUCTION_SLOPE_UP_TIP                  ),
    MakeWidget        ({119,  88}, {     24,  24}, WindowWidgetType::FlatBtn,  WindowColour::Secondary, ImageId(SPR_RIDE_CONSTRUCTION_SLOPE_UP_STEEP),    STR_RIDE_CONSTRUCTION_STEEP_SLOPE_UP_TIP            ),
    MakeWidget        ({134,  88}, {     24,  24}, WindowWidgetType::FlatBtn,  WindowColour::Secondary, ImageId(SPR_CHAIN_LIFT),                          STR_RIDE_CONSTRUCTION_CHAIN_LIFT_TIP                ),
    MakeWidget        ({ 47, 132}, {     24,  24}, WindowWidgetType::FlatBtn,  WindowColour::Secondary, ImageId(SPR_RIDE_CONSTRUCTION_LEFT_BANK),         STR_RIDE_CONSTRUCTION_ROLL_FOR_LEFT_CURVE_TIP       ),
    MakeWidget        ({ 71, 132}, {     24,  24}, WindowWidgetType::FlatBtn,  WindowColour::Secondary, ImageId(SPR_RIDE_CONSTRUCTION_NO_BANK),           STR_RIDE_CONSTRUCTION_NO_ROLL_TIP                   ),
    MakeWidget        ({ 95, 132}, {     24,  24}, WindowWidgetType::FlatBtn,  WindowColour::Secondary, ImageId(SPR_RIDE_CONSTRUCTION_RIGHT_BANK),        STR_RIDE_CONSTRUCTION_ROLL_FOR_RIGHT_CURVE_TIP      ),
    MakeWidget        ({  3, 164}, {     GW, 170}, WindowWidgetType::ImgBtn,   WindowColour::Secondary, 0xFFFFFFFF,                                       STR_RIDE_CONSTRUCTION_CONSTRUCT_SELECTED_SECTION_TIP),
    MakeWidget        ({ 60, 338}, {     46,  24}, WindowWidgetType::FlatBtn,  WindowColour::Secondary, ImageId(SPR_DEMOLISH_CURRENT_SECTION),            STR_RIDE_CONSTRUCTION_REMOVE_HIGHLIGHTED_SECTION_TIP),
    MakeWidget        ({ 30, 338}, {     24,  24}, WindowWidgetType::FlatBtn,  WindowColour::Secondary, ImageId(SPR_PREVIOUS),                            STR_RIDE_CONSTRUCTION_MOVE_TO_PREVIOUS_SECTION_TIP  ),
    MakeWidget        ({112, 338}, {     24,  24}, WindowWidgetType::FlatBtn,  WindowColour::Secondary, ImageId(SPR_NEXT),                                STR_RIDE_CONSTRUCTION_MOVE_TO_NEXT_SECTION_TIP      ),
    MakeWidget        ({  3, 362}, {     GW,  28}, WindowWidgetType::Groupbox, WindowColour::Primary                                                                                                          ),
    MakeWidget        ({  9, 372}, {     70,  12}, WindowWidgetType::Button,   WindowColour::Secondary, STR_RIDE_CONSTRUCTION_ENTRANCE,                   STR_RIDE_CONSTRUCTION_ENTRANCE_TIP                  ),
    MakeWidget        ({ 87, 372}, {     70,  12}, WindowWidgetType::Button,   WindowColour::Secondary, STR_RIDE_CONSTRUCTION_EXIT,                       STR_RIDE_CONSTRUCTION_EXIT_TIP                      ),
    MakeWidget        ({ 72, 338}, {     24,  24}, WindowWidgetType::FlatBtn,  WindowColour::Secondary, ImageId(SPR_ROTATE_ARROW),                        STR_ROTATE_90_TIP                                   ),
    MakeWidget        ({ 19, 132}, {     24,  24}, WindowWidgetType::FlatBtn,  WindowColour::Secondary, ImageId(SPR_RIDE_CONSTRUCTION_U_SHAPED_TRACK),    STR_RIDE_CONSTRUCTION_U_SHAPED_OPEN_TRACK_TIP       ),
    MakeWidget        ({123, 132}, {     24,  24}, WindowWidgetType::FlatBtn,  WindowColour::Secondary, ImageId(SPR_RIDE_CONSTRUCTION_O_SHAPED_TRACK),    STR_RIDE_CONSTRUCTION_O_SHAPED_ENCLOSED_TRACK_TIP   ),
    MakeWidget        ({ 96, 120}, {     67,  41}, WindowWidgetType::Groupbox, WindowColour::Primary  , STR_RIDE_CONSTRUCTION_SEAT_ROT                                                                        ),
    MakeSpinnerWidgets({101, 138}, {     58,  12}, WindowWidgetType::Spinner,  WindowColour::Secondary, 0,                                                STR_RIDE_CONSTRUCTION_SELECT_SEAT_ROTATION_ANGLE_TIP),
    MakeWidget        ({139, 338}, {     24,  24}, WindowWidgetType::FlatBtn,  WindowColour::Secondary, ImageId(SPR_G2_SIMULATE),                         STR_SIMULATE_RIDE_TIP                               ),
    WIDGETS_END,
};
// clang-format on

#pragma endregion

static bool _trackPlaceCtrlState;
static int32_t _trackPlaceCtrlZ;
static bool _trackPlaceShiftState;
static ScreenCoordsXY _trackPlaceShiftStart;
static int32_t _trackPlaceShiftZ;
static int32_t _trackPlaceZ;
static money64 _trackPlaceCost;
static StringId _trackPlaceErrorMessage;
static bool _autoRotatingShop;
static bool _gotoStartPlacementMode = false;

static constexpr StringId RideConstructionSeatAngleRotationStrings[] = {
    STR_RIDE_CONSTRUCTION_SEAT_ROTATION_ANGLE_NEG_180, STR_RIDE_CONSTRUCTION_SEAT_ROTATION_ANGLE_NEG_135,
    STR_RIDE_CONSTRUCTION_SEAT_ROTATION_ANGLE_NEG_90,  STR_RIDE_CONSTRUCTION_SEAT_ROTATION_ANGLE_NEG_45,
    STR_RIDE_CONSTRUCTION_SEAT_ROTATION_ANGLE_0,       STR_RIDE_CONSTRUCTION_SEAT_ROTATION_ANGLE_45,
    STR_RIDE_CONSTRUCTION_SEAT_ROTATION_ANGLE_90,      STR_RIDE_CONSTRUCTION_SEAT_ROTATION_ANGLE_135,
    STR_RIDE_CONSTRUCTION_SEAT_ROTATION_ANGLE_180,     STR_RIDE_CONSTRUCTION_SEAT_ROTATION_ANGLE_225,
    STR_RIDE_CONSTRUCTION_SEAT_ROTATION_ANGLE_270,     STR_RIDE_CONSTRUCTION_SEAT_ROTATION_ANGLE_315,
    STR_RIDE_CONSTRUCTION_SEAT_ROTATION_ANGLE_360,     STR_RIDE_CONSTRUCTION_SEAT_ROTATION_ANGLE_405,
    STR_RIDE_CONSTRUCTION_SEAT_ROTATION_ANGLE_450,     STR_RIDE_CONSTRUCTION_SEAT_ROTATION_ANGLE_495,
};

static void WindowRideConstructionMouseUpDemolishNextPiece(const CoordsXYZD& piecePos, int32_t type);

static int32_t RideGetAlternativeType(const Ride& ride)
{
    return (_currentTrackAlternative & RIDE_TYPE_ALTERNATIVE_TRACK_TYPE) ? ride.GetRideTypeDescriptor().AlternateType
                                                                         : ride.type;
}

/* move to ride.c */
static void CloseRideWindowForConstruction(RideId rideId)
{
    WindowBase* w = WindowFindByNumber(WindowClass::Ride, rideId.ToUnderlying());
    if (w != nullptr && w->page == 1)
        WindowClose(*w);
}

static void RideConstructPlacedForwardGameActionCallback(const GameAction* ga, const GameActions::Result* result);
static void RideConstructPlacedBackwardGameActionCallback(const GameAction* ga, const GameActions::Result* result);
static void CloseConstructWindowOnCompletion(const Ride& ride);

class RideConstructionWindow final : public Window
{
private:
    uint8_t _currentlyShowingBrakeOrBoosterSpeed{};
    SpecialElementsDropdownState _specialElementDropdownState;
    bool _autoOpeningShop{};

public:
    void OnOpen() override
    {
        auto currentRide = GetRide(_currentRideIndex);
        if (currentRide == nullptr)
        {
            return;
        }

        widgets = _rideConstructionWidgets;
        number = _currentRideIndex.ToUnderlying();

        InitScrollWidgets();

        WindowPushOthersRight(*this);
        ShowGridlines();

        _currentTrackPrice = MONEY64_UNDEFINED;
        _currentBrakeSpeed2 = 8;
        _currentSeatRotationAngle = 4;

        _currentTrackCurve = currentRide->GetRideTypeDescriptor().StartTrackPiece | RideConstructionSpecialPieceSelected;
        _currentTrackSlopeEnd = 0;
        _currentTrackBankEnd = 0;
        _currentTrackLiftHill = 0;
        _currentTrackAlternative = RIDE_TYPE_NO_ALTERNATIVES;

        if (currentRide->GetRideTypeDescriptor().HasFlag(RIDE_TYPE_FLAG_START_CONSTRUCTION_INVERTED))
            _currentTrackAlternative |= RIDE_TYPE_ALTERNATIVE_TRACK_TYPE;

        _previousTrackBankEnd = 0;
        _previousTrackSlopeEnd = 0;

        _currentTrackPieceDirection = 0;
        _rideConstructionState = RideConstructionState::Place;
        _currentTrackSelectionFlags = 0;
        _autoOpeningShop = false;
        _autoRotatingShop = true;
        _trackPlaceCtrlState = false;
        _trackPlaceShiftState = false;
    }

    void OnClose() override
    {
        RideConstructionInvalidateCurrentTrack();
        ViewportSetVisibility(ViewportVisibility::Default);

        MapInvalidateMapSelectionTiles();
        gMapSelectFlags &= ~MAP_SELECT_FLAG_ENABLE_CONSTRUCT;
        gMapSelectFlags &= ~MAP_SELECT_FLAG_ENABLE_ARROW;

        // In order to cancel the yellow arrow correctly the
        // selection tool should be cancelled. Don't do a tool cancel if
        // another window has already taken control of tool.
        if (classification == gCurrentToolWidget.window_classification && number == gCurrentToolWidget.window_number)
            ToolCancel();

        HideGridlines();

        // If we demolish a currentRide all windows will be closed including the construction window,
        // the currentRide at this point is already gone.
        auto currentRide = GetRide(_currentRideIndex);
        if (currentRide == nullptr)
        {
            return;
        }

        if (RideTryGetOriginElement(*currentRide, nullptr))
        {
            // Auto open shops if required.
            if (currentRide->mode == RideMode::ShopStall && gConfigGeneral.AutoOpenShops)
            {
                // HACK: Until we find a good a way to defer the game command for opening the shop, stop this
                //       from getting stuck in an infinite loop as opening the currentRide will try to close this window
                if (!_autoOpeningShop)
                {
                    _autoOpeningShop = true;
                    auto gameAction = RideSetStatusAction(currentRide->id, RideStatus::Open);
                    GameActions::Execute(&gameAction);
                    _autoOpeningShop = false;
                }
            }

            currentRide->SetToDefaultInspectionInterval();
            auto intent = Intent(WindowClass::Ride);
            intent.PutExtra(INTENT_EXTRA_RIDE_ID, currentRide->id.ToUnderlying());
            ContextOpenIntent(&intent);
        }
        else
        {
            auto gameAction = RideDemolishAction(currentRide->id, RIDE_MODIFY_DEMOLISH);
            gameAction.SetFlags(GAME_COMMAND_FLAG_ALLOW_DURING_PAUSED);
            GameActions::Execute(&gameAction);
        }
    }

    void OnResize() override
    {
        ResizeFrame();
        WindowRideConstructionUpdateEnabledTrackPieces();

        auto currentRide = GetRide(_currentRideIndex);
        if (currentRide == nullptr)
        {
            return;
        }

        int32_t rideType = RideGetAlternativeType(*currentRide);

        uint64_t disabledWidgets = 0;

        if (_currentTrackCurve & RideConstructionSpecialPieceSelected)
        {
            disabledWidgets |= (1uLL << WIDX_SLOPE_GROUPBOX) | (1uLL << WIDX_BANKING_GROUPBOX) | (1uLL << WIDX_SLOPE_DOWN_STEEP)
                | (1uLL << WIDX_SLOPE_DOWN) | (1uLL << WIDX_LEVEL) | (1uLL << WIDX_SLOPE_UP) | (1uLL << WIDX_SLOPE_UP_STEEP)
                | (1uLL << WIDX_CHAIN_LIFT) | (1uLL << WIDX_BANK_LEFT) | (1uLL << WIDX_BANK_STRAIGHT)
                | (1uLL << WIDX_BANK_RIGHT);
        }

        // Disable large curves if the start or end of the track is sloped and large sloped curves are not available
        if ((_previousTrackSlopeEnd != TRACK_SLOPE_NONE || _currentTrackSlopeEnd != TRACK_SLOPE_NONE))
        {
            if (!IsTrackEnabled(TRACK_SLOPE_CURVE_LARGE)
                || !(_previousTrackSlopeEnd == TRACK_SLOPE_UP_25 || _previousTrackSlopeEnd == TRACK_SLOPE_DOWN_25)
                || !(_currentTrackSlopeEnd == TRACK_SLOPE_UP_25 || _currentTrackSlopeEnd == TRACK_SLOPE_DOWN_25))
            {
                disabledWidgets |= (1uLL << WIDX_LEFT_CURVE_LARGE) | (1uLL << WIDX_RIGHT_CURVE_LARGE);
            }
        }
        if (IsTrackEnabled(TRACK_SLOPE_CURVE) && IsTrackEnabled(TRACK_CURVE_VERY_SMALL))
        {
            // Disable small curves if the start or end of the track is sloped.
            if (_previousTrackSlopeEnd != TRACK_SLOPE_NONE || _currentTrackSlopeEnd != TRACK_SLOPE_NONE)
            {
                disabledWidgets |= (1uLL << WIDX_LEFT_CURVE_VERY_SMALL) | (1uLL << WIDX_LEFT_CURVE) | (1uLL << WIDX_RIGHT_CURVE)
                    | (1uLL << WIDX_RIGHT_CURVE_VERY_SMALL);
            }
        }
        if (!IsTrackEnabled(TRACK_SLOPE_CURVE))
        {
            if (IsTrackEnabled(TRACK_CURVE_VERTICAL))
            {
                // Disable all curves only on vertical track
                if (_previousTrackSlopeEnd != TRACK_SLOPE_UP_90 || _currentTrackSlopeEnd != TRACK_SLOPE_UP_90)
                {
                    if (_previousTrackSlopeEnd != TRACK_SLOPE_DOWN_90 || _currentTrackSlopeEnd != TRACK_SLOPE_DOWN_90)
                    {
                        disabledWidgets |= (1uLL << WIDX_LEFT_CURVE_VERY_SMALL) | (1uLL << WIDX_LEFT_CURVE_SMALL)
                            | (1uLL << WIDX_LEFT_CURVE) | (1uLL << WIDX_RIGHT_CURVE) | (1uLL << WIDX_RIGHT_CURVE_SMALL)
                            | (1uLL << WIDX_RIGHT_CURVE_VERY_SMALL);
                    }
                }
            }
            else
            {
                // Disable all curves on sloped track
                if (_previousTrackSlopeEnd != TRACK_SLOPE_NONE || _currentTrackSlopeEnd != TRACK_SLOPE_NONE)
                {
                    disabledWidgets |= (1uLL << WIDX_LEFT_CURVE_VERY_SMALL) | (1uLL << WIDX_LEFT_CURVE_SMALL)
                        | (1uLL << WIDX_LEFT_CURVE) | (1uLL << WIDX_RIGHT_CURVE) | (1uLL << WIDX_RIGHT_CURVE_SMALL)
                        | (1uLL << WIDX_RIGHT_CURVE_VERY_SMALL);
                }
            }
        }
        if (!IsTrackEnabled(TRACK_FLAT_ROLL_BANKING))
        {
            // Disable banking
            disabledWidgets |= (1uLL << WIDX_BANKING_GROUPBOX) | (1uLL << WIDX_BANK_LEFT) | (1uLL << WIDX_BANK_STRAIGHT)
                | (1uLL << WIDX_BANK_RIGHT);
        }
        // Disable banking if the start track is steep and the end of the track becomes flat.
        if ((_previousTrackSlopeEnd == TRACK_SLOPE_DOWN_60 || _previousTrackSlopeEnd == TRACK_SLOPE_UP_60)
            && _currentTrackSlopeEnd == TRACK_SLOPE_NONE)
        {
            disabledWidgets |= (1uLL << WIDX_BANK_LEFT) | (1uLL << WIDX_BANK_RIGHT);
        }
        if (!IsTrackEnabled(TRACK_SLOPE) && !IsTrackEnabled(TRACK_SLOPE_STEEP_DOWN) && !IsTrackEnabled(TRACK_SLOPE_STEEP_UP))
        {
            if (!currentRide->GetRideTypeDescriptor().SupportsTrackPiece(TRACK_REVERSE_FREEFALL))
            {
                // Disable all slopes
                disabledWidgets |= (1uLL << WIDX_SLOPE_GROUPBOX) | (1uLL << WIDX_SLOPE_DOWN_STEEP) | (1uLL << WIDX_SLOPE_DOWN)
                    | (1uLL << WIDX_LEVEL) | (1uLL << WIDX_SLOPE_UP) | (1uLL << WIDX_SLOPE_UP_STEEP);
            }
        }
        // If ride type does not have access to diagonal sloped turns, disallow simultaneous use of banked and sloped diagonals
        if (!IsTrackEnabled(TRACK_SLOPE_CURVE_LARGE) && TrackPieceDirectionIsDiagonal(_currentTrackPieceDirection))
        {
            if (_currentTrackSlopeEnd != TRACK_SLOPE_NONE)
            {
                disabledWidgets |= (1uLL << WIDX_BANK_LEFT) | (1uLL << WIDX_BANK_RIGHT);
            }
            else if (_currentTrackBankEnd != TRACK_BANK_NONE)
            {
                disabledWidgets |= (1uLL << WIDX_SLOPE_DOWN) | (1uLL << WIDX_SLOPE_UP);
            }
        }
        if (currentRide->GetRideTypeDescriptor().HasFlag(RIDE_TYPE_FLAG_UP_INCLINE_REQUIRES_LIFT)
            && !gCheatsEnableAllDrawableTrackPieces)
        {
            // Disable lift hill toggle and banking if current track piece is uphill
            if (_previousTrackSlopeEnd == TRACK_SLOPE_UP_25 || _previousTrackSlopeEnd == TRACK_SLOPE_UP_60
                || _currentTrackSlopeEnd == TRACK_SLOPE_UP_25 || _currentTrackSlopeEnd == TRACK_SLOPE_UP_60)
                disabledWidgets |= 1uLL << WIDX_CHAIN_LIFT | (1uLL << WIDX_BANK_LEFT) | (1uLL << WIDX_BANK_RIGHT);
            // Disable upward slope if current track piece is not flat
            if ((_previousTrackSlopeEnd != TRACK_SLOPE_NONE || _previousTrackBankEnd != TRACK_BANK_NONE)
                && !(_currentTrackLiftHill & CONSTRUCTION_LIFT_HILL_SELECTED))
                disabledWidgets |= (1uLL << WIDX_SLOPE_UP);
        }
        if (_rideConstructionState == RideConstructionState::State0)
        {
            disabledWidgets |= (1uLL << WIDX_CONSTRUCT) | (1uLL << WIDX_DEMOLISH) | (1uLL << WIDX_PREVIOUS_SECTION)
                | (1uLL << WIDX_NEXT_SECTION);
        }
        switch (_currentTrackCurve)
        {
            case TRACK_CURVE_LEFT_VERY_SMALL:
            case TRACK_CURVE_LEFT_SMALL:
            case TRACK_CURVE_LEFT:
            case TRACK_CURVE_LEFT_LARGE:
                disabledWidgets |= (1uLL << WIDX_BANK_RIGHT);
                if (_previousTrackBankEnd == TRACK_BANK_NONE)
                {
                    disabledWidgets |= (1uLL << WIDX_BANK_LEFT);
                }
                else
                {
                    disabledWidgets |= (1uLL << WIDX_BANK_STRAIGHT);
                }
                break;
            case TRACK_CURVE_RIGHT_LARGE:
            case TRACK_CURVE_RIGHT:
            case TRACK_CURVE_RIGHT_SMALL:
            case TRACK_CURVE_RIGHT_VERY_SMALL:
                disabledWidgets |= (1uLL << WIDX_BANK_LEFT);
                if (_previousTrackBankEnd == TRACK_BANK_NONE)
                {
                    disabledWidgets |= (1uLL << WIDX_BANK_RIGHT);
                }
                else
                {
                    disabledWidgets |= (1uLL << WIDX_BANK_STRAIGHT);
                }
                break;
        }
        if (!IsTrackEnabled(TRACK_SLOPE_ROLL_BANKING))
        {
            if (_currentTrackBankEnd != TRACK_BANK_NONE)
            {
                disabledWidgets |= (1uLL << WIDX_SLOPE_DOWN) | (1uLL << WIDX_SLOPE_UP);
            }
        }
        if (_previousTrackSlopeEnd == _currentTrackSlopeEnd)
        {
            switch (_currentTrackSlopeEnd)
            {
                case TRACK_SLOPE_UP_60:
                case TRACK_SLOPE_DOWN_60:
                    disabledWidgets |= (1uLL << WIDX_LEFT_CURVE_VERY_SMALL) | (1uLL << WIDX_LEFT_CURVE)
                        | (1uLL << WIDX_RIGHT_CURVE) | (1uLL << WIDX_RIGHT_CURVE_VERY_SMALL);
                    if (!IsTrackEnabled(TRACK_SLOPE_CURVE_STEEP))
                    {
                        disabledWidgets |= (1uLL << WIDX_LEFT_CURVE_SMALL) | (1uLL << WIDX_RIGHT_CURVE_SMALL);
                    }
                    break;
                case TRACK_SLOPE_UP_90:
                case TRACK_SLOPE_DOWN_90:
                    disabledWidgets |= (1uLL << WIDX_LEFT_CURVE_VERY_SMALL) | (1uLL << WIDX_LEFT_CURVE)
                        | (1uLL << WIDX_RIGHT_CURVE) | (1uLL << WIDX_RIGHT_CURVE_VERY_SMALL);
                    if (!IsTrackEnabled(TRACK_CURVE_VERTICAL))
                    {
                        disabledWidgets |= (1uLL << WIDX_LEFT_CURVE_SMALL) | (1uLL << WIDX_RIGHT_CURVE_SMALL);
                    }
                    break;
            }
        }
        else
        {
            // Disable all curves
            disabledWidgets |= (1uLL << WIDX_LEFT_CURVE_VERY_SMALL) | (1uLL << WIDX_LEFT_CURVE_SMALL)
                | (1uLL << WIDX_LEFT_CURVE) | (1uLL << WIDX_RIGHT_CURVE) | (1uLL << WIDX_RIGHT_CURVE_SMALL)
                | (1uLL << WIDX_RIGHT_CURVE_VERY_SMALL);
        }

        switch (_previousTrackSlopeEnd)
        {
            case TRACK_SLOPE_NONE:
                if (_currentTrackCurve != TRACK_CURVE_NONE
                    || (IsTrackEnabled(TRACK_SLOPE_STEEP_LONG) && TrackPieceDirectionIsDiagonal(_currentTrackPieceDirection)))
                {
                    disabledWidgets |= (1uLL << WIDX_SLOPE_DOWN_STEEP) | (1uLL << WIDX_SLOPE_UP_STEEP);
                }
                break;
            case TRACK_SLOPE_DOWN_25:
                disabledWidgets |= (1uLL << WIDX_SLOPE_UP) | (1uLL << WIDX_SLOPE_UP_STEEP);
                break;
            case TRACK_SLOPE_DOWN_60:
                disabledWidgets |= (1uLL << WIDX_SLOPE_UP) | (1uLL << WIDX_SLOPE_UP_STEEP);
                if (!IsTrackEnabled(TRACK_SLOPE_LONG)
                    && !(IsTrackEnabled(TRACK_SLOPE_STEEP_LONG) && !TrackPieceDirectionIsDiagonal(_currentTrackPieceDirection)))
                {
                    disabledWidgets |= (1uLL << WIDX_LEVEL);
                }
                break;
            case TRACK_SLOPE_UP_25:
                disabledWidgets |= (1uLL << WIDX_SLOPE_DOWN_STEEP) | (1uLL << WIDX_SLOPE_DOWN);
                break;
            case TRACK_SLOPE_UP_60:
                disabledWidgets |= (1uLL << WIDX_SLOPE_DOWN_STEEP) | (1uLL << WIDX_SLOPE_DOWN);
                if (!IsTrackEnabled(TRACK_SLOPE_LONG)
                    && !(IsTrackEnabled(TRACK_SLOPE_STEEP_LONG) && !TrackPieceDirectionIsDiagonal(_currentTrackPieceDirection)))
                {
                    disabledWidgets |= (1uLL << WIDX_LEVEL);
                }
                break;
            case TRACK_SLOPE_DOWN_90:
            case TRACK_SLOPE_UP_90:
                disabledWidgets |= (1uLL << WIDX_SLOPE_DOWN) | (1uLL << WIDX_LEVEL) | (1uLL << WIDX_SLOPE_UP);
                break;
        }
        if (_previousTrackSlopeEnd == TRACK_SLOPE_NONE)
        {
            if (!IsTrackEnabled(TRACK_SLOPE_LONG) && !IsTrackEnabled(TRACK_SLOPE_STEEP_LONG))
            {
                disabledWidgets |= (1uLL << WIDX_SLOPE_DOWN_STEEP) | (1uLL << WIDX_SLOPE_UP_STEEP);
            }
        }
        if (IsTrackEnabled(TRACK_SLOPE_VERTICAL))
        {
            if (_previousTrackSlopeEnd == TRACK_SLOPE_UP_60 && _currentTrackPieceDirection < 4)
            {
                disabledWidgets &= ~(1uLL << WIDX_SLOPE_DOWN_STEEP);
            }
            if (_previousTrackSlopeEnd == TRACK_SLOPE_UP_90)
            {
                disabledWidgets &= ~(1uLL << WIDX_SLOPE_DOWN_STEEP);
            }
            if (_previousTrackSlopeEnd == TRACK_SLOPE_DOWN_60 && _currentTrackPieceDirection < 4)
            {
                disabledWidgets &= ~(1uLL << WIDX_SLOPE_UP_STEEP);
            }
        }
        if (_previousTrackBankEnd == TRACK_BANK_LEFT)
        {
            disabledWidgets |= (1uLL << WIDX_RIGHT_CURVE_SMALL) | (1uLL << WIDX_RIGHT_CURVE) | (1uLL << WIDX_RIGHT_CURVE_LARGE)
                | (1uLL << WIDX_BANK_RIGHT);
        }
        if (_previousTrackBankEnd == TRACK_BANK_RIGHT)
        {
            disabledWidgets |= (1uLL << WIDX_LEFT_CURVE_SMALL) | (1uLL << WIDX_LEFT_CURVE) | (1uLL << WIDX_LEFT_CURVE_LARGE)
                | (1uLL << WIDX_BANK_LEFT);
        }
        if (_currentTrackBankEnd != _previousTrackBankEnd)
        {
            disabledWidgets |= (1uLL << WIDX_RIGHT_CURVE_SMALL) | (1uLL << WIDX_RIGHT_CURVE) | (1uLL << WIDX_RIGHT_CURVE_LARGE)
                | (1uLL << WIDX_LEFT_CURVE_SMALL) | (1uLL << WIDX_LEFT_CURVE) | (1uLL << WIDX_LEFT_CURVE_LARGE);
        }
        if (_currentTrackSlopeEnd != TRACK_SLOPE_NONE)
        {
            if (IsTrackEnabled(TRACK_SLOPE_ROLL_BANKING))
            {
                if (_previousTrackSlopeEnd == TRACK_SLOPE_NONE)
                {
                    if (_currentTrackSlopeEnd != TRACK_SLOPE_UP_25 && _currentTrackSlopeEnd != TRACK_SLOPE_DOWN_25)
                    {
                        disabledWidgets |= (1uLL << WIDX_BANK_LEFT) | (1uLL << WIDX_BANK_RIGHT);
                    }
                }
                else
                {
                    if (_currentTrackSlopeEnd != _previousTrackSlopeEnd)
                    {
                        disabledWidgets |= (1uLL << WIDX_BANK_LEFT) | (1uLL << WIDX_BANK_RIGHT);
                    }
                    else
                    {
                        if (_currentTrackSlopeEnd != TRACK_SLOPE_UP_25 && _currentTrackSlopeEnd != TRACK_SLOPE_DOWN_25)
                        {
                            disabledWidgets |= (1uLL << WIDX_BANK_LEFT) | (1uLL << WIDX_BANK_RIGHT);
                        }
                    }
                }
            }
            else
            {
                disabledWidgets |= (1uLL << WIDX_BANK_LEFT) | (1uLL << WIDX_BANK_RIGHT);
            }
        }
        if (_currentTrackBankEnd != TRACK_BANK_NONE || _previousTrackBankEnd != TRACK_BANK_NONE)
        {
            disabledWidgets |= (1uLL << WIDX_SLOPE_DOWN_STEEP) | (1uLL << WIDX_SLOPE_UP_STEEP) | (1uLL << WIDX_CHAIN_LIFT);
        }
        if (_currentTrackCurve != TRACK_CURVE_NONE)
        {
            if (!IsTrackEnabled(TRACK_LIFT_HILL_CURVE))
            {
                disabledWidgets |= (1uLL << WIDX_CHAIN_LIFT);
            }
            if (_currentTrackSlopeEnd == TRACK_SLOPE_NONE)
            {
                disabledWidgets |= (1uLL << WIDX_CHAIN_LIFT);
            }
            if (_currentTrackSlopeEnd == TRACK_SLOPE_UP_60)
            {
                disabledWidgets |= (1uLL << WIDX_CHAIN_LIFT);
            }
            if (_currentTrackSlopeEnd == TRACK_SLOPE_DOWN_60)
            {
                disabledWidgets |= (1uLL << WIDX_CHAIN_LIFT);
            }
        }
        if (_currentTrackSlopeEnd == TRACK_SLOPE_UP_90 || _previousTrackSlopeEnd == TRACK_SLOPE_UP_90)
        {
            disabledWidgets |= (1uLL << WIDX_CHAIN_LIFT);
        }
        if (!IsTrackEnabled(TRACK_LIFT_HILL_STEEP))
        {
            if (_previousTrackSlopeEnd == TRACK_SLOPE_UP_60 || _currentTrackSlopeEnd == TRACK_SLOPE_UP_60)
            {
                disabledWidgets |= (1uLL << WIDX_CHAIN_LIFT);
            }
        }
        if (_previousTrackBankEnd == TRACK_BANK_UPSIDE_DOWN)
        {
            disabledWidgets |= (1uLL << WIDX_LEFT_CURVE_SMALL) | (1uLL << WIDX_LEFT_CURVE) | (1uLL << WIDX_LEFT_CURVE_LARGE)
                | (1uLL << WIDX_STRAIGHT) | (1uLL << WIDX_RIGHT_CURVE_SMALL) | (1uLL << WIDX_RIGHT_CURVE)
                | (1uLL << WIDX_RIGHT_CURVE_LARGE);
        }
        if (_currentTrackCurve != TRACK_CURVE_NONE)
        {
            if (_currentTrackSlopeEnd == TRACK_SLOPE_NONE)
            {
                disabledWidgets |= (1uLL << WIDX_SLOPE_DOWN) | (1uLL << WIDX_SLOPE_UP);
            }
            if (_currentTrackSlopeEnd == _previousTrackSlopeEnd)
            {
                if (_currentTrackSlopeEnd == TRACK_SLOPE_UP_25)
                {
                    disabledWidgets |= (1uLL << WIDX_SLOPE_UP_STEEP);
                    if (_currentTrackCurve == TRACK_CURVE_LEFT || _currentTrackCurve == TRACK_CURVE_RIGHT
                        || _rideConstructionState != RideConstructionState::Back || !IsTrackEnabled(TRACK_SLOPE_CURVE_BANKED))
                    {
                        disabledWidgets |= (1uLL << WIDX_LEVEL);
                    }
                }
                if (_currentTrackSlopeEnd == TRACK_SLOPE_DOWN_25)
                {
                    disabledWidgets |= (1uLL << WIDX_SLOPE_DOWN_STEEP);
                    if (_currentTrackCurve == TRACK_CURVE_LEFT || _currentTrackCurve == TRACK_CURVE_RIGHT
                        || _rideConstructionState != RideConstructionState::Front || !IsTrackEnabled(TRACK_SLOPE_CURVE_BANKED))
                    {
                        disabledWidgets |= (1uLL << WIDX_LEVEL);
                    }
                }
            }
            else if (IsTrackEnabled(TRACK_SLOPE_CURVE_BANKED))
            {
                disabledWidgets |= (1uLL << WIDX_SLOPE_DOWN_STEEP) | (1uLL << WIDX_SLOPE_UP_STEEP);
                if (_currentTrackBankEnd == TRACK_BANK_LEFT)
                {
                    disabledWidgets |= (1uLL << WIDX_BANK_STRAIGHT) | (1uLL << WIDX_BANK_RIGHT);
                    disabledWidgets &= ~(1uLL << WIDX_BANK_LEFT);
                }
                if (_currentTrackBankEnd == TRACK_BANK_RIGHT)
                {
                    disabledWidgets |= (1uLL << WIDX_BANK_LEFT) | (1uLL << WIDX_BANK_STRAIGHT);
                    disabledWidgets &= ~(1uLL << WIDX_BANK_RIGHT);
                }
                if (_currentTrackBankEnd == TRACK_BANK_NONE)
                {
                    disabledWidgets |= (1uLL << WIDX_BANK_LEFT) | (1uLL << WIDX_BANK_RIGHT);
                    disabledWidgets &= ~(1uLL << WIDX_BANK_STRAIGHT);
                }
                if (_currentTrackSlopeEnd == TRACK_SLOPE_NONE)
                {
                    disabledWidgets |= (1uLL << WIDX_SLOPE_DOWN) | (1uLL << WIDX_SLOPE_UP);
                    disabledWidgets &= ~(1uLL << WIDX_LEVEL);
                }
                if (_currentTrackSlopeEnd == TRACK_SLOPE_UP_25)
                {
                    disabledWidgets |= (1uLL << WIDX_SLOPE_DOWN) | (1uLL << WIDX_LEVEL);
                    disabledWidgets &= ~(1uLL << WIDX_SLOPE_UP);
                }
                if (_currentTrackSlopeEnd == TRACK_SLOPE_DOWN_25)
                {
                    disabledWidgets |= (1uLL << WIDX_LEVEL) | (1uLL << WIDX_SLOPE_UP);
                    disabledWidgets &= ~(1uLL << WIDX_SLOPE_DOWN);
                }
                if (_currentTrackCurve == TRACK_CURVE_LEFT_SMALL)
                {
                    disabledWidgets &= ~(1uLL << WIDX_LEFT_CURVE_SMALL);
                }
                if (_currentTrackCurve == TRACK_CURVE_RIGHT_SMALL)
                {
                    disabledWidgets &= ~(1uLL << WIDX_RIGHT_CURVE_SMALL);
                }
            }
        }
        if (_currentTrackCurve != TRACK_CURVE_NONE && _currentTrackSlopeEnd == TRACK_SLOPE_UP_60)
        {
            disabledWidgets |= (1uLL << WIDX_SLOPE_UP);
        }
        if (_currentTrackCurve != TRACK_CURVE_NONE && _currentTrackSlopeEnd == TRACK_SLOPE_DOWN_60)
        {
            disabledWidgets |= (1uLL << WIDX_SLOPE_DOWN);
        }
        if ((_currentTrackLiftHill & CONSTRUCTION_LIFT_HILL_SELECTED) && !gCheatsEnableChainLiftOnAllTrack)
        {
            if (_currentTrackSlopeEnd != TRACK_SLOPE_NONE && !IsTrackEnabled(TRACK_LIFT_HILL_CURVE))
            {
                disabledWidgets |= (1uLL << WIDX_LEFT_CURVE_SMALL) | (1uLL << WIDX_LEFT_CURVE) | (1uLL << WIDX_LEFT_CURVE_LARGE)
                    | (1uLL << WIDX_RIGHT_CURVE_SMALL) | (1uLL << WIDX_RIGHT_CURVE) | (1uLL << WIDX_RIGHT_CURVE_LARGE);
            }
            if (!IsTrackEnabled(TRACK_LIFT_HILL_STEEP))
            {
                if (widgets[WIDX_SLOPE_UP_STEEP].tooltip == STR_RIDE_CONSTRUCTION_STEEP_SLOPE_UP_TIP)
                {
                    disabledWidgets |= (1uLL << WIDX_SLOPE_UP_STEEP);
                }
            }
        }
        if (_previousTrackSlopeEnd == TRACK_SLOPE_UP_60 && _currentTrackCurve != TRACK_CURVE_NONE)
        {
            disabledWidgets |= (1uLL << WIDX_SLOPE_DOWN_STEEP) | (1uLL << WIDX_LEVEL);
        }
        if (_previousTrackSlopeEnd == TRACK_SLOPE_DOWN_60 && _currentTrackCurve != TRACK_CURVE_NONE)
        {
            disabledWidgets |= (1uLL << WIDX_LEVEL) | (1uLL << WIDX_SLOPE_UP_STEEP);
        }
        if (_currentTrackSlopeEnd == TRACK_SLOPE_UP_90 || _previousTrackSlopeEnd == TRACK_SLOPE_UP_90)
        {
            if (_currentTrackCurve != TRACK_CURVE_NONE)
            {
                disabledWidgets |= (1uLL << WIDX_SLOPE_UP_STEEP);
            }
            disabledWidgets |= (1uLL << WIDX_LEFT_CURVE_LARGE) | (1uLL << WIDX_RIGHT_CURVE_LARGE);
            if (currentRide->GetRideTypeDescriptor().SupportsTrackPiece(TRACK_REVERSE_FREEFALL))
            {
                disabledWidgets |= (1uLL << WIDX_STRAIGHT) | (1uLL << WIDX_RIGHT_CURVE) | (1uLL << WIDX_RIGHT_CURVE_SMALL)
                    | (1uLL << WIDX_LEFT_CURVE_SMALL) | (1uLL << WIDX_LEFT_CURVE);
            }
        }
        else if (_currentTrackSlopeEnd == TRACK_SLOPE_DOWN_90 || _previousTrackSlopeEnd == TRACK_SLOPE_DOWN_90)
        {
            if (_currentTrackCurve != TRACK_CURVE_NONE)
            {
                disabledWidgets |= (1uLL << WIDX_SLOPE_DOWN_STEEP);
            }
            disabledWidgets |= (1uLL << WIDX_LEFT_CURVE_LARGE) | (1uLL << WIDX_RIGHT_CURVE_LARGE);
            if (currentRide->GetRideTypeDescriptor().SupportsTrackPiece(TRACK_REVERSE_FREEFALL))
            {
                disabledWidgets |= (1uLL << WIDX_STRAIGHT) | (1uLL << WIDX_RIGHT_CURVE) | (1uLL << WIDX_RIGHT_CURVE_SMALL)
                    | (1uLL << WIDX_LEFT_CURVE_SMALL) | (1uLL << WIDX_LEFT_CURVE);
            }
        }
        // If the previous track is flat and the next track is flat, attempt to show buttons for helixes
        if (_currentTrackSlopeEnd == TRACK_SLOPE_NONE && _currentTrackSlopeEnd == _previousTrackSlopeEnd)
        {
            // If the bank is none, attempt to show unbanked quarter helixes
            if (_currentTrackBankEnd == TRACK_BANK_NONE
                && (_currentTrackCurve == TRACK_CURVE_LEFT || _currentTrackCurve == TRACK_CURVE_RIGHT))
            {
                if (IsTrackEnabled(TRACK_HELIX_DOWN_UNBANKED_QUARTER))
                    disabledWidgets &= ~(1uLL << WIDX_SLOPE_DOWN_STEEP);
                if (IsTrackEnabled(TRACK_HELIX_UP_UNBANKED_QUARTER))
                    disabledWidgets &= ~(1uLL << WIDX_SLOPE_UP_STEEP);
            }
            // If the track is banked left or right and curvature is standard size (2.5 tile radius), attempt to show buttons
            // for half or quarter helixes
            else if (
                (_currentTrackBankEnd == TRACK_BANK_LEFT || _currentTrackBankEnd == TRACK_BANK_RIGHT)
                && (_currentTrackCurve == TRACK_CURVE_LEFT || _currentTrackCurve == TRACK_CURVE_RIGHT))
            {
                if (IsTrackEnabled(TRACK_HELIX_DOWN_BANKED_HALF) || IsTrackEnabled(TRACK_HELIX_DOWN_BANKED_QUARTER))
                    disabledWidgets &= ~(1uLL << WIDX_SLOPE_DOWN_STEEP);
                if (IsTrackEnabled(TRACK_HELIX_UP_BANKED_HALF) || IsTrackEnabled(TRACK_HELIX_UP_BANKED_QUARTER))
                    disabledWidgets &= ~(1uLL << WIDX_SLOPE_UP_STEEP);
            }
            // If the track is banked left or right and curvature is small size (1.5 tile radius), attempt to show buttons for
            // half helixes
            else if (
                (_currentTrackBankEnd == TRACK_BANK_LEFT || _currentTrackBankEnd == TRACK_BANK_RIGHT)
                && (_currentTrackCurve == TRACK_CURVE_LEFT_SMALL || _currentTrackCurve == TRACK_CURVE_RIGHT_SMALL))
            {
                if (IsTrackEnabled(TRACK_HELIX_DOWN_BANKED_HALF))
                    disabledWidgets &= ~(1uLL << WIDX_SLOPE_DOWN_STEEP);
                if (IsTrackEnabled(TRACK_HELIX_UP_BANKED_HALF))
                    disabledWidgets &= ~(1uLL << WIDX_SLOPE_UP_STEEP);
            }
        }
        if (IsTrackEnabled(TRACK_SLOPE_CURVE_BANKED))
        {
            if (_rideConstructionState == RideConstructionState::Front)
            {
                if (_currentTrackCurve == TRACK_CURVE_LEFT_SMALL || _currentTrackCurve == TRACK_CURVE_RIGHT_SMALL)
                {
                    if (_currentTrackSlopeEnd == TRACK_SLOPE_NONE && _previousTrackBankEnd != TRACK_BANK_NONE
                        && (!currentRide->GetRideTypeDescriptor().HasFlag(RIDE_TYPE_FLAG_UP_INCLINE_REQUIRES_LIFT)
                            || gCheatsEnableAllDrawableTrackPieces))
                    {
                        disabledWidgets &= ~(1uLL << WIDX_SLOPE_UP);
                    }
                }
            }
            else if (_rideConstructionState == RideConstructionState::Back)
            {
                if (_currentTrackCurve == TRACK_CURVE_LEFT_SMALL || _currentTrackCurve == TRACK_CURVE_RIGHT_SMALL)
                {
                    if (_currentTrackSlopeEnd == TRACK_SLOPE_NONE && _previousTrackBankEnd != TRACK_BANK_NONE)
                    {
                        disabledWidgets &= ~(1uLL << WIDX_SLOPE_DOWN);
                    }
                }
            }
        }
        if (_currentTrackPieceDirection >= 4)
        {
            disabledWidgets |= (1uLL << WIDX_LEFT_CURVE_VERY_SMALL) | (1uLL << WIDX_LEFT_CURVE_SMALL)
                | (1uLL << WIDX_LEFT_CURVE) | (1uLL << WIDX_RIGHT_CURVE) | (1uLL << WIDX_RIGHT_CURVE_SMALL)
                | (1uLL << WIDX_RIGHT_CURVE_VERY_SMALL);
        }
        if (_rideConstructionState == RideConstructionState::Front)
        {
            disabledWidgets |= (1uLL << WIDX_NEXT_SECTION);
            if (WindowRideConstructionUpdateState(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr))
            {
                disabledWidgets |= (1uLL << WIDX_CONSTRUCT);
            }
        }
        else if (_rideConstructionState == RideConstructionState::Back)
        {
            disabledWidgets |= (1uLL << WIDX_PREVIOUS_SECTION);
            if (WindowRideConstructionUpdateState(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr))
            {
                disabledWidgets |= (1uLL << WIDX_CONSTRUCT);
            }
        }
        if (GetRideTypeDescriptor(rideType).HasFlag(RIDE_TYPE_FLAG_TRACK_ELEMENTS_HAVE_TWO_VARIETIES))
        {
            disabledWidgets &= ~(1uLL << WIDX_BANKING_GROUPBOX);
        }
        if (_rideConstructionState == RideConstructionState::EntranceExit
            || _rideConstructionState == RideConstructionState::Selected)
        {
            disabledWidgets |= (1uLL << WIDX_DIRECTION_GROUPBOX) | (1uLL << WIDX_SLOPE_GROUPBOX)
                | (1uLL << WIDX_BANKING_GROUPBOX) | (1uLL << WIDX_LEFT_CURVE_VERY_SMALL) | (1uLL << WIDX_LEFT_CURVE_SMALL)
                | (1uLL << WIDX_LEFT_CURVE) | (1uLL << WIDX_STRAIGHT) | (1uLL << WIDX_RIGHT_CURVE)
                | (1uLL << WIDX_RIGHT_CURVE_SMALL) | (1uLL << WIDX_RIGHT_CURVE_VERY_SMALL)
                | (1uLL << WIDX_SPECIAL_TRACK_DROPDOWN) | (1uLL << WIDX_SLOPE_DOWN_STEEP) | (1uLL << WIDX_SLOPE_DOWN)
                | (1uLL << WIDX_LEVEL) | (1uLL << WIDX_SLOPE_UP) | (1uLL << WIDX_SLOPE_UP_STEEP) | (1uLL << WIDX_CHAIN_LIFT)
                | (1uLL << WIDX_BANK_LEFT) | (1uLL << WIDX_BANK_STRAIGHT) | (1uLL << WIDX_BANK_RIGHT)
                | (1uLL << WIDX_LEFT_CURVE_LARGE) | (1uLL << WIDX_RIGHT_CURVE_LARGE);
        }
        if (_currentlyShowingBrakeOrBoosterSpeed)
        {
            disabledWidgets &= ~(1uLL << WIDX_BANKING_GROUPBOX);
            disabledWidgets &= ~(1uLL << WIDX_BANK_LEFT);
            disabledWidgets &= ~(1uLL << WIDX_BANK_STRAIGHT);
            disabledWidgets &= ~(1uLL << WIDX_BANK_RIGHT);
        }

        // If chain lift cheat is enabled then show the chain lift widget no matter what
        if (gCheatsEnableChainLiftOnAllTrack)
        {
            disabledWidgets &= ~(1uLL << WIDX_CHAIN_LIFT);
        }

        // Set and invalidate the changed widgets
        uint64_t currentDisabledWidgets = disabled_widgets;
        if (currentDisabledWidgets == disabledWidgets)
            return;

        for (WidgetIndex i = 0; i < 64; i++)
        {
            if ((disabledWidgets & (1uLL << i)) != (currentDisabledWidgets & (1uLL << i)))
            {
                WidgetInvalidate(*this, i);
            }
        }
        disabled_widgets = disabledWidgets;
    }

    void OnUpdate() override
    {
        auto currentRide = GetRide(_currentRideIndex);
        if (currentRide == nullptr)
        {
            return;
        }

        // Close construction window if currentRide is not closed,
        // editing currentRide while open will cause many issues until properly handled
        if (currentRide->status != RideStatus::Closed && currentRide->status != RideStatus::Simulating)
        {
            Close();
            return;
        }

        switch (_currentTrackCurve)
        {
            case TrackElemType::SpinningTunnel | RideConstructionSpecialPieceSelected:
            case TrackElemType::Whirlpool | RideConstructionSpecialPieceSelected:
            case TrackElemType::Rapids | RideConstructionSpecialPieceSelected:
            case TrackElemType::Waterfall | RideConstructionSpecialPieceSelected:
                WidgetInvalidate(*this, WIDX_CONSTRUCT);
                break;
        }

        if (_rideConstructionState == RideConstructionState::Place)
        {
            if (!WidgetIsActiveTool(*this, WIDX_CONSTRUCT))
            {
                Close();
                return;
            }
        }

        if (_rideConstructionState == RideConstructionState::EntranceExit)
        {
            if (!WidgetIsActiveTool(*this, WIDX_ENTRANCE) && !WidgetIsActiveTool(*this, WIDX_EXIT))
            {
                _rideConstructionState = gRideEntranceExitPlacePreviousRideConstructionState;
                WindowRideConstructionUpdateActiveElements();
            }
        }

        switch (_rideConstructionState)
        {
            case RideConstructionState::Front:
            case RideConstructionState::Back:
            case RideConstructionState::Selected:
                if ((InputTestFlag(INPUT_FLAG_TOOL_ACTIVE))
                    && gCurrentToolWidget.window_classification == WindowClass::RideConstruction)
                {
                    ToolCancel();
                }
                break;
            default:
                break;
        }

        UpdateGhostTrackAndArrow();
    }

    void OnMouseUp(WidgetIndex widgetIndex) override
    {
        WindowRideConstructionUpdateEnabledTrackPieces();
        switch (widgetIndex)
        {
            case WIDX_CLOSE:
                Close();
                break;
            case WIDX_NEXT_SECTION:
                RideSelectNextSection();
                break;
            case WIDX_PREVIOUS_SECTION:
                RideSelectPreviousSection();
                break;
            case WIDX_CONSTRUCT:
                Construct();
                // Force any footpath construction to recheck the area.
                gProvisionalFootpath.Flags |= PROVISIONAL_PATH_FLAG_2;
                break;
            case WIDX_DEMOLISH:
                MouseUpDemolish();
                break;
            case WIDX_ROTATE:
                Rotate();
                break;
            case WIDX_ENTRANCE:
                EntranceClick();
                break;
            case WIDX_EXIT:
                ExitClick();
                break;
            case WIDX_SIMULATE:
            {
                auto currentRide = GetRide(_currentRideIndex);
                if (currentRide != nullptr)
                {
                    auto status = currentRide->status == RideStatus::Simulating ? RideStatus::Closed : RideStatus::Simulating;
                    auto gameAction = RideSetStatusAction(currentRide->id, status);
                    GameActions::Execute(&gameAction);
                }
                break;
            }
        }
    }

    void OnMouseDown(WidgetIndex widgetIndex) override
    {
        auto currentRide = GetRide(_currentRideIndex);
        if (currentRide == nullptr)
        {
            return;
        }

        WindowRideConstructionUpdateEnabledTrackPieces();
        switch (widgetIndex)
        {
            case WIDX_LEFT_CURVE:
                RideConstructionInvalidateCurrentTrack();
                _currentTrackCurve = TRACK_CURVE_LEFT;
                _currentTrackPrice = MONEY64_UNDEFINED;
                WindowRideConstructionUpdateActiveElements();
                break;
            case WIDX_RIGHT_CURVE:
                RideConstructionInvalidateCurrentTrack();
                _currentTrackCurve = TRACK_CURVE_RIGHT;
                _currentTrackPrice = MONEY64_UNDEFINED;
                WindowRideConstructionUpdateActiveElements();
                break;
            case WIDX_LEFT_CURVE_SMALL:
                RideConstructionInvalidateCurrentTrack();
                _currentTrackCurve = TRACK_CURVE_LEFT_SMALL;
                _currentTrackPrice = MONEY64_UNDEFINED;
                WindowRideConstructionUpdateActiveElements();
                break;
            case WIDX_RIGHT_CURVE_SMALL:
                RideConstructionInvalidateCurrentTrack();
                _currentTrackCurve = TRACK_CURVE_RIGHT_SMALL;
                _currentTrackPrice = MONEY64_UNDEFINED;
                WindowRideConstructionUpdateActiveElements();
                break;
            case WIDX_LEFT_CURVE_VERY_SMALL:
                RideConstructionInvalidateCurrentTrack();
                _currentTrackCurve = TRACK_CURVE_LEFT_VERY_SMALL;
                _currentTrackPrice = MONEY64_UNDEFINED;
                WindowRideConstructionUpdateActiveElements();
                break;
            case WIDX_RIGHT_CURVE_VERY_SMALL:
                RideConstructionInvalidateCurrentTrack();
                _currentTrackCurve = TRACK_CURVE_RIGHT_VERY_SMALL;
                _currentTrackPrice = MONEY64_UNDEFINED;
                WindowRideConstructionUpdateActiveElements();
                break;
            case WIDX_LEFT_CURVE_LARGE:
                RideConstructionInvalidateCurrentTrack();
                _currentTrackCurve = TRACK_CURVE_LEFT_LARGE;
                _currentTrackPrice = MONEY64_UNDEFINED;
                WindowRideConstructionUpdateActiveElements();
                break;
            case WIDX_RIGHT_CURVE_LARGE:
                RideConstructionInvalidateCurrentTrack();
                _currentTrackCurve = TRACK_CURVE_RIGHT_LARGE;
                _currentTrackPrice = MONEY64_UNDEFINED;
                WindowRideConstructionUpdateActiveElements();
                break;
            case WIDX_STRAIGHT:
                RideConstructionInvalidateCurrentTrack();
                if (_currentTrackCurve != TRACK_CURVE_NONE)
                    _currentTrackBankEnd = TRACK_BANK_NONE;
                _currentTrackCurve = TRACK_CURVE_NONE;
                _currentTrackPrice = MONEY64_UNDEFINED;
                WindowRideConstructionUpdateActiveElements();
                break;
            case WIDX_SLOPE_DOWN_STEEP:
                RideConstructionInvalidateCurrentTrack();
                if (IsTrackEnabled(TRACK_HELIX_DOWN_BANKED_HALF) || IsTrackEnabled(TRACK_HELIX_UP_BANKED_HALF))
                {
                    if (_currentTrackCurve == TRACK_CURVE_LEFT && _currentTrackBankEnd == TRACK_BANK_LEFT)
                    {
                        _currentTrackCurve = TrackElemType::LeftHalfBankedHelixDownLarge | RideConstructionSpecialPieceSelected;
                        _currentTrackPrice = MONEY64_UNDEFINED;
                        WindowRideConstructionUpdateActiveElements();
                        break;
                    }
                    if (_currentTrackCurve == TRACK_CURVE_RIGHT && _currentTrackBankEnd == TRACK_BANK_RIGHT)
                    {
                        _currentTrackCurve = TrackElemType::RightHalfBankedHelixDownLarge
                            | RideConstructionSpecialPieceSelected;
                        _currentTrackPrice = MONEY64_UNDEFINED;
                        WindowRideConstructionUpdateActiveElements();
                        break;
                    }
                    if (_currentTrackCurve == TRACK_CURVE_LEFT_SMALL && _currentTrackBankEnd == TRACK_BANK_LEFT)
                    {
                        _currentTrackCurve = TrackElemType::LeftHalfBankedHelixDownSmall | RideConstructionSpecialPieceSelected;
                        _currentTrackPrice = MONEY64_UNDEFINED;
                        WindowRideConstructionUpdateActiveElements();
                        break;
                    }
                    if (_currentTrackCurve == TRACK_CURVE_RIGHT_SMALL && _currentTrackBankEnd == TRACK_BANK_RIGHT)
                    {
                        _currentTrackCurve = TrackElemType::RightHalfBankedHelixDownSmall
                            | RideConstructionSpecialPieceSelected;
                        _currentTrackPrice = MONEY64_UNDEFINED;
                        WindowRideConstructionUpdateActiveElements();
                        break;
                    }
                }
                if (IsTrackEnabled(TRACK_HELIX_DOWN_BANKED_QUARTER) || IsTrackEnabled(TRACK_HELIX_UP_BANKED_QUARTER))
                {
                    if (_currentTrackCurve == TRACK_CURVE_LEFT && _currentTrackBankEnd == TRACK_BANK_LEFT)
                    {
                        _currentTrackCurve = TrackElemType::LeftQuarterBankedHelixLargeDown
                            | RideConstructionSpecialPieceSelected;
                        _currentTrackPrice = MONEY64_UNDEFINED;
                        WindowRideConstructionUpdateActiveElements();
                        break;
                    }
                    if (_currentTrackCurve == TRACK_CURVE_RIGHT && _currentTrackBankEnd == TRACK_BANK_RIGHT)
                    {
                        _currentTrackCurve = TrackElemType::RightQuarterBankedHelixLargeDown
                            | RideConstructionSpecialPieceSelected;
                        _currentTrackPrice = MONEY64_UNDEFINED;
                        WindowRideConstructionUpdateActiveElements();
                        break;
                    }
                }
                if (IsTrackEnabled(TRACK_HELIX_DOWN_UNBANKED_QUARTER) || IsTrackEnabled(TRACK_HELIX_UP_UNBANKED_QUARTER))
                {
                    if (_currentTrackBankEnd == TRACK_BANK_NONE)
                    {
                        if (_currentTrackCurve == TRACK_CURVE_LEFT)
                        {
                            _currentTrackCurve = TrackElemType::LeftQuarterHelixLargeDown
                                | RideConstructionSpecialPieceSelected;
                            _currentTrackPrice = MONEY64_UNDEFINED;
                            WindowRideConstructionUpdateActiveElements();
                            break;
                        }
                        if (_currentTrackCurve == TRACK_CURVE_RIGHT)
                        {
                            _currentTrackCurve = TrackElemType::RightQuarterHelixLargeDown
                                | RideConstructionSpecialPieceSelected;
                            _currentTrackPrice = MONEY64_UNDEFINED;
                            WindowRideConstructionUpdateActiveElements();
                            break;
                        }
                    }
                }
                if (widgets[WIDX_SLOPE_DOWN_STEEP].tooltip == STR_RIDE_CONSTRUCTION_STEEP_SLOPE_DOWN_TIP)
                {
                    UpdateLiftHillSelected(TRACK_SLOPE_DOWN_60);
                }
                else
                {
                    UpdateLiftHillSelected(TRACK_SLOPE_UP_90);
                }
                break;
            case WIDX_SLOPE_DOWN:
                RideConstructionInvalidateCurrentTrack();
                if (_rideConstructionState == RideConstructionState::Back && _currentTrackBankEnd != TRACK_BANK_NONE)
                {
                    _currentTrackBankEnd = TRACK_BANK_NONE;
                }
                UpdateLiftHillSelected(TRACK_SLOPE_DOWN_25);
                break;
            case WIDX_LEVEL:
                RideConstructionInvalidateCurrentTrack();
                if (_rideConstructionState == RideConstructionState::Front && _previousTrackSlopeEnd == 6)
                {
                    if (_currentTrackCurve == TRACK_CURVE_LEFT_SMALL)
                    {
                        _currentTrackBankEnd = TRACK_BANK_LEFT;
                    }
                    else if (_currentTrackCurve == TRACK_CURVE_RIGHT_SMALL)
                    {
                        _currentTrackBankEnd = TRACK_BANK_RIGHT;
                    }
                }
                else if (_rideConstructionState == RideConstructionState::Back && _previousTrackSlopeEnd == 2)
                {
                    if (_currentTrackCurve == TRACK_CURVE_LEFT_SMALL)
                    {
                        _currentTrackBankEnd = TRACK_BANK_LEFT;
                    }
                    else if (_currentTrackCurve == TRACK_CURVE_RIGHT_SMALL)
                    {
                        _currentTrackBankEnd = TRACK_BANK_RIGHT;
                    }
                }
                UpdateLiftHillSelected(TRACK_SLOPE_NONE);
                break;
            case WIDX_SLOPE_UP:
                RideConstructionInvalidateCurrentTrack();
                if (_rideConstructionState == RideConstructionState::Front && _currentTrackBankEnd != TRACK_BANK_NONE)
                {
                    _currentTrackBankEnd = TRACK_BANK_NONE;
                }
                if (currentRide->GetRideTypeDescriptor().SupportsTrackPiece(TRACK_REVERSE_FREEFALL))
                {
                    if (_rideConstructionState == RideConstructionState::Front && _currentTrackCurve == TRACK_CURVE_NONE)
                    {
                        _currentTrackCurve = TrackElemType::ReverseFreefallSlope | RideConstructionSpecialPieceSelected;
                        WindowRideConstructionUpdateActiveElements();
                    }
                }
                else
                {
                    UpdateLiftHillSelected(TRACK_SLOPE_UP_25);
                }
                break;
            case WIDX_SLOPE_UP_STEEP:
                RideConstructionInvalidateCurrentTrack();
                if (IsTrackEnabled(TRACK_HELIX_DOWN_BANKED_HALF) || IsTrackEnabled(TRACK_HELIX_UP_BANKED_HALF))
                {
                    if (_currentTrackCurve == TRACK_CURVE_LEFT && _currentTrackBankEnd == TRACK_BANK_LEFT)
                    {
                        _currentTrackCurve = TrackElemType::LeftHalfBankedHelixUpLarge | RideConstructionSpecialPieceSelected;
                        _currentTrackPrice = MONEY64_UNDEFINED;
                        WindowRideConstructionUpdateActiveElements();
                        break;
                    }
                    if (_currentTrackCurve == TRACK_CURVE_RIGHT && _currentTrackBankEnd == TRACK_BANK_RIGHT)
                    {
                        _currentTrackCurve = TrackElemType::RightHalfBankedHelixUpLarge | RideConstructionSpecialPieceSelected;
                        _currentTrackPrice = MONEY64_UNDEFINED;
                        WindowRideConstructionUpdateActiveElements();
                        break;
                    }
                    if (_currentTrackCurve == TRACK_CURVE_LEFT_SMALL && _currentTrackBankEnd == TRACK_BANK_LEFT)
                    {
                        _currentTrackCurve = TrackElemType::LeftHalfBankedHelixUpSmall | RideConstructionSpecialPieceSelected;
                        _currentTrackPrice = MONEY64_UNDEFINED;
                        WindowRideConstructionUpdateActiveElements();
                        break;
                    }
                    if (_currentTrackCurve == TRACK_CURVE_RIGHT_SMALL && _currentTrackBankEnd == TRACK_BANK_RIGHT)
                    {
                        _currentTrackCurve = TrackElemType::RightHalfBankedHelixUpSmall | RideConstructionSpecialPieceSelected;
                        _currentTrackPrice = MONEY64_UNDEFINED;
                        WindowRideConstructionUpdateActiveElements();
                        break;
                    }
                }
                if (IsTrackEnabled(TRACK_HELIX_DOWN_BANKED_QUARTER) || IsTrackEnabled(TRACK_HELIX_UP_BANKED_QUARTER))
                {
                    if (_currentTrackCurve == TRACK_CURVE_LEFT && _currentTrackBankEnd == TRACK_BANK_LEFT)
                    {
                        _currentTrackCurve = TrackElemType::LeftQuarterBankedHelixLargeUp
                            | RideConstructionSpecialPieceSelected;
                        _currentTrackPrice = MONEY64_UNDEFINED;
                        WindowRideConstructionUpdateActiveElements();
                        break;
                    }
                    if (_currentTrackCurve == TRACK_CURVE_RIGHT && _currentTrackBankEnd == TRACK_BANK_RIGHT)
                    {
                        _currentTrackCurve = TrackElemType::RightQuarterBankedHelixLargeUp
                            | RideConstructionSpecialPieceSelected;
                        _currentTrackPrice = MONEY64_UNDEFINED;
                        WindowRideConstructionUpdateActiveElements();
                        break;
                    }
                }
                if (IsTrackEnabled(TRACK_HELIX_DOWN_UNBANKED_QUARTER) || IsTrackEnabled(TRACK_HELIX_UP_UNBANKED_QUARTER))
                {
                    if (_currentTrackBankEnd == TRACK_BANK_NONE)
                    {
                        if (_currentTrackCurve == TRACK_CURVE_LEFT)
                        {
                            _currentTrackCurve = TrackElemType::LeftQuarterHelixLargeUp | RideConstructionSpecialPieceSelected;
                            _currentTrackPrice = MONEY64_UNDEFINED;
                            WindowRideConstructionUpdateActiveElements();
                            break;
                        }
                        if (_currentTrackCurve == TRACK_CURVE_RIGHT)
                        {
                            _currentTrackCurve = TrackElemType::RightQuarterHelixLargeUp | RideConstructionSpecialPieceSelected;
                            _currentTrackPrice = MONEY64_UNDEFINED;
                            WindowRideConstructionUpdateActiveElements();
                            break;
                        }
                    }
                }
                if (widgets[WIDX_SLOPE_UP_STEEP].tooltip == STR_RIDE_CONSTRUCTION_STEEP_SLOPE_UP_TIP)
                {
                    UpdateLiftHillSelected(TRACK_SLOPE_UP_60);
                }
                else
                {
                    UpdateLiftHillSelected(TRACK_SLOPE_DOWN_90);
                }
                break;
            case WIDX_CHAIN_LIFT:
                RideConstructionInvalidateCurrentTrack();
                _currentTrackLiftHill ^= CONSTRUCTION_LIFT_HILL_SELECTED;
                if ((_currentTrackLiftHill & CONSTRUCTION_LIFT_HILL_SELECTED) && !gCheatsEnableChainLiftOnAllTrack)
                    _currentTrackAlternative &= ~RIDE_TYPE_ALTERNATIVE_TRACK_PIECES;
                _currentTrackPrice = MONEY64_UNDEFINED;
                WindowRideConstructionUpdateActiveElements();
                break;
            case WIDX_BANK_LEFT:
                RideConstructionInvalidateCurrentTrack();
                if (!_currentlyShowingBrakeOrBoosterSpeed)
                {
                    _currentTrackBankEnd = TRACK_BANK_LEFT;
                    _currentTrackPrice = MONEY64_UNDEFINED;
                    WindowRideConstructionUpdateActiveElements();
                }
                break;
            case WIDX_BANK_STRAIGHT:
                RideConstructionInvalidateCurrentTrack();
                if (!_currentlyShowingBrakeOrBoosterSpeed)
                {
                    _currentTrackBankEnd = TRACK_BANK_NONE;
                    _currentTrackPrice = MONEY64_UNDEFINED;
                    WindowRideConstructionUpdateActiveElements();
                }
                else
                {
                    uint8_t* brakesSpeedPtr = &_currentBrakeSpeed2;
                    uint8_t maxBrakesSpeed = 30;
                    uint8_t brakesSpeed = *brakesSpeedPtr + 2;
                    if (brakesSpeed <= maxBrakesSpeed)
                    {
                        if (_rideConstructionState == RideConstructionState::Selected)
                        {
                            SetBrakeSpeed(brakesSpeed);
                        }
                        else
                        {
                            *brakesSpeedPtr = brakesSpeed;
                            WindowRideConstructionUpdateActiveElements();
                        }
                    }
                }
                break;
            case WIDX_BANK_RIGHT:
                RideConstructionInvalidateCurrentTrack();
                if (!_currentlyShowingBrakeOrBoosterSpeed)
                {
                    _currentTrackBankEnd = TRACK_BANK_RIGHT;
                    _currentTrackPrice = MONEY64_UNDEFINED;
                    WindowRideConstructionUpdateActiveElements();
                }
                else
                {
                    uint8_t* brakesSpeedPtr = &_currentBrakeSpeed2;
                    uint8_t brakesSpeed = *brakesSpeedPtr - 2;
                    if (brakesSpeed >= 2)
                    {
                        if (_rideConstructionState == RideConstructionState::Selected)
                        {
                            SetBrakeSpeed(brakesSpeed);
                        }
                        else
                        {
                            *brakesSpeedPtr = brakesSpeed;
                            WindowRideConstructionUpdateActiveElements();
                        }
                    }
                }
                break;
            case WIDX_SPECIAL_TRACK_DROPDOWN:
                ShowSpecialTrackDropdown(&widgets[widgetIndex]);
                break;
            case WIDX_U_TRACK:
                RideConstructionInvalidateCurrentTrack();
                _currentTrackAlternative &= ~RIDE_TYPE_ALTERNATIVE_TRACK_PIECES;
                _currentTrackPrice = MONEY64_UNDEFINED;
                WindowRideConstructionUpdateActiveElements();
                break;
            case WIDX_O_TRACK:
                RideConstructionInvalidateCurrentTrack();
                _currentTrackAlternative |= RIDE_TYPE_ALTERNATIVE_TRACK_PIECES;
                if (!gCheatsEnableChainLiftOnAllTrack)
                    _currentTrackLiftHill &= ~CONSTRUCTION_LIFT_HILL_SELECTED;
                _currentTrackPrice = MONEY64_UNDEFINED;
                WindowRideConstructionUpdateActiveElements();
                break;
            case WIDX_SEAT_ROTATION_ANGLE_SPINNER_UP:
                if (_currentSeatRotationAngle < 15)
                {
                    if (_rideConstructionState == RideConstructionState::Selected)
                    {
                        RideSelectedTrackSetSeatRotation(_currentSeatRotationAngle + 1);
                    }
                    else
                    {
                        _currentSeatRotationAngle++;
                        WindowRideConstructionUpdateActiveElements();
                    }
                }
                break;
            case WIDX_SEAT_ROTATION_ANGLE_SPINNER_DOWN:
                if (_currentSeatRotationAngle > 0)
                {
                    if (_rideConstructionState == RideConstructionState::Selected)
                    {
                        RideSelectedTrackSetSeatRotation(_currentSeatRotationAngle - 1);
                    }
                    else
                    {
                        _currentSeatRotationAngle--;
                        WindowRideConstructionUpdateActiveElements();
                    }
                }
                break;
        }
    }

    void OnDropdown(WidgetIndex widgetIndex, int32_t selectedIndex) override
    {
        if (widgetIndex != WIDX_SPECIAL_TRACK_DROPDOWN)
            return;
        if (selectedIndex == -1)
            return;

        RideConstructionInvalidateCurrentTrack();
        _currentTrackPrice = MONEY64_UNDEFINED;
        track_type_t trackPiece = _specialElementDropdownState.Elements[selectedIndex].TrackType;
        switch (trackPiece)
        {
            case TrackElemType::EndStation:
            case TrackElemType::SBendLeft:
            case TrackElemType::SBendRight:
                _currentTrackSlopeEnd = 0;
                break;
            case TrackElemType::LeftVerticalLoop:
            case TrackElemType::RightVerticalLoop:
                _currentTrackBankEnd = TRACK_BANK_NONE;
                _currentTrackLiftHill &= ~CONSTRUCTION_LIFT_HILL_SELECTED;
                break;
            case TrackElemType::BlockBrakes:
            case TrackElemType::DiagBlockBrakes:
                _currentBrakeSpeed2 = kRCT2DefaultBlockBrakeSpeed;
        }
        _currentTrackCurve = trackPiece | RideConstructionSpecialPieceSelected;
        WindowRideConstructionUpdateActiveElements();
    }

    void OnToolUpdate(WidgetIndex widgetIndex, const ScreenCoordsXY& screenCoords) override
    {
        switch (widgetIndex)
        {
            case WIDX_CONSTRUCT:
                RideConstructionToolupdateConstruct(screenCoords);
                break;
            case WIDX_ENTRANCE:
            case WIDX_EXIT:
                RideConstructionToolupdateEntranceExit(screenCoords);
                break;
        }
    }

    void OnToolDown(WidgetIndex widgetIndex, const ScreenCoordsXY& screenCoords) override
    {
        switch (widgetIndex)
        {
            case WIDX_CONSTRUCT:
                RideConstructionTooldownConstruct(screenCoords);
                break;
            case WIDX_ENTRANCE:
            case WIDX_EXIT:
                ToolDownEntranceExit(screenCoords);
                break;
        }
    }

    void OnPrepareDraw() override
    {
        auto currentRide = GetRide(_currentRideIndex);
        if (currentRide == nullptr)
        {
            return;
        }

        StringId stringId = STR_RIDE_CONSTRUCTION_SPECIAL;
        if (_currentTrackCurve & RideConstructionSpecialPieceSelected)
        {
            const auto& rtd = currentRide->GetRideTypeDescriptor();
            const auto& ted = GetTrackElementDescriptor(_currentTrackCurve & ~RideConstructionSpecialPieceSelected);
            stringId = ted.Description;
            if (stringId == STR_RAPIDS && rtd.Category != RIDE_CATEGORY_WATER)
            {
                stringId = STR_LOG_BUMPS;
            }
        }
        auto ft = Formatter::Common();
        ft.Add<uint16_t>(stringId);

        if (_currentlyShowingBrakeOrBoosterSpeed)
        {
            uint16_t brakeSpeed2 = ((_currentBrakeSpeed2 * 9) >> 2) & 0xFFFF;
            if (_selectedTrackType == TrackElemType::Booster
                || _currentTrackCurve == (RideConstructionSpecialPieceSelected | TrackElemType::Booster))
            {
                brakeSpeed2 = GetBoosterSpeed(currentRide->type, brakeSpeed2);
            }
            ft.Add<uint16_t>(brakeSpeed2);
        }

        widgets[WIDX_SEAT_ROTATION_ANGLE_SPINNER].text = RideConstructionSeatAngleRotationStrings[_currentSeatRotationAngle];

        // Simulate button
        auto& simulateWidget = widgets[WIDX_SIMULATE];
        simulateWidget.type = WindowWidgetType::Empty;
        if (currentRide->SupportsStatus(RideStatus::Simulating))
        {
            simulateWidget.type = WindowWidgetType::FlatBtn;
            if (currentRide->status == RideStatus::Simulating)
            {
                pressed_widgets |= (1uLL << WIDX_SIMULATE);
            }
            else
            {
                pressed_widgets &= ~(1uLL << WIDX_SIMULATE);
            }
        }

        // Set window title arguments
        ft = Formatter::Common();
        ft.Increment(4);
        currentRide->FormatNameTo(ft);
    }

    void OnDraw(DrawPixelInfo& dpi) override
    {
        DrawPixelInfo clipdpi;
        Widget* widget;
        int32_t widgetWidth, widgetHeight;

        DrawWidgets(dpi);

        widget = &widgets[WIDX_CONSTRUCT];
        if (widget->type == WindowWidgetType::Empty)
            return;

        RideId rideIndex;
        int32_t trackType, trackDirection, liftHillAndInvertedState;
        if (WindowRideConstructionUpdateState(
                &trackType, &trackDirection, &rideIndex, &liftHillAndInvertedState, nullptr, nullptr))
            return;

        // Draw track piece
        auto screenCoords = ScreenCoordsXY{ windowPos.x + widget->left + 1, windowPos.y + widget->top + 1 };
        widgetWidth = widget->width() - 1;
        widgetHeight = widget->height() - 1;
        if (ClipDrawPixelInfo(clipdpi, dpi, screenCoords, widgetWidth, widgetHeight))
        {
            DrawTrackPiece(clipdpi, rideIndex, trackType, trackDirection, liftHillAndInvertedState, widgetWidth, widgetHeight);
        }

        // Draw cost
        screenCoords = { windowPos.x + widget->midX(), windowPos.y + widget->bottom - 23 };
        if (_rideConstructionState != RideConstructionState::Place)
            DrawTextBasic(dpi, screenCoords, STR_BUILD_THIS, {}, { TextAlignment::CENTRE });

        screenCoords.y += 11;
        if (_currentTrackPrice != MONEY64_UNDEFINED && !(gParkFlags & PARK_FLAGS_NO_MONEY))
        {
            auto ft = Formatter();
            ft.Add<money64>(_currentTrackPrice);
            DrawTextBasic(dpi, screenCoords, STR_COST_LABEL, ft, { TextAlignment::CENTRE });
        }
    }

    void UpdateWidgets()
    {
        auto currentRide = GetRide(_currentRideIndex);
        if (currentRide == nullptr)
        {
            return;
        }
        int32_t rideType = RideGetAlternativeType(*currentRide);

        hold_down_widgets = 0;
        if (GetRideTypeDescriptor(rideType).HasFlag(RIDE_TYPE_FLAG_IS_SHOP_OR_FACILITY) || !currentRide->HasStation())
        {
            widgets[WIDX_ENTRANCE_EXIT_GROUPBOX].type = WindowWidgetType::Empty;
            widgets[WIDX_ENTRANCE].type = WindowWidgetType::Empty;
            widgets[WIDX_EXIT].type = WindowWidgetType::Empty;
        }
        else
        {
            widgets[WIDX_ENTRANCE_EXIT_GROUPBOX].type = WindowWidgetType::Groupbox;
            widgets[WIDX_ENTRANCE].type = WindowWidgetType::Button;
            widgets[WIDX_EXIT].type = WindowWidgetType::Button;
        }

        if (_specialElementDropdownState.HasActiveElements)
        {
            widgets[WIDX_SPECIAL_TRACK_DROPDOWN].type = WindowWidgetType::Button;
        }
        else
        {
            widgets[WIDX_SPECIAL_TRACK_DROPDOWN].type = WindowWidgetType::Empty;
        }

        if (IsTrackEnabled(TRACK_STRAIGHT))
        {
            widgets[WIDX_STRAIGHT].type = WindowWidgetType::FlatBtn;
        }
        else
        {
            widgets[WIDX_STRAIGHT].type = WindowWidgetType::Empty;
        }

        if (IsTrackEnabled(TRACK_CURVE_LARGE))
        {
            widgets[WIDX_LEFT_CURVE_LARGE].type = WindowWidgetType::FlatBtn;
            widgets[WIDX_RIGHT_CURVE_LARGE].type = WindowWidgetType::FlatBtn;
        }
        else
        {
            widgets[WIDX_LEFT_CURVE_LARGE].type = WindowWidgetType::Empty;
            widgets[WIDX_RIGHT_CURVE_LARGE].type = WindowWidgetType::Empty;
        }

        widgets[WIDX_LEFT_CURVE].type = WindowWidgetType::Empty;
        widgets[WIDX_RIGHT_CURVE].type = WindowWidgetType::Empty;
        widgets[WIDX_LEFT_CURVE_SMALL].type = WindowWidgetType::Empty;
        widgets[WIDX_RIGHT_CURVE_SMALL].type = WindowWidgetType::Empty;
        widgets[WIDX_LEFT_CURVE_VERY_SMALL].type = WindowWidgetType::Empty;
        widgets[WIDX_RIGHT_CURVE_VERY_SMALL].type = WindowWidgetType::Empty;
        widgets[WIDX_LEFT_CURVE_SMALL].left = 28;
        widgets[WIDX_LEFT_CURVE_SMALL].right = 49;
        widgets[WIDX_RIGHT_CURVE_SMALL].left = 116;
        widgets[WIDX_RIGHT_CURVE_SMALL].right = 137;
        widgets[WIDX_LEFT_CURVE_SMALL].image = ImageId(SPR_RIDE_CONSTRUCTION_LEFT_CURVE);
        widgets[WIDX_RIGHT_CURVE_SMALL].image = ImageId(SPR_RIDE_CONSTRUCTION_RIGHT_CURVE);
        if (IsTrackEnabled(TRACK_CURVE_VERTICAL))
        {
            widgets[WIDX_LEFT_CURVE_SMALL].type = WindowWidgetType::FlatBtn;
            widgets[WIDX_LEFT_CURVE_SMALL].left = 6;
            widgets[WIDX_LEFT_CURVE_SMALL].right = 27;
            widgets[WIDX_LEFT_CURVE_SMALL].image = ImageId(SPR_RIDE_CONSTRUCTION_LEFT_CURVE_SMALL);
            widgets[WIDX_RIGHT_CURVE_SMALL].type = WindowWidgetType::FlatBtn;
            widgets[WIDX_RIGHT_CURVE_SMALL].left = 138;
            widgets[WIDX_RIGHT_CURVE_SMALL].right = 159;
            widgets[WIDX_RIGHT_CURVE_SMALL].image = ImageId(SPR_RIDE_CONSTRUCTION_RIGHT_CURVE_SMALL);
        }
        if (IsTrackEnabled(TRACK_CURVE))
        {
            widgets[WIDX_LEFT_CURVE].type = WindowWidgetType::FlatBtn;
            widgets[WIDX_RIGHT_CURVE].type = WindowWidgetType::FlatBtn;
            widgets[WIDX_LEFT_CURVE_SMALL].left = 6;
            widgets[WIDX_LEFT_CURVE_SMALL].right = 27;
            widgets[WIDX_LEFT_CURVE_SMALL].image = ImageId(SPR_RIDE_CONSTRUCTION_LEFT_CURVE_SMALL);
            widgets[WIDX_RIGHT_CURVE_SMALL].left = 138;
            widgets[WIDX_RIGHT_CURVE_SMALL].right = 159;
            widgets[WIDX_RIGHT_CURVE_SMALL].image = ImageId(SPR_RIDE_CONSTRUCTION_RIGHT_CURVE_SMALL);
        }
        if (IsTrackEnabled(TRACK_CURVE_SMALL))
        {
            widgets[WIDX_LEFT_CURVE_SMALL].type = WindowWidgetType::FlatBtn;
            widgets[WIDX_RIGHT_CURVE_SMALL].type = WindowWidgetType::FlatBtn;
        }
        if (IsTrackEnabled(TRACK_CURVE_VERY_SMALL))
        {
            widgets[WIDX_LEFT_CURVE_VERY_SMALL].type = WindowWidgetType::FlatBtn;
            widgets[WIDX_RIGHT_CURVE_VERY_SMALL].type = WindowWidgetType::FlatBtn;
        }

        widgets[WIDX_SLOPE_DOWN_STEEP].type = WindowWidgetType::Empty;
        widgets[WIDX_SLOPE_DOWN].type = WindowWidgetType::Empty;
        widgets[WIDX_LEVEL].type = WindowWidgetType::Empty;
        widgets[WIDX_SLOPE_UP].type = WindowWidgetType::Empty;
        widgets[WIDX_SLOPE_UP_STEEP].type = WindowWidgetType::Empty;
        widgets[WIDX_SLOPE_DOWN_STEEP].image = ImageId(SPR_RIDE_CONSTRUCTION_SLOPE_DOWN_STEEP);
        widgets[WIDX_SLOPE_DOWN_STEEP].tooltip = STR_RIDE_CONSTRUCTION_STEEP_SLOPE_DOWN_TIP;
        widgets[WIDX_SLOPE_UP_STEEP].image = ImageId(SPR_RIDE_CONSTRUCTION_SLOPE_UP_STEEP);
        widgets[WIDX_SLOPE_UP_STEEP].tooltip = STR_RIDE_CONSTRUCTION_STEEP_SLOPE_UP_TIP;
        if (GetRideTypeDescriptor(rideType).SupportsTrackPiece(TRACK_REVERSE_FREEFALL))
        {
            widgets[WIDX_LEVEL].type = WindowWidgetType::FlatBtn;
            widgets[WIDX_SLOPE_UP].type = WindowWidgetType::FlatBtn;
        }
        if (IsTrackEnabled(TRACK_SLOPE) || IsTrackEnabled(TRACK_SLOPE_STEEP_DOWN) || IsTrackEnabled(TRACK_SLOPE_STEEP_UP))
        {
            widgets[WIDX_LEVEL].type = WindowWidgetType::FlatBtn;
        }
        if (IsTrackEnabled(TRACK_SLOPE))
        {
            widgets[WIDX_SLOPE_DOWN].type = WindowWidgetType::FlatBtn;
            widgets[WIDX_SLOPE_UP].type = WindowWidgetType::FlatBtn;
        }
        if ((IsTrackEnabled(TRACK_HELIX_DOWN_BANKED_HALF) || IsTrackEnabled(TRACK_HELIX_UP_BANKED_HALF))
            && _currentTrackBankEnd != TRACK_BANK_NONE && _currentTrackSlopeEnd == TRACK_SLOPE_NONE)
        {
            if (_currentTrackCurve >= TRACK_CURVE_LEFT && _currentTrackCurve <= TRACK_CURVE_RIGHT_SMALL)
            {
                // Enable helix
                widgets[WIDX_SLOPE_DOWN_STEEP].type = WindowWidgetType::FlatBtn;
                if (rideType != RIDE_TYPE_SPLASH_BOATS && rideType != RIDE_TYPE_RIVER_RAFTS)
                    widgets[WIDX_SLOPE_UP_STEEP].type = WindowWidgetType::FlatBtn;
            }
        }

        if (IsTrackEnabled(TRACK_SLOPE_STEEP_DOWN))
        {
            widgets[WIDX_SLOPE_DOWN_STEEP].type = WindowWidgetType::FlatBtn;
        }
        if (IsTrackEnabled(TRACK_SLOPE_STEEP_UP))
        {
            widgets[WIDX_SLOPE_UP_STEEP].type = WindowWidgetType::FlatBtn;
        }

        if (currentRide->GetRideTypeDescriptor().HasFlag(RIDE_TYPE_FLAG_UP_INCLINE_REQUIRES_LIFT)
            && (_currentTrackSlopeEnd == TRACK_SLOPE_UP_25 || _currentTrackSlopeEnd == TRACK_SLOPE_UP_60)
            && !gCheatsEnableAllDrawableTrackPieces)
        {
            _currentTrackLiftHill |= CONSTRUCTION_LIFT_HILL_SELECTED;
        }

        int32_t x;
        if ((IsTrackEnabled(TRACK_LIFT_HILL) && (_currentTrackCurve & RideConstructionSpecialPieceSelected) == 0)
            || (gCheatsEnableChainLiftOnAllTrack && currentRide->GetRideTypeDescriptor().HasFlag(RIDE_TYPE_FLAG_HAS_TRACK)))
        {
            widgets[WIDX_CHAIN_LIFT].type = WindowWidgetType::FlatBtn;
            x = 9;
        }
        else
        {
            widgets[WIDX_CHAIN_LIFT].type = WindowWidgetType::Empty;
            x = 23;
        }

        for (int32_t i = WIDX_SLOPE_DOWN_STEEP; i <= WIDX_SLOPE_UP_STEEP; i++)
        {
            widgets[i].left = x;
            widgets[i].right = x + 23;
            x += 24;
        }

        widgets[WIDX_SLOPE_UP_STEEP].image = ImageId(SPR_RIDE_CONSTRUCTION_SLOPE_UP_STEEP);
        widgets[WIDX_SLOPE_UP_STEEP].tooltip = STR_RIDE_CONSTRUCTION_STEEP_SLOPE_UP_TIP;
        widgets[WIDX_SLOPE_DOWN_STEEP].image = ImageId(SPR_RIDE_CONSTRUCTION_SLOPE_DOWN_STEEP);
        widgets[WIDX_SLOPE_DOWN_STEEP].tooltip = STR_RIDE_CONSTRUCTION_STEEP_SLOPE_DOWN_TIP;
        if (IsTrackEnabled(TRACK_SLOPE_VERTICAL))
        {
            if (_previousTrackSlopeEnd == TRACK_SLOPE_UP_60 || _previousTrackSlopeEnd == TRACK_SLOPE_UP_90)
            {
                int32_t originalSlopeUpSteepLeft = widgets[WIDX_SLOPE_UP_STEEP].left;
                int32_t originalSlopeUpSteepRight = widgets[WIDX_SLOPE_UP_STEEP].right;
                for (int32_t i = WIDX_SLOPE_UP_STEEP; i > WIDX_SLOPE_DOWN_STEEP; i--)
                {
                    widgets[i].left = widgets[i - 1].left;
                    widgets[i].right = widgets[i - 1].right;
                }
                widgets[WIDX_SLOPE_DOWN_STEEP].left = originalSlopeUpSteepLeft;
                widgets[WIDX_SLOPE_DOWN_STEEP].right = originalSlopeUpSteepRight;
                widgets[WIDX_SLOPE_DOWN_STEEP].image = ImageId(SPR_RIDE_CONSTRUCTION_VERTICAL_RISE);
                widgets[WIDX_SLOPE_DOWN_STEEP].tooltip = STR_RIDE_CONSTRUCTION_VERTICAL_RISE_TIP;
            }
            else if (_previousTrackSlopeEnd == TRACK_SLOPE_DOWN_60 || _previousTrackSlopeEnd == TRACK_SLOPE_DOWN_90)
            {
                int32_t originalSlopeDownSteepLeft = widgets[WIDX_SLOPE_DOWN_STEEP].left;
                int32_t originalSlopeDownSteepRight = widgets[WIDX_SLOPE_DOWN_STEEP].right;
                for (int32_t i = WIDX_SLOPE_DOWN_STEEP; i < WIDX_SLOPE_UP_STEEP; i++)
                {
                    widgets[i].left = widgets[i + 1].left;
                    widgets[i].right = widgets[i + 1].right;
                }
                widgets[WIDX_SLOPE_UP_STEEP].left = originalSlopeDownSteepLeft;
                widgets[WIDX_SLOPE_UP_STEEP].right = originalSlopeDownSteepRight;
                widgets[WIDX_SLOPE_UP_STEEP].image = ImageId(SPR_RIDE_CONSTRUCTION_VERTICAL_DROP);
                widgets[WIDX_SLOPE_UP_STEEP].tooltip = STR_RIDE_CONSTRUCTION_VERTICAL_DROP_TIP;
            }
        }

        if ((IsTrackEnabled(TRACK_HELIX_DOWN_UNBANKED_QUARTER) || IsTrackEnabled(TRACK_HELIX_UP_UNBANKED_QUARTER))
            && _currentTrackSlopeEnd == TRACK_SLOPE_NONE && _currentTrackBankEnd == TRACK_BANK_NONE
            && (_currentTrackCurve == TRACK_CURVE_LEFT || _currentTrackCurve == TRACK_CURVE_RIGHT))
        {
            widgets[WIDX_SLOPE_DOWN_STEEP].image = ImageId(SPR_RIDE_CONSTRUCTION_HELIX_DOWN);
            widgets[WIDX_SLOPE_DOWN_STEEP].tooltip = STR_RIDE_CONSTRUCTION_HELIX_DOWN_TIP;
            widgets[WIDX_SLOPE_UP_STEEP].image = ImageId(SPR_RIDE_CONSTRUCTION_HELIX_UP);
            widgets[WIDX_SLOPE_UP_STEEP].tooltip = STR_RIDE_CONSTRUCTION_HELIX_UP_TIP;

            int32_t tmp = widgets[WIDX_SLOPE_DOWN_STEEP].left;
            widgets[WIDX_SLOPE_DOWN_STEEP].left = widgets[WIDX_SLOPE_DOWN].left;
            widgets[WIDX_SLOPE_DOWN].left = tmp;

            tmp = widgets[WIDX_SLOPE_DOWN_STEEP].right;
            widgets[WIDX_SLOPE_DOWN_STEEP].right = widgets[WIDX_SLOPE_DOWN].right;
            widgets[WIDX_SLOPE_DOWN].right = tmp;

            tmp = widgets[WIDX_SLOPE_UP_STEEP].left;
            widgets[WIDX_SLOPE_UP_STEEP].left = widgets[WIDX_SLOPE_UP].left;
            widgets[WIDX_SLOPE_UP].left = tmp;

            tmp = widgets[WIDX_SLOPE_UP_STEEP].right;
            widgets[WIDX_SLOPE_UP_STEEP].right = widgets[WIDX_SLOPE_UP].right;
            widgets[WIDX_SLOPE_UP].right = tmp;
        }

        if ((IsTrackEnabled(TRACK_HELIX_DOWN_BANKED_QUARTER) || IsTrackEnabled(TRACK_HELIX_UP_BANKED_QUARTER)
             || IsTrackEnabled(TRACK_HELIX_DOWN_BANKED_HALF) || IsTrackEnabled(TRACK_HELIX_UP_BANKED_HALF))
            && (_currentTrackCurve >= TRACK_CURVE_LEFT && _currentTrackCurve <= TRACK_CURVE_RIGHT_SMALL)
            && _currentTrackSlopeEnd == TRACK_SLOPE_NONE && _currentTrackBankEnd != TRACK_BANK_NONE)
        {
            widgets[WIDX_SLOPE_DOWN_STEEP].image = ImageId(SPR_RIDE_CONSTRUCTION_HELIX_DOWN);
            widgets[WIDX_SLOPE_DOWN_STEEP].tooltip = STR_RIDE_CONSTRUCTION_HELIX_DOWN_TIP;
            widgets[WIDX_SLOPE_UP_STEEP].image = ImageId(SPR_RIDE_CONSTRUCTION_HELIX_UP);
            widgets[WIDX_SLOPE_UP_STEEP].tooltip = STR_RIDE_CONSTRUCTION_HELIX_UP_TIP;

            int32_t tmp = widgets[WIDX_SLOPE_DOWN_STEEP].left;
            widgets[WIDX_SLOPE_DOWN_STEEP].left = widgets[WIDX_SLOPE_DOWN].left;
            widgets[WIDX_SLOPE_DOWN].left = tmp;

            tmp = widgets[WIDX_SLOPE_DOWN_STEEP].right;
            widgets[WIDX_SLOPE_DOWN_STEEP].right = widgets[WIDX_SLOPE_DOWN].right;
            widgets[WIDX_SLOPE_DOWN].right = tmp;

            tmp = widgets[WIDX_SLOPE_UP_STEEP].left;
            widgets[WIDX_SLOPE_UP_STEEP].left = widgets[WIDX_SLOPE_UP].left;
            widgets[WIDX_SLOPE_UP].left = tmp;

            tmp = widgets[WIDX_SLOPE_UP_STEEP].right;
            widgets[WIDX_SLOPE_UP_STEEP].right = widgets[WIDX_SLOPE_UP].right;
            widgets[WIDX_SLOPE_UP].right = tmp;
        }

        widgets[WIDX_BANKING_GROUPBOX].image = ImageId(STR_RIDE_CONSTRUCTION_ROLL_BANKING);
        widgets[WIDX_BANK_LEFT].image = ImageId(SPR_RIDE_CONSTRUCTION_LEFT_BANK);
        widgets[WIDX_BANK_LEFT].tooltip = STR_RIDE_CONSTRUCTION_ROLL_FOR_LEFT_CURVE_TIP;
        widgets[WIDX_BANK_LEFT].left = 47;
        widgets[WIDX_BANK_LEFT].right = 70;
        widgets[WIDX_BANK_LEFT].top = 132;
        widgets[WIDX_BANK_LEFT].bottom = 155;
        widgets[WIDX_BANK_STRAIGHT].image = ImageId(SPR_RIDE_CONSTRUCTION_NO_BANK);
        widgets[WIDX_BANK_STRAIGHT].tooltip = STR_RIDE_CONSTRUCTION_NO_ROLL_TIP;
        widgets[WIDX_BANK_STRAIGHT].left = 71;
        widgets[WIDX_BANK_STRAIGHT].right = 94;
        widgets[WIDX_BANK_STRAIGHT].top = 132;
        widgets[WIDX_BANK_STRAIGHT].bottom = 155;
        widgets[WIDX_BANK_RIGHT].image = ImageId(SPR_RIDE_CONSTRUCTION_RIGHT_BANK);
        widgets[WIDX_BANK_RIGHT].tooltip = STR_RIDE_CONSTRUCTION_ROLL_FOR_RIGHT_CURVE_TIP;
        widgets[WIDX_BANK_RIGHT].left = 95;
        widgets[WIDX_BANK_RIGHT].right = 118;
        widgets[WIDX_BANK_RIGHT].top = 132;
        widgets[WIDX_BANK_RIGHT].bottom = 155;
        widgets[WIDX_BANK_LEFT].type = WindowWidgetType::Empty;
        widgets[WIDX_BANK_STRAIGHT].type = WindowWidgetType::Empty;
        widgets[WIDX_BANK_RIGHT].type = WindowWidgetType::Empty;
        widgets[WIDX_U_TRACK].type = WindowWidgetType::Empty;
        widgets[WIDX_O_TRACK].type = WindowWidgetType::Empty;

        bool trackHasSpeedSetting = TrackTypeHasSpeedSetting(_selectedTrackType)
            || TrackTypeHasSpeedSetting(_currentTrackCurve & ~RideConstructionSpecialPieceSelected);
        bool boosterTrackSelected = _selectedTrackType == TrackElemType::Booster
            || _currentTrackCurve == (RideConstructionSpecialPieceSelected | TrackElemType::Booster);

        // Only necessary because TD6 writes speed and seat rotation to the same bits. Remove for new track design format.
        bool trackHasSpeedAndSeatRotation = _selectedTrackType == TrackElemType::BlockBrakes
            || _currentTrackCurve == (RideConstructionSpecialPieceSelected | TrackElemType::BlockBrakes)
            || _selectedTrackType > TrackElemType::HighestAlias
            || _currentTrackCurve > (RideConstructionSpecialPieceSelected | TrackElemType::HighestAlias);

        const auto& rtd = GetRideTypeDescriptor(rideType);
        bool rideHasSeatRotation = rtd.HasFlag(RIDE_TYPE_FLAG_HAS_SEAT_ROTATION);

        if (!trackHasSpeedSetting)
        {
            if (IsTrackEnabled(TRACK_FLAT_ROLL_BANKING))
            {
                widgets[WIDX_BANK_LEFT].type = WindowWidgetType::FlatBtn;
                widgets[WIDX_BANK_STRAIGHT].type = WindowWidgetType::FlatBtn;
                widgets[WIDX_BANK_RIGHT].type = WindowWidgetType::FlatBtn;
            }
            if (GetRideTypeDescriptor(rideType).HasFlag(RIDE_TYPE_FLAG_TRACK_ELEMENTS_HAVE_TWO_VARIETIES))
            {
                if (rideType == RIDE_TYPE_WATER_COASTER)
                {
                    widgets[WIDX_U_TRACK].image = ImageId(SPR_RIDE_CONSTRUCTION_RC_TRACK);
                    widgets[WIDX_O_TRACK].image = ImageId(SPR_RIDE_CONSTRUCTION_WATER_CHANNEL);
                    widgets[WIDX_U_TRACK].tooltip = STR_RIDE_CONSTRUCTION_STANDARD_RC_TRACK_TIP;
                    widgets[WIDX_O_TRACK].tooltip = STR_RIDE_CONSTRUCTION_WATER_CHANNEL_TIP;
                    if ((_currentTrackCurve < TRACK_CURVE_LEFT_SMALL
                         || _currentTrackCurve == (RideConstructionSpecialPieceSelected | TrackElemType::SBendLeft)
                         || _currentTrackCurve == (RideConstructionSpecialPieceSelected | TrackElemType::SBendRight))
                        && _currentTrackSlopeEnd == TRACK_SLOPE_NONE && _currentTrackBankEnd == TRACK_BANK_NONE)
                    {
                        widgets[WIDX_BANKING_GROUPBOX].text = STR_RIDE_CONSTRUCTION_TRACK_STYLE;
                        widgets[WIDX_U_TRACK].type = WindowWidgetType::FlatBtn;
                        widgets[WIDX_O_TRACK].type = WindowWidgetType::FlatBtn;
                    }
                }
                else
                {
                    widgets[WIDX_U_TRACK].image = ImageId(SPR_RIDE_CONSTRUCTION_U_SHAPED_TRACK);
                    widgets[WIDX_O_TRACK].image = ImageId(SPR_RIDE_CONSTRUCTION_O_SHAPED_TRACK);
                    widgets[WIDX_U_TRACK].tooltip = STR_RIDE_CONSTRUCTION_U_SHAPED_OPEN_TRACK_TIP;
                    widgets[WIDX_O_TRACK].tooltip = STR_RIDE_CONSTRUCTION_O_SHAPED_ENCLOSED_TRACK_TIP;
                    widgets[WIDX_BANKING_GROUPBOX].text = STR_RIDE_CONSTRUCTION_TRACK_STYLE;
                    widgets[WIDX_U_TRACK].type = WindowWidgetType::FlatBtn;
                    widgets[WIDX_O_TRACK].type = WindowWidgetType::FlatBtn;
                }
            }
        }
        else
        {
            if (!boosterTrackSelected)
            {
                widgets[WIDX_SPEED_GROUPBOX].text = STR_RIDE_CONSTRUCTION_BRAKE_SPEED;
                widgets[WIDX_SPEED_SETTING_SPINNER].tooltip = STR_RIDE_CONSTRUCTION_BRAKE_SPEED_LIMIT_TIP;
                widgets[WIDX_SPEED_SETTING_SPINNER_UP].tooltip = STR_RIDE_CONSTRUCTION_BRAKE_SPEED_LIMIT_TIP;
                widgets[WIDX_SPEED_SETTING_SPINNER_DOWN].tooltip = STR_RIDE_CONSTRUCTION_BRAKE_SPEED_LIMIT_TIP;
            }
            else
            {
                widgets[WIDX_SPEED_GROUPBOX].text = STR_RIDE_CONSTRUCTION_BOOSTER_SPEED;
                widgets[WIDX_SPEED_SETTING_SPINNER].tooltip = STR_RIDE_CONSTRUCTION_BOOSTER_SPEED_LIMIT_TIP;
                widgets[WIDX_SPEED_SETTING_SPINNER_UP].tooltip = STR_RIDE_CONSTRUCTION_BOOSTER_SPEED_LIMIT_TIP;
                widgets[WIDX_SPEED_SETTING_SPINNER_DOWN].tooltip = STR_RIDE_CONSTRUCTION_BOOSTER_SPEED_LIMIT_TIP;
            }

            _currentlyShowingBrakeOrBoosterSpeed = true;
            widgets[WIDX_SPEED_SETTING_SPINNER].text = STR_RIDE_CONSTRUCTION_BRAKE_SPEED_VELOCITY;

            widgets[WIDX_SPEED_SETTING_SPINNER].type = WindowWidgetType::Spinner;
            widgets[WIDX_SPEED_SETTING_SPINNER_UP].type = WindowWidgetType::Button;
            widgets[WIDX_SPEED_SETTING_SPINNER_UP].text = STR_NUMERIC_UP;
            widgets[WIDX_SPEED_SETTING_SPINNER_DOWN].type = WindowWidgetType::Button;
            widgets[WIDX_SPEED_SETTING_SPINNER_DOWN].text = STR_NUMERIC_DOWN;

            ResizeSpinner(WIDX_SPEED_SETTING_SPINNER, { 12, 138 }, { 85, SPINNER_HEIGHT });

            hold_down_widgets |= (1uLL << WIDX_SPEED_SETTING_SPINNER_UP) | (1uLL << WIDX_SPEED_SETTING_SPINNER_DOWN);
        }

        static constexpr int16_t bankingGroupboxRightNoSeatRotation = 162;
        static constexpr int16_t bankingGroupboxRightWithSeatRotation = 92;

        widgets[WIDX_BANKING_GROUPBOX].right = bankingGroupboxRightNoSeatRotation;
        widgets[WIDX_SEAT_ROTATION_GROUPBOX].type = WindowWidgetType::Empty;
        widgets[WIDX_SEAT_ROTATION_ANGLE_SPINNER].type = WindowWidgetType::Empty;
        widgets[WIDX_SEAT_ROTATION_ANGLE_SPINNER_UP].type = WindowWidgetType::Empty;
        widgets[WIDX_SEAT_ROTATION_ANGLE_SPINNER_DOWN].type = WindowWidgetType::Empty;

        // Simplify this condition to "rideHasSeatRotation" for new track design format
        if ((rideHasSeatRotation && !trackHasSpeedSetting)
            || (rideHasSeatRotation && trackHasSpeedSetting && trackHasSpeedAndSeatRotation))
        {
            widgets[WIDX_SEAT_ROTATION_GROUPBOX].type = WindowWidgetType::Groupbox;
            widgets[WIDX_SEAT_ROTATION_ANGLE_SPINNER].type = WindowWidgetType::Spinner;
            widgets[WIDX_SEAT_ROTATION_ANGLE_SPINNER_UP].type = WindowWidgetType::Button;
            widgets[WIDX_SEAT_ROTATION_ANGLE_SPINNER_DOWN].type = WindowWidgetType::Button;
            widgets[WIDX_BANKING_GROUPBOX].right = bankingGroupboxRightWithSeatRotation;

            // squishes the track speed spinner slightly to make room for the seat rotation widgets
            if (trackHasSpeedSetting)
            {
                widgets[WIDX_SPEED_SETTING_SPINNER].left -= 4;
                widgets[WIDX_SPEED_SETTING_SPINNER].right -= 8;
                widgets[WIDX_SPEED_SETTING_SPINNER_UP].right -= 8;
                widgets[WIDX_SPEED_SETTING_SPINNER_DOWN].right -= 8;
                widgets[WIDX_SPEED_SETTING_SPINNER_UP].left -= 8;
                widgets[WIDX_SPEED_SETTING_SPINNER_DOWN].left -= 8;
            }
            // moves banking buttons to the left to make room for the seat rotation widgets
            else if (IsTrackEnabled(TRACK_FLAT_ROLL_BANKING))
            {
                for (int32_t i = WIDX_BANK_LEFT; i <= WIDX_BANK_RIGHT; i++)
                {
                    widgets[i].left -= 36;
                    widgets[i].right -= 36;
                }
            }
        }

        uint64_t pressedWidgets = pressed_widgets
            & ((1uLL << WIDX_BACKGROUND) | (1uLL << WIDX_TITLE) | (1uLL << WIDX_CLOSE) | (1uLL << WIDX_DIRECTION_GROUPBOX)
               | (1uLL << WIDX_SLOPE_GROUPBOX) | (1uLL << WIDX_BANKING_GROUPBOX) | (1uLL << WIDX_CONSTRUCT)
               | (1uLL << WIDX_DEMOLISH) | (1uLL << WIDX_PREVIOUS_SECTION) | (1uLL << WIDX_NEXT_SECTION)
               | (1uLL << WIDX_ENTRANCE_EXIT_GROUPBOX) | (1uLL << WIDX_ENTRANCE) | (1uLL << WIDX_EXIT));

        widgets[WIDX_CONSTRUCT].type = WindowWidgetType::Empty;
        widgets[WIDX_DEMOLISH].type = WindowWidgetType::FlatBtn;
        widgets[WIDX_ROTATE].type = WindowWidgetType::Empty;
        if (GetRideTypeDescriptor(rideType).HasFlag(RIDE_TYPE_FLAG_CANNOT_HAVE_GAPS))
        {
            widgets[WIDX_PREVIOUS_SECTION].type = WindowWidgetType::Empty;
            widgets[WIDX_NEXT_SECTION].type = WindowWidgetType::Empty;
        }
        else
        {
            widgets[WIDX_PREVIOUS_SECTION].type = WindowWidgetType::FlatBtn;
            widgets[WIDX_NEXT_SECTION].type = WindowWidgetType::FlatBtn;
        }

        switch (_rideConstructionState)
        {
            case RideConstructionState::Front:
                widgets[WIDX_CONSTRUCT].type = WindowWidgetType::ImgBtn;
                widgets[WIDX_NEXT_SECTION].type = WindowWidgetType::Empty;
                break;
            case RideConstructionState::Back:
                widgets[WIDX_CONSTRUCT].type = WindowWidgetType::ImgBtn;
                widgets[WIDX_PREVIOUS_SECTION].type = WindowWidgetType::Empty;
                break;
            case RideConstructionState::Place:
                widgets[WIDX_CONSTRUCT].type = WindowWidgetType::ImgBtn;
                widgets[WIDX_DEMOLISH].type = WindowWidgetType::Empty;
                widgets[WIDX_NEXT_SECTION].type = WindowWidgetType::Empty;
                widgets[WIDX_PREVIOUS_SECTION].type = WindowWidgetType::Empty;
                widgets[WIDX_ROTATE].type = WindowWidgetType::FlatBtn;
                break;
            case RideConstructionState::EntranceExit:
                widgets[WIDX_DEMOLISH].type = WindowWidgetType::Empty;
                widgets[WIDX_NEXT_SECTION].type = WindowWidgetType::Empty;
                widgets[WIDX_PREVIOUS_SECTION].type = WindowWidgetType::Empty;
                break;
            default:
                pressed_widgets = pressedWidgets;
                Invalidate();
                return;
        }

        WidgetIndex widgetIndex;
        switch (_currentTrackCurve)
        {
            case TRACK_CURVE_NONE:
                widgetIndex = WIDX_STRAIGHT;
                break;
            case TRACK_CURVE_LEFT:
                widgetIndex = WIDX_LEFT_CURVE;
                break;
            case TRACK_CURVE_RIGHT:
                widgetIndex = WIDX_RIGHT_CURVE;
                break;
            case TRACK_CURVE_LEFT_SMALL:
                widgetIndex = WIDX_LEFT_CURVE_SMALL;
                break;
            case TRACK_CURVE_RIGHT_SMALL:
                widgetIndex = WIDX_RIGHT_CURVE_SMALL;
                break;
            case TRACK_CURVE_LEFT_VERY_SMALL:
                widgetIndex = WIDX_LEFT_CURVE_VERY_SMALL;
                break;
            case TRACK_CURVE_RIGHT_VERY_SMALL:
                widgetIndex = WIDX_RIGHT_CURVE_VERY_SMALL;
                break;
            case TRACK_CURVE_LEFT_LARGE:
                widgetIndex = WIDX_LEFT_CURVE_LARGE;
                break;
            case TRACK_CURVE_RIGHT_LARGE:
                widgetIndex = WIDX_RIGHT_CURVE_LARGE;
                break;
            default:
                widgetIndex = WIDX_SPECIAL_TRACK_DROPDOWN;
                break;
        }
        pressedWidgets |= (1uLL << widgetIndex);

        switch (_currentTrackSlopeEnd)
        {
            case TRACK_SLOPE_DOWN_60:
            case TRACK_SLOPE_UP_90:
                widgetIndex = WIDX_SLOPE_DOWN_STEEP;
                break;
            case TRACK_SLOPE_DOWN_25:
                widgetIndex = WIDX_SLOPE_DOWN;
                break;
            case TRACK_SLOPE_UP_25:
                widgetIndex = WIDX_SLOPE_UP;
                break;
            case TRACK_SLOPE_UP_60:
            case TRACK_SLOPE_DOWN_90:
                widgetIndex = WIDX_SLOPE_UP_STEEP;
                break;
            default:
                widgetIndex = WIDX_LEVEL;
                break;
        }
        pressedWidgets |= (1uLL << widgetIndex);

        if (!_currentlyShowingBrakeOrBoosterSpeed)
        {
            if (GetRideTypeDescriptor(rideType).HasFlag(RIDE_TYPE_FLAG_TRACK_ELEMENTS_HAVE_TWO_VARIETIES))
            {
                if (_currentTrackAlternative & RIDE_TYPE_ALTERNATIVE_TRACK_PIECES)
                {
                    pressed_widgets |= (1uLL << WIDX_O_TRACK);
                }
                else
                {
                    pressed_widgets |= (1uLL << WIDX_U_TRACK);
                }
            }
            switch (_currentTrackBankEnd)
            {
                case TRACK_BANK_LEFT:
                    widgetIndex = WIDX_BANK_LEFT;
                    break;
                case TRACK_BANK_NONE:
                    widgetIndex = WIDX_BANK_STRAIGHT;
                    break;
                default:
                    widgetIndex = WIDX_BANK_RIGHT;
                    break;
            }
            pressedWidgets |= (1uLL << widgetIndex);
        }

        if (_currentTrackLiftHill & CONSTRUCTION_LIFT_HILL_SELECTED)
            pressedWidgets |= (1uLL << WIDX_CHAIN_LIFT);

        pressed_widgets = pressedWidgets;
        Invalidate();
    }

    void UpdatePossibleRideConfigurations()
    {
        auto currentRide = GetRide(_currentRideIndex);
        if (currentRide == nullptr)
        {
            return;
        }
        _specialElementDropdownState = BuildSpecialElementsList(
            *currentRide, _currentTrackPieceDirection, _previousTrackSlopeEnd, _previousTrackBankEnd, _rideConstructionState);
        _currentlyShowingBrakeOrBoosterSpeed = false;
    }

    void UpdateMapSelection()
    {
        int32_t trackType, trackDirection;
        CoordsXYZ trackPos{};

        MapInvalidateMapSelectionTiles();
        gMapSelectFlags |= MAP_SELECT_FLAG_ENABLE_CONSTRUCT;
        gMapSelectFlags |= MAP_SELECT_FLAG_GREEN;

        switch (_rideConstructionState)
        {
            case RideConstructionState::State0:
                trackDirection = _currentTrackPieceDirection;
                trackType = 0;
                trackPos = _currentTrackBegin;
                break;
            case RideConstructionState::Selected:
                trackDirection = _currentTrackPieceDirection;
                trackType = _currentTrackPieceType;
                trackPos = _currentTrackBegin;
                break;
            case RideConstructionState::EntranceExit:
                gMapSelectionTiles.clear();
                return;
            default:
                if (WindowRideConstructionUpdateState(&trackType, &trackDirection, nullptr, nullptr, &trackPos, nullptr))
                {
                    trackDirection = _currentTrackPieceDirection;
                    trackType = 0;
                    trackPos = _currentTrackBegin;
                }
                break;
        }

        if (GetRide(_currentRideIndex))
        {
            SelectMapTiles(trackType, trackDirection, trackPos);
            MapInvalidateMapSelectionTiles();
        }
    }

    void SelectMapTiles(int32_t trackType, int32_t trackDirection, const CoordsXY& tileCoords)
    {
        // If the scenery tool is active, we do not display our tiles as it
        // will conflict with larger scenery objects selecting tiles
        if (SceneryToolIsActive())
        {
            return;
        }

        const PreviewTrack* trackBlock;

        const auto& ted = GetTrackElementDescriptor(trackType);
        trackBlock = ted.Block;
        trackDirection &= 3;
        gMapSelectionTiles.clear();
        while (trackBlock->index != 255)
        {
            CoordsXY offsets = { trackBlock->x, trackBlock->y };
            CoordsXY currentTileCoords = tileCoords + offsets.Rotate(trackDirection);

            gMapSelectionTiles.push_back(currentTileCoords);
            trackBlock++;
        }
    }

private:
    void Construct()
    {
        RideId rideIndex;
        int32_t trackType, trackDirection, liftHillAndAlternativeState, properties;
        CoordsXYZ trackPos{};

        _currentTrackPrice = MONEY64_UNDEFINED;
        _trackPlaceCost = MONEY64_UNDEFINED;
        _trackPlaceErrorMessage = STR_NONE;
        RideConstructionInvalidateCurrentTrack();
        if (WindowRideConstructionUpdateState(
                &trackType, &trackDirection, &rideIndex, &liftHillAndAlternativeState, &trackPos, &properties))
        {
            WindowRideConstructionUpdateActiveElements();
            return;
        }

        auto currentRide = GetRide(_currentRideIndex);
        if (currentRide == nullptr)
        {
            return;
        }

        auto trackPlaceAction = TrackPlaceAction(
            rideIndex, trackType, currentRide->type, { trackPos, static_cast<uint8_t>(trackDirection) }, (properties)&0xFF,
            (properties >> 8) & 0x0F, (properties >> 12) & 0x0F, liftHillAndAlternativeState, false);
        if (_rideConstructionState == RideConstructionState::Back)
        {
            trackPlaceAction.SetCallback(RideConstructPlacedBackwardGameActionCallback);
        }
        else if (_rideConstructionState == RideConstructionState::Front)
        {
            trackPlaceAction.SetCallback(RideConstructPlacedForwardGameActionCallback);
        }
        auto res = GameActions::Execute(&trackPlaceAction);
        // Used by some functions
        if (res.Error != GameActions::Status::Ok)
        {
            _trackPlaceCost = MONEY64_UNDEFINED;
            _trackPlaceErrorMessage = std::get<StringId>(res.ErrorMessage);
        }
        else
        {
            _trackPlaceCost = res.Cost;
            _trackPlaceErrorMessage = STR_NONE;
        }

        if (res.Error != GameActions::Status::Ok)
        {
            return;
        }
        OpenRCT2::Audio::Play3D(OpenRCT2::Audio::SoundId::PlaceItem, trackPos);

        if (NetworkGetMode() != NETWORK_MODE_NONE)
        {
            _currentTrackSelectionFlags |= TRACK_SELECTION_FLAG_TRACK_PLACE_ACTION_QUEUED;
        }

        const auto resultData = res.GetData<TrackPlaceActionResult>();
        if (resultData.GroundFlags & ELEMENT_IS_UNDERGROUND)
        {
            ViewportSetVisibility(ViewportVisibility::UndergroundViewOn);
        }

        const bool helixSelected = (_currentTrackCurve & RideConstructionSpecialPieceSelected)
            && TrackTypeIsHelix(_currentTrackCurve & ~RideConstructionSpecialPieceSelected);
        if (helixSelected || (_currentTrackSlopeEnd != TRACK_SLOPE_NONE))
        {
            ViewportSetVisibility(ViewportVisibility::TrackHeights);
        }
    }

    void MouseUpDemolish()
    {
        int32_t direction;
        TileElement* tileElement;
        CoordsXYE inputElement, outputElement;
        TrackBeginEnd trackBeginEnd;

        _currentTrackPrice = MONEY64_UNDEFINED;
        RideConstructionInvalidateCurrentTrack();

        // Select the track element that is to be deleted
        _rideConstructionState2 = RideConstructionState::Selected;
        if (_rideConstructionState == RideConstructionState::Front)
        {
            if (!RideSelectBackwardsFromFront())
            {
                WindowRideConstructionUpdateActiveElements();
                return;
            }
            _rideConstructionState2 = RideConstructionState::Front;
        }
        else if (_rideConstructionState == RideConstructionState::Back)
        {
            if (!RideSelectForwardsFromBack())
            {
                WindowRideConstructionUpdateActiveElements();
                return;
            }
            _rideConstructionState2 = RideConstructionState::Back;
        }

        // Invalidate the selected track element or make sure it's at origin???
        direction = _currentTrackPieceDirection;
        // The direction is reset by ride_initialise_construction_window(), but we need it to remove flat rides properly.
        Direction currentDirection = _currentTrackPieceDirection;
        track_type_t type = _currentTrackPieceType;
        auto newCoords = GetTrackElementOriginAndApplyChanges(
            { _currentTrackBegin, static_cast<Direction>(direction & 3) }, type, 0, &tileElement, 0);
        if (!newCoords.has_value())
        {
            WindowRideConstructionUpdateActiveElements();
            return;
        }

        // Get the previous track element to go to after the selected track element is deleted
        inputElement.x = newCoords->x;
        inputElement.y = newCoords->y;
        inputElement.element = tileElement;
        if (TrackBlockGetPrevious({ *newCoords, tileElement }, &trackBeginEnd))
        {
            *newCoords = { trackBeginEnd.begin_x, trackBeginEnd.begin_y, trackBeginEnd.begin_z };
            direction = trackBeginEnd.begin_direction;
            type = trackBeginEnd.begin_element->AsTrack()->GetTrackType();
            _gotoStartPlacementMode = false;
        }
        else if (TrackBlockGetNext(&inputElement, &outputElement, &newCoords->z, &direction))
        {
            newCoords->x = outputElement.x;
            newCoords->y = outputElement.y;
            direction = outputElement.element->GetDirection();
            type = outputElement.element->AsTrack()->GetTrackType();
            _gotoStartPlacementMode = false;
        }
        else
        {
            direction = _currentTrackPieceDirection;
            type = _currentTrackPieceType;
            newCoords = GetTrackElementOriginAndApplyChanges(
                { _currentTrackBegin, static_cast<Direction>(direction & 3) }, type, 0, &tileElement, 0);

            if (!newCoords.has_value())
            {
                WindowRideConstructionUpdateActiveElements();
                return;
            }

            const auto& ted = GetTrackElementDescriptor(tileElement->AsTrack()->GetTrackType());
            const PreviewTrack* trackBlock = ted.Block;
            newCoords->z = (tileElement->GetBaseZ()) - trackBlock->z;
            _gotoStartPlacementMode = true;

            // When flat rides are deleted, the window should be reset so the currentRide can be placed again.
            auto currentRide = GetRide(_currentRideIndex);
            const auto& rtd = currentRide->GetRideTypeDescriptor();
            if (rtd.HasFlag(RIDE_TYPE_FLAG_FLAT_RIDE) && !rtd.HasFlag(RIDE_TYPE_FLAG_IS_SHOP_OR_FACILITY))
            {
                RideInitialiseConstructionWindow(*currentRide);
            }
        }

        auto trackRemoveAction = TrackRemoveAction(
            _currentTrackPieceType, 0, { _currentTrackBegin.x, _currentTrackBegin.y, _currentTrackBegin.z, currentDirection });

        trackRemoveAction.SetCallback([=](const GameAction* ga, const GameActions::Result* result) {
            if (result->Error != GameActions::Status::Ok)
            {
                WindowRideConstructionUpdateActiveElements();
            }
            else
            {
                auto currentRide = GetRide(_currentRideIndex);
                if (currentRide != nullptr)
                {
                    WindowRideConstructionMouseUpDemolishNextPiece({ *newCoords, static_cast<Direction>(direction) }, type);
                }
            }
        });

        GameActions::Execute(&trackRemoveAction);
    }

    void Rotate()
    {
        _autoRotatingShop = false;
        _currentTrackPieceDirection = (_currentTrackPieceDirection + 1) & 3;
        RideConstructionInvalidateCurrentTrack();
        _currentTrackPrice = MONEY64_UNDEFINED;
        WindowRideConstructionUpdateActiveElements();
    }

    void EntranceClick()
    {
        if (ToolSet(*this, WIDX_ENTRANCE, Tool::Crosshair))
        {
            auto currentRide = GetRide(_currentRideIndex);
            if (currentRide != nullptr && !RideTryGetOriginElement(*currentRide, nullptr))
            {
                RideInitialiseConstructionWindow(*currentRide);
            }
        }
        else
        {
            gRideEntranceExitPlaceType = ENTRANCE_TYPE_RIDE_ENTRANCE;
            gRideEntranceExitPlaceRideIndex = _currentRideIndex;
            gRideEntranceExitPlaceStationIndex = StationIndex::FromUnderlying(0);
            InputSetFlag(INPUT_FLAG_6, true);
            RideConstructionInvalidateCurrentTrack();
            if (_rideConstructionState != RideConstructionState::EntranceExit)
            {
                gRideEntranceExitPlacePreviousRideConstructionState = _rideConstructionState;
                _rideConstructionState = RideConstructionState::EntranceExit;
            }
            WindowRideConstructionUpdateActiveElements();
        }
    }

    void ExitClick()
    {
        if (ToolSet(*this, WIDX_EXIT, Tool::Crosshair))
        {
            auto currentRide = GetRide(_currentRideIndex);
            if (!RideTryGetOriginElement(*currentRide, nullptr))
            {
                RideInitialiseConstructionWindow(*currentRide);
            }
        }
        else
        {
            gRideEntranceExitPlaceType = ENTRANCE_TYPE_RIDE_EXIT;
            gRideEntranceExitPlaceRideIndex = _currentRideIndex;
            gRideEntranceExitPlaceStationIndex = StationIndex::FromUnderlying(0);
            InputSetFlag(INPUT_FLAG_6, true);
            RideConstructionInvalidateCurrentTrack();
            if (_rideConstructionState != RideConstructionState::EntranceExit)
            {
                gRideEntranceExitPlacePreviousRideConstructionState = _rideConstructionState;
                _rideConstructionState = RideConstructionState::EntranceExit;
            }
            WindowRideConstructionUpdateActiveElements();
        }
    }

    void UpdateLiftHillSelected(int32_t slope)
    {
        _currentTrackSlopeEnd = slope;
        _currentTrackPrice = MONEY64_UNDEFINED;
        if (_rideConstructionState == RideConstructionState::Front && !gCheatsEnableChainLiftOnAllTrack)
        {
            switch (slope)
            {
                case TRACK_SLOPE_NONE:
                case TRACK_SLOPE_UP_25:
                case TRACK_SLOPE_UP_60:
                    break;
                default:
                    _currentTrackLiftHill &= ~CONSTRUCTION_LIFT_HILL_SELECTED;
                    break;
            }
        }
        WindowRideConstructionUpdateActiveElements();
    }

    void SetBrakeSpeed(int32_t brakesSpeed)
    {
        TileElement* tileElement;

        if (GetTrackElementOriginAndApplyChanges(
                { _currentTrackBegin, static_cast<Direction>(_currentTrackPieceDirection & 3) }, _currentTrackPieceType, 0,
                &tileElement, 0)
            != std::nullopt)
        {
            auto trackSetBrakeSpeed = TrackSetBrakeSpeedAction(
                _currentTrackBegin, tileElement->AsTrack()->GetTrackType(), brakesSpeed);
            trackSetBrakeSpeed.SetCallback(
                [](const GameAction* ga, const GameActions::Result* result) { WindowRideConstructionUpdateActiveElements(); });
            GameActions::Execute(&trackSetBrakeSpeed);
            return;
        }
        WindowRideConstructionUpdateActiveElements();
    }

    void ShowSpecialTrackDropdown(Widget* widget)
    {
        int32_t defaultIndex = -1;
        for (size_t i = 0; i < _specialElementDropdownState.Elements.size(); i++)
        {
            track_type_t trackPiece = _specialElementDropdownState.Elements[i].TrackType;

            const auto& ted = GetTrackElementDescriptor(trackPiece);
            StringId trackPieceStringId = ted.Description;
            if (trackPieceStringId == STR_RAPIDS)
            {
                auto currentRide = GetRide(_currentRideIndex);
                if (currentRide != nullptr)
                {
                    const auto& rtd = currentRide->GetRideTypeDescriptor();
                    if (rtd.Category != RIDE_CATEGORY_WATER)
                        trackPieceStringId = STR_LOG_BUMPS;
                }
            }
            gDropdownItems[i].Format = trackPieceStringId;
            if ((trackPiece | RideConstructionSpecialPieceSelected) == _currentTrackCurve)
            {
                defaultIndex = static_cast<int32_t>(i);
            }
        }

        WindowDropdownShowTextCustomWidth(
            { windowPos.x + widget->left, windowPos.y + widget->top }, widget->height() + 1, colours[1], 0, 0,
            _specialElementDropdownState.Elements.size(), widget->width());

        for (size_t i = 0; i < _specialElementDropdownState.Elements.size(); i++)
        {
            Dropdown::SetDisabled(static_cast<int32_t>(i), _specialElementDropdownState.Elements[i].Disabled);
        }
        gDropdownDefaultIndex = defaultIndex;
    }

    void RideSelectedTrackSetSeatRotation(int32_t seatRotation)
    {
        GetTrackElementOriginAndApplyChanges(
            { _currentTrackBegin, static_cast<Direction>(_currentTrackPieceDirection & 3) }, _currentTrackPieceType,
            seatRotation, nullptr, TRACK_ELEMENT_SET_SEAT_ROTATION);
        WindowRideConstructionUpdateActiveElements();
    }

    void ToolDownEntranceExit(const ScreenCoordsXY& screenCoords)
    {
        RideConstructionInvalidateCurrentTrack();
        MapInvalidateSelectionRect();
        gMapSelectFlags &= ~MAP_SELECT_FLAG_ENABLE;
        gMapSelectFlags &= ~MAP_SELECT_FLAG_ENABLE_ARROW;

        CoordsXYZD entranceOrExitCoords = RideGetEntranceOrExitPositionFromScreenPosition(screenCoords);
        if (gRideEntranceExitPlaceDirection == INVALID_DIRECTION)
            return;

        auto rideEntranceExitPlaceAction = RideEntranceExitPlaceAction(
            entranceOrExitCoords, DirectionReverse(gRideEntranceExitPlaceDirection), gRideEntranceExitPlaceRideIndex,
            gRideEntranceExitPlaceStationIndex, gRideEntranceExitPlaceType == ENTRANCE_TYPE_RIDE_EXIT);

        rideEntranceExitPlaceAction.SetCallback([=](const GameAction* ga, const GameActions::Result* result) {
            if (result->Error != GameActions::Status::Ok)
                return;

            OpenRCT2::Audio::Play3D(OpenRCT2::Audio::SoundId::PlaceItem, result->Position);

            auto currentRide = GetRide(gRideEntranceExitPlaceRideIndex);
            if (currentRide != nullptr && RideAreAllPossibleEntrancesAndExitsBuilt(*currentRide).Successful)
            {
                ToolCancel();
                if (!currentRide->GetRideTypeDescriptor().HasFlag(RIDE_TYPE_FLAG_HAS_TRACK))
                {
                    WindowCloseByClass(WindowClass::RideConstruction);
                }
            }
            else
            {
                gRideEntranceExitPlaceType = gRideEntranceExitPlaceType ^ 1;
                WindowInvalidateByClass(WindowClass::RideConstruction);
                gCurrentToolWidget.widget_index = (gRideEntranceExitPlaceType == ENTRANCE_TYPE_RIDE_ENTRANCE)
                    ? WC_RIDE_CONSTRUCTION__WIDX_ENTRANCE
                    : WC_RIDE_CONSTRUCTION__WIDX_EXIT;
            }
        });
        auto res = GameActions::Execute(&rideEntranceExitPlaceAction);
    }

    void DrawTrackPiece(
        DrawPixelInfo& dpi, RideId rideIndex, int32_t trackType, int32_t trackDirection, int32_t liftHillAndInvertedState,
        int32_t widgetWidth, int32_t widgetHeight)
    {
        auto currentRide = GetRide(rideIndex);
        if (currentRide == nullptr)
        {
            return;
        }

        const auto& ted = GetTrackElementDescriptor(trackType);
        const auto* trackBlock = ted.Block;
        while ((trackBlock + 1)->index != 0xFF)
            trackBlock++;

        CoordsXYZ mapCoords{ trackBlock->x, trackBlock->y, trackBlock->z };
        if (trackBlock->flags & RCT_PREVIEW_TRACK_FLAG_1)
        {
            mapCoords.x = 0;
            mapCoords.y = 0;
        }

        auto rotatedMapCoords = mapCoords.Rotate(trackDirection);
        // this is actually case 0, but the other cases all jump to it
        mapCoords.x = 4112 + (rotatedMapCoords.x / 2);
        mapCoords.y = 4112 + (rotatedMapCoords.y / 2);
        mapCoords.z = 1024 + mapCoords.z;

        int16_t previewZOffset = ted.Definition.preview_z_offset;
        mapCoords.z -= previewZOffset;

        const ScreenCoordsXY rotatedScreenCoords = Translate3DTo2DWithZ(GetCurrentRotation(), mapCoords);

        dpi.x += rotatedScreenCoords.x - widgetWidth / 2;
        dpi.y += rotatedScreenCoords.y - widgetHeight / 2 - 16;

        DrawTrackPieceHelper(dpi, rideIndex, trackType, trackDirection, liftHillAndInvertedState, { 4096, 4096 }, 1024);
    }

    void DrawTrackPieceHelper(
        DrawPixelInfo& dpi, RideId rideIndex, int32_t trackType, int32_t trackDirection, int32_t liftHillAndInvertedState,
        const CoordsXY& originCoords, int32_t originZ)
    {
        TileElement tempSideTrackTileElement{ 0x80, 0x8F, 128, 128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
        TileElement tempTrackTileElement{};
        TileElement* backupTileElementArrays[5]{};
        PaintSession* session = PaintSessionAlloc(dpi, 0);
        trackDirection &= 3;

        auto currentRide = GetRide(rideIndex);
        if (currentRide == nullptr)
        {
            return;
        }

        auto preserveMapSize = gMapSize;

        gMapSize = { MAXIMUM_MAP_SIZE_TECHNICAL, MAXIMUM_MAP_SIZE_TECHNICAL };

        // Setup non changing parts of the temporary track tile element
        tempTrackTileElement.SetType(TileElementType::Track);
        tempTrackTileElement.SetDirection(trackDirection);
        tempTrackTileElement.AsTrack()->SetHasChain((liftHillAndInvertedState & CONSTRUCTION_LIFT_HILL_SELECTED) != 0);
        tempTrackTileElement.SetLastForTile(true);
        tempTrackTileElement.AsTrack()->SetTrackType(trackType);
        tempTrackTileElement.AsTrack()->SetRideType(currentRide->type);
        tempTrackTileElement.AsTrack()->SetHasCableLift(false);
        tempTrackTileElement.AsTrack()->SetInverted((liftHillAndInvertedState & CONSTRUCTION_INVERTED_TRACK_SELECTED) != 0);
        tempTrackTileElement.AsTrack()->SetColourScheme(RIDE_COLOUR_SCHEME_MAIN);
        // Skipping seat rotation, should not be necessary for a temporary piece.
        tempTrackTileElement.AsTrack()->SetRideIndex(rideIndex);

        const auto& ted = GetTrackElementDescriptor(trackType);
        const auto* trackBlock = ted.Block;
        const auto* rideEntry = currentRide->GetRideEntry();
        auto clearanceHeight = (rideEntry != nullptr) ? rideEntry->Clearance
                                                      : currentRide->GetRideTypeDescriptor().Heights.ClearanceHeight;

        while (trackBlock->index != 255)
        {
            auto quarterTile = trackBlock->var_08.Rotate(trackDirection);
            CoordsXY offsets = { trackBlock->x, trackBlock->y };
            CoordsXY coords = originCoords + offsets.Rotate(trackDirection);

            int32_t baseZ = originZ + trackBlock->z;
            int32_t clearanceZ = trackBlock->var_07 + clearanceHeight + baseZ + (4 * COORDS_Z_STEP);

            auto centreTileCoords = TileCoordsXY{ coords };
            auto eastTileCoords = centreTileCoords + TileDirectionDelta[TILE_ELEMENT_DIRECTION_EAST];
            auto westTileCoords = centreTileCoords + TileDirectionDelta[TILE_ELEMENT_DIRECTION_WEST];
            auto northTileCoords = centreTileCoords + TileDirectionDelta[TILE_ELEMENT_DIRECTION_NORTH];
            auto southTileCoords = centreTileCoords + TileDirectionDelta[TILE_ELEMENT_DIRECTION_SOUTH];

            // Replace map elements with temporary ones containing track
            backupTileElementArrays[0] = MapGetFirstElementAt(centreTileCoords);
            backupTileElementArrays[1] = MapGetFirstElementAt(eastTileCoords);
            backupTileElementArrays[2] = MapGetFirstElementAt(westTileCoords);
            backupTileElementArrays[3] = MapGetFirstElementAt(northTileCoords);
            backupTileElementArrays[4] = MapGetFirstElementAt(southTileCoords);
            MapSetTileElement(centreTileCoords, &tempTrackTileElement);
            MapSetTileElement(eastTileCoords, &tempSideTrackTileElement);
            MapSetTileElement(westTileCoords, &tempSideTrackTileElement);
            MapSetTileElement(northTileCoords, &tempSideTrackTileElement);
            MapSetTileElement(southTileCoords, &tempSideTrackTileElement);

            // Set the temporary track element
            tempTrackTileElement.SetOccupiedQuadrants(quarterTile.GetBaseQuarterOccupied());
            tempTrackTileElement.SetBaseZ(baseZ);
            tempTrackTileElement.SetClearanceZ(clearanceZ);
            tempTrackTileElement.AsTrack()->SetSequenceIndex(trackBlock->index);

            // Draw this map tile
            TileElementPaintSetup(*session, coords, true);

            // Restore map elements
            MapSetTileElement(centreTileCoords, backupTileElementArrays[0]);
            MapSetTileElement(eastTileCoords, backupTileElementArrays[1]);
            MapSetTileElement(westTileCoords, backupTileElementArrays[2]);
            MapSetTileElement(northTileCoords, backupTileElementArrays[3]);
            MapSetTileElement(southTileCoords, backupTileElementArrays[4]);

            trackBlock++;
        }

        gMapSize = preserveMapSize;

        PaintSessionArrange(*session);
        PaintDrawStructs(*session);
        PaintSessionFree(session);
    }
};

static void WindowRideConstructionUpdateDisabledPieces(ObjectEntryIndex rideType)
{
    RideTrackGroup disabledPieces{};
    const auto& rtd = GetRideTypeDescriptor(rideType);
    if (rtd.HasFlag(RIDE_TYPE_FLAG_HAS_TRACK))
    {
        // Set all pieces as “disabled”. When looping over the ride entries,
        // pieces will be re-enabled as soon as a single entry supports it.
        disabledPieces.flip();

        auto& objManager = OpenRCT2::GetContext()->GetObjectManager();
        auto& rideEntries = objManager.GetAllRideEntries(rideType);
        // If there are no vehicles for this ride type, enable all the track pieces.
        if (rideEntries.size() == 0)
        {
            disabledPieces.reset();
        }

        for (auto rideEntryIndex : rideEntries)
        {
            const auto* currentRideEntry = GetRideEntryByIndex(rideEntryIndex);
            if (currentRideEntry == nullptr)
                continue;

            // Non-default vehicle visuals do not use this system, so we have to assume it supports all the track pieces.
            if (currentRideEntry->Cars[0].PaintStyle != VEHICLE_VISUAL_DEFAULT || rideType == RIDE_TYPE_CHAIRLIFT
                || (currentRideEntry->Cars[0].flags & CAR_ENTRY_FLAG_SLIDE_SWING))
            {
                disabledPieces.reset();
                break;
            }

            // Any pieces that this ride entry supports must be taken out of the array.
            auto supportedPieces = RideEntryGetSupportedTrackPieces(*currentRideEntry);
            disabledPieces &= supportedPieces.flip();
        }
    }

    UpdateDisabledRidePieces(disabledPieces);
}

/**
 *
 *  rct2: 0x006CB481
 */
WindowBase* WindowRideConstructionOpen()
{
    RideId rideIndex = _currentRideIndex;
    CloseRideWindowForConstruction(rideIndex);

    auto currentRide = GetRide(rideIndex);
    if (currentRide == nullptr)
    {
        return nullptr;
    }

    WindowRideConstructionUpdateDisabledPieces(currentRide->type);

    const auto& rtd = currentRide->GetRideTypeDescriptor();
    switch (rtd.ConstructionWindowContext)
    {
        case RideConstructionWindowContext::Maze:
            return ContextOpenWindowView(WV_MAZE_CONSTRUCTION);
        case RideConstructionWindowContext::Default:
            return WindowCreate<RideConstructionWindow>(
                WindowClass::RideConstruction, ScreenCoordsXY(0, 29), WW, WH, WF_NO_AUTO_CLOSE);
    }
    return WindowCreate<RideConstructionWindow>(WindowClass::RideConstruction, ScreenCoordsXY(0, 29), WW, WH, WF_NO_AUTO_CLOSE);
}

static void CloseConstructWindowOnCompletion(const Ride& ride)
{
    if (_rideConstructionState == RideConstructionState::State0)
    {
        auto w = WindowFindByClass(WindowClass::RideConstruction);
        if (w != nullptr)
        {
            if (RideAreAllPossibleEntrancesAndExitsBuilt(ride).Successful)
            {
                WindowClose(*w);
            }
            else
            {
                WindowEventMouseUpCall(w, WIDX_ENTRANCE);
            }
        }
    }
}

static void WindowRideConstructionDoEntranceExitCheck()
{
    auto w = WindowFindByClass(WindowClass::RideConstruction);
    auto ride = GetRide(_currentRideIndex);
    if (w == nullptr || ride == nullptr)
    {
        return;
    }

    if (_rideConstructionState == RideConstructionState::State0)
    {
        w = WindowFindByClass(WindowClass::RideConstruction);
        if (w != nullptr)
        {
            if (!RideAreAllPossibleEntrancesAndExitsBuilt(*ride).Successful)
            {
                WindowEventMouseUpCall(w, WC_RIDE_CONSTRUCTION__WIDX_ENTRANCE);
            }
        }
    }
}

static void RideConstructPlacedForwardGameActionCallback(const GameAction* ga, const GameActions::Result* result)
{
    if (result->Error != GameActions::Status::Ok)
    {
        WindowRideConstructionUpdateActiveElements();
        return;
    }
    auto ride = GetRide(_currentRideIndex);
    if (ride != nullptr)
    {
        int32_t trackDirection = _currentTrackPieceDirection;
        auto trackPos = _currentTrackBegin;
        if (!(trackDirection & 4))
        {
            trackPos -= CoordsDirectionDelta[trackDirection];
        }

        CoordsXYE next_track;
        if (TrackBlockGetNextFromZero(trackPos, *ride, trackDirection, &next_track, &trackPos.z, &trackDirection, false))
        {
            _currentTrackBegin.x = next_track.x;
            _currentTrackBegin.y = next_track.y;
            _currentTrackBegin.z = trackPos.z;
            _currentTrackPieceDirection = next_track.element->GetDirection();
            _currentTrackPieceType = next_track.element->AsTrack()->GetTrackType();
            _currentTrackSelectionFlags = 0;
            _rideConstructionState = RideConstructionState::Selected;
            _rideConstructionNextArrowPulse = 0;
            gMapSelectFlags &= ~MAP_SELECT_FLAG_ENABLE_ARROW;
            RideSelectNextSection();
        }
        else
        {
            _rideConstructionState = RideConstructionState::State0;
        }

        WindowRideConstructionDoEntranceExitCheck();
        WindowRideConstructionUpdateActiveElements();
    }

    WindowCloseByClass(WindowClass::Error);
    if (ride != nullptr)
        CloseConstructWindowOnCompletion(*ride);
}

static void RideConstructPlacedBackwardGameActionCallback(const GameAction* ga, const GameActions::Result* result)
{
    if (result->Error != GameActions::Status::Ok)
    {
        WindowRideConstructionUpdateActiveElements();
        return;
    }
    auto ride = GetRide(_currentRideIndex);
    if (ride != nullptr)
    {
        auto trackDirection = DirectionReverse(_currentTrackPieceDirection);
        auto trackPos = _currentTrackBegin;
        if (!(trackDirection & 4))
        {
            trackPos += CoordsDirectionDelta[trackDirection];
        }

        TrackBeginEnd trackBeginEnd;
        if (TrackBlockGetPreviousFromZero(trackPos, *ride, trackDirection, &trackBeginEnd))
        {
            _currentTrackBegin.x = trackBeginEnd.begin_x;
            _currentTrackBegin.y = trackBeginEnd.begin_y;
            _currentTrackBegin.z = trackBeginEnd.begin_z;
            _currentTrackPieceDirection = trackBeginEnd.begin_direction;
            _currentTrackPieceType = trackBeginEnd.begin_element->AsTrack()->GetTrackType();
            _currentTrackSelectionFlags = 0;
            _rideConstructionState = RideConstructionState::Selected;
            _rideConstructionNextArrowPulse = 0;
            gMapSelectFlags &= ~MAP_SELECT_FLAG_ENABLE_ARROW;
            RideSelectPreviousSection();
        }
        else
        {
            _rideConstructionState = RideConstructionState::State0;
        }

        WindowRideConstructionUpdateActiveElements();
    }

    WindowCloseByClass(WindowClass::Error);
    if (ride != nullptr)
        CloseConstructWindowOnCompletion(*ride);
}

/**
 *
 *  rct2: 0x006CC538
 */
static std::optional<CoordsXY> RideGetPlacePositionFromScreenPosition(ScreenCoordsXY screenCoords)
{
    CoordsXY mapCoords;

    if (!_trackPlaceCtrlState)
    {
        if (gInputPlaceObjectModifier & PLACE_OBJECT_MODIFIER_COPY_Z)
        {
            auto info = GetMapCoordinatesFromPos(screenCoords, 0xFCCA);
            if (info.SpriteType != ViewportInteractionItem::None)
            {
                _trackPlaceCtrlZ = info.Element->GetBaseZ();
                _trackPlaceCtrlState = true;
            }
        }
    }
    else
    {
        if (!(gInputPlaceObjectModifier & PLACE_OBJECT_MODIFIER_COPY_Z))
        {
            _trackPlaceCtrlState = false;
        }
    }

    if (!_trackPlaceShiftState)
    {
        if (gInputPlaceObjectModifier & PLACE_OBJECT_MODIFIER_SHIFT_Z)
        {
            _trackPlaceShiftState = true;
            _trackPlaceShiftStart = screenCoords;
            _trackPlaceShiftZ = 0;
        }
    }
    else
    {
        if (gInputPlaceObjectModifier & PLACE_OBJECT_MODIFIER_SHIFT_Z)
        {
            uint16_t maxHeight = ZoomLevel::max().ApplyTo(std::numeric_limits<decltype(TileElement::BaseHeight)>::max() - 32);

            _trackPlaceShiftZ = _trackPlaceShiftStart.y - screenCoords.y + 4;
            // Scale delta by zoom to match mouse position.
            auto* mainWnd = WindowGetMain();
            if (mainWnd != nullptr && mainWnd->viewport != nullptr)
            {
                _trackPlaceShiftZ = mainWnd->viewport->zoom.ApplyTo(_trackPlaceShiftZ);
            }
            _trackPlaceShiftZ = Floor2(_trackPlaceShiftZ, 8);

            // Clamp to maximum possible value of BaseHeight can offer.
            _trackPlaceShiftZ = std::min<int16_t>(_trackPlaceShiftZ, maxHeight);

            screenCoords = _trackPlaceShiftStart;
        }
        else
        {
            _trackPlaceShiftState = false;
        }
    }

    if (!_trackPlaceCtrlState)
    {
        mapCoords = ViewportInteractionGetTileStartAtCursor(screenCoords);
        if (mapCoords.IsNull())
            return std::nullopt;

        _trackPlaceZ = 0;
        if (_trackPlaceShiftState)
        {
            auto surfaceElement = MapGetSurfaceElementAt(mapCoords);
            if (surfaceElement == nullptr)
                return std::nullopt;
            auto mapZ = Floor2(surfaceElement->GetBaseZ(), 16);
            mapZ += _trackPlaceShiftZ;
            mapZ = std::max<int16_t>(mapZ, 16);
            _trackPlaceZ = mapZ;
        }
    }
    else
    {
        auto mapZ = _trackPlaceCtrlZ;
        auto mapXYCoords = ScreenGetMapXYWithZ(screenCoords, mapZ);
        if (mapXYCoords.has_value())
        {
            mapCoords = mapXYCoords.value();
        }
        else
        {
            return std::nullopt;
        }

        if (_trackPlaceShiftState != 0)
        {
            mapZ += _trackPlaceShiftZ;
        }
        _trackPlaceZ = std::max<int32_t>(mapZ, 16);
    }

    if (mapCoords.x == LOCATION_NULL)
        return std::nullopt;

    return mapCoords.ToTileStart();
}

/**
 *
 *  rct2: 0x006C84CE
 */
void WindowRideConstructionUpdateActiveElementsImpl()
{
    WindowRideConstructionUpdateEnabledTrackPieces();
    if (auto currentRide = GetRide(_currentRideIndex);
        !currentRide || currentRide->GetRideTypeDescriptor().HasFlag(RIDE_TYPE_FLAG_IS_MAZE))
    {
        return;
    }

    auto window = static_cast<RideConstructionWindow*>(WindowFindByClass(WindowClass::RideConstruction));
    if (!window)
    {
        return;
    }

    window->UpdateMapSelection();

    _selectedTrackType = TrackElemType::None;
    if (_rideConstructionState == RideConstructionState::Selected)
    {
        TileElement* tileElement;
        if (GetTrackElementOriginAndApplyChanges(
                { _currentTrackBegin, static_cast<Direction>(_currentTrackPieceDirection & 3) }, _currentTrackPieceType, 0,
                &tileElement, 0)
            != std::nullopt)
        {
            _selectedTrackType = tileElement->AsTrack()->GetTrackType();
            if (TrackTypeHasSpeedSetting(tileElement->AsTrack()->GetTrackType()))
                _currentBrakeSpeed2 = tileElement->AsTrack()->GetBrakeBoosterSpeed();
            _currentSeatRotationAngle = tileElement->AsTrack()->GetSeatRotation();
        }
    }

    window->UpdatePossibleRideConfigurations();
    window->UpdateWidgets();
}

/**
 *
 *  rct2: 0x006C6A77
 */
void WindowRideConstructionUpdateEnabledTrackPieces()
{
    auto ride = GetRide(_currentRideIndex);
    if (ride == nullptr)
        return;

    auto rideEntry = ride->GetRideEntry();
    if (rideEntry == nullptr)
        return;

    int32_t rideType = RideGetAlternativeType(*ride);
    UpdateEnabledRidePieces(rideType);
}

/**
 *
 *  rct2: 0x006C94D8
 */
void UpdateGhostTrackAndArrow()
{
    RideId rideIndex;
    int32_t direction, type, liftHillAndAlternativeState;
    CoordsXYZ trackPos{};

    if (_currentTrackSelectionFlags & TRACK_SELECTION_FLAG_TRACK_PLACE_ACTION_QUEUED)
    {
        return;
    }

    // Recheck if area is fine for new track.
    // Set by footpath placement
    if (_currentTrackSelectionFlags & TRACK_SELECTION_FLAG_RECHECK)
    {
        RideConstructionInvalidateCurrentTrack();
        _currentTrackSelectionFlags &= ~TRACK_SELECTION_FLAG_RECHECK;
    }

    switch (_rideConstructionState)
    {
        case RideConstructionState::Front:
        case RideConstructionState::Back:
        {
            // place ghost piece
            if (!(_currentTrackSelectionFlags & TRACK_SELECTION_FLAG_TRACK))
            {
                if (WindowRideConstructionUpdateState(
                        &type, &direction, &rideIndex, &liftHillAndAlternativeState, &trackPos, nullptr))
                {
                    RideConstructionRemoveGhosts();
                }
                else
                {
                    _currentTrackPrice = PlaceProvisionalTrackPiece(
                        rideIndex, type, direction, liftHillAndAlternativeState, trackPos);
                    WindowRideConstructionUpdateActiveElements();
                }
            }
            // update flashing arrow
            auto curTime = Platform::GetTicks();
            if (_rideConstructionNextArrowPulse >= curTime)
                break;
            _rideConstructionNextArrowPulse = curTime + ARROW_PULSE_DURATION;

            _currentTrackSelectionFlags ^= TRACK_SELECTION_FLAG_ARROW;
            trackPos = _currentTrackBegin;
            direction = _currentTrackPieceDirection;
            type = _currentTrackPieceType;
            // diagonal pieces trigger this
            if (direction >= 4)
                direction += 4;
            if (_rideConstructionState == RideConstructionState::Back)
                direction = DirectionReverse(direction);
            gMapSelectArrowPosition = trackPos;
            gMapSelectArrowDirection = direction;
            gMapSelectFlags &= ~MAP_SELECT_FLAG_ENABLE_ARROW;
            if (_currentTrackSelectionFlags & TRACK_SELECTION_FLAG_ARROW)
                gMapSelectFlags |= MAP_SELECT_FLAG_ENABLE_ARROW;
            MapInvalidateTileFull(trackPos);
            break;
        }
        case RideConstructionState::Selected:
        {
            auto curTime = Platform::GetTicks();
            if (_rideConstructionNextArrowPulse >= curTime)
                break;
            _rideConstructionNextArrowPulse = curTime + ARROW_PULSE_DURATION;

            _currentTrackSelectionFlags ^= TRACK_SELECTION_FLAG_ARROW;
            direction = _currentTrackPieceDirection & 3;
            type = _currentTrackPieceType;
            uint16_t flags = _currentTrackSelectionFlags & TRACK_SELECTION_FLAG_ARROW ? TRACK_ELEMENT_SET_HIGHLIGHT_TRUE
                                                                                      : TRACK_ELEMENT_SET_HIGHLIGHT_FALSE;
            auto newCoords = GetTrackElementOriginAndApplyChanges(
                { _currentTrackBegin, static_cast<Direction>(direction) }, type, 0, nullptr, flags);
            if (!newCoords.has_value())
            {
                RideConstructionRemoveGhosts();
                _rideConstructionState = RideConstructionState::State0;
            }
            break;
        }
        case RideConstructionState::MazeBuild:
        case RideConstructionState::MazeMove:
        case RideConstructionState::MazeFill:
        {
            auto curTime = Platform::GetTicks();
            if (_rideConstructionNextArrowPulse >= curTime)
                break;
            _rideConstructionNextArrowPulse = curTime + ARROW_PULSE_DURATION;

            _currentTrackSelectionFlags ^= TRACK_SELECTION_FLAG_ARROW;
            trackPos = CoordsXYZ{ _currentTrackBegin.x & 0xFFE0, _currentTrackBegin.y & 0xFFE0, _currentTrackBegin.z + 15 };
            gMapSelectArrowPosition = trackPos;
            gMapSelectArrowDirection = 4;
            if (((_currentTrackBegin.x & 0x1F) | (_currentTrackBegin.y & 0x1F)) != 0)
            {
                gMapSelectArrowDirection = 6;
                if (((_currentTrackBegin.x & 0x1F) & (_currentTrackBegin.y & 0x1F)) == 0)
                {
                    gMapSelectArrowDirection = 5;
                    if ((_currentTrackBegin.y & 0x1F) == 0)
                        gMapSelectArrowDirection = 7;
                }
            }
            gMapSelectFlags &= ~MAP_SELECT_FLAG_ENABLE_ARROW;
            if (_currentTrackSelectionFlags & TRACK_SELECTION_FLAG_ARROW)
                gMapSelectFlags |= MAP_SELECT_FLAG_ENABLE_ARROW;
            MapInvalidateTileFull(trackPos);
            break;
        }
        default:
            break;
    }
}

/**
 *
 *  rct2: 0x006CC6A8
 */
void RideConstructionToolupdateConstruct(const ScreenCoordsXY& screenCoords)
{
    int32_t z;
    const PreviewTrack* trackBlock;

    MapInvalidateMapSelectionTiles();
    gMapSelectFlags &= ~MAP_SELECT_FLAG_ENABLE;
    gMapSelectFlags &= ~MAP_SELECT_FLAG_ENABLE_CONSTRUCT;
    gMapSelectFlags &= ~MAP_SELECT_FLAG_ENABLE_ARROW;
    auto mapCoords = RideGetPlacePositionFromScreenPosition(screenCoords);
    if (!mapCoords)
    {
        RideConstructionInvalidateCurrentTrack();
        MapInvalidateMapSelectionTiles();
        return;
    }

    z = _trackPlaceZ;
    if (z == 0)
        z = MapGetHighestZ(*mapCoords);

    gMapSelectFlags |= MAP_SELECT_FLAG_ENABLE_CONSTRUCT;
    gMapSelectFlags |= MAP_SELECT_FLAG_ENABLE_ARROW;
    gMapSelectFlags &= ~MAP_SELECT_FLAG_GREEN;
    gMapSelectArrowPosition = CoordsXYZ{ *mapCoords, z };
    gMapSelectArrowDirection = _currentTrackPieceDirection;
    gMapSelectionTiles.clear();
    gMapSelectionTiles.push_back(*mapCoords);

    RideId rideIndex;
    int32_t trackType, trackDirection, liftHillAndAlternativeState;
    if (WindowRideConstructionUpdateState(
            &trackType, &trackDirection, &rideIndex, &liftHillAndAlternativeState, nullptr, nullptr))
    {
        RideConstructionInvalidateCurrentTrack();
        MapInvalidateMapSelectionTiles();
        return;
    }
    _currentTrackPieceType = trackType;
    auto ride = GetRide(_currentRideIndex);
    if (!ride)
    {
        return;
    }

    const auto& rtd = ride->GetRideTypeDescriptor();
    if (!rtd.HasFlag(RIDE_TYPE_FLAG_IS_MAZE))
    {
        auto window = static_cast<RideConstructionWindow*>(WindowFindByClass(WindowClass::RideConstruction));
        if (!window)
        {
            return;
        }
        // Re-using this other code, very slight difference from original
        //   - Original code checks for MSB mask instead of 255 on trackPart->var_00
        //   - Original code checks this first as its already set origin tile, probably just a micro optimisation
        window->SelectMapTiles(trackType, trackDirection, *mapCoords);
    }

    gMapSelectArrowPosition.z = z;
    if (_trackPlaceZ == 0)
    {
        // Raise z above all slopes and water
        if (gMapSelectFlags & MAP_SELECT_FLAG_ENABLE_CONSTRUCT)
        {
            int32_t highestZ = 0;
            for (const auto& selectedTile : gMapSelectionTiles)
            {
                if (MapIsLocationValid(selectedTile))
                {
                    z = MapGetHighestZ(selectedTile);
                    if (z > highestZ)
                        highestZ = z;
                }
            }
        }
        // Loc6CC8BF:
        // z = MapGetHighestZ(x >> 5, y >> 5);
    }
    // Loc6CC91B:
    const auto& ted = GetTrackElementDescriptor(trackType);
    trackBlock = ted.Block;
    int32_t bx = 0;
    do
    {
        bx = std::min<int32_t>(bx, trackBlock->z);
        trackBlock++;
    } while (trackBlock->index != 255);
    z -= bx;

    gMapSelectArrowPosition.z = z;
    _currentTrackBegin.x = mapCoords->x;
    _currentTrackBegin.y = mapCoords->y;
    _currentTrackBegin.z = z;
    if ((_currentTrackSelectionFlags & TRACK_SELECTION_FLAG_TRACK) && _currentTrackBegin == _previousTrackPiece)
    {
        MapInvalidateMapSelectionTiles();
        return;
    }

    _previousTrackPiece = _currentTrackBegin;
    // search for appropriate z value for ghost, up to max ride height
    int numAttempts = (z <= MAX_TRACK_HEIGHT ? ((MAX_TRACK_HEIGHT - z) / COORDS_Z_STEP + 1) : 2);

    if (rtd.HasFlag(RIDE_TYPE_FLAG_IS_MAZE))
    {
        for (int zAttempts = 0; zAttempts < numAttempts; ++zAttempts)
        {
            CoordsXYZ trackPos{};
            WindowRideConstructionUpdateState(
                &trackType, &trackDirection, &rideIndex, &liftHillAndAlternativeState, &trackPos, nullptr);
            _currentTrackPrice = PlaceProvisionalTrackPiece(
                rideIndex, trackType, trackDirection, liftHillAndAlternativeState, trackPos);
            if (_currentTrackPrice != MONEY64_UNDEFINED)
                break;

            _currentTrackBegin.z -= 8;
            if (_currentTrackBegin.z < 0)
                break;

            _currentTrackBegin.z += 16;
        }

        auto intent = Intent(INTENT_ACTION_UPDATE_MAZE_CONSTRUCTION);
        ContextBroadcastIntent(&intent);
        MapInvalidateMapSelectionTiles();
        return;
    }

    for (int zAttempts = 0; zAttempts < numAttempts; ++zAttempts)
    {
        CoordsXYZ trackPos{};
        WindowRideConstructionUpdateState(
            &trackType, &trackDirection, &rideIndex, &liftHillAndAlternativeState, &trackPos, nullptr);
        _currentTrackPrice = PlaceProvisionalTrackPiece(
            rideIndex, trackType, trackDirection, liftHillAndAlternativeState, trackPos);
        mapCoords = trackPos;
        z = trackPos.z;
        if (_currentTrackPrice != MONEY64_UNDEFINED)
            break;

        _currentTrackBegin.z -= 8;
        if (_currentTrackBegin.z < 0)
            break;

        _currentTrackBegin.z += 16;
    }

    if (_autoRotatingShop && _rideConstructionState == RideConstructionState::Place
        && ride->GetRideTypeDescriptor().HasFlag(RIDE_TYPE_FLAG_IS_SHOP_OR_FACILITY))
    {
        PathElement* pathsByDir[NumOrthogonalDirections];

        bool keepOrientation = false;
        for (int8_t i = 0; i < NumOrthogonalDirections; i++)
        {
            const auto testLoc = CoordsXYZ{ *mapCoords + CoordsDirectionDelta[i], z };
            if (!MapIsLocationOwned(testLoc))
            {
                pathsByDir[i] = nullptr;
                continue;
            }

            pathsByDir[i] = MapGetFootpathElement(testLoc);

            if (pathsByDir[i] != nullptr && pathsByDir[i]->IsSloped() && pathsByDir[i]->GetSlopeDirection() != i)
            {
                pathsByDir[i] = nullptr;
            }

            // Sloped path on the level below
            if (pathsByDir[i] == nullptr)
            {
                pathsByDir[i] = MapGetFootpathElement({ *mapCoords + CoordsDirectionDelta[i], z - PATH_HEIGHT_STEP });

                if (pathsByDir[i] != nullptr
                    && (!pathsByDir[i]->IsSloped() || pathsByDir[i]->GetSlopeDirection() != DirectionReverse(i)))
                {
                    pathsByDir[i] = nullptr;
                }
            }

            if (pathsByDir[i] != nullptr && pathsByDir[i]->IsQueue())
            {
                pathsByDir[i] = nullptr;
            }

            if (pathsByDir[i] != nullptr && i == _currentTrackPieceDirection)
            {
                keepOrientation = true;
                break;
            }
        }

        if (!keepOrientation)
        {
            for (int8_t i = 0; i < NumOrthogonalDirections; i++)
            {
                if (pathsByDir[i] != nullptr)
                {
                    _currentTrackPieceDirection = i;

                    CoordsXYZ trackPos{};
                    WindowRideConstructionUpdateState(
                        &trackType, &trackDirection, &rideIndex, &liftHillAndAlternativeState, &trackPos, nullptr);
                    PlaceProvisionalTrackPiece(rideIndex, trackType, trackDirection, liftHillAndAlternativeState, trackPos);
                    gMapSelectArrowDirection = _currentTrackPieceDirection;
                    break;
                }
            }
        }
    }

    WindowRideConstructionUpdateActiveElements();
    MapInvalidateMapSelectionTiles();
}

/**
 *
 *  rct2: 0x006CD354
 */
void RideConstructionToolupdateEntranceExit(const ScreenCoordsXY& screenCoords)
{
    MapInvalidateSelectionRect();
    MapInvalidateMapSelectionTiles();
    gMapSelectFlags &= ~MAP_SELECT_FLAG_ENABLE;
    gMapSelectFlags &= ~MAP_SELECT_FLAG_ENABLE_CONSTRUCT;
    gMapSelectFlags &= ~MAP_SELECT_FLAG_ENABLE_ARROW;
    CoordsXYZD entranceOrExitCoords = RideGetEntranceOrExitPositionFromScreenPosition(screenCoords);
    if (gRideEntranceExitPlaceDirection == INVALID_DIRECTION)
    {
        RideConstructionInvalidateCurrentTrack();
        return;
    }
    gMapSelectFlags |= MAP_SELECT_FLAG_ENABLE;
    gMapSelectFlags |= MAP_SELECT_FLAG_ENABLE_ARROW;
    gMapSelectType = MAP_SELECT_TYPE_FULL;
    gMapSelectPositionA = entranceOrExitCoords;
    gMapSelectPositionB = entranceOrExitCoords;
    gMapSelectArrowPosition = entranceOrExitCoords;
    gMapSelectArrowDirection = DirectionReverse(entranceOrExitCoords.direction);
    MapInvalidateSelectionRect();

    entranceOrExitCoords.direction = DirectionReverse(gRideEntranceExitPlaceDirection);
    StationIndex stationNum = gRideEntranceExitPlaceStationIndex;
    if (!(_currentTrackSelectionFlags & TRACK_SELECTION_FLAG_ENTRANCE_OR_EXIT)
        || entranceOrExitCoords != gRideEntranceExitGhostPosition || stationNum != gRideEntranceExitGhostStationIndex)
    {
        auto ride = GetRide(_currentRideIndex);
        if (ride != nullptr)
        {
            _currentTrackPrice = RideEntranceExitPlaceGhost(
                *ride, entranceOrExitCoords, entranceOrExitCoords.direction, gRideEntranceExitPlaceType, stationNum);
        }
        WindowRideConstructionUpdateActiveElements();
    }
}

/**
 *
 *  rct2: 0x006CCA73
 */
void RideConstructionTooldownConstruct(const ScreenCoordsXY& screenCoords)
{
    const CursorState* state = ContextGetCursorState();

    WindowBase* w;

    MapInvalidateMapSelectionTiles();
    RideConstructionInvalidateCurrentTrack();

    CoordsXYZ mapCoords{};
    int32_t trackType, z, highestZ;

    if (WindowRideConstructionUpdateState(&trackType, nullptr, nullptr, nullptr, nullptr, nullptr))
        return;

    z = mapCoords.z;
    _currentTrackPieceType = trackType;

    // Raise z above all slopes and water
    highestZ = 0;
    if (gMapSelectFlags & MAP_SELECT_FLAG_ENABLE_CONSTRUCT)
    {
        for (const auto& selectedTile : gMapSelectionTiles)
        {
            if (!MapIsLocationValid(selectedTile))
                continue;

            z = MapGetHighestZ(selectedTile);
            if (z > highestZ)
                highestZ = z;
        }
    }

    gMapSelectFlags &= ~MAP_SELECT_FLAG_ENABLE;
    gMapSelectFlags &= ~MAP_SELECT_FLAG_ENABLE_CONSTRUCT;
    gMapSelectFlags &= ~MAP_SELECT_FLAG_ENABLE_ARROW;
    auto ridePlacePosition = RideGetPlacePositionFromScreenPosition(screenCoords);
    if (!ridePlacePosition)
        return;

    mapCoords = { *ridePlacePosition, z };

    z = _trackPlaceZ;
    if (z == 0)
        z = MapGetHighestZ(mapCoords);

    ToolCancel();

    auto ride = GetRide(_currentRideIndex);
    if (ride == nullptr)
        return;

    if (_trackPlaceZ == 0)
    {
        const auto& ted = GetTrackElementDescriptor(_currentTrackPieceType);
        const PreviewTrack* trackBlock = ted.Block;
        int32_t bx = 0;
        do
        {
            bx = std::min<int32_t>(bx, trackBlock->z);
            trackBlock++;
        } while (trackBlock->index != 255);
        z -= bx;

        // FIX not sure exactly why it starts trial and error place from a lower Z, but it causes issues with disable clearance
        if (!gCheatsDisableClearanceChecks && z > MINIMUM_LAND_HEIGHT_BIG)
        {
            z -= LAND_HEIGHT_STEP;
        }
    }
    else
    {
        z = _trackPlaceZ;
    }

    // search for z value to build at, up to max ride height
    int numAttempts = (z <= MAX_TRACK_HEIGHT ? ((MAX_TRACK_HEIGHT - z) / COORDS_Z_STEP + 1) : 2);

    const auto& rtd = ride->GetRideTypeDescriptor();
    if (rtd.HasFlag(RIDE_TYPE_FLAG_IS_MAZE))
    {
        for (int32_t zAttempts = 0; zAttempts < numAttempts; ++zAttempts)
        {
            _rideConstructionState = RideConstructionState::MazeBuild;
            _currentTrackBegin.x = mapCoords.x;
            _currentTrackBegin.y = mapCoords.y;
            _currentTrackBegin.z = z;
            _currentTrackSelectionFlags = 0;
            auto intent = Intent(INTENT_ACTION_UPDATE_MAZE_CONSTRUCTION);
            ContextBroadcastIntent(&intent);
            w = WindowFindByClass(WindowClass::RideConstruction);
            if (w == nullptr)
                break;

            gDisableErrorWindowSound = true;

            auto gameAction = MazeSetTrackAction(
                CoordsXYZD{ _currentTrackBegin, 0 }, true, _currentRideIndex, GC_SET_MAZE_TRACK_BUILD);
            auto mazeSetTrackResult = GameActions::Execute(&gameAction);
            if (mazeSetTrackResult.Error == GameActions::Status::Ok)
            {
                _trackPlaceCost = mazeSetTrackResult.Cost;
                _trackPlaceErrorMessage = STR_NONE;
            }
            else
            {
                _trackPlaceCost = MONEY64_UNDEFINED;
                _trackPlaceErrorMessage = std::get<StringId>(mazeSetTrackResult.ErrorMessage);
            }

            gDisableErrorWindowSound = false;

            if (mazeSetTrackResult.Error != GameActions::Status::Ok)
            {
                _rideConstructionState = RideConstructionState::Place;
                StringId errorText = std::get<StringId>(mazeSetTrackResult.ErrorMessage);
                z -= 8;
                if (errorText == STR_NOT_ENOUGH_CASH_REQUIRES || errorText == STR_CAN_ONLY_BUILD_THIS_UNDERWATER
                    || errorText == STR_CAN_ONLY_BUILD_THIS_ON_WATER || errorText == STR_RIDE_CANT_BUILD_THIS_UNDERWATER
                    || errorText == STR_CAN_ONLY_BUILD_THIS_ABOVE_GROUND || errorText == STR_TOO_HIGH_FOR_SUPPORTS
                    || zAttempts == (numAttempts - 1) || z < 0)
                {
                    OpenRCT2::Audio::Play(OpenRCT2::Audio::SoundId::Error, 0, state->position.x);
                    w = WindowFindByClass(WindowClass::RideConstruction);
                    if (w != nullptr)
                    {
                        ToolSet(*w, WIDX_CONSTRUCT, Tool::Crosshair);
                        InputSetFlag(INPUT_FLAG_6, true);
                        _trackPlaceCtrlState = false;
                        _trackPlaceShiftState = false;
                    }
                    auto intent2 = Intent(INTENT_ACTION_UPDATE_MAZE_CONSTRUCTION);
                    ContextBroadcastIntent(&intent2);
                    break;
                }
                z += 16;
            }
            else
            {
                WindowCloseByClass(WindowClass::Error);
                OpenRCT2::Audio::Play3D(OpenRCT2::Audio::SoundId::PlaceItem, _currentTrackBegin);
                break;
            }
        }
        return;
    }

    for (int32_t zAttempts = 0; zAttempts < numAttempts; ++zAttempts)
    {
        _rideConstructionState = RideConstructionState::Front;
        _currentTrackBegin.x = mapCoords.x;
        _currentTrackBegin.y = mapCoords.y;
        _currentTrackBegin.z = z;
        _currentTrackSelectionFlags = 0;
        WindowRideConstructionUpdateActiveElements();
        w = WindowFindByClass(WindowClass::RideConstruction);
        if (w == nullptr)
            break;

        gDisableErrorWindowSound = true;
        WindowEventMouseUpCall(w, WIDX_CONSTRUCT);
        gDisableErrorWindowSound = false;

        if (_trackPlaceCost == MONEY64_UNDEFINED)
        {
            StringId errorText = _trackPlaceErrorMessage;
            z -= 8;
            if (errorText == STR_NOT_ENOUGH_CASH_REQUIRES || errorText == STR_CAN_ONLY_BUILD_THIS_UNDERWATER
                || errorText == STR_CAN_ONLY_BUILD_THIS_ON_WATER || errorText == STR_CAN_ONLY_BUILD_THIS_ABOVE_GROUND
                || errorText == STR_TOO_HIGH_FOR_SUPPORTS || errorText == STR_TOO_HIGH
                || errorText == STR_LOCAL_AUTHORITY_WONT_ALLOW_CONSTRUCTION_ABOVE_TREE_HEIGHT || zAttempts == (numAttempts - 1)
                || z < 0)
            {
                int32_t saveTrackDirection = _currentTrackPieceDirection;
                auto saveCurrentTrackCurve = _currentTrackCurve;
                int32_t savePreviousTrackSlopeEnd = _previousTrackSlopeEnd;
                int32_t saveCurrentTrackSlopeEnd = _currentTrackSlopeEnd;
                int32_t savePreviousTrackBankEnd = _previousTrackBankEnd;
                int32_t saveCurrentTrackBankEnd = _currentTrackBankEnd;
                int32_t saveCurrentTrackAlternative = _currentTrackAlternative;
                int32_t saveCurrentTrackLiftHill = _currentTrackLiftHill;

                RideInitialiseConstructionWindow(*ride);

                _currentTrackPieceDirection = saveTrackDirection;
                _currentTrackCurve = saveCurrentTrackCurve;
                _previousTrackSlopeEnd = savePreviousTrackSlopeEnd;
                _currentTrackSlopeEnd = saveCurrentTrackSlopeEnd;
                _previousTrackBankEnd = savePreviousTrackBankEnd;
                _currentTrackBankEnd = saveCurrentTrackBankEnd;
                _currentTrackAlternative = saveCurrentTrackAlternative;
                _currentTrackLiftHill = saveCurrentTrackLiftHill;

                OpenRCT2::Audio::Play(OpenRCT2::Audio::SoundId::Error, 0, state->position.x);
                break;
            }

            z += 16;
        }
        else
        {
            break;
        }
    }
}

void WindowRideConstructionKeyboardShortcutTurnLeft()
{
    WindowBase* w = WindowFindByClass(WindowClass::RideConstruction);
    if (w == nullptr || WidgetIsDisabled(*w, WIDX_STRAIGHT) || w->widgets[WIDX_STRAIGHT].type == WindowWidgetType::Empty)
    {
        return;
    }

    switch (_currentTrackCurve)
    {
        case TRACK_CURVE_LEFT_SMALL:
            if (!WidgetIsDisabled(*w, WIDX_LEFT_CURVE_VERY_SMALL)
                && w->widgets[WIDX_LEFT_CURVE_VERY_SMALL].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_LEFT_CURVE_VERY_SMALL);
            }
            break;
        case TRACK_CURVE_LEFT:
            if (!WidgetIsDisabled(*w, WIDX_LEFT_CURVE_SMALL)
                && w->widgets[WIDX_LEFT_CURVE_SMALL].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_LEFT_CURVE_SMALL);
            }
            else if (
                !WidgetIsDisabled(*w, WIDX_LEFT_CURVE_VERY_SMALL)
                && w->widgets[WIDX_LEFT_CURVE_VERY_SMALL].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_LEFT_CURVE_VERY_SMALL);
            }
            else
            {
                return;
            }
            break;
        case TRACK_CURVE_LEFT_LARGE:
            if (!WidgetIsDisabled(*w, WIDX_LEFT_CURVE) && w->widgets[WIDX_LEFT_CURVE].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_LEFT_CURVE);
            }
            else if (
                !WidgetIsDisabled(*w, WIDX_LEFT_CURVE_SMALL)
                && w->widgets[WIDX_LEFT_CURVE_SMALL].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_LEFT_CURVE_SMALL);
            }
            else if (
                !WidgetIsDisabled(*w, WIDX_LEFT_CURVE_VERY_SMALL)
                && w->widgets[WIDX_LEFT_CURVE_VERY_SMALL].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_LEFT_CURVE_VERY_SMALL);
            }
            else
            {
                return;
            }
            break;
        case TRACK_CURVE_NONE:
            if (!WidgetIsDisabled(*w, WIDX_LEFT_CURVE_LARGE)
                && w->widgets[WIDX_LEFT_CURVE_LARGE].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_LEFT_CURVE_LARGE);
            }
            else if (!WidgetIsDisabled(*w, WIDX_LEFT_CURVE) && w->widgets[WIDX_LEFT_CURVE].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_LEFT_CURVE);
            }
            else if (
                !WidgetIsDisabled(*w, WIDX_LEFT_CURVE_SMALL)
                && w->widgets[WIDX_LEFT_CURVE_SMALL].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_LEFT_CURVE_SMALL);
            }
            else if (
                !WidgetIsDisabled(*w, WIDX_LEFT_CURVE_VERY_SMALL)
                && w->widgets[WIDX_LEFT_CURVE_VERY_SMALL].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_LEFT_CURVE_VERY_SMALL);
            }
            else
            {
                return;
            }
            break;
        case TRACK_CURVE_RIGHT_LARGE:
            if (!WidgetIsDisabled(*w, WIDX_STRAIGHT) && w->widgets[WIDX_STRAIGHT].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_STRAIGHT);
            }
            else if (
                !WidgetIsDisabled(*w, WIDX_LEFT_CURVE_LARGE)
                && w->widgets[WIDX_LEFT_CURVE_LARGE].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_LEFT_CURVE_LARGE);
            }
            else if (!WidgetIsDisabled(*w, WIDX_LEFT_CURVE) && w->widgets[WIDX_LEFT_CURVE].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_LEFT_CURVE);
            }
            else if (
                !WidgetIsDisabled(*w, WIDX_LEFT_CURVE_SMALL)
                && w->widgets[WIDX_LEFT_CURVE_SMALL].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_LEFT_CURVE_SMALL);
            }
            else if (
                !WidgetIsDisabled(*w, WIDX_LEFT_CURVE_VERY_SMALL)
                && w->widgets[WIDX_LEFT_CURVE_VERY_SMALL].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_LEFT_CURVE_VERY_SMALL);
            }
            else
            {
                return;
            }
            break;
        case TRACK_CURVE_RIGHT:
            if (!WidgetIsDisabled(*w, WIDX_RIGHT_CURVE_LARGE)
                && w->widgets[WIDX_RIGHT_CURVE_LARGE].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_RIGHT_CURVE_LARGE);
            }
            else if (!WidgetIsDisabled(*w, WIDX_STRAIGHT) && w->widgets[WIDX_STRAIGHT].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_STRAIGHT);
            }
            else if (
                !WidgetIsDisabled(*w, WIDX_LEFT_CURVE_LARGE)
                && w->widgets[WIDX_LEFT_CURVE_LARGE].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_LEFT_CURVE_LARGE);
            }
            else if (!WidgetIsDisabled(*w, WIDX_LEFT_CURVE) && w->widgets[WIDX_LEFT_CURVE].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_LEFT_CURVE);
            }
            else if (
                !WidgetIsDisabled(*w, WIDX_LEFT_CURVE_SMALL)
                && w->widgets[WIDX_LEFT_CURVE_SMALL].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_LEFT_CURVE_SMALL);
            }
            else if (
                !WidgetIsDisabled(*w, WIDX_LEFT_CURVE_VERY_SMALL)
                && w->widgets[WIDX_LEFT_CURVE_VERY_SMALL].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_LEFT_CURVE_VERY_SMALL);
            }
            else
            {
                return;
            }
            break;
        case TRACK_CURVE_RIGHT_SMALL:
            if (!WidgetIsDisabled(*w, WIDX_RIGHT_CURVE) && w->widgets[WIDX_RIGHT_CURVE].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_RIGHT_CURVE);
            }
            else if (
                !WidgetIsDisabled(*w, WIDX_RIGHT_CURVE_LARGE)
                && w->widgets[WIDX_RIGHT_CURVE_LARGE].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_RIGHT_CURVE_LARGE);
            }
            else if (!WidgetIsDisabled(*w, WIDX_STRAIGHT) && w->widgets[WIDX_STRAIGHT].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_STRAIGHT);
            }
            else if (
                !WidgetIsDisabled(*w, WIDX_LEFT_CURVE_LARGE)
                && w->widgets[WIDX_LEFT_CURVE_LARGE].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_LEFT_CURVE_LARGE);
            }
            else if (!WidgetIsDisabled(*w, WIDX_LEFT_CURVE) && w->widgets[WIDX_LEFT_CURVE].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_LEFT_CURVE);
            }
            else if (
                !WidgetIsDisabled(*w, WIDX_LEFT_CURVE_SMALL)
                && w->widgets[WIDX_LEFT_CURVE_SMALL].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_LEFT_CURVE_SMALL);
            }
            else if (
                !WidgetIsDisabled(*w, WIDX_LEFT_CURVE_VERY_SMALL)
                && w->widgets[WIDX_LEFT_CURVE_VERY_SMALL].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_LEFT_CURVE_VERY_SMALL);
            }
            else
            {
                return;
            }
            break;
        case TRACK_CURVE_RIGHT_VERY_SMALL:
            if (!WidgetIsDisabled(*w, WIDX_RIGHT_CURVE_SMALL)
                && w->widgets[WIDX_RIGHT_CURVE_SMALL].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_RIGHT_CURVE_SMALL);
            }
            else if (!WidgetIsDisabled(*w, WIDX_RIGHT_CURVE) && w->widgets[WIDX_RIGHT_CURVE].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_RIGHT_CURVE);
            }
            else if (
                !WidgetIsDisabled(*w, WIDX_RIGHT_CURVE_LARGE)
                && w->widgets[WIDX_RIGHT_CURVE_LARGE].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_RIGHT_CURVE_LARGE);
            }
            else if (!WidgetIsDisabled(*w, WIDX_STRAIGHT) && w->widgets[WIDX_STRAIGHT].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_STRAIGHT);
            }
            else if (
                !WidgetIsDisabled(*w, WIDX_LEFT_CURVE_LARGE)
                && w->widgets[WIDX_LEFT_CURVE_LARGE].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_LEFT_CURVE_LARGE);
            }
            else if (!WidgetIsDisabled(*w, WIDX_LEFT_CURVE) && w->widgets[WIDX_LEFT_CURVE].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_LEFT_CURVE);
            }
            else if (
                !WidgetIsDisabled(*w, WIDX_LEFT_CURVE_SMALL)
                && w->widgets[WIDX_LEFT_CURVE_SMALL].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_LEFT_CURVE_SMALL);
            }
            else if (
                !WidgetIsDisabled(*w, WIDX_LEFT_CURVE_VERY_SMALL)
                && w->widgets[WIDX_LEFT_CURVE_VERY_SMALL].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_LEFT_CURVE_VERY_SMALL);
            }
            else
            {
                return;
            }
            break;
        default:
            return;
    }
}

void WindowRideConstructionKeyboardShortcutTurnRight()
{
    WindowBase* w = WindowFindByClass(WindowClass::RideConstruction);
    if (w == nullptr || WidgetIsDisabled(*w, WIDX_STRAIGHT) || w->widgets[WIDX_STRAIGHT].type == WindowWidgetType::Empty)
    {
        return;
    }

    switch (_currentTrackCurve)
    {
        case TRACK_CURVE_RIGHT_SMALL:
            if (!WidgetIsDisabled(*w, WIDX_RIGHT_CURVE_VERY_SMALL)
                && w->widgets[WIDX_RIGHT_CURVE_VERY_SMALL].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_RIGHT_CURVE_VERY_SMALL);
            }
            break;
        case TRACK_CURVE_RIGHT:
            if (!WidgetIsDisabled(*w, WIDX_RIGHT_CURVE_SMALL)
                && w->widgets[WIDX_RIGHT_CURVE_SMALL].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_RIGHT_CURVE_SMALL);
            }
            else if (
                !WidgetIsDisabled(*w, WIDX_RIGHT_CURVE_VERY_SMALL)
                && w->widgets[WIDX_RIGHT_CURVE_VERY_SMALL].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_RIGHT_CURVE_VERY_SMALL);
            }
            else
            {
                return;
            }
            break;
        case TRACK_CURVE_RIGHT_LARGE:
            if (!WidgetIsDisabled(*w, WIDX_RIGHT_CURVE) && w->widgets[WIDX_RIGHT_CURVE].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_RIGHT_CURVE);
            }
            else if (
                !WidgetIsDisabled(*w, WIDX_RIGHT_CURVE_SMALL)
                && w->widgets[WIDX_RIGHT_CURVE_SMALL].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_RIGHT_CURVE_SMALL);
            }
            else if (
                !WidgetIsDisabled(*w, WIDX_RIGHT_CURVE_VERY_SMALL)
                && w->widgets[WIDX_RIGHT_CURVE_VERY_SMALL].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_RIGHT_CURVE_VERY_SMALL);
            }
            else
            {
                return;
            }
            break;
        case TRACK_CURVE_NONE:
            if (!WidgetIsDisabled(*w, WIDX_RIGHT_CURVE_LARGE)
                && w->widgets[WIDX_RIGHT_CURVE_LARGE].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_RIGHT_CURVE_LARGE);
            }
            else if (!WidgetIsDisabled(*w, WIDX_LEFT_CURVE) && w->widgets[WIDX_RIGHT_CURVE].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_RIGHT_CURVE);
            }
            else if (
                !WidgetIsDisabled(*w, WIDX_RIGHT_CURVE_SMALL)
                && w->widgets[WIDX_RIGHT_CURVE_SMALL].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_RIGHT_CURVE_SMALL);
            }
            else if (
                !WidgetIsDisabled(*w, WIDX_RIGHT_CURVE_VERY_SMALL)
                && w->widgets[WIDX_RIGHT_CURVE_VERY_SMALL].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_RIGHT_CURVE_VERY_SMALL);
            }
            else
            {
                return;
            }
            break;
        case TRACK_CURVE_LEFT_LARGE:
            if (!WidgetIsDisabled(*w, WIDX_STRAIGHT) && w->widgets[WIDX_STRAIGHT].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_STRAIGHT);
            }
            else if (
                !WidgetIsDisabled(*w, WIDX_RIGHT_CURVE_LARGE)
                && w->widgets[WIDX_RIGHT_CURVE_LARGE].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_RIGHT_CURVE_LARGE);
            }
            else if (!WidgetIsDisabled(*w, WIDX_RIGHT_CURVE) && w->widgets[WIDX_RIGHT_CURVE].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_RIGHT_CURVE);
            }
            else if (
                !WidgetIsDisabled(*w, WIDX_RIGHT_CURVE_SMALL)
                && w->widgets[WIDX_RIGHT_CURVE_SMALL].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_RIGHT_CURVE_SMALL);
            }
            else if (
                !WidgetIsDisabled(*w, WIDX_RIGHT_CURVE_VERY_SMALL)
                && w->widgets[WIDX_RIGHT_CURVE_VERY_SMALL].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_RIGHT_CURVE_VERY_SMALL);
            }
            else
            {
                return;
            }
            break;
        case TRACK_CURVE_LEFT:
            if (!WidgetIsDisabled(*w, WIDX_LEFT_CURVE_LARGE)
                && w->widgets[WIDX_LEFT_CURVE_LARGE].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_LEFT_CURVE_LARGE);
            }
            else if (!WidgetIsDisabled(*w, WIDX_STRAIGHT) && w->widgets[WIDX_STRAIGHT].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_STRAIGHT);
            }
            else if (
                !WidgetIsDisabled(*w, WIDX_RIGHT_CURVE_LARGE)
                && w->widgets[WIDX_RIGHT_CURVE_LARGE].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_RIGHT_CURVE_LARGE);
            }
            else if (!WidgetIsDisabled(*w, WIDX_RIGHT_CURVE) && w->widgets[WIDX_RIGHT_CURVE].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_RIGHT_CURVE);
            }
            else if (
                !WidgetIsDisabled(*w, WIDX_RIGHT_CURVE_SMALL)
                && w->widgets[WIDX_RIGHT_CURVE_SMALL].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_RIGHT_CURVE_SMALL);
            }
            else if (
                !WidgetIsDisabled(*w, WIDX_RIGHT_CURVE_VERY_SMALL)
                && w->widgets[WIDX_RIGHT_CURVE_VERY_SMALL].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_RIGHT_CURVE_VERY_SMALL);
            }
            else
            {
                return;
            }
            break;
        case TRACK_CURVE_LEFT_SMALL:
            if (!WidgetIsDisabled(*w, WIDX_LEFT_CURVE) && w->widgets[WIDX_LEFT_CURVE].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_LEFT_CURVE);
            }
            else if (
                !WidgetIsDisabled(*w, WIDX_LEFT_CURVE_LARGE)
                && w->widgets[WIDX_LEFT_CURVE_LARGE].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_LEFT_CURVE_LARGE);
            }
            else if (!WidgetIsDisabled(*w, WIDX_STRAIGHT) && w->widgets[WIDX_STRAIGHT].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_STRAIGHT);
            }
            else if (
                !WidgetIsDisabled(*w, WIDX_RIGHT_CURVE_LARGE)
                && w->widgets[WIDX_RIGHT_CURVE_LARGE].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_RIGHT_CURVE_LARGE);
            }
            else if (!WidgetIsDisabled(*w, WIDX_RIGHT_CURVE) && w->widgets[WIDX_RIGHT_CURVE].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_RIGHT_CURVE);
            }
            else if (
                !WidgetIsDisabled(*w, WIDX_RIGHT_CURVE_SMALL)
                && w->widgets[WIDX_RIGHT_CURVE_SMALL].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_RIGHT_CURVE_SMALL);
            }
            else if (
                !WidgetIsDisabled(*w, WIDX_RIGHT_CURVE_VERY_SMALL)
                && w->widgets[WIDX_RIGHT_CURVE_VERY_SMALL].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_RIGHT_CURVE_VERY_SMALL);
            }
            else
            {
                return;
            }
            break;
        case TRACK_CURVE_LEFT_VERY_SMALL:
            if (!WidgetIsDisabled(*w, WIDX_LEFT_CURVE_SMALL)
                && w->widgets[WIDX_LEFT_CURVE_SMALL].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_LEFT_CURVE_SMALL);
            }
            else if (!WidgetIsDisabled(*w, WIDX_LEFT_CURVE) && w->widgets[WIDX_LEFT_CURVE].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_LEFT_CURVE);
            }
            else if (
                !WidgetIsDisabled(*w, WIDX_LEFT_CURVE_LARGE)
                && w->widgets[WIDX_LEFT_CURVE_LARGE].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_LEFT_CURVE_LARGE);
            }
            else if (!WidgetIsDisabled(*w, WIDX_STRAIGHT) && w->widgets[WIDX_STRAIGHT].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_STRAIGHT);
            }
            else if (
                !WidgetIsDisabled(*w, WIDX_RIGHT_CURVE_LARGE)
                && w->widgets[WIDX_RIGHT_CURVE_LARGE].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_RIGHT_CURVE_LARGE);
            }
            else if (!WidgetIsDisabled(*w, WIDX_RIGHT_CURVE) && w->widgets[WIDX_RIGHT_CURVE].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_RIGHT_CURVE);
            }
            else if (
                !WidgetIsDisabled(*w, WIDX_RIGHT_CURVE_SMALL)
                && w->widgets[WIDX_RIGHT_CURVE_SMALL].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_RIGHT_CURVE_SMALL);
            }
            else if (
                !WidgetIsDisabled(*w, WIDX_RIGHT_CURVE_VERY_SMALL)
                && w->widgets[WIDX_RIGHT_CURVE_VERY_SMALL].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_RIGHT_CURVE_VERY_SMALL);
            }
            else
            {
                return;
            }
            break;
        default:
            return;
    }
}

void WindowRideConstructionKeyboardShortcutUseTrackDefault()
{
    WindowBase* w = WindowFindByClass(WindowClass::RideConstruction);
    if (w == nullptr || WidgetIsDisabled(*w, WIDX_STRAIGHT) || w->widgets[WIDX_STRAIGHT].type == WindowWidgetType::Empty)
    {
        return;
    }

    if (!WidgetIsDisabled(*w, WIDX_STRAIGHT) && w->widgets[WIDX_STRAIGHT].type != WindowWidgetType::Empty)
    {
        WindowEventMouseDownCall(w, WIDX_STRAIGHT);
    }

    if (!WidgetIsDisabled(*w, WIDX_LEVEL) && w->widgets[WIDX_LEVEL].type != WindowWidgetType::Empty)
    {
        WindowEventMouseDownCall(w, WIDX_LEVEL);
    }

    if (!WidgetIsDisabled(*w, WIDX_CHAIN_LIFT) && w->widgets[WIDX_CHAIN_LIFT].type != WindowWidgetType::Empty
        && _currentTrackLiftHill & CONSTRUCTION_LIFT_HILL_SELECTED)
    {
        WindowEventMouseDownCall(w, WIDX_CHAIN_LIFT);
    }

    if (!WidgetIsDisabled(*w, WIDX_BANK_STRAIGHT) && w->widgets[WIDX_BANK_STRAIGHT].type != WindowWidgetType::Empty)
    {
        WindowEventMouseDownCall(w, WIDX_BANK_STRAIGHT);
    }
}

void WindowRideConstructionKeyboardShortcutSlopeDown()
{
    WindowBase* w = WindowFindByClass(WindowClass::RideConstruction);
    if (w == nullptr || WidgetIsDisabled(*w, WIDX_STRAIGHT) || w->widgets[WIDX_STRAIGHT].type == WindowWidgetType::Empty)
    {
        return;
    }

    switch (_currentTrackSlopeEnd)
    {
        case TRACK_SLOPE_DOWN_60:
            if (IsTrackEnabled(TRACK_SLOPE_VERTICAL) && !WidgetIsDisabled(*w, WIDX_SLOPE_UP_STEEP)
                && w->widgets[WIDX_SLOPE_UP_STEEP].image.GetIndex() == SPR_RIDE_CONSTRUCTION_VERTICAL_DROP
                && w->widgets[WIDX_SLOPE_UP_STEEP].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_SLOPE_UP_STEEP);
            }
            break;
        case TRACK_SLOPE_DOWN_25:
            if (!WidgetIsDisabled(*w, WIDX_SLOPE_DOWN_STEEP)
                && w->widgets[WIDX_SLOPE_DOWN_STEEP].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_SLOPE_DOWN_STEEP);
            }
            break;
        case TRACK_SLOPE_NONE:
            if (!WidgetIsDisabled(*w, WIDX_SLOPE_DOWN) && w->widgets[WIDX_SLOPE_DOWN].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_SLOPE_DOWN);
            }
            else if (
                IsTrackEnabled(TRACK_SLOPE_VERTICAL)
                && w->widgets[WIDX_SLOPE_DOWN_STEEP].image.GetIndex() == SPR_RIDE_CONSTRUCTION_VERTICAL_RISE)
            {
                return;
            }
            else if (
                !WidgetIsDisabled(*w, WIDX_SLOPE_DOWN_STEEP)
                && w->widgets[WIDX_SLOPE_DOWN_STEEP].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_SLOPE_DOWN_STEEP);
            }
            else
            {
                return;
            }
            break;
        case TRACK_SLOPE_UP_25:
            if (!WidgetIsDisabled(*w, WIDX_LEVEL) && w->widgets[WIDX_LEVEL].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_LEVEL);
            }
            else if (!WidgetIsDisabled(*w, WIDX_SLOPE_DOWN) && w->widgets[WIDX_SLOPE_DOWN].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_SLOPE_DOWN);
            }
            else if (
                !WidgetIsDisabled(*w, WIDX_SLOPE_DOWN_STEEP)
                && w->widgets[WIDX_SLOPE_DOWN_STEEP].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_SLOPE_DOWN_STEEP);
            }
            else
            {
                return;
            }
            break;
        case TRACK_SLOPE_UP_60:
            if (!WidgetIsDisabled(*w, WIDX_SLOPE_UP) && w->widgets[WIDX_SLOPE_UP].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_SLOPE_UP);
            }
            else if (!WidgetIsDisabled(*w, WIDX_LEVEL) && w->widgets[WIDX_LEVEL].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_LEVEL);
            }
            else if (!WidgetIsDisabled(*w, WIDX_SLOPE_DOWN) && w->widgets[WIDX_SLOPE_DOWN].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_SLOPE_DOWN);
            }
            else if (
                IsTrackEnabled(TRACK_SLOPE_VERTICAL)
                && w->widgets[WIDX_SLOPE_DOWN_STEEP].image.GetIndex() == SPR_RIDE_CONSTRUCTION_VERTICAL_RISE)
            {
                return;
            }
            else if (
                !WidgetIsDisabled(*w, WIDX_SLOPE_DOWN_STEEP)
                && w->widgets[WIDX_SLOPE_DOWN_STEEP].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_SLOPE_DOWN_STEEP);
            }
            else
            {
                return;
            }
            break;
        case TRACK_SLOPE_UP_90:
            if (IsTrackEnabled(TRACK_SLOPE_VERTICAL) && !WidgetIsDisabled(*w, WIDX_SLOPE_UP_STEEP)
                && w->widgets[WIDX_SLOPE_DOWN_STEEP].image.GetIndex() == SPR_RIDE_CONSTRUCTION_VERTICAL_RISE
                && w->widgets[WIDX_SLOPE_DOWN_STEEP].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_SLOPE_UP_STEEP);
            }
            break;
        default:
            return;
    }
}

void WindowRideConstructionKeyboardShortcutSlopeUp()
{
    WindowBase* w = WindowFindByClass(WindowClass::RideConstruction);
    if (w == nullptr || WidgetIsDisabled(*w, WIDX_STRAIGHT) || w->widgets[WIDX_STRAIGHT].type == WindowWidgetType::Empty)
    {
        return;
    }

    switch (_currentTrackSlopeEnd)
    {
        case TRACK_SLOPE_UP_60:
            if (IsTrackEnabled(TRACK_SLOPE_VERTICAL) && !WidgetIsDisabled(*w, WIDX_SLOPE_DOWN_STEEP)
                && w->widgets[WIDX_SLOPE_DOWN_STEEP].image.GetIndex() == SPR_RIDE_CONSTRUCTION_VERTICAL_RISE
                && w->widgets[WIDX_SLOPE_DOWN_STEEP].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_SLOPE_DOWN_STEEP);
            }
            break;
        case TRACK_SLOPE_UP_25:
            if (!WidgetIsDisabled(*w, WIDX_SLOPE_UP_STEEP) && w->widgets[WIDX_SLOPE_UP_STEEP].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_SLOPE_UP_STEEP);
            }
            break;
        case TRACK_SLOPE_NONE:
            if (!WidgetIsDisabled(*w, WIDX_SLOPE_UP) && w->widgets[WIDX_SLOPE_UP].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_SLOPE_UP);
            }
            else if (
                IsTrackEnabled(TRACK_SLOPE_VERTICAL)
                && w->widgets[WIDX_SLOPE_UP_STEEP].image.GetIndex() == SPR_RIDE_CONSTRUCTION_VERTICAL_DROP)
            {
                return;
            }
            else if (
                !WidgetIsDisabled(*w, WIDX_SLOPE_UP_STEEP) && w->widgets[WIDX_SLOPE_UP_STEEP].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_SLOPE_UP_STEEP);
            }
            else
            {
                return;
            }
            break;
        case TRACK_SLOPE_DOWN_25:
            if (!WidgetIsDisabled(*w, WIDX_LEVEL) && w->widgets[WIDX_LEVEL].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_LEVEL);
            }
            else if (!WidgetIsDisabled(*w, WIDX_SLOPE_UP) && w->widgets[WIDX_SLOPE_UP].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_SLOPE_UP);
            }
            else if (
                !WidgetIsDisabled(*w, WIDX_SLOPE_UP_STEEP) && w->widgets[WIDX_SLOPE_UP_STEEP].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_SLOPE_UP_STEEP);
            }
            else
            {
                return;
            }
            break;
        case TRACK_SLOPE_DOWN_60:
            if (!WidgetIsDisabled(*w, WIDX_SLOPE_DOWN) && w->widgets[WIDX_SLOPE_DOWN].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_SLOPE_DOWN);
            }
            else if (!WidgetIsDisabled(*w, WIDX_LEVEL) && w->widgets[WIDX_LEVEL].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_LEVEL);
            }
            else if (!WidgetIsDisabled(*w, WIDX_SLOPE_UP) && w->widgets[WIDX_SLOPE_UP].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_SLOPE_UP);
            }
            else if (
                IsTrackEnabled(TRACK_SLOPE_VERTICAL)
                && w->widgets[WIDX_SLOPE_UP_STEEP].image.GetIndex() == SPR_RIDE_CONSTRUCTION_VERTICAL_DROP)
            {
                return;
            }
            else if (
                !WidgetIsDisabled(*w, WIDX_SLOPE_UP_STEEP) && w->widgets[WIDX_SLOPE_UP_STEEP].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_SLOPE_UP_STEEP);
            }
            else
            {
                return;
            }
            break;
        case TRACK_SLOPE_DOWN_90:
            if (IsTrackEnabled(TRACK_SLOPE_VERTICAL) && !WidgetIsDisabled(*w, WIDX_SLOPE_DOWN_STEEP)
                && w->widgets[WIDX_SLOPE_UP_STEEP].image.GetIndex() == SPR_RIDE_CONSTRUCTION_VERTICAL_DROP
                && w->widgets[WIDX_SLOPE_DOWN_STEEP].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_SLOPE_DOWN_STEEP);
            }
            break;
        default:
            return;
    }
}

void WindowRideConstructionKeyboardShortcutChainLiftToggle()
{
    WindowBase* w = WindowFindByClass(WindowClass::RideConstruction);
    if (w == nullptr || WidgetIsDisabled(*w, WIDX_CHAIN_LIFT) || w->widgets[WIDX_CHAIN_LIFT].type == WindowWidgetType::Empty)
    {
        return;
    }

    WindowEventMouseDownCall(w, WIDX_CHAIN_LIFT);
}

void WindowRideConstructionKeyboardShortcutBankLeft()
{
    WindowBase* w = WindowFindByClass(WindowClass::RideConstruction);
    if (w == nullptr || WidgetIsDisabled(*w, WIDX_BANK_STRAIGHT)
        || w->widgets[WIDX_BANK_STRAIGHT].type == WindowWidgetType::Empty)
    {
        return;
    }

    switch (_currentTrackBankEnd)
    {
        case TRACK_BANK_NONE:
            if (!WidgetIsDisabled(*w, WIDX_BANK_LEFT) && w->widgets[WIDX_BANK_LEFT].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_BANK_LEFT);
            }
            break;
        case TRACK_BANK_RIGHT:
            if (!WidgetIsDisabled(*w, WIDX_BANK_STRAIGHT) && w->widgets[WIDX_BANK_STRAIGHT].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_BANK_STRAIGHT);
            }
            else if (!WidgetIsDisabled(*w, WIDX_BANK_LEFT) && w->widgets[WIDX_BANK_LEFT].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_BANK_LEFT);
            }
            else
            {
                return;
            }
            break;
        default:
            return;
    }
}

void WindowRideConstructionKeyboardShortcutBankRight()
{
    WindowBase* w = WindowFindByClass(WindowClass::RideConstruction);
    if (w == nullptr || WidgetIsDisabled(*w, WIDX_BANK_STRAIGHT)
        || w->widgets[WIDX_BANK_STRAIGHT].type == WindowWidgetType::Empty)
    {
        return;
    }

    switch (_currentTrackBankEnd)
    {
        case TRACK_BANK_NONE:
            if (!WidgetIsDisabled(*w, WIDX_BANK_RIGHT) && w->widgets[WIDX_BANK_RIGHT].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_BANK_RIGHT);
            }
            break;
        case TRACK_BANK_LEFT:
            if (!WidgetIsDisabled(*w, WIDX_BANK_STRAIGHT) && w->widgets[WIDX_BANK_STRAIGHT].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_BANK_STRAIGHT);
            }
            else if (!WidgetIsDisabled(*w, WIDX_BANK_RIGHT) && w->widgets[WIDX_BANK_RIGHT].type != WindowWidgetType::Empty)
            {
                WindowEventMouseDownCall(w, WIDX_BANK_RIGHT);
            }
            else
            {
                return;
            }
            break;
        default:
            return;
    }
}

void WindowRideConstructionKeyboardShortcutPreviousTrack()
{
    WindowBase* w = WindowFindByClass(WindowClass::RideConstruction);
    if (w == nullptr || WidgetIsDisabled(*w, WIDX_PREVIOUS_SECTION)
        || w->widgets[WIDX_PREVIOUS_SECTION].type == WindowWidgetType::Empty)
    {
        return;
    }

    WindowEventMouseUpCall(w, WIDX_PREVIOUS_SECTION);
}

void WindowRideConstructionKeyboardShortcutNextTrack()
{
    WindowBase* w = WindowFindByClass(WindowClass::RideConstruction);
    if (w == nullptr || WidgetIsDisabled(*w, WIDX_NEXT_SECTION)
        || w->widgets[WIDX_NEXT_SECTION].type == WindowWidgetType::Empty)
    {
        return;
    }

    WindowEventMouseUpCall(w, WIDX_NEXT_SECTION);
}

void WindowRideConstructionKeyboardShortcutBuildCurrent()
{
    WindowBase* w = WindowFindByClass(WindowClass::RideConstruction);
    if (w == nullptr || WidgetIsDisabled(*w, WIDX_CONSTRUCT) || w->widgets[WIDX_CONSTRUCT].type == WindowWidgetType::Empty)
    {
        return;
    }

    WindowEventMouseUpCall(w, WIDX_CONSTRUCT);
}

void WindowRideConstructionKeyboardShortcutDemolishCurrent()
{
    WindowBase* w = WindowFindByClass(WindowClass::RideConstruction);
    if (w == nullptr || WidgetIsDisabled(*w, WIDX_DEMOLISH) || w->widgets[WIDX_DEMOLISH].type == WindowWidgetType::Empty)
    {
        return;
    }

    WindowEventMouseUpCall(w, WIDX_DEMOLISH);
}

static void WindowRideConstructionMouseUpDemolishNextPiece(const CoordsXYZD& piecePos, int32_t type)
{
    if (_gotoStartPlacementMode)
    {
        _currentTrackBegin.z = Floor2(piecePos.z, COORDS_Z_STEP);
        _rideConstructionState = RideConstructionState::Front;
        _currentTrackSelectionFlags = 0;
        _currentTrackPieceDirection = piecePos.direction & 3;
        auto savedCurrentTrackCurve = _currentTrackCurve;
        int32_t savedPreviousTrackSlopeEnd = _previousTrackSlopeEnd;
        int32_t savedCurrentTrackSlopeEnd = _currentTrackSlopeEnd;
        int32_t savedPreviousTrackBankEnd = _previousTrackBankEnd;
        int32_t savedCurrentTrackBankEnd = _currentTrackBankEnd;
        int32_t savedCurrentTrackAlternative = _currentTrackAlternative;
        int32_t savedCurrentTrackLiftHill = _currentTrackLiftHill;
        RideConstructionSetDefaultNextPiece();
        WindowRideConstructionUpdateActiveElements();
        auto ride = GetRide(_currentRideIndex);
        if (!RideTryGetOriginElement(*ride, nullptr))
        {
            RideInitialiseConstructionWindow(*ride);
            _currentTrackPieceDirection = piecePos.direction & 3;
            if (!(savedCurrentTrackCurve & RideConstructionSpecialPieceSelected))
            {
                _currentTrackCurve = savedCurrentTrackCurve;
                _previousTrackSlopeEnd = savedPreviousTrackSlopeEnd;
                _currentTrackSlopeEnd = savedCurrentTrackSlopeEnd;
                _previousTrackBankEnd = savedPreviousTrackBankEnd;
                _currentTrackBankEnd = savedCurrentTrackBankEnd;
                _currentTrackAlternative = savedCurrentTrackAlternative;
                _currentTrackLiftHill = savedCurrentTrackLiftHill;
                WindowRideConstructionUpdateActiveElements();
            }
        }
    }
    else
    {
        if (_rideConstructionState2 == RideConstructionState::Selected
            || _rideConstructionState2 == RideConstructionState::Front)
        {
            if (type == TrackElemType::MiddleStation || type == TrackElemType::BeginStation)
            {
                type = TrackElemType::EndStation;
            }
        }
        if (_rideConstructionState2 == RideConstructionState::Back)
        {
            if (type == TrackElemType::MiddleStation)
            {
                type = TrackElemType::BeginStation;
            }
        }
        if (NetworkGetMode() == NETWORK_MODE_CLIENT)
        {
            // rideConstructionState needs to be set again to the proper value, this only affects the client
            _rideConstructionState = RideConstructionState::Selected;
        }
        _currentTrackBegin = piecePos;
        _currentTrackPieceDirection = piecePos.direction;
        _currentTrackPieceType = type;
        _currentTrackSelectionFlags = 0;
        if (_rideConstructionState2 == RideConstructionState::Front)
        {
            RideSelectNextSection();
        }
        else if (_rideConstructionState2 == RideConstructionState::Back)
        {
            RideSelectPreviousSection();
        }
        WindowRideConstructionUpdateActiveElements();
    }
}
