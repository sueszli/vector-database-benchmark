/*****************************************************************************
 * Copyright (c) 2014-2023 OpenRCT2 developers
 *
 * For a complete list of all authors, please refer to contributors.md
 * Interested in contributing? Visit https://github.com/OpenRCT2/OpenRCT2
 *
 * OpenRCT2 is licensed under the GNU General Public License version 3.
 *****************************************************************************/

#include "TrackSetBrakeSpeedAction.h"

#include "../management/Finance.h"

TrackSetBrakeSpeedAction::TrackSetBrakeSpeedAction(const CoordsXYZ& loc, track_type_t trackType, uint8_t brakeSpeed)
    : _loc(loc)
    , _trackType(trackType)
    , _brakeSpeed(brakeSpeed)
{
}

void TrackSetBrakeSpeedAction::AcceptParameters(GameActionParameterVisitor& visitor)
{
    visitor.Visit(_loc);
    visitor.Visit("trackType", _trackType);
    visitor.Visit("brakeSpeed", _brakeSpeed);
}

uint16_t TrackSetBrakeSpeedAction::GetActionFlags() const
{
    return GameAction::GetActionFlags() | GameActions::Flags::AllowWhilePaused;
}

void TrackSetBrakeSpeedAction::Serialise(DataSerialiser& stream)
{
    GameAction::Serialise(stream);
    stream << DS_TAG(_loc) << DS_TAG(_trackType) << DS_TAG(_brakeSpeed);
}

GameActions::Result TrackSetBrakeSpeedAction::Query() const
{
    return QueryExecute(false);
}

GameActions::Result TrackSetBrakeSpeedAction::Execute() const
{
    return QueryExecute(true);
}

GameActions::Result TrackSetBrakeSpeedAction::QueryExecute(bool isExecuting) const
{
    auto res = GameActions::Result();

    res.Position = _loc;
    res.Position.x += 16;
    res.Position.y += 16;
    res.Expenditure = ExpenditureType::RideConstruction;

    if (!LocationValid(_loc))
    {
        return GameActions::Result(GameActions::Status::NotOwned, STR_NONE, STR_NONE);
    }

    TileElement* tileElement = MapGetTrackElementAtOfType(_loc, _trackType);
    if (tileElement == nullptr)
    {
        LOG_WARNING("Invalid game command for setting brakes speed. x = %d, y = %d", _loc.x, _loc.y);
        return GameActions::Result(GameActions::Status::InvalidParameters, STR_NONE, STR_NONE);
    }

    if (isExecuting)
    {
        GetTrackElementOriginAndApplyChanges(
            { _loc, tileElement->GetDirection() }, tileElement->AsTrack()->GetTrackType(), _brakeSpeed, nullptr,
            TRACK_ELEMENT_SET_BRAKE_BOOSTER_SPEED);
    }
    return res;
}
