/*****************************************************************************
 * Copyright (c) 2014-2023 OpenRCT2 developers
 *
 * For a complete list of all authors, please refer to contributors.md
 * Interested in contributing? Visit https://github.com/OpenRCT2/OpenRCT2
 *
 * OpenRCT2 is licensed under the GNU General Public License version 3.
 *****************************************************************************/

#include "../../entity/EntityRegistry.h"
#include "../../interface/Viewport.h"
#include "../../localisation/Localisation.h"
#include "../../paint/Boundbox.h"
#include "../../paint/Paint.h"
#include "../../paint/Supports.h"
#include "../../sprites.h"
#include "../../world/Map.h"
#include "../RideData.h"
#include "../TrackData.h"
#include "../TrackPaint.h"
#include "../Vehicle.h"

static int16_t TopSpinSeatHeightOffset[] = {
    -10, -10, -9, -7, -4, -1, 2,  6,  11, 16, 21, 26, 31, 37, 42, 47, 52, 57, 61, 64, 67, 70, 72, 73,
    73,  73,  72, 70, 67, 64, 61, 57, 52, 47, 42, 37, 31, 26, 21, 16, 11, 6,  2,  -1, -4, -7, -9, -10,
};

/**
 * Can be calculated as Rounddown(34*sin(x)+0.5)
 * where x is in 7.5 deg segments.
 */
static int8_t TopSpinSeatPositionOffset[] = {
    0, 4,  9,  13,  17,  21,  24,  27,  29,  31,  33,  34,  34,  34,  33,  31,  29,  27,  24,  21,  17,  13,  9,  4,
    0, -3, -8, -12, -16, -20, -23, -26, -28, -30, -32, -33, -33, -33, -32, -30, -28, -26, -23, -20, -16, -12, -8, -3,
};

static void PaintTopSpinRiders(
    PaintSession& session, const Vehicle& vehicle, ImageIndex seatImageIndex, const CoordsXYZ& seatCoords,
    const BoundBoxXYZ& bb)
{
    if (session.DPI.zoom_level >= ZoomLevel{ 2 })
        return;

    for (int i = 0; i < 4; i++)
    {
        auto peepIndex = i * 2;
        if (vehicle.num_peeps > peepIndex)
        {
            auto imageIndex = seatImageIndex + ((i + 1) * 76);
            auto imageId = ImageId(
                imageIndex, vehicle.peep_tshirt_colours[peepIndex], vehicle.peep_tshirt_colours[peepIndex + 1]);
            PaintAddImageAsChild(session, imageId, seatCoords, bb);
        }
        else
        {
            break;
        }
    }
}

static void PaintTopSpinSeat(
    PaintSession& session, const Ride& ride, const RideObjectEntry& rideEntry, const Vehicle* vehicle, Direction direction,
    uint32_t armRotation, uint32_t seatRotation, const CoordsXYZ& offset, const BoundBoxXYZ& bb)
{
    if (armRotation >= std::size(TopSpinSeatHeightOffset))
        return;

    const auto& carEntry = rideEntry.Cars[0];

    ImageIndex seatImageIndex;
    if (vehicle != nullptr && vehicle->restraints_position >= 64)
    {
        // Open Restraints
        seatImageIndex = carEntry.base_image_id + 64;
        seatImageIndex += (vehicle->restraints_position - 64) >> 6;
        seatImageIndex += direction * 3;
    }
    else
    {
        // Var_20 Rotation of seats
        seatImageIndex = carEntry.base_image_id;
        seatImageIndex += direction * 16;
        seatImageIndex += seatRotation;
    }

    auto seatCoords = offset;
    seatCoords.z += TopSpinSeatHeightOffset[armRotation];
    switch (direction)
    {
        case 0:
            seatCoords.x -= TopSpinSeatPositionOffset[armRotation];
            break;
        case 1:
            seatCoords.y += TopSpinSeatPositionOffset[armRotation];
            break;
        case 2:
            seatCoords.x += TopSpinSeatPositionOffset[armRotation];
            break;
        case 3:
            seatCoords.y -= TopSpinSeatPositionOffset[armRotation];
            break;
    }

    auto imageFlags = session.TrackColours[SCHEME_MISC];
    auto imageTemplate = ImageId(0, ride.vehicle_colours[0].Body, ride.vehicle_colours[0].Trim);
    if (imageFlags != TrackGhost)
    {
        imageTemplate = imageFlags;
    }

    PaintAddImageAsChild(session, imageTemplate.WithIndex(seatImageIndex), seatCoords, bb);
    if (vehicle != nullptr)
    {
        PaintTopSpinRiders(session, *vehicle, seatImageIndex, seatCoords, bb);
    }
}

static void PaintTopSpinVehicle(
    PaintSession& session, int32_t al, int32_t cl, const Ride& ride, uint8_t direction, int32_t height,
    const TrackElement& tileElement)
{
    const auto* rideEntry = GetRideEntryByIndex(ride.subtype);
    if (rideEntry == nullptr)
        return;

    const auto& carEntry = rideEntry->Cars[0];

    height += 3;
    uint8_t seatRotation = 0;
    uint8_t armRotation = 0;
    auto* vehicle = GetEntity<Vehicle>(ride.vehicles[0]);
    if (ride.lifecycle_flags & RIDE_LIFECYCLE_ON_TRACK && vehicle != nullptr)
    {
        session.InteractionType = ViewportInteractionItem::Entity;
        session.CurrentlyDrawnEntity = vehicle;

        armRotation = vehicle->Pitch;
        seatRotation = vehicle->bank_rotation;
    }

    int32_t armImageOffset = armRotation;
    if (direction & 2)
    {
        armImageOffset = -armImageOffset;
        if (armImageOffset != 0)
            armImageOffset += 48;
    }

    CoordsXYZ offset = { al, cl, height };
    BoundBoxXYZ bb = { { al + 16, cl + 16, height }, { 24, 24, 90 } };

    auto imageFlags = session.TrackColours[SCHEME_MISC];
    auto supportImageTemplate = ImageId(0, ride.track_colour[0].main, ride.track_colour[0].supports);
    auto armImageTemplate = ImageId(0, ride.track_colour[0].main, ride.track_colour[0].additional);
    if (imageFlags != TrackGhost)
    {
        supportImageTemplate = imageFlags;
        armImageTemplate = supportImageTemplate;
    }

    // Left back bottom support
    auto imageIndex = carEntry.base_image_id + 572 + ((direction & 1) << 1);
    PaintAddImageAsParent(session, supportImageTemplate.WithIndex(imageIndex), offset, bb);

    // Left hand arm
    imageIndex = carEntry.base_image_id + 380 + armImageOffset + ((direction & 1) * 48);
    PaintAddImageAsChild(session, armImageTemplate.WithIndex(imageIndex), offset, bb);

    // Seat
    PaintTopSpinSeat(session, ride, *rideEntry, vehicle, direction, armRotation, seatRotation, offset, bb);

    // Right hand arm
    imageIndex = carEntry.base_image_id + 476 + armImageOffset + ((direction & 1) * 48);
    PaintAddImageAsChild(session, armImageTemplate.WithIndex(imageIndex), offset, bb);

    // Right back bottom support
    imageIndex = carEntry.base_image_id + 573 + ((direction & 1) << 1);
    PaintAddImageAsChild(session, supportImageTemplate.WithIndex(imageIndex), offset, bb);

    session.CurrentlyDrawnEntity = nullptr;
    session.InteractionType = ViewportInteractionItem::Ride;
}

static void PaintTopSpin(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    trackSequence = track_map_3x3[direction][trackSequence];

    int32_t edges = edges_3x3[trackSequence];

    WoodenASupportsPaintSetup(session, direction & 1, 0, height, session.TrackColours[SCHEME_MISC]);

    const StationObject* stationObject = ride.GetStationObject();

    TrackPaintUtilPaintFloor(session, edges, session.TrackColours[SCHEME_TRACK], height, floorSpritesCork, stationObject);

    TrackPaintUtilPaintFences(
        session, edges, session.MapPosition, trackElement, ride, session.TrackColours[SCHEME_MISC], height, fenceSpritesRope,
        session.CurrentRotation);

    switch (trackSequence)
    {
        case 1:
            PaintTopSpinVehicle(session, 32, 32, ride, direction, height, trackElement);
            break;
        case 3:
            PaintTopSpinVehicle(session, 32, -32, ride, direction, height, trackElement);
            break;
        case 5:
            PaintTopSpinVehicle(session, 0, -32, ride, direction, height, trackElement);
            break;
        case 6:
            PaintTopSpinVehicle(session, -32, 32, ride, direction, height, trackElement);
            break;
        case 7:
            PaintTopSpinVehicle(session, -32, -32, ride, direction, height, trackElement);
            break;
        case 8:
            PaintTopSpinVehicle(session, -32, 0, ride, direction, height, trackElement);
            break;
    }

    int32_t cornerSegments = 0;
    switch (trackSequence)
    {
        case 1:
            // top
            cornerSegments = SEGMENT_B4 | SEGMENT_C8 | SEGMENT_CC;
            break;
        case 3:
            // right
            cornerSegments = SEGMENT_CC | SEGMENT_BC | SEGMENT_D4;
            break;
        case 6:
            // left
            cornerSegments = SEGMENT_C8 | SEGMENT_B8 | SEGMENT_D0;
            break;
        case 7:
            // bottom
            cornerSegments = SEGMENT_D0 | SEGMENT_C0 | SEGMENT_D4;
            break;
    }

    PaintUtilSetSegmentSupportHeight(session, cornerSegments, height + 2, 0x20);
    PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL & ~cornerSegments, 0xFFFF, 0);
    PaintUtilSetGeneralSupportHeight(session, height + 112, 0x20);
}

TRACK_PAINT_FUNCTION GetTrackPaintFunctionTopspin(int32_t trackType)
{
    if (trackType != TrackElemType::FlatTrack3x3)
    {
        return nullptr;
    }

    return PaintTopSpin;
}
