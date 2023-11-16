/*****************************************************************************
 * Copyright (c) 2014-2023 OpenRCT2 developers
 *
 * For a complete list of all authors, please refer to contributors.md
 * Interested in contributing? Visit https://github.com/OpenRCT2/OpenRCT2
 *
 * OpenRCT2 is licensed under the GNU General Public License version 3.
 *****************************************************************************/

#include "../../common.h"
#include "../../config/Config.h"
#include "../../interface/Viewport.h"
#include "../../paint/Paint.h"
#include "../../paint/Supports.h"
#include "../Ride.h"
#include "../Track.h"
#include "../TrackPaint.h"
#include "../Vehicle.h"
#include "../VehiclePaint.h"

enum
{
    SPR_LAUNCHED_FREEFALL_TOWER_BASE = 14564,
    SPR_LAUNCHED_FREEFALL_TOWER_SEGMENT = 14565,
    SPR_LAUNCHED_FREEFALL_TOWER_SEGMENT_TOP = 14566,
};

/**
 *
 *  rct2: 0x006D5FAB
 */
void VehicleVisualLaunchedFreefall(
    PaintSession& session, int32_t x, int32_t imageDirection, int32_t y, int32_t z, const Vehicle* vehicle,
    const CarEntry* carEntry)
{
    auto imageFlags = ImageId(0, vehicle->colours.Body, vehicle->colours.Trim);
    if (vehicle->IsGhost())
    {
        imageFlags = ConstructionMarker;
    }

    // Draw back:
    int32_t baseImage_id = carEntry->base_image_id + ((vehicle->restraints_position / 64) * 2);
    auto image_id = imageFlags.WithIndex(baseImage_id + 2);
    PaintAddImageAsParent(session, image_id, { 0, 0, z }, { { -11, -11, z + 1 }, { 2, 2, 41 } });

    // Draw front:
    image_id = imageFlags.WithIndex(baseImage_id + 1);
    PaintAddImageAsParent(session, image_id, { 0, 0, z }, { { -5, -5, z + 1 }, { 16, 16, 41 } });

    // Draw peeps:
    if (session.DPI.zoom_level < ZoomLevel{ 2 } && vehicle->num_peeps > 0 && !vehicle->IsGhost())
    {
        baseImage_id = carEntry->base_image_id + 9;
        if ((vehicle->restraints_position / 64) == 3)
        {
            baseImage_id += 2; // Draw peeps sitting without transparent area between them for restraints
        }
        auto directionOffset = OpenRCT2::Entity::Yaw::YawTo4(imageDirection);
        image_id = ImageId(
            baseImage_id + (((directionOffset + 0) & 3) * 3), vehicle->peep_tshirt_colours[0], vehicle->peep_tshirt_colours[1]);
        PaintAddImageAsChild(session, image_id, { 0, 0, z }, { { -5, -5, z + 1 }, { 16, 16, 41 } });
        if (vehicle->num_peeps > 2)
        {
            image_id = ImageId(
                baseImage_id + (((directionOffset + 1) & 3) * 3), vehicle->peep_tshirt_colours[2],
                vehicle->peep_tshirt_colours[3]);
            PaintAddImageAsChild(session, image_id, { 0, 0, z }, { { -5, -5, z + 1 }, { 16, 16, 41 } });
        }
        if (vehicle->num_peeps > 4)
        {
            image_id = ImageId(
                baseImage_id + (((directionOffset + 2) & 3) * 3), vehicle->peep_tshirt_colours[4],
                vehicle->peep_tshirt_colours[5]);
            PaintAddImageAsChild(session, image_id, { 0, 0, z }, { { -5, -5, z + 1 }, { 16, 16, 41 } });
        }
        if (vehicle->num_peeps > 6)
        {
            image_id = ImageId(
                baseImage_id + (((directionOffset + 3) & 3) * 3), vehicle->peep_tshirt_colours[6],
                vehicle->peep_tshirt_colours[7]);
            PaintAddImageAsChild(session, image_id, { 0, 0, z }, { { -5, -5, z + 1 }, { 16, 16, 41 } });
        }
    }

    assert(carEntry->effect_visual == 1);
}

/** rct2: 0x006FD1F8 */
static void PaintLaunchedFreefallBase(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    trackSequence = track_map_3x3[direction][trackSequence];

    int32_t edges = edges_3x3[trackSequence];

    WoodenASupportsPaintSetup(session, (direction & 1), 0, height, session.TrackColours[SCHEME_MISC]);

    const StationObject* stationObject = ride.GetStationObject();

    TrackPaintUtilPaintFloor(session, edges, session.TrackColours[SCHEME_SUPPORTS], height, floorSpritesMetal, stationObject);

    TrackPaintUtilPaintFences(
        session, edges, session.MapPosition, trackElement, ride, session.TrackColours[SCHEME_TRACK], height, fenceSpritesMetal,
        session.CurrentRotation);

    if (trackSequence == 0)
    {
        auto imageId = session.TrackColours[SCHEME_TRACK].WithIndex(SPR_LAUNCHED_FREEFALL_TOWER_BASE);
        PaintAddImageAsParent(session, imageId, { 0, 0, height }, { { 8, 8, height + 3 }, { 2, 2, 27 } });

        height += 32;
        imageId = session.TrackColours[SCHEME_TRACK].WithIndex(SPR_LAUNCHED_FREEFALL_TOWER_SEGMENT);
        PaintAddImageAsParent(session, imageId, { 0, 0, height }, { { 8, 8, height }, { 2, 2, 30 } });

        height += 32;
        imageId = session.TrackColours[SCHEME_TRACK].WithIndex(SPR_LAUNCHED_FREEFALL_TOWER_SEGMENT);
        PaintAddImageAsParent(session, imageId, { 0, 0, height }, { { 8, 8, height }, { 2, 2, 30 } });

        PaintUtilSetVerticalTunnel(session, height + 32);

        height -= 64;
    }

    int32_t blockedSegments = 0;
    switch (trackSequence)
    {
        case 0:
            blockedSegments = SEGMENTS_ALL;
            break;
        case 1:
            blockedSegments = SEGMENT_B8 | SEGMENT_C8 | SEGMENT_B4 | SEGMENT_CC | SEGMENT_BC;
            break;
        case 2:
            blockedSegments = SEGMENT_B4 | SEGMENT_CC | SEGMENT_BC;
            break;
        case 3:
            blockedSegments = SEGMENT_B4 | SEGMENT_CC | SEGMENT_BC | SEGMENT_D4 | SEGMENT_C0;
            break;
        case 4:
            blockedSegments = SEGMENT_B4 | SEGMENT_C8 | SEGMENT_B8;
            break;
        case 5:
            blockedSegments = SEGMENT_BC | SEGMENT_D4 | SEGMENT_C0;
            break;
        case 6:
            blockedSegments = SEGMENT_B4 | SEGMENT_C8 | SEGMENT_B8 | SEGMENT_D0 | SEGMENT_C0;
            break;
        case 7:
            blockedSegments = SEGMENT_B8 | SEGMENT_D0 | SEGMENT_C0 | SEGMENT_D4 | SEGMENT_BC;
            break;
        case 8:
            blockedSegments = SEGMENT_B8 | SEGMENT_D0 | SEGMENT_C0;
            break;
    }
    PaintUtilSetSegmentSupportHeight(session, blockedSegments, 0xFFFF, 0);
    PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL & ~blockedSegments, height + 2, 0x20);
    PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
}

/** rct2: 0x006FD208 */
static void PaintLaunchedFreefallTowerSection(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    if (trackSequence == 1)
    {
        return;
    }

    auto imageId = session.TrackColours[SCHEME_TRACK].WithIndex(SPR_LAUNCHED_FREEFALL_TOWER_SEGMENT);
    PaintAddImageAsParent(session, imageId, { 0, 0, height }, { { 8, 8, height }, { 2, 2, 30 } });

    const TileElement* nextTileElement = reinterpret_cast<const TileElement*>(&trackElement) + 1;
    if (trackElement.IsLastForTile() || trackElement.GetClearanceZ() != nextTileElement->GetBaseZ())
    {
        imageId = session.TrackColours[SCHEME_TRACK].WithIndex(SPR_LAUNCHED_FREEFALL_TOWER_SEGMENT_TOP);
        PaintAddImageAsChild(session, imageId, { 0, 0, height }, { { 8, 8, height }, { 2, 2, 30 } });
    }

    PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);

    PaintUtilSetVerticalTunnel(session, height + 32);
    PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
}

/**
 * rct2: 0x006FD0E8
 */
TRACK_PAINT_FUNCTION GetTrackPaintFunctionLaunchedFreefall(int32_t trackType)
{
    switch (trackType)
    {
        case TrackElemType::TowerBase:
            return PaintLaunchedFreefallBase;

        case TrackElemType::TowerSection:
            return PaintLaunchedFreefallTowerSection;
    }

    return nullptr;
}
