/*****************************************************************************
 * Copyright (c) 2014-2023 OpenRCT2 developers
 *
 * For a complete list of all authors, please refer to contributors.md
 * Interested in contributing? Visit https://github.com/OpenRCT2/OpenRCT2
 *
 * OpenRCT2 is licensed under the GNU General Public License version 3.
 *****************************************************************************/

#include "../Paint.h"

#include "../../Context.h"
#include "../../Game.h"
#include "../../config/Config.h"
#include "../../core/Numerics.hpp"
#include "../../entity/PatrolArea.h"
#include "../../interface/Viewport.h"
#include "../../localisation/Formatter.h"
#include "../../localisation/Formatting.h"
#include "../../localisation/Localisation.h"
#include "../../object/FootpathObject.h"
#include "../../object/FootpathRailingsObject.h"
#include "../../object/FootpathSurfaceObject.h"
#include "../../object/PathAdditionEntry.h"
#include "../../profiling/Profiling.h"
#include "../../ride/Ride.h"
#include "../../ride/Track.h"
#include "../../ride/TrackDesign.h"
#include "../../ride/TrackPaint.h"
#include "../../world/Footpath.h"
#include "../../world/Map.h"
#include "../../world/Scenery.h"
#include "../../world/Surface.h"
#include "../../world/TileInspector.h"
#include "../Boundbox.h"
#include "../Paint.SessionFlags.h"
#include "../Supports.h"
#include "Paint.PathAddition.h"
#include "Paint.Surface.h"
#include "Paint.TileElement.h"

using namespace OpenRCT2;

bool gPaintWidePathsAsGhost = false;

const uint8_t PathSlopeToLandSlope[] = {
    TILE_ELEMENT_SLOPE_SW_SIDE_UP,
    TILE_ELEMENT_SLOPE_NW_SIDE_UP,
    TILE_ELEMENT_SLOPE_NE_SIDE_UP,
    TILE_ELEMENT_SLOPE_SE_SIDE_UP,
};

static constexpr uint8_t Byte98D6E0[] = {
    0, 1, 2, 3, 4, 5, 6,  7,  8, 9,  10, 11, 12, 13, 14, 15, 0, 1, 2, 20, 4, 5, 6, 22, 8, 9, 10, 26, 12, 13, 14, 36,
    0, 1, 2, 3, 4, 5, 21, 23, 8, 9,  10, 11, 12, 13, 33, 37, 0, 1, 2, 3,  4, 5, 6, 24, 8, 9, 10, 11, 12, 13, 14, 38,
    0, 1, 2, 3, 4, 5, 6,  7,  8, 9,  10, 11, 29, 30, 34, 39, 0, 1, 2, 3,  4, 5, 6, 7,  8, 9, 10, 11, 12, 13, 14, 40,
    0, 1, 2, 3, 4, 5, 6,  7,  8, 9,  10, 11, 12, 13, 35, 41, 0, 1, 2, 3,  4, 5, 6, 7,  8, 9, 10, 11, 12, 13, 14, 42,
    0, 1, 2, 3, 4, 5, 6,  7,  8, 25, 10, 27, 12, 31, 14, 43, 0, 1, 2, 3,  4, 5, 6, 7,  8, 9, 10, 28, 12, 13, 14, 44,
    0, 1, 2, 3, 4, 5, 6,  7,  8, 9,  10, 11, 12, 13, 14, 45, 0, 1, 2, 3,  4, 5, 6, 7,  8, 9, 10, 11, 12, 13, 14, 46,
    0, 1, 2, 3, 4, 5, 6,  7,  8, 9,  10, 11, 12, 32, 14, 47, 0, 1, 2, 3,  4, 5, 6, 7,  8, 9, 10, 11, 12, 13, 14, 48,
    0, 1, 2, 3, 4, 5, 6,  7,  8, 9,  10, 11, 12, 13, 14, 49, 0, 1, 2, 3,  4, 5, 6, 7,  8, 9, 10, 11, 12, 13, 14, 50,
};

// clang-format off
static constexpr BoundBoxXY stru_98D804[] = {
    { { 3, 3 }, { 26, 26 } },
    { { 0, 3 }, { 29, 26 } },
    { { 3, 3 }, { 26, 29 } },
    { { 0, 3 }, { 29, 29 } },
    { { 3, 3 }, { 29, 26 } },
    { { 0, 3 }, { 32, 26 } },
    { { 3, 3 }, { 29, 29 } },
    { { 0, 3 }, { 32, 29 } },
    { { 3, 0 }, { 26, 29 } },
    { { 0, 0 }, { 29, 29 } },
    { { 3, 0 }, { 26, 32 } },
    { { 0, 0 }, { 29, 32 } },
    { { 3, 0 }, { 29, 29 } },
    { { 0, 0 }, { 32, 29 } },
    { { 3, 0 }, { 29, 32 } },
    { { 0, 0 }, { 32, 32 } },
};

static constexpr uint8_t Byte98D8A4[] = {
    0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0
};
// clang-format on

void PathPaintBoxSupport(
    PaintSession& session, const PathElement& pathElement, int32_t height, const FootpathPaintInfo& pathPaintInfo,
    bool hasSupports, ImageId imageTemplate, ImageId sceneryImageTemplate);
void PathPaintPoleSupport(
    PaintSession& session, const PathElement& pathElement, int16_t height, const FootpathPaintInfo& pathPaintInfo,
    bool hasSupports, ImageId imageTemplate, ImageId sceneryImageTemplate);

/**
 * rct2: 0x006A4101
 * @param tile_element (esi)
 */
static void PathPaintFencesAndQueueBanners(
    PaintSession& session, const PathElement& pathElement, uint16_t height, uint32_t connectedEdges, bool hasSupports,
    const FootpathPaintInfo& pathPaintInfo, ImageId imageTemplate)
{
    PROFILED_FUNCTION();

    auto imageId = imageTemplate.WithIndex(pathPaintInfo.RailingsImageId);
    if (pathElement.IsQueue())
    {
        if (pathElement.IsSloped())
        {
            switch ((pathElement.GetSlopeDirection() + session.CurrentRotation) & FOOTPATH_PROPERTIES_SLOPE_DIRECTION_MASK)
            {
                case 0:
                    PaintAddImageAsParent(
                        session, imageId.WithIndexOffset(22), { 0, 4, height }, { { 0, 4, height + 2 }, { 32, 1, 23 } });
                    PaintAddImageAsParent(
                        session, imageId.WithIndexOffset(22), { 0, 28, height }, { { 0, 28, height + 2 }, { 32, 1, 23 } });
                    break;
                case 1:
                    PaintAddImageAsParent(
                        session, imageId.WithIndexOffset(21), { 4, 0, height }, { { 4, 0, height + 2 }, { 1, 32, 23 } });
                    PaintAddImageAsParent(
                        session, imageId.WithIndexOffset(21), { 28, 0, height }, { { 28, 0, height + 2 }, { 1, 32, 23 } });
                    break;
                case 2:
                    PaintAddImageAsParent(
                        session, imageId.WithIndexOffset(23), { 0, 4, height }, { { 0, 4, height + 2 }, { 32, 1, 23 } });
                    PaintAddImageAsParent(
                        session, imageId.WithIndexOffset(23), { 0, 28, height }, { { 0, 28, height + 2 }, { 32, 1, 23 } });
                    break;
                case 3:
                    PaintAddImageAsParent(
                        session, imageId.WithIndexOffset(20), { 4, 0, height }, { { 4, 0, height + 2 }, { 1, 32, 23 } });
                    PaintAddImageAsParent(
                        session, imageId.WithIndexOffset(20), { 28, 0, height }, { { 28, 0, height + 2 }, { 1, 32, 23 } });
                    break;
            }
        }
        else
        {
            const auto pathEdges = connectedEdges & FOOTPATH_PROPERTIES_EDGES_EDGES_MASK;
            switch (pathEdges)
            {
                case 0b0001:
                    PaintAddImageAsParent(
                        session, imageId.WithIndexOffset(17), { 0, 4, height }, { { 0, 4, height + 2 }, { 28, 1, 7 } });
                    PaintAddImageAsParent(
                        session, imageId.WithIndexOffset(17), { 0, 28, height }, { { 0, 28, height + 2 }, { 28, 1, 7 } });
                    break;
                case 0b0010:
                    PaintAddImageAsParent(
                        session, imageId.WithIndexOffset(18), { 4, 0, height }, { { 4, 0, height + 2 }, { 1, 28, 7 } });
                    PaintAddImageAsParent(
                        session, imageId.WithIndexOffset(18), { 28, 0, height }, { { 28, 0, height + 2 }, { 1, 28, 7 } });
                    break;
                case 0b0011:
                    PaintAddImageAsParent(
                        session, imageId.WithIndexOffset(17), { 0, 4, height }, { { 0, 4, height + 2 }, { 28, 1, 7 } });
                    PaintAddImageAsParent(
                        session, imageId.WithIndexOffset(18), { 28, 0, height },
                        { { 28, 4, height + 2 }, { 1, 28, 7 } }); // bound_box_offset_y seems to be a bug
                    PaintAddImageAsParent(
                        session, imageId.WithIndexOffset(25), { 0, 0, height }, { { 0, 28, height + 2 }, { 4, 4, 7 } });
                    break;
                case 0b0100:
                    PaintAddImageAsParent(
                        session, imageId.WithIndexOffset(19), { 0, 4, height }, { { 0, 4, height + 2 }, { 28, 1, 7 } });
                    PaintAddImageAsParent(
                        session, imageId.WithIndexOffset(19), { 0, 28, height }, { { 0, 28, height + 2 }, { 28, 1, 7 } });
                    break;
                case 0b0101:
                    PaintAddImageAsParent(
                        session, imageId.WithIndexOffset(15), { 0, 4, height }, { { 0, 4, height + 2 }, { 32, 1, 7 } });
                    PaintAddImageAsParent(
                        session, imageId.WithIndexOffset(15), { 0, 28, height }, { { 0, 28, height + 2 }, { 32, 1, 7 } });
                    break;
                case 0b0110:
                    PaintAddImageAsParent(
                        session, imageId.WithIndexOffset(18), { 4, 0, height }, { { 4, 0, height + 2 }, { 1, 28, 7 } });
                    PaintAddImageAsParent(
                        session, imageId.WithIndexOffset(19), { 0, 4, height }, { { 0, 4, height + 2 }, { 28, 1, 7 } });
                    PaintAddImageAsParent(
                        session, imageId.WithIndexOffset(26), { 0, 0, height }, { { 28, 28, height + 2 }, { 4, 4, 7 } });
                    break;
                case 0b0111:
                    if (pathElement.HasJunctionRailings())
                    {
                        PaintAddImageAsParent(
                            session, imageId.WithIndexOffset(15), { 0, 4, height }, { { 0, 4, height + 2 }, { 32, 1, 7 } });
                        PaintAddImageAsParent(
                            session, imageId.WithIndexOffset(25), { 0, 0, height }, { { 0, 28, height + 2 }, { 4, 4, 7 } });
                        PaintAddImageAsParent(
                            session, imageId.WithIndexOffset(26), { 0, 0, height }, { { 28, 28, height + 2 }, { 4, 4, 7 } });
                    }
                    break;
                case 0b1000:
                    PaintAddImageAsParent(
                        session, imageId.WithIndexOffset(16), { 4, 0, height }, { { 4, 0, height + 2 }, { 1, 28, 7 } });
                    PaintAddImageAsParent(
                        session, imageId.WithIndexOffset(16), { 28, 0, height }, { { 28, 0, height + 2 }, { 1, 28, 7 } });
                    break;
                case 0b1001:
                    PaintAddImageAsParent(
                        session, imageId.WithIndexOffset(16), { 28, 0, height }, { { 28, 0, height + 2 }, { 1, 28, 7 } });
                    PaintAddImageAsParent(
                        session, imageId.WithIndexOffset(17), { 0, 28, height }, { { 0, 28, height + 2 }, { 28, 1, 7 } });
                    PaintAddImageAsParent(
                        session, imageId.WithIndexOffset(24), { 0, 0, height }, { { 0, 0, height + 2 }, { 4, 4, 7 } });
                    break;
                case 0b1010:
                    PaintAddImageAsParent(
                        session, imageId.WithIndexOffset(14), { 4, 0, height }, { { 4, 0, height + 2 }, { 1, 32, 7 } });
                    PaintAddImageAsParent(
                        session, imageId.WithIndexOffset(14), { 28, 0, height }, { { 28, 0, height + 2 }, { 1, 32, 7 } });
                    break;
                case 0b1011:
                    if (pathElement.HasJunctionRailings())
                    {
                        PaintAddImageAsParent(
                            session, imageId.WithIndexOffset(14), { 28, 0, height }, { { 28, 0, height + 2 }, { 1, 32, 7 } });
                        PaintAddImageAsParent(
                            session, imageId.WithIndexOffset(24), { 0, 0, height }, { { 0, 0, height + 2 }, { 4, 4, 7 } });
                        PaintAddImageAsParent(
                            session, imageId.WithIndexOffset(25), { 0, 0, height }, { { 0, 28, height + 2 }, { 4, 4, 7 } });
                    }
                    break;
                case 0b1100:
                    PaintAddImageAsParent(
                        session, imageId.WithIndexOffset(16), { 4, 0, height }, { { 4, 0, height + 2 }, { 1, 28, 7 } });
                    PaintAddImageAsParent(
                        session, imageId.WithIndexOffset(19), { 0, 28, height },
                        { { 4, 28, height + 2 }, { 28, 1, 7 } }); // bound_box_offset_x seems to be a bug
                    PaintAddImageAsParent(
                        session, imageId.WithIndexOffset(27), { 0, 0, height }, { { 28, 0, height + 2 }, { 4, 4, 7 } });
                    break;
                case 0b1101:
                    if (pathElement.HasJunctionRailings())
                    {
                        PaintAddImageAsParent(
                            session, imageId.WithIndexOffset(15), { 0, 28, height }, { { 0, 28, height + 2 }, { 32, 1, 7 } });
                        PaintAddImageAsParent(
                            session, imageId.WithIndexOffset(24), { 0, 0, height }, { { 0, 0, height + 2 }, { 4, 4, 7 } });
                        PaintAddImageAsParent(
                            session, imageId.WithIndexOffset(27), { 0, 0, height }, { { 28, 0, height + 2 }, { 4, 4, 7 } });
                    }
                    break;
                case 0b1110:
                    if (pathElement.HasJunctionRailings())
                    {
                        PaintAddImageAsParent(
                            session, imageId.WithIndexOffset(14), { 4, 0, height }, { { 4, 0, height + 2 }, { 1, 32, 7 } });
                        PaintAddImageAsParent(
                            session, imageId.WithIndexOffset(26), { 0, 0, height }, { { 28, 28, height + 2 }, { 4, 4, 7 } });
                        PaintAddImageAsParent(
                            session, imageId.WithIndexOffset(27), { 0, 0, height }, { { 28, 0, height + 2 }, { 4, 4, 7 } });
                    }
                    break;
                case 0b1111:
                    if (pathElement.HasJunctionRailings())
                    {
                        PaintAddImageAsParent(
                            session, imageId.WithIndexOffset(24), { 0, 0, height }, { { 0, 0, height + 2 }, { 4, 4, 7 } });
                        PaintAddImageAsParent(
                            session, imageId.WithIndexOffset(25), { 0, 0, height }, { { 0, 28, height + 2 }, { 4, 4, 7 } });
                        PaintAddImageAsParent(
                            session, imageId.WithIndexOffset(26), { 0, 0, height }, { { 28, 28, height + 2 }, { 4, 4, 7 } });
                        PaintAddImageAsParent(
                            session, imageId.WithIndexOffset(27), { 0, 0, height }, { { 28, 0, height + 2 }, { 4, 4, 7 } });
                    }
            }
        }

        if (!pathElement.HasQueueBanner() || (pathPaintInfo.RailingFlags & RAILING_ENTRY_FLAG_NO_QUEUE_BANNER))
        {
            return;
        }

        uint8_t direction = pathElement.GetQueueBannerDirection();
        // Draw ride sign
        session.InteractionType = ViewportInteractionItem::Ride;
        if (pathElement.IsSloped())
        {
            if (pathElement.GetSlopeDirection() == direction)
                height += COORDS_Z_STEP * 2;
        }
        direction += session.CurrentRotation;
        direction &= 3;

        CoordsXYZ boundBoxOffsets = CoordsXYZ(BannerBoundBoxes[direction][0], height + 2);

        imageId = imageId.WithIndexOffset(28 + (direction << 1));

        // Draw pole in the back
        PaintAddImageAsParent(session, imageId, { 0, 0, height }, { boundBoxOffsets, { 1, 1, 21 } });

        // Draw pole in the front and banner
        boundBoxOffsets.x = BannerBoundBoxes[direction][1].x;
        boundBoxOffsets.y = BannerBoundBoxes[direction][1].y;
        imageId = imageId.WithIndexOffset(1);
        PaintAddImageAsParent(session, imageId, { 0, 0, height }, { boundBoxOffsets, { 1, 1, 21 } });

        direction--;
        // If text shown
        auto ride = GetRide(pathElement.GetRideIndex());
        if (direction < 2 && ride != nullptr && !imageTemplate.IsRemap())
        {
            uint16_t scrollingMode = pathPaintInfo.ScrollingMode;
            scrollingMode += direction;

            auto ft = Formatter();

            if (ride->status == RideStatus::Open && !(ride->lifecycle_flags & RIDE_LIFECYCLE_BROKEN_DOWN))
            {
                ft.Add<StringId>(STR_RIDE_ENTRANCE_NAME);
                ride->FormatNameTo(ft);
            }
            else
            {
                ft.Add<StringId>(STR_RIDE_ENTRANCE_CLOSED);
            }
            if (gConfigGeneral.UpperCaseBanners)
            {
                FormatStringToUpper(
                    gCommonStringFormatBuffer, sizeof(gCommonStringFormatBuffer), STR_BANNER_TEXT_FORMAT, ft.Data());
            }
            else
            {
                FormatStringLegacy(
                    gCommonStringFormatBuffer, sizeof(gCommonStringFormatBuffer), STR_BANNER_TEXT_FORMAT, ft.Data());
            }

            uint16_t stringWidth = GfxGetStringWidth(gCommonStringFormatBuffer, FontStyle::Tiny);
            uint16_t scroll = stringWidth > 0 ? (gCurrentTicks / 2) % stringWidth : 0;

            PaintAddImageAsChild(
                session, ScrollingTextSetup(session, STR_BANNER_TEXT_FORMAT, ft, scroll, scrollingMode, COLOUR_BLACK),
                { 0, 0, height + 7 }, { boundBoxOffsets, { 1, 1, 21 } });
        }

        session.InteractionType = ViewportInteractionItem::Footpath;
        if (imageTemplate.IsRemap())
        {
            session.InteractionType = ViewportInteractionItem::None;
        }
        return;
    }

    uint32_t drawnCorners = 0;
    // If the path is not drawn over the supports, then no corner sprites will be drawn (making double-width paths
    // look like connected series of intersections).
    if (pathPaintInfo.RailingFlags & RAILING_ENTRY_FLAG_DRAW_PATH_OVER_SUPPORTS)
    {
        drawnCorners = (connectedEdges & FOOTPATH_PROPERTIES_EDGES_CORNERS_MASK) >> 4;
    }

    auto slopeRailingsSupported = !(pathPaintInfo.SurfaceFlags & FOOTPATH_ENTRY_FLAG_NO_SLOPE_RAILINGS);
    if ((hasSupports || slopeRailingsSupported) && pathElement.IsSloped())
    {
        switch ((pathElement.GetSlopeDirection() + session.CurrentRotation) & FOOTPATH_PROPERTIES_SLOPE_DIRECTION_MASK)
        {
            case 0:
                PaintAddImageAsParent(
                    session, imageId.WithIndexOffset(8), { 0, 4, height }, { { 0, 4, height + 2 }, { 32, 1, 23 } });
                PaintAddImageAsParent(
                    session, imageId.WithIndexOffset(8), { 0, 28, height }, { { 0, 28, height + 2 }, { 32, 1, 23 } });
                break;
            case 1:
                PaintAddImageAsParent(
                    session, imageId.WithIndexOffset(7), { 4, 0, height }, { { 4, 0, height + 2 }, { 1, 32, 23 } });
                PaintAddImageAsParent(
                    session, imageId.WithIndexOffset(7), { 28, 0, height }, { { 28, 0, height + 2 }, { 1, 32, 23 } });
                break;
            case 2:
                PaintAddImageAsParent(
                    session, imageId.WithIndexOffset(9), { 0, 4, height }, { { 0, 4, height + 2 }, { 32, 1, 23 } });
                PaintAddImageAsParent(
                    session, imageId.WithIndexOffset(9), { 0, 28, height }, { { 0, 28, height + 2 }, { 32, 1, 23 } });
                break;
            case 3:
                PaintAddImageAsParent(
                    session, imageId.WithIndexOffset(6), { 4, 0, height }, { { 4, 0, height + 2 }, { 1, 32, 23 } });
                PaintAddImageAsParent(
                    session, imageId.WithIndexOffset(6), { 28, 0, height }, { { 28, 0, height + 2 }, { 1, 32, 23 } });
                break;
        }
    }
    else
    {
        if (!hasSupports)
        {
            return;
        }

        switch (connectedEdges & FOOTPATH_PROPERTIES_EDGES_EDGES_MASK)
        {
            case 0:
                // purposely left empty
                break;
            case 1:
                PaintAddImageAsParent(
                    session, imageId.WithIndexOffset(3), { 0, 4, height }, { { 0, 4, height + 2 }, { 28, 1, 7 } });
                PaintAddImageAsParent(
                    session, imageId.WithIndexOffset(3), { 0, 28, height }, { { 0, 28, height + 2 }, { 28, 1, 7 } });
                break;
            case 2:
                PaintAddImageAsParent(
                    session, imageId.WithIndexOffset(4), { 4, 0, height }, { { 4, 0, height + 2 }, { 1, 28, 7 } });
                PaintAddImageAsParent(
                    session, imageId.WithIndexOffset(4), { 28, 0, height }, { { 28, 0, height + 2 }, { 1, 28, 7 } });
                break;
            case 4:
                PaintAddImageAsParent(
                    session, imageId.WithIndexOffset(5), { 0, 4, height }, { { 0, 4, height + 2 }, { 28, 1, 7 } });
                PaintAddImageAsParent(
                    session, imageId.WithIndexOffset(5), { 0, 28, height }, { { 0, 28, height + 2 }, { 28, 1, 7 } });
                break;
            case 5:
                PaintAddImageAsParent(
                    session, imageId.WithIndexOffset(1), { 0, 4, height }, { { 0, 4, height + 2 }, { 32, 1, 7 } });
                PaintAddImageAsParent(
                    session, imageId.WithIndexOffset(1), { 0, 28, height }, { { 0, 28, height + 2 }, { 32, 1, 7 } });
                break;
            case 8:
                PaintAddImageAsParent(
                    session, imageId.WithIndexOffset(2), { 4, 0, height }, { { 4, 0, height + 2 }, { 1, 28, 7 } });
                PaintAddImageAsParent(
                    session, imageId.WithIndexOffset(2), { 28, 0, height }, { { 28, 0, height + 2 }, { 1, 28, 7 } });
                break;
            case 10:
                PaintAddImageAsParent(
                    session, imageId.WithIndexOffset(0), { 4, 0, height }, { { 4, 0, height + 2 }, { 1, 32, 7 } });
                PaintAddImageAsParent(
                    session, imageId.WithIndexOffset(0), { 28, 0, height }, { { 28, 0, height + 2 }, { 1, 32, 7 } });
                break;

            case 3:
                PaintAddImageAsParent(
                    session, imageId.WithIndexOffset(3), { 0, 4, height }, { { 0, 4, height + 2 }, { 28, 1, 7 } });
                PaintAddImageAsParent(
                    session, imageId.WithIndexOffset(4), { 28, 0, height },
                    { { 28, 4, height + 2 }, { 1, 28, 7 } }); // bound_box_offset_y seems to be a bug
                if (!(drawnCorners & FOOTPATH_CORNER_0))
                {
                    PaintAddImageAsParent(
                        session, imageId.WithIndexOffset(11), { 0, 0, height }, { { 0, 28, height + 2 }, { 4, 4, 7 } });
                }
                break;
            case 6:
                PaintAddImageAsParent(
                    session, imageId.WithIndexOffset(4), { 4, 0, height }, { { 4, 0, height + 2 }, { 1, 28, 7 } });
                PaintAddImageAsParent(
                    session, imageId.WithIndexOffset(5), { 0, 4, height }, { { 0, 4, height + 2 }, { 28, 1, 7 } });
                if (!(drawnCorners & FOOTPATH_CORNER_1))
                {
                    PaintAddImageAsParent(
                        session, imageId.WithIndexOffset(12), { 0, 0, height }, { { 28, 28, height + 2 }, { 4, 4, 7 } });
                }
                break;
            case 9:
                PaintAddImageAsParent(
                    session, imageId.WithIndexOffset(2), { 28, 0, height }, { { 28, 0, height + 2 }, { 1, 28, 7 } });
                PaintAddImageAsParent(
                    session, imageId.WithIndexOffset(3), { 0, 28, height }, { { 0, 28, height + 2 }, { 28, 1, 7 } });
                if (!(drawnCorners & FOOTPATH_CORNER_3))
                {
                    PaintAddImageAsParent(
                        session, imageId.WithIndexOffset(10), { 0, 0, height }, { { 0, 0, height + 2 }, { 4, 4, 7 } });
                }
                break;
            case 12:
                PaintAddImageAsParent(
                    session, imageId.WithIndexOffset(2), { 4, 0, height }, { { 4, 0, height + 2 }, { 1, 28, 7 } });
                PaintAddImageAsParent(
                    session, imageId.WithIndexOffset(5), { 0, 28, height },
                    { { 4, 28, height + 2 }, { 28, 1, 7 } }); // bound_box_offset_x seems to be a bug
                if (!(drawnCorners & FOOTPATH_CORNER_2))
                {
                    PaintAddImageAsParent(
                        session, imageId.WithIndexOffset(13), { 0, 0, height }, { { 28, 0, height + 2 }, { 4, 4, 7 } });
                }
                break;

            case 7:
                PaintAddImageAsParent(
                    session, imageId.WithIndexOffset(1), { 0, 4, height }, { { 0, 4, height + 2 }, { 32, 1, 7 } });
                if (!(drawnCorners & FOOTPATH_CORNER_0))
                {
                    PaintAddImageAsParent(
                        session, imageId.WithIndexOffset(11), { 0, 0, height }, { { 0, 28, height + 2 }, { 4, 4, 7 } });
                }
                if (!(drawnCorners & FOOTPATH_CORNER_1))
                {
                    PaintAddImageAsParent(
                        session, imageId.WithIndexOffset(12), { 0, 0, height }, { { 28, 28, height + 2 }, { 4, 4, 7 } });
                }
                break;
            case 13:
                PaintAddImageAsParent(
                    session, imageId.WithIndexOffset(1), { 0, 28, height }, { { 0, 28, height + 2 }, { 32, 1, 7 } });
                if (!(drawnCorners & FOOTPATH_CORNER_2))
                {
                    PaintAddImageAsParent(
                        session, imageId.WithIndexOffset(13), { 0, 0, height }, { { 28, 0, height + 2 }, { 4, 4, 7 } });
                }
                if (!(drawnCorners & FOOTPATH_CORNER_3))
                {
                    PaintAddImageAsParent(
                        session, imageId.WithIndexOffset(10), { 0, 0, height }, { { 0, 0, height + 2 }, { 4, 4, 7 } });
                }
                break;
            case 14:
                PaintAddImageAsParent(
                    session, imageId.WithIndexOffset(0), { 4, 0, height }, { { 4, 0, height + 2 }, { 1, 32, 7 } });
                if (!(drawnCorners & FOOTPATH_CORNER_1))
                {
                    PaintAddImageAsParent(
                        session, imageId.WithIndexOffset(12), { 0, 0, height }, { { 28, 28, height + 2 }, { 4, 4, 7 } });
                }
                if (!(drawnCorners & FOOTPATH_CORNER_2))
                {
                    PaintAddImageAsParent(
                        session, imageId.WithIndexOffset(13), { 0, 0, height }, { { 28, 0, height + 2 }, { 4, 4, 7 } });
                }
                break;
            case 11:
                PaintAddImageAsParent(
                    session, imageId.WithIndexOffset(0), { 28, 0, height }, { { 28, 0, height + 2 }, { 1, 32, 7 } });
                if (!(drawnCorners & FOOTPATH_CORNER_0))
                {
                    PaintAddImageAsParent(
                        session, imageId.WithIndexOffset(11), { 0, 0, height }, { { 0, 28, height + 2 }, { 4, 4, 7 } });
                }
                if (!(drawnCorners & FOOTPATH_CORNER_3))
                {
                    PaintAddImageAsParent(
                        session, imageId.WithIndexOffset(10), { 0, 0, height }, { { 0, 0, height + 2 }, { 4, 4, 7 } });
                }
                break;

            case 15:
                if (!(drawnCorners & FOOTPATH_CORNER_0))
                {
                    PaintAddImageAsParent(
                        session, imageId.WithIndexOffset(11), { 0, 0, height }, { { 0, 28, height + 2 }, { 4, 4, 7 } });
                }
                if (!(drawnCorners & FOOTPATH_CORNER_1))
                {
                    PaintAddImageAsParent(
                        session, imageId.WithIndexOffset(12), { 0, 0, height }, { { 28, 28, height + 2 }, { 4, 4, 7 } });
                }
                if (!(drawnCorners & FOOTPATH_CORNER_2))
                {
                    PaintAddImageAsParent(
                        session, imageId.WithIndexOffset(13), { 0, 0, height }, { { 28, 0, height + 2 }, { 4, 4, 7 } });
                }
                if (!(drawnCorners & FOOTPATH_CORNER_3))
                {
                    PaintAddImageAsParent(
                        session, imageId.WithIndexOffset(10), { 0, 0, height }, { { 0, 0, height + 2 }, { 4, 4, 7 } });
                }
                break;
        }
    }
}

/**
 * rct2: 0x006A3F61
 * @param pathElement (esp[0])
 * @param connectedEdges (bp) (relative to the camera's rotation)
 * @param height (dx)
 * @param pathPaintInfo (0x00F3EF6C)
 * @param imageFlags (0x00F3EF70)
 * @param sceneryImageFlags (0x00F3EF74)
 */
static void Sub6A3F61(
    PaintSession& session, const PathElement& pathElement, uint16_t connectedEdges, uint16_t height,
    const FootpathPaintInfo& pathPaintInfo, ImageId imageTemplate, ImageId sceneryImageTemplate, bool hasSupports)
{
    // eax --
    // ebx --
    // ecx
    // edx
    // esi --
    // edi --
    // ebp
    // esp: [ esi, ???, 000]

    // Probably drawing benches etc.
    PROFILED_FUNCTION();

    if (session.DPI.zoom_level <= ZoomLevel{ 1 })
    {
        if (!gTrackDesignSaveMode)
        {
            if (pathElement.HasAddition())
            {
                Sub6A3F61PathAddition(session, pathElement, height, sceneryImageTemplate);
            }
        }

        // Redundant zoom-level check removed

        PathPaintFencesAndQueueBanners(session, pathElement, height, connectedEdges, hasSupports, pathPaintInfo, imageTemplate);
    }

    // This is about tunnel drawing
    uint8_t direction = (pathElement.GetSlopeDirection() + session.CurrentRotation) & FOOTPATH_PROPERTIES_SLOPE_DIRECTION_MASK;
    bool sloped = pathElement.IsSloped();

    if (connectedEdges & EDGE_SE)
    {
        // Bottom right of tile is a tunnel
        if (sloped && direction == EDGE_NE)
        {
            // Path going down into the tunnel
            PaintUtilPushTunnelRight(session, height + 16, TUNNEL_PATH_AND_MINI_GOLF);
        }
        else if (connectedEdges & EDGE_NE)
        {
            // Flat path with edge to the right (north-east)
            PaintUtilPushTunnelRight(session, height, TUNNEL_PATH_11);
        }
        else
        {
            // Path going up, or flat and not connected to the right
            PaintUtilPushTunnelRight(session, height, TUNNEL_PATH_AND_MINI_GOLF);
        }
    }

    if (!(connectedEdges & EDGE_SW))
    {
        return;
    }

    // Bottom left of the tile is a tunnel
    if (sloped && direction == EDGE_SE)
    {
        // Path going down into the tunnel
        PaintUtilPushTunnelLeft(session, height + 16, TUNNEL_PATH_AND_MINI_GOLF);
    }
    else if (connectedEdges & EDGE_NW)
    {
        // Flat path with edge to the left (north-west)
        PaintUtilPushTunnelLeft(session, height, TUNNEL_PATH_11);
    }
    else
    {
        // Path going up, or flat and not connected to the left
        PaintUtilPushTunnelLeft(session, height, TUNNEL_PATH_AND_MINI_GOLF);
    }
}

static FootpathPaintInfo GetFootpathPaintInfo(const PathElement& pathEl)
{
    FootpathPaintInfo pathPaintInfo;

    const auto* surfaceDescriptor = pathEl.GetSurfaceDescriptor();
    if (surfaceDescriptor != nullptr)
    {
        pathPaintInfo.SurfaceImageId = surfaceDescriptor->Image;
        pathPaintInfo.SurfaceFlags = surfaceDescriptor->Flags;
    }

    const auto* railingsDescriptor = pathEl.GetRailingsDescriptor();
    if (railingsDescriptor != nullptr)
    {
        pathPaintInfo.BridgeImageId = railingsDescriptor->BridgeImage;
        pathPaintInfo.RailingsImageId = railingsDescriptor->RailingsImage;
        pathPaintInfo.RailingFlags = railingsDescriptor->Flags;
        pathPaintInfo.ScrollingMode = railingsDescriptor->ScrollingMode;
        pathPaintInfo.SupportType = railingsDescriptor->SupportType;
        pathPaintInfo.SupportColour = railingsDescriptor->SupportColour;
    }

    return pathPaintInfo;
}

static bool ShouldDrawSupports(PaintSession& session, const PathElement& pathEl, uint16_t height)
{
    auto surface = MapGetSurfaceElementAt(session.MapPosition);
    if (surface == nullptr)
    {
        return true;
    }
    else if (surface->GetBaseZ() != height)
    {
        const auto* surfaceEntry = pathEl.GetSurfaceEntry();
        const bool showUndergroundRailings = surfaceEntry == nullptr
            || !(surfaceEntry->Flags & FOOTPATH_ENTRY_FLAG_NO_SLOPE_RAILINGS);
        if (surface->GetBaseZ() < height || showUndergroundRailings)
            return true;
    }
    else if (pathEl.IsSloped())
    {
        // Diagonal path
        if (surface->GetSlope() != PathSlopeToLandSlope[pathEl.GetSlopeDirection()])
        {
            return true;
        }
    }
    else if (surface->GetSlope() != TILE_ELEMENT_SLOPE_FLAT)
    {
        return true;
    }
    return false;
}

static void PaintPatrolAreas(PaintSession& session, const PathElement& pathEl)
{
    auto colour = GetPatrolAreaTileColour(session.MapPosition);
    if (colour)
    {
        uint32_t baseImageIndex = SPR_TERRAIN_STAFF;
        auto patrolAreaBaseZ = pathEl.GetBaseZ();
        if (pathEl.IsSloped())
        {
            baseImageIndex = SPR_TERRAIN_STAFF_SLOPED + ((pathEl.GetSlopeDirection() + session.CurrentRotation) & 3);
            patrolAreaBaseZ += 16;
        }

        auto imageId = ImageId(baseImageIndex, *colour);
        PaintAddImageAsParent(session, imageId, { 16, 16, patrolAreaBaseZ + 2 }, { 1, 1, 0 });
    }
}

static void PaintHeightMarkers(PaintSession& session, const PathElement& pathEl)
{
    PROFILED_FUNCTION();

    if (PaintShouldShowHeightMarkers(session, VIEWPORT_FLAG_PATH_HEIGHTS))
    {
        uint16_t heightMarkerBaseZ = pathEl.GetBaseZ() + 3;
        if (pathEl.IsSloped())
        {
            heightMarkerBaseZ += 8;
        }

        uint32_t baseImageIndex = SPR_HEIGHT_MARKER_BASE;
        baseImageIndex += heightMarkerBaseZ / 16;
        baseImageIndex += GetHeightMarkerOffset();
        baseImageIndex -= gMapBaseZ;
        auto imageId = ImageId(baseImageIndex, COLOUR_GREY);
        PaintAddImageAsParent(session, imageId, { 16, 16, heightMarkerBaseZ }, { 1, 1, 0 });
    }
}

/**
 * rct2: 0x0006A3590
 */
void PaintPath(PaintSession& session, uint16_t height, const PathElement& tileElement)
{
    PROFILED_FUNCTION();

    session.InteractionType = ViewportInteractionItem::Footpath;

    ImageId imageTemplate, sceneryImageTemplate;
    if (gTrackDesignSaveMode)
    {
        // Do not display queues for other rides
        if (tileElement.IsQueue() && tileElement.GetRideIndex() != gTrackDesignSaveRideIndex)
        {
            return;
        }

        if (!TrackDesignSaveContainsTileElement(reinterpret_cast<const TileElement*>(&tileElement)))
        {
            imageTemplate = ImageId().WithRemap(FilterPaletteID::Palette46);
        }
    }

    if (session.ViewFlags & VIEWPORT_FLAG_HIGHLIGHT_PATH_ISSUES)
    {
        imageTemplate = ImageId().WithRemap(FilterPaletteID::Palette46);
    }

    if (tileElement.AdditionIsGhost())
    {
        sceneryImageTemplate = ImageId().WithRemap(FilterPaletteID::PaletteGhost);
    }

    if (tileElement.IsGhost())
    {
        session.InteractionType = ViewportInteractionItem::None;
        imageTemplate = ImageId().WithRemap(FilterPaletteID::PaletteGhost);
    }
    else if (session.SelectedElement == reinterpret_cast<const TileElement*>(&tileElement))
    {
        imageTemplate = ImageId().WithRemap(FilterPaletteID::PaletteGhost);
        sceneryImageTemplate = ImageId().WithRemap(FilterPaletteID::PaletteGhost);
    }

    // For debugging purpose, show blocked tiles with a colour
    if (gPaintBlockedTiles && tileElement.IsBlockedByVehicle())
    {
        imageTemplate = ImageId().WithRemap(FilterPaletteID::Palette46);
    }

    // Draw wide flags as ghosts, leaving only the "walkable" paths to be drawn normally
    if (gPaintWidePathsAsGhost && tileElement.IsWide())
    {
        imageTemplate = ImageId().WithRemap(FilterPaletteID::PaletteGhost);
    }

    PaintPatrolAreas(session, tileElement);
    PaintHeightMarkers(session, tileElement);

    auto hasSupports = ShouldDrawSupports(session, tileElement, height);
    auto pathPaintInfo = GetFootpathPaintInfo(tileElement);
    if (pathPaintInfo.SupportType == RailingEntrySupportType::Pole)
    {
        PathPaintPoleSupport(session, tileElement, height, pathPaintInfo, hasSupports, imageTemplate, sceneryImageTemplate);
    }
    else
    {
        PathPaintBoxSupport(session, tileElement, height, pathPaintInfo, hasSupports, imageTemplate, sceneryImageTemplate);
    }

    PaintLampLightEffects(session, tileElement, height);
}

void PathPaintBoxSupport(
    PaintSession& session, const PathElement& pathElement, int32_t height, const FootpathPaintInfo& pathPaintInfo,
    bool hasSupports, ImageId imageTemplate, ImageId sceneryImageTemplate)
{
    PROFILED_FUNCTION();

    // Rol edges around rotation
    uint8_t edges = ((pathElement.GetEdges() << session.CurrentRotation) & 0xF)
        | (((pathElement.GetEdges()) << session.CurrentRotation) >> 4);

    uint8_t corners = (((pathElement.GetCorners()) << session.CurrentRotation) & 0xF)
        | (((pathElement.GetCorners()) << session.CurrentRotation) >> 4);

    CoordsXY boundBoxOffset = stru_98D804[edges].offset;
    CoordsXY boundBoxSize = stru_98D804[edges].length;

    uint16_t edi = edges | (corners << 4);

    ImageIndex surfaceBaseImageIndex = pathPaintInfo.SurfaceImageId;
    if (pathElement.IsSloped())
    {
        auto directionOffset = (pathElement.GetSlopeDirection() + session.CurrentRotation)
            & FOOTPATH_PROPERTIES_SLOPE_DIRECTION_MASK;
        surfaceBaseImageIndex += 16 + directionOffset;
    }
    else
    {
        surfaceBaseImageIndex += Byte98D6E0[edi];
    }

    const bool hasPassedSurface = (session.Flags & PaintSessionFlags::PassedSurface) != 0;
    if (!hasPassedSurface)
    {
        boundBoxOffset.x = 3;
        boundBoxOffset.y = 3;
        boundBoxSize.x = 26;
        boundBoxSize.y = 26;
    }

    // By default, add 1 to the z bounding box to always clip above the surface
    uint8_t boundingBoxZOffset = 1;

    // If we are on the same tile as a straight track, add the offset 2 so we
    //  can clip above gravel part of the track sprite
    if (session.TrackElementOnSameHeight != nullptr)
    {
        if (session.TrackElementOnSameHeight->AsTrack()->GetTrackType() == TrackElemType::Flat)
        {
            boundingBoxZOffset = 2;
        }
    }

    if (!hasSupports || !hasPassedSurface)
    {
        PaintAddImageAsParent(
            session, imageTemplate.WithIndex(surfaceBaseImageIndex), { 0, 0, height },
            { { boundBoxOffset, height + boundingBoxZOffset }, { boundBoxSize, 0 } });
    }
    else
    {
        ImageIndex bridgeBaseImageIndex;
        if (pathElement.IsSloped())
        {
            auto directionOffset
                = ((pathElement.GetSlopeDirection() + session.CurrentRotation) & FOOTPATH_PROPERTIES_SLOPE_DIRECTION_MASK);
            bridgeBaseImageIndex = pathPaintInfo.BridgeImageId + 51 + directionOffset;
        }
        else
        {
            bridgeBaseImageIndex = Byte98D8A4[edges] + pathPaintInfo.BridgeImageId + 49;
        }

        PaintAddImageAsParent(
            session, imageTemplate.WithIndex(bridgeBaseImageIndex), { 0, 0, height },
            { { boundBoxOffset, height + boundingBoxZOffset }, { boundBoxSize, 0 } });

        if (pathElement.IsQueue() || (pathPaintInfo.RailingFlags & RAILING_ENTRY_FLAG_DRAW_PATH_OVER_SUPPORTS))
        {
            PaintAddImageAsChild(
                session, imageTemplate.WithIndex(surfaceBaseImageIndex), { 0, 0, height },
                { { boundBoxOffset, height + boundingBoxZOffset }, { boundBoxSize, 0 } });
        }
    }

    Sub6A3F61(session, pathElement, edi, height, pathPaintInfo, imageTemplate, sceneryImageTemplate, hasSupports);

    uint16_t ax = 0;
    if (pathElement.IsSloped())
    {
        ax = ((pathElement.GetSlopeDirection() + session.CurrentRotation) & 0x3) + 1;
    }

    auto supportType = Byte98D8A4[edges] == 0 ? 0 : 1;
    PathASupportsPaintSetup(session, supportType, ax, height, imageTemplate, pathPaintInfo, nullptr);

    height += 32;
    if (pathElement.IsSloped())
    {
        height += 16;
    }

    PaintUtilSetGeneralSupportHeight(session, height, 0x20);

    if (pathElement.IsQueue() || (pathElement.GetEdgesAndCorners() != 0xFF && hasSupports))
    {
        PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
        return;
    }

    if (pathElement.GetEdgesAndCorners() == 0xFF)
    {
        PaintUtilSetSegmentSupportHeight(session, SEGMENT_C8 | SEGMENT_CC | SEGMENT_D0 | SEGMENT_D4, 0xFFFF, 0);
        return;
    }

    PaintUtilSetSegmentSupportHeight(session, SEGMENT_C4, 0xFFFF, 0);

    if (edges & 1)
    {
        PaintUtilSetSegmentSupportHeight(session, SEGMENT_CC, 0xFFFF, 0);
    }

    if (edges & 2)
    {
        PaintUtilSetSegmentSupportHeight(session, SEGMENT_D4, 0xFFFF, 0);
    }

    if (edges & 4)
    {
        PaintUtilSetSegmentSupportHeight(session, SEGMENT_D0, 0xFFFF, 0);
    }

    if (edges & 8)
    {
        PaintUtilSetSegmentSupportHeight(session, SEGMENT_C8, 0xFFFF, 0);
    }
}

void PathPaintPoleSupport(
    PaintSession& session, const PathElement& pathElement, int16_t height, const FootpathPaintInfo& pathPaintInfo,
    bool hasSupports, ImageId imageTemplate, ImageId sceneryImageTemplate)
{
    PROFILED_FUNCTION();

    // Rol edges around rotation
    uint8_t edges = ((pathElement.GetEdges() << session.CurrentRotation) & 0xF)
        | (((pathElement.GetEdges()) << session.CurrentRotation) >> 4);

    CoordsXY boundBoxOffset = stru_98D804[edges].offset;
    CoordsXY boundBoxSize = stru_98D804[edges].length;

    uint8_t corners = (((pathElement.GetCorners()) << session.CurrentRotation) & 0xF)
        | (((pathElement.GetCorners()) << session.CurrentRotation) >> 4);

    uint16_t edi = edges | (corners << 4);

    ImageIndex surfaceBaseImageIndex = pathPaintInfo.SurfaceImageId;
    if (pathElement.IsSloped())
    {
        auto directionOffset
            = ((pathElement.GetSlopeDirection() + session.CurrentRotation) & FOOTPATH_PROPERTIES_SLOPE_DIRECTION_MASK);
        surfaceBaseImageIndex += 16 + directionOffset;
    }
    else
    {
        surfaceBaseImageIndex += Byte98D6E0[edi];
    }

    // Below Surface
    const bool hasPassedSurface = (session.Flags & PaintSessionFlags::PassedSurface) != 0;
    if (!hasPassedSurface)
    {
        boundBoxOffset.x = 3;
        boundBoxOffset.y = 3;
        boundBoxSize.x = 26;
        boundBoxSize.y = 26;
    }

    // By default, add 1 to the z bounding box to always clip above the surface
    uint8_t boundingBoxZOffset = 1;

    // If we are on the same tile as a straight track, add the offset 2 so we
    //  can clip above gravel part of the track sprite
    if (session.TrackElementOnSameHeight != nullptr)
    {
        if (session.TrackElementOnSameHeight->AsTrack()->GetTrackType() == TrackElemType::Flat)
        {
            boundingBoxZOffset = 2;
        }
    }

    if (!hasSupports || !hasPassedSurface)
    {
        PaintAddImageAsParent(
            session, imageTemplate.WithIndex(surfaceBaseImageIndex), { 0, 0, height },
            { { boundBoxOffset.x, boundBoxOffset.y, height + boundingBoxZOffset }, { boundBoxSize.x, boundBoxSize.y, 0 } });
    }
    else
    {
        ImageIndex bridgeBaseImageIndex;
        if (pathElement.IsSloped())
        {
            bridgeBaseImageIndex = ((pathElement.GetSlopeDirection() + session.CurrentRotation)
                                    & FOOTPATH_PROPERTIES_SLOPE_DIRECTION_MASK)
                + pathPaintInfo.BridgeImageId + 16;
        }
        else
        {
            bridgeBaseImageIndex = edges + pathPaintInfo.BridgeImageId;
        }

        PaintAddImageAsParent(
            session, imageTemplate.WithIndex(bridgeBaseImageIndex), { 0, 0, height },
            { { boundBoxOffset, height + boundingBoxZOffset }, { boundBoxSize, 0 } });

        if (pathElement.IsQueue() || (pathPaintInfo.RailingFlags & RAILING_ENTRY_FLAG_DRAW_PATH_OVER_SUPPORTS))
        {
            PaintAddImageAsChild(
                session, imageTemplate.WithIndex(surfaceBaseImageIndex), { 0, 0, height },
                { { boundBoxOffset, height + boundingBoxZOffset }, { boundBoxSize, 0 } });
        }
    }

    Sub6A3F61(
        session, pathElement, edi, height, pathPaintInfo, imageTemplate, sceneryImageTemplate,
        hasSupports); // TODO: arguments

    uint16_t ax = 0;
    if (pathElement.IsSloped())
    {
        ax = 8;
    }

    uint8_t supports[] = {
        6,
        8,
        7,
        5,
    };

    for (int8_t i = 3; i > -1; --i)
    {
        if (!(edges & (1 << i)))
        {
            // Only colour the supports if not already remapped (e.g. ghost remap)
            auto supportColour = pathPaintInfo.SupportColour;
            if (supportColour != COLOUR_NULL && !imageTemplate.IsRemap())
            {
                imageTemplate = ImageId().WithPrimary(supportColour);
            }
            PathBSupportsPaintSetup(session, supports[i], ax, height, imageTemplate, pathPaintInfo);
        }
    }

    height += 32;
    if (pathElement.IsSloped())
    {
        height += 16;
    }

    PaintUtilSetGeneralSupportHeight(session, height, 0x20);

    if (pathElement.IsQueue() || (pathElement.GetEdgesAndCorners() != 0xFF && hasSupports))
    {
        PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
        return;
    }

    if (pathElement.GetEdgesAndCorners() == 0xFF)
    {
        PaintUtilSetSegmentSupportHeight(session, SEGMENT_C8 | SEGMENT_CC | SEGMENT_D0 | SEGMENT_D4, 0xFFFF, 0);
        return;
    }

    PaintUtilSetSegmentSupportHeight(session, SEGMENT_C4, 0xFFFF, 0);

    if (edges & EDGE_NE)
    {
        PaintUtilSetSegmentSupportHeight(session, SEGMENT_CC, 0xFFFF, 0);
    }

    if (edges & EDGE_SE)
    {
        PaintUtilSetSegmentSupportHeight(session, SEGMENT_D4, 0xFFFF, 0);
    }

    if (edges & EDGE_SW)
    {
        PaintUtilSetSegmentSupportHeight(session, SEGMENT_D0, 0xFFFF, 0);
    }

    if (edges & EDGE_NW)
    {
        PaintUtilSetSegmentSupportHeight(session, SEGMENT_C8, 0xFFFF, 0);
    }
}
