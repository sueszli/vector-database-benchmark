/*****************************************************************************
 * Copyright (c) 2014-2023 OpenRCT2 developers
 *
 * For a complete list of all authors, please refer to contributors.md
 * Interested in contributing? Visit https://github.com/OpenRCT2/OpenRCT2
 *
 * OpenRCT2 is licensed under the GNU General Public License version 3.
 *****************************************************************************/

#include "../../drawing/Drawing.h"
#include "../../interface/Viewport.h"
#include "../../paint/Paint.h"
#include "../../paint/Supports.h"
#include "../../paint/tile_element/Paint.TileElement.h"
#include "../../sprites.h"
#include "../../world/Map.h"
#include "../RideData.h"
#include "../TrackData.h"
#include "../TrackPaint.h"

/** rct2: 0x008B08D0 */
static void MineRideTrackFlat(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    switch (direction)
    {
        case 0:
        case 2:
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19338), { 0, 0, height },
                { { 0, 6, height }, { 32, 20, 3 } });
            if (TrackPaintUtilShouldPaintSupports(session.MapPosition))
            {
                MetalASupportsPaintSetup(
                    session, MetalSupportType::Fork, MetalSupportPlace::Centre, 6, height,
                    session.TrackColours[SCHEME_SUPPORTS]);
            }
            break;
        case 1:
        case 3:
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19339), { 0, 0, height },
                { { 0, 6, height }, { 32, 20, 3 } });
            if (TrackPaintUtilShouldPaintSupports(session.MapPosition))
            {
                MetalASupportsPaintSetup(
                    session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 6, height,
                    session.TrackColours[SCHEME_SUPPORTS]);
            }
            break;
    }
    PaintUtilPushTunnelRotated(session, direction, height, TUNNEL_SQUARE_FLAT);
    PaintUtilSetSegmentSupportHeight(
        session, PaintUtilRotateSegments(SEGMENT_C4 | SEGMENT_CC | SEGMENT_D0, direction), 0xFFFF, 0);
    PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
}

static void MineRideTrackStation(
    PaintSession& session, const Ride& ride, [[maybe_unused]] uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    static constexpr uint32_t imageIds[4][3] = {
        { 19338, SPR_STATION_BASE_A_SW_NE },
        { 19339, SPR_STATION_BASE_A_NW_SE },
        { 19338, SPR_STATION_BASE_A_SW_NE },
        { 19339, SPR_STATION_BASE_A_NW_SE },
    };

    PaintAddImageAsParentRotated(
        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(imageIds[direction][0]), { 0, 0, height },
        { { 0, 6, height + 3 }, { 32, 20, 1 } });
    PaintAddImageAsParentRotated(
        session, direction, session.TrackColours[SCHEME_MISC].WithIndex(imageIds[direction][1]), { 0, 0, height },
        { 32, 32, 1 });
    DrawSupportsSideBySide(session, direction, height, session.TrackColours[SCHEME_SUPPORTS], MetalSupportType::Tubes);
    TrackPaintUtilDrawStation2(session, ride, direction, height, trackElement, 9, 11);
    PaintUtilPushTunnelRotated(session, direction, height, TUNNEL_SQUARE_FLAT);
    PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
    PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
}

/** rct2: 0x008B08E0 */
static void MineRideTrack25DegUp(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    switch (direction)
    {
        case 0:
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19388), { 0, 0, height },
                { { 0, 6, height }, { 32, 20, 3 } });
            if (TrackPaintUtilShouldPaintSupports(session.MapPosition))
            {
                MetalASupportsPaintSetup(
                    session, MetalSupportType::Fork, MetalSupportPlace::Centre, 14, height,
                    session.TrackColours[SCHEME_SUPPORTS]);
            }
            break;
        case 1:
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19389), { 0, 0, height },
                { { 0, 6, height }, { 32, 20, 3 } });
            if (TrackPaintUtilShouldPaintSupports(session.MapPosition))
            {
                MetalASupportsPaintSetup(
                    session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 14, height,
                    session.TrackColours[SCHEME_SUPPORTS]);
            }
            break;
        case 2:
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19390), { 0, 0, height },
                { { 0, 6, height }, { 32, 20, 3 } });
            if (TrackPaintUtilShouldPaintSupports(session.MapPosition))
            {
                MetalASupportsPaintSetup(
                    session, MetalSupportType::Fork, MetalSupportPlace::Centre, 14, height,
                    session.TrackColours[SCHEME_SUPPORTS]);
            }
            break;
        case 3:
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19391), { 0, 0, height },
                { { 0, 6, height }, { 32, 20, 3 } });
            if (TrackPaintUtilShouldPaintSupports(session.MapPosition))
            {
                MetalASupportsPaintSetup(
                    session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 14, height,
                    session.TrackColours[SCHEME_SUPPORTS]);
            }
            break;
    }
    if (direction == 0 || direction == 3)
    {
        PaintUtilPushTunnelRotated(session, direction, height - 8, TUNNEL_SQUARE_7);
    }
    else
    {
        PaintUtilPushTunnelRotated(session, direction, height + 8, TUNNEL_SQUARE_8);
    }
    PaintUtilSetSegmentSupportHeight(
        session, PaintUtilRotateSegments(SEGMENT_C4 | SEGMENT_CC | SEGMENT_D0, direction), 0xFFFF, 0);
    PaintUtilSetGeneralSupportHeight(session, height + 56, 0x20);
}

/** rct2: 0x008B08F0 */
static void MineRideTrackFlatTo25DegUp(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    switch (direction)
    {
        case 0:
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19380), { 0, 0, height },
                { { 0, 6, height }, { 32, 20, 3 } });
            if (TrackPaintUtilShouldPaintSupports(session.MapPosition))
            {
                MetalASupportsPaintSetup(
                    session, MetalSupportType::Fork, MetalSupportPlace::Centre, 9, height,
                    session.TrackColours[SCHEME_SUPPORTS]);
            }
            break;
        case 1:
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19381), { 0, 0, height },
                { { 0, 6, height }, { 32, 20, 3 } });
            if (TrackPaintUtilShouldPaintSupports(session.MapPosition))
            {
                MetalASupportsPaintSetup(
                    session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 9, height,
                    session.TrackColours[SCHEME_SUPPORTS]);
            }
            break;
        case 2:
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19382), { 0, 0, height },
                { { 0, 6, height }, { 32, 20, 3 } });
            if (TrackPaintUtilShouldPaintSupports(session.MapPosition))
            {
                MetalASupportsPaintSetup(
                    session, MetalSupportType::Fork, MetalSupportPlace::Centre, 9, height,
                    session.TrackColours[SCHEME_SUPPORTS]);
            }
            break;
        case 3:
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19383), { 0, 0, height },
                { { 0, 6, height }, { 32, 20, 3 } });
            if (TrackPaintUtilShouldPaintSupports(session.MapPosition))
            {
                MetalASupportsPaintSetup(
                    session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 9, height,
                    session.TrackColours[SCHEME_SUPPORTS]);
            }
            break;
    }
    if (direction == 0 || direction == 3)
    {
        PaintUtilPushTunnelRotated(session, direction, height, TUNNEL_SQUARE_FLAT);
    }
    else
    {
        PaintUtilPushTunnelRotated(session, direction, height, TUNNEL_SQUARE_8);
    }
    PaintUtilSetSegmentSupportHeight(
        session, PaintUtilRotateSegments(SEGMENT_C4 | SEGMENT_CC | SEGMENT_D0, direction), 0xFFFF, 0);
    PaintUtilSetGeneralSupportHeight(session, height + 48, 0x20);
}

/** rct2: 0x008B0900 */
static void MineRideTrack25DegUpToFlat(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    switch (direction)
    {
        case 0:
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19384), { 0, 0, height },
                { { 0, 6, height }, { 32, 20, 3 } });
            if (TrackPaintUtilShouldPaintSupports(session.MapPosition))
            {
                MetalASupportsPaintSetup(
                    session, MetalSupportType::Fork, MetalSupportPlace::Centre, 12, height,
                    session.TrackColours[SCHEME_SUPPORTS]);
            }
            break;
        case 1:
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19385), { 0, 0, height },
                { { 0, 6, height }, { 32, 20, 3 } });
            if (TrackPaintUtilShouldPaintSupports(session.MapPosition))
            {
                MetalASupportsPaintSetup(
                    session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 12, height,
                    session.TrackColours[SCHEME_SUPPORTS]);
            }
            break;
        case 2:
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19386), { 0, 0, height },
                { { 0, 6, height }, { 32, 20, 3 } });
            if (TrackPaintUtilShouldPaintSupports(session.MapPosition))
            {
                MetalASupportsPaintSetup(
                    session, MetalSupportType::Fork, MetalSupportPlace::Centre, 12, height,
                    session.TrackColours[SCHEME_SUPPORTS]);
            }
            break;
        case 3:
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19387), { 0, 0, height },
                { { 0, 6, height }, { 32, 20, 3 } });
            if (TrackPaintUtilShouldPaintSupports(session.MapPosition))
            {
                MetalASupportsPaintSetup(
                    session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 12, height,
                    session.TrackColours[SCHEME_SUPPORTS]);
            }
            break;
    }
    if (direction == 0 || direction == 3)
    {
        PaintUtilPushTunnelRotated(session, direction, height - 8, TUNNEL_SQUARE_FLAT);
    }
    else
    {
        PaintUtilPushTunnelRotated(session, direction, height + 8, TUNNEL_14);
    }
    PaintUtilSetSegmentSupportHeight(
        session, PaintUtilRotateSegments(SEGMENT_C4 | SEGMENT_CC | SEGMENT_D0, direction), 0xFFFF, 0);
    PaintUtilSetGeneralSupportHeight(session, height + 40, 0x20);
}

/** rct2: 0x008B0910 */
static void MineRideTrack25DegDown(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    MineRideTrack25DegUp(session, ride, trackSequence, (direction + 2) & 3, height, trackElement);
}

/** rct2: 0x008B0920 */
static void MineRideTrackFlatTo25DegDown(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    MineRideTrack25DegUpToFlat(session, ride, trackSequence, (direction + 2) & 3, height, trackElement);
}

/** rct2: 0x008B0930 */
static void MineRideTrack25DegDownToFlat(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    MineRideTrackFlatTo25DegUp(session, ride, trackSequence, (direction + 2) & 3, height, trackElement);
}

/** rct2: 0x008B0940 */
static void MineRideTrackLeftQuarterTurn5(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    switch (trackSequence)
    {
        case 0:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19427), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19432), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19437), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19422), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            if (direction == 0 || direction == 3)
            {
                PaintUtilPushTunnelRotated(session, direction, height, TUNNEL_SQUARE_FLAT);
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B4 | SEGMENT_C4 | SEGMENT_CC | SEGMENT_D0, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 1:
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 2:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19426), { 0, 0, height },
                        { { 0, 0, height }, { 32, 16, 3 } });
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19431), { 0, 0, height },
                        { { 0, 0, height }, { 32, 16, 3 } });
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19436), { 0, 0, height },
                        { { 0, 16, height }, { 32, 16, 3 } });
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19421), { 0, 0, height },
                        { { 0, 16, height }, { 32, 16, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session,
                PaintUtilRotateSegments(SEGMENT_B4 | SEGMENT_B8 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_CC | SEGMENT_D0, direction),
                0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 3:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19425), { 0, 0, height },
                        { { 0, 16, height }, { 16, 16, 3 } });
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19430), { 0, 0, height },
                        { { 6, 6, height }, { 16, 16, 3 } });
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19435), { 0, 0, height },
                        { { 16, 0, height }, { 16, 16, 3 } });
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19420), { 0, 0, height },
                        { { 0, 0, height }, { 16, 16, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_BC | SEGMENT_C4 | SEGMENT_CC | SEGMENT_D4, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 4:
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 5:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19424), { 0, 0, height },
                        { { 16, 0, height }, { 16, 32, 3 } });
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19429), { 0, 0, height },
                        { { 0, 0, height }, { 16, 32, 3 } });
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19434), { 0, 0, height },
                        { { 0, 0, height }, { 16, 32, 3 } });
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19419), { 0, 0, height },
                        { { 16, 0, height }, { 16, 32, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session,
                PaintUtilRotateSegments(SEGMENT_B8 | SEGMENT_C0 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_D0 | SEGMENT_D4, direction),
                0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 6:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19423), { 0, 0, height },
                        { { 6, 0, height }, { 20, 32, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19428), { 0, 0, height },
                        { { 6, 0, height }, { 20, 32, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19433), { 0, 0, height },
                        { { 6, 0, height }, { 20, 32, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19418), { 0, 0, height },
                        { { 6, 0, height }, { 20, 32, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            switch (direction)
            {
                case 2:
                    PaintUtilPushTunnelRight(session, height, TUNNEL_SQUARE_FLAT);
                    break;
                case 3:
                    PaintUtilPushTunnelLeft(session, height, TUNNEL_SQUARE_FLAT);
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_C0 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_D4, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
    }
}

/** rct2: 0x008B0950 */
static void MineRideTrackRightQuarterTurn5(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    trackSequence = mapLeftQuarterTurn5TilesToRightQuarterTurn5Tiles[trackSequence];
    MineRideTrackLeftQuarterTurn5(session, ride, trackSequence, (direction - 1) & 3, height, trackElement);
}

/** rct2: 0x008B0960 */
static void MineRideTrackFlatToLeftBank(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    switch (direction)
    {
        case 0:
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19340), { 0, 0, height },
                { { 0, 6, height }, { 32, 20, 3 } });
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19348), { 0, 0, height },
                { { 0, 27, height }, { 32, 1, 26 } });
            if (TrackPaintUtilShouldPaintSupports(session.MapPosition))
            {
                MetalASupportsPaintSetup(
                    session, MetalSupportType::Fork, MetalSupportPlace::Centre, 6, height,
                    session.TrackColours[SCHEME_SUPPORTS]);
            }
            break;
        case 1:
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19341), { 0, 0, height },
                { { 0, 6, height }, { 32, 20, 3 } });
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19349), { 0, 0, height },
                { { 0, 27, height }, { 32, 1, 26 } });
            if (TrackPaintUtilShouldPaintSupports(session.MapPosition))
            {
                MetalASupportsPaintSetup(
                    session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 6, height,
                    session.TrackColours[SCHEME_SUPPORTS]);
            }
            break;
        case 2:
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19342), { 0, 0, height },
                { { 0, 6, height }, { 32, 20, 3 } });
            if (TrackPaintUtilShouldPaintSupports(session.MapPosition))
            {
                MetalASupportsPaintSetup(
                    session, MetalSupportType::Fork, MetalSupportPlace::Centre, 6, height,
                    session.TrackColours[SCHEME_SUPPORTS]);
            }
            break;
        case 3:
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19343), { 0, 0, height },
                { { 0, 6, height }, { 32, 20, 3 } });
            if (TrackPaintUtilShouldPaintSupports(session.MapPosition))
            {
                MetalASupportsPaintSetup(
                    session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 6, height,
                    session.TrackColours[SCHEME_SUPPORTS]);
            }
            break;
    }
    PaintUtilPushTunnelRotated(session, direction, height, TUNNEL_SQUARE_FLAT);
    PaintUtilSetSegmentSupportHeight(
        session, PaintUtilRotateSegments(SEGMENT_C4 | SEGMENT_CC | SEGMENT_D0, direction), 0xFFFF, 0);
    PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
}

/** rct2: 0x008B0970 */
static void MineRideTrackFlatToRightBank(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    switch (direction)
    {
        case 0:
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19344), { 0, 0, height },
                { { 0, 6, height }, { 32, 20, 3 } });
            if (TrackPaintUtilShouldPaintSupports(session.MapPosition))
            {
                MetalASupportsPaintSetup(
                    session, MetalSupportType::Fork, MetalSupportPlace::Centre, 6, height,
                    session.TrackColours[SCHEME_SUPPORTS]);
            }
            break;
        case 1:
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19345), { 0, 0, height },
                { { 0, 6, height }, { 32, 20, 3 } });
            if (TrackPaintUtilShouldPaintSupports(session.MapPosition))
            {
                MetalASupportsPaintSetup(
                    session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 6, height,
                    session.TrackColours[SCHEME_SUPPORTS]);
            }
            break;
        case 2:
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19346), { 0, 0, height },
                { { 0, 6, height }, { 32, 20, 3 } });
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19350), { 0, 0, height },
                { { 0, 27, height }, { 32, 1, 26 } });
            if (TrackPaintUtilShouldPaintSupports(session.MapPosition))
            {
                MetalASupportsPaintSetup(
                    session, MetalSupportType::Fork, MetalSupportPlace::Centre, 6, height,
                    session.TrackColours[SCHEME_SUPPORTS]);
            }
            break;
        case 3:
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19347), { 0, 0, height },
                { { 0, 6, height }, { 32, 20, 3 } });
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19351), { 0, 0, height },
                { { 0, 27, height }, { 32, 1, 26 } });
            if (TrackPaintUtilShouldPaintSupports(session.MapPosition))
            {
                MetalASupportsPaintSetup(
                    session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 6, height,
                    session.TrackColours[SCHEME_SUPPORTS]);
            }
            break;
    }
    PaintUtilPushTunnelRotated(session, direction, height, TUNNEL_SQUARE_FLAT);
    PaintUtilSetSegmentSupportHeight(
        session, PaintUtilRotateSegments(SEGMENT_C4 | SEGMENT_CC | SEGMENT_D0, direction), 0xFFFF, 0);
    PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
}

/** rct2: 0x008B0980 */
static void MineRideTrackLeftBankToFlat(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    switch (direction)
    {
        case 0:
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19346), { 0, 0, height },
                { { 0, 6, height }, { 32, 20, 3 } });
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19350), { 0, 0, height },
                { { 0, 27, height }, { 32, 1, 26 } });
            if (TrackPaintUtilShouldPaintSupports(session.MapPosition))
            {
                MetalASupportsPaintSetup(
                    session, MetalSupportType::Fork, MetalSupportPlace::Centre, 6, height,
                    session.TrackColours[SCHEME_SUPPORTS]);
            }
            break;
        case 1:
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19347), { 0, 0, height },
                { { 0, 6, height }, { 32, 20, 3 } });
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19351), { 0, 0, height },
                { { 0, 27, height }, { 32, 1, 26 } });
            if (TrackPaintUtilShouldPaintSupports(session.MapPosition))
            {
                MetalASupportsPaintSetup(
                    session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 6, height,
                    session.TrackColours[SCHEME_SUPPORTS]);
            }
            break;
        case 2:
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19344), { 0, 0, height },
                { { 0, 6, height }, { 32, 20, 3 } });
            if (TrackPaintUtilShouldPaintSupports(session.MapPosition))
            {
                MetalASupportsPaintSetup(
                    session, MetalSupportType::Fork, MetalSupportPlace::Centre, 6, height,
                    session.TrackColours[SCHEME_SUPPORTS]);
            }
            break;
        case 3:
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19345), { 0, 0, height },
                { { 0, 6, height }, { 32, 20, 3 } });
            if (TrackPaintUtilShouldPaintSupports(session.MapPosition))
            {
                MetalASupportsPaintSetup(
                    session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 6, height,
                    session.TrackColours[SCHEME_SUPPORTS]);
            }
            break;
    }
    PaintUtilPushTunnelRotated(session, direction, height, TUNNEL_SQUARE_FLAT);
    PaintUtilSetSegmentSupportHeight(
        session, PaintUtilRotateSegments(SEGMENT_C4 | SEGMENT_CC | SEGMENT_D0, direction), 0xFFFF, 0);
    PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
}

/** rct2: 0x008B0990 */
static void MineRideTrackRightBankToFlat(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    switch (direction)
    {
        case 0:
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19342), { 0, 0, height },
                { { 0, 6, height }, { 32, 20, 3 } });
            if (TrackPaintUtilShouldPaintSupports(session.MapPosition))
            {
                MetalASupportsPaintSetup(
                    session, MetalSupportType::Fork, MetalSupportPlace::Centre, 6, height,
                    session.TrackColours[SCHEME_SUPPORTS]);
            }
            break;
        case 1:
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19343), { 0, 0, height },
                { { 0, 6, height }, { 32, 20, 3 } });
            if (TrackPaintUtilShouldPaintSupports(session.MapPosition))
            {
                MetalASupportsPaintSetup(
                    session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 6, height,
                    session.TrackColours[SCHEME_SUPPORTS]);
            }
            break;
        case 2:
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19340), { 0, 0, height },
                { { 0, 6, height }, { 32, 20, 3 } });
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19348), { 0, 0, height },
                { { 0, 27, height }, { 32, 1, 26 } });
            if (TrackPaintUtilShouldPaintSupports(session.MapPosition))
            {
                MetalASupportsPaintSetup(
                    session, MetalSupportType::Fork, MetalSupportPlace::Centre, 6, height,
                    session.TrackColours[SCHEME_SUPPORTS]);
            }
            break;
        case 3:
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19341), { 0, 0, height },
                { { 0, 6, height }, { 32, 20, 3 } });
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19349), { 0, 0, height },
                { { 0, 27, height }, { 32, 1, 26 } });
            if (TrackPaintUtilShouldPaintSupports(session.MapPosition))
            {
                MetalASupportsPaintSetup(
                    session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 6, height,
                    session.TrackColours[SCHEME_SUPPORTS]);
            }
            break;
    }
    PaintUtilPushTunnelRotated(session, direction, height, TUNNEL_SQUARE_FLAT);
    PaintUtilSetSegmentSupportHeight(
        session, PaintUtilRotateSegments(SEGMENT_C4 | SEGMENT_CC | SEGMENT_D0, direction), 0xFFFF, 0);
    PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
}

/** rct2: 0x008B09A0 */
static void MineRideTrackBankedLeftQuarterTurn5(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    switch (trackSequence)
    {
        case 0:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19447), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19458), { 0, 0, height },
                        { { 0, 27, height }, { 32, 1, 26 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19452), { 0, 0, height },
                        { { 0, 27, height }, { 32, 1, 26 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19457), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19442), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            if (direction == 0 || direction == 3)
            {
                PaintUtilPushTunnelRotated(session, direction, height, TUNNEL_SQUARE_FLAT);
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B4 | SEGMENT_C4 | SEGMENT_CC | SEGMENT_D0, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 1:
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 2:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19446), { 0, 0, height },
                        { { 0, 0, height }, { 32, 16, 3 } });
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19451), { 0, 0, height },
                        { { 0, 0, height + 27 }, { 32, 16, 1 } });
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19456), { 0, 0, height },
                        { { 0, 16, height }, { 32, 16, 3 } });
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19441), { 0, 0, height },
                        { { 0, 16, height }, { 32, 16, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session,
                PaintUtilRotateSegments(SEGMENT_B4 | SEGMENT_B8 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_CC | SEGMENT_D0, direction),
                0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 3:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19445), { 0, 0, height },
                        { { 0, 16, height }, { 16, 16, 3 } });
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19450), { 0, 0, height },
                        { { 16, 16, height + 27 }, { 16, 16, 1 } });
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19455), { 0, 0, height },
                        { { 16, 0, height }, { 16, 16, 3 } });
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19440), { 0, 0, height },
                        { { 0, 0, height }, { 16, 16, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_BC | SEGMENT_C4 | SEGMENT_CC | SEGMENT_D4, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 4:
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 5:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19444), { 0, 0, height },
                        { { 16, 0, height }, { 16, 32, 3 } });
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19449), { 0, 0, height },
                        { { 0, 0, height + 27 }, { 16, 32, 1 } });
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19454), { 0, 0, height },
                        { { 0, 0, height }, { 16, 32, 3 } });
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19439), { 0, 0, height },
                        { { 16, 0, height }, { 16, 32, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session,
                PaintUtilRotateSegments(SEGMENT_B8 | SEGMENT_C0 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_D0 | SEGMENT_D4, direction),
                0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 6:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19443), { 0, 0, height },
                        { { 6, 0, height }, { 20, 32, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19448), { 0, 0, height },
                        { { 27, 0, height }, { 1, 32, 26 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19453), { 0, 0, height },
                        { { 6, 0, height }, { 20, 32, 3 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19459), { 0, 0, height },
                        { { 27, 0, height }, { 1, 32, 26 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19438), { 0, 0, height },
                        { { 6, 0, height }, { 20, 32, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            switch (direction)
            {
                case 2:
                    PaintUtilPushTunnelRight(session, height, TUNNEL_SQUARE_FLAT);
                    break;
                case 3:
                    PaintUtilPushTunnelLeft(session, height, TUNNEL_SQUARE_FLAT);
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_C0 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_D4, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
    }
}

/** rct2: 0x008B09B0 */
static void MineRideTrackBankedRightQuarterTurn5(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    trackSequence = mapLeftQuarterTurn5TilesToRightQuarterTurn5Tiles[trackSequence];
    MineRideTrackBankedLeftQuarterTurn5(session, ride, trackSequence, (direction - 1) & 3, height, trackElement);
}

/** rct2: 0x008B09C0 */
static void MineRideTrackLeftBankTo25DegUp(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    switch (direction)
    {
        case 0:
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19352), { 0, 0, height },
                { { 0, 6, height }, { 32, 20, 3 } });
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19356), { 0, 0, height },
                { { 0, 27, height }, { 32, 1, 34 } });
            if (TrackPaintUtilShouldPaintSupports(session.MapPosition))
            {
                MetalASupportsPaintSetup(
                    session, MetalSupportType::Fork, MetalSupportPlace::Centre, 9, height,
                    session.TrackColours[SCHEME_SUPPORTS]);
            }
            break;
        case 1:
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19353), { 0, 0, height },
                { { 0, 6, height }, { 32, 20, 3 } });
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19357), { 0, 0, height },
                { { 0, 27, height }, { 32, 1, 34 } });
            if (TrackPaintUtilShouldPaintSupports(session.MapPosition))
            {
                MetalASupportsPaintSetup(
                    session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 9, height,
                    session.TrackColours[SCHEME_SUPPORTS]);
            }
            break;
        case 2:
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19354), { 0, 0, height },
                { { 0, 6, height }, { 32, 20, 3 } });
            if (TrackPaintUtilShouldPaintSupports(session.MapPosition))
            {
                MetalASupportsPaintSetup(
                    session, MetalSupportType::Fork, MetalSupportPlace::Centre, 9, height,
                    session.TrackColours[SCHEME_SUPPORTS]);
            }
            break;
        case 3:
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19355), { 0, 0, height },
                { { 0, 6, height }, { 32, 20, 3 } });
            if (TrackPaintUtilShouldPaintSupports(session.MapPosition))
            {
                MetalASupportsPaintSetup(
                    session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 9, height,
                    session.TrackColours[SCHEME_SUPPORTS]);
            }
            break;
    }
    if (direction == 0 || direction == 3)
    {
        PaintUtilPushTunnelRotated(session, direction, height, TUNNEL_SQUARE_FLAT);
    }
    else
    {
        PaintUtilPushTunnelRotated(session, direction, height, TUNNEL_SQUARE_8);
    }
    PaintUtilSetSegmentSupportHeight(
        session, PaintUtilRotateSegments(SEGMENT_C4 | SEGMENT_CC | SEGMENT_D0, direction), 0xFFFF, 0);
    PaintUtilSetGeneralSupportHeight(session, height + 48, 0x20);
}

/** rct2: 0x008B09D0 */
static void MineRideTrackRightBankTo25DegUp(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    switch (direction)
    {
        case 0:
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19358), { 0, 0, height },
                { { 0, 6, height }, { 32, 20, 3 } });
            if (TrackPaintUtilShouldPaintSupports(session.MapPosition))
            {
                MetalASupportsPaintSetup(
                    session, MetalSupportType::Fork, MetalSupportPlace::Centre, 9, height,
                    session.TrackColours[SCHEME_SUPPORTS]);
            }
            break;
        case 1:
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19359), { 0, 0, height },
                { { 0, 6, height }, { 32, 20, 3 } });
            if (TrackPaintUtilShouldPaintSupports(session.MapPosition))
            {
                MetalASupportsPaintSetup(
                    session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 9, height,
                    session.TrackColours[SCHEME_SUPPORTS]);
            }
            break;
        case 2:
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19360), { 0, 0, height },
                { { 0, 6, height }, { 32, 20, 3 } });
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19362), { 0, 0, height },
                { { 0, 27, height }, { 32, 1, 34 } });
            if (TrackPaintUtilShouldPaintSupports(session.MapPosition))
            {
                MetalASupportsPaintSetup(
                    session, MetalSupportType::Fork, MetalSupportPlace::Centre, 9, height,
                    session.TrackColours[SCHEME_SUPPORTS]);
            }
            break;
        case 3:
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19361), { 0, 0, height },
                { { 0, 6, height }, { 32, 20, 3 } });
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19363), { 0, 0, height },
                { { 0, 27, height }, { 32, 1, 34 } });
            if (TrackPaintUtilShouldPaintSupports(session.MapPosition))
            {
                MetalASupportsPaintSetup(
                    session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 9, height,
                    session.TrackColours[SCHEME_SUPPORTS]);
            }
            break;
    }
    if (direction == 0 || direction == 3)
    {
        PaintUtilPushTunnelRotated(session, direction, height, TUNNEL_SQUARE_FLAT);
    }
    else
    {
        PaintUtilPushTunnelRotated(session, direction, height, TUNNEL_SQUARE_8);
    }
    PaintUtilSetSegmentSupportHeight(
        session, PaintUtilRotateSegments(SEGMENT_C4 | SEGMENT_CC | SEGMENT_D0, direction), 0xFFFF, 0);
    PaintUtilSetGeneralSupportHeight(session, height + 48, 0x20);
}

/** rct2: 0x008B09E0 */
static void MineRideTrack25DegUpToLeftBank(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    switch (direction)
    {
        case 0:
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19364), { 0, 0, height },
                { { 0, 6, height }, { 32, 20, 3 } });
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19368), { 0, 0, height },
                { { 0, 27, height }, { 32, 1, 34 } });
            if (TrackPaintUtilShouldPaintSupports(session.MapPosition))
            {
                MetalASupportsPaintSetup(
                    session, MetalSupportType::Fork, MetalSupportPlace::Centre, 12, height,
                    session.TrackColours[SCHEME_SUPPORTS]);
            }
            break;
        case 1:
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19365), { 0, 0, height },
                { { 0, 6, height }, { 32, 20, 3 } });
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19369), { 0, 0, height },
                { { 0, 27, height }, { 32, 1, 34 } });
            if (TrackPaintUtilShouldPaintSupports(session.MapPosition))
            {
                MetalASupportsPaintSetup(
                    session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 12, height,
                    session.TrackColours[SCHEME_SUPPORTS]);
            }
            break;
        case 2:
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19366), { 0, 0, height },
                { { 0, 6, height }, { 32, 20, 3 } });
            if (TrackPaintUtilShouldPaintSupports(session.MapPosition))
            {
                MetalASupportsPaintSetup(
                    session, MetalSupportType::Fork, MetalSupportPlace::Centre, 12, height,
                    session.TrackColours[SCHEME_SUPPORTS]);
            }
            break;
        case 3:
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19367), { 0, 0, height },
                { { 0, 6, height }, { 32, 20, 3 } });
            if (TrackPaintUtilShouldPaintSupports(session.MapPosition))
            {
                MetalASupportsPaintSetup(
                    session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 12, height,
                    session.TrackColours[SCHEME_SUPPORTS]);
            }
            break;
    }
    if (direction == 0 || direction == 3)
    {
        PaintUtilPushTunnelRotated(session, direction, height - 8, TUNNEL_SQUARE_FLAT);
    }
    else
    {
        PaintUtilPushTunnelRotated(session, direction, height + 8, TUNNEL_14);
    }
    PaintUtilSetSegmentSupportHeight(
        session, PaintUtilRotateSegments(SEGMENT_C4 | SEGMENT_CC | SEGMENT_D0, direction), 0xFFFF, 0);
    PaintUtilSetGeneralSupportHeight(session, height + 40, 0x20);
}

/** rct2: 0x008B09F0 */
static void MineRideTrack25DegUpToRightBank(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    switch (direction)
    {
        case 0:
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19370), { 0, 0, height },
                { { 0, 6, height }, { 32, 20, 3 } });
            if (TrackPaintUtilShouldPaintSupports(session.MapPosition))
            {
                MetalASupportsPaintSetup(
                    session, MetalSupportType::Fork, MetalSupportPlace::Centre, 12, height,
                    session.TrackColours[SCHEME_SUPPORTS]);
            }
            break;
        case 1:
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19371), { 0, 0, height },
                { { 0, 6, height }, { 32, 20, 3 } });
            if (TrackPaintUtilShouldPaintSupports(session.MapPosition))
            {
                MetalASupportsPaintSetup(
                    session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 12, height,
                    session.TrackColours[SCHEME_SUPPORTS]);
            }
            break;
        case 2:
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19372), { 0, 0, height },
                { { 0, 6, height }, { 32, 20, 3 } });
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19374), { 0, 0, height },
                { { 0, 27, height }, { 32, 1, 34 } });
            if (TrackPaintUtilShouldPaintSupports(session.MapPosition))
            {
                MetalASupportsPaintSetup(
                    session, MetalSupportType::Fork, MetalSupportPlace::Centre, 12, height,
                    session.TrackColours[SCHEME_SUPPORTS]);
            }
            break;
        case 3:
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19373), { 0, 0, height },
                { { 0, 6, height }, { 32, 20, 3 } });
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19375), { 0, 0, height },
                { { 0, 27, height }, { 32, 1, 34 } });
            if (TrackPaintUtilShouldPaintSupports(session.MapPosition))
            {
                MetalASupportsPaintSetup(
                    session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 12, height,
                    session.TrackColours[SCHEME_SUPPORTS]);
            }
            break;
    }
    if (direction == 0 || direction == 3)
    {
        PaintUtilPushTunnelRotated(session, direction, height - 8, TUNNEL_SQUARE_FLAT);
    }
    else
    {
        PaintUtilPushTunnelRotated(session, direction, height + 8, TUNNEL_14);
    }
    PaintUtilSetSegmentSupportHeight(
        session, PaintUtilRotateSegments(SEGMENT_C4 | SEGMENT_CC | SEGMENT_D0, direction), 0xFFFF, 0);
    PaintUtilSetGeneralSupportHeight(session, height + 40, 0x20);
}

/** rct2: 0x008B0A00 */
static void MineRideTrackLeftBankTo25DegDown(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    MineRideTrack25DegUpToRightBank(session, ride, trackSequence, (direction + 2) & 3, height, trackElement);
}

/** rct2: 0x008B0A10 */
static void MineRideTrackRightBankTo25DegDown(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    MineRideTrack25DegUpToLeftBank(session, ride, trackSequence, (direction + 2) & 3, height, trackElement);
}

/** rct2: 0x008B0A20 */
static void MineRideTrack25DegDownToLeftBank(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    MineRideTrackRightBankTo25DegUp(session, ride, trackSequence, (direction + 2) & 3, height, trackElement);
}

/** rct2: 0x008B0A30 */
static void MineRideTrack25DegDownToRightBank(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    MineRideTrackLeftBankTo25DegUp(session, ride, trackSequence, (direction + 2) & 3, height, trackElement);
}

/** rct2: 0x008B0A40 */
static void MineRideTrackLeftBank(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    switch (direction)
    {
        case 0:
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19376), { 0, 0, height },
                { { 0, 27, height }, { 32, 1, 26 } });
            if (TrackPaintUtilShouldPaintSupports(session.MapPosition))
            {
                MetalASupportsPaintSetup(
                    session, MetalSupportType::Fork, MetalSupportPlace::Centre, 6, height,
                    session.TrackColours[SCHEME_SUPPORTS]);
            }
            break;
        case 1:
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19377), { 0, 0, height },
                { { 0, 27, height }, { 32, 1, 26 } });
            if (TrackPaintUtilShouldPaintSupports(session.MapPosition))
            {
                MetalASupportsPaintSetup(
                    session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 6, height,
                    session.TrackColours[SCHEME_SUPPORTS]);
            }
            break;
        case 2:
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19378), { 0, 0, height },
                { { 0, 6, height }, { 32, 20, 3 } });
            if (TrackPaintUtilShouldPaintSupports(session.MapPosition))
            {
                MetalASupportsPaintSetup(
                    session, MetalSupportType::Fork, MetalSupportPlace::Centre, 6, height,
                    session.TrackColours[SCHEME_SUPPORTS]);
            }
            break;
        case 3:
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19379), { 0, 0, height },
                { { 0, 6, height }, { 32, 20, 3 } });
            if (TrackPaintUtilShouldPaintSupports(session.MapPosition))
            {
                MetalASupportsPaintSetup(
                    session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 6, height,
                    session.TrackColours[SCHEME_SUPPORTS]);
            }
            break;
    }
    PaintUtilPushTunnelRotated(session, direction, height, TUNNEL_SQUARE_FLAT);
    PaintUtilSetSegmentSupportHeight(
        session, PaintUtilRotateSegments(SEGMENT_C4 | SEGMENT_CC | SEGMENT_D0, direction), 0xFFFF, 0);
    PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
}

/** rct2: 0x008B0A50 */
static void MineRideTrackRightBank(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    MineRideTrackLeftBank(session, ride, trackSequence, (direction + 2) & 3, height, trackElement);
}

/** rct2: 0x008B0A60 */
static void MineRideTrackSBendLeft(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    switch (trackSequence)
    {
        case 0:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19460), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19464), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19463), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19467), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            if (direction == 0 || direction == 3)
            {
                PaintUtilPushTunnelRotated(session, direction, height, TUNNEL_SQUARE_FLAT);
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B4 | SEGMENT_C4 | SEGMENT_CC | SEGMENT_D0, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 1:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19461), { 0, 0, height },
                        { 32, 26, 3 });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::TopLeftSide, 3, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19465), { 0, 0, height },
                        { 32, 26, 3 });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::TopRightSide, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19462), { 0, 0, height },
                        { { 0, 6, height }, { 32, 26, 3 } });
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19466), { 0, 0, height },
                        { { 0, 6, height }, { 32, 26, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session,
                PaintUtilRotateSegments(SEGMENT_B4 | SEGMENT_B8 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_CC | SEGMENT_D0, direction),
                0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 2:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19462), { 0, 0, height },
                        { { 0, 6, height }, { 32, 26, 3 } });
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19466), { 0, 0, height },
                        { { 0, 6, height }, { 32, 26, 3 } });
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19461), { 0, 0, height },
                        { 32, 26, 3 });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::TopLeftSide, 3, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19465), { 0, 0, height },
                        { 32, 26, 3 });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::TopRightSide, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session,
                PaintUtilRotateSegments(SEGMENT_BC | SEGMENT_C0 | SEGMENT_C4 | SEGMENT_CC | SEGMENT_D0 | SEGMENT_D4, direction),
                0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 3:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19463), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19467), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19460), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19464), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            switch (direction)
            {
                case 1:
                    PaintUtilPushTunnelRight(session, height, TUNNEL_SQUARE_FLAT);
                    break;
                case 2:
                    PaintUtilPushTunnelLeft(session, height, TUNNEL_SQUARE_FLAT);
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_C0 | SEGMENT_C4 | SEGMENT_CC | SEGMENT_D0, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
    }
}

/** rct2: 0x008B0A70 */
static void MineRideTrackSBendRight(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    switch (trackSequence)
    {
        case 0:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19468), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19472), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19471), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19475), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            if (direction == 0 || direction == 3)
            {
                PaintUtilPushTunnelRotated(session, direction, height, TUNNEL_SQUARE_FLAT);
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_BC | SEGMENT_C4 | SEGMENT_CC | SEGMENT_D0, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 1:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19469), { 0, 0, height },
                        { { 0, 6, height }, { 32, 26, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::BottomRightSide, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19473), { 0, 0, height },
                        { { 0, 6, height }, { 32, 26, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::BottomLeftSide, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19470), { 0, 0, height },
                        { 32, 26, 3 });
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19474), { 0, 0, height },
                        { 32, 26, 3 });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session,
                PaintUtilRotateSegments(SEGMENT_BC | SEGMENT_C0 | SEGMENT_C4 | SEGMENT_CC | SEGMENT_D0 | SEGMENT_D4, direction),
                0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 2:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19470), { 0, 0, height },
                        { 32, 26, 3 });
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19474), { 0, 0, height },
                        { 32, 26, 3 });
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19469), { 0, 0, height },
                        { { 0, 6, height }, { 32, 26, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::BottomRightSide, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19473), { 0, 0, height },
                        { { 0, 6, height }, { 32, 26, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::BottomLeftSide, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session,
                PaintUtilRotateSegments(SEGMENT_B4 | SEGMENT_B8 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_CC | SEGMENT_D0, direction),
                0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 3:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19471), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19475), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19468), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19472), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            switch (direction)
            {
                case 1:
                    PaintUtilPushTunnelRight(session, height, TUNNEL_SQUARE_FLAT);
                    break;
                case 2:
                    PaintUtilPushTunnelLeft(session, height, TUNNEL_SQUARE_FLAT);
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B8 | SEGMENT_C4 | SEGMENT_CC | SEGMENT_D0, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
    }
}

/** rct2: 0x008B0AB0 */
static void MineRideTrackLeftQuarterTurn3(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    switch (trackSequence)
    {
        case 0:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19397), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19400), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19403), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19394), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            if (direction == 0 || direction == 3)
            {
                PaintUtilPushTunnelRotated(session, direction, height, TUNNEL_SQUARE_FLAT);
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B4 | SEGMENT_C4 | SEGMENT_CC | SEGMENT_D0, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 1:
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 2:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19396), { 0, 0, height },
                        { { 16, 0, height }, { 16, 16, 3 } });
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19399), { 0, 0, height },
                        { { 0, 0, height }, { 16, 16, 3 } });
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19402), { 0, 0, height },
                        { { 0, 16, height }, { 16, 16, 3 } });
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19393), { 0, 0, height },
                        { { 16, 16, height }, { 16, 16, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B8 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_D0, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 3:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19395), { 0, 0, height },
                        { { 6, 0, height }, { 20, 32, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19398), { 0, 0, height },
                        { { 6, 0, height }, { 20, 32, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19401), { 0, 0, height },
                        { { 6, 0, height }, { 20, 32, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19392), { 0, 0, height },
                        { { 6, 0, height }, { 20, 32, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            switch (direction)
            {
                case 2:
                    PaintUtilPushTunnelRight(session, height, TUNNEL_SQUARE_FLAT);
                    break;
                case 3:
                    PaintUtilPushTunnelLeft(session, height, TUNNEL_SQUARE_FLAT);
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_C0 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_D4, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
    }
}

/** rct2: 0x008B0AC0 */
static void MineRideTrackRightQuarterTurn3(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    trackSequence = mapLeftQuarterTurn3TilesToRightQuarterTurn3Tiles[trackSequence];
    MineRideTrackLeftQuarterTurn3(session, ride, trackSequence, (direction - 1) & 3, height, trackElement);
}

/** rct2: 0x008B0AD0 */
static void MineRideTrackLeftQuarterTurn3Bank(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    switch (trackSequence)
    {
        case 0:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19409), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19416), { 0, 0, height },
                        { { 0, 27, height }, { 32, 1, 26 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19412), { 0, 0, height },
                        { { 0, 27, height }, { 32, 1, 26 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19415), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19406), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            if (direction == 0 || direction == 3)
            {
                PaintUtilPushTunnelRotated(session, direction, height, TUNNEL_SQUARE_FLAT);
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B4 | SEGMENT_C4 | SEGMENT_CC | SEGMENT_D0, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 1:
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 2:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19408), { 0, 0, height },
                        { { 16, 0, height }, { 16, 16, 3 } });
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19411), { 0, 0, height },
                        { { 0, 0, height + 27 }, { 16, 16, 1 } });
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19414), { 0, 0, height },
                        { { 0, 16, height }, { 16, 16, 3 } });
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19405), { 0, 0, height },
                        { { 16, 16, height }, { 16, 16, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B8 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_D0, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 3:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19407), { 0, 0, height },
                        { { 6, 0, height }, { 20, 32, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19410), { 0, 0, height },
                        { { 27, 0, height }, { 1, 32, 26 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19413), { 0, 0, height },
                        { { 6, 0, height }, { 20, 32, 3 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19417), { 0, 0, height },
                        { { 27, 0, height }, { 1, 32, 26 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19404), { 0, 0, height },
                        { { 6, 0, height }, { 20, 32, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            switch (direction)
            {
                case 2:
                    PaintUtilPushTunnelRight(session, height, TUNNEL_SQUARE_FLAT);
                    break;
                case 3:
                    PaintUtilPushTunnelLeft(session, height, TUNNEL_SQUARE_FLAT);
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_C0 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_D4, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
    }
}

/** rct2: 0x008B0AE0 */
static void MineRideTrackRightQuarterTurn3Bank(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    trackSequence = mapLeftQuarterTurn3TilesToRightQuarterTurn3Tiles[trackSequence];
    MineRideTrackLeftQuarterTurn3Bank(session, ride, trackSequence, (direction - 1) & 3, height, trackElement);
}

/** rct2: 0x008B0AF0 */
static void MineRideTrackLeftHalfBankedHelixUpSmall(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    switch (trackSequence)
    {
        case 0:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19495), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19502), { 0, 0, height },
                        { { 0, 27, height }, { 32, 1, 26 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 8, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19498), { 0, 0, height },
                        { { 0, 27, height }, { 32, 1, 26 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 8, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19501), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 8, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19492), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 8, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            if (direction == 0 || direction == 3)
            {
                PaintUtilPushTunnelRotated(session, direction, height, TUNNEL_SQUARE_FLAT);
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B4 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_CC | SEGMENT_D0, direction),
                0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 1:
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 2:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19494), { 0, 0, height },
                        { { 16, 0, height }, { 16, 16, 3 } });
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19497), { 0, 0, height },
                        { { 0, 0, height + 27 }, { 16, 16, 1 } });
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19500), { 0, 0, height },
                        { { 0, 16, height }, { 16, 16, 3 } });
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19491), { 0, 0, height },
                        { { 16, 16, height }, { 16, 16, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B8 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_D0, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 3:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19493), { 0, 0, height },
                        { { 6, 0, height }, { 20, 32, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 12, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19496), { 0, 0, height },
                        { { 27, 0, height }, { 1, 32, 26 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 12, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19499), { 0, 0, height },
                        { { 6, 0, height }, { 20, 32, 3 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19503), { 0, 0, height },
                        { { 27, 0, height }, { 1, 32, 26 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 15, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19490), { 0, 0, height },
                        { { 6, 0, height + 8 }, { 20, 32, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 12, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            switch (direction)
            {
                case 2:
                    PaintUtilPushTunnelRight(session, height + 8, TUNNEL_SQUARE_FLAT);
                    break;
                case 3:
                    PaintUtilPushTunnelLeft(session, height + 8, TUNNEL_SQUARE_FLAT);
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_C0 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_D0 | SEGMENT_D4, direction),
                0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 4:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19492), { 0, 0, height },
                        { { 6, 0, height }, { 20, 32, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 8, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19495), { 0, 0, height },
                        { { 6, 0, height }, { 20, 32, 3 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19502), { 0, 0, height },
                        { { 27, 0, height }, { 1, 32, 26 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 8, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19498), { 0, 0, height },
                        { { 27, 0, height }, { 1, 32, 26 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 8, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19501), { 0, 0, height },
                        { { 6, 0, height }, { 20, 32, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 8, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            switch (direction)
            {
                case 0:
                    PaintUtilPushTunnelRight(session, height, TUNNEL_SQUARE_FLAT);
                    break;
                case 1:
                    PaintUtilPushTunnelLeft(session, height, TUNNEL_SQUARE_FLAT);
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B8 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_D0 | SEGMENT_D4, direction),
                0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 5:
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 6:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19491), { 0, 0, height },
                        { { 16, 16, height }, { 16, 16, 3 } });
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19494), { 0, 0, height },
                        { { 0, 16, height }, { 16, 16, 3 } });
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19497), { 0, 0, height },
                        { { 0, 0, height + 27 }, { 16, 16, 1 } });
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19500), { 0, 0, height },
                        { { 16, 0, height }, { 16, 16, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_C0 | SEGMENT_C4 | SEGMENT_D0 | SEGMENT_D4, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 7:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19490), { 0, 0, height },
                        { { 0, 6, height + 8 }, { 32, 20, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 12, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19493), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 12, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19496), { 0, 0, height },
                        { { 0, 27, height }, { 32, 1, 26 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 12, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19499), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19503), { 0, 0, height },
                        { { 0, 27, height }, { 32, 1, 26 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 15, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            if (direction == 0 || direction == 3)
            {
                PaintUtilPushTunnelRotated(session, direction, height + 8, TUNNEL_SQUARE_FLAT);
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_BC | SEGMENT_C4 | SEGMENT_CC | SEGMENT_D0 | SEGMENT_D4, direction),
                0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
    }
}

/** rct2: 0x008B0B00 */
static void MineRideTrackRightHalfBankedHelixUpSmall(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    switch (trackSequence)
    {
        case 0:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19476), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 8, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19479), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 8, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19482), { 0, 0, height },
                        { { 0, 27, height }, { 32, 1, 26 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 9, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19485), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19489), { 0, 0, height },
                        { { 0, 27, height }, { 32, 1, 26 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 11, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            if (direction == 0 || direction == 3)
            {
                PaintUtilPushTunnelRotated(session, direction, height, TUNNEL_SQUARE_FLAT);
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_BC | SEGMENT_C4 | SEGMENT_CC | SEGMENT_D0 | SEGMENT_D4, direction),
                0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 1:
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 2:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19477), { 0, 0, height },
                        { { 16, 16, height }, { 16, 16, 3 } });
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19480), { 0, 0, height },
                        { { 0, 16, height }, { 16, 16, 3 } });
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19483), { 0, 0, height },
                        { { 0, 0, height + 27 }, { 16, 16, 1 } });
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19486), { 0, 0, height },
                        { { 16, 0, height }, { 16, 16, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_C0 | SEGMENT_C4 | SEGMENT_D0 | SEGMENT_D4, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 3:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19478), { 0, 0, height },
                        { { 6, 0, height + 8 }, { 20, 32, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 12, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19481), { 0, 0, height },
                        { { 6, 0, height }, { 20, 32, 3 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19488), { 0, 0, height },
                        { { 27, 0, height }, { 1, 32, 26 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 15, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19484), { 0, 0, height },
                        { { 27, 0, height }, { 1, 32, 26 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 14, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19487), { 0, 0, height },
                        { { 6, 0, height }, { 20, 32, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 12, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            switch (direction)
            {
                case 0:
                    PaintUtilPushTunnelRight(session, height + 8, TUNNEL_SQUARE_FLAT);
                    break;
                case 1:
                    PaintUtilPushTunnelLeft(session, height + 8, TUNNEL_SQUARE_FLAT);
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B8 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_D0 | SEGMENT_D4, direction),
                0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 4:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19479), { 0, 0, height },
                        { { 6, 0, height }, { 20, 32, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 8, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19482), { 0, 0, height },
                        { { 27, 0, height }, { 1, 32, 26 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 9, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19485), { 0, 0, height },
                        { { 6, 0, height }, { 20, 32, 3 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19489), { 0, 0, height },
                        { { 27, 0, height }, { 1, 32, 26 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 11, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19476), { 0, 0, height },
                        { { 6, 0, height }, { 20, 32, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 8, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            switch (direction)
            {
                case 2:
                    PaintUtilPushTunnelRight(session, height, TUNNEL_SQUARE_FLAT);
                    break;
                case 3:
                    PaintUtilPushTunnelLeft(session, height, TUNNEL_SQUARE_FLAT);
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_C0 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_D0 | SEGMENT_D4, direction),
                0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 5:
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 6:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19480), { 0, 0, height },
                        { { 16, 0, height }, { 16, 16, 3 } });
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19483), { 0, 0, height },
                        { { 0, 0, height + 27 }, { 16, 16, 1 } });
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19486), { 0, 0, height },
                        { { 0, 16, height }, { 16, 16, 3 } });
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19477), { 0, 0, height },
                        { { 16, 16, height }, { 16, 16, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B8 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_D0, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 7:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19481), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19488), { 0, 0, height },
                        { { 0, 27, height }, { 32, 1, 26 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 15, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19484), { 0, 0, height },
                        { { 0, 27, height }, { 32, 1, 26 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 14, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19487), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 12, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19478), { 0, 0, height },
                        { { 0, 6, height + 8 }, { 32, 20, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 12, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            if (direction == 0 || direction == 3)
            {
                PaintUtilPushTunnelRotated(session, direction, height + 8, TUNNEL_SQUARE_FLAT);
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B4 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_CC | SEGMENT_D0, direction),
                0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
    }
}

/** rct2: 0x008B0B10 */
static void MineRideTrackLeftHalfBankedHelixDownSmall(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    if (trackSequence >= 4)
    {
        trackSequence -= 4;
        direction = (direction - 1) & 3;
    }
    trackSequence = mapLeftQuarterTurn3TilesToRightQuarterTurn3Tiles[trackSequence];
    MineRideTrackRightHalfBankedHelixUpSmall(session, ride, trackSequence, (direction + 1) & 3, height, trackElement);
}

/** rct2: 0x008B0B20 */
static void MineRideTrackRightHalfBankedHelixDownSmall(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    if (trackSequence >= 4)
    {
        trackSequence -= 4;
        direction = (direction + 1) & 3;
    }
    trackSequence = mapLeftQuarterTurn3TilesToRightQuarterTurn3Tiles[trackSequence];
    MineRideTrackLeftHalfBankedHelixUpSmall(session, ride, trackSequence, (direction - 1) & 3, height, trackElement);
}

/** rct2: 0x008B0B30 */
static void MineRideTrackLeftHalfBankedHelixUpLarge(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    switch (trackSequence)
    {
        case 0:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19535), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19546), { 0, 0, height },
                        { { 0, 27, height }, { 32, 1, 26 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 7, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19540), { 0, 0, height },
                        { { 0, 27, height }, { 32, 1, 26 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 7, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19545), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 7, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19530), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 7, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            if (direction == 0 || direction == 3)
            {
                PaintUtilPushTunnelRotated(session, direction, height, TUNNEL_SQUARE_FLAT);
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B4 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_CC | SEGMENT_D0, direction),
                0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 1:
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 2:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19534), { 0, 0, height },
                        { { 0, 0, height }, { 32, 16, 3 } });
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19539), { 0, 0, height },
                        { { 0, 0, height + 27 }, { 32, 16, 1 } });
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19544), { 0, 0, height },
                        { { 0, 16, height }, { 32, 16, 3 } });
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19529), { 0, 0, height },
                        { { 0, 16, height }, { 32, 16, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session,
                PaintUtilRotateSegments(SEGMENT_B4 | SEGMENT_B8 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_CC | SEGMENT_D0, direction),
                0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 3:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19533), { 0, 0, height },
                        { { 0, 16, height }, { 16, 16, 3 } });
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19538), { 0, 0, height },
                        { { 16, 16, height + 27 }, { 16, 16, 1 } });
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19543), { 0, 0, height },
                        { { 16, 0, height }, { 16, 16, 3 } });
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19528), { 0, 0, height },
                        { { 0, 0, height }, { 16, 16, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_BC | SEGMENT_C4 | SEGMENT_CC | SEGMENT_D4, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 4:
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 5:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19532), { 0, 0, height },
                        { { 16, 0, height }, { 16, 32, 3 } });
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19537), { 0, 0, height },
                        { { 0, 0, height + 27 }, { 16, 32, 1 } });
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19542), { 0, 0, height },
                        { { 0, 0, height }, { 16, 32, 3 } });
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19527), { 0, 0, height },
                        { { 16, 0, height }, { 16, 32, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session,
                PaintUtilRotateSegments(SEGMENT_B8 | SEGMENT_C0 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_D0 | SEGMENT_D4, direction),
                0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 6:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19531), { 0, 0, height },
                        { { 6, 0, height }, { 20, 32, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 13, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19536), { 0, 0, height },
                        { { 27, 0, height }, { 1, 32, 26 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 13, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19541), { 0, 0, height },
                        { { 6, 0, height }, { 20, 32, 3 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19547), { 0, 0, height },
                        { { 27, 0, height }, { 1, 32, 26 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 13, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19526), { 0, 0, height },
                        { { 6, 0, height + 8 }, { 20, 32, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 13, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            switch (direction)
            {
                case 2:
                    PaintUtilPushTunnelRight(session, height + 8, TUNNEL_SQUARE_FLAT);
                    break;
                case 3:
                    PaintUtilPushTunnelLeft(session, height + 8, TUNNEL_SQUARE_FLAT);
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_C0 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_D0 | SEGMENT_D4, direction),
                0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 7:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19530), { 0, 0, height },
                        { { 6, 0, height }, { 20, 32, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 7, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19535), { 0, 0, height },
                        { { 6, 0, height }, { 20, 32, 3 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19546), { 0, 0, height },
                        { { 27, 0, height }, { 1, 32, 26 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 7, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19540), { 0, 0, height },
                        { { 27, 0, height }, { 1, 32, 26 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 7, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19545), { 0, 0, height },
                        { { 6, 0, height }, { 20, 32, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 7, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            switch (direction)
            {
                case 0:
                    PaintUtilPushTunnelRight(session, height, TUNNEL_SQUARE_FLAT);
                    break;
                case 1:
                    PaintUtilPushTunnelLeft(session, height, TUNNEL_SQUARE_FLAT);
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B8 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_D0 | SEGMENT_D4, direction),
                0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 8:
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 9:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19529), { 0, 0, height },
                        { { 16, 0, height }, { 16, 32, 3 } });
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19534), { 0, 0, height },
                        { { 0, 0, height }, { 16, 32, 3 } });
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19539), { 0, 0, height },
                        { { 0, 0, height + 27 }, { 16, 32, 1 } });
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19544), { 0, 0, height },
                        { { 16, 0, height }, { 16, 32, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session,
                PaintUtilRotateSegments(SEGMENT_B8 | SEGMENT_C0 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_D0 | SEGMENT_D4, direction),
                0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 10:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19528), { 0, 0, height },
                        { { 0, 0, height }, { 16, 16, 3 } });
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19533), { 0, 0, height },
                        { { 16, 0, height }, { 16, 16, 3 } });
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19538), { 0, 0, height },
                        { { 16, 16, height + 27 }, { 16, 16, 1 } });
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19543), { 0, 0, height },
                        { { 0, 16, height }, { 16, 16, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B4 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_CC, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 11:
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 12:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19527), { 0, 0, height },
                        { { 0, 16, height }, { 32, 16, 3 } });
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19532), { 0, 0, height },
                        { { 0, 16, height }, { 32, 16, 3 } });
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19537), { 0, 0, height },
                        { { 0, 0, height + 27 }, { 32, 16, 1 } });
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19542), { 0, 0, height },
                        { { 0, 0, height }, { 32, 16, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session,
                PaintUtilRotateSegments(SEGMENT_BC | SEGMENT_C0 | SEGMENT_C4 | SEGMENT_CC | SEGMENT_D0 | SEGMENT_D4, direction),
                0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 13:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19526), { 0, 0, height },
                        { { 0, 6, height + 8 }, { 32, 20, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 13, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19531), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 13, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19536), { 0, 0, height },
                        { { 0, 27, height }, { 32, 1, 26 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 13, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19541), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19547), { 0, 0, height },
                        { { 0, 27, height }, { 32, 1, 26 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 13, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            if (direction == 0 || direction == 3)
            {
                PaintUtilPushTunnelRotated(session, direction, height + 8, TUNNEL_SQUARE_FLAT);
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_BC | SEGMENT_C4 | SEGMENT_CC | SEGMENT_D0 | SEGMENT_D4, direction),
                0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
    }
}

/** rct2: 0x008B0B40 */
static void MineRideTrackRightHalfBankedHelixUpLarge(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    switch (trackSequence)
    {
        case 0:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19504), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 7, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19509), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 7, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19514), { 0, 0, height },
                        { { 0, 27, height }, { 32, 1, 26 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 7, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19519), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19525), { 0, 0, height },
                        { { 0, 27, height }, { 32, 1, 26 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 7, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            if (direction == 0 || direction == 3)
            {
                PaintUtilPushTunnelRotated(session, direction, height, TUNNEL_SQUARE_FLAT);
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_BC | SEGMENT_C4 | SEGMENT_CC | SEGMENT_D0 | SEGMENT_D4, direction),
                0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 1:
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 2:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19505), { 0, 0, height },
                        { { 0, 16, height }, { 32, 16, 3 } });
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19510), { 0, 0, height },
                        { { 0, 16, height }, { 32, 16, 3 } });
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19515), { 0, 0, height },
                        { { 0, 0, height + 27 }, { 32, 16, 1 } });
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19520), { 0, 0, height },
                        { { 0, 0, height }, { 32, 16, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session,
                PaintUtilRotateSegments(SEGMENT_BC | SEGMENT_C0 | SEGMENT_C4 | SEGMENT_CC | SEGMENT_D0 | SEGMENT_D4, direction),
                0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 3:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19506), { 0, 0, height },
                        { { 0, 0, height }, { 16, 16, 3 } });
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19511), { 0, 0, height },
                        { { 16, 0, height }, { 16, 16, 3 } });
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19516), { 0, 0, height },
                        { { 16, 16, height + 27 }, { 16, 16, 1 } });
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19521), { 0, 0, height },
                        { { 0, 16, height }, { 16, 16, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B4 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_CC, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 4:
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 5:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19507), { 0, 0, height },
                        { { 16, 0, height }, { 16, 32, 3 } });
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19512), { 0, 0, height },
                        { { 0, 0, height }, { 16, 32, 3 } });
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19517), { 0, 0, height },
                        { { 0, 0, height + 27 }, { 16, 32, 1 } });
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19522), { 0, 0, height },
                        { { 16, 0, height }, { 16, 32, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session,
                PaintUtilRotateSegments(SEGMENT_B8 | SEGMENT_C0 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_D0 | SEGMENT_D4, direction),
                0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 6:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19508), { 0, 0, height },
                        { { 6, 0, height + 8 }, { 20, 32, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 13, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19513), { 0, 0, height },
                        { { 6, 0, height }, { 20, 32, 3 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19524), { 0, 0, height },
                        { { 27, 0, height }, { 1, 32, 26 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 13, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19518), { 0, 0, height },
                        { { 27, 0, height }, { 1, 32, 26 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 13, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19523), { 0, 0, height },
                        { { 6, 0, height }, { 20, 32, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 13, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            switch (direction)
            {
                case 0:
                    PaintUtilPushTunnelRight(session, height + 8, TUNNEL_SQUARE_FLAT);
                    break;
                case 1:
                    PaintUtilPushTunnelLeft(session, height + 8, TUNNEL_SQUARE_FLAT);
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B8 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_D0 | SEGMENT_D4, direction),
                0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 7:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19509), { 0, 0, height },
                        { { 6, 0, height }, { 20, 32, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 7, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19514), { 0, 0, height },
                        { { 27, 0, height }, { 1, 32, 26 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 7, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19519), { 0, 0, height },
                        { { 6, 0, height }, { 20, 32, 3 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19525), { 0, 0, height },
                        { { 27, 0, height }, { 1, 32, 26 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 7, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19504), { 0, 0, height },
                        { { 6, 0, height }, { 20, 32, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 7, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            switch (direction)
            {
                case 2:
                    PaintUtilPushTunnelRight(session, height, TUNNEL_SQUARE_FLAT);
                    break;
                case 3:
                    PaintUtilPushTunnelLeft(session, height, TUNNEL_SQUARE_FLAT);
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_C0 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_D0 | SEGMENT_D4, direction),
                0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 8:
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 9:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19510), { 0, 0, height },
                        { { 16, 0, height }, { 16, 32, 3 } });
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19515), { 0, 0, height },
                        { { 0, 0, height + 27 }, { 16, 32, 1 } });
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19520), { 0, 0, height },
                        { { 0, 0, height }, { 16, 32, 3 } });
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19505), { 0, 0, height },
                        { { 16, 0, height }, { 16, 32, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session,
                PaintUtilRotateSegments(SEGMENT_B8 | SEGMENT_C0 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_D0 | SEGMENT_D4, direction),
                0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 10:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19511), { 0, 0, height },
                        { { 0, 16, height }, { 16, 16, 3 } });
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19516), { 0, 0, height },
                        { { 16, 16, height + 27 }, { 16, 16, 1 } });
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19521), { 0, 0, height },
                        { { 16, 0, height }, { 16, 16, 3 } });
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19506), { 0, 0, height },
                        { { 0, 0, height }, { 16, 16, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_BC | SEGMENT_C4 | SEGMENT_CC | SEGMENT_D4, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 11:
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 12:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19512), { 0, 0, height },
                        { { 0, 0, height }, { 32, 16, 3 } });
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19517), { 0, 0, height },
                        { { 0, 0, height + 27 }, { 32, 16, 1 } });
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19522), { 0, 0, height },
                        { { 0, 16, height }, { 32, 16, 3 } });
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19507), { 0, 0, height },
                        { { 0, 16, height }, { 32, 16, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session,
                PaintUtilRotateSegments(SEGMENT_B4 | SEGMENT_B8 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_CC | SEGMENT_D0, direction),
                0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 13:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19513), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19524), { 0, 0, height },
                        { { 0, 27, height }, { 32, 1, 26 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 13, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19518), { 0, 0, height },
                        { { 0, 27, height }, { 32, 1, 26 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 13, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19523), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 13, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19508), { 0, 0, height },
                        { { 0, 6, height + 8 }, { 32, 20, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 13, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            if (direction == 0 || direction == 3)
            {
                PaintUtilPushTunnelRotated(session, direction, height + 8, TUNNEL_SQUARE_FLAT);
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B4 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_CC | SEGMENT_D0, direction),
                0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
    }
}

/** rct2: 0x008B0B50 */
static void MineRideTrackLeftHalfBankedHelixDownLarge(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    if (trackSequence >= 7)
    {
        trackSequence -= 7;
        direction = (direction - 1) & 3;
    }
    trackSequence = mapLeftQuarterTurn5TilesToRightQuarterTurn5Tiles[trackSequence];
    MineRideTrackRightHalfBankedHelixUpLarge(session, ride, trackSequence, (direction + 1) & 3, height, trackElement);
}

/** rct2: 0x008B0B60 */
static void MineRideTrackRightHalfBankedHelixDownLarge(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    if (trackSequence >= 7)
    {
        trackSequence -= 7;
        direction = (direction + 1) & 3;
    }
    trackSequence = mapLeftQuarterTurn5TilesToRightQuarterTurn5Tiles[trackSequence];
    MineRideTrackLeftHalfBankedHelixUpLarge(session, ride, trackSequence, (direction - 1) & 3, height, trackElement);
}

/** rct2: 0x008B0B70 */
static void MineRideTrackOnRidePhoto(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    PaintAddImageAsParentRotated(
        session, direction, ImageId(SPR_STATION_BASE_D, COLOUR_BLACK), { 0, 0, height }, { 32, 32, 1 });
    DrawSupportsSideBySide(session, direction, height, session.TrackColours[SCHEME_SUPPORTS], MetalSupportType::Fork, 6);

    switch (direction)
    {
        case 0:
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19338), { 0, 0, height },
                { { 0, 6, height + 3 }, { 32, 20, 0 } });
            break;
        case 1:
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19339), { 0, 0, height },
                { { 0, 6, height + 3 }, { 32, 20, 0 } });
            break;
        case 2:
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19338), { 0, 0, height },
                { { 0, 6, height + 3 }, { 32, 20, 0 } });
            break;
        case 3:
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19339), { 0, 0, height },
                { { 0, 6, height + 3 }, { 32, 20, 0 } });
            break;
    }
    TrackPaintUtilOnridePhotoPaint(session, direction, height + 3, trackElement);
    PaintUtilPushTunnelRotated(session, direction, height, TUNNEL_SQUARE_FLAT);
    PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
    PaintUtilSetGeneralSupportHeight(session, height + 48, 0x20);
}

/** rct2: 0x008B0B90 */
static void MineRideTrackLeftEighthToDiag(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    switch (trackSequence)
    {
        case 0:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19564), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19568), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19572), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19576), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            if (direction == 0 || direction == 3)
            {
                PaintUtilPushTunnelRotated(session, direction, height, TUNNEL_SQUARE_FLAT);
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_C4 | SEGMENT_CC | SEGMENT_D0, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 1:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19565), { 0, 0, height },
                        { { 0, 0, height }, { 32, 16, 3 } });
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19569), { 0, 0, height },
                        { { 0, 0, height }, { 34, 16, 3 } });
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19573), { 0, 0, height },
                        { { 0, 16, height }, { 32, 16, 3 } });
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19577), { 0, 0, height },
                        { { 0, 16, height }, { 32, 16, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session,
                PaintUtilRotateSegments(SEGMENT_B4 | SEGMENT_B8 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_CC | SEGMENT_D0, direction),
                0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 2:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19566), { 0, 0, height },
                        { { 0, 16, height }, { 16, 16, 3 } });
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19570), { 0, 0, height },
                        { { 16, 16, height }, { 16, 16, 3 } });
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19574), { 0, 0, height },
                        { { 16, 0, height }, { 16, 16, 3 } });
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19578), { 0, 0, height },
                        { { 0, 0, height }, { 16, 16, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_BC | SEGMENT_C0 | SEGMENT_C4 | SEGMENT_CC | SEGMENT_D4, direction),
                0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 3:
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B8 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_D0, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 4:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19567), { 0, 0, height },
                        { { 16, 16, height }, { 16, 16, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::BottomCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19571), { 0, 0, height },
                        { { 0, 16, height }, { 16, 18, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::LeftCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19575), { 0, 0, height },
                        { { 0, 0, height }, { 16, 16, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::TopCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19579), { 0, 0, height },
                        { { 16, 0, height }, { 16, 16, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::RightCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session,
                PaintUtilRotateSegments(SEGMENT_C0 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_CC | SEGMENT_D0 | SEGMENT_D4, direction),
                0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
    }
}

/** rct2: 0x008B0BA0 */
static void MineRideTrackRightEighthToDiag(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    switch (trackSequence)
    {
        case 0:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19548), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19552), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19556), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19560), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            if (direction == 0 || direction == 3)
            {
                PaintUtilPushTunnelRotated(session, direction, height, TUNNEL_SQUARE_FLAT);
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_C4 | SEGMENT_CC | SEGMENT_D0, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 1:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19549), { 0, 0, height },
                        { { 0, 16, height }, { 32, 16, 3 } });
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19553), { 0, 0, height },
                        { { 0, 16, height }, { 32, 16, 3 } });
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19557), { 0, 0, height },
                        { { 0, 0, height }, { 34, 16, 3 } });
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19561), { 0, 0, height },
                        { { 0, 0, height }, { 32, 16, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session,
                PaintUtilRotateSegments(SEGMENT_BC | SEGMENT_C0 | SEGMENT_C4 | SEGMENT_CC | SEGMENT_D0 | SEGMENT_D4, direction),
                0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 2:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19550), { 0, 0, height },
                        { { 0, 0, height }, { 16, 16, 3 } });
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19554), { 0, 0, height },
                        { { 16, 0, height }, { 16, 16, 3 } });
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19558), { 0, 0, height },
                        { { 4, 4, height }, { 28, 28, 3 } });
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19562), { 0, 0, height },
                        { { 0, 16, height }, { 16, 16, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B4 | SEGMENT_B8 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_CC, direction),
                0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 3:
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_C0 | SEGMENT_C4 | SEGMENT_D0 | SEGMENT_D4, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 4:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19551), { 0, 0, height },
                        { { 16, 0, height }, { 16, 16, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::LeftCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19555), { 0, 0, height },
                        { { 0, 0, height }, { 16, 16, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::TopCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19559), { 0, 0, height },
                        { { 0, 16, height }, { 16, 18, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::RightCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19563), { 0, 0, height },
                        { { 16, 16, height }, { 16, 16, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::BottomCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session,
                PaintUtilRotateSegments(SEGMENT_B8 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_CC | SEGMENT_D0 | SEGMENT_D4, direction),
                0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
    }
}

/** rct2: 0x008B0BB0 */
static void MineRideTrackLeftEighthToOrthogonal(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    trackSequence = mapLeftEighthTurnToOrthogonal[trackSequence];
    MineRideTrackRightEighthToDiag(session, ride, trackSequence, (direction + 2) & 3, height, trackElement);
}

/** rct2: 0x008B0BC0 */
static void MineRideTrackRightEighthToOrthogonal(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    trackSequence = mapLeftEighthTurnToOrthogonal[trackSequence];
    MineRideTrackLeftEighthToDiag(session, ride, trackSequence, (direction + 3) & 3, height, trackElement);
}

/** rct2: 0x008B0BD0 */
static void MineRideTrackLeftEighthBankToDiag(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    switch (trackSequence)
    {
        case 0:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19596), { 0, 0, height },
                        { { 0, 27, height }, { 32, 1, 26 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19600), { 0, 0, height },
                        { { 0, 27, height }, { 32, 1, 26 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19604), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19608), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            if (direction == 0 || direction == 3)
            {
                PaintUtilPushTunnelRotated(session, direction, height, TUNNEL_SQUARE_FLAT);
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_C4 | SEGMENT_CC | SEGMENT_D0, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 1:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19597), { 0, 0, height },
                        { { 0, 0, height }, { 32, 16, 3 } });
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19601), { 0, 0, height },
                        { { 0, 0, height + 27 }, { 34, 16, 0 } });
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19605), { 0, 0, height },
                        { { 0, 16, height }, { 32, 16, 3 } });
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19609), { 0, 0, height },
                        { { 0, 16, height }, { 32, 16, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session,
                PaintUtilRotateSegments(SEGMENT_B4 | SEGMENT_B8 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_CC | SEGMENT_D0, direction),
                0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 2:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19598), { 0, 0, height },
                        { { 0, 16, height }, { 16, 16, 3 } });
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19602), { 0, 0, height },
                        { { 16, 16, height + 27 }, { 16, 16, 0 } });
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19606), { 0, 0, height },
                        { { 16, 0, height }, { 16, 16, 3 } });
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19610), { 0, 0, height },
                        { { 0, 0, height }, { 16, 16, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_BC | SEGMENT_C0 | SEGMENT_C4 | SEGMENT_CC | SEGMENT_D4, direction),
                0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 3:
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B8 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_D0, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 4:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19599), { 0, 0, height },
                        { { 16, 16, height }, { 16, 16, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::BottomCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19603), { 0, 0, height },
                        { { 0, 16, height + 27 }, { 16, 18, 0 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::LeftCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19607), { 0, 0, height },
                        { { 0, 0, height }, { 16, 16, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::TopCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19611), { 0, 0, height },
                        { { 16, 0, height }, { 16, 16, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::RightCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session,
                PaintUtilRotateSegments(SEGMENT_C0 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_CC | SEGMENT_D0 | SEGMENT_D4, direction),
                0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
    }
}

/** rct2: 0x008B0BE0 */
static void MineRideTrackRightEighthBankToDiag(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    switch (trackSequence)
    {
        case 0:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19580), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19584), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19588), { 0, 0, height },
                        { { 0, 27, height }, { 32, 1, 26 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19592), { 0, 0, height },
                        { { 0, 27, height }, { 32, 1, 26 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::Centre, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            if (direction == 0 || direction == 3)
            {
                PaintUtilPushTunnelRotated(session, direction, height, TUNNEL_SQUARE_FLAT);
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_C4 | SEGMENT_CC | SEGMENT_D0, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 1:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19581), { 0, 0, height },
                        { { 0, 16, height }, { 32, 16, 3 } });
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19585), { 0, 0, height },
                        { { 0, 16, height }, { 32, 16, 3 } });
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19589), { 0, 0, height },
                        { { 0, 0, height + 27 }, { 34, 16, 0 } });
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19593), { 0, 0, height },
                        { { 0, 0, height }, { 32, 16, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session,
                PaintUtilRotateSegments(SEGMENT_BC | SEGMENT_C0 | SEGMENT_C4 | SEGMENT_CC | SEGMENT_D0 | SEGMENT_D4, direction),
                0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 2:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19582), { 0, 0, height },
                        { { 0, 0, height }, { 16, 16, 3 } });
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19586), { 0, 0, height },
                        { { 16, 0, height }, { 16, 16, 3 } });
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19590), { 0, 0, height },
                        { { 4, 4, height + 27 }, { 28, 28, 0 } });
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19594), { 0, 0, height },
                        { { 0, 16, height }, { 16, 16, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B4 | SEGMENT_B8 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_CC, direction),
                0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 3:
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_C0 | SEGMENT_C4 | SEGMENT_D0 | SEGMENT_D4, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 4:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19583), { 0, 0, height },
                        { { 16, 0, height }, { 16, 16, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::LeftCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19587), { 0, 0, height },
                        { { 0, 0, height }, { 16, 16, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::TopCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19591), { 0, 0, height },
                        { { 0, 16, height + 27 }, { 16, 18, 0 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::RightCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19595), { 0, 0, height },
                        { { 16, 16, height }, { 16, 16, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::BottomCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session,
                PaintUtilRotateSegments(SEGMENT_B8 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_CC | SEGMENT_D0 | SEGMENT_D4, direction),
                0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
    }
}

/** rct2: 0x008B0BF0 */
static void MineRideTrackLeftEighthBankToOrthogonal(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    trackSequence = mapLeftEighthTurnToOrthogonal[trackSequence];
    MineRideTrackRightEighthBankToDiag(session, ride, trackSequence, (direction + 2) & 3, height, trackElement);
}

/** rct2: 0x008B0C00 */
static void MineRideTrackRightEighthBankToOrthogonal(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    trackSequence = mapLeftEighthTurnToOrthogonal[trackSequence];
    MineRideTrackLeftEighthBankToDiag(session, ride, trackSequence, (direction + 3) & 3, height, trackElement);
}

/** rct2: 0x008B0B80 */
static void MineRideTrackDiagFlat(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    switch (trackSequence)
    {
        case 0:
            switch (direction)
            {
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19615), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_BC | SEGMENT_C4 | SEGMENT_CC | SEGMENT_D4, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 1:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19612), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B4 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_CC, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 2:
            switch (direction)
            {
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19614), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_C0 | SEGMENT_C4 | SEGMENT_D0 | SEGMENT_D4, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 3:
            switch (direction)
            {
                case 0:
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::LeftCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19613), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::TopCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::RightCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::BottomCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B8 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_D0, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
    }
}

/** rct2: 0x008B0C30 */
static void MineRideTrackDiag25DegUp(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    switch (trackSequence)
    {
        case 0:
            switch (direction)
            {
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19627), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_BC | SEGMENT_C4 | SEGMENT_CC | SEGMENT_D4, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 56, 0x20);
            break;
        case 1:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19624), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B4 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_CC, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 56, 0x20);
            break;
        case 2:
            switch (direction)
            {
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19626), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_C0 | SEGMENT_C4 | SEGMENT_D0 | SEGMENT_D4, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 56, 0x20);
            break;
        case 3:
            switch (direction)
            {
                case 0:
                    MetalBSupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::LeftCorner, 14, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19625), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    MetalBSupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::TopCorner, 14, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    MetalBSupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::RightCorner, 14, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    MetalBSupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::BottomCorner, 14, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B8 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_D0, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 56, 0x20);
            break;
    }
}

/** rct2: 0x008B0C10 */
static void MineRideTrackDiagFlatTo25DegUp(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    switch (trackSequence)
    {
        case 0:
            switch (direction)
            {
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19619), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_BC | SEGMENT_C4 | SEGMENT_CC | SEGMENT_D4, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 48, 0x20);
            break;
        case 1:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19616), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B4 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_CC, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 48, 0x20);
            break;
        case 2:
            switch (direction)
            {
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19618), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_C0 | SEGMENT_C4 | SEGMENT_D0 | SEGMENT_D4, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 48, 0x20);
            break;
        case 3:
            switch (direction)
            {
                case 0:
                    MetalBSupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::LeftCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19617), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    MetalBSupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::TopCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    MetalBSupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::RightCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    MetalBSupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::BottomCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B8 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_D0, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 48, 0x20);
            break;
    }
}

/** rct2: 0x008B0C20 */
static void MineRideTrackDiag25DegUpToFlat(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    switch (trackSequence)
    {
        case 0:
            switch (direction)
            {
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19623), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_BC | SEGMENT_C4 | SEGMENT_CC | SEGMENT_D4, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 56, 0x20);
            break;
        case 1:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19620), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B4 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_CC, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 56, 0x20);
            break;
        case 2:
            switch (direction)
            {
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19622), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_C0 | SEGMENT_C4 | SEGMENT_D0 | SEGMENT_D4, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 56, 0x20);
            break;
        case 3:
            switch (direction)
            {
                case 0:
                    MetalBSupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::LeftCorner, 10, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19621), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    MetalBSupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::TopCorner, 10, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    MetalBSupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::RightCorner, 10, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    MetalBSupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::BottomCorner, 10, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B8 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_D0, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 56, 0x20);
            break;
    }
}

/** rct2: 0x008B0C60 */
static void MineRideTrackDiag25DegDown(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    switch (trackSequence)
    {
        case 0:
            switch (direction)
            {
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19625), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_BC | SEGMENT_C4 | SEGMENT_CC | SEGMENT_D4, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 56, 0x20);
            break;
        case 1:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19626), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B4 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_CC, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 56, 0x20);
            break;
        case 2:
            switch (direction)
            {
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19624), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_C0 | SEGMENT_C4 | SEGMENT_D0 | SEGMENT_D4, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 56, 0x20);
            break;
        case 3:
            switch (direction)
            {
                case 0:
                    MetalBSupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::LeftCorner, 14, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19627), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    MetalBSupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::TopCorner, 14, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    MetalBSupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::RightCorner, 14, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    MetalBSupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::BottomCorner, 14, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B8 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_D0, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 56, 0x20);
            break;
    }
}

/** rct2: 0x008B0C40 */
static void MineRideTrackDiagFlatTo25DegDown(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    switch (trackSequence)
    {
        case 0:
            switch (direction)
            {
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19621), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_BC | SEGMENT_C4 | SEGMENT_CC | SEGMENT_D4, direction), 0xFFFF, 0);
            break;
        case 1:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19622), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B4 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_CC, direction), 0xFFFF, 0);
            break;
        case 2:
            switch (direction)
            {
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19620), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_C0 | SEGMENT_C4 | SEGMENT_D0 | SEGMENT_D4, direction), 0xFFFF, 0);
            break;
        case 3:
            switch (direction)
            {
                case 0:
                    MetalBSupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::LeftCorner, 10, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19623), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    MetalBSupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::TopCorner, 10, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    MetalBSupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::RightCorner, 10, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    MetalBSupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::BottomCorner, 10, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B8 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_D0, direction), 0xFFFF, 0);
            break;
    }

    PaintUtilSetGeneralSupportHeight(session, height + 56, 0x20);
}

/** rct2: 0x008B0C50 */
static void MineRideTrackDiag25DegDownToFlat(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    switch (trackSequence)
    {
        case 0:
            switch (direction)
            {
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19617), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_BC | SEGMENT_C4 | SEGMENT_CC | SEGMENT_D4, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 48, 0x20);
            break;
        case 1:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19618), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B4 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_CC, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 48, 0x20);
            break;
        case 2:
            switch (direction)
            {
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19616), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_C0 | SEGMENT_C4 | SEGMENT_D0 | SEGMENT_D4, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 48, 0x20);
            break;
        case 3:
            switch (direction)
            {
                case 0:
                    MetalBSupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::LeftCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19619), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    MetalBSupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::TopCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    MetalBSupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::RightCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    MetalBSupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::BottomCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B8 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_D0, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 48, 0x20);
            break;
    }
}

/** rct2: 0x008B0C90 */
static void MineRideTrackDiagFlatToLeftBank(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    switch (trackSequence)
    {
        case 0:
            switch (direction)
            {
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19635), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_BC | SEGMENT_C4 | SEGMENT_CC | SEGMENT_D4, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 1:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19632), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19636), { -16, -16, height },
                        { { -16, -16, height + 27 }, { 32, 32, 0 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B4 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_CC, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 2:
            switch (direction)
            {
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19634), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_C0 | SEGMENT_C4 | SEGMENT_D0 | SEGMENT_D4, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 3:
            switch (direction)
            {
                case 0:
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::LeftCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19633), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::TopCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::RightCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::BottomCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B8 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_D0, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
    }
}

/** rct2: 0x008B0CA0 */
static void MineRideTrackDiagFlatToRightBank(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    switch (trackSequence)
    {
        case 0:
            switch (direction)
            {
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19640), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_BC | SEGMENT_C4 | SEGMENT_CC | SEGMENT_D4, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 1:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19637), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B4 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_CC, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 2:
            switch (direction)
            {
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19639), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19641), { -16, -16, height },
                        { { -16, -16, height + 27 }, { 32, 32, 0 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_C0 | SEGMENT_C4 | SEGMENT_D0 | SEGMENT_D4, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 3:
            switch (direction)
            {
                case 0:
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::LeftCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19638), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::TopCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::RightCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::BottomCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B8 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_D0, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
    }
}

/** rct2: 0x008B0CB0 */
static void MineRideTrackDiagLeftBankToFlat(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    switch (trackSequence)
    {
        case 0:
            switch (direction)
            {
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19638), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_BC | SEGMENT_C4 | SEGMENT_CC | SEGMENT_D4, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 1:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19639), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19641), { -16, -16, height },
                        { { -16, -16, height + 27 }, { 32, 32, 0 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B4 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_CC, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 2:
            switch (direction)
            {
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19637), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_C0 | SEGMENT_C4 | SEGMENT_D0 | SEGMENT_D4, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 3:
            switch (direction)
            {
                case 0:
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::LeftCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19640), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::TopCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::RightCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::BottomCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B8 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_D0, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
    }
}

/** rct2: 0x008B0CC0 */
static void MineRideTrackDiagRightBankToFlat(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    switch (trackSequence)
    {
        case 0:
            switch (direction)
            {
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19633), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_BC | SEGMENT_C4 | SEGMENT_CC | SEGMENT_D4, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 1:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19634), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B4 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_CC, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 2:
            switch (direction)
            {
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19632), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19636), { -16, -16, height },
                        { { -16, -16, height + 27 }, { 32, 32, 0 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_C0 | SEGMENT_C4 | SEGMENT_D0 | SEGMENT_D4, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 3:
            switch (direction)
            {
                case 0:
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::LeftCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19635), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::TopCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::RightCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::BottomCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B8 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_D0, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
    }
}

/** rct2: 0x008B0CF0 */
static void MineRideTrackDiagLeftBankTo25DegUp(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    switch (trackSequence)
    {
        case 0:
            switch (direction)
            {
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19655), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_BC | SEGMENT_C4 | SEGMENT_CC | SEGMENT_D4, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 48, 0x20);
            break;
        case 1:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19652), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19656), { -16, -16, height },
                        { { -16, -16, height + 35 }, { 32, 32, 0 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B4 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_CC, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 48, 0x20);
            break;
        case 2:
            switch (direction)
            {
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19654), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_C0 | SEGMENT_C4 | SEGMENT_D0 | SEGMENT_D4, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 48, 0x20);
            break;
        case 3:
            switch (direction)
            {
                case 0:
                    MetalBSupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::LeftCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19653), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    MetalBSupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::TopCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    MetalBSupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::RightCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    MetalBSupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::BottomCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B8 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_D0, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 48, 0x20);
            break;
    }
}

/** rct2: 0x008B0D00 */
static void MineRideTrackDiagRightBankTo25DegUp(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    switch (trackSequence)
    {
        case 0:
            switch (direction)
            {
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19660), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_BC | SEGMENT_C4 | SEGMENT_CC | SEGMENT_D4, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 48, 0x20);
            break;
        case 1:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19657), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B4 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_CC, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 48, 0x20);
            break;
        case 2:
            switch (direction)
            {
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19659), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19661), { -16, -16, height },
                        { { -16, -16, height + 35 }, { 32, 32, 0 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_C0 | SEGMENT_C4 | SEGMENT_D0 | SEGMENT_D4, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 48, 0x20);
            break;
        case 3:
            switch (direction)
            {
                case 0:
                    MetalBSupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::LeftCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19658), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    MetalBSupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::TopCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    MetalBSupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::RightCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    MetalBSupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::BottomCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B8 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_D0, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 48, 0x20);
            break;
    }
}

/** rct2: 0x008B0CD0 */
static void MineRideTrackDiag25DegUpToLeftBank(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    switch (trackSequence)
    {
        case 0:
            switch (direction)
            {
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19645), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_BC | SEGMENT_C4 | SEGMENT_CC | SEGMENT_D4, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 56, 0x20);
            break;
        case 1:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19642), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19646), { -16, -16, height },
                        { { -16, -16, height + 35 }, { 32, 32, 0 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B4 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_CC, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 56, 0x20);
            break;
        case 2:
            switch (direction)
            {
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19644), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_C0 | SEGMENT_C4 | SEGMENT_D0 | SEGMENT_D4, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 56, 0x20);
            break;
        case 3:
            switch (direction)
            {
                case 0:
                    MetalBSupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::LeftCorner, 10, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19643), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    MetalBSupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::TopCorner, 10, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    MetalBSupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::RightCorner, 10, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    MetalBSupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::BottomCorner, 10, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B8 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_D0, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 56, 0x20);
            break;
    }
}

/** rct2: 0x008B0CE0 */
static void MineRideTrackDiag25DegUpToRightBank(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    switch (trackSequence)
    {
        case 0:
            switch (direction)
            {
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19650), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_BC | SEGMENT_C4 | SEGMENT_CC | SEGMENT_D4, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 56, 0x20);
            break;
        case 1:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19647), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B4 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_CC, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 56, 0x20);
            break;
        case 2:
            switch (direction)
            {
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19649), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19651), { -16, -16, height },
                        { { -16, -16, height + 35 }, { 32, 32, 0 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_C0 | SEGMENT_C4 | SEGMENT_D0 | SEGMENT_D4, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 56, 0x20);
            break;
        case 3:
            switch (direction)
            {
                case 0:
                    MetalBSupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::LeftCorner, 10, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19648), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    MetalBSupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::TopCorner, 10, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    MetalBSupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::RightCorner, 10, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    MetalBSupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::BottomCorner, 10, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B8 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_D0, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 56, 0x20);
            break;
    }
}

/** rct2: 0x008B0D10 */
static void MineRideTrackDiagLeftBankTo25DegDown(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    switch (trackSequence)
    {
        case 0:
            switch (direction)
            {
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19648), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_BC | SEGMENT_C4 | SEGMENT_CC | SEGMENT_D4, direction), 0xFFFF, 0);
            break;
        case 1:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19649), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19651), { -16, -16, height },
                        { { -16, -16, height + 35 }, { 32, 32, 0 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B4 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_CC, direction), 0xFFFF, 0);
            break;
        case 2:
            switch (direction)
            {
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19647), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_C0 | SEGMENT_C4 | SEGMENT_D0 | SEGMENT_D4, direction), 0xFFFF, 0);
            break;
        case 3:
            switch (direction)
            {
                case 0:
                    MetalBSupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::LeftCorner, 10, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19650), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    MetalBSupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::TopCorner, 10, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    MetalBSupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::RightCorner, 10, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    MetalBSupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::BottomCorner, 10, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B8 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_D0, direction), 0xFFFF, 0);
            break;
    }

    PaintUtilSetGeneralSupportHeight(session, height + 56, 0x20);
}

/** rct2: 0x008B0D20 */
static void MineRideTrackDiagRightBankTo25DegDown(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    switch (trackSequence)
    {
        case 0:
            switch (direction)
            {
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19643), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_BC | SEGMENT_C4 | SEGMENT_CC | SEGMENT_D4, direction), 0xFFFF, 0);
            break;
        case 1:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19644), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B4 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_CC, direction), 0xFFFF, 0);
            break;
        case 2:
            switch (direction)
            {
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19642), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19646), { -16, -16, height },
                        { { -16, -16, height + 35 }, { 32, 32, 0 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_C0 | SEGMENT_C4 | SEGMENT_D0 | SEGMENT_D4, direction), 0xFFFF, 0);
            break;
        case 3:
            switch (direction)
            {
                case 0:
                    MetalBSupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::LeftCorner, 10, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19645), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    MetalBSupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::TopCorner, 10, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    MetalBSupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::RightCorner, 10, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    MetalBSupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::BottomCorner, 10, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B8 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_D0, direction), 0xFFFF, 0);
            break;
    }

    PaintUtilSetGeneralSupportHeight(session, height + 56, 0x20);
}

/** rct2: 0x008B0D30 */
static void MineRideTrackDiag25DegDownToLeftBank(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    switch (trackSequence)
    {
        case 0:
            switch (direction)
            {
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19658), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_BC | SEGMENT_C4 | SEGMENT_CC | SEGMENT_D4, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 48, 0x20);
            break;
        case 1:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19659), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19661), { -16, -16, height },
                        { { -16, -16, height + 35 }, { 32, 32, 0 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B4 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_CC, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 48, 0x20);
            break;
        case 2:
            switch (direction)
            {
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19657), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_C0 | SEGMENT_C4 | SEGMENT_D0 | SEGMENT_D4, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 48, 0x20);
            break;
        case 3:
            switch (direction)
            {
                case 0:
                    MetalBSupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::LeftCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19660), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    MetalBSupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::TopCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    MetalBSupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::RightCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    MetalBSupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::BottomCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B8 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_D0, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 48, 0x20);
            break;
    }
}

/** rct2: 0x008B0D40 */
static void MineRideTrackDiag25DegDownToRightBank(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    switch (trackSequence)
    {
        case 0:
            switch (direction)
            {
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19653), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_BC | SEGMENT_C4 | SEGMENT_CC | SEGMENT_D4, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 48, 0x20);
            break;
        case 1:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19654), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B4 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_CC, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 48, 0x20);
            break;
        case 2:
            switch (direction)
            {
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19652), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19656), { -16, -16, height },
                        { { -16, -16, height + 35 }, { 32, 32, 0 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_C0 | SEGMENT_C4 | SEGMENT_D0 | SEGMENT_D4, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 48, 0x20);
            break;
        case 3:
            switch (direction)
            {
                case 0:
                    MetalBSupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::LeftCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19655), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    MetalBSupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::TopCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    MetalBSupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::RightCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    MetalBSupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::BottomCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B8 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_D0, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 48, 0x20);
            break;
    }
}

/** rct2: 0x008B0C70 */
static void MineRideTrackDiagLeftBank(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    switch (trackSequence)
    {
        case 0:
            switch (direction)
            {
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19631), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_BC | SEGMENT_C4 | SEGMENT_CC | SEGMENT_D4, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 1:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19628), { -16, -16, height },
                        { { -16, -16, height + 27 }, { 32, 32, 0 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B4 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_CC, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 2:
            switch (direction)
            {
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19630), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_C0 | SEGMENT_C4 | SEGMENT_D0 | SEGMENT_D4, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 3:
            switch (direction)
            {
                case 0:
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::LeftCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19629), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::TopCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::RightCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::BottomCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B8 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_D0, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
    }
}

/** rct2: 0x008B0C80 */
static void MineRideTrackDiagRightBank(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    switch (trackSequence)
    {
        case 0:
            switch (direction)
            {
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19629), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_BC | SEGMENT_C4 | SEGMENT_CC | SEGMENT_D4, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 1:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19630), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B4 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_CC, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 2:
            switch (direction)
            {
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19628), { -16, -16, height },
                        { { -16, -16, height + 27 }, { 32, 32, 0 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_C0 | SEGMENT_C4 | SEGMENT_D0 | SEGMENT_D4, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 3:
            switch (direction)
            {
                case 0:
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::LeftCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(19631), { -16, -16, height },
                        { { -16, -16, height }, { 32, 32, 3 } });
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::TopCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::Fork, MetalSupportPlace::RightCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    MetalASupportsPaintSetup(
                        session, MetalSupportType::ForkAlt, MetalSupportPlace::BottomCorner, 6, height,
                        session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_B8 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_D0, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
    }
}

TRACK_PAINT_FUNCTION GetTrackPaintFunctionMineRide(int32_t trackType)
{
    switch (trackType)
    {
        case TrackElemType::Flat:
            return MineRideTrackFlat;
        case TrackElemType::EndStation:
        case TrackElemType::BeginStation:
        case TrackElemType::MiddleStation:
            return MineRideTrackStation;
        case TrackElemType::Up25:
            return MineRideTrack25DegUp;
        case TrackElemType::FlatToUp25:
            return MineRideTrackFlatTo25DegUp;
        case TrackElemType::Up25ToFlat:
            return MineRideTrack25DegUpToFlat;
        case TrackElemType::Down25:
            return MineRideTrack25DegDown;
        case TrackElemType::FlatToDown25:
            return MineRideTrackFlatTo25DegDown;
        case TrackElemType::Down25ToFlat:
            return MineRideTrack25DegDownToFlat;
        case TrackElemType::LeftQuarterTurn5Tiles:
            return MineRideTrackLeftQuarterTurn5;
        case TrackElemType::RightQuarterTurn5Tiles:
            return MineRideTrackRightQuarterTurn5;
        case TrackElemType::FlatToLeftBank:
            return MineRideTrackFlatToLeftBank;
        case TrackElemType::FlatToRightBank:
            return MineRideTrackFlatToRightBank;
        case TrackElemType::LeftBankToFlat:
            return MineRideTrackLeftBankToFlat;
        case TrackElemType::RightBankToFlat:
            return MineRideTrackRightBankToFlat;
        case TrackElemType::BankedLeftQuarterTurn5Tiles:
            return MineRideTrackBankedLeftQuarterTurn5;
        case TrackElemType::BankedRightQuarterTurn5Tiles:
            return MineRideTrackBankedRightQuarterTurn5;
        case TrackElemType::LeftBankToUp25:
            return MineRideTrackLeftBankTo25DegUp;
        case TrackElemType::RightBankToUp25:
            return MineRideTrackRightBankTo25DegUp;
        case TrackElemType::Up25ToLeftBank:
            return MineRideTrack25DegUpToLeftBank;
        case TrackElemType::Up25ToRightBank:
            return MineRideTrack25DegUpToRightBank;
        case TrackElemType::LeftBankToDown25:
            return MineRideTrackLeftBankTo25DegDown;
        case TrackElemType::RightBankToDown25:
            return MineRideTrackRightBankTo25DegDown;
        case TrackElemType::Down25ToLeftBank:
            return MineRideTrack25DegDownToLeftBank;
        case TrackElemType::Down25ToRightBank:
            return MineRideTrack25DegDownToRightBank;
        case TrackElemType::LeftBank:
            return MineRideTrackLeftBank;
        case TrackElemType::RightBank:
            return MineRideTrackRightBank;
        case TrackElemType::SBendLeft:
            return MineRideTrackSBendLeft;
        case TrackElemType::SBendRight:
            return MineRideTrackSBendRight;
        case TrackElemType::LeftQuarterTurn3Tiles:
            return MineRideTrackLeftQuarterTurn3;
        case TrackElemType::RightQuarterTurn3Tiles:
            return MineRideTrackRightQuarterTurn3;
        case TrackElemType::LeftBankedQuarterTurn3Tiles:
            return MineRideTrackLeftQuarterTurn3Bank;
        case TrackElemType::RightBankedQuarterTurn3Tiles:
            return MineRideTrackRightQuarterTurn3Bank;
        case TrackElemType::LeftHalfBankedHelixUpSmall:
            return MineRideTrackLeftHalfBankedHelixUpSmall;
        case TrackElemType::RightHalfBankedHelixUpSmall:
            return MineRideTrackRightHalfBankedHelixUpSmall;
        case TrackElemType::LeftHalfBankedHelixDownSmall:
            return MineRideTrackLeftHalfBankedHelixDownSmall;
        case TrackElemType::RightHalfBankedHelixDownSmall:
            return MineRideTrackRightHalfBankedHelixDownSmall;
        case TrackElemType::LeftHalfBankedHelixUpLarge:
            return MineRideTrackLeftHalfBankedHelixUpLarge;
        case TrackElemType::RightHalfBankedHelixUpLarge:
            return MineRideTrackRightHalfBankedHelixUpLarge;
        case TrackElemType::LeftHalfBankedHelixDownLarge:
            return MineRideTrackLeftHalfBankedHelixDownLarge;
        case TrackElemType::RightHalfBankedHelixDownLarge:
            return MineRideTrackRightHalfBankedHelixDownLarge;
        case TrackElemType::OnRidePhoto:
            return MineRideTrackOnRidePhoto;
        case TrackElemType::LeftEighthToDiag:
            return MineRideTrackLeftEighthToDiag;
        case TrackElemType::RightEighthToDiag:
            return MineRideTrackRightEighthToDiag;
        case TrackElemType::LeftEighthToOrthogonal:
            return MineRideTrackLeftEighthToOrthogonal;
        case TrackElemType::RightEighthToOrthogonal:
            return MineRideTrackRightEighthToOrthogonal;
        case TrackElemType::LeftEighthBankToDiag:
            return MineRideTrackLeftEighthBankToDiag;
        case TrackElemType::RightEighthBankToDiag:
            return MineRideTrackRightEighthBankToDiag;
        case TrackElemType::LeftEighthBankToOrthogonal:
            return MineRideTrackLeftEighthBankToOrthogonal;
        case TrackElemType::RightEighthBankToOrthogonal:
            return MineRideTrackRightEighthBankToOrthogonal;
        case TrackElemType::DiagFlat:
            return MineRideTrackDiagFlat;
        case TrackElemType::DiagUp25:
            return MineRideTrackDiag25DegUp;
        case TrackElemType::DiagFlatToUp25:
            return MineRideTrackDiagFlatTo25DegUp;
        case TrackElemType::DiagUp25ToFlat:
            return MineRideTrackDiag25DegUpToFlat;
        case TrackElemType::DiagDown25:
            return MineRideTrackDiag25DegDown;
        case TrackElemType::DiagFlatToDown25:
            return MineRideTrackDiagFlatTo25DegDown;
        case TrackElemType::DiagDown25ToFlat:
            return MineRideTrackDiag25DegDownToFlat;
        case TrackElemType::DiagFlatToLeftBank:
            return MineRideTrackDiagFlatToLeftBank;
        case TrackElemType::DiagFlatToRightBank:
            return MineRideTrackDiagFlatToRightBank;
        case TrackElemType::DiagLeftBankToFlat:
            return MineRideTrackDiagLeftBankToFlat;
        case TrackElemType::DiagRightBankToFlat:
            return MineRideTrackDiagRightBankToFlat;
        case TrackElemType::DiagLeftBankToUp25:
            return MineRideTrackDiagLeftBankTo25DegUp;
        case TrackElemType::DiagRightBankToUp25:
            return MineRideTrackDiagRightBankTo25DegUp;
        case TrackElemType::DiagUp25ToLeftBank:
            return MineRideTrackDiag25DegUpToLeftBank;
        case TrackElemType::DiagUp25ToRightBank:
            return MineRideTrackDiag25DegUpToRightBank;
        case TrackElemType::DiagLeftBankToDown25:
            return MineRideTrackDiagLeftBankTo25DegDown;
        case TrackElemType::DiagRightBankToDown25:
            return MineRideTrackDiagRightBankTo25DegDown;
        case TrackElemType::DiagDown25ToLeftBank:
            return MineRideTrackDiag25DegDownToLeftBank;
        case TrackElemType::DiagDown25ToRightBank:
            return MineRideTrackDiag25DegDownToRightBank;
        case TrackElemType::DiagLeftBank:
            return MineRideTrackDiagLeftBank;
        case TrackElemType::DiagRightBank:
            return MineRideTrackDiagRightBank;
    }
    return nullptr;
}
