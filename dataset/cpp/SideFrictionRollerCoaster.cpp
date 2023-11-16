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

constexpr int32_t SPR_SIDE_FRICTION_60_DEG_UP_DIR_0_B = 21658;
constexpr int32_t SPR_SIDE_FRICTION_60_DEG_UP_DIR_1_A = 21647;
constexpr int32_t SPR_SIDE_FRICTION_60_DEG_UP_DIR_1_B = 21659;
constexpr int32_t SPR_SIDE_FRICTION_60_DEG_UP_DIR_2_A = 21648;
constexpr int32_t SPR_SIDE_FRICTION_60_DEG_UP_DIR_0_A = 21646;
constexpr int32_t SPR_SIDE_FRICTION_60_DEG_UP_DIR_2_B = 21660;
constexpr int32_t SPR_SIDE_FRICTION_60_DEG_UP_DIR_3_A = 21649;
constexpr int32_t SPR_SIDE_FRICTION_60_DEG_UP_DIR_3_B = 21661;

constexpr int32_t SPR_SIDE_FRICTION_25_DEG_UP_TO_60_DEG_UP_DIR_0_A = 21638;
constexpr int32_t SPR_SIDE_FRICTION_25_DEG_UP_TO_60_DEG_UP_DIR_0_B = 21650;
constexpr int32_t SPR_SIDE_FRICTION_25_DEG_UP_TO_60_DEG_UP_DIR_1_A = 21639;
constexpr int32_t SPR_SIDE_FRICTION_25_DEG_UP_TO_60_DEG_UP_DIR_1_B = 21651;
constexpr int32_t SPR_SIDE_FRICTION_25_DEG_UP_TO_60_DEG_UP_DIR_2_A = 21640;
constexpr int32_t SPR_SIDE_FRICTION_25_DEG_UP_TO_60_DEG_UP_DIR_2_B = 21652;
constexpr int32_t SPR_SIDE_FRICTION_25_DEG_UP_TO_60_DEG_UP_DIR_3_A = 21641;
constexpr int32_t SPR_SIDE_FRICTION_25_DEG_UP_TO_60_DEG_UP_DIR_3_B = 21653;

constexpr int32_t SPR_SIDE_FRICTION_60_DEG_UP_TO_25_DEG_UP_DIR_0_A = 21642;
constexpr int32_t SPR_SIDE_FRICTION_60_DEG_UP_TO_25_DEG_UP_DIR_0_B = 21654;
constexpr int32_t SPR_SIDE_FRICTION_60_DEG_UP_TO_25_DEG_UP_DIR_1_A = 21643;
constexpr int32_t SPR_SIDE_FRICTION_60_DEG_UP_TO_25_DEG_UP_DIR_1_B = 21655;
constexpr int32_t SPR_SIDE_FRICTION_60_DEG_UP_TO_25_DEG_UP_DIR_2_A = 21644;
constexpr int32_t SPR_SIDE_FRICTION_60_DEG_UP_TO_25_DEG_UP_DIR_2_B = 21656;
constexpr int32_t SPR_SIDE_FRICTION_60_DEG_UP_TO_25_DEG_UP_DIR_3_A = 21645;
constexpr int32_t SPR_SIDE_FRICTION_60_DEG_UP_TO_25_DEG_UP_DIR_3_B = 21657;

constexpr int32_t SPR_SIDE_FRICTION_DIAG_60_DEG_UP_DIR_0_A = 21882;
constexpr int32_t SPR_SIDE_FRICTION_DIAG_60_DEG_UP_DIR_0_B = 21886;
constexpr int32_t SPR_SIDE_FRICTION_DIAG_60_DEG_UP_DIR_1_A = 21883; // Needs no B piece
constexpr int32_t SPR_SIDE_FRICTION_DIAG_60_DEG_UP_DIR_2_A = 21884;
constexpr int32_t SPR_SIDE_FRICTION_DIAG_60_DEG_UP_DIR_2_B = 21887;
constexpr int32_t SPR_SIDE_FRICTION_DIAG_60_DEG_UP_DIR_3_A = 21885; // Needs no B piece

// Whole block appears to lack B counterparts
constexpr int32_t SPR_SIDE_FRICTION_DIAG_25_DEG_UP_TO_60_DEG_UP_DIR_0_A = 21870;
constexpr int32_t SPR_SIDE_FRICTION_DIAG_25_DEG_UP_TO_60_DEG_UP_DIR_1_A = 21871;
constexpr int32_t SPR_SIDE_FRICTION_DIAG_25_DEG_UP_TO_60_DEG_UP_DIR_2_A = 21872;
constexpr int32_t SPR_SIDE_FRICTION_DIAG_25_DEG_UP_TO_60_DEG_UP_DIR_3_A = 21873;

constexpr int32_t SPR_SIDE_FRICTION_DIAG_60_DEG_UP_TO_25_DEG_UP_DIR_0_A = 21876;
constexpr int32_t SPR_SIDE_FRICTION_DIAG_60_DEG_UP_TO_25_DEG_UP_DIR_0_B = 21880;
constexpr int32_t SPR_SIDE_FRICTION_DIAG_60_DEG_UP_TO_25_DEG_UP_DIR_1_A = 21877; // Needs no B piece
constexpr int32_t SPR_SIDE_FRICTION_DIAG_60_DEG_UP_TO_25_DEG_UP_DIR_2_A = 21878;
constexpr int32_t SPR_SIDE_FRICTION_DIAG_60_DEG_UP_TO_25_DEG_UP_DIR_2_B = 21881;
constexpr int32_t SPR_SIDE_FRICTION_DIAG_60_DEG_UP_TO_25_DEG_UP_DIR_3_A = 21879; // Needs no B piece

/** rct2: 0x0077839C */
static void SideFrictionRCTrackFlat(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    if (trackElement.HasChain())
    {
        switch (direction)
        {
            case 0:
                PaintAddImageAsParentRotated(
                    session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21662), { 0, 0, height },
                    { { 0, 2, height }, { 32, 27, 2 } });
                PaintAddImageAsParentRotated(
                    session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21664), { 0, 0, height },
                    { { 0, 2, height + 27 }, { 32, 27, 0 } });
                WoodenASupportsPaintSetup(session, 0, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                break;
            case 1:
                PaintAddImageAsParentRotated(
                    session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21663), { 0, 0, height },
                    { { 0, 2, height }, { 32, 27, 2 } });
                PaintAddImageAsParentRotated(
                    session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21665), { 0, 0, height },
                    { { 0, 2, height + 27 }, { 32, 27, 0 } });
                WoodenASupportsPaintSetup(session, 1, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                break;
            case 2:
                PaintAddImageAsParentRotated(
                    session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21666), { 0, 0, height },
                    { { 0, 2, height }, { 32, 27, 2 } });
                PaintAddImageAsParentRotated(
                    session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21668), { 0, 0, height },
                    { { 0, 2, height + 27 }, { 32, 27, 0 } });
                WoodenASupportsPaintSetup(session, 0, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                break;
            case 3:
                PaintAddImageAsParentRotated(
                    session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21667), { 0, 0, height },
                    { { 0, 2, height }, { 32, 27, 2 } });
                PaintAddImageAsParentRotated(
                    session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21669), { 0, 0, height },
                    { { 0, 2, height + 27 }, { 32, 27, 0 } });
                WoodenASupportsPaintSetup(session, 1, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                break;
        }
    }
    else
    {
        switch (direction)
        {
            case 0:
            case 2:
                PaintAddImageAsParentRotated(
                    session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21606), { 0, 0, height },
                    { { 0, 2, height }, { 32, 27, 2 } });
                PaintAddImageAsParentRotated(
                    session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21608), { 0, 0, height },
                    { { 0, 2, height + 27 }, { 32, 27, 0 } });
                WoodenASupportsPaintSetup(session, 0, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                break;
            case 1:
            case 3:
                PaintAddImageAsParentRotated(
                    session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21607), { 0, 0, height },
                    { { 0, 2, height }, { 32, 27, 2 } });
                PaintAddImageAsParentRotated(
                    session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21609), { 0, 0, height },
                    { { 0, 2, height + 27 }, { 32, 27, 0 } });
                WoodenASupportsPaintSetup(session, 1, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                break;
        }
    }
    PaintUtilPushTunnelRotated(session, direction, height, TUNNEL_SQUARE_FLAT);
    PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
    PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
}

/** rct2: 0x007784AC, 0x007784BC, 0x007784CC */
static void SideFrictionRCTrackStation(
    PaintSession& session, const Ride& ride, [[maybe_unused]] uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    static constexpr uint32_t imageIds[4] = {
        21610,
        21611,
        21610,
        21611,
    };

    PaintAddImageAsParentRotated(
        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(imageIds[direction]), { 0, 0, height },
        { { 0, 2, height }, { 32, 27, 2 } });
    WoodenASupportsPaintSetup(session, direction & 1, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
    TrackPaintUtilDrawStation2(session, ride, direction, height, trackElement, 9, 11);
    PaintUtilPushTunnelRotated(session, direction, height, TUNNEL_SQUARE_FLAT);
    PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
    PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
}

/** rct2: 0x007783AC */
static void SideFrictionRCTrack25DegUp(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    if (trackElement.HasChain())
    {
        switch (direction)
        {
            case 0:
                PaintAddImageAsParentRotated(
                    session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21678), { 0, 0, height },
                    { { 0, 2, height }, { 32, 27, 2 } });
                PaintAddImageAsParentRotated(
                    session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21690), { 0, 0, height },
                    { { 0, 26, height + 5 }, { 32, 1, 9 } });
                WoodenASupportsPaintSetup(session, 0, 9, height, session.TrackColours[SCHEME_SUPPORTS]);
                break;
            case 1:
                PaintAddImageAsParentRotated(
                    session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21679), { 0, 0, height },
                    { { 0, 2, height }, { 32, 27, 2 } });
                PaintAddImageAsParentRotated(
                    session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21691), { 0, 0, height },
                    { { 0, 26, height + 5 }, { 32, 1, 9 } });
                WoodenASupportsPaintSetup(session, 1, 10, height, session.TrackColours[SCHEME_SUPPORTS]);
                break;
            case 2:
                PaintAddImageAsParentRotated(
                    session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21680), { 0, 0, height },
                    { { 0, 2, height }, { 32, 27, 2 } });
                PaintAddImageAsParentRotated(
                    session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21692), { 0, 0, height },
                    { { 0, 26, height + 5 }, { 32, 1, 9 } });
                WoodenASupportsPaintSetup(session, 0, 11, height, session.TrackColours[SCHEME_SUPPORTS]);
                break;
            case 3:
                PaintAddImageAsParentRotated(
                    session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21681), { 0, 0, height },
                    { { 0, 2, height }, { 32, 27, 2 } });
                PaintAddImageAsParentRotated(
                    session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21693), { 0, 0, height },
                    { { 0, 26, height + 5 }, { 32, 1, 9 } });
                WoodenASupportsPaintSetup(session, 1, 12, height, session.TrackColours[SCHEME_SUPPORTS]);
                break;
        }
    }
    else
    {
        switch (direction)
        {
            case 0:
                PaintAddImageAsParentRotated(
                    session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21622), { 0, 0, height },
                    { { 0, 2, height }, { 32, 27, 2 } });
                PaintAddImageAsParentRotated(
                    session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21634), { 0, 0, height },
                    { { 0, 26, height + 5 }, { 32, 1, 9 } });
                WoodenASupportsPaintSetup(session, 0, 9, height, session.TrackColours[SCHEME_SUPPORTS]);
                break;
            case 1:
                PaintAddImageAsParentRotated(
                    session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21623), { 0, 0, height },
                    { { 0, 2, height }, { 32, 27, 2 } });
                PaintAddImageAsParentRotated(
                    session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21635), { 0, 0, height },
                    { { 0, 26, height + 5 }, { 32, 1, 9 } });
                WoodenASupportsPaintSetup(session, 1, 10, height, session.TrackColours[SCHEME_SUPPORTS]);
                break;
            case 2:
                PaintAddImageAsParentRotated(
                    session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21624), { 0, 0, height },
                    { { 0, 2, height }, { 32, 27, 2 } });
                PaintAddImageAsParentRotated(
                    session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21636), { 0, 0, height },
                    { { 0, 26, height + 5 }, { 32, 1, 9 } });
                WoodenASupportsPaintSetup(session, 0, 11, height, session.TrackColours[SCHEME_SUPPORTS]);
                break;
            case 3:
                PaintAddImageAsParentRotated(
                    session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21625), { 0, 0, height },
                    { { 0, 2, height }, { 32, 27, 2 } });
                PaintAddImageAsParentRotated(
                    session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21637), { 0, 0, height },
                    { { 0, 26, height + 5 }, { 32, 1, 9 } });
                WoodenASupportsPaintSetup(session, 1, 12, height, session.TrackColours[SCHEME_SUPPORTS]);
                break;
        }
    }
    if (direction == 0 || direction == 3)
    {
        PaintUtilPushTunnelRotated(session, direction, height - 8, TUNNEL_SQUARE_7);
    }
    else
    {
        PaintUtilPushTunnelRotated(session, direction, height + 8, TUNNEL_SQUARE_8);
    }
    PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
    PaintUtilSetGeneralSupportHeight(session, height + 56, 0x20);
}

/** rct2: 0x007783CC */
static void SideFrictionRCTrackFlatTo25DegUp(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    if (trackElement.HasChain())
    {
        switch (direction)
        {
            case 0:
                PaintAddImageAsParentRotated(
                    session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21670), { 0, 0, height },
                    { { 0, 2, height }, { 32, 27, 2 } });
                PaintAddImageAsParentRotated(
                    session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21682), { 0, 0, height },
                    { { 0, 26, height + 5 }, { 32, 1, 9 } });
                WoodenASupportsPaintSetup(session, 0, 1, height, session.TrackColours[SCHEME_SUPPORTS]);
                break;
            case 1:
                PaintAddImageAsParentRotated(
                    session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21671), { 0, 0, height },
                    { { 0, 2, height }, { 32, 27, 2 } });
                PaintAddImageAsParentRotated(
                    session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21683), { 0, 0, height },
                    { { 0, 26, height + 5 }, { 32, 1, 9 } });
                WoodenASupportsPaintSetup(session, 1, 2, height, session.TrackColours[SCHEME_SUPPORTS]);
                break;
            case 2:
                PaintAddImageAsParentRotated(
                    session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21672), { 0, 0, height },
                    { { 0, 2, height }, { 32, 27, 2 } });
                PaintAddImageAsParentRotated(
                    session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21684), { 0, 0, height },
                    { { 0, 26, height + 5 }, { 32, 1, 9 } });
                WoodenASupportsPaintSetup(session, 0, 3, height, session.TrackColours[SCHEME_SUPPORTS]);
                break;
            case 3:
                PaintAddImageAsParentRotated(
                    session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21673), { 0, 0, height },
                    { { 0, 2, height }, { 32, 27, 2 } });
                PaintAddImageAsParentRotated(
                    session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21685), { 0, 0, height },
                    { { 0, 26, height + 5 }, { 32, 1, 9 } });
                WoodenASupportsPaintSetup(session, 1, 4, height, session.TrackColours[SCHEME_SUPPORTS]);
                break;
        }
    }
    else
    {
        switch (direction)
        {
            case 0:
                PaintAddImageAsParentRotated(
                    session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21614), { 0, 0, height },
                    { { 0, 2, height }, { 32, 27, 2 } });
                PaintAddImageAsParentRotated(
                    session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21626), { 0, 0, height },
                    { { 0, 26, height + 5 }, { 32, 1, 9 } });
                WoodenASupportsPaintSetup(session, 0, 1, height, session.TrackColours[SCHEME_SUPPORTS]);
                break;
            case 1:
                PaintAddImageAsParentRotated(
                    session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21615), { 0, 0, height },
                    { { 0, 2, height }, { 32, 27, 2 } });
                PaintAddImageAsParentRotated(
                    session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21627), { 0, 0, height },
                    { { 0, 26, height + 5 }, { 32, 1, 9 } });
                WoodenASupportsPaintSetup(session, 1, 2, height, session.TrackColours[SCHEME_SUPPORTS]);
                break;
            case 2:
                PaintAddImageAsParentRotated(
                    session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21616), { 0, 0, height },
                    { { 0, 2, height }, { 32, 27, 2 } });
                PaintAddImageAsParentRotated(
                    session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21628), { 0, 0, height },
                    { { 0, 26, height + 5 }, { 32, 1, 9 } });
                WoodenASupportsPaintSetup(session, 0, 3, height, session.TrackColours[SCHEME_SUPPORTS]);
                break;
            case 3:
                PaintAddImageAsParentRotated(
                    session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21617), { 0, 0, height },
                    { { 0, 2, height }, { 32, 27, 2 } });
                PaintAddImageAsParentRotated(
                    session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21629), { 0, 0, height },
                    { { 0, 26, height + 5 }, { 32, 1, 9 } });
                WoodenASupportsPaintSetup(session, 1, 4, height, session.TrackColours[SCHEME_SUPPORTS]);
                break;
        }
    }
    if (direction == 0 || direction == 3)
    {
        PaintUtilPushTunnelRotated(session, direction, height, TUNNEL_SQUARE_FLAT);
    }
    else
    {
        PaintUtilPushTunnelRotated(session, direction, height, TUNNEL_SQUARE_8);
    }
    PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
    PaintUtilSetGeneralSupportHeight(session, height + 48, 0x20);
}

/** rct2: 0x007783FC */
static void SideFrictionRCTrack25DegUpToFlat(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    if (trackElement.HasChain())
    {
        switch (direction)
        {
            case 0:
                PaintAddImageAsParentRotated(
                    session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21674), { 0, 0, height },
                    { { 0, 2, height }, { 32, 27, 2 } });
                PaintAddImageAsParentRotated(
                    session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21686), { 0, 0, height },
                    { { 0, 26, height + 5 }, { 32, 1, 9 } });
                WoodenASupportsPaintSetup(session, 0, 5, height, session.TrackColours[SCHEME_SUPPORTS]);
                break;
            case 1:
                PaintAddImageAsParentRotated(
                    session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21675), { 0, 0, height },
                    { { 0, 2, height }, { 32, 27, 2 } });
                PaintAddImageAsParentRotated(
                    session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21687), { 0, 0, height },
                    { { 0, 26, height + 5 }, { 32, 1, 9 } });
                WoodenASupportsPaintSetup(session, 1, 6, height, session.TrackColours[SCHEME_SUPPORTS]);
                break;
            case 2:
                PaintAddImageAsParentRotated(
                    session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21676), { 0, 0, height },
                    { { 0, 2, height }, { 32, 27, 2 } });
                PaintAddImageAsParentRotated(
                    session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21688), { 0, 0, height },
                    { { 0, 26, height + 5 }, { 32, 1, 9 } });
                WoodenASupportsPaintSetup(session, 0, 7, height, session.TrackColours[SCHEME_SUPPORTS]);
                break;
            case 3:
                PaintAddImageAsParentRotated(
                    session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21677), { 0, 0, height },
                    { { 0, 2, height }, { 32, 27, 2 } });
                PaintAddImageAsParentRotated(
                    session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21689), { 0, 0, height },
                    { { 0, 26, height + 5 }, { 32, 1, 9 } });
                WoodenASupportsPaintSetup(session, 1, 8, height, session.TrackColours[SCHEME_SUPPORTS]);
                break;
        }
    }
    else
    {
        switch (direction)
        {
            case 0:
                PaintAddImageAsParentRotated(
                    session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21618), { 0, 0, height },
                    { { 0, 2, height }, { 32, 27, 2 } });
                PaintAddImageAsParentRotated(
                    session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21630), { 0, 0, height },
                    { { 0, 26, height + 5 }, { 32, 1, 9 } });
                WoodenASupportsPaintSetup(session, 0, 5, height, session.TrackColours[SCHEME_SUPPORTS]);
                break;
            case 1:
                PaintAddImageAsParentRotated(
                    session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21619), { 0, 0, height },
                    { { 0, 2, height }, { 32, 27, 2 } });
                PaintAddImageAsParentRotated(
                    session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21631), { 0, 0, height },
                    { { 0, 26, height + 5 }, { 32, 1, 9 } });
                WoodenASupportsPaintSetup(session, 1, 6, height, session.TrackColours[SCHEME_SUPPORTS]);
                break;
            case 2:
                PaintAddImageAsParentRotated(
                    session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21620), { 0, 0, height },
                    { { 0, 2, height }, { 32, 27, 2 } });
                PaintAddImageAsParentRotated(
                    session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21632), { 0, 0, height },
                    { { 0, 26, height + 5 }, { 32, 1, 9 } });
                WoodenASupportsPaintSetup(session, 0, 7, height, session.TrackColours[SCHEME_SUPPORTS]);
                break;
            case 3:
                PaintAddImageAsParentRotated(
                    session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21621), { 0, 0, height },
                    { { 0, 2, height }, { 32, 27, 2 } });
                PaintAddImageAsParentRotated(
                    session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21633), { 0, 0, height },
                    { { 0, 26, height + 5 }, { 32, 1, 9 } });
                WoodenASupportsPaintSetup(session, 1, 8, height, session.TrackColours[SCHEME_SUPPORTS]);
                break;
        }
    }
    if (direction == 0 || direction == 3)
    {
        PaintUtilPushTunnelRotated(session, direction, height - 8, TUNNEL_SQUARE_FLAT);
    }
    else
    {
        PaintUtilPushTunnelRotated(session, direction, height + 8, TUNNEL_14);
    }
    PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
    PaintUtilSetGeneralSupportHeight(session, height + 40, 0x20);
}

/** rct2: 0x0077840C */
static void SideFrictionRCTrack25DegDown(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    SideFrictionRCTrack25DegUp(session, ride, trackSequence, (direction + 2) & 3, height, trackElement);
}

/** rct2: 0x0077842C */
static void SideFrictionRCTrackFlatTo25DegDown(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    SideFrictionRCTrack25DegUpToFlat(session, ride, trackSequence, (direction + 2) & 3, height, trackElement);
}

/** rct2: 0x0077845C */
static void SideFrictionRCTrack25DegDownToFlat(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    SideFrictionRCTrackFlatTo25DegUp(session, ride, trackSequence, (direction + 2) & 3, height, trackElement);
}

/** rct2: 0x0077846C */
static void SideFrictionRCTrackLeftQuarterTurn5(
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
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21727), { 0, 2, height },
                        { 32, 27, 2 });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21747), { 0, 2, height },
                        { { 0, 2, height + 27 }, { 32, 27, 0 } });
                    WoodenASupportsPaintSetup(session, 0, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21732), { 0, 2, height },
                        { 32, 27, 2 });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21752), { 0, 2, height },
                        { { 0, 2, height + 27 }, { 32, 27, 0 } });
                    WoodenASupportsPaintSetup(session, 1, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21737), { 0, 2, height },
                        { 32, 32, 2 });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21757), { 0, 2, height },
                        { { 0, 2, height + 27 }, { 32, 32, 0 } });
                    WoodenASupportsPaintSetup(session, 0, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21722), { 0, 2, height },
                        { 32, 32, 2 });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21742), { 0, 2, height },
                        { { 0, 2, height + 27 }, { 32, 32, 0 } });
                    WoodenASupportsPaintSetup(session, 1, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            if (direction == 0 || direction == 3)
            {
                PaintUtilPushTunnelRotated(session, direction, height, TUNNEL_SQUARE_FLAT);
            }
            PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 1:
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_BC | SEGMENT_CC | SEGMENT_D4, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 2:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21726), { 0, 0, height },
                        { 32, 16, 2 });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21746), { 0, 0, height },
                        { { 0, 0, height + 27 }, { 32, 16, 0 } });
                    WoodenASupportsPaintSetup(session, 5, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21731), { 0, 0, height },
                        { 32, 16, 2 });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21751), { 0, 0, height },
                        { { 0, 0, height + 27 }, { 32, 16, 0 } });
                    WoodenASupportsPaintSetup(session, 2, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21736), { 0, 16, height },
                        { 32, 16, 2 });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21756), { 0, 16, height },
                        { { 0, 16, height + 27 }, { 32, 16, 0 } });
                    WoodenASupportsPaintSetup(session, 3, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21721), { 0, 16, height },
                        { 32, 16, 2 });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21741), { 0, 16, height },
                        { { 0, 16, height + 27 }, { 32, 16, 0 } });
                    WoodenASupportsPaintSetup(session, 4, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
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
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21725), { 0, 16, height },
                        { 16, 16, 2 });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21745), { 0, 16, height },
                        { { 0, 16, height + 27 }, { 16, 16, 0 } });
                    WoodenASupportsPaintSetup(session, 3, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21730), { 16, 16, height },
                        { 16, 16, 2 });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21750), { 16, 16, height },
                        { { 16, 16, height + 27 }, { 16, 16, 0 } });
                    WoodenASupportsPaintSetup(session, 4, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21735), { 16, 0, height },
                        { 16, 16, 2 });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21755), { 16, 0, height },
                        { { 16, 0, height + 27 }, { 16, 16, 0 } });
                    WoodenASupportsPaintSetup(session, 5, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21720), { 0, 0, height },
                        { 16, 16, 2 });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21740), { 0, 0, height },
                        { { 0, 0, height + 27 }, { 16, 16, 0 } });
                    WoodenASupportsPaintSetup(session, 2, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            PaintUtilSetSegmentSupportHeight(
                session,
                PaintUtilRotateSegments(
                    SEGMENT_B4 | SEGMENT_BC | SEGMENT_C0 | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_CC | SEGMENT_D0 | SEGMENT_D4,
                    direction),
                0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 4:
            PaintUtilSetSegmentSupportHeight(
                session, PaintUtilRotateSegments(SEGMENT_BC | SEGMENT_CC | SEGMENT_D4, direction), 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 5:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21724), { 16, 0, height },
                        { 16, 34, 2 });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21744), { 16, 0, height },
                        { { 16, 0, height + 27 }, { 16, 34, 0 } });
                    WoodenASupportsPaintSetup(session, 5, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21729), { 0, 0, height },
                        { 16, 32, 2 });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21749), { 0, 0, height },
                        { { 0, 0, height + 27 }, { 16, 32, 0 } });
                    WoodenASupportsPaintSetup(session, 2, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21734), { 0, 0, height },
                        { 16, 32, 2 });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21754), { 0, 0, height },
                        { { 0, 0, height + 27 }, { 16, 32, 0 } });
                    WoodenASupportsPaintSetup(session, 3, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21719), { 16, 0, height },
                        { 16, 32, 2 });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21739), { 16, 0, height },
                        { { 16, 0, height + 27 }, { 16, 32, 0 } });
                    WoodenASupportsPaintSetup(session, 4, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
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
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21723), { 2, 0, height },
                        { 32, 32, 2 });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21743), { 2, 0, height },
                        { { 2, 0, height + 27 }, { 32, 32, 0 } });
                    WoodenASupportsPaintSetup(session, 1, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21728), { 2, 0, height },
                        { 27, 32, 2 });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21748), { 2, 0, height },
                        { { 2, 0, height + 27 }, { 27, 32, 0 } });
                    WoodenASupportsPaintSetup(session, 0, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21733), { 2, 0, height },
                        { 27, 32, 2 });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21753), { 2, 0, height },
                        { { 2, 0, height + 27 }, { 27, 32, 0 } });
                    WoodenASupportsPaintSetup(session, 1, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21718), { 2, 0, height },
                        { 32, 32, 2 });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21738), { 2, 0, height },
                        { { 2, 0, height + 27 }, { 32, 32, 0 } });
                    WoodenASupportsPaintSetup(session, 0, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
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
            PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
    }
}

/** rct2: 0x0077847C */
static void SideFrictionRCTrackRightQuarterTurn5(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    trackSequence = mapLeftQuarterTurn5TilesToRightQuarterTurn5Tiles[trackSequence];
    SideFrictionRCTrackLeftQuarterTurn5(session, ride, trackSequence, (direction - 1) & 3, height, trackElement);
}

/** rct2: 0x0077848C */
static void SideFrictionRCTrackSBendLeft(
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
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21888), { 0, 0, height },
                        { { 0, 2, height }, { 32, 27, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21904), { 0, 0, height },
                        { { 0, 2, height + 27 }, { 32, 27, 0 } });
                    WoodenASupportsPaintSetup(session, 0, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21892), { 0, 0, height },
                        { { 0, 2, height }, { 32, 27, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21908), { 0, 0, height },
                        { { 0, 2, height + 27 }, { 32, 27, 0 } });
                    WoodenASupportsPaintSetup(session, 1, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21891), { 0, 0, height },
                        { { 0, 2, height }, { 32, 27, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21907), { 0, 0, height },
                        { { 0, 2, height + 27 }, { 32, 27, 0 } });
                    WoodenASupportsPaintSetup(session, 0, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21895), { 0, 0, height },
                        { { 0, 2, height }, { 32, 27, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21911), { 0, 0, height },
                        { { 0, 2, height + 27 }, { 32, 27, 0 } });
                    WoodenASupportsPaintSetup(session, 1, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            if (direction == 0 || direction == 3)
            {
                PaintUtilPushTunnelRotated(session, direction, height, TUNNEL_SQUARE_FLAT);
            }
            PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 1:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21889), { 0, 0, height },
                        { { 0, 0, height }, { 32, 26, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21905), { 0, 0, height },
                        { { 0, 0, height + 27 }, { 32, 26, 0 } });
                    WoodenASupportsPaintSetup(session, 5, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21893), { 0, 0, height },
                        { { 0, 0, height }, { 32, 26, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21909), { 0, 0, height },
                        { { 0, 0, height + 27 }, { 32, 26, 0 } });
                    WoodenASupportsPaintSetup(session, 2, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21890), { 0, 0, height },
                        { { 0, 6, height }, { 32, 26, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21906), { 0, 0, height },
                        { { 0, 6, height + 27 }, { 32, 26, 0 } });
                    WoodenASupportsPaintSetup(session, 3, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21894), { 0, 0, height },
                        { { 0, 6, height }, { 32, 26, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21910), { 0, 0, height },
                        { { 0, 6, height + 27 }, { 32, 26, 0 } });
                    WoodenASupportsPaintSetup(session, 4, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
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
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21890), { 0, 0, height },
                        { { 0, 6, height }, { 32, 26, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21906), { 0, 0, height },
                        { { 0, 6, height + 27 }, { 32, 26, 0 } });
                    WoodenASupportsPaintSetup(session, 3, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21894), { 0, 0, height },
                        { { 0, 6, height }, { 32, 26, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21910), { 0, 0, height },
                        { { 0, 6, height + 27 }, { 32, 26, 0 } });
                    WoodenASupportsPaintSetup(session, 4, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21889), { 0, 0, height },
                        { { 0, 0, height }, { 32, 26, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21905), { 0, 0, height },
                        { { 0, 0, height + 27 }, { 32, 26, 0 } });
                    WoodenASupportsPaintSetup(session, 5, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21893), { 0, 0, height },
                        { { 0, 0, height }, { 32, 26, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21909), { 0, 0, height },
                        { { 0, 0, height + 27 }, { 32, 26, 0 } });
                    WoodenASupportsPaintSetup(session, 2, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
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
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21891), { 0, 0, height },
                        { { 0, 2, height }, { 32, 27, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21907), { 0, 0, height },
                        { { 0, 2, height + 27 }, { 32, 27, 0 } });
                    WoodenASupportsPaintSetup(session, 0, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21895), { 0, 0, height },
                        { { 0, 2, height }, { 32, 27, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21911), { 0, 0, height },
                        { { 0, 2, height + 27 }, { 32, 27, 0 } });
                    WoodenASupportsPaintSetup(session, 1, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21888), { 0, 0, height },
                        { { 0, 2, height }, { 32, 27, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21904), { 0, 0, height },
                        { { 0, 2, height + 27 }, { 32, 27, 0 } });
                    WoodenASupportsPaintSetup(session, 0, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21892), { 0, 0, height },
                        { { 0, 2, height }, { 32, 27, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21908), { 0, 0, height },
                        { { 0, 2, height + 27 }, { 32, 27, 0 } });
                    WoodenASupportsPaintSetup(session, 1, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
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
            PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
    }
}

/** rct2: 0x0077849C */
static void SideFrictionRCTrackSBendRight(
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
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21896), { 0, 0, height },
                        { { 0, 2, height }, { 32, 27, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21912), { 0, 0, height },
                        { { 0, 2, height + 27 }, { 32, 27, 0 } });
                    WoodenASupportsPaintSetup(session, 0, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21900), { 0, 0, height },
                        { { 0, 2, height }, { 32, 27, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21916), { 0, 0, height },
                        { { 0, 2, height + 27 }, { 32, 27, 0 } });
                    WoodenASupportsPaintSetup(session, 1, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21899), { 0, 0, height },
                        { { 0, 2, height }, { 32, 27, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21915), { 0, 0, height },
                        { { 0, 2, height + 27 }, { 32, 27, 0 } });
                    WoodenASupportsPaintSetup(session, 0, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21903), { 0, 0, height },
                        { { 0, 2, height }, { 32, 27, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21919), { 0, 0, height },
                        { { 0, 2, height + 27 }, { 32, 27, 0 } });
                    WoodenASupportsPaintSetup(session, 1, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            if (direction == 0 || direction == 3)
            {
                PaintUtilPushTunnelRotated(session, direction, height, TUNNEL_SQUARE_FLAT);
            }
            PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 1:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21897), { 0, 0, height },
                        { { 0, 6, height }, { 32, 26, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21913), { 0, 0, height },
                        { { 0, 6, height + 27 }, { 32, 26, 0 } });
                    WoodenASupportsPaintSetup(session, 4, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21901), { 0, 0, height },
                        { { 0, 6, height }, { 32, 26, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21917), { 0, 0, height },
                        { { 0, 6, height + 27 }, { 32, 26, 0 } });
                    WoodenASupportsPaintSetup(session, 5, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21898), { 0, 0, height },
                        { { 0, 0, height }, { 32, 26, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21914), { 0, 0, height },
                        { { 0, 0, height + 27 }, { 32, 26, 0 } });
                    WoodenASupportsPaintSetup(session, 2, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21902), { 0, 0, height },
                        { { 0, 0, height }, { 32, 26, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21918), { 0, 0, height },
                        { { 0, 0, height + 27 }, { 32, 26, 0 } });
                    WoodenASupportsPaintSetup(session, 3, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
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
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21898), { 0, 0, height },
                        { { 0, 0, height }, { 32, 26, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21914), { 0, 0, height },
                        { { 0, 0, height + 27 }, { 32, 26, 0 } });
                    WoodenASupportsPaintSetup(session, 2, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21902), { 0, 0, height },
                        { { 0, 0, height }, { 32, 26, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21918), { 0, 0, height },
                        { { 0, 0, height + 27 }, { 32, 26, 0 } });
                    WoodenASupportsPaintSetup(session, 3, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21897), { 0, 0, height },
                        { { 0, 6, height }, { 32, 26, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21913), { 0, 0, height },
                        { { 0, 6, height + 27 }, { 32, 26, 0 } });
                    WoodenASupportsPaintSetup(session, 4, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21901), { 0, 0, height },
                        { { 0, 6, height }, { 32, 26, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21917), { 0, 0, height },
                        { { 0, 6, height + 27 }, { 32, 26, 0 } });
                    WoodenASupportsPaintSetup(session, 5, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
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
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21899), { 0, 0, height },
                        { { 0, 2, height }, { 32, 27, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21915), { 0, 0, height },
                        { { 0, 2, height + 27 }, { 32, 27, 0 } });
                    WoodenASupportsPaintSetup(session, 0, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21903), { 0, 0, height },
                        { { 0, 2, height }, { 32, 27, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21919), { 0, 0, height },
                        { { 0, 2, height + 27 }, { 32, 27, 0 } });
                    WoodenASupportsPaintSetup(session, 1, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21896), { 0, 0, height },
                        { { 0, 2, height }, { 32, 27, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21912), { 0, 0, height },
                        { { 0, 2, height + 27 }, { 32, 27, 0 } });
                    WoodenASupportsPaintSetup(session, 0, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21900), { 0, 0, height },
                        { { 0, 2, height }, { 32, 27, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21916), { 0, 0, height },
                        { { 0, 2, height + 27 }, { 32, 27, 0 } });
                    WoodenASupportsPaintSetup(session, 1, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
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
            PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
    }
}

/** rct2: 0x007784DC */
static void SideFrictionRCTrackLeftQuarterTurn3(
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
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21699), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21711), { 0, 0, height },
                        { { 0, 6, height + 27 }, { 32, 20, 0 } });
                    WoodenASupportsPaintSetup(session, 5, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21702), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21714), { 0, 0, height },
                        { { 0, 6, height + 27 }, { 32, 20, 0 } });
                    WoodenASupportsPaintSetup(session, 2, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21705), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21717), { 0, 0, height },
                        { { 0, 6, height + 27 }, { 32, 20, 0 } });
                    WoodenASupportsPaintSetup(session, 3, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21696), { 0, 0, height },
                        { { 0, 6, height }, { 32, 20, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21708), { 0, 0, height },
                        { { 0, 6, height + 27 }, { 32, 20, 0 } });
                    WoodenASupportsPaintSetup(session, 4, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            if (direction == 0 || direction == 3)
            {
                PaintUtilPushTunnelRotated(session, direction, height, TUNNEL_SQUARE_FLAT);
            }
            PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
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
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21698), { 0, 0, height },
                        { { 16, 0, height }, { 16, 16, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21710), { 0, 0, height },
                        { { 16, 0, height + 27 }, { 16, 16, 0 } });
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21701), { 0, 0, height },
                        { { 0, 0, height }, { 16, 16, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21713), { 0, 0, height },
                        { { 0, 0, height + 27 }, { 16, 16, 0 } });
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21704), { 0, 0, height },
                        { { 0, 16, height }, { 16, 16, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21716), { 0, 0, height },
                        { { 0, 16, height + 27 }, { 16, 16, 0 } });
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21695), { 0, 0, height },
                        { { 16, 16, height }, { 16, 16, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21707), { 0, 0, height },
                        { { 16, 16, height + 27 }, { 16, 16, 0 } });
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
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21697), { 0, 0, height },
                        { { 6, 0, height }, { 20, 32, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21709), { 0, 0, height },
                        { { 6, 0, height + 27 }, { 20, 32, 0 } });
                    WoodenASupportsPaintSetup(session, 5, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21700), { 0, 0, height },
                        { { 6, 0, height }, { 20, 32, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21712), { 0, 0, height },
                        { { 6, 0, height + 27 }, { 20, 32, 0 } });
                    WoodenASupportsPaintSetup(session, 2, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21703), { 0, 0, height },
                        { { 6, 0, height }, { 20, 32, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21715), { 0, 0, height },
                        { { 6, 0, height + 27 }, { 20, 32, 0 } });
                    WoodenASupportsPaintSetup(session, 3, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21694), { 0, 0, height },
                        { { 6, 0, height }, { 20, 32, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21706), { 0, 0, height },
                        { { 6, 0, height + 27 }, { 20, 32, 0 } });
                    WoodenASupportsPaintSetup(session, 4, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
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
            PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
    }
}

/** rct2: 0x007784EC */
static void SideFrictionRCTrackRightQuarterTurn3(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    trackSequence = mapLeftQuarterTurn3TilesToRightQuarterTurn3Tiles[trackSequence];
    SideFrictionRCTrackLeftQuarterTurn3(session, ride, trackSequence, (direction - 1) & 3, height, trackElement);
}

/** rct2: 0x007784FC */
static void SideFrictionRCTrackBrakes(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    switch (direction)
    {
        case 0:
        case 2:
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21610), { 0, 0, height },
                { { 0, 2, height }, { 32, 27, 2 } });
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21612), { 0, 0, height },
                { { 0, 2, height + 27 }, { 32, 27, 0 } });
            WoodenASupportsPaintSetup(session, 0, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
            break;
        case 1:
        case 3:
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21611), { 0, 0, height },
                { { 0, 2, height }, { 32, 27, 2 } });
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21613), { 0, 0, height },
                { { 0, 2, height + 27 }, { 32, 27, 0 } });
            WoodenASupportsPaintSetup(session, 1, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
            break;
    }
    PaintUtilPushTunnelRotated(session, direction, height, TUNNEL_SQUARE_FLAT);
    PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
    PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
}

/** rct2: 0x007785DC */
static void SideFrictionRCTrackLeftEighthToDiag(
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
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21790), { 0, 0, height },
                        { { 0, 0, height }, { 32, 32, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21806), { 0, 0, height },
                        { { 0, 0, height + 27 }, { 32, 32, 0 } });
                    WoodenASupportsPaintSetup(session, 0, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21794), { 0, 0, height },
                        { { 0, 0, height }, { 32, 32, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21810), { 0, 0, height },
                        { { 0, 0, height + 27 }, { 32, 32, 0 } });
                    WoodenASupportsPaintSetup(session, 1, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21798), { 0, 0, height },
                        { { 0, 0, height }, { 32, 32, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21814), { 0, 0, height },
                        { { 0, 0, height + 27 }, { 32, 32, 0 } });
                    WoodenASupportsPaintSetup(session, 0, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21802), { 0, 0, height },
                        { { 0, 0, height }, { 32, 32, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21818), { 0, 0, height },
                        { { 0, 0, height + 27 }, { 32, 32, 0 } });
                    WoodenASupportsPaintSetup(session, 1, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            if (direction == 0 || direction == 3)
            {
                PaintUtilPushTunnelRotated(session, direction, height, TUNNEL_SQUARE_FLAT);
            }
            PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 1:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21791), { 0, 0, height },
                        { { 0, 0, height }, { 32, 16, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21807), { 0, 0, height },
                        { { 0, 0, height + 27 }, { 32, 16, 0 } });
                    WoodenASupportsPaintSetup(session, 0, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21795), { 0, 0, height },
                        { { 0, 0, height }, { 34, 16, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21811), { 0, 0, height },
                        { { 0, 0, height + 27 }, { 32, 16, 0 } });
                    WoodenASupportsPaintSetup(session, 1, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21799), { 0, 0, height },
                        { { 0, 16, height }, { 32, 16, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21815), { 0, 0, height },
                        { { 0, 16, height + 27 }, { 32, 16, 0 } });
                    WoodenASupportsPaintSetup(session, 0, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21803), { 0, 0, height },
                        { { 0, 16, height }, { 32, 16, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21819), { 0, 0, height },
                        { { 0, 16, height + 27 }, { 32, 16, 0 } });
                    WoodenASupportsPaintSetup(session, 1, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 2:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21792), { 0, 0, height },
                        { { 0, 16, height }, { 16, 16, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21808), { 0, 0, height },
                        { { 0, 16, height + 27 }, { 16, 16, 0 } });
                    WoodenASupportsPaintSetup(session, 3, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21796), { 0, 0, height },
                        { { 16, 16, height }, { 16, 16, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21812), { 0, 0, height },
                        { { 16, 16, height + 27 }, { 16, 16, 0 } });
                    WoodenASupportsPaintSetup(session, 4, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21800), { 0, 0, height },
                        { { 16, 0, height }, { 16, 16, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21816), { 0, 0, height },
                        { { 16, 0, height + 27 }, { 16, 16, 0 } });
                    WoodenASupportsPaintSetup(session, 5, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21804), { 0, 0, height },
                        { { 0, 0, height }, { 16, 16, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21820), { 0, 0, height },
                        { { 0, 0, height + 27 }, { 16, 16, 0 } });
                    WoodenASupportsPaintSetup(session, 2, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 3:
            switch (direction)
            {
                case 0:
                    WoodenASupportsPaintSetup(session, 5, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    WoodenASupportsPaintSetup(session, 2, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    WoodenASupportsPaintSetup(session, 3, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    WoodenASupportsPaintSetup(session, 4, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 4:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21793), { 0, 0, height },
                        { { 16, 16, height }, { 16, 16, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21809), { 0, 0, height },
                        { { 16, 16, height + 27 }, { 16, 16, 0 } });
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21797), { 0, 0, height },
                        { { 0, 16, height }, { 16, 18, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21813), { 0, 0, height },
                        { { 0, 16, height + 27 }, { 16, 16, 0 } });
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21801), { 0, 0, height },
                        { { 0, 0, height }, { 16, 16, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21817), { 0, 0, height },
                        { { 0, 0, height + 27 }, { 16, 16, 0 } });
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21805), { 0, 0, height },
                        { { 16, 0, height }, { 16, 16, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21821), { 0, 0, height },
                        { { 16, 0, height + 27 }, { 16, 16, 0 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
    }
}

/** rct2: 0x007785EC */
static void SideFrictionRCTrackRightEighthToDiag(
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
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21758), { 0, 0, height },
                        { { 0, 0, height }, { 32, 32, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21774), { 0, 0, height },
                        { { 0, 0, height + 27 }, { 32, 32, 0 } });
                    WoodenASupportsPaintSetup(session, 0, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21762), { 0, 0, height },
                        { { 0, 0, height }, { 32, 32, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21778), { 0, 0, height },
                        { { 0, 0, height + 27 }, { 32, 32, 0 } });
                    WoodenASupportsPaintSetup(session, 1, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21766), { 0, 0, height },
                        { { 0, 0, height }, { 32, 32, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21782), { 0, 0, height },
                        { { 0, 0, height + 27 }, { 32, 32, 0 } });
                    WoodenASupportsPaintSetup(session, 0, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21770), { 0, 0, height },
                        { { 0, 0, height }, { 32, 32, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21786), { 0, 0, height },
                        { { 0, 0, height + 27 }, { 32, 32, 0 } });
                    WoodenASupportsPaintSetup(session, 1, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            if (direction == 0 || direction == 3)
            {
                PaintUtilPushTunnelRotated(session, direction, height, TUNNEL_SQUARE_FLAT);
            }
            PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 1:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21759), { 0, 0, height },
                        { { 0, 16, height }, { 32, 16, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21775), { 0, 0, height },
                        { { 0, 16, height + 27 }, { 32, 16, 0 } });
                    WoodenASupportsPaintSetup(session, 0, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21763), { 0, 0, height },
                        { { 0, 16, height }, { 32, 16, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21779), { 0, 0, height },
                        { { 0, 16, height + 27 }, { 32, 16, 0 } });
                    WoodenASupportsPaintSetup(session, 1, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21767), { 0, 0, height },
                        { { 0, 0, height }, { 34, 16, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21783), { 0, 0, height },
                        { { 0, 0, height + 27 }, { 32, 16, 0 } });
                    WoodenASupportsPaintSetup(session, 0, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21771), { 0, 0, height },
                        { { 0, 0, height }, { 32, 16, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21787), { 0, 0, height },
                        { { 0, 0, height + 27 }, { 32, 16, 0 } });
                    WoodenASupportsPaintSetup(session, 1, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 2:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21760), { 0, 0, height },
                        { { 0, 0, height }, { 16, 16, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21776), { 0, 0, height },
                        { { 0, 0, height + 27 }, { 16, 16, 0 } });
                    WoodenASupportsPaintSetup(session, 2, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21764), { 0, 0, height },
                        { { 16, 0, height }, { 16, 16, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21780), { 0, 0, height },
                        { { 16, 0, height + 27 }, { 16, 16, 0 } });
                    WoodenASupportsPaintSetup(session, 3, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21768), { 0, 0, height },
                        { { 4, 4, height }, { 28, 28, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21784), { 0, 0, height },
                        { { 4, 4, height + 27 }, { 28, 28, 0 } });
                    WoodenASupportsPaintSetup(session, 4, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21772), { 0, 0, height },
                        { { 0, 16, height }, { 16, 16, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21788), { 0, 0, height },
                        { { 0, 16, height + 27 }, { 16, 16, 0 } });
                    WoodenASupportsPaintSetup(session, 5, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 3:
            switch (direction)
            {
                case 0:
                    WoodenASupportsPaintSetup(session, 4, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    WoodenASupportsPaintSetup(session, 5, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    WoodenASupportsPaintSetup(session, 2, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    WoodenASupportsPaintSetup(session, 3, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 4:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21761), { 0, 0, height },
                        { { 16, 0, height }, { 16, 16, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21777), { 0, 0, height },
                        { { 16, 0, height + 27 }, { 16, 16, 0 } });
                    break;
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21765), { 0, 0, height },
                        { { 0, 0, height }, { 16, 16, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21781), { 0, 0, height },
                        { { 0, 0, height + 27 }, { 16, 16, 0 } });
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21769), { 0, 0, height },
                        { { 0, 16, height }, { 16, 18, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21785), { 0, 0, height },
                        { { 0, 16, height + 27 }, { 16, 16, 0 } });
                    break;
                case 3:
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21773), { 0, 0, height },
                        { { 16, 16, height }, { 16, 16, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21789), { 0, 0, height },
                        { { 16, 16, height + 27 }, { 16, 16, 0 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
    }
}

/** rct2: 0x007785FC */
static void SideFrictionRCTrackLeftEighthToOrthogonal(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    trackSequence = mapLeftEighthTurnToOrthogonal[trackSequence];
    SideFrictionRCTrackRightEighthToDiag(session, ride, trackSequence, (direction + 2) & 3, height, trackElement);
}

/** rct2: 0x0077860C */
static void SideFrictionRCTrackRightEighthToOrthogonal(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    trackSequence = mapLeftEighthTurnToOrthogonal[trackSequence];
    SideFrictionRCTrackLeftEighthToDiag(session, ride, trackSequence, (direction + 3) & 3, height, trackElement);
}

/** rct2: 0x0077850C */
static void SideFrictionRCTrackDiagFlat(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    switch (trackSequence)
    {
        case 0:
            if (trackElement.HasChain())
            {
                switch (direction)
                {
                    case 3:
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21849), { -16, -16, height },
                            { { -16, -16, height }, { 32, 32, 2 } });
                        break;
                }
            }
            else
            {
                switch (direction)
                {
                    case 3:
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21825), { -16, -16, height },
                            { { -16, -16, height }, { 32, 32, 2 } });
                        break;
                }
            }
            PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 1:
            if (trackElement.HasChain())
            {
                switch (direction)
                {
                    case 0:
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21846), { -16, -16, height },
                            { { -16, -16, height }, { 32, 32, 2 } });
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21850), { -16, -16, height },
                            { { -16, -16, height + 27 }, { 32, 32, 0 } });
                        WoodenASupportsPaintSetup(session, 2, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 1:
                        WoodenASupportsPaintSetup(session, 3, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 2:
                        WoodenASupportsPaintSetup(session, 4, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 3:
                        WoodenASupportsPaintSetup(session, 5, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                }
            }
            else
            {
                switch (direction)
                {
                    case 0:
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21822), { -16, -16, height },
                            { { -16, -16, height }, { 32, 32, 2 } });
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21826), { -16, -16, height },
                            { { -16, -16, height + 27 }, { 32, 32, 0 } });
                        WoodenASupportsPaintSetup(session, 2, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 1:
                        WoodenASupportsPaintSetup(session, 3, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 2:
                        WoodenASupportsPaintSetup(session, 4, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 3:
                        WoodenASupportsPaintSetup(session, 5, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                }
            }
            PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 2:
            if (trackElement.HasChain())
            {
                switch (direction)
                {
                    case 0:
                        WoodenASupportsPaintSetup(session, 4, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 1:
                        WoodenASupportsPaintSetup(session, 5, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 2:
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21848), { -16, -16, height },
                            { { -16, -16, height }, { 32, 32, 2 } });
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21851), { -16, -16, height },
                            { { -16, -16, height + 27 }, { 32, 32, 0 } });
                        WoodenASupportsPaintSetup(session, 2, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 3:
                        WoodenASupportsPaintSetup(session, 3, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                }
            }
            else
            {
                switch (direction)
                {
                    case 0:
                        WoodenASupportsPaintSetup(session, 4, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 1:
                        WoodenASupportsPaintSetup(session, 5, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 2:
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21824), { -16, -16, height },
                            { { -16, -16, height }, { 32, 32, 2 } });
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21827), { -16, -16, height },
                            { { -16, -16, height + 27 }, { 32, 32, 0 } });
                        WoodenASupportsPaintSetup(session, 2, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 3:
                        WoodenASupportsPaintSetup(session, 3, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                }
            }
            PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 3:
            if (trackElement.HasChain())
            {
                switch (direction)
                {
                    case 1:
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21847), { -16, -16, height },
                            { { -16, -16, height }, { 32, 32, 2 } });
                        break;
                }
            }
            else
            {
                switch (direction)
                {
                    case 1:
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21823), { -16, -16, height },
                            { { -16, -16, height }, { 32, 32, 2 } });
                        break;
                }
            }
            PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
    }
}

/** rct2: 0x0077853C */
static void SideFrictionRCTrackDiag25DegUp(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    switch (trackSequence)
    {
        case 0:
            if (trackElement.HasChain())
            {
                switch (direction)
                {
                    case 3:
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21867), { -16, -16, height },
                            { { -16, -16, height }, { 32, 32, 2 } });
                        break;
                }
            }
            else
            {
                switch (direction)
                {
                    case 3:
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21843), { -16, -16, height },
                            { { -16, -16, height }, { 32, 32, 2 } });
                        break;
                }
            }
            PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 56, 0x20);
            break;
        case 1:
            if (trackElement.HasChain())
            {
                switch (direction)
                {
                    case 0:
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21864), { -16, -16, height },
                            { { -16, -16, height }, { 32, 32, 2 } });
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21868), { -16, -16, height },
                            { { -16, -16, height + 43 }, { 32, 32, 0 } });
                        WoodenBSupportsPaintSetup(session, 2, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 1:
                        WoodenBSupportsPaintSetup(session, 3, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 2:
                        WoodenBSupportsPaintSetup(session, 4, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 3:
                        WoodenBSupportsPaintSetup(session, 5, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                }
            }
            else
            {
                switch (direction)
                {
                    case 0:
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21840), { -16, -16, height },
                            { { -16, -16, height }, { 32, 32, 2 } });
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21844), { -16, -16, height },
                            { { -16, -16, height + 43 }, { 32, 32, 0 } });
                        WoodenBSupportsPaintSetup(session, 2, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 1:
                        WoodenBSupportsPaintSetup(session, 3, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 2:
                        WoodenBSupportsPaintSetup(session, 4, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 3:
                        WoodenBSupportsPaintSetup(session, 5, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                }
            }
            PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 56, 0x20);
            break;
        case 2:
            if (trackElement.HasChain())
            {
                switch (direction)
                {
                    case 0:
                        WoodenBSupportsPaintSetup(session, 4, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 1:
                        WoodenBSupportsPaintSetup(session, 5, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 2:
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21866), { -16, -16, height },
                            { { -16, -16, height }, { 32, 32, 2 } });
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21869), { -16, -16, height },
                            { { -16, -16, height + 43 }, { 32, 32, 0 } });
                        WoodenBSupportsPaintSetup(session, 2, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 3:
                        WoodenBSupportsPaintSetup(session, 3, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                }
            }
            else
            {
                switch (direction)
                {
                    case 0:
                        WoodenBSupportsPaintSetup(session, 4, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 1:
                        WoodenBSupportsPaintSetup(session, 5, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 2:
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21842), { -16, -16, height },
                            { { -16, -16, height }, { 32, 32, 2 } });
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21845), { -16, -16, height },
                            { { -16, -16, height + 43 }, { 32, 32, 0 } });
                        WoodenBSupportsPaintSetup(session, 2, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 3:
                        WoodenBSupportsPaintSetup(session, 3, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                }
            }
            PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 56, 0x20);
            break;
        case 3:
            if (trackElement.HasChain())
            {
                switch (direction)
                {
                    case 1:
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21865), { -16, -16, height },
                            { { -16, -16, height }, { 32, 32, 2 } });
                        break;
                }
            }
            else
            {
                switch (direction)
                {
                    case 1:
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21841), { -16, -16, height },
                            { { -16, -16, height }, { 32, 32, 2 } });
                        break;
                }
            }
            PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 56, 0x20);
            break;
    }
}

/** rct2: 0x0077851C */
static void SideFrictionRCTrackDiagFlatTo25DegUp(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    switch (trackSequence)
    {
        case 0:
            if (trackElement.HasChain())
            {
                switch (direction)
                {
                    case 3:
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21855), { -16, -16, height },
                            { { -16, -16, height }, { 32, 32, 2 } });
                        break;
                }
            }
            else
            {
                switch (direction)
                {
                    case 3:
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21831), { -16, -16, height },
                            { { -16, -16, height }, { 32, 32, 2 } });
                        break;
                }
            }
            PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 48, 0x20);
            break;
        case 1:
            if (trackElement.HasChain())
            {
                switch (direction)
                {
                    case 0:
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21852), { -16, -16, height },
                            { { -16, -16, height }, { 32, 32, 2 } });
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21856), { -16, -16, height },
                            { { -16, -16, height + 35 }, { 32, 32, 0 } });
                        WoodenASupportsPaintSetup(session, 2, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 1:
                        WoodenASupportsPaintSetup(session, 3, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 2:
                        WoodenASupportsPaintSetup(session, 4, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 3:
                        WoodenASupportsPaintSetup(session, 5, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                }
            }
            else
            {
                switch (direction)
                {
                    case 0:
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21828), { -16, -16, height },
                            { { -16, -16, height }, { 32, 32, 2 } });
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21832), { -16, -16, height },
                            { { -16, -16, height + 35 }, { 32, 32, 0 } });
                        WoodenASupportsPaintSetup(session, 2, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 1:
                        WoodenASupportsPaintSetup(session, 3, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 2:
                        WoodenASupportsPaintSetup(session, 4, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 3:
                        WoodenASupportsPaintSetup(session, 5, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                }
            }
            PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 48, 0x20);
            break;
        case 2:
            if (trackElement.HasChain())
            {
                switch (direction)
                {
                    case 0:
                        WoodenASupportsPaintSetup(session, 4, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 1:
                        WoodenASupportsPaintSetup(session, 5, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 2:
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21854), { -16, -16, height },
                            { { -16, -16, height }, { 32, 32, 2 } });
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21857), { -16, -16, height },
                            { { -16, -16, height + 35 }, { 32, 32, 0 } });
                        WoodenASupportsPaintSetup(session, 2, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 3:
                        WoodenASupportsPaintSetup(session, 3, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                }
            }
            else
            {
                switch (direction)
                {
                    case 0:
                        WoodenASupportsPaintSetup(session, 4, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 1:
                        WoodenASupportsPaintSetup(session, 5, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 2:
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21830), { -16, -16, height },
                            { { -16, -16, height }, { 32, 32, 2 } });
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21833), { -16, -16, height },
                            { { -16, -16, height + 35 }, { 32, 32, 0 } });
                        WoodenASupportsPaintSetup(session, 2, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 3:
                        WoodenASupportsPaintSetup(session, 3, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                }
            }
            PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 48, 0x20);
            break;
        case 3:
            if (trackElement.HasChain())
            {
                switch (direction)
                {
                    case 1:
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21853), { -16, -16, height },
                            { { -16, -16, height }, { 32, 32, 2 } });
                        break;
                }
            }
            else
            {
                switch (direction)
                {
                    case 1:
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21829), { -16, -16, height },
                            { { -16, -16, height }, { 32, 32, 2 } });
                        break;
                }
            }
            PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 48, 0x20);
            break;
    }
}

/** rct2: 0x0077852C */
static void SideFrictionRCTrackDiag25DegUpToFlat(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    switch (trackSequence)
    {
        case 0:
            if (trackElement.HasChain())
            {
                switch (direction)
                {
                    case 3:
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21861), { -16, -16, height },
                            { { -16, -16, height }, { 32, 32, 2 } });
                        break;
                }
            }
            else
            {
                switch (direction)
                {
                    case 3:
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21837), { -16, -16, height },
                            { { -16, -16, height }, { 32, 32, 2 } });
                        break;
                }
            }
            PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 56, 0x20);
            break;
        case 1:
            if (trackElement.HasChain())
            {
                switch (direction)
                {
                    case 0:
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21858), { -16, -16, height },
                            { { -16, -16, height }, { 32, 32, 2 } });
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21862), { -16, -16, height },
                            { { -16, -16, height + 35 }, { 32, 32, 0 } });
                        WoodenBSupportsPaintSetup(session, 2, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 1:
                        WoodenBSupportsPaintSetup(session, 3, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 2:
                        WoodenBSupportsPaintSetup(session, 4, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 3:
                        WoodenBSupportsPaintSetup(session, 5, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                }
            }
            else
            {
                switch (direction)
                {
                    case 0:
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21834), { -16, -16, height },
                            { { -16, -16, height }, { 32, 32, 2 } });
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21838), { -16, -16, height },
                            { { -16, -16, height + 35 }, { 32, 32, 0 } });
                        WoodenBSupportsPaintSetup(session, 2, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 1:
                        WoodenBSupportsPaintSetup(session, 3, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 2:
                        WoodenBSupportsPaintSetup(session, 4, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 3:
                        WoodenBSupportsPaintSetup(session, 5, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                }
            }
            PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 56, 0x20);
            break;
        case 2:
            if (trackElement.HasChain())
            {
                switch (direction)
                {
                    case 0:
                        WoodenBSupportsPaintSetup(session, 4, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 1:
                        WoodenBSupportsPaintSetup(session, 5, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 2:
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21860), { -16, -16, height },
                            { { -16, -16, height }, { 32, 32, 2 } });
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21863), { -16, -16, height },
                            { { -16, -16, height + 35 }, { 32, 32, 0 } });
                        WoodenBSupportsPaintSetup(session, 2, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 3:
                        WoodenBSupportsPaintSetup(session, 3, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                }
            }
            else
            {
                switch (direction)
                {
                    case 0:
                        WoodenBSupportsPaintSetup(session, 4, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 1:
                        WoodenBSupportsPaintSetup(session, 5, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 2:
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21836), { -16, -16, height },
                            { { -16, -16, height }, { 32, 32, 2 } });
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21839), { -16, -16, height },
                            { { -16, -16, height + 35 }, { 32, 32, 0 } });
                        WoodenBSupportsPaintSetup(session, 2, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 3:
                        WoodenBSupportsPaintSetup(session, 3, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                }
            }
            PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 56, 0x20);
            break;
        case 3:
            if (trackElement.HasChain())
            {
                switch (direction)
                {
                    case 1:
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21859), { -16, -16, height },
                            { { -16, -16, height }, { 32, 32, 2 } });
                        break;
                }
            }
            else
            {
                switch (direction)
                {
                    case 1:
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21835), { -16, -16, height },
                            { { -16, -16, height }, { 32, 32, 2 } });
                        break;
                }
            }
            PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 56, 0x20);
            break;
    }
}

/** rct2: 0x0077859C */
static void SideFrictionRCTrackDiag25DegDown(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    switch (trackSequence)
    {
        case 0:
            if (trackElement.HasChain())
            {
                switch (direction)
                {
                    case 3:
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21865), { -16, -16, height },
                            { { -16, -16, height }, { 32, 32, 2 } });
                        break;
                }
            }
            else
            {
                switch (direction)
                {
                    case 3:
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21841), { -16, -16, height },
                            { { -16, -16, height }, { 32, 32, 2 } });
                        break;
                }
            }
            PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 56, 0x20);
            break;
        case 1:
            if (trackElement.HasChain())
            {
                switch (direction)
                {
                    case 0:
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21866), { -16, -16, height },
                            { { -16, -16, height }, { 32, 32, 2 } });
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21869), { -16, -16, height },
                            { { -16, -16, height + 43 }, { 32, 32, 0 } });
                        WoodenBSupportsPaintSetup(session, 2, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 1:
                        WoodenBSupportsPaintSetup(session, 3, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 2:
                        WoodenBSupportsPaintSetup(session, 4, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 3:
                        WoodenBSupportsPaintSetup(session, 5, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                }
            }
            else
            {
                switch (direction)
                {
                    case 0:
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21842), { -16, -16, height },
                            { { -16, -16, height }, { 32, 32, 2 } });
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21845), { -16, -16, height },
                            { { -16, -16, height + 43 }, { 32, 32, 0 } });
                        WoodenBSupportsPaintSetup(session, 2, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 1:
                        WoodenBSupportsPaintSetup(session, 3, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 2:
                        WoodenBSupportsPaintSetup(session, 4, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 3:
                        WoodenBSupportsPaintSetup(session, 5, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                }
            }
            PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 56, 0x20);
            break;
        case 2:
            if (trackElement.HasChain())
            {
                switch (direction)
                {
                    case 0:
                        WoodenBSupportsPaintSetup(session, 4, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 1:
                        WoodenBSupportsPaintSetup(session, 5, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 2:
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21864), { -16, -16, height },
                            { { -16, -16, height }, { 32, 32, 2 } });
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21868), { -16, -16, height },
                            { { -16, -16, height + 43 }, { 32, 32, 0 } });
                        WoodenBSupportsPaintSetup(session, 2, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 3:
                        WoodenBSupportsPaintSetup(session, 3, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                }
            }
            else
            {
                switch (direction)
                {
                    case 0:
                        WoodenBSupportsPaintSetup(session, 4, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 1:
                        WoodenBSupportsPaintSetup(session, 5, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 2:
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21840), { -16, -16, height },
                            { { -16, -16, height }, { 32, 32, 2 } });
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21844), { -16, -16, height },
                            { { -16, -16, height + 43 }, { 32, 32, 0 } });
                        WoodenBSupportsPaintSetup(session, 2, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 3:
                        WoodenBSupportsPaintSetup(session, 3, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                }
            }
            PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 56, 0x20);
            break;
        case 3:
            if (trackElement.HasChain())
            {
                switch (direction)
                {
                    case 1:
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21867), { -16, -16, height },
                            { { -16, -16, height }, { 32, 32, 2 } });
                        break;
                }
            }
            else
            {
                switch (direction)
                {
                    case 1:
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21843), { -16, -16, height },
                            { { -16, -16, height }, { 32, 32, 2 } });
                        break;
                }
            }
            PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 56, 0x20);
            break;
    }
}

/** rct2: 0x0077857C */
static void SideFrictionRCTrackDiagFlatTo25DegDown(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    switch (trackSequence)
    {
        case 0:
            if (trackElement.HasChain())
            {
                switch (direction)
                {
                    case 3:
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21859), { -16, -16, height },
                            { { -16, -16, height }, { 32, 32, 2 } });
                        break;
                }
            }
            else
            {
                switch (direction)
                {
                    case 3:
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21835), { -16, -16, height },
                            { { -16, -16, height }, { 32, 32, 2 } });
                        break;
                }
            }
            break;
        case 1:
            if (trackElement.HasChain())
            {
                switch (direction)
                {
                    case 0:
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21860), { -16, -16, height },
                            { { -16, -16, height }, { 32, 32, 2 } });
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21863), { -16, -16, height },
                            { { -16, -16, height + 35 }, { 32, 32, 0 } });
                        WoodenBSupportsPaintSetup(session, 2, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 1:
                        WoodenBSupportsPaintSetup(session, 3, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 2:
                        WoodenBSupportsPaintSetup(session, 4, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 3:
                        WoodenBSupportsPaintSetup(session, 5, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                }
            }
            else
            {
                switch (direction)
                {
                    case 0:
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21836), { -16, -16, height },
                            { { -16, -16, height }, { 32, 32, 2 } });
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21839), { -16, -16, height },
                            { { -16, -16, height + 35 }, { 32, 32, 0 } });
                        WoodenBSupportsPaintSetup(session, 2, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 1:
                        WoodenBSupportsPaintSetup(session, 3, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 2:
                        WoodenBSupportsPaintSetup(session, 4, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 3:
                        WoodenBSupportsPaintSetup(session, 5, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                }
            }
            break;
        case 2:
            if (trackElement.HasChain())
            {
                switch (direction)
                {
                    case 0:
                        WoodenBSupportsPaintSetup(session, 4, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 1:
                        WoodenBSupportsPaintSetup(session, 5, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 2:
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21858), { -16, -16, height },
                            { { -16, -16, height }, { 32, 32, 2 } });
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21862), { -16, -16, height },
                            { { -16, -16, height + 35 }, { 32, 32, 0 } });
                        WoodenBSupportsPaintSetup(session, 2, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 3:
                        WoodenBSupportsPaintSetup(session, 3, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                }
            }
            else
            {
                switch (direction)
                {
                    case 0:
                        WoodenBSupportsPaintSetup(session, 4, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 1:
                        WoodenBSupportsPaintSetup(session, 5, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 2:
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21834), { -16, -16, height },
                            { { -16, -16, height }, { 32, 32, 2 } });
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21838), { -16, -16, height },
                            { { -16, -16, height + 35 }, { 32, 32, 0 } });
                        WoodenBSupportsPaintSetup(session, 2, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 3:
                        WoodenBSupportsPaintSetup(session, 3, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                }
            }
            break;
        case 3:
            if (trackElement.HasChain())
            {
                switch (direction)
                {
                    case 1:
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21861), { -16, -16, height },
                            { { -16, -16, height }, { 32, 32, 2 } });
                        break;
                }
            }
            else
            {
                switch (direction)
                {
                    case 1:
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21837), { -16, -16, height },
                            { { -16, -16, height }, { 32, 32, 2 } });
                        break;
                }
            }
            break;
    }

    PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
    PaintUtilSetGeneralSupportHeight(session, height + 56, 0x20);
}

/** rct2: 0x0077858C */
static void SideFrictionRCTrackDiag25DegDownToFlat(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    switch (trackSequence)
    {
        case 0:
            if (trackElement.HasChain())
            {
                switch (direction)
                {
                    case 3:
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21853), { -16, -16, height },
                            { { -16, -16, height }, { 32, 32, 2 } });
                        break;
                }
            }
            else
            {
                switch (direction)
                {
                    case 3:
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21829), { -16, -16, height },
                            { { -16, -16, height }, { 32, 32, 2 } });
                        break;
                }
            }
            PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 48, 0x20);
            break;
        case 1:
            if (trackElement.HasChain())
            {
                switch (direction)
                {
                    case 0:
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21854), { -16, -16, height },
                            { { -16, -16, height }, { 32, 32, 2 } });
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21857), { -16, -16, height },
                            { { -16, -16, height + 35 }, { 32, 32, 0 } });
                        WoodenASupportsPaintSetup(session, 2, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 1:
                        WoodenASupportsPaintSetup(session, 3, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 2:
                        WoodenASupportsPaintSetup(session, 4, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 3:
                        WoodenASupportsPaintSetup(session, 5, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                }
            }
            else
            {
                switch (direction)
                {
                    case 0:
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21830), { -16, -16, height },
                            { { -16, -16, height }, { 32, 32, 2 } });
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21833), { -16, -16, height },
                            { { -16, -16, height + 35 }, { 32, 32, 0 } });
                        WoodenASupportsPaintSetup(session, 2, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 1:
                        WoodenASupportsPaintSetup(session, 3, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 2:
                        WoodenASupportsPaintSetup(session, 4, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 3:
                        WoodenASupportsPaintSetup(session, 5, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                }
            }
            PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 48, 0x20);
            break;
        case 2:
            if (trackElement.HasChain())
            {
                switch (direction)
                {
                    case 0:
                        WoodenASupportsPaintSetup(session, 4, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 1:
                        WoodenASupportsPaintSetup(session, 5, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 2:
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21852), { -16, -16, height },
                            { { -16, -16, height }, { 32, 32, 2 } });
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21856), { -16, -16, height },
                            { { -16, -16, height + 35 }, { 32, 32, 0 } });
                        WoodenASupportsPaintSetup(session, 2, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 3:
                        WoodenASupportsPaintSetup(session, 3, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                }
            }
            else
            {
                switch (direction)
                {
                    case 0:
                        WoodenASupportsPaintSetup(session, 4, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 1:
                        WoodenASupportsPaintSetup(session, 5, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 2:
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21828), { -16, -16, height },
                            { { -16, -16, height }, { 32, 32, 2 } });
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21832), { -16, -16, height },
                            { { -16, -16, height + 35 }, { 32, 32, 0 } });
                        WoodenASupportsPaintSetup(session, 2, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                    case 3:
                        WoodenASupportsPaintSetup(session, 3, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
                        break;
                }
            }
            PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 48, 0x20);
            break;
        case 3:
            if (trackElement.HasChain())
            {
                switch (direction)
                {
                    case 1:
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21855), { -16, -16, height },
                            { { -16, -16, height }, { 32, 32, 2 } });
                        break;
                }
            }
            else
            {
                switch (direction)
                {
                    case 1:
                        PaintAddImageAsParentRotated(
                            session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(21831), { -16, -16, height },
                            { { -16, -16, height }, { 32, 32, 2 } });
                        break;
                }
            }
            PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 48, 0x20);
            break;
    }
}

static void SideFrictionRCTrack60DegUp(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    switch (direction)
    {
        case 0:
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(SPR_SIDE_FRICTION_60_DEG_UP_DIR_0_A),
                { 0, 0, height }, { { 0, 2, height }, { 32, 27, 2 } });
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(SPR_SIDE_FRICTION_60_DEG_UP_DIR_0_B),
                { 0, 0, height }, { { 0, 26, height + 5 }, { 32, 1, 9 } });
            WoodenASupportsPaintSetup(session, 0, 21, height, session.TrackColours[SCHEME_SUPPORTS]);
            break;
        case 1:
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(SPR_SIDE_FRICTION_60_DEG_UP_DIR_1_A),
                { 0, 0, height }, { { 0, 2, height }, { 32, 27, 2 } });

            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(SPR_SIDE_FRICTION_60_DEG_UP_DIR_1_B),
                { 0, 0, height }, { { 0, 26, height + 5 }, { 32, 1, 9 } });

            WoodenASupportsPaintSetup(session, 1, 22, height, session.TrackColours[SCHEME_SUPPORTS]);
            break;
        case 2:
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(SPR_SIDE_FRICTION_60_DEG_UP_DIR_2_A),
                { 0, 0, height }, { { 0, 2, height }, { 32, 27, 2 } });
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(SPR_SIDE_FRICTION_60_DEG_UP_DIR_2_B),
                { 0, 0, height }, { { 0, 26, height + 5 }, { 32, 1, 9 } });

            WoodenASupportsPaintSetup(session, 0, 23, height, session.TrackColours[SCHEME_SUPPORTS]);
            break;
        case 3:
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(SPR_SIDE_FRICTION_60_DEG_UP_DIR_3_A),
                { 0, 0, height }, { { 0, 2, height }, { 32, 27, 2 } });
            PaintAddImageAsParentRotated(
                session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(SPR_SIDE_FRICTION_60_DEG_UP_DIR_3_B),
                { 0, 0, height }, { { 0, 26, height + 5 }, { 32, 1, 9 } });
            WoodenASupportsPaintSetup(session, 1, 24, height, session.TrackColours[SCHEME_SUPPORTS]);
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
    PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
    PaintUtilSetGeneralSupportHeight(session, height + 104, 0x20);
}

static void SideFrictionRCTrack60DegDown(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    SideFrictionRCTrack60DegUp(session, ride, trackSequence, (direction + 2) % 4, height, trackElement);
}

static void SideFrictionRCTrack25DegUpTo60DegUp(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    switch (direction)
    {
        case 0:
            PaintAddImageAsParentRotated(
                session, direction,
                session.TrackColours[SCHEME_TRACK].WithIndex(SPR_SIDE_FRICTION_25_DEG_UP_TO_60_DEG_UP_DIR_0_A),
                { 0, 0, height }, { { 0, 2, height }, { 32, 27, 2 } });
            PaintAddImageAsParentRotated(
                session, direction,
                session.TrackColours[SCHEME_TRACK].WithIndex(SPR_SIDE_FRICTION_25_DEG_UP_TO_60_DEG_UP_DIR_0_B),
                { 0, 0, height }, { { 0, 26, height + 5 }, { 32, 1, 9 } });
            WoodenASupportsPaintSetup(session, 0, 13, height, session.TrackColours[SCHEME_SUPPORTS]);
            break;
        case 1:
            PaintAddImageAsParentRotated(
                session, direction,
                session.TrackColours[SCHEME_TRACK].WithIndex(SPR_SIDE_FRICTION_25_DEG_UP_TO_60_DEG_UP_DIR_1_A),
                { 0, 0, height }, { { 0, 2, height }, { 32, 27, 2 } });
            PaintAddImageAsParentRotated(
                session, direction,
                session.TrackColours[SCHEME_TRACK].WithIndex(SPR_SIDE_FRICTION_25_DEG_UP_TO_60_DEG_UP_DIR_1_B),
                { 0, 0, height }, { { 0, 26, height + 5 }, { 32, 1, 9 } });
            WoodenASupportsPaintSetup(session, 1, 14, height, session.TrackColours[SCHEME_SUPPORTS]);
            break;
        case 2:
            PaintAddImageAsParentRotated(
                session, direction,
                session.TrackColours[SCHEME_TRACK].WithIndex(SPR_SIDE_FRICTION_25_DEG_UP_TO_60_DEG_UP_DIR_2_A),
                { 0, 0, height }, { { 0, 2, height }, { 32, 27, 2 } });
            PaintAddImageAsParentRotated(
                session, direction,
                session.TrackColours[SCHEME_TRACK].WithIndex(SPR_SIDE_FRICTION_25_DEG_UP_TO_60_DEG_UP_DIR_2_B),
                { 0, 0, height }, { { 0, 26, height + 5 }, { 32, 1, 9 } });
            WoodenASupportsPaintSetup(session, 0, 15, height, session.TrackColours[SCHEME_SUPPORTS]);
            break;
        case 3:
            PaintAddImageAsParentRotated(
                session, direction,
                session.TrackColours[SCHEME_TRACK].WithIndex(SPR_SIDE_FRICTION_25_DEG_UP_TO_60_DEG_UP_DIR_3_A),
                { 0, 0, height }, { { 0, 2, height }, { 32, 27, 2 } });
            PaintAddImageAsParentRotated(
                session, direction,
                session.TrackColours[SCHEME_TRACK].WithIndex(SPR_SIDE_FRICTION_25_DEG_UP_TO_60_DEG_UP_DIR_3_B),
                { 0, 0, height }, { { 0, 26, height + 5 }, { 32, 1, 9 } });
            WoodenASupportsPaintSetup(session, 1, 16, height, session.TrackColours[SCHEME_SUPPORTS]);
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
    PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
    PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
}

static void SideFrictionRCTrack60DegDownTo25DegDown(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    SideFrictionRCTrack25DegUpTo60DegUp(session, ride, trackSequence, (direction + 2) % 4, height, trackElement);
}

static void SideFrictionRCTrack60DegUpTo25DegUp(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    switch (direction)
    {
        case 0:
            PaintAddImageAsParentRotated(
                session, direction,
                session.TrackColours[SCHEME_TRACK].WithIndex(SPR_SIDE_FRICTION_60_DEG_UP_TO_25_DEG_UP_DIR_0_A),
                { 0, 0, height }, { { 0, 2, height }, { 32, 27, 2 } });
            PaintAddImageAsParentRotated(
                session, direction,
                session.TrackColours[SCHEME_TRACK].WithIndex(SPR_SIDE_FRICTION_60_DEG_UP_TO_25_DEG_UP_DIR_0_B),
                { 0, 0, height }, { { 0, 26, height + 5 }, { 32, 1, 9 } });
            WoodenASupportsPaintSetup(session, 0, 17, height, session.TrackColours[SCHEME_SUPPORTS]);
            break;
        case 1:
            PaintAddImageAsParentRotated(
                session, direction,
                session.TrackColours[SCHEME_TRACK].WithIndex(SPR_SIDE_FRICTION_60_DEG_UP_TO_25_DEG_UP_DIR_1_A),
                { 0, 0, height }, { { 0, 2, height }, { 32, 27, 2 } });
            PaintAddImageAsParentRotated(
                session, direction,
                session.TrackColours[SCHEME_TRACK].WithIndex(SPR_SIDE_FRICTION_60_DEG_UP_TO_25_DEG_UP_DIR_1_B),
                { 0, 0, height }, { { 0, 26, height + 5 }, { 32, 1, 9 } });
            WoodenASupportsPaintSetup(session, 1, 18, height, session.TrackColours[SCHEME_SUPPORTS]);
            break;
        case 2:
            PaintAddImageAsParentRotated(
                session, direction,
                session.TrackColours[SCHEME_TRACK].WithIndex(SPR_SIDE_FRICTION_60_DEG_UP_TO_25_DEG_UP_DIR_2_A),
                { 0, 0, height }, { { 0, 2, height }, { 32, 27, 2 } });
            PaintAddImageAsParentRotated(
                session, direction,
                session.TrackColours[SCHEME_TRACK].WithIndex(SPR_SIDE_FRICTION_60_DEG_UP_TO_25_DEG_UP_DIR_2_B),
                { 0, 0, height }, { { 0, 26, height + 5 }, { 32, 1, 9 } });
            WoodenASupportsPaintSetup(session, 0, 19, height, session.TrackColours[SCHEME_SUPPORTS]);
            break;
        case 3:
            PaintAddImageAsParentRotated(
                session, direction,
                session.TrackColours[SCHEME_TRACK].WithIndex(SPR_SIDE_FRICTION_60_DEG_UP_TO_25_DEG_UP_DIR_3_A),
                { 0, 0, height }, { { 0, 2, height }, { 32, 27, 2 } });
            PaintAddImageAsParentRotated(
                session, direction,
                session.TrackColours[SCHEME_TRACK].WithIndex(SPR_SIDE_FRICTION_60_DEG_UP_TO_25_DEG_UP_DIR_3_B),
                { 0, 0, height }, { { 0, 26, height + 5 }, { 32, 1, 9 } });
            WoodenASupportsPaintSetup(session, 1, 20, height, session.TrackColours[SCHEME_SUPPORTS]);
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
    PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
    PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
}

static void SideFrictionRCTrack25DegDownTo60DegDown(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    SideFrictionRCTrack60DegUpTo25DegUp(session, ride, trackSequence, (direction + 2) % 4, height, trackElement);
}

static void SideFrictionRCTrackDiag60DegUp(
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
                        session, direction,
                        session.TrackColours[SCHEME_TRACK].WithIndex(SPR_SIDE_FRICTION_DIAG_60_DEG_UP_DIR_3_A),
                        { -16, -16, height }, { { -16, -16, height }, { 32, 32, 2 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 104, 0x20);
            break;
        case 1:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction,
                        session.TrackColours[SCHEME_TRACK].WithIndex(SPR_SIDE_FRICTION_DIAG_60_DEG_UP_DIR_0_A),
                        { -16, -16, height }, { { -16, -16, height }, { 32, 32, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction,
                        session.TrackColours[SCHEME_TRACK].WithIndex(SPR_SIDE_FRICTION_DIAG_60_DEG_UP_DIR_0_B),
                        { -16, -16, height }, { { -16, -16, height + 43 }, { 32, 32, 0 } });
                    WoodenBSupportsPaintSetup(session, 2, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    WoodenBSupportsPaintSetup(session, 3, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    WoodenBSupportsPaintSetup(session, 4, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    WoodenBSupportsPaintSetup(session, 5, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }

            PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 104, 0x20);
            break;
        case 2:
            switch (direction)
            {
                case 0:
                    WoodenBSupportsPaintSetup(session, 4, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    WoodenBSupportsPaintSetup(session, 5, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction,
                        session.TrackColours[SCHEME_TRACK].WithIndex(SPR_SIDE_FRICTION_DIAG_60_DEG_UP_DIR_2_A),
                        { -16, -16, height }, { { -16, -16, height }, { 32, 32, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction,
                        session.TrackColours[SCHEME_TRACK].WithIndex(SPR_SIDE_FRICTION_DIAG_60_DEG_UP_DIR_2_B),
                        { -16, -16, height }, { { -16, -16, height + 43 }, { 32, 32, 0 } });
                    WoodenBSupportsPaintSetup(session, 2, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    WoodenBSupportsPaintSetup(session, 3, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 104, 0x20);
            break;
        case 3:
            switch (direction)
            {
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction,
                        session.TrackColours[SCHEME_TRACK].WithIndex(SPR_SIDE_FRICTION_DIAG_60_DEG_UP_DIR_1_A),
                        { -16, -16, height }, { { -16, -16, height }, { 32, 32, 2 } });
                    break;
            }

            PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 104, 0x20);
            break;
    }
}

static void SideFrictionRCTrackDiag60DegDown(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    SideFrictionRCTrackDiag60DegUp(session, ride, 3 - trackSequence, (direction + 2) % 4, height, trackElement);
}

static void SideFrictionRCTrackDiag60DegUpTo25DegUp(
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
                        session, direction,
                        session.TrackColours[SCHEME_TRACK].WithIndex(SPR_SIDE_FRICTION_DIAG_60_DEG_UP_TO_25_DEG_UP_DIR_3_A),
                        { -16, -16, height }, { { -16, -16, height }, { 32, 32, 2 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 1:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction,
                        session.TrackColours[SCHEME_TRACK].WithIndex(SPR_SIDE_FRICTION_DIAG_60_DEG_UP_TO_25_DEG_UP_DIR_0_A),
                        { -16, -16, height }, { { -16, -16, height }, { 32, 32, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction,
                        session.TrackColours[SCHEME_TRACK].WithIndex(SPR_SIDE_FRICTION_DIAG_60_DEG_UP_TO_25_DEG_UP_DIR_0_B),
                        { -16, -16, height }, { { -16, -16, height + 59 }, { 32, 32, 0 } });
                    WoodenBSupportsPaintSetup(session, 2, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    WoodenBSupportsPaintSetup(session, 3, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    WoodenBSupportsPaintSetup(session, 4, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    WoodenBSupportsPaintSetup(session, 5, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 2:
            switch (direction)
            {
                case 0:
                    WoodenBSupportsPaintSetup(session, 4, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    WoodenBSupportsPaintSetup(session, 5, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction,
                        session.TrackColours[SCHEME_TRACK].WithIndex(SPR_SIDE_FRICTION_DIAG_60_DEG_UP_TO_25_DEG_UP_DIR_2_A),
                        { -16, -16, height }, { { -16, -16, height }, { 32, 32, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction,
                        session.TrackColours[SCHEME_TRACK].WithIndex(SPR_SIDE_FRICTION_DIAG_60_DEG_UP_TO_25_DEG_UP_DIR_2_B),
                        { -16, -16, height }, { { -16, -16, height + 59 }, { 32, 32, 0 } });
                    WoodenBSupportsPaintSetup(session, 2, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    WoodenBSupportsPaintSetup(session, 3, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 3:
            switch (direction)
            {
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction,
                        session.TrackColours[SCHEME_TRACK].WithIndex(SPR_SIDE_FRICTION_DIAG_60_DEG_UP_TO_25_DEG_UP_DIR_1_A),
                        { -16, -16, height }, { { 0, 0, height }, { 16, 16, 2 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
    }
}

static void SideFrictionRCTrackDiag25DegDownTo60DegDown(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    SideFrictionRCTrackDiag60DegUpTo25DegUp(session, ride, 3 - trackSequence, (direction + 2) & 3, height, trackElement);
};

static void SideFrictionRCTrackDiag25DegUpTo60DegUp(
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
                        session, direction,
                        session.TrackColours[SCHEME_TRACK].WithIndex(SPR_SIDE_FRICTION_DIAG_25_DEG_UP_TO_60_DEG_UP_DIR_3_A),
                        { -16 + 4, -16 + 2, height }, { { -16, -16, height }, { 32, 32, 2 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 1:
            switch (direction)
            {
                case 0:
                    PaintAddImageAsParentRotated(
                        session, direction,
                        session.TrackColours[SCHEME_TRACK].WithIndex(SPR_SIDE_FRICTION_DIAG_25_DEG_UP_TO_60_DEG_UP_DIR_0_A),
                        { -16, -16, height }, { { -16, -16, height }, { 32, 32, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(24033), { -16, -16, height },
                        { { -16, -16, height + 59 }, { 32, 32, 0 } });
                    WoodenBSupportsPaintSetup(session, 2, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    WoodenBSupportsPaintSetup(session, 3, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    WoodenBSupportsPaintSetup(session, 4, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    WoodenBSupportsPaintSetup(session, 5, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 2:
            switch (direction)
            {
                case 0:
                    WoodenBSupportsPaintSetup(session, 4, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 1:
                    WoodenBSupportsPaintSetup(session, 5, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 2:
                    PaintAddImageAsParentRotated(
                        session, direction,
                        session.TrackColours[SCHEME_TRACK].WithIndex(SPR_SIDE_FRICTION_DIAG_25_DEG_UP_TO_60_DEG_UP_DIR_2_A),
                        { -16, -16, height }, { { -16, -16, height }, { 32, 32, 2 } });
                    PaintAddImageAsParentRotated(
                        session, direction, session.TrackColours[SCHEME_TRACK].WithIndex(24034), { -16, -16, height },
                        { { -16, -16, height + 59 }, { 32, 32, 0 } });
                    WoodenBSupportsPaintSetup(session, 2, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
                case 3:
                    WoodenBSupportsPaintSetup(session, 3, 0, height + 16, session.TrackColours[SCHEME_SUPPORTS]);
                    break;
            }
            PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
        case 3:
            switch (direction)
            {
                case 1:
                    PaintAddImageAsParentRotated(
                        session, direction,
                        session.TrackColours[SCHEME_TRACK].WithIndex(SPR_SIDE_FRICTION_DIAG_25_DEG_UP_TO_60_DEG_UP_DIR_1_A),
                        { -16, -16, height }, { { -16, -16, height }, { 32, 32, 2 } });
                    break;
            }
            PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
            PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
            break;
    }
}

static void SideFrictionRCTrackDiag60DegDownTo25DegDown(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    SideFrictionRCTrackDiag25DegUpTo60DegUp(session, ride, 3 - trackSequence, (direction + 2) & 3, height, trackElement);
};

TRACK_PAINT_FUNCTION GetTrackPaintFunctionSideFrictionRC(int32_t trackType)
{
    switch (trackType)
    {
        case TrackElemType::Flat:
            return SideFrictionRCTrackFlat;
        case TrackElemType::EndStation:
        case TrackElemType::BeginStation:
        case TrackElemType::MiddleStation:
            return SideFrictionRCTrackStation;
        case TrackElemType::Up25:
            return SideFrictionRCTrack25DegUp;
        case TrackElemType::FlatToUp25:
            return SideFrictionRCTrackFlatTo25DegUp;
        case TrackElemType::Up25ToFlat:
            return SideFrictionRCTrack25DegUpToFlat;
        case TrackElemType::Down25:
            return SideFrictionRCTrack25DegDown;
        case TrackElemType::FlatToDown25:
            return SideFrictionRCTrackFlatTo25DegDown;
        case TrackElemType::Down25ToFlat:
            return SideFrictionRCTrack25DegDownToFlat;
        case TrackElemType::LeftQuarterTurn5Tiles:
            return SideFrictionRCTrackLeftQuarterTurn5;
        case TrackElemType::RightQuarterTurn5Tiles:
            return SideFrictionRCTrackRightQuarterTurn5;
        case TrackElemType::SBendLeft:
            return SideFrictionRCTrackSBendLeft;
        case TrackElemType::SBendRight:
            return SideFrictionRCTrackSBendRight;
        case TrackElemType::LeftQuarterTurn3Tiles:
            return SideFrictionRCTrackLeftQuarterTurn3;
        case TrackElemType::RightQuarterTurn3Tiles:
            return SideFrictionRCTrackRightQuarterTurn3;
        case TrackElemType::Brakes:
            return SideFrictionRCTrackBrakes;
        case TrackElemType::LeftEighthToDiag:
            return SideFrictionRCTrackLeftEighthToDiag;
        case TrackElemType::RightEighthToDiag:
            return SideFrictionRCTrackRightEighthToDiag;
        case TrackElemType::LeftEighthToOrthogonal:
            return SideFrictionRCTrackLeftEighthToOrthogonal;
        case TrackElemType::RightEighthToOrthogonal:
            return SideFrictionRCTrackRightEighthToOrthogonal;
        case TrackElemType::DiagFlat:
            return SideFrictionRCTrackDiagFlat;
        case TrackElemType::DiagUp25:
            return SideFrictionRCTrackDiag25DegUp;
        case TrackElemType::DiagFlatToUp25:
            return SideFrictionRCTrackDiagFlatTo25DegUp;
        case TrackElemType::DiagUp25ToFlat:
            return SideFrictionRCTrackDiag25DegUpToFlat;
        case TrackElemType::DiagDown25:
            return SideFrictionRCTrackDiag25DegDown;
        case TrackElemType::DiagFlatToDown25:
            return SideFrictionRCTrackDiagFlatTo25DegDown;
        case TrackElemType::DiagDown25ToFlat:
            return SideFrictionRCTrackDiag25DegDownToFlat;

        // Added by OpenRCT2
        case TrackElemType::Down25ToDown60:
            return SideFrictionRCTrack25DegDownTo60DegDown;
        case TrackElemType::Down60ToDown25:
            return SideFrictionRCTrack60DegDownTo25DegDown;
        case TrackElemType::Up25ToUp60:
            return SideFrictionRCTrack25DegUpTo60DegUp;
        case TrackElemType::Up60ToUp25:
            return SideFrictionRCTrack60DegUpTo25DegUp;
        case TrackElemType::Up60:
            return SideFrictionRCTrack60DegUp;
        case TrackElemType::Down60:
            return SideFrictionRCTrack60DegDown;

        case TrackElemType::DiagUp60:
            return SideFrictionRCTrackDiag60DegUp;
        case TrackElemType::DiagDown60:
            return SideFrictionRCTrackDiag60DegDown;
        case TrackElemType::DiagUp60ToUp25:
            return SideFrictionRCTrackDiag60DegUpTo25DegUp;
        case TrackElemType::DiagDown25ToDown60:
            return SideFrictionRCTrackDiag25DegDownTo60DegDown;
        case TrackElemType::DiagUp25ToUp60:
            return SideFrictionRCTrackDiag25DegUpTo60DegUp;
        case TrackElemType::DiagDown60ToDown25:
            return SideFrictionRCTrackDiag60DegDownTo25DegDown;
    }
    return nullptr;
}
