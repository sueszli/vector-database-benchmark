/*****************************************************************************
 * Copyright (c) 2014-2023 OpenRCT2 developers
 *
 * For a complete list of all authors, please refer to contributors.md
 * Interested in contributing? Visit https://github.com/OpenRCT2/OpenRCT2
 *
 * OpenRCT2 is licensed under the GNU General Public License version 3.
 *****************************************************************************/

#include "../../common.h"
#include "../../interface/Viewport.h"
#include "../../paint/Paint.h"
#include "../../paint/Supports.h"
#include "../../sprites.h"
#include "../../world/Map.h"
#include "../RideData.h"
#include "../Track.h"
#include "../TrackPaint.h"
#include "../VehiclePaint.h"

enum
{
    SPR_MINIATURE_RAILWAY_FLAT_SW_NE = 23341,
    SPR_MINIATURE_RAILWAY_FLAT_NW_SE = 23342,
    SPR_MINIATURE_RAILWAY_QUARTER_TURN_5_TILES_SW_SE_PART_0 = 23343,
    SPR_MINIATURE_RAILWAY_QUARTER_TURN_5_TILES_SW_SE_PART_1 = 23344,
    SPR_MINIATURE_RAILWAY_QUARTER_TURN_5_TILES_SW_SE_PART_2 = 23345,
    SPR_MINIATURE_RAILWAY_QUARTER_TURN_5_TILES_SW_SE_PART_3 = 23346,
    SPR_MINIATURE_RAILWAY_QUARTER_TURN_5_TILES_SW_SE_PART_4 = 23347,
    SPR_MINIATURE_RAILWAY_QUARTER_TURN_5_TILES_NW_SW_PART_0 = 23348,
    SPR_MINIATURE_RAILWAY_QUARTER_TURN_5_TILES_NW_SW_PART_1 = 23349,
    SPR_MINIATURE_RAILWAY_QUARTER_TURN_5_TILES_NW_SW_PART_2 = 23350,
    SPR_MINIATURE_RAILWAY_QUARTER_TURN_5_TILES_NW_SW_PART_3 = 23351,
    SPR_MINIATURE_RAILWAY_QUARTER_TURN_5_TILES_NW_SW_PART_4 = 23352,
    SPR_MINIATURE_RAILWAY_QUARTER_TURN_5_TILES_NE_NW_PART_0 = 23353,
    SPR_MINIATURE_RAILWAY_QUARTER_TURN_5_TILES_NE_NW_PART_1 = 23354,
    SPR_MINIATURE_RAILWAY_QUARTER_TURN_5_TILES_NE_NW_PART_2 = 23355,
    SPR_MINIATURE_RAILWAY_QUARTER_TURN_5_TILES_NE_NW_PART_3 = 23356,
    SPR_MINIATURE_RAILWAY_QUARTER_TURN_5_TILES_NE_NW_PART_4 = 23357,
    SPR_MINIATURE_RAILWAY_QUARTER_TURN_5_TILES_SE_NE_PART_0 = 23358,
    SPR_MINIATURE_RAILWAY_QUARTER_TURN_5_TILES_SE_NE_PART_1 = 23359,
    SPR_MINIATURE_RAILWAY_QUARTER_TURN_5_TILES_SE_NE_PART_2 = 23360,
    SPR_MINIATURE_RAILWAY_QUARTER_TURN_5_TILES_SE_NE_PART_3 = 23361,
    SPR_MINIATURE_RAILWAY_QUARTER_TURN_5_TILES_SE_NE_PART_4 = 23362,
    SPR_MINIATURE_RAILWAY_S_BEND_RIGHT_SW_NE_PART_0 = 23363,
    SPR_MINIATURE_RAILWAY_S_BEND_RIGHT_SW_NE_PART_1 = 23364,
    SPR_MINIATURE_RAILWAY_S_BEND_RIGHT_SW_NE_PART_2 = 23365,
    SPR_MINIATURE_RAILWAY_S_BEND_RIGHT_SW_NE_PART_3 = 23366,
    SPR_MINIATURE_RAILWAY_S_BEND_LEFT_SW_NE_PART_0 = 23367,
    SPR_MINIATURE_RAILWAY_S_BEND_LEFT_SW_NE_PART_1 = 23368,
    SPR_MINIATURE_RAILWAY_S_BEND_LEFT_SW_NE_PART_2 = 23369,
    SPR_MINIATURE_RAILWAY_S_BEND_LEFT_SW_NE_PART_3 = 23370,
    SPR_MINIATURE_RAILWAY_S_BEND_LEFT_SE_NW_PART_0 = 23371,
    SPR_MINIATURE_RAILWAY_S_BEND_LEFT_SE_NW_PART_1 = 23372,
    SPR_MINIATURE_RAILWAY_S_BEND_LEFT_SE_NW_PART_2 = 23373,
    SPR_MINIATURE_RAILWAY_S_BEND_LEFT_SE_NW_PART_3 = 23374,
    SPR_MINIATURE_RAILWAY_S_BEND_RIGHT_SE_NW_PART_0 = 23375,
    SPR_MINIATURE_RAILWAY_S_BEND_RIGHT_SE_NW_PART_1 = 23376,
    SPR_MINIATURE_RAILWAY_S_BEND_RIGHT_SE_NW_PART_2 = 23377,
    SPR_MINIATURE_RAILWAY_S_BEND_RIGHT_SE_NW_PART_3 = 23378,
    SPR_MINIATURE_RAILWAY_25_DEG_UP_SW_NE = 23379,
    SPR_MINIATURE_RAILWAY_FLAT_TO_25_DEG_UP_SW_NE = 23380,
    SPR_MINIATURE_RAILWAY_25_DEG_UP_NE_SW = 23381,
    SPR_MINIATURE_RAILWAY_FLAT_TO_25_DEG_UP_NE_SW = 23382,
    SPR_MINIATURE_RAILWAY_25_DEG_UP_TO_FLAT_NE_SW = 23383,
    SPR_MINIATURE_RAILWAY_25_DEG_UP_TO_FLAT_SW_NE = 23384,
    SPR_MINIATURE_RAILWAY_25_DEG_UP_NW_SE = 23385,
    SPR_MINIATURE_RAILWAY_25_DEG_UP_SE_NW = 23386,
    SPR_MINIATURE_RAILWAY_FLAT_TO_25_DEG_UP_NW_SE = 23387,
    SPR_MINIATURE_RAILWAY_FLAT_TO_25_DEG_UP_SE_NW = 23388,
    SPR_MINIATURE_RAILWAY_25_DEG_UP_TO_FLAT_SE_NW = 23389,
    SPR_MINIATURE_RAILWAY_25_DEG_UP_TO_FLAT_NW_SE = 23390,
    SPR_MINIATURE_RAILWAY_QUARTER_TURN_3_TILES_SW_SE_PART_0 = 23391,
    SPR_MINIATURE_RAILWAY_QUARTER_TURN_3_TILES_SW_SE_PART_1 = 23392,
    SPR_MINIATURE_RAILWAY_QUARTER_TURN_3_TILES_SW_SE_PART_2 = 23393,
    SPR_MINIATURE_RAILWAY_QUARTER_TURN_3_TILES_NW_SW_PART_0 = 23394,
    SPR_MINIATURE_RAILWAY_QUARTER_TURN_3_TILES_NW_SW_PART_1 = 23395,
    SPR_MINIATURE_RAILWAY_QUARTER_TURN_3_TILES_NW_SW_PART_2 = 23396,
    SPR_MINIATURE_RAILWAY_QUARTER_TURN_3_TILES_NE_NW_PART_0 = 23397,
    SPR_MINIATURE_RAILWAY_QUARTER_TURN_3_TILES_NE_NW_PART_1 = 23398,
    SPR_MINIATURE_RAILWAY_QUARTER_TURN_3_TILES_NE_NW_PART_2 = 23399,
    SPR_MINIATURE_RAILWAY_QUARTER_TURN_3_TILES_SE_NE_PART_0 = 23400,
    SPR_MINIATURE_RAILWAY_QUARTER_TURN_3_TILES_SE_NE_PART_1 = 23401,
    SPR_MINIATURE_RAILWAY_QUARTER_TURN_3_TILES_SE_NE_PART_2 = 23402,
    SPR_MINIATURE_RAILWAY_FLAT_NO_BASE_SW_NE = 23403,
    SPR_MINIATURE_RAILWAY_FLAT_NO_BASE_NW_SE = 23404,
    SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_SW_E_PART_0 = 23405,
    SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_SW_E_PART_1 = 23406,
    SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_SW_E_PART_2 = 23407,
    SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_SW_E_PART_3 = 23408,
    SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_NW_S_PART_0 = 23409,
    SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_NW_S_PART_1 = 23410,
    SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_NW_S_PART_2 = 23411,
    SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_NW_S_PART_3 = 23412,
    SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_NE_W_PART_0 = 23413,
    SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_NE_W_PART_1 = 23414,
    SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_NE_W_PART_2 = 23415,
    SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_NE_W_PART_3 = 23416,
    SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_SE_N_PART_0 = 23417,
    SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_SE_N_PART_1 = 23418,
    SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_SE_N_PART_2 = 23419,
    SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_SE_N_PART_3 = 23420,
    SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_SW_N_PART_0 = 23421,
    SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_SW_N_PART_1 = 23422,
    SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_SW_N_PART_2 = 23423,
    SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_SW_N_PART_3 = 23424,
    SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_NW_E_PART_0 = 23425,
    SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_NW_E_PART_1 = 23426,
    SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_NW_E_PART_2 = 23427,
    SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_NW_E_PART_3 = 23428,
    SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_NE_S_PART_0 = 23429,
    SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_NE_S_PART_1 = 23430,
    SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_NE_S_PART_2 = 23431,
    SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_NE_S_PART_3 = 23432,
    SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_SE_W_PART_0 = 23433,
    SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_SE_W_PART_1 = 23434,
    SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_SE_W_PART_2 = 23435,
    SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_SE_W_PART_3 = 23436,
    SPR_MINIATURE_RAILWAY_DIAG_FLAT_W_E = 23437,
    SPR_MINIATURE_RAILWAY_DIAG_FLAT_N_S = 23438,
    SPR_MINIATURE_RAILWAY_DIAG_FLAT_E_W = 23439,
    SPR_MINIATURE_RAILWAY_DIAG_FLAT_S_N = 23440,
    SPR_MINIATURE_RAILWAY_DIAG_FLAT_TO_25_DEG_UP_W_E = 23441,
    SPR_MINIATURE_RAILWAY_DIAG_FLAT_TO_25_DEG_UP_N_S = 23442,
    SPR_MINIATURE_RAILWAY_DIAG_FLAT_TO_25_DEG_UP_E_W = 23443,
    SPR_MINIATURE_RAILWAY_DIAG_FLAT_TO_25_DEG_UP_S_N = 23444,
    SPR_MINIATURE_RAILWAY_DIAG_25_DEG_UP_TO_FLAT_W_E = 23445,
    SPR_MINIATURE_RAILWAY_DIAG_25_DEG_UP_TO_FLAT_N_S = 23446,
    SPR_MINIATURE_RAILWAY_DIAG_25_DEG_UP_TO_FLAT_E_W = 23447,
    SPR_MINIATURE_RAILWAY_DIAG_25_DEG_UP_TO_FLAT_S_N = 23448,
    SPR_MINIATURE_RAILWAY_DIAG_25_DEG_UP_W_E = 23449,
    SPR_MINIATURE_RAILWAY_DIAG_25_DEG_UP_N_S = 23450,
    SPR_MINIATURE_RAILWAY_DIAG_25_DEG_UP_E_W = 23451,
    SPR_MINIATURE_RAILWAY_DIAG_25_DEG_UP_S_N = 23452,
};

static constexpr uint32_t miniature_railway_track_floor[4] = {
    SPR_FLOOR_PLANKS,
    SPR_FLOOR_PLANKS_90_DEG,
    SPR_FLOOR_PLANKS,
    SPR_FLOOR_PLANKS_90_DEG,
};

static constexpr uint32_t miniature_railway_track_pieces_flat[4] = {
    SPR_MINIATURE_RAILWAY_FLAT_SW_NE,
    SPR_MINIATURE_RAILWAY_FLAT_NW_SE,
    SPR_MINIATURE_RAILWAY_FLAT_SW_NE,
    SPR_MINIATURE_RAILWAY_FLAT_NW_SE,
};

static constexpr uint32_t miniature_railway_station_floor[4] = {
    SPR_STATION_BASE_A_SW_NE,
    SPR_STATION_BASE_A_NW_SE,
    SPR_STATION_BASE_A_SW_NE,
    SPR_STATION_BASE_A_NW_SE,
};

static constexpr uint32_t miniature_railway_track_pieces_flat_station[4] = {
    SPR_MINIATURE_RAILWAY_FLAT_NO_BASE_SW_NE,
    SPR_MINIATURE_RAILWAY_FLAT_NO_BASE_NW_SE,
    SPR_MINIATURE_RAILWAY_FLAT_NO_BASE_SW_NE,
    SPR_MINIATURE_RAILWAY_FLAT_NO_BASE_NW_SE,
};

static constexpr uint32_t miniature_railway_track_pieces_flat_quarter_turn_5_tiles[4][5] = {
    {
        SPR_MINIATURE_RAILWAY_QUARTER_TURN_5_TILES_SW_SE_PART_0,
        SPR_MINIATURE_RAILWAY_QUARTER_TURN_5_TILES_SW_SE_PART_1,
        SPR_MINIATURE_RAILWAY_QUARTER_TURN_5_TILES_SW_SE_PART_2,
        SPR_MINIATURE_RAILWAY_QUARTER_TURN_5_TILES_SW_SE_PART_3,
        SPR_MINIATURE_RAILWAY_QUARTER_TURN_5_TILES_SW_SE_PART_4,
    },
    {
        SPR_MINIATURE_RAILWAY_QUARTER_TURN_5_TILES_NW_SW_PART_0,
        SPR_MINIATURE_RAILWAY_QUARTER_TURN_5_TILES_NW_SW_PART_1,
        SPR_MINIATURE_RAILWAY_QUARTER_TURN_5_TILES_NW_SW_PART_2,
        SPR_MINIATURE_RAILWAY_QUARTER_TURN_5_TILES_NW_SW_PART_3,
        SPR_MINIATURE_RAILWAY_QUARTER_TURN_5_TILES_NW_SW_PART_4,
    },
    {
        SPR_MINIATURE_RAILWAY_QUARTER_TURN_5_TILES_NE_NW_PART_0,
        SPR_MINIATURE_RAILWAY_QUARTER_TURN_5_TILES_NE_NW_PART_1,
        SPR_MINIATURE_RAILWAY_QUARTER_TURN_5_TILES_NE_NW_PART_2,
        SPR_MINIATURE_RAILWAY_QUARTER_TURN_5_TILES_NE_NW_PART_3,
        SPR_MINIATURE_RAILWAY_QUARTER_TURN_5_TILES_NE_NW_PART_4,
    },
    {
        SPR_MINIATURE_RAILWAY_QUARTER_TURN_5_TILES_SE_NE_PART_0,
        SPR_MINIATURE_RAILWAY_QUARTER_TURN_5_TILES_SE_NE_PART_1,
        SPR_MINIATURE_RAILWAY_QUARTER_TURN_5_TILES_SE_NE_PART_2,
        SPR_MINIATURE_RAILWAY_QUARTER_TURN_5_TILES_SE_NE_PART_3,
        SPR_MINIATURE_RAILWAY_QUARTER_TURN_5_TILES_SE_NE_PART_4,
    },
};

static constexpr uint32_t miniature_railway_track_pieces_25_deg_up[4] = {
    SPR_MINIATURE_RAILWAY_25_DEG_UP_SW_NE,
    SPR_MINIATURE_RAILWAY_25_DEG_UP_NW_SE,
    SPR_MINIATURE_RAILWAY_25_DEG_UP_NE_SW,
    SPR_MINIATURE_RAILWAY_25_DEG_UP_SE_NW,
};

static constexpr uint32_t miniature_railway_track_pieces_flat_to_25_deg_up[4] = {
    SPR_MINIATURE_RAILWAY_FLAT_TO_25_DEG_UP_SW_NE,
    SPR_MINIATURE_RAILWAY_FLAT_TO_25_DEG_UP_NW_SE,
    SPR_MINIATURE_RAILWAY_FLAT_TO_25_DEG_UP_NE_SW,
    SPR_MINIATURE_RAILWAY_FLAT_TO_25_DEG_UP_SE_NW,
};

static constexpr uint32_t miniature_railway_track_pieces_25_deg_up_to_flat[4] = {
    SPR_MINIATURE_RAILWAY_25_DEG_UP_TO_FLAT_SW_NE,
    SPR_MINIATURE_RAILWAY_25_DEG_UP_TO_FLAT_NW_SE,
    SPR_MINIATURE_RAILWAY_25_DEG_UP_TO_FLAT_NE_SW,
    SPR_MINIATURE_RAILWAY_25_DEG_UP_TO_FLAT_SE_NW,
};

static constexpr uint32_t miniature_railway_track_pieces_s_bend_left[2][4] = {
    {
        SPR_MINIATURE_RAILWAY_S_BEND_LEFT_SW_NE_PART_0,
        SPR_MINIATURE_RAILWAY_S_BEND_LEFT_SW_NE_PART_1,
        SPR_MINIATURE_RAILWAY_S_BEND_LEFT_SW_NE_PART_2,
        SPR_MINIATURE_RAILWAY_S_BEND_LEFT_SW_NE_PART_3,
    },
    {
        SPR_MINIATURE_RAILWAY_S_BEND_LEFT_SE_NW_PART_3,
        SPR_MINIATURE_RAILWAY_S_BEND_LEFT_SE_NW_PART_2,
        SPR_MINIATURE_RAILWAY_S_BEND_LEFT_SE_NW_PART_1,
        SPR_MINIATURE_RAILWAY_S_BEND_LEFT_SE_NW_PART_0,
    },
};

static constexpr uint32_t miniature_railway_track_pieces_s_bend_right[2][4] = {
    {
        SPR_MINIATURE_RAILWAY_S_BEND_RIGHT_SW_NE_PART_0,
        SPR_MINIATURE_RAILWAY_S_BEND_RIGHT_SW_NE_PART_1,
        SPR_MINIATURE_RAILWAY_S_BEND_RIGHT_SW_NE_PART_2,
        SPR_MINIATURE_RAILWAY_S_BEND_RIGHT_SW_NE_PART_3,
    },
    {
        SPR_MINIATURE_RAILWAY_S_BEND_RIGHT_SE_NW_PART_3,
        SPR_MINIATURE_RAILWAY_S_BEND_RIGHT_SE_NW_PART_2,
        SPR_MINIATURE_RAILWAY_S_BEND_RIGHT_SE_NW_PART_1,
        SPR_MINIATURE_RAILWAY_S_BEND_RIGHT_SE_NW_PART_0,
    },
};

static constexpr uint32_t miniature_railway_track_pieces_flat_quarter_turn_3_tiles[4][3] = {
    {
        SPR_MINIATURE_RAILWAY_QUARTER_TURN_3_TILES_SW_SE_PART_0,
        SPR_MINIATURE_RAILWAY_QUARTER_TURN_3_TILES_SW_SE_PART_1,
        SPR_MINIATURE_RAILWAY_QUARTER_TURN_3_TILES_SW_SE_PART_2,
    },
    {
        SPR_MINIATURE_RAILWAY_QUARTER_TURN_3_TILES_NW_SW_PART_0,
        SPR_MINIATURE_RAILWAY_QUARTER_TURN_3_TILES_NW_SW_PART_1,
        SPR_MINIATURE_RAILWAY_QUARTER_TURN_3_TILES_NW_SW_PART_2,
    },
    {
        SPR_MINIATURE_RAILWAY_QUARTER_TURN_3_TILES_NE_NW_PART_0,
        SPR_MINIATURE_RAILWAY_QUARTER_TURN_3_TILES_NE_NW_PART_1,
        SPR_MINIATURE_RAILWAY_QUARTER_TURN_3_TILES_NE_NW_PART_2,
    },
    {
        SPR_MINIATURE_RAILWAY_QUARTER_TURN_3_TILES_SE_NE_PART_0,
        SPR_MINIATURE_RAILWAY_QUARTER_TURN_3_TILES_SE_NE_PART_1,
        SPR_MINIATURE_RAILWAY_QUARTER_TURN_3_TILES_SE_NE_PART_2,
    },
};

static constexpr uint32_t miniature_railway_track_pieces_right_eight_to_diag[4][4] = {
    {
        SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_SW_E_PART_0,
        SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_SW_E_PART_1,
        SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_SW_E_PART_2,
        SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_SW_E_PART_3,
    },
    {
        SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_NW_S_PART_0,
        SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_NW_S_PART_1,
        SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_NW_S_PART_2,
        SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_NW_S_PART_3,
    },
    {
        SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_NE_W_PART_0,
        SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_NE_W_PART_1,
        SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_NE_W_PART_2,
        SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_NE_W_PART_3,
    },
    {
        SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_SE_N_PART_0,
        SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_SE_N_PART_1,
        SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_SE_N_PART_2,
        SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_SE_N_PART_3,
    },
};

static constexpr CoordsXYZ miniature_railway_track_pieces_right_eight_to_diag_bounds[4][4] = {
    {
        { 32, 32, 2 },
        { 32, 16, 2 },
        { 16, 16, 2 },
        { 32, 32, 0 },
    },
    {
        { 32, 32, 2 },
        { 16, 32, 2 },
        { 16, 16, 2 },
        { 16, 16, 2 },
    },
    {
        { 32, 32, 2 },
        { 34, 16, 2 },
        { 28, 28, 2 },
        { 32, 34, 0 },
    },
    {
        { 32, 32, 2 },
        { 16, 32, 2 },
        { 16, 16, 2 },
        { 30, 30, 0 },
    },
};

static constexpr CoordsXY miniature_railway_track_pieces_right_eight_to_diag_offset[4][4] = {
    {
        { 0, 0 },
        { 0, 16 },
        { 0, 0 },
        { 0, 0 },
    },
    {
        { 0, 0 },
        { 16, 0 },
        { 0, 16 },
        { 0, 0 },
    },
    {
        { 0, 0 },
        { 0, 0 },
        { 4, 4 },
        { 0, 0 },
    },
    {
        { 0, 0 },
        { 0, 0 },
        { 16, 0 },
        { 0, 0 },
    },
};

static constexpr uint32_t miniature_railway_track_pieces_left_eight_to_diag[4][4] = {
    {
        SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_SW_N_PART_0,
        SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_SW_N_PART_1,
        SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_SW_N_PART_2,
        SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_SW_N_PART_3,
    },
    {
        SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_NW_E_PART_0,
        SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_NW_E_PART_1,
        SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_NW_E_PART_2,
        SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_NW_E_PART_3,
    },
    {
        SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_NE_S_PART_0,
        SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_NE_S_PART_1,
        SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_NE_S_PART_2,
        SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_NE_S_PART_3,
    },
    {
        SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_SE_W_PART_0,
        SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_SE_W_PART_1,
        SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_SE_W_PART_2,
        SPR_MINIATURE_RAILWAY_EIGHT_TO_DIAG_SE_W_PART_3,
    },
};

static constexpr CoordsXYZ miniature_railway_track_pieces_left_eight_to_diag_bounds[4][4] = {
    {
        { 32, 32, 2 },
        { 32, 16, 2 },
        { 16, 16, 2 },
        { 30, 30, 0 },
    },
    {
        { 32, 32, 2 },
        { 16, 34, 2 },
        { 14, 14, 2 },
        { 34, 32, 0 },
    },
    {
        { 32, 32, 2 },
        { 32, 16, 2 },
        { 16, 16, 2 },
        { 16, 16, 2 },
    },
    {
        { 32, 32, 2 },
        { 16, 32, 2 },
        { 16, 16, 2 },
        { 32, 32, 0 },
    },
};

static constexpr CoordsXY miniature_railway_track_pieces_left_eight_to_diag_offset[4][4] = {
    {
        { 0, 0 },
        { 0, 0 },
        { 0, 16 },
        { 0, 0 },
    },
    {
        { 0, 0 },
        { 0, 0 },
        { 16, 16 },
        { 0, 0 },
    },
    {
        { 0, 0 },
        { 0, 16 },
        { 16, 0 },
        { 0, 0 },
    },
    {
        { 0, 0 },
        { 16, 0 },
        { 0, 0 },
        { 0, 0 },
    },
};

static constexpr CoordsXYZ miniature_railway_track_pieces_right_eight_to_orthog_bounds[4][4] = {
    {
        { 32, 32, 2 },
        { 32, 16, 2 },
        { 16, 16, 2 },
        { 14, 14, 2 },
    },
    {
        { 32, 32, 2 },
        { 16, 34, 2 },
        { 14, 14, 2 },
        { 18, 16, 2 },
    },
    {
        { 32, 32, 2 },
        { 32, 16, 2 },
        { 16, 16, 2 },
        { 16, 16, 2 },
    },
    {
        { 32, 32, 2 },
        { 16, 32, 2 },
        { 16, 16, 2 },
        { 16, 16, 2 },
    },
};

static constexpr CoordsXY miniature_railway_track_pieces_right_eight_to_orthog_offset[4][4] = {
    {
        { 0, 0 },
        { 0, 0 },
        { 0, 16 },
        { 16, 16 },
    },
    {
        { 0, 0 },
        { 0, 0 },
        { 16, 16 },
        { 16, 0 },
    },
    {
        { 0, 0 },
        { 0, 16 },
        { 16, 0 },
        { 0, 0 },
    },
    {
        { 0, 0 },
        { 16, 0 },
        { 0, 0 },
        { 0, 16 },
    },
};

static constexpr CoordsXYZ miniature_railway_track_pieces_left_eight_to_orthog_bounds[4][4] = {
    {
        { 32, 32, 2 },
        { 32, 16, 2 },
        { 16, 16, 2 },
        { 16, 16, 2 },
    },
    {
        { 32, 32, 2 },
        { 16, 32, 2 },
        { 16, 16, 2 },
        { 16, 16, 2 },
    },
    {
        { 32, 32, 2 },
        { 34, 16, 2 },
        { 28, 28, 2 },
        { 16, 18, 2 },
    },
    {
        { 32, 32, 2 },
        { 16, 32, 2 },
        { 16, 16, 2 },
        { 14, 14, 2 },
    },
};

static constexpr CoordsXY miniature_railway_track_pieces_left_eight_to_orthog_offset[4][4] = {
    {
        { 0, 0 },
        { 0, 16 },
        { 0, 0 },
        { 16, 0 },
    },
    {
        { 0, 0 },
        { 16, 0 },
        { 0, 16 },
        { 0, 0 },
    },
    {
        { 0, 0 },
        { 0, 0 },
        { 4, 4 },
        { 0, 16 },
    },
    {
        { 0, 0 },
        { 0, 0 },
        { 16, 0 },
        { 16, 16 },
    },
};

static constexpr uint32_t miniature_railway_track_pieces_diag_flat[4] = {
    SPR_MINIATURE_RAILWAY_DIAG_FLAT_W_E,
    SPR_MINIATURE_RAILWAY_DIAG_FLAT_N_S,
    SPR_MINIATURE_RAILWAY_DIAG_FLAT_E_W,
    SPR_MINIATURE_RAILWAY_DIAG_FLAT_S_N,
};

static constexpr uint32_t miniature_railway_track_pieces_diag_flat_to_25_deg_up[4] = {
    SPR_MINIATURE_RAILWAY_DIAG_FLAT_TO_25_DEG_UP_W_E,
    SPR_MINIATURE_RAILWAY_DIAG_FLAT_TO_25_DEG_UP_N_S,
    SPR_MINIATURE_RAILWAY_DIAG_FLAT_TO_25_DEG_UP_E_W,
    SPR_MINIATURE_RAILWAY_DIAG_FLAT_TO_25_DEG_UP_S_N,
};

static constexpr uint32_t miniature_railway_track_pieces_diag_25_deg_up_to_flat[4] = {
    SPR_MINIATURE_RAILWAY_DIAG_25_DEG_UP_TO_FLAT_W_E,
    SPR_MINIATURE_RAILWAY_DIAG_25_DEG_UP_TO_FLAT_N_S,
    SPR_MINIATURE_RAILWAY_DIAG_25_DEG_UP_TO_FLAT_E_W,
    SPR_MINIATURE_RAILWAY_DIAG_25_DEG_UP_TO_FLAT_S_N,
};

static constexpr uint32_t miniature_railway_track_pieces_diag_25_deg_up[4] = {
    SPR_MINIATURE_RAILWAY_DIAG_25_DEG_UP_W_E,
    SPR_MINIATURE_RAILWAY_DIAG_25_DEG_UP_N_S,
    SPR_MINIATURE_RAILWAY_DIAG_25_DEG_UP_E_W,
    SPR_MINIATURE_RAILWAY_DIAG_25_DEG_UP_S_N,
};

static ImageId MiniatureRailwayTrackToGravel(ImageId imageId)
{
    return ImageId(imageId.GetIndex() - SPR_MINIATURE_RAILWAY_FLAT_SW_NE + SPR_G2_MINIATURE_RAILWAY_GRAVEL_SW_NE);
}

static ImageId MiniatureRailwayTrackToGrooved(ImageId imageId)
{
    return ImageId(imageId.GetIndex() - SPR_MINIATURE_RAILWAY_FLAT_SW_NE + SPR_G2_MINIATURE_RAILWAY_GROOVED_SW_NE);
}

static ImageId MiniatureRailwayTrackToGroovedIndent(const TileElement* path, uint8_t direction, uint8_t rotation)
{
    if (path == nullptr)
    {
        return ImageId(0);
    }

    uint32_t imageIdAlt = SPR_G2_MINIATURE_RAILWAY_GROOVED_SW_NE;

    uint8_t correctedEdges = path->AsPath()->GetEdges();
    correctedEdges |= correctedEdges << 4;
    correctedEdges >>= 4 - rotation;
    correctedEdges &= 0x0F;

    if (direction & 0x1)
    {
        static constexpr uint32_t imageIds[2][2] = {
            {
                SPR_G2_MINIATURE_RAILWAY_INSET_NW_SE,
                SPR_G2_MINIATURE_RAILWAY_INSET_END_NW,
            },
            {
                SPR_G2_MINIATURE_RAILWAY_INSET_END_SE,
                SPR_G2_MINIATURE_RAILWAY_INSET_END_NW_SE,
            },
        };
        imageIdAlt = imageIds[(correctedEdges & 0x2) ? 0 : 1][(correctedEdges & 0x8) ? 0 : 1];
    }
    else
    {
        static constexpr uint32_t imageIds[2][2] = {
            {
                SPR_G2_MINIATURE_RAILWAY_INSET_SW_NE,
                SPR_G2_MINIATURE_RAILWAY_INSET_END_SW,
            },
            {
                SPR_G2_MINIATURE_RAILWAY_INSET_END_NE,
                SPR_G2_MINIATURE_RAILWAY_INSET_END_SW_NE,
            },
        };
        imageIdAlt = imageIds[(correctedEdges & 0x1) ? 0 : 1][(correctedEdges & 0x4) ? 0 : 1];
    }

    return ImageId(imageIdAlt);
}

/** rct2: 0x008AD0C0 */
static void PaintMiniatureRailwayTrackFlat(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    bool paintAsGravel = false;
    bool paintGrooved = false;

    if (session.PathElementOnSameHeight != nullptr)
    {
        paintAsGravel = true;
        paintGrooved = true;
    }

    bool isSupported = WoodenASupportsPaintSetup(session, direction & 1, 0, height, session.TrackColours[SCHEME_SUPPORTS]);
    ImageId imageId, imageIdAlt;

    // In the following 3 calls to PaintAddImageAsParentRotated/PaintAddImageAsChildRotated, we add 1 to the
    //  bound_box_offset_z argument to make straight tracks draw above footpaths
    if (isSupported)
    {
        imageId = session.TrackColours[SCHEME_SUPPORTS].WithIndex(miniature_railway_track_floor[direction]);
        PaintAddImageAsParentRotated(session, direction, imageId, { 0, 0, height }, { { 0, 6, height }, { 32, 20, 2 } });

        imageId = session.TrackColours[SCHEME_TRACK].WithIndex(miniature_railway_track_pieces_flat[direction]);
        if (!paintAsGravel)
        {
            PaintAddImageAsChildRotated(session, direction, imageId, { 0, 6, height }, { { 0, 6, height }, { 32, 20, 2 } });
        }
        else
        {
            imageIdAlt = MiniatureRailwayTrackToGravel(imageId);
            PaintAddImageAsChildRotated(session, direction, imageIdAlt, { 0, 6, height }, { { 0, 6, height }, { 32, 20, 2 } });
        }
        if (paintGrooved)
        {
            imageIdAlt = MiniatureRailwayTrackToGrooved(imageId);
            PaintAddImageAsChildRotated(
                session, direction, imageIdAlt, { 0, 6, height }, { { 0, 6, height + 2 }, { 32, 20, 2 } });
            imageIdAlt = MiniatureRailwayTrackToGroovedIndent(
                session.PathElementOnSameHeight, direction, session.CurrentRotation);
            PaintAddImageAsChildRotated(
                session, direction, imageIdAlt.WithTransparency(FilterPaletteID::PaletteDarken2), { 0, 6, height },
                { { 0, 6, height + 2 }, { 32, 20, 2 } });
        }
    }
    else
    {
        imageId = session.TrackColours[SCHEME_TRACK].WithIndex(miniature_railway_track_pieces_flat[direction]);
        if (!paintAsGravel)
        {
            PaintAddImageAsParentRotated(session, direction, imageId, { 0, 6, height }, { { 0, 6, height }, { 32, 20, 2 } });
        }
        else
        {
            imageIdAlt = MiniatureRailwayTrackToGravel(imageId);
            PaintAddImageAsParentRotated(session, direction, imageIdAlt, { 0, 6, height }, { { 0, 6, height }, { 32, 20, 2 } });
        }
        if (paintGrooved)
        {
            imageIdAlt = MiniatureRailwayTrackToGrooved(imageId);
            PaintAddImageAsParentRotated(
                session, direction, imageIdAlt, { 0, 6, height }, { { 0, 6, height + 2 }, { 32, 20, 2 } });
            imageIdAlt = MiniatureRailwayTrackToGroovedIndent(
                session.PathElementOnSameHeight, direction, session.CurrentRotation);
            PaintAddImageAsParentRotated(
                session, direction, imageIdAlt.WithTransparency(FilterPaletteID::PaletteDarken2), { 0, 6, height },
                { { 0, 6, height + 2 }, { 32, 20, 2 } });
        }
    }

    PaintUtilPushTunnelRotated(session, direction, height, TUNNEL_SQUARE_FLAT);

    PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
    PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
}

/** rct2: 0x008AD170, 0x008AD180, 0x008AD190 */
static void PaintMiniatureRailwayStation(
    PaintSession& session, const Ride& ride, [[maybe_unused]] uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    ImageId imageId;

    WoodenASupportsPaintSetup(session, direction & 1, 0, height, session.TrackColours[SCHEME_SUPPORTS]);

    imageId = session.TrackColours[SCHEME_MISC].WithIndex(miniature_railway_station_floor[direction]);
    PaintAddImageAsParentRotated(session, direction, imageId, { 0, 0, height - 2 }, { { 0, 2, height }, { 32, 28, 2 } });

    imageId = session.TrackColours[SCHEME_TRACK].WithIndex(miniature_railway_track_pieces_flat_station[direction]);
    PaintAddImageAsChildRotated(session, direction, imageId, { 0, 6, height }, { { 0, 0, height }, { 32, 20, 2 } });

    PaintUtilPushTunnelRotated(session, direction, height, TUNNEL_SQUARE_FLAT);

    TrackPaintUtilDrawStation3(session, ride, direction, height + 2, height, trackElement);
    // covers shouldn't be offset by +2

    PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
    PaintUtilSetGeneralSupportHeight(session, height + 30, 0x20);
}

/** rct2: 0x008AD0D0 */
static void PaintMiniatureRailwayTrack25DegUp(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    auto imageId = session.TrackColours[SCHEME_TRACK].WithIndex(miniature_railway_track_pieces_25_deg_up[direction]);

    PaintAddImageAsParentRotated(session, direction, imageId, { 0, 2, height }, { { 0, 3, height }, { 32, 25, 2 } });

    switch (direction)
    {
        case 0:
            PaintUtilPushTunnelLeft(session, height - 8, TUNNEL_SQUARE_7);
            break;
        case 1:
            PaintUtilPushTunnelRight(session, height + 8, TUNNEL_SQUARE_8);
            break;
        case 2:
            PaintUtilPushTunnelLeft(session, height + 8, TUNNEL_SQUARE_8);
            break;
        case 3:
            PaintUtilPushTunnelRight(session, height - 8, TUNNEL_SQUARE_7);
            break;
    }

    WoodenASupportsPaintSetup(session, direction & 1, 45 + direction, height, session.TrackColours[SCHEME_SUPPORTS]);

    PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
    PaintUtilSetGeneralSupportHeight(session, height + 56, 0x20);
}

/** rct2: 0x008AD0E0 */
static void PaintMiniatureRailwayTrackFlatTo25DegUp(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    auto imageId = session.TrackColours[SCHEME_TRACK].WithIndex(miniature_railway_track_pieces_flat_to_25_deg_up[direction]);

    PaintAddImageAsParentRotated(session, direction, imageId, { 0, 2, height }, { { 0, 3, height }, { 32, 25, 2 } });

    switch (direction)
    {
        case 0:
            PaintUtilPushTunnelLeft(session, height, TUNNEL_SQUARE_FLAT);
            break;
        case 1:
            PaintUtilPushTunnelRight(session, height, TUNNEL_SQUARE_8);
            break;
        case 2:
            PaintUtilPushTunnelLeft(session, height, TUNNEL_SQUARE_8);
            break;
        case 3:
            PaintUtilPushTunnelRight(session, height, TUNNEL_SQUARE_FLAT);
            break;
    }

    WoodenASupportsPaintSetup(session, direction & 1, 37 + direction, height, session.TrackColours[SCHEME_SUPPORTS]);

    PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
    PaintUtilSetGeneralSupportHeight(session, height + 48, 0x20);
}

/** rct2: 0x008AD0F0 */
static void PaintMiniatureRailwayTrack25DegUpToFlat(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    auto imageId = session.TrackColours[SCHEME_TRACK].WithIndex(miniature_railway_track_pieces_25_deg_up_to_flat[direction]);

    PaintAddImageAsParentRotated(session, direction, imageId, { 0, 2, height }, { { 0, 3, height }, { 32, 25, 2 } });

    switch (direction)
    {
        case 0:
            PaintUtilPushTunnelLeft(session, height - 8, TUNNEL_SQUARE_FLAT);
            break;
        case 1:
            PaintUtilPushTunnelRight(session, height + 8, TUNNEL_14);
            break;
        case 2:
            PaintUtilPushTunnelLeft(session, height + 8, TUNNEL_14);
            break;
        case 3:
            PaintUtilPushTunnelRight(session, height - 8, TUNNEL_SQUARE_FLAT);
            break;
    }

    WoodenASupportsPaintSetup(session, direction & 1, 41 + direction, height, session.TrackColours[SCHEME_SUPPORTS]);

    PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
    PaintUtilSetGeneralSupportHeight(session, height + 40, 0x20);
}

/** rct2: 0x008AD100 */
static void PaintMiniatureRailwayTrack25DegDown(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    PaintMiniatureRailwayTrack25DegUp(session, ride, trackSequence, (direction + 2) % 4, height, trackElement);
}

/** rct2: 0x008AD110 */
static void PaintMiniatureRailwayTrackFlatTo25DegDown(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    PaintMiniatureRailwayTrack25DegUpToFlat(session, ride, trackSequence, (direction + 2) % 4, height, trackElement);
}

/** rct2: 0x008AD120 */
static void PaintMiniatureRailwayTrack25DegDownToFlat(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    PaintMiniatureRailwayTrackFlatTo25DegUp(session, ride, trackSequence, (direction + 2) % 4, height, trackElement);
}

static constexpr CoordsXY miniature_railway_right_quarter_turn_5_tiles_offsets[4][5] = {
    {
        { 0, 2 },
        { 0, 16 },
        { 0, 0 },
        { 16, 0 },
        { 2, 0 },
    },
    {
        { 2, 0 },
        { 16, 0 },
        { 0, 16 },
        { 0, 0 },
        { 0, 2 },
    },
    {
        { 0, 2 },
        { 0, 0 },
        { 16, 16 },
        { 0, 0 },
        { 2, 0 },
    },
    {
        { 2, 0 },
        { 0, 0 },
        { 16, 0 },
        { 0, 16 },
        { 0, 2 },
    },
};

static constexpr CoordsXYZ miniature_railway_right_quarter_turn_5_tiles_bound_offsets[4][5] = {
    {
        { 0, 2, 0 },
        { 0, 16, 0 },
        { 0, 0, 0 },
        { 16, 0, 0 },
        { 2, 0, 0 },
    },
    {
        { 2, 0, 0 },
        { 16, 0, 0 },
        { 0, 16, 0 },
        { 0, 0, 0 },
        { 0, 2, 0 },
    },
    {
        { 0, 2, 0 },
        { 0, 0, 0 },
        { 16, 16, 0 },
        { 0, 0, 0 },
        { 2, 0, 0 },
    },
    {
        { 2, 0, 0 },
        { 0, 0, 0 },
        { 16, 0, 0 },
        { 0, 16, 0 },
        { 0, 2, 0 },
    },
};

static constexpr CoordsXY miniature_railway_right_quarter_turn_5_tiles_bound_lengths[4][5] = {
    {
        { 32, 32 },
        { 32, 16 },
        { 16, 16 },
        { 16, 32 },
        { 32, 32 },
    },
    {
        { 32, 32 },
        { 16, 34 },
        { 16, 16 },
        { 32, 16 },
        { 32, 27 },
    },
    {
        { 32, 27 },
        { 32, 16 },
        { 16, 16 },
        { 16, 32 },
        { 27, 32 },
    },
    {
        { 27, 32 },
        { 16, 32 },
        { 16, 16 },
        { 32, 16 },
        { 32, 32 },
    },
};

static constexpr uint32_t miniature_railway_right_quarter_turn_5_tiles_track_floor[4][5] = {
    {
        SPR_FLOOR_PLANKS,
        SPR_FLOOR_PLANKS_S_SEGMENT,
        SPR_FLOOR_PLANKS_N_SEGMENT,
        SPR_FLOOR_PLANKS_S_SEGMENT,
        SPR_FLOOR_PLANKS_90_DEG,
    },
    {
        SPR_FLOOR_PLANKS_90_DEG,
        SPR_FLOOR_PLANKS_W_SEGMENT,
        SPR_FLOOR_PLANKS_E_SEGMENT,
        SPR_FLOOR_PLANKS_W_SEGMENT,
        SPR_FLOOR_PLANKS,
    },
    {
        SPR_FLOOR_PLANKS,
        SPR_FLOOR_PLANKS_N_SEGMENT,
        SPR_FLOOR_PLANKS_S_SEGMENT,
        SPR_FLOOR_PLANKS_N_SEGMENT,
        SPR_FLOOR_PLANKS_90_DEG,
    },
    {
        SPR_FLOOR_PLANKS_90_DEG,
        SPR_FLOOR_PLANKS_E_SEGMENT,
        SPR_FLOOR_PLANKS_W_SEGMENT,
        SPR_FLOOR_PLANKS_E_SEGMENT,
        SPR_FLOOR_PLANKS,
    },
};

static constexpr int8_t right_quarter_turn_5_supports_type[4][7] = {
    { 0, -1, 4, 2, -1, 4, 1 },
    { 1, -1, 5, 3, -1, 5, 0 },
    { 0, -1, 2, 4, -1, 2, 1 },
    { 1, -1, 3, 5, -1, 3, 0 },
};

static constexpr int8_t miniature_railway_right_quarter_turn_5_tiles_sprite_map[] = {
    0, -1, 1, 2, -1, 3, 4,
};

/** rct2: 0x008AD140 */
static void PaintMiniatureRailwayTrackRightQuarterTurn5Tiles(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    if (right_quarter_turn_5_supports_type[direction][trackSequence] != -1)
    {
        bool isSupported = WoodenASupportsPaintSetup(
            session, right_quarter_turn_5_supports_type[direction][trackSequence], 0, height,
            session.TrackColours[SCHEME_SUPPORTS]);

        if (!isSupported || (trackSequence == 3 && direction == 2))
        {
            TrackPaintUtilRightQuarterTurn5TilesPaint(
                session, 2, height, direction, trackSequence, session.TrackColours[SCHEME_TRACK],
                miniature_railway_track_pieces_flat_quarter_turn_5_tiles, miniature_railway_right_quarter_turn_5_tiles_offsets,
                miniature_railway_right_quarter_turn_5_tiles_bound_lengths, nullptr);
        }
        else
        {
            TrackPaintUtilRightQuarterTurn5TilesPaint(
                session, 2, height, direction, trackSequence, session.TrackColours[SCHEME_SUPPORTS],
                miniature_railway_right_quarter_turn_5_tiles_track_floor, nullptr,
                miniature_railway_right_quarter_turn_5_tiles_bound_lengths,
                miniature_railway_right_quarter_turn_5_tiles_bound_offsets);

            int32_t index = miniature_railway_right_quarter_turn_5_tiles_sprite_map[trackSequence];
            auto imageId = session.TrackColours[SCHEME_TRACK].WithIndex(
                miniature_railway_track_pieces_flat_quarter_turn_5_tiles[direction][index]);
            CoordsXY offset = miniature_railway_right_quarter_turn_5_tiles_offsets[direction][index];
            CoordsXY boundsLength = miniature_railway_right_quarter_turn_5_tiles_bound_lengths[direction][index];
            CoordsXYZ boundsOffset = CoordsXYZ(offset, 0);

            PaintAddImageAsChild(
                session, imageId, { offset, height }, { boundsOffset + CoordsXYZ{ 0, 0, height }, { boundsLength, 2 } });
        }
    }
    if (direction == 0 && trackSequence == 0)
    {
        PaintUtilPushTunnelLeft(session, height, TUNNEL_SQUARE_FLAT);
    }

    if (direction == 0 && trackSequence == 6)
    {
        PaintUtilPushTunnelRight(session, height, TUNNEL_SQUARE_FLAT);
    }

    if (direction == 1 && trackSequence == 6)
    {
        PaintUtilPushTunnelLeft(session, height, TUNNEL_SQUARE_FLAT);
    }

    if (direction == 3 && trackSequence == 0)
    {
        PaintUtilPushTunnelRight(session, height, TUNNEL_SQUARE_FLAT);
    }

    int32_t blockedSegments = 0;
    switch (trackSequence)
    {
        case 0:
            blockedSegments = SEGMENTS_ALL;
            break;
        case 1:
            blockedSegments = SEGMENT_B4 | SEGMENT_C8 | SEGMENT_CC;
            break;
        case 2:
            blockedSegments = SEGMENT_D0 | SEGMENT_C4 | SEGMENT_D4 | SEGMENT_BC | SEGMENT_C0 | SEGMENT_CC;
            break;
        case 3:
            blockedSegments = SEGMENT_B4 | SEGMENT_B8 | SEGMENT_BC | SEGMENT_C4 | SEGMENT_C8 | SEGMENT_CC | SEGMENT_D0
                | SEGMENT_D4;
            break;
        case 4:
            blockedSegments = SEGMENT_B4 | SEGMENT_C8 | SEGMENT_CC;
            break;
        case 5:
            blockedSegments = SEGMENT_D4 | SEGMENT_C4 | SEGMENT_D0 | SEGMENT_B8 | SEGMENT_C0 | SEGMENT_C8;
            break;
        case 6:
            blockedSegments = SEGMENTS_ALL;
            break;
    }

    PaintUtilSetSegmentSupportHeight(session, PaintUtilRotateSegments(blockedSegments, direction), 0xFFFF, 0);

    PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
}

/** rct2: 0x008AD130 */
static void PaintMiniatureRailwayTrackLeftQuarterTurn5Tiles(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    trackSequence = mapLeftQuarterTurn5TilesToRightQuarterTurn5Tiles[trackSequence];
    PaintMiniatureRailwayTrackRightQuarterTurn5Tiles(session, ride, trackSequence, (direction + 1) % 4, height, trackElement);
}

static constexpr int8_t s_bend_left_supports_type[4][4] = {
    { 0, 5, 3, 0 },
    { 1, 2, 4, 1 },
    { 0, 5, 3, 0 },
    { 1, 2, 4, 1 },
};

static constexpr uint32_t miniature_railway_s_bend_left_tiles_track_floor[2][4] = {
    {
        SPR_FLOOR_PLANKS,
        SPR_FLOOR_PLANKS_W_SEGMENT,
        SPR_FLOOR_PLANKS_E_SEGMENT,
        SPR_FLOOR_PLANKS,
    },
    {
        SPR_FLOOR_PLANKS_90_DEG,
        SPR_FLOOR_PLANKS_N_SEGMENT,
        SPR_FLOOR_PLANKS_S_SEGMENT,
        SPR_FLOOR_PLANKS_90_DEG,
    },
};

/** rct2: 0x8AD150 */
static void PaintMiniatureRailwayTrackSBendLeft(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    if (direction == 2 || direction == 3)
    {
        trackSequence = 3 - trackSequence;
    }

    bool isSupported = WoodenASupportsPaintSetup(
        session, s_bend_left_supports_type[direction][trackSequence], 0, height, session.TrackColours[SCHEME_SUPPORTS]);

    static constexpr CoordsXY offsetList[] = {
        { 0, 2 },
        { 0, 0 },
        { 0, 6 },
        { 0, 2 },
    };

    static constexpr CoordsXY boundsList[] = {
        { 32, 27 },
        { 32, 26 },
        { 32, 26 },
        { 32, 27 },
    };

    auto imageId = session.TrackColours[SCHEME_TRACK].WithIndex(
        miniature_railway_track_pieces_s_bend_left[direction & 1][trackSequence]);
    CoordsXY offset = offsetList[trackSequence];
    CoordsXY bounds = boundsList[trackSequence];

    if (!isSupported)
    {
        PaintAddImageAsParentRotated(
            session, direction, imageId, { offset.x, offset.y, height },
            { { offset.x, offset.y, height }, { bounds.x, bounds.y, 2 } });
    }
    else
    {
        imageId = session.TrackColours[SCHEME_SUPPORTS].WithIndex(
            miniature_railway_s_bend_left_tiles_track_floor[direction & 1][trackSequence]);
        PaintAddImageAsParentRotated(
            session, direction, imageId, { 0, 0, height }, { { offset.x, offset.y, height }, { bounds.x, bounds.y, 2 } });

        imageId = session.TrackColours[SCHEME_TRACK].WithIndex(
            miniature_railway_track_pieces_s_bend_left[direction & 1][trackSequence]);
        PaintAddImageAsChildRotated(
            session, direction, imageId, { offset.x, offset.y, height },
            { { offset.x, offset.y, height }, { bounds.x, bounds.y, 2 } });
    }
    if (direction == 0 || direction == 2)
    {
        if (trackSequence == 0)
        {
            PaintUtilPushTunnelLeft(session, height, TUNNEL_SQUARE_FLAT);
        }
    }
    else
    {
        if (trackSequence == 3)
        {
            PaintUtilPushTunnelRight(session, height, TUNNEL_SQUARE_FLAT);
        }
    }

    int32_t blockedSegments = 0;
    switch (trackSequence)
    {
        case 0:
            blockedSegments = SEGMENTS_ALL;
            break;
        case 1:
            blockedSegments = SEGMENT_D0 | SEGMENT_C4 | SEGMENT_CC | SEGMENT_B8 | SEGMENT_C8 | SEGMENT_B4;
            break;
        case 2:
            blockedSegments = SEGMENT_D0 | SEGMENT_C4 | SEGMENT_CC | SEGMENT_C0 | SEGMENT_D4 | SEGMENT_BC;
            break;
        case 3:
            blockedSegments = SEGMENTS_ALL;
            break;
    }
    PaintUtilSetSegmentSupportHeight(session, PaintUtilRotateSegments(blockedSegments, direction & 1), 0xFFFF, 0);

    PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
}

static constexpr int8_t s_bend_right_supports_type[4][4] = {
    { 0, 4, 2, 0 },
    { 1, 5, 3, 1 },
    { 0, 4, 2, 0 },
    { 1, 5, 3, 1 },
};

static constexpr uint32_t miniature_railway_s_bend_right_tiles_track_floor[2][4] = {
    {
        SPR_FLOOR_PLANKS,
        SPR_FLOOR_PLANKS_S_SEGMENT,
        SPR_FLOOR_PLANKS_N_SEGMENT,
        SPR_FLOOR_PLANKS,
    },
    {
        SPR_FLOOR_PLANKS_90_DEG,
        SPR_FLOOR_PLANKS_W_SEGMENT,
        SPR_FLOOR_PLANKS_E_SEGMENT,
        SPR_FLOOR_PLANKS_90_DEG,
    },
};

/** rct2: 0x008AD160 */
static void PaintMiniatureRailwayTrackSBendRight(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    if (direction == 2 || direction == 3)
    {
        trackSequence = 3 - trackSequence;
    }

    bool isSupported = WoodenASupportsPaintSetup(
        session, s_bend_right_supports_type[direction][trackSequence], 0, height, session.TrackColours[SCHEME_SUPPORTS]);

    static constexpr CoordsXY offsetList[] = {
        { 0, 2 },
        { 0, 6 },
        { 0, 0 },
        { 0, 2 },
    };

    static constexpr CoordsXY boundsList[] = {
        { 32, 27 },
        { 32, 26 },
        { 32, 26 },
        { 32, 27 },
    };

    auto imageId = session.TrackColours[SCHEME_TRACK].WithIndex(
        miniature_railway_track_pieces_s_bend_right[direction & 1][trackSequence]);
    CoordsXY offset = offsetList[trackSequence];
    CoordsXY bounds = boundsList[trackSequence];
    if (!isSupported)
    {
        PaintAddImageAsParentRotated(
            session, direction, imageId, { offset.x, offset.y, height },
            { { offset.x, offset.y, height }, { bounds.x, bounds.y, 2 } });
    }
    else
    {
        imageId = session.TrackColours[SCHEME_SUPPORTS].WithIndex(
            miniature_railway_s_bend_right_tiles_track_floor[direction & 1][trackSequence]);
        PaintAddImageAsParentRotated(
            session, direction, imageId, { 0, 0, height }, { { offset.x, offset.y, height }, { bounds.x, bounds.y, 2 } });

        imageId = session.TrackColours[SCHEME_TRACK].WithIndex(
            miniature_railway_track_pieces_s_bend_right[direction & 1][trackSequence]);
        PaintAddImageAsChildRotated(
            session, direction, imageId, { offset.x, offset.y, height },
            { { offset.x, offset.y, height }, { bounds.x, bounds.y, 2 } });
    }

    if (direction == 0 || direction == 2)
    {
        if (trackSequence == 0)
        {
            PaintUtilPushTunnelLeft(session, height, TUNNEL_SQUARE_FLAT);
        }
    }
    else
    {
        if (trackSequence == 3)
        {
            PaintUtilPushTunnelRight(session, height, TUNNEL_SQUARE_FLAT);
        }
    }

    int32_t blockedSegments = 0;
    switch (trackSequence)
    {
        case 0:
            blockedSegments = SEGMENTS_ALL;
            break;
        case 1:
            blockedSegments = SEGMENT_D0 | SEGMENT_C4 | SEGMENT_CC | SEGMENT_C0 | SEGMENT_D4 | SEGMENT_BC;
            break;
        case 2:
            blockedSegments = SEGMENT_D0 | SEGMENT_C4 | SEGMENT_CC | SEGMENT_B8 | SEGMENT_C8 | SEGMENT_B4;
            break;
        case 3:
            blockedSegments = SEGMENTS_ALL;
            break;
    }
    PaintUtilSetSegmentSupportHeight(session, PaintUtilRotateSegments(blockedSegments, direction & 1), 0xFFFF, 0);

    PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
}

static constexpr uint32_t miniature_railway_right_quarter_turn_3_tile_track_floor[4][3] = {
    {
        SPR_FLOOR_PLANKS_S_SEGMENT,
        0,
        SPR_FLOOR_PLANKS_S_SEGMENT,
    },
    {
        SPR_FLOOR_PLANKS_W_SEGMENT,
        0,
        SPR_FLOOR_PLANKS_W_SEGMENT,
    },
    {
        SPR_FLOOR_PLANKS_N_SEGMENT,
        0,
        SPR_FLOOR_PLANKS_N_SEGMENT,
    },
    {
        SPR_FLOOR_PLANKS_E_SEGMENT,
        0,
        SPR_FLOOR_PLANKS_E_SEGMENT,
    },
};

static constexpr CoordsXYZ miniature_railway_right_quarter_turn_3_tile_bound_offsets[4][3] = {
    {
        { 0, 6, 0 },
        { 16, 16, 0 },
        { 6, 0, 0 },
    },
    {
        { 6, 0, 0 },
        { 16, 0, 0 },
        { 0, 6, 0 },
    },
    {
        { 0, 6, 0 },
        { 0, 0, 0 },
        { 6, 0, 0 },
    },
    {
        { 6, 0, 0 },
        { 0, 16, 0 },
        { 0, 6, 0 },
    },
};

/** rct2: 0x008AD1B0 */
static void PaintMiniatureRailwayTrackRightQuarterTurn3Tiles(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    bool isSupported = false;
    if (trackSequence != 1 && trackSequence != 2)
    {
        static constexpr uint8_t supportType[] = { 4, 5, 2, 3 };
        isSupported = WoodenASupportsPaintSetup(
            session, supportType[direction], 0, height, session.TrackColours[SCHEME_SUPPORTS]);
    }
    if (!isSupported)
    {
        TrackPaintUtilRightQuarterTurn3TilesPaint(
            session, 3, height, direction, trackSequence, session.TrackColours[SCHEME_TRACK],
            miniature_railway_track_pieces_flat_quarter_turn_3_tiles, defaultRightQuarterTurn3TilesOffsets,
            defaultRightQuarterTurn3TilesBoundLengths, nullptr);

        // The following piece was missing in vanilla RCT2
        if (trackSequence == 1 && direction == 0)
        {
            auto imageId = session.TrackColours[SCHEME_TRACK].WithIndex(
                SPR_G2_MINIATURE_RAILWAY_QUARTER_TURN_3_TILES_SW_SE_PART_3);
            PaintAddImageAsParent(session, imageId, { 0, 0, height }, { { 0, 0, height }, { 8, 8, 2 } });
        }
    }
    else
    {
        TrackPaintUtilRightQuarterTurn3TilesPaint(
            session, 3, height, direction, trackSequence, session.TrackColours[SCHEME_SUPPORTS],
            miniature_railway_right_quarter_turn_3_tile_track_floor, nullptr, defaultRightQuarterTurn3TilesBoundLengths,
            miniature_railway_right_quarter_turn_3_tile_bound_offsets);

        static constexpr int8_t _right_quarter_turn_3_tiles_sprite_map[] = { 0, -1, 1, 2 };

        int32_t index = _right_quarter_turn_3_tiles_sprite_map[trackSequence];

        auto imageId = session.TrackColours[SCHEME_TRACK].WithIndex(
            miniature_railway_track_pieces_flat_quarter_turn_3_tiles[direction][index]);
        CoordsXY offset = defaultRightQuarterTurn3TilesOffsets[direction][index];
        CoordsXY boundsLength = defaultRightQuarterTurn3TilesBoundLengths[direction][index];
        CoordsXYZ boundsOffset(offset, 0);

        PaintAddImageAsChild(
            session, imageId, { offset, height }, { boundsOffset + CoordsXYZ{ 0, 0, height }, { boundsLength, 3 } });
    }
    TrackPaintUtilRightQuarterTurn3TilesTunnel(session, height, direction, trackSequence, TUNNEL_SQUARE_FLAT);

    int32_t blockedSegments = 0;
    switch (trackSequence)
    {
        case 0:
            blockedSegments = SEGMENTS_ALL;
            break;
        case 2:
            blockedSegments = SEGMENT_D0 | SEGMENT_C4 | SEGMENT_D4 | SEGMENT_C0;
            break;
        case 3:
            blockedSegments = SEGMENTS_ALL;
            break;
    }
    PaintUtilSetSegmentSupportHeight(session, PaintUtilRotateSegments(blockedSegments, direction), 0xFFFF, 0);

    PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
}

/** rct2: 0x008AD1A0 */
static void PaintMiniatureRailwayTrackLeftQuarterTurn3Tiles(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    trackSequence = mapLeftQuarterTurn3TilesToRightQuarterTurn3Tiles[trackSequence];
    PaintMiniatureRailwayTrackRightQuarterTurn3Tiles(session, ride, trackSequence, (direction + 1) % 4, height, trackElement);
}

static constexpr int8_t paint_miniature_railway_eighth_to_diag_index[] = {
    0, 1, 2, -1, 3,
};

static constexpr uint32_t miniature_railway_floor_track_pieces_left_eight_to_diag[4][5] = {
    {
        SPR_FLOOR_PLANKS,
        SPR_FLOOR_PLANKS,
        SPR_FLOOR_PLANKS_E_SEGMENT,
        SPR_FLOOR_PLANKS_W_SEGMENT,
        SPR_FLOOR_PLANKS_90_DEG,
    },
    {
        SPR_FLOOR_PLANKS_90_DEG,
        SPR_FLOOR_PLANKS_90_DEG,
        SPR_FLOOR_PLANKS_S_SEGMENT,
        SPR_FLOOR_PLANKS_N_SEGMENT,
        SPR_FLOOR_PLANKS,
    },
    {
        SPR_FLOOR_PLANKS,
        SPR_FLOOR_PLANKS,
        SPR_FLOOR_PLANKS_W_SEGMENT,
        SPR_FLOOR_PLANKS_E_SEGMENT,
        SPR_FLOOR_PLANKS_90_DEG,
    },
    {
        SPR_FLOOR_PLANKS_90_DEG,
        SPR_FLOOR_PLANKS_90_DEG,
        SPR_FLOOR_PLANKS_N_SEGMENT,
        SPR_FLOOR_PLANKS_S_SEGMENT,
        SPR_FLOOR_PLANKS,
    },
};

static constexpr CoordsXYZ miniature_railway_track_floor_pieces_left_eight_to_diag_bounds[4][5] = {
    {
        { 32, 32, 2 },
        { 32, 16, 2 },
        { 16, 16, 2 },
        { 16, 16, 0 },
        { 30, 30, 0 },
    },
    {
        { 32, 32, 2 },
        { 16, 34, 2 },
        { 14, 14, 2 },
        { 16, 16, 0 },
        { 34, 32, 0 },
    },
    {
        { 32, 32, 2 },
        { 32, 16, 2 },
        { 16, 16, 2 },
        { 16, 16, 0 },
        { 16, 16, 2 },
    },
    {
        { 32, 32, 2 },
        { 16, 32, 2 },
        { 16, 16, 2 },
        { 16, 16, 0 },
        { 32, 32, 0 },
    },
};

static constexpr CoordsXY miniature_railway_track_floor_pieces_left_eight_to_diag_offset[4][5] = {
    {
        { 0, 0 },
        { 0, 0 },
        { 0, 16 },
        { 16, 0 },
        { 0, 0 },
    },
    {
        { 0, 0 },
        { 0, 0 },
        { 16, 16 },
        { 0, 0 },
        { 0, 0 },
    },
    {
        { 0, 0 },
        { 0, 16 },
        { 16, 0 },
        { 0, 16 },
        { 0, 0 },
    },
    {
        { 0, 0 },
        { 16, 0 },
        { 0, 0 },
        { 16, 16 },
        { 0, 0 },
    },
};

/** rct2: 0x008AD1C0 */
static void PaintMiniatureRailwayTrackLeftEighthToDiag(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    static constexpr uint8_t supportType[4][5] = {
        { 0, 0, 3, 5, 1 },
        { 1, 1, 4, 2, 0 },
        { 0, 0, 5, 3, 1 },
        { 1, 1, 2, 4, 0 },
    };

    bool isSupported = false;
    bool isRightEighthToOrthog = trackElement.GetTrackType() == TrackElemType::RightEighthToOrthogonal;
    // Right eighth to orthogonal calls this function but we do not want to have a support call for it
    // for track sequence 4
    if (trackSequence != 4 || !isRightEighthToOrthog)
    {
        isSupported = WoodenASupportsPaintSetup(
            session, supportType[direction][trackSequence], 0, height, session.TrackColours[SCHEME_SUPPORTS]);
    }
    ImageId imageId;
    if (!isSupported)
    {
        int8_t index = paint_miniature_railway_eighth_to_diag_index[trackSequence];
        if (index >= 0)
        {
            imageId = session.TrackColours[SCHEME_TRACK].WithIndex(
                miniature_railway_track_pieces_left_eight_to_diag[direction][index]);
            CoordsXY offset = miniature_railway_track_pieces_left_eight_to_diag_offset[direction][index];
            CoordsXYZ bounds = miniature_railway_track_pieces_left_eight_to_diag_bounds[direction][index];
            if (isRightEighthToOrthog)
            {
                bounds = miniature_railway_track_pieces_right_eight_to_orthog_bounds[direction][index];
                offset = miniature_railway_track_pieces_right_eight_to_orthog_offset[direction][index];
            }
            PaintAddImageAsParent(session, imageId, { 0, 0, height }, { { offset, height }, bounds });
        }
    }
    else
    {
        imageId = session.TrackColours[SCHEME_SUPPORTS].WithIndex(
            miniature_railway_floor_track_pieces_left_eight_to_diag[direction][trackSequence]);
        CoordsXY offset = miniature_railway_track_floor_pieces_left_eight_to_diag_offset[direction][trackSequence];
        CoordsXYZ bounds = miniature_railway_track_floor_pieces_left_eight_to_diag_bounds[direction][trackSequence];
        PaintAddImageAsParent(session, imageId, { 0, 0, height }, { { offset, height }, bounds });

        int8_t index = paint_miniature_railway_eighth_to_diag_index[trackSequence];
        if (index >= 0)
        {
            imageId = session.TrackColours[SCHEME_TRACK].WithIndex(
                miniature_railway_track_pieces_left_eight_to_diag[direction][index]);
            offset = miniature_railway_track_pieces_left_eight_to_diag_offset[direction][index];
            bounds = miniature_railway_track_pieces_left_eight_to_diag_bounds[direction][index];
            PaintAddImageAsChild(session, imageId, { 0, 0, height }, { { offset, height }, bounds });
        }
    }

    if (direction == 0 && trackSequence == 0)
    {
        PaintUtilPushTunnelLeft(session, height, TUNNEL_SQUARE_FLAT);
    }
    if (direction == 3 && trackSequence == 0)
    {
        PaintUtilPushTunnelRight(session, height, TUNNEL_SQUARE_FLAT);
    }

    PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
    PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
}

static constexpr uint32_t miniature_railway_floor_track_pieces_right_eight_to_diag[4][5] = {
    {
        SPR_FLOOR_PLANKS_90_DEG,
        SPR_FLOOR_PLANKS_90_DEG,
        SPR_FLOOR_PLANKS_N_SEGMENT,
        SPR_FLOOR_PLANKS_S_SEGMENT,
        SPR_FLOOR_PLANKS_90_DEG,
    },
    {
        SPR_FLOOR_PLANKS_90_DEG,
        SPR_FLOOR_PLANKS_90_DEG,
        SPR_FLOOR_PLANKS_E_SEGMENT,
        SPR_FLOOR_PLANKS_W_SEGMENT,
        SPR_FLOOR_PLANKS,
    },
    {
        SPR_FLOOR_PLANKS,
        SPR_FLOOR_PLANKS,
        SPR_FLOOR_PLANKS_S_SEGMENT,
        SPR_FLOOR_PLANKS_N_SEGMENT,
        SPR_FLOOR_PLANKS_90_DEG,
    },
    {
        SPR_FLOOR_PLANKS_90_DEG,
        SPR_FLOOR_PLANKS_90_DEG,
        SPR_FLOOR_PLANKS_W_SEGMENT,
        SPR_FLOOR_PLANKS_E_SEGMENT,
        SPR_FLOOR_PLANKS,
    },
};

static constexpr CoordsXYZ miniature_railway_track_floor_pieces_right_eight_to_diag_bounds[4][5] = {
    {
        { 32, 32, 2 },
        { 32, 16, 2 },
        { 16, 16, 2 },
        { 16, 16, 0 },
        { 32, 32, 0 },
    },
    {
        { 32, 32, 2 },
        { 16, 32, 2 },
        { 16, 16, 2 },
        { 16, 16, 0 },
        { 16, 16, 2 },
    },
    {
        { 32, 32, 2 },
        { 34, 16, 2 },
        { 28, 28, 2 },
        { 16, 16, 0 },
        { 32, 34, 0 },
    },
    {
        { 32, 32, 2 },
        { 16, 32, 2 },
        { 16, 16, 2 },
        { 16, 16, 0 },
        { 30, 30, 0 },
    },
};

static constexpr CoordsXY miniature_railway_track_floor_pieces_right_eight_to_diag_offset[4][5] = {
    {
        { 0, 0 },
        { 0, 16 },
        { 0, 0 },
        { 16, 16 },
        { 0, 0 },
    },
    {
        { 0, 0 },
        { 16, 0 },
        { 0, 16 },
        { 16, 0 },
        { 0, 0 },
    },
    {
        { 0, 0 },
        { 0, 0 },
        { 4, 4 },
        { 0, 0 },
        { 0, 0 },
    },
    {
        { 0, 0 },
        { 0, 0 },
        { 16, 0 },
        { 0, 16 },
        { 0, 0 },
    },
};

/** rct2: 0x008AD1D0 */
static void PaintMiniatureRailwayTrackRightEighthToDiag(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    static constexpr uint8_t supportType[4][5] = {
        { 0, 0, 2, 4, 1 },
        { 1, 1, 3, 5, 0 },
        { 0, 0, 4, 2, 1 },
        { 1, 1, 5, 3, 0 },
    };

    bool isSupported = false;
    bool isLeftEighthToOrthog = trackElement.GetTrackType() == TrackElemType::LeftEighthToOrthogonal;
    // Right eighth to orthogonal calls this function but we do not want to have a support call for it
    // for track sequence 4
    if (trackSequence != 4 || !isLeftEighthToOrthog)
    {
        isSupported = WoodenASupportsPaintSetup(
            session, supportType[direction][trackSequence], 0, height, session.TrackColours[SCHEME_SUPPORTS]);
    }

    ImageId imageId;
    if (!isSupported)
    {
        int8_t index = paint_miniature_railway_eighth_to_diag_index[trackSequence];
        if (index >= 0)
        {
            imageId = session.TrackColours[SCHEME_TRACK].WithIndex(
                miniature_railway_track_pieces_right_eight_to_diag[direction][index]);
            CoordsXY offset = miniature_railway_track_pieces_right_eight_to_diag_offset[direction][index];
            CoordsXYZ bounds = miniature_railway_track_pieces_right_eight_to_diag_bounds[direction][index];
            if (isLeftEighthToOrthog)
            {
                bounds = miniature_railway_track_pieces_left_eight_to_orthog_bounds[direction][index];
                offset = miniature_railway_track_pieces_left_eight_to_orthog_offset[direction][index];
            }
            PaintAddImageAsParent(session, imageId, { 0, 0, height }, { { offset, height }, bounds });
        }
    }
    else
    {
        imageId = session.TrackColours[SCHEME_SUPPORTS].WithIndex(
            miniature_railway_floor_track_pieces_right_eight_to_diag[direction][trackSequence]);
        CoordsXY offset = miniature_railway_track_floor_pieces_right_eight_to_diag_offset[direction][trackSequence];
        CoordsXYZ bounds = miniature_railway_track_floor_pieces_right_eight_to_diag_bounds[direction][trackSequence];
        PaintAddImageAsParent(session, imageId, { 0, 0, height }, { { offset, height }, bounds });

        int8_t index = paint_miniature_railway_eighth_to_diag_index[trackSequence];
        if (index >= 0)
        {
            imageId = session.TrackColours[SCHEME_TRACK].WithIndex(
                miniature_railway_track_pieces_right_eight_to_diag[direction][index]);
            offset = miniature_railway_track_pieces_right_eight_to_diag_offset[direction][index];
            bounds = miniature_railway_track_pieces_right_eight_to_diag_bounds[direction][index];
            PaintAddImageAsChild(session, imageId, { 0, 0, height }, { { offset, height }, bounds });
        }
    }

    if (direction == 0 && trackSequence == 0)
    {
        PaintUtilPushTunnelLeft(session, height, TUNNEL_SQUARE_FLAT);
    }
    if (direction == 3 && trackSequence == 0)
    {
        PaintUtilPushTunnelRight(session, height, TUNNEL_SQUARE_FLAT);
    }

    PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
    PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
}

/** rct2: 0x008AD1E0 */
static void PaintMiniatureRailwayTrackLeftEighthToOrthogonal(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    trackSequence = mapLeftEighthTurnToOrthogonal[trackSequence];
    PaintMiniatureRailwayTrackRightEighthToDiag(session, ride, trackSequence, (direction + 2) % 4, height, trackElement);
}

/** rct2: 0x008AD1F0 */
static void PaintMiniatureRailwayTrackRightEighthToOrthogonal(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    trackSequence = mapLeftEighthTurnToOrthogonal[trackSequence];
    PaintMiniatureRailwayTrackLeftEighthToDiag(session, ride, trackSequence, (direction + 3) % 4, height, trackElement);
}

static constexpr bool miniature_railway_diag_image_segment[][4] = {
    { false, true, false, false },
    { false, false, false, true },
    { false, false, true, false },
    { true, false, false, false },
};

static constexpr std::optional<WoodenSupportSubType> _diagSupportTypes[NumOrthogonalDirections][4] = {
    { std::nullopt, WoodenSupportSubType::Corner0, WoodenSupportSubType::Corner2, WoodenSupportSubType::NeSw },
    { std::nullopt, WoodenSupportSubType::Corner1, WoodenSupportSubType::Corner3, WoodenSupportSubType::NwSe },
    { std::nullopt, WoodenSupportSubType::Corner2, WoodenSupportSubType::Corner0, WoodenSupportSubType::NeSw },
    { std::nullopt, WoodenSupportSubType::Corner3, WoodenSupportSubType::Corner1, WoodenSupportSubType::NwSe },
};

struct FloorDescriptor
{
    uint32_t image_id;
    CoordsXY bound_size;
    CoordsXY bound_offset;
};

static constexpr FloorDescriptor floors[] = {
    { SPR_FLOOR_PLANKS, { 32, 32 }, { 0, 0 } },
    { SPR_FLOOR_PLANKS_90_DEG, { 32, 32 }, { 0, 0 } }, // -16,-16(1:3)
    { SPR_FLOOR_PLANKS_N_SEGMENT, { 32, 32 }, { -16, -16 } },
    { SPR_FLOOR_PLANKS_E_SEGMENT, { 16, 16 }, { 0, 16 } },
    { SPR_FLOOR_PLANKS_S_SEGMENT, { 16, 16 }, { 16, 16 } },
    { SPR_FLOOR_PLANKS_W_SEGMENT, { 16, 16 }, { 16, 0 } },
};

/** rct2: 0x008AD200 */
static void MiniatureRailwayTrackDiagFlat(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    bool isSupported = false;
    auto supportType = _diagSupportTypes[direction][trackSequence];

    uint32_t floorImage = 0;
    CoordsXY floorBoundSize;
    CoordsXY floorBoundOffset;

    if (supportType.has_value())
    {
        const auto enumValue = EnumValue(supportType.value());
        floorImage = floors[enumValue].image_id;
        floorBoundSize = floors[enumValue].bound_size;
        floorBoundOffset = floors[enumValue].bound_offset;
        isSupported = WoodenASupportsPaintSetup(
            session, WoodenSupportType::Truss, supportType.value(), height, session.TrackColours[SCHEME_SUPPORTS]);
    }

    if (direction == 1 && trackSequence == 3)
    {
        floorBoundOffset = { -16, -16 };
    }

    uint32_t imageId = miniature_railway_track_pieces_diag_flat[direction];
    bool drawRail = miniature_railway_diag_image_segment[direction][trackSequence];

    if (isSupported)
    {
        PaintAddImageAsParent(
            session, session.TrackColours[SCHEME_SUPPORTS].WithIndex(floorImage), { 0, 0, height },
            { { floorBoundOffset, height }, { floorBoundSize, (drawRail ? 2 : 0) } });
        if (drawRail)
        {
            PaintAddImageAsChild(
                session, session.TrackColours[SCHEME_TRACK].WithIndex(imageId), { -16, -16, height },
                { { -16, -16, height }, { 32, 32, 2 } });
        }
    }
    else if (drawRail)
    {
        PaintAddImageAsParent(
            session, session.TrackColours[SCHEME_TRACK].WithIndex(imageId), { -16, -16, height },
            { { -16, -16, height }, { 32, 32, 2 } });
    }

    PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
    PaintUtilSetGeneralSupportHeight(session, height + 32, 0x20);
}

enum
{
    WOOD_A = 0,
    WOOD_B = 1,
};

static bool WoodenSupportsPaintSetup(
    PaintSession& session, int32_t woodType, int32_t supportType, int32_t special, int32_t height, ImageId imageColourFlags)
{
    switch (woodType)
    {
        case WOOD_A:
            return WoodenASupportsPaintSetup(session, supportType, special, height, imageColourFlags);

        case WOOD_B:
            return WoodenBSupportsPaintSetup(session, supportType, special, height, imageColourFlags);
    }

    return false;
}

/** rct2: 0x008AD230 */
static void MiniatureRailwayTrackDiag25DegUp(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    static constexpr int8_t heightDiffs[] = { +8, -8, +8, -8 };
    int8_t heightDiff = heightDiffs[direction];
    if (trackSequence == 3)
    {
        heightDiff = 8;
    }

    bool hasSupports = false;

    uint32_t floorImage = 0;
    CoordsXY floorBoundSize;
    CoordsXY floorBoundOffset;

    auto supportType = _diagSupportTypes[direction][trackSequence];
    if (supportType.has_value())
    {
        // TODO: The supports are inconsistent for different rotations
        int8_t supportFunction = WOOD_B;
        if (supportType == WoodenSupportSubType::NeSw || supportType == WoodenSupportSubType::NwSe)
        {
            supportFunction = WOOD_A;
        }

        const auto enumValue = EnumValue(supportType.value());
        floorImage = floors[enumValue].image_id;
        floorBoundSize = floors[enumValue].bound_size;
        floorBoundOffset = floors[enumValue].bound_offset;
        hasSupports = WoodenSupportsPaintSetup(
            session, supportFunction, enumValue, 0, height + heightDiff, session.TrackColours[SCHEME_SUPPORTS]);
    }

    if (direction == 1 && trackSequence == 3)
    {
        floorBoundOffset = { -16, -16 };
    }

    static constexpr int8_t offsetsB[4][4][2] = {
        { { 0, 0 }, { +8, +16 }, { +8, +8 }, { +8, +8 } },
        { { 0, 0 }, { -8, -8 }, { -8, -8 }, { +8, +8 } },
        { { 0, 0 }, { +8, +8 }, { +8, +16 }, { +8, +8 } },
        {
            { 0, 0 },
            { -8, -8 },
            { -8, -8 },
            { +8, +8 },
        },
    };

    uint32_t imageId = miniature_railway_track_pieces_diag_25_deg_up[direction];
    bool drawRail = miniature_railway_diag_image_segment[direction][trackSequence];
    static constexpr int8_t offsetB[] = { +8, 0, +8, +8 };
    if (hasSupports)
    {
        PaintAddImageAsParent(
            session, session.TrackColours[SCHEME_SUPPORTS].WithIndex(floorImage),
            { 0, 0, height + offsetsB[direction][trackSequence][0] },
            { { floorBoundOffset, height + offsetsB[direction][trackSequence][1] }, { floorBoundSize, (drawRail ? 2 : 0) } });
        if (drawRail)
        {
            PaintAddImageAsChild(
                session, session.TrackColours[SCHEME_TRACK].WithIndex(imageId), { -16, -16, height },
                { { -16, -16, height + offsetB[direction] }, { 32, 32, 2 } });
        }
    }
    else if (drawRail)
    {
        PaintAddImageAsParent(
            session, session.TrackColours[SCHEME_TRACK].WithIndex(imageId), { -16, -16, height },
            { { -16, -16, height + offsetB[direction] }, { 32, 32, 2 } });
    }

    PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
    PaintUtilSetGeneralSupportHeight(session, height + 56, 0x20);
}

/** rct2: 0x008AD210 */
static void MiniatureRailwayTrackDiagFlatTo25DegUp(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    bool hasSupports = false;

    uint32_t floorImage = 0;
    CoordsXY floorBoundSize;
    CoordsXY floorBoundOffset;

    auto supportType = _diagSupportTypes[direction][trackSequence];
    if (supportType.has_value())
    {
        const auto enumValue = EnumValue(supportType.value());
        floorImage = floors[enumValue].image_id;
        floorBoundSize = floors[enumValue].bound_size;
        floorBoundOffset = floors[enumValue].bound_offset;
        hasSupports = WoodenASupportsPaintSetup(
            session, WoodenSupportType::Truss, supportType.value(), height, session.TrackColours[SCHEME_SUPPORTS]);
    }

    if (direction == 1 && trackSequence == 3)
    {
        floorBoundOffset = CoordsXY(-16, -16);
    }

    uint32_t imageId = miniature_railway_track_pieces_diag_flat_to_25_deg_up[direction];
    bool drawRail = miniature_railway_diag_image_segment[direction][trackSequence];

    if (hasSupports)
    {
        PaintAddImageAsParent(
            session, session.TrackColours[SCHEME_SUPPORTS].WithIndex(floorImage), { 0, 0, height },
            { { floorBoundOffset, height }, { floorBoundSize, (drawRail ? 2 : 0) } });
        if (drawRail)
        {
            PaintAddImageAsChild(
                session, session.TrackColours[SCHEME_TRACK].WithIndex(imageId), { -16, -16, height },
                { { -16, -16, height }, { 32, 32, 2 } });
        }
    }
    else if (drawRail)
    {
        PaintAddImageAsParent(
            session, session.TrackColours[SCHEME_TRACK].WithIndex(imageId), { -16, -16, height },
            { { -16, -16, height }, { 32, 32, 2 } });
    }

    PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
    PaintUtilSetGeneralSupportHeight(session, height + 48, 0x20);
}

/** rct2: 0x008AD220 */
static void MiniatureRailwayTrackDiag25DegUpToFlat(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    static constexpr int8_t supportOffsets[][4] = {
        { 0, +8, +8, +8 },
        { 0, -8, -8, 0 },
        { 0, +8, +8, +8 },
        { 0, -8, -8, +8 },
    };

    int8_t supportFunction = WOOD_B;
    if (trackSequence == 3)
    {
        supportFunction = WOOD_A;
    }

    bool hasSupports = false;
    auto supportType = _diagSupportTypes[direction][trackSequence];

    uint32_t floorImage = 0;
    CoordsXY floorBoundSize;
    CoordsXY floorBoundOffset;

    if (supportType.has_value())
    {
        const auto enumValue = EnumValue(supportType.value());
        floorImage = floors[enumValue].image_id;
        floorBoundSize = floors[enumValue].bound_size;
        floorBoundOffset = floors[enumValue].bound_offset;
        hasSupports = WoodenSupportsPaintSetup(
            session, supportFunction, enumValue, 0, height + supportOffsets[direction][trackSequence],
            session.TrackColours[SCHEME_SUPPORTS]);
    }

    if (direction == 1 && trackSequence == 3)
    {
        floorBoundOffset = CoordsXY(-16, -16);
    }

    static constexpr int8_t offsetsB[4][4][2] = {
        { { 0, 0 }, { +8, +16 }, { +8, +8 }, { +8, +8 } },
        { { 0, 0 }, { -8, -8 }, { -8, -8 }, { +8, +8 } },
        { { 0, 0 }, { +8, +8 }, { +8, +16 }, { +8, +8 } },
        { { 0, 0 }, { -8, -8 }, { -8, -8 }, { +8, +8 } },
    };

    uint32_t imageId = miniature_railway_track_pieces_diag_25_deg_up_to_flat[direction];
    bool drawRail = miniature_railway_diag_image_segment[direction][trackSequence];

    static constexpr int8_t railOffsets[] = { +8, 0, +8, +8 };

    if (hasSupports)
    {
        PaintAddImageAsParent(
            session, session.TrackColours[SCHEME_SUPPORTS].WithIndex(floorImage),
            { 0, 0, height + offsetsB[direction][trackSequence][0] },
            { { floorBoundOffset, height + offsetsB[direction][trackSequence][1] }, { floorBoundSize, (drawRail ? 2 : 0) } });
        if (drawRail)
        {
            PaintAddImageAsChild(
                session, session.TrackColours[SCHEME_TRACK].WithIndex(imageId), { -16, -16, height },
                { { -16, -16, height + railOffsets[direction] }, { 32, 32, 2 } });
        }
    }
    else if (drawRail)
    {
        PaintAddImageAsParent(
            session, session.TrackColours[SCHEME_TRACK].WithIndex(imageId), { -16, -16, height },
            { { -16, -16, height + railOffsets[direction] }, { 32, 32, 2 } });
    }

    PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
    PaintUtilSetGeneralSupportHeight(session, height + 56, 0x20);
}

/** rct2: 0x008AD260 */
static void MiniatureRailwayTrackDiag25DegDown(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    static constexpr int8_t supportOffsets[][4] = {
        { 0, +16, +16, 0 },
        { 0, -8, -8, 0 },
        { 0, +16, +16, 0 },
        { 0, -8, -8, -8 },
    };

    int8_t supportFunction = WOOD_B;
    if (trackSequence == 3)
    {
        supportFunction = WOOD_A;
    }

    bool hasSupports = false;

    uint32_t floorImage = 0;
    CoordsXY floorBoundSize;
    CoordsXY floorBoundOffset;
    auto supportType = _diagSupportTypes[direction][trackSequence];

    if (supportType.has_value())
    {
        const auto enumValue = EnumValue(supportType.value());
        floorImage = floors[enumValue].image_id;
        floorBoundSize = floors[enumValue].bound_size;
        floorBoundOffset = floors[enumValue].bound_offset;
        hasSupports = WoodenSupportsPaintSetup(
            session, supportFunction, enumValue, 0, height + supportOffsets[direction][trackSequence],
            session.TrackColours[SCHEME_SUPPORTS]);
    }

    if (direction == 1 && trackSequence == 3)
    {
        floorBoundOffset = CoordsXY(-16, -16);
    }

    uint32_t imageId = miniature_railway_track_pieces_diag_25_deg_up[(direction + 2) % 4];
    bool drawRail = miniature_railway_diag_image_segment[direction][trackSequence];
    static constexpr int8_t railOffsets[] = { 0, +8, +8, +8 };

    static constexpr int8_t offsetsB[4][4][2] = {
        { { 0, 0 }, { +8, +8 }, { +8, +8 }, { -8, -8 } },
        { { 0, 0 }, { -8, -8 }, { -8, -8 }, { -8, 0 } },
        { { 0, 0 }, { +8, +8 }, { +8, +16 }, { -8, -8 } },
        { { 0, 0 }, { -8, -8 }, { -8, -8 }, { -8, -8 } },
    };

    if (hasSupports)
    {
        PaintAddImageAsParent(
            session, session.TrackColours[SCHEME_SUPPORTS].WithIndex(floorImage),
            { 0, 0, height + offsetsB[direction][trackSequence][0] },
            { { floorBoundOffset, height + offsetsB[direction][trackSequence][1] }, { floorBoundSize, (drawRail ? 2 : 0) } });
        if (drawRail)
        {
            PaintAddImageAsChild(
                session, session.TrackColours[SCHEME_TRACK].WithIndex(imageId), { -16, -16, height },
                { { -16, -16, height + railOffsets[direction] }, { 32, 32, 2 } });
        }
    }
    else if (drawRail)
    {
        PaintAddImageAsParent(
            session, session.TrackColours[SCHEME_TRACK].WithIndex(imageId), { -16, -16, height },
            { { -16, -16, height + railOffsets[direction] }, { 32, 32, 2 } });
    }

    PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
    PaintUtilSetGeneralSupportHeight(session, height + 56, 0x20);
}

/** rct2: 0x008AD240 */
static void MiniatureRailwayTrackDiagFlatTo25DegDown(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    static constexpr int8_t supportOffsets[][4] = {
        { 0, +16, +16, 0 },
        { 0, -8, -8, -8 },
        { 0, +8, +8, 0 },
        { 0, -8, -8, -8 },
    };

    int8_t supportFunction = WOOD_B;
    if (trackSequence == 3)
    {
        supportFunction = WOOD_A;
    }

    bool hasSupports = false;
    uint32_t floorImage = 0;
    CoordsXY floorBoundSize;
    CoordsXY floorBoundOffset;
    auto supportType = _diagSupportTypes[direction][trackSequence];

    if (supportType.has_value())
    {
        const auto enumValue = EnumValue(supportType.value());
        floorImage = floors[enumValue].image_id;
        floorBoundSize = floors[enumValue].bound_size;
        floorBoundOffset = floors[enumValue].bound_offset;
        hasSupports = WoodenSupportsPaintSetup(
            session, supportFunction, enumValue, 0, height + supportOffsets[direction][trackSequence],
            session.TrackColours[SCHEME_SUPPORTS]);
    }

    if (direction == 1 && trackSequence == 3)
    {
        floorBoundOffset = CoordsXY(-16, -16);
    }

    static constexpr int8_t offsetsB[4][4][2] = {
        { { 0, 0 }, { +8, +8 }, { +8, +8 }, { -8, -8 } },
        { { 0, 0 }, { -8, -8 }, { -8, -8 }, { -8, 0 } },
        { { 0, 0 }, { +8, +8 }, { +8, +16 }, { -8, -8 } },
        { { 0, 0 }, { -8, -8 }, { -8, -8 }, { -8, -8 } },
    };

    uint32_t imageId = miniature_railway_track_pieces_diag_25_deg_up_to_flat[(direction + 2) % 4];
    bool drawRail = miniature_railway_diag_image_segment[direction][trackSequence];
    static constexpr int8_t railOffsets[] = { 0, +8, +8, +8 };

    if (hasSupports)
    {
        PaintAddImageAsParent(
            session, session.TrackColours[SCHEME_SUPPORTS].WithIndex(floorImage),
            { 0, 0, height + offsetsB[direction][trackSequence][0] },
            { { floorBoundOffset, height + offsetsB[direction][trackSequence][1] }, { floorBoundSize, (drawRail ? 2 : 0) } });
        if (drawRail)
        {
            PaintAddImageAsChild(
                session, session.TrackColours[SCHEME_TRACK].WithIndex(imageId), { -16, -16, height },
                { { -16, -16, height + railOffsets[direction] }, { 32, 32, 2 } });
        }
    }
    else if (drawRail)
    {
        PaintAddImageAsParent(
            session, session.TrackColours[SCHEME_TRACK].WithIndex(imageId), { -16, -16, height },
            { { -16, -16, height + railOffsets[direction] }, { 32, 32, 2 } });
    }

    PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
    PaintUtilSetGeneralSupportHeight(session, height + 56, 0x20);
}

static void MiniatureRailwayTrackDiag25DegDownToFlat(
    PaintSession& session, const Ride& ride, uint8_t trackSequence, uint8_t direction, int32_t height,
    const TrackElement& trackElement)
{
    bool hasSupports = false;
    uint32_t floorImage = 0;
    CoordsXY floorBoundSize;
    CoordsXY floorBoundOffset;
    auto supportType = _diagSupportTypes[direction][trackSequence];

    if (supportType.has_value())
    {
        const auto enumValue = EnumValue(supportType.value());
        floorImage = floors[enumValue].image_id;
        floorBoundSize = floors[enumValue].bound_size;
        floorBoundOffset = floors[enumValue].bound_offset;
        hasSupports = WoodenASupportsPaintSetup(
            session, WoodenSupportType::Truss, supportType.value(), height, session.TrackColours[SCHEME_SUPPORTS]);
    }

    if (direction == 1 && trackSequence == 3)
    {
        floorBoundOffset = CoordsXY(-16, -16);
    }

    uint32_t imageId = miniature_railway_track_pieces_diag_flat_to_25_deg_up[(direction + 2) % 4];
    bool drawRail = miniature_railway_diag_image_segment[direction][trackSequence];

    if (hasSupports)
    {
        PaintAddImageAsParent(
            session, session.TrackColours[SCHEME_SUPPORTS].WithIndex(floorImage), { 0, 0, height },
            { { floorBoundOffset, height }, { floorBoundSize, (drawRail ? 2 : 0) } });
        if (drawRail)
        {
            PaintAddImageAsChild(
                session, session.TrackColours[SCHEME_TRACK].WithIndex(imageId), { -16, -16, height },
                { { -16, -16, height }, { 32, 32, 2 } });
        }
    }
    else if (drawRail)
    {
        PaintAddImageAsParent(
            session, session.TrackColours[SCHEME_TRACK].WithIndex(imageId), { -16, -16, height },
            { { -16, -16, height }, { 32, 32, 2 } });
    }

    PaintUtilSetSegmentSupportHeight(session, SEGMENTS_ALL, 0xFFFF, 0);
    PaintUtilSetGeneralSupportHeight(session, height + 48, 0x20);
}

/**
 * rct2: 0x008ACE48
 */
TRACK_PAINT_FUNCTION GetTrackPaintFunctionMiniatureRailway(int32_t trackType)
{
    switch (trackType)
    {
        case TrackElemType::Flat:
            return PaintMiniatureRailwayTrackFlat;

        case TrackElemType::EndStation:
        case TrackElemType::BeginStation:
        case TrackElemType::MiddleStation:
            return PaintMiniatureRailwayStation;

        case TrackElemType::Up25:
            return PaintMiniatureRailwayTrack25DegUp;
        case TrackElemType::FlatToUp25:
            return PaintMiniatureRailwayTrackFlatTo25DegUp;
        case TrackElemType::Up25ToFlat:
            return PaintMiniatureRailwayTrack25DegUpToFlat;

        case TrackElemType::Down25:
            return PaintMiniatureRailwayTrack25DegDown;
        case TrackElemType::FlatToDown25:
            return PaintMiniatureRailwayTrackFlatTo25DegDown;
        case TrackElemType::Down25ToFlat:
            return PaintMiniatureRailwayTrack25DegDownToFlat;

        case TrackElemType::LeftQuarterTurn5Tiles:
            return PaintMiniatureRailwayTrackLeftQuarterTurn5Tiles;
        case TrackElemType::RightQuarterTurn5Tiles:
            return PaintMiniatureRailwayTrackRightQuarterTurn5Tiles;

        case TrackElemType::SBendLeft:
            return PaintMiniatureRailwayTrackSBendLeft;
        case TrackElemType::SBendRight:
            return PaintMiniatureRailwayTrackSBendRight;

        case TrackElemType::LeftQuarterTurn3Tiles:
            return PaintMiniatureRailwayTrackLeftQuarterTurn3Tiles;
        case TrackElemType::RightQuarterTurn3Tiles:
            return PaintMiniatureRailwayTrackRightQuarterTurn3Tiles;

        case TrackElemType::LeftEighthToDiag:
            return PaintMiniatureRailwayTrackLeftEighthToDiag;
        case TrackElemType::RightEighthToDiag:
            return PaintMiniatureRailwayTrackRightEighthToDiag;
        case TrackElemType::LeftEighthToOrthogonal:
            return PaintMiniatureRailwayTrackLeftEighthToOrthogonal;
        case TrackElemType::RightEighthToOrthogonal:
            return PaintMiniatureRailwayTrackRightEighthToOrthogonal;

        case TrackElemType::DiagFlat:
            return MiniatureRailwayTrackDiagFlat;
        case TrackElemType::DiagUp25:
            return MiniatureRailwayTrackDiag25DegUp;
        case TrackElemType::DiagFlatToUp25:
            return MiniatureRailwayTrackDiagFlatTo25DegUp;
        case TrackElemType::DiagUp25ToFlat:
            return MiniatureRailwayTrackDiag25DegUpToFlat;
        case TrackElemType::DiagDown25:
            return MiniatureRailwayTrackDiag25DegDown;
        case TrackElemType::DiagFlatToDown25:
            return MiniatureRailwayTrackDiagFlatTo25DegDown;
        case TrackElemType::DiagDown25ToFlat:
            return MiniatureRailwayTrackDiag25DegDownToFlat;
    }

    return nullptr;
}
