﻿/*
neogfx C++ App/Game Engine - Examples - Games - Chess
Copyright(C) 2020 Leigh Johnston

This program is free software: you can redistribute it and / or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <vector>

#include <chess/mailbox.hpp>

namespace chess
{
    template<>
    mailbox_position const& setup_position<mailbox_rep>()
    {
        static const mailbox_position position
        {
            {{
                { piece::WhiteRook, piece::WhiteKnight, piece::WhiteBishop, piece::WhiteQueen, piece::WhiteKing, piece::WhiteBishop, piece::WhiteKnight, piece::WhiteRook },
                { piece::WhitePawn, piece::WhitePawn, piece::WhitePawn, piece::WhitePawn, piece::WhitePawn, piece::WhitePawn, piece::WhitePawn, piece::WhitePawn },
                {}, {}, {}, {},
                { piece::BlackPawn, piece::BlackPawn, piece::BlackPawn, piece::BlackPawn, piece::BlackPawn, piece::BlackPawn, piece::BlackPawn, piece::BlackPawn },
                { piece::BlackRook, piece::BlackKnight, piece::BlackBishop, piece::BlackQueen, piece::BlackKing, piece::BlackBishop, piece::BlackKnight, piece::BlackRook },
            }},
            {{
                { 4u, 0u }, { 4u, 7u }
            }},
            player::White
        };
        return position;
    }

    template<>
    move_tables<mailbox_rep> generate_move_tables<mailbox_rep>()
    {
        typedef move_tables<mailbox_rep>::move_coordinates move_coordinates;
        move_tables<mailbox_rep> result
        {
            // unit moves
            {{
                {{
                    { { 0, 1 } }, // pawn
                    { { 2, 1 }, { 2, -1 }, { 1, 2 }, { 1, -2 }, { -1, 2 }, { -1, -2 }, { -2, 1 }, { -2, -1 } }, // knight
                    { { -1, 1 }, { 1, 1 }, { 1, -1 }, { -1, -1 } }, // bishop
                    { { 1, 0 }, { 0, 1 }, { -1, 0 }, { 0, -1 } }, // rook
                    { { 1, 0 }, { 0, 1 }, { -1, 0 }, { 0, -1 }, { 1, 1 }, { -1, 1 }, { -1, -1 }, { 1, -1 } }, // queen
                    { { 1, 0 }, { 0, 1 }, { -1, 0 }, { 0, -1 }, { 1, 1 }, { -1, 1 }, { -1, -1 }, { 1, -1 } } // king
                }},
                {{
                    { { 0, -1 } }, // pawn
                    { { 2, 1 }, { 2, -1 }, { 1, 2 }, { 1, -2 }, { -1, 2 }, { -1, -2 }, { -2, 1 }, { -2, -1 } }, // knight
                    { { -1, 1 }, { 1, 1 }, { 1, -1 }, { -1, -1 } }, // bishop
                    { { 1, 0 }, { 0, 1 }, { -1, 0 }, { 0, -1 } }, // rook
                    { { 1, 0 }, { 0, 1 }, { -1, 0 }, { 0, -1 }, { 1, 1 }, { -1, 1 }, { -1, -1 }, { 1, -1 } }, // queen
                    { { 1, 0 }, { 0, 1 }, { -1, 0 }, { 0, -1 }, { 1, 1 }, { -1, 1 }, { -1, -1 }, { 1, -1 } } // king
                }},
            }},
            // capture moves
            {{
                {{
                    { { -1, 1 }, { 1, 1 } }, // pawn
                    { { 2, 1 }, { 2, -1 }, { 1, 2 }, { 1, -2 }, { -1, 2 }, { -1, -2 }, { -2, 1 }, { -2, -1 } }, // knight
                    { { -1, 1 }, { 1, 1 }, { 1, -1 }, { -1, -1 } }, // bishop
                    { { 1, 0 }, { 0, 1 }, { -1, 0 }, { 0, -1 } }, // rook
                    { { 1, 0 }, { 0, 1 }, { -1, 0 }, { 0, -1 }, { 1, 1 }, { -1, 1 }, { -1, -1 }, { 1, -1 } }, // queen
                    { { 1, 0 }, { 0, 1 }, { -1, 0 }, { 0, -1 }, { 1, 1 }, { -1, 1 }, { -1, -1 }, { 1, -1 } } // king
                }},
                {{
                    { { -1, -1 }, { 1, -1 } }, // pawn
                    { { 2, 1 }, { 2, -1 }, { 1, 2 }, { 1, -2 }, { -1, 2 }, { -1, -2 }, { -2, 1 }, { -2, -1 } }, // knight
                    { { -1, 1 }, { 1, 1 }, { 1, -1 }, { -1, -1 } }, // bishop
                    { { 1, 0 }, { 0, 1 }, { -1, 0 }, { 0, -1 } }, // rook
                    { { 1, 0 }, { 0, 1 }, { -1, 0 }, { 0, -1 }, { 1, 1 }, { -1, 1 }, { -1, -1 }, { 1, -1 } }, // queen
                    { { 1, 0 }, { 0, 1 }, { -1, 0 }, { 0, -1 }, { 1, 1 }, { -1, 1 }, { -1, -1 }, { 1, -1 } } // king
                }}
            }},
            // can move multiple of unit move
            {{
                false, // pawn
                false, // knight
                true, // bishop
                true, // rook
                true, // queen
                false // king
            }}
        };
        // all pieces
        for (std::size_t pieceColorIndex = 0u; pieceColorIndex < PIECE_COLORS; ++pieceColorIndex)
            for (std::size_t pieceTypeIndex = 0u; pieceTypeIndex < PIECE_TYPES; ++pieceTypeIndex)
                for (coordinate yFrom = 0u; yFrom <= 7u; ++yFrom)
                    for (coordinate xFrom = 0u; xFrom <= 7u; ++xFrom)
                        for (coordinate yTo = 0u; yTo <= 7u; ++yTo)
                            for (coordinate xTo = 0u; xTo <= 7u; ++xTo)
                            {
                                auto calc_validity = [&](move_tables<mailbox_rep>::unit_moves const& aUnitMoves, bool& aResult) 
                                {
                                    aResult = false;
                                    auto const delta = move_coordinates{ static_cast<int32_t>(xTo), static_cast<int32_t>(yTo) } - move_coordinates{ static_cast<int32_t>(xFrom), static_cast<int32_t>(yFrom) };
                                    auto const& unitMoves = aUnitMoves[pieceColorIndex][pieceTypeIndex];
                                    if (std::find(unitMoves.begin(), unitMoves.end(), delta) != unitMoves.end())
                                        aResult = true;
                                    else if (result.canMoveMultiple[pieceTypeIndex] && (std::abs(delta.x) == std::abs(delta.y) || delta.x == 0 || delta.y == 0))
                                    {
                                        auto const& deltaUnity = neogfx::delta_i32{ delta.x != 0 ? delta.x / std::abs(delta.x) : 0, delta.y != 0 ? delta.y / std::abs(delta.y) : 0 };
                                        if (std::find(unitMoves.begin(), unitMoves.end(), deltaUnity) != unitMoves.end())
                                            aResult = true;
                                    }
                                };
                                calc_validity(result.unitMoves, result.validMoves[pieceColorIndex][pieceTypeIndex][yFrom][xFrom][yTo][xTo]);
                                calc_validity(result.unitCaptureMoves, result.validCaptureMoves[pieceColorIndex][pieceTypeIndex][yFrom][xFrom][yTo][xTo]);
                            }
        // pawn (first move)
        for (coordinate x = 0u; x <= 7u; ++x)
        {
            result.validMoves[static_cast<std::size_t>(piece_color_cardinal::White)][static_cast<std::size_t>(piece_cardinal::Pawn)][1u][x][3u][x] = true;
            result.validMoves[static_cast<std::size_t>(piece_color_cardinal::Black)][static_cast<std::size_t>(piece_cardinal::Pawn)][6u][x][4u][x] = true;
        }
        for (coordinate_i32 yFrom = 0; yFrom <= 7; ++yFrom)
            for (coordinate_i32 xFrom = 0; xFrom <= 7; ++xFrom)
                for (coordinate_i32 yTo = 0; yTo <= 7; ++yTo)
                    for (coordinate_i32 xTo = 0; xTo <= 7; ++xTo)
                    {
                        bool aKnight = false;
                        if (!can_move_trivial(coordinates_i32{ xFrom, yFrom }.as<coordinate>(), coordinates_i32{ xTo, yTo }.as<coordinate>(), aKnight))
                            continue;
                        result.trivialMoves[yFrom][xFrom][yTo][xTo] = true;
                        auto const start = coordinates_i32{ xFrom, yFrom };
                        auto const end = coordinates_i32{ xTo, yTo };
                        if (aKnight)
                        {
                            result.movePaths[yFrom][xFrom][yTo][xTo].second[result.movePaths[yFrom][xFrom][yTo][xTo].first++] = start;
                            result.movePaths[yFrom][xFrom][yTo][xTo].second[result.movePaths[yFrom][xFrom][yTo][xTo].first++] = end;
                            continue;
                        }
                        auto const delta = move_tables<mailbox_rep>::move_coordinates{ xTo, yTo } - move_tables<mailbox_rep>::move_coordinates{ xFrom, yFrom };
                        auto const& deltaUnity = neogfx::delta_i32{ delta.x != 0 ? delta.x / std::abs(delta.x) : 0, delta.y != 0 ? delta.y / std::abs(delta.y) : 0 };
                        auto pos = start;
                        for(;;)
                        {
                            result.movePaths[yFrom][xFrom][yTo][xTo].second[result.movePaths[yFrom][xFrom][yTo][xTo].first++] = pos;
                            if (pos == end)
                                break;
                            pos += deltaUnity;
                        }
                    }
        return result;
    }

    template <player Player>
    struct eval<mailbox_rep, Player>
    {
        eval_result operator()(move_tables<mailbox_rep> const& aTables, mailbox_position& aPosition, double aPly, eval_info* aEvalInfo = nullptr)
        {
            auto const start = !aEvalInfo ? std::chrono::steady_clock::time_point{} : std::chrono::steady_clock::now();

            eval_result result = {};

            double constexpr scaleMaterial = 100.0; // todo
            double constexpr scalePromotion = 0.01; // todo
            double constexpr scaleMobility = 0.01; // todo
            double constexpr scaleAttack = 0.04; // todo
            double constexpr scaleDefend = 0.02; // todo
            double constexpr scaleCheck = 1.0; // todo
            double constexpr scaleAttackAdvantage = 0.04; // todo
            double const scaleMate = 1.0 / std::pow(10.0, aPly);
            double constexpr stalemate = 0.0;
            double material = 0.0;
            double mobility = 0.0;
            double attack = 0.0;
            double defend = 0.0;
            bool mobilityPlayer = false;
            bool mobilityPlayerKing = false;
            double checkedPlayerKing = 0.0;
            bool mobilityOpponent = false;
            bool mobilityOpponentKing = false;
            double checkedOpponentKing = 0.0;
            if (chess::draw(aPosition))
            {
                if (aEvalInfo)
                {
                    auto const end_us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start);
                    *aEvalInfo = eval_info{ material, mobility, attack, defend, mobilityPlayer, mobilityOpponent, mobilityPlayerKing, mobilityOpponentKing, checkedPlayerKing, checkedOpponentKing, result.eval, end_us };
                }
                return eval_result{ eval_node::Terminal, stalemate };
            }
            for (coordinate yFrom = 0u; yFrom <= 7u; ++yFrom)
                for (coordinate xFrom = 0u; xFrom <= 7u; ++xFrom)
                {
                    auto const from = piece_at(aPosition, coordinates{ xFrom, yFrom });
                    if (from == piece::None)
                        continue;
                    auto const playerFrom = static_cast<chess::player>(piece_color(from));
                    auto const valueFrom = player_piece_value<Player>(from);
                    material += valueFrom;
                    for (coordinate yTo = 0u; yTo <= 7u; ++yTo)
                        for (coordinate xTo = 0u; xTo <= 7u; ++xTo)
                        {
                            move const candidateMove{ { xFrom, yFrom }, { xTo, yTo } };
                            if (!can_move_trivial(aTables, candidateMove.from, candidateMove.to))
                                continue;
                            auto const to = piece_at(aPosition, candidateMove.to );
                            auto const playerTo = static_cast<chess::player>(piece_color(to));
                            auto const valueTo = player_piece_value<Player>(to);
                            if (can_move<true, false, true>(aTables, Player, aPosition, candidateMove))
                            {
                                if (playerFrom != playerTo)
                                {
                                    if (from == (piece::King | static_cast<piece>(Player)))
                                        mobilityPlayerKing = true;
                                    else
                                        mobilityPlayer = true;
                                    if (to == (piece::King | static_cast<piece>(opponent_v<Player>)))
                                        checkedOpponentKing = 1.0;
                                    if (from == (piece::Pawn | static_cast<piece>(Player)) && yTo == promotion_rank_v<Player>)
                                        material += (player_piece_value<Player>(piece::Queen) * scalePromotion);
                                    mobility += 1.0;
                                    if (playerTo == opponent_v<Player>)
                                        attack -= valueTo * scaleAttackAdvantage;
                                }
                                else
                                    defend += valueTo;
                            }
                            else if (can_move<true, false, true>(aTables, opponent_v<Player>, aPosition, candidateMove))
                            {
                                if (playerFrom != playerTo)
                                {
                                    if (from == (piece::King | static_cast<piece>(opponent_v<Player>)))
                                        mobilityOpponentKing = true;
                                    else
                                        mobilityOpponent = true;
                                    if (to == (piece::King | static_cast<piece>(Player)))
                                        checkedPlayerKing = 1.0;
                                    if (from == (piece::Pawn | static_cast<piece>(opponent_v<Player>)) && yTo == promotion_rank_v<opponent_v<Player>>)
                                        material -= (player_piece_value<Player>(piece::Queen) * scalePromotion);
                                    mobility -= 1.0;
                                    if (playerTo == Player)
                                        attack -= valueTo;
                                }
                                else
                                    defend += valueTo;
                            }
                        }
                }
            material *= scaleMaterial;
            mobility *= scaleMobility;
            attack *= scaleAttack;
            defend *= scaleDefend;
            result.eval = material + mobility + attack + defend;
            result.eval -= (checkedPlayerKing * scaleCheck);
            result.eval += (checkedOpponentKing * scaleCheck);
            if (!mobilityPlayerKing)
            {
                if (checkedPlayerKing != 0.0)
                {
                    if (!mobilityPlayer)
                    {
                        result.node = eval_node::Terminal;
                        result.eval = -std::numeric_limits<double>::max() * scaleMate;
                    }
                }
                else if (!mobilityPlayer)
                {
                    result.node = eval_node::Terminal;
                    result.eval = stalemate;
                }
            }
            if (!mobilityOpponentKing)
            {
                if (checkedOpponentKing != 0.0)
                {
                    if (!mobilityOpponent)
                    {
                        result.node = eval_node::Terminal;
                        result.eval = +std::numeric_limits<double>::max() * scaleMate;
                    }
                }
                else if (!mobilityOpponent)
                {
                    result.node = eval_node::Terminal;
                    result.eval = stalemate;
                }
            }

            if (aEvalInfo)
            {
                auto const end_us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start);
                *aEvalInfo = eval_info{ material, mobility, attack, defend, mobilityPlayer, mobilityOpponent, mobilityPlayerKing, mobilityOpponentKing, checkedPlayerKing, checkedOpponentKing, result.eval, end_us };
            }

            return result;
        }
        eval_result operator()(move_tables<mailbox_rep> const& aTables, mailbox_position& aPosition, double aPly, eval_info& aEvalInfo)
        {
            return eval{}(aTables, aPosition, aPly, &aEvalInfo);
        }
    };


    template struct eval<mailbox_rep, player::White>;
    template struct eval<mailbox_rep, player::Black>;
}