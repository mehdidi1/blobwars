#include "strategy.h"
#include "greedy.h"
#include <sys/time.h>
#include <fstream>
#include "minimax.h"
#include "minimax_para.h"
#include "alpha_beta.h"
#include "alpha_beta_para.h"
#include "feldman.h"
#include "final_ai.h"
#include "test.h"
#include <vector> // Make sure vector is included

void Strategy::applyMove(const movement &mv)
{
    // Copy or jump blob to the new position
    _blobs.set(mv.nx, mv.ny, _current_player);

    // If it's a jump (distance 2), remove the blob from original position
    int dist = std::max(abs(mv.nx - mv.ox), abs(mv.ny - mv.oy));
    if (dist == 2)
        _blobs.set(mv.ox, mv.oy, -1);

    // Convert adjacent opponent blobs
    for (int dx = -1; dx <= 1; dx++)
    {
        for (int dy = -1; dy <= 1; dy++)
        {
            int x = mv.nx + dx;
            int y = mv.ny + dy;

            // Check bounds
            if (x < 0 || x >= 8 || y < 0 || y >= 8)
                continue;

            // If opponent's blob, convert it
            if (_blobs.get(x, y) == (1 - _current_player))
                _blobs.set(x, y, _current_player);
        }
    }
    _current_player = !_current_player;
}

// Default scoring function
Sint32 Strategy::estimateCurrentScore(Sint32 player) const
{
    Sint32 player0_score = 0;
    Sint32 player1_score = 0;

    // Count blobs for each player
    for (int x = 0; x < 8; x++)
    {
        for (int y = 0; y < 8; y++)
        {
            if (_blobs.get(x, y) == 0)
                player0_score++;
            else if (_blobs.get(x, y) == 1)
                player1_score++;
        }
    }

    if (player == 1)
    {
        return player1_score - player0_score;
    }

    return player0_score - player1_score;
}

Sint32 Strategy::estimateCurrentScoreImproved(Sint32 player) const
{
    Sint32 opponent = 1 - player;
    Sint32 my_blobs = 0, opp_blobs = 0;
    Sint32 my_corner_blobs = 0, opp_corner_blobs = 0;
    Sint32 my_potential_captures = 0;         // Opponent blobs adjacent to mine
    Sint32 my_frontier = 0, opp_frontier = 0; // Blobs adjacent to empty squares

    // --- Iterate Board ---
    for (int x = 0; x < 8; ++x)
    {
        for (int y = 0; y < 8; ++y)
        {
            Sint32 blob_owner = _blobs.get(x, y);
            bool is_corner = (x == 0 || x == 7) && (y == 0 || y == 7);

            if (blob_owner == player)
            {
                my_blobs++;
                if (is_corner)
                    my_corner_blobs++;

                bool is_frontier = false;
                for (int dx = -1; dx <= 1; ++dx)
                {
                    for (int dy = -1; dy <= 1; ++dy)
                    {
                        if (dx == 0 && dy == 0)
                            continue;
                        int nx = x + dx, ny = y + dy;
                        if (nx >= 0 && nx < 8 && ny >= 0 && ny < 8)
                        {
                            if (_blobs.get(nx, ny) == opponent)
                            {
                                my_potential_captures++;
                            }
                            else if (_blobs.get(nx, ny) == -1 && !_holes.get(nx, ny))
                            {
                                is_frontier = true; // Adjacent to empty non-hole
                            }
                        }
                    }
                }
                if (is_frontier)
                    my_frontier++;
            }
            else if (blob_owner == opponent)
            {
                opp_blobs++;
                if (is_corner)
                    opp_corner_blobs++;

                bool is_frontier = false;
                for (int dx = -1; dx <= 1; ++dx)
                {
                    for (int dy = -1; dy <= 1; ++dy)
                    {
                        if (dx == 0 && dy == 0)
                            continue;
                        int nx = x + dx, ny = y + dy;
                        if (nx >= 0 && nx < 8 && ny >= 0 && ny < 8)
                        {
                            if (_blobs.get(nx, ny) == -1 && !_holes.get(nx, ny))
                            {
                                is_frontier = true;
                            }
                        }
                    }
                }
                if (is_frontier)
                    opp_frontier++;
            }
        }
    }

    // --- Calculate Scores ---
    // 1. Material Score (Blob difference) - High weight
    Sint32 material_score = 100 * (my_blobs - opp_blobs);
    if (my_blobs == 0)
        return -99999; // Loss condition
    if (opp_blobs == 0)
        return 99999; // Win condition

    // 2. Corner Score - Very high weight
    Sint32 corner_score = 50 * (my_corner_blobs - opp_corner_blobs);

    // 3. Mobility Score - Moderate weight
    Sint32 my_moves = 0, opp_moves = 0;
    {
        Strategy tmp = *this;
        vector<movement> moves;
        tmp._current_player = player;
        tmp.computeValidMoves(moves);
        my_moves = moves.size();
        moves.clear();
        tmp._current_player = opponent;
        tmp.computeValidMoves(moves);
        opp_moves = moves.size();
    }
    Sint32 mobility_score = 20 * (my_moves - opp_moves);


    // 5. Potential Capture Score - Lower weight
    Sint32 capture_score = 30 * my_potential_captures; // Only count ours for simplicity

    // 6. Frontier Score (Penalty for vulnerable blobs) - Lower negative weight
    Sint32 frontier_score = -50 * (my_frontier - opp_frontier);

    // Determine current game phase
    GamePhase phase = detectGamePhase();

    // Phase-specific weights
    Sint32 material_weight, corner_weight, mobility_weight;
    Sint32 capture_weight, frontier_weight;

    switch (phase)
    {


    case EARLY_GAME:
        // Early game: emphasize position and corners
        material_weight = 120;
        corner_weight = 129; // Higher emphasis on corners
        mobility_weight = 23;
        capture_weight = 13;
        frontier_weight = -5; // Less penalty for frontiers early
        break;

    case MID_GAME:
        // Mid game: emphasize mobility and potential captures
        material_weight = 97;
        corner_weight = 84;
        mobility_weight = 5; // Higher emphasis on mobility
        capture_weight = 3; // Higher emphasis on potential captures
        frontier_weight = -50;
        break;

    case LATE_GAME:
        // Late game: emphasize material count and reduce mobility importance
        material_weight = 167; // Higher emphasis on material
        corner_weight = 74;
        mobility_weight = 9; // Less emphasis on mobility
        capture_weight = 6;
        frontier_weight = -29; // More penalty for vulnerable blobs
        break;
    }

    // Apply phase-specific weights to scores
    material_score = material_weight * (my_blobs - opp_blobs);
    if (my_blobs == 0)
        return -99999; // Loss condition
    if (opp_blobs == 0)
        return 99999; // Win condition

    corner_score = corner_weight * (my_corner_blobs - opp_corner_blobs);
    mobility_score = mobility_weight * (my_moves - opp_moves);
    capture_score = capture_weight * my_potential_captures;
    frontier_score = frontier_weight * (my_frontier - opp_frontier);

    // --- Combine Scores ---
    Sint32 total_score = material_score + corner_score + mobility_score + capture_score + frontier_score ;

    return total_score;
}


vector<movement> &Strategy::computeValidMoves(vector<movement> &valid_moves) const
{
    valid_moves.clear();

    // Iterate over all positions on the board
    for (int ox = 0; ox < 8; ox++)
    {
        for (int oy = 0; oy < 8; oy++)
        {
            // Check if current position contains a blob of the current player
            if (_blobs.get(ox, oy) == (int)_current_player)
            {

                // Try all possible destination positions within distance 2
                for (int nx = std::max(0, ox - 2); nx <= std::min(7, ox + 2); nx++)
                {
                    for (int ny = std::max(0, oy - 2); ny <= std::min(7, oy + 2); ny++)
                    {
                        // Skip the original position
                        if (nx == ox && ny == oy)
                            continue;

                        // Skip if destination is a hole
                        if (_holes.get(nx, ny))
                            continue;

                        // Skip if destination already has a blob
                        if (_blobs.get(nx, ny) != -1)
                            continue;

                        // Check distance for move validity
                        int dist = std::max(abs(nx - ox), abs(ny - oy));
                        if (dist <= 2)
                        {
                            valid_moves.push_back(movement(ox, oy, nx, ny));
                        }
                    }
                }
            }
        }
    }
    return valid_moves;
}

// Strategy selector
// void Strategy::computeBestMove()
// {
//     // Add timing code
//     struct timeval start_time, end_time;
//     gettimeofday(&start_time, NULL);

//     // Existing code
// #ifdef USE_GREEDY
//     greedy::computeBestMoveWithScore(*this);
// #elif defined(USE_MINIMAX)
//     minimax::computeBestMoveWithScore(*this);
// #else
//     // Default strategy implementation...
//     movement best_move(0, 0, 0, 0);
//     vector<movement> valid_moves;
//     computeValidMoves(valid_moves);

//     if (!valid_moves.empty())
//     {
//         best_move = valid_moves[0]; // Just pick the first valid move
//     }

//     _saveBestMove(best_move);
// #endif

//     // Calculate and display elapsed time
//     gettimeofday(&end_time, NULL);
//     double elapsed = ((end_time.tv_sec - start_time.tv_sec) * 1000.0) +
//                      ((end_time.tv_usec - start_time.tv_usec) / 1000.0);
//     std::cout << "Best move computation took " << elapsed << " milliseconds" << std::endl;
// }

void Strategy::computeBestMove()
{
    // Player 1 is blue
    // Player 0 is red

    // Read previous statistics - separate for each player
    double total_time_p0 = 0.0;
    int total_moves_p0 = 0;
    double total_time_p1 = 0.0;
    int total_moves_p1 = 0;

    std::ifstream stats_file("move_stats.txt");
    if (stats_file.good())
    {
        stats_file >> total_time_p0 >> total_moves_p0 >> total_time_p1 >> total_moves_p1;
        stats_file.close();
    }

    // Add timing code
    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);

    // Select strategy based on the current player
    if (_current_player == 0)
    {
        alpha_beta::computeBestMoveWithScore(*this);
    }
    else if (_current_player == 1)
    {
        alpha_beta_para::computeBestMoveWithScore(*this);
    }

    // Calculate elapsed time
    gettimeofday(&end_time, NULL);
    double elapsed = ((end_time.tv_sec - start_time.tv_sec) * 1000.0) +
                     ((end_time.tv_usec - start_time.tv_usec) / 1000.0);

    // Update metrics for the specific player
    if (_current_player == 0)
    {
        total_time_p0 += elapsed;
        total_moves_p0++;
        double avg_time = total_time_p0 / total_moves_p0;
        std::cout << "Player 0 (Red) move took " << elapsed << " ms" << std::endl;
        std::cout << "Player 0 average time: " << avg_time << " ms"
                  << " (over " << total_moves_p0 << " moves)" << std::endl;
    }
    else
    {
        total_time_p1 += elapsed;
        total_moves_p1++;
        double avg_time = total_time_p1 / total_moves_p1;
        std::cout << "Player 1 (Blue) move took " << elapsed << " ms" << std::endl;
        std::cout << "Player 1 average time: " << avg_time << " ms"
                  << " (over " << total_moves_p1 << " moves)" << std::endl;
    }

    // Save updated statistics for both players
    std::ofstream stats_out("move_stats.txt");
    stats_out << total_time_p0 << " " << total_moves_p0 << " "
              << total_time_p1 << " " << total_moves_p1;
    stats_out.close();
}

// Add this function to detect game phase
Strategy::GamePhase Strategy::detectGamePhase() const
{
    // Count total number of blobs on board
    int total_blobs = 0;
    int empty_spaces = 0;

    for (int x = 0; x < 8; x++)
    {
        for (int y = 0; y < 8; y++)
        {
            if (_blobs.get(x, y) != -1)
            {
                total_blobs++;
            }
            else if (!_holes.get(x, y))
            {
                empty_spaces++;
            }
        }
    }

    // Determine phase based on board occupation
    if (total_blobs <= 16)
    {
        return EARLY_GAME;
    }
    else if (total_blobs >= 45)
    {
        return LATE_GAME;
    }
    else
    {
        return MID_GAME;
    }
}