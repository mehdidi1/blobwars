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
#include "ga_weight_optimizer.h"

// Initialize static member
ChromosomeWeights* Strategy::evaluation_weights = nullptr;

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
    Sint32 my_position_score = 0, opp_position_score = 0;
    Sint32 my_potential_captures = 0;         // Opponent blobs adjacent to mine
    Sint32 my_frontier = 0, opp_frontier = 0; // Blobs adjacent to empty squares

    // --- Positional Weights (Center and Edges) ---
    // Higher values for center, moderate for edges, low for near-corners
    static const int position_weights[8][8] = {
        {8, -1, 6, 4, 4, 6, -1, 8}, // Corners are handled separately
        {-1, -1, 0, 1, 1, 0, -1, -1},
        {6, 0, 2, 3, 3, 2, 0, 6},
        {4, 1, 3, 4, 4, 3, 1, 4},
        {4, 1, 3, 4, 4, 3, 1, 4},
        {6, 0, 2, 3, 3, 2, 0, 6},
        {-1, -1, 0, 1, 1, 0, -1, -1},
        {8, -1, 6, 4, 4, 6, -1, 8}};

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
                my_position_score += position_weights[y][x];

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
                opp_position_score += position_weights[y][x];

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

    // 4. Positional Score (using weights table) - Lower weight
    Sint32 positional_score = 10 * (my_position_score - opp_position_score);

    // 5. Potential Capture Score - Lower weight
    Sint32 capture_score = 30 * my_potential_captures; // Only count ours for simplicity

    // 6. Frontier Score (Penalty for vulnerable blobs) - Lower negative weight
    Sint32 frontier_score = -50 * (my_frontier - opp_frontier);

    // Determine current game phase
    GamePhase phase = detectGamePhase();

    // Phase-specific weights
    Sint32 material_weight, corner_weight, mobility_weight;
    Sint32 positional_weight, capture_weight, frontier_weight;

    // Use optimization weights if available
    if (Strategy::evaluation_weights != nullptr) {
        switch (phase) {
        case EARLY_GAME:
            material_weight = Strategy::evaluation_weights->early_game.material_weight;
            corner_weight = Strategy::evaluation_weights->early_game.corner_weight;
            mobility_weight = Strategy::evaluation_weights->early_game.mobility_weight;
            positional_weight = Strategy::evaluation_weights->early_game.positional_weight;
            capture_weight = Strategy::evaluation_weights->early_game.capture_weight;
            frontier_weight = Strategy::evaluation_weights->early_game.frontier_weight;
            break;
            
        case MID_GAME:
            material_weight = Strategy::evaluation_weights->mid_game.material_weight;
            corner_weight = Strategy::evaluation_weights->mid_game.corner_weight;
            mobility_weight = Strategy::evaluation_weights->mid_game.mobility_weight;
            positional_weight = Strategy::evaluation_weights->mid_game.positional_weight;
            capture_weight = Strategy::evaluation_weights->mid_game.capture_weight;
            frontier_weight = Strategy::evaluation_weights->mid_game.frontier_weight;
            break;
            
        case LATE_GAME:
            material_weight = Strategy::evaluation_weights->late_game.material_weight;
            corner_weight = Strategy::evaluation_weights->late_game.corner_weight;
            mobility_weight = Strategy::evaluation_weights->late_game.mobility_weight;
            positional_weight = Strategy::evaluation_weights->late_game.positional_weight;
            capture_weight = Strategy::evaluation_weights->late_game.capture_weight;
            frontier_weight = Strategy::evaluation_weights->late_game.frontier_weight;
            break;
        }
    } else {
        // Use default weights (your existing code)
        switch (phase) {
        case EARLY_GAME:
            material_weight = 80;
            corner_weight = 120; // Higher emphasis on corners
            mobility_weight = 15;
            positional_weight = 20; // Higher emphasis on position
            capture_weight = 20;
            frontier_weight = -30; // Less penalty for frontiers early
            break;

        case MID_GAME:
            // Mid game: emphasize mobility and potential captures
            material_weight = 100;
            corner_weight = 70;
            mobility_weight = 30; // Higher emphasis on mobility
            positional_weight = 10;
            capture_weight = 40; // Higher emphasis on potential captures
            frontier_weight = -40;
            break;

        case LATE_GAME:
            // Late game: emphasize material count and reduce mobility importance
            material_weight = 120; // Higher emphasis on material
            corner_weight = 60;
            mobility_weight = 10; // Less emphasis on mobility
            positional_weight = 5;
            capture_weight = 20;
            frontier_weight = -50; // More penalty for vulnerable blobs
            break;
        }
    }

    // Apply phase-specific weights to scores
    material_score = material_weight * (my_blobs - opp_blobs);
    if (my_blobs == 0)
        return -99999; // Loss condition
    if (opp_blobs == 0)
        return 99999; // Win condition

    corner_score = corner_weight * (my_corner_blobs - opp_corner_blobs);
    mobility_score = mobility_weight * (my_moves - opp_moves);
    positional_score = positional_weight * (my_position_score - opp_position_score);
    capture_score = capture_weight * my_potential_captures;
    frontier_score = frontier_weight * (my_frontier - opp_frontier);

    // --- Combine Scores ---
    Sint32 total_score = material_score + corner_score + mobility_score + positional_score + capture_score + frontier_score + recognizePatterns(player);

    return total_score;
}

// Add these pattern recognition functions after your existing evaluation functions

// Pattern recognition scoring function
Sint32 Strategy::recognizePatterns(Sint32 player) const
{
    Sint32 opponent = 1 - player;
    Sint32 total_pattern_score = 0;

    // Pattern weights
    Sint32 FORTRESS_WEIGHT, WALL_WEIGHT, INVASION_PLATFORM_WEIGHT, 
           PINCER_WEIGHT, EXPANSION_HUB_WEIGHT;
           
    if (Strategy::evaluation_weights != nullptr) {
        FORTRESS_WEIGHT = Strategy::evaluation_weights->fortress_weight;
        WALL_WEIGHT = Strategy::evaluation_weights->wall_weight;
        INVASION_PLATFORM_WEIGHT = Strategy::evaluation_weights->invasion_platform_weight;
        PINCER_WEIGHT = Strategy::evaluation_weights->pincer_weight;
        EXPANSION_HUB_WEIGHT = Strategy::evaluation_weights->expansion_hub_weight;
    } else {
        // Default values
        FORTRESS_WEIGHT = 40;
        WALL_WEIGHT = 25;
        INVASION_PLATFORM_WEIGHT = 30;
        PINCER_WEIGHT = 35;
        EXPANSION_HUB_WEIGHT = 20;
    }

    // --- Detect Fortress Formations (2x2 blocks or L-shapes) ---
    Sint32 my_fortress = 0, opp_fortress = 0;
    for (int x = 0; x < 7; x++)
    {
        for (int y = 0; y < 7; y++)
        {
            // Check for 2x2
            int my_count = 0, opp_count = 0;
            for (int dx = 0; dx <= 1; dx++)
            {
                for (int dy = 0; dy <= 1; dy++)
                {
                    if (_blobs.get(x + dx, y + dy) == player)
                        my_count++;
                    else if (_blobs.get(x + dx, y + dy) == opponent)
                        opp_count++;
                }
            }
            if (my_count >= 3)
                my_fortress++; // 3 or 4 blobs in a 2x2 area
            if (opp_count >= 3)
                opp_fortress++;

            // Check for L-shapes (separate from the 2x2 check)
            if (x < 6 && y < 6)
            {
                // L-shape pattern 1 (for player)
                if (_blobs.get(x, y) == player &&
                    _blobs.get(x, y + 1) == player &&
                    _blobs.get(x + 1, y) == player)
                    my_fortress++;

                // L-shape pattern 2 (for player)
                if (_blobs.get(x, y) == player &&
                    _blobs.get(x + 1, y) == player &&
                    _blobs.get(x + 1, y + 1) == player)
                    my_fortress++;

                // Similar checks for opponent L-shapes
                if (_blobs.get(x, y) == opponent &&
                    _blobs.get(x, y + 1) == opponent &&
                    _blobs.get(x + 1, y) == opponent)
                    opp_fortress++;

                if (_blobs.get(x, y) == opponent &&
                    _blobs.get(x + 1, y) == opponent &&
                    _blobs.get(x + 1, y + 1) == opponent)
                    opp_fortress++;
            }
        }
    }
    total_pattern_score += FORTRESS_WEIGHT * (my_fortress - opp_fortress);

    // --- Detect Wall Formations ---
    Sint32 my_walls = 0, opp_walls = 0;
    // Horizontal walls
    for (int y = 0; y < 8; y++)
    {
        int consecutive_mine = 0;
        int consecutive_opp = 0;
        for (int x = 0; x < 8; x++)
        {
            if (_blobs.get(x, y) == player)
            {
                consecutive_mine++;
                consecutive_opp = 0;
                if (consecutive_mine >= 3)
                    my_walls++; // 3+ consecutive blobs form a wall
            }
            else if (_blobs.get(x, y) == opponent)
            {
                consecutive_opp++;
                consecutive_mine = 0;
                if (consecutive_opp >= 3)
                    opp_walls++;
            }
            else
            {
                consecutive_mine = consecutive_opp = 0;
            }
        }
    }
    // Vertical walls (similar logic)
    for (int x = 0; x < 8; x++)
    {
        int consecutive_mine = 0;
        int consecutive_opp = 0;
        for (int y = 0; y < 8; y++)
        {
            if (_blobs.get(x, y) == player)
            {
                consecutive_mine++;
                consecutive_opp = 0;
                if (consecutive_mine >= 3)
                    my_walls++;
            }
            else if (_blobs.get(x, y) == opponent)
            {
                consecutive_opp++;
                consecutive_mine = 0;
                if (consecutive_opp >= 3)
                    opp_walls++;
            }
            else
            {
                consecutive_mine = consecutive_opp = 0;
            }
        }
    }
    total_pattern_score += WALL_WEIGHT * (my_walls - opp_walls);

    // --- Detect Invasion Platforms ---
    Sint32 my_invasion_platforms = 0, opp_invasion_platforms = 0;
    for (int x = 0; x < 8; x++)
    {
        for (int y = 0; y < 8; y++)
        {
            if (_blobs.get(x, y) == player)
            {
                int potential_captures = 0;
                // Check positions at exactly distance 2
                for (int dx = -2; dx <= 2; dx++)
                {
                    for (int dy = -2; dy <= 2; dy++)
                    {
                        // Skip if not exactly distance 2
                        if (std::max(abs(dx), abs(dy)) != 2)
                            continue;

                        int nx = x + dx, ny = y + dy;
                        // Check bounds
                        if (nx < 0 || nx >= 8 || ny < 0 || ny >= 8)
                            continue;

                        // Check if position is empty and has opponent blobs adjacent
                        if (_blobs.get(nx, ny) == -1 && !_holes.get(nx, ny))
                        {
                            int adjacent_opponents = 0;
                            for (int adx = -1; adx <= 1; adx++)
                            {
                                for (int ady = -1; ady <= 1; ady++)
                                {
                                    if (adx == 0 && ady == 0)
                                        continue;
                                    int ax = nx + adx, ay = ny + ady;
                                    if (ax >= 0 && ax < 8 && ay >= 0 && ay < 8)
                                    {
                                        if (_blobs.get(ax, ay) == opponent)
                                            adjacent_opponents++;
                                    }
                                }
                            }
                            if (adjacent_opponents >= 2)
                                potential_captures++;
                        }
                    }
                }
                if (potential_captures > 0)
                    my_invasion_platforms++;
            }
            else if (_blobs.get(x, y) == opponent)
            {
                // Similar check for opponent invasion platforms
                int potential_captures = 0;
                // Check positions at exactly distance 2
                for (int dx = -2; dx <= 2; dx++)
                {
                    for (int dy = -2; dy <= 2; dy++)
                    {
                        if (std::max(abs(dx), abs(dy)) != 2)
                            continue;

                        int nx = x + dx, ny = y + dy;
                        if (nx < 0 || nx >= 8 || ny < 0 || ny >= 8)
                            continue;

                        if (_blobs.get(nx, ny) == -1 && !_holes.get(nx, ny))
                        {
                            int adjacent_mine = 0;
                            for (int adx = -1; adx <= 1; adx++)
                            {
                                for (int ady = -1; ady <= 1; ady++)
                                {
                                    if (adx == 0 && ady == 0)
                                        continue;
                                    int ax = nx + adx, ay = ny + ady;
                                    if (ax >= 0 && ax < 8 && ay >= 0 && ay < 8)
                                    {
                                        if (_blobs.get(ax, ay) == player)
                                            adjacent_mine++;
                                    }
                                }
                            }
                            if (adjacent_mine >= 2)
                                potential_captures++;
                        }
                    }
                }
                if (potential_captures > 0)
                    opp_invasion_platforms++;
            }
        }
    }
    total_pattern_score += INVASION_PLATFORM_WEIGHT * (my_invasion_platforms - opp_invasion_platforms);

    // --- Detect Pincer Patterns ---
    Sint32 my_pincers = 0, opp_pincers = 0;
    // Check horizontal pincers
    for (int y = 0; y < 8; y++)
    {
        for (int x = 0; x < 4; x++)
        { // Changed from x < 6 to x < 4
            // Check for pattern: [player][empty][opponent][empty][player]
            if (x + 4 < 8 && // Added bounds check
                _blobs.get(x, y) == player &&
                _blobs.get(x + 1, y) == -1 && !_holes.get(x + 1, y) &&
                _blobs.get(x + 2, y) == opponent &&
                _blobs.get(x + 3, y) == -1 && !_holes.get(x + 3, y) &&
                _blobs.get(x + 4, y) == player)
            {
                my_pincers++;
            }
            // Check for pattern: [opponent][empty][player][empty][opponent]
            if (x + 4 < 8 && // Added bounds check
                _blobs.get(x, y) == opponent &&
                _blobs.get(x + 1, y) == -1 && !_holes.get(x + 1, y) &&
                _blobs.get(x + 2, y) == player &&
                _blobs.get(x + 3, y) == -1 && !_holes.get(x + 3, y) &&
                _blobs.get(x + 4, y) == opponent)
            {
                opp_pincers++;
            }
        }
    }
    // Check vertical pincers (similar logic)
    for (int x = 0; x < 8; x++)
    {
        for (int y = 0; y < 4; y++)
        {                    // Changed from y < 6 to y < 4
            if (y + 4 < 8 && // Added bounds check
                _blobs.get(x, y) == player &&
                _blobs.get(x, y + 1) == -1 && !_holes.get(x, y + 1) &&
                _blobs.get(x, y + 2) == opponent &&
                _blobs.get(x, y + 3) == -1 && !_holes.get(x, y + 3) &&
                _blobs.get(x, y + 4) == player)
            {
                my_pincers++;
            }
            if (y + 4 < 8 && // Added bounds check
                _blobs.get(x, y) == opponent &&
                _blobs.get(x, y + 1) == -1 && !_holes.get(x, y + 1) &&
                _blobs.get(x, y + 2) == player &&
                _blobs.get(x, y + 3) == -1 && !_holes.get(x, y + 3) &&
                _blobs.get(x, y + 4) == opponent)
            {
                opp_pincers++;
            }
        }
    }
    total_pattern_score += PINCER_WEIGHT * (my_pincers - opp_pincers);

    // --- Count Expansion Hubs ---
    Sint32 my_hubs = 0, opp_hubs = 0;
    for (int x = 0; x < 8; x++)
    {
        for (int y = 0; y < 8; y++)
        {
            if (_blobs.get(x, y) == player)
            {
                int empty_adjacent = 0;
                for (int dx = -1; dx <= 1; dx++)
                {
                    for (int dy = -1; dy <= 1; dy++)
                    {
                        if (dx == 0 && dy == 0)
                            continue;
                        int nx = x + dx, ny = y + dy;
                        if (nx >= 0 && nx < 8 && ny >= 0 && ny < 8)
                        {
                            if (_blobs.get(nx, ny) == -1 && !_holes.get(nx, ny))
                                empty_adjacent++;
                        }
                    }
                }
                if (empty_adjacent >= 3)
                    my_hubs++; // 3+ empty adjacent squares
            }
            else if (_blobs.get(x, y) == opponent)
            {
                int empty_adjacent = 0;
                for (int dx = -1; dx <= 1; dx++)
                {
                    for (int dy = -1; dy <= 1; dy++)
                    {
                        if (dx == 0 && dy == 0)
                            continue;
                        int nx = x + dx, ny = y + dy;
                        if (nx >= 0 && nx < 8 && ny >= 0 && ny < 8)
                        {
                            if (_blobs.get(nx, ny) == -1 && !_holes.get(nx, ny))
                                empty_adjacent++;
                        }
                    }
                }
                if (empty_adjacent >= 3)
                    opp_hubs++;
            }
        }
    }
    total_pattern_score += EXPANSION_HUB_WEIGHT * (my_hubs - opp_hubs);

    return total_pattern_score;
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
        alpha_beta::computeBestMoveWithScore(*this);
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
GamePhase Strategy::detectGamePhase() const
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

bool Strategy::useOptimizedWeights(ChromosomeWeights* weights) {
    // Validate pointer
    if (weights == nullptr) {
        std::cerr << "Warning: Attempting to set null weights" << std::endl;
        Strategy::evaluation_weights = nullptr;
        return false;
    }
    
    // Set weights
    Strategy::evaluation_weights = weights;
    return true;
}