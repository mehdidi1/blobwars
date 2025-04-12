#include "strategy.h"
#include "greedy.h"
#include <sys/time.h>
#include "minimax.h"
#include "minimax_para.h"

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
    _current_player =  !_current_player;
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

    if (player == 1){
        return player1_score - player0_score;
    }

    return player0_score - player1_score;
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

    // Add timing code
    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);

    // Select strategy based on the current player
    if (_current_player == 0)
    {
        minimax::computeBestMoveWithScore(*this);
    }
    else if (_current_player == 1)
    {
        minimax_para::computeBestMoveWithScore(*this);
    }
    // Calculate and display elapsed time
    gettimeofday(&end_time, NULL);
    double elapsed = ((end_time.tv_sec - start_time.tv_sec) * 1000.0) +
                     ((end_time.tv_usec - start_time.tv_usec) / 1000.0);
    std::cout << "Best move computation took " << elapsed << " milliseconds" << std::endl;
}