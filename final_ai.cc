#include "final_ai.h"
#include <limits>
#include <tbb/tbb.h>
#include <tbb/parallel_for.h>
#include <tbb/spin_mutex.h>
#include <atomic>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <sys/time.h>

namespace final_ai
{

    // Minimum depth to guarantee
    const int MIN_DEPTH = 3;
    const int MAX_DEPTH = 4;

    Sint32 alphabeta(Strategy &strategy, int depth, Sint32 alpha, Sint32 beta, bool maximizingPlayer, Sint32 root_player)
    {
        // Base case: if we reach the maximum depth or no valid moves are left
        vector<movement> valid_moves;
        strategy.computeValidMoves(valid_moves);
        if (depth == 0 || valid_moves.empty())
        {
            return strategy.estimateCurrentScore(root_player);
        }

        // Maximizing player
        if (maximizingPlayer)
        {
            Sint32 value = std::numeric_limits<Sint32>::min();
            for (const movement &mv : valid_moves)
            {
                Strategy sim_strategy(strategy); // Simulate the move
                sim_strategy.applyMove(mv);
                value = std::max(value, alphabeta(sim_strategy, depth - 1, alpha, beta, false, root_player));
                alpha = std::max(alpha, value);
                if (alpha >= beta)
                {
                    // Beta cut-off
                    break;
                }
            }
            return value;
        }
        // Minimizing player
        else
        {
            Sint32 value = std::numeric_limits<Sint32>::max();
            for (const movement &mv : valid_moves)
            {
                Strategy sim_strategy(strategy); // Simulate the move
                sim_strategy.applyMove(mv);
                value = std::min(value, alphabeta(sim_strategy, depth - 1, alpha, beta, true, root_player));
                beta = std::min(beta, value);
                if (beta <= alpha)
                {
                    // Alpha cut-off
                    break;
                }
            }
            return value;
        }
    }

    // Compute the best move using iterative deepening with guaranteed minimum depth
    void computeBestMoveWithScore(Strategy &strategy)
    {
        movement best_move(0, 0, 0, 0);
        vector<movement> valid_moves;

        // Get all valid moves
        strategy.computeValidMoves(valid_moves);

        // If no valid moves, return an empty move
        if (valid_moves.empty())
        {
            strategy._saveBestMove(best_move);
            std::cout << "No valid move available" << std::endl;
            return;
        }



        // First guarantee the minimum depth search
        int starting_depth = MIN_DEPTH;

        // Iterative deepening
        for (int current_depth = starting_depth; current_depth <= MAX_DEPTH; current_depth++)
        {
            std::cout << "Searching at depth " << current_depth << std::endl;

            std::atomic<Sint32> best_score{std::numeric_limits<Sint32>::min()};
            movement depth_best_move = best_move; // Start with previous best
            tbb::spin_mutex best_move_mutex;

            // Parallel evaluation at this depth
            tbb::parallel_for(
                tbb::blocked_range<size_t>(0, valid_moves.size()),
                [&](const tbb::blocked_range<size_t> &range)
                {
                    for (size_t i = range.begin(); i < range.end(); ++i)
                    {

                        const movement &mv = valid_moves[i];
                        Strategy sim_strategy(strategy);
                        sim_strategy.applyMove(mv);

                        // Call Alpha-Beta for the opponent's turn
                        Sint32 score = alphabeta(sim_strategy, current_depth - 1,
                                                 std::numeric_limits<Sint32>::min(),
                                                 std::numeric_limits<Sint32>::max(),
                                                 false, strategy._current_player);

                        // Keep track of the best move
                        if (score > best_score)
                        {
                            tbb::spin_mutex::scoped_lock lock(best_move_mutex);
                            if (score > best_score)
                            { // Double-check after obtaining lock
                                best_score = score;
                                depth_best_move = mv;
                            }
                        }
                    }
                });

            // Only update the best move if this iteration completed
            best_move = depth_best_move;
            strategy._saveBestMove(best_move);
            std::cout << "Depth " << current_depth << " completed. Best score: "
                      << best_score.load() << std::endl;
        }
    }
}