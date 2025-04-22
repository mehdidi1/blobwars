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
    const int MAX_DEPTH = 10;


    // Store the best move found in the previous iteration of iterative deepening
    movement previous_iteration_best_move(0, 0, 0, 0);
    bool has_previous_iteration_best = false;


    // Sort moves based ONLY on the best move from the previous iterative deepening depth.
    void order_moves(vector<movement>& moves) {
        // Check if we have a best move from the previous iteration
        if (has_previous_iteration_best) {
            // Find the previous best move in the current list
            for (size_t i = 0; i < moves.size(); ++i) {
                if (moves[i].ox == previous_iteration_best_move.ox &&
                    moves[i].oy == previous_iteration_best_move.oy &&
                    moves[i].nx == previous_iteration_best_move.nx &&
                    moves[i].ny == previous_iteration_best_move.ny)
                {
                    // If found and not already at the front, swap it to the front
                    if (i > 0) {
                        std::swap(moves[0], moves[i]);
                    }
                    // No further sorting needed based on history, so we can break
                    break;
                }
            }
        }
    }

    Sint32 alphabeta(Strategy &strategy, int depth, Sint32 alpha, Sint32 beta, bool maximizingPlayer, Sint32 root_player)
    {
        // Base case: if we reach the maximum depth or no valid moves are left
        vector<movement> valid_moves;
        strategy.computeValidMoves(valid_moves);
        if (depth == 0 || valid_moves.empty())
        {
            return strategy.estimateCurrentScore(root_player);
        }

        // Order moves based *only* on previous iteration's best move
        order_moves(valid_moves);

        // Maximizing player
        if (maximizingPlayer)
        {
            Sint32 value = std::numeric_limits<Sint32>::min();
            for (const movement &mv : valid_moves)
            {
                Strategy sim_strategy(strategy); // Simulate the move
                sim_strategy.applyMove(mv);
                Sint32 child_value = alphabeta(sim_strategy, depth - 1, alpha, beta, false, root_player);
                value = std::max(value, child_value);
                alpha = std::max(alpha, value);

                if (alpha >= beta)
                {
                    // Beta cut-off - update_history call removed
                    break; // Prune
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
                Sint32 child_value = alphabeta(sim_strategy, depth - 1, alpha, beta, true, root_player);
                value = std::min(value, child_value);
                beta = std::min(beta, value);

                if (beta <= alpha)
                {
                    // Alpha cut-off - update_history call removed
                    break; // Prune
                }
            }
            return value;
        }
    }

    // Compute the best move using iterative deepening with simplified move ordering
    void computeBestMoveWithScore(Strategy &strategy)
    {
        movement best_move(0, 0, 0, 0);
        vector<movement> valid_moves;

        // Get all valid moves for the root node
        strategy.computeValidMoves(valid_moves);

        if (valid_moves.empty())
        {
            strategy._saveBestMove(best_move);
            std::cout << "No valid move available" << std::endl;
            return;
        }

        // Reset previous best move flag for the new turn
        has_previous_iteration_best = false;

        // Iterative deepening loop
        for (int current_depth = MIN_DEPTH; current_depth <= MAX_DEPTH; ++current_depth)
        {
            std::cout << "Searching at depth " << current_depth << std::endl;

            std::atomic<Sint32> current_best_score{std::numeric_limits<Sint32>::min()};
            movement current_depth_best_move(0,0,0,0); // Best move found at this depth
            tbb::spin_mutex best_move_mutex;

            // Order root moves based *only* on previous iteration's best move
            order_moves(valid_moves); // Order moves for the root level search

            // Parallel evaluation of root moves
            tbb::parallel_for(
                tbb::blocked_range<size_t>(0, valid_moves.size()),
                [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t i = range.begin(); i != range.end(); ++i) {
                        const movement& mv = valid_moves[i];
                        Strategy sim_strategy(strategy);
                        sim_strategy.applyMove(mv);

                        // Call Alpha-Beta for the opponent's turn (minimizing node)
                        Sint32 score = alphabeta(sim_strategy, current_depth - 1,
                                                 std::numeric_limits<Sint32>::min(),
                                                 std::numeric_limits<Sint32>::max(),
                                                 false, // Opponent is minimizing player
                                                 strategy._current_player); // Root player perspective

                        // Update the best score and move found so far at this depth
                        Sint32 local_best_score = current_best_score.load();
                        if (score > local_best_score) {
                            tbb::spin_mutex::scoped_lock lock(best_move_mutex);
                            // Double-check after acquiring the lock
                            if (score > current_best_score) {
                                current_best_score = score;
                                current_depth_best_move = mv;
                            }
                        }
                    }
                }
            );

            // Update the overall best move found across all depths
            best_move = current_depth_best_move;
            strategy._saveBestMove(best_move); // Save the best move found at this depth

            // Store the best move found at this depth to be used for ordering in the next iteration
            previous_iteration_best_move = best_move;
            // Ensure the flag is set only if a valid move was actually found (handle edge case of no moves)
            has_previous_iteration_best = !(best_move.ox == 0 && best_move.oy == 0 && best_move.nx == 0 && best_move.ny == 0);


            std::cout << "Depth " << current_depth << " completed. Best score: "
                      << current_best_score.load() << std::endl;

        }

    }
}