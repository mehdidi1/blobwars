#include "test.h"
#include <limits>
#include <tbb/tbb.h>
#include <tbb/parallel_for.h>
#include <tbb/spin_mutex.h>
#include <atomic>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <sys/time.h>
#include <unordered_map>
#include <string>
#include <algorithm>

namespace test {

    // Minimum depth to guarantee
    const int MIN_DEPTH = 1;
    const int MAX_DEPTH = 10;

    // Move history table to track move quality across iterations
    std::unordered_map<std::string, int> move_history;

    // Create a unique key for a move
    std::string create_move_key(const movement& mv) {
        return std::to_string(mv.ox) + "," + 
               std::to_string(mv.oy) + "," + 
               std::to_string(mv.nx) + "," + 
               std::to_string(mv.ny);
    }

    // Update history table when a move causes a cutoff
    void update_history(const movement& mv, int depth) {
        std::string key = create_move_key(mv);
        move_history[key] += depth * depth; // Square the depth for greater impact on deeper searches
    }

    // Sort moves based on history scores
    void order_moves(vector<movement>& moves) {
        std::sort(moves.begin(), moves.end(), [](const movement& a, const movement& b) {
            std::string key_a = create_move_key(a);
            std::string key_b = create_move_key(b);
            
            // If move doesn't exist in history, default to 0
            int score_a = move_history.count(key_a) ? move_history[key_a] : 0;
            int score_b = move_history.count(key_b) ? move_history[key_b] : 0;
            
            return score_a > score_b;
        });
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

        // Order moves based on historical performance
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
                    // Beta cut-off - update history as this move was good enough to cause a cutoff
                    update_history(mv, depth);
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
                Sint32 child_value = alphabeta(sim_strategy, depth - 1, alpha, beta, true, root_player);
                value = std::min(value, child_value);
                beta = std::min(beta, value);
                
                if (beta <= alpha)
                {
                    // Alpha cut-off - update history as this move was good enough to cause a cutoff
                    update_history(mv, depth);
                    break;
                }
            }
            return value;
        }
    }

    // Compute the best move using iterative deepening with move ordering
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
            
            // Order moves based on previous iterations results
            order_moves(valid_moves);

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
                                
                                // Update history for best moves at root level
                                update_history(mv, current_depth * 2); // Extra weight for best moves
                            }
                        }
                    }
                });

            // Update the best move for this depth
            best_move = depth_best_move;
            strategy._saveBestMove(best_move);
            std::cout << "Depth " << current_depth << " completed. Best score: "
                      << best_score.load() << std::endl;
        }
    }
}