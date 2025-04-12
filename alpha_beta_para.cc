#include "alpha_beta_para.h"
#include <limits>
#include <tbb/tbb.h>
#include <tbb/parallel_for.h>
#include <tbb/spin_mutex.h>
#include <atomic>
#include <cstdlib>
#include <ctime>
#include <iostream>

namespace alpha_beta_para {

    // Depth limit for the Alpha-Beta algorithm
    const int MAX_DEPTH = 4;

    // Alpha-Beta pruning recursive function
    Sint32 alphabeta(Strategy& strategy, int depth, Sint32 alpha, Sint32 beta, bool maximizingPlayer, Sint32 root_player) {
        // Base case: if we reach the maximum depth or no valid moves are left
        vector<movement> valid_moves;
        strategy.computeValidMoves(valid_moves);
        if (depth == 0 || valid_moves.empty()) {
            return strategy.estimateCurrentScore(root_player);
        }

        // Maximizing player
        if (maximizingPlayer) {
            Sint32 value = std::numeric_limits<Sint32>::min();
            for (const movement& mv : valid_moves) {
                Strategy sim_strategy(strategy); // Simulate the move
                sim_strategy.applyMove(mv);
                value = std::max(value, alphabeta(sim_strategy, depth - 1, alpha, beta, false, root_player));
                alpha = std::max(alpha, value);
                if (alpha >= beta) {
                    // Beta cut-off
                    break;
                }
            }
            return value;
        }
        // Minimizing player
        else {
            Sint32 value = std::numeric_limits<Sint32>::max();
            for (const movement& mv : valid_moves) {
                Strategy sim_strategy(strategy); // Simulate the move
                sim_strategy.applyMove(mv);
                value = std::min(value, alphabeta(sim_strategy, depth - 1, alpha, beta, true, root_player));
                beta = std::min(beta, value);
                if (beta <= alpha) {
                    // Alpha cut-off
                    break;
                }
            }
            return value;
        }
    }

    // Compute the best move using parallel Alpha-Beta pruning
    void computeBestMoveWithScore(Strategy& strategy) {
        movement best_move(0, 0, 0, 0);
        std::atomic<Sint32> best_score{std::numeric_limits<Sint32>::min()};
        vector<movement> valid_moves;
        tbb::spin_mutex best_move_mutex;

        // Get all valid moves
        strategy.computeValidMoves(valid_moves);

        // If no valid moves, return an empty move
        if (valid_moves.empty()) {
            strategy._saveBestMove(best_move);
            std::cout << "No valid move available" << std::endl;
            return;
        }

        // Save a random move as fallback in case computation is interrupted
        if (!valid_moves.empty()) {
            srand(time(0)); // Seed for randomness
            int random_index = rand() % valid_moves.size();
            strategy._saveBestMove(valid_moves[random_index]);
        }

        // Parallel evaluation of all moves using TBB
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, valid_moves.size()),
            [&](const tbb::blocked_range<size_t>& range) {
                for (size_t i = range.begin(); i < range.end(); ++i) {
                    const movement& mv = valid_moves[i];
                    Strategy sim_strategy(strategy); // Simulate the move
                    sim_strategy.applyMove(mv);

                    // Call Alpha-Beta for the opponent's turn
                    Sint32 score = alphabeta(sim_strategy, MAX_DEPTH - 1, 
                                        std::numeric_limits<Sint32>::min(), 
                                        std::numeric_limits<Sint32>::max(), 
                                        false, strategy._current_player);

                    // Keep track of the best move
                    if (score > best_score) {
                        tbb::spin_mutex::scoped_lock lock(best_move_mutex);
                        if (score > best_score) {  // Double-check after obtaining lock
                            best_score = score;
                            best_move = mv;
                        }
                    }
                }
            }
        );

        // Save the best move
        strategy._saveBestMove(best_move);
    }
}