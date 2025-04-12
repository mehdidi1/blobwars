#include "minimax.h"
#include <limits>
#include <tbb/tbb.h>
#include <tbb/parallel_for.h>
#include <tbb/mutex.h>
#include <atomic>

namespace minimax_para {

    // Depth limit for the Minimax algorithm
    const int MAX_DEPTH = 2;

    // Minimax recursive function
    Sint32 minimax(Strategy& strategy, int depth, bool maximizingPlayer,Sint32 root_player) {
        // Base case: if we reach the maximum depth or no valid moves are left
        vector<movement> valid_moves;
        strategy.computeValidMoves(valid_moves);
        if (depth == 0 || valid_moves.empty()) {
            return strategy.estimateCurrentScore(root_player);
        }

        // Maximizing player
        if (maximizingPlayer) {
            Sint32 maxEval = std::numeric_limits<Sint32>::min();
            for (const movement& mv : valid_moves) {
                Strategy sim_strategy(strategy); // Simulate the move
                sim_strategy.applyMove(mv);
                Sint32 eval = minimax(sim_strategy, depth - 1, false,root_player);
                maxEval = std::max(maxEval, eval);
            }
            return maxEval;
        }
        // Minimizing player
        else {
            Sint32 minEval = std::numeric_limits<Sint32>::max();
            for (const movement& mv : valid_moves) {
                Strategy sim_strategy(strategy); // Simulate the move
                sim_strategy.applyMove(mv);
                Sint32 eval = minimax(sim_strategy, depth - 1, true,root_player);
                minEval = std::min(minEval, eval);
            }
            return minEval;
        }
    }

    // Compute the best move using Minimax
    void computeBestMoveWithScore(Strategy& strategy) {
        movement best_move(0, 0, 0, 0);
        std::atomic<Sint32> best_score{std::numeric_limits<Sint32>::min()};
        vector<movement> valid_moves;
        tbb::mutex best_move_mutex;

        // Get all valid moves
        strategy.computeValidMoves(valid_moves);

        // If no valid moves, return an empty move
        if (valid_moves.empty()) {
            strategy._saveBestMove(best_move);
            std::cout << "No valid move available" << endl;
            return;
        }

        //Save a random move in case we don't finish computing
        if (!valid_moves.empty()) {
            srand(time(0)); // Seed for randomness
            int random_index = rand() % valid_moves.size();
            strategy._saveBestMove(valid_moves[random_index]);
        }

        // Evaluate each move in parallel
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, valid_moves.size()),
            [&](const tbb::blocked_range<size_t>& range) {
                for (size_t i = range.begin(); i < range.end(); ++i) {
                    const movement& mv = valid_moves[i];
                    Strategy sim_strategy(strategy); // Simulate the move
                    sim_strategy.applyMove(mv);

                    // Call Minimax for the opponent's turn
                    Sint32 score = minimax(sim_strategy, MAX_DEPTH - 1, false, strategy._current_player);

                    // Keep track of the best move
                    if (score > best_score) {
                        tbb::mutex::scoped_lock lock(best_move_mutex);
                        if (score > best_score) {
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