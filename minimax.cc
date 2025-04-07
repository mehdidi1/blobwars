#include "minimax.h"
#include <limits>

namespace minimax {

    // Depth limit for the Minimax algorithm
    const int MAX_DEPTH = 2;

    // Minimax recursive function
    Sint32 minimax(Strategy& strategy, int depth, bool maximizingPlayer) {
        // Base case: if we reach the maximum depth or no valid moves are left
        vector<movement> valid_moves;
        strategy.computeValidMoves(valid_moves);
        if (depth == 0 || valid_moves.empty()) {
            return strategy.estimateCurrentScore();
        }

        // Maximizing player
        if (maximizingPlayer) {
            Sint32 maxEval = std::numeric_limits<Sint32>::min();
            for (const movement& mv : valid_moves) {
                Strategy sim_strategy(strategy); // Simulate the move
                sim_strategy.applyMove(mv);
                Sint32 eval = minimax(sim_strategy, depth - 1, false);
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
                Sint32 eval = minimax(sim_strategy, depth - 1, true);
                minEval = std::min(minEval, eval);
            }
            return minEval;
        }
    }

    // Compute the best move using Minimax
    void computeBestMoveWithScore(Strategy& strategy) {
        movement best_move(0, 0, 0, 0);
        Sint32 best_score = std::numeric_limits<Sint32>::min();
        vector<movement> valid_moves;

        // Get all valid moves
        strategy.computeValidMoves(valid_moves);

        // If no valid moves, return an empty move
        if (valid_moves.empty()) {
            strategy._saveBestMove(best_move);
            return;
        }

        // Evaluate each move
        for (const movement& mv : valid_moves) {
            Strategy sim_strategy(strategy); // Simulate the move
            sim_strategy.applyMove(mv);

            // Call Minimax for the opponent's turn
            Sint32 score = minimax(sim_strategy, MAX_DEPTH - 1, false);

            // For player 1, invert the score
            if (strategy._current_player == 1) {
                score = -score;
            }

            // Keep track of the best move
            if (score > best_score) {
                best_score = score;
                best_move = mv;
            }
        }

        // Save the best move
        strategy._saveBestMove(best_move);
    }
}