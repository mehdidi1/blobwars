#ifndef MINIMAX_H
#define MINIMAX_H

#include "strategy.h"

namespace minimax {
    /**
     * Compute the best move using the Minimax algorithm.
     * @param strategy The current game state.
     */
    void computeBestMoveWithScore(Strategy& strategy);

    /**
     * Minimax recursive function to evaluate the best move.
     * @param strategy The current game state.
     * @param depth The current depth of the search.
     * @param maximizingPlayer True if the current player is maximizing their score.
     * @param root_player player for whom we are making the best move
     * @return The score of the best move.
     */
    Sint32 minimax(Strategy& strategy, int depth, bool maximizingPlayer,Sint32 root_player);
}

#endif // MINIMAX_H