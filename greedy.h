#ifndef GREEDY_H
#define GREEDY_H

#include "strategy.h"

namespace greedy {
    /**
     * Custom scoring function for greedy strategy
     * @param strategy The game strategy state to evaluate
     * @return A score value, higher is better for current player
     */
    Sint32 estimateScore(const Strategy& strategy);
    
    /**
     * Computes the best move for the current player using
     * a greedy approach that maximizes immediate score gain
     * @param strategy The game strategy containing the current board state
     */
    void computeBestMoveWithScore(Strategy& strategy);
}

#endif // GREEDY_H