#ifndef MINIMAX_PARA_H
#define MINIMAX_PARA_H

#include "strategy.h"
#include <tbb/tbb.h>

namespace minimax_para {
    /**
     * Depth limit for the Minimax algorithm
     * Can be adjusted based on available computation time
     */
    extern const int MAX_DEPTH;
    
    /**
     * Recursive minimax function with alpha-beta pruning
     * 
     * @param strategy Current game state
     * @param depth Current depth in the search tree
     * @param maximizingPlayer Whether current player is maximizing (true) or minimizing (false)
     * @param root_player The original player who initiated the search
     * @return Score evaluation for the given position
     */
    Sint32 minimax(Strategy& strategy, int depth, bool maximizingPlayer, Sint32 root_player);
    
    /**
     * Compute the best move using parallelized minimax with Intel TBB
     * 
     * @param strategy Current game state
     */
    void computeBestMoveWithScore(Strategy& strategy);
    
    /**
     * Optional: Alpha-beta pruning version of minimax for even better performance
     * 
     * @param strategy Current game state
     * @param depth Current depth in the search tree
     * @param alpha Best score for maximizing player
     * @param beta Best score for minimizing player
     * @param maximizingPlayer Whether current player is maximizing (true) or minimizing (false)
     * @param root_player The original player who initiated the search
     * @return Score evaluation for the given position
     */
    Sint32 alphabeta(Strategy& strategy, int depth, Sint32 alpha, Sint32 beta, 
                     bool maximizingPlayer, Sint32 root_player);
}

#endif // MINIMAX_PARA_H