#ifndef ALPHA_BETA_PARA_H
#define ALPHA_BETA_PARA_H

#include "strategy.h"

namespace alpha_beta_para {
    /**
     * Depth limit for the Alpha-Beta algorithm
     * Can be adjusted based on available computation time
     */
    extern const int MAX_DEPTH;
    
    /**
     * Alpha-Beta pruning recursive function
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
    
    /**
     * Compute the best move using parallel Alpha-Beta pruning
     * 
     * @param strategy Current game state
     */
    void computeBestMoveWithScore(Strategy& strategy);
}

#endif // ALPHA_BETA_PARA_H