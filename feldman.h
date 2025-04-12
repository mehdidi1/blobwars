#ifndef FELDMAN_H
#define FELDMAN_H

#include "strategy.h"
#include <tbb/tbb.h>

namespace feldman {
    /**
     * Depth limit for the Alpha-Beta algorithm
     * Can be adjusted based on available computation time
     */
    extern const int MAX_DEPTH;
    
    /**
     * Minimum depth for parallelization using the Feldmann-Mysliwietz-Monien heuristic
     * Only subtrees at or above this depth will be considered for parallelization
     */
    extern const int PARALLEL_THRESHOLD;
    
    /**
     * Alpha-Beta pruning with Feldmann-Mysliwietz-Monien parallelization heuristic
     * 
     * Implementation of the FMM heuristic:
     * 1. Process the first child sequentially
     * 2. If the first child doesn't cause a cutoff, process remaining children in parallel
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
     * Compute the best move using parallel Alpha-Beta pruning with FMM heuristic
     * 
     * @param strategy Current game state
     */
    void computeBestMoveWithScore(Strategy& strategy);
    
    /**
     * Counts the number of nodes explored during search
     * Used for performance analysis
     */
    extern unsigned long long nodes_explored;
    
    /**
     * Counts the number of cutoffs that occurred during search
     * Used for performance analysis
     */
    extern unsigned long long cutoffs;
}

#endif // FELDMAN_H