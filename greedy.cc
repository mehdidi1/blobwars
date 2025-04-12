#include "strategy.h"
#include "greedy.h"

namespace greedy {
    Sint32 estimateScore(const Strategy& strategy,Sint32 player) {
        return strategy.estimateCurrentScore(player); //Use default 
    }
    
    void computeBestMoveWithScore(Strategy& strategy) {
        movement best_move(0, 0, 0, 0);
        Sint32 best_score = -1000000;
        vector<movement> valid_moves;
        
        strategy.computeValidMoves(valid_moves);
        
        for (const movement& mv : valid_moves) {
            Strategy sim_strategy(strategy);  // Create a copy
            sim_strategy.applyMove(mv);
            
            Sint32 score = estimateScore(sim_strategy,strategy._current_player);

                
            if (score > best_score) {
                best_score = score;
                best_move = mv;
            }
        }
        
        strategy._saveBestMove(best_move);
    }
}