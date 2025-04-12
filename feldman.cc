#include "alpha_beta_para.h"
#include <limits>
#include <tbb/tbb.h>
#include <tbb/parallel_for.h>
#include <tbb/spin_mutex.h>
#include <atomic>
#include <cstdlib>
#include <ctime>
#include <iostream>

namespace feldman
{
    // Depth limit for the Alpha-Beta algorithm
    const int MAX_DEPTH = 4;

    // Minimum depth for parallelization
    const int PARALLEL_THRESHOLD = 2;

    // Alpha-Beta pruning recursive function with FMM parallelization heuristic
    Sint32 alphabeta(Strategy &strategy, int depth, Sint32 alpha, Sint32 beta, bool maximizingPlayer, Sint32 root_player)
    {
        // Base case: if we reach the maximum depth or no valid moves are left
        vector<movement> valid_moves;
        strategy.computeValidMoves(valid_moves);
        if (depth == 0 || valid_moves.empty())
        {
            return strategy.estimateCurrentScore(root_player);
        }

        // Only parallelize when enough work exists (deeper in tree)
        if (depth >= PARALLEL_THRESHOLD && valid_moves.size() > 3)
        {
            // Maximizing player
            if (maximizingPlayer)
            {
                Sint32 value = std::numeric_limits<Sint32>::min();

                // First try the first move sequentially
                if (!valid_moves.empty())
                {
                    const movement &mv = valid_moves[0];
                    Strategy sim_strategy(strategy);
                    sim_strategy.applyMove(mv);
                    value = alphabeta(sim_strategy, depth - 1, alpha, beta, false, root_player);
                    alpha = std::max(alpha, value);

                    // If this first move caused a cutoff, we're done
                    if (alpha >= beta)
                    {
                        return value;
                    }
                }

                // No cutoff after first move, explore remaining moves in parallel
                if (valid_moves.size() > 1)
                {
                    std::atomic<Sint32> shared_alpha{alpha};

                    // Parallel exploration of remaining moves
                    Sint32 parallel_value = tbb::parallel_reduce(
                        tbb::blocked_range<size_t>(1, valid_moves.size()), // Start from 1 (skipping first move)
                        std::numeric_limits<Sint32>::min(),
                        [&](const tbb::blocked_range<size_t> &range, Sint32 local_value)
                        {
                            for (size_t i = range.begin(); i < range.end(); ++i)
                            {
                                // Early cutoff check based on shared alpha/beta
                                if (shared_alpha.load() >= beta)
                                {
                                    continue; // Skip this branch
                                }

                                const movement &mv = valid_moves[i];
                                Strategy sim_strategy(strategy);
                                sim_strategy.applyMove(mv);

                                Sint32 eval = alphabeta(sim_strategy, depth - 1, shared_alpha.load(), beta, false, root_player);
                                local_value = std::max(local_value, eval);

                                // Update shared alpha if we found a better value
                                Sint32 current_alpha = shared_alpha.load();
                                while (local_value > current_alpha &&
                                       !shared_alpha.compare_exchange_weak(current_alpha, local_value))
                                {
                                    // If CAS failed, current_alpha has the latest value, try again
                                }
                            }
                            return local_value;
                        },
                        [](Sint32 x, Sint32 y)
                        { return std::max(x, y); });

                    // Combine the results from sequential and parallel parts
                    value = std::max(value, parallel_value);
                }

                return value;
            }
            // Minimizing player
            else
            {
                Sint32 value = std::numeric_limits<Sint32>::max();

                // First try the first move sequentially
                if (!valid_moves.empty())
                {
                    const movement &mv = valid_moves[0];
                    Strategy sim_strategy(strategy);
                    sim_strategy.applyMove(mv);
                    value = alphabeta(sim_strategy, depth - 1, alpha, beta, true, root_player);
                    beta = std::min(beta, value);

                    // If this first move caused a cutoff, we're done
                    if (beta <= alpha)
                    {
                        return value;
                    }
                }

                // No cutoff after first move, explore remaining moves in parallel
                if (valid_moves.size() > 1)
                {
                    std::atomic<Sint32> shared_beta{beta};

                    // Parallel exploration of remaining moves
                    Sint32 parallel_value = tbb::parallel_reduce(
                        tbb::blocked_range<size_t>(1, valid_moves.size()), // Start from 1 (skipping first move)
                        std::numeric_limits<Sint32>::max(),
                        [&](const tbb::blocked_range<size_t> &range, Sint32 local_value)
                        {
                            for (size_t i = range.begin(); i < range.end(); ++i)
                            {
                                // Early cutoff check based on shared alpha/beta
                                if (alpha >= shared_beta.load())
                                {
                                    continue; // Skip this branch
                                }

                                const movement &mv = valid_moves[i];
                                Strategy sim_strategy(strategy);
                                sim_strategy.applyMove(mv);

                                Sint32 eval = alphabeta(sim_strategy, depth - 1, alpha, shared_beta.load(), true, root_player);
                                local_value = std::min(local_value, eval);

                                // Update shared beta if we found a better value
                                Sint32 current_beta = shared_beta.load();
                                while (local_value < current_beta &&
                                       !shared_beta.compare_exchange_weak(current_beta, local_value))
                                {
                                    // If CAS failed, current_beta has the latest value, try again
                                }
                            }
                            return local_value;
                        },
                        [](Sint32 x, Sint32 y)
                        { return std::min(x, y); });

                    // Combine the results from sequential and parallel parts
                    value = std::min(value, parallel_value);
                }

                return value;
            }
        }
        else
        {
            // Sequential alpha-beta for small subtrees
            // Maximizing player
            if (maximizingPlayer)
            {
                Sint32 value = std::numeric_limits<Sint32>::min();
                for (const movement &mv : valid_moves)
                {
                    Strategy sim_strategy(strategy);
                    sim_strategy.applyMove(mv);
                    value = std::max(value, alphabeta(sim_strategy, depth - 1, alpha, beta, false, root_player));
                    alpha = std::max(alpha, value);
                    if (alpha >= beta)
                    {
                        break; // Beta cut-off
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
                    Strategy sim_strategy(strategy);
                    sim_strategy.applyMove(mv);
                    value = std::min(value, alphabeta(sim_strategy, depth - 1, alpha, beta, true, root_player));
                    beta = std::min(beta, value);
                    if (beta <= alpha)
                    {
                        break; // Alpha cut-off
                    }
                }
                return value;
            }
        }
    }

    // Compute the best move using parallel Alpha-Beta pruning
    void computeBestMoveWithScore(Strategy &strategy)
    {
        movement best_move(0, 0, 0, 0);
        std::atomic<Sint32> best_score{std::numeric_limits<Sint32>::min()};
        vector<movement> valid_moves;
        tbb::spin_mutex best_move_mutex;

        // Get all valid moves
        strategy.computeValidMoves(valid_moves);

        // If no valid moves, return an empty move
        if (valid_moves.empty())
        {
            strategy._saveBestMove(best_move);
            std::cout << "No valid move available" << std::endl;
            return;
        }

        // Parallel evaluation of all moves using TBB
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, valid_moves.size()),
            [&](const tbb::blocked_range<size_t> &range)
            {
                for (size_t i = range.begin(); i < range.end(); ++i)
                {
                    const movement &mv = valid_moves[i];
                    Strategy sim_strategy(strategy); // Simulate the move
                    sim_strategy.applyMove(mv);

                    // Call Alpha-Beta for the opponent's turn
                    Sint32 score = alphabeta(sim_strategy, MAX_DEPTH - 1,
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
                            best_move = mv;
                            strategy._saveBestMove(best_move);
                        }
                    }
                }
            });
    }
}