#include "ga_weight_optimizer.h"
#include <algorithm>
#include <chrono>
#include <iostream>
// Add these includes if needed
#include "strategy.h"
#include "alpha_beta.h"
#include "alpha_beta_para.h"
#include "feldman.h"

// Global variables for move saving
static movement current_saved_move;

static void move_saver_function(movement& mv) {
    current_saved_move = mv;
}

// Game simulation helper functions
bool isGameOver(Strategy& strategy) {
    // Game is over when no valid moves for either player
    std::vector<movement> moves;
    
    // Check player 0
    Strategy tmp = strategy;
    tmp._current_player = 0;
    tmp.computeValidMoves(moves);
    bool p0_can_move = !moves.empty();
    
    // Check player 1
    moves.clear();
    tmp._current_player = 1;
    tmp.computeValidMoves(moves);
    bool p1_can_move = !moves.empty();
    
    return !p0_can_move && !p1_can_move;
}

int getWinner(Strategy& strategy) {
    int p0_score = 0, p1_score = 0;
    
    // Count blobs
    for (int x = 0; x < 8; x++) {
        for (int y = 0; y < 8; y++) {
            if (strategy._blobs.get(x, y) == 0)
                p0_score++;
            else if (strategy._blobs.get(x, y) == 1)
                p1_score++;
        }
    }
    
    if (p0_score > p1_score) return 0;
    else if (p1_score > p0_score) return 1;
    else return -1; // Draw
}

void playFullGame(Strategy& strategy, ChromosomeWeights* weights0, ChromosomeWeights* weights1) {
    // Limit moves to prevent infinite games
    int max_moves = 100;
    int move_count = 0;
    
    // Play until game over or move limit reached
    while (!isGameOver(strategy) && move_count < max_moves) {
        // Set weights for current player
        if (strategy._current_player == 0) {
            strategy.useOptimizedWeights(weights0);
        } else {
            strategy.useOptimizedWeights(weights1);
        }
        
        // Compute and apply best move
        strategy.computeBestMove();
        
        // Apply the move that was saved by the move_saver_function
        strategy.applyMove(current_saved_move);
        
        move_count++;
    }
}

void ChromosomeWeights::initializeDefault() {
    // Initialize with current weights from the strategy.cc file


    // Early game
    early_game.material_weight = 84;
    early_game.corner_weight = 121;
    early_game.mobility_weight = -6;
    early_game.capture_weight = -10;
    early_game.frontier_weight = -20;
    
    // Mid game
    mid_game.material_weight = 129;
    mid_game.corner_weight = 40;
    mid_game.mobility_weight = 5;
    mid_game.capture_weight = 10;
    mid_game.frontier_weight = -86;
    
    // Late game
    late_game.material_weight = 151;
    late_game.corner_weight = 81;
    late_game.mobility_weight = 5;
    late_game.capture_weight = 12;
    late_game.frontier_weight = -55;

    
    // Reset fitness
    wins = draws = losses = games = 0;
}

void ChromosomeWeights::randomize(std::mt19937& rng, int range) {
    // Initialize with random values for all weights
    std::uniform_int_distribution<Sint32> dist(-range, range);
    
    // Phase weights
    early_game.material_weight = 80 + dist(rng);
    early_game.corner_weight = 120 + dist(rng);
    early_game.mobility_weight = 15 + dist(rng);
    early_game.capture_weight = 20 + dist(rng);
    early_game.frontier_weight = -30 + dist(rng);
    
    mid_game.material_weight = 100 + dist(rng);
    mid_game.corner_weight = 70 + dist(rng);
    mid_game.mobility_weight = 30 + dist(rng);
    mid_game.capture_weight = 40 + dist(rng);
    mid_game.frontier_weight = -40 + dist(rng);
    
    late_game.material_weight = 120 + dist(rng);
    late_game.corner_weight = 60 + dist(rng);
    late_game.mobility_weight = 10 + dist(rng);
    late_game.capture_weight = 20 + dist(rng);
    late_game.frontier_weight = -50 + dist(rng);
}

ChromosomeWeights ChromosomeWeights::crossover(
    const ChromosomeWeights& parent1, 
    const ChromosomeWeights& parent2,
    std::mt19937& rng) {
    
    ChromosomeWeights child;
    
    // Use uniform distribution for random crossover points
    std::uniform_int_distribution<int> dist(0, 1);
    
    // Randomly select weights from either parent
    // Early game weights
    child.early_game.material_weight = 
        dist(rng) ? parent1.early_game.material_weight : parent2.early_game.material_weight;
    child.early_game.corner_weight = 
        dist(rng) ? parent1.early_game.corner_weight : parent2.early_game.corner_weight;
    child.early_game.mobility_weight = 
        dist(rng) ? parent1.early_game.mobility_weight : parent2.early_game.mobility_weight;
    child.early_game.capture_weight = 
        dist(rng) ? parent1.early_game.capture_weight : parent2.early_game.capture_weight;
    child.early_game.frontier_weight = 
        dist(rng) ? parent1.early_game.frontier_weight : parent2.early_game.frontier_weight;
    
    // Mid game weights
    child.mid_game.material_weight = 
        dist(rng) ? parent1.mid_game.material_weight : parent2.mid_game.material_weight;
    child.mid_game.corner_weight = 
        dist(rng) ? parent1.mid_game.corner_weight : parent2.mid_game.corner_weight;
    child.mid_game.mobility_weight = 
        dist(rng) ? parent1.mid_game.mobility_weight : parent2.mid_game.mobility_weight;
    child.mid_game.capture_weight = 
        dist(rng) ? parent1.mid_game.capture_weight : parent2.mid_game.capture_weight;
    child.mid_game.frontier_weight = 
        dist(rng) ? parent1.mid_game.frontier_weight : parent2.mid_game.frontier_weight;
    
    // Late game weights
    child.late_game.material_weight = 
        dist(rng) ? parent1.late_game.material_weight : parent2.late_game.material_weight;
    child.late_game.corner_weight = 
        dist(rng) ? parent1.late_game.corner_weight : parent2.late_game.corner_weight;
    child.late_game.mobility_weight = 
        dist(rng) ? parent1.late_game.mobility_weight : parent2.late_game.mobility_weight;
    child.late_game.capture_weight = 
        dist(rng) ? parent1.late_game.capture_weight : parent2.late_game.capture_weight;
    child.late_game.frontier_weight = 
        dist(rng) ? parent1.late_game.frontier_weight : parent2.late_game.frontier_weight;
    
    // Reset fitness
    child.wins = child.draws = child.losses = child.games = 0;
    
    return child;
}

void ChromosomeWeights::mutate(std::mt19937& rng, int mutation_chance, int mutation_range) {
    // Probability distribution for mutation chance
    std::uniform_int_distribution<int> chance(1, 100);
    
    // Distribution for mutation amount
    std::uniform_int_distribution<int> amount(-mutation_range, mutation_range);
    
    // Potentially mutate each weight
    if (chance(rng) <= mutation_chance) early_game.material_weight += amount(rng);
    if (chance(rng) <= mutation_chance) early_game.corner_weight += amount(rng);
    if (chance(rng) <= mutation_chance) early_game.mobility_weight += amount(rng);
    if (chance(rng) <= mutation_chance) early_game.capture_weight += amount(rng);
    if (chance(rng) <= mutation_chance) early_game.frontier_weight += amount(rng);
    
    if (chance(rng) <= mutation_chance) mid_game.material_weight += amount(rng);
    if (chance(rng) <= mutation_chance) mid_game.corner_weight += amount(rng);
    if (chance(rng) <= mutation_chance) mid_game.mobility_weight += amount(rng);
    if (chance(rng) <= mutation_chance) mid_game.capture_weight += amount(rng);
    if (chance(rng) <= mutation_chance) mid_game.frontier_weight += amount(rng);
    
    if (chance(rng) <= mutation_chance) late_game.material_weight += amount(rng);
    if (chance(rng) <= mutation_chance) late_game.corner_weight += amount(rng);
    if (chance(rng) <= mutation_chance) late_game.mobility_weight += amount(rng);
    if (chance(rng) <= mutation_chance) late_game.capture_weight += amount(rng);
    if (chance(rng) <= mutation_chance) late_game.frontier_weight += amount(rng);
}

bool ChromosomeWeights::saveToFile(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) return false;
    
    // Write all weights to file
    file << early_game.material_weight << " " << early_game.corner_weight << " "  << early_game.mobility_weight << " "
         << early_game.capture_weight << " " << early_game.frontier_weight << "\n";
    
    file << mid_game.material_weight << " " << mid_game.corner_weight << " " << mid_game.mobility_weight << " "
         << mid_game.capture_weight << " " << mid_game.frontier_weight << "\n";
    
    file << late_game.material_weight << " " << late_game.corner_weight << " " << late_game.mobility_weight << " "
         << late_game.capture_weight << " " << late_game.frontier_weight << "\n";
    
    file << wins << " " << draws << " " << losses << " " << games << "\n";
    
    file.close();
    return true;
}

bool ChromosomeWeights::loadFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) return false;
    
    file >> early_game.material_weight >> early_game.corner_weight >> early_game.mobility_weight
         >> early_game.capture_weight >> early_game.frontier_weight;
    
    file >> mid_game.material_weight >> mid_game.corner_weight >> mid_game.mobility_weight
         >> mid_game.capture_weight >> mid_game.frontier_weight;
    
    file >> late_game.material_weight >> late_game.corner_weight >> late_game.mobility_weight
         >> late_game.capture_weight >> late_game.frontier_weight;
    
    file >> wins >> draws >> losses >> games;
    
    file.close();
    return true;
}

WeightOptimizer::WeightOptimizer(int pop_size, int tourn_games, int gens, 
                               int mut_chance, int mut_range, int var_range)
    : population_size(pop_size), tournament_games(tourn_games), generations(gens),
      mutation_chance(mut_chance), mutation_range(mut_range), variation_range(var_range) {
    
    // Initialize RNG with current time
    rng.seed(static_cast<unsigned int>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count()));
}

void WeightOptimizer::initialize() {
    population.clear();
    population.reserve(population_size);
    
    // Create initial population with random variations
    for (int i = 0; i < population_size; i++) {
        ChromosomeWeights chromosome;
        chromosome.randomize(rng, variation_range);
        population.push_back(chromosome);
    }
}

void WeightOptimizer::runTournament() {
    // Reset all fitness values
    for (auto& chromosome : population) {
        chromosome.wins = chromosome.draws = chromosome.losses = chromosome.games = 0;
    }
    
    // Each chromosome plays against the neutral algorithm
    for (size_t i = 0; i < population.size(); i++) {
        ChromosomeWeights& weights = population[i];
        
        std::cout << "Evaluating weight set " << i << std::endl;
        
        // Play multiple games against neutral algorithm
        for (int game = 0; game < tournament_games; game++) {
            // Each weight set plays as both first and second player
            playGameAgainstNeutral(&weights, true, weights.wins, weights.draws, weights.losses);
            weights.games++;
            
            playGameAgainstNeutral(&weights, false, weights.wins, weights.draws, weights.losses);
            weights.games++;
            
            std::cout << "Game " << game << " complete. Current record: " 
                      << weights.wins << " wins, " << weights.draws << " draws, "
                      << weights.losses << " losses" << std::endl;
        }
        
        std::cout << "Weight set " << i << " final fitness: " << weights.getFitness() << std::endl;
    }
}

void WeightOptimizer::evolve() {
    // Sort population by fitness
    std::sort(population.begin(), population.end(), 
              [](const ChromosomeWeights& a, const ChromosomeWeights& b) {
                  return a.getFitness() > b.getFitness();
              });
    
    // Keep only the top half
    int keep = population_size / 2;
    population.resize(keep);
    
    // Create new individuals through crossover and mutation
    while ( (int)population.size() < population_size) {
        // Select two random parents from survivors
        std::uniform_int_distribution<int> select(0, keep - 1);
        int parent1_idx = select(rng);
        int parent2_idx = select(rng);
        
        // Create child through crossover
        ChromosomeWeights child = ChromosomeWeights::crossover(
            population[parent1_idx], population[parent2_idx], rng);
        
        // Potentially mutate child
        child.mutate(rng, mutation_chance, mutation_range);
        
        // Add child to population
        population.push_back(child);
    }
}

void WeightOptimizer::run() {
    // Initialize population
    initialize();
    
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();
    auto last_gen_time = start_time;
    
    // Run for specified number of generations
    for (int gen = 0; gen < generations; gen++) {
        std::cout << "Generation " << gen + 1 << "/" << generations << std::endl;
        
        // Evaluate fitness through tournament
        runTournament();
        
        // Print best fitness
        std::cout << "Best fitness: " << getBestChromosome().getFitness() << std::endl;
        
        // Calculate time statistics
        auto current_time = std::chrono::high_resolution_clock::now();
        auto gen_duration = std::chrono::duration_cast<std::chrono::seconds>(
            current_time - last_gen_time).count();
        auto total_elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            current_time - start_time).count();
        
        // Estimate remaining time
        double avg_gen_time = total_elapsed / (gen + 1.0);
        int remaining_gens = generations - (gen + 1);
        int estimated_remaining_seconds = avg_gen_time * remaining_gens;
        
        // Format and display time information
        int hours = estimated_remaining_seconds / 3600;
        int minutes = (estimated_remaining_seconds % 3600) / 60;
        int seconds = estimated_remaining_seconds % 60;
        
        std::cout << "Generation " << gen + 1 << " completed in " << gen_duration << " seconds" << std::endl;
        std::cout << "Estimated time remaining: " 
                  << hours << "h " << minutes << "m " << seconds << "s" << std::endl;
        
        // Update last generation time
        last_gen_time = current_time;
        
        // Create next generation (except for last iteration)
        if (gen < generations - 1) {
            evolve();
        }
    }
    
    // Calculate total runtime
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_seconds = std::chrono::duration_cast<std::chrono::seconds>(
        end_time - start_time).count();
    int hours = total_seconds / 3600;
    int minutes = (total_seconds % 3600) / 60;
    int seconds = total_seconds % 60;
    
    // Save best chromosome
    saveBestChromosome("best_weights.txt");
    std::cout << "Best weights saved to best_weights.txt" << std::endl;
    std::cout << "Total optimization time: "
              << hours << "h " << minutes << "m " << seconds << "s" << std::endl;
}

const ChromosomeWeights& WeightOptimizer::getBestChromosome() const {
    auto best_it = std::max_element(population.begin(), population.end(),
        [](const ChromosomeWeights& a, const ChromosomeWeights& b) {
            return a.getFitness() < b.getFitness();
        });
    
    return *best_it;
}

void WeightOptimizer::saveBestChromosome(const std::string& filename) const {
    getBestChromosome().saveToFile(filename);
}

void WeightOptimizer::playGameAgainstNeutral(ChromosomeWeights* weights, bool weights_play_first, 
                                           int& wins, int& draws, int& losses) const {
    // Initialize board state
    bidiarray<Sint16> blobs;
    bidiarray<bool> holes;
    for (int x = 0; x < 8; x++) {
        for (int y = 0; y < 8; y++) {
            blobs.set(x, y, -1);
            holes.set(x, y, false);
        }
    }
    
    // Set initial positions
    blobs.set(0, 0, 0); // Player 0 (Red)
    blobs.set(0, 7, 0);
    blobs.set(7, 7, 1); // Player 1 (Blue)
    blobs.set(7, 0, 1);
    
    // Create game with move saver
    Strategy gameState(blobs, holes, 0, move_saver_function);
    
    // Limit moves to prevent infinite games
    int max_moves = 100;
    int move_count = 0;
    
    // Play until game over or move limit reached
    while (!isGameOver(gameState) && move_count < max_moves) {
        // Check if current player has valid moves
        std::vector<movement> valid_moves;
        gameState.computeValidMoves(valid_moves);
        
        if (valid_moves.empty()) {
            // No moves available - switch players and continue
            gameState._current_player = 1 - gameState._current_player;
            continue;
        }
        
        bool is_weights_turn = (gameState._current_player == 0 && weights_play_first) || 
                              (gameState._current_player == 1 && !weights_play_first);
        
        // Choose algorithm based on whose turn it is
        if (is_weights_turn) {
            // Optimized weights player's turn
            gameState.useOptimizedWeights(weights);
            alpha_beta_para::computeBestMoveWithScore(gameState);
        } else {
            // Neutral algorithm turn - use greedy instead of feldman
            feldman::computeBestMoveWithScore(gameState);
        }
        
        // Apply the move
        gameState.applyMove(current_saved_move);
        move_count++;
    }
    
    // Determine winner
    int winner = getWinner(gameState);
    int weights_player = weights_play_first ? 0 : 1;
    
    if (winner == -1) {
        draws++;
    } else if (winner == weights_player) {
        wins++;
    } else {
        losses++;
    }
}