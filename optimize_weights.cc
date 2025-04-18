#include "ga_weight_optimizer.h"
#include <iostream>
#include <ctime>

int main(int argc, char** argv) {
    // Parse command line arguments
    int population_size = 20;
    int tournament_games = 5;
    int generations = 10;
    int mutation_chance = 20;   // 20% chance
    int mutation_range = 20;    // +/- 20 units
    int variation_range = 30;   // +/- 30 units for initial population
    
    if (argc > 1) population_size = std::atoi(argv[1]);
    if (argc > 2) tournament_games = std::atoi(argv[2]);
    if (argc > 3) generations = std::atoi(argv[3]);
    
    // Record start time
    time_t start_time = time(nullptr);
    
    // Create optimizer with parameters
    WeightOptimizer optimizer(population_size, tournament_games, generations, 
                             mutation_chance, mutation_range, variation_range);
    
    // Run optimization
    std::cout << "Starting weight optimization with:" << std::endl
              << "  Population size: " << population_size << std::endl
              << "  Tournament games per pair: " << tournament_games << std::endl
              << "  Generations: " << generations << std::endl;
    
    optimizer.run();
    
    // Print results
    const ChromosomeWeights& best = optimizer.getBestChromosome();
    std::cout << "\nOptimization complete!" << std::endl
              << "Best fitness: " << best.getFitness() << std::endl
              << "Wins: " << best.wins << ", Draws: " << best.draws 
              << ", Losses: " << best.losses << std::endl
              << "Total runtime: " << (time(nullptr) - start_time) << " seconds" << std::endl;
    
    return 0;
}