#ifndef GA_WEIGHT_OPTIMIZER_H
#define GA_WEIGHT_OPTIMIZER_H

#include <SDL/SDL.h>
#include <vector>
#include <random>
#include <fstream>
#include <iostream>

// Represents a complete set of weights for evaluation across all game phases
struct ChromosomeWeights {
    // Per-phase weights
    struct PhaseWeights {
        Sint32 material_weight;
        Sint32 corner_weight;
        Sint32 mobility_weight;
        Sint32 capture_weight;
        Sint32 frontier_weight;
    };
    
    // Phase-specific weights
    PhaseWeights early_game;
    PhaseWeights mid_game;
    PhaseWeights late_game;
    
    // Fitness metrics
    int wins = 0;
    int draws = 0;
    int losses = 0;
    int games = 0;
    
    double getFitness() const {
        if (games == 0) return 0.0;
        return (wins + 0.5 * draws) / games;
    }
    
    // Initialize with current weights or variations
    void initializeDefault();
    void randomize(std::mt19937& rng, int variation_range);
    
    // Genetic operators
    static ChromosomeWeights crossover(const ChromosomeWeights& parent1, 
                                      const ChromosomeWeights& parent2,
                                      std::mt19937& rng);
    void mutate(std::mt19937& rng, int mutation_chance, int mutation_range);
    
    // Save/load from file
    bool saveToFile(const std::string& filename) const;
    bool loadFromFile(const std::string& filename);
};

// Main GA class to run tournaments and evolution
class WeightOptimizer {
private:
    std::vector<ChromosomeWeights> population;
    std::mt19937 rng;
    int population_size;
    int tournament_games;
    int generations;
    int mutation_chance; // percentage
    int mutation_range;
    int variation_range;
    
public:
    WeightOptimizer(int pop_size = 20, int tourn_games = 10, int gens = 10, 
                  int mut_chance = 20, int mut_range = 20, int var_range = 30);
    
    void initialize();
    void runTournament();
    void evolve();
    void run();
    
    const ChromosomeWeights& getBestChromosome() const;
    void saveBestChromosome(const std::string& filename) const;
    void playGameAgainstNeutral(ChromosomeWeights* weights, bool weights_play_first, int& wins, int& draws, int& losses) const;
};

#endif // GA_WEIGHT_OPTIMIZER_H