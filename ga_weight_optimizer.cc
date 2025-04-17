#include "ga_weight_optimizer.h"
#include <algorithm>
#include <chrono>

void ChromosomeWeights::initializeDefault() {
    // Initialize with current weights from the strategy.cc file
    
    // Early game
    early_game.material_weight = 80;
    early_game.corner_weight = 120;
    early_game.mobility_weight = 15;
    early_game.positional_weight = 20;
    early_game.capture_weight = 20;
    early_game.frontier_weight = -30;
    
    // Mid game
    mid_game.material_weight = 100;
    mid_game.corner_weight = 70;
    mid_game.mobility_weight = 30;
    mid_game.positional_weight = 10;
    mid_game.capture_weight = 40;
    mid_game.frontier_weight = -40;
    
    // Late game
    late_game.material_weight = 120;
    late_game.corner_weight = 60;
    late_game.mobility_weight = 10;
    late_game.positional_weight = 5;
    late_game.capture_weight = 20;
    late_game.frontier_weight = -50;
    
    // Pattern weights
    fortress_weight = 40;
    wall_weight = 25;
    invasion_platform_weight = 30;
    pincer_weight = 35;
    expansion_hub_weight = 20;
    
    // Reset fitness
    wins = draws = losses = games = 0;
}

void ChromosomeWeights::randomize(std::mt19937& rng, int variation_range) {
    // Start with defaults
    initializeDefault();
    
    // Create distribution for variations
    std::uniform_int_distribution<int> variation(-variation_range, variation_range);
    
    // Apply random variations to each weight
    early_game.material_weight += variation(rng);
    early_game.corner_weight += variation(rng);
    early_game.mobility_weight += variation(rng);
    early_game.positional_weight += variation(rng);
    early_game.capture_weight += variation(rng);
    early_game.frontier_weight += variation(rng);
    
    mid_game.material_weight += variation(rng);
    mid_game.corner_weight += variation(rng);
    mid_game.mobility_weight += variation(rng);
    mid_game.positional_weight += variation(rng);
    mid_game.capture_weight += variation(rng);
    mid_game.frontier_weight += variation(rng);
    
    late_game.material_weight += variation(rng);
    late_game.corner_weight += variation(rng);
    late_game.mobility_weight += variation(rng);
    late_game.positional_weight += variation(rng);
    late_game.capture_weight += variation(rng);
    late_game.frontier_weight += variation(rng);
    
    fortress_weight += variation(rng);
    wall_weight += variation(rng);
    invasion_platform_weight += variation(rng);
    pincer_weight += variation(rng);
    expansion_hub_weight += variation(rng);
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
    child.early_game.positional_weight = 
        dist(rng) ? parent1.early_game.positional_weight : parent2.early_game.positional_weight;
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
    child.mid_game.positional_weight = 
        dist(rng) ? parent1.mid_game.positional_weight : parent2.mid_game.positional_weight;
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
    child.late_game.positional_weight = 
        dist(rng) ? parent1.late_game.positional_weight : parent2.late_game.positional_weight;
    child.late_game.capture_weight = 
        dist(rng) ? parent1.late_game.capture_weight : parent2.late_game.capture_weight;
    child.late_game.frontier_weight = 
        dist(rng) ? parent1.late_game.frontier_weight : parent2.late_game.frontier_weight;
    
    // Pattern weights
    child.fortress_weight = 
        dist(rng) ? parent1.fortress_weight : parent2.fortress_weight;
    child.wall_weight = 
        dist(rng) ? parent1.wall_weight : parent2.wall_weight;
    child.invasion_platform_weight = 
        dist(rng) ? parent1.invasion_platform_weight : parent2.invasion_platform_weight;
    child.pincer_weight = 
        dist(rng) ? parent1.pincer_weight : parent2.pincer_weight;
    child.expansion_hub_weight = 
        dist(rng) ? parent1.expansion_hub_weight : parent2.expansion_hub_weight;
    
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
    if (chance(rng) <= mutation_chance) early_game.positional_weight += amount(rng);
    if (chance(rng) <= mutation_chance) early_game.capture_weight += amount(rng);
    if (chance(rng) <= mutation_chance) early_game.frontier_weight += amount(rng);
    
    if (chance(rng) <= mutation_chance) mid_game.material_weight += amount(rng);
    if (chance(rng) <= mutation_chance) mid_game.corner_weight += amount(rng);
    if (chance(rng) <= mutation_chance) mid_game.mobility_weight += amount(rng);
    if (chance(rng) <= mutation_chance) mid_game.positional_weight += amount(rng);
    if (chance(rng) <= mutation_chance) mid_game.capture_weight += amount(rng);
    if (chance(rng) <= mutation_chance) mid_game.frontier_weight += amount(rng);
    
    if (chance(rng) <= mutation_chance) late_game.material_weight += amount(rng);
    if (chance(rng) <= mutation_chance) late_game.corner_weight += amount(rng);
    if (chance(rng) <= mutation_chance) late_game.mobility_weight += amount(rng);
    if (chance(rng) <= mutation_chance) late_game.positional_weight += amount(rng);
    if (chance(rng) <= mutation_chance) late_game.capture_weight += amount(rng);
    if (chance(rng) <= mutation_chance) late_game.frontier_weight += amount(rng);
    
    if (chance(rng) <= mutation_chance) fortress_weight += amount(rng);
    if (chance(rng) <= mutation_chance) wall_weight += amount(rng);
    if (chance(rng) <= mutation_chance) invasion_platform_weight += amount(rng);
    if (chance(rng) <= mutation_chance) pincer_weight += amount(rng);
    if (chance(rng) <= mutation_chance) expansion_hub_weight += amount(rng);
}

bool ChromosomeWeights::saveToFile(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) return false;
    
    // Write all weights to file
    file << early_game.material_weight << " " << early_game.corner_weight << " " 
         << early_game.mobility_weight << " " << early_game.positional_weight << " "
         << early_game.capture_weight << " " << early_game.frontier_weight << "\n";
    
    file << mid_game.material_weight << " " << mid_game.corner_weight << " " 
         << mid_game.mobility_weight << " " << mid_game.positional_weight << " "
         << mid_game.capture_weight << " " << mid_game.frontier_weight << "\n";
    
    file << late_game.material_weight << " " << late_game.corner_weight << " " 
         << late_game.mobility_weight << " " << late_game.positional_weight << " "
         << late_game.capture_weight << " " << late_game.frontier_weight << "\n";
    
    file << fortress_weight << " " << wall_weight << " " << invasion_platform_weight << " "
         << pincer_weight << " " << expansion_hub_weight << "\n";
    
    file << wins << " " << draws << " " << losses << " " << games << "\n";
    
    file.close();
    return true;
}

bool ChromosomeWeights::loadFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) return false;
    
    file >> early_game.material_weight >> early_game.corner_weight 
         >> early_game.mobility_weight >> early_game.positional_weight
         >> early_game.capture_weight >> early_game.frontier_weight;
    
    file >> mid_game.material_weight >> mid_game.corner_weight 
         >> mid_game.mobility_weight >> mid_game.positional_weight
         >> mid_game.capture_weight >> mid_game.frontier_weight;
    
    file >> late_game.material_weight >> late_game.corner_weight 
         >> late_game.mobility_weight >> late_game.positional_weight
         >> late_game.capture_weight >> late_game.frontier_weight;
    
    file >> fortress_weight >> wall_weight >> invasion_platform_weight
         >> pincer_weight >> expansion_hub_weight;
    
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
    
    // Each chromosome plays against all others
    for (size_t i = 0; i < population.size(); i++) {
        for (size_t j = i + 1; j < population.size(); j++) {
            // Play multiple games between i and j
            for (int game = 0; game < tournament_games; game++) {
                // TODO: Implement actual game logic here
                // This would involve modifying Strategy to use the weights from 
                // population[i] and population[j], then running a game
                
                // For now, we'll simulate with random outcomes
                std::uniform_int_distribution<int> result(0, 2); // 0=i wins, 1=draw, 2=j wins
                int outcome = result(rng);
                
                population[i].games++;
                population[j].games++;
                
                if (outcome == 0) {
                    population[i].wins++;
                    population[j].losses++;
                } else if (outcome == 1) {
                    population[i].draws++;
                    population[j].draws++;
                } else {
                    population[i].losses++;
                    population[j].wins++;
                }
            }
        }
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
    
    // Run for specified number of generations
    for (int gen = 0; gen < generations; gen++) {
        std::cout << "Generation " << gen + 1 << "/" << generations << std::endl;
        
        // Evaluate fitness through tournament
        runTournament();
        
        // Print best fitness
        std::cout << "Best fitness: " << getBestChromosome().getFitness() << std::endl;
        
        // Create next generation (except for last iteration)
        if (gen < generations - 1) {
            evolve();
        }
    }
    
    // Save best chromosome
    saveBestChromosome("best_weights.txt");
    std::cout << "Best weights saved to best_weights.txt" << std::endl;
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