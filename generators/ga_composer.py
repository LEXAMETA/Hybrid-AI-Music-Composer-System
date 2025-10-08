
import numpy as np
import random
import tensorflow as tf

# Constants matching your system
SEQUENCE_LENGTH = 32
PITCH_DIM = 48
CHROMO_DIM = PITCH_DIM + 2  # pitch one-hot + duration + velocity

def pitch_entropy(pitches_onehot):
    probs = np.sum(pitches_onehot, axis=0) / (np.sum(pitches_onehot) + 1e-9)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

def interval_variance(pitches_onehot):
    pitches = np.argmax(pitches_onehot, axis=1)
    intervals = np.diff(pitches)
    return np.var(intervals) if len(intervals) > 0 else 0.0

class GeneticMusicGenerator:
    def __init__(self, evaluator_model, population_size=40, sequence_length=SEQUENCE_LENGTH, mutation_rate=0.07):
        self.evaluator = evaluator_model
        self.population_size = population_size
        self.sequence_length = sequence_length
        self.mutation_rate = mutation_rate
        self.population = [self._random_chromosome() for _ in range(population_size)]

    def _random_chromosome(self):
        chrom = np.zeros((self.sequence_length, CHROMO_DIM))
        for t in range(self.sequence_length):
            pitch = np.random.randint(0, PITCH_DIM)
            chrom[t, pitch] = 1.0
            chrom[t, PITCH_DIM] = np.random.uniform(0.2, 1.0)  # duration
            chrom[t, PITCH_DIM + 1] = np.random.uniform(0.5, 1.0)  # velocity
        return chrom

    def _fitness(self, chromosome):
        pitches = chromosome[:, :PITCH_DIM]
        input_tensor = tf.convert_to_tensor(pitches[np.newaxis, ...], dtype=tf.float32)
        neural_score = float(tf.reduce_mean(self.evaluator(input_tensor)).numpy())
        entropy = pitch_entropy(pitches)
        interval_var = interval_variance(pitches)
        entropy_score = entropy / np.log2(PITCH_DIM)
        interval_score = 1.0 - np.tanh(interval_var / 10.0)
        fitness = 0.5 * neural_score + 0.3 * entropy_score + 0.2 * interval_score
        return fitness

    def _selection(self, fitnesses, num_parents=10):
        parents_idx = np.argsort(fitnesses)[-num_parents:]
        return [self.population[i] for i in parents_idx]

    def _crossover(self, p1, p2):
        point = np.random.randint(1, self.sequence_length - 1)
        c1 = np.vstack((p1[:point, :], p2[point:, :]))
        c2 = np.vstack((p2[:point, :], p1[point:, :]))
        return c1, c2

    def _mutation(self, chromosome):
        for t in range(self.sequence_length):
            if np.random.rand() < self.mutation_rate:
                chromosome[t, :PITCH_DIM] = 0
                new_p = np.random.randint(0, PITCH_DIM)
                chromosome[t, new_p] = 1.0
                chromosome[t, PITCH_DIM] = np.clip(chromosome[t, PITCH_DIM] + np.random.uniform(-0.1, 0.1), 0.1, 1.0)
                chromosome[t, PITCH_DIM+1] = np.clip(chromosome[t, PITCH_DIM+1] + np.random.uniform(-0.1, 0.1), 0.3, 1.0)
        return chromosome

    def evolve(self, generations=30):
        for gen in range(generations):
            fitnesses = [self._fitness(ind) for ind in self.population]
            print(f"Generation {gen+1}/{generations} Max fitness: {max(fitnesses):.4f}")
            parents = self._selection(fitnesses)
            new_pop = parents.copy()
            while len(new_pop) < self.population_size:
                p1, p2 = random.sample(parents, 2)
                c1, c2 = self._crossover(p1, p2)
                new_pop.append(self._mutation(c1))
                if len(new_pop) < self.population_size:
                    new_pop.append(self._mutation(c2))
            self.population = new_pop
        fitnesses = [self._fitness(ind) for ind in self.population]
        best_idx = np.argmax(fitnesses)
        return self.population[best_idx]

def chromosome_to_pitch_sequence(chromosome):
    return [int(np.argmax(note[:PITCH_DIM])) for note in chromosome]
