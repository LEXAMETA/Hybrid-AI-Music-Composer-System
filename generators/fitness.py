
import numpy as np

def pitch_entropy(pitches_onehot):
    """
    Compute entropy of pitch distribution over generated sequence.
    Higher entropy indicates diverse pitch usage.
    """
    probs = np.sum(pitches_onehot, axis=0) / (np.sum(pitches_onehot) + 1e-9)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

def interval_variance(pitches_onehot):
    """
    Compute variance of pitch intervals.
    Lower variance suggests smoother melodic contour.
    """
    pitches = np.argmax(pitches_onehot, axis=1)
    intervals = np.diff(pitches)
    return np.var(intervals) if len(intervals) > 0 else 0.0

def rhythmic_consistency(durations):
    """
    Evaluate how closely durations stick to standard rhythmic values
    (e.g., quarter note, eighth note).
    """
    # Defined standard durations (seconds or normalized)
    standard_durations = np.array([1, 0.5, 0.25, 0.125])
    rhythm_score = 0.0

    for d in durations:
        dist = np.min(np.abs(standard_durations - d))
        rhythm_score += 1 - dist  # Closer duration gets higher score

    avg_score = rhythm_score / len(durations) if len(durations) > 0 else 0
    return np.clip(avg_score, 0.0, 1.0)

def tonal_distance(pitches_onehot):
    """
    Estimate tonal distance for harmonicity between pitches —
    lower values = more harmonically consistent.
    Here using simplified pitch-class co-occurrence statistics.
    """
    pitches = np.argmax(pitches_onehot, axis=1)
    pitch_classes = pitches % 12
    unique, counts = np.unique(pitch_classes, return_counts=True)
    freqs = counts / counts.sum()
    entropy = -np.sum(freqs * np.log2(freqs + 1e-9))
    return entropy  # Using entropy as tonal distance proxy

def fitness_function(chromosome):
    """
    Composite fitness scoring mixing neural score, pitch entropy,
    interval smoothness, rhythmic consistency, and tonal distance.
    """

    pitches = chromosome[:, :pitches_onehot.shape[1]]
    durations = chromosome[:, PITCH_DIM]  # Example: duration vector
    neural_score = chromosome.fitness_neural  # Provided externally

    ent = pitch_entropy(pitches)
    iv = interval_variance(pitches)
    rc = rhythmic_consistency(durations)
    td = tonal_distance(pitches)

    # Weighted sum (weights can be tuned)
    fitness = 0.4 * neural_score + \
              0.15 * ent + \
              0.15 * (1.0 - np.tanh(iv / 10.0)) + \
              0.15 * rc + \
              0.15 * (1.0 - td / 4.0)  # Normalized tonal dist
    return fitness
