import pickle
import numpy as np

ELITE_FILE = "elite_genomes.pkl"
elite_genomes = []


def save_to_archive(genome, fitness, generation=0, max_elites=8):
    """
    Save a genome to the elite archive with its fitness and generation.
    Keeps only the top `max_elites` genomes.
    """
    global elite_genomes

    # Sanitize fitness
    if fitness is None or not np.isfinite(fitness):
        fitness = 0.0

    # Remove duplicate if same genome key exists
    elite_genomes = [
        (g, f, gen) for (g, f, gen) in elite_genomes if g.key != genome.key
    ]

    # Add new genome
    elite_genomes.append((genome, float(fitness), generation))

    # Sort and trim
    elite_genomes.sort(key=lambda x: x[1], reverse=True)
    elite_genomes = elite_genomes[:max_elites]

    # Save to file
    try:
        with open(ELITE_FILE, "wb") as f:
            pickle.dump(elite_genomes, f)
    except Exception as e:
        print(f"❌ Failed to save elite archive: {e}")


def load_elite():
    """
    Load elites from file into memory.
    Returns a list of (genome, fitness, generation).
    """
    global elite_genomes
    try:
        with open(ELITE_FILE, "rb") as f:
            elite_genomes = pickle.load(f)
    except FileNotFoundError:
        elite_genomes = []
    except Exception as e:
        print(f"❌ Failed to load elite archive: {e}")
        elite_genomes = []

    # Sanitize fitness values
    sanitized = []
    for g, f, gen in elite_genomes:
        if f is None or not np.isfinite(f):
            f = 0.0
        sanitized.append((g, float(f), gen))

    elite_genomes = sanitized
    return elite_genomes
