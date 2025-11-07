import pickle

ELITE_FILE = "elite_genomes.pkl"
elite_genomes = []


def save_to_archive(genome, fitness):
    global elite_genomes
    elite_genomes.append((genome, fitness))
    # Sort by fitness descending
    elite_genomes = sorted(elite_genomes, key=lambda x: x[1], reverse=True)
    # Keep only top 5 genomes
    elite_genomes = elite_genomes[:5]
    # Save to file
    with open(ELITE_FILE, "wb") as f:
        pickle.dump(elite_genomes, f)


def load_elite():
    global elite_genomes
    try:
        with open(ELITE_FILE, "rb") as f:
            elite_genomes = pickle.load(f)
    except FileNotFoundError:
        elite_genomes = []
