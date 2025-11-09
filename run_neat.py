import neat
import cv2
import numpy as np
import traceback
from car_env import CarEnv
from elite_archive import save_to_archive, load_elite

EXIT_FLAG = False  


def eval_genomes(genomes, config, population=None):
    global EXIT_FLAG
    if EXIT_FLAG:
        return

    cars = []
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        env = CarEnv()
        if population is not None:
            env.current_generation = population.generation

        state = env.reset()
        cars.append({
            'genome': genome,
            'net': net,
            'env': env,
            'state': state,
            'fitness': 0.0,
            'done': False,
            'steps': 0
        })

    vis_env = CarEnv()
    gen = population.generation if population is not None else 0
    if gen <= 50:
        step_limit = 8000
    elif gen <= 75:
        step_limit = 7000
    else:
        step_limit = 6000

    active_cars = len(cars)

    while active_cars > 0 and not EXIT_FLAG:
        car_positions = []

        for car in cars:
            if not car['done'] and car['steps'] < step_limit:
                outputs = car['net'].activate(car['state'])
                action_idx = int(np.argmax(outputs))

                car['state'], reward, car['done'] = car['env'].step_discrete(
                    action_idx)
                car['fitness'] += reward
                car['steps'] += 1

                if not car['done']:
                    car_positions.append(
                        (car['env'].car_pos, car['env'].car_angle))

                if car['steps'] >= step_limit or car['done']:
                    car['done'] = True
                    active_cars -= 1

            elif not car['done']:
                car_positions.append(
                    (car['env'].car_pos, car['env'].car_angle))

        key = vis_env.render(car_positions)

        if key == ord('q') or key == 27:
            EXIT_FLAG = True
            cv2.destroyAllWindows()
            break
        elif key == ord('s'):
            print("Skipping generation...")
            break

    for car in cars:
        if car['fitness'] is None or not np.isfinite(car['fitness']):
            car['fitness'] = 0.0
        car['genome'].fitness = float(car['fitness'])

        if population is not None:
            save_to_archive(car['genome'], car['fitness'],
                            population.generation)
        else:
            save_to_archive(car['genome'], car['fitness'])


def run(config_file):
    global EXIT_FLAG

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))

    elites = load_elite()
    for genome, fitness, gen in elites:
        if fitness is None or not np.isfinite(fitness):
            fitness = 0.0
        genome.fitness = float(fitness)
        population.population[genome.key] = genome
        p = neat.Population(config)

    population.species.speciate(config, population.population, generation=0)

    while not EXIT_FLAG:
        try:
            winner = population.run(
                lambda g, c: eval_genomes(g, c, population), 1)
        except Exception as e:
            print(f"An error occurred: {e}")
            traceback.print_exc()
            break

        if population.generation >= 120:
            break

    cv2.destroyAllWindows()
    print("\nBest genome found:\n", winner if 'winner' in locals()
          else "No winner found (early exit).")


if __name__ == "__main__":
    run("neat_config.txt")
