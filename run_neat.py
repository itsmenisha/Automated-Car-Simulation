import neat
import cv2  # Import cv2 for destroyAllWindows
import numpy as np
from car_env import CarEnv
import pickle
from elite_archive import save_to_archive

# Global flag to signal a clean exit from the training loop
EXIT_FLAG = False


def eval_genomes(genomes, config):
    global EXIT_FLAG
    if EXIT_FLAG:
        return  # Skip evaluation if we're quitting

    cars = []
    nets = []
    envs = []

    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        env = CarEnv()
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

    # Shared environment for visualization
    vis_env = CarEnv()

    step_limit = 500  # Max steps per run
    active_cars = len(cars)

    while active_cars > 0 and not EXIT_FLAG:
        # List to store positions of all active cars for visualization
        car_positions = []

        # Update all cars
        for car in cars:
            if not car['done'] and car['steps'] < step_limit:
                # Get neural network output (discrete actions)
                outputs = car['net'].activate(car['state'])
                # Choose the action with highest output
                try:
                    action_idx = int(np.argmax(outputs))
                except Exception:
                    action_idx = 4

                # Update car using the discrete action step helper
                car['state'], reward, car['done'] = car['env'].step_discrete(
                    action_idx)
                car['fitness'] += reward
                car['steps'] += 1

                # Add position for visualization if car is still active
                if not car['done']:
                    car_positions.append(
                        (car['env'].car_pos, car['env'].car_angle))

                # Check if car is done
                if car['done'] or car['steps'] >= step_limit:
                    active_cars -= 1

                    # Bonus for surviving full duration
                    if not car['done'] and car['steps'] >= step_limit:
                        car['fitness'] += 10.0

            # If car is done but still in simulation, add its final position
            elif not car['done']:
                car_positions.append(
                    (car['env'].car_pos, car['env'].car_angle))

        # Render all active cars
        key = vis_env.render(car_positions)

        # Check for exit
        if key == ord('q') or key == 27:
            EXIT_FLAG = True
            cv2.destroyAllWindows()
            break

    # Update fitness for all genomes
    for car in cars:
        car['genome'].fitness = car['fitness']
        # Save to elite archive if good enough
        save_to_archive(car['genome'], car['fitness'])


def run(config_file):
    global EXIT_FLAG

    config = neat.Config(neat.DefaultGenome,
                         neat.DefaultReproduction,
                         neat.DefaultSpeciesSet,
                         neat.DefaultStagnation,
                         config_file)

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())

    while not EXIT_FLAG:
        try:
            # Run for 1 generation at a time
            winner = population.run(eval_genomes, 1)
        except Exception as e:
            print(f"An error occurred: {e}")
            break

        # We manually break the loop after 50 generations or if EXIT_FLAG is set
        if population.generation >= 600:
            break

    # Clean up at the end
    cv2.destroyAllWindows()
    print("\nBest genome found:\n", winner if 'winner' in locals()
          else "No winner found (early exit).")


if __name__ == "__main__":
    run("neat_config.txt")
# thissssss
