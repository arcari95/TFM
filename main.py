import argparse
import time
import os
import random
import numpy as np
from eda.builder import build_logger, build_objective, build_optimizer
from eda.experimenter import Experimenter


"""Parameter configuration"""
parser = argparse.ArgumentParser(description='Compute optimization.')

parser.add_argument('--img_directory', type=str, default="D:/images/",
                    help='Image directory')
parser.add_argument('--dataset_path', type=str, default="NeuralNetwork/data/AVA_sample3.txt",
                    help='Dataset path')
parser.add_argument('--before_after_path', type=str, default="./results/before_after.csv",
                    help='Before After Path')
parser.add_argument('--iterations_path', type=str, default="./results/iterations.csv",
                    help='Iterations Path')
parser.add_argument('-single_mode', action='store_true',
                    help='Batch mode or single mode')
parser.add_argument('--img_path', type=str, default="",
                    help='Image path for the single mode')

parser.add_argument("--objective-type", type=str,
                    choices=["ObjectiveFunction"], default="ObjectiveFunction",
                    help="specify a objective function.")
parser.add_argument("--dim", type=int, default=40,
                    help="dimension of the objective function (variables).")
parser.add_argument("--minimize", action="store_false",
                    help="the problem is whether minimization problem or maximization problem.")
parser.add_argument("--optim-type", type=str,
                    choices=["umda", "pbil", "cga", "ecga"], default="ecga",
                    help="specify a optimizer.")
parser.add_argument("--lam", type=int, default=20,
                    help="population size.")
parser.add_argument("--lr", type=float, default=0.5,
                    help="learning rate in PBIL and UMDA.")
parser.add_argument("--negative-lr", type=float, default=None,
                    help="learning rate for negative example in PBIL.")
parser.add_argument("--selection", type=str, default="roulette",
                    choices=["none", "block", "tournament", "roulette", "top"],
                    help="selection method which chooses individuals from a population based on the evaluation value.")
parser.add_argument("--selection-rate", type=float, default=0.1,
                    help="selection rate, i.e., how many individuals are chosen when the selection method is applied to a population.")
parser.add_argument("--tournament-size", type=int, default=2,
                    help="tournament size in the tournament selection.")
parser.add_argument("--selection-criterion", type=str, default="rank",
                    choices=["eval", "rank"],
                    help="criterion to generate a roulette in roulette selection.")
parser.add_argument("--with-replacement", action="store_true",
                    help="sampling with replacement or not")
parser.add_argument("--mutation-prob", type=float, default=0.5,
                    help="mutation probability in PBIL.")
parser.add_argument("--mutation-shift", type=float, default=0.05,
                    help="amount of shift for mutation in PBIL.")
parser.add_argument("--replacement", type=str, default="truncation",
                    choices=["truncation", "restricted"],
                    help="replacement method which replaces individuals in parent population with ones in candidate population.")
parser.add_argument("--replace-rate", type=float, default=0.5,
                    help="replacement rate, i.e., how many individuals are replaced when the replacement method is applied to a population.")
parser.add_argument("--window-size", type=int, default=2,
                    help="a user parameter, which determines trade-off between the goodness and the diversity in the population, in the restricted tournament replacement.")
parser.add_argument("--max-num-evals", type=int, default=1000,
                    help="Maximum number of evaluations.")

parser.add_argument("--seed", type=int, default=19, # original = -1
                    help="a random number seed for trials.")
parser.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="a random number seed for each trials. the length of the seeds must match the number of trials.")
parser.add_argument("--trials", type=int, default=1,
                    help="how many independent trials.")

parser.add_argument("--log-dir", type=str, default=None,
                    help="directory path to output logs.")
parser.add_argument("--logging-step", type=int, default=1,
                    help="interval of outputting logs to directory.")
parser.add_argument("--display-step", type=int, default=1,
                    help="interval of displaying logs to stdout.")

args = parser.parse_args()

iterations = []
before_after = []
step = 0
image_ids = np.loadtxt(args.dataset_path, dtype=int)

my_file = "./results/iterations_Best.csv"
my_file1 = "./results/iterations.csv"

# check if file exists
if os.path.exists(my_file):
    os.remove(my_file)
if os.path.exists(my_file1):
    os.remove(my_file1)

def main():
    # generate a random seed of each trial
    if args.seeds is None:
        if args.seed < 0:
            args.seed = np.random.randint(2 ** 31)
        print("seed", args.seed)
        np.random.seed(args.seed)
    seeds = list(np.random.randint(0, 2 ** 31, args.trials))

    assert args.trials == len(seeds), \
        "The length of the seeds ({}) must match the number of trials ({}).\n".format(len(seeds), args.trials)
    # build each component
    logger = build_logger(args)
    logger.info("building now...")
    objective = build_objective(args, image_id=" ")

    logger.info("built an objective function: {}".format(objective))
    optim_dummy = build_optimizer(args, objective)
    logger.info("built an optimizer: {}".format(optim_dummy))

    for image in range(0, image_ids.size-1):

        print("IMAGE", image + 1, "OF 1000.")
        image_id = image_ids[image]
        objective = build_objective(args, image_id)
        # independent trials
        iters, best_evals, num_evals, times = [], [], [], []
        for i, seed in enumerate(seeds, 1):
            logger.open(i)
            # set a random seed
            np.random.seed(seed)
            random.seed(seed)

            optim = build_optimizer(args, objective)
            start = time.time()

            exp = Experimenter(objective=objective, optim=optim,
                               max_num_evals=args.max_num_evals,
                               img_id=image_id,
                               logger=logger)
            success, iteration = exp.execute()

            if success:
                iters.append(iteration)
                num_evals.append(optim.num_evals)
                times.append(time.time() - start)
            best_evals.append(optim.best_eval)
            info = {
                "success": success,
                "iters": iteration,
                "num_evals": optim.num_evals,
                "elapsed_time": time.time() - start,
                "best_eval": optim.best_eval,
                "best_indiv": np.argmax(optim.best_indiv, axis=-1),
            }
            logger.result(info)
            logger.info("Finished {}/{}".format(i, args.trials))
            logger.close()

        if len(iters):
            logger.info("Success rate\t{:.2f}  ({}/{})".format(len(iters) / args.trials, len(iters), args.trials))
            logger.info("Iterations\t{:.2f}±{:.2f}".format(np.mean(iters), np.std(iters)))
            logger.info("Number of evaluations\t{:.2f}±{:.2f}".format(np.mean(num_evals), np.std(num_evals)))
            logger.info("Elapsed time\t{:.2f}±{:.2f}".format(np.mean(times), np.std(times)))
        logger.info("Best evaluation\t{:.2f}±{:.2f}".format(np.mean(best_evals), np.std(best_evals)))


if __name__ == "__main__":
    main()
