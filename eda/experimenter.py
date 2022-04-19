import numpy as np
from matplotlib import pyplot as plt

from eda.optimizer import *
from ObjectiveFunction.objective_function import bin_to_float
import time
from time import strftime
from time import gmtime
import csv


class Experimenter(object):
    """
    A class that executes an independent experimental trial.
    """

    def __init__(self, objective, optim, max_num_evals=1e5, img_id="", logger=None):
        """
        Parameters
        ----------
        objective : eda.objective.objective_base.ObjectiveBase
            A objective function.
        optim : eda.optimizer.eda_base.EDABase
            A optimizer.
        max_num_evals : int, default 1e5
            Maximum number of evaluations.
        logger : eda.logger.Logger, default None
            A logger.
        """
        self.objective = objective
        self.img_id = img_id
        self.optim = optim
        self.best_ind_gen = []
        self.best_ind_all = []
        self.stop_condition = lambda o: o.num_evals > max_num_evals or o.is_convergence()
        self.logger = logger
        columns = ["step", "num_evals", "best_eval_gen", "best_indiv_gen",
                   "best_eval_all", "best_indiv_all", "convergence", "probability",
                   "order", "uni_freq", "bi_freq", "cluster",
                   "network_structure", "network_score"]

        self.logger.set_log_columns(columns)
        self.logger.set_display_columns(columns)

    def execute(self):
        is_success = False
        step = 0
        start = time.time()
        lap = time.time()
        iterations_best = []
        history = []

        while not self.stop_condition(self.optim):
            step += 1
            lam = self.optim.lam
            c = np.array([self.optim.sampling() for _ in range(lam)])
            #print(c)
            evals, obj_info = self.objective(c)
            optim_info = self.optim.update(c, evals)  # c es una matrix dimension x poblacion (4x2)

            if self.logger:
                self.log(c, evals, obj_info, optim_info, step, start, lap)
            best_eval = np.min(evals)
            history.append(best_eval)

            # a optimum was found
            if np.abs(best_eval - self.objective.optimal_value) < 1e-8:
                is_success = True
                if self.logger and step % self.logger.logging_step:
                    self.log(c, evals, obj_info, optim_info, step, start, lap, force=True)
                break

            last_lap = time.time()
        gap = last_lap - lap
        current_best = [self.img_id, str(self.optim)[0:4], self.optim.best_eval, str(self.best_ind_all),
                        strftime("%H:%M:%S", gmtime(gap))]
        iterations_best.append(current_best)
        # self.graphics(history)

        with open("./results/iterations_Best.csv", "a") as f1:
            fieldnames_best = ['Img', 'algorithm','best_eval_all', 'best_indiv_all', 'gap']
            writer_best = csv.DictWriter(f1, fieldnames=fieldnames_best)
            if f1.tell() == 0:
                writer_best.writeheader()
            writer_best = csv.writer(f1)
            writer_best.writerows(iterations_best)
            iterations_best = []
            print("Iterations best saved.")

        return is_success, step

    def log(self, c, evals, obj_info, optim_info, step, start, lap, force=False):
        iterations = []
        previous_lap = lap
        lap = time.time()
        gap = lap - previous_lap
        best_idx = np.argmin(evals)
        # print(c[best_idx][:,1])
        self.logger.add("step", step, step, force=force)
        self.logger.add("num_evals", self.optim.num_evals, step, force=force)
        self.logger.add("best_eval_gen", evals[best_idx], step, force=force)
        ci = bin_to_float(1*c[best_idx][:, 1][0:10])
        bi = bin_to_float(1*c[best_idx][:, 1][10:20])
        si = bin_to_float(1*c[best_idx][:, 1][20:30])
        clri = bin_to_float(1*c[best_idx][:, 1][30:40])
        self.best_ind_gen = [ci,bi,si,clri]
        self.logger.add("best_indiv_gen", str(self.best_ind_gen), step, force=force)
        self.logger.add("best_eval_all", str(self.optim.best_eval), step, force=force)
        c = bin_to_float(1*self.optim.best_indiv[:, 1][0:10])
        b = bin_to_float(1*self.optim.best_indiv[:, 1][10:20])
        s = bin_to_float(1*self.optim.best_indiv[:, 1][20:30])
        clr = bin_to_float(1*self.optim.best_indiv[:, 1][30:40])
        self.best_ind_all = [c,b,s,clr]
        # print('Mejor = ', self.best_ind_all)
        self.logger.add("best_indiv_all", str(self.best_ind_all), step, force=force)
        self.logger.add("convergence", self.optim.convergence(), step, force=force)
        if isinstance(self.optim, (PBIL, UMDA, CGA, ECGA)):
            self.logger.add("probability", self.optim.theta, step, force=force)
        self.logger.output(step, force=force)
        current = [self.img_id, step, self.optim.convergence(), self.optim.best_eval, self.best_ind_all,
                   evals[best_idx], self.best_ind_gen, strftime("%H:%M:%S", gmtime(gap))]
        iterations.append(current)

        with open("./results/iterations.csv", "a") as f:
            fieldnames = ['Img', 'step', 'Convergence', 'best_eval_all', 'best_indiv_all', 'best_eval_gen',
                          'best_indiv_gen',
                          'gap']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if f.tell() == 0:
                writer.writeheader()
            writer = csv.writer(f)
            writer.writerows(iterations)
            iterations = []
            #print("Iterations saved.")

    def graphics(self, history):

        relative_plot = []
        mx = 0
        for i in range(len(history)):
            if history[i] > mx:
                mx = history[i]
                relative_plot.append(mx)
            else:
                relative_plot.append(mx)

        plt.figure(figsize=(14, 6))

        ax = plt.subplot(121)
        ax.plot(list(range(1,len(history)+1)), history)
        ax.title.set_text('Local cost found')
        ax.set_xlabel('iteration')
        ax.set_ylabel('SCORE')

        ax = plt.subplot(122)
        ax.plot(list(range(1, len(relative_plot)+1)), relative_plot)
        ax.title.set_text('Best global cost found')
        ax.set_xlabel('iteration')
        ax.set_ylabel('SCORE')

        plt.show()
