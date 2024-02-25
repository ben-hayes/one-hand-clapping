import time
from typing import Literal

import evotorch
import matplotlib.pyplot as plt
import torch
from evotorch.logging import Logger
import evotorch.operators as op

# import wandb

# matplotlib.use("TkAgg")
# plt.ion()


class MetricsLivePlotter(Logger):
    def __init__(self, searcher, target_status: str, max_generations: int):
        # Call the super constructor
        super().__init__(searcher)

        # Set up the target status
        self._target_status = target_status
        self.max_generations = max_generations

        # Create a figure and axis
        self._fig = plt.figure(figsize=(10, 4), dpi=80)

        self._ax = self._fig.add_subplot(111)

        # Set the labels of the x and y axis
        self._ax.set_xlabel("iter")
        self._ax.set_ylabel(target_status)

        # Create a line with (initially) no data in it
        (self._line,) = self._ax.plot([], [])

        # Update the TkAgg window name to something more interesting
        self._fig.canvas.manager.window.title(f"LivePlotter: {target_status}")

        self._iter_hist = []
        self._status_hist = []

    def _log(self, status: dict):
        # Update the histories of the status
        self._iter_hist.append(status["iter"])
        self._status_hist.append(status[self._target_status])

        # Update the x and y data
        self._line.set_xdata(self._iter_hist)
        self._line.set_ydata(self._status_hist)

        # Rescale the limits of the x and y axis
        self._ax.set_xlim(0.99, status["iter"])
        self._ax.set_ylim(min(self._status_hist) * 0.99, max(self._status_hist) * 1.01)

        # Draw the figure and flush its events
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

        # Sleeping here will make the updates easier to watch
        time.sleep(0.05)
        if status["iter"] == self.max_generations:
            plt.show()
            plt.pause(5)


class FindMySound(evotorch.Problem):
    def __init__(
        self,
        objective_sense: str,
        num_params: int,
        min_value: float,
        max_value: float,
        fitness_function: torch.nn.Module,
        need_bounds: bool = False,
        device: str = "cpu",
    ):
        """
        Args:
            objective_sense (str): The objective sense of the problem, either
                "max" or "min"
            num_params (int): The number of parameters in the solution; in our case,
                the number of paramters in the VST
            min_value (float): The minimum value of each parameter
            max_value (float): The maximum value of each parameter
            fitness_function (torch.nn.Module): The fitness function to be used for
                optimisation
            batch_size (int): The batch size for the fitness function
        """
        if need_bounds:
            super().__init__(
                objective_sense=objective_sense,
                solution_length=num_params,
                initial_bounds=(min_value, max_value),
                bounds=(min_value, max_value),
                device=device,
            )
        else:
            super().__init__(
                objective_sense=objective_sense,
                solution_length=num_params,
                initial_bounds=(min_value, max_value),
                device=device,
            )
        self.fitness_fn = fitness_function

    def fitness_function(self, params_tensor: torch.Tensor):
        return self.fitness_fn.compute(params_tensor)

    def _evaluate_batch(self, solutions: evotorch.SolutionBatch):
        fitness = self.fitness_function(solutions)
        solutions.set_evals(fitness)
        return solutions


# build a live custom logger that plots the population in real time. You can use
# dimensionality reduction to plot the population in 3D


class EvolutionarySearch:
    def __init__(
        self,
        problem: evotorch.Problem,
        logger: str = "StdOutLogger",
        searcher: str = "SNES",
        population_size: int = 10,
        max_generations: int = 10,
        stdev_init: float = 1.0,
        crossover: Literal["one_point", "two_point", "sbx"] = "one_point",
        mutation: Literal[
            "gaussian_sparse_narrow",
            "gaussian_sparse_wide",
            "gaussian_dense_narrow",
            "gaussian_dense_wide",
            "polynomial_sparse",
            "polynomial_dense",
        ] = "gaussian",
    ):
        """
        Args:
            problem (evotorch.Problem): The problem to be solved
            logger (str): The logger to be used
            searcher (str): The searcher to be used
            population_size (int): The size of the population
            max_generations (int): The maximum number of generations
            initial_solution_path (str): The path to the initial solution
            batch_size (int): The batch size for the fitness function
            stdev_init (float): The initial standard deviation for the SNES algorithm
            objective_sense (str): The objective sense of the problem,
                either "max" or "min"
        """

        self._problem = problem
        if searcher == "SNES":
            self._searcher = evotorch.algorithms.SNES(
                self._problem, stdev_init=stdev_init, popsize=population_size
            )
        elif searcher == "GeneticAlgorithm":
            # make sure need_bounds is set to True in Problem class
            crossover = (
                op.OnePointCrossOver(
                    self._problem,
                    tournament_size=8,
                )
                if crossover == "one_point"
                else (
                    op.TwoPointCrossOver(self._problem, tournament_size=8)
                    if crossover == "two_point"
                    else op.SimulatedBinaryCrossOver(
                        self._problem, tournament_size=8, eta=10.0
                    )
                )
            )

            mutation = (
                op.GaussianMutation(
                    self._problem,
                    stdev=0.1 if "narrow" in mutation else 1.0,
                    mutation_probability=0.1 if "sparse" in mutation else 0.9,
                )
                if "gaussian" in mutation
                else op.PolynomialMutation(
                    self._problem,
                    mutation_probability=0.1 if "sparse" in mutation else 0.9,
                )
            )
            self._searcher = evotorch.algorithms.GeneticAlgorithm(
                self._problem,
                popsize=population_size,
                operators=[
                    crossover,
                    mutation,
                ],
                re_evaluate=False,
            )
        else:
            raise NotImplementedError("Searcher not recognised")

        # self._population_size = population_size
        self._max_generations = max_generations

        if logger == "StdOutLogger":
            self._logger = evotorch.logging.StdOutLogger(self._searcher)
        elif logger == "Custom":
            self._logger = MetricsLivePlotter(
                self._searcher, "best_eval", max_generations
            )

        else:
            raise NotImplementedError("Logger not recognised")

    def run(self):
        self._searcher.run(self._max_generations)
        # for _ in range(self._max_generations):
        #     self._searcher.step()
        #     status = self._searcher.status
        #     print(status)
        #     population = self._searcher._population.access_values(keep_evals=True)
        #     print(population)

    # print("Search complete")
