import torch
import evotorch
from EvoCLAP.fitness import FitnessFunction  # Assuming FitnessFunction is a module with a class or function named 'FitnessFunction'
from evotorch.callbacks import DefaultLogger  # Importing the default logger from evotorch


class FindMySound(evotorch.Problem):
    def __init__(self, 
                 num_params: int, 
                 min_value: float, 
                 max_value: float, 
                 fitness_function: torch.nn.Module, 
                 batch_size: int = 1):
        super().__init__(
            objective_sense="max",
            solution_length=num_params,
            initial_bounds=(min_value, max_value),
        )
        self.fitness_fn = fitness_function
        self._batch_size = batch_size

    def fitness_function(self, params_tensor: torch.Tensor):
        return self.fitness_fn.compute(params_tensor)

    def _evaluate_batch(self, solutions: evotorch.SolutionBatch):
        batched_solutions = solutions.values.split(self._batch_size)
        batched_fitness = []
        for batch in batched_solutions:
            fitness = self.fitness_function(batch)
            batched_fitness.append(fitness)
        fitness = torch.cat(batched_fitness)
        solutions.set_evals(fitness)


class EvoCLAP(evotorch.Algorithm):
    def __init__(self,
                 problem: evotorch.Problem,
                 logger: evotorch.callbacks.Callback = DefaultLogger(),  # Using DefaultLogger
                 searcher_class: torch.nn.Module = evotorch.algorithms.SNES,  # Use provided searcher class
                 population_size: int = 100,
                 max_generations: int = 100,
                 initial_solution_path=None):
        
        self._problem = problem
        # https://docs.evotorch.ai/v0.1.1/reference/evotorch/algorithms/distributed/gaussian/#evotorch.algorithms.distributed.gaussian.SNES.__init__

        self._searcher = searcher_class(self._problem, stdev_init=5)  # Use provided searcher class
        self._population_size = population_size
        self._max_generations = max_generations
        self._population = self._initialize_population()
        self._logger = logger
        self.initial_solution_path = initial_solution_path

    def _initialize_population(self):
        if self.initial_solution_path is not None:
            initial_solution = torch.load(self.initial_solution_path)
            return evotorch.SolutionBatch(
                problem=self._problem,
                size=self._population_size,
                values=initial_solution
            )
        else:
            return evotorch.SolutionBatch(
                problem=self._problem,
                size=self._population_size,
            )

    def run(self):
        for generation in range(self._max_generations):
            self._searcher.run(1)
            self._logger.log_generation(self._searcher)
        return self._searcher.best_solution
