from problem import EvolutionaryProblem
from utils import add_timer
import random

@add_timer
def BRKGA(problem: EvolutionaryProblem):
    solutions = []
    for i in range(problem.base_population):
        cost, sol = problem.generate_chromosome()
        solutions.append([cost, sol])

    for i in range(problem.generations):
        children = []
        for j in range(problem.children_by_generation):
            (x, y) = problem.selection(solutions)
            s = problem.crossover(x, y)
            if random.random() < problem.mutation_rate:
                s = problem.mutate(s)
            children.append(s)
        solutions = problem.update(solutions + children)

    best = min(solutions, key=lambda x: x[0])
    best[1] = problem.solution_array_to_dict(best[1])
    return [round(best[0], 2), best[1]]