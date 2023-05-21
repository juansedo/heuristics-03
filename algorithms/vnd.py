from typing import Dict, List
from utils import add_timer
from problem import LocalSearchProblem

@add_timer
def VND(problem: LocalSearchProblem, initial_solution: Dict[int, List[int]]):
    j = 0
    best_solution = initial_solution
    
    hoods = [
        problem.swapping,
        problem.external_swapping,
        problem.external_insertion,
        problem.insertion,
    ]

    while j < len(hoods):
        best_neighbor = hoods[j](best_solution)
        if problem.Z(best_neighbor) < problem.Z(best_solution):
            j = 0
            best_solution = best_neighbor
        else:
            j += 1

    final_Z = problem.Z(best_solution)
    return [round(final_Z, 2), best_solution]
