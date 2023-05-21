#%% Algorithm
import ast
import argparse
import random
import matplotlib.pyplot as plt
import pandas as pd
from algorithms.vnd import VND
from algorithms.genetic import BRKGA
from utils import get_test_file, generate_distance_matrix, paths_and_time_plot, comparison_boxplot
from files import ExcelBook
from problem import Problem, EvolutionaryProblem, LocalSearchProblem


def solveGenetic(problem: EvolutionaryProblem):
    # Genetic Algorithm with Biased Random Key Generation
    ([best_cost_genetic, best_sol_genetic], time_genetic) = BRKGA(problem)
    problem.add_to_table(problem.generations, best_cost_genetic, best_sol_genetic, time_genetic, phase='GA-Genetic')
    
    # Local Search with Variable Neighborhood Descent (VND)
    ls_problem = LocalSearchProblem(
        distance_matrix=problem.distance_matrix,
        demands=problem.demands,
        n=problem.n,
        R=problem.R,
        Q=problem.Q,
        Th=problem.Th,
        data=problem.data
    )
    ([best_cost_vnd, best_vnd], time_vnd) = VND(ls_problem, best_sol_genetic)
    problem.add_to_table(problem.generations, best_cost_vnd, best_vnd, time_vnd, phase='GA-VND')
    
    ## Reduce penalized time (Th) using Constructive Sorting
    [best_cost, best_sol, time_sorting] = [best_cost_vnd, best_vnd, 0]
    
    best_distances = problem.get_distance_array(best_sol)
    sol = [best_sol, best_distances, time_genetic + time_vnd + time_sorting]
    
    return best_cost, sol

def main():
    parser = argparse.ArgumentParser(description= "Heuristics local search algorithms (by juansedo)")
    parser.add_argument("-f", "--file", nargs="+", type=int, help="File id to read the data from")
    parser.add_argument("-s", "--seed", type=str, help="String for the random seed")
    parser.add_argument("--summary", action="store_true", help="Generate the summary file")
    args = parser.parse_args()
    if (args.file): args.file = list(set(args.file))
    if (args.seed): random.seed(args.seed)
    test_params = [
        ('I', 250, 100, 0.2, 0.25),
        ('P1', 250, 1000, 0.2, 0.25),
        ('G1', 1000, 100, 0.2, 0.25),
        ('M1', 250, 100, 0.8, 0.25),
        ('E1', 250, 100, 0.2, 0.8),
    ]

    if (args.summary):
        author = 'JSDIAZO'
        df = pd.read_csv('./outputs/summary.csv', index_col=0)
        geneticExcel = ExcelBook(f'mtVRP_{author}_BRKGA.xls')
        for i in range(1, 12):
            headers, data = get_test_file(i)
            n, R, Q, Th = headers
            distance_matrix, demands = generate_distance_matrix(data, n)
            problem = Problem(
                distance_matrix=distance_matrix,
                demands=demands,
                n=n,
                R=R,
                Q=Q,
                Th=Th,
                data=data,
            )

            filtered_item = df[df['instance'] == f'mtVRP{i}'].sort_values(by=['bks_gap'])
            if len(filtered_item) > 0:
                filtered_item = filtered_item.iloc[0]
                filtered_set = df[(df['instance'] == f'mtVRP{i}') & (df['test_type'] == filtered_item['test_type'])]
                best_solution = ast.literal_eval(filtered_item['solution'])
                best_time = round(filtered_set['time_elapsed'].sum(), 2)

                geneticExcel.add_sheet(i, [best_solution, problem.get_distance_array(best_solution), best_time], Th)
        comparison_boxplot(df, test_params)
        geneticExcel.save()
        exit()

    iterator = args.file if args.file else [1]
    for i in iterator:
        for (test_type, generations, base_population, mutation_rate, elitist_rate) in test_params:
            headers, data = get_test_file(i)
            n, R, Q, Th = headers
            distance_matrix, demands = generate_distance_matrix(data, n)
            
            problem = EvolutionaryProblem(
                distance_matrix=distance_matrix,
                demands=demands,
                n=n,
                R=R,
                Q=Q,
                Th=Th,
                data=data,
                generations=generations,
                base_population=base_population,
                children_by_generation=int(base_population * (1 - elitist_rate)),
                mutation_rate=mutation_rate,
                elitist_rate=elitist_rate,
                index=i,
                test_type=test_type
            )
            cost, sol = solveGenetic(problem)

        # Get best solution from problem.table() dataframe and plot it
        best = problem.table()[problem.table()['instance'] == f'mtVRP{i}'].sort_values(by=['bks_gap']).iloc[0]
        best_test_type = best['test_type']
        sol = [best['solution'], problem.get_distance_array(best['solution']), best['time_elapsed']]
        paths_and_time_plot(f'BRKGA + VND mtVRP{i} ({best_test_type})', data, cost, sol, Th, problem.calculate_penalized_time(sol[0]), i)

    problem.table().to_csv('./outputs/results.csv', index_label='index')
    problem.print_table()

if __name__ == "__main__":
    main()
