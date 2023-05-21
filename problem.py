from typing import List, Dict
import pandas as pd
import random
import copy
from utils import BKS, LB

genetic_df = pd.DataFrame(columns=[
    'instance',
    'test_type',
    'phase',
    'time_elapsed',
    'gens',
    'cg',
    'population',
    'mr',
    'er',
    'value',
    'bks_gap',
    'lb_gap',
    'penalized_time',
    'solution'
])

class Problem:
    def __init__(self, distance_matrix, demands, n, R, Q, Th, data):
        self.distance_matrix = distance_matrix
        self.demands = demands
        self.n = n
        self.R = R
        self.Q = Q
        self.Th = Th
        self.data = data
        
    def get_optimal(self, possible_paths: list):
        optimal = possible_paths[0]
        for path in possible_paths:
            if self.calculate_path_distance(path) < self.calculate_path_distance(optimal):
                optimal = path
        return optimal

    def get_optimal_solution(self, solutions: List[dict]):
        optimal = solutions[0]
        for sol in solutions:
            if self.Z(sol) < self.Z(optimal):
                optimal = sol
        return optimal

    def check_consistency(self, path: list):
        amount = self.Q
        for i in range(len(path) - 1):
            actualNode, nextNode = [path[i], path[i + 1]]
            if actualNode == 0: amount = self.Q
            if nextNode == 0: continue
            amount -= self.demands[nextNode]
            if amount < 0:
                return False
        return True

    def check_consistency_solution(self, solution: dict):
        values: List[bool] = [self.check_consistency(path) for path in solution.values()]
        return all(values)

    def calculate_path_distance(self, path: list):
        return sum([self.distance_matrix[path[i]][path[i + 1]] for i in range(len(path) - 1)])
    
    def get_distance_array(self, solution: dict):
        return [self.calculate_path_distance(path) for path in solution.values()]
    
    def calculate_penalized_time(self, solution: Dict[int, List[int]]):
        time = 0
        for path in solution.values():
            t = self.calculate_path_distance(path)
            if t > self.Th:
                time += t - self.Th
        return round(time, 2)

    def Z(self, solution: Dict[int, List[int]]):
        """Returns the objective function value of a solution"""
        total = 0
        for path in solution.values():
            value = self.calculate_path_distance(path)
            total += round(value, 2)
        return total
    
    def solution_array_to_dict(self, solution: list):
        solution_dict = {t: [0] for t in range(self.R)}
        truck = 0
        for i in range(1, len(solution)):
            if solution[i] == 0:
                if solution_dict[truck][-1] != 0: solution_dict[truck] += [0]
                truck += 1
                truck %= self.R
                if solution_dict[truck][-1] != 0: solution_dict[truck] += [0]
                continue
            else:
                solution_dict[truck].append(solution[i])
        return solution_dict
    
    def solution_dict_to_array(self, solution: Dict[int, List[int]]):
        solution_array = [0]
        for path in solution.values():
            solution_array += path[1:]
        return solution_array


class EvolutionaryProblem(Problem):
    def __init__(self, test_type='C1', index=1, generations=100, children_by_generation=40, base_population=50, mutation_rate=0.1, elitist_rate=0.25, *args, **kwargs):
        # Params
        self.index = index
        self.generations = generations
        self.children_by_generation = children_by_generation
        self.base_population = base_population
        self.mutation_rate = mutation_rate
        self.elitist_rate = elitist_rate
        self.test_type = test_type
        self.bks = BKS.get_by_index(index)
        self.lb = LB.get_by_index(index)
        
        # Dataframe
        # cg: children by generation
        # mr: mutation rate
        self.formatters = {
            'time_elapsed': '{:.2f} s'.format,
            'bks_gap': '{:.2f} %'.format,
            'lb_gap': '{:.2f} %'.format,
        }
        super(EvolutionaryProblem, self).__init__(*args, **kwargs)
    
    def generate_chromosome(self):
        chromosome = [i + 1 for i in list(range(self.n))]
        random.shuffle(chromosome)
        cost, solution = self.fitness(chromosome)
        return cost, solution

    def selection(self, solutions):
        # Elitist selection
        fits = sorted(solutions, key=lambda i: i[0], reverse=True)[:int(self.base_population * self.elitist_rate)]
        x = random.choice(fits)
        
        # Roulette wheel selection
        fitness_total = sum([i[0] for i in solutions])
        fitness_weights = [i[0] / fitness_total for i in solutions]
        y = random.choices(solutions, weights=fitness_weights)[0]
        return x, y

    def crossover(self, c1, c2):
        random_c1 = self.get_random_key(c1[1])
        random_c2 = self.get_random_key(c2[1])
        
        half = len(random_c1) // 2
        selected = random_c1[:half] + random_c2[half:]        
        _sorted = selected.copy()
        _sorted.sort()
        new_child = [_sorted.index(i) + 1 for i in selected]
        return new_child

    def mutate(self, solution):
        # Simple swapping
        new_solution = solution.copy()
        i, j = random.sample(range(1, len(solution) - 1), 2)
        new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
        return new_solution
    
    def update(self, solutions):
        # Elitist selection
        fits = sorted(solutions, key=lambda i: i[0], reverse=True)[:self.base_population]
        return fits
    
    def fitness(self, solution):
        new_sol = [0, solution[0]]
        cost = self.distance_matrix[0][solution[0]]
        i = 0
        capacity = self.Q
        for i in range(len(solution) - 1):
            actualNode, nextNode = [solution[i], solution[i + 1]]
            if capacity >= self.demands[nextNode]:
                capacity -= self.demands[nextNode]
                cost += self.distance_matrix[actualNode][nextNode]
                new_sol += [nextNode]
            else:
                capacity = self.Q - self.demands[nextNode]
                cost += self.distance_matrix[actualNode][0] + self.distance_matrix[0][nextNode]
                new_sol += [0, nextNode]
        cost += self.distance_matrix[solution[-1]][0]
        new_sol += [0]
        return cost, new_sol
    
    def get_random_key(self, solution):
        random_key = []
        truck = -1
        for i in range(len(solution)):
            if solution[i] == 0:
                truck += 1
                truck %= self.R
                continue
            random_key.append(round(truck + random.random(), 4))
        return random_key
    
    def add_to_table(self, generations, cost, solution, time_elapsed, phase='Genetic'):
        bks_gap = round(((cost - self.bks) / self.bks) * 100, 2) 
        lb_gap = round(((cost - self.lb) / self.lb) * 100, 2)
        penalized_time = self.calculate_penalized_time(solution)

        genetic_df.loc[len(genetic_df)] = {
            'instance': f'mtVRP{self.index}',
            'test_type': self.test_type,
            'phase': phase,
            'time_elapsed': time_elapsed,
            'gens': generations,
            'cg': self.children_by_generation,
            'population': self.base_population,
            'mr': self.mutation_rate,
            'er': self.elitist_rate,
            'value': cost,
            'bks_gap': bks_gap,
            'lb_gap': lb_gap,
            'penalized_time': penalized_time,
            'solution': solution
        }
    
    def print_table(self, columns=['instance', 'test_type', 'phase', 'time_elapsed', 'penalized_time', 'bks_gap', 'lb_gap', 'value']):
        table = genetic_df[columns].to_string(formatters=self.formatters)
        print(table)
    
    def table(self):
        return genetic_df

class LocalSearchProblem(Problem):
    def _handle_solution(self, solution: dict, transformer, rand: bool = False):
        truck = 0
        truck_offset = 1
        best_solution = {}
        hood = []

        i = 1
        j = 1

        while truck_offset < len(solution):
            if (i < len(solution[truck_offset]) - 1 and solution[truck_offset][i] == 0): i += 1
            if (j < len(solution[truck_offset]) - 1 and solution[truck_offset][j] == 0): j += 1
            if (j >= len(solution[truck_offset]) - 1):
                i += 1
                j = 1
            if (i >= len(solution[truck]) - 1):
                truck_offset += 1
                i = 1
                j = 1
                continue
            new_sol = transformer(solution, truck, truck_offset, i, j)
            
            if self.check_consistency_solution(new_sol):
                hood.append(new_sol)
            j += 1

        # Noise
        if rand:
            temp = []
            for i in range(len(hood)):
                if random.random() < 0.7:
                    temp.append(hood[i])
            hood = temp
        
        if len(hood) > 0:
            best_solution = self.get_optimal_solution(hood)
        else:
            best_solution = solution
        return best_solution
    
    def _external_swapping(self, solution, truck, truck_offset, i, j):
        try:
            new_sol = copy.deepcopy(solution)
            elem1 = new_sol[truck][i]
            new_sol[truck][i] = new_sol[truck_offset][j]
            new_sol[truck_offset][j] = elem1
            return new_sol
        except:
            print(f'WARNING: ({truck},{truck_offset},{i},{j}) ON')
            print(solution)
            return solution
    
    def _external_insertion(self, solution, truck, truck_offset, i, j):
        new_sol = copy.deepcopy(solution)
        new_sol[truck_offset].insert(j, new_sol[truck].pop(i))
        return new_sol

    def external_swapping(self, solution: dict, rand: bool = False):
        return self._handle_solution(solution, self._external_swapping, rand)
    
    def external_insertion(self, solution: dict, rand: bool = False):
        return self._handle_solution(solution, self._external_insertion, rand)


    def _handle_internal_solution(self, solution: dict, transformer, rand: bool = False):
        truck = 0
        best_solution = {}
        for sol in solution.values():
            hood = []
            for i in range(1, len(sol) - 2):
                for j in range(i + 1, len(sol) - 1):
                    neighbor = transformer(sol, i, j)
                    hood.append(neighbor)
            hood = list(filter(lambda x: self.check_consistency(x), hood))
            # Noise
            if rand:
                temp = []
                for i in range(len(hood)):
                    if random.random() < 0.7:
                        temp.append(hood[i])
                temp = hood

            if len(hood) > 0:
                best_solution[truck] = self.get_optimal(hood)
            else:
                best_solution[truck] = sol
            truck += 1
        return best_solution
    
    def _swapping(self, solution: list, i, j):
        new_sol = solution.copy()
        aux = new_sol[i]
        new_sol[i] = new_sol[j]
        new_sol[j] = aux
        return new_sol
    
    def _insertion(self, solution: list, i, j):
        new_sol = solution.copy()
        new_sol.insert(i, new_sol.pop(j))
        return new_sol

    def _reversion(self, solution: list, i, j):
        new_sol = solution.copy()
        new_sol[i:j] = new_sol[i:j][::-1]
        return new_sol
    
    def swapping(self, solution: dict, rand: bool = False):
        return self._handle_internal_solution(solution, self._swapping, rand)

    def insertion(self, solution: dict, rand: bool = False):
        return self._handle_internal_solution(solution, self._insertion, rand)

    def reversion(self, solution: dict, rand: bool = False):
        return self._handle_internal_solution(solution, self._reversion, rand)


    def shuffle(self, solution: dict):
        truck = 0
        shuffled_solution = copy.deepcopy(solution)
        trucks = len(shuffled_solution)
        
        for s in solution.values():
            arr = s.copy()
            arr = arr[1:-1]
            random.shuffle(arr)
            shuffled_solution[truck] = [0] + arr + [0]
            truck += 1
        
        for i in range(3):
            try:
                path1 = random.randint(0, trucks - 1)
                path2 = random.randint(0, trucks - 1)

                if len(shuffled_solution[path1]) <= 3: continue
                lb = random.randint(1, len(shuffled_solution[path1]) - 2)
                node = shuffled_solution[path1].pop(lb)
                lb = random.randint(1, len(shuffled_solution[path2]) - 1)
                shuffled_solution[path2].insert(lb, node)
            except:
                pass
        
        return shuffled_solution

