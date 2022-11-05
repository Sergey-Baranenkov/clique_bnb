import cplex
from parse_graph import *
from paint_graph import *
from cplex._internal._subinterfaces import VarTypes

class Solver:
    def __init__(self, graph: nx.Graph, colors_grouped: list[set]):
        # cplex
        self.cplex: cplex.Cplex = None

        self.graph = graph

        # Независимые множества из раскраски
        self.ind_sets = colors_grouped

        # Размер текущей максимальной клики
        self.current_max_clique_size = 0

        # Макс клика
        self.best_solution = None

        self.branch_num = 0

    def initialize_variables(self, c: cplex.Cplex):
        nodes = self.graph.nodes
        nodes_len = len(nodes)
        # Максимизируем целевую функцию (количество вершин в клике)
        c.objective.set_sense(c.objective.sense.maximize)
        c.variables.add(
            names=[f'x{v}' for v in nodes],
            types=VarTypes.continuous * nodes_len,
            obj=[1.0] * nodes_len,
            ub=[1.0] * nodes_len
        )

    def initialize_constraints(self, c: cplex.Cplex):
        complement_graph = nx.complement(self.graph)
        complement_graph_edges = complement_graph.edges

        constraints = []
        # Ограничения по независимым множествам
        for ind_set in self.ind_sets:
            var_names = [f'x{v}' for v in ind_set]
            constraints.append([var_names, [1.0] * len(var_names)])

        # Ограничения по дополнению графа
        for v1, v2 in complement_graph_edges:
            constraints.append([[f'x{v1}', f'x{v2}'], [1.0] * 2])

        boundaries = [1.0] * len(constraints)
        names = [f'c{i}' for i in range(len(constraints))]

        # <=
        senses = ['L'] * len(constraints)

        c.linear_constraints.add(lin_expr=constraints,
                                 senses=senses,
                                 rhs=boundaries,
                                 names=names)

    def initialize_cplex(self):
        c = cplex.Cplex()
        c.set_log_stream(None)
        c.set_results_stream(None)
        c.set_warning_stream(None)
        c.set_error_stream(None)

        self.initialize_variables(c)
        self.initialize_constraints(c)
        self.cplex = c

    def get_branching_var(self, solution: list):
        enumerated_not_ints = [x for x in enumerate(solution) if not x[1].is_integer()]
        return max(enumerated_not_ints, key=lambda x: x[1], default=(None, None))[0]

    def add_branching_constraint(self, value: float, boundary: float, branch_num: int):
        self.cplex.linear_constraints.add(lin_expr=[[[value], [1.0]]],
                                          senses=['E'],
                                          rhs=[boundary],
                                          names=[f'b_{branch_num}'])

    def go_left(self, value: float, branch_num: int):
        self.add_branching_constraint(value, 0.0, branch_num)
        branch = self.branching()
        self.cplex.linear_constraints.delete(f'b_{branch_num}')
        return branch

    def go_right(self, value: float, branch_num: int):
        self.add_branching_constraint(value, 1.0, branch_num)
        branch = self.branching()
        self.cplex.linear_constraints.delete(f'b_{branch_num}')
        return branch

    def calculate_current_clique_size(self):
        return sum([1 for x in self.cplex.solution.get_values() if x == 1.0])

    def branching(self):
        self.cplex.solve()
        solution = self.cplex.solution.get_values()
        cur_objective_value = self.cplex.solution.get_objective_value()

        # Если можно улучшить решение
        if cur_objective_value > self.current_max_clique_size:
            # Получаем индекс максимальной не целой переменной
            branching_var = self.get_branching_var(solution)

            # Если все элементы целочисленные - конец
            if branching_var is None:
                self.current_max_clique_size = self.calculate_current_clique_size()
                self.best_solution = solution
                return self.current_max_clique_size
            else:
                # Ищем решение в левой и правой ветках
                self.branch_num += 1
                cur_branch = self.branch_num
                right_branch = self.go_right(branching_var, cur_branch)
                left_branch = self.go_left(branching_var, cur_branch)
                return max([right_branch, left_branch])

        return 0

    def get_clique_indices(self):
        return [v[0] + 1 for v in enumerate(self.best_solution) if v[1] == 1.0]

    def solve(self):
        self.initialize_cplex()
        self.branching()

        clique = self.get_clique_indices()

        return clique, len(clique)

def main():
    filenames = ['myciel4.clq']
    for filename in filenames:
        # Парсим и строим граф
        v, e, edges = parse_file(f'resources/{filename}')

        # Начальная раскраска графа для независимых множеств
        current_color, colors_grouped = paint_graph(v, edges)
        graph = get_graph(edges)

        # Получаем решение
        solver = Solver(graph, colors_grouped)
        clique, l = solver.solve()

        # Печатаем результат (arg where = 1)
        print('res', l, clique)


if __name__ == '__main__':
    main()
