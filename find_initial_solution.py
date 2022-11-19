from collections import defaultdict
from datetime import datetime

from typing import List

import networkx


def find_initial_solution(graph: networkx.Graph):
    adj_like = defaultdict(set)
    vertex_degree = defaultdict(int)

    # Заполняем словарь смежности и список степеней вершин
    for edge in graph.edges:
        [v1, v2] = map(int, edge)
        adj_like[v1].add(v2)
        vertex_degree[v1] += 1
        adj_like[v2].add(v1)
        vertex_degree[v2] += 1

    def sort_by_degree(l):
        return sorted(l, key=lambda key: vertex_degree.get(key), reverse=True)

    vertices_to_iterate = range(len(graph.nodes))

    max_clique = None

    def step(neighbours: List[int], initial_neighbour_idx=0):
        clique = []
        # Дефолтный жадный алгоритм построения клики кроме того, что первое разветвление можно регулировать индексом
        while len(neighbours):
            next_vertex = neighbours[initial_neighbour_idx]
            initial_neighbour_idx = 0
            clique.append(next_vertex)
            neighbours = list(set(neighbours).intersection(adj_like[next_vertex]))
            neighbours = sort_by_degree(neighbours)
        return clique

    # Проходим каждую вершину (начальную)
    for vertex in vertices_to_iterate:
        # Получаем ее соседей
        neighbours = sort_by_degree(adj_like[vertex])
        max_clique_local = []
        n_first_neighbours = 50

        # Для первой вершины просматриваем n_first_neighbours ее соседей
        for idx in range(len(neighbours[:n_first_neighbours])):
            # В клике изначально наша вершина
            clique = [vertex]
            # Достраиваем клику, идя из vertex в idx соседа
            # ( idx from 0 to min(len(neighbours), n_first_neighbours) - 1 )
            clique += step(neighbours, idx)

            # Если найденная клика, начиная с вершины vertex больше чем другая клика с этой вершины - перезаписываем
            # локально-максимальную клику для этой вершины
            if len(max_clique_local) < len(clique):
                max_clique_local = clique

        # Если максимальная клика этой вершины больше чем глобальная клика любой вершины - перезаписываем
        if max_clique is None or len(max_clique_local) > len(max_clique):
            max_clique = max_clique_local

    return max_clique