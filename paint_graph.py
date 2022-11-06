from collections import defaultdict

from typing import List


def paint_graph(number_of_vertices: int, edges: List[str]):
    adj_like = defaultdict(set)
    vertex_degree = defaultdict(int)

    # Заполняем словарь смежности и список степеней вершин
    for edge in edges:
        [v1, v2] = map(int, edge.split(' '))
        adj_like[v1].add(v2)
        vertex_degree[v1] += 1
        adj_like[v2].add(v1)
        vertex_degree[v2] += 1

    # Список цветов вершин
    vertex_color = [None for _ in range(number_of_vertices)]  # n - 1

    # Список вершин, отсортированных по степеням от большей к меньшей
    vertex_sorted_by_degree = sorted(vertex_degree.keys(), key=lambda key: vertex_degree.get(key), reverse=True)

    current_color = 0
    colors_grouped = []

    # Пока есть хотя-бы одна вершина
    while len(vertex_sorted_by_degree):

        # Создаем сет вершин текущего цвета
        vertex_of_color = set()
        # Список индексов вершин текущего цвета (чтобы удалить потом из рассматриваемого списка)
        indices_of_vertex_of_color = []

        # Для всех оставшихся непокрашенных вершин
        for i, vertex in enumerate(vertex_sorted_by_degree):
            # Если хотя-бы одна соседняя вершина покрашена в текущий цвет - пропускаем текущую вершину
            # .intersection так как смотрим только те, что уже были покрашены на данном шаге алгоритма
            for adj_vertex in adj_like[vertex].intersection(vertex_of_color):
                if vertex_color[adj_vertex - 1] == current_color:
                    break
            else:  # Иначе красим текущую вершину в текущий цвет
                vertex_color[vertex - 1] = current_color
                vertex_of_color.add(vertex)

                # Добавляем в список просмотренных вершин, чтобы
                vertex_of_color.add(vertex)
                # Добавляем в список индексов вершин, которые нужно убрать перед следующим шагом
                indices_of_vertex_of_color.append(i)

        # Удаляем нужные индексы в обратном порядке чтобы лист не сдвигался и индексы не багались
        for index in sorted(indices_of_vertex_of_color, reverse=True):
            del vertex_sorted_by_degree[index]

        colors_grouped.append(vertex_of_color)
        current_color += 1

    return current_color, colors_grouped
