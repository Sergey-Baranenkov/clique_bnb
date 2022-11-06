import networkx as nx
from typing import List


def parse_file(filename):
    with open(filename, 'r') as file:
        edges = []
        number_of_vertices = None
        number_of_edges = None
        for line in file.readlines():
            strip_line = line.strip()
            if strip_line[0] == 'e':
                edges.append(strip_line[2:])
            elif strip_line[0] == 'p':
                [a, b, c, d] = strip_line.split(' ')
                number_of_vertices = c
                number_of_edges = d

        return int(number_of_vertices), int(number_of_edges), edges


def get_graph(edges: List[str]) -> nx.Graph:
    g = nx.Graph()
    for edge in edges:
        [v1, v2] = map(int, edge.split(' '))
        g.add_edge(v1, v2)
    return g
