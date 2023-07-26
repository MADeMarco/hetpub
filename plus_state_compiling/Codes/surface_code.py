import networkx as nx
import numpy as np
from sklearn.cluster import SpectralClustering

def square_lattice_graph(Lx, Ly):
    graph = nx.Graph()

    for x in range(Lx):
        for y in range(Ly):
            qubit_index = x * Ly + y

            # Connect qubit to its right neighbor
            if x < Lx - 1:
                right_neighbor_index = (x + 1) * Ly + y
                graph.add_edge(qubit_index, right_neighbor_index, check=('Z', (x, y)))

            # Connect qubit to its upper neighbor
            if y < Ly - 1:
                upper_neighbor_index = x * Ly + (y + 1)
                graph.add_edge(qubit_index, upper_neighbor_index, check=('X', (x, y)))

    return graph


def stabilizer_checks(Lx, Ly):
    checks = []

    for x in range(Lx):
        for y in range(Ly):
            # Calculate the indices of the face's qubits
            bottom_left = x * Ly + y

            if x < Lx - 1 and y < Ly - 1:
                bottom_right = (x + 1) * Ly + y
                top_left = x * Ly + (y + 1)
                top_right = (x + 1) * Ly + (y + 1)

                check = [bottom_left, bottom_right, top_right, top_left]

                if bottom_left % 2 == 0:
                    checks.append(('X', check))
                else:
                    checks.append(('Z', check))
            else:
                if x < Lx - 1:  # Boundary check on the right edge
                    bottom_right = (x + 1) * Ly + y
                    check = [bottom_left, bottom_right]

                    if bottom_left % 2 == 0:
                        checks.append(('X', check))
                    else:
                        checks.append(('Z', check))

                if y < Ly - 1:  # Boundary check on the top edge
                    top_left = x * Ly + (y + 1)
                    check = [bottom_left, top_left]

                    if bottom_left % 2 == 0:
                        checks.append(('X', check))
                    else:
                        checks.append(('Z', check))

    return checks


