import numpy as np

def reed_muller_checks():

    # This script calculates the stabilizers for the 15-qubit doubled color code.
    # The code is based on the Reed-Muller code, a quantum error-correcting code.
    # For more information, see: https://arxiv.org/pdf/1509.03239.pdf

    # Define the faces of the 3x4 lattice
    faces = np.array([[1, 3, 7, 5], [4, 5, 7, 6], [2, 3, 7, 6]])

    # Get the size of the faces matrix
    sizes = faces.shape

    # Initialize the links matrix
    links = np.zeros((np.prod(sizes), 2), dtype=int)

    # Populate the links matrix
    for irow in range(3):
        for icol in range(4):
            linknum = icol + sizes[1] * irow
            links[linknum, 0] = faces[irow, icol]
            links[linknum, 1] = faces[irow, (icol+1) % 4]

    # Remove duplicate rows from the links matrix
    links = np.unique(np.sort(links, axis=1), axis=0)

    # Define the X-checks and Z-checks matrices
    X_checks = np.vstack([np.hstack([faces, faces + 7]), np.arange(8, 16).reshape(1, -1)])
    G = np.hstack([links, links + 7])
    T = np.vstack([faces, faces + 7, [8, 11, 12, 15]])
    Z_checks = np.vstack([G, T])

    print("X_checks: ", X_checks)
    print("Z_checks: ", Z_checks)

    X_checks = [('X', row.tolist()) for row in X_checks]
    Z_checks = [('Z', row.tolist()) for row in Z_checks]

    return X_checks, Z_checks


# Instead of generating graphs, just call:
# get_storage_assignments
# initialize and assign qubits
# schedule individual gates

