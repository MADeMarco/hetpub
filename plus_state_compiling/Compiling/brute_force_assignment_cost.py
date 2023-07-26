import numpy as np
import itertools

def cost_function(N, assignments, checks, mem_limit=None):
    # Initialize the cost to zero
    cost = 0

    if mem_limit:
        # Check if assignment obeys the register memory limit
        total_bins_count = [0] * N
        for cbin in assignments:
            total_bins_count[cbin] += 1
        #if np.any([tbc > mem_limit for tbc in total_bins_count]):
        #    return None
        for num_qubits_in_bin in total_bins_count:
            cost += num_qubits_in_bin ** 2

    # Iterate through each check
    for check in checks:
        # Initialize a list to store the number of objects in each bin for the current check
        bins_count = [0] * N

        # Calculate the number of objects in each bin for the current check
        for obj in check:
            bin_idx = assignments[obj - 1]  # Subtract 1 to account for 0-indexing
            bins_count[bin_idx] += 1

        # Calculate the number of non-empty bins
        non_empty_bins = sum(1 for count in bins_count if count > 0)

        # If more than 2 bins contain any objects, return None
        if non_empty_bins > 2:
            return None

        if non_empty_bins == 2:
            idxs_with_objects = [idx for idx, count in enumerate(bins_count) if count > 0]
            if abs(idxs_with_objects[0] - idxs_with_objects[1]) > 1:
                return None

            # Otherwise, calculate the cost as the absolute difference between the number of objects in the two bins
            else:
                bins_with_objects = [count for count in bins_count if count > 0]
                cost += abs(bins_with_objects[0] - bins_with_objects[1])
        # If only 1 bin contains any objects, add the length of the check to the cost
        elif non_empty_bins == 1:
            cost += len(check)
        else:
            raise Exception("something went wrong!")

    return cost


def bin_assignments_iterator(N, Q):
    # Generate all possible combinations of assigning Q objects to N bins
    for assignment in itertools.product(range(N), repeat=Q):
        yield assignment


if __name__ == "__main__":

    # These example checks are from distance 3 surface code
    sc_checks = [
        (1,2,4,5),
        (2,3,5,6),
        (4,5,7,8),
        (5,6,8,9),
    ]

    # These checks are for Reed-Muller
    rm_checks = [
        [ 1,  3,  7,  5,  8, 10, 14, 12],
        [ 4,  5,  7,  6, 11, 12, 14, 13],
        [ 2,  3,  7,  6,  9, 10, 14, 13],
        [ 8,  9, 10, 11, 12, 13, 14, 15],
        [ 1,  3,  8, 10],
        [ 1,  5,  8, 12],
        [ 2,  3,  9, 10],
        [ 2,  6,  9, 13],
        [ 3,  7, 10, 14],
        [ 4,  5, 11, 12],
        [ 4,  6, 11, 13],
        [ 5,  7, 12, 14],
        [ 6,  7, 13, 14],
        [ 1,  3,  7,  5],
        [ 4,  5,  7,  6],
        [ 2,  3,  7,  6],
        [ 8, 10, 14, 12],
        [11, 12, 14, 13],
        [ 9, 10, 14, 13],
        [ 8, 11, 12, 15],
    ]

    checks = sc_checks
    num_qubits = 9 # Set this number carefully, not checking if it matches the given checks
    mem_limit = True
    for N in [2, 3, 4]: # Number of memory registers
        best_cost = 1e9
        best_assignments = []
        count = 0
        print(f'Computing assignments to {N} registers')
        for assignment in bin_assignments_iterator(N, num_qubits):
            cost = cost_function(
                    N,
                    assignment,
                    checks,
                    mem_limit=mem_limit,
                )
            if cost:
                if cost == best_cost:
                    best_assignments.append((assignment, cost))
                elif cost < best_cost:
                    best_assignments = [(assignment, cost)]
                    best_cost = cost
            count += 1
            #if count % 100:
            #    print(f"Checked {count} assignments")
        print(f"\tChecked {count} assignments,\n\t{len(best_assignments)} optimal assignments with cost {best_cost}")
        solns = {n: 0 for n in range(2, N+1)}
        for assignment, _ in best_assignments:
            nregs = len(set(assignment))
            solns[nregs] += 1
        for key, val in solns.items():
            print(f'\t\t{val} assignments used {key} registers')

       # print(f"{len(best_assignments)} optimal assignments with cost of {best_cost}")
    #for a, c in best_assignments:
    #    print(a, c)








