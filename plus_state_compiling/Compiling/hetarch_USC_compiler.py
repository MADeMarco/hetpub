import networkx as nx
import numpy as np
from sklearn.cluster import SpectralClustering
from collections import defaultdict
import pandas as pd

import itertools


def cost_function(N, assignments, checks, mem_limit=None):
    # Initialize the cost to zero
    cost = 0

    # Hyperparameter that can be changed
    storageEqualizerFactor = 0.001
    if False:  # mem_limit:
        # Check if assignment obeys the register memory limit
        total_bins_count = [0] * N
        for cbin in assignments:
            total_bins_count[cbin] += 1
        # if np.any([tbc > mem_limit for tbc in total_bins_count]):
        #    return None
        for num_qubits_in_bin in total_bins_count:
            cost += num_qubits_in_bin ** 2

    cost *= storageEqualizerFactor
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

            # Otherwise, calculate the cost as the absolute difference
            # between the number of objects in the two bins
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


# def bin_assignments_iterator(N, Q, mem_limit=None):
#    if not mem_limit:
#        mem_limit = Q
#    # Generate all possible combinations of assigning Q objects to N bins with limit on memory size=R
#    A = np.repeat(np.arange(N), mem_limit)     # N=number of registers
#    x = itertools.combinations(A, Q)
#    for combination in x:
#        y = itertools.permutations(combination)
#        for assignment in y:
#            yield assignment



def initialize_assign_qubits(assignments, Q, N, F):
    qubit_df = pd.DataFrame(columns=['QbitID', 'Type', 'QubitIDPerType', 'Status', 'Location'])
    # for now location is referenced based on storage id, this can change later

    # Data Qubits: [0,Q-1]  (inclusive)
    # Dummy Qubits: [Q, Q+N-1]   (inclusive)
    # Ancilla Qubits: [Q+N, Q+2N-2]   (inclusive)
    # Flag Qubits: [Q+2N-1, Q+2N-2+F*(N-1)]   (inclusive)

    for idx, storage in enumerate(assignments):
        qubit_df.loc[len(qubit_df.index)] = [idx, 'Data', idx, 'Storage', storage]

    total_qubits = Q
    # adding dummy qubits for computes:
    idx = 0
    for i in range(total_qubits, total_qubits + N):
        qubit_df.loc[len(qubit_df.index)] = [i, 'Dummy', idx, 'Compute', idx]
        idx += 1

    total_qubits += N
    # ancilla qubits:
    idx = 0
    for i in range(total_qubits, total_qubits + N - 1):
        qubit_df.loc[len(qubit_df.index)] = [i, 'Ancilla', idx, 'Compute', idx]
        idx += 1

    total_qubits += N - 1
    # flag qubits:
    idx = 0
    for i in range(total_qubits, total_qubits + N - 1):
        qubit_df.loc[len(qubit_df.index)] = [i, 'Flag', idx, 'Compute', idx]
        idx += 1

    total_qubits += N - 1
    # flag qubits:
    idx = 0
    for i in range(total_qubits, total_qubits + (F - 1) * (N - 1)):
        qubit_df.loc[len(qubit_df.index)] = [i, 'Flag', idx, 'Storage', idx]
        idx += 1

    total_qubits += F * (N - 1)
    return qubit_df


def get_storage_assignments(checks_in, Q, N, mem_limit=None):
    # May need to check if number of qubits, Q, matches the given checks
    # Isn't this supposed to be a max qubit number (integer)?

    # Strip out the X or Z labels:
    checks = [tup[1] for tup in checks_in]

    # We need a data structure to be initialized and populated here:

    best_cost = 1e9
    best_assignment = []
    count = 0
    print(f'Computing assignments to {N} registers')
    for assignment in bin_assignments_iterator(N, Q):
        cost = cost_function(
            N,
            assignment,
            checks,
            mem_limit=mem_limit,
        )
        if cost:
            if (cost == best_cost) & (len(np.unique(assignment)) < len(np.unique(assignment))):
                best_assignment = [(assignment, cost)]
            elif cost < best_cost:
                best_assignment = [(assignment, cost)]
                best_cost = cost
            count += 1


    return np.array(best_assignment[0][0])



def can_be_performed_concurrently(check1, check2, memory_assignment):
    qubits1 = set(check1[1])
    qubits2 = set(check2[1])

    memories1 = set(memory_assignment[qubit] for qubit in qubits1)
    memories2 = set(memory_assignment[qubit] for qubit in qubits2)

    if len(memories1.intersection(memories2)) == 0:
        concurrent = True

        # edge case: if memory lists are both size one and sequentially numbered and
        # are the first and last memory, then they cannot be performed concurrently
        if (len(memories1) == 1) & (len(memories2) == 1):
            memories = memories1.union(memories2)
            first_last = set([0, max(memory_assignment)])
            if memories == first_last:
                concurrent = False
    else:
        concurrent = False

    return concurrent


def build_graph(checks, memory_assignment):
    G = nx.Graph()

    for i, check in enumerate(checks):
        G.add_node(i, check=check)

    for i in range(len(checks)):
        for j in range(i + 1, len(checks)):
            if not can_be_performed_concurrently(checks[i], checks[j], memory_assignment):
                G.add_edge(i, j)
    return G


def schedule_checks(checks, memory_assignment):
    G = build_graph(checks, memory_assignment)
    coloring = nx.algorithms.coloring.greedy_color(G, strategy='largest_first')

    color_to_checks = defaultdict(list)
    for node, color in coloring.items():
        check = G.nodes[node]['check']
        color_to_checks[color].append(check)

    scheduled_checks = [color_to_checks[color] for color in sorted(color_to_checks.keys())]

    return scheduled_checks


def schedule_individual_gates(assignments, check_schedule, qubit_df):
    gate_schedule = []
    assignments = np.array(assignments)

    print("check_schedule: ", check_schedule)

    involved_memories = np.unique(assignments)
    time_step = 1
    Q = len(assignments)
    for round_idx, round_checks in enumerate(check_schedule):
        # round_gates = []

        XorZ = round_checks[0]
        qubits = np.array(round_checks[1])

        print("Qubits: ", qubits)

        # need to figure out partition versus assignment
        involved_memories_for_check = np.unique(assignments[qubits])

        if len(involved_memories_for_check) > 2:
            print("Warning: Check has more than two involved memories")
            continue

        N = len(involved_memories_for_check)
        if N==2:
            m1, m2 = involved_memories_for_check
        else:
            m1 = involved_memories_for_check[0]

        q_m1 = np.array([q for q in qubits if assignments[q] == m1]).flatten()
        #q_m1 = np.array(np.argwhere(assignments[qubits] == m1)).flatten()
        q_m1 = np.insert(q_m1, len(q_m1), Q+m1)
        n_m1 = len(q_m1)-1
        #print(q_m1)


        if N>1:
            q_m2 = np.array([q for q in qubits if assignments[q] == m2]).flatten()
            #q_m2 = np.array(np.argwhere(assignments[qubits] == m2)).flatten()
            q_m2 = np.insert(q_m2, len(q_m2), Q+m2)
            n_m2 = len(q_m2)-1
            par_CNOTs = 2*min(n_m1, n_m2)
            #print(q_m2)

        else:
            par_CNOTs =0


        par_CNOTs_per_storage = int(par_CNOTs/2)

        # Ancilla Qubits: [Q+N, Q+2N-2]   (inclusive)
        AQ = qubit_df.loc[qubit_df['QbitID'] == (Q + 2 + m1)].to_numpy()

        
        if (N >1):
            if (n_m1 >= n_m2):
                gate_schedule.append([time_step,
                                     ['SWAP',
                                       [qubit_df.loc[qubit_df['QbitID'] == q_m1[0]].to_numpy(),
                                        qubit_df.loc[qubit_df['QbitID'] == (Q + m1)].to_numpy()]]])
                toCNOT = qubit_df.loc[qubit_df['QbitID'] == q_m1[0]].to_numpy()
                toSWAP_CtoS = qubit_df.loc[qubit_df['QbitID'] == (Q + m2)].to_numpy()
                toSWAP_StoC = qubit_df.loc[qubit_df['QbitID'] == q_m2[0]].to_numpy()
                first = True
            else:
                gate_schedule.append([time_step,
                                      [['SWAP',
                                       qubit_df.loc[qubit_df['QbitID'] == q_m2[0]].to_numpy(),
                                       qubit_df.loc[qubit_df['QbitID'] == (Q + m2)].to_numpy()]]])
                toCNOT = qubit_df.loc[qubit_df['QbitID'] == q_m2[0]].to_numpy()
                toSWAP_CtoS = qubit_df.loc[qubit_df['QbitID'] == (Q + m1)].to_numpy()
                toSWAP_StoC = qubit_df.loc[qubit_df['QbitID'] == q_m1[0]].to_numpy()
                first = False
    
            i = 1
            #print("par_CNOTs: ", par_CNOTs)
    
            while True:
                time_step += 1
                gate_schedule.append([time_step,
                                      ['SWAP',
                                       [toSWAP_CtoS,
                                        toSWAP_StoC]],
                                      ['CNOT',
                                       [AQ,
                                        toCNOT
                                        ]],
                                      ])
                # CNOT(ancilla, toCNOT)  #these two are supposed to happen in parallel↓
                # SWAP(toSWAP_C, toSWAP_S(i/2))  #these two are supposed to happen in parallel↑
                i += 1
                # tmp = toCNOT
                toSWAP_CtoS = toCNOT
                toCNOT = toSWAP_StoC
                if ((((i % 2) == 0) & first) | (((i % 2) == 1) & (not first))):
                    toSWAP_StoC = qubit_df.loc[qubit_df['QbitID'] == q_m1[int(i / 2)]].to_numpy()
                    # print("swapping from storage 0", toSWAP_StoC)
                else:
                    toSWAP_StoC = qubit_df.loc[qubit_df['QbitID'] == q_m2[int(i / 2)]].to_numpy()
                    # print("swapping from storage 1", toSWAP_StoC)
                if (i >(par_CNOTs)):
                    break

    
        time_step += 1
        
    
        
        if N==1:
            biggerStorage = q_m1
            toSWAP_StoC = qubit_df.loc[qubit_df['QbitID'] == q_m1[0]].to_numpy()
            gate_schedule.append([time_step,
                                     ['SWAP',
                                       [qubit_df.loc[qubit_df['QbitID'] == q_m1[0]].to_numpy(),
                                        qubit_df.loc[qubit_df['QbitID'] == (Q + m1)].to_numpy()]]])
            toCNOT = toSWAP_StoC
            n_m2=0
            i=0
            time_step += 1
            
        elif(n_m1 != n_m2):
            i = par_CNOTs_per_storage + 1
            gate_schedule.append([time_step,
                                      ['SWAP',
                                       [toSWAP_CtoS,
                                        toSWAP_StoC]],
                                      ['CNOT',
                                       [AQ,
                                        toCNOT
                                        ]],
                                      ])
            toSWAP_CtoS = toCNOT
            toCNOT = toSWAP_StoC
            if (n_m1 > par_CNOTs_per_storage):
                biggerStorage = q_m1
            elif (n_m2 >= par_CNOTs_per_storage):
                biggerStorage = q_m2
            time_step += 1



            
        if (N==1) | (n_m1 != n_m2):
            while True :
                toSWAP_StoC = qubit_df.loc[qubit_df['QbitID'] == biggerStorage[i]].to_numpy()
                toSWAP_CtoS = toCNOT
                print("toSWAP_StoC", toSWAP_StoC)
                print("toSWAP_CtoS", toSWAP_CtoS)
                print('-----------------')

                gate_schedule.append([time_step,
                                  ['SWAP',
                                   [toSWAP_CtoS,
                                    toSWAP_StoC]],
                                  ])
                i +=1
                time_step += 1
                if i>=len(biggerStorage):
                    break
                toCNOT = toSWAP_StoC
                gate_schedule.append([time_step,
                                  ['CNOT',
                                   [AQ,
                                    toCNOT]],
                                  ])
                time_step += 1
                
        else:
            Q1 = qubit_df.loc[qubit_df['QbitID'] == q_m2[-2]].to_numpy()
            Q2 = qubit_df.loc[qubit_df['QbitID'] == q_m2[-1]].to_numpy()
            gate_schedule.append([time_step,
                                  ['SWAP',
                                   [Q1, Q2]]])
            time_step += 1
            
    # for ii in range(len(gate_schedule)):
    #     print("time step: ", gate_schedule[ii][0])
    #     for jj in range(1, len(gate_schedule[ii])):
    #         print(gate_schedule[ii][jj])
    return gate_schedule
