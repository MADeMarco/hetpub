import stim
import sinter
import scienceplots
import copy
from surface_code_sim import LogicalQubit
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import FuncFormatter, MaxNLocator
import matplotlib as mpl
import os
from datetime import datetime
from matplotlib.colors import LinearSegmentedColormap

def simulate_sc(param_sets):
    task_list = []
    for params in param_sets:
        d = params['d']

        num_cycles = 2*d

        LQ = LogicalQubit(d, params)
        circ = LQ.generate_stim(rounds=num_cycles)
        dem = circ.detector_error_model(
            approximate_disjoint_errors=True,
            decompose_errors=True,
            ignore_decomposition_failures=True
        )

        task_list.append(sinter.Task(
            circuit=circ,
            detector_error_model=dem,
            json_metadata=params)
        )

    samples = sinter.collect(
        num_workers=8,
        max_shots=1_000_000, # Did 1 trillion shots for figures 6 and 7 data
        max_errors=1000,
        tasks=task_list,
        decoders=['pymatching'],
    )

    sample_data = [{'error_prob': (float(sample.errors) / float(sample.shots)), 'num_cycles': num_cycles, **(sample.json_metadata)} for sample in
                   samples]

    return (sample_data)

def ancilla_vs_data(base_params, T_ratio_values, d):
    data_param_set = []
    anci_param_set = []
    for T_ratio in T_ratio_values:
        params = copy.deepcopy(base_params)
        params['data_T1'] = T_ratio * params['data_T1']
        params['data_T2'] = T_ratio * params['data_T2']
        params['d'] = d
        params['T_ratio'] = T_ratio
        data_param_set.append(params)

        params = copy.deepcopy(base_params)
        params['ancilla_T1'] = T_ratio * params['ancilla_T1']
        params['ancilla_T2'] = T_ratio * params['ancilla_T2']
        params['d'] = d
        params['T_ratio'] = T_ratio
        anci_param_set.append(params)

    sample_data = simulate_sc(data_param_set)
    sample_ancilla = simulate_sc(anci_param_set)

    # get error rates:
    data_error = [x['error_prob']/x['num_cycles'] for x in sample_data]
    ancilla_error = [x['error_prob']/x['num_cycles'] for x in sample_ancilla]

    # Plotting:
    with plt.style.context(['science', 'no-latex']):
        fig, ax = plt.subplots(figsize=(8, 5))
        fig_options(fig, ax)
        pparam = dict(xlabel=r'$\alpha$', ylabel='$Logical\ Error\ Rate$')
        ax.set(**pparam)
        ax.set_ylim(0.8*min(data_error), 1.2 * max(data_error))
        ax.set_xlim(0, max(T_ratio_values))
        ax.scatter(T_ratio_values, data_error,
                   s=10, label=r'$T_{CD} =\alpha \cdot 100\mu s$, $T_{CA} = 100\mu s$',
                   color='#2FA990', alpha=0.9)
        ax.scatter(T_ratio_values, ancilla_error,
                   s=10, label=r'$T_{CD} = 100\mu s$, $T_{CA} = \alpha \cdot 100 \mu s$',
                   color='#8A5DA2', alpha=0.9)
        ax.plot([1, 1], [0, 1], color='#000000', linestyle='--', marker = 'None', linewidth=1)
        ax.annotate('Homogeneous', xy=(1, data_error[0]+0.0015), xytext=(1.50, data_error[0]+0.002),
                    arrowprops=dict(facecolor='black', arrowstyle='->'),
                    fontsize=12)
        ax.legend(title='', title_fontsize=18,fontsize=16,frameon=True)
        fig.tight_layout()
        # plt.show()
        save_plot_with_timestamp(fig, 'ancilla_vs_data')

def generate_colors(N, color1, color2):
    """
    Generate N colors interpolating between color1 and color2.

    Args:
    N (int): Number of colors to generate.
    color1 (str): Starting color in hex format.
    color2 (str): Ending color in hex format.

    Returns:
    list: List of N colors in hex format.
    """
    # Create a colormap that interpolates between color1 and color2
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", [color1, color2])

    # Generate N colors from the colormap
    colors = custom_cmap(np.linspace(0, 1, N))

    # Convert the colors to hex format
    hex_colors = [mpl.colors.rgb2hex(color) for color in colors]

    return hex_colors
#
# def plot_error_ratio(base_params, min_d, max_d, gate2_Time_values, T_ratio_values):
#
#     color_array = generate_colors(len(T_ratio_values), '#2F5080', '#1B7A74')
#
#     param_set = []
#     for T_ratio in T_ratio_values:
#         for gate2_Time in gate2_Time_values:
#             for d in [min_d, max_d]:
#                 params = copy.deepcopy(base_params)
#                 params['data_T1'] = T_ratio * params['data_T1']
#                 params['data_T2'] = T_ratio * params['data_T2']
#                 params['d'] = d
#                 params['time_2q'] = gate2_Time
#                 params['T_ratio'] = T_ratio
#                 param_set.append(params)
#
#     sample_data = simulate_sc(param_set)
#
#     with plt.style.context(['science', 'no-latex']):
#         for i_T, T_ratio in enumerate(T_ratio_values):
#             error_ratio = []
#             for gate2_Time in gate2_Time_values:
#                 select_data = [x for x in sample_data if x['T_ratio'] == T_ratio and x['time_2q'] == gate2_Time]
#                 data_error = [x['error_prob'] for x in select_data]
#                 error_ratio.append(data_error[-1] / data_error[0])
#             # if i_T == 0:
#             #     fig, ax = plt.subplots(figsize=(8, 5))
#             #     fig_options(fig, ax)
#             #     pparam = dict(xlabel='$Gate\ Time\ (\mu s)$', ylabel='$Error\ Ratio$')
#             #     ax.set(**pparam)
#             #     ax.set_xlim(min(gate2_Time_values), max(gate2_Time_values))
#
#         for i_g, gate2_Time in enumerate(gate2_Time_values):
#             error_ratio = []
#             for T_ratio in T_ratio_values:
#                 select_data = [x for x in sample_data if x['T_ratio'] == T_ratio and x['time_2q'] == gate2_Time]
#                 data_error = [x['error_prob'] for x in select_data]
#                 error_ratio.append(data_error[-1] / data_error[0])
#             if i_g == 0:
#                 fig, ax = plt.subplots(figsize=(8, 5))
#                 fig_options(fig, ax)
#                 pparam = dict(xlabel='$T_{CD}/T_{CA} $', ylabel='$Error\ Ratio$')
#                 ax.set(**pparam)
#                 ax.set_xlim(min(T_ratio_values)-0.25, max(T_ratio_values)+0.25)
#
#             ax.plot(T_ratio_values, error_ratio, label=f'${gate2_Time}\mu s$', color = color_array[i_T])
#         # ax.plot(T_ratio_values, [1] * len(T_ratio_values), color='black', linestyle='--', marker = 'None', linewidth=1)
#         ax.hlines(y=1,xmin=min(T_ratio_values)-0.25, xmax=max(T_ratio_values)+0.25, color='black', linestyle='--')
#         ax.legend(frameon=True, title="Gate time")
#         plt.show()
#         filename = 'error_ratio'
#         save_plot_with_timestamp(fig, filename)


def plot_distance(base_params, T_ratio_values, d_values):

    color_array = generate_colors(len(T_ratio_values), '#3C8DAD', '#E44D71')

    param_set = []
    for T_ratio in T_ratio_values:
        for d in d_values:
            params = copy.deepcopy(base_params)
            params['data_T1'] = T_ratio * params['data_T1']
            params['data_T2'] = T_ratio * params['data_T2']
            params['d'] = d
            params['T_ratio'] = T_ratio
            param_set.append(params)

    sample_data = simulate_sc(param_set)

    with plt.style.context(['science', 'no-latex']):

        for i_T, T_ratio in enumerate(T_ratio_values):
            select_data = [x for x in sample_data if x['T_ratio'] == T_ratio]
            data_error = [x['error_prob'] / x['num_cycles'] for x in select_data]
            distance = [x['d'] for x in select_data]
            xy = sorted(zip(distance, data_error))
            distance_sorted = [x for x, y in xy]
            data_error_sorted = [y for x, y in xy]
            if i_T == 0:
                fig, ax = plt.subplots(figsize=(8, 5))
                fig_options(fig, ax)
                pparam = dict(xlabel='$Distance$', ylabel='$Logical\ Error\ Rate$')
                ax.set(**pparam)
                ax.set_xlim(min(d_values)-1, max(d_values)+1)
                ax.set_xticks(ticks=[5,10,15,20])

            ax.plot(distance_sorted, data_error_sorted, label=f'${T_ratio}$',
                    color = color_array[i_T],
                    linestyle='--', linewidth=3,
                    alpha=0.9)
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
                  frameon=True, title="$T_{CD}/T_{CA}$", title_fontsize=18,fontsize=16)
        # fig.tight_layout()
        plt.show()
        filename = 'distance'
        save_plot_with_timestamp(fig, filename)

def is_decreasing(data):
    if len(data) < 3:
        raise ValueError("Data should contain at least 3 elements.")

    middle_index = len(data) // 2
    first_element = data[0]
    middle_element = data[middle_index]
    last_element = data[-1]

    return first_element > middle_element > last_element

def save_plot_with_timestamp(fig, filename):
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    pdf_filename = f"Figure_Saves/{filename}_{timestamp}.pdf"
    fig.savefig(pdf_filename)
    print(f"Plot saved as {pdf_filename}")

def fig_options(fig, ax):
    mpl.rcParams['axes.labelsize'] = 16
    mpl.rcParams['xtick.labelsize'] = 14
    mpl.rcParams['ytick.labelsize'] = 14

    ax.yaxis.grid(visible=True, which='minor', color='0.775', linestyle='-')
    ax.xaxis.grid(visible=True, which='major', color='0.775', linestyle='-')
    ax.xaxis.grid(visible=True, which='minor', color='0.875', linestyle='-')

    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    ax.xaxis.label.set_size(16)  # Increase X label font size
    ax.yaxis.label.set_size(16)  # Increase Y label font size

    marker_cycle = plt.cycler(marker=['o', 's', '^', 'v', '*', 'D', 'X', 'P'])
    color_cycle = plt.cycler(color=['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow'])
    combined_cycle = marker_cycle + color_cycle

    # Apply the marker and color cycle to the axes
    plt.gca().set_prop_cycle(combined_cycle)

    # plt.subplots_adjust(left=0.1, right=.95, bottom=0.1, top=.95)
    
def main():
    base_params = {  # units are microseconds
            'data_T1': 100.0,
            'data_T2': 100.0,
            'ancilla_T1': 100.0,
            'ancilla_T2': 100.0,
            'time_1q_ancilla': 0.04,
            'time_2q': 0.1,
            'time_measurement_ancilla': 1, #0.5
            'gate2_err': 0,
            'time_reset_ancilla': 0.04, #0.5
            'ancilla_1q_err': 0,#0.001,
            'data_ancilla_2q_err': 0,
            'readout_err': 0,
            'reset_err': 0
        }
    manual_gate2_err = 0.01#.0075

    base_params['gate2_err'] = manual_gate2_err

    T_ratio_values_fine = np.linspace(1, 10, 100)
    T_ratio_coarse = range(1,8,1)
    gate2_Time_values = np.linspace(.1, 1, 5) #5
    d_values = range(5,20,2) #2

    # ancilla_vs_data(base_params=base_params, T_ratio_values=T_ratio_values_fine, d=13)
    # plot_error_ratio(base_params=base_params, min_d=5, max_d=9,
    #                  gate2_Time_values=gate2_Time_values, T_ratio_values=T_ratio_coarse)
    plot_distance(base_params=base_params, d_values= d_values, T_ratio_values=T_ratio_coarse)



if __name__ == '__main__':
    main()
