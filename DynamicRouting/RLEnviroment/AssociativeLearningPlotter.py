import sys
import traceback

from DynamicSynapse.Utils.MatToVideo2 import matrix2video2
from DynamicSynapse.Utils.tracereader import TraceReader
from matplotlib.backends.backend_pdf import PdfPages
import os
import matplotlib.pyplot as plt
import matplotlib

import numpy as np
from tqdm import tqdm

from DynamicRouting.Utils.TaxiUtils import decode
from multiprocessing import Pool, TimeoutError



def analysis_plot(trace_file_folder=None, maxstep=40, number_of_maggots=30, len_protocol=8, num_locations=5):

    step_traces = [[None for _ in range(number_of_maggots)] for _ in range(len_protocol)]
    step_locations = [[None for _ in range(number_of_maggots)] for _ in range(len_protocol)]
    location_counts_per_maggot = [[None for _ in range(number_of_maggots)] for _ in
                                  range(len_protocol)]  # numbers of maggots on AM side or on OCT side
    location_counts_per_step = [[None for _ in range(maxstep)] for _ in range(len_protocol)]
    learning_pref_per_maggot = np.zeros((len_protocol, number_of_maggots))
    learning_pref_per_step = np.zeros((len_protocol, maxstep))
    learning_index_per_step = {}
    learning_index_per_maggot = {}
    for i_protocol in range(len_protocol):
        for i_maggot in range(number_of_maggots):
            log_file_path = os.path.join(trace_file_folder,
                                         "protocol%d_maggot%d_experiment_step.pkl" % (i_protocol, i_maggot))
            aTR = TraceReader(log_file_path=log_file_path)
            step_trace = aTR.get_trace()
    
            step_traces[i_protocol][i_maggot] = step_trace  # = step_loggers[i_protocol][i_maggot].retrieve_record()

            step_locations[i_protocol][i_maggot] = np.array(step_trace['observation'])[-maxstep:]
    
            locations, temp_counts = np.unique(step_locations[i_protocol][i_maggot], return_counts=True)
            counts = [0 for _ in range(num_locations)]
            for i in range(len(locations)):
                counts[locations[i]] = temp_counts[i]
            print('locations')
            print(locations)
    
            location_counts_per_maggot[i_protocol][i_maggot] = counts
            print('location_counts_per_maggot[%d][%d]' % (i_protocol, i_maggot))
            print(location_counts_per_maggot[i_protocol][i_maggot])
            # # AM side  counts[0:2] # # OCT side counts[3:5]
            learning_pref_per_maggot[i_protocol][i_maggot] = (counts[0]+counts[1]  - counts[3]-counts[4]) / maxstep
            print('learning_pref_per_maggot[%d][%d]' % (i_protocol, i_maggot))
            print(learning_pref_per_maggot[i_protocol][i_maggot])

    step_locations = np.array(step_locations)
    for i_protocol in range(len_protocol):
        for i_step in range(maxstep):
            locations, temp_counts = np.unique(step_locations[i_protocol, :, i_step], return_counts=True)
            counts = [0 for _ in range(num_locations)]
            for i in range(len(locations)):
                counts[locations[i]] = temp_counts[i]
            location_counts_per_step[i_protocol][i_step] = counts
            print('learning_pref_per_step[%d][%d]' % (i_protocol, i_step))
            print(learning_pref_per_step[i_protocol][i_step])
            # # AM side  counts[0:2] # # OCT side counts[3:5]
            learning_pref_per_step[i_protocol][i_step] = (counts[0]+counts[1]  - counts[3]-counts[4]) / number_of_maggots
            print('learning_pref_per_step[%d][%d]' % (i_protocol, i_step))
            print(learning_pref_per_step[i_protocol][i_step])
    
    learning_pref_per_step = np.array(learning_pref_per_step)
    print('learning_pref_per_step')
    print(learning_pref_per_step)
    learning_index_per_step['FN'] = (learning_pref_per_step[0] - learning_pref_per_step[1]) / 2
    learning_index_per_step['FF'] = (learning_pref_per_step[2] - learning_pref_per_step[3]) / 2
    learning_index_per_step['QN'] = (learning_pref_per_step[4] - learning_pref_per_step[5]) / 2
    learning_index_per_step['QQ'] = (learning_pref_per_step[6] - learning_pref_per_step[7]) / 2
    print(learning_index_per_step)
    
    labels = ['FN', 'FF', 'QN', 'QQ']
    learning_index_per_step_array = np.vstack((learning_index_per_step['FN'],
                                     learning_index_per_step['FF'],
                                     learning_index_per_step['QN'],
                                     learning_index_per_step['QQ'])).T
    
    fig, ax = plt.subplots(nrows=1, ncols=1,  figsize=(4, 3))
    bplot = ax.boxplot(learning_index_per_step_array,
                       notch=False,  # notch shape
                       vert=True,  # vertical box alignment
                       patch_artist=True,  # fill with color
                       labels=labels)  # will be used to label x-ticks
    # ax.set_title('learning_index_per_step')
    ylim = list(ax.get_ylim())
    if ylim[0] > -0.4:
        ylim[0] = -0.4
    if ylim[1] < 0.7:
        ylim[1] = 0.7
    plt.ylim(ylim)
    ax.yaxis.grid(True)
    ax.set_title('learning_pref_per_step')
    ax.set_xlabel('Training / testing reinforcers')
    ax.set_ylabel('Learning index')
    plt.tight_layout()

    learning_pref_per_maggot = np.array(learning_pref_per_maggot)
    print('learning_pref_per_step')
    print(learning_pref_per_maggot)
    learning_index_per_maggot['FN'] = (learning_pref_per_maggot[0] - learning_pref_per_maggot[1]) / 2
    learning_index_per_maggot['FF'] = (learning_pref_per_maggot[2] - learning_pref_per_maggot[3]) / 2
    learning_index_per_maggot['QN'] = (learning_pref_per_maggot[4] - learning_pref_per_maggot[5]) / 2
    learning_index_per_maggot['QQ'] = (learning_pref_per_maggot[6] - learning_pref_per_maggot[7]) / 2
    print(learning_index_per_maggot)

    labels = ['FN', 'FF', 'QN', 'QQ']
    learning_index_per_maggot_array = np.vstack((learning_index_per_maggot['FN'],
                                      learning_index_per_maggot['FF'],
                                      learning_index_per_maggot['QN'],
                                      learning_index_per_maggot['QQ'])).T

    fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(4, 3))
    bplot2 = ax2.boxplot(learning_index_per_maggot_array,
                       notch=False,  # notch shape
                       vert=True,  # vertical box alignment
                       patch_artist=True,  # fill with color
                       labels=labels)  # will be used to label x-ticks
    ax2.set_title('learning_index_per_maggot')
    ax2.yaxis.grid(True)
    ax2.set_xlabel('Training / testing reinforcers')
    ax2.set_ylabel('Learning index')
    plt.tight_layout()
    plt.show()
    pass


if __name__ == "__main__":
    experiment_time = ""  # Fill in the experiment time, as the folder name shows
    trace_file_folder = "recording\\MaggotInPetriDish-v1_DynamicRouting\\"+experiment_time+"\\trace\\"
    analysis_plot(trace_file_folder=trace_file_folder, maxstep=40, number_of_maggots=30, len_protocol=8)

