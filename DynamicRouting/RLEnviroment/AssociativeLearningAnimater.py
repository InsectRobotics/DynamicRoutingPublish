import sys
import traceback

from DynamicSynapse.Utils.MatToVideo2 import matrix2video2
from DynamicSynapse.Utils.tracereader import TraceReader
from cycler import cycler
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
import numpy as np
from tqdm import tqdm

from DynamicRouting.Utils.TaxiUtils import decode
from multiprocessing import Pool, TimeoutError
from DynamicRouting.RLEnviroment.AssociativeLearning import Maggot

def save_as_video(traces, video_path, prefix='', method=''):
    video_filename = os.path.join(video_path, prefix + 'output.avi')
    print('video_filename', video_filename)
    matrix2video2(traces['connection_matrix'], video_filename=video_filename)

def matshow_images(trace, video_path, prefix=''):
    video_file_path = os.path.join(video_path, prefix)
    if not os.path.exists(video_file_path):
        os.makedirs(video_file_path)
    print('video_file_path', video_file_path)
    vmax = np.max(trace['connection_matrix'])
    for index in tqdm(range((trace['connection_matrix'].shape[0]))):
        figure = plt.matshow(trace['connection_matrix'][index, :, :], vmax=vmax)
        plt.colorbar()
        plt.savefig(os.path.join(video_file_path, f'step{index:06d}.png'))
        plt.close()
    os.chdir(video_file_path)
    os.system('ffmpeg -f image2 -framerate 25 -i step%06d.png -vcodec libx265 -crf 22 -y ../'+prefix+'video.mp4')

def matshow_images2(trace, video_path, prefix=''):
    video_file_path = os.path.join(video_path, prefix)
    if not os.path.exists(video_file_path):
        os.makedirs(video_file_path)
    print('video_file_path', video_file_path)
    vmax = np.max(trace['weight'])
    for index in tqdm(range((trace['weight'].shape[0]))):
        weights_in_a_step = trace['weight'][index, :, :, :]
        # to state neuron, from state neuron, action neuron
        weights_in_a_step_new = np.moveaxis(weights_in_a_step, 2, 0)
        # action neuron, to state neuron, from state neuron
        weights_in_a_step_new_2d = weights_in_a_step_new.reshape(weights_in_a_step_new.shape[0],
                                   weights_in_a_step_new.shape[1] * weights_in_a_step_new.shape[2])
        figure = plt.matshow(weights_in_a_step_new_2d, vmax=vmax)
        plt.colorbar()
        plt.savefig(os.path.join(video_file_path, f'step{index:06d}.png'))
        plt.close()
    os.chdir(video_file_path)
    os.system('ffmpeg -f image2 -framerate 25 -i step%06d.png -vcodec libx265 -crf 22 -y ../'+prefix+'video.mp4')

def plot(plot_path='', trace=None, trace2=None, DRN=None, petri_dish_steps=40):
    figure_dict = dict()
    vmax = np.max(trace['connection_matrix'])
    # figure_dict['connection_matrix']  = plt.figure()
    plt.matshow(trace['connection_matrix'][-1, :, :], vmax=vmax)
    plt.colorbar()
    figure_dict['connection_matrix'] = plt.gcf()

    # default_cycler = (cycler(color=['r', 'g', 'b', 'y']) +
    #                   cycler(linestyle=['-', '--', ':', '-.']))
    default_cycler = (cycler(color=mcolors.TABLEAU_COLORS) *
                      cycler(linestyle=['-', '--', ':', '-.']))
    # plt.rc('lines', linewidth=4)
    plt.rc('axes', prop_cycle=default_cycler)

    connection_matrix_shape = trace['connection_matrix'].shape

    flatten_connection_matrix = trace['connection_matrix'].reshape(connection_matrix_shape[0],
                                                                   connection_matrix_shape[1]*connection_matrix_shape[2])
    fig, ax = plt.subplots()
    figure_dict['matrix_weight'] = fig
    lines = ax.plot(flatten_connection_matrix)
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.70])
    ax.legend(lines, [str(i)+' to '+str(j) for i in range(connection_matrix_shape[1]) for j in range(connection_matrix_shape[2])],
              ncol=6, loc='upper center', bbox_to_anchor=(0.48, 1.6))
    ax.vlines(np.arange(petri_dish_steps, connection_matrix_shape[0], petri_dish_steps), 0, vmax, colors='salmon', linestyles=':', linewidth=1)

    if trace2 is not None:

        action_weight_matrix = trace2['weight']
        # action_weight_matrix_shape = action_weight_matrix.shape
        #
        # action_weight_matrix_flatten = action_weight_matrix.reshape(connection_matrix_shape[0],
        #                                                                connection_matrix_shape[1] *
        #                                                                connection_matrix_shape[2] *
        #                                                             connection_matrix_shape[3])
        wmin = np.min(action_weight_matrix)
        wmax = np.max(action_weight_matrix)

        if wmin>wmax*0.01:
            threshould = wmin
        else:
            threshould = wmax*0.01
        weights_in_a_step =action_weight_matrix[-1, :, :, :]
        # to state neuron, from state neuron, action neuron
        weights_in_a_step_new = np.moveaxis(weights_in_a_step, 2, 0)
        # action neuron, to state neuron, from state neuron
        weights_in_a_step_new_2d = weights_in_a_step_new.reshape(weights_in_a_step_new.shape[0],
                                                                 weights_in_a_step_new.shape[1] *
                                                                 weights_in_a_step_new.shape[2])
        fig, ax = plt.subplots()
        plt.matshow(weights_in_a_step_new_2d, vmax=wmax)
        plt.colorbar()
        figure_dict['action_weight_final'] = plt.gcf()

        learned_weight_bool = np.any(np.greater(action_weight_matrix,threshould), axis=0 )
        learned_weight_index = np.where(learned_weight_bool)
        fig, ax = plt.subplots(figsize=(6.4,3.2))
        figure_dict['action_weight'] = fig
        lines = []
        legend = []
        for from_index, to_index, action_index in zip(*learned_weight_index):
            lines.append(*ax.plot(action_weight_matrix[:,from_index, to_index, action_index]))
            legend.append('edge ' + str(from_index) + ' to ' + str(to_index) +' to a' + str(action_index))
        pos = ax.get_position()
        # ax.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.70])
        ax.legend(lines, legend)
                  # ncol=6, loc='upper center', bbox_to_anchor=(0.48, 1.6))
        ax.vlines(np.arange(petri_dish_steps, connection_matrix_shape[0], petri_dish_steps), 0, wmax, colors='salmon',
                  linestyles=':', linewidth=1)

    if DRN is not None:
        figure_dict['DRN_graph'] = DRN.plot_graph()
        figure_dict['DRN_heatmap'] = DRN.plot_heatmap(decoder=decode, task='Taxi-v3')

    if plot_path:
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        pp = PdfPages(os.path.join(plot_path, "Emviroment.pdf"))
        for key in figure_dict:
            figure_dict[key].savefig(pp, format='pdf')
        pp.close()

    # plt.show()
    return figure_dict

def data_generator(number_of_protocol, number_of_maggot, model_path, log_path, plot_path):
    index_protocol = 0
    index_maggot = 0
    while index_protocol < number_of_protocol and index_maggot < number_of_maggot:
        log_file_path = os.path.join(log_path, "protocol" + str(index_protocol) + "_maggot" + str(
            index_maggot) + ".pkl")
        log_file_path2 = os.path.join(log_path, "2ndOrderDynapseProtocol" + str(index_protocol) + "_maggot" + str(
            index_maggot) + ".pkl")
        plot_floder_path = os.path.join(plot_path,
                                        "protocol" + str(index_protocol) + "_maggot" + str(index_maggot))
        model_file_path = os.path.join(model_path, "maggots.pkl")
        data = [index_protocol, index_maggot, model_file_path, log_file_path, log_file_path2, plot_floder_path]
        print(data)
        yield data
        if index_maggot < number_of_maggot - 1:
            index_maggot += 1
        else:
            # if index0<len(gm)-1:
            index_maggot = 0
            index_protocol += 1


def video_maker(arguments):
    prefixdDRNWeight = 'DRNWeight'
    prefixd2ndWeight = '2ndWeight'
    print(1)
    plot_matshow_images = False
    plot_matshow_images2 = False
    # try:
    index_protocol, index_maggot, model_file_path, log_file_path, log_file_path2, plot_floder_path = arguments
    print(model_file_path)
    print(log_file_path)
    print(log_file_path2)
    print(plot_floder_path)
    if not os.path.exists(plot_floder_path):
        os.makedirs(plot_floder_path)
    if not os.path.exists(log_file_path):
        log_file_exist = False
    else:
        log_file_exist = True
    if not os.path.exists(log_file_path2):
        log_file2_exist = False
    else:
        log_file2_exist = True


    if not os.path.exists(os.path.join(plot_floder_path, prefixdDRNWeight+'video.mp4')):
        plot_matshow_images = True
    if not os.path.exists(os.path.join(plot_floder_path, prefixd2ndWeight+'video.mp4')):
        plot_matshow_images2 = True

    if log_file_exist:
        trace_reader = TraceReader(log_file_path)
        trace_reader.get_trace()
        trace = trace_reader.reconstruct_trace()
    else:
        trace = None
    if log_file2_exist:
        trace_reader2 = TraceReader(log_file_path2)
        trace_reader2.get_trace()
        trace2 = trace_reader2.reconstruct_trace()
    else:
        trace2 = None
    model_parameter_reader = TraceReader(model_file_path)
    model_parameter = model_parameter_reader.get_trace()
    # model_parameter= model_parameter_reader.reconstruct_trace()
    DRN=model_parameter[index_protocol][index_maggot]
    if log_file_exist or log_file2_exist:
        plot(plot_path=plot_floder_path, trace=trace, trace2=trace2, DRN=DRN)
    if log_file_exist and plot_matshow_images:
        matshow_images(trace, plot_floder_path, prefix=prefixdDRNWeight)
    if log_file2_exist and plot_matshow_images2:
        matshow_images2(trace2, plot_floder_path, prefix=prefixd2ndWeight)
    return 1
    # except:
    #     traceback.print_exc(file=sys.stderr)

if __name__ == "__main__":

    experiment_time = ""  # Fill in the experiment time, as the folder name shows

    trace_file_folder = "recording\\MaggotInPetriDish-v1_DynamicRouting\\"+experiment_time+"\\trace\\"
    log_path = "recording\\MaggotInPetriDish-v1_DynamicRouting\\"+experiment_time+"\\trace\\"
    model_path = "recording\\MaggotInPetriDish-v1_DynamicRouting\\"+experiment_time+"\\model\\"
    plot_path = "recording\\MaggotInPetriDish-v1_DynamicRouting\\"+experiment_time+"\\plot\\"

    number_of_protocol = 8
    number_of_maggot = 30
    PARALLEL = True

    if PARALLEL:
        numberOfProcess = 15
        pool = Pool(processes=numberOfProcess)
        results = pool.imap_unordered(video_maker, data_generator(number_of_protocol, number_of_maggot, model_path, log_path, plot_path))
        for a_result in results:
            print(a_result)
    else:

        for index_of_protocol in range(number_of_protocol):
            for index_of_maggot in range(number_of_maggot):
                log_file_path = os.path.join(log_path, "protocol"+str(index_of_protocol)+"_maggot"+str(index_of_maggot)+".pkl")
                log_file_path2 = os.path.join(log_path, "2ndOrderDynapseProtocol" + str(index_of_protocol) + "_maggot" + str(
                                                  index_of_maggot) + ".pkl")
                plot_floder_path = os.path.join(plot_path, "protocol"+str(index_of_protocol)+"_maggot"+str(index_of_maggot))
                model_file_path = os.path.join(model_path, "maggots.pkl")
                video_maker((index_of_protocol, index_of_maggot, model_file_path, log_file_path, log_file_path2,plot_floder_path))
