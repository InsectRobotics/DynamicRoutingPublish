from DynamicSynapse.Utils.tracereader import TraceReader
from matplotlib.backends.backend_pdf import PdfPages
import os
import matplotlib.pyplot as plt
import numpy as np
from DynamicRouting.Utils.TaxiUtils import decode

def plot(plot_path='', episode_trace=None, DRN=None):

    figure_dict = dict()
    figure_dict['episode_reward'] = plt.figure()
    figure12lines1, = plt.plot(episode_trace['episode_reward'])
    figure12lines2, = plt.plot(episode_trace['average_episode_reward'])
    plt.legend([figure12lines1, figure12lines2], ['Episode Reward', 'Average Episode Reward'], loc=4)
    plt.xlabel('Episode')
    plt.title('Episode Reward')
    plt.grid()

    figure_dict['episode_reward_in_step'] = plt.figure()
    episode_reward_in_step = np.cumsum(episode_trace['step'])
    figure14lines1, = plt.plot(episode_reward_in_step, episode_trace['episode_reward'])
    figure14lines2, = plt.plot(episode_reward_in_step, episode_trace['average_episode_reward'])
    plt.legend([figure14lines1, figure14lines2], ['Episode Reward', 'Average Episode Reward'], loc=4)
    plt.xlabel('Steps')
    plt.title('Episode Reward at Steps')
    plt.grid()

    figure_dict['episode_step'] = plt.figure()
    figure13lines1, = plt.plot(episode_trace['step'])
    plt.legend([figure13lines1], ['steps'], loc=4)
    plt.xlabel('Episode')
    plt.title('Episode Steps')
    plt.grid()

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

    plt.show()
    return figure_dict

if __name__=="__main__":
    # log_file_path = "F:\\recording\\Taxi-v3_DynamicRouting\\2021-07-04_00-55-05\\trace\\experiment_episode.pkl"
    log_file_path = "F:\\recording\\LunarLander-v2_DynamicRouting\\2021-07-29_10-53-19\\trace\\experiment_episode.pkl"
    plot_path = "F:\\recording\\LunarLander-v2_DynamicRouting\\2021-07-29_10-53-19\\plot"
    trace_reader = TraceReader(log_file_path)
    trace_reader.get_trace()
    trace = trace_reader.reconstruct_trace()
    plot(plot_path=plot_path, episode_trace=trace, DRN=None)
