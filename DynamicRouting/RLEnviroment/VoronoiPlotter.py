import os
import pickle

from DynamicSynapse.Utils.filereader import fileload
from DynamicSynapse.Utils.loggable import Loggable
from DynamicRouting.RLEnviroment.TaxiPlotter import plot
from DynamicSynapse.Utils.tracereader import TraceReader


experiment_time = "" # Fill in the experiment time, as the folder name shows
experiment_name = "VoronoiWorldGoal-v1_DynamicRouting"

trace_file_folder = os.path.join("recording", experiment_name, experiment_time, "trace")
model_path = os.path.join("recording", experiment_name, experiment_time, "model")
plot_path = os.path.join("recording", experiment_name, experiment_time, "plot")

episode_TR = TraceReader(log_file_path=os.path.join(trace_file_folder, "experiment_episode.pkl"))
episode_trace = episode_TR.get_trace()

step_TR = TraceReader(log_file_path=os.path.join(trace_file_folder, "experiment_step.pkl"))
step_trace = step_TR.get_trace()

plot(plot_path = plot_path, episode_trace=episode_trace, DRN=None, linthresh={'reward':10, 'step':100})