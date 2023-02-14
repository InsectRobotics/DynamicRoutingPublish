import platform
import os
import shutil
import time


def choose_recording_path(experiment, TimeOfRecording):
    path = '../../recording/' + experiment + '/' + TimeOfRecording + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def choose_result_path(experiment, TimeOfRecording):
    path = '../../recording/' + experiment + '/' + TimeOfRecording + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    return os.path.normpath(path)


def common_path(experiment):
    TIME_OF_RECORDING = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    result_path = choose_result_path(experiment, TIME_OF_RECORDING)
    code_path = os.path.join(result_path, 'src')
    if not os.path.exists(code_path):
        os.makedirs(code_path)
    data_path = os.path.join(result_path, 'trace')
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    plot_path = os.path.join(result_path, 'plot')
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    model_path = os.path.join(result_path, 'model')
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    recording_path = choose_recording_path(experiment, TIME_OF_RECORDING)
    video_path = os.path.join(recording_path, 'video')
    if not os.path.exists(video_path):
        os.makedirs(video_path)
    task_path = os.path.join(recording_path, 'task')
    if not os.path.exists(task_path):
        os.makedirs(task_path)
    pathes = {'TIME_OF_RECORDING': TIME_OF_RECORDING,
              'result_path': result_path,
              'code_path': code_path,
              'data_path': data_path,
              'plot_path': plot_path,
              'model_path': model_path,
              'recording_path': recording_path,
              'video_path': video_path,
              'task_path': task_path}
    return pathes


def backup_code(source_code_path='', backup_path='', experiment='Default'):
    if not source_code_path:
        source_code_path = os.getcwd()
    if not backup_path:
        TIME_OF_RECORDING = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        result_path = choose_result_path(experiment, TIME_OF_RECORDING)
        backup_path = os.path.join(result_path, 'src')
    for subdir, dirs, files in os.walk(source_code_path):  # replace the . with your starting directory
        for file in files:
            if file.endswith(".py"):
                path_file = os.path.join(subdir, file)
                relative_path = os.path.relpath(subdir, source_code_path)
                destPath = os.path.join(backup_path, relative_path)
                if not os.path.exists(destPath):
                    os.makedirs(destPath)
                shutil.copy2(path_file, os.path.join(backup_path, relative_path))
