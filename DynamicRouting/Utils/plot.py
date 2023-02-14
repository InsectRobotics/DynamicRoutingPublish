import os
import warnings

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

def save_plots_as_pdf(figure_dict, plot_path, file_name='plots.pdf'):
    from matplotlib.backends.backend_pdf import PdfPages
    pdf_filepath = os.path.join(plot_path, file_name)
    pp = PdfPages(pdf_filepath)
    for key in figure_dict:
        figure_dict[key].savefig(pp, format='pdf')
    pp.close()
    print(plot_path)


def make_legend_arrow(legend, orig_handle,
                      xdescent, ydescent,
                      width, height, fontsize):
    p = mpatches.FancyArrow(0, 0.5 * height, 0.7*width, 0, length_includes_head=True)
    # print('legend', legend)
    return p

def kmeans_representer(data, n_clusters):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters).fit(data)
    distance = kmeans.transform(data)
    labels = kmeans.labels_
    group_index = []
    for index in range(n_clusters):
        a_group_index = np.where(np.equal(labels, index))[0]
        if len(a_group_index)>0:
            group_index.append(a_group_index)
    if len(group_index)<n_clusters:
        warnings.warn('Clustered data has %d groups, which is less then the expected %d groups.'
                      %(len(group_index), n_clusters))
    closet_index = np.zeros(len(group_index), dtype=int)
    for index in range(len(group_index)):
            closet_index[index] = group_index[index][np.argmin(distance[group_index[index], index])]
    return closet_index


if __name__ == "__main__":
    X = np.array([[1, 2], [1, 4], [1, 0],
                  [10, 2], [10, 4], [10, 0]])
    closet_index = kmeans_representer(X, n_clusters=3)
