import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def multiple_histogram_graph():
    t_number_of_bins = 20
    h_number_of_bins = 20

    def read(name, sheet):
        data = pd.read_excel(name, sheet_name=sheet, header=None)
        matrix = data.values
        return matrix

    t_dataset = read('../Prove_Data.xlsx', 4)[:50, :].T
    h_dataset = read('../Prove_Data.xlsx', 3)[:180, :].T

    t_hist_range = (np.min(t_dataset), np.max(t_dataset))
    h_hist_range = (np.min(h_dataset), np.max(h_dataset))

    t_binned_data_sets = [np.histogram(d, range=t_hist_range, bins=t_number_of_bins)[0] for d in t_dataset]
    h_binned_data_sets = [np.histogram(d, range=h_hist_range, bins=h_number_of_bins)[0] for d in h_dataset]

    t_binned_maximums = np.max(t_binned_data_sets, axis=1)
    h_binned_maximums = np.max(h_binned_data_sets, axis=1)

    # t_x_locations = np.arange(0, sum(t_binned_maximums), np.max(t_binned_maximums))
    # h_x_locations = np.arange(0, sum(h_binned_maximums), np.max(h_binned_maximums))
    t_x_locations = np.array([0, 16, 32, 48])
    h_x_locations = np.array([0, 35, 70, 105])
    t_bin_edges = np.linspace(t_hist_range[0], t_hist_range[1], t_number_of_bins + 1)
    t_centers = 0.5 * (t_bin_edges + np.roll(t_bin_edges, 1))[:-1]
    h_bin_edges = np.linspace(h_hist_range[0], h_hist_range[1], h_number_of_bins + 1)
    h_centers = 0.5 * (h_bin_edges + np.roll(h_bin_edges, 1))[:-1]

    t_heights = np.diff(t_bin_edges)
    h_heights = np.diff(h_bin_edges)

    fig, axes = plt.subplots(2)
    ax = axes[0]
    for x_loc, binned_data in zip(t_x_locations, t_binned_data_sets):
        lefts = x_loc - 0.5 * binned_data
        ax.barh(t_centers, binned_data, height=t_heights, left=lefts)

    ax.set_xticks(t_x_locations)
    ax.set_xticklabels(['Dataset at 303K', 'Designed at 303K', 'Dataset at 363K', 'Designed at 363K'])
    ax.set_ylabel(u"Thermal Conductivity(W·m\u207B\u00B9·K\u207B\u00B9)")
    # ax.set_xlabel("Data sets")

    ax = axes[1]
    for x_loc, binned_data in zip(h_x_locations, h_binned_data_sets):
        lefts = x_loc - 0.5 * binned_data
        ax.barh(h_centers, binned_data, height=h_heights, left=lefts)

    ax.set_xticks(h_x_locations)
    ax.set_xticklabels(['Dataset at 303K', 'Designed at 303K', 'Dataset at 363K', 'Designed at 363K'])
    ax.set_ylabel(u"Heat Capacity(J·kg\u207B\u00B9·K\u207B\u00B9)")
    # ax.set_xlabel("Data sets")

    plt.show()


def t_violin_graph():
    t_dataset = pd.read_csv('../t.csv')
    f, axes = plt.subplots()
    sns.set_theme(style='white', font='sans-serif')
    sns.violinplot(x=t_dataset.Temperature, y=t_dataset.Thermal_Conductivity, hue=t_dataset.Source, data=t_dataset,
                   palette="Paired", bw=.2, cut=1, linewidth=1, inner="quartile", split=True)
    axes.set(ylim=(0.1, 0.25))
    axes.set_ylabel(u"Thermal Conductivity (W·m\u207B\u00B9·K\u207B\u00B9)")
    axes.set_xlabel('Temperature (K)')
    # sns.despine(left=True, bottom=True)


def h_violin_graph():
    h_dataset = pd.read_csv('../h.csv')
    f, axes = plt.subplots()
    sns.set_theme(style='white', font='serif')
    sns.violinplot(x=h_dataset.Temperature, y=h_dataset.Heat_Capacity, hue=h_dataset.Source, data=h_dataset,
                   palette="Paired", bw=.2, cut=1, linewidth=1, inner="quartile", split=True)
    axes.set(ylim=(200, 800))
    axes.set_ylabel(u"Heat Capacity (J·kg\u207B\u00B9·K\u207B\u00B9)")
    axes.set_xlabel('Temperature (K)')
    # sns.despine(left=True, bottom=True)


def score_violin_graph():
    s_dataset = pd.read_csv('../score.csv')
    f, axes = plt.subplots()
    sns.set_theme(style='white', font='serif')
    sns.violinplot(x=s_dataset.Temperature, y=s_dataset.Score, hue=s_dataset.Source, data=s_dataset,
                   palette="Paired", bw=.2, cut=1, linewidth=1, inner="quartile", split=True)
    axes.set(ylim=(0.02, 0.06))
    axes.set_ylabel("Property Score")
    axes.set_xlabel('Temperature (K)')


def t_cross_violin_graph():
    t_dataset = pd.read_csv('../tc.csv')
    f, axes = plt.subplots()
    sns.set_theme(style='white', font='serif')
    sns.violinplot(x=t_dataset.Temperature, y=t_dataset.Thermal_Conductivity,
                   hue=t_dataset.Standard_Temperature, data=t_dataset,
                   palette="Paired", bw=.2, linewidth=1, inner="quartile", split=True)
    axes.set(ylim=(0.1, 0.25))
    axes.set_ylabel(u"Thermal Conductivity (W·m\u207B\u00B9·K\u207B\u00B9)")
    axes.set_xlabel('Temperature (K)')
    plt.show()
    # sns.despine(left=True, bottom=True)


def h_cross_violin_graph():
    h_dataset = pd.read_csv('../hc.csv')
    f, axes = plt.subplots()
    sns.set_theme(style='white', font='serif')
    sns.violinplot(x=h_dataset.Temperature, y=h_dataset.Heat_Capacity,
                   hue=h_dataset.Standard_Temperature, data=h_dataset,
                   palette="Paired", bw=.2, linewidth=1, inner="quartile", split=True)
    axes.set(ylim=(200, 800))
    axes.set_ylabel(u"Heat Capacity (J·kg\u207B\u00B9·K\u207B\u00B9)")
    axes.set_xlabel('Temperature (K)')
    plt.show()
    # sns.despine(left=True, bottom=True)


def score_cross_violin_graph():
    s_dataset = pd.read_csv('../score_c.csv')
    f, axes = plt.subplots()
    sns.set_theme(style='white', font='serif')
    sns.violinplot(x=s_dataset.Temperature, y=s_dataset.Score,
                   hue=s_dataset.Standard_Temperature, data=s_dataset,
                   palette="Paired", bw=.2, linewidth=1, inner="quartile", split=True)
    axes.set(ylim=(0.02, 0.06))
    axes.set_ylabel("Property Score")
    axes.set_xlabel('Temperature (K)')
    plt.show()
    # sns.despine(left=True, bottom=True)


if __name__ == '__main__':
    # multiple_histogram_graph()
    t_violin_graph()
    # h_violin_graph()
    # score_violin_graph()
    # t_cross_violin_graph()
    # h_cross_violin_graph()
    # score_cross_violin_graph()
