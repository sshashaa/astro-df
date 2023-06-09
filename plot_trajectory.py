import sys
import os.path as o
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))

def plot_with_conf_interval(file1, file2, ylabel, output):
    conf_level = 0.95

    # Load data from the first file
    data1 = np.load(file1)
    mean1 = np.mean(data1, axis=0)
    std1 = np.std(data1, axis=0)
    conf_interval1 = []

    # Calculate confidence interval for the first file
    for i in range(len(mean1)):
        critical_value1 = stats.t.ppf((1 + conf_level) / 2, len(data1[:,0]) - 1)
        margin_of_error1 = critical_value1 * (std1[i] / np.sqrt(len(data1[:,0])))
        conf_interval1.append((mean1[i] - margin_of_error1, mean1[i] + margin_of_error1))

    # Load data from the second file
    data2 = np.load(file2)
    mean2 = np.mean(data2, axis=0)
    std2 = np.std(data2, axis=0)
    conf_interval2 = []

    # Calculate confidence interval for the second file
    for i in range(len(mean2)):
        critical_value2 = stats.t.ppf((1 + conf_level) / 2, len(data2[:,0]) - 1)
        margin_of_error2 = critical_value2 * (std2[i] / np.sqrt(len(data2[:,0])))
        conf_interval2.append((mean2[i] - margin_of_error2, mean2[i] + margin_of_error2))

    # Generate the plot
    x = range(len(conf_interval1))
    y_lower1 = [d[0] for d in conf_interval1]
    y_upper1 = [d[1] for d in conf_interval1]

    y_lower2 = [d[0] for d in conf_interval2]
    y_upper2 = [d[1] for d in conf_interval2]

    plt.plot(x, mean1, color='C0', label='ASTRO-DF with Direct Search')
    plt.fill_between(x, y_lower1, y_upper1, alpha=0.2, color='C0')

    plt.plot(x, mean2, color='C1', label='ASTRO-DF without Direct Search')
    plt.fill_between(x, y_lower2, y_upper2, alpha=0.2, color='C1')

    plt.xlabel('Iteration k')
    plt.ylabel(ylabel)
    plt.legend(loc="upper left")
    plt.savefig(output, format='pdf')
    plt.clf()  # Clear the figure for the next iteration

# Plot RSBR
plot_with_conf_interval('delta_rsbr_rf.npy', 'delta_rsbr_dh.npy', 'Delta', f"experiments/plots/delta_rsbr_trajectory.pdf")
plot_with_conf_interval('function_rsbr_rf.npy', 'function_rsbr_dh.npy', 'Function Estimates', f"experiments/plots/function_rsbr_trajectory.pdf")

# Plot SAN
plot_with_conf_interval('delta_san_rf.npy', 'delta_san_dh.npy', 'Delta', f"experiments/plots/delta_san_trajectory.pdf")
plot_with_conf_interval('function_san_rf.npy', 'function_san_dh.npy', 'Function Estimates', f"experiments/plots/function_san_trajectory.pdf")
plot_with_conf_interval('budget_san_rf.npy', 'budget_san_dh.npy', 'Expended Budget', f"experiments/plots/budget_san_trajectory.pdf")
