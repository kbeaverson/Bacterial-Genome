import numpy as np
from scipy import stats


def calculate_normal_confidence_intervals(confusion_matrices_list):
    accuracy_list = []
    sensitivity_list = []
    specificity_list = []
    bal_acc_list = []

    for confusion_matrix in confusion_matrices_list:
        sensitivity = confusion_matrix[1,1] / confusion_matrix[1,:].sum()
        specificity = confusion_matrix[0,0] / confusion_matrix[0,:].sum()

        accuracy_list.append((confusion_matrix[0,0] + confusion_matrix[1,1]) / confusion_matrix.sum())
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)
        bal_acc_list.append(np.mean([sensitivity, specificity]))

    location, std = np.mean(accuracy_list), np.std(accuracy_list)
    accuracy_ci = stats.norm.interval(0.95, loc = location, scale = std)

    location, std = np.mean(sensitivity_list), np.std(sensitivity_list)
    sensitivity_ci = stats.norm.interval(0.95, loc = location, scale = std)

    location, std = np.mean(specificity_list), np.std(specificity_list)
    specificity_ci = stats.norm.interval(0.95, loc = location, scale = std)

    location, std = np.mean(bal_acc_list), np.std(bal_acc_list)
    bal_acc_ci = stats.norm.interval(0.95, loc = location, scale = std)

    return accuracy_ci, sensitivity_ci, specificity_ci, bal_acc_ci