# implement a classification model based on the regression for the best frequency section
#%% imports
import numpy as np 
import matplotlib.pyplot as plt 
import sys
sys.path.insert(0, '/Users/jakobschaefer/Documents/rf-ultrasound')  # necessary to import functions from other directory
import functions.definitions as definitions
from scipy import stats
import sklearn.metrics as metrics

#%% load data 
version = '230330'
input_directory = '/Users/jakobschaefer/Documents/RF_US_AI/MLP_QUS_Parameter/Input_data/'
file_path_C3 = input_directory+'mean_slopes_all_recordings_freqspec_C3_'+version+'.npy'
file_path_L15 = input_directory+'mean_slopes_all_recordings_freqspec_L15_'+version+'.npy'
C3_values = np.load(file_path_C3, allow_pickle=True)
L15_values = np.load(file_path_L15, allow_pickle=True)

# load CAP values
target_directory = '/Users/jakobschaefer/Documents/RF_US_AI/MLP_QUS_Parameter/Targets/'
target_path_C3 = target_directory+'targets_CAP_freqspec_C3_'+version+'.npy'
target_path_L15 = target_directory+'targets_CAP_freqspec_L15_'+version+'.npy'
target_values_C3 = np.load(target_path_C3, allow_pickle=True)
target_values_L15 = np.load(target_path_L15, allow_pickle=True)
CAP_values_C3 = list(target_values_C3.item().values())
CAP_values_L15 = list(target_values_L15.item().values())

#%% perform spearman correlation
#C3
spearman_rho_C3_freqspec = []
spearman_p_C3_freqspec = []
for frequency in range(0, len(C3_values[0])):
    spearman_rho_C3_one_freq, spearman_p_C3_one_freq = stats.spearmanr(C3_values[:,frequency], CAP_values_C3)
    spearman_rho_C3_freqspec.append(spearman_rho_C3_one_freq)
    spearman_p_C3_freqspec.append(spearman_p_C3_one_freq)
# print best spearman rho and corresponding frequency section
best_spearman_rho_C3 = np.max(np.abs(spearman_rho_C3_freqspec))
best_frequency_section_C3 = np.argmax(np.abs(spearman_rho_C3_freqspec))
print('Best spearman rho for C3: '+str(best_spearman_rho_C3)+' at frequency section '+str(best_frequency_section_C3))

# L15 
spearman_rho_L15_freqspec = []
spearman_p_L15_freqspec = []
for frequency in range(0, len(L15_values[0])):
    spearman_rho_L15_one_freq, spearman_p_L15_one_freq = stats.spearmanr(L15_values[:,frequency], CAP_values_L15)
    spearman_rho_L15_freqspec.append(spearman_rho_L15_one_freq)
    spearman_p_L15_freqspec.append(spearman_p_L15_one_freq)
# print best spearman rho and corresponding frequency section
best_spearman_rho_L15 = np.max(np.abs(spearman_rho_L15_freqspec))
best_frequency_section_L15 = np.argmax(np.abs(spearman_rho_L15_freqspec))
print('Best spearman rho for L15: '+str(best_spearman_rho_L15)+' at frequency section '+str(best_frequency_section_L15))



#%% perform frequency spezific linear regression
transducer = 'C3'

pearson_per_frequency, attenuation_values_best_freq, redcap_value,best_slope, best_intercept, best_r_value, best_p_value, best_mymodel = definitions.get_attenuation_best_freq(transducer, version, input_directory, target_directory)

# calculate the predicted CAP values based on the model for the best frequency section
predicted_CAP_values = best_slope*attenuation_values_best_freq+best_intercept

# calculate the mean absolute error
if transducer == 'C3':
    CAP_values = CAP_values_C3
elif transducer == 'L15':
    CAP_values = CAP_values_L15

mean_absolute_error = np.mean(np.abs(CAP_values-predicted_CAP_values))

# split the output into 4 clases based on the CAP values 
# define cutoff values
# Karlas et al. (2017): Individual patient data meta-analysis of controlled attenuation parameter (CAP) technology for assessing steatosis
# cutoff_1 = 248
# cutoff_2 = 268
# cutoff_3 = 280
# Kuroda et al. (2021): Diagnostic accuracy of ultrasound-guided attenuation parameter as a noninvasive test for steatosis in non-alcoholic fatty liver disease
cutoff_1 = 246
cutoff_2 = 274
cutoff_3 = 287

# define the classes
pred_class_1 = []
pred_class_2 = []
pred_class_3 = []
pred_class_4 = []

# split the predicted CAP values into the classes
for i in range(len(predicted_CAP_values)):
    if predicted_CAP_values[i] < cutoff_1:
        pred_class_1.append(i)
    elif predicted_CAP_values[i] < cutoff_2:
        pred_class_2.append(i)
    elif predicted_CAP_values[i] < cutoff_3:
        pred_class_3.append(i)
    else:
        pred_class_4.append(i)

# split the actual CAP values into the classes
actual_class_1 = []
actual_class_2 = []
actual_class_3 = []
actual_class_4 = []

for i in range(len(CAP_values)):
    if CAP_values[i] < cutoff_1:
        actual_class_1.append(i)
    elif CAP_values[i] < cutoff_2:
        actual_class_2.append(i)
    elif CAP_values[i] < cutoff_3:
        actual_class_3.append(i)
    else:
        actual_class_4.append(i)

# calculate the accuracy of the classification
# class 1 vs. class 2-4 
correct_class_1 = 0
for i in range(len(pred_class_1)):
    if pred_class_1[i] in actual_class_1:
        correct_class_1 += 1
accuracy_class_1 = correct_class_1/len(pred_class_1)

# class 1,2 vs. class 3,4
correct_class_2 = 0
for i in range(len(pred_class_2)):
    if pred_class_2[i] in actual_class_2:
        correct_class_2 += 1
accuracy_class_2 = correct_class_2/len(pred_class_2)

# class 1,2,3 vs. class 4
correct_class_3 = 0
for i in range(len(pred_class_3)):
    if pred_class_3[i] in actual_class_3:
        correct_class_3 += 1
accuracy_class_3 = correct_class_3/len(pred_class_3)

# %%
# calculate AUROC for a classification
cutoff = cutoff_3

def calculate_AUROC(cutoff, predicted_CAP_values, CAP_values):
    pred_classes = []
    actual_classes = []
    for i in range(len(CAP_values)):
        if CAP_values[i] < cutoff:
            actual_classes.append(0)
        else:
            actual_classes.append(1)
        
        if predicted_CAP_values[i] < cutoff:
            pred_classes.append(0)
        else:
            pred_classes.append(1)
    
    AUROC = metrics.roc_auc_score(actual_classes, pred_classes)
    sensitivity = metrics.recall_score(actual_classes, pred_classes)
    specificity = metrics.recall_score(actual_classes, pred_classes, pos_label=0)

    return AUROC, sensitivity, specificity

AUROC, sensitifity, specificity = calculate_AUROC(cutoff, predicted_CAP_values, CAP_values)

print('AUROC: '+str(AUROC))
print('Sensitivity for cutoff '+str(cutoff)+': '+str(sensitifity))
print('Specificity for cutoff '+str(cutoff)+': '+str(specificity))
# %%
