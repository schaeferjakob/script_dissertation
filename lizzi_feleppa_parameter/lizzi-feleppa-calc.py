#%%
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, '/Users/jakobschaefer/Documents/rf-ultrasound')  # necessary to import functions from other directory
import functions.definitions as definitions
from scipy import stats
from tqdm import tqdm
from scipy import ndimage
import copy
import functions.def_notion_API as notion_def
#%%
# choose if plots should be displayed or not
show_plots = False # True or False

# choose recording to display 
version = '230330'
directory = '/Volumes/Extreme_SSD/Aufnahmen_vorverarbeitet/C3_and_L15_without_TGC_'+version
transducer = 'L15_large'

ref_parameter = 'CAP' # 'E' or 'CAP'
window_function = 'hamming'


#low_frequency_cutoff = 0 # sometimes the fft gives out really high values at the beginning, so we cut them out 
Phantom = 'Wat_Phan' # 'Phantom' or 'Met_Phan' or 'AIR00000' or 'Wat_Phan' or 'fft_all_recordings'

# define where to choose ROI from
ROI_type = 2 # 0: whole image, 1: middle of scanline (=around focal point), 2: specific ROI (defined below)
len_ROI = 500 # the length of the examined ROI in px (only for ROI_type = 1)
freq_shift_comp = 1000 # for calibration with Wat_Phan -> has much smaler freq. downshift than Patient recordings



if transducer == 'L15_large':
    specific_ROI = [193,1626] # [start, end] of ROI
    start_scanline = 14
    number_of_scanlines = 78
    border = 14
if transducer == 'C3_large':
    specific_ROI = [0,1086] # [start, end] of ROI
    start_scanline = 43
    number_of_scanlines = 192
    border = 30 # dB
# define where is the middle of reflection of the calibration phantom
if transducer == 'L15_large':
    reflection_max = 1316
if transducer == 'C3_large':
    reflection_max = 1800


end_scanline = start_scanline + number_of_scanlines

# define if results should be saved to notion
safe_to_notion = True # True or False
database_id = 'f94121511c1f49e09d1b1ba9af40157b'
# define notion parameters
notion_transducer = transducer
notion_window_func = str(window_function+' (not important)')
notion_cropping_properties  = str('Min. depth: '+str(specific_ROI[0])+'\nMax. depth: '+str(specific_ROI[1])+'\nStart scanline: '+str(start_scanline)+'\nEnd scanline: '+str(end_scanline)+'\nBorder: '+str(border)+'dB')
#%%
# get values from redcap survey
dir_redcap_survey = '/Users/jakobschaefer/Documents/REDCap_Survey_'+version+'.csv'

if Phantom != 'fft_all_recordings':   
    # calculate fft of phantom recording
    phantom_file = '/Volumes/Extreme_SSD/Aufnahmen_vorverarbeitet/Calibration/'+Phantom+'/'+transducer+'.rf_no_tgc.npy'
    phantom_data = np.load(phantom_file)
    phantom_data = phantom_data[:,:,0]
    # cut out specific ROI from phantom_data
    if ROI_type == 1:
        phantom_data = phantom_data[:,len(phantom_data[0,:])//2-(len_ROI//2):len(phantom_data[0,:])//2+(len_ROI//2)]
    if ROI_type == 2:
        if Phantom == 'Wat_Phan':
            # if specific_ROI[1]+freq_shift_comp > len(phantom_data[0,:]):
            #     freq_shift_comp = len(phantom_data[0,:])-specific_ROI[1]
            # phantom_data = phantom_data[:,specific_ROI[0]+freq_shift_comp:specific_ROI[1]+freq_shift_comp]
            len_specific_ROI = specific_ROI[1]-specific_ROI[0]
            phantom_data = phantom_data[:,reflection_max-(len_specific_ROI//2):reflection_max+(len_specific_ROI//2)]
            if len(phantom_data[0,:]) > len_specific_ROI:
                phantom_data = phantom_data[:,:-1]
        else:
            phantom_data = phantom_data[:,specific_ROI[0]:specific_ROI[1]]



    # get sampling frequency from yaml file
    yaml_file = phantom_file[:-14]+'.rf.yaml'
    sampling_rate, t = definitions.read_yaml_file(yaml_file, phantom_data)

    phantom_y_fft = []
    for scanline in range(len(phantom_data[:,0])):
        phantom_x_fft, phantom_y_fft_scanline = definitions.amplitude_spectrum(sampling_rate, phantom_data[scanline,:], window_function)
        phantom_y_fft.append(phantom_y_fft_scanline)
    phantom_y_fft = np.array(phantom_y_fft)

    phantom_y_fft_square = np.square(phantom_y_fft)
    calibration_spectrum = np.mean(phantom_y_fft_square, axis = 0)
    #calibration_spectrum = calibration_spectrum[low_frequency_cutoff:]

elif Phantom == 'fft_all_recordings':
    try:
        phantom_fft = np.load('/Volumes/Extreme_SSD/Aufnahmen_vorverarbeitet/Calibration/mean_fft_all_recordings/mean_fft_all_recordings_'+str(specific_ROI[0])+'-'+str(specific_ROI[1])+'_'+version+'_'+transducer+'.npy')
    except:
        # calculate FFT for all recordings
        fft_all_recordings = []
        for recording in tqdm(os.listdir(directory)):
            # load file
            array = np.load(directory+'/'+recording+'/'+transducer+'.rf_no_tgc.npy')

            # cut off sides of array 
            array = array[start_scanline:end_scanline,:,:]

            # sort out recordings where array is shorter than a certain length
            if len(array[0,:,0]) >= 2704:
            
                array = array[:,specific_ROI[0]:specific_ROI[1],:]

                # get sampling rate out of yaml file
                yaml_file = str(directory)+'/'+recording+'/'+transducer+'.rf.yaml'
                sampling_rate, t = definitions.read_yaml_file(yaml_file, array)

                yfft_all_scanlines = []
                # get fft from every scanline and calculate mean
                for scanline in range(len(array[:,0,0])):

                    # perform FFT on first scanline in first frame
                    x_fft,y_fft = definitions.amplitude_spectrum(sampling_rate, array[scanline,:,0], window_function)

                    yfft_all_scanlines.append(y_fft)
                yfft_all_scanlines = np.array(yfft_all_scanlines)
                mean_yfft_all_scanlines = np.mean(yfft_all_scanlines, axis=0)



                # append FFT to list
                fft_all_recordings.append(mean_yfft_all_scanlines)
        fft_all_recordings = np.array(fft_all_recordings)

        fft_all_recordings = np.square(fft_all_recordings)
        mean_fft_all_recordings = np.mean(fft_all_recordings, axis = 0)

        # save mean_fft_all_recordings
        np.save('/Volumes/Extreme_SSD/Aufnahmen_vorverarbeitet/Calibration/mean_fft_all_recordings/mean_fft_all_recordings_'+str(specific_ROI[0])+'-'+str(specific_ROI[1])+'_'+version+'_'+transducer+'.npy', (mean_fft_all_recordings, x_fft))
        phantom_fft = mean_fft_all_recordings, x_fft

    phantom_y_fft = phantom_fft[0]
    phantom_x_fft = phantom_fft[1]
    calibration_spectrum = np.square(phantom_y_fft)
#phantom_x_fft = phantom_x_fft[low_frequency_cutoff:]

# get max_value from calibration_spectrum and calculate -15dB border'
max_value_calibration = np.max(calibration_spectrum)
index_max_value_calibration = np.argmax(calibration_spectrum)
border_value = max_value_calibration/np.power(10, border/20)
# define upper border of calibration_spectrum
for frequency in range(index_max_value_calibration, len(calibration_spectrum)):
    if calibration_spectrum[frequency] < border_value:
        upper_index_border = frequency
        break
    else: 
        upper_index_border = len(calibration_spectrum)-1
# define lower border of calibration_spectrum
for frequency in range(index_max_value_calibration, 0, -1):
    if calibration_spectrum[frequency] < border_value:
        lower_index_border = frequency
        break
    else: 
        lower_index_border = 0

new_phantom_x_fft = phantom_x_fft[lower_index_border:upper_index_border]
new_calibration_spectrum = calibration_spectrum[lower_index_border:upper_index_border]

# get redcap values out of csv file
Patient_ID, Median_E, Median_CAP, IQR_E, IQR_CAP, scan_conditions = definitions.get_fibroscan_values(dir_redcap_survey)

#%%
# calculate fft of recording
quotient_all_recordings = []
MBF_all_recordings = []
slopes_all_recordings = []
intercepts_all_recordings = []
redcap_value = []
list_patient_ID_different_order = []
for recording in tqdm(os.listdir(directory)):

    file = directory+'/'+recording+'/'+transducer+'.rf_no_tgc.npy'
    data = np.load(file)

    # cut off sides of array 
    data = data[start_scanline:end_scanline,:,:]

    # define sampling frequency
    yaml_file = directory+'/'+recording+'/'+transducer+'.rf.yaml'
    sampling_rate, t = definitions.read_yaml_file(yaml_file, data)
    
    # get redcap value out of lists from csv file
    number_redcap_value = Patient_ID.index(recording) 
    list_patient_ID_different_order.append(recording)
    #choose from which list to get redcap values
    if ref_parameter == 'E':
        list_values = Median_E
    elif ref_parameter == 'CAP': 
        list_values = Median_CAP
    else:
        print('ERROR: ref_parameter not defined correctly!')
    redcap_value.append(list_values[number_redcap_value])

    # just take first frame
    data = data[:,:,0]

    # cut out specific ROI from data
    if ROI_type == 1:
        data = data[:,len(data[0,:])//2-(len_ROI//2):len(data[0,:])//2+(len_ROI//2)]
    if ROI_type == 2:
        if Phantom == 'Wat_Phan':
            data = data[:,specific_ROI[0]:specific_ROI[1]]
        else:
            data = data[:,specific_ROI[0]:specific_ROI[1]]



    # # adapt the length of pahntom_data and data together (by zero-padding)
    # if len(data[0,:]) > len(phantom_data[0,:]):
    #     phantom_data = np.pad(phantom_data, ((0,0),(0,len(data[0,:])-len(phantom_data[0,:]))), 'constant', constant_values=0)
    # if len(data[0,:]) < len(phantom_data[0,:]):
    #     data = np.pad(data, ((0,0),(0,len(phantom_data[0,:])-len(data[0,:]))), 'constant', constant_values=0)

    # calculate fft of recording
    y_fft = []
    for scanline in range(len(data[:,0])-1):
        x_fft, y_fft_scanline = definitions.amplitude_spectrum(sampling_rate, data[scanline,:], window_function)
        y_fft.append(y_fft_scanline)
    y_fft = np.array(y_fft)

    # square of fft
    y_fft_square = np.square(y_fft)
    # average of fft (by that all scanlines are averaged together to one spectrum)
    average_power_spectrum = np.mean(y_fft_square, axis = 0)
    #average_power_spectrum = average_power_spectrum[low_frequency_cutoff:]
    #x_fft = x_fft[low_frequency_cutoff:]

    # # JUST TO TRY SOMETHING -> DELETE LATER!!
    # artificial_frequency_shift = np.where(x_fft > 0.05e6)[0][0]
    # average_power_spectrum = np.append(np.zeros(artificial_frequency_shift),average_power_spectrum[:-artificial_frequency_shift])    

    # define index of border
    new_x_fft = x_fft[lower_index_border:upper_index_border]
    new_average_power_spectrum = average_power_spectrum[lower_index_border:upper_index_border]

    # quotient of recording fft and phantom fft 
    try:
        quotient = new_average_power_spectrum/new_calibration_spectrum
   
    
        # Calculate Lizzi-Feleppa-Parameters
        x_quotient = new_x_fft

        # linear regression
        slope_quotient, intercept_quotient, r_value, p_value, mymodel = definitions.linear_regression(x_quotient, quotient)

        # define regression line
        def myfunc(x):
            return slope_quotient * x + intercept_quotient


        # calculate Midband Fit
        MBF = slope_quotient * (x_quotient[round(len(x_quotient)/2)]) + intercept_quotient

        # collect values for all recordings
        quotient_all_recordings.append(quotient)
        MBF_all_recordings.append(MBF)
        slopes_all_recordings.append(slope_quotient)
        intercepts_all_recordings.append(myfunc(0))

        if show_plots == True:
        
            # plot arrays
            plt.figure(figsize=(9,3)) # figsize in inches (width, height) -> multiply with  0.393701 to get cm
            amplification_fft_recording = np.max(calibration_spectrum)/np.max(average_power_spectrum)
            plt.plot(x_fft, average_power_spectrum*amplification_fft_recording, color = '#00305D', label = 'Participant recording') #(*'+str(int(amplification_fft_recording))+')')
            plt.plot(phantom_x_fft, calibration_spectrum, color = '#FF0000', label = 'Calibration recording')
            plt.title('Frequency Spectrum', fontsize = 13, fontweight = 'bold')#'FFT of '+Phantom+' and '+recording+' '+transducer)
            plt.xlabel('Frequency [Hz]', fontsize = 12)
            plt.ylabel('Amplitude', fontsize = 12)
            plt.legend( loc='upper right', fontsize = 12)
            plt.show()

            # plot shortened ffts
            plt.figure(figsize=(9,3)) # figsize in inches (width, height) -> multiply with  0.393701 to get cm
            amplification_new_fft_recording = amplification_fft_recording
            plt.plot(new_x_fft, new_average_power_spectrum*amplification_new_fft_recording, color = '#00305D', label = 'Participant recording') #(*'+str(int(amplification_new_fft_recording))+')')
            plt.plot(new_phantom_x_fft, new_calibration_spectrum, color = '#FF0000', label = 'Calibration recording')
            #plt.title('Scanline '+str(scanline)+' of '+recording+' '+transducer)
            plt.xlabel('Frequency [Hz]', fontsize = 12)
            plt.ylabel('Amplitude', fontsize = 12)
            plt.legend( loc='upper right', fontsize = 12)
            plt.title('Shortened Frequency Spectrum', fontsize = 13, fontweight = 'bold')#of '+Phantom+' and '+recording+' '+transducer)
            plt.show()

            # plot quotient
            plt.figure(figsize=(9,3)) # figsize in inches (width, height) -> multiply with  0.393701 to get cm
            plt.plot(new_x_fft, quotient, color = '#00305D', label = 'Quotient of Part.recording/Calibration', zorder = 1)
            plt.plot(x_quotient, mymodel, color = 'orange', label = 'Regression line', zorder  = 2)
            # intercept
            plt.scatter(0,myfunc(0), color = '#FF0000', zorder = 3)
            plt.annotate('INT', xy=(0,myfunc(0)), xytext=(0,myfunc(0)+max(quotient)/20), color = '#FF0000',fontsize=12)
            # MBF
            plt.scatter( x_quotient[round(len(x_quotient)/2)],MBF, color = '#FF0000', zorder = 3)
            plt.annotate('MBF', xy=( x_quotient[round(len(x_quotient)/2)],MBF), xytext=( x_quotient[round(len(x_quotient)/2)],MBF+max(quotient)/20), color = '#FF0000',fontsize=12)
            plt.title('Calibrated Spectrum and Regression', fontsize = 13, fontweight = 'bold') #of '+Phantom+' and '+recording+' '+transducer)
            plt.xlabel('Frequency [Hz]', fontsize = 12)
            plt.ylabel('Quotient', fontsize = 12)
            plt.legend( loc='upper right', fontsize = 12)
            plt.show()

    except:
        # remove last element of redcap_value and list_patient_ID_different_order
        redcap_value.pop()
        list_patient_ID_different_order.pop()

        print('Error in calculating quotient of '+recording+' '+transducer)
        continue    
quotient_all_recordings = np.array(quotient_all_recordings)
slopes_all_recordings = np.array(slopes_all_recordings)
intercepts_all_recordings = np.array(intercepts_all_recordings)

# save results into AI/MLP folder
result_directory = '/Users/jakobschaefer/Documents/RF_US_AI/MLP_QUS_Parameter/Lizzi_Feleppa_Results'
# create dict with results
dict_results = {'Patient_ID': list_patient_ID_different_order, 'MBF': MBF_all_recordings, 'slope': slopes_all_recordings, 'intercept': intercepts_all_recordings, ref_parameter: redcap_value}
# save dict as npy file
np.save(result_directory+'/Lizzi_Feleppa_Results_'+version+'_'+transducer+'.npy', dict_results)



#%% 
# plot regression results of lizzi_feleppa parameters
# define which parameter should be plotted
lizzi_feleppa_parameter = 'MBF' # 'MBF', 'intercept', 'slope'

# load dict_results
result_directory = '/Users/jakobschaefer/Documents/RF_US_AI/MLP_QUS_Parameter/Lizzi_Feleppa_Results'
dict_results = np.load(result_directory+'/Lizzi_Feleppa_Results_'+version+'_'+transducer+'.npy', allow_pickle = True).item()

# get last key of dict_results
ref_parameter = list(dict_results.keys())[-1]

r_value, p_value, mean_lizzi_feleppa_value, std_lizzi_feleppa_value = definitions.plot_one_lizzi_feleppa_results(dict_results, lizzi_feleppa_parameter, ref_parameter, transducer, Phantom, safe_to_notion, database_id, version, notion_cropping_properties, notion_window_func, notion_transducer)
#%%


# plot all regressions into one big figure
result_directory = '/Users/jakobschaefer/Documents/RF_US_AI/MLP_QUS_Parameter/Lizzi_Feleppa_Results'

definitions.plot_all_lizzi_feleppa_results(result_directory, ref_parameter, transducer, Phantom, version)



#%%
# evaluate outliers
lizzi_feleppa_parameter = 'MBF' # 'MBF', 'intercept', 'slope'

# load dict_results
result_directory = '/Users/jakobschaefer/Documents/RF_US_AI/MLP_QUS_Parameter/Lizzi_Feleppa_Results'
dict_results_C3 = np.load(result_directory+'/Lizzi_Feleppa_Results_'+version+'_'+'C3_large'+'.npy', allow_pickle = True).item()
dict_results_L15 = np.load(result_directory+'/Lizzi_Feleppa_Results_'+version+'_'+'L15_large'+'.npy', allow_pickle = True).item()

array_C3 = dict_results_C3[lizzi_feleppa_parameter]
array_L15 = dict_results_L15[lizzi_feleppa_parameter]

removal_based_on = 'IQR' # 'IQR' or 'std'
threshold = 'higher' # 'lower' -> IQR-quartiles at 25% and 75% and 2*std, 'higher' -> IQR-quartiles at 5% and 95% and 3*std

evaluation_parameters_no_outliers_C3, redcap_value_outlier_removal_C3, n_outliers_C3, id_outliers_C3 = definitions.detect_and_remove_outliers(dict_results_C3['Patient_ID'], array_C3,dict_results_C3[ref_parameter], removal_based_on, threshold)
evaluation_parameters_no_outliers_L15, redcap_value_outlier_removal_L15, n_outliers_L15, id_outliers_L15 = definitions.detect_and_remove_outliers(dict_results_L15['Patient_ID'], array_L15,dict_results_L15[ref_parameter], removal_based_on, threshold)


print('Outliers C3: ', n_outliers_C3, '(IDs: ', id_outliers_C3, ')')
print('Outliers L15: ', n_outliers_L15, '(IDs: ', id_outliers_L15, ')')

# get values of outliers
outlier_values_C3 = []
for outlier in id_outliers_C3:
    outlier_values_C3.append(array_C3[dict_results_C3['Patient_ID'].index(outlier)])
outlier_values_L15 = []
for outlier in id_outliers_L15:
    outlier_values_L15.append(array_L15[dict_results_L15['Patient_ID'].index(outlier)])



# plot boxplots for C3 and L15 frequency-specific Attenuation
# define specific whiskers
std_array_C3 = np.std(array_C3)
std_array_L15 = np.std(array_L15)

mean_array_C3 = np.mean(array_C3)
mean_array_L15 = np.mean(array_L15)

if removal_based_on == 'IQR':
    if threshold == 'higher':
        perzentiles_for_cutoff = [10,90]
    elif threshold == 'lower':
        perzentiles_for_cutoff = [25,75]
    P_C3 = np.percentile(array_C3, perzentiles_for_cutoff)
    IPR_C3 = np.abs(P_C3[1]-P_C3[0])
    P_L15 = np.percentile(array_L15, perzentiles_for_cutoff)
    IPR_L15 = np.abs(P_L15[1]-P_L15[0])

    whis_3std_C3 = np.interp([P_C3[0]-np.abs(1.5*IPR_C3), P_C3[1]+np.abs(1.5*IPR_C3)], np.sort(array_C3), np.linspace(0,1,len(array_C3))) * 100
    whis_3std_L15 = np.interp([P_L15[0]-np.abs(1.5*IPR_L15), P_L15[1]+1.5*IPR_L15], np.sort(array_L15), np.linspace(0,1,len(array_L15))) * 100
elif removal_based_on == 'std':
    if threshold == 'higher':
        nr_std = 3
    elif threshold == 'lower':
        nr_std = 2

    whis_3std_C3 = np.interp([mean_array_C3-nr_std*std_array_C3, mean_array_C3+nr_std*std_array_C3], np.sort(array_C3), np.linspace(0,1,len(array_C3))) * 100
    whis_3std_L15 = np.interp([mean_array_L15-nr_std*std_array_L15, mean_array_L15+nr_std*std_array_L15], np.sort(array_L15), np.linspace(0,1,len(array_L15))) * 100


fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# Set the random seed (for random distribution of scatterers)
seed_value = 42
np.random.seed(seed_value)
# boxplot C3
plt.sca(ax[0])
plt.boxplot(array_C3, whis = whis_3std_C3)

# check if some texts would overlap and if they do, then place them next to each other
n_overlaps = 0
for outlier in np.arange(len(outlier_values_C3)):
    position = 'normal'
    for outlier_value in outlier_values_C3:
        if outlier_value != outlier_values_C3[outlier]:
            if outlier_value > outlier_values_C3[outlier]-(max(np.abs(array_C3))/10) and outlier_value < outlier_values_C3[outlier]+(max(np.abs(array_C3))/10):
                position = 'overlapping'
    
    if position == 'overlapping':
        plt.text(1.03+n_overlaps*0.17,outlier_values_C3[outlier], id_outliers_C3[outlier], fontsize = 11) 
        n_overlaps += 1
    else:
        plt.text(1.03,outlier_values_C3[outlier], id_outliers_C3[outlier], fontsize = 11)

plt.scatter(np.random.normal(1,0.04,len(evaluation_parameters_no_outliers_C3)), evaluation_parameters_no_outliers_C3, color = '#00305D', alpha = 0.3)
plt.scatter(np.ones(len(outlier_values_C3)), outlier_values_C3, color = '#00305D', alpha = 0.3)
plt.title('C3 Lizzi-Feleppa '+lizzi_feleppa_parameter+' Regression', fontsize=12, fontweight='bold')
plt.ylabel('Lizzi-Feleppa '+lizzi_feleppa_parameter, fontsize=12)
plt.setp(plt.gca().lines, linewidth=2)
plt.xticks([])
#plt.xlabel('C3')

# boxplot L15
plt.sca(ax[1])
plt.boxplot(array_L15,  whis = whis_3std_L15)

# check if some texts would overlap and if they do, then place them next to each other
n_overlaps = 0
for outlier in np.arange(len(outlier_values_L15)):
    position = 'normal'
    for outlier_value in outlier_values_L15:
        if outlier_value != outlier_values_L15[outlier]: # exclude the outlier itself
            if outlier_value > outlier_values_L15[outlier]-(max(np.abs(array_L15))/10) and outlier_value < outlier_values_L15[outlier]+(max(np.abs(array_L15))/10): # check if outlier values are close to each other
                position = 'overlapping'

    if position == 'overlapping': 
        plt.text(1.03+n_overlaps*0.17,outlier_values_L15[outlier], id_outliers_L15[outlier], fontsize = 11) 
        n_overlaps += 1
    else:
        plt.text(1.03,outlier_values_L15[outlier], id_outliers_L15[outlier], fontsize = 11)

plt.scatter(np.random.normal(1,0.04,len(evaluation_parameters_no_outliers_L15)), evaluation_parameters_no_outliers_L15, color = '#00305D', alpha = 0.3)
plt.scatter(np.ones(len(outlier_values_L15)), outlier_values_L15, color = '#00305D', alpha = 0.3)
plt.title('L15 Lizzi-Feleppa '+lizzi_feleppa_parameter+' Regression', fontsize=12, fontweight='bold')
plt.ylabel('Lizzi-Feleppa '+lizzi_feleppa_parameter, fontsize=12)
plt.setp(plt.gca().lines, linewidth=2)
plt.xticks([])
#plt.xlabel('L15')

fig.tight_layout()
plt.show()

# linear regression of parameter and redcap values for C3
slope_C3, intercept_C3, r_value_C3, p_value_C3, mymodel_C3 = definitions.linear_regression(redcap_value_outlier_removal_C3, evaluation_parameters_no_outliers_C3)
print('C3 - no outliers: Pearson correlation coefficient '+ lizzi_feleppa_parameter+ ' vs. '+ref_parameter+': ', r_value_C3)
try: 
    if transducer == 'C3_large':
        print('Difference to C3 with outliers: ', np.abs(r_value_C3-r_value))
except:
    pass

# linear regression of parameter and redcap values for L15
slope_L15, intercept_L15, r_value_L15, p_value_L15, mymodel_L15 = definitions.linear_regression(redcap_value_outlier_removal_L15, evaluation_parameters_no_outliers_L15)
print('L15 - no outliers: Pearson correlation coefficient '+ lizzi_feleppa_parameter+ ' vs. '+ref_parameter+': ', r_value_L15)
try:
    if transducer == 'L15_large':
        print('Difference to L15 with outliers: ', np.abs(r_value_L15-r_value))
except:
    pass

# # safe results (with outliers removed) to notion
# if Phantom == 'Wat_Phan':
#     notion_evaluation_parameter = lizzi_feleppa_parameter+' Lizzi Feleppa ('+str(n_outliers)+' outliers removed) \nCalibration: '+Phantom
# else:
#     notion_evaluation_parameter = lizzi_feleppa_parameter+' Lizzi Feleppa ('+str(n_outliers)+' outliers removed) \nCalibration: '+Phantom
# notion_results = str('Pearson coef.: '+str(round(np.abs(r_value), 3))+ '\n(p-value: '+ str(round(p_value, 3))+')')
# notion_version = str(version+'\n('+str(len(redcap_value_outlier_removal))+' Pat.)')
# notion_def.safe_to_notion_def(notion_version, notion_evaluation_parameter, notion_results, notion_transducer, notion_cropping_properties,notion_window_func, safe_to_notion, database_id)

# %% calculate spearman coefficients
version = '230330'
ref_parameter = 'CAP'
lizzi_feleppa_parameter = 'slope'

# load dict_results
result_directory = '/Users/jakobschaefer/Documents/RF_US_AI/MLP_QUS_Parameter/Lizzi_Feleppa_Results'
dict_results_C3 = np.load(result_directory+'/Lizzi_Feleppa_Results_'+version+'_'+'C3_large'+'.npy', allow_pickle = True).item()
dict_results_L15 = np.load(result_directory+'/Lizzi_Feleppa_Results_'+version+'_'+'L15_large'+'.npy', allow_pickle = True).item()

array_C3 = dict_results_C3[lizzi_feleppa_parameter]
array_L15 = dict_results_L15[lizzi_feleppa_parameter]

# calculate spearman coefficients
spearman_C3= stats.spearmanr(array_C3, dict_results_C3[ref_parameter])
spearman_L15 = stats.spearmanr(array_L15, dict_results_L15[ref_parameter])

print('C3: ', spearman_C3)
print('L15: ', spearman_L15)


# %%
