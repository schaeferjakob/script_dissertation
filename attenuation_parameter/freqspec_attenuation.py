#%% 
# evaluate if the attenuation slope is dependent on the frequency section and the CAP value (-> fat content)
#%%
import sys
sys.path.insert(0, '/Users/jakobschaefer/Documents/rf-ultrasound')  # necessary to import functions from other directory
import functions.definitions as definitions
import numpy as np
import matplotlib.pyplot as plt
#%%
# load values
version = '230330'
recording_nr = 0

file_path_C3 = '/Users/jakobschaefer/Documents/RF_US_AI/MLP_QUS_Parameter/Input_data/mean_slopes_all_recordings_freqspec_C3_'+version+'.npy'
file_path_L15 = '/Users/jakobschaefer/Documents/RF_US_AI/MLP_QUS_Parameter/Input_data/mean_slopes_all_recordings_freqspec_L15_'+version+'.npy'

mean_freqspec_slopes_C3 = np.load(file_path_C3)
mean_freqspec_slopes_L15 = np.load(file_path_L15)

attenuation_freqspec_C3 = np.abs(mean_freqspec_slopes_C3)
attenuation_freqspec_L15 = np.abs(mean_freqspec_slopes_L15)

mean_attenuation_freqspec_C3 = np.mean(attenuation_freqspec_C3, axis = 0)
mean_attenuation_freqspec_L15 = np.mean(attenuation_freqspec_L15, axis = 0)

x_L15 = np.arange(0,15,15/len(attenuation_freqspec_L15[recording_nr,:]))
x_C3 = np.arange(0,7.5,7.5/len(attenuation_freqspec_C3[recording_nr,:]))

# plot mean slopes for each frequency section of one recording for C3 and L15 in one array
plt.figure(figsize=(10,5))
plt.scatter(x_C3, attenuation_freqspec_C3[recording_nr,:], color = '#00305D', label = 'Frequency specific attenuation C3')
plt.scatter(x_L15, attenuation_freqspec_L15[recording_nr,:], color = 'red', label = 'Frequency specific attenuation L15')
plt.xlabel('Frequency [MHz]', fontsize = 12)
plt.ylabel('Attenuation slope', fontsize = 12)
plt.title('Frequency specific attenuation of recording '+str(recording_nr), fontweight = 'bold')
plt.legend()
plt.show()

# plot mean slopes for each frequency section of all recordings for C3 and L15 in one array
plt.figure(figsize=(10,5))
plt.scatter(x_C3, mean_attenuation_freqspec_C3, color = '#00305D', label = 'Frequency specific attenuation C3')
plt.scatter(x_L15, mean_attenuation_freqspec_L15, color = 'red', label = 'Frequency specific attenuation L15')
plt.xlabel('Frequency [MHz]', fontsize = 12)
plt.ylabel('Attenuation slope', fontsize = 12)
plt.title('Mean frequency specific attenuation of all recordings', fontweight = 'bold')
plt.legend()
plt.show()
#%% 
# display the mean_fft_all_recordings for C3 and L15 in one array 
version = '230330'
#load C3
transducer = 'C3_large'
ROI_C3 = [505,2184] # start and end
mean_fft_all_recordings_C3, x_fft_C3 = np.load('/Volumes/Extreme_SSD/Aufnahmen_vorverarbeitet/Calibration/mean_fft_all_recordings/mean_fft_all_recordings_'+str(ROI_C3[0])+'-'+str(ROI_C3[1])+'_'+version+'_'+transducer+'.npy', allow_pickle=True)

#load L15
transducer = 'L15_large'
ROI_L15 =  [640,2546] # start and end
mean_fft_all_recordings_L15, x_fft_L15 = np.load('/Volumes/Extreme_SSD/Aufnahmen_vorverarbeitet/Calibration/mean_fft_all_recordings/mean_fft_all_recordings_'+str(ROI_L15[0])+'-'+str(ROI_L15[1])+'_'+version+'_'+transducer+'.npy', allow_pickle=True)

# plot mean fft all recordings for C3 and L15 in one array with different y-axis
fig, ax1 = plt.subplots(figsize=(10,5))
ax1.plot(x_fft_C3, mean_fft_all_recordings_C3, color = '#00305D', label = 'C3')
ax1.set_xlabel('Frequency [Hz]', fontsize = 12)
ax1.set_ylabel('Amplitude C3', color = '#00305D', fontsize = 12)
ax1.tick_params(axis='y', labelcolor = '#00305D')
ax1.set_title('Mean FFT of all recordings', fontweight = 'bold')
ax2 = ax1.twinx()
ax2.plot(x_fft_L15, mean_fft_all_recordings_L15, color = 'red', label = 'L15')
ax2.set_ylabel('Amplitude L15', color = 'red', fontsize = 12)
ax2.tick_params(axis='y', labelcolor = 'red')
# add legend
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)

plt.show()
# %%
# display the mean slopes and the mean fft in one plot (C3 and L15 in different arrays)
version = '230330'

#display mean_fft_all_recordings_C3 and attenuation_freqspec_C3 in one plot with different y-axis
fig, ax1 = plt.subplots(figsize=(5,5))
ax1.plot(x_fft_C3*1e-6, mean_fft_all_recordings_C3, color = '#00305D', label = 'Mean fft C3')
ax1.set_xlabel('Frequency [MHz]', fontsize = 12)
ax1.set_ylabel('Amplitude C3', color = '#00305D', fontsize = 12)
ax1.tick_params(axis='y', labelcolor = '#00305D')
ax1.set_title('FFT all recordings and freq.spec. Attenuation C3', fontweight = 'bold')
ax2 = ax1.twinx()
ax2.scatter(x_C3, mean_attenuation_freqspec_C3, color = 'red', label = 'Frequency specific attenuation C3')
ax2.set_ylabel('Attenuation slope C3', color = 'red', fontsize = 12)
ax2.tick_params(axis='y', labelcolor = 'red')
# # vertical line at 5.2 MHz
# ax1.axvline(x=5.2, color = 'black', linestyle = '--')
# add legend
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
plt.show()

# display mean_fft_all_recordings_L15 and attenuation_freqspec_L15 in one plot with different y-axis
fig, ax1 = plt.subplots(figsize=(5,5))
ax1.plot(x_fft_L15*1e-6, mean_fft_all_recordings_L15, color = '#00305D', label = 'Mean fft L15')
ax1.set_xlabel('Frequency [MHz]', fontsize = 12)
ax1.set_ylabel('Amplitude L15', color = '#00305D', fontsize = 12)
ax1.tick_params(axis='y', labelcolor = '#00305D')
ax1.set_title('FFT all recordings and freq.spec. Attenuation L15', fontweight = 'bold')
ax2 = ax1.twinx()
ax2.scatter(x_L15, mean_attenuation_freqspec_L15, color = 'red', label = 'Frequency specific attenuation L15')
ax2.set_ylabel('Attenuation slope L15', color = 'red', fontsize = 12)
ax2.tick_params(axis='y', labelcolor = 'red')
# # vertical line at 5.2 MHz
# ax1.axvline(x=5.2, color = 'black', linestyle = '--')
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
plt.show()


# %%
# plot the mean slopes for different levels of CAP (C3 and L15 in different arrays)
# define CAP levels
lower_border = 248
upper_border = 280

version = '230330'
# load values C3
file_path_C3 = '/Users/jakobschaefer/Documents/RF_US_AI/MLP_QUS_Parameter/Input_data/mean_slopes_all_recordings_freqspec_C3_'+version+'.npy'
cap_path_C3 = '/Users/jakobschaefer/Documents/RF_US_AI/MLP_QUS_Parameter/Targets/targets_CAP_freqspec_C3_'+version+'.npy'
mean_freqspec_slopes_C3 = np.load(file_path_C3)
cap_values_C3 = np.load(cap_path_C3, allow_pickle=True).item()
cap_values_C3 = list(cap_values_C3.values())

low_CAP_C3 = []
medium_CAP_C3 = []
high_CAP_C3 = []
for i in range(len(cap_values_C3)):
    if cap_values_C3[i] < lower_border:
        low_CAP_C3.append(mean_freqspec_slopes_C3[i,:])
    elif cap_values_C3[i] > upper_border:
        high_CAP_C3.append(mean_freqspec_slopes_C3[i,:])
    else:
        medium_CAP_C3.append(mean_freqspec_slopes_C3[i,:])

low_CAP_C3 = np.array(low_CAP_C3)
medium_CAP_C3 = np.array(medium_CAP_C3)
high_CAP_C3 = np.array(high_CAP_C3)

mean_low_CAP_C3 = np.abs(np.mean(low_CAP_C3, axis = 0))
mean_medium_CAP_C3 = np.abs(np.mean(medium_CAP_C3, axis = 0))
mean_high_CAP_C3 = np.abs(np.mean(high_CAP_C3, axis = 0))

# load values L15
file_path_L15 = '/Users/jakobschaefer/Documents/RF_US_AI/MLP_QUS_Parameter/Input_data/mean_slopes_all_recordings_freqspec_L15_'+version+'.npy'
cap_path_L15 = '/Users/jakobschaefer/Documents/RF_US_AI/MLP_QUS_Parameter/Targets/targets_CAP_freqspec_L15_'+version+'.npy'
mean_freqspec_slopes_L15 = np.load(file_path_L15)
cap_values_L15 = np.load(cap_path_L15, allow_pickle=True).item()
cap_values_L15 = list(cap_values_L15.values())

low_CAP_L15 = []
medium_CAP_L15 = []
high_CAP_L15 = []
for i in range(len(cap_values_L15)):
    if cap_values_L15[i] < lower_border:
        low_CAP_L15.append(mean_freqspec_slopes_L15[i,:])
    elif cap_values_L15[i] > upper_border:
        high_CAP_L15.append(mean_freqspec_slopes_L15[i,:])
    else:
        medium_CAP_L15.append(mean_freqspec_slopes_L15[i,:])

low_CAP_L15 = np.array(low_CAP_L15)
medium_CAP_L15 = np.array(medium_CAP_L15)
high_CAP_L15 = np.array(high_CAP_L15)

mean_low_CAP_L15 = np.abs(np.mean(low_CAP_L15, axis = 0))
mean_medium_CAP_L15 = np.abs(np.mean(medium_CAP_L15, axis = 0))
mean_high_CAP_L15 = np.abs(np.mean(high_CAP_L15, axis = 0))


# plot mean slopes for different levels of CAP for C3 and L15 in two subplots 
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))
fig.suptitle('Mean frequency specific attenuation at different CAP levels', fontweight = 'bold', fontsize = 14)
ax1.plot(x_C3, mean_low_CAP_C3, color = 'green', label = 'Low CAP')
ax1.plot(x_C3, mean_medium_CAP_C3, color = '#00305D', label = 'Medium CAP')
ax1.plot(x_C3, mean_high_CAP_C3, color = 'red', label = 'High CAP')
ax1.set_xlabel('Frequency [MHz]', fontsize = 12)
ax1.set_ylabel('Attenuation slope C3', fontsize = 12)
ax1.set_title('C3', fontweight = 'bold')
ax1.legend()
ax1.grid()
ax2.plot(x_L15, mean_low_CAP_L15, color = 'green', label = 'Low CAP')
ax2.plot(x_L15, mean_medium_CAP_L15, color = '#00305D', label = 'Medium CAP')
ax2.plot(x_L15, mean_high_CAP_L15, color = 'red', label = 'High CAP')
ax2.set_xlabel('Frequency [MHz]', fontsize = 12)
ax2.set_ylabel('Attenuation slope L15', fontsize = 12)
ax2.set_title('L15', fontweight = 'bold')
ax2.legend(loc = 'upper right')
ax2.grid()
plt.tight_layout()
plt.show()



# defnie difference of low and high CAP Attenuation from medium CAP Attenuation
#C3
diff_low_CAP_C3 = mean_low_CAP_C3 - mean_medium_CAP_C3
diff_high_CAP_C3 = mean_high_CAP_C3 - mean_medium_CAP_C3

diff_low_CAP_C3 = np.abs(diff_low_CAP_C3)
diff_high_CAP_C3 = np.abs(diff_high_CAP_C3)

#L15
diff_low_CAP_L15 = mean_low_CAP_L15 - mean_medium_CAP_L15
diff_high_CAP_L15 = mean_high_CAP_L15 - mean_medium_CAP_L15

diff_low_CAP_L15 = np.abs(diff_low_CAP_L15)
diff_high_CAP_L15 = np.abs(diff_high_CAP_L15)


# plot difference of low and high CAP Attenuation from medium CAP Attenuation for C3 and L15 in two subplots
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))
fig.suptitle('Difference of low and high CAP attenuation from medium CAP attenuation', fontweight = 'bold', fontsize = 14)
ax1.plot(x_C3, diff_low_CAP_C3, color = 'green', label = 'Low CAP')
# horizontal line at 0 for medium CAP attenuation
ax1.axhline(y=0, color = '#00305D', label = 'Medium CAP')
ax1.plot(x_C3, diff_high_CAP_C3, color = 'red', label = 'High CAP')
ax1.set_xlabel('Frequency [MHz]', fontsize = 12)
ax1.set_ylabel('Attenuation slope C3', fontsize = 12)
ax1.set_title('C3', fontweight = 'bold')
ax1.legend()
ax1.grid()
ax2.plot(x_L15, diff_low_CAP_L15, color = 'green', label = 'Low CAP')
# horizontal line at 0 for medium CAP attenuation
ax2.axhline(y=0, color = '#00305D', label = 'Medium CAP')
ax2.plot(x_L15, diff_high_CAP_L15, color = 'red', label = 'High CAP')
ax2.set_xlabel('Frequency [MHz]', fontsize = 12)
ax2.set_ylabel('Attenuation slope L15', fontsize = 12)
ax2.set_title('L15', fontweight = 'bold')
ax2.legend(loc = 'upper right')
ax2.grid()
plt.tight_layout()
plt.show()

# %% plot the frequency specific Pearson coefficients together with the mean_fft_all_recordings for C3 and L15 
version = '230330'
#load data
file_path_C3 = '/Users/jakobschaefer/Documents/RF_US_AI/MLP_QUS_Parameter/Input_data/mean_slopes_all_recordings_freqspec_C3_'+version+'.npy'
file_path_L15 = '/Users/jakobschaefer/Documents/RF_US_AI/MLP_QUS_Parameter/Input_data/mean_slopes_all_recordings_freqspec_L15_'+version+'.npy'

mean_freqspec_slopes_C3 = np.load(file_path_C3)
mean_freqspec_slopes_L15 = np.load(file_path_L15)

# %%
