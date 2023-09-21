#%%
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, '/Users/jakobschaefer/Documents/rf-ultrasound')  # necessary to import functions from other d irectory
import functions.definitions as definitions
from tqdm import tqdm

#%%
# choose recording to display 
version = '230330'
start_directory = '/Volumes/Extreme_SSD/Aufnahmen_vorverarbeitet/C3_and_L15_without_TGC_'+version
transducer = 'C3_large'
window_function = 'hamming'

# define ROI for FFT
# attenuation C3: [505,2184]
# attenuation L15: [640,2546]
ROI = [505,2184] # start and end 
#%%
# calculate FFT for all recordings
fft_all_recordings = []
nr_removed_recordings = 0
for recording in tqdm(os.listdir(start_directory)):
    # load file
    array = np.load(start_directory+'/'+recording+'/'+transducer+'.rf_no_tgc.npy')

    # sort out recordings where array is shorter than a certain length
    if len(array[0,:,0]) >= ROI[1]:
       
        array = array[:,ROI[0]:ROI[1],:]

        # get sampling rate out of yaml file
        yaml_file = str(start_directory)+'/'+recording+'/'+transducer+'.rf.yaml'
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
    else:
        nr_removed_recordings += 1
fft_all_recordings = np.array(fft_all_recordings)

fft_all_recordings = np.square(fft_all_recordings)
mean_fft_all_recordings = np.mean(fft_all_recordings, axis = 0)

print('Number of recordings which were removed because they were too short: '+str(nr_removed_recordings))

# plot mean fft all recordings
plt.figure()
plt.plot(x_fft, mean_fft_all_recordings)
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.title('Mean FFT of all recordings')
plt.show()
# %%
np.save('/Volumes/Extreme_SSD/Aufnahmen_vorverarbeitet/Calibration/mean_fft_all_recordings/mean_fft_all_recordings_'+str(ROI[0])+'-'+str(ROI[1])+'_'+version+'_'+transducer+'.npy', (mean_fft_all_recordings, x_fft))

# %%
# display the mean_fft_all_recordings for C3 and L15 in one array 
version = '230330'
#load C3
transducer = 'C3_large'
ROI_C3 = [505,2184]#[0,2000] # start and end
mean_fft_all_recordings_C3, x_fft_C3 = np.load('/Volumes/Extreme_SSD/Aufnahmen_vorverarbeitet/Calibration/mean_fft_all_recordings/mean_fft_all_recordings_'+str(ROI_C3[0])+'-'+str(ROI_C3[1])+'_'+version+'_'+transducer+'.npy', allow_pickle=True)

#load L15
transducer = 'L15_large'
ROI_L15 =  [640,2546]#[0,2704] # start and end
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

#%%
