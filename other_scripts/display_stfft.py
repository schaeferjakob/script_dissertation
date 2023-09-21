#%%
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('../') # necessary to import functions from other directory
import functions.definitions as definitions
from scipy import stats
from scipy.signal import hilbert
from scipy import signal
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

#%%
# choose recording to display 
version = '230612'
directory = '/Volumes/Extreme_SSD/Aufnahmen_vorverarbeitet/Finished_data_'+version

examination = 'UKD227'
recording = 'L15_large'

# start and end of time segment to display
rf_start_pixel = 1000
rf_end_pixel = 2000

# stfft parameters
window_size = 50
hop_size = 25 # hopsize/window_size = overlap
if recording.startswith('L15'):
    sampling_frequency = 30e6
elif recording.startswith('C3'):
    sampling_frequency = 15e6
window_function = 'hamming' # 'hamming', 'hanning', 'blackman' or 'none'
#%%
# display DICOM image and mark time segment of one scanline
display_one_dcm_scanline = False
mark_time_segment = True

# load dcm data
dcm_file = directory+'/'+examination+'/'+recording+'.dcm.npy'
dcm_data = np.load(dcm_file, allow_pickle=True)
# extract the part with image information
dcm_data = definitions.crop_dcm(dcm_data, recording)
# choose one scanline
dcm_scanline = len(dcm_data[0,:])//2
# len in px of one dcm-scanline
dcm_scanline_length = len(dcm_data[:,0])

# load rd data
rf_file = directory+'/'+examination+'/'+recording+'.rf_no_tgc.npy'
rf_data = np.load(rf_file) 
# reduce data to first frame
rf_data = rf_data[:,:,0]
# choose one scanline
rf_scanline = len(rf_data[:,0])//2
# len in px of one rf-scanline
rf_scanline_length = len(rf_data[0,:])


# display dcm_data
plt.figure(figsize = (7.5,4))
plt.imshow(dcm_data.T, cmap='gray')
plt.title('DICOM image '+examination+' '+recording)
plt.xlabel('Depth [px]', fontsize= 12)
plt.ylabel('Lateral [px]', fontsize= 12)

# display time-segment of one scanline
if mark_time_segment == True:
    dcm_start_pixel = int(rf_start_pixel*dcm_scanline_length/rf_scanline_length)
    dcm_end_pixel = int(rf_end_pixel*dcm_scanline_length/rf_scanline_length)
    plt.plot([dcm_start_pixel, dcm_end_pixel], [dcm_scanline, dcm_scanline], linewidth=2)
    plt.text(dcm_start_pixel+15,dcm_scanline-15,'A-Scanline', c='#1f77b4', fontsize = 12)

    # display rectangle of imaginary ROI
    plt.gca().add_patch(Rectangle((dcm_start_pixel, 100), (dcm_end_pixel-dcm_start_pixel), dcm_scanline-55, edgecolor = 'green', facecolor = 'none', lw = 2))
    plt.text(dcm_start_pixel+15,dcm_scanline-65, 'Example ROI', color= 'green', fontsize = 12)

# display one scanline
if display_one_dcm_scanline == True:
    scanline = len(dcm_data[:,0])//2
    plt.figure(figsize=(5,5)) # figsize in inches (width, height) -> multiply with  0.393701 to get cm
    plt.plot(dcm_data[:,scanline], color = '#00305D')
    #plt.title('Scanline '+str(scanline)+' of '+recording+' '+transducer)
    plt.xlabel('Depth [px]')
    plt.ylabel('Grey value')
    plt.show()

# %%
# display one rf-scanline and mark one time segment
hilbert_envelope = False
display_scanline_segment = True


plt.figure(figsize=(5,5)) # figsize in inches (width, height) -> multiply with  0.393701 to get cm
plt.plot(rf_data[rf_scanline,:])#, color = '#00305D')

if mark_time_segment == True:
    #markers_on = [rf_start_pixel, rf_end_pixel]
    plt.plot(rf_start_pixel,0, color = '#FF0000', marker='o', markersize=5)
    plt.plot(rf_end_pixel,0, color = '#FF0000', marker='o', markersize=5)
#plt.title('Scanline '+str(scanline)+' of '+recording+' '+transducer)
plt.xlabel('Depth [px]')
plt.ylabel('Amplitude')
#plt.show()

# display hilbert envelope of rf data
if hilbert_envelope == True:
    envelope = hilbert(rf_data)
    abs_envelope = np.abs(envelope)

    # display one scanline
    #plt.figure(figsize=(5,5)) # figsize in inches (width, height) -> multiply with  0.393701 to get cm
    plt.plot(abs_envelope[rf_scanline,:])#, color = '#00305D')
    #plt.title('Scanline '+str(scanline)+' of '+recording+' '+transducer)
    plt.xlabel('Depth [px]')
    plt.ylabel('Amplitude')
    plt.show()

# display one scanline segment
if display_scanline_segment == True:
    time_segment = rf_data[rf_scanline, rf_start_pixel:rf_end_pixel]

    plt.figure(figsize=(5.3,4)) # figsize in inches (width, height) -> multiply with  0.393701 to get cm
    plt.plot(np.arange(rf_start_pixel,rf_end_pixel),time_segment)#, color = '#00305D')
    plt.title('Scanline '+str(rf_scanline)+' from '+str(rf_start_pixel)+'px to '+str(rf_end_pixel)+'px of '+examination+' '+ recording) 
    plt.xlabel('Depth [px]', fontsize= 12)
    plt.ylabel('Amplitude', fontsize= 12)
    plt.show()
# %%
# display stfft of scanline segment
num_xticks = 5
num_yticks = 6

stfft_scanline_segment = []
for i in range(0,round(len(time_segment)/hop_size)):
    segment_array = time_segment[i*hop_size:i*hop_size+window_size]
    if len(segment_array) == window_size:
        x_fft, y_fft_plot = definitions.amplitude_spectrum(sampling_frequency, segment_array, window_function)
    
        y_fft_line = y_fft_plot.reshape(len(y_fft_plot),1)
        
        stfft_scanline_segment.append(y_fft_line)
    
stfft_scanline_segment = np.array(stfft_scanline_segment)
stfft_scanline_segment = np.squeeze(stfft_scanline_segment)

plt.figure(figsize=(6,4.5)) # figsize in inches (width, height) -> multiply with  0.393701 to get cm
plt.imshow(np.abs(stfft_scanline_segment).T)
plt.title('STFFT of scanline '+str(rf_scanline)+' from '+str(rf_start_pixel)+'px to '+str(rf_end_pixel)+'px of '+examination+' '+ recording)
plt.xlabel('Depth [px]', fontsize = 12)
plt.xticks(np.arange(stfft_scanline_segment.shape[0], step =stfft_scanline_segment.shape[0]/num_xticks ),np.arange(rf_start_pixel,rf_end_pixel, step = (rf_end_pixel-rf_start_pixel)/num_xticks))
plt.ylabel('Frequency [MHz]', fontsize = 12)
plt.yticks(np.arange(stfft_scanline_segment.shape[1], step =stfft_scanline_segment.shape[1]/num_yticks ),np.arange(0,(sampling_frequency/2)*1e-6, step = (sampling_frequency/2/num_yticks)*1e-6))
plt.show()


# %%
