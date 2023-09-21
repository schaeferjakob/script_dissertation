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

#%%
# choose recording to display 
version = '230904'
directory = '/Volumes/Extreme_SSD/Aufnahmen_vorverarbeitet/Finished_data_'+version #'/Volumes/Extreme_SSD/Aufnahmen_vorverarbeitet/Calibration'# '/Volumes/Extreme_SSD/Aufnahmen_vorverarbeitet/Finished_data_'+version

examination = 'UKD098'
recording = 'L15_large'

# phantom recordings
calibration_type = 'Wat_Phan' # 'Phantom' or 'Met_Phan' or 'AIR00000' or 'Wat_Phan' 
phantom_file = '/Volumes/Extreme_SSD/Aufnahmen_vorverarbeitet/Calibration/'+calibration_type+'/'+recording[:9]+'.rf_no_tgc.npy'

# define window function

window_function = 'hanning'

# define where to choose ROI from
ROI_type = 2 # 0: whole image, 1: middle of scanline (=around focal point), 2: specific ROI (defined below)
len_ROI = 500 # the length of the examined ROI in px (only for ROI_type = 1)
specific_ROI = [1000,1500] # [start, end] of ROI

# define border of the calibration spectrum
border = 15 # dB

#%% display dicom, rf and rf_no_tgc data in one figure
# display DICOM image
display_one_scanline = False
file = directory+'/'+examination+'/'+recording+'.dcm.npy'

# load data
dicom_data = np.load(file, allow_pickle=True)

# extract the part with image information

dicom_data = definitions.crop_dcm(dicom_data, recording)

# display one DICOM scanline
if display_one_scanline == True:
    scanline = len(dicom_data[:,0])//2
    plt.figure(figsize=(5,5)) # figsize in inches (width, height) -> multiply with  0.393701 to get cm
    plt.plot(dicom_data[:,scanline], color = '#00305D')
    #plt.title('Scanline '+str(scanline)+' of '+recording+' '+transducer)
    plt.xlabel('Depth [px]')
    plt.ylabel('Grey value')
    plt.show()

# load RF data
rf_file = os.path.join(directory, examination, recording+'.rf.npy')
rf_data = np.load(rf_file)
rf_data = rf_data[:,:,0]

rf_picture = 20 * np.log10(np.abs(1 + hilbert(rf_data)))
rf_picture = np.rot90(rf_picture, 3)

# load RF_no_tgc data
rf_no_tgc_file = os.path.join(directory, examination, recording+'.rf_no_tgc.npy')
rf_no_tgc_data = np.load(rf_no_tgc_file)
rf_no_tgc_data = rf_no_tgc_data[:,:,0]

rf_no_tgc_picture = 20 * np.log10(np.abs(1 + hilbert(rf_no_tgc_data)))
rf_no_tgc_picture = np.rot90(rf_no_tgc_picture, 3)

# display dicom, rf and rf_no_tgc data in one figure with 3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))
ax1.imshow(dicom_data, cmap='gray')
ax1.set_title('DICOM image '+examination+' '+recording)
extent = [0, len(rf_no_tgc_picture[0,:])*10, 0, len(rf_no_tgc_picture[:,0])]
ax2.imshow(rf_picture,extent= extent, cmap='gray')
ax2.set_title('RF image '+examination+' '+recording)
extent = [0, len(rf_no_tgc_picture[0,:])*10, 0, len(rf_no_tgc_picture[:,0])]
ax3.imshow(rf_no_tgc_picture, extent=extent,cmap='gray')
ax3.set_title('RF_no_tgc image '+examination+' '+recording)
plt.show()

# just display the DICOM image
plt.figure(figsize=(5,5)) # figsize in inches (width, height) -> multiply with  0.393701 to get cm
plt.imshow(dicom_data, cmap='gray')
plt.title('DICOM image '+examination+' '+recording)
plt.show()


#%%
# display rf data
hilbert_envelope = False
show_rf_picture = True

display_scanline_segment = True
start_pixel = 500
end_pixel = 1000

file = directory+'/'+examination+'/'+recording+'.rf_no_tgc.npy'

# load data
data = np.load(file)

# reduce data to first frame
data = data[:,:,0]

# display one scanline
scanline = len(data[:,0])//2

plt.figure(figsize=(5,5)) # figsize in inches (width, height) -> multiply with  0.393701 to get cm
plt.plot(data[scanline,:])#, color = '#00305D')
#plt.title('Scanline '+str(scanline)+' of '+recording+' '+transducer)
plt.xlabel('Depth [px]')
plt.ylabel('Amplitude')
#plt.show()

# display hilbert envelope of rf data
if hilbert_envelope == True:
    envelope = hilbert(data)
    abs_envelope = np.abs(envelope)

    # display one scanline
    scanline = 150
    #plt.figure(figsize=(5,5)) # figsize in inches (width, height) -> multiply with  0.393701 to get cm
    plt.plot(abs_envelope[scanline,:])#, color = '#00305D')
    #plt.title('Scanline '+str(scanline)+' of '+recording+' '+transducer)
    plt.xlabel('Depth [px]')
    plt.ylabel('Amplitude')
    plt.show()

# display one scanline segment
if display_scanline_segment == True:
    time_segment = data[scanline, start_pixel:end_pixel]

    plt.figure(figsize=(5,5)) # figsize in inches (width, height) -> multiply with  0.393701 to get cm
    plt.plot(np.arange(start_pixel,end_pixel),time_segment)#, color = '#00305D')
    plt.title('Scanline '+str(scanline)+' from '+str(start_pixel)+'px to '+str(end_pixel)+'px of '+examination+' '+ recording) 
    plt.xlabel('Depth [px]')
    plt.ylabel('Amplitude')
    plt.show()

#%%
# display fft 
file = directory+'/'+examination+'/'+recording+'.rf_no_tgc.npy'
data = np.load(file)
data = data[:,:,0]

# define sampling frequency
if 'C3' in recording:
    fs = 15e6
if 'L15' in recording:
    fs = 30e6

# cut out specific ROI from data
if ROI_type == 1:
    data = data[:,len(data[0,:])//2-(len_ROI//2):len(data[0,:])//2+(len_ROI//2)]
if ROI_type == 2:
    data = data[:,specific_ROI[0]:specific_ROI[1]]

# calculate fft of recording
y_fft = []
for scanline in range(len(data[:,0])):
    x_fft, y_fft_scanline = definitions.amplitude_spectrum(fs, data[scanline,:],window_function)
    y_fft.append(y_fft_scanline)
y_fft = np.array(y_fft)
y_fft_square = np.square(y_fft)
average_power_spectrum = np.mean(y_fft_square, axis = 0)

# plot average power spectrum
plt.figure(figsize=(5,5)) # figsize in inches (width, height) -> multiply with  0.393701 to get cm
plt.plot(x_fft, average_power_spectrum)
plt.title('Average power spectrum of '+examination+' '+recording)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude')
plt.show()


#%% 
# # frequency downshift phantom block

# file = '/Volumes/Extreme_SSD/Aufnahmen_vorverarbeitet/Phantom/'+transducer+'.rf_no_tgc.npy'
# phantom_data = np.load(file)

# segment_width = 500 # in samples
# overlap = 0 # in percent

# segments = len(phantom_data[0,:])//(segment_width-segment_width*overlap/100)

# for segment in range(segments):
#     segment_phantom_data = 

# phantom_data = phantom_data[:,:,0]
# #phantom_data = phantom_data[:,len(phantom_data[0,:])//2-250:len(phantom_data[0,:])//2+250]
# #phantom_data = phantom_data[:,500:1000]

# phantom_y_fft = []
# for scanline in range(len(phantom_data[:,0])):
#     phantom_x_fft, phantom_y_fft_scanline = definitions.amplitude_spectrum(fs, phantom_data[scanline,:], hamming=True)
#     phantom_y_fft.append(phantom_y_fft_scanline[5:])
# phantom_y_fft = np.array(phantom_y_fft)
# phantom_y_fft_square = np.square(phantom_y_fft)
# calibration_spectrum = np.mean(phantom_y_fft_square, axis = 0)

# %%
# display stFFT data

# load data
#file = '/Volumes/Extreme_SSD/Aufnahmen_vorverarbeitet/Phantom/'+transducer+'.rf_no_tgc.npy'
file = directory+'/'+examination+'/'+recording+'.rf_no_tgc.npy'
file = phantom_file
data = np.load(file)


# define window size and hop size
window_size = 200
hop_size = 1

# perform stFFT
stfft = definitions.stFFT(data, window_size, hop_size, fs)
stfft_square = np.square(stfft)
stfft_mean = np.mean(stfft_square, axis = 0)

for depth in range(len(stfft_mean[:,0])):
    plt.figure()
    plt.plot(stfft_mean[depth,:])
    plt.xlabel('Frequency section')
    plt.ylabel('Amplitude')
    plt.title('Depth '+str(depth)+'/'+str(len(stfft_mean[:,0])))
    plt.show()



# # display one scanline
# scanline = 150
# stfft = stfft[scanline,:,:]
# stfft = stfft.T
# plt.figure(figsize=(5,5)) # figsize in inches (width, height) -> multiply with  0.393701 to get cm
# plt.imshow(stfft)
# #plt.title('Scanline '+str(scanline)+' of '+recording+' '+transducer)
# plt.xlabel('Depth')
# plt.ylabel('Frequency section')
# plt.show()

# %%
# plot log of one frequency line + regression line
frequency_section = 23

frequency_line = stfft[frequency_section,:]

log_frequency_line = np.log10(frequency_line)

# perform linear regression
x = np.arange(0,len(log_frequency_line))
y = log_frequency_line
m, b = np.polyfit(x, y, 1)


# plot frequency line
plt.figure(figsize=(5,5)) # figsize in inches (width, height) -> multiply with  0.393701 to get cm
plt.plot(log_frequency_line, color = '#00305D', label = 'Log. frequency line ')
#plt.title('Scanline '+str(scanline)+' of '+recording+' '+transducer)

# plot regression line
plt.plot(x, m*x + b, color = '#FF0000', label = 'Linear fit (Slope = '+str(round(m,4))+')')

plt.xlabel('Depth')
plt.ylabel('Amplitude')
plt.legend()
plt.show()

# %%