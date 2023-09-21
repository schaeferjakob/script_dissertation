import csv
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft
import sys
sys.path.insert(0, '/Users/jakobschaefer/Documents/rf-ultrasound')  # necessary to import functions from other directory
import functions.definitions as definitions
import sys

def get_a_scan(directory, sampling_rate):
    #search rf-file in chosen directory
    for root, dirs, files in os.walk(directory):  
            for name in files:
                if name.endswith('rf.raw'):
                    name_length = len(name)
                    file_path_1 = os.path.abspath(name)
                    file_path_2 = file_path_1[:-name_length]+ root +'/'+file_path_1[-name_length:]
    rf_filename = file_path_2

    # get sampling_rate and soundspeed out of rf.yml file (sometimes soundspeed is not provided)
    soundspeed = 154e3 #in cm/s geschätzt

    hdr, timestamps, data = definitions.read_rf(rf_filename) # read rf-file 
    middle_line = len(data[:,0,0])/2
    a_scan = data[int(middle_line)]
    a_scan = a_scan[:,0]
    a_scan = a_scan.tolist()

    time_per_pixel = 1/sampling_rate
    time_axis = np.linspace(0, len(a_scan)*time_per_pixel*soundspeed, len(a_scan))

    #plt.plot(time_axis, a_scan)
    #plt.xlabel('Depth ($cm$)')
    #plt.ylabel('Amplitude ($Unit$)')
    return a_scan


# c3-Directory:
directory_c3 = 'Data/Phantomblock/raw_0_0' #relative path
sampling_rate_c3 = 15e6
# l15-Directory:
directory_l15 = 'Data/Phantomblock_L15_2-7.5_MHz_Phantom/raw_0_0' #relative path
sampling_rate_l15 = 30e6

a_scan_c3 = get_a_scan(directory_c3, sampling_rate_c3)
a_scan_l15 = get_a_scan(directory_l15, sampling_rate_l15)

x_fft_c3, y_fft_plot_c3 = definitions.amplitude_spectrum(sampling_rate_c3, a_scan_c3)
x_fft_l15, y_fft_plot_l15 = definitions.amplitude_spectrum(sampling_rate_l15, a_scan_l15)

plt.plot(x_fft_c3/1e6, y_fft_plot_c3)
plt.plot(x_fft_l15/1e6, y_fft_plot_l15)
plt.xlabel('Frequency ($MHz$)')
plt.ylabel('Amplitude ($Unit$)')
plt.show()

