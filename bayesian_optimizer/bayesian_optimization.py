#%%
from bayes_opt import BayesianOptimization
import sys
sys.path.insert(0, '/Users/jakobschaefer/Documents/rf-ultrasound')  # necessary to import functions from other directory
import functions_bayesian_optimization as functions_bayesian_optimization
import os
#%%

f = functions_bayesian_optimization.lizzi_feleppa


def config_parameters():
    version = '230330'
    transducer = 'C3_large' # with '_large' for spectral_shift and lizzi_feleppa and without '_large' for the others
    ref_parameter = 'CAP'

    return version, transducer, ref_parameter

def config_lizzi_feleppa():
    lizzi_feleppa_parameter = 'intercept' # MBF, slope or intercept
    # define where to choose ROI from
    ROI_type = 2 # 0: whole image, 1: middle of scanline (=around focal point), 2: specific ROI (defined below)
    len_ROI = 500 # the length of the examined ROI in px (only for ROI_type = 1)
    remove_outliers = False # True or False

    return lizzi_feleppa_parameter, ROI_type, len_ROI, remove_outliers

# window function = 0: none, 1: hanning, 2: hamming, 3: blackman

# pbounds for frequency independent evaluation
if f == functions_bayesian_optimization.pearson_correlation:
    pbounds = {'max_depth': (2002, 2704), 'min_depth': (0, 2000), 'number_of_scanlines': (2, 192), 'start_scanline': (0, 190)}

# pbounds for frequency specific evaluation
if f == functions_bayesian_optimization.frequency_specific_pearson_correlation:
    pbounds = {'hop_size': (1,10),'max_depth': (1600, 2704), 'min_depth': (0, 1500), 'number_of_scanlines': (2, 192), 'start_scanline': (0, 190), 'window_size': (10, 300), 'window_function': (0, 3)}

# pbounds for spectral shift evaluation
if f == functions_bayesian_optimization.spectral_shift:
    pbounds = {'spectral_shift_border': (2,10), 'hop_size': (1,8),'max_depth': (2001, 2704), 'min_depth': (0, 1500), 'number_of_scanlines': (2, 192), 'start_scanline': (0, 190), 'window_size': (30, 100), 'window_function': (0,3) }

# pbounds for spectral shift evaluation
if f == functions_bayesian_optimization.lizzi_feleppa:
    pbounds = {'start_ROI': (0,1500), 'end_ROI': (1600,2704),'start_scanline': (0, 190), 'number_of_scanlines': (2, 192), 'calibration_border': (2, 30), 'reflection_max': (1000,2500)}#, 'freq_shift_comp': (0,2000)}


optimizer = BayesianOptimization(
    f=f,
    pbounds=pbounds,
    random_state=0,
)

optimizer.maximize(
    init_points=4,
    n_iter=40,
)   
optimizer_max = optimizer.max
print(optimizer.max)



#%%
try:
    os.system('say "Optimierung abgeschlossen"')
except:
    import winsound
    frequency = 2500  # Set Frequency To 2500 Hertz
    duration = 1000  # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)

# write something into notion to inform about process 
import functions.def_notion_API as notion_def

# define if results should be saved to notion
safe_to_notion = True # True or False

# define notion parameters
version, notion_transducer, ref_parameter = config_parameters()

notion_window_func = 'Something'
notion_cropping_properties = 'Something'

 
notion_evaluation_parameter = str(f)[10:-19]
notion_results = 'Pearson coeff.: '+str(round(optimizer_max['target'], 3))

# round optimal parameters
rounded_optimal_params = dict()
for key in optimizer_max['params']:
    rounded_optimal_params[key] = round(optimizer_max['params'][key])

# read out optimal parameters
notion_optimal_params = str()
for key in rounded_optimal_params:
    if key == 'window_function':
        if rounded_optimal_params[key] == 0:
            rounded_optimal_params[key] = 'None'
        if rounded_optimal_params[key] == 1:
            rounded_optimal_params[key] = 'Hanning'
        if rounded_optimal_params[key] == 2:
            rounded_optimal_params[key] = 'Hamming'
        if rounded_optimal_params[key] == 3:
            rounded_optimal_params[key] = 'Blackman'

    elif key == 'Phantom':
        if rounded_optimal_params[key] == 0:
            rounded_optimal_params[key] = 'Phantom'
        if rounded_optimal_params[key] == 1:
            rounded_optimal_params[key] = 'Met_Phan'
        if rounded_optimal_params[key] == 2:
            rounded_optimal_params[key] = 'AIR00000'
        if rounded_optimal_params[key] == 3:
            rounded_optimal_params[key] = 'Wat_Phan'
        if rounded_optimal_params[key] == 4:
            rounded_optimal_params[key] = 'fft_all_recordings'
for key in rounded_optimal_params:
    notion_optimal_params += key + ': ' + str(rounded_optimal_params[key]) + '\n'

# define text to write into notion
text = 'Version: ' + version+' ('+notion_transducer + ')\n' + 'Ref. parameter: '+ref_parameter +'\n'+notion_results+'\n'+notion_optimal_params

# define notion database id
if f == functions_bayesian_optimization.pearson_correlation:
    block_id = '92597b4e6b714ba9ad85adabd908e3a3'
if f == functions_bayesian_optimization.frequency_specific_pearson_correlation:
    block_id = '92597b4e6b714ba9ad85adabd908e3a3'
if f == functions_bayesian_optimization.spectral_shift:
    block_id = 'fd8ae4c7c8af48f5bea350ea64b37d28'
if f == functions_bayesian_optimization.lizzi_feleppa:
    block_id = '60b310b80dbc46838f368d5cf2e0faa9'
    lizzi_feleppa_parameter, ROI_type, len_ROI, remove_outliers = config_lizzi_feleppa()

    if remove_outliers:
        ref_parameter += ' (outliers removed)'
    text = 'Version: ' + version+' ('+notion_transducer + ')\n' + 'Ref. parameter: '+ 'square and sum of all Lizzi-Feleppa Params'+' '+ref_parameter+'\n'+notion_results+'\n'

    text += notion_optimal_params



notion_def.write_text(block_id, text)
#%%