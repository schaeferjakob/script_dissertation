#%% 
import numpy as np
import pandas as pd
from statsmodels.stats.anova import AnovaRM
import scipy.stats as stats
import matplotlib.pyplot as plt


# local definitions 
def test_norm_dist(parameter, data):
    # plot Q-Q plot
    stats.probplot(data, dist="norm", plot=plt)
    plt.title('Q-Q Plot '+parameter)
    plt.show()

    # test with shapiro-wilk test
    shapiro_stats = stats.shapiro(data)
    # p-value < 0.05 -> not normally distributed
    print('P-value of Shapiro-Wilk test for '+parameter+': '+str(shapiro_stats[1]))
    if shapiro_stats[1] < 0.05:
        print('P-Value < 0.05 -> not normally distributed')
    else:
        print('P-Value > 0.05 -> normally distributed')

#%% sample size calculation for pearson correlation

# correlation coefficient
r0 = 0.634
r1 = 0.693
# significance level
alpha = 0.05
z_alpha_2 = stats.norm.ppf(1-alpha)
# power
beta = 0.2
z_beta = stats.norm.ppf(1-beta)

# calculate z-score
z0 = 0.5*np.log((1+r0)/(1-r0))
z1 = 0.5*np.log((1+r1)/(1-r1))

# calculate sample size
n = 3+((z_alpha_2+z_beta)/np.abs(z1-z0))**2

print('Sample size: '+str(n))

#%% test if difference between correlation coefficients is significant
r1 = 0.707
r2 = 0.548

n1 = 111
n2 = 111

z1 = 0.5*np.log((1+r1)/(1-r1))
z2 = 0.5*np.log((1+r2)/(1-r2))

Zobs = (z1-z2)/np.sqrt(1/(n1-3)+1/(n2-3))

print(Zobs)

if np.abs(Zobs) > 1.96:
    print('Significant difference between correlation coefficients')
else:
    print('No significant difference between correlation coefficients')
#%% 
transducer = 'C3'
version = '230330'

if transducer == 'C3':
    sampling_rate = 15e6
if transducer == 'L15':
    sampling_rate = 30e6
# %% test normal distibution of mean slopes
# load values
filepath = '/Users/jakobschaefer/Documents/RF_US_AI/MLP_QUS_Parameter/Input_data/mean_slopes_all_recordings_'+transducer+'_'+version+'.npy'
mean_slopes = np.load(filepath)
parameter = 'Attenuation'
test_norm_dist(parameter, mean_slopes)

# %% test normal distibution of freqspec mean slopes
freqspec_filepath = '/Users/jakobschaefer/Documents/RF_US_AI/MLP_QUS_Parameter/Input_data/mean_slopes_all_recordings_freqspec_'+transducer+'_'+version+'.npy'

mean_freqspec_slopes = np.load(freqspec_filepath)

nr_patients = len(mean_freqspec_slopes[:,0])
nr_freqs = len(mean_freqspec_slopes[0,:])

nr_normally_distributed = 0
not_normally_distributed = []
for i in range(nr_freqs):
    # test with shapiro-wilk test
    shapiro_stats = stats.shapiro(mean_freqspec_slopes[:,i])

    if shapiro_stats[1] < 0.05:
        not_normally_distributed.append(i)
        # perform Q-Q plot
        stats.probplot(mean_freqspec_slopes[:,i], dist="norm", plot=plt)
        frequency_section = (i+1)*(sampling_rate/2/nr_freqs)/1e6 # MHz
        plt.title('Frequency section '+str(round(frequency_section, 3))+' ± '+str(round(sampling_rate/nr_freqs/1e6, 3))+' MHz', fontweight = 'bold', fontsize = 12)
        plt.xlabel('Theoretical quantiles', fontsize = 12)
        plt.ylabel('Ordered values', fontsize = 12)
        plt.show()
        # p-value < 0.05 -> not normally distributed
        print('P-value of Shapiro-Wilk test for frequency specific attenuation: '+str(shapiro_stats[1]))
    else:
        nr_normally_distributed += 1 

print('Number of normally distributed frequency specific mean slopes: '+str(nr_normally_distributed))
print('Not normally distributed frequency specific mean slopes: '+str(not_normally_distributed)+'('+str(len(not_normally_distributed))+'/'+str(nr_freqs)+')')
# %% test normal distibution of lizzi-feleppa parameters
# load values
version= '230330'
transducer = 'C3_large'
lizzi_filepath = '/Users/jakobschaefer/Documents/RF_US_AI/MLP_QUS_Parameter/Lizzi_Feleppa_Results/Lizzi_Feleppa_Results_'+version+'_'+transducer+'.npy'

lizzi_feleppa_results = np.load(lizzi_filepath, allow_pickle=True)
lizzi_feleppa_results = lizzi_feleppa_results.item()

# MBF distribution
MBF_data = lizzi_feleppa_results['MBF']
test_norm_dist('MBF', MBF_data)

# slope distribution
slope_data = lizzi_feleppa_results['slope']
test_norm_dist('slope', slope_data)

# intercept distribution
intercept_data = lizzi_feleppa_results['intercept']
test_norm_dist('intercept', intercept_data)


# %%
# perform an ANOVA on the frequency specific attenuation values
# load values
freqspec_filepath = '/Users/jakobschaefer/Documents/RF_US_AI/MLP_QUS_Parameter/Input_data/mean_slopes_all_recordings_freqspec_'+transducer+'_'+version+'.npy'

mean_freqspec_slopes = np.load(freqspec_filepath)

nr_patients = len(mean_freqspec_slopes[:,0])
nr_freqs = len(mean_freqspec_slopes[0,:])

df = pd.DataFrame({'patient': np.repeat(np.arange(nr_patients),nr_freqs),
                   'frequency': np.tile(np.arange(nr_freqs),nr_patients),
                   'mean_freqspec_slopes': mean_freqspec_slopes.flatten()})



# perform ANOVA
print(AnovaRM(data=df, depvar='mean_freqspec_slopes', subject='patient', within=['frequency']).fit())

#%% since data is not normally distributed, perform a friedman test

# perform friedman test
print(stats.friedmanchisquare(*mean_freqspec_slopes))

# %%
