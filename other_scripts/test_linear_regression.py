# goal is to test wether the quotient of slopes of different linear regressions changes changes, when the same normalization is applied on all regression lines

#%% imports
import numpy as np
import matplotlib.pyplot as plt

# create a array with a slope of -1 and some noise
x1 = np.arange(30,100)
y1 = -1*x1 + np.random.normal(0, 10, 70)
# create a array with a slope of -2 and some noise
x2 = np.arange(30,100)
y2 = -2*x2 + np.random.normal(0, 10, 70)
# create a array with a slope of -0.5 and some noise
x3 = np.arange(30,100)
y3 = -0.5*x3 + np.random.normal(0, 10, 70)

# calculate the linear regression for all three arrays
slope1, intercept1 = np.polyfit(x1, y1, 1)
slope2, intercept2 = np.polyfit(x2, y2, 1)
slope3, intercept3 = np.polyfit(x3, y3, 1)

# define the regression lines for all three arrays
regression_line1 = slope1*x1+intercept1
regression_line2 = slope2*x2+intercept2
regression_line3 = slope3*x3+intercept3

#%% plot the three arrays and the regression lines in three different plots
plt.figure()
plt.plot(x1, y1)
plt.plot(x1, regression_line1)
plt.xlabel('x1')
plt.ylabel('y1')
plt.show()
print('Slope of regression line 1: '+str(slope1))

plt.figure()
plt.plot(x2, y2)
plt.plot(x2, regression_line2)
plt.xlabel('x2')
plt.ylabel('y2')
plt.show()
print('Slope of regression line 2: '+str(slope2))

plt.figure()
plt.plot(x3, y3)
plt.plot(x3, regression_line3)
plt.xlabel('x3')
plt.ylabel('y3')
plt.show()
print('Slope of regression line 3: '+str(slope3))

#%% calculate the quotient of the slopes of the three regression lines
quotient1 = slope1/slope2
quotient2 = slope1/slope3
quotient3 = slope2/slope3

print('Quotient of slope1/slope2: '+str(quotient1))
print('Quotient of slope1/slope3: '+str(quotient2))
print('Quotient of slope2/slope3: '+str(quotient3))

#%% apply normalization to the three arrays and calculate the linear regression again
# create a fourth random array with a slope of s_norm and some noise for normalization
s_norm = -1
x_norm = np.arange(30,100)
y_norm = s_norm * x_norm + np.random.normal(0, 0.001, 70)

plt.figure()
plt.plot(x_norm, y_norm)
plt.xlabel('x_norm')
plt.ylabel('y_norm')
plt.show()


# divide the three arrays by the normalization array
y1_norm = y1/y_norm
y2_norm = y2/y_norm
y3_norm = y3/y_norm

# plot the three normalized arrays with their original arrays
plt.figure()
#plt.plot(x1, y1, label = 'y1')
plt.plot(x1, y1_norm, label = 'y1_norm')
plt.xlabel('x1')
plt.ylabel('y1')
plt.legend()
plt.show()

plt.figure()
#plt.plot(x2, y2, label = 'y2')
plt.plot(x2, y2_norm, label = 'y2_norm')
plt.xlabel('x2')
plt.ylabel('y2')
plt.legend()
plt.show()

plt.figure()
#plt.plot(x3, y3, label = 'y3')
plt.plot(x3, y3_norm, label = 'y3_norm')
plt.xlabel('x3')
plt.ylabel('y3')
plt.legend()
plt.show()



# calculate the linear regression for all three normalized arrays
slope1_norm, intercept1_norm = np.polyfit(x_norm, y1_norm, 1)
slope2_norm, intercept2_norm = np.polyfit(x_norm, y2_norm, 1)
slope3_norm, intercept3_norm = np.polyfit(x_norm, y3_norm, 1)

# define the regression lines for all three normalized arrays
regression_line1_norm = slope1_norm*x_norm+intercept1_norm
regression_line2_norm = slope2_norm*x_norm+intercept2_norm
regression_line3_norm = slope3_norm*x_norm+intercept3_norm

# print the slopes of the three normalized regression lines and the quotient of the slopes
print('Slope of regression line 1 normalized: '+str(slope1_norm))
print('Slope of regression line 2 normalized: '+str(slope2_norm))
print('Slope of regression line 3 normalized: '+str(slope3_norm))

quotient1_norm = slope1_norm/slope2_norm
quotient2_norm = slope1_norm/slope3_norm
quotient3_norm = slope2_norm/slope3_norm

print('Quotient of slope1_norm/slope2_norm: '+str(quotient1_norm))
print('Quotient of slope1_norm/slope3_norm: '+str(quotient2_norm))
print('Quotient of slope2_norm/slope3_norm: '+str(quotient3_norm))





# Quotient of slope1/slope2: 0.4993369437908597
# Quotient of slope1/slope3: 2.023667586391417
# Quotient of slope2/slope3: 4.05270952120659



# %%
# create a exponantially decreasing signal
noise_factor = 0.0
signal_length = 1000
factor_signal_2 = 5

x = np.arange(0,signal_length)
y = np.exp(-0.01*x)



y_noise_1 = y + np.random.normal(0, noise_factor, signal_length)
y_noise_1 = np.abs(y_noise_1)
y_noise_2 = y*factor_signal_2 + np.random.normal(0, noise_factor, signal_length)
y_noise_2 = np.abs(y_noise_2)

plt.figure(figsize=(5,5))
plt.plot(x, y_noise_1, label = 'Signal 1')
plt.plot(x, y_noise_2, label = 'Signal 2')
plt.title('Two exponentially decreasing signals with noise')
plt.legend()
plt.show()

# create log10 of both signals
y_log_noise_1 = np.log10(y_noise_1)
y_log_noise_2 = np.log10(y_noise_2)

# calculate linear regression
slope_y_noise, intercept_y_noise = np.polyfit(x, y_log_noise_1, 1)
slope_y_noise_2, intercept_y_noise_2 = np.polyfit(x, y_log_noise_2, 1)

# plot
plt.figure(figsize=(5,5))
plt.plot(x, y_log_noise_1, label = 'Log10 of Signal 1')
plt.plot(x, y_log_noise_2, label = 'Log10 of Signal 2')
plt.plot(x, slope_y_noise*x+intercept_y_noise, color = '#00305D', label = 'Linear regression of Signal 1')
plt.plot(x, slope_y_noise_2*x+intercept_y_noise_2,  color = 'red', label = 'Linear regression of Signal 2')
plt.title('Log10 of Signals 1 and 2 with linear regression')
plt.legend()
plt.show()


print('Slope of regression line 1: '+str(slope_y_noise))
print('Slope of regression line 2: '+str(slope_y_noise_2))

# %%
