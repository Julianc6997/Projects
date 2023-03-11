import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# Function that calculates population variance and standard deviation, and estimates percentile 90 - percentile 10 range, mean
# absolute deviation, median absolute deviation and inter quartile range
def parameterGenerator(a, loc, scale, size):
    sample = sp.stats.skewnorm.rvs(a, loc, scale, size)
    mean = np.mean(sample)

    var = sp.stats.skewnorm.var(a, loc, scale)
    std = sp.stats.skewnorm.std(a, loc, scale)
    p90p10range = np.percentile(sample, 90) - np.percentile(sample, 10)
    meanabsdevs = np.mean([np.absolute(mean - i) for i in sample])
    medianabsdevs = np.median([np.absolute(mean - i) for i in sample])
    iqr = sp.stats.iqr(sample)

    return var, std, p90p10range, meanabsdevs, medianabsdevs, iqr

# Store dispersion measures in variables
var, std, p90p10range, meanabsdevs, medianabsdevs, iqr = parameterGenerator(a=0.4, loc=0, scale=1, size=1000)

# Generate random samples
samples = sp.stats.skewnorm.rvs(a=0.8, loc=0, scale=1, size=(100000,10))

# Prepare absolute deviations to compute mean absolute deviations
means = np.mean(samples, axis=1)
absdevs = [np.absolute(x - means) for x, means in zip(samples, means)]

# Compute the proportion between the real and observed value
# For readability: In every variable np.divide takes as argument a np.array that contains the given dispersion method
# for every sample generated
StDv = np.divide(np.std(samples, axis=1), std)
Variance = np.divide(np.var(samples, axis=1), var)
IQRS = np.divide(sp.stats.iqr(samples, axis=1), iqr)
MedianAbsDevs = np.divide(sp.stats.median_abs_deviation(samples, axis=1), medianabsdevs)
P90P10Range = np.divide([x - y for x, y in zip(np.percentile(samples, 90, axis =1), np.percentile(samples, 10, axis = 1))], p90p10range)
MeanAbsDevs = np.divide(np.mean(absdevs, axis=1), meanabsdevs)

# Return the 97.5, 75, 50, 25, 0.5th percentiles for every dispersion measure
print('Percentiles of Standard deviation: ' + str(np.percentile(StDv, [97.5, 75, 50, 25, 0.5])))
print('Percentiles of Variance: ' + str(np.percentile(Variance, [97.5, 75, 50, 25, 0.5])))
print('Percentiles of IQ Range: ' + str(np.percentile(IQRS, [97.5, 75, 50, 25, 0.5])))
print('Percentiles of Median Absolute Devs: ' + str(np.percentile(MedianAbsDevs, [97.5, 75, 50, 25, 0.5])))
print('Percentiles of P90-10 Range: ' + str(np.percentile(P90P10Range, [97.5, 75, 50, 25, 0.5])))
print('Percentiles of Mean Absolute Devs: ' + str(np.percentile(MeanAbsDevs, [97.5, 75, 50, 25, 0.5])))

# Create a list of datasets and their labels
data = [StDv, Variance, IQRS, MedianAbsDevs, P90P10Range, MeanAbsDevs]
labels = ['Standard Deviation', 'Variance', 'IQ Range', 'Median Abs Devs', 'P90-10 Range', 'Mean Abs Devs']

# Sort the list of datasets and labels based on their standard deviations
data_labels = sorted(zip(data, labels), key=lambda x: np.std(x[0]))
data, labels = zip(*data_labels)

# Plot the box plots side by side with labels
fig, ax = plt.subplots(figsize=(15, 5))
ax.boxplot(data, labels=labels)
ax.set_xticklabels(labels)
plt.show()