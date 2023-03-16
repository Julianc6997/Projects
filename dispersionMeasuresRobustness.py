import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

distribution = str(input('Choose the probability distribution from Normal, Skewnorm, Gennormal, Beta, Levy, Lognormal or t: '))

if distribution not in ('Normal', 'Skewnorm', 'Gennormal', 'Beta', 'Levy', 'Lognormal', 't'):
    raise Exception('We could not find the distribution, please use one of the options given')

tuningSamples = int(input('Number of samples to determine dispersion measures: '))
randomSamples = int(input('Number of samples to evaluate disperison measures: '))
sampleSize = int(input('Size of the random samples: '))

# Function that defines required input parameter, calculates and estimates dispersion measures, it finally creates the samples
def Generator(distr, tuningSamples, randomSamples, sampleSize):

    if distr == 'Normal':

        location = float(input('Location parameter(mu): '))
        scale = float(input('Scale parameter(sigma): '))
        if not (isinstance(location, (float, int)) and isinstance(scale, (float, int))):
            raise Exception('Please use real numbers as inputs')
        if scale < 0:
            raise Exception('Normal distribution is not defined for scale < 0')

        sample = sp.stats.norm.rvs(loc=location, scale=scale, size=tuningSamples)
        mean = sp.stats.norm.mean(loc=location, scale=scale)

        var = sp.stats.norm.var(loc=location, scale=scale)
        std = sp.stats.norm.std(loc=location, scale=scale)
        p90p10range = np.percentile(sample, 90) - np.percentile(sample, 10)
        meanabsdevs = np.mean([np.absolute(mean - i) for i in sample])
        medianabsdevs = np.median([np.absolute(mean - i) for i in sample])
        iqr = sp.stats.iqr(sample)

        samples = sp.stats.norm.rvs(loc=location, scale=scale, size=(randomSamples,sampleSize))
        del location, scale, randomSamples, sampleSize

    elif distr == 'Skewnorm':

        skew = float(input('Skew parameter(a): '))
        location = float(input('Location parameter(xi): '))
        scale = float(input('Scale parameter(omega): '))
        if not (isinstance(skew, (float, int)) and isinstance(location, (float, int)) and isinstance(scale, (float, int))):
            raise Exception('Please use real numbers as imputs')
        if scale < 0:
            raise Exception('Skew normal distribution is not defined for scale < 0')
        
        sample = sp.stats.skewnorm.rvs(a=skew, loc=location, scale=scale, size=tuningSamples)
        mean = sp.stats.skewnorm.mean(a=skew, loc=location, scale=scale)

        var = sp.stats.skewnorm.var(a=skew, loc=location, scale=scale)
        std = sp.stats.skewnorm.std(a=skew, loc=location, scale=scale)
        p90p10range = np.percentile(sample, 90) - np.percentile(sample, 10)
        meanabsdevs = np.mean([np.absolute(mean - i) for i in sample])
        medianabsdevs = np.median([np.absolute(mean - i) for i in sample])
        iqr = sp.stats.iqr(sample)

        samples = sp.stats.skewnorm.rvs(a=skew, loc=location, scale=scale, size=(randomSamples,sampleSize))
        del skew, location, scale, randomSamples, sampleSize

    elif distr == 'Gennormal':
        
        kurtosis = input('Kurtosis parameter(beta): ')
        location = input('Location parameter(a): ')
        scale = input('Scale parameter(mu): ')
        if not (isinstance(kurtosis, (float, int)) and isinstance(location, (float, int)) and isinstance(scale, (float, int))):
            raise Exception('Please use real numbers as inputs')
        if kurtosis < 0:
            raise Exception('Generalized normal distribution is not defined for kurtosis < 0')
        if scale < 0:
            raise Exception('Generalized normal distribution is not defined for scale < 0')

        sample = sp.stats.gennorm.rvs(beta=kurtosis, loc=location, scale=scale, size=tuningSamples)
        mean = sp.stats.gennorm.mean(beta=kurtosis, loc=location, scale=scale)

        var = sp.stats.gennorm.var(beta=kurtosis, loc=location, scale=scale)
        std = sp.stats.gennorm.std(beta=kurtosis, loc=location, scale=scale)
        p90p10range = np.percentile(sample, 90) - np.percentile(sample, 10)
        meanabsdevs = np.mean([np.absolute(mean - i) for i in sample])
        medianabsdevs = np.median([np.absolute(mean - i) for i in sample])
        iqr = sp.stats.iqr(sample)

        samples = sp.stats.gennorm.rvs(beta=kurtosis, loc=location, scale=scale, size=(randomSamples,sampleSize))
        del kurtosis, location, scale, randomSamples, sampleSize

    elif distr == 'Beta':

        alpha = float(input('Shape parameter(alpha): '))
        beta = float(input('Inverse scale(beta): '))
        location = float(input('Location parameter: '))
        scale = float(input('Scale parameter: '))
        if not (isinstance(alpha, (float, int)) and isinstance(beta, (float, int)) and isinstance(location, (float, int)) and isinstance(scale, (float, int))):
            raise Exception('Please use real numbers as inputs')
        if (alpha or beta) < 0:
            raise Exception('Beta distribution is not defined for alpha or beta < 0')
        sample = sp.stats.beta.rvs(a=alpha, b=beta, loc=location, scale=scale, size=tuningSamples)
        mean = sp.stats.beta.mean(a=alpha, b=beta, loc=location, scale=scale)

        var = sp.stats.beta.var(a=alpha, b=beta, loc=location, scale=scale)
        std = sp.stats.beta.std(a=alpha, b=beta, loc=location, scale=scale)
        p90p10range = np.percentile(sample, 90) - np.percentile(sample, 10)
        meanabsdevs = np.mean([np.absolute(mean - i) for i in sample])
        medianabsdevs = np.median([np.absolute(mean - i) for i in sample])
        iqr = sp.stats.iqr(sample)

        samples = sp.stats.beta.rvs(a=alpha, b=beta, loc=location, scale=scale, size=(randomSamples,sampleSize))
        del alpha, beta, location, scale, randomSamples, sampleSize

    elif distr == 'Gamma':

        alpha = float(input('Alpha parameter: '))
        location = float(input('Location paramter: '))
        scale = float(input('Scale parameter: '))
        if not (isinstance(alpha, (float, int)) and isinstance(location, (float, int)) and isinstance(scale, (float, int))):
            raise Exception('Please use real numbers as inputs')
        if alpha < 0:
            raise Exception('Gamma distribution is no defined for alpha < 0')
        sample = sp.stats.gamma.rvs(a=alpha, loc=location, scale=scale, size=tuningSamples)
        mean = sp.stats.gamma.mean(a=alpha, loc=location, scale=scale)

        var = sp.stats.gamma.var(a=alpha, loc=location, scale=scale)
        std = sp.stats.gamma.std(a=alpha, loc=location, scale=scale)
        p90p10range = np.percentile(sample, 90) - np.percentile(sample, 10)
        meanabsdevs = np.mean([np.absolute(mean - i) for i in sample])
        medianabsdevs = np.median([np.absolute(mean - i) for i in sample])
        iqr = sp.stats.iqr(sample)

        samples = sp.stats.gamma.rvs(a=alpha, loc=location, scale=scale, size=(randomSamples,sampleSize))
        del alpha, location, scale, randomSamples, sampleSize

    elif distr == 'Levy':

        location = float(input('Location parameter(mu): '))
        scale = float(input('Scale parameter(c): '))
        if not (isinstance(location, (float, int)) and isinstance(scale, (float, int))):
            raise Exception('Please use real numbers as inputs')
        if scale < 0:
            raise Exception('Levy distribution is not defined for c < 0')
        sample = sp.stats.levy.rvs(loc=location, scale=scale, size=tuningSamples)
        mean = sp.stats.levy.mean(loc=location, scale=scale)

        var = sp.stats.levy.var(loc=location, scale=scale)
        std = sp.stats.levy.std(loc=location, scale=scale)
        p90p10range = np.percentile(sample, 90) - np.percentile(sample, 10)
        meanabsdevs = np.mean([np.absolute(mean - i) for i in sample])
        medianabsdevs = np.median([np.absolute(mean - i) for i in sample])
        iqr = sp.stats.iqr(sample)

        samples = sp.stats.levy.rvs(loc=location, scale=scale, size=(randomSamples,sampleSize))
        del location, scale, randomSamples, sampleSize

    elif distr == 'Lognormal':

        mu = float(input('Location parameter(mu): '))
        sigma = float(input('Scale, skew and kurtosis parameter(sigma): '))
        scale = float(input('Scale parameter: '))
        if not (isinstance(mu, (float, int)) and isinstance(sigma, (float, int)) and isinstance(scale, (float, int))):
            raise Exception('Please use real numbers as inputs')
        if sigma < 0:
            raise Exception('Lognormal distribution is not defined for sigma < 0')
        
        sample = sp.stats.lognorm.rvs(s=sigma, loc=mu, scale=scale, size=tuningSamples)
        mean = sp.stats.lognorm.mean(s=sigma, loc=mu, scale=scale)

        var = sp.stats.lognorm.var(s=sigma, loc=mu, scale=scale)
        std = sp.stats.lognorm.std(s=sigma, loc=mu, scale=scale)
        p90p10range = np.percentile(sample, 90) - np.percentile(sample, 10)
        meanabsdevs = np.mean([np.absolute(mean - i) for i in sample])
        medianabsdevs = np.median([np.absolute(mean - i) for i in sample])
        iqr = sp.stats.iqr(sample)

        samples = sp.stats.lognorm.rvs(s=sigma, loc=mu, scale=scale, size=(randomSamples,sampleSize))
        del mu, sigma, scale, randomSamples, sampleSize

    elif distr == 't':
        
        df = float(input('Degrees of freedom parameter(nu): '))
        location = float(input('Location parameter: '))
        scale = float(input('Scale parameter: '))
        if not (isinstance(df, (float, int)) and isinstance(location, (float, int)) and isinstance(scale, (float, int))):
            raise Exception('Please use real numbers as inputs')
        if df < 0:
            raise Exception('Lognormal distribution is not defined for sigma < 0')
        sample = sp.stats.t.rvs(df=df, loc=location, scale=scale, size=tuningSamples)
        mean = sp.stats.t.mean(df=df, loc=location, scale=scale)

        var = sp.stats.t.var(df=df, loc=location, scale=scale)
        std = sp.stats.t.std(df=df, loc=location, scale=scale)
        p90p10range = np.percentile(sample, 90) - np.percentile(sample, 10)
        meanabsdevs = np.mean([np.absolute(mean - i) for i in sample])
        medianabsdevs = np.median([np.absolute(mean - i) for i in sample])
        iqr = sp.stats.iqr(sample) 

        samples = sp.stats.t.rvs(df=df, loc=location, scale=scale, size=(randomSamples,sampleSize))  
        del df, location, scale, randomSamples, sampleSize    

    return var, std, p90p10range, meanabsdevs, medianabsdevs, iqr, samples

# Store dispersion measures in variables
var, std, p90p10range, meanabsdevs, medianabsdevs, iqr, samples = Generator(distr=distribution, tuningSamples=tuningSamples, randomSamples=randomSamples, sampleSize=sampleSize)

# Prepare absolute deviations to compute mean absolute deviations
means = np.mean(samples, axis=1)
absdevs = [np.absolute(x - means) for x, means in zip(samples, means)]

# Compute the proportion between the real and observed value
# For readability: In every variable np.divide takes as argument a np.array that contains the given dispersion method
# for every sample generated
def results(samples, var, std, p90p10range, meanabsdevs, medianabsdevs, iqr):
    
    StDv = np.divide(np.std(samples, axis=1), std)
    Variance = np.divide(np.var(samples, axis=1), var)
    IQRS = np.divide(sp.stats.iqr(samples, axis=1), iqr)
    MedianAbsDevs = np.divide(sp.stats.median_abs_deviation(samples, axis=1), medianabsdevs)
    P90P10Range = np.divide([x - y for x, y in zip(np.percentile(samples, 90, axis =1), np.percentile(samples, 10, axis = 1))], p90p10range)
    MeanAbsDevs = np.divide(np.mean(absdevs, axis=1), meanabsdevs)

    return StDv, Variance, IQRS, MedianAbsDevs, P90P10Range, MeanAbsDevs

StDv, Variance, IQRS, MedianAbsDevs, P90P10Range, MeanAbsDevs = results(samples, var, std, p90p10range, meanabsdevs, medianabsdevs, iqr)

# Delete variables that are not going to be used again 
del means, absdevs, samples

# Return the 97.5, 75, 50, 25 and 2.5th percentiles for every dispersion measure
print('Percentiles of Standard deviation: ' + str(np.percentile(StDv, [97.5, 75, 50, 25, 2.5])))
print('Percentiles of Variance: ' + str(np.percentile(Variance, [97.5, 75, 50, 25, 2.5])))
print('Percentiles of IQ Range: ' + str(np.percentile(IQRS, [97.5, 75, 50, 25, 2.5])))
print('Percentiles of Median Absolute Devs: ' + str(np.percentile(MedianAbsDevs, [97.5, 75, 50, 25, 2.5])))
print('Percentiles of P90-10 Range: ' + str(np.percentile(P90P10Range, [97.5, 75, 50, 25, 2.5])))
print('Percentiles of Mean Absolute Devs: ' + str(np.percentile(MeanAbsDevs, [97.5, 75, 50, 25, 2.5])))

# Print Standard deviation of dispersion measure from lowest to gratest
results = [(np.std(StDv), 'StDv'), (np.std(Variance), 'Variance'), (np.std(IQRS), 'IQ Range'), (np.std(MedianAbsDevs), 'Median Abs Devs'), (np.std(P90P10Range), 'P90-P10 Range'), (np.std(MeanAbsDevs), 'Mean Abs Devs')]
results = sorted(results)
for result in results:
    print(str(result[1]) + ' has a standard deviation of ' + str(result[0]))

# Create a list of datasets and their labels
data = [StDv, Variance, IQRS, MedianAbsDevs, P90P10Range, MeanAbsDevs]
labels = ['Standard Deviation', 'Variance', 'IQ Range', 'Median Abs Devs', 'P90-10 Range', 'Mean Abs Devs']

# Delete variables that are not going to be used again 
del StDv, Variance, IQRS, MedianAbsDevs, P90P10Range, MeanAbsDevs

# Sort the list of datasets and labels based on their standard deviations
data_labels = sorted(zip(data, labels), key=lambda x: np.std(x[0]))
data, labels = zip(*data_labels)

# Plot the box plots side by side with labels
fig, ax = plt.subplots(figsize=(15, 5))
ax.boxplot(data, labels=labels)
ax.set_xticklabels(labels)
plt.show()

# Delete variables that are not going to be used again 
del data, labels, data_labels, fig, ax
