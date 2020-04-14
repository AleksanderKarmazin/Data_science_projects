import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
import scipy as sp
from datetime import datetime 
from scipy.optimize import minimize

corona = pd.read_csv('D:/coronavirus.csv',sep=";")
corona.set_index('Date', inplace=True)
corona.index = pd.to_datetime(corona.index)

X = corona['X']
chi = corona['China']
fr = corona['France']
ir = corona['Iran']
it = corona['Italy']
sp = corona['Spain']
uk = corona['UK']
us = corona['US']
bg = corona['Belgium']
gm = corona['Germany']
nt = corona['Netherlands']
sw = corona['Switzerland']
tot = corona['Total']

dchi = chi[1:].values - chi[:-1].values
dfr = fr[1:].values - fr[:-1].values
dit = it[1:].values - it[:-1].values
diran = ir[1:].values - ir[:-1].values
dsp = sp[1:].values - sp[:-1].values
duk = uk[1:].values - uk[:-1].values
dus = us[1:].values - us[:-1].values
dbg = bg[1:].values - bg[:-1].values
dgm = gm[1:].values - gm[:-1].values
dnt = nt[1:].values - nt[:-1].values
dsw = sw[1:].values - sw[:-1].values
dtot = tot[1:].values - tot[:-1].values


X_long = np.arange(20, 200)
time_long = pd.date_range('2020-01-20', periods=180)


def resLogistic(coefficents):
    A0 = coefficents[0]
    A1 = coefficents[1]
    A2 = coefficents[2]
    teor = A0 / (1 + np.exp(A1 * (X.ravel() - A2)))
    
    return np.sum((teor - chi) ** 2)

minim = minimize(resLogistic, [3200, -.16, 46])
minim.x

plt.figure(figsize=(15,10))
teorChi = minim.x[0] / (1 + np.exp(minim.x[1] * (X_long - minim.x[2])))
plt.plot(X,chi,'ro', label='Фактические данные')
plt.plot(X_long[:80], teorChi[:80],'b', label='Аппроксимация и прогноз')
plt.xticks(X_long[:80][::2], time_long.date[:80][::2], rotation='90');
plt.title('Количество умерших всего, Китай', Size=20);
plt.ylabel('Количество умерших человек')
plt.legend()
plt.grid()


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
import scipy as sp
from datetime import datetime 
from scipy.optimize import minimize

corona = pd.read_csv('D:/coronavirus.csv',sep=";")
corona.set_index('Date', inplace=True)
corona.index = pd.to_datetime(corona.index)

X = corona['X']
chi = corona['China']


dchi = chi[1:].values - chi[:-1].values



X_long = np.arange(94, 274) #!!!!!!!!!!!!!!!
time_long = pd.date_range('2020-04-03', periods=180) #!!!!!!!!!!!!!!!


def resLogistic(coefficents):
    A0 = coefficents[0]
    A1 = coefficents[1]
    A2 = coefficents[2]
    teor = A0 / (1 + np.exp(A1 * (X.ravel() - A2)))
    
    return np.sum((teor - chi) ** 2)

minim = minimize(resLogistic, [211, -.16, 80]) #!!!!!!!!!!!!!!!
minim.x

plt.figure(figsize=(15,10))
teorChi = minim.x[0] / (1 + np.exp(minim.x[1] * (X_long - minim.x[2])))
plt.plot(X,chi,'ro', label='Фактические данные')
plt.plot(X_long[:80], teorChi[:80],'b', label='Аппроксимация и прогноз')
plt.xticks(X_long[:80][::2], time_long.date[:80][::2], rotation='90');
plt.title('Количество умерших всего, Китай', Size=20);
plt.ylabel('Количество умерших человек')
plt.legend()
plt.grid()



import math
import numpy as np
import pandas as pd
import scipy.optimize as optim
import matplotlib.pyplot as plt

# Import the data
data = pd.read_csv('D:/full_data_logistic.csv', sep=';')
data = data['total_cases']
data = data.reset_index(drop=False)
data.columns = ['Timestep', 'Total Cases']
data.head(11)

# Define funcion with the coefficients to estimate
def my_logistic(t, a, b, c):
    return c / (1 + a * np.exp(-b*t))


# Randomly initialize the coefficients
p0 = np.random.exponential(size=3)
p0

# Set min bound 0 on all coefficients, and set different max bounds for each coefficient
bounds = (0, [100000., 3., 1000000000.])


# Convert pd.Series to np.Array and use Scipy's curve fit to find the best Nonlinear Least Squares coefficients
x = np.array(data['Timestep']) + 1
y = np.array(data['Total Cases'])

(a,b,c),cov = optim.curve_fit(my_logistic, x, y, bounds=bounds, p0=p0)

# Show the coefficients
a,b,c

# Redefine the function with the new a, b and c
def my_logistic(t):
    return c / (1 + a * np.exp(-b*t))
    

plt.scatter(x, y)
plt.plot(x, my_logistic(x))
plt.title('Logistic Model vs Real Observations of China Coronavirus')
plt.legend([ 'Logistic Model', 'Real data'])
plt.xlabel('Time')
plt.ylabel('Infections')

#===========================================================================

import math
import numpy as np
import pandas as pd
import scipy.optimize as optim
import matplotlib.pyplot as plt

# Import the data
data = pd.read_csv('D:/full_data_logistic.csv', sep=';')
data = data['total_cases']
data = data.reset_index(drop=False)
data.columns = ['Timestep', 'Total Cases']
data.head(11)

# Define funcion with the coefficients to estimate
def my_logistic(t, a, b, c):
    return c / (1 + a * np.exp(-b*t))


# Randomly initialize the coefficients
p0 = np.random.exponential(size=3)
p0

# Set min bound 0 on all coefficients, and set different max bounds for each coefficient
bounds = (0, [100000., 3., 1000000000.])


# Convert pd.Series to np.Array and use Scipy's curve fit to find the best Nonlinear Least Squares coefficients
x = np.array(data['Timestep']) + 1
y = np.array(data['Total Cases'])

(a,b,c),cov = optim.curve_fit(my_logistic, x, y, bounds=bounds, p0=p0)

# Show the coefficients
a,b,c

# Redefine the function with the new a, b and c
def my_logistic(t):
    return c / (1 + a * np.exp(-b*t))
    
plt.scatter(x, y)
plt.plot(x, my_logistic(x))
plt.plot(X_long[:80], my_logistic[:80],'b')
plt.xticks(X_long[:80][::2], time_long.date[:80][::2], rotation='90');
plt.title('Logistic Model vs Real Observations of China Coronavirus')
plt.legend([ 'Logistic Model', 'Real data'])
plt.xlabel('Time')
plt.ylabel('Infections')

#===========================================================================

import math
import numpy as np
import pandas as pd
import scipy.optimize as optim
import matplotlib.pyplot as plt

# Import the data
data = pd.read_csv('D:/full_data_logistic.csv', sep=';')
data = data['total_cases']
data = data.reset_index(drop=False)
data.columns = ['Timestep', 'Total Cases']
data.head(11)

# Define funcion with the coefficients to estimate
def my_logistic(t, a, b, c):
    return c / (1 + a * np.exp(-b*t))


# Randomly initialize the coefficients
p0 = np.random.exponential(size=3)
p0

# Set min bound 0 on all coefficients, and set different max bounds for each coefficient
bounds = (0, [100000., 3., 1000000000.])


# Convert pd.Series to np.Array and use Scipy's curve fit to find the best Nonlinear Least Squares coefficients
x = np.array(data['Timestep']) + 1
y = np.array(data['Total Cases'])

(a,b,c),cov = optim.curve_fit(my_logistic, x, y, bounds=bounds, p0=p0)

# Show the coefficients
a,b,c

# Redefine the function with the new a, b and c
def my_logistic(t):
    return c / (1 + a * np.exp(-b*t))
    
plt.scatter(x, y)
plt.plot(x, my_logistic(x))
plt.plot(X_long[:80], my_logistic[:80],'b')
plt.xticks(X_long[:80][::2], time_long.date[:80][::2], rotation='90');
plt.title('Logistic Model vs Real Observations of China Coronavirus')
plt.legend([ 'Logistic Model', 'Real data'])
plt.xlabel('Time')
plt.ylabel('Infections')

# =========================================================================================

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
import scipy as sp
from datetime import datetime 
from scipy.optimize import minimize

corona = pd.read_csv('D:/coronavirus.csv',sep=";")
corona.set_index('Date', inplace=True)
corona.index = pd.to_datetime(corona.index)

X = corona['X']
chi = corona['China']
fr = corona['France']
ir = corona['Iran']
it = corona['Italy']
sp = corona['Spain']
uk = corona['UK']
us = corona['US']
bg = corona['Belgium']
gm = corona['Germany']
nt = corona['Netherlands']
sw = corona['Switzerland']
tot = corona['Total']

dchi = chi[1:].values - chi[:-1].values
dfr = fr[1:].values - fr[:-1].values
dit = it[1:].values - it[:-1].values
diran = ir[1:].values - ir[:-1].values
dsp = sp[1:].values - sp[:-1].values
duk = uk[1:].values - uk[:-1].values
dus = us[1:].values - us[:-1].values
dbg = bg[1:].values - bg[:-1].values
dgm = gm[1:].values - gm[:-1].values
dnt = nt[1:].values - nt[:-1].values
dsw = sw[1:].values - sw[:-1].values
dtot = tot[1:].values - tot[:-1].values


X_long = np.arange(20, 200)
time_long = pd.date_range('2020-01-20', periods=180)




def resLogisticUs(coefficents):
    A0 = coefficents[0]
    A1 = coefficents[1]
    A2 = coefficents[2]
    teor = A0 / (1 + np.exp(A1 * (X.ravel() - A2)))
    return np.sum((teor - us) ** 2)

minim = minimize(resLogisticUs, [3200, -.16, 100])
minim.x

plt.figure(figsize=(15,10))
teorUS = minim.x[0] / (1 + np.exp(minim.x[1] * (X_long - minim.x[2])))
plt.plot(X_long[:120], teorUS[:120],'b', label='Аппроксимация и прогноз')
plt.xticks(X_long[:120][::3], time_long.date[:120][::3], rotation='90');
plt.title('Количество умерших всего, США', Size=20);
plt.plot(X,us,'ro', label='Фактические данные')
plt.grid()
plt.legend()
plt.ylabel('Количество умерших человек')