import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta

df = pd.read_csv("./Data/Nat_Gas.csv")

df['Dates'] = pd.to_datetime(df['Dates'])
df['Prices'] = pd.to_numeric(df['Prices'])
df = df.dropna()

plt.plot(df['Dates'], df['Prices'], marker='o')
plt.title("Natural Gas Prices")
plt.xlabel("Date")
plt.ylabel("Price")
plt.tight_layout()
plt.show()

df['DayIndex'] = df['Dates'].map(lambda d: d.toordinal())
x = df['DayIndex'].values
y = df['Prices'].values

# --- Linear Trend Fit ---
def fitLinear(x, y):
    xMean, yMean = np.mean(x), np.mean(y)
    slope = np.sum((x - xMean) * (y - yMean)) / np.sum((x - xMean)**2)
    intercept = yMean - slope * xMean
    return slope, intercept

slope, intercept = fitLinear(x, y)
df['LinearTrend'] = x * slope + intercept

residual = y - df['LinearTrend'].values
angFreq = 2 * np.pi / 365
df['sin'] = np.sin(angFreq * x)
df['cos'] = np.cos(angFreq * x)

def fitSeasonality(y, sinX, cosX):
    a = np.sum(y * sinX) / np.sum(sinX ** 2)
    b = np.sum(y * cosX) / np.sum(cosX ** 2)
    return a, b

a, b = fitSeasonality(residual, df['sin'], df['cos'])
amplitude = np.sqrt(a**2 + b**2)
phase = np.arctan2(b, a)

lastDate = df['Dates'].max()
futureDates = pd.date_range(start=lastDate + pd.Timedelta(days=1), end=lastDate + pd.DateOffset(years=2), freq='D')
futureDatesID = futureDates.map(lambda d: d.toordinal())

# Estimate prices for future dates
# Reconstruct model estimate for current data
df['Estimate'] = [
    slope * d + amplitude * np.sin(angFreq * d + phase) + intercept
    for d in df['DayIndex']
]

extrapolations = [
    slope * day + amplitude * np.sin(angFreq * day + phase) + intercept
    for day in futureDatesID
]

# Combine with existing data for full view
allDates = pd.concat([df['Dates'], pd.Series(futureDates)])
allPrices = np.concatenate([df['Prices'].values, [np.nan] * len(futureDates)])
allExtrapolations = np.concatenate([df['Estimate'].values, extrapolations])

plt.plot(df['Dates'], df['Prices'], 'o', label='Original Prices')
plt.plot(allDates, allExtrapolations, '-', label='Trend + Season Estimate')
plt.title("Extrapolated Natural Gas Prices")
plt.xlabel("Date")
plt.ylabel("Price")
plt.tight_layout()
plt.show()

def estimatePrice(date: pd.Timestamp):
    dateID = date.toordinal()
    linear = slope * dateID + intercept
    seasonal = amplitude * np.sin(angFreq * dateID + phase)
    return linear + seasonal
