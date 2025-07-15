import pandas as pd
from TaskOne import estimatePrice

def ContractCalculation(injectionDates, withdrawalDates, injRate, withDrawRate, storeCost, maxStore, priceModel):
    
    injectionDates = pd.to_datetime(injectionDates)
    withdrawalDates = pd.to_datetime(withdrawalDates)
    
    dates = pd.date_range(start=min(injectionDates.min(), withdrawalDates.min()),
                        end=max(injectionDates.max(), withdrawalDates.max()),
                        freq='D')
    
    storage = 0
    cash = []
    storageMonths = set()

    for date in dates:
        price = priceModel(date)
        flow = 0

        if date in injectionDates:
            injected = min(injRate, maxStore - storage)
            storage += injected
            flow -= price * injected  
        elif date in withdrawalDates:
            withdrawn = min(withDrawRate, storage)
            storage -= withdrawn
            flow += price * withdrawn 
        
        if storage > 0:
            storageMonths.add((date.year, date.month))
        
        cash.append(flow)
    
    totalStorageCost = len(storageMonths) * storeCost
    return sum(cash) - totalStorageCost


val = ContractCalculation(injectionDates=pd.date_range("2024-11-01", periods=5, freq='D'),
    withdrawalDates=pd.date_range("2025-02-01", periods=5, freq='D'),
    injRate=1000,
    withDrawRate=1000,
    storeCost=100,
    maxStore=50000000,
    priceModel=estimatePrice)

print(f"Estimated contract value: ${val:.2f}")