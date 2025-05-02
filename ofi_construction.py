import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

def OFI(df, level, symbol='AAPL'): 
    filtered_df = df[df['symbol'] == symbol]
    bid_size = filtered_df[f'bid_sz_0{level-1}'].to_numpy()
    ask_size = filtered_df[f'ask_sz_0{level-1}'].to_numpy()

    bid_price = filtered_df[f'bid_px_0{level-1}'].to_numpy()
    ask_price = filtered_df[f'ask_px_0{level-1}'].to_numpy()

    bid_price_diff = np.diff(bid_price, prepend=bid_price[0])[1:]
    ask_price_diff = np.diff(ask_price, prepend=ask_price[0])[1:]

    bid_size_diff = np.diff(bid_size, prepend=bid_size[0])[1:]
    ask_size_diff = np.diff(ask_size, prepend=ask_size[0])[1:]

    bid_size = bid_size[1:]
    ask_size = ask_size[1:]
    
    of_b = (bid_price_diff>0)*bid_size - (bid_price_diff < 0) * bid_size + (bid_price_diff==0) * bid_size_diff
    of_a = (ask_price_diff>0)*ask_size - (ask_price_diff < 0) * ask_size + (ask_price_diff==0) * ask_size_diff

    return of_b - of_a

def Q(df, level, symbol='AAPL'):
    filtered_df = df[df['symbol'] == symbol]
    res = np.zeros(len(filtered_df)-1)
    for i in range(level):
        res+= filtered_df[f'bid_sz_0{i}'].to_numpy()[1:] + filtered_df[f'ask_sz_0{i}'].to_numpy()[1:]
    return res/level/2

def ofi(df, level, symbol='AAPL'):
    return OFI(df, level, symbol)/Q(df, level, symbol)

def best_level_ofi(df, symbol='AAPL'):
    return OFI(df, 1, symbol)

def multi_level_ofi(df, symbol='AAPL'):
    res = []
    for i in range(1,11):
        res.append(ofi(df,i, symbol))
    return np.array(res).T

def integrated_ofi(df, symbol='AAPL'):
    multi_level_ofi_values = multi_level_ofi(df, symbol)
    pca = PCA(n_components=1)
    pca.fit(multi_level_ofi_values)

    w =  pca.components_[0]
    return w @ multi_level_ofi_values.T / np.linalg.norm(w, ord=1)

def logrithmic_return(df, symbol='AAPL'):
    filtered_df = df[df['symbol'] == symbol]
    bid_price = filtered_df[f'bid_px_00'].to_numpy()
    ask_price = filtered_df[f'ask_px_00'].to_numpy()
    midprice = (bid_price + ask_price) / 2
    log_return = np.log(midprice[1:] / midprice[:-1])
    return log_return

def regression(y, X):
    lasso = Lasso(alpha=0.1)
    lasso.fit(X, y)


    params = list(lasso.coef_)
    params.insert(0, lasso.intercept_)
    return params

def best_level_cross_asset_ofi(df, symbol='AAPL'):
    ofi_self = best_level_ofi(df, symbol)
    X = [ofi_self]

    symbols = set(df['symbol'])
    symbols.remove(symbol)
    for s in symbols:
        ofi_cross = best_level_ofi(df, s)
        X.append(ofi_cross)
    X = np.array(X).T
    
    y = logrithmic_return(df, symbol)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return regression(y, X_scaled)

def integrated_cross_asset_ofi(df, symbol='AAPL'):
    ofi_self = integrated_ofi(df, symbol)
    X = [ofi_self]

    symbols = set(df['symbol'])
    symbols.remove(symbol)
    for s in symbols:
        ofi_cross = integrated_ofi(df, s)
        X.append(ofi_cross)
    X = np.array(X).T
    
    y = logrithmic_return(df, symbol)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return regression(y, X_scaled)
