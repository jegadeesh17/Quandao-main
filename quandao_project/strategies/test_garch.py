import pandas as pd
import numpy as np
from arch import arch_model
import scipy.stats as stats
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Find the project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath('')))
load_dotenv(os.path.join(project_root, '.env'))

DB_URL = os.getenv('DATABASE_URL')
engine = create_engine(DB_URL)

SYMBOL    = 'NSE:NIFTY50-INDEX'
FROM_DATE = '2022-01-01 00:00:00+00:00'
TO_DATE   = '2036-04-06 00:00:00+00:00'

sql = text("""
    SELECT time, open, high, low, close, volume
    FROM   market_candles
    WHERE  symbol     = :symbol
    AND    resolution = '1'
    AND    time >= :from_date
    AND    time <  :to_date
    ORDER  BY time
""")

df = pd.read_sql(sql, engine, params={'symbol': SYMBOL, 'from_date': FROM_DATE, 'to_date': TO_DATE})
df['time'] = pd.to_datetime(df['time'], utc=True)
df = df.sort_values('time').reset_index(drop=True)

df_daily = (
    df.set_index('time')
      .resample('1D')
      .agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
      .dropna()
)

df_daily['returns'] = 100 * np.log(df_daily['close'] / df_daily['close'].shift(1))
df_daily = df_daily.dropna()

print(f"Total days: {len(df_daily)}")

model = arch_model(
    df_daily['returns'],
    mean='Constant',
    vol='EGARCH',
    p=1,
    o=1,             
    q=1,
    dist='t'
)
result = model.fit(disp='off')
print(result.summary())

df_daily['cond_vol'] = result.conditional_volatility
nu_param = result.params['nu']
T_MULTIPLIER = stats.t.ppf(0.95, df=nu_param)
df_daily['VaR_95'] = df_daily['cond_vol'] * T_MULTIPLIER
df_daily['VaR_Breach'] = np.abs(df_daily['returns']) > df_daily['VaR_95']

breach_pct = (df_daily['VaR_Breach'].sum() / len(df_daily)) * 100
print(f"EGARCH Breach rate: {breach_pct:.2f}%")

model_garch = arch_model(
    df_daily['returns'],
    mean='Constant',
    vol='GARCH',
    p=1,            
    q=1,
    dist='t'
)
result_garch = model_garch.fit(disp='off')
df_daily['cond_vol_garch'] = result_garch.conditional_volatility
nu_param_garch = result_garch.params['nu']
T_MULTIPLIER_garch = stats.t.ppf(0.95, df=nu_param_garch)
df_daily['VaR_95_garch'] = df_daily['cond_vol_garch'] * T_MULTIPLIER_garch
df_daily['VaR_Breach_garch'] = np.abs(df_daily['returns']) > df_daily['VaR_95_garch']

breach_pct_garch = (df_daily['VaR_Breach_garch'].sum() / len(df_daily)) * 100
print(f"GARCH Breach rate: {breach_pct_garch:.2f}%")

model_garch_vix = arch_model(
    df_daily['returns'],
    mean='Constant',
    vol='GARCH',
    p=1,            
    q=1,
    dist='t'
)
