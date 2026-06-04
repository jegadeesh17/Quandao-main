from IPython.display import display
import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
sns.set_theme(style='darkgrid', palette='viridis')
plt.rcParams.update({
    'figure.figsize': (16, 6),
    'font.size': 12,
    'axes.titlesize': 15,
    'axes.labelsize': 13,
})

# ── Load .env ──────────────────────────────────────────────────────────
_this_dir     = os.path.abspath('')
_project_root = os.path.dirname(os.path.dirname(_this_dir))  # adjust if needed

# Try multiple possible .env locations
for candidate in [
    os.path.join(_project_root, '.env'),
    os.path.join(os.path.dirname(_this_dir), '.env'),
    os.path.join(_this_dir, '..', '..', '.env'),
    os.path.join(_this_dir, '..', '.env'),
    '.env',
]:
    if os.path.exists(candidate):
        load_dotenv(candidate)
        print(f'✅ Loaded .env from: {candidate}')
        break

DB_URL = os.getenv('DATABASE_URL')
assert DB_URL, '❌ DATABASE_URL not found in environment!'

engine = create_engine(DB_URL)
print(f'✅ Connected to DB: {DB_URL.split("@")[1] if "@" in DB_URL else DB_URL}')
SYMBOL = 'NSE:NIFTY50-INDEX'

# ── Fetch all 1-min candles ────────────────────────────────────────────
sql = text("""
    SELECT time, open, high, low, close, volume
    FROM   market_candles
    WHERE  symbol     = :symbol
    AND    resolution = '1'
    ORDER  BY time
""")

df_1m = pd.read_sql(sql, engine, params={'symbol': SYMBOL})
print(f'📥 Fetched {len(df_1m):,} one-minute candles')

# ── Ensure timezone-aware datetime ─────────────────────────────────────
df_1m['time'] = pd.to_datetime(df_1m['time'], utc=True)
df_1m['time_ist'] = df_1m['time'].dt.tz_convert('Asia/Kolkata')

# ── Filter to NSE session (09:15 – 15:30 IST) ─────────────────────────
session_mask = (
    (df_1m['time_ist'].dt.time >= pd.Timestamp('09:15').time()) &
    (df_1m['time_ist'].dt.time <= pd.Timestamp('15:30').time())
)
df_1m = df_1m[session_mask].copy()
print(f'🕘 After session filter: {len(df_1m):,} bars  |  '
      f'{df_1m["time_ist"].dt.date.nunique()} trading days')

# ── Resample to 15-min candles ─────────────────────────────────────────
df_1m = df_1m.set_index('time_ist').sort_index()

df_15m = df_1m.resample('15min', label='left', closed='left').agg({
    'open':   'first',
    'high':   'max',
    'low':    'min',
    'close':  'last',
    'volume': 'sum',
}).dropna(subset=['open']).copy()

# Re-apply session filter on the 15-min bars
df_15m = df_15m[
    (df_15m.index.time >= pd.Timestamp('09:15').time()) &
    (df_15m.index.time < pd.Timestamp('15:30').time())
].copy()

# ── Derived columns ───────────────────────────────────────────────────
df_15m['candle_size']  = df_15m['high'] - df_15m['low']          # range in points
df_15m['body_size']    = (df_15m['close'] - df_15m['open']).abs() # absolute body
df_15m['direction']    = np.where(df_15m['close'] >= df_15m['open'], 'Bullish', 'Bearish')
df_15m['date']         = df_15m.index.date
df_15m['day_name']     = df_15m.index.day_name()
df_15m['time_slot']    = df_15m.index.strftime('%H:%M')

print(f'\n✅ Resampled to {len(df_15m):,} fifteen-minute candles')
print(f'   Date range: {df_15m.index.min():%Y-%m-%d} → {df_15m.index.max():%Y-%m-%d}')
df_15m.head()
cs = df_15m['candle_size']

statistics = {
    'Count':              len(cs),
    'Mean (pts)':         cs.mean(),
    'Median (pts)':       cs.median(),
    'Variance':           cs.var(),
    'Std Deviation':      cs.std(),
    'Skewness':           cs.skew(),
    'Kurtosis':           cs.kurtosis(),
    'Min (pts)':          cs.min(),
    'Max (pts)':          cs.max(),
    'IQR':                cs.quantile(0.75) - cs.quantile(0.25),
    'P25':                cs.quantile(0.25),
    'P50 (Median)':       cs.quantile(0.50),
    'P75':                cs.quantile(0.75),
    'P90':                cs.quantile(0.90),
    'P95':                cs.quantile(0.95),
    'P99':                cs.quantile(0.99),
}

stats_df = pd.DataFrame.from_dict(statistics, orient='index', columns=['Value'])
stats_df['Value'] = stats_df['Value'].map(lambda x: f'{x:,.4f}' if isinstance(x, float) else x)

print('=' * 50)
print(' NIFTY50 15-min Candle Size — Descriptive Stats')
print('=' * 50)
display(stats_df)
fig, ax = plt.subplots(figsize=(18, 6))

colors = np.where(df_15m['direction'] == 'Bullish', '#26a69a', '#ef5350')
ax.scatter(df_15m.index, df_15m['candle_size'], c=colors, alpha=0.35, s=4, edgecolors='none')

# Rolling 200-candle mean overlay
rolling_mean = df_15m['candle_size'].rolling(200, min_periods=50).mean()
ax.plot(df_15m.index, rolling_mean, color='#ff9800', linewidth=2, label='200-bar Rolling Mean')

ax.axhline(cs.mean(), color='white', linestyle='--', linewidth=1, alpha=0.7, label=f'Overall Mean = {cs.mean():.1f} pts')
ax.set_title('NIFTY50 — 15-Min Candle Size (High − Low) Over Time', fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Candle Size (points)')
ax.legend(loc='upper left')
plt.tight_layout()
plt.draw()
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
day_stats = df_15m.groupby('day_name')['candle_size'].agg(['mean', 'median', 'std', 'count'])
day_stats = day_stats.reindex(day_order)

print('\n📅 Average 15-min candle size by day of week:\n')
display(day_stats.round(2))

# ── Bar chart ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

palette = ['#1e88e5', '#43a047', '#fb8c00', '#8e24aa', '#e53935']

# Mean candle size
axes[0].bar(day_stats.index, day_stats['mean'], color=palette, edgecolor='white', linewidth=0.5)
axes[0].set_title('Mean 15-Min Candle Size by Day', fontweight='bold')
axes[0].set_ylabel('Points')
for i, v in enumerate(day_stats['mean']):
    axes[0].text(i, v + 0.3, f'{v:.1f}', ha='center', fontweight='bold', fontsize=11)

# Count of candles
axes[1].bar(day_stats.index, day_stats['count'], color=palette, edgecolor='white', linewidth=0.5)
axes[1].set_title('Number of 15-Min Candles by Day', fontweight='bold')
axes[1].set_ylabel('Count')
for i, v in enumerate(day_stats['count']):
    axes[1].text(i, v + 50, f'{int(v):,}', ha='center', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.draw()
pivot = df_15m.pivot_table(
    values='candle_size', index='time_slot', columns='day_name', aggfunc='mean'
).reindex(columns=day_order)

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(pivot, cmap='YlOrRd', annot=True, fmt='.1f', linewidths=0.5,
            cbar_kws={'label': 'Avg Candle Size (pts)'}, ax=ax)
ax.set_title('NIFTY50 — Avg 15-Min Candle Size: Time Slot × Day of Week', fontweight='bold')
ax.set_xlabel('Day of Week')
ax.set_ylabel('Time Slot (IST)')
plt.tight_layout()
plt.draw()
# ── Define bins (auto-adjusted to data range) ─────────────────────────
max_size = cs.max()
bins = [0, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500]
if max_size > 500:
    bins.append(max_size + 1)
labels = [f'{bins[i]}–{bins[i+1]}' for i in range(len(bins)-1)]

df_15m['size_bin'] = pd.cut(df_15m['candle_size'], bins=bins, labels=labels, right=False)

bin_counts = df_15m['size_bin'].value_counts().reindex(labels).fillna(0).astype(int)
bin_pct    = (bin_counts / len(df_15m) * 100).round(2)

dist_df = pd.DataFrame({'Count': bin_counts, 'Percentage (%)': bin_pct})
print('📊 Distribution of 15-min candle sizes:\n')
display(dist_df)

# ── Plot ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Bar chart of bin counts
colors_bar = sns.color_palette('viridis', len(labels))
axes[0].bar(labels, bin_counts.values, color=colors_bar, edgecolor='white')
axes[0].set_title('Candle Size Distribution — Bin Counts', fontweight='bold')
axes[0].set_xlabel('Candle Size Range (pts)')
axes[0].set_ylabel('Number of Candles')
axes[0].tick_params(axis='x', rotation=45)
for i, v in enumerate(bin_counts.values):
    if v > 0:
        axes[0].text(i, v + max(bin_counts.values)*0.01, f'{int(v):,}', ha='center', fontsize=9, fontweight='bold')

# KDE + Histogram
axes[1].hist(cs, bins=80, density=True, alpha=0.5, color='#42a5f5', edgecolor='white', label='Histogram')
cs.plot.kde(ax=axes[1], color='#e53935', linewidth=2, label='KDE')
axes[1].axvline(cs.mean(), color='#ff9800', linestyle='--', linewidth=1.5, label=f'Mean = {cs.mean():.1f}')
axes[1].axvline(cs.median(), color='#66bb6a', linestyle='--', linewidth=1.5, label=f'Median = {cs.median():.1f}')
axes[1].set_title('Candle Size — Histogram + KDE', fontweight='bold')
axes[1].set_xlabel('Candle Size (pts)')
axes[1].set_ylabel('Density')
axes[1].legend()

plt.tight_layout()
plt.draw()
fig, ax = plt.subplots(figsize=(14, 6))
sns.boxplot(
    data=df_15m, x='day_name', y='candle_size',
    order=day_order, palette=palette, showfliers=False, ax=ax
)
sns.stripplot(
    data=df_15m, x='day_name', y='candle_size',
    order=day_order, color='black', alpha=0.05, size=2, jitter=True, ax=ax
)
ax.set_title('NIFTY50 — 15-Min Candle Size Distribution by Day', fontweight='bold')
ax.set_xlabel('Day of Week')
ax.set_ylabel('Candle Size (pts)')
plt.tight_layout()
plt.draw()
# ── Define extreme threshold (95th percentile) ─────────────────────────
threshold_95 = cs.quantile(0.95)
threshold_05 = cs.quantile(0.05)

df_15m['is_extreme_high'] = df_15m['candle_size'] >= threshold_95
df_15m['is_extreme_low']  = df_15m['candle_size'] <= threshold_05
df_15m['candle_group'] = np.select(
    [df_15m['is_extreme_high'], df_15m['is_extreme_low']],
    ['Extreme High (≥P95)', 'Extreme Low (≤P5)'],
    default='Normal'
)

print(f'🔺 Extreme HIGH threshold (P95): {threshold_95:.1f} pts')
print(f'🔻 Extreme LOW  threshold (P5) : {threshold_05:.1f} pts')
print(f'   Extreme high candles: {df_15m["is_extreme_high"].sum():,}')
print(f'   Extreme low  candles: {df_15m["is_extreme_low"].sum():,}')
print(f'   Normal candles:       {(~df_15m["is_extreme_high"] & ~df_15m["is_extreme_low"]).sum():,}')
extreme_high = df_15m[df_15m['is_extreme_high']].copy()

# Day distribution of extreme candles vs. overall
day_pct_overall = df_15m['day_name'].value_counts(normalize=True).reindex(day_order) * 100
day_pct_extreme = extreme_high['day_name'].value_counts(normalize=True).reindex(day_order) * 100

compare_day = pd.DataFrame({
    'Overall %': day_pct_overall.round(2),
    'Extreme High %': day_pct_extreme.round(2),
    'Over-represented': (day_pct_extreme / day_pct_overall).round(2)
})
print('📅 Day-of-week distribution — Extreme High vs Overall:\n')
display(compare_day)

# Time-slot distribution
slot_pct_overall = df_15m['time_slot'].value_counts(normalize=True).sort_index() * 100
slot_pct_extreme = extreme_high['time_slot'].value_counts(normalize=True).sort_index() * 100

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Day comparison
x_pos = np.arange(len(day_order))
w = 0.35
axes[0].bar(x_pos - w/2, day_pct_overall.values, w, label='Overall', color='#42a5f5', alpha=0.8)
axes[0].bar(x_pos + w/2, day_pct_extreme.values, w, label='Extreme High', color='#e53935', alpha=0.8)
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(day_order)
axes[0].set_title('Day Distribution: Extreme High vs Overall', fontweight='bold')
axes[0].set_ylabel('Percentage (%)')
axes[0].legend()

# Time slot comparison
common_slots = slot_pct_overall.index.intersection(slot_pct_extreme.index)
axes[1].plot(slot_pct_overall.loc[common_slots].values, label='Overall', color='#42a5f5', linewidth=2)
axes[1].plot(slot_pct_extreme.loc[common_slots].values, label='Extreme High', color='#e53935', linewidth=2)
axes[1].set_xticks(range(0, len(common_slots), max(1, len(common_slots)//10)))
axes[1].set_xticklabels(common_slots[::max(1, len(common_slots)//10)], rotation=45)
axes[1].set_title('Time Slot Distribution: Extreme High vs Overall', fontweight='bold')
axes[1].set_ylabel('Percentage (%)')
axes[1].legend()

plt.tight_layout()
plt.draw()
max_lag = 10
acf_values = [cs.autocorr(lag=i) for i in range(1, max_lag + 1)]

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(range(1, max_lag + 1), acf_values, color='#7e57c2', edgecolor='white')
ax.axhline(0, color='white', linewidth=0.5)
ax.axhline(1.96 / np.sqrt(len(cs)), color='#ef5350', linestyle='--', alpha=0.7, label='95% confidence')
ax.axhline(-1.96 / np.sqrt(len(cs)), color='#ef5350', linestyle='--', alpha=0.7)
ax.set_title('Autocorrelation of 15-Min Candle Size (Lag 1 → 10)', fontweight='bold')
ax.set_xlabel('Lag (number of 15-min candles)')
ax.set_ylabel('Autocorrelation')
ax.legend()

for i, v in enumerate(acf_values):
    ax.text(i + 1, v + 0.01, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.draw()

print('\n📈 Lag Autocorrelations:')
for i, v in enumerate(acf_values, 1):
    significance = '✅ Significant' if abs(v) > 1.96 / np.sqrt(len(cs)) else '  Not significant'
    print(f'   Lag {i:2d}: {v:+.4f}  {significance}')
# ── Compute next-candle stats after extreme highs ─────────────────────
df_15m_flat = df_15m.reset_index()
extreme_idx = df_15m_flat[df_15m_flat['is_extreme_high']].index

next_candle_data = []
for idx in extreme_idx:
    for lag in [1, 2, 3]:
        next_idx = idx + lag
        if next_idx < len(df_15m_flat):
            # Ensure same trading day (avoid overnight gap)
            if df_15m_flat.loc[idx, 'date'] == df_15m_flat.loc[next_idx, 'date']:
                next_candle_data.append({
                    'lag': lag,
                    'candle_size': df_15m_flat.loc[next_idx, 'candle_size'],
                    'direction': df_15m_flat.loc[next_idx, 'direction'],
                    'extreme_size': df_15m_flat.loc[idx, 'candle_size'],
                })

df_next = pd.DataFrame(next_candle_data)

if not df_next.empty:
    overall_mean = cs.mean()
    print('📊 Average candle size AFTER an extreme high candle:\n')
    for lag in [1, 2, 3]:
        subset = df_next[df_next['lag'] == lag]
        avg = subset['candle_size'].mean()
        pct_bull = (subset['direction'] == 'Bullish').mean() * 100
        print(f'   Next+{lag}: avg = {avg:.1f} pts '
              f'({"ABOVE" if avg > overall_mean else "BELOW"} overall mean of {overall_mean:.1f})  |  '
              f'Bullish {pct_bull:.1f}% / Bearish {100-pct_bull:.1f}%')

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    lag_means = df_next.groupby('lag')['candle_size'].mean()
    ax.bar(lag_means.index, lag_means.values, color=['#e53935', '#fb8c00', '#43a047'], edgecolor='white')
    ax.axhline(overall_mean, color='white', linestyle='--', linewidth=1.5, label=f'Overall Mean = {overall_mean:.1f}')
    ax.set_title('Average Candle Size After Extreme Candle (Lag 1–3)', fontweight='bold')
    ax.set_xlabel('Lag (candles after extreme)')
    ax.set_ylabel('Avg Candle Size (pts)')
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Next +1', 'Next +2', 'Next +3'])
    ax.legend()
    for i, (x, v) in enumerate(zip(lag_means.index, lag_means.values)):
        ax.text(x, v + 0.5, f'{v:.1f}', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.draw()
else:
    print('⚠️ Not enough data to compute next-candle statistics.')
# ── Build day × time_slot pivot ───────────────────────────────────────
pivot_corr = df_15m.pivot_table(
    values='candle_size', index='time_slot', columns='day_name', aggfunc='mean'
).reindex(columns=day_order).dropna()

day_corr = pivot_corr.corr()

fig, ax = plt.subplots(figsize=(8, 7))
mask = np.triu(np.ones_like(day_corr, dtype=bool), k=1)
sns.heatmap(day_corr, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.5, vmax=1.0,
            linewidths=1, square=True, ax=ax)
ax.set_title('Intraday Profile Similarity (Pearson Correlation)', fontweight='bold')
plt.tight_layout()
plt.draw()

# Find most and least similar day pairs
import itertools
pairs = list(itertools.combinations(day_order, 2))
pair_corrs = [(a, b, day_corr.loc[a, b]) for a, b in pairs]
pair_corrs.sort(key=lambda x: x[2], reverse=True)

print('\n🔗 Most similar day pairs (by intraday volatility profile):')
for a, b, c in pair_corrs[:3]:
    print(f'   {a} ↔ {b}: r = {c:.4f}')
print('\n🔀 Least similar day pairs:')
for a, b, c in pair_corrs[-3:]:
    print(f'   {a} ↔ {b}: r = {c:.4f}')
extreme_high = df_15m[df_15m['is_extreme_high']].copy()
extreme_high['body_pct'] = (extreme_high['body_size'] / extreme_high['candle_size'] * 100)

print('🔺 EXTREME HIGH CANDLE FINGERPRINT')
print('=' * 50)

# Direction bias
dir_dist = extreme_high['direction'].value_counts(normalize=True) * 100
print(f'\n Direction: {dir_dist.to_dict()}')

# Top days
top_days = extreme_high['day_name'].value_counts().head(3)
print(f'\n Top days:  {top_days.to_dict()}')

# Top time slots
top_slots = extreme_high['time_slot'].value_counts().head(5)
print(f'\n Top slots: {top_slots.to_dict()}')

# Body % stats
print(f'\n Body/Range ratio: mean={extreme_high["body_pct"].mean():.1f}%, '
      f'median={extreme_high["body_pct"].median():.1f}%')

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Direction pie
dir_dist.plot.pie(ax=axes[0], autopct='%1.1f%%', colors=['#26a69a', '#ef5350'],
                  startangle=90, textprops={'fontsize': 12})
axes[0].set_title('Direction of Extreme Candles', fontweight='bold')
axes[0].set_ylabel('')

# Top time slots bar
top_slots.plot.barh(ax=axes[1], color='#7e57c2', edgecolor='white')
axes[1].set_title('Top Time Slots for Extreme Candles', fontweight='bold')
axes[1].set_xlabel('Count')
axes[1].invert_yaxis()

# Body % histogram
axes[2].hist(extreme_high['body_pct'], bins=20, color='#ff7043', edgecolor='white', alpha=0.8)
axes[2].set_title('Body-to-Range Ratio of Extreme Candles', fontweight='bold')
axes[2].set_xlabel('Body / Range (%)')
axes[2].set_ylabel('Count')

plt.tight_layout()
plt.draw()
print('=' * 60)
print(' NIFTY50 15-MINUTE CANDLE — SUMMARY REPORT')
print('=' * 60)

print(f'\n📅 Data range:     {df_15m.index.min():%Y-%m-%d} → {df_15m.index.max():%Y-%m-%d}')
print(f'   Total candles:  {len(df_15m):,}')
print(f'   Trading days:   {df_15m["date"].nunique():,}')

print(f'\n📏 Candle Size (High − Low):')
print(f'   Mean:     {cs.mean():.2f} pts')
print(f'   Median:   {cs.median():.2f} pts')
print(f'   Std Dev:  {cs.std():.2f} pts')
print(f'   Skewness: {cs.skew():.2f}  ({"> 0 → right-skewed, more extreme UP moves" if cs.skew() > 0 else "< 0 → left-skewed"})')

print(f'\n📊 Day Rankings (by avg candle size):')
day_rank = day_stats['mean'].sort_values(ascending=False)
for rank, (day, val) in enumerate(day_rank.items(), 1):
    print(f'   {rank}. {day:12s} — {val:.2f} pts')

print(f'\n🔺 Extreme candle concentration:')
if not extreme_high.empty:
    top_day_extreme = extreme_high['day_name'].value_counts().idxmax()
    top_slot_extreme = extreme_high['time_slot'].value_counts().idxmax()
    print(f'   Most common day:  {top_day_extreme}')
    print(f'   Most common slot: {top_slot_extreme} IST')

print(f'\n🔗 Autocorrelation (lag-1): {acf_values[0]:.4f}')
if acf_values[0] > 0.1:
    print('   → Volatility clustering detected: a large candle tends to be followed by another large candle.')
    print('   → TRADING INSIGHT: After an extreme candle, expect continued high volatility in the next 1-2 bars.')
else:
    print('   → Weak autocorrelation: candle sizes are relatively independent.')

print(f'\n💡 TRADING INSIGHTS:')
print(f'   1. Focus on the highest-volatility day ({day_rank.index[0]}) for breakout trades.')
print(f'   2. Avoid low-volatility slots where candle size < median ({cs.median():.0f} pts).')
if not extreme_high.empty:
    print(f'   3. Extreme candles cluster at {top_slot_extreme} IST on {top_day_extreme}s — prime reversal/breakout window.')
print(f'   4. The distribution is {"right-skewed" if cs.skew() > 0 else "left-skewed"} → '
      f'occasional large moves are {"more common" if cs.skew() > 0 else "less common"} than a normal distribution.')

print('\n' + '=' * 60)
print(' END OF ANALYSIS')
print('=' * 60)