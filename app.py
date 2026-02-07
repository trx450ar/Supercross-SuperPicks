import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pulp import LpMinimize, LpProblem, LpVariable, lpSum, value

st.title("Supercross Predictor")
st.write("Grok model • Updated live in thread with @JumpTruck1776")

if st.button("GO - Run Predictions"):
    # === UPDATE THIS SECTION EVERY WEEK ===
    riders = [
        {'Rider': 'Eli Tomac', 'Points': 88, 'Recent Positions': [1, 3, 1, 5], 'Qual Lap Time': 52.181, 'Behavioral Score': 0.90, 'Injury Penalty': 0.00},
        {'Rider': 'Hunter Lawrence', 'Points': 84, 'Recent Positions': [2, 2, 2, 4], 'Qual Lap Time': 52.800, 'Behavioral Score': 0.85, 'Injury Penalty': 0.00},
        {'Rider': 'Ken Roczen', 'Points': 76, 'Recent Positions': [3, 1, 3, 3], 'Qual Lap Time': 52.000, 'Behavioral Score': 0.80, 'Injury Penalty': 0.00},
        {'Rider': 'Chase Sexton', 'Points': 74, 'Recent Positions': [8, 1, 2, 5], 'Qual Lap Time': 52.500, 'Behavioral Score': 0.75, 'Injury Penalty': 0.00},
        {'Rider': 'Cooper Webb', 'Points': 71, 'Recent Positions': [7, 4, 1, 4], 'Qual Lap Time': 53.100, 'Behavioral Score': 0.85, 'Injury Penalty': 0.00},
        {'Rider': 'Jason Anderson', 'Points': 62, 'Recent Positions': [5, 6, 7, 6], 'Qual Lap Time': 53.300, 'Behavioral Score': 0.70, 'Injury Penalty': 0.00},
        {'Rider': 'Justin Cooper', 'Points': 57, 'Recent Positions': [6, 9, 4, 9], 'Qual Lap Time': 53.500, 'Behavioral Score': 0.75, 'Injury Penalty': 0.00},
        {'Rider': 'Jorge Prado', 'Points': 55, 'Recent Positions': [3, 5, 8, 7], 'Qual Lap Time': 52.417, 'Behavioral Score': 0.80, 'Injury Penalty': 0.00},
        {'Rider': 'Malcolm Stewart', 'Points': 52, 'Recent Positions': [22, 8, 6, 8], 'Qual Lap Time': 53.700, 'Behavioral Score': 0.70, 'Injury Penalty': -0.15},
        {'Rider': 'Joey Savatgy', 'Points': 50, 'Recent Positions': [9, 10, 5, 10], 'Qual Lap Time': 53.900, 'Behavioral Score': 0.65, 'Injury Penalty': 0.00},
        {'Rider': 'Dylan Ferrandis', 'Points': 45, 'Recent Positions': [10, 11, 9, 11], 'Qual Lap Time': 54.000, 'Behavioral Score': 0.70, 'Injury Penalty': 0.00},
        {'Rider': 'Aaron Plessinger', 'Points': 40, 'Recent Positions': [11, 22, 10, 12], 'Qual Lap Time': 54.200, 'Behavioral Score': 0.65, 'Injury Penalty': 0.00},
        {'Rider': 'Christian Craig', 'Points': 38, 'Recent Positions': [12, 12, 11, 13], 'Qual Lap Time': 54.300, 'Behavioral Score': 0.60, 'Injury Penalty': 0.00},
        {'Rider': 'Vince Friese', 'Points': 35, 'Recent Positions': [14, 13, 12, 14], 'Qual Lap Time': 54.500, 'Behavioral Score': 0.70, 'Injury Penalty': 0.00},
        {'Rider': 'Shane McElrath', 'Points': 32, 'Recent Positions': [13, 14, 13, 15], 'Qual Lap Time': 54.600, 'Behavioral Score': 0.65, 'Injury Penalty': 0.00},
        {'Rider': 'Justin Hill', 'Points': 30, 'Recent Positions': [15, 15, 14, 16], 'Qual Lap Time': 54.800, 'Behavioral Score': 0.60, 'Injury Penalty': 0.00},
        {'Rider': 'RJ Hampshire', 'Points': 28, 'Recent Positions': [16, 16, 15, 17], 'Qual Lap Time': 55.000, 'Behavioral Score': 0.55, 'Injury Penalty': 0.00},
        {'Rider': 'Freddie Noren', 'Points': 25, 'Recent Positions': [17, 17, 16, 18], 'Qual Lap Time': 55.200, 'Behavioral Score': 0.50, 'Injury Penalty': 0.00},
        {'Rider': 'Kyle Chisholm', 'Points': 22, 'Recent Positions': [18, 18, 17, 19], 'Qual Lap Time': 55.400, 'Behavioral Score': 0.50, 'Injury Penalty': 0.00},
        {'Rider': 'Benny Bloss', 'Points': 20, 'Recent Positions': [19, 19, 18, 20], 'Qual Lap Time': 55.600, 'Behavioral Score': 0.50, 'Injury Penalty': 0.00},
        {'Rider': 'Justin Starling', 'Points': 18, 'Recent Positions': [20, 20, 19, 21], 'Qual Lap Time': 55.800, 'Behavioral Score': 0.45, 'Injury Penalty': 0.00},
        {'Rider': 'Cade Clason', 'Points': 16, 'Recent Positions': [21, 21, 20, 22], 'Qual Lap Time': 56.000, 'Behavioral Score': 0.45, 'Injury Penalty': 0.00}
    ]

    df = pd.DataFrame(riders)

    df['Avg Position'] = df['Recent Positions'].apply(np.mean)
    df['Position Variance'] = df['Recent Positions'].apply(np.var)

    max_points = df['Points'].max()
    df['Points Score'] = df['Points'] / max_points

    odds = {'Eli Tomac': 190, 'Chase Sexton': 198, 'Hunter Lawrence': 308, 'Ken Roczen': 477, 'Cooper Webb': 750}
    def odds_to_prob(o): return 100 / (o + 100) if o > 0 else 0
    df['Odds Prob'] = df['Rider'].apply(lambda r: odds_to_prob(odds.get(r, 1000)))

    min_qual = df['Qual Lap Time'].min()
    df['Inverted Qual'] = min_qual / df['Qual Lap Time']

    df['Inverted Avg Pos'] = 1 / (df['Avg Position'] + 1)
    df['Variance Penalty'] = 1 / (1 + df['Position Variance'])

    track_scale = 80
    df['Track Fit Multiplier'] = 1.0
    if track_scale >= 75:
        df.loc[df['Rider'].isin(['Eli Tomac', 'Ken Roczen', 'Chase Sexton', 'Cooper Webb']), 'Track Fit Multiplier'] = 1.05

    df['Top Score'] = (
        0.3 * df['Inverted Avg Pos'] +
        0.3 * df['Points Score'] +
        0.15 * df['Odds Prob'] +
        0.2 * df['Inverted Qual'] +
        0.1 * df['Variance Penalty'] +
        0.1 * df['Behavioral Score'] +
        df['Injury Penalty'] +
        0.05 * (df['Track Fit Multiplier'] - 1)
    )

    top_df = df.sort_values('Top Score', ascending=False).reset_index(drop=True)

    # Monte Carlo
    scores = top_df['Top Score'].values[:10]
    sims = []
    for _ in range(1000):
        noisy = scores + np.random.normal(0, 0.1, len(scores))
        sims.append(np.argsort(noisy)[::-1] + 1)
    prob_df = pd.DataFrame(sims, columns=top_df['Rider'][:10])
    position_probs = prob_df.apply(lambda c: c.value_counts(normalize=True).sort_index()).fillna(0)

    # Wildcard PuLP
    prob = LpProblem("Wildcard", LpMinimize)
    vars = LpVariable.dicts("sel", df.index, cat='Binary')
    df['Dist14'] = abs(df['Avg Position'] - 14)
    prob += lpSum([df['Dist14'][i] * vars[i] + 0.5 * df['Position Variance'][i] * vars[i] + abs(df['Injury Penalty'][i]) * vars[i] for i in df.index])
    prob += lpSum([vars[i] for i in df.index]) == 1
    prob.solve()
    selected = df.loc[[i for i in df.index if value(vars[i]) == 1][0], 'Rider']

    # Display
    st.subheader("Top 5")
    st.dataframe(top_df.head(5)[['Rider', 'Top Score']].round(3))

    st.subheader("Monte Carlo % Chance (positions 1-10)")
    st.dataframe((position_probs * 100).round(1))

    st.subheader("Wildcard (exact 14th)")
    st.success(selected)

    fig, ax = plt.subplots()
    ax.bar(top_df['Rider'].head(5), top_df['Top Score'].head(5))
    ax.set_title('Top 5 Scores')
    st.pyplot(fig)
