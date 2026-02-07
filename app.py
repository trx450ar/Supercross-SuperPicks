import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pulp import LpMinimize, LpProblem, LpVariable, lpSum, value

# === PALETTE (minimal bronze accents, no slider overrides) ===
BLACK_BG = "#000000"
CHARCOAL_DEEP = "#1A1A1A"
TITANIUM_DARK = "#2E2E2E"
BURNT_BRONZE = "#8C5523"
BRONZE_LIGHT = "#A67C52"
HONDA_RED_DARK = "#8B0000"
TEXT_LIGHT = "#D0D0D0"
STEALTH_GREY = "#3A3A3A"

st.set_page_config(page_title="Supercross SuperPicks", layout="wide")

# Custom CSS - subtle bronze + red accents, no slider color override
st.markdown(f"""
    <style>
        .stApp {{
            background-color: {BLACK_BG};
            color: {TEXT_LIGHT};
            font-family: 'Segoe UI', 'Roboto', sans-serif;
        }}
        h1, h2, h3 {{
            color: {BURNT_BRONZE};
            font-weight: 600;
            letter-spacing: 0.5px;
        }}
        h1 {{
            border-bottom: 1px solid {HONDA_RED_DARK};  /* Subtle red underline under title */
            padding-bottom: 8px;
        }}
        .stButton > button {{
            background-color: {CHARCOAL_DEEP};
            color: {TEXT_LIGHT};
            border: 1px solid {TITANIUM_DARK};
            border-radius: 4px;
            transition: all 0.2s;
        }}
        .stButton > button:hover {{
            background-color: {STEALTH_GREY};
            border-color: {HONDA_RED_DARK};
            box-shadow: 0 0 8px rgba(139, 0, 0, 0.15);
        }}
        .stSidebar {{
            background-color: {CHARCOAL_DEEP};
            border-right: 1px solid {TITANIUM_DARK};
        }}
        .stExpander {{
            background-color: {CHARCOAL_DEEP};
            border: 1px solid {TITANIUM_DARK};
            border-radius: 4px;
        }}
        hr {{
            border-color: {TITANIUM_DARK};
        }}
        footer {{
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background-color: #0A0A0A;
            color: #555555;
            text-align: center;
            padding: 8px 0;
            font-size: 0.8em;
            border-top: 1px solid {HONDA_RED_DARK};
        }}
        .stDivider {{
            background: linear-gradient(to right, transparent, {TITANIUM_DARK}, transparent);
            height: 1px;
        }}
        .stDataFrame {{
            border: 1px solid {BRONZE_LIGHT};
            border-radius: 4px;
        }}
    </style>
""", unsafe_allow_html=True)

st.title("Supercross SuperPicks")
st.markdown("Grok model • Built in thread with @JumpTruck1776")

# Sidebar - original first version you liked (clean, weights expander)
with st.sidebar:
    st.header("Live Inputs")
    wildcard_pos = st.number_input("Wildcard Position", value=14, step=1, help="Exact position wildcard is this week (from RMFantasySMX)")
    track_scale = st.slider("Track Conditions Scale", 1, 100, 80, help="1 = muddy/wet, 100 = dry/hard-pack (adjust based on preview)")

    with st.expander("Model Weights (Adjust if Needed)"):
        st.markdown("Higher weight = more influence on final score. Hover for explanation.")
        w_pos = st.slider("Recent Position Weight", 0.0, 0.5, 0.30, step=0.05, help="Higher = favors recent momentum. Ex: 0.4 drops inconsistent riders more.")
        w_points = st.slider("Season Points Weight", 0.0, 0.5, 0.30, step=0.05, help="Higher = rewards overall standings. Ex: 0.4 boosts Tomac's lead.")
        w_odds = st.slider("Betting Odds Weight", 0.0, 0.3, 0.15, step=0.05, help="Higher = trusts market favorites more.")
        w_qual = st.slider("Qual Time Weight", 0.0, 0.4, 0.20, step=0.05, help="Higher = rewards fast qualifiers.")
        w_var = st.slider("Variance Penalty Weight", 0.0, 0.3, 0.10, step=0.05, help="Higher = penalizes boom-or-bust riders.")
        w_beh = st.slider("Behavioral Score Weight", 0.0, 0.3, 0.10, step=0.05, help="Higher = rewards rider traits (smooth vs aggressive).")
        w_inj = st.slider("Injury Penalty Strength", 0.0, 1.0, 1.0, step=0.1, help="Higher = harsher drop for injury risks.")
        w_track = st.slider("Track Fit Weight", 0.0, 0.2, 0.05, step=0.01, help="Higher = more boost from track conditions match.")

# Main logic
if st.button("GO - Run Predictions"):
    with st.spinner("Running model..."):
        rider_names = ['Eli Tomac', 'Hunter Lawrence', 'Ken Roczen', 'Chase Sexton', 'Cooper Webb', 'Jason Anderson', 'Justin Cooper', 'Jorge Prado', 'Malcolm Stewart', 'Joey Savatgy', 'Dylan Ferrandis', 'Aaron Plessinger', 'Christian Craig', 'Vince Friese', 'Shane McElrath', 'Justin Hill', 'RJ Hampshire', 'Freddie Noren', 'Kyle Chisholm', 'Benny Bloss', 'Justin Starling', 'Cade Clason']
        qual_times = [52.181, 52.800, 52.000, 52.500, 53.100, 53.300, 53.500, 52.417, 53.700, 53.900, 54.000, 54.200, 54.300, 54.500, 54.600, 54.800, 55.000, 55.200, 55.400, 55.600, 55.800, 56.000]  # Placeholder

        riders = []
        for i, name in enumerate(rider_names):
            riders.append({
                'Rider': name,
                'Points': 88 - i*2,
                'Recent Positions': [i+1, i+2, i+3, i+4],
                'Qual Lap Time': qual_times[i],
                'Behavioral Score': 0.90 - i*0.05,
                'Injury Penalty': -0.15 if name == 'Malcolm Stewart' else 0.00
            })

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

        # Track Fit Multiplier
        df['Track Fit Multiplier'] = 1.0
        if track_scale >= 75:
            df.loc[df['Rider'].isin(['Eli Tomac', 'Ken Roczen', 'Chase Sexton', 'Cooper Webb']), 'Track Fit Multiplier'] = 1.05

        # Top Score with custom weights
        df['Top Score'] = (
            w_pos * df['Inverted Avg Pos'] +
            w_points * df['Points Score'] +
            w_odds * df['Odds Prob'] +
            w_qual * df['Inverted Qual'] +
            w_var * df['Variance Penalty'] +
            w_beh * df['Behavioral Score'] +
            w_inj * df['Injury Penalty'] +
            w_track * (df['Track Fit Multiplier'] - 1)
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
        df['DistTarget'] = abs(df['Avg Position'] - wildcard_pos)
        prob += lpSum([df['DistTarget'][i] * vars[i] + 0.5 * df['Position Variance'][i] * vars[i] + abs(df['Injury Penalty'][i]) * vars[i] for i in df.index])
        prob += lpSum([vars[i] for i in df.index]) == 1
        prob.solve()
        selected_index = None
        for i in df.index:
            if value(vars[i]) == 1:
                selected_index = i
                break
        selected_wildcard = df.loc[selected_index, 'Rider'] if selected_index is not None else "No valid selection"

        # Display
        st.subheader("Top 5 Predictions")
        st.dataframe(top_df.head(5)[['Rider', 'Top Score']].round(3))

        st.subheader("Monte Carlo % Chance (positions 1-10)")
        st.dataframe((position_probs * 100).round(1))

        st.subheader("Optimized Wildcard")
        st.success(f"Recommended for position {wildcard_pos}: {selected_wildcard}")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(top_df['Rider'].head(5), top_df['Top Score'].head(5), color=BURNT_BRONZE)
        ax.set_title('Top 5 Rider Scores', color=TEXT_LIGHT)
        ax.set_xlabel('Rider', color=TEXT_LIGHT)
        ax.set_ylabel('Top Score', color=TEXT_LIGHT)
        ax.tick_params(axis='x', rotation=45, colors=TEXT_LIGHT)
        ax.tick_params(axis='y', colors=TEXT_LIGHT)
        ax.set_facecolor(CHARCOAL_DEEP)
        fig.set_facecolor(BLACK_BG)
        fig.tight_layout()
        st.pyplot(fig)

st.markdown(
    """
    <footer>
        Built with precision • Grok + @JumpTruck1776 • Function over flash
    </footer>
    """,
    unsafe_allow_html=True
)
