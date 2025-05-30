import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Solar Dashboard", layout="wide")

# Title
st.title("☀️ Solar Irradiance Dashboard")
st.markdown("Compare GHI across West African countries and view regional insights.")

# Sidebar country selection
countries = {
    "Ghana": r"C:\Users\soswo\OneDrive\Desktop\projects\solar-challenge-week1\data\processed\Ghana_clean.csv",
    "Benin": r"C:\Users\soswo\OneDrive\Desktop\projects\solar-challenge-week1\data\processed\Benin_clean.csv",
    "Liberia": r"C:\Users\soswo\OneDrive\Desktop\projects\solar-challenge-week1\data\processed\Liberia_clean.csv"
}

selected_countries = st.sidebar.multiselect(
    "Select Countries to Compare",
    options=list(countries.keys()),
    default=["Ghana", "Benin", "Liberia"]
)

# Load and combine selected data
dfs = []
for country in selected_countries:
    path = countries[country]
    df = pd.read_csv(path)
    df = df.iloc[1:]  # Drop header row if needed
    df['GHI'] = pd.to_numeric(df['GHI'], errors='coerce')
    df['Country'] = country
    dfs.append(df)

if not dfs:
    st.warning("Please select at least one country.")
    st.stop()

df_all = pd.concat(dfs)

# --- Boxplot of GHI ---
st.subheader("📦 GHI Distribution by Country")

fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=df_all, x="Country", y="GHI", ax=ax)
ax.set_ylabel("GHI (W/m²)")
ax.set_title("Boxplot of Global Horizontal Irradiance (GHI)")
st.pyplot(fig)

# --- Top Regions Table ---
st.subheader("📊 Top Regions by Average GHI")

# Normalize column names (strip spaces, standardize case)
# --- Top 10 Timestamps with Highest GHI ---
st.subheader("📍 Top 10 Timestamps with Highest GHI")

# Ensure GHI is numeric and Timestamp is parsed
df_all['GHI'] = pd.to_numeric(df_all['GHI'], errors='coerce')
df_all['Timestamp'] = pd.to_datetime(df_all['Timestamp'], errors='coerce')

# Drop NaN values in Timestamp and GHI
df_top_ghi = df_all.dropna(subset=['Timestamp', 'GHI'])

# Sort and select top 10
top_ghi_times = (
    df_top_ghi[['Timestamp', 'GHI']]
    .sort_values(by='GHI', ascending=False)
    .head(10)
)

st.dataframe(top_ghi_times)

# --- Footer ---
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Data: Ghana, Benin, Liberia")
