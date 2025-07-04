
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

os.makedirs("outputs/figures", exist_ok=True)
df = pd.read_csv("global_aqi_data.csv")
df["Date"] = pd.to_datetime(df[["Year", "Month"]].assign(DAY=1))

summary = df.groupby("City")["AQI"].describe()
summary.to_csv("outputs/aqi_summary_stats.csv")

plt.figure(figsize=(12, 6))
for city in df["City"].unique():
    city_data = df[df["City"] == city].groupby("Year")["AQI"].mean()
    plt.plot(city_data.index, city_data.values, marker='o', label=city)
plt.title("Average Yearly AQI by City")
plt.xlabel("Year")
plt.ylabel("Average AQI")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/figures/yearly_aqi_trend.png")
plt.close()

plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x="City", y="AQI")
plt.title("AQI Distribution by City")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/figures/aqi_boxplot_by_city.png")
plt.close()

pivot = df.pivot_table(values='AQI', index='Month', columns='City', aggfunc='mean')
corr = pivot.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation of Monthly AQI Between Cities")
plt.tight_layout()
plt.savefig("outputs/figures/aqi_correlation_heatmap.png")
plt.close()

delhi_aqi = df[df["City"] == "Delhi"]["AQI"]
paris_aqi = df[df["City"] == "Paris"]["AQI"]
t_stat, p_value = stats.ttest_ind(delhi_aqi, paris_aqi)

with open("outputs/results_summary.txt", "w") as f:
    f.write("T-test between Delhi and Paris AQI\n")
    f.write(f"T-statistic: {t_stat:.2f}, P-value: {p_value:.4f}\n")
    if p_value < 0.05:
        f.write("Result: Significant difference in AQI\n")
    else:
        f.write("Result: No significant difference in AQI\n")
