import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

def process_wifi_data(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} not found.")
        return

    # Load data
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    if df.empty:
        print("CSV is empty.")
        return

    # 1. Signal Strength Over Time
    plt.figure(figsize=(12, 6))
    for bssid in df['bssid'].unique():
        subset = df[df['bssid'] == bssid]
        plt.plot(subset['timestamp'], subset['signal'], label=f"{bssid}")
    
    plt.title("WiFi Signal Strength Over Time")
    plt.xlabel("Time")
    plt.ylabel("Signal Strength (%)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("signal_over_time.png")
    print("Saved: signal_over_time.png")
    plt.close()

    # 2. Basic "Heatmap" (Signal Strength Distribution)
    # Note: Since we don't have XY coordinates yet, this is a 
    # Signal-Intensity-by-BSSID heatmap (Pivot Table)
    plt.figure(figsize=(10, 8))
    pivot_df = df.pivot_table(index='bssid', columns='timestamp', values='signal')
    sns.heatmap(pivot_df, cmap="YlGnBu", annot=False)
    plt.title("Signal Strength Heatmap (BSSID vs Time)")
    plt.xlabel("Timestamp")
    plt.ylabel("BSSID")
    plt.tight_layout()
    plt.savefig("signal_heatmap.png")
    print("Saved: signal_heatmap.png")
    plt.close()

if __name__ == "__main__":
    csv_file = "wifi_scan_results.csv"
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    
    process_wifi_data(csv_file)
