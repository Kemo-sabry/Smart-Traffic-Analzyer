import pandas as pd
import matplotlib.pyplot as plt

# Load Data
try:
    df = pd.read_csv("traffic_data.csv")
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: 'traffic_data.csv' not found. Run main.py first.")
    exit()

# Setup Plot Style
plt.style.use('ggplot')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
fig.suptitle('Smart Traffic Analysis Report', fontsize=16)

# Plot 1: Vehicle Count over Time (Frames)
ax1.plot(df['Frame'], df['VehicleCount'], color='tab:blue', linewidth=2)
ax1.set_title('Traffic Density Over Time')
ax1.set_xlabel('Frame Number')
ax1.set_ylabel('Number of Vehicles')
ax1.grid(True)

# Plot 2: Traffic Status Distribution (Pie Chart)
status_counts = df['Status'].value_counts()
colors = {'LOW': '#2ecc71', 'MEDIUM': '#f1c40f', 'HIGH': '#e74c3c'}
# Map colors to the labels present in data
pie_colors = [colors.get(x, 'gray') for x in status_counts.index]

ax2.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', colors=pie_colors, startangle=90)
ax2.set_title('Traffic Condition Distribution')

# Save Report
plt.tight_layout()
plt.savefig('traffic_report.png')
print("Report saved as 'traffic_report.png'. Check your folder!")

plt.show()
