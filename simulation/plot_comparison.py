import matplotlib.pyplot as plt
import numpy as np


# Results from comparison
methods = ["Reactive", "Proactive"]

handovers = [257, 62]
ping_pong = [41, 0]
avg_rssi = [-89.32, -92.45]
outages = [2384, 3141]


# --------------------------------------------------
# 1. Total handovers
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(methods, handovers)

plt.ylabel("Number of Handovers")
plt.title("Reactive vs Proactive: Total Handovers")
plt.grid(axis="y")

plt.show()


# --------------------------------------------------
# 2. Ping-pong handovers
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(methods, ping_pong)

plt.ylabel("Ping-Pong Handovers")
plt.title("Reactive vs Proactive: Ping-Pong Handovers")
plt.grid(axis="y")

plt.show()


# --------------------------------------------------
# 3. Average serving RSSI
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(methods, avg_rssi)

plt.ylabel("Average RSSI (dBm)")
plt.title("Reactive vs Proactive: Average Serving RSSI")
plt.grid(axis="y")

plt.show()


# --------------------------------------------------
# 4. Outage samples
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(methods, outages)

plt.ylabel("Outage Samples")
plt.title("Reactive vs Proactive: Outage Samples")
plt.grid(axis="y")

plt.show()