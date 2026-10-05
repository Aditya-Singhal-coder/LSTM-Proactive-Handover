import matplotlib.pyplot as plt


# -----------------------------
# Final results
# -----------------------------

metrics = [
    "Total Handovers",
    "Ping-Pong Handovers",
    "Outage Samples"
]

reactive = [
    257,
    41,
    2384
]

proactive = [
    121,
    0,
    2302
]


# -----------------------------
# Comparison graph
# -----------------------------

x = range(len(metrics))

width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(
    [i - width / 2 for i in x],
    reactive,
    width,
    label="Reactive"
)

plt.bar(
    [i + width / 2 for i in x],
    proactive,
    width,
    label="Proactive"
)

plt.xticks(
    x,
    metrics
)

plt.ylabel("Count")

plt.title(
    "Reactive vs Proactive Handover Performance"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "../data/final_handover_comparison.png",
    dpi=300
)

plt.show()


# -----------------------------
# Average RSSI comparison
# -----------------------------

methods = [
    "Reactive",
    "Proactive"
]

rssi = [
    -89.32,
    -89.15
]

plt.figure(figsize=(8, 6))

plt.bar(
    methods,
    rssi
)

plt.ylabel("Average Serving RSSI (dBm)")

plt.title(
    "Average Serving RSSI Comparison"
)

plt.tight_layout()

plt.savefig(
    "../data/final_rssi_comparison.png",
    dpi=300
)

plt.show()