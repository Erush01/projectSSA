import matplotlib.pyplot as plt
from datetime import datetime

# Your data
data = [
    ('17:05:19.464000', 7.191),
    ('17:05:19.564000', 7.214),
    ('17:05:19.664000', 7.275),
    ('17:05:19.764000', 7.258)
]

# Extracting times and magnitudes
times = [d[0] for d in data]
magnitudes = [d[1] for d in data]

# Plotting
plt.plot(magnitudes, marker='o', linestyle='-')
plt.title('Mag Values Over Time')
plt.xlabel('Time')
plt.ylabel('Magnitude')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()