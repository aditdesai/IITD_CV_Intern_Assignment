from DINO.util.plot_utils import plot_logs
from pathlib import Path
import matplotlib.pyplot as plt


fig, axs = plot_logs(logs=Path(), fields=['loss'])
plt.show()