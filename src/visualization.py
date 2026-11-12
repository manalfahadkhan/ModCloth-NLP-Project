import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_json(
    "data/raw/modcloth_final_data.json",
    lines=True
)

df["fit"].value_counts().plot(kind="bar")

plt.title("Clothing Fit Distribution")
plt.xlabel("Fit Category")
plt.ylabel("Number of Reviews")

plt.tight_layout()

plt.savefig("data/processed/fit_distribution.png")

plt.show()


df["quality"].value_counts().sort_index().plot(kind="bar")

plt.title("Product Quality Distribution")
plt.xlabel("Quality Rating")
plt.ylabel("Number of Reviews")

plt.tight_layout()

plt.savefig("data/processed/quality_distribution.png")

plt.show()