import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# 1. Load the public Iris dataset
iris = load_iris(as_frame=True)
df = iris.frame.copy()

df.columns = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "target"
]

df["class"] = df["target"].map(dict(enumerate(iris.target_names)))
df = df.drop(columns=["target"])

print("First 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

# 2. Data cleaning
clean = df.copy()

clean["class"] = clean["class"].astype(str).str.strip().str.lower()

numeric_cols = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
]

for col in numeric_cols:
    clean[col] = pd.to_numeric(clean[col], errors="coerce")

# Handle missing values if present
for col in numeric_cols:
    if clean[col].isna().any():
        clean[col] = clean[col].fillna(clean[col].median())

if clean["class"].isna().any():
    clean["class"] = clean["class"].fillna(clean["class"].mode()[0])

# Remove duplicates
clean = clean.drop_duplicates().reset_index(drop=True)

# Save cleaned data
clean.to_csv("iris_cleaned.csv", index=False)

# 3. Exploratory data analysis
print("\nDescriptive statistics:")
print(clean[numeric_cols].describe())

print("\nClass counts:")
print(clean["class"].value_counts())

print("\nClass-wise means:")
print(clean.groupby("class")[numeric_cols].mean())

print("\nCorrelation matrix:")
print(clean[numeric_cols].corr())

# 4. Visualization 1: Class distribution
clean["class"].value_counts().sort_index().plot(kind="bar")
plt.title("Iris Samples by Class")
plt.xlabel("Iris class")
plt.ylabel("Number of samples")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 5. Visualization 2: Petal length box plot
groups = [
    clean.loc[clean["class"] == cls, "petal_length"]
    for cls in ["setosa", "versicolor", "virginica"]
]

plt.boxplot(groups, labels=["setosa", "versicolor", "virginica"])
plt.title("Petal Length Distribution by Iris Class")
plt.xlabel("Iris class")
plt.ylabel("Petal length (cm)")
plt.tight_layout()
plt.show()

# 6. Visualization 3: Petal length vs petal width
for cls in ["setosa", "versicolor", "virginica"]:
    subset = clean[clean["class"] == cls]
    plt.scatter(
        subset["petal_length"],
        subset["petal_width"],
        label=cls,
        alpha=0.75
    )

plt.title("Petal Length vs Petal Width")
plt.xlabel("Petal length (cm)")
plt.ylabel("Petal width (cm)")
plt.legend(title="Class")
plt.tight_layout()
plt.show()

# 7. Visualization 4: Correlation heatmap
corr = clean[numeric_cols].corr()

plt.imshow(corr.values, aspect="auto")
plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=35, ha="right")
plt.yticks(range(len(numeric_cols)), numeric_cols)
plt.title("Correlation Matrix of Numeric Features")

for i in range(len(numeric_cols)):
    for j in range(len(numeric_cols)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}",
                 ha="center", va="center")

plt.colorbar()
plt.tight_layout()
plt.show()
