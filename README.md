# 🛍️ Syntecxhub Customer Segmentation

A machine learning project that applies **K-Means clustering** to segment mall customers based on their age, annual income, and spending score — enabling data-driven marketing and business strategy.

---

## 📁 Project Structure

```
Syntecxhub_Customer_Segmentation/
│
├── Mall_Customers.csv          # Raw input dataset
├── customer_segments.csv       # Output dataset with cluster labels
├── customer_segmentation.py    # Main segmentation script
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 📊 Dataset

**Source:** `Mall_Customers.csv`

| Column | Description |
|---|---|
| `CustomerID` | Unique customer identifier (dropped during preprocessing) |
| `Genre` | Gender of the customer (`Male` / `Female`) |
| `Age` | Age of the customer |
| `Annual Income (k$)` | Annual income in thousands of dollars |
| `Spending Score (1-100)` | Mall-assigned spending score (1 = low, 100 = high) |

**Size:** 200 customers

---

## ⚙️ How It Works

### 1. Data Preprocessing
- Drops the `CustomerID` column (no predictive value)
- Encodes `Genre` as numeric (`Male → 0`, `Female → 1`)
- Selects features: `Age`, `Annual Income (k$)`, `Spending Score (1-100)`
- Scales features using `StandardScaler`

### 2. Finding Optimal Clusters
- Iterates over `k = 2` to `10`
- Evaluates each `k` using:
  - **Elbow Method** (WCSS / Inertia)
  - **Silhouette Score**
- Optimal `k = 5` selected based on both metrics

### 3. Final Model
- Trains `KMeans` with `k=5`, `init='k-means++'`, `random_state=42`
- Assigns cluster labels to each customer

### 4. Visualization
- **PCA scatter plot** — 2D projection of all clusters
- **Income vs Spending scatter plot** — direct feature view of segments

---

## 👥 Customer Segments

| Cluster | Segment Name | Traits |
|---|---|---|
| 0 | Standard Customers | Average income, average spending |
| 1 | High Value Customers | Low income, high spending |
| 2 | Careful Spenders | High income, low spending |
| 3 | Young Spenders | Young, high spending |
| 4 | Budget Customers | Low income, low spending |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/parthpetkar24/Syntecxhub_Customer_Segmentation.git
cd Syntecxhub_Customer_Segmentation

# Install dependencies
pip install -r requirements.txt
```

### Run the Script

```bash
python customer_segmentation.py
```

The script will:
1. Load and preprocess the dataset
2. Plot the Elbow and Silhouette graphs
3. Train the final K-Means model
4. Display cluster visualizations
5. Print a segment report to the console
6. Save results to `customer_segments.csv`

---

## 📦 Output

**`customer_segments.csv`** — the original dataset enriched with:
- `Cluster` — numeric cluster label (0–4)
- `PCA1`, `PCA2` — principal component coordinates
- `Segment Name` — human-readable segment label

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Plotting (Elbow, scatter charts) |
| `seaborn` | Enhanced visualizations |
| `scikit-learn` | Scaling, K-Means clustering, PCA, silhouette score |

---

