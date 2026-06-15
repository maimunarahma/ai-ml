# 📊 K-Means Clustering Project (Customer Segmentation / Iris Dataset)

This project implements a complete **K-Means Clustering Machine Learning pipeline** using Python and Scikit-Learn. It includes data preprocessing, optimal cluster selection, model training, visualization, and real-world prediction using custom data.

---

## 🚀 Project Objective

To:
- Implement K-Means clustering from scratch using Scikit-Learn
- Find optimal number of clusters using Elbow Method
- Train clustering model on a standard dataset
- Test model using real-world custom data
- Visualize clusters and interpret results

---

## 📁 Dataset Used

### 🔹 Standard Dataset (Training)
You can use:
- Iris Dataset 🌸 OR
- Mall Customer Dataset 🛍️

Features used (example for Iris):
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

OR (for Customer Dataset):
- Age
- Annual Income
- Spending Score

---

### 🔹 Custom Dataset (Testing)
A manually collected dataset containing 10 real-world samples.

Example (Customer Data):
- Age
- Income
- Spending Score (1–100)

Stored in:


---

## ⚙️ Project Workflow

### 1️⃣ Data Loading
- Dataset is loaded directly from GitHub using raw URL
- No manual file upload required

---

### 2️⃣ Data Preprocessing
- Handling missing values
- Feature scaling using `StandardScaler`

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
3️⃣ Optimal K Selection (Elbow Method)
Tried different values of K (1–10)
Plotted WCSS (Within Cluster Sum of Squares)
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
4️⃣ Model Training

Final model trained using optimal K:

kmeans = KMeans(n_clusters=3)
kmeans.fit(X_scaled)
5️⃣ Cluster Prediction (Custom Data)

Custom dataset is scaled using SAME scaler and predicted:

custom_scaled = scaler.transform(custom_data)
predictions = kmeans.predict(custom_scaled)
custom_data["Cluster"] = predictions
📊 Visualizations
📌 Elbow Curve

Shows optimal number of clusters.

📌 Cluster Scatter Plot
2D visualization of clusters
Centroids marked with X
plt.scatter(X_scaled[:,0], X_scaled[:,1], c=clusters)
plt.scatter(centroids[:,0], centroids[:,1], marker='X', s=300)
📌 Custom Data Prediction Table

Displays real-world data with cluster assignments.

Age	Income	Spending Score	Cluster
21	18	80	0
45	70	30	1
🧠 Cluster Interpretation

Based on centroids:

Cluster 0 → High income, high spending customers (Premium group)
Cluster 1 → High income, low spending customers (Savers)
Cluster 2 → Low income, moderate spending customers
🛠️ Technologies Used
Python 🐍
Pandas
NumPy
Scikit-Learn
Matplotlib
📂 Project Structure
📁 dataset/
   └── custom_data.csv

📁 model/
   └── kmeans_model.pkl

📄 KMeans_Clustering.ipynb
📄 README.md
🔥 Key Features
Automatic dataset loading from GitHub
Elbow Method for optimal K selection
Fully trained K-Means clustering model
Real-world custom data prediction
Visualization of clusters and centroids
Cluster interpretation in human language
📌 How to Run
Clone repository:
git clone https://github.com/your-username/kmeans-project.git
Open notebook in Google Colab
Click:
Runtime → Run All

Everything runs automatically:

dataset loads
model trains
clusters generated
visualizations displayed
📈 Outcome

This project demonstrates:

Unsupervised learning (K-Means)
Real-world data clustering
Feature scaling importance
Cluster interpretation skills
