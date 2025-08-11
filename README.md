# KMeans Electrofacies Clustering

🔬 **Advanced K-Means clustering analysis for electrofacies classification in well log data**

This project provides a comprehensive solution for performing K-Means clustering on petrophysical well log data to classify electrofacies and identify similar lithology/reservoir zones.

## 🎯 Features

### 🧠 Machine Learning Capabilities
- **K-Means Clustering**: Unsupervised learning for electrofacies classification
- **Automatic Cluster Optimization**: Elbow method and silhouette analysis
- **Feature Preprocessing**: Missing value handling and standardization
- **Data Quality Analysis**: Comprehensive null value and out-of-bounds detection
- **Model Persistence**: Save and load trained models
- **Memory-efficient sampling**: Automatic sampling for large datasets
- **Stratified sampling**: Maintains cluster balance in sampled data

### 📊 Visualization Suite
- **Interactive Log Curves**: Plotly-based depth vs log curves with cluster overlays
- **Crossplots**: RHOB vs NPHI, GR vs RDEP with cluster coloring
- **Depth Tracks**: Cluster vs depth visualization for zone identification
- **Optimization Plots**: Elbow curves and silhouette analysis

### 🌐 Web Application
- **Streamlit Interface**: User-friendly web application
- **File Upload**: Support for CSV files with flexible delimiters
- **Interactive Parameters**: Configurable clustering settings
- **Data Quality Analysis**: Comprehensive data validation with downloadable reports
- **Real-time Analysis**: Live clustering and visualization
- **Export Functionality**: Download clustered datasets
- **Enhanced Export Options**: Download annotated datasets with cluster assignments and quality scores

## 📋 Dataset Requirements

Your well log dataset should contain the following columns:

| Column | Description | Units |
|--------|-------------|-------|
| `DEPTH_MD` | Measured depth | meters |
| `GR` | Gamma Ray | API units |
| `RDEP` | Deep Resistivity | Ohm·m |
| `RHOB` | Bulk Density | g/cc |
| `NPHI` | Neutron Porosity | v/v |
| `PEF` | Photoelectric Factor | - |
| `DTC` | Compressional Sonic Transit Time | μs/ft |
| `CALI` | Caliper (borehole diameter) | inches |

## 🚀 Quick Start

### 1. Installation

```bash
# Navigate to the correct directory
cd KMeans_Electrofacies_Clustering\KMeansModel

# Install required packages
pip install -r requirements.txt
```

### 2. Run Standalone Analysis

```python
from kmeans_electrofacies import ElectrofaciesKMeans

# Initialize the clustering class
kmeans_analyzer = ElectrofaciesKMeans(n_clusters=5, random_state=42)

# Run complete analysis
cluster_summary = kmeans_analyzer.run_complete_analysis(
    file_path='train.csv',
    find_optimal=True,
    max_clusters=8
)
```

### 3. Launch Web Application


```bash
# Make sure you are in the correct directory:
cd KMeans_Electrofacies_Clustering\KMeansModel

# Run the Streamlit app
streamlit run streamlit_kmeans_app.py
```

Then open your browser to `http://localhost:8501`

## 📁 Project Structure

```
KMeansModel/
├── kmeans_electrofacies.py      # Main clustering class
├── streamlit_kmeans_app.py      # Web application
├── train.csv                    # Sample dataset
├── requirements.txt             # Dependencies
├── README.md                    # This file
└── Generated Files:
    ├── clustered_data.csv       # Results with cluster labels
    ├── log_curves_clusters.html # Interactive log plots
    ├── crossplots_clusters.html # Interactive crossplots
    ├── cluster_depth_track.html # Depth track visualization
    ├── cluster_optimization.png # Optimization analysis
    ├── kmeans_model.joblib      # Trained model
    └── scaler.joblib           # Feature scaler
```

## 🔧 Usage Examples

### Basic Clustering

```python
from kmeans_electrofacies import ElectrofaciesKMeans

# Initialize with specific number of clusters
analyzer = ElectrofaciesKMeans(n_clusters=4)

# Load and preprocess data
analyzer.load_data('your_data.csv', delimiter=';')
analyzer.preprocess_data()

# Fit clustering model
analyzer.fit_kmeans()

# Analyze results
cluster_summary = analyzer.analyze_clusters()

# Create visualizations
analyzer.plot_log_curves_with_clusters()
analyzer.plot_crossplots_with_clusters()
analyzer.plot_cluster_depth_track()

# Save results
analyzer.save_results('output_clustered.csv')
```

### Optimal Cluster Detection

```python
# Find optimal number of clusters
optimal_k, scores = analyzer.find_optimal_clusters(max_clusters=10)
print(f"Optimal clusters: {optimal_k}")

# Update and refit
analyzer.n_clusters = optimal_k
analyzer.fit_kmeans()
```

### Custom Feature Selection

```python
# Use specific features only
analyzer.feature_columns = ['GR', 'RHOB', 'NPHI', 'RDEP']
analyzer.preprocess_data()
analyzer.fit_kmeans()
```

## 📊 Output Files

### 1. Clustered Dataset (`clustered_data.csv`)
- Original data with added `Cluster` column
- Semicolon-delimited format
- Ready for further analysis
### 1a. Enhanced Clustered Dataset
- Includes cluster assignments, data quality annotations, and quality scores
- Downloadable from the Streamlit app

### 2. Interactive Visualizations
- **Log Curves**: Multi-track log display with cluster colors
- **Crossplots**: Scatter plots for petrophysical analysis
- **Depth Track**: Vertical cluster distribution

### 3. Model Files
- **kmeans_model.joblib**: Trained K-Means model
- **scaler.joblib**: Feature standardization scaler

## 🎨 Customization

### Modify Clustering Parameters

```python
analyzer = ElectrofaciesKMeans(
    n_clusters=6,           # Number of clusters
    random_state=123        # For reproducibility
)
```

### Add Custom Features

```python
# Extend feature list
analyzer.feature_columns.extend(['SP', 'ROP', 'MUDWEIGHT'])
```

### Custom Preprocessing

```python
# Override preprocessing method
def custom_preprocess(self):
    # Your custom preprocessing logic
    pass

analyzer.preprocess_data = custom_preprocess
```

## 🔍 Cluster Interpretation

### Typical Electrofacies Clusters:

1. **Cluster 0**: High GR, Low RDEP → **Shale/Clay**
2. **Cluster 1**: Low GR, High RDEP, Low NPHI → **Clean Sand**
3. **Cluster 2**: Medium GR, Medium RDEP → **Silty Sand**
4. **Cluster 3**: High RHOB, Low NPHI → **Tight Formation**
5. **Cluster 4**: Low RHOB, High NPHI → **Porous Formation**

## 🔍 Data Quality Analysis

The application now includes comprehensive data quality analysis to help identify and address data issues:

### **Features:**
- **Null Value Detection**: Identifies missing values in all well log parameters
- **Out-of-Bounds Analysis**: Detects values outside expected geological ranges
- **Quality Scoring**: Provides overall data quality assessment
- **Downloadable Reports**: Export problematic data for review and cleaning
- **Cluster Quality Annotation**: Annotates each row with data quality and cluster assignment
- **Technique Performance Tracking**: Logs imputation and scaling methods used

### **Expected Parameter Ranges:**
| Parameter | Range | Units | Description |
|-----------|-------|-------|-------------|
| GR | 0 - 300 | API | Gamma Ray |
| RDEP | 0.1 - 10,000 | Ohm·m | Deep Resistivity |
| RHOB | 1.0 - 3.5 | g/cc | Bulk Density |
| NPHI | -0.1 - 1.0 | v/v | Neutron Porosity |
| PEF | 0.5 - 10.0 | - | Photoelectric Factor |
| DTC | 40 - 300 | μs/ft | Sonic Transit Time |
| CALI | 4 - 20 | inches | Caliper |

### **Quality Recommendations:**
- **🔴 Critical Issues**: >50% missing values or >30% total issues
- **🟡 Moderate Issues**: 20-50% missing values or 10-30% total issues
- **✅ Good Quality**: <20% missing values and <10% total issues

### **Download Options:**
1. **Null Values Data**: CSV file containing all rows with missing values
2. **Out-of-Bounds Data**: CSV file containing rows with values outside expected ranges
3. **Quality Summary**: Detailed statistics and recommendations
4. **Enhanced Clustered Dataset**: CSV file with cluster assignments, quality annotations, and scores

## 🛠️ Troubleshooting

### Common Issues:

1. **CSV Delimiter Error**
   ```python
   # Try different delimiters
   analyzer.load_data('data.csv', delimiter=',')  # or '\t'
   ```

2. **Missing Columns**
   ```python
   # Check available columns
   print(data.columns.tolist())
   
   # Update feature list
   analyzer.feature_columns = [col for col in analyzer.feature_columns if col in data.columns]
   ```

3. **Memory Issues with Large Datasets**
```python
# Sample data for analysis
sampled_data = data.sample(n=100000, random_state=42)
# Or set max_samples in analyzer for automatic sampling
analyzer = ElectrofaciesKMeans(max_samples=100000)
```

## 📝 Recent Feature Additions

- Memory-efficient sampling and data type optimization
- Stratified sampling for balanced clusters
- Comprehensive quality annotation and scoring
- Technique performance tracking for imputation and scaling
- Enhanced export options for annotated clustered datasets
- Interactive and static visualizations for cluster and data quality

## 📈 Performance Tips

- **Data Sampling**: Use representative samples for large datasets (>1M rows)
- **Feature Selection**: Focus on most discriminative features
- **Cluster Range**: Test 3-8 clusters for most geological applications
- **Preprocessing**: Remove extreme outliers before clustering

## 🤝 Contributing

Feel free to contribute improvements:
1. Fork the repository
2. Create feature branch
3. Submit pull request

## 📄 License

This project is open source and available under the MIT License.

---

**Happy Clustering! 🎯**
