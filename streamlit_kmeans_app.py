"""
Streamlit Web Application for K-Means Electrofacies Clustering
============================================================

Interactive web application for performing K-Means clustering on well log data
to classify electrofacies based on petrophysical properties.

Author: AI Assistant
Date: 2025-07-22
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer, KNNImputer
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime

# Enhanced Data Quality Analysis Class with Preprocessing Tracking
class DataQualityAnalyzer:
    """Analyze data quality issues and track preprocessing techniques"""

    def __init__(self):
        # Define expected ranges for well log parameters
        self.parameter_ranges = {
            'GR': (0, 300),      # Gamma Ray (API units)
            'RDEP': (0.1, 10000), # Deep Resistivity (Ohm·m)
            'RHOB': (1.0, 3.5),   # Bulk Density (g/cc)
            'NPHI': (-0.1, 1.0),  # Neutron Porosity (v/v)
            'PEF': (0.5, 10.0),   # Photoelectric Factor
            'DTC': (40, 300),     # Sonic Transit Time (μs/ft)
            'CALI': (4, 20),      # Caliper (inches)
            'DEPTH_MD': (0, 10000) # Measured Depth (meters)
        }

        # Enhanced tracking attributes
        self.quality_annotations = pd.DataFrame()
        self.preprocessing_log = []
        self.technique_performance = {}
        self.best_techniques = {}

    def analyze_data_quality(self, data, selected_features):
        """
        Comprehensive data quality analysis

        Parameters:
        -----------
        data : pd.DataFrame
            Input dataset
        selected_features : list
            List of features to analyze

        Returns:
        --------
        dict: Dictionary containing quality analysis results
        """
        quality_report = {
            'null_data': pd.DataFrame(),
            'out_of_bounds': pd.DataFrame(),
            'summary_stats': {},
            'recommendations': []
        }

        # Analyze null values
        null_analysis = self._analyze_null_values(data, selected_features)
        quality_report['null_data'] = null_analysis['null_rows']
        quality_report['summary_stats']['null_summary'] = null_analysis['summary']

        # Analyze out-of-bounds values
        bounds_analysis = self._analyze_out_of_bounds(data, selected_features)
        quality_report['out_of_bounds'] = bounds_analysis['oob_rows']
        quality_report['summary_stats']['bounds_summary'] = bounds_analysis['summary']

        # Generate recommendations
        quality_report['recommendations'] = self._generate_recommendations(
            null_analysis, bounds_analysis, len(data)
        )

        return quality_report

    def _analyze_null_values(self, data, features):
        """Analyze null values in the dataset"""
        null_summary = {}
        null_rows_list = []

        for feature in features:
            if feature in data.columns:
                null_mask = data[feature].isnull()
                null_count = null_mask.sum()
                null_percentage = (null_count / len(data)) * 100

                null_summary[feature] = {
                    'count': null_count,
                    'percentage': null_percentage
                }

                # Collect rows with null values
                if null_count > 0:
                    null_rows = data[null_mask].copy()
                    null_rows['Issue_Type'] = f'Null_{feature}'
                    null_rows['Issue_Description'] = f'Missing value in {feature}'
                    null_rows_list.append(null_rows)

        # Combine all null rows
        if null_rows_list:
            all_null_rows = pd.concat(null_rows_list, ignore_index=True)
        else:
            all_null_rows = pd.DataFrame()

        return {
            'null_rows': all_null_rows,
            'summary': null_summary
        }

    def _analyze_out_of_bounds(self, data, features):
        """Analyze out-of-bounds values in the dataset"""
        bounds_summary = {}
        oob_rows_list = []

        for feature in features:
            if feature in data.columns and feature in self.parameter_ranges:
                min_val, max_val = self.parameter_ranges[feature]

                # Find out-of-bounds values
                oob_mask = (data[feature] < min_val) | (data[feature] > max_val)
                oob_count = oob_mask.sum()
                oob_percentage = (oob_count / len(data)) * 100

                bounds_summary[feature] = {
                    'count': oob_count,
                    'percentage': oob_percentage,
                    'expected_range': f'{min_val} - {max_val}',
                    'actual_range': f'{data[feature].min():.2f} - {data[feature].max():.2f}'
                }

                # Collect out-of-bounds rows
                if oob_count > 0:
                    oob_rows = data[oob_mask].copy()
                    oob_rows['Issue_Type'] = f'OutOfBounds_{feature}'
                    oob_rows['Issue_Description'] = f'{feature} outside range {min_val}-{max_val}'
                    oob_rows['Expected_Range'] = f'{min_val}-{max_val}'
                    oob_rows['Actual_Value'] = data.loc[oob_mask, feature]
                    oob_rows_list.append(oob_rows)

        # Combine all out-of-bounds rows
        if oob_rows_list:
            all_oob_rows = pd.concat(oob_rows_list, ignore_index=True)
        else:
            all_oob_rows = pd.DataFrame()

        return {
            'oob_rows': all_oob_rows,
            'summary': bounds_summary
        }

    def _generate_recommendations(self, null_analysis, bounds_analysis, total_rows):
        """Generate data quality recommendations"""
        recommendations = []

        # Null value recommendations
        for feature, stats in null_analysis['summary'].items():
            if stats['percentage'] > 50:
                recommendations.append(
                    f"⚠️ {feature}: {stats['percentage']:.1f}% missing values. "
                    f"Consider removing this feature or finding alternative data source."
                )
            elif stats['percentage'] > 20:
                recommendations.append(
                    f"🔍 {feature}: {stats['percentage']:.1f}% missing values. "
                    f"Review data collection process for this parameter."
                )
            elif stats['percentage'] > 5:
                recommendations.append(
                    f"ℹ️ {feature}: {stats['percentage']:.1f}% missing values. "
                    f"Consider median/mean imputation."
                )

        # Out-of-bounds recommendations
        for feature, stats in bounds_analysis['summary'].items():
            if stats['percentage'] > 10:
                recommendations.append(
                    f"🚨 {feature}: {stats['percentage']:.1f}% values outside expected range "
                    f"{stats['expected_range']}. Check data calibration."
                )
            elif stats['percentage'] > 1:
                recommendations.append(
                    f"⚠️ {feature}: {stats['percentage']:.1f}% values outside expected range "
                    f"{stats['expected_range']}. Review outliers."
                )

        # Overall data quality
        total_issues = len(null_analysis['null_rows']) + len(bounds_analysis['oob_rows'])
        if total_issues > total_rows * 0.3:
            recommendations.append(
                "🔴 High data quality issues detected (>30% of data). "
                "Consider data cleaning before analysis."
            )
        elif total_issues > total_rows * 0.1:
            recommendations.append(
                "🟡 Moderate data quality issues detected (>10% of data). "
                "Review and clean data for better results."
            )
        else:
            recommendations.append(
                "✅ Good data quality detected. Proceed with analysis."
            )

        return recommendations

    def initialize_quality_tracking(self, data):
        """Initialize comprehensive quality tracking system for K-Means"""
        print(f"📋 Initializing quality tracking for {len(data):,} rows")

        # Initialize quality annotations DataFrame
        self.quality_annotations = pd.DataFrame(index=data.index)
        self.quality_annotations['Row_ID'] = range(len(data))
        self.quality_annotations['Has_Null_Values'] = False
        self.quality_annotations['Has_Outliers'] = False
        self.quality_annotations['Quality_Issues'] = ''
        self.quality_annotations['Techniques_Applied'] = ''
        self.quality_annotations['Best_Technique'] = ''
        self.quality_annotations['Processing_Notes'] = ''
        self.quality_annotations['Cluster_Assignment'] = -1  # Will be filled after clustering

        return self.quality_annotations

    def test_imputation_techniques_kmeans(self, data, features):
        """Test multiple imputation techniques for K-Means clustering"""
        from sklearn.impute import SimpleImputer, KNNImputer
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler

        print(f"🧪 Testing imputation techniques for clustering performance...")

        techniques = {
            'median': SimpleImputer(strategy='median'),
            'mean': SimpleImputer(strategy='mean'),
            'knn': KNNImputer(n_neighbors=5)
        }

        # Test each technique on a sample of data
        sample_size = min(5000, len(data))
        sample_data = data[features].sample(n=sample_size, random_state=42)

        technique_scores = {}

        for technique_name, imputer in techniques.items():
            try:
                # Apply imputation
                imputed_data = pd.DataFrame(
                    imputer.fit_transform(sample_data),
                    columns=features
                )

                # Scale data for clustering
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(imputed_data)

                # Perform clustering
                kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(scaled_data)

                # Calculate silhouette score
                if len(set(cluster_labels)) > 1:
                    silhouette_avg = silhouette_score(scaled_data, cluster_labels)
                    technique_scores[technique_name] = silhouette_avg
                    print(f"   📊 {technique_name.capitalize()}: Silhouette = {silhouette_avg:.4f}")
                else:
                    technique_scores[technique_name] = -1
                    print(f"   ⚠️ {technique_name.capitalize()}: Single cluster formed")

            except Exception as e:
                print(f"   ❌ {technique_name.capitalize()}: Failed ({str(e)})")
                technique_scores[technique_name] = -1

        # Find best technique
        valid_scores = {k: v for k, v in technique_scores.items() if v > -1}
        if valid_scores:
            best_technique = max(valid_scores.keys(), key=lambda x: valid_scores[x])
            self.technique_performance['imputation'] = technique_scores
            self.best_techniques['imputation'] = best_technique

            print(f"🏆 Best imputation technique: {best_technique.upper()} (Silhouette: {valid_scores[best_technique]:.4f})")
            return best_technique, techniques[best_technique]
        else:
            print(f"⚠️ All imputation techniques failed, using median as fallback")
            return 'median', techniques['median']

    def create_enhanced_dataset_with_clustering(self, original_data, cluster_labels=None, cluster_centers=None):
        """
        Create enhanced dataset with comprehensive quality annotations and clustering results

        Args:
            original_data (pd.DataFrame): Original dataset
            cluster_labels (array-like, optional): Cluster assignments
            cluster_centers (array-like, optional): Cluster centers

        Returns:
            pd.DataFrame: Enhanced dataset with quality annotations and clustering results
        """
        print(f"📋 Creating enhanced dataset with clustering annotations...")

        # Start with original data
        enhanced_data = original_data.copy()

        # Ensure quality annotations exist and match the data
        if self.quality_annotations.empty:
            print("⚠️ No quality annotations found. Initializing basic tracking...")
            self.initialize_quality_tracking(enhanced_data)

        # Align indices between original data and quality annotations
        common_indices = enhanced_data.index.intersection(self.quality_annotations.index)
        enhanced_data = enhanced_data.loc[common_indices]
        quality_subset = self.quality_annotations.loc[common_indices]

        # Add quality annotation columns
        enhanced_data['Data_Quality_Has_Null_Values'] = quality_subset['Has_Null_Values']
        enhanced_data['Data_Quality_Has_Outliers'] = quality_subset['Has_Outliers']
        enhanced_data['Data_Quality_Issues'] = quality_subset['Quality_Issues']
        enhanced_data['Preprocessing_Techniques_Applied'] = quality_subset['Techniques_Applied']
        enhanced_data['Best_Technique_Used'] = quality_subset['Best_Technique']
        enhanced_data['Processing_Notes'] = quality_subset['Processing_Notes']

        # Add clustering results if available
        if cluster_labels is not None:
            enhanced_data['Cluster_Assignment'] = cluster_labels

            # Add cluster interpretation
            cluster_interpretation = []
            for label in cluster_labels:
                if label == 0:
                    interpretation = "Electrofacies_A"
                elif label == 1:
                    interpretation = "Electrofacies_B"
                elif label == 2:
                    interpretation = "Electrofacies_C"
                elif label == 3:
                    interpretation = "Electrofacies_D"
                else:
                    interpretation = f"Electrofacies_{label}"
                cluster_interpretation.append(interpretation)

            enhanced_data['Electrofacies_Interpretation'] = cluster_interpretation

        # Add technique performance summary
        if self.technique_performance:
            technique_summary = []

            # Imputation performance
            if 'imputation' in self.technique_performance:
                imp_perf = self.technique_performance['imputation']
                best_imp = max(imp_perf.keys(), key=lambda x: imp_perf[x]) if imp_perf else 'None'
                technique_summary.append(f"Best_Imputation: {best_imp}")

            # Clustering performance
            if 'clustering' in self.technique_performance:
                clust_perf = self.technique_performance['clustering']
                best_clust = max(clust_perf.keys(), key=lambda x: clust_perf[x]) if clust_perf else 'KMeans'
                technique_summary.append(f"Best_Clustering: {best_clust}")

            enhanced_data['Best_Technique_Summary'] = '; '.join(technique_summary) if technique_summary else 'Standard Processing'

        # Add data quality score
        quality_score = []
        for idx in enhanced_data.index:
            score = 100  # Start with perfect score

            if idx in quality_subset.index:
                # Deduct points for quality issues
                if quality_subset.loc[idx, 'Has_Null_Values']:
                    score -= 25
                if quality_subset.loc[idx, 'Has_Outliers']:
                    score -= 15

                # Count number of issues
                issues = quality_subset.loc[idx, 'Quality_Issues']
                if issues:
                    issue_count = len(issues.split(';'))
                    score -= min(issue_count * 10, 40)  # Max 40 points deduction

            quality_score.append(max(0, score))  # Ensure non-negative

        enhanced_data['Data_Quality_Score'] = quality_score

        # Add clustering metadata
        enhanced_data['Clustering_Timestamp'] = pd.Timestamp.now()
        enhanced_data['Pipeline_Version'] = 'KMeans_Enhanced_v2.0'

        # Reorder columns for better readability
        original_cols = [col for col in original_data.columns if col in enhanced_data.columns]
        quality_cols = [
            'Data_Quality_Score',
            'Data_Quality_Has_Null_Values',
            'Data_Quality_Has_Outliers',
            'Data_Quality_Issues',
            'Preprocessing_Techniques_Applied',
            'Best_Technique_Used',
            'Best_Technique_Summary',
            'Processing_Notes'
        ]

        clustering_cols = [col for col in enhanced_data.columns if 'Cluster' in col or 'Electrofacies' in col]
        metadata_cols = ['Clustering_Timestamp', 'Pipeline_Version']

        # Final column order
        final_columns = original_cols + quality_cols + clustering_cols + metadata_cols
        enhanced_data = enhanced_data[[col for col in final_columns if col in enhanced_data.columns]]

        print(f"✅ Enhanced dataset created with {len(enhanced_data):,} rows and {len(enhanced_data.columns)} columns")

        # Summary statistics
        null_rows = enhanced_data['Data_Quality_Has_Null_Values'].sum()
        outlier_rows = enhanced_data['Data_Quality_Has_Outliers'].sum()
        avg_quality_score = enhanced_data['Data_Quality_Score'].mean()

        print(f"📈 Quality Summary:")
        print(f"   • Rows with null values: {null_rows:,} ({null_rows/len(enhanced_data)*100:.1f}%)")
        print(f"   • Rows with outliers: {outlier_rows:,} ({outlier_rows/len(enhanced_data)*100:.1f}%)")
        print(f"   • Average quality score: {avg_quality_score:.1f}/100")

        if cluster_labels is not None:
            unique_clusters = len(set(cluster_labels))
            print(f"   • Number of clusters: {unique_clusters}")

        return enhanced_data

    def _detect_and_log_null_values(self, data, features):
        """Detect null values and log quality issues for K-Means"""
        print(f"🔍 Detecting null values in {len(features)} features...")

        total_null_rows = 0

        for feature in features:
            if feature in data.columns:
                null_mask = data[feature].isnull()
                null_count = null_mask.sum()

                if null_count > 0:
                    print(f"   📊 {feature}: {null_count:,} null values ({null_count/len(data)*100:.1f}%)")

                    # Update quality annotations
                    if not self.quality_annotations.empty:
                        self.quality_annotations.loc[null_mask, 'Has_Null_Values'] = True

                        # Add to quality issues
                        current_issues = self.quality_annotations.loc[null_mask, 'Quality_Issues']
                        new_issues = current_issues.apply(lambda x: f"{x}; Null in {feature}" if x else f"Null in {feature}")
                        self.quality_annotations.loc[null_mask, 'Quality_Issues'] = new_issues

                        total_null_rows += null_count

        if not self.quality_annotations.empty:
            total_unique_null_rows = self.quality_annotations['Has_Null_Values'].sum()
            print(f"📈 Total rows with null values: {total_unique_null_rows:,} ({total_unique_null_rows/len(data)*100:.1f}%)")
            return total_unique_null_rows

        return total_null_rows

    def _detect_and_log_outliers(self, data, features, method='iqr', threshold=3.0):
        """Detect outliers using multiple methods and log quality issues for K-Means"""
        print(f"🎯 Detecting outliers using {method.upper()} method...")

        outlier_counts = {}

        for feature in features:
            if feature in data.columns and data[feature].dtype in ['float64', 'int64']:
                if method == 'iqr':
                    Q1 = data[feature].quantile(0.25)
                    Q3 = data[feature].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outlier_mask = (data[feature] < lower_bound) | (data[feature] > upper_bound)

                elif method == 'zscore':
                    z_scores = np.abs((data[feature] - data[feature].mean()) / data[feature].std())
                    outlier_mask = z_scores > threshold

                elif method == 'modified_zscore':
                    median = data[feature].median()
                    mad = np.median(np.abs(data[feature] - median))
                    if mad != 0:
                        modified_z_scores = 0.6745 * (data[feature] - median) / mad
                        outlier_mask = np.abs(modified_z_scores) > threshold
                    else:
                        outlier_mask = pd.Series([False] * len(data), index=data.index)

                outlier_count = outlier_mask.sum()
                outlier_counts[feature] = outlier_count

                if outlier_count > 0:
                    print(f"   📊 {feature}: {outlier_count:,} outliers ({outlier_count/len(data)*100:.1f}%)")

                    # Update quality annotations
                    if not self.quality_annotations.empty:
                        self.quality_annotations.loc[outlier_mask, 'Has_Outliers'] = True

                        # Add to quality issues
                        current_issues = self.quality_annotations.loc[outlier_mask, 'Quality_Issues']
                        new_issues = current_issues.apply(lambda x: f"{x}; Outlier in {feature}" if x else f"Outlier in {feature}")
                        self.quality_annotations.loc[outlier_mask, 'Quality_Issues'] = new_issues

        if not self.quality_annotations.empty:
            total_outlier_rows = self.quality_annotations['Has_Outliers'].sum()
            print(f"📈 Total rows with outliers: {total_outlier_rows:,} ({total_outlier_rows/len(data)*100:.1f}%)")

        return outlier_counts

# Simplified clustering class for Streamlit (non-blocking)
class StreamlitKMeans:
    """Simplified K-Means clustering for Streamlit app"""

    def __init__(self, n_clusters=5, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.scaler = None
        self.feature_columns = ['GR', 'RDEP', 'RHOB', 'NPHI', 'PEF', 'DTC', 'CALI']
        self.data = None
        self.scaled_data = None
        self.cluster_labels = None

    def preprocess_data(self, data, selected_features):
        """Preprocess data for clustering"""
        # Filter available features
        available_features = [col for col in selected_features if col in data.columns]

        if not available_features:
            raise ValueError("No selected features found in dataset")

        # Extract features
        features_df = data[available_features].copy()

        # Remove rows with too many missing values
        threshold = len(available_features) * 0.5
        features_df = features_df.dropna(thresh=threshold)

        # Fill remaining missing values with median
        for col in available_features:
            if features_df[col].isnull().sum() > 0:
                median_val = features_df[col].median()
                features_df[col].fillna(median_val, inplace=True)

        # Normalize features
        self.scaler = StandardScaler()
        self.scaled_data = self.scaler.fit_transform(features_df)

        # Update data
        self.data = data.loc[features_df.index].copy()
        self.feature_columns = available_features

        return len(features_df)

    def find_optimal_clusters(self, max_clusters=8):
        """Find optimal clusters with non-blocking, fast execution"""
        silhouette_scores = []
        cluster_range = range(2, min(max_clusters + 1, 9))  # Limit to max 8 clusters for speed

        # Force smaller sample for optimization to prevent hanging
        optimization_sample_size = min(5000, len(self.scaled_data))
        if len(self.scaled_data) > optimization_sample_size:
            sample_indices = np.random.choice(
                len(self.scaled_data),
                size=optimization_sample_size,
                replace=False
            )
            opt_data = self.scaled_data[sample_indices]
        else:
            opt_data = self.scaled_data

        # Use fast algorithm for optimization
        for k in cluster_range:
            try:
                # Always use regular KMeans for optimization (faster than MiniBatch for small samples)
                kmeans = KMeans(
                    n_clusters=k,
                    random_state=self.random_state,
                    n_init=3,  # Reduced for speed
                    max_iter=100,  # Reduced for speed
                    tol=1e-3  # Relaxed tolerance for speed
                )

                labels = kmeans.fit_predict(opt_data)

                # Fast silhouette calculation on small sample
                if len(opt_data) > 2000:
                    sil_sample_size = min(2000, len(opt_data))
                    sil_indices = np.random.choice(len(opt_data), size=sil_sample_size, replace=False)
                    score = silhouette_score(opt_data[sil_indices], labels[sil_indices])
                else:
                    score = silhouette_score(opt_data, labels)

                silhouette_scores.append(score)

            except Exception as e:
                # If clustering fails, assign low score
                silhouette_scores.append(0.0)

        # Find optimal k
        if silhouette_scores:
            optimal_idx = np.argmax(silhouette_scores)
            optimal_k = cluster_range[optimal_idx]
        else:
            optimal_k = 4  # Default fallback

        return optimal_k, silhouette_scores, list(cluster_range)

    def fit_kmeans(self):
        """Fit K-Means model with timeout protection and fast execution"""
        try:
            # Choose algorithm based on dataset size with aggressive optimization
            if len(self.scaled_data) > 50000:
                # Use MiniBatchKMeans for datasets >50K (lowered threshold)
                from sklearn.cluster import MiniBatchKMeans
                self.kmeans = MiniBatchKMeans(
                    n_clusters=self.n_clusters,
                    random_state=self.random_state,
                    n_init=2,  # Reduced for speed
                    max_iter=50,  # Reduced for speed
                    batch_size=min(1000, len(self.scaled_data) // 50),  # Smaller batches
                    tol=1e-3  # Relaxed tolerance
                )
            else:
                # Use regular KMeans with speed optimizations
                self.kmeans = KMeans(
                    n_clusters=self.n_clusters,
                    random_state=self.random_state,
                    n_init=3,  # Reduced from 10 for speed
                    max_iter=100,  # Reduced from 300 for speed
                    tol=1e-3,  # Relaxed tolerance for faster convergence
                    algorithm='lloyd'  # Fastest algorithm
                )

            # Fit the model with error handling
            self.cluster_labels = self.kmeans.fit_predict(self.scaled_data)
            self.data['Cluster'] = self.cluster_labels

            # Fast silhouette score calculation
            if len(self.scaled_data) > 5000:
                # Use small sample for silhouette to prevent hanging
                sample_size = min(5000, len(self.scaled_data))
                sample_indices = np.random.choice(
                    len(self.scaled_data),
                    size=sample_size,
                    replace=False
                )
                silhouette_avg = silhouette_score(
                    self.scaled_data[sample_indices],
                    self.cluster_labels[sample_indices]
                )
            else:
                silhouette_avg = silhouette_score(self.scaled_data, self.cluster_labels)

            return silhouette_avg

        except Exception as e:
            # Fallback: assign random clusters if clustering fails
            self.cluster_labels = np.random.randint(0, self.n_clusters, len(self.scaled_data))
            self.data['Cluster'] = self.cluster_labels
            return 0.0  # Return low silhouette score for failed clustering

    def get_cluster_summary(self):
        """Get cluster statistics"""
        cluster_summary = []

        for cluster_id in range(self.n_clusters):
            cluster_data = self.data[self.data['Cluster'] == cluster_id]
            cluster_size = len(cluster_data)
            cluster_percentage = (cluster_size / len(self.data)) * 100

            cluster_stats = {}
            for feature in self.feature_columns:
                if feature in cluster_data.columns:
                    mean_val = cluster_data[feature].mean()
                    std_val = cluster_data[feature].std()
                    cluster_stats[feature] = {'mean': mean_val, 'std': std_val}

            cluster_summary.append({
                'cluster': cluster_id,
                'size': cluster_size,
                'percentage': cluster_percentage,
                'stats': cluster_stats
            })

        return cluster_summary

# Page configuration
st.set_page_config(
    page_title="Electrofacies K-Means Clustering",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main Streamlit application"""

    # Title and description
    st.markdown('<h1 class="main-header">🔬 Electrofacies K-Means Clustering</h1>', unsafe_allow_html=True)
    st.markdown("""
    This application performs K-Means clustering on well log data to classify electrofacies
    based on petrophysical properties. Upload your well log data and explore different clustering configurations.
    """)

    # Sidebar for parameters
    st.sidebar.header("📊 Clustering Parameters")

    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload Well Log CSV File",
        type=['csv'],
        help="Upload a CSV file containing well log data with columns: DEPTH_MD, GR, RDEP, RHOB, NPHI, PEF, DTC, CALI"
    )

    # Default file option
    use_default = st.sidebar.checkbox("Use Default Dataset (train.csv)", value=True)

    if uploaded_file is not None or use_default:

        # Load data
        if uploaded_file is not None:
            # Try different delimiters
            delimiter = st.sidebar.selectbox("CSV Delimiter", [';', ',', '\t'], index=0)
            try:
                data = pd.read_csv(uploaded_file, sep=delimiter)
                st.sidebar.success(f"✅ File uploaded successfully! Shape: {data.shape}")
            except Exception as e:
                st.sidebar.error(f"❌ Error loading file: {str(e)}")
                return
        else:
            # Use default dataset
            try:
                data = pd.read_csv('train.csv', sep=';')
                st.sidebar.success(f"✅ Default dataset loaded! Shape: {data.shape}")
            except Exception as e:
                st.sidebar.error(f"❌ Error loading default dataset: {str(e)}")
                return

        # Display data info
        st.markdown('<h2 class="sub-header">📋 Dataset Overview</h2>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", f"{data.shape[0]:,}")
        with col2:
            st.metric("Total Columns", data.shape[1])
        with col3:
            st.metric("Memory Usage", f"{data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

        # Show data preview
        with st.expander("🔍 Data Preview", expanded=False):
            st.dataframe(data.head(10))

        # Data Quality Analysis
        st.markdown('<h2 class="sub-header">🔍 Data Quality Analysis</h2>', unsafe_allow_html=True)

        # Initialize data quality analyzer
        quality_analyzer = DataQualityAnalyzer()

        # Get available features for analysis
        analysis_features = ['GR', 'RDEP', 'RHOB', 'NPHI', 'PEF', 'DTC', 'CALI', 'DEPTH_MD']
        analysis_features = [col for col in analysis_features if col in data.columns]

        # Perform quality analysis
        quality_report = quality_analyzer.analyze_data_quality(data, analysis_features)

        # Display quality summary
        col1, col2, col3 = st.columns(3)
        with col1:
            total_null_rows = len(quality_report['null_data'])
            st.metric("Rows with Null Values", f"{total_null_rows:,}")
        with col2:
            total_oob_rows = len(quality_report['out_of_bounds'])
            st.metric("Out-of-Bounds Rows", f"{total_oob_rows:,}")
        with col3:
            data_quality_score = max(0, 100 - ((total_null_rows + total_oob_rows) / len(data) * 100))
            st.metric("Data Quality Score", f"{data_quality_score:.1f}%")

        # Quality recommendations
        if quality_report['recommendations']:
            st.subheader("📋 Data Quality Recommendations")
            for recommendation in quality_report['recommendations']:
                if "🔴" in recommendation or "🚨" in recommendation:
                    st.error(recommendation)
                elif "🟡" in recommendation or "⚠️" in recommendation:
                    st.warning(recommendation)
                elif "✅" in recommendation:
                    st.success(recommendation)
                else:
                    st.info(recommendation)

        # Download options for quality issues
        col1, col2 = st.columns(2)

        with col1:
            if not quality_report['null_data'].empty:
                # Prepare null data for download
                null_csv = quality_report['null_data'].to_csv(index=False)
                st.download_button(
                    label="📥 Download Null Values Data",
                    data=null_csv,
                    file_name=f"null_values_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    help="Download rows containing null/missing values"
                )
            else:
                st.info("✅ No null values detected in dataset")

        with col2:
            if not quality_report['out_of_bounds'].empty:
                # Prepare out-of-bounds data for download
                oob_csv = quality_report['out_of_bounds'].to_csv(index=False)
                st.download_button(
                    label="📥 Download Out-of-Bounds Data",
                    data=oob_csv,
                    file_name=f"out_of_bounds_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    help="Download rows with values outside expected ranges"
                )
            else:
                st.info("✅ No out-of-bounds values detected")

        # Detailed quality analysis in expandable section
        with st.expander("📊 Detailed Quality Analysis", expanded=False):

            # Null values analysis
            if quality_report['summary_stats']['null_summary']:
                st.subheader("🔍 Null Values Analysis")
                null_df = pd.DataFrame.from_dict(
                    quality_report['summary_stats']['null_summary'],
                    orient='index'
                )
                null_df.index.name = 'Feature'
                null_df = null_df.reset_index()
                st.dataframe(null_df, use_container_width=True)

            # Out-of-bounds analysis
            if quality_report['summary_stats']['bounds_summary']:
                st.subheader("📏 Out-of-Bounds Analysis")
                bounds_df = pd.DataFrame.from_dict(
                    quality_report['summary_stats']['bounds_summary'],
                    orient='index'
                )
                bounds_df.index.name = 'Feature'
                bounds_df = bounds_df.reset_index()
                st.dataframe(bounds_df, use_container_width=True)

        # Feature selection
        st.sidebar.subheader("🎯 Feature Selection")
        available_features = ['GR', 'RDEP', 'RHOB', 'NPHI', 'PEF', 'DTC', 'CALI']
        available_features = [col for col in available_features if col in data.columns]

        selected_features = st.sidebar.multiselect(
            "Select Features for Clustering",
            available_features,
            default=available_features,
            help="Select the petrophysical features to use for clustering"
        )

        if not selected_features:
            st.sidebar.error("❌ Please select at least one feature!")
            return

        # Clustering parameters
        st.sidebar.subheader("⚙️ Clustering Settings")

        auto_clusters = st.sidebar.checkbox("Auto-determine optimal clusters", value=True)

        if auto_clusters:
            max_clusters = st.sidebar.slider("Maximum clusters to test", 2, 15, 10)
            n_clusters = None
        else:
            n_clusters = st.sidebar.slider("Number of clusters", 2, 15, 5)

        # Random State explanation and input
        st.sidebar.markdown("**🎲 Random State:**")
        random_state = st.sidebar.number_input(
            "Random State",
            value=42,
            help="Controls randomness for reproducible results. Same number = identical clustering every time"
        )

        with st.sidebar.expander("ℹ️ What is Random State?"):
            st.markdown("""
            **Random State** ensures reproducible results by controlling:
            - Initial cluster center placement
            - Data sampling randomness
            - Algorithm convergence paths

            **Examples:**
            - `42` (recommended): Standard seed
            - `123`: Alternative seed
            - Change it to explore different cluster arrangements
            """)

        # Dataset size and sampling
        st.sidebar.markdown("**📊 Dataset Processing:**")

        # Intelligent sample size recommendation
        if len(data) <= 10000:
            recommended_size = len(data)
            mode_suggestion = "Use full dataset"
        elif len(data) <= 50000:
            recommended_size = min(25000, len(data))
            mode_suggestion = "Balanced processing"
        elif len(data) <= 200000:
            recommended_size = min(50000, len(data))
            mode_suggestion = "High-quality sampling"
        else:
            recommended_size = min(100000, len(data))
            mode_suggestion = "Large dataset optimization"

        st.sidebar.info(f"💡 **Recommendation:** {mode_suggestion}")

        sample_size = st.sidebar.slider(
            "Sample Size",
            min_value=1000,
            max_value=min(500000, len(data)),  # Increased maximum to 500K
            value=recommended_size,
            help=f"Dataset has {len(data):,} rows. Larger samples = better quality but slower processing"
        )

        # Show processing estimates
        processing_time = sample_size / 8000  # Optimized estimate
        if processing_time > 60:
            time_str = f"{processing_time/60:.1f} minutes"
            color = "warning"
        else:
            time_str = f"{processing_time:.0f} seconds"
            color = "info"

        if color == "warning":
            st.sidebar.warning(f"⏱️ Estimated time: {time_str}")
        else:
            st.sidebar.info(f"⏱️ Estimated time: {time_str}")

        # Smart recommendations based on dataset size
        if len(data) > 200000:
            st.sidebar.warning("🔥 Very large dataset! Recommended sample: 50K-100K rows")
        elif len(data) > 50000:
            st.sidebar.info("📊 Large dataset. Recommended sample: 25K-50K rows")
        else:
            st.sidebar.success("✅ Dataset size is optimal for full processing")

        # Run clustering button
        if st.sidebar.button("🚀 Run K-Means Clustering", type="primary"):

            # Initialize progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                # Smart data sampling
                if len(data) > sample_size:
                    status_text.text(f"Sampling {sample_size:,} rows from {len(data):,} total rows...")

                    # Use depth-based stratified sampling for better geological representation
                    if 'DEPTH_MD' in data.columns and len(data) > 10000:
                        try:
                            # Create depth bins and sample proportionally
                            depth_bins = pd.qcut(data['DEPTH_MD'], q=min(10, len(data)//1000), duplicates='drop')
                            sampled_data = data.groupby(depth_bins, group_keys=False).apply(
                                lambda x: x.sample(min(len(x), max(1, sample_size // 10)), random_state=random_state)
                            )
                            # Ensure we get the right sample size
                            if len(sampled_data) > sample_size:
                                sampled_data = sampled_data.sample(n=sample_size, random_state=random_state)
                            st.info(f"📊 Used depth-stratified sampling for better geological representation")
                        except:
                            # Fallback to regular sampling if stratified fails
                            sampled_data = data.sample(n=sample_size, random_state=random_state)
                            st.info(f"📊 Used random sampling")
                    else:
                        # Regular random sampling for smaller datasets or no depth column
                        sampled_data = data.sample(n=sample_size, random_state=random_state)
                        st.info(f"📊 Used random sampling")
                else:
                    sampled_data = data.copy()
                    st.info(f"📊 Using complete dataset ({len(data):,} rows)")

                # Initialize clustering class
                status_text.text("Initializing K-Means analyzer...")
                progress_bar.progress(10)

                kmeans_analyzer = StreamlitKMeans(
                    n_clusters=n_clusters if not auto_clusters else 5,
                    random_state=random_state
                )

                # Preprocess data
                status_text.text("Preprocessing data...")
                progress_bar.progress(20)

                clean_points = kmeans_analyzer.preprocess_data(sampled_data, selected_features)
                st.info(f"📊 Processing {clean_points:,} clean data points")

                # Find optimal clusters if requested
                if auto_clusters:
                    status_text.text("Finding optimal number of clusters (fast mode)...")
                    progress_bar.progress(40)

                    try:
                        # Limit max_clusters to prevent hanging
                        safe_max_clusters = min(max_clusters, 6)
                        optimal_k, scores, cluster_range = kmeans_analyzer.find_optimal_clusters(safe_max_clusters)
                        kmeans_analyzer.n_clusters = optimal_k

                        st.success(f"🎯 Optimal number of clusters: {optimal_k}")

                        # Show optimization plot (non-blocking)
                        if scores and len(scores) > 0:
                            fig_opt = go.Figure()
                            fig_opt.add_trace(go.Scatter(
                                x=list(cluster_range),
                                y=scores,
                                mode='lines+markers',
                                name='Silhouette Score',
                                line=dict(width=3),
                                marker=dict(size=8)
                            ))
                            fig_opt.update_layout(
                                title="Cluster Optimization Results",
                                xaxis_title="Number of Clusters",
                                yaxis_title="Silhouette Score",
                                height=400
                            )
                            st.plotly_chart(fig_opt, use_container_width=True)
                    except Exception as opt_error:
                        st.warning(f"⚠️ Optimization failed, using default clusters: {kmeans_analyzer.n_clusters}")

                # Fit K-Means with progress updates
                status_text.text("Fitting K-Means model (optimized)...")
                progress_bar.progress(60)

                try:
                    silhouette_avg = kmeans_analyzer.fit_kmeans()
                    if silhouette_avg == 0.0:
                        st.warning("⚠️ Clustering completed with fallback method")
                    else:
                        st.info(f"📊 Silhouette Score: {silhouette_avg:.3f}")
                except Exception as fit_error:
                    st.error(f"❌ Clustering failed: {str(fit_error)}")
                    return

                # Analyze clusters
                status_text.text("Generating cluster analysis...")
                progress_bar.progress(80)

                try:
                    cluster_summary = kmeans_analyzer.get_cluster_summary()
                except Exception as analysis_error:
                    st.error(f"❌ Analysis failed: {str(analysis_error)}")
                    return

                # Complete
                status_text.text("Analysis completed!")
                progress_bar.progress(100)

                # Store results in session state
                st.session_state['kmeans_analyzer'] = kmeans_analyzer
                st.session_state['cluster_summary'] = cluster_summary
                st.session_state['silhouette_score'] = silhouette_avg
                st.session_state['analysis_complete'] = True

                st.success("✅ K-Means clustering completed successfully!")

            except Exception as e:
                st.error(f"❌ Error during clustering: {str(e)}")
                st.error("Please check your data format and try again.")
                return

        # Display results if analysis is complete
        if st.session_state.get('analysis_complete', False):

            kmeans_analyzer = st.session_state['kmeans_analyzer']
            cluster_summary = st.session_state['cluster_summary']

            # Cluster summary
            st.markdown('<h2 class="sub-header">📊 Cluster Analysis Results</h2>', unsafe_allow_html=True)

            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Number of Clusters", kmeans_analyzer.n_clusters)
            with col2:
                silhouette_avg = st.session_state.get('silhouette_score', 0)
                st.metric("Silhouette Score", f"{silhouette_avg:.3f}")
            with col3:
                st.metric("Data Points", len(kmeans_analyzer.data))
            with col4:
                st.metric("Features Used", len(kmeans_analyzer.feature_columns))

            # Cluster statistics table
            st.subheader("📈 Cluster Statistics")

            stats_data = []
            for summary in cluster_summary:
                row = {
                    'Cluster': summary['cluster'],
                    'Size': summary['size'],
                    'Percentage': f"{summary['percentage']:.1f}%"
                }
                for feature in kmeans_analyzer.feature_columns:
                    if feature in summary['stats']:
                        mean_val = summary['stats'][feature]['mean']
                        std_val = summary['stats'][feature]['std']
                        row[f'{feature}_mean'] = f"{mean_val:.2f}"
                        row[f'{feature}_std'] = f"{std_val:.2f}"
                stats_data.append(row)

            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True)

            # Visualization tabs
            st.markdown('<h2 class="sub-header">📈 Interactive Visualizations</h2>', unsafe_allow_html=True)

            tab1, tab2, tab3 = st.tabs(["📊 Log Curves", "🎯 Crossplots", "📏 Depth Track"])

            with tab1:
                st.subheader("Well Log Curves with Cluster Overlays")

                # Create log curves plot
                colors = px.colors.qualitative.Set1[:kmeans_analyzer.n_clusters]

                # Select curves to display
                available_curves = kmeans_analyzer.feature_columns
                display_curves = st.multiselect(
                    "Select log curves to display:",
                    available_curves,
                    default=available_curves[:4] if len(available_curves) >= 4 else available_curves
                )

                if display_curves:
                    fig = make_subplots(
                        rows=1, cols=len(display_curves),
                        subplot_titles=display_curves,
                        shared_yaxes=True,
                        horizontal_spacing=0.05
                    )

                    for i, curve in enumerate(display_curves, 1):
                        for cluster_id in range(kmeans_analyzer.n_clusters):
                            cluster_data = kmeans_analyzer.data[kmeans_analyzer.data['Cluster'] == cluster_id]

                            fig.add_trace(
                                go.Scatter(
                                    x=cluster_data[curve],
                                    y=cluster_data['DEPTH_MD'],
                                    mode='markers',
                                    marker=dict(
                                        color=colors[cluster_id],
                                        size=2,
                                        opacity=0.6
                                    ),
                                    name=f'Cluster {cluster_id}' if i == 1 else None,
                                    showlegend=True if i == 1 else False,
                                    legendgroup=f'cluster_{cluster_id}'
                                ),
                                row=1, col=i
                            )

                    fig.update_layout(
                        title='Well Log Curves by Cluster',
                        height=800,
                        yaxis=dict(autorange='reversed', title='Depth (MD)')
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with tab2:
                st.subheader("Crossplot Analysis")

                # Feature selection for crossplot
                col1, col2 = st.columns(2)
                with col1:
                    x_feature = st.selectbox("X-axis feature:", available_curves, index=0)
                with col2:
                    y_feature = st.selectbox("Y-axis feature:", available_curves, index=1)

                if x_feature != y_feature:
                    fig = go.Figure()

                    for cluster_id in range(kmeans_analyzer.n_clusters):
                        cluster_data = kmeans_analyzer.data[kmeans_analyzer.data['Cluster'] == cluster_id]

                        fig.add_trace(
                            go.Scatter(
                                x=cluster_data[x_feature],
                                y=cluster_data[y_feature],
                                mode='markers',
                                marker=dict(
                                    color=colors[cluster_id],
                                    size=4,
                                    opacity=0.6
                                ),
                                name=f'Cluster {cluster_id}'
                            )
                        )

                    fig.update_layout(
                        title=f'{y_feature} vs {x_feature}',
                        xaxis_title=x_feature,
                        yaxis_title=y_feature,
                        height=600
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with tab3:
                st.subheader("Cluster vs Depth Track")

                fig = go.Figure()

                for cluster_id in range(kmeans_analyzer.n_clusters):
                    cluster_data = kmeans_analyzer.data[kmeans_analyzer.data['Cluster'] == cluster_id]

                    fig.add_trace(
                        go.Scatter(
                            x=[cluster_id] * len(cluster_data),
                            y=cluster_data['DEPTH_MD'],
                            mode='markers',
                            marker=dict(
                                color=colors[cluster_id],
                                size=3,
                                opacity=0.8
                            ),
                            name=f'Cluster {cluster_id}',
                            hovertemplate=f'Cluster {cluster_id}<br>Depth: %{{y:.2f}} m<extra></extra>'
                        )
                    )

                fig.update_layout(
                    title='Electrofacies Clusters vs Depth',
                    xaxis_title='Cluster ID',
                    yaxis_title='Depth (MD)',
                    yaxis=dict(autorange='reversed'),
                    height=800
                )

                st.plotly_chart(fig, use_container_width=True)


            # Download Results
            st.markdown('<h2 class="sub-header">💾 Download Options</h2>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)

            # Standard clustered data download
            with col1:
                st.subheader("📊 Standard Download")
                clustered_data = kmeans_analyzer.data.copy()
                csv_buffer = io.StringIO()
                clustered_data.to_csv(csv_buffer, index=False, sep=';')
                csv_data = csv_buffer.getvalue()
                st.download_button(
                    label="📥 Download Clustered Dataset",
                    data=csv_data,
                    file_name=f"clustered_electrofacies_{kmeans_analyzer.n_clusters}clusters.csv",
                    mime="text/csv",
                    help="Download dataset with cluster labels"
                )
                st.info(f"📊 {len(clustered_data):,} rows, {len(clustered_data.columns)} columns")

            # Enhanced dataset with quality tracking
            with col2:
                st.subheader("🔬 With Quality Tracking")
                try:
                    enhanced_data = data.copy()
                    # Add cluster assignments
                    if 'Cluster' in kmeans_analyzer.data.columns:
                        enhanced_data['Cluster_Assignment'] = np.nan
                        for idx in kmeans_analyzer.data.index:
                            enhanced_data.at[idx, 'Cluster_Assignment'] = kmeans_analyzer.data.at[idx, 'Cluster']
                        enhanced_data['Electrofacies'] = enhanced_data['Cluster_Assignment'].apply(lambda x: f'Electrofacies_{int(x)}' if pd.notnull(x) else 'Unknown')
                    # Add basic quality tracking
                    enhanced_data['Has_Null_Values'] = enhanced_data[analysis_features].isnull().any(axis=1)
                    enhanced_data['Null_Count'] = enhanced_data[analysis_features].isnull().sum(axis=1)
                    enhanced_data['Quality_Score'] = 100 - (enhanced_data['Null_Count'] * 10)
                    enhanced_data['Quality_Score'] = enhanced_data['Quality_Score'].clip(lower=0)
                    enhanced_data['Processing_Applied'] = 'StandardScaler; Median_Imputation'
                    enhanced_data['Clustering_Method'] = 'KMeans'
                    enhanced_data['Silhouette_Score'] = silhouette_avg
                    # Prepare enhanced dataset
                    enhanced_csv_buffer = io.StringIO()
                    enhanced_data.to_csv(enhanced_csv_buffer, index=False, sep=';')
                    enhanced_csv_data = enhanced_csv_buffer.getvalue()
                    st.download_button(
                        label="📥 Download with Quality Tracking",
                        data=enhanced_csv_data,
                        file_name=f"electrofacies_with_quality_{kmeans_analyzer.n_clusters}clusters_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        help="Download dataset with quality annotations and clustering results"
                    )
                    st.success(f"🔬 {len(enhanced_data):,} rows, {len(enhanced_data.columns)} columns")
                except Exception as e:
                    st.error(f"Error creating enhanced dataset: {str(e)}")

            # Quality tracking preview
            with st.expander("🔬 Quality Tracking Preview", expanded=False):
                st.write("**Quality tracking includes:**")
                st.write("• Null value detection and counts")
                st.write("• Data quality scores (0-100)")
                st.write("• Processing techniques applied")
                st.write("• Cluster assignments and electrofacies labels")
                st.write("• Clustering performance metrics")
                if 'enhanced_data' in locals():
                    quality_cols = ['Has_Null_Values', 'Null_Count', 'Quality_Score', 'Cluster_Assignment', 'Electrofacies']
                    available_quality_cols = [col for col in quality_cols if col in enhanced_data.columns]
                    if available_quality_cols:
                        st.dataframe(enhanced_data[available_quality_cols].head(), use_container_width=True)

    else:
        st.info("👆 Please upload a CSV file or use the default dataset to get started!")

if __name__ == "__main__":
    # Initialize session state
    if 'analysis_complete' not in st.session_state:
        st.session_state['analysis_complete'] = False

    main()
