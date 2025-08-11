"""
K-Means Clustering for Electrofacies Classification
==================================================

This script performs K-Means clustering on well log data to classify electrofacies
based on petrophysical properties.

Author: AI Assistant
Date: 2025-07-22
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class ElectrofaciesKMeans:
    """
    K-Means clustering class for electrofacies classification
    """

    def __init__(self, n_clusters=5, random_state=42):
        """
        Initialize the ElectrofaciesKMeans class

        Parameters:
        -----------
        n_clusters : int, default=5
            Number of clusters for K-Means
        random_state : int, default=42
            Random state for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.scaler = None
        self.feature_columns = ['GR', 'RDEP', 'RHOB', 'NPHI', 'PEF', 'DTC', 'CALI']
        self.data = None
        self.scaled_data = None
        self.cluster_labels = None

    def load_data(self, file_path, delimiter=';'):
        """
        Load well log data from CSV file

        Parameters:
        -----------
        file_path : str
            Path to the CSV file
        delimiter : str, default=';'
            Delimiter used in the CSV file
        """
        print(f"Loading data from {file_path}...")
        self.data = pd.read_csv(file_path, sep=delimiter)
        print(f"Data loaded successfully. Shape: {self.data.shape}")
        print(f"Columns: {self.data.columns.tolist()}")

    def preprocess_data(self):
        """
        Preprocess the data: handle missing values and normalize features
        """
        print("\nPreprocessing data...")

        # Check for required columns
        missing_cols = [col for col in self.feature_columns if col not in self.data.columns]
        if missing_cols:
            print(f"Warning: Missing columns: {missing_cols}")
            self.feature_columns = [col for col in self.feature_columns if col in self.data.columns]
            print(f"Using available columns: {self.feature_columns}")

        # Extract features and handle missing values
        features_df = self.data[self.feature_columns].copy()

        print(f"Missing values before cleaning:")
        print(features_df.isnull().sum())

        # Remove rows with too many missing values (>50% of features)
        threshold = len(self.feature_columns) * 0.5
        features_df = features_df.dropna(thresh=threshold)

        # Fill remaining missing values with median
        for col in self.feature_columns:
            if features_df[col].isnull().sum() > 0:
                median_val = features_df[col].median()
                features_df[col].fillna(median_val, inplace=True)
                print(f"Filled {col} missing values with median: {median_val:.2f}")

        print(f"\nData shape after cleaning: {features_df.shape}")
        print(f"Missing values after cleaning:")
        print(features_df.isnull().sum())

        # Normalize features using StandardScaler
        self.scaler = StandardScaler()
        self.scaled_data = self.scaler.fit_transform(features_df)

        # Update the main dataframe to keep only clean rows
        self.data = self.data.loc[features_df.index].copy()

        print("Data preprocessing completed successfully!")

    def find_optimal_clusters(self, max_clusters=10):
        """
        Find optimal number of clusters using elbow method and silhouette analysis

        Parameters:
        -----------
        max_clusters : int, default=10
            Maximum number of clusters to test
        """
        print(f"\nFinding optimal number of clusters (testing 2 to {max_clusters})...")

        inertias = []
        silhouette_scores = []
        cluster_range = range(2, max_clusters + 1)

        for k in cluster_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            kmeans.fit(self.scaled_data)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.scaled_data, kmeans.labels_))

        # Plot elbow curve and silhouette scores
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Elbow curve
        ax1.plot(cluster_range, inertias, 'bo-')
        ax1.set_xlabel('Number of Clusters')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method for Optimal Clusters')
        ax1.grid(True)

        # Silhouette scores
        ax2.plot(cluster_range, silhouette_scores, 'ro-')
        ax2.set_xlabel('Number of Clusters')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Analysis')
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig('cluster_optimization.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Find optimal k based on silhouette score
        optimal_k = cluster_range[np.argmax(silhouette_scores)]
        print(f"Optimal number of clusters based on silhouette score: {optimal_k}")
        print(f"Best silhouette score: {max(silhouette_scores):.3f}")

        return optimal_k, silhouette_scores

    def fit_kmeans(self):
        """
        Fit K-Means clustering model
        """
        print(f"\nFitting K-Means with {self.n_clusters} clusters...")

        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
            max_iter=300
        )

        self.cluster_labels = self.kmeans.fit_predict(self.scaled_data)

        # Add cluster labels to the dataframe
        self.data['Cluster'] = self.cluster_labels

        # Calculate silhouette score
        silhouette_avg = silhouette_score(self.scaled_data, self.cluster_labels)
        print(f"Silhouette Score: {silhouette_avg:.3f}")

        print("K-Means clustering completed successfully!")

    def analyze_clusters(self):
        """
        Analyze and summarize cluster characteristics
        """
        print("\n" + "="*60)
        print("CLUSTER ANALYSIS SUMMARY")
        print("="*60)

        cluster_summary = []

        for cluster_id in range(self.n_clusters):
            cluster_data = self.data[self.data['Cluster'] == cluster_id]
            cluster_size = len(cluster_data)
            cluster_percentage = (cluster_size / len(self.data)) * 100

            print(f"\nCluster {cluster_id}:")
            print(f"  Size: {cluster_size} points ({cluster_percentage:.1f}%)")
            print(f"  Feature Statistics:")

            cluster_stats = {}
            for feature in self.feature_columns:
                if feature in cluster_data.columns:
                    mean_val = cluster_data[feature].mean()
                    std_val = cluster_data[feature].std()
                    print(f"    {feature}: {mean_val:.2f} ± {std_val:.2f}")
                    cluster_stats[feature] = {'mean': mean_val, 'std': std_val}

            cluster_summary.append({
                'cluster': cluster_id,
                'size': cluster_size,
                'percentage': cluster_percentage,
                'stats': cluster_stats
            })

        return cluster_summary

    def plot_log_curves_with_clusters(self):
        """
        Plot log curves vs depth with cluster overlays
        """
        print("\nCreating log curves with cluster overlays...")

        # Define colors for clusters
        colors = px.colors.qualitative.Set1[:self.n_clusters]

        # Create subplots for log curves
        fig = make_subplots(
            rows=1, cols=4,
            subplot_titles=['Gamma Ray (GR)', 'Deep Resistivity (RDEP)',
                          'Bulk Density (RHOB)', 'Neutron Porosity (NPHI)'],
            shared_yaxes=True,
            horizontal_spacing=0.05
        )

        log_curves = ['GR', 'RDEP', 'RHOB', 'NPHI']

        for i, log_curve in enumerate(log_curves, 1):
            if log_curve in self.data.columns:
                for cluster_id in range(self.n_clusters):
                    cluster_data = self.data[self.data['Cluster'] == cluster_id]

                    fig.add_trace(
                        go.Scatter(
                            x=cluster_data[log_curve],
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

        # Update layout
        fig.update_layout(
            title='Well Log Curves with K-Means Clusters',
            height=800,
            showlegend=True
        )

        # Invert y-axis for depth
        fig.update_yaxes(autorange='reversed', title_text='Depth (MD)')

        # Update x-axis titles
        fig.update_xaxes(title_text='GR (API)', row=1, col=1)
        fig.update_xaxes(title_text='RDEP (Ohm·m)', row=1, col=2)
        fig.update_xaxes(title_text='RHOB (g/cc)', row=1, col=3)
        fig.update_xaxes(title_text='NPHI (v/v)', row=1, col=4)

        fig.write_html('log_curves_clusters.html')
        fig.show()

    def plot_crossplots_with_clusters(self):
        """
        Plot crossplots with cluster coloring
        """
        print("\nCreating crossplots with cluster coloring...")

        # Create subplots for crossplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['RHOB vs NPHI', 'GR vs RDEP'],
            horizontal_spacing=0.1
        )

        colors = px.colors.qualitative.Set1[:self.n_clusters]

        # RHOB vs NPHI crossplot
        if 'RHOB' in self.data.columns and 'NPHI' in self.data.columns:
            for cluster_id in range(self.n_clusters):
                cluster_data = self.data[self.data['Cluster'] == cluster_id]

                fig.add_trace(
                    go.Scatter(
                        x=cluster_data['RHOB'],
                        y=cluster_data['NPHI'],
                        mode='markers',
                        marker=dict(
                            color=colors[cluster_id],
                            size=3,
                            opacity=0.6
                        ),
                        name=f'Cluster {cluster_id}',
                        legendgroup=f'cluster_{cluster_id}'
                    ),
                    row=1, col=1
                )

        # GR vs RDEP crossplot
        if 'GR' in self.data.columns and 'RDEP' in self.data.columns:
            for cluster_id in range(self.n_clusters):
                cluster_data = self.data[self.data['Cluster'] == cluster_id]

                fig.add_trace(
                    go.Scatter(
                        x=cluster_data['GR'],
                        y=cluster_data['RDEP'],
                        mode='markers',
                        marker=dict(
                            color=colors[cluster_id],
                            size=3,
                            opacity=0.6
                        ),
                        name=f'Cluster {cluster_id}',
                        showlegend=False,
                        legendgroup=f'cluster_{cluster_id}'
                    ),
                    row=1, col=2
                )

        # Update layout
        fig.update_layout(
            title='Petrophysical Crossplots with K-Means Clusters',
            height=600,
            showlegend=True
        )

        # Update axis labels
        fig.update_xaxes(title_text='RHOB (g/cc)', row=1, col=1)
        fig.update_yaxes(title_text='NPHI (v/v)', row=1, col=1)
        fig.update_xaxes(title_text='GR (API)', row=1, col=2)
        fig.update_yaxes(title_text='RDEP (Ohm·m)', type='log', row=1, col=2)

        fig.write_html('crossplots_clusters.html')
        fig.show()

    def plot_cluster_depth_track(self):
        """
        Plot cluster vs depth track to show electrofacies zones
        """
        print("\nCreating cluster vs depth track...")

        colors = px.colors.qualitative.Set1[:self.n_clusters]

        fig = go.Figure()

        for cluster_id in range(self.n_clusters):
            cluster_data = self.data[self.data['Cluster'] == cluster_id]

            fig.add_trace(
                go.Scatter(
                    x=[cluster_id] * len(cluster_data),
                    y=cluster_data['DEPTH_MD'],
                    mode='markers',
                    marker=dict(
                        color=colors[cluster_id],
                        size=2,
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
            height=800,
            showlegend=True
        )

        fig.write_html('cluster_depth_track.html')
        fig.show()

    def save_results(self, output_file='clustered_data.csv'):
        """
        Save the cluster-labeled dataset

        Parameters:
        -----------
        output_file : str, default='clustered_data.csv'
            Output filename for the clustered dataset
        """
        self.data.to_csv(output_file, index=False, sep=';')
        print(f"\nCluster-labeled dataset saved to: {output_file}")

        # Also save the model and scaler
        joblib.dump(self.kmeans, 'kmeans_model.joblib')
        joblib.dump(self.scaler, 'scaler.joblib')
        print("Model and scaler saved successfully!")

    def run_complete_analysis(self, file_path, find_optimal=True, max_clusters=10):
        """
        Run the complete electrofacies clustering analysis

        Parameters:
        -----------
        file_path : str
            Path to the input CSV file
        find_optimal : bool, default=True
            Whether to find optimal number of clusters
        max_clusters : int, default=10
            Maximum number of clusters to test for optimization
        """
        print("Starting Complete Electrofacies K-Means Analysis")
        print("=" * 60)

        # Load and preprocess data
        self.load_data(file_path)
        self.preprocess_data()

        # Find optimal clusters if requested
        if find_optimal:
            optimal_k, _ = self.find_optimal_clusters(max_clusters)
            self.n_clusters = optimal_k

        # Fit K-Means
        self.fit_kmeans()

        # Analyze clusters
        cluster_summary = self.analyze_clusters()

        # Create visualizations
        self.plot_log_curves_with_clusters()
        self.plot_crossplots_with_clusters()
        self.plot_cluster_depth_track()

        # Save results
        self.save_results()

        print("\n" + "="*60)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Generated files:")
        print("- clustered_data.csv: Dataset with cluster labels")
        print("- log_curves_clusters.html: Interactive log curves plot")
        print("- crossplots_clusters.html: Interactive crossplots")
        print("- cluster_depth_track.html: Cluster vs depth track")
        print("- cluster_optimization.png: Cluster optimization plots")
        print("- kmeans_model.joblib: Trained K-Means model")
        print("- scaler.joblib: Feature scaler")

        return cluster_summary


# Example usage
if __name__ == "__main__":
    # Initialize the clustering class
    kmeans_analyzer = ElectrofaciesKMeans(n_clusters=5, random_state=42)

    # Run complete analysis
    cluster_summary = kmeans_analyzer.run_complete_analysis(
        file_path='train.csv',
        find_optimal=True,
        max_clusters=8
    )
