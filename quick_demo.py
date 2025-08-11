"""
Quick Demo of K-Means Electrofacies Clustering
==============================================

This script demonstrates the clustering functionality with a smaller sample
for faster execution and testing.

Author: AI Assistant
Date: 2025-07-22
"""

import pandas as pd
import numpy as np
from kmeans_electrofacies import ElectrofaciesKMeans
import matplotlib.pyplot as plt

def run_quick_demo():
    """Run a quick demonstration with sampled data"""
    
    print("🚀 Quick Demo: K-Means Electrofacies Clustering")
    print("=" * 60)
    
    try:
        # Load and sample data for faster processing
        print("1. Loading and sampling data...")
        data = pd.read_csv('train.csv', sep=';')
        print(f"   Original dataset size: {data.shape}")
        
        # Sample 10,000 rows for quick demo
        sample_size = min(10000, len(data))
        sampled_data = data.sample(n=sample_size, random_state=42)
        sampled_data.to_csv('demo_sample.csv', sep=';', index=False)
        print(f"   Created sample with {sample_size} rows")
        
        # Initialize clustering
        print("\n2. Initializing K-Means analyzer...")
        kmeans_analyzer = ElectrofaciesKMeans(n_clusters=5, random_state=42)
        
        # Load sampled data
        print("\n3. Loading sample data...")
        kmeans_analyzer.load_data('demo_sample.csv', delimiter=';')
        
        # Preprocess
        print("\n4. Preprocessing data...")
        kmeans_analyzer.preprocess_data()
        
        # Find optimal clusters (quick version)
        print("\n5. Finding optimal clusters...")
        optimal_k, silhouette_scores = kmeans_analyzer.find_optimal_clusters(max_clusters=8)
        kmeans_analyzer.n_clusters = optimal_k
        
        # Fit K-Means
        print(f"\n6. Fitting K-Means with {optimal_k} clusters...")
        kmeans_analyzer.fit_kmeans()
        
        # Analyze clusters
        print("\n7. Analyzing clusters...")
        cluster_summary = kmeans_analyzer.analyze_clusters()
        
        # Create basic visualizations (without interactive plots for speed)
        print("\n8. Creating basic visualizations...")
        
        # Simple matplotlib plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: GR vs Depth
        colors = plt.cm.Set1(np.linspace(0, 1, optimal_k))
        for i, cluster_id in enumerate(range(optimal_k)):
            cluster_data = kmeans_analyzer.data[kmeans_analyzer.data['Cluster'] == cluster_id]
            axes[0, 0].scatter(cluster_data['GR'], cluster_data['DEPTH_MD'], 
                             c=[colors[i]], label=f'Cluster {cluster_id}', alpha=0.6, s=1)
        axes[0, 0].set_xlabel('Gamma Ray (API)')
        axes[0, 0].set_ylabel('Depth (MD)')
        axes[0, 0].set_title('GR vs Depth with Clusters')
        axes[0, 0].invert_yaxis()
        axes[0, 0].legend()
        
        # Plot 2: RHOB vs NPHI
        if 'RHOB' in kmeans_analyzer.data.columns and 'NPHI' in kmeans_analyzer.data.columns:
            for i, cluster_id in enumerate(range(optimal_k)):
                cluster_data = kmeans_analyzer.data[kmeans_analyzer.data['Cluster'] == cluster_id]
                axes[0, 1].scatter(cluster_data['RHOB'], cluster_data['NPHI'], 
                                 c=[colors[i]], label=f'Cluster {cluster_id}', alpha=0.6, s=1)
            axes[0, 1].set_xlabel('Bulk Density (g/cc)')
            axes[0, 1].set_ylabel('Neutron Porosity (v/v)')
            axes[0, 1].set_title('RHOB vs NPHI Crossplot')
            axes[0, 1].legend()
        
        # Plot 3: GR vs RDEP
        if 'RDEP' in kmeans_analyzer.data.columns:
            for i, cluster_id in enumerate(range(optimal_k)):
                cluster_data = kmeans_analyzer.data[kmeans_analyzer.data['Cluster'] == cluster_id]
                axes[1, 0].scatter(cluster_data['GR'], cluster_data['RDEP'], 
                                 c=[colors[i]], label=f'Cluster {cluster_id}', alpha=0.6, s=1)
            axes[1, 0].set_xlabel('Gamma Ray (API)')
            axes[1, 0].set_ylabel('Deep Resistivity (Ohm·m)')
            axes[1, 0].set_title('GR vs RDEP Crossplot')
            axes[1, 0].set_yscale('log')
            axes[1, 0].legend()
        
        # Plot 4: Cluster distribution
        cluster_counts = kmeans_analyzer.data['Cluster'].value_counts().sort_index()
        axes[1, 1].bar(cluster_counts.index, cluster_counts.values, color=colors[:len(cluster_counts)])
        axes[1, 1].set_xlabel('Cluster ID')
        axes[1, 1].set_ylabel('Number of Points')
        axes[1, 1].set_title('Cluster Size Distribution')
        
        plt.tight_layout()
        plt.savefig('demo_clustering_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save results
        print("\n9. Saving results...")
        kmeans_analyzer.save_results('demo_clustered_data.csv')
        
        # Summary statistics
        print("\n" + "="*60)
        print("DEMO RESULTS SUMMARY")
        print("="*60)
        print(f"Dataset size: {len(kmeans_analyzer.data):,} points")
        print(f"Optimal clusters: {optimal_k}")
        print(f"Features used: {', '.join(kmeans_analyzer.feature_columns)}")
        
        print(f"\nCluster sizes:")
        for cluster_id in range(optimal_k):
            count = (kmeans_analyzer.data['Cluster'] == cluster_id).sum()
            percentage = (count / len(kmeans_analyzer.data)) * 100
            print(f"  Cluster {cluster_id}: {count:,} points ({percentage:.1f}%)")
        
        print(f"\nGenerated files:")
        print("  - demo_clustered_data.csv: Clustered dataset")
        print("  - demo_clustering_results.png: Visualization plots")
        print("  - kmeans_model.joblib: Trained model")
        print("  - scaler.joblib: Feature scaler")
        
        print("\n🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_quick_demo()
    
    if success:
        print("\n✨ The K-Means clustering system is working correctly!")
        print("You can now:")
        print("1. Run the full analysis with: python kmeans_electrofacies.py")
        print("2. Launch the web app with: streamlit run streamlit_kmeans_app.py")
        print("3. Or use the batch files: run_streamlit_app.bat")
    else:
        print("\n⚠️  Please check the error messages and fix any issues.")
