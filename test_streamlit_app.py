"""
Test script for the fixed Streamlit app
======================================

This script tests the core functionality of the Streamlit app components
without actually running the web interface.
"""

import pandas as pd
import numpy as np
from streamlit_kmeans_app import StreamlitKMeans

def test_streamlit_kmeans():
    """Test the StreamlitKMeans class"""
    
    print("🧪 Testing StreamlitKMeans Class")
    print("=" * 40)
    
    try:
        # Load sample data
        print("1. Loading test data...")
        data = pd.read_csv('train.csv', sep=';')
        
        # Sample for testing
        sample_data = data.sample(n=1000, random_state=42)
        print(f"   ✅ Loaded {len(sample_data)} sample rows")
        
        # Initialize clustering
        print("\n2. Initializing StreamlitKMeans...")
        kmeans = StreamlitKMeans(n_clusters=4, random_state=42)
        print("   ✅ Initialization successful")
        
        # Test preprocessing
        print("\n3. Testing data preprocessing...")
        selected_features = ['GR', 'RDEP', 'RHOB', 'NPHI']
        clean_points = kmeans.preprocess_data(sample_data, selected_features)
        print(f"   ✅ Preprocessed {clean_points} clean points")
        
        # Test optimal cluster finding
        print("\n4. Testing optimal cluster detection...")
        optimal_k, scores, cluster_range = kmeans.find_optimal_clusters(max_clusters=6)
        print(f"   ✅ Optimal clusters: {optimal_k}")
        print(f"   ✅ Silhouette scores: {[f'{s:.3f}' for s in scores]}")
        
        # Test K-Means fitting
        print("\n5. Testing K-Means fitting...")
        kmeans.n_clusters = optimal_k
        silhouette_avg = kmeans.fit_kmeans()
        print(f"   ✅ K-Means fitted with silhouette score: {silhouette_avg:.3f}")
        
        # Test cluster analysis
        print("\n6. Testing cluster analysis...")
        cluster_summary = kmeans.get_cluster_summary()
        print(f"   ✅ Generated summary for {len(cluster_summary)} clusters")
        
        # Display cluster info
        print("\n7. Cluster Summary:")
        for summary in cluster_summary:
            print(f"   Cluster {summary['cluster']}: {summary['size']} points ({summary['percentage']:.1f}%)")
        
        print("\n" + "=" * 40)
        print("🎉 ALL TESTS PASSED!")
        print("The StreamlitKMeans class is working correctly.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loading():
    """Test data loading functionality"""
    
    print("\n🔍 Testing Data Loading")
    print("-" * 30)
    
    try:
        # Test CSV loading with different delimiters
        data_semicolon = pd.read_csv('train.csv', sep=';')
        print(f"   ✅ Semicolon delimiter: {data_semicolon.shape}")
        
        # Check required columns
        required_cols = ['DEPTH_MD', 'GR', 'RDEP', 'RHOB', 'NPHI', 'PEF', 'DTC', 'CALI']
        available_cols = [col for col in required_cols if col in data_semicolon.columns]
        print(f"   ✅ Available features: {available_cols}")
        
        # Check data types
        numeric_cols = data_semicolon[available_cols].select_dtypes(include=[np.number]).columns
        print(f"   ✅ Numeric columns: {len(numeric_cols)}/{len(available_cols)}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Data loading test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Testing Fixed Streamlit App Components")
    print("=" * 60)
    
    # Run tests
    test1_passed = test_data_loading()
    test2_passed = test_streamlit_kmeans()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Data Loading Test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"StreamlitKMeans Test: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("The fixed Streamlit app should work correctly now.")
        print("\nTo run the app:")
        print("  streamlit run streamlit_kmeans_app.py")
    else:
        print("\n⚠️  SOME TESTS FAILED!")
        print("Please check the error messages above.")
    
    print("=" * 60)
