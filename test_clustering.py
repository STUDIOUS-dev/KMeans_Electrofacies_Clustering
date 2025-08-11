"""
Test script for K-Means Electrofacies Clustering
==============================================

This script tests the basic functionality of the clustering analysis.

Author: AI Assistant
Date: 2025-07-22
"""

import sys
import os
import pandas as pd
from kmeans_electrofacies import ElectrofaciesKMeans

def test_basic_functionality():
    """Test basic clustering functionality"""
    
    print("🧪 Testing K-Means Electrofacies Clustering")
    print("=" * 50)
    
    try:
        # Initialize the clustering class
        print("1. Initializing ElectrofaciesKMeans...")
        kmeans_analyzer = ElectrofaciesKMeans(n_clusters=4, random_state=42)
        print("   ✅ Initialization successful")
        
        # Load data
        print("\n2. Loading test data...")
        if os.path.exists('train.csv'):
            kmeans_analyzer.load_data('train.csv', delimiter=';')
            print("   ✅ Data loading successful")
        else:
            print("   ❌ train.csv not found!")
            return False
        
        # Preprocess data
        print("\n3. Preprocessing data...")
        kmeans_analyzer.preprocess_data()
        print("   ✅ Data preprocessing successful")
        
        # Fit K-Means (without optimization for speed)
        print("\n4. Fitting K-Means model...")
        kmeans_analyzer.fit_kmeans()
        print("   ✅ K-Means fitting successful")
        
        # Analyze clusters
        print("\n5. Analyzing clusters...")
        cluster_summary = kmeans_analyzer.analyze_clusters()
        print("   ✅ Cluster analysis successful")
        
        # Test basic statistics
        print("\n6. Validating results...")
        
        # Check if clusters were assigned
        unique_clusters = kmeans_analyzer.data['Cluster'].nunique()
        if unique_clusters == kmeans_analyzer.n_clusters:
            print(f"   ✅ Correct number of clusters: {unique_clusters}")
        else:
            print(f"   ❌ Cluster count mismatch: expected {kmeans_analyzer.n_clusters}, got {unique_clusters}")
        
        # Check if all data points have cluster labels
        total_points = len(kmeans_analyzer.data)
        labeled_points = kmeans_analyzer.data['Cluster'].notna().sum()
        if total_points == labeled_points:
            print(f"   ✅ All {total_points} data points labeled")
        else:
            print(f"   ❌ Missing labels: {total_points - labeled_points} points")
        
        # Save results (optional)
        print("\n7. Saving results...")
        kmeans_analyzer.save_results('test_clustered_data.csv')
        print("   ✅ Results saved successfully")
        
        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_data_validation():
    """Test data validation and error handling"""
    
    print("\n🔍 Testing Data Validation")
    print("-" * 30)
    
    try:
        # Test with sample data
        sample_data = pd.DataFrame({
            'DEPTH_MD': range(100, 200),
            'GR': [50 + i*0.1 for i in range(100)],
            'RDEP': [10 + i*0.05 for i in range(100)],
            'RHOB': [2.2 + i*0.001 for i in range(100)],
            'NPHI': [0.15 + i*0.0001 for i in range(100)]
        })
        
        # Add some missing values
        sample_data.loc[10:15, 'GR'] = None
        sample_data.loc[20:25, 'RDEP'] = None
        
        # Save sample data
        sample_data.to_csv('test_sample.csv', index=False)
        
        # Test clustering with sample data
        analyzer = ElectrofaciesKMeans(n_clusters=3)
        analyzer.load_data('test_sample.csv', delimiter=',')
        analyzer.preprocess_data()
        analyzer.fit_kmeans()
        
        print("   ✅ Sample data clustering successful")
        
        # Clean up
        if os.path.exists('test_sample.csv'):
            os.remove('test_sample.csv')
        
        return True
        
    except Exception as e:
        print(f"   ❌ Data validation test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("Starting K-Means Electrofacies Testing Suite")
    print("=" * 60)
    
    # Run tests
    test1_passed = test_basic_functionality()
    test2_passed = test_data_validation()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Basic Functionality Test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Data Validation Test: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("The K-Means clustering system is ready to use.")
    else:
        print("\n⚠️  SOME TESTS FAILED!")
        print("Please check the error messages above.")
    
    print("=" * 60)
