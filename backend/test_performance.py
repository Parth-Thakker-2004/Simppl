"""
Performance Test Script for Optimized RAG System
Tests response times with and without caching
"""

import requests
import time
import json

def test_performance():
    """Test performance improvements with caching"""
    
    base_url = "http://localhost:5000"
    
    # Test queries
    test_queries = [
        "Who is Donald Trump?",
        "What are the current tariff policies?",
        "Tell me about climate change policies",
        "Who is responsible for the tariffs?",  # This should trigger image generation
        "What do people think about the economy?"
    ]
    
    print("🚀 Testing RAG System Performance")
    print("=" * 50)
    
    # Check if server is running
    try:
        response = requests.get(f"{base_url}/api/status", timeout=5)
        if response.status_code != 200:
            print("❌ Server not running. Please start the backend first.")
            return
    except requests.exceptions.RequestException:
        print("❌ Server not accessible. Please start the backend first.")
        return
    
    print("✅ Server is running\n")
    
    # Test each query twice to see caching benefits
    for i, query in enumerate(test_queries, 1):
        print(f"🔍 Test {i}: {query}")
        
        # First run (no cache)
        start_time = time.time()
        try:
            response = requests.post(
                f"{base_url}/api/chat",
                json={"query": query, "top_k": 5},
                timeout=30
            )
            first_run_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                processing_time = data.get("processing_time", "N/A")
                cached_ops = data.get("cached_operations", {})
                
                print(f"   First run: {first_run_time:.2f}s (Backend: {processing_time}s)")
                print(f"   Cached ops: {cached_ops}")
            else:
                print(f"   ❌ Error: {response.status_code}")
                continue
                
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Request failed: {e}")
            continue
        
        # Second run (with cache)
        start_time = time.time()
        try:
            response = requests.post(
                f"{base_url}/api/chat",
                json={"query": query, "top_k": 5},
                timeout=30
            )
            second_run_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                processing_time = data.get("processing_time", "N/A")
                cached_ops = data.get("cached_operations", {})
                
                print(f"   Second run: {second_run_time:.2f}s (Backend: {processing_time}s)")
                print(f"   Cached ops: {cached_ops}")
                
                # Calculate improvement
                if isinstance(processing_time, (int, float)) and processing_time > 0:
                    improvement = ((first_run_time - second_run_time) / first_run_time) * 100
                    print(f"   ⚡ Improvement: {improvement:.1f}%")
                
            else:
                print(f"   ❌ Error: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Request failed: {e}")
        
        print()
    
    # Get cache statistics
    try:
        response = requests.get(f"{base_url}/api/performance/cache", timeout=5)
        if response.status_code == 200:
            cache_stats = response.json()["cache_stats"]
            print("📊 Cache Statistics:")
            print(f"   Total items: {cache_stats['total_items']}")
            print(f"   Active items: {cache_stats['active_items']}")
            print(f"   Expired items: {cache_stats['expired_items']}")
        else:
            print("❌ Could not get cache statistics")
    except requests.exceptions.RequestException:
        print("❌ Could not connect to cache stats endpoint")

if __name__ == "__main__":
    test_performance()