"""
Performance Cache Module for RAG System
Implements caching for embeddings, API responses, and query results
"""

import pickle
import hashlib
import os
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import numpy as np

@dataclass
class CacheEntry:
    """Represents a cached item with expiration"""
    data: Any
    timestamp: float
    ttl: float  # Time to live in seconds
    
    def is_expired(self) -> bool:
        return time.time() - self.timestamp > self.ttl

class PerformanceCache:
    """High-performance cache for RAG system operations"""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.max_memory_items = 1000  # Limit memory usage
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
    def _get_cache_key(self, prefix: str, data: Any) -> str:
        """Generate cache key from data"""
        if isinstance(data, str):
            content = data
        elif isinstance(data, dict):
            content = str(sorted(data.items()))
        elif isinstance(data, list):
            content = str(data)
        else:
            content = str(data)
            
        return f"{prefix}_{hashlib.md5(content.encode()).hexdigest()}"
    
    def get_embedding_cache(self, query: str) -> Optional[np.ndarray]:
        """Get cached embedding for query"""
        key = self._get_cache_key("embedding", query)
        
        # Check memory cache first
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            if not entry.is_expired():
                return entry.data
            else:
                del self.memory_cache[key]
        
        # Check disk cache
        cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    entry = pickle.load(f)
                    if not entry.is_expired():
                        # Store in memory for faster access
                        self.memory_cache[key] = entry
                        return entry.data
                    else:
                        os.remove(cache_file)
            except:
                pass
        
        return None
    
    def set_embedding_cache(self, query: str, embedding: np.ndarray, ttl: float = 3600):
        """Cache embedding for query (1 hour TTL)"""
        key = self._get_cache_key("embedding", query)
        entry = CacheEntry(embedding, time.time(), ttl)
        
        # Store in memory
        self.memory_cache[key] = entry
        
        # Store on disk
        cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(entry, f)
        except:
            pass
        
        # Cleanup memory if too large
        if len(self.memory_cache) > self.max_memory_items:
            # Remove oldest entries
            sorted_items = sorted(self.memory_cache.items(), 
                                key=lambda x: x[1].timestamp)
            for old_key, _ in sorted_items[:100]:
                del self.memory_cache[old_key]
    
    def get_news_cache(self, query: str) -> Optional[List[Dict]]:
        """Get cached news results"""
        key = self._get_cache_key("news", query)
        
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            if not entry.is_expired():
                return entry.data
            else:
                del self.memory_cache[key]
        
        return None
    
    def set_news_cache(self, query: str, results: List[Dict], ttl: float = 1800):
        """Cache news results (30 minutes TTL)"""
        key = self._get_cache_key("news", query)
        entry = CacheEntry(results, time.time(), ttl)
        self.memory_cache[key] = entry
    
    def get_rag_results_cache(self, query: str, params: Dict) -> Optional[List]:
        """Get cached RAG search results"""
        cache_data = {"query": query, "params": params}
        key = self._get_cache_key("rag", cache_data)
        
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            if not entry.is_expired():
                return entry.data
            else:
                del self.memory_cache[key]
        
        return None
    
    def set_rag_results_cache(self, query: str, params: Dict, results: List, ttl: float = 1800):
        """Cache RAG search results (30 minutes TTL)"""
        cache_data = {"query": query, "params": params}
        key = self._get_cache_key("rag", cache_data)
        entry = CacheEntry(results, time.time(), ttl)
        self.memory_cache[key] = entry
    
    def get_gemini_response_cache(self, prompt: str) -> Optional[str]:
        """Get cached Gemini response"""
        key = self._get_cache_key("gemini", prompt)
        
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            if not entry.is_expired():
                return entry.data
            else:
                del self.memory_cache[key]
        
        return None
    
    def set_gemini_response_cache(self, prompt: str, response: str, ttl: float = 3600):
        """Cache Gemini response (1 hour TTL)"""
        key = self._get_cache_key("gemini", prompt)
        entry = CacheEntry(response, time.time(), ttl)
        self.memory_cache[key] = entry
    
    def clear_expired(self):
        """Clear all expired cache entries"""
        expired_keys = []
        for key, entry in self.memory_cache.items():
            if entry.is_expired():
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.memory_cache[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_items = len(self.memory_cache)
        expired_items = sum(1 for entry in self.memory_cache.values() if entry.is_expired())
        
        return {
            "total_items": total_items,
            "active_items": total_items - expired_items,
            "expired_items": expired_items,
            "cache_directory": self.cache_dir
        }