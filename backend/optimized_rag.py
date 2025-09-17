import csv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import json
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import time
import random
from urllib.parse import unquote
from insight_graphs import GroqInsightGraphGenerator
from image_generator import ImageGenerator
from performance_cache import PerformanceCache
import ast

@dataclass
class SocialMediaChunk:
    """Represents a chunk of social media content for RAG with precomputed embeddings"""
    id: str
    platform: str
    content: str
    metadata: Dict[str, Any]
    embedding: np.ndarray

class OptimizedSocialMediaRAG:
    """Ultra-fast RAG system using precomputed embeddings from CSV files"""
    
    def __init__(self):
        self.chunks: List[SocialMediaChunk] = []
        self.embeddings_matrix: np.ndarray = None
        self.data_loaded = False
        self.cache = PerformanceCache()  # Add performance cache
        
    def _parse_embedding(self, embedding_str: str) -> np.ndarray:
        """Parse embedding string from CSV into numpy array"""
        if not embedding_str or embedding_str.strip() == '' or embedding_str.strip() == '[]':
            return np.array([])  # Return empty array for missing embeddings
        
        try:
            # Clean the string
            embedding_str = embedding_str.strip()
            
            # Try parsing as JSON first
            if embedding_str.startswith('[') and embedding_str.endswith(']'):
                embedding_list = json.loads(embedding_str)
                if isinstance(embedding_list, list) and len(embedding_list) > 0:
                    return np.array(embedding_list, dtype=np.float32)
                else:
                    return np.array([])  # Empty list
            
            # Try parsing as literal_eval (for Python list format)
            embedding_list = ast.literal_eval(embedding_str)
            if isinstance(embedding_list, list) and len(embedding_list) > 0:
                return np.array(embedding_list, dtype=np.float32)
            else:
                return np.array([])  # Empty list
            
        except json.JSONDecodeError:
            # Only print warning for truly unexpected formats, not empty arrays
            if embedding_str != '[]' and embedding_str != '':
                print(f"JSON decode error for embedding: {embedding_str[:50]}...")
            return np.array([])
        except (ValueError, SyntaxError):
            # Only print warning for truly unexpected formats
            if embedding_str != '[]' and embedding_str != '':
                print(f"Parse error for embedding: {embedding_str[:50]}...")
            return np.array([])
        except Exception as e:
            print(f"Unexpected error parsing embedding: {e}")
            return np.array([])
    
    def load_reddit_data_with_embeddings(self, csv_path: str = None):
        """Load Reddit data directly from CSV with precomputed embeddings"""
        if csv_path is None:
            data_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
            posts_path = os.path.join(data_folder, "posts Data Dump - Reddit.csv")
            comments_path = os.path.join(data_folder, "comments Data Dump - Reddit.csv")
        
        
        # Load posts
        posts_count = 0
        with open(posts_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                # Get post ID
                post_id = None
                for id_col in ['\ufeffid', 'id', 'post_id']:
                    if id_col in row and row.get(id_col, '').strip():
                        post_id = row[id_col].strip()
                        break
                
                if not post_id:
                    continue
                
                # Extract text content
                text_content = row.get('text', '')
                if text_content.startswith('{'):
                    try:
                        text_json = json.loads(text_content)
                        if 'title' in text_json:
                            text_content = text_json['title']
                    except:
                        pass
                
                # Extract keywords from external links to enhance searchability
                link_keywords = ""
                
                # Parse embedding
                embedding_str = row.get('embeddings_vec', '') or row.get('embeddings', '')
                if not embedding_str:
                    continue  # Skip rows without embeddings
                
                embedding = self._parse_embedding(embedding_str)
                if embedding.size == 0:
                    continue
                
                # Create chunk
                chunk = SocialMediaChunk(
                    id=f"reddit_post_{post_id}",
                    platform="Reddit",
                    content=f"Reddit Post by {row.get('username', 'Unknown')}: {text_content}{link_keywords}",
                    metadata={
                        "type": "post",
                        "username": row.get('username', ''),
                        "timestamp": row.get('timestamp', ''),
                        "engagement": row.get('engagement', ''),
                        "text_analysis": row.get('text_analysis', ''),
                        "original_id": post_id
                    },
                    embedding=embedding
                )
                
                self.chunks.append(chunk)
                posts_count += 1
        
        
        comments_count = 0
        
        with open(comments_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                # Parse embedding
                embedding_str = row.get('embeddings_vec', '') or row.get('embeddings', '')
                if not embedding_str:
                    continue
                
                embedding = self._parse_embedding(embedding_str)
                if embedding.size == 0:
                    continue
                
                # Create chunk
                chunk = SocialMediaChunk(
                    id=f"reddit_comment_{row.get('comment_id', comments_count)}",
                    platform="Reddit",
                    content=f"Reddit Comment by {row.get('username', 'Unknown')}: {row.get('text', '')}",
                    metadata={
                        "type": "comment",
                        "username": row.get('username', ''),
                        "date": row.get('date_of_comment', ''),
                        "post_id": row.get('post_id', ''),
                        "reactions": row.get('reactions', ''),
                        "text_analysis": row.get('text_analysis', '')
                    },
                    embedding=embedding
                )
                
                self.chunks.append(chunk)
                comments_count += 1
                
        
        
    def load_youtube_data_with_embeddings(self, csv_path: str = None):
        """Load YouTube data directly from CSV with precomputed embeddings"""
        if csv_path is None:
            data_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
            posts_path = os.path.join(data_folder, "posts Data Dump - Youtube.csv")
            comments_path = os.path.join(data_folder, "comments Data Dump - Youtube.csv")
        
        # Load posts
        posts_count = 0
        with open(posts_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                # Get post ID
                post_id = None
                for id_col in ['\ufeffid', 'id', 'post_id']:
                    if id_col in row and row.get(id_col, '').strip():
                        post_id = row[id_col].strip()
                        break
                
                if not post_id:
                    continue
                
                # Parse embedding
                embedding_str = row.get('embeddings_vec', '') or row.get('embeddings', '')
                if not embedding_str:
                    continue
                
                embedding = self._parse_embedding(embedding_str)
                if embedding.size == 0:
                    continue
                
                # Create chunk
                chunk = SocialMediaChunk(
                    id=f"youtube_post_{post_id}",
                    platform="YouTube",
                    content=f"YouTube Post by {row.get('username', 'Unknown')}: {row.get('text', '')}",
                    metadata={
                        "type": "post",
                        "username": row.get('username', ''),
                        "timestamp": row.get('timestamp', ''),
                        "link": row.get('link', ''),
                        "engagement": row.get('engagement', ''),
                        "reactions": row.get('reactions', ''),
                        "text_analysis": row.get('text_analysis', ''),
                        "original_id": post_id
                    },
                    embedding=embedding
                )
                
                self.chunks.append(chunk)
                posts_count += 1
                
        
        
        comments_count = 0
        
        with open(comments_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                # Parse embedding
                embedding_str = row.get('embeddings_vec', '') or row.get('embeddings', '')
                if not embedding_str:
                    continue
                
                embedding = self._parse_embedding(embedding_str)
                if embedding.size == 0:
                    continue
                
                # Create chunk
                chunk = SocialMediaChunk(
                    id=f"youtube_comment_{row.get('comment_id', comments_count)}",
                    platform="YouTube",
                    content=f"YouTube Comment by {row.get('username', 'Unknown')}: {row.get('text', '')}",
                    metadata={
                        "type": "comment",
                        "username": row.get('username', ''),
                        "date": row.get('date_of_comment', ''),
                        "post_id": row.get('post_id', ''),
                        "reactions": row.get('reactions', ''),
                        "text_analysis": row.get('text_analysis', '')
                    },
                    embedding=embedding
                )
                
                self.chunks.append(chunk)
                comments_count += 1
                
        
    
    def build_embeddings_matrix(self):
        """Build the embeddings matrix for fast similarity search"""
        if not self.chunks:
            raise ValueError("No data loaded. Call load_*_data_with_embeddings first.")
        
        
        # Stack all embeddings into a matrix
        embeddings_list = [chunk.embedding for chunk in self.chunks]
        self.embeddings_matrix = np.vstack(embeddings_list)
        
        self.data_loaded = True
        
        # Save the processed data for future fast loading
        self._save_processed_data()
    
    def _save_processed_data(self):
        """Save processed chunks and embeddings matrix for fast loading"""
        try:
            data_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
            
            # Save embeddings matrix
            embeddings_path = os.path.join(data_folder, "optimized_embeddings.npy")
            np.save(embeddings_path, self.embeddings_matrix)
            
            # Save chunks metadata (without embeddings to save space)
            chunks_data = []
            for chunk in self.chunks:
                chunks_data.append({
                    'id': chunk.id,
                    'platform': chunk.platform,
                    'content': chunk.content,
                    'metadata': chunk.metadata
                })
            
            chunks_path = os.path.join(data_folder, "optimized_chunks.json")
            with open(chunks_path, 'w', encoding='utf-8') as f:
                json.dump(chunks_data, f, ensure_ascii=False, indent=2)
            
            
        except Exception as e:
            print(f"Could not save processed data: {e}")
    
    def load_processed_data(self):
        """Load previously processed data for instant startup"""
        try:
            data_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
            embeddings_path = os.path.join(data_folder, "optimized_embeddings.npy")
            chunks_path = os.path.join(data_folder, "optimized_chunks.json")
            
            if not (os.path.exists(embeddings_path) and os.path.exists(chunks_path)):
                return False
            
            print("Loading preprocessed data...")
            
            # Load embeddings matrix
            self.embeddings_matrix = np.load(embeddings_path)
            
            # Load chunks
            with open(chunks_path, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            
            # Reconstruct chunks (embeddings will be referenced from matrix)
            self.chunks = []
            for i, chunk_data in enumerate(chunks_data):
                chunk = SocialMediaChunk(
                    id=chunk_data['id'],
                    platform=chunk_data['platform'],
                    content=chunk_data['content'],
                    metadata=chunk_data['metadata'],
                    embedding=self.embeddings_matrix[i]  # Reference to matrix row
                )
                self.chunks.append(chunk)
            
            self.data_loaded = True
            print(f"Loaded {len(self.chunks)} chunks with {self.embeddings_matrix.shape} embeddings matrix")
            return True
            
        except Exception as e:
            print(f"Could not load processed data: {e}")
            return False
    
    def enhance_query_with_gemini(self, user_query: str, gemini_model=None) -> str:
        """Use Gemini to enhance user query with relevant keywords for better RAG retrieval"""
        
        if not gemini_model:
            return user_query  # Return original if Gemini not available
        
        enhancement_prompt = f"""
You are a search query enhancement expert for social media content analysis. 

Original Question: "{user_query}"

Your task is to generate focused search keywords that will find RELEVANT social media posts and comments. 

Guidelines:
1. Keep the core topic/subject of the original question
2. Add related terms that people would actually use when discussing this topic on Reddit/YouTube
3. Include synonyms and alternative phrasings specific to the topic
4. Avoid generic terms that could match unrelated content
5. Focus on the MAIN SUBJECT MATTER of the question

For example:
- If asking about "tariffs", include: trade policy, import taxes, customs duties, trade war
- If asking about "AI regulation", include: artificial intelligence policy, tech regulation, AI governance
- If asking about "climate change", include: global warming, carbon emissions, environmental policy

Generate enhanced search terms that will specifically find content related to "{user_query}".

Return ONLY the enhanced keywords separated by spaces (max 30 words). Be precise and topic-focused.

Enhanced search query:"""

        try:
            response = gemini_model.generate_content(enhancement_prompt)
            enhanced_query = response.text.strip()
            
            # Clean up the response - remove quotes, extra formatting
            enhanced_query = enhanced_query.replace('"', '').replace('\n', ' ')
            enhanced_query = ' '.join(enhanced_query.split())  # Normalize whitespace
            
            print(f"Original query: {user_query}")
            print(f"Enhanced query: {enhanced_query}")
            
            return enhanced_query
            
        except Exception as e:
            print(f"Error enhancing query with Gemini: {e}")
            return user_query  # Fallback to original query
    def search(self, query_embedding: np.ndarray, top_k: int = 5, platform_filter: str = None, min_score: float = 0.05) -> List[Tuple[SocialMediaChunk, float]]:
        """Enhanced search using precomputed embeddings with dimension handling"""
        if not self.data_loaded:
            raise ValueError("Data must be loaded first")
        
        # Handle dimension mismatch by padding or truncating query embedding
        query_dim = query_embedding.shape[0]
        data_dim = self.embeddings_matrix.shape[1]
        
        if query_dim != data_dim:
            if query_dim < data_dim:
                # Pad with zeros
                padded_embedding = np.zeros(data_dim)
                padded_embedding[:query_dim] = query_embedding
                query_embedding = padded_embedding
            else:
                # Truncate
                query_embedding = query_embedding[:data_dim]
        
        # Compute similarities with all embeddings at once
        similarities = cosine_similarity([query_embedding], self.embeddings_matrix)[0]
        
        # Get all indices sorted by similarity
        all_indices = np.argsort(similarities)[::-1]
        
        results = []
        seen_content = set()  # To avoid duplicate content
        
        for idx in all_indices:
            chunk = self.chunks[idx]
            score = similarities[idx]
            
            # Skip if similarity score is too low
            if score < min_score:
                break
            
            # Apply platform filter if specified
            if platform_filter and chunk.platform.lower() != platform_filter.lower():
                continue
            
            # Check for content diversity (avoid very similar content)
            content_key = chunk.content[:100].lower().strip()
            if content_key in seen_content:
                continue
            seen_content.add(content_key)
            
            # Additional quality filters
            if self._is_quality_content(chunk):
                results.append((chunk, score))
                
                if len(results) >= top_k:
                    break
        
        # If we don't have enough results, relax the constraints further
        if len(results) < top_k // 2 and min_score > 0.02:
            return self.search(query_embedding, top_k, platform_filter, min_score=0.02)
                
        return results
    
    def _is_quality_content(self, chunk: SocialMediaChunk) -> bool:
        """Filter for quality content"""
        content = chunk.content.lower()
        
        # Skip very short content
        if len(content.strip()) < 20:
            return False
        
        # Skip content that's mostly punctuation or numbers
        alpha_chars = sum(c.isalpha() for c in content)
        if alpha_chars < len(content) * 0.5:
            return False
        
        # Skip common spam patterns
        spam_patterns = [
            'click here', 'buy now', 'subscribe', 'like and share',
            'first!', 'early!', 'notification squad'
        ]
        
        for pattern in spam_patterns:
            if pattern in content:
                return False
        
        return True
    
    def search_with_query_expansion(self, query_embedding: np.ndarray, original_query: str, top_k: int = 5, platform_filter: str = None) -> List[Tuple[SocialMediaChunk, float]]:
        """Search with query expansion for better retrieval"""
        
        # Get initial results
        initial_results = self.search(query_embedding, top_k * 2, platform_filter, min_score=0.1)
        
        if len(initial_results) < top_k:
            # If we don't have enough good results, try different approaches
            
            # 1. Try with lower threshold
            backup_results = self.search(query_embedding, top_k, platform_filter, min_score=0.05)
            
            # 2. Try searching both platforms if filter was applied
            if platform_filter:
                cross_platform_results = self.search(query_embedding, top_k, None, min_score=0.1)
                backup_results.extend(cross_platform_results)
            
            # Combine and deduplicate
            all_results = initial_results + backup_results
            seen_ids = set()
            final_results = []
            
            for chunk, score in all_results:
                if chunk.id not in seen_ids:
                    seen_ids.add(chunk.id)
                    final_results.append((chunk, score))
                    if len(final_results) >= top_k:
                        break
            
            return final_results
        
        return initial_results[:top_k]
    
    def generate_context(self, search_results: List[Tuple[SocialMediaChunk, float]]) -> str:
        """Generate enhanced context string from search results"""
        if not search_results:
            return ""
        
        context_parts = []
        
        # Group results by platform and type for better organization
        reddit_posts = []
        reddit_comments = []
        youtube_posts = []
        youtube_comments = []
        
        for chunk, score in search_results:
            metadata = chunk.metadata
            
            # Add relevance score and enhanced metadata
            enhanced_content = f"[{chunk.platform} {metadata['type']} - Relevance: {score:.3f}] {chunk.content}"
            
            # Add contextual metadata
            context_info = []
            if metadata.get('timestamp'):
                context_info.append(f"Posted: {metadata['timestamp']}")
            if metadata.get('engagement'):
                context_info.append(f"Engagement: {metadata['engagement']}")
            if metadata.get('reactions'):
                context_info.append(f"Reactions: {metadata['reactions']}")
            if metadata.get('text_analysis'):
                context_info.append(f"Sentiment: {metadata['text_analysis']}")
            
            if context_info:
                enhanced_content += f" ({', '.join(context_info)})"
            
            # Categorize by platform and type
            if chunk.platform == "Reddit":
                if metadata['type'] == 'post':
                    reddit_posts.append(enhanced_content)
                else:
                    reddit_comments.append(enhanced_content)
            else:  # YouTube
                if metadata['type'] == 'post':
                    youtube_posts.append(enhanced_content)
                else:
                    youtube_comments.append(enhanced_content)
        
        # Build organized context
        if reddit_posts:
            context_parts.append("=== REDDIT POSTS ===")
            context_parts.extend(reddit_posts)
        
        if reddit_comments:
            context_parts.append("\n=== REDDIT COMMENTS ===")
            context_parts.extend(reddit_comments)
        
        if youtube_posts:
            context_parts.append("\n=== YOUTUBE POSTS ===")
            context_parts.extend(youtube_posts)
        
        if youtube_comments:
            context_parts.append("\n=== YOUTUBE COMMENTS ===")
            context_parts.extend(youtube_comments)
        
        return "\n\n".join(context_parts)

    def filter_relevant_results(self, results: List[Tuple[SocialMediaChunk, float]], original_query: str, query_encoder=None, relevance_threshold: float = 0.3) -> List[Tuple[SocialMediaChunk, float]]:
        """Filter results to ensure they're actually relevant to the original query"""
        if not query_encoder or not results:
            return results
        
        try:
            # Encode the original query
            original_query_embedding = query_encoder.encode([original_query.lower()])[0]
            
            filtered_results = []
            for chunk, score in results:
                # Encode the chunk content
                chunk_embedding = query_encoder.encode([chunk.content[:500].lower()])[0]  # Use first 500 chars
                
                # Calculate semantic similarity between original query and chunk content
                semantic_similarity = cosine_similarity([original_query_embedding], [chunk_embedding])[0][0]
                
                # Only keep results that are semantically similar to the original query
                if semantic_similarity >= relevance_threshold:
                    filtered_results.append((chunk, score))
                    print(f"Relevant (sim: {semantic_similarity:.3f}): {chunk.content[:100]}...")
                else:
                    print(f"Filtered out (sim: {semantic_similarity:.3f}): {chunk.content[:100]}...")
            
            print(f"Filtered {len(results)} → {len(filtered_results)} relevant results")
            return filtered_results
            
        except Exception as e:
            print(f"Error in relevance filtering: {e}")
            return results  # Return original results if filtering fails

class OnlineSearchAgent:
    """Agent to fetch information from online sources"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def search_news(self, query: str, max_results: int = 3) -> List[Dict[str, str]]:
        """Search for news articles related to the query"""
        try:
            # Enhanced search query focusing on major news sites with a wider net
            news_sites = [
                "reuters.com", "bbc.com", "cnn.com", "apnews.com", "nytimes.com", 
                "theguardian.com", "washingtonpost.com", "bloomberg.com", "ft.com",
                "wsj.com", "economist.com", "npr.org", "cbsnews.com", "nbcnews.com",
                "abcnews.go.com", "aljazeera.com", "time.com", "forbes.com", "cnbc.com",
                "thehindu.com", "indianexpress.com", "ndtv.com", "timesofindia.indiatimes.com"
            ]
            
            # Create a site-specific search query
            site_query = " OR ".join([f"site:{site}" for site in news_sites[:10]])  # Limit to avoid overly long query
            search_url = f"https://duckduckgo.com/html/?q={query}+news+({site_query})"
            
            response = self.session.get(search_url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            results = []
            news_links = soup.find_all('a', {'class': 'result__a'})[:max_results+5]  # Fetch extra to account for possible failures
            
            for link in news_links:
                title = link.get_text().strip()
                url = link.get('href', '')
                if url.startswith('/l/?uddg='):
                    url = url.split('uddg=')[1] if 'uddg=' in url else url
                    # URL decode
                    url = unquote(url)
                
                # Skip if not from a news site
                if not any(news_site in url for news_site in news_sites):
                    continue
                    
                try:
                    # Extract publisher from URL
                    domain = url.split('//')[1].split('/')[0]
                    publisher = domain.replace('www.', '').split('.')[0].capitalize()
                    if publisher in ['Nytimes', 'Wsj', 'Npr', 'Bbc', 'Cnn']:
                        publisher_map = {
                            'Nytimes': 'The New York Times',
                            'Wsj': 'Wall Street Journal',
                            'Npr': 'NPR',
                            'Bbc': 'BBC News',
                            'Cnn': 'CNN'
                        }
                        publisher = publisher_map.get(publisher, publisher)
                        
                    # Get snippet and try to extract date
                    snippet, pub_date = self._extract_article_snippet_and_date(url)
                    
                    if snippet and len(snippet) > 50:  # Ensure we have a valid snippet
                        results.append({
                            'title': title,
                            'url': url,
                            'snippet': snippet,
                            'type': 'news',
                            'publisher': publisher,
                            'publication_date': pub_date
                        })
                    
                    if len(results) >= max_results:
                        break
                        
                except Exception as e:
                    print(f"Error processing news item {url}: {e}")
                    continue
                
                time.sleep(0.3)  # Gentle throttling
            
            return results
            
        except Exception as e:
            print(f"Error searching news: {e}")
            return self._get_synthetic_news(query, max_results)
    
    def _extract_article_snippet_and_date(self, url: str, max_length: int = 300) -> Tuple[str, str]:
        """Extract a snippet and publication date from a news article"""
        try:
            response = self.session.get(url, timeout=8)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract snippet
            article_selectors = [
                'article', '.article-body', '.story-body', 
                '.content', '.post-content', 'main', 
                '.article__content', '.article-text'
            ]
            
            snippet = ""
            for selector in article_selectors:
                article_content = soup.select_one(selector)
                if article_content:
                    paragraphs = article_content.find_all('p')
                    if paragraphs:
                        snippet = ' '.join([p.get_text().strip() for p in paragraphs[:3]])
                        if snippet:
                            break
            
            if not snippet:
                # Fallback to all paragraphs
                paragraphs = soup.find_all('p')
                if paragraphs:
                    snippet = ' '.join([p.get_text().strip() for p in paragraphs[:3]])
            
            # Clean and truncate snippet
            if snippet:
                # Remove multiple spaces
                snippet = ' '.join(snippet.split())
                # Truncate if needed
                if len(snippet) > max_length:
                    snippet = snippet[:max_length] + '...'
            
            # Try to extract publication date
            pub_date = ""
            date_selectors = [
                'meta[property="article:published_time"]',
                'meta[name="publishdate"]',
                'meta[name="date"]',
                'time',
                '.date',
                '.publish-date',
                '.published',
                '.article-date',
                'meta[name="DC.date.issued"]'
            ]
            
            for selector in date_selectors:
                date_elem = soup.select_one(selector)
                if date_elem:
                    if selector.startswith('meta'):
                        pub_date = date_elem.get('content', '')
                    else:
                        pub_date = date_elem.get_text().strip()
                        if date_elem.get('datetime'):
                            pub_date = date_elem.get('datetime')
                    if pub_date:
                        break
            
            # Basic cleaning of date format
            if pub_date:
                try:
                    # Try common date formats
                    if len(pub_date) >= 10:
                        parsed_date = datetime.strptime(pub_date[:10], '%Y-%m-%d')
                        pub_date = parsed_date.strftime('%Y-%m-%d')
                except Exception:
                    # Keep original format if parsing fails
                    pass
            
            return snippet, pub_date
            
        except Exception as e:
            print(f"Error extracting article content: {e}")
            return "", ""

class OptimizedConversationalAgent:
    """Optimized conversational agent using precomputed embeddings"""
    
    def __init__(self, rag_system: OptimizedSocialMediaRAG, gemini_api_key: str = None):
        self.rag = rag_system
        self.online_search = OnlineSearchAgent()
        self.query_encoder = None
        self.embedding_dimension = None
        self.graph_generator = GroqInsightGraphGenerator()
        self.image_generator = ImageGenerator()
        
        # Initialize conversation context
        self.conversation_history = []
        self.max_history_length = 15  # Keep last 15 exchanges (increased from 10)
        
        # Initialize Gemini first
        if gemini_api_key:
            try:
                genai.configure(api_key=gemini_api_key)
                self.model = genai.GenerativeModel('gemini-2.5-flash')
            except Exception as e:
                self.model = None
        else:
            self.model = None
        
    def initialize_query_encoder(self):
        """Initialize query encoder after we know the embedding dimension"""
        if self.query_encoder is not None:
            return True
            
        if not self.rag.data_loaded or self.rag.embeddings_matrix is None:
            print("RAG system not loaded")
            return False
            
        # Detect embedding dimension from the data
        self.embedding_dimension = self.rag.embeddings_matrix.shape[1]
        
        # Try to load query encoder with better error handling
        try:
            from sentence_transformers import SentenceTransformer
            
            # Use the best 1024D model to match precomputed embeddings
            model_name = 'sentence-transformers/all-roberta-large-v1'
            
            self.query_encoder = SentenceTransformer(model_name)
            
            # Verify dimensions match
            test_embedding = self.query_encoder.encode(["test"])
            return True
            
        except ImportError:
            print("sentence-transformers not installed. Install with: pip install sentence-transformers")
            return False
        except Exception as e:
            print(f"Could not load query encoder: {e}")
            return False
    
    def add_to_conversation_history(self, user_query: str, ai_response: str):
        """Add user query and AI response to conversation history"""
        self.conversation_history.append({"role": "user", "content": user_query})
        self.conversation_history.append({"role": "assistant", "content": ai_response})
        
        # Keep only the last max_history_length exchanges
        if len(self.conversation_history) > self.max_history_length * 2:
            self.conversation_history = self.conversation_history[-(self.max_history_length * 2):]
    
    def get_conversation_context(self) -> str:
        """Format conversation history for use in prompts"""
        if not self.conversation_history:
            return ""
        
        context_parts = []
        context_parts.append("=== CONVERSATION HISTORY ===")
        
        # Include up to 10 exchanges (20 messages) in the context
        for exchange in self.conversation_history[-20:]:  # Increased from last 3 exchanges
            if exchange["role"] == "user":
                context_parts.append(f"User: {exchange['content']}")
            else:
                # Truncate very long responses but keep more content
                context_parts.append(f"Assistant: {exchange['content'][:500]}...")
        
        context_parts.append("=== END CONVERSATION HISTORY ===")
        return "\n".join(context_parts)
    
    def clear_conversation_history(self):
        """Clear all conversation history"""
        self.conversation_history = []
        
    def chat(self, user_query: str, platform_filter: str = None, top_k: int = 5, use_online: bool = True) -> Dict[str, Any]:
        """Enhanced chat interaction with optimized RAG and performance caching"""
        
        start_time = time.time()
        
        # Initialize query encoder if not done yet
        if not self.query_encoder:
            if not self.initialize_query_encoder():
                return {
                    "error": "Query encoder not available. Please ensure sentence-transformers is installed and embeddings are loaded.",
                    "response": "I'm sorry, but the query encoder is not available. Please contact the administrator.",
                    "sources": [],
                    "rag_context": "",
                    "online_sources": [],
                    "num_social_results": 0,
                    "num_news_results": 0
                }

        try:
            
            # 1. Check cache for query embedding first
            query_for_cache = f"{user_query}_{platform_filter or 'all'}"
            cached_embedding = self.rag.cache.get_embedding_cache(query_for_cache)
            
            if cached_embedding is not None:
                print("Using cached embedding")
                enhanced_query_embedding = cached_embedding
                enhanced_query = user_query  # Skip enhancement for cached queries
            else:
                # Enhance query using Gemini for better RAG retrieval
                enhanced_query = self.rag.enhance_query_with_gemini(user_query, self.model)
                enhanced_query_embedding = self.query_encoder.encode([enhanced_query])[0]
                # Cache the embedding
                self.rag.cache.set_embedding_cache(query_for_cache, enhanced_query_embedding)
            
            # 2. Check cache for RAG results
            rag_params = {"top_k": top_k, "platform_filter": platform_filter}
            cached_rag_results = self.rag.cache.get_rag_results_cache(user_query, rag_params)
            
            if cached_rag_results is not None:
                rag_results = cached_rag_results
            else:
                # Search RAG system using enhanced query
                initial_results = self.rag.search(
                    enhanced_query_embedding, top_k=top_k*2, platform_filter=platform_filter
                )
                
                # Filter results for relevance to original query
                rag_results = self.rag.filter_relevant_results(
                    initial_results, user_query, self.query_encoder, relevance_threshold=0.10
                )[:top_k]
                
                # Cache the results
                self.rag.cache.set_rag_results_cache(user_query, rag_params, rag_results)
            
            rag_context = self.rag.generate_context(rag_results) if rag_results else ""
            
            # 3. Check cache for news results (only if online search is enabled)
            online_results = []
            if use_online:
                cached_news = self.rag.cache.get_news_cache(user_query)
                if cached_news is not None:
                    online_results = cached_news
                else:
                    try:
                        online_results = self.online_search.search_news(user_query, max_results=3)
                        # Cache the news results
                        self.rag.cache.set_news_cache(user_query, online_results)
                    except Exception as e:
                        print(f"Online search failed: {e}")
                        online_results = self.online_search._get_synthetic_news(user_query, max_results=2)
            
            # 4. Generate images/graphs (skip for similar recent queries to save time)
            generated_image = None
            graphs = []
            
            try:
                # Only generate images for person/visual queries
                query_lower = user_query.lower()
                visual_keywords = ['who is', 'picture', 'image', 'photo', 'looks like', 'appearance']
                if any(keyword in query_lower for keyword in visual_keywords):
                    generated_image = self.image_generator.generate_image_for_query(user_query, self.model)
                    
                
                # Only generate graphs if we have enough data and it's explicitly requested
                graph_keywords = ['chart', 'graph', 'analysis', 'trend', 'comparison', 'data', 'statistics']
                if any(keyword in query_lower for keyword in graph_keywords):
                    graphs = self.graph_generator.generate_insights_for_query(
                        user_query, rag_results, self.model
                    )
                        
            except Exception as e:
                print(f"Error with image/graph generation: {e}")
                generated_image = None
                graphs = []
            
            # 4. Create comprehensive context
            context_parts = []
            
            if rag_context:
                context_parts.append("=== SOCIAL MEDIA DISCUSSIONS ===")
                context_parts.append(rag_context)
            
            if online_results:
                context_parts.append("\n=== RECENT NEWS & INFORMATION ===")
                for result in online_results:
                    context_parts.append(f"[{result['title']}]")
                    context_parts.append(f"Source: {result['url']}")
                    context_parts.append(f"Content: {result['snippet']}")
                    context_parts.append("---")
            
            full_context = "\n".join(context_parts)
            
            # 5. Create enhanced prompt for Gemini with conversation context
            conversation_context = self.get_conversation_context()
            
            enhanced_prompt = f"""
You are an intelligent politics assistant that provides comprehensive, balanced answers by analyzing both social media discussions and current news sources. You maintain conversation context to provide coherent, contextual responses.

IMPORTANT GUIDELINES:
1. ALWAYS prioritize information from NEWS SOURCES over social media
2. Cite your sources clearly, especially news publishers
3. Maintain a balanced, factual tone
4. Use news sources as primary evidence for your claims
5. When available, reference publication dates from news sources
6. Provide nuanced analysis that considers multiple perspectives

{conversation_context}

CURRENT USER QUESTION: {user_query}

AVAILABLE CONTEXT:
{full_context}

INSTRUCTIONS:
1. CONVERSATION HISTORY: Use the complete conversation history to understand context and follow-up questions
2. DIRECT ANSWERS: Directly address the CURRENT USER QUESTION above
3. CONTINUITY: Maintain coherent narrative between questions - reference previous information when relevant
4. BALANCED PERSPECTIVE: Always incorporate BOTH social media insights AND news sources in your response
5. NEWS PRIORITY: Explicitly cite news sources and their findings in your response
6. FACTS FIRST: Lead with factual information from news sources, then add social media sentiment
7. CLEAR ATTRIBUTION: Clearly indicate when information comes from news versus social media
8. OBJECTIVITY: Present multiple perspectives when available
7. CITATIONS: Cite specific sources when making claims
8. OBJECTIVITY: Be objective and acknowledge different viewpoints
9. FOLLOW-UPS: For follow-up questions, explicitly reference previous context

Note: The context above was retrieved using enhanced search terms, but your response should directly address the original user question.

Please provide a comprehensive response that addresses the user's question using both social media insights and current information, while maintaining conversation continuity.

I want you return the result in the form of a paragraph. Don't include your opinions on the data provided. Use it if it is relevant to the user's question.
"""
            
            # 6. Generate response using Gemini
            try:
                if self.model:
                    response = self.model.generate_content(enhanced_prompt)
                    ai_response = response.text
                else:
                    ai_response = self._generate_fallback_response(user_query, rag_context, online_results)
                    
            except Exception as e:
                print(f"Error with Gemini API: {e}")
                ai_response = self._generate_fallback_response(user_query, rag_context, online_results)
            
            # 7. Create enhanced sources list
            sources = []
            
            for chunk, score in rag_results:
                # Generate a mock URL for social media content using available metadata
                platform = chunk.platform.lower()
                content_type = chunk.metadata["type"].lower()
                
                # For comments, use the post_id rather than comment_id if available
                post_id = chunk.metadata.get("post_id", chunk.id)
                username = chunk.metadata.get("username", "user")
                
                # Create platform-specific URL formats
                if platform == "reddit":
                    if content_type == "comment":
                        # For Reddit comments, link to the parent post
                        subreddit = chunk.metadata.get('subreddit', 'all')
                        url = f"https://www.reddit.com/r/{subreddit}/comments/{post_id}"
                    else:
                        url = f"https://www.reddit.com/r/{chunk.metadata.get('subreddit', 'all')}/comments/{post_id}"
                elif platform == "youtube":
                    # For YouTube, always link to the video, not the comment
                    url = f"https://www.youtube.com/watch?v={post_id}"
                elif platform == "twitter" or platform == "x":
                    url = f"https://twitter.com/{username}/status/{post_id}"
                else:
                    url = f"https://{platform}.com/{username}/{content_type}/{post_id}"
                
                sources.append({
                    "type": "social_media",
                    "platform": chunk.platform,
                    "content_type": chunk.metadata["type"],
                    "username": chunk.metadata.get("username", "Unknown"),
                    "score": float(score),
                    "preview": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                    "url": url
                })
            
            for result in online_results:
                sources.append({
                    "type": "news",
                    "title": result["title"],
                    "url": result["url"],
                    "preview": result["snippet"]
                })
            
            return {
                "response": ai_response,
                "sources": sources,
                "rag_context": rag_context,
                "online_sources": online_results,
                "graphs": graphs,
                "generated_image": generated_image,
                "num_social_results": len(rag_results),
                "num_news_results": len(online_results),
                "num_graphs": len(graphs),
                "processing_time": round(time.time() - start_time, 2),
                "cached_operations": {
                    "embedding": cached_embedding is not None,
                    "rag_results": cached_rag_results is not None,
                    "news": cached_news is not None if use_online else False
                }
            }
            
        except Exception as e:
            print(f"Error in chat method: {e}")
            return {
                "error": f"An error occurred: {str(e)}",
                "response": "I'm sorry, but I encountered an error while processing your request. Please try again.",
                "sources": [],
                "rag_context": "",
                "online_sources": [],
                "num_social_results": 0,
                "num_news_results": 0
            }
    
    def _generate_fallback_response(self, query: str, rag_context: str, online_results: List[Dict]) -> str:
        """Generate a basic response when Gemini is not available"""
        response_parts = []
        
        response_parts.append(f"Based on your question: '{query}'\n")
        
        if rag_context:
            response_parts.append("**Social Media Insights:**")
            response_parts.append(rag_context[:1000] + "..." if len(rag_context) > 1000 else rag_context)
        
        if online_results:
            response_parts.append("\n**Recent News & Information:**")
            for result in online_results:
                response_parts.append(f"• {result['title']}")
                response_parts.append(f"  {result['snippet'][:200]}...")
        
        if not rag_context and not online_results:
            response_parts.append("I couldn't find specific information about your query in either social media discussions or recent news.")
        
        return "\n\n".join(response_parts)

# Flask API with optimized RAG
app = Flask(__name__)
CORS(app)

# Global optimized RAG system
optimized_rag_system = None
optimized_agent = None
initialization_status = {
    'initialized': False,
    'loading': False,
    'error': None
}

def initialize_optimized_rag_system():
    """Initialize optimized RAG system with precomputed embeddings"""
    global optimized_rag_system, optimized_agent, initialization_status
    
    if initialization_status['initialized']:
        return True
    
    if initialization_status['loading']:
        return False
    
    try:
        initialization_status['loading'] = True
        
        optimized_rag_system = OptimizedSocialMediaRAG()
        
        # Try to load preprocessed data first
        if optimized_rag_system.load_processed_data():
            print("Loaded preprocessed data - instant startup!")
        else:
            print("Loading data from CSV files with embeddings...")
            optimized_rag_system.load_reddit_data_with_embeddings()
            optimized_rag_system.load_youtube_data_with_embeddings()
            optimized_rag_system.build_embeddings_matrix()
        
        optimized_agent = OptimizedConversationalAgent(
            optimized_rag_system, 
            gemini_api_key=os.getenv("GEMINI_API_KEY")  # Use environment variable for security
        )
        
        # Initialize the query encoder now that we have the data loaded
        if not optimized_agent.initialize_query_encoder():
            raise Exception("Failed to initialize query encoder with correct dimensions")
        
        initialization_status['initialized'] = True
        initialization_status['loading'] = False
        initialization_status['error'] = None
        
        print("RAG system initialized successfully!")
        return True
        
    except Exception as e:
        initialization_status['loading'] = False
        initialization_status['error'] = str(e)
        print(f"Failed to initialize RAG system: {e}")
        return False

@app.route('/api/status', methods=['GET'])
def status():
    """Get initialization status"""
    return jsonify(initialization_status)

@app.route('/api/initialize', methods=['POST'])
def initialize():
    """Initialize the optimized RAG system"""
    if initialization_status['initialized']:
        return jsonify({
            "status": "success", 
            "message": "Optimized RAG system already initialized",
            "cached": True
        })
    
    success = initialize_optimized_rag_system()
    
    if success:
        return jsonify({
            "status": "success", 
            "message": "Optimized RAG system initialized successfully",
            "cached": False
        })
    else:
        return jsonify({
            "status": "error", 
            "message": initialization_status['error']
        }), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Enhanced chat endpoint with optimized RAG"""
    if optimized_agent is None:
        return jsonify({"error": "Optimized RAG system not initialized. Call /api/initialize first"}), 400
    
    data = request.json
    query = data.get('query', '')
    platform_filter = data.get('platform_filter')
    top_k = data.get('top_k', 5)
    use_online = data.get('use_online', True)
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    try:
        result = optimized_agent.chat(query, platform_filter=platform_filter, top_k=top_k, use_online=use_online)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/search', methods=['POST'])
def search():
    """Direct search endpoint with optimized RAG"""
    try:
        if optimized_rag_system is None or not optimized_rag_system.data_loaded:
            return jsonify({"error": "Optimized RAG system not initialized"}), 400
        
        data = request.json
        query = data.get('query', '')
        platform_filter = data.get('platform_filter')
        top_k = data.get('top_k', 10)
        
        if not query:
            return jsonify({"error": "Query is required"}), 400
        
        # Check if query encoder is available
        if not optimized_agent or not optimized_agent.query_encoder:
            # Try to initialize query encoder
            if optimized_agent and not optimized_agent.initialize_query_encoder():
                return jsonify({"error": "Query encoder not available"}), 500
        
        # Encode query
        query_embedding = optimized_agent.query_encoder.encode([query])[0]
        
        results = optimized_rag_system.search(query_embedding, top_k=top_k, platform_filter=platform_filter)
        
        formatted_results = []
        for chunk, score in results:
            formatted_results.append({
                "id": chunk.id,
                "platform": chunk.platform,
                "content": chunk.content,
                "score": float(score),
                "metadata": chunk.metadata
            })
        
        return jsonify({"results": formatted_results})
        
    except Exception as e:
        print(f"Error in search endpoint: {e}")
        return jsonify({"error": f"Search failed: {str(e)}"}), 500

@app.route('/api/conversation/clear', methods=['POST'])
def clear_conversation():
    """Clear conversation history"""
    if optimized_agent is None:
        return jsonify({"error": "Optimized RAG system not initialized"}), 400
    
    try:
        optimized_agent.clear_conversation_history()
        return jsonify({"success": True, "message": "Conversation history cleared successfully"})
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500

@app.route('/api/conversation/history', methods=['GET'])
def get_conversation_history():
    """Get current conversation history"""
    if optimized_agent is None:
        return jsonify({"error": "Optimized RAG system not initialized"}), 400
    
    try:
        return jsonify({
            "conversation_history": optimized_agent.conversation_history,
            "history_length": len(optimized_agent.conversation_history)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/performance/cache', methods=['GET'])
def get_cache_stats():
    """Get performance cache statistics"""
    if optimized_rag_system is None:
        return jsonify({"error": "Optimized RAG system not initialized"}), 400
    
    try:
        stats = optimized_rag_system.cache.get_stats()
        return jsonify({
            "cache_stats": stats,
            "status": "active"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/performance/cache/clear', methods=['POST'])
def clear_cache():
    """Clear performance cache"""
    if optimized_rag_system is None:
        return jsonify({"error": "Optimized RAG system not initialized"}), 400
    
    try:
        old_stats = optimized_rag_system.cache.get_stats()
        optimized_rag_system.cache.clear_expired()
        optimized_rag_system.cache.memory_cache.clear()
        new_stats = optimized_rag_system.cache.get_stats()
        
        return jsonify({
            "message": "Cache cleared successfully",
            "old_stats": old_stats,
            "new_stats": new_stats
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting optimized RAG system with precomputed embeddings...")
    
    success = initialize_optimized_rag_system()
    
    if success:
        print("Optimized RAG system ready!")
        print("Server will be available at: http://127.0.0.1:5000")
        print("Open frontend/index.html in your browser to start chatting")
        app.run(debug=True, port=5000)
    else:
        print("Failed to initialize optimized RAG system. Check your data files.")
        print(f"Error: {initialization_status['error']}")
        exit(1)