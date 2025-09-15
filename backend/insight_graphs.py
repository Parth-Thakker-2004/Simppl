import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import base64
from io import BytesIO
from collections import Counter, defaultdict
import re
from typing import List, Dict, Any, Tuple
import matplotlib.dates as mdates
from wordcloud import WordCloud

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")

class InsightGraphGenerator:
    """Generate insight-driven visualizations from social media data"""
    
    def __init__(self):
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8',
            'reddit': '#ff4500',
            'youtube': '#ff0000'
        }
    
    def generate_insights_for_query(self, query: str, rag_results: List[Tuple], gemini_model=None) -> List[Dict[str, Any]]:
        """Generate multiple insight-driven graphs for a query"""
        
        if not rag_results:
            return []
        
        # Extract data from rag_results
        data = self._extract_data_from_results(rag_results)
        
        if not data:
            return []
        
        graphs = []
        
        try:
            # 1. Sentiment Evolution Analysis
            sentiment_graph = self._create_sentiment_evolution_graph(data, query)
            if sentiment_graph:
                graphs.append(sentiment_graph)
            
            # 2. Platform Engagement Comparison
            engagement_graph = self._create_platform_engagement_analysis(data, query)
            if engagement_graph:
                graphs.append(engagement_graph)
            
            # 3. Topic Sentiment Heatmap
            sentiment_heatmap = self._create_sentiment_topic_heatmap(data, query)
            if sentiment_heatmap:
                graphs.append(sentiment_heatmap)
            
            # 4. Engagement vs Sentiment Correlation
            correlation_graph = self._create_engagement_sentiment_correlation(data, query)
            if correlation_graph:
                graphs.append(correlation_graph)
            
            # 5. Content Type Distribution with Insights
            content_analysis = self._create_content_type_analysis(data, query)
            if content_analysis:
                graphs.append(content_analysis)
                
        except Exception as e:
            print(f"Error generating graphs: {e}")
        
        return graphs
    
    def _extract_data_from_results(self, rag_results: List[Tuple]) -> List[Dict]:
        """Extract structured data from RAG results"""
        data = []
        
        for chunk, score in rag_results:
            try:
                # Parse metadata for additional insights
                metadata = chunk.metadata if hasattr(chunk, 'metadata') else {}
                
                # Extract sentiment if available
                sentiment_data = {}
                if 'text_analysis' in metadata:
                    try:
                        analysis = json.loads(metadata['text_analysis']) if isinstance(metadata['text_analysis'], str) else metadata['text_analysis']
                        if 'Sentiment' in analysis:
                            sentiment_data = analysis['Sentiment']
                        if 'Emotion' in analysis:
                            sentiment_data.update(analysis['Emotion'])
                    except:
                        pass
                
                # Extract engagement metrics
                engagement = metadata.get('engagement', {})
                if isinstance(engagement, str):
                    try:
                        engagement = json.loads(engagement)
                    except:
                        engagement = {}
                
                # Create data point
                data_point = {
                    'platform': chunk.platform,
                    'content': chunk.content,
                    'type': metadata.get('type', 'unknown'),
                    'username': metadata.get('username', 'unknown'),
                    'timestamp': metadata.get('timestamp', None),
                    'relevance_score': score,
                    'sentiment': sentiment_data,
                    'engagement': engagement,
                    'content_length': len(chunk.content),
                    'metadata': metadata
                }
                
                data.append(data_point)
                
            except Exception as e:
                print(f"Error extracting data point: {e}")
                continue
        
        return data
    
    def _create_sentiment_evolution_graph(self, data: List[Dict], query: str) -> Dict[str, Any]:
        """Create sentiment evolution over time analysis"""
        
        # Filter data with timestamps and sentiment
        time_data = []
        for item in data:
            if item.get('timestamp') and item.get('sentiment'):
                try:
                    # Parse timestamp
                    if isinstance(item['timestamp'], str):
                        timestamp = datetime.fromisoformat(item['timestamp'].replace('Z', '+00:00'))
                    else:
                        timestamp = item['timestamp']
                    
                    sentiment = item['sentiment']
                    time_data.append({
                        'timestamp': timestamp,
                        'positive': sentiment.get('positive', 0),
                        'negative': sentiment.get('negative', 0),
                        'neutral': sentiment.get('neutral', 0),
                        'platform': item['platform']
                    })
                except:
                    continue
        
        if len(time_data) < 3:
            return None
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(time_data)
        df = df.sort_values('timestamp')
        
        # Group by day and calculate average sentiment
        df['date'] = df['timestamp'].dt.date
        daily_sentiment = df.groupby('date')[['positive', 'negative', 'neutral']].mean()
        
        # Plot sentiment evolution
        dates = daily_sentiment.index
        ax.plot(dates, daily_sentiment['positive'], label='Positive Sentiment', color=self.colors['success'], linewidth=2)
        ax.plot(dates, daily_sentiment['negative'], label='Negative Sentiment', color=self.colors['danger'], linewidth=2)
        ax.plot(dates, daily_sentiment['neutral'], label='Neutral Sentiment', color=self.colors['info'], linewidth=2)
        
        # Customize the plot
        ax.set_title(f'Sentiment Evolution for "{query}"', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Average Sentiment Score', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.tick_params(axis='x', rotation=45)
        
        # Add insights text
        latest_sentiment = daily_sentiment.iloc[-1]
        dominant_sentiment = latest_sentiment.idxmax()
        insight_text = f"Current trend: {dominant_sentiment.title()} sentiment dominates\n"
        insight_text += f"Latest scores - Positive: {latest_sentiment['positive']:.2f}, "
        insight_text += f"Negative: {latest_sentiment['negative']:.2f}"
        
        ax.text(0.02, 0.98, insight_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Convert to base64
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return {
            'type': 'sentiment_evolution',
            'title': f'Sentiment Evolution for "{query}"',
            'description': f'Analysis of how public sentiment about {query} has evolved over time across social media platforms',
            'insights': [
                f"Current dominant sentiment: {dominant_sentiment.title()}",
                f"Total posts analyzed: {len(time_data)}",
                f"Time range: {min(df['timestamp']).strftime('%Y-%m-%d')} to {max(df['timestamp']).strftime('%Y-%m-%d')}"
            ],
            'image': f"data:image/png;base64,{img_base64}",
            'data_points': len(time_data)
        }
    
    def _create_platform_engagement_analysis(self, data: List[Dict], query: str) -> Dict[str, Any]:
        """Create platform engagement comparison analysis"""
        
        # Group by platform and calculate engagement metrics
        platform_stats = defaultdict(lambda: {'posts': 0, 'total_engagement': 0, 'avg_sentiment': []})
        
        for item in data:
            platform = item['platform']
            platform_stats[platform]['posts'] += 1
            
            # Calculate engagement score
            engagement = item.get('engagement', {})
            engagement_score = 0
            if isinstance(engagement, dict):
                likes = engagement.get('likes', 0) or 0
                dislikes = engagement.get('dislikes', 0) or 0
                engagement_score = likes - dislikes
            
            platform_stats[platform]['total_engagement'] += engagement_score
            
            # Add sentiment
            sentiment = item.get('sentiment', {})
            if sentiment:
                sentiment_score = sentiment.get('positive', 0) - sentiment.get('negative', 0)
                platform_stats[platform]['avg_sentiment'].append(sentiment_score)
        
        if len(platform_stats) < 2:
            return None
        
        # Calculate averages
        for platform in platform_stats:
            stats = platform_stats[platform]
            stats['avg_engagement'] = stats['total_engagement'] / stats['posts'] if stats['posts'] > 0 else 0
            stats['avg_sentiment_score'] = np.mean(stats['avg_sentiment']) if stats['avg_sentiment'] else 0
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        platforms = list(platform_stats.keys())
        post_counts = [platform_stats[p]['posts'] for p in platforms]
        avg_engagements = [platform_stats[p]['avg_engagement'] for p in platforms]
        avg_sentiments = [platform_stats[p]['avg_sentiment_score'] for p in platforms]
        
        colors = [self.colors['reddit'] if 'reddit' in p.lower() else self.colors['youtube'] for p in platforms]
        
        # Plot 1: Post count vs Average engagement
        scatter = ax1.scatter(post_counts, avg_engagements, c=colors, s=200, alpha=0.7, edgecolors='black')
        
        for i, platform in enumerate(platforms):
            ax1.annotate(platform, (post_counts[i], avg_engagements[i]), 
                        xytext=(5, 5), textcoords='offset points', fontweight='bold')
        
        ax1.set_title(f'Platform Engagement Analysis for "{query}"', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Number of Posts', fontsize=12)
        ax1.set_ylabel('Average Engagement Score', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Sentiment by platform
        bars = ax2.bar(platforms, avg_sentiments, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_title('Average Sentiment by Platform', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Platform', fontsize=12)
        ax2.set_ylabel('Average Sentiment Score', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, avg_sentiments):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.03),
                    f'{value:.2f}', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
        
        plt.tight_layout()
        
        # Convert to base64
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        # Generate insights
        most_active_platform = max(platforms, key=lambda p: platform_stats[p]['posts'])
        most_engaged_platform = max(platforms, key=lambda p: platform_stats[p]['avg_engagement'])
        most_positive_platform = max(platforms, key=lambda p: platform_stats[p]['avg_sentiment_score'])
        
        return {
            'type': 'platform_analysis',
            'title': f'Platform Engagement Analysis for "{query}"',
            'description': f'Comparative analysis of how different platforms discuss {query}',
            'insights': [
                f"Most active platform: {most_active_platform} ({platform_stats[most_active_platform]['posts']} posts)",
                f"Highest engagement: {most_engaged_platform} (avg: {platform_stats[most_engaged_platform]['avg_engagement']:.1f})",
                f"Most positive sentiment: {most_positive_platform} (score: {platform_stats[most_positive_platform]['avg_sentiment_score']:.2f})"
            ],
            'image': f"data:image/png;base64,{img_base64}",
            'data_points': len(data)
        }
    
    def _create_sentiment_topic_heatmap(self, data: List[Dict], query: str) -> Dict[str, Any]:
        """Create sentiment analysis heatmap by content type and platform"""
        
        # Extract sentiment by platform and content type
        sentiment_matrix = defaultdict(lambda: defaultdict(list))
        
        for item in data:
            platform = item['platform']
            content_type = item.get('type', 'unknown')
            sentiment = item.get('sentiment', {})
            
            if sentiment:
                # Calculate net sentiment score
                net_sentiment = sentiment.get('positive', 0) - sentiment.get('negative', 0)
                sentiment_matrix[platform][content_type].append(net_sentiment)
        
        if len(sentiment_matrix) < 1:
            return None
        
        # Create matrix for heatmap
        platforms = list(sentiment_matrix.keys())
        content_types = set()
        for platform_data in sentiment_matrix.values():
            content_types.update(platform_data.keys())
        content_types = list(content_types)
        
        if len(platforms) < 1 or len(content_types) < 1:
            return None
        
        # Calculate average sentiment for each cell
        heatmap_data = np.zeros((len(platforms), len(content_types)))
        for i, platform in enumerate(platforms):
            for j, content_type in enumerate(content_types):
                sentiments = sentiment_matrix[platform][content_type]
                if sentiments:
                    heatmap_data[i, j] = np.mean(sentiments)
        
        # Create the heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        
        im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(content_types)))
        ax.set_yticks(np.arange(len(platforms)))
        ax.set_xticklabels(content_types)
        ax.set_yticklabels(platforms)
        
        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Average Sentiment Score', rotation=-90, va="bottom")
        
        # Add text annotations
        for i in range(len(platforms)):
            for j in range(len(content_types)):
                count = len(sentiment_matrix[platforms[i]][content_types[j]])
                if count > 0:
                    text = ax.text(j, i, f'{heatmap_data[i, j]:.2f}\n({count})',
                                 ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_title(f'Sentiment Heatmap by Platform and Content Type\nTopic: "{query}"', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Convert to base64
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        # Find insights
        max_sentiment_idx = np.unravel_index(np.argmax(heatmap_data), heatmap_data.shape)
        min_sentiment_idx = np.unravel_index(np.argmin(heatmap_data), heatmap_data.shape)
        
        most_positive = f"{platforms[max_sentiment_idx[0]]} {content_types[max_sentiment_idx[1]]}"
        most_negative = f"{platforms[min_sentiment_idx[0]]} {content_types[min_sentiment_idx[1]]}"
        
        return {
            'type': 'sentiment_heatmap',
            'title': f'Sentiment Analysis Heatmap for "{query}"',
            'description': f'Sentiment distribution across platforms and content types for {query}',
            'insights': [
                f"Most positive: {most_positive} (score: {heatmap_data[max_sentiment_idx]:.2f})",
                f"Most negative: {most_negative} (score: {heatmap_data[min_sentiment_idx]:.2f})",
                f"Analyzed {len(platforms)} platforms and {len(content_types)} content types"
            ],
            'image': f"data:image/png;base64,{img_base64}",
            'data_points': len(data)
        }
    
    def _create_engagement_sentiment_correlation(self, data: List[Dict], query: str) -> Dict[str, Any]:
        """Create correlation analysis between engagement and sentiment"""
        
        # Extract engagement and sentiment data
        engagement_sentiment_data = []
        for item in data:
            engagement = item.get('engagement', {})
            sentiment = item.get('sentiment', {})
            
            if engagement and sentiment:
                # Calculate engagement score
                likes = engagement.get('likes', 0) or 0
                dislikes = engagement.get('dislikes', 0) or 0
                engagement_score = likes + abs(dislikes)  # Total engagement
                
                # Calculate sentiment score
                sentiment_score = sentiment.get('positive', 0) - sentiment.get('negative', 0)
                
                if engagement_score > 0:  # Only include posts with engagement
                    engagement_sentiment_data.append({
                        'engagement': engagement_score,
                        'sentiment': sentiment_score,
                        'platform': item['platform'],
                        'type': item.get('type', 'unknown'),
                        'content_length': item['content_length']
                    })
        
        if len(engagement_sentiment_data) < 5:
            return None
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Separate by platform for different colors
        platforms = set(item['platform'] for item in engagement_sentiment_data)
        colors_map = {list(platforms)[0]: self.colors['reddit'], 
                     list(platforms)[1] if len(platforms) > 1 else list(platforms)[0]: self.colors['youtube']}
        
        for platform in platforms:
            platform_data = [item for item in engagement_sentiment_data if item['platform'] == platform]
            x = [item['engagement'] for item in platform_data]
            y = [item['sentiment'] for item in platform_data]
            
            ax.scatter(x, y, c=colors_map.get(platform, self.colors['primary']), 
                      label=platform, alpha=0.6, s=60, edgecolors='black')
        
        # Add trend line
        all_engagement = [item['engagement'] for item in engagement_sentiment_data]
        all_sentiment = [item['sentiment'] for item in engagement_sentiment_data]
        
        if len(all_engagement) > 1:
            z = np.polyfit(all_engagement, all_sentiment, 1)
            p = np.poly1d(z)
            ax.plot(all_engagement, p(all_engagement), "r--", alpha=0.8, linewidth=2, label='Trend')
            
            # Calculate correlation
            correlation = np.corrcoef(all_engagement, all_sentiment)[0, 1]
        else:
            correlation = 0
        
        ax.set_title(f'Engagement vs Sentiment Correlation\nTopic: "{query}"', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Total Engagement (Likes + Dislikes)', fontsize=12)
        ax.set_ylabel('Sentiment Score (Positive - Negative)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add correlation info
        correlation_text = f'Correlation: {correlation:.3f}\n'
        if abs(correlation) > 0.5:
            correlation_text += 'Strong correlation'
        elif abs(correlation) > 0.3:
            correlation_text += 'Moderate correlation'
        else:
            correlation_text += 'Weak correlation'
        
        ax.text(0.02, 0.98, correlation_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        # Convert to base64
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        # Generate insights
        high_engagement_items = [item for item in engagement_sentiment_data 
                               if item['engagement'] > np.percentile(all_engagement, 75)]
        avg_sentiment_high_engagement = np.mean([item['sentiment'] for item in high_engagement_items]) if high_engagement_items else 0
        
        return {
            'type': 'engagement_correlation',
            'title': f'Engagement vs Sentiment Analysis for "{query}"',
            'description': f'Analysis of how engagement levels correlate with sentiment for {query}',
            'insights': [
                f"Correlation coefficient: {correlation:.3f}",
                f"High-engagement posts average sentiment: {avg_sentiment_high_engagement:.2f}",
                f"Total posts analyzed: {len(engagement_sentiment_data)}"
            ],
            'image': f"data:image/png;base64,{img_base64}",
            'data_points': len(engagement_sentiment_data)
        }
    
    def _create_content_type_analysis(self, data: List[Dict], query: str) -> Dict[str, Any]:
        """Create content type distribution and performance analysis"""
        
        # Analyze content types
        content_analysis = defaultdict(lambda: {
            'count': 0, 
            'total_engagement': 0, 
            'sentiment_scores': [],
            'avg_length': [],
            'relevance_scores': []
        })
        
        for item in data:
            content_type = item.get('type', 'unknown')
            analysis = content_analysis[content_type]
            
            analysis['count'] += 1
            analysis['avg_length'].append(item['content_length'])
            analysis['relevance_scores'].append(item['relevance_score'])
            
            # Add engagement
            engagement = item.get('engagement', {})
            if isinstance(engagement, dict):
                likes = engagement.get('likes', 0) or 0
                dislikes = engagement.get('dislikes', 0) or 0
                analysis['total_engagement'] += (likes + abs(dislikes))
            
            # Add sentiment
            sentiment = item.get('sentiment', {})
            if sentiment:
                sentiment_score = sentiment.get('positive', 0) - sentiment.get('negative', 0)
                analysis['sentiment_scores'].append(sentiment_score)
        
        # Calculate averages
        for content_type in content_analysis:
            analysis = content_analysis[content_type]
            analysis['avg_engagement'] = analysis['total_engagement'] / analysis['count']
            analysis['avg_sentiment'] = np.mean(analysis['sentiment_scores']) if analysis['sentiment_scores'] else 0
            analysis['avg_length'] = np.mean(analysis['avg_length'])
            analysis['avg_relevance'] = np.mean(analysis['relevance_scores'])
        
        if len(content_analysis) < 2:
            return None
        
        # Create the plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        content_types = list(content_analysis.keys())
        counts = [content_analysis[ct]['count'] for ct in content_types]
        avg_sentiments = [content_analysis[ct]['avg_sentiment'] for ct in content_types]
        avg_engagements = [content_analysis[ct]['avg_engagement'] for ct in content_types]
        avg_relevance = [content_analysis[ct]['avg_relevance'] for ct in content_types]
        
        # Plot 1: Content type distribution (pie chart)
        colors = plt.cm.Set3(np.linspace(0, 1, len(content_types)))
        wedges, texts, autotexts = ax1.pie(counts, labels=content_types, autopct='%1.1f%%', 
                                          colors=colors, startangle=90)
        ax1.set_title('Content Type Distribution', fontsize=12, fontweight='bold')
        
        # Plot 2: Average sentiment by content type
        bars2 = ax2.bar(content_types, avg_sentiments, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_title('Average Sentiment by Content Type', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Average Sentiment Score')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Average engagement by content type
        bars3 = ax3.bar(content_types, avg_engagements, color=colors, alpha=0.7, edgecolor='black')
        ax3.set_title('Average Engagement by Content Type', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Average Engagement Score')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Average relevance by content type
        bars4 = ax4.bar(content_types, avg_relevance, color=colors, alpha=0.7, edgecolor='black')
        ax4.set_title('Average Relevance Score by Content Type', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Average Relevance Score')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(f'Content Analysis for "{query}"', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Convert to base64
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        # Generate insights
        most_common_type = max(content_types, key=lambda ct: content_analysis[ct]['count'])
        most_engaging_type = max(content_types, key=lambda ct: content_analysis[ct]['avg_engagement'])
        most_relevant_type = max(content_types, key=lambda ct: content_analysis[ct]['avg_relevance'])
        
        return {
            'type': 'content_analysis',
            'title': f'Content Type Analysis for "{query}"',
            'description': f'Comprehensive analysis of how different content types perform when discussing {query}',
            'insights': [
                f"Most common content type: {most_common_type} ({content_analysis[most_common_type]['count']} posts)",
                f"Most engaging content type: {most_engaging_type} (avg: {content_analysis[most_engaging_type]['avg_engagement']:.1f})",
                f"Most relevant content type: {most_relevant_type} (score: {content_analysis[most_relevant_type]['avg_relevance']:.3f})"
            ],
            'image': f"data:image/png;base64,{img_base64}",
            'data_points': len(data)
        }