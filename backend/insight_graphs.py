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
from typing import List, Dict, Any, Tuple, Optional
import matplotlib.dates as mdates
import os
from groq import Groq

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")

class GroqInsightGraphGenerator:
    """Generate insight-driven visualizations using Groq for intelligent decisions"""
    
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
        
        # Initialize Groq client
        self.groq_client = None
        groq_api_key = os.getenv('GROQ_API_KEY') or "Groq key here"
        if groq_api_key:
            try:
                self.groq_client = Groq(api_key=groq_api_key)
                print("✅ Groq client initialized for graph analysis")
            except Exception as e:
                print(f"⚠️ Groq initialization failed: {e}")
    
    def generate_insights_for_query(self, query: str, rag_results: List[Tuple], gemini_model=None) -> List[Dict[str, Any]]:
        """Generate graphs using Groq for intelligent decisions"""
        
        if not rag_results or not self.groq_client:
            return []
        
        # Extract data from rag_results
        data = self._extract_data_from_results(rag_results)
        
        if not data or len(data) < 3:
            print("📊 Insufficient data for meaningful graph generation")
            return []
        
        # Use Groq to analyze query and determine graph needs
        print("🚀 Using Groq for intelligent graph analysis...")
        groq_analysis = self._analyze_query_with_groq(query, data)
        
        if not groq_analysis.get("needs_graph"):
            print(f"🚫 Groq determined no graph needed for: {query}")
            return []
        
        # Generate graph based on Groq's analysis
        graph_result = self._generate_graph_from_groq_analysis(groq_analysis, data, query)
        
        if graph_result:
            print(f"📊 Generated {groq_analysis['graph_type']} graph for query: {query}")
            return [graph_result]
        
        return []
    
    def _analyze_query_with_groq(self, query: str, data: List[Dict]) -> Dict[str, Any]:
        """Use Groq to determine graph needs and specifications"""
        
        try:
            # Create data summary for Groq
            data_summary = self._create_data_summary(data)
            
            prompt = f"""
Analyze this query and data to determine if a graph would be valuable and what type.

QUERY: "{query}"

AVAILABLE DATA:
- Total posts/comments: {len(data)}
- Platforms: {data_summary['platforms']}
- Time range: {data_summary['time_range']}
- Has sentiment data: {data_summary['has_sentiment']}
- Has engagement data: {data_summary['has_engagement']}
- Content types: {data_summary['content_types']}

INSTRUCTIONS:
1. Determine if a graph would add meaningful insights
2. If yes, choose the best graph type and specify axes
3. Provide inferential meaning for data points, not raw data
4. Respond with ONLY valid JSON - no comments allowed

GRAPH TYPES AVAILABLE:
- line_chart: For trends over time
- bar_chart: For comparisons between categories
- scatter_plot: For correlations between two variables
- pie_chart: For distribution/proportions
- heatmap: For multi-dimensional data patterns

Response format (JSON ONLY):
{{
    "needs_graph": true/false,
    "graph_type": "line_chart/bar_chart/scatter_plot/pie_chart/heatmap",
    "title": "Graph title",
    "x_axis": {{
        "label": "X-axis label",
        "data_points": ["point1", "point2"],
        "meaning": "What these points represent inferentially"
    }},
    "y_axis": {{
        "label": "Y-axis label", 
        "data_points": [value1, value2],
        "meaning": "What these values represent inferentially"
    }},
    "insights": ["insight1", "insight2"],
    "reasoning": "Why this graph type and data representation was chosen"
}}

EXAMPLES:
- For "how do people feel about inflation": line_chart showing sentiment evolution over time
- For "reddit vs youtube engagement": bar_chart comparing average engagement by platform
- For "most discussed topics": pie_chart showing topic distribution

Respond only with valid JSON:
"""
            
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
                temperature=0.1,
                max_tokens=800
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Clean JSON by removing comments and extra text
            response_text = self._clean_json_response(response_text)
            
            try:
                result = json.loads(response_text)
                print(f"🚀 Groq graph analysis: {result}")
                return result
            except json.JSONDecodeError:
                print(f"⚠️ JSON parsing failed, response: {response_text}")
                return {"needs_graph": False}
                
        except Exception as e:
            print(f"⚠️ Groq graph analysis failed: {e}")
            return {"needs_graph": False}
    
    def _clean_json_response(self, response_text: str) -> str:
        """Clean JSON response by removing comments and extra text"""
        # Remove // comments
        lines = response_text.split('\n')
        cleaned_lines = []
        for line in lines:
            # Remove comments after //
            if '//' in line:
                line = line[:line.index('//')]
            cleaned_lines.append(line)
        
        cleaned_text = '\n'.join(cleaned_lines)
        
        # Try to extract JSON from response if it contains extra text
        if '{' in cleaned_text and '}' in cleaned_text:
            start = cleaned_text.find('{')
            end = cleaned_text.rfind('}') + 1
            cleaned_text = cleaned_text[start:end]
        
        return cleaned_text
    
    def _create_data_summary(self, data: List[Dict]) -> Dict[str, Any]:
        """Create a summary of available data for Groq analysis"""
        
        platforms = list(set(item.get('platform', 'unknown') for item in data))
        content_types = list(set(item.get('type', 'unknown') for item in data))
        
        # Check for sentiment data
        has_sentiment = any(item.get('sentiment') for item in data)
        
        # Check for engagement data
        has_engagement = any(item.get('engagement') for item in data)
        
        # Get time range
        timestamps = []
        for item in data:
            if item.get('timestamp'):
                try:
                    if isinstance(item['timestamp'], str):
                        timestamp = datetime.fromisoformat(item['timestamp'].replace('Z', '+00:00'))
                    else:
                        timestamp = item['timestamp']
                    timestamps.append(timestamp)
                except:
                    continue
        
        time_range = "No temporal data"
        if timestamps:
            min_time = min(timestamps)
            max_time = max(timestamps)
            time_range = f"{min_time.strftime('%Y-%m-%d')} to {max_time.strftime('%Y-%m-%d')}"
        
        return {
            'platforms': platforms,
            'content_types': content_types,
            'has_sentiment': has_sentiment,
            'has_engagement': has_engagement,
            'time_range': time_range
        }
    
    def _generate_graph_from_groq_analysis(self, analysis: Dict[str, Any], data: List[Dict], query: str) -> Optional[Dict[str, Any]]:
        """Generate graph based on Groq's analysis"""
        
        graph_type = analysis.get("graph_type")
        
        if graph_type == "line_chart":
            return self._create_line_chart(analysis, data, query)
        elif graph_type == "bar_chart":
            return self._create_bar_chart(analysis, data, query)
        elif graph_type == "scatter_plot":
            return self._create_scatter_plot(analysis, data, query)
        elif graph_type == "pie_chart":
            return self._create_pie_chart(analysis, data, query)
        elif graph_type == "heatmap":
            return self._create_heatmap(analysis, data, query)
        else:
            print(f"⚠️ Unknown graph type: {graph_type}")
            return None
    
    def _create_line_chart(self, analysis: Dict[str, Any], data: List[Dict], query: str) -> Dict[str, Any]:
        """Create line chart based on Groq analysis"""
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Extract data points from analysis
        x_data = analysis.get("x_axis", {}).get("data_points", [])
        y_data = analysis.get("y_axis", {}).get("data_points", [])
        
        # If Groq didn't provide specific data, generate from actual data
        if not x_data or not y_data:
            x_data, y_data = self._extract_time_series_data(data)
        
        # Ensure we have valid data
        if len(x_data) != len(y_data) or len(x_data) == 0:
            x_data = list(range(len(data)))
            y_data = [item.get('relevance_score', 0.5) for item in data]
        
        # Create line plot
        ax.plot(x_data, y_data, marker='o', linewidth=2, markersize=6, color=self.colors['primary'])
        
        # Customize plot
        ax.set_title(analysis.get("title", f"Line Chart Analysis for '{query}'"), fontsize=14, fontweight='bold')
        ax.set_xlabel(analysis.get("x_axis", {}).get("label", "Data Points"), fontsize=12)
        ax.set_ylabel(analysis.get("y_axis", {}).get("label", "Values"), fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add trend line if enough points
        if len(x_data) > 3:
            z = np.polyfit(range(len(x_data)), y_data, 1)
            p = np.poly1d(z)
            ax.plot(range(len(x_data)), p(range(len(x_data))), "--", alpha=0.8, color=self.colors['danger'])
        
        plt.tight_layout()
        
        # Convert to base64
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return {
            'type': 'line_chart',
            'title': analysis.get("title", f"Line Chart for '{query}'"),
            'description': analysis.get("reasoning", "Line chart showing trends in the data"),
            'insights': analysis.get("insights", [f"Analysis of {len(data)} data points"]),
            'image': f"data:image/png;base64,{img_base64}",
            'data_points': len(data)
        }
    
    def _create_bar_chart(self, analysis: Dict[str, Any], data: List[Dict], query: str) -> Dict[str, Any]:
        """Create bar chart based on Groq analysis"""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract data points from analysis
        x_data = analysis.get("x_axis", {}).get("data_points", [])
        y_data = analysis.get("y_axis", {}).get("data_points", [])
        
        # If Groq didn't provide specific data, generate from actual data
        if not x_data or not y_data:
            x_data, y_data = self._extract_categorical_data(data)
        
        # Create bar plot
        bars = ax.bar(x_data, y_data, color=[self.colors['primary'], self.colors['secondary'], 
                                            self.colors['success'], self.colors['warning']][:len(x_data)])
        
        # Customize plot
        ax.set_title(analysis.get("title", f"Bar Chart Analysis for '{query}'"), fontsize=14, fontweight='bold')
        ax.set_xlabel(analysis.get("x_axis", {}).get("label", "Categories"), fontsize=12)
        ax.set_ylabel(analysis.get("y_axis", {}).get("label", "Values"), fontsize=12)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Convert to base64
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return {
            'type': 'bar_chart',
            'title': analysis.get("title", f"Bar Chart for '{query}'"),
            'description': analysis.get("reasoning", "Bar chart showing categorical comparisons"),
            'insights': analysis.get("insights", [f"Comparison of {len(x_data)} categories"]),
            'image': f"data:image/png;base64,{img_base64}",
            'data_points': len(data)
        }
    
    def _create_scatter_plot(self, analysis: Dict[str, Any], data: List[Dict], query: str) -> Dict[str, Any]:
        """Create scatter plot based on Groq analysis"""
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Extract data points from analysis
        x_data = analysis.get("x_axis", {}).get("data_points", [])
        y_data = analysis.get("y_axis", {}).get("data_points", [])
        
        # If Groq didn't provide specific data, generate from actual data
        if not x_data or not y_data:
            x_data, y_data = self._extract_correlation_data(data)
        
        # Create scatter plot
        ax.scatter(x_data, y_data, alpha=0.6, s=60, color=self.colors['primary'])
        
        # Add correlation line if enough points
        if len(x_data) > 3:
            z = np.polyfit(x_data, y_data, 1)
            p = np.poly1d(z)
            ax.plot(x_data, p(x_data), "--", alpha=0.8, color=self.colors['danger'])
            
            # Calculate correlation coefficient
            correlation = np.corrcoef(x_data, y_data)[0, 1]
            ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Customize plot
        ax.set_title(analysis.get("title", f"Scatter Plot Analysis for '{query}'"), fontsize=14, fontweight='bold')
        ax.set_xlabel(analysis.get("x_axis", {}).get("label", "X Values"), fontsize=12)
        ax.set_ylabel(analysis.get("y_axis", {}).get("label", "Y Values"), fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to base64
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return {
            'type': 'scatter_plot',
            'title': analysis.get("title", f"Scatter Plot for '{query}'"),
            'description': analysis.get("reasoning", "Scatter plot showing correlations in the data"),
            'insights': analysis.get("insights", [f"Correlation analysis of {len(data)} data points"]),
            'image': f"data:image/png;base64,{img_base64}",
            'data_points': len(data)
        }
    
    def _create_pie_chart(self, analysis: Dict[str, Any], data: List[Dict], query: str) -> Dict[str, Any]:
        """Create pie chart based on Groq analysis"""
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Extract data points from analysis
        labels = analysis.get("x_axis", {}).get("data_points", [])
        sizes = analysis.get("y_axis", {}).get("data_points", [])
        
        # If Groq didn't provide specific data, generate from actual data
        if not labels or not sizes:
            labels, sizes = self._extract_distribution_data(data)
        
        # Create pie chart
        colors = [self.colors['primary'], self.colors['secondary'], self.colors['success'], 
                 self.colors['warning'], self.colors['info'], self.colors['danger']]
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                         colors=colors[:len(labels)], startangle=90)
        
        # Customize plot
        ax.set_title(analysis.get("title", f"Distribution Analysis for '{query}'"), fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Convert to base64
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return {
            'type': 'pie_chart',
            'title': analysis.get("title", f"Pie Chart for '{query}'"),
            'description': analysis.get("reasoning", "Pie chart showing distribution of data"),
            'insights': analysis.get("insights", [f"Distribution analysis of {len(data)} data points"]),
            'image': f"data:image/png;base64,{img_base64}",
            'data_points': len(data)
        }
    
    def _create_heatmap(self, analysis: Dict[str, Any], data: List[Dict], query: str) -> Dict[str, Any]:
        """Create heatmap based on Groq analysis"""
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create sample heatmap data
        heatmap_data = self._extract_heatmap_data(data)
        
        # Create heatmap
        sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd', ax=ax, fmt='.2f')
        
        # Customize plot
        ax.set_title(analysis.get("title", f"Heatmap Analysis for '{query}'"), fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Convert to base64
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return {
            'type': 'heatmap',
            'title': analysis.get("title", f"Heatmap for '{query}'"),
            'description': analysis.get("reasoning", "Heatmap showing multi-dimensional data patterns"),
            'insights': analysis.get("insights", [f"Pattern analysis of {len(data)} data points"]),
            'image': f"data:image/png;base64,{img_base64}",
            'data_points': len(data)
        }
    
    # Data extraction helper methods
    def _extract_time_series_data(self, data: List[Dict]) -> Tuple[List, List]:
        """Extract time series data for line charts"""
        timestamps = []
        values = []
        
        for item in data:
            if item.get('timestamp'):
                try:
                    if isinstance(item['timestamp'], str):
                        timestamp = datetime.fromisoformat(item['timestamp'].replace('Z', '+00:00'))
                    else:
                        timestamp = item['timestamp']
                    timestamps.append(timestamp)
                    
                    # Use relevance score or sentiment as value
                    if item.get('sentiment'):
                        sentiment = item['sentiment']
                        value = sentiment.get('positive', 0) - sentiment.get('negative', 0)
                    else:
                        value = item.get('relevance_score', 0.5)
                    values.append(value)
                except:
                    continue
        
        if not timestamps:
            return list(range(len(data))), [item.get('relevance_score', 0.5) for item in data]
        
        # Sort by timestamp
        sorted_data = sorted(zip(timestamps, values))
        timestamps, values = zip(*sorted_data)
        
        return list(timestamps), list(values)
    
    def _extract_categorical_data(self, data: List[Dict]) -> Tuple[List[str], List[float]]:
        """Extract categorical data for bar charts"""
        # Group by platform
        platform_counts = Counter(item.get('platform', 'unknown') for item in data)
        
        if len(platform_counts) == 1:
            # Group by content type instead
            type_counts = Counter(item.get('type', 'unknown') for item in data)
            return list(type_counts.keys()), list(type_counts.values())
        
        return list(platform_counts.keys()), list(platform_counts.values())
    
    def _extract_correlation_data(self, data: List[Dict]) -> Tuple[List[float], List[float]]:
        """Extract correlation data for scatter plots"""
        x_values = []
        y_values = []
        
        for item in data:
            # Use relevance score as x
            x_val = item.get('relevance_score', 0.5)
            
            # Use engagement or sentiment as y
            if item.get('engagement'):
                engagement = item['engagement']
                if isinstance(engagement, dict):
                    y_val = engagement.get('score', 0) or engagement.get('likes', 0) or engagement.get('upvotes', 0)
                else:
                    y_val = float(engagement) if engagement else 0
            elif item.get('sentiment'):
                sentiment = item['sentiment']
                y_val = sentiment.get('positive', 0) - sentiment.get('negative', 0)
            else:
                y_val = len(item.get('content', '')) / 100  # Content length proxy
            
            x_values.append(x_val)
            y_values.append(y_val)
        
        return x_values, y_values
    
    def _extract_distribution_data(self, data: List[Dict]) -> Tuple[List[str], List[int]]:
        """Extract distribution data for pie charts"""
        platform_counts = Counter(item.get('platform', 'unknown') for item in data)
        return list(platform_counts.keys()), list(platform_counts.values())
    
    def _extract_heatmap_data(self, data: List[Dict]) -> pd.DataFrame:
        """Extract data for heatmap visualization"""
        # Create a simple platform vs time heatmap
        platforms = list(set(item.get('platform', 'unknown') for item in data))
        
        # Create sample data matrix
        matrix_data = []
        for platform in platforms:
            platform_data = [item for item in data if item.get('platform') == platform]
            row = [len(platform_data), 
                   np.mean([item.get('relevance_score', 0.5) for item in platform_data]),
                   len([item for item in platform_data if item.get('sentiment', {}).get('positive', 0) > 0.5])]
            matrix_data.append(row)
        
        return pd.DataFrame(matrix_data, 
                           index=platforms, 
                           columns=['Count', 'Avg Relevance', 'Positive Posts'])
    
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