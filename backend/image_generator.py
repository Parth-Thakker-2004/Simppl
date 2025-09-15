import requests
import json
from typing import Optional, Dict, Any
import os
from groq import Groq

class ImageGenerator:
    """Generate relevant images for queries using streamlined Groq + Wikipedia approach"""
    
    def __init__(self):
        self.groq_client = None
        groq_api_key = os.getenv('GROQ_API_KEY') or "Groq key here"
        if groq_api_key:
            try:
                self.groq_client = Groq(api_key=groq_api_key)
                print("✅ Groq client initialized for fast AI decisions")
            except Exception as e:
                print(f"⚠️ Groq initialization failed: {e}")
    
    def analyze_query_with_groq(self, query: str) -> Dict[str, Any]:
        """Use Groq to determine if image is needed, answer question, and get search terms"""
        
        if not self.groq_client:
            return {"needs_image": False, "answer": None, "search_term": None, "image_type": None}
        
        try:
            prompt = f"""
Analyze this query and determine:
1. Does it need an image? (YES/NO)
2. If yes, what type? (person/place/object)
3. What's the answer to the question?
4. What search term should be used for image search?

QUERY: "{query}"

INSTRUCTIONS:
- For "who" questions about people: Answer with the person's name and use that name for search
- For "where" questions: Answer with location and use that for search  
- For "what does X look like" or "show me X": Always needs image
- For data/statistics/opinion questions: NO image needed

Response format (JSON):
{{
    "needs_image": true/false,
    "image_type": "person/place/object/null",
    "answer": "direct answer to the question",
    "search_term": "term to search for image"
}}

Examples:
- "who is elon musk" → {{"needs_image": true, "image_type": "person", "answer": "Elon Musk", "search_term": "Elon Musk"}}
- "who was the last prime minister of india" → {{"needs_image": true, "image_type": "person", "answer": "Narendra Modi", "search_term": "Narendra Modi"}}
- "where is paris" → {{"needs_image": true, "image_type": "place", "answer": "Paris, France", "search_term": "Paris France"}}
- "what are inflation rates" → {{"needs_image": false, "image_type": null, "answer": "Current inflation data...", "search_term": null}}

Respond only with valid JSON:
"""
            
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
                temperature=0.1,
                max_tokens=300
            )
            
            response_text = response.choices[0].message.content.strip()
            try:
                result = json.loads(response_text)
                print(f"🚀 Groq analysis: {result}")
                return result
            except json.JSONDecodeError:
                print(f"⚠️ JSON parsing failed, response: {response_text}")
                return {"needs_image": False, "answer": None, "search_term": None, "image_type": None}
                
        except Exception as e:
            print(f"⚠️ Groq analysis failed: {e}")
            return {"needs_image": False, "answer": None, "search_term": None, "image_type": None}
    
    def should_generate_graph_with_groq(self, query: str, num_results: int) -> bool:
        """Use Groq to determine if a graph would be valuable for the query"""
        
        if not self.groq_client or num_results < 3:
            return False
        
        try:
            prompt = f"""
Analyze this query and determine if a graph/chart would be valuable to visualize the data.

QUERY: "{query}"
DATA_POINTS: {num_results} social media posts/comments available

GUIDELINES:
- Generate graphs for: trends, comparisons, sentiment analysis, data patterns, statistics
- DON'T generate graphs for: simple factual questions, person identification, definitions, where questions

Respond with only YES or NO:
"""
            
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
                temperature=0.1,
                max_tokens=10
            )
            
            response_text = response.choices[0].message.content.strip().upper()
            should_generate = response_text.startswith("YES")
            
            print(f"🚀 Groq graph decision for '{query}': {should_generate}")
            return should_generate
                
        except Exception as e:
            print(f"⚠️ Groq graph decision failed: {e}")
            return False
    
    # Removed - now handled by Groq in analyze_query_with_groq
    
    # Removed - now handled by Groq in analyze_query_with_groq
    
    def search_image(self, search_term: str, image_type: str = "general") -> Optional[Dict[str, Any]]:
        """Search for relevant images using free APIs with streamlined approach"""
        
        print(f"🔍 Searching for image: '{search_term}' (type: {image_type})")
        
        # Try Wikipedia first (most reliable for people/places)
        result = self._search_wikimedia(search_term)
        if result:
            print(f"✅ Found Wikipedia image for: {search_term}")
            return result
        
        # For people, avoid random images - try name variations only
        if image_type == "person":
            name_variations = [search_term.title(), search_term.replace(" ", "_")]
            for variation in name_variations:
                if variation != search_term:
                    result = self._search_wikimedia(variation)
                    if result:
                        print(f"✅ Found Wikipedia image with variation: {variation}")
                        return result
        
        # Final fallback to enhanced placeholder
        return self._get_enhanced_placeholder(search_term, image_type)
    
    # Removed unused API methods - keeping only Wikipedia and placeholder
    
    def _search_wikimedia(self, search_term: str) -> Optional[Dict[str, Any]]:
        """Search Wikimedia Commons for free images"""
        try:
            # Search Wikipedia first to get main image
            wiki_url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + search_term.replace(" ", "_")
            headers = {
                'User-Agent': 'MediaMeld/1.0 (https://github.com/Parth-Thakker-2004/Simppl; journalism-research-tool) Python/requests'
            }
            response = requests.get(wiki_url, headers=headers, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if "originalimage" in data:
                    print(f"✅ Found Wikipedia image for: {search_term}")
                    return {
                        "url": data["originalimage"]["source"],
                        "thumbnail": data.get("thumbnail", {}).get("source", data["originalimage"]["source"]),
                        "description": data.get("description", search_term),
                        "photographer": "Wikimedia Commons",
                        "source": "Wikipedia"
                    }
                else:
                    print(f"📷 No image found in Wikipedia for: {search_term}")
            else:
                print(f"⚠️ Wikipedia API returned status {response.status_code} for: {search_term}")
        except Exception as e:
            print(f"⚠️ Wikimedia search failed: {e}")
        
        return None
    
    # Removed unused methods - keeping only essential Wikipedia and placeholder
    
    def _get_enhanced_placeholder(self, search_term: str, image_type: str) -> Dict[str, Any]:
        """Generate an enhanced placeholder image with better styling"""
        
        # Final fallback to styled placeholder
        width, height = 600, 400
        text = search_term.replace(" ", "+")
        
        # Use a more attractive color scheme
        if image_type == "person":
            bg_color = "4a90e2"  # Blue for people
            text_color = "ffffff"
        elif image_type == "place":
            bg_color = "27ae60"  # Green for places
            text_color = "ffffff"
        else:
            bg_color = "e74c3c"  # Red for objects/general
            text_color = "ffffff"
        
        placeholder_url = f"https://via.placeholder.com/{width}x{height}/{bg_color}/{text_color}?text={text}"
        
        return {
            "url": placeholder_url,
            "thumbnail": placeholder_url,
            "description": f"Visual representation of {search_term}",
            "photographer": "Generated Content",
            "source": "Enhanced Placeholder"
        }
    
    def generate_image_for_query(self, query: str, gemini_model=None) -> Optional[Dict[str, Any]]:
        """Main method to generate image for a query using streamlined Groq-only approach"""
        
        # Use Groq for all analysis - single API call
        if self.groq_client:
            print("🚀 Using Groq for intelligent image analysis...")
            groq_analysis = self.analyze_query_with_groq(query)
            
            if groq_analysis["needs_image"] and groq_analysis["search_term"]:
                search_term = groq_analysis["search_term"]
                image_type = groq_analysis["image_type"] or "general"
                
                print(f"🎯 Groq decision: Image needed for '{search_term}' (type: {image_type})")
                
                # Single streamlined image search
                image_result = self.search_image(search_term, image_type)
                
                if image_result:
                    image_result["search_term"] = search_term
                    image_result["image_type"] = image_type
                    image_result["groq_answer"] = groq_analysis.get("answer", "")
                    print(f"🖼️ Generated image for query: {query} -> {search_term}")
                    return image_result
            else:
                print(f"🚫 Groq determined no image needed for: {query}")
                return None
        
        # No fallback needed - Groq handles everything
        print("⚠️ Groq not available, no image generated")
        return None