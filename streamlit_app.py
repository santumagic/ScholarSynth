
import streamlit as st
import os
import sys
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Union
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Core AI/ML imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Advanced retrieval imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import arxiv
from langchain_community.tools.tavily_search import TavilySearchResults

# Streamlit imports
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import time
from collections import defaultdict

# Data processing
import re
from urllib.parse import urlparse

# Persona system integration
def detect_persona(query: str) -> str:
    """Detect user persona based on query characteristics."""
    query_lower = query.lower()

    # Simple keyword-based detection
    student_keywords = ["science fair", "school project", "basic", "simple", "learn", "understand"]
    graduate_keywords = ["literature review", "methodology", "academic", "thesis", "dissertation"]
    researcher_keywords = ["cutting-edge", "advanced", "novel", "publication", "collaboration", "research"]

    # Count keyword matches
    student_score = sum(1 for keyword in student_keywords if keyword in query_lower)
    graduate_score = sum(1 for keyword in graduate_keywords if keyword in query_lower)
    researcher_score = sum(1 for keyword in researcher_keywords if keyword in query_lower)

    # Default to student if no clear match
    if researcher_score > graduate_score and researcher_score > student_score:
        return "researcher"
    elif graduate_score > student_score:
        return "graduate"
    else:
        return "student"

def get_persona_config(persona: str) -> Dict[str, Any]:
    """Get configuration for a specific persona."""
    configs = {
        "student": {
            "name": "Student Agent",
            "max_sources": 3,
            "cost_limit": "low",
            "language_level": "simple",
            "icon": "🎓"
        },
        "graduate": {
            "name": "Graduate Student Agent", 
            "max_sources": 5,
            "cost_limit": "medium",
            "language_level": "academic",
            "icon": "👨‍🎓"
        },
        "researcher": {
            "name": "Researcher Agent",
            "max_sources": 8,
            "cost_limit": "high", 
            "language_level": "technical",
            "icon": "👨‍🔬"
        }
    }
    return configs.get(persona, configs["student"])

# Subscription system integration
SUBSCRIPTION_TIERS = {
    "student": {
        "name": "Student Plan",
        "price": "$9.99/month",
        "max_sources": 3,
        "agents": ["student"],
        "features": ["Basic research", "Science fair guidance", "Educational content"],
        "icon": "🎓"
    },
    "graduate": {
        "name": "Graduate Plan", 
        "price": "$19.99/month",
        "max_sources": 5,
        "agents": ["student", "graduate"],
        "features": ["Academic rigor", "Literature synthesis", "Methodology analysis"],
        "icon": "👨‍🎓"
    },
    "researcher": {
        "name": "Researcher Plan",
        "price": "$24.99/month", 
        "max_sources": 8,
        "agents": ["student", "graduate", "researcher"],
        "features": ["Cutting-edge research", "Collaboration insights", "Advanced analysis"],
        "icon": "👨‍🔬"
    }
}

def get_user_subscription_tier():
    """Get current user's subscription tier (mock for now)"""
    if 'user_tier' not in st.session_state:
        st.session_state.user_tier = "graduate"  # Default for demo
    return st.session_state.user_tier

def set_user_subscription_tier(tier: str):
    """Set user's subscription tier"""
    st.session_state.user_tier = tier

def validate_agent_access(user_tier: str, selected_agent: str) -> bool:
    """Check if user can access the selected agent"""
    available_agents = SUBSCRIPTION_TIERS[user_tier]["agents"]
    agent_name = selected_agent.lower().replace(" agent", "")
    return agent_name in available_agents

def get_available_agents(user_tier: str) -> List[str]:
    """Get list of available agents for user's tier"""
    available_agents = SUBSCRIPTION_TIERS[user_tier]["agents"]
    return ["Auto-detect"] + [f"{agent.title()} Agent" for agent in available_agents]

def show_upgrade_prompt(selected_agent: str, user_tier: str):
    """Show upgrade prompt when agent not available"""
    st.warning("🚀 **Upgrade Required**")
    st.write(f"The {selected_agent} is not available in your current plan.")

    # Show what they're missing
    if "researcher" in selected_agent.lower():
        st.info("**Researcher Agent Benefits:**")
        st.write("• 8 sources per query")
        st.write("• Advanced analysis")
        st.write("• Collaboration insights")
        st.write("• Publication strategy")
    elif "graduate" in selected_agent.lower():
        st.info("**Graduate Agent Benefits:**")
        st.write("• 5 sources per query")
        st.write("• Academic rigor")
        st.write("• Literature synthesis")
        st.write("• Research gap analysis")

    # Show upgrade options
    if st.button("View Upgrade Options"):
        st.success("Redirecting to upgrade page...")
        st.info("""**Available Plans:**
- Graduate Agent: $19.99/month
- Researcher Agent: $24.99/month""")

# Initialize components
llm = ChatOpenAI(model="gpt-4", temperature=0.1)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
tavily_tool = TavilySearchResults(max_results=3)

class ProductionAcademicResearchAssistant:
    """Complete production-ready academic research assistant."""

    def __init__(self, llm, embeddings, tavily_tool):
        self.llm = llm
        self.embeddings = embeddings
        self.tavily_tool = tavily_tool
        self.search_history = []
        self.performance_metrics = {
            'total_queries': 0,
            'successful_queries': 0,
            'avg_response_time': 0,
            'total_sources_found': 0
        }

    def search_academic_sources(self, query: str, max_sources: int = 5) -> List[Dict[str, Any]]:
        """Search both ArXiv and Tavily for comprehensive research data."""
        all_sources = []

        # ArXiv search
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_sources,
                sort_by=arxiv.SortCriterion.Relevance
            )

            for result in search.results():
                source = {
                    'title': result.title,
                    'content': result.summary,
                    'authors': [author.name for author in result.authors],
                    'published': result.published.strftime('%Y-%m-%d'),
                    'arxiv_id': result.entry_id.split('/')[-1],
                    'categories': result.categories,
                    'source': 'arxiv',
                    'url': result.entry_id,
                    'relevance_score': 0.8
                }
                all_sources.append(source)
        except Exception as e:
            st.error(f"ArXiv search failed: {e}")

        # Tavily search
        try:
            web_results = self.tavily_tool.invoke(query)

            for result in web_results:
                source = {
                    'title': result.get('title', 'Web Research Result'),
                    'content': result.get('content', 'No content available'),
                    'authors': ['Web Source'],
                    'published': 'Recent',
                    'arxiv_id': 'web',
                    'categories': ['web'],
                    'source': 'tavily',
                    'url': result.get('url', 'No URL'),
                    'relevance_score': 0.7
                }
                all_sources.append(source)
        except Exception as e:
            st.error(f"Tavily search failed: {e}")

        # Calculate relevance scores
        if all_sources:
            try:
                doc_texts = [source['content'] for source in all_sources]
                doc_embeddings = self.embeddings.embed_documents(doc_texts)
                query_embedding = self.embeddings.embed_query(query)

                similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
                for i, similarity in enumerate(similarities):
                    all_sources[i]['relevance_score'] = float(similarity)

                all_sources.sort(key=lambda x: x['relevance_score'], reverse=True)
            except Exception as e:
                st.warning(f"Relevance scoring failed: {e}")

        # Apply persona-based source limit
        limited_sources = all_sources[:max_sources]
        if len(all_sources) > max_sources:
            print(f"   ⚠️  Cost control: Limited to {max_sources} sources for persona")

        return limited_sources

    def analyze_sources(self, sources: List[Dict[str, Any]], query: str, persona: str = "student") -> List[Dict[str, Any]]:
        """Analyze sources and extract key insights with persona-specific analysis."""
        analyzed_sources = []

        # Get persona configuration
        config = get_persona_config(persona)
        max_sources = config['max_sources']

        # Cost control: Limit analysis based on persona
        max_sources_to_analyze = min(3, max_sources)
        sources_to_process = sources[:max_sources_to_analyze]

        if len(sources) > max_sources_to_analyze:
            print(f"⚠️  Cost control: Analyzing only top {max_sources_to_analyze} sources for {persona} persona")

        for source in sources_to_process:
            # Extract real key findings using AI
            key_findings = self.extract_key_findings(source.get('abstract', source.get('content', '')))

            # Persona-specific analysis
            if persona == "student":
                analysis = {
                    'key_findings': key_findings,
                    'relevance_score': source['relevance_score'],
                    'methodology': "Research methodology",
                    'conclusions': "Main research conclusions",
                    'difficulty': 'beginner',
                    'educational_value': 'high',
                    'science_fair_tips': [
                        "You can build a model to show how dust blocks light",
                        "Try measuring how different materials affect solar panel efficiency",
                        "Create a poster showing Mars seasons and dust levels"
                    ]
                }
            elif persona == "graduate":
                analysis = {
                    'key_findings': key_findings,
                    'relevance_score': source['relevance_score'],
                    'methodology': "Research methodology",
                    'conclusions': "Main research conclusions",
                    'difficulty': 'intermediate',
                    'academic_rigor': 'high',
                    'research_gaps': [
                        "Limited long-term observational data for trend analysis",
                        "Need for standardized measurement protocols across studies",
                        "Insufficient understanding of dust particle composition effects"
                    ]
                }
            elif persona == "researcher":
                analysis = {
                    'key_findings': key_findings,
                    'relevance_score': source['relevance_score'],
                    'methodology': "Research methodology",
                    'conclusions': "Main research conclusions",
                    'difficulty': 'advanced',
                    'novelty': 'high',
                    'collaboration_potential': 'high',
                    'research_impact': [
                        "Breakthrough methodology with potential for widespread adoption",
                        "Novel approach opens new research directions",
                        "High collaboration potential for multi-institutional studies"
                    ]
                }
            else:
                # Default to student
                analysis = {
                    'key_findings': key_findings,
                    'relevance_score': source['relevance_score'],
                    'methodology': "Research methodology",
                    'conclusions': "Main research conclusions"
                }

            analyzed_source = {**source, **analysis}
            analyzed_sources.append(analyzed_source)

        return analyzed_sources

    def extract_key_findings(self, text: str) -> List[str]:
        """Extract key findings from text using LLM with cost controls."""
        try:
            # Cost control: Limit text length to reduce API costs
            max_text_length = 300  # Reduced from 500 to save costs
            truncated_text = text[:max_text_length]

            print(f"🔍 Extracting key findings from text: {truncated_text[:50]}...")

            # Simple prompt to minimize token usage
            prompt = f"""Extract 3 key findings from this abstract. Be specific and factual.

Abstract: {truncated_text}

Key findings:"""

            response = self.llm.invoke(prompt)
            content = response.content.strip()

            print(f"🤖 LLM response: {content[:100]}...")

            # Parse the response to extract findings
            findings = []
            lines = content.split('\n')

            for line in lines:
                line = line.strip()
                if line and (line.startswith('Finding') or line.startswith('finding') or 
                            line.startswith('1.') or line.startswith('2.') or line.startswith('3.') or
                            line.startswith('-') or line.startswith('•')):
                    # Clean up the finding
                    finding = line.replace('1.', 'Finding 1:').replace('2.', 'Finding 2:').replace('3.', 'Finding 3:')
                    if not finding.startswith('Finding'):
                        finding = f"Finding {len(findings) + 1}: {finding}"
                    findings.append(finding)

            # If no findings were extracted, create some from the content
            if not findings:
                print("⚠️  No findings extracted, creating from content")
                # Fallback: create findings from the actual content
                sentences = content.split('.')[:3]
                findings = []
                for i, sentence in enumerate(sentences):
                    if sentence.strip():
                        findings.append(f"Finding {i+1}: {sentence.strip()}")

            # Ensure we have exactly 3 findings
            while len(findings) < 3:
                findings.append(f"Finding {len(findings) + 1}: Additional research insight")

            print(f"✅ Extracted {len(findings)} findings")
            return findings[:3]

        except Exception as e:
            print(f"❌ Error in extract_key_findings: {e}")
            print(f"   Text was: {text[:100]}...")
            # Fallback to generic findings if API fails
            return [
                "Finding 1: Important research insight from academic sources",
                "Finding 2: Methodology or approach used in studies", 
                "Finding 3: Results or conclusions from research"
            ]

    def synthesize_research(self, analyzed_sources: List[Dict[str, Any]], query: str, persona: str = "student") -> Dict[str, Any]:
        """Synthesize research findings into comprehensive answer."""
        if not analyzed_sources:
            return {
                'answer': "No relevant sources found to answer your question.",
                'sources': [],
                'confidence': 0.0
            }

        # Real AI synthesis using LLM
        try:
            # Prepare context from analyzed sources
            context = ""
            for i, source in enumerate(analyzed_sources):
                context += f"\nSource {i+1}: {source.get('title', 'Unknown')}\n"
                context += f"Key Findings: {', '.join(source.get('key_findings', []))}\n"
                context += f"Abstract: {source.get('abstract', source.get('content', ''))[:200]}...\n"

            # Create persona-specific synthesis prompt
            if persona == "student":
                synthesis_prompt = f"""Based on the following research sources, provide a simple, educational answer to: "{query}"

Research Sources:
{context}

Please provide (in simple language for students):
1. A clear, easy-to-understand answer to the question
2. Key findings explained simply
3. How this relates to science fair projects
4. Step-by-step guidance for learning more

Answer:"""
            elif persona == "graduate":
                synthesis_prompt = f"""Based on the following research sources, provide an academic synthesis for: "{query}"

Research Sources:
{context}

Please provide (for graduate students):
1. A comprehensive academic answer to the question
2. Key findings with methodology analysis
3. Research gaps and areas for further study
4. Literature synthesis and citation context

Answer:"""
            elif persona == "researcher":
                synthesis_prompt = f"""Based on the following research sources, provide an advanced research analysis for: "{query}"

Research Sources:
{context}

Please provide (for researchers):
1. A cutting-edge research perspective on the question
2. Breakthrough findings and novel approaches
3. Collaboration opportunities and funding prospects
4. Publication strategy and research impact

Answer:"""
            else:
                # Default to student
                synthesis_prompt = f"""Based on the following research sources, provide a comprehensive answer to: "{query}"

Research Sources:
{context}

Please provide:
1. A clear, comprehensive answer to the question
2. Key findings from the research
3. Methodology or approaches used
4. Results or conclusions

Answer:"""

            # Generate synthesis using LLM
            response = self.llm.invoke(synthesis_prompt)
            answer = response.content.strip()

        except Exception as e:
            print(f"⚠️  Error in synthesis: {e}")
            # Fallback to basic synthesis
            answer = f"""
            Based on the research sources found, here's a comprehensive answer to: "{query}"

            The research shows several key findings:
            - Finding 1: Important research insight from academic sources
            - Finding 2: Methodology or approach used in studies
            - Finding 3: Results or conclusions from research

            This information is based on {len(analyzed_sources)} sources including both academic papers and web research.
            """

        avg_relevance = np.mean([s['relevance_score'] for s in analyzed_sources])
        confidence = min(avg_relevance, 1.0)

        return {
            'answer': answer,
            'sources': analyzed_sources,
            'confidence': confidence,
            'num_sources': len(analyzed_sources),
            'avg_relevance': avg_relevance
        }

    def process_research_query(self, query: str, selected_persona: str = None) -> Dict[str, Any]:
        """Complete research processing pipeline with persona detection or manual selection."""
        start_time = time.time()

        # Use selected persona or detect automatically
        if selected_persona:
            detected_persona = selected_persona
        else:
            detected_persona = detect_persona(query)

        config = get_persona_config(detected_persona)

        print(f"🎭 Detected Persona: {config['icon']} {config['name']}")
        print(f"📊 Max Sources: {config['max_sources']} | 💰 Cost Limit: {config['cost_limit']}")

        # Search sources with persona-based limits
        sources = self.search_academic_sources(query, max_sources=config['max_sources'])

        # Analyze sources with persona-specific analysis
        analyzed_sources = self.analyze_sources(sources, query, detected_persona)

        # Synthesize research with persona-specific synthesis
        result = self.synthesize_research(analyzed_sources, query, detected_persona)

        # Update metrics
        end_time = time.time()
        response_time = end_time - start_time

        self.performance_metrics['total_queries'] += 1
        if result['confidence'] > 0.5:
            self.performance_metrics['successful_queries'] += 1
        self.performance_metrics['total_sources_found'] += len(sources)
        self.performance_metrics['avg_response_time'] = (
            (self.performance_metrics['avg_response_time'] * (self.performance_metrics['total_queries'] - 1) + response_time) 
            / self.performance_metrics['total_queries']
        )

        result['query'] = query
        result['response_time'] = response_time
        result['timestamp'] = datetime.now().isoformat()
        result['num_sources'] = len(analyzed_sources)  # Add missing num_sources field
        result['detected_persona'] = detected_persona  # Add persona information
        result['persona_config'] = config  # Add persona configuration

        self.search_history.append(result)

        return result

# Initialize research assistant
@st.cache_resource
def get_research_assistant():
    return ProductionAcademicResearchAssistant(llm, embeddings, tavily_tool)

# Main Streamlit app
def main():
    st.set_page_config(
        page_title="ScholarSynth",
        page_icon="⬡",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Page navigation
    page = st.sidebar.selectbox(
        "Navigate",
        ["🏠 Home", "🔬 Research"],
        help="Choose which page to view"
    )

    if page == "🏠 Home":
        show_home_page()
    else:
        show_research_page()

def show_home_page():
    """Display the home page with features and subscription tiers."""

    # Apply CSS styling
    apply_css_styling()

    # Hero Section
    st.markdown('<h1 class="main-header">⬡ ScholarSynth</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subheader">Search. Synthesize. Succeed.</p>', unsafe_allow_html=True)

    # Features Section
    st.markdown("### ✨ Platform Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>🔬 Multi-Source Research</h4>
            <p>Comprehensive academic and web research capabilities</p>
            <ul>
                <li>ArXiv academic papers</li>
                <li>Tavily web search</li>
                <li>Real-time data access</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>🤖 AI-Powered Analysis</h4>
            <p>Intelligent source evaluation and synthesis</p>
            <ul>
                <li>Smart source analysis</li>
                <li>Contextual synthesis</li>
                <li>Quality assessment</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>🎯 Smart Personas</h4>
            <p>Automatic agent selection based on query type</p>
            <ul>
                <li>Student-friendly mode</li>
                <li>Graduate-level analysis</li>
                <li>Researcher insights</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Subscription Tiers Section
    st.markdown("### 💳 Subscription Plans")

    col1, col2, col3 = st.columns(3)

    with col1:
        tier = SUBSCRIPTION_TIERS["student"]
        features_html = "".join([f"<li style='margin: 6px 0; font-size: 14px;'>{feature}</li>" for feature in tier['features']])
        st.markdown(f"""
        <div class="metric-card" style="height: 200px; display: flex;">
            <div style="flex: 1; padding: 20px; border-right: 1px solid #e5e7eb; display: flex; flex-direction: column; justify-content: center; text-align: center;">
                <h3 style="margin: 0 0 8px 0; font-size: 18px;">{tier['icon']} {tier['name']}</h3>
                <h2 style="color: #3b82f6; margin: 0; font-size: 24px;">{tier['price']}</h2>
            </div>
            <div style="flex: 1; padding: 20px; display: flex; align-items: center;">
                <ul style="margin: 0; padding: 0; list-style: none;">
                    <li style="margin: 6px 0; font-size: 14px; font-weight: bold; color: #3b82f6;">Max Sources: {tier['max_sources']}</li>
                    {features_html}
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        tier = SUBSCRIPTION_TIERS["graduate"]
        features_html = "".join([f"<li style='margin: 6px 0; font-size: 14px;'>{feature}</li>" for feature in tier['features']])
        st.markdown(f"""
        <div class="metric-card" style="border: 2px solid #3b82f6; height: 200px; display: flex;">
            <div style="flex: 1; padding: 20px; border-right: 1px solid #e5e7eb; display: flex; flex-direction: column; justify-content: center; text-align: center;">
                <h3 style="margin: 0 0 8px 0; font-size: 18px;">{tier['icon']} {tier['name']}</h3>
                <h2 style="color: #3b82f6; margin: 0; font-size: 24px;">{tier['price']}</h2>
            </div>
            <div style="flex: 1; padding: 20px; display: flex; align-items: center;">
                <ul style="margin: 0; padding: 0; list-style: none;">
                    <li style="margin: 6px 0; font-size: 14px; font-weight: bold; color: #3b82f6;">Max Sources: {tier['max_sources']}</li>
                    {features_html}
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        tier = SUBSCRIPTION_TIERS["researcher"]
        features_html = "".join([f"<li style='margin: 6px 0; font-size: 14px;'>{feature}</li>" for feature in tier['features']])
        st.markdown(f"""
        <div class="metric-card" style="height: 200px; display: flex;">
            <div style="flex: 1; padding: 20px; border-right: 1px solid #e5e7eb; display: flex; flex-direction: column; justify-content: center; text-align: center;">
                <h3 style="margin: 0 0 8px 0; font-size: 18px;">{tier['icon']} {tier['name']}</h3>
                <h2 style="color: #3b82f6; margin: 0; font-size: 24px;">{tier['price']}</h2>
            </div>
            <div style="flex: 1; padding: 20px; display: flex; align-items: center;">
                <ul style="margin: 0; padding: 0; list-style: none;">
                    <li style="margin: 6px 0; font-size: 14px; font-weight: bold; color: #3b82f6;">Max Sources: {tier['max_sources']}</li>
                    {features_html}
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Call to Action
    st.markdown("### 🚀 Ready to Start Research?")

    if st.button("🔬 Go to Research", type="primary", use_container_width=True):
        st.session_state.page = "🔬 Research"
        st.rerun()

    # Powered by footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 14px; margin: 20px 0;">
        <p>Powered by <strong>ArXiv API</strong> • <strong>Tavily Search</strong> • <strong>OpenAI GPT-4</strong> • <strong>LangChain</strong></p>
    </div>
    """, unsafe_allow_html=True)

def show_research_page():
    """Display the research page with full functionality."""

    # Apply CSS styling
    apply_css_styling()

    # Professional Header
    st.markdown('<h1 class="main-header">⬡ ScholarSynth</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subheader">Search. Synthesize. Succeed.</p>', unsafe_allow_html=True)

    # Initialize research assistant
    research_assistant = get_research_assistant()

    # Professional Sidebar - Subscription System
    with st.sidebar:
        # Create display options with icons
        plan_options = [f"{SUBSCRIPTION_TIERS[tier]['icon']} {SUBSCRIPTION_TIERS[tier]['name']}" for tier in SUBSCRIPTION_TIERS.keys()]
        tier_keys = list(SUBSCRIPTION_TIERS.keys())

        selected_display = st.selectbox(
            "Current Plan:",
            plan_options,
            index=tier_keys.index("researcher"),
            help="Select your subscription tier for demo"
        )

        # Get the actual tier key from the selected display
        user_tier = tier_keys[plan_options.index(selected_display)]

        # Update session state
        set_user_subscription_tier(user_tier)

        st.markdown("### 💡 **Smart Detection Tips**")

        # Show tips only for available agents in current tier
        available_agents = SUBSCRIPTION_TIERS[user_tier]["agents"]
        tips_text = "**Our AI automatically detects the best approach:**\n\n"

        if "student" in available_agents:
            tips_text += """**🎓 Student Queries:**
- "What is..." or "How does..."
- Simple, clear language
- Educational questions

"""

        if "graduate" in available_agents:
            tips_text += """**👨‍🎓 Graduate Queries:**
- "literature review" or "thesis"
- Methodology questions
- Academic analysis

"""

        if "researcher" in available_agents:
            tips_text += """**👨‍🔬 Researcher Queries:**
- "cutting-edge" or "research"
- Collaboration topics
- Innovation questions

"""

        st.markdown(tips_text)

    # Professional Main Content
    st.markdown("### 🔍 Research Query")

    # Query input with professional styling
    query = st.text_area(
        "",
        height=100,
        placeholder="Simply enter your research question here and click search! Our AI will automatically detect the best research approach for your query.",
        label_visibility="collapsed",
        help="Ask any research question and our AI will find relevant academic sources"
    )

    # Professional search button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        search_button = st.button("🔍 Start Research", type="primary", width='stretch', use_container_width=True)

    # Process query with smart auto-detection
    if search_button and query:
        with st.spinner("🔍 Searching academic sources..."):
            # Auto-detect persona from query
            result = research_assistant.process_research_query(query)

            # Check if detected persona is available in user's tier
            if 'detected_persona' in result:
                detected_persona = result['detected_persona']

                # Validate agent access
                if not validate_agent_access(user_tier, f"{detected_persona.title()} Agent"):
                    # Show upgrade prompt for detected persona
                    show_upgrade_prompt(f"{detected_persona.title()} Agent", user_tier)
                    st.stop() # Stop execution if agent not allowed

                # If detected persona is not available, use the highest available persona
                available_agents = SUBSCRIPTION_TIERS[user_tier]["agents"]
                if detected_persona not in available_agents:
                    # Use the highest available agent in their tier
                    if "researcher" in available_agents:
                        fallback_persona = "researcher"
                    elif "graduate" in available_agents:
                        fallback_persona = "graduate"
                    else:
                        fallback_persona = "student"

                    st.warning(f"🎯 **Smart Fallback**: Your query suggests a {detected_persona.title()} approach, but you're using a {tier_info['name']}. Using {fallback_persona.title()} Agent instead.")

                    # Re-process with fallback persona
                    result = research_assistant.process_research_query(query, fallback_persona)

            st.session_state.last_result = result

    # Professional Results Display
    if 'last_result' in st.session_state:
        result = st.session_state.last_result

        # Professional results header
        st.markdown("### 📊 Research Results")

        # Answer with professional styling
        st.markdown("#### 📝 Research Answer")
        st.markdown(f"""
        <div class="metric-card">
            {result['answer']}
        </div>
        """, unsafe_allow_html=True)

        # Sources with professional styling
        st.markdown("#### 📚 Research Sources")
        st.markdown(f"**Found {len(result['sources'])} relevant sources**")

        for i, source in enumerate(result['sources'], 1):
            with st.expander(f"📄 Source {i}: {source['title'][:60]}...", expanded=False):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown(f"**👥 Authors:** {', '.join(source['authors'][:3])}")
                    st.markdown(f"**📅 Published:** {source['published']}")
                    st.markdown(f"**🔗 Source:** {source['source'].title()}")
                    if source['source'] == 'arxiv':
                        st.markdown(f"**📋 ArXiv ID:** {source['arxiv_id']}")
                    else:
                        st.markdown(f"**🌐 URL:** {source['url']}")

                with col2:
                    st.metric("🎯 Relevance", f"{source['relevance_score']:.2f}")

                st.markdown("**📖 Content:**")
                st.markdown(f"""
                <div class="source-card">
                    {source['content'][:500] + "..." if len(source['content']) > 500 else source['content']}
                </div>
                """, unsafe_allow_html=True)

        # Persona information with professional styling
        if 'detected_persona' in result:
            persona = result['detected_persona']
            config = result.get('persona_config', {})
            st.markdown("#### 🎭 Persona Analysis")
            st.markdown(f"""
            <div class="persona-card">
                <h3>{config.get('icon', '👤')} Detected Persona: {config.get('name', 'Unknown')}</h3>
                <div style="display: flex; justify-content: space-between; margin-top: 1rem;">
                    <div><strong>Language Level:</strong> {config.get('language_level', 'Unknown')}</div>
                    <div><strong>Cost Limit:</strong> {config.get('cost_limit', 'Unknown')}</div>
                    <div><strong>Max Sources:</strong> {config.get('max_sources', 'Unknown')}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Professional confidence indicator
        confidence = result['confidence']
        st.markdown("#### 🎯 Research Confidence")
        if confidence > 0.7:
            st.markdown(f"""
            <div class="confidence-high">
                ✅ High Confidence Research Results ({confidence:.2f})
            </div>
            """, unsafe_allow_html=True)
        elif confidence > 0.4:
            st.markdown(f"""
            <div class="confidence-medium">
                ⚠️ Medium Confidence Research Results ({confidence:.2f})
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="confidence-low">
                ❌ Low Confidence Research Results ({confidence:.2f})
            </div>
            """, unsafe_allow_html=True)


def apply_css_styling():
    """Apply the beautiful CSS styling."""
    st.markdown("""
    <style>
        /* Import Google Fonts - Clean and Professional */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

        /* Global Styles - Clean Blue Theme */
        .main {
            font-family: 'Inter', sans-serif;
            background-color: #f1f5f9; /* Clean slate background */
        }

        /* Ultra-aggressive approach - target all possible spacing sources */
        .stApp > div:first-child {
            padding-top: 0 !important;
            margin-top: 0 !important;
        }

        .stApp > div[data-testid="stAppViewContainer"] {
            padding-top: 0 !important;
            margin-top: 0 !important;
        }

        .main .block-container {
            padding-top: 0 !important;
            margin-top: 0 !important;
        }

        .main .block-container > div {
            padding-top: 0 !important;
            margin-top: 0 !important;
        }

        /* Target the main header specifically */
        .main-header {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }

        /* Target any h1 elements */
        h1 {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }

        /* Target Streamlit's main content area */
        .main .element-container:first-child {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }

        /* Additional aggressive targeting */
        .stApp > div:first-child > div {
            padding-top: 0 !important;
            margin-top: 0 !important;
        }

        .main {
            padding-top: 0 !important;
            margin-top: 0 !important;
        }

        /* Target the specific div containing our content */
        .main .block-container > div:first-child {
            padding-top: 0 !important;
            margin-top: 0 !important;
        }

        /* Force remove any remaining spacing */
        .stApp {
            padding-top: 0 !important;
            margin-top: 0 !important;
        }

        /* Target the header container specifically */
        .main-header {
            line-height: 1 !important;
        }

        /* Ultra-minimal bottom spacing - just enough to not touch edge */
        .main .block-container {
            padding-bottom: 0.1rem !important;
        }

        /* Almost no app bottom margin */
        .stApp {
            margin-bottom: 0.05rem !important;
        }

        /* Remove any footer spacing */
        .footer {
            margin-bottom: 0 !important;
            padding-bottom: 0 !important;
        }

        /* Even more aggressive top spacing removal */
        .main .block-container > div:first-child > div:first-child {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }

        /* Header Styling - Professional Blue */
        .main-header {
            font-family: 'Poppins', sans-serif;
            font-size: 2.8rem;
            font-weight: 700;
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); /* Clean blue gradient */
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-align: center;
            margin-bottom: 0.5rem !important;
            margin-top: 0.5rem !important;
            text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        /* Subheader Styling - Clean Typography */
        .subheader {
            font-family: 'Inter', sans-serif;
            font-size: 1rem;
            color: #334155; /* Clean slate text */
            text-align: center;
            margin-bottom: 1rem !important;
            margin-top: 0 !important;
            font-weight: 400;
        }

        /* Card Styling - Clean White Cards */
        .metric-card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 0.25rem 0 !important;
            border: 1px solid #e2e8f0; /* Subtle border */
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.08); /* Blue shadow */
            transition: all 0.3s ease;
        }

        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(59, 130, 246, 0.12); /* Enhanced blue shadow */
        }

        /* Persona Card Styling - Blue Gradient */
        .persona-card {
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); /* Clean blue gradient */
            color: white;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 6px 20px rgba(59, 130, 246, 0.25); /* Blue shadow */
            border: none;
        }

        .persona-card h3 {
            color: white;
            margin: 0 0 0.5rem 0;
            font-weight: 600;
        }

        /* Source Card Styling - Clean White */
        .source-card {
            background: white;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
            border-left: 4px solid #3b82f6; /* Blue accent */
            box-shadow: 0 2px 8px rgba(59, 130, 246, 0.08); /* Blue shadow */
        }

        /* Button Styling - Clean Blue */
        .stButton > button {
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); /* Clean blue gradient */
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 2rem;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.25); /* Blue shadow */
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(59, 130, 246, 0.35); /* Enhanced blue shadow */
        }

        /* Confidence Indicators - Clean Colors */
        .confidence-high { 
            color: #059669; /* Green for high confidence */
            font-weight: 600;
            background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); /* Green background */
            padding: 0.5rem 1rem;
            border-radius: 6px;
            border-left: 4px solid #059669;
        }
        .confidence-medium { 
            color: #f59e0b; /* Amber for medium confidence */
            font-weight: 600;
            background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); /* Amber background */
            padding: 0.5rem 1rem;
            border-radius: 6px;
            border-left: 4px solid #f59e0b;
        }
        .confidence-low { 
            color: #dc2626; /* Red for low confidence */
            font-weight: 600;
            background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); /* Red background */
            padding: 0.5rem 1rem;
            border-radius: 6px;
            border-left: 4px solid #dc2626;
        }

        /* Sidebar Styling - Clean Background */
        .css-1d391kg {
            background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%); /* Clean slate gradient */
        }

        /* Expander Styling - Clean Blue */
        .streamlit-expanderHeader {
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); /* Clean slate gradient */
            border-radius: 8px 8px 0 0;
            font-weight: 600;
            color: #3b82f6; /* Blue text */
        }

        /* Clean Loading - No Spinning Animation */
        .stSpinner {
            display: none;
        }

        /* Text Area Styling - Clean Blue Focus */
        .stTextArea > div > div > textarea {
            border-radius: 8px;
            border: 2px solid #e2e8f0; /* Subtle border */
            transition: all 0.3s ease;
            font-family: 'Inter', sans-serif;
        }

        .stTextArea > div > div > textarea:focus {
            border-color: #3b82f6; /* Blue focus */
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1); /* Blue focus ring */
        }

        /* Selectbox Styling - Clean Blue */
        .stSelectbox > div > div {
            border-radius: 8px;
            border: 2px solid #e2e8f0; /* Subtle border */
            font-family: 'Inter', sans-serif;
        }

        /* Info Box Styling - Clean Blue */
        .stInfo {
            background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); /* Blue background */
            border: 1px solid #93c5fd; /* Blue border */
            border-radius: 8px;
            border-left: 4px solid #3b82f6; /* Blue accent */
        }

        /* Warning Box Styling - Clean Amber */
        .stWarning {
            background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); /* Amber background */
            border: 1px solid #fcd34d; /* Amber border */
            border-radius: 8px;
            border-left: 4px solid #f59e0b; /* Amber accent */
        }

        /* Success Box Styling - Clean Green */
        .stSuccess {
            background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); /* Green background */
            border: 1px solid #6ee7b7; /* Green border */
            border-radius: 8px;
            border-left: 4px solid #059669; /* Green accent */
        }

        /* Error Box Styling - Clean Red */
        .stError {
            background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); /* Red background */
            border: 1px solid #fca5a5; /* Red border */
            border-radius: 8px;
            border-left: 4px solid #dc2626; /* Red accent */
        }

        /* Footer Styling - Clean and Professional */
        .footer {
            text-align: center;
            color: #334155; /* Clean slate text */
            font-size: 0.9rem;
            margin-top: 1rem;
            padding: 1rem 0;
            border-top: 1px solid #e2e8f0; /* Subtle border */
            font-family: 'Inter', sans-serif;
        }

        /* Responsive Design */
        @media (max-width: 768px) {
            .main-header {
                font-size: 2.5rem;
            }
            .metric-card {
                padding: 1rem;
            }
        }
    </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
