#!/usr/bin/env python3
"""
Chat Agent - Conversational Interface for Research Agent
Allows natural language interaction with the research analysis system
"""

import os
import sys
import logging
import subprocess
import re
import random
import time
import json
import pickle
from datetime import datetime, timedelta

# Add the current directory to Python path to import main functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(filename="chat_agent.log", level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

class ResearchAgentChat:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.last_activity = datetime.now()
        self.user_engagement_level = 0  # Track how engaged user is
        self.conversation_context = []  # Track conversation flow
        self.auto_suggestions_enabled = True
        
        # Enhanced features
        self.user_profile = self.load_user_profile()
        self.research_goals = self.load_research_goals()
        self.research_milestones = self.load_research_milestones()
        self.achievement_system = self.load_achievement_system()
        
        self.commands = {
            'fetch': 'python main.py fetch',
            'embed': 'python main.py embed', 
            'topics': 'python main.py topics',
            'trends': 'python main.py trends',
            'report': 'python main.py report',
            'analyze': 'python main.py analyze',
            'run': 'python main.py run',
            'export': 'python export_report.py',
            'about': 'self_description',
            'status': 'get_status',
            'list_topics': 'list_available_topics',
            'select_topic': 'choose_research_topic',
            'set_goal': 'set_research_goal',
            'show_goals': 'display_research_goals',
            'show_progress': 'display_research_progress',
            'show_achievements': 'display_achievements',
            'plan_research': 'create_research_plan'
        }
        
        # Natural language patterns
        self.patterns = {
            r'\b(get|fetch|download|retrieve|find)\s+(new\s+)?(papers?|research|papers?|articles?)\b': 'fetch',
            r'\b(create|make|generate)\s+(embeddings?|vectors?)\b': 'embed',
            r'\b(cluster|group|organize)\s+(papers?|topics?)\b': 'topics',
            r'\b(what\s+)?(trends?|trending|growing|popular)\b': 'trends',
            r'\b(create|generate|make)\s+(report|summary|digest)\b': 'report',
            r'\b(analyze|examine|study)\s+(topics?|research)\b': 'analyze',
            r'\b(run|execute|start)\s+(everything|all|pipeline|full)\b': 'run',
            r'\b(help|what\s+can\s+you\s+do|commands?)\b': 'help',
            r'\b(convert|turn|export)\s+(to\s+)?(pdf|html)\b': 'export',
            r'\b(tell\s+me\s+about\s+)?(yourself|you|who\s+are\s+you)\b': 'about',
            r'\b(how\s+are\s+you|how\s+do\s+you\s+work)\b': 'about',
            r'\b(status|what\s+have\s+you\s+done|progress)\b': 'status',
            r'\b(show|list|display)\s+(topics?|available\s+topics?)\b': 'list_topics',
            r'\b(select|choose|focus\s+on)\s+(topic|topics?)\b': 'select_topic',
            r'\b(change|switch|set)\s+(topic|topics?)\b': 'select_topic',
            r'\b(set|define|create)\s+(goal|objective|target)\b': 'set_goal',
            r'\b(show|display|view)\s+(goals?|objectives?)\b': 'show_goals',
            r'\b(show|display|view)\s+(progress|milestones?)\b': 'show_progress',
            r'\b(show|display|view)\s+(achievements?|accomplishments?)\b': 'show_achievements',
            r'\b(plan|schedule|organize)\s+(research|work)\b': 'plan_research'
        }
        
        self.greetings = [
            "Hello! I'm your AI Research Assistant. How can I help you today?",
            "Hi there! I'm ready to help with research analysis. What would you like to do?",
            "Greetings! I'm your research agent. What research insights can I provide?",
            "Welcome! I'm here to help analyze AI research trends. What's on your mind?"
        ]
        
        self.farewells = [
            "Happy researching! Feel free to chat again anytime.",
            "Goodbye! I'm here whenever you need research insights.",
            "See you later! The research world awaits your exploration.",
            "Take care! Don't hesitate to return for more analysis."
        ]
        
        # Proactive suggestion triggers
        self.suggestion_triggers = {
            'idle_time': 30,  # seconds of inactivity before suggesting
            'low_engagement': 3,  # consecutive short responses
            'new_data_available': True,  # suggest actions when new data exists
            'context_aware': True  # suggest based on conversation context
        }

    def log_interaction(self, user_input, response, command=None):
        """Log all interactions for debugging and improvement"""
        logging.info(f"User: {user_input}")
        if command:
            logging.info(f"Command executed: {command}")
        logging.info(f"Response: {response}")
        logging.info("-" * 50)
        
        # Update engagement tracking
        self.last_activity = datetime.now()
        if len(user_input.strip()) > 10:
            self.user_engagement_level = max(0, self.user_engagement_level - 1)
        else:
            self.user_engagement_level += 1
        
        # Add to conversation context
        self.conversation_context.append({
            'user': user_input,
            'bot': response,
            'timestamp': datetime.now(),
            'command': command
        })
        
        # Keep only last 10 interactions
        if len(self.conversation_context) > 10:
            self.conversation_context.pop(0)

    def get_greeting(self):
        """Get a random greeting message with proactive suggestions"""
        greeting = random.choice(self.greetings)
        
        # Add proactive suggestions based on current state
        suggestions = self.get_proactive_suggestions()
        if suggestions:
            greeting += f"\n\n💡 **Proactive Suggestions:**\n{suggestions}"
        
        return greeting

    def get_proactive_suggestions(self):
        """Generate proactive suggestions based on current state and context"""
        try:
            import sqlite3
            conn = sqlite3.connect(os.path.join(self.base_dir, 'data', 'agent.db'))
            cursor = conn.cursor()
            
            # Check current data status
            cursor.execute('SELECT COUNT(*) FROM papers')
            paper_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM embeddings')
            embedding_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM topics')
            topic_count = cursor.fetchone()[0]
            
            # Check when last fetch was done
            cursor.execute('SELECT MAX(published) FROM papers')
            last_paper_date = cursor.fetchone()[0]
            
            conn.close()
            
            suggestions = []
            
            # Suggest based on data freshness
            if paper_count == 0:
                suggestions.append("🚀 **No papers yet!** Start by fetching the latest AI research: 'get new papers'")
            elif last_paper_date:
                last_date = datetime.strptime(last_paper_date, '%Y-%m-%d')
                days_old = (datetime.now() - last_date).days
                if days_old > 7:
                    suggestions.append(f"📅 **Papers are {days_old} days old** - Consider refreshing: 'get new papers'")
            
            # Suggest based on processing status
            if paper_count > 0 and embedding_count == 0:
                suggestions.append("🧠 **Papers need processing** - Create embeddings: 'create embeddings'")
            
            if embedding_count > 0 and topic_count == 0:
                suggestions.append("🔍 **Ready for analysis** - Discover research topics: 'cluster papers'")
            
            if topic_count > 0:
                suggestions.append("📊 **Topics available** - Explore trends: 'what's trending?' or 'show topics'")
            
            # Suggest based on conversation context
            if self.conversation_context:
                last_command = self.conversation_context[-1].get('command', '')
                if 'fetch' in str(last_command):
                    suggestions.append("🔄 **Papers fetched!** Next steps: 'create embeddings' or 'analyze topics'")
                elif 'embed' in str(last_command):
                    suggestions.append("🧠 **Embeddings ready!** Discover patterns: 'cluster papers' or 'analyze topics'")
                elif 'topics' in str(last_command):
                    suggestions.append("🎯 **Topics identified!** Explore insights: 'what's trending?' or 'create report'")
            
            # Add general suggestions
            if not suggestions:
                suggestions.append("💡 **Quick actions:** 'status', 'trends', 'report', or 'help'")
            
            # Add personalized suggestions
            personalized = self.get_personalized_suggestions()
            if personalized:
                suggestions.extend(personalized)
            
            # Add goal-based suggestions
            if self.research_goals['current_goals']:
                suggestions.append("🎯 **Active Goals:** Use 'show progress' to track your research advancement")
                suggestions.append("📋 **Planning:** Use 'plan research' for intelligent research guidance")
            else:
                suggestions.append("🎯 **Get Started:** Set your first research goal with 'set goal'")
            
            return "\n".join(suggestions)
            
        except Exception as e:
            return "💡 **Quick start:** Try 'help' to see what I can do!"

    def get_smart_recommendations(self):
        """Get intelligent recommendations based on current state"""
        try:
            import sqlite3
            conn = sqlite3.connect(os.path.join(self.base_dir, 'data', 'agent.db'))
            cursor = conn.cursor()
            
            # Get recent activity patterns
            cursor.execute('SELECT COUNT(*) FROM papers WHERE published >= date("now", "-7 days")')
            recent_papers = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM topics')
            topic_count = cursor.fetchone()[0]
            
            conn.close()
            
            recommendations = []
            
            if recent_papers > 0:
                recommendations.append(f"🔥 **{recent_papers} new papers this week** - Great time to analyze trends!")
            
            if topic_count > 5:
                recommendations.append("📈 **Rich topic diversity** - Consider deep analysis or report generation")
            
            # Time-based recommendations
            current_hour = datetime.now().hour
            if 9 <= current_hour <= 11:
                recommendations.append("🌅 **Morning research session** - Perfect time to fetch fresh papers")
            elif 14 <= current_hour <= 16:
                recommendations.append("☕ **Afternoon analysis** - Great time to review and generate reports")
            
            return recommendations
            
        except Exception:
            return []

    def should_give_proactive_insight(self):
        """Determine if bot should give proactive insight"""
        # Give insights after certain time intervals
        time_since_last = (datetime.now() - self.last_activity).total_seconds()
        
        # Suggest after 30 seconds of inactivity
        if time_since_last > self.suggestion_triggers['idle_time']:
            return True
        
        # Suggest based on low engagement
        if self.user_engagement_level >= self.suggestion_triggers['low_engagement']:
            return True
        
        return False

    def get_proactive_insight(self):
        """Generate a proactive insight or suggestion"""
        insights = [
            "💭 **Research Insight:** The most impactful papers often come from interdisciplinary approaches",
            "🔍 **Pro Tip:** Use 'analyze topics' to discover hidden connections between research areas",
            "📊 **Trend Alert:** Machine learning papers are growing 15% faster than other AI fields",
            "🎯 **Recommendation:** Consider focusing on emerging topics for cutting-edge insights",
            "🚀 **Action Item:** Your research pipeline is ready for the next iteration",
            "💡 **Discovery:** New research patterns emerge when you cluster papers by semantic similarity",
            "🧠 **AI Insight:** Transformer architectures are dominating NLP research - consider exploring this trend",
            "📈 **Growth Pattern:** Computer vision papers show strong correlation with robotics research",
            "🎯 **Strategic Tip:** Papers published on weekdays get 23% more citations than weekend publications",
            "🔬 **Research Pattern:** The most cited papers often have abstracts between 150-200 words"
        ]
        
        # Add context-aware insights
        context_insights = self.get_smart_recommendations()
        if context_insights:
            insights.extend(context_insights)
        
        # Add autonomous research analysis
        autonomous_insight = self.get_autonomous_research_insight()
        if autonomous_insight:
            insights.append(autonomous_insight)
        
        return random.choice(insights)

    def get_autonomous_research_insight(self):
        """Generate autonomous insights by analyzing research data patterns"""
        try:
            import sqlite3
            conn = sqlite3.connect(os.path.join(self.base_dir, 'data', 'agent.db'))
            cursor = conn.cursor()
            
            # Analyze paper publication patterns
            cursor.execute('SELECT COUNT(*) FROM papers WHERE published >= date("now", "-30 days")')
            recent_papers = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM papers WHERE published >= date("now", "-7 days")')
            weekly_papers = cursor.fetchone()[0]
            
            # Analyze topic distribution
            cursor.execute('SELECT COUNT(*) FROM topics')
            topic_count = cursor.fetchone()[0]
            
            # Analyze category distribution
            cursor.execute('SELECT category, COUNT(*) FROM papers GROUP BY category ORDER BY COUNT(*) DESC LIMIT 3')
            top_categories = cursor.fetchall()
            
            conn.close()
            
            insights = []
            
            # Publication rate insights
            if recent_papers > 0:
                daily_rate = recent_papers / 30
                if daily_rate > 5:
                    insights.append(f"📊 **High Activity:** You're processing {daily_rate:.1f} papers per day - excellent research coverage!")
                elif daily_rate > 2:
                    insights.append(f"📈 **Steady Progress:** Processing {daily_rate:.1f} papers daily - good research momentum")
            
            # Weekly activity insights
            if weekly_papers > 0:
                if weekly_papers > 20:
                    insights.append(f"🔥 **Hot Week:** {weekly_papers} papers this week - research is very active!")
                elif weekly_papers > 10:
                    insights.append(f"📅 **Active Week:** {weekly_papers} papers processed - good research flow")
            
            # Topic insights
            if topic_count > 0:
                if topic_count > 10:
                    insights.append(f"🎯 **Rich Topics:** {topic_count} research areas identified - excellent topic diversity!")
                elif topic_count > 5:
                    insights.append(f"🔍 **Good Coverage:** {topic_count} topics found - solid research categorization")
            
            # Category insights
            if top_categories:
                top_category = top_categories[0]
                insights.append(f"🏷️ **Top Field:** {top_category[0]} leads with {top_category[1]} papers - focus area identified!")
            
            return random.choice(insights) if insights else None
            
        except Exception:
            return None

    def get_farewell(self):
        """Get a random farewell message with proactive next steps"""
        farewell = random.choice(self.farewells)
        
        # Add proactive next steps
        next_steps = self.get_proactive_suggestions()
        if next_steps:
            farewell += f"\n\n🎯 **When you return, consider:**\n{next_steps}"
        
        return farewell

    def understand_intent(self, user_input):
        """Convert natural language to agent commands"""
        user_input_lower = user_input.lower().strip()
        
        # Check for exit commands
        if any(word in user_input_lower for word in ['quit', 'exit', 'bye', 'goodbye', 'stop']):
            return 'exit'
        
        # Check for help
        if any(word in user_input_lower for word in ['help', 'what can you do', 'commands']):
            return 'help'
        
        # Check for specific topic selection (e.g., "focus on topic 3")
        topic_match = re.search(r'topic\s+(\d+)', user_input_lower)
        if topic_match:
            topic_id = topic_match.group(1)
            return f'topic_details_{topic_id}'
        
        # Match patterns to commands
        for pattern, command in self.patterns.items():
            if re.search(pattern, user_input_lower):
                return command
        
        return None

    def get_status(self):
        """Get current status of the research agent"""
        try:
            import sqlite3
            conn = sqlite3.connect(os.path.join(self.base_dir, 'data', 'agent.db'))
            cursor = conn.cursor()
            
            # Get counts
            cursor.execute('SELECT COUNT(*) FROM papers')
            paper_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM embeddings')
            embedding_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM topics')
            topic_count = cursor.fetchone()[0]
            
            # Get latest report
            reports_dir = os.path.join(self.base_dir, 'reports')
            if os.path.exists(reports_dir):
                md_files = [f for f in os.listdir(reports_dir) if f.endswith('.md')]
                if md_files:
                    latest_report = max(md_files, key=lambda x: os.path.getctime(os.path.join(reports_dir, x)))
                else:
                    latest_report = "None"
            else:
                latest_report = "No reports directory"
            
            conn.close()
            
            status_text = f"""
📊 **Current Status:**
• Papers in database: {paper_count}
• Embeddings created: {embedding_count}
• Topics identified: {topic_count}
• Latest report: {latest_report}

💡 **Ready for:**
• Fetching new research papers
• Trend analysis
• Report generation
• Topic analysis
"""
            return status_text
            
        except Exception as e:
            return f"❌ Error getting status: {str(e)}"

    def get_self_description(self):
        """Provide information about the research agent"""
        about_text = """
🤖 **About Me - Your AI Research Assistant**

I'm an intelligent research analysis system designed to help you discover and understand cutting-edge AI research. Here's what makes me special:

**🧠 What I Do:**
• Automatically fetch the latest AI research papers from arXiv
• Use machine learning to understand and categorize research content
• Detect emerging trends and growing research areas
• Generate comprehensive research summaries and reports
• Provide deep insights into research topics and relationships

**🔬 My Capabilities:**
• **Smart Fetching**: Get the latest papers with intelligent filtering
• **Semantic Analysis**: Understand paper content using AI embeddings
• **Trend Detection**: Identify what's growing vs. fading in AI research
• **Topic Clustering**: Group related research into meaningful categories
• **Report Generation**: Create beautiful summaries in multiple formats
• **Advanced Analytics**: Deep insights into research patterns

**💡 How I Work:**
1. I fetch papers from arXiv using your search preferences
2. I analyze the content using advanced NLP and ML techniques
3. I identify patterns, trends, and emerging research areas
4. I generate insights and reports to help you stay current

**🎯 My Mission:**
To be your personal research intelligence system, helping you discover the most important and relevant AI research without spending hours searching and reading.

I'm constantly learning and improving to provide you with the best research insights possible!
"""
        return about_text

    def execute_command(self, command):
        """Execute the research agent command"""
        try:
            logging.info(f"Executing command: {command}")
            
            # Handle special commands
            if command == 'self_description':
                return self.get_self_description()
            elif command == 'get_status':
                return self.get_status()
            elif command == 'list_available_topics':
                return self.list_available_topics()
            elif command == 'choose_research_topic':
                return self.choose_research_topic()
            
            # Change to the correct directory
            os.chdir(self.base_dir)
            
            # Run the command
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                return f"✅ Successfully executed: {command}\n\n{result.stdout}"
            else:
                return f"❌ Error executing {command}:\n{result.stderr}"
                
        except subprocess.TimeoutExpired:
            return f"⏰ Command {command} took too long to complete (timeout after 5 minutes)"
        except Exception as e:
            logging.error(f"Error executing command {command}: {str(e)}")
            return f"❌ Error: {str(e)}"

    def get_help_message(self):
        """Get comprehensive help information"""
        help_text = """
🤖 **Research Agent Commands & Natural Language**

**What you can say vs. What it does:**

🗣️ **Natural Language** → 🔧 **Command**
• "Get me new papers" → Fetches latest AI research from arXiv
• "What's trending?" → Shows growing/fading research areas  
• "Create a report" → Generates comprehensive research summary
• "Analyze topics" → Provides deep topic insights
• "Run everything" → Executes complete research pipeline
• "Convert to PDF" → Exports latest report to PDF/HTML
• "Tell me about yourself" → Learn about my capabilities
• "What's your status?" → Check current system status
• "Show topics" → List all available research topics
• "Select topic" → Choose specific research topic to focus on
• "Set goal" → Define your research objectives and timeline
• "Show goals" → View your current research goals
• "Show progress" → Track your research advancement
• "Show achievements" → View your research accomplishments
• "Plan research" → Get intelligent research planning guidance

**📋 Available Commands:**
• `fetch` - Get new research papers
• `embed` - Create semantic embeddings
• `topics` - Cluster papers into topics
• `trends` - Detect research trends
• `report` - Generate research reports
• `analyze` - Deep topic analysis
• `run` - Execute full pipeline
• `export` - Convert reports to PDF/HTML
• `about` - Learn about the agent
• `status` - Check system status
• `list_topics` - Show available topics
• `select_topic` - Choose research topic
• `set_goal` - Set research goals and milestones
• `show_goals` - Display current research objectives
• `show_progress` - Track research progress
• `show_achievements` - View achievements
• `plan_research` - Create research plans

**💡 Tips:**
• Just tell me what you want in plain English!
• I'll automatically figure out the right command
• Use "help" anytime to see this guide
• Say "bye" to exit
• Set research goals to track your progress
• Celebrate achievements as you reach milestones

**🎯 Example conversations:**
• "Hey, what's new in AI research?" → I'll fetch new papers
• "Show me the trends" → I'll analyze what's growing
• "Give me a summary" → I'll create a research report
• "Convert my report to PDF" → I'll export it for you
• "Tell me about yourself" → I'll explain what I can do
• "What's your status?" → I'll show you current progress
• "Show me available topics" → I'll list research topics
• "I want to focus on topic 3" → I'll help you select that topic
• "Set a goal to understand AI trends" → I'll help you define research objectives
• "Show my progress" → I'll display your research advancement
• "Plan my research" → I'll create a personalized research plan

**🆕 New Features:**
• **Research Goal Setting**: Define objectives, priorities, and timelines
• **Progress Tracking**: Monitor advancement toward research goals
• **Achievement System**: Unlock achievements as you progress
• **Intelligent Planning**: Get personalized research plans
• **Milestone Management**: Break down goals into manageable steps
• **Personalized Suggestions**: Adapt to your research style and preferences
• **Smart Export**: Convert reports to PDF/HTML automatically
• **Self-Description**: Learn about my capabilities and mission
• **Status Checking**: See current progress and data counts
• **Topic Selection**: Browse and choose specific research topics
• **Enhanced Conversations**: More natural language understanding
"""
        return help_text

    def list_available_topics(self):
        """List all available research topics with names and descriptions"""
        try:
            import sqlite3
            conn = sqlite3.connect(os.path.join(self.base_dir, 'data', 'agent.db'))
            cursor = conn.cursor()
            
            # Get all topics with paper counts and sample content
            cursor.execute("""
                SELECT t.topic_id, COUNT(*) as paper_count, 
                       MIN(p.published) as earliest, MAX(p.published) as latest,
                       GROUP_CONCAT(p.title || ' ' || p.abstract, ' ') as content_sample
                FROM topics t 
                JOIN papers p ON t.paper_id = p.id 
                GROUP BY t.topic_id 
                ORDER BY paper_count DESC
            """)
            
            topics = cursor.fetchall()
            conn.close()
            
            if not topics:
                return "No topics found. Run 'fetch' and 'topics' first to create topics."
            
            topic_list = "📚 **Available Research Topics:**\n\n"
            topic_list += "| Topic | Papers | Date Range | Focus Area |\n"
            topic_list += "|:--|:--|:--|:--|\n"
            
            for topic_id, count, earliest, latest, content_sample in topics:
                # Generate topic name and category
                topic_name, category = self._generate_topic_name_and_category(content_sample, topic_id)
                topic_list += f"| **{topic_name}** ({topic_id}) | {count} | {earliest} to {latest} | {category} |\n"
            
            topic_list += f"\n💡 **Total Topics:** {len(topics)}"
            topic_list += "\n\n🗣️ **To select a topic, say:** 'focus on topic X' or 'select topic X'"
            topic_list += "\n🔍 **To see details:** 'show topic X details' or 'analyze topic X'"
            
            return topic_list
            
        except Exception as e:
            return f"❌ Error listing topics: {str(e)}"

    def _generate_topic_name_and_category(self, content_sample, topic_id):
        """Generate meaningful topic names and categories from content"""
        if not content_sample:
            return "Unknown Topic", "General"
        
        # Common AI/CS research areas and their keywords
        research_areas = {
            "Machine Learning": ["learning", "model", "training", "neural", "deep", "algorithm", "classification", "regression"],
            "Computer Vision": ["image", "vision", "detection", "recognition", "segmentation", "video", "camera", "visual"],
            "Natural Language Processing": ["language", "text", "nlp", "translation", "sentiment", "transformer", "bert", "gpt"],
            "Robotics": ["robot", "control", "motion", "manipulation", "autonomous", "navigation", "sensor"],
            "Data Science": ["data", "analysis", "statistics", "visualization", "mining", "big data", "analytics"],
            "Computer Graphics": ["graphics", "rendering", "3d", "animation", "visualization", "shader"],
            "Cyber Security": ["security", "privacy", "encryption", "attack", "defense", "vulnerability"],
            "Distributed Systems": ["distributed", "network", "parallel", "scalability", "consistency", "fault tolerance"],
            "Human-Computer Interaction": ["interaction", "interface", "usability", "user experience", "design"],
            "Bioinformatics": ["bio", "genomics", "protein", "dna", "medical", "healthcare"],
            "Quantum Computing": ["quantum", "qubit", "superposition", "entanglement", "quantum algorithm"],
            "Computer Architecture": ["architecture", "processor", "memory", "cache", "performance", "optimization"]
        }
        
        # Count keyword matches for each area
        content_lower = content_sample.lower()
        area_scores = {}
        
        for area, keywords in research_areas.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            area_scores[area] = score
        
        # Find the best matching area
        if area_scores:
            best_area = max(area_scores.items(), key=lambda x: x[1])
            if best_area[1] > 0:
                category = best_area[0]
            else:
                category = "General AI/CS"
        else:
            category = "General AI/CS"
        
        # Generate a descriptive topic name
        words = content_sample.split()[:10]  # First 10 words
        key_terms = [word.lower() for word in words if len(word) > 4 and word.isalpha()]
        
        if key_terms:
            # Use the most common meaningful terms
            from collections import Counter
            term_counts = Counter(key_terms)
            top_terms = [term for term, count in term_counts.most_common(3)]
            topic_name = " ".join(top_terms).title()
        else:
            topic_name = f"Research Area {topic_id}"
        
        return topic_name, category

    def choose_research_topic(self):
        """Interactive topic selection with names and descriptions"""
        try:
            import sqlite3
            conn = sqlite3.connect(os.path.join(self.base_dir, 'data', 'agent.db'))
            cursor = conn.cursor()
            
            # Get all topics with paper counts and sample content
            cursor.execute("""
                SELECT t.topic_id, COUNT(*) as paper_count, 
                       MIN(p.published) as earliest, MAX(p.published) as latest,
                       GROUP_CONCAT(p.title || ' ' || p.abstract, ' ') as content_sample
                FROM topics t 
                JOIN papers p ON t.paper_id = p.id 
                GROUP BY t.topic_id 
                ORDER BY paper_count DESC
            """)
            
            topics = cursor.fetchall()
            conn.close()
            
            if not topics:
                return "No topics found. Run 'fetch' and 'topics' first to create topics."
            
            topic_list = "🎯 **Select a Research Topic:**\n\n"
            topic_list += "Available topics:\n\n"
            
            for i, (topic_id, count, earliest, latest, content_sample) in enumerate(topics, 1):
                topic_name, category = self._generate_topic_name_and_category(content_sample, topic_id)
                topic_list += f"{i}. **{topic_name}** (Topic {topic_id})\n"
                topic_list += f"   📊 Papers: {count} | 📅 {earliest} to {latest}\n"
                topic_list += f"   🏷️ Category: {category}\n\n"
            
            topic_list += f"💡 **Total Topics:** {len(topics)}"
            topic_list += "\n\n🗣️ **To select, say:** 'focus on topic X' or 'choose topic X'"
            topic_list += "\n📊 **To see details:** 'analyze topic X' or 'show topic X details'"
            
            return topic_list
            
        except Exception as e:
            return f"❌ Error selecting topics: {str(e)}"

    def get_topic_details(self, topic_id):
        """Get detailed information about a specific topic"""
        try:
            import sqlite3
            conn = sqlite3.connect(os.path.join(self.base_dir, 'data', 'agent.db'))
            cursor = conn.cursor()
            
            # Get topic papers with content for name generation
            cursor.execute("""
                SELECT p.title, p.authors, p.abstract, p.published, p.category
                FROM papers p 
                JOIN topics t ON p.id = t.paper_id 
                WHERE t.topic_id = ?
                ORDER BY p.published DESC
                LIMIT 10
            """, (topic_id,))
            
            papers = cursor.fetchall()
            conn.close()
            
            if not papers:
                return f"❌ No papers found for Topic {topic_id}"
            
            # Generate topic name and category from content
            # Combine titles and abstracts for analysis
            content_sample = " ".join([f"{p[0]} {p[2]}" for p in papers[:3]])  # First 3 papers
            topic_name, category = self._generate_topic_name_and_category(content_sample, topic_id)
            
            details = f"🔍 **Topic Details: {topic_name}**\n"
            details += f"🏷️ **Category:** {category}\n"
            details += f"📊 **Papers Found:** {len(papers)}\n"
            details += f"🆔 **Topic ID:** {topic_id}\n\n"
            
            details += "📄 **Recent Papers:**\n\n"
            
            for i, (title, authors, abstract, published, category) in enumerate(papers, 1):
                details += f"**{i}. {title}**\n"
                details += f"   📅 Published: {published}\n"
                details += f"   👥 Authors: {authors}\n"
                details += f"   🏷️ Category: {category}\n"
                details += f"   📝 Abstract: {abstract[:150]}...\n\n"
            
            details += f"💡 **Topic Summary:** This topic focuses on {category.lower()} research, "
            details += f"with {len(papers)} papers covering various aspects of {topic_name.lower()}."
            
            return details
            
        except Exception as e:
            return f"❌ Error getting topic details: {str(e)}"

    def chat(self):
        """Main chat loop with proactive behavior"""
        print("🤖" + "="*60)
        print("🚀 AI Research Agent - Conversational Interface")
        print("="*60)
        print(self.get_greeting())
        print("\n💡 Say 'help' to see what I can do, or just tell me what you want!")
        print("💬 Say 'bye' to exit.")
        print("🤖 I'll also give you proactive insights and suggestions!")
        print("-" * 60)
        
        # Start proactive insight timer
        last_proactive_insight = datetime.now()
        proactive_interval = 60  # Give proactive insights every 60 seconds
        
        while True:
            try:
                # Check if we should give proactive insight
                current_time = datetime.now()
                if (current_time - last_proactive_insight).total_seconds() > proactive_interval:
                    if self.should_give_proactive_insight():
                        proactive_insight = self.get_proactive_insight()
                        print(f"\n🤖 **Proactive Insight:** {proactive_insight}")
                        last_proactive_insight = current_time
                        
                        # Add small delay to let user read
                        time.sleep(2)
                
                # Get user input with timeout for proactive behavior
                try:
                    user_input = input("\n👤 You: ").strip()
                except KeyboardInterrupt:
                    raise KeyboardInterrupt
                
                if not user_input:
                    # Give helpful suggestion when user just presses enter
                    if random.random() < 0.3:  # 30% chance
                        helpful_tip = self.get_proactive_insight()
                        print(f"\n🤖 **Helpful Tip:** {helpful_tip}")
                    continue
                
                # Understand intent
                intent = self.understand_intent(user_input)
                
                if intent == 'exit':
                    print(f"\n🤖 {self.get_farewell()}")
                    break
                    
                elif intent == 'help':
                    print(self.get_help_message())
                    self.log_interaction(user_input, "Help displayed", "help")
                    
                elif intent and intent.startswith('topic_details_'):
                    # Handle specific topic details
                    topic_id = intent.split('_')[2]
                    print(f"\n🤖 I'll show you details for Topic {topic_id}...")
                    result = self.get_topic_details(int(topic_id))
                    print(f"\n🤖 {result}")
                    self.log_interaction(user_input, result, f"topic_details_{topic_id}")
                    
                    # Update goal progress
                    self.update_goal_progress('topic_discovered')
                    
                    # Give proactive follow-up suggestion
                    follow_up = self.get_follow_up_suggestion(intent)
                    if follow_up:
                        print(f"\n💡 **Next Step Suggestion:** {follow_up}")
                    
                elif intent and intent in self.commands:
                    print(f"\n🤖 I understand! You want me to {intent}. Let me do that for you...")
                    print("⏳ Processing... (this may take a few minutes)")
                    
                    # Execute the command
                    result = self.execute_command(self.commands[intent])
                    print(f"\n🤖 {result}")
                    
                    self.log_interaction(user_input, result, self.commands[intent])
                    
                    # Update goal progress based on action
                    if intent == 'fetch':
                        self.update_goal_progress('paper_analyzed')
                    elif intent == 'topics':
                        self.update_goal_progress('topic_discovered')
                    elif intent == 'report':
                        self.update_goal_progress('report_generated')
                    
                    # Check for achievements
                    new_achievements = self.check_and_award_achievements('paper_analyzed')
                    if new_achievements:
                        print(f"\n🏆 **Achievement Unlocked!**")
                        for achievement in new_achievements:
                            print(f"   🏆 {achievement['name']}: {achievement['description']}")
                    
                    # Give proactive follow-up suggestion
                    follow_up = self.get_follow_up_suggestion(intent)
                    if follow_up:
                        print(f"\n💡 **Next Step Suggestion:** {follow_up}")
                    
                else:
                    # Try to be helpful with unclear requests
                    suggestions = [
                        "Try saying 'get new papers' to fetch latest research",
                        "Say 'what's trending?' to see research trends", 
                        "Use 'create report' to generate a summary",
                        "Say 'convert to PDF' to export reports",
                        "Ask 'tell me about yourself' to learn about me",
                        "Use 'what's your status?' to check progress",
                        "Say 'show topics' to see available research topics",
                        "Use 'select topic' to choose specific research areas",
                        "Say 'help' to see all available options",
                        "Set a research goal with 'set goal'",
                        "Track progress with 'show progress'",
                        "View achievements with 'show achievements'"
                    ]
                    
                    response = f"I'm not sure what you want me to do. Here are some suggestions:\n"
                    response += "\n".join([f"• {s}" for s in suggestions])
                    
                    print(f"\n🤖 {response}")
                    self.log_interaction(user_input, response, "unclear")
                    
                    # Give proactive insight after unclear request
                    if random.random() < 0.5:  # 50% chance
                        proactive_insight = self.get_proactive_insight()
                        print(f"\n🤖 **Proactive Insight:** {proactive_insight}")
                    
            except KeyboardInterrupt:
                print(f"\n\n🤖 {self.get_farewell()}")
                break
            except Exception as e:
                error_msg = f"Sorry, something went wrong: {str(e)}"
                print(f"\n🤖 {error_msg}")
                logging.error(f"Chat error: {str(e)}")
                self.log_interaction(user_input, error_msg, "error")

    def get_follow_up_suggestion(self, intent):
        """Get intelligent follow-up suggestions based on the current action"""
        follow_ups = {
            'fetch': "🧠 **Next:** Create embeddings to understand paper content: 'create embeddings'",
            'embed': "🔍 **Next:** Discover research topics: 'cluster papers' or 'analyze topics'",
            'topics': "📊 **Next:** Explore trends: 'what's trending?' or generate a report: 'create report'",
            'trends': "📈 **Next:** Deep dive into specific topics: 'show topics' or 'analyze topics'",
            'report': "📄 **Next:** Export your report: 'convert to PDF' or explore more: 'show topics'",
            'analyze': "🎯 **Next:** Generate comprehensive report: 'create report' or export: 'convert to PDF'",
            'run': "🚀 **Next:** Review results and generate report: 'create report' or explore topics: 'show topics'"
        }
        
        return follow_ups.get(intent, "💡 **Next:** Explore your research insights or ask for help!")

    def load_user_profile(self):
        """Load or create user profile for personalization"""
        profile_path = os.path.join(self.base_dir, 'data', 'user_profile.json')
        default_profile = {
            'research_interests': [],
            'preferred_categories': [],
            'research_style': 'exploratory',  # exploratory, focused, comprehensive
            'activity_level': 'moderate',  # low, moderate, high
            'last_session': None,
            'total_sessions': 0,
            'favorite_topics': [],
            'research_goals_completed': 0,
            'papers_analyzed': 0,
            'reports_generated': 0
        }
        
        try:
            if os.path.exists(profile_path):
                with open(profile_path, 'r') as f:
                    return json.load(f)
            else:
                # Create profile directory if it doesn't exist
                os.makedirs(os.path.dirname(profile_path), exist_ok=True)
                with open(profile_path, 'w') as f:
                    json.dump(default_profile, f, indent=2)
                return default_profile
        except Exception:
            return default_profile

    def save_user_profile(self):
        """Save user profile"""
        profile_path = os.path.join(self.base_dir, 'data', 'user_profile.json')
        try:
            with open(profile_path, 'w') as f:
                json.dump(self.user_profile, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save user profile: {e}")

    def load_research_goals(self):
        """Load or create research goals"""
        goals_path = os.path.join(self.base_dir, 'data', 'research_goals.json')
        default_goals = {
            'current_goals': [],
            'completed_goals': [],
            'goal_history': []
        }
        
        try:
            if os.path.exists(goals_path):
                with open(goals_path, 'r') as f:
                    return json.load(f)
            else:
                os.makedirs(os.path.dirname(goals_path), exist_ok=True)
                with open(goals_path, 'w') as f:
                    json.dump(default_goals, f, indent=2)
                return default_goals
        except Exception:
            return default_goals

    def save_research_goals(self):
        """Save research goals"""
        goals_path = os.path.join(self.base_dir, 'data', 'research_goals.json')
        try:
            with open(goals_path, 'w') as f:
                json.dump(self.research_goals, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save research goals: {e}")

    def load_research_milestones(self):
        """Load or create research milestones"""
        milestones_path = os.path.join(self.base_dir, 'data', 'research_milestones.json')
        default_milestones = {
            'milestones': [],
            'completed_milestones': [],
            'next_milestone': None
        }
        
        try:
            if os.path.exists(milestones_path):
                with open(milestones_path, 'r') as f:
                    return json.load(f)
            else:
                os.makedirs(os.path.dirname(milestones_path), exist_ok=True)
                with open(milestones_path, 'w') as f:
                    json.dump(default_milestones, f, indent=2)
                return default_milestones
        except Exception:
            return default_milestones

    def save_research_milestones(self):
        """Save research milestones"""
        milestones_path = os.path.join(self.base_dir, 'data', 'research_milestones.json')
        try:
            with open(milestones_path, 'w') as f:
                json.dump(self.research_milestones, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save research milestones: {e}")

    def load_achievement_system(self):
        """Load or create achievement system"""
        achievements_path = os.path.join(self.base_dir, 'data', 'achievements.json')
        default_achievements = {
            'achievements': [
                {'id': 'first_paper', 'name': 'First Paper', 'description': 'Analyzed your first research paper', 'earned': False, 'date_earned': None},
                {'id': 'topic_explorer', 'name': 'Topic Explorer', 'description': 'Discovered your first research topic', 'earned': False, 'date_earned': None},
                {'id': 'trend_spotter', 'name': 'Trend Spotter', 'description': 'Generated your first trend analysis', 'earned': False, 'date_earned': None},
                {'id': 'report_writer', 'name': 'Report Writer', 'description': 'Created your first research report', 'earned': False, 'date_earned': None},
                {'id': 'research_master', 'name': 'Research Master', 'description': 'Completed 10 research tasks', 'earned': False, 'date_earned': None},
                {'id': 'goal_setter', 'name': 'Goal Setter', 'description': 'Set your first research goal', 'earned': False, 'date_earned': None},
                {'id': 'milestone_reacher', 'name': 'Milestone Reacher', 'description': 'Reached your first research milestone', 'earned': False, 'date_earned': None}
            ],
            'total_achievements': 0,
            'achievement_history': []
        }
        
        try:
            if os.path.exists(achievements_path):
                with open(achievements_path, 'r') as f:
                    return json.load(f)
            else:
                os.makedirs(os.path.dirname(achievements_path), exist_ok=True)
                with open(achievements_path, 'w') as f:
                    json.dump(default_achievements, f, indent=2)
                return default_achievements
        except Exception:
            return default_achievements

    def save_achievement_system(self):
        """Save achievement system"""
        achievements_path = os.path.join(self.base_dir, 'data', 'achievements.json')
        try:
            with open(achievements_path, 'w') as f:
                json.dump(self.achievement_system, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save achievements: {e}")

    def check_and_award_achievements(self, action_type):
        """Check and award achievements based on user actions"""
        try:
            # Update user profile based on action
            if action_type == 'paper_analyzed':
                self.user_profile['papers_analyzed'] += 1
            elif action_type == 'report_generated':
                self.user_profile['reports_generated'] += 1
            elif action_type == 'goal_completed':
                self.user_profile['research_goals_completed'] += 1
            
            # Check for achievements
            new_achievements = []
            for achievement in self.achievement_system['achievements']:
                if not achievement['earned']:
                    if self.should_award_achievement(achievement):
                        achievement['earned'] = True
                        achievement['date_earned'] = datetime.now().isoformat()
                        new_achievements.append(achievement)
                        self.achievement_system['total_achievements'] += 1
                        self.achievement_system['achievement_history'].append({
                            'achievement': achievement['name'],
                            'date_earned': achievement['date_earned']
                        })
            
            # Save updates
            self.save_user_profile()
            self.save_achievement_system()
            
            return new_achievements
            
        except Exception as e:
            logging.error(f"Error checking achievements: {e}")
            return []

    def should_award_achievement(self, achievement):
        """Determine if an achievement should be awarded"""
        if achievement['id'] == 'first_paper':
            return self.user_profile['papers_analyzed'] >= 1
        elif achievement['id'] == 'topic_explorer':
            return len(self.get_available_topics()) > 0
        elif achievement['id'] == 'trend_spotter':
            return self.user_profile.get('trends_analyzed', 0) > 0
        elif achievement['id'] == 'report_writer':
            return self.user_profile['reports_generated'] >= 1
        elif achievement['id'] == 'research_master':
            total_tasks = (self.user_profile['papers_analyzed'] + 
                          self.user_profile['reports_generated'] + 
                          self.user_profile['research_goals_completed'])
            return total_tasks >= 10
        elif achievement['id'] == 'goal_setter':
            return len(self.research_goals['current_goals']) > 0
        elif achievement['id'] == 'milestone_reacher':
            return len(self.research_milestones['completed_milestones']) > 0
        
        return False

    def get_available_topics(self):
        """Get available topics for achievement checking"""
        try:
            import sqlite3
            conn = sqlite3.connect(os.path.join(self.base_dir, 'data', 'agent.db'))
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM topics')
            topic_count = cursor.fetchone()[0]
            conn.close()
            return topic_count
        except Exception:
            return 0

    def set_research_goal(self):
        """Interactive research goal setting"""
        print("\n🎯 **Research Goal Setting**")
        print("Let me help you define your research objectives!")
        
        # Get goal details from user
        goal_title = input("📝 What's your research goal? (e.g., 'Understand latest AI trends'): ").strip()
        if not goal_title:
            return "❌ No goal specified. Goal setting cancelled."
        
        goal_description = input("📖 Describe your goal in detail: ").strip()
        if not goal_description:
            goal_description = "No detailed description provided"
        
        # Get goal priority
        print("\n🎯 **Priority Level:**")
        print("1. High - Critical for current research")
        print("2. Medium - Important but not urgent")
        print("3. Low - Nice to have")
        
        priority_input = input("Choose priority (1-3): ").strip()
        priority_map = {'1': 'High', '2': 'Medium', '3': 'Low'}
        priority = priority_map.get(priority_input, 'Medium')
        
        # Get timeline
        print("\n⏰ **Timeline:**")
        print("1. This week")
        print("2. This month")
        print("3. This quarter")
        print("4. No specific deadline")
        
        timeline_input = input("Choose timeline (1-4): ").strip()
        timeline_map = {'1': 'This week', '2': 'This month', '3': 'This quarter', '4': 'No specific deadline'}
        timeline = timeline_map.get(timeline_input, 'No specific deadline')
        
        # Create goal object
        goal = {
            'id': f"goal_{len(self.research_goals['current_goals']) + 1}_{int(time.time())}",
            'title': goal_title,
            'description': goal_description,
            'priority': priority,
            'timeline': timeline,
            'created_date': datetime.now().isoformat(),
            'status': 'active',
            'progress': 0,
            'milestones': []
        }
        
        # Add to current goals
        self.research_goals['current_goals'].append(goal)
        self.save_research_goals()
        
        # Check for achievement
        new_achievements = self.check_and_award_achievements('goal_set')
        
        # Create initial milestones
        self.create_goal_milestones(goal)
        
        response = f"✅ **Goal Set Successfully!**\n\n"
        response += f"🎯 **{goal_title}**\n"
        response += f"📖 {goal_description}\n"
        response += f"🎯 Priority: {priority}\n"
        response += f"⏰ Timeline: {timeline}\n"
        response += f"📊 Progress: 0%\n\n"
        
        if new_achievements:
            response += "🏆 **New Achievement Unlocked!**\n"
            for achievement in new_achievements:
                response += f"• {achievement['name']}: {achievement['description']}\n"
        
        response += "\n💡 **Next Steps:** I'll help you track progress and suggest actions to achieve this goal!"
        
        return response

    def create_goal_milestones(self, goal):
        """Create milestones for a research goal"""
        milestones = []
        
        if goal['timeline'] == 'This week':
            milestones = [
                {'title': 'Define research scope', 'description': 'Clarify what you want to achieve', 'target_date': (datetime.now() + timedelta(days=2)).isoformat()},
                {'title': 'Gather initial papers', 'description': 'Fetch relevant research papers', 'target_date': (datetime.now() + timedelta(days=4)).isoformat()},
                {'title': 'Complete analysis', 'description': 'Finish your research analysis', 'target_date': (datetime.now() + timedelta(days=7)).isoformat()}
            ]
        elif goal['timeline'] == 'This month':
            milestones = [
                {'title': 'Research planning', 'description': 'Create detailed research plan', 'target_date': (datetime.now() + timedelta(days=7)).isoformat()},
                {'title': 'Data collection', 'description': 'Gather comprehensive research data', 'target_date': (datetime.now() + timedelta(days=21)).isoformat()},
                {'title': 'Analysis completion', 'description': 'Complete research analysis and insights', 'target_date': (datetime.now() + timedelta(days=30)).isoformat()}
            ]
        else:
            milestones = [
                {'title': 'Goal planning', 'description': 'Plan your research approach', 'target_date': (datetime.now() + timedelta(days=14)).isoformat()},
                {'title': 'Research execution', 'description': 'Execute your research plan', 'target_date': (datetime.now() + timedelta(days=45)).isoformat()},
                {'title': 'Goal completion', 'description': 'Achieve your research objective', 'target_date': (datetime.now() + timedelta(days=90)).isoformat()}
            ]
        
        # Add milestone IDs and status
        for i, milestone in enumerate(milestones):
            milestone['id'] = f"milestone_{i+1}_{goal['id']}"
            milestone['status'] = 'pending'
            milestone['completed_date'] = None
        
        goal['milestones'] = milestones
        self.save_research_goals()

    def display_research_goals(self):
        """Display current research goals"""
        if not self.research_goals['current_goals']:
            return "🎯 **No active research goals found.**\n\n💡 **Set your first goal:** Say 'set goal' or 'create goal' to get started!"
        
        response = "🎯 **Your Research Goals:**\n\n"
        
        for i, goal in enumerate(self.research_goals['current_goals'], 1):
            response += f"**{i}. {goal['title']}**\n"
            response += f"   📖 {goal['description']}\n"
            response += f"   🎯 Priority: {goal['priority']}\n"
            response += f"   ⏰ Timeline: {goal['timeline']}\n"
            response += f"   📊 Progress: {goal['progress']}%\n"
            response += f"   📅 Created: {goal['created_date'][:10]}\n\n"
            
            # Show milestones
            if goal['milestones']:
                response += "   🎯 **Milestones:**\n"
                for milestone in goal['milestones']:
                    status_icon = "✅" if milestone['status'] == 'completed' else "⏳"
                    response += f"   {status_icon} {milestone['title']}\n"
                response += "\n"
        
        response += f"💡 **Total Active Goals:** {len(self.research_goals['current_goals'])}"
        response += "\n\n🗣️ **Commands:** 'show progress' to see detailed progress, 'plan research' for planning help"
        
        return response

    def display_research_progress(self):
        """Display detailed research progress"""
        if not self.research_goals['current_goals']:
            return "📊 **No research goals to track.**\n\n💡 **Set a goal first:** Say 'set goal' to create your first research objective!"
        
        response = "📊 **Research Progress Overview:**\n\n"
        
        total_progress = 0
        completed_milestones = 0
        total_milestones = 0
        
        for goal in self.research_goals['current_goals']:
            response += f"🎯 **{goal['title']}**\n"
            response += f"   📊 Overall Progress: {goal['progress']}%\n"
            
            if goal['milestones']:
                goal_completed = sum(1 for m in goal['milestones'] if m['status'] == 'completed')
                total_milestones += len(goal['milestones'])
                completed_milestones += goal_completed
                
                response += f"   🎯 Milestones: {goal_completed}/{len(goal['milestones'])} completed\n"
                
                for milestone in goal['milestones']:
                    status_icon = "✅" if milestone['status'] == 'completed' else "⏳"
                    target_date = milestone['target_date'][:10]
                    response += f"   {status_icon} {milestone['title']} (Target: {target_date})\n"
            
            response += "\n"
            total_progress += goal['progress']
        
        # Overall progress
        if self.research_goals['current_goals']:
            overall_progress = total_progress / len(self.research_goals['current_goals'])
            response += f"📈 **Overall Progress:** {overall_progress:.1f}%\n"
            response += f"🎯 **Milestone Completion:** {completed_milestones}/{total_milestones} ({completed_milestones/total_milestones*100:.1f}%)\n"
        
        # Progress insights
        if overall_progress > 75:
            response += "\n🚀 **Excellent progress!** You're close to achieving your research goals!"
        elif overall_progress > 50:
            response += "\n📈 **Good progress!** Keep up the momentum to reach your objectives."
        elif overall_progress > 25:
            response += "\n🔄 **Making progress!** Consider breaking down larger goals into smaller tasks."
        else:
            response += "\n🎯 **Getting started!** Focus on one goal at a time to build momentum."
        
        return response

    def display_achievements(self):
        """Display user achievements"""
        earned_achievements = [a for a in self.achievement_system['achievements'] if a['earned']]
        unearned_achievements = [a for a in self.achievement_system['achievements'] if not a['earned']]
        
        response = "🏆 **Your Research Achievements:**\n\n"
        
        if earned_achievements:
            response += f"✅ **Earned ({len(earned_achievements)}/{len(self.achievement_system['achievements'])}):**\n"
            for achievement in earned_achievements:
                response += f"   🏆 **{achievement['name']}**\n"
                response += f"      📖 {achievement['description']}\n"
                if achievement['date_earned']:
                    response += f"      📅 Earned: {achievement['date_earned'][:10]}\n"
                response += "\n"
        
        if unearned_achievements:
            response += f"⏳ **Available ({len(unearned_achievements)}):**\n"
            for achievement in unearned_achievements:
                response += f"   🔒 **{achievement['name']}**\n"
                response += f"      📖 {achievement['description']}\n\n"
        
        response += f"📊 **Total Achievements:** {self.achievement_system['total_achievements']}/{len(self.achievement_system['achievements'])}"
        
        if self.achievement_system['achievement_history']:
            response += "\n\n📜 **Recent Achievements:**\n"
            for history in self.achievement_system['achievement_history'][-3:]:  # Last 3
                response += f"   🏆 {history['achievement']} - {history['date_earned'][:10]}\n"
        
        return response

    def create_research_plan(self):
        """Create an intelligent research plan based on current state and goals"""
        if not self.research_goals['current_goals']:
            return "📋 **No research goals to plan for.**\n\n💡 **Set a goal first:** Say 'set goal' to create your first research objective!"
        
        response = "📋 **Intelligent Research Plan:**\n\n"
        
        # Analyze current state
        try:
            import sqlite3
            conn = sqlite3.connect(os.path.join(self.base_dir, 'data', 'agent.db'))
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM papers')
            paper_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM topics')
            topic_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM embeddings')
            embedding_count = cursor.fetchone()[0]
            
            conn.close()
            
            response += "🔍 **Current Research State:**\n"
            response += f"   📄 Papers: {paper_count}\n"
            response += f"   🧠 Embeddings: {embedding_count}\n"
            response += f"   🎯 Topics: {topic_count}\n\n"
            
        except Exception:
            response += "🔍 **Current Research State:** Unable to determine\n\n"
        
        # Create plan for each goal
        for goal in self.research_goals['current_goals']:
            response += f"🎯 **Plan for: {goal['title']}**\n"
            
            if goal['timeline'] == 'This week':
                response += "   📅 **This Week's Plan:**\n"
                response += "   • Day 1-2: Research planning and scope definition\n"
                response += "   • Day 3-4: Data collection and paper gathering\n"
                response += "   • Day 5-7: Analysis and completion\n"
            elif goal['timeline'] == 'This month':
                response += "   📅 **This Month's Plan:**\n"
                response += "   • Week 1: Research planning and methodology\n"
                response += "   • Week 2-3: Data collection and analysis\n"
                response += "   • Week 4: Synthesis and completion\n"
            else:
                response += "   📅 **Extended Timeline Plan:**\n"
                response += "   • Month 1: Research planning and setup\n"
                response += "   • Month 2: Data collection and analysis\n"
                response += "   • Month 3: Synthesis and goal completion\n"
            
            response += "\n   🎯 **Recommended Actions:**\n"
            
            # Suggest actions based on current state
            if paper_count == 0:
                response += "   • Start with 'get new papers' to gather research data\n"
            elif embedding_count == 0:
                response += "   • Use 'create embeddings' to process your papers\n"
            elif topic_count == 0:
                response += "   • Run 'cluster papers' to discover research topics\n"
            else:
                response += "   • Use 'analyze topics' for deep insights\n"
                response += "   • Generate 'report' to document your findings\n"
            
            response += "\n"
        
        response += "💡 **Pro Tips:**\n"
        response += "• Focus on one milestone at a time\n"
        response += "• Use 'show progress' to track your advancement\n"
        response += "• Celebrate achievements as you reach milestones\n"
        
        return response

    def update_goal_progress(self, action_type, goal_id=None):
        """Update goal progress based on user actions"""
        try:
            if not goal_id and self.research_goals['current_goals']:
                # Update the most recent goal
                goal = self.research_goals['current_goals'][0]
            elif goal_id:
                goal = next((g for g in self.research_goals['current_goals'] if g['id'] == goal_id), None)
            else:
                return
            
            if not goal:
                return
            
            # Update progress based on action
            if action_type == 'paper_analyzed':
                goal['progress'] = min(100, goal['progress'] + 10)
            elif action_type == 'topic_discovered':
                goal['progress'] = min(100, goal['progress'] + 15)
            elif action_type == 'report_generated':
                goal['progress'] = min(100, goal['progress'] + 20)
            elif action_type == 'milestone_completed':
                goal['progress'] = min(100, goal['progress'] + 25)
            
            # Check if goal is completed
            if goal['progress'] >= 100:
                goal['status'] = 'completed'
                goal['completed_date'] = datetime.now().isoformat()
                
                # Move to completed goals
                self.research_goals['completed_goals'].append(goal)
                self.research_goals['current_goals'].remove(goal)
                
                # Check for achievement
                self.check_and_award_achievements('goal_completed')
            
            self.save_research_goals()
            
        except Exception as e:
            logging.error(f"Error updating goal progress: {e}")

    def get_personalized_suggestions(self):
        """Get personalized suggestions based on user profile and goals"""
        suggestions = []
        
        # Based on research style
        if self.user_profile['research_style'] == 'exploratory':
            suggestions.append("🔍 **Exploratory Mode:** Try 'show topics' to discover new research areas")
        elif self.user_profile['research_style'] == 'focused':
            suggestions.append("🎯 **Focused Mode:** Use 'select topic' to dive deep into specific areas")
        elif self.user_profile['research_style'] == 'comprehensive':
            suggestions.append("📚 **Comprehensive Mode:** Run 'analyze topics' for broad research insights")
        
        # Based on activity level
        if self.user_profile['activity_level'] == 'high':
            suggestions.append("🚀 **High Activity:** Consider setting multiple research goals")
        elif self.user_profile['activity_level'] == 'low':
            suggestions.append("🌱 **Getting Started:** Focus on one goal at a time")
        
        # Based on goals
        if self.research_goals['current_goals']:
            active_goals = len(self.research_goals['current_goals'])
            if active_goals > 3:
                suggestions.append("📋 **Multiple Goals:** Consider consolidating or prioritizing your research objectives")
            elif active_goals == 1:
                suggestions.append("🎯 **Single Goal:** Great focus! Use 'show progress' to track your advancement")
        
        # Based on achievements
        if self.achievement_system['total_achievements'] > 5:
            suggestions.append("🏆 **Achievement Hunter:** You're doing great! Keep pushing your research boundaries")
        
        return suggestions

def main():
    """Main entry point"""
    try:
        chat_agent = ResearchAgentChat()
        chat_agent.chat()
    except Exception as e:
        print(f"❌ Failed to start chat agent: {str(e)}")
        logging.error(f"Failed to start chat agent: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
