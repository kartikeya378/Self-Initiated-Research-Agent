#!/usr/bin/env python3
"""
Web UI for Research Agent Chat
Modern, responsive web interface for the research analysis system
"""

import os
import sys
import json
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO, emit
import threading
import time

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the chat agent
from chat_agent import ResearchAgentChat

# Configure Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'research_agent_secret_key_2025'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global chat agent instance
chat_agent = None

def initialize_chat_agent():
    """Initialize the chat agent"""
    global chat_agent
    try:
        chat_agent = ResearchAgentChat()
        return True
    except Exception as e:
        logging.error(f"Failed to initialize chat agent: {e}")
        return False

@app.route('/')
def index():
    """Main chat interface"""
    return render_template('chat.html')

@app.route('/api/status')
def get_status():
    """Get current system status"""
    try:
        if not chat_agent:
            return jsonify({'error': 'Chat agent not initialized'}), 500
        
        # Get basic status
        status = {
            'timestamp': datetime.now().isoformat(),
            'user_profile': chat_agent.user_profile,
            'active_goals': len(chat_agent.research_goals['current_goals']),
            'total_achievements': chat_agent.achievement_system['total_achievements'],
            'total_achievements_available': len(chat_agent.achievement_system['achievements'])
        }
        
        # Get database status
        try:
            import sqlite3
            conn = sqlite3.connect(os.path.join(chat_agent.base_dir, 'data', 'agent.db'))
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM papers')
            status['papers_count'] = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM topics')
            status['topics_count'] = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM embeddings')
            status['embeddings_count'] = cursor.fetchone()[0]
            
            conn.close()
        except Exception:
            status['papers_count'] = 0
            status['topics_count'] = 0
            status['embeddings_count'] = 0
        
        return jsonify(status)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/goals', methods=['GET', 'POST'])
def handle_goals():
    """Handle research goals"""
    try:
        if not chat_agent:
            return jsonify({'error': 'Chat agent not initialized'}), 500
        
        if request.method == 'GET':
            return jsonify(chat_agent.research_goals)
        
        elif request.method == 'POST':
            data = request.json
            # This would integrate with the goal setting system
            return jsonify({'message': 'Goal creation endpoint - integrate with chat agent'})
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/achievements')
def get_achievements():
    """Get user achievements"""
    try:
        if not chat_agent:
            return jsonify({'error': 'Chat agent not initialized'}), 500
        
        return jsonify(chat_agent.achievement_system)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/command', methods=['POST'])
def execute_command():
    """Execute a research command"""
    try:
        if not chat_agent:
            return jsonify({'error': 'Chat agent not initialized'}), 500
        
        data = request.json
        command = data.get('command')
        
        if not command:
            return jsonify({'error': 'No command specified'}), 400
        
        # Execute the command
        result = chat_agent.execute_command(command)
        
        return jsonify({
            'success': True,
            'result': result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@socketio.on('send_message')
def handle_message(data):
    """Handle incoming chat messages"""
    try:
        if not chat_agent:
            emit('error', {'message': 'Chat agent not initialized'})
            return
        
        user_input = data.get('message', '').strip()
        if not user_input:
            return
        
        # Log the interaction
        chat_agent.log_interaction(user_input, "Processing...", None)
        
        # Understand intent
        intent = chat_agent.understand_intent(user_input)
        
        response = None
        command_executed = None
        
        if intent == 'exit':
            response = "Goodbye! Feel free to return anytime for more research insights."
            
        elif intent == 'help':
            response = chat_agent.get_help_message()
            command_executed = 'help'
            
        elif intent and intent.startswith('topic_details_'):
            topic_id = intent.split('_')[2]
            response = chat_agent.get_topic_details(int(topic_id))
            command_executed = f'topic_details_{topic_id}'
            
            # Update goal progress
            chat_agent.update_goal_progress('topic_discovered')
            
        elif intent and intent in chat_agent.commands:
            # Execute the command
            response = chat_agent.execute_command(chat_agent.commands[intent])
            command_executed = chat_agent.commands[intent]
            
            # Update goal progress based on action
            if intent == 'fetch':
                chat_agent.update_goal_progress('paper_analyzed')
            elif intent == 'topics':
                chat_agent.update_goal_progress('topic_discovered')
            elif intent == 'report':
                chat_agent.update_goal_progress('report_generated')
            
            # Check for achievements
            new_achievements = chat_agent.check_and_award_achievements('paper_analyzed')
            if new_achievements:
                achievement_msg = "🏆 **New Achievement Unlocked!**\n"
                for achievement in new_achievements:
                    achievement_msg += f"• {achievement['name']}: {achievement['description']}\n"
                response += f"\n\n{achievement_msg}"
            
        else:
            # Handle unclear requests
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
            command_executed = 'unclear'
        
        # Log the interaction
        if response:
            chat_agent.log_interaction(user_input, response, command_executed)
        
        # Send response back to client
        emit('message_response', {
            'message': response,
            'timestamp': datetime.now().isoformat(),
            'command_executed': command_executed
        })
        
        # Send proactive suggestions if appropriate
        if chat_agent.should_give_proactive_insight():
            proactive_insight = chat_agent.get_proactive_insight()
            emit('proactive_insight', {
                'message': proactive_insight,
                'timestamp': datetime.now().isoformat()
            })
        
    except Exception as e:
        error_msg = f"Sorry, something went wrong: {str(e)}"
        logging.error(f"Error handling message: {e}")
        emit('error', {'message': error_msg})

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    emit('connected', {'message': 'Connected to Research Agent'})
    
    # Send initial greeting
    if chat_agent:
        greeting = chat_agent.get_greeting()
        emit('greeting', {'message': greeting})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

def create_templates_directory():
    """Create templates directory if it doesn't exist"""
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    return templates_dir

def create_static_directory():
    """Create static directory if it doesn't exist"""
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    os.makedirs(static_dir, exist_ok=True)
    return static_dir

def main():
    """Main entry point"""
    try:
        # Initialize chat agent
        if not initialize_chat_agent():
            print("❌ Failed to initialize chat agent")
            return
        
        # Create necessary directories
        create_templates_directory()
        create_static_directory()
        
        print("🚀 Starting Research Agent Web UI...")
        print("📱 Open your browser and go to: http://localhost:5000")
        print("💡 The web interface provides a modern chat experience!")
        
        # Run the Flask app
        socketio.run(app, host='0.0.0.0', port=5000, debug=True)
        
    except Exception as e:
        print(f"❌ Failed to start web UI: {str(e)}")
        logging.error(f"Failed to start web UI: {str(e)}")

if __name__ == '__main__':
    main()
