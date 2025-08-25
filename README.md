Self-Initiated Research Agent
Overview

The Self-Initiated Research Agent is an autonomous Python-based assistant that helps manage, track, and generate reports for research activities. It is designed to support long-term projects by storing goals, milestones, achievements, and user profiles, while also offering a reporting system in multiple formats (Markdown, HTML, PDF). The agent can run in both command-line mode and via a web UI for interactive usage.

Features

Autonomous agent for research assistance.

Data storage for achievements, research goals, milestones, and user profiles.

Report generation in Markdown, HTML, and PDF formats.

Logs to track agent activity.

Web UI for ease of interaction.

Configurable settings via config.yaml.

Task Scheduler / Automation support for periodic execution.

Project Structure
self_initiated_research_agent/
    chat_agent.log
    self_initiated_research_agent/
        agent.log
        chat.bat
        chat_agent.log
        chat_agent.py
        config.yaml
        export_report.py
        launch_web_ui.bat
        main.py
        README.md
        requirements.txt
        web_requirements.txt
        web_ui.py
        data/
            achievements.json
            agent.db
            research_goals.json
            research_milestones.json
            user_profile.json
        reports/
            report_*.md / .html / .pdf

Installation

Clone or extract the project.

Install Python dependencies:

pip install -r requirements.txt
pip install -r web_requirements.txt   # For web UI

Usage
CLI Mode

Run the agent from the command line:

python main.py

Web UI Mode

Launch the interactive web interface:

python web_ui.py


Or use the provided batch script:

launch_web_ui.bat

Report Export

To generate/export reports:

python export_report.py

Data Files

achievements.json – stores recorded achievements.

agent.db – local database for structured storage.

research_goals.json – user-defined goals.

research_milestones.json – progress tracking milestones.

user_profile.json – stores researcher profile & preferences.

Reports

Reports are auto-generated and stored under reports/ in multiple formats:

.md → raw markdown

.html → web view

.pdf → printable version

Automating with Task Scheduler

To run the agent automatically at scheduled times, you can use Windows Task Scheduler:

Open Task Scheduler from the Start Menu.

Click Create Basic Task.

Give it a name (e.g., Research Agent Auto Run).

Choose the trigger (e.g., daily at 9 AM).

Select Start a Program and point it to:

Program: python

Arguments: path\\to\\main.py

Start in: project folder path

Save the task.

This ensures the agent runs automatically at your chosen intervals.

(For Linux/macOS, you can use cron instead.)

Requirements

Python 3.8+

Dependencies listed in requirements.txt and web_requirements.txt

License

This project is for educational and research purposes only.
