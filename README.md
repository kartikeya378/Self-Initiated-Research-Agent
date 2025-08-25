# 🤖Self-Initiated Research Agent

# AI Research Agent

An **autonomous research assistant** that fetches, analyzes, and summarizes research papers (e.g., from arXiv).  
It detects **emerging topics, trends, and research gaps** while providing reports and interactive insights.

---

## 🚀 Features
- 📄 Fetch latest research papers automatically
- 🔍 Topic detection & trend analysis over time
- 📊 Generate summary reports (PDF/Markdown)
- 🤖 Self-initiated (runs daily via scheduler)
- 💬 Interactive chat interface for research exploration
- ⚡ Smart suggestions: "What's trending?", "Show topics", "Set research goals"

---

## 📸 Demo Screenshot

![AI Research Agent UI](https://github.com/kartikeya378/Self-Initiated-Research-Agent/blob/main/Screenshot%202025-08-25%20112208.png)

*(Screenshot of the running interface – shows chat, system status, and quick actions)*

---

## 🗂️ Project Structure
```
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
```

---

## Installation
1. Clone or extract the project.
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r web_requirements.txt   # For web UI
   ```

---

## Usage

### CLI Mode
Run the agent from the command line:
```bash
python main.py
```

### Web UI Mode
Launch the interactive web interface:
```bash
python web_ui.py
```
Or use the provided batch script:
```bash
launch_web_ui.bat
```

### Report Export
To generate/export reports:
```bash
python export_report.py
```

---

## Data Files
- **achievements.json** – stores recorded achievements.
- **agent.db** – local database for structured storage.
- **research_goals.json** – user-defined goals.
- **research_milestones.json** – progress tracking milestones.
- **user_profile.json** – stores researcher profile & preferences.

---

## Reports
Reports are auto-generated and stored under `reports/` in multiple formats:
- `.md` → raw markdown
- `.html` → web view
- `.pdf` → printable version

---

## Automating with Task Scheduler

To run the agent automatically at scheduled times, you can use **Windows Task Scheduler**:

1. Open **Task Scheduler** from the Start Menu.
2. Click **Create Basic Task**.
3. Give it a name (e.g., *Research Agent Auto Run*).
4. Choose the trigger (e.g., daily at 9 AM).
5. Select **Start a Program** and point it to:
   - Program: `python`
   - Arguments: `path\to\main.py`
   - Start in: project folder path
6. Save the task.

This ensures the agent runs automatically at your chosen intervals.

*(For Linux/macOS, you can use `cron` instead.)*

---

## Requirements
- Python 3.11.9 is the best version as of now
- Dependencies listed in `requirements.txt` and `web_requirements.txt`.

---

## License
This project is licensed under the MIT License - free to use and modify
