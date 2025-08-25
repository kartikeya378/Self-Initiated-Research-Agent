import argparse
import os
import sqlite3
from datetime import datetime, timedelta
import yaml
import pandas as pd
import numpy as np
from dateutil import parser as dateparser
from tqdm import tqdm
import nltk

# --- config & paths ---
BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, "data", "agent.db")
REPORTS_DIR = os.path.join(BASE, "reports")
os.makedirs(os.path.join(BASE, "data"), exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# --- DB schema ---
def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS papers (
        id TEXT PRIMARY KEY,
        title TEXT,
        authors TEXT,
        abstract TEXT,
        published TEXT,
        category TEXT
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS embeddings (
        paper_id TEXT PRIMARY KEY,
        vector BLOB,
        FOREIGN KEY(paper_id) REFERENCES papers(id)
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS topics (
        paper_id TEXT PRIMARY KEY,
        topic_id INTEGER,
        FOREIGN KEY(paper_id) REFERENCES papers(id)
    )""")
    con.commit()
    con.close()

# --- helpers ---
def read_config():
    with open(os.path.join(BASE, "config.yaml"), "r") as f:
        return yaml.safe_load(f)

def df_from_db(query, params=()):
    con = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(query, con, params=params)
    except Exception as e:
        print(f"DB error: {e}")
        df = pd.DataFrame()
    con.close()
    return df

def executemany(query, rows):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.executemany(query, rows)
    con.commit()
    con.close()

def execute(query, params=()):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(query, params)
    con.commit()
    con.close()

def ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    except Exception as e:
        print(f"nltk error: {e}")

# --- step 1: fetch from arXiv ---
def cmd_fetch(cfg):
    import arxiv  # local import
    init_db()
    days_back = int(cfg.get("days_back", 7))
    max_results = int(cfg.get("max_results", 100))
    query = cfg.get("query", "artificial intelligence")

    since = datetime.utcnow() - timedelta(days=days_back)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    rows = []
    for r in tqdm(search.results(), desc="Fetching arXiv"):
        pub = r.published.replace(tzinfo=None)
        if pub < since:
            continue
        rid = r.get_short_id()
        title = (r.title or "").strip().replace("\n", " ")
        authors = ", ".join([a.name for a in (r.authors or [])])
        abstract = (r.summary or "").strip().replace("\n", " ")
        published = pub.strftime("%Y-%m-%d")
        category = getattr(r, "primary_category", "") or ""
        rows.append((rid, title, authors, abstract, published, category))

    if rows:
        executemany("""INSERT OR IGNORE INTO papers(id,title,authors,abstract,published,category)
                       VALUES(?,?,?,?,?,?)""", rows)
    print(f"Inserted {len(rows)} new papers.")

# --- step 2: embeddings ---
def cmd_embed(cfg):
    from sentence_transformers import SentenceTransformer
    init_db()
    df = df_from_db("""SELECT p.id, p.abstract
                       FROM papers p
                       LEFT JOIN embeddings e ON p.id = e.paper_id
                       WHERE e.paper_id IS NULL AND LENGTH(p.abstract) > 0""")
    if df.empty:
        print("No new papers to embed.")
        return

    model = SentenceTransformer("all-MiniLM-L6-v2")
    vecs = model.encode(df["abstract"].tolist(), show_progress_bar=True, normalize_embeddings=True)
    rows = [(pid, np.asarray(v, dtype=np.float32).tobytes()) for pid, v in zip(df["id"], vecs)]
    executemany("INSERT OR REPLACE INTO embeddings(paper_id, vector) VALUES(?,?)", rows)
    print(f"Embedded {len(rows)} papers.")

# --- step 3: topics via k-means ---
def cmd_topics(cfg):
    from sklearn.cluster import MiniBatchKMeans
    init_db()
    df = df_from_db("""SELECT p.id, p.published, e.vector
                       FROM papers p JOIN embeddings e ON p.id = e.paper_id""")
    if df.empty:
        print("No embeddings available.")
        return

    # reconstruct vectors
    X = np.vstack([np.frombuffer(v, dtype=np.float32) for v in df["vector"]])
    n = X.shape[0]
    k_min = int(cfg.get("min_topics", 6))
    k_max = int(cfg.get("max_topics", 25))
    # heuristic: sqrt(n) clamped
    k = max(k_min, min(k_max, int(np.sqrt(n))))
    km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=512, n_init=10)
    labels = km.fit_predict(X)

    rows = list(zip(df["id"].tolist(), labels.tolist()))
    executemany("INSERT OR REPLACE INTO topics(paper_id, topic_id) VALUES(?,?)", rows)
    print(f"Assigned {len(rows)} papers to {k} topics.")

# --- step 4: trend detection ---
def cmd_trends(cfg):
    init_db()
    df = df_from_db("""
        SELECT p.id, p.published, t.topic_id
        FROM papers p JOIN topics t ON p.id = t.paper_id
    """)
    if df.empty:
        print("No topics available.")
        return

    df["published"] = pd.to_datetime(df["published"])
    today = pd.Timestamp.utcnow().normalize().replace(tzinfo=None)
    recent_days = int(cfg.get("lookback_recent_days", 7))
    baseline_days = int(cfg.get("lookback_baseline_days", 14))
    min_count = int(cfg.get("min_count", 3))
    ratio_thresh = float(cfg.get("ratio_thresh", 1.5))

    recent_start = today - pd.Timedelta(days=recent_days)
    base_start = recent_start - pd.Timedelta(days=baseline_days)

    # Enhanced trend analysis with multiple time windows
    windows = [
        ("recent", recent_start, today),
        ("baseline", base_start, recent_start),
        ("extended", base_start - pd.Timedelta(days=baseline_days), base_start)
    ]
    
    window_counts = {}
    for name, start, end in windows:
        window_data = df[(df["published"] >= start) & (df["published"] < end)]
        window_counts[name] = window_data.groupby("topic_id")["id"].count()

    # Calculate multiple trend metrics
    all_topics = sorted(set().union(*[set(counts.index) for counts in window_counts.values()]))
    trend_analysis = []
    
    for topic_id in all_topics:
        recent_count = int(window_counts["recent"].get(topic_id, 0))
        baseline_count = int(window_counts["baseline"].get(topic_id, 0))
        extended_count = int(window_counts["extended"].get(topic_id, 0))
        
        # Multiple trend indicators
        ratio = (recent_count + 1) / (baseline_count + 1)
        growth_rate = (recent_count - baseline_count) / (baseline_count + 1)
        acceleration = (baseline_count - extended_count) / (extended_count + 1)
        
        # Novelty score (higher for topics that appeared recently)
        novelty_score = 1.0 if extended_count == 0 else (recent_count / (extended_count + 1))
        
        # Momentum score (combination of growth and acceleration)
        momentum_score = growth_rate + acceleration
        
        if recent_count >= min_count and ratio >= ratio_thresh:
            trend_analysis.append({
                'topic_id': topic_id,
                'recent_count': recent_count,
                'baseline_count': baseline_count,
                'extended_count': extended_count,
                'ratio': ratio,
                'growth_rate': growth_rate,
                'acceleration': acceleration,
                'novelty_score': novelty_score,
                'momentum_score': momentum_score,
                'trend_strength': ratio * momentum_score * novelty_score
            })
    
    # Sort by trend strength (comprehensive ranking)
    trend_analysis.sort(key=lambda x: x['trend_strength'], reverse=True)
    
    if trend_analysis:
        print("Enhanced Trend Analysis:")
        print("=" * 80)
        print(f"{'Topic':<6} {'Recent':<8} {'Baseline':<10} {'Growth':<8} {'Momentum':<10} {'Novelty':<8} {'Strength':<10}")
        print("-" * 80)
        
        for trend in trend_analysis:
            print(f"{trend['topic_id']:<6} {trend['recent_count']:<8} {trend['baseline_count']:<10} "
                  f"{trend['growth_rate']:<8.2f} {trend['momentum_score']:<10.2f} "
                  f"{trend['novelty_score']:<8.2f} {trend['trend_strength']:<10.2f}")
        
        print("\nTrend Categories:")
        # Categorize trends
        emerging = [t for t in trend_analysis if t['novelty_score'] > 0.8 and t['growth_rate'] > 0.5]
        accelerating = [t for t in trend_analysis if t['acceleration'] > 0.3 and t['growth_rate'] > 0.2]
        stable = [t for t in trend_analysis if 0.8 <= t['ratio'] <= 1.2 and t['momentum_score'] < 0.1]
        
        if emerging:
            print(f"Emerging Topics: {', '.join([str(t['topic_id']) for t in emerging[:3]])}")
        if accelerating:
            print(f"Accelerating Topics: {', '.join([str(t['topic_id']) for t in accelerating[:3]])}")
        if stable:
            print(f"Stable Topics: {', '.join([str(t['topic_id']) for t in stable[:3]])}")
            
    else:
        print("No trending topics met thresholds.")

    # Store enhanced trend data for reporting
    execute("DROP TABLE IF EXISTS trend_cache")
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""CREATE TABLE trend_cache (
        topic_id INTEGER,
        recent_count INTEGER,
        baseline_count INTEGER,
        extended_count INTEGER,
        ratio REAL,
        growth_rate REAL,
        acceleration REAL,
        novelty_score REAL,
        momentum_score REAL,
        trend_strength REAL
    )""")
    
    if trend_analysis:
        rows = [(t['topic_id'], t['recent_count'], t['baseline_count'], t['extended_count'],
                t['ratio'], t['growth_rate'], t['acceleration'], t['novelty_score'], 
                t['momentum_score'], t['trend_strength']) for t in trend_analysis]
        cur.executemany("INSERT INTO trend_cache VALUES(?,?,?,?,?,?,?,?,?,?)", rows)
    
    con.commit()
    con.close()

# --- enhanced topic analysis and summarization ---
def analyze_topic_coherence(texts, max_sentences=3):
    """Analyze topic coherence and extract key insights"""
    ensure_nltk()
    import nltk, re
    from collections import Counter
    
    # Combine all texts for analysis
    combined_text = " ".join(texts)
    sentences = nltk.sent_tokenize(combined_text or "")
    
    if len(sentences) <= max_sentences:
        return " ".join(sentences), 1.0
    
    # Extract keywords and their frequencies
    words = re.findall(r"[A-Za-z]{4,}", combined_text.lower())
    word_freq = Counter(words)
    
    # Remove common stopwords (simplified)
    stopwords = {'this', 'that', 'with', 'have', 'from', 'they', 'will', 'been', 'were', 'said', 'each', 'which', 'their', 'time', 'there', 'could', 'would', 'made', 'over', 'very', 'after', 'also', 'into', 'through', 'during', 'before', 'above', 'below', 'between', 'among', 'within', 'without', 'against', 'toward', 'towards', 'upon', 'about', 'around', 'across', 'behind', 'beneath', 'beside', 'beyond', 'inside', 'outside', 'underneath'}
    word_freq = {k: v for k, v in word_freq.items() if k not in stopwords}
    
    # Score sentences based on keyword density and position
    sentence_scores = []
    for i, sentence in enumerate(sentences):
        sentence_words = re.findall(r"[A-Za-z]{4,}", sentence.lower())
        sentence_words = [w for w in sentence_words if w not in stopwords]
        
        if not sentence_words:
            continue
            
        # Keyword frequency score
        keyword_score = sum(word_freq.get(w, 0) for w in sentence_words) / len(sentence_words)
        
        # Position bonus (earlier sentences get higher scores)
        position_bonus = 1.0 - (i / len(sentences))
        
        # Length penalty (prefer medium-length sentences)
        length_penalty = 1.0 if 10 <= len(sentence_words) <= 25 else 0.7
        
        final_score = keyword_score * position_bonus * length_penalty
        sentence_scores.append((final_score, i, sentence))
    
    # Select top sentences
    top_sentences = sorted(sentence_scores, key=lambda x: x[0], reverse=True)[:max_sentences]
    top_sentences = sorted(top_sentences, key=lambda x: x[1])  # Maintain order
    
    summary = " ".join([s[2] for s in top_sentences])
    
    # Calculate coherence score
    coherence_score = sum(s[0] for s in top_sentences) / len(top_sentences) if top_sentences else 0
    
    return summary, coherence_score

def extract_topic_keywords(texts, top_n=10):
    """Extract most relevant keywords for a topic"""
    ensure_nltk()
    import nltk, re
    from collections import Counter
    
    combined_text = " ".join(texts)
    words = re.findall(r"[A-Za-z]{4,}", combined_text.lower())
    
    # Remove stopwords
    stopwords = {'this', 'that', 'with', 'have', 'from', 'they', 'will', 'been', 'were', 'said', 'each', 'which', 'their', 'time', 'there', 'could', 'would', 'made', 'over', 'very', 'after', 'also', 'into', 'through', 'during', 'before', 'above', 'below', 'between', 'among', 'within', 'without', 'against', 'toward', 'towards', 'upon', 'about', 'around', 'across', 'behind', 'beneath', 'beside', 'beyond', 'inside', 'outside', 'underneath'}
    word_freq = Counter([w for w in words if w not in stopwords])
    
    return [word for word, _ in word_freq.most_common(top_n)]

def summarize_text(text, max_sentences=2):
    """Legacy function - kept for compatibility"""
    summary, _ = analyze_topic_coherence([text], max_sentences)
    return summary

# --- step 5: reporting ---
def cmd_report(cfg):
    init_db()
    # read trending topics from cache (created in cmd_trends)
    cache = df_from_db("SELECT * FROM trend_cache ORDER BY trend_strength DESC")
    if cache.empty:
        print("No trend_cache found or it's empty. Run 'trends' first; generating baseline report instead.")
        # create a baseline: top populous topics
        df = df_from_db("""SELECT t.topic_id, COUNT(*) as c
                           FROM topics t GROUP BY t.topic_id ORDER BY c DESC""")
        if df.empty:
            print("No topics found. Did you run fetch/embed/topics?")
            return
        top_topics = df.head(int(cfg.get("top_n_topics", 3)))["topic_id"].tolist()
        # Create dummy trend data for baseline
        trend_data = []
        for tid in top_topics:
            trend_data.append({
                'topic_id': tid,
                'recent_count': df[df['topic_id'] == tid]['c'].iloc[0],
                'baseline_count': 0,
                'extended_count': 0,
                'ratio': 1.0,
                'growth_rate': 0.0,
                'acceleration': 0.0,
                'novelty_score': 1.0,
                'momentum_score': 0.0,
                'trend_strength': 1.0
            })
    else:
        top_topics = cache.head(int(cfg.get("top_n_topics", 3)))["topic_id"].tolist()
        trend_data = cache.to_dict('records')

    # fetch papers for each top topic
    n_per = int(cfg.get("papers_per_topic", 5))
    all_rows = []
    for tid in top_topics:
        q = """
            SELECT p.id, p.title, p.authors, p.abstract, p.published, p.category
            FROM papers p JOIN topics t ON p.id = t.paper_id
            WHERE t.topic_id = ?
            ORDER BY p.published DESC
            LIMIT ?
        """
        dfp = df_from_db(q, (int(tid), int(n_per)))
        dfp["topic_id"] = tid
        all_rows.append(dfp)
    if not all_rows:
        print("No papers for selected topics.")
        return
    dfall = pd.concat(all_rows, ignore_index=True)

    # build enhanced markdown report
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(REPORTS_DIR, f"report_{ts}.md")

    lines = []
    lines.append(f"# Research Digest — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")
    lines.append("This automatically generated digest highlights trending topics and recent papers with enhanced analysis.")
    lines.append("")

    # Enhanced trending analysis table
    if not cache.empty:
        lines.append("## Trending Topics Analysis")
        lines.append("")
        lines.append("| Topic | Recent | Baseline | Growth | Momentum | Novelty | Strength |")
        lines.append("|:--|:--|:--|:--|:--|:--|:--|")
        for _, row in cache.iterrows():
            lines.append(f"| {int(row['topic_id'])} | {int(row['recent_count'])} | {int(row['baseline_count'])} | {row['growth_rate']:.2f} | {row['momentum_score']:.2f} | {row['novelty_score']:.2f} | {row['trend_strength']:.2f} |")
        lines.append("")
        
        # Trend insights
        lines.append("### Trend Insights")
        lines.append("")
        if not cache.empty:
            top_trend = cache.iloc[0]
            lines.append(f"- **Most Dynamic Topic**: Topic {int(top_trend['topic_id'])} shows the highest trend strength ({top_trend['trend_strength']:.2f})")
            lines.append(f"- **Growth Leader**: Topic {int(top_trend['topic_id'])} with {top_trend['growth_rate']:.2f} growth rate")
            if top_trend['novelty_score'] > 0.8:
                lines.append(f"- **Emerging Field**: Topic {int(top_trend['topic_id'])} appears to be a newly emerging research area")
        lines.append("")

    # Enhanced topic details
    lines.append("## Topic Deep Dive")
    lines.append("")
    
    for i, tid in enumerate(top_topics):
        subset = dfall[dfall["topic_id"] == tid].copy()
        if subset.empty:
            continue
            
        # Get trend data for this topic
        topic_trend = next((t for t in trend_data if t['topic_id'] == tid), None)
        
        lines.append(f"### Topic {tid}")
        if topic_trend:
            lines.append("")
            lines.append(f"**Trend Metrics:**")
            lines.append(f"- Recent Papers: {topic_trend['recent_count']}")
            lines.append(f"- Growth Rate: {topic_trend['growth_rate']:.2f}")
            lines.append(f"- Momentum Score: {topic_trend['momentum_score']:.2f}")
            lines.append(f"- Novelty: {topic_trend['novelty_score']:.2f}")
            lines.append("")
        
        # Enhanced topic summary using new analysis
        titles_and_abstracts = (subset["title"].fillna("") + ". " + subset["abstract"].fillna("")).tolist()
        summary, coherence = analyze_topic_coherence(titles_and_abstracts, max_sentences=3)
        keywords = extract_topic_keywords(titles_and_abstracts, top_n=8)
        
        lines.append(f"**Synopsis:** {summary}")
        lines.append("")
        lines.append(f"**Key Terms:** {', '.join(keywords)}")
        lines.append(f"**Coherence Score:** {coherence:.2f}")
        lines.append("")
        
        lines.append("| Published | Title | Authors | Category |")
        lines.append("|:--|:--|:--|:--|")
        for _, r in subset.iterrows():
            lines.append(f"| {r['published']} | {r['title'].replace('|','-')} | {r['authors'].replace('|',', ')} | {r['category']} |")
        lines.append("")

    # Add methodology section
    lines.append("## Analysis Methodology")
    lines.append("")
    lines.append("This report uses advanced trend detection algorithms including:")
    lines.append("- **Multi-window Analysis**: Recent (7 days) vs Baseline (7 days) vs Extended (14 days)")
    lines.append("- **Growth Metrics**: Rate of change, acceleration, and momentum scoring")
    lines.append("- **Novelty Detection**: Identification of emerging research areas")
    lines.append("- **Topic Coherence**: Semantic analysis of topic consistency")
    lines.append("- **Keyword Extraction**: Automated identification of key research terms")
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Enhanced report written: {path}")

# --- step 6: advanced topic analysis ---
def cmd_analyze(cfg):
    """Advanced topic analysis with deep insights"""
    init_db()
    
    # Get all topics with their papers
    df = df_from_db("""
        SELECT t.topic_id, p.id, p.title, p.abstract, p.published, p.category
        FROM topics t JOIN papers p ON t.paper_id = p.id
        ORDER BY t.topic_id, p.published DESC
    """)
    
    if df.empty:
        print("No topics found. Did you run fetch/embed/topics?")
        return
    
    print("Advanced Topic Analysis")
    print("=" * 80)
    
    # Analyze each topic
    topic_insights = {}
    for topic_id in df['topic_id'].unique():
        topic_papers = df[df['topic_id'] == topic_id]
        
        # Extract texts for analysis
        texts = (topic_papers["title"].fillna("") + ". " + topic_papers["abstract"].fillna("")).tolist()
        
        # Get summary and coherence
        summary, coherence = analyze_topic_coherence(texts, max_sentences=2)
        keywords = extract_topic_keywords(texts, top_n=6)
        
        # Calculate topic statistics
        total_papers = len(topic_papers)
        date_range = topic_papers['published'].agg(['min', 'max'])
        categories = topic_papers['category'].value_counts()
        
        topic_insights[topic_id] = {
            'summary': summary,
            'coherence': coherence,
            'keywords': keywords,
            'total_papers': total_papers,
            'date_range': date_range,
            'categories': categories,
            'papers': topic_papers
        }
        
        print(f"\nTopic {topic_id}")
        print(f"   Papers: {total_papers}")
        print(f"   Coherence: {coherence:.2f}")
        print(f"   Date Range: {date_range['min']} to {date_range['max']}")
        print(f"   Top Categories: {', '.join(categories.head(3).index.tolist())}")
        print(f"   Key Terms: {', '.join(keywords)}")
        print(f"   Summary: {summary[:100]}...")
    
    # Find topic relationships (shared keywords)
    print("\nTopic Relationships")
    print("-" * 40)
    
    for i, (tid1, insights1) in enumerate(topic_insights.items()):
        for tid2, insights2 in list(topic_insights.items())[i+1:]:
            shared_keywords = set(insights1['keywords']) & set(insights2['keywords'])
            if len(shared_keywords) >= 2:
                print(f"Topics {tid1} & {tid2}: {len(shared_keywords)} shared terms ({', '.join(list(shared_keywords)[:3])})")
    
    # Save detailed analysis
    analysis_path = os.path.join(REPORTS_DIR, f"topic_analysis_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.md")
    
    lines = []
    lines.append(f"# Advanced Topic Analysis — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")
    
    for topic_id, insights in topic_insights.items():
        lines.append(f"## Topic {topic_id}")
        lines.append("")
        lines.append(f"**Statistics:**")
        lines.append(f"- Total Papers: {insights['total_papers']}")
        lines.append(f"- Coherence Score: {insights['coherence']:.2f}")
        lines.append(f"- Date Range: {insights['date_range']['min']} to {insights['date_range']['max']}")
        lines.append("")
        
        lines.append(f"**Keywords:** {', '.join(insights['keywords'])}")
        lines.append("")
        
        lines.append(f"**Summary:** {insights['summary']}")
        lines.append("")
        
        lines.append(f"**Top Categories:**")
        for cat, count in insights['categories'].head(5).items():
            lines.append(f"- {cat}: {count} papers")
        lines.append("")
        
        lines.append("**Recent Papers:**")
        for _, paper in insights['papers'].head(5).iterrows():
            lines.append(f"- **{paper['title']}** ({paper['published']})")
            if 'authors' in paper:
                lines.append(f"  - Authors: {paper['authors']}")
            if 'category' in paper:
                lines.append(f"  - Category: {paper['category']}")
            lines.append("")
    
    with open(analysis_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"\nDetailed analysis saved: {analysis_path}")

# --- orchestration ---
def cmd_run(cfg):
    cmd_fetch(cfg)
    cmd_embed(cfg)
    cmd_topics(cfg)
    cmd_trends(cfg)
    cmd_report(cfg)

def main():
    parser = argparse.ArgumentParser(description="Self-Initiated Research Agent (MVP) - Enhanced with Smart Analysis")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("fetch")
    sub.add_parser("embed")
    sub.add_parser("topics")
    sub.add_parser("trends")
    sub.add_parser("report")
    sub.add_parser("analyze")
    sub.add_parser("run")

    args = parser.parse_args()
    cfg = read_config()
    init_db()

    if args.cmd == "fetch":
        cmd_fetch(cfg)
    elif args.cmd == "embed":
        cmd_embed(cfg)
    elif args.cmd == "topics":
        cmd_topics(cfg)
    elif args.cmd == "trends":
        cmd_trends(cfg)
    elif args.cmd == "report":
        cmd_report(cfg)
    elif args.cmd == "analyze":
        cmd_analyze(cfg)
    elif args.cmd == "run":
        cmd_run(cfg)
    else:
        parser.print_help()
        
if __name__ == "__main__":
    main()



