"""
Data storage and retrieval utilities
"""
import json
import sqlite3
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import aiofiles
import asyncio
from loguru import logger


class DataStorage:
    def __init__(self, db_path: str = "data/community_manager.db"):
        self.db_path = db_path
        self.ensure_directories()
        self.init_database()

    def ensure_directories(self):
        """Create necessary directories"""
        directories = ['data/raw', 'data/processed', 'data/exports', 'logs']
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def init_database(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Raw data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS raw_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                type TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                processed BOOLEAN DEFAULT FALSE
            )
        ''')

        # Analysis results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id TEXT UNIQUE NOT NULL,
                results TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Recommendations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id TEXT NOT NULL,
                recommendation_type TEXT NOT NULL,
                content TEXT NOT NULL,
                priority INTEGER DEFAULT 0,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")

    async def save_raw_data(self, source: str, data_type: str, data: List[Dict[str, Any]]):
        """Save raw scraped data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for item in data:
            cursor.execute('''
                INSERT INTO raw_data (source, type, content, metadata)
                VALUES (?, ?, ?, ?)
            ''', (
                source,
                data_type,
                item.get('content', ''),
                json.dumps(item)
            ))

        conn.commit()
        conn.close()

        # Also save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/raw/{source}_{data_type}_{timestamp}.json"

        async with aiofiles.open(filename, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(data, indent=2, ensure_ascii=False))

        logger.info(f"Saved {len(data)} items from {source} ({data_type})")

    async def save_analysis_results(self, results: Dict[str, Any]):
        """Save analysis results"""
        analysis_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO analysis_results (analysis_id, results)
            VALUES (?, ?)
        ''', (analysis_id, json.dumps(results)))

        # Save recommendations separately
        recommendations = results.get('recommendations', [])
        for rec in recommendations:
            cursor.execute('''
                INSERT INTO recommendations (analysis_id, recommendation_type, content, priority)
                VALUES (?, ?, ?, ?)
            ''', (
                analysis_id,
                rec.get('type', 'general'),
                rec.get('content', ''),
                rec.get('priority', 0)
            ))

        conn.commit()
        conn.close()

        # Export to JSON
        filename = f"data/exports/{analysis_id}.json"
        async with aiofiles.open(filename, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(results, indent=2, ensure_ascii=False))

        logger.info(f"Analysis results saved: {analysis_id}")
        return analysis_id

    def load_raw_data(self, source: Optional[str] = None, limit: int = 1000) -> pd.DataFrame:
        """Load raw data from database"""
        conn = sqlite3.connect(self.db_path)

        if source:
            query = "SELECT * FROM raw_data WHERE source = ? ORDER BY timestamp DESC LIMIT ?"
            df = pd.read_sql_query(query, conn, params=(source, limit))
        else:
            query = "SELECT * FROM raw_data ORDER BY timestamp DESC LIMIT ?"
            df = pd.read_sql_query(query, conn, params=(limit,))

        conn.close()
        return df

    def load_analysis_results(self, analysis_id: Optional[str] = None) -> Dict[str, Any]:
        """Load analysis results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if analysis_id:
            cursor.execute("SELECT results FROM analysis_results WHERE analysis_id = ?", (analysis_id,))
        else:
            cursor.execute("SELECT results FROM analysis_results ORDER BY timestamp DESC LIMIT 1")

        result = cursor.fetchone()
        conn.close()

        if result:
            return json.loads(result[0])
        return {}

    async def export_to_csv(self, data: pd.DataFrame, filename: str):
        """Export data to CSV"""
        filepath = f"data/exports/{filename}"
        data.to_csv(filepath, index=False)
        logger.info(f"Data exported to {filepath}")

    def get_data_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        stats = {}

        # Raw data stats
        cursor.execute("SELECT COUNT(*) FROM raw_data")
        stats['total_raw_records'] = cursor.fetchone()[0]

        cursor.execute("SELECT source, COUNT(*) FROM raw_data GROUP BY source")
        stats['records_by_source'] = dict(cursor.fetchall())

        # Analysis stats
        cursor.execute("SELECT COUNT(*) FROM analysis_results")
        stats['total_analyses'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM recommendations")
        stats['total_recommendations'] = cursor.fetchone()[0]

        conn.close()
        return stats


def generate_insights(analysis_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate actionable insights from analysis data"""
    insights = []

    # Sentiment insights
    sentiment_data = analysis_data.get('sentiment_analysis', {})
    if sentiment_data:
        positive_pct = sentiment_data.get('sentiment_percentages', {}).get('positive', 0)
        negative_pct = sentiment_data.get('sentiment_percentages', {}).get('negative', 0)

        if positive_pct > 70:
            insights.append({
                'type': 'positive_sentiment',
                'title': 'Strong Positive Sentiment',
                'description': f'Community shows {positive_pct:.1f}% positive sentiment. Great opportunity for engagement.',
                'priority': 8,
                'action': 'Amplify positive content and engage with enthusiastic users'
            })
        elif negative_pct > 40:
            insights.append({
                'type': 'negative_sentiment',
                'title': 'Negative Sentiment Alert',
                'description': f'High negative sentiment detected ({negative_pct:.1f}%). Requires attention.',
                'priority': 9,
                'action': 'Address concerns, provide support, and improve communication'
            })

    # Trending topics insights
    trending_topics = analysis_data.get('trending_topics', [])
    if trending_topics:
        top_topic = trending_topics[0]
        insights.append({
            'type': 'trending_topic',
            'title': f'Trending: "{top_topic[0]}"',
            'description': f'Most discussed topic with {top_topic[1]} mentions.',
            'priority': 7,
            'action': f'Create content around "{top_topic[0]}" to ride the trend'
        })

    # Activity timing insights
    peak_times = analysis_data.get('peak_times', {})
    if peak_times:
        peak_hours = peak_times.get('peak_hours', [])
        if peak_hours:
            top_hour = peak_hours[0][0]
            insights.append({
                'type': 'optimal_timing',
                'title': f'Peak Activity at {top_hour}:00',
                'description': f'Highest engagement occurs around {top_hour}:00.',
                'priority': 6,
                'action': f'Schedule important posts around {top_hour}:00 for maximum reach'
            })

    # Sort by priority descending
    return sorted(insights, key=lambda x: x['priority'], reverse=True)
