"""
Analytics utilities for AI Community Manager
Provides statistical analysis, trend detection, and reporting capabilities
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta, timezone
from collections import Counter, defaultdict
import re
from textblob import TextBlob
from loguru import logger
import json
from dataclasses import dataclass
from enum import Enum


class TrendDirection(Enum):
    """Enum for trend directions"""
    RISING = "rising"
    FALLING = "falling"
    STABLE = "stable"
    VOLATILE = "volatile"


@dataclass
class TrendMetrics:
    """Data class for trend metrics"""
    topic: str
    mentions: int
    velocity: float  # Change rate per hour
    sentiment: float
    direction: TrendDirection
    confidence: float
    peak_time: Optional[datetime] = None
    related_keywords: List[str] = None


@dataclass
class AnalyticsReport:
    """Data class for analytics reports"""
    period_start: datetime
    period_end: datetime
    total_articles: int
    total_mentions: int
    top_trends: List[TrendMetrics]
    sentiment_distribution: Dict[str, float]
    source_breakdown: Dict[str, int]
    insights: List[str]
    generated_at: datetime


class Analytics:
    """Main analytics engine for community data analysis"""

    def __init__(self):
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
        }
        self.tech_keywords = {
            'ai', 'artificial intelligence', 'machine learning', 'deep learning',
            'python', 'javascript', 'react', 'nodejs', 'docker', 'kubernetes',
            'blockchain', 'cryptocurrency', 'cybersecurity', 'iot', 'cloud',
            'aws', 'azure', 'gcp', 'devops', 'api', 'microservices'
        }

    def normalize_datetime(self, dt_input: Any) -> Optional[datetime]:
        """Normalize datetime to timezone-naive UTC datetime"""
        if dt_input is None:
            return None

        try:
            # If it's already a datetime object
            if isinstance(dt_input, datetime):
                if dt_input.tzinfo is not None:
                    # Convert to UTC and make naive
                    return dt_input.astimezone(timezone.utc).replace(tzinfo=None)
                return dt_input

            # If it's a string, try to parse it
            if isinstance(dt_input, str):
                # Try pandas to_datetime which handles many formats
                parsed = pd.to_datetime(dt_input, errors='coerce')
                if pd.notna(parsed):
                    if parsed.tzinfo is not None:
                        return parsed.astimezone(timezone.utc).replace(tzinfo=None)
                    return parsed.replace(tzinfo=None)

        except Exception as e:
            logger.debug(f"Failed to normalize datetime {dt_input}: {e}")

        return None

    def calculate_basic_stats(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate basic statistics for the dataset"""
        if not data:
            return {}

        df = pd.DataFrame(data)

        # Normalize dates for comparison
        valid_dates = []
        for item in data:
            normalized_date = self.normalize_datetime(item.get('date'))
            if normalized_date:
                valid_dates.append(normalized_date)

        stats = {
            'total_articles': len(data),
            'date_range': {
                'start': min(valid_dates).isoformat() if valid_dates else 'Unknown',
                'end': max(valid_dates).isoformat() if valid_dates else 'Unknown'
            },
            'sources': dict(Counter(item.get('source', 'Unknown') for item in data)),
            'categories': dict(Counter(item.get('category', 'General') for item in data)),
            'avg_content_length': np.mean([len(str(item.get('content', ''))) for item in data]),
            'total_unique_sources': len(set(item.get('source') for item in data if item.get('source')))
        }

        # Word count statistics
        if 'word_count' in df.columns:
            stats['word_count_stats'] = {
                'mean': float(df['word_count'].mean()),
                'median': float(df['word_count'].median()),
                'std': float(df['word_count'].std()),
                'min': int(df['word_count'].min()),
                'max': int(df['word_count'].max())
            }

        # Sentiment statistics
        if 'sentiment_polarity' in df.columns:
            sentiment_data = df['sentiment_polarity'].dropna()
            if len(sentiment_data) > 0:
                stats['sentiment_stats'] = {
                    'avg_sentiment': float(sentiment_data.mean()),
                    'positive_ratio': float((sentiment_data > 0.1).mean()),
                    'negative_ratio': float((sentiment_data < -0.1).mean()),
                    'neutral_ratio': float((sentiment_data.abs() <= 0.1).mean())
                }

        logger.info(f"Calculated basic stats for {len(data)} articles")
        return stats

    def extract_keywords(self, text: str, min_length: int = 3, top_n: int = 20) -> List[Tuple[str, int]]:
        """Extract and rank keywords from text"""
        if not text or not isinstance(text, str):
            return []

        # Clean and tokenize
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = [word for word in text.split()
                 if len(word) >= min_length and word not in self.stop_words]

        # Count frequencies
        word_counts = Counter(words)

        # Also look for multi-word tech terms
        text_lower = text.lower()
        for keyword in self.tech_keywords:
            if keyword in text_lower:
                count = text_lower.count(keyword)
                if count > 0:
                    word_counts[keyword] += count

        return word_counts.most_common(top_n)

    def analyze_trends(self, data: List[Dict[str, Any]],
                       time_window_hours: int = 24,
                       min_mentions: int = 2) -> List[TrendMetrics]:  # Reduced min_mentions
        """Analyze trending topics and their velocity"""
        if not data:
            return []

        # Normalize dates first
        processed_data = []
        for item in data:
            normalized_date = self.normalize_datetime(item.get('date'))
            if normalized_date:
                item_copy = item.copy()
                item_copy['normalized_date'] = normalized_date
                processed_data.append(item_copy)
            else:
                # For items without valid dates, use current time
                item_copy = item.copy()
                item_copy['normalized_date'] = datetime.now()
                processed_data.append(item_copy)

        if not processed_data:
            logger.warning("No valid data for trend analysis")
            return []

        # Extract all keywords from all content
        all_keywords = Counter()
        keyword_articles = defaultdict(list)

        for item in processed_data:
            content = str(item.get('content', '')) + ' ' + str(item.get('title', ''))
            keywords = self.extract_keywords(content, top_n=50)

            for keyword, count in keywords:
                all_keywords[keyword] += count
                keyword_articles[keyword].append({
                    'date': item['normalized_date'],
                    'sentiment': item.get('sentiment_polarity', 0),
                    'source': item.get('source', 'Unknown')
                })

        # Analyze trends for top keywords
        trends = []
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(hours=time_window_hours)

        for keyword, total_count in all_keywords.most_common(30):  # Check more keywords
            if total_count < min_mentions:
                continue

            articles = keyword_articles[keyword]
            recent_articles = [a for a in articles if a['date'] >= cutoff_time]

            if len(recent_articles) == 0:
                continue

            # Calculate velocity (mentions per hour)
            recent_count = len(recent_articles)
            velocity = recent_count / max(1, time_window_hours)  # Avoid division by zero

            # Calculate sentiment
            sentiments = [a['sentiment'] for a in articles if a['sentiment'] is not None and a['sentiment'] != 0]
            avg_sentiment = float(np.mean(sentiments)) if sentiments else 0.0

            # Determine trend direction with more lenient thresholds
            recent_ratio = recent_count / max(1, len(articles))
            if recent_ratio >= 0.6:  # 60% of mentions are recent
                direction = TrendDirection.RISING
                confidence = min(0.9, recent_count / 5)  # Lower threshold for confidence
            elif recent_ratio <= 0.2:  # Only 20% are recent
                direction = TrendDirection.FALLING
                confidence = min(0.7, total_count / 10)
            else:
                direction = TrendDirection.STABLE
                confidence = 0.5

            # Find peak time
            peak_time = None
            if recent_articles:
                # Group by hour and find peak
                hourly_counts = Counter()
                for article in recent_articles:
                    hour = article['date'].replace(minute=0, second=0, microsecond=0)
                    hourly_counts[hour] += 1

                if hourly_counts:
                    peak_time = max(hourly_counts, key=hourly_counts.get)

            # Find related keywords (simplified)
            related_keywords = []
            keyword_words = set(keyword.lower().split())
            for other_keyword, _ in all_keywords.most_common(15):
                if other_keyword != keyword:
                    other_words = set(other_keyword.lower().split())
                    if keyword_words & other_words:  # Has common words
                        related_keywords.append(other_keyword)
                    if len(related_keywords) >= 3:  # Limit to top 3
                        break

            trend = TrendMetrics(
                topic=keyword,
                mentions=total_count,
                velocity=velocity,
                sentiment=avg_sentiment,
                direction=direction,
                confidence=confidence,
                peak_time=peak_time,
                related_keywords=related_keywords
            )
            trends.append(trend)

        # Sort by velocity and confidence
        trends.sort(key=lambda x: (x.velocity * x.confidence), reverse=True)

        logger.info(f"Identified {len(trends)} trending topics")
        return trends[:15]  # Return top 15 trends

    def analyze_sentiment_trends(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sentiment trends over time"""
        if not data:
            return {}

        # Process data with normalized dates
        processed_data = []
        for item in data:
            sentiment = item.get('sentiment_polarity')
            if sentiment is not None and sentiment != 0:
                normalized_date = self.normalize_datetime(item.get('date'))
                if normalized_date:
                    processed_data.append({
                        'date': normalized_date,
                        'sentiment': sentiment,
                        'source': item.get('source', 'Unknown')
                    })

        if not processed_data:
            return {'sentiment_distribution': {'positive': 0, 'negative': 0, 'neutral': 1}}

        df = pd.DataFrame(processed_data)

        # Group by hour and calculate average sentiment
        df['hour'] = df['date'].dt.floor('H')
        hourly_sentiment = df.groupby('hour')['sentiment'].agg(['mean', 'count', 'std']).reset_index()

        # Calculate sentiment distribution
        positive_count = (df['sentiment'] > 0.1).sum()
        negative_count = (df['sentiment'] < -0.1).sum()
        neutral_count = len(df) - positive_count - negative_count

        sentiment_analysis = {
            'overall_sentiment': float(df['sentiment'].mean()),
            'sentiment_distribution': {
                'positive': float(positive_count / len(df)) if len(df) > 0 else 0,
                'negative': float(negative_count / len(df)) if len(df) > 0 else 0,
                'neutral': float(neutral_count / len(df)) if len(df) > 0 else 1
            },
            'hourly_trends': [
                {
                    'hour': row['hour'].isoformat(),
                    'avg_sentiment': float(row['mean']),
                    'article_count': int(row['count']),
                    'sentiment_volatility': float(row['std']) if pd.notna(row['std']) else 0.0
                }
                for _, row in hourly_sentiment.iterrows()
            ],
            'sentiment_by_source': {}
        }

        # Sentiment by source
        if len(df) > 0:
            source_sentiment = df.groupby('source')['sentiment'].agg(['mean', 'count']).reset_index()
            sentiment_analysis['sentiment_by_source'] = {
                row['source']: {
                    'avg_sentiment': float(row['mean']),
                    'article_count': int(row['count'])
                }
                for _, row in source_sentiment.iterrows()
            }

        return sentiment_analysis

    def generate_insights(self, data: List[Dict[str, Any]], trends: List[TrendMetrics]) -> List[str]:
        """Generate actionable insights from the analysis"""
        insights = []

        if not data:
            return ["No data available for analysis."]

        # Basic stats insights
        total_articles = len(data)
        sources = set(item.get('source') for item in data if item.get('source'))

        insights.append(f"Analyzed {total_articles} articles from {len(sources)} different sources")

        # Trending topics insights
        if trends:
            top_trend = trends[0]
            insights.append(
                f"Top trending topic: '{top_trend.topic}' with {top_trend.mentions} mentions "
                f"and {top_trend.direction.value} trend"
            )

            # Rising trends
            rising_trends = [t for t in trends[:5] if t.direction == TrendDirection.RISING]
            if rising_trends:
                rising_topics = [t.topic for t in rising_trends[:3]]  # Limit to top 3
                insights.append(f"Rising trends: {', '.join(rising_topics)}")

            # High-velocity trends
            high_velocity = [t for t in trends[:10] if t.velocity > 0.5]  # Lowered threshold
            if high_velocity:
                insights.append(f"High-velocity topics generating significant discussion: {len(high_velocity)} topics")
        else:
            insights.append("No significant trends detected in current dataset")

        # Sentiment insights
        sentiments = [item.get('sentiment_polarity', 0) for item in data
                      if item.get('sentiment_polarity') is not None and item.get('sentiment_polarity') != 0]
        if sentiments:
            avg_sentiment = np.mean(sentiments)
            if avg_sentiment > 0.2:
                insights.append("Overall sentiment is positive - good engagement opportunity")
            elif avg_sentiment < -0.2:
                insights.append("Overall sentiment is negative - may need attention")
            else:
                insights.append("Sentiment is neutral - opportunity for engagement")

        # Source diversity insights
        source_counts = Counter(item.get('source') for item in data if item.get('source'))
        if source_counts:
            top_source = source_counts.most_common(1)[0]
            insights.append(f"Primary source: {top_source[0]} ({top_source[1]} articles)")

        # Timing insights
        valid_dates = [self.normalize_datetime(item.get('date')) for item in data]
        valid_dates = [d for d in valid_dates if d is not None]

        if valid_dates and len(valid_dates) > 5:
            hours = [d.hour for d in valid_dates]
            most_common_hour = Counter(hours).most_common(1)[0][0]
            insights.append(f"Peak activity around {most_common_hour}:00 - optimal time for engagement")

        # Content insights
        word_counts = [item.get('word_count', 0) for item in data if item.get('word_count')]
        if word_counts:
            avg_words = np.mean(word_counts)
            if avg_words > 100:
                insights.append("Long-form content dominates - detailed discussions preferred")
            else:
                insights.append("Short-form content prevalent - quick updates and news")

        return insights

    def create_analytics_report(self, data: List[Dict[str, Any]],
                                period_hours: int = 24) -> AnalyticsReport:
        """Create a comprehensive analytics report"""
        logger.info(f"Generating analytics report for {len(data)} articles")

        # Calculate time period
        current_time = datetime.now()
        period_start = current_time - timedelta(hours=period_hours)

        # Filter data to time period if dates available
        filtered_data = []
        for item in data:
            normalized_date = self.normalize_datetime(item.get('date'))
            if normalized_date and normalized_date >= period_start:
                filtered_data.append(item)
            elif not item.get('date'):  # Include items without dates
                filtered_data.append(item)

        if not filtered_data:
            filtered_data = data  # Use all data if no valid dates

        # Run analysis
        trends = self.analyze_trends(filtered_data, time_window_hours=period_hours)
        sentiment_analysis = self.analyze_sentiment_trends(filtered_data)
        basic_stats = self.calculate_basic_stats(filtered_data)
        insights = self.generate_insights(filtered_data, trends)

        # Calculate total mentions
        total_mentions = sum(t.mentions for t in trends) if trends else 0

        # Create report
        report = AnalyticsReport(
            period_start=period_start,
            period_end=current_time,
            total_articles=len(filtered_data),
            total_mentions=total_mentions,
            top_trends=trends[:10],
            sentiment_distribution=sentiment_analysis.get('sentiment_distribution', {}),
            source_breakdown=basic_stats.get('sources', {}),
            insights=insights,
            generated_at=current_time
        )

        logger.info("Analytics report generated successfully")
        return report

    def export_report_json(self, report: AnalyticsReport) -> str:
        """Export analytics report as JSON string"""
        report_dict = {
            'period_start': report.period_start.isoformat(),
            'period_end': report.period_end.isoformat(),
            'total_articles': report.total_articles,
            'total_mentions': report.total_mentions,
            'top_trends': [
                {
                    'topic': trend.topic,
                    'mentions': trend.mentions,
                    'velocity': trend.velocity,
                    'sentiment': trend.sentiment,
                    'direction': trend.direction.value,
                    'confidence': trend.confidence,
                    'peak_time': trend.peak_time.isoformat() if trend.peak_time else None,
                    'related_keywords': trend.related_keywords or []
                }
                for trend in report.top_trends
            ],
            'sentiment_distribution': report.sentiment_distribution,
            'source_breakdown': report.source_breakdown,
            'insights': report.insights,
            'generated_at': report.generated_at.isoformat()
        }

        return json.dumps(report_dict, indent=2, ensure_ascii=False)

    def calculate_engagement_score(self, data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate engagement scores for different topics and sources"""
        if not data:
            return {}

        engagement_scores = {}

        # Group by source
        source_groups = defaultdict(list)
        for item in data:
            source = item.get('source', 'Unknown')
            source_groups[source].append(item)

        for source, articles in source_groups.items():
            # Calculate metrics
            total_articles = len(articles)
            word_counts = [item.get('word_count', 0) for item in articles]
            avg_word_count = np.mean(word_counts) if word_counts else 0

            sentiments = [item.get('sentiment_polarity', 0) for item in articles
                          if item.get('sentiment_polarity') is not None]
            avg_sentiment = np.mean(sentiments) if sentiments else 0

            # Engagement score based on volume, length, and sentiment
            volume_score = min(1.0, total_articles / 10)  # Normalize to 0-1
            length_score = min(1.0, avg_word_count / 500)  # Normalize to 0-1
            sentiment_score = (avg_sentiment + 1) / 2  # Convert -1,1 to 0,1

            engagement_score = (volume_score * 0.4 + length_score * 0.3 + sentiment_score * 0.3)
            engagement_scores[source] = float(engagement_score)

        return engagement_scores


# Create global analytics instance
analytics = Analytics()


# Convenience functions
def analyze_data(data: List[Dict[str, Any]]) -> AnalyticsReport:
    """Quick access to full data analysis"""
    return analytics.create_analytics_report(data)


def get_trends(data: List[Dict[str, Any]], min_mentions: int = 2) -> List[TrendMetrics]:
    """Quick access to trend analysis"""
    return analytics.analyze_trends(data, min_mentions=min_mentions)


def get_basic_stats(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Quick access to basic statistics"""
    return analytics.calculate_basic_stats(data)


def main():
    """Test the analytics module independently"""
    print("Testing Analytics Module")
    print("=" * 40)

    # Sample data for testing
    sample_data = [
        {
            'title': 'AI Revolution in Tech',
            'content': 'Artificial intelligence is transforming the technology industry with machine learning advances',
            'date': datetime.now().isoformat(),
            'source': 'TechNews',
            'sentiment_polarity': 0.8,
            'word_count': 50
        },
        {
            'title': 'Python Programming Growth',
            'content': 'Python continues to grow as the most popular programming language for data science',
            'date': (datetime.now() - timedelta(hours=2)).isoformat(),
            'source': 'DevNews',
            'sentiment_polarity': 0.6,
            'word_count': 45
        }
    ]

    # Run analysis
    report = analytics.create_analytics_report(sample_data)
    print(f"Generated report with {len(report.top_trends)} trends")
    print(f"Top trend: {report.top_trends[0].topic if report.top_trends else 'None'}")


if __name__ == "__main__":
    main()