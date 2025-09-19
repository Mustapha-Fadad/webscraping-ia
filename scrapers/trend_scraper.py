"""
Trend Scraper Module
Analyzes trends and identifies emerging topics
"""

import asyncio
import aiohttp
import requests
import json
import time
import random
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
import re

from config.settings import settings
from utils.cleaning import cleaner
from utils.analytics import analytics  # Fixed import


class TrendScraper:
    """Scraper for trend analysis and emerging topics"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(settings.DEFAULT_HEADERS)
        self.analytics = analytics  # Use the global analytics instance

    def analyze_hashtag_trends(self, time_window_hours: int = 24) -> List[Dict[str, Any]]:
        """Analyze hashtag trends across platforms"""
        logger.info(f"Analyzing hashtag trends for the past {time_window_hours} hours...")

        # In a real implementation, this would collect data from multiple sources
        # For now, we'll simulate trend data based on common patterns

        trending_hashtags = self._simulate_hashtag_trends()

        # Calculate trend scores
        for hashtag in trending_hashtags:
            hashtag['trend_score'] = self._calculate_trend_score(hashtag)
            hashtag['velocity'] = self._calculate_velocity(hashtag)
            hashtag['prediction'] = self._predict_trend_direction(hashtag)

        # Sort by trend score
        trending_hashtags.sort(key=lambda x: x['trend_score'], reverse=True)

        logger.info(f"Analyzed {len(trending_hashtags)} hashtag trends")
        return trending_hashtags

    def _simulate_hashtag_trends(self) -> List[Dict[str, Any]]:
        """Simulate hashtag trend data"""

        tech_hashtags = [
            "#AI", "#MachineLearning", "#Python", "#JavaScript", "#React",
            "#Docker", "#Kubernetes", "#AWS", "#CloudComputing", "#DevOps",
            "#DataScience", "#Blockchain", "#IoT", "#Cybersecurity", "#OpenSource"
        ]

        general_hashtags = [
            "#Technology", "#Innovation", "#Digital", "#Future", "#Tech",
            "#Programming", "#Development", "#Software", "#Startup", "#Business"
        ]

        all_hashtags = tech_hashtags + general_hashtags
        trending_data = []

        for hashtag in all_hashtags:
            # Simulate trend metrics
            current_mentions = random.randint(100, 10000)
            previous_mentions = random.randint(50, current_mentions)

            trend_data = {
                'hashtag': hashtag,
                'current_mentions': current_mentions,
                'previous_mentions': previous_mentions,
                'growth_rate': ((current_mentions - previous_mentions) / max(previous_mentions, 1)) * 100,
                'platforms': random.sample(['twitter', 'instagram', 'linkedin', 'tiktok'], random.randint(2, 4)),
                'sentiment_score': random.uniform(-0.5, 0.8),
                'engagement_rate': random.uniform(0.02, 0.15),
                'geographic_spread': random.randint(5, 50),  # Number of countries
                'peak_time': (datetime.now() - timedelta(hours=random.randint(1, 24))).isoformat(),
                'related_keywords': self._generate_related_keywords(hashtag),
                'category': self._categorize_hashtag(hashtag),
                'analyzed_at': datetime.now().isoformat()
            }

            trending_data.append(trend_data)

        return trending_data

    def _generate_related_keywords(self, hashtag: str) -> List[str]:
        """Generate related keywords for a hashtag"""

        keyword_map = {
            '#AI': ['artificial intelligence', 'machine learning', 'deep learning', 'neural networks'],
            '#Python': ['programming', 'coding', 'development', 'data science'],
            '#JavaScript': ['web development', 'frontend', 'backend', 'nodejs'],
            '#React': ['frontend', 'components', 'jsx', 'hooks'],
            '#Docker': ['containers', 'devops', 'deployment', 'microservices'],
            '#DataScience': ['analytics', 'big data', 'statistics', 'visualization'],
            '#Blockchain': ['cryptocurrency', 'bitcoin', 'ethereum', 'smart contracts']
        }

        return keyword_map.get(hashtag, ['technology', 'innovation', 'digital'])

    def _categorize_hashtag(self, hashtag: str) -> str:
        """Categorize hashtag by topic"""

        categories = {
            'Programming': ['#Python', '#JavaScript', '#React', '#Programming'],
            'DevOps': ['#Docker', '#Kubernetes', '#AWS', '#DevOps'],
            'Data & AI': ['#AI', '#MachineLearning', '#DataScience'],
            'Security': ['#Cybersecurity', '#Security'],
            'Business': ['#Startup', '#Business', '#Innovation'],
            'General Tech': ['#Technology', '#Digital', '#Tech']
        }

        for category, tags in categories.items():
            if hashtag in tags:
                return category

        return 'Other'

    def _calculate_trend_score(self, hashtag_data: Dict[str, Any]) -> float:
        """Calculate a comprehensive trend score"""

        # Factors contributing to trend score
        growth_rate = hashtag_data.get('growth_rate', 0)
        current_mentions = hashtag_data.get('current_mentions', 0)
        engagement_rate = hashtag_data.get('engagement_rate', 0)
        sentiment_score = hashtag_data.get('sentiment_score', 0)
        platform_count = len(hashtag_data.get('platforms', []))
        geographic_spread = hashtag_data.get('geographic_spread', 0)

        # Weighted score calculation
        score = (
                (growth_rate * 0.3) +  # 30% weight for growth
                (min(current_mentions / 1000, 10) * 0.2) +  # 20% weight for volume (capped)
                (engagement_rate * 100 * 0.2) +  # 20% weight for engagement
                ((sentiment_score + 1) * 5 * 0.1) +  # 10% weight for sentiment
                (platform_count * 2 * 0.1) +  # 10% weight for platform diversity
                (min(geographic_spread / 10, 5) * 0.1)  # 10% weight for geographic spread
        )

        return max(0, min(score, 100))  # Normalize to 0-100

    def _calculate_velocity(self, hashtag_data: Dict[str, Any]) -> str:
        """Calculate trend velocity (how fast it's growing)"""

        growth_rate = hashtag_data.get('growth_rate', 0)

        if growth_rate > 200:
            return 'explosive'
        elif growth_rate > 50:
            return 'rapid'
        elif growth_rate > 10:
            return 'moderate'
        elif growth_rate > 0:
            return 'slow'
        else:
            return 'declining'

    def _predict_trend_direction(self, hashtag_data: Dict[str, Any]) -> str:
        """Predict future trend direction"""

        growth_rate = hashtag_data.get('growth_rate', 0)
        engagement_rate = hashtag_data.get('engagement_rate', 0)
        sentiment_score = hashtag_data.get('sentiment_score', 0)

        # Simple prediction logic
        if growth_rate > 30 and engagement_rate > 0.05 and sentiment_score > 0.2:
            return 'rising'
        elif growth_rate < -10 or sentiment_score < -0.3:
            return 'declining'
        elif growth_rate > 5:
            return 'stable_growth'
        else:
            return 'stable'

    def detect_emerging_topics(self, data_sources: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """Detect emerging topics from content analysis"""
        logger.info("Detecting emerging topics...")

        if not data_sources:
            # Simulate emerging topics
            return self._simulate_emerging_topics()

        # Analyze provided data for emerging topics
        all_content = []
        for source in data_sources:
            if isinstance(source, list):
                for item in source:
                    content = item.get('content', '') + ' ' + item.get('title', '')
                    if content.strip():
                        all_content.append(content)
            elif isinstance(source, dict):
                content = source.get('content', '') + ' ' + source.get('title', '')
                if content.strip():
                    all_content.append(content)

        # Extract topics using text analysis
        emerging_topics = self._extract_emerging_topics(all_content)

        logger.info(f"Detected {len(emerging_topics)} emerging topics")
        return emerging_topics

    def _simulate_emerging_topics(self) -> List[Dict[str, Any]]:
        """Simulate emerging topic data"""

        emerging_topics = [
            {
                'topic': 'Edge AI Computing',
                'description': 'AI processing at the edge of networks for real-time applications',
                'growth_indicators': ['increased mentions', 'new product launches', 'research papers'],
                'confidence_score': 0.85,
                'emergence_date': (datetime.now() - timedelta(days=7)).isoformat(),
                'related_companies': ['NVIDIA', 'Intel', 'Qualcomm'],
                'market_size_prediction': '$15B by 2025',
                'key_use_cases': ['autonomous vehicles', 'smart cities', 'IoT devices']
            },
            {
                'topic': 'Quantum Machine Learning',
                'description': 'Integration of quantum computing with machine learning algorithms',
                'growth_indicators': ['academic interest', 'patent filings', 'startup funding'],
                'confidence_score': 0.72,
                'emergence_date': (datetime.now() - timedelta(days=14)).isoformat(),
                'related_companies': ['IBM', 'Google', 'Rigetti'],
                'market_size_prediction': '$2.5B by 2030',
                'key_use_cases': ['optimization', 'cryptography', 'drug discovery']
            },
            {
                'topic': 'Sustainable Tech Solutions',
                'description': 'Technology solutions focused on environmental sustainability',
                'growth_indicators': ['ESG requirements', 'carbon neutral goals', 'green funding'],
                'confidence_score': 0.91,
                'emergence_date': (datetime.now() - timedelta(days=21)).isoformat(),
                'related_companies': ['Tesla', 'Microsoft', 'Amazon'],
                'market_size_prediction': '$50B by 2026',
                'key_use_cases': ['carbon tracking', 'renewable energy', 'circular economy']
            },
            {
                'topic': 'No-Code AI Platforms',
                'description': 'Democratizing AI through visual, code-free development platforms',
                'growth_indicators': ['user adoption', 'platform launches', 'enterprise adoption'],
                'confidence_score': 0.78,
                'emergence_date': (datetime.now() - timedelta(days=10)).isoformat(),
                'related_companies': ['Microsoft', 'Google', 'Salesforce'],
                'market_size_prediction': '$8B by 2025',
                'key_use_cases': ['business automation', 'data analysis', 'customer service']
            }
        ]

        # Add analysis metadata
        for topic in emerging_topics:
            topic.update({
                'analyzed_at': datetime.now().isoformat(),
                'data_sources': ['news', 'research', 'social media', 'patents'],
                'geographic_regions': ['North America', 'Europe', 'Asia-Pacific'],
                'trend_category': self._categorize_emerging_topic(topic['topic'])
            })

        return emerging_topics

    def _categorize_emerging_topic(self, topic: str) -> str:
        """Categorize emerging topic"""

        if any(keyword in topic.lower() for keyword in ['ai', 'machine learning', 'quantum']):
            return 'AI & Computing'
        elif any(keyword in topic.lower() for keyword in ['sustainable', 'green', 'carbon']):
            return 'Sustainability'
        elif any(keyword in topic.lower() for keyword in ['no-code', 'platform', 'automation']):
            return 'Developer Tools'
        else:
            return 'General Technology'

    def _extract_emerging_topics(self, content_list: List[str]) -> List[Dict[str, Any]]:
        """Extract emerging topics from actual content"""

        # Combine all content
        combined_text = ' '.join(content_list).lower()

        # Extract potential topics using keyword patterns
        # This is a simplified approach - in production, you'd use more sophisticated NLP

        tech_patterns = [
            r'\b(\w+)\s+ai\b', r'\b(\w+)\s+learning\b', r'\b(\w+)\s+computing\b',
            r'\b(\w+)\s+platform\b', r'\b(\w+)\s+technology\b', r'\b(\w+)\s+solution\b'
        ]

        potential_topics = []
        for pattern in tech_patterns:
            matches = re.findall(pattern, combined_text)
            potential_topics.extend(matches)

        # Count occurrences and filter
        topic_counts = Counter(potential_topics)

        emerging_topics = []
        for topic, count in topic_counts.most_common(10):
            if count >= 3 and len(topic) > 3:  # Minimum threshold and length
                emerging_topic = {
                    'topic': topic.title() + ' Technology',
                    'description': f'Emerging technology trend related to {topic}',
                    'mention_count': count,
                    'confidence_score': min(count / 20, 1.0),  # Normalize to 0-1
                    'emergence_date': datetime.now().isoformat(),
                    'data_sources': ['content_analysis'],
                    'analyzed_at': datetime.now().isoformat(),
                    'trend_category': self._categorize_emerging_topic(topic)
                }
                emerging_topics.append(emerging_topic)

        return emerging_topics

    def analyze_content_velocity(self, content_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the velocity of content creation and engagement"""
        logger.info("Analyzing content velocity...")

        # Group content by time periods
        hourly_counts = defaultdict(int)
        daily_counts = defaultdict(int)

        total_engagement = 0
        content_count = len(content_data)

        for item in content_data:
            timestamp = item.get('created_at') or item.get('timestamp') or item.get('date')
            if timestamp:
                try:
                    # Handle different timestamp formats
                    if isinstance(timestamp, str):
                        if 'Z' in timestamp:
                            timestamp = timestamp.replace('Z', '+00:00')
                        dt = datetime.fromisoformat(timestamp) if '+' in timestamp or '-' in timestamp[
                                                                                             -6:] else datetime.fromisoformat(
                            timestamp + '+00:00')
                    else:
                        dt = timestamp

                    hour_key = dt.strftime('%Y-%m-%d-%H')
                    day_key = dt.strftime('%Y-%m-%d')

                    hourly_counts[hour_key] += 1
                    daily_counts[day_key] += 1

                    # Calculate engagement
                    engagement = (item.get('likes', 0) +
                                  item.get('comments', 0) +
                                  item.get('shares', 0) +
                                  item.get('retweets', 0))
                    total_engagement += engagement

                except Exception as e:
                    logger.debug(f"Error parsing timestamp {timestamp}: {e}")
                    continue

        # Calculate velocity metrics
        if hourly_counts:
            hourly_values = list(hourly_counts.values())
            avg_hourly_rate = sum(hourly_values) / len(hourly_values)
            peak_hour = max(hourly_counts.items(), key=lambda x: x[1])
        else:
            avg_hourly_rate = 0
            peak_hour = ('N/A', 0)

        if daily_counts:
            daily_values = list(daily_counts.values())
            avg_daily_rate = sum(daily_values) / len(daily_values)
        else:
            avg_daily_rate = 0

        velocity_analysis = {
            'content_count': content_count,
            'total_engagement': total_engagement,
            'avg_engagement_per_post': total_engagement / max(content_count, 1),
            'avg_hourly_posting_rate': avg_hourly_rate,
            'avg_daily_posting_rate': avg_daily_rate,
            'peak_hour': {
                'time': peak_hour[0],
                'post_count': peak_hour[1]
            },
            'velocity_trend': self._determine_velocity_trend(daily_counts),
            'analyzed_at': datetime.now().isoformat(),
            'time_period_analyzed': {
                'start': min(hourly_counts.keys()) if hourly_counts else 'N/A',
                'end': max(hourly_counts.keys()) if hourly_counts else 'N/A'
            }
        }

        return velocity_analysis

    def _determine_velocity_trend(self, daily_counts: Dict[str, int]) -> str:
        """Determine if content velocity is increasing, decreasing, or stable"""

        if len(daily_counts) < 2:
            return 'insufficient_data'

        # Get recent days vs earlier days
        sorted_days = sorted(daily_counts.items())

        if len(sorted_days) >= 4:
            recent_avg = sum(count for _, count in sorted_days[-2:]) / 2
            earlier_avg = sum(count for _, count in sorted_days[:2]) / 2

            if recent_avg > earlier_avg * 1.2:
                return 'increasing'
            elif recent_avg < earlier_avg * 0.8:
                return 'decreasing'
            else:
                return 'stable'
        else:
            return 'stable'

    def run_full_analysis(self, content_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Run complete trend analysis"""
        logger.info("Starting full trend analysis...")

        analysis_results = {
            'timestamp': datetime.now().isoformat(),
            'hashtag_trends': [],
            'emerging_topics': [],
            'content_velocity': {},
            'trend_summary': {}
        }

        try:
            # 1. Analyze hashtag trends
            hashtag_trends = self.analyze_hashtag_trends()
            analysis_results['hashtag_trends'] = hashtag_trends

            # 2. Detect emerging topics
            emerging_topics = self.detect_emerging_topics([content_data] if content_data else None)
            analysis_results['emerging_topics'] = emerging_topics

            # 3. Analyze content velocity if data provided
            if content_data:
                velocity_analysis = self.analyze_content_velocity(content_data)
                analysis_results['content_velocity'] = velocity_analysis

            # 4. Create trend summary
            analysis_results['trend_summary'] = self._create_trend_summary(
                hashtag_trends, emerging_topics, analysis_results.get('content_velocity', {})
            )

            logger.info("Trend analysis completed successfully")

        except Exception as e:
            logger.error(f"Error in trend analysis: {e}")
            analysis_results['error'] = str(e)

        return analysis_results

    def _create_trend_summary(self, hashtag_trends: List[Dict],
                              emerging_topics: List[Dict],
                              velocity_data: Dict) -> Dict[str, Any]:
        """Create a summary of trend analysis"""

        # Top trending hashtags
        top_hashtags = [
            {
                'hashtag': trend['hashtag'],
                'score': trend['trend_score'],
                'velocity': trend['velocity']
            }
            for trend in hashtag_trends[:5]
        ]

        # High-confidence emerging topics
        high_confidence_topics = [
            {
                'topic': topic['topic'],
                'confidence': topic['confidence_score']
            }
            for topic in emerging_topics
            if topic.get('confidence_score', 0) > 0.7
        ]

        # Velocity insights
        velocity_insights = []
        if velocity_data:
            trend = velocity_data.get('velocity_trend', 'stable')
            avg_engagement = velocity_data.get('avg_engagement_per_post', 0)

            if trend == 'increasing':
                velocity_insights.append('Content velocity is increasing - good time for active engagement')
            elif trend == 'decreasing':
                velocity_insights.append('Content velocity is decreasing - may need to boost content creation')

            if avg_engagement > 50:
                velocity_insights.append('High engagement rate detected - content resonating well')
            elif avg_engagement < 10:
                velocity_insights.append('Low engagement rate - consider content strategy adjustment')

        summary = {
            'total_hashtags_analyzed': len(hashtag_trends),
            'total_emerging_topics': len(emerging_topics),
            'top_trending_hashtags': top_hashtags,
            'high_confidence_emerging_topics': high_confidence_topics,
            'velocity_insights': velocity_insights,
            'analysis_timestamp': datetime.now().isoformat(),
            'key_recommendations': self._generate_trend_recommendations(
                hashtag_trends, emerging_topics, velocity_data
            )
        }

        return summary

    def _generate_trend_recommendations(self, hashtag_trends: List[Dict],
                                        emerging_topics: List[Dict],
                                        velocity_data: Dict) -> List[str]:
        """Generate actionable recommendations based on trend analysis"""

        recommendations = []

        # Hashtag recommendations
        if hashtag_trends:
            top_hashtag = hashtag_trends[0]
            if top_hashtag['velocity'] == 'explosive':
                recommendations.append(
                    f"Capitalize on explosive trend: {top_hashtag['hashtag']} - create content immediately")
            elif top_hashtag['velocity'] == 'rapid':
                recommendations.append(
                    f"Join trending conversation: {top_hashtag['hashtag']} - high engagement potential")

        # Emerging topic recommendations
        high_confidence = [t for t in emerging_topics if t.get('confidence_score', 0) > 0.8]
        if high_confidence:
            topic = high_confidence[0]
            recommendations.append(
                f"Early mover advantage: Create content about '{topic['topic']}' - high confidence emerging trend")

        # Velocity recommendations
        if velocity_data:
            trend = velocity_data.get('velocity_trend', 'stable')
            if trend == 'increasing':
                recommendations.append("Content velocity increasing - scale up posting schedule to match momentum")
            elif trend == 'decreasing':
                recommendations.append(
                    "Content velocity decreasing - focus on quality over quantity, boost engagement tactics")

        # General recommendations
        recommendations.extend([
            "Monitor trend velocity daily for rapid response opportunities",
            "Cross-reference hashtag trends with emerging topics for content synergy",
            "Prepare content templates for quick deployment during trend spikes"
        ])

        return recommendations[:5]  # Return top 5 recommendations


# Create global trend scraper instance
trend_scraper = TrendScraper()


# Convenience functions
def analyze_hashtag_trends(time_window_hours: int = 24) -> List[Dict[str, Any]]:
    """Quick access to hashtag trend analysis"""
    return trend_scraper.analyze_hashtag_trends(time_window_hours)


def detect_emerging_topics(data_sources: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Quick access to emerging topic detection"""
    return trend_scraper.detect_emerging_topics(data_sources)


def run_trend_analysis(content_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Quick access to full trend analysis"""
    return trend_scraper.run_full_analysis(content_data)


def main():
    """Test the trend scraper independently"""
    scraper = TrendScraper()

    print("📈 Testing Trend Scraper")
    print("=" * 40)

    # Run full analysis
    results = scraper.run_full_analysis()

    print(f"\n📊 Analysis Results:")
    print(f"  Hashtag trends analyzed: {len(results.get('hashtag_trends', []))}")
    print(f"  Emerging topics detected: {len(results.get('emerging_topics', []))}")

    # Show top trending hashtags
    hashtag_trends = results.get('hashtag_trends', [])
    if hashtag_trends:
        print(f"\n🔥 Top 5 Trending Hashtags:")
        for i, trend in enumerate(hashtag_trends[:5], 1):
            print(f"  {i}. {trend['hashtag']} - Score: {trend['trend_score']:.1f}, Velocity: {trend['velocity']}")

    # Show emerging topics
    emerging_topics = results.get('emerging_topics', [])
    if emerging_topics:
        print(f"\n🚀 Emerging Topics:")
        for topic in emerging_topics[:3]:
            print(f"  • {topic['topic']} (Confidence: {topic['confidence_score']:.2f})")
            print(f"    {topic['description']}")

    # Show key recommendations
    summary = results.get('trend_summary', {})
    recommendations = summary.get('key_recommendations', [])
    if recommendations:
        print(f"\n💡 Key Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")


if __name__ == "__main__":
    main()