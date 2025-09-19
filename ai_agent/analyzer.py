"""
Intelligent analysis module for community data
"""
import asyncio
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta
import numpy as np
from textblob import TextBlob
from collections import defaultdict, Counter
import re
from loguru import logger


class CommunityAnalyzer:
    def __init__(self):
        self.sentiment_cache = {}
        self.topic_models = {}

    async def perform_advanced_analysis(self, collected_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Perform advanced analysis on collected data"""
        logger.info("Starting advanced analysis...")

        analysis = {
            'community_health': {},
            'content_performance': {},
            'user_behavior': {},
            'competitive_analysis': {},
            'anomaly_detection': {},
            'predictive_insights': {}
        }

        try:
            # Community Health Analysis
            analysis['community_health'] = await self._analyze_community_health(collected_data)

            # Content Performance Analysis
            analysis['content_performance'] = await self._analyze_content_performance(collected_data)

            # User Behavior Analysis
            analysis['user_behavior'] = await self._analyze_user_behavior(collected_data)

            # Competitive Analysis
            analysis['competitive_analysis'] = await self._analyze_competition(collected_data)

            # Anomaly Detection
            analysis['anomaly_detection'] = await self._detect_anomalies(collected_data)

            # Predictive Insights
            analysis['predictive_insights'] = await self._generate_predictive_insights(collected_data)

            logger.info("Advanced analysis completed")

        except Exception as e:
            logger.error(f"Error in advanced analysis: {e}")

        return analysis

    async def _analyze_community_health(self, data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Analyze overall community health metrics"""
        health_metrics = {
            'engagement_rate': 0.0,
            'sentiment_score': 0.0,
            'activity_level': 'low',
            'growth_indicators': {},
            'health_score': 0.0
        }

        try:
            all_posts = []
            for source_data in data.values():
                all_posts.extend(source_data)

            if not all_posts:
                return health_metrics

            # Calculate engagement rate
            total_engagement = 0
            total_posts = len(all_posts)

            for post in all_posts:
                likes = post.get('likes', 0)
                comments = post.get('comments', 0)
                shares = post.get('shares', 0)
                total_engagement += likes + comments + shares

            avg_engagement = total_engagement / total_posts if total_posts > 0 else 0
            health_metrics['engagement_rate'] = avg_engagement

            # Calculate sentiment score
            sentiments = []
            for post in all_posts:
                content = post.get('content', '')
                if content:
                    blob = TextBlob(content)
                    sentiments.append(blob.sentiment.polarity)

            avg_sentiment = np.mean(sentiments) if sentiments else 0.0
            health_metrics['sentiment_score'] = avg_sentiment

            # Activity level assessment
            if total_posts > 100:
                health_metrics['activity_level'] = 'high'
            elif total_posts > 50:
                health_metrics['activity_level'] = 'medium'
            else:
                health_metrics['activity_level'] = 'low'

            # Overall health score (0-100)
            engagement_component = min(avg_engagement / 10, 40)  # Max 40 points
            sentiment_component = ((avg_sentiment + 1) / 2) * 30  # Max 30 points
            activity_component = min(total_posts / 10, 30)  # Max 30 points

            health_metrics['health_score'] = engagement_component + sentiment_component + activity_component

        except Exception as e:
            logger.error(f"Error in community health analysis: {e}")

        return health_metrics

    async def _analyze_content_performance(self, data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Analyze content performance patterns"""
        performance_metrics = {
            'top_performing_content': [],
            'content_types': {},
            'optimal_length': {},
            'hashtag_performance': {},
            'media_impact': {}
        }

        try:
            all_content = []
            for source_data in data.values():
                all_content.extend(source_data)

            # Top performing content
            scored_content = []
            for item in all_content:
                score = (item.get('likes', 0) +
                         item.get('comments', 0) * 2 +
                         item.get('shares', 0) * 3)

                scored_content.append({
                    'content': item.get('content', '')[:100] + '...',
                    'score': score,
                    'source': item.get('source', 'unknown'),
                    'engagement': {
                        'likes': item.get('likes', 0),
                        'comments': item.get('comments', 0),
                        'shares': item.get('shares', 0)
                    }
                })

            performance_metrics['top_performing_content'] = sorted(
                scored_content, key=lambda x: x['score'], reverse=True
            )[:10]

            # Content length analysis
            lengths_and_scores = []
            for item in all_content:
                content_length = len(item.get('content', ''))
                score = (item.get('likes', 0) + item.get('comments', 0) + item.get('shares', 0))
                lengths_and_scores.append((content_length, score))

            # Find optimal length ranges
            length_ranges = {
                'short (0-100)': [],
                'medium (101-300)': [],
                'long (301+)': []
            }

            for length, score in lengths_and_scores:
                if length <= 100:
                    length_ranges['short (0-100)'].append(score)
                elif length <= 300:
                    length_ranges['medium (101-300)'].append(score)
                else:
                    length_ranges['long (301+)'].append(score)

            for range_name, scores in length_ranges.items():
                avg_score = np.mean(scores) if scores else 0
                performance_metrics['optimal_length'][range_name] = {
                    'average_score': avg_score,
                    'count': len(scores)
                }

            # Hashtag performance
            hashtag_scores = defaultdict(list)
            for item in all_content:
                hashtags = item.get('hashtags', [])
                score = (item.get('likes', 0) + item.get('comments', 0) + item.get('shares', 0))

                for hashtag in hashtags:
                    hashtag_scores[hashtag].append(score)

            for hashtag, scores in hashtag_scores.items():
                performance_metrics['hashtag_performance'][hashtag] = {
                    'average_score': np.mean(scores),
                    'usage_count': len(scores),
                    'total_score': sum(scores)
                }

        except Exception as e:
            logger.error(f"Error in content performance analysis: {e}")

        return performance_metrics

    async def _analyze_user_behavior(self, data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Analyze user behavior patterns"""
        behavior_metrics = {
            'active_users': {},
            'engagement_patterns': {},
            'user_segments': {},
            'interaction_types': {}
        }

        try:
            user_activities = defaultdict(list)
            all_interactions = []

            for source_data in data.values():
                for item in source_data:
                    username = item.get('username', item.get('author', 'anonymous'))
                    timestamp = item.get('timestamp', item.get('created_at'))

                    if username != 'anonymous':
                        user_activities[username].append({
                            'timestamp': timestamp,
                            'engagement': item.get('likes', 0) + item.get('comments', 0),
                            'content_length': len(item.get('content', ''))
                        })

                    all_interactions.append(item)

            # Active users analysis
            behavior_metrics['active_users'] = {
                'total_unique_users': len(user_activities),
                'most_active_users': sorted(
                    [(user, len(activities)) for user, activities in user_activities.items()],
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
            }

            # Engagement patterns
            hourly_activity = defaultdict(int)
            daily_activity = defaultdict(int)

            for interaction in all_interactions:
                timestamp = interaction.get('timestamp', interaction.get('created_at'))
                if timestamp:
                    try:
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        hourly_activity[dt.hour] += 1
                        daily_activity[dt.strftime('%A')] += 1
                    except:
                        continue

            behavior_metrics['engagement_patterns'] = {
                'peak_hours': sorted(hourly_activity.items(), key=lambda x: x[1], reverse=True)[:5],
                'peak_days': sorted(daily_activity.items(), key=lambda x: x[1], reverse=True)
            }

        except Exception as e:
            logger.error(f"Error in user behavior analysis: {e}")

        return behavior_metrics

    async def _analyze_competition(self, data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Analyze competitive landscape"""
        competitive_metrics = {
            'competitor_mentions': {},
            'market_share_indicators': {},
            'competitive_topics': [],
            'opportunity_gaps': []
        }

        try:
            # Define common competitors (this would be configurable)
            competitors = ['openai', 'google', 'microsoft', 'meta', 'anthropic', 'amazon']

            all_content = []
            for source_data in data.values():
                all_content.extend([item.get('content', '').lower() for item in source_data])

            combined_content = ' '.join(all_content)

            # Count competitor mentions
            for competitor in competitors:
                mentions = combined_content.count(competitor.lower())
                if mentions > 0:
                    competitive_metrics['competitor_mentions'][competitor] = mentions

            # Identify competitive topics
            competitive_topics = []
            for item in all_content:
                for competitor in competitors:
                    if competitor in item.lower():
                        # Extract context around competitor mention
                        words = item.split()
                        for i, word in enumerate(words):
                            if competitor in word.lower():
                                context = ' '.join(words[max(0, i - 3):i + 4])
                                competitive_topics.append(context)

            competitive_metrics['competitive_topics'] = competitive_topics[:20]

        except Exception as e:
            logger.error(f"Error in competitive analysis: {e}")

        return competitive_metrics

    async def _detect_anomalies(self, data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Detect anomalies in community data"""
        anomalies = {
            'unusual_activity_spikes': [],
            'sentiment_anomalies': [],
            'engagement_anomalies': [],
            'content_anomalies': []
        }

        try:
            # Analyze activity patterns for spikes
            hourly_counts = defaultdict(int)
            all_items = []

            for source_data in data.values():
                for item in source_data:
                    all_items.append(item)
                    timestamp = item.get('timestamp', item.get('created_at'))
                    if timestamp:
                        try:
                            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                            hour_key = dt.strftime('%Y-%m-%d-%H')
                            hourly_counts[hour_key] += 1
                        except:
                            continue

            # Detect activity spikes (simple threshold-based)
            if hourly_counts:
                counts = list(hourly_counts.values())
                mean_count = np.mean(counts)
                std_count = np.std(counts)
                threshold = mean_count + 2 * std_count

                for hour, count in hourly_counts.items():
                    if count > threshold:
                        anomalies['unusual_activity_spikes'].append({
                            'hour': hour,
                            'count': count,
                            'deviation': count - mean_count
                        })

            # Detect sentiment anomalies
            sentiments = []
            for item in all_items:
                content = item.get('content', '')
                if content:
                    blob = TextBlob(content)
                    sentiment = blob.sentiment.polarity
                    sentiments.append({
                        'content': content[:100] + '...',
                        'sentiment': sentiment,
                        'timestamp': item.get('timestamp', item.get('created_at'))
                    })

            if sentiments:
                sentiment_scores = [s['sentiment'] for s in sentiments]
                mean_sentiment = np.mean(sentiment_scores)
                std_sentiment = np.std(sentiment_scores)

                # Find extremely positive or negative sentiments
                for item in sentiments:
                    if abs(item['sentiment'] - mean_sentiment) > 2 * std_sentiment:
                        anomalies['sentiment_anomalies'].append(item)

        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")

        return anomalies

    async def _generate_predictive_insights(self, data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Generate predictive insights based on trends"""
        predictions = {
            'trending_predictions': [],
            'engagement_forecast': {},
            'content_recommendations': [],
            'timing_predictions': {}
        }

        try:
            # Simple trend analysis for predictions
            all_items = []
            for source_data in data.values():
                all_items.extend(source_data)

            # Analyze growth trends in topics
            topic_mentions = defaultdict(list)
            for item in all_items:
                content = item.get('content', '').lower()
                timestamp = item.get('timestamp', item.get('created_at'))

                # Extract potential trending topics (simple keyword extraction)
                words = re.findall(r'\b[a-z]{3,}\b', content)
                for word in words:
                    if len(word) > 3:  # Filter short words
                        topic_mentions[word].append(timestamp)

            # Predict trending topics based on recent activity
            recent_topics = []
            for topic, timestamps in topic_mentions.items():
                if len(timestamps) >= 3:  # Minimum threshold for trend
                    recent_topics.append({
                        'topic': topic,
                        'mentions': len(timestamps),
                        'predicted_growth': 'increasing' if len(timestamps) > 5 else 'stable'
                    })

            predictions['trending_predictions'] = sorted(
                recent_topics, key=lambda x: x['mentions'], reverse=True
            )[:10]

            # Simple engagement forecast
            engagements = []
            for item in all_items:
                total_engagement = (item.get('likes', 0) +
                                    item.get('comments', 0) +
                                    item.get('shares', 0))
                engagements.append(total_engagement)

            if engagements:
                predictions['engagement_forecast'] = {
                    'average_current': np.mean(engagements),
                    'predicted_next_period': np.mean(engagements) * 1.1,  # Simple 10% growth prediction
                    'confidence': 'medium'
                }

        except Exception as e:
            logger.error(f"Error in predictive insights: {e}")

        return predictions
