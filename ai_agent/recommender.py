#!/usr/bin/env python3
"""
Content Recommender Module
Generates content recommendations based on community analysis
"""

import random
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger
import json


class ContentRecommender:
    """AI-powered content recommendation engine"""

    def __init__(self):
        self.content_templates = {
            'tutorial': [
                "Step-by-step guide to {topic}",
                "Beginner's tutorial: Understanding {topic}",
                "Advanced {topic} techniques for professionals",
                "Complete {topic} masterclass for developers",
                "From zero to hero: {topic} in 2024"
            ],
            'discussion': [
                "What's your experience with {topic}?",
                "Hot take: The future of {topic}",
                "Community discussion: Best practices for {topic}",
                "Debate: {topic} vs alternatives",
                "Weekly thread: Share your {topic} projects"
            ],
            'news': [
                "Breaking: Latest developments in {topic}",
                "Industry update: {topic} trends for 2024",
                "Research spotlight: New {topic} breakthrough",
                "Market analysis: {topic} adoption rates",
                "Expert opinion: Where {topic} is heading"
            ],
            'showcase': [
                "Show and tell: My {topic} project",
                "Community showcase: Best {topic} implementations",
                "Portfolio spotlight: {topic} success stories",
                "Before and after: {topic} transformation",
                "Case study: How {topic} solved our problem"
            ],
            'qa': [
                "FAQ: Common {topic} questions answered",
                "Ask me anything: {topic} expert session",
                "Troubleshooting guide: {topic} common issues",
                "Q&A compilation: {topic} best practices",
                "Expert answers: Your {topic} questions"
            ]
        }

        self.engagement_strategies = [
            "Ask for community input and experiences",
            "Include interactive elements like polls or quizzes",
            "Share behind-the-scenes content",
            "Create comparison posts",
            "Host live Q&A sessions",
            "Share user-generated content",
            "Create challenge or contest posts",
            "Provide exclusive insights or tips",
            "Share failure stories and lessons learned",
            "Create collaborative content with community members"
        ]

        self.optimal_times = {
            'weekday_morning': {'hour': 9, 'engagement_boost': 1.2},
            'weekday_afternoon': {'hour': 15, 'engagement_boost': 1.1},
            'weekday_evening': {'hour': 19, 'engagement_boost': 1.3},
            'weekend_morning': {'hour': 10, 'engagement_boost': 1.0},
            'weekend_afternoon': {'hour': 14, 'engagement_boost': 1.1}
        }

    def generate_recommendations(self, analysis_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate content recommendations based on analysis"""
        logger.info("Generating content recommendations...")

        recommendations = []

        try:
            # Extract key information from analysis
            trends = analysis_data.get('trends', [])
            sentiment = analysis_data.get('sentiment_analysis', {})
            insights = analysis_data.get('insights', [])
            basic_stats = analysis_data.get('basic_stats', {})

            # 1. Trend-based recommendations
            trend_recs = self._generate_trend_recommendations(trends)
            recommendations.extend(trend_recs)

            # 2. Sentiment-based recommendations
            sentiment_recs = self._generate_sentiment_recommendations(sentiment)
            recommendations.extend(sentiment_recs)

            # 3. Engagement-based recommendations
            engagement_recs = self._generate_engagement_recommendations(basic_stats)
            recommendations.extend(engagement_recs)

            # 4. Timing recommendations
            timing_recs = self._generate_timing_recommendations()
            recommendations.extend(timing_recs)

            # 5. Content type recommendations
            content_recs = self._generate_content_type_recommendations(analysis_data)
            recommendations.extend(content_recs)

            # Sort by priority and limit results
            recommendations.sort(key=lambda x: x.get('priority', 0), reverse=True)
            recommendations = recommendations[:15]  # Limit to top 15

            # Add metadata
            for i, rec in enumerate(recommendations):
                rec['rank'] = i + 1
                rec['generated_at'] = datetime.now().isoformat()
                rec['confidence_score'] = self._calculate_confidence_score(rec, analysis_data)

            logger.info(f"Generated {len(recommendations)} recommendations")

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            # Fallback to basic recommendations
            recommendations = self._generate_fallback_recommendations()

        return recommendations

    def _generate_trend_recommendations(self, trends: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate recommendations based on trending topics"""
        recommendations = []

        for trend in trends[:5]:  # Top 5 trends
            topic = trend.get('topic', 'technology')
            mentions = trend.get('mentions', 0)
            velocity = trend.get('velocity', 0)
            direction = trend.get('direction', 'stable')

            # Determine recommendation type based on trend characteristics
            if direction == 'rising' and velocity > 2:
                rec_type = 'news'
                priority = 9
                urgency = 'high'
            elif direction == 'rising':
                rec_type = 'discussion'
                priority = 8
                urgency = 'medium'
            else:
                rec_type = 'tutorial'
                priority = 6
                urgency = 'low'

            template = random.choice(self.content_templates[rec_type])
            title = template.format(topic=topic)

            recommendation = {
                'id': f"trend_{trend.get('topic', 'unknown').replace(' ', '_')}_{int(datetime.now().timestamp())}",
                'type': 'trend_based',
                'content_type': rec_type,
                'title': title,
                'topic': topic,
                'priority': priority,
                'urgency': urgency,
                'reasoning': f"Trending topic with {mentions} mentions, {direction} trend",
                'target_audience': 'general',
                'estimated_engagement': self._estimate_engagement(priority, velocity),
                'hashtags': [f"#{topic.replace(' ', '')}", "#trending", "#tech"],
                'trend_data': trend
            }

            recommendations.append(recommendation)

        return recommendations

    def _generate_sentiment_recommendations(self, sentiment_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations based on sentiment analysis"""
        recommendations = []

        if not sentiment_data:
            return recommendations

        overall_sentiment = sentiment_data.get('overall_sentiment', 0)
        distribution = sentiment_data.get('sentiment_distribution', {})

        negative_ratio = distribution.get('negative', 0)
        positive_ratio = distribution.get('positive', 0)

        if negative_ratio > 0.4:  # High negative sentiment
            recommendation = {
                'id': f"sentiment_negative_{int(datetime.now().timestamp())}",
                'type': 'sentiment_response',
                'content_type': 'discussion',
                'title': "Addressing community concerns: Let's talk",
                'priority': 9,
                'urgency': 'high',
                'reasoning': f"High negative sentiment ({negative_ratio:.1%}) requires attention",
                'target_audience': 'concerned_users',
                'estimated_engagement': 'high',
                'hashtags': ["#community", "#feedback", "#improvement"],
                'action_required': 'immediate_response',
                'strategy': 'transparency_and_support'
            }
            recommendations.append(recommendation)

        elif positive_ratio > 0.7:  # High positive sentiment
            recommendation = {
                'id': f"sentiment_positive_{int(datetime.now().timestamp())}",
                'type': 'sentiment_amplification',
                'content_type': 'showcase',
                'title': "Celebrating our amazing community achievements",
                'priority': 7,
                'urgency': 'medium',
                'reasoning': f"High positive sentiment ({positive_ratio:.1%}) - good time to amplify",
                'target_audience': 'engaged_community',
                'estimated_engagement': 'high',
                'hashtags': ["#celebration", "#community", "#success"],
                'strategy': 'amplify_positive_momentum'
            }
            recommendations.append(recommendation)

        return recommendations

    def _generate_engagement_recommendations(self, basic_stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations based on engagement patterns"""
        recommendations = []

        if not basic_stats:
            return recommendations

        total_articles = basic_stats.get('total_articles', 0)
        sources = basic_stats.get('sources', {})

        if total_articles > 50:  # High activity
            recommendation = {
                'id': f"engagement_high_activity_{int(datetime.now().timestamp())}",
                'type': 'engagement_optimization',
                'content_type': 'discussion',
                'title': "Community roundup: This week's highlights",
                'priority': 6,
                'urgency': 'low',
                'reasoning': f"High community activity ({total_articles} posts) - good time for summary content",
                'target_audience': 'active_community',
                'estimated_engagement': 'medium',
                'hashtags': ["#roundup", "#community", "#highlights"],
                'strategy': 'content_curation'
            }
            recommendations.append(recommendation)

        # Source diversity recommendations
        if len(sources) > 5:
            recommendation = {
                'id': f"engagement_diverse_sources_{int(datetime.now().timestamp())}",
                'type': 'content_diversity',
                'content_type': 'showcase',
                'title': "Community spotlight: Diverse voices and perspectives",
                'priority': 5,
                'urgency': 'low',
                'reasoning': f"Content from {len(sources)} different sources shows good diversity",
                'target_audience': 'general',
                'estimated_engagement': 'medium',
                'hashtags': ["#diversity", "#community", "#voices"],
                'strategy': 'highlight_diversity'
            }
            recommendations.append(recommendation)

        return recommendations

    def _generate_timing_recommendations(self) -> List[Dict[str, Any]]:
        """Generate optimal timing recommendations"""
        recommendations = []

        current_time = datetime.now()
        current_hour = current_time.hour

        # Find best upcoming time slot
        best_time = None
        max_boost = 0

        for time_slot, data in self.optimal_times.items():
            slot_hour = data['hour']
            boost = data['engagement_boost']

            # Calculate hours until this time slot
            if slot_hour > current_hour:
                hours_until = slot_hour - current_hour
            else:
                hours_until = 24 - current_hour + slot_hour

            if boost > max_boost and hours_until <= 12:  # Within next 12 hours
                max_boost = boost
                best_time = {
                    'slot': time_slot,
                    'hour': slot_hour,
                    'boost': boost,
                    'hours_until': hours_until
                }

        if best_time:
            recommendation = {
                'id': f"timing_optimal_{int(datetime.now().timestamp())}",
                'type': 'timing_optimization',
                'content_type': 'meta',
                'title': f"Optimal posting time: {best_time['hour']}:00",
                'priority': 4,
                'urgency': 'low',
                'reasoning': f"Peak engagement time with {best_time['boost']:.1f}x boost",
                'target_audience': 'content_creators',
                'estimated_engagement': f"{best_time['boost']:.1f}x normal",
                'schedule_for': (current_time + timedelta(hours=best_time['hours_until'])).isoformat(),
                'strategy': 'timing_optimization'
            }
            recommendations.append(recommendation)

        return recommendations

    def _generate_content_type_recommendations(self, analysis_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations for different content types"""
        recommendations = []

        # Educational content recommendation
        recommendations.append({
            'id': f"content_educational_{int(datetime.now().timestamp())}",
            'type': 'content_type',
            'content_type': 'tutorial',
            'title': "Create comprehensive learning resource",
            'priority': 7,
            'urgency': 'medium',
            'reasoning': "Educational content typically performs well in tech communities",
            'target_audience': 'learners',
            'estimated_engagement': 'high',
            'hashtags': ["#tutorial", "#learning", "#education"],
            'strategy': 'knowledge_sharing'
        })

        # Interactive content recommendation
        recommendations.append({
            'id': f"content_interactive_{int(datetime.now().timestamp())}",
            'type': 'content_type',
            'content_type': 'qa',
            'title': "Host interactive Q&A session",
            'priority': 6,
            'urgency': 'medium',
            'reasoning': "Interactive content drives engagement and community participation",
            'target_audience': 'active_community',
            'estimated_engagement': 'very_high',
            'hashtags': ["#AMA", "#QA", "#interactive"],
            'strategy': 'community_engagement'
        })

        return recommendations

    def _generate_fallback_recommendations(self) -> List[Dict[str, Any]]:
        """Generate basic recommendations when analysis data is insufficient"""
        logger.info("Generating fallback recommendations...")

        fallback_recommendations = [
            {
                'id': f"fallback_welcome_{int(datetime.now().timestamp())}",
                'type': 'community_building',
                'content_type': 'discussion',
                'title': "Welcome new community members",
                'priority': 5,
                'urgency': 'low',
                'reasoning': "Regular community building activity",
                'target_audience': 'new_members',
                'estimated_engagement': 'medium',
                'hashtags': ["#welcome", "#community", "#newmembers"],
                'strategy': 'community_building'
            },
            {
                'id': f"fallback_tips_{int(datetime.now().timestamp())}",
                'type': 'educational',
                'content_type': 'tutorial',
                'title': "Share useful tips and tricks",
                'priority': 6,
                'urgency': 'low',
                'reasoning': "Educational content is always valuable",
                'target_audience': 'general',
                'estimated_engagement': 'medium',
                'hashtags': ["#tips", "#tricks", "#learning"],
                'strategy': 'knowledge_sharing'
            },
            {
                'id': f"fallback_showcase_{int(datetime.now().timestamp())}",
                'type': 'community_engagement',
                'content_type': 'showcase',
                'title': "Showcase community projects",
                'priority': 7,
                'urgency': 'medium',
                'reasoning': "Highlighting community work drives engagement",
                'target_audience': 'creators',
                'estimated_engagement': 'high',
                'hashtags': ["#showcase", "#projects", "#community"],
                'strategy': 'community_highlights'
            }
        ]

        # Add standard metadata
        for i, rec in enumerate(fallback_recommendations):
            rec['rank'] = i + 1
            rec['generated_at'] = datetime.now().isoformat()
            rec['confidence_score'] = 0.6  # Medium confidence for fallback

        return fallback_recommendations

    def _estimate_engagement(self, priority: int, velocity: float) -> str:
        """Estimate engagement level based on priority and velocity"""
        score = priority + (velocity * 2)

        if score >= 12:
            return 'very_high'
        elif score >= 9:
            return 'high'
        elif score >= 6:
            return 'medium'
        else:
            return 'low'

    def _calculate_confidence_score(self, recommendation: Dict[str, Any], analysis_data: Dict[str, Any]) -> float:
        """Calculate confidence score for recommendation"""
        base_confidence = 0.7

        # Boost confidence based on data quality
        total_items = analysis_data.get('summary', {}).get('total_items', 0)
        if total_items > 100:
            base_confidence += 0.2
        elif total_items > 50:
            base_confidence += 0.1

        # Boost confidence for trend-based recommendations
        if recommendation.get('type') == 'trend_based':
            trend_data = recommendation.get('trend_data', {})
            if trend_data.get('confidence', 0) > 0.8:
                base_confidence += 0.1

        # Cap confidence at 0.95
        return min(0.95, base_confidence)

    def export_recommendations(self, recommendations: List[Dict[str, Any]]) -> str:
        """Export recommendations as JSON string"""
        export_data = {
            'generated_at': datetime.now().isoformat(),
            'total_recommendations': len(recommendations),
            'recommendations': recommendations,
            'metadata': {
                'generator': 'AI Community Manager',
                'version': '1.0',
                'confidence_levels': {
                    'high': len([r for r in recommendations if r.get('confidence_score', 0) > 0.8]),
                    'medium': len([r for r in recommendations if 0.6 <= r.get('confidence_score', 0) <= 0.8]),
                    'low': len([r for r in recommendations if r.get('confidence_score', 0) < 0.6])
                }
            }
        }

        return json.dumps(export_data, indent=2, ensure_ascii=False)


def main():
    """Test the recommender independently"""
    recommender = ContentRecommender()

    print("💡 Testing Content Recommender")
    print("=" * 40)

    # Sample analysis data for testing
    sample_analysis = {
        'trends': [
            {
                'topic': 'artificial intelligence',
                'mentions': 150,
                'velocity': 3.2,
                'direction': 'rising',
                'confidence': 0.85
            },
            {
                'topic': 'machine learning',
                'mentions': 89,
                'velocity': 1.8,
                'direction': 'stable',
                'confidence': 0.72
            }
        ],
        'sentiment_analysis': {
            'overall_sentiment': 0.65,
            'sentiment_distribution': {
                'positive': 0.75,
                'negative': 0.15,
                'neutral': 0.10
            }
        },
        'summary': {
            'total_items': 125
        }
    }

    # Generate recommendations
    recommendations = recommender.generate_recommendations(sample_analysis)

    print(f"\n📊 Generated {len(recommendations)} recommendations")

    if recommendations:
        print("\n🔝 Top 3 recommendations:")
        for i, rec in enumerate(recommendations[:3], 1):
            print(f"\n{i}. {rec.get('title', 'No title')}")
            print(f"   Type: {rec.get('content_type', 'Unknown')}")
            print(f"   Priority: {rec.get('priority', 0)}")
            print(f"   Confidence: {rec.get('confidence_score', 0):.2f}")
            print(f"   Reasoning: {rec.get('reasoning', 'No reasoning provided')}")

    print("\n✅ Recommender test completed")


if __name__ == "__main__":
    main()