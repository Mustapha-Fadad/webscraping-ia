#!/usr/bin/env python3
"""
Community Manager Orchestrator
Coordinates all AI community management activities
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from loguru import logger

# Import scrapers
from scrapers.news_scraper import NewsScraper
from scrapers.social_scraper import SocialScraper
from scrapers.forum_scraper import ForumScraper
from scrapers.trend_scraper import TrendScraper

# Import AI agent components
from ai_agent.recommender import ContentRecommender

# Import utilities
from utils.analytics import analytics
from utils.storage import DataStorage
from utils.cleaning import cleaner

# Import settings
from config.settings import settings


class CommunityOrchestrator:
    """Main orchestrator for AI community management system"""

    def __init__(self):
        # Initialize components
        self.news_scraper = NewsScraper()
        self.social_scraper = SocialScraper()
        self.forum_scraper = ForumScraper()
        self.trend_scraper = TrendScraper()
        self.recommender = ContentRecommender()
        self.storage = DataStorage()

        # Configuration
        self.subreddits = ["artificial", "MachineLearning", "programming", "datascience", "technology"]
        self.keywords = ["AI", "artificial intelligence", "machine learning", "data science", "python"]

    async def collect_all_data(self) -> List[Dict[str, Any]]:
        """Collect data from all sources asynchronously"""
        logger.info("Starting data collection from all sources...")

        all_data = []

        # Create collection tasks
        tasks = [
            self._collect_news_data(),
            self._collect_social_data(),
            self._collect_forum_data(),
            self._collect_trend_data()
        ]

        # Execute tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Collection task {i} failed: {result}")
            elif isinstance(result, list):
                all_data.extend(result)
                logger.info(f"Task {i} collected {len(result)} items")

        # Remove duplicates and clean data
        if all_data:
            cleaned_data = cleaner.remove_duplicates(all_data, key_field='content')
            enriched_data = cleaner.enrich_data(cleaned_data)
            all_data = enriched_data

        logger.info(f"Data collection completed: {len(all_data)} items collected")
        return all_data

    async def _collect_news_data(self) -> List[Dict[str, Any]]:
        """Collect news data"""
        try:
            logger.info("Collecting news data...")

            # Run news scraping in executor to avoid blocking
            loop = asyncio.get_event_loop()
            news_data = await loop.run_in_executor(None, self.news_scraper.run_full_scraping)

            # Save to storage
            if news_data:
                await self.storage.save_raw_data('news', 'articles', news_data)

            logger.info(f"Collected {len(news_data)} news articles")
            return news_data

        except Exception as e:
            logger.error(f"News collection error: {e}")
            return []

    async def _collect_social_data(self) -> List[Dict[str, Any]]:
        """Collect social media data"""
        try:
            logger.info("Collecting social media data...")

            # Use async social scraping
            social_data = await self.social_scraper.scrape_social_async(self.keywords)

            # Save to storage
            if social_data:
                await self.storage.save_raw_data('social', 'posts', social_data)

            logger.info(f"Collected {len(social_data)} social media posts")
            return social_data

        except Exception as e:
            logger.error(f"Social media collection error: {e}")
            return []

    async def _collect_forum_data(self) -> List[Dict[str, Any]]:
        """Collect forum data"""
        try:
            logger.info("Collecting forum data...")

            # Use async forum scraping
            forum_data = await self.forum_scraper.scrape_forums_async(self.subreddits, self.keywords)

            # Save to storage
            if forum_data:
                await self.storage.save_raw_data('forums', 'posts', forum_data)

            logger.info(f"Collected {len(forum_data)} forum posts")
            return forum_data

        except Exception as e:
            logger.error(f"Forum collection error: {e}")
            return []

    async def _collect_trend_data(self) -> List[Dict[str, Any]]:
        """Collect trend data"""
        try:
            logger.info("Collecting trend data...")

            # Run trend analysis in executor
            loop = asyncio.get_event_loop()
            trend_data = await loop.run_in_executor(
                None,
                self.trend_scraper.analyze_hashtag_trends,
                24
            )

            logger.info(f"Collected {len(trend_data)} trend items")
            return trend_data

        except Exception as e:
            logger.error(f"Trend collection error: {e}")
            return []

    async def analyze_data(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze collected data"""
        logger.info("Starting data analysis...")

        if not data:
            logger.warning("No data to analyze")
            return {
                'summary': {
                    'total_items': 0,
                    'analysis_completed': False
                },
                'insights': [],
                'trends': [],
                'sentiment_analysis': {}
            }

        try:
            # Run analytics in executor to avoid blocking
            loop = asyncio.get_event_loop()

            # Basic statistics
            basic_stats = await loop.run_in_executor(
                None,
                analytics.calculate_basic_stats,
                data
            )

            # Trend analysis - only on text data
            text_data = [item for item in data if item.get('content') or item.get('title')]

            if text_data:
                trends = await loop.run_in_executor(
                    None,
                    analytics.analyze_trends,
                    text_data,
                    24,
                    3
                )

                # Sentiment analysis
                sentiment_analysis = await loop.run_in_executor(
                    None,
                    analytics.analyze_sentiment_trends,
                    text_data
                )

                # Generate insights
                insights = await loop.run_in_executor(
                    None,
                    analytics.generate_insights,
                    text_data,
                    trends
                )
            else:
                logger.warning("No text data found for analysis")
                trends = []
                sentiment_analysis = {}
                insights = []

            # Compile analysis results
            analysis_results = {
                'summary': {
                    'total_items': len(data),
                    'text_items': len(text_data),
                    'analysis_completed': True,
                    'analysis_timestamp': datetime.now().isoformat()
                },
                'basic_stats': basic_stats,
                'trends': [
                    {
                        'topic': trend.topic,
                        'mentions': trend.mentions,
                        'velocity': trend.velocity,
                        'sentiment': trend.sentiment,
                        'direction': trend.direction.value,
                        'confidence': trend.confidence
                    }
                    for trend in trends[:10]
                ],
                'sentiment_analysis': sentiment_analysis,
                'insights': [
                    {
                        'title': insight,
                        'type': 'general',
                        'priority': 5
                    }
                    for insight in insights[:10]
                ]
            }

            logger.info(f"Analysis completed: {len(trends)} trends, {len(insights)} insights")
            return analysis_results

        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return {
                'summary': {
                    'total_items': len(data),
                    'analysis_completed': False,
                    'error': str(e)
                },
                'insights': [],
                'trends': [],
                'sentiment_analysis': {}
            }

    async def generate_recommendations(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate content recommendations based on analysis"""
        logger.info("Generating content recommendations...")

        try:
            # Run recommendation generation in executor
            loop = asyncio.get_event_loop()
            recommendations = await loop.run_in_executor(
                None,
                self.recommender.generate_recommendations,
                analysis_results
            )

            logger.info(f"Generated {len(recommendations)} recommendations")
            return recommendations

        except Exception as e:
            logger.error(f"Recommendation generation error: {e}")
            return []

    async def run_analysis_cycle(self) -> Dict[str, Any]:
        """Run complete analysis cycle"""
        cycle_id = f"cycle_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()

        logger.info("Starting complete analysis cycle...")

        try:
            # 1. Collect data from all sources
            all_data = await self.collect_all_data()

            # 2. Analyze the collected data
            analysis_results = await self.analyze_data(all_data)

            # 3. Generate recommendations
            recommendations = await self.generate_recommendations(analysis_results)

            # 4. Compile final results
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            final_results = {
                'cycle_id': cycle_id,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration,
                'summary': {
                    'total_items_collected': len(all_data),
                    'total_insights': len(analysis_results.get('insights', [])),
                    'total_recommendations': len(recommendations),
                    'analysis_success': analysis_results.get('summary', {}).get('analysis_completed', False)
                },
                'data_collection': {
                    'total_items': len(all_data),
                    'sources_accessed': self._get_sources_summary(all_data)
                },
                'analysis': analysis_results,
                'recommendations': recommendations
            }

            # 5. Save results
            analysis_id = await self.storage.save_analysis_results(final_results)
            final_results['analysis_id'] = analysis_id

            logger.info(f"Analysis cycle completed in {duration:.2f} seconds")
            return final_results

        except Exception as e:
            logger.error(f"Analysis cycle failed: {e}")
            return {
                'cycle_id': cycle_id,
                'error': str(e),
                'start_time': start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'summary': {
                    'total_items_collected': 0,
                    'total_insights': 0,
                    'total_recommendations': 0,
                    'analysis_success': False
                }
            }

    def _get_sources_summary(self, data: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get summary of data sources"""
        sources = {}
        for item in data:
            source = item.get('source', 'Unknown')
            sources[source] = sources.get(source, 0) + 1
        return sources

    async def run_continuous_monitoring(self, interval_hours: int = 6):
        """Run continuous monitoring with specified interval"""
        logger.info(f"Starting continuous monitoring with {interval_hours}h interval")

        while True:
            try:
                # Run analysis cycle
                results = await self.run_analysis_cycle()

                if 'error' not in results:
                    logger.info("Monitoring cycle completed successfully")
                else:
                    logger.error(f"Monitoring cycle failed: {results.get('error')}")

                # Wait for next cycle
                await asyncio.sleep(interval_hours * 3600)

            except KeyboardInterrupt:
                logger.info("Continuous monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retry

    async def get_status(self) -> Dict[str, Any]:
        """Get current system status"""
        try:
            stats = self.storage.get_data_statistics()

            status = {
                'timestamp': datetime.now().isoformat(),
                'database_stats': stats,
                'scrapers_status': {
                    'news': 'active',
                    'social': 'active' if self.social_scraper.twitter_setup else 'limited',
                    'forums': 'active' if self.forum_scraper.reddit_setup else 'limited',
                    'trends': 'active'
                },
                'last_analysis': self.storage.load_analysis_results(),
                'configuration': {
                    'subreddits': self.subreddits,
                    'keywords': self.keywords,
                    'scraping_interval': settings.SCRAPING_INTERVAL_HOURS
                }
            }

            return status

        except Exception as e:
            logger.error(f"Status check failed: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'status': 'error'
            }


# Convenience functions for external use
async def run_quick_analysis() -> Dict[str, Any]:
    """Quick analysis function for external use"""
    orchestrator = CommunityOrchestrator()
    return await orchestrator.run_analysis_cycle()


async def collect_data_only() -> List[Dict[str, Any]]:
    """Collect data without full analysis"""
    orchestrator = CommunityOrchestrator()
    return await orchestrator.collect_all_data()


def main():
    """Test the orchestrator independently"""

    async def test_orchestrator():
        orchestrator = CommunityOrchestrator()

        print("🤖 Testing Community Orchestrator")
        print("=" * 50)

        # Test data collection
        print("📊 Testing data collection...")
        data = await orchestrator.collect_all_data()
        print(f"Collected {len(data)} items")

        if data:
            # Test analysis
            print("🔍 Testing analysis...")
            analysis = await orchestrator.analyze_data(data)
            print(f"Analysis completed: {analysis.get('summary', {}).get('analysis_completed', False)}")

            # Test recommendations
            print("💡 Testing recommendations...")
            recommendations = await orchestrator.generate_recommendations(analysis)
            print(f"Generated {len(recommendations)} recommendations")

            # Show sample results
            if recommendations:
                print("\nSample recommendation:")
                sample = recommendations[0]
                for key, value in sample.items():
                    print(f"  {key}: {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}")

        print("\n✅ Orchestrator test completed")

    # Run the test
    asyncio.run(test_orchestrator())


if __name__ == "__main__":
    main()