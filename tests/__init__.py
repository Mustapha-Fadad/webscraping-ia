"""
Integration test for all scrapers
"""

import asyncio
import time
from datetime import datetime


def test_all_scrapers():
    """Test all scrapers in sequence"""

    print("🤖 AI Community Manager - Scraper Integration Test")
    print("=" * 60)

    # Test News Scraper
    print("\n1. Testing News Scraper...")
    try:
        from scrapers.news_scraper import NewsScraper
        news_scraper = NewsScraper()
        news_articles = news_scraper.scrape_techcrunch()
        print(f"   ✅ News Scraper: {len(news_articles)} articles collected")
    except Exception as e:
        print(f"   ❌ News Scraper failed: {e}")

    # Test Social Scraper
    print("\n2. Testing Social Scraper...")
    try:
        from scrapers.social_scraper import SocialScraper
        social_scraper = SocialScraper()
        social_posts = social_scraper.search_tweets("AI", limit=10)
        print(f"   ✅ Social Scraper: {len(social_posts)} posts collected")
    except Exception as e:
        print(f"   ❌ Social Scraper failed: {e}")

    # Test Forum Scraper
    print("\n3. Testing Forum Scraper...")
    try:
        from scrapers.forum_scraper import ForumScraper
        forum_scraper = ForumScraper()
        forum_posts = forum_scraper.scrape_reddit("artificial", limit=10)
        print(f"   ✅ Forum Scraper: {len(forum_posts)} posts collected")
    except Exception as e:
        print(f"   ❌ Forum Scraper failed: {e}")

    # Test Trend Scraper
    print("\n4. Testing Trend Scraper...")
    try:
        from scrapers.trend_scraper import TrendScraper
        trend_scraper = TrendScraper()
        trend_analysis = trend_scraper.analyze_hashtag_trends()
        print(f"   ✅ Trend Scraper: {len(trend_analysis)} trends analyzed")
    except Exception as e:
        print(f"   ❌ Trend Scraper failed: {e}")

    print("\n" + "=" * 60)
    print("Integration test completed!")
    print(f"Test run at: {datetime.now().isoformat()}")


if __name__ == "__main__":
    test_all_scrapers()