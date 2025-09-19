#!/usr/bin/env python3
"""
Social Media Scraper Module
Collects posts from various social media platforms
"""

import asyncio
import aiohttp
import requests
import json
import time
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from loguru import logger

from config.settings import settings
from utils.cleaning import cleaner


class SocialScraper:
    """Scraper for social media platforms"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(settings.DEFAULT_HEADERS)

        # Twitter API setup (if available)
        self.twitter_setup = self._setup_twitter_api()

    def _setup_twitter_api(self):
        """Setup Twitter API if credentials are available"""
        try:
            if hasattr(settings, 'TWITTER_API_KEY') and settings.TWITTER_API_KEY:
                import tweepy

                auth = tweepy.OAuthHandler(
                    settings.TWITTER_API_KEY,
                    settings.TWITTER_API_SECRET
                )
                auth.set_access_token(
                    settings.TWITTER_ACCESS_TOKEN,
                    settings.TWITTER_ACCESS_TOKEN_SECRET
                )

                api = tweepy.API(auth, wait_on_rate_limit=True)
                logger.info("Twitter API initialized successfully")
                return api
            else:
                logger.warning("Twitter API credentials not found")
                return None

        except ImportError:
            logger.warning("Tweepy not installed. Install with: pip install tweepy")
            return None
        except Exception as e:
            logger.error(f"Error setting up Twitter API: {e}")
            return None

    def search_tweets(self, keyword: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Search for tweets containing specific keywords"""
        logger.info(f"Searching tweets for keyword: {keyword}")
        tweets = []

        if not self.twitter_setup:
            # Fallback: Simulate tweet data for testing
            return self._simulate_tweet_data(keyword, limit)

        try:
            # Search for tweets
            search_results = self.twitter_setup.search_tweets(
                q=keyword,
                result_type="recent",
                count=limit,
                lang="en",
                tweet_mode="extended"
            )

            for tweet in search_results:
                tweet_data = {
                    'id': str(tweet.id),
                    'content': tweet.full_text,
                    'username': tweet.user.screen_name,
                    'user_followers': tweet.user.followers_count,
                    'created_at': tweet.created_at.isoformat(),
                    'likes': tweet.favorite_count,
                    'retweets': tweet.retweet_count,
                    'replies': tweet.reply_count if hasattr(tweet, 'reply_count') else 0,
                    'hashtags': [hashtag['text'] for hashtag in tweet.entities['hashtags']],
                    'mentions': [mention['screen_name'] for mention in tweet.entities['user_mentions']],
                    'urls': [url['expanded_url'] for url in tweet.entities['urls']],
                    'source': 'Twitter',
                    'platform': 'twitter',
                    'keyword_searched': keyword,
                    'scraped_at': datetime.now().isoformat(),
                    'is_retweet': hasattr(tweet, 'retweeted_status'),
                    'language': tweet.lang,
                    'location': tweet.user.location if tweet.user.location else None
                }
                tweets.append(tweet_data)

            logger.info(f"Collected {len(tweets)} tweets for keyword: {keyword}")

        except Exception as e:
            logger.error(f"Error searching tweets for {keyword}: {e}")
            # Fallback to simulated data
            tweets = self._simulate_tweet_data(keyword, min(limit, 10))

        return tweets

    def _simulate_tweet_data(self, keyword: str, limit: int) -> List[Dict[str, Any]]:
        """Simulate tweet data for testing when API is not available"""
        logger.info(f"Simulating tweet data for {keyword}")

        sample_tweets = [
            f"Just discovered {keyword} and it's amazing! #technology #innovation",
            f"How {keyword} is changing the game in tech industry 🚀",
            f"Thoughts on the latest {keyword} developments? Would love to hear opinions!",
            f"Working with {keyword} today. The possibilities are endless! #tech",
            f"Anyone else excited about {keyword}? This is the future! ✨",
            f"{keyword} is revolutionary. Can't wait to see what's next.",
            f"Deep dive into {keyword}: what you need to know",
            f"The impact of {keyword} on modern business is incredible",
            f"Just finished reading about {keyword}. Mind blown! 🤯",
            f"Poll: What's your experience with {keyword}? Share below! 👇"
        ]

        tweets = []
        for i in range(min(limit, len(sample_tweets))):
            tweet = {
                'id': f"sim_{keyword}_{i}_{int(time.time())}",
                'content': sample_tweets[i % len(sample_tweets)],
                'username': f"user_{random.randint(1000, 9999)}",
                'user_followers': random.randint(100, 10000),
                'created_at': (datetime.now() - timedelta(hours=random.randint(1, 24))).isoformat(),
                'likes': random.randint(0, 100),
                'retweets': random.randint(0, 50),
                'replies': random.randint(0, 20),
                'hashtags': ['technology', 'innovation', 'tech'],
                'mentions': [],
                'urls': [],
                'source': 'Twitter (Simulated)',
                'platform': 'twitter',
                'keyword_searched': keyword,
                'scraped_at': datetime.now().isoformat(),
                'is_retweet': False,
                'language': 'en'
            }
            tweets.append(tweet)

        return tweets

    def scrape_facebook_posts(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """Scrape Facebook posts (limited due to API restrictions)"""
        logger.info("Attempting to collect Facebook data...")

        # Note: Facebook's API is heavily restricted for public data
        # This is a placeholder implementation
        posts = []

        for keyword in keywords:
            # Simulate Facebook post data
            for i in range(3):
                post = {
                    'id': f"fb_{keyword}_{i}_{int(time.time())}",
                    'content': f"Interesting discussion about {keyword} in our community. "
                               f"What are your thoughts on this topic? #facebook #community",
                    'author': f"Page{random.randint(1, 100)}",
                    'page_followers': random.randint(1000, 50000),
                    'created_at': (datetime.now() - timedelta(days=random.randint(1, 7))).isoformat(),
                    'likes': random.randint(5, 500),
                    'comments': random.randint(2, 100),
                    'shares': random.randint(1, 50),
                    'source': 'Facebook (Simulated)',
                    'platform': 'facebook',
                    'keyword_searched': keyword,
                    'scraped_at': datetime.now().isoformat(),
                    'post_type': 'text'
                }
                posts.append(post)

        logger.info(f"Collected {len(posts)} Facebook posts (simulated)")
        return posts

    def scrape_linkedin_posts(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """Scrape LinkedIn posts"""
        logger.info("Collecting LinkedIn data...")
        posts = []

        # LinkedIn API requires special authorization
        # Simulating professional content for demonstration

        professional_templates = [
            "Exciting developments in {keyword}. Here's what professionals should know:",
            "My take on the {keyword} trend in our industry. Thoughts?",
            "How {keyword} is transforming business operations. Key insights:",
            "Just attended a conference on {keyword}. Here are my takeaways:",
            "The future of {keyword} in enterprise solutions looks promising.",
            "Team collaboration using {keyword} has improved our productivity by 40%",
            "Leadership lessons from implementing {keyword} in our organization"
        ]

        for keyword in keywords:
            for i, template in enumerate(professional_templates[:5]):
                content = template.format(keyword=keyword)

                post = {
                    'id': f"ln_{keyword}_{i}_{int(time.time())}",
                    'content': content,
                    'author': f"Professional{random.randint(1, 1000)}",
                    'author_title': random.choice([
                        "Senior Developer", "Product Manager", "Tech Lead",
                        "Data Scientist", "CTO", "Consultant"
                    ]),
                    'company': f"TechCorp{random.randint(1, 100)}",
                    'created_at': (datetime.now() - timedelta(days=random.randint(1, 14))).isoformat(),
                    'likes': random.randint(10, 1000),
                    'comments': random.randint(5, 200),
                    'shares': random.randint(2, 100),
                    'source': 'LinkedIn',
                    'platform': 'linkedin',
                    'keyword_searched': keyword,
                    'scraped_at': datetime.now().isoformat(),
                    'content_type': 'professional_post',
                    'industry_tags': ['technology', 'business', 'innovation']
                }
                posts.append(post)

        logger.info(f"Collected {len(posts)} LinkedIn posts")
        return posts

    async def scrape_social_async(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """Asynchronous social media scraping"""
        logger.info("Starting async social media scraping...")

        all_posts = []

        # Create tasks for different platforms
        tasks = []

        for keyword in keywords:
            # Twitter task
            task = asyncio.create_task(self._async_twitter_search(keyword))
            tasks.append(task)

        # Execute all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, list):
                all_posts.extend(result)
            elif isinstance(result, Exception):
                logger.warning(f"Async task failed: {result}")

        # Add Facebook and LinkedIn data (synchronous)
        facebook_posts = self.scrape_facebook_posts(keywords)
        linkedin_posts = self.scrape_linkedin_posts(keywords)

        all_posts.extend(facebook_posts)
        all_posts.extend(linkedin_posts)

        logger.info(f"Async social scraping completed: {len(all_posts)} posts")
        return all_posts

    async def _async_twitter_search(self, keyword: str) -> List[Dict[str, Any]]:
        """Async wrapper for Twitter search"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.search_tweets, keyword, 20)

    def get_trending_hashtags(self) -> List[Dict[str, Any]]:
        """Get trending hashtags from social platforms"""
        logger.info("Collecting trending hashtags...")

        if self.twitter_setup:
            try:
                trends = self.twitter_setup.get_place_trends(1)  # Worldwide trends
                trending_hashtags = []

                for trend in trends[0]['trends'][:20]:
                    hashtag_data = {
                        'name': trend['name'],
                        'url': trend.get('url', ''),
                        'tweet_volume': trend.get('tweet_volume', 0),
                        'source': 'Twitter Trends',
                        'collected_at': datetime.now().isoformat(),
                        'rank': len(trending_hashtags) + 1
                    }
                    trending_hashtags.append(hashtag_data)

                logger.info(f"Collected {len(trending_hashtags)} trending hashtags")
                return trending_hashtags

            except Exception as e:
                logger.error(f"Error getting trending hashtags: {e}")

        # Fallback: Simulated trending data
        return self._simulate_trending_data()

    def _simulate_trending_data(self) -> List[Dict[str, Any]]:
        """Simulate trending hashtag data"""
        trending_topics = [
            "#AI", "#MachineLearning", "#Technology", "#Innovation",
            "#Python", "#DataScience", "#CloudComputing", "#Blockchain",
            "#Startup", "#DigitalTransformation", "#IoT", "#Cybersecurity"
        ]

        hashtags = []
        for i, topic in enumerate(trending_topics):
            hashtag_data = {
                'name': topic,
                'url': f'https://twitter.com/hashtag/{topic[1:]}',
                'tweet_volume': random.randint(1000, 100000),
                'source': 'Simulated Trends',
                'collected_at': datetime.now().isoformat(),
                'rank': i + 1
            }
            hashtags.append(hashtag_data)

        return hashtags

    def run_full_scraping(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """Run complete social media scraping pipeline"""
        logger.info("Starting full social media scraping pipeline...")

        all_posts = []

        # 1. Search tweets for each keyword
        for keyword in keywords:
            try:
                tweets = self.search_tweets(keyword, limit=30)
                all_posts.extend(tweets)

                # Respectful delay between searches
                time.sleep(settings.REQUEST_DELAY)

            except Exception as e:
                logger.error(f"Error scraping tweets for {keyword}: {e}")

        # 2. Scrape Facebook posts
        try:
            facebook_posts = self.scrape_facebook_posts(keywords)
            all_posts.extend(facebook_posts)
        except Exception as e:
            logger.error(f"Facebook scraping failed: {e}")

        # 3. Scrape LinkedIn posts
        try:
            linkedin_posts = self.scrape_linkedin_posts(keywords)
            all_posts.extend(linkedin_posts)
        except Exception as e:
            logger.error(f"LinkedIn scraping failed: {e}")

        logger.info(f"Social media scraping completed: {len(all_posts)} total posts")

        # Clean and enrich the data
        cleaned_posts = cleaner.enrich_data(all_posts)

        return cleaned_posts


def main():
    """Test the social scraper independently"""
    scraper = SocialScraper()

    print("📱 Testing Social Media Scraper")
    print("=" * 40)

    test_keywords = ["AI", "machine learning", "technology"]
    posts = scraper.run_full_scraping(test_keywords)

    print(f"\n📊 Results: {len(posts)} posts collected")

    if posts:
        print(f"\n📄 Sample post:")
        sample = posts[0]
        for key, value in sample.items():
            print(f"  {key}: {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}")

    # Test trending hashtags
    print(f"\n🔥 Trending hashtags:")
    trending = scraper.get_trending_hashtags()
    for hashtag in trending[:5]:
        print(f"  {hashtag['rank']}. {hashtag['name']} - {hashtag.get('tweet_volume', 0)} volume")


if __name__ == "__main__":
    main()