#!/usr/bin/env python3
"""
Forum Scraper Module
Collects discussions from various forums and Q&A sites
"""

import asyncio
import aiohttp
import requests
import time
import random
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from loguru import logger

from config.settings import settings
from utils.cleaning import cleaner


class ForumScraper:
    """Scraper for forums and Q&A platforms"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(settings.DEFAULT_HEADERS)

        # Reddit API setup
        self.reddit_setup = self._setup_reddit_api()

    def _setup_reddit_api(self):
        """Setup Reddit API if credentials are available"""
        try:
            if hasattr(settings, 'REDDIT_CLIENT_ID') and settings.REDDIT_CLIENT_ID:
                import praw

                reddit = praw.Reddit(
                    client_id=settings.REDDIT_CLIENT_ID,
                    client_secret=settings.REDDIT_CLIENT_SECRET,
                    user_agent=settings.REDDIT_USER_AGENT
                )

                logger.info("Reddit API initialized successfully")
                return reddit
            else:
                logger.warning("Reddit API credentials not found")
                return None

        except ImportError:
            logger.warning("PRAW not installed. Install with: pip install praw")
            return None
        except Exception as e:
            logger.error(f"Error setting up Reddit API: {e}")
            return None

    def scrape_reddit(self, subreddit_name: str, limit: int = 25) -> List[Dict[str, Any]]:
        """Scrape posts from a specific subreddit"""
        # Clean subreddit name - remove r/ prefix if present
        if subreddit_name.startswith('r/'):
            subreddit_name = subreddit_name[2:]

        logger.info(f"Scraping Reddit: r/{subreddit_name}")
        posts = []

        if not self.reddit_setup:
            return self._simulate_reddit_data(subreddit_name, limit)

        try:
            subreddit = self.reddit_setup.subreddit(subreddit_name)

            # Get hot posts
            hot_posts = subreddit.hot(limit=limit)

            for submission in hot_posts:
                post_data = {
                    'id': submission.id,
                    'title': submission.title,
                    'content': submission.selftext if submission.selftext else submission.title,
                    'url': submission.url,
                    'permalink': f"https://reddit.com{submission.permalink}",
                    'author': str(submission.author) if submission.author else '[deleted]',
                    'subreddit': subreddit_name,
                    'score': submission.score,
                    'upvote_ratio': submission.upvote_ratio,
                    'num_comments': submission.num_comments,
                    'created_at': datetime.fromtimestamp(submission.created_utc).isoformat(),
                    'flair': submission.link_flair_text,
                    'is_self': submission.is_self,
                    'is_video': submission.is_video,
                    'domain': submission.domain,
                    'source': 'Reddit',
                    'platform': 'reddit',
                    'scraped_at': datetime.now().isoformat(),
                    'post_type': 'self_post' if submission.is_self else 'link_post'
                }

                # Get top comments
                submission.comments.replace_more(limit=0)
                top_comments = []
                for comment in submission.comments[:5]:
                    if hasattr(comment, 'body'):
                        comment_data = {
                            'body': comment.body,
                            'score': comment.score,
                            'author': str(comment.author) if comment.author else '[deleted]',
                            'created_at': datetime.fromtimestamp(comment.created_utc).isoformat()
                        }
                        top_comments.append(comment_data)

                post_data['top_comments'] = top_comments
                posts.append(post_data)

            logger.info(f"Collected {len(posts)} posts from r/{subreddit_name}")

        except Exception as e:
            logger.error(f"Error scraping r/{subreddit_name}: {e}")
            # Fallback to simulated data
            posts = self._simulate_reddit_data(subreddit_name, min(limit, 10))

        return posts

    def _simulate_reddit_data(self, subreddit_name: str, limit: int) -> List[Dict[str, Any]]:
        """Simulate Reddit data for testing"""
        logger.info(f"Simulating Reddit data for r/{subreddit_name}")

        sample_titles = [
            f"Discussion: What's your experience with AI in {subreddit_name}?",
            f"Tutorial: Getting started with machine learning",
            f"News: Latest developments in technology sector",
            f"Question: Best practices for software development?",
            f"Show and Tell: My latest project using Python",
            f"AMA: I'm a tech entrepreneur, ask me anything!",
            f"Weekly Thread: Share your coding wins and fails",
            f"Resource: Comprehensive guide to data science",
            f"Opinion: The future of artificial intelligence",
            f"Help: Stuck on a complex algorithm problem"
        ]

        posts = []
        for i in range(min(limit, len(sample_titles))):
            post = {
                'id': f"sim_{subreddit_name}_{i}_{int(time.time())}",
                'title': sample_titles[i % len(sample_titles)],
                'content': f"This is a simulated discussion post from r/{subreddit_name}. "
                           f"The community is actively discussing various topics related to the subreddit theme.",
                'url': f"https://reddit.com/r/{subreddit_name}/comments/sim{i}",
                'permalink': f"/r/{subreddit_name}/comments/sim{i}",
                'author': f"user_{random.randint(1000, 9999)}",
                'subreddit': subreddit_name,
                'score': random.randint(1, 1000),
                'upvote_ratio': random.uniform(0.7, 0.95),
                'num_comments': random.randint(0, 200),
                'created_at': (datetime.now() - timedelta(hours=random.randint(1, 72))).isoformat(),
                'flair': random.choice([None, "Discussion", "Tutorial", "News", "Question"]),
                'is_self': True,
                'is_video': False,
                'domain': 'self.' + subreddit_name,
                'source': 'Reddit (Simulated)',
                'platform': 'reddit',
                'scraped_at': datetime.now().isoformat(),
                'post_type': 'self_post',
                'top_comments': []
            }
            posts.append(post)

        return posts

    def search_stackoverflow(self, keyword: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search StackOverflow for questions containing keywords"""
        logger.info(f"Searching StackOverflow for: {keyword}")
        questions = []

        try:
            # StackOverflow API endpoint
            api_url = "https://api.stackexchange.com/2.3/search"
            params = {
                'order': 'desc',
                'sort': 'activity',
                'intitle': keyword,
                'site': 'stackoverflow',
                'pagesize': limit
            }

            response = self.session.get(api_url, params=params, timeout=settings.TIMEOUT)

            if response.status_code == 200:
                data = response.json()

                for item in data.get('items', []):
                    question = {
                        'id': str(item.get('question_id', '')),
                        'title': item.get('title', ''),
                        'content': BeautifulSoup(item.get('body', ''), 'html.parser').get_text()[:1000],
                        'url': item.get('link', ''),
                        'author': item.get('owner', {}).get('display_name', 'Anonymous'),
                        'author_reputation': item.get('owner', {}).get('reputation', 0),
                        'score': item.get('score', 0),
                        'view_count': item.get('view_count', 0),
                        'answer_count': item.get('answer_count', 0),
                        'comment_count': item.get('comment_count', 0),
                        'created_at': datetime.fromtimestamp(item.get('creation_date', 0)).isoformat(),
                        'last_activity': datetime.fromtimestamp(item.get('last_activity_date', 0)).isoformat(),
                        'tags': item.get('tags', []),
                        'is_answered': item.get('is_answered', False),
                        'accepted_answer_id': item.get('accepted_answer_id'),
                        'source': 'StackOverflow',
                        'platform': 'stackoverflow',
                        'keyword_searched': keyword,
                        'scraped_at': datetime.now().isoformat()
                    }
                    questions.append(question)

                logger.info(f"Collected {len(questions)} questions from StackOverflow")

            else:
                logger.warning(f"StackOverflow API returned status {response.status_code}")
                questions = self._simulate_stackoverflow_data(keyword, min(limit, 5))

        except Exception as e:
            logger.error(f"Error searching StackOverflow: {e}")
            questions = self._simulate_stackoverflow_data(keyword, min(limit, 5))

        return questions

    def _simulate_stackoverflow_data(self, keyword: str, limit: int) -> List[Dict[str, Any]]:
        """Simulate StackOverflow data for testing"""
        logger.info(f"Simulating StackOverflow data for: {keyword}")

        question_templates = [
            f"How to implement {keyword} in Python?",
            f"Best practices for {keyword} development",
            f"Error when using {keyword} - need help debugging",
            f"Performance optimization for {keyword} applications",
            f"{keyword} vs alternatives - which to choose?",
            f"Beginner question about {keyword}",
            f"Advanced {keyword} techniques and patterns",
            f"Integration issues with {keyword} library",
            f"Testing strategies for {keyword} projects",
            f"Security considerations for {keyword}"
        ]

        questions = []
        for i in range(min(limit, len(question_templates))):
            question = {
                'id': f"so_{keyword}_{i}_{int(time.time())}",
                'title': question_templates[i % len(question_templates)],
                'content': f"I'm working on a project involving {keyword} and need some guidance. "
                           f"Has anyone encountered similar challenges? Any advice would be appreciated.",
                'url': f"https://stackoverflow.com/questions/{random.randint(60000000, 70000000)}",
                'author': f"developer_{random.randint(1000, 9999)}",
                'author_reputation': random.randint(1, 50000),
                'score': random.randint(-5, 100),
                'view_count': random.randint(10, 10000),
                'answer_count': random.randint(0, 15),
                'comment_count': random.randint(0, 10),
                'created_at': (datetime.now() - timedelta(days=random.randint(1, 365))).isoformat(),
                'last_activity': (datetime.now() - timedelta(hours=random.randint(1, 24))).isoformat(),
                'tags': [keyword.lower(), 'programming', random.choice(['python', 'javascript', 'java'])],
                'is_answered': random.choice([True, False]),
                'accepted_answer_id': random.randint(1000000, 2000000) if random.choice([True, False]) else None,
                'source': 'StackOverflow (Simulated)',
                'platform': 'stackoverflow',
                'keyword_searched': keyword,
                'scraped_at': datetime.now().isoformat()
            }
            questions.append(question)

        return questions

    def scrape_hackernews(self, limit: int = 30) -> List[Dict[str, Any]]:
        """Scrape Hacker News front page"""
        logger.info("Scraping Hacker News...")
        stories = []

        try:
            url = "https://hacker-news.firebaseio.com/v0/topstories.json"
            response = self.session.get(url, timeout=settings.TIMEOUT)

            if response.status_code == 200:
                story_ids = response.json()[:limit]

                for story_id in story_ids:
                    try:
                        story_url = f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
                        story_response = self.session.get(story_url, timeout=settings.TIMEOUT)

                        if story_response.status_code == 200:
                            story_data = story_response.json()

                            story = {
                                'id': str(story_data.get('id', '')),
                                'title': story_data.get('title', ''),
                                'content': story_data.get('text', ''),
                                'url': story_data.get('url', ''),
                                'hn_url': f"https://news.ycombinator.com/item?id={story_id}",
                                'author': story_data.get('by', 'Anonymous'),
                                'score': story_data.get('score', 0),
                                'comment_count': story_data.get('descendants', 0),
                                'created_at': datetime.fromtimestamp(story_data.get('time', 0)).isoformat(),
                                'story_type': story_data.get('type', 'story'),
                                'source': 'Hacker News',
                                'platform': 'hackernews',
                                'scraped_at': datetime.now().isoformat()
                            }
                            stories.append(story)

                        # Rate limiting
                        time.sleep(0.1)

                    except Exception as e:
                        logger.warning(f"Error fetching HN story {story_id}: {e}")
                        continue

                logger.info(f"Collected {len(stories)} stories from Hacker News")

        except Exception as e:
            logger.error(f"Error scraping Hacker News: {e}")
            stories = self._simulate_hackernews_data(limit)

        return stories

    def _simulate_hackernews_data(self, limit: int) -> List[Dict[str, Any]]:
        """Simulate Hacker News data for testing"""
        logger.info("Simulating Hacker News data...")

        story_titles = [
            "New breakthrough in artificial intelligence research",
            "Open source project that's changing software development",
            "Startup raises $50M Series A for innovative tech solution",
            "The future of remote work in tech companies",
            "Security vulnerability discovered in popular framework",
            "Machine learning model achieves human-level performance",
            "Developer tools that boost productivity by 10x",
            "Cloud computing trends for the next decade",
            "Cryptocurrency adoption in mainstream finance",
            "Quantum computing milestone reached by researchers"
        ]

        stories = []
        for i in range(min(limit, len(story_titles) * 3)):
            title_index = i % len(story_titles)
            story = {
                'id': f"hn_{i}_{int(time.time())}",
                'title': story_titles[title_index],
                'content': f"Discussion about {story_titles[title_index].lower()}. "
                           f"Community members share their insights and experiences.",
                'url': f"https://example.com/article/{i}",
                'hn_url': f"https://news.ycombinator.com/item?id={30000000 + i}",
                'author': f"hn_user_{random.randint(1000, 9999)}",
                'score': random.randint(1, 500),
                'comment_count': random.randint(0, 200),
                'created_at': (datetime.now() - timedelta(hours=random.randint(1, 48))).isoformat(),
                'story_type': 'story',
                'source': 'Hacker News (Simulated)',
                'platform': 'hackernews',
                'scraped_at': datetime.now().isoformat()
            }
            stories.append(story)

        return stories

    def scrape_quora_topics(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """Scrape Quora questions and answers (limited due to anti-scraping measures)"""
        logger.info(f"Searching Quora for topics: {keywords}")

        # Quora has strong anti-scraping measures, so we'll simulate data
        questions = []

        question_templates = [
            "What is the best way to learn {keyword}?",
            "How does {keyword} compare to other technologies?",
            "What are the career prospects in {keyword}?",
            "What are some common mistakes when working with {keyword}?",
            "Can someone explain {keyword} in simple terms?",
            "What resources do you recommend for {keyword}?",
            "How has {keyword} changed the industry?",
            "What are the advantages and disadvantages of {keyword}?",
            "Is {keyword} worth learning in 2024?",
            "What skills are needed to work with {keyword}?"
        ]

        for keyword in keywords:
            for i, template in enumerate(question_templates[:5]):
                question_title = template.format(keyword=keyword)

                question = {
                    'id': f"quora_{keyword}_{i}_{int(time.time())}",
                    'title': question_title,
                    'content': f"I'm interested in learning more about {keyword} and would appreciate "
                               f"insights from the community. Any advice or resources would be helpful.",
                    'url': f"https://quora.com/question/{keyword.lower()}-{i}",
                    'author': f"QuoraUser{random.randint(1000, 9999)}",
                    'upvotes': random.randint(0, 500),
                    'views': random.randint(100, 50000),
                    'answer_count': random.randint(1, 25),
                    'follower_count': random.randint(5, 1000),
                    'created_at': (datetime.now() - timedelta(days=random.randint(1, 180))).isoformat(),
                    'topics': [keyword, 'Technology', 'Learning'],
                    'source': 'Quora (Simulated)',
                    'platform': 'quora',
                    'keyword_searched': keyword,
                    'scraped_at': datetime.now().isoformat(),
                    'question_type': 'advice_seeking'
                }
                questions.append(question)

        logger.info(f"Collected {len(questions)} Quora questions (simulated)")
        return questions

    async def scrape_forums_async(self, subreddits: List[str], keywords: List[str]) -> List[Dict[str, Any]]:
        """Asynchronous forum scraping"""
        logger.info("Starting async forum scraping...")

        all_posts = []

        # Create tasks
        tasks = []

        # Reddit tasks
        for subreddit in subreddits:
            task = asyncio.create_task(self._async_reddit_scrape(subreddit))
            tasks.append(task)

        # StackOverflow tasks
        for keyword in keywords:
            task = asyncio.create_task(self._async_stackoverflow_search(keyword))
            tasks.append(task)

        # Execute tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, list):
                all_posts.extend(result)
            elif isinstance(result, Exception):
                logger.warning(f"Async forum task failed: {result}")

        # Add Hacker News and Quora data (synchronous)
        try:
            hn_stories = self.scrape_hackernews(20)
            all_posts.extend(hn_stories)
        except Exception as e:
            logger.error(f"Hacker News scraping failed: {e}")

        try:
            quora_questions = self.scrape_quora_topics(keywords)
            all_posts.extend(quora_questions)
        except Exception as e:
            logger.error(f"Quora scraping failed: {e}")

        logger.info(f"Async forum scraping completed: {len(all_posts)} posts")
        return all_posts

    async def _async_reddit_scrape(self, subreddit: str) -> List[Dict[str, Any]]:
        """Async wrapper for Reddit scraping"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.scrape_reddit, subreddit, 20)

    async def _async_stackoverflow_search(self, keyword: str) -> List[Dict[str, Any]]:
        """Async wrapper for StackOverflow search"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.search_stackoverflow, keyword, 15)

    def run_full_scraping(self, subreddits: List[str], keywords: List[str]) -> List[Dict[str, Any]]:
        """Run complete forum scraping pipeline"""
        logger.info("Starting full forum scraping pipeline...")

        all_posts = []

        # 1. Scrape Reddit subreddits
        for subreddit in subreddits:
            try:
                posts = self.scrape_reddit(subreddit, limit=25)
                all_posts.extend(posts)

                # Respectful delay between subreddits
                time.sleep(settings.REQUEST_DELAY + random.uniform(0.5, 1.0))

            except Exception as e:
                logger.error(f"Error scraping r/{subreddit}: {e}")

        # 2. Search StackOverflow for keywords
        for keyword in keywords:
            try:
                questions = self.search_stackoverflow(keyword, limit=20)
                all_posts.extend(questions)

                # API rate limiting
                time.sleep(settings.REQUEST_DELAY)

            except Exception as e:
                logger.error(f"Error searching StackOverflow for {keyword}: {e}")

        # 3. Scrape Hacker News
        try:
            hn_stories = self.scrape_hackernews(30)
            all_posts.extend(hn_stories)
        except Exception as e:
            logger.error(f"Hacker News scraping failed: {e}")

        # 4. Scrape Quora topics
        try:
            quora_questions = self.scrape_quora_topics(keywords)
            all_posts.extend(quora_questions)
        except Exception as e:
            logger.error(f"Quora scraping failed: {e}")

        logger.info(f"Forum scraping completed: {len(all_posts)} total posts")

        # Clean and enrich the data
        cleaned_posts = cleaner.enrich_data(all_posts)

        return cleaned_posts


def main():
    """Test the forum scraper independently"""
    scraper = ForumScraper()

    print("💬 Testing Forum Scraper")
    print("=" * 40)

    test_subreddits = ["artificial", "MachineLearning", "programming"]
    test_keywords = ["AI", "Python", "data science"]

    posts = scraper.run_full_scraping(test_subreddits, test_keywords)

    print(f"\n📊 Results: {len(posts)} posts collected")

    # Show breakdown by source
    sources = {}
    for post in posts:
        source = post.get('source', 'Unknown')
        sources[source] = sources.get(source, 0) + 1

    print(f"\n📈 Breakdown by source:")
    for source, count in sources.items():
        print(f"  {source}: {count} posts")

    if posts:
        print(f"\n📄 Sample post:")
        sample = posts[0]
        for key, value in sample.items():
            if key not in ['top_comments']:  # Skip complex nested data
                print(f"  {key}: {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}")


if __name__ == "__main__":
    main()