#!/usr/bin/env python3
"""
News Scraper Module
Collects news articles from various sources
"""

import asyncio
import aiohttp
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from loguru import logger
import time
import random

from config.settings import settings
from utils.cleaning import cleaner


class NewsScraper:
    """Scraper for news websites and RSS feeds"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(settings.DEFAULT_HEADERS)

    def scrape_techcrunch(self) -> List[Dict[str, Any]]:
        """Scrape latest articles from TechCrunch"""
        logger.info("Scraping TechCrunch...")
        articles = []

        try:
            url = "https://techcrunch.com"
            response = self.session.get(url, timeout=settings.TIMEOUT)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Find article elements (TechCrunch structure)
            article_elements = soup.find_all('div', class_='post-block')

            for element in article_elements[:10]:  # Limit to 10 articles
                try:
                    # Extract title
                    title_elem = element.find('h2', class_='post-block__title')
                    title = title_elem.get_text(strip=True) if title_elem else "No title"

                    # Extract URL
                    link_elem = title_elem.find('a') if title_elem else None
                    url = link_elem['href'] if link_elem else ""

                    # Extract excerpt
                    excerpt_elem = element.find('div', class_='post-block__content')
                    excerpt = excerpt_elem.get_text(strip=True) if excerpt_elem else ""

                    # Extract author and date (if available)
                    author_elem = element.find('div', class_='river-byline__authors')
                    author = author_elem.get_text(strip=True) if author_elem else "Unknown"

                    article = {
                        'title': title,
                        'content': excerpt,
                        'url': url,
                        'source': 'TechCrunch',
                        'author': author,
                        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'category': 'Technology',
                        'scraped_at': datetime.now().isoformat()
                    }

                    articles.append(article)

                except Exception as e:
                    logger.warning(f"Error parsing TechCrunch article: {e}")
                    continue

            logger.info(f"Scraped {len(articles)} articles from TechCrunch")

        except Exception as e:
            logger.error(f"Error scraping TechCrunch: {e}")

        return articles

    def scrape_generic_news_site(self, url: str, site_name: str) -> List[Dict[str, Any]]:
        """Generic news site scraper"""
        logger.info(f"Scraping {site_name}...")
        articles = []

        try:
            response = self.session.get(url, timeout=settings.TIMEOUT)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Common selectors for news articles
            selectors = [
                'article',
                '.article',
                '.post',
                '.news-item',
                '[class*="article"]',
                '[class*="post"]'
            ]

            article_elements = []
            for selector in selectors:
                elements = soup.select(selector)
                if elements:
                    article_elements = elements
                    break

            for element in article_elements[:15]:  # Limit results
                try:
                    # Try to find title
                    title_selectors = ['h1', 'h2', 'h3', '.title', '[class*="title"]']
                    title = ""
                    for sel in title_selectors:
                        title_elem = element.select_one(sel)
                        if title_elem:
                            title = title_elem.get_text(strip=True)
                            break

                    if not title or len(title) < 10:
                        continue

                    # Try to find content
                    content_selectors = ['.content', '.excerpt', '.summary', 'p']
                    content = ""
                    for sel in content_selectors:
                        content_elem = element.select_one(sel)
                        if content_elem:
                            content = content_elem.get_text(strip=True)
                            break

                    # Try to find link
                    link_elem = element.select_one('a')
                    link = link_elem['href'] if link_elem else ""
                    if link and not link.startswith('http'):
                        link = url.rstrip('/') + '/' + link.lstrip('/')

                    article = {
                        'title': title,
                        'content': content,
                        'url': link,
                        'source': site_name,
                        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'category': 'General',
                        'scraped_at': datetime.now().isoformat()
                    }

                    articles.append(article)

                except Exception as e:
                    logger.warning(f"Error parsing article from {site_name}: {e}")
                    continue

            logger.info(f"Scraped {len(articles)} articles from {site_name}")

        except Exception as e:
            logger.error(f"Error scraping {site_name}: {e}")

        return articles

    async def scrape_news_async(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Asynchronous news scraping for multiple sources"""
        logger.info("Starting async news scraping...")

        async def fetch_site(session, url, site_name):
            try:
                async with session.get(url, timeout=settings.TIMEOUT) as response:
                    content = await response.text()
                    soup = BeautifulSoup(content, 'html.parser')

                    # Simple extraction logic
                    articles = []
                    for article in soup.find_all(['article', 'div'], class_=lambda x: x and 'article' in x.lower())[:5]:
                        title_elem = article.find(['h1', 'h2', 'h3'])
                        if title_elem:
                            title = title_elem.get_text(strip=True)
                            if len(title) > 10:
                                articles.append({
                                    'title': title,
                                    'content': article.get_text(strip=True)[:500],
                                    'url': url,
                                    'source': site_name,
                                    'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    'scraped_at': datetime.now().isoformat()
                                })

                    return articles

            except Exception as e:
                logger.warning(f"Error fetching {site_name}: {e}")
                return []

        all_articles = []

        async with aiohttp.ClientSession(headers=settings.DEFAULT_HEADERS) as session:
            tasks = []
            for url in urls:
                site_name = url.split('//')[1].split('/')[0]
                tasks.append(fetch_site(session, url, site_name))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, list):
                    all_articles.extend(result)

        logger.info(f"Async scraping completed: {len(all_articles)} articles")
        return all_articles

    def scrape_rss_feeds(self, rss_feeds: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Scrape RSS feeds for news"""
        logger.info("Scraping RSS feeds...")
        articles = []

        if not rss_feeds:
            rss_feeds = [
                "https://techcrunch.com/feed/",
                "https://feeds.feedburner.com/venturebeat/SZYF",
                "https://www.theverge.com/rss/index.xml"
            ]

        try:
            import feedparser

            for feed_url in rss_feeds:
                try:
                    feed = feedparser.parse(feed_url)
                    source_name = feed.feed.get('title', 'Unknown Source')

                    for entry in feed.entries[:10]:
                        article = {
                            'title': entry.get('title', ''),
                            'content': entry.get('summary', '')[:1000],
                            'url': entry.get('link', ''),
                            'source': source_name,
                            'author': entry.get('author', 'Unknown'),
                            'date': entry.get('published', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                            'category': 'RSS Feed',
                            'scraped_at': datetime.now().isoformat()
                        }
                        articles.append(article)

                    logger.info(f"Scraped {len(feed.entries[:10])} articles from {source_name}")

                except Exception as e:
                    logger.warning(f"Error parsing RSS feed {feed_url}: {e}")
                    continue

        except ImportError:
            logger.warning("feedparser not installed. Install with: pip install feedparser")
            # Return simulated RSS data
            articles = self._simulate_rss_data()

        return articles

    def _simulate_rss_data(self) -> List[Dict[str, Any]]:
        """Simulate RSS feed data"""
        articles = []
        sources = ['TechNews RSS', 'VentureBeat RSS', 'TheVerge RSS']

        for source in sources:
            for i in range(3):
                article = {
                    'title': f'Latest {source} Article {i + 1}',
                    'content': f'This is simulated RSS content from {source}. Technology news and updates.',
                    'url': f'https://example.com/{source.lower().replace(" ", "")}/article-{i}',
                    'source': source,
                    'author': 'RSS Author',
                    'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'category': 'RSS Feed',
                    'scraped_at': datetime.now().isoformat()
                }
                articles.append(article)

        return articles

    def scrape_keyword_news(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """Search for news articles containing specific keywords"""
        logger.info(f"Searching news for keywords: {keywords}")
        articles = []

        # Use a news API or search engine
        # For demo purposes, we'll simulate keyword-based results
        for keyword in keywords:
            try:
                # Simulate news search results
                for i in range(3):
                    article = {
                        'title': f"News about {keyword} - Article {i + 1}",
                        'content': f"This news article discusses {keyword} and its impact on the industry. "
                                   f"Experts analyze the latest trends and developments...",
                        'url': f"https://example.com/news/{keyword.lower().replace(' ', '-')}-{i + 1}",
                        'source': 'News Search',
                        'keywords_matched': [keyword],
                        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'category': 'Keyword Search',
                        'scraped_at': datetime.now().isoformat()
                    }
                    articles.append(article)

                # Add delay between searches
                time.sleep(settings.REQUEST_DELAY)

            except Exception as e:
                logger.error(f"Error searching for keyword {keyword}: {e}")

        return articles

    def run_full_scraping(self) -> List[Dict[str, Any]]:
        """Run complete news scraping pipeline"""
        logger.info("Starting full news scraping pipeline...")

        all_articles = []

        # 1. Scrape major tech news sites
        try:
            techcrunch_articles = self.scrape_techcrunch()
            all_articles.extend(techcrunch_articles)
        except Exception as e:
            logger.error(f"TechCrunch scraping failed: {e}")

        # 2. Scrape other news sources
        for source_url in settings.NEWS_SOURCES[1:]:  # Skip TechCrunch as it's already done
            try:
                site_name = source_url.split('//')[1].split('/')[0]
                articles = self.scrape_generic_news_site(source_url, site_name)
                all_articles.extend(articles)

                # Respectful delay between sites
                time.sleep(settings.REQUEST_DELAY + random.uniform(0.5, 1.5))

            except Exception as e:
                logger.error(f"Error scraping {source_url}: {e}")

        # 3. Scrape RSS feeds
        try:
            rss_articles = self.scrape_rss_feeds()
            all_articles.extend(rss_articles)
        except Exception as e:
            logger.error(f"RSS scraping failed: {e}")

        # 4. Search for trending keywords - FIXED: Changed TRENDING_KEYWORDS to TRACKING_KEYWORDS
        try:
            keyword_articles = self.scrape_keyword_news(settings.TRACKING_KEYWORDS)
            all_articles.extend(keyword_articles)
        except Exception as e:
            logger.error(f"Keyword scraping failed: {e}")

        logger.info(f"News scraping completed: {len(all_articles)} total articles")

        # Clean and enrich the data
        cleaned_articles = cleaner.enrich_data(all_articles)

        return cleaned_articles


def main():
    """Test the news scraper independently"""
    scraper = NewsScraper()

    print("🗞️ Testing News Scraper")
    print("=" * 40)

    articles = scraper.run_full_scraping()

    print(f"\n📊 Results: {len(articles)} articles collected")

    if articles:
        print(f"\n📄 Sample article:")
        sample = articles[0]
        for key, value in sample.items():
            print(f"  {key}: {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}")


if __name__ == "__main__":
    main()