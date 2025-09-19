"""
Configuration settings for AI Community Manager
"""
import os
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import List, Dict, Any


load_dotenv()


@dataclass
class Settings:
    # API Keys
    TWITTER_API_KEY: str = os.getenv('TWITTER_API_KEY', '')
    TWITTER_API_SECRET: str = os.getenv('TWITTER_API_SECRET', '')
    TWITTER_ACCESS_TOKEN: str = os.getenv('TWITTER_ACCESS_TOKEN', '')
    TWITTER_ACCESS_TOKEN_SECRET: str = os.getenv('TWITTER_ACCESS_TOKEN_SECRET', '')

    REDDIT_CLIENT_ID: str = os.getenv('REDDIT_CLIENT_ID', '')
    REDDIT_CLIENT_SECRET: str = os.getenv('REDDIT_CLIENT_SECRET', '')
    REDDIT_USER_AGENT: str = os.getenv('REDDIT_USER_AGENT', 'AI_Community_Manager')

    # Database
    DATABASE_URL: str = os.getenv('DATABASE_URL', 'sqlite:///data/community_manager.db')

    # Scraping Settings
    MAX_ARTICLES_PER_SOURCE: int = int(os.getenv('MAX_ARTICLES_PER_SOURCE', '50'))
    SCRAPING_INTERVAL_HOURS: int = int(os.getenv('SCRAPING_INTERVAL_HOURS', '6'))
    MAX_WORKERS: int = int(os.getenv('MAX_WORKERS', '5'))

    # Analysis Settings
    SENTIMENT_THRESHOLD: float = float(os.getenv('SENTIMENT_THRESHOLD', '0.6'))
    TRENDING_SCORE_THRESHOLD: float = float(os.getenv('TRENDING_SCORE_THRESHOLD', '0.7'))

    # Additional settings for scrapers
    DEFAULT_HEADERS: Dict[str, str] = field(default_factory=lambda: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    })

    # Request settings
    TIMEOUT: int = 30
    REQUEST_DELAY: float = 2.0  # Seconds between requests
    MAX_RETRIES: int = 3
    BACKOFF_FACTOR: float = 0.3

    # Content limits
    MAX_CONTENT_LENGTH: int = 5000
    MIN_CONTENT_LENGTH: int = 50
    MAX_ARTICLES_PER_RUN: int = 200

    # Trending keywords for searches
    TRACKING_KEYWORDS: List[str] = field(default_factory=lambda: [
        'artificial intelligence', 'machine learning', 'deep learning',
        'data science', 'python', 'javascript', 'react', 'nodejs',
        'docker', 'kubernetes', 'aws', 'cloud computing', 'devops',
        'blockchain', 'cryptocurrency', 'cybersecurity', 'iot',
        'startup', 'tech innovation', 'digital transformation'
    ])

    TARGET_SUBREDDITS = ["r/artificial", "r/MachineLearning", "r/technology"]

    # News Sources - Fixed the missing comma
    NEWS_SOURCES: List[str] = field(default_factory=lambda: [
        'https://techcrunch.com',
        'https://feeds.feedburner.com/venturebeat/SZYF',
        'https://www.theverge.com/rss/index.xml',  # Added missing comma
        'https://www.wired.com',
        'https://arstechnica.com',
        'https://www.engadget.com',
        'https://www.zdnet.com',
        'https://techreport.com'
    ])

    # Social media platforms configuration
    SOCIAL_PLATFORMS: Dict[str, Dict] = field(default_factory=lambda: {
        'twitter': {
            'enabled': True,
            'rate_limit': 300,  # Requests per 15-minute window
            'search_operators': ['lang:en', '-is:retweet']
        },
        'facebook': {
            'enabled': False,  # Requires special permissions
            'rate_limit': 100
        },
        'linkedin': {
            'enabled': False,  # Requires API access
            'rate_limit': 100
        }
    })

    # Forum configuration
    FORUM_CONFIG: Dict[str, Dict] = field(default_factory=lambda: {
        'reddit': {
            'rate_limit': 60,  # Requests per minute
            'max_posts_per_subreddit': 25
        },
        'stackoverflow': {
            'rate_limit': 30,  # Requests per second
            'max_questions_per_search': 20
        },
        'hackernews': {
            'rate_limit': 10,  # Requests per second
            'max_stories': 30
        }
    })

    # Trend analysis settings
    TREND_ANALYSIS: Dict[str, Any] = field(default_factory=lambda: {
        'min_mentions_for_trend': 10,
        'trend_time_window_hours': 24,
        'confidence_threshold': 0.7,
        'velocity_calculation_hours': 6
    })


# Create a global settings instance
settings = Settings()

# Alternative: You can also create a function to get settings
def get_settings() -> Settings:
    """Get application settings"""
    return settings

# For backward compatibility, you might want to expose some commonly used settings
DEFAULT_HEADERS = settings.DEFAULT_HEADERS
TIMEOUT = settings.TIMEOUT
REQUEST_DELAY = settings.REQUEST_DELAY
NEWS_SOURCES = settings.NEWS_SOURCES
TRACKING_KEYWORDS = settings.TRACKING_KEYWORDS