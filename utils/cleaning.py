"""
Data cleaning and preprocessing utilities
"""
import re
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from textblob import TextBlob
import unicodedata
from loguru import logger
from datetime import datetime
import hashlib
from difflib import SequenceMatcher


class DataCleaner:
    def __init__(self, similarity_threshold: float = 0.8):
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.mention_pattern = re.compile(r'@[A-Za-z0-9_]+')
        self.hashtag_pattern = re.compile(r'#[A-Za-z0-9_]+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}')
        self.similarity_threshold = similarity_threshold

        # HTML and special character patterns
        self.html_pattern = re.compile(r'<[^>]+>')
        self.excessive_punctuation = re.compile(r'([.!?]){3,}')
        self.excessive_caps = re.compile(r'([A-Z]){4,}')

        # Common stop words for better duplicate detection
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being'
        }

    def clean_text(self, text: str, preserve_entities: bool = True) -> str:
        """Enhanced text cleaning with more options"""
        if not text or not isinstance(text, str):
            return ""

        # Store entities before cleaning if preserving
        entities = {}
        if preserve_entities:
            entities = self.extract_entities(text)

        # Remove HTML tags
        text = self.html_pattern.sub(' ', text)

        # Normalize unicode
        text = unicodedata.normalize('NFKD', text)

        # Fix excessive punctuation and caps
        text = self.excessive_punctuation.sub(r'\1\1\1', text)  # Max 3 repeated punctuation
        text = self.excessive_caps.sub(lambda m: m.group(0)[:2] + m.group(0)[2:].lower(), text)

        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text).strip()

        # Remove URLs if not preserving entities
        if not preserve_entities:
            text = self.url_pattern.sub('', text)

        # Remove special characters but keep basic punctuation and entity markers
        if preserve_entities:
            text = re.sub(r'[^\w\s.,!?@#\-:\/]', '', text)
        else:
            text = re.sub(r'[^\w\s.,!?\-]', '', text)

        return text.strip()

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Enhanced entity extraction with emails and phone numbers"""
        entities = {
            'mentions': self.mention_pattern.findall(text),
            'hashtags': self.hashtag_pattern.findall(text),
            'urls': self.url_pattern.findall(text),
            'emails': self.email_pattern.findall(text),
            'phones': self.phone_pattern.findall(text)
        }
        return entities

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using sequence matcher"""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def create_content_hash(self, text: str) -> str:
        """Create a hash for content deduplication"""
        # Normalize text for hashing
        normalized = self.clean_text(text, preserve_entities=False).lower()
        # Remove stop words for better matching
        words = [word for word in normalized.split() if word not in self.stop_words]
        normalized = ' '.join(words)
        return hashlib.md5(normalized.encode()).hexdigest()

    def remove_duplicates(self, data: List[Dict[str, Any]],
                          key_field: str = 'content',
                          use_similarity: bool = True) -> List[Dict[str, Any]]:
        """Enhanced duplicate removal with similarity checking"""
        if not data:
            return data

        unique_data = []
        seen_hashes = set()
        seen_content = []

        for item in data:
            content = item.get(key_field, '')
            if not content:
                continue

            # Fast exact duplicate check
            content_hash = self.create_content_hash(content)
            if content_hash in seen_hashes:
                continue

            # Similarity check for near-duplicates
            is_duplicate = False
            if use_similarity:
                for existing_content in seen_content:
                    if self.calculate_similarity(content, existing_content) > self.similarity_threshold:
                        is_duplicate = True
                        break

            if not is_duplicate:
                seen_hashes.add(content_hash)
                seen_content.append(content)
                unique_data.append(item)

        logger.info(f"Removed {len(data) - len(unique_data)} duplicates from {len(data)} items")
        return unique_data

    def validate_article(self, article: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate article quality and return issues found"""
        issues = []

        # Check required fields
        required_fields = ['title', 'content']
        for field in required_fields:
            if not article.get(field):
                issues.append(f"Missing {field}")

        # Check content quality
        content = article.get('content', '')
        if content:
            if len(content.split()) < 10:
                issues.append("Content too short (< 10 words)")
            if len(content) > 10000:
                issues.append("Content too long (> 10000 chars)")
            if content.count('.') < 2:
                issues.append("Content lacks proper sentences")

        # Check title quality
        title = article.get('title', '')
        if title:
            if len(title.split()) < 3:
                issues.append("Title too short")
            if len(title) > 200:
                issues.append("Title too long")

        # Check URL validity
        url = article.get('url', '')
        if url and not url.startswith(('http://', 'https://')):
            issues.append("Invalid URL format")

        return len(issues) == 0, issues

    def classify_sentiment(self, polarity: float) -> str:
        """Classify sentiment based on polarity score"""
        if polarity > 0.1:
            return 'positive'
        elif polarity < -0.1:
            return 'negative'
        else:
            return 'neutral'

    def calculate_readability(self, text: str) -> float:
        """Simple readability score based on sentence and word complexity"""
        if not text:
            return 0.0

        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if not sentences:
            return 0.0

        words = text.split()
        if not words:
            return 0.0

        # Simple metrics
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)

        # Simple readability score (lower is more readable)
        readability = (avg_sentence_length * 0.6) + (avg_word_length * 0.4)

        # Normalize to 0-100 scale (inverted so higher is more readable)
        return max(0, 100 - min(100, readability * 5))

    def enrich_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhanced data enrichment with quality validation"""
        enriched_data = []
        invalid_count = 0

        for item in data:
            # Validate article first
            is_valid, issues = self.validate_article(item)
            if not is_valid:
                invalid_count += 1
                logger.debug(f"Invalid article: {issues}")
                continue

            enriched_item = item.copy()
            content = item.get('content', '')
            title = item.get('title', '')

            # Clean the text content
            enriched_item['content'] = self.clean_text(content)
            enriched_item['title'] = self.clean_text(title)

            if content:
                # Add text statistics
                words = content.split()
                enriched_item['word_count'] = len(words)
                enriched_item['char_count'] = len(content)
                enriched_item['sentence_count'] = len([s for s in content.split('.') if s.strip()])
                enriched_item['avg_word_length'] = sum(len(word) for word in words) / len(words) if words else 0

                # Extract entities
                entities = self.extract_entities(content)
                enriched_item.update(entities)

                # Content quality metrics
                enriched_item['has_entities'] = any(entities.values())
                enriched_item['readability_score'] = self.calculate_readability(content)

                # Enhanced sentiment analysis
                try:
                    blob = TextBlob(content)
                    sentiment = blob.sentiment
                    enriched_item['sentiment_polarity'] = sentiment.polarity
                    enriched_item['sentiment_subjectivity'] = sentiment.subjectivity
                    enriched_item['sentiment_label'] = self.classify_sentiment(sentiment.polarity)
                except Exception as e:
                    logger.debug(f"Sentiment analysis failed: {e}")
                    enriched_item['sentiment_polarity'] = 0.0
                    enriched_item['sentiment_subjectivity'] = 0.0
                    enriched_item['sentiment_label'] = 'neutral'

                # Language detection with confidence
                try:
                    blob = TextBlob(content)
                    enriched_item['language'] = blob.detect_language()
                    # Simple confidence based on text length and character patterns
                    enriched_item['language_confidence'] = min(1.0, len(content) / 200)
                except Exception:
                    enriched_item['language'] = 'unknown'
                    enriched_item['language_confidence'] = 0.0

            # Add processing metadata
            enriched_item['processed_at'] = datetime.now().isoformat()
            enriched_item['content_hash'] = self.create_content_hash(content)

            enriched_data.append(enriched_item)

        logger.info(f"Enriched {len(enriched_data)} articles, rejected {invalid_count} invalid articles")
        return enriched_data

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced DataFrame cleaning with validation"""
        logger.info(f"Cleaning DataFrame with {len(df)} rows")

        # Remove rows where essential content is empty
        initial_count = len(df)
        df = df.dropna(subset=['content', 'title'])
        logger.info(f"Removed {initial_count - len(df)} rows with missing content/title")

        # Clean text columns
        text_columns = ['content', 'title', 'description', 'author']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: self.clean_text(str(x)) if pd.notna(x) else '')

        # Remove empty content after cleaning
        df = df[df['content'].str.len() > 10]

        # Remove duplicates based on content similarity
        df = df.drop_duplicates(subset=['content'], keep='first')

        # Add quality metrics
        if 'content' in df.columns:
            df['word_count'] = df['content'].apply(lambda x: len(str(x).split()))
            df['readability_score'] = df['content'].apply(self.calculate_readability)

        logger.info(f"Final DataFrame has {len(df)} rows")
        return df


# Create a global cleaner instance
cleaner = DataCleaner()


# Alternative: You can also create a function to get cleaner
def get_cleaner(similarity_threshold: float = 0.8) -> DataCleaner:
    """Get data cleaner instance"""
    return DataCleaner(similarity_threshold=similarity_threshold)


# For backward compatibility, expose commonly used functions
def clean_text(text: str) -> str:
    """Quick access to text cleaning"""
    return cleaner.clean_text(text)


def remove_duplicates(data: List[Dict[str, Any]], key_field: str = 'content') -> List[Dict[str, Any]]:
    """Quick access to duplicate removal"""
    return cleaner.remove_duplicates(data, key_field)


def enrich_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Quick access to data enrichment"""
    return cleaner.enrich_data(data)