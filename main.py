"""
AI Community Manager - Main Application Entry Point
"""
import asyncio
import schedule
import time
import argparse
from datetime import datetime
from typing import List, Dict, Any
from loguru import logger

from config.settings import Settings
from ai_agent.orchestrator import CommunityOrchestrator
from utils.storage import DataStorage

# Configure logging
logger.add("logs/app_{time}.log", rotation="1 day", retention="30 days")


class CommunityManagerApp:
    def __init__(self):
        self.settings = Settings()
        self.orchestrator = CommunityOrchestrator()
        self.storage = DataStorage()

    async def run_analysis_cycle(self):
        """Run a complete analysis cycle"""
        try:
            logger.info("Starting analysis cycle...")

            # Run the orchestrator
            results = await self.orchestrator.run_analysis_cycle()

            if 'error' not in results:
                logger.info("Analysis cycle completed successfully")
                self._print_summary(results)
                return True
            else:
                logger.error(f"Analysis cycle failed: {results.get('error')}")
                return False

        except Exception as e:
            logger.error(f"Error in analysis cycle: {e}")
            return False

    def _print_summary(self, results: Dict[str, Any]):
        """Print a summary of the analysis results"""
        summary = results.get('summary', {})

        print("\n" + "=" * 50)
        print("AI COMMUNITY MANAGER - ANALYSIS SUMMARY")
        print("=" * 50)
        print(f"Cycle ID: {results.get('cycle_id', 'Unknown')}")
        print(f"Duration: {results.get('start_time', '')} - {results.get('end_time', '')}")
        print(f"Total Items Collected: {summary.get('total_items_collected', 0)}")
        print(f"Total Insights Generated: {summary.get('total_insights', 0)}")
        print(f"Total Recommendations: {summary.get('total_recommendations', 0)}")

        # Show top recommendations
        recommendations = results.get('recommendations', [])[:5]
        if recommendations:
            print("\nTOP RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec.get('title', 'No title')} (Priority: {rec.get('priority', 0)})")

        # Show insights
        insights = results.get('analysis', {}).get('insights', [])[:3]
        if insights:
            print("\nKEY INSIGHTS:")
            for i, insight in enumerate(insights, 1):
                print(f"{i}. {insight.get('title', 'No title')}")

        print("=" * 50 + "\n")

    def schedule_jobs(self):
        """Schedule periodic jobs"""
        interval_hours = self.settings.SCRAPING_INTERVAL_HOURS
        schedule.every(interval_hours).hours.do(lambda: asyncio.run(self.run_analysis_cycle()))
        logger.info(f"Scheduled analysis cycles every {interval_hours} hours")

    async def run_once(self):
        """Run analysis once for testing"""
        return await self.run_analysis_cycle()

    def start_scheduler(self):
        """Start the scheduled jobs"""
        self.schedule_jobs()
        logger.info("Starting scheduler...")

        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

    async def interactive_mode(self):
        """Run in interactive mode"""
        print("AI Community Manager - Interactive Mode")
        print("Commands: run, status, stats, help, quit")

        while True:
            command = input("\n> ").strip().lower()

            if command == 'run':
                print("Running analysis cycle...")
                await self.run_analysis_cycle()

            elif command == 'status':
                stats = self.storage.get_data_statistics()
                print(f"Database Statistics:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")

            elif command == 'stats':
                stats = self.storage.get_data_statistics()
                print("Detailed Statistics:")
                for key, value in stats.items():
                    print(f"{key}: {value}")

            elif command == 'help':
                print("Available commands:")
                print("  run - Run a single analysis cycle")
                print("  status - Show database status")
                print("  stats - Show detailed statistics")
                print("  help - Show this help message")
                print("  quit - Exit the application")

            elif command in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            else:
                print("Unknown command. Type 'help' for available commands.")


def main():
    parser = argparse.ArgumentParser(description="AI Community Manager")
    parser.add_argument('--mode', choices=['once', 'schedule', 'interactive'],
                        default='interactive', help='Execution mode')

    args = parser.parse_args()
    app = CommunityManagerApp()

    if args.mode == 'once':
        # Run once and exit
        asyncio.run(app.run_once())

    elif args.mode == 'schedule':
        # Run scheduled jobs
        app.start_scheduler()

    elif args.mode == 'interactive':
        # Interactive mode
        asyncio.run(app.interactive_mode())


if __name__ == "__main__":
    main()
