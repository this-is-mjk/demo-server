"""
Image Scraping Script for Jaundice Detection Dataset
Scrapes face images from various sources for building training datasets.

Usage:
    python scrape_images.py --query "jaundice face" --output raw/jaundice --num 100
    python scrape_images.py --query "healthy face" --output raw/normal --num 100
"""

import os
import re
import time
import uuid
import argparse
import hashlib
import requests
from pathlib import Path
from urllib.parse import urlparse, quote_plus
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Set

# Try to import optional dependencies
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    print("Warning: BeautifulSoup not installed. Install with: pip install beautifulsoup4")

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("Warning: Selenium not installed. Install with: pip install selenium")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class ImageScraper:
    """Multi-source image scraper for building datasets."""
    
    def __init__(self, output_dir: str, min_size: int = 100):
        """
        Initialize the scraper.
        
        Args:
            output_dir: Directory to save downloaded images
            min_size: Minimum image dimension (width or height)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.min_size = min_size
        self.downloaded_hashes: Set[str] = set()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        })
    
    def _get_image_hash(self, content: bytes) -> str:
        """Get MD5 hash of image content for deduplication."""
        return hashlib.md5(content).hexdigest()
    
    def _is_valid_image(self, content: bytes) -> bool:
        """Check if content is a valid image of sufficient size."""
        if not PIL_AVAILABLE:
            return len(content) > 5000  # Basic size check
        
        try:
            from io import BytesIO
            img = Image.open(BytesIO(content))
            width, height = img.size
            return width >= self.min_size and height >= self.min_size
        except Exception:
            return False
    
    def _download_image(self, url: str, prefix: str = "img") -> Optional[str]:
        """
        Download a single image.
        
        Args:
            url: Image URL
            prefix: Filename prefix
            
        Returns:
            Path to saved image or None if failed
        """
        try:
            response = self.session.get(url, timeout=10, stream=True)
            response.raise_for_status()
            
            content = response.content
            
            # Check for duplicates
            img_hash = self._get_image_hash(content)
            if img_hash in self.downloaded_hashes:
                return None
            
            # Validate image
            if not self._is_valid_image(content):
                return None
            
            self.downloaded_hashes.add(img_hash)
            
            # Determine extension
            content_type = response.headers.get('content-type', '')
            if 'png' in content_type:
                ext = 'png'
            elif 'gif' in content_type:
                ext = 'gif'
            elif 'webp' in content_type:
                ext = 'webp'
            else:
                ext = 'jpg'
            
            # Save image
            filename = f"{prefix}_{len(self.downloaded_hashes)}_{uuid.uuid4().hex[:8]}.{ext}"
            filepath = self.output_dir / filename
            
            with open(filepath, 'wb') as f:
                f.write(content)
            
            return str(filepath)
            
        except Exception as e:
            return None
    
    def scrape_google_images(self, query: str, num_images: int = 100) -> List[str]:
        """
        Scrape images from Google Images using Selenium.
        
        Args:
            query: Search query
            num_images: Number of images to download
            
        Returns:
            List of downloaded image paths
        """
        if not SELENIUM_AVAILABLE:
            print("Selenium not available. Skipping Google Images.")
            return []
        
        print(f"Scraping Google Images for: {query}")
        
        # Setup Chrome options
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        
        try:
            driver = webdriver.Chrome(options=options)
        except Exception as e:
            print(f"Failed to start Chrome: {e}")
            print("Make sure chromedriver is installed: sudo apt install chromium-chromedriver")
            return []
        
        downloaded = []
        
        try:
            # Navigate to Google Images
            search_url = f"https://www.google.com/search?q={quote_plus(query)}&tbm=isch"
            driver.get(search_url)
            time.sleep(2)
            
            # Scroll to load more images
            last_height = driver.execute_script("return document.body.scrollHeight")
            while len(downloaded) < num_images:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1)
                
                new_height = driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    # Try clicking "Show more results"
                    try:
                        show_more = driver.find_element(By.CSS_SELECTOR, '.mye4qd')
                        show_more.click()
                        time.sleep(2)
                    except:
                        break
                last_height = new_height
                
                # Get image elements
                images = driver.find_elements(By.CSS_SELECTOR, 'img.rg_i')
                
                for img in images:
                    if len(downloaded) >= num_images:
                        break
                    
                    try:
                        # Click on thumbnail to get full image
                        img.click()
                        time.sleep(0.5)
                        
                        # Find the full-size image
                        full_img = driver.find_elements(By.CSS_SELECTOR, 'img.n3VNCb')
                        if full_img:
                            src = full_img[0].get_attribute('src')
                            if src and src.startswith('http'):
                                result = self._download_image(src, query.replace(' ', '_'))
                                if result:
                                    downloaded.append(result)
                                    print(f"Downloaded: {len(downloaded)}/{num_images}")
                    except:
                        continue
        
        finally:
            driver.quit()
        
        return downloaded
    
    def scrape_bing_images(self, query: str, num_images: int = 100) -> List[str]:
        """
        Scrape images from Bing Images (no Selenium required).
        
        Args:
            query: Search query
            num_images: Number of images to download
            
        Returns:
            List of downloaded image paths
        """
        if not BS4_AVAILABLE:
            print("BeautifulSoup not available. Skipping Bing Images.")
            return []
        
        print(f"Scraping Bing Images for: {query}")
        downloaded = []
        
        page = 0
        while len(downloaded) < num_images:
            url = f"https://www.bing.com/images/search?q={quote_plus(query)}&first={page * 35}"
            
            try:
                response = self.session.get(url)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find image URLs
                img_tags = soup.find_all('a', class_='iusc')
                
                if not img_tags:
                    break
                
                for tag in img_tags:
                    if len(downloaded) >= num_images:
                        break
                    
                    try:
                        import json
                        m = tag.get('m')
                        if m:
                            data = json.loads(m)
                            img_url = data.get('murl')
                            if img_url:
                                result = self._download_image(img_url, query.replace(' ', '_'))
                                if result:
                                    downloaded.append(result)
                                    print(f"Downloaded: {len(downloaded)}/{num_images}")
                    except:
                        continue
                
                page += 1
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error on page {page}: {e}")
                break
        
        return downloaded
    
    def scrape_duckduckgo_images(self, query: str, num_images: int = 100) -> List[str]:
        """
        Scrape images from DuckDuckGo.
        
        Args:
            query: Search query
            num_images: Number of images to download
            
        Returns:
            List of downloaded image paths
        """
        print(f"Scraping DuckDuckGo Images for: {query}")
        downloaded = []
        
        # DuckDuckGo image search API
        url = "https://duckduckgo.com/"
        
        try:
            # Get token
            response = self.session.get(url)
            token_match = re.search(r'vqd=([\d-]+)', response.text)
            if not token_match:
                print("Could not get DuckDuckGo token")
                return []
            
            token = token_match.group(1)
            
            # Search images
            search_url = f"https://duckduckgo.com/i.js?l=us-en&o=json&q={quote_plus(query)}&vqd={token}&f=,,,&p=1"
            
            response = self.session.get(search_url)
            data = response.json()
            
            results = data.get('results', [])
            
            for result in results:
                if len(downloaded) >= num_images:
                    break
                
                img_url = result.get('image')
                if img_url:
                    filepath = self._download_image(img_url, query.replace(' ', '_'))
                    if filepath:
                        downloaded.append(filepath)
                        print(f"Downloaded: {len(downloaded)}/{num_images}")
            
        except Exception as e:
            print(f"DuckDuckGo error: {e}")
        
        return downloaded
    
    def scrape_from_urls_file(self, urls_file: str, prefix: str = "img") -> List[str]:
        """
        Download images from a file containing URLs (one per line).
        
        Args:
            urls_file: Path to file with URLs
            prefix: Filename prefix
            
        Returns:
            List of downloaded image paths
        """
        print(f"Downloading from URLs file: {urls_file}")
        
        with open(urls_file, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
        
        downloaded = []
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(self._download_image, url, prefix): url for url in urls}
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    downloaded.append(result)
                    print(f"Downloaded: {len(downloaded)}/{len(urls)}")
        
        return downloaded
    
    def scrape_all_sources(self, query: str, num_images: int = 100) -> List[str]:
        """
        Scrape from all available sources.
        
        Args:
            query: Search query
            num_images: Total number of images to download
            
        Returns:
            List of downloaded image paths
        """
        all_downloaded = []
        images_per_source = num_images // 3 + 1
        
        # Try each source
        sources = [
            ('Bing', self.scrape_bing_images),
            ('DuckDuckGo', self.scrape_duckduckgo_images),
            ('Google', self.scrape_google_images),
        ]
        
        for name, scraper in sources:
            if len(all_downloaded) >= num_images:
                break
            
            remaining = num_images - len(all_downloaded)
            try:
                results = scraper(query, min(images_per_source, remaining))
                all_downloaded.extend(results)
                print(f"{name}: Downloaded {len(results)} images")
            except Exception as e:
                print(f"{name} failed: {e}")
        
        return all_downloaded[:num_images]


def main():
    parser = argparse.ArgumentParser(description='Scrape images for jaundice detection dataset')
    parser.add_argument('--query', type=str, required=True, help='Search query')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--num', type=int, default=100, help='Number of images to download')
    parser.add_argument('--source', type=str, default='all', 
                        choices=['all', 'google', 'bing', 'duckduckgo', 'urls'],
                        help='Image source')
    parser.add_argument('--urls-file', type=str, help='File with URLs (for --source urls)')
    parser.add_argument('--min-size', type=int, default=100, help='Minimum image dimension')
    
    args = parser.parse_args()
    
    scraper = ImageScraper(args.output, min_size=args.min_size)
    
    if args.source == 'all':
        results = scraper.scrape_all_sources(args.query, args.num)
    elif args.source == 'google':
        results = scraper.scrape_google_images(args.query, args.num)
    elif args.source == 'bing':
        results = scraper.scrape_bing_images(args.query, args.num)
    elif args.source == 'duckduckgo':
        results = scraper.scrape_duckduckgo_images(args.query, args.num)
    elif args.source == 'urls':
        if not args.urls_file:
            print("Error: --urls-file required when using --source urls")
            return
        results = scraper.scrape_from_urls_file(args.urls_file, args.query.replace(' ', '_'))
    
    print(f"\n✓ Downloaded {len(results)} images to {args.output}")


if __name__ == '__main__':
    main()
