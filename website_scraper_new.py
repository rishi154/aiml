import asyncio
from playwright.async_api import async_playwright
from urllib.parse import urljoin, urlparse, urldefrag
from langchain.schema import Document


async def scrape_body(page, url):
    """Fast scraping: Removes unwanted elements, avoids unnecessary waits, and extracts body content efficiently."""
    await page.goto(url, wait_until="domcontentloaded", timeout=60000)  # Faster load

    try:
        body_handle = await page.evaluate_handle("""
            () => {
                // Remove unnecessary elements before extracting text
                const removeElements = (selectors) => {
                    selectors.forEach(selector => {
                        document.querySelectorAll(selector).forEach(el => el.remove());
                    });
                };
                removeElements(["style", "script", "[class^='navbar']", "main > div > div > ul"]);

                return document.body.innerText.trim();
            }
        """)
        body_content = await body_handle.json_value()  # Faster than returning full innerText
        return body_content if body_content else None
    except Exception as e:
        print(f"❌ Error extracting body from {url}: {e}")
        return None


async def get_links(page):
    """Extract all unique links, removing duplicates and fragments (#section)."""
    raw_links = await page.eval_on_selector_all("a", "elements => elements.map(el => el.href)")
    return list(set(urldefrag(link)[0] for link in raw_links))  # Remove duplicates & fragments


async def scrape_website(start_url, max_depth=3, concurrency=5):
    visited = set()
    queue = [(start_url, 0)]
    documents = []
    success_count = 0  # Track successful scrapes
    fail_count = 0  # Track failed scrapes

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()

        # Block unnecessary resources (images, styles, fonts)
        await context.route("**/*",
                            lambda route: route.abort() if route.request.resource_type in ["image", "stylesheet",
                                                                                           "font"] else route.continue_())

        async def process_page(url, depth):
            """Process a single page asynchronously."""
            nonlocal success_count, fail_count

            if depth > max_depth or url in visited:
                return
            visited.add(url)

            page = await context.new_page()
            try:
                content = await scrape_body(page, url)
                if content:
                    documents.append(Document(page_content=content, metadata={"url": url, "depth": depth}))
                    success_count += 1
                    print(f"✅ Successfully scraped: {url}")
                else:
                    fail_count += 1
                    print(f"❌ Failed to scrape: {url}")

                if depth < max_depth:
                    links = await get_links(page)
                    queue.extend((urljoin(url, link), depth + 1) for link in links if
                                 urlparse(link).netloc == urlparse(start_url).netloc and link not in visited)

            except Exception as e:
                fail_count += 1
                print(f"❌ Error scraping {url}: {e}")
            finally:
                await page.close()

        while queue:
            tasks = [process_page(url, depth) for url, depth in
                     queue[:concurrency]]  # Process `concurrency` pages at a time
            queue = queue[concurrency:]
            await asyncio.gather(*tasks)

        await browser.close()

    # Print final results
    print("\nScraping Summary:")
    print(f"✅ Successfully scraped pages: {success_count}")
    print(f"❌ Failed to scrape pages: {fail_count}")

    return documents

# # Run the scraper
# start_url = "https://help.globalpaymentsintegrated.com/merchantportal/accounts/"
# documents = asyncio.run(scrape_website(start_url))
# print("Number of documents  ", len(documents))
