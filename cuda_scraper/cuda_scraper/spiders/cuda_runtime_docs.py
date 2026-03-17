from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from ..items import WebItem

class CudaRuntimeDocsSpider(CrawlSpider):
    ''' Scrapy Spider to crawl CUDA Runtime API documentation '''
    name = "cuda_runtime_docs"

    allowed_domains = ["docs.nvidia.com"]
    start_urls = ["https://docs.nvidia.com/cuda/cuda-runtime-api/"]

    rules = (
        # Follow links within CUDA Runtime API documentation while avoiding non-content pages
        Rule(LinkExtractor(allow = (r"/cuda/cuda-runtime-api/.*",),
                           deny = ("genindex", "search", "_static", ".pdf", ".png", ".jpg", ".jpeg", ".svg")),
             callback='parse_item',
             follow=True),
    )

    custom_settings = {
        # Specifying depth limit to avoid crawling too deep into unrelated pages
        'DEPTH_LIMIT': 5,
    }

    def parse_item(self, response):
        '''Extract title, headings, and main content article from documentation page.'''
        item = WebItem()

        item["url"] = response.url
        item["title"] = response.css("title::text").get()
        
        # Extract section heading for additional documentation metadata (h2 actually contains the title of the main article)
        item["h2"] = response.css("article#contents h2::text").getall()
        item["h3"] = response.css("article#contents h3::text").getall()

        # Collect all relevant text
        content = response.xpath('//article[@id="contents"]//text()').getall()

        # Clean and concatenate text
        cleaned = " ".join(
            t.strip() for t in content if t.strip()
        )

        item["content"] = cleaned

        yield item
