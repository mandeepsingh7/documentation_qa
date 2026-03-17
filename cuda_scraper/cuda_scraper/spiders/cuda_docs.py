# This is old version. It scrapes the entire NVIDIA CUDA Docs

from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from ..items import WebItem

class CudaDocsSpider(CrawlSpider):
    name = "cuda_docs"
    allowed_domains = ["docs.nvidia.com",
                       "nvidia.github.io",
                       "nvlabs.github.io",
                       "developer.nvidia.com"]
    start_urls = ["https://docs.nvidia.com/cuda/"]

    rules = (
        Rule(LinkExtractor(allow = (r"/cuda/.*",),
                           deny = ("genindex", "search", "_static", ".pdf", ".png", ".jpg", ".jpeg", ".svg")),
             callback='parse_item',
             follow=True),
    )

    custom_settings = {
        'DEPTH_LIMIT': 5,
    }

    def parse_item(self, response):
        web_item = WebItem()
        web_item['url'] = response.url
        web_item['title'] = response.css('title::text').get()
        web_item['h1'] = response.css('h1::text').getall()
        web_item['h2'] = response.css('h2::text').getall()
        web_item['h3'] = response.css('h3::text').getall()
        web_item['h4'] = response.css('h4::text').getall()
        web_item['h5'] = response.css('h5::text').getall()
        web_item['h6'] = response.css('h6::text').getall()

        content = response.css("section ::text, p ::text, li ::text, dt ::text, dd ::text, pre ::text").getall()
        cleaned_content = " ".join(t.strip() for t in content if t.strip())
        web_item['content'] = cleaned_content
        yield web_item
