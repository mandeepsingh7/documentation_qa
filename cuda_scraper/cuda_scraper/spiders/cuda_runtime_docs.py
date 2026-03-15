import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from ..items import WebItem

class CudaRuntimeDocsSpider(CrawlSpider):
    name = "cuda_runtime_docs"
    allowed_domains = ["docs.nvidia.com"]
    start_urls = ["https://docs.nvidia.com/cuda/cuda-runtime-api/"]

    rules = (
        Rule(LinkExtractor(allow = (r"/cuda/cuda-runtime-api/.*",),
                           deny = ("genindex", "search", "_static", ".pdf", ".png", ".jpg", ".jpeg", ".svg")),
             callback='parse_item',
             follow=True),
    )

    custom_settings = {
        'DEPTH_LIMIT': 5,
    }

    def parse_item(self, response):

        item = WebItem()

        item["url"] = response.url
        item["title"] = response.css("title::text").get()
        
        item["h2"] = response.css("article#contents h2::text").getall()
        item["h3"] = response.css("article#contents h3::text").getall()

        content = response.xpath('//article[@id="contents"]//text()').getall()

        cleaned = " ".join(
            t.strip() for t in content if t.strip()
        )

        item["content"] = cleaned

        yield item
