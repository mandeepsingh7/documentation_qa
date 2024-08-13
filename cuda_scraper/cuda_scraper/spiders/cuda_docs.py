import scrapy
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
        Rule(LinkExtractor(allow = ()), callback='parse_item'),
    )

    custom_settings = {
        'DEPTH_LIMIT': 5,
    }

    def parse_item(self, response):
        html = response
        web_item = WebItem()
        web_item['url'] = response.url
        web_item['title'] = html.css('head title::text').get()
        web_item['h1'] = html.css('h1::text').getall()
        web_item['h2'] = html.css('h2::text').getall()
        web_item['h3'] = html.css('h3::text').getall()
        web_item['h4'] = html.css('h4::text').getall()
        web_item['h5'] = html.css('h5::text').getall()
        web_item['h6'] = html.css('h6::text').getall()
        content = response.xpath('//body//text()').getall()
        web_item['content'] = ' '.join(content).strip()
        yield web_item
