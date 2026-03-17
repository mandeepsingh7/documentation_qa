'''
Item definitions for the CUDA Documentation scraper.
Each item represents a scraped documentaton page.
'''

import scrapy

class WebItem(scrapy.Item):
    '''Structured representation of a documentation page.'''
    url = scrapy.Field()
    title = scrapy.Field()

    # Heading hierarchy extracted from the page 
    h1 = scrapy.Field()
    h2 = scrapy.Field()
    h3 = scrapy.Field()
    h4 = scrapy.Field()
    h5 = scrapy.Field()
    h6 = scrapy.Field()

    # Full cleaned text content of the page
    content = scrapy.Field()

class CudaScraperItem(scrapy.Item):
    '''Placeholder for additional CUDA scraper items.'''
    pass
