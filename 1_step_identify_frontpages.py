#Step 1: Identify Front Pages via XML Parsing
from lxml import etree
import os

def extract_frontpages(xml_dir):
    frontpages = []
    for xml_file in os.listdir(xml_dir):
        tree = etree.parse(os.path.join(xml_dir, xml_file))
        for df in tree.xpath('//datafield[@tag="200"]'):
            title = df.xpath('subfield[@code="a"]/text()')
            if title:
                frontpages.append({
                    'file': xml_file.replace('.xml', '.jpg'),
                    'title': title[0],
                    'desc': df.xpath('subfield[@code="e"]/text()')[0]
                })
    return frontpages