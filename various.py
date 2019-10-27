
"""Various useful functions"""

#Author: John Sibony <john.sibony@hotmail.fr>

import os
import urllib.request
import re

def send_imessage(message, name):
	""" Send a message through imessage app.
	:param message: Message to send (trype string).
	:param name: Name of the contact to send the message.  
	"""
    cmd = '''osascript<<END
    tell application "Messages"
    send '''+''' "'''+message+'''" '''+''' to buddy'''+''' "'''+name+'''" '''+'''
    end tell
    END'''
    os.system(cmd)

def scrapping_wikipedia(url):
    """ Scrap a wikipedia article.
    :param url: Url of the wikipedia article
    Returns: List of sentences of the wikipedia article. 
    """
    page = urllib.request.urlopen(url)
    page = page.read().decode()
    page = page.split('\n')
    begin, end = [], []
    for ind,phrase in enumerate(page):
        page[ind] = re.sub(r'\<.*?\>', r'', phrase)
        if (re.findall(r'<p>', phrase)) : begin.append(ind)
        if (re.findall(r'</p>', phrase)) : end.append(ind)
    page = [page[ind] for ind in range(begin[0],end[-1]) if page[ind]!='']
    return page

if __name__ == '__main__':
	send_imessage('Hey Johnny :)', "John Sibony")
    scrapping_wikipedia('https://en.wikipedia.org/wiki/Deinococcus_radiodurans')






