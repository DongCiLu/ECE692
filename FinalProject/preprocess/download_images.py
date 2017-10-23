import json
import pandas as pd
import re
import os
import requests
import argparse
import urllib
import logging
from datetime import datetime

downloaded_images = set()

def save_media_info(dict_tweet, image_path, entity):
    # get the text of tweets
    if ('text' in dict_tweet):
        tweet_text = dict_tweet['text']
    elif ('full_text' in dict_tweet):
        tweet_text = dict_tweet['full_text']
    # print (dict_tweet['entities'])
    # check the media elements 
    if ('media' in dict_tweet[entity]):
        # dict_tweet['entities']['media'] is a list 
        for media_item in dict_tweet[entity]['media']: 
            if (media_item['type'] == 'photo'):
                image_url = media_item['media_url']
                tweet_id = dict_tweet['id']
                # a new image
                if (image_url not in downloaded_images):
                    downloaded_images.add(image_url)
                    # image_data = requests.get(image_url).content
                    ipath = "{}{}.jpg".format(image_path, tweet_id)
                    tpath = "{}{}.txt".format(image_path, tweet_id)
                    try:
                        urllib.urlretrieve(image_url, ipath)
                    except Exception as e:
                        logging.warning("image file writing error for {} ({}): {}".format(tweet_id, type(e), str(e)))
                    with open(tpath, 'w') as f:
                        try:
                            f.write(tweet_text.encode('utf8'))
                            # f.write(tweet_text)
                        except Exception as e:
                            logging.warning("text file writing error for {} ({}): {}".format(tweet_id, type(e), str(e)))

def extract_media_info(dict_tweet, image_path):
    if ('entities' in dict_tweet):
        entity = 'entities'
        save_media_info(dict_tweet, image_path, entity)

    if ('extended_entities' in dict_tweet):
        entity = 'extended_entities'
        save_media_info(dict_tweet, image_path, entity)

    for key in dict_tweet:
        if (isinstance(dict_tweet[key], dict)): 
            extract_media_info(dict_tweet[key], image_path)

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('source_dirname', type=str)
    arg_parser.add_argument('dst_dirname', type=str)
    args = arg_parser.parse_args()

    log_filename = 'figure_crawl_{}.log'.format(datetime.now())
    logging.basicConfig(filename=log_filename, level=logging.WARNING)

    last_tweet_cnt = 0
    last_error_cnt = 0
    tweet_cnt = 0
    error_cnt = 0

    for subdir, dirs, files in os.walk(args.source_dirname):
        print subdir, ': '
        for filename in files:
            full_fn = os.path.join(subdir, filename)
            print "+++ ", full_fn
            with open(full_fn) as f:
                for line in f:
                    try:
                        tweet = json.loads(line)
                        extract_media_info(tweet, args.dst_dirname)
                        tweet_cnt += 1
                        if tweet_cnt % 100000 == 0:
                            print tweet_cnt, '(', error_cnt, '):',
                        if tweet_cnt % 1000 == 0:
                            sys.stdout.write('.')
                            sys.stdout.flush()
                    except:
                        error_cnt += 1
                        continue
            print "Total counts in this file: ", \
                    tweet_cnt - last_tweet_cnt, \
                    '(', error_cnt - last_error_cnt, ').'
            last_tweet_cnt = tweet_cnt
            last_error_cnt = error_cnt
    print "Total counts: ", tweet_cnt, '(', error_cnt, ').'
