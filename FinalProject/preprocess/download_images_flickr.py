import os
import sys

from subprocess import call
from tqdm import tqdm
import urllib
import validators

if __name__ == '__main__':
    url_path = "../dataset/sentibank_flickr/image_url"
    img_path = "../dataset/sentibank_flickr/image"

    for subdir, dirs, files in os.walk(url_path):
        pbar = tqdm(range(len(files)), desc=subdir)
        for i in pbar:
            fn = os.path.join(subdir, files[i])
            img_anp = files[i].split('.')[0]
            img_subdir = os.path.join(img_path, img_anp)
            os.mkdir(img_subdir)
            with open(fn) as f:
                for line in f:
                    img_url = line.split(' ')[1]
                    img_name = img_url.split('/')[-1]
                    img_fn = os.path.join(img_subdir, img_name)
                    try:
                        urllib.urlretrieve(img_url, img_fn)
                    except Exception as e:
                        print "Failed downloading image for {} ({}): {}".format(
                                image_url, type(e), str(e))
