import os
import sys
import cPickle as pickle
from PIL import Image

dirname = "../dataset/all/"
image_cnt = 0
error_cnt = 0
image_sizes = {};
loaded = True;

if not loaded:
    for subdir, dirs, files in os.walk(dirname):
        for f in files:  
            if image_cnt % 1000 == 0: 
                sys.stdout.write('.') 
                sys.stdout.flush()

            fn = os.path.join(subdir, f)
            if fn.find("txt") != -1:
                continue;

            try:
                im = Image.open(fn)
            except:
                error_cnt += 1 
                continue;

            image_cnt += 1
            width, height = im.size;
            
            if (width, height) not in image_sizes:
                image_sizes[(width, height)] = 0
            image_sizes[(width, height)] += 1

    print ''

    print "number of images:", image_cnt, error_cnt
    print "number of different sizes: ", len(image_sizes)
    for size in image_sizes:
        print size, image_sizes[size]

    f = open('imagesize.p', 'w')
    pickle.dump(image_sizes, f)
    f.close()

else:
    f = open('imagesize.p', 'r')
    image_sizes = pickle.load(f)
    f.close()
    image_size_ratios = {}
    in_cnt = 0

    for size in image_sizes:
        ratio = float(size[0]) / float(size[1])
        # almost identical
        ratio *= 10
        ratio = int(ratio)
        if ratio not in image_size_ratios:
            image_size_ratios[ratio] = 0
        image_size_ratios[ratio] += image_sizes[size]

    for ratio in image_size_ratios:
        print ratio, image_size_ratios[ratio]
        if ratio >= 5 and ratio <= 20:
            in_cnt += image_size_ratios[ratio]

    print "number of different ratio (width / height):", len(image_size_ratios)
    print "number of images with ratio in [0.5, 2]:", in_cnt
