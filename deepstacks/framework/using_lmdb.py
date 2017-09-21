#!/usr/bin/env python
# coding:utf-8
# vi:tabstop=4:shiftwidth=4:expandtab:sts=4

#import cv2
import sys
import numpy as np
import random
import lmdb
import caffe
import theano
from .main import register_batch_iterator,register_batches
from ..utils.curry import curry
from ..utils.async_iterate import AsyncIterate
from StringIO import StringIO
import PIL.Image
from ..utils.localshuffler import LocalShuffler

#from deepstacks.framework.main import register_batch_iterator,register_batches
#from deepstacks.utils.curry import curry
#from deepstacks.utils.async_iterate import AsyncIterate
#from StringIO import StringIO
#import PIL.Image
#from deepstacks.utils.localshuffler import LocalShuffler

minibatch_handlers=[]
def register_minibatch_handler(h):
    global minibatch_handlers
    minibatch_handlers+=[h]


def iterate_minibatches(batchsize, database, shufflesize=1, use_caffe=True):

    in_db_data = lmdb.open(database, readonly=True)
    n = int(in_db_data.stat()['entries'])
    #register_batches(n//batchsize)

    with in_db_data.begin() as in_txn:
        cursor = in_txn.cursor()
        cursor.first()
        (key, value) = cursor.item()

        if use_caffe:
            raw_datum = bytes(value)

            datum = caffe.proto.caffe_pb2.Datum()
            datum.ParseFromString(raw_datum)

            s = StringIO()
            s.write(datum.data)
            s.seek(0)
            img = np.array(PIL.Image.open(s))
            if len(img.shape) == 2:
                img = img[:,:,np.newaxis]
            img = img.transpose(2,0,1)

            label = datum.label
        else:
            img_dat = bytes(value)
            (img, label) = pickle.loads(img_dat)

        img_height = img.shape[1] # img = np array with (channels, height, width)
        img_width = img.shape[2]
        img_channels = img.shape[0]

    in_txn = in_db_data.begin()
    cursor = in_txn.cursor()
    cursor.first()
    pos=0

    assert n > batchsize

    localshuffler = LocalShuffler(batchsize*shufflesize,(img_channels, img_width, img_height))

    for start_idx in range(0, n - batchsize + 1, batchsize):
        data_batches = np.zeros((batchsize, img_channels, img_width, img_height), dtype=theano.config.floatX)
        labels_batches = np.zeros((batchsize,), dtype=np.int32)
        ids_batches = np.zeros((batchsize,), dtype=np.int64)

        try: 
            for i in xrange(batchsize):
                (key, value) = cursor.item()

                if use_caffe:
                    raw_datum = bytes(value)

                    datum = caffe.proto.caffe_pb2.Datum()
                    datum.ParseFromString(raw_datum)

                    s = StringIO()
                    s.write(datum.data)
                    s.seek(0)
                    img = np.array(PIL.Image.open(s))
                    if len(img.shape) == 2:
                        img = img[:,:,np.newaxis]
                    img = img.transpose(2,0,1)

                    label = datum.label
                else:
                    img_dat = bytes(value)
                    (img, label) = pickle.loads(img_dat)

                data_batches[i,:,:,:] = img
                labels_batches[i] = np.int32(label)
                ids_batches[i] = np.int64(pos)
                cursor.next()
                pos+=1
        except:
            cursor.close() 
            in_db_data.close()
            print "Unexpected error:", sys.exc_info()[0]
            raise 
        X = {
                'image':data_batches,
                'target':labels_batches.astype('int64'),
                }
        for h in minibatch_handlers:
            h(X,ids_batches)
        yield X,ids_batches

#        if not localshuffler.is_full():
#            #print localshuffler.fill
#            for i in range(batchsize):
#                res=localshuffler.feed(data_batches[i],labels_batches[i],0)
#                assert res is None
#        else:
#            for i in range(batchsize):
#                x,y,ig=localshuffler.feed(data_batches[i],labels_batches[i],0)
#                data_batches[i]=x
#                labels_batches[i]=y
#            X = {
#                    'image':data_batches,
#                    'target':labels_batches.astype('int64'),
#                    }
#            yield X
#    while not localshuffler.is_empty():
#        #print localshuffler.fill
#        data_batches=np.zeros_like(data_batches)
#        labels_batches=np.zeros_like(labels_batches)
#        for i in range(batchsize):
#            x,y,ig=localshuffler.feed(None,None,None)
#            data_batches[i]=x
#            labels_batches[i]=y
#        X = {
#                'image':data_batches,
#                'target':labels_batches.astype('int64'),
#                }
#        yield X

from scipy.misc import imsave,imread
import os
def iterate_one(batchsize, database):
    #print 'iterate_one',batchsize,database
    if os.path.isdir(database):
        a=map(lambda x:os.path.join(database,x),sorted(os.listdir(database)))
    else:
        a=[database]
    for database in a:
        image=imread(database)
        if len(image.shape)==2:
            image=image[...,np.newaxis]
        image=image.transpose(2,0,1)[np.newaxis,...]
        ids = np.zeros((batchsize,),dtype='int64')
        X = {
                'image':image,
                'target':np.zeros((1,1),dtype=theano.config.floatX),
                }
        #print X,ids
        for h in minibatch_handlers:
            h(X,ids)
        #print X,ids
        yield X,ids

register_batch_iterator(
        AsyncIterate(curry(iterate_minibatches,shufflesize=1)),
        AsyncIterate(curry(iterate_minibatches,shufflesize=1)),
        AsyncIterate(curry(iterate_minibatches,shufflesize=1)),
        AsyncIterate(curry(iterate_one)),
        #iterate_one,
        )

if __name__=='__main__':
#    count = 0
#    for x in iterate_minibatches(512,'/home/ubuntu/work/DIGITS/digits/jobs/20170825-163329-a5a4/train_db',shufflesize=2):
#        print x['image'].shape
#        print x['target'].shape
#        print count
#        count += 1
    for x,ids in AsyncIterate(curry(iterate_one))(1,'gogopher.jpg'):
        print x,ids

