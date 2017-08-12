#!/usr/bin/env python
# coding:utf-8
# vi:tabstop=4:shiftwidth=4:expandtab:sts=4

import os
import sys
import random
import numpy as np
import cv
import cv2
import time
import theano
import thread
import threading
import Queue
from .main import register_batch_iterator
from ..util.curry import curry
from ..util.async_iterate import AsyncIterate
from ..util.localshuffler import LocalShuffler

queue=Queue.Queue(256)

def backend():
    while True:
        v=queue.get()
        v.shuffle2()
        v.busy=False
for i in range(3):
    print 'start_new_thread'
    thread.start_new_thread(backend,())

def filter_notbusy(vs):
    vres=[]
    cres=[0]
    for v in vs:
        #v.sync()
        if not v.busy:
            vres+=[v]
            cres+=[cres[-1]+(v.endpos-v.beginpos)]
    cres=cres[1:]
    return vres,cres

#semaphore = threading.Semaphore(3)

class VideoCaptureReader(object):
    def __init__(self,vurl):
        print vurl
        self.vurl=vurl
        self.vcap=cv2.VideoCapture(vurl)
        self.fps=self.vcap.get(cv.CV_CAP_PROP_FPS)
        self.begin=time.time()
        self.last=self.begin
        self.image=None
        self.ctime=time.time()
        self.beginpos=0
        self.endpos=3600*30*2
        ret,image=self.vcap.read()
        if ret:
            self.image=image.copy()
            self.localshuffler=LocalShuffler(4,self.image.shape)
            for i in range(4):
                ret,image=self.vcap.read()
                assert ret
                self.localshuffler.feed(image,0)

            thread.start_new_thread(self.backend,())
            print 'ok'
        else:
            print 'skip'
    @property
    def busy(self):
        if time.time()-self.ctime>600:
            print >>sys.stderr,'warning: vcap fail, try reload'
            self.ctime=time.time()
            self.vcap=cv2.VideoCapture(self.vurl)
        return self.image is None
    def shuffle(self):
        pass
    def backend(self):
        while True:
            ret,image=self.vcap.read()
            if ret:
                self.image=image.copy()
            #else:
            #    self.image=None
            time.sleep(1.0/self.fps)
    def read(self):
        if self.image is None:
            return False,None
        image,ig=self.localshuffler.feed(self.image,0)
        self.image=None
        return True,image

class SliceLooper(object):
    def __init__(self,filename,beginpos,endpos):
        print filename,beginpos,endpos
        self.video=cv2.VideoCapture(filename)
        self.beginpos=beginpos
        self.endpos=endpos
        self.currpos=beginpos+int(random.random()*(endpos-beginpos))
        self.video.set(cv.CV_CAP_PROP_POS_FRAMES,self.currpos)
        self.busy=False
        #self.shuffle_done=False
        #self.shuffle_thread=thread.start_new_thread(self.backend,())
    def shuffle2(self):
        self.currpos=self.beginpos+int(random.random()*(self.endpos-self.beginpos))
        self.video.set(cv.CV_CAP_PROP_POS_FRAMES,self.currpos)
#    def backend(self):
#        while True:
#            if self.shuffle_done:
#                time.sleep(0.1)
#            elif self.busy:
#                semaphore.acquire()
#                #print 'shuffle'
#                self.currpos=self.beginpos+int(random.random()*(self.endpos-self.beginpos))
#                self.video.set(cv.CV_CAP_PROP_POS_FRAMES,self.currpos)
#                self.shuffle_done=True
#                semaphore.release()
#            else:
#                time.sleep(0.1)
    def shuffle(self):
        #assert not self.shuffle_done
        if not self.busy:
            self.busy=True
            queue.put(self)
#    def sync(self):
#        if self.shuffle_done:
#            #print 'sync'
#            self.busy=False
#            self.shuffle_done=False
    def read(self,shuffle=False):
        ret,image=self.video.read()
        count=0
        while True:
            if ret:
                self.currpos+=1
            update=False
            if shuffle and random.random()<0.1 or not ret:
                update=True
                self.currpos=self.beginpos+int(random.random()*(self.endpos-self.beginpos))
            if self.currpos>=self.endpos:
                update=True
                self.currpos=self.beginpos
            if update:
                self.video.set(cv.CV_CAP_PROP_POS_FRAMES,self.currpos)
            if ret:
                return ret,image
            else:
                ret,image=self.video.read()
                count+=1
                if count>=1000:
                    print 'read retry timeout'
                    os._exit(-1)
        return ret,image

def iterate_one(path,y,width,height,batchsize,xdtype,ydtype):
    v = cv2.VideoCapture(path)
    X = np.zeros((batchsize,3,width,height),dtype=xdtype)
    if type(y)==np.array:
        Y = np.zeros((batchsize,y.shape),dtype=ydtype)
    else:
        Y = np.zeros((batchsize,),dtype=ydtype)

    batches = int(v.get(cv.CV_CAP_PROP_FRAME_COUNT)//batchsize)
    print 'batchs',batches
    for i in range(batches):
        time0=time.time()
        for j in range(batchsize):
            r,image=v.read()
            assert r
            image=cv2.resize(image, (width,height), interpolation = cv2.INTER_NEAREST)
            image=image.transpose(2,1,0)
            X[j,:,:,:]=image
            Y[j]=y
        time1=time.time()
        #print time1-time0
        #yield X,Y
        #X/=256
        yield {'image':(X/256.0).astype(xdtype),'target':Y}

class iterate(object):
    def __init__(self,paths,ys,width,height,batchsize,batches,xdtype,ydtype,num_slice,fixed_shuffle_rate=False):
        assert len(paths)==len(ys)
        m = {}
        count = {}
        for p,y in zip(paths,ys):
            m[y]=[]
            count[y]=[0]
            a=os.listdir(p)
            eff_num_slice = max(num_slice,(batchsize*2)//len(a))
            # print a
            for t in a:
                if t.endswith('.url'):
                    url=open(os.path.join(p,t)).read().strip()
                    v = VideoCaptureReader(url)
                    m[y]+=[v]
                    beginpos=v.beginpos
                    endpos=v.endpos
                    count[y]+=[count[y][-1]+(endpos-beginpos)]
                else:
                    v = cv2.VideoCapture(os.path.join(p,t))

                    r = t.split('.')[-2]
                    if r[0]=='[' and r[-1]==']':
                        begin_time,end_time=r[1:-1].split('-')
                        begin_time=float(begin_time)
                        v.set(cv.CV_CAP_PROP_POS_MSEC,int(begin_time*1000))
                        begin=v.get(cv.CV_CAP_PROP_POS_FRAMES)
                        if end_time != '':
                            end_time=float(end_time)
                            v.set(cv.CV_CAP_PROP_POS_MSEC,int(end_time*1000))
                            end=v.get(cv.CV_CAP_PROP_POS_FRAMES)
                        else:
                            end = v.get(cv.CV_CAP_PROP_FRAME_COUNT)
                        n=end-begin
                    else:
                        begin=0
                        n = v.get(cv.CV_CAP_PROP_FRAME_COUNT)
                        end=n
                    slices = []
                    for i in range(0,int(n)-int(n)//eff_num_slice+1,int(n)//eff_num_slice):
                        slices+=[(begin+i,begin+min(n,i+int(n)//eff_num_slice))]
                    for beginpos,endpos in slices:
                        # print beginpos,'/',n,t
                        v = SliceLooper(os.path.join(p,t),beginpos,endpos)
                        m[y]+=[v]
                        count[y]+=[count[y][-1]+(endpos-beginpos)]
            count[y]=count[y][1:]
        self.batches=batches
        self.X = np.zeros((batchsize,3,width,height),dtype=xdtype)
        if type(ys[0])==np.array:
            self.Y = np.zeros((batchsize,ys[0].shape),dtype=ydtype)
        else:
            self.Y = np.zeros((batchsize,),dtype=ydtype)
        self.batchsize=batchsize
        self.paths=paths
        self.ys=ys
        self.m=m
        self.count=count
        self.width=width
        self.height=height
        self.xdtype=xdtype
        self.ydtype=ydtype
        self.shuffle_rate=1.0
        self.fixed_shuffle_rate=fixed_shuffle_rate

    def __call__(self):
        for i in range(self.batches):
            #X = self.X.copy()
            #Y = self.Y.copy()
            X = np.zeros((self.batchsize,3,self.width,self.height),dtype=self.xdtype)
            if type(self.ys[0])==np.array:
                Y = np.zeros((self.batchsize,self.ys[0].shape),dtype=self.ydtype)
            else:
                Y = np.zeros((self.batchsize,),dtype=self.ydtype)
            time0=time.time()
            for j in range(self.batchsize):
                k = int(random.random()*len(self.paths))
                y = self.ys[k]
                Y[j]=y
                vs = self.m[y]
                cs = self.count[y]

                vs,cs = filter_notbusy(vs)
                if len(vs)==0:
                    if not self.fixed_shuffle_rate:
                        if self.shuffle_rate>0.1:
                            self.shuffle_rate*=0.9
                    count=0
                    while len(filter(lambda x:isinstance(x,SliceLooper),vs))!=len(filter(lambda x:isinstance(x,SliceLooper),self.m[y])):
                        vs,cs = filter_notbusy(self.m[y])
                        count+=1
                        if count>=1000:
                            print 'wait shuffle timeout'
                            os._exit(-1)
                        time.sleep(0.01)
                print >>sys.stderr,y,len(vs),cs[-1],self.shuffle_rate

                pos = int(random.random()*cs[-1])
                for p,v in zip(cs,vs):
                    if p>pos:
                        break
                r,image=v.read()
                assert r
                if random.random()<self.shuffle_rate:
                    v.shuffle()
                #image=cv2.resize(image, (width,height), interpolation = cv2.INTER_AREA)
                image=cv2.resize(image, (self.width,self.height), interpolation = cv2.INTER_NEAREST)
                #image=cv2.resize(image, (width,height))
                image=image.transpose(2,1,0)
                X[j,:,:,:]=image
            if not self.fixed_shuffle_rate:
                if self.shuffle_rate<1.0:
                    self.shuffle_rate*=1.001
            time1=time.time()
            #print time1-time0
            #yield X,Y
            #X/=256
            yield {'image':(X/256.0).astype(self.xdtype),'target':Y}

train_iter=None
def iterate_minibatches_train(batchsize,database):
    global train_iter
    if train_iter is None:
        a=os.listdir(database)
        paths=map(lambda x:os.path.join(database,x),a)
        ys=map(lambda x:int(x),a)
        print paths,ys
        train_iter=iterate(paths,ys,256,256,batchsize,400,theano.config.floatX,'int64',1)
    return train_iter()

val_iter=None
def iterate_minibatches_val(batchsize,database):
    global val_iter
    if val_iter is None:
        a=os.listdir(database)
        paths=map(lambda x:os.path.join(database,x),a)
        ys=map(lambda x:int(x),a)
        val_iter=iterate(paths,ys,256,256,batchsize,40,theano.config.floatX,'int64',1,fixed_shuffle_rate=True)
    return val_iter()

def iterate_minibatches_inference(batchsize,database):
    #a=os.listdir(database)
    return iterate_one(os.path.join(database,database),0,256,256,batchsize,'float32','int64')

register_batch_iterator(AsyncIterate(iterate_minibatches_train),AsyncIterate(iterate_minibatches_val),AsyncIterate(iterate_minibatches_inference))
