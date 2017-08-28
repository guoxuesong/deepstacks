#!/usr/bin/env python
# coding:utf-8
# vi:tabstop=4:shiftwidth=4:expandtab:sts=4

import Queue
import thread

class AsyncIterate:

    def __init__(self,it):
        self.it=it
        self.instance=None
        self.quitflag=False
        self.q=Queue.Queue(2)

    def __copy__(self):
        return AsyncIterate(self.it)

    def __call__(self,*args,**kwargs):
        assert self.instance is None
        self.instance=self.it(*args,**kwargs)
        if self.instance is not None:
            thread.start_new_thread(self.backend,())
            data=self.q.get()
            while data is not None:
                try:
                    yield data
                except GeneratorExit:
                    #print 'GeneratorExit'
                    self.quitflag=True
                    #self.instance.close()
                    while self.q.get() is not None:
                        pass
                    break
                data=self.q.get()
            #print 'end'
            self.instance=None
            self.quitflag=False
    
    def backend(self):
        try:
            while True:
                data=self.instance.next()
                if self.quitflag:
                    self.instance.close()
                if data is not None:
                    self.q.put(data)
        except (GeneratorExit,StopIteration):
            self.q.put(None)

if __name__=='__main__':
    def myit(n):
        for x in range(n):
            yield x

    iterator=AsyncIterate(myit)

    it=iterator(5)
    for t in it:
        print t
        it.close()
        break
    it=iterator(5)
    for t in it:
        print t
