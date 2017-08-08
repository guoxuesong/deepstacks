#!/usr/bin/env python
# coding:utf-8
# vi:tabstop=4:shiftwidth=4:expandtab:sts=4

import theano
import lasagne
import numpy as np
import cv2
import main
#from .main import batch_iterator_train,batch_iterator_test
from .main import register_training_callbacks, register_model_handler
from ..util.curry import curry

floatX=theano.config.floatX

using_fingerprint=False

def imnorm(x):#{{{
    M=np.max(x)
    m=np.min(x)
    l=M-m
    if l==0:
        l=1.0
    res=((x-m)*1.0/l*255.0).astype('uint8')
    return res#}}}

visual_keys=[]
visual_vars=[]
visual_varnames=[]
def register_visual(key):
    global visual_keys
    visual_keys+=[key]
def register_visual_var(name,var):
    global visual_vars
    global visual_varnames
    visual_vars+=[var]
    visual_varnames+=[name]

num_batchsize = None
walker_fn = None
predict_fn = None
visual_fn = None
def handle_model(inputs,network,stacks,layers,errors,watchpoints):
    global predict_fn,visual_fn,num_batchsize

    num_batchsize=lasagne.layers.get_output_shape(network)[0]

    if 'predict' in stacks:
        key='predict'
    else:
        key='output'
    predict_fn = theano.function([inputs['source_image'].input_var,inputs['action'].input_var], 
        lasagne.layers.get_output(stacks[key][0],deterministic=True),
        on_unused_input='warn', allow_input_downcast=True)
    visual_fn = theano.function([inputs['source_image'].input_var,inputs['action'].input_var], 
        [lasagne.layers.get_output(stacks[key][0],deterministic=True) for key in visual_keys],
        on_unused_input='warn', allow_input_downcast=True)

register_model_handler(handle_model)

src=None
norms=None
predictsloop=[]
predictsloop2=[]
predictsloop3=[]
bottom=None
right=None

sn=0
def mybatchok():
    global sn
    show(src,norms,predictsloop,predictsloop2,predictsloop3,sn,bottom=bottom,right=right)
    cv2.waitKey(100)
    sn+=1

def imshow64x64(name,img):
    w,h,c=img.shape
    if w==256:
        img=img[::4,::4,:]
    cv2.imshow(name,img)

def show(src,norms,predictsloop,predictsloop2,predictsloop3,t,bottom=None,right=None):
    t=t%len(predictsloop)
    w=64
    h=64
    xscreenbase=0
    yscreenbase=0
    for i in range(num_batchsize):
        for j in range(len(src)):
            #j=0
            imshow64x64("sample-"+str(i)+"-"+str(j),
                norms[j](src[j][i,0:3,:,:].transpose(1,2,0)))
            cv2.moveWindow("sample-"+str(i)+"-"+str(j),xscreenbase+j*w,yscreenbase+i*h)

        #j=1
        #cv2.imshow("sample-"+str(i)+"-"+str(j),
        #    imnorm(srchide0[i,0:3].transpose(1,2,0)))
        #cv2.moveWindow("sample-"+str(i)+"-"+str(j),xscreenbase+j*w,yscreenbase+i*h)
        #j=1
        #cv2.imshow("sample-"+str(i)+"-"+str(j),
        #    imnorm(recon[i,0:3].transpose(2,1,0)))
        #cv2.moveWindow("sample-"+str(i)+"-"+str(j),xscreenbase+j*w,yscreenbase+i*h)

        n=j+1

        #for p in range(1):
        #    j=p+n
        #    base=srchide1.shape[1]-1
        #    cv2.imshow("sample-"+str(i)+"-"+str(j),
        #        imnorm(enlarge(srchide1[i,base+p:base+p+1].transpose(1,2,0),4)))
        #    cv2.moveWindow("sample-"+str(i)+"-"+str(j),xscreenbase+j*w,yscreenbase+i*h)
        #n+=1


        for p in range(4):
            j=p+n
            imshow64x64("sample-"+str(i)+"-"+str(j),
                imnorm(predictsloop[t][p][i,0:3].transpose(2,1,0)))
            cv2.moveWindow("sample-"+str(i)+"-"+str(j),xscreenbase+j*w,yscreenbase+i*h)
        n+=4
        for p in range(4): #num_actions+4
            j=p+n
            imshow64x64("sample-"+str(i)+"-"+str(j),
                imnorm(predictsloop2[t][p][i,0:3].transpose(2,1,0)))
            cv2.moveWindow("sample-"+str(i)+"-"+str(j),xscreenbase+j*w,yscreenbase+i*h)
        n+=4 #num_actions+4
        if i>=7:
            break
    if bottom is not None:
        cv2.imshow('bottom',bottom)
        cv2.moveWindow("bottom",0,64*8)
    if right is not None:
        cv2.imshow('right',right)
        cv2.moveWindow("right",64*10,0)

visualize_validation_set=False

def set_visualize_validation_set(v):
    global visualize_validation_set
    visualize_validation_set=v

def myupdate():
    global src,norms,predictsloop,predictsloop2,predictsloop3
    if visualize_validation_set:
        iterate_fn=main.batch_iterator_test
        db=main.args.train_db
    else:
        iterate_fn=main.batch_iterator_train
        db=main.args.validation_db
    src=None
    norms=None
    predictsloop=[]
    predictsloop2=[]
    predictsloop3=[]
    it=iterate_fn(num_batchsize,db)
    for batch in it:
        #inputs, inputs2, actions, outputs, outputs2, rewards, targets, flags = batch

        inputs = batch['source_image']
        actions = batch['action']
        outputs = batch['target_image']

        #if inputs.shape[2]==256:
        #    inputs=inputs[:,:,::4,::4]
        #    outputs=outputs[:,:,::4,::4]

        vis_args = [batch[key] for key in visual_varnames]
        vis = visual_fn(inputs,actions,*vis_args)
        #srchide0=hides[0]
        #srchide1=hides[1]
        #srchide2=hides[2]
        #srchide0=srchide0.transpose(0,1,3,2)
        #srchide1=srchide1.transpose(0,1,3,2)
        #srchide2=srchide2.transpose(0,1,3,2)

        #recon=recon_fn(inputs,
        #        np.concatenate((actions,np.zeros((num_batchsize,2,1,1),dtype=floatX)),axis=1),
        #        outputs)

        num_actions=6

        p=[inputs]*(num_actions+4)
        for t in range(5):
            #sources=[] 
            predicts=[] 
            predicts2=[] 
            for i in range(num_actions+4):

                if t>0:
                    actions1=np.concatenate((np.eye(num_actions+4,dtype=floatX)[i:i+1],)*num_batchsize,axis=0).reshape(num_batchsize,num_actions+4,1,1)*(0.1)
                else:
                    actions1=np.concatenate((np.eye(num_actions+4,dtype=floatX)[i:i+1],)*num_batchsize,axis=0).reshape(num_batchsize,num_actions+4,1,1)*(0.0)
                    #print 'predict actions',actions1
                if not using_fingerprint:
                    predict=predict_fn(
                        p[i],
                        actions1,
                        )
                else:
                    predict=predict_fn(
                        p[i],
                        batch['source_fingerprint'], #XXX
                        outputs,#XXX
                        batch['target_fingerprint'], #XXX
                        actions1,
                        )
                predicts+=[predict]

                actions2=np.concatenate((np.eye(num_actions+4,dtype=floatX)[i:i+1],)*num_batchsize,axis=0).reshape(num_batchsize,num_actions+4,1,1)*(0.1*t)
                if not using_fingerprint:
                    predict=predict_fn(
                        inputs,
                        actions2,
                        )
                else:
                    predict=predict_fn(
                        inputs,
                        batch['source_fingerprint'],
                        outputs,
                        batch['target_fingerprint'],
                        actions2,
                        )
                predicts2+=[predict]
            p=predicts
            predictsloop+=[predicts]
            #predictsloop+=[map(deconv3_fn,predicts)]
            predictsloop2+=[predicts2]

        #tmp=list_transpose(sources) # (num_batchsize,actions,[features,width,height])
        #for j in range(8):
        #    print [(x**2).sum()**0.5 for x in tmp[j][0:4]]
        #print np. asarray(bn_std)
        #src=[inputs.transpose(0,1,3,2),np.roll(inputs.transpose(0,1,3,2),1,axis=0)]
        src=[inputs.transpose(0,1,3,2),outputs.transpose(0,1,3,2)]+[t.transpose(0,1,3,2) for t in vis]
        norms=[imnorm]*len(src)

        #print action
        for t in range(len(predictsloop)):
            show(src,norms,predictsloop,predictsloop2,predictsloop3,t)
            if 0xFF & cv2.waitKey(100) == 27:
                break_flag = True
                break

        it.close()
        break

register_training_callbacks(
        [ mybatchok ], 
        [ myupdate ],
        [ myupdate ],
        [],)


