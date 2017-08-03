#!/usr/bin/env python
# coding:utf-8
# vi:tabstop=4:shiftwidth=4:expandtab:sts=4

import numpy as np
from .main import register_batch_iterator

frames=270
rl_dummy=16


def load_dataset():
    if frames==90:
        npyfile=('200x64x'+str(frames)+'.npy',)
    else:
        npyfile=('200x64x'+str(frames)+'-0.npy',
                '200x64x'+str(frames)+'-1.npy')
    if not os.path.exists(npyfile[0]):
        #aa=[] #(frames, 3, 256, 256)*16
        base=0
        for dir in ['turnleft','turnright','up','down']:
            for i in range(200):
                a=np.load('200x64x'+str(frames)+'/'+dir+'/'+dir+'-'+str(i)+'.npz')['arr_0']
                writenpy(npyfile,(3,64,64),np.arange(len(a))+base,a,fast=True)
                base+=len(a)
    return curry(readnpy,npyfile,(3,64,64))

minibatch_handlers=[]
def register_minibatch_handler(h):
    global minibatch_handlers
    minibatch_handlers+=[h]

centralize=False
one_direction=False

def iterate_minibatches(aa,stage,batchsize,iteratesize=400, shuffle=False, idx=False):
    rangeframes=frames/90*10
    unitframes=frames/90*10
    i=0
    last=None
    batchsize0=batchsize
    while i<iteratesize:
        if stage=='train':
            if last is not None:
                batchsize=batchsize0#/2
            else:
                batchsize=batchsize0
        else:
            batchsize=batchsize0
        if stage=='train':
            actions1=np.zeros((batchsize,handledata.num_curr_actions,1,1),dtype=floatX)
            #actions2=np.zeros((batchsize,handledata.num_curr_actions,1,1),dtype=floatX)
            k=(np.random.rand(batchsize)*800).astype('int')
            beginpos=rangeframes+(np.random.rand(batchsize)*(frames-rangeframes*2)).astype('int')
            if centralize:
                k[::2]=k[0]
                beginpos[::2]=beginpos[0]
            action1=(np.random.rand(batchsize)*rangeframes).astype('int')
            #action2=(np.random.rand(batchsize)*rangeframes).astype('int')
            #endpos=beginpos+action2
            beforpos=beginpos-action1
            faction1=action1.astype(floatX)/unitframes
            #faction2=action2.astype(floatX)/unitframes
            isleft=(k<200)
            isright=(k>=200)*(k<400)
            isup=(k>=400)*(k<600)
            isdown=(k>=600)
            actions1[:,0,0,0]=isleft*faction1
            #actions2[:,0,0,0]=isleft*faction2
            actions1[:,1,0,0]=isright*faction1
            #actions2[:,1,0,0]=isright*faction2
            actions1[:,2,0,0]=isup*faction1
            #actions2[:,2,0,0]=isup*faction2
            actions1[:,3,0,0]=isdown*faction1
            #actions2[:,3,0,0]=isdown*faction2
            assert idx==False
            actions1=np.concatenate((actions1[:,0:4],
                np.zeros((batchsize,6,1,1),dtype=floatX)
                ),axis=1)
            #actions2=np.concatenate((actions2[:,0:4],
            #    np.zeros((batchsize,6,1,1),dtype=floatX)
            #    ),axis=1)
        else:
            #actions1=np.zeros((batchsize,handledata.num_curr_actions,1,1),dtype=floatX)
            actions2=np.zeros((batchsize,handledata.num_curr_actions,1,1),dtype=floatX)
            k=(np.random.rand(batchsize)*800).astype('int')
            #action1=(np.random.rand(batchsize)*rangeframes).astype('int')
            action2=(np.random.rand(batchsize)*rangeframes).astype('int')
            #beginpos=rangeframes+(np.random.rand(batchsize)*(frames-rangeframes*2)).astype('int')
            endpos=frames-1-(np.random.rand(batchsize)*rangeframes).astype('int')
            beginpos=endpos-action2
            #endpos=beginpos+action2
            #beforpos=beginpos-action1
            #faction1=action1.astype(floatX)/unitframes
            faction2=action2.astype(floatX)/unitframes
            isleft=(k<200)
            isright=(k>=200)*(k<400)
            isup=(k>=400)*(k<600)
            isdown=(k>=600)
            #actions1[:,0,0,0]=isleft*faction1
            actions2[:,0,0,0]=isleft*faction2
            #actions1[:,1,0,0]=isright*faction1
            actions2[:,1,0,0]=isright*faction2
            #actions1[:,2,0,0]=isup*faction1
            actions2[:,2,0,0]=isup*faction2
            #actions1[:,3,0,0]=isdown*faction1
            actions2[:,3,0,0]=isdown*faction2
            assert idx==False
            #actions1=np.concatenate((actions1[:,0:4],
            #    np.zeros((batchsize,6,1,1),dtype=floatX)
            #    ),axis=1)
            actions2=np.concatenate((actions2[:,0:4],
                np.zeros((batchsize,6,1,1),dtype=floatX)
                ),axis=1)

        if stage=='train':
            if is_autoencoder:
                batch = ((aa(k*frames+beforpos)/256.0).astype(floatX),None,actions1,"(aa(k*frames+beginpos)/256.0).astype(floatX)",None,"actions2","(aa(k*frames+endpos)/256.0).astype(floatX)",None,None,None,None)
                images1, ig1, actions1, images2, ig2, actions2, images3, ig3, rewards, targets, flags = batch
                images2=images1
                actions1=np.zeros_like(actions1)
            else:
                batch = ((aa(k*frames+beforpos)/256.0).astype(floatX),None,actions1,(aa(k*frames+beginpos)/256.0).astype(floatX),None,"actions2","(aa(k*frames+endpos)/256.0).astype(floatX)",None,None,None,None)
                images1, ig1, actions1, images2, ig2, actions2, images3, ig3, rewards, targets, flags = batch
            idx1 = create_fingerprint(k*frames+beforpos,32)
            idx2 = create_fingerprint(k*frames+beginpos,32)
            if images1.shape[2]==256:
                images1=images1[:,:,::4,::4]
                images2=images2[:,:,::4,::4]
                #images3=images3[:,:,::4,::4]

            action_samples=build_action_sample(rl_dummy,4,zero=True)

            #if sigma_const is not None:
            #    sigma=np.ones_like(sigma)*sigma_const

            inputs=images1
            outputs=images2
            actions=actions1

            #actions[::8]=np.zeros_like(actions[::8])

            drdscxcy=np.zeros((batchsize,4,1,1),dtype=floatX)
            dxdy=np.zeros((batchsize,2,1,1),dtype=floatX)

            actions=np.concatenate((actions[:,0:4],
                drdscxcy[:,2:4].reshape(batchsize,2,1,1),
                dxdy.reshape(batchsize,2,1,1),
                drdscxcy[:,0:2].reshape(batchsize,2,1,1),
                ),axis=1)
            samples=np.concatenate((action_samples,np.zeros((rl_dummy,num_extra_actions,1,1),dtype=floatX)),axis=1)

            if last is None:
                for j in range(batchsize):
                    if not one_direction:
                        if random.random()*2.0<=1.0:
                            actions[j]=-actions[j]
                            tmp=inputs[j]
                            inputs[j]=outputs[j]
                            outputs[j]=tmp

                X={
                        'source_image':inputs,
                        'target_image':outputs,
                        'action':actions,
                        }
                if using_fingerprint:
                    X['source_fingerprint']=idx1
                    X['target_fingerprint']=idx2
                for h in minibatch_handlers:
                    h(X)
            else:
                inputs=np.concatenate((last['source_image'][batchsize:],inputs),axis=0)
                outputs=np.concatenate((last['target_image'][batchsize:],outputs),axis=0)
                actions=np.concatenate((last['action'][batchsize:],actions),axis=0)

                for j in range(batchsize):
                    if not one_direction:
                        if random.random()*2.0<=1.0:
                            actions[j]=-actions[j]
                            tmp=inputs[j]
                            inputs[j]=outputs[j]
                            outputs[j]=tmp

                X={
                        'source_image':inputs,
                        'target_image':outputs,
                        'action':actions,
                        }
                if using_fingerprint:
                    X['source_fingerprint']=idx1
                    X['target_fingerprint']=idx2
                for h in minibatch_handlers:
                    h(X)
            last=X

            yield X
        else:
            X={
                    'source_image':(aa(k*frames+beginpos)/256.0).astype(floatX),
                    'target_image':(aa(k*frames+endpos)/256.0).astype(floatX),
                    'action':actions2,
                    }
            idx1 = create_fingerprint(k*frames+beginpos,32)
            idx2 = create_fingerprint(k*frames+endpos,32)
            if using_fingerprint:
                X['source_fingerprint']=idx1
                X['target_fingerprint']=idx2
            for h in minibatch_handlers:
                h(X)
            yield X
        i+=1

aa = load_dataset()
register_batch_iterator(AsyncIterate(curry(iterate_minibatches,aa,'train',iteratesize=400,shuffle=True)),AsyncIterate(curry(iterate_minibatches,aa,'val',iteratesize=1,shuffle=True)))