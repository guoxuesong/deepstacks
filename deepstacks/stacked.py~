#!/usr/bin/env python
# coding:utf-8
# vi:tabstop=4:shiftwidth=4:expandtab:sts=4

class Layers(object):
    def __init__(self,network,stacks):
        self.layers=[network]
        self.stacks=stacks
        self.future={}

    def add(self,network):
        self.layers+=[network]

    def get_layer(self,k):
        if type(k)==list and len(k)==1:
            raise NotImplementedError
        if type(k)==int:
            return self.layers[::-1][k]
        if type(k)==list and len(k)>1:
            assert len(k)==3
            raise NotImplementedError
        return self.stacks[k][-1] 

    def finish(self):
        pass

def deep_eval(a,m):
    if type(a)==tuple or type(a)==list:
        if type(a)==tuple and (type(a[0])==type(lambda:0) or type(a[0])==type):
            args=[(m[x] if x in m else x) for x in a[1:-1]]
            kwargs=a[-1]
            for k in kwargs:
                if kwargs[k] in m:
                    kwargs[k]=m[kwargs[k]]
            a=a[0](*args,**kwargs)
        else:
            a=type(a)(map(lambda x:deep_eval(x,m),a))
        #a=type(a)(map(lambda x: x(*args,**kwargs) if type(x)==type(lambda:0) else deep_eval(x,*args,**kwargs),a))
    elif type(a)==dict:
        out={}
        for k in a:
            out[k]=deep_eval(a[k],m)
        a=out
    return a

def curr_layer():
    return 'curr_layer'
def curr_stacks():
    return 'curr_stacks'
def curr_flags():
    return 'curr_flags'
def curr_model():
    return 'curr_model'

flag_list=[]
flag_handler={}
flag_excluding={}
def register_flag_handler(flag,handler,excluding=()):
    global flag_list
    if flag not in flag_list:
        flag_list=[flag]+flag_list
    assert flag not in flag_handler
    flag_handler[flag]=handler
    flag_excluding[flag]=set(excluding)

inputs_handler={}
def register_inputs_handler(flag,handler):
    assert flag not in inputs_handler
    inputs_handler[flag]=handler
concat_handler=None
def register_concat_handler(handler):
    concat_handler=handler

def layer_handler(network,flags,stacks,this_model):
    return flags['layer'],{}
def push_handler(network,flags,stacks,this_model):
    keys=flags['push']
    if type(keys)!=tuple and type(keys)!=list:
        keys=[keys]
    for key in keys:
        assert type(key)==str
        if key not in stacks:
            stacks[key]=[]
        stacks[key]+=[network]
    return network,()
def pop_handler(network,flags,stacks,this_model):
    key=flags['pop']
    assert key in stacks
    stacks[key]=stacks[key][:-1]
    if len(stacks[key])==0:
        stacks.pop(key)
    return network,()

register_flag_handler('push',push_handler)
register_flag_handler('pop',pop_handler)
register_flag_handler('layer',layer_handler)

macros=[]
def register_macro_handler(handler):
    global macros
    macros=[handler]+macros

layers_class=Layers
def register_layers_class(l):
    global layers_class
    layers_class=l

network_wrapper=None
def register_network_wrapper(f):
    global network_wrapper 
    network_wrapper=f

def build_network(network,a,m={},**kwargs):
    # a=((INPUTS,flags),)*N

    this_model=kwargs
    this_model['errors']={}
    this_model['watchpoints']={}

    for h in macros:
        a=h(a)

    paramlayers=[]

    stacks={}
    for key in m:
        stacks[key]=[m[key]]
    stacks['input']=[network]

    all_layers=layers_class(network,stacks)

    this_model['get_layer']=all_layers.get_layer
    def get_layer(k):
        return all_layers.get_layer(k)

    count=0
    p=0
    for info in a:
        print count,info

        inputs=info[0]
        if type(info[-1])==set or type(info[-1])==dict:
            pass
        else:
            info=info+({},)

        if type(inputs) == int or type(inputs) == str or type(inputs) == list:
            network=get_layer(inputs)
        elif type(inputs) == tuple:
            layers = map(get_layer,inputs)
            network = None
            for flag in inputs_handler:
                if flag in flags:
                    network = inputs_handler[flag](layers,info[-1],stacks,this_model)
                    break
            if network is None:
                network = concat_handler(layers,info[-1],stacks,this_model)

        info=info[0:1]+deep_eval(info[1:],{
            curr_stacks:stacks,
            curr_layer:network,
            curr_flags:info[-1],
            curr_model:this_model
            })
        flags=info[-1]

        for flag in flag_list:
            if flag in flags:
                if len(flag_excluding[flag] & set(flags))==0:
                    h=flag_handler[flag]
                    network,layers=h(network,info[-1],stacks,this_model)
                    paramlayers+=layers
        all_layers.add(network)
    stacks['output']=[network]
    if 'finish' in kwargs:
        kwargs['finish'](stacks)
    all_layers.finish()

    if network_wrapper:
        network=network_wrapper(network,stacks,this_model)

    return network,stacks,paramlayers,this_model['errors'],this_model['watchpoints']
