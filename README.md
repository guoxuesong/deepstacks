DEEPSTACKS
==========

A build_network() for Lasagne and noen. Define your network model in a
datacheet with stack machine mechanisms.  Support reuse part of model as
function, and share parameters.

Philosophy
----------

The philosophy of deepstacks is: thinking using eyes.

About 1/3 of human brain is relevant to vision. So if you think using vision,
you should be at least 50% brighter than who not.

To think using eyes, the best way is to manage the infomation that showed to
yourself, as compact as posible.

That is why we intend to represent neural networks using a datacheet-like
struct, instead of code. For example, a inception block in googLenet will be
represent this way:

	((0,0,3,1,0,0,{'maxpool'}),
	(0,nfilters[0],1,1,0,0,{}),
	(2,nfilters[1],1,1,0,0,{}),
	(3,nfilters[2],1,1,0,0,{}),
	(0,nfilters[3],3,1,0,0,{}),
	(5,nfilters[4],1,1,0,0,{}),
	(0,nfilters[5],5,1,0,0,{}),
	((4,2,0,5),0,0,0,name,0,{}),)

Keep the items and relationships in a small viewport, so that a single look can
catch the meaning. That would be much better than tedious documents.

Professional
------------

Professional not means good, not always, Professional means you do this all
day. When you do something all day, what you care about is different from who
do this once a week. the latter similar with beginners care about 'easy to
learn', 'readable', 'remember less'. 

What a professional care about is 'performance', 'save energy', 'easy to
change'. A professional totaly not care about 'not easy to learn' or 'remember
too much', because compare with the working time, the cost of learning can be
ignored, and the repeated using of the knowlege makes it really hard to forget.

For example, Vim is a tool for professional, the user of Vim coding all day,
remember a lot of shortcut keys, totally not need menu.

Deepstacks intend to be this kind of tool, for professional. The user need to
remember the meaning of ever fields, and what flags can be used in the {}. But
after that, you can get good performance.

Fields
------

A deepstacks line composed with 6 fields and a flags dict, the 6 fields are:

  bottom_layer, num_filters, filter_size, stride, push_to, share_id

* bottom_layer: int, str or tuple. if int, 0 means use the last line as the
bottom layer, N means use the last N+1 line as the bottom layer; if str, use
the top of stack that bottom_layer equals to the name of the stack; if tuple,
every item is int or str each stand for a layer, using the foregoing rules, the
tuple of these layers ast as inputs of ConcatLayer by default or MergeLayer if
'op','add' or 'sub' in flags.

* num_filters: int, slice or tuple. if int >0, and none of 'maxpool' 'meanpool' or
'upscale' is in flags, a DenseLayer or Conv[123]DLayer will be
constructed, depend on the output_shape of bottom_layer, if 'dense' in flags,
use DenseLayer; if slice, a SliceLayer will be constructed, num_filters used as
indices; if tuple, a ReshapeLayer will be constructed, (-1,)+num_filters used
as shape.

* filter_size: int, used to construct Conv[123]DLayer.

* stride: int, used to construct Conv[123]DLayer.

* putsh_to: str or 0, if not 0, push the result of this line to stack whose name equals to push_to.

* share_id: str or 0, if not 0, the layers has same share_id share params. Keep
it 0, macro 'share' and 'call' will handle it.

Flags
-----

Following flags take effect in order:

* 'op': callable, bottom_layer should be tuple, ElemwiseMergeLayer, use
bottom_layer as incomings, flags['op'] as merge_function.

* 'add': equal to flags['op':theano.tensor.add]

* 'sub': equal to flags['op':theano.tensor.sub]

If any of the following three flags take effect, 'num_filters' would be IGNORED:

* 'maxpool': Pool[123]DLayer mode='max',use filter_size as pool_size, use stride

* 'meanpool': Pool[123]DLayer mode='average_inc_pad',use filter_size as pool_size, use stride

* 'upscale': Upscale[123]DLayer mode='repeat', use filter_size as scale_factor

Following flags take effect AFTER 'num_filters'.

* 'dimshuffle': DimshuffleLayer use flags['dimshuffle'] as pattern

* 'max': goroshin_max layer, see deepstacks/lasagne/argmax.py

* 'argmax': goroshin_argmax layer, see deepstacks/lasagne/argmax.py

* 'unargmax': goroshin_unargmax layer, see deepstacks/lasagne/argmax.py

* 'noise': GaussianNoiseLayer layer, use flags['noise'] as sigma

* 'nonlinearity': add a extra NonlinearityLayer or ExpressionLayer if
num_filters==0, use flags['nonlinearity'] as nonlinearity, use flags['shape']
as out_shape of ExpressionLayer if exists, if num_filters!=0, used by default
handler of num_filters.

* 'relu': add a extra NonlinearityLayer with nonlinearity=relu, the default relu
is rectify.

* 'watch': [equal_to, name, eq], add a named watchpoint represent
eq(curr_layer,equal_to), see deepstacks/lasagne/implment.py for detail.

* 'equal': [equal_to, name, eq], add a named constraint represent
eq(curry_layer,equal_to), see deepstacks/lasagne/implment.py for detail, see
exaples/lasagne/3.constraint.py for usage.
