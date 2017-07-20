# DEEPSTACKS
A build_network() for Lasagne and noen. Define your network model in a
datacheet with stack machine mechanisms.  Support reuse part of model as
function, and share parameters.

# Philosophy
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
	((5,4,2,0,6),0,0,0,name,0,{}),)

Keep the items and relationships in a small viewport, so that a single look can
catch the meaning. That would be much better than tedious documents.

# Professional
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

