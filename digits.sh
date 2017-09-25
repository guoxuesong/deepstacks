#! /bin/bash

USERNAME=`whoami`
while getopts :u:h FLAG; do
  case $FLAG in
    u)  #set option "a"
      USERNAME=$OPTARG
      ;;
    h)  #show help
      echo usage: digits.sh [-u USERNAME] network.py DATASET BATCHSIZE
      exit
      ;;
    \?) #unrecognized option - show help
      echo -e \\n"Option -${BOLD}$OPTARG${NORM} not allowed."
      exit
      ;;
  esac
done

shift $((OPTIND-1))  

DATASET=$2
BATCHSIZE=$3

#echo $DATASET
#echo $BATCHSIZE

curl localhost:5000/datasets/$DATASET.json|grep directory|cut -d '"' -f 4 |sed s/\\/digits\\/jobs\\/$DATASET// >tmp.txt
DIGITS=`cat tmp.txt`

if [ "$DIGITS" == "" ];then
	echo 'Bad dataset'
	exit
fi

cat $1|sed s/#.*$// >tmp.txt
group_name=`basename -s .py $1`
model_name=`basename -s .py $1`

if [ -a $group_name-model-layers-0.npz ];then
	CUSTOM_NETWORK_SNAPSHOT="-F custom_network_snapshot=`pwd`/$group_name"
	ln $group_name-model-global-0.npz tmp.npz -sf
	EPOCH_BEGIN=`python << EOF
import numpy as np
print np.load("tmp.npz")["arr_0"]
EOF
	`
	model_name=$group_name-$EPOCH_BEGIN
else
	CUSTOM_NETWORK_SNAPSHOT=''
fi
curl -s localhost:5000/login -c digits.cookie -XPOST -F username=$USERNAME >/dev/null
curl -s localhost:5000/models/images/classification.json -b digits.cookie -XPOST -F use_mean=none -F custom_network="$(<tmp.txt)" $CUSTOM_NETWORK_SNAPSHOT -F batch_size=$BATCHSIZE -F method=custom -F batch_accumulation=1 -F solver_type=ADAM -F learning_rate=1e-4 -F train_epochs=100 -F framework=deepstacks -F model_name=$model_name -F group_name=$group_name -F dataset=$DATASET|grep "job id"|cut -d ':' -f 2 |cut -d '"' -f 2 >jobid.txt
ln $DIGITS/digits/jobs/`cat jobid.txt`/deepstacks_output.log $group_name.log -sf
while [ ! -e $DIGITS/digits/jobs/`cat jobid.txt`/snapshot-model-layers-0.npz ];do
	curl -s localhost:5000/models/`cat jobid.txt`/status|grep -E 'Initialized|Running' >status.txt
	if [ "`cat status.txt`" = "" ]; then
		exit
	fi
	sleep 1
done
ln $DIGITS/digits/jobs/`cat jobid.txt`/snapshot-model-layers-0.npz $group_name-model-layers-0.npz -sf
ln $DIGITS/digits/jobs/`cat jobid.txt`/snapshot-model-global-0.npz $group_name-model-global-0.npz -sf
