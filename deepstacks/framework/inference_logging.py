#!/usr/bin/env python
# coding:utf-8
# vi:tabstop=4:shiftwidth=4:expandtab:sts=4

from .main import register_inference_handler
import logging
import json

def my_inference_handler(out,ids):
    for i in range(out.shape[0]):
        probs=out[i]
        logging.info('Predictions for image ' + str(ids[i]) +
                ': ' + json.dumps(probs.tolist()))

register_inference_handler(my_inference_handler)
