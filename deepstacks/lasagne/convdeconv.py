#!/usr/bin/env python
# coding:utf-8
# vi:tabstop=4:shiftwidth=4:expandtab:sts=4

from .implement import register_macro_handler
from .implement import *
from ..macros import macros
from ..fields import fields

register_macro_handler(fields)
register_macro_handler(macros)
