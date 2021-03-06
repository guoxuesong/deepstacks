#!/usr/bin/env python
# coding:utf-8
# vi:tabstop=4:shiftwidth=4:expandtab:sts=4


def fields(a):
    res = ()
    count = 0
    for line in a:
        #print count, line
        if type(line[-1]) == set or type(line[-1]) == dict:
            pass
        else:
            line = line+({},)
        while len(line) < 7:
            line = line[:-1]+(0, line[-1])

        # line[-1] in a maybe same if functions in macros not write carefully.
        # because return value of
        #
        # def foo(m={})
        #     return m
        #
        # is same object

        flags = {}
        for k in line[-1]:
            if type(line[-1]) == set:
                flags[k] = True
            else:
                flags[k] = line[-1][k]

        if line[1] is not None and line[1] != 0:
            if type(line[1]) == tuple:
                if len(line[1]) == 0 or line[1][0] != slice:
                    flags['reshape'] = line[1]
                else:
                    flags['slice'] = line[1]
            elif type(line[1]) == slice:
                flags['slice'] = line[1]
            else:
                flags['num_filters'] = line[1]
        if line[2]:
            flags['filter_size'] = line[2]
        if line[3]:
            flags['stride'] = line[3]
        if line[4]:
            flags['push'] = line[4]
        if line[5]:
            flags['sharegroup'] = line[5]
        res += (
                (line[0], flags),
                )
        #print count, res[-1]
        count += 1
    return res
