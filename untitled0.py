# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 10:26:45 2019

@author: yifei
"""

import os

if __name__ == "__main__":
    
    path = r'C:\Users\yifei\Desktop\temp\temp.txt'
    outpath = r'C:\Users\yifei\Desktop\temp\temp_new.txt'
    outfile = open(outpath, 'w')
    file = open(path, 'r', encoding='UTF-8')
    lines = file.readlines()
    for line in lines:
        print(line)
        items = line.split('\t')
        if "-" in items[3]:
            head = items[3].split('-')[0]
            tail = items[3].split('-')[1]
            new_item = tail + '-' + head
        else:
            new_item = items[3]
        items[3] = new_item
        for k in range(0, len(items)):
            if k < len(items)-1:
                outfile.write(items[k] + '\t')
            else:
                outfile.write(items[k])
        outfile.write('\n')
    outfile.close()
    file.close()