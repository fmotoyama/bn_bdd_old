# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 21:45:16 2020

@author: motoyama
"""

import time

class stopwatch():
    def __init__(self):
        self.start = time.time()
    
    def rap(self, title):
        now = time.time() - self.start
        print("{0}:{1}".format(title.ljust(10),now) + "[sec]")