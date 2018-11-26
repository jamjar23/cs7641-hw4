# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 21:41:59 2018

@author: James
"""

import pandas as pd 
import timeit
import os
this_dir='D:\\GoogleDrive\\_Study\\GT_ML\\hw4'
os.chdir(this_dir)
from rlworlds import reset_state_Class_example
from rlworlds import reset_state_frozen_lake_4x4
from rlworlds import reset_state_frozen_lake_8x8
from rlworlds import reset_state_suhail
from rlworlds import reset_state_ML_Hacker_11x22
from rlfuncs import map_grid
from rlfuncs import value_iteration
from rlfuncs import policy_iteration
from rlfuncs import Q_learn
from rlfuncs import play
from rlfuncs import plot_QL

# change display settings
pd.set_option('display.width', 999)
pd.set_option('display.max_columns', 99)
pd.set_option('display.max_rows', 40)
#np.set_printoptions(precision=3, suppress=True)
#pd.options.display.float_format = lambda x: '{:,.0f}'.format(x) if x > 1e3 else '{:,.2f}'.format(x)
pd.options.display.float_format = lambda x: '{:,.2f}'.format(x)

  
# probabilistic model of intended actions and actual movements
# lecture example
mv_lecture = {'up':[('up',0.8),('le',0.1),('ri',0.1)],
            'dn':[('dn',0.8),('ri',0.1),('le',0.1)],
            'le':[('le',0.8),('dn',0.1),('up',0.1)],
            'ri':[('ri',0.8),('up',0.1),('dn',0.1)]}
# FrozenLake
mv_ice = {'up':[('up',0.334),('le',0.333),('ri',0.333)],
            'dn':[('dn',0.334),('ri',0.333),('le',0.333)],
            'le':[('le',0.334),('dn',0.333),('up',0.333)],
            'ri':[('ri',0.334),('up',0.333),('dn',0.333)]}
# Perfect movement
mv_perfect = {'up':[('up',1.0),('le',0.0),('ri',0.0)],
            'dn':[('dn',1.0),('ri',0.0),('le',0.0)],
            'le':[('le',1.0),('dn',0.0),('up',0.0)],
            'ri':[('ri',1.0),('up',0.0),('dn',0.0)]}
gamma = 0.95

env = reset_state_suhail(lc=1)
env, hist, _ = value_iteration(env, movement, 2000, 0.01)
print(map_grid(env, 2))

env = reset_state_Class_example(lc=0.04)


''' 4x4 LAKE:
    S F F F     1  2  3  4     S - Start
    F X F X     5  6  7  8     G - Goal (finish)
    F F F X     9 10 11 12     F - Frozen
    X F F G    13 14 15 16     X - Hole (dead)
'''
env = reset_state_frozen_lake_4x4(lc=0.0)
env, hist, _ = value_iteration(env, movement, 2000, 0.9)
print(map_grid(env, 4))

%timeit calc_U(env, 0.9)
%timeit calc_U_dense(env, 0.9)

envPI = reset_state_frozen_lake_4x4(lc=0.00)
envPI, hist, _ = policy_iteration(envPI, movement, 36, 0.8)
print(map_grid(envPI, 4))

envQ = reset_state_frozen_lake_4x4(lc=0.0)
envQ, hist, Q = Q_learn(envQ, movement, iter=15000, gamma=0.999, eps=0.3, init_state=0)
print(map_grid(envPI, 4))
print(map_grid(envQ, 4))

hist[-20:]

hist[[1,2,3,4,5,7,9,10,11,14,15]][:10].transpose()
pd.DataFrame(env)

''' 8x8 LAKE:                     
    S F F F F F F F    1  2   3   4   5   6   7   8    S - Start
    F F F F F F F F    9 10  11  12  13  14  15  16    G - Goal (finish)  
    F F F X F F F F   17 18  19  20x 21  22  23  24    F - Frozen 
    F F F F F X F F   25 26  27  28  29  30x 31  32    X - Hole (dead)
    F F F X F F F F   33 34  35  36x 37  38  39  40    
    F X X F F F X F   41 42x 43x 44  45  46  47x 48
    F X F F X F X F   49 50x 51  52  53x 54  55x 56
    F F F X F F F G   57 58  59  60x 61  62  63  64
'''
movement = mv_ice
envVI = reset_state_frozen_lake_8x8(lc=0.00)
envVI, hist, _ = value_iteration(envVI, movement, 2000, 0.999)
print(map_grid(envVI, 8))

envPI = reset_state_frozen_lake_8x8(lc=0.00)
envPI, hist, _ = policy_iteration(envPI, movement, 100, 0.999)
print(map_grid(envPI, 8))

envQ = reset_state_frozen_lake_8x8(lc=0.0)
envQ, hist, Q = Q_learn(envQ, movement, iter=55000, gamma=0.999, eps=0.3, init_state=0)
print(map_grid(envPI, 8))
print(map_grid(envQ, 8))

def print_visits(env, Q):
  msg=''
  for s in env:
    v = sum([Q[s,v][0][0] for v in movement])
    if env[s]['pol'] in 'updnleri':
      if v>999:
        v = ' {0:1.1e}'.format(v)
        v = v.replace('e+0','e')
      else:
        v = ' {:5d}'.format(v)
      msg = msg + v
    else:
      msg = msg + '     ' + envQ[s]['pol']
    if s%22==0:
      msg += '\n'
  print(msg)

movement = mv_perfect
movement = mv_ice
envVI = reset_state_ML_Hacker_11x22(lc=0)
envVI, hist, _ = value_iteration(envVI, movement, 2000, 0.999)
print(map_grid(envVI, 22, dec=1, title='Value Iteration Results'))

envPI = reset_state_ML_Hacker_11x22(lc=0.00)
envPI, hist, _ = policy_iteration(envPI, movement, 100, 0.999)
print(map_grid(envPI, 22, dec=1, title='Policy Iteration Results'))

envQ = reset_state_ML_Hacker_11x22(lc=0.0); Q={}; histQ=[]
envQ, histQ, Q = Q_learn(envQ, movement, iter=50000, gamma=0.9999, 
                        Q=Q, hist=histQ, eps=0.6, 
                        init_state=1, alpha_decay=70, eps_decay=0.997)
print(map_grid(envQ, 22, dec=2, title='Q-Learning Results'))
#print(map_grid(envPI, 22, dec=2, title='Policy Iteration Results'))
print_visits(envQ, Q)
plot_QL(histQ, w=5, h=4)

print(play(envPI, movement, init_state=1))
print(play(envQ,  movement, init_state=1))
