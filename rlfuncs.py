# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 21:41:59 2018

@author: James
"""

import pandas as pd 
from collections import defaultdict
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import time
import random
import matplotlib.pyplot as plt 
from statistics import stdev
from statistics import mean
  
sym = {'up':'↑', 'le':'←', 'ri':'→', 'dn':'↓'}
rev = {'↑':'up', '←':'le', '→':'ri', '↓':'dn', 'G':'up', 'X':'up'}

def map_grid(env, width, dec=3, title=''):
  # width: width of world (set when \n inserted)
  # dec : number of decimals places to show
  grid = '\n'; vals = ''
  for s in env:
    if s==0: continue
    pol = env[s]['pol']
    if pol in sym:
      pol = sym[pol]
    grid = grid + ' ' + pol 
    val = ' %.' + str(dec) + 'f'
    v = (val % env[s]['u'])
    v = v.replace('-0.','-.')
    v = v.replace(' 0.','  .')
    v = v.replace('-1.00','-1.0')
    vals = vals + v
    if s%width==0:
      grid = grid + '  ' + vals + '\n'
      vals=''
  if title>'':
    grid = '\n' + title + grid
  return grid
#print(map_grid(env, 4))

def value_iteration(env, movement, iter, gamma):
  start_time = time.time()
  u = {'up':0,'dn':0,'le':0,'ri':0}
  U = defaultdict(int)
  cols = list(env.keys())
  hist = pd.DataFrame(columns=cols)
  for t in range(iter):
    for sk, sv in env.items():
      # iterate over env states
      for ak, av in movement.items():
        # iterate over intended actions
        u[ak]=0
        for m in av:
          # Iterate over actual movements resulting from intended action.
          # Find expected utility given:
          # - actual movements and that probability: m[1], and 
          # - the utility the the new state: env[sv[m[0]]]['u']
          dir = m[0]
          ns = sv[dir]
          u[ak] += (m[1] * env[ns]['u'])
          # gives expected utility of each intended action (policy)
      # return max (best value) and argmax (best policy) from utilities
      U[sk] = ((max(u.values()) * gamma) + env[sk]['rew'], max(u, key=lambda key: u[key]))

    # After finding new expected best utilities (stored in U), update 
    converged = True
    for sk, sv in env.items():
      if abs(env[sk]['u'] - U[sk][0]) > 1e-7: converged = False
      env[sk]['u'] = U[sk][0]
      if env[sk]['pol'] in '↑ ← → ↓':
        pol = U[sk][1]
        env[sk]['pol'] = sym[pol]
    hist.loc[t]= pd.Series([(env[k]['pol'], round(env[k]['u'],2)) for k in env], cols)
    if converged:
      print ('converged after' , t, 'iterations')
      break
  if not converged:
    print ('failed to converge by', t+1, 'iterations')
  end_time = time.time()
  print("--- Value Iteration %.2f seconds ---" % (end_time - start_time))
  env[0]['iter']=t
  env[0]['gamma']=gamma
  return env, hist, t

def calc_U_dense(env, movement, gamma):
  # solve for U simultaneously, for policy evaluation phase of policy iteration
  n = len(env)
  T = np.zeros([n,n])     # n x n matrix of transition probabilities for current policies
  r = np.zeros([n,1])     # n x 1 matrix of rewards
  # populate transition matrix
  dir = {v:k for k,v in sym.items()}
  for sk, sv in env.items():
    r[sk] = sv['rew']
    if sk==0: continue
    pol = sv['pol']
    if pol in dir:
      pol = dir[pol]
    if not(pol in movement): continue
    for mv in movement[pol]:
      # set probability for possible movements for selected policy
      d = mv[0]         # direction
      ns = sv[d]      # next state
      T[sk,ns] = T[sk,ns] + mv[1]  # update T with the probability
  # U = (I - gamma x T) r
  I = np.identity(n)

  U = np.linalg.inv(I -(gamma * T)) @ r
  return U

def calc_U(env, movement, gamma):
  # solve for U simultaneously, using sparse (coo) matrices
  # (for policy evaluation phase of policy iteration)
  n = len(env)
  r = np.zeros([n,1])     # n x 1 matrix of rewards
  coo_vals = []
  coo_rows = []
  coo_cols = []  
  # dictonary to translate symbols (arrows) to text 'up' 'dn' etc. 
  dir = {v:k for k,v in sym.items()}
  # build transition matrix T (as sparse coo matrix)
  for sk, sv in env.items():
    # for each state:
    r[sk] = sv['rew']
    if sk==0: continue
    pol = sv['pol']
    if pol in dir:
      pol = dir[pol]
    if not(pol in movement): continue
    ps = defaultdict(int)
    for mv in movement[pol]:
      # set probability for possible movements for selected policy
      d = mv[0]         # direction
      ns = sv[d]      # next state
      ps[ns] += mv[1] 
    for k,v in sorted(ps.items(), key=lambda tup: tup[0]):
      coo_rows.append(sk)
      coo_cols.append(k)
      coo_vals.append(v)
  # U = (I - gamma x T) r
  #print(  list(zip(coo_vals, coo_rows, coo_cols)))
  T = sparse.coo_matrix((coo_vals, (coo_rows,coo_cols)), shape=(n,n))
  I = sparse.identity(n, format='coo')
  A = I -(gamma * T)
  u = spsolve(A, r)
  return u


def policy_iteration(env, movement, iter, gamma):
  start_time = time.time()
  u = {'up':0,'dn':0,'le':0,'ri':0}
  P = defaultdict(int)
  cols = list(env.keys())
  hist = pd.DataFrame(columns=cols)
  for t in range(iter):
    # policy evaluation: update utility values
    converged = True
    U = calc_U(env, movement, gamma)
    for s,v in enumerate(U):
      if abs(env[s]['u'] - U[s]) > 1e-7: converged = False
      env[s]['u'] = v
    if converged:
      print ('converged after' , t, 'iterations')
      break
    # policy improvement
    for sk, sv in env.items():
      # skip terminal states:
      if sv['up']==0: continue
      # for each non-terminal state:
      for ak, av in movement.items():
        # iterate over intended actions
        u[ak]=0
        for m in av:
            # find expected utility of each intended action (policy)
            dir = m[0]
            ns = sv[dir]
            u[ak] += (m[1] * env[ns]['u'])
        # return max (best value) and argmax (best policy) from utilities
        P[sk] = ((max(u.values()) * gamma) + env[sk]['rew'], max(u, key=lambda key: u[key]))
  
    # After finding new expected best policies (stored in P), update 
    for sk, sv in env.items():
      if sk in P:
        pol = P[sk][1]
        env[sk]['pol'] = sym[pol]
    hist.loc[t]= pd.Series([(env[k]['pol'], round(env[k]['u'],2)) for k in env], cols)
  if not converged:
    print ('failed to converge by', t+1, 'iterations')
  end_time = time.time()
  print("--- Policy Iteration %.3f seconds ---" % (end_time - start_time))
  env[0]['iter']=t
  env[0]['gamma']=gamma
  return env, hist, t

def actual_move(dir, movement):
  # actual move made given intent to take a specified action
  mv = np.random.random()
  for m in movement[dir]:
    tp = m[1]
    if mv < tp:
      mv = m[0]; break
    else:
      mv -= tp
  return mv
  
def Q_learn(env, movement, iter, gamma, Q={}, hist=[], eps=0.2, init_state=0, 
            alpha_decay=50, eps_decay=0.996, hist_int=5000, explore_until=200, rho=.005):
  # alpha decay: high (~50) equals slower; alpha=1/t; t(s,a) increments by 1/alpha decay each visit
  # eps_decay (0-1): high equal slower; eps = eps * eps_decay at end of each trial/run
  # init_state: set starting point, or set to zero to use random starting points
  # Q, hist: supply if continuing exploration after exhausting iterations
  # explore_until: artificially grant reward up until this many visits have been experienced
  # rho: proportion of max_rew to encourage exploration if (visits<explore_until)
  start_time = time.time()
  q = {'up':0,'dn':0,'le':0,'ri':0}
  # initialize Q and randomize initial start directions
  actions = list(q.keys())
  if Q=={}:
    # initialize new Q/experience if not provided
    max_rew = 0
    for s in env:
      for a in movement:
        Q[(s,a)] = 0
      if env[s]['pol'] in '↑ ← → ↓ up dn le ri':
        pol = random.choice(actions)
        env[s]['pol'] = pol
        Q[(s,pol)] = random.random()/100
    experience = {(s,a):[0,0,0] for s,a in Q}     # (chosen, policy, actual dir)
    beg = 1
  else:
    # or reset based on Q provided
    experience = {k:v[0] for k,v in Q.items()}
    Q = {k:v[1] for k,v in Q.items()}
    beg = hist[-1][0] + 1
    max_rew = max(Q.values())
  chgs=0
  explorations=0
  
  def best_Q(next_states):
    # find max Q(s',a') for all a, for each state we can move to
    bq = {s:0 for s in next_states}
    for s in bq:
      nq = 0
      for a in movement:
        nq = max(Q[(s,a)], nq)
      bq[s] = nq 
    return bq  

  for run in range(beg, iter+beg):
    # restart exploration
#    nts = [k for k,v in env.items() if v['up']!=0]   # non-terminal states
    nts = [k for k,v in env.items() if v['pol'] in '↑ ← → ↓ up dn le ri']   # non-terminal states
    if init_state==0:
      cs = np.random.choice(nts)
    else:
      cs = init_state
    i=0
#    # find (s,a) pair from visited states that has lowest experience
#    min_visits = 9e99
#    for s in env:
#      mvs = 9e99
#      for a in actions:
#        n = experience[s,a][0]
#        if (n>0) & (n<mvs):
#          mvs = n
#      if mvs < min_visits:
#        min_visits = mvs

    while i >=0:
      # explore in steps/iterations i
      # find max Q(s',a') for each state we can move to
      bq = best_Q({env[cs]['up'], env[cs]['dn'], env[cs]['le'], env[cs]['ri']})

      # q value for each action/movement from current state
      for a in q:
        q[a] = Q[cs,a] 
      # find policy pointing to best q values for our (s,a)
      pol = max(q, key=lambda key: q[key])      # best move based on current Q (highest q value)
      if (env[cs]['up']!=0) & (env[cs]['pol']!=pol):
        env[cs]['pol'] = pol
        chgs += 1
      experience[cs,pol][1] += 1                      # best policy direction

      # choose to explore randomly with Pr(eps), in direction other than best policy
      # first, decay epsilon based on available action that has been executed least
      m = min([experience[cs,a][0] for a in movement]) + 1
      eps_adj = eps * eps_decay**m
#      eps_adj = eps * eps_decay**min_visits
      # now set dir = pol and check for random alternative movement based on adjusted epsilon
      dir = pol
      if np.random.random() < eps_adj:
        dir = random.choice(list({k for k in movement} - {pol}))
        explorations += 1
      experience[cs,dir][0] += 1                      # intended direction after epsilon

      # find actual movement based on selected direction (may end up elsewhere due to stochasticity)
      mv = actual_move(dir, movement)
      experience[cs,mv][2] += 1                       # actual direction

      # complete move to new state
      ps = cs                                         # prior state 
      cs = env[cs][mv]                                # new current state
      
      rew = env[ps]['rew']
      # record maximum reward found so far
      if rew > max_rew:
        max_rew = rew

      # update Q(s,a) based on reward from prior state s and Q from new state s'
      # note: a is intended direction, not necessarily actual direction!
      # t is based on t(s,a), the number of times this (s,a) has been explored,
      # but reduced after t=5 (to slow down decay of alpha)
      t = experience[ps,dir][0]

      # artificially set/enhance reward if (s,a) pair not visited more than threshold set by explore_until
      if t < explore_until:
        rew = max_rew * rho
      if t>3:
        t = 3 + ((t-3)/alpha_decay)              
      alpha = 1/t
      x = rew + (gamma * bq[cs])
      x = min(x, max(max_rew, 0.01)) 
      Q[ps,dir] = ((1 - alpha) * Q[ps,dir]) + (alpha * x)

      # check for termination of this run
      if cs==0:
        Q[ps,dir] = env[ps]['rew']
        break   # quit this run and move to next
      i += 1
      if i > (len(env)**3):
        # maxed iterations for this run (unlikely, but just in case)
        msg = 'run %s maxed iterations at %s after %s steps' % (run, cs, i)
        print (msg)
#        hist.append([msg, (run, cs, i)])
        break   # quit this run and move to next
    if run%hist_int==0:
      print ('%s trials, +%s policy changes, %s explorations, %.0f s' % (run,chgs,explorations, time.time()-start_time))
      hist.append((run, chgs, explorations, round(time.time()-start_time),1))
      chgs=0
      explorations=0

  # update utility based on max(Q) 
  for s in env:
    env[s]['u'] = max([q for k,q in Q.items() if k[0]==s])

  # add experience to Q
  for k in Q:
    Q[k] = (experience[k], Q[k])
  end_time = time.time()
  print("--- Q Learning %.3f seconds, %s trials, gamma=%s ---" % (end_time - start_time, iter, gamma))
  # record parameters in env[0]
  env[0]['iter']=iter
  env[0]['gamma']=gamma
  env[0]['init_state']=init_state
  env[0]['alpha_decay']=alpha_decay
  env[0]['eps_decay']=eps_decay
  return env, hist,  Q
  
def play(env, movement, init_state, iter=100000):
  # run the policy selected by an environment
  results = {}
  for i in range(iter):
    # start new game
    cs = init_state
    rew = 0
    moves = 0
    while cs!=0:
      pol = env[cs]['pol']
      if pol in rev:
        pol = rev[pol]
      if pol in sym:
        mv = actual_move(pol, movement)
        ps = cs
        cs = env[cs][mv]
        rew += env[cs]['rew']
        moves += 1
    results[i] ={'end':ps, 'fate':env[ps]['pol'], 'rew':rew, 'moves':moves}
  wins = sum([1 for i in results if results[i]['fate']=='G'])
  rew_win =  sum([results[i]['rew'] for i in results if results[i]['fate']=='G'])
  rew_loss =  sum([results[i]['rew'] for i in results if results[i]['fate']!='G'])
  mv_av = round(mean([results[i]['moves'] for i in results]),1)
  mv_sd = round(stdev([results[i]['moves'] for i in results]),1)
#  rew_tot =  sum([results[i]['rew'] for i in results])
  return wins, rew_win, rew_loss, mv_av, mv_sd
  
      
def plot_QL(hist, t='QL - total changes accumulated between plot points', h=2, w=3, xlab='trials'):
  # plot the history output from Q_learn
  fig = plt.figure(t, figsize=(w, h))
  plt.clf()
  x = [x[0] for x in hist]
  c = [x[1] for x in hist]
  e = [x[2]/1 for x in hist]
  ax = fig.add_subplot(111)
  ax.plot(x, c,'-o', markersize=3)
  ax.plot(x, e,'-o', markersize=3)
  plt.title(t, size='small')
  plt.legend(['Policy changes','Explorations (1s)'], fontsize='x-small', frameon=False, title='')
  plt.xlabel(xlab)
  plt.tick_params(labelsize=8)
  return fig
