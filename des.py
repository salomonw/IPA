# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import simpy as sy
import numpy as np 
import random as rnd
import matplotlib.pyplot as plt
import sys


def gen_inter_arrival(lambda_arr):
    return np.random.exponential(lambda_arr)

def generate_service(lambda_ser):
    return np.random.exponential(lambda_ser)

def sim(env,servers,lambda_arr,lambda_ser):
    i = 0
    while True:
        i += 1
        yield env.timeout(gen_inter_arrival(lambda_arr))
        env.process(arrival(env,i,servers,q_length,max_q_len))

wait_t = []

def arrival(env,arrival,servers,q_length,max_q_len):
    with servers.request() as request:
        if queue_lent_calc(servers.queue)>=max_q_len+1: 
            t_kout = env.now
            #print env.now, '{} kicked out'.format(arrival)
            env.exit()
            
        else:
            t_arrival = env.now
            #print env.now, '{} arrives'.format(arrival)
            yield request
            #print env.now, '{} being served' .format(arrival)
            yield env.timeout(generate_service(lambda_ser))
            #print env.now, '{} departs'.format(arrival)
            t_depart = env.now
            wait_t.append(t_depart-t_arrival)
            


def queue_lent_calc(servers):
    return len(servers)


obs_times = []
q_len = []
event_1 = []
ev_1=[]
q_length  = 0
event_2 = []
ev_2=[] 
cost = []
C=[]
T=[]
tao = []


def observe(env,servers,time_to_obs,max_q_len,overf_cost):
    i = -1
    C.append(0)
    T.append(0)
    tao.append(0)

    cost.append(0)
    while True:
        event_1.append(0)
        event_2.append(0)
        i += 1
        #print(i)
        obs_times.append(env.now)
        q_length =  queue_lent_calc(servers.queue)
        #q_length = len(servers.queue)
        q_len.append(q_length)
        
        
        # Event type 1 
        if q_len[i] == 0:
            if event_1[i-1] == 0 and q_len[i-1]>0:
                event_1[i]=1
                if tao[0]>0:
                    C[0] = C[0]-1
                    T[0] = T[0] + (env.now-tao[0])
                    tao[0] = 0
            else:
                event_1[i]=0
        else:
            ev_1.append(0)
            event_1[i]=0
        yield env.timeout(time_to_obs)
        
        # Event type 2
        if q_len[i] == max_q_len:
            if event_2[i-1] == 0 and q_len[i-1]<max_q_len:
                event_2[i]=1
                tao[0] = env.now
                cost[0] += overf_cost
            else:
                event_2[i]=0
                cost[0] += overf_cost
        else:
            ev_2.append(0)
            event_2[i]=0
        yield env.timeout(time_to_obs)
        
        # Last time      
        if env.now==int_ipa_time-1 and tao>0:
            C[0] = C[0]-1
            T[0] = T[0] + (env.now-tao[0])
        dat = np.matrix('%s %s %s %s' % (env.now,C[0],T[0],tao[0]))
        #print dat
        #print cost
        #cost += q_len
        #yield cost
    
# Parameters
        
        
lambda_arr=1./10.0
lambda_ser=1./10.0
cap = 1 # capacity of servers
gen_inter_arrival(lambda_arr)
time_to_obs = 0.5
int_ipa_time = 100
overf_cost = 100
q_len_cost = 1
R = 50

np.random.seed(1)

results = []
num_iteration = 1

max_q_len = 15

for i in range(0,num_iteration):
    wait_t = []
    obs_times = []
    q_len = []
    event_1 = []
    ev_1=[]
    event_2 = []
    ev_2=[] 
    cost = []
    C=[]
    T=[]
    tao = [] 
    cost = []
    cost.append(0)
    C.append(0)
    T.append(0)
    tao.append(0)
    np.random.seed(1)
    env = sy.Environment()
    
    servers = sy.Resource(env, capacity=cap)
    
    env.process(sim(env,servers,lambda_arr,lambda_ser))
    
    env.process(observe(env,servers,time_to_obs,max_q_len,overf_cost))
    
    env.run(until=int_ipa_time)
    
    cost[0] += np.mean(q_len)*q_len_cost
    
    results.append([i,cost[0],max_q_len])
    
    #results.append(np.matrix('%s %s %s' % (i,cost[0],max_q_len)))
    
    max_q_len -= (T[0] + R*C[0])/int_ipa_time
    

    plt.figure()
    plt.hist(wait_t)
    plt.xlabel('Waiting time')
    plt.ylabel('Number of customers')
    plt.show()
    
    plt.figure()
    plt.step(obs_times, q_len , where='post')
    plt.xlabel('Time')
    plt.ylabel('Queue length')
    plt.show()
    
print results
