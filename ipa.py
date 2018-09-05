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
            n_kout[0] += 1
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
        obs_times.append(env.now)
        q_length =  queue_lent_calc(servers.queue)
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
        if q_len[i] >= np.floor(max_q_len):
            if event_2[i-1] == 0 and q_len[i-1]<np.floor(max_q_len):
                event_2[i]=1
                tao[0] = env.now
                cost[0] += overf_cost
            else:
                event_2[i]=0
        else:
            ev_2.append(0)
            event_2[i]=0
        yield env.timeout(time_to_obs)
        
        # Last moment      
        if env.now==int_ipa_time-time_to_obs and tao>0:
            C[0] = C[0]-1
            T[0] = T[0] + (env.now-tao[0])
        dat = np.matrix('%s %s %s %s' % (env.now,C[0],T[0],tao[0]))
        #print dat
        
#----------------------------------------------     
# Parameters        
lambda_arr=1./5.0
lambda_ser=1./5.0
cap = 1 # capacity of servers
gen_inter_arrival(lambda_arr)
time_to_obs = 2 # mod = 0.5
int_ipa_time = 5000
overf_cost = 25
q_len_cost = 2
R = 25
max_q_len = 25
results = []
num_iteration = 35

for i in range(0,num_iteration):
    np.random.seed(2)
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
    n_kout = []
    cost.append(0)
    C.append(0)
    T.append(0)
    tao.append(0)
    n_kout.append(0)
    env = sy.Environment()
    
    servers = sy.Resource(env, capacity=cap)
    
    env.process(sim(env,servers,lambda_arr,lambda_ser))
    
    env.process(observe(env,servers,time_to_obs,max_q_len,overf_cost))
    
    env.run(until=int_ipa_time)
    
    cost[0] += np.mean(q_len)*int_ipa_time*q_len_cost + n_kout[0]*R
    
    gradient = float (T[0] + R*C[0])/int_ipa_time
    
    gamma = 8
    
    results.append([i,cost[0],T[0],C[0],gradient,max_q_len])
    results_mat = np.array(results)
    
    max_q_len += -gamma*gradient
    
"""
    plt.figure()
    plt.hist(wait_t)
    plt.xlabel('Waiting time')
    plt.ylabel('Number of customers')
    
    plt.figure()
    plt.step(obs_times, q_len , where='post')
    plt.xlabel('Time')
    plt.ylabel('Queue length')
"""
plt.figure()
plt.plot(results_mat[:,5],results_mat[:,1])
plt.xlabel('Buffer Size')
plt.ylabel('Cost')
plt.show()
print results
