from bliss import setup_globals
from bliss.setup_globals import *
import gevent
import time
import datetime
import sys
import numpy as np
from xmlrpc.client import ServerProxy
from gevent import sleep

eiger4m_v2.processing.saving_sparse.file_exists_policy='overwrite'

def now():
    return str(datetime.datetime.now())

def myprint(*args):
    print(now(),*args)


def take_data_and_move(nimages,dt=0.01,dz=0.02,n_moves=4):
    period = dt+120e-6
    initial_zs = zs.position
    def wait_and_move():
        for _i in range(n_moves-1):
            gevent.sleep(period*nimages/n_moves)
            umvr(zs,dz)
            myprint(f"New zs position {zs.position:.3f}")
    gwait_and_move = gevent.spawn(wait_and_move)
    fasttimescan(nimages, dt, save=True)
    umv(zs,initial_zs)
    print("Waiting for backgroud job to finish ...",end="")
    gwait_and_move.join()
    print("done")
    return gwait_and_move


   
    
def take_data(exp_time=0.001,n_frames=20_000):
    exp_pil = max(1,int(n_frames*exp_time/60_000+1))
    mtimescan(exp_time,int(n_frames),exp_pil)
    
    
def qscan_macro():
    delcoups = [1.7, 3.5, 8]
    Zs = [2.7, 2.8, 2.9]
    num_frames = 16_000_000
    
    itime = 0.001
    
    for ii in range(len(delcoups)):
        print("Measuremnt at delcoup", delcoups[ii], "for ", num_frames*itime/3600, "hours")
        umv(delcoup, delcoups[ii])
        umv(zs, Zs[ii])
        take_data(itime, num_frames)
     
