import psutil, getpass
import pandas as pd

WHITE_LIST = ['decontot', 'armstonj']

def list_processes_by_user(username):
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'username']):
        try:
            if proc.info['username'] == username:
                processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return processes

def tell_user(n_max=10, mem_max=50, verbose=True):
    user = getpass.getuser()   
    processes = list_processes_by_user(user)
        
    ulist = []
    for proc in processes:
        if proc.name() == 'python' and '--multiprocessing-fork' in proc.cmdline():
            obj = dict(pmem = proc.memory_percent(), pcpu = proc.cpu_percent())
            ulist.append(obj)
    
    if len(ulist) == 0:
        if verbose: 
            print(f"no parallel forks spawned by {user}")
        return False
            
    udf = pd.DataFrame(ulist)
    sum_mem = udf.pmem.sum()
    mean_cpu = udf.pcpu.mean()
    n_cpu = len(udf)
    
    if verbose:
        print(f"user `{user}` already spawned {n_cpu} parallel forks: {mean_cpu:.2f}% average cpu load and {sum_mem:.2f}% used memory")
    return n_cpu >= n_max and sum_mem >= mem_max

def clear_user():
    return getpass.getuser() in WHITE_LIST