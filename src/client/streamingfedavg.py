from argparse import Namespace
from typing import Dict, List
import numpy as np
from copy import deepcopy
from functools import cmp_to_key
from config.models import DecoupledModel
from config.utils import Logger
from fedavg import FedAvgClient

class StreamingFedAvgClient(FedAvgClient):
    
    def __init__(self, model: DecoupledModel, args: Namespace, logger: Logger):
        super().__init__(model, args, logger)
     
        self.counters:Dict[int,List[int]] = {}
        
        self.init_counter:List[int] = [0]*args.counter_capacity
        self.counter_capacity = args.counter_capacity

        total_class = len(self.dataset.classes)
        for i,index in enumerate(self.data_indices):
            
            index['train']=np.array(sorted(index['train'],key=cmp_to_key(lambda a,b: 
                self.dataset.targets[a]-self.dataset.targets[b])))
            split_class = i % total_class
            split_point = 0
            
            for j,pos in enumerate(index['train']):
                if self.dataset.targets[pos] == split_class:
                    split_point = j
                    break
            index['train']=np.concatenate((index['train'][split_point:],index['train'][0:split_point]))
            
        self.total_data_indices = deepcopy(self.data_indices)
        for i,index in enumerate(self.data_indices):
            index['train'] = np.zeros(0,dtype=int)
        
        logger.log("StreamingFedAvgfClient init.")
    
    
    def data_streaming(self,client_id:int,add_fraction:float):
        """模拟数据流到达

        Args:
            add_fraction (float): 增加add_float的数据
            client_id(int): id
        """
        self.client_id = client_id
        total_num = len(self.total_data_indices[self.client_id]['train'])
        cur_num = len(self.data_indices[self.client_id]['train'])
        new_num = min(int(total_num*add_fraction+cur_num),total_num)
        
        cache_indices = []
        if self.client_id not in self.counters.keys():
            self.counters[self.client_id] = deepcopy(self.init_counter)
        counter = self.counters[self.client_id]
        for indice in self.total_data_indices[self.client_id]['train'][cur_num:new_num]:
            # if indice ==6223:
            #     print(type(indice))
            cache_indices.append(indice)
            _,y = self.dataset[indice]
            counter[y%self.counter_capacity] += 1
        
        
        cache_indices = np.array(cache_indices,dtype=int)
        self.data_indices[self.client_id]['train'] = np.append(self.data_indices[self.client_id]['train'],cache_indices)
        
    def load_dataset(self):
        
        return super().load_dataset()
        
        
            