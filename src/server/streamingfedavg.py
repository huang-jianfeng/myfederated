from typing import Dict, List

from argparse import Namespace
from copy import deepcopy
from torch.utils.data import DataLoader, Subset

import torch
from fedavg import FedAvgServer
from src.client.streamingfedavg import StreamingFedAvgClient
from src.config.utils import (
    OUT_DIR,
    evaluate,
    # Logger,
    # fix_random_seed,
    # trainable_params,
    # get_best_device,
)
class StreamingFedAvgServer(FedAvgServer):
    
    def __init__(self, algo: str = "StreamingFedAvg", args: Namespace = None, unique_model=False, default_trainer=False):
        super().__init__(algo, args, unique_model, default_trainer)
        
        self.trainer = StreamingFedAvgClient(deepcopy(self.model),self.args, self.logger)
        self.train_results:Dict[int,Dict]={}
        self.test_results:Dict[int,Dict]={}
        self.global_results:Dict[int,Dict]={}
    
    def train(self):
        for E in self.train_progress_bar:
            self.current_epoch = E

            if (E + 1) % self.args.verbose_gap == 0:
                self.logger.log("-" * 26, f"TRAINING EPOCH: {E + 1}", "-" * 26)
                
            self.selected_clients = self.client_sampling(E)
            for client_id in range(self.client_num):
                self.trainer.data_streaming(client_id,0.005)
                
            self.train_one_round()
            if (E + 1) % self.args.test_gap == 0:
                self.test()
                self.log_test_info()

            self.log_info()
                
            # if E == 34:
            #     self.logger.log("debug: rounds#{} selected client{}",E,self.selected_clients)
          
                
            
           
    def custom(self):
        self.save_train_and_test_results()
        self.test_on_global_test()
    
    def test_on_global_test(self):
        if self.args.global_testset:
            loader = DataLoader(self.trainer.global_testset(),self.args.batch_size)
            criterion = torch.nn.CrossEntropyLoss(reduction="sum")
            test_loss, test_correct, test_sample_num = evaluate(
                model=self.model,
                dataloader=loader,
                criterion=criterion,
                device=self.device,
            )
        
    def test(self):
        """The function for testing FL method's output (a single global model or personalized client models)."""
        self.test_flag = True
        loss_before, loss_after = [], []
        correct_before, correct_after = [], []
        train_loss_before,train_loss_after=[],[]
        train_correct_before,train_correct_after=[],[]
        train_num_samples=[]
        num_samples = []
        for client_id in self.test_clients:
            client_local_params = self.generate_client_params(client_id)
            stats = self.trainer.test(client_id, client_local_params)

            correct_before.append(stats["before"]["test_correct"])
            correct_after.append(stats["after"]["test_correct"])
            loss_before.append(stats["before"]["test_loss"])
            loss_after.append(stats["after"]["test_loss"])
            num_samples.append(stats["before"]["test_size"])
            
            
            train_correct_before.append(stats["before"]["train_correct"])
            train_correct_after.append(stats["after"]["train_correct"])
            train_loss_before.append(stats["before"]["train_loss"])
            train_loss_after.append(stats["after"]["train_loss"])
            train_num_samples.append(stats["before"]["train_size"])

        loss_before = torch.tensor(loss_before)
        loss_after = torch.tensor(loss_after)
        correct_before = torch.tensor(correct_before)
        correct_after = torch.tensor(correct_after)
        num_samples = torch.tensor(num_samples)
        
        train_loss_before = torch.tensor(train_loss_before)
        train_loss_after = torch.tensor(train_loss_after)
        train_correct_before = torch.tensor(train_correct_before)
        train_correct_after = torch.tensor(train_correct_after)
        train_num_samples = torch.tensor(train_num_samples)
        
        self.train_results[self.current_epoch + 1] = {
            'before_loss':(train_loss_before.sum() / train_num_samples.sum()).item(),
            'after_loss':(train_loss_after.sum() / train_num_samples.sum()).item(),
            "before_accuracy": (train_correct_before.sum() / train_num_samples.sum() * 100).item(),
            'after_accuracy':  (train_correct_after.sum() / train_num_samples.sum() * 100).item(),
            }
        
        self.test_results[self.current_epoch + 1] = {
            "before_loss": (loss_before.sum() / num_samples.sum()).item(),
            "after_loss":(loss_after.sum() / num_samples.sum()).item(),
            "before_accuracy": (correct_before.sum() / num_samples.sum() * 100).item(),
            "after_accuracy":(correct_after.sum() / num_samples.sum() * 100).item(),
        }
        self.test_flag = False
            
    def log_test_info(self):
        self.logger.log("train_loss:{:<10.6f}(before)->{:<10.6f}(after)".format(self.train_results[self.current_epoch+1]['before_loss'],
                                                           self.train_results[self.current_epoch+1]['after_loss']))
        self.logger.log("train_accuracy:{:<10.6f}(before->{:<10.6f}(after))".format(self.train_results[self.current_epoch+1]['before_accuracy'],
                                                                      self.train_results[self.current_epoch+1]['after_accuracy']))
        self.logger.log("test_loss:{:<10.6f}(before)->{:<10.6f}(after)".format(self.test_results[self.current_epoch+1]['before_loss'],
                                                                 self.test_results[self.current_epoch+1]['after_loss']))
        self.logger.log("test_accuracy:{:<10.6f}(before)->{:<10.6f}(after)".format(self.test_results[self.current_epoch+1]['before_accuracy'],
                                                                     self.test_results[self.current_epoch+1]['after_accuracy']))
        
    def save_train_and_test_results(self):
        import pandas as pd
        import matplotlib.pyplot as plt
        
        plt.clf()
        train_df = pd.DataFrame.from_dict(self.train_results, orient='index')
        plt.plot(train_df.index, train_df['before_loss'],label='before_loss_train')
        
        test_df = pd.DataFrame.from_dict(self.test_results, orient='index')
        plt.plot(test_df.index, test_df['before_loss'],label='before_loss_test')
        
       
        plt.xlabel('Communication Rounds')
        plt.ylabel('loss')
        plt.title(f'{self.algo}_{self.args.dataset}')
        plt.legend()
        plt.savefig(OUT_DIR/self.algo/f"{self.args.dataset}_loss.jpg")
        plt.clf()
        
        
        plt.plot(train_df.index, train_df['before_accuracy'],label='before_acc_train')
        plt.plot(test_df.index, test_df['before_accuracy'],label='before_acc_test')
        
        plt.xlabel('Communication Rounds')
        plt.ylabel('accuracy')
        plt.title(f'{self.algo}_{self.args.dataset}')
        plt.legend()
        plt.savefig(OUT_DIR/self.algo/f"{self.args.dataset}_accuracy.jpg")
        plt.clf()
        
        train_df.to_csv(OUT_DIR/self.algo/f"{self.args.dataset}_train_loss_acc.csv")
        test_df.to_csv(OUT_DIR/self.algo/f"{self.args.dataset}_test_loss_acc.csv")

    def client_sampling(self, E: int) -> List:
        return super().client_sampling(E)
if __name__ == "__main__":
    server = StreamingFedAvgServer()
    server.run()