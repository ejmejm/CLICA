import datasets
import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.pin_memory import pin_memory
from torch.utils.data.dataloader import _SingleProcessDataLoaderIter
from transformers.trainer import seed_worker, Trainer as HFTrainer


class BufferedSequentialIterator(_SingleProcessDataLoaderIter):
    def __init__(self, dataloader: DataLoader):
        super().__init__(dataloader)
        # We will manually apply the collate function.
        # Overriding it here will ensure it is not called during the call to `fetch`.
        self._dataset_fetcher.collate_fn = lambda x: x
        self.sequence_buffer = [[] for _ in range(dataloader.batch_size)]
        self.final_sequence_loaded = False

    def _fill_sequence_buffer_if_empty(self):
        if self.final_sequence_loaded:
            return
        
        for i in range(len(self.sequence_buffer)):
            if len(self.sequence_buffer[i]) == 0:
                try:
                    self.sequence_buffer[i] = self._fetch_single_sequence()
                except StopIteration:
                    self.final_sequence_loaded = True
    
    def _fetch_single_sequence(self):
        index = self._next_index()  # may raise StopIteration
        data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
        if self._pin_memory:
            data = pin_memory(data, self._pin_memory_device)
        return data

    def _next_data(self):
        self._fill_sequence_buffer_if_empty()
        batch = [
            self.sequence_buffer[i].pop(0)
            for i in range(len(self.sequence_buffer))
            if len(self.sequence_buffer[i]) > 0
        ]

        if len(batch) == 0:
            raise StopIteration

        return self.collate_fn(batch)


class SequentialDataloader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_sampler = None
        
    def _get_iterator(self):
        return BufferedSequentialIterator(self)


# class Trainer(HFTrainer):
#     def __init__(self, *args, dataloader_cls=DataLoader, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.dataloader_cls = dataloader_cls

#     def get_train_dataloader(self) -> DataLoader:
#         """
#         Returns the training [`~torch.utils.data.DataLoader`].

#         Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
#         training if necessary) otherwise.

#         Subclass and override this method if you want to inject some custom behavior.
#         """
#         if self.train_dataset is None:
#             raise ValueError("Trainer: training requires a train_dataset.")

#         train_dataset = self.train_dataset
#         data_collator = self.data_collator
#         if isinstance(train_dataset, datasets.Dataset):
#             train_dataset = self._remove_unused_columns(train_dataset, description="training")
#         else:
#             data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

#         dataloader_params = {
#             "batch_size": self._train_batch_size,
#             "collate_fn": data_collator,
#             "num_workers": self.args.dataloader_num_workers,
#             "pin_memory": self.args.dataloader_pin_memory,
#             "persistent_workers": self.args.dataloader_persistent_workers,
#         }

#         if not isinstance(train_dataset, torch.utils.data.IterableDataset):
#             dataloader_params["sampler"] = self._get_train_sampler()
#             dataloader_params["drop_last"] = self.args.dataloader_drop_last
#             dataloader_params["worker_init_fn"] = seed_worker
#             dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

#         return self.accelerator.prepare(self.dataloader_cls(train_dataset, **dataloader_params))


# class SequentialTrainer(Trainer):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, dataloader_cls=SequentialDataloader, **kwargs)
