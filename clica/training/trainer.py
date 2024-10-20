from typing import Any, Dict, List

from datasets import Dataset
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import _utils
from torch.utils.data._utils.pin_memory import pin_memory
from torch.utils.data.dataloader import _SingleProcessDataLoaderIter
from transformers.trainer import seed_worker, Trainer as HFTrainer


class BufferedMultiSequenceIterator(_SingleProcessDataLoaderIter):
    def __init__(self, dataloader: DataLoader):
        super().__init__(dataloader)
        # We will manually apply the collate function.
        # Overriding it here will ensure it is not called during the call to `fetch`.
        self._dataset_fetcher.collate_fn = lambda x: x
        self.sequence_buffer = [{} for _ in range(dataloader.batch_size)]
        self.final_sequence_loaded = False
        self.dataset_keys = dataloader.dataset.column_names

    def _fill_sequence_buffer_if_empty(self):
        if self.final_sequence_loaded:
            return
        
        for i in range(len(self.sequence_buffer)):
            if len(self.sequence_buffer[i]) == 0:
                try:
                    self.sequence_buffer[i] = self._fetch_single_sequence()
                except StopIteration:
                    self.final_sequence_loaded = True
                    break
    
    def _fetch_single_sequence(self):
        index = self._next_index()  # may raise StopIteration
        data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
        if self._pin_memory:
            data = pin_memory(data, self._pin_memory_device)
        return data

    def _next_data(self):
        self._fill_sequence_buffer_if_empty()
        batch = []
        sample_count = 0
        for i in range(len(self.sequence_buffer)):
            # First check if the sequence buffer is empty
            if not self.sequence_buffer[i]:
                # Need to append an empty sequence ensure the batch size is consistent
                # And each separate sequence stays at the same index
                batch.append({k: [] for k in self.dataset_keys})
                continue
            
            # If not empty, add the first item to the batch
            batch.append({k: v.pop(0) for k, v in self.sequence_buffer[i].items()})
            sample_count += 1
            # TODO: Eventually, when we get to recurrent models, we will need to return
            #       something that indicates the end of the sequence
            # Reset the buffer to empty if the sequence is exhausted
            first_key = self.dataset_keys[0]
            if len(self.sequence_buffer[i][first_key]) == 0:
                self.sequence_buffer[i] = {}

        if sample_count == 0:
            raise StopIteration

        return self._collate_fn(batch)


# Note that this dataloader does not work with Accelerate.
# It will not throw an error, but it will not load the data correctly.
class MultiSequenceDataloader(DataLoader):
    def __init__(self, *args, **kwargs):
        if 'batch_sampler' in kwargs:
            assert kwargs['batch_sampler'] is None, "batch_sampler not supported"
        super().__init__(*args, **kwargs)

        object.__setattr__(self, 'batch_sampler', None)
        if self.collate_fn == _utils.collate.default_collate:
            self.collate_fn = _utils.collate.default_convert
        
    def _get_iterator(self):
        return BufferedMultiSequenceIterator(self)


class MultiSequenceDataset(Dataset):
    def __init__(self, *args, sequence_indices: List[int], **kwargs):
        super().__init__(*args, **kwargs)
        self.sequence_indices = sequence_indices
        self.sequence_indices.append(self.num_rows)
    
    def __len__(self):
        return len(self.sequence_indices) - 1

    def __getitem__(self, index):
        # Convert integer indices into slices that retrieve the entire sequence
        if isinstance(index, int):
            index = slice(self.sequence_indices[index], self.sequence_indices[index + 1])
        return super().__getitem__(index)

    @classmethod
    def from_nested_list(
        cls,
        nested_list: List[List[Dict[str, Any]]],
        **kwargs,
    ) -> "MultiSequenceDataset":
        flattened_list = [item for sublist in nested_list for item in sublist]
        sequence_indices = list(np.cumsum([0] + [len(seq) for seq in nested_list[:-1]]))
        
        dataset = Dataset.from_list(flattened_list, **kwargs)
        return cls(
            dataset._data,
            info = dataset._info,
            split = dataset._split,
            sequence_indices = sequence_indices,
        )


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
