"""
Data Loading and Preprocessing Pipeline for Knowledge Distillation
Uses real datasets from HuggingFace with efficient batching and tokenization
"""

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset, concatenate_datasets
from typing import Dict, List, Optional, Union
import numpy as np
from tqdm import tqdm


class DistillationDataset(Dataset):
    """Custom dataset for knowledge distillation"""

    def __init__(self,
                 dataset_names: List[str],
                 tokenizer: AutoTokenizer,
                 max_length: int = 512,
                 subset_size: Optional[int] = None,
                 cache_dir: str = "./cache"):
        """
        Args:
            dataset_names: List of HuggingFace dataset names to use
            tokenizer: Tokenizer for processing text
            max_length: Maximum sequence length
            subset_size: Optional size limit for dataset
            cache_dir: Directory for caching datasets
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_dir = cache_dir

        # Load and combine datasets
        self.dataset = self._load_and_combine_datasets(dataset_names, subset_size)

        # Preprocess if needed
        self.dataset = self.dataset.map(
            self._preprocess_function,
            batched=True,
            remove_columns=self.dataset.column_names,
            desc="Tokenizing dataset"
        )

    def _load_and_combine_datasets(self, dataset_names: List[str], subset_size: Optional[int]):
        """Load multiple datasets and combine them"""
        datasets_list = []

        for dataset_name in dataset_names:
            try:
                # Handle different dataset configurations
                if dataset_name == "wikitext":
                    ds = load_dataset("wikitext", "wikitext-103-raw-v1",
                                     split="train", cache_dir=self.cache_dir)
                elif dataset_name == "c4":
                    # C4 is large, only load if explicitly requested and not in recommended
                    ds = load_dataset("c4", "en", split="train", cache_dir=self.cache_dir)
                elif dataset_name == "bookcorpus":
                    ds = load_dataset("bookcorpus", "plain_text", split="train",
                                     cache_dir=self.cache_dir)
                elif dataset_name == "wikipedia":
                    ds = load_dataset("wikipedia", "20220301.en", split="train",
                                     cache_dir=self.cache_dir)
                else:
                    # Try to load custom dataset
                    ds = load_dataset(dataset_name, split="train",
                                     cache_dir=self.cache_dir)

                # Limit dataset size if specified
                if subset_size and len(ds) > subset_size:
                    ds = ds.select(range(subset_size))

                datasets_list.append(ds)
                print(f"Loaded {dataset_name}: {len(ds)} samples")

            except Exception as e:
                print(f"Warning: Could not load dataset {dataset_name}: {e}")
                continue

        if not datasets_list:
            # Fallback to a simple default dataset
            print("Loading default dataset: wikitext-2")
            ds = load_dataset("wikitext", "wikitext-2-raw-v1",
                            split="train", cache_dir=self.cache_dir)
            if subset_size:
                ds = ds.select(range(min(len(ds), subset_size)))
            datasets_list = [ds]

        # Combine all datasets
        if len(datasets_list) > 1:
            combined = concatenate_datasets(datasets_list)
        else:
            combined = datasets_list[0]

        return combined

    def _preprocess_function(self, examples):
        """Tokenize and prepare examples for training"""
        # Get text field (handle different dataset formats)
        if "text" in examples:
            texts = examples["text"]
        elif "content" in examples:
            texts = examples["content"]
        elif "passage" in examples:
            texts = examples["passage"]
        else:
            # Try to find any text field
            for key in examples.keys():
                if isinstance(examples[key][0], str):
                    texts = examples[key]
                    break
            else:
                raise ValueError("No text field found in dataset")

        # Filter out empty texts
        texts = [t for t in texts if t and len(t.strip()) > 0]

        if not texts:
            return {"input_ids": [], "attention_mask": [], "labels": []}

        # Tokenize
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        # For language modeling, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].clone()

        return tokenized

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"]),
            "attention_mask": torch.tensor(item["attention_mask"]),
            "labels": torch.tensor(item["labels"])
        }


class DynamicBatchSampler:
    """Dynamic batch sampler that groups sequences by length for efficiency"""

    def __init__(self, dataset, batch_size: int, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        # Group indices by sequence length
        self.length_groups = self._group_by_length()

    def _group_by_length(self):
        """Group dataset indices by sequence length"""
        length_to_indices = {}

        for idx in range(len(self.dataset)):
            item = self.dataset[idx]
            length = (item["attention_mask"] == 1).sum().item()

            if length not in length_to_indices:
                length_to_indices[length] = []
            length_to_indices[length].append(idx)

        return length_to_indices

    def __iter__(self):
        """Yield batches of similar-length sequences"""
        all_batches = []

        for length, indices in self.length_groups.items():
            # Shuffle indices within each length group
            np.random.shuffle(indices)

            # Create batches
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    all_batches.append(batch)

        # Shuffle all batches
        np.random.shuffle(all_batches)

        for batch in all_batches:
            yield batch

    def __len__(self):
        total_batches = 0
        for indices in self.length_groups.values():
            n_batches = len(indices) // self.batch_size
            if not self.drop_last and len(indices) % self.batch_size > 0:
                n_batches += 1
            total_batches += n_batches
        return total_batches


def create_distillation_dataloader(
    dataset_names: List[str],
    tokenizer: AutoTokenizer,
    batch_size: int = 8,
    max_length: int = 512,
    subset_size: Optional[int] = None,
    shuffle: bool = True,
    num_workers: int = 4,
    use_dynamic_batching: bool = True,
    cache_dir: str = "./cache"
) -> DataLoader:
    """
    Create a DataLoader for distillation training

    Args:
        dataset_names: List of dataset names to use
        tokenizer: Tokenizer for text processing
        batch_size: Batch size for training
        max_length: Maximum sequence length
        subset_size: Optional limit on dataset size
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        use_dynamic_batching: Use dynamic batching by sequence length
        cache_dir: Cache directory for datasets

    Returns:
        DataLoader instance
    """
    # Create dataset
    dataset = DistillationDataset(
        dataset_names=dataset_names,
        tokenizer=tokenizer,
        max_length=max_length,
        subset_size=subset_size,
        cache_dir=cache_dir
    )

    # Create data collator for padding
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
        pad_to_multiple_of=8  # Efficient for tensor cores
    )

    # Create dataloader
    if use_dynamic_batching:
        batch_sampler = DynamicBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            drop_last=True
        )

        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=data_collator,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=data_collator,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True
        )

    return dataloader


def get_recommended_datasets() -> List[str]:
    """Get recommended datasets for distillation"""
    return [
        "wikitext",      # High-quality Wikipedia text
        "bookcorpus",    # Large, clean text corpus
        # "c4",          # Large-scale web corpus (commented due to size)
        # "openwebtext", # Replaced by bookcorpus due to instability
        # "wikipedia",   # Full Wikipedia (commented due to size)
    ]


def prepare_eval_dataloader(
    tokenizer: AutoTokenizer,
    batch_size: int = 16,
    max_length: int = 512,
    cache_dir: str = "./cache"
) -> DataLoader:
    """
    Prepare evaluation dataloader

    Args:
        tokenizer: Tokenizer for text processing
        batch_size: Batch size for evaluation
        max_length: Maximum sequence length
        cache_dir: Cache directory

    Returns:
        DataLoader for evaluation
    """
    # Use wikitext validation set for evaluation
    eval_dataset = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split="validation",
        cache_dir=cache_dir
    )

    # Limit size for faster evaluation
    eval_dataset = eval_dataset.select(range(min(1000, len(eval_dataset))))

    # Create dataset wrapper
    class EvalDataset(Dataset):
        def __init__(self, dataset, tokenizer, max_length):
            self.dataset = dataset
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.processed = []

            # Preprocess all data
            for item in tqdm(dataset, desc="Processing eval data"):
                text = item.get("text", "")
                if text and len(text.strip()) > 0:
                    tokens = tokenizer(
                        text,
                        truncation=True,
                        padding="max_length",
                        max_length=max_length,
                        return_tensors="pt"
                    )
                    self.processed.append({
                        "input_ids": tokens["input_ids"].squeeze(0),
                        "attention_mask": tokens["attention_mask"].squeeze(0),
                        "labels": tokens["input_ids"].squeeze(0)
                    })

        def __len__(self):
            return len(self.processed)

        def __getitem__(self, idx):
            return self.processed[idx]

    eval_dataset_wrapped = EvalDataset(eval_dataset, tokenizer, max_length)

    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )

    # Create dataloader
    eval_dataloader = DataLoader(
        eval_dataset_wrapped,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    return eval_dataloader
