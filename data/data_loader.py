"""
Data Loading and Preprocessing Pipeline for Knowledge Distillation
Uses real datasets from HuggingFace with efficient batching and tokenization
"""

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets
from typing import Dict, List, Optional, Union, Callable
import numpy as np
from tqdm import tqdm


class DistillationDataset(Dataset):
    """Custom dataset for knowledge distillation"""

    def __init__(self,
                 dataset_names: List[str],
                 student_tokenizer: AutoTokenizer,
                 teacher_tokenizer: AutoTokenizer,
                 max_length: int = 512,
                 teacher_max_length: Optional[int] = None,
                 subset_size: Optional[int] = None,
                 cache_dir: str = "./cache",
                 split: str = "train"):
        """
        Args:
            dataset_names: List of HuggingFace dataset names to use
            student_tokenizer: Tokenizer for student model (used for lengths)
            teacher_tokenizer: Tokenizer for teacher model (kept for reference)
            max_length: Maximum sequence length
            teacher_max_length: Optional teacher sequence cap (default = max_length)
            subset_size: Optional size limit for dataset
            cache_dir: Directory for caching datasets
            split: Dataset split to load (train/validation/test)
        """
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.max_length = max_length
        self.teacher_max_length = teacher_max_length or max_length
        self.cache_dir = cache_dir
        self.split = split

        # Load, combine and extract raw texts
        combined_dataset = self._load_and_combine_datasets(dataset_names, subset_size)
        self.samples: List[str] = []
        for example in combined_dataset:
            text = self._extract_text(example)
            if text:
                self.samples.append(text)

        if subset_size is not None:
            self.samples = self.samples[:subset_size]

        if not self.samples:
            raise ValueError("No valid text samples were loaded for the specified datasets.")

        # Pre-compute student token lengths for dynamic batching
        self.student_lengths: List[int] = [
            self._compute_token_length(text)
            for text in self.samples
        ]

    def _load_and_combine_datasets(self, dataset_names: List[str], subset_size: Optional[int]):
        """Load multiple datasets and combine them"""
        datasets_list = []

        for dataset_name in dataset_names:
            try:
                load_kwargs = {
                    "cache_dir": self.cache_dir
                }

                if dataset_name in {"bookcorpus"}:
                    load_kwargs["trust_remote_code"] = True

                # Handle different dataset configurations
                if dataset_name == "wikitext":
                    ds = load_dataset("wikitext", "wikitext-103-raw-v1",
                                     split=self.split, **load_kwargs)
                elif dataset_name == "c4":
                    # C4 is large, only load if explicitly requested and not in recommended
                    ds = load_dataset("c4", "en", split=self.split, **load_kwargs)
                elif dataset_name == "bookcorpus":
                    ds = load_dataset("bookcorpus", "plain_text", split="train",
                                     **load_kwargs)
                elif dataset_name == "wikipedia":
                    ds = load_dataset("wikipedia", "20220301.en", split="train",
                                     **load_kwargs)
                else:
                    # Try to load custom dataset
                    ds = load_dataset(dataset_name, split=self.split,
                                     **load_kwargs)

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
                            split=self.split, cache_dir=self.cache_dir)
            if subset_size:
                ds = ds.select(range(min(len(ds), subset_size)))
            datasets_list = [ds]

        # Combine all datasets
        if len(datasets_list) > 1:
            combined = concatenate_datasets(datasets_list)
        else:
            combined = datasets_list[0]

        return combined

    def _extract_text(self, example: Dict) -> Optional[str]:
        """Extract text field from dataset example"""
        if "text" in example and isinstance(example["text"], str):
            text = example["text"]
        elif "content" in example and isinstance(example["content"], str):
            text = example["content"]
        elif "passage" in example and isinstance(example["passage"], str):
            text = example["passage"]
        else:
            text = None
            for value in example.values():
                if isinstance(value, str):
                    text = value
                    break

        if text is None:
            return None

        text = text.strip()
        return text if text else None

    def _compute_token_length(self, text: str) -> int:
        tokens = self.student_tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_attention_mask=False,
            add_special_tokens=True
        )
        return min(len(tokens["input_ids"]), self.max_length)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def get_length(self, idx: int) -> int:
        return self.student_lengths[idx]


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
            length = self.dataset.get_length(idx)

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


class DualTokenizerCollator:
    """Collate function that tokenizes with both student and teacher tokenizers"""

    def __init__(self,
                 student_tokenizer: AutoTokenizer,
                 teacher_tokenizer: AutoTokenizer,
                 student_max_length: int,
                 teacher_max_length: int,
                 pad_to_multiple_of: int = 8):
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.student_max_length = student_max_length
        self.teacher_max_length = teacher_max_length
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch: List[str]) -> Dict[str, torch.Tensor]:
        texts = batch

        student_tokens = self.student_tokenizer(
            texts,
            padding='longest',
            truncation=True,
            max_length=self.student_max_length,
            return_attention_mask=True,
            add_special_tokens=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt'
        )

        teacher_tokens = self.teacher_tokenizer(
            texts,
            padding='longest',
            truncation=True,
            max_length=self.teacher_max_length,
            return_attention_mask=True,
            add_special_tokens=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt'
        )

        labels = student_tokens["input_ids"].clone()

        return {
            "student_input_ids": student_tokens["input_ids"],
            "student_attention_mask": student_tokens["attention_mask"],
            "teacher_input_ids": teacher_tokens["input_ids"],
            "teacher_attention_mask": teacher_tokens["attention_mask"],
            "labels": labels
        }


def create_distillation_dataloader(
    dataset_names: List[str],
    student_tokenizer: AutoTokenizer,
    teacher_tokenizer: AutoTokenizer,
    batch_size: int = 8,
    max_length: int = 512,
    teacher_max_length: Optional[int] = None,
    subset_size: Optional[int] = None,
    shuffle: bool = True,
    num_workers: int = 4,
    use_dynamic_batching: bool = True,
    cache_dir: str = "./cache",
    split: str = "train"
) -> DataLoader:
    """
    Create a DataLoader for distillation training

    Args:
        dataset_names: List of dataset names to use
        student_tokenizer: Student tokenizer for text processing
        teacher_tokenizer: Teacher tokenizer for text processing
        batch_size: Batch size for training
        max_length: Maximum sequence length
        teacher_max_length: Maximum teacher sequence length (defaults to max_length)
        subset_size: Optional limit on dataset size
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        use_dynamic_batching: Use dynamic batching by sequence length
        cache_dir: Cache directory for datasets
        split: Dataset split to load

    Returns:
        DataLoader instance
    """
    # Create dataset
    dataset = DistillationDataset(
        dataset_names=dataset_names,
        student_tokenizer=student_tokenizer,
        teacher_tokenizer=teacher_tokenizer,
        max_length=max_length,
        teacher_max_length=teacher_max_length,
        subset_size=subset_size,
        cache_dir=cache_dir,
        split=split
    )

    data_collator = DualTokenizerCollator(
        student_tokenizer=student_tokenizer,
        teacher_tokenizer=teacher_tokenizer,
        student_max_length=max_length,
        teacher_max_length=teacher_max_length or max_length,
        pad_to_multiple_of=8
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
    student_tokenizer: AutoTokenizer,
    teacher_tokenizer: AutoTokenizer,
    batch_size: int = 16,
    max_length: int = 512,
    teacher_max_length: Optional[int] = None,
    cache_dir: str = "./cache"
) -> DataLoader:
    """
    Prepare evaluation dataloader

    Args:
        student_tokenizer: Student tokenizer for text processing
        teacher_tokenizer: Teacher tokenizer for text processing
        batch_size: Batch size for evaluation
        max_length: Maximum sequence length
        teacher_max_length: Maximum teacher sequence length
        cache_dir: Cache directory

    Returns:
        DataLoader for evaluation
    """
    eval_texts = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split="validation",
        cache_dir=cache_dir
    )

    eval_dataset = DistillationDataset(
        dataset_names=["wikitext"],
        student_tokenizer=student_tokenizer,
        teacher_tokenizer=teacher_tokenizer,
        max_length=max_length,
        teacher_max_length=teacher_max_length or max_length,
        subset_size=min(1000, len(eval_texts)),
        cache_dir=cache_dir,
        split="validation"
    )

    collator = DualTokenizerCollator(
        student_tokenizer=student_tokenizer,
        teacher_tokenizer=teacher_tokenizer,
        student_max_length=max_length,
        teacher_max_length=teacher_max_length or max_length,
        pad_to_multiple_of=8
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    return eval_dataloader
