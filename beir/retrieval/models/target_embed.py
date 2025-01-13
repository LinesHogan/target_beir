from sentence_transformers import SentenceTransformer
from torch import Tensor
import torch.multiprocessing as mp
from typing import List, Dict, Union, Tuple
import numpy as np
import logging
from datasets import Dataset
from tqdm import tqdm
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, Qwen2ForCausalLM
import time

logger = logging.getLogger(__name__)

class BaseEmbed(ABC):
    def __init__(self, model_path, device_map="cuda"):
        self.model_path = model_path
        self.device_map = device_map

    @abstractmethod
    def embedding(self, prompts, **kwargs):
        """抽象方法，获取embedding"""
        pass

    @abstractmethod
    def query_embed(self, text):
        """抽象方法，获取query的embedding"""
        pass

    @abstractmethod
    def content_embed(self, text):
        """抽象方法，获取content的embedding"""
        pass

    def __call__(self, text):
        """通用调用接口"""
        return self.embedding(text)

class TargetEmbed(BaseEmbed):
    def __init__(self, model_path, device_map="cuda", repeat_prefix=None, mode="sum", target_token_index=None):
        super().__init__(model_path, device_map)
        self._validate_mode(mode)  # 检查 mode 是否有效
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model:Qwen2ForCausalLM = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype="auto",
            output_hidden_states=True,
            trust_remote_code=True,
        )
        self.repeat_prefix = repeat_prefix or (
            "Please repeat everything I say from now on. Do not add any extra content, just repeat exactly what I say. "
            "\n\n I am about to start speaking: "
        )
        self.mode = mode
        prompts = [
            "Two roads diverged in a yellow wood, And sorry I could not travel both",
            "Hope is the thing with feathers That perches in the soul"
        ]
        self.max_common_suffix_length = self._compute_max_common_suffix_length(prompts)
        self.target_token_index = self._validate_target_token_index(
            target_token_index, self.max_common_suffix_length, self.mode
        )
        self.hidden_size = self.model.config.hidden_size

    def _compute_max_common_suffix_length(self, prompts):
        tokenized_prompts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            tokenized_text = self.tokenizer(text, return_tensors="pt", padding=False)["input_ids"]
            tokenized_prompts.append(tokenized_text[0].tolist())
        tokens1, tokens2 = tokenized_prompts
        min_length = min(len(tokens1), len(tokens2))
        common_suffix_length = 0
        for i in range(1, min_length + 1): 
            if tokens1[-i] == tokens2[-i]:
                common_suffix_length += 1
            else:
                break
        return common_suffix_length
        
    def _validate_target_token_index(self, target_token_index, max_suffix_length, mode):
        if mode == "token":
            target_token_index = target_token_index or 1
            if not (1 <= target_token_index <= max_suffix_length):
                raise ValueError(
                    f"target_token_index must be within the range of 1 and max_common_suffix_length ({max_suffix_length}). "
                    f"Got {target_token_index}."
                )
        else:
            target_token_index = None
        return target_token_index

    def _validate_mode(self, mode):
        valid_modes = ["sum", "concat", "token"]
        if mode not in valid_modes:
            raise ValueError(
                f"Invalid mode: {mode}. Supported modes are: {', '.join(valid_modes)}."
            )

    def embedding(self, prompts, layer=-1, prefix=""):
        text = []
        for prompt in prompts:
            prompt = prefix + prompt
            messages = [{"role": "user", "content": prompt}]
            text.append(
                self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )
        inputs = self.tokenizer(
            text, padding=True, padding_side="left", return_tensors="pt"
        ).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        hidden_states:torch.Tensor = outputs.hidden_states
        if self.mode == "sum":
            # print(f"shape: {hidden_states[layer][:, -self.max_common_suffix_length:].sum(axis=-2).shape}")
            return hidden_states[layer][:, -self.max_common_suffix_length:].sum(axis=-2)
        if self.mode == "concat":
            # print(f"shape: {hidden_states[layer][:, -self.max_common_suffix_length:].reshape(-1, self.hidden_size*self.max_common_suffix_length).shape}")
            return hidden_states[layer][:, -self.max_common_suffix_length:].reshape(-1, self.hidden_size*self.max_common_suffix_length)
        if self.mode == "token":
            return hidden_states[layer][:, -self.target_token_index]
            
    def query_embed(self, text):
        time1 = time.time()
        embeddings = self.embedding(text)
        # print("Time taken for query_embed: ", time.time()-time1)
        return embeddings

    def content_embed(self, text):
        time1 = time.time()
        embeddings = self.embedding(text, prefix=self.repeat_prefix)
        # print("Time taken for content_embed: ", time.time()-time1)
        return embeddings

class TargetEmbedBEIR:
    def __init__(self, model_path: str, sep: str = " ", mode="concat", target_token_index=None, **kwargs):
        self.sep = sep
        self.mode = mode
        self.target_token_index = target_token_index
        self.model = TargetEmbed(model_path, mode=mode, target_token_index=target_token_index, **kwargs)

    def start_multi_process_pool(self, target_devices: List[str] = None) -> Dict[str, object]:
        # Multi-processing is more complex with TargetEmbed and might require significant changes
        # to the way it handles tokenization and model inference. 
        # For now, we'll raise a NotImplementedError. You can explore adapting it later.
        raise NotImplementedError("Multi-processing is not currently supported for TargetEmbedBEIR.")

    def stop_multi_process_pool(self, pool: Dict[str, object]):
        raise NotImplementedError("Multi-processing is not currently supported for TargetEmbedBEIR.")

    def encode_queries(self, queries: List[str], batch_size: int = 16, **kwargs) -> Union[List[Tensor], np.ndarray, Tensor]:
        # Use TargetEmbed's query_embed method in batches
        results = []
        for i in tqdm(range(0, len(queries), batch_size), desc="Encoding Queries"):
            batch = queries[i: i + batch_size]
            batch_embeddings = self.model.query_embed(batch)
            results.append(batch_embeddings)

        return torch.cat(results, dim=0)

    def encode_corpus(self, corpus: Union[List[Dict[str, str]], Dict[str, List]], batch_size: int = 8, **kwargs) -> Union[List[Tensor], np.ndarray, Tensor]:
        if type(corpus) is dict:
            sentences = [(corpus["title"][i] + self.sep + corpus["text"][i]).strip() if "title" in corpus else corpus["text"][i].strip() for i in range(len(corpus['text']))]
        else:
            sentences = [(doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip() for doc in corpus]
        
        # Use TargetEmbed's content_embed method in batches
        results = []
        print(f"batch_size: {batch_size}")
        for i in tqdm(range(0, len(sentences), batch_size), desc="Encoding Corpus"):
            batch = sentences[i: i + batch_size]
            batch_embeddings = self.model.content_embed(batch)
            results.append(batch_embeddings)

        return torch.cat(results, dim=0)

    def encode_corpus_parallel(self, corpus: Union[List[Dict[str, str]], Dataset], pool: Dict[str, str], batch_size: int = 8, chunk_id: int = None, **kwargs):
        raise NotImplementedError("Parallel corpus encoding is not currently supported for TargetEmbedBEIR.")