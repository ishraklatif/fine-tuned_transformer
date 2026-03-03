import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, AutoTokenizer
from sklearn import preprocessing
from six.moves.urllib.request import urlretrieve
from datasets import Dataset
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class DataManager:
    """
    Manages downloading, preprocessing, tokenising, and splitting
    the question-classification dataset for the Transformer models.
    """

    DATA_URL = "http://cogcomp.org/Data/QA/QC/"
    DATA_FILE = "train_2000.label"

    def __init__(self, verbose: bool = True, random_state: int = 6789):
        self.verbose = verbose
        self.max_sentence_len = 0
        self.str_questions: list = []
        self.str_labels: list = []
        self.numeral_labels = None
        self.numeral_data = None
        self.random_state = random_state
        self.random = np.random.RandomState(random_state)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def maybe_download(dir_name: str, file_name: str, url: str, verbose: bool = True) -> None:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        filepath = os.path.join(dir_name, file_name)
        if not os.path.exists(filepath):
            import requests
            response = requests.get(url + file_name, verify=False)
            with open(filepath, 'wb') as f:
                f.write(response.content)
        if verbose:
            print(f"Downloaded successfully: {file_name}")

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def read_data(self, dir_name: str, file_names: list) -> None:
        self.str_questions = []
        self.str_labels = []
        for file_name in file_names:
            file_path = os.path.join(dir_name, file_name)
            with open(file_path, "r", encoding="latin-1") as f:
                for row in f:
                    row_str = row.split(":")
                    label, question = row_str[0], row_str[1]
                    question = question.lower()
                    self.str_labels.append(label)
                    self.str_questions.append(question[:-1])
                    if self.max_sentence_len < len(self.str_questions[-1]):
                        self.max_sentence_len = len(self.str_questions[-1])

        le = preprocessing.LabelEncoder()
        le.fit(self.str_labels)
        self.numeral_labels = np.array(le.transform(self.str_labels))
        self.str_classes = le.classes_
        self.num_classes = len(self.str_classes)

        if self.verbose:
            print("\nSample questions and labels:")
            print(self.str_questions[:5])
            print(self.str_labels[:5])

    def manipulate_data(self) -> None:
        """Tokenise with BERT tokeniser and pad sequences."""
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        vocab = self.tokenizer.get_vocab()
        self.word2idx = {w: i for i, w in enumerate(vocab)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)

        num_seqs = []
        for text in self.str_questions:
            text_seqs = self.tokenizer.tokenize(str(text))
            token_ids = self.tokenizer.convert_tokens_to_ids(text_seqs)
            num_seqs.append(torch.LongTensor(token_ids))

        if num_seqs:
            self.numeral_data = pad_sequence(num_seqs, batch_first=True)
            self.num_sentences, self.max_seq_len = self.numeral_data.shape

    def train_valid_test_split(self, train_ratio: float = 0.8, test_ratio: float = 0.1) -> None:
        """Split into train / valid / test DataLoaders (for Transformer classifier)."""
        train_size = int(self.num_sentences * train_ratio) + 1
        test_size = int(self.num_sentences * test_ratio) + 1

        data_indices = list(range(self.num_sentences))
        random.shuffle(data_indices)

        def _make_loader(indices, shuffle):
            data = self.numeral_data[indices]
            labels = torch.from_numpy(self.numeral_labels[indices])
            dataset = torch.utils.data.TensorDataset(data, labels)
            return DataLoader(dataset, batch_size=64, shuffle=shuffle)

        self.train_loader = _make_loader(data_indices[:train_size], shuffle=True)
        self.test_loader = _make_loader(data_indices[-test_size:], shuffle=False)
        self.valid_loader = _make_loader(data_indices[train_size:-test_size], shuffle=False)

        # Store string questions for reference
        self.train_str_questions = [self.str_questions[i] for i in data_indices[:train_size]]
        self.test_str_questions = [self.str_questions[i] for i in data_indices[-test_size:]]
        self.valid_str_questions = [self.str_questions[i] for i in data_indices[train_size:-test_size]]

    # ------------------------------------------------------------------
    # BERT fine-tuning helpers
    # ------------------------------------------------------------------

    def get_bert_loaders(self, model_name: str = "bert-base-uncased",
                         max_length: int = 36,
                         train_ratio: float = 0.8,
                         test_ratio: float = 0.1,
                         batch_size: int = 64):
        """Build HuggingFace Dataset-based loaders for BERT prefix-tuning."""
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        dataset = Dataset.from_dict({
            "text": self.str_questions,
            "label": self.numeral_labels
        })

        def tokenize_fn(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=max_length
            )

        dataset = dataset.map(tokenize_fn, batched=True)
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

        n = len(dataset)
        train_size = int(n * train_ratio) + 1
        test_size = int(n * test_ratio) + 1

        def _hf_loader(subset, shuffle):
            return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)

        train_loader = _hf_loader(Dataset.from_dict(dataset[:train_size]).with_format("torch", columns=["input_ids", "attention_mask", "label"]), True)
        test_loader = _hf_loader(Dataset.from_dict(dataset[-test_size:]).with_format("torch", columns=["input_ids", "attention_mask", "label"]), False)
        valid_loader = _hf_loader(Dataset.from_dict(dataset[train_size:-test_size]).with_format("torch", columns=["input_ids", "attention_mask", "label"]), False)

        return train_loader, valid_loader, test_loader
