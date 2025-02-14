import numpy as np
from tqdm import tqdm

from src.preprocessing.embedding.embedder_interface import EmbedderInterface, EmbeddingInputTypes


class Dataset:

    def __init__(self, texts_train: list[list[str]], texts_test: list[list[str]], labels_train: list[str | int],
                 labels_test: list[str | int], highlight_masks_train: list[list[np.ndarray]],
                 highlight_masks_test: list[list[np.ndarray]], embedder: EmbedderInterface):

        assert len(texts_train) == len(labels_train) == len(highlight_masks_train)
        assert len(texts_test) == len(labels_test) == len(highlight_masks_test)

        self.n_samples_train = len(texts_train)
        self.n_samples_test = len(texts_test)
        self.texts = texts_train + texts_test
        self.labels = labels_train + labels_test
        self.highlight_masks = highlight_masks_train + highlight_masks_test
        self.embedder = embedder

        self.tokenizer, self.detokenizer, self.max_len = self.__build_tokenizer()
        self.vocab_size = len(self.tokenizer) + 1

        tokenized_texts, self.masks = self.__tokenize_and_pad()
        self.embedded_texts = self.__embed_texts(tokenized_texts)

        self.labels_encoder = self.__build_labels_encoder()
        self.encoded_labels = self.__encode_labels()

    @property
    def embedding_dim(self):
        return self.embedder.embedding_dim

    def get_train_and_test_sets(self) -> tuple[tuple, tuple]:

        print("{} samples in the training set".format(self.n_samples_train))
        print("{} samples in the test set".format(self.n_samples_test))

        texts_train = self.embedded_texts[0:self.n_samples_train]
        masks_train = self.masks[0:self.n_samples_train]
        labels_train = self.encoded_labels[0:self.n_samples_train]
        highlight_masks_train = self.highlight_masks[0:self.n_samples_train]

        texts_test = self.embedded_texts[self.n_samples_train:]
        masks_test = self.masks[self.n_samples_train:]
        labels_test = self.encoded_labels[self.n_samples_train:]
        highlight_masks_test = self.highlight_masks[self.n_samples_train:]

        x_train = [texts_train, masks_train]
        x_test = [texts_test, masks_test]

        train_set = (x_train, labels_train, highlight_masks_train)
        test_set = (x_test, labels_test, highlight_masks_test)

        return train_set, test_set

    def get_train_and_test_texts(self) -> tuple[list[list[str]], list[list[str]]]:

        texts_train = self.texts[0:self.n_samples_train]
        texts_test = self.texts[self.n_samples_train:]

        return texts_train, texts_test

    def get_classes_count(self) -> int:
        return len(self.labels_encoder)

    def __build_tokenizer(self) -> tuple[dict[str, int], dict[int, str], int]:

        tokenizer: dict[str, int] = dict()
        detokenizer: dict[int, str] = dict()
        max_len = 0

        token_number = 1

        print("Building tokenizer...")

        for text in tqdm(self.texts[0: self.n_samples_train]):
            if len(text) > max_len:
                max_len = len(text)
            for word in text:
                if word not in tokenizer:
                    tokenizer[word] = token_number
                    detokenizer[token_number] = word
                    token_number += 1

        print("Max length: {}".format(max_len))

        return tokenizer, detokenizer, max_len

    def __tokenize_and_pad(self) -> tuple[list[list[int]], np.ndarray]:

        tokenized_texts: list[list[int]] = []
        masks: list[np.ndarray] = []

        print("Tokenizing texts...")

        for text in tqdm(self.texts):

            tokenized_sequence: list[int] = []

            for word in text:
                if word in self.tokenizer:
                    token = self.tokenizer[word]
                else:
                    token = self.vocab_size  # UNK token

                tokenized_sequence.append(token)

            mask = [1] * len(text)

            pad_len = self.max_len - len(text)
            if pad_len > 0:
                pad_sequence = [0] * pad_len
                tokenized_sequence.extend(pad_sequence)
                mask.extend(pad_sequence)
            elif pad_len < 0:
                tokenized_sequence = tokenized_sequence[0:pad_len]
                mask = mask[0:pad_len]

            tokenized_texts.append(tokenized_sequence)
            masks.append(np.array(mask))

        return tokenized_texts, np.array(masks)

    def __embed_texts(self, tokenized_texts: list[list[int]]) -> np.ndarray:

        embedded_texts: list[list[np.ndarray]] = []

        print("Embedding texts...")

        for text in tqdm(tokenized_texts):
            embedded_text: list[np.ndarray] = []
            for token in text:
                embedding_type = self.embedder.get_input_type()
                if embedding_type == EmbeddingInputTypes.TOKEN:
                    embedding = self.embedder.embed(token)
                elif embedding_type == EmbeddingInputTypes.TEXT:
                    word = self.detokenizer.get(token, None)
                    embedding = self.embedder.embed(word)
                else:
                    raise Exception("Invalid EmbeddingInputType: {}".format(embedding_type))
                embedded_text.append(embedding)
            embedded_texts.append(embedded_text)

        return np.array(embedded_texts)

    def __build_labels_encoder(self) -> dict[str | int, np.ndarray]:

        label_encoder: dict[str | int, np.ndarray] = dict()

        unique_labels: list[str | int] = list(set(self.labels))

        for i, label in enumerate(unique_labels):
            one_hot_encoding = [0] * len(unique_labels)
            one_hot_encoding[i] = 1
            label_encoder[label] = np.array(one_hot_encoding)

        return label_encoder

    def __encode_labels(self) -> np.ndarray:

        encoded_labels: list[np.ndarray] = []

        print("Encoding labels...")

        for label in tqdm(self.labels):
            encoded_label = self.labels_encoder[label]
            encoded_labels.append(encoded_label)

        return np.array(encoded_labels)
