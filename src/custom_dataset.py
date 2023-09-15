import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):

    def __init__(self, X_note, X_content, tokenizer, max_len):
        
        self.texts_note = X_note
        self.texts_content = X_content
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts_note)

    def __getitem__(self, index):
        texts_note = str(self.texts_note[index])
        texts_note = " ".join(texts_note.split())

        texts_content = str(self.texts_content[index])
        texts_content = " ".join(texts_content.split())

        inputs_note = self.tokenizer.encode_plus(
            texts_note,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )

        inputs_content = self.tokenizer.encode_plus(
            texts_content,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )

        ids_note = inputs_note['input_ids']
        mask_note = inputs_note['attention_mask']

        ids_content = inputs_content['input_ids']
        mask_content = inputs_content['attention_mask']

        return {
            "input_ids_note": torch.tensor(ids_note, dtype=torch.long),
            "attention_mask_note": torch.tensor(mask_note, dtype=torch.long),
            "input_ids_content": torch.tensor(ids_content, dtype=torch.long),
            "attention_mask_content": torch.tensor(mask_content, dtype=torch.long),
        }
