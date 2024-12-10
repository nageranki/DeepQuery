import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import math
import json
import re
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ---------------------------
# Normalize Text Function
# ---------------------------
def normalize_text(text):
    """
    Normalizes text by lowercasing, removing extra spaces, and handling punctuation.
    """
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'\s([?.!,"])', r'\1', text)  # Remove spaces before punctuation
    return text.strip()

# ---------------------------
# Custom Embedding Layer
# ---------------------------
class CustomEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.embedding_dim)

# ---------------------------
# Positional Encoding
# ---------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_seq_length=512):
        super().__init__()
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim))
        pe = torch.zeros(max_seq_length, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :]

# ---------------------------
# Multi-Head Attention
# ---------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        assert self.head_dim * num_heads == embedding_dim, "Embedding dimension must be divisible by number of heads."

        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        self.output = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        Q = self.query(query).view(batch_size, -1, self.num_heads, self.head_dim)
        K = self.key(key).view(batch_size, -1, self.num_heads, self.head_dim)
        V = self.value(value).view(batch_size, -1, self.num_heads, self.head_dim)

        Q = Q.transpose(1, 2)  # (batch_size, num_heads, seq_length, head_dim)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (batch_size, num_heads, seq_length, seq_length)

        if mask is not None:
            # mask should be broadcastable to (batch_size, num_heads, seq_length, seq_length)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention, V)  # (batch_size, num_heads, seq_length, head_dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.embedding_dim)  # (batch_size, seq_length, embedding_dim)
        return self.output(output)

# ---------------------------
# Transformer Encoder Layer
# ---------------------------
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, ff_dim):
        super().__init__()
        self.self_attention = MultiHeadAttention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embedding_dim)
        )

    def forward(self, x, mask=None):
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + attn_output)
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        return x

# ---------------------------
# Question Answering Model
# ---------------------------
class QuestionAnsweringModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, ff_dim, max_seq_length):
        super().__init__()
        self.embedding = CustomEmbedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim, max_seq_length)

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(embedding_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])

        self.start_predictor = nn.Linear(embedding_dim, 1)
        self.end_predictor = nn.Linear(embedding_dim, 1)

    def forward(self, input_ids, attention_mask):
        x = self.embedding(input_ids)  # (batch_size, seq_length, embedding_dim)
        x = self.positional_encoding(x)

        # Create a mask compatible with multi-head attention
        # attention_mask: (batch_size, seq_length)
        # Convert to (batch_size, 1, 1, seq_length) for broadcasting
        attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        for layer in self.encoder_layers:
            x = layer(x, attn_mask)

        start_logits = self.start_predictor(x).squeeze(-1)  # (batch_size, seq_length)
        end_logits = self.end_predictor(x).squeeze(-1)

        return start_logits, end_logits

# ---------------------------
# Tokenizer
# ---------------------------
class CustomTokenizer:
    def __init__(self, vocab_file):
        with open(vocab_file, 'r') as f:
            self.vocab = json.load(f)
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}

        for token in ['[PAD]', '[UNK]', '[SEP]']:
            if token not in self.word_to_idx:
                raise ValueError(f"Missing special token {token} in vocabulary.")

    def tokenize(self, text):
        # Improved tokenization: preserve punctuation as separate tokens
        tokens = re.findall(r"\w+(?:'\w+)?|[^\w\s]", text.lower())
        return tokens

    def encode(self, tokens, max_length):
        token_ids = [self.word_to_idx.get(token, self.word_to_idx['[UNK]']) for token in tokens]
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        else:
            token_ids += [self.word_to_idx['[PAD]']] * (max_length - len(token_ids))
        return token_ids

# ---------------------------
# Custom Dataset
# ---------------------------
class QuestionAnsweringDataset(Dataset):
    def __init__(self, contexts, questions, answers, answer_starts, tokenizer, max_length):
        """
        Args:
            contexts (List[str]): List of context paragraphs.
            questions (List[str]): List of questions.
            answers (List[str]): List of answers.
            answer_starts (List[int]): List of answer start positions in the context.
            tokenizer (CustomTokenizer): Tokenizer instance.
            max_length (int): Maximum sequence length.
        """
        self.contexts = contexts
        self.questions = questions
        self.answers = answers
        self.answer_starts = answer_starts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        context = self.contexts[idx]
        question = self.questions[idx]
        answer = self.answers[idx]
        answer_start = self.answer_starts[idx]

        # Tokenize question and context
        question_tokens = self.tokenizer.tokenize(question)
        context_tokens = self.tokenizer.tokenize(context)

        # Combine tokens with [SEP]
        combined_tokens = question_tokens + ['[SEP]'] + context_tokens

        # Initialize start and end positions
        start_idx, end_idx = -1, -1

        if answer and answer_start != -1:
            # Calculate the character end position
            answer_end_char = answer_start + len(answer)

            # Tokenize context segments
            tokens_before_answer = self.tokenizer.tokenize(context[:answer_start])
            tokens_in_answer = self.tokenizer.tokenize(context[answer_start:answer_end_char])

            # Calculate token start and end indices
            token_start = len(question_tokens) + 1 + len(tokens_before_answer)
            token_end = token_start + len(tokens_in_answer) - 1  # Inclusive

            # Ensure within max_length
            if token_start < self.max_length and token_end < self.max_length:
                start_idx = token_start
                end_idx = token_end

        # Encode tokens
        input_ids = self.tokenizer.encode(combined_tokens, self.max_length)

        # Create attention mask
        attention_mask = [1 if tok != self.tokenizer.word_to_idx['[PAD]'] else 0 for tok in input_ids]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'start_position': torch.tensor(start_idx, dtype=torch.long),
            'end_position': torch.tensor(end_idx, dtype=torch.long)
        }

# ---------------------------
# Training Function
# ---------------------------
def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    count = 0

    for batch in tqdm(dataloader, desc="Training Batches"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_position'].to(device)
        end_positions = batch['end_position'].to(device)

        # Filter out samples with no answer
        valid_indices = (start_positions != -1) & (end_positions != -1)
        if valid_indices.sum() == 0:
            continue

        input_ids = input_ids[valid_indices]
        attention_mask = attention_mask[valid_indices]
        start_positions = start_positions[valid_indices]
        end_positions = end_positions[valid_indices]

        optimizer.zero_grad()
        start_logits, end_logits = model(input_ids, attention_mask)

        seq_length = start_logits.size(1)
        # Double-check targets are within bounds
        if (start_positions >= seq_length).any() or (end_positions >= seq_length).any():
            continue

        start_loss = criterion(start_logits, start_positions)
        end_loss = criterion(end_logits, end_positions)
        loss = start_loss + end_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Prevent exploding gradients
        optimizer.step()

        total_loss += loss.item()
        count += 1

    return total_loss / count if count > 0 else 0

# ---------------------------
# Vocabulary Creation
# ---------------------------
def create_vocabulary(contexts, questions, answers, vocab_file='vocab.json'):
    all_text = " ".join(contexts + questions + answers)
    # Improved tokenization to include punctuation
    tokens = re.findall(r"\w+(?:'\w+)?|[^\w\s]", all_text.lower())
    unique_tokens = sorted(set(tokens))
    vocab = ['[PAD]', '[UNK]', '[SEP]'] + unique_tokens
    with open(vocab_file, 'w') as f:
        json.dump(vocab, f)
    return vocab

# ---------------------------
# Dataset Verification
# ---------------------------
def verify_dataset(dataset, contexts, questions, answers, answer_starts, tokenizer):
    print("Verifying dataset...")
    mismatches = 0
    invalid_spans = 0
    for idx in range(len(dataset)):
        sample = dataset[idx]
        context = contexts[idx]
        question = questions[idx]
        answer = answers[idx]
        answer_start = answer_starts[idx]
        start = sample['start_position'].item()
        end = sample['end_position'].item()

        if answer and answer_start != -1:
            tokens_before_answer = tokenizer.tokenize(context[:answer_start])
            tokens_in_answer = tokenizer.tokenize(context[answer_start:answer_start + len(answer)])
            expected_tokens = tokens_in_answer

            extracted_tokens = tokenizer.tokenize(question) + ['[SEP]'] + tokenizer.tokenize(context)
            extracted_answer_tokens = extracted_tokens[start:end+1]
            extracted_answer = " ".join(extracted_answer_tokens)
            extracted_answer = normalize_text(extracted_answer)
            expected_answer = " ".join(expected_tokens)
            expected_answer = normalize_text(expected_answer)

            if extracted_answer != expected_answer:
                print(f"Sample {idx}: Mismatch")
                print(f"Question: {question}")
                print(f"Expected: {expected_answer}")
                print(f"Extracted: {extracted_answer}\n")
                mismatches += 1
        else:
            if answer_start != -1:
                print(f"Sample {idx}: Valid answer but positions set to -1.")
                print(f"Question: {question}")
                print(f"Answer: {answer}")
                print(f"Context: {context}\n")
                invalid_spans += 1
    print(f"Verification completed with {mismatches} mismatches and {invalid_spans} invalid spans.\n")

# ---------------------------
# Inference Function
# ---------------------------
def predict_answer(model, tokenizer, context, question, max_length=256, device='cpu'):
    """
    Predicts the answer to a question given a context using the trained model.
    """
    model.eval()

    # Tokenize question and context
    question_tokens = tokenizer.tokenize(question)
    context_tokens = tokenizer.tokenize(context)
    combined_tokens = question_tokens + ['[SEP]'] + context_tokens

    # Encode tokens
    input_ids = tokenizer.encode(combined_tokens, max_length)
    attention_mask = [1 if tok != tokenizer.word_to_idx['[PAD]'] else 0 for tok in input_ids]

    input_ids_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    attention_mask_tensor = torch.tensor([attention_mask], dtype=torch.long).to(device)

    with torch.no_grad():
        start_logits, end_logits = model(input_ids_tensor, attention_mask_tensor)

    start_pred = torch.argmax(start_logits, dim=1).item()
    end_pred = torch.argmax(end_logits, dim=1).item()

    if end_pred < start_pred:
        return ""

    # Extract predicted tokens
    extracted_tokens = combined_tokens[start_pred:end_pred+1]
    predicted_answer = " ".join(extracted_tokens)
    predicted_answer = normalize_text(predicted_answer)

    return predicted_answer

# ---------------------------
# Main Function
# ---------------------------
def main():
    EMBEDDING_DIM = 256
    NUM_HEADS = 8
    NUM_LAYERS = 3
    FF_DIM = 512
    MAX_SEQ_LENGTH = 256  # Increased from 128
    BATCH_SIZE = 8  # Increased batch size for efficiency
    LEARNING_RATE = 0.001
    EPOCHS = 2
    PATIENCE = 3  # Early stopping patience

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    json_path = "C:\\Users\\kumar\\Downloads\\small_train.json"
    with open(json_path, 'r', encoding='utf-8') as f:
        dataset_json = json.load(f)

    contexts = []
    questions = []
    answers = []
    answer_starts = []

    for article in dataset_json['data']:
        for paragraph in article['paragraphs']:
            context_text = paragraph['context']
            for qa in paragraph['qas']:
                question_text = qa['question']
                if not qa.get('is_impossible', False) and qa['answers']:
                    answer_text = qa['answers'][0]['text']
                    answer_start = qa['answers'][0]['answer_start']
                else:
                    answer_text = ""
                    answer_start = -1
                contexts.append(context_text)
                questions.append(question_text)
                answers.append(answer_text)
                answer_starts.append(answer_start)

    # Create vocabulary
    vocab = create_vocabulary(contexts, questions, answers)
    tokenizer = CustomTokenizer('vocab.json')
    tokenizer.max_length = MAX_SEQ_LENGTH  # Set max_length attribute for verification

    # Create dataset and dataloader
    dataset = QuestionAnsweringDataset(contexts, questions, answers, answer_starts, tokenizer, MAX_SEQ_LENGTH)
    train_indices, val_indices = train_test_split(list(range(len(dataset))), test_size=0.1, random_state=42)
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    train_dataloader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_dataloader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Initialize model
    model = QuestionAnsweringModel(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        ff_dim=FF_DIM,
        max_seq_length=MAX_SEQ_LENGTH
    ).to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Verify dataset before training
    verify_dataset(dataset, contexts, questions, answers, answer_starts, tokenizer)

    # Implement Early Stopping
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        # Training
        avg_train_loss = train_model(model, train_dataloader, optimizer, criterion, device)
        print(f"Average Training Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        total_val_loss = 0
        val_count = 0
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation Batches"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                start_positions = batch['start_position'].to(device)
                end_positions = batch['end_position'].to(device)

                # Filter out samples with no answer
                valid_indices = (start_positions != -1) & (end_positions != -1)
                if valid_indices.sum() == 0:
                    continue

                input_ids_ = input_ids[valid_indices]
                attention_mask_ = attention_mask[valid_indices]
                start_positions_ = start_positions[valid_indices]
                end_positions_ = end_positions[valid_indices]

                start_logits, end_logits = model(input_ids_, attention_mask_)

                seq_length = start_logits.size(1)
                if (start_positions_ >= seq_length).any() or (end_positions_ >= seq_length).any():
                    continue

                start_loss = criterion(start_logits, start_positions_)
                end_loss = criterion(end_logits, end_positions_)
                loss = start_loss + end_loss

                total_val_loss += loss.item()
                val_count += 1

        avg_val_loss = total_val_loss / val_count if val_count > 0 else float('inf')
        print(f"Average Validation Loss: {avg_val_loss:.4f}")

        # Check for improvement
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), 'best_qa_model.pth')
            print("Validation loss improved. Model saved.\n")
        else:
            patience_counter += 1
            print(f"No improvement in validation loss for {patience_counter} epoch(s).")
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    # Inference after training
    model.eval()

    test_context = (
        "P. Christiaan Klieger, an anthropologist and scholar of the California Academy of Sciences in San Francisco, "
        "writes that the vice royalty of the Sakya regime installed by the Mongols established a patron and priest "
        "relationship between Tibetans and Mongol converts to Tibetan Buddhism. According to him, the Tibetan lamas "
        "and Mongol khans upheld a \"mutual role of religious prelate and secular patron,\" respectively. He adds that "
        "\"Although agreements were made between Tibetan leaders and Mongol khans, Ming and Qing emperors, it was the "
        "Republic of China and its Communist successors that assumed the former imperial tributaries and subject "
        "states as integral parts of the Chinese nation-state."
    )
    test_question = "What is anthropology a study of?"

    predicted_answer = predict_answer(model, tokenizer, test_context, test_question, MAX_SEQ_LENGTH, device)
    print("Predicted Answer:", predicted_answer)

if __name__ == "__main__":
    main()
