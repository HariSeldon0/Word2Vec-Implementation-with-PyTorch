import torch
from torch import nn
from utils.dataloader import get_dataloader_and_tokenizer
from utils.models import SkipGram
from utils.trainer import Trainer

batch_size = 35000
half_context_window = 4
min_freq = 50
embedding_size = 100
epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 0.025

dataloader, tokenizer = get_dataloader_and_tokenizer(
    batch_size=batch_size, half_context_window=half_context_window, min_freq=min_freq
)
print(f"vocab size {tokenizer.get_vocab_size()}")

model = SkipGram(vocab_size=tokenizer.get_vocab_size(), embedding_size=embedding_size)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()

trainer = Trainer(
    dataloader=dataloader,
    model=model,
    epochs=epochs,
    device=device,
    optimizer=optimizer,
    loss_fn=loss_fn,
)

trainer.train()

trainer.save_model(
    "/root/autodl-tmp/Word2Vec-Implementation-with-PyTorch/models/skipgram-5e.pth"
)
