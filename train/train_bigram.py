"""训练 Bigram baseline,保存到 checkpoints/bigram.pt。"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import BigramCfg, CHECKPOINT_DIR, TOKENIZER_FILE
from model.bigram import BigramModel
from tokenizer.tokenizer import CharTokenizer
from train.utils import get_device, load_data, make_get_batch, train_loop


def main() -> None:
    device = get_device()
    print(f'device = {device}')

    tokenizer = CharTokenizer.load(TOKENIZER_FILE)
    train_data, val_data = load_data(device)
    get_batch = make_get_batch(train_data, val_data)

    model = BigramModel(tokenizer.vocab_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'参数量: {n_params / 1e6:.2f}M  (vocab^2 = {tokenizer.vocab_size}^2)')

    optimizer = torch.optim.AdamW(model.parameters(), lr=BigramCfg.learning_rate)
    train_loop(model, optimizer, get_batch, num_steps=BigramCfg.num_steps)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / 'bigram.pt'
    torch.save(model.state_dict(), ckpt_path)
    print(f'[DONE] 模型已保存到 {ckpt_path}')


if __name__ == '__main__':
    main()
