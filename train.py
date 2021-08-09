from utils.tokenizer import Tokenizer
from utils.data_loader import DataLoader
from models.transformer import Transformer
from conf import *
import torch
import torch.nn as nn
from torch import optim
from utils.timer import epoch_time

import math
import time


# 모델 학습(train) 함수
def train(model, iterator, optimizer, criterion, clip):
    model.train() # 학습 모드
    epoch_loss = 0

    # 전체 학습 데이터를 확인하며
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        # 출력 단어의 마지막 인덱스(<eos>)는 제외
        # 입력을 할 때는 <sos>부터 시작하도록 처리
        output = model(src, trg[:,:-1])

        # output: [배치 크기, trg_len - 1, output_dim]
        # trg: [배치 크기, trg_len]

        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        # 출력 단어의 인덱스 0(<sos>)은 제외
        trg = trg[:,1:].contiguous().view(-1)

        # output: [배치 크기 * trg_len - 1, output_dim]
        # trg: [배치 크기 * trg len - 1]

        # 모델의 출력 결과와 타겟 문장을 비교하여 손실 계산
        loss = criterion(output, trg)
        loss.backward() # 기울기(gradient) 계산

        # 기울기(gradient) clipping 진행
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        # 파라미터 업데이트
        optimizer.step()

        # 전체 손실 값 계산
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


# 모델 평가(evaluate) 함수
def evaluate(model, iterator, criterion):
    model.eval() # 평가 모드
    epoch_loss = 0

    with torch.no_grad():
        # 전체 평가 데이터를 확인하며
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            # 출력 단어의 마지막 인덱스(<eos>)는 제외
            # 입력을 할 때는 <sos>부터 시작하도록 처리
            output= model(src, trg[:,:-1])

            # output: [배치 크기, trg_len - 1, output_dim]
            # trg: [배치 크기, trg_len]

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            # 출력 단어의 인덱스 0(<sos>)은 제외
            trg = trg[:,1:].contiguous().view(-1)

            # output: [배치 크기 * trg_len - 1, output_dim]
            # trg: [배치 크기 * trg len - 1]

            # 모델의 출력 결과와 타겟 문장을 비교하여 손실 계산
            loss = criterion(output, trg)

            # 전체 손실 값 계산
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


if __name__ == '__main__':
    tokenizer = Tokenizer()
    dataloader = DataLoader(tokenizer_de=tokenizer.tokenize_de, tokenizer_en=tokenizer.tokenize_en, init_token='<sos>', eos_token='<eos>')
    train_dataset, valid_dataset, test_dataset = dataloader.make_dataset()
    dataloader.build_vocab(train_dataset=train_dataset, min_freq=2)
    train_iterator, valid_iterator, test_iterator = dataloader.make_iter(train_dataset=train_dataset, valid_dataset=valid_dataset, test_dataset=test_dataset, batch_size=batch_size, device=device)

    src_pad_idx = dataloader.source.vocab.stoi[dataloader.source.pad_token]
    trg_pad_idx = dataloader.target.vocab.stoi[dataloader.target.pad_token]

    src_vocab_size = len(dataloader.source.vocab)
    trg_vocab_size = len(dataloader.target.vocab)

    model = Transformer(src_vocab_size=src_vocab_size,
                        trg_vocab_size=trg_vocab_size,
                        src_pad_idx=src_pad_idx,
                        trg_pad_idx=trg_pad_idx,
                        embed_size=embed_size,
                        num_layers=num_layers,
                        forward_expansion=forward_expansion,
                        heads=heads,
                        dropout=dropout,
                        device=device,
                        max_length=max_length).to(device)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    def initialize_weights(m):
        if hasattr(m, 'weight') and m.weight.dim() > 1:
            nn.init.xavier_uniform_(m.weight.data)
    model.apply(initialize_weights)

    # Adam optimizer로 학습 최적화
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=weight_decay, eps=adam_eps)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                     verbose=True,
                                                     factor=factor,
                                                     patience=patience)

    criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)

    for epoch in range(N_EPOCHS):
        start_time = time.time() # 시작 시간 기록

        train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, valid_iterator, criterion)

        end_time = time.time() # 종료 시간 기록
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'transformer_german_to_english.pt')

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):.3f}')
        print(f'\tValidation Loss: {valid_loss:.3f} | Validation PPL: {math.exp(valid_loss):.3f}')


