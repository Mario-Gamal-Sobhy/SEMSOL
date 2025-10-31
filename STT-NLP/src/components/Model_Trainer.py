import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.config.entity import ModelTrainerConfig
from src import get_logger
from src.components.CustomDataset import CustomDataset
from src.model.SpeechToText import SpeechToText
from pathlib import Path
import os

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.logger = get_logger("ModelTrainer")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _get_data_loaders(self):
        train_dataset = CustomDataset(self.config.processed_train_path, self.config.char_map_file)
        test_dataset = CustomDataset(self.config.processed_test_path, self.config.char_map_file)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.params.model_trainer.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.params.model_trainer.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn
        )
        return train_loader, test_loader

    def _collate_fn(self, batch):
        spectrograms = [item[0].squeeze(0) for item in batch]
        labels = [item[1] for item in batch]
        # time dimension (T) is dim 0 after dataset transposes to (T, n_mels)
        input_lengths = [spec.shape[0] for spec in spectrograms]
        label_lengths = [len(label) for label in labels]

        # Pad spectrograms => (B, 1, max_T, n_mels)
        padded_spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1)

        # Concatenate labels for CTC targets and ensure Long dtype
        concatenated_labels = torch.cat([lbl.long() for lbl in labels])

        return (
            padded_spectrograms,
            concatenated_labels,
            torch.IntTensor(input_lengths),
            torch.IntTensor(label_lengths),
        )

    def train(self):
        self.logger.info("Starting model training...")
        self.logger.info(f"ModelTrainerConfig params: {self.config.params.model_trainer}")

        train_loader, test_loader = self._get_data_loaders()

        # Extract parameters
        p = self.config.params.model_trainer
        model = SpeechToText(
            n_cnn_layers=p.n_cnn_layers,
            n_rnn_layers=p.n_rnn_layers,
            rnn_dim=p.rnn_dim,
            n_class=p.n_class,
            n_feats=p.n_feats,
            cnn_out_channels=p.cnn_out_channels,
            stride=p.stride,
            dropout=p.dropout
        ).to(self.device)
        self.logger.info(f"Model type: {type(model)}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=p.learning_rate)

        for epoch in range(p.epochs):
            model.train()
            train_loss = 0
            for batch_idx, (data, target, input_lengths, target_lengths) in enumerate(train_loader):
                data, target = data.to(self.device), target.long().to(self.device)

                output, output_lengths = model(data, input_lengths)
                optimizer.zero_grad()
                output = output.transpose(0, 1).log_softmax(2)
                loss = torch.nn.functional.ctc_loss(
                    log_probs=output,
                    targets=target,
                    input_lengths=output_lengths.cpu(),
                    target_lengths=target_lengths.cpu(),
                    blank=p.blank_index,
                    zero_infinity=True,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
                optimizer.step()
                train_loss += loss.item()

            self.logger.info(f"Epoch {epoch+1} Train Loss: {train_loss / max(1, len(train_loader))}")

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_idx, (data, target, input_lengths, target_lengths) in enumerate(test_loader):
                    data, target = data.to(self.device), target.long().to(self.device)
                    output, output_lengths = model(data, input_lengths)
                    output = output.transpose(0, 1).log_softmax(2)
                    loss = torch.nn.functional.ctc_loss(
                        log_probs=output,
                        targets=target,
                        input_lengths=output_lengths.cpu(),
                        target_lengths=target_lengths.cpu(),
                        blank=p.blank_index,
                        zero_infinity=True,
                    )
                    val_loss += loss.item()

            self.logger.info(f"Epoch {epoch+1} Validation Loss: {val_loss / max(1, len(test_loader))}")

        # Save the trained model
        model_save_path = Path(self.config.model_name)
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), str(model_save_path))
        self.logger.info(f"Model saved to {model_save_path}")
        self.logger.info("Model training completed.")
