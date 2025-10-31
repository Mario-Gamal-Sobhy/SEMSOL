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

        worker_count = max(1, (os.cpu_count() or 2) - 1)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.params.model_trainer.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
            num_workers=worker_count,
            persistent_workers=True,
            prefetch_factor=2,
            pin_memory=False,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.params.model_trainer.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
            num_workers=worker_count,
            persistent_workers=True,
            prefetch_factor=2,
            pin_memory=False,
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

        loaders = self._get_data_loaders()
        if isinstance(loaders, tuple) and len(loaders) == 2:
            train_loader, test_loader = loaders
        else:
            train_loader, test_loader = loaders, None

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

        # Checkpoint setup and auto-resume
        ckpt_dir = Path(self.config.root_dir) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        latest_ckpt = ckpt_dir / "latest.pt"
        start_epoch = 1
        if latest_ckpt.exists():
            try:
                state = torch.load(latest_ckpt, map_location=self.device)
                model.load_state_dict(state["model_state_dict"]) 
                optimizer.load_state_dict(state["optimizer_state_dict"]) 
                start_epoch = int(state.get("epoch", 0)) + 1
                self.logger.info(f"Resuming from checkpoint: epoch {start_epoch}")
            except Exception as e:
                self.logger.info(f"Could not load checkpoint (starting fresh): {e}")

        # Training loop with epoch-level checkpoints
        for epoch in range(start_epoch, p.epochs + 1):
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

            self.logger.info(f"Epoch {epoch} Train Loss: {train_loss / max(1, len(train_loader))}")

            model.eval()
            if test_loader is not None:
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
                self.logger.info(f"Epoch {epoch} Validation Loss: {val_loss / max(1, len(test_loader))}")
            else:
                self.logger.info("No test loader available; skipping validation.")

            # Save checkpoint (epoch)
            epoch_ckpt = ckpt_dir / f"epoch_{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "params": dict(p.__dict__) if hasattr(p, "__dict__") else None,
            }, str(epoch_ckpt))
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "params": dict(p.__dict__) if hasattr(p, "__dict__") else None,
            }, str(latest_ckpt))

            # Keep only last 3 checkpoints
            ckpts = sorted(ckpt_dir.glob("epoch_*.pt"), key=lambda x: int(x.stem.split("_")[-1]))
            while len(ckpts) > 3:
                old = ckpts.pop(0)
                try:
                    old.unlink()
                except Exception:
                    pass

        # Save the trained model
        model_save_path = Path(self.config.model_name)
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), str(model_save_path))
        self.logger.info(f"Model saved to {model_save_path}")
        self.logger.info("Model training completed.")
