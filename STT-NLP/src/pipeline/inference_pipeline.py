from src.config.Configuration import ConfigurationManager
from src.components.FeaturesExctractor import FeaturesExtractor
from src.model.SpeechToText import SpeechToText
import torch
import torchaudio
from src.components.Audio_Preprocess import AudioPreprocess
from src.exceptions import FileOperationError
from pathlib import Path
import tempfile
import io

class InferencePipeline:
    def __init__(self):
        self.config_manager = ConfigurationManager()
        self.model_trainer_config = self.config_manager.get_model_trainer_config()
        self.data_transformation_config = self.config_manager.get_data_transformation_config()
        self.features_extractor = FeaturesExtractor(config=self.data_transformation_config)
        self.preprocessor = torch.load(self.data_transformation_config.preprocessor_object_file , weights_only=False)
        self.audio_preprocessor = AudioPreprocess(config=self.data_transformation_config)
        self.model = self.load_model()
        self.char_map = {v: k for k, v in self.features_extractor.char_map.items()}

    def load_model(self) -> SpeechToText:
        try:
            p = self.model_trainer_config.params.model_trainer
            model = SpeechToText(
                n_cnn_layers=p.n_cnn_layers,
                n_rnn_layers=p.n_rnn_layers,
                rnn_dim=p.rnn_dim,
                n_class=p.n_class,
                n_feats=p.n_feats,
                cnn_out_channels=p.cnn_out_channels,
                stride=p.stride,
                dropout=p.dropout,
            )
            model.load_state_dict(torch.load(self.model_trainer_config.model_name))
            model.eval()
            return model
        except Exception as e:
            raise FileOperationError(f"Error loading model from {self.model_trainer_config.model_name}: {e}")

    def predict(self, audio_input) -> str:
        try:
            # Accept path-like or in-memory BytesIO
            if isinstance(audio_input, (str, Path)):
                path = Path(audio_input)
            elif isinstance(audio_input, io.BytesIO):
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(audio_input.getvalue())
                    tmp.flush()
                    path = Path(tmp.name)
            else:
                raise FileOperationError("Unsupported audio input type; provide file path or BytesIO.")

            # Preprocess to mel-spec (n_mels, time)
            spec_nt: torch.Tensor = self.audio_preprocessor._process_audio(path, self.preprocessor)
            # Shape to (1, 1, time, n_mels) like training
            spec_bctf = spec_nt.unsqueeze(0).unsqueeze(0).transpose(2, 3)

            # Make prediction
            with torch.no_grad():
                input_lengths = torch.tensor([spec_bctf.shape[2]], dtype=torch.int32)
                output, _ = self.model(spec_bctf, input_lengths)

            # Greedy decode
            output = torch.argmax(output, dim=2).detach().cpu().numpy()
            text: str = "".join([self.char_map.get(int(i), "") for i in output[0]])
            return text
        except Exception as e:
            raise FileOperationError(f"Error during inference for audio input: {e}")
