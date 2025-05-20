import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
from transformers import ClapModel, ClapProcessor as HFClapProcessor

class ClapAudioLoader:
    """
    Audio loader class that loads and processes audio files for CLAP embedding
    """
    def __init__(self, target_sample_rate: int = 48000) -> None:
        self.target_sample_rate = target_sample_rate

    def load_audio(self, uri: str) -> Optional[Dict[str, Any]]:
        """Load audio file from URI"""
        if uri is None:
            return None

        try:
            import librosa
            waveform, sample_rate = librosa.load(uri, sr=self.target_sample_rate, mono=True)
            return {"waveform": waveform, "uri": uri}
        except Exception as e:
            print(f"Error loading audio file {uri}: {str(e)}")
            return None

    def __call__(self, uris: List[Optional[str]]) -> List[Optional[Dict[str, Any]]]:
        """Process multiple URIs"""
        return [self.load_audio(uri) for uri in uris]


class ClapEmbeddingFunction:
    """
    Embedding function for CLAP that can embed both audio and text
    """
    def __init__(
        self,
        model_name: str = "laion/larger_clap_general",
        device: str = None
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading CLAP model from {model_name} on {device}...")
        self.model = ClapModel.from_pretrained(model_name).to(device)
        self.processor = HFClapProcessor.from_pretrained(model_name)
        self.device = device
        print("Model loaded successfully")

    def encode_audio(self, audio: np.ndarray) -> np.ndarray:
        """Encode audio using CLAP"""
        inputs = self.processor(audios=audio, sampling_rate=48000, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            audio_embedding = self.model.get_audio_features(**inputs)
        
        return audio_embedding.squeeze().cpu().numpy()

    def encode_text(self, text: str) -> np.ndarray:
        """Encode text using CLAP"""
        inputs = self.processor(text=text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_embedding = self.model.get_text_features(**inputs)
        
        return text_embedding.squeeze().cpu().numpy()

    def __call__(self, input_data: Union[List[str], List[Dict[str, Any]]]) -> List[np.ndarray]:
        """Process either text or audio inputs"""
        embeddings = []
        
        for item in input_data:
            if isinstance(item, dict) and "waveform" in item:
                embeddings.append(self.encode_audio(item["waveform"]))
            elif isinstance(item, str):
                embeddings.append(self.encode_text(item))
            else:
                raise ValueError(f"Unsupported input type: {type(item)}")
        
        return embeddings 