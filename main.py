“””
Articulatory-Enhanced Mispronunciation Detection and Diagnosis Models:
A Multi-dimensional Error Analysis

Complete implementation based on:
Wei, Cucchiarini, van Hout, Strik (SLaTE 2025)

This program implements:

1. Audio feature extraction (39-dim MFCC, 83-dim FBank+Pitch)
1. AF (Articulatory Feature) classifiers (6 DNN-HMM heads)
1. Five MDD model configurations: RS, FP, M1, FT, M2
1. Two output frameworks: PHN (phoneme-based) and ART (articulatory-based)
1. Full evaluation metrics: DA, FAR, FRR, DER, MCC
1. Dataset handling for L2-ARCTIC and LibriSpeech
1. Training loops with joint CTC/Attention loss
   “””

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset

torch.manual_seed(42)

# =============================================================================

# 1. Phoneme & Articulatory Label Definitions (Section 2.2)

# =============================================================================

# CMU phoneme set used in L2-ARCTIC

PHONEME_LIST = [
“<blank>”, “<sos>”, “<eos>”,
“AA”, “AE”, “AH”, “AO”, “AW”, “AY”,
“B”, “CH”, “D”, “DH”,
“EH”, “ER”, “EY”,
“F”,
“G”,
“HH”,
“IH”, “IY”,
“JH”,
“K”,
“L”,
“M”, “N”, “NG”,
“OW”, “OY”,
“P”,
“R”,
“S”, “SH”, “SIL”,
“T”, “TH”,
“UH”, “UW”,
“V”,
“W”,
“Y”,
“Z”, “ZH”,
]
PHONEME_TO_IDX = {p: i for i, p in enumerate(PHONEME_LIST)}
NUM_PHONEMES = len(PHONEME_LIST)

# — AF category definitions (Section 2.2) —

AF_CONFIG = {
“backness”: [“Front”, “Central”, “Back”, “Back2front”],          # 4
“height”: [“High”, “Middle”, “Low”, “Low2high”],                 # 4
“roundness”: [“Rounded”, “Unrounded”, “Rounded2unrounded”],      # 3
“manner”: [“Affricate”, “Fricative”, “Nasal”, “Stop”, “Approximant”],  # 5
“place”: [
“Alveolar”, “Bilabial”, “Dental”, “Glottal”,
“Labiodental”, “Palatal”, “Post-Alveolar”, “Velar”,
],  # 8
“voicing”: [“Voiced”, “Unvoiced”],                               # 2
}
AF_NUM_CLASSES = {k: len(v) for k, v in AF_CONFIG.items()}
TOTAL_AF_DIM = sum(AF_NUM_CLASSES.values())  # 26

# Vowel AFs: backness, height, roundness

# Consonant AFs: manner, place, voicing

VOWEL_AF_KEYS = [“backness”, “height”, “roundness”]
CONSONANT_AF_KEYS = [“manner”, “place”, “voicing”]

# Phoneme-to-AF mapping for ART output framework

# Each phoneme maps to its articulatory label string

VOWELS = {“AA”, “AE”, “AH”, “AO”, “AW”, “AY”, “EH”, “ER”, “EY”,
“IH”, “IY”, “OW”, “OY”, “UH”, “UW”}

# Articulatory labels for ART framework output

# Format: “backness_height_roundness” for vowels,

# “manner_place_voicing” for consonants

PHONEME_TO_ART = {
# Vowels: backness_height_roundness
“AA”: “Back_Low_Unrounded”,
“AE”: “Front_Low_Unrounded”,
“AH”: “Central_Low_Unrounded”,
“AO”: “Back_Low_Rounded”,
“AW”: “Back2front_Low2high_Rounded2unrounded”,
“AY”: “Back2front_Low2high_Unrounded”,
“EH”: “Front_Middle_Unrounded”,
“ER”: “Central_Middle_Unrounded”,
“EY”: “Front_Middle_Unrounded”,
“IH”: “Front_High_Unrounded”,
“IY”: “Front_High_Unrounded”,
“OW”: “Back_Middle_Rounded”,
“OY”: “Back2front_Low2high_Rounded2unrounded”,
“UH”: “Back_High_Rounded”,
“UW”: “Back_High_Rounded”,
# Consonants: manner_place_voicing
“B”: “Stop_Bilabial_Voiced”,
“CH”: “Affricate_Post-Alveolar_Unvoiced”,
“D”: “Stop_Alveolar_Voiced”,
“DH”: “Fricative_Dental_Voiced”,
“F”: “Fricative_Labiodental_Unvoiced”,
“G”: “Stop_Velar_Voiced”,
“HH”: “Fricative_Glottal_Unvoiced”,
“JH”: “Affricate_Post-Alveolar_Voiced”,
“K”: “Stop_Velar_Unvoiced”,
“L”: “Approximant_Alveolar_Voiced”,
“M”: “Nasal_Bilabial_Voiced”,
“N”: “Nasal_Alveolar_Voiced”,
“NG”: “Nasal_Velar_Voiced”,
“P”: “Stop_Bilabial_Unvoiced”,
“R”: “Approximant_Alveolar_Voiced”,
“S”: “Fricative_Alveolar_Unvoiced”,
“SH”: “Fricative_Post-Alveolar_Unvoiced”,
“T”: “Stop_Alveolar_Unvoiced”,
“TH”: “Fricative_Dental_Unvoiced”,
“V”: “Fricative_Labiodental_Voiced”,
“W”: “Approximant_Bilabial_Voiced”,
“Y”: “Approximant_Palatal_Voiced”,
“Z”: “Fricative_Alveolar_Voiced”,
“ZH”: “Fricative_Post-Alveolar_Voiced”,
“SIL”: “SIL”,
}

# Build ART label set

ART_LABELS = sorted(set(PHONEME_TO_ART.values()))
ART_LABELS = [”<blank>”, “<sos>”, “<eos>”] + ART_LABELS
ART_TO_IDX = {a: i for i, a in enumerate(ART_LABELS)}
NUM_ART_LABELS = len(ART_LABELS)

def phoneme_seq_to_art_seq(phoneme_seq: List[str]) -> List[str]:
“”“Convert phoneme sequence to articulatory label sequence for ART framework.”””
return [PHONEME_TO_ART.get(p, “SIL”) for p in phoneme_seq]

# =============================================================================

# 2. Audio Feature Extraction (Section 2.2)

# =============================================================================

class AudioFeatureExtractor:
“””
Extracts acoustic features as described in Section 2.2:
- 39-dim MFCCs: 13 coefficients + delta + delta-delta (for AF classifiers)
- 83-dim FBank+Pitch: 80 FBank + 3 pitch features (for Conformer M1)
“””

```
def __init__(self, sample_rate: int = 16000):
    self.sample_rate = sample_rate

    # 13-dim MFCC extraction
    self.mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=13,
        melkwargs={
            "n_fft": 400,       # 25ms window at 16kHz
            "hop_length": 160,  # 10ms hop
            "n_mels": 23,
            "center": False,
        },
    )

    # 80-dim FBank extraction
    self.fbank_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=400,
        hop_length=160,
        n_mels=80,
    )

def compute_deltas(self, features: torch.Tensor) -> torch.Tensor:
    """
    Compute delta and delta-delta features.
    Input:  (Channels, Time)
    Output: (3*Channels, Time)
    """
    delta = torchaudio.functional.compute_deltas(features)
    delta_delta = torchaudio.functional.compute_deltas(delta)
    return torch.cat([features, delta, delta_delta], dim=0)

def extract_mfcc_39(self, waveform: torch.Tensor) -> torch.Tensor:
    """
    Extract 39-dim MFCCs for AF classifier training.
    Input:  waveform (1, Samples) or (Samples,)
    Output: (Time, 39)
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    mfcc_13 = self.mfcc_transform(waveform).squeeze(0)  # (13, Time)
    mfcc_39 = self.compute_deltas(mfcc_13)               # (39, Time)
    return mfcc_39.transpose(0, 1)                        # (Time, 39)

def extract_fbank_80(self, waveform: torch.Tensor) -> torch.Tensor:
    """
    Extract 80-dim log-FBank features.
    Input:  waveform (1, Samples) or (Samples,)
    Output: (Time, 80)
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    mel_spec = self.fbank_transform(waveform).squeeze(0)  # (80, Time)
    log_fbank = torch.log(mel_spec + 1e-8)
    return log_fbank.transpose(0, 1)                       # (Time, 80)

def extract_pitch_3(self, waveform: torch.Tensor) -> torch.Tensor:
    """
    Extract 3-dim pitch features (pitch, POV, delta-pitch).
    Uses Kaldi-style pitch extraction.
    Input:  waveform (1, Samples) or (Samples,)
    Output: (Time, 3)
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    try:
        # torchaudio Kaldi-compatible pitch extraction
        pitch_feat = torchaudio.functional.compute_kaldi_pitch(
            waveform, sample_rate=self.sample_rate,
            frame_length=25.0, frame_shift=10.0
        )
        # pitch_feat: (Batch, Time, 2) -> [NCCF, pitch]
        pitch_feat = pitch_feat.squeeze(0)  # (Time, 2)
        nccf = pitch_feat[:, 0:1]   # Normalized cross-correlation (POV proxy)
        pitch = pitch_feat[:, 1:2]   # Pitch frequency
        delta_pitch = torchaudio.functional.compute_deltas(
            pitch.transpose(0, 1)
        ).transpose(0, 1)
        return torch.cat([pitch, nccf, delta_pitch], dim=1)  # (Time, 3)
    except Exception:
        # Fallback: zero features if Kaldi pitch unavailable
        num_frames = (waveform.shape[-1] - 400) // 160 + 1
        return torch.zeros(max(num_frames, 1), 3)

def extract_fbank_pitch_83(self, waveform: torch.Tensor) -> torch.Tensor:
    """
    Extract 83-dim FBank+Pitch features (Section 2.2).
    80-dim FBank concatenated with 3-dim pitch features.
    Input:  waveform (1, Samples) or (Samples,)
    Output: (Time, 83)
    """
    fbank = self.extract_fbank_80(waveform)     # (T, 80)
    pitch = self.extract_pitch_3(waveform)       # (T', 3)

    # Align lengths (may differ by 1-2 frames)
    min_len = min(fbank.size(0), pitch.size(0))
    fbank = fbank[:min_len]
    pitch = pitch[:min_len]

    return torch.cat([fbank, pitch], dim=1)      # (T, 83)
```

# =============================================================================

# 3. AF Classifier - DNN-HMM (Section 2.2)

# =============================================================================

class AFClassifier(nn.Module):
“””
DNN-HMM Articulatory Feature Classifier (Section 2.2).

```
Architecture: 6 hidden layers, 2048 sigmoid units each.
Trained on 100-hour LibriSpeech clean subset with 39-dim MFCC input.
Produces 6 separate softmax posteriors, concatenated into a 26-dim
composite AF vector per frame.

In the DNN-HMM paradigm, the DNN replaces the GMM in a conventional
HMM system. Frame-level AF labels are obtained via forced alignment
using an HMM topology, and the DNN is trained to predict these labels.
"""

def __init__(self, input_dim: int = 39, hidden_dim: int = 2048,
             num_hidden_layers: int = 6):
    super().__init__()

    # Build shared encoder: 6 hidden layers with sigmoid activation
    layers = []
    layers.append(nn.Linear(input_dim, hidden_dim))
    layers.append(nn.Sigmoid())
    for _ in range(num_hidden_layers - 1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.Sigmoid())
    self.encoder = nn.Sequential(*layers)

    # 6 independent classification heads (one per AF category)
    self.heads = nn.ModuleDict({
        name: nn.Linear(hidden_dim, num_cls)
        for name, num_cls in AF_NUM_CLASSES.items()
    })

def forward(self, mfcc: torch.Tensor) -> torch.Tensor:
    """
    Args:
        mfcc: (Batch, Time, 39) MFCC features
    Returns:
        af_vector: (Batch, Time, 26) composite AF posteriors
    """
    h = self.encoder(mfcc)  # (B, T, 2048)

    posteriors = []
    for name in AF_CONFIG.keys():  # Fixed iteration order
        logits = self.heads[name](h)            # (B, T, num_classes)
        prob = F.softmax(logits, dim=-1)         # (B, T, num_classes)
        posteriors.append(prob)

    return torch.cat(posteriors, dim=-1)  # (B, T, 26)

def compute_loss(self, mfcc: torch.Tensor,
                 targets: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Compute cross-entropy loss for each AF head separately.
    Args:
        mfcc: (Batch, Time, 39)
        targets: dict mapping AF category name -> (Batch, Time) integer labels
    Returns:
        total_loss: scalar
    """
    h = self.encoder(mfcc)
    total_loss = torch.tensor(0.0, device=mfcc.device)

    for name in AF_CONFIG.keys():
        logits = self.heads[name](h)  # (B, T, C)
        B, T, C = logits.shape
        loss = F.cross_entropy(
            logits.reshape(B * T, C),
            targets[name].reshape(B * T),
            ignore_index=-1,
        )
        total_loss = total_loss + loss

    return total_loss / len(AF_CONFIG)
```

# =============================================================================

# 4. MDD Models (Section 2.2)

# =============================================================================

# — Conformer-based E2E Models: RS, FP, M1 —

class ConformerMDD(nn.Module):
“””
Custom E2E Conformer model with joint CTC/Attention (Section 2.2).

```
Unified architecture for RS, FP, and M1 configurations:
  - RS: input_dim = raw speech features (e.g., 80-dim spectrogram)
  - FP: input_dim = 83 (80 FBank + 3 pitch)
  - M1: input_dim = 109 (83 FP + 26 AF posteriors)

Architecture:
  - Conformer encoder (12 layers, 4 heads, FFN 1024, kernel 31)
  - Transformer decoder (6 layers)
  - CTC head for joint CTC/Attention decoding
"""

def __init__(self, input_dim: int, num_classes: int, d_model: int = 256,
             encoder_layers: int = 12, decoder_layers: int = 6,
             num_heads: int = 4, ffn_dim: int = 1024,
             depthwise_conv_kernel: int = 31, dropout: float = 0.1):
    super().__init__()

    self.d_model = d_model

    # Input projection to d_model dimension
    self.input_proj = nn.Linear(input_dim, d_model)

    # Conformer encoder
    self.encoder = torchaudio.models.Conformer(
        input_dim=d_model,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        num_layers=encoder_layers,
        depthwise_conv_kernel_size=depthwise_conv_kernel,
        dropout=dropout,
    )

    # CTC head (operates on encoder output)
    self.ctc_head = nn.Linear(d_model, num_classes)

    # Transformer decoder for attention-based decoding
    decoder_layer = nn.TransformerDecoderLayer(
        d_model=d_model, nhead=num_heads, dim_feedforward=ffn_dim,
        dropout=dropout, batch_first=False,
    )
    self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)

    # Decoder output projection
    self.decoder_embed = nn.Embedding(num_classes, d_model)
    self.decoder_proj = nn.Linear(d_model, num_classes)

    # Positional encoding for decoder
    self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)

def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor,
            decoder_targets: Optional[torch.Tensor] = None
            ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Args:
        inputs:          (Batch, Time, input_dim) acoustic features
        input_lengths:   (Batch,) valid lengths per sample
        decoder_targets: (Batch, TargetLen) for teacher forcing (training)
    Returns:
        ctc_logits:      (Batch, Time, num_classes) CTC output
        enc_lengths:     (Batch,) encoder output lengths
        att_logits:      (Batch, TargetLen, num_classes) or None
    """
    # Project input to d_model
    x = self.input_proj(inputs)  # (B, T, d_model)

    # Conformer encoder
    encoder_out, enc_lengths = self.encoder(x, input_lengths)  # (B, T, d_model)

    # CTC branch
    ctc_logits = self.ctc_head(encoder_out)  # (B, T, num_classes)

    # Attention decoder branch (teacher-forcing during training)
    att_logits = None
    if decoder_targets is not None:
        # Embed target tokens and add positional encoding
        tgt_embed = self.decoder_embed(decoder_targets)  # (B, S, d_model)
        tgt_embed = tgt_embed.permute(1, 0, 2)           # (S, B, d_model)
        tgt_embed = self.pos_encoding(tgt_embed)

        # Encoder output as memory
        memory = encoder_out.permute(1, 0, 2)  # (T, B, d_model)

        # Causal mask for autoregressive decoding
        tgt_len = tgt_embed.size(0)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(
            inputs.device
        )

        # Memory key padding mask
        max_len = encoder_out.size(1)
        memory_key_padding_mask = torch.arange(max_len, device=inputs.device).unsqueeze(0) >= enc_lengths.unsqueeze(1)

        decoded = self.decoder(
            tgt=tgt_embed, memory=memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )  # (S, B, d_model)

        att_logits = self.decoder_proj(decoded.permute(1, 0, 2))  # (B, S, num_classes)

    return ctc_logits, enc_lengths, att_logits
```

class PositionalEncoding(nn.Module):
“”“Standard sinusoidal positional encoding.”””

```
def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
    super().__init__()
    self.dropout = nn.Dropout(p=dropout)

    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
    self.register_buffer("pe", pe)

def forward(self, x: torch.Tensor) -> torch.Tensor:
    """x: (Seq, Batch, d_model)"""
    x = x + self.pe[:x.size(0)]
    return self.dropout(x)
```

# — Fine-tuned Wav2Vec 2.0 Models: FT, M2 —

class XLSRMDD(nn.Module):
“””
Fine-tuned XLSR (Wav2Vec 2.0) for MDD (Section 2.2).

```
FT configuration: XLSR embeddings only -> Transformer decoder + CTC
M2 configuration: XLSR embeddings + AF posteriors -> Fusion -> Decoder + CTC

XLSR-53 produces 1024-dim embeddings at 20ms frame rate.
"""

def __init__(self, num_classes: int, af_dim: int = 0,
             d_model: int = 512, decoder_layers: int = 6,
             num_heads: int = 8, freeze_feature_extractor: bool = True):
    """
    Args:
        num_classes: output vocabulary size
        af_dim: AF vector dimension (0 for FT baseline, 26 for M2)
        d_model: decoder hidden dimension
    """
    super().__init__()

    self.af_dim = af_dim
    self.d_model = d_model
    self.w2v_dim = 1024  # XLSR-53 hidden size

    # Load pretrained XLSR model
    try:
        from transformers import Wav2Vec2Model
        self.wav2vec = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-large-xlsr-53"
        )
        if freeze_feature_extractor:
            self.wav2vec.feature_extractor._freeze_parameters()
        self._use_real_w2v = True
    except Exception:
        print("Warning: Could not load XLSR model. Using mock encoder.")
        self.wav2vec = nn.Linear(1, self.w2v_dim)
        self._use_real_w2v = False

    # Fusion projection: (w2v_dim + af_dim) -> d_model
    fusion_input_dim = self.w2v_dim + af_dim
    self.fusion_proj = nn.Sequential(
        nn.Linear(fusion_input_dim, d_model),
        nn.LayerNorm(d_model),
        nn.ReLU(),
        nn.Dropout(0.1),
    )

    # CTC head
    self.ctc_head = nn.Linear(d_model, num_classes)

    # Transformer decoder
    decoder_layer = nn.TransformerDecoderLayer(
        d_model=d_model, nhead=num_heads,
        dim_feedforward=2048, dropout=0.1, batch_first=False,
    )
    self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)

    # Decoder components
    self.decoder_embed = nn.Embedding(num_classes, d_model)
    self.decoder_proj = nn.Linear(d_model, num_classes)
    self.pos_encoding = PositionalEncoding(d_model)

def _extract_wav2vec_features(self, raw_audio: torch.Tensor) -> torch.Tensor:
    """Extract embeddings from XLSR encoder."""
    if self._use_real_w2v:
        outputs = self.wav2vec(raw_audio)
        return outputs.last_hidden_state  # (B, T_w2v, 1024)
    else:
        # Mock: simulate w2v output shape
        B = raw_audio.shape[0]
        T = raw_audio.shape[-1] // 320  # ~20ms stride at 16kHz
        return torch.randn(B, max(T, 1), self.w2v_dim, device=raw_audio.device)

def _align_and_fuse(self, embeddings: torch.Tensor,
                    af_vectors: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Align XLSR embeddings (20ms stride) with AF vectors (10ms stride)
    and perform frame-by-frame fusion.

    XLSR outputs at ~20ms, MFCCs at 10ms. Downsample AF by factor of 2.
    """
    if af_vectors is not None and self.af_dim > 0:
        # Downsample AF vectors: take every 2nd frame to match XLSR rate
        af_downsampled = af_vectors[:, ::2, :]

        # Truncate to minimum length
        min_len = min(embeddings.size(1), af_downsampled.size(1))
        embeddings = embeddings[:, :min_len, :]
        af_downsampled = af_downsampled[:, :min_len, :]

        # Fusion by concatenation (Section 2.2: T × (D1 + D2))
        fused = torch.cat([embeddings, af_downsampled], dim=-1)
    else:
        fused = embeddings

    return fused

def forward(self, raw_audio: torch.Tensor,
            af_vectors: Optional[torch.Tensor] = None,
            decoder_targets: Optional[torch.Tensor] = None
            ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Args:
        raw_audio:       (Batch, Samples) raw waveform
        af_vectors:      (Batch, Time_AF, 26) AF posteriors (None for FT)
        decoder_targets: (Batch, TargetLen) for teacher forcing
    Returns:
        ctc_logits:  (Batch, Time, num_classes)
        att_logits:  (Batch, TargetLen, num_classes) or None
    """
    # 1. Extract XLSR embeddings
    embeddings = self._extract_wav2vec_features(raw_audio)

    # 2. Fuse with AFs if M2 configuration
    fused = self._align_and_fuse(embeddings, af_vectors)

    # 3. Project to decoder dimension
    projected = self.fusion_proj(fused)  # (B, T, d_model)

    # 4. CTC branch
    ctc_logits = self.ctc_head(projected)  # (B, T, num_classes)

    # 5. Attention decoder branch
    att_logits = None
    if decoder_targets is not None:
        memory = projected.permute(1, 0, 2)  # (T, B, d_model)

        tgt_embed = self.decoder_embed(decoder_targets).permute(1, 0, 2)
        tgt_embed = self.pos_encoding(tgt_embed)

        tgt_len = tgt_embed.size(0)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(
            raw_audio.device
        )

        decoded = self.decoder(tgt=tgt_embed, memory=memory, tgt_mask=tgt_mask)
        att_logits = self.decoder_proj(decoded.permute(1, 0, 2))

    return ctc_logits, att_logits
```

# =============================================================================

# 5. Joint CTC/Attention Loss

# =============================================================================

class JointCTCAttentionLoss(nn.Module):
“””
Joint CTC/Attention loss for training MDD models.
Loss = α * CTC_loss + (1 - α) * Attention_loss
“””

```
def __init__(self, ctc_weight: float = 0.3, blank_id: int = 0,
             ignore_id: int = -1):
    super().__init__()
    self.ctc_weight = ctc_weight
    self.ctc_loss = nn.CTCLoss(blank=blank_id, zero_infinity=True)
    self.att_loss = nn.CrossEntropyLoss(ignore_index=ignore_id)

def forward(self, ctc_logits: torch.Tensor, ctc_lengths: torch.Tensor,
            att_logits: Optional[torch.Tensor],
            targets: torch.Tensor, target_lengths: torch.Tensor
            ) -> torch.Tensor:
    """
    Args:
        ctc_logits:     (B, T, C) CTC output
        ctc_lengths:    (B,) encoder output lengths
        att_logits:     (B, S, C) attention decoder output (or None)
        targets:        (B, S) target label indices
        target_lengths: (B,) target lengths
    """
    # CTC loss: expects (T, B, C) log-probs
    ctc_log_probs = F.log_softmax(ctc_logits, dim=-1).permute(1, 0, 2)
    loss_ctc = self.ctc_loss(ctc_log_probs, targets, ctc_lengths, target_lengths)

    if att_logits is not None:
        B, S, C = att_logits.shape
        loss_att = self.att_loss(att_logits.reshape(B * S, C), targets.reshape(B * S))
        loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att
    else:
        loss = loss_ctc

    return loss
```

# =============================================================================

# 6. Dataset (Section 2.1)

# =============================================================================

class L2ArcticDataset(Dataset):
“””
Dataset for L2-ARCTIC corpus (Section 2.1).

```
Supports both PHN and ART output frameworks.
Data split: 12 train / 6 validation / 6 test speakers.

Each sample contains:
  - waveform (raw audio)
  - canonical phoneme sequence
  - actual (annotated) phoneme sequence
  - mispronunciation annotations
"""

def __init__(self, data_entries: List[Dict], feature_extractor: AudioFeatureExtractor,
             af_classifier: Optional[AFClassifier] = None,
             output_framework: str = "PHN",
             model_config: str = "FP"):
    """
    Args:
        data_entries: list of dicts with keys:
            'audio_path', 'canonical', 'actual', 'mispronunciations'
        feature_extractor: AudioFeatureExtractor instance
        af_classifier: pretrained AF classifier (for M1/M2 configs)
        output_framework: "PHN" or "ART"
        model_config: "RS", "FP", "M1", "FT", "M2"
    """
    self.data = data_entries
    self.extractor = feature_extractor
    self.af_classifier = af_classifier
    self.output_framework = output_framework
    self.model_config = model_config

    # Choose label mapping based on framework
    if output_framework == "PHN":
        self.label_to_idx = PHONEME_TO_IDX
        self.num_classes = NUM_PHONEMES
    else:  # ART
        self.label_to_idx = ART_TO_IDX
        self.num_classes = NUM_ART_LABELS

def __len__(self):
    return len(self.data)

def __getitem__(self, idx: int) -> Dict:
    entry = self.data[idx]

    # Load audio
    waveform, sr = torchaudio.load(entry["audio_path"])
    if sr != self.extractor.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, self.extractor.sample_rate)
    waveform = waveform.squeeze(0)  # (Samples,)

    result = {"waveform": waveform}

    # Extract features based on model configuration
    if self.model_config in ("RS", "FP", "M1"):
        if self.model_config == "RS":
            # Raw speech spectrogram (80-dim as baseline)
            features = self.extractor.extract_fbank_80(waveform)
        elif self.model_config == "FP":
            features = self.extractor.extract_fbank_pitch_83(waveform)
        elif self.model_config == "M1":
            fp_features = self.extractor.extract_fbank_pitch_83(waveform)

            # Compute AF posteriors
            mfcc = self.extractor.extract_mfcc_39(waveform).unsqueeze(0)
            with torch.no_grad():
                af_posteriors = self.af_classifier(mfcc).squeeze(0)  # (T, 26)

            # Align and fuse: T × (83 + 26) = T × 109
            min_len = min(fp_features.size(0), af_posteriors.size(0))
            features = torch.cat([
                fp_features[:min_len],
                af_posteriors[:min_len]
            ], dim=1)

        result["features"] = features
        result["feature_lengths"] = torch.tensor(features.size(0))

    elif self.model_config in ("FT", "M2"):
        result["raw_audio"] = waveform

        if self.model_config == "M2" and self.af_classifier is not None:
            mfcc = self.extractor.extract_mfcc_39(waveform).unsqueeze(0)
            with torch.no_grad():
                af_posteriors = self.af_classifier(mfcc).squeeze(0)
            result["af_vectors"] = af_posteriors

    # Prepare target labels
    actual_phonemes = entry["actual"]
    if self.output_framework == "ART":
        labels = phoneme_seq_to_art_seq(actual_phonemes)
        label_indices = [self.label_to_idx.get(l, 0) for l in labels]
    else:
        label_indices = [self.label_to_idx.get(p, 0) for p in actual_phonemes]

    result["targets"] = torch.tensor(label_indices, dtype=torch.long)
    result["target_lengths"] = torch.tensor(len(label_indices))
    result["canonical"] = entry["canonical"]
    result["actual"] = entry["actual"]
    result["mispronunciations"] = entry.get("mispronunciations", {})
    result["speaker"] = entry.get("speaker", "unknown")

    return result
```

class LibriSpeechAFDataset(Dataset):
“””
Dataset for training AF classifiers on LibriSpeech 100h clean subset.

```
Provides frame-level AF labels obtained via forced alignment (HMM).
In practice, labels come from a Kaldi-style forced alignment pipeline.
"""

def __init__(self, data_entries: List[Dict],
             feature_extractor: AudioFeatureExtractor):
    self.data = data_entries
    self.extractor = feature_extractor

def __len__(self):
    return len(self.data)

def __getitem__(self, idx: int) -> Dict:
    entry = self.data[idx]

    waveform, sr = torchaudio.load(entry["audio_path"])
    if sr != self.extractor.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, self.extractor.sample_rate)
    waveform = waveform.squeeze(0)

    mfcc = self.extractor.extract_mfcc_39(waveform)  # (T, 39)

    # Frame-level AF labels from forced alignment
    # Each key maps to a (T,) tensor of class indices
    af_labels = entry["af_labels"]  # Dict[str, Tensor]

    return {"mfcc": mfcc, "af_labels": af_labels}
```

# =============================================================================

# 7. Evaluation Metrics (Section 2.3, Equations 1-5)

# =============================================================================

def levenshtein_alignment(ref: List[str], hyp: List[str]
) -> List[Tuple[Optional[str], Optional[str], str]]:
“””
Perform Levenshtein alignment between reference and hypothesis sequences.

```
Returns list of (ref_token, hyp_token, operation) tuples where
operation is one of: 'match', 'substitution', 'insertion', 'deletion'.
"""
n, m = len(ref), len(hyp)

# DP table
dp = np.zeros((n + 1, m + 1), dtype=int)
for i in range(n + 1):
    dp[i][0] = i
for j in range(m + 1):
    dp[0][j] = j

for i in range(1, n + 1):
    for j in range(1, m + 1):
        if ref[i - 1] == hyp[j - 1]:
            dp[i][j] = dp[i - 1][j - 1]
        else:
            dp[i][j] = 1 + min(dp[i - 1][j],       # deletion
                               dp[i][j - 1],       # insertion
                               dp[i - 1][j - 1])   # substitution

# Backtrace
alignment = []
i, j = n, m
while i > 0 or j > 0:
    if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
        alignment.append((ref[i - 1], hyp[j - 1], "match"))
        i -= 1
        j -= 1
    elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
        alignment.append((ref[i - 1], hyp[j - 1], "substitution"))
        i -= 1
        j -= 1
    elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
        alignment.append((ref[i - 1], None, "deletion"))
        i -= 1
    else:
        alignment.append((None, hyp[j - 1], "insertion"))
        j -= 1

alignment.reverse()
return alignment
```

class MDDEvaluator:
“””
Evaluation metrics for MDD (Section 2.3).

```
Classification categories:
  - CA: Correct Acceptance (correct phoneme correctly predicted)
  - FR: False Rejection (correct phoneme incorrectly rejected)
  - CR: Correct Rejection (mispronunciation correctly detected)
    - CD: Correct Diagnosis (specific error correctly identified)
    - DE: Diagnosis Error (error detected but misidentified)
  - FA: False Acceptance (mispronunciation missed)

Metrics: DA, FAR, FRR, DER, MCC (Equations 1-5)
"""

def __init__(self):
    self.reset()

def reset(self):
    self.stats = {"CA": 0, "FR": 0, "CR": 0, "FA": 0, "CD": 0, "DE": 0}

def update(self, canonical_seq: List[str], predicted_seq: List[str],
           actual_seq: List[str]):
    """
    Update evaluation counts using sequence alignment.

    Args:
        canonical_seq: intended/reference phoneme sequence
        predicted_seq: model-predicted phoneme sequence
        actual_seq:    ground truth annotated phoneme sequence
                      (what the speaker actually produced)

    For each position in the canonical sequence:
      - If canonical == actual (correct pronunciation):
          - canonical == predicted -> CA
          - canonical != predicted -> FR
      - If canonical != actual (mispronunciation):
          - canonical == predicted -> FA (error missed)
          - canonical != predicted -> CR
              - predicted == actual -> CD (correct diagnosis)
              - predicted != actual -> DE (wrong diagnosis)
    """
    # Align canonical with predicted
    alignment_pred = levenshtein_alignment(canonical_seq, predicted_seq)
    # Align canonical with actual (ground truth)
    alignment_gt = levenshtein_alignment(canonical_seq, actual_seq)

    # Build position-based lookup from ground truth alignment
    gt_map = {}  # canonical_position -> actual_token
    can_pos = 0
    for ref_tok, hyp_tok, op in alignment_gt:
        if op == "match":
            gt_map[can_pos] = ref_tok  # correct pronunciation
            can_pos += 1
        elif op == "substitution":
            gt_map[can_pos] = hyp_tok  # mispronounced as hyp_tok
            can_pos += 1
        elif op == "deletion":
            gt_map[can_pos] = None     # phoneme deleted by speaker
            can_pos += 1
        # insertion: extra phoneme added by speaker (handled separately)

    # Evaluate predictions against ground truth
    can_pos = 0
    for ref_tok, hyp_tok, op in alignment_pred:
        if op in ("match", "substitution"):
            actual_tok = gt_map.get(can_pos)

            if actual_tok == ref_tok:
                # Ground truth: correctly pronounced
                if ref_tok == hyp_tok:
                    self.stats["CA"] += 1
                else:
                    self.stats["FR"] += 1
            else:
                # Ground truth: mispronounced
                if ref_tok == hyp_tok:
                    self.stats["FA"] += 1
                else:
                    self.stats["CR"] += 1
                    # Check diagnosis
                    if hyp_tok == actual_tok:
                        self.stats["CD"] += 1
                    else:
                        self.stats["DE"] += 1

            can_pos += 1
        elif op == "deletion":
            actual_tok = gt_map.get(can_pos)
            if actual_tok == ref_tok:
                self.stats["FR"] += 1   # correct phoneme erroneously deleted
            else:
                self.stats["CR"] += 1   # detected something wrong
                self.stats["DE"] += 1   # but deletion ≠ correct diagnosis
            can_pos += 1
        elif op == "insertion":
            # Model inserted extra phoneme - contributes to FR
            self.stats["FR"] += 1

def compute_metrics(self) -> Dict[str, float]:
    """
    Compute all metrics per Section 2.3 (Equations 1-5).

    DA  = (CA + CR) / (CA + CR + FA + FR)      (1)
    FAR = FA / (CR + FA)                        (2)
    FRR = FR / (CA + FR)                        (3)
    DER = DE / (CD + DE)                        (4)
    MCC = (CA*CR - FA*FR) /                     (5)
          sqrt((CA+FA)(CA+FR)(CR+FA)(CR+FR))
    """
    s = self.stats
    eps = 1e-10

    total = s["CA"] + s["CR"] + s["FA"] + s["FR"]
    da = (s["CA"] + s["CR"]) / (total + eps)

    far = s["FA"] / (s["CR"] + s["FA"] + eps)
    frr = s["FR"] / (s["CA"] + s["FR"] + eps)
    der = s["DE"] / (s["CD"] + s["DE"] + eps)

    # MCC (Equation 5)
    numerator = s["CA"] * s["CR"] - s["FA"] * s["FR"]
    denominator = np.sqrt(
        (s["CA"] + s["FA"]) * (s["CA"] + s["FR"]) *
        (s["CR"] + s["FA"]) * (s["CR"] + s["FR"])
    ) + eps
    mcc = numerator / denominator

    return {
        "DA": round(da * 100, 2),
        "FAR": round(far * 100, 2),
        "FRR": round(frr * 100, 2),
        "DER": round(der * 100, 2),
        "MCC": round(mcc, 4),
    }

def compute_per_error_der(self, error_type_counts: Dict[str, Dict[str, int]]
                          ) -> Dict[str, float]:
    """
    Compute DER for specific mispronunciation types (Section 3.4).
    E.g., DH/D, Z/S, IH/IY as in Table 4.

    Args:
        error_type_counts: dict mapping "X/Y" -> {"CD": n, "DE": m}
    Returns:
        dict mapping error type to DER percentage
    """
    results = {}
    for error_type, counts in error_type_counts.items():
        cd = counts.get("CD", 0)
        de = counts.get("DE", 0)
        if cd + de > 0:
            results[error_type] = round(de / (cd + de) * 100, 2)
        else:
            results[error_type] = 0.0
    return results
```

# =============================================================================

# 8. Training Pipeline

# =============================================================================

class MDDTrainer:
“”“Training pipeline for MDD models with joint CTC/Attention.”””

```
def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
             loss_fn: JointCTCAttentionLoss, device: torch.device,
             model_config: str = "FP"):
    self.model = model
    self.optimizer = optimizer
    self.loss_fn = loss_fn
    self.device = device
    self.model_config = model_config

def train_epoch(self, dataloader: DataLoader) -> float:
    """Run one training epoch. Returns average loss."""
    self.model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        self.optimizer.zero_grad()

        targets = batch["targets"].to(self.device)
        target_lengths = batch["target_lengths"].to(self.device)

        if self.model_config in ("RS", "FP", "M1"):
            features = batch["features"].to(self.device)
            feat_lengths = batch["feature_lengths"].to(self.device)

            # Prepare decoder input: shift right with <sos>
            sos_id = 1  # <sos> index
            decoder_input = torch.cat([
                torch.full((targets.size(0), 1), sos_id,
                           dtype=torch.long, device=self.device),
                targets[:, :-1]
            ], dim=1)

            ctc_logits, enc_lengths, att_logits = self.model(
                features, feat_lengths, decoder_input
            )

            loss = self.loss_fn(
                ctc_logits, enc_lengths, att_logits, targets, target_lengths
            )

        elif self.model_config in ("FT", "M2"):
            raw_audio = batch["raw_audio"].to(self.device)
            af_vectors = batch.get("af_vectors")
            if af_vectors is not None:
                af_vectors = af_vectors.to(self.device)

            sos_id = 1
            decoder_input = torch.cat([
                torch.full((targets.size(0), 1), sos_id,
                           dtype=torch.long, device=self.device),
                targets[:, :-1]
            ], dim=1)

            ctc_logits, att_logits = self.model(
                raw_audio, af_vectors, decoder_input
            )

            # Estimate CTC lengths from output
            ctc_lengths = torch.full(
                (ctc_logits.size(0),), ctc_logits.size(1),
                dtype=torch.long, device=self.device
            )

            loss = self.loss_fn(
                ctc_logits, ctc_lengths, att_logits, targets, target_lengths
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)

@torch.no_grad()
def evaluate(self, dataloader: DataLoader, idx_to_label: Dict[int, str]
             ) -> Dict[str, float]:
    """Run evaluation and compute MDD metrics."""
    self.model.eval()
    evaluator = MDDEvaluator()

    for batch in dataloader:
        if self.model_config in ("RS", "FP", "M1"):
            features = batch["features"].to(self.device)
            feat_lengths = batch["feature_lengths"].to(self.device)
            ctc_logits, enc_lengths, _ = self.model(features, feat_lengths)
        else:
            raw_audio = batch["raw_audio"].to(self.device)
            af_vectors = batch.get("af_vectors")
            if af_vectors is not None:
                af_vectors = af_vectors.to(self.device)
            ctc_logits, _ = self.model(raw_audio, af_vectors)

        # CTC greedy decode
        predictions = ctc_greedy_decode(ctc_logits, idx_to_label)

        # Update evaluator per sample
        batch_size = ctc_logits.size(0)
        for i in range(batch_size):
            canonical = batch["canonical"][i]
            actual = batch["actual"][i]
            predicted = predictions[i]

            if isinstance(canonical, str):
                canonical = canonical.split()
            if isinstance(actual, str):
                actual = actual.split()

            evaluator.update(canonical, predicted, actual)

    return evaluator.compute_metrics()
```

def ctc_greedy_decode(logits: torch.Tensor, idx_to_label: Dict[int, str],
blank_id: int = 0) -> List[List[str]]:
“””
CTC greedy decoding: take argmax at each timestep, remove blanks
and consecutive duplicates.

```
Args:
    logits: (Batch, Time, Classes)
    idx_to_label: mapping from class index to label string
Returns:
    decoded: list of label sequences per batch item
"""
predictions = logits.argmax(dim=-1)  # (B, T)
batch_decoded = []

for b in range(predictions.size(0)):
    decoded = []
    prev_token = blank_id
    for t in range(predictions.size(1)):
        token = predictions[b, t].item()
        if token != blank_id and token != prev_token:
            label = idx_to_label.get(token, "<unk>")
            if label not in ("<blank>", "<sos>", "<eos>"):
                decoded.append(label)
        prev_token = token
    batch_decoded.append(decoded)

return batch_decoded
```

# =============================================================================

# 9. AF Classifier Training Pipeline

# =============================================================================

class AFTrainer:
“”“Training pipeline for AF classifiers on LibriSpeech.”””

```
def __init__(self, model: AFClassifier, optimizer: torch.optim.Optimizer,
             device: torch.device):
    self.model = model
    self.optimizer = optimizer
    self.device = device

def train_epoch(self, dataloader: DataLoader) -> float:
    self.model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        self.optimizer.zero_grad()
        mfcc = batch["mfcc"].to(self.device)
        af_labels = {k: v.to(self.device) for k, v in batch["af_labels"].items()}

        loss = self.model.compute_loss(mfcc, af_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)
```

# =============================================================================

# 10. Model Factory

# =============================================================================

def build_model(config: str, output_framework: str = “PHN”,
d_model: int = 256) -> nn.Module:
“””
Build MDD model based on configuration (Section 2.2).

```
Five configurations × two frameworks = ten models:
  PHN-RS, PHN-FP, PHN-M1, PHN-FT, PHN-M2
  ART-RS, ART-FP, ART-M1, ART-FT, ART-M2

Args:
    config: "RS", "FP", "M1", "FT", "M2"
    output_framework: "PHN" or "ART"
    d_model: hidden dimension for Conformer/Transformer
"""
if output_framework == "PHN":
    num_classes = NUM_PHONEMES
else:
    num_classes = NUM_ART_LABELS

if config == "RS":
    # Raw speech baseline: 80-dim FBank
    return ConformerMDD(input_dim=80, num_classes=num_classes, d_model=d_model)
elif config == "FP":
    # FBank + Pitch: 83-dim
    return ConformerMDD(input_dim=83, num_classes=num_classes, d_model=d_model)
elif config == "M1":
    # FP + AF fusion: 83 + 26 = 109
    return ConformerMDD(input_dim=109, num_classes=num_classes, d_model=d_model)
elif config == "FT":
    # Fine-tuned XLSR (no AF)
    return XLSRMDD(num_classes=num_classes, af_dim=0, d_model=512)
elif config == "M2":
    # XLSR + AF fusion
    return XLSRMDD(num_classes=num_classes, af_dim=TOTAL_AF_DIM, d_model=512)
else:
    raise ValueError(f"Unknown config: {config}")
```

# =============================================================================

# 11. Utterance-Length Analysis (Section 3.3)

# =============================================================================

def analyze_by_utterance_length(results: List[Dict]) -> Dict[str, Dict[str, float]]:
“””
Partition results by utterance length and compute metrics per category.

```
Categories (Section 3.3):
  - Short:  < 21 labels
  - Medium: 21-40 labels
  - Long:   > 40 labels

Args:
    results: list of dicts with 'num_labels', 'canonical', 'predicted', 'actual'
Returns:
    metrics per length category
"""
categories = {"short": [], "medium": [], "long": []}

for r in results:
    n = r["num_labels"]
    if n < 21:
        categories["short"].append(r)
    elif n <= 40:
        categories["medium"].append(r)
    else:
        categories["long"].append(r)

metrics = {}
for cat_name, entries in categories.items():
    evaluator = MDDEvaluator()
    for e in entries:
        evaluator.update(e["canonical"], e["predicted"], e["actual"])
    metrics[cat_name] = evaluator.compute_metrics()
    metrics[cat_name]["count"] = len(entries)

return metrics
```

# =============================================================================

# 12. Speaker-Specific Analysis (Section 3.2)

# =============================================================================

def analyze_by_speaker(results: List[Dict]) -> Dict[str, Dict[str, float]]:
“””
Compute per-speaker evaluation metrics (Section 3.2).

```
Test speakers: PNV, THV, TLV (Vietnamese), RRBI, SVBI (Hindi), SKA (Arabic)
"""
speaker_results = {}
for r in results:
    spk = r["speaker"]
    if spk not in speaker_results:
        speaker_results[spk] = []
    speaker_results[spk].append(r)

metrics = {}
for spk, entries in speaker_results.items():
    evaluator = MDDEvaluator()
    for e in entries:
        evaluator.update(e["canonical"], e["predicted"], e["actual"])
    metrics[spk] = evaluator.compute_metrics()
    metrics[spk]["num_utterances"] = len(entries)

return metrics
```

# =============================================================================

# 13. Integration Demo

# =============================================================================

def main():
“””
Demonstrate the complete pipeline with mock data.
In production, replace mock data with real L2-ARCTIC/LibriSpeech loading.
“””
device = torch.device(“cuda” if torch.cuda.is_available() else “cpu”)
print(f”Device: {device}”)

```
# --- Step 1: Feature Extractor ---
print("\n[Step 1] Initializing feature extractor...")
extractor = AudioFeatureExtractor(sample_rate=16000)

# Test feature extraction
mock_waveform = torch.randn(16000)  # 1 second
mfcc = extractor.extract_mfcc_39(mock_waveform)
fp = extractor.extract_fbank_pitch_83(mock_waveform)
print(f"  MFCC shape: {mfcc.shape}  (expect ~[T, 39])")
print(f"  FBank+Pitch shape: {fp.shape}  (expect ~[T, 83])")

# --- Step 2: AF Classifier ---
print("\n[Step 2] Building AF classifier...")
af_model = AFClassifier(input_dim=39, hidden_dim=2048, num_hidden_layers=6).to(device)
num_params = sum(p.numel() for p in af_model.parameters())
print(f"  AF Classifier parameters: {num_params:,}")

# Test AF inference
mock_mfcc_batch = torch.randn(2, 50, 39).to(device)
with torch.no_grad():
    af_vectors = af_model(mock_mfcc_batch)
print(f"  AF output shape: {af_vectors.shape}  (expect [2, 50, 26])")

# --- Step 3: Build all 10 models ---
print("\n[Step 3] Building all 10 model configurations...")
models = {}
for framework in ("PHN", "ART"):
    for config in ("RS", "FP", "M1"):
        name = f"{framework}-{config}"
        model = build_model(config, framework, d_model=256).to(device)
        models[name] = model
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  {name}: {n_params:,} params")

# FT and M2 models (skip real XLSR loading for demo)
print("  (Skipping XLSR-based FT/M2 in demo mode)")

# --- Step 4: Test forward pass for Conformer models ---
print("\n[Step 4] Testing forward passes...")

# M1: FP(83) + AF(26) = 109
m1_input = torch.randn(2, 50, 109).to(device)
lengths = torch.tensor([50, 45]).to(device)
targets = torch.randint(0, NUM_PHONEMES, (2, 20)).to(device)

model_m1 = models["PHN-M1"]
ctc_out, enc_len, att_out = model_m1(m1_input, lengths, targets)
print(f"  PHN-M1 CTC output: {ctc_out.shape}  (expect [2, ~50, {NUM_PHONEMES}])")
print(f"  PHN-M1 ATT output: {att_out.shape}  (expect [2, 20, {NUM_PHONEMES}])")

# --- Step 5: Test joint loss ---
print("\n[Step 5] Testing joint CTC/Attention loss...")
loss_fn = JointCTCAttentionLoss(ctc_weight=0.3, blank_id=0)
target_lengths = torch.tensor([20, 18]).to(device)
loss = loss_fn(ctc_out, enc_len, att_out, targets, target_lengths)
print(f"  Joint loss: {loss.item():.4f}")

# --- Step 6: Test CTC greedy decoding ---
print("\n[Step 6] Testing CTC greedy decode...")
idx_to_phn = {i: p for p, i in PHONEME_TO_IDX.items()}
decoded = ctc_greedy_decode(ctc_out, idx_to_phn)
for i, seq in enumerate(decoded):
    print(f"  Sample {i}: {' '.join(seq[:10])}{'...' if len(seq) > 10 else ''}")

# --- Step 7: Test evaluation metrics ---
print("\n[Step 7] Testing evaluation metrics...")
evaluator = MDDEvaluator()

# Mock: canonical vs predicted vs actual (ground truth)
canonical = ["DH", "AH", "K", "AE", "T", "S", "IH", "T", "S"]
actual    = ["D",  "AH", "K", "AE", "T", "Z", "IY", "T", "S"]
predicted = ["D",  "AH", "K", "AE", "T", "S", "IY", "T", "S"]
# GT errors: DH->D (sub), S->Z (sub), IH->IY (sub)
# Pred: D (detected DH error, diagnosed as D=correct), S (missed Z error=FA),
#       IY (detected IH error, diagnosed as IY=correct)

evaluator.update(canonical, predicted, actual)
metrics = evaluator.compute_metrics()
print(f"  Metrics: {metrics}")

# --- Step 8: Test utterance-length analysis ---
print("\n[Step 8] Testing utterance-length analysis...")
mock_results = [
    {"num_labels": 15, "canonical": canonical, "predicted": predicted,
     "actual": actual, "speaker": "THV"},
    {"num_labels": 30, "canonical": canonical, "predicted": predicted,
     "actual": actual, "speaker": "TLV"},
    {"num_labels": 55, "canonical": canonical, "predicted": predicted,
     "actual": actual, "speaker": "RRBI"},
]
length_metrics = analyze_by_utterance_length(mock_results)
for cat, m in length_metrics.items():
    print(f"  {cat}: DA={m['DA']}%, FAR={m['FAR']}%, count={m['count']}")

# --- Step 9: Test speaker analysis ---
print("\n[Step 9] Testing speaker-specific analysis...")
speaker_metrics = analyze_by_speaker(mock_results)
for spk, m in speaker_metrics.items():
    print(f"  {spk}: DA={m['DA']}%, MCC={m['MCC']}")

# --- Summary ---
print("\n" + "=" * 60)
print("Pipeline verification complete.")
print("=" * 60)
print(f"  Output frameworks: PHN ({NUM_PHONEMES} classes), "
      f"ART ({NUM_ART_LABELS} classes)")
print(f"  AF dimensions: {TOTAL_AF_DIM}")
print(f"  Model configs: RS, FP, M1, FT, M2 × PHN/ART = 10 models")
print(f"  Metrics: DA, FAR, FRR, DER, MCC")
print(f"  Analysis: per-speaker, per-utterance-length, per-error-type DER")
print("=" * 60)
```

if **name** == “**main**”:
main()