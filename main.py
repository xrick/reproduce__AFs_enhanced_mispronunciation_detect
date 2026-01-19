from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset
from transformers import Wav2Vec2Model

# 設定亂數種子以確保可重現性
torch.manual_seed(42)

# =============================================================================
# 1. 資料準備與特徵提取 (Section 2.1 & 2.2)
# =============================================================================


class AudioFeatureExtractor:
    """
    負責提取論文中提到的聲學特徵：
    1. 39-dim MFCCs (用於 AF Classifier)
    2. 83-dim FBank + Pitch (用於 Conformer Model M1)
    """

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

        # 定義 39-dim MFCC (13 coeffs + delta + delta-delta)
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=13,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
        )

        # 定義 80-dim MelSpectrogram (搭配 Pitch 變成 83-dim)
        self.melspec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=80, n_fft=400, hop_length=160
        )

    def compute_deltas(self, spec):
        """計算 Delta 和 Delta-Delta"""
        delta = torchaudio.functional.compute_deltas(spec)
        delta2 = torchaudio.functional.compute_deltas(delta)
        # 拼接成 (Channels, Time)
        return torch.cat([spec, delta, delta2], dim=0)

    def extract_mfcc_39(self, waveform):
        """提取 39-dim MFCCs (Section 2.2)"""
        mfcc = self.mfcc_transform(waveform)
        mfcc_39 = self.compute_deltas(mfcc)
        return mfcc_39.transpose(0, 1)  # Return (Time, 39)

    def extract_fbank_pitch_83(self, waveform):
        """
        提取 83-dim 特徵 (80 FBank + 3 Pitch features) (Section 2.2)
        注意：這裡使用隨機值模擬 Pitch，實際請使用 torchaudio.functional.compute_kaldi_pitch
        """
        # 1. FBank (80 dims)
        fbank = self.melspec_transform(waveform).transpose(0, 1)  # (Time, 80)

        # 2. Pitch (3 dims: pitch, pov, delta pitch) - Mock implementation
        T = fbank.size(0)
        pitch_feats = torch.zeros(T, 3)

        # 融合
        combined = torch.cat([fbank, pitch_feats], dim=1)  # (Time, 83)
        return combined


# --- AF 類別定義 (Section 2.2) ---
# 用於定義分類器的輸出頭結構
AF_CONFIG = {
    "backness": 4,  # Front, Central, Back, Back2front
    "height": 4,  # High, Middle, Low, Low2high
    "roundness": 3,  # Rounded, Unrounded, Rounded2unrounded
    "manner": 5,  # Affricate, Fricative, Nasal, Stop, Approximant
    "place": 8,  # Alveolar, Bilabial, Dental, ...
    "voicing": 2,  # Voiced, Unvoiced
}
# 總特徵維度: 4+4+3+5+8+2 = 26
TOTAL_AF_DIM = sum(AF_CONFIG.values())

# =============================================================================
# 2. 構音特徵 (AF) 分類器 (Section 2.2)
# =============================================================================


class AFClassifier(nn.Module):
    """
    獨立 DNN-HMM 分類器。
    輸入: 39-dim MFCC
    輸出: 6 個類別的 Posteriors (Concatenated)
    結構: 6 hidden layers, 2048 units, Sigmoid activation
    """

    def __init__(self, input_dim=39, hidden_dim=2048, num_layers=6):
        super(AFClassifier, self).__init__()

        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Sigmoid())

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Sigmoid())

        self.encoder = nn.Sequential(*layers)

        # Output layers (6 separate classifiers)
        self.heads = nn.ModuleDict(
            {
                name: nn.Linear(hidden_dim, num_classes)
                for name, num_classes in AF_CONFIG.items()
            }
        )

    def forward(self, x):
        """
        x: (Batch, Time, 39)
        Returns: Composite AF vector (Batch, Time, 26)
        """
        feat = self.encoder(x)

        outputs = []
        # 依照固定順序提取並做 Softmax
        for name in AF_CONFIG.keys():
            logits = self.heads[name](feat)
            posteriors = F.softmax(logits, dim=-1)
            outputs.append(posteriors)

        # Fusion: Concatenate all posteriors
        af_vector = torch.cat(outputs, dim=-1)
        return af_vector


# =============================================================================
# 3. MDD 模型架構 (Section 2.2 - Models)
# =============================================================================

# --- Model A: Custom E2E Conformer (M1) ---


class ConformerMDD(nn.Module):
    """
    Conformer Encoder + Transformer Decoder + CTC
    支援 M1 配置: Fusion of FP (83) + AFs (26) -> Input dim 109
    """

    def __init__(self, input_dim, num_classes, d_model=256):
        super(ConformerMDD, self).__init__()

        self.encoder = torchaudio.models.Conformer(
            input_dim=input_dim,
            num_heads=4,
            ffn_dim=1024,
            num_layers=12,
            depthwise_conv_kernel_size=31,
        )

        # CTC Head
        self.ctc_head = nn.Linear(input_dim, num_classes)

        # Decoder (Optional/Hybrid part)
        # 論文提及 Transformer decoder，若為 Joint CTC/Attention，通常在此實作 Attention 機制
        decoder_layer = nn.TransformerDecoderLayer(d_model=input_dim, nhead=4)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        self.decoder_proj = nn.Linear(input_dim, num_classes)

    def forward(self, inputs, lengths):
        # inputs: (Batch, Time, Dim)
        encoder_out, lengths = self.encoder(inputs, lengths)

        # CTC Logits
        ctc_logits = self.ctc_head(encoder_out)

        return ctc_logits, lengths


# --- Model B: Fine-tuned XLSR (M2) ---


class XLSRWithAF(nn.Module):
    """
    Wav2Vec 2.0 (XLSR) + AF Fusion
    M2 配置: Wav2Vec2 Embeddings + AFs -> Transformer Decoder
    """

    def __init__(self, num_classes, af_dim=26, freeze_encoder=True):
        super(XLSRWithAF, self).__init__()

        # Load Pre-trained XLSR
        # 實際執行時需下載模型，這裡使用字串作為 placeholder
        try:
            self.wav2vec = Wav2Vec2Model.from_pretrained(
                "facebook/wav2vec2-large-xlsr-53"
            )
        except:
            print("Warning: Could not load HuggingFace model. Using mock layer.")
            self.wav2vec = nn.Linear(1, 1024)  # Mock for testing without download

        if freeze_encoder and isinstance(self.wav2vec, Wav2Vec2Model):
            self.wav2vec.feature_extractor._freeze_parameters()

        w2v_dim = 1024  # XLSR-53 hidden size

        # Fusion Layer: Project concatenated features (1024 + 26) to decoder dimension
        self.fusion_proj = nn.Linear(w2v_dim + af_dim, 512)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

        # Output Head
        self.output_head = nn.Linear(512, num_classes)

    def forward(self, raw_audio, af_vectors):
        """
        raw_audio: (Batch, Time_Raw)
        af_vectors: (Batch, Time_Frame, AF_Dim)
        """
        if isinstance(self.wav2vec, Wav2Vec2Model):
            outputs = self.wav2vec(raw_audio)
            embeddings = outputs.last_hidden_state  # (Batch, Frames, 1024)
        else:
            # Mock behavior
            embeddings = torch.randn(raw_audio.shape[0], af_vectors.shape[1], 1024)

        # 時間對齊處理 (簡易截斷)
        min_len = min(embeddings.size(1), af_vectors.size(1))
        embeddings = embeddings[:, :min_len, :]
        af_vectors = af_vectors[:, :min_len, :]

        # Fusion
        fused = torch.cat([embeddings, af_vectors], dim=-1)  # (Batch, Frames, 1050)

        # Projection
        projected = self.fusion_proj(fused)  # (Batch, Frames, 512)
        projected = projected.permute(1, 0, 2)  # (Seq, Batch, Dim) for Transformer

        # Decoder (Self-attention only for this snippet)
        decoded = self.decoder(projected, projected)

        logits = self.output_head(decoded.permute(1, 0, 2))
        return logits


# =============================================================================
# 4. 評估指標與邏輯 (Section 2.3 & 5)
# =============================================================================


class MDDEvaluator:
    def __init__(self):
        self.reset()

    def reset(self):
        # Counts for metrics
        self.stats = {"CA": 0, "FR": 0, "CR": 0, "FA": 0, "CD": 0, "DE": 0}

    def update(self, canonical_seq, predicted_seq, ground_truth_mispronunciations):
        """
        對比標準序列與預測序列。此處為簡化邏輯，實際需透過 Needleman-Wunsch
        或 Levenshtein Distance 進行 alignment 才能判斷每個音素的狀態。
        """
        # 假設已對齊，直接比較 (Mock Logic)
        for i in range(min(len(canonical_seq), len(predicted_seq))):
            can = canonical_seq[i]
            pred = predicted_seq[i]
            is_error_gt = (
                i in ground_truth_mispronunciations
            )  # Index i is an error in GT

            if not is_error_gt:  # Ground Truth: Correct
                if can == pred:
                    self.stats["CA"] += 1
                else:
                    self.stats["FR"] += 1
            else:  # Ground Truth: Mispronounced
                if can == pred:
                    self.stats["FA"] += 1
                else:
                    self.stats["CR"] += 1
                    # Diagnosis check
                    # 假設我們知道這是什麼錯誤，若 pred == actual_error_sound 則為 CD
                    self.stats["CD"] += 1  # Mock optimistic diagnosis

    def compute_metrics(self):
        s = self.stats
        eps = 1e-8

        # Section 2.3 Equations
        da = (s["CA"] + s["CR"]) / (
            sum(s.values()) - s["CD"] - s["DE"] + eps
        )  # Total samples
        # Note: Denominator logic simplifies to Total instances
        total = s["CA"] + s["FR"] + s["FA"] + s["CR"]

        da = (s["CA"] + s["CR"]) / (total + eps)
        far = s["FA"] / (s["CR"] + s["FA"] + eps)
        frr = s["FR"] / (s["CA"] + s["FR"] + eps)
        der = s["DE"] / (s["CD"] + s["DE"] + eps)

        # MCC
        num = (s["CA"] * s["CR"]) - (s["FA"] * s["FR"])
        den = (
            np.sqrt(
                (s["CA"] + s["FA"])
                * (s["CA"] + s["FR"])
                * (s["CR"] + s["FA"])
                * (s["CR"] + s["FR"])
            )
            + eps
        )

        return {"DA": da, "FAR": far, "FRR": frr, "DER": der, "MCC": num / den}


# =============================================================================
# 5. 整合測試 (Main Pipeline)
# =============================================================================


def main():
    print(">>> 初始化模型...")
    # 假設輸出類別 (Phonemes) 為 40
    af_model = AFClassifier()
    # M1: Input 83 (FP) + 26 (AF) = 109
    mdd_m1 = ConformerMDD(input_dim=109, num_classes=40)
    # M2: XLSR + AF
    mdd_m2 = XLSRWithAF(num_classes=40)

    print(">>> 模擬數據流 (Batch Size=2, Time=50)...")
    # Mock Data
    waveform = torch.randn(2, 16000)  # 1 sec audio
    extractor = AudioFeatureExtractor()

    # 1. Feature Extraction
    mfcc = torch.randn(2, 50, 39)  # Mock MFCC output
    fp = torch.randn(2, 50, 83)  # Mock FP output

    # 2. AF Classifier Inference
    with torch.no_grad():
        af_vectors = af_model(mfcc)  # (2, 50, 26)

    # 3. M1 Fusion & Forward
    m1_input = torch.cat([fp, af_vectors], dim=-1)  # (2, 50, 109)
    lengths = torch.tensor([50, 50])
    m1_out, _ = mdd_m1(m1_input, lengths)
    print(f"[M1] Output Shape: {m1_out.shape} (Expect: 2, 50, 40)")

    # 4. M2 Fusion & Forward
    # Note: raw audio length differs from frame length, handling inside model
    m2_out = mdd_m2(waveform, af_vectors)
    print(f"[M2] Output Shape: {m2_out.shape} (Expect: 2, 50, 40)")

    print(">>> 測試完成。此框架已準備好接上真實數據載入器。")


if __name__ == "__main__":
    main()
