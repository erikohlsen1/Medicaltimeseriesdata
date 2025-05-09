import torch
import os
import torch.nn as nn
from sklearn.metrics import silhouette_score
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import time
from scipy import stats
from sklearn.manifold import TSNE
import warnings
import seaborn as sns
from scipy.stats import ttest_ind, f_oneway
import matplotlib.cm as cm

warnings.filterwarnings('ignore')

# Global switches
TRAIN_TSTCC = True  # Switch for TS-TCC Training
TRAIN_RAUTO = True  # Switch for Recurrent Autoencoder Training
TRAIN_IMISS = True  # Switch for Informative Missingness Model
TRAIN_SIGNATURE = True  # Switch for Signature Transform Clustering
TRAIN_STATFEAT = True  # Switch for Statistical Feature Representation
TRAIN_RGAN = True  # Switch for Recurrent GAN
batch_size = 32  # Standard batch size

# Dictionary to store original feature names
FEATURE_NAMES = {}  # Will be populated from dataset


# Function to get actual feature names
def get_feature_names(data):
    """Extract and store the actual feature names from the dataset"""
    # Remove non-numeric or non-training relevant columns
    drop_cols = ['SepsisLabel', 'Patient_ID', 'ICULOS', 'Hour', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime']
    feature_cols = [col for col in data.columns if col not in drop_cols]

    # Store in global dictionary
    global FEATURE_NAMES
    FEATURE_NAMES = {i: name for i, name in enumerate(feature_cols)}
    return feature_cols


# ---------------------- Zusätzliche Datenvorverarbeitung für TS-TCC und Rekurrenten Autoencoder ----------------------

# 1. Spezielle Dataset-Klasse für TS-TCC mit Augmentierungen
class TSTCCDataset(Dataset):
    def __init__(self, sequences, masks):
        self.sequences = sequences
        self.masks = masks
        self.feature_dim = sequences.shape[2]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.sequences[idx])
        mask = torch.FloatTensor(self.masks[idx])

        # Erstelle augmentierte Versionen direkt im Dataset
        weak_aug_seq = self._weak_augmentation(sequence, mask)
        strong_aug_seq = self._strong_augmentation(sequence, mask)

        return sequence, weak_aug_seq, strong_aug_seq, mask

    def _weak_augmentation(self, x, mask):
        # Anwenden der Maske, damit nur gültige Werte augmentiert werden
        valid_indices = (mask == 1).nonzero(as_tuple=True)[0]

        if len(valid_indices) == 0:
            return x.clone()  # Keine gültigen Werte, gib Original zurück

        # Jittering nur auf gültige Werte
        x_aug = x.clone()
        noise = torch.randn_like(x_aug[valid_indices]) * 0.05
        x_aug[valid_indices] = x_aug[valid_indices] + noise

        # Skalierung nur auf gültige Werte
        scaling_factor = torch.randn(1) * 0.1 + 1.0
        x_aug[valid_indices] = x_aug[valid_indices] * scaling_factor

        return x_aug

    def _strong_augmentation(self, x, mask):
        # Anwenden der Maske, damit nur gültige Werte augmentiert werden
        valid_indices = (mask == 1).nonzero(as_tuple=True)[0]

        if len(valid_indices) == 0:
            return x.clone()  # Keine gültigen Werte, gib Original zurück

        x_aug = x.clone()

        # Stärkeres Jittering (verwende nur dieses als starke Augmentierung)
        noise = torch.randn_like(x_aug[valid_indices]) * 0.15
        x_aug[valid_indices] = x_aug[valid_indices] + noise

        # Hinzufügen zusätzlicher Augmentierungen ohne Permutation der Segmente

        # 1. Skalierung mit größerer Varianz
        scaling_factor = torch.randn(1) * 0.2 + 1.0
        x_aug[valid_indices] = x_aug[valid_indices] * scaling_factor

        # 2. Zeitliche Maskierung - setze zufällig einige Zeitschritte auf Null
        if len(valid_indices) > 5:
            num_to_mask = random.randint(1, len(valid_indices) // 5)
            mask_indices = random.sample(valid_indices.tolist(), num_to_mask)
            x_aug[mask_indices] = 0

        return x_aug


# 2. Spezieller DataLoader für den Rekurrenten Autoencoder, der Sequenzen nach Länge sortiert
def create_length_based_batches(sequences, masks, batch_size=32):
    # Berechne die tatsächliche Länge jeder Sequenz
    seq_lengths = masks.sum(axis=1)

    # Sortiere die Sequenzen nach Länge
    indices = np.argsort(seq_lengths)
    sorted_sequences = sequences[indices]
    sorted_masks = masks[indices]

    # Erstelle Batches mit ähnlichen Längen
    batches = []
    for i in range(0, len(sorted_sequences), batch_size):
        end_idx = min(i + batch_size, len(sorted_sequences))
        batch_sequences = sorted_sequences[i:end_idx]
        batch_masks = sorted_masks[i:end_idx]
        batches.append((batch_sequences, batch_masks))

    # Mische die Batches (nicht innerhalb der Batches)
    random.shuffle(batches)

    return batches


# Einfache Dataset-Klasse für den Autoencoder
class SimpleDataset(Dataset):
    def __init__(self, sequences, masks):
        self.sequences = sequences
        self.masks = masks

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.sequences[idx])
        mask = torch.FloatTensor(self.masks[idx])
        return sequence, mask


def pad_sequences(sequences, max_len=None):
    """Fügt Padding zu Sequenzen hinzu, um gleiche Länge zu erreichen."""
    # Bestimme die maximale Sequenzlänge (falls nicht angegeben)
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)

    # Hole die Feature-Dimension aus der ersten Sequenz
    feature_dim = sequences[0].shape[1]

    # Erstelle gepaddte Sequenzen und Masken
    padded_sequences = []
    attention_masks = []

    for seq in sequences:
        # Wenn die Sequenz kürzer als max_len ist, füge Padding hinzu
        seq_len = len(seq)
        if seq_len < max_len:
            # Erstelle Padding
            padding = np.zeros((max_len - seq_len, feature_dim))
            # Füge das Padding zur Sequenz hinzu
            padded_seq = np.vstack((seq, padding))
            # Erstelle eine Aufmerksamkeitsmaske: 1 für echte Daten, 0 für Padding
            attention_mask = np.ones(max_len)
            attention_mask[seq_len:] = 0
        else:
            # Wenn die Sequenz gleich oder länger als max_len ist, schneide sie ab
            padded_seq = seq[:max_len]
            attention_mask = np.ones(max_len)

        padded_sequences.append(padded_seq)
        attention_masks.append(attention_mask)

    return np.array(padded_sequences), np.array(attention_masks)


# Hilfsfunktion für das unüberwachte Training mit Masken
def train_with_masks_batch(model, sequences, masks, optimizer, loss_fn, device):
    """
    Gemeinsame Trainingsfunktion für maskierte Batches bei unüberwachtem Lernen
    """
    sequences = torch.FloatTensor(sequences).to(device)
    masks = torch.FloatTensor(masks).to(device)

    optimizer.zero_grad()

    # Forward pass (spezifisch für jedes Modell implementiert)
    outputs = model(sequences, masks)

    # Loss-Berechnung nur auf gültige (nicht maskierte) Werte
    loss = loss_fn(outputs, sequences, masks)

    loss.backward()
    optimizer.step()

    return loss.item()


# MSE Verlustfunktion mit Masken für den Rekurrenten Autoencoder
def masked_mse_loss(outputs, targets, masks):
    """
    MSE-Verlust, der nur gültige (nicht maskierte) Zeitpunkte berücksichtigt
    """
    if isinstance(outputs, tuple):  # Falls der Autoencoder auch latent zurückgibt
        outputs = outputs[0]

    # Elementweiser MSE
    mse = (outputs - targets) ** 2

    # Anwenden der Maske (erweitern auf Feature-Dimension)
    mask_expanded = masks.unsqueeze(-1)
    masked_mse = mse * mask_expanded

    # Mittlerer Verlust über alle gültigen Einträge
    num_valid = torch.sum(mask_expanded) * targets.size(-1)
    if num_valid > 0:
        return torch.sum(masked_mse) / num_valid
    else:
        return torch.tensor(0.0, device=outputs.device)


# Extraktion von Features aus trainierten unüberwachten Modellen (für spätere Analyse)
def extract_unsupervised_features(model, dataloader, device, model_type='tstcc'):
    """
    Extrahiert die gelernten Repräsentationen aus den unüberwachten Modellen
    """
    model.eval()
    features_list = []

    with torch.no_grad():
        if model_type == 'tstcc':
            for batch_data in dataloader:
                if len(batch_data) == 4:  # TSTCCDataset
                    sequences, _, _, masks = batch_data
                else:  # Standard Dataset
                    sequences, masks = batch_data

                sequences = sequences.to(device)

                # Extrahiere Features (angenommen, model() gibt die Repräsentationen zurück)
                features = model(sequences)
                features_list.append(features.cpu().numpy())

        elif model_type == 'rauto':
            for sequences, masks in dataloader:
                sequences = sequences.to(device)
                masks = masks.to(device)

                # Extrahiere latente Repräsentationen
                _, latent = model(sequences, masks)
                features_list.append(latent.cpu().numpy())

    return np.vstack(features_list) if features_list else np.array([])


# ---------------------- Modell 1: Temporal Contrastive Learning (TS-TCC) ----------------------
class TemporalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, feature_dim=64):
        super(TemporalEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU()
        )

    def forward(self, x):
        # Input: [batch_size, seq_len, input_dim]
        # Convert to [batch_size, input_dim, seq_len] for Conv1d
        x = x.transpose(1, 2)
        features = self.encoder(x)
        return features  # [batch_size, feature_dim, seq_len]


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim=64):
        super(ProjectionHead, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.BatchNorm1d(input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, output_dim)
        )

    def forward(self, x):
        return self.projection(x)


class TS_TCC(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, feature_dim=64, temperature=0.1):
        super(TS_TCC, self).__init__()
        self.encoder = TemporalEncoder(input_dim, hidden_dim, feature_dim)
        self.proj_head = ProjectionHead(feature_dim, feature_dim)
        self.temperature = temperature

    def forward(self, x):
        # Standard forward pass for inference
        features = self.encoder(x)
        # Convert [batch_size, feature_dim, seq_len] to [batch_size, feature_dim]
        # by mean pooling across sequence length
        features = torch.mean(features, dim=2)
        return features

    def get_representations(self, x, proj=False):
        features = self.encoder(x)
        features = torch.mean(features, dim=2)
        if proj:
            features = self.proj_head(features)
        return features

    def weak_augmentation(self, x):
        # Jittering
        noise = torch.randn_like(x) * 0.05
        x_jitter = x + noise

        # Scaling
        scale_factor = torch.randn(x.size(0), 1, x.size(2), device=x.device) * 0.1 + 1.0
        x_scaled = x_jitter * scale_factor.unsqueeze(1)

        return x_scaled

    def strong_augmentation(self, x):
        batch_size, seq_len, feat_dim = x.shape

        # Permutation - randomly permute segments of the sequence
        num_segments = random.randint(2, 5)
        segment_len = seq_len // num_segments

        x_perm = x.clone()
        for b in range(batch_size):
            segments = list(range(num_segments))
            random.shuffle(segments)

            for i, seg_idx in enumerate(segments):
                orig_start = seg_idx * segment_len
                orig_end = (seg_idx + 1) * segment_len if seg_idx < num_segments - 1 else seq_len
                new_start = i * segment_len
                new_end = (i + 1) * segment_len if i < num_segments - 1 else seq_len

                x_perm[b, new_start:new_end] = x[b, orig_start:orig_end]

        # Adding stronger jittering
        noise = torch.randn_like(x_perm) * 0.1
        x_perm = x_perm + noise

        return x_perm

    def temporal_contrastive_loss(self, weak_feats, strong_feats):
        # Temporal Contrastive Loss
        batch_size = weak_feats.size(0)

        # L2 normalize the features
        weak_feats = nn.functional.normalize(weak_feats, dim=1)
        strong_feats = nn.functional.normalize(strong_feats, dim=1)

        # Positive similarity (between weak and strong views of same sample)
        pos_sim = torch.bmm(weak_feats.unsqueeze(1), strong_feats.unsqueeze(2)).squeeze(-1)

        # Compute similarity matrix between all samples
        similarity_matrix = torch.matmul(weak_feats, strong_feats.transpose(0, 1)) / self.temperature

        # Create mask for positive pairs
        mask = torch.eye(batch_size, device=weak_feats.device)

        # Compute contrastive loss
        exp_sim = torch.exp(similarity_matrix)
        log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        loss = -(log_prob * mask).sum() / batch_size

        return loss


def train_ts_tcc(model, train_loader, device, num_epochs=50, learning_rate=0.001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    loss_history = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        for sequences, weak_augs, strong_augs, masks in train_loader:
            # Verschiebe Daten zum Gerät
            sequences = sequences.to(device)
            weak_augs = weak_augs.to(device)
            strong_augs = strong_augs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            # Erhalte Projektionen für beide Augmentierungen
            weak_feats = model.get_representations(weak_augs, proj=True)
            strong_feats = model.get_representations(strong_augs, proj=True)

            # Berechne Contrastive Loss
            loss = model.temporal_contrastive_loss(weak_feats, strong_feats)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_epoch_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}")

    return model, loss_history


# Verwendung des TS-TCC Modells
def use_ts_tcc_model(input_dim, train_loader, device, num_epochs=50):
    print("\n==== Training des TS-TCC Modells (Unsupervised) ====")

    # Modell initialisieren
    ts_tcc_model = TS_TCC(input_dim).to(device)

    # Unüberwachtes Training
    ts_tcc_model, loss_history = train_ts_tcc(
        ts_tcc_model, train_loader, device, num_epochs=num_epochs)

    print(f"Finaler Trainingsverlust: {loss_history[-1]:.4f}")

    return ts_tcc_model, loss_history


# ---------------------- Modell 2: Rekurrenter Autoencoder für medizinische Zeitreihen ----------------------
class RecurrentAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, latent_dim=64, n_layers=2, dropout=0.2):
        super(RecurrentAutoencoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.n_layers = n_layers

        # Encoder (LSTM oder GRU)
        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )

        # Attention Mechanismus
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softmax(dim=1)
        )

        # Projektion zum latenten Raum
        self.to_latent = nn.Linear(hidden_dim, latent_dim)

        # Projektion vom latenten Raum zum Decoder-Eingang
        self.from_latent = nn.Linear(latent_dim, hidden_dim)

        # Decoder (LSTM oder GRU)
        self.decoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )

        # Output-Projektion
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def encode(self, x, mask=None):
        # x: [batch_size, seq_len, input_dim]
        # Encoder-Ausgabe: all hidden states
        if mask is not None:
            # Packzustand für variable Sequenzlängen
            lengths = mask.sum(dim=1).long()
            packed_x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_output, hidden = self.encoder(packed_x)
            encoder_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        else:
            encoder_output, hidden = self.encoder(x)

        # Attention Mechanismus
        attention_weights = self.attention(encoder_output)
        context = torch.sum(attention_weights * encoder_output, dim=1)

        # Latente Repräsentation
        latent = self.to_latent(context)

        return latent, hidden

    def decode(self, latent, seq_len, hidden=None):
        # Transformiere latent zu Decoder-Input
        decoder_input = self.from_latent(latent)

        # Wiederhole den Decoder-Input für jede Zeitposition
        decoder_input = decoder_input.unsqueeze(1).repeat(1, seq_len, 1)

        # Decoder-Schritt
        decoder_output, _ = self.decoder(decoder_input, hidden)

        # Projiziere zu Originalformat
        reconstructed = self.output_layer(decoder_output)

        return reconstructed

    def forward(self, x, mask=None):
        seq_len = x.size(1)

        # Encoding
        latent, hidden = self.encode(x, mask)

        # Decoding
        reconstructed = self.decode(latent, seq_len, hidden)

        if mask is not None:
            # Anwenden der Maske auf die Rekonstruktion (optional)
            reconstructed = reconstructed * mask.unsqueeze(-1)

        return reconstructed, latent

    def get_latent(self, x, mask=None):
        with torch.no_grad():
            latent, _ = self.encode(x, mask)
        return latent


def train_recurrent_autoencoder(model, train_loader, device, num_epochs=50, learning_rate=0.001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    reconstruction_criterion = nn.MSELoss(reduction='none')

    model.train()
    loss_history = []

    for epoch in range(num_epochs):
        epoch_loss = 0

        for sequences, masks in train_loader:  # Angepasst für unüberwachtes Lernen ohne Labels
            sequences = sequences.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            # Forward pass
            reconstructed, _ = model(sequences, masks)

            # Berechne rekonstruktionsverlust nur für gültige (nicht maskierte) Zeitpunkte
            reconstruction_loss = reconstruction_criterion(reconstructed, sequences)

            # Anwenden der Maske
            masked_loss = reconstruction_loss * masks.unsqueeze(-1)

            # Mittlerer Verlust über alle gültigen Einträge
            n_valid_entries = masks.sum() * sequences.size(2)
            if n_valid_entries > 0:  # Vermeide Division durch Null
                loss = masked_loss.sum() / n_valid_entries
            else:
                loss = torch.tensor(0.0, device=device)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_epoch_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}")

    return model, loss_history


# Angepasste Funktion für längenbasierte Batches
def train_recurrent_autoencoder_with_length_batches(model, batches, device, num_epochs=50, learning_rate=0.001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    reconstruction_criterion = nn.MSELoss(reduction='none')

    model.train()
    loss_history = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        batch_count = 0

        # Shuffeln der Batches vor jeder Epoche
        random.shuffle(batches)

        for batch_sequences, batch_masks in batches:
            sequences = torch.FloatTensor(batch_sequences).to(device)
            masks = torch.FloatTensor(batch_masks).to(device)

            optimizer.zero_grad()

            # Forward pass
            reconstructed, _ = model(sequences, masks)

            # Berechne rekonstruktionsverlust nur für gültige (nicht maskierte) Zeitpunkte
            reconstruction_loss = reconstruction_criterion(reconstructed, sequences)

            # Anwenden der Maske
            masked_loss = reconstruction_loss * masks.unsqueeze(-1)

            # Mittlerer Verlust über alle gültigen Einträge
            n_valid_entries = masks.sum() * sequences.size(2)
            if n_valid_entries > 0:  # Vermeide Division durch Null
                loss = masked_loss.sum() / n_valid_entries
            else:
                loss = torch.tensor(0.0, device=device)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        avg_epoch_loss = epoch_loss / batch_count
        loss_history.append(avg_epoch_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}")

    return model, loss_history


def use_recurrent_autoencoder(input_dim, batches, device, num_epochs=50):
    print("\n==== Training des Rekurrenten Autoencoders (Unsupervised) ====")

    # Modell initialisieren
    autoencoder = RecurrentAutoencoder(input_dim).to(device)

    # Unüberwachtes Training mit längenbasierten Batches
    autoencoder, loss_history = train_recurrent_autoencoder_with_length_batches(
        autoencoder, batches, device, num_epochs=num_epochs)

    print(f"Finaler Rekonstruktionsverlust: {loss_history[-1]:.4f}")

    return autoencoder, loss_history


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from sklearn.cluster import KMeans


class InformativeMissingnessAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, latent_dim=64, dropout=0.2):
        super(InformativeMissingnessAutoencoder, self).__init__()

        # Bei diesem Modell berücksichtigen wir sowohl die Werte als auch die Maskierungsmuster
        # Daher ist der tatsächliche Eingangsdimension input_dim * 2 (Werte + Masken)
        self.input_dim = input_dim
        self.combined_dim = input_dim * 2

        # Encoder für ursprüngliche Werte und Masken
        self.encoder = nn.Sequential(
            nn.Linear(self.combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, latent_dim)
        )

        # Decoder für Werte
        self.decoder_values = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )

        # Decoder für Missingness-Muster (als Binärklassifikation)
        self.decoder_missingness = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # Wahrscheinlichkeit für das Vorhandensein eines Wertes
        )

        # Sliding Window Attention
        self.window_size = 3  # Fenster vor und nach aktuellem Zeitpunkt
        self.attention = nn.Sequential(
            nn.Linear(latent_dim * (2 * self.window_size + 1), hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2 * self.window_size + 1),
            nn.Softmax(dim=1)
        )

    def encode(self, x, mask):
        # Kombiniere Eingabewerte und Masken
        # x hat Form [batch_size, seq_len, features]
        # mask hat Form [batch_size, seq_len] oder [batch_size, seq_len, features]

        batch_size, seq_len, feature_dim = x.shape

        # Überprüfe die Form der Maske und erweitere sie bei Bedarf
        if len(mask.shape) == 2:
            # Wenn die Maske nur 2D ist (batch_size, seq_len), erweitere sie auf 3D
            # um sie mit features zu multiplizieren
            mask = mask.unsqueeze(-1).expand_as(x)
        elif mask.shape[2] == 1:
            # Wenn die Maske 3D ist, aber nur eine Featuredimension hat, erweitere sie
            mask = mask.expand_as(x)

        # Bette fehlende Werte standardmäßig mit 0 ein
        x_masked = x * mask

        # Kombiniere Werte und Masken
        # Sicherstellen, dass die Masken die gleiche Größe wie x haben
        combined = torch.cat([x_masked, mask], dim=2)

        # Anwenden des Encoders zeitschrittweise
        encoded_seq = []
        for t in range(seq_len):
            # Extrahiere die aktuelle Zeitscheibe
            current_slice = combined[:, t, :]
            encoded_t = self.encoder(current_slice)
            encoded_seq.append(encoded_t)

        # Stapel aller Zeitpunkte
        encoded_seq = torch.stack(encoded_seq, dim=1)

        return encoded_seq

    def apply_temporal_attention(self, encoded_seq):
        batch_size, seq_len, latent_dim = encoded_seq.shape

        # Anwenden der temporalen Aufmerksamkeit
        attended_seq = []

        for t in range(seq_len):
            # Fenster um den aktuellen Zeitpunkt definieren
            window_start = max(0, t - self.window_size)
            window_end = min(seq_len, t + self.window_size + 1)

            # Extrahiere Zeitfenster-Encodings
            if window_start == 0:
                # Auffüllen am Anfang
                padding_size = self.window_size - t
                padding = torch.zeros(batch_size, padding_size, latent_dim, device=encoded_seq.device)
                window_encodings = torch.cat([padding, encoded_seq[:, :window_end]], dim=1)
            elif window_end == seq_len:
                # Auffüllen am Ende
                padding_size = self.window_size - (seq_len - t - 1)
                padding = torch.zeros(batch_size, padding_size, latent_dim, device=encoded_seq.device)
                window_encodings = torch.cat([encoded_seq[:, window_start:], padding], dim=1)
            else:
                # Kein Auffüllen nötig
                window_encodings = encoded_seq[:, window_start:window_end]

            # Reshape für Attention-Berechnung
            window_flat = window_encodings.reshape(batch_size, -1)

            # Attention-Gewichte berechnen
            if window_flat.size(1) != (2 * self.window_size + 1) * latent_dim:
                # Temporäre Lösung: Wenn das Fenster nicht die erwartete Größe hat,
                # verwenden wir einfach den aktuellen Encoding ohne Attention
                attended_t = encoded_seq[:, t]
            else:
                attn_weights = self.attention(window_flat)

                # Reshape Attention-Gewichte und window_encodings für Matrix-Multiplikation
                attn_weights = attn_weights.unsqueeze(2)  # [batch, window, 1]

                # Wende Attention an
                attended_t = torch.bmm(
                    window_encodings.transpose(1, 2),  # [batch, latent, window]
                    attn_weights  # [batch, window, 1]
                ).squeeze(2)  # [batch, latent]

            attended_seq.append(attended_t)

        # Stapel aller Zeitpunkte
        attended_seq = torch.stack(attended_seq, dim=1)  # [batch, seq, latent]

        return attended_seq

    def decode(self, z):
        # z hat Form [batch_size, seq_len, latent_dim]
        batch_size, seq_len, _ = z.shape

        # Decodieren der Werte zeitschrittweise
        decoded_values = []
        decoded_missingness = []

        for t in range(seq_len):
            # Extrahiere den aktuellen Zeitschritt
            z_t = z[:, t]

            # Decodiere Werte und Missingness
            values_t = self.decoder_values(z_t)
            missingness_t = self.decoder_missingness(z_t)

            decoded_values.append(values_t)
            decoded_missingness.append(missingness_t)

        # Stapel aller Zeitpunkte
        decoded_values = torch.stack(decoded_values, dim=1)
        decoded_missingness = torch.stack(decoded_missingness, dim=1)

        return decoded_values, decoded_missingness

    def forward(self, x, mask):
        # Encoding
        z = self.encode(x, mask)

        # Temporale Attention
        z_attended = self.apply_temporal_attention(z)

        # Decoding
        x_hat, mask_hat = self.decode(z_attended)

        return x_hat, mask_hat, z_attended

    def get_latent(self, x, mask):
        with torch.no_grad():
            z = self.encode(x, mask)
            z_attended = self.apply_temporal_attention(z)
        return z_attended


def train_informative_missingness_model(model, batches, device, num_epochs=50, learning_rate=0.001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Verlustfunktionen
    # Für Werte: MSE auf vorhandene Werte
    value_criterion = nn.MSELoss(reduction='none')

    # Für Missingness: Binary Cross-Entropy
    missingness_criterion = nn.BCELoss(reduction='none')

    model.train()
    loss_history = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        value_loss_total = 0
        missingness_loss_total = 0
        batch_count = 0

        # Batch-Shuffle für jede Epoche
        random.shuffle(batches)

        for batch_sequences, batch_masks in batches:
            sequences = torch.FloatTensor(batch_sequences).to(device)
            masks = torch.FloatTensor(batch_masks).to(device)

            # Überprüfe und passe die Maskenform an
            if len(masks.shape) == 2:
                # Wenn die Maske 2D ist, erweitere sie auf 3D für Features
                masks_expanded = masks.unsqueeze(-1).expand_as(sequences)
            else:
                masks_expanded = masks

            optimizer.zero_grad()

            # Forward-Pass
            x_hat, mask_hat, _ = model(sequences, masks)

            # Verlust für Werte (nur auf vorhandene Werte)
            value_loss = value_criterion(x_hat, sequences)
            masked_value_loss = (value_loss * masks_expanded).sum() / (masks_expanded.sum() + 1e-8)

            # Verlust für Missingness-Muster
            # Stelle sicher, dass mask_hat und masks die gleiche Form haben
            if len(masks.shape) == 2:
                # Reduziere mask_hat auf 2D wenn nötig
                mask_hat_reshaped = mask_hat.mean(dim=-1)
                missingness_loss = missingness_criterion(mask_hat_reshaped, masks)
            else:
                missingness_loss = missingness_criterion(mask_hat, masks_expanded)

            missingness_loss = missingness_loss.mean()

            # Gesamtverlust
            loss = masked_value_loss + missingness_loss

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            value_loss_total += masked_value_loss.item()
            missingness_loss_total += missingness_loss.item()
            batch_count += 1

        avg_epoch_loss = epoch_loss / batch_count
        avg_value_loss = value_loss_total / batch_count
        avg_missingness_loss = missingness_loss_total / batch_count

        loss_history.append((avg_epoch_loss, avg_value_loss, avg_missingness_loss))

        print(f"Epoch {epoch + 1}/{num_epochs}, Total Loss: {avg_epoch_loss:.4f}, "
              f"Value Loss: {avg_value_loss:.4f}, Missingness Loss: {avg_missingness_loss:.4f}")

    return model, loss_history


def use_informative_missingness_model(input_dim, batches, device, num_epochs=50):
    print("\n==== Training des Informative Missingness Pattern Learning Models ====")

    # Modell initialisieren
    model = InformativeMissingnessAutoencoder(input_dim).to(device)

    # Unüberwachtes Training mit längenbasierten Batches
    model, loss_history = train_informative_missingness_model(
        model, batches, device, num_epochs=num_epochs
    )

    print(f"Finaler Gesamtverlust: {loss_history[-1][0]:.4f}")
    print(f"Finaler Werteverlust: {loss_history[-1][1]:.4f}")
    print(f"Finaler Missingness-Verlust: {loss_history[-1][2]:.4f}")

    return model, loss_history


def cluster_latent_representations(model, batches, device, n_clusters=5):
    """
    Wendet Clustering auf die latenten Repräsentationen des Modells an
    """
    all_latents = []

    model.eval()
    with torch.no_grad():
        for batch_sequences, batch_masks in batches:
            sequences = torch.FloatTensor(batch_sequences).to(device)
            masks = torch.FloatTensor(batch_masks).to(device)

            # Extrahiere latente Repräsentationen
            latent = model.get_latent(sequences, masks)

            # Berechne den Mittelwert über alle Zeitschritte, um eine Sequenz in einen Vektor zu transformieren
            latent_mean = latent.mean(dim=1)

            all_latents.append(latent_mean.cpu().numpy())

    # Verkette alle latenten Repräsentationen
    all_latents = np.vstack(all_latents)

    # Wende K-Means Clustering an
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(all_latents)

    return clusters, kmeans.cluster_centers_, all_latents


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class SignatureTransform:
    def __init__(self, truncation_level=3):
        self.truncation_level = truncation_level

    def lead_lag_transform(self, time_series):
        """
        Transformiert eine Zeitreihe in ihre Lead-Lag-Repräsentation.
        Dies verdoppelt jeden Punkt: einen für den aktuellen Wert (lag) und einen für den nächsten Wert (lead).

        Args:
            time_series: Array der Form [n_timesteps, n_features]

        Returns:
            Array der Form [2*n_timesteps-1, n_features*2]
        """
        n_timesteps, n_features = time_series.shape

        # Erstelle führende und verzögerte Versionen
        lead = time_series[1:, :]  # entferne ersten Zeitschritt
        lag = time_series[:-1, :]  # entferne letzten Zeitschritt

        # Interpoliere, um alternierende Lead-Lag-Darstellung zu erhalten
        transformed = np.zeros((2 * n_timesteps - 1, n_features * 2))

        # Fülle alternierende Positionen
        for i in range(n_timesteps - 1):
            # Lag-Einträge an ungeraden Positionen
            transformed[2 * i, :n_features] = lag[i]
            # Lead-Einträge an geraden Positionen
            transformed[2 * i + 1, n_features:] = lead[i]

        # Letzter Lag-Eintrag (falls vorhanden)
        if n_timesteps > 0:
            transformed[-1, :n_features] = time_series[-1]

        return transformed

    def compute_cumsum(self, time_series):
        """
        Berechnet die kumulierte Summe einer Zeitreihe entlang der Zeitdimension.
        Dies ist eine einfache Approximation der Signatur für erste-Ordnung-Terme.

        Args:
            time_series: Array der Form [n_timesteps, n_features]

        Returns:
            Array der Form [n_timesteps, n_features] mit kumulierten Summen
        """
        return np.cumsum(time_series, axis=0)

    def compute_second_order_terms(self, time_series):
        """
        Berechnet die Interaktionsterme zweiter Ordnung zwischen Features.
        Dies ist eine Approximation der Signatur für zweite-Ordnung-Terme.

        Args:
            time_series: Array der Form [n_timesteps, n_features]

        Returns:
            Array der Form [n_timesteps, n_features*(n_features+1)/2] mit Interaktionstermen
        """
        n_timesteps, n_features = time_series.shape
        n_interactions = n_features * (n_features + 1) // 2

        # Kumulierte Summen für Einzelterme
        cumsum = self.compute_cumsum(time_series)

        # Initialisiere Array für Interaktionsterme
        interactions = np.zeros((n_timesteps, n_interactions))

        # Fülle Interaktionsterme
        idx = 0
        for i in range(n_features):
            for j in range(i, n_features):
                # Berechne Area zweier kumulierter Summen als Näherung für Doppelintegral
                area_product = np.zeros(n_timesteps)
                for t in range(1, n_timesteps):
                    # Approximiere das Itô-Integral: ∫∫ f_i(s) df_j(t)
                    area_product[t] = area_product[t - 1] + cumsum[t - 1, i] * (
                            time_series[t, j] - time_series[t - 1, j])

                interactions[:, idx] = area_product
                idx += 1

        return interactions

    def compute_path_signature(self, time_series, mask=None):
        """
        Berechnet die Pfadsignatur einer Zeitreihe bis zum angegebenen Trunkierungslevel.

        Args:
            time_series: Array der Form [n_timesteps, n_features]
            mask: Optionale Maske der Form [n_timesteps] oder [n_timesteps, n_features], 1 für gültige Werte

        Returns:
            Dictionary mit Signaturtermen verschiedener Ordnung
        """
        n_timesteps, n_features = time_series.shape

        # Behandle fehlende Werte
        if mask is not None:
            # Prüfe die Dimension der Maske und erweitere sie bei Bedarf
            if len(mask.shape) == 1:
                # Erweitere die Maske auf [n_timesteps, n_features]
                mask = np.tile(mask[:, np.newaxis], (1, n_features))

            # Ersetze fehlende Werte durch lineare Interpolation
            for i in range(n_features):
                valid_indices = np.where(mask[:, i])[0]
                if len(valid_indices) > 0:
                    # Falls es mindestens einen gültigen Wert gibt
                    for j in range(n_timesteps):
                        if not mask[j, i]:
                            # Finde nächsten gültigen Wert vor und nach
                            prev_valid = valid_indices[valid_indices < j]
                            next_valid = valid_indices[valid_indices > j]

                            if len(prev_valid) > 0 and len(next_valid) > 0:
                                # Interpoliere zwischen vorherigem und nächstem gültigen Wert
                                prev_idx = prev_valid[-1]
                                next_idx = next_valid[0]
                                weight = (j - prev_idx) / (next_idx - prev_idx)
                                time_series[j, i] = (1 - weight) * time_series[prev_idx, i] + weight * time_series[
                                    next_idx, i]
                            elif len(prev_valid) > 0:
                                # Verwende letzten gültigen Wert
                                time_series[j, i] = time_series[prev_valid[-1], i]
                            elif len(next_valid) > 0:
                                # Verwende nächsten gültigen Wert
                                time_series[j, i] = time_series[next_valid[0], i]
                            else:
                                # Keine gültigen Werte für dieses Feature
                                time_series[j, i] = 0

        # Standardisiere die Zeitreihe pro Feature
        scaler = StandardScaler()
        time_series = scaler.fit_transform(time_series)

        # Berechne Lead-Lag-Transformation
        lead_lag_series = self.lead_lag_transform(time_series)

        # Berechne Signaturen verschiedener Ordnungen
        signatures = {}

        # Level 0: Mittelwert (zerostufe)
        signatures['level_0'] = np.mean(time_series, axis=0)

        # Level 1: Kumulierte Summe (Approximation der Signatur erster Ordnung)
        cumsum = self.compute_cumsum(lead_lag_series)
        signatures['level_1'] = cumsum[-1]  # Endwert der kumulierten Summe

        # Falls gefordert, berechne höhere Ordnungen
        if self.truncation_level >= 2:
            signatures['level_2'] = self.compute_second_order_terms(lead_lag_series)[-1]

        # Für Level 3 erstellen wir eine Näherung durch Merkmale wie Krümmung, Geschwindigkeit, etc.
        if self.truncation_level >= 3:
            # Berechne Differenzen (Geschwindigkeit)
            velocity = np.diff(time_series, axis=0, prepend=time_series[0:1])

            # Berechne zweite Differenzen (Beschleunigung)
            acceleration = np.diff(velocity, axis=0, prepend=velocity[0:1])

            # Berechne Krümmung: ||a x v|| / ||v||^3
            velocity_norm = np.linalg.norm(velocity, axis=1, keepdims=True)
            velocity_norm = np.maximum(velocity_norm, 1e-10)  # Vermeide Division durch Null

            # Verwende einige repräsentative Statistiken
            signatures['level_3'] = np.concatenate([
                np.mean(velocity, axis=0),
                np.mean(acceleration, axis=0),
                np.std(velocity, axis=0),
                np.std(acceleration, axis=0)
            ])

        # Alle Signaturen zu einem Feature-Vektor verketten
        all_features = []
        for level in range(self.truncation_level + 1):
            key = f'level_{level}'
            if key in signatures:
                all_features.append(signatures[key])

        feature_vector = np.concatenate([feat.flatten() for feat in all_features])

        return feature_vector


class SignatureTransformClustering:
    def __init__(self, truncation_level=3, n_clusters=5, pca_components=30, random_state=42):
        self.signature_transform = SignatureTransform(truncation_level=truncation_level)
        self.n_clusters = n_clusters
        self.pca_components = pca_components
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=pca_components)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)

    def fit(self, sequences, masks=None):
        """
        Berechnet die Signatur-Transformation für jede Sequenz und wendet Clustering an
        """
        # Berechne Signaturen für alle Sequenzen
        signatures = []

        for i, seq in enumerate(sequences):
            mask = masks[i] if masks is not None else None
            signature = self.signature_transform.compute_path_signature(seq, mask)
            signatures.append(signature)

        # Konvertiere in NumPy-Array
        signatures = np.array(signatures)

        # Standardisiere die Features
        signatures_scaled = self.scaler.fit_transform(signatures)

        # Dimensionsreduktion mit PCA, falls nötig
        n_samples, n_features = signatures_scaled.shape
        if n_features > self.pca_components and n_samples > self.pca_components:
            signatures_reduced = self.pca.fit_transform(signatures_scaled)
            print(f"PCA angewendet: Reduzierung von {n_features} auf {self.pca_components} Dimensionen")
            print(f"Erklärte Varianz: {np.sum(self.pca.explained_variance_ratio_):.4f}")
        else:
            signatures_reduced = signatures_scaled

        # K-Means Clustering
        self.kmeans.fit(signatures_reduced)

        # Speichere die Signatur-Vektoren
        self.signatures = signatures_reduced

        return self

    def predict(self, sequences, masks=None):
        """
        Weist neue Sequenzen den gelernten Clustern zu
        """
        # Berechne Signaturen für alle Sequenzen
        signatures = []

        for i, seq in enumerate(sequences):
            mask = masks[i] if masks is not None else None
            signature = self.signature_transform.compute_path_signature(seq, mask)
            signatures.append(signature)

        # Konvertiere in NumPy-Array
        signatures = np.array(signatures)

        # Standardisiere mit dem trainierten Scaler
        signatures_scaled = self.scaler.transform(signatures)

        # PCA-Transformation, falls zuvor angewendet
        if hasattr(self, 'pca') and hasattr(self.pca, 'components_'):
            signatures_reduced = self.pca.transform(signatures_scaled)
        else:
            signatures_reduced = signatures_scaled

        # Cluster-Zuweisung
        cluster_labels = self.kmeans.predict(signatures_reduced)

        return cluster_labels

    def fit_predict(self, sequences, masks=None):
        """
        Führt Training und Vorhersage in einem Schritt durch
        """
        self.fit(sequences, masks)
        return self.kmeans.labels_


import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import random
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


class StatisticalFeatureRepresentation:
    def __init__(self, use_pca=True, n_components=10, n_clusters=5, random_state=42):
        self.use_pca = use_pca
        self.n_components = n_components
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components) if use_pca else None
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        # Store feature importance information
        self.feature_importances = None
        self.feature_names = None

    def extract_statistical_features(self, sequence, mask=None):
        """
        Extrahiert 14 statistische Features für jede Variable in der Sequenz.

        Args:
            sequence: Array der Form [n_timesteps, n_features]
            mask: Optionale Maske der Form [n_timesteps, n_features] mit 1 für gültige Werte

        Returns:
            Array mit statistischen Features pro Variable
        """
        # Wenn keine Maske vorhanden, nehme an, dass alle Werte gültig sind
        if mask is None:
            mask = np.ones_like(sequence)
        elif len(mask.shape) == 1:
            # Wenn die Maske eindimensional ist, erweitere sie
            mask = np.repeat(mask[:, np.newaxis], sequence.shape[1], axis=1)

        n_timesteps, n_features = sequence.shape
        stats_features = []

        for i in range(n_features):
            # Extrahiere nur die gültigen Werte für diese Variable
            valid_indices = mask[:, i] > 0
            valid_values = sequence[valid_indices, i]

            if len(valid_values) == 0:
                # Keine gültigen Werte, verwende Nullwerte
                feature_stats = np.zeros(14)
            else:
                try:
                    # 1. Zentrale Tendenz (4 Features)
                    mean = np.mean(valid_values)
                    median = np.median(valid_values)

                    # Modus (häufigster Wert) - für kontinuierliche Daten approximiert
                    hist, bin_edges = np.histogram(valid_values, bins=min(10, len(valid_values)))
                    mode_idx = np.argmax(hist)
                    mode = (bin_edges[mode_idx] + bin_edges[mode_idx + 1]) / 2

                    # 25% Quantil
                    q25 = np.percentile(valid_values, 25)

                    # 2. Streuungstendenz (6 Features)
                    min_val = np.min(valid_values)
                    max_val = np.max(valid_values)
                    std = np.std(valid_values)

                    # Variationskoeffizient (Standardabweichung / Mittelwert)
                    cv = std / (mean if mean != 0 else 1)

                    # Bereich (Range)
                    range_val = max_val - min_val

                    # Interquartilsabstand (IQR)
                    q75 = np.percentile(valid_values, 75)
                    iqr = q75 - q25

                    # 3. Verteilungsform (4 Features)
                    # Schiefe (Skewness)
                    skewness = stats.skew(valid_values) if len(valid_values) > 2 else 0

                    # Kurtosis (Wölbung)
                    kurtosis = stats.kurtosis(valid_values) if len(valid_values) > 2 else 0

                    # Median absolute deviation (MAD)
                    mad = np.median(np.abs(valid_values - median))

                    # Zero crossing rate - Prozentuales Durchqueren der Nulllinie
                    zero_crossings = np.where(np.diff(np.signbit(valid_values)))[0].shape[0]
                    zcr = zero_crossings / (len(valid_values) - 1) if len(valid_values) > 1 else 0

                    # Kombiniere alle Features
                    feature_stats = np.array([
                        mean, median, mode, q25,  # Zentrale Tendenz
                        min_val, max_val, std, cv, range_val, iqr,  # Streuungstendenz
                        skewness, kurtosis, mad, zcr  # Verteilungsform
                    ])
                except:
                    # Bei Berechnungsproblemen verwende Nullwerte
                    feature_stats = np.zeros(14)

            stats_features.append(feature_stats)

        # Verkette Features für alle Variablen
        return np.concatenate(stats_features)

    def fit(self, sequences, masks=None):
        """
        Extrahiert statistische Features für alle Sequenzen und wendet Clustering an.

        Args:
            sequences: Liste oder Array von Sequenzen, jede mit Form [n_timesteps, n_features]
            masks: Optionale Liste oder Array von Masken, jede mit Form [n_timesteps, n_features]

        Returns:
            self
        """
        print("Extrahiere statistische Features...")
        all_stats_features = []

        # Store input feature names if available (from global dictionary)
        if 'FEATURE_NAMES' in globals() and FEATURE_NAMES:
            self.feature_names = [FEATURE_NAMES.get(i, f"Feature_{i}") for i in range(sequences[0].shape[1])]
        else:
            self.feature_names = [f"Feature_{i}" for i in range(sequences[0].shape[1])]

        for i, seq in enumerate(sequences):
            mask = masks[i] if masks is not None else None
            stats_features = self.extract_statistical_features(seq, mask)
            all_stats_features.append(stats_features)

        # Konvertiere zu Numpy-Array
        all_stats_features = np.array(all_stats_features)

        # Behandle NaN-Werte
        all_stats_features = np.nan_to_num(all_stats_features)

        print(f"Extrahierte Feature-Form: {all_stats_features.shape}")

        # Standardisiere Features
        self.stats_features = self.scaler.fit_transform(all_stats_features)

        # Dimensionsreduktion mit PCA, falls gewünscht
        if self.use_pca and self.stats_features.shape[1] > self.n_components:
            print(
                f"Wende PCA an, um Dimensionen von {self.stats_features.shape[1]} auf {self.n_components} zu reduzieren...")
            self.feature_vectors = self.pca.fit_transform(self.stats_features)
            explained_var = np.sum(self.pca.explained_variance_ratio_)
            print(f"Erklärte Varianz durch PCA: {explained_var:.4f}")

            # Calculate feature importance from PCA components
            pca_importance = np.abs(self.pca.components_).mean(axis=0)
            self.feature_importances = pca_importance
        else:
            self.feature_vectors = self.stats_features
            # When not using PCA, use feature variance as importance
            self.feature_importances = np.var(self.stats_features, axis=0)

        # Anwendung von K-Means-Clustering
        print(f"Wende K-Means-Clustering mit {self.n_clusters} Clustern an...")
        self.kmeans.fit(self.feature_vectors)

        return self

    def fit_predict(self, sequences, masks=None):
        """
        Führt Fitting und Prediction in einem Schritt durch.

        Returns:
            Cluster-Labels für die Eingabesequenzen
        """
        self.fit(sequences, masks)
        return self.kmeans.labels_

    def predict(self, sequences, masks=None):
        """
        Weist neue Sequenzen den gelernten Clustern zu.

        Returns:
            Cluster-Labels für die Eingabesequenzen
        """
        all_stats_features = []

        for i, seq in enumerate(sequences):
            mask = masks[i] if masks is not None else None
            stats_features = self.extract_statistical_features(seq, mask)
            all_stats_features.append(stats_features)

        all_stats_features = np.array(all_stats_features)
        all_stats_features = np.nan_to_num(all_stats_features)

        # Standardisiere mit dem trainierten Scaler
        scaled_features = self.scaler.transform(all_stats_features)

        # PCA-Transformation, falls zuvor angewendet
        if self.use_pca:
            feature_vectors = self.pca.transform(scaled_features)
        else:
            feature_vectors = scaled_features

        # Cluster-Zuweisung
        return self.kmeans.predict(feature_vectors)

    def get_feature_importance(self):
        """
        Extrahiert die Wichtigkeit der Features basierend auf der Distanz zu den Clusterzentren.

        Returns:
            DataFrame mit Feature-Namen und deren Wichtigkeitswerten
        """
        if not hasattr(self, 'kmeans') or not hasattr(self, 'feature_vectors'):
            print("Modell wurde noch nicht trainiert.")
            return None

        if self.feature_importances is None:
            # Fallback if feature importances weren't calculated during fit
            self.feature_importances = np.var(self.stats_features, axis=0)

        # Create feature names for all statistical features
        feature_names = []
        stat_names = ["Mittelwert", "Median", "Modus", "Q25",
                      "Min", "Max", "Std", "CV", "Range", "IQR",
                      "Schiefe", "Kurtosis", "MAD", "ZCR"]

        for i in range(len(self.feature_names)):
            var_name = self.feature_names[i]
            for stat_name in stat_names:
                feature_names.append(f"{var_name}_{stat_name}")

        # Ensure we have the right number of feature names
        feature_names = feature_names[:len(self.feature_importances)]

        # Sort features by importance
        importance_idx = np.argsort(self.feature_importances)[::-1]

        # Create DataFrame with feature names and importance
        importance_df = pd.DataFrame({
            'Feature': [feature_names[i] for i in importance_idx],
            'Importance': self.feature_importances[importance_idx],
            'Original_Feature': [self.feature_names[i // 14] for i in importance_idx]
        })

        return importance_df

    def visualize_feature_importance(self, top_n=20):
        """
        Visualize the most important features

        Args:
            top_n: Number of top features to display
        """
        importance_df = self.get_feature_importance()
        if importance_df is None:
            return

        top_features = importance_df.head(top_n)

        plt.figure(figsize=(12, 8))
        bars = plt.barh(top_features['Feature'], top_features['Importance'])

        # Color bars by original feature
        unique_orig_features = top_features['Original_Feature'].unique()
        color_map = plt.cm.get_cmap('tab10', len(unique_orig_features))
        feature_to_color = {feat: color_map(i) for i, feat in enumerate(unique_orig_features)}

        for i, bar in enumerate(bars):
            orig_feature = top_features.iloc[i]['Original_Feature']
            bar.set_color(feature_to_color[orig_feature])

        plt.xlabel('Importance')
        plt.title('Top Statistical Features by Importance')
        plt.gca().invert_yaxis()  # Display highest importance at top

        # Add a legend for original features
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color=color_map(i), lw=4, label=feat)
                           for i, feat in enumerate(unique_orig_features)]
        plt.legend(handles=legend_elements, title="Original Features",
                   loc='lower right', bbox_to_anchor=(1.1, 0))

        plt.tight_layout()
        plt.savefig('statistical_feature_importance.png')
        plt.close()

    def visualize_clusters(self, data=None, cluster_labels=None):
        """
        Visualize clusters with original feature information

        Args:
            data: Original data sequences (optional)
            cluster_labels: Cluster labels (if None, uses the model's labels)
        """
        if cluster_labels is None and hasattr(self, 'kmeans'):
            cluster_labels = self.kmeans.labels_

        if cluster_labels is None:
            print("No cluster labels available for visualization")
            return

        # PCA visualization of feature vectors
        plt.figure(figsize=(12, 10))

        # Get 2D representation of feature vectors
        if self.feature_vectors.shape[1] > 2:
            viz_pca = PCA(n_components=2)
            viz_data = viz_pca.fit_transform(self.feature_vectors)
        else:
            viz_data = self.feature_vectors

        # Create scatter plot with cluster colors
        unique_clusters = np.unique(cluster_labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_clusters)))

        for i, cluster_id in enumerate(unique_clusters):
            cluster_points = viz_data[cluster_labels == cluster_id]
            plt.scatter(
                cluster_points[:, 0],
                cluster_points[:, 1],
                color=colors[i],
                label=f'Cluster {cluster_id}',
                alpha=0.7
            )

            # Add cluster center
            if hasattr(self, 'kmeans'):
                center = viz_pca.transform([self.kmeans.cluster_centers_[cluster_id]])[0]
                plt.scatter(
                    center[0], center[1],
                    s=200, color=colors[i],
                    edgecolor='black', linewidth=2,
                    marker='*'
                )

                # Add cluster number
                plt.annotate(
                    f"{cluster_id}",
                    (center[0], center[1]),
                    fontsize=14,
                    fontweight='bold',
                    ha='center', va='center'
                )

        plt.title('Statistical Feature Clusters Visualization')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.legend()

        # Add explanatory text for each cluster
        if data is not None and hasattr(self, 'feature_names'):
            for cluster_id in unique_clusters:
                # Get data for this cluster
                cluster_data = data[cluster_labels == cluster_id]

                # Find key features for this cluster
                feature_means = np.mean(cluster_data, axis=(0, 1))
                # Get top 3 features by absolute mean value
                top_indices = np.argsort(np.abs(feature_means))[-3:][::-1]

                # Calculate centroid
                centroid = np.mean(viz_data[cluster_labels == cluster_id], axis=0)

                # Create annotation with top features
                annotation = "\n".join([
                    f"{self.feature_names[i]}: {feature_means[i]:.2f}"
                    for i in top_indices
                ])

                # Add annotation with offset
                plt.annotate(
                    annotation,
                    xy=(centroid[0], centroid[1]),
                    xytext=(centroid[0] + 0.5, centroid[1] + 0.5),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3")
                )

        plt.tight_layout()
        plt.savefig('statistical_feature_clusters.png')
        plt.close()


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import mean_squared_error
from sklearn.manifold import TSNE


class RGANGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim=100, output_dim=None, n_layers=1, dropout=0.1):
        super(RGANGenerator, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.n_layers = n_layers

        # Noise-zu-Hidden Layer
        self.noise_to_hidden = nn.Linear(input_dim, hidden_dim)

        # GRU Layer - generiert Sequenzen
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )

        # Output Layer - projiziert GRU-Ausgabe auf die Zieldimension
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # Aktivierungsfunktionen
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()  # Tanh für normalisierte Ausgabe zwischen -1 und 1

    def forward(self, noise, seq_len):
        """
        Args:
            noise: Rauschen der Form [batch_size, input_dim]
            seq_len: Länge der zu generierenden Sequenzen
        """
        batch_size = noise.size(0)

        # Transformiere Rauschen
        noise_hidden = self.relu(self.noise_to_hidden(noise))

        # Initialisiere versteckten Zustand für GRU
        h_0 = noise_hidden.unsqueeze(0).repeat(self.n_layers, 1, 1)

        # Wiederhole den Input für jeden Zeitschritt
        repeated_input = noise_hidden.unsqueeze(1).repeat(1, seq_len, 1)

        # GRU forward pass
        gru_out, _ = self.gru(repeated_input, h_0)

        # Projektion auf Ausgabedimension
        output = self.output_layer(gru_out)

        # Tanh-Aktivierung für normalisierte Ausgabe
        return self.tanh(output)


class RGANDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=100, n_layers=1, dropout=0.1):
        super(RGANDiscriminator, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # GRU Layer - verarbeitet die Sequenzen
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )

        # Attention Mechanismus
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softmax(dim=1)
        )

        # Fully Connected Layers für die Klassifikation
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Ausgabe zwischen 0 und 1 (echt oder gefälscht)
        )

    def forward(self, x):
        """
        Args:
            x: Sequenzen der Form [batch_size, seq_len, input_dim]
        """
        # GRU forward pass
        gru_out, _ = self.gru(x)

        # Attention-Mechanismus
        attention_weights = self.attention(gru_out)
        context = torch.sum(attention_weights * gru_out, dim=1)

        # Klassifikation
        validity = self.fc(context)

        return validity


class RGAN:
    def __init__(self, input_dim, hidden_dim=100, n_layers=1, dropout=0.1, device='cpu'):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.device = device
        self.feature_names = None

        # Initialisiere Generator und Diskriminator
        self.generator = RGANGenerator(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            n_layers=n_layers,
            dropout=dropout
        ).to(device)

        self.discriminator = RGANDiscriminator(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout
        ).to(device)

        # Optimierer
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        # Verlustfunktionen
        self.adversarial_loss = nn.BCELoss()

        # Training-Historie
        self.g_losses = []
        self.d_losses = []
        self.mmd_scores = []

        # Store feature importance based on discriminator
        self.feature_importances = None

    def train(self, batches, epochs=50, sample_interval=10):
        """
        Trainiert das RGAN-Modell.

        Args:
            batches: Liste von Batches (sequences, masks)
            epochs: Anzahl der Trainingsepochen
            sample_interval: Intervall für das Sampling und die Auswertung
        """
        # Label für echte und gefälschte Daten
        valid = torch.ones(1, 1).to(self.device)
        fake = torch.zeros(1, 1).to(self.device)

        # Store feature names if available (from global dictionary)
        if 'FEATURE_NAMES' in globals() and FEATURE_NAMES:
            sample_seq = batches[0][0][0]  # First sequence in first batch
            self.feature_names = [FEATURE_NAMES.get(i, f"Feature_{i}") for i in range(sample_seq.shape[1])]

        for epoch in range(epochs):
            epoch_g_loss = 0
            epoch_d_loss = 0
            batch_count = 0

            # Batch-Shuffle für jede Epoche
            random.shuffle(batches)

            for batch_sequences, batch_masks in batches:
                # Konvertiere zu Torch Tensoren
                real_seqs = torch.FloatTensor(batch_sequences).to(self.device)
                masks = torch.FloatTensor(batch_masks).to(self.device)

                # Ignoriere Batches, die zu klein sind
                if real_seqs.size(0) < 2:
                    continue

                batch_size, seq_len, _ = real_seqs.shape

                # Erweitere Masken, falls sie 2D sind
                if len(masks.shape) == 2:
                    masks = masks.unsqueeze(-1).expand_as(real_seqs)

                # Anwenden der Masken auf echte Sequenzen
                real_seqs = real_seqs * masks

                # ---------------------
                #  Trainiere Diskriminator
                # ---------------------
                self.d_optimizer.zero_grad()

                # Echte Sequenzen
                real_validity = self.discriminator(real_seqs)
                d_real_loss = self.adversarial_loss(real_validity, valid.expand_as(real_validity))

                # Generiere gefälschte Sequenzen
                z = torch.randn(batch_size, self.input_dim).to(self.device)
                gen_seqs = self.generator(z, seq_len)

                # Anwenden der Masken auf generierte Sequenzen
                gen_seqs = gen_seqs * masks

                # Gefälschte Sequenzen
                fake_validity = self.discriminator(gen_seqs.detach())
                d_fake_loss = self.adversarial_loss(fake_validity, fake.expand_as(fake_validity))

                # Gesamtverlust Diskriminator
                d_loss = (d_real_loss + d_fake_loss) / 2
                d_loss.backward()
                self.d_optimizer.step()

                # ---------------------
                #  Trainiere Generator
                # ---------------------
                self.g_optimizer.zero_grad()

                # Generiere Sequenzen und berechne Verlust
                gen_validity = self.discriminator(gen_seqs)
                g_loss = self.adversarial_loss(gen_validity, valid.expand_as(gen_validity))

                g_loss.backward()
                self.g_optimizer.step()

                # Verluste für diese Iteration
                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()
                batch_count += 1

            # Durchschnittliche Verluste für diese Epoche
            avg_g_loss = epoch_g_loss / max(1, batch_count)
            avg_d_loss = epoch_d_loss / max(1, batch_count)

            self.g_losses.append(avg_g_loss)
            self.d_losses.append(avg_d_loss)

            # Ausgabe für jede Epoche
            print(f"Epoch {epoch + 1}/{epochs} - G_loss: {avg_g_loss:.4f}, D_loss: {avg_d_loss:.4f}")

            # Sample und Auswertung in bestimmten Intervallen
            if (epoch + 1) % sample_interval == 0:
                self._evaluate(batches)

        # After training, compute feature importances
        self._compute_feature_importance(batches)

    def _evaluate(self, batches, n_samples=10):
        """
        Evaluiert das RGAN durch Sampling und MMD-Berechnung.
        """
        # Extrahiere einige echte Sequenzen aus den Batches für Vergleich
        real_samples = []
        for batch_sequences, batch_masks in batches[:5]:  # Nur die ersten 5 Batches
            real_samples.extend(batch_sequences)

        real_samples = real_samples[:n_samples]
        real_samples = np.array(real_samples)

        # Generiere synthetische Sequenzen
        seq_len = real_samples.shape[1] if real_samples.shape[0] > 0 else 50

        z = torch.randn(n_samples, self.input_dim).to(self.device)
        with torch.no_grad():
            gen_samples = self.generator(z, seq_len).cpu().numpy()

        # Berechne MMD zwischen echten und generierten Proben
        if len(real_samples) > 0:
            mmd_score = self.compute_mmd(real_samples.reshape(n_samples, -1),
                                         gen_samples.reshape(n_samples, -1))
            self.mmd_scores.append(mmd_score)
            print(f"MMD Score: {mmd_score:.4f}")

    def compute_mmd(self, x, y, kernel='rbf', sigma=None):
        """
        Berechnet Maximum Mean Discrepancy zwischen zwei Verteilungen.
        """
        xx, yy, xy = 0, 0, 0

        # Standardabweichung des Kernels schätzen
        if sigma is None:
            sigma = np.median(np.sqrt(np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=2)))
            sigma = max(sigma, 1e-5)  # Vermeide zu kleine Werte

        gamma = 1 / (2 * sigma ** 2)

        # xx-Term
        for i in range(x.shape[0]):
            for j in range(i + 1, x.shape[0]):
                if kernel == 'rbf':
                    s = np.exp(-gamma * np.sum((x[i] - x[j]) ** 2))
                else:
                    s = np.dot(x[i], x[j])
                xx += s

        # yy-Term
        for i in range(y.shape[0]):
            for j in range(i + 1, y.shape[0]):
                if kernel == 'rbf':
                    s = np.exp(-gamma * np.sum((y[i] - y[j]) ** 2))
                else:
                    s = np.dot(y[i], y[j])
                yy += s

        # xy-Term
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                if kernel == 'rbf':
                    s = np.exp(-gamma * np.sum((x[i] - y[j]) ** 2))
                else:
                    s = np.dot(x[i], y[j])
                xy += s

        # Normalisiere die Terme
        n = x.shape[0]
        m = y.shape[0]

        xx = 2 * xx / (n * (n - 1)) if n > 1 else 0
        yy = 2 * yy / (m * (m - 1)) if m > 1 else 0
        xy = xy / (n * m)

        return xx + yy - 2 * xy

    def generate_samples(self, n_samples, seq_len):
        """
        Generiert n_samples Sequenzen der Länge seq_len.
        """
        z = torch.randn(n_samples, self.input_dim).to(self.device)

        with torch.no_grad():
            samples = self.generator(z, seq_len).cpu().numpy()

        return samples

    def plot_losses(self):
        """
        Plottet die Verluste während des Trainings.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.g_losses, label='Generator')
        plt.plot(self.d_losses, label='Diskriminator')
        plt.title('RGAN Verluste')
        plt.xlabel('Epoche')
        plt.ylabel('Verlust')
        plt.legend()
        plt.savefig('rgan_losses.png')
        plt.close()

        if self.mmd_scores:
            plt.figure(figsize=(10, 5))
            plt.plot(self.mmd_scores)
            plt.title('MMD Score (lower is better)')
            plt.xlabel('Evaluierung')
            plt.ylabel('MMD')
            plt.savefig('rgan_mmd.png')
            plt.close()

    def visualize_samples(self, real_samples, n_samples=5):
        """
        Visualisiert generierte Samples im Vergleich zu echten Samples.
        """
        # Generiere neue Samples
        seq_len = real_samples.shape[1]
        gen_samples = self.generate_samples(n_samples, seq_len)

        # Plot für jede Variable
        for var_idx in range(min(3, self.input_dim)):  # Begrenze auf max. 3 Variablen
            var_name = self.feature_names[var_idx] if self.feature_names else f"Variable {var_idx + 1}"

            plt.figure(figsize=(12, 6))

            # Echte Samples
            for i in range(min(n_samples, len(real_samples))):
                plt.subplot(2, 1, 1)
                plt.plot(real_samples[i, :, var_idx], label=f'Sample {i + 1}' if i == 0 else None)

            plt.title(f'Echte Samples: {var_name}')
            plt.xlabel('Zeitschritt')
            plt.ylabel(f'Wert')
            if var_idx == 0:
                plt.legend()

            # Generierte Samples
            for i in range(n_samples):
                plt.subplot(2, 1, 2)
                plt.plot(gen_samples[i, :, var_idx], label=f'Sample {i + 1}' if i == 0 else None)

            plt.title(f'Generierte Samples: {var_name}')
            plt.xlabel('Zeitschritt')
            plt.ylabel(f'Wert')
            if var_idx == 0:
                plt.legend()

            plt.tight_layout()
            plt.savefig(f'rgan_samples_{var_name}.png')
            plt.close()

    def get_latent_representation(self, sequences, masks=None):
        """
        Extrahiert latente Repräsentationen von Sequenzen mit dem Diskriminator.
        Dies kann als Feature-Extraktion für Downstream-Tasks verwendet werden.
        """
        self.discriminator.eval()
        latent_representations = []

        with torch.no_grad():
            for i in range(len(sequences)):
                seq = torch.FloatTensor(sequences[i]).unsqueeze(0).to(self.device)

                if masks is not None:
                    mask = torch.FloatTensor(masks[i]).unsqueeze(0).to(self.device)

                    # Erweitere Masken, falls sie 2D sind
                    if len(mask.shape) == 2:
                        mask = mask.unsqueeze(-1).expand_as(seq)

                    # Anwenden der Masken
                    seq = seq * mask

                # Extrahiere Features vom Diskriminator (vor der letzten Schicht)
                gru_out, _ = self.discriminator.gru(seq)
                attention_weights = self.discriminator.attention(gru_out)
                context = torch.sum(attention_weights * gru_out, dim=1)

                latent_representations.append(context.cpu().numpy())

        return np.vstack(latent_representations)

    def _compute_feature_importance(self, batches):
        """
        Compute feature importance by analyzing the discriminator's sensitivity to each feature
        """
        sample_sequences, sample_masks = batches[0]
        sample_sequences = torch.FloatTensor(sample_sequences).to(self.device)
        sample_masks = torch.FloatTensor(sample_masks).to(self.device)

        if len(sample_masks.shape) == 2:
            sample_masks = sample_masks.unsqueeze(-1).expand_as(sample_sequences)

        # Apply masks to sequences
        sample_sequences = sample_sequences * sample_masks

        # Get baseline predictions
        self.discriminator.eval()
        with torch.no_grad():
            baseline_preds = self.discriminator(sample_sequences).mean().item()

        # For each feature, measure impact when perturbed
        feature_impacts = []
        for feat_idx in range(self.input_dim):
            # Create perturbed sequences
            perturbed_sequences = sample_sequences.clone()

            # Zero out this feature
            perturbed_sequences[:, :, feat_idx] = 0

            # Get predictions for perturbed sequences
            with torch.no_grad():
                perturbed_preds = self.discriminator(perturbed_sequences).mean().item()

            # Impact is the change in prediction
            impact = abs(baseline_preds - perturbed_preds)
            feature_impacts.append(impact)

        # Store feature importances
        self.feature_importances = np.array(feature_impacts)

    def visualize_feature_importance(self):
        """Visualize feature importance from the discriminator"""
        if self.feature_importances is None:
            print("No feature importance data available. Train the model first.")
            return

        # Sort features by importance
        sorted_indices = np.argsort(self.feature_importances)[::-1]
        sorted_importances = self.feature_importances[sorted_indices]

        # Get feature names
        feature_names = self.feature_names if self.feature_names else [f"Feature {i}" for i in range(self.input_dim)]
        sorted_names = [feature_names[i] for i in sorted_indices]

        # Plot feature importance
        plt.figure(figsize=(12, 8))
        plt.bar(range(len(sorted_importances)), sorted_importances)
        plt.xticks(range(len(sorted_importances)), sorted_names, rotation=90)
        plt.title('Feature Importance in RGAN Discriminator')
        plt.xlabel('Features')
        plt.ylabel('Importance Score')
        plt.tight_layout()
        plt.savefig('rgan_feature_importance.png')
        plt.close()

        # Return top 5 most important features
        top_features = [(sorted_names[i], sorted_importances[i]) for i in range(min(5, len(sorted_importances)))]
        return top_features

    def save_model(self, path):
        """
        Speichert das trainierte Modell.
        """
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'g_losses': self.g_losses,
            'd_losses': self.d_losses,
            'mmd_scores': self.mmd_scores
        }, path)

    def load_model(self, path):
        """
        Lädt ein vortrainiertes Modell.
        """
        checkpoint = torch.load(path)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        self.g_losses = checkpoint['g_losses']
        self.d_losses = checkpoint['d_losses']
        self.mmd_scores = checkpoint['mmd_scores']


# Zunächst müssen wir eine angepasste RGAN-Klasse für das Benchmarking erstellen
class BenchmarkRGAN(RGAN):
    """
    Eine spezielle Version von RGAN für das Benchmarking, die flexibler mit der Anzahl der Samples umgeht.
    """

    def _evaluate(self, batches, n_samples=None):
        """
        Evaluiert das RGAN durch Sampling und MMD-Berechnung.
        Diese Version ist angepasst, um mit weniger Samples zu funktionieren.
        """
        # Extrahiere einige echte Sequenzen aus den Batches für Vergleich
        real_samples = []
        for batch_sequences, batch_masks in batches[:5]:  # Nur die ersten 5 Batches
            real_samples.extend(batch_sequences)

        # Bestimme die Anzahl der zu vergleichenden Samples
        if n_samples is None:
            n_samples = min(10, len(real_samples))  # Verwende höchstens 10 Samples

        if len(real_samples) == 0 or n_samples == 0:
            print("Keine Samples für die Evaluation verfügbar.")
            return

        real_samples = real_samples[:n_samples]
        real_samples = np.array(real_samples)

        # Generiere synthetische Sequenzen
        seq_len = real_samples.shape[1]

        z = torch.randn(n_samples, self.input_dim).to(self.device)
        with torch.no_grad():
            gen_samples = self.generator(z, seq_len).cpu().numpy()

        # Berechne MMD zwischen echten und generierten Proben
        try:
            mmd_score = self.compute_mmd(real_samples.reshape(n_samples, -1),
                                         gen_samples.reshape(n_samples, -1))
            self.mmd_scores.append(mmd_score)
            print(f"MMD Score: {mmd_score:.4f}")
        except Exception as e:
            print(f"Fehler bei der MMD-Berechnung: {e}")


# Diese Funktionen müssen außerhalb aller Klassen definiert werden
def use_signature_transform_clustering(batches, n_clusters=5, truncation_level=3):
    print("\n==== Signature Transform Clustering ====")

    # Batchesdaten in Sequenzen und Masken aufteilen
    all_sequences = []
    all_masks = []

    for batch_sequences, batch_masks in batches:
        all_sequences.extend(batch_sequences)
        all_masks.extend(batch_masks)

    all_sequences = np.array(all_sequences)
    all_masks = np.array(all_masks)

    print(f"Signature Transform: Verarbeite {len(all_sequences)} Sequenzen mit Trunkierungslevel {truncation_level}")

    # Überprüfen der Maskenform und korrigieren, falls notwendig
    if len(all_masks.shape) > 2:
        # Wenn die Masken 3D sind (batch, seq_len, features), reduziere sie auf 2D
        all_masks = all_masks.mean(axis=2) > 0  # Mittelwert über Features und binarisieren

    # Modell initialisieren und anwenden
    model = SignatureTransformClustering(
        truncation_level=truncation_level,
        n_clusters=n_clusters
    )

    # Store feature names if available
    if 'FEATURE_NAMES' in globals() and FEATURE_NAMES:
        model.feature_names = [FEATURE_NAMES.get(i, f"Feature_{i}") for i in range(all_sequences[0].shape[1])]

    # Training und Clustering
    try:
        start_time = time.time()
        cluster_labels = model.fit_predict(all_sequences, all_masks)
        end_time = time.time()

        print(f"Clustering abgeschlossen in {end_time - start_time:.2f} Sekunden")

        # Analysiere Cluster-Zuweisungen
        cluster_counts = np.bincount(cluster_labels)
        print(f"Cluster-Zuweisungen: {cluster_counts}")

        # Berechne Silhouette-Score, wenn mehr als ein Cluster vorhanden ist
        if n_clusters > 1 and len(np.unique(cluster_labels)) > 1:
            from sklearn.metrics import silhouette_score
            try:
                silhouette_avg = silhouette_score(model.signatures, cluster_labels)
                print(f"Silhouette-Score: {silhouette_avg:.4f}")
            except:
                print("Silhouette-Score konnte nicht berechnet werden")

        # Berechne die Cluster-Zentren im Originalraum (falls möglich)
        if hasattr(model, 'signatures'):
            cluster_centers = model.kmeans.cluster_centers_
            print(f"Cluster-Zentren-Form: {cluster_centers.shape}")

        # Visualize clusters with feature interpretation
        visualize_signature_clusters(model, all_sequences, cluster_labels)

        return model, cluster_labels
    except Exception as e:
        print(f"Fehler beim Clustering: {e}")
        print("Versuche es mit einfacheren Einstellungen...")

        # Fallback mit vereinfachten Einstellungen
        model = SignatureTransformClustering(
            truncation_level=2,  # Reduzierte Trunkierungsebene
            n_clusters=min(n_clusters, len(all_sequences) // 2),  # Weniger Cluster
            pca_components=min(20, len(all_sequences) - 1)  # Weniger PCA-Komponenten
        )

        # Store feature names if available
        if 'FEATURE_NAMES' in globals() and FEATURE_NAMES:
            model.feature_names = [FEATURE_NAMES.get(i, f"Feature_{i}") for i in range(all_sequences[0].shape[1])]

        # Direktes K-Means-Clustering auf den Sequenzen, falls die Signatur-Transformation fehlschlägt
        from sklearn.cluster import KMeans

        # Flatten der Sequenzen für einfaches Clustering
        flattened_seqs = np.array([seq.flatten() for seq in all_sequences])

        # Begrenze die Dimension, falls nötig
        if flattened_seqs.shape[1] > 100:
            pca = PCA(n_components=100)
            flattened_seqs = pca.fit_transform(flattened_seqs)

        kmeans = KMeans(n_clusters=min(n_clusters, len(flattened_seqs) // 2), random_state=42)
        cluster_labels = kmeans.fit_predict(flattened_seqs)

        print("Fallback-Clustering durchgeführt")
        cluster_counts = np.bincount(cluster_labels)
        print(f"Cluster-Zuweisungen: {cluster_counts}")

        # Erstelle ein vereinfachtes Modell-Objekt mit den notwendigen Attributen
        class SimplifiedModel:
            def __init__(self, kmeans, data, feature_names=None):
                self.kmeans = kmeans
                self.signatures = data
                self.feature_names = feature_names

        model = SimplifiedModel(kmeans, flattened_seqs,
                                [FEATURE_NAMES.get(i, f"Feature_{i}") for i in range(all_sequences[0].shape[1])]
                                if 'FEATURE_NAMES' in globals() and FEATURE_NAMES else None)

        # Visualize fallback clusters
        visualize_signature_clusters(model, all_sequences, cluster_labels)

        return model, cluster_labels


def visualize_signature_clusters(model, sequences, cluster_labels):
    """
    Visualize signature transform clusters with feature interpretation

    Args:
        model: Trained SignatureTransformClustering model
        sequences: Original sequences data
        cluster_labels: Cluster assignments
    """
    # Prepare PCA projection of signatures for visualization
    if hasattr(model, 'signatures'):
        if model.signatures.shape[1] > 2:
            viz_pca = PCA(n_components=2)
            viz_data = viz_pca.fit_transform(model.signatures)
        else:
            viz_data = model.signatures

        # Plot clusters
        plt.figure(figsize=(12, 10))

        # Get feature names if available
        feature_names = model.feature_names if hasattr(model, 'feature_names') and model.feature_names else None

        # Color by cluster
        unique_clusters = np.unique(cluster_labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_clusters)))

        for i, cluster_id in enumerate(unique_clusters):
            # Plot points for this cluster
            cluster_points = viz_data[cluster_labels == cluster_id]
            plt.scatter(
                cluster_points[:, 0],
                cluster_points[:, 1],
                color=colors[i],
                label=f'Cluster {cluster_id}',
                alpha=0.7
            )

            # Add cluster center
            if hasattr(model, 'kmeans') and hasattr(model.kmeans, 'cluster_centers_'):
                center_idx = cluster_id
                if center_idx < len(model.kmeans.cluster_centers_):
                    # Transform center to 2D if needed
                    if model.signatures.shape[1] > 2:
                        center = viz_pca.transform([model.kmeans.cluster_centers_[center_idx]])[0]
                    else:
                        center = model.kmeans.cluster_centers_[center_idx]

                    # Plot center
                    plt.scatter(
                        center[0], center[1],
                        s=200, color=colors[i],
                        edgecolor='black', linewidth=2,
                        marker='X'
                    )

            # Add annotations if we have feature names
            if feature_names and sequences is not None:
                # Get data for this cluster
                cluster_seqs = sequences[cluster_labels == cluster_id]

                # Calculate mean time series for each feature
                mean_series = np.mean(cluster_seqs, axis=0)

                # Find features with highest variance or trend
                if mean_series.shape[0] > 1:  # At least 2 time points
                    # Calculate trends (end - start)
                    trends = mean_series[-1] - mean_series[0]
                    # Find top features by absolute trend
                    top_features_idx = np.argsort(np.abs(trends))[-3:][::-1]

                    # Create annotation text
                    annotation = "\n".join([
                        f"{feature_names[idx]}: {'↑' if trends[idx] > 0 else '↓'}{abs(trends[idx]):.2f}"
                        for idx in top_features_idx
                    ])

                    # Calculate centroid for annotation
                    if len(cluster_points) > 0:
                        centroid = np.mean(cluster_points, axis=0)

                        # Add annotation with offset
                        plt.annotate(
                            annotation,
                            xy=(centroid[0], centroid[1]),
                            xytext=(centroid[0] + 0.5, centroid[1] + 0.5),
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7),
                            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3")
                        )

        plt.title('Signature Transform Clusters Visualization')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.legend()
        plt.tight_layout()
        plt.savefig('signature_clusters.png')
        plt.close()

    # Also visualize time series patterns by cluster for key features
    if sequences is not None and feature_names:
        # Find top features by variance across all sequences
        feature_variance = np.var(np.mean(sequences, axis=1), axis=0)
        top_features = np.argsort(feature_variance)[-3:][::-1]  # Top 3 features

        # Plot time series for each top feature by cluster
        for feat_idx in top_features:
            plt.figure(figsize=(14, 8))

            for i, cluster_id in enumerate(unique_clusters):
                # Get sequences for this cluster
                cluster_seqs = sequences[cluster_labels == cluster_id]

                # Calculate mean and std
                mean_series = np.mean(cluster_seqs[:, :, feat_idx], axis=0)
                std_series = np.std(cluster_seqs[:, :, feat_idx], axis=0)

                # Plot with confidence interval
                time_points = range(len(mean_series))
                plt.plot(time_points, mean_series, label=f'Cluster {cluster_id}',
                         color=colors[i], linewidth=2)
                plt.fill_between(
                    time_points,
                    mean_series - std_series,
                    mean_series + std_series,
                    color=colors[i], alpha=0.2
                )

            feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f"Feature {feat_idx}"
            plt.title(f'Time Series Patterns by Cluster: {feat_name}')
            plt.xlabel('Time Steps')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'signature_timeseries_{feat_name.replace(" ", "_")}.png')
            plt.close()


def use_statistical_feature_representation(batches, n_clusters=5, use_pca=True, n_components=10):
    print("\n==== Statistische Feature-Repräsentation ====")

    # Extrahiere Sequenzen und Masken aus den Batches
    all_sequences = []
    all_masks = []

    for batch_sequences, batch_masks in batches:
        all_sequences.extend(batch_sequences)
        all_masks.extend(batch_masks)

    all_sequences = np.array(all_sequences)
    all_masks = np.array(all_masks)

    print(f"Verarbeite {len(all_sequences)} Sequenzen...")

    # Erstelle und trainiere das Modell
    model = StatisticalFeatureRepresentation(
        use_pca=use_pca,
        n_components=min(n_components, all_sequences.shape[0] - 1, all_sequences.shape[1] * 14),
        n_clusters=min(n_clusters, all_sequences.shape[0] // 2),
        random_state=42
    )

    # Messe die Zeit für das Clustering
    start_time = time.time()
    cluster_labels = model.fit_predict(all_sequences, all_masks)
    end_time = time.time()

    print(f"Clustering abgeschlossen in {end_time - start_time:.2f} Sekunden")

    # Analysiere die Clustering-Ergebnisse
    cluster_counts = np.bincount(cluster_labels)
    print(f"Cluster-Verteilung: {cluster_counts}")

    # Berechne Silhouette-Score, wenn mehr als ein Cluster vorhanden ist
    if model.n_clusters > 1 and len(np.unique(cluster_labels)) > 1:
        try:
            silhouette_avg = silhouette_score(model.feature_vectors, cluster_labels)
            print(f"Silhouette-Score: {silhouette_avg:.4f}")
        except Exception as e:
            print(f"Silhouette-Score konnte nicht berechnet werden: {e}")

    # Visualize feature importance
    model.visualize_feature_importance()

    # Visualize clusters
    model.visualize_clusters(all_sequences, cluster_labels)

    return model, cluster_labels


def use_rgan(batches, input_dim, device, epochs=30, sample_interval=5):
    print("\n==== Rekurrente Generative Adversarial Networks (RGAN) ====")

    # Extrahiere einige echte Sequenzen für die spätere Visualisierung
    real_samples = []
    for batch_sequences, _ in batches[:5]:
        real_samples.extend(batch_sequences)

    real_samples = np.array(real_samples[:10])  # Beschränke auf max. 10 Samples

    # Initialisiere und trainiere das RGAN
    rgan = RGAN(
        input_dim=input_dim,
        hidden_dim=128,
        n_layers=2,
        dropout=0.1,
        device=device
    )

    print(f"Training RGAN mit {len(batches)} Batches für {epochs} Epochen...")
    start_time = time.time()

    rgan.train(batches, epochs=epochs, sample_interval=sample_interval)

    end_time = time.time()
    print(f"Training abgeschlossen in {end_time - start_time:.2f} Sekunden")

    # Plotte die Verluste
    rgan.plot_losses()

    # Visualisiere generierte Samples
    rgan.visualize_samples(real_samples)

    # Visualize feature importance
    top_features = rgan.visualize_feature_importance()
    if top_features:
        print("Top 5 most important features for the RGAN discriminator:")
        for i, (feature, importance) in enumerate(top_features):
            print(f"{i + 1}. {feature}: {importance:.4f}")

    # Extrahiere latente Repräsentationen für alle Sequenzen
    print("Extrahiere latente Repräsentationen...")

    all_sequences = []
    all_masks = []

    for batch_sequences, batch_masks in batches:
        all_sequences.extend(batch_sequences)
        all_masks.extend(batch_masks)

    latent_reps = rgan.get_latent_representation(all_sequences, all_masks)

    # Visualisiere die latenten Repräsentationen (verwende PCA statt t-SNE)
    if latent_reps.shape[0] > 2:  # Mindestens 3 Samples für sinnvolle Visualisierung
        print("Visualisiere latente Repräsentationen mit PCA...")

        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(latent_reps)

            plt.figure(figsize=(10, 8))
            plt.scatter(pca_result[:, 0], pca_result[:, 1])
            plt.title('PCA der RGAN latenten Repräsentationen')
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.savefig('rgan_pca.png')
            plt.close()

            print(f"PCA-Visualisierung erfolgreich, erklärte Varianz: {sum(pca.explained_variance_ratio_):.2f}")

            # Optional: Wenn es genügend Samples für t-SNE gibt, versuche es auch
            if latent_reps.shape[0] > 50:  # Mehr als 50 Samples für t-SNE
                try:
                    perplexity = min(30, latent_reps.shape[0] // 3)  # Sicherer Wert für Perplexität
                    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                    tsne_result = tsne.fit_transform(latent_reps)

                    plt.figure(figsize=(10, 8))
                    plt.scatter(tsne_result[:, 0], tsne_result[:, 1])
                    plt.title('t-SNE der RGAN latenten Repräsentationen')
                    plt.xlabel('Dimension 1')
                    plt.ylabel('Dimension 2')
                    plt.savefig('rgan_tsne.png')
                    plt.close()

                    print(f"t-SNE-Visualisierung mit Perplexität {perplexity} erfolgreich")
                except Exception as e:
                    print(f"t-SNE-Visualisierung fehlgeschlagen: {e}")
        except Exception as e:
            print(f"Visualisierung fehlgeschlagen: {e}")

    return rgan, latent_reps


import torch
import torch.nn as nn
from sklearn.metrics import silhouette_score
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import time
from scipy import stats
from sklearn.manifold import TSNE
import warnings
import seaborn as sns
from scipy.stats import ttest_ind, f_oneway
import matplotlib.cm as cm

warnings.filterwarnings('ignore')

# Global switches
TRAIN_TSTCC = True  # Switch for TS-TCC Training
TRAIN_RAUTO = True  # Switch for Recurrent Autoencoder Training
TRAIN_IMISS = True  # Switch for Informative Missingness Model
TRAIN_SIGNATURE = True  # Switch for Signature Transform Clustering
TRAIN_STATFEAT = True  # Switch for Statistical Feature Representation
TRAIN_RGAN = True  # Switch for Recurrent GAN
batch_size = 32  # Standard batch size

# Dictionary to store original feature names
FEATURE_NAMES = {}  # Will be populated from dataset


# Function to get actual feature names
def get_feature_names(data):
    """Extract and store the actual feature names from the dataset"""
    # Remove non-numeric or non-training relevant columns
    drop_cols = ['SepsisLabel', 'Patient_ID', 'ICULOS', 'Hour', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime']
    feature_cols = [col for col in data.columns if col not in drop_cols]

    # Store in global dictionary
    global FEATURE_NAMES
    FEATURE_NAMES = {i: name for i, name in enumerate(feature_cols)}
    return feature_cols


# ---------------------- Cluster interpretation functions ----------------------

def analyze_cluster_characteristics(model, data, cluster_labels, feature_names=None, model_type='imiss'):
    """
    Analyzes what variables and patterns characterize each cluster.

    Args:
        model: The trained model
        data: The original data (could be padded sequences or original patient data)
        cluster_labels: The cluster assignments
        feature_names: Names of the features (if None, will use indices)
        model_type: Type of model being analyzed

    Returns:
        Dictionary with cluster characteristics
    """
    n_clusters = len(np.unique(cluster_labels))
    characteristics = {}

    # If feature names not provided, create generic ones
    if feature_names is None:
        if isinstance(data, np.ndarray) and len(data.shape) >= 3:
            feature_names = [f"Feature_{i}" for i in range(data.shape[2])]
        else:
            feature_names = [f"Feature_{i}" for i in range(data.shape[1] if len(data.shape) > 1 else 1)]

    # For each cluster, identify key characteristics
    for cluster_id in range(n_clusters):
        # Get samples belonging to this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_samples = data[cluster_mask]
        other_samples = data[~cluster_mask]

        # Skip if not enough samples
        if len(cluster_samples) < 5 or len(other_samples) < 5:
            characteristics[cluster_id] = {"description": f"Too few samples in cluster {cluster_id}"}
            continue

        # Different analysis based on model type
        if model_type in ['rauto', 'imiss']:
            # For autoencoders, analyze latent space
            cluster_characteristics = analyze_autoencoder_cluster(model, cluster_samples, other_samples,
                                                                  feature_names, model_type)
        elif model_type == 'signature':
            # For signature transform, analyze signature components
            cluster_characteristics = analyze_signature_cluster(model, cluster_samples, other_samples,
                                                                feature_names)
        elif model_type == 'statfeat':
            # For statistical features, analyze statistical properties
            cluster_characteristics = analyze_statistical_cluster(model, cluster_samples, other_samples,
                                                                  feature_names)
        elif model_type == 'rgan':
            # For RGAN, analyze discriminator features
            cluster_characteristics = analyze_rgan_cluster(model, cluster_samples, other_samples,
                                                           feature_names)
        else:
            # Generic analysis for other models
            cluster_characteristics = analyze_generic_cluster(cluster_samples, other_samples, feature_names)

        # Store results
        characteristics[cluster_id] = cluster_characteristics

    return characteristics

def analyze_generic_cluster(cluster_samples, other_samples, feature_names):
    """Generic cluster analysis that works for any model type"""
    characteristics = {}

    # For 3D data (samples, time, features)
    if len(cluster_samples.shape) == 3:
        # 1. Calculate mean time series for each feature in this cluster
        mean_time_series = np.nanmean(cluster_samples, axis=0)  # [time, features]
        other_mean_time_series = np.nanmean(other_samples, axis=0)  # [time, features]

        # 2. Calculate overall mean for each feature
        feature_means = np.nanmean(mean_time_series, axis=0)  # [features]
        other_feature_means = np.nanmean(other_mean_time_series, axis=0)  # [features]

        # 3. Calculate feature trends (difference between end and start)
        feature_trends = mean_time_series[-1, :] - mean_time_series[0, :]  # [features]
        other_feature_trends = other_mean_time_series[-1, :] - other_mean_time_series[0, :]  # [features]

        # 4. Calculate feature volatility (standard deviation over time)
        feature_volatility = np.nanstd(mean_time_series, axis=0)  # [features]
        other_feature_volatility = np.nanstd(other_mean_time_series, axis=0)  # [features]

        # 5. Find the most distinguishing features (largest differences)
        mean_diffs = feature_means - other_feature_means
        trend_diffs = feature_trends - other_feature_trends
        vol_diffs = feature_volatility - other_feature_volatility

        # Sort indices by absolute difference
        mean_indices = np.argsort(np.abs(mean_diffs))[::-1]
        trend_indices = np.argsort(np.abs(trend_diffs))[::-1]
        vol_indices = np.argsort(np.abs(vol_diffs))[::-1]

        # 6. Build description of key distinguishing features
        distinguishing_by_mean = []
        for i in mean_indices[:5]:  # Top 5 by mean
            if i < len(feature_names):
                distinguishing_by_mean.append({
                    "feature": feature_names[i],
                    "cluster_mean": float(feature_means[i]),
                    "others_mean": float(other_feature_means[i]),
                    "difference": float(mean_diffs[i])
                })

        distinguishing_by_trend = []
        for i in trend_indices[:5]:  # Top 5 by trend
            if i < len(feature_names):
                distinguishing_by_trend.append({
                    "feature": feature_names[i],
                    "cluster_trend": float(feature_trends[i]),
                    "others_trend": float(other_feature_trends[i]),
                    "difference": float(trend_diffs[i])
                })

        distinguishing_by_volatility = []
        for i in vol_indices[:5]:  # Top 5 by volatility
            if i < len(feature_names):
                distinguishing_by_volatility.append({
                    "feature": feature_names[i],
                    "cluster_volatility": float(feature_volatility[i]),
                    "others_volatility": float(other_feature_volatility[i]),
                    "difference": float(vol_diffs[i])
                })

        characteristics["distinguishing_by_mean"] = distinguishing_by_mean
        characteristics["distinguishing_by_trend"] = distinguishing_by_trend
        characteristics["distinguishing_by_volatility"] = distinguishing_by_volatility

    return characteristics

def analyze_missingness_patterns(cluster_samples, other_samples, feature_names):
    """Analyze patterns of missing values in the data"""
    missingness_patterns = {}

    # Check if we have masks available (if not, can't analyze missingness)
    if isinstance(cluster_samples, tuple) and len(cluster_samples) == 2:
        sequences, masks = cluster_samples
    else:
        # No explicit masks available, check for NaN or zero values as proxy
        sequences = cluster_samples
        # Create masks where non-zero values are considered valid
        masks = (sequences != 0).astype(float)

    # Calculate missingness rate for each feature
    if len(masks.shape) == 3:  # [samples, time, features]
        # Calculate missingness rate per feature
        missingness_rate = 1 - np.mean(masks, axis=(0, 1))  # [features]

        # Do the same for other samples
        if isinstance(other_samples, tuple) and len(other_samples) == 2:
            other_sequences, other_masks = other_samples
        else:
            other_sequences = other_samples
            other_masks = (other_sequences != 0).astype(float)

        other_missingness_rate = 1 - np.mean(other_masks, axis=(0, 1))  # [features]

        # Calculate differences in missingness rates
        missingness_diffs = missingness_rate - other_missingness_rate

        # Find features with largest missingness differences
        miss_indices = np.argsort(np.abs(missingness_diffs))[::-1]

        # Build description of key missingness patterns
        distinguishing_by_missingness = []
        for i in miss_indices[:5]:  # Top 5 by missingness difference
            if i < len(feature_names):
                distinguishing_by_missingness.append({
                    "feature": feature_names[i],
                    "cluster_missingness": float(missingness_rate[i]),
                    "others_missingness": float(other_missingness_rate[i]),
                    "difference": float(missingness_diffs[i])
                })

        missingness_patterns["distinguishing_by_missingness"] = distinguishing_by_missingness

        # Also analyze missingness correlations
        # This identifies pairs of features that tend to be missing together
        missingness_corr = {}

        # Convert masks to binary (0=missing, 1=present)
        binary_masks = (masks > 0.5).astype(int)

        # Flatten across time
        flat_masks = binary_masks.reshape(binary_masks.shape[0], -1)  # [samples, time*features]

        # Calculate correlation matrix
        try:
            corr_matrix = np.corrcoef(flat_masks.T)

            # Find highly correlated missingness pairs
            n_features = sequences.shape[2]
            high_corr_pairs = []

            for i in range(n_features):
                for j in range(i + 1, n_features):
                    if i < len(feature_names) and j < len(feature_names):
                        # Calculate correlation between missingness patterns
                        i_missing = 1 - binary_masks[:, :, i].flatten()
                        j_missing = 1 - binary_masks[:, :, j].flatten()

                        # Only consider valid pairs (not all 0 or all 1)
                        if np.var(i_missing) > 0 and np.var(j_missing) > 0:
                            corr = np.corrcoef(i_missing, j_missing)[0, 1]

                            if abs(corr) > 0.5:  # Only report strong correlations
                                high_corr_pairs.append({
                                    "feature1": feature_names[i],
                                    "feature2": feature_names[j],
                                    "correlation": float(corr)
                                })

            # Sort by absolute correlation
            high_corr_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
            missingness_patterns["correlated_missingness"] = high_corr_pairs[:5]  # Top 5 pairs
        except:
            # In case correlation calculation fails
            missingness_patterns["correlated_missingness"] = []

    return missingness_patterns


def analyze_rgan_cluster(model, cluster_samples, other_samples, feature_names):
    """Analyze what the RGAN model has learned about a cluster"""
    characteristics = {}

    # For RGAN, we can analyze what the discriminator focuses on
    if hasattr(model, 'discriminator') and isinstance(model.discriminator, nn.Module):
        # Run samples through the discriminator to see what activates it
        if len(cluster_samples) > 0:
            # Take a subset of samples to analyze
            subset_size = min(20, len(cluster_samples))
            subset_indices = np.random.choice(len(cluster_samples), subset_size, replace=False)
            subset = cluster_samples[subset_indices]

            # Add analysis of generated samples that are similar to this cluster
            characteristics["generated_analysis"] = "This cluster's patterns could be generated by the RGAN model."

    # Generic feature analysis
    if len(cluster_samples.shape) == 3:  # [samples, time, features]
        characteristics.update(analyze_generic_cluster(cluster_samples, other_samples, feature_names))

    # If model has feature importances, add them
    if hasattr(model, 'feature_importances') and model.feature_importances is not None:
        feature_importances = model.feature_importances

        # Get the most important features
        top_features_idx = np.argsort(feature_importances)[::-1][:5]  # Top 5 features
        top_features = []

        for idx in top_features_idx:
            if idx < len(feature_names):
                importance = feature_importances[idx]
                top_features.append({
                    "feature": feature_names[idx],
                    "importance": float(importance)
                })

        characteristics["important_features"] = top_features

    return characteristics


def analyze_autoencoder_cluster(model, cluster_samples, other_samples, feature_names, model_type):
    """Analyze what an autoencoder-based model has learned about a cluster"""
    characteristics = {}

    # Convert samples to tensors if needed
    if not isinstance(cluster_samples, torch.Tensor):
        cluster_tensor = torch.FloatTensor(cluster_samples)
    else:
        cluster_tensor = cluster_samples

    # Extract original features (not padded sequences)
    if len(cluster_samples.shape) == 3:  # [samples, time, features]
        # Calculate mean values across time for each feature
        cluster_feature_means = np.mean(cluster_samples, axis=1)  # [samples, features]
        other_feature_means = np.mean(other_samples, axis=1)  # [samples, features]

        # Perform statistical tests to find distinguishing features
        p_values = []
        effect_sizes = []

        for feat_idx in range(cluster_feature_means.shape[1]):
            # Get feature values for this cluster and others
            cluster_vals = cluster_feature_means[:, feat_idx]
            other_vals = other_feature_means[:, feat_idx]

            # T-test to see if the means are significantly different
            try:
                t_stat, p_val = ttest_ind(cluster_vals, other_vals, equal_var=False)
                # Calculate effect size (Cohen's d)
                mean_diff = np.mean(cluster_vals) - np.mean(other_vals)
                pooled_std = np.sqrt(((len(cluster_vals) - 1) * np.std(cluster_vals, ddof=1) ** 2 +
                                      (len(other_vals) - 1) * np.std(other_vals, ddof=1) ** 2) /
                                     (len(cluster_vals) + len(other_vals) - 2))
                effect_size = mean_diff / pooled_std if pooled_std != 0 else 0
            except:
                p_val = 1.0
                effect_size = 0

            p_values.append(p_val)
            effect_sizes.append(effect_size)

        # Rank features by effect size and p-value
        feature_importance = [(i, es, pv) for i, (es, pv) in
                              enumerate(zip(effect_sizes, p_values))]
        feature_importance.sort(key=lambda x: (x[1], -x[2]),
                                reverse=True)  # Sort by effect size (desc) then p-value (asc)

        # Extract top 5 distinguishing features
        top_features = feature_importance[:5]
        characteristics["distinguishing_features"] = [
            {
                "feature": feature_names[feat_idx],
                "effect_size": effect,
                "p_value": p_val,
                "cluster_mean": float(np.mean(cluster_feature_means[:, feat_idx])),
                "others_mean": float(np.mean(other_feature_means[:, feat_idx]))
            }
            for feat_idx, effect, p_val in top_features if effect != 0 and p_val < 0.05
        ]

        # Analyze temporal patterns if applicable
        if model_type == 'imiss':
            # For InformativeMissingness, also analyze missingness patterns
            characteristics["missingness_patterns"] = analyze_missingness_patterns(
                cluster_samples, other_samples, feature_names)

    return characteristics


def analyze_signature_cluster(model, cluster_samples, other_samples, feature_names):
    """Analyze what the signature transform model has learned about a cluster"""
    characteristics = {}

    # Get signature terms for each feature
    if hasattr(model, 'signature_transform'):
        # Extract meaningful signature components
        level_names = {
            'level_0': 'Mean',
            'level_1': 'Trend',
            'level_2': 'Quadratic',
            'level_3': 'Volatility'
        }

        # Calculate signature for a representative sample
        if len(cluster_samples) > 0:
            # Take a representative sample
            sample = cluster_samples[0]
            # Compute signature
            signature = model.signature_transform.compute_path_signature(sample)

            # Map signature values to features where possible
            if isinstance(signature, dict):
                sig_features = {}

                for level, values in signature.items():
                    if level in level_names:
                        level_type = level_names[level]
                        if level == 'level_0':  # Mean - map directly to features
                            for i, val in enumerate(values):
                                if i < len(feature_names):
                                    sig_features[f"{feature_names[i]} ({level_type})"] = val
                        elif level == 'level_1':  # Trend
                            for i, val in enumerate(values):
                                if i < len(feature_names):
                                    sig_features[f"{feature_names[i]} Trend"] = val

                characteristics["signature_analysis"] = sig_features

    # Generic feature analysis as fallback
    if len(cluster_samples.shape) == 3:  # [samples, time, features]
        characteristics.update(analyze_generic_cluster(cluster_samples, other_samples, feature_names))

    return characteristics


def analyze_statistical_cluster(model, cluster_samples, other_samples, feature_names):
    """Analyze what the statistical feature model has learned about a cluster"""
    characteristics = {}

    # If the model has feature importance, extract it
    if hasattr(model, 'get_feature_importance'):
        importance_df = model.get_feature_importance()
        if importance_df is not None:
            # Extract top 10 important features
            top_features = importance_df.head(10).to_dict('records')
            characteristics["important_statistical_features"] = top_features

    # Calculate statistical measures for each feature
    if len(cluster_samples.shape) == 3:  # [samples, time, features]
        # Calculate statistics across time for each feature
        stats_map = {}

        for feat_idx in range(cluster_samples.shape[2]):
            if feat_idx < len(feature_names):
                feat_name = feature_names[feat_idx]

                # Extract values for this feature across all samples and time
                cluster_vals = cluster_samples[:, :, feat_idx].flatten()
                cluster_vals = cluster_vals[~np.isnan(cluster_vals)]  # Remove NaNs

                # Calculate statistics
                if len(cluster_vals) > 0:
                    mean = np.mean(cluster_vals)
                    std = np.std(cluster_vals)
                    skew = stats.skew(cluster_vals) if len(cluster_vals) > 2 else 0
                    kurt = stats.kurtosis(cluster_vals) if len(cluster_vals) > 2 else 0
                    p5 = np.percentile(cluster_vals, 5)
                    p95 = np.percentile(cluster_vals, 95)

                    stats_map[feat_name] = {
                        "mean": float(mean),
                        "std": float(std),
                        "skewness": float(skew),
                        "kurtosis": float(kurt),
                        "p5": float(p5),
                        "p95": float(p95)
                    }

                characteristics["feature_statistics"] = stats_map

                return characteristics

def analyze_rgan_cluster(model, cluster_samples, other_samples, feature_names):
    """Analyze what the RGAN model has learned about a cluster"""
    characteristics = {}

    # For RGAN, we can analyze what the discriminator focuses on
    if hasattr(model, 'discriminator') and isinstance(model.discriminator, nn.Module):
        # Run samples through the discriminator to see what activates it
        if len(cluster_samples) > 0:
            # Take a subset of samples to analyze
            subset_size = min(20, len(cluster_samples))
            subset_indices = np.random.choice(len(cluster_samples), subset_size, replace=False)
            subset = cluster_samples[subset_indices]

            # Add analysis of generated samples that are similar to this cluster
            characteristics[
                "generated_analysis"] = "This cluster's patterns could be generated by the RGAN model."

    # Generic feature analysis
    if len(cluster_samples.shape) == 3:  # [samples, time, features]
        characteristics.update(analyze_generic_cluster(cluster_samples, other_samples, feature_names))

    # If model has feature importances, add them
    if hasattr(model, 'feature_importances') and model.feature_importances is not None:
        feature_importances = model.feature_importances

        # Get the most important features
        top_features_idx = np.argsort(feature_importances)[::-1][:5]  # Top 5 features
        top_features = []

        for idx in top_features_idx:
            if idx < len(feature_names):
                importance = feature_importances[idx]
                top_features.append({
                    "feature": feature_names[idx],
                    "importance": float(importance)
                })

        characteristics["important_features"] = top_features

    return characteristics

def analyze_generic_cluster(cluster_samples, other_samples, feature_names):
    """Generic cluster analysis that works for any model type"""
    characteristics = {}

    # For 3D data (samples, time, features)
    if len(cluster_samples.shape) == 3:
        # 1. Calculate mean time series for each feature in this cluster
        mean_time_series = np.nanmean(cluster_samples, axis=0)  # [time, features]
        other_mean_time_series = np.nanmean(other_samples, axis=0)  # [time, features]

        # 2. Calculate overall mean for each feature
        feature_means = np.nanmean(mean_time_series, axis=0)  # [features]
        other_feature_means = np.nanmean(other_mean_time_series, axis=0)  # [features]

        # 3. Calculate feature trends (difference between end and start)
        feature_trends = mean_time_series[-1, :] - mean_time_series[0, :]  # [features]
        other_feature_trends = other_mean_time_series[-1, :] - other_mean_time_series[0, :]  # [features]

        # 4. Calculate feature volatility (standard deviation over time)
        feature_volatility = np.nanstd(mean_time_series, axis=0)  # [features]
        other_feature_volatility = np.nanstd(other_mean_time_series, axis=0)  # [features]

        # 5. Find the most distinguishing features (largest differences)
        mean_diffs = feature_means - other_feature_means
        trend_diffs = feature_trends - other_feature_trends
        vol_diffs = feature_volatility - other_feature_volatility

        # Sort indices by absolute difference
        mean_indices = np.argsort(np.abs(mean_diffs))[::-1]
        trend_indices = np.argsort(np.abs(trend_diffs))[::-1]
        vol_indices = np.argsort(np.abs(vol_diffs))[::-1]

        # 6. Build description of key distinguishing features
        distinguishing_by_mean = []
        for i in mean_indices[:5]:  # Top 5 by mean
            if i < len(feature_names):
                distinguishing_by_mean.append({
                    "feature": feature_names[i],
                    "cluster_mean": float(feature_means[i]),
                    "others_mean": float(other_feature_means[i]),
                    "difference": float(mean_diffs[i])
                })

        distinguishing_by_trend = []
        for i in trend_indices[:5]:  # Top 5 by trend
            if i < len(feature_names):
                distinguishing_by_trend.append({
                    "feature": feature_names[i],
                    "cluster_trend": float(feature_trends[i]),
                    "others_trend": float(other_feature_trends[i]),
                    "difference": float(trend_diffs[i])
                })

        distinguishing_by_volatility = []
        for i in vol_indices[:5]:  # Top 5 by volatility
            if i < len(feature_names):
                distinguishing_by_volatility.append({
                    "feature": feature_names[i],
                    "cluster_volatility": float(feature_volatility[i]),
                    "others_volatility": float(other_feature_volatility[i]),
                    "difference": float(vol_diffs[i])
                })

        characteristics["distinguishing_by_mean"] = distinguishing_by_mean
        characteristics["distinguishing_by_trend"] = distinguishing_by_trend
        characteristics["distinguishing_by_volatility"] = distinguishing_by_volatility

    return characteristics

def analyze_missingness_patterns(cluster_samples, other_samples, feature_names):
    """Analyze patterns of missing values in the data"""
    missingness_patterns = {}

    # Check if we have masks available (if not, can't analyze missingness)
    if isinstance(cluster_samples, tuple) and len(cluster_samples) == 2:
        sequences, masks = cluster_samples
    else:
        # No explicit masks available, check for NaN or zero values as proxy
        sequences = cluster_samples
        # Create masks where non-zero values are considered valid
        masks = (sequences != 0).astype(float)

    # Calculate missingness rate for each feature
    if len(masks.shape) == 3:  # [samples, time, features]
        # Calculate missingness rate per feature
        missingness_rate = 1 - np.mean(masks, axis=(0, 1))  # [features]

        # Do the same for other samples
        if isinstance(other_samples, tuple) and len(other_samples) == 2:
            other_sequences, other_masks = other_samples
        else:
            other_sequences = other_samples
            other_masks = (other_sequences != 0).astype(float)

        other_missingness_rate = 1 - np.mean(other_masks, axis=(0, 1))  # [features]

        # Calculate differences in missingness rates
        missingness_diffs = missingness_rate - other_missingness_rate

        # Find features with largest missingness differences
        miss_indices = np.argsort(np.abs(missingness_diffs))[::-1]

        # Build description of key missingness patterns
        distinguishing_by_missingness = []
        for i in miss_indices[:5]:  # Top 5 by missingness difference
            if i < len(feature_names):
                distinguishing_by_missingness.append({
                    "feature": feature_names[i],
                    "cluster_missingness": float(missingness_rate[i]),
                    "others_missingness": float(other_missingness_rate[i]),
                    "difference": float(missingness_diffs[i])
                })

        missingness_patterns["distinguishing_by_missingness"] = distinguishing_by_missingness

        # Also analyze missingness correlations
        # This identifies pairs of features that tend to be missing together
        missingness_corr = {}

        # Convert masks to binary (0=missing, 1=present)
        binary_masks = (masks > 0.5).astype(int)

        # Flatten across time
        flat_masks = binary_masks.reshape(binary_masks.shape[0], -1)  # [samples, time*features]

        # Calculate correlation matrix
        try:
            corr_matrix = np.corrcoef(flat_masks.T)

            # Find highly correlated missingness pairs
            n_features = sequences.shape[2]
            high_corr_pairs = []

            for i in range(n_features):
                for j in range(i + 1, n_features):
                    if i < len(feature_names) and j < len(feature_names):
                        # Calculate correlation between missingness patterns
                        i_missing = 1 - binary_masks[:, :, i].flatten()
                        j_missing = 1 - binary_masks[:, :, j].flatten()

                        # Only consider valid pairs (not all 0 or all 1)
                        if np.var(i_missing) > 0 and np.var(j_missing) > 0:
                            corr = np.corrcoef(i_missing, j_missing)[0, 1]

                            if abs(corr) > 0.5:  # Only report strong correlations
                                high_corr_pairs.append({
                                    "feature1": feature_names[i],
                                    "feature2": feature_names[j],
                                    "correlation": float(corr)
                                })

            # Sort by absolute correlation
            high_corr_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
            missingness_patterns["correlated_missingness"] = high_corr_pairs[:5]  # Top 5 pairs
        except:
            # In case correlation calculation fails
            missingness_patterns["correlated_missingness"] = []

    return missingness_patterns

def generate_cluster_description(characteristics, cluster_id):
    """Generate human-readable description of cluster characteristics"""
    description = f"Cluster {cluster_id} characteristics:\n\n"

    # Add distinguishing features by mean
    if "distinguishing_by_mean" in characteristics and characteristics["distinguishing_by_mean"]:
        description += "Key features (by mean value):\n"
        for feat in characteristics["distinguishing_by_mean"][:3]:  # Top 3 for readability
            direction = "higher" if feat["difference"] > 0 else "lower"
            description += f"- {feat['feature']}: {direction} than other clusters "
            description += f"({feat['cluster_mean']:.2f} vs {feat['others_mean']:.2f})\n"
        description += "\n"

    # Add distinguishing features by trend
    if "distinguishing_by_trend" in characteristics and characteristics["distinguishing_by_trend"]:
        description += "Key features (by trend):\n"
        for feat in characteristics["distinguishing_by_trend"][:3]:  # Top 3 for readability
            trend_desc = "increasing" if feat["cluster_trend"] > 0 else "decreasing"
            other_trend_desc = "increasing" if feat["others_trend"] > 0 else "decreasing"
            description += f"- {feat['feature']}: {trend_desc} trend "
            if np.sign(feat["cluster_trend"]) != np.sign(feat["others_trend"]):
                description += f"(unlike other clusters which are {other_trend_desc})\n"
            else:
                magnitude = "more rapidly" if abs(feat["cluster_trend"]) > abs(
                    feat["others_trend"]) else "more slowly"
                description += f"({magnitude} than other clusters)\n"
        description += "\n"

    # Add distinguishing features by volatility
    if "distinguishing_by_volatility" in characteristics and characteristics[
        "distinguishing_by_volatility"]:
        description += "Key features (by volatility):\n"
        for feat in characteristics["distinguishing_by_volatility"][:3]:  # Top 3 for readability
            direction = "more variable" if feat["difference"] > 0 else "more stable"
            description += f"- {feat['feature']}: {direction} than other clusters "
            description += f"(volatility: {feat['cluster_volatility']:.2f} vs {feat['others_volatility']:.2f})\n"
        description += "\n"

    # Add missingness patterns if available
    if "missingness_patterns" in characteristics:
        missing_patterns = characteristics["missingness_patterns"]
        if "distinguishing_by_missingness" in missing_patterns and missing_patterns[
            "distinguishing_by_missingness"]:
            description += "Key missingness patterns:\n"
            for feat in missing_patterns["distinguishing_by_missingness"][:3]:  # Top 3 for readability
                direction = "more frequently missing" if feat[
                                                             "difference"] > 0 else "less frequently missing"
                description += f"- {feat['feature']}: {direction} than in other clusters "
                description += f"({feat['cluster_missingness'] * 100:.1f}% vs {feat['others_missingness'] * 100:.1f}%)\n"
            description += "\n"

        if "correlated_missingness" in missing_patterns and missing_patterns["correlated_missingness"]:
            description += "Correlated missingness patterns:\n"
            for pair in missing_patterns["correlated_missingness"][:2]:  # Top 2 for readability
                if pair["correlation"] > 0:
                    relation = "tend to be missing together"
                else:
                    relation = "tend to have opposite missingness patterns"
                description += f"- {pair['feature1']} and {pair['feature2']} {relation} "
                description += f"(corr: {pair['correlation']:.2f})\n"
            description += "\n"

    # Add signature analysis if available
    if "signature_analysis" in characteristics and characteristics["signature_analysis"]:
        description += "Signature analysis:\n"
        sig_features = characteristics["signature_analysis"]
        # Sort by absolute value
        sorted_sigs = sorted(sig_features.items(), key=lambda x: abs(x[1]), reverse=True)
        for feat, val in sorted_sigs[:3]:  # Top 3 for readability
            direction = "high" if val > 0 else "low"
            description += f"- {feat}: {direction} ({val:.2f})\n"
        description += "\n"

    # Add statistical features if available
    if "important_statistical_features" in characteristics and characteristics[
        "important_statistical_features"]:
        description += "Important statistical features:\n"
        for feat in characteristics["important_statistical_features"][:3]:  # Top 3 for readability
            description += f"- {feat['Feature']}: importance score {feat['Importance']:.3f}\n"
        description += "\n"

    # Add feature statistics if available
    if "feature_statistics" in characteristics and characteristics["feature_statistics"]:
        description += "Key feature statistics:\n"
        # Sort features by standard deviation
        sorted_stats = sorted(characteristics["feature_statistics"].items(),
                              key=lambda x: x[1]['std'], reverse=True)
        for feat, stats in sorted_stats[:3]:  # Top 3 for readability
            description += f"- {feat}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, "
            description += f"skewness={stats['skewness']:.2f}, 5-95th percentile=[{stats['p5']:.2f}, {stats['p95']:.2f}]\n"
        description += "\n"

    # Add RGAN-specific analysis if available
    if "generated_analysis" in characteristics:
        description += characteristics["generated_analysis"] + "\n\n"

    # Add important features from RGAN if available
    if "important_features" in characteristics and characteristics["important_features"]:
        description += "Important features for this cluster:\n"
        for feat in characteristics["important_features"]:
            description += f"- {feat['feature']}: {feat['importance']:.4f}\n"
        description += "\n"

    # Generate clinical interpretation suggestions
    description += "Potential clinical interpretation:\n"
    description += "This cluster may represent a distinct patient subgroup with the above characteristics. "
    description += "Consider examining clinical outcomes and treatment responses for patients in this cluster."

    return description

def visualize_cluster_characteristics(data, cluster_labels, feature_names=None, save_path=None):
    """
    Create visualizations for the characteristics of each cluster

    Args:
        data: The original data (padded sequences) [samples, time, features]
        cluster_labels: The cluster assignments
        feature_names: Names of the features (if None, will use indices)
        save_path: Path to save the visualizations
    """
    n_clusters = len(np.unique(cluster_labels))

    # If feature names not provided, create generic ones
    if feature_names is None:
        if isinstance(data, np.ndarray) and len(data.shape) >= 3:
            feature_names = [f"Feature_{i}" for i in range(data.shape[2])]
        else:
            feature_names = [f"Feature_{i}" for i in range(data.shape[1] if len(data.shape) > 1 else 1)]

    # Create save directory if needed
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    # 1. Plot cluster sizes
    plt.figure(figsize=(12, 6))
    cluster_sizes = np.bincount(cluster_labels)
    plt.bar(range(n_clusters), cluster_sizes)
    plt.title('Cluster Sizes')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Samples')
    plt.xticks(range(n_clusters))
    plt.grid(True, axis='y')
    if save_path:
        plt.savefig(f"{save_path}/cluster_sizes.png")
    plt.close()

    # 2. For 3D data (samples, time, features), visualize time series patterns per cluster
    if len(data.shape) == 3:
        # Select top features by variance
        feature_variance = np.var(data, axis=(0, 1))
        top_feature_indices = np.argsort(feature_variance)[-min(5, len(feature_variance)):]

        # For each top feature, plot mean time series per cluster
        for feat_idx in top_feature_indices:
            if feat_idx < len(feature_names):
                plt.figure(figsize=(14, 8))

                # Create custom color map for clusters
                colors = plt.cm.rainbow(np.linspace(0, 1, n_clusters))

                for cluster_id in range(n_clusters):
                    # Get samples in this cluster
                    cluster_mask = cluster_labels == cluster_id
                    cluster_samples = data[cluster_mask]

                    if len(cluster_samples) > 0:
                        # Calculate mean time series for this feature
                        mean_series = np.nanmean(cluster_samples[:, :, feat_idx], axis=0)
                        # Calculate standard deviation for confidence interval
                        std_series = np.nanstd(cluster_samples[:, :, feat_idx], axis=0)

                        # Plot mean with confidence interval
                        time_points = range(len(mean_series))
                        plt.plot(time_points, mean_series, label=f'Cluster {cluster_id}',
                                 linewidth=2, color=colors[cluster_id])
                        plt.fill_between(time_points, mean_series - std_series, mean_series + std_series,
                                         alpha=0.2, color=colors[cluster_id])

                plt.title(f'Time Series for {feature_names[feat_idx]} by Cluster')
                plt.xlabel('Time Step')
                plt.ylabel('Value')
                plt.legend()
                plt.grid(True)
                if save_path:
                    plt.savefig(
                        f"{save_path}/feature_{feat_idx}_{feature_names[feat_idx].replace(' ', '_')}_by_cluster.png")
                plt.close()

        # 3. Visualize missingness patterns per cluster
        # Create masks where non-zero and non-NaN values are considered valid
        masks = (~np.isnan(data) & (data != 0)).astype(float)

        # Calculate missingness rate per feature and cluster
        missingness_by_cluster = []
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_samples = data[cluster_mask]
            cluster_masks = masks[cluster_mask]

            if len(cluster_samples) > 0:
                # Calculate missingness rate per feature
                missingness_rate = 1 - np.mean(cluster_masks, axis=(0, 1))  # [features]
                missingness_by_cluster.append(missingness_rate)

        if missingness_by_cluster:
            # Convert to array
            missingness_by_cluster = np.array(missingness_by_cluster)

            # Select top features by missingness variance
            miss_variance = np.var(missingness_by_cluster, axis=0)
            top_miss_indices = np.argsort(miss_variance)[-min(10, len(miss_variance)):]

            # Create heatmap of missingness patterns
            plt.figure(figsize=(15, 8))
            selected_names = [feature_names[i] if i < len(feature_names) else f"Feature_{i}"
                              for i in top_miss_indices]
            selected_missingness = missingness_by_cluster[:, top_miss_indices]

            sns.heatmap(selected_missingness, annot=True, fmt=".2f", cmap="YlOrRd",
                        xticklabels=selected_names, yticklabels=[f"Cluster {i}" for i in range(n_clusters)])
            plt.title('Missingness Rates by Cluster and Feature')
            plt.ylabel('Cluster')
            plt.xlabel('Feature')
            plt.tight_layout()
            if save_path:
                plt.savefig(f"{save_path}/missingness_heatmap.png")
            plt.close()

    # 4. Create a detailed version of the PCA visualization with cluster descriptions
    if len(data.shape) > 1:
        # Use the first two dimensions if data already has 2D
        if data.shape[1] == 2:
            pca_result = data
        else:
            # Flatten the data if it's 3D
            if len(data.shape) == 3:
                flattened_data = data.reshape(data.shape[0], -1)
            else:
                flattened_data = data

            # Apply PCA
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(flattened_data)

        # Create a new figure with increased size for detailed visualization
        plt.figure(figsize=(16, 12))

        # Set up colors for clusters
        colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

        # Plot each cluster with a different color
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            plt.scatter(pca_result[cluster_mask, 0], pca_result[cluster_mask, 1],
                        c=[colors[cluster_id]], label=f'Cluster {cluster_id}', alpha=0.7)

            # Calculate centroid for this cluster
            if np.sum(cluster_mask) > 0:
                centroid = np.mean(pca_result[cluster_mask], axis=0)

                # Add cluster ID label at centroid
                plt.text(centroid[0], centroid[1], str(cluster_id),
                         fontsize=15, fontweight='bold',
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='black',
                                   boxstyle='round,pad=0.5'))

                # Calculate the top 3 features for this cluster
                if len(data.shape) == 3:
                    cluster_data = data[cluster_mask]
                    other_data = data[~cluster_mask]

                    # Calculate feature means for this cluster and others
                    cluster_means = np.nanmean(cluster_data, axis=(0, 1))
                    other_means = np.nanmean(other_data, axis=(0, 1))

                    # Calculate differences
                    diff = cluster_means - other_means

                    # Find top features (largest absolute differences)
                    top_indices = np.argsort(np.abs(diff))[-3:][::-1]

                    # Add annotation with top features
                    annotation = "\n".join([
                        f"{feature_names[i] if i < len(feature_names) else f'Feature_{i}'}: "
                        f"{'↑' if diff[i] > 0 else '↓'}{abs(diff[i]):.2f}"
                        for i in top_indices
                    ])

                    # Calculate position for annotation (offset from centroid)
                    offset_x = (np.random.random() - 0.5) * 2  # Random offset to avoid overlap
                    offset_y = (np.random.random() - 0.5) * 2
                    scale = 2.0  # Scale factor for offset

                    plt.annotate(annotation,
                                 xy=(centroid[0], centroid[1]),
                                 xytext=(centroid[0] + offset_x * scale, centroid[1] + offset_y * scale),
                                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7),
                                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3"))

        plt.title('PCA Visualization with Cluster Characteristics')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        if save_path:
            plt.savefig(f"{save_path}/detailed_pca_visualization.png")
        plt.close()

    # 5. Create a cluster comparison heatmap for key features
    if len(data.shape) == 3:
        # Calculate mean values for each feature by cluster
        cluster_feature_means = np.zeros((n_clusters, data.shape[2]))

        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_data = data[cluster_mask]

            if len(cluster_data) > 0:
                # Calculate mean across samples and time
                cluster_feature_means[cluster_id] = np.nanmean(cluster_data, axis=(0, 1))

        # Select top differentiating features
        # Calculate variance of feature means across clusters
        feature_variance = np.var(cluster_feature_means, axis=0)
        top_indices = np.argsort(feature_variance)[-min(15, len(feature_variance)):]

        # Create heatmap
        plt.figure(figsize=(16, 10))
        selected_names = [feature_names[i] if i < len(feature_names) else f"Feature_{i}"
                          for i in top_indices]
        selected_means = cluster_feature_means[:, top_indices]

        # Normalize data for better visualization
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        normalized_means = scaler.fit_transform(selected_means)

        # Create heatmap with detailed annotations
        sns.heatmap(normalized_means,
                    annot=np.round(selected_means, 2),  # Show actual values
                    fmt=".2f",
                    cmap="coolwarm",
                    xticklabels=selected_names,
                    yticklabels=[f"Cluster {i}" for i in range(n_clusters)])

        plt.title('Feature Comparison Across Clusters (Normalized)')
        plt.ylabel('Cluster')
        plt.xlabel('Feature')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}/cluster_feature_comparison.png")
        plt.close()

# ---------------------- Add cluster interpretation to existing clustering functions ----------------------

# Add cluster interpretation to the InformativeMissingness model
def cluster_latent_representations_enhanced(model, batches, device, n_clusters=5, feature_names=None):
    """
    Enhanced version of cluster_latent_representations that includes cluster interpretation
    """
    # Call the original function to get clusters
    clusters, centers, latents = cluster_latent_representations(model, batches, device, n_clusters)

    # Extract all sequences and masks for interpretation
    all_sequences = []
    all_masks = []
    for batch_sequences, batch_masks in batches:
        all_sequences.extend(batch_sequences)
        all_masks.extend(batch_masks)

    all_sequences = np.array(all_sequences)
    all_masks = np.array(all_masks)

    # Analyze cluster characteristics
    characteristics = analyze_cluster_characteristics(model, all_sequences, clusters,
                                                      feature_names=feature_names, model_type='imiss')

    # Generate human-readable descriptions
    descriptions = {}
    for cluster_id in range(n_clusters):
        if cluster_id in characteristics:
            descriptions[cluster_id] = generate_cluster_description(
                characteristics[cluster_id], cluster_id)

    # Create visualizations
    save_path = 'cluster_visualizations/imiss'
    os.makedirs(save_path, exist_ok=True)
    visualize_cluster_characteristics(all_sequences, clusters, feature_names, save_path)

    # Print descriptions to console
    for cluster_id, description in descriptions.items():
        print("\n" + "=" * 80)
        print(description)
        print("=" * 80)

    # Save descriptions to file
    with open(f'{save_path}/cluster_descriptions.txt', 'w') as f:
        for cluster_id, description in descriptions.items():
            f.write("\n" + "=" * 80 + "\n")
            f.write(description)
            f.write("\n" + "=" * 80 + "\n")

    return clusters, centers, latents, characteristics, descriptions

# Add enhanced versions for signature transform clustering
def use_signature_transform_clustering_enhanced(batches, n_clusters=5, truncation_level=3,
                                                feature_names=None):
    # Call the original function
    model, cluster_labels = use_signature_transform_clustering(batches, n_clusters, truncation_level)

    # Extract all sequences and masks for interpretation
    all_sequences = []
    all_masks = []
    for batch_sequences, batch_masks in batches:
        all_sequences.extend(batch_sequences)
        all_masks.extend(batch_masks)

    all_sequences = np.array(all_sequences)
    all_masks = np.array(all_masks)

    # Analyze cluster characteristics
    characteristics = analyze_cluster_characteristics(model, all_sequences, cluster_labels,
                                                      feature_names=feature_names, model_type='signature')

    # Generate human-readable descriptions
    descriptions = {}
    for cluster_id in range(n_clusters):
        if cluster_id in characteristics:
            descriptions[cluster_id] = generate_cluster_description(
                characteristics[cluster_id], cluster_id)

    # Create visualizations
    save_path = 'cluster_visualizations/signature'
    os.makedirs(save_path, exist_ok=True)
    visualize_cluster_characteristics(all_sequences, cluster_labels, feature_names, save_path)

    # Print descriptions to console
    for cluster_id, description in descriptions.items():
        print("\n" + "=" * 80)
        print(description)
        print("=" * 80)

    # Save descriptions to file
    with open(f'{save_path}/cluster_descriptions.txt', 'w') as f:
        for cluster_id, description in descriptions.items():
            f.write("\n" + "=" * 80 + "\n")
            f.write(description)
            f.write("\n" + "=" * 80 + "\n")

    return model, cluster_labels, characteristics, descriptions

# Add enhanced version of statistical feature representation
def use_statistical_feature_representation_enhanced(batches, n_clusters=5, use_pca=True, n_components=10,
                                                    feature_names=None):
    # Call the original function
    model, cluster_labels = use_statistical_feature_representation(batches, n_clusters, use_pca,
                                                                   n_components)

    # Extract all sequences and masks for interpretation
    all_sequences = []
    all_masks = []
    for batch_sequences, batch_masks in batches:
        all_sequences.extend(batch_sequences)
        all_masks.extend(batch_masks)

    all_sequences = np.array(all_sequences)
    all_masks = np.array(all_masks)

    # Analyze cluster characteristics
    characteristics = analyze_cluster_characteristics(model, all_sequences, cluster_labels,
                                                      feature_names=feature_names, model_type='statfeat')

    # Generate human-readable descriptions
    descriptions = {}
    for cluster_id in range(n_clusters):
        if cluster_id in characteristics:
            descriptions[cluster_id] = generate_cluster_description(
                characteristics[cluster_id], cluster_id)

    # Create visualizations
    save_path = 'cluster_visualizations/statfeat'
    os.makedirs(save_path, exist_ok=True)
    visualize_cluster_characteristics(all_sequences, cluster_labels, feature_names, save_path)

    # Print descriptions to console
    for cluster_id, description in descriptions.items():
        print("\n" + "=" * 80)
        print(description)
        print("=" * 80)

    # Save descriptions to file
    with open(f'{save_path}/cluster_descriptions.txt', 'w') as f:
        for cluster_id, description in descriptions.items():
            f.write("\n" + "=" * 80 + "\n")
            f.write(description)
            f.write("\n" + "=" * 80 + "\n")

    return model, cluster_labels, characteristics, descriptions

# Add enhanced version for RGAN clustering
def use_rgan_enhanced(batches, input_dim, device, epochs=30, sample_interval=5, feature_names=None):
    # Call the original function
    rgan_model, rgan_latents = use_rgan(batches, input_dim, device, epochs, sample_interval)

    # Perform clustering on latent representations
    if len(rgan_latents) > 10:
        from sklearn.cluster import KMeans

        n_clusters = min(5, len(rgan_latents) // 2)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        rgan_clusters = kmeans.fit_predict(rgan_latents)

        # Extract all sequences and masks for interpretation
        all_sequences = []
        all_masks = []
        for batch_sequences, batch_masks in batches:
            all_sequences.extend(batch_sequences)
            all_masks.extend(batch_masks)

        all_sequences = np.array(all_sequences)
        all_masks = np.array(all_masks)

        # Analyze cluster characteristics
        characteristics = analyze_cluster_characteristics(rgan_model, all_sequences, rgan_clusters,
                                                          feature_names=feature_names, model_type='rgan')

        # Generate human-readable descriptions
        descriptions = {}
        for cluster_id in range(n_clusters):
            if cluster_id in characteristics:
                descriptions[cluster_id] = generate_cluster_description(
                    characteristics[cluster_id], cluster_id)

        # Create visualizations
        save_path = 'cluster_visualizations/rgan'
        os.makedirs(save_path, exist_ok=True)
        visualize_cluster_characteristics(all_sequences, rgan_clusters, feature_names, save_path)

        # Print descriptions to console
        for cluster_id, description in descriptions.items():
            print("\n" + "=" * 80)
            print(description)
            print("=" * 80)

        # Save descriptions to file
        with open(f'{save_path}/cluster_descriptions.txt', 'w') as f:
            for cluster_id, description in descriptions.items():
                f.write("\n" + "=" * 80 + "\n")
                f.write(description)
                f.write("\n" + "=" * 80 + "\n")

        return rgan_model, rgan_latents, rgan_clusters, characteristics, descriptions

    return rgan_model, rgan_latents

def run_benchmark(ts_tcc_model, autoencoder, imiss_model, signature_model, statfeat_model, rgan_model,
                  X_train_pad, X_test_pad, y_train, y_test,
                  mask_train, mask_test, patient_ids_train, patient_ids_test,
                  tstcc_train_loader, rauto_train_loader, rauto_batches, device):
    """
    Runs a comprehensive benchmark for all six models
    """
    print("\n==== Starting Benchmark Process ====")

    # Import necessary metrics
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.linear_model import LogisticRegression

    # Extract all sequences and masks for interpretation
    all_sequences = []
    all_masks = []
    for batch_sequences, batch_masks in rauto_batches:
        all_sequences.extend(batch_sequences)
        all_masks.extend(batch_masks)

    all_sequences = np.array(all_sequences)
    all_masks = np.array(all_masks)

    # Create directories for results
    os.makedirs('benchmark_results', exist_ok=True)
    os.makedirs('benchmark_interpretations', exist_ok=True)

    # Dictionary to store benchmark results
    benchmark_results = {
        'clustering_metrics': {},
        'downstream_classification': {},
        'feature_importance': {},
        'computational_efficiency': {}
    }

    # Feature names for interpretation
    feature_names = [FEATURE_NAMES.get(i, f"Feature_{i}") for i in range(all_sequences.shape[2])]

    # ================ 1. TS-TCC Model ================
    print("\nAnalyzing TS-TCC model...")
    start_time = time.time()

    # Extract features and do clustering on TS-TCC representations
    ts_tcc_features = extract_unsupervised_features(ts_tcc_model, tstcc_train_loader, device, 'tstcc')
    kmeans_tstcc = KMeans(n_clusters=5, random_state=42)
    tstcc_clusters = kmeans_tstcc.fit_predict(ts_tcc_features)

    # Calculate processing time
    tstcc_time = time.time() - start_time

    # Check silhouette score and other metrics
    try:
        tstcc_silhouette = silhouette_score(ts_tcc_features, tstcc_clusters)
        tstcc_db = davies_bouldin_score(ts_tcc_features, tstcc_clusters)
        tstcc_ch = calinski_harabasz_score(ts_tcc_features, tstcc_clusters)
        print(
            f"TS-TCC Clustering Metrics - Silhouette: {tstcc_silhouette:.4f}, DB: {tstcc_db:.4f}, CH: {tstcc_ch:.4f}")

        # Store results
        benchmark_results['clustering_metrics']['TS-TCC'] = {
            'silhouette': tstcc_silhouette,
            'davies_bouldin': tstcc_db,
            'calinski_harabasz': tstcc_ch,
            'n_clusters': len(np.unique(tstcc_clusters)),
            'processing_time': tstcc_time
        }
    except Exception as e:
        print(f"Could not calculate clustering metrics for TS-TCC: {e}")

    # Save analysis
    tstcc_characteristics = analyze_cluster_characteristics(
        ts_tcc_model, all_sequences, tstcc_clusters, feature_names, 'tstcc')

    # Downstream classification task
    try:
        # Split features into train/test
        train_size = int(0.8 * len(ts_tcc_features))
        X_train_feat, X_test_feat = ts_tcc_features[:train_size], ts_tcc_features[train_size:]
        y_train_feat, y_test_feat = y_train[:train_size], y_train[train_size:]

        # Train classifier
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train_feat, y_train_feat)

        # Predict
        y_pred = clf.predict(X_test_feat)

        # Calculate metrics
        accuracy = accuracy_score(y_test_feat, y_pred)
        f1 = f1_score(y_test_feat, y_pred, average='weighted')

        print(f"TS-TCC Downstream Classification - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

        # Store results
        benchmark_results['downstream_classification']['TS-TCC'] = {
            'accuracy': accuracy,
            'f1_score': f1
        }
    except Exception as e:
        print(f"Could not perform downstream classification for TS-TCC: {e}")

    # Save visualizations
    save_path = 'benchmark_results/tstcc'
    os.makedirs(save_path, exist_ok=True)
    visualize_cluster_characteristics(all_sequences, tstcc_clusters, feature_names, save_path)

    # Save cluster descriptions
    with open(f'{save_path}/cluster_descriptions.txt', 'w') as f:
        for cluster_id, characteristics in tstcc_characteristics.items():
            description = generate_cluster_description(characteristics, cluster_id)
            f.write(f"\n{'=' * 80}\n")
            f.write(description)
            f.write(f"\n{'=' * 80}\n")

    # ================ 2. Recurrent Autoencoder ================
    print("\nAnalyzing Recurrent Autoencoder model...")
    start_time = time.time()

    # Extract latent representations
    rauto_features = []
    autoencoder.eval()
    with torch.no_grad():
        for sequences, masks in rauto_train_loader:
            sequences = sequences.to(device)
            masks = masks.to(device)
            _, latent = autoencoder(sequences, masks)
            rauto_features.append(latent.cpu().numpy())

    rauto_features = np.vstack(rauto_features)

    # Clustering
    kmeans_rauto = KMeans(n_clusters=5, random_state=42)
    rauto_clusters = kmeans_rauto.fit_predict(rauto_features)

    # Calculate processing time
    rauto_time = time.time() - start_time

    # Calculate metrics
    try:
        rauto_silhouette = silhouette_score(rauto_features, rauto_clusters)
        rauto_db = davies_bouldin_score(rauto_features, rauto_clusters)
        rauto_ch = calinski_harabasz_score(rauto_features, rauto_clusters)
        print(
            f"Autoencoder Clustering Metrics - Silhouette: {rauto_silhouette:.4f}, DB: {rauto_db:.4f}, CH: {rauto_ch:.4f}")

        # Store results
        benchmark_results['clustering_metrics']['Autoencoder'] = {
            'silhouette': rauto_silhouette,
            'davies_bouldin': rauto_db,
            'calinski_harabasz': rauto_ch,
            'n_clusters': len(np.unique(rauto_clusters)),
            'processing_time': rauto_time
        }
    except Exception as e:
        print(f"Could not calculate clustering metrics for Autoencoder: {e}")

    # Analysis
    rauto_characteristics = analyze_cluster_characteristics(
        autoencoder, all_sequences, rauto_clusters, feature_names, 'rauto')

    # Downstream classification
    try:
        # Split features into train/test
        train_size = int(0.8 * len(rauto_features))
        X_train_feat, X_test_feat = rauto_features[:train_size], rauto_features[train_size:]
        y_train_feat, y_test_feat = y_train[:train_size], y_train[train_size:]

        # Train classifier
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train_feat, y_train_feat)

        # Predict
        y_pred = clf.predict(X_test_feat)

        # Calculate metrics
        accuracy = accuracy_score(y_test_feat, y_pred)
        f1 = f1_score(y_test_feat, y_pred, average='weighted')

        print(f"Autoencoder Downstream Classification - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

        # Store results
        benchmark_results['downstream_classification']['Autoencoder'] = {
            'accuracy': accuracy,
            'f1_score': f1
        }
    except Exception as e:
        print(f"Could not perform downstream classification for Autoencoder: {e}")

    # Save visualizations and descriptions
    save_path = 'benchmark_results/autoencoder'
    os.makedirs(save_path, exist_ok=True)
    visualize_cluster_characteristics(all_sequences, rauto_clusters, feature_names, save_path)

    with open(f'{save_path}/cluster_descriptions.txt', 'w') as f:
        for cluster_id, characteristics in rauto_characteristics.items():
            description = generate_cluster_description(characteristics, cluster_id)
            f.write(f"\n{'=' * 80}\n")
            f.write(description)
            f.write(f"\n{'=' * 80}\n")

    # ================ 3. Informative Missingness Model ================
    print("\nAnalyzing Informative Missingness model...")
    start_time = time.time()

    # Extract features and clusters
    imiss_clusters, _, imiss_features = cluster_latent_representations(imiss_model, rauto_batches, device,
                                                                       n_clusters=5)

    # Calculate processing time
    imiss_time = time.time() - start_time

    # Calculate metrics
    try:
        imiss_silhouette = silhouette_score(imiss_features, imiss_clusters)
        imiss_db = davies_bouldin_score(imiss_features, imiss_clusters)
        imiss_ch = calinski_harabasz_score(imiss_features, imiss_clusters)
        print(
            f"IMISS Clustering Metrics - Silhouette: {imiss_silhouette:.4f}, DB: {imiss_db:.4f}, CH: {imiss_ch:.4f}")

        # Store results
        benchmark_results['clustering_metrics']['IMISS'] = {
            'silhouette': imiss_silhouette,
            'davies_bouldin': imiss_db,
            'calinski_harabasz': imiss_ch,
            'n_clusters': len(np.unique(imiss_clusters)),
            'processing_time': imiss_time
        }
    except Exception as e:
        print(f"Could not calculate clustering metrics for IMISS: {e}")

    # Analysis
    imiss_characteristics = analyze_cluster_characteristics(
        imiss_model, all_sequences, imiss_clusters, feature_names, 'imiss')

    # Downstream classification
    try:
        # Split features into train/test
        train_size = int(0.8 * len(imiss_features))
        X_train_feat, X_test_feat = imiss_features[:train_size], imiss_features[train_size:]
        y_train_feat, y_test_feat = y_train[:train_size], y_train[train_size:]

        # Train classifier
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train_feat, y_train_feat)

        # Predict
        y_pred = clf.predict(X_test_feat)

        # Calculate metrics
        accuracy = accuracy_score(y_test_feat, y_pred)
        f1 = f1_score(y_test_feat, y_pred, average='weighted')

        print(f"IMISS Downstream Classification - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

        # Store results
        benchmark_results['downstream_classification']['IMISS'] = {
            'accuracy': accuracy,
            'f1_score': f1
             }

        # Save visualizations and descriptions
        save_path = 'benchmark_results/imiss'
        os.makedirs(save_path, exist_ok=True)
        visualize_cluster_characteristics(all_sequences, imiss_clusters, feature_names, save_path)

        with open(f'{save_path}/cluster_descriptions.txt', 'w') as f:
            for cluster_id, characteristics in imiss_characteristics.items():
                description = generate_cluster_description(characteristics, cluster_id)
                f.write(f"\n{'=' * 80}\n")
                f.write(description)
                f.write(f"\n{'=' * 80}\n")

        # ================ 4. Signature Transform ================
        print("\nAnalyzing Signature Transform model...")
        start_time = time.time()

        # Get clusters from the signature model (they're already computed)
        if hasattr(signature_model, 'kmeans') and hasattr(signature_model.kmeans, 'labels_'):
            sig_clusters = signature_model.kmeans.labels_
            sig_features = signature_model.signatures
        else:
            print("Signature model doesn't have precomputed clusters, recomputing...")
            _, sig_clusters = use_signature_transform_clustering(rauto_batches, n_clusters=5)
            sig_features = None

        # Calculate processing time
        sig_time = time.time() - start_time

        # Calculate metrics (if features are available)
        if sig_features is not None:
            try:
                sig_silhouette = silhouette_score(sig_features, sig_clusters)
                sig_db = davies_bouldin_score(sig_features, sig_clusters)
                sig_ch = calinski_harabasz_score(sig_features, sig_clusters)
                print(
                    f"Signature Clustering Metrics - Silhouette: {sig_silhouette:.4f}, DB: {sig_db:.4f}, CH: {sig_ch:.4f}")

                # Store results
                benchmark_results['clustering_metrics']['Signature'] = {
                    'silhouette': sig_silhouette,
                    'davies_bouldin': sig_db,
                    'calinski_harabasz': sig_ch,
                    'n_clusters': len(np.unique(sig_clusters)),
                    'processing_time': sig_time
                }
            except Exception as e:
                print(f"Could not calculate clustering metrics for Signature: {e}")

        # Analysis
        sig_characteristics = analyze_cluster_characteristics(
            signature_model, all_sequences, sig_clusters, feature_names, 'signature')

        # Skip downstream classification if features aren't available

        # Save visualizations and descriptions
        save_path = 'benchmark_results/signature'
        os.makedirs(save_path, exist_ok=True)
        visualize_cluster_characteristics(all_sequences, sig_clusters, feature_names, save_path)

        with open(f'{save_path}/cluster_descriptions.txt', 'w') as f:
            for cluster_id, characteristics in sig_characteristics.items():
                description = generate_cluster_description(characteristics, cluster_id)
                f.write(f"\n{'=' * 80}\n")
                f.write(description)
                f.write(f"\n{'=' * 80}\n")

        # ================ 5. Statistical Feature Representation ================
        print("\nAnalyzing Statistical Feature model...")
        start_time = time.time()

        # Get clusters from the statistical model (they're already computed)
        if hasattr(statfeat_model, 'kmeans') and hasattr(statfeat_model.kmeans, 'labels_'):
            stat_clusters = statfeat_model.kmeans.labels_
            stat_features = statfeat_model.feature_vectors
        else:
            print("Statistical model doesn't have precomputed clusters, recomputing...")
            _, stat_clusters = use_statistical_feature_representation(rauto_batches, n_clusters=5)
            stat_features = None

        # Calculate processing time
        stat_time = time.time() - start_time

        # Calculate metrics (if features are available)
        if stat_features is not None:
            try:
                stat_silhouette = silhouette_score(stat_features, stat_clusters)
                stat_db = davies_bouldin_score(stat_features, stat_clusters)
                stat_ch = calinski_harabasz_score(stat_features, stat_clusters)
                print(
                    f"Statistical Clustering Metrics - Silhouette: {stat_silhouette:.4f}, DB: {stat_db:.4f}, CH: {stat_ch:.4f}")

                # Store results
                benchmark_results['clustering_metrics']['Statistical'] = {
                    'silhouette': stat_silhouette,
                    'davies_bouldin': stat_db,
                    'calinski_harabasz': stat_ch,
                    'n_clusters': len(np.unique(stat_clusters)),
                    'processing_time': stat_time
                }
            except Exception as e:
                print(f"Could not calculate clustering metrics for Statistical: {e}")

        # Analysis
        stat_characteristics = analyze_cluster_characteristics(
            statfeat_model, all_sequences, stat_clusters, feature_names, 'statfeat')

        # Downstream classification if features are available
        if stat_features is not None:
            try:
                # Split features into train/test (if we have enough samples)
                if len(stat_features) > 10:  # Ensure we have enough samples
                    train_size = int(0.8 * len(stat_features))
                    X_train_feat, X_test_feat = stat_features[:train_size], stat_features[train_size:]
                    y_train_feat, y_test_feat = y_train[:train_size], y_train[train_size:]

                    # Train classifier
                    clf = LogisticRegression(max_iter=1000)
                    clf.fit(X_train_feat, y_train_feat)

                    # Predict
                    y_pred = clf.predict(X_test_feat)

                    # Calculate metrics
                    accuracy = accuracy_score(y_test_feat, y_pred)
                    f1 = f1_score(y_test_feat, y_pred, average='weighted')

                    print(f"Statistical Downstream Classification - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

                    # Store results
                    benchmark_results['downstream_classification']['Statistical'] = {
                        'accuracy': accuracy,
                        'f1_score': f1
                    }
            except Exception as e:
                print(f"Could not perform downstream classification for Statistical: {e}")

        # Get feature importance if available
        if hasattr(statfeat_model, 'get_feature_importance'):
            importance_df = statfeat_model.get_feature_importance()
            if importance_df is not None:
                top_features = importance_df.head(10)
                print("\nTop 10 important statistical features:")
                print(top_features)

                # Store in results
                benchmark_results['feature_importance']['Statistical'] = top_features.to_dict('records')

        # Save visualizations and descriptions
        save_path = 'benchmark_results/statistical'
        os.makedirs(save_path, exist_ok=True)
        visualize_cluster_characteristics(all_sequences, stat_clusters, feature_names, save_path)

        with open(f'{save_path}/cluster_descriptions.txt', 'w') as f:
            for cluster_id, characteristics in stat_characteristics.items():
                description = generate_cluster_description(characteristics, cluster_id)
                f.write(f"\n{'=' * 80}\n")
                f.write(description)
                f.write(f"\n{'=' * 80}\n")

        # ================ 6. RGAN Model ================
        print("\nAnalyzing RGAN model...")
        start_time = time.time()

        # Extract latent representations from RGAN
        rgan_features = rgan_model.get_latent_representation(all_sequences, all_masks)

        # Perform clustering on RGAN features
        kmeans_rgan = KMeans(n_clusters=5, random_state=42)
        rgan_clusters = kmeans_rgan.fit_predict(rgan_features)

        # Calculate processing time
        rgan_time = time.time() - start_time

        # Calculate metrics
        try:
            rgan_silhouette = silhouette_score(rgan_features, rgan_clusters)
            rgan_db = davies_bouldin_score(rgan_features, rgan_clusters)
            rgan_ch = calinski_harabasz_score(rgan_features, rgan_clusters)
            print(
                f"RGAN Clustering Metrics - Silhouette: {rgan_silhouette:.4f}, DB: {rgan_db:.4f}, CH: {rgan_ch:.4f}")

            # Store results
            benchmark_results['clustering_metrics']['RGAN'] = {
                'silhouette': rgan_silhouette,
                'davies_bouldin': rgan_db,
                'calinski_harabasz': rgan_ch,
                'n_clusters': len(np.unique(rgan_clusters)),
                'processing_time': rgan_time
            }
        except Exception as e:
            print(f"Could not calculate clustering metrics for RGAN: {e}")

        # Analysis
        rgan_characteristics = analyze_cluster_characteristics(
            rgan_model, all_sequences, rgan_clusters, feature_names, 'rgan')

        # Downstream classification
        try:
            # Split features into train/test
            train_size = int(0.8 * len(rgan_features))
            X_train_feat, X_test_feat = rgan_features[:train_size], rgan_features[train_size:]
            y_train_feat, y_test_feat = y_train[:train_size], y_train[train_size:]

            # Train classifier
            clf = LogisticRegression(max_iter=1000)
            clf.fit(X_train_feat, y_train_feat)

            # Predict
            y_pred = clf.predict(X_test_feat)

            # Calculate metrics
            accuracy = accuracy_score(y_test_feat, y_pred)
            f1 = f1_score(y_test_feat, y_pred, average='weighted')

            print(f"RGAN Downstream Classification - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

            # Store results
            benchmark_results['downstream_classification']['RGAN'] = {
                'accuracy': accuracy,
                'f1_score': f1
            }
        except Exception as e:
            print(f"Could not perform downstream classification for RGAN: {e}")

        # Get feature importance if available
        if hasattr(rgan_model, 'feature_importances') and rgan_model.feature_importances is not None:
            feature_importances = rgan_model.feature_importances
            top_indices = np.argsort(feature_importances)[::-1][:10]

            # Create a feature importance summary
            top_features = []
            for i, idx in enumerate(top_indices):
                if idx < len(feature_names):
                    top_features.append({
                        'rank': i + 1,
                        'feature': feature_names[idx],
                        'importance': float(feature_importances[idx])
                    })

            print("\nTop 10 important RGAN features:")
            for feat in top_features:
                print(f"{feat['rank']}. {feat['feature']}: {feat['importance']:.4f}")

            # Store in results
            benchmark_results['feature_importance']['RGAN'] = top_features

        # Save visualizations and descriptions
        save_path = 'benchmark_results/rgan'
        os.makedirs(save_path, exist_ok=True)
        visualize_cluster_characteristics(all_sequences, rgan_clusters, feature_names, save_path)

        with open(f'{save_path}/cluster_descriptions.txt', 'w') as f:
            for cluster_id, characteristics in rgan_characteristics.items():
                description = generate_cluster_description(characteristics, cluster_id)
                f.write(f"\n{'=' * 80}\n")
                f.write(description)
                f.write(f"\n{'=' * 80}\n")

        # ================ Final Comparison Visualizations ================
        print("\nCreating final comparison visualizations...")

        # 1. Clustering Performance Comparison
        plt.figure(figsize=(15, 10))

        # Silhouette scores
        silhouettes = {
            model: metrics.get('silhouette', 0)
            for model, metrics in benchmark_results['clustering_metrics'].items()
        }

        # Processing times
        times = {
            model: metrics.get('processing_time', 0)
            for model, metrics in benchmark_results['clustering_metrics'].items()
        }

        # Silhouette score comparison
        plt.subplot(2, 1, 1)
        models = list(silhouettes.keys())
        scores = list(silhouettes.values())

        plt.bar(models, scores, color='skyblue')
        plt.title('Clustering Quality (Silhouette Score)')
        plt.ylabel('Silhouette Score')
        plt.grid(axis='y', alpha=0.3)

        # Add values on top of bars
        for i, v in enumerate(scores):
            plt.text(i, v + 0.01, f"{v:.3f}", ha='center')

        # Processing time comparison
        plt.subplot(2, 1, 2)
        proc_times = list(times.values())

        plt.bar(models, proc_times, color='salmon')
        plt.title('Computational Efficiency (Processing Time)')
        plt.ylabel('Time (seconds)')
        plt.grid(axis='y', alpha=0.3)

        # Add values on top of bars
        for i, v in enumerate(proc_times):
            plt.text(i, v + 0.5, f"{v:.1f}s", ha='center')

        plt.tight_layout()
        plt.savefig('benchmark_results/clustering_performance_comparison.png')
        plt.close()

        # 2. Downstream classification performance
        plt.figure(figsize=(12, 6))

        # Collect metrics
        models = []
        accuracies = []
        f1_scores = []

        for model, metrics in benchmark_results['downstream_classification'].items():
            models.append(model)
            accuracies.append(metrics.get('accuracy', 0))
            f1_scores.append(metrics.get('f1_score', 0))

        # Set up bar positions
        x = np.arange(len(models))
        width = 0.35

        # Create bars
        plt.bar(x - width / 2, accuracies, width, label='Accuracy', color='lightblue')
        plt.bar(x + width / 2, f1_scores, width, label='F1 Score', color='lightgreen')

        # Add details
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title('Downstream Classification Performance')
        plt.xticks(x, models)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)

        # Add values on top of bars
        for i, v in enumerate(accuracies):
            plt.text(i - width / 2, v + 0.01, f"{v:.3f}", ha='center')

        for i, v in enumerate(f1_scores):
            plt.text(i + width / 2, v + 0.01, f"{v:.3f}", ha='center')

        plt.tight_layout()
        plt.savefig('benchmark_results/classification_performance_comparison.png')
        plt.close()

        # 3. Save benchmark results as JSON
        import json

        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        # Recursively convert all values
        def convert_dict_recursively(d):
            result = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    result[k] = convert_dict_recursively(v)
                else:
                    result[k] = convert_to_serializable(v)
            return result

        # Convert and save
        serializable_results = convert_dict_recursively(benchmark_results)

        with open('benchmark_results/benchmark_summary.json', 'w') as f:
            json.dump(serializable_results, f, indent=4)

        # Create summary report
        with open('benchmark_results/benchmark_report.md', 'w') as f:
            f.write("# Unsupervised Clustering Models Benchmark Report\n\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Clustering Performance Metrics\n\n")
            f.write(
                "| Model | Silhouette Score | Davies-Bouldin Index | Calinski-Harabasz Index | Processing Time |\n")
            f.write(
                "|-------|-----------------|----------------------|------------------------|----------------|\n")

            for model, metrics in benchmark_results['clustering_metrics'].items():
                f.write(
                    f"| {model} | {metrics.get('silhouette', 'N/A'):.4f} | {metrics.get('davies_bouldin', 'N/A'):.4f} | {metrics.get('calinski_harabasz', 'N/A'):.1f} | {metrics.get('processing_time', 'N/A'):.2f}s |\n")

            f.write("\n## Downstream Classification Performance\n\n")
            f.write("| Model | Accuracy | F1 Score |\n")
            f.write("|-------|----------|----------|\n")

            for model, metrics in benchmark_results['downstream_classification'].items():
                f.write(
                    f"| {model} | {metrics.get('accuracy', 'N/A'):.4f} | {metrics.get('f1_score', 'N/A'):.4f} |\n")

            f.write("\n## Feature Importance Analysis\n\n")
            for model, features in benchmark_results['feature_importance'].items():
                f.write(f"### {model} Model - Top Features\n\n")

                if isinstance(features, list):
                    # If features is a list of dictionaries
                    if len(features) > 0 and isinstance(features[0], dict):
                        if 'feature' in features[0] and 'importance' in features[0]:
                            f.write("| Rank | Feature | Importance |\n")
                            f.write("|------|---------|------------|\n")

                            for i, feat in enumerate(features[:10]):  # Show top 10
                                f.write(f"| {i + 1} | {feat['feature']} | {feat['importance']:.4f} |\n")

                f.write("\n")

            f.write("\n## Conclusion and Recommendations\n\n")

            # Find best model for each category
            best_silhouette = max(benchmark_results['clustering_metrics'].items(),
                                  key=lambda x: x[1].get('silhouette', 0))

            best_classification = max(benchmark_results['downstream_classification'].items(),
                                      key=lambda x: x[1].get('accuracy', 0))

            fastest_model = min(benchmark_results['clustering_metrics'].items(),
                                key=lambda x: x[1].get('processing_time', float('inf')))

            f.write(
                f"- **Best clustering quality**: {best_silhouette[0]} (Silhouette score: {best_silhouette[1]['silhouette']:.4f})\n")
            f.write(
                f"- **Best downstream performance**: {best_classification[0]} (Accuracy: {best_classification[1]['accuracy']:.4f})\n")
            f.write(
                f"- **Most computationally efficient**: {fastest_model[0]} (Processing time: {fastest_model[1]['processing_time']:.2f}s)\n\n")

            f.write("### Overall Recommendation\n\n")
            f.write("Based on the benchmark results, we recommend using:\n\n")

            # Simple weighted scoring to pick overall best model
            scores = {}
            for model in benchmark_results['clustering_metrics']:
                silhouette = benchmark_results['clustering_metrics'][model].get('silhouette', 0)

                # Get accuracy if available
                accuracy = 0
                if model in benchmark_results['downstream_classification']:
                    accuracy = benchmark_results['downstream_classification'][model].get('accuracy', 0)

                # Processing time (normalize inversely)
                proc_time = benchmark_results['clustering_metrics'][model].get('processing_time', float('inf'))
                time_score = 1.0 / (1.0 + proc_time / 10)  # Normalize: higher is better

                # Weighted sum
                scores[model] = 0.4 * silhouette + 0.4 * accuracy + 0.2 * time_score

            # Find overall best model
            best_model = max(scores.items(), key=lambda x: x[1])

            f.write(f"**{best_model[0]}** model as the best overall performer considering clustering quality, "
                    f"predictive power, and computational efficiency.\n")

        print("\nBenchmark complete! Results saved in benchmark_results directory.")
        print("See benchmark_results/benchmark_report.md for a detailed summary.")

        return benchmark_results
    except Exception as e:
        print(f"Could not calculate clustering metrics for Signature: {e}")
        
#----------- Main Program: Updated to incorporate feature naming and interpretability ----------------------
if __name__ == "__main__":
    # Path to your sepsis dataset
    file_path = r"C:\Users\ErikOhlsen\Downloads\sepsisalldatacsv.csv"

    print("Loading data...")
    # Load the data
    data = pd.read_csv(file_path)

    # Check for missing values
    missing_values = data.isnull().sum() / len(data) * 100
    print(f"Missing values per column (%):\n{missing_values}")

    # Extract feature names before dropping columns
    feature_cols = get_feature_names(data)
    print(f"Identified {len(feature_cols)} physiological features: {feature_cols[:5]}...")

    # Remove columns with too many missing values (>70%)
    threshold = 70.0
    cols_to_drop = missing_values[missing_values > threshold].index.tolist()
    data = data.drop(columns=cols_to_drop)

    # Fill remaining missing values
    data = data.fillna(method='ffill').fillna(method='bfill').fillna(0)

    # Separate time series features and labels
    label_col = 'SepsisLabel'
    id_col = 'Patient_ID'
    time_col = 'ICULOS'
    y = data[label_col].values
    patient_ids = data[id_col].values
    time_steps = data[time_col].values

    # Remove non-numeric or non-training relevant columns
    drop_cols = [label_col, id_col, time_col, 'Hour', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime']
    X = data.drop(columns=drop_cols).select_dtypes(include=[np.number])

    # Update feature names to include only the ones that remain after dropping
    feature_names = list(X.columns)
    print(f"Using {len(feature_names)} features after removing high-missingness columns")

    # Normalize feature values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create a list of time series per patient
    unique_patients = np.unique(patient_ids)
    sequences = []
    labels = []

    for patient in unique_patients:
        idx = patient_ids == patient
        patient_seq = X_scaled[idx]
        patient_label = y[idx][-1]  # Take the last label for the patient
        sequences.append(patient_seq)
        labels.append(patient_label)

    # Determine a reasonable maximum sequence length
    seq_lengths = [len(seq) for seq in sequences]
    print(
        f"Min sequence length: {min(seq_lengths)}, Max sequence length: {max(seq_lengths)}, Average: {np.mean(seq_lengths):.2f}")

    # Choose a reasonable maximum length (e.g., 95th percentile)
    max_sequence_length = int(np.percentile(seq_lengths, 95))
    print(f"Selected maximum sequence length: {max_sequence_length}")

    # Create padded sequences and masks
    padded_sequences, attention_masks = pad_sequences(sequences, max_sequence_length)

    # Split into training and test data
    X_train_pad, X_test_pad, y_train, y_test, mask_train, mask_test, patient_ids_train, patient_ids_test = train_test_split(
        padded_sequences, labels, attention_masks, unique_patients, test_size=0.2, random_state=42,
        stratify=labels
    )

    print(f"Shape of padded training sequences: {X_train_pad.shape}")
    print(f"Shape of padded test sequences: {X_test_pad.shape}")

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create DataLoaders
    if TRAIN_TSTCC:
        # TS-TCC specific DataLoader with augmentations
        tstcc_train_dataset = TSTCCDataset(X_train_pad, mask_train)
        tstcc_train_loader = DataLoader(tstcc_train_dataset, batch_size=batch_size, shuffle=True)
        print(f"TS-TCC Dataset created: {len(tstcc_train_dataset)} sequences")

    if TRAIN_RAUTO:
        # For Recurrent Autoencoder - length-based batches for more efficient training
        rauto_batches = create_length_based_batches(X_train_pad, mask_train, batch_size)
        print(f"Recurrent Autoencoder: {len(rauto_batches)} length-based batches created")

        # Also create a standard DataLoader
        rauto_train_dataset = SimpleDataset(X_train_pad, mask_train)
        rauto_train_loader = DataLoader(rauto_train_dataset, batch_size=batch_size, shuffle=True)

    # Training the models
    # Extract input dimension from training data
    input_dim = X_train_pad.shape[2]

    # Number of training cycles
    num_epochs = 20

    if TRAIN_TSTCC:
        print("\n==== Training TS-TCC Modell ====")
        ts_tcc_model, ts_tcc_loss = use_ts_tcc_model(
            input_dim, tstcc_train_loader, device, num_epochs=num_epochs)

        # Visualisieren des Trainingsverlaufs
        plt.figure(figsize=(10, 5))
        plt.plot(ts_tcc_loss)
        plt.title('TS-TCC Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig('tstcc_training_loss.png')
        plt.close()

    if TRAIN_RAUTO:
        print("\n==== Training Recurrent Autoencoder ====")
        autoencoder, autoencoder_loss = use_recurrent_autoencoder(
            input_dim, rauto_batches, device, num_epochs=num_epochs)

        # Visualisieren des Trainingsverlaufs
        plt.figure(figsize=(10, 5))
        plt.plot(autoencoder_loss)
        plt.title('Recurrent Autoencoder Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Reconstruction Loss')
        plt.grid(True)
        plt.savefig('autoencoder_training_loss.png')
        plt.close()

    if TRAIN_IMISS:
        print("\n==== Training Informative Missingness Model ====")
        imiss_model, imiss_loss = use_informative_missingness_model(
            input_dim, rauto_batches, device, num_epochs=num_epochs)

        # Extract loss components for visualization
        total_losses = [loss[0] for loss in imiss_loss]
        value_losses = [loss[1] for loss in imiss_loss]
        missingness_losses = [loss[2] for loss in imiss_loss]

        # Visualize training losses (all components)
        plt.figure(figsize=(12, 6))
        plt.plot(total_losses, label='Total Loss')
        plt.plot(value_losses, label='Value Loss')
        plt.plot(missingness_losses, label='Missingness Loss')
        plt.title('Informative Missingness Model Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('imiss_training_loss.png')
        plt.close()

        # Clustering of latent representations with enhanced interpretation
        print("\n==== Analyzing Informative Missingness Representations ====")
        clusters, centers, latents, characteristics, descriptions = cluster_latent_representations_enhanced(
            imiss_model, rauto_batches, device, n_clusters=5, feature_names=feature_names)

    if TRAIN_SIGNATURE:
        print("\n==== Signature Transform Clustering with Interpretation ====")
        # Use the enhanced version that includes cluster interpretation
        signature_model, signature_clusters, sig_characteristics, sig_descriptions = use_signature_transform_clustering_enhanced(
            rauto_batches, n_clusters=5, truncation_level=3, feature_names=feature_names)

    if TRAIN_STATFEAT:
        print("\n==== Statistical Feature Representation with Interpretation ====")
        # Use the enhanced version that includes cluster interpretation
        statfeat_model, statfeat_clusters, stat_characteristics, stat_descriptions = use_statistical_feature_representation_enhanced(
            rauto_batches, n_clusters=5, use_pca=True, n_components=min(10, input_dim * 14 // 2),
            feature_names=feature_names)

    if TRAIN_RGAN:
        print("\n==== Recurrent GAN with Cluster Interpretation ====")
        # Use the enhanced version that includes cluster interpretation
        rgan_model, rgan_latents, rgan_clusters, rgan_characteristics, rgan_descriptions = use_rgan_enhanced(
            rauto_batches, input_dim, device, epochs=20, sample_interval=5, feature_names=feature_names)

    # Run a comprehensive benchmark with interpretation
    run_benchmark(
        ts_tcc_model, autoencoder, imiss_model, signature_model, statfeat_model, rgan_model,
        X_train_pad, X_test_pad, y_train, y_test,
        mask_train, mask_test, patient_ids_train, patient_ids_test,
        tstcc_train_loader, rauto_train_loader, rauto_batches, device
    )

    print("All models trained and evaluated with enhanced cluster interpretation.")

