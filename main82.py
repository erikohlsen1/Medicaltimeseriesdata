import torch
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
import torch.nn as nn
import numpy as np
import time
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings

warnings.filterwarnings('ignore')

# Ergänze diese Schalter am Anfang des Codes (bei den globalen Parametern):
TRAIN_TSTCC = True  # Schalter für TS-TCC Training
TRAIN_RAUTO = True  # Schalter für rekurrenten Autoencoder Training
TRAIN_IMISS = True  # Schalter für Informative Missingness Modell
TRAIN_SIGNATURE = True  # Schalter für Signature Transform Clustering
TRAIN_STATFEAT = True  # Schalter für Statistische Feature-Repräsentation
TRAIN_RGAN = True  # Schalter für Rekurrentes GAN
batch_size = 32  # Standard-Batchgröße


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

        print(
            f"Signature Transform: Verarbeite {len(all_sequences)} Sequenzen mit Trunkierungslevel {truncation_level}")

        # Überprüfen der Maskenform und korrigieren, falls notwendig
        if len(all_masks.shape) > 2:
            # Wenn die Masken 3D sind (batch, seq_len, features), reduziere sie auf 2D
            all_masks = all_masks.mean(axis=2) > 0  # Mittelwert über Features und binarisieren

        # Modell initialisieren und anwenden
        model = SignatureTransformClustering(
            truncation_level=truncation_level,
            n_clusters=n_clusters
        )

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
                def __init__(self, kmeans, data):
                    self.kmeans = kmeans
                    self.signatures = data

            model = SimplifiedModel(kmeans, flattened_seqs)

            return model, cluster_labels

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
        else:
            self.feature_vectors = self.stats_features

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

        if not self.use_pca:
            # Für Nicht-PCA-Modelle, berechne die Feature-Wichtigkeit direkt
            feature_names = []
            var_names = ["Variable_" + str(i // 14 + 1) for i in range(self.stats_features.shape[1])]
            stat_names = ["Mittelwert", "Median", "Modus", "Q25",
                          "Min", "Max", "Std", "CV", "Range", "IQR",
                          "Schiefe", "Kurtosis", "MAD", "ZCR"]

            for i in range(self.stats_features.shape[1]):
                var_idx = i // 14
                stat_idx = i % 14
                feature_names.append(f"{var_names[var_idx]}_{stat_names[stat_idx]}")

            # Berechne die Standardabweichung jedes Features über alle Cluster
            centroids = self.kmeans.cluster_centers_
            feature_std = np.std(centroids, axis=0)

            # Sortiere Features nach ihrer Wichtigkeit
            importance_idx = np.argsort(feature_std)[::-1]

            # Erstelle DataFrame mit Feature-Namen und Wichtigkeit
            importance_df = pd.DataFrame({
                'Feature': [feature_names[i] for i in importance_idx],
                'Importance': feature_std[importance_idx]
            })

            return importance_df
        else:
            # Für PCA-Modelle, zeige die Feature-Gewichtung in den Hauptkomponenten
            if not hasattr(self, 'pca') or not hasattr(self.pca, 'components_'):
                print("PCA wurde nicht angewendet oder hat keine Komponenten.")
                return None

            # Erstelle Feature-Namen wie oben
            feature_names = []
            var_names = ["Variable_" + str(i // 14 + 1) for i in range(self.stats_features.shape[1])]
            stat_names = ["Mittelwert", "Median", "Modus", "Q25",
                          "Min", "Max", "Std", "CV", "Range", "IQR",
                          "Schiefe", "Kurtosis", "MAD", "ZCR"]

            for i in range(self.stats_features.shape[1]):
                var_idx = i // 14
                stat_idx = i % 14
                feature_names.append(f"{var_names[var_idx]}_{stat_names[stat_idx]}")

            # Berechne die Feature-Wichtigkeit basierend auf PCA-Komponenten
            pca_components = self.pca.components_
            pca_importance = np.abs(pca_components).mean(axis=0)

            # Sortiere Features nach ihrer Wichtigkeit
            importance_idx = np.argsort(pca_importance)[::-1]

            # Erstelle DataFrame mit Feature-Namen und Wichtigkeit
            importance_df = pd.DataFrame({
                'Feature': [feature_names[i] for i in importance_idx],
                'Importance': pca_importance[importance_idx]
            })

            return importance_df

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

            # Visualisiere die Feature-Wichtigkeit
            importance_df = model.get_feature_importance()
            if importance_df is not None:
                top_n = min(20, len(importance_df))
                print(f"\nTop {top_n} wichtigste Features:")
                print(importance_df.head(top_n))

                # Plot Feature-Wichtigkeit
                plt.figure(figsize=(12, 6))
                plt.bar(importance_df['Feature'].head(top_n), importance_df['Importance'].head(top_n))
                plt.title('Top Features nach Wichtigkeit')
                plt.xlabel('Features')
                plt.ylabel('Wichtigkeit')
                plt.xticks(rotation=90)
                plt.tight_layout()
                plt.savefig('statistical_feature_importance.png')
                plt.close()

        # Visualisiere die Cluster mit PCA (auf 2 Dimensionen reduzieren für Visualisierung)
        if model.feature_vectors.shape[1] > 2:
            viz_pca = PCA(n_components=2)
            viz_vectors = viz_pca.fit_transform(model.feature_vectors)

            plt.figure(figsize=(10, 8))
            for i in range(model.n_clusters):
                plt.scatter(
                    viz_vectors[cluster_labels == i, 0],
                    viz_vectors[cluster_labels == i, 1],
                    label=f'Cluster {i}'
                )

            plt.title('PCA-Visualisierung der Statistischen Feature-Cluster')
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.legend()
            plt.savefig('statistical_feature_clusters.png')
            plt.close()

        return model, cluster_labels


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
            plt.figure(figsize=(12, 6))

            # Echte Samples
            for i in range(min(n_samples, len(real_samples))):
                plt.subplot(2, 1, 1)
                plt.plot(real_samples[i, :, var_idx], label=f'Sample {i + 1}' if i == 0 else None)

            plt.title('Echte Samples')
            plt.xlabel('Zeitschritt')
            plt.ylabel(f'Variable {var_idx + 1}')
            if var_idx == 0:
                plt.legend()

            # Generierte Samples
            for i in range(n_samples):
                plt.subplot(2, 1, 2)
                plt.plot(gen_samples[i, :, var_idx], label=f'Sample {i + 1}' if i == 0 else None)

            plt.title('Generierte Samples')
            plt.xlabel('Zeitschritt')
            plt.ylabel(f'Variable {var_idx + 1}')
            if var_idx == 0:
                plt.legend()

            plt.tight_layout()
            plt.savefig(f'rgan_samples_var{var_idx + 1}.png')
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

        # Extrahiere latente Repräsentationen für alle Sequenzen
        print("Extrahiere latente Repräsentationen...")

        all_sequences = []
        all_masks = []

        for batch_sequences, batch_masks in batches:
            all_sequences.extend(batch_sequences)
            all_masks.extend(batch_masks)

        latent_reps = rgan.get_latent_representation(all_sequences, all_masks)

        # Visualisiere die latenten Repräsentationen mit t-SNE
        if latent_reps.shape[0] > 10:  # Nur wenn genügend Samples vorhanden sind
            print("Visualisiere latente Repräsentationen mit t-SNE...")

            # Setze eine niedrigere Perplexität, die kleiner als die Anzahl der Samples ist
            perplexity = min(5, latent_reps.shape[0] - 1)  # Mindestens 1 kleiner als Anzahl der Samples

            try:
                tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                latent_2d = tsne.fit_transform(latent_reps)

                plt.figure(figsize=(10, 8))
                plt.scatter(latent_2d[:, 0], latent_2d[:, 1])
                plt.title('t-SNE der RGAN latenten Repräsentationen')
                plt.xlabel('Dimension 1')
                plt.ylabel('Dimension 2')
                plt.savefig('rgan_tsne.png')
                plt.close()
            except Exception as e:
                print(f"Fehler bei der t-SNE-Visualisierung: {e}")
                print("Versuche stattdessen PCA zur Visualisierung...")

                # Fallback: Verwende PCA zur Visualisierung
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                try:
                    pca_result = pca.fit_transform(latent_reps)

                    plt.figure(figsize=(10, 8))
                    plt.scatter(pca_result[:, 0], pca_result[:, 1])
                    plt.title('PCA der RGAN latenten Repräsentationen')
                    plt.xlabel('PC1')
                    plt.ylabel('PC2')
                    plt.savefig('rgan_pca.png')
                    plt.close()

                    print(f"PCA-Visualisierung erfolgreich, erklärte Varianz: {sum(pca.explained_variance_ratio_):.2f}")
                except Exception as e2:
                    print(f"Auch PCA-Visualisierung fehlgeschlagen: {e2}")

        return rgan, latent_reps


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
            def __init__(self, kmeans, data):
                self.kmeans = kmeans
                self.signatures = data

        model = SimplifiedModel(kmeans, flattened_seqs)

        return model, cluster_labels


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

    # Visualisiere die Feature-Wichtigkeit
    importance_df = model.get_feature_importance()
    if importance_df is not None:
        top_n = min(20, len(importance_df))
        print(f"\nTop {top_n} wichtigste Features:")
        print(importance_df.head(top_n))

        # Plot Feature-Wichtigkeit
        plt.figure(figsize=(12, 6))
        plt.bar(importance_df['Feature'].head(top_n), importance_df['Importance'].head(top_n))
        plt.title('Top Features nach Wichtigkeit')
        plt.xlabel('Features')
        plt.ylabel('Wichtigkeit')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig('statistical_feature_importance.png')
        plt.close()

    # Visualisiere die Cluster mit PCA (auf 2 Dimensionen reduzieren für Visualisierung)
    if model.feature_vectors.shape[1] > 2:
        viz_pca = PCA(n_components=2)
        viz_vectors = viz_pca.fit_transform(model.feature_vectors)

        plt.figure(figsize=(10, 8))
        for i in range(model.n_clusters):
            plt.scatter(
                viz_vectors[cluster_labels == i, 0],
                viz_vectors[cluster_labels == i, 1],
                label=f'Cluster {i}'
            )

        plt.title('PCA-Visualisierung der Statistischen Feature-Cluster')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend()
        plt.savefig('statistical_feature_clusters.png')
        plt.close()

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


# Hauptprogramm
if __name__ == "__main__":
    # Path to your sepsis dataset
    file_path = r"C:\Users\ErikOhlsen\Downloads\sepsisalldatacsv.csv"

    print("Lade Daten...")
    # Einlesen der Daten
    data = pd.read_csv(file_path)

    # Überprüfung der fehlenden Werte
    missing_values = data.isnull().sum() / len(data) * 100
    print(f"Fehlende Werte pro Spalte (%):\n{missing_values}")

    # Entfernen von Spalten mit zu vielen fehlenden Werten (>70%)
    threshold = 70.0
    cols_to_drop = missing_values[missing_values > threshold].index.tolist()
    data = data.drop(columns=cols_to_drop)

    # Füllen der übrigen fehlenden Werte
    data = data.fillna(method='ffill').fillna(method='bfill').fillna(0)

    # Separation von Zeitreihenfeatures und Labels
    label_col = 'SepsisLabel'
    id_col = 'Patient_ID'
    time_col = 'ICULOS'
    y = data[label_col].values
    patient_ids = data[id_col].values
    time_steps = data[time_col].values

    # Entfernen von nicht-numerischen oder nicht für das Training relevanten Spalten
    drop_cols = [label_col, id_col, time_col, 'Hour', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime']
    X = data.drop(columns=drop_cols).select_dtypes(include=[np.number])

    # Normalisierung der Feature-Werte
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Erstellen einer Liste von Zeitreihen pro Patient
    unique_patients = np.unique(patient_ids)
    sequences = []
    labels = []

    for patient in unique_patients:
        idx = patient_ids == patient
        patient_seq = X_scaled[idx]
        patient_label = y[idx][-1]  # Nehme das letzte Label für den Patienten
        sequences.append(patient_seq)
        labels.append(patient_label)

    # Bestimme eine maximale Sequenzlänge basierend auf den Daten
    seq_lengths = [len(seq) for seq in sequences]
    print(
        f"Min Sequenzlänge: {min(seq_lengths)}, Max Sequenzlänge: {max(seq_lengths)}, Mittelwert: {np.mean(seq_lengths):.2f}")

    # Wähle eine vernünftige maximale Länge (z.B. 95. Perzentil)
    max_sequence_length = int(np.percentile(seq_lengths, 95))
    print(f"Gewählte maximale Sequenzlänge: {max_sequence_length}")

    # Erstelle gepaddte Sequenzen und Masken
    padded_sequences, attention_masks = pad_sequences(sequences, max_sequence_length)

    # Aufteilen in Trainings- und Testdaten mit den gepaddten Sequenzen
    X_train_pad, X_test_pad, y_train, y_test, mask_train, mask_test = train_test_split(
        padded_sequences, labels, attention_masks, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"Form der gepaddten Trainingssequenzen: {X_train_pad.shape}")
    print(f"Form der gepaddten Testsequenzen: {X_test_pad.shape}")

    # Device-Konfiguration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Verwendetes Gerät: {device}")

    # Erstellen der speziellen DataLoaders für die unüberwachten Modelle
    if TRAIN_TSTCC:
        # TS-TCC spezifischer DataLoader mit Augmentierungen
        tstcc_train_dataset = TSTCCDataset(X_train_pad, mask_train)
        tstcc_train_loader = DataLoader(tstcc_train_dataset, batch_size=batch_size, shuffle=True)
        print(f"TS-TCC Dataset erstellt: {len(tstcc_train_dataset)} Sequenzen")

    if TRAIN_RAUTO:
        # Für Rekurrenten Autoencoder - längenbasierte Batches für effizienteres Training
        rauto_batches = create_length_based_batches(X_train_pad, mask_train, batch_size)
        print(f"Rekurrenter Autoencoder: {len(rauto_batches)} längenbasierte Batches erstellt")

        # Auch einen Standard-Dataloader erstellen
        rauto_train_dataset = SimpleDataset(X_train_pad, mask_train)
        rauto_train_loader = DataLoader(rauto_train_dataset, batch_size=batch_size, shuffle=True)

    # Training der Modelle
    # Eingangsdimension aus den Trainingsdaten extrahieren
    input_dim = X_train_pad.shape[2]

    # Anzahl der Trainingszyklen
    num_epochs = 20

    if TRAIN_TSTCC:
        print("\n==== Training TS-TCC Modell ====")
        ts_tcc_model, ts_tcc_loss = use_ts_tcc_model(
            input_dim, tstcc_train_loader, device, num_epochs=num_epochs)

        # Visualisieren des Trainingsverlaufs
        plt.figure(figsize=(10, 5))
        plt.plot(ts_tcc_loss)
        plt.title('TS-TCC Trainingsverlauf')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig('tstcc_training_loss.png')
        plt.close()

    if TRAIN_RAUTO:
        print("\n==== Training Rekurrenter Autoencoder ====")
        autoencoder, autoencoder_loss = use_recurrent_autoencoder(
            input_dim, rauto_batches, device, num_epochs=num_epochs)

        # Visualisieren des Trainingsverlaufs
        plt.figure(figsize=(10, 5))
        plt.plot(autoencoder_loss)
        plt.title('Rekurrenter Autoencoder Trainingsverlauf')
        plt.xlabel('Epoch')
        plt.ylabel('Rekonstruktionsverlust')
        plt.grid(True)
        plt.savefig('autoencoder_training_loss.png')
        plt.close()

    if TRAIN_IMISS:
        print("\n==== Training Informative Missingness Modell ====")
        imiss_model, imiss_loss = use_informative_missingness_model(
            input_dim, rauto_batches, device, num_epochs=num_epochs)

        # Extrahiere die verschiedenen Verlustkomponenten für die Visualisierung
        total_losses = [loss[0] for loss in imiss_loss]
        value_losses = [loss[1] for loss in imiss_loss]
        missingness_losses = [loss[2] for loss in imiss_loss]

        # Visualisieren des Trainingsverlaufs (alle Verlustkomponenten)
        plt.figure(figsize=(12, 6))
        plt.plot(total_losses, label='Gesamtverlust')
        plt.plot(value_losses, label='Werteverlust')
        plt.plot(missingness_losses, label='Missingness-Verlust')
        plt.title('Informative Missingness Modell Trainingsverlauf')
        plt.xlabel('Epoch')
        plt.ylabel('Verlust')
        plt.legend()
        plt.grid(True)
        plt.savefig('imiss_training_loss.png')
        plt.close()

        # Clustering der latenten Repräsentationen
        print("\n==== Clustering der Informative Missingness Repräsentationen ====")
        n_clusters = 5  # Anzahl der Cluster
        clusters, centers, latents = cluster_latent_representations(
            imiss_model, rauto_batches, device, n_clusters=n_clusters)

        # Analyse der Cluster-Verteilung
        cluster_counts = np.bincount(clusters)
        print(f"Cluster-Verteilung: {cluster_counts}")

        # Visualisierung der Cluster-Verteilung
        plt.figure(figsize=(10, 6))
        plt.bar(range(n_clusters), cluster_counts)
        plt.title('Cluster-Verteilung des Informative Missingness Modells')
        plt.xlabel('Cluster')
        plt.ylabel('Anzahl der Sequenzen')
        plt.xticks(range(n_clusters))
        plt.grid(True, axis='y')
        plt.savefig('imiss_cluster_distribution.png')
        plt.close()

        # Optional: PCA für 2D-Visualisierung
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        latents_2d = pca.fit_transform(latents)

        plt.figure(figsize=(10, 8))
        for i in range(n_clusters):
            plt.scatter(latents_2d[clusters == i, 0], latents_2d[clusters == i, 1], label=f'Cluster {i}')
        plt.title('PCA-Visualisierung der Informative Missingness Repräsentationen')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend()
        plt.savefig('imiss_pca_visualization.png')
        plt.close()

    if TRAIN_SIGNATURE:
        print("\n==== Signature Transform Clustering ====")
        # Für Signature Transform benötigen wir eine Liste von Sequenzen
        signature_model, signature_clusters = use_signature_transform_clustering(
            rauto_batches, n_clusters=5, truncation_level=3)

        # Analyse der Cluster-Verteilung
        cluster_counts = np.bincount(signature_clusters)
        print(f"Signature Transform Cluster-Verteilung: {cluster_counts}")

        # Visualisierung der Cluster-Verteilung
        n_clusters = len(cluster_counts)
        plt.figure(figsize=(10, 6))
        plt.bar(range(n_clusters), cluster_counts)
        plt.title('Cluster-Verteilung des Signature Transform Modells')
        plt.xlabel('Cluster')
        plt.ylabel('Anzahl der Sequenzen')
        plt.xticks(range(n_clusters))
        plt.grid(True, axis='y')
        plt.savefig('signature_cluster_distribution.png')
        plt.close()

        # Visualisierung der Cluster-Zentren (wenn verfügbar)
        if hasattr(signature_model, 'signatures'):
            # Verwende PCA zur Dimensionsreduktion der Cluster-Zentren
            from sklearn.decomposition import PCA

            signatures_2d = PCA(n_components=2).fit_transform(signature_model.signatures)

            plt.figure(figsize=(10, 8))
            for i in range(n_clusters):
                plt.scatter(
                    signatures_2d[signature_clusters == i, 0],
                    signatures_2d[signature_clusters == i, 1],
                    label=f'Cluster {i}'
                )

            # Optional: Visualisiere auch die Cluster-Zentren
            centers_2d = PCA(n_components=2).fit_transform(signature_model.kmeans.cluster_centers_)
            plt.scatter(
                centers_2d[:, 0], centers_2d[:, 1],
                marker='x', s=100, c='black', label='Cluster-Zentren'
            )

            plt.title('PCA-Visualisierung der Signature Transform Cluster')
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.legend()
            plt.savefig('signature_pca_visualization.png')
            plt.close()

        if TRAIN_STATFEAT:
            print("\n==== Training Statistische Feature-Repräsentation ====")
            statfeat_model, statfeat_clusters = use_statistical_feature_representation(
                rauto_batches, n_clusters=5, use_pca=True, n_components=min(10, input_dim * 14 // 2)
            )

            # Visualisierung der Cluster-Verteilung
            n_clusters = len(np.bincount(statfeat_clusters))
            plt.figure(figsize=(10, 6))
            plt.bar(range(n_clusters), np.bincount(statfeat_clusters))
            plt.title('Cluster-Verteilung der Statistischen Feature-Repräsentation')
            plt.xlabel('Cluster')
            plt.ylabel('Anzahl der Sequenzen')
            plt.xticks(range(n_clusters))
            plt.grid(True, axis='y')
            plt.savefig('statfeat_cluster_distribution.png')
            plt.close()

        if TRAIN_RGAN:
            print("\n==== Training Rekurrentes GAN ====")
            rgan_model, rgan_latents = use_rgan(
                rauto_batches, input_dim, device, epochs=20, sample_interval=5
            )

            # Optional: Clustering der latenten Repräsentationen
            if len(rgan_latents) > 10:
                from sklearn.cluster import KMeans

                n_clusters = min(5, len(rgan_latents) // 2)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                rgan_clusters = kmeans.fit_predict(rgan_latents)

                # Visualisierung der Cluster-Verteilung
                plt.figure(figsize=(10, 6))
                plt.bar(range(n_clusters), np.bincount(rgan_clusters))
                plt.title('Cluster-Verteilung der RGAN latenten Repräsentationen')
                plt.xlabel('Cluster')
                plt.ylabel('Anzahl der Sequenzen')
                plt.xticks(range(n_clusters))
                plt.grid(True, axis='y')
                plt.savefig('rgan_cluster_distribution.png')
                plt.close()

    print("Datenvorverarbeitung und Training abgeschlossen.")

# ---------------------- Benchmark Framework für unüberwachte Modelle ----------------------
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import torch
import psutil
import os
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import tracemalloc
import gc


class ModelBenchmark:
    """
    Umfassendes Benchmark-Framework für unüberwachte Lernmodelle
    """

    def __init__(self):
        self.results = {
            'representation_quality': {},
            'computational_efficiency': {},
            'scalability': {},
            'generalization': {}
        }

        # Speichere Ergebnisse für jedes Modell
        self.model_results = {}

    def _measure_time(self, func, *args, **kwargs):
        """Misst die Ausführungszeit einer Funktion"""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time

    def _measure_memory(self, func, *args, **kwargs):
        """Misst den Speicherverbrauch einer Funktion"""
        tracemalloc.start()
        result = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return result, peak / 1024 / 1024  # MB

    def benchmark_representation_quality(self, model_name, model, X_train, X_test, y_train, y_test,
                                         mask_train, mask_test, device, model_type='tstcc'):
        """
        Bewertet die Qualität der gelernten Repräsentationen
        """
        print(f"\n==== Bewertung der Repräsentationsqualität für {model_name} ====")

        # Extrahiere Feature-Repräsentationen
        train_dataset = SimpleDataset(X_train, mask_train)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
        test_dataset = SimpleDataset(X_test, mask_test)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Extrahiere Features basierend auf dem Modelltyp
        if model_type == 'tstcc' or model_type == 'rauto':
            train_features = extract_unsupervised_features(model, train_loader, device, model_type)
            test_features = extract_unsupervised_features(model, test_loader, device, model_type)
        elif model_type == 'imiss':
            # Für Informative Missingness Modell
            train_features = []
            with torch.no_grad():
                for sequences, masks in train_loader:
                    sequences = sequences.to(device)
                    masks = masks.to(device)
                    latent = model.get_latent(sequences, masks)
                    train_features.append(latent.mean(dim=1).cpu().numpy())
                train_features = np.vstack(train_features)

            test_features = []
            with torch.no_grad():
                for sequences, masks in test_loader:
                    sequences = sequences.to(device)
                    masks = masks.to(device)
                    latent = model.get_latent(sequences, masks)
                    test_features.append(latent.mean(dim=1).cpu().numpy())
                test_features = np.vstack(test_features)
        elif model_type == 'signature':
            # Für Signature Transform Clustering
            train_features = model.signatures
            # Für Test-Features müssen wir neue Signaturen berechnen
            all_test_sequences = []
            all_test_masks = []
            for sequences, masks in test_loader:
                all_test_sequences.extend(sequences.numpy())
                all_test_masks.extend(masks.numpy())

            test_features = []
            for i, seq in enumerate(all_test_sequences):
                signature = model.signature_transform.compute_path_signature(seq, all_test_masks[i])
                test_features.append(signature)

            test_features = np.array(test_features)
            test_features = model.scaler.transform(test_features)
            if hasattr(model, 'pca') and hasattr(model.pca, 'components_'):
                test_features = model.pca.transform(test_features)
        elif model_type == 'statfeat':
            # Für Statistical Feature Representation
            train_features = model.feature_vectors

            # Extrahiere Test-Features
            all_test_sequences = []
            all_test_masks = []
            for sequences, masks in test_loader:
                all_test_sequences.extend(sequences.numpy())
                all_test_masks.extend(masks.numpy())

            test_features = []
            for i, seq in enumerate(all_test_sequences):
                features = model.extract_statistical_features(seq, all_test_masks[i])
                test_features.append(features)

            test_features = np.array(test_features)
            test_features = np.nan_to_num(test_features)
            test_features = model.scaler.transform(test_features)
            if model.use_pca:
                test_features = model.pca.transform(test_features)
        elif model_type == 'rgan':
            # Für RGAN
            all_train_sequences = []
            all_train_masks = []
            for sequences, masks in train_loader:
                all_train_sequences.extend(sequences.numpy())
                all_train_masks.extend(masks.numpy())

            all_test_sequences = []
            all_test_masks = []
            for sequences, masks in test_loader:
                all_test_sequences.extend(sequences.numpy())
                all_test_masks.extend(masks.numpy())

            train_features = model.get_latent_representation(all_train_sequences, all_train_masks)
            test_features = model.get_latent_representation(all_test_sequences, all_test_masks)
        else:
            print(f"Unbekannter Modelltyp: {model_type}")
            return None

        # 1. Downstream-Klassifikation mit verschiedenen Klassifikatoren
        classifiers = {
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'RandomForest': RandomForestClassifier(random_state=42)
        }

        classification_results = {}
        for clf_name, clf in classifiers.items():
            clf.fit(train_features, y_train)
            y_pred = clf.predict(test_features)
            try:
                y_prob = clf.predict_proba(test_features)[:, 1]
                auc = roc_auc_score(y_test, y_prob)
            except:
                auc = 0  # Falls der Klassifikator keine Wahrscheinlichkeiten unterstützt

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            classification_results[clf_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc
            }

            print(f"{clf_name} mit {model_name}-Features: Accuracy={accuracy:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

        # 2. Clustering-Metriken
        n_clusters = len(np.unique(y_test))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(test_features)

        silhouette = silhouette_score(test_features, cluster_labels)
        db_score = davies_bouldin_score(test_features, cluster_labels)
        ch_score = calinski_harabasz_score(test_features, cluster_labels)

        clustering_results = {
            'silhouette_score': silhouette,
            'davies_bouldin_score': db_score,
            'calinski_harabasz_score': ch_score
        }

        print(f"Clustering-Metriken: Silhouette={silhouette:.4f}, DB={db_score:.4f}, CH={ch_score:.4f}")

        # 3. Rekonstruktionsfehler (für Autoencoder)
        reconstruction_error = None
        if model_type == 'rauto':
            total_error = 0
            total_samples = 0
            model.eval()
            with torch.no_grad():
                for sequences, masks in test_loader:
                    sequences = sequences.to(device)
                    masks = masks.to(device)
                    reconstructed, _ = model(sequences, masks)

                    # Berechne MSE nur für gültige Werte
                    mse = ((reconstructed - sequences) ** 2) * masks.unsqueeze(-1)
                    total_error += mse.sum().item()
                    total_samples += masks.sum().item() * sequences.size(2)

            if total_samples > 0:
                reconstruction_error = total_error / total_samples
                print(f"Rekonstruktionsfehler: {reconstruction_error:.6f}")
        elif model_type == 'imiss':
            total_error = 0
            total_samples = 0
            model.eval()
            with torch.no_grad():
                for sequences, masks in test_loader:
                    sequences = sequences.to(device)
                    masks = masks.to(device)
                    x_hat, _, _ = model(sequences, masks)

                    # Berechne MSE nur für gültige Werte
                    if len(masks.shape) == 2:
                        masks_expanded = masks.unsqueeze(-1).expand_as(sequences)
                    else:
                        masks_expanded = masks

                    mse = ((x_hat - sequences) ** 2) * masks_expanded
                    total_error += mse.sum().item()
                    total_samples += masks_expanded.sum().item()

            if total_samples > 0:
                reconstruction_error = total_error / total_samples
                print(f"Rekonstruktionsfehler: {reconstruction_error:.6f}")

        # Visualisieren der gelernten Features
        self._visualize_features(test_features, y_test, model_name)

        # Speichern der Ergebnisse
        representation_results = {
            'classification': classification_results,
            'clustering': clustering_results,
            'reconstruction_error': reconstruction_error
        }

        self.results['representation_quality'][model_name] = representation_results
        return representation_results

    def benchmark_computational_efficiency(self, model_name, model_builder, input_dim,
                                           train_loader, batches, device, model_type='tstcc',
                                           num_epochs=5):
        """
        Bewertet die Recheneffizienz des Modells
        """
        print(f"\n==== Bewertung der Recheneffizienz für {model_name} ====")

        # Trainingszeit messen
        gc.collect()  # Garbage collection vor dem Test
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Extrahiere Sequenzen und Masken für alle Modelle einmal zu Beginn
        all_sequences = []
        all_masks = []

        # Extrahiere Sequenzen für alle Modelle, die sie benötigen könnten
        if batches is not None and model_type in ['signature', 'statfeat', 'rgan']:
            for batch_sequences, batch_masks in batches:
                all_sequences.extend(batch_sequences)
                all_masks.extend(batch_masks)

            all_sequences = np.array(all_sequences)
            all_masks = np.array(all_masks)

        if model_type == 'tstcc':
            # Instanziiere ein neues Modell für das Benchmark
            model = TS_TCC(input_dim).to(device)

            # Messe Trainingszeit
            start_time = time.time()
            train_ts_tcc(model, train_loader, device, num_epochs=num_epochs)
            train_time = time.time() - start_time

        elif model_type == 'rauto':
            # Instanziiere ein neues Modell für das Benchmark
            model = RecurrentAutoencoder(input_dim).to(device)

            # Messe Trainingszeit
            start_time = time.time()
            train_recurrent_autoencoder_with_length_batches(model, batches, device, num_epochs=num_epochs)
            train_time = time.time() - start_time

        elif model_type == 'imiss':
            # Instanziiere ein neues Modell für das Benchmark
            model = InformativeMissingnessAutoencoder(input_dim).to(device)

            # Messe Trainingszeit
            start_time = time.time()
            train_informative_missingness_model(model, batches, device, num_epochs=num_epochs)
            train_time = time.time() - start_time

        elif model_type == 'signature':
            # Messe Trainingszeit
            start_time = time.time()
            model = SignatureTransformClustering(
                truncation_level=3,
                n_clusters=min(5, len(all_sequences) // 2),  # Sicherstellen dass n_clusters < n_samples
                pca_components=min(30, all_sequences.shape[0] - 1)
            )
            model.fit(all_sequences, all_masks)
            train_time = time.time() - start_time

        elif model_type == 'statfeat':
            # Messe Trainingszeit
            start_time = time.time()
            model = StatisticalFeatureRepresentation(
                use_pca=True,
                n_components=min(10, all_sequences.shape[1] * 14),
                n_clusters=min(5, len(all_sequences) // 2)  # Sicherstellen dass n_clusters < n_samples
            )
            model.fit(all_sequences, all_masks)
            train_time = time.time() - start_time

        elif model_type == 'rgan':
            # Instanziiere ein neues Modell für das Benchmark
            rgan = RGAN(
                input_dim=input_dim,
                hidden_dim=128,
                n_layers=2,
                dropout=0.1,
                device=device
            )

            # Messe Trainingszeit
            start_time = time.time()
            rgan.train(batches, epochs=num_epochs, sample_interval=num_epochs)
            train_time = time.time() - start_time
            model = rgan

        else:
            print(f"Unbekannter Modelltyp: {model_type}")
            return None

        print(f"Trainingszeit für {num_epochs} Epochen: {train_time:.2f} Sekunden")

        # Inferenzzeit messen
        inference_times = []

        # Erstelle einen kleinen Batch für die Inferenz
        if model_type == 'tstcc':
            model.eval()  # Setze das Modell in den Evaluierungsmodus
            for sequences, _, _, masks in train_loader:
                sequences = sequences.to(device)

                # Warmup
                model(sequences)

                # Messe Inferenzzeit
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                with torch.no_grad():
                    for _ in range(10):  # Mehrere Durchläufe für stabilere Messungen
                        _ = model(sequences)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                inference_times.append((time.time() - start_time) / 10)
                break  # Nur ersten Batch verwenden

        elif model_type == 'rauto' or model_type == 'imiss':
            model.eval()  # Setze das Modell in den Evaluierungsmodus
            for sequences, masks in zip([batches[0][0]], [batches[0][1]]):
                sequences = torch.FloatTensor(sequences).to(device)
                masks = torch.FloatTensor(masks).to(device)

                # Warmup
                if model_type == 'rauto':
                    model(sequences, masks)
                else:
                    model(sequences, masks)

                # Messe Inferenzzeit
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                with torch.no_grad():
                    for _ in range(10):  # Mehrere Durchläufe für stabilere Messungen
                        if model_type == 'rauto':
                            _ = model(sequences, masks)
                        else:
                            _ = model(sequences, masks)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                inference_times.append((time.time() - start_time) / 10)

        elif model_type == 'signature':
            # Messe Zeit für die Berechnung einer einzelnen Signature
            seq = all_sequences[0]
            mask = all_masks[0]

            # Warmup
            _ = model.signature_transform.compute_path_signature(seq, mask)

            # Messe Inferenzzeit
            start_time = time.time()
            for _ in range(10):
                _ = model.signature_transform.compute_path_signature(seq, mask)
            inference_times.append((time.time() - start_time) / 10)

        elif model_type == 'statfeat':
            # Messe Zeit für die Extraktion statistischer Features
            seq = all_sequences[0]
            mask = all_masks[0]

            # Warmup
            _ = model.extract_statistical_features(seq, mask)

            # Messe Inferenzzeit
            start_time = time.time()
            for _ in range(10):
                _ = model.extract_statistical_features(seq, mask)
            inference_times.append((time.time() - start_time) / 10)

        elif model_type == 'rgan':
            model.generator.eval()  # Setze nur den Generator in den Evaluierungsmodus
            # Messe Zeit für die Generierung von Samples
            batch_size = 10

            # Bestimme Sequenzlänge aus den ersten Sequenzdaten
            if len(all_sequences) > 0:
                seq_len = all_sequences[0].shape[0]
            else:
                # Fallback, falls keine Sequenzen verfügbar sind
                seq_len = 50

            # Warmup
            with torch.no_grad():
                z = torch.randn(batch_size, input_dim).to(device)
                _ = model.generator(z, seq_len)

            # Messe Inferenzzeit
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            with torch.no_grad():
                for _ in range(10):
                    z = torch.randn(batch_size, input_dim).to(device)
                    _ = model.generator(z, seq_len)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            inference_times.append((time.time() - start_time) / 10)

        avg_inference_time = np.mean(inference_times)
        print(f"Durchschnittliche Inferenzzeit pro Batch: {avg_inference_time:.6f} Sekunden")

        # Speicherverbrauch messen
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        print(f"Speicherverbrauch: {memory_usage:.2f} MB")

        # Anzahl der Parameter - angepasst für verschiedene Modelltypen
        if model_type in ['tstcc', 'rauto', 'imiss']:
            num_params = sum(p.numel() for p in model.parameters())
        elif model_type == 'rgan':
            num_params = sum(p.numel() for p in model.generator.parameters()) + sum(
                p.numel() for p in model.discriminator.parameters())
        elif model_type == 'signature':
            # Für Signature Transform können wir Anzahl der Komponenten schätzen
            if hasattr(model, 'pca') and hasattr(model.pca, 'components_'):
                num_params = model.pca.components_.size + model.kmeans.cluster_centers_.size
            else:
                num_params = model.kmeans.cluster_centers_.size
        elif model_type == 'statfeat':
            # Für Statistical Feature können wir Anzahl der Komponenten schätzen
            if model.use_pca and hasattr(model, 'pca') and hasattr(model.pca, 'components_'):
                num_params = model.pca.components_.size + model.kmeans.cluster_centers_.size
            else:
                num_params = model.kmeans.cluster_centers_.size
        else:
            num_params = 0

        print(f"Anzahl der Modellparameter: {num_params}")

        # Speichern der Ergebnisse
        efficiency_results = {
            'training_time': train_time,
            'inference_time': avg_inference_time,
            'memory_usage': memory_usage,
            'parameter_count': num_params
        }

        self.results['computational_efficiency'][model_name] = efficiency_results
        return efficiency_results

    def benchmark_scalability(self, model_name, model_builder, input_dim, X_train, mask_train,
                              device, model_type='tstcc', num_epochs=5):
        """
        Bewertet die Skalierbarkeit des Modells mit zunehmender Datenmenge
        """
        print(f"\n==== Bewertung der Skalierbarkeit für {model_name} ====")

        # Teste verschiedene Datengrößen
        data_sizes = [0.2, 0.4, 0.6, 0.8, 1.0]
        scalability_results = {'data_sizes': {}, 'missing_values': {}}

        for size in data_sizes:
            n_samples = int(len(X_train) * size)
            X_subset = X_train[:n_samples]
            mask_subset = mask_train[:n_samples]

            if model_type == 'tstcc':
                subset_dataset = TSTCCDataset(X_subset, mask_subset)
                subset_loader = DataLoader(subset_dataset, batch_size=32, shuffle=True)

                # Instanziiere ein neues Modell
                model = TS_TCC(input_dim).to(device)

                # Messe Trainingszeit
                start_time = time.time()
                train_ts_tcc(model, subset_loader, device, num_epochs=num_epochs)
                train_time = time.time() - start_time

            elif model_type == 'rauto':
                subset_batches = create_length_based_batches(X_subset, mask_subset, batch_size=32)

                # Instanziiere ein neues Modell
                model = RecurrentAutoencoder(input_dim).to(device)

                # Messe Trainingszeit
                start_time = time.time()
                train_recurrent_autoencoder_with_length_batches(model, subset_batches, device,
                                                                num_epochs=num_epochs)
                train_time = time.time() - start_time

            elif model_type == 'imiss':
                subset_batches = create_length_based_batches(X_subset, mask_subset, batch_size=32)

                # Instanziiere ein neues Modell
                model = InformativeMissingnessAutoencoder(input_dim).to(device)

                # Messe Trainingszeit
                start_time = time.time()
                train_informative_missingness_model(model, subset_batches, device, num_epochs=num_epochs)
                train_time = time.time() - start_time

            elif model_type == 'signature':
                subset_batches = create_length_based_batches(X_subset, mask_subset, batch_size=32)

                # Extrahiere Sequenzen und Masken
                all_sequences = []
                all_masks = []
                for batch_sequences, batch_masks in subset_batches:
                    all_sequences.extend(batch_sequences)
                    all_masks.extend(batch_masks)

                all_sequences = np.array(all_sequences)
                all_masks = np.array(all_masks)

                # Berechne Anzahl der Cluster basierend auf Datengröße
                # Stelle sicher, dass n_clusters < n_samples
                n_clusters = min(5, max(2, len(all_sequences) // 2))

                # Messe Trainingszeit
                start_time = time.time()
                try:
                    model = SignatureTransformClustering(
                        truncation_level=3,
                        n_clusters=n_clusters,
                        pca_components=min(30, len(all_sequences) - 1)
                    )
                    model.fit(all_sequences, all_masks)
                    train_time = time.time() - start_time
                except Exception as e:
                    print(f"Fehler beim Training für {size * 100:.0f}% der Daten: {e}")
                    train_time = float('nan')  # NaN für fehlgeschlagene Versuche

            elif model_type == 'statfeat':
                subset_batches = create_length_based_batches(X_subset, mask_subset, batch_size=32)

                # Extrahiere Sequenzen und Masken
                all_sequences = []
                all_masks = []
                for batch_sequences, batch_masks in subset_batches:
                    all_sequences.extend(batch_sequences)
                    all_masks.extend(batch_masks)

                all_sequences = np.array(all_sequences)
                all_masks = np.array(all_masks)

                # Berechne Anzahl der Cluster basierend auf Datengröße
                # Stelle sicher, dass n_clusters < n_samples
                n_clusters = min(5, max(2, len(all_sequences) // 2))

                # Messe Trainingszeit
                start_time = time.time()
                try:
                    model = StatisticalFeatureRepresentation(
                        use_pca=True,
                        n_components=min(10, all_sequences.shape[1] * 14),
                        n_clusters=n_clusters
                    )
                    model.fit(all_sequences, all_masks)
                    train_time = time.time() - start_time
                except Exception as e:
                    print(f"Fehler beim Training für {size * 100:.0f}% der Daten: {e}")
                    train_time = float('nan')  # NaN für fehlgeschlagene Versuche

            elif model_type == 'rgan':
                subset_batches = create_length_based_batches(X_subset, mask_subset, batch_size=32)

                # Instanziiere ein neues Modell, verwende hier BenchmarkRGAN statt normaler RGAN
                rgan = BenchmarkRGAN(
                    input_dim=input_dim,
                    hidden_dim=128,
                    n_layers=2,
                    dropout=0.1,
                    device=device
                )

                # Messe Trainingszeit
                try:
                    start_time = time.time()
                    rgan.train(subset_batches, epochs=num_epochs, sample_interval=num_epochs)
                    train_time = time.time() - start_time
                except Exception as e:
                    print(f"Fehler beim Training für {size * 100:.0f}% der Daten: {e}")
                    train_time = float('nan')  # NaN für fehlgeschlagene Versuche

            scalability_results['data_sizes'][size] = train_time
            print(f"Trainingszeit für {size * 100:.0f}% der Daten: {train_time:.2f} Sekunden")

        # Test mit verschiedenen Anteilen fehlender Werte
        missing_rates = [0.0, 0.1, 0.2, 0.3, 0.4]

        for rate in missing_rates:
            if rate == 0.0:
                # Keine zusätzlichen fehlenden Werte
                X_missing = X_train.copy()
                mask_missing = mask_train.copy()
            else:
                # Füge künstlich fehlende Werte hinzu, indem Masken aktualisiert werden
                X_missing = X_train.copy()
                mask_missing = mask_train.copy()

                # Finde ursprünglich gültige Punkte
                valid_indices = np.where(mask_missing == 1)
                n_valid = len(valid_indices[0])
                n_to_mask = int(n_valid * rate)

                # Wähle zufällig einige davon aus, um sie zu maskieren
                to_mask_indices = np.random.choice(range(n_valid), n_to_mask, replace=False)

                for idx in to_mask_indices:
                    i, j = valid_indices[0][idx], valid_indices[1][idx]
                    mask_missing[i, j] = 0
                    X_missing[i, j] = 0  # Setze diese Werte auf 0

            if model_type == 'tstcc':
                missing_dataset = TSTCCDataset(X_missing, mask_missing)
                missing_loader = DataLoader(missing_dataset, batch_size=32, shuffle=True)

                # Instanziiere ein neues Modell
                model = TS_TCC(input_dim).to(device)

                # Messe Trainingszeit
                start_time = time.time()
                train_ts_tcc(model, missing_loader, device, num_epochs=num_epochs)
                train_time = time.time() - start_time

            elif model_type == 'rauto':
                missing_batches = create_length_based_batches(X_missing, mask_missing, batch_size=32)

                # Instanziiere ein neues Modell
                model = RecurrentAutoencoder(input_dim).to(device)

                # Messe Trainingszeit
                start_time = time.time()
                train_recurrent_autoencoder_with_length_batches(model, missing_batches, device,
                                                                num_epochs=num_epochs)
                train_time = time.time() - start_time

            elif model_type == 'imiss':
                missing_batches = create_length_based_batches(X_missing, mask_missing, batch_size=32)

                # Instanziiere ein neues Modell
                model = InformativeMissingnessAutoencoder(input_dim).to(device)

                # Messe Trainingszeit
                start_time = time.time()
                train_informative_missingness_model(model, missing_batches, device, num_epochs=num_epochs)
                train_time = time.time() - start_time

            elif model_type == 'signature':
                missing_batches = create_length_based_batches(X_missing, mask_missing, batch_size=32)

                # Extrahiere Sequenzen und Masken
                all_sequences = []
                all_masks = []
                for batch_sequences, batch_masks in missing_batches:
                    all_sequences.extend(batch_sequences)
                    all_masks.extend(batch_masks)

                all_sequences = np.array(all_sequences)
                all_masks = np.array(all_masks)

                # Berechne Anzahl der Cluster basierend auf Datengröße
                # Stelle sicher, dass n_clusters < n_samples
                n_clusters = min(5, max(2, len(all_sequences) // 2))

                # Messe Trainingszeit
                start_time = time.time()
                try:
                    model = SignatureTransformClustering(
                        truncation_level=3,
                        n_clusters=n_clusters,
                        pca_components=min(30, len(all_sequences) - 1)
                    )
                    model.fit(all_sequences, all_masks)
                    train_time = time.time() - start_time
                except Exception as e:
                    print(f"Fehler beim Training mit {rate * 100:.0f}% fehlenden Werten: {e}")
                    train_time = float('nan')  # NaN für fehlgeschlagene Versuche

            elif model_type == 'statfeat':
                missing_batches = create_length_based_batches(X_missing, mask_missing, batch_size=32)

                # Extrahiere Sequenzen und Masken
                all_sequences = []
                all_masks = []
                for batch_sequences, batch_masks in missing_batches:
                    all_sequences.extend(batch_sequences)
                    all_masks.extend(batch_masks)

                all_sequences = np.array(all_sequences)
                all_masks = np.array(all_masks)

                # Berechne Anzahl der Cluster basierend auf Datengröße
                # Stelle sicher, dass n_clusters < n_samples
                n_clusters = min(5, max(2, len(all_sequences) // 2))

                # Messe Trainingszeit
                start_time = time.time()
                try:
                    model = StatisticalFeatureRepresentation(
                        use_pca=True,
                        n_components=min(10, all_sequences.shape[1] * 14),
                        n_clusters=n_clusters
                    )
                    model.fit(all_sequences, all_masks)
                    train_time = time.time() - start_time
                except Exception as e:
                    print(f"Fehler beim Training mit {rate * 100:.0f}% fehlenden Werten: {e}")
                    train_time = float('nan')  # NaN für fehlgeschlagene Versuche

            elif model_type == 'rgan':
                missing_batches = create_length_based_batches(X_missing, mask_missing, batch_size=32)

                # Instanziiere ein neues Modell, verwende hier BenchmarkRGAN statt normaler RGAN
                rgan = BenchmarkRGAN(
                    input_dim=input_dim,
                    hidden_dim=128,
                    n_layers=2,
                    dropout=0.1,
                    device=device
                )

                # Messe Trainingszeit
                try:
                    start_time = time.time()
                    rgan.train(missing_batches, epochs=num_epochs, sample_interval=num_epochs)
                    train_time = time.time() - start_time
                except Exception as e:
                    print(f"Fehler beim Training mit {rate * 100:.0f}% fehlenden Werten: {e}")
                    train_time = float('nan')  # NaN für fehlgeschlagene Versuche

            scalability_results['missing_values'][rate] = train_time
            print(f"Trainingszeit mit {rate * 100:.0f}% zusätzlichen fehlenden Werten: {train_time:.2f} Sekunden")

        # Visualisieren der Skalierbarkeit
        self._plot_scalability(scalability_results, model_name)

        self.results['scalability'][model_name] = scalability_results
        return scalability_results

    def benchmark_generalization(self, model_name, model, X_train, X_test, y_train, y_test,
                                 mask_train, mask_test, patient_ids_train, patient_ids_test,
                                 device, model_type='tstcc'):
        """
        Bewertet die Generalisierungsfähigkeit des Modells
        """
        print(f"\n==== Bewertung der Generalisierungsfähigkeit für {model_name} ====")

        # 1. Erstelle Feature-Repräsentationen
        train_dataset = SimpleDataset(X_train, mask_train)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
        test_dataset = SimpleDataset(X_test, mask_test)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Extrahiere Features basierend auf dem Modelltyp
        if model_type == 'tstcc' or model_type == 'rauto':
            train_features = extract_unsupervised_features(model, train_loader, device, model_type)
            test_features = extract_unsupervised_features(model, test_loader, device, model_type)
        elif model_type == 'imiss':
            # Für Informative Missingness Modell
            train_features = []
            with torch.no_grad():
                for sequences, masks in train_loader:
                    sequences = sequences.to(device)
                    masks = masks.to(device)
                    latent = model.get_latent(sequences, masks)
                    train_features.append(latent.mean(dim=1).cpu().numpy())
                train_features = np.vstack(train_features)

            test_features = []
            with torch.no_grad():
                for sequences, masks in test_loader:
                    sequences = sequences.to(device)
                    masks = masks.to(device)
                    latent = model.get_latent(sequences, masks)
                    test_features.append(latent.mean(dim=1).cpu().numpy())
                test_features = np.vstack(test_features)
        elif model_type == 'signature':
            # Für Signature Transform Clustering
            train_features = model.signatures
            # Für Test-Features müssen wir neue Signaturen berechnen
            all_test_sequences = []
            all_test_masks = []
            for sequences, masks in test_loader:
                all_test_sequences.extend(sequences.numpy())
                all_test_masks.extend(masks.numpy())

            test_features = []
            for i, seq in enumerate(all_test_sequences):
                signature = model.signature_transform.compute_path_signature(seq, all_test_masks[i])
                test_features.append(signature)

            test_features = np.array(test_features)
            test_features = model.scaler.transform(test_features)
            if hasattr(model, 'pca') and hasattr(model.pca, 'components_'):
                test_features = model.pca.transform(test_features)
        elif model_type == 'statfeat':
            # Für Statistical Feature Representation
            train_features = model.feature_vectors

            # Extrahiere Test-Features
            all_test_sequences = []
            all_test_masks = []
            for sequences, masks in test_loader:
                all_test_sequences.extend(sequences.numpy())
                all_test_masks.extend(masks.numpy())

            test_features = []
            for i, seq in enumerate(all_test_sequences):
                features = model.extract_statistical_features(seq, all_test_masks[i])
                test_features.append(features)

            test_features = np.array(test_features)
            test_features = np.nan_to_num(test_features)
            test_features = model.scaler.transform(test_features)
            if model.use_pca:
                test_features = model.pca.transform(test_features)
        elif model_type == 'rgan':
            # Für RGAN
            all_train_sequences = []
            all_train_masks = []
            for sequences, masks in train_loader:
                all_train_sequences.extend(sequences.numpy())
                all_train_masks.extend(masks.numpy())

            all_test_sequences = []
            all_test_masks = []
            for sequences, masks in test_loader:
                all_test_sequences.extend(sequences.numpy())
                all_test_masks.extend(masks.numpy())

            train_features = model.get_latent_representation(all_train_sequences, all_train_masks)
            test_features = model.get_latent_representation(all_test_sequences, all_test_masks)
        else:
            print(f"Unbekannter Modelltyp: {model_type}")
            return None

        # 2. Trainiere einen einfachen Klassifikator
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(train_features, y_train)

        # 3. Bewerte auf Testdaten
        y_pred = clf.predict(test_features)

        # Basismessungen
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        print(f"Basisleistung - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

        # 4. Patientenweise Analyse
        # Stellen wir sicher, dass patient_ids_test in passendem Format vorliegt
        unique_patients = np.unique(patient_ids_test)
        patient_performances = []

        for patient in unique_patients:
            # Finde Indizes für diesen Patienten
            patient_mask = patient_ids_test == patient
            # Prüfen, ob Masken-Array gültige Werte enthält
            if np.sum(patient_mask) == 0:
                continue

            # Extrahiere Features und Labels für diesen Patienten
            patient_features = test_features[patient_mask]
            patient_y_test = np.array(y_test)[patient_mask]

            # Überspringe Patienten mit zu wenig Daten
            if len(patient_features) < 2:
                continue

            # Prüfen, ob alle Labels gleich sind (nur eine Klasse)
            if len(np.unique(patient_y_test)) < 2:
                patient_preds = clf.predict(patient_features)
                try:
                    patient_acc = accuracy_score(patient_y_test, patient_preds)
                    patient_performances.append((patient_acc, 1.0))  # Wenn nur eine Klasse: F1 = 1.0 oder N/A
                except:
                    continue
            else:
                # Mehrere Klassen vorhanden
                patient_preds = clf.predict(patient_features)
                try:
                    patient_acc = accuracy_score(patient_y_test, patient_preds)
                    patient_f1 = f1_score(patient_y_test, patient_preds, average='weighted')
                    patient_performances.append((patient_acc, patient_f1))
                except:
                    continue

        # Wenn keine gültigen Patientenperformances: Dummy-Werte verwenden
        if not patient_performances:
            print("Keine ausreichenden patientenweisen Daten für die Analyse verfügbar.")
            patient_acc_mean, patient_acc_std = 0.0, 0.0
            patient_f1_mean, patient_f1_std = 0.0, 0.0
        else:
            patient_acc_mean = np.mean([p[0] for p in patient_performances])
            patient_acc_std = np.std([p[0] for p in patient_performances])
            patient_f1_mean = np.mean([p[1] for p in patient_performances])
            patient_f1_std = np.std([p[1] for p in patient_performances])

        print(f"Patientenweise Leistung - Accuracy: {patient_acc_mean:.4f} ± {patient_acc_std:.4f}")
        print(f"Patientenweise Leistung - F1: {patient_f1_mean:.4f} ± {patient_f1_std:.4f}")

        # 5. Robustheit - Trainiere mit verschiedenen Untermengen an Patienten
        robustness_results = []

        unique_patients_train = np.unique(patient_ids_train)

        # Mindestens 2 Patienten für Robustheitstests benötigt
        n_splits = min(5, len(unique_patients_train))

        if n_splits < 2:
            print("Nicht genügend Patienten für Robustheitstests.")
            rob_acc_mean, rob_acc_std = 0.0, 0.0
            rob_f1_mean, rob_f1_std = 0.0, 0.0
        else:
            for i in range(n_splits):
                # Wähle eine Teilmenge der Patienten (mindestens 1)
                subset_size = max(1, len(unique_patients_train) // 2)
                subset_patients = np.random.choice(unique_patients_train,
                                                   size=subset_size,
                                                   replace=False)

                # Finde Indizes in den Trainingsdaten
                subset_mask = np.isin(patient_ids_train, subset_patients)

                # Prüfen, ob Maske gültige Werte enthält
                if np.sum(subset_mask) == 0:
                    continue

                subset_features = train_features[subset_mask]
                subset_labels = np.array(y_train)[subset_mask]

                # Überspringen, wenn zu wenig Daten
                if len(subset_features) < 5:
                    continue

                try:
                    subset_clf = LogisticRegression(max_iter=1000, random_state=42)
                    subset_clf.fit(subset_features, subset_labels)

                    subset_preds = subset_clf.predict(test_features)
                    subset_acc = accuracy_score(y_test, subset_preds)
                    subset_f1 = f1_score(y_test, subset_preds, average='weighted')

                    robustness_results.append((subset_acc, subset_f1))
                except Exception as e:
                    print(f"Fehler bei Robustheitstest: {e}")
                    continue

            # Dummy-Werte, falls keine Robustheitstests möglich waren
            if not robustness_results:
                rob_acc_mean, rob_acc_std = 0.0, 0.0
                rob_f1_mean, rob_f1_std = 0.0, 0.0
            else:
                rob_acc_mean = np.mean([r[0] for r in robustness_results])
                rob_acc_std = np.std([r[0] for r in robustness_results])
                rob_f1_mean = np.mean([r[1] for r in robustness_results])
                rob_f1_std = np.std([r[1] for r in robustness_results])

        print(f"Robustheit - Accuracy: {rob_acc_mean:.4f} ± {rob_acc_std:.4f}")
        print(f"Robustheit - F1: {rob_f1_mean:.4f} ± {rob_f1_std:.4f}")

        # Speichern der Ergebnisse
        generalization_results = {
            'base_performance': {'accuracy': accuracy, 'f1': f1},
            'patient_performance': {
                'accuracy_mean': patient_acc_mean,
                'accuracy_std': patient_acc_std,
                'f1_mean': patient_f1_mean,
                'f1_std': patient_f1_std
            },
            'robustness': {
                'accuracy_mean': rob_acc_mean,
                'accuracy_std': rob_acc_std,
                'f1_mean': rob_f1_mean,
                'f1_std': rob_f1_std
            }
        }

        self.results['generalization'][model_name] = generalization_results
        return generalization_results

    def evaluate_all(self, model_names, models, model_builders, X_train, X_test, y_train, y_test,
                     mask_train, mask_test, patient_ids_train, patient_ids_test, train_loaders,
                     length_batches, device, model_types, save_path='benchmark_results'):
        """
        Führt alle Benchmarks für alle Modelle aus
        """
        os.makedirs(save_path, exist_ok=True)

        for i, model_name in enumerate(model_names):
            print(f"\n\n======= Benchmarking {model_name} =======")

            model = models[i]
            model_builder = model_builders[i]
            model_type = model_types[i]
            input_dim = X_train.shape[2]

            # 1. Repräsentationsqualität
            rep_quality = self.benchmark_representation_quality(
                model_name, model, X_train, X_test, y_train, y_test,
                mask_train, mask_test, device, model_type
            )

            # 2. Recheneffizienz
            comp_efficiency = self.benchmark_computational_efficiency(
                model_name, model_builder, input_dim,
                train_loaders[i] if i < len(train_loaders) else None,
                length_batches[i] if i < len(length_batches) and length_batches[i] is not None else None,
                device, model_type
            )

            # 3. Skalierbarkeit
            scalability = self.benchmark_scalability(
                model_name, model_builder, input_dim, X_train, mask_train,
                device, model_type
            )

            # 4. Generalisierungsfähigkeit
            generalization = self.benchmark_generalization(
                model_name, model, X_train, X_test, y_train, y_test,
                mask_train, mask_test, patient_ids_train, patient_ids_test,
                device, model_type
            )

            # Sammle alle Ergebnisse für dieses Modell
            self.model_results[model_name] = {
                'representation_quality': rep_quality,
                'computational_efficiency': comp_efficiency,
                'scalability': scalability,
                'generalization': generalization
            }

        # Erstelle zusammenfassende Visualisierungen
        self._create_comparison_plots(save_path)

        # Speichere Ergebnisse
        self._save_results(save_path)

        return self.results, self.model_results

    def _visualize_features(self, features, labels, model_name):
        """Visualisiert die gelernten Feature-Repräsentationen"""
        plt.figure(figsize=(12, 10))

        # PCA-Visualisierung
        plt.subplot(2, 1, 1)
        pca = PCA(n_components=2)
        pca_features = pca.fit_transform(features)

        for label in np.unique(labels):
            plt.scatter(
                pca_features[labels == label, 0],
                pca_features[labels == label, 1],
                label=f'Klasse {label}',
                alpha=0.7
            )

        plt.title(f'PCA-Visualisierung der {model_name}-Features')
        plt.xlabel('Komponente 1')
        plt.ylabel('Komponente 2')
        plt.legend()

        # t-SNE-Visualisierung
        plt.subplot(2, 1, 2)

        # Anpassen der Perplexität an die Datensatzgröße
        n_samples = len(features)
        perplexity = min(30, n_samples // 4)  # Sicherstellen, dass perplexity < n_samples
        perplexity = max(5, perplexity)  # Mindestens 5 für sinnvolle Ergebnisse

        print(f"Verwende t-SNE mit Perplexität {perplexity} für {n_samples} Samples")

        try:
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            tsne_features = tsne.fit_transform(features)

            for label in np.unique(labels):
                plt.scatter(
                    tsne_features[labels == label, 0],
                    tsne_features[labels == label, 1],
                    label=f'Klasse {label}',
                    alpha=0.7
                )

            plt.title(f't-SNE-Visualisierung der {model_name}-Features')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            plt.legend()
        except Exception as e:
            plt.text(0.5, 0.5, f"t-SNE konnte nicht ausgeführt werden: {str(e)}",
                     horizontalalignment='center', verticalalignment='center')
            print(f"t-SNE-Fehler: {str(e)}. Überspringe Visualisierung.")

        plt.tight_layout()
        plt.savefig(f'{model_name}_feature_visualization.png')
        plt.close()

    def _plot_scalability(self, scalability_results, model_name):
        """Erstellt Plots für die Skalierbarkeitsanalyse"""
        plt.figure(figsize=(12, 5))

        # Plot für verschiedene Datengrößen
        plt.subplot(1, 2, 1)
        sizes = list(scalability_results['data_sizes'].keys())
        times = list(scalability_results['data_sizes'].values())

        plt.plot(sizes, times, 'o-')
        plt.xlabel('Anteil der Daten')
        plt.ylabel('Trainingszeit (s)')
        plt.title('Trainingszeit vs. Datengröße')
        plt.grid(True)

        # Plot für verschiedene fehlende Werte
        plt.subplot(1, 2, 2)
        rates = list(scalability_results['missing_values'].keys())
        times = list(scalability_results['missing_values'].values())

        plt.plot(rates, times, 'o-')
        plt.xlabel('Anteil fehlender Werte')
        plt.ylabel('Trainingszeit (s)')
        plt.title('Trainingszeit vs. Fehlende Werte')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f'{model_name}_scalability.png')
        plt.close()

    def _create_comparison_plots(self, save_path):
        """Erstellt Vergleichsplots für alle Modelle"""
        # 1. Vergleich der Repräsentationsqualität (Downstream-Klassifikation)
        plt.figure(figsize=(15, 8))

        classifiers = ['LogisticRegression', 'RandomForest', 'SVM']
        metrics = ['accuracy', 'f1_score', 'auc']

        # Vorbereiten der Daten für den Plot
        model_names = list(self.results['representation_quality'].keys())
        data_by_clf_metric = {}

        for clf in classifiers:
            for metric in metrics:
                data_by_clf_metric[(clf, metric)] = []

                for model in model_names:
                    try:
                        value = self.results['representation_quality'][model]['classification'][clf][metric]
                        data_by_clf_metric[(clf, metric)].append(value)
                    except:
                        data_by_clf_metric[(clf, metric)].append(0)

        # Erstellen des Plots
        n_groups = len(model_names)
        bar_width = 0.1
        index = np.arange(n_groups)

        colors = plt.cm.tab10(np.linspace(0, 1, len(classifiers) * len(metrics)))
        color_idx = 0

        for clf in classifiers:
            for i, metric in enumerate(metrics):
                plt.bar(
                    index + (i + len(metrics) * classifiers.index(clf)) * bar_width - bar_width * len(
                        metrics) * len(classifiers) / 2,
                    data_by_clf_metric[(clf, metric)],
                    bar_width,
                    color=colors[color_idx],
                    label=f'{clf} - {metric}'
                )
                color_idx += 1

        plt.xlabel('Modell')
        plt.ylabel('Metrikwert')
        plt.title('Vergleich der Modelle - Downstream-Klassifikationsleistung')
        plt.xticks(index, model_names)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig(f'{save_path}/classification_comparison.png')
        plt.close()

        # 2. Vergleich der Recheneffizienz
        plt.figure(figsize=(14, 6))

        metrics = ['training_time', 'inference_time', 'memory_usage', 'parameter_count']

        # Normalisieren der Daten für bessere Visualisierung
        normalized_data = {metric: [] for metric in metrics}

        for metric in metrics:
            values = [self.results['computational_efficiency'][model][metric] for model in model_names]
            max_val = max(values) if max(values) > 0 else 1
            normalized_data[metric] = [v / max_val for v in values]

        # Erstellen des Plots
        width = 0.2
        x = np.arange(len(model_names))

        for i, metric in enumerate(metrics):
            plt.bar(
                x + (i - len(metrics) / 2 + 0.5) * width,
                normalized_data[metric],
                width,
                label=metric
            )

        plt.xlabel('Modell')
        plt.ylabel('Normalisierter Wert')
        plt.title('Vergleich der Recheneffizienz (normalisiert)')
        plt.xticks(x, model_names)
        plt.legend()
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig(f'{save_path}/efficiency_comparison.png')
        plt.close()

        # 3. Detaillierte Vergleiche als Radar-Charts
        self._create_radar_plots(save_path)

    def _create_radar_plots(self, save_path):
        """Erstellt Radar-Charts für einen schnellen visuellen Vergleich der Modelle"""
        model_names = list(self.model_results.keys())

        # Definiere Kategorien und Metriken für das Radar-Chart
        categories = [
            # Repräsentationsqualität
            'Classification Acc',
            'Classification F1',
            'Silhouette Score',
            # Recheneffizienz
            'Training Speed',  # Inverse der Trainingszeit
            'Inference Speed',  # Inverse der Inferenzzeit
            'Memory Efficiency',  # Inverse des Speicherverbrauchs
            # Skalierbarkeit
            'Data Size Scaling',  # Inverse der Trainingszeit bei voller Datengröße
            'Missing Value Handling',  # Inverse der Trainingszeit mit 40% fehlenden Werten
            # Generalisierung
            'Generalization Acc',
            'Robustness'
        ]

        # Sammle die Daten für jedes Modell
        data = []
        for model in model_names:
            model_data = []
            results = self.model_results[model]

            # Repräsentationsqualität
            try:
                model_data.append(
                    results['representation_quality']['classification']['LogisticRegression']['accuracy'])
            except:
                model_data.append(0)

            try:
                model_data.append(
                    results['representation_quality']['classification']['LogisticRegression']['f1_score'])
            except:
                model_data.append(0)

            try:
                model_data.append(results['representation_quality']['clustering']['silhouette_score'])
            except:
                model_data.append(0)

            # Recheneffizienz - normalisiert (niedriger ist besser, daher inversiv)
            try:
                train_times = [r['computational_efficiency']['training_time'] for r in self.model_results.values()]
                max_train = max(train_times)
                model_data.append(1 - results['computational_efficiency']['training_time'] / max_train)
            except:
                model_data.append(0)

            try:
                inf_times = [r['computational_efficiency']['inference_time'] for r in self.model_results.values()]
                max_inf = max(inf_times)
                model_data.append(1 - results['computational_efficiency']['inference_time'] / max_inf)
            except:
                model_data.append(0)

            try:
                mem_usages = [r['computational_efficiency']['memory_usage'] for r in self.model_results.values()]
                max_mem = max(mem_usages)
                model_data.append(1 - results['computational_efficiency']['memory_usage'] / max_mem)
            except:
                model_data.append(0)

            # Skalierbarkeit
            try:
                data_size_times = [r['scalability']['data_sizes'][1.0] for r in self.model_results.values()]
                max_time = max(data_size_times)
                model_data.append(1 - results['scalability']['data_sizes'][1.0] / max_time)
            except:
                model_data.append(0)

            try:
                missing_times = [r['scalability']['missing_values'][0.4] for r in self.model_results.values()]
                max_miss = max(missing_times)
                model_data.append(1 - results['scalability']['missing_values'][0.4] / max_miss)
            except:
                model_data.append(0)

            # Generalisierung
            try:
                model_data.append(results['generalization']['base_performance']['accuracy'])
            except:
                model_data.append(0)

            try:
                model_data.append(results['generalization']['robustness']['accuracy_mean'])
            except:
                model_data.append(0)

            data.append(model_data)

        # Erstelle Radar-Chart
        num_vars = len(categories)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Schließe den Plot

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        # Hinzufügen von Labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)

        # Zeichnen für jedes Modell
        for i, model in enumerate(model_names):
            values = data[i]
            values += values[:1]  # Schließe den Plot

            ax.plot(angles, values, linewidth=2, linestyle='solid', label=model)
            ax.fill(angles, values, alpha=0.1)

        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title('Vergleich aller Modelle nach Metriken')

        plt.tight_layout()
        plt.savefig(f'{save_path}/radar_comparison.png')
        plt.close()

    def _save_results(self, save_path):
        """Speichert die Benchmark-Ergebnisse als CSV und JSON"""
        import json
        import numpy as np

        # Numpy-Arrays und spezielle Datentypen in Python-Standard-Typen konvertieren
        def convert_for_json(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (set, tuple)):
                return list(obj)
            elif hasattr(obj, 'tolist'):
                return obj.tolist()
            else:
                return obj

        # Rekursiv durch die Ergebnisse gehen und konvertieren
        def deep_convert(obj):
            if isinstance(obj, dict):
                return {k: deep_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [deep_convert(item) for item in obj]
            else:
                return convert_for_json(obj)

        # Konvertierte Ergebnisse für JSON vorbereiten
        json_results = deep_convert(self.results)

        # Speichern der Ergebnisse als JSON für leichte Wiederverwendung
        try:
            with open(f'{save_path}/benchmark_results.json', 'w') as f:
                json.dump(json_results, f, indent=4)
            print(f"JSON-Ergebnisse gespeichert in: {save_path}/benchmark_results.json")
        except Exception as e:
            print(f"Fehler beim Speichern der JSON-Ergebnisse: {e}")

        # Erstellen einer zusammenfassenden CSV-Datei
        try:
            summary_data = []
            model_names = list(self.model_results.keys())

            for model in model_names:
                row = {'model': model}

                # Repräsentationsqualität
                try:
                    row['clf_accuracy'] = float(
                        self.model_results[model]['representation_quality']['classification']['LogisticRegression'][
                            'accuracy'])
                    row['clf_f1'] = float(
                        self.model_results[model]['representation_quality']['classification']['LogisticRegression'][
                            'f1_score'])
                    row['silhouette'] = float(
                        self.model_results[model]['representation_quality']['clustering']['silhouette_score'])
                except:
                    pass

                # Recheneffizienz
                try:
                    row['train_time'] = float(
                        self.model_results[model]['computational_efficiency']['training_time'])
                    row['inference_time'] = float(
                        self.model_results[model]['computational_efficiency']['inference_time'])
                    row['memory_usage'] = float(
                        self.model_results[model]['computational_efficiency']['memory_usage'])
                    row['param_count'] = int(
                        self.model_results[model]['computational_efficiency']['parameter_count'])
                except:
                    pass

                # Generalisierung
                try:
                    row['generalization_acc'] = float(
                        self.model_results[model]['generalization']['base_performance']['accuracy'])
                    row['robustness_acc'] = float(
                        self.model_results[model]['generalization']['robustness']['accuracy_mean'])
                except:
                    pass

                summary_data.append(row)

            # Erstelle DataFrame und speichere als CSV
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(f'{save_path}/benchmark_summary.csv', index=False)
            print(f"CSV-Zusammenfassung gespeichert in: {save_path}/benchmark_summary.csv")
        except Exception as e:
            print(f"Fehler beim Erstellen der CSV-Zusammenfassung: {e}")

        # Erstelle eine detailliertere HTML-Berichtsdatei
        try:
            self._create_html_report(save_path)
            print(f"HTML-Bericht gespeichert in: {save_path}/benchmark_report.html")
        except Exception as e:
            print(f"Fehler beim Erstellen des HTML-Berichts: {e}")

    def _create_html_report(self, save_path):
        """Erstellt einen HTML-Bericht mit interaktiven Visualisierungen"""
        model_names = list(self.model_results.keys())

        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Benchmark-Ergebnisse für unüberwachte Modelle</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2, h3 { color: #333; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .section { margin-bottom: 30px; }
                .image-container { display: flex; flex-wrap: wrap; justify-content: center; }
                .image-container img { margin: 10px; max-width: 100%; box-shadow: 0 0 5px rgba(0,0,0,0.2); }
                .summary { background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
            </style>
        </head>
        <body>
            <h1>Benchmark-Ergebnisse für unüberwachte Modelle</h1>

            <div class="summary">
                <h2>Zusammenfassung</h2>
                <p>Dieser Bericht enthält die Benchmark-Ergebnisse für die folgenden unüberwachten Lernmodelle:</p>
                <ul>
        """

        for model in model_names:
            html_content += f"<li>{model}</li>\n"

        html_content += """
                </ul>
                <p>Die Modelle wurden anhand der folgenden Kriterien bewertet:</p>
                <ul>
                    <li><strong>Repräsentationsqualität:</strong> Downstream-Klassifikation, Clustering-Metriken, Rekonstruktionsfehler</li>
                    <li><strong>Recheneffizienz:</strong> Trainingszeit, Inferenzzeit, Speicherverbrauch</li>
                    <li><strong>Skalierbarkeit:</strong> Leistung mit zunehmender Datengröße, Umgang mit fehlenden Werten</li>
                    <li><strong>Generalisierung:</strong> Robustheit, Patientenübergreifende Leistung</li>
                </ul>
            </div>

            <div class="section">
                <h2>Modellvergleiche</h2>
                <div class="image-container">
                    <img src="radar_comparison.png" alt="Radar-Vergleich der Modelle">
                    <img src="classification_comparison.png" alt="Klassifikationsvergleich">
                    <img src="efficiency_comparison.png" alt="Effizienzvergleich">
                </div>
            </div>

            <div class="section">
                <h2>Detaillierte Ergebnisse</h2>
        """

        # Tabellen für jede Bewertungskategorie

        # 1. Repräsentationsqualität
        html_content += """
                <h3>Repräsentationsqualität</h3>
                <table>
                    <tr>
                        <th>Modell</th>
                        <th>LR Accuracy</th>
                        <th>LR F1-Score</th>
                        <th>RF Accuracy</th>
                        <th>RF F1-Score</th>
                        <th>SVM Accuracy</th>
                        <th>SVM F1-Score</th>
                        <th>Silhouette Score</th>
                        <th>Davies-Bouldin</th>
                        <th>Calinski-Harabasz</th>
                        <th>Rekonstruktionsfehler</th>
                    </tr>
        """

        for model in model_names:
            html_content += f"<tr><td>{model}</td>"

            try:
                lr_acc = \
                    self.model_results[model]['representation_quality']['classification']['LogisticRegression'][
                        'accuracy']
                lr_f1 = self.model_results[model]['representation_quality']['classification']['LogisticRegression'][
                    'f1_score']
                html_content += f"<td>{lr_acc:.4f}</td><td>{lr_f1:.4f}</td>"
            except:
                html_content += "<td>-</td><td>-</td>"

            try:
                rf_acc = self.model_results[model]['representation_quality']['classification']['RandomForest'][
                    'accuracy']
                rf_f1 = self.model_results[model]['representation_quality']['classification']['RandomForest'][
                    'f1_score']
                html_content += f"<td>{rf_acc:.4f}</td><td>{rf_f1:.4f}</td>"
            except:
                html_content += "<td>-</td><td>-</td>"

            try:
                svm_acc = self.model_results[model]['representation_quality']['classification']['SVM']['accuracy']
                svm_f1 = self.model_results[model]['representation_quality']['classification']['SVM']['f1_score']
                html_content += f"<td>{svm_acc:.4f}</td><td>{svm_f1:.4f}</td>"
            except:
                html_content += "<td>-</td><td>-</td>"

            try:
                silhouette = self.model_results[model]['representation_quality']['clustering']['silhouette_score']
                db = self.model_results[model]['representation_quality']['clustering']['davies_bouldin_score']
                ch = self.model_results[model]['representation_quality']['clustering']['calinski_harabasz_score']
                html_content += f"<td>{silhouette:.4f}</td><td>{db:.4f}</td><td>{ch:.4f}</td>"
            except:
                html_content += "<td>-</td><td>-</td><td>-</td>"

            try:
                recon_err = self.model_results[model]['representation_quality']['reconstruction_error']
                if recon_err is not None:
                    html_content += f"<td>{recon_err:.6f}</td>"
                else:
                    html_content += "<td>-</td>"
            except:
                html_content += "<td>-</td>"

            html_content += "</tr>\n"

        html_content += """
                </table>

                <h3>Recheneffizienz</h3>
                <table>
                    <tr>
                        <th>Modell</th>
                        <th>Trainingszeit (s)</th>
                        <th>Inferenzzeit (s)</th>
                        <th>Speicherverbrauch (MB)</th>
                        <th>Anzahl Parameter</th>
                    </tr>
        """

        for model in model_names:
            html_content += f"<tr><td>{model}</td>"

            try:
                train_time = self.model_results[model]['computational_efficiency']['training_time']
                inf_time = self.model_results[model]['computational_efficiency']['inference_time']
                mem = self.model_results[model]['computational_efficiency']['memory_usage']
                params = self.model_results[model]['computational_efficiency']['parameter_count']

                html_content += f"<td>{train_time:.2f}</td><td>{inf_time:.6f}</td><td>{mem:.2f}</td><td>{params}</td>"
            except:
                html_content += "<td>-</td><td>-</td><td>-</td><td>-</td>"

            html_content += "</tr>\n"

        html_content += """
                </table>

                <h3>Generalisierung</h3>
                <table>
                    <tr>
                        <th>Modell</th>
                        <th>Basis Accuracy</th>
                        <th>Basis F1-Score</th>
                        <th>Patientenweise Accuracy</th>
                        <th>Patientenweise F1-Score</th>
                        <th>Robustheit Accuracy</th>
                        <th>Robustheit F1-Score</th>
                    </tr>
        """

        for model in model_names:
            html_content += f"<tr><td>{model}</td>"

            try:
                base_acc = self.model_results[model]['generalization']['base_performance']['accuracy']
                base_f1 = self.model_results[model]['generalization']['base_performance']['f1']
                pat_acc = f"{self.model_results[model]['generalization']['patient_performance']['accuracy_mean']:.4f} ± {self.model_results[model]['generalization']['patient_performance']['accuracy_std']:.4f}"
                pat_f1 = f"{self.model_results[model]['generalization']['patient_performance']['f1_mean']:.4f} ± {self.model_results[model]['generalization']['patient_performance']['f1_std']:.4f}"
                rob_acc = f"{self.model_results[model]['generalization']['robustness']['accuracy_mean']:.4f} ± {self.model_results[model]['generalization']['robustness']['accuracy_std']:.4f}"
                rob_f1 = f"{self.model_results[model]['generalization']['robustness']['f1_mean']:.4f} ± {self.model_results[model]['generalization']['robustness']['f1_std']:.4f}"

                html_content += f"<td>{base_acc:.4f}</td><td>{base_f1:.4f}</td><td>{pat_acc}</td><td>{pat_f1}</td><td>{rob_acc}</td><td>{rob_f1}</td>"
            except:
                html_content += "<td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td>"

            html_content += "</tr>\n"

        html_content += """
                </table>
            </div>

            <div class="section">
                <h2>Visualisierungen der einzelnen Modelle</h2>
                <div class="image-container">
        """

        for model in model_names:
            html_content += f"""
                    <div>
                        <h3>{model}</h3>
                        <img src="{model}_feature_visualization.png" alt="{model} Feature-Visualisierung">
                        <img src="{model}_scalability.png" alt="{model} Skalierbarkeit">
                    </div>
            """

        html_content += """
                </div>
            </div>

            <div class="section">
                <h2>Fazit und Empfehlungen</h2>
                <p>Basierend auf den Benchmark-Ergebnissen können folgende Empfehlungen gegeben werden:</p>
                <ul>
                    <li><strong>Beste Repräsentationsqualität:</strong> [Wird basierend auf den Ergebnissen ausgefüllt]</li>
                    <li><strong>Beste Recheneffizienz:</strong> [Wird basierend auf den Ergebnissen ausgefüllt]</li>
                    <li><strong>Beste Skalierbarkeit:</strong> [Wird basierend auf den Ergebnissen ausgefüllt]</li>
                    <li><strong>Beste Generalisierung:</strong> [Wird basierend auf den Ergebnissen ausgefüllt]</li>
                </ul>
                <p>Die Auswahl des optimalen Modells hängt von den spezifischen Anforderungen des Anwendungsfalls ab.</p>
            </div>

            <footer>
                <p>Benchmark-Bericht erstellt am [Datum]</p>
            </footer>
        </body>
        </html>
        """

        with open(f'{save_path}/benchmark_report.html', 'w') as f:
            f.write(html_content)


# Code für die Ausführung des Benchmarks
def run_benchmark(ts_tcc_model, autoencoder, imiss_model, signature_model, statfeat_model, rgan_model,
                  X_train_pad, X_test_pad, y_train, y_test,
                  mask_train, mask_test, patient_ids_train, patient_ids_test,
                  tstcc_train_loader, rauto_train_loader, rauto_batches, device):
    """
    Führt ein vollständiges Benchmark für alle sechs Modelle aus
    """
    print("\n==== Starte Benchmark-Verfahren ====")

    # Benchmark-Objekt
    benchmark = ModelBenchmark()

    # Modell-Informationen
    model_names = ['TS-TCC', 'RecurrentAutoencoder', 'InformativeMissingness', 'SignatureTransform',
                   'StatisticalFeature', 'RGAN']
    models = [ts_tcc_model, autoencoder, imiss_model, signature_model, statfeat_model, rgan_model]

    # Modell-Builder sind Funktionen, die neue Instanzen der Modelle erstellen
    def build_tstcc(input_dim):
        return TS_TCC(input_dim).to(device)

    def build_rauto(input_dim):
        return RecurrentAutoencoder(input_dim).to(device)

    def build_imiss(input_dim):
        return InformativeMissingnessAutoencoder(input_dim).to(device)

    def build_signature(input_dim):
        return SignatureTransformClustering(truncation_level=3, n_clusters=5)

    def build_statfeat(input_dim):
        return StatisticalFeatureRepresentation(use_pca=True, n_components=10, n_clusters=5)

    def build_rgan(input_dim):
        # Verwende BenchmarkRGAN statt normaler RGAN
        return BenchmarkRGAN(input_dim=input_dim, hidden_dim=128, n_layers=2, dropout=0.1, device=device)

    model_builders = [build_tstcc, build_rauto, build_imiss, build_signature, build_statfeat, build_rgan]
    model_types = ['tstcc', 'rauto', 'imiss', 'signature', 'statfeat', 'rgan']

    # DataLoader
    train_loaders = [tstcc_train_loader, rauto_train_loader, None, None, None, None]
    length_batches = [None, rauto_batches, rauto_batches, rauto_batches, rauto_batches, rauto_batches]

    # Ausführen des Benchmarks
    results, model_results = benchmark.evaluate_all(
        model_names, models, model_builders,
        X_train_pad, X_test_pad, y_train, y_test,
        mask_train, mask_test, patient_ids_train, patient_ids_test,
        train_loaders, length_batches, device, model_types,
        save_path='benchmark_results'
    )

    print("\n==== Benchmark abgeschlossen ====")
    print(f"Ergebnisse wurden im Verzeichnis 'benchmark_results' gespeichert.")

    return results, model_results


# Hauptprogramm
if __name__ == "__main__":
    # ... [Code zum Laden und Vorbereiten der Daten]

    # Nach dem Training der Modelle:
    if TRAIN_TSTCC and TRAIN_RAUTO and TRAIN_IMISS and TRAIN_SIGNATURE and TRAIN_STATFEAT and TRAIN_RGAN:
        print("\n==== Starte Benchmark aller sechs Modelle ====")

        # Dataset für das Benchmarking
        test_dataset = SimpleDataset(X_test_pad, mask_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Künstliche Patienten-IDs für die Generalisierungstests erstellen
        # Einfache Zuweisung: jede 5. Sequenz gehört zum selben Patienten
        unique_ids = np.array(range(len(padded_sequences) // 5 + 1))
        patient_ids = np.repeat(unique_ids, 5)[:len(padded_sequences)]

        # Aufteilen in train/test entsprechend der Daten
        patient_indices_train = []
        patient_indices_test = []

        for i, seq in enumerate(padded_sequences):
            for j, train_seq in enumerate(X_train_pad):
                if np.array_equal(seq, train_seq):
                    patient_indices_train.append(i)
                    break

            for j, test_seq in enumerate(X_test_pad):
                if np.array_equal(seq, test_seq):
                    patient_indices_test.append(i)
                    break

        patient_ids_train = patient_ids[patient_indices_train]
        patient_ids_test = patient_ids[patient_indices_test]

        # Ausführen des Benchmarks für alle sechs Modelle
        benchmark_results, model_details = run_benchmark(
            ts_tcc_model, autoencoder, imiss_model, signature_model, statfeat_model, rgan_model,
            X_train_pad, X_test_pad, y_train, y_test,
            mask_train, mask_test, patient_ids_train, patient_ids_test,
            tstcc_train_loader, rauto_train_loader, rauto_batches,
            device
        )

        print("\nBenchmark-Zusammenfassung:")
        for model in benchmark_results['representation_quality']:
            print(f"\n{model}:")

            # Klassifikationsleistung
            try:
                clf_metrics = benchmark_results['representation_quality'][model]['classification']['LogisticRegression']
                print(f"  - Klassifikation: Accuracy={clf_metrics['accuracy']:.4f}, F1={clf_metrics['f1_score']:.4f}")
            except:
                print("  - Klassifikationsmetriken nicht verfügbar")

            # Recheneffizienz
            try:
                comp_metrics = benchmark_results['computational_efficiency'][model]
                print(f"  - Trainingszeit: {comp_metrics['training_time']:.2f}s")
                print(f"  - Inferenzzeit: {comp_metrics['inference_time']:.6f}s")
            except:
                print("  - Effizienzmetriken nicht verfügbar")
    else:
        print("Bitte aktivieren Sie alle sechs Modelle, um ein vollständiges Benchmark durchzuführen.")

