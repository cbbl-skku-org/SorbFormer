import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModel,
    T5EncoderModel,
)
import selfies as sf
import logging
from pathlib import Path
from tqdm import tqdm
import hashlib
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class MolecularFeatureExtractor(nn.Module):
    """
    Unified wrapper for extracting molecular features from two pretrained models:
    - QizhiPei/biot5-plus-base (SELFIES-based T5 encoder)
    - DeepChem/ChemBERTa-77M-MLM (SMILES-based RoBERTa-style)

    All models are frozen (gradients disabled) and used for feature extraction only.
    """

    MODEL_CONFIGS = {
        "biot5-plus-base": {
            "hf_path": "QizhiPei/biot5-plus-base",
            "type": "t5_encoder",
            "input_format": "selfies",
            "feature_dim": 768,
            "pool_strategy": "mean",
            "max_length": 512,
            "add_special_tokens": True,
            "special_token_format": "<bom>{}<eom>",
            "trust_remote_code": False,
        },
        "chemberta-77m-mlm": {
            "hf_path": "DeepChem/ChemBERTa-77M-MLM",
            "type": "bert",
            "input_format": "smiles",
            "feature_dim": 384,
            "pool_strategy": "cls",
            "max_length": 512,
            "add_special_tokens": False,
            "special_token_format": "{}",
            "trust_remote_code": False,
        },
    }

    def __init__(
        self,
        model_name="chemberta-77m-mlm",
        device=None,
        cache_dir=None,
        output_hidden_states=False,
    ):
        """
        Initialize the feature extractor.

        Args:
            model_name (str): Key from MODEL_CONFIGS
            device (str): 'cuda', 'cpu', or None (auto-detect)
            cache_dir (str): Directory to cache extracted features
            output_hidden_states (bool): Return all hidden states (for layer analysis)
        """
        super().__init__()

        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(
                f"Model '{model_name}' not supported. "
                f"Available models: {list(self.MODEL_CONFIGS.keys())}"
            )

        self.config = self.MODEL_CONFIGS[model_name]
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.output_hidden_states = output_hidden_states

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"Loading {model_name} ({self.config['hf_path']})...")

        # Load tokenizer
        tokenizer_kwargs = (
            {"trust_remote_code": True} if self.config["trust_remote_code"] else {}
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["hf_path"], **tokenizer_kwargs
        )

        # Load model based on type
        model_kwargs = {}
        if self.config["trust_remote_code"]:
            model_kwargs["trust_remote_code"] = True

        if self.config["type"] == "t5_encoder":
            self.model = T5EncoderModel.from_pretrained(
                self.config["hf_path"], **model_kwargs
            )
        elif self.config["type"] == "bert":
            self.model = AutoModel.from_pretrained(
                self.config["hf_path"], **model_kwargs
            )
        else:
            raise ValueError(f"Unknown model type: {self.config['type']}")

        # Move to requested device or CPU by default
        target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(target_device)
        self.model.eval()

        # Freeze all parameters initially
        for param in self.model.parameters():
            param.requires_grad = False

        logging.info("Model loaded and frozen successfully.")
        logging.info(f"Feature dimension: {self.config['feature_dim']}")

    def _preprocess_input(self, smiles_list):
        """
        Convert SMILES to required format (SMILES/SELFIES) with proper formatting.

        Returns:
            processed_list: List of formatted strings
            failed_indices: List of indices where conversion failed
        """
        processed = []
        failed_indices = []

        for idx, smi in enumerate(smiles_list):
            try:
                # Format conversion
                if self.config["input_format"] == "selfies":
                    val = sf.encoder(smi)
                    if val is None or val == "":
                        logging.warning(
                            f"Failed to convert SMILES to SELFIES at index {idx}: '{smi}'"
                        )
                        val = "[C]"  # Fallback to methane
                        failed_indices.append(idx)
                else:
                    val = smi

                # Add special tokens if needed (e.g., BioT5+ uses <bom>...<eom>)
                if self.config["add_special_tokens"]:
                    special_format = self.config["special_token_format"]
                    val = special_format.format(val)

                processed.append(val)

            except Exception as e:
                logging.error(f"Error processing molecule at index {idx} '{smi}': {e}")
                if self.config["input_format"] == "selfies":
                    processed.append("[C]")
                else:
                    processed.append("C")
                failed_indices.append(idx)

        if failed_indices:
            logging.warning(
                f"Total failed conversions: {len(failed_indices)}/{len(smiles_list)}"
            )

        return processed, failed_indices

    def _get_cache_key(self, smiles_list):
        """Generate unique hash for caching"""
        smiles_str = "||".join(sorted(smiles_list))
        hash_obj = hashlib.md5(smiles_str.encode())
        return f"{self.model_name}_{hash_obj.hexdigest()}.pt"

    @torch.no_grad()
    def extract_features(
        self,
        smiles_list,
        batch_size=32,
        normalize=False,
        show_progress=False,
        use_cache=False,
    ):
        """
        Extract molecular features from SMILES strings.

        Args:
            smiles_list (list): List of SMILES strings
            batch_size (int): Batch size for inference
            normalize (bool): L2-normalize features
            show_progress (bool): Show progress bar
            use_cache (bool): Use cached features if available

        Returns:
            torch.Tensor: Features of shape [N, feature_dim]
        """
        # Check cache
        if use_cache and self.cache_dir:
            cache_file = self.cache_dir / self._get_cache_key(smiles_list)
            if cache_file.exists():
                logging.info(f"Loading cached features from {cache_file}")
                return torch.load(cache_file, map_location="cpu")

        # Preprocess
        input_texts, failed_idx = self._preprocess_input(smiles_list)
        num_samples = len(input_texts)

        # Pre-allocate output tensor
        all_features = torch.zeros(num_samples, self.config["feature_dim"])

        # Batch processing
        iterator = range(0, num_samples, batch_size)
        if show_progress:
            iterator = tqdm(
                iterator,
                total=(num_samples + batch_size - 1) // batch_size,
                desc=f"Extracting {self.model_name} features",
            )

        for i in iterator:
            batch = input_texts[i : i + batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config["max_length"],
            ).to(self.model.device)

            # Forward pass
            if self.config["type"] == "t5_encoder":
                outputs = self.model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    return_dict=True,
                    output_hidden_states=self.output_hidden_states,
                )
                hidden_states = outputs.last_hidden_state

            elif self.config["type"] == "bert":
                outputs = self.model(
                    **inputs,
                    return_dict=True,
                    output_hidden_states=self.output_hidden_states,
                )
                hidden_states = outputs.last_hidden_state

            # Pooling strategy
            mask = inputs.attention_mask.unsqueeze(-1).float()

            pooled = self._pool_features(hidden_states, mask)

            # Write to pre-allocated tensor
            batch_end = min(i + batch_size, num_samples)
            all_features[i:batch_end] = pooled.to(all_features.device)

        # Normalize if requested
        if normalize:
            all_features = torch.nn.functional.normalize(all_features, p=2, dim=1)

        # Cache
        if use_cache and self.cache_dir:
            cache_file = self.cache_dir / self._get_cache_key(smiles_list)
            torch.save(all_features, cache_file)
            logging.info(f"Saved features to cache: {cache_file}")

        return all_features

    def _pool_features(self, hidden_states, mask):
        """Helper to pool internal states dynamically"""
        if self.config["pool_strategy"] == "mean":
            sum_hidden = torch.sum(hidden_states * mask, dim=1)
            sum_mask = torch.sum(mask, dim=1).clamp(min=1e-9)
            return sum_hidden / sum_mask
        elif self.config["pool_strategy"] == "cls":
            return hidden_states[:, 0, :]
        return hidden_states

    def unfreeze_layers(self, unfreeze_all=False, num_layers_to_unfreeze=None):
        """Unfreeze the LLM for fine-tuning.
        
        By default, unfreezes only the last N transformer blocks:
          - BioT5/MolT5 (T5 encoder, 12 blocks): last 3 blocks
          - ChemBERTa (RoBERTa, 3 layers): last 1 layer
        
        Args:
            unfreeze_all: If True, unfreeze all parameters (full fine-tuning).
            num_layers_to_unfreeze: Override the default number of layers to unfreeze.
        """
        if unfreeze_all:
            for param in self.model.parameters():
                param.requires_grad = True
            logging.info("Fully unfroze all language model parameters for fine-tuning.")
        else:
            # Architecture-specific selective unfreezing
            model_type = self.config["type"]
            
            if model_type == "t5_encoder":
                # T5 encoder: blocks named "encoder.block.N"
                all_blocks = [n for n, _ in self.model.named_modules()
                              if n.startswith("encoder.block.") and n.count(".") == 2]
                n_unfreeze = num_layers_to_unfreeze or 3
                blocks_to_unfreeze = set(all_blocks[-n_unfreeze:])
                
                unfrozen_count = 0
                for name, param in self.model.named_parameters():
                    # Check if this param belongs to one of the blocks to unfreeze
                    # Also unfreeze the final layer norm
                    if any(name.startswith(block) for block in blocks_to_unfreeze) or \
                       "encoder.final_layer_norm" in name:
                        param.requires_grad = True
                        unfrozen_count += 1
                
                logging.info(f"Unfroze last {n_unfreeze} T5 encoder blocks "
                           f"({unfrozen_count} params): {sorted(blocks_to_unfreeze)}")
                           
            elif model_type == "bert":
                # RoBERTa/BERT: layers named "encoder.layer.N"
                all_layers = [n for n, _ in self.model.named_modules()
                              if n.startswith("encoder.layer.") and n.count(".") == 2]
                n_unfreeze = num_layers_to_unfreeze or 1
                layers_to_unfreeze = set(all_layers[-n_unfreeze:])
                
                unfrozen_count = 0
                for name, param in self.model.named_parameters():
                    if any(name.startswith(layer) for layer in layers_to_unfreeze) or \
                       "pooler" in name:
                        param.requires_grad = True
                        unfrozen_count += 1
                
                logging.info(f"Unfroze last {n_unfreeze} BERT/RoBERTa layers "
                           f"({unfrozen_count} params): {sorted(layers_to_unfreeze)}")
            else:
                logging.warning(f"Unknown model type '{model_type}', unfreezing all.")
                for param in self.model.parameters():
                    param.requires_grad = True

        self.model.train()

    @torch.no_grad()
    def extract_sequence_features(self, smiles_list, max_length=None):
        """
        Extract token-level sequence features (not pooled) from SMILES strings.

        Unlike extract_features() which returns mean-pooled vectors (B, feature_dim),
        this returns the full sequence of hidden states so that sequential models
        (e.g., BiLSTM, RNN) can process them.

        Args:
            smiles_list (list): List of SMILES strings
            max_length (int): Optional max sequence length override

        Returns:
            hidden_states: (B, max_seq_len, feature_dim) - padded token embeddings
            lengths: (B,) - actual sequence lengths (before padding)
        """
        input_texts, failed_idx = self._preprocess_input(smiles_list)
        max_len = max_length or self.config["max_length"]

        inputs = self.tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        ).to(self.model.device)

        if self.config["type"] == "t5_encoder":
            outputs = self.model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                return_dict=True,
            )
        elif self.config["type"] == "bert":
            outputs = self.model(
                **inputs,
                return_dict=True,
            )

        hidden_states = outputs.last_hidden_state  # (B, seq_len, feature_dim)
        lengths = inputs.attention_mask.sum(dim=1)  # (B,)

        return hidden_states, lengths

    @property
    def device(self):
        """Dynamic device property based on model parameters."""
        return next(self.model.parameters()).device

    def get_model_info(self):
        """Return model configuration information"""
        return {
            "model_name": self.model_name,
            "hf_path": self.config["hf_path"],
            "input_format": self.config["input_format"],
            "feature_dim": self.config["feature_dim"],
            "pool_strategy": self.config["pool_strategy"],
            "max_length": self.config["max_length"],
            "device": str(self.device),
        }


# ============================================================================ #
# USAGE EXAMPLES                                                               #
# ============================================================================ #

if __name__ == "__main__":
    molecules = [
        "CCO",  # Ethanol
        "c1ccccc1",  # Benzene
        "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
    ]

    print("=" * 80)
    print("Molecular Feature Extraction - Model Comparison")
    print("=" * 80)

    # Test only the two models you care about
    models_to_test = [
        "chemberta-77m-mlm",
        "biot5-plus-base",
    ]

    for model_name in models_to_test:
        print(f"\n{'='*80}")
        print(f"Testing: {model_name}")
        print(f"{'='*80}")

        try:
            extractor = MolecularFeatureExtractor(
                model_name=model_name,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )

            # Extract features
            features = extractor.extract_features(
                molecules, batch_size=4, show_progress=True, normalize=True
            )

            # Display results
            info = extractor.get_model_info()
            print(f"\nModel Info:")
            print(f"  - HuggingFace Path: {info['hf_path']}")
            print(f"  - Input Format: {info['input_format']}")
            print(f"  - Feature Dimension: {info['feature_dim']}")
            print(f"  - Pooling Strategy: {info['pool_strategy']}")
            print(f"\nExtracted Features Shape: {features.shape}")
            print(f"Feature Statistics:")
            print(f"  - Mean: {features.mean():.4f}")
            print(f"  - Std: {features.std():.4f}")
            print(f"  - Min: {features.min():.4f}")
            print(f"  - Max: {features.max():.4f}")

            # Compute pairwise similarities
            similarities = torch.mm(features, features.t())
            print(f"\nPairwise Cosine Similarities:")
            print(similarities.numpy().round(3))

            # Clean up
            del extractor
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"ERROR testing {model_name}: {e}")
            import traceback

            traceback.print_exc()

    print(f"\n{'='*80}")
    print("All tests completed!")
    print(f"{'='*80}")
