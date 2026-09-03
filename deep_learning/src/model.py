import torch
import torch.nn as nn
import numpy as np
from typing import Dict
from lightning import LightningModule
from torch.nn import HuberLoss
from src.metrics import METRIC_DICT
import json
from captum.attr import IntegratedGradients


class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class CrossAttentionFusion(nn.Module):
    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value):
        attn_out, attn_weights = self.multihead_attn(query, key_value, key_value)
        return self.norm(query + self.dropout(attn_out)), attn_weights


class BilinearFusion(nn.Module):
    def __init__(self, dim1, dim2, out_dim, dropout):
        super().__init__()
        self.bilinear = nn.Bilinear(dim1, dim2, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2):
        fused = self.bilinear(x1, x2)
        return self.dropout(self.norm(fused))


class GatedFusion(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LeakyReLU(0.01),
            nn.Linear(dim, 2),
            nn.Softmax(dim=1),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2):
        combined = torch.cat([x1, x2], dim=1)
        gates = self.gate(combined)  # (B, 2)
        out = gates[:, 0:1] * x1 + gates[:, 1:2] * x2
        return self.dropout(out), gates



class PredictionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, num_layers=3):
        super().__init__()
        layers = []
        curr_dim = input_dim
        
        # We always want at least 2 layers (one hidden, one output)
        # But we'll support 2, 3, 4 layers
        for i in range(num_layers - 1):
            out_dim = hidden_dim // (2**i) if i > 0 else hidden_dim
            layers.append(LinearBlock(curr_dim, out_dim, dropout))
            curr_dim = out_dim
            
        layers.append(nn.Linear(curr_dim, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x).squeeze(-1)


class PlantUptakeModel(LightningModule):
    def __init__(self, args, molecular_extractor):
        super().__init__()
        self.args = args
        self.target_name = args.target  # Store target name
        self.save_hyperparameters(args)
        self.molecular_extractor = molecular_extractor

        with open(args.metadata_file, "r") as f:
            self.metadata = json.load(f)

        if args.loss_type == "huber":
            self.loss_fn = HuberLoss(delta=args.huber_delta)
        else:
            raise ValueError(f"Unknown loss: {args.loss_type}")

        self.metric_list = list(METRIC_DICT.keys())
        self.modality_names = []

        # ── 1. Molecular branch ──
        if args.use_molecule:
            self.modality_names.append("molecule")
            if getattr(self.args, "finetune_llm", False):
                self.molecular_extractor.unfreeze_layers(
                    unfreeze_all=getattr(self.args, "unfreeze_all_llm", False),
                    num_layers_to_unfreeze=getattr(self.args, "num_layers_to_unfreeze", None)
                )
                self.llm = self.molecular_extractor.model
            
            bottleneck_dim = getattr(self.args, "bottleneck_dim", None)
            if bottleneck_dim is not None and bottleneck_dim > 0 and bottleneck_dim < args.mol_dim:
                self.mol_encoder = nn.Sequential(
                    LinearBlock(args.mol_dim, bottleneck_dim, args.dropout),
                    LinearBlock(bottleneck_dim, args.hidden_dim, args.dropout),
                )
            else:
                self.mol_encoder = nn.Sequential(
                    LinearBlock(args.mol_dim, args.hidden_dim, args.dropout),
                )

        # ── 2. Conventional branch (simple concat-based) ──
        if args.use_conventional:
            self.modality_names.append("conventional")

            cat_cols = self.metadata["categorical_columns"]
            num_cols = self.metadata["numerical_columns"]

            # Categorical embeddings with variable dimensions
            self.cat_embeddings = nn.ModuleDict()
            total_cat_dim = 0
            for col in cat_cols:
                vocab_size = len(self.metadata["categorical_maps"][col])
                emb_dim = max(4, min(args.categorical_dim, vocab_size // 2 + 1))

                self.cat_embeddings[col] = nn.Embedding(
                    vocab_size,
                    emb_dim,
                    padding_idx=(
                        0
                        if 0 in self.metadata["categorical_maps"][col].values()
                        else None
                    ),
                )
                total_cat_dim += emb_dim

            # Numerical: project to a higher dimension to balance with categorical
            self.num_projection = nn.Sequential(
                nn.Linear(len(num_cols), args.hidden_dim // 2),
                nn.LayerNorm(args.hidden_dim // 2),
                nn.LeakyReLU(0.01),
            )

            # Conventional encoder: concat(cat_embeddings, num_projection) → hidden_dim
            conv_input_dim = total_cat_dim + (args.hidden_dim // 2)
            self.conventional_encoder = LinearBlock(
                conv_input_dim, args.hidden_dim, args.dropout
            )

        assert len(self.modality_names) > 0, "Must enable at least one modality"

        # ── 3. Fusion ──
        self.fusion_type = args.fusion_type
        num_modalities = len(self.modality_names)

        if num_modalities > 1:
            if self.fusion_type == "cross_attention":
                self.cross_attn = CrossAttentionFusion(
                    args.hidden_dim, args.num_heads, args.dropout
                )
                self.fusion_proj = LinearBlock(
                    2 * args.hidden_dim, args.hidden_dim, args.dropout
                )
            elif self.fusion_type == "bilinear":
                self.bilinear_fusion = BilinearFusion(
                    args.hidden_dim, args.hidden_dim, args.hidden_dim, args.dropout
                )
            elif self.fusion_type == "gated":
                self.gated_fusion = GatedFusion(args.hidden_dim, args.dropout)
                self.gate_l1_lambda = args.gate_l1_lambda

            # Always have a concat projection as fallback or for 'concat' mode
            self.concat_proj = LinearBlock(
                args.hidden_dim * num_modalities, args.hidden_dim, args.dropout
            )

        # ── 4. Prediction head ──
        self.prediction_head = PredictionHead(
            args.hidden_dim, args.hidden_dim // 2, args.dropout, num_layers=args.num_layers
        )

        self._init_weights()
        self._init_metrics()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Match LeakyReLU(0.01) slope
                nn.init.kaiming_normal_(
                    m.weight, a=0.01, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Bilinear):
                nn.init.kaiming_normal_(
                    m.weight, a=0.01, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.MultiheadAttention):
                for p in m.parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)
                if hasattr(m, "padding_idx") and m.padding_idx is not None:
                    with torch.no_grad():
                        m.weight[m.padding_idx].fill_(0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _init_metrics(self):
        for stage in ["train", "val", "test"]:
            # Standard metrics (or 'raw' metrics if RCF)
            prefix = "raw_" if self.target_name == "rcf" else ""
            for metric_name in self.metric_list:
                setattr(
                    self, f"{stage}/{prefix}{metric_name}", METRIC_DICT[metric_name]()
                )

            # Additional 'revert' metrics for RCF
            if self.target_name == "rcf":
                for metric_name in self.metric_list:
                    setattr(
                        self,
                        f"{stage}/revert_{metric_name}",
                        METRIC_DICT[metric_name](),
                    )

    def _update_metrics(self, stage, preds, targets):
        if self.target_name == "rcf":
            # Update raw metrics
            for metric_name in self.metric_list:
                getattr(self, f"{stage}/raw_{metric_name}").update(preds, targets)

            # Update revert metrics
            reverted_preds = torch.pow(10.0, preds)
            reverted_targets = torch.pow(10.0, targets)
            for metric_name in self.metric_list:
                getattr(self, f"{stage}/revert_{metric_name}").update(
                    reverted_preds, reverted_targets
                )
        else:
            # Update standard metrics for other targets (e.g. TSCF)
            for metric_name in self.metric_list:
                getattr(self, f"{stage}/{metric_name}").update(preds, targets)

    def forward(
        self,
        molecules,
        categorical_features,
        numerical_features,
        molecular_features=None,
        mixup_params=None,
    ):
        mol_emb = None
        conv_emb = None

        # ── 1. Molecular embedding ──
        if self.args.use_molecule:
            if getattr(self.args, "finetune_llm", False):
                # Tokenize incrementally in the batch
                input_texts, _ = self.molecular_extractor._preprocess_input(molecules)
                inputs = self.molecular_extractor.tokenizer(
                    input_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.molecular_extractor.config.get("max_length", 512),
                ).to(self.device)
                
                if self.molecular_extractor.config["type"] == "t5_encoder":
                    outputs = self.llm(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        return_dict=True,
                        output_hidden_states=self.molecular_extractor.output_hidden_states,
                    )
                    hidden_states = outputs.last_hidden_state
                elif self.molecular_extractor.config["type"] == "bert":
                    outputs = self.llm(
                        **inputs,
                        return_dict=True,
                        output_hidden_states=self.molecular_extractor.output_hidden_states,
                    )
                    hidden_states = outputs.last_hidden_state
                    
                mask = inputs.attention_mask.unsqueeze(-1).float()
                mol_features = self.molecular_extractor._pool_features(hidden_states, mask)
            else:
                if molecular_features is not None:
                    mol_features = molecular_features.to(self.device)
                else:
                    mol_features = self.molecular_extractor.extract_features(molecules).to(
                        self.device
                    )
            
            mol_emb = self.mol_encoder(mol_features)

            if mixup_params is not None:
                lam, index = mixup_params["lambda"], mixup_params["index"]
                mol_emb = lam * mol_emb + (1 - lam) * mol_emb[index]

        # ── 2. Conventional embedding ──
        if self.args.use_conventional:
            cat_cols = self.metadata["categorical_columns"]
            num_cols = self.metadata["numerical_columns"]

            parts = []

            # Categorical embeddings
            for col in cat_cols:
                x = categorical_features[f"cat_{col}"]
                e = self.cat_embeddings[col](x)  # (B, cat_dim)
                parts.append(e)

            # Numerical features -> projection
            num_values = []
            for col in num_cols:
                x = numerical_features[f"num_{col}"]
                if x.dim() == 1:
                    x = x.unsqueeze(1)
                num_values.append(x)

            if num_values:
                num_input = torch.cat(num_values, dim=1)
                num_projected = self.num_projection(num_input)
                parts.append(num_projected)

            # Concat all → encoder
            conv_input = torch.cat(parts, dim=1)
            conv_emb = self.conventional_encoder(conv_input)

            if mixup_params is not None:
                lam, index = mixup_params["lambda"], mixup_params["index"]
                conv_emb = lam * conv_emb + (1 - lam) * conv_emb[index]

        # ── 3. Fusion ──
        prediction, gates = self._fuse_and_predict(mol_emb, conv_emb)
        return prediction, gates

    def _fuse_and_predict(self, mol_emb, conv_emb):
        gates = None

        # Single modality
        if mol_emb is not None and conv_emb is None:
            fused = mol_emb
        elif conv_emb is not None and mol_emb is None:
            fused = conv_emb
        # Two modalities → fusion
        elif self.fusion_type == "cross_attention":
            attn_out, _ = self.cross_attn(mol_emb.unsqueeze(1), conv_emb.unsqueeze(1))
            mol_attended = attn_out.squeeze(1)
            fused = self.fusion_proj(torch.cat([mol_attended, conv_emb], dim=1))
            # Residual mol connection specifically for cross-attention
            fused = fused + mol_emb
        elif self.fusion_type == "bilinear":
            fused = self.bilinear_fusion(mol_emb, conv_emb)
        elif self.fusion_type == "gated":
            fused, gates = self.gated_fusion(mol_emb, conv_emb)
        else:
            # concat
            fused = self.concat_proj(torch.cat([mol_emb, conv_emb], dim=1))

        prediction = self.prediction_head(fused)
        return prediction, gates

    def get_modality_importance(self, batch):
        self.eval()
        with torch.no_grad():
            _, gates = self.forward(
                batch["smiles"],
                batch["categorical_feats"],
                batch["numerical_feats"],
                molecular_features=batch.get("molecular_feats"),
            )
        return gates.mean(dim=0).cpu().numpy() if gates is not None else None

    def explain_molecule(
        self, smiles_list, categorical_feats=None, numerical_feats=None, n_baselines=1
    ):
        self.eval()
        extractor = self.molecular_extractor
        device = next(extractor.model.parameters()).device

        input_texts, _ = extractor._preprocess_input(smiles_list)
        inputs = extractor.tokenizer(
            input_texts, return_tensors="pt", padding=True, truncation=True
        ).to(device)

        all_explanations = []

        def forward_func(embeds, attn_mask, d_conv):
            if extractor.config["type"] == "t5_encoder":
                outputs = extractor.model.encoder(
                    inputs_embeds=embeds, attention_mask=attn_mask, return_dict=True
                )
            else:
                outputs = extractor.model(
                    inputs_embeds=embeds, attention_mask=attn_mask, return_dict=True
                )

            hf = outputs.last_hidden_state
            mask = attn_mask.unsqueeze(-1).float()
            mol_features = torch.sum(hf * mask, dim=1) / torch.sum(mask, dim=1).clamp(
                min=1e-9
            )

            m_emb = self.mol_encoder(mol_features)
            pred, _ = self._fuse_and_predict(m_emb, d_conv)
            return pred

        ig = IntegratedGradients(forward_func)

        # Pre-compute conventional features if provided
        d_conv_context = None
        if self.args.use_conventional:
            if categorical_feats is not None and numerical_feats is not None:
                # Move to device and ensure batch dimension
                cat_f = {k: v.to(device) for k, v in categorical_feats.items()}
                num_f = {k: v.to(device) for k, v in numerical_feats.items()}

                # Encode (logic same as forward)
                cat_embs = [
                    self.cat_embeddings[col](cat_f[f"cat_{col}"])
                    for col in self.metadata["categorical_columns"]
                ]
                cat_out = torch.cat(cat_embs, dim=1)

                num_values = []
                for col in self.metadata["numerical_columns"]:
                    x = num_f[f"num_{col}"]
                    if x.dim() == 1:
                        x = x.unsqueeze(1)
                    num_values.append(x)
                num_input = torch.cat(num_values, dim=1)
                num_out = self.num_projection(num_input)

                d_conv_context = self.conventional_encoder(
                    torch.cat([cat_out, num_out], dim=1)
                )
            else:
                # Default to zeros if not provided
                d_conv_context = torch.zeros(
                    (len(smiles_list), self.args.hidden_dim), device=device
                )

        for i in range(len(smiles_list)):
            input_ids = inputs.input_ids[i : i + 1]
            attn_mask = inputs.attention_mask[i : i + 1]
            sample_d_conv = (
                d_conv_context[i : i + 1] if d_conv_context is not None else None
            )

            if extractor.config["type"] == "t5_encoder":
                embeds = extractor.model.get_input_embeddings()(input_ids)
            else:
                embeds = extractor.model.embeddings(input_ids)

            # Support multiple baselines for uncertainty
            baselines = [torch.zeros_like(embeds)]
            if n_baselines > 1:
                # Use random token embeddings as alternative baselines
                for _ in range(n_baselines - 1):
                    rand_ids = torch.randint(
                        0,
                        extractor.tokenizer.vocab_size,
                        input_ids.shape,
                        device=device,
                    )
                    if extractor.config["type"] == "t5_encoder":
                        baselines.append(
                            extractor.model.get_input_embeddings()(rand_ids)
                        )
                    else:
                        baselines.append(extractor.model.embeddings(rand_ids))

            multi_scores = []
            for base in baselines:
                attr = ig.attribute(
                    embeds,
                    baselines=base,
                    additional_forward_args=(attn_mask, sample_d_conv),
                    n_steps=200,
                )
                s = attr.sum(dim=-1).squeeze(0)
                s = s[: attn_mask.sum()].detach().cpu().numpy()
                multi_scores.append(s)

            tokens = extractor.tokenizer.convert_ids_to_tokens(
                input_ids[0, : attn_mask.sum()]
            )

            if n_baselines == 1:
                all_explanations.append(list(zip(tokens, multi_scores[0])))
            else:
                # Return list of scores per token: [(tok, [s1, s2, ...]), ...]
                all_explanations.append(
                    list(zip(tokens, np.stack(multi_scores, axis=1).tolist()))
                )

        return all_explanations

    def shared_step(self, batch: Dict, batch_idx: int, stage: str) -> torch.Tensor:
        use_mixup = stage == "train" and self.args.mixup_alpha > 0

        if use_mixup:
            lam = np.random.beta(self.args.mixup_alpha, self.args.mixup_alpha)
            batch_size = batch["targets"].size(0)
            index = torch.randperm(batch_size, device=self.device)

            mixup_params = {"lambda": lam, "index": index}
            mixed_targets = lam * batch["targets"] + (1 - lam) * batch["targets"][index]

            predictions, gates = self(
                batch["smiles"],
                batch["categorical_feats"],
                batch["numerical_feats"],
                molecular_features=batch.get("molecular_feats"),
                mixup_params=mixup_params,
            )
            loss = self.loss_fn(predictions, mixed_targets)

            with torch.no_grad():
                self.eval()
                clean_predictions, _ = self(
                    batch["smiles"],
                    batch["categorical_feats"],
                    batch["numerical_feats"],
                    molecular_features=batch.get("molecular_feats"),
                    mixup_params=None,
                )
                self.train()

            metric_target_preds = clean_predictions
        else:
            predictions, gates = self(
                batch["smiles"],
                batch["categorical_feats"],
                batch["numerical_feats"],
                molecular_features=batch.get("molecular_feats"),
                mixup_params=None,
            )
            loss = self.loss_fn(predictions, batch["targets"])
            metric_target_preds = predictions

        # Gate L1 regularization (for gated fusion only)
        if gates is not None and getattr(self, "gate_l1_lambda", 0) > 0:
            loss = loss + self.gate_l1_lambda * gates.abs().mean()

        batch_size = batch["targets"].size(0)
        self.log(
            f"{stage}/loss", loss, on_epoch=True, batch_size=batch_size, on_step=False
        )

        if gates is not None:
            mean_gates = gates.mean(dim=0)
            for i, modality_name in enumerate(self.modality_names):
                self.log(
                    f"{stage}/gate_{modality_name}",
                    mean_gates[i],
                    on_epoch=True,
                    on_step=False,
                    batch_size=batch_size,
                )

        self._update_metrics(stage, metric_target_preds, batch["targets"])

        return loss

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        return self.shared_step(batch, batch_idx, "train")

    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        return self.shared_step(batch, batch_idx, "val")

    def test_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        return self.shared_step(batch, batch_idx, "test")

    def on_train_epoch_end(self):
        self._log_metrics("train")

    def on_validation_epoch_end(self):
        self._log_metrics("val")

    def on_test_epoch_end(self):
        self._log_metrics("test")

    def _log_metrics(self, stage: str):
        prefixes = ["raw_", "revert_"] if self.target_name == "rcf" else [""]
        for prefix in prefixes:
            for metric_name in self.metric_list:
                metric_key = f"{stage}/{prefix}{metric_name}"
                metric = getattr(self, metric_key)
                try:
                    self.log(
                        metric_key,
                        metric.compute(),
                        prog_bar=(metric_name == "r2"),  # Progress bar for R2
                        on_epoch=True,
                        on_step=False,
                    )
                except (ValueError, RuntimeError):
                    pass
                metric.reset()

    def configure_optimizers(self) -> Dict:
        optimizer_grouped_parameters = []
        llm_params = []
        head_params = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "llm." in name or "molecular_extractor." in name:
                llm_params.append(param)
            else:
                head_params.append(param)
                
        if llm_params:
            optimizer_grouped_parameters.append(
                {"params": llm_params, "lr": getattr(self.args, "llm_lr", 1e-5), "weight_decay": self.args.weight_decay}
            )
        
        if head_params:
            optimizer_grouped_parameters.append(
                {"params": head_params, "lr": self.args.lr, "weight_decay": self.args.weight_decay}
            )

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.args.max_epochs,
            eta_min=self.args.lr_eta_min,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "frequency": 1,
                "interval": "epoch",
            },
        }

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
