import os
import sys
import torch
import numpy as np
from pathlib import Path

# Import necessary modules - now using package imports
from .vqvae_model import VQVAEModel
from .protein_chain import WrappedProteinChain
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from esm.utils.constants import esm3 as C
import hydra
import omegaconf

def load_aminoseed_model(checkpoint_path, device='cuda'):
    """Load the AminoAseed model from checkpoint"""
    
    # Load the model configuration
    model_cfg = {
        "quantizer": {
            "quantizer_type": "StraightThroughQuantizer",
            "loss_weight": {
                "commitment_loss_weight": 0.25,
                "quantization_loss_weight": 1.0,
                "reconstruction_loss_weight": 1.0
            },
            "codebook_size": 512,
            "codebook_embed_size": 1024,
            "_need_init": False,
            "freeze_codebook": True,
            "use_linear_project": True  # True for AminoAseed
        },
        "encoder": {
            "d_model": 1024,
            "n_heads": 1,
            "v_heads": 128,
            "n_layers": 2,
            "d_out": 1024
        },
        "decoder": {
            "encoder_d_out": 1024,
            "d_model": 1024,
            "n_heads": 16,
            "n_layers": 8
        }
    }
    
    # Convert dict to OmegaConf for compatibility
    model_cfg = omegaconf.OmegaConf.create(model_cfg)
    
    # Initialize model
    model = VQVAEModel(model_cfg=model_cfg)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model state dict
    if "module" in checkpoint:
        state_dict = checkpoint["module"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    
    # Remove 'model.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v
    
    # Load weights
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()
    
    return model

def prepare_pdb_input(pdb_path, chain_id='A', device='cuda'):
    """Load and prepare PDB file for model input"""
    
    # Load protein chain
    if pdb_path.endswith('.pdb'):
        pdb_chain = WrappedProteinChain.from_pdb(pdb_path)
    else:  # .cif file
        pdb_chain = WrappedProteinChain.from_cif(pdb_path, chain_id=chain_id)
    
    print(f"Loaded protein with {len(pdb_chain.sequence)} residues")
    print(f"Sequence: {pdb_chain.sequence[:50]}..." if len(pdb_chain.sequence) > 50 else f"Sequence: {pdb_chain.sequence}")
    
    # Get structure encoder inputs with normalization
    coords, plddt, residue_index = pdb_chain.to_structure_encoder_inputs(
        device, 
        should_normalize_coordinates=True
    )
    
    # Check for missing atoms (NaN values are expected for missing atoms)
    nan_mask = torch.isnan(coords[0, :, :, 0])  # Check first coordinate of each atom
    if nan_mask.any():
        print(f"Note: Found {nan_mask.sum().item()} atoms with missing coordinates (this is normal)")
    
    # Create attention mask
    attention_mask = coords[:, :, 0, 0] == torch.inf
    
    # Prepare sequence tokens
    seq_tokenizer = EsmSequenceTokenizer()
    sequence = pdb_chain.sequence.replace(C.MASK_STR_SHORT, "<mask>")
    seq_ids = seq_tokenizer.encode(sequence, add_special_tokens=False)
    seq_ids = torch.tensor(seq_ids, dtype=torch.int64, device=device).unsqueeze(0)
    
    print(f"Valid residues for processing: {(~attention_mask).sum().item()}/{coords.shape[1]}")
    
    return coords, attention_mask, residue_index, seq_ids, pdb_chain

def encode_structure(model, coords, attention_mask, residue_index, seq_ids, pdb_chain):
    """Encode structure to tokens and continuous representations"""
    
    with torch.no_grad():
        # Prepare input
        input_list = (coords, attention_mask, residue_index, seq_ids, [pdb_chain])
        
        # Get tokens and representations
        quantized_reprs, quantized_indices, continuous_reprs = model(
            input_list, use_as_tokenizer=True
        )
        
    return quantized_reprs, quantized_indices, continuous_reprs

def decode_tokens(model, quantized_reprs, structure_tokens, attention_mask, sequence_id=None):
    """Decode tokens back to structure"""
    
    if sequence_id is None:
        sequence_id = torch.zeros_like(structure_tokens, dtype=torch.int64)
    
    with torch.no_grad():
        # Decode using the model's decoder
        decoded_states = model.decoder.decode(
            quantized_reprs, 
            structure_tokens, 
            attention_mask, 
            sequence_id
        )
    
    return decoded_states

def analyze_results(decoded_states, pdb_chain, decoded_coords, quantized_reprs, quantized_indices, continuous_reprs):
    """Analyze and print detailed results"""
    
    print("\n=== Reconstruction Results ===")
    
    # Calculate actual RMSD
    from .protein_chain import WrappedProteinChain
    decoded_chain = WrappedProteinChain.from_backbone_atom_coordinates(
        decoded_coords[0].cpu()
    )
    decoded_chain = decoded_chain[:len(pdb_chain)]
    bb_rmsd = decoded_chain.rmsd(pdb_chain, only_compute_backbone_rmsd=True)
    
    print(f"Backbone RMSD: {bb_rmsd:.3f} Å")
    
    # Model's predicted confidence scores
    print("\n=== Model's Predicted Confidence Scores ===")
    print(f"Predicted LDDT (pLDDT): {decoded_states['plddt'][0].mean().item():.3f}")
    print(f"  - Range: [{decoded_states['plddt'][0].min().item():.3f}, {decoded_states['plddt'][0].max().item():.3f}]")
    
    print(f"\nPredicted TM-score (pTM): {decoded_states['ptm'].item():.3f}")
    
    # Analyze PAE if available
    if 'predicted_aligned_error' in decoded_states and decoded_states['predicted_aligned_error'] is not None:
        pae = decoded_states['predicted_aligned_error'][0]
        print(f"\nPredicted Aligned Error (PAE):")
        print(f"  - Mean: {pae.mean().item():.2f} Å")
        print(f"  - Range: [{pae.min().item():.2f}, {pae.max().item():.2f}] Å")
    
    # === NEW: Analyze Continuous Representations ===
    print(f"\n=== Continuous Representations Analysis ===")
    print(f"Shape: {continuous_reprs.shape}")
    print(f"Data type: {continuous_reprs.dtype}")
    print(f"Device: {continuous_reprs.device}")
    
    # Statistics
    continuous_flat = continuous_reprs[0].cpu().numpy()  # [L, D] -> flatten to analyze
    print(f"Statistics:")
    print(f"  - Mean: {continuous_flat.mean():.4f}")
    print(f"  - Std: {continuous_flat.std():.4f}")
    print(f"  - Min: {continuous_flat.min():.4f}")
    print(f"  - Max: {continuous_flat.max():.4f}")
    
    # Per-residue analysis
    residue_norms = torch.norm(continuous_reprs[0], dim=1).cpu().numpy()
    print(f"Per-residue L2 norms:")
    print(f"  - Mean norm: {residue_norms.mean():.4f}")
    print(f"  - Std norm: {residue_norms.std():.4f}")
    print(f"  - Range: [{residue_norms.min():.4f}, {residue_norms.max():.4f}]")
    
    # === NEW: Analyze Quantized Representations ===
    print(f"\n=== Quantized Representations Analysis ===")
    print(f"Shape: {quantized_reprs.shape}")
    print(f"Data type: {quantized_reprs.dtype}")
    
    # Statistics
    quantized_flat = quantized_reprs[0].cpu().numpy()
    print(f"Statistics:")
    print(f"  - Mean: {quantized_flat.mean():.4f}")
    print(f"  - Std: {quantized_flat.std():.4f}")
    print(f"  - Min: {quantized_flat.min():.4f}")
    print(f"  - Max: {quantized_flat.max():.4f}")
    
    # Quantization loss analysis
    quantization_error = torch.nn.functional.mse_loss(quantized_reprs, continuous_reprs, reduction='none')
    per_residue_error = quantization_error.mean(dim=-1)[0].cpu().numpy()
    total_error = quantization_error.mean().item()
    
    print(f"Quantization Error (MSE):")
    print(f"  - Total MSE: {total_error:.6f}")
    print(f"  - Per-residue MSE mean: {per_residue_error.mean():.6f}")
    print(f"  - Per-residue MSE std: {per_residue_error.std():.6f}")
    print(f"  - Worst residue error: {per_residue_error.max():.6f} (position {per_residue_error.argmax()})")
    print(f"  - Best residue error: {per_residue_error.min():.6f} (position {per_residue_error.argmin()})")
    
    # === NEW: Analyze Discrete Tokens ===
    print(f"\n=== Discrete Token Analysis ===")
    tokens = quantized_indices[0].cpu().numpy()
    print(f"Token sequence length: {len(tokens)}")
    print(f"Token range: [{tokens.min()}, {tokens.max()}]")
    print(f"Unique tokens used: {len(np.unique(tokens))}/512 possible tokens")
    
    # Calculate token entropy
    import math
    token_counts = np.bincount(tokens)
    token_probs = token_counts / len(tokens)
    token_probs = token_probs[token_probs > 0]  # Remove zeros
    entropy = -sum(p * math.log2(p) for p in token_probs)  # Use math.log2 directly
    print(f"Token entropy: {entropy:.3f} bits")
    
    # Most common tokens - handle numpy unique more carefully
    unique_tokens = np.unique(tokens)
    counts = np.array([np.sum(tokens == token) for token in unique_tokens])
    sorted_idx = np.argsort(counts)[::-1]
    print(f"Most frequent tokens:")
    for i in range(min(5, len(unique_tokens))):
        token_id = unique_tokens[sorted_idx[i]]
        count = counts[sorted_idx[i]]
        freq = count / len(tokens) * 100
        print(f"  - Token {token_id}: {count} times ({freq:.1f}%)")
    
    # Token transition analysis
    transitions = {}
    for i in range(len(tokens)-1):
        curr, next_tok = tokens[i], tokens[i+1]
        if curr not in transitions:
            transitions[curr] = {}
        if next_tok not in transitions[curr]:
            transitions[curr][next_tok] = 0
        transitions[curr][next_tok] += 1
    
    print(f"Token transitions: {len(transitions)} different starting tokens have transitions")
    
    # === NEW: Compare Representations ===
    print(f"\n=== Representation Comparison ===")
    
    # Cosine similarity between continuous and quantized
    continuous_norm = torch.nn.functional.normalize(continuous_reprs[0], dim=1)
    quantized_norm = torch.nn.functional.normalize(quantized_reprs[0], dim=1)
    cosine_sim = torch.sum(continuous_norm * quantized_norm, dim=1).cpu().numpy()
    
    print(f"Cosine similarity (continuous vs quantized):")
    print(f"  - Mean: {cosine_sim.mean():.4f}")
    print(f"  - Std: {cosine_sim.std():.4f}")
    print(f"  - Min: {cosine_sim.min():.4f} (position {cosine_sim.argmin()})")
    print(f"  - Max: {cosine_sim.max():.4f} (position {cosine_sim.argmax()})")
    
    # Token usage statistics
    print(f"\n=== Tokenization Statistics ===")
    print(f"Number of structure tokens: {decoded_states['last_hidden_state'].shape[1]}")
    print(f"Decoder hidden state shape: {decoded_states['last_hidden_state'].shape}")
    
    return bb_rmsd

def save_decoded_structure(decoded_coords, original_pdb_chain, output_path):
    """Save decoded coordinates as PDB file"""
    
    # Three letter amino acid codes
    aa_3letter = {
        'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE',
        'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU',
        'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
        'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR',
        'X': 'UNK'
    }
    
    # Create a simple PDB writer
    with open(output_path, 'w') as f:
        f.write("REMARK   Generated by AminoAseed decoder\\n")
        atom_idx = 1
        for i, residue in enumerate(original_pdb_chain.sequence):
            res_idx = original_pdb_chain.residue_index[i]
            res_3letter = aa_3letter.get(residue, 'UNK')
            
            # Write backbone atoms
            for atom_name, atom_idx_in_res in [('N', 0), ('CA', 1), ('C', 2)]:
                x, y, z = decoded_coords[0, i, atom_idx_in_res].cpu().numpy()
                f.write(f"ATOM  {atom_idx:5d}  {atom_name:<3s} {res_3letter:3s} A{res_idx:4d}    "
                       f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {atom_name[0]}\\n")
                atom_idx += 1
        
        f.write("END\\n")

def main():
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Update these paths to your actual files
    checkpoint_path = "struct_token_bench_release_ckpt/codebook_512x1024-1e+19-linear-fixed-last.ckpt/checkpoint/mp_rank_00_model_states.pt"
    pdb_path = "/hpc/mydata/kyle.hippe/data/pdb/9XIM.pdb"  # Change to your PDB file
    output_path = "decoded_structure.pdb"
    
    # Check if files exist
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return
    
    if not os.path.exists(pdb_path):
        print(f"Error: PDB file not found at {pdb_path}")
        return
    
    print("=== AminoAseed Structure Tokenization Demo ===\\n")
    
    print("Loading AminoAseed model...")
    model = load_aminoseed_model(checkpoint_path, device)
    
    print("\\nLoading and preparing PDB file...")
    coords, attention_mask, residue_index, seq_ids, pdb_chain = prepare_pdb_input(
        pdb_path, chain_id='A', device=device
    )
    
    print("\\nEncoding structure to tokens...")
    quantized_reprs, quantized_indices, continuous_reprs = encode_structure(
        model, coords, attention_mask, residue_index, seq_ids, pdb_chain
    )
    
    print(f"\\nTokenization complete:")
    print(f"  - Generated {quantized_indices.shape[1]} discrete tokens")
    print(f"  - Continuous representation shape: {continuous_reprs.shape}")
    print(f"  - Sample tokens: {quantized_indices[0][:20].cpu().numpy()}...")
    
    print("\\nDecoding tokens back to structure...")
    decoded_states = decode_tokens(
        model, quantized_reprs, quantized_indices, attention_mask
    )
    
    # Extract decoded backbone coordinates
    decoded_coords = decoded_states["bb_pred"]

    # Analyze results
    _bb_rmsd = analyze_results(decoded_states, pdb_chain, decoded_coords, quantized_reprs, quantized_indices, continuous_reprs)
    
    print(f"\\nSaving decoded structure to: {output_path}")
    save_decoded_structure(decoded_coords, pdb_chain, output_path)
    
if __name__ == "__main__":
    main()