"""Command-line interface for the original ``main_T2MRec.py`` entry point."""

import argparse

from .config import DEFAULTS


def build_parser() -> argparse.ArgumentParser:
    defaults = DEFAULTS
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--embedding_type', type=str, default=defaults.embedding_type,
                        choices=['tfidf', 'sbert'])
    parser.add_argument('--sbert_model', type=str,
                        default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--fit_all_mcp_for_embedding', type=int, default=1,
                        help='1=fit TF-IDF vocabulary with all MCP texts; 0=train-linked MCPs only')
    parser.add_argument('--tfidf_max_features', type=int, default=50000)
    parser.add_argument('--tfidf_encode_batch_size', type=int, default=512)
    parser.add_argument('--alpha_semantic', type=float, default=defaults.semantic_weight)
    parser.add_argument('--alpha_struct', type=float, default=defaults.structural_weight)
    parser.add_argument('--epochs', type=int, default=defaults.epochs)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use_llm_selfcheck', type=int, default=1)
    parser.add_argument('--topk1', type=int, default=defaults.anchor_k)
    parser.add_argument('--topk2', type=int, default=defaults.expansion_k,
                        help='number of additional candidates retrieved by centroid expansion')
    parser.add_argument('--final_k', type=int, default=defaults.final_k,
                        help='size of the final recommendation list')
    parser.add_argument('--use_two_tower', type=int, default=1)
    parser.add_argument('--lr', type=float, default=defaults.learning_rate)
    parser.add_argument('--hidden_dim', type=int, default=defaults.hidden_dim)
    parser.add_argument('--out_dim', type=int, default=defaults.output_dim)
    parser.add_argument('--num_layers', type=int, default=defaults.num_layers)
    parser.add_argument('--dropout', type=float, default=defaults.dropout)
    parser.add_argument('--loss_type', type=str, default='contrastive',
                        choices=['bce', 'contrastive'])
    parser.add_argument('--temperature', type=float, default=defaults.temperature)
    parser.add_argument('--batch_size', type=int, default=defaults.batch_size)
    parser.add_argument('--num_neg', type=int, default=10,
                        help='number of sampled negatives per positive for optional BCE training')
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not 0.0 <= args.alpha_semantic <= 1.0 or not 0.0 <= args.alpha_struct <= 1.0:
        parser.error('fusion weights must be in [0, 1]')
    if abs((args.alpha_semantic + args.alpha_struct) - 1.0) > 1e-8:
        parser.error('--alpha_semantic and --alpha_struct must sum to 1')
    if min(args.topk1, args.topk2, args.final_k) <= 0:
        parser.error('--topk1, --topk2, and --final_k must be positive')
    if args.final_k > args.topk1 + args.topk2:
        parser.error('--final_k cannot exceed the expanded candidate-pool size')
