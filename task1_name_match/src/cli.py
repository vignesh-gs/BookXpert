# CLI: interactive (python -m src.cli) or --name "Geetha B.S" --top_k 5, --json, --data path.

import argparse
import json
import sys
from pathlib import Path

from src.index import NameIndex
from src.matcher import NameMatcher


def print_welcome():
    """Banner for interactive mode."""
    print("\n" + "=" * 60)
    print("       NAME MATCHING SYSTEM - Interactive Mode")
    print("=" * 60)
    print("\nThis system finds the most similar names from a dataset.")
    print("\nCommands:")
    print("  - Type a name to search (e.g., 'Geetha B.S')")
    print("  - Type 'top N' to change number of results (e.g., 'top 10')")
    print("  - Type 'quit' or 'exit' to exit")
    print("=" * 60 + "\n")


def print_human_readable(result: dict) -> None:
    """Pretty-print match result."""
    print(f"\n{'='*60}")
    print(f"Query: \"{result['query']}\"")
    print(f"Normalized: \"{result['query_normalized']}\"")
    print(f"{'='*60}")
    
    if result.get('error'):
        print(f"\nError: {result['error']}")
        return
    
    if not result['best_match']:
        print("\nNo matches found.")
        return
    
    best = result['best_match']
    print(f"\nBest Match: \"{best['name']}\" (Score: {best['score']:.1f})")
    
    print(f"\n{'─'*60}")
    print("Top Matches:")
    print(f"{'─'*60}")
    print(f"{'Rank':<5} {'Name':<30} {'Score':<8} {'First':<6} {'Edit':<6} {'Init':<6}")
    print(f"{'─'*60}")
    
    for i, match in enumerate(result['matches'], 1):
        name = match['name'][:28] + '..' if len(match['name']) > 30 else match['name']
        bd = match['breakdown']
        print(f"{i:<5} {name:<30} {match['score']:<8.1f} {bd['first_name_score']:<6.1f} {bd['edit_distance_score']:<6.1f} {bd['initials_score']:<6.1f}")
    
    print(f"{'─'*60}")
    
    # Show breakdown for best match
    print(f"\nBreakdown for best match \"{best['name']}\":")
    bd = best['breakdown']
    print(f"  First Name Score:    {bd['first_name_score']:>6.1f}")
    print(f"  Edit Distance Score: {bd['edit_distance_score']:>6.1f}")
    print(f"  Other Core Score:    {bd['other_core_score']:>6.1f}")
    print(f"  Initials Score:      {bd['initials_score']:>6.1f}")
    print(f"  Phonetic Core Score: {bd['phonetic_core_score']:>6.1f}")
    print(f"  Full String Score:   {bd['full_string_score']:>6.1f}")
    print(f"  ─────────────────────────────")
    
    penalties = []
    if bd['missing_initial_penalty'] > 0:
        penalties.append(f"Missing initials: -{bd['missing_initial_penalty']:.1f}")
    if bd['extra_initial_penalty'] > 0:
        penalties.append(f"Extra initials: -{bd['extra_initial_penalty']:.1f}")
    if bd['missing_core_penalty'] > 0:
        penalties.append(f"Missing cores: -{bd['missing_core_penalty']:.1f}")
    if bd['overlong_penalty'] > 0:
        penalties.append(f"Overlong: -{bd['overlong_penalty']:.1f}")
    if bd['length_diff_penalty'] > 0:
        penalties.append(f"Length diff: -{bd['length_diff_penalty']:.1f}")
    
    if penalties:
        print(f"  Penalties: {', '.join(penalties)}")
    else:
        print(f"  Penalties: None")
    
    print(f"  ─────────────────────────────")
    print(f"  Final Score:         {bd['final_score']:>6.1f}")
    
    # Show token info
    print(f"\n  Query cores:     {bd['query_cores']}")
    print(f"  Candidate cores: {bd['candidate_cores']}")
    print(f"  Query initials:  {bd['query_initials']}")
    print(f"  Cand. initials:  {bd['candidate_initials']}")
    
    if bd['missing_initials']:
        print(f"  Missing:         {bd['missing_initials']}")


def resolve_data_path(data_arg: str) -> Path:
    """Resolve data path (script dir or cwd)."""
    data_path = Path(data_arg)
    if not data_path.is_absolute():
        # Try relative to script location first
        script_dir = Path(__file__).parent.parent
        candidate_path = script_dir / data_arg
        if candidate_path.exists():
            data_path = candidate_path
        else:
            # Try current directory
            data_path = Path(data_arg)
    return data_path


def load_index(data_path: Path) -> NameIndex:
    """Load index from CSV."""
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    index = NameIndex()
    count = index.load_from_csv(str(data_path))
    print(f"Loaded {count} names from {data_path.name}")
    return index


def run_interactive_mode(data_path: str = 'data/Indian_Names.csv'):
    """Interactive loop: prompt for name, show matches."""
    resolved_path = resolve_data_path(data_path)
    
    try:
        index = load_index(resolved_path)
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        sys.exit(1)
    
    matcher = NameMatcher(index)
    top_k = 5  # Default number of results
    
    print_welcome()
    
    while True:
        try:
            # Get user input
            user_input = input("\nEnter a name to search: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ('quit', 'exit', 'q'):
                print("\nThank you for using Name Matching System. Goodbye!")
                break
            
            # Check for empty input
            if not user_input:
                print("Please enter a name to search.")
                continue
            
            # Check for 'top N' command to change result count
            if user_input.lower().startswith('top '):
                try:
                    new_top_k = int(user_input.split()[1])
                    if new_top_k < 1:
                        print("Number of results must be at least 1.")
                    else:
                        top_k = new_top_k
                        print(f"Now showing top {top_k} results.")
                except (ValueError, IndexError):
                    print("Usage: top N (e.g., 'top 10')")
                continue
            
            # Run the matcher
            result = matcher.match(user_input, top_k)
            result_dict = result.to_dict()
            
            # Handle errors
            if result.error:
                print(f"\nError: {result.error}")
                continue
            
            # Print results
            print_human_readable(result_dict)
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except EOFError:
            print("\n\nEnd of input. Goodbye!")
            break


def run_command_line_mode(args):
    """One-shot match from --name, output to stdout or JSON."""
    data_path = resolve_data_path(args.data)
    
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}", file=sys.stderr)
        sys.exit(1)
    
    # Load index
    try:
        index = NameIndex()
        index.load_from_csv(str(data_path))
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Run matcher
    matcher = NameMatcher(index)
    result = matcher.match(args.name, args.top_k)
    result_dict = result.to_dict()
    
    # Handle errors
    if result.error:
        if args.json:
            print(json.dumps(result_dict, indent=2))
        else:
            print(f"Error: {result.error}", file=sys.stderr)
        sys.exit(2)
    
    # Output results
    if args.json:
        print(json.dumps(result_dict, indent=2))
    else:
        print_human_readable(result_dict)
    
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description='Name Matching System - Find similar names from a dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Interactive mode (prompts for input):
    python -m src.cli
    python -m src.cli --data data/names.csv
  
  Command-line mode:
    python -m src.cli --name "Geetha" --top_k 5
    python -m src.cli --name "Aaditya" --json
    python -m src.cli --name "Krishna" --data data/names.csv
        """
    )
    
    parser.add_argument(
        '--name', '-n',
        required=False,
        default=None,
        help='Name to search for (if not provided, runs in interactive mode)'
    )
    
    parser.add_argument(
        '--top_k', '-k',
        type=int,
        default=5,
        help='Number of top matches to return (default: 5)'
    )
    
    parser.add_argument(
        '--data', '-d',
        default='data/Indian_Names.csv',
        help='Path to names CSV file (default: data/Indian_Names.csv)'
    )
    
    parser.add_argument(
        '--json', '-j',
        action='store_true',
        help='Output in JSON format (only for command-line mode)'
    )
    
    args = parser.parse_args()
    
    # If no name provided, run interactive mode
    if args.name is None:
        run_interactive_mode(args.data)
    else:
        run_command_line_mode(args)


if __name__ == '__main__':
    main()
