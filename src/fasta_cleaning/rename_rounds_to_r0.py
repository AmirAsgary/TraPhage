import sys
import os

def rename_rounds(r0_file, round_files):
    print(f"Loading Master Dictionary from {r0_file}...")
    
    # Dictionary to hold the mapping: Sequence (String) -> Master ID (String)
    seq_to_id = {}
    
    # 1. Parse R0 and build the mapping dictionary
    with open(r0_file, 'r') as f:
        current_id = None
        seq_parts = []
        
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id and seq_parts:
                    seq_to_id["".join(seq_parts).upper()] = current_id
                    seq_parts.clear()
                
                # Extract the base ID from the R0 header (e.g., ">sequence15_10" -> "sequence15")
                # Split by '_' and take the first part, removing the '>'
                header_parts = line.split('_')
                if header_parts:
                    current_id = header_parts[0][1:] 
            else:
                seq_parts.append(line)
                
        # Catch the very last sequence in the R0 file
        if current_id and seq_parts:
            seq_to_id["".join(seq_parts).upper()] = current_id

    print(f"Built mapping for {len(seq_to_id)} unique sequences from R0.")
    

    # 2. Process each Round file (K files)
    for r_file in round_files:
        # Create a safe output filename so we don't overwrite your raw data
        output_file = r_file.replace('.fasta', '_renamed.fasta')
        if output_file == r_file:
            output_file = r_file + ".renamed"
            
        print(f"Processing {r_file} -> Saving to {output_file}...")
        
        with open(r_file, 'r') as f_in, open(output_file, 'w') as f_out:
            current_count = "1" # Fallback count
            seq_parts = []
            
            for line in f_in:
                line = line.strip()
                if line.startswith(">"):
                    if seq_parts:
                        seq = "".join(seq_parts).upper()
                        # Look up the R0 ID; if somehow missing, flag it
                        r0_id = seq_to_id.get(seq, "UNKNOWN_SEQ") 
                        
                        # Write the unified header and the sequence
                        f_out.write(f">{r0_id}_{current_count}\n{seq}\n")
                        seq_parts.clear()
                    
                    # Extract the round-specific count from the current file's header
                    # e.g., ">sequence99_450" -> "450"
                    header_parts = line.split('_')
                    if len(header_parts) > 1:
                        current_count = header_parts[1]
                    else:
                        current_count = "1"
                else:
                    seq_parts.append(line)
                    
            # Catch the very last sequence in the round file
            if seq_parts:
                seq = "".join(seq_parts).upper()
                r0_id = seq_to_id.get(seq, "UNKNOWN_SEQ")
                f_out.write(f">{r0_id}_{current_count}\n{seq}\n")

    print("All files renamed successfully! Your pipeline is fully unified.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python rename_rounds_to_r0.py <path_to_R0.fasta> <path_to_R1.fasta> [path_to_R2.fasta ...]")
        sys.exit(1)
    
    r0 = sys.argv[1]
    rounds = sys.argv[2:]
    rename_rounds(r0, rounds)