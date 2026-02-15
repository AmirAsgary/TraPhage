import sys
import os
import re

def map_rounds(r0_file, k_files):
    print(f"Loading {r0_file} into memory...")
    r0_seqs = set()
    max_id = 0
    
    # 1. Parse R0 to get all known sequences and the highest sequence ID
    with open(r0_file, 'r') as f:
        seq_parts = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq_parts:
                    r0_seqs.add("".join(seq_parts).upper())
                    seq_parts.clear()
                
                # Extract the sequence number to know where to continue
                m = re.search(r'>sequence(\d+)', line, re.IGNORECASE)
                if m:
                    max_id = max(max_id, int(m.group(1)))
            else:
                seq_parts.append(line)
        
        # Catch the very last sequence in R0
        if seq_parts:
            r0_seqs.add("".join(seq_parts).upper())

    # Fallback if no '>sequenceN' headers exist yet
    if max_id == 0:
        max_id = len(r0_seqs)

    print(f"Loaded {len(r0_seqs)} unique sequences from R0.")
    print(f"Highest sequence ID found: {max_id}. New sequences will start at {max_id + 1}.")

    # 2. Check if R0 needs a newline at the EOF to prevent gluing headers to sequences
    needs_newline = False
    if os.path.getsize(r0_file) > 0:
        with open(r0_file, 'rb') as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b'\n':
                needs_newline = True

    # 3. Process K files and append missing sequences
    next_id = max_id + 1
    added_count = 0

    # Open R0 in Append mode ('a')
    with open(r0_file, 'a') as out_f:
        if needs_newline:
            out_f.write('\n')

        for k_file in k_files:
            print(f"Scanning {k_file}...")
            with open(k_file, 'r') as f:
                seq_parts = []
                for line in f:
                    line = line.strip()
                    if line.startswith(">"):
                        if seq_parts:
                            seq = "".join(seq_parts).upper()
                            # If sequence is completely novel:
                            if seq not in r0_seqs:
                                out_f.write(f">sequence{next_id}_0.5\n{seq}\n")
                                r0_seqs.add(seq) # Add to set to prevent adding duplicates later
                                next_id += 1
                                added_count += 1
                            seq_parts.clear()
                    else:
                        seq_parts.append(line)
                
                # Catch the last sequence in the K file
                if seq_parts:
                    seq = "".join(seq_parts).upper()
                    if seq not in r0_seqs:
                        out_f.write(f">sequence{next_id}_0.5\n{seq}\n")
                        r0_seqs.add(seq)
                        next_id += 1
                        added_count += 1

    print(f"Done! Appended {added_count} new unique sequences to {r0_file}.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python map_rounds_to_r0.py <path_to_R0.fasta> <path_to_K1.fasta> [path_to_K2.fasta ...]")
        sys.exit(1)
    
    r0 = sys.argv[1]
    ks = sys.argv[2:]
    map_rounds(r0, ks)