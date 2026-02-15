mkdir -p data
# R0
mkdir -p data/R0/
vsearch --derep_fulllength ~/.project/dir.project/aastha/aastha_amir/non_aligned_fasta/R0_library/initial.singleline.fasta \
        --sizeout \
        --relabel sequence \
        --fasta_width 0 \
        --output - | sed 's/;size=/_/g' > data/R0/unique.fasta


# Bm75
mkdir -p data/Bm75/
vsearch --derep_fulllength ~/.project/dir.project/aastha/aastha_amir/non_aligned_fasta/Bm75/Bm75_1_S4_merged_constant_no_stop_aa.fasta.singleline.fasta \
        --sizeout \
        --relabel sequence \
        --fasta_width 0 \
        --output - | sed 's/;size=/_/g' > data/Bm75/R1_unique.fasta

vsearch --derep_fulllength ~/.project/dir.project/aastha/aastha_amir/non_aligned_fasta/Bm75/Bm75_3_S9_merged_constant_no_stop_aa.fasta.singleline.fasta \
        --sizeout \
        --relabel sequence \
        --fasta_width 0 \
        --output - | sed 's/;size=/_/g' > data/Bm75/R3_unique.fasta


# Bm76
mkdir -p data/Bm76/
vsearch --derep_fulllength ~/.project/dir.project/aastha/aastha_amir/non_aligned_fasta/Bm76/Bm76_1_S5_merged_constant_no_stop_aa.fasta.singleline.fasta \
        --sizeout \
        --relabel sequence \
        --fasta_width 0 \
        --output - | sed 's/;size=/_/g' > data/Bm76/R1_unique.fasta

vsearch --derep_fulllength ~/.project/dir.project/aastha/aastha_amir/non_aligned_fasta/Bm76/Bm76_3_S10_merged_constant_no_stop_aa.fasta.singleline.fasta \
        --sizeout \
        --relabel sequence \
        --fasta_width 0 \
        --output - | sed 's/;size=/_/g' > data/Bm76/R3_unique.fasta

# Re77
mkdir -p data/Re77/
vsearch --derep_fulllength ~/.project/dir.project/aastha/aastha_amir/non_aligned_fasta/Re77/Re77_1_S1_merged_constant_no_stop_aa.fasta.singleline.fasta \
        --sizeout \
        --relabel sequence \
        --fasta_width 0 \
        --output - | sed 's/;size=/_/g' > data/Re77/R1_unique.fasta

vsearch --derep_fulllength ~/.project/dir.project/aastha/aastha_amir/non_aligned_fasta/Re77/Re77_3_S6_merged_constant_no_stop_aa.fasta.singleline.fasta \
        --sizeout \
        --relabel sequence \
        --fasta_width 0 \
        --output - | sed 's/;size=/_/g' > data/Re77/R3_unique.fasta


# Re78
mkdir -p data/Re78/
vsearch --derep_fulllength ~/.project/dir.project/aastha/aastha_amir/non_aligned_fasta/Re78/Re78_1_S2_merged_constant_no_stop_aa.fasta.singleline.fasta \
        --sizeout \
        --relabel sequence \
        --fasta_width 0 \
        --output - | sed 's/;size=/_/g' > data/Re78/R1_unique.fasta

vsearch --derep_fulllength ~/.project/dir.project/aastha/aastha_amir/non_aligned_fasta/Re78/Re78_3_S7_merged_constant_no_stop_aa.fasta.singleline.fasta \
        --sizeout \
        --relabel sequence \
        --fasta_width 0 \
        --output - | sed 's/;size=/_/g' > data/Re78/R3_unique.fasta

# Re79
mkdir -p data/Re79/
vsearch --derep_fulllength ~/.project/dir.project/aastha/aastha_amir/non_aligned_fasta/Re79/Re79_1_S3_merged_constant_no_stop_aa.fasta.fasta \
        --sizeout \
        --relabel sequence \
        --fasta_width 0 \
        --output - | sed 's/;size=/_/g' > data/Re79/R1_unique.fasta

vsearch --derep_fulllength ~/.project/dir.project/aastha/aastha_amir/non_aligned_fasta/Re79/Re79_3_S8_merged_constant_no_stop_aa.fasta.singleline.fasta \
        --sizeout \
        --relabel sequence \
        --fasta_width 0 \
        --output - | sed 's/;size=/_/g' > data/Re79/R3_unique.fasta



##### Add undefined sequences from different rounds to R0 #####
python src/fasta_cleaning/map_rounds_to_r0.py data/R0/unique.fasta data/Bm75/R1_unique.fasta
python src/fasta_cleaning/map_rounds_to_r0.py data/R0/unique.fasta data/Bm75/R3_unique.fasta
python src/fasta_cleaning/map_rounds_to_r0.py data/R0/unique.fasta data/Bm76/R1_unique.fasta
python src/fasta_cleaning/map_rounds_to_r0.py data/R0/unique.fasta data/Bm76/R3_unique.fasta
python src/fasta_cleaning/map_rounds_to_r0.py data/R0/unique.fasta data/Re77/R1_unique.fasta
python src/fasta_cleaning/map_rounds_to_r0.py data/R0/unique.fasta data/Re77/R3_unique.fasta
python src/fasta_cleaning/map_rounds_to_r0.py data/R0/unique.fasta data/Re78/R1_unique.fasta
python src/fasta_cleaning/map_rounds_to_r0.py data/R0/unique.fasta data/Re78/R3_unique.fasta
python src/fasta_cleaning/map_rounds_to_r0.py data/R0/unique.fasta data/Re79/R1_unique.fasta
python src/fasta_cleaning/map_rounds_to_r0.py data/R0/unique.fasta data/Re79/R3_unique.fasta


##### Rename rounds sequences to R0 names
python src/fasta_cleaning/rename_rounds_to_r0.py data/R0/unique.fasta \
    data/Bm75/R1_unique.fasta \
    data/Bm75/R3_unique.fasta \
    data/Bm76/R1_unique.fasta \
    data/Bm76/R3_unique.fasta \
    data/Re77/R1_unique.fasta \
    data/Re77/R3_unique.fasta \
    data/Re78/R1_unique.fasta \
    data/Re78/R3_unique.fasta \
    data/Re79/R1_unique.fasta \
    data/Re79/R3_unique.fasta


##### Run ANARCI on R0 to get IMGT numbering
ANARCI -i data/R0/unique.fasta -o data/R0/R0_imgt_alignment.csv --scheme imgt --csv --ncpu 30