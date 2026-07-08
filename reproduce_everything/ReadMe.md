# Here are the instructions for reproducting my results:

Also i have provided you with the two files I have after comparing the predictions of EvOR and CasFinder, if you want to compare that to yours aswell/use that, since the downloading and running casfinder took a very long time for me. -> located here in the folder "output_dataset_10k" as "intermediate_array_cas_annotations.tsv" and "curated_dataset.tsv"
-> the first one has the predictions of casfinder and adds the crispr type and the second one combines both predictions in a file, ready for processing.
-> if you want to use them you can skip to step #5, and uncomment the paths for these at the beginning of run_all.py (line 45+46)

If you want to also do the getting the predictions of both tools steps, i have provided my documentation here, But I did this now 2 months ago and it isnt much present in my memory anymore, so i have not cleaned them up properly. (for instance a lot of the compare-results.py script is just analyzing and printing out stats of the predictions) though i think i have documented it enough to use them, so here you go. the scripts used are located in the "scripts" subfolder here. they are supposed to be in the same folder as the results they are used for, they are only placed here in this subfolder for a better overview for you.


# 1. download all bacterial and archea data from ncbi

I first got all accesion numbers and then split them into batches as i dont have enough free memory to save them all. depending on your setup that might differ, I split them into batches of approx. 20GB (6k genomes). 

getting all acc numbers:

```bash
./datasets summary genome taxon bacteria archaea \
  --assembly-level complete \
  --assembly-source refseq \
  --annotated \
  --as-json-lines \
  | jq -r '.accession' > all_accessions.txt
```

split into batches:

```bash
split -l 6000 --numeric-suffixes=1 --suffix-length=3 --additional-suffix=.txt all_accessions.txt batch_
```

download batch (change batch name for each one):

```bash
./datasets download genome accession --inputfile batch_001.txt --include genome --dehydrated --filename batch_001.zip
```
```bash
unzip batch_001.zip -d batch_001/
```
```bash
./datasets rehydrate --directory batch_001/
```

# 2. run CRISPRCasFinder on it

-> configure casfinder like specified in their repo, then activate env. also here I destinctly remember, that when not running from inside the folder where the tool is placed, that it sometimes didnt work, so i always ran it from there and with cardcoded paths.

again, I did it in batches, so here it has to be modified for each bartch name then. also its run as a script for better visibility, specify paths to your needs:

this are just exapmles of my scripts used, obviously you need to specify the paths of your machine in them (just mine provided here as exapmles)

```bash
nano /home/benni/ccf.sh
```

then inside is this (had to specify the supplementary files here as otherwise predictions dont work, also # of jobs could be different depending on compute power available, again specify the batch youre using and paths)

```bash
#!/bin/bash
FASTA_DIR=/home/benni/CRISPRCasFinder/batch_003/ncbi_dataset/data
CF_DIR=/home/benni/CRISPRCasFinder

find $FASTA_DIR -name "*.fna" | parallel \
  --jobs 16 \
  --progress \
  --joblog /home/benni/crispr_parallel.log \
  'SAMPLE=$(basename {.}) && \
   WORKDIR=/tmp/crispr_work/$SAMPLE && \
   mkdir -p $WORKDIR && \
   cp '"$CF_DIR"'/sel392v2.so $WORKDIR/ && \
   cd $WORKDIR && \
   perl '"$CF_DIR"'/CRISPRCasFinder.pl \
     -in {} \
     -soFile $WORKDIR/sel392v2.so \
     -cas \
     -dbc '"$CF_DIR"'/supplementary_files/CRISPR_crisprdb.csv \
     -rpts '"$CF_DIR"'/supplementary_files/Repeat_List.csv \
     -drpt '"$CF_DIR"'/supplementary_files/repeatDirection.tsv && \
   mv $WORKDIR/Result_* '"$CF_DIR"'/test_out/ && \
   rm -rf $WORKDIR'
```

then run this (includes clean up for temp data from last batch):

```bash
rm -rf /tmp/crispr_work/
rm -f /home/benni/crispr_parallel.log
bash /home/benni/ccf.sh
```

# 3. run EvOr (in spacerplacer)

-> configure spacerplacer like specified in their repo, then activate env.

check for bad json files (I encountered some broken json files that then crash spacerPlacer, so remove those)

```bash
mkdir -p bad_json
```

```bash
for f in $(find . -name "result.json"); do
    if ! jq empty "$f" 2>/dev/null; then
        dir=$(dirname "$f")
        echo "Moving folder: $dir"
        mv "$dir" bad_json/
    fi
done
```

then onyl get all of the evidence level 4 arrays (test_out here is the folder with all of the results from casfinder):
-> description of extract-arrays.py:
Reads test_out/Result_*/result.json for all present json files in that folder
has to be placed in the same directory as the test_out folder of CasFinder

```bash
python extract-arrays.py test_out extracted_arrays.txt 4
```

when using the native SpacerPlacer clustering, with all the data I get too large of clusters, that then crash the process, so I build my own clustering trying to copy theirs with max-size constraints (dont know if you can use the native one of them, otherwise mine is provided here as cluster_like_spacerplacer.py. this can then be run with this command, i used max-size 300 in the end.)
--> description of cluster_like_spacerplacer.py:
Reads test_out/Result_*/result.json for all present json files in that folder
writes the clustering and metadata
has to be placed in the same directory as the test_out folder of CasFinder 

```bash
python cluster_like_spacerplacer.py \
  --input-dir test_out \
  --output-dir sp_style_clusters \
  --min-evidence 4 \
  --max-size 300 \
  --cluster-spacers \
  --max-distance 1 \
  --workers 8 \
  --group-by-cas-type
```

this should then get us a "singleton" folder, where all of the arrays we cant cluster properly are then located. -> cant make a prediction for them
then run spacerPlacer on all clustered groups (again i put this into a script)

```bash
nano /home/benni/execute_groups.sh
```

then inside, (here --determine_orientation is the important flag to get the orientation prediction):

```bash
MAX_JOBS=4
OUTPUT_BASE="/home/benni/SpacerPlacer/out-sp_style_clusters"
mkdir -p "$OUTPUT_BASE"

for group in /home/benni/SpacerPlacer/sp_style_clusters/*.fa; do
    [ -f "$group" ] || continue
    
    while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do sleep 1; done
    
    python spacerplacer.py "$group" "$OUTPUT_BASE/$(basename $group .fa)" \
        -it spacer_fasta --determine_orientation &
done
wait
```

then run this script
```bash
bash /home/benni/execute_groups.sh
```

# 4. compare predictions and prepare dataset creation

here i only have the scipts left, run these in this order, since the script depends on the other script (bad programming i know, but it is what it is.) also here the distance for identifying the subtype is 10k bp. 
--> again both have to be placed in the same folder as test_out for the CasFinder results and sp_style_clusters for the EvOr predictions

this script is mainly used to generate a comparison between the CasFinder results and the evOr results, but can also be used for comparing multiple different clusterings. Most of the code is not nescessary, as it it just stats and comparison, but the next script needs the comparison table generated by this script.

```bash
python compare-results.py
```

this script then generates the final .tsv files that are used by the model to construct its dataset. the cluster_csv here is generated by the above script, it should have the same name, but if it gets a different name, then just specify the name of the file you have.

```bash
python generate_dataset.py   --results_dir ./test_out   --cluster_csv ./crispr_comparison_results/ccf_vs_out-sp_style_clusters_detail.csv   --out_dir ./output_dataset_fixed_10k --max_distance_bp 10000
```



# 5. actual dataset creation and training of the model

here you only need to run the file run_all.py once, that then builds the dataset and train/test splits, imports the carbon model and does LoRA on its weights. the hyperparameters are specified at the beginning of the file, they are set to what i last used.

```bash
python reproduce_everything/run_all.py
```