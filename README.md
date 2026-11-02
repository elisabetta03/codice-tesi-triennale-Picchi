# Codice della tesi

Script utilizzati per la preparazione dei dati e per le analisi della tesi sui bias sociali nei Vision-Language Models.

## Pipeline

1. `annotate_images.py` - annotazione manuale di genere percepito (M/F) e appartenenza razziale percepita (W/NW).
2. `vad_analysis.py` - calcolo delle statistiche NRC-VAD sulle risposte generate.
3. `merge_datasets.py` - unione dei punteggi VAD con i metadati delle immagini e creazione dei dataset finali long/wide.
4. `rq1_prompt_effect.py` - RQ1: effetto della formulazione del prompt utente.
5. `rq2_demographic_effects.py` - RQ2: confronti per genere e appartenenza razziale percepiti.
6. `rq3_system_prompt_effects.py` - RQ3: effetto dei prompt di sistema.
7. `rq4_image_source_effect.py` - RQ4: confronto tra immagini reali e generate artificialmente.
8. `forced_choice_analysis.py` - analisi separata delle risposte forced-choice YES/NO.

`validate_dataset.py` esegue controlli di consistenza sul dataset e non fa parte della pipeline analitica principale.

## Dipendenze

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Il NRC-VAD Lexicon non è incluso nel repository. Deve essere scaricato dalla fonte ufficiale e posizionato in uno dei percorsi configurati in `vad_analysis.py`, oppure deve essere modificato il percorso nella configurazione iniziale dello script.

## Dati

I file di output dei modelli e le immagini utilizzate nell'esperimento non sono inclusi nel repository. I percorsi attesi sono definiti nella configurazione iniziale dei singoli script.

## Riproducibilità

Il repository contiene il codice utilizzato per l'annotazione delle immagini, il calcolo delle statistiche NRC-VAD, la preparazione dei dataset e le analisi presentate nella tesi.
