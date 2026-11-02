# Codice della tesi

Script utilizzati per la preparazione dei dati e per le analisi della tesi sui bias sociali nei Vision-Language Models.

## Pipeline

1. `src/annotate_images.py` - annotazione manuale di genere percepito (M/F) e appartenenza razziale percepita (W/NW).
2. `src/vad_analysis.py` - calcolo delle statistiche NRC-VAD sulle risposte generate.
3. `src/merge_datasets.py` - unione dei punteggi VAD con i metadati delle immagini e creazione dei dataset finali long/wide.
4. `src/rq1_prompt_effect.py` - RQ1: effetto della formulazione del prompt utente.
5. `src/rq2_demographic_effects.py` - RQ2: confronti per genere e appartenenza razziale percepiti.
6. `src/rq3_system_prompt_effects.py` - RQ3: effetto dei prompt di sistema.
7. `src/rq4_image_source_effect.py` - RQ4: confronto tra immagini reali e generate artificialmente.
8. `src/forced_choice_analysis.py` - analisi separata delle risposte forced-choice YES/NO.

`tools/validate_dataset.py` è un controllo di consistenza opzionale e non fa parte della pipeline analitica principale.

## Dipendenze

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Il NRC-VAD Lexicon non è incluso nel repository. Va scaricato dalla fonte ufficiale e posizionato in uno dei percorsi configurati in `vad_analysis.py`, oppure va modificata la configurazione iniziale dello script.

## Dati

I file di output dei modelli e le immagini non sono inclusi nel repository. I percorsi attesi sono definiti nella sezione di configurazione iniziale dei singoli script.

## Riproducibilità

Il repository, nello stato attuale, contiene la pipeline di annotazione, scoring e analisi. Nei file forniti non è presente lo script vLLM utilizzato per generare le risposte originali dei due modelli: prima di descrivere la repository come riproduzione end-to-end dell'esperimento è opportuno aggiungere anche quello script, insieme all'eventuale codice usato per compilare i prompt.
