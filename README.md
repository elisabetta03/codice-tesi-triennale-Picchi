# Codice della tesi

Codice utilizzato per la preparazione dei dati e per le analisi della tesi **“Bias sociali nei Vision-Language Models: analisi della sensibilità ai prompt su immagini di soggetti associati a diverse professioni”**.

Il repository contiene la pipeline di annotazione, scoring NRC-VAD, costruzione dei dataset analitici e analisi descritte nei Capitoli 3 e 4 della tesi. I file originali delle immagini e gli output completi dei modelli non sono inclusi.

## Disegno sperimentale

L'esperimento considera:

- **50 professioni**;
- **1500 immagini** (30 per professione), di cui 836 generate artificialmente e 664 reali;
- **25 prompt utente**, organizzati in 5 condizioni con 5 parafrasi ciascuna:
  - baseline neutro;
  - occupazione neutra;
  - modificatore positivo;
  - modificatore negativo;
  - forced-choice;
- **10 prompt di sistema** più la condizione `NO_SYS`;
- due varianti dello stesso modello:
  - `Qwen/Qwen3.8-27B` (Qwen Standard);
  - `OBLITERATUS/Qwen3.8-27B-OBLITERATED` (Qwen Obliterated).

La generazione originale è stata eseguita localmente tramite vLLM con temperatura 0, top-p 1 e seed 42. Lo script utilizzato per l'inferenza vLLM non è incluso in questo repository: il codice qui pubblicato parte dagli output già generati.

## Pipeline

### 1. Annotazione delle immagini

`annotate_images.py`

Interfaccia grafica utilizzata per annotare manualmente ogni immagine rispetto a:

- genere percepito: `M` / `F`;
- appartenenza razziale percepita: `W` / `NW`.

Le etichette rappresentano attributi percepiti a partire dallo stimolo visivo e non informazioni certe sull'identità dei soggetti raffigurati.

### 2. Scoring NRC-VAD

`vad_analysis.py`

Elabora le risposte narrative utilizzando spaCy e la versione inglese 2018 del **NRC-VAD Lexicon**.

Per ogni token alfabetico:

1. viene cercata nel lessico la forma superficiale normalizzata in minuscolo;
2. se non viene trovata, viene utilizzato il lemma spaCy come fallback;
3. i token non presenti nel lessico vengono ignorati.

Per ciascuna risposta e per ciascuna dimensione **Valence, Arousal e Dominance** vengono calcolate:

- media;
- deviazione standard della popolazione;
- mediana;
- massimo.

Si ottengono quindi 12 variabili NRC-VAD per risposta.

Esecuzione:

```bash
python vad_analysis.py qwen
python vad_analysis.py obliterated
```

### 3. Costruzione dei dataset analitici

`merge_datasets.py`

Unisce:

- output dei modelli con statistiche NRC-VAD;
- annotazioni demografiche delle immagini;
- provenienza dell'immagine (`REAL` / `AI`);
- metadati relativi a professione, prompt utente e prompt di sistema.

Genera i dataset finali utilizzati dalle analisi successive.

## Analisi della tesi

### RQ1 — Effetto della formulazione del prompt

`rq1_prompt_effect.py`

Confronta, nella sola condizione `NO_SYS`:

- baseline;
- occupazione neutra;
- modificatore positivo;
- modificatore negativo.

Le cinque parafrasi vengono prima mediate a livello di immagine. La figura finale confronta Qwen Standard e Qwen Obliterated per Valence, Arousal e Dominance, riportando media, mediana e massimo; la deviazione standard lessicale è riportata nelle tabelle.

Output principale:

`analysis_outputs/rq1/rq1_vad_statistics_grouped.pdf`

### RQ2 — Attributi demografici percepiti

`rq2_demographic_effects.py`

Analizza i contrasti:

- `M-F` per il genere percepito;
- `W-NW` per l'appartenenza razziale percepita.

Per ogni professione vengono calcolati:

- contrasto nella condizione di occupazione neutra;
- differenza nell'effetto del modificatore positivo, `POS - OCC`;
- differenza nell'effetto del modificatore negativo, `NEG - OCC`.

I contrasti vengono mediati assegnando lo stesso peso a ciascuna professione.

Gli intervalli di confidenza al 95% sono stimati mediante **bootstrap percentile con 10.000 ricampionamenti con reinserimento delle professioni**. Gli intervalli sono utilizzati in senso descrittivo e non come test confermativi.

Output principali:

- `analysis_outputs/rq2/rq2_gender_forest_plot.pdf`
- `analysis_outputs/rq2/rq2_race_forest_plot.pdf`

### RQ3 — Effetto dei prompt di sistema

`rq3_system_prompt_effects.py`

Utilizza esclusivamente la condizione utente di occupazione neutra e confronta i dieci prompt di sistema con `NO_SYS`.

Le abbreviazioni usate nelle figure sono:

- `G1-G7` per i prompt relativi al genere;
- `R1-R3` per i prompt relativi all'appartenenza razziale percepita.

L'analisi comprende:

1. variazione complessiva di ciascun system prompt rispetto a `NO_SYS`;
2. differenze demografiche nell'effetto dei prompt:
   - `M-F` per `G1-G7`;
   - `W-NW` per `R1-R3`.

Anche in questo caso vengono utilizzati intervalli bootstrap al 95% con 10.000 ricampionamenti delle professioni. Nelle heatmap i valori in grassetto corrispondono a intervalli che non includono lo zero.

Output principali:

- `analysis_outputs/rq3/rq3_system_effect_heatmap.pdf`
- `analysis_outputs/rq3/rq3_demographic_effect_heatmap.pdf`

### RQ4 — Provenienza REAL vs AI

`rq4_image_source_effect.py`

Confronta immagini reali e generate artificialmente controllando:

- professione;
- condizione visiva dell'immagine (neutral, positive, negative);
- condizione del prompt utente.

Sono mantenute soltanto le professioni per le quali sono presenti sia immagini `REAL` sia immagini `AI` in tutte e tre le condizioni visive. Nel dataset finale della tesi il requisito è soddisfatto da **46 professioni**.

Le tre condizioni visive ricevono lo stesso peso all'interno di ciascuna professione. Il contrasto analizzato è:

`AI - REAL`

Gli intervalli di confidenza al 95% sono stimati con 10.000 ricampionamenti bootstrap delle professioni.

Output principale:

`analysis_outputs/rq4/rq4_source_forest_plot.pdf`

### Forced-choice

`forced_choice_analysis.py`

Analizza separatamente i cinque prompt forced-choice nella condizione `NO_SYS`.

La scelta binaria viene estratta con criteri conservativi:

1. riconoscimento delle risposte che iniziano esplicitamente con `YES` o `NO`;
2. recupero di formulazioni introduttive nelle quali la scelta binaria è comunque inequivocabile;
3. esclusione degli output privi di una scelta esplicita.

Nel dataset utilizzato nella tesi il parser classifica:

- **7410/7485** risposte di Qwen Standard (99,0%);
- **7495/7495** risposte di Qwen Obliterated.

Per l'aggregazione vengono mantenute soltanto le immagini con tutte le 5 parafrasi classificabili:

- **1432** immagini per Qwen Standard;
- **1499** per Qwen Obliterated.

Per ogni immagine viene calcolata la proporzione di risposte YES (`pYES`), poi mediata prima entro professione e successivamente tra professioni.

Lo script produce:

- risultati complessivi;
- confronto per genere percepito;
- confronto per appartenenza razziale percepita;
- analisi per i cinque macrogruppi professionali.

I macrogruppi contengono rispettivamente **9, 10, 8, 9 e 14 professioni**, per un totale di 50.

Output principali:

- `analysis_outputs/forced_choice/forced_choice_demographic_yes_rates.pdf`
- `analysis_outputs/forced_choice/forced_choice_macrogroup_yes_rates.pdf`

#### Controllo qualitativo delle risposte NO

Come riportato nella tesi, è stato effettuato separatamente un controllo manuale su un campione casuale di 100 delle 2554 risposte NO classificabili di Qwen Standard:

- 49: rifiuto o impossibilità esplicita di inferire la competenza dall'immagine;
- 26: invalidazione del compito perché il soggetto viene riconosciuto come sintetico/non reale;
- 25: effettivo giudizio negativo sulla competenza.

Questo controllo ha finalità interpretativa e **non viene utilizzato per ricodificare le risposte o ricalcolare le percentuali quantitative**. La classificazione manuale del campione non è automatizzata nello script.

## Controllo del dataset

`validate_dataset.py`

Esegue controlli di consistenza sul dataset, inclusi:

- numero di risposte per immagine;
- combinazioni prompt/system prompt;
- coerenza delle annotazioni a livello di immagine;
- riconoscimento delle 50 professioni e dei cinque macrogruppi.

Non fa parte della pipeline analitica principale.

## Dipendenze

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Dipendenze principali:

- pandas
- numpy
- matplotlib
- spaCy
- tqdm
- Pillow

Il **NRC-VAD Lexicon** non è incluso nel repository. Deve essere scaricato dalla fonte ufficiale e posizionato in uno dei percorsi configurati in `vad_analysis.py`, oppure deve essere modificato il percorso nella configurazione iniziale dello script.

## Dati non inclusi

Per dimensioni e modalità di raccolta non sono inclusi:

- le 1500 immagini;
- gli output completi generati dai due modelli;
- i dataset intermedi e finali presenti in `merged_outputs/`;
- il file `image_annotations.csv`;
- il NRC-VAD Lexicon.

I percorsi attesi sono definiti nella configurazione iniziale dei singoli script.

## Riproducibilità

Il repository permette di riprodurre la pipeline di **annotazione, scoring NRC-VAD, costruzione dei dataset analitici e analisi statistiche/visualizzazioni** descritte nella tesi, a condizione di disporre dei dati originali.

La generazione iniziale delle risposte tramite vLLM non è inclusa nel repository.
