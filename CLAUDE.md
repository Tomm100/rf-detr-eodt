# EoDT — Istruzioni per l'agente

Questo è un **fork di RF-DETR per una tesi magistrale** (Tommaso Centonze, PoliTo): implementa EoDT, l'approccio encoder-only di EoMT applicato alla detection di RF-DETR. Non è un contributo alla repo ufficiale.

**Queste istruzioni hanno la precedenza su `AGENTS.md`**, che descrive le regole per contribuire a RF-DETR upstream (pre-commit, PR, CI). Consulta `AGENTS.md` solo come documentazione dell'architettura di RF-DETR, se serve.

## Fonte di verità

- **`EODT_SPEC.md`** contiene tutte le decisioni: architettura, baseline, training, ordine di lavoro (§9) e ambienti (§11). Prima di ogni passo leggi la sezione che lo riguarda.
- Le voci **[DA CONFERMARE]** e **[DA DECIDERE]** non si cambiano di propria iniziativa. Se una scelta non è scritta nella specifica, **chiedi**: non decidere al posto mio.
- Se trovi un errore o una contraddizione nella specifica, segnalalo prima di scrivere codice.

## Cartelle

| Cartella | Uso |
|---|---|
| `C:/dev/tesi/rf-detr-eodt` (questa) | l'unica in cui si scrive |
| `C:/dev/tesi/ref-rfdetr` | RF-DETR originale, stesso commit del fork: **sola lettura**, per confronti e test di equivalenza |
| `C:/dev/tesi/ref-eomt` | EoMT originale: **sola lettura**, riferimento per le query nei blocchi (`models/eomt.py`, `models/vit.py`) |

- Codice nuovo in `src/rfdetr/models/eodt/`, test in `tests/eodt/`, configurazioni in `configs/eodt/` e `configs/baseline/`, script per Leonardo in `scripts/leonardo/`.
- I file esistenti di RF-DETR **non si modificano**, salvo i punti di integrazione della §8 della specifica, e solo dopo averli proposti nel piano.

## Come si lavora

1. **Un passo della §9 alla volta.** Parti sempre in plan mode: proponi il piano del passo, aspetta la mia approvazione, poi implementa.
2. **Test prima di dichiarare finito.** Un passo è chiuso solo quando il suo test di verifica (colonna "Test di verifica" della §9) passa. Riporta l'output dei test, non un riassunto ottimistico: se qualcosa fallisce o è stato saltato, dillo.
3. **A fine passo ti fermi** e riassumi: file toccati, cosa fa ogni pezzo, quali test passano. Il commit lo faccio io dopo aver rivisto il diff.
4. **Git:** niente `commit`, `push`, `rebase`, `reset`, `merge` (sono anche bloccati nei permessi). Puoi usare `git status`, `git diff`, `git log`.
5. **Subagent:** solo per compiti chiusi e di **sola lettura**:
   - esplorazione ("come `RFDETRModelModule` costruisce il modello?"), così il contesto principale resta pulito;
   - revisione a fine passo: un subagent confronta il diff con la specifica e segnala cosa non torna.

   Il codice lo scrive solo l'agente principale. Niente orchestrazioni con più agenti che scrivono in parallelo.
6. **Niente lavoro fuori dal passo corrente:** se noti altro da sistemare, segnalalo in fondo al riepilogo invece di farlo.

## Ambiente locale (PC)

- Python 3.11 in `.venv` (creato con uv). Esegui sempre con `.venv/Scripts/python.exe`, per esempio:
  - test veloci di EoDT (solo ViT-S): `.venv/Scripts/python.exe -m pytest tests/eodt -q -p no:cacheprovider --confcutdir=tests/eodt -m "not gpu"`
  - tutti i test di EoDT (anche ViT-B e ViT-L, lenti su CPU): lo stesso comando senza `-m "not gpu"`
  - `--confcutdir=tests/eodt` serve a saltare `tests/conftest.py` di RF-DETR, che importa `fuse_augmentations` (dipendenza di sviluppo non installata) e a ogni sessione scarica `facebook/dinov2-with-registers-base`. I test di EoDT non lo usano: se servono fixture comuni, vanno in `tests/eodt/conftest.py`.
  - uno script: `.venv/Scripts/python.exe path/script.py`
- GPU: **GTX 1650, 4 GB, niente bf16**. In locale usa fp32 (o fp16), batch piccoli e, per l'overfit, risoluzione ridotta. I passi 1-4 devono girare anche su CPU.
- **Dipendenze:** non installare né aggiornare pacchetti senza chiedere. Le versioni sono fissate in `requirements-eodt.txt` e devono restare identiche a Leonardo (§11 della specifica).
- I pesi di DINOv2 si scaricano da timm/Hugging Face al primo uso: va bene in locale.
- **Dati per il test di overfit (passo 5):** `C:/dev/tesi/data/coco_mini`, mini-COCO in formato RF-DETR (`train2017/`, `val2017/`, `annotations/instances_{train,val}2017.json`): 20 immagini di val2017 con 2-10 oggetti ciascuna (85 oggetti, 32 classi), identiche in train e val. Non modificarla.

## Leonardo (training)

- Non hai accesso a Leonardo: **gli script SLURM li scrivi, li lancio io**. Non usare `sbatch` o `ssh`.
- Ambiente: `module load python cuda && source /leonardo_work/IscrC_EoDT/tcentonz/eodt-venv/bin/activate`; codice in `/leonardo_work/IscrC_EoDT/tcentonz/rf-detr-eodt`.
- I nodi di calcolo **non hanno internet**: pesi e dati vanno preparati prima dal nodo di login; W&B gira offline (§7.3 della specifica).
- Limite di 24 ore per job: ogni training deve poter ripartire dall'ultimo checkpoint.
- Gli script `.sh` devono avere gli a capo Unix (garantito da `.gitattributes`).

## Stile del codice

- Segui lo stile del codice vicino. Type hint su funzioni e metodi, docstring brevi in stile Google.
- Codice minimo e leggibile: devo poter spiegare ogni pezzo nella tesi. Niente astrazioni non richieste dalla specifica.
- Ogni valore che può cambiare tra esperimenti va nella configurazione, mai scritto nel codice (§3.7).
